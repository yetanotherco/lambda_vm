//! Proving `Σ_x w(x)·f(x) = y` about a committed polynomial, which is what
//! settles the residual claims the other arguments hand back.
//!
//! The sumcheck produces the folding randomness; the fully folded codeword is
//! the constant `f(α)`, and the sumcheck's residual is `w(α)·f(α)`. Both must
//! name the same `f(α)`.
//!
//! An evaluation claim is the weight `w = eq(z, ·)`, which is what [`prove`]
//! and [`verify`] specialize to. A **stacked** claim — many columns packed into
//! one committed polynomial, each read from its own subcube — is a weight that
//! sums several of those, and settling it costs the same one sumcheck. That is
//! why the general form is the one implemented.
//!
//! One round, folding all the way down, so a block is the whole message — which
//! is what [`whir_chain`](crate::whir_chain) exists to fix. This module is the
//! degenerate one-round case of it, kept as the reference for the identity the
//! whole thing rests on.

use crypto::fiat_shamir::is_transcript::IsTranscript;
use math::{
    field::{
        element::FieldElement,
        traits::{IsFFTField, IsField, IsPrimeField, IsSubFieldOf},
    },
    traits::AsBytes,
};

use crate::{
    Error,
    eq::{eq_eval, eq_mle},
    mle::Mle,
    poly::EqScaled,
    sumcheck::{self, SumcheckProof},
    virtual_poly::{Term, VirtualPolynomial},
    whir::{Domain, encode, fold_codeword_k, lift_coefficients},
    whir_commit::{CodewordCommitment, Commitment, CosetOpening, fold_coset, verify_opening},
    whir_hash::WhirHash,
};

/// Blowup and query count.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EvalConfig {
    /// `log2` of the code's inverse rate.
    pub log_blowup: usize,
    /// Codeword blocks checked against the fold.
    pub num_queries: usize,
}

/// A proof that a committed polynomial takes a claimed value at a point.
#[derive(
    Clone,
    Debug,
    serde::Serialize,
    serde::Deserialize,
    rkyv::Archive,
    rkyv::Serialize,
    rkyv::Deserialize,
)]
#[serde(bound = "")]
pub struct EvalProof<E: IsField> {
    pub sumcheck: SumcheckProof<E>,
    /// The constant the codeword folds to — the prover's claim for `f(α)`.
    pub final_value: FieldElement<E>,
    pub openings: Vec<CosetOpening<E>>,
}

/// Commits to `f`, ready to answer evaluation claims.
pub fn commit<F, E, H>(
    f: &Mle<E>,
    config: &EvalConfig,
) -> Result<(CodewordCommitment<E, H>, Domain<F>), Error>
where
    F: IsFFTField + IsPrimeField + IsSubFieldOf<E> + 'static,
    E: IsField + Send + Sync + 'static,
    H: WhirHash,
    FieldElement<E>: AsBytes + Sync + Send,
{
    let domain = Domain::<F>::new(f.num_vars() + config.log_blowup)?;
    let codeword = encode::<F, E>(&lift_coefficients(f), &domain)?;
    // One block per fold target: folding all the way leaves `2^log_blowup`.
    let commitment = CodewordCommitment::new(&codeword, f.num_vars())?;
    Ok((commitment, domain))
}

/// `Σ_x w(x)·f(x)`, the sumcheck a weighted claim becomes.
fn weighted<F: IsField>(
    f: &Mle<F>,
    weight: Mle<F>,
) -> Result<EqScaled<F, VirtualPolynomial<F>>, Error> {
    let inner = VirtualPolynomial::new(vec![f.clone()], vec![Term::single(0)])?;
    EqScaled::new(inner, weight)
}

/// Proves `f(z) = y`.
///
/// The caller must have absorbed the commitment root and `z` into `transcript`
/// already; both sides must do the same.
pub fn prove<F, E, T, H>(
    f: &Mle<E>,
    z: &[FieldElement<E>],
    commitment: &CodewordCommitment<E, H>,
    domain: &Domain<F>,
    config: &EvalConfig,
    transcript: &mut T,
) -> Result<EvalProof<E>, Error>
where
    F: IsFFTField + IsPrimeField + IsSubFieldOf<E> + 'static,
    E: IsField + Send + Sync + 'static,
    FieldElement<E>: AsBytes + Sync + Send,
    T: IsTranscript<E>,
    H: WhirHash,
{
    prove_weighted::<F, E, T, H>(f, eq_mle(z)?, commitment, domain, config, transcript)
}

/// Proves `Σ_x w(x)·f(x) = y` for a weight the verifier can evaluate itself.
pub fn prove_weighted<F, E, T, H>(
    f: &Mle<E>,
    weight: Mle<E>,
    commitment: &CodewordCommitment<E, H>,
    domain: &Domain<F>,
    config: &EvalConfig,
    transcript: &mut T,
) -> Result<EvalProof<E>, Error>
where
    F: IsFFTField + IsPrimeField + IsSubFieldOf<E> + 'static,
    E: IsField + Send + Sync + 'static,
    FieldElement<E>: AsBytes + Sync + Send,
    T: IsTranscript<E>,
    H: WhirHash,
{
    let (sumcheck, alphas) = sumcheck::prove(weighted(f, weight)?, transcript)?;

    // From the commitment, not a second encoding: it is the same array, and a
    // prover that folded a different one could not then answer the openings.
    // This path builds its own commitment on the host, so the codeword is here.
    let values = commitment
        .codeword()
        .host()
        .ok_or(crate::Error::EmptyPolynomial)?;
    let (folded, _) = fold_codeword_k::<F, E, E>(values, domain, &alphas)?;
    let final_value = folded[0].clone();
    transcript.append_field_element(&final_value);

    let queries = sample_queries(transcript, config.num_queries, commitment.num_leaves());
    let openings = queries
        .into_iter()
        .map(|q| commitment.open(q))
        .collect::<Result<_, _>>()?;

    Ok(EvalProof {
        sumcheck,
        final_value,
        openings,
    })
}

fn sample_queries<F, T>(transcript: &mut T, num_queries: usize, bound: usize) -> Vec<usize>
where
    F: IsField,
    T: IsTranscript<F>,
{
    (0..num_queries)
        .map(|_| transcript.sample_u64(bound as u64) as usize)
        .collect()
}

/// Verifies `f(z) = y` against a commitment.
pub fn verify<F, E, T, H>(
    proof: &EvalProof<E>,
    root: &Commitment,
    z: &[FieldElement<E>],
    y: FieldElement<E>,
    domain: &Domain<F>,
    config: &EvalConfig,
    transcript: &mut T,
) -> Result<(), Error>
where
    F: IsFFTField + IsPrimeField + IsSubFieldOf<E> + 'static,
    E: IsField + Send + Sync + 'static,
    FieldElement<E>: AsBytes + Sync + Send,
    T: IsTranscript<E>,
    H: WhirHash,
{
    verify_weighted::<F, E, T, _, H>(
        proof,
        root,
        |alphas: &[FieldElement<E>]| eq_eval(z, alphas),
        y,
        z.len(),
        domain,
        config,
        transcript,
    )
}

/// Verifies `Σ_x w(x)·f(x) = y`.
///
/// `weight_at` is the weight's closed form; the verifier evaluates it at the
/// sumcheck point rather than holding its table.
#[allow(clippy::too_many_arguments)]
pub fn verify_weighted<F, E, T, W, H>(
    proof: &EvalProof<E>,
    root: &Commitment,
    weight_at: W,
    y: FieldElement<E>,
    num_vars: usize,
    domain: &Domain<F>,
    config: &EvalConfig,
    transcript: &mut T,
) -> Result<(), Error>
where
    F: IsFFTField + IsPrimeField + IsSubFieldOf<E> + 'static,
    E: IsField + Send + Sync + 'static,
    FieldElement<E>: AsBytes + Sync + Send,
    T: IsTranscript<E>,
    W: FnOnce(&[FieldElement<E>]) -> Result<FieldElement<E>, Error>,
    H: WhirHash,
{
    // The weight raises the degree of the plain `f` term to two.
    let claim = sumcheck::verify(&proof.sumcheck, y, num_vars, 2, transcript)?;
    let alphas = &claim.point;

    // The sumcheck's residual is w(α)·f(α); the verifier knows w.
    let weight = weight_at(alphas)?;
    let required = claim
        .expected_evaluation
        .clone()
        .mul_by_inverse_of(&weight)
        .ok_or(Error::DegenerateEvaluationPoint)?;

    transcript.append_field_element(&proof.final_value);

    // The wire: the folded codeword and the sumcheck must name the same f(α).
    if proof.final_value != required {
        return Err(Error::EvaluationMismatch);
    }

    if proof.openings.len() != config.num_queries {
        return Err(Error::QueryCountMismatch {
            expected: config.num_queries,
            got: proof.openings.len(),
        });
    }
    let num_leaves = domain.size() >> num_vars;
    let queries = sample_queries(transcript, config.num_queries, num_leaves);

    for (i, (&q, opening)) in queries.iter().zip(&proof.openings).enumerate() {
        if !verify_opening::<E, H>(root, q, opening) {
            return Err(Error::OpeningRejected { query: i });
        }
        if fold_coset::<F, E, E>(&opening.values, domain, q, alphas)? != proof.final_value {
            return Err(Error::FoldInconsistent { query: i });
        }
    }

    Ok(())
}

/// Small helper so the division reads as one step and the degenerate case is
/// explicit rather than a panic.
trait MulByInverse: Sized {
    fn mul_by_inverse_of(self, other: &Self) -> Option<Self>;
}

impl<F: IsField> MulByInverse for FieldElement<F> {
    fn mul_by_inverse_of(self, other: &Self) -> Option<Self> {
        other.inv().ok().map(|inv| self * inv)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crypto::fiat_shamir::default_transcript::DefaultTranscript;
    use math::field::goldilocks::GoldilocksField as F;

    use crate::whir_hash::KeccakWhir;

    type FE = FieldElement<F>;

    fn transcript() -> DefaultTranscript<F> {
        DefaultTranscript::<F>::new(b"whir-eval-test")
    }

    fn config() -> EvalConfig {
        EvalConfig {
            log_blowup: 2,
            num_queries: 3,
        }
    }

    /// Deterministic pseudo-random values. The seed is mixed *before* the
    /// shift — mixing it after would let nearby seeds collapse to the same
    /// polynomial, which silently turns a negative test into a tautology.
    fn pseudo_mle(num_vars: usize, seed: u64) -> Mle<F> {
        let vals: Vec<FE> = (0..(1u64 << num_vars))
            .map(|i| {
                let mixed = i
                    .wrapping_add(seed)
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15));
                FE::from(mixed >> 13)
            })
            .collect();
        Mle::new(vals).unwrap()
    }

    fn point(num_vars: usize) -> Vec<FE> {
        (0..num_vars).map(|i| FE::from(101 + i as u64)).collect()
    }

    #[test]
    fn an_honest_evaluation_verifies() {
        for num_vars in 1..=4usize {
            let f = pseudo_mle(num_vars, 11);
            let z = point(num_vars);
            let y = f.evaluate(&z).unwrap();

            let (commitment, domain) = commit::<F, F, KeccakWhir>(&f, &config()).unwrap();
            let proof = prove::<F, F, _, KeccakWhir>(
                &f,
                &z,
                &commitment,
                &domain,
                &config(),
                &mut transcript(),
            )
            .unwrap();

            verify::<F, F, _, KeccakWhir>(
                &proof,
                &commitment.root(),
                &z,
                y,
                &domain,
                &config(),
                &mut transcript(),
            )
            .unwrap_or_else(|e| panic!("num_vars={num_vars}: {e:?}"));
        }
    }

    /// The point of the whole file: claiming a value the polynomial does not
    /// take must fail, even though the prover runs the protocol honestly around
    /// that lie.
    #[test]
    fn a_false_evaluation_is_rejected() {
        let f = pseudo_mle(3, 13);
        let z = point(3);
        let y = f.evaluate(&z).unwrap();

        let (commitment, domain) = commit::<F, F, KeccakWhir>(&f, &config()).unwrap();
        let proof = prove::<F, F, _, KeccakWhir>(
            &f,
            &z,
            &commitment,
            &domain,
            &config(),
            &mut transcript(),
        )
        .unwrap();

        let err = verify::<F, F, _, KeccakWhir>(
            &proof,
            &commitment.root(),
            &z,
            y + FE::one(),
            &domain,
            &config(),
            &mut transcript(),
        )
        .unwrap_err();
        // The lie propagates into the residual, so it is the wire between the
        // sumcheck and the folded codeword that breaks.
        assert_eq!(err, Error::EvaluationMismatch);
    }

    #[test]
    fn a_forged_final_value_is_rejected() {
        let f = pseudo_mle(3, 17);
        let z = point(3);
        let y = f.evaluate(&z).unwrap();

        let (commitment, domain) = commit::<F, F, KeccakWhir>(&f, &config()).unwrap();
        let mut proof = prove::<F, F, _, KeccakWhir>(
            &f,
            &z,
            &commitment,
            &domain,
            &config(),
            &mut transcript(),
        )
        .unwrap();
        proof.final_value += FE::one();

        let err = verify::<F, F, _, KeccakWhir>(
            &proof,
            &commitment.root(),
            &z,
            y,
            &domain,
            &config(),
            &mut transcript(),
        )
        .unwrap_err();
        assert_eq!(err, Error::EvaluationMismatch);
    }

    #[test]
    fn a_tampered_opening_is_rejected() {
        let f = pseudo_mle(3, 19);
        let z = point(3);
        let y = f.evaluate(&z).unwrap();

        let (commitment, domain) = commit::<F, F, KeccakWhir>(&f, &config()).unwrap();
        let mut proof = prove::<F, F, _, KeccakWhir>(
            &f,
            &z,
            &commitment,
            &domain,
            &config(),
            &mut transcript(),
        )
        .unwrap();
        proof.openings[0].values[0] += FE::one();

        let err = verify::<F, F, _, KeccakWhir>(
            &proof,
            &commitment.root(),
            &z,
            y,
            &domain,
            &config(),
            &mut transcript(),
        )
        .unwrap_err();
        assert!(matches!(err, Error::OpeningRejected { query: 0 }));
    }

    /// A prover who commits one polynomial and argues about another.
    #[test]
    fn a_proof_about_a_different_polynomial_is_rejected() {
        let f = pseudo_mle(3, 23);
        let g = pseudo_mle(3, 29);
        let z = point(3);

        let (f_commitment, domain) = commit::<F, F, KeccakWhir>(&f, &config()).unwrap();
        // Argue g's evaluation while presenting f's commitment.
        let proof = prove::<F, F, _, KeccakWhir>(
            &g,
            &z,
            &f_commitment,
            &domain,
            &config(),
            &mut transcript(),
        )
        .unwrap();

        let err = verify::<F, F, _, KeccakWhir>(
            &proof,
            &f_commitment.root(),
            &z,
            g.evaluate(&z).unwrap(),
            &domain,
            &config(),
            &mut transcript(),
        )
        .unwrap_err();
        // The codeword comes out of the commitment, so a prover cannot be
        // inconsistent between the two: the lie surfaces on the wire between
        // the sumcheck and the folded value.
        assert_eq!(err, Error::EvaluationMismatch);
    }

    #[test]
    fn a_proof_replayed_under_another_transcript_is_rejected() {
        let f = pseudo_mle(3, 31);
        let z = point(3);
        let y = f.evaluate(&z).unwrap();

        let (commitment, domain) = commit::<F, F, KeccakWhir>(&f, &config()).unwrap();
        let proof = prove::<F, F, _, KeccakWhir>(
            &f,
            &z,
            &commitment,
            &domain,
            &config(),
            &mut transcript(),
        )
        .unwrap();

        let mut other = DefaultTranscript::<F>::new(b"a-different-statement");
        assert!(
            verify::<F, F, _, KeccakWhir>(
                &proof,
                &commitment.root(),
                &z,
                y,
                &domain,
                &config(),
                &mut other
            )
            .is_err()
        );
    }

    #[test]
    fn a_proof_with_too_few_openings_is_rejected() {
        let f = pseudo_mle(3, 37);
        let z = point(3);
        let y = f.evaluate(&z).unwrap();

        let (commitment, domain) = commit::<F, F, KeccakWhir>(&f, &config()).unwrap();
        let mut proof = prove::<F, F, _, KeccakWhir>(
            &f,
            &z,
            &commitment,
            &domain,
            &config(),
            &mut transcript(),
        )
        .unwrap();
        proof.openings.pop();

        let err = verify::<F, F, _, KeccakWhir>(
            &proof,
            &commitment.root(),
            &z,
            y,
            &domain,
            &config(),
            &mut transcript(),
        )
        .unwrap_err();
        assert!(matches!(err, Error::QueryCountMismatch { .. }));
    }

    #[test]
    fn the_commitment_has_one_block_per_fold_target() {
        let f = pseudo_mle(4, 41);
        let cfg = config();
        let (commitment, domain) = commit::<F, F, KeccakWhir>(&f, &cfg).unwrap();
        assert_eq!(commitment.num_leaves(), 1 << cfg.log_blowup);
        assert_eq!(domain.log_size(), 4 + cfg.log_blowup);
    }
}
