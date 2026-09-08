//! The evaluation argument, end to end: proving `f(z) = y` about a *committed*
//! polynomial.
//!
//! Every argument built so far ends by handing back a claim it cannot settle —
//! [`zerocheck`](crate::zerocheck) says "`C` must take this value at this
//! point", [`gkr`](crate::gkr) says the same about its input layer. This is what
//! settles them, and it is the piece that makes the rest mean anything: without
//! it a prover can answer any residual claim with whatever number closes the
//! proof.
//!
//! # The wire
//!
//! Proving `f(z) = y` runs a sumcheck on
//!
//! ```text
//! Σ_x eq(z, x)·f(x) = y
//! ```
//!
//! whose round challenges `α` bind `f`'s variables one at a time. The prover
//! folds the **committed codeword** with those same `α`. Folding every variable
//! leaves a constant codeword, and that constant is `f(α)`.
//!
//! So two independent computations must agree on `f(α)`:
//!
//! - the sumcheck's residual claim, which is `expected / eq(z, α)`;
//! - the folded codeword, spot-checked against the commitment by opening blocks.
//!
//! A prover who lies about `y` fails the first; one who lies about the codeword
//! fails the second; one who lies about both has to make them collide.
//!
//! # Scope
//!
//! One round, folding all the way down. Real WHIR folds `k` variables at a time
//! over several rounds, committing an intermediate codeword each time — that
//! keeps blocks small, since here a block is the whole message. Chaining rounds
//! is the next step and [`whir_round`](crate::whir_round) already does one link
//! of it. Out-of-domain sampling and grinding are also still absent.

use crypto::fiat_shamir::is_transcript::IsTranscript;
use math::{
    field::{
        element::FieldElement,
        traits::{IsFFTField, IsField, IsPrimeField},
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
#[derive(Clone, Debug)]
pub struct EvalProof<F: IsField> {
    pub sumcheck: SumcheckProof<F>,
    /// The constant the codeword folds to — the prover's claim for `f(α)`.
    pub final_value: FieldElement<F>,
    pub openings: Vec<CosetOpening<F>>,
}

/// Commits to `f`, ready to answer evaluation claims.
pub fn commit<F>(
    f: &Mle<F>,
    config: &EvalConfig,
) -> Result<(CodewordCommitment<F>, Domain<F>), Error>
where
    F: IsFFTField + IsPrimeField,
    FieldElement<F>: AsBytes + Sync + Send,
{
    let domain = Domain::<F>::new(f.num_vars() + config.log_blowup)?;
    let codeword = encode(&lift_coefficients(f), &domain)?;
    // One block per fold target: folding all the way leaves `2^log_blowup`.
    let commitment = CodewordCommitment::new(&codeword, f.num_vars())?;
    Ok((commitment, domain))
}

/// `Σ_x eq(z, x)·f(x)`, the sumcheck an evaluation claim becomes.
fn eq_weighted<F: IsField>(
    f: &Mle<F>,
    z: &[FieldElement<F>],
) -> Result<EqScaled<F, VirtualPolynomial<F>>, Error> {
    let inner = VirtualPolynomial::new(vec![f.clone()], vec![Term::single(0)])?;
    EqScaled::new(inner, eq_mle(z)?)
}

/// Proves `f(z) = y`.
///
/// The caller must have absorbed the commitment root and `z` into `transcript`
/// already; both sides must do the same.
pub fn prove<F, T>(
    f: &Mle<F>,
    z: &[FieldElement<F>],
    commitment: &CodewordCommitment<F>,
    domain: &Domain<F>,
    config: &EvalConfig,
    transcript: &mut T,
) -> Result<EvalProof<F>, Error>
where
    F: IsFFTField + IsPrimeField,
    FieldElement<F>: AsBytes + Sync + Send,
    T: IsTranscript<F>,
{
    let (sumcheck, alphas) = sumcheck::prove(eq_weighted(f, z)?, transcript)?;

    let codeword = encode(&lift_coefficients(f), domain)?;
    let (folded, _) = fold_codeword_k(&codeword, domain, &alphas)?;
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
pub fn verify<F, T>(
    proof: &EvalProof<F>,
    root: &Commitment,
    z: &[FieldElement<F>],
    y: FieldElement<F>,
    domain: &Domain<F>,
    config: &EvalConfig,
    transcript: &mut T,
) -> Result<(), Error>
where
    F: IsFFTField + IsPrimeField + 'static,
    FieldElement<F>: AsBytes + Sync + Send,
    T: IsTranscript<F>,
{
    let num_vars = z.len();
    // `eq` raises the degree of the plain `f` term to two.
    let claim = sumcheck::verify(&proof.sumcheck, y, num_vars, 2, transcript)?;
    let alphas = &claim.point;

    // The sumcheck's residual is eq(z, α)·f(α); the verifier knows eq.
    let eq_at = eq_eval(z, alphas)?;
    let required = claim
        .expected_evaluation
        .clone()
        .mul_by_inverse_of(&eq_at)
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
        if !verify_opening::<F>(root, q, opening) {
            return Err(Error::OpeningRejected { query: i });
        }
        if fold_coset(&opening.values, domain, q, alphas)? != proof.final_value {
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

            let (commitment, domain) = commit(&f, &config()).unwrap();
            let proof = prove(&f, &z, &commitment, &domain, &config(), &mut transcript()).unwrap();

            verify(
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

        let (commitment, domain) = commit(&f, &config()).unwrap();
        let proof = prove(&f, &z, &commitment, &domain, &config(), &mut transcript()).unwrap();

        let err = verify(
            &proof,
            &commitment.root(),
            &z,
            y + FE::one(),
            &domain,
            &config(),
            &mut transcript(),
        )
        .unwrap_err();
        assert!(matches!(err, Error::RoundSumMismatch { .. }));
    }

    #[test]
    fn a_forged_final_value_is_rejected() {
        let f = pseudo_mle(3, 17);
        let z = point(3);
        let y = f.evaluate(&z).unwrap();

        let (commitment, domain) = commit(&f, &config()).unwrap();
        let mut proof = prove(&f, &z, &commitment, &domain, &config(), &mut transcript()).unwrap();
        proof.final_value += FE::one();

        let err = verify(
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

        let (commitment, domain) = commit(&f, &config()).unwrap();
        let mut proof = prove(&f, &z, &commitment, &domain, &config(), &mut transcript()).unwrap();
        proof.openings[0].values[0] += FE::one();

        let err = verify(
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

        let (f_commitment, domain) = commit(&f, &config()).unwrap();
        // Argue g's evaluation while presenting f's commitment.
        let proof = prove(&g, &z, &f_commitment, &domain, &config(), &mut transcript()).unwrap();

        let err = verify(
            &proof,
            &f_commitment.root(),
            &z,
            g.evaluate(&z).unwrap(),
            &domain,
            &config(),
            &mut transcript(),
        )
        .unwrap_err();
        assert!(matches!(err, Error::FoldInconsistent { .. }));
    }

    #[test]
    fn a_proof_replayed_under_another_transcript_is_rejected() {
        let f = pseudo_mle(3, 31);
        let z = point(3);
        let y = f.evaluate(&z).unwrap();

        let (commitment, domain) = commit(&f, &config()).unwrap();
        let proof = prove(&f, &z, &commitment, &domain, &config(), &mut transcript()).unwrap();

        let mut other = DefaultTranscript::<F>::new(b"a-different-statement");
        assert!(
            verify(
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

        let (commitment, domain) = commit(&f, &config()).unwrap();
        let mut proof = prove(&f, &z, &commitment, &domain, &config(), &mut transcript()).unwrap();
        proof.openings.pop();

        let err = verify(
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
        let (commitment, domain) = commit(&f, &cfg).unwrap();
        assert_eq!(commitment.num_leaves(), 1 << cfg.log_blowup);
        assert_eq!(domain.log_size(), 4 + cfg.log_blowup);
    }
}
