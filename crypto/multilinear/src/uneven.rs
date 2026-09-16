//! Sumchecks over cubes of **different heights**, proved as one.
//!
//! [`batch`](crate::batch) shares the *factors* between statements over one
//! cube. This shares the *rounds* between statements over cubes that are not
//! the same size: `k` of them on `n_1, .., n_k` variables become one sumcheck
//! of `n = max n_i` rounds instead of `k` sumchecks of `Σ n_i` rounds between
//! them. What that buys is not arithmetic — a round still sums over every
//! statement's cube — it is the **per-round cost**, which for this prover is a
//! kernel launch, a synchronization and a transcript step, and which one
//! measurement put at about half of what the rounds cost at all.
//!
//! ## How a short statement fits on the big cube
//!
//! Statement `i` is read on `n` variables as the function that **ignores the
//! last `n − n_i`**: its table with each entry repeated `2^{n−n_i}` times.
//! That extension is still multilinear and nothing is padded in memory — the
//! statement keeps its own cube and the repetition is only how the rounds count
//! it:
//!
//! - its sum over the big cube is `2^{n−n_i}` times its own, so its claim
//!   enters the batch scaled by a constant the verifier computes;
//! - while it still has variables it contributes **its own round polynomial**,
//!   at the cost of its own cube and no more;
//! - once they are bound it is a constant, contributing `2^{rest} · v_i` to
//!   every interpolation node — **degree 0**, and all the exhausted statements
//!   together are one accumulator, not one term each.
//!
//! Statements are walked tallest first, so the ones still binding variables are
//! a prefix and the exhausted ones a suffix: a cut point that advances, not a
//! test per statement per round.
//!
//! ## What the point means
//!
//! Every statement ends bound at a **prefix** of the one challenge point:
//! statement `i` at `r_1..r_{n_i}`, the tallest at all of `r`. So the identity
//! the verifier is left with carries no scale at all,
//! `G(r) = Σ λ^i f_i(r_1..r_{n_i})` — the scale is in the claim, not in the
//! point — and the openings a caller still has to discharge are nested rather
//! than unrelated.
//!
//! ## Soundness
//!
//! Two terms on top of one sumcheck's. The claims are absorbed before `λ` is
//! drawn, so a prover that lies about any of them needs `λ` to be a root of a
//! nonzero polynomial of degree `k − 1`: at most `(k−1)/|F|`. The sumcheck over
//! the combination costs the usual `d·n/|F|`, with `d` the largest statement's
//! degree. Total `(k − 1 + d·n)/|F|`.
//!
//! Over the degree-3 extension of Goldilocks that field has about `2^192`
//! elements, so for this prover's shape — some tens of statements, degree under
//! ten, cubes of at most `2^21` — the whole term is below `2^-180`. The
//! batching does not move the security level; what fixes it stays the query and
//! grinding side of the commitment scheme.

use crypto::fiat_shamir::is_transcript::IsTranscript;
use math::field::{element::FieldElement, traits::IsField};

use crate::{
    Error, challenge_powers,
    poly::SumcheckPolynomial,
    sumcheck::{self, RoundProof, SumcheckProof, round_evaluations},
};

/// `2^k` as a field element.
fn two_to<F: IsField>(k: usize) -> FieldElement<F> {
    FieldElement::<F>::from(2).pow(k as u64)
}

/// Absorbs the statement — the heights, then the claims — and expands the
/// batching challenge.
///
/// Prover and verifier both come through here so the order is written once.
/// The heights scale the claims, so they are bound rather than assumed: a
/// caller that has already bound them elsewhere pays two hashes for it.
fn absorb_statement<F, T>(
    heights: &[usize],
    claims: &[FieldElement<F>],
    transcript: &mut T,
) -> Vec<FieldElement<F>>
where
    F: IsField,
    T: IsTranscript<F>,
{
    let mut bytes = Vec::with_capacity(heights.len() * 8);
    for height in heights {
        bytes.extend_from_slice(&(*height as u64).to_le_bytes());
    }
    transcript.append_bytes(&bytes);
    for claim in claims {
        transcript.append_field_element(claim);
    }
    challenge_powers(&transcript.sample_field_element(), claims.len())
}

/// What the big cube's sum comes to, given each statement's own.
fn claimed_sum<F: IsField>(
    heights: &[usize],
    claims: &[FieldElement<F>],
    lambdas: &[FieldElement<F>],
    n: usize,
) -> FieldElement<F> {
    heights
        .iter()
        .zip(claims)
        .zip(lambdas)
        .fold(FieldElement::zero(), |acc, ((height, claim), lambda)| {
            acc + lambda * claim * two_to::<F>(n - height)
        })
}

/// Proves every statement in one sumcheck of `max num_vars` rounds.
///
/// `claims[i]` is what `Σ_x f_i(x)` over statement `i`'s **own** cube must come
/// to. They are absorbed before the batching challenge, so the prover cannot
/// pick a statement after seeing it.
///
/// Returns the proof and the whole challenge point. `parts` is left folded:
/// statement `i` is bound at `point[..n_i]`, which is where the caller reads
/// the factor values the verifier will ask it to justify.
pub fn prove<F, T, P>(
    parts: &mut [P],
    claims: &[FieldElement<F>],
    transcript: &mut T,
) -> Result<(SumcheckProof<F>, Vec<FieldElement<F>>), Error>
where
    F: IsField + 'static,
    T: IsTranscript<F>,
    P: SumcheckPolynomial<F> + Sync,
    FieldElement<F>: Send + Sync,
{
    if parts.is_empty() {
        return Err(Error::EmptyPolynomial);
    }
    if claims.len() != parts.len() {
        return Err(Error::VariableCountMismatch {
            expected: parts.len(),
            got: claims.len(),
        });
    }
    let heights: Vec<usize> = parts.iter().map(|p| p.num_vars()).collect();
    let n = heights.iter().copied().max().unwrap_or(0);
    let degree = parts.iter().map(|p| p.degree()).max().unwrap_or(1).max(1);
    let lambdas = absorb_statement(&heights, claims, transcript);

    // Tallest first, so the statements still binding variables are a prefix and
    // the exhausted ones a suffix.
    let mut order: Vec<usize> = (0..parts.len()).collect();
    order.sort_by_key(|&i| std::cmp::Reverse(heights[i]));
    // What a statement's own round polynomial is worth on the big cube. The
    // powers of lambda stay in the caller's order, which is the order the
    // verifier recombines in; this sort is only how the rounds walk.
    let weights: Vec<FieldElement<F>> = heights
        .iter()
        .zip(&lambdas)
        .map(|(height, lambda)| lambda * two_to::<F>(n - height))
        .collect();

    // The identity the verifier takes on faith. Checking it costs the pass over
    // the cube the protocol exists to skip, so it runs in debug only — and here
    // it is what catches a wrong scale, which is otherwise a proof that fails
    // to verify with nothing pointing at why.
    #[cfg(debug_assertions)]
    let mut running = claimed_sum(&heights, claims, &lambdas, n);
    let with_zero = cfg!(debug_assertions);
    let width = degree + usize::from(with_zero);

    let mut active = parts.len();
    // Σ λ^i · v_i over the statements that have run out of variables.
    let mut tail = FieldElement::<F>::zero();
    let mut rounds = Vec::with_capacity(n);
    let mut challenges = Vec::with_capacity(n);

    for round in 0..n {
        // Whatever finished binding last round joins the tail, once and for all.
        while active > 0 && parts[order[active - 1]].num_vars() == 0 {
            active -= 1;
            let i = order[active];
            tail += &lambdas[i] * parts[i].eval_at_index(0);
        }

        let mut all = vec![FieldElement::<F>::zero(); width];
        for &i in &order[..active] {
            let evaluations = round_evaluations(&parts[i], degree, with_zero);
            for (slot, value) in all.iter_mut().zip(evaluations) {
                *slot += &weights[i] * value;
            }
        }
        // The exhausted ones are one constant for the whole batch: their
        // combined value over the cube the rounds after this one still sum.
        if tail != FieldElement::zero() {
            let rest = &tail * two_to::<F>(n - round - 1);
            for slot in all.iter_mut() {
                *slot += rest.clone();
            }
        }

        let sent = all[width - degree..].to_vec();
        for e in &sent {
            transcript.append_field_element(e);
        }
        let r = transcript.sample_field_element();

        #[cfg(debug_assertions)]
        {
            debug_assert_eq!(
                &all[0] + &all[1],
                running,
                "uneven sumcheck: g(0) + g(1) is not the claim carried into round {round}"
            );
            running = sumcheck::interpolate(&all, &r);
        }

        for &i in &order[..active] {
            parts[i].fix_first_variable(&r)?;
        }
        rounds.push(RoundProof { evaluations: sent });
        challenges.push(r);
    }

    Ok((SumcheckProof { rounds }, challenges))
}

/// Checks the batch against the value each statement takes at its own prefix of
/// the point.
///
/// `values_at` is handed the whole point and returns one value per statement:
/// statement `i`'s at `point[..heights[i]]`. Where those values come from —
/// another sumcheck, a commitment opening — is the caller's, and so is
/// discharging them; this only checks that they rebuild what the rounds left.
///
/// Returns the point.
pub fn verify<F, T, V>(
    proof: &SumcheckProof<F>,
    heights: &[usize],
    claims: &[FieldElement<F>],
    degree: usize,
    values_at: V,
    transcript: &mut T,
) -> Result<Vec<FieldElement<F>>, Error>
where
    F: IsField,
    T: IsTranscript<F>,
    V: FnOnce(&[FieldElement<F>]) -> Result<Vec<FieldElement<F>>, Error>,
{
    if heights.is_empty() {
        return Err(Error::EmptyPolynomial);
    }
    if claims.len() != heights.len() {
        return Err(Error::VariableCountMismatch {
            expected: heights.len(),
            got: claims.len(),
        });
    }
    let n = heights.iter().copied().max().unwrap_or(0);
    let lambdas = absorb_statement(heights, claims, transcript);
    let claimed = claimed_sum(heights, claims, &lambdas, n);
    let claim = sumcheck::verify(proof, claimed, n, degree.max(1), transcript)?;

    let values = values_at(&claim.point)?;
    if values.len() != heights.len() {
        return Err(Error::VariableCountMismatch {
            expected: heights.len(),
            got: values.len(),
        });
    }
    // No scale here: the big-cube reading of statement `i` at `r` is its own
    // extension at `r_1..r_{n_i}`, because the variables it ignores are the
    // ones past that. The scale rode in the claim.
    let rebuilt = lambdas
        .iter()
        .zip(&values)
        .fold(FieldElement::<F>::zero(), |acc, (lambda, value)| {
            acc + lambda * value
        });
    if rebuilt != claim.expected_evaluation {
        return Err(Error::BatchMismatch);
    }

    Ok(claim.point)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crypto::fiat_shamir::default_transcript::DefaultTranscript;
    use math::field::goldilocks::GoldilocksField as F;

    use crate::{
        mle::Mle,
        poly::Composed,
        virtual_poly::{Term, VirtualPolynomial},
    };

    type FE = FieldElement<F>;

    fn transcript() -> DefaultTranscript<F> {
        DefaultTranscript::<F>::new(b"uneven-test")
    }

    /// Pseudo-random table of `2^n` entries, deterministic across runs.
    fn pseudo_mle(n: usize, seed: u64) -> Mle<F> {
        Mle::new(
            (0..(1u64 << n))
                .map(|i| FE::from((i.wrapping_mul(6364136223846793005).wrapping_add(seed)) >> 11))
                .collect(),
        )
        .unwrap()
    }

    /// `2·a·b` over two tables of `num_vars` variables — degree 2.
    fn part(num_vars: usize, seed: u64) -> VirtualPolynomial<F> {
        VirtualPolynomial::new(
            vec![pseudo_mle(num_vars, seed), pseudo_mle(num_vars, seed + 1)],
            vec![Term::new(FE::from(2), vec![0, 1])],
        )
        .unwrap()
    }

    /// A constant statement: no variables at all.
    fn flat(value: u64) -> VirtualPolynomial<F> {
        VirtualPolynomial::new(
            vec![Mle::new(vec![FE::from(value)]).unwrap()],
            vec![Term::single(0)],
        )
        .unwrap()
    }

    fn claims_of(parts: &[VirtualPolynomial<F>]) -> Vec<FE> {
        parts.iter().map(|p| p.sum_over_hypercube()).collect()
    }

    /// The verifier's side of the round trip: each statement's own extension at
    /// its own prefix of the point.
    fn values_at(parts: &[VirtualPolynomial<F>], point: &[FE]) -> Result<Vec<FE>, Error> {
        parts
            .iter()
            .map(|p| p.evaluate(&point[..p.num_vars()]))
            .collect()
    }

    /// The table read on `n` variables as the function ignoring the last
    /// `n − num_vars`: each entry repeated. This is what the protocol claims a
    /// short statement *is*, written out — only the tests pay for it.
    fn repeated(table: &Mle<F>, n: usize) -> Mle<F> {
        let shift = n - table.num_vars();
        Mle::new(
            (0..(1usize << n))
                .map(|j| table.evals()[j >> shift])
                .collect(),
        )
        .unwrap()
    }

    #[test]
    fn uneven_cubes_round_trip() {
        // Deliberately not sorted: the batch orders its own walk.
        let mut parts = vec![part(3, 1), part(5, 10), part(4, 20)];
        let heights: Vec<usize> = parts.iter().map(|p| p.num_vars()).collect();
        let claims = claims_of(&parts);
        let originals = parts.clone();

        let (proof, point) = prove(&mut parts, &claims, &mut transcript()).unwrap();
        assert_eq!(proof.rounds.len(), 5);

        let verified = verify(
            &proof,
            &heights,
            &claims,
            2,
            |p| values_at(&originals, p),
            &mut transcript(),
        )
        .unwrap();
        assert_eq!(verified, point);
    }

    #[test]
    fn it_is_the_sumcheck_of_the_repeated_tables() {
        // The whole construction, checked against writing the padding out: the
        // batch must produce exactly the proof one plain sumcheck over the
        // statements read on the big cube would.
        let mut parts = vec![part(5, 10), part(3, 1), part(4, 20)];
        let heights: Vec<usize> = parts.iter().map(|p| p.num_vars()).collect();
        let claims = claims_of(&parts);
        let n = 5;

        let (proof, point) = prove(&mut parts, &claims, &mut transcript()).unwrap();

        let mut other = transcript();
        let lambdas = absorb_statement(&heights, &claims, &mut other);
        let polys: Vec<Mle<F>> = [part(5, 10), part(3, 1), part(4, 20)]
            .iter()
            .flat_map(|p| p.polys().iter().map(|f| repeated(f, n)).collect::<Vec<_>>())
            .collect();
        let combined = Composed::new(
            polys,
            move |v: &[FE]| {
                (0..3).fold(FE::zero(), |acc, i| {
                    acc + lambdas[i] * FE::from(2) * v[2 * i] * v[2 * i + 1]
                })
            },
            2,
        )
        .unwrap();
        let (expected, expected_point) = sumcheck::prove(combined, &mut other).unwrap();

        assert_eq!(proof, expected);
        assert_eq!(point, expected_point);
    }

    #[test]
    fn a_statement_with_no_variables_is_carried_as_a_constant() {
        let mut parts = vec![part(4, 7), flat(31), part(2, 9)];
        let heights: Vec<usize> = parts.iter().map(|p| p.num_vars()).collect();
        assert_eq!(heights[1], 0);
        let claims = claims_of(&parts);
        let originals = parts.clone();

        let (proof, _) = prove(&mut parts, &claims, &mut transcript()).unwrap();
        verify(
            &proof,
            &heights,
            &claims,
            2,
            |p| values_at(&originals, p),
            &mut transcript(),
        )
        .unwrap();
    }

    #[test]
    fn statements_of_equal_height_round_trip() {
        // The degenerate case the uneven machinery must not special-case wrong.
        let mut parts = vec![part(4, 1), part(4, 5)];
        let heights: Vec<usize> = parts.iter().map(|p| p.num_vars()).collect();
        let claims = claims_of(&parts);
        let originals = parts.clone();

        let (proof, _) = prove(&mut parts, &claims, &mut transcript()).unwrap();
        verify(
            &proof,
            &heights,
            &claims,
            2,
            |p| values_at(&originals, p),
            &mut transcript(),
        )
        .unwrap();
    }

    #[test]
    fn statements_of_different_degree_round_trip() {
        // The batch runs at the worst degree; the cheaper statement is simply
        // sampled at more nodes than it needs.
        let cubic = VirtualPolynomial::new(
            vec![pseudo_mle(4, 1), pseudo_mle(4, 2), pseudo_mle(4, 3)],
            vec![Term::new(FE::from(7), vec![0, 1, 2])],
        )
        .unwrap();
        let linear = VirtualPolynomial::new(vec![pseudo_mle(2, 4)], vec![Term::single(0)]).unwrap();
        let mut parts = vec![cubic, linear];
        let heights: Vec<usize> = parts.iter().map(|p| p.num_vars()).collect();
        let claims = claims_of(&parts);
        let originals = parts.clone();

        let (proof, _) = prove(&mut parts, &claims, &mut transcript()).unwrap();
        assert_eq!(proof.rounds[0].evaluations.len(), 3);
        verify(
            &proof,
            &heights,
            &claims,
            3,
            |p| values_at(&originals, p),
            &mut transcript(),
        )
        .unwrap();
    }

    #[test]
    fn the_parts_come_back_bound_at_their_own_prefix() {
        // What the prover reads off the folded statements has to be the very
        // thing the verifier asks `values_at` for.
        let mut parts = vec![part(5, 3), part(2, 4), part(4, 5)];
        let originals = parts.clone();

        let claims = claims_of(&parts);
        let (_, point) = prove(&mut parts, &claims, &mut transcript()).unwrap();

        for (folded, original) in parts.iter().zip(&originals) {
            assert_eq!(folded.num_vars(), 0);
            assert_eq!(
                folded.eval_at_index(0),
                original.evaluate(&point[..original.num_vars()]).unwrap()
            );
        }
    }

    /// **The statement is bound to the batching challenge**, both halves of it.
    ///
    /// The claims and the heights are absorbed before `lambda` is drawn — the
    /// claims so a prover cannot pick a statement after seeing the challenge,
    /// the heights because they scale the claims. **No proof can show this**:
    /// take an absorption out of both sides and everything still verifies,
    /// because both sides stay in step; what is gone is the soundness, not the
    /// agreement. The transcript shows it, so the test is against
    /// [`absorb_statement`], which is the one place both sides go through.
    #[test]
    fn the_statement_is_bound_to_the_batching_challenge() {
        let draw = |heights: &[usize], claims: &[FE]| {
            absorb_statement(heights, claims, &mut transcript())[1]
        };
        let claims = [FE::from(41), FE::from(43)];
        let base = draw(&[4, 3], &claims);

        // A claim restated.
        let moved = [claims[0], claims[1] + FE::one()];
        assert_ne!(base, draw(&[4, 3], &moved), "the claims are not bound");

        // A height restated. They scale the claims, so they are statement too.
        assert_ne!(base, draw(&[4, 4], &claims), "the heights are not bound");

        // And the same statement draws the same challenge, or the two sides
        // would never agree in the first place.
        assert_eq!(base, draw(&[4, 3], &claims));
    }

    #[test]
    fn a_wrong_claim_is_rejected() {
        let mut parts = vec![part(4, 1), part(2, 2)];
        let heights: Vec<usize> = parts.iter().map(|p| p.num_vars()).collect();
        let claims = claims_of(&parts);
        let originals = parts.clone();
        let (proof, _) = prove(&mut parts, &claims, &mut transcript()).unwrap();

        // The short statement's claim is the one scaled by 2^{n−n_i}; getting
        // that scale wrong is the mistake this construction invites, so the
        // rejection has to come from the residual and not from a round.
        let lied = vec![claims[0], claims[1] + FE::one()];
        let err = verify(
            &proof,
            &heights,
            &lied,
            2,
            |p| values_at(&originals, p),
            &mut transcript(),
        )
        .unwrap_err();
        assert_eq!(err, Error::BatchMismatch);
    }

    #[test]
    fn a_tampered_round_is_rejected() {
        let mut parts = vec![part(4, 1), part(3, 2)];
        let heights: Vec<usize> = parts.iter().map(|p| p.num_vars()).collect();
        let claims = claims_of(&parts);
        let originals = parts.clone();
        let (mut proof, _) = prove(&mut parts, &claims, &mut transcript()).unwrap();

        proof.rounds[2].evaluations[0] += FE::one();
        let err = verify(
            &proof,
            &heights,
            &claims,
            2,
            |p| values_at(&originals, p),
            &mut transcript(),
        )
        .unwrap_err();
        assert_eq!(err, Error::BatchMismatch);
    }

    #[test]
    fn a_value_from_the_wrong_prefix_is_rejected() {
        // Reading a short statement at the *whole* point instead of its prefix
        // is the other easy mistake, and it has to fail.
        let mut parts = vec![part(4, 1), part(2, 2)];
        let heights: Vec<usize> = parts.iter().map(|p| p.num_vars()).collect();
        let claims = claims_of(&parts);
        let originals = parts.clone();
        let (proof, _) = prove(&mut parts, &claims, &mut transcript()).unwrap();

        let err = verify(
            &proof,
            &heights,
            &claims,
            2,
            |p| {
                Ok(vec![
                    originals[0].evaluate(&p[..4]).unwrap(),
                    // The last two challenges, not the first two.
                    originals[1].evaluate(&p[2..]).unwrap(),
                ])
            },
            &mut transcript(),
        )
        .unwrap_err();
        assert_eq!(err, Error::BatchMismatch);
    }

    #[test]
    fn a_proof_replayed_under_another_transcript_is_rejected() {
        let mut parts = vec![part(4, 1), part(3, 2)];
        let heights: Vec<usize> = parts.iter().map(|p| p.num_vars()).collect();
        let claims = claims_of(&parts);
        let originals = parts.clone();
        let (proof, _) = prove(&mut parts, &claims, &mut transcript()).unwrap();

        let mut other = DefaultTranscript::<F>::new(b"a-different-statement");
        let err = verify(
            &proof,
            &heights,
            &claims,
            2,
            |p| values_at(&originals, p),
            &mut other,
        )
        .unwrap_err();
        assert_eq!(err, Error::BatchMismatch);
    }

    #[test]
    fn a_claim_per_statement_is_required() {
        let mut parts = vec![part(3, 1), part(2, 2)];
        let err = prove(&mut parts, &[FE::one()], &mut transcript()).unwrap_err();
        assert_eq!(
            err,
            Error::VariableCountMismatch {
                expected: 2,
                got: 1
            }
        );
    }

    #[test]
    fn an_empty_batch_is_rejected() {
        let mut parts: Vec<VirtualPolynomial<F>> = Vec::new();
        let err = prove(&mut parts, &[], &mut transcript()).unwrap_err();
        assert_eq!(err, Error::EmptyPolynomial);
    }
}
