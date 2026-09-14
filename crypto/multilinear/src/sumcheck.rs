//! The sumcheck protocol, reducing `Σ_x f(x) = S` to one evaluation of `f`.
//!
//! Round polynomials travel as evaluations at `1, .., d`. `g(0)` is **not**
//! sent: `g(0) + g(1)` is the claim carried into the round, which fixes it. So
//! the prover skips a whole pass over the cube and the proof loses one field
//! element per round — and the rejection that used to happen per round now
//! happens **only** against the final claim, which [`verify`] returns and the
//! caller must discharge.

use crypto::fiat_shamir::is_transcript::IsTranscript;
use math::field::{element::FieldElement, traits::IsField};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::{Error, mle::Mle, poly::SumcheckPolynomial};

/// One round: the round polynomial as evaluations at `1, .., degree`.
///
/// `g(0)` is absent by construction — see the module docs.
#[derive(
    Clone,
    Debug,
    PartialEq,
    Eq,
    serde::Serialize,
    serde::Deserialize,
    rkyv::Archive,
    rkyv::Serialize,
    rkyv::Deserialize,
)]
#[serde(bound = "")]
pub struct RoundProof<F: IsField> {
    pub evaluations: Vec<FieldElement<F>>,
}

/// A full sumcheck transcript.
#[derive(
    Clone,
    Debug,
    PartialEq,
    Eq,
    serde::Serialize,
    serde::Deserialize,
    rkyv::Archive,
    rkyv::Serialize,
    rkyv::Deserialize,
)]
#[serde(bound = "")]
pub struct SumcheckProof<F: IsField> {
    pub rounds: Vec<RoundProof<F>>,
}

/// What the verifier is left holding: `f(point)` must equal `expected_evaluation`.
///
/// Discharging this is the **only** place a sumcheck rejects, so dropping it
/// silently accepts anything.
#[must_use]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SumcheckClaim<F: IsField> {
    pub point: Vec<FieldElement<F>>,
    pub expected_evaluation: FieldElement<F>,
}

/// Lagrange-interpolates values at `0, 1, .., d` and evaluates at `x`.
///
/// Round polynomials are small, so the quadratic form is the cheap one. The
/// denominators depend only on how many nodes there are, so they go through one
/// batched inversion instead of one each — and the quadratic numerators are
/// kept rather than switching to the barycentric form, which would divide by
/// `x − x_i` and so need a special case for an `x` that lands on a node.
fn interpolate<F: IsField>(values: &[FieldElement<F>], x: &FieldElement<F>) -> FieldElement<F> {
    let n = values.len();
    let node = |j: usize| FieldElement::<F>::from(j as u64);
    let others = |i: usize, at: &FieldElement<F>| {
        (0..n)
            .filter(|&j| j != i)
            .fold(FieldElement::<F>::one(), |acc, j| acc * (at - node(j)))
    };

    let mut denominators: Vec<FieldElement<F>> = (0..n).map(|i| others(i, &node(i))).collect();
    // The nodes are distinct, so none of them is zero.
    FieldElement::inplace_batch_inverse(&mut denominators).expect("distinct interpolation nodes");

    values
        .iter()
        .zip(&denominators)
        .enumerate()
        .fold(FieldElement::zero(), |acc, (i, (y_i, inv))| {
            acc + y_i * others(i, x) * inv
        })
}

/// Sums `f(r_0..r_{j-1}, t, rest)` over the remaining cube.
/// Sums `f(r_0..r_{j-1}, t, rest)` over the remaining cube.
///
/// `t` runs over `1..=degree`, or `0..=degree` when `with_zero` — which only
/// the prover's debug self-check asks for.
///
/// `poly` has already been folded on the earlier variables, so its first
/// variable is the one this round binds.
///
/// One pass over the cube serves **every** `t`: each factor's `lo` and `hi` are
/// read once per index and the extensions come off `hi - lo`, rather than
/// re-reading the tables once per `t`. On a real trace the factors are hundreds
/// of megabytes, so the reads are the cost, not the arithmetic.
fn round_evaluations<F, P>(poly: &P, degree: usize, with_zero: bool) -> Vec<FieldElement<F>>
where
    F: IsField + 'static,
    P: SumcheckPolynomial<F> + Sync,
    FieldElement<F>: Send + Sync,
{
    let half = 1usize << (poly.num_vars() - 1);
    let first = usize::from(!with_zero);
    let steps: Vec<FieldElement<F>> = (first..=degree)
        .map(|t| FieldElement::<F>::from(t as u64))
        .collect();

    // A slice of the cube, summed independently. The rounds are the whole cost
    // of the prover, and every index is independent, so this is where the cores
    // go in.
    let slice = |range: std::ops::Range<usize>| -> Vec<FieldElement<F>> {
        let width = poly.polys().len();
        // Buffers for the whole slice. Collecting a fresh `Vec` per cube index
        // would be one heap allocation per index, which on a real trace is
        // millions of them per round and dwarfs the arithmetic.
        let mut totals = vec![FieldElement::<F>::zero(); steps.len()];
        let mut lo = vec![FieldElement::<F>::zero(); width];
        let mut hi = vec![FieldElement::<F>::zero(); width];
        let mut delta = vec![FieldElement::<F>::zero(); width];
        let mut values = vec![FieldElement::<F>::zero(); width];
        // A compiled rule's steps go here, once for the whole slice.
        let mut scratch = Vec::new();

        for j in range {
            for (k, p) in poly.polys().iter().enumerate() {
                lo[k] = p.evals()[j].clone();
                hi[k] = p.evals()[j + half].clone();
                delta[k] = &hi[k] - &lo[k];
            }
            for (total, t) in totals.iter_mut().zip(&steps) {
                // `t = 0` and `t = 1` are the halves as they are, so they skip
                // the multiplication entirely.
                if t == &FieldElement::<F>::zero() {
                    values.clone_from(&lo);
                } else if t == &FieldElement::<F>::one() {
                    values.clone_from(&hi);
                } else {
                    for (v, (l, d)) in values.iter_mut().zip(lo.iter().zip(&delta)) {
                        *v = l + t * d;
                    }
                }
                *total += poly.combine_in(&values, &mut scratch);
            }
        }
        totals
    };

    let add = |mut acc: Vec<FieldElement<F>>, part: Vec<FieldElement<F>>| {
        for (a, p) in acc.iter_mut().zip(part) {
            *a += p;
        }
        acc
    };

    #[cfg(feature = "parallel")]
    {
        if half < crate::SERIAL_BELOW {
            return slice(0..half);
        }
        let chunk = half.div_ceil(rayon::current_num_threads().max(1));
        (0..half)
            .into_par_iter()
            .step_by(chunk)
            .map(|start| slice(start..(start + chunk).min(half)))
            .reduce(|| vec![FieldElement::<F>::zero(); steps.len()], add)
    }
    #[cfg(not(feature = "parallel"))]
    {
        let _ = add;
        slice(0..half)
    }
}

/// The round polynomial's evaluations at `1..=degree` for one compiled rule
/// over `factors`, through the host round loop.
///
/// This is the reference the device round is checked against
/// (`math-cuda/tests/sumcheck.rs`): the comparison has to be against the loop
/// the prover actually runs, not a second copy of the formula.
pub fn round_evaluations_for_program<F>(
    factors: &[Mle<F>],
    program: &crate::program::Program<F>,
    degree: usize,
) -> Result<Vec<FieldElement<F>>, Error>
where
    F: IsField + 'static,
    FieldElement<F>: Send + Sync,
{
    let batched = crate::batch::Batched::new(
        factors.to_vec(),
        vec![crate::batch::Rule::compiled(degree, program.clone())],
        vec![FieldElement::one()],
    )?;
    Ok(round_evaluations(&batched, degree, false))
}

/// Runs the prover, absorbing each round polynomial and drawing each challenge
/// from `transcript`.
///
/// Returns the proof and the challenge point. The caller is responsible for
/// having absorbed the claimed sum and any statement binding beforehand.
pub fn prove<F, T, P>(
    mut poly: P,
    transcript: &mut T,
) -> Result<(SumcheckProof<F>, Vec<FieldElement<F>>), Error>
where
    F: IsField + 'static,
    T: IsTranscript<F>,
    P: SumcheckPolynomial<F> + Sync,
    FieldElement<F>: Send + Sync,
{
    let num_vars = poly.num_vars();
    let (rounds, challenges) = prove_rounds(&mut poly, num_vars, transcript)?;
    Ok((SumcheckProof { rounds }, challenges))
}

/// A group of round polynomials and the challenges they drew.
pub type RoundGroup<F> = (Vec<RoundProof<F>>, Vec<FieldElement<F>>);

/// Runs `rounds` rounds, leaving `poly` folded on them.
///
/// A chained WHIR interleaves: between groups of rounds it folds its codeword
/// and commits the successor, so it cannot run the whole sumcheck in one call.
pub fn prove_rounds<F, T, P>(
    poly: &mut P,
    rounds: usize,
    transcript: &mut T,
) -> Result<RoundGroup<F>, Error>
where
    F: IsField + 'static,
    T: IsTranscript<F>,
    P: SumcheckPolynomial<F> + Sync,
    FieldElement<F>: Send + Sync,
{
    if rounds > poly.num_vars() {
        return Err(Error::RoundCountMismatch {
            expected: poly.num_vars(),
            got: rounds,
        });
    }
    let degree = poly.degree().max(1);

    // The rounds on device when the polynomial says which program it is and the
    // device takes it. Only a whole sumcheck: a group of rounds leaves tables
    // the caller needs back, and downloading them between groups costs more
    // than the rounds do.
    let attempt = if rounds == poly.num_vars() {
        match poly.program() {
            Some(program) => {
                let cube = poly.host_cube();
                crate::gpu::prove_sumcheck(poly.polys(), program, degree, cube, |evaluations| {
                    for e in evaluations {
                        transcript.append_field_element(e);
                    }
                    transcript.sample_field_element()
                })
            }
            None => None,
        }
    } else {
        None
    };
    let mut proofs = Vec::with_capacity(rounds);
    let mut challenges = Vec::with_capacity(rounds);
    // What a device ran of this, if it ran any. It stops where the cube stops
    // being worth sending and hands the factors back folded, so what is left
    // carries on below from exactly where it left off.
    if let Some(outcome) = attempt {
        let (device_proofs, device_challenges, folded) = outcome?;
        poly.accept_folded(folded)?;
        proofs = device_proofs;
        challenges = device_challenges;
    }
    let rounds = rounds - proofs.len();
    // The identity the verifier now takes on faith. Checking it costs the pass
    // over the cube the protocol exists to skip, so it runs in debug only —
    // where it turns a silent prover bug into a local failure.
    #[cfg(debug_assertions)]
    let mut running: Option<FieldElement<F>> = None;

    for _ in 0..rounds {
        let all = round_evaluations(poly, degree, cfg!(debug_assertions));
        let sent = all[all.len() - degree..].to_vec();
        for e in &sent {
            transcript.append_field_element(e);
        }
        let r = transcript.sample_field_element();

        #[cfg(debug_assertions)]
        {
            let sum = &all[0] + &all[1];
            if let Some(expected) = &running {
                debug_assert_eq!(
                    &sum, expected,
                    "sumcheck: g(0) + g(1) is not the claim carried into the round"
                );
            }
            running = Some(interpolate(&all, &r));
        }

        poly.fix_first_variable(&r)?;
        proofs.push(RoundProof { evaluations: sent });
        challenges.push(r);
    }

    Ok((proofs, challenges))
}

/// Checks every round against the running claim and returns the final claim.
///
/// Verifying `claimed_sum` requires one more step the caller must perform:
/// obtain `f` at [`SumcheckClaim::point`] and compare it against
/// [`SumcheckClaim::expected_evaluation`].
pub fn verify<F, T>(
    proof: &SumcheckProof<F>,
    claimed_sum: FieldElement<F>,
    num_vars: usize,
    degree: usize,
    transcript: &mut T,
) -> Result<SumcheckClaim<F>, Error>
where
    F: IsField,
    T: IsTranscript<F>,
{
    if proof.rounds.len() != num_vars {
        return Err(Error::RoundCountMismatch {
            expected: num_vars,
            got: proof.rounds.len(),
        });
    }
    verify_rounds(&proof.rounds, claimed_sum, degree, transcript)
}

/// Verifies a group of rounds against a running claim.
///
/// The returned claim carries this group's challenges and the claim it leaves,
/// which is what the next group starts from. Round indices in errors are
/// relative to the group.
pub fn verify_rounds<F, T>(
    rounds: &[RoundProof<F>],
    claimed_sum: FieldElement<F>,
    degree: usize,
    transcript: &mut T,
) -> Result<SumcheckClaim<F>, Error>
where
    F: IsField,
    T: IsTranscript<F>,
{
    let degree = degree.max(1);

    let mut current = claimed_sum;
    let mut point = Vec::with_capacity(rounds.len());

    for (round, r_proof) in rounds.iter().enumerate() {
        if r_proof.evaluations.len() != degree {
            return Err(Error::RoundDegreeMismatch {
                round,
                expected: degree,
                got: r_proof.evaluations.len(),
            });
        }
        // `g(0)` is recovered rather than checked: `g(0) + g(1)` is the claim
        // carried in. So nothing is rejected here, and everything rides on the
        // claim this returns.
        let mut all = Vec::with_capacity(degree + 1);
        all.push(&current - &r_proof.evaluations[0]);
        all.extend(r_proof.evaluations.iter().cloned());

        for e in &r_proof.evaluations {
            transcript.append_field_element(e);
        }
        let r = transcript.sample_field_element();

        current = interpolate(&all, &r);
        point.push(r);
    }

    Ok(SumcheckClaim {
        point,
        expected_evaluation: current,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crypto::fiat_shamir::default_transcript::DefaultTranscript;
    use math::field::goldilocks::GoldilocksField as F;

    use crate::{
        mle::Mle,
        virtual_poly::{Term, VirtualPolynomial},
    };

    type FE = FieldElement<F>;

    fn transcript() -> DefaultTranscript<F> {
        DefaultTranscript::<F>::new(b"sumcheck-test")
    }

    fn mle(vals: &[u64]) -> Mle<F> {
        Mle::new(vals.iter().map(|v| FE::from(*v)).collect()).unwrap()
    }

    /// Pseudo-random table of `2^n` entries, deterministic across runs.
    fn pseudo_mle(n: usize, seed: u64) -> Mle<F> {
        let vals: Vec<u64> = (0..(1u64 << n))
            .map(|i| (i.wrapping_mul(6364136223846793005).wrapping_add(seed)) >> 11)
            .collect();
        mle(&vals)
    }

    #[test]
    fn interpolation_reproduces_the_nodes() {
        let values: Vec<FE> = [3u64, 1, 4, 1].iter().map(|v| FE::from(*v)).collect();
        for (i, v) in values.iter().enumerate() {
            assert_eq!(interpolate(&values, &FE::from(i as u64)), *v);
        }
    }

    #[test]
    fn interpolation_of_a_line_is_affine() {
        // g(t) = 5 + 2t sampled at 0,1 must give 5 + 2·7 at t = 7.
        let values = vec![FE::from(5), FE::from(7)];
        assert_eq!(interpolate(&values, &FE::from(7)), FE::from(19));
    }

    #[test]
    fn linear_polynomial_round_trips() {
        let f = VirtualPolynomial::new(vec![pseudo_mle(4, 1)], vec![Term::single(0)]).unwrap();
        let claimed = f.sum_over_hypercube();
        let num_vars = f.num_vars();
        let degree = f.degree();

        let (proof, _) = prove(f.clone(), &mut transcript()).unwrap();
        let claim = verify(&proof, claimed, num_vars, degree, &mut transcript()).unwrap();

        assert_eq!(f.evaluate(&claim.point).unwrap(), claim.expected_evaluation);
    }

    #[test]
    fn product_of_three_polynomials_round_trips() {
        let f = VirtualPolynomial::new(
            vec![pseudo_mle(5, 1), pseudo_mle(5, 2), pseudo_mle(5, 3)],
            vec![Term::new(FE::from(7), vec![0, 1, 2])],
        )
        .unwrap();
        let claimed = f.sum_over_hypercube();
        let (num_vars, degree) = (f.num_vars(), f.degree());
        assert_eq!(degree, 3);

        let (proof, challenges) = prove(f.clone(), &mut transcript()).unwrap();
        let claim = verify(&proof, claimed, num_vars, degree, &mut transcript()).unwrap();

        // Prover and verifier must derive the same challenges from the transcript.
        assert_eq!(challenges, claim.point);
        assert_eq!(f.evaluate(&claim.point).unwrap(), claim.expected_evaluation);
    }

    #[test]
    fn sum_of_terms_of_mixed_degree_round_trips() {
        let f = VirtualPolynomial::new(
            vec![pseudo_mle(4, 10), pseudo_mle(4, 20), pseudo_mle(4, 30)],
            vec![
                Term::new(FE::from(2), vec![0, 1]),
                Term::new(FE::from(3), vec![2]),
                Term::new(FE::from(5), vec![]),
            ],
        )
        .unwrap();
        let claimed = f.sum_over_hypercube();
        let (num_vars, degree) = (f.num_vars(), f.degree());

        let (proof, _) = prove(f.clone(), &mut transcript()).unwrap();
        let claim = verify(&proof, claimed, num_vars, degree, &mut transcript()).unwrap();

        assert_eq!(f.evaluate(&claim.point).unwrap(), claim.expected_evaluation);
    }

    #[test]
    fn single_variable_round_trips() {
        let f = VirtualPolynomial::new(vec![mle(&[3, 11])], vec![Term::single(0)]).unwrap();
        let claimed = f.sum_over_hypercube();
        assert_eq!(claimed, FE::from(14));

        let (proof, _) = prove(f.clone(), &mut transcript()).unwrap();
        let claim = verify(&proof, claimed, 1, 1, &mut transcript()).unwrap();
        assert_eq!(f.evaluate(&claim.point).unwrap(), claim.expected_evaluation);
    }

    /// `g(0)` is derived from the claim, so a wrong claim is not caught in the
    /// round — it is caught by the residual, which stops being the
    /// polynomial's value. Every caller discharges that; this is the property
    /// they rely on.
    #[test]
    fn a_wrong_claimed_sum_corrupts_the_residual() {
        let f = VirtualPolynomial::new(vec![pseudo_mle(3, 1)], vec![Term::single(0)]).unwrap();
        let claimed = f.sum_over_hypercube();
        let (proof, _) = prove(f.clone(), &mut transcript()).unwrap();

        let honest = verify(&proof, claimed, 3, 1, &mut transcript()).unwrap();
        assert_eq!(
            f.evaluate(&honest.point).unwrap(),
            honest.expected_evaluation
        );

        let lied = verify(&proof, claimed + FE::one(), 3, 1, &mut transcript()).unwrap();
        assert_ne!(f.evaluate(&lied.point).unwrap(), lied.expected_evaluation);
    }

    #[test]
    fn a_tampered_round_polynomial_corrupts_the_residual() {
        let f = VirtualPolynomial::new(
            vec![pseudo_mle(4, 1), pseudo_mle(4, 2)],
            vec![Term::new(FE::one(), vec![0, 1])],
        )
        .unwrap();
        let claimed = f.sum_over_hypercube();
        let (mut proof, _) = prove(f.clone(), &mut transcript()).unwrap();

        proof.rounds[2].evaluations[0] += FE::one();
        let claim = verify(&proof, claimed, 4, 2, &mut transcript()).unwrap();
        assert_ne!(f.evaluate(&claim.point).unwrap(), claim.expected_evaluation);
    }

    #[test]
    fn a_proof_with_the_wrong_round_count_is_rejected() {
        let f = VirtualPolynomial::new(vec![pseudo_mle(3, 1)], vec![Term::single(0)]).unwrap();
        let claimed = f.sum_over_hypercube();
        let (mut proof, _) = prove(f, &mut transcript()).unwrap();
        proof.rounds.pop();

        let err = verify(&proof, claimed, 3, 1, &mut transcript()).unwrap_err();
        assert_eq!(
            err,
            Error::RoundCountMismatch {
                expected: 3,
                got: 2
            }
        );
    }

    #[test]
    fn a_round_polynomial_of_the_wrong_degree_is_rejected() {
        let f = VirtualPolynomial::new(vec![pseudo_mle(3, 1)], vec![Term::single(0)]).unwrap();
        let claimed = f.sum_over_hypercube();
        let (mut proof, _) = prove(f, &mut transcript()).unwrap();
        proof.rounds[0].evaluations.push(FE::from(1));

        let err = verify(&proof, claimed, 3, 1, &mut transcript()).unwrap_err();
        assert!(matches!(err, Error::RoundDegreeMismatch { round: 0, .. }));
    }

    #[test]
    fn a_different_transcript_seed_yields_a_different_point() {
        // The challenge point is bound to the statement, not just to the
        // polynomial: proving the same claim under a different seed must land
        // somewhere else.
        let f = VirtualPolynomial::new(vec![pseudo_mle(4, 1)], vec![Term::single(0)]).unwrap();

        let (_, a) = prove(f.clone(), &mut transcript()).unwrap();
        let mut other = DefaultTranscript::<F>::new(b"a-different-statement");
        let (_, b) = prove(f, &mut other).unwrap();

        assert_ne!(a, b);
    }

    #[test]
    fn a_proof_replayed_under_another_transcript_is_rejected() {
        // Fiat-Shamir binding: the verifier redraws challenges, so a proof
        // lifted onto a different statement stops matching.
        let f = VirtualPolynomial::new(vec![pseudo_mle(4, 1)], vec![Term::single(0)]).unwrap();
        let claimed = f.sum_over_hypercube();
        let (proof, _) = prove(f.clone(), &mut transcript()).unwrap();

        let honest = verify(&proof, claimed, 4, 1, &mut transcript()).unwrap();
        assert_eq!(
            f.evaluate(&honest.point).unwrap(),
            honest.expected_evaluation
        );

        // Replayed, the verifier redraws different challenges, so the residual
        // stops describing the polynomial.
        let mut other = DefaultTranscript::<F>::new(b"a-different-statement");
        let replayed = verify(&proof, claimed, 4, 1, &mut other).unwrap();
        assert_ne!(
            f.evaluate(&replayed.point).unwrap(),
            replayed.expected_evaluation
        );
    }
}
