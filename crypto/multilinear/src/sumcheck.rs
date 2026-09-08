//! The sumcheck protocol.
//!
//! Reduces the claim `Σ_{x ∈ {0,1}^n} f(x) = S` to a single evaluation of `f`
//! at a random point. One round per variable: the prover sends the univariate
//! `g_j(t) = Σ_{rest} f(r_0..r_{j-1}, t, rest)`, the verifier checks
//! `g_j(0) + g_j(1)` against the running claim and answers with a challenge.
//!
//! Round polynomials travel as evaluations at `0, 1, .., d`, where `d` is the
//! polynomial's degree — the natural form, since the prover produces them by
//! summing over the remaining cube at each of those points.
//!
//! What the verifier is left with is a claim about `f` at the challenge point.
//! Discharging it needs an oracle for `f` there; in a full proof system that is
//! the polynomial commitment scheme. [`verify`] returns the claim rather than
//! deciding it.

use crypto::fiat_shamir::is_transcript::IsTranscript;
use math::field::{element::FieldElement, traits::IsField};

use crate::{Error, poly::SumcheckPolynomial};

/// One round: the round polynomial as evaluations at `0, 1, .., degree`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RoundProof<F: IsField> {
    pub evaluations: Vec<FieldElement<F>>,
}

/// A full sumcheck transcript.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SumcheckProof<F: IsField> {
    pub rounds: Vec<RoundProof<F>>,
}

/// What the verifier is left holding: `f(point)` must equal `expected_evaluation`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SumcheckClaim<F: IsField> {
    pub point: Vec<FieldElement<F>>,
    pub expected_evaluation: FieldElement<F>,
}

/// Interpolates a polynomial given by its values at `0, 1, .., d` and evaluates
/// it at `x`, via the Lagrange basis for that node set.
///
/// Round polynomials are small (degree = the AIR's constraint degree), so the
/// quadratic-time barycentric-free form is the cheap one here.
fn interpolate<F: IsField>(values: &[FieldElement<F>], x: &FieldElement<F>) -> FieldElement<F> {
    let n = values.len();
    let mut acc = FieldElement::<F>::zero();
    for (i, y_i) in values.iter().enumerate() {
        let x_i = FieldElement::<F>::from(i as u64);
        let mut num = FieldElement::<F>::one();
        let mut den = FieldElement::<F>::one();
        for j in 0..n {
            if i == j {
                continue;
            }
            let x_j = FieldElement::<F>::from(j as u64);
            num *= x - &x_j;
            den *= &x_i - &x_j;
        }
        // The nodes are distinct, so `den` is never zero.
        acc += y_i * num * den.inv().expect("distinct interpolation nodes");
    }
    acc
}

/// Sums `f(r_0..r_{j-1}, t, rest)` over the remaining cube, for `t = 0..=degree`.
///
/// `poly` has already been folded on the earlier variables, so its first
/// variable is the one this round binds. For `t` in `{0, 1}` the sum reads the
/// corresponding half of each table directly; for `t >= 2` each factor is
/// extended along the axis first.
fn round_evaluations<F: IsField, P: SumcheckPolynomial<F>>(
    poly: &P,
    degree: usize,
) -> Vec<FieldElement<F>> {
    let half = 1usize << (poly.num_vars() - 1);
    let mut out = Vec::with_capacity(degree + 1);

    for t in 0..=degree {
        let t_fe = FieldElement::<F>::from(t as u64);
        let mut total = FieldElement::<F>::zero();
        for j in 0..half {
            // Extend every factor along the bound axis: lo + t·(hi − lo).
            let values: Vec<FieldElement<F>> = poly
                .polys()
                .iter()
                .map(|p| {
                    let lo = &p.evals()[j];
                    let hi = &p.evals()[j + half];
                    lo + &t_fe * &(hi - lo)
                })
                .collect();
            total += poly.combine(&values);
        }
        out.push(total);
    }
    out
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
    F: IsField,
    T: IsTranscript<F>,
    P: SumcheckPolynomial<F>,
{
    let num_vars = poly.num_vars();
    let degree = poly.degree().max(1);
    let mut rounds = Vec::with_capacity(num_vars);
    let mut challenges = Vec::with_capacity(num_vars);

    for _ in 0..num_vars {
        let evaluations = round_evaluations(&poly, degree);
        for e in &evaluations {
            transcript.append_field_element(e);
        }
        let r = transcript.sample_field_element();

        poly.fix_first_variable(&r)?;
        rounds.push(RoundProof { evaluations });
        challenges.push(r);
    }

    Ok((SumcheckProof { rounds }, challenges))
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
    let degree = degree.max(1);

    let mut current = claimed_sum;
    let mut point = Vec::with_capacity(num_vars);

    for (round, r_proof) in proof.rounds.iter().enumerate() {
        if r_proof.evaluations.len() != degree + 1 {
            return Err(Error::RoundDegreeMismatch {
                round,
                expected: degree,
                got: r_proof.evaluations.len(),
            });
        }
        // g_j(0) + g_j(1) must reproduce the claim carried into this round.
        let sum = &r_proof.evaluations[0] + &r_proof.evaluations[1];
        if sum != current {
            return Err(Error::RoundSumMismatch {
                round,
                claimed: format!("{current:?}"),
                got: format!("{sum:?}"),
            });
        }

        for e in &r_proof.evaluations {
            transcript.append_field_element(e);
        }
        let r = transcript.sample_field_element();

        current = interpolate(&r_proof.evaluations, &r);
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

    #[test]
    fn a_wrong_claimed_sum_is_rejected_in_the_first_round() {
        let f = VirtualPolynomial::new(vec![pseudo_mle(3, 1)], vec![Term::single(0)]).unwrap();
        let claimed = f.sum_over_hypercube();
        let (proof, _) = prove(f, &mut transcript()).unwrap();

        let err = verify(&proof, claimed + FE::one(), 3, 1, &mut transcript()).unwrap_err();
        assert!(matches!(err, Error::RoundSumMismatch { round: 0, .. }));
    }

    #[test]
    fn a_tampered_round_polynomial_is_rejected() {
        let f = VirtualPolynomial::new(
            vec![pseudo_mle(4, 1), pseudo_mle(4, 2)],
            vec![Term::new(FE::one(), vec![0, 1])],
        )
        .unwrap();
        let claimed = f.sum_over_hypercube();
        let (mut proof, _) = prove(f, &mut transcript()).unwrap();

        // Shift one endpoint of a later round: round 0 still passes, so this
        // exercises the running-claim check rather than the initial one.
        proof.rounds[2].evaluations[0] += FE::one();
        let err = verify(&proof, claimed, 4, 2, &mut transcript()).unwrap_err();
        assert!(matches!(err, Error::RoundSumMismatch { round: 2, .. }));
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
        let (proof, _) = prove(f, &mut transcript()).unwrap();

        verify(&proof, claimed, 4, 1, &mut transcript()).unwrap();

        let mut other = DefaultTranscript::<F>::new(b"a-different-statement");
        assert!(verify(&proof, claimed, 4, 1, &mut other).is_err());
    }
}
