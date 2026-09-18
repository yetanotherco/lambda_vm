//! `f` vanishes on the whole hypercube, via `Σ_x eq(r, x)·f(x) = 0`.
//!
//! The multilinear replacement for a quotient argument. `eq` adds one degree.
//! The residual claim about `f` is returned, not decided.
//!
//! **This is the reference, not the path.** The prover's zerocheck does not
//! come through here: a table's constraints go in as a compiled
//! [`batch::Rule`](crate::batch::Rule) alongside its two bus claims, so all
//! three share one pass over one factor list — which is the whole point of
//! batching them. What this module is for is saying the argument once, in the
//! shape it has on paper, and being the thing the batched version is checked
//! against. Nothing outside tests calls it.

use crypto::fiat_shamir::is_transcript::IsTranscript;
use math::field::{element::FieldElement, traits::IsField};

use crate::{
    Error,
    eq::eq_mle,
    poly::{EqScaled, SumcheckPolynomial},
    sumcheck::{self, SumcheckProof},
};

/// A zerocheck proof: the sumcheck transcript for `eq(r, ·)·C`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ZeroCheckProof<F: IsField> {
    pub sumcheck: SumcheckProof<F>,
}

/// What the verifier is left holding.
///
/// `eq(r, point)` is computable by the verifier alone, so the residual claim is
/// entirely about `C` — [`constraint_evaluation`](Self::constraint_evaluation)
/// is the value the commitment scheme must confirm.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ZeroCheckClaim<F: IsField> {
    /// The zerocheck challenge `r`.
    pub r: Vec<FieldElement<F>>,
    /// The sumcheck challenge point.
    pub point: Vec<FieldElement<F>>,
    /// Required value of `eq(r, point)·C(point)`.
    pub expected_evaluation: FieldElement<F>,
    /// `eq(r, point)`, recomputed by the verifier.
    pub eq_at_point: FieldElement<F>,
}

impl<F: IsField> ZeroCheckClaim<F> {
    /// The value `C(point)` must take, isolated from the `eq` factor.
    ///
    /// `None` when `eq(r, point)` is zero, which leaves `C(point)`
    /// unconstrained by this claim. It happens only if the sumcheck challenges
    /// land exactly on a cube corner other than `r`.
    pub fn constraint_evaluation(&self) -> Option<FieldElement<F>> {
        self.eq_at_point
            .inv()
            .ok()
            .map(|inv| &self.expected_evaluation * inv)
    }
}

/// What proving leaves the caller holding.
pub struct ZeroCheckOutput<F: IsField> {
    pub proof: ZeroCheckProof<F>,
    /// The zerocheck challenge.
    pub r: Vec<FieldElement<F>>,
    /// The sumcheck challenge point — where the residual claim about the
    /// constraint lives, and therefore where the commitment scheme must open.
    pub point: Vec<FieldElement<F>>,
}

/// Proves that `constraint` vanishes on `{0,1}^n`.
///
/// Draws `r` from the transcript first, so the prover cannot choose the
/// constraint after seeing it.
pub fn prove<F, T, P>(constraint: P, transcript: &mut T) -> Result<ZeroCheckOutput<F>, Error>
where
    F: IsField + 'static,
    T: IsTranscript<F>,
    P: SumcheckPolynomial<F> + Sync,
    FieldElement<F>: Send + Sync,
{
    let num_vars = constraint.num_vars();
    let r: Vec<FieldElement<F>> = (0..num_vars)
        .map(|_| transcript.sample_field_element())
        .collect();

    let combined = EqScaled::new(constraint, eq_mle(&r)?)?;
    let (sumcheck, point) = sumcheck::prove(combined, transcript)?;
    Ok(ZeroCheckOutput {
        proof: ZeroCheckProof { sumcheck },
        r,
        point,
    })
}

/// Verifies a zerocheck, returning the residual claim about `C`.
///
/// `constraint_degree` is the degree of `C`; `eq` adds one on top.
pub fn verify<F, T>(
    proof: &ZeroCheckProof<F>,
    num_vars: usize,
    constraint_degree: usize,
    transcript: &mut T,
) -> Result<ZeroCheckClaim<F>, Error>
where
    F: IsField + 'static,
    T: IsTranscript<F>,
{
    let r: Vec<FieldElement<F>> = (0..num_vars)
        .map(|_| transcript.sample_field_element())
        .collect();

    // The claimed sum is zero — that is the whole statement.
    let claim = sumcheck::verify(
        &proof.sumcheck,
        FieldElement::<F>::zero(),
        num_vars,
        constraint_degree + 1,
        transcript,
    )?;

    let eq_at_point = crate::eq::eq_eval(&r, &claim.point)?;
    Ok(ZeroCheckClaim {
        r,
        point: claim.point,
        expected_evaluation: claim.expected_evaluation,
        eq_at_point,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crypto::fiat_shamir::default_transcript::DefaultTranscript;
    use math::field::goldilocks::GoldilocksField as F;

    use crate::{mle::Mle, poly::Composed};

    type FE = FieldElement<F>;

    fn transcript() -> DefaultTranscript<F> {
        DefaultTranscript::<F>::new(b"zerocheck-test")
    }

    fn mle(vals: &[u64]) -> Mle<F> {
        Mle::new(vals.iter().map(|v| FE::from(*v)).collect()).unwrap()
    }

    /// `C = a·b − c`, with `c` set to the product so it vanishes everywhere.
    /// This is the shape of a real AIR constraint: a relation among columns.
    ///
    /// Rebuilt on each call rather than cloned: the rule is a closure, and a
    /// closure is not `Clone`.
    fn satisfied_constraint(n: usize) -> Composed<F, impl Fn(&[FE]) -> FE> {
        let size = 1usize << n;
        let a: Vec<u64> = (0..size as u64).map(|i| i * 3 + 1).collect();
        let b: Vec<u64> = (0..size as u64).map(|i| i * 5 + 2).collect();
        let c: Vec<u64> = a.iter().zip(&b).map(|(x, y)| x * y).collect();
        Composed::new(
            vec![mle(&a), mle(&b), mle(&c)],
            |v: &[FE]| v[0] * v[1] - v[2],
            2,
        )
        .unwrap()
    }

    #[test]
    fn a_satisfied_constraint_verifies() {
        let c = satisfied_constraint(4);
        assert_eq!(c.sum_over_hypercube(), FE::zero());

        let out = prove(satisfied_constraint(4), &mut transcript()).unwrap();
        let (proof, r_prover) = (out.proof, out.r);
        let claim = verify(&proof, 4, c.degree(), &mut transcript()).unwrap();

        assert_eq!(claim.r, r_prover);
        // The residual claim must be exactly what C evaluates to there.
        assert_eq!(
            claim.constraint_evaluation().unwrap(),
            c.evaluate(&claim.point).unwrap()
        );
    }

    #[test]
    fn the_residual_claim_factors_as_eq_times_c() {
        let c = satisfied_constraint(3);
        let proof = prove(satisfied_constraint(3), &mut transcript())
            .unwrap()
            .proof;
        let claim = verify(&proof, 3, c.degree(), &mut transcript()).unwrap();

        let c_at_point = c.evaluate(&claim.point).unwrap();
        assert_eq!(claim.eq_at_point * c_at_point, claim.expected_evaluation);
    }

    #[test]
    fn a_constraint_violated_in_one_row_is_rejected() {
        let size = 8usize;
        let a: Vec<u64> = (0..size as u64).map(|i| i * 3 + 1).collect();
        let b: Vec<u64> = (0..size as u64).map(|i| i * 5 + 2).collect();
        let mut c: Vec<u64> = a.iter().zip(&b).map(|(x, y)| x * y).collect();
        c[5] += 1; // one bad row

        let build = || {
            Composed::new(
                vec![mle(&a), mle(&b), mle(&c)],
                |v: &[FE]| v[0] * v[1] - v[2],
                2,
            )
            .unwrap()
        };
        let broken = build();
        assert_ne!(broken.sum_over_hypercube(), FE::zero());

        // The prover runs the protocol honestly on a false statement. `g(0)` is
        // derived from the claim, so nothing is rejected in the round — the lie
        // surfaces in the residual, which stops describing the constraint.
        let proof = prove(build(), &mut transcript()).unwrap().proof;
        let claim = verify(&proof, 3, broken.degree(), &mut transcript()).unwrap();
        assert_ne!(
            claim.constraint_evaluation(),
            Some(broken.evaluate(&claim.point).unwrap()),
            "a violated constraint produced a consistent claim"
        );
    }

    #[test]
    fn a_nonzero_polynomial_that_happens_to_sum_to_zero_is_still_rejected() {
        // Σ C = 0 but C is not identically zero: exactly the case a plain
        // sumcheck-for-zero would miss and eq(r, ·) is there to catch.
        let build = || {
            Composed::new(
                vec![Mle::new(vec![FE::from(7), -FE::from(7)]).unwrap()],
                |v: &[FE]| v[0],
                1,
            )
            .unwrap()
        };
        let c = build();
        assert_eq!(c.sum_over_hypercube(), FE::zero());

        let proof = prove(build(), &mut transcript()).unwrap().proof;
        let result = verify(&proof, 1, c.degree(), &mut transcript());

        // The residual claim must be inconsistent with the real polynomial:
        // that is where a non-vanishing constraint gets caught.
        let claim = result.unwrap();
        assert_ne!(
            claim.constraint_evaluation(),
            Some(c.evaluate(&claim.point).unwrap()),
            "a non-vanishing constraint produced a consistent claim"
        );
    }

    #[test]
    fn degree_accounts_for_the_eq_factor() {
        let c = satisfied_constraint(3);
        let proof = prove(satisfied_constraint(3), &mut transcript())
            .unwrap()
            .proof;
        // C has degree 2; with eq the round polynomials are degree 3, so each
        // carries three evaluations — `g(1), g(2), g(3)`, with `g(0)` derived.
        assert_eq!(c.degree(), 2);
        assert_eq!(proof.sumcheck.rounds[0].evaluations.len(), 3);
    }

    #[test]
    fn verifying_with_the_wrong_degree_is_rejected() {
        let c = satisfied_constraint(3);
        let proof = prove(satisfied_constraint(3), &mut transcript())
            .unwrap()
            .proof;
        let err = verify(&proof, 3, c.degree() + 1, &mut transcript()).unwrap_err();
        assert!(matches!(err, Error::RoundDegreeMismatch { round: 0, .. }));
    }
}
