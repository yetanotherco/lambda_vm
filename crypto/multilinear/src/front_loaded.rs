//! One sumcheck over several polynomials of DIFFERENT sizes, front-loaded
//! (SWIRL Protocol 2.7.3; D-BATCH §2.2, phase C).
//!
//! Polynomial `t` has `n_t` variables and weight `w_t`. It is padded virtually
//! to `n_max` by the product of the variables it lacks:
//!
//! ```text
//! f̃_t(X) = f_t(X_{<n_t}) · Π_{j ≥ n_t} X_j
//! ```
//!
//! which sums over the cube to exactly `Σ f_t` — only the all-ones corner of the
//! padding survives — so the batched claim is `Σ_t w_t·claim_t` with no scaling.
//! The first `n_t` rounds bind `f_t`'s own variables, so its point is the
//! PREFIX `r_{<n_t}` of the shared one. After that its round polynomial is
//! `v_t·Π_{n_t ≤ j' < j} r_{j'}·X`, linear in `X`, with `v_t = f_t(r_{<n_t})`.
//!
//! The verifier's last step is the caller's: the residual must be
//! `Σ_t w_t·f_t(r_{<n_t})·Π_{j ≥ n_t} r_j` ([`padding_product`]).
//!
//! Soundness: SWIRL Thm 4.1.1 (the RBR error of a front-loaded batch is that of
//! the tallest sumcheck), so each round costs `degree/|F|`.

use crypto::fiat_shamir::is_transcript::IsTranscript;
use math::field::{element::FieldElement, traits::IsField};

use crate::{
    Error,
    poly::SumcheckPolynomial,
    sumcheck::{RoundProof, SumcheckProof, round_evaluations},
};

/// `Π_{j = vars}^{n_max − 1} point_j`: what a polynomial of `vars` variables
/// is padded by at `point` (one when it is the tallest).
pub fn padding_product<F: IsField>(point: &[FieldElement<F>], vars: usize) -> FieldElement<F> {
    point
        .iter()
        .skip(vars)
        .fold(FieldElement::one(), |acc, r| acc * r)
}

/// One polynomial's side of a front-loaded batch: its round messages on
/// demand, bound by the batch's shared challenges.
///
/// The host's is any [`SumcheckPolynomial`]; a device-resident table brings
/// its own (the fused zerocheck's stepper), and the batch cannot tell them
/// apart — which is what makes a device proof the host reference's bytes.
pub trait RoundStepper<F: IsField> {
    /// Variables left to bind.
    fn num_vars(&self) -> usize;
    /// The degree of its round polynomials.
    fn degree(&self) -> usize;
    /// This round's polynomial at `1..=degree` (`degree` at least its own).
    fn round(&mut self, degree: usize) -> Result<Vec<FieldElement<F>>, Error>;
    /// Binds this round's variable to `r`.
    fn bind(&mut self, r: &FieldElement<F>) -> Result<(), Error>;
    /// Its value once every variable is bound.
    fn value(&self) -> Result<FieldElement<F>, Error>;
    /// Queues this round's work on a device without waiting for it; the
    /// batch queues every table's before reading any back, so a round costs
    /// one wait rather than one a table.
    fn prefetch(&mut self) -> Result<(), Error> {
        Ok(())
    }
}

/// A host polynomial's rounds: [`round_evaluations`] and a fold, as the
/// plain sumcheck runs them.
macro_rules! host_stepper {
    ($($ty:ty),*) => {$(
        impl<F> RoundStepper<F> for $ty
        where
            F: IsField + 'static,
            FieldElement<F>: Send + Sync,
        {
            fn num_vars(&self) -> usize {
                SumcheckPolynomial::num_vars(self)
            }

            fn degree(&self) -> usize {
                SumcheckPolynomial::degree(self)
            }

            fn round(&mut self, degree: usize) -> Result<Vec<FieldElement<F>>, Error> {
                Ok(round_evaluations(self, degree, false))
            }

            fn bind(&mut self, r: &FieldElement<F>) -> Result<(), Error> {
                self.fix_first_variable(r)
            }

            fn value(&self) -> Result<FieldElement<F>, Error> {
                if SumcheckPolynomial::num_vars(self) != 0 {
                    return Err(Error::VariableCountMismatch {
                        expected: 0,
                        got: SumcheckPolynomial::num_vars(self),
                    });
                }
                Ok(self.eval_at_index(0))
            }
        }
    )*};
}

host_stepper!(
    crate::batch::Batched<'_, F>,
    crate::virtual_poly::VirtualPolynomial<F>
);

/// One table's rule in the batched constraint sumcheck (D-BATCH phase C): its
/// factors and weights on the host, or its fused rounds on a device down to
/// today's crossover and the host's over what the card folded from there.
/// Either way the same messages, so a device proof is the host reference's.
pub enum TableRounds<'a, E: IsField> {
    Host(crate::batch::Batched<'a, E>),
    #[cfg(feature = "cuda")]
    Fused {
        stepper: Option<Box<crate::gpu_fused::FusedStepper<'a, E>>>,
        /// The rules and weights, adopting the factors at the crossover.
        batched: crate::batch::Batched<'a, E>,
        /// Variables left to bind.
        vars: usize,
    },
}

impl<'a, E> TableRounds<'a, E>
where
    E: IsField + 'static,
    FieldElement<E>: Send + Sync,
{
    /// A table's rounds, on its device factors when the fused rounds take
    /// them, else on the host over `factors()` and the two weights built from
    /// their points.
    ///
    /// `claim` is the table's own `Σ λ^j·claim_j` — what the fused rounds
    /// carry to reconstruct `A(1)`. The rules are `[zerocheck, numerator,
    /// denominator]` and `lambdas` their `[1, λ, λ²]`, today's batch.
    #[allow(clippy::too_many_arguments)]
    pub fn new<B, P>(
        rules: Vec<crate::batch::Rule<'a, E>>,
        lambdas: Vec<FieldElement<E>>,
        claim: &FieldElement<E>,
        weights: [&[FieldElement<E>]; 2],
        num_vars: usize,
        device: Option<&'a crate::gpu::DeviceFactors>,
        fused: Option<crate::gpu_fused::FusedInput<'_, B, E>>,
        factors: P,
    ) -> Result<Self, Error>
    where
        B: IsField + math::field::traits::IsSubFieldOf<E> + 'static,
        P: FnOnce() -> Result<Vec<crate::mle::Mle<E>>, Error>,
    {
        #[cfg(feature = "cuda")]
        if let (Some(device), Some(input)) = (device, fused.as_ref())
            && crate::gpu_fused::argue_fused()
        {
            let degree = rules
                .iter()
                .map(crate::batch::Rule::degree)
                .max()
                .unwrap_or(0)
                .max(1);
            let check = crate::gpu_fused::argue_fused_xcheck();
            match crate::gpu_fused::FusedStepper::new(device, input, &lambdas, claim, degree, check)
            {
                Some(Ok(stepper)) => {
                    let batched = crate::batch::Batched::new(Vec::new(), rules, lambdas)?;
                    return Ok(Self::Fused {
                        stepper: Some(Box::new(stepper)),
                        batched,
                        vars: num_vars,
                    });
                }
                Some(Err(error)) => return Err(error),
                None => {
                    crate::gpu_fused::note_decline();
                    crate::whir_split::bump(&crate::whir_split::ZC_FUSED_DECLINED);
                }
            }
        }
        let _ = (device, &fused, claim, num_vars);
        let mut polys = factors()?;
        polys.push(crate::eq::eq_mle(weights[0])?);
        polys.push(crate::eq::eq_mle(weights[1])?);
        Ok(Self::Host(crate::batch::Batched::new(
            polys, rules, lambdas,
        )?))
    }

    /// Every factor slot, once all are bound: the trace's, then the weights.
    pub fn polys(&self) -> &[crate::mle::Mle<E>] {
        match self {
            Self::Host(batched) => batched.polys(),
            #[cfg(feature = "cuda")]
            Self::Fused { batched, .. } => batched.polys(),
        }
    }

    /// Whether the card runs its rounds.
    pub fn on_device(&self) -> bool {
        match self {
            Self::Host(_) => false,
            #[cfg(feature = "cuda")]
            Self::Fused { .. } => true,
        }
    }
}

impl<E> RoundStepper<E> for TableRounds<'_, E>
where
    E: IsField + 'static,
    FieldElement<E>: Send + Sync,
{
    fn num_vars(&self) -> usize {
        match self {
            Self::Host(batched) => SumcheckPolynomial::num_vars(batched),
            #[cfg(feature = "cuda")]
            Self::Fused { vars, .. } => *vars,
        }
    }

    fn degree(&self) -> usize {
        match self {
            Self::Host(batched) => SumcheckPolynomial::degree(batched),
            #[cfg(feature = "cuda")]
            Self::Fused { batched, .. } => SumcheckPolynomial::degree(batched),
        }
    }

    fn round(&mut self, degree: usize) -> Result<Vec<FieldElement<E>>, Error> {
        match self {
            Self::Host(batched) => Ok(round_evaluations(batched, degree, false)),
            #[cfg(feature = "cuda")]
            Self::Fused {
                stepper, batched, ..
            } => match stepper {
                None => Ok(round_evaluations(batched, degree, false)),
                Some(stepper) => {
                    let g = stepper.message()?;
                    if g.len() > degree {
                        return Err(Error::RoundDegreeMismatch {
                            round: 0,
                            expected: degree,
                            got: g.len(),
                        });
                    }
                    // Past its own degree the message is the same polynomial,
                    // read at more nodes: `g(0)` from the claim, then
                    // interpolation.
                    let mut all = Vec::with_capacity(g.len() + 1);
                    all.push(stepper.claim() - &g[0]);
                    all.extend(g.iter().cloned());
                    let mut out = g;
                    for node in out.len() + 1..=degree {
                        out.push(crate::sumcheck::interpolate(
                            &all,
                            &FieldElement::<E>::from(node as u64),
                        ));
                    }
                    Ok(out)
                }
            },
        }
    }

    fn bind(&mut self, r: &FieldElement<E>) -> Result<(), Error> {
        match self {
            Self::Host(batched) => batched.fix_first_variable(r),
            #[cfg(feature = "cuda")]
            Self::Fused {
                stepper,
                batched,
                vars,
            } => {
                *vars -= 1;
                match stepper.as_mut() {
                    None => batched.fix_first_variable(r),
                    Some(on_card) => {
                        on_card.bind(r)?;
                        if on_card.rounds_left() == 0 {
                            let (factors, _) = stepper
                                .take()
                                .ok_or(Error::DeviceFailed { stage: "fused" })?
                                .into_factors()?;
                            batched.adopt(factors)?;
                        }
                        Ok(())
                    }
                }
            }
        }
    }

    fn prefetch(&mut self) -> Result<(), Error> {
        match self {
            Self::Host(_) => Ok(()),
            #[cfg(feature = "cuda")]
            Self::Fused { stepper, .. } => match stepper {
                Some(stepper) => stepper.prefetch(),
                None => Ok(()),
            },
        }
    }

    fn value(&self) -> Result<FieldElement<E>, Error> {
        if RoundStepper::num_vars(self) != 0 {
            return Err(Error::VariableCountMismatch {
                expected: 0,
                got: RoundStepper::num_vars(self),
            });
        }
        let batched = match self {
            Self::Host(batched) => batched,
            #[cfg(feature = "cuda")]
            Self::Fused { batched, .. } => batched,
        };
        Ok(batched.eval_at_index(0))
    }
}

/// ⛔ A FAULT for the negative test: the named polynomial is padded the `2^Δ`
/// way — constant in the variables it lacks, its round polynomial `v·2^{rest}` —
/// which is a different batch than the one the verifier checks.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[doc(hidden)]
pub struct PaddingFault {
    pub scaled: Option<usize>,
}

/// Proves `Σ_t w_t·Σ_x f̃_t(x)` over `n_max = max_t n_t` rounds of `degree`,
/// on the host, leaving every polynomial bound at its own prefix of the point.
///
/// `degree` must bound every polynomial's (and be at least one, for the
/// padding's `X`). The caller has absorbed the claims and drawn the weights.
pub fn prove<F, T, P>(
    polys: &mut [P],
    weights: &[FieldElement<F>],
    degree: usize,
    transcript: &mut T,
) -> Result<(SumcheckProof<F>, Vec<FieldElement<F>>), Error>
where
    F: IsField + 'static,
    T: IsTranscript<F>,
    P: RoundStepper<F>,
{
    prove_with(polys, weights, degree, transcript, PaddingFault::default())
}

/// [`prove`], with a fault armed only by the negative test.
#[doc(hidden)]
pub fn prove_with<F, T, P>(
    polys: &mut [P],
    weights: &[FieldElement<F>],
    degree: usize,
    transcript: &mut T,
    fault: PaddingFault,
) -> Result<(SumcheckProof<F>, Vec<FieldElement<F>>), Error>
where
    F: IsField + 'static,
    T: IsTranscript<F>,
    P: RoundStepper<F>,
{
    if polys.len() != weights.len() {
        return Err(Error::VariableCountMismatch {
            expected: polys.len(),
            got: weights.len(),
        });
    }
    let degree = degree.max(1);
    if let Some(p) = polys.iter().find(|p| p.degree() > degree) {
        return Err(Error::RoundDegreeMismatch {
            round: 0,
            expected: degree,
            got: p.degree(),
        });
    }
    let rounds = polys.iter().map(|p| p.num_vars()).max().unwrap_or(0);
    let steps: Vec<FieldElement<F>> = (1..=degree)
        .map(|t| FieldElement::<F>::from(t as u64))
        .collect();
    // Per polynomial, once bound: its value, times the challenges drawn since.
    let mut done: Vec<Option<FieldElement<F>>> = vec![None; polys.len()];
    let mut proofs = Vec::with_capacity(rounds);
    let mut point = Vec::with_capacity(rounds);

    for j in 0..rounds {
        for poly in polys.iter_mut() {
            if poly.num_vars() > 0 {
                poly.prefetch()?;
            }
        }
        let mut sent = vec![FieldElement::<F>::zero(); degree];
        for (t, (poly, weight)) in polys.iter_mut().zip(weights).enumerate() {
            if poly.num_vars() > 0 {
                let own = poly.round(degree)?;
                if own.len() != degree {
                    return Err(Error::RoundDegreeMismatch {
                        round: j,
                        expected: degree,
                        got: own.len(),
                    });
                }
                for (s, v) in sent.iter_mut().zip(&own) {
                    *s += weight * v;
                }
                continue;
            }
            let value = match &done[t] {
                Some(value) => value.clone(),
                None => {
                    let value = poly.value()?;
                    done[t] = Some(value.clone());
                    value
                }
            };
            if fault.scaled == Some(t) {
                let rest = FieldElement::<F>::from(1u64 << (rounds - 1 - j));
                for s in sent.iter_mut() {
                    *s += weight * &value * &rest;
                }
                continue;
            }
            for (s, x) in sent.iter_mut().zip(&steps) {
                *s += weight * &value * x;
            }
        }
        for e in &sent {
            transcript.append_field_element(e);
        }
        let r: FieldElement<F> = transcript.sample_field_element();
        for (t, poly) in polys.iter_mut().enumerate() {
            if poly.num_vars() > 0 {
                poly.bind(&r)?;
            } else if let Some(value) = done[t].as_mut() {
                *value *= &r;
            }
        }
        proofs.push(RoundProof { evaluations: sent });
        point.push(r);
    }
    Ok((SumcheckProof { rounds: proofs }, point))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        mle::Mle,
        sumcheck,
        virtual_poly::{Term, VirtualPolynomial},
    };
    use crypto::fiat_shamir::default_transcript::DefaultTranscript;
    use math::field::goldilocks::GoldilocksField as F;

    type FE = FieldElement<F>;

    fn transcript() -> DefaultTranscript<F> {
        DefaultTranscript::<F>::new(b"front-loaded-test")
    }

    fn pseudo_mle(n: usize, seed: u64) -> Mle<F> {
        Mle::new(
            (0..(1u64 << n))
                .map(|i| FE::from((i.wrapping_mul(6364136223846793005).wrapping_add(seed)) >> 11))
                .collect(),
        )
        .unwrap()
    }

    /// `a·b·c` over `n` variables — degree three.
    fn cubic(n: usize, seed: u64) -> VirtualPolynomial<F> {
        VirtualPolynomial::new(
            vec![
                pseudo_mle(n, seed),
                pseudo_mle(n, seed + 1),
                pseudo_mle(n, seed + 2),
            ],
            vec![Term::new(FE::one(), vec![0, 1, 2])],
        )
        .unwrap()
    }

    fn run(heights: &[usize], fault: PaddingFault) -> Result<(), String> {
        let polys: Vec<VirtualPolynomial<F>> = heights
            .iter()
            .enumerate()
            .map(|(t, &n)| cubic(n, 7 * t as u64))
            .collect();
        let weights: Vec<FE> = (0..heights.len()).map(|t| FE::from(3 + t as u64)).collect();
        let claim = polys
            .iter()
            .zip(&weights)
            .fold(FE::zero(), |acc, (p, w)| acc + w * p.sum_over_hypercube());
        let n_max = heights.iter().copied().max().unwrap_or(0);

        let mut bound = polys.clone();
        let (proof, point) = prove_with(&mut bound, &weights, 3, &mut transcript(), fault)
            .map_err(|e| format!("{e:?}"))?;
        let checked = sumcheck::verify(&proof, claim, n_max, 3, &mut transcript())
            .map_err(|e| format!("{e:?}"))?;
        assert_eq!(checked.point, point);
        let want = polys
            .iter()
            .zip(&weights)
            .zip(heights)
            .fold(FE::zero(), |acc, ((p, w), &n)| {
                acc + w * p.evaluate(&point[..n]).unwrap() * padding_product(&point, n)
            });
        // What the prover's own bound polynomials say agrees with the direct
        // evaluation at the prefix.
        for ((b, p), &n) in bound.iter().zip(&polys).zip(heights) {
            assert_eq!(b.eval_at_index(0), p.evaluate(&point[..n]).unwrap());
        }
        if want == checked.expected_evaluation {
            Ok(())
        } else {
            Err("residual".into())
        }
    }

    /// Polynomials of different heights, one with no variables, batched into
    /// one sumcheck: the residual is each at its prefix, padded by the rest.
    #[test]
    fn polynomials_of_different_heights_batch_front_loaded() {
        run(&[4, 1, 0, 4, 2], PaddingFault::default()).unwrap();
        run(&[3], PaddingFault::default()).unwrap();
        run(&[0, 0], PaddingFault::default()).unwrap();
    }

    /// A prover padding one short polynomial the `2^Δ` way is refused.
    #[test]
    fn a_polynomial_padded_by_scaling_is_refused() {
        assert_eq!(
            run(&[4, 1, 4], PaddingFault { scaled: Some(1) }),
            Err("residual".into())
        );
    }

    #[test]
    fn the_padding_product_is_the_suffix() {
        let point = [FE::from(2), FE::from(3), FE::from(5)];
        assert_eq!(padding_product(&point, 0), FE::from(30));
        assert_eq!(padding_product(&point, 2), FE::from(5));
        assert_eq!(padding_product(&point, 3), FE::one());
    }
}
