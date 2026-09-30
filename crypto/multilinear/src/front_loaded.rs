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
    P: SumcheckPolynomial<F> + Sync,
    FieldElement<F>: Send + Sync,
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
    P: SumcheckPolynomial<F> + Sync,
    FieldElement<F>: Send + Sync,
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
        let mut sent = vec![FieldElement::<F>::zero(); degree];
        for (t, (poly, weight)) in polys.iter().zip(weights).enumerate() {
            if poly.num_vars() > 0 {
                let own = round_evaluations(poly, degree, false);
                for (s, v) in sent.iter_mut().zip(&own) {
                    *s += weight * v;
                }
                continue;
            }
            let value = match &done[t] {
                Some(value) => value.clone(),
                None => {
                    let value = poly.eval_at_index(0);
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
                poly.fix_first_variable(&r)?;
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
