use math::fft::cpu::{
    bit_reversing::in_place_bit_reverse_permute, roots_of_unity::get_powers_of_primitive_root_coset,
};
use math::field::{
    element::FieldElement,
    traits::{IsFFTField, IsField, IsSubFieldOf},
};
use math::polynomial::Polynomial;

/// In-place FRI polynomial folding with fused doubling: 2 * (P_even(x) + beta * P_odd(x))
///
/// This modifies the polynomial in place, avoiding memory allocation.
/// The polynomial degree is halved after this operation.
///
/// Note: This is the coefficient-form fold, retained for tests and reference.
/// Production FRI now uses `fold_evaluations_in_place` (evaluation-form).
#[allow(unused)]
pub fn fold_polynomial_doubled_inplace<F>(
    poly: &mut Polynomial<FieldElement<F>>,
    beta: &FieldElement<F>,
) where
    F: IsField,
{
    let coefficients = &mut poly.coefficients;
    if coefficients.is_empty() {
        return;
    }

    let new_len = coefficients.len().div_ceil(2);

    // Fold in place: process pairs and write results back to the beginning
    for i in 0..new_len {
        let idx = i * 2;
        let folded = if idx + 1 < coefficients.len() {
            (&coefficients[idx] + &(&coefficients[idx + 1] * beta)).double()
        } else {
            coefficients[idx].double()
        };
        coefficients[i] = folded;
    }

    // Truncate to the new length
    coefficients.truncate(new_len);
}

/// Evaluation-form FRI fold: given evaluations in bit-reversed order where
/// consecutive pairs (2j, 2j+1) are conjugates (p(x_j), p(-x_j)), compute
/// the folded evaluations: (lo + hi) + inv_twiddle[j] * zeta * (lo - hi)
/// = 2 * (p_even(x_j²) + zeta * p_odd(x_j²))
///
/// After folding, the N/2 results are evaluations on the squared coset
/// in bit-reversed order, preserving conjugate pairing for the next fold.
pub fn fold_evaluations_in_place<F: IsSubFieldOf<E>, E: IsField>(
    evals: &mut Vec<FieldElement<E>>,
    zeta: &FieldElement<E>,
    inv_twiddles: &[FieldElement<F>],
) where
    FieldElement<E>: Send + Sync,
    FieldElement<F>: Sync,
{
    let half = evals.len() / 2;

    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        // Parallel fold: split evals into pairs, compute folded value for each.
        // Write results into a new Vec to avoid aliasing (evals[j] overlaps evals[2*j]).
        let folded: Vec<FieldElement<E>> = (0..half)
            .into_par_iter()
            .map(|j| {
                let lo = &evals[2 * j];
                let hi = &evals[2 * j + 1];
                let sum = lo + hi;
                let diff = lo - hi;
                &sum + &(&inv_twiddles[j] * &(zeta * &diff))
            })
            .collect();
        evals.truncate(half);
        evals[..half].clone_from_slice(&folded);
    }

    #[cfg(not(feature = "parallel"))]
    {
        for j in 0..half {
            let lo = &evals[2 * j];
            let hi = &evals[2 * j + 1];
            let sum = lo + hi;
            let diff = lo - hi;
            evals[j] = &sum + &(&inv_twiddles[j] * &(zeta * &diff));
        }
        evals.truncate(half);
    }
}

/// Compute inverse twiddle factors for evaluation-form FRI folding.
///
/// For a coset of size N with offset g, the twiddle factors are 1/x_j where
/// x_j are the coset points at even bit-reversed positions. Specifically:
/// generate g·w^i for i=0..N/2 (half the coset points), bit-reverse with
/// (logN-1) bits, then batch-invert.
pub fn compute_coset_twiddles_inv<F: IsFFTField>(
    coset_offset: &FieldElement<F>,
    domain_size: usize,
) -> Vec<FieldElement<F>> {
    let half = domain_size / 2;
    let order = domain_size.trailing_zeros() as u64;
    let mut points = get_powers_of_primitive_root_coset(order, half, coset_offset).unwrap();
    in_place_bit_reverse_permute(&mut points);
    FieldElement::inplace_batch_inverse(&mut points).unwrap();
    points
}

/// Update inverse twiddle factors for the next FRI layer.
///
/// Between levels: new_tw[j'] = tw[2j']² (take even-indexed, square).
/// This corresponds to the squared coset offset and halved domain.
pub fn update_twiddles_in_place<F: IsField>(twiddles: &mut Vec<FieldElement<F>>) {
    let new_len = twiddles.len() / 2;
    for j in 0..new_len {
        twiddles[j] = twiddles[2 * j].square();
    }
    twiddles.truncate(new_len);
}

#[cfg(test)]
mod tests {
    use super::fold_polynomial_doubled_inplace;
    use math::field::element::FieldElement;
    use math::field::goldilocks::GoldilocksField;
    use math::polynomial::Polynomial;

    type FE = FieldElement<GoldilocksField>;

    /// FRI polynomial folding: computes P_even(x) + beta * P_odd(x)
    /// where P(x) = P_even(x^2) + x * P_odd(x^2)
    fn fold_polynomial<F>(
        poly: &Polynomial<FieldElement<F>>,
        beta: &FieldElement<F>,
    ) -> Polynomial<FieldElement<F>>
    where
        F: math::field::traits::IsField,
    {
        let coefficients = poly.coefficients();
        if coefficients.is_empty() {
            return Polynomial::new(&[]);
        }

        let mut result = Vec::with_capacity(coefficients.len().div_ceil(2));

        for chunk in coefficients.chunks(2) {
            let folded = if chunk.len() == 2 {
                &chunk[0] + &(&chunk[1] * beta)
            } else {
                chunk[0].clone()
            };
            result.push(folded);
        }

        Polynomial::new(&result)
    }

    /// FRI polynomial folding with fused doubling: 2 * (P_even(x) + beta * P_odd(x))
    ///
    /// Uses `double()` which is more efficient than multiplication by 2.
    fn fold_polynomial_doubled<F>(
        poly: &Polynomial<FieldElement<F>>,
        beta: &FieldElement<F>,
    ) -> Polynomial<FieldElement<F>>
    where
        F: math::field::traits::IsField,
    {
        let coefficients = poly.coefficients();
        if coefficients.is_empty() {
            return Polynomial::new(&[]);
        }

        let mut result = Vec::with_capacity(coefficients.len().div_ceil(2));

        for chunk in coefficients.chunks(2) {
            let folded = if chunk.len() == 2 {
                (&chunk[0] + &(&chunk[1] * beta)).double()
            } else {
                chunk[0].double()
            };
            result.push(folded);
        }

        Polynomial::new(&result)
    }

    #[test]
    fn test_fold_power_of_2() {
        let p0 = Polynomial::new(&[
            FE::new(3),
            FE::new(1),
            FE::new(2),
            FE::new(7),
            FE::new(3),
            FE::new(5),
            FE::new(4),
            FE::new(2),
        ]);
        let beta = FE::new(4);
        let p1 = fold_polynomial(&p0, &beta);
        assert_eq!(
            p1,
            Polynomial::new(&[FE::new(7), FE::new(30), FE::new(23), FE::new(12)])
        );

        let gamma = FE::new(3);
        let p2 = fold_polynomial(&p1, &gamma);
        assert_eq!(p2, Polynomial::new(&[FE::new(97), FE::new(59)]));

        let delta = FE::new(2);
        let p3 = fold_polynomial(&p2, &delta);
        assert_eq!(p3, Polynomial::new(&[FE::new(215)]));
        assert_eq!(p3.degree(), 0);
    }

    #[test]
    fn test_fold_size_2() {
        let p2 = Polynomial::new(&[FE::new(10), FE::new(20)]);
        let beta = FE::new(3);
        let result = fold_polynomial(&p2, &beta);
        assert_eq!(result, Polynomial::new(&[FE::new(70)]));
    }

    #[test]
    fn test_inplace_matches_regular() {
        let p0 = Polynomial::new(&[
            FE::new(3),
            FE::new(1),
            FE::new(2),
            FE::new(7),
            FE::new(3),
            FE::new(5),
            FE::new(4),
            FE::new(2),
        ]);
        let beta = FE::new(4);

        // Test that in-place matches regular folding with doubling
        let expected = fold_polynomial_doubled(&p0, &beta);
        let mut p_inplace = p0.clone();
        fold_polynomial_doubled_inplace(&mut p_inplace, &beta);
        assert_eq!(p_inplace, expected);

        // Test multiple folds
        let gamma = FE::new(3);
        let expected2 = fold_polynomial_doubled(&expected, &gamma);
        fold_polynomial_doubled_inplace(&mut p_inplace, &gamma);
        assert_eq!(p_inplace, expected2);

        let delta = FE::new(2);
        let expected3 = fold_polynomial_doubled(&expected2, &delta);
        fold_polynomial_doubled_inplace(&mut p_inplace, &delta);
        assert_eq!(p_inplace, expected3);
    }

    #[test]
    fn test_inplace_empty() {
        let mut p: Polynomial<FE> = Polynomial::new(&[]);
        let beta = FE::new(4);
        fold_polynomial_doubled_inplace(&mut p, &beta);
        assert!(p.coefficients.is_empty());
    }

    #[test]
    fn test_inplace_single() {
        let mut p = Polynomial::new(&[FE::new(5)]);
        let beta = FE::new(4);
        fold_polynomial_doubled_inplace(&mut p, &beta);
        assert_eq!(p, Polynomial::new(&[FE::new(10)])); // 5 * 2 = 10
    }
}
