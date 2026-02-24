use super::Polynomial;
use math::field::{element::FieldElement, traits::IsField};

/// In-place FRI polynomial folding with fused doubling: 2 * (P_even(x) + beta * P_odd(x))
///
/// This modifies the polynomial in place, avoiding memory allocation.
/// The polynomial degree is halved after this operation.
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

#[cfg(test)]
mod tests {
    use super::fold_polynomial_doubled_inplace;
    use math::field::element::FieldElement;
    use math::field::goldilocks::GoldilocksField;
    use math::polynomial::Polynomial;

    type FE = FieldElement<GoldilocksField>;

    /// FRI polynomial folding: computes P_even(x) + beta * P_odd(x)
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
