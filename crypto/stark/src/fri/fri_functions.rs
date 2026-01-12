use super::Polynomial;
use math::field::{element::FieldElement, traits::IsField};

/// FRI polynomial folding: computes P_even(x) + beta * P_odd(x)
/// where P(x) = P_even(x^2) + x * P_odd(x^2)
pub fn fold_polynomial<F>(
    poly: &Polynomial<FieldElement<F>>,
    beta: &FieldElement<F>,
) -> Polynomial<FieldElement<F>>
where
    F: IsField,
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
pub fn fold_polynomial_doubled<F>(
    poly: &Polynomial<FieldElement<F>>,
    beta: &FieldElement<F>,
) -> Polynomial<FieldElement<F>>
where
    F: IsField,
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

/// Legacy implementation kept for benchmark comparison.
#[cfg(any(test, feature = "benchmark"))]
pub fn fold_polynomial_legacy<F>(
    poly: &Polynomial<FieldElement<F>>,
    beta: &FieldElement<F>,
) -> Polynomial<FieldElement<F>>
where
    F: IsField,
{
    use math::polynomial;

    let coefficients = poly.coefficients();
    let even: Vec<_> = coefficients.iter().step_by(2).cloned().collect();
    let odd_scaled: Vec<_> = coefficients
        .iter()
        .skip(1)
        .step_by(2)
        .map(|c| c.clone() * beta)
        .collect();

    let (even_poly, odd_poly) = polynomial::pad_with_zero_coefficients(
        &Polynomial::new(&even),
        &Polynomial::new(&odd_scaled),
    );
    even_poly + odd_poly
}

#[cfg(test)]
mod tests {
    use super::{fold_polynomial, fold_polynomial_legacy};
    use math::field::element::FieldElement;
    use math::field::fields::u64_prime_field::U64PrimeField;
    use math::polynomial::Polynomial;

    const MODULUS: u64 = 293;
    type FE = FieldElement<U64PrimeField<MODULUS>>;

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
    fn test_matches_legacy() {
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

        assert_eq!(fold_polynomial_legacy(&p0, &beta), fold_polynomial(&p0, &beta));

        let p4 = Polynomial::new(&[FE::new(1), FE::new(2), FE::new(3), FE::new(4)]);
        assert_eq!(fold_polynomial_legacy(&p4, &beta), fold_polynomial(&p4, &beta));

        let p2 = Polynomial::new(&[FE::new(10), FE::new(20)]);
        assert_eq!(fold_polynomial_legacy(&p2, &beta), fold_polynomial(&p2, &beta));
    }

    #[test]
    fn test_fold_size_2() {
        let p2 = Polynomial::new(&[FE::new(10), FE::new(20)]);
        let beta = FE::new(3);
        let result = fold_polynomial(&p2, &beta);
        assert_eq!(result, Polynomial::new(&[FE::new(70)]));
    }
}
