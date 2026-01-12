use super::Polynomial;
use math::field::{element::FieldElement, traits::IsField};

/// Optimized FRI fold: single allocation, no intermediate polynomials
/// Computes P_even(x) + beta * P_odd(x) where P(x) = P_even(x²) + x*P_odd(x²)
///
/// Note: FRI polynomial lengths are always powers of 2, and fold requires at least 2 coefficients
pub fn fold_polynomial<F>(
    poly: &Polynomial<FieldElement<F>>,
    beta: &FieldElement<F>,
) -> Polynomial<FieldElement<F>>
where
    F: IsField,
{
    let coef = poly.coefficients();
    let n = coef.len();

    if n == 0 {
        return Polynomial::new(&[]);
    }

    let result_len = (n + 1) / 2;
    let mut result = Vec::with_capacity(result_len);

    // Process pairs: result[i] = coef[2i] + beta * coef[2i+1]
    let mut i = 0;
    while i + 1 < n {
        result.push(&coef[i] + &(&coef[i + 1] * beta));
        i += 2;
    }

    // Handle last coefficient if n is odd (no pair)
    if n % 2 == 1 {
        result.push(coef[n - 1].clone());
    }

    Polynomial::new(&result)
}

/// Optimized FRI fold with fused doubling: 2 * (P_even(x) + beta * P_odd(x))
/// Uses double() which is more efficient than multiplication by 2.
/// This is the pattern used in FRI commit phase.
pub fn fold_polynomial_doubled<F>(
    poly: &Polynomial<FieldElement<F>>,
    beta: &FieldElement<F>,
) -> Polynomial<FieldElement<F>>
where
    F: IsField,
{
    let coef = poly.coefficients();
    let n = coef.len();

    if n == 0 {
        return Polynomial::new(&[]);
    }

    let result_len = (n + 1) / 2;
    let mut result = Vec::with_capacity(result_len);

    // Process pairs: result[i] = 2 * (coef[2i] + beta * coef[2i+1])
    let mut i = 0;
    while i + 1 < n {
        let folded = &coef[i] + &(&coef[i + 1] * beta);
        result.push(folded.double());
        i += 2;
    }

    // Handle last coefficient if n is odd (no pair)
    if n % 2 == 1 {
        result.push(coef[n - 1].double());
    }

    Polynomial::new(&result)
}

/// Original implementation for benchmarking comparison
pub fn fold_polynomial_original<F>(
    poly: &Polynomial<FieldElement<F>>,
    beta: &FieldElement<F>,
) -> Polynomial<FieldElement<F>>
where
    F: IsField,
{
    use math::polynomial;

    let coef = poly.coefficients();
    let even_coef: Vec<FieldElement<F>> = coef.iter().step_by(2).cloned().collect();

    // odd coeficients of poly are multiplied by beta
    let odd_coef_mul_beta: Vec<FieldElement<F>> = coef
        .iter()
        .skip(1)
        .step_by(2)
        .map(|v| (v.clone()) * beta)
        .collect();

    let (even_poly, odd_poly) = polynomial::pad_with_zero_coefficients(
        &Polynomial::new(&even_coef),
        &Polynomial::new(&odd_coef_mul_beta),
    );
    even_poly + odd_poly
}

#[cfg(test)]
mod tests {
    use super::{fold_polynomial, fold_polynomial_original};
    use math::field::element::FieldElement;
    use math::field::fields::u64_prime_field::U64PrimeField;
    use math::polynomial::Polynomial;

    const MODULUS: u64 = 293;
    type FE = FieldElement<U64PrimeField<MODULUS>>;

    #[test]
    fn test_fold_power_of_2() {
        // FRI uses power-of-2 lengths: 8 -> 4 -> 2 -> 1
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
        // p1[i] = p0[2i] + beta * p0[2i+1]
        // p1[0] = 3 + 4*1 = 7
        // p1[1] = 2 + 4*7 = 30
        // p1[2] = 3 + 4*5 = 23
        // p1[3] = 4 + 4*2 = 12
        assert_eq!(
            p1,
            Polynomial::new(&[FE::new(7), FE::new(30), FE::new(23), FE::new(12)])
        );

        let gamma = FE::new(3);
        let p2 = fold_polynomial(&p1, &gamma);
        // p2[0] = 7 + 3*30 = 97
        // p2[1] = 23 + 3*12 = 59
        assert_eq!(p2, Polynomial::new(&[FE::new(97), FE::new(59)]));

        let delta = FE::new(2);
        let p3 = fold_polynomial(&p2, &delta);
        // p3[0] = 97 + 2*59 = 215
        assert_eq!(p3, Polynomial::new(&[FE::new(215)]));
        assert_eq!(p3.degree(), 0);
    }

    #[test]
    fn test_optimized_matches_original() {
        // Test with power-of-2 length (FRI-compatible)
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

        let original = fold_polynomial_original(&p0, &beta);
        let optimized = fold_polynomial(&p0, &beta);
        assert_eq!(original, optimized);

        // Test with size 4
        let p4 = Polynomial::new(&[FE::new(1), FE::new(2), FE::new(3), FE::new(4)]);
        let orig4 = fold_polynomial_original(&p4, &beta);
        let opt4 = fold_polynomial(&p4, &beta);
        assert_eq!(orig4, opt4);

        // Test with size 2
        let p2 = Polynomial::new(&[FE::new(10), FE::new(20)]);
        let orig2 = fold_polynomial_original(&p2, &beta);
        let opt2 = fold_polynomial(&p2, &beta);
        assert_eq!(orig2, opt2);
    }

    #[test]
    fn test_fold_size_2() {
        // Minimum valid FRI fold: 2 -> 1
        let p2 = Polynomial::new(&[FE::new(10), FE::new(20)]);
        let beta = FE::new(3);
        let result = fold_polynomial(&p2, &beta);
        // result[0] = 10 + 3*20 = 70
        assert_eq!(result, Polynomial::new(&[FE::new(70)]));
    }
}
