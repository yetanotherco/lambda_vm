use crate::fri::fri_functions::{
    compute_coset_twiddles_inv, fold_evaluations_in_place, fold_polynomial_doubled_inplace,
};
use math::fft::cpu::bit_reversing::in_place_bit_reverse_permute;
use math::field::element::FieldElement;
use math::field::fields::fft_friendly::u64_goldilocks::GoldilocksField;
use math::field::traits::IsField;
use math::polynomial::Polynomial;

type FE = FieldElement<GoldilocksField>;

/// FRI polynomial folding: computes P_even(x) + beta * P_odd(x)
/// where P(x) = P_even(x^2) + x * P_odd(x^2)
fn fold_polynomial<F>(
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
fn fold_polynomial_doubled<F>(
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

/// Verifies that evaluation-form folding produces the same result as
/// coefficient-form fold + FFT. This is the key correctness invariant:
/// both paths must produce identical FRI layer evaluations.
#[test]
fn test_eval_fold_matches_coeff_fold() {
    type Gfe = FieldElement<GoldilocksField>;

    // A degree-7 polynomial (8 coefficients)
    let poly = Polynomial::new(&[
        Gfe::from(3u64),
        Gfe::from(1u64),
        Gfe::from(2u64),
        Gfe::from(7u64),
        Gfe::from(3u64),
        Gfe::from(5u64),
        Gfe::from(4u64),
        Gfe::from(2u64),
    ]);
    let domain_size = 8usize;
    let coset_offset = Gfe::from(7u64); // arbitrary nonzero offset
    let zeta = Gfe::from(13u64);

    // Path A: coefficient fold + FFT + bit-reverse (old approach)
    let mut poly_coeff = poly.clone();
    fold_polynomial_doubled_inplace(&mut poly_coeff, &zeta);
    let coset_offset_squared = coset_offset.square();
    let mut coeff_evals = Polynomial::evaluate_offset_fft(
        &poly_coeff,
        1,
        Some(domain_size / 2),
        &coset_offset_squared,
    )
    .unwrap();
    in_place_bit_reverse_permute(&mut coeff_evals);

    // Path B: FFT + bit-reverse + eval fold (new approach)
    let mut eval_evals =
        Polynomial::evaluate_offset_fft(&poly, 1, Some(domain_size), &coset_offset).unwrap();
    in_place_bit_reverse_permute(&mut eval_evals);
    let inv_twiddles = compute_coset_twiddles_inv(&coset_offset, domain_size);
    fold_evaluations_in_place(&mut eval_evals, &zeta, &inv_twiddles);

    // Both paths must produce identical results
    assert_eq!(coeff_evals.len(), eval_evals.len());
    for (i, (a, b)) in coeff_evals.iter().zip(eval_evals.iter()).enumerate() {
        assert_eq!(a, b, "Mismatch at index {i}: coeff={a:?}, eval={b:?}");
    }
}
