use crate::fri::fri_functions::{compute_coset_twiddles_inv, fold_evaluations_in_place};
use math::fft::bit_reversing::in_place_bit_reverse_permute;
use math::field::element::FieldElement;
use math::field::goldilocks::GoldilocksField;
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

/// Reference coefficient-form FRI fold with doubling: 2 * (P_even(x) + beta * P_odd(x))
fn fold_polynomial_doubled_reference<F: IsField>(
    poly: &Polynomial<FieldElement<F>>,
    beta: &FieldElement<F>,
) -> Polynomial<FieldElement<F>> {
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
fn test_eval_fold_matches_coeff_fold() {
    let coset_offset = FE::from(3u64);
    let beta = FE::from(7u64);

    // Use a degree-7 polynomial (8 coefficients)
    let poly = Polynomial::new(&[
        FE::from(1u64),
        FE::from(2u64),
        FE::from(3u64),
        FE::from(4u64),
        FE::from(5u64),
        FE::from(6u64),
        FE::from(7u64),
        FE::from(8u64),
    ]);
    let n = 8usize;

    // Evaluate polynomial on coset via FFT
    let evals_fft =
        Polynomial::evaluate_offset_fft::<GoldilocksField>(&poly, 1, None, &coset_offset).unwrap();

    // Path A: reference coeff fold -> FFT -> bit-reverse
    let folded_poly = fold_polynomial_doubled_reference(&poly, &beta);
    let squared_offset = coset_offset.square();
    let mut path_a_evals =
        Polynomial::evaluate_offset_fft::<GoldilocksField>(&folded_poly, 1, None, &squared_offset)
            .unwrap();
    in_place_bit_reverse_permute(&mut path_a_evals);

    // Path B: FFT -> bit-reverse -> eval fold (live fold_evaluations_in_place)
    let mut path_b_evals = evals_fft;
    in_place_bit_reverse_permute(&mut path_b_evals);
    let inv_twiddles = compute_coset_twiddles_inv::<GoldilocksField>(&coset_offset, n);
    fold_evaluations_in_place(&mut path_b_evals, &beta, &inv_twiddles);

    assert_eq!(path_a_evals, path_b_evals);
}
