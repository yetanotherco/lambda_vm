use crate::{
    fft::cpu::roots_of_unity::{get_powers_of_primitive_root, get_powers_of_primitive_root_coset},
    field::{
        element::FieldElement,
        traits::{IsFFTField, RootsConfig},
    },
    polynomial::Polynomial,
};
use alloc::vec::Vec;

/// Calculates the (non-unitary) Discrete Fourier Transform of `input` via the DFT matrix.
pub fn naive_matrix_dft_test<F: IsFFTField>(input: &[FieldElement<F>]) -> Vec<FieldElement<F>> {
    let n = input.len();
    assert!(n.is_power_of_two());
    let order = n.trailing_zeros();

    let twiddles =
        get_powers_of_primitive_root::<F>(order.into(), n, RootsConfig::Natural).unwrap();

    let mut output = Vec::with_capacity(n);
    for row in 0..n {
        let mut sum = FieldElement::zero();

        for (col, element) in input.iter().enumerate() {
            let i = (row * col) % n; // w^i = w^(i mod n)
            sum += element.clone() * twiddles[i].clone();
        }

        output.push(sum);
    }

    output
}

/// Compares FFT evaluation against naive polynomial evaluation.
/// Returns (fft_result, naive_result) for comparison.
pub fn gen_fft_and_naive_evaluation<F: IsFFTField>(
    poly: Polynomial<FieldElement<F>>,
) -> (Vec<FieldElement<F>>, Vec<FieldElement<F>>) {
    let len = poly.coeff_len().next_power_of_two();
    let order = len.trailing_zeros();
    let twiddles = get_powers_of_primitive_root(order.into(), len, RootsConfig::Natural).unwrap();

    let fft_eval = Polynomial::evaluate_fft::<F>(&poly, 1, None).unwrap();
    let naive_eval = poly.evaluate_slice(&twiddles);

    (fft_eval, naive_eval)
}

/// Compares FFT coset evaluation against naive polynomial evaluation.
/// Returns (fft_result, naive_result) for comparison.
pub fn gen_fft_coset_and_naive_evaluation<F: IsFFTField>(
    poly: Polynomial<FieldElement<F>>,
    offset: FieldElement<F>,
    blowup_factor: usize,
) -> (Vec<FieldElement<F>>, Vec<FieldElement<F>>) {
    let len = poly.coeff_len().next_power_of_two();
    let order = (len * blowup_factor).trailing_zeros();
    let twiddles =
        get_powers_of_primitive_root_coset(order.into(), len * blowup_factor, &offset).unwrap();

    let fft_eval =
        Polynomial::evaluate_offset_fft::<F>(&poly, blowup_factor, None, &offset).unwrap();
    let naive_eval = poly.evaluate_slice(&twiddles);

    (fft_eval, naive_eval)
}

/// Compares FFT interpolation against naive interpolation.
/// Returns (fft_result, naive_result) for comparison.
pub fn gen_fft_and_naive_interpolate<F: IsFFTField>(
    fft_evals: &[FieldElement<F>],
) -> (Polynomial<FieldElement<F>>, Polynomial<FieldElement<F>>) {
    let order = fft_evals.len().trailing_zeros() as u64;
    let twiddles = get_powers_of_primitive_root(order, 1 << order, RootsConfig::Natural).unwrap();

    let naive_poly = Polynomial::interpolate(&twiddles, fft_evals).unwrap();
    let fft_poly = Polynomial::interpolate_fft::<F>(fft_evals).unwrap();

    (fft_poly, naive_poly)
}

/// Compares FFT coset interpolation against naive interpolation.
/// Returns (fft_result, naive_result) for comparison.
pub fn gen_fft_and_naive_coset_interpolate<F: IsFFTField>(
    fft_evals: &[FieldElement<F>],
    offset: &FieldElement<F>,
) -> (Polynomial<FieldElement<F>>, Polynomial<FieldElement<F>>) {
    let order = fft_evals.len().trailing_zeros() as u64;
    let twiddles = get_powers_of_primitive_root_coset(order, 1 << order, offset).unwrap();

    let naive_poly = Polynomial::interpolate(&twiddles, fft_evals).unwrap();
    let fft_poly = Polynomial::interpolate_offset_fft(fft_evals, offset).unwrap();

    (fft_poly, naive_poly)
}

/// Verifies that FFT interpolation is the inverse of evaluation.
/// Returns (original_poly, recovered_poly) for comparison.
pub fn gen_fft_interpolate_and_evaluate<F: IsFFTField>(
    poly: Polynomial<FieldElement<F>>,
) -> (Polynomial<FieldElement<F>>, Polynomial<FieldElement<F>>) {
    let eval = Polynomial::evaluate_fft::<F>(&poly, 1, None).unwrap();
    let new_poly = Polynomial::interpolate_fft::<F>(&eval).unwrap();

    (poly, new_poly)
}
