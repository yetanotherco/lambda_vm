use crate::{
    fft::roots_of_unity::get_powers_of_primitive_root,
    field::{
        element::FieldElement,
        traits::{IsFFTField, RootsConfig},
    },
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
