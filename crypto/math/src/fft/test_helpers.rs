use crate::{
    fft::{bit_reversing::in_place_bit_reverse_permute, errors::FFTError},
    field::{
        element::FieldElement,
        traits::{IsFFTField, RootsConfig},
    },
};
use alloc::vec::Vec;

/// Returns a `Vec` of the powers of a `2^n`th primitive root of unity in some configuration
/// `config`. For example, in a `Natural` config this would yield: w^0, w^1, w^2...
///
/// Test-only: production twiddle generation goes through `bowers_fft::LayerTwiddles`.
pub fn get_powers_of_primitive_root<F: IsFFTField>(
    n: u64,
    count: usize,
    config: RootsConfig,
) -> Result<Vec<FieldElement<F>>, FFTError> {
    if count == 0 {
        return Ok(Vec::new());
    }

    let root = match config {
        RootsConfig::Natural | RootsConfig::BitReverse => F::get_primitive_root_of_unity(n)?,
        _ => F::get_primitive_root_of_unity(n)?.inv().unwrap(),
    };
    let up_to = match config {
        RootsConfig::Natural | RootsConfig::NaturalInversed => count,
        // In bit reverse form we could need as many as `(1 << count.bits()) - 1` roots
        _ => count.next_power_of_two(),
    };

    let mut results = Vec::with_capacity(up_to);
    // NOTE: a nice version would be using `core::iter::successors`. However, this is 10% faster.
    results.extend((0..up_to).scan(FieldElement::one(), |state, _| {
        let res = state.clone();
        *state = &(*state) * &root;
        Some(res)
    }));

    if matches!(
        config,
        RootsConfig::BitReverse | RootsConfig::BitReverseInversed
    ) {
        in_place_bit_reverse_permute(&mut results);
    }

    Ok(results)
}

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
