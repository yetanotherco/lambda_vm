use crate::field::{element::FieldElement, traits::IsFFTField};
use alloc::vec::Vec;

use crate::fft::errors::FFTError;

// `RootsConfig` and the bit-reverse permutation are only used by the test-only
// `get_powers_of_primitive_root` below.
#[cfg(test)]
use super::bit_reversing::in_place_bit_reverse_permute;
#[cfg(test)]
use crate::field::traits::RootsConfig;

/// Returns a `Vec` of the powers of a `2^n`th primitive root of unity in some configuration
/// `config`. For example, in a `Natural` config this would yield: w^0, w^1, w^2...
///
/// Test-only: production twiddle generation goes through `bowers_fft::LayerTwiddles`.
#[cfg(test)]
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

/// Returns a `Vec` of the powers of a `2^n`th primitive root of unity, scaled `offset` times,
/// in a Natural configuration.
pub fn get_powers_of_primitive_root_coset<F: IsFFTField>(
    n: u64,
    count: usize,
    offset: &FieldElement<F>,
) -> Result<Vec<FieldElement<F>>, FFTError> {
    if count == 0 {
        return Ok(Vec::new());
    }

    let root = F::get_primitive_root_of_unity(n)?;
    let mut results = Vec::with_capacity(count);
    let mut current = offset.clone();
    for _ in 0..count {
        results.push(current.clone());
        current = &current * &root;
    }

    Ok(results)
}
