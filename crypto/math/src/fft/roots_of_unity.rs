use crate::field::{element::FieldElement, traits::IsFFTField};
use alloc::vec::Vec;

use crate::fft::errors::FFTError;

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
