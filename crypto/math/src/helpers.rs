#[cfg(feature = "alloc")]
use crate::field::{element::FieldElement, traits::IsFFTField};

/// Computes the power of two that is equal or greater than n
pub fn next_power_of_two(n: u64) -> u64 {
    if n <= 1 {
        1
    } else {
        (u64::MAX >> (n - 1).leading_zeros()) + 1
    }
}

/// Pads the trace table with zeros until the length of the columns of the trace
/// is equal to a power of 2.
/// This is required to ensure that we can use the radix-2 Cooley-Tukey FFT algorithm.
#[cfg(feature = "alloc")]
pub fn resize_to_next_power_of_two<F: IsFFTField>(
    trace_colums: &mut [alloc::vec::Vec<FieldElement<F>>],
) {
    trace_colums.iter_mut().for_each(|col| {
        // Safe: usize always fits in u64 (usize is at most 64 bits)
        let col_len = col.len() as u64;
        let next_power_of_two_len = next_power_of_two(col_len);
        // On 32-bit platforms, this saturates if the result exceeds usize::MAX,
        // but such large traces wouldn't fit in memory anyway.
        let target_len = usize::try_from(next_power_of_two_len).unwrap_or(usize::MAX);
        col.resize(target_len, FieldElement::<F>::zero())
    })
}
