#[cfg(feature = "alloc")]
use crate::field::{element::FieldElement, traits::IsFFTField};

/// Pads the trace table with zeros until the length of the columns of the trace
/// is equal to a power of 2
/// This is required to ensure that we can use the radix-2 Cooley-Tukey FFT algorithm
#[cfg(feature = "alloc")]
pub fn resize_to_next_power_of_two<F: IsFFTField>(
    trace_colums: &mut [alloc::vec::Vec<FieldElement<F>>],
) {
    trace_colums.iter_mut().for_each(|col| {
        let next_power_of_two_len = col.len().next_power_of_two();
        col.resize(next_power_of_two_len, FieldElement::<F>::zero())
    })
}
