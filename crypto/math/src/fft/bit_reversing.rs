#[cfg(all(feature = "alloc", feature = "parallel"))]
use rayon::prelude::*;

/// In-place bit-reverse permutation algorithm. Requires input length to be a power of two.
pub fn in_place_bit_reverse_permute<E>(input: &mut [E]) {
    for i in 0..input.len() {
        let bit_reversed_index = reverse_index(i, input.len() as u64);
        if bit_reversed_index > i {
            input.swap(i, bit_reversed_index);
        }
    }
}

/// Reverses the `log2(size)` first bits of `i`
pub fn reverse_index(i: usize, size: u64) -> usize {
    if size == 1 {
        i
    } else {
        i.reverse_bits() >> (usize::BITS - size.trailing_zeros())
    }
}

/// Row-major variant of [`in_place_bit_reverse_permute`]: permute a flat
/// `n * num_cols` row-major buffer by bit-reversing the row index, swapping
/// whole rows (`num_cols` consecutive elements) at a time.
///
/// `buf.len()` must equal `n * num_cols` for some power-of-two `n`. Row `i` is
/// swapped with row `reverse_index(i, n)` when that index is greater (so each
/// pair is swapped exactly once). Used by the batched row-major FFT/LDE.
///
/// Parallel path: bit-reverse is a permutation, so the `(i, br(i))` pairs with
/// `br(i) > i` are pairwise disjoint; each swap touches two distinct,
/// non-overlapping row slices, so they can be dispatched via raw-pointer
/// indexing without a synchronization barrier.
#[cfg(feature = "alloc")]
pub fn in_place_bit_reverse_permute_row_major<E: Send + Sync>(buf: &mut [E], num_cols: usize) {
    if num_cols == 0 || buf.is_empty() {
        return;
    }
    debug_assert!(
        buf.len().is_multiple_of(num_cols),
        "buf.len() must be a multiple of num_cols"
    );
    let n = buf.len() / num_cols;
    if n <= 1 {
        return;
    }
    debug_assert!(n.is_power_of_two(), "row count must be a power of two");

    #[cfg(feature = "parallel")]
    {
        // No upfront Vec<(usize, usize)> collection (saves ~16 MB at log21 n=64).
        if n >= 2048 {
            use core::sync::atomic::{AtomicPtr, Ordering};
            let raw = AtomicPtr::new(buf.as_mut_ptr());
            (0..n).into_par_iter().for_each(|i| {
                let j = reverse_index(i, n as u64);
                if j > i {
                    let ptr = raw.load(Ordering::Relaxed);
                    let lo = i * num_cols;
                    let hi = j * num_cols;
                    // SAFETY: (lo..lo+M) and (hi..hi+M) point into the same
                    // buffer but are disjoint (lo != hi); the par_iter visits
                    // each unordered pair exactly once (we filter on j > i),
                    // so no two threads touch overlapping ranges.
                    unsafe {
                        let lo_row = core::slice::from_raw_parts_mut(ptr.add(lo), num_cols);
                        let hi_row = core::slice::from_raw_parts_mut(ptr.add(hi), num_cols);
                        lo_row.swap_with_slice(hi_row);
                    }
                }
            });
            return;
        }
    }

    for i in 0..n {
        let j = reverse_index(i, n as u64);
        if j > i {
            let lo = i * num_cols;
            let hi = j * num_cols;
            let (left, right) = buf.split_at_mut(hi);
            left[lo..lo + num_cols].swap_with_slice(&mut right[..num_cols]);
        }
    }
}
