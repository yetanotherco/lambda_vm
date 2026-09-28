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
/// Parallel path: over a power-of-two row count, bit-reverse is an *involution*
/// (`br(br(i)) == i`), so every non-trivial orbit is a 2-cycle `{i, br(i)}`.
/// Filtering on `br(i) > i` selects one representative per orbit, so the swapped
/// pairs are pairwise disjoint; each swap touches two distinct, non-overlapping
/// row slices, so they can be dispatched via raw-pointer indexing without a
/// synchronization barrier.
///
/// The power-of-two row count is the precondition that makes bit-reverse an
/// involution, so it is enforced with a runtime `assert!` (not just a
/// `debug_assert!`): a non-power-of-two `n` would break the disjointness the
/// parallel path relies on, turning a bad caller's input into a data race.
#[cfg(feature = "alloc")]
pub(crate) fn in_place_bit_reverse_permute_row_major<E: Send + Sync>(
    buf: &mut [E],
    num_cols: usize,
) {
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
    // Safety-critical, not just correctness: the parallel raw-pointer path below
    // relies on bit-reverse being an involution, which holds only when `n` is a
    // power of two. Enforce at runtime so a bad caller panics here rather than
    // triggering a data race in the unsafe block.
    assert!(n.is_power_of_two(), "row count must be a power of two");

    #[cfg(feature = "parallel")]
    {
        // No upfront Vec<(usize, usize)> collection (saves ~32 MB at log_n=21 on 64-bit).
        if n >= 2048 {
            use core::sync::atomic::{AtomicPtr, Ordering};
            let raw = AtomicPtr::new(buf.as_mut_ptr());
            (0..n).into_par_iter().for_each(|i| {
                let j = reverse_index(i, n as u64);
                if j > i {
                    let ptr = raw.load(Ordering::Relaxed);
                    let lo = i * num_cols;
                    let hi = j * num_cols;
                    // SAFETY: (lo..lo+M) and (hi..hi+M) are disjoint, so no two
                    // threads ever touch overlapping ranges:
                    //   1. `n` is a power of two (asserted above), so bit-reverse
                    //      is an involution (`br(br(i)) == i`); every non-trivial
                    //      orbit is a 2-cycle `{i, br(i)}`. The `j > i` filter
                    //      keeps one representative per orbit, so the chosen pairs
                    //      are pairwise disjoint and `lo != hi`. (`j = br(i) < n`,
                    //      so both rows are in bounds.)
                    //   2. Rows are `num_cols` wide and don't overlap, so the two
                    //      M-element ranges are disjoint.
                    //   3. `Ordering::Relaxed` on the load is sound: the pointer is
                    //      written before `into_par_iter()` starts, and Rayon's
                    //      thread spawn provides the happens-before edge that makes
                    //      every worker observe the initial value.
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

/// Out-of-place row-major bit-reverse: appends the rows of `src` (`n` rows of
/// `num_cols` elements, `n` a power of two) to the EMPTY `dst` in bit-reversed
/// order — `dst` row `i` = `src` row `reverse_index(i, n)`. One read and one
/// write per element, where copying `src` in and then running
/// [`in_place_bit_reverse_permute_row_major`] costs a full extra pass. `dst`
/// keeps (and grows to at least) the capacity it was given.
#[cfg(feature = "alloc")]
pub(crate) fn bit_reverse_rows_into<E: Clone + Send + Sync>(
    src: &[E],
    dst: &mut alloc::vec::Vec<E>,
    num_cols: usize,
) {
    assert!(dst.is_empty(), "dst must start empty");
    if num_cols == 0 || src.is_empty() {
        return;
    }
    assert!(
        src.len().is_multiple_of(num_cols),
        "src.len() must be a multiple of num_cols"
    );
    let n = src.len() / num_cols;
    assert!(n.is_power_of_two(), "row count must be a power of two");
    dst.reserve(src.len());
    let rows = &mut dst.spare_capacity_mut()[..src.len()];
    let copy_row = |(i, row): (usize, &mut [core::mem::MaybeUninit<E>])| {
        let j = reverse_index(i, n as u64) * num_cols;
        for (d, s) in row.iter_mut().zip(&src[j..j + num_cols]) {
            d.write(s.clone());
        }
    };
    #[cfg(feature = "parallel")]
    if n >= 2048 {
        rows.par_chunks_exact_mut(num_cols)
            .enumerate()
            .for_each(copy_row);
    } else {
        rows.chunks_exact_mut(num_cols)
            .enumerate()
            .for_each(copy_row);
    }
    #[cfg(not(feature = "parallel"))]
    rows.chunks_exact_mut(num_cols)
        .enumerate()
        .for_each(copy_row);
    // SAFETY: every one of the first `src.len()` spare slots was written above.
    unsafe { dst.set_len(src.len()) };
}
