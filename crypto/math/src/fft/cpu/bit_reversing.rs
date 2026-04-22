/// In-place bit-reverse permutation algorithm. Requires input length to be a power of two.
pub fn in_place_bit_reverse_permute<E: Send>(input: &mut [E]) {
    let n = input.len();
    #[cfg(feature = "parallel")]
    {
        // Pair-parallel swap: each pair (i, br(i)) with i < br(i) is independent of all
        // other pairs (disjoint indices), so threads can swap concurrently provided they
        // never touch the same memory location. `if br > i` selects exactly one owner
        // per pair, so no two threads ever write the same slot.
        const PARALLEL_BITREV_THRESHOLD: usize = 1 << 14;
        if n >= PARALLEL_BITREV_THRESHOLD {
            use rayon::prelude::*;
            struct SendPtr<E>(*mut E);
            impl<E> Copy for SendPtr<E> {}
            impl<E> Clone for SendPtr<E> {
                fn clone(&self) -> Self {
                    *self
                }
            }
            unsafe impl<E> Send for SendPtr<E> {}
            unsafe impl<E> Sync for SendPtr<E> {}
            let ptr = SendPtr(input.as_mut_ptr());
            (0..n).into_par_iter().for_each(|i| {
                let br = reverse_index(i, n as u64);
                if br > i {
                    // SAFETY: (i, br) uniquely identifies this pair (smaller index is owner),
                    // so no two threads race on the same `ptr.0.add(k)` slot. Both indices
                    // are in-bounds since i < n and br < n.
                    let p = ptr;
                    unsafe {
                        core::ptr::swap(p.0.add(i), p.0.add(br));
                    }
                }
            });
            return;
        }
    }
    for i in 0..n {
        let bit_reversed_index = reverse_index(i, n as u64);
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

#[cfg(all(test, feature = "alloc"))]
mod test {
    use super::*;
    use alloc::vec::Vec;

    // TODO: proptest would be better.
    #[test]
    fn bit_reverse_permutation_works() {
        let mut reversed: Vec<usize> = Vec::with_capacity(16);
        for i in 0..reversed.capacity() {
            reversed.push(reverse_index(i, reversed.capacity() as u64));
        }
        assert_eq!(
            reversed[..],
            [0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15]
        );

        in_place_bit_reverse_permute(&mut reversed[..]);
        assert_eq!(
            reversed[..],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        );
    }

    #[test]
    fn bit_reverse_permutation_edge_case() {
        let mut edge_case = [0];

        in_place_bit_reverse_permute(&mut edge_case[..]);
        assert_eq!(edge_case[..], [0]);
    }
}
