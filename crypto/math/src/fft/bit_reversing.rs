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
