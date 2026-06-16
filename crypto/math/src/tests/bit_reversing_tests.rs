//! Tests for the bit-reverse permutation, relocated from `fft/bit_reversing.rs`.

use crate::fft::bit_reversing::{in_place_bit_reverse_permute, reverse_index};
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
