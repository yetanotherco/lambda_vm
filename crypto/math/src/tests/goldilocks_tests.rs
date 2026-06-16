//! Tests for `field/goldilocks.rs`, relocated from its inline test module.

use crate::field::element::FieldElement;
use crate::field::goldilocks::{GOLDILOCKS_PRIME, GoldilocksField};
use crate::traits::{AsBytes, ByteConversion};

#[test]
fn write_bytes_be_matches_as_bytes() {
    let cases = [
        FieldElement::<GoldilocksField>::from(0u64),
        FieldElement::<GoldilocksField>::from(1u64),
        FieldElement::<GoldilocksField>::from(GOLDILOCKS_PRIME - 1),
    ];
    for elem in &cases {
        let mut buf = [0u8; 8];
        elem.write_bytes_be(&mut buf);
        assert_eq!(&buf[..], elem.as_bytes().as_slice());
    }
}

#[test]
fn write_bytes_be_matches_as_bytes_noncanonical() {
    // Values stored as-is via from_raw are non-canonical (>= p) until serialized.
    let elem = FieldElement::<GoldilocksField>::from_raw(GOLDILOCKS_PRIME + 5);
    let mut buf = [0u8; 8];
    elem.write_bytes_be(&mut buf);
    assert_eq!(&buf[..], elem.as_bytes().as_slice());
}
