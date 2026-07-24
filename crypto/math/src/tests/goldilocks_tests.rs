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

#[test]
fn eq_matches_field_value_across_representations() {
    type Fe = FieldElement<GoldilocksField>;
    // The raw-compare short-circuit in `eq` must still equate the two lazy
    // representations `x` and `x + p` of the same element, and keep distinct
    // elements distinct — identical to the pure `canonical(a) == canonical(b)`
    // compare. `x + p` only fits in u64 for `x < 2^64 - p = 2^32 - 1`, so the
    // second representation is exercised with small `x`.
    for x in [0u64, 1, 5, 12345, 0xFFFF_FFFEu64] {
        let canonical = Fe::from_raw(x);
        let lazy = Fe::from_raw(x + GOLDILOCKS_PRIME); // same element, non-canonical rep
        assert_eq!(canonical, lazy, "x={x}: lazy rep must equal canonical rep");
        assert_eq!(canonical, canonical);
        let other = Fe::from_raw(x + 1); // distinct element (x+1 < p here)
        assert_ne!(canonical, other, "x={x}: distinct elements must differ");
        // A distinct element compared against a lazy rep must also differ.
        assert_ne!(lazy, other, "x={x}: distinct vs lazy must differ");
    }
    // Also directly: the largest canonical value and its (single) representation.
    let top = Fe::from_raw(GOLDILOCKS_PRIME - 1);
    assert_eq!(top, top);
    assert_ne!(top, Fe::from_raw(0));
}
