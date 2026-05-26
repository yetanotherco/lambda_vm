//! Tests for `field/extensions_goldilocks.rs`, relocated from its inline test module.

use crate::field::element::FieldElement;
use crate::field::extensions_goldilocks::{Degree3GoldilocksExtensionField, FpE};
use crate::traits::{AsBytes, ByteConversion};

#[test]
fn write_bytes_be_matches_as_bytes() {
    let cases = [
        FieldElement::<Degree3GoldilocksExtensionField>::zero(),
        FieldElement::<Degree3GoldilocksExtensionField>::one(),
        FieldElement::<Degree3GoldilocksExtensionField>::new([
            FpE::from(1u64),
            FpE::from(2u64),
            FpE::from(3u64),
        ]),
    ];
    for elem in &cases {
        let mut buf = [0u8; 24];
        elem.write_bytes_be(&mut buf);
        assert_eq!(&buf[..], elem.as_bytes().as_slice());
    }
}
