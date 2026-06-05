//! Tests for constraint templates.

use crate::constraints::templates::*;
use crate::tables::types::GoldilocksField;
use math::field::element::FieldElement;

#[test]
fn test_inv_shift_32_is_correct() {
    let inv = FieldElement::<GoldilocksField>::from(INV_SHIFT_32);
    let shift = FieldElement::<GoldilocksField>::from(SHIFT_32);
    assert_eq!(inv * shift, FieldElement::one());
}
