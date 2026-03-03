use super::curve::{BN254FieldElement, BN254TwistCurveFieldElement};
use crate::field::traits::LegendreSymbol;
use core::cmp::Ordering;

pub const TWO_INV: BN254FieldElement = BN254FieldElement::from_hex_unchecked(
    "183227397098D014DC2822DB40C0AC2ECBC0B548B438E5469E10460B6C3E7EA4",
);

#[must_use]
pub fn select_sqrt_value_from_third_bit(
    sqrt_1: BN254FieldElement,
    sqrt_2: BN254FieldElement,
    third_bit: u8,
) -> BN254FieldElement {
    match (sqrt_1.canonical().cmp(&sqrt_2.canonical()), third_bit) {
        (Ordering::Greater, 0) => sqrt_2,
        (Ordering::Greater, _) | (Ordering::Less, 0) | (Ordering::Equal, _) => sqrt_1,
        (Ordering::Less, _) => sqrt_2,
    }
}

/// * `third_bit` - if 1, then the square root is the greater one, otherwise it is the smaller one.
#[must_use]
pub fn sqrt_qfe(
    input: &BN254TwistCurveFieldElement,
    third_bit: u8,
) -> Option<BN254TwistCurveFieldElement> {
    // Algorithm 8, https://eprint.iacr.org/2012/685.pdf
    if *input == BN254TwistCurveFieldElement::zero() {
        Some(BN254TwistCurveFieldElement::zero())
    } else {
        let a = input.value()[0].clone();
        let b = input.value()[1].clone();
        if b == BN254FieldElement::zero() {
            // second part is zero
            let (y_sqrt_1, y_sqrt_2) = a.sqrt()?;
            let y_aux = select_sqrt_value_from_third_bit(y_sqrt_1, y_sqrt_2, third_bit);

            Some(BN254TwistCurveFieldElement::new([
                y_aux,
                BN254FieldElement::zero(),
            ]))
        } else {
            // second part of the input field number is non-zero
            // instead of "sum" is: -beta
            let alpha = a.square() + b.square();
            let gamma = alpha.legendre_symbol();
            match gamma {
                LegendreSymbol::One => {
                    let two = BN254FieldElement::from(2u64);
                    // calculate the square root of alpha
                    let (y_sqrt1, y_sqrt2) = alpha.sqrt()?;
                    let mut delta = (&a + y_sqrt1) * TWO_INV;

                    let legendre_delta = delta.legendre_symbol();
                    if legendre_delta == LegendreSymbol::MinusOne {
                        delta = (a + y_sqrt2) * TWO_INV;
                    };
                    let (x_sqrt_1, x_sqrt_2) = delta.sqrt()?;
                    let x_0 = select_sqrt_value_from_third_bit(x_sqrt_1, x_sqrt_2, third_bit);
                    let x_1 = b * (two * &x_0).inv().unwrap();
                    Some(BN254TwistCurveFieldElement::new([x_0, x_1]))
                }
                LegendreSymbol::MinusOne => None,
                LegendreSymbol::Zero => {
                    unreachable!("The input is zero, but we already handled this case.")
                }
            }
        }
    }
}
