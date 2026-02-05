use super::{field_extension::BN254PrimeField, twist::BN254TwistCurve};
use crate::{
    elliptic_curve::short_weierstrass::{
        curves::bn_254::{curve::BN254Curve, field_extension::Degree2ExtensionField, sqrt},
        point::ShortWeierstrassProjectivePoint,
        traits::{Compress, IsShortWeierstrass},
    },
    field::element::FieldElement,
};
use core::cmp::Ordering;

use crate::{
    cyclic_group::IsGroup, elliptic_curve::traits::FromAffine, errors::ByteConversionError,
    traits::ByteConversion,
};

type G1Point = ShortWeierstrassProjectivePoint<BN254Curve>;
type G2Point = ShortWeierstrassProjectivePoint<BN254TwistCurve>;
type BN254FieldElement = FieldElement<BN254PrimeField>;

/// As we have less than 3 bits available in our coordinate x, we can't follow BLS12-381 style encoding.
/// We use the 2 most significant bits instead
/// 00: uncompressed
/// 10: compressed and y_neg >= y
/// 11: compressed and y_neg < y
/// 01: compressed infinity point
/// the "uncompressed infinity point" will just have 00 (uncompressed) followed by zeroes (infinity = 0,0 in affine coordinates).
/// adapted from gnark https://github.com/consensys/gnark-crypto/blob/v0.13.0/ecc/bn254/marshal.go
impl Compress for BN254Curve {
    type G1Point = G1Point;

    type G2Point = G2Point;

    type G1Compressed = [u8; 32];

    type G2Compressed = [u8; 64];

    type Error = ByteConversionError;

    #[cfg(feature = "alloc")]
    fn compress_g1_point(point: &Self::G1Point) -> Self::G1Compressed {
        if *point == G1Point::neutral_element() {
            // Point is at infinity
            let mut x_bytes = [0_u8; 32];
            x_bytes[0] |= 1 << 6; // x_bytes = 01000000
            x_bytes
        } else {
            // Point is not at infinity
            let point_affine = point.to_affine();
            let x = point_affine.x();
            let y = point_affine.y();

            let mut x_bytes = [0u8; 32];
            let bytes = x.to_bytes_be();
            x_bytes.copy_from_slice(&bytes);
            // Set first bit to 1 to indicate this is a compressed element.
            x_bytes[0] |= 1 << 7; // x_bytes = 10000000

            let y_neg = core::ops::Neg::neg(y);
            if y_neg.canonical() < y.canonical() {
                x_bytes[0] |= 1 << 6; // x_bytes = 11000000
            }
            x_bytes
        }
    }

    fn decompress_g1_point(input_bytes: &mut [u8]) -> Result<Self::G1Point, Self::Error> {
        // We check that input_bytes has 32 bytes.
        if !input_bytes.len() == 32 {
            return Err(ByteConversionError::InvalidValue);
        }

        let first_byte = input_bytes.first().unwrap();
        // We get the 2 most significant bits
        let prefix_bits = first_byte >> 6;

        // If first two bits are 00, then the value is not compressed.
        if prefix_bits == 0_u8 {
            return Err(ByteConversionError::ValueNotCompressed);
        }

        // If first two bits are 01, then the compressed point is the
        // point at infinity and we return it directly.
        if prefix_bits == 1_u8 {
            return Ok(G1Point::neutral_element());
        }

        let first_byte_without_control_bits = (first_byte << 2) >> 2;
        input_bytes[0] = first_byte_without_control_bits;

        let x = BN254FieldElement::from_bytes_be(input_bytes)?;

        // We apply the elliptic curve formula to know the y^2 value.
        let y_squared = x.pow(3_u16) + BN254FieldElement::from(3);

        let (y_sqrt_1, y_sqrt_2) = &y_squared.sqrt().ok_or(ByteConversionError::InvalidValue)?;

        // If the frist two bits are 10, we take the smaller root.
        // If the first two bits are 11, we take the grater one.
        let y = match (y_sqrt_1.canonical().cmp(&y_sqrt_2.canonical()), prefix_bits) {
            (Ordering::Greater, 2_u8) => y_sqrt_2,
            (Ordering::Greater, _) => y_sqrt_1,
            (Ordering::Less, 2_u8) => y_sqrt_1,
            (Ordering::Less, _) => y_sqrt_2,
            (Ordering::Equal, _) => y_sqrt_1,
        };

        let point =
            G1Point::from_affine(x, y.clone()).map_err(|_| ByteConversionError::InvalidValue)?;

        Ok(point)
    }

    #[cfg(feature = "alloc")]
    fn compress_g2_point(point: &Self::G2Point) -> Self::G2Compressed {
        if *point == G2Point::neutral_element() {
            // Point is at infinity
            let mut x_bytes = [0_u8; 64];
            x_bytes[0] |= 1 << 6; // x_bytes = 01000000
            x_bytes
        } else {
            // Point is not at infinity
            let point_affine = point.to_affine();
            let x = point_affine.x();
            let y = point_affine.y();

            let mut x_bytes = [0u8; 64];
            let bytes = x.to_bytes_be();
            x_bytes.copy_from_slice(&bytes);

            // Set first bit to to 1 indicate this is compressed element.
            x_bytes[0] |= 1 << 7;

            // We see if y_neg < y lexicographically where the lexicographic order is as follows:
            // Let a = a0 + a1 * u and b = b0 + b1 * u in Fp2, then a < b if a0 < b0 or
            // a0 = b0 and a1 < b1.
            let y_neg = -y;
            match (
                y.value()[0].canonical().cmp(&y_neg.value()[0].canonical()),
                y.value()[1].canonical().cmp(&y_neg.value()[1].canonical()),
            ) {
                (Ordering::Greater, _) | (Ordering::Equal, Ordering::Greater) => {
                    x_bytes[0] |= 1 << 6; // Prefix: 11
                }
                (_, _) => (),
            }
            x_bytes
        }
    }

    #[allow(unused)]
    fn decompress_g2_point(input_bytes: &mut [u8]) -> Result<Self::G2Point, Self::Error> {
        if !input_bytes.len() == 64 {
            return Err(ByteConversionError::InvalidValue);
        }

        let first_byte = input_bytes.first().unwrap();

        // We get the first 2 bits.
        let prefix_bits = first_byte >> 6;

        // If first two bits are 00, then the value is not compressed.
        if prefix_bits == 0_u8 {
            return Err(ByteConversionError::InvalidValue);
        }

        // If the first two bits are 01, then the compressed point is the
        // point at infinity and we return it directly.
        if prefix_bits == 1_u8 {
            return Ok(Self::G2Point::neutral_element());
        }

        let second_bit = prefix_bits & 1_u8;
        let first_byte_without_control_bits = (first_byte << 2) >> 2;
        input_bytes[0] = first_byte_without_control_bits;

        let input1 = &input_bytes[0..32];
        let input0 = &input_bytes[32..];
        let x0 = BN254FieldElement::from_bytes_be(input0).unwrap();
        let x1 = BN254FieldElement::from_bytes_be(input1).unwrap();
        let x: FieldElement<Degree2ExtensionField> = FieldElement::new([x0, x1]);

        let b_param_qfe = BN254TwistCurve::b();

        // If the first two bits are 11, then the square root chosen is the greater one.
        // So we should use sqrt_qfe with the input 1.
        let y = sqrt::sqrt_qfe(&(x.pow(3_u64) + b_param_qfe), second_bit)
            .ok_or(ByteConversionError::InvalidValue)?;

        Self::G2Point::from_affine(x, y).map_err(|_| ByteConversionError::InvalidValue)
    }
}
