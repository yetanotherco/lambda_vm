use crate::field::{
    element::FieldElement,
    extensions::{
        cubic::{CubicExtensionField, HasCubicNonResidue},
        quadratic::{HasQuadraticNonResidue, QuadraticExtensionField},
    },
    fields::fft_friendly::u64_goldilocks::U64GoldilocksPrimeField,
};

// =====================================================
// QUADRATIC EXTENSION (Fp2)
// =====================================================
// The quadratic extension is constructed using x^2 - 7,
// where 7 is a quadratic non-residue in the Goldilocks field.
// This means Fp2 = Fp[x] / (x^2 - 7)

/// Quadratic field extension of Goldilocks (degree 2)
/// Elements are represented as a + b*w where w^2 = 7
pub type Degree2GoldilocksExtensionField =
    QuadraticExtensionField<U64GoldilocksPrimeField, U64GoldilocksPrimeField>;

impl HasQuadraticNonResidue<U64GoldilocksPrimeField> for U64GoldilocksPrimeField {
    /// Returns the quadratic non-residue used for the extension.
    /// We use 7, which is verified to be a quadratic non-residue in the Goldilocks field.
    fn residue() -> FieldElement<U64GoldilocksPrimeField> {
        FieldElement::from(7u64)
    }
}

/// Field element type for the quadratic extension of Goldilocks
pub type Degree2GoldilocksExtensionFieldElement =
    FieldElement<Degree2GoldilocksExtensionField>;

// =====================================================
// CUBIC EXTENSION (Fp3)
// =====================================================
// The cubic extension is constructed using x^3 - 2,
// where 2 is a cubic non-residue in the Goldilocks field.
// This means Fp3 = Fp[x] / (x^3 - 2)

#[derive(Debug, Clone)]
pub struct GoldilocksCubicNonResidue;

impl HasCubicNonResidue<U64GoldilocksPrimeField> for GoldilocksCubicNonResidue {
    /// Returns the cubic non-residue used for the extension.
    /// We use 2, which is a cubic non-residue in the Goldilocks field.
    fn residue() -> FieldElement<U64GoldilocksPrimeField> {
        FieldElement::from(2u64)
    }
}

/// Cubic field extension of Goldilocks (degree 3)
/// Elements are represented as a + b*w + c*w^2 where w^3 = 2
pub type Degree3GoldilocksExtensionField =
    CubicExtensionField<U64GoldilocksPrimeField, GoldilocksCubicNonResidue>;

/// Field element type for the cubic extension of Goldilocks
pub type Degree3GoldilocksExtensionFieldElement =
    FieldElement<Degree3GoldilocksExtensionField>;

#[cfg(test)]
mod tests {
    use super::*;

    type FpE = FieldElement<U64GoldilocksPrimeField>;
    type Fp2E = Degree2GoldilocksExtensionFieldElement;
    type Fp3E = Degree3GoldilocksExtensionFieldElement;

    // =====================================================
    // QUADRATIC EXTENSION TESTS
    // =====================================================

    #[test]
    fn test_fp2_add() {
        let a = Fp2E::new([FpE::from(0u64), FpE::from(3u64)]);
        let b = Fp2E::new([FpE::from(2u64), FpE::from(8u64)]);
        let expected = Fp2E::new([FpE::from(2u64), FpE::from(11u64)]);
        assert_eq!(a + b, expected);
    }

    #[test]
    fn test_fp2_sub() {
        let a = Fp2E::new([FpE::from(10u64), FpE::from(5u64)]);
        let b = Fp2E::new([FpE::from(3u64), FpE::from(2u64)]);
        let expected = Fp2E::new([FpE::from(7u64), FpE::from(3u64)]);
        assert_eq!(a - b, expected);
    }

    #[test]
    fn test_fp2_mul() {
        // (a0 + a1*w) * (b0 + b1*w) = (a0*b0 + a1*b1*7) + (a0*b1 + a1*b0)*w
        let a = Fp2E::new([FpE::from(2u64), FpE::from(3u64)]);
        let b = Fp2E::new([FpE::from(4u64), FpE::from(5u64)]);
        // c0 = 2*4 + 3*5*7 = 8 + 105 = 113
        // c1 = 2*5 + 3*4 = 10 + 12 = 22
        let expected = Fp2E::new([FpE::from(113u64), FpE::from(22u64)]);
        assert_eq!(a * b, expected);
    }

    #[test]
    fn test_fp2_mul_by_one() {
        let a = Fp2E::new([FpE::from(12u64), FpE::from(5u64)]);
        let one = Fp2E::one();
        assert_eq!(&a * &one, a);
    }

    #[test]
    fn test_fp2_mul_by_zero() {
        let a = Fp2E::new([FpE::from(12u64), FpE::from(5u64)]);
        let zero = Fp2E::zero();
        assert_eq!(&a * &zero, zero);
    }

    #[test]
    fn test_fp2_inv() {
        let a = Fp2E::new([FpE::from(12u64), FpE::from(5u64)]);
        let a_inv = a.inv().unwrap();
        assert_eq!(&a * &a_inv, Fp2E::one());
    }

    #[test]
    fn test_fp2_inv_one() {
        let one = Fp2E::one();
        assert_eq!(one.inv().unwrap(), one);
    }

    #[test]
    fn test_fp2_div() {
        let a = Fp2E::new([FpE::from(12u64), FpE::from(5u64)]);
        let b = Fp2E::new([FpE::from(4u64), FpE::from(2u64)]);
        let result = (&a / &b).unwrap();
        // Verify: result * b = a
        assert_eq!(&result * &b, a);
    }

    #[test]
    fn test_fp2_pow() {
        let a = Fp2E::new([FpE::from(2u64), FpE::from(3u64)]);
        let a_squared = &a * &a;
        let a_cubed = &a_squared * &a;
        assert_eq!(a.pow(2u64), a_squared);
        assert_eq!(a.pow(3u64), a_cubed);
    }

    #[test]
    fn test_fp2_conjugate() {
        let a = Fp2E::new([FpE::from(12u64), FpE::from(5u64)]);
        let expected = Fp2E::new([FpE::from(12u64), -FpE::from(5u64)]);
        assert_eq!(a.conjugate(), expected);
    }

    #[test]
    fn test_fp2_neg() {
        let a = Fp2E::new([FpE::from(12u64), FpE::from(5u64)]);
        let neg_a = -&a;
        assert_eq!(&a + &neg_a, Fp2E::zero());
    }

    #[test]
    fn test_fp2_square_equals_mul() {
        let a = Fp2E::new([FpE::from(7u64), FpE::from(11u64)]);
        assert_eq!(a.square(), &a * &a);
    }

    // =====================================================
    // CUBIC EXTENSION TESTS
    // =====================================================

    #[test]
    fn test_fp3_add() {
        let a = Fp3E::new([FpE::from(1u64), FpE::from(2u64), FpE::from(3u64)]);
        let b = Fp3E::new([FpE::from(4u64), FpE::from(5u64), FpE::from(6u64)]);
        let expected = Fp3E::new([FpE::from(5u64), FpE::from(7u64), FpE::from(9u64)]);
        assert_eq!(a + b, expected);
    }

    #[test]
    fn test_fp3_sub() {
        let a = Fp3E::new([FpE::from(10u64), FpE::from(8u64), FpE::from(6u64)]);
        let b = Fp3E::new([FpE::from(3u64), FpE::from(2u64), FpE::from(1u64)]);
        let expected = Fp3E::new([FpE::from(7u64), FpE::from(6u64), FpE::from(5u64)]);
        assert_eq!(a - b, expected);
    }

    #[test]
    fn test_fp3_mul_by_one() {
        let a = Fp3E::new([FpE::from(12u64), FpE::from(5u64), FpE::from(7u64)]);
        let one = Fp3E::one();
        assert_eq!(&a * &one, a);
    }

    #[test]
    fn test_fp3_mul_by_zero() {
        let a = Fp3E::new([FpE::from(12u64), FpE::from(5u64), FpE::from(7u64)]);
        let zero = Fp3E::zero();
        assert_eq!(&a * &zero, zero);
    }

    #[test]
    fn test_fp3_mul() {
        let a = Fp3E::new([FpE::from(1u64), FpE::from(2u64), FpE::from(3u64)]);
        let b = Fp3E::new([FpE::from(4u64), FpE::from(5u64), FpE::from(6u64)]);
        // Verify multiplication is commutative
        assert_eq!(&a * &b, &b * &a);
    }

    #[test]
    fn test_fp3_inv() {
        let a = Fp3E::new([FpE::from(12u64), FpE::from(5u64), FpE::from(7u64)]);
        let a_inv = a.inv().unwrap();
        assert_eq!(&a * &a_inv, Fp3E::one());
    }

    #[test]
    fn test_fp3_inv_one() {
        let one = Fp3E::one();
        assert_eq!(one.inv().unwrap(), one);
    }

    #[test]
    fn test_fp3_div() {
        let a = Fp3E::new([FpE::from(12u64), FpE::from(5u64), FpE::from(7u64)]);
        let b = Fp3E::new([FpE::from(4u64), FpE::from(2u64), FpE::from(3u64)]);
        let result = (&a / &b).unwrap();
        // Verify: result * b = a
        assert_eq!(&result * &b, a);
    }

    #[test]
    fn test_fp3_pow() {
        let a = Fp3E::new([FpE::from(2u64), FpE::from(3u64), FpE::from(4u64)]);
        let a_squared = &a * &a;
        let a_cubed = &a_squared * &a;
        assert_eq!(a.pow(2u64), a_squared);
        assert_eq!(a.pow(3u64), a_cubed);
    }

    #[test]
    fn test_fp3_neg() {
        let a = Fp3E::new([FpE::from(12u64), FpE::from(5u64), FpE::from(7u64)]);
        let neg_a = -&a;
        assert_eq!(&a + &neg_a, Fp3E::zero());
    }

    // =====================================================
    // EMBEDDING TESTS (Base field into extension)
    // =====================================================

    #[test]
    fn test_fp2_from_base() {
        let base = FpE::from(42u64);
        let ext = Fp2E::from(42u64);
        assert_eq!(ext.value()[0], base);
        assert_eq!(ext.value()[1], FpE::zero());
    }

    #[test]
    fn test_fp3_from_base() {
        let base = FpE::from(42u64);
        let ext = Fp3E::from(42u64);
        assert_eq!(ext.value()[0], base);
        assert_eq!(ext.value()[1], FpE::zero());
        assert_eq!(ext.value()[2], FpE::zero());
    }

    #[test]
    fn test_fp2_base_mul() {
        // Test that base field elements multiply correctly in the extension
        let a = FpE::from(5u64);
        let b = Fp2E::new([FpE::from(2u64), FpE::from(3u64)]);
        let result = a * &b;
        let expected = Fp2E::new([FpE::from(10u64), FpE::from(15u64)]);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_fp3_base_mul() {
        // Test that base field elements multiply correctly in the extension
        let a = FpE::from(5u64);
        let b = Fp3E::new([FpE::from(2u64), FpE::from(3u64), FpE::from(4u64)]);
        let result = a * &b;
        let expected = Fp3E::new([FpE::from(10u64), FpE::from(15u64), FpE::from(20u64)]);
        assert_eq!(result, expected);
    }

    // =====================================================
    // ASSOCIATIVITY AND DISTRIBUTIVITY TESTS
    // =====================================================

    #[test]
    fn test_fp2_mul_associativity() {
        let a = Fp2E::new([FpE::from(2u64), FpE::from(3u64)]);
        let b = Fp2E::new([FpE::from(4u64), FpE::from(5u64)]);
        let c = Fp2E::new([FpE::from(6u64), FpE::from(7u64)]);
        assert_eq!(&(&a * &b) * &c, &a * &(&b * &c));
    }

    #[test]
    fn test_fp3_mul_associativity() {
        let a = Fp3E::new([FpE::from(2u64), FpE::from(3u64), FpE::from(4u64)]);
        let b = Fp3E::new([FpE::from(5u64), FpE::from(6u64), FpE::from(7u64)]);
        let c = Fp3E::new([FpE::from(8u64), FpE::from(9u64), FpE::from(10u64)]);
        assert_eq!(&(&a * &b) * &c, &a * &(&b * &c));
    }

    #[test]
    fn test_fp2_distributivity() {
        let a = Fp2E::new([FpE::from(2u64), FpE::from(3u64)]);
        let b = Fp2E::new([FpE::from(4u64), FpE::from(5u64)]);
        let c = Fp2E::new([FpE::from(6u64), FpE::from(7u64)]);
        // a * (b + c) = a * b + a * c
        assert_eq!(&a * &(&b + &c), &(&a * &b) + &(&a * &c));
    }

    #[test]
    fn test_fp3_distributivity() {
        let a = Fp3E::new([FpE::from(2u64), FpE::from(3u64), FpE::from(4u64)]);
        let b = Fp3E::new([FpE::from(5u64), FpE::from(6u64), FpE::from(7u64)]);
        let c = Fp3E::new([FpE::from(8u64), FpE::from(9u64), FpE::from(10u64)]);
        // a * (b + c) = a * b + a * c
        assert_eq!(&a * &(&b + &c), &(&a * &b) + &(&a * &c));
    }

    // =====================================================
    // RANDOM ELEMENT TESTS WITH LARGER VALUES
    // =====================================================

    #[test]
    fn test_fp2_large_values() {
        let a = Fp2E::new([FpE::from(18446744069414584300u64), FpE::from(12345678901234567u64)]);
        let b = Fp2E::new([FpE::from(9876543210987654u64), FpE::from(11111111111111111u64)]);

        // Test that a * a^-1 = 1
        let a_inv = a.inv().unwrap();
        assert_eq!(&a * &a_inv, Fp2E::one());

        // Test division
        let result = (&a / &b).unwrap();
        assert_eq!(&result * &b, a);
    }

    #[test]
    fn test_fp3_large_values() {
        let a = Fp3E::new([
            FpE::from(18446744069414584300u64),
            FpE::from(12345678901234567u64),
            FpE::from(98765432109876543u64),
        ]);
        let b = Fp3E::new([
            FpE::from(9876543210987654u64),
            FpE::from(11111111111111111u64),
            FpE::from(22222222222222222u64),
        ]);

        // Test that a * a^-1 = 1
        let a_inv = a.inv().unwrap();
        assert_eq!(&a * &a_inv, Fp3E::one());

        // Test division
        let result = (&a / &b).unwrap();
        assert_eq!(&result * &b, a);
    }
}
