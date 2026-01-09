#[cfg(test)]
mod tests {
    use crate::field::fields::binary::field::{BinaryFieldError, TowerFieldElement};
    use proptest::prelude::*;

    #[test]
    fn test_new_safe() {
        // Test with level too large
        let elem = TowerFieldElement::new(0, 8);
        assert_eq!(elem.num_level, 7); // Should be capped at 7

        // Test with value too large for level
        let elem = TowerFieldElement::new(4, 1); // Level 1 can only store 0-3
        assert_eq!(elem.value, 0); // Should mask to 0 (100 & 11 = 00)
    }

    #[test]
    fn test_addition() {
        let a = TowerFieldElement::new(5, 9); // 8 bits
        let b = TowerFieldElement::new(3, 2); // 4 bits

        let c = a + b;
        // 5 (0101) + 3 (0011) should be 6 (0110) at level 3
        assert_eq!(c.value, 6);
        assert_eq!(c.num_level, 7);

        // Test commutative property
        let d = b + a;
        assert_eq!(d, c);
    }

    #[test]
    fn mul_in_level_0() {
        let a = TowerFieldElement::new(0, 0);
        let b = TowerFieldElement::new(1, 0);
        assert_eq!(a * a, a);
        assert_eq!(a * b, a);
        assert_eq!(b * b, b);
    }

    #[test]
    fn mul_in_level_1() {
        let a = TowerFieldElement::new(0b00, 1); // 0
        let b = TowerFieldElement::new(0b01, 1); // 1
        let c = TowerFieldElement::new(0b10, 1); // x
        let d = TowerFieldElement::new(0b11, 1); // x + 1
        assert_eq!(a * a, a);
        assert_eq!(a * b, a);
        assert_eq!(b * c, c);
        assert_eq!(c * d, b);
    }

    #[test]
    fn mul_in_level_2() {
        let a = TowerFieldElement::new(0b0000, 2); // 0
        let b = TowerFieldElement::new(0b0001, 2); // 1
        let c = TowerFieldElement::new(0b0010, 2); // x
        let d = TowerFieldElement::new(0b0011, 2); // x + 1
        let e = TowerFieldElement::new(0b0100, 2); // y
        let f = TowerFieldElement::new(0b0101, 2); // y + 1
        let g = TowerFieldElement::new(0b0110, 2); // y + x
        let h = TowerFieldElement::new(0b0111, 2); // y + x + 1
        let i = TowerFieldElement::new(0b1000, 2); // yx
        let j = TowerFieldElement::new(0b1001, 2); // yx + 1
        let k = TowerFieldElement::new(0b1010, 2); // yx + x
        let l = TowerFieldElement::new(0b1011, 2); // yx + x + 1
        let n = TowerFieldElement::new(0b1100, 2); // yx + y
        let m = TowerFieldElement::new(0b1101, 2); // yx + y + 1
        let o = TowerFieldElement::new(0b1110, 2); // yx + y + x
        let p = TowerFieldElement::new(0b1111, 2); // yx + y + x + 1

        assert_eq!(a * p, a); // 0 * (yx + y + x + 1) = 0
        assert_eq!(a * l, a); // 0 * (yx + x + 1) = 0
        assert_eq!(b * m, m); // 1 * 1 = 1
        assert_eq!(c * e, i); // x * y = xy
        assert_eq!(c * c, d); // x * x = x + 1
        assert_eq!(g * h, n); //(y + x)(y + x + 1) = yx + y
        assert_eq!(k * j, b); // (yx + x)(yx + 1) = 1
        assert_eq!(j * f, d); // (yx + 1)(y + 1) = x + 1
        assert_eq!(e * e, j); // y * y = yx + 1
        assert_eq!(n * o, k); // (yx + y)(yx + y + x) = yx + x
    }

    #[test]
    fn mul_between_different_levels() {
        let a = TowerFieldElement::new(0b10, 1); // x
        let b = TowerFieldElement::new(0b0100, 2); // y
        let c = TowerFieldElement::new(0b1000, 2); // yx
        assert_eq!(a * b, c);
    }

    #[test]
    fn test_correct_level_mul() {
        let a = TowerFieldElement::new(0b1111, 5);
        let b = TowerFieldElement::new(0b1010, 2);
        assert_eq!((a * b).num_level, 5);
    }

    #[test]
    fn mul_is_asociative() {
        let a = TowerFieldElement::new(83, 7);
        let b = TowerFieldElement::new(31, 5);
        let c = TowerFieldElement::new(3, 2);
        let ab = a * b;
        let bc = b * c;
        assert_eq!(ab * c, a * bc);
    }

    #[test]
    fn mul_is_conmutative() {
        let a = TowerFieldElement::new(127, 7);
        let b = TowerFieldElement::new(6, 3);
        let ab = a * b;
        let ba = b * a;
        assert_eq!(ab, ba);
    }

    #[test]
    fn test_inverse() {
        let a0 = TowerFieldElement::new(1, 0);
        let inv_a0 = a0.inv().unwrap();
        assert_eq!(inv_a0.value, 1);
        assert_eq!(inv_a0.num_level, 0);

        let a1 = TowerFieldElement::new(2, 1);
        let inv_a1 = a1.inv().unwrap();
        assert_eq!(inv_a1.value, 3); // because 10 * 11 = 01.
        assert_eq!(inv_a1.num_level, 1);

        // Verify a * a^(-1) = 1
        let a2 = TowerFieldElement::new(15, 4);
        let inv_a2 = a2.inv().unwrap();
        let one = TowerFieldElement::new(1, 4);
        assert_eq!(a2 * inv_a2, one);

        let a3 = TowerFieldElement::new(30, 5);
        let inv_a3 = a3.inv().unwrap();
        let one = TowerFieldElement::new(1, 5);
        assert_eq!(a3 * inv_a3, one);

        let zero = TowerFieldElement::zero();
        assert!(matches!(zero.inv(), Err(BinaryFieldError::InverseOfZero)));
    }

    #[test]
    fn test_multiplication_overflow() {
        for level in 0..7 {
            let max_value = (1u128 << (1 << level)) - 1; // Maximum value for this level
            let a = TowerFieldElement::new(max_value, level);
            let b = TowerFieldElement::new(max_value, level);

            let result = a * b;

            // Result should be properly reduced
            assert!(result.value < (1u128 << result.num_bits()));
        }
    }

    #[test]
    fn test_split_join_consistency() {
        // Test that join and split are consistent operations
        for i in 0..20 {
            let original = TowerFieldElement::new(i, 3);
            let (hi, lo) = original.split();
            let rejoined = hi.join(&lo);

            assert_eq!(rejoined, original);
        }
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_bin_representation() {
        let a = TowerFieldElement::new(0b1010, 5);
        assert_eq!(a.to_binary_string(), "00000000000000000000000000001010");
        let b = TowerFieldElement::new(0b1010, 4);
        assert_eq!(b.to_binary_string(), "0000000000001010");
    }

    // Strategy to generate a TowerFieldElement with a random level between 0 and 7.
    // For a given level:
    // - The number of bits is computed as 1 << level.
    // - For level 0, valid values are 0 to (1 << 1) - 1 = 1.
    // - For level > 0, valid values are 0 to (1 << (1 << level)) - 1.
    fn arb_tower_element_any() -> impl Strategy<Value = TowerFieldElement> {
        (0usize..=7)
            .prop_flat_map(|level| {
                let max_val = if level == 0 {
                    1
                } else if (1usize << level) >= 128 {
                    u128::MAX
                } else {
                    (1u128 << (1 << level)) - 1
                };
                (Just(level), 0u128..=max_val)
            })
            .prop_map(|(level, val)| TowerFieldElement::new(val, level))
    }

    #[cfg(feature = "std")]
    proptest! {
        // Test that multiplication is commutative:
        // For any two randomly generated elements, a * b should equal b * a.
        #[test]
        fn test_mul_commutative(a in arb_tower_element_any(), b in arb_tower_element_any()) {
            prop_assert_eq!(a * b, b * a);
        }

        // Test that multiplication is associative:
        // For any three randomly generated elements, (a * b) * c should equal a * (b * c).
        #[test]
        fn test_mul_associative(a in arb_tower_element_any(), b in arb_tower_element_any(), c in arb_tower_element_any()) {
            prop_assert_eq!((a * b) * c, a * (b * c));
        }
    }
}
