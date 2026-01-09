#[cfg(test)]
mod tests {
    use crate::cyclic_group::IsGroup;
    use crate::field::element::FieldElement;
    use crate::field::errors::FieldError;
    use crate::field::fields::u64_prime_field::U64PrimeField;
    use crate::field::traits::IsPrimeField;
    use crate::traits::ByteConversion;

    const MODULUS: u64 = 13;
    type F = U64PrimeField<MODULUS>;
    type FE = FieldElement<F>;

    #[test]
    fn from_hex_for_b_is_11() {
        assert_eq!(F::from_hex("B").unwrap(), 11);
    }

    #[test]
    fn from_hex_for_0x1_a_is_26() {
        assert_eq!(F::from_hex("0x1a").unwrap(), 26);
    }

    #[cfg(feature = "std")]
    #[test]
    fn to_hex_test_works_1() {
        let num = F::from_hex("B").unwrap();
        assert_eq!(F::to_hex(&num), "B");
    }

    #[cfg(feature = "std")]
    #[test]
    fn to_hex_test_works_2() {
        let num = F::from_hex("0x1a").unwrap();
        assert_eq!(F::to_hex(&num), "1A");
    }

    #[test]
    fn bit_size_of_mod_13_field_is_4() {
        assert_eq!(
            <U64PrimeField<MODULUS> as crate::field::traits::IsPrimeField>::field_bit_size(),
            4
        );
    }

    #[test]
    fn bit_size_of_big_mod_field_is_64() {
        const MODULUS: u64 = 10000000000000000000;
        assert_eq!(
            <U64PrimeField<MODULUS> as crate::field::traits::IsPrimeField>::field_bit_size(),
            64
        );
    }

    #[test]
    fn bit_size_of_63_bit_mod_field_is_63() {
        const MODULUS: u64 = 9000000000000000000;
        assert_eq!(
            <U64PrimeField<MODULUS> as crate::field::traits::IsPrimeField>::field_bit_size(),
            63
        );
    }

    #[test]
    fn two_plus_one_is_three() {
        assert_eq!(FE::new(2) + FE::new(1), FE::new(3));
    }

    #[test]
    fn max_order_plus_1_is_0() {
        assert_eq!(FE::new(MODULUS - 1) + FE::new(1), FE::new(0));
    }

    #[test]
    fn when_comparing_13_and_13_they_are_equal() {
        let a: FE = FE::new(13);
        let b: FE = FE::new(13);
        assert_eq!(a, b);
    }

    #[test]
    fn when_comparing_13_and_8_they_are_different() {
        let a: FE = FE::new(13);
        let b: FE = FE::new(8);
        assert_ne!(a, b);
    }

    #[test]
    fn mul_neutral_element() {
        let a: FE = FE::new(1);
        let b: FE = FE::new(2);
        assert_eq!(a * b, FE::new(2));
    }

    #[test]
    fn mul_2_3_is_6() {
        let a: FE = FE::new(2);
        let b: FE = FE::new(3);
        assert_eq!(a * b, FE::new(6));
    }

    #[test]
    fn mul_order_minus_1() {
        let a: FE = FE::new(MODULUS - 1);
        let b: FE = FE::new(MODULUS - 1);
        assert_eq!(a * b, FE::new(1));
    }

    #[test]
    fn inv_0_error() {
        let result = FE::new(0).inv();
        assert!(matches!(result, Err(FieldError::InvZeroError)));
    }

    #[test]
    fn inv_2() {
        let a: FE = FE::new(2);
        assert_eq!(a * a.inv().unwrap(), FE::new(1));
    }

    #[test]
    fn pow_2_3() {
        assert_eq!(FE::new(2).pow(3_u64), FE::new(8))
    }

    #[test]
    fn pow_p_minus_1() {
        assert_eq!(FE::new(2).pow(MODULUS - 1), FE::new(1))
    }

    #[test]
    fn div_1() {
        assert_eq!(FE::new(2) * FE::new(1).inv().unwrap(), FE::new(2))
    }

    #[test]
    fn div_4_2() {
        assert_eq!(FE::new(4) * FE::new(2).inv().unwrap(), FE::new(2))
    }

    #[test]
    fn div_4_3() {
        assert_eq!(
            FE::new(4) * FE::new(3).inv().unwrap() * FE::new(3),
            FE::new(4)
        )
    }

    #[test]
    fn two_plus_its_additive_inv_is_0() {
        let two = FE::new(2);

        assert_eq!(two + (-two), FE::new(0))
    }

    #[test]
    fn four_minus_three_is_1() {
        let four = FE::new(4);
        let three = FE::new(3);

        assert_eq!(four - three, FE::new(1))
    }

    #[test]
    fn zero_minus_1_is_order_minus_1() {
        let zero = FE::new(0);
        let one = FE::new(1);

        assert_eq!(zero - one, FE::new(MODULUS - 1))
    }

    #[test]
    fn neg_zero_is_zero() {
        let zero = FE::new(0);

        assert_eq!(-zero, zero);
    }

    #[test]
    fn zero_constructor_returns_zero() {
        assert_eq!(FE::new(0), FE::new(0));
    }

    #[test]
    fn field_element_as_group_element_multiplication_by_scalar_works_as_multiplication_in_finite_fields()
     {
        let a = FE::new(3);
        let b = FE::new(12);
        assert_eq!(a * b, a.operate_with_self(12_u16));
    }

    #[test]
    #[cfg(feature = "alloc")]
    fn to_bytes_from_bytes_be_is_the_identity() {
        let x = FE::new(12345);
        assert_eq!(FE::from_bytes_be(&x.to_bytes_be()).unwrap(), x);
    }

    #[test]
    #[cfg(feature = "alloc")]
    fn from_bytes_to_bytes_be_is_the_identity_for_one() {
        let bytes = [0, 0, 0, 0, 0, 0, 0, 1];
        assert_eq!(FE::from_bytes_be(&bytes).unwrap().to_bytes_be(), bytes);
    }

    #[test]
    #[cfg(feature = "alloc")]
    fn to_bytes_from_bytes_le_is_the_identity() {
        let x = FE::new(12345);
        assert_eq!(FE::from_bytes_le(&x.to_bytes_le()).unwrap(), x);
    }

    #[test]
    #[cfg(feature = "alloc")]
    fn from_bytes_to_bytes_le_is_the_identity_for_one() {
        let bytes = [1, 0, 0, 0, 0, 0, 0, 0];
        assert_eq!(FE::from_bytes_le(&bytes).unwrap().to_bytes_le(), bytes);
    }

    #[test]
    fn creating_a_field_element_from_its_representative_returns_the_same_element_1() {
        let change = 1;
        let f1 = FE::new(MODULUS + change);
        let f2 = FE::new(f1.representative());
        assert_eq!(f1, f2);
    }

    #[test]
    fn creating_a_field_element_from_its_representative_returns_the_same_element_2() {
        let change = 8;
        let f1 = FE::new(MODULUS + change);
        let f2 = FE::new(f1.representative());
        assert_eq!(f1, f2);
    }
}
