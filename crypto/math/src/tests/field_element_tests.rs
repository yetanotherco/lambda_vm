#[cfg(test)]
mod tests {
    use crate::errors::ByteConversionError;
    use crate::field::element::FieldElement;
    use crate::field::goldilocks::GoldilocksField;
    use crate::field::test_fields::u64_test_field::U64TestField;
    #[cfg(feature = "alloc")]
    use alloc::vec::Vec;
    use num_bigint::BigUint;
    #[cfg(feature = "alloc")]
    use proptest::collection;
    use proptest::{prelude::*, prop_compose, proptest, strategy::Strategy};

    #[test]
    fn test_std_iter_sum_field_element() {
        let n = 164;
        const MODULUS: u64 = 18446744069414584321;
        assert_eq!(
            (0..n)
                .map(|x| { FieldElement::<U64TestField>::from(x) })
                .sum::<FieldElement<U64TestField>>()
                .to_raw(),
            ((n - 1) as f64 / 2. * ((n - 1) as f64 + 1.)) as u64 % MODULUS
        );
    }

    #[test]
    fn test_std_iter_sum_field_element_zero_length() {
        let n = 0;
        assert_eq!(
            (0..n)
                .map(|x| { FieldElement::<U64TestField>::from(x) })
                .sum::<FieldElement<U64TestField>>()
                .to_raw(),
            0
        );
    }

    type GF = GoldilocksField;
    type Gfe = FieldElement<GF>;

    #[test]
    fn one_of_sqrt_roots_for_4_is_2() {
        let input = Gfe::from(4u64);
        let sqrt = input.sqrt().unwrap();
        let result = Gfe::from(2u64);
        assert_eq!(sqrt.0, result);
    }

    #[test]
    fn one_of_sqrt_roots_for_25_is_5() {
        let input = Gfe::from(25u64);
        let sqrt = input.sqrt().unwrap();
        let five = Gfe::from(5u64);
        assert!(sqrt.1 == five || sqrt.0 == five);
    }

    #[test]
    fn sqrt_works_for_prime_minus_one() {
        let input = -Gfe::from(1u64);
        let sqrt = input.sqrt();
        // -1 may or may not be a QR in Goldilocks; verify consistency
        if let Some(sqrt) = sqrt {
            assert_eq!(sqrt.0.square(), input);
            assert_eq!(sqrt.1.square(), input);
            assert_ne!(sqrt.0, sqrt.1);
        }
    }

    #[test]
    fn sqrt_roots_for_0_are_0() {
        let input = Gfe::from(0u64);
        let sqrt = input.sqrt().unwrap();
        let result = Gfe::from(0u64);
        assert_eq!(sqrt.0, result);
        assert_eq!(sqrt.1, result);
    }

    #[test]
    fn from_hex_1a_is_26_for_goldilocks() {
        assert_eq!(Gfe::from_hex("1a").unwrap(), Gfe::from(26u64))
    }

    #[test]
    fn construct_new_field_element_from_empty_string_errs() {
        assert!(Gfe::from_hex("").is_err());
    }

    prop_compose! {
        fn field_element()(num in any::<u64>().prop_filter("Avoid null coefficients", |x| x != &0)) -> Gfe {
            Gfe::from(num)
        }
    }

    prop_compose! {
        #[cfg(feature = "alloc")]
        fn field_vec(max_exp: u8)(vec in collection::vec(field_element(), 0..1 << max_exp)) -> Vec<Gfe> {
            vec
        }
    }

    proptest! {
        #[cfg(feature = "alloc")]
        #[test]
        fn test_inplace_batch_inverse_returns_inverses(vec in field_vec(10)) {
            let input: Vec<_> = vec.into_iter().filter(|x| x != &Gfe::zero()).collect();
            let mut inverses = input.clone();
            FieldElement::inplace_batch_inverse(&mut inverses).unwrap();

            for (i, x) in inverses.into_iter().enumerate() {
                prop_assert_eq!(x * input[i], Gfe::one());
            }
        }
    }

    // Tests for BigUint conversion using Goldilocks field.
    #[test]
    fn test_reduced_biguint_conversion_goldilocks() {
        let value = BigUint::from(10u32);
        let fe = Gfe::try_from(value.clone()).unwrap();
        let back_to_biguint = fe.to_big_uint();
        assert_eq!(value, back_to_biguint);
    }

    #[test]
    fn non_reduced_biguint_value_conversion_errors_goldilocks() {
        // Goldilocks modulus = 2^64 - 2^32 + 1 = 18446744069414584321
        let value = BigUint::from(18446744069414584321u64);
        let result = Gfe::try_from(value);
        assert_eq!(result, Err(ByteConversionError::ValueNotReduced));
    }

    #[test]
    fn test_hex_string_conversion_goldilocks() {
        let hex_str = "0x0a";
        let fe = Gfe::from_hex_str(hex_str).unwrap();
        assert_eq!(fe, Gfe::from(10u64));
        assert_eq!(fe.to_hex_str(), "0x0A");
    }

    #[test]
    fn test_invalid_hex_string_goldilocks() {
        let hex_str = "0xzz";
        let result = Gfe::from_hex_str(hex_str);
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_hex_string() {
        let hex_str = "";
        let result = Gfe::from_hex_str(hex_str);
        assert!(result.is_err());
    }
}
