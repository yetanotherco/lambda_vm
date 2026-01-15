#[cfg(test)]
mod tests {
    use crate::elliptic_curve::short_weierstrass::curves::bn_254::field_extension::BN254PrimeField;
    use crate::errors::ByteConversionError;
    use crate::field::element::FieldElement;
    use crate::field::fields::fft_friendly::{
        babybear_u32::Babybear31PrimeField, stark_252_prime_field::Stark252PrimeField,
    };
    use crate::field::fields::montgomery_backed_prime_fields::{IsModulus, U384PrimeField};
    use crate::field::fields::goldilocks::Goldilocks64Field;
    use crate::field::fields::u64_prime_field::U64PrimeField;
    use crate::unsigned_integer::element::U384;
    #[cfg(feature = "alloc")]
    use crate::unsigned_integer::element::UnsignedInteger;
    #[cfg(feature = "alloc")]
    use alloc::vec::Vec;
    use num_bigint::BigUint;
    #[cfg(feature = "alloc")]
    use proptest::collection;
    use proptest::{prelude::*, prop_compose, proptest, strategy::Strategy};

    #[test]
    fn test_std_iter_sum_field_element() {
        let n = 164;
        const MODULUS: u64 = 18446744069414584321; // Goldilocks prime
        assert_eq!(
            (0..n)
                .map(|x| { FieldElement::<Goldilocks64Field>::from(x) })
                .sum::<FieldElement<Goldilocks64Field>>()
                .representative(),
            ((n - 1) as f64 / 2. * ((n - 1) as f64 + 1.)) as u64 % MODULUS
        );
    }

    #[test]
    fn test_std_iter_sum_field_element_zero_length() {
        let n = 0;
        assert_eq!(
            (0..n)
                .map(|x| { FieldElement::<Goldilocks64Field>::from(x) })
                .sum::<FieldElement<Goldilocks64Field>>()
                .representative(),
            0
        );
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_display_montgomery_field() {
        use alloc::format;

        let zero_field_element = FieldElement::<Stark252PrimeField>::from(0);
        assert_eq!(format!("{zero_field_element}"), "0x0");

        let some_field_element =
            FieldElement::<Stark252PrimeField>::from(&UnsignedInteger::from_limbs([
                0x0, 0x1, 0x0, 0x1,
            ]));

        // it should start with the first non-zero digit. Each limb has 16 digits in hex.
        assert_eq!(
            format!("{some_field_element}"),
            format!("0x{}{}{}{}", "1", "0".repeat(16), "0".repeat(15), "1")
        );
    }

    #[test]
    fn one_of_sqrt_roots_for_4_is_2() {
        type FrField = Stark252PrimeField;
        type FrElement = FieldElement<FrField>;

        let input = FrElement::from(4);
        let sqrt = input.sqrt().unwrap();
        let result = FrElement::from(2);
        assert_eq!(sqrt.0, result);
    }

    #[test]
    fn one_of_sqrt_roots_for_5_is_28_mod_41() {
        let input = FieldElement::<U64PrimeField<41>>::from(5);
        let sqrt = input.sqrt().unwrap();
        let result = FieldElement::from(28);
        assert_eq!(sqrt.0, result);
        assert_eq!(sqrt.1, -result);
    }

    #[test]
    fn one_of_sqrt_roots_for_25_is_5() {
        type FrField = Stark252PrimeField;
        type FrElement = FieldElement<FrField>;
        let input = FrElement::from(25);
        let sqrt = input.sqrt().unwrap();
        let five = FrElement::from(5);
        assert!(sqrt.1 == five || sqrt.0 == five);
    }

    #[test]
    fn sqrt_works_for_prime_minus_one() {
        type FrField = Stark252PrimeField;
        type FrElement = FieldElement<FrField>;

        let input = -FrElement::from(1);
        let sqrt = input.sqrt().unwrap();
        assert_eq!(sqrt.0.square(), input);
        assert_eq!(sqrt.1.square(), input);
        assert_ne!(sqrt.0, sqrt.1);
    }

    #[test]
    fn one_of_sqrt_roots_for_25_is_5_in_stark_field() {
        type FrField = Stark252PrimeField;
        type FrElement = FieldElement<FrField>;

        let input = FrElement::from(25);
        let sqrt = input.sqrt().unwrap();
        let result = FrElement::from(5);
        assert_eq!(sqrt.0, result);
        assert_eq!(sqrt.1, -result);
    }

    #[test]
    fn sqrt_roots_for_0_are_0_in_stark_field() {
        type FrField = Stark252PrimeField;
        type FrElement = FieldElement<FrField>;

        let input = FrElement::from(0);
        let sqrt = input.sqrt().unwrap();
        let result = FrElement::from(0);
        assert_eq!(sqrt.0, result);
        assert_eq!(sqrt.1, result);
    }

    #[test]
    fn sqrt_of_27_for_stark_field_does_not_exist() {
        type FrField = Stark252PrimeField;
        type FrElement = FieldElement<FrField>;

        let input = FrElement::from(27);
        let sqrt = input.sqrt();
        assert!(sqrt.is_none());
    }

    #[test]
    fn from_hex_1a_is_26_for_stark252_prime_field_element() {
        type F = Stark252PrimeField;
        type FE = FieldElement<F>;
        assert_eq!(FE::from_hex("1a").unwrap(), FE::from(26))
    }

    #[test]
    fn from_hex_unchecked_zero_x_1a_is_26_for_stark252_prime_field_element() {
        type F = Stark252PrimeField;
        type FE = FieldElement<F>;
        assert_eq!(FE::from_hex_unchecked("0x1a"), FE::from(26))
    }

    #[test]
    fn construct_new_field_element_from_empty_string_errs() {
        type F = Stark252PrimeField;
        type FE = FieldElement<F>;
        assert!(FE::from_hex("").is_err());
    }

    #[test]
    fn construct_new_field_element_from_value_bigger_than_modulus() {
        type F = Stark252PrimeField;
        type FE = FieldElement<F>;
        // A number that consists of 255 1s is bigger than the `Stark252PrimeField` modulus
        assert!(FE::from_hex(&format!("0x{}", "f".repeat(65))).is_err());
    }

    prop_compose! {
        fn field_element()(num in any::<u64>().prop_filter("Avoid null coefficients", |x| x != &0)) -> FieldElement::<Stark252PrimeField> {
            FieldElement::<Stark252PrimeField>::from(num)
        }
    }

    prop_compose! {
        #[cfg(feature = "alloc")]
        fn field_vec(max_exp: u8)(vec in collection::vec(field_element(), 0..1 << max_exp)) -> Vec<FieldElement::<Stark252PrimeField>> {
            vec
        }
    }

    proptest! {
        #[cfg(feature = "alloc")]
        #[test]
        fn test_inplace_batch_inverse_returns_inverses(vec in field_vec(10)) {
            let input: Vec<_> = vec.into_iter().filter(|x| x != &FieldElement::<Stark252PrimeField>::zero()).collect();
            let mut inverses = input.clone();
            FieldElement::inplace_batch_inverse(&mut inverses).unwrap();

            for (i, x) in inverses.into_iter().enumerate() {
                prop_assert_eq!(x * input[i], FieldElement::<Stark252PrimeField>::one());
            }
        }
    }

    // Tests for BigUint conversion.
    // We define different fields to test the conversion.

    // Prime field with modulus 17 and base type u64.
    type U64F17 = U64PrimeField<17>;
    type U64F17Element = FieldElement<U64F17>;

    // Baby Bear Prime field with u32 montgomery backend.
    type BabyBear = Babybear31PrimeField;
    type BabyBearElement = FieldElement<BabyBear>;

    // Prime field with modulus 23, using u64 montgomery backend of 6 limbs.
    #[derive(Clone, Debug)]
    struct U384Modulus23;
    impl IsModulus<U384> for U384Modulus23 {
        const MODULUS: U384 = UnsignedInteger::from_u64(23);
    }
    type U384F23 = U384PrimeField<U384Modulus23>;
    type U384F23Element = FieldElement<U384F23>;

    #[test]
    fn test_reduced_biguint_conversion_u64_field() {
        let value = BigUint::from(10u32);
        let fe = U64F17Element::try_from(value.clone()).unwrap();
        let back_to_biguint = fe.to_big_uint();
        assert_eq!(value, back_to_biguint);
    }

    #[test]
    fn test_reduced_biguint_conversion_baby_bear() {
        let value = BigUint::from(1000u32);
        let fe = BabyBearElement::from_reduced_big_uint(&value).unwrap();
        assert_eq!(fe, BabyBearElement::from(1000));
        let back_to_biguint = fe.to_big_uint();
        assert_eq!(value, back_to_biguint);
    }

    #[test]
    fn test_reduced_biguint_conversion_u384_field() {
        let value = BigUint::from(22u32);
        let fe = U384F23Element::from_reduced_big_uint(&value).unwrap();
        let back_to_biguint = fe.to_big_uint();
        assert_eq!(value, back_to_biguint);
    }
    #[test]
    fn test_bn254_field_biguint_conversion() {
        type BN254Element = FieldElement<BN254PrimeField>;
        let value = BigUint::from(1001u32);
        let fe = BN254Element::from_reduced_big_uint(&value).unwrap();
        let back_to_biguint = fe.to_big_uint();
        assert_eq!(value, back_to_biguint);
    }

    #[test]
    fn non_reduced_biguint_value_conversion_errors_u64_field() {
        let value = BigUint::from(17u32);
        let result = U64F17Element::from_reduced_big_uint(&value);
        assert_eq!(result, Err(ByteConversionError::ValueNotReduced));
    }

    #[test]
    fn non_reduced_biguint_value_conversion_errors_baby_bear() {
        let value = BigUint::from(2013265921u32);
        let result = BabyBearElement::try_from(value);
        assert_eq!(result, Err(ByteConversionError::ValueNotReduced));
    }

    #[test]
    fn non_reduced_biguint_value_conversion_errors_u384_field() {
        let value = BigUint::from(30u32);
        let result = U384F23Element::try_from(value);
        assert_eq!(result, Err(ByteConversionError::ValueNotReduced));
    }

    #[test]
    fn test_hex_string_conversion_u64_field() {
        let hex_str = "0x0a";
        let fe = U64F17Element::from_hex_str(hex_str).unwrap();
        assert_eq!(fe, U64F17Element::from(10));
        assert_eq!(fe.to_hex_str(), "0x0A");
    }

    #[test]
    fn test_hex_string_conversion_baby_bear() {
        let hex_str = "0x77FFFFFF"; // 2013265919
        let fe = BabyBearElement::from_hex_str(hex_str).unwrap();
        assert_eq!(fe, BabyBearElement::from(2013265919));
        assert_eq!(fe.to_hex_str(), "0x77FFFFFF");
    }

    #[test]
    fn test_hex_string_conversion_u384_field() {
        let hex_str = "0x14"; // 20
        let fe = U384F23Element::from_hex_str(hex_str).unwrap();
        assert_eq!(fe, U384F23Element::from(20));
        assert_eq!(fe.to_hex_str(), "0x14");
    }

    #[test]
    fn test_invalid_hex_string_u64_field() {
        let hex_str = "0xzz";
        let result = U64F17Element::from_hex_str(hex_str);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_hex_string_baby_bear() {
        // modulus = 0x78000001
        let hex_str = "0x78000001";
        let result = BabyBearElement::from_hex_str(hex_str);
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_hex_string() {
        let hex_str = "";
        let result = U64F17Element::from_hex_str(hex_str);
        assert!(result.is_err());
    }
}
