#[cfg(test)]
mod tests_u384_prime_fields {
    use crate::field::element::FieldElement;
    use crate::field::errors::FieldError;
    use crate::field::fields::fft_friendly::stark_252_prime_field::Stark252PrimeField;
    use crate::field::fields::montgomery_backed_prime_fields::{
        IsModulus, U256PrimeField, U384PrimeField,
    };
    use crate::field::traits::HasDefaultTranscript;
    use crate::field::traits::IsField;
    use crate::field::traits::IsPrimeField;
    #[cfg(feature = "alloc")]
    use crate::traits::ByteConversion;
    use crate::unsigned_integer::element::U384;
    use crate::unsigned_integer::element::{U256, UnsignedInteger};

    use rand::Rng;
    use rand_chacha::{ChaCha20Rng, rand_core::SeedableRng};

    #[derive(Clone, Debug)]
    struct U384Modulus23;
    impl IsModulus<U384> for U384Modulus23 {
        const MODULUS: U384 = UnsignedInteger::from_u64(23);
    }

    type U384F23 = U384PrimeField<U384Modulus23>;
    type U384F23Element = FieldElement<U384F23>;

    #[test]
    fn u384_mod_23_uses_5_bits() {
        assert_eq!(U384F23::field_bit_size(), 5);
    }

    #[test]
    fn stark_252_prime_field_uses_252_bits() {
        assert_eq!(Stark252PrimeField::field_bit_size(), 252);
    }

    #[test]
    fn u256_mod_2_uses_1_bit() {
        #[derive(Clone, Debug)]
        struct U256Modulus1;
        impl IsModulus<U256> for U256Modulus1 {
            const MODULUS: U256 = UnsignedInteger::from_u64(2);
        }
        type U256OneField = U256PrimeField<U256Modulus1>;
        assert_eq!(U256OneField::field_bit_size(), 1);
    }

    #[test]
    fn u256_with_first_bit_set_uses_256_bit() {
        #[derive(Clone, Debug)]
        struct U256ModulusBig;
        impl IsModulus<U256> for U256ModulusBig {
            const MODULUS: U256 = UnsignedInteger::from_hex_unchecked(
                "F0000000F0000000F0000000F0000000F0000000F0000000F0000000F0000000",
            );
        }
        type U256OneField = U256PrimeField<U256ModulusBig>;
        assert_eq!(U256OneField::field_bit_size(), 256);
    }

    #[test]
    fn montgomery_backend_primefield_compute_r2_parameter() {
        let r2: U384 = UnsignedInteger {
            limbs: [0, 0, 0, 0, 0, 6],
        };
        assert_eq!(U384F23::R2, r2);
    }

    #[test]
    fn montgomery_backend_primefield_compute_mu_parameter() {
        assert_eq!(U384F23::MU, 3208129404123400281);
    }

    #[test]
    fn montgomery_backend_primefield_compute_zero_parameter() {
        let zero: U384 = UnsignedInteger {
            limbs: [0, 0, 0, 0, 0, 0],
        };
        assert_eq!(U384F23::ZERO, zero);
    }

    #[test]
    fn montgomery_backend_primefield_from_u64() {
        let a: U384 = UnsignedInteger {
            limbs: [0, 0, 0, 0, 0, 17],
        };
        assert_eq!(U384F23::from_u64(770_u64), a);
    }

    #[test]
    fn montgomery_backend_primefield_representative() {
        let a: U384 = UnsignedInteger {
            limbs: [0, 0, 0, 0, 0, 11],
        };
        assert_eq!(U384F23::representative(&U384F23::from_u64(770_u64)), a);
    }

    #[test]
    fn montgomery_backend_multiplication_works_0() {
        let x = U384F23Element::from(11_u64);
        let y = U384F23Element::from(10_u64);
        let c = U384F23Element::from(110_u64);
        assert_eq!(x * y, c);
    }

    #[test]
    #[cfg(feature = "lambdaworks-serde-string")]
    fn montgomery_backend_serialization_deserialization() {
        let x = U384F23Element::from(11_u64);
        let x_serialized = serde_json::to_string(&x).unwrap();
        let x_deserialized: U384F23Element = serde_json::from_str(&x_serialized).unwrap();
        // assert_eq!(x_serialized, "{\"value\":\"0xb\"}"); // serialization is no longer as hex string
        assert_eq!(x_deserialized, x);
    }

    #[test]
    fn doubling() {
        assert_eq!(
            U384F23Element::from(2).double(),
            U384F23Element::from(2) + U384F23Element::from(2),
        );
    }

    const ORDER: usize = 23;
    #[test]
    fn two_plus_one_is_three() {
        assert_eq!(
            U384F23Element::from(2) + U384F23Element::from(1),
            U384F23Element::from(3)
        );
    }

    #[test]
    fn max_order_plus_1_is_0() {
        assert_eq!(
            U384F23Element::from((ORDER - 1) as u64) + U384F23Element::from(1),
            U384F23Element::from(0)
        );
    }

    #[test]
    fn when_comparing_13_and_13_they_are_equal() {
        let a: U384F23Element = U384F23Element::from(13);
        let b: U384F23Element = U384F23Element::from(13);
        assert_eq!(a, b);
    }

    #[test]
    fn when_comparing_13_and_8_they_are_different() {
        let a: U384F23Element = U384F23Element::from(13);
        let b: U384F23Element = U384F23Element::from(8);
        assert_ne!(a, b);
    }

    #[test]
    fn mul_neutral_element() {
        let a: U384F23Element = U384F23Element::from(1);
        let b: U384F23Element = U384F23Element::from(2);
        assert_eq!(a * b, U384F23Element::from(2));
    }

    #[test]
    fn mul_2_3_is_6() {
        let a: U384F23Element = U384F23Element::from(2);
        let b: U384F23Element = U384F23Element::from(3);
        assert_eq!(a * b, U384F23Element::from(6));
    }

    #[test]
    fn mul_order_minus_1() {
        let a: U384F23Element = U384F23Element::from((ORDER - 1) as u64);
        let b: U384F23Element = U384F23Element::from((ORDER - 1) as u64);
        assert_eq!(a * b, U384F23Element::from(1));
    }

    #[test]
    fn inv_0_error() {
        let result = U384F23Element::from(0).inv();
        assert!(matches!(result, Err(FieldError::InvZeroError)))
    }

    #[test]
    fn inv_2() {
        let a: U384F23Element = U384F23Element::from(2);
        assert_eq!(&a * a.inv().unwrap(), U384F23Element::from(1));
    }

    #[test]
    fn pow_2_3() {
        assert_eq!(U384F23Element::from(2).pow(3_u64), U384F23Element::from(8))
    }

    #[test]
    fn pow_p_minus_1() {
        assert_eq!(
            U384F23Element::from(2).pow(ORDER - 1),
            U384F23Element::from(1)
        )
    }

    #[test]
    fn div_1() {
        assert_eq!(
            (U384F23Element::from(2) / U384F23Element::from(1)).unwrap(),
            U384F23Element::from(2)
        )
    }

    #[test]
    fn div_4_2() {
        assert_eq!(
            (U384F23Element::from(4) / U384F23Element::from(2)).unwrap(),
            U384F23Element::from(2)
        )
    }

    #[test]
    fn three_inverse() {
        let a = U384F23Element::from(3);
        let expected = U384F23Element::from(8);
        assert_eq!(a.inv().unwrap(), expected)
    }

    #[test]
    fn div_4_3() {
        assert_eq!(
            (U384F23Element::from(4) / U384F23Element::from(3)).unwrap() * U384F23Element::from(3),
            U384F23Element::from(4)
        )
    }

    #[test]
    fn two_plus_its_additive_inv_is_0() {
        let two = U384F23Element::from(2);

        assert_eq!(&two + (-&two), U384F23Element::from(0))
    }

    #[test]
    fn four_minus_three_is_1() {
        let four = U384F23Element::from(4);
        let three = U384F23Element::from(3);

        assert_eq!(four - three, U384F23Element::from(1))
    }

    #[test]
    fn zero_minus_1_is_order_minus_1() {
        let zero = U384F23Element::from(0);
        let one = U384F23Element::from(1);

        assert_eq!(zero - one, U384F23Element::from((ORDER - 1) as u64))
    }

    #[test]
    fn neg_zero_is_zero() {
        let zero = U384F23Element::from(0);

        assert_eq!(-&zero, zero);
    }

    #[test]
    fn test_random_field_element_from_rng_0() {
        // This seed generates a sample that is less than the modulus;
        let mut rng = <ChaCha20Rng as SeedableRng>::from_seed([1; 32]);
        let mut expected_rng = rng.clone();

        let mut buffer = [0u8; 48];
        let sample_bytes = &mut buffer[40..];
        expected_rng.fill(sample_bytes);

        let mut expected_uint = UnsignedInteger::from_bytes_be(&buffer).unwrap();

        expected_uint.limbs[5] &= 31_u64;

        let expected = FieldElement::new(U384F23::from_base_type(expected_uint));

        let result = U384F23::get_random_field_element_from_rng(&mut rng);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_random_field_element_from_rng_1() {
        // This seed generates a sample that is grater than the modulus;
        let mut rng = <ChaCha20Rng as SeedableRng>::from_seed([5; 32]);
        let mut expected_rng = rng.clone();

        let mut buffer = [0u8; 48];
        let sample_bytes = &mut buffer[40..];
        expected_rng.fill(sample_bytes);

        let mut expected_uint = UnsignedInteger::from_bytes_be(&buffer).unwrap();

        expected_uint.limbs[5] &= 31_u64;

        let expected = FieldElement::new(U384F23::from_base_type(expected_uint));

        let result = U384F23::get_random_field_element_from_rng(&mut rng);

        assert_ne!(result, expected);
    }

    // FP1
    #[derive(Clone, Debug)]
    struct U384ModulusP1;
    impl IsModulus<U384> for U384ModulusP1 {
        const MODULUS: U384 = UnsignedInteger {
            limbs: [
                0,
                0,
                0,
                3450888597,
                5754816256417943771,
                15923941673896418529,
            ],
        };
    }

    type U384FP1 = U384PrimeField<U384ModulusP1>;
    type U384FP1Element = FieldElement<U384FP1>;

    #[test]
    fn montgomery_prime_field_from_bad_hex_errs() {
        assert!(U384FP1Element::from_hex("0xTEST").is_err());
    }

    #[test]
    fn montgomery_prime_field_addition_works_0() {
        let x = U384FP1Element::new(UnsignedInteger::from_hex_unchecked(
            "05ed176deb0e80b4deb7718cdaa075165f149c",
        ));
        let y = U384FP1Element::new(UnsignedInteger::from_hex_unchecked(
            "5f103b0bd4397d4df560eb559f38353f80eeb6",
        ));
        let c = U384FP1Element::new(UnsignedInteger::from_hex_unchecked(
            "64fd5279bf47fe02d4185ce279d8aa55e00352",
        ));
        assert_eq!(x + y, c);
    }

    #[test]
    fn montgomery_prime_field_multiplication_works_0() {
        let x = U384FP1Element::new(UnsignedInteger::from_hex_unchecked(
            "05ed176deb0e80b4deb7718cdaa075165f149c",
        ));
        let y = U384FP1Element::new(UnsignedInteger::from_hex_unchecked(
            "5f103b0bd4397d4df560eb559f38353f80eeb6",
        ));
        let c = U384FP1Element::new(UnsignedInteger::from_hex_unchecked(
            "73d23e8d462060dc23d5c15c00fc432d95621a3c",
        ));
        assert_eq!(x * y, c);
    }

    // FP2
    #[derive(Clone, Debug)]
    struct U384ModulusP2;
    impl IsModulus<U384> for U384ModulusP2 {
        const MODULUS: U384 = UnsignedInteger {
            limbs: [
                18446744073709551615,
                18446744073709551615,
                18446744073709551615,
                18446744073709551615,
                18446744073709551615,
                18446744073709551275,
            ],
        };
    }

    type U384FP2 = U384PrimeField<U384ModulusP2>;
    type U384FP2Element = FieldElement<U384FP2>;

    #[test]
    fn montgomery_prime_field_addition_works_1() {
        let x = U384FP2Element::new(UnsignedInteger::from_hex_unchecked(
            "05ed176deb0e80b4deb7718cdaa075165f149c",
        ));
        let y = U384FP2Element::new(UnsignedInteger::from_hex_unchecked(
            "5f103b0bd4397d4df560eb559f38353f80eeb6",
        ));
        let c = U384FP2Element::new(UnsignedInteger::from_hex_unchecked(
            "64fd5279bf47fe02d4185ce279d8aa55e00352",
        ));
        assert_eq!(x + y, c);
    }

    #[test]
    fn montgomery_prime_field_multiplication_works_1() {
        let x = U384FP2Element::one();
        let y = U384FP2Element::new(UnsignedInteger::from_hex_unchecked(
            "5f103b0bd4397d4df560eb559f38353f80eeb6",
        ));
        assert_eq!(&y * x, y);
    }

    #[test]
    #[cfg(feature = "alloc")]
    fn to_bytes_from_bytes_be_is_the_identity() {
        let x = U384FP2Element::new(UnsignedInteger::from_hex_unchecked(
            "5f103b0bd4397d4df560eb559f38353f80eeb6",
        ));
        assert_eq!(U384FP2Element::from_bytes_be(&x.to_bytes_be()).unwrap(), x);
    }

    #[test]
    #[cfg(feature = "alloc")]
    fn from_bytes_to_bytes_be_is_the_identity_for_one() {
        let bytes = [
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        ];
        assert_eq!(
            U384FP2Element::from_bytes_be(&bytes).unwrap().to_bytes_be(),
            bytes
        );
    }

    #[test]
    #[cfg(feature = "alloc")]
    fn to_bytes_from_bytes_le_is_the_identity() {
        let x = U384FP2Element::new(UnsignedInteger::from_hex_unchecked(
            "5f103b0bd4397d4df560eb559f38353f80eeb6",
        ));
        assert_eq!(U384FP2Element::from_bytes_le(&x.to_bytes_le()).unwrap(), x);
    }

    #[test]
    #[cfg(feature = "alloc")]
    fn from_bytes_to_bytes_le_is_the_identity_for_one() {
        let bytes = [
            1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ];
        assert_eq!(
            U384FP2Element::from_bytes_le(&bytes).unwrap().to_bytes_le(),
            bytes
        );
    }
}

#[cfg(test)]
mod tests_u256_prime_fields {
    use crate::field::element::FieldElement;
    use crate::field::errors::FieldError;
    use crate::field::fields::montgomery_backed_prime_fields::{
        IsModulus, U64PrimeField, U256PrimeField,
    };
    use crate::field::traits::HasDefaultTranscript;
    use crate::field::traits::IsField;
    use crate::field::traits::IsPrimeField;
    #[cfg(feature = "alloc")]
    use crate::traits::ByteConversion;
    use crate::unsigned_integer::element::U256;
    use crate::unsigned_integer::element::{U64, UnsignedInteger};
    use rand::Rng;
    use rand_chacha::{ChaCha20Rng, rand_core::SeedableRng};

    #[derive(Clone, Debug)]
    struct U256Modulus29;
    impl IsModulus<U256> for U256Modulus29 {
        const MODULUS: U256 = UnsignedInteger::from_u64(29);
    }

    type U256F29 = U256PrimeField<U256Modulus29>;
    type U256F29Element = FieldElement<U256F29>;

    #[test]
    fn montgomery_backend_primefield_compute_r2_parameter() {
        let r2: U256 = UnsignedInteger {
            limbs: [0, 0, 0, 24],
        };
        assert_eq!(U256F29::R2, r2);
    }

    #[test]
    fn montgomery_backend_primefield_compute_mu_parameter() {
        // modular multiplicative inverse
        assert_eq!(U256F29::MU, 14630176334321368523);
    }

    #[test]
    fn montgomery_backend_primefield_compute_zero_parameter() {
        let zero: U256 = UnsignedInteger {
            limbs: [0, 0, 0, 0],
        };
        assert_eq!(U256F29::ZERO, zero);
    }

    #[test]
    fn montgomery_backend_primefield_from_u64() {
        // (770*2**(256))%29
        let a: U256 = UnsignedInteger {
            limbs: [0, 0, 0, 24],
        };
        assert_eq!(U256F29::from_u64(770_u64), a);
    }

    #[test]
    fn montgomery_backend_primefield_representative() {
        // 770%29
        let a: U256 = UnsignedInteger {
            limbs: [0, 0, 0, 16],
        };
        assert_eq!(U256F29::representative(&U256F29::from_u64(770_u64)), a);
    }

    #[test]
    fn montgomery_backend_multiplication_works_0() {
        let x = U256F29Element::from(11_u64);
        let y = U256F29Element::from(10_u64);
        let c = U256F29Element::from(110_u64);
        assert_eq!(x * y, c);
    }

    #[test]
    fn doubling() {
        assert_eq!(
            U256F29Element::from(2).double(),
            U256F29Element::from(2) + U256F29Element::from(2),
        );
    }

    const ORDER: usize = 29;
    #[test]
    fn two_plus_one_is_three() {
        assert_eq!(
            U256F29Element::from(2) + U256F29Element::from(1),
            U256F29Element::from(3)
        );
    }

    #[test]
    fn max_order_plus_1_is_0() {
        assert_eq!(
            U256F29Element::from((ORDER - 1) as u64) + U256F29Element::from(1),
            U256F29Element::from(0)
        );
    }

    #[test]
    fn when_comparing_13_and_13_they_are_equal() {
        let a: U256F29Element = U256F29Element::from(13);
        let b: U256F29Element = U256F29Element::from(13);
        assert_eq!(a, b);
    }

    #[test]
    fn when_comparing_13_and_8_they_are_different() {
        let a: U256F29Element = U256F29Element::from(13);
        let b: U256F29Element = U256F29Element::from(8);
        assert_ne!(a, b);
    }

    #[test]
    fn mul_neutral_element() {
        let a: U256F29Element = U256F29Element::from(1);
        let b: U256F29Element = U256F29Element::from(2);
        assert_eq!(a * b, U256F29Element::from(2));
    }

    #[test]
    fn mul_2_3_is_6() {
        let a: U256F29Element = U256F29Element::from(2);
        let b: U256F29Element = U256F29Element::from(3);
        assert_eq!(a * b, U256F29Element::from(6));
    }

    #[test]
    fn mul_order_minus_1() {
        let a: U256F29Element = U256F29Element::from((ORDER - 1) as u64);
        let b: U256F29Element = U256F29Element::from((ORDER - 1) as u64);
        assert_eq!(a * b, U256F29Element::from(1));
    }

    #[test]
    fn inv_0_error() {
        let result = U256F29Element::from(0).inv();
        assert!(matches!(result, Err(FieldError::InvZeroError)));
    }

    #[test]
    fn inv_2() {
        let a: U256F29Element = U256F29Element::from(2);
        assert_eq!(&a * a.inv().unwrap(), U256F29Element::from(1));
    }

    #[test]
    fn pow_2_3() {
        assert_eq!(U256F29Element::from(2).pow(3_u64), U256F29Element::from(8))
    }

    #[test]
    fn pow_p_minus_1() {
        assert_eq!(
            U256F29Element::from(2).pow(ORDER - 1),
            U256F29Element::from(1)
        )
    }

    #[test]
    fn div_1() {
        assert_eq!(
            (U256F29Element::from(2) / U256F29Element::from(1)).unwrap(),
            U256F29Element::from(2)
        )
    }

    #[test]
    fn div_4_2() {
        let a = U256F29Element::from(4);
        let b = U256F29Element::from(2);
        assert_eq!((a / &b).unwrap(), b)
    }

    #[test]
    fn div_4_3() {
        assert_eq!(
            (U256F29Element::from(4) / U256F29Element::from(3)).unwrap() * U256F29Element::from(3),
            U256F29Element::from(4)
        )
    }

    #[test]
    fn two_plus_its_additive_inv_is_0() {
        let two = U256F29Element::from(2);

        assert_eq!(&two + (-&two), U256F29Element::from(0))
    }

    #[test]
    fn four_minus_three_is_1() {
        let four = U256F29Element::from(4);
        let three = U256F29Element::from(3);

        assert_eq!(four - three, U256F29Element::from(1))
    }

    #[test]
    fn zero_minus_1_is_order_minus_1() {
        let zero = U256F29Element::from(0);
        let one = U256F29Element::from(1);

        assert_eq!(zero - one, U256F29Element::from((ORDER - 1) as u64))
    }

    #[test]
    fn neg_zero_is_zero() {
        let zero = U256F29Element::from(0);

        assert_eq!(-&zero, zero);
    }

    #[test]
    fn test_random_field_element_from_rng_0() {
        // This seed generates a sample that is less than the modulus;
        let mut rng = <ChaCha20Rng as SeedableRng>::from_seed([1; 32]);
        let mut expected_rng = rng.clone();

        let mut buffer = [0u8; 48];
        let sample_bytes = &mut buffer[24..32];
        expected_rng.fill(sample_bytes);

        let mut expected_uint = UnsignedInteger::from_bytes_be(&buffer).unwrap();

        expected_uint.limbs[3] &= 31_u64;

        let expected = FieldElement::new(U256F29::from_base_type(expected_uint));

        let result = U256F29::get_random_field_element_from_rng(&mut rng);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_random_field_element_from_rng_1() {
        // This seed generates a sample that is grater than the modulus;
        let mut rng = <ChaCha20Rng as SeedableRng>::from_seed([5; 32]);
        let mut expected_rng = rng.clone();

        let mut buffer = [0u8; 48];
        let sample_bytes = &mut buffer[24..32];
        expected_rng.fill(sample_bytes);

        let mut expected_uint = UnsignedInteger::from_bytes_be(&buffer).unwrap();

        expected_uint.limbs[3] &= 31_u64;

        let expected = FieldElement::new(U256F29::from_base_type(expected_uint));

        let result = U256F29::get_random_field_element_from_rng(&mut rng);

        assert_ne!(result, expected);
    }

    #[derive(Clone, Debug)]
    struct U256ModulusP1;
    impl IsModulus<U256> for U256ModulusP1 {
        const MODULUS: U256 = UnsignedInteger {
            limbs: [
                8366,
                8155137382671976874,
                227688614771682406,
                15723111795979912613,
            ],
        };
    }

    type U256FP1 = U256PrimeField<U256ModulusP1>;
    type U256FP1Element = FieldElement<U256FP1>;

    #[test]
    fn montgomery_prime_field_addition_works_0() {
        let x = U256FP1Element::new(UnsignedInteger::from_hex_unchecked(
            "93e712950bf3fe589aa030562a44b1cec66b09192c4bcf705a5",
        ));
        let y = U256FP1Element::new(UnsignedInteger::from_hex_unchecked(
            "10a712235c1f6b4172a1e35da6aef1a7ec6b09192c4bb88cfa5",
        ));
        let c = U256FP1Element::new(UnsignedInteger::from_hex_unchecked(
            "a48e24b86813699a0d4213b3d0f3a376b2d61232589787fd54a",
        ));
        assert_eq!(x + y, c);
    }

    #[test]
    fn montgomery_prime_field_multiplication_works_0() {
        let x = U256FP1Element::new(UnsignedInteger::from_hex_unchecked(
            "93e712950bf3fe589aa030562a44b1cec66b09192c4bcf705a5",
        ));
        let y = U256FP1Element::new(UnsignedInteger::from_hex_unchecked(
            "10a712235c1f6b4172a1e35da6aef1a7ec6b09192c4bb88cfa5",
        ));
        let c = U256FP1Element::new(UnsignedInteger::from_hex_unchecked(
            "7808e74c3208d9a66791ef9cc15a46acc9951ee312102684021",
        ));
        assert_eq!(x * y, c);
    }

    // FP2
    #[derive(Clone, Debug)]
    struct ModulusP2;
    impl IsModulus<U256> for ModulusP2 {
        const MODULUS: U256 = UnsignedInteger {
            limbs: [
                18446744073709551615,
                18446744073709551615,
                18446744073709551615,
                18446744073709551427,
            ],
        };
    }

    type FP2 = U256PrimeField<ModulusP2>;
    type FP2Element = FieldElement<FP2>;

    #[test]
    fn montgomery_prime_field_addition_works_1() {
        let x = FP2Element::new(UnsignedInteger::from_hex_unchecked(
            "acbbb7ca01c65cfffffc72815b397fff9ab130ad53a5ffffffb8f21b207dfedf",
        ));
        let y = FP2Element::new(UnsignedInteger::from_hex_unchecked(
            "d65ddbe509d3fffff21f494c588cbdbfe43e929b0543e3ffffffffffffffff43",
        ));
        let c = FP2Element::new(UnsignedInteger::from_hex_unchecked(
            "831993af0b9a5cfff21bbbcdb3c63dbf7eefc34858e9e3ffffb8f21b207dfedf",
        ));
        assert_eq!(x + y, c);
    }

    #[test]
    fn montgomery_prime_field_multiplication_works_1() {
        let x = FP2Element::new(UnsignedInteger::from_hex_unchecked(
            "acbbb7ca01c65cfffffc72815b397fff9ab130ad53a5ffffffb8f21b207dfedf",
        ));
        let y = FP2Element::new(UnsignedInteger::from_hex_unchecked(
            "d65ddbe509d3fffff21f494c588cbdbfe43e929b0543e3ffffffffffffffff43",
        ));
        let c = FP2Element::new(UnsignedInteger::from_hex_unchecked(
            "2b1e80d553ecab2e4d41eb53c4c8ad89ebacac6cf6b91dcf2213f311093aa05d",
        ));
        assert_eq!(&y * x, c);
    }

    #[test]
    #[cfg(feature = "alloc")]
    fn to_bytes_from_bytes_be_is_the_identity() {
        let x = FP2Element::new(UnsignedInteger::from_hex_unchecked(
            "5f103b0bd4397d4df560eb559f38353f80eeb6",
        ));
        assert_eq!(FP2Element::from_bytes_be(&x.to_bytes_be()).unwrap(), x);
    }

    #[test]
    #[cfg(feature = "alloc")]
    fn from_bytes_to_bytes_be_is_the_identity_for_one() {
        let bytes = [
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 1,
        ];
        assert_eq!(
            FP2Element::from_bytes_be(&bytes).unwrap().to_bytes_be(),
            bytes
        );
    }

    #[test]
    #[cfg(feature = "alloc")]
    fn to_bytes_from_bytes_le_is_the_identity() {
        let x = FP2Element::new(UnsignedInteger::from_hex_unchecked(
            "5f103b0bd4397d4df560eb559f38353f80eeb6",
        ));
        assert_eq!(FP2Element::from_bytes_le(&x.to_bytes_le()).unwrap(), x);
    }

    #[test]
    #[cfg(feature = "alloc")]
    fn from_bytes_to_bytes_le_is_the_identity_for_one() {
        let bytes = [
            1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0,
        ];
        assert_eq!(
            FP2Element::from_bytes_le(&bytes).unwrap().to_bytes_le(),
            bytes
        );
    }

    #[test]
    #[cfg(feature = "alloc")]
    fn creating_a_field_element_from_its_representative_returns_the_same_element_1() {
        let change = U256::from_u64(1);
        let f1 = U256FP1Element::new(U256ModulusP1::MODULUS + change);
        let f2 = U256FP1Element::new(f1.representative());
        assert_eq!(f1, f2);
    }

    #[test]
    fn creating_a_field_element_from_its_representative_returns_the_same_element_2() {
        let change = U256::from_u64(27);
        let f1 = U256F29Element::new(U256Modulus29::MODULUS + change);
        let f2 = U256F29Element::new(f1.representative());
        assert_eq!(f1, f2);
    }

    #[test]
    fn creating_a_field_element_from_hex_works_1() {
        let a = U256FP1Element::from_hex_unchecked("eb235f6144d9e91f4b14");
        let b = U256FP1Element::new(U256 {
            limbs: [0, 0, 60195, 6872850209053821716],
        });
        assert_eq!(a, b);
    }

    #[test]
    fn creating_a_field_element_from_hex_too_big_errors() {
        let a = U256FP1Element::from_hex(&"f".repeat(65));
        assert!(a.is_err());
        assert_eq!(
            a.unwrap_err(),
            crate::errors::CreationError::HexStringIsTooBig
        )
    }

    #[test]
    fn creating_a_field_element_from_hex_bigger_than_modulus_errors() {
        // A number that consists of 255 1s is bigger than the `U256FP1` modulus
        let a = U256FP1Element::from_hex(&"f".repeat(64));
        assert!(a.is_err());
        assert_eq!(
            a.unwrap_err(),
            crate::errors::CreationError::RepresentativeOutOfRange
        )
    }

    #[test]
    fn creating_a_field_element_from_hex_works_2() {
        let a = U256F29Element::from_hex_unchecked("aa");
        let b = U256F29Element::from(25);
        assert_eq!(a, b);
    }

    #[test]
    fn creating_a_field_element_from_hex_works_3() {
        let a = U256F29Element::from_hex_unchecked("1d");
        let b = U256F29Element::zero();
        assert_eq!(a, b);
    }

    #[cfg(feature = "std")]
    #[test]
    fn to_hex_test_works_1() {
        let a = U256FP1Element::from_hex_unchecked("eb235f6144d9e91f4b14");
        let b = U256FP1Element::new(U256 {
            limbs: [0, 0, 60195, 6872850209053821716],
        });

        assert_eq!(U256FP1Element::to_hex(&a), U256FP1Element::to_hex(&b));
    }

    #[cfg(feature = "std")]
    #[test]
    fn to_hex_test_works_2() {
        let a = U256F29Element::from_hex_unchecked("1d");
        let b = U256F29Element::zero();

        assert_eq!(U256F29Element::to_hex(&a), U256F29Element::to_hex(&b));
    }

    #[cfg(feature = "std")]
    #[test]
    fn to_hex_test_works_3() {
        let a = U256F29Element::from_hex_unchecked("aa");
        let b = U256F29Element::from(25);

        assert_eq!(U256F29Element::to_hex(&a), U256F29Element::to_hex(&b));
    }

    // Goldilocks
    #[derive(Clone, Debug)]
    struct GoldilocksModulus;
    impl IsModulus<U64> for GoldilocksModulus {
        const MODULUS: U64 = UnsignedInteger {
            limbs: [18446744069414584321],
        };
    }

    type GoldilocksField = U64PrimeField<GoldilocksModulus>;
    type GoldilocksElement = FieldElement<GoldilocksField>;

    #[derive(Clone, Debug)]
    struct SecpModulus;
    impl IsModulus<U256> for SecpModulus {
        const MODULUS: U256 = UnsignedInteger::from_hex_unchecked(
            "0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F",
        );
    }
    type SecpMontField = U256PrimeField<SecpModulus>;
    type SecpMontElement = FieldElement<SecpMontField>;

    #[test]
    fn secp256k1_minus_three_pow_2_is_9_with_all_operations() {
        let minus_3 = -SecpMontElement::from_hex_unchecked("0x3");
        let minus_3_mul_minus_3 = &minus_3 * &minus_3;
        let minus_3_squared = minus_3.square();
        let minus_3_pow_2 = minus_3.pow(2_u32);
        let nine = SecpMontElement::from_hex_unchecked("0x9");

        assert_eq!(minus_3_mul_minus_3, nine);
        assert_eq!(minus_3_squared, nine);
        assert_eq!(minus_3_pow_2, nine);
    }

    #[test]
    fn secp256k1_inv_works() {
        let a = SecpMontElement::from_hex_unchecked("0x456");
        let a_inv = a.inv().unwrap();

        assert_eq!(a * a_inv, SecpMontElement::one());
    }

    #[test]
    fn test_cios_overflow_case() {
        let a = GoldilocksElement::from(732582227915286439);
        let b = GoldilocksElement::from(3906369333256140342);
        let expected_sum = GoldilocksElement::from(4638951561171426781);
        assert_eq!(a + b, expected_sum);
    }
}
