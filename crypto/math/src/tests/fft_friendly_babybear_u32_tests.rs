use crate::field::fields::fft_friendly::babybear_u32::Babybear31PrimeField;

mod test_babybear_31_ops {
    use super::*;
    use crate::{
        errors::CreationError,
        field::{element::FieldElement, errors::FieldError, traits::IsPrimeField},
        traits::ByteConversion,
    };
    type FE = FieldElement<Babybear31PrimeField>;

    #[test]
    fn two_plus_one_is_three() {
        assert_eq!(FE::from(2) + FE::one(), FE::from(3));
    }

    #[test]
    fn one_minus_two_is_minus_one() {
        assert_eq!(FE::one() - FE::from(2), FE::from(2013265920));
    }

    #[test]
    fn mul_by_zero_is_zero() {
        let zero = FE::zero();
        assert_eq!(FE::from(2) * zero, zero);
    }

    #[test]
    fn neg_zero_is_zero() {
        let zero = FE::zero();
        assert_eq!(-&zero, zero);
    }

    #[test]
    fn doubling() {
        assert_eq!(FE::from(2).double(), FE::from(4));
    }

    const ORDER: usize = 2013265921;

    #[test]
    fn order_is_0() {
        assert_eq!(FE::from((ORDER - 1) as u64) + FE::one(), FE::zero());
    }

    #[test]
    fn when_comparing_13_and_13_they_are_equal() {
        assert_eq!(FE::from(13), FE::from(13));
    }

    #[test]
    fn when_comparing_13_and_8_they_are_different() {
        assert_ne!(FE::from(13), FE::from(8));
    }

    #[test]
    fn mul_neutral_element() {
        assert_eq!(FE::one() * FE::from(2), FE::from(2));
    }

    #[test]
    fn mul_2_3_is_6() {
        assert_eq!(FE::from(2) * FE::from(3), FE::from(6));
    }

    #[test]
    fn mul_order_minus_1() {
        let a = FE::from((ORDER - 1) as u64);
        assert_eq!(a * a, FE::one());
    }

    #[test]
    fn inv_0_error() {
        assert!(matches!(FE::zero().inv(), Err(FieldError::InvZeroError)));
    }

    #[test]
    fn inv_2_mul_2_is_1() {
        let a = FE::from(2);
        assert_eq!(a * a.inv().unwrap(), FE::one());
    }

    #[test]
    fn square_2_is_4() {
        assert_eq!(FE::from(2).square(), FE::from(4));
    }

    #[test]
    fn pow_2_3_is_8() {
        assert_eq!(FE::from(2).pow(3_u64), FE::from(8));
    }

    #[test]
    fn pow_p_minus_1() {
        assert_eq!(FE::from(2).pow(ORDER - 1), FE::one());
    }

    #[test]
    fn div_1() {
        assert_eq!((FE::from(2) / FE::one()).unwrap(), FE::from(2));
    }

    #[test]
    fn div_4_2() {
        assert_eq!((FE::from(4) / FE::from(2)).unwrap(), FE::from(2));
    }

    #[test]
    fn two_plus_its_additive_inv_is_0() {
        let two = FE::from(2);
        assert_eq!(two + (-&two), FE::zero());
    }

    #[test]
    fn four_minus_three_is_1() {
        assert_eq!(FE::from(4) - FE::from(3), FE::one());
    }

    #[test]
    fn zero_minus_1_is_order_minus_1() {
        assert_eq!(FE::zero() - FE::one(), FE::from((ORDER - 1) as u64));
    }

    #[test]
    fn babybear_uses_31_bits() {
        assert_eq!(Babybear31PrimeField::field_bit_size(), 31);
    }

    #[test]
    fn montgomery_backend_prime_field_compute_mu_parameter() {
        assert_eq!(Babybear31PrimeField::MU, 2281701377);
    }

    #[test]
    fn montgomery_backend_prime_field_compute_r2_parameter() {
        assert_eq!(Babybear31PrimeField::R2, 1172168163);
    }

    #[test]
    #[cfg(feature = "alloc")]
    fn from_hex_bigger_than_u64_returns_error() {
        let x = FE::from_hex("5f103b0bd4397d4df560eb559f38353f80eeb6");
        assert!(matches!(x, Err(CreationError::InvalidHexString)));
    }

    #[test]
    #[cfg(feature = "alloc")]
    fn to_bytes_from_bytes_be_is_the_identity() {
        let x = FE::from_hex("5f103b").unwrap();
        assert_eq!(FE::from_bytes_be(&x.to_bytes_be()).unwrap(), x);
    }

    #[test]
    #[cfg(feature = "alloc")]
    fn from_bytes_to_bytes_be_is_the_identity() {
        let bytes = [0, 0, 0, 1];
        assert_eq!(FE::from_bytes_be(&bytes).unwrap().to_bytes_be(), bytes);
    }

    #[test]
    #[cfg(feature = "alloc")]
    fn to_bytes_from_bytes_le_is_the_identity() {
        let x = FE::from_hex("5f103b").unwrap();
        assert_eq!(FE::from_bytes_le(&x.to_bytes_le()).unwrap(), x);
    }

    #[test]
    #[cfg(feature = "alloc")]
    fn from_bytes_to_bytes_le_is_the_identity_4_bytes() {
        let bytes = [1, 0, 0, 0];
        assert_eq!(FE::from_bytes_le(&bytes).unwrap().to_bytes_le(), bytes);
    }

    #[test]
    #[cfg(feature = "alloc")]
    fn byte_serialization_for_a_number_matches_with_byte_conversion_implementation_le() {
        let element = FE::from_hex("0123456701234567").unwrap();
        let bytes = element.to_bytes_le();
        let expected_bytes: [u8; 4] = ByteConversion::to_bytes_le(&element).try_into().unwrap();
        assert_eq!(bytes, expected_bytes);
    }

    #[test]
    #[cfg(feature = "alloc")]
    fn byte_serialization_for_a_number_matches_with_byte_conversion_implementation_be() {
        let element = FE::from_hex("0123456701234567").unwrap();
        let bytes = element.to_bytes_be();
        let expected_bytes: [u8; 4] = ByteConversion::to_bytes_be(&element).try_into().unwrap();
        assert_eq!(bytes, expected_bytes);
    }

    #[test]
    fn byte_serialization_and_deserialization_works_le() {
        let element = FE::from_hex("0x7654321076543210").unwrap();
        let bytes = element.to_bytes_le();
        let from_bytes = FE::from_bytes_le(&bytes).unwrap();
        assert_eq!(element, from_bytes);
    }

    #[test]
    fn byte_serialization_and_deserialization_works_be() {
        let element = FE::from_hex("7654321076543210").unwrap();
        let bytes = element.to_bytes_be();
        let from_bytes = FE::from_bytes_be(&bytes).unwrap();
        assert_eq!(element, from_bytes);
    }
}

#[cfg(feature = "std")]
mod test_babybear_31_fft {
    use super::*;
    use crate::fft::test_helpers::{
        gen_fft_and_naive_coset_interpolate, gen_fft_and_naive_evaluation,
        gen_fft_and_naive_interpolate, gen_fft_coset_and_naive_evaluation,
        gen_fft_interpolate_and_evaluate,
    };
    use crate::field::element::FieldElement;
    use crate::polynomial::Polynomial;
    use proptest::{collection, prelude::*, std_facade::Vec};

    type FE = FieldElement<Babybear31PrimeField>;

    prop_compose! {
        fn powers_of_two(max_exp: u8)(exp in 1..max_exp) -> usize { 1 << exp }
    }
    prop_compose! {
        fn field_element()(num in any::<u64>().prop_filter("Avoid null coefficients", |x| x != &0)) -> FE {
            FE::from(num)
        }
    }
    prop_compose! {
        fn offset()(num in any::<u64>(), factor in any::<u64>()) -> FE { FE::from(num).pow(factor) }
    }
    prop_compose! {
        fn field_vec(max_exp: u8)(vec in collection::vec(field_element(), 0..1 << max_exp)) -> Vec<FE> {
            vec
        }
    }
    prop_compose! {
        fn poly(max_exp: u8)(coeffs in field_vec(max_exp)) -> Polynomial<FE> {
            Polynomial::new(&coeffs)
        }
    }

    proptest! {
        #[test]
        fn test_fft_matches_naive_evaluation(poly in poly(8)) {
            let (fft_eval, naive_eval) = gen_fft_and_naive_evaluation(poly);
            prop_assert_eq!(fft_eval, naive_eval);
        }

        #[test]
        fn test_fft_coset_matches_naive_evaluation(poly in poly(4), offset in offset(), blowup_factor in powers_of_two(4)) {
            let (fft_eval, naive_eval) = gen_fft_coset_and_naive_evaluation(poly, offset, blowup_factor);
            prop_assert_eq!(fft_eval, naive_eval);
        }

        #[test]
        fn test_fft_interpolate_matches_naive(fft_evals in field_vec(4)
                                                       .prop_filter("Avoid polynomials of size not power of two",
                                                                    |evals| evals.len().is_power_of_two())) {
            let (fft_poly, naive_poly) = gen_fft_and_naive_interpolate(&fft_evals);
            prop_assert_eq!(fft_poly, naive_poly);
        }

        #[test]
        fn test_fft_interpolate_coset_matches_naive(offset in offset(), fft_evals in field_vec(4)
                                                       .prop_filter("Avoid polynomials of size not power of two",
                                                                    |evals| evals.len().is_power_of_two())) {
            let (fft_poly, naive_poly) = gen_fft_and_naive_coset_interpolate(&fft_evals, &offset);
            prop_assert_eq!(fft_poly, naive_poly);
        }

        #[test]
        fn test_fft_interpolate_is_inverse_of_evaluate(
            poly in poly(4).prop_filter("Avoid non pows of two", |poly| poly.coeff_len().is_power_of_two())) {
            let (poly, new_poly) = gen_fft_interpolate_and_evaluate(poly);
            prop_assert_eq!(poly, new_poly);
        }
    }
}
