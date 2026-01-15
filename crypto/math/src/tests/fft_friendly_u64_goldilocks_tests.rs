use crate::field::element::FieldElement;
use crate::field::fields::goldilocks::Goldilocks64Field;
use crate::field::traits::IsFFTField;
use crate::traits::ByteConversion;

type F = Goldilocks64Field;
type FE = FieldElement<F>;

#[test]
fn two_adic_primitve_root_of_unity_is_correct() {
    let root = F::get_primitive_root_of_unity(F::TWO_ADICITY).unwrap();
    let order = 1u64 << F::TWO_ADICITY;

    // root^(2^TWO_ADICITY) should be 1
    assert_eq!(root.pow(order), FE::one());

    // root^(2^(TWO_ADICITY-1)) should NOT be 1 (it should be -1)
    assert_ne!(root.pow(order / 2), FE::one());
}

#[test]
fn primitive_root_of_unity_powers() {
    for order in 1..=16 {
        let root = F::get_primitive_root_of_unity(order).unwrap();
        let n = 1u64 << order;

        assert_eq!(root.pow(n), FE::one(), "Root of order {} failed", order);

        if order > 0 {
            assert_ne!(
                root.pow(n / 2),
                FE::one(),
                "Root of order {} is not primitive",
                order
            );
        }
    }
}

#[test]
#[cfg(feature = "alloc")]
fn byte_serialization_for_a_number_matches_with_byte_conversion_implementation_le() {
    let element = FE::from_hex("0123456701234567").unwrap();
    let bytes = element.to_bytes_le_array();
    let expected_bytes: [u8; 8] = ByteConversion::to_bytes_le(&element).try_into().unwrap();
    assert_eq!(bytes, expected_bytes);
}

#[test]
#[cfg(feature = "alloc")]
fn byte_serialization_for_a_number_matches_with_byte_conversion_implementation_be() {
    let element = FE::from_hex("0123456701234567").unwrap();
    let bytes = element.to_bytes_be_array();
    let expected_bytes: [u8; 8] = ByteConversion::to_bytes_be(&element).try_into().unwrap();
    assert_eq!(bytes, expected_bytes);
}

#[test]
#[cfg(feature = "alloc")]
fn byte_serialization_and_deserialization_works_le() {
    let element = FE::from_hex("7654321076543210").unwrap();
    let bytes = ByteConversion::to_bytes_le(&element);
    let from_bytes: FE = ByteConversion::from_bytes_le(&bytes).unwrap();
    assert_eq!(element, from_bytes);
}

#[test]
#[cfg(feature = "alloc")]
fn byte_serialization_and_deserialization_works_be() {
    let element = FE::from_hex("7654321076543210").unwrap();
    let bytes = ByteConversion::to_bytes_be(&element);
    let from_bytes: FE = ByteConversion::from_bytes_be(&bytes).unwrap();
    assert_eq!(element, from_bytes);
}

#[cfg(feature = "std")]
mod fft_tests {
    use super::*;
    use crate::fft::test_helpers::{
        gen_fft_and_naive_coset_interpolate, gen_fft_and_naive_evaluation,
        gen_fft_and_naive_interpolate, gen_fft_coset_and_naive_evaluation,
        gen_fft_interpolate_and_evaluate,
    };
    use crate::polynomial::Polynomial;
    use alloc::vec::Vec;
    use proptest::{collection, prelude::*};

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
