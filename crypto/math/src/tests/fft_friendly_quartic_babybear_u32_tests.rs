use crate::field::{
    element::FieldElement,
    fields::fft_friendly::{
        babybear_u32::Babybear31PrimeField, quartic_babybear_u32::Degree4BabyBearU32ExtensionField,
    },
    traits::IsFFTField,
};
use crate::traits::ByteConversion;

type FpE = FieldElement<Babybear31PrimeField>;
type Fp4E = FieldElement<Degree4BabyBearU32ExtensionField>;

#[test]
fn test_add() {
    let a = Fp4E::new([FpE::from(0), FpE::from(1), FpE::from(2), FpE::from(3)]);
    let b = Fp4E::new([-FpE::from(2), FpE::from(4), FpE::from(6), -FpE::from(8)]);
    let expected_result = Fp4E::new([
        FpE::from(0) - FpE::from(2),
        FpE::from(1) + FpE::from(4),
        FpE::from(2) + FpE::from(6),
        FpE::from(3) - FpE::from(8),
    ]);
    assert_eq!(a + b, expected_result);
}

#[test]
fn test_sub() {
    let a = Fp4E::new([FpE::from(0), FpE::from(1), FpE::from(2), FpE::from(3)]);
    let b = Fp4E::new([-FpE::from(2), FpE::from(4), FpE::from(6), -FpE::from(8)]);
    let expected_result = Fp4E::new([
        FpE::from(0) + FpE::from(2),
        FpE::from(1) - FpE::from(4),
        FpE::from(2) - FpE::from(6),
        FpE::from(3) + FpE::from(8),
    ]);
    assert_eq!(a - b, expected_result);
}

#[test]
fn test_mul_by_0() {
    let a = Fp4E::new([FpE::from(4), FpE::from(1), FpE::from(2), FpE::from(3)]);
    let b = Fp4E::zero();
    assert_eq!(a * b, b);
}

#[test]
fn test_mul_by_1() {
    let a = Fp4E::new([FpE::from(4), FpE::from(1), FpE::from(2), FpE::from(3)]);
    let b = Fp4E::one();
    assert_eq!(a * b, a);
}

#[test]
fn test_mul() {
    let a = Fp4E::new([FpE::from(0), FpE::from(1), FpE::from(2), FpE::from(3)]);
    let b = Fp4E::new([FpE::from(2), FpE::from(4), FpE::from(6), FpE::from(8)]);
    let expected_result = Fp4E::new([
        -FpE::from(352),
        -FpE::from(372),
        -FpE::from(256),
        FpE::from(20),
    ]);
    assert_eq!(a * b, expected_result);
}

#[test]
fn test_pow() {
    let a = Fp4E::new([FpE::from(0), FpE::from(1), FpE::from(2), FpE::from(3)]);
    let expected_result = a * a * a;
    assert_eq!(a.pow(3u64), expected_result);
}

#[test]
fn test_inv_of_one_is_one() {
    let a = Fp4E::one();
    assert_eq!(a.inv().unwrap(), a);
}

#[test]
fn test_inv_of_zero_error() {
    assert!(Fp4E::zero().inv().is_err());
}

#[test]
fn test_mul_by_inv_is_identity() {
    let a = Fp4E::from(123456);
    assert_eq!(a * a.inv().unwrap(), Fp4E::one());
}

#[test]
fn test_mul_as_subfield() {
    let a = FpE::from(2);
    let b = Fp4E::new([FpE::from(2), FpE::from(4), FpE::from(6), FpE::from(8)]);
    let expected_result = Fp4E::new([FpE::from(4), FpE::from(8), FpE::from(12), FpE::from(16)]);
    assert_eq!(a * b, expected_result);
}

#[test]
fn test_double_equals_sum_two_times() {
    let a = Fp4E::new([FpE::from(2), FpE::from(4), FpE::from(6), FpE::from(8)]);
    assert_eq!(a.double(), a + a);
}

#[test]
fn test_mul_group_generator_pow_order_is_one() {
    let generator = Fp4E::new([FpE::from(8), FpE::from(1), FpE::zero(), FpE::zero()]);
    let extension_order: u128 = 2013265921_u128.pow(4);
    assert_eq!(generator.pow(extension_order), generator);
}

#[test]
fn test_two_adic_primitve_root_of_unity() {
    let generator = Fp4E::new(Degree4BabyBearU32ExtensionField::TWO_ADIC_PRIMITVE_ROOT_OF_UNITY);
    assert_eq!(
        generator.pow(2u64.pow(Degree4BabyBearU32ExtensionField::TWO_ADICITY as u32)),
        Fp4E::one()
    );
}

#[test]
#[cfg(feature = "alloc")]
fn to_bytes_from_bytes_be_is_the_identity() {
    let x = Fp4E::new([FpE::from(2), FpE::from(4), FpE::from(6), FpE::from(8)]);
    assert_eq!(Fp4E::from_bytes_be(&x.to_bytes_be()).unwrap(), x);
}

#[test]
#[cfg(feature = "alloc")]
fn from_bytes_to_bytes_be_is_the_identity() {
    let bytes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
    assert_eq!(Fp4E::from_bytes_be(&bytes).unwrap().to_bytes_be(), bytes);
}

#[test]
#[cfg(feature = "alloc")]
fn to_bytes_from_bytes_le_is_the_identity() {
    let x = Fp4E::new([FpE::from(2), FpE::from(4), FpE::from(6), FpE::from(8)]);
    assert_eq!(Fp4E::from_bytes_le(&x.to_bytes_le()).unwrap(), x);
}

#[test]
#[cfg(feature = "alloc")]
fn from_bytes_to_bytes_le_is_the_identity() {
    let bytes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
    assert_eq!(Fp4E::from_bytes_le(&bytes).unwrap().to_bytes_le(), bytes);
}

#[cfg(feature = "std")]
mod test_babybear_31_fft {
    use super::*;
    use crate::fft::test_helpers::{
        gen_fft_and_naive_coset_interpolate, gen_fft_and_naive_evaluation,
        gen_fft_and_naive_interpolate, gen_fft_coset_and_naive_evaluation,
        gen_fft_interpolate_and_evaluate,
    };
    use crate::polynomial::Polynomial;
    use proptest::{collection, prelude::*, std_facade::Vec};

    prop_compose! {
        fn powers_of_two(max_exp: u8)(exp in 1..max_exp) -> usize { 1 << exp }
    }
    prop_compose! {
        fn field_element()(coeffs in [any::<u64>(); 4]) -> Fp4E {
            Fp4E::new([
                FpE::from(coeffs[0]),
                FpE::from(coeffs[1]),
                FpE::from(coeffs[2]),
                FpE::from(coeffs[3]),
            ])
        }
    }
    prop_compose! {
        fn offset()(num in field_element(), factor in any::<u64>()) -> Fp4E { num.pow(factor) }
    }
    prop_compose! {
        fn field_vec(max_exp: u8)(vec in collection::vec(field_element(), 0..1 << max_exp)) -> Vec<Fp4E> {
            vec
        }
    }
    prop_compose! {
        fn poly(max_exp: u8)(coeffs in field_vec(max_exp)) -> Polynomial<Fp4E> {
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
