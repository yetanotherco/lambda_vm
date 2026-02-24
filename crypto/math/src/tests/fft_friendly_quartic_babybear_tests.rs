use crate::field::{
    element::FieldElement,
    fields::fft_friendly::{
        babybear::Babybear31PrimeField, quartic_babybear::Degree4BabyBearExtensionField,
    },
    traits::IsFFTField,
};

type FpE = FieldElement<Babybear31PrimeField>;
type Fp4E = FieldElement<Degree4BabyBearExtensionField>;

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
    let b = Fp4E::new([FpE::zero(), FpE::zero(), FpE::zero(), FpE::zero()]);
    assert_eq!(&a * &b, b);
}

#[test]
fn test_mul_by_1() {
    let a = Fp4E::new([FpE::from(4), FpE::from(1), FpE::from(2), FpE::from(3)]);
    let b = Fp4E::new([FpE::one(), FpE::zero(), FpE::zero(), FpE::zero()]);
    assert_eq!(&a * b, a);
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
    let expected_result = &a * &a * &a;
    assert_eq!(a.pow(3u64), expected_result);
}

#[test]
fn test_inv_of_one_is_one() {
    let a = Fp4E::one();
    assert_eq!(a.inv().unwrap(), a);
}

#[test]
fn test_inv_of_zero_error() {
    let result = Fp4E::zero().inv();
    assert!(result.is_err());
}

#[test]
fn test_mul_by_inv_is_identity() {
    let a = Fp4E::from(123456);
    assert_eq!(&a * a.inv().unwrap(), Fp4E::one());
}

#[test]
fn test_mul_as_subfield() {
    let a = FpE::from(2);
    let b = Fp4E::new([FpE::from(2), FpE::from(4), FpE::from(6), FpE::from(8)]);
    let expected_result = Fp4E::new([
        FpE::from(2) * FpE::from(2),
        FpE::from(4) * FpE::from(2),
        FpE::from(6) * FpE::from(2),
        FpE::from(8) * FpE::from(2),
    ]);
    assert_eq!(a * b, expected_result);
}

#[test]
fn test_double_equals_sum_two_times() {
    let a = Fp4E::new([FpE::from(2), FpE::from(4), FpE::from(6), FpE::from(8)]);

    assert_eq!(a.double(), &a + &a);
}

#[test]
fn test_mul_group_generator_pow_order_is_one() {
    let generator = Fp4E::new([FpE::from(8), FpE::from(1), FpE::zero(), FpE::zero()]);
    let extension_order: u128 = 2013265921_u128.pow(4);
    assert_eq!(generator.pow(extension_order), generator);
}

#[test]
fn test_two_adic_primitve_root_of_unity() {
    let generator = Fp4E::new(Degree4BabyBearExtensionField::TWO_ADIC_PRIMITVE_ROOT_OF_UNITY);
    assert_eq!(
        generator.pow(2u64.pow(Degree4BabyBearExtensionField::TWO_ADICITY as u32)),
        Fp4E::one()
    );
}

#[cfg(all(feature = "std", not(feature = "instruments")))]
mod test_babybear_31_fft {
    use super::*;
    use crate::fft::cpu::roots_of_unity::{
        get_powers_of_primitive_root, get_powers_of_primitive_root_coset,
    };
    use crate::field::element::FieldElement;
    use crate::field::traits::{IsFFTField, RootsConfig};
    use crate::polynomial::Polynomial;
    use proptest::{collection, prelude::*, std_facade::Vec};

    /// Evaluates a polynomial at a slice of points
    fn evaluate_slice<F: IsFFTField>(
        poly: &Polynomial<FieldElement<F>>,
        input: &[FieldElement<F>],
    ) -> Vec<FieldElement<F>> {
        input.iter().map(|x| poly.evaluate(x)).collect()
    }

    fn gen_fft_and_naive_evaluation<F: IsFFTField>(
        poly: Polynomial<FieldElement<F>>,
    ) -> (Vec<FieldElement<F>>, Vec<FieldElement<F>>) {
        let len = poly.coeff_len().next_power_of_two();
        let order = len.trailing_zeros();
        let twiddles =
            get_powers_of_primitive_root(order.into(), len, RootsConfig::Natural).unwrap();

        let fft_eval = Polynomial::evaluate_fft::<F>(&poly, 1, None).unwrap();
        let naive_eval = evaluate_slice(&poly, &twiddles);

        (fft_eval, naive_eval)
    }

    fn gen_fft_coset_and_naive_evaluation<F: IsFFTField>(
        poly: Polynomial<FieldElement<F>>,
        offset: FieldElement<F>,
        blowup_factor: usize,
    ) -> (Vec<FieldElement<F>>, Vec<FieldElement<F>>) {
        let len = poly.coeff_len().next_power_of_two();
        let order = (len * blowup_factor).trailing_zeros();
        let twiddles =
            get_powers_of_primitive_root_coset(order.into(), len * blowup_factor, &offset).unwrap();

        let fft_eval =
            Polynomial::evaluate_offset_fft::<F>(&poly, blowup_factor, None, &offset).unwrap();
        let naive_eval = evaluate_slice(&poly, &twiddles);

        (fft_eval, naive_eval)
    }

    fn gen_fft_and_naive_interpolate<F: IsFFTField>(
        fft_evals: &[FieldElement<F>],
    ) -> (Polynomial<FieldElement<F>>, Polynomial<FieldElement<F>>) {
        let order = fft_evals.len().trailing_zeros() as u64;
        let twiddles =
            get_powers_of_primitive_root(order, 1 << order, RootsConfig::Natural).unwrap();

        let naive_poly = Polynomial::interpolate(&twiddles, fft_evals).unwrap();
        let fft_poly = Polynomial::interpolate_fft::<F>(fft_evals).unwrap();

        (fft_poly, naive_poly)
    }

    fn gen_fft_and_naive_coset_interpolate<F: IsFFTField>(
        fft_evals: &[FieldElement<F>],
        offset: &FieldElement<F>,
    ) -> (Polynomial<FieldElement<F>>, Polynomial<FieldElement<F>>) {
        let order = fft_evals.len().trailing_zeros() as u64;
        let twiddles = get_powers_of_primitive_root_coset(order, 1 << order, offset).unwrap();

        let naive_poly = Polynomial::interpolate(&twiddles, fft_evals).unwrap();
        let fft_poly = Polynomial::interpolate_offset_fft(fft_evals, offset).unwrap();

        (fft_poly, naive_poly)
    }

    fn gen_fft_interpolate_and_evaluate<F: IsFFTField>(
        poly: Polynomial<FieldElement<F>>,
    ) -> (Polynomial<FieldElement<F>>, Polynomial<FieldElement<F>>) {
        let eval = Polynomial::evaluate_fft::<F>(&poly, 1, None).unwrap();
        let new_poly = Polynomial::interpolate_fft::<F>(&eval).unwrap();

        (poly, new_poly)
    }

    prop_compose! {
        fn powers_of_two(max_exp: u8)(exp in 1..max_exp) -> usize { 1 << exp }
        // max_exp cannot be multiple of the bits that represent a usize, generally 64 or 32.
        // also it can't exceed the test field's two-adicity.
    }
    prop_compose! {
        fn field_element()(coeffs in [any::<u64>(); 4]) -> Fp4E {
            Fp4E::new([
                FpE::from(coeffs[0]),
                FpE::from(coeffs[1]),
                FpE::from(coeffs[2]),
                FpE::from(coeffs[3])]
            )
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
        fn non_power_of_two_sized_field_vec(max_exp: u8)(vec in collection::vec(field_element(), 2..1<<max_exp).prop_filter("Avoid polynomials of size power of two", |vec| !vec.len().is_power_of_two())) -> Vec<Fp4E> {
            vec
        }
    }
    prop_compose! {
        fn poly(max_exp: u8)(coeffs in field_vec(max_exp)) -> Polynomial<Fp4E> {
            Polynomial::new(&coeffs)
        }
    }
    prop_compose! {
        fn poly_with_non_power_of_two_coeffs(max_exp: u8)(coeffs in non_power_of_two_sized_field_vec(max_exp)) -> Polynomial<Fp4E> {
            Polynomial::new(&coeffs)
        }
    }

    proptest! {
        // Property-based test that ensures FFT eval. gives same result as a naive polynomial evaluation.
        #[test]
        fn test_fft_matches_naive_evaluation(poly in poly(8)) {
            let (fft_eval, naive_eval) = gen_fft_and_naive_evaluation(poly);
            prop_assert_eq!(fft_eval, naive_eval);
        }

        // Property-based test that ensures FFT eval. with coset gives same result as a naive polynomial evaluation.
        #[test]
        fn test_fft_coset_matches_naive_evaluation(poly in poly(4), offset in offset(), blowup_factor in powers_of_two(4)) {
            let (fft_eval, naive_eval) = gen_fft_coset_and_naive_evaluation(poly, offset, blowup_factor);
            prop_assert_eq!(fft_eval, naive_eval);
        }

        // Property-based test that ensures FFT interpolation is the same as naive..
        #[test]
        fn test_fft_interpolate_matches_naive(fft_evals in field_vec(4)
                                                       .prop_filter("Avoid polynomials of size not power of two",
                                                                    |evals| evals.len().is_power_of_two())) {
            let (fft_poly, naive_poly) = gen_fft_and_naive_interpolate(&fft_evals);
            prop_assert_eq!(fft_poly, naive_poly);
        }

        // Property-based test that ensures FFT interpolation with an offset is the same as naive.
        #[test]
        fn test_fft_interpolate_coset_matches_naive(offset in offset(), fft_evals in field_vec(4)
                                                       .prop_filter("Avoid polynomials of size not power of two",
                                                                    |evals| evals.len().is_power_of_two())) {
            let (fft_poly, naive_poly) = gen_fft_and_naive_coset_interpolate(&fft_evals, &offset);
            prop_assert_eq!(fft_poly, naive_poly);
        }

        // Property-based test that ensures interpolation is the inverse operation of evaluation.
        #[test]
        fn test_fft_interpolate_is_inverse_of_evaluate(
            poly in poly(4).prop_filter("Avoid non pows of two", |poly| poly.coeff_len().is_power_of_two())) {
            let (poly, new_poly) = gen_fft_interpolate_and_evaluate(poly);
            prop_assert_eq!(poly, new_poly);
        }
    }
}
