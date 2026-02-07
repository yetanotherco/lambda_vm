#[cfg(test)]
mod fft_helpers_test {
    use crate::fft::cpu::roots_of_unity::get_powers_of_primitive_root;
    use crate::fft::test_helpers::naive_matrix_dft_test;
    use crate::field::element::FieldElement;
    use crate::field::test_fields::u64_test_field::U64TestField;
    use crate::field::traits::RootsConfig;
    use crate::polynomial::Polynomial;

    use proptest::{collection, prelude::*};

    type F = U64TestField;
    type FE = FieldElement<F>;

    prop_compose! {
        fn powers_of_two(max_exp: u8)(exp in 1..max_exp) -> usize { 1 << exp }
        // max_exp cannot be multiple of the bits that represent a usize, generally 64 or 32.
        // also it can't exceed the test field's two-adicity.
    }
    prop_compose! {
        fn field_element()(num in any::<u64>().prop_filter("Avoid null coefficients", |x| x != &0)) -> FE {
            FE::from(num)
        }
    }
    prop_compose! {
        fn field_vec(max_exp: u8)(vec in collection::vec(field_element(), 2..1<<max_exp).prop_filter("Avoid polynomials of size not power of two", |vec| vec.len().is_power_of_two())) -> Vec<FE> {
            vec
        }
    }

    proptest! {
        // Property-based test that ensures dft() gives the same result as a naive polynomial evaluation.
        #[test]
        fn test_dft_same_as_eval(coeffs in field_vec(8)) {
            let dft = naive_matrix_dft_test(&coeffs);

            let poly = Polynomial::new(&coeffs);
            let order = coeffs.len().trailing_zeros();
            let twiddles = get_powers_of_primitive_root(order.into(), coeffs.len(), RootsConfig::Natural).unwrap();
            let evals: Vec<FE> = twiddles.iter().map(|x| poly.evaluate(x)).collect();

            prop_assert_eq!(evals, dft);
        }
    }
}

#[cfg(test)]
mod fft_polynomial_tests {
    #[cfg(not(feature = "cuda"))]
    use crate::field::traits::IsField;

    use crate::fft::cpu::roots_of_unity::{
        get_powers_of_primitive_root, get_powers_of_primitive_root_coset,
    };
    use crate::fft::polynomial::compose_fft;
    use crate::field::element::FieldElement;
    use crate::field::extensions_goldilocks::Degree2GoldilocksExtensionField;
    use crate::field::traits::{IsFFTField, RootsConfig};
    use crate::polynomial::Polynomial;
    use proptest::{collection, prelude::*};

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

    #[cfg(not(feature = "cuda"))]
    mod u64_field_tests {
        use super::*;
        use crate::field::test_fields::u64_test_field::U64TestField;

        // FFT related tests
        type F = U64TestField;
        type FE = FieldElement<F>;

        prop_compose! {
            fn powers_of_two(max_exp: u8)(exp in 1..max_exp) -> usize { 1 << exp }
            // max_exp cannot be multiple of the bits that represent a usize, generally 64 or 32.
            // also it can't exceed the test field's two-adicity.
        }
        prop_compose! {
            fn field_element()(num in any::<u64>().prop_filter("Avoid null coefficients", |x| x != &0)) -> FE {
                FE::from(num)
            }
        }
        prop_compose! {
            fn offset()(num in 1..F::neg(&1)) -> FE { FE::from(num) }
        }
        prop_compose! {
            fn field_vec(max_exp: u8)(vec in collection::vec(field_element(), 0..1 << max_exp)) -> Vec<FE> {
                vec
            }
        }
        prop_compose! {
            fn non_empty_field_vec(max_exp: u8)(vec in collection::vec(field_element(), 1 << max_exp)) -> Vec<FE> {
                vec
            }
        }
        prop_compose! {
            fn non_power_of_two_sized_field_vec(max_exp: u8)(vec in collection::vec(field_element(), 2..1<<max_exp).prop_filter("Avoid polynomials of size power of two", |vec| !vec.len().is_power_of_two())) -> Vec<FE> {
                vec
            }
        }
        prop_compose! {
            fn poly(max_exp: u8)(coeffs in field_vec(max_exp)) -> Polynomial<FE> {
                Polynomial::new(&coeffs)
            }
        }
        prop_compose! {
            fn non_zero_poly(max_exp: u8)(coeffs in non_empty_field_vec(max_exp)) -> Polynomial<FE> {
                Polynomial::new(&coeffs)
            }
        }
        prop_compose! {
            fn poly_with_non_power_of_two_coeffs(max_exp: u8)(coeffs in non_power_of_two_sized_field_vec(max_exp)) -> Polynomial<FE> {
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
            fn test_fft_coset_matches_naive_evaluation(poly in poly(6), offset in offset(), blowup_factor in powers_of_two(4)) {
                let (fft_eval, naive_eval) = gen_fft_coset_and_naive_evaluation(poly, offset, blowup_factor);
                prop_assert_eq!(fft_eval, naive_eval);
            }

            // Property-based test that ensures FFT interpolation is the same as naive.
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
            fn test_fft_interpolate_is_inverse_of_evaluate(poly in poly(4)
                                                           .prop_filter("Avoid polynomials of size not power of two",
                                                                        |poly| poly.coeff_len().is_power_of_two())) {
                let (poly, new_poly) = gen_fft_interpolate_and_evaluate(poly);

                prop_assert_eq!(poly, new_poly);
            }

            #[test]
            fn test_fft_multiplication_works(poly in poly(7), other in poly(7)) {
                prop_assert_eq!(poly.fast_fft_multiplication::<F>(&other).unwrap(), poly * other);
            }

            #[test]
            fn test_fft_division_works(poly in non_zero_poly(7), other in non_zero_poly(7)) {
                prop_assert_eq!(poly.fast_division::<F>(&other).unwrap(), poly.long_division_with_remainder(&other));
            }

            #[test]
            fn test_invert_polynomial_mod_works(poly in non_zero_poly(7), k in powers_of_two(4)) {
                let inverted_poly = poly.invert_polynomial_mod::<F>(k).unwrap();
                prop_assert_eq!((poly * inverted_poly).truncate(k), Polynomial::new(&[FE::one()]));
            }
        }

        #[test]
        fn composition_fft_works() {
            let p = Polynomial::new(&[FE::new(0), FE::new(2)]);
            let q = Polynomial::new(&[FE::new(0), FE::new(0), FE::new(0), FE::new(1)]);
            assert_eq!(
                compose_fft::<F, F>(&p, &q),
                Polynomial::new(&[FE::new(0), FE::new(0), FE::new(0), FE::new(2)])
            );
        }
    }

    #[test]
    fn test_fft_with_values_in_field_extension_over_domain_in_prime_field() {
        use crate::field::goldilocks::GoldilocksField;
        type TF = GoldilocksField;
        type TL = Degree2GoldilocksExtensionField;

        let a = FieldElement::<TL>::from(&[FieldElement::one(), FieldElement::one()]);
        let b = FieldElement::<TL>::from(&[-FieldElement::from(2), FieldElement::from(17)]);
        let c = FieldElement::<TL>::one();
        let poly = Polynomial::new(&[a, b, c]);

        let eval = Polynomial::evaluate_offset_fft::<TF>(&poly, 8, Some(4), &FieldElement::from(2))
            .unwrap();
        let new_poly =
            Polynomial::interpolate_offset_fft::<TF>(&eval, &FieldElement::from(2)).unwrap();
        assert_eq!(poly, new_poly);
    }
}

#[cfg(test)]
mod roots_of_unity_tests {
    use crate::fft::cpu::bit_reversing::in_place_bit_reverse_permute;
    use crate::fft::cpu::roots_of_unity::get_twiddles;
    use crate::fft::errors::FFTError;
    use crate::field::test_fields::u64_test_field::U64TestField;
    use crate::field::traits::RootsConfig;
    use proptest::prelude::*;

    type F = U64TestField;

    proptest! {
        #[test]
        fn test_gen_twiddles_bit_reversed_validity(n in 1..8_u64) {
            let twiddles = get_twiddles::<F>(n, RootsConfig::Natural).unwrap();
            let mut twiddles_to_reorder = get_twiddles(n, RootsConfig::BitReverse).unwrap();
            in_place_bit_reverse_permute(&mut twiddles_to_reorder); // so now should be naturally ordered

            prop_assert_eq!(twiddles, twiddles_to_reorder);
        }
    }

    #[test]
    fn gen_twiddles_with_order_greater_than_63_should_fail() {
        let twiddles = get_twiddles::<F>(64, RootsConfig::Natural);

        assert!(matches!(twiddles, Err(FFTError::OrderError(_))));
    }
}
