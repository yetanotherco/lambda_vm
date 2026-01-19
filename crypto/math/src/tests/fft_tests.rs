#[cfg(test)]
mod fft_helpers_test {
    use crate::fft::cpu::roots_of_unity::get_powers_of_primitive_root;
    use crate::fft::test_helpers::naive_matrix_dft_test;
    use crate::field::element::FieldElement;
    use crate::field::fields::fft_friendly::babybear::Babybear31PrimeField;
    use crate::field::traits::RootsConfig;
    use crate::polynomial::Polynomial;
    use proptest::{collection, prelude::*};

    type F = Babybear31PrimeField;
    type FE = FieldElement<F>;

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
    use crate::fft::polynomial::compose_fft;
    use crate::fft::test_helpers::{
        gen_fft_and_naive_coset_interpolate, gen_fft_and_naive_evaluation,
        gen_fft_and_naive_interpolate, gen_fft_coset_and_naive_evaluation,
        gen_fft_interpolate_and_evaluate,
    };
    use crate::field::element::FieldElement;
    use crate::field::fields::fft_friendly::babybear::Babybear31PrimeField;
    use crate::field::fields::fft_friendly::quartic_babybear::Degree4BabyBearExtensionField;
    use crate::polynomial::Polynomial;
    use proptest::{collection, prelude::*};

    mod babybear_field_tests {
        use super::*;

        type F = Babybear31PrimeField;
        type FE = FieldElement<F>;

        prop_compose! {
            fn powers_of_two(max_exp: u8)(exp in 1..max_exp) -> usize { 1 << exp }
        }
        prop_compose! {
            fn field_element()(num in any::<u64>().prop_filter("Avoid null coefficients", |x| x != &0)) -> FE {
                FE::from(num)
            }
        }
        prop_compose! {
            fn offset()(num in 1..((1u64 << 31) - (1u64 << 27))) -> FE { FE::from(num) }
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
            fn poly(max_exp: u8)(coeffs in field_vec(max_exp)) -> Polynomial<FE> {
                Polynomial::new(&coeffs)
            }
        }
        prop_compose! {
            fn non_zero_poly(max_exp: u8)(coeffs in non_empty_field_vec(max_exp)) -> Polynomial<FE> {
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
            fn test_fft_coset_matches_naive_evaluation(poly in poly(6), offset in offset(), blowup_factor in powers_of_two(4)) {
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
            let p = Polynomial::new(&[FE::from(0u64), FE::from(2u64)]);
            let q = Polynomial::new(&[
                FE::from(0u64),
                FE::from(0u64),
                FE::from(0u64),
                FE::from(1u64),
            ]);
            assert_eq!(
                compose_fft::<F, F>(&p, &q),
                Polynomial::new(&[
                    FE::from(0u64),
                    FE::from(0u64),
                    FE::from(0u64),
                    FE::from(2u64)
                ])
            );
        }
    }

    mod u256_field_tests {
        use super::*;
        use crate::field::fields::fft_friendly::stark_252_prime_field::Stark252PrimeField;

        type F = Stark252PrimeField;
        type FE = FieldElement<F>;

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
            fn non_empty_field_vec(max_exp: u8)(vec in collection::vec(field_element(), 1 << max_exp)) -> Vec<FE> {
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

            #[test]
            fn test_fft_multiplication_works(poly in poly(7), other in poly(7)) {
                prop_assert_eq!(poly.fast_fft_multiplication::<F>(&other).unwrap(), poly * other);
            }

            #[test]
            fn test_fft_division_works(poly in poly(7), other in non_zero_poly(7)) {
                prop_assert_eq!(poly.fast_division::<F>(&other).unwrap(), poly.long_division_with_remainder(&other));
            }

            #[test]
            fn test_invert_polynomial_mod_works(poly in non_zero_poly(7), k in powers_of_two(4)) {
                let inverted_poly = poly.invert_polynomial_mod::<F>(k).unwrap();
                prop_assert_eq!((poly * inverted_poly).truncate(k), Polynomial::new(&[FE::one()]));
            }
        }
    }

    #[test]
    fn test_fft_with_values_in_field_extension_over_domain_in_prime_field() {
        type TF = Babybear31PrimeField;
        type TL = Degree4BabyBearExtensionField;

        let a = FieldElement::<TL>::from(&[
            FieldElement::one(),
            FieldElement::one(),
            FieldElement::zero(),
            FieldElement::zero(),
        ]);
        let b = FieldElement::<TL>::from(&[
            -FieldElement::from(2),
            FieldElement::from(17),
            FieldElement::zero(),
            FieldElement::zero(),
        ]);
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
    use crate::field::fields::fft_friendly::babybear::Babybear31PrimeField;
    use crate::field::traits::RootsConfig;
    use proptest::prelude::*;

    type F = Babybear31PrimeField;

    proptest! {
        #[test]
        fn test_gen_twiddles_bit_reversed_validity(n in 1..8_u64) {
            let twiddles = get_twiddles::<F>(n, RootsConfig::Natural).unwrap();
            let mut twiddles_to_reorder = get_twiddles(n, RootsConfig::BitReverse).unwrap();
            in_place_bit_reverse_permute(&mut twiddles_to_reorder);

            prop_assert_eq!(twiddles, twiddles_to_reorder);
        }
    }

    #[test]
    fn gen_twiddles_with_order_greater_than_63_should_fail() {
        let twiddles = get_twiddles::<F>(64, RootsConfig::Natural);
        assert!(matches!(twiddles, Err(FFTError::OrderError(_))));
    }
}
