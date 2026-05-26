#[cfg(test)]
mod fft_helpers_test {
    use crate::fft::roots_of_unity::get_powers_of_primitive_root;
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
    use crate::field::traits::IsField;

    use crate::fft::roots_of_unity::{
        get_powers_of_primitive_root, get_powers_of_primitive_root_coset,
    };
    use crate::field::element::FieldElement;
    use crate::field::extensions_goldilocks::Degree2GoldilocksExtensionField;
    use crate::field::traits::{IsFFTField, RootsConfig};
    use crate::polynomial::Polynomial;
    use crate::polynomial::compose_fft;
    use proptest::{collection, prelude::*};

    /// Evaluates a polynomial at a slice of points
    fn evaluate_slice<F: IsFFTField + Send + Sync>(
        poly: &Polynomial<FieldElement<F>>,
        input: &[FieldElement<F>],
    ) -> Vec<FieldElement<F>> {
        input.iter().map(|x| poly.evaluate(x)).collect()
    }

    fn gen_fft_and_naive_evaluation<F: IsFFTField + Send + Sync>(
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

    fn gen_fft_coset_and_naive_evaluation<F: IsFFTField + Send + Sync>(
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

    /// FFT interpolation round-trip: interpolate `fft_evals` back to a
    /// polynomial via FFT, then re-evaluate at every twiddle via Horner.
    /// `(recovered, original)` agree iff `interpolate_fft` is correct — an
    /// independent check using a different algorithm.
    fn gen_fft_interpolate_round_trip<F: IsFFTField + Send + Sync>(
        fft_evals: &[FieldElement<F>],
    ) -> (Vec<FieldElement<F>>, Vec<FieldElement<F>>) {
        let order = fft_evals.len().trailing_zeros() as u64;
        let twiddles =
            get_powers_of_primitive_root(order, 1 << order, RootsConfig::Natural).unwrap();

        let fft_poly = Polynomial::interpolate_fft::<F>(fft_evals).unwrap();
        let recovered = evaluate_slice(&fft_poly, &twiddles);

        (recovered, fft_evals.to_vec())
    }

    fn gen_fft_coset_interpolate_round_trip<F: IsFFTField + Send + Sync>(
        fft_evals: &[FieldElement<F>],
        offset: &FieldElement<F>,
    ) -> (Vec<FieldElement<F>>, Vec<FieldElement<F>>) {
        let order = fft_evals.len().trailing_zeros() as u64;
        let twiddles = get_powers_of_primitive_root_coset(order, 1 << order, offset).unwrap();

        let fft_poly = Polynomial::interpolate_offset_fft(fft_evals, offset).unwrap();
        let recovered = evaluate_slice(&fft_poly, &twiddles);

        (recovered, fft_evals.to_vec())
    }

    fn gen_fft_interpolate_and_evaluate<F: IsFFTField + Send + Sync>(
        poly: Polynomial<FieldElement<F>>,
    ) -> (Polynomial<FieldElement<F>>, Polynomial<FieldElement<F>>) {
        let eval = Polynomial::evaluate_fft::<F>(&poly, 1, None).unwrap();
        let new_poly = Polynomial::interpolate_fft::<F>(&eval).unwrap();

        (poly, new_poly)
    }

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
                let (recovered, original) = gen_fft_interpolate_round_trip(&fft_evals);
                prop_assert_eq!(recovered, original);
            }

            // Property-based test that ensures FFT interpolation with an offset is the same as naive.
            #[test]
            fn test_fft_interpolate_coset_matches_naive(offset in offset(), fft_evals in field_vec(4)
                                                           .prop_filter("Avoid polynomials of size not power of two",
                                                                        |evals| evals.len().is_power_of_two())) {
                let (recovered, original) = gen_fft_coset_interpolate_round_trip(&fft_evals, &offset);
                prop_assert_eq!(recovered, original);
            }

            // Property-based test that ensures interpolation is the inverse operation of evaluation.
            #[test]
            fn test_fft_interpolate_is_inverse_of_evaluate(poly in poly(4)
                                                           .prop_filter("Avoid polynomials of size not power of two",
                                                                        |poly| poly.coeff_len().is_power_of_two())) {
                let (poly, new_poly) = gen_fft_interpolate_and_evaluate(poly);

                prop_assert_eq!(poly, new_poly);
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
    use crate::fft::bit_reversing::in_place_bit_reverse_permute;
    use crate::fft::roots_of_unity::get_powers_of_primitive_root;
    use crate::field::test_fields::u64_test_field::U64TestField;
    use crate::field::traits::RootsConfig;
    use proptest::prelude::*;

    type F = U64TestField;

    proptest! {
        #[test]
        fn test_gen_twiddles_bit_reversed_validity(n in 1..8_u64) {
            let count = (1 << n) / 2;
            let twiddles = get_powers_of_primitive_root::<F>(n, count, RootsConfig::Natural).unwrap();
            let mut twiddles_to_reorder = get_powers_of_primitive_root::<F>(n, count, RootsConfig::BitReverse).unwrap();
            in_place_bit_reverse_permute(&mut twiddles_to_reorder);
            prop_assert_eq!(twiddles, twiddles_to_reorder);
        }
    }

    #[test]
    fn gen_twiddles_with_order_greater_than_field_adicity_should_fail() {
        // U64TestField has TWO_ADICITY = 32, so order 33 should fail
        let result = get_powers_of_primitive_root::<F>(33, 1, RootsConfig::Natural);
        assert!(result.is_err());
    }
}

#[cfg(all(test, feature = "alloc"))]
mod coset_lde_tests {
    use crate::fft::bowers_fft::LayerTwiddles;
    use crate::field::element::FieldElement;
    use crate::field::goldilocks::GoldilocksField;
    use crate::polynomial::Polynomial;
    use alloc::vec::Vec;

    type F = GoldilocksField;
    type FE = FieldElement<F>;

    #[test]
    fn coset_lde_full_into_matches_coset_lde_full() {
        let offset = FE::from(3u64);
        let blowup_factor = 2;

        for order in 1..=10 {
            let n = 1usize << order;
            let evals: Vec<FE> = (0..n).map(|i| FE::from((i * 7 + 13) as u64)).collect();

            let lde_size = n * blowup_factor;
            let inv_tw = LayerTwiddles::<F>::new_inverse(n.trailing_zeros() as u64).unwrap();
            let fwd_tw = LayerTwiddles::<F>::new(lde_size.trailing_zeros() as u64).unwrap();

            let n_inv = FE::from(n as u64).inv().unwrap();
            let mut weights = Vec::with_capacity(n);
            let mut offset_power = n_inv;
            for _ in 0..n {
                weights.push(offset_power);
                offset_power = &offset_power * &offset;
            }

            let reference = Polynomial::<FE>::coset_lde_full::<F>(
                &evals,
                blowup_factor,
                &weights,
                &inv_tw,
                &fwd_tw,
            )
            .unwrap();

            // Test with pre-allocated buffer
            let mut buffer = Vec::with_capacity(lde_size);
            Polynomial::<FE>::coset_lde_full_into::<F>(
                &evals,
                blowup_factor,
                &weights,
                &inv_tw,
                &fwd_tw,
                &mut buffer,
            )
            .unwrap();

            assert_eq!(reference, buffer, "Mismatch at order {}", order);
        }
    }

    #[test]
    fn coset_lde_full_into_reuses_buffer() {
        let offset = FE::from(5u64);
        let blowup_factor = 2usize;
        let n = 16usize;
        let lde_size = n * blowup_factor;

        let inv_tw = LayerTwiddles::<F>::new_inverse(n.trailing_zeros() as u64).unwrap();
        let fwd_tw = LayerTwiddles::<F>::new(lde_size.trailing_zeros() as u64).unwrap();

        let n_inv = FE::from(n as u64).inv().unwrap();
        let mut weights = Vec::with_capacity(n);
        let mut offset_power = n_inv;
        for _ in 0..n {
            weights.push(offset_power);
            offset_power = &offset_power * &offset;
        }

        // Pre-allocate buffer once, reuse for two different inputs
        let mut buffer = Vec::with_capacity(lde_size);

        for seed in [13u64, 42u64] {
            let evals: Vec<FE> = (0..n).map(|i| FE::from(i as u64 * seed + 1)).collect();

            let reference = Polynomial::<FE>::coset_lde_full::<F>(
                &evals,
                blowup_factor,
                &weights,
                &inv_tw,
                &fwd_tw,
            )
            .unwrap();

            Polynomial::<FE>::coset_lde_full_into::<F>(
                &evals,
                blowup_factor,
                &weights,
                &inv_tw,
                &fwd_tw,
                &mut buffer,
            )
            .unwrap();

            assert_eq!(reference, buffer, "Mismatch for seed {}", seed);
        }
    }
}
