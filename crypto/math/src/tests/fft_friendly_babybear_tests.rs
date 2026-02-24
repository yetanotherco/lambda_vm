use crate::field::element::FieldElement;
use crate::field::fields::fft_friendly::babybear::Babybear31PrimeField;
use crate::traits::ByteConversion;

#[test]
#[cfg(feature = "alloc")]
fn byte_serialization_for_a_number_matches_with_byte_conversion_implementation_le() {
    let element = FieldElement::<Babybear31PrimeField>::from_hex_unchecked("0123456701234567");
    let bytes = element.to_bytes_le();
    let expected_bytes: [u8; 8] = ByteConversion::to_bytes_le(&element).try_into().unwrap();
    assert_eq!(bytes, expected_bytes);
}

#[test]
#[cfg(feature = "alloc")]
fn byte_serialization_for_a_number_matches_with_byte_conversion_implementation_be() {
    let element = FieldElement::<Babybear31PrimeField>::from_hex_unchecked("0123456701234567");
    let bytes = element.to_bytes_be();
    let expected_bytes: [u8; 8] = ByteConversion::to_bytes_be(&element).try_into().unwrap();
    assert_eq!(bytes, expected_bytes);
}

#[test]
fn byte_serialization_and_deserialization_works_le() {
    let element = FieldElement::<Babybear31PrimeField>::from_hex_unchecked("7654321076543210");
    let bytes = element.to_bytes_le();
    let from_bytes = FieldElement::<Babybear31PrimeField>::from_bytes_le(&bytes).unwrap();
    assert_eq!(element, from_bytes);
}

#[test]
fn byte_serialization_and_deserialization_works_be() {
    let element = FieldElement::<Babybear31PrimeField>::from_hex_unchecked("7654321076543210");
    let bytes = element.to_bytes_be();
    let from_bytes = FieldElement::<Babybear31PrimeField>::from_bytes_be(&bytes).unwrap();
    assert_eq!(element, from_bytes);
}

#[cfg(all(feature = "std", not(feature = "instruments")))]
mod fft_tests {
    use super::*;
    use crate::fft::cpu::roots_of_unity::{
        get_powers_of_primitive_root, get_powers_of_primitive_root_coset,
    };
    use crate::field::traits::{IsFFTField, RootsConfig};
    use crate::polynomial::Polynomial;
    use alloc::vec::Vec;
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

    prop_compose! {
        fn powers_of_two(max_exp: u8)(exp in 1..max_exp) -> usize { 1 << exp }
    }
    prop_compose! {
        fn field_element()(num in any::<u64>().prop_filter("Avoid null coefficients", |x| x != &0)) -> FieldElement<Babybear31PrimeField> {
            FieldElement::<Babybear31PrimeField>::from(num)
        }
    }
    prop_compose! {
        fn offset()(num in any::<u64>(), factor in any::<u64>()) -> FieldElement<Babybear31PrimeField> { FieldElement::<Babybear31PrimeField>::from(num).pow(factor) }
    }
    prop_compose! {
        fn field_vec(max_exp: u8)(vec in collection::vec(field_element(), 0..1 << max_exp)) -> Vec<FieldElement<Babybear31PrimeField>> {
            vec
        }
    }
    prop_compose! {
        fn non_power_of_two_sized_field_vec(max_exp: u8)(vec in collection::vec(field_element(), 2..1<<max_exp).prop_filter("Avoid polynomials of size power of two", |vec| !vec.len().is_power_of_two())) -> Vec<FieldElement<Babybear31PrimeField>> {
            vec
        }
    }
    prop_compose! {
        fn poly(max_exp: u8)(coeffs in field_vec(max_exp)) -> Polynomial<FieldElement<Babybear31PrimeField>> {
            Polynomial::new(&coeffs)
        }
    }
    prop_compose! {
        fn poly_with_non_power_of_two_coeffs(max_exp: u8)(coeffs in non_power_of_two_sized_field_vec(max_exp)) -> Polynomial<FieldElement<Babybear31PrimeField>> {
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
