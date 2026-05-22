use crate::field::element::FieldElement;
use crate::field::goldilocks::{
    GOLDILOCKS_PRIME, GoldilocksElement, GoldilocksField, inv_addition_chain,
};
use crate::field::traits::{IsFFTField, IsField, IsPrimeField};
use crate::traits::ByteConversion;

const EPSILON: u64 = 0xFFFF_FFFF;

type F = GoldilocksField;
type FE = FieldElement<F>;

fn fe_from_hex(hex: &str) -> FE {
    let value = u64::from_str_radix(hex, 16).unwrap();
    FE::from(value)
}

#[test]
fn two_adic_primitve_root_of_unity_is_correct() {
    // The primitive root should have order 2^TWO_ADICITY
    let root = F::get_primitive_root_of_unity(F::TWO_ADICITY).unwrap();
    let order = 1u64 << F::TWO_ADICITY;

    // root^(2^TWO_ADICITY) should be 1
    assert_eq!(root.pow(order), FE::one());

    // root^(2^(TWO_ADICITY-1)) should NOT be 1 (it should be -1)
    let half_order = order / 2;
    assert_ne!(root.pow(half_order), FE::one());
}

#[test]
fn primitive_root_of_unity_powers() {
    // Test that we can get roots of unity for various orders
    for order in 1..=16 {
        let root = F::get_primitive_root_of_unity(order).unwrap();
        let n = 1u64 << order;

        // root^n should be 1
        assert_eq!(root.pow(n), FE::one(), "Root of order {} failed", order);

        // root^(n/2) should not be 1 for order > 0
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
    let element = fe_from_hex("0123456701234567");
    let bytes = element.to_bytes_le();
    let expected_bytes: [u8; 8] = ByteConversion::to_bytes_le(&element).try_into().unwrap();
    assert_eq!(bytes, expected_bytes);
}

#[test]
#[cfg(feature = "alloc")]
fn byte_serialization_for_a_number_matches_with_byte_conversion_implementation_be() {
    let element = fe_from_hex("0123456701234567");
    let bytes = element.to_bytes_be();
    let expected_bytes: [u8; 8] = ByteConversion::to_bytes_be(&element).try_into().unwrap();
    assert_eq!(bytes, expected_bytes);
}

#[test]
fn byte_serialization_and_deserialization_works_le() {
    let element = fe_from_hex("7654321076543210");
    let bytes = element.to_bytes_le();
    let from_bytes = FieldElement::<GoldilocksField>::from_bytes_le(&bytes).unwrap();
    assert_eq!(element, from_bytes);
}

#[test]
fn byte_serialization_and_deserialization_works_be() {
    let element = fe_from_hex("7654321076543210");
    let bytes = element.to_bytes_be();
    let from_bytes = FieldElement::<GoldilocksField>::from_bytes_be(&bytes).unwrap();
    assert_eq!(element, from_bytes);
}

#[test]
fn test_add_basic() {
    let a = 5u64;
    let b = 7u64;
    assert_eq!(GoldilocksField::add(&a, &b), 12);
}

#[test]
fn test_add_overflow() {
    let a = GOLDILOCKS_PRIME - 1;
    let b = 2u64;
    let result = GoldilocksField::add(&a, &b);
    assert_eq!(GoldilocksField::canonical(&result), 1);
}

#[test]
fn test_sub_basic() {
    let a = 10u64;
    let b = 3u64;
    assert_eq!(GoldilocksField::sub(&a, &b), 7);
}

#[test]
fn test_sub_underflow() {
    let a = 3u64;
    let b = 10u64;
    let result = GoldilocksField::sub(&a, &b);
    assert_eq!(GoldilocksField::canonical(&result), GOLDILOCKS_PRIME - 7);
}

#[test]
fn test_mul_basic() {
    let a = 5u64;
    let b = 7u64;
    assert_eq!(GoldilocksField::mul(&a, &b), 35);
}

#[test]
fn test_mul_large() {
    let a = 1u64 << 40;
    let b = 1u64 << 40;
    let result = GoldilocksField::mul(&a, &b);
    let expected = ((a as u128 * b as u128) % GOLDILOCKS_PRIME as u128) as u64;
    assert_eq!(GoldilocksField::canonical(&result), expected);
}

#[test]
fn test_inv() {
    let a = 5u64;
    let a_inv = GoldilocksField::inv(&a).unwrap();
    let product = GoldilocksField::mul(&a, &a_inv);
    assert_eq!(GoldilocksField::canonical(&product), 1);
}

#[test]
fn test_inv_larger() {
    let a = 123456789u64;
    let a_inv = GoldilocksField::inv(&a).unwrap();
    let product = GoldilocksField::mul(&a, &a_inv);
    assert_eq!(GoldilocksField::canonical(&product), 1);
}

#[test]
fn test_zero_inv() {
    assert!(GoldilocksField::inv(&0).is_err());
}

#[test]
fn test_neg() {
    let a = 5u64;
    let neg_a = GoldilocksField::neg(&a);
    let sum = GoldilocksField::add(&a, &neg_a);
    assert_eq!(GoldilocksField::canonical(&sum), 0);
}

#[test]
fn test_primitive_root() {
    let root = GoldilocksField::get_primitive_root_of_unity(GoldilocksField::TWO_ADICITY).unwrap();
    let mut result = *root.value();
    for _ in 0..32 {
        result = GoldilocksField::square(&result);
    }
    assert_eq!(GoldilocksField::canonical(&result), 1);
}

#[test]
fn test_inv_addition_chain() {
    for a in [5u64, 123456789, GOLDILOCKS_PRIME - 1, 0xDEADBEEF, 1, 2] {
        let a_inv = inv_addition_chain(a);
        let product = GoldilocksField::mul(&a, &a_inv);
        assert_eq!(
            GoldilocksField::canonical(&product),
            1,
            "Failed for a = {}",
            a
        );
    }
}

#[test]
fn test_square() {
    for a in [5u64, 123456789, GOLDILOCKS_PRIME - 1, 0xDEADBEEF, 1, 2] {
        let sq = GoldilocksField::square(&a);
        let mul = GoldilocksField::mul(&a, &a);
        assert_eq!(
            GoldilocksField::canonical(&sq),
            GoldilocksField::canonical(&mul),
            "Square mismatch for a = {}",
            a
        );
    }
}

#[test]
fn test_from_i64_positive() {
    let fe_from_i64 = GoldilocksElement::from(42i64);
    let fe_from_u64 = GoldilocksElement::from(42u64);
    assert_eq!(fe_from_i64, fe_from_u64);
}

#[test]
fn test_from_i64_zero() {
    let fe = GoldilocksElement::from(0i64);
    assert_eq!(fe, GoldilocksElement::zero());
}

#[test]
fn test_from_i64_negative_one() {
    let fe = GoldilocksElement::from(-1i64);
    let expected = GoldilocksElement::from(GOLDILOCKS_PRIME - 1);
    assert_eq!(fe, expected);
    let one = GoldilocksElement::one();
    assert_eq!(fe + one, GoldilocksElement::zero());
}

#[test]
fn test_from_i64_negative_values() {
    for x in [1i64, 5, 100, 1000, 123456789] {
        let pos = GoldilocksElement::from(x);
        let neg = GoldilocksElement::from(-x);
        assert_eq!(pos + neg, GoldilocksElement::zero(), "Failed for x = {}", x);
    }
}

#[test]
fn test_from_i64_negative_equals_negation() {
    for x in [1i64, 42, 1000, 999999] {
        let from_neg = GoldilocksElement::from(-x);
        let neg_from = -GoldilocksElement::from(x);
        assert_eq!(from_neg, neg_from, "Failed for x = {}", x);
    }
}

#[test]
fn test_from_i64_arithmetic() {
    let five = GoldilocksElement::from(5i64);
    let ten = GoldilocksElement::from(10i64);
    let minus_five = GoldilocksElement::from(-5i64);
    assert_eq!(five - ten, minus_five);
}

#[test]
fn test_from_i64_large_negative() {
    let large_neg = GoldilocksElement::from(-1_000_000_000i64);
    let large_pos = GoldilocksElement::from(1_000_000_000i64);
    assert_eq!(large_neg + large_pos, GoldilocksElement::zero());
    assert_eq!(large_neg, -large_pos);
}

#[test]
fn test_from_i64_min_value() {
    let min_val = GoldilocksElement::from(i64::MIN);
    let expected_val = GOLDILOCKS_PRIME - (1u64 << 63);
    let expected = GoldilocksElement::from(expected_val);
    assert_eq!(min_val, expected);
}

#[test]
fn test_from_i64_max_value() {
    let max_val = GoldilocksElement::from(i64::MAX);
    let expected = GoldilocksElement::from(i64::MAX as u64);
    assert_eq!(max_val, expected);
}

#[cfg(all(feature = "std", not(feature = "instruments")))]
mod fft_tests {
    use super::*;
    use crate::fft::roots_of_unity::{
        get_powers_of_primitive_root, get_powers_of_primitive_root_coset,
    };
    use crate::field::traits::{IsFFTField, RootsConfig};
    use crate::polynomial::Polynomial;
    use alloc::vec::Vec;
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

    /// FFT interpolation round-trip: interpolate the evals back into a
    /// coefficient-form polynomial via FFT, then re-evaluate that polynomial
    /// at every twiddle via Horner (`Polynomial::evaluate`). The two
    /// representations agree iff `interpolate_fft` produced the right
    /// polynomial — an independent check using a different algorithm, with
    /// no naive Lagrange `interpolate` needed.
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

    prop_compose! {
        fn powers_of_two(max_exp: u8)(exp in 1..max_exp) -> usize { 1 << exp }
    }
    prop_compose! {
        fn field_element()(num in any::<u64>().prop_filter("Avoid null coefficients", |x| x != &0)) -> FieldElement<GoldilocksField> {
            FieldElement::<GoldilocksField>::from(num)
        }
    }
    prop_compose! {
        fn offset()(num in any::<u64>(), factor in any::<u64>()) -> FieldElement<GoldilocksField> { FieldElement::<GoldilocksField>::from(num).pow(factor) }
    }
    prop_compose! {
        fn field_vec(max_exp: u8)(vec in collection::vec(field_element(), 0..1 << max_exp)) -> Vec<FieldElement<GoldilocksField>> {
            vec
        }
    }
    prop_compose! {
        fn non_power_of_two_sized_field_vec(max_exp: u8)(vec in collection::vec(field_element(), 2..1<<max_exp).prop_filter("Avoid polynomials of size power of two", |vec| !vec.len().is_power_of_two())) -> Vec<FieldElement<GoldilocksField>> {
            vec
        }
    }
    prop_compose! {
        fn poly(max_exp: u8)(coeffs in field_vec(max_exp)) -> Polynomial<FieldElement<GoldilocksField>> {
            Polynomial::new(&coeffs)
        }
    }
    prop_compose! {
        fn poly_with_non_power_of_two_coeffs(max_exp: u8)(coeffs in non_power_of_two_sized_field_vec(max_exp)) -> Polynomial<FieldElement<GoldilocksField>> {
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
            let (recovered, original) = gen_fft_interpolate_round_trip(&fft_evals);
            prop_assert_eq!(recovered, original);
        }

        #[test]
        fn test_fft_interpolate_coset_matches_naive(offset in offset(), fft_evals in field_vec(4)
                                                       .prop_filter("Avoid polynomials of size not power of two",
                                                                    |evals| evals.len().is_power_of_two())) {
            let (recovered, original) = gen_fft_coset_interpolate_round_trip(&fft_evals, &offset);
            prop_assert_eq!(recovered, original);
        }

        #[test]
        fn test_fft_interpolate_is_inverse_of_evaluate(
            poly in poly(4).prop_filter("Avoid non pows of two", |poly| poly.coeff_len().is_power_of_two())) {
            let (poly, new_poly) = gen_fft_interpolate_and_evaluate(poly);
            prop_assert_eq!(poly, new_poly);
        }
    }
}

// =====================================================
// NON-CANONICAL / EDGE-CASE TESTS
// =====================================================
//
// These test the double-overflow and double-underflow paths in add/sub,
// which are only reachable with non-canonical inputs (values >= p).

/// Reference modular add using u128 to avoid overflow.
/// Canonicalizes inputs first so non-canonical u64 values work correctly.
fn ref_add(a: u64, b: u64) -> u64 {
    let a = (a % GOLDILOCKS_PRIME) as u128;
    let b = (b % GOLDILOCKS_PRIME) as u128;
    ((a + b) % GOLDILOCKS_PRIME as u128) as u64
}

/// Reference modular sub using u128 to avoid underflow.
/// Canonicalizes inputs first so non-canonical u64 values work correctly.
fn ref_sub(a: u64, b: u64) -> u64 {
    let a = (a % GOLDILOCKS_PRIME) as u128;
    let b = (b % GOLDILOCKS_PRIME) as u128;
    let p = GOLDILOCKS_PRIME as u128;
    ((a + p - b) % p) as u64
}

#[test]
fn add_double_overflow_both_max() {
    // a = b = u64::MAX, the most extreme double-overflow case.
    let a = u64::MAX;
    let b = u64::MAX;
    let result = F::add(&a, &b);
    assert_eq!(F::canonical(&result), ref_add(a, b));
}

#[test]
fn add_double_overflow_threshold() {
    // Double overflow fires when a + b >= 2^65 - EPSILON.
    // Minimum case: a = p + 1, b = u64::MAX.
    let a = GOLDILOCKS_PRIME + 1;
    let b = u64::MAX;
    let result = F::add(&a, &b);
    assert_eq!(F::canonical(&result), ref_add(a, b));
}

#[test]
fn add_no_double_overflow_at_boundary() {
    // a = p, b = u64::MAX should NOT trigger double overflow
    // (a + b = 2^65 - EPSILON - 1, one below the threshold).
    let a = GOLDILOCKS_PRIME;
    let b = u64::MAX;
    let result = F::add(&a, &b);
    assert_eq!(F::canonical(&result), ref_add(a, b));
}

#[test]
fn add_non_canonical_symmetric() {
    // Both inputs just above p.
    for offset in 1..=EPSILON {
        let a = GOLDILOCKS_PRIME + offset;
        let b = GOLDILOCKS_PRIME + offset;
        let result = F::add(&a, &b);
        assert_eq!(
            F::canonical(&result),
            ref_add(a, b),
            "failed for offset={}",
            offset
        );
        // Only test a few representative values, not all 2^32-1.
        if offset > 16 {
            break;
        }
    }
}

#[test]
fn sub_double_underflow_extreme() {
    // a = 0, b = u64::MAX: maximum underflow.
    let a = 0u64;
    let b = u64::MAX;
    let result = F::sub(&a, &b);
    assert_eq!(F::canonical(&result), ref_sub(a, b));
}

#[test]
fn sub_double_underflow_small_a_large_b() {
    // a just below EPSILON - 1, b well above p.
    let a = EPSILON - 2; // 2^32 - 3
    let b = u64::MAX;
    let result = F::sub(&a, &b);
    assert_eq!(F::canonical(&result), ref_sub(a, b));
}

#[test]
fn sub_no_double_underflow_at_boundary() {
    // a = EPSILON - 1 = 2^32 - 2, b = u64::MAX.
    // diff1 = EPSILON, diff1 - EPSILON = 0 (no second underflow).
    let a = EPSILON - 1;
    let b = u64::MAX;
    let result = F::sub(&a, &b);
    assert_eq!(F::canonical(&result), ref_sub(a, b));
}

#[test]
fn sub_non_canonical_b() {
    // b values just above p, a = 0.
    for offset in 1..=16u64 {
        let a = 0u64;
        let b = GOLDILOCKS_PRIME + offset;
        let result = F::sub(&a, &b);
        assert_eq!(
            F::canonical(&result),
            ref_sub(a, b),
            "failed for b = p + {}",
            offset
        );
    }
}

#[test]
fn add_sub_roundtrip_non_canonical() {
    // For non-canonical inputs, (a + b) - b should equal a (mod p).
    let test_cases: Vec<(u64, u64)> = vec![
        (u64::MAX, u64::MAX),
        (u64::MAX, GOLDILOCKS_PRIME + 1),
        (u64::MAX - 1, GOLDILOCKS_PRIME + 1),
        (0, u64::MAX),
        (1, u64::MAX - 1),
    ];
    for (a, b) in test_cases {
        let sum = F::add(&a, &b);
        let back = F::sub(&sum, &b);
        assert_eq!(
            F::canonical(&back),
            F::canonical(&a),
            "roundtrip failed for a={:#x}, b={:#x}",
            a,
            b
        );
    }
}

#[test]
fn mul_non_canonical_inputs() {
    // Multiplication should produce correct results even with non-canonical inputs.
    let test_cases: Vec<(u64, u64)> = vec![
        (u64::MAX, u64::MAX),
        (GOLDILOCKS_PRIME, GOLDILOCKS_PRIME),
        (GOLDILOCKS_PRIME + 1, u64::MAX),
        (u64::MAX - 1, 2),
    ];
    let ref_mul = |a: u64, b: u64| ((a as u128 * b as u128) % GOLDILOCKS_PRIME as u128) as u64;
    for (a, b) in test_cases {
        let result = F::mul(&a, &b);
        assert_eq!(
            F::canonical(&result),
            ref_mul(a, b),
            "mul failed for a={:#x}, b={:#x}",
            a,
            b
        );
    }
}
