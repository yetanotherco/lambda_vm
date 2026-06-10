//! Tests for Bowers FFT implementation
//!
//! Separated from the main implementation to keep the library code concise.

use crate::fft::bit_reversing::in_place_bit_reverse_permute;
use crate::fft::bowers_fft::*;
use crate::field::element::FieldElement;
use crate::field::goldilocks::GoldilocksField;
use crate::field::traits::IsFFTField;
use alloc::vec;
use alloc::vec::Vec;
use proptest::{collection, prelude::*};

type F = GoldilocksField;
type FE = FieldElement<F>;

/// Naive O(n²) DFT for correctness verification
pub fn naive_dft(input: &[FE]) -> Vec<FE> {
    let n = input.len();
    if n == 0 {
        return Vec::new();
    }
    let root = F::get_primitive_root_of_unity(n.trailing_zeros() as u64).unwrap();
    let mut result = vec![FE::zero(); n];

    for (k, res) in result.iter_mut().enumerate() {
        for (j, inp) in input.iter().enumerate() {
            *res = &*res + &(inp * &root.pow((j * k) as u64));
        }
    }

    result
}

// =========================================================================
// LayerTwiddles tests
// =========================================================================

#[test]
fn test_layer_twiddles_creation() {
    let order = 4u64;
    let layer_twiddles = LayerTwiddles::<F>::new(order).unwrap();

    assert_eq!(layer_twiddles.layers.len(), 4);
    assert_eq!(layer_twiddles.layers[0].len(), 8);
    assert_eq!(layer_twiddles.layers[1].len(), 4);
    assert_eq!(layer_twiddles.layers[2].len(), 2);
    assert_eq!(layer_twiddles.layers[3].len(), 1);

    // First twiddle of each layer should be 1
    for layer in &layer_twiddles.layers {
        assert_eq!(layer[0], FE::one());
    }
}

#[test]
fn test_layer_twiddles_overflow_protection() {
    // Order 64 would overflow on 64-bit systems
    let result = LayerTwiddles::<F>::new(64);
    assert!(result.is_none());
}

#[test]
#[should_panic(expected = "Layer index out of bounds")]
fn test_layer_twiddles_get_layer_out_of_bounds() {
    let layer_twiddles = LayerTwiddles::<F>::new(4).unwrap();
    let _ = layer_twiddles.get_layer(10); // Should panic
}

// =========================================================================
// Bowers FFT tests
// =========================================================================

#[test]
fn test_bowers_fft_opt_fused_small() {
    let input: Vec<FE> = (0..4).map(|i| FE::from(i as u64)).collect();
    let expected = naive_dft(&input);

    let layer_twiddles = LayerTwiddles::<F>::new(2).unwrap();
    let mut result = input.clone();
    bowers_fft_opt_fused(&mut result, &layer_twiddles).unwrap();
    in_place_bit_reverse_permute(&mut result);

    assert_eq!(result, expected);
}

#[test]
fn test_bowers_fft_opt_fused_medium() {
    let input: Vec<FE> = (0..16).map(|i| FE::from(i as u64)).collect();
    let expected = naive_dft(&input);

    let layer_twiddles = LayerTwiddles::<F>::new(4).unwrap();
    let mut result = input.clone();
    bowers_fft_opt_fused(&mut result, &layer_twiddles).unwrap();
    in_place_bit_reverse_permute(&mut result);

    assert_eq!(result, expected);
}

#[test]
fn test_bowers_fft_opt_fused_large() {
    let input: Vec<FE> = (0..256).map(|i| FE::from(i as u64)).collect();
    let expected = naive_dft(&input);

    let layer_twiddles = LayerTwiddles::<F>::new(8).unwrap();
    let mut result = input.clone();
    bowers_fft_opt_fused(&mut result, &layer_twiddles).unwrap();
    in_place_bit_reverse_permute(&mut result);

    assert_eq!(result, expected);
}

#[test]
fn test_bowers_fft_size_one() {
    let input: Vec<FE> = vec![FE::from(42u64)];
    let layer_twiddles = LayerTwiddles::<F>::new(0).unwrap();
    let mut result = input.clone();
    bowers_fft_opt_fused(&mut result, &layer_twiddles).unwrap();
    assert_eq!(result, input);
}

#[test]
fn test_bowers_fft_size_two() {
    let input: Vec<FE> = vec![FE::from(1u64), FE::from(2u64)];
    let expected = naive_dft(&input);

    let layer_twiddles = LayerTwiddles::<F>::new(1).unwrap();
    let mut result = input.clone();
    bowers_fft_opt_fused(&mut result, &layer_twiddles).unwrap();
    in_place_bit_reverse_permute(&mut result);

    assert_eq!(result, expected);
}

#[test]
fn test_bowers_fft_non_power_of_two() {
    use crate::fft::errors::FFTError;
    let input: Vec<FE> = (0..7).map(|i| FE::from(i as u64)).collect();
    let layer_twiddles = LayerTwiddles::<F>::new(3).unwrap();
    let mut result = input;
    let err = bowers_fft_opt_fused(&mut result, &layer_twiddles);
    assert!(matches!(err, Err(FFTError::InputError(7))));
}

#[test]
fn test_bowers_fft_twiddle_size_mismatch() {
    use crate::fft::errors::FFTError;
    // Input of size 16 (order 4) with twiddles for order 3
    let mut input: Vec<FE> = (0..16).map(|i| FE::from(i as u64)).collect();
    let wrong_twiddles = LayerTwiddles::<F>::new(3).unwrap();
    let err = bowers_fft_opt_fused(&mut input, &wrong_twiddles);
    assert!(matches!(err, Err(FFTError::InputError(16))));
}

#[test]
fn test_bowers_ifft_twiddle_size_mismatch() {
    use crate::fft::errors::FFTError;
    // Input of size 16 (order 4) with inverse twiddles for order 3
    let mut input: Vec<FE> = (0..16).map(|i| FE::from(i as u64)).collect();
    let wrong_twiddles = LayerTwiddles::<F>::new_inverse(3).unwrap();
    let err = bowers_ifft_opt(&mut input, &wrong_twiddles);
    assert!(matches!(err, Err(FFTError::InputError(16))));
}

// =========================================================================
// IFFT tests
// =========================================================================

/// Naive O(n²) inverse DFT for correctness verification
pub fn naive_idft(input: &[FE]) -> Vec<FE> {
    let n = input.len();
    if n == 0 {
        return Vec::new();
    }
    let root = F::get_primitive_root_of_unity(n.trailing_zeros() as u64).unwrap();
    let inv_root = root.inv().unwrap();
    let n_inv = FE::from(n as u64).inv().unwrap();
    let mut result = vec![FE::zero(); n];

    for (k, res) in result.iter_mut().enumerate() {
        for (j, inp) in input.iter().enumerate() {
            *res = &*res + &(inp * &inv_root.pow((j * k) as u64));
        }
        // Scale by 1/n
        *res = &*res * &n_inv;
    }

    result
}

#[test]
fn test_bowers_ifft_basic() {
    // FFT of [1, 2, 3, 4] then IFFT should give back [1, 2, 3, 4]
    let input: Vec<FE> = (1..=4).map(|i| FE::from(i as u64)).collect();
    let order = 2u64;
    let n = input.len();

    // Forward twiddles for FFT
    let fwd_twiddles = LayerTwiddles::<F>::new(order).unwrap();
    // Inverse twiddles for IFFT
    let inv_twiddles = LayerTwiddles::<F>::new_inverse(order).unwrap();

    // FFT
    let mut fft_result = input.clone();
    bowers_fft_opt_fused(&mut fft_result, &fwd_twiddles).unwrap();
    in_place_bit_reverse_permute(&mut fft_result);

    // IFFT (bit-reverse first, then IFFT)
    in_place_bit_reverse_permute(&mut fft_result);
    bowers_ifft_opt(&mut fft_result, &inv_twiddles).unwrap();

    // Scale by 1/n
    let n_inv = FE::from(n as u64).inv().unwrap();
    for val in fft_result.iter_mut() {
        *val = &*val * &n_inv;
    }

    assert_eq!(fft_result, input);
}

#[test]
fn test_bowers_fft_ifft_roundtrip_small() {
    for order in 1..=4u64 {
        let n = 1 << order;
        let input: Vec<FE> = (0..n).map(|i| FE::from(i as u64)).collect();

        let fwd_twiddles = LayerTwiddles::<F>::new(order).unwrap();
        let inv_twiddles = LayerTwiddles::<F>::new_inverse(order).unwrap();

        // FFT
        let mut result = input.clone();
        bowers_fft_opt_fused(&mut result, &fwd_twiddles).unwrap();
        in_place_bit_reverse_permute(&mut result);

        // IFFT
        in_place_bit_reverse_permute(&mut result);
        bowers_ifft_opt(&mut result, &inv_twiddles).unwrap();

        // Scale by 1/n
        let n_inv = FE::from(n as u64).inv().unwrap();
        for val in result.iter_mut() {
            *val = &*val * &n_inv;
        }

        assert_eq!(result, input, "Roundtrip failed for order {}", order);
    }
}

#[test]
fn test_bowers_fft_ifft_roundtrip_medium() {
    for order in 5..=8u64 {
        let n = 1 << order;
        let input: Vec<FE> = (0..n).map(|i| FE::from(i as u64)).collect();

        let fwd_twiddles = LayerTwiddles::<F>::new(order).unwrap();
        let inv_twiddles = LayerTwiddles::<F>::new_inverse(order).unwrap();

        // FFT -> IFFT roundtrip
        let mut result = input.clone();
        bowers_fft_opt_fused(&mut result, &fwd_twiddles).unwrap();
        in_place_bit_reverse_permute(&mut result);

        in_place_bit_reverse_permute(&mut result);
        bowers_ifft_opt(&mut result, &inv_twiddles).unwrap();

        // Scale by 1/n
        let n_inv = FE::from(n as u64).inv().unwrap();
        for val in result.iter_mut() {
            *val = &*val * &n_inv;
        }

        assert_eq!(
            result, input,
            "FFT->IFFT roundtrip failed for order {}",
            order
        );
    }
}

#[test]
fn test_bowers_ifft_fft_roundtrip() {
    // Test IFFT -> FFT roundtrip (reverse order)
    // This verifies that FFT(IFFT(x)) = x (with proper scaling and permutations)
    for order in 2..=6u64 {
        let n = 1 << order;
        let input: Vec<FE> = (0..n).map(|i| FE::from(i as u64)).collect();

        let fwd_twiddles = LayerTwiddles::<F>::new(order).unwrap();
        let inv_twiddles = LayerTwiddles::<F>::new_inverse(order).unwrap();

        // For IFFT -> FFT roundtrip, we treat input as frequency-domain values.
        // IFFT: bit-reverse input, apply inverse butterflies
        let mut result = input.clone();
        in_place_bit_reverse_permute(&mut result);
        bowers_ifft_opt(&mut result, &inv_twiddles).unwrap();

        // FFT: apply forward butterflies, bit-reverse output
        bowers_fft_opt_fused(&mut result, &fwd_twiddles).unwrap();
        in_place_bit_reverse_permute(&mut result);

        // The FFT and IFFT should cancel out (both contribute n factor, so we need 1/n)
        let n_inv = FE::from(n as u64).inv().unwrap();
        for val in result.iter_mut() {
            *val = &*val * &n_inv;
        }

        assert_eq!(
            result, input,
            "IFFT->FFT roundtrip failed for order {}",
            order
        );
    }
}

#[test]
fn test_bowers_ifft_matches_naive() {
    // Compare Bowers IFFT against naive IDFT
    for order in 2..=6u64 {
        let n = 1 << order;
        // Start with FFT output (evaluations)
        let input: Vec<FE> = (0..n).map(|i| FE::from(i as u64)).collect();
        let fwd_twiddles = LayerTwiddles::<F>::new(order).unwrap();

        let mut evals = input.clone();
        bowers_fft_opt_fused(&mut evals, &fwd_twiddles).unwrap();
        in_place_bit_reverse_permute(&mut evals);

        // Naive IDFT
        let expected = naive_idft(&evals);

        // Bowers IFFT
        let inv_twiddles = LayerTwiddles::<F>::new_inverse(order).unwrap();
        let mut result = evals;
        in_place_bit_reverse_permute(&mut result);
        bowers_ifft_opt(&mut result, &inv_twiddles).unwrap();

        // Scale by 1/n
        let n_inv = FE::from(n as u64).inv().unwrap();
        for val in result.iter_mut() {
            *val = &*val * &n_inv;
        }

        assert_eq!(
            result, expected,
            "IFFT differs from naive IDFT for order {}",
            order
        );
    }
}

#[test]
fn test_layer_twiddles_inverse_creation() {
    let order = 4u64;
    let inv_twiddles = LayerTwiddles::<F>::new_inverse(order).unwrap();

    assert_eq!(inv_twiddles.layers.len(), 4);
    assert_eq!(inv_twiddles.layers[0].len(), 8);
    assert_eq!(inv_twiddles.layers[1].len(), 4);
    assert_eq!(inv_twiddles.layers[2].len(), 2);
    assert_eq!(inv_twiddles.layers[3].len(), 1);

    // First twiddle of each layer should still be 1
    for layer in &inv_twiddles.layers {
        assert_eq!(layer[0], FE::one());
    }
}

#[test]
fn test_fft_ifft_roundtrip_edge_cases() {
    let order = 6u64;
    let n = 1 << order;

    let fwd_twiddles = LayerTwiddles::<F>::new(order).unwrap();
    let inv_twiddles = LayerTwiddles::<F>::new_inverse(order).unwrap();
    let n_inv = FE::from(n as u64).inv().unwrap();

    // All zeros
    let zeros: Vec<FE> = vec![FE::zero(); n];
    let mut result = zeros.clone();
    bowers_fft_opt_fused(&mut result, &fwd_twiddles).unwrap();
    in_place_bit_reverse_permute(&mut result);
    in_place_bit_reverse_permute(&mut result);
    bowers_ifft_opt(&mut result, &inv_twiddles).unwrap();
    for val in result.iter_mut() {
        *val = &*val * &n_inv;
    }
    assert_eq!(result, zeros, "Roundtrip failed for all zeros");

    // All ones
    let ones: Vec<FE> = vec![FE::one(); n];
    let mut result = ones.clone();
    bowers_fft_opt_fused(&mut result, &fwd_twiddles).unwrap();
    in_place_bit_reverse_permute(&mut result);
    in_place_bit_reverse_permute(&mut result);
    bowers_ifft_opt(&mut result, &inv_twiddles).unwrap();
    for val in result.iter_mut() {
        *val = &*val * &n_inv;
    }
    assert_eq!(result, ones, "Roundtrip failed for all ones");

    // Single non-zero element
    let mut sparse = vec![FE::zero(); n];
    sparse[0] = FE::from(42u64);
    let mut result = sparse.clone();
    bowers_fft_opt_fused(&mut result, &fwd_twiddles).unwrap();
    in_place_bit_reverse_permute(&mut result);
    in_place_bit_reverse_permute(&mut result);
    bowers_ifft_opt(&mut result, &inv_twiddles).unwrap();
    for val in result.iter_mut() {
        *val = &*val * &n_inv;
    }
    assert_eq!(result, sparse, "Roundtrip failed for sparse input");
}

// =========================================================================
// Property-based tests (proptest)
// =========================================================================

prop_compose! {
    fn field_element()(num in any::<u64>()) -> FE {
        FE::from(num)
    }
}

prop_compose! {
    fn field_vec(max_exp: u8)(exp in 1u8..=max_exp)(
        vec in collection::vec(field_element(), 1usize << exp)
    ) -> Vec<FE> {
        vec
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    #[test]
    fn proptest_bowers_matches_naive_dft(coeffs in field_vec(8)) {
        let expected = naive_dft(&coeffs);

        let order = coeffs.len().trailing_zeros() as u64;
        let layer_twiddles = LayerTwiddles::<F>::new(order).unwrap();

        let mut result = coeffs;
        bowers_fft_opt_fused(&mut result, &layer_twiddles).unwrap();
        in_place_bit_reverse_permute(&mut result);

        prop_assert_eq!(result, expected);
    }

    #[test]
    fn proptest_bowers_fft_ifft_roundtrip(coeffs in field_vec(8)) {
        let n = coeffs.len();
        let order = n.trailing_zeros() as u64;

        let fwd_twiddles = LayerTwiddles::<F>::new(order).unwrap();
        let inv_twiddles = LayerTwiddles::<F>::new_inverse(order).unwrap();

        // FFT -> bit-reverse -> bit-reverse -> IFFT -> scale
        let mut result = coeffs.clone();
        bowers_fft_opt_fused(&mut result, &fwd_twiddles).unwrap();
        in_place_bit_reverse_permute(&mut result);

        in_place_bit_reverse_permute(&mut result);
        bowers_ifft_opt(&mut result, &inv_twiddles).unwrap();

        let n_inv = FE::from(n as u64).inv().unwrap();
        for val in result.iter_mut() {
            *val = &*val * &n_inv;
        }

        prop_assert_eq!(result, coeffs);
    }
}

// =========================================================================
// Parallel tests
// =========================================================================

#[cfg(feature = "parallel")]
mod parallel_tests {
    use super::*;

    // ---- Parallel IFFT tests ----

    #[test]
    fn test_bowers_ifft_opt_parallel_roundtrip() {
        for order in 2..=10u64 {
            let n = 1 << order;
            let input: Vec<FE> = (0..n).map(|i| FE::from(i as u64)).collect();

            let fwd_twiddles = LayerTwiddles::<F>::new(order).unwrap();
            let inv_twiddles = LayerTwiddles::<F>::new_inverse(order).unwrap();

            // FFT
            let mut result = input.clone();
            bowers_fft_opt_fused_parallel(&mut result, &fwd_twiddles).unwrap();
            in_place_bit_reverse_permute(&mut result);

            // Parallel IFFT
            in_place_bit_reverse_permute(&mut result);
            bowers_ifft_opt_parallel(&mut result, &inv_twiddles).unwrap();

            // Scale by 1/n
            let n_inv = FE::from(n as u64).inv().unwrap();
            for val in result.iter_mut() {
                *val = &*val * &n_inv;
            }

            assert_eq!(
                result, input,
                "Parallel FFT->IFFT roundtrip failed for order {}",
                order
            );
        }
    }

    #[test]
    fn test_parallel_ifft_matches_sequential() {
        for order in 2..=10u64 {
            let n = 1 << order;
            let input: Vec<FE> = (0..n).map(|i| FE::from(i as u64)).collect();
            let inv_twiddles = LayerTwiddles::<F>::new_inverse(order).unwrap();

            let mut result_seq = input.clone();
            bowers_ifft_opt(&mut result_seq, &inv_twiddles).unwrap();

            let mut result_par = input.clone();
            bowers_ifft_opt_parallel(&mut result_par, &inv_twiddles).unwrap();

            assert_eq!(
                result_seq, result_par,
                "Parallel IFFT differs from sequential for order {}",
                order
            );
        }
    }

    // ---- Parallel FFT tests ----

    #[test]
    fn test_bowers_fft_opt_fused_parallel_small() {
        // Small input - should use sequential path
        let input: Vec<FE> = (0..16).map(|i| FE::from(i as u64)).collect();
        let expected = naive_dft(&input);

        let layer_twiddles = LayerTwiddles::<F>::new(4).unwrap();
        let mut result = input.clone();
        bowers_fft_opt_fused_parallel(&mut result, &layer_twiddles).unwrap();
        in_place_bit_reverse_permute(&mut result);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_bowers_fft_opt_fused_parallel_medium() {
        // Medium input
        let input: Vec<FE> = (0..256).map(|i| FE::from(i as u64)).collect();
        let expected = naive_dft(&input);

        let layer_twiddles = LayerTwiddles::<F>::new(8).unwrap();
        let mut result = input.clone();
        bowers_fft_opt_fused_parallel(&mut result, &layer_twiddles).unwrap();
        in_place_bit_reverse_permute(&mut result);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_bowers_fft_opt_fused_parallel_large() {
        // Large input - should exercise parallel paths
        let input: Vec<FE> = (0..4096).map(|i| FE::from(i as u64)).collect();
        let expected = naive_dft(&input);

        let layer_twiddles = LayerTwiddles::<F>::new(12).unwrap();
        let mut result = input.clone();
        bowers_fft_opt_fused_parallel(&mut result, &layer_twiddles).unwrap();
        in_place_bit_reverse_permute(&mut result);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_parallel_matches_sequential() {
        // Verify parallel and sequential produce identical results
        let input: Vec<FE> = (0..1024).map(|i| FE::from(i as u64)).collect();

        let layer_twiddles = LayerTwiddles::<F>::new(10).unwrap();

        let mut result_seq = input.clone();
        bowers_fft_opt_fused(&mut result_seq, &layer_twiddles).unwrap();
        in_place_bit_reverse_permute(&mut result_seq);

        let mut result_par = input.clone();
        bowers_fft_opt_fused_parallel(&mut result_par, &layer_twiddles).unwrap();
        in_place_bit_reverse_permute(&mut result_par);

        assert_eq!(result_seq, result_par);
    }
}

// =========================================================================
// Adaptive threshold tests
// =========================================================================

#[cfg(feature = "parallel")]
#[test]
fn test_adaptive_threshold_scales_with_threads() {
    use crate::fft::bowers_fft::bowers_fft_opt_fused_parallel;
    use rayon::ThreadPoolBuilder;

    // Test with different thread counts
    for num_threads in [1, 2, 4, 8] {
        let pool = ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .unwrap();

        pool.install(|| {
            // Verify FFT still works with adaptive threshold
            let input: Vec<FE> = (0..256).map(|i| FE::from(i as u64)).collect();
            let expected = naive_dft(&input);

            let layer_twiddles = LayerTwiddles::<F>::new(8).unwrap();
            let mut result = input.clone();
            bowers_fft_opt_fused_parallel(&mut result, &layer_twiddles).unwrap();
            in_place_bit_reverse_permute(&mut result);

            assert_eq!(
                result, expected,
                "FFT correctness with {} threads",
                num_threads
            );
        });
    }
}

#[test]
fn test_mismatched_twiddle_table_error() {
    // Test that FFT returns error instead of panicking when twiddle table size doesn't match input
    let order_input = 8u64; // Input size: 256 elements
    let order_twiddles = 6u64; // Twiddle table for 64 elements (WRONG!)

    let n = 1 << order_input;
    let input: Vec<FE> = (0..n).map(|i| FE::from(i as u64)).collect();

    // Create mismatched twiddle table (order 6 instead of 8)
    let wrong_twiddles = LayerTwiddles::<F>::new(order_twiddles).unwrap();

    // Forward FFT should return error
    let mut data = input.clone();
    let result = bowers_fft_opt_fused(&mut data, &wrong_twiddles);
    assert!(
        result.is_err(),
        "FFT should return error with mismatched twiddle table"
    );

    // Inverse FFT should also return error
    let wrong_inv_twiddles = LayerTwiddles::<F>::new_inverse(order_twiddles).unwrap();
    let mut data = input.clone();
    let result = bowers_ifft_opt(&mut data, &wrong_inv_twiddles);
    assert!(
        result.is_err(),
        "IFFT should return error with mismatched twiddle table"
    );
}

#[cfg(feature = "parallel")]
#[test]
fn test_mismatched_twiddle_table_error_parallel() {
    // Test parallel FFT with mismatched twiddle table
    let order_input = 10u64; // Input size: 1024 elements
    let order_twiddles = 8u64; // Twiddle table for 256 elements (WRONG!)

    let n = 1 << order_input;
    let input: Vec<FE> = (0..n).map(|i| FE::from(i as u64)).collect();
    let wrong_twiddles = LayerTwiddles::<F>::new(order_twiddles).unwrap();

    let mut data = input;
    let result = bowers_fft_opt_fused_parallel(&mut data, &wrong_twiddles);
    assert!(
        result.is_err(),
        "Parallel FFT should return error with mismatched twiddle table"
    );
}
