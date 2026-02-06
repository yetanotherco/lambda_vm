//! Tests for Metal FFT implementation
//!
//! Includes differential fuzzing tests comparing Metal GPU FFT against
//! the CPU Bowers FFT implementation to verify correctness.

use super::device::MetalError;
use super::fft::MetalFft;
use crate::fft::cpu::bit_reversing::in_place_bit_reverse_permute;
use crate::fft::cpu::fft::in_place_nr_2radix_fft;
use crate::fft::cpu::roots_of_unity::get_twiddles;
use crate::field::element::FieldElement;
use crate::field::fields::fft_friendly::u64_goldilocks::GoldilocksField;
use crate::field::traits::RootsConfig;
use alloc::vec;
use alloc::vec::Vec;

#[cfg(feature = "proptest")]
use proptest::{collection, prelude::*};

type F = GoldilocksField;
type FE = FieldElement<F>;

/// Goldilocks prime for canonicalization
const GOLDILOCKS_PRIME: u64 = 0xFFFF_FFFF_0000_0001;

/// Canonicalize a field element to [0, p)
fn canonicalize(x: u64) -> u64 {
    if x >= GOLDILOCKS_PRIME {
        x - GOLDILOCKS_PRIME
    } else {
        x
    }
}

/// CPU reference FFT using existing Cooley-Tukey implementation
fn cpu_fft(input: &[u64]) -> Vec<u64> {
    assert!(input.len().is_power_of_two(), "cpu_fft requires power-of-two input");
    let order = input.len().trailing_zeros() as u64;

    // Convert to FieldElements
    let mut data: Vec<FE> = input.iter().map(|&x| FE::from(x)).collect();

    // Get twiddles and perform FFT
    let twiddles = get_twiddles(order, RootsConfig::BitReverse)
        .expect("get_twiddles failed: order should be valid in test context");
    in_place_nr_2radix_fft::<F, F>(&mut data, &twiddles);
    in_place_bit_reverse_permute(&mut data);

    // Convert back to u64 and canonicalize
    data.iter().map(|fe| canonicalize(*fe.value())).collect()
}

// =========================================================================
// Metal FFT tests
// =========================================================================

#[test]
fn test_metal_fft_medium() {
    match MetalFft::new() {
        Ok(fft) => {
            let mut data: Vec<u64> = (0..16).collect();
            match fft.fft_natural_order(&mut data) {
                Ok(()) => {
                    // Sum should be preserved (first element after FFT)
                    // For [0, 1, 2, ..., 15], sum = 120
                    assert_eq!(canonicalize(data[0]), 120);
                }
                Err(e) => panic!("FFT failed: {:?}", e),
            }
        }
        Err(MetalError::NoDevice) => {
            println!("Skipping test: no Metal device available");
        }
        Err(e) => panic!("Unexpected error: {:?}", e),
    }
}

#[test]
fn test_metal_fft_large() {
    match MetalFft::new() {
        Ok(fft) => {
            let mut data: Vec<u64> = (0..256).collect();
            match fft.fft_natural_order(&mut data) {
                Ok(()) => {
                    // Sum = 0 + 1 + ... + 255 = 255 * 256 / 2 = 32640
                    assert_eq!(canonicalize(data[0]), 32640);
                }
                Err(e) => panic!("FFT failed: {:?}", e),
            }
        }
        Err(MetalError::NoDevice) => {
            println!("Skipping test: no Metal device available");
        }
        Err(e) => panic!("Unexpected error: {:?}", e),
    }
}

#[test]
fn test_metal_fft_size_one() {
    match MetalFft::new() {
        Ok(fft) => {
            let mut data = vec![42u64];
            match fft.fft_natural_order(&mut data) {
                Ok(()) => {
                    assert_eq!(data, vec![42u64]);
                }
                Err(e) => panic!("FFT failed: {:?}", e),
            }
        }
        Err(MetalError::NoDevice) => {
            println!("Skipping test: no Metal device available");
        }
        Err(e) => panic!("Unexpected error: {:?}", e),
    }
}

#[test]
fn test_metal_fft_non_power_of_two() {
    match MetalFft::new() {
        Ok(fft) => {
            let mut data: Vec<u64> = (0..7).collect();
            match fft.fft_natural_order(&mut data) {
                Ok(()) => panic!("Should have failed for non-power-of-two input"),
                Err(MetalError::InvalidInput(_)) => {
                    // Expected
                }
                Err(e) => panic!("Unexpected error type: {:?}", e),
            }
        }
        Err(MetalError::NoDevice) => {
            println!("Skipping test: no Metal device available");
        }
        Err(e) => panic!("Unexpected error: {:?}", e),
    }
}

// =========================================================================
// Differential fuzzing tests: Metal vs CPU FFT
// =========================================================================

/// Compare Metal FFT output against CPU FFT for correctness
fn compare_metal_vs_cpu(input: &[u64]) {
    let metal_fft = match MetalFft::new() {
        Ok(fft) => fft,
        Err(MetalError::NoDevice) => {
            println!("Skipping differential test: no Metal device available");
            return;
        }
        Err(e) => panic!("Metal FFT creation failed: {:?}", e),
    };

    // Run Metal FFT
    let mut metal_result = input.to_vec();
    match metal_fft.fft_natural_order(&mut metal_result) {
        Ok(()) => {}
        Err(e) => panic!("Metal FFT failed: {:?}", e),
    }

    // Run CPU FFT
    let cpu_result = cpu_fft(input);

    // Canonicalize Metal results for comparison
    let metal_canonical: Vec<u64> = metal_result.iter().map(|&x| canonicalize(x)).collect();

    // Compare
    assert_eq!(
        metal_canonical,
        cpu_result,
        "Metal FFT output differs from CPU FFT!\n\
         Input length: {}\n\
         First differing index: {:?}",
        input.len(),
        metal_canonical
            .iter()
            .zip(cpu_result.iter())
            .position(|(m, c)| m != c)
    );
}

#[test]
fn test_differential_fft_small_sizes() {
    for order in 1..=6u32 {
        let n = 1 << order;
        let input: Vec<u64> = (0..n).collect();
        compare_metal_vs_cpu(&input);
    }
}

#[test]
fn test_differential_fft_medium_sizes() {
    for order in 7..=10u32 {
        let n = 1 << order;
        let input: Vec<u64> = (0..n).collect();
        compare_metal_vs_cpu(&input);
    }
}

#[test]
fn test_differential_fft_edge_cases() {
    // All zeros
    let zeros: Vec<u64> = vec![0; 64];
    compare_metal_vs_cpu(&zeros);

    // All ones
    let ones: Vec<u64> = vec![1; 64];
    compare_metal_vs_cpu(&ones);

    // Alternating
    let alternating: Vec<u64> = (0..64).map(|i| if i % 2 == 0 { 0 } else { 1 }).collect();
    compare_metal_vs_cpu(&alternating);

    // Large values (close to prime)
    let large: Vec<u64> = (0..64).map(|i| GOLDILOCKS_PRIME - 1 - i as u64).collect();
    compare_metal_vs_cpu(&large);
}

#[test]
fn test_differential_fft_random_small() {
    // Simple pseudo-random sequence for reproducibility
    let mut rng_state = 12345u64;
    let next_rand = |state: &mut u64| -> u64 {
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        *state
    };

    for order in 2..=8u32 {
        let n = 1 << order;
        let input: Vec<u64> = (0..n)
            .map(|_| next_rand(&mut rng_state) % GOLDILOCKS_PRIME)
            .collect();
        compare_metal_vs_cpu(&input);
    }
}

// =========================================================================
// IFFT and Roundtrip tests
// =========================================================================

/// Scale all elements by 1/n (for IFFT normalization)
fn scale_by_inverse_n(data: &mut [u64], n: usize) {
    let n_inv = goldilocks_inverse(n as u64);
    for val in data.iter_mut() {
        *val = goldilocks_mul(*val, n_inv);
    }
}

/// Goldilocks field multiplication (for scaling)
fn goldilocks_mul(a: u64, b: u64) -> u64 {
    let product = (a as u128) * (b as u128);
    goldilocks_reduce128(product)
}

/// Goldilocks field inversion using Fermat's little theorem
fn goldilocks_inverse(a: u64) -> u64 {
    const P_MINUS_2: u64 = 0xFFFF_FFFE_FFFF_FFFF;
    goldilocks_pow(a, P_MINUS_2)
}

/// Binary exponentiation in Goldilocks field
fn goldilocks_pow(mut base: u64, mut exp: u64) -> u64 {
    let mut result = 1u64;
    while exp > 0 {
        if exp & 1 == 1 {
            result = goldilocks_mul(result, base);
        }
        base = goldilocks_square(base);
        exp >>= 1;
    }
    result
}

/// Square in Goldilocks field
fn goldilocks_square(a: u64) -> u64 {
    goldilocks_mul(a, a)
}

/// Reduce 128-bit to Goldilocks field element
fn goldilocks_reduce128(x: u128) -> u64 {
    const EPSILON: u64 = 0xFFFF_FFFF;

    let x_lo = x as u64;
    let x_hi = (x >> 64) as u64;
    let x_hi_hi = x_hi >> 32;
    let x_hi_lo = x_hi & EPSILON;

    let (t0, borrow) = x_lo.overflowing_sub(x_hi_hi);
    let t0 = if borrow { t0.wrapping_sub(EPSILON) } else { t0 };

    let t1 = (x_hi_lo << 32).wrapping_sub(x_hi_lo);

    let (result, carry) = t0.overflowing_add(t1);
    if carry {
        result.wrapping_add(EPSILON)
    } else {
        result
    }
}

#[test]
fn test_metal_ifft_basic() {
    match MetalFft::new() {
        Ok(fft) => {
            let mut data: Vec<u64> = (0..16).collect();
            match fft.ifft_natural_order(&mut data) {
                Ok(()) => {
                    // IFFT should also preserve the sum in the first element
                    // (before scaling by 1/n)
                    assert_eq!(canonicalize(data[0]), 120);
                }
                Err(e) => panic!("IFFT failed: {:?}", e),
            }
        }
        Err(MetalError::NoDevice) => {
            println!("Skipping test: no Metal device available");
        }
        Err(e) => panic!("Unexpected error: {:?}", e),
    }
}

#[test]
fn test_metal_fft_ifft_roundtrip_small() {
    let metal_fft = match MetalFft::new() {
        Ok(fft) => fft,
        Err(MetalError::NoDevice) => {
            println!("Skipping test: no Metal device available");
            return;
        }
        Err(e) => panic!("Metal FFT creation failed: {:?}", e),
    };

    for order in 1..=6u32 {
        let n = 1 << order;
        let original: Vec<u64> = (0..n).map(|i| i as u64).collect();

        // FFT -> IFFT -> scale by 1/n should return original
        let mut result = original.clone();
        if let Err(e) = metal_fft.fft_natural_order(&mut result) {
            panic!("FFT failed: {:?}", e);
        }
        if let Err(e) = metal_fft.ifft_natural_order(&mut result) {
            panic!("IFFT failed: {:?}", e);
        }
        scale_by_inverse_n(&mut result, n);

        // Canonicalize for comparison
        let result_canonical: Vec<u64> = result.iter().map(|&x| canonicalize(x)).collect();
        let original_canonical: Vec<u64> = original.iter().map(|&x| canonicalize(x)).collect();

        assert_eq!(
            result_canonical, original_canonical,
            "FFT->IFFT roundtrip failed for order {}",
            order
        );
    }
}

#[test]
fn test_metal_fft_ifft_roundtrip_medium() {
    let metal_fft = match MetalFft::new() {
        Ok(fft) => fft,
        Err(MetalError::NoDevice) => {
            println!("Skipping test: no Metal device available");
            return;
        }
        Err(e) => panic!("Metal FFT creation failed: {:?}", e),
    };

    for order in 7..=10u32 {
        let n = 1 << order;
        let original: Vec<u64> = (0..n).map(|i| (i as u64) % GOLDILOCKS_PRIME).collect();

        // FFT -> IFFT -> scale by 1/n should return original
        let mut result = original.clone();
        if let Err(e) = metal_fft.fft_natural_order(&mut result) {
            panic!("FFT failed: {:?}", e);
        }
        if let Err(e) = metal_fft.ifft_natural_order(&mut result) {
            panic!("IFFT failed: {:?}", e);
        }
        scale_by_inverse_n(&mut result, n);

        // Canonicalize for comparison
        let result_canonical: Vec<u64> = result.iter().map(|&x| canonicalize(x)).collect();
        let original_canonical: Vec<u64> = original.iter().map(|&x| canonicalize(x)).collect();

        assert_eq!(
            result_canonical, original_canonical,
            "FFT->IFFT roundtrip failed for order {}",
            order
        );
    }
}

#[test]
fn test_metal_ifft_fft_roundtrip() {
    let metal_fft = match MetalFft::new() {
        Ok(fft) => fft,
        Err(MetalError::NoDevice) => {
            println!("Skipping test: no Metal device available");
            return;
        }
        Err(e) => panic!("Metal FFT creation failed: {:?}", e),
    };

    for order in 1..=8u32 {
        let n = 1 << order;
        let original: Vec<u64> = (0..n)
            .map(|i| (i as u64 * 7 + 3) % GOLDILOCKS_PRIME)
            .collect();

        // IFFT -> FFT -> scale by 1/n should return original
        let mut result = original.clone();
        if let Err(e) = metal_fft.ifft_natural_order(&mut result) {
            panic!("IFFT failed: {:?}", e);
        }
        if let Err(e) = metal_fft.fft_natural_order(&mut result) {
            panic!("FFT failed: {:?}", e);
        }
        scale_by_inverse_n(&mut result, n);

        // Canonicalize for comparison
        let result_canonical: Vec<u64> = result.iter().map(|&x| canonicalize(x)).collect();
        let original_canonical: Vec<u64> = original.iter().map(|&x| canonicalize(x)).collect();

        assert_eq!(
            result_canonical, original_canonical,
            "IFFT->FFT roundtrip failed for order {}",
            order
        );
    }
}

#[test]
fn test_metal_fft_ifft_roundtrip_edge_cases() {
    let metal_fft = match MetalFft::new() {
        Ok(fft) => fft,
        Err(MetalError::NoDevice) => {
            println!("Skipping test: no Metal device available");
            return;
        }
        Err(e) => panic!("Metal FFT creation failed: {:?}", e),
    };

    // All zeros
    let n = 64;
    let zeros: Vec<u64> = vec![0; n];
    let mut result = zeros.clone();
    if let Err(e) = metal_fft.fft_natural_order(&mut result) {
        panic!("FFT failed: {:?}", e);
    }
    if let Err(e) = metal_fft.ifft_natural_order(&mut result) {
        panic!("IFFT failed: {:?}", e);
    }
    scale_by_inverse_n(&mut result, n);
    let result_canonical: Vec<u64> = result.iter().map(|&x| canonicalize(x)).collect();
    assert_eq!(result_canonical, zeros, "Roundtrip failed for all zeros");

    // All ones
    let ones: Vec<u64> = vec![1; n];
    let mut result = ones.clone();
    if let Err(e) = metal_fft.fft_natural_order(&mut result) {
        panic!("FFT failed: {:?}", e);
    }
    if let Err(e) = metal_fft.ifft_natural_order(&mut result) {
        panic!("IFFT failed: {:?}", e);
    }
    scale_by_inverse_n(&mut result, n);
    let result_canonical: Vec<u64> = result.iter().map(|&x| canonicalize(x)).collect();
    assert_eq!(result_canonical, ones, "Roundtrip failed for all ones");

    // Alternating
    let alternating: Vec<u64> = (0..n as u64)
        .map(|i| if i % 2 == 0 { 0 } else { 1 })
        .collect();
    let mut result = alternating.clone();
    if let Err(e) = metal_fft.fft_natural_order(&mut result) {
        panic!("FFT failed: {:?}", e);
    }
    if let Err(e) = metal_fft.ifft_natural_order(&mut result) {
        panic!("IFFT failed: {:?}", e);
    }
    scale_by_inverse_n(&mut result, n);
    let result_canonical: Vec<u64> = result.iter().map(|&x| canonicalize(x)).collect();
    assert_eq!(
        result_canonical, alternating,
        "Roundtrip failed for alternating"
    );

    // Large values near prime
    let large: Vec<u64> = (0..n as u64).map(|i| GOLDILOCKS_PRIME - 1 - i).collect();
    let mut result = large.clone();
    if let Err(e) = metal_fft.fft_natural_order(&mut result) {
        panic!("FFT failed: {:?}", e);
    }
    if let Err(e) = metal_fft.ifft_natural_order(&mut result) {
        panic!("IFFT failed: {:?}", e);
    }
    scale_by_inverse_n(&mut result, n);
    let result_canonical: Vec<u64> = result.iter().map(|&x| canonicalize(x)).collect();
    let large_canonical: Vec<u64> = large.iter().map(|&x| canonicalize(x)).collect();
    assert_eq!(
        result_canonical, large_canonical,
        "Roundtrip failed for large values"
    );
}

#[test]
fn test_metal_fft_ifft_roundtrip_random() {
    let metal_fft = match MetalFft::new() {
        Ok(fft) => fft,
        Err(MetalError::NoDevice) => {
            println!("Skipping test: no Metal device available");
            return;
        }
        Err(e) => panic!("Metal FFT creation failed: {:?}", e),
    };

    // Simple pseudo-random sequence for reproducibility
    let mut rng_state = 98765u64;
    let next_rand = |state: &mut u64| -> u64 {
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        *state
    };

    for order in 2..=8u32 {
        let n = 1 << order;
        let original: Vec<u64> = (0..n)
            .map(|_| next_rand(&mut rng_state) % GOLDILOCKS_PRIME)
            .collect();

        let mut result = original.clone();
        if let Err(e) = metal_fft.fft_natural_order(&mut result) {
            panic!("FFT failed: {:?}", e);
        }
        if let Err(e) = metal_fft.ifft_natural_order(&mut result) {
            panic!("IFFT failed: {:?}", e);
        }
        scale_by_inverse_n(&mut result, n);

        let result_canonical: Vec<u64> = result.iter().map(|&x| canonicalize(x)).collect();
        let original_canonical: Vec<u64> = original.iter().map(|&x| canonicalize(x)).collect();

        assert_eq!(
            result_canonical, original_canonical,
            "Random roundtrip failed for order {}",
            order
        );
    }
}

// =========================================================================
// Batch FFT tests
// =========================================================================

#[test]
fn test_metal_batch_fft() {
    match MetalFft::new() {
        Ok(fft) => {
            let poly_len = 8;
            let num_polys = 4;
            let mut data: Vec<u64> = (0..(poly_len * num_polys) as u64).collect();

            match fft.batch_fft(&mut data, poly_len, num_polys) {
                Ok(()) => {
                    // Each polynomial should have its sum as first element
                    // Poly 0: 0+1+2+...+7 = 28
                    // Poly 1: 8+9+...+15 = 92
                    // etc.
                    for poly_idx in 0..num_polys {
                        let start = poly_idx * poly_len;
                        let expected_sum: u64 = (start..(start + poly_len)).map(|i| i as u64).sum();
                        assert_eq!(
                            canonicalize(data[start]),
                            expected_sum,
                            "Polynomial {} sum mismatch",
                            poly_idx
                        );
                    }
                }
                Err(e) => panic!("Batch FFT failed: {:?}", e),
            }
        }
        Err(MetalError::NoDevice) => {
            println!("Skipping test: no Metal device available");
        }
        Err(e) => panic!("Unexpected error: {:?}", e),
    }
}

// =========================================================================
// Property-based tests (proptest)
// =========================================================================

#[cfg(feature = "proptest")]
prop_compose! {
    fn field_element()(num in any::<u64>()) -> u64 {
        num % GOLDILOCKS_PRIME
    }
}

#[cfg(feature = "proptest")]
prop_compose! {
    fn field_vec(max_exp: u8)(exp in 1u8..=max_exp)(
        vec in collection::vec(field_element(), 1usize << exp)
    ) -> Vec<u64> {
        vec
    }
}

#[cfg(feature = "proptest")]
proptest! {
    #![proptest_config(ProptestConfig::with_cases(20))]

    #[test]
    fn proptest_metal_vs_cpu_fft(coeffs in field_vec(8)) {
        compare_metal_vs_cpu(&coeffs);
    }

    #[test]
    fn proptest_metal_fft_preserves_sum(coeffs in field_vec(8)) {
        let metal_fft = match MetalFft::new() {
            Ok(fft) => fft,
            Err(MetalError::NoDevice) => {
                return Ok(());
            }
            Err(e) => panic!("Metal FFT creation failed: {:?}", e),
        };

        // Calculate expected sum
        let expected_sum: u64 = coeffs.iter().fold(0u64, |acc, &x| {
            let sum = acc as u128 + x as u128;
            if sum >= GOLDILOCKS_PRIME as u128 {
                (sum - GOLDILOCKS_PRIME as u128) as u64
            } else {
                sum as u64
            }
        });

        let mut result = coeffs;
        metal_fft.fft_natural_order(&mut result).map_err(|e| {
            TestCaseError::fail(format!("FFT failed: {:?}", e))
        })?;

        // First element should be the sum
        prop_assert_eq!(canonicalize(result[0]), expected_sum);
    }
}
