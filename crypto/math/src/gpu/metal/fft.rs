//! Metal-accelerated FFT/NTT implementation for Goldilocks field.
//!
//! This module provides GPU-accelerated FFT operations using Apple's Metal API.
//! It implements the Cooley-Tukey radix-2 DIT (decimation-in-time) algorithm.
//!
//! # Algorithm
//!
//! The FFT is performed in stages:
//! 1. Copy input data to GPU buffer
//! 2. Execute log2(n) butterfly stages on GPU
//! 3. Perform bit-reversal permutation on GPU
//! 4. Copy results back to CPU
//!
//! For small sizes (< 8192), a single-kernel approach using shared memory
//! is more efficient due to reduced global memory traffic.

use super::device::{MetalState, GPU_THRESHOLD, MAX_FFT_SIZE};
use super::errors::MetalError;
use super::shaders::FFTPipelines;
use crate::field::element::FieldElement;
use crate::field::fields::fft_friendly::u64_goldilocks_native::GoldilocksField;
use alloc::vec::Vec;
use metal::{Buffer, MTLSize};

/// Threshold for using shared memory FFT kernel
const SHARED_MEMORY_THRESHOLD: usize = 1 << 13; // 8192 elements

/// Metal-accelerated FFT for Goldilocks field.
pub struct MetalFFT {
    /// Metal state (device, command queue, library)
    state: MetalState,
    /// Compiled compute pipelines
    pipelines: FFTPipelines,
    /// Cached twiddle factors (keyed by log2 of size)
    twiddle_cache: Vec<Option<Buffer>>,
}

impl MetalFFT {
    /// Create a new Metal FFT instance.
    pub fn new() -> Result<Self, MetalError> {
        let state = MetalState::new()?;
        let pipelines = FFTPipelines::new(&state.device, &state.library)?;

        // Pre-allocate twiddle cache for sizes up to 2^24
        let twiddle_cache = (0..25).map(|_| None).collect();

        Ok(Self {
            state,
            pipelines,
            twiddle_cache,
        })
    }

    /// Check if GPU FFT is recommended for the given size.
    pub fn should_use_gpu(size: usize) -> bool {
        size >= GPU_THRESHOLD && size <= MAX_FFT_SIZE && size.is_power_of_two()
    }

    /// Perform forward FFT on Goldilocks field elements.
    ///
    /// # Arguments
    /// * `input` - Input polynomial coefficients (must be power of 2 length)
    ///
    /// # Returns
    /// * Evaluations at roots of unity (in bit-reversed order, then permuted to natural)
    pub fn fft(
        &mut self,
        input: &[FieldElement<GoldilocksField>],
    ) -> Result<Vec<FieldElement<GoldilocksField>>, MetalError> {
        let n = input.len();

        // Validate input
        if !n.is_power_of_two() {
            return Err(MetalError::InvalidInputSize(n));
        }
        if n > MAX_FFT_SIZE {
            return Err(MetalError::InputTooLarge {
                size: n,
                max: MAX_FFT_SIZE,
            });
        }
        if n <= 1 {
            return Ok(input.to_vec());
        }

        let log_n = n.trailing_zeros() as usize;

        // Convert input to raw u64 values
        let input_raw: Vec<u64> = input.iter().map(|e| *e.value()).collect();

        // Get or generate twiddle factors
        let twiddles = self.get_or_generate_twiddles(log_n)?;

        // Create data buffer
        let data_buffer = self.state.create_buffer_with_data(&input_raw)?;

        // Execute FFT
        if n <= SHARED_MEMORY_THRESHOLD {
            self.execute_fft_small(&data_buffer, &twiddles, n, log_n)?;
        } else {
            self.execute_fft_staged(&data_buffer, &twiddles, n, log_n)?;
        }

        // Bit-reverse permutation
        self.execute_bit_reverse(&data_buffer, n, log_n)?;

        // Read results
        let mut output_raw = vec![0u64; n];
        self.state.read_buffer(&data_buffer, &mut output_raw)?;

        // Convert back to field elements
        let output: Vec<FieldElement<GoldilocksField>> = output_raw
            .into_iter()
            .map(FieldElement::from)
            .collect();

        Ok(output)
    }

    /// Get cached twiddle factors or generate them.
    fn get_or_generate_twiddles(&mut self, log_n: usize) -> Result<Buffer, MetalError> {
        // Check cache
        if let Some(ref buffer) = self.twiddle_cache[log_n] {
            return Ok(buffer.clone());
        }

        // Generate twiddle factors on CPU (more reliable for initial implementation)
        let twiddles = self.generate_twiddles_cpu(log_n)?;
        let buffer = self.state.create_buffer_with_data(&twiddles)?;

        // Cache for future use
        self.twiddle_cache[log_n] = Some(buffer.clone());

        Ok(buffer)
    }

    /// Generate twiddle factors on CPU.
    /// Twiddles are in bit-reversed order for NR DIT FFT.
    fn generate_twiddles_cpu(&self, log_n: usize) -> Result<Vec<u64>, MetalError> {
        use crate::fft::cpu::roots_of_unity::get_twiddles;
        use crate::field::traits::RootsConfig;

        let twiddles = get_twiddles::<GoldilocksField>(log_n as u64, RootsConfig::BitReverse)
            .map_err(|e| MetalError::TwiddleGenerationFailed(format!("{:?}", e)))?;

        Ok(twiddles.into_iter().map(|e| *e.value()).collect())
    }

    /// Execute FFT using staged approach (for large sizes).
    fn execute_fft_staged(
        &self,
        data_buffer: &Buffer,
        twiddles: &Buffer,
        n: usize,
        log_n: usize,
    ) -> Result<(), MetalError> {
        let mut group_count: u32 = 1;
        let mut group_size: u32 = n as u32;

        for _stage in 0..log_n {
            let command_buffer = self.state.command_queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();

            encoder.set_compute_pipeline_state(&self.pipelines.fft_radix2_stage);
            encoder.set_buffer(0, Some(data_buffer), 0);
            encoder.set_buffer(1, Some(twiddles), 0);

            let n_val = n as u32;
            encoder.set_bytes(2, 4, &n_val as *const u32 as *const _);
            encoder.set_bytes(3, 4, &group_count as *const u32 as *const _);
            encoder.set_bytes(4, 4, &group_size as *const u32 as *const _);

            // Calculate grid size
            let total_butterflies = (n / 2) as u64;
            let threadgroup_size = self.state.recommended_threadgroup_size() as u64;
            let grid_size = MTLSize::new(total_butterflies, 1, 1);
            let threadgroup = MTLSize::new(threadgroup_size.min(total_butterflies), 1, 1);

            encoder.dispatch_threads(grid_size, threadgroup);
            encoder.end_encoding();

            command_buffer.commit();
            command_buffer.wait_until_completed();

            // Update for next stage
            group_count *= 2;
            group_size /= 2;
        }

        Ok(())
    }

    /// Execute FFT using shared memory kernel (for small sizes).
    fn execute_fft_small(
        &self,
        data_buffer: &Buffer,
        twiddles: &Buffer,
        n: usize,
        log_n: usize,
    ) -> Result<(), MetalError> {
        let command_buffer = self.state.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.pipelines.fft_radix2_small);
        encoder.set_buffer(0, Some(data_buffer), 0);
        encoder.set_buffer(1, Some(twiddles), 0);

        let n_val = n as u32;
        let log_n_val = log_n as u32;
        encoder.set_bytes(2, 4, &n_val as *const u32 as *const _);
        encoder.set_bytes(3, 4, &log_n_val as *const u32 as *const _);

        // Allocate threadgroup memory for data
        let shared_memory_size = n * core::mem::size_of::<u64>();
        encoder.set_threadgroup_memory_length(0, shared_memory_size as u64);

        // Use single threadgroup for small FFT
        let threadgroup_size = self.state.recommended_threadgroup_size().min(n);
        let grid_size = MTLSize::new(threadgroup_size as u64, 1, 1);
        let threadgroup = MTLSize::new(threadgroup_size as u64, 1, 1);

        encoder.dispatch_threads(grid_size, threadgroup);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(())
    }

    /// Execute bit-reverse permutation on GPU.
    fn execute_bit_reverse(
        &self,
        data_buffer: &Buffer,
        n: usize,
        log_n: usize,
    ) -> Result<(), MetalError> {
        let command_buffer = self.state.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.pipelines.bit_reverse_permute);
        encoder.set_buffer(0, Some(data_buffer), 0);

        let n_val = n as u32;
        let log_n_val = log_n as u32;
        encoder.set_bytes(1, 4, &n_val as *const u32 as *const _);
        encoder.set_bytes(2, 4, &log_n_val as *const u32 as *const _);

        let threadgroup_size = self.state.recommended_threadgroup_size() as u64;
        let grid_size = MTLSize::new(n as u64, 1, 1);
        let threadgroup = MTLSize::new(threadgroup_size.min(n as u64), 1, 1);

        encoder.dispatch_threads(grid_size, threadgroup);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(())
    }

    /// Get device name for logging/debugging.
    pub fn device_name(&self) -> &str {
        self.state.device_name()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fft::cpu::ops::fft as cpu_fft;
    use crate::fft::cpu::roots_of_unity::get_twiddles;
    use crate::field::traits::{IsField, RootsConfig};

    fn make_test_input(size: usize) -> Vec<FieldElement<GoldilocksField>> {
        (0..size as u64)
            .map(|i| FieldElement::from(i + 1))
            .collect()
    }

    /// Test field multiplication on GPU matches CPU
    fn test_metal_multiplication(state: &MetalState, a: u64, b: u64) -> u64 {
        use super::super::shaders::KERNEL_TEST_MULTIPLY;
        use metal::MTLSize;

        // Get pipeline
        let function = state.library.get_function(KERNEL_TEST_MULTIPLY, None).unwrap();
        let pipeline = state.device.new_compute_pipeline_state_with_function(&function).unwrap();

        // Create buffer with single value
        let data = vec![a];
        let buffer = state.create_buffer_with_data(&data).unwrap();

        // Execute
        let command_buffer = state.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(&buffer), 0);
        encoder.set_bytes(1, 8, &b as *const u64 as *const _);
        let n: u32 = 1;
        encoder.set_bytes(2, 4, &n as *const u32 as *const _);
        encoder.dispatch_threads(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Read result
        let mut result = vec![0u64; 1];
        state.read_buffer(&buffer, &mut result).unwrap();
        result[0]
    }

    /// Test field addition on GPU matches CPU
    fn test_metal_addition(state: &MetalState, a: u64, b: u64) -> u64 {
        use super::super::shaders::KERNEL_TEST_ADD;
        use metal::MTLSize;

        let function = state.library.get_function(KERNEL_TEST_ADD, None).unwrap();
        let pipeline = state.device.new_compute_pipeline_state_with_function(&function).unwrap();

        let data = vec![a];
        let buffer = state.create_buffer_with_data(&data).unwrap();

        let command_buffer = state.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(&buffer), 0);
        encoder.set_bytes(1, 8, &b as *const u64 as *const _);
        let n: u32 = 1;
        encoder.set_bytes(2, 4, &n as *const u32 as *const _);
        encoder.dispatch_threads(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let mut result = vec![0u64; 1];
        state.read_buffer(&buffer, &mut result).unwrap();
        result[0]
    }

    fn canonicalize(x: u64) -> u64 {
        const P: u64 = 0xFFFFFFFF00000001;
        if x >= P { x - P } else { x }
    }

    #[test]
    fn test_metal_field_multiplication() {
        let state = match MetalState::new() {
            Ok(s) => s,
            Err(_) => return,
        };

        let test_cases: Vec<(u64, u64)> = vec![
            (2, 3), (5, 7), (1, 1), (0, 12345),
            (1u64 << 32, 2), (1u64 << 40, 1u64 << 30),
        ];

        for (a, b) in test_cases {
            let metal_result = test_metal_multiplication(&state, a, b);
            let cpu_result = GoldilocksField::mul(&a, &b);
            assert_eq!(
                canonicalize(cpu_result), canonicalize(metal_result),
                "Mul mismatch for {} * {}", a, b
            );
        }
    }

    #[test]
    fn test_metal_field_addition() {
        let state = match MetalState::new() {
            Ok(s) => s,
            Err(_) => return,
        };

        let test_cases: Vec<(u64, u64)> = vec![
            (2, 3), (5, 7), (0, 0),
            (0xFFFFFFFF00000000, 1), // Near prime
            (0xFFFFFFFF00000000, 2), // Wraps past prime
        ];

        for (a, b) in test_cases {
            let metal_result = test_metal_addition(&state, a, b);
            let cpu_result = GoldilocksField::add(&a, &b);
            assert_eq!(
                canonicalize(cpu_result), canonicalize(metal_result),
                "Add mismatch for {} + {}: CPU={}, Metal={}",
                a, b, canonicalize(cpu_result), canonicalize(metal_result)
            );
        }
    }

    #[test]
    fn test_metal_butterfly_operation() {
        let state = match MetalState::new() {
            Ok(s) => s,
            Err(_) => return,
        };

        use super::super::shaders::KERNEL_TEST_BUTTERFLY;
        use metal::MTLSize;

        let function = state.library.get_function(KERNEL_TEST_BUTTERFLY, None).unwrap();
        let pipeline = state.device.new_compute_pipeline_state_with_function(&function).unwrap();

        // Test butterfly: (a, b) -> (a + w*b, a - w*b)
        let test_cases = vec![
            (1u64, 2u64, 1u64),  // w=1: (1,2) -> (3, -1)
            (5u64, 3u64, 1u64),  // w=1: (5,3) -> (8, 2)
            (10u64, 4u64, 2u64), // w=2: (10,4) -> (10+8=18, 10-8=2)
        ];

        for (a, b, w) in test_cases {
            let a_buf = state.create_buffer_with_data(&[a]).unwrap();
            let b_buf = state.create_buffer_with_data(&[b]).unwrap();
            let w_buf = state.create_buffer_with_data(&[w]).unwrap();

            let command_buffer = state.command_queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_buffer(0, Some(&a_buf), 0);
            encoder.set_buffer(1, Some(&b_buf), 0);
            encoder.set_buffer(2, Some(&w_buf), 0);
            let n: u32 = 1;
            encoder.set_bytes(3, 4, &n as *const u32 as *const _);
            encoder.dispatch_threads(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();

            let mut a_result = vec![0u64; 1];
            let mut b_result = vec![0u64; 1];
            state.read_buffer(&a_buf, &mut a_result).unwrap();
            state.read_buffer(&b_buf, &mut b_result).unwrap();

            // CPU computation
            let wb = GoldilocksField::mul(&w, &b);
            let expected_a = GoldilocksField::add(&a, &wb);
            let expected_b = GoldilocksField::sub(&a, &wb);

            assert_eq!(
                canonicalize(expected_a), canonicalize(a_result[0]),
                "Butterfly a mismatch for ({}, {}, w={}): expected {}, got {}",
                a, b, w, canonicalize(expected_a), canonicalize(a_result[0])
            );
            assert_eq!(
                canonicalize(expected_b), canonicalize(b_result[0]),
                "Butterfly b mismatch for ({}, {}, w={}): expected {}, got {}",
                a, b, w, canonicalize(expected_b), canonicalize(b_result[0])
            );
        }
        println!("Butterfly tests passed!");
    }

    #[test]
    fn test_metal_fft_basic() {
        let metal_fft = match MetalFFT::new() {
            Ok(fft) => fft,
            Err(e) => {
                println!("Metal not available: {:?}", e);
                return;
            }
        };

        println!("Testing on device: {}", metal_fft.device_name());
    }

    #[test]
    fn test_metal_fft_small() {
        let mut metal_fft = match MetalFFT::new() {
            Ok(fft) => fft,
            Err(_) => return, // Skip if Metal not available
        };

        // Test small size
        let input = make_test_input(8);
        let result = metal_fft.fft(&input);

        assert!(result.is_ok(), "FFT failed: {:?}", result.err());
        let output = result.unwrap();
        assert_eq!(output.len(), 8);
    }

    #[test]
    fn test_metal_fft_matches_cpu() {
        let mut metal_fft = match MetalFFT::new() {
            Ok(fft) => fft,
            Err(_) => return, // Skip if Metal not available
        };

        for log_n in 2..=10 {
            let n = 1 << log_n;
            let input = make_test_input(n);

            // CPU FFT
            let twiddles = get_twiddles::<GoldilocksField>(log_n as u64, RootsConfig::BitReverse)
                .expect("Failed to get twiddles");
            let cpu_result = cpu_fft::<GoldilocksField, GoldilocksField>(&input, &twiddles)
                .expect("CPU FFT failed");

            // Metal FFT
            let metal_result = metal_fft.fft(&input).expect("Metal FFT failed");

            // Compare results
            for (i, (cpu, metal)) in cpu_result.iter().zip(metal_result.iter()).enumerate() {
                assert_eq!(
                    cpu, metal,
                    "Mismatch at index {} for size {}: CPU={:?}, Metal={:?}",
                    i, n, cpu, metal
                );
            }

            println!("FFT size {} passed", n);
        }
    }
}
