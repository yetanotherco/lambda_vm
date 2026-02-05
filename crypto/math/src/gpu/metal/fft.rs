//! Metal GPU FFT implementation using Bowers algorithm
//!
//! This module provides GPU-accelerated FFT for the Goldilocks field using
//! the Bowers G network algorithm with 2-layer fusion for improved performance.
//!
//! # Limitations
//!
//! - Maximum FFT size is 2^32 (Goldilocks field's two-adic order)
//! - Input sizes larger than 2^32 elements will return an error
//! - Requires macOS with Metal-capable GPU

use super::device::{MetalContext, MetalError, MetalState};
use crate::field::element::FieldElement;
use crate::field::fields::fft_friendly::u64_goldilocks::GoldilocksField;
use metal::{Buffer, MTLCommandBufferStatus, MTLSize};

/// Goldilocks field primitive 2^32-th root of unity
const GOLDILOCKS_TWO_ADIC_ROOT: u64 = 1753635133440165772;

/// Maximum FFT order supported by Goldilocks field (2-adic order = 32)
pub(crate) const MAX_FFT_ORDER: u64 = 32;

/// Metal-accelerated Bowers FFT for Goldilocks field
pub struct MetalFft {
    ctx: MetalContext,
}

impl MetalFft {
    /// Create a new Metal FFT instance
    pub fn new() -> Result<Self, MetalError> {
        Ok(Self {
            ctx: MetalContext::new()?,
        })
    }

    /// Create from existing context (for reuse)
    pub fn from_context(ctx: MetalContext) -> Self {
        Self { ctx }
    }

    /// Perform FFT on the input data using Bowers algorithm
    ///
    /// The input is modified in-place and returned in bit-reversed order.
    /// Apply bit-reversal permutation afterwards to get natural order.
    ///
    /// # Errors
    ///
    /// Returns `InvalidInput` if:
    /// - Input length is not a power of two
    /// - Input length exceeds 2^32 (Goldilocks field's two-adic order)
    pub fn fft(&self, input: &mut [u64]) -> Result<(), MetalError> {
        let n = input.len();
        if !n.is_power_of_two() {
            return Err(MetalError::InvalidInput(format!(
                "Input length {} is not a power of two",
                n
            )));
        }

        if n <= 1 {
            return Ok(());
        }

        let state = self.ctx.state();
        let log_n = n.trailing_zeros() as usize;
        let order = log_n as u64;

        // Validate order is within Goldilocks field's two-adic order
        if order > MAX_FFT_ORDER {
            return Err(MetalError::InvalidInput(format!(
                "FFT order {} exceeds Goldilocks field's two-adic order {}",
                order, MAX_FFT_ORDER
            )));
        }

        // Get primitive root of unity for this FFT size
        let root = compute_root_of_unity(order);

        // Allocate GPU buffers
        let input_buffer = state.create_buffer_with_data(input)?;

        // Compute layer twiddles on GPU
        let layer_twiddles = self.compute_layer_twiddles(state, order, root)?;

        // Execute Bowers FFT layers
        self.execute_bowers_fft(state, &input_buffer, &layer_twiddles, n, log_n)?;

        // Copy results back
        self.copy_buffer_to_slice(&input_buffer, input);

        Ok(())
    }

    /// Perform FFT and bit-reversal in one call, returning natural-ordered output
    pub fn fft_natural_order(&self, input: &mut [u64]) -> Result<(), MetalError> {
        self.fft(input)?;
        self.bitrev_permutation_inplace(input)?;
        Ok(())
    }

    /// Perform inverse FFT on the input data using Bowers algorithm
    ///
    /// The input is modified in-place and returned in bit-reversed order.
    /// Apply bit-reversal permutation afterwards to get natural order.
    ///
    /// Note: This does NOT include the 1/n scaling factor. Caller must multiply
    /// by 1/n after calling this function to get the true inverse.
    ///
    /// # Errors
    ///
    /// Returns `InvalidInput` if:
    /// - Input length is not a power of two
    /// - Input length exceeds 2^32 (Goldilocks field's two-adic order)
    pub fn ifft(&self, input: &mut [u64]) -> Result<(), MetalError> {
        let n = input.len();
        if !n.is_power_of_two() {
            return Err(MetalError::InvalidInput(format!(
                "Input length {} is not a power of two",
                n
            )));
        }

        if n <= 1 {
            return Ok(());
        }

        let state = self.ctx.state();
        let log_n = n.trailing_zeros() as usize;
        let order = log_n as u64;

        // Validate order is within Goldilocks field's two-adic order
        if order > MAX_FFT_ORDER {
            return Err(MetalError::InvalidInput(format!(
                "FFT order {} exceeds Goldilocks field's two-adic order {}",
                order, MAX_FFT_ORDER
            )));
        }

        // Get INVERSE primitive root of unity for this FFT size
        let inv_root = compute_inverse_root_of_unity(order);

        // Allocate GPU buffers
        let input_buffer = state.create_buffer_with_data(input)?;

        // Compute layer twiddles on GPU using inverse root
        let layer_twiddles = self.compute_layer_twiddles(state, order, inv_root)?;

        // Execute Bowers FFT layers (same algorithm, just with inverse twiddles)
        self.execute_bowers_fft(state, &input_buffer, &layer_twiddles, n, log_n)?;

        // Copy results back
        self.copy_buffer_to_slice(&input_buffer, input);

        Ok(())
    }

    /// Perform inverse FFT and bit-reversal in one call, returning natural-ordered output
    ///
    /// Note: This does NOT include the 1/n scaling factor. Caller must multiply
    /// by 1/n after calling this function to get the true inverse.
    pub fn ifft_natural_order(&self, input: &mut [u64]) -> Result<(), MetalError> {
        self.ifft(input)?;
        self.bitrev_permutation_inplace(input)?;
        Ok(())
    }

    /// Perform batch FFT on multiple polynomials (SoA layout)
    ///
    /// `data` contains `num_polys` polynomials each of length `poly_len`,
    /// stored contiguously in SoA format.
    ///
    /// # Errors
    ///
    /// Returns `InvalidInput` if:
    /// - Polynomial length is not a power of two
    /// - Polynomial length exceeds 2^32 (Goldilocks field's two-adic order)
    /// - Data length doesn't match poly_len * num_polys
    pub fn batch_fft(
        &self,
        data: &mut [u64],
        poly_len: usize,
        num_polys: usize,
    ) -> Result<(), MetalError> {
        if !poly_len.is_power_of_two() {
            return Err(MetalError::InvalidInput(format!(
                "Polynomial length {} is not a power of two",
                poly_len
            )));
        }

        if data.len() != poly_len * num_polys {
            return Err(MetalError::InvalidInput(format!(
                "Data length {} doesn't match poly_len {} * num_polys {}",
                data.len(),
                poly_len,
                num_polys
            )));
        }

        if poly_len <= 1 {
            return Ok(());
        }

        let state = self.ctx.state();
        let log_n = poly_len.trailing_zeros() as usize;
        let order = log_n as u64;

        // Validate order is within Goldilocks field's two-adic order
        if order > MAX_FFT_ORDER {
            return Err(MetalError::InvalidInput(format!(
                "FFT order {} exceeds Goldilocks field's two-adic order {}",
                order, MAX_FFT_ORDER
            )));
        }

        let root = compute_root_of_unity(order);

        let data_buffer = state.create_buffer_with_data(data)?;
        let layer_twiddles = self.compute_layer_twiddles(state, order, root)?;

        // Execute batch Bowers FFT
        self.execute_batch_bowers_fft(
            state,
            &data_buffer,
            &layer_twiddles,
            poly_len,
            num_polys,
            log_n,
        )?;

        // Copy results back
        self.copy_buffer_to_slice(&data_buffer, data);

        Ok(())
    }

    /// Compute layer twiddles on GPU
    fn compute_layer_twiddles(
        &self,
        state: &MetalState,
        order: u64,
        root: u64,
    ) -> Result<Vec<Buffer>, MetalError> {
        let n = 1usize << order;
        let mut layer_twiddles = Vec::with_capacity(order as usize);

        for layer in 0..order as usize {
            let count = n >> (layer + 1);
            let twiddle_buffer =
                state.create_buffer((count * std::mem::size_of::<u64>()) as usize)?;

            let command_buffer = state.command_queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();

            encoder.set_compute_pipeline_state(state.calc_layer_twiddles_pipeline());
            encoder.set_buffer(0, Some(&twiddle_buffer), 0);
            encoder.set_bytes(
                1,
                std::mem::size_of::<u64>() as u64,
                &root as *const u64 as *const _,
            );
            encoder.set_bytes(
                2,
                std::mem::size_of::<u32>() as u64,
                &(layer as u32) as *const u32 as *const _,
            );
            encoder.set_bytes(
                3,
                std::mem::size_of::<u32>() as u64,
                &(count as u32) as *const u32 as *const _,
            );

            let threads_per_grid = MTLSize::new(count as u64, 1, 1);
            let threads_per_group = MTLSize::new(
                state
                    .calc_layer_twiddles_pipeline()
                    .max_total_threads_per_threadgroup()
                    .min(count as u64),
                1,
                1,
            );

            encoder.dispatch_threads(threads_per_grid, threads_per_group);
            encoder.end_encoding();

            command_buffer.commit();
            wait_and_check_completion(&command_buffer)?;

            layer_twiddles.push(twiddle_buffer);
        }

        Ok(layer_twiddles)
    }

    /// Execute Bowers FFT with 2-layer fusion
    fn execute_bowers_fft(
        &self,
        state: &MetalState,
        input_buffer: &Buffer,
        layer_twiddles: &[Buffer],
        n: usize,
        log_n: usize,
    ) -> Result<(), MetalError> {
        let mut layer = 0;

        // Process pairs of layers with 2-layer fusion
        while layer + 1 < log_n {
            let block_size = n >> layer;

            if block_size >= 4 {
                let command_buffer = state.command_queue.new_command_buffer();
                let encoder = command_buffer.new_compute_command_encoder();

                encoder.set_compute_pipeline_state(state.bowers_fused_pipeline());
                encoder.set_buffer(0, Some(input_buffer), 0);
                encoder.set_buffer(1, Some(&layer_twiddles[layer]), 0);
                encoder.set_buffer(2, Some(&layer_twiddles[layer + 1]), 0);
                encoder.set_bytes(
                    3,
                    std::mem::size_of::<u32>() as u64,
                    &(block_size as u32) as *const u32 as *const _,
                );
                encoder.set_bytes(
                    4,
                    std::mem::size_of::<u32>() as u64,
                    &(n as u32) as *const u32 as *const _,
                );

                let threads_per_grid = MTLSize::new((n / 4) as u64, 1, 1);
                let threads_per_group = MTLSize::new(
                    state
                        .bowers_fused_pipeline()
                        .max_total_threads_per_threadgroup()
                        .min((n / 4) as u64),
                    1,
                    1,
                );

                encoder.dispatch_threads(threads_per_grid, threads_per_group);
                encoder.end_encoding();

                command_buffer.commit();
                wait_and_check_completion(&command_buffer)?;

                layer += 2;
            } else {
                break;
            }
        }

        // Process remaining single layers
        while layer < log_n {
            let block_size = n >> layer;

            let command_buffer = state.command_queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();

            encoder.set_compute_pipeline_state(state.bowers_single_pipeline());
            encoder.set_buffer(0, Some(input_buffer), 0);
            encoder.set_buffer(1, Some(&layer_twiddles[layer]), 0);
            encoder.set_bytes(
                2,
                std::mem::size_of::<u32>() as u64,
                &(block_size as u32) as *const u32 as *const _,
            );
            encoder.set_bytes(
                3,
                std::mem::size_of::<u32>() as u64,
                &(n as u32) as *const u32 as *const _,
            );

            let threads_per_grid = MTLSize::new((n / 2) as u64, 1, 1);
            let threads_per_group = MTLSize::new(
                state
                    .bowers_single_pipeline()
                    .max_total_threads_per_threadgroup()
                    .min((n / 2) as u64),
                1,
                1,
            );

            encoder.dispatch_threads(threads_per_grid, threads_per_group);
            encoder.end_encoding();

            command_buffer.commit();
            wait_and_check_completion(&command_buffer)?;

            layer += 1;
        }

        Ok(())
    }

    /// Execute batch Bowers FFT for SoA layout
    fn execute_batch_bowers_fft(
        &self,
        state: &MetalState,
        data_buffer: &Buffer,
        layer_twiddles: &[Buffer],
        poly_len: usize,
        num_polys: usize,
        log_n: usize,
    ) -> Result<(), MetalError> {
        let mut layer = 0;

        // Process pairs of layers with 2-layer fusion
        while layer + 1 < log_n {
            let block_size = poly_len >> layer;

            if block_size >= 4 {
                let command_buffer = state.command_queue.new_command_buffer();
                let encoder = command_buffer.new_compute_command_encoder();

                encoder.set_compute_pipeline_state(state.batch_bowers_fused_pipeline());
                encoder.set_buffer(0, Some(data_buffer), 0);
                encoder.set_buffer(1, Some(&layer_twiddles[layer]), 0);
                encoder.set_buffer(2, Some(&layer_twiddles[layer + 1]), 0);
                encoder.set_bytes(
                    3,
                    std::mem::size_of::<u32>() as u64,
                    &(poly_len as u32) as *const u32 as *const _,
                );
                encoder.set_bytes(
                    4,
                    std::mem::size_of::<u32>() as u64,
                    &(num_polys as u32) as *const u32 as *const _,
                );
                encoder.set_bytes(
                    5,
                    std::mem::size_of::<u32>() as u64,
                    &(block_size as u32) as *const u32 as *const _,
                );

                let threads_per_grid = MTLSize::new((poly_len / 4) as u64, num_polys as u64, 1);
                let threads_per_group = MTLSize::new(
                    state
                        .batch_bowers_fused_pipeline()
                        .max_total_threads_per_threadgroup()
                        .min((poly_len / 4) as u64),
                    1,
                    1,
                );

                encoder.dispatch_threads(threads_per_grid, threads_per_group);
                encoder.end_encoding();

                command_buffer.commit();
                wait_and_check_completion(&command_buffer)?;

                layer += 2;
            } else {
                break;
            }
        }

        // Process remaining single layers for each polynomial
        while layer < log_n {
            let block_size = poly_len >> layer;

            for poly_idx in 0..num_polys {
                let offset = poly_idx * poly_len * std::mem::size_of::<u64>();

                let command_buffer = state.command_queue.new_command_buffer();
                let encoder = command_buffer.new_compute_command_encoder();

                encoder.set_compute_pipeline_state(state.bowers_single_pipeline());
                encoder.set_buffer(0, Some(data_buffer), offset as u64);
                encoder.set_buffer(1, Some(&layer_twiddles[layer]), 0);
                encoder.set_bytes(
                    2,
                    std::mem::size_of::<u32>() as u64,
                    &(block_size as u32) as *const u32 as *const _,
                );
                encoder.set_bytes(
                    3,
                    std::mem::size_of::<u32>() as u64,
                    &(poly_len as u32) as *const u32 as *const _,
                );

                let threads_per_grid = MTLSize::new((poly_len / 2) as u64, 1, 1);
                let threads_per_group = MTLSize::new(
                    state
                        .bowers_single_pipeline()
                        .max_total_threads_per_threadgroup()
                        .min((poly_len / 2) as u64),
                    1,
                    1,
                );

                encoder.dispatch_threads(threads_per_grid, threads_per_group);
                encoder.end_encoding();

                command_buffer.commit();
                wait_and_check_completion(&command_buffer)?;
            }

            layer += 1;
        }

        Ok(())
    }

    /// Perform bit-reversal permutation
    ///
    /// Uses a two-buffer approach to avoid race conditions. The in-place GPU swap
    /// has a race where multiple threads can simultaneously access overlapping
    /// memory locations. Instead, we read from input_buffer and write to
    /// output_buffer at bit-reversed positions.
    pub fn bitrev_permutation_inplace(&self, data: &mut [u64]) -> Result<(), MetalError> {
        let n = data.len();
        if !n.is_power_of_two() {
            return Err(MetalError::InvalidInput(format!(
                "Length {} is not a power of two",
                n
            )));
        }

        if n <= 1 {
            return Ok(());
        }

        let state = self.ctx.state();
        let log_n = n.trailing_zeros() as u32;

        // Two-buffer approach to avoid race conditions in GPU swaps
        let input_buffer = state.create_buffer_with_data(data)?;
        let output_buffer = state.create_buffer(n * std::mem::size_of::<u64>())?;

        let command_buffer = state.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        // Use the two-buffer bit-reversal kernel (not the race-prone in-place one)
        encoder.set_compute_pipeline_state(state.bitrev_pipeline());
        encoder.set_buffer(0, Some(&input_buffer), 0);
        encoder.set_buffer(1, Some(&output_buffer), 0);
        encoder.set_bytes(
            2,
            std::mem::size_of::<u32>() as u64,
            &(n as u32) as *const u32 as *const _,
        );
        encoder.set_bytes(
            3,
            std::mem::size_of::<u32>() as u64,
            &log_n as *const u32 as *const _,
        );

        let threads_per_grid = MTLSize::new(n as u64, 1, 1);
        let threads_per_group = MTLSize::new(
            state
                .bitrev_pipeline()
                .max_total_threads_per_threadgroup()
                .min(n as u64),
            1,
            1,
        );

        encoder.dispatch_threads(threads_per_grid, threads_per_group);
        encoder.end_encoding();

        command_buffer.commit();
        wait_and_check_completion(&command_buffer)?;

        self.copy_buffer_to_slice(&output_buffer, data);

        Ok(())
    }

    /// Copy buffer contents to a slice
    ///
    /// # Safety
    ///
    /// This is safe because:
    /// - The buffer was allocated with StorageModeShared, so contents() returns valid CPU-accessible memory
    /// - The buffer length is guaranteed to be >= dest.len() * size_of::<u64>() by construction
    /// - No other threads access the buffer during copy (we've waited for GPU completion)
    fn copy_buffer_to_slice(&self, buffer: &Buffer, dest: &mut [u64]) {
        let ptr = buffer.contents() as *const u64;
        // SAFETY: See function-level safety documentation
        unsafe {
            std::ptr::copy_nonoverlapping(ptr, dest.as_mut_ptr(), dest.len());
        }
    }
}

/// Compute primitive root of unity for order (FFT size = 2^order)
pub(crate) fn compute_root_of_unity(order: u64) -> u64 {
    // Start with the 2^32-th root and square down to the desired order
    let mut root = GOLDILOCKS_TWO_ADIC_ROOT;
    for _ in order..32 {
        root = goldilocks_square(root);
    }
    root
}

/// Compute inverse primitive root of unity for order (FFT size = 2^order)
fn compute_inverse_root_of_unity(order: u64) -> u64 {
    let root = compute_root_of_unity(order);
    goldilocks_inverse(root)
}

/// Compute multiplicative inverse in Goldilocks field using Fermat's little theorem
/// a^(-1) = a^(p-2) mod p
fn goldilocks_inverse(a: u64) -> u64 {
    const P_MINUS_2: u64 = 0xFFFF_FFFE_FFFF_FFFF; // p - 2
    goldilocks_pow(a, P_MINUS_2)
}

/// Compute a^exp mod p using binary exponentiation
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

/// Multiply two Goldilocks field elements
fn goldilocks_mul(a: u64, b: u64) -> u64 {
    let product = (a as u128) * (b as u128);
    goldilocks_reduce128(product)
}

/// Wait for command buffer completion and check for errors
fn wait_and_check_completion(command_buffer: &metal::CommandBufferRef) -> Result<(), MetalError> {
    command_buffer.wait_until_completed();

    match command_buffer.status() {
        MTLCommandBufferStatus::Completed => Ok(()),
        MTLCommandBufferStatus::Error => Err(MetalError::ExecutionFailed),
        status => {
            // Should not happen after wait_until_completed
            Err(MetalError::InvalidInput(format!(
                "Unexpected command buffer status: {:?}",
                status as u32
            )))
        }
    }
}

/// Square in Goldilocks field (CPU helper for root computation)
pub(crate) fn goldilocks_square(a: u64) -> u64 {
    let product = (a as u128) * (a as u128);
    goldilocks_reduce128(product)
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

/// High-level FFT function using Metal GPU
///
/// Converts FieldElement slice to u64, performs FFT, and converts back.
pub fn metal_fft(input: &mut [FieldElement<GoldilocksField>]) -> Result<(), MetalError> {
    let metal_fft = MetalFft::new()?;

    // Convert to raw u64 values
    let mut data: Vec<u64> = input.iter().map(|fe| *fe.value()).collect();

    // Perform FFT
    metal_fft.fft_natural_order(&mut data)?;

    // Copy back to FieldElements
    for (dest, &val) in input.iter_mut().zip(data.iter()) {
        *dest = FieldElement::from(val);
    }

    Ok(())
}
