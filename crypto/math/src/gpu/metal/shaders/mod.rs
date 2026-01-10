//! Metal shader management and pipeline creation.
//!
//! This module handles loading and compiling Metal shaders for FFT operations.

use super::errors::MetalError;
use alloc::format;
use metal::{ComputePipelineState, Device, Library};

/// Names of the kernel functions in the Metal shader
pub const KERNEL_FFT_RADIX2_STAGE: &str = "fft_radix2_stage";
pub const KERNEL_FFT_RADIX4_STAGE: &str = "fft_radix4_stage";
pub const KERNEL_BIT_REVERSE_PERMUTE: &str = "bit_reverse_permute";
pub const KERNEL_GENERATE_TWIDDLES: &str = "generate_twiddles";
pub const KERNEL_FFT_RADIX2_SMALL: &str = "fft_radix2_small";
pub const KERNEL_CANONICALIZE: &str = "canonicalize_buffer";
pub const KERNEL_FFT_STOCKHAM_STAGE: &str = "fft_stockham_stage";
pub const KERNEL_TEST_MULTIPLY: &str = "test_multiply";
pub const KERNEL_TEST_ADD: &str = "test_add";
pub const KERNEL_TEST_BUTTERFLY: &str = "test_butterfly";

/// Container for all FFT compute pipelines.
pub struct FFTPipelines {
    /// Pipeline for single FFT stage (radix-2)
    pub fft_radix2_stage: ComputePipelineState,
    /// Pipeline for single FFT stage (radix-4)
    pub fft_radix4_stage: ComputePipelineState,
    /// Pipeline for bit-reverse permutation
    pub bit_reverse_permute: ComputePipelineState,
    /// Pipeline for twiddle factor generation
    pub generate_twiddles: ComputePipelineState,
    /// Pipeline for small FFT (shared memory)
    pub fft_radix2_small: ComputePipelineState,
    /// Pipeline for canonicalization
    pub canonicalize: ComputePipelineState,
    /// Pipeline for Stockham FFT stage (auto-sorting, out-of-place)
    pub fft_stockham_stage: ComputePipelineState,
}

impl FFTPipelines {
    /// Create all FFT pipelines from a compiled shader library.
    pub fn new(device: &Device, library: &Library) -> Result<Self, MetalError> {
        Ok(Self {
            fft_radix2_stage: create_pipeline(device, library, KERNEL_FFT_RADIX2_STAGE)?,
            fft_radix4_stage: create_pipeline(device, library, KERNEL_FFT_RADIX4_STAGE)?,
            bit_reverse_permute: create_pipeline(device, library, KERNEL_BIT_REVERSE_PERMUTE)?,
            generate_twiddles: create_pipeline(device, library, KERNEL_GENERATE_TWIDDLES)?,
            fft_radix2_small: create_pipeline(device, library, KERNEL_FFT_RADIX2_SMALL)?,
            canonicalize: create_pipeline(device, library, KERNEL_CANONICALIZE)?,
            fft_stockham_stage: create_pipeline(device, library, KERNEL_FFT_STOCKHAM_STAGE)?,
        })
    }
}

/// Create a compute pipeline for a specific kernel function.
fn create_pipeline(
    device: &Device,
    library: &Library,
    kernel_name: &str,
) -> Result<ComputePipelineState, MetalError> {
    let function = library
        .get_function(kernel_name, None)
        .map_err(|e| MetalError::ShaderCompilationFailed(
            format!("Failed to get function '{}': {}", kernel_name, e)
        ))?;

    device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| MetalError::PipelineCreationFailed(
            format!("Failed to create pipeline for '{}': {}", kernel_name, e)
        ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::metal::device::MetalState;

    #[test]
    fn test_pipeline_creation() {
        if let Ok(state) = MetalState::new() {
            let pipelines = FFTPipelines::new(&state.device, &state.library);
            assert!(pipelines.is_ok(), "Failed to create pipelines: {:?}", pipelines.err());
        }
    }
}
