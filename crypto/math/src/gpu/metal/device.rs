//! Metal device management for GPU FFT operations
//!
//! Provides abstraction over Metal device initialization, buffer management,
//! and shader compilation for Goldilocks field FFT.

use metal::{
    Buffer, CommandQueue, CompileOptions, ComputePipelineState, Device, Library, MTLResourceOptions,
};
use std::sync::Arc;

/// Errors that can occur during Metal operations
#[derive(Debug)]
pub enum MetalError {
    /// No Metal-capable device found
    NoDevice,
    /// Failed to create command queue
    CommandQueueCreation,
    /// Failed to compile Metal shader library
    ShaderCompilation(String),
    /// Failed to create compute pipeline
    PipelineCreation(String),
    /// Buffer allocation failed
    BufferAllocation,
    /// Command buffer execution failed
    ExecutionFailed,
    /// Invalid input (e.g., not power of two)
    InvalidInput(String),
}

impl std::fmt::Display for MetalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MetalError::NoDevice => write!(f, "No Metal-capable device found"),
            MetalError::CommandQueueCreation => write!(f, "Failed to create command queue"),
            MetalError::ShaderCompilation(msg) => write!(f, "Shader compilation failed: {}", msg),
            MetalError::PipelineCreation(msg) => write!(f, "Pipeline creation failed: {}", msg),
            MetalError::BufferAllocation => write!(f, "Buffer allocation failed"),
            MetalError::ExecutionFailed => write!(f, "Command execution failed"),
            MetalError::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
        }
    }
}

impl std::error::Error for MetalError {}

/// Metal state containing device, command queue, and compiled pipelines
pub struct MetalState {
    pub device: Device,
    pub command_queue: CommandQueue,
    pub library: Library,
    pipelines: MetalPipelines,
}

/// Pre-compiled compute pipelines for FFT operations
struct MetalPipelines {
    radix2_dit_butterfly: ComputePipelineState,
    bowers_fft_fused_layer: ComputePipelineState,
    bowers_fft_single_layer: ComputePipelineState,
    bitrev_permutation: ComputePipelineState,
    bitrev_permutation_inplace: ComputePipelineState,
    calc_twiddles: ComputePipelineState,
    calc_layer_twiddles: ComputePipelineState,
    batch_bowers_fft_fused_layer: ComputePipelineState,
    batch_bowers_fft_single_layer: ComputePipelineState,
}

impl MetalState {
    /// Create a new Metal state with compiled FFT pipelines
    pub fn new() -> Result<Self, MetalError> {
        let device = Device::system_default().ok_or(MetalError::NoDevice)?;
        let command_queue = device.new_command_queue();

        // Compile the shader library
        let shader_source = include_str!("shaders/goldilocks_fft.metal");
        let options = CompileOptions::new();
        let library = device
            .new_library_with_source(shader_source, &options)
            .map_err(|e| MetalError::ShaderCompilation(e.to_string()))?;

        // Create compute pipelines
        let pipelines = MetalPipelines {
            radix2_dit_butterfly: Self::create_pipeline(&device, &library, "radix2_dit_butterfly")?,
            bowers_fft_fused_layer: Self::create_pipeline(
                &device,
                &library,
                "bowers_fft_fused_layer",
            )?,
            bowers_fft_single_layer: Self::create_pipeline(
                &device,
                &library,
                "bowers_fft_single_layer",
            )?,
            bitrev_permutation: Self::create_pipeline(&device, &library, "bitrev_permutation")?,
            bitrev_permutation_inplace: Self::create_pipeline(
                &device,
                &library,
                "bitrev_permutation_inplace",
            )?,
            calc_twiddles: Self::create_pipeline(&device, &library, "calc_twiddles")?,
            calc_layer_twiddles: Self::create_pipeline(&device, &library, "calc_layer_twiddles")?,
            batch_bowers_fft_fused_layer: Self::create_pipeline(
                &device,
                &library,
                "batch_bowers_fft_fused_layer",
            )?,
            batch_bowers_fft_single_layer: Self::create_pipeline(
                &device,
                &library,
                "batch_bowers_fft_single_layer",
            )?,
        };

        Ok(Self {
            device,
            command_queue,
            library,
            pipelines,
        })
    }

    fn create_pipeline(
        device: &Device,
        library: &Library,
        name: &str,
    ) -> Result<ComputePipelineState, MetalError> {
        let function = library
            .get_function(name, None)
            .map_err(|e| MetalError::PipelineCreation(format!("{}: {}", name, e)))?;
        device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|e| MetalError::PipelineCreation(format!("{}: {}", name, e)))
    }

    /// Allocate a buffer on the GPU
    pub fn create_buffer(&self, size: usize) -> Result<Buffer, MetalError> {
        let buffer = self
            .device
            .new_buffer(size as u64, MTLResourceOptions::StorageModeShared);
        Ok(buffer)
    }

    /// Allocate a buffer initialized with data
    pub fn create_buffer_with_data(&self, data: &[u64]) -> Result<Buffer, MetalError> {
        let buffer = self.device.new_buffer_with_data(
            data.as_ptr() as *const _,
            (data.len() * std::mem::size_of::<u64>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        Ok(buffer)
    }

    /// Get pipeline reference
    pub fn radix2_pipeline(&self) -> &ComputePipelineState {
        &self.pipelines.radix2_dit_butterfly
    }

    pub fn bowers_fused_pipeline(&self) -> &ComputePipelineState {
        &self.pipelines.bowers_fft_fused_layer
    }

    pub fn bowers_single_pipeline(&self) -> &ComputePipelineState {
        &self.pipelines.bowers_fft_single_layer
    }

    pub fn bitrev_pipeline(&self) -> &ComputePipelineState {
        &self.pipelines.bitrev_permutation
    }

    pub fn bitrev_inplace_pipeline(&self) -> &ComputePipelineState {
        &self.pipelines.bitrev_permutation_inplace
    }

    pub fn calc_twiddles_pipeline(&self) -> &ComputePipelineState {
        &self.pipelines.calc_twiddles
    }

    pub fn calc_layer_twiddles_pipeline(&self) -> &ComputePipelineState {
        &self.pipelines.calc_layer_twiddles
    }

    pub fn batch_bowers_fused_pipeline(&self) -> &ComputePipelineState {
        &self.pipelines.batch_bowers_fft_fused_layer
    }

    pub fn batch_bowers_single_pipeline(&self) -> &ComputePipelineState {
        &self.pipelines.batch_bowers_fft_single_layer
    }
}

/// Thread-safe wrapper for MetalState
pub struct MetalContext {
    state: Arc<MetalState>,
}

impl MetalContext {
    pub fn new() -> Result<Self, MetalError> {
        Ok(Self {
            state: Arc::new(MetalState::new()?),
        })
    }

    pub fn state(&self) -> &MetalState {
        &self.state
    }
}

impl Clone for MetalContext {
    fn clone(&self) -> Self {
        Self {
            state: Arc::clone(&self.state),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metal_state_creation() {
        // This test will fail on non-macOS or systems without Metal
        match MetalState::new() {
            Ok(state) => {
                println!("Metal device: {:?}", state.device.name());
            }
            Err(MetalError::NoDevice) => {
                println!("No Metal device available (expected on non-macOS)");
            }
            Err(e) => {
                panic!("Unexpected error: {:?}", e);
            }
        }
    }
}
