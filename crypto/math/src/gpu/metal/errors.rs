//! Error types for Metal GPU operations.

use alloc::string::String;
use core::fmt;

/// Errors that can occur during Metal GPU operations.
#[derive(Debug, Clone)]
pub enum MetalError {
    /// No Metal-compatible GPU device found
    DeviceNotFound,
    /// Failed to create Metal device
    DeviceCreationFailed(String),
    /// Failed to create command queue
    CommandQueueCreationFailed,
    /// Failed to compile shader
    ShaderCompilationFailed(String),
    /// Failed to create compute pipeline
    PipelineCreationFailed(String),
    /// Failed to allocate GPU buffer
    BufferAllocationFailed(String),
    /// Invalid input size (must be power of two)
    InvalidInputSize(usize),
    /// Input size exceeds maximum supported
    InputTooLarge { size: usize, max: usize },
    /// Command buffer execution failed
    ExecutionFailed(String),
    /// Twiddle factor generation failed
    TwiddleGenerationFailed(String),
    /// Data transfer failed
    DataTransferFailed(String),
    /// Invalid input
    InvalidInput(String),
}

impl fmt::Display for MetalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MetalError::DeviceNotFound => {
                write!(f, "No Metal-compatible GPU device found")
            }
            MetalError::DeviceCreationFailed(msg) => {
                write!(f, "Failed to create Metal device: {}", msg)
            }
            MetalError::CommandQueueCreationFailed => {
                write!(f, "Failed to create command queue")
            }
            MetalError::ShaderCompilationFailed(msg) => {
                write!(f, "Failed to compile shader: {}", msg)
            }
            MetalError::PipelineCreationFailed(msg) => {
                write!(f, "Failed to create compute pipeline: {}", msg)
            }
            MetalError::BufferAllocationFailed(msg) => {
                write!(f, "Failed to allocate GPU buffer: {}", msg)
            }
            MetalError::InvalidInputSize(size) => {
                write!(f, "Invalid input size {}: must be power of two", size)
            }
            MetalError::InputTooLarge { size, max } => {
                write!(f, "Input size {} exceeds maximum {}", size, max)
            }
            MetalError::ExecutionFailed(msg) => {
                write!(f, "Command buffer execution failed: {}", msg)
            }
            MetalError::TwiddleGenerationFailed(msg) => {
                write!(f, "Twiddle factor generation failed: {}", msg)
            }
            MetalError::DataTransferFailed(msg) => {
                write!(f, "Data transfer failed: {}", msg)
            }
            MetalError::InvalidInput(msg) => {
                write!(f, "Invalid input: {}", msg)
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for MetalError {}
