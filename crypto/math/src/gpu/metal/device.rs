//! Metal device management and state.
//!
//! Handles GPU device initialization, command queue creation, and buffer management.

use super::errors::MetalError;
use alloc::string::ToString;
use metal::{Buffer, CommandQueue, Device, Library, MTLResourceOptions};

/// Maximum supported FFT size (2^24 = 16M elements)
pub const MAX_FFT_SIZE: usize = 1 << 24;

/// Threshold below which CPU FFT is likely faster due to transfer overhead
pub const GPU_THRESHOLD: usize = 1 << 12; // 4096 elements

/// Metal state containing device, command queue, and compiled shaders.
pub struct MetalState {
    /// The Metal device (GPU)
    pub device: Device,
    /// Command queue for submitting work
    pub command_queue: CommandQueue,
    /// Compiled shader library
    pub library: Library,
}

impl MetalState {
    /// Create a new Metal state with the default system GPU.
    pub fn new() -> Result<Self, MetalError> {
        // Get the default Metal device
        let device = Device::system_default().ok_or(MetalError::DeviceNotFound)?;

        // Create command queue
        let command_queue = device.new_command_queue();

        // Compile shaders
        let library = Self::compile_shaders(&device)?;

        Ok(Self {
            device,
            command_queue,
            library,
        })
    }

    /// Compile the Metal shaders for FFT operations.
    fn compile_shaders(device: &Device) -> Result<Library, MetalError> {
        let shader_source = include_str!("shaders/goldilocks_fft.metal");

        let options = metal::CompileOptions::new();

        device
            .new_library_with_source(shader_source, &options)
            .map_err(|e| MetalError::ShaderCompilationFailed(e.to_string()))
    }

    /// Create a buffer for field elements (u64 values).
    ///
    /// Uses shared memory mode for efficient CPU-GPU data transfer on Apple Silicon.
    pub fn create_buffer(&self, size: usize) -> Result<Buffer, MetalError> {
        let byte_size = size * core::mem::size_of::<u64>();

        if byte_size == 0 {
            return Err(MetalError::BufferAllocationFailed(
                "Cannot allocate zero-size buffer".to_string(),
            ));
        }

        // Use shared storage mode for Apple Silicon unified memory
        let options = MTLResourceOptions::StorageModeShared;

        let buffer = self.device.new_buffer(byte_size as u64, options);

        Ok(buffer)
    }

    /// Create a buffer initialized with data.
    pub fn create_buffer_with_data(&self, data: &[u64]) -> Result<Buffer, MetalError> {
        let byte_size = data.len() * core::mem::size_of::<u64>();

        if byte_size == 0 {
            return Err(MetalError::BufferAllocationFailed(
                "Cannot allocate zero-size buffer".to_string(),
            ));
        }

        let options = MTLResourceOptions::StorageModeShared;

        let buffer =
            self.device
                .new_buffer_with_data(data.as_ptr() as *const _, byte_size as u64, options);

        Ok(buffer)
    }

    /// Read data from a GPU buffer back to CPU memory.
    pub fn read_buffer(&self, buffer: &Buffer, output: &mut [u64]) -> Result<(), MetalError> {
        let byte_size = output.len() * core::mem::size_of::<u64>();

        if buffer.length() < byte_size as u64 {
            return Err(MetalError::DataTransferFailed(
                "Buffer too small for output".to_string(),
            ));
        }

        let ptr = buffer.contents() as *const u64;
        unsafe {
            core::ptr::copy_nonoverlapping(ptr, output.as_mut_ptr(), output.len());
        }

        Ok(())
    }

    /// Get device name for logging/debugging.
    pub fn device_name(&self) -> &str {
        self.device.name()
    }

    /// Check if this is Apple Silicon (unified memory architecture).
    pub fn is_unified_memory(&self) -> bool {
        self.device.has_unified_memory()
    }

    /// Get recommended threadgroup size for compute operations.
    pub fn recommended_threadgroup_size(&self) -> usize {
        // Apple Silicon typically works well with 256 threads per threadgroup
        // for compute operations
        256
    }

    /// Get maximum threadgroup memory size.
    pub fn max_threadgroup_memory(&self) -> u64 {
        self.device.max_threadgroup_memory_length()
    }
}

/// Thread-safe wrapper for MetalState.
pub type SharedMetalState = alloc::sync::Arc<MetalState>;

/// Create a shared Metal state that can be used across threads.
pub fn create_shared_state() -> Result<SharedMetalState, MetalError> {
    Ok(alloc::sync::Arc::new(MetalState::new()?))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metal_device_creation() {
        // This test will only pass on systems with Metal support
        if let Ok(state) = MetalState::new() {
            println!("Metal device: {}", state.device_name());
            println!("Unified memory: {}", state.is_unified_memory());
            println!(
                "Max threadgroup memory: {} bytes",
                state.max_threadgroup_memory()
            );
        } else {
            println!("Metal not available on this system");
        }
    }

    #[test]
    fn test_buffer_creation() {
        if let Ok(state) = MetalState::new() {
            let buffer = state.create_buffer(1024).expect("Failed to create buffer");
            assert_eq!(buffer.length(), 1024 * 8); // 1024 u64s = 8192 bytes
        }
    }

    #[test]
    fn test_buffer_roundtrip() {
        if let Ok(state) = MetalState::new() {
            let input: Vec<u64> = (0..1024).collect();
            let buffer = state
                .create_buffer_with_data(&input)
                .expect("Failed to create buffer");

            let mut output = vec![0u64; 1024];
            state
                .read_buffer(&buffer, &mut output)
                .expect("Failed to read buffer");

            assert_eq!(input, output);
        }
    }
}
