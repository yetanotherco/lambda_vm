//! Metal GPU acceleration for cryptographic operations
//!
//! This module provides GPU-accelerated implementations of field operations
//! and FFT using Apple's Metal framework.
//!
//! # Features
//!
//! - Goldilocks field arithmetic on GPU
//! - Bowers FFT with 2-layer fusion
//! - Batch FFT for SoA layouts
//! - In-place bit-reversal permutation
//!
//! # Requirements
//!
//! - macOS with Metal-capable GPU
//! - Requires the `metal` feature flag
//!
//! # Example
//!
//! ```ignore
//! use math::gpu::metal::fft::MetalFft;
//!
//! let fft = MetalFft::new()?;
//! let mut data: Vec<u64> = (0..1024).collect();
//! fft.fft_natural_order(&mut data)?;
//! ```

pub mod device;
pub mod fft;

pub use device::{MetalContext, MetalError, MetalState};
pub use fft::{MetalFft, metal_fft};
