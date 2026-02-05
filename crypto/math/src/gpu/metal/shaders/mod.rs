//! Metal shader compilation utilities
//!
//! The shader source is embedded at compile time using `include_str!`
//! in the device module. This module provides additional utilities
//! for shader management if needed.

/// Path to the Goldilocks FFT shader source
pub const GOLDILOCKS_FFT_SHADER: &str = include_str!("goldilocks_fft.metal");
