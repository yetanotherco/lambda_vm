//! Metal GPU backend for accelerated cryptographic operations.
//!
//! This module provides GPU-accelerated FFT/NTT operations using Apple's Metal API.
//! It targets Apple Silicon (M1/M2/M3) and Intel Macs with discrete GPUs.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    Metal FFT Module                         │
//! ├─────────────────────────────────────────────────────────────┤
//! │  Device       │  Shader      │  FFT           │  Errors    │
//! │  └─ init      │  └─ compile  │  └─ radix2     │            │
//! │  └─ buffers   │  └─ field    │  └─ twiddles   │            │
//! │  └─ commands  │     ops      │  └─ bit_rev    │            │
//! └─────────────────────────────────────────────────────────────┘
//! ```

pub mod device;
pub mod errors;
pub mod fft;
pub mod fuzzing;
pub mod shaders;

pub use device::MetalState;
pub use errors::MetalError;
pub use fft::MetalFFT;
pub use fuzzing::{DifferentialFuzzer, FuzzConfig, FuzzReport, FuzzResult};
