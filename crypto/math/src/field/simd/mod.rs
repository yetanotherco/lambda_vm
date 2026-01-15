//! SIMD-accelerated field arithmetic.
//!
//! This module provides vectorized implementations of field operations
//! for improved performance on modern CPUs with SIMD support.
//!
//! Currently supported:
//! - ARM NEON (aarch64): 2-wide Goldilocks operations via `uint64x2_t`
//!
//! # Usage
//!
//! The SIMD types process multiple field elements in parallel:
//!
//! ```ignore
//! use math::field::simd::PackedGoldilocks2;
//!
//! let a = PackedGoldilocks2::from_array([1, 2]);
//! let b = PackedGoldilocks2::from_array([3, 4]);
//! let c = a + b;  // [4, 6]
//! ```

#[cfg(target_arch = "aarch64")]
pub mod packed_goldilocks;

#[cfg(target_arch = "aarch64")]
pub use packed_goldilocks::PackedGoldilocks2;

// Fallback for non-aarch64 architectures
#[cfg(not(target_arch = "aarch64"))]
pub mod packed_goldilocks_scalar;

#[cfg(not(target_arch = "aarch64"))]
pub use packed_goldilocks_scalar::PackedGoldilocks2;

// Quadratic extension Fp2
pub mod packed_fp2;
pub use packed_fp2::{Fp2Raw, PackedFp2x2};

// Cubic extension Fp3
pub mod packed_fp3;
pub use packed_fp3::{Fp3Raw, PackedFp3x2};
