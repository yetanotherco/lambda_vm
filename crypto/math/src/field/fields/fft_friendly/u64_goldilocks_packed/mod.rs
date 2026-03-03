//! Platform-dispatched packed Goldilocks field types.
//!
//! Selects the best SIMD implementation at compile time:
//! - AVX2 (x86-64): WIDTH=4
//! - AVX-512 (x86-64): WIDTH=8
//! - NEON (AArch64): WIDTH=2
//! - Scalar fallback: WIDTH=1

mod scalar;
pub mod fp3;

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    not(target_feature = "avx512f")
))]
mod x86_64_avx2;

// TODO: Task 5 — AVX-512
// #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
// mod x86_64_avx512;

#[cfg(target_arch = "aarch64")]
mod aarch64_neon;

// Re-export the platform-appropriate packed type as `PackedGoldilocks`.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    not(target_feature = "avx512f")
))]
pub use x86_64_avx2::PackedGoldilocksAVX2 as PackedGoldilocks;

#[cfg(target_arch = "aarch64")]
pub use aarch64_neon::PackedGoldilocksNeon as PackedGoldilocks;

// Scalar fallback for all other platforms
#[cfg(not(any(
    all(
        target_arch = "x86_64",
        target_feature = "avx2",
        not(target_feature = "avx512f")
    ),
    // all(target_arch = "x86_64", target_feature = "avx512f"),
    target_arch = "aarch64",
)))]
pub use scalar::PackedGoldilocksScalar as PackedGoldilocks;
