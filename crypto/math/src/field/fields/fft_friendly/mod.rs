/// Quadratic and cubic extensions of the Goldilocks field
pub mod extensions_goldilocks;
/// Optimized Goldilocks field p = 2^64 - 2^32 + 1 (no Montgomery form)
pub mod u64_goldilocks;

/// NEON 2-wide packed Goldilocks arithmetic
#[cfg(target_arch = "aarch64")]
pub mod goldilocks_neon;

/// AVX2 4-wide packed Goldilocks arithmetic
#[cfg(target_arch = "x86_64")]
pub mod goldilocks_avx2;
