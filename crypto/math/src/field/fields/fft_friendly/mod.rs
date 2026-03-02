/// Quadratic and cubic extensions of the Goldilocks field
pub mod extensions_goldilocks;
/// AVX2-accelerated Goldilocks arithmetic (4-wide packed operations)
#[cfg(target_arch = "x86_64")]
pub mod goldilocks_avx2;
/// Optimized Goldilocks field p = 2^64 - 2^32 + 1 (no Montgomery form)
pub mod u64_goldilocks;
