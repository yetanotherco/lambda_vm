/// Implemenation of the Babybear Prime Field p = 2^31 - 2^27 + 1
pub mod babybear;
/// Implementation of the quadratic extension of the babybear field
pub mod quadratic_babybear;
/// Implementation of the extension of degree 4 of the babybear field using u64.
pub mod quartic_babybear;
/// Implementation of two-adic prime field over 256 bit unsigned integers.
pub mod stark_252_prime_field;
/// Implemenation of the Mersenne Prime field p = 2^31 - 1
pub mod u64_mersenne_montgomery_field;

/// Inmplementation of the Babybear Prime Field p = 2^31 - 2^27 + 1 using u32
pub mod babybear_u32;

/// Implementation of the extension of degree 4 of the babybear field using u32.
pub mod quartic_babybear_u32;

/// Quadratic and cubic extensions of the Goldilocks field
pub mod extensions_goldilocks;
/// Optimized Goldilocks field p = 2^64 - 2^32 + 1 (no Montgomery form)
pub mod u64_goldilocks;

/// ARM64 assembly optimizations for Goldilocks extensions (enabled with asm-arm64 feature)
#[cfg(all(feature = "asm-arm64", target_arch = "aarch64"))]
pub mod goldilocks_extensions_asm;
/// ARM64 assembly optimizations for Goldilocks field (enabled with asm-arm64 feature)
#[cfg(all(feature = "asm-arm64", target_arch = "aarch64"))]
pub mod u64_goldilocks_asm;
