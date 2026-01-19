/// Implemenation of the Babybear Prime Field p = 2^31 - 2^27 + 1
pub mod babybear;
/// Implementation of the quadratic and cubic extensions of the Goldilocks field (Montgomery)
pub mod extensions_goldilocks;
/// Implementation of the quadratic and cubic extensions of the native Goldilocks field
pub mod extensions_goldilocks_native;
/// Differential fuzzing for ARM64 assembly Goldilocks operations
#[cfg(all(feature = "asm-arm64", target_arch = "aarch64"))]
pub mod goldilocks_asm_fuzzing;
/// ARM64 assembly-optimized Goldilocks extension field operations (Fp2, Fp3)
#[cfg(all(feature = "asm-arm64", target_arch = "aarch64"))]
pub mod goldilocks_extensions_asm;
/// Implementation of the quadratic extension of the babybear field
pub mod quadratic_babybear;
/// Implementation of the extension of degree 4 of the babybear field using u64.
pub mod quartic_babybear;
/// Implementation of two-adic prime field over 256 bit unsigned integers.
pub mod stark_252_prime_field;
/// Implemenation of the Goldilocks Prime Field p = 2^64 - 2^32 + 1 (Montgomery form)
pub mod u64_goldilocks;
/// ARM64 assembly-optimized Goldilocks field operations (Apple Silicon)
/// Uses SBC mask trick for add/sub (~10% faster), native Rust for mul (LLVM optimal)
#[cfg(all(feature = "asm-arm64", target_arch = "aarch64"))]
pub mod u64_goldilocks_asm;
/// Optimized Goldilocks Prime Field using native u64 arithmetic (no Montgomery)
pub mod u64_goldilocks_native;
/// Implemenation of the Mersenne Prime field p = 2^31 - 1
pub mod u64_mersenne_montgomery_field;

/// Inmplementation of the Babybear Prime Field p = 2^31 - 2^27 + 1 using u32
pub mod babybear_u32;

/// Implementation of the extension of degree 4 of the babybear field using u32.
pub mod quartic_babybear_u32;
