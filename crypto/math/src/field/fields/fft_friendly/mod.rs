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

/// AVX2-optimized Goldilocks field operations (x86/x86_64 only)
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub mod u64_goldilocks_avx;
