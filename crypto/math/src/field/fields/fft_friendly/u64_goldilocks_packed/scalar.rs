//! Scalar fallback for packed Goldilocks — used on platforms without SIMD.
//! Simply re-exports ScalarPacked<GoldilocksField> from the packed module.

use crate::field::fields::fft_friendly::u64_goldilocks::GoldilocksField;
pub use crate::field::packed::ScalarPacked;

#[allow(dead_code)]
pub type PackedGoldilocksScalar = ScalarPacked<GoldilocksField>;
