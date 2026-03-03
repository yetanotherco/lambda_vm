//! Scalar fallback for packed Goldilocks — used on platforms without SIMD.
//! Simply re-exports ScalarPacked<GoldilocksField> from the packed module.

pub use crate::field::packed::ScalarPacked;
use crate::field::fields::fft_friendly::u64_goldilocks::GoldilocksField;

pub type PackedGoldilocksScalar = ScalarPacked<GoldilocksField>;
