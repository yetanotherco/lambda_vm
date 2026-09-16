use crate::hash::platform_keccak::PlatformKeccak256 as Keccak256;

use super::{
    field_element::FieldElementBackend,
    field_element_vector::{FieldElementPairBackend, FieldElementVectorBackend},
};

// Field element backend definitions
pub type Keccak256Backend<F> = FieldElementBackend<F, Keccak256, 32>;

// Vector of field elements backend definitions
pub type BatchKeccak256Backend<F> = FieldElementVectorBackend<F, Keccak256, 32>;

// Fixed-size pair backends (more efficient for FRI layers)
pub type PairKeccak256Backend<F> = FieldElementPairBackend<F, Keccak256, 32>;

/// RPX256 over a vector of field elements — the algebraic backend, for a proof
/// that is going to be verified inside another proof. See
/// [`crate::hash::rpx`] for why an algebraic hash is worth its host cost, and
/// only there.
pub type BatchRpx256Backend<F> = super::rpx::RpxVectorBackend<F>;
