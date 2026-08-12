use crate::hash::platform_keccak::PlatformKeccak256 as Keccak256;

use super::field_element_vector::{FieldElementPairBackend, FieldElementVectorBackend};

// Vector of field elements backend definitions
pub type BatchKeccak256Backend<F> = FieldElementVectorBackend<F, Keccak256, 32>;

// Fixed-size pair backends (more efficient for FRI layers)
pub type PairKeccak256Backend<F> = FieldElementPairBackend<F, Keccak256, 32>;
