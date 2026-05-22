use sha3::Keccak256;

use super::{
    field_element::FieldElementBackend,
    field_element_vector::{
        FieldElementPairBackend, FieldElementQuadBackend, FieldElementVectorBackend,
    },
};

// Field element backend definitions
pub type Keccak256Backend<F> = FieldElementBackend<F, Keccak256, 32>;

// Vector of field elements backend definitions
pub type BatchKeccak256Backend<F> = FieldElementVectorBackend<F, Keccak256, 32>;

// Fixed-size pair backends (more efficient for FRI layers)
pub type PairKeccak256Backend<F> = FieldElementPairBackend<F, Keccak256, 32>;

// Fixed-size quad backend: the leaf for arity-4 FRI fold orbits.
pub type QuadKeccak256Backend<F> = FieldElementQuadBackend<F, Keccak256, 32>;
