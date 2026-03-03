use sha3::Keccak256;

use super::{field_element::FieldElementBackend, field_element_vector::FieldElementVectorBackend};

// 4-ary Keccak256 backends (used by quaternary-merkle feature)

pub type Keccak256Backend<F> = FieldElementBackend<F, Keccak256, 32>;
pub type BatchKeccak256Backend<F> = FieldElementVectorBackend<F, Keccak256, 32>;

// Binary (arity=2) Keccak256 backends (used by default / Stone compatibility)

pub type BinaryKeccak256Backend<F> = FieldElementBackend<F, Keccak256, 32, 2>;
pub type BinaryBatchKeccak256Backend<F> = FieldElementVectorBackend<F, Keccak256, 32, 2>;
