use sha2::{Sha256, Sha512};
use sha3::{Keccak256, Keccak512, Sha3_256, Sha3_512};

use super::{
    field_element::FieldElementBackend,
    field_element_vector::{FieldElementPairBackend, FieldElementVectorBackend},
};

// 4-ary field element backends (default)

pub type Sha3_256Backend<F> = FieldElementBackend<F, Sha3_256, 32>;
pub type Keccak256Backend<F> = FieldElementBackend<F, Keccak256, 32>;
pub type Sha2_256Backend<F> = FieldElementBackend<F, Sha256, 32>;

pub type Sha3_512Backend<F> = FieldElementBackend<F, Sha3_512, 64>;
pub type Keccak512Backend<F> = FieldElementBackend<F, Keccak512, 64>;
pub type Sha2_512Backend<F> = FieldElementBackend<F, Sha512, 64>;

// 4-ary vector backends (default)

pub type BatchSha3_256Backend<F> = FieldElementVectorBackend<F, Sha3_256, 32>;
pub type BatchKeccak256Backend<F> = FieldElementVectorBackend<F, Keccak256, 32>;
pub type BatchSha2_256Backend<F> = FieldElementVectorBackend<F, Sha256, 32>;

pub type BatchSha3_512Backend<F> = FieldElementVectorBackend<F, Sha3_512, 64>;
pub type BatchKeccak512Backend<F> = FieldElementVectorBackend<F, Keccak512, 64>;
pub type BatchSha2_512Backend<F> = FieldElementVectorBackend<F, Sha512, 64>;

// Binary (arity=2) backends for Stone compatibility

pub type BinaryKeccak256Backend<F> = FieldElementBackend<F, Keccak256, 32, 2>;
pub type BinaryBatchKeccak256Backend<F> = FieldElementVectorBackend<F, Keccak256, 32, 2>;

// Pair backends (arity=2, fixed-size pair leaves)

pub type PairKeccak256Backend<F> = FieldElementPairBackend<F, Keccak256, 32>;
pub type PairSha3_256Backend<F> = FieldElementPairBackend<F, Sha3_256, 32>;
pub type PairSha2_256Backend<F> = FieldElementPairBackend<F, Sha256, 32>;
