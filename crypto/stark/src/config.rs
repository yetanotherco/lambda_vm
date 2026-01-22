use crypto::merkle_tree::merkle::MerkleTree;

// ============================================================
// Keccak256 Configuration (default)
// ============================================================
#[cfg(not(feature = "poseidon2"))]
mod keccak_config {
    use super::*;
    use crypto::merkle_tree::backends::types::{
        BatchKeccak256Backend, Keccak256Backend, PairKeccak256Backend,
    };

    pub type FriMerkleTreeBackend<F> = Keccak256Backend<F>;
    pub type FriMerkleTree<F> = MerkleTree<FriMerkleTreeBackend<F>>;

    /// If using hashes with 256-bit security, commitment size should be 32
    /// If using hashes with 512-bit security, commitment size should be 64
    pub const COMMITMENT_SIZE: usize = 32;
    pub type Commitment = [u8; COMMITMENT_SIZE];

    pub type BatchedMerkleTreeBackend<F> = BatchKeccak256Backend<F>;
    pub type BatchedMerkleTree<F> = MerkleTree<BatchedMerkleTreeBackend<F>>;

    /// FRI layer uses fixed-size pairs for efficiency (avoids Vec allocation per pair)
    pub type FriLayerMerkleTreeBackend<F> = PairKeccak256Backend<F>;
    pub type FriLayerMerkleTree<F> = MerkleTree<FriLayerMerkleTreeBackend<F>>;
}

#[cfg(not(feature = "poseidon2"))]
pub use keccak_config::*;

// ============================================================
// Poseidon2 Configuration (feature = "poseidon2")
// ============================================================
//
// IMPORTANT: Poseidon2 backends are Goldilocks-field specific.
// Using --features poseidon2 restricts the STARK to Goldilocks field only.
// Attempting to use with Stark252 or other fields will cause compilation errors.
//
// The types below use PhantomData to carry generic type information for API
// compatibility, but internally always use Goldilocks-specific backends.
#[cfg(feature = "poseidon2")]
mod poseidon2_config {
    use super::*;
    use crypto::hash::poseidon2::Fp;
    use crypto::merkle_tree::backends::types::{BatchPoseidon2Backend, Poseidon2Backend};
    use math::traits::AsBytes;
    use std::marker::PhantomData;
    use std::ops::Deref;

    // Re-export Fp for use in other modules that need the actual Goldilocks type
    pub use crypto::hash::poseidon2::Fp as PoseidonFp;

    // Re-export backend types for use in fri/mod.rs and other modules
    pub use crypto::merkle_tree::backends::types::PairPoseidon2Backend;

    /// Commitment type for Poseidon2: a single Goldilocks field element
    pub type Commitment = Fp;

    // Type aliases for actual backends - no generic parameter
    pub type FriMerkleTreeBackendInner = Poseidon2Backend;
    pub type FriMerkleTreeInner = MerkleTree<Poseidon2Backend>;

    pub type BatchedMerkleTreeBackendInner = BatchPoseidon2Backend;
    pub type BatchedMerkleTreeInner = MerkleTree<BatchPoseidon2Backend>;

    pub type FriLayerMerkleTreeBackendInner = BatchPoseidon2Backend;
    pub type FriLayerMerkleTreeInner = MerkleTree<BatchPoseidon2Backend>;

    // ============================================================
    // Wrapper types with PhantomData for API compatibility
    // ============================================================
    // These allow code using `BatchedMerkleTree<F>` to compile, while
    // internally using the Goldilocks-specific Poseidon2 backend.
    // The wrappers implement Deref to transparently expose inner fields.

    /// FRI Merkle tree backend wrapper
    pub struct FriMerkleTreeBackend<F>(PhantomData<F>);

    /// FRI Merkle tree wrapper with phantom type parameter
    pub struct FriMerkleTree<F> {
        inner: FriMerkleTreeInner,
        _phantom: PhantomData<F>,
    }

    impl<F> FriMerkleTree<F> {
        pub fn new(inner: FriMerkleTreeInner) -> Self {
            Self {
                inner,
                _phantom: PhantomData,
            }
        }
    }

    impl<F> Deref for FriMerkleTree<F> {
        type Target = FriMerkleTreeInner;
        fn deref(&self) -> &Self::Target {
            &self.inner
        }
    }

    /// Batched Merkle tree backend wrapper
    pub struct BatchedMerkleTreeBackend<F>(PhantomData<F>);

    /// Batched Merkle tree wrapper with phantom type parameter.
    /// Implements Deref to allow transparent access to inner MerkleTree methods.
    pub struct BatchedMerkleTree<F> {
        inner: BatchedMerkleTreeInner,
        /// Public root field for API compatibility with MerkleTree
        pub root: Commitment,
        _phantom: PhantomData<F>,
    }

    impl<F> Clone for BatchedMerkleTree<F> {
        fn clone(&self) -> Self {
            Self {
                inner: self.inner.clone(),
                root: self.root,
                _phantom: PhantomData,
            }
        }
    }

    impl<F> BatchedMerkleTree<F> {
        /// Create wrapper from inner MerkleTree
        pub fn from_inner(inner: BatchedMerkleTreeInner) -> Self {
            let root = inner.root;
            Self {
                inner,
                root,
                _phantom: PhantomData,
            }
        }

        /// Build a new Merkle tree from generic field element vectors.
        /// Converts each FieldElement into a vector of Fp limbs (8-byte big-endian chunks).
        pub fn build<G: math::field::traits::IsField>(
            unhashed_leaves: &[Vec<math::field::element::FieldElement<G>>],
        ) -> Option<Self>
        where
            math::field::element::FieldElement<G>: math::traits::AsBytes,
        {
            // Convert each row of field elements into a flat Vec<Fp> (lossless encoding)
            let converted: Vec<Vec<Fp>> = unhashed_leaves
                .iter()
                .map(|row| field_elements_to_fps(row))
                .collect();
            BatchedMerkleTreeInner::build(&converted).map(Self::from_inner)
        }

        /// Get proof by position
        pub fn get_proof_by_pos(
            &self,
            pos: usize,
        ) -> Option<crypto::merkle_tree::proof::Proof<Commitment>> {
            self.inner.get_proof_by_pos(pos)
        }
    }

    /// Convert a FieldElement into a vector of PoseidonFp limbs.
    /// The encoding is lossless as long as `as_bytes()` is canonical.
    pub fn field_element_to_fps<G: math::field::traits::IsField>(
        elem: &math::field::element::FieldElement<G>,
    ) -> Vec<Fp>
    where
        math::field::element::FieldElement<G>: math::traits::AsBytes,
    {
        let bytes = elem.as_bytes();
        debug_assert!(
            bytes.len() % 8 == 0,
            "as_bytes() length must be a multiple of 8 for Poseidon2 encoding"
        );
        let mut limbs = Vec::with_capacity((bytes.len() + 7) / 8);
        for chunk in bytes.chunks(8) {
            let mut buf = [0u8; 8];
            // Left-pad for big-endian chunks shorter than 8 bytes.
            buf[8 - chunk.len()..].copy_from_slice(chunk);
            let val = u64::from_be_bytes(buf);
            limbs.push(Fp::from(val));
        }
        limbs
    }

    /// Convert a slice of FieldElements into a flat Vec<Fp> (concatenated limbs).
    pub fn field_elements_to_fps<G: math::field::traits::IsField>(
        elems: &[math::field::element::FieldElement<G>],
    ) -> Vec<Fp>
    where
        math::field::element::FieldElement<G>: math::traits::AsBytes,
    {
        let mut out = Vec::new();
        for elem in elems {
            out.extend(field_element_to_fps(elem));
        }
        out
    }

    impl<F> Deref for BatchedMerkleTree<F> {
        type Target = BatchedMerkleTreeInner;
        fn deref(&self) -> &Self::Target {
            &self.inner
        }
    }

    /// FRI layer Merkle tree backend wrapper
    pub struct FriLayerMerkleTreeBackend<F>(PhantomData<F>);

    /// FRI layer Merkle tree wrapper with phantom type parameter
    pub struct FriLayerMerkleTree<F> {
        inner: FriLayerMerkleTreeInner,
        /// Public root field for API compatibility with MerkleTree
        pub root: Commitment,
        _phantom: PhantomData<F>,
    }

    impl<F> FriLayerMerkleTree<F> {
        /// Create wrapper from inner MerkleTree
        pub fn from_inner(inner: FriLayerMerkleTreeInner) -> Self {
            let root = inner.root;
            Self {
                inner,
                root,
                _phantom: PhantomData,
            }
        }

        /// Build a new Merkle tree from leaves represented as Vec<Fp>
        pub fn build(unhashed_leaves: &[Vec<Fp>]) -> Option<Self> {
            FriLayerMerkleTreeInner::build(unhashed_leaves).map(Self::from_inner)
        }

        /// Get proof by position
        pub fn get_proof_by_pos(
            &self,
            pos: usize,
        ) -> Option<crypto::merkle_tree::proof::Proof<Commitment>> {
            self.inner.get_proof_by_pos(pos)
        }
    }

    impl<F> Deref for FriLayerMerkleTree<F> {
        type Target = FriLayerMerkleTreeInner;
        fn deref(&self) -> &Self::Target {
            &self.inner
        }
    }
}

#[cfg(feature = "poseidon2")]
pub use poseidon2_config::*;

// ============================================================
// Commitment to bytes conversion
// ============================================================

#[cfg(not(feature = "poseidon2"))]
pub fn commitment_to_bytes(commitment: &Commitment) -> Vec<u8> {
    commitment.to_vec()
}

#[cfg(feature = "poseidon2")]
pub fn commitment_to_bytes(commitment: &Commitment) -> Vec<u8> {
    use math::traits::AsBytes;
    commitment.as_bytes()
}
