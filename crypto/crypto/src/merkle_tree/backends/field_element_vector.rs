use core::marker::PhantomData;

use crate::hash::poseidon::Poseidon;
use crate::merkle_tree::traits::IsMerkleTreeBackend;
use alloc::vec::Vec;
use digest::{Digest, Output};
use math::{
    field::{element::FieldElement, traits::IsField},
    traits::AsBytes,
};

/// A backend for Merkle trees that uses fixed-size pairs of field elements.
/// This is more efficient than `FieldElementVectorBackend` when the batch size is always 2,
/// as it avoids Vec allocation overhead.
#[derive(Clone)]
pub struct FieldElementPairBackend<F, D: Digest, const NUM_BYTES: usize> {
    phantom1: PhantomData<F>,
    phantom2: PhantomData<D>,
}

impl<F, D: Digest, const NUM_BYTES: usize> Default for FieldElementPairBackend<F, D, NUM_BYTES> {
    fn default() -> Self {
        Self {
            phantom1: PhantomData,
            phantom2: PhantomData,
        }
    }
}

impl<F, D: Digest, const NUM_BYTES: usize> IsMerkleTreeBackend
    for FieldElementPairBackend<F, D, NUM_BYTES>
where
    F: IsField,
    FieldElement<F>: AsBytes,
    [u8; NUM_BYTES]: From<Output<D>>,
{
    type Node = [u8; NUM_BYTES];
    type Data = [FieldElement<F>; 2];

    fn hash_data(input: &[FieldElement<F>; 2]) -> [u8; NUM_BYTES] {
        let mut hasher = D::new();
        input[0].stream_bytes(&mut |b| hasher.update(b));
        input[1].stream_bytes(&mut |b| hasher.update(b));
        let mut result_hash = [0_u8; NUM_BYTES];
        result_hash.copy_from_slice(&hasher.finalize());
        result_hash
    }

    fn hash_new_parent(left: &[u8; NUM_BYTES], right: &[u8; NUM_BYTES]) -> [u8; NUM_BYTES] {
        let mut hasher = D::new();
        hasher.update(left);
        hasher.update(right);
        let mut result_hash = [0_u8; NUM_BYTES];
        result_hash.copy_from_slice(&hasher.finalize());
        result_hash
    }
}

#[derive(Clone)]
pub struct FieldElementVectorBackend<F, D: Digest, const NUM_BYTES: usize> {
    phantom1: PhantomData<F>,
    phantom2: PhantomData<D>,
}

impl<F, D: Digest, const NUM_BYTES: usize> Default for FieldElementVectorBackend<F, D, NUM_BYTES> {
    fn default() -> Self {
        Self {
            phantom1: PhantomData,
            phantom2: PhantomData,
        }
    }
}

impl<F, D: Digest, const NUM_BYTES: usize> FieldElementVectorBackend<F, D, NUM_BYTES>
where
    [u8; NUM_BYTES]: From<Output<D>>,
{
    /// Hash raw bytes using the same digest (`D`) as this backend's leaf hashing.
    /// Enables callers to pre-serialize field elements into a byte buffer and hash
    /// once, avoiding per-element allocations while staying consistent with the
    /// backend's hash function.
    pub fn hash_bytes(data: &[u8]) -> [u8; NUM_BYTES] {
        let mut hasher = D::new();
        hasher.update(data);
        let mut result = [0u8; NUM_BYTES];
        result.copy_from_slice(&hasher.finalize());
        result
    }
}

impl<F, D: Digest, const NUM_BYTES: usize> FieldElementVectorBackend<F, D, NUM_BYTES>
where
    F: IsField,
    FieldElement<F>: AsBytes,
    [u8; NUM_BYTES]: From<Output<D>>,
{
    /// Leaf-hash the concatenation of two field-element slices `a ‖ b` without
    /// materializing it. Streams every element of `a` then every element of `b`
    /// into the digest, so the result is byte-identical to
    /// `hash_data(&[a, b].concat())`: the sponge absorbs the same element bytes
    /// in the same order, just without the intermediate `Vec`.
    pub fn hash_data_from_slices(a: &[FieldElement<F>], b: &[FieldElement<F>]) -> [u8; NUM_BYTES] {
        let mut hasher = D::new();
        for element in a.iter().chain(b.iter()) {
            element.stream_bytes(&mut |bytes| hasher.update(bytes));
        }
        let mut result_hash = [0_u8; NUM_BYTES];
        result_hash.copy_from_slice(&hasher.finalize());
        result_hash
    }
}

impl<F, D: Digest, const NUM_BYTES: usize> IsMerkleTreeBackend
    for FieldElementVectorBackend<F, D, NUM_BYTES>
where
    F: IsField,
    FieldElement<F>: AsBytes,
    [u8; NUM_BYTES]: From<Output<D>>,
    Vec<FieldElement<F>>: Sync + Send,
{
    type Node = [u8; NUM_BYTES];
    type Data = Vec<FieldElement<F>>;

    fn hash_data(input: &Vec<FieldElement<F>>) -> [u8; NUM_BYTES] {
        // Delegate to the two-slice hash so the leaf-hash byte layout has a
        // single source of truth: a plain leaf is the concatenation with an
        // empty second slice.
        Self::hash_data_from_slices(input, &[])
    }

    fn hash_new_parent(left: &[u8; NUM_BYTES], right: &[u8; NUM_BYTES]) -> [u8; NUM_BYTES] {
        let mut hasher = D::new();
        hasher.update(left);
        hasher.update(right);
        let mut result_hash = [0_u8; NUM_BYTES];
        result_hash.copy_from_slice(&hasher.finalize());
        result_hash
    }
}

#[derive(Clone, Default)]
pub struct BatchPoseidonTree<P: Poseidon + Default> {
    _poseidon: PhantomData<P>,
}

impl<P> IsMerkleTreeBackend for BatchPoseidonTree<P>
where
    P: Poseidon + Default,
    Vec<FieldElement<P::F>>: Sync + Send,
    FieldElement<P::F>: Sync + Send,
{
    type Node = FieldElement<P::F>;
    type Data = Vec<FieldElement<P::F>>;

    fn hash_data(input: &Vec<FieldElement<P::F>>) -> FieldElement<P::F> {
        P::hash_many(input)
    }

    fn hash_new_parent(
        left: &FieldElement<P::F>,
        right: &FieldElement<P::F>,
    ) -> FieldElement<P::F> {
        P::hash(left, right)
    }
}
