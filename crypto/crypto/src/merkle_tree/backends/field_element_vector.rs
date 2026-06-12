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
        // One reused buffer; `write_bytes` is non-allocating for the field types
        // used in production (Goldilocks / ext3), unlike `as_bytes()`.
        let mut buf = Vec::new();
        let mut hasher = D::new();
        input[0].write_bytes(&mut buf);
        hasher.update(&buf);
        buf.clear();
        input[1].write_bytes(&mut buf);
        hasher.update(&buf);
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

impl<F, D, const NUM_BYTES: usize> FieldElementVectorBackend<F, D, NUM_BYTES>
where
    F: IsField,
    FieldElement<F>: AsBytes,
    D: Digest,
    [u8; NUM_BYTES]: From<Output<D>>,
{
    /// Hash a sequence of borrowed field elements into a leaf node, identical
    /// to `hash_data` over the same sequence — but without materializing a Vec.
    pub fn hash_elements<'a, I>(elements: I) -> [u8; NUM_BYTES]
    where
        I: IntoIterator<Item = &'a FieldElement<F>>,
        F: 'a,
    {
        let mut buf = Vec::new();
        let mut hasher = D::new();
        for element in elements {
            buf.clear();
            element.write_bytes(&mut buf);
            hasher.update(&buf);
        }
        let mut result_hash = [0u8; NUM_BYTES];
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
        let mut buf = Vec::new();
        let mut hasher = D::new();
        for element in input.iter() {
            buf.clear();
            element.write_bytes(&mut buf);
            hasher.update(&buf);
        }
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
