use crate::hash::poseidon::Poseidon;
use crate::merkle_tree::traits::IsMerkleTreeBackend;
use core::marker::PhantomData;
use digest::{Digest, Output};
use math::{
    field::{element::FieldElement, traits::IsField},
    traits::ByteConversion,
};

#[derive(Clone)]
pub struct FieldElementBackend<F, D: Digest, const NUM_BYTES: usize> {
    phantom1: PhantomData<F>,
    phantom2: PhantomData<D>,
}

impl<F, D: Digest, const NUM_BYTES: usize> Default for FieldElementBackend<F, D, NUM_BYTES> {
    fn default() -> Self {
        Self {
            phantom1: PhantomData,
            phantom2: PhantomData,
        }
    }
}

impl<F, D: Digest, const NUM_BYTES: usize> IsMerkleTreeBackend
    for FieldElementBackend<F, D, NUM_BYTES>
where
    F: IsField,
    FieldElement<F>: ByteConversion + Sync + Send,
    [u8; NUM_BYTES]: From<Output<D>>,
{
    type Node = [u8; NUM_BYTES];
    type Data = FieldElement<F>;

    fn hash_data(input: &FieldElement<F>) -> [u8; NUM_BYTES] {
        let mut hasher = D::new();
        // Hash the big-endian bytes directly from the fixed-size array (no
        // allocation). Same bytes as the previous `as_bytes()` (= to_bytes_be).
        hasher.update(input.to_bytes_be().as_ref());
        hasher.finalize().into()
    }

    fn hash_new_parent(left: &[u8; NUM_BYTES], right: &[u8; NUM_BYTES]) -> [u8; NUM_BYTES] {
        let mut hasher = D::new();
        hasher.update(left);
        hasher.update(right);
        hasher.finalize().into()
    }
}

#[derive(Clone, Default)]
pub struct TreePoseidon<P: Poseidon + Default> {
    _poseidon: PhantomData<P>,
}

impl<P> IsMerkleTreeBackend for TreePoseidon<P>
where
    P: Poseidon + Default,
    FieldElement<P::F>: Sync + Send,
{
    type Node = FieldElement<P::F>;
    type Data = FieldElement<P::F>;

    fn hash_data(input: &FieldElement<P::F>) -> FieldElement<P::F> {
        P::hash_single(input)
    }

    fn hash_new_parent(
        left: &FieldElement<P::F>,
        right: &FieldElement<P::F>,
    ) -> FieldElement<P::F> {
        P::hash(left, right)
    }
}
