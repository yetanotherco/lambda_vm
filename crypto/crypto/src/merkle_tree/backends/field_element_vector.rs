use core::any::TypeId;
use core::marker::PhantomData;

use crate::hash::poseidon::Poseidon;
use crate::merkle_tree::traits::IsMerkleTreeBackend;
use alloc::vec::Vec;
use digest::{Digest, Output};
use math::{
    field::{element::FieldElement, traits::IsField},
    traits::AsBytes,
};
use sha3::Keccak256;

/// Keccak256 of exactly 64 bytes (two concatenated 32-byte nodes) fits in a
/// single keccak-f[1600] permutation: this bypasses the `sha3`/`block_buffer`
/// incremental-update machinery, which is pure overhead for one fixed-size
/// block. Byte-identical to `Keccak256::new().chain_update(left).chain_update(right).finalize()`.
fn keccak256_pair(left: &[u8; 32], right: &[u8; 32]) -> [u8; 32] {
    const RATE: usize = 136;
    let mut block = [0u8; RATE];
    block[..32].copy_from_slice(left);
    block[32..64].copy_from_slice(right);
    block[64] = 0x01;
    block[RATE - 1] |= 0x80;

    let mut state = [0u64; 25];
    for (lane, chunk) in state.iter_mut().zip(block.chunks_exact(8)) {
        *lane = u64::from_le_bytes(chunk.try_into().unwrap());
    }
    keccak::f1600(&mut state);

    let mut out = [0u8; 32];
    for (chunk, lane) in out.chunks_exact_mut(8).zip(state.iter()) {
        chunk.copy_from_slice(&lane.to_le_bytes());
    }
    out
}

/// Dispatches to [`keccak256_pair`] when `D` is exactly `Keccak256` and the
/// node size is 32 bytes (the only shape `hash_new_parent` is ever called
/// with in practice); falls back to the generic `Digest` path otherwise.
/// `TypeId::of::<D>()` is a per-monomorphization compile-time constant, so
/// this branch is fully resolved (and the untaken side dead-code-eliminated)
/// at codegen time — no runtime dispatch cost.
fn hash_new_parent_bytes<D: Digest + 'static, const NUM_BYTES: usize>(
    left: &[u8; NUM_BYTES],
    right: &[u8; NUM_BYTES],
) -> [u8; NUM_BYTES] {
    if NUM_BYTES == 32 && TypeId::of::<D>() == TypeId::of::<Keccak256>() {
        let l: [u8; 32] = left.as_slice().try_into().unwrap();
        let r: [u8; 32] = right.as_slice().try_into().unwrap();
        let hash = keccak256_pair(&l, &r);
        let mut result_hash = [0_u8; NUM_BYTES];
        result_hash.as_mut_slice().copy_from_slice(&hash);
        return result_hash;
    }

    let mut hasher = D::new();
    hasher.update(left);
    hasher.update(right);
    let mut result_hash = [0_u8; NUM_BYTES];
    result_hash.copy_from_slice(&hasher.finalize());
    result_hash
}

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

impl<F, D: Digest + 'static, const NUM_BYTES: usize> IsMerkleTreeBackend
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
        hasher.update(input[0].as_bytes());
        hasher.update(input[1].as_bytes());
        let mut result_hash = [0_u8; NUM_BYTES];
        result_hash.copy_from_slice(&hasher.finalize());
        result_hash
    }

    fn hash_new_parent(left: &[u8; NUM_BYTES], right: &[u8; NUM_BYTES]) -> [u8; NUM_BYTES] {
        hash_new_parent_bytes::<D, NUM_BYTES>(left, right)
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

impl<F, D: Digest + 'static, const NUM_BYTES: usize> IsMerkleTreeBackend
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
        let mut hasher = D::new();
        for element in input.iter() {
            hasher.update(element.as_bytes());
        }
        let mut result_hash = [0_u8; NUM_BYTES];
        result_hash.copy_from_slice(&hasher.finalize());
        result_hash
    }

    fn hash_new_parent(left: &[u8; NUM_BYTES], right: &[u8; NUM_BYTES]) -> [u8; NUM_BYTES] {
        hash_new_parent_bytes::<D, NUM_BYTES>(left, right)
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

#[cfg(test)]
mod tests {
    use super::keccak256_pair;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;
    use sha3::{Digest, Keccak256};

    #[test]
    fn keccak256_pair_matches_sha3_keccak256_on_random_inputs() {
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        for _ in 0..1000 {
            let mut left = [0u8; 32];
            let mut right = [0u8; 32];
            rng.fill(&mut left);
            rng.fill(&mut right);

            let fast = keccak256_pair(&left, &right);

            let mut hasher = Keccak256::new();
            hasher.update(left);
            hasher.update(right);
            let expected: [u8; 32] = hasher.finalize().into();

            assert_eq!(fast, expected);
        }
    }
}
