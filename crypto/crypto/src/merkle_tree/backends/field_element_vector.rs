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

/// Keccak rate for a 512-bit-capacity sponge (Keccak256): (1600 - 2*256) / 8.
const RATE: usize = 136;

/// Keccak256(left ‖ right) for two 32-byte nodes (always 64 bytes, one
/// rate block): builds the padded state directly from `left`/`right`,
/// skipping both the `sha3`/`block_buffer` incremental-update machinery and
/// the 136-byte staging buffer `keccak256_one_block` needs for
/// variable-length input. Byte-identical to
/// `Keccak256::new().chain_update(left).chain_update(right).finalize()`.
#[inline]
fn keccak256_pair(left: &[u8], right: &[u8]) -> [u8; 32] {
    debug_assert_eq!(left.len(), 32);
    debug_assert_eq!(right.len(), 32);

    let mut state = [0u64; 25];
    for i in 0..4 {
        state[i] = u64::from_le_bytes(left[i * 8..i * 8 + 8].try_into().unwrap());
        state[i + 4] = u64::from_le_bytes(right[i * 8..i * 8 + 8].try_into().unwrap());
    }
    // pad10*1: single 0x01 at byte 64 (lane 8), single 0x80 at byte 135 (top
    // byte of lane 16) — the two never collide since 64 != RATE - 1.
    state[8] = 0x01;
    state[16] = 0x8000_0000_0000_0000;
    keccak::f1600(&mut state);

    let mut out = [0u8; 32];
    for i in 0..4 {
        out[i * 8..i * 8 + 8].copy_from_slice(&state[i].to_le_bytes());
    }
    out
}

/// Keccak256 of a single sub-rate (`< RATE` byte) message: one keccak-f[1600]
/// permutation via a staged buffer, needed because the caller's byte count
/// isn't known at compile time (unlike [`keccak256_pair`]'s fixed 64 bytes).
/// Byte-identical to `Keccak256::digest(data)` for any `data.len() < RATE`.
#[inline]
fn keccak256_one_block(data: &[u8]) -> [u8; 32] {
    debug_assert!(data.len() < RATE);

    let mut block = [0u8; RATE];
    block[..data.len()].copy_from_slice(data);
    // `|=` (not `=`) so the len == RATE - 1 case combines both pad bits
    // (0x01 | 0x80 = 0x81) in the single byte they then share.
    block[data.len()] |= 0x01;
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
#[inline]
fn hash_new_parent_bytes<D: Digest + 'static, const NUM_BYTES: usize>(
    left: &[u8; NUM_BYTES],
    right: &[u8; NUM_BYTES],
) -> [u8; NUM_BYTES] {
    if NUM_BYTES == 32 && TypeId::of::<D>() == TypeId::of::<Keccak256>() {
        let hash = keccak256_pair(left.as_slice(), right.as_slice());
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
        if NUM_BYTES == 32 && TypeId::of::<D>() == TypeId::of::<Keccak256>() {
            let a = input[0].as_bytes();
            let b = input[1].as_bytes();
            if a.len() + b.len() < RATE {
                let mut block = [0u8; RATE];
                block[..a.len()].copy_from_slice(&a);
                block[a.len()..a.len() + b.len()].copy_from_slice(&b);
                let hash = keccak256_one_block(&block[..a.len() + b.len()]);
                let mut result_hash = [0_u8; NUM_BYTES];
                result_hash.as_mut_slice().copy_from_slice(&hash);
                return result_hash;
            }
        }

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

    // Unlike `FieldElementPairBackend::hash_data` (always 2 elements, always
    // sub-rate), real callers here hash whole trace rows (tens of columns),
    // reliably exceeding the 136-byte rate — a "try one block, else fall
    // back" attempt would pay a wasted scan on every call for no payoff, so
    // this stays on the generic multi-block `Digest` path unconditionally.
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
    use super::{
        FieldElementPairBackend, FieldElementVectorBackend, IsMerkleTreeBackend, RATE,
        hash_new_parent_bytes, keccak256_one_block, keccak256_pair,
    };
    use alloc::vec::Vec;
    use math::field::{element::FieldElement, goldilocks::GoldilocksField};
    use math::traits::AsBytes;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;
    use sha3::{Digest, Keccak256};

    type F = GoldilocksField;

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

    #[test]
    fn keccak256_one_block_matches_sha3_keccak256_on_random_short_inputs() {
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        for len in 0..RATE {
            let mut data = alloc::vec![0u8; len];
            rng.fill(data.as_mut_slice());

            let fast = keccak256_one_block(&data);
            let expected: [u8; 32] = Keccak256::digest(&data).into();

            assert_eq!(fast, expected, "mismatch at len={len}");
        }
    }

    /// Pins the `TypeId` dispatch itself (not just the inner helper) against
    /// an independently-computed reference, so a broken dispatch condition
    /// can't hide behind `keccak256_pair` and `keccak256_one_block` each
    /// individually being correct.
    #[test]
    fn hash_new_parent_bytes_dispatch_matches_reference_keccak256() {
        let mut rng = ChaCha8Rng::seed_from_u64(2);
        for _ in 0..100 {
            let mut left = [0u8; 32];
            let mut right = [0u8; 32];
            rng.fill(&mut left);
            rng.fill(&mut right);

            let dispatched = hash_new_parent_bytes::<Keccak256, 32>(&left, &right);

            let mut hasher = Keccak256::new();
            hasher.update(left);
            hasher.update(right);
            let expected: [u8; 32] = hasher.finalize().into();

            assert_eq!(dispatched, expected);
        }
    }

    #[test]
    fn pair_backend_hash_data_fast_path_matches_reference_keccak256() {
        type Backend = FieldElementPairBackend<F, Keccak256, 32>;

        let input = [
            FieldElement::<F>::from(11u64),
            FieldElement::<F>::from(42u64),
        ];
        let actual = Backend::hash_data(&input);

        let mut hasher = Keccak256::new();
        hasher.update(input[0].as_bytes());
        hasher.update(input[1].as_bytes());
        let expected: [u8; 32] = hasher.finalize().into();

        assert_eq!(actual, expected);
    }

    #[test]
    fn vector_backend_hash_data_matches_reference_keccak256() {
        type Backend = FieldElementVectorBackend<F, Keccak256, 32>;

        // 32 Goldilocks elements * 8 bytes = 256 bytes, representative of a
        // real (multi-block) trace-row leaf; this backend has no single-block
        // fast path (see the comment on its `hash_data`), so this just pins
        // the untouched generic `Digest` path.
        let input: Vec<FieldElement<F>> =
            (0..32).map(|i| FieldElement::<F>::from(i as u64)).collect();
        let actual = Backend::hash_data(&input);

        let mut hasher = Keccak256::new();
        for element in input.iter() {
            hasher.update(element.as_bytes());
        }
        let expected: [u8; 32] = hasher.finalize().into();

        assert_eq!(actual, expected);
    }
}
