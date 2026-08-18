use core::marker::PhantomData;

use crate::hash::poseidon::Poseidon;
use crate::merkle_tree::traits::{IsMerkleTreeBackend, IsStreamingLeafBackend};
use alloc::vec::Vec;
use digest::{Digest, Output};
use math::{
    field::{element::FieldElement, traits::IsField},
    traits::AsBytes,
};

#[cfg(target_arch = "riscv64")]
use crate::hash::blake3::chain::{Blake3Chain, blake3_parent};
#[cfg(target_arch = "riscv64")]
use crate::hash::platform_keccak::PlatformKeccak256;
#[cfg(target_arch = "riscv64")]
use core::any::TypeId;
#[cfg(target_arch = "riscv64")]
use lambda_vm_syscalls::keccak::Keccak256 as SyscallKeccak256;

/// Absorb `feed`'s byte stream into a fresh `D` and return the digest as a
/// fixed `[u8; NUM_BYTES]`.
///
/// On the riscv64 guest, when `D` is the platform keccak digest and the node
/// is 32 bytes, this drives the syscall sponge directly and squeezes straight
/// into the result array. That skips the `Digest::finalize` blanket, which
/// allocates a zeroed `GenericArray`, has the adapter fill a local `[u8; 32]`
/// and copy it into that `Output`, then leaves the caller to copy the `Output`
/// once more into its own array — two 32-byte memcpys plus a 32-byte memset of
/// pure plumbing around the one permutation. Byte-identical to the generic
/// path; every other digest / node size (and the entire host build) takes the
/// generic path unchanged.
///
/// DO NOT replace this `TypeId` dispatch with a generic `Digest::finalize_into`
/// fix "at the adapter altitude" — that exact refactor was implemented and
/// MEASURED SLOWER on the guest across every formulation tried (best:
/// +60k min = +0.14%, +1.25M blowup8 = +0.48%), including `#[inline]`
/// adapters and a check-free `AsMut` output conversion. The residual is
/// intrinsic: `FixedOutput::finalize_into` moves the 208-byte sponge by value
/// through the newtype + trait layer into a non-inlined cross-crate call, and
/// without LTO the placement isn't elided; the direct branch below builds the
/// sponge in place at the call's ABI slot. Deleting the dispatch also cannot
/// remove the `'static` bounds — `hash_new_parent_bytes` needs them regardless.
#[inline]
fn hash_streamed<D: Digest + 'static, const NUM_BYTES: usize>(
    feed: impl Fn(&mut dyn FnMut(&[u8])),
) -> [u8; NUM_BYTES] {
    #[cfg(target_arch = "riscv64")]
    if NUM_BYTES == 32 && TypeId::of::<D>() == TypeId::of::<PlatformKeccak256>() {
        let mut hasher = SyscallKeccak256::new();
        feed(&mut |bytes| hasher.update(bytes));
        let mut result = [0u8; NUM_BYTES];
        // NUM_BYTES == 32 in this branch, so the slice is exactly a [u8; 32].
        let out: &mut [u8; 32] = (&mut result[..]).try_into().unwrap();
        hasher.finalize(out);
        return result;
    }

    let mut hasher = D::new();
    feed(&mut |bytes| hasher.update(bytes));
    let mut result_hash = [0_u8; NUM_BYTES];
    result_hash.copy_from_slice(&hasher.finalize());
    result_hash
}

/// Hash a Merkle parent — always exactly two concatenated 32-byte nodes.
///
/// On the riscv64 guest, when `D` is the platform keccak digest and nodes are
/// 32 bytes, this is one fixed-shape 64-byte compression ([`keccak256_pair`]):
/// a single permutation with the input lanes and padding written straight into
/// the state, skipping the incremental sponge's per-byte absorb, running
/// offset, and separate padding pass. Byte-identical to streaming both nodes
/// through the digest; every other digest / node size (and the host build)
/// takes the generic streaming-and-finalize path unchanged.
#[inline]
fn hash_new_parent_bytes<D: Digest + 'static, const NUM_BYTES: usize>(
    left: &[u8; NUM_BYTES],
    right: &[u8; NUM_BYTES],
) -> [u8; NUM_BYTES] {
    #[cfg(target_arch = "riscv64")]
    if NUM_BYTES == 32 && TypeId::of::<D>() == TypeId::of::<PlatformKeccak256>() {
        let l: &[u8; 32] = left[..].try_into().unwrap();
        let r: &[u8; 32] = right[..].try_into().unwrap();
        let hash = lambda_vm_syscalls::keccak::keccak256_pair(l, r);
        let mut result = [0u8; NUM_BYTES];
        result.copy_from_slice(&hash);
        return result;
    }

    // The BLAKE3 twin of the branch above, and it is the same trade: a 64-byte
    // parent is ONE compression (P2), so the sponge around it is pure plumbing.
    // In-guest that plumbing measured ~55% of a 64-byte hash — ~556 cycles for
    // ~248 of compression — and a Merkle parent is the shape every FRI query
    // path step is made of.
    //
    // Keyed on the CONCRETE digest, exactly like keccak's: `Blake3Chain` is the
    // type `BatchBlake3Backend`/`PairBlake3Backend` instantiate, so no other
    // digest can fall into this branch. `blake3_parent` goes through
    // `chain.rs`'s single `compress_block` dispatch, so this is the accelerator
    // ecall on a guest that has it and the software path otherwise — one
    // framing, not a second transcription.
    #[cfg(target_arch = "riscv64")]
    if NUM_BYTES == 32 && TypeId::of::<D>() == TypeId::of::<Blake3Chain>() {
        let l: &[u8; 32] = left[..].try_into().unwrap();
        let r: &[u8; 32] = right[..].try_into().unwrap();
        let hash = blake3_parent(l, r);
        let mut result = [0u8; NUM_BYTES];
        result.copy_from_slice(&hash);
        return result;
    }

    hash_streamed::<D, NUM_BYTES>(|sink| {
        sink(left);
        sink(right);
    })
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
        hash_streamed::<D, NUM_BYTES>(|sink| {
            input[0].stream_bytes(sink);
            input[1].stream_bytes(sink);
        })
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

impl<F, D: Digest + 'static, const NUM_BYTES: usize> FieldElementVectorBackend<F, D, NUM_BYTES>
where
    [u8; NUM_BYTES]: From<Output<D>>,
{
    /// Hash raw bytes using the same digest (`D`) as this backend's leaf hashing.
    /// Enables callers to pre-serialize field elements into a byte buffer and hash
    /// once, avoiding per-element allocations while staying consistent with the
    /// backend's hash function.
    pub fn hash_bytes(data: &[u8]) -> [u8; NUM_BYTES] {
        hash_streamed::<D, NUM_BYTES>(|sink| sink(data))
    }
}

impl<F, D: Digest + 'static, const NUM_BYTES: usize> FieldElementVectorBackend<F, D, NUM_BYTES>
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
        hash_streamed::<D, NUM_BYTES>(|sink| {
            for element in a.iter().chain(b.iter()) {
                element.stream_bytes(sink);
            }
        })
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
        // Delegate to the two-slice hash so the leaf-hash byte layout has a
        // single source of truth: a plain leaf is the concatenation with an
        // empty second slice.
        Self::hash_data_from_slices(input, &[])
    }

    fn hash_new_parent(left: &[u8; NUM_BYTES], right: &[u8; NUM_BYTES]) -> [u8; NUM_BYTES] {
        hash_new_parent_bytes::<D, NUM_BYTES>(left, right)
    }
}

/// Exposes the streaming leaf routes to callers that reach this backend through
/// a commitment configuration rather than by name. Both bodies go through
/// [`hash_streamed`], which is where the absorbed byte layout is defined, so
/// they agree with `hash_data` by construction.
impl<F, D: Digest + 'static, const NUM_BYTES: usize> IsStreamingLeafBackend<F>
    for FieldElementVectorBackend<F, D, NUM_BYTES>
where
    F: IsField,
    FieldElement<F>: AsBytes,
    [u8; NUM_BYTES]: From<Output<D>>,
    Vec<FieldElement<F>>: Sync + Send,
{
    fn hash_bytes(data: &[u8]) -> [u8; NUM_BYTES] {
        hash_streamed::<D, NUM_BYTES>(|sink| sink(data))
    }

    fn hash_data_from_slices(a: &[FieldElement<F>], b: &[FieldElement<F>]) -> [u8; NUM_BYTES] {
        hash_streamed::<D, NUM_BYTES>(|sink| {
            for element in a.iter().chain(b.iter()) {
                element.stream_bytes(sink);
            }
        })
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
