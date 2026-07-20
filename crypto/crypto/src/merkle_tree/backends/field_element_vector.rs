use core::marker::PhantomData;

use crate::hash::poseidon::Poseidon;
use crate::merkle_tree::traits::IsMerkleTreeBackend;
use alloc::vec::Vec;
use digest::{Digest, Output};
use math::{
    field::{element::FieldElement, traits::IsField},
    traits::AsBytes,
};

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
    let mut out = [0u8; NUM_BYTES];
    hash_new_parent_bytes_into::<D, NUM_BYTES>(left, right, &mut out);
    out
}

/// Like [`hash_new_parent_bytes`] but writes the parent digest straight into
/// `out`. Folding a Merkle path can then accumulate the running hash in place
/// (ping-ponging two buffers) instead of reassigning the by-value return each
/// step — that per-step 32-byte copy compiled to a `memcpy` call and dominated
/// `verify_merkle_path_from_leaf_hash`. Byte-identical to the returning form.
#[inline]
fn hash_new_parent_bytes_into<D: Digest + 'static, const NUM_BYTES: usize>(
    left: &[u8; NUM_BYTES],
    right: &[u8; NUM_BYTES],
    out: &mut [u8; NUM_BYTES],
) {
    #[cfg(target_arch = "riscv64")]
    if NUM_BYTES == 32 && TypeId::of::<D>() == TypeId::of::<PlatformKeccak256>() {
        let l: &[u8; 32] = left[..].try_into().unwrap();
        let r: &[u8; 32] = right[..].try_into().unwrap();
        let out32: &mut [u8; 32] = (&mut out[..]).try_into().unwrap();
        // EXPERIMENT 1: HASH_PAIR ecall replaces the in-guest 64-byte
        // compression (byte-identical). Off-feature keeps the guest sponge.
        #[cfg(feature = "sim-hash-ecalls")]
        lambda_vm_syscalls::syscalls::sim_hash_pair(l.as_ptr(), r.as_ptr(), out32.as_mut_ptr());
        #[cfg(not(feature = "sim-hash-ecalls"))]
        {
            *out32 = lambda_vm_syscalls::keccak::keccak256_pair(l, r);
        }
        return;
    }

    *out = hash_streamed::<D, NUM_BYTES>(|sink| {
        sink(left);
        sink(right);
    });
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
        // EXPERIMENT 1: one-shot HASH_FELTS ecall over the two contiguous
        // elements. `FieldElement<F>` is `#[repr(transparent)]` over its
        // Goldilocks limbs, so the array pointer is the raw-limb pointer and
        // `size_of / 8` is the per-element limb count (kind). Byte-identical to
        // the `stream_bytes` streaming path below.
        #[cfg(all(target_arch = "riscv64", feature = "sim-hash-ecalls"))]
        if NUM_BYTES == 32 && TypeId::of::<D>() == TypeId::of::<PlatformKeccak256>() {
            let kind = core::mem::size_of::<FieldElement<F>>() / 8;
            let hash = lambda_vm_syscalls::keccak::sim_hash_felts(
                core::ptr::from_ref(input).cast::<u8>(),
                2,
                core::ptr::null(),
                0,
                kind,
            );
            let mut result = [0u8; NUM_BYTES];
            result.copy_from_slice(&hash);
            return result;
        }

        hash_streamed::<D, NUM_BYTES>(|sink| {
            input[0].stream_bytes(sink);
            input[1].stream_bytes(sink);
        })
    }

    fn hash_new_parent(left: &[u8; NUM_BYTES], right: &[u8; NUM_BYTES]) -> [u8; NUM_BYTES] {
        hash_new_parent_bytes::<D, NUM_BYTES>(left, right)
    }

    fn hash_new_parent_into(
        left: &[u8; NUM_BYTES],
        right: &[u8; NUM_BYTES],
        out: &mut [u8; NUM_BYTES],
    ) {
        hash_new_parent_bytes_into::<D, NUM_BYTES>(left, right, out);
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
        // EXPERIMENT 1: one-shot HASH_FELTS ecall over the two slices `a ‖ b`
        // (the verifier's `evaluations ‖ evaluations_sym` leaf shape; the plain
        // leaf case passes an empty `b`). Elements are contiguous
        // `#[repr(transparent)]` limbs, so each slice's `as_ptr()` is its
        // raw-limb pointer and `size_of / 8` is the kind. Byte-identical to the
        // streaming path below.
        #[cfg(all(target_arch = "riscv64", feature = "sim-hash-ecalls"))]
        if NUM_BYTES == 32 && TypeId::of::<D>() == TypeId::of::<PlatformKeccak256>() {
            let kind = core::mem::size_of::<FieldElement<F>>() / 8;
            let hash = lambda_vm_syscalls::keccak::sim_hash_felts(
                a.as_ptr().cast::<u8>(),
                a.len(),
                b.as_ptr().cast::<u8>(),
                b.len(),
                kind,
            );
            let mut result = [0u8; NUM_BYTES];
            result.copy_from_slice(&hash);
            return result;
        }

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

    fn hash_new_parent_into(
        left: &[u8; NUM_BYTES],
        right: &[u8; NUM_BYTES],
        out: &mut [u8; NUM_BYTES],
    ) {
        hash_new_parent_bytes_into::<D, NUM_BYTES>(left, right, out);
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
