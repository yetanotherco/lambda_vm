//! ★ The hash the WHIR path runs on — the one name a proof's Merkle trees, its
//! Fiat-Shamir sponge and its proof-of-work all answer to.
//!
//! # Why one trait and not three parameters
//!
//! A WHIR proof has three hash consumers: the Merkle backend that builds its
//! roots, the transcript sponge that draws its challenges, and the grind that
//! gates each redrawable challenge. Parameterising them separately would make
//! the **half-flip** spellable — one hash's trees under another hash's sponge.
//! That configuration is self-consistent between prover and verifier, so it
//! verifies, so nothing fails; it is silent by construction, and the only thing
//! wrong with it is that no single name describes the proof. Here there is one
//! name to write, so there is nothing to assert against: the bad state is
//! unreachable rather than checked.
//!
//! # What a hash may NOT change
//!
//! [`Commitment`] stays `[u8; 32]` for every implementation. A keccak digest is
//! 32 bytes and an algebraic digest is four canonical Goldilocks felts, which is
//! also 32 bytes — so every proof type on this path (`ChainProof`,
//! `StackedProof`, `CosetOpening`, `Proof<Commitment>`, and `MultiProof` above
//! them) keeps its layout, its rkyv derives and its serialized length. **A hash
//! swap is not a proof-format change**, and `stacked_eval`'s
//! `the_two_hashes_serialize_to_the_same_length` is what holds that to it.
//!
//! # The device
//!
//! Under `cuda` the leaf and parent hashing happens in KERNELS, and the host
//! backend is only the label on the tree they built. So a configuration must
//! also name which kernel family the device has to run:
//! [`WhirHash::DEVICE`] is that name, handed down to `math-cuda`'s tree entry
//! points, which match on it exhaustively. A tree labelled `Self` was therefore
//! hashed by `Self`'s kernels or was not built on the device at all — never by
//! another hash's kernels wearing this name.
//!
//! The key is a type of this crate's own rather than `math_cuda::DeviceHash`
//! directly, because `math-cuda` is an optional dependency and the trait has to
//! exist on a build without it. [`DeviceHashKey::into_math_cuda`] is the bridge,
//! and it is total in both directions with the pairing asserted at compile time,
//! so the two enums cannot drift apart or be cross-wired.

use crypto::fiat_shamir::transcript_hash::{
    KeccakTranscriptHash, RpxTranscriptHash, TranscriptHash,
};
use crypto::merkle_tree::backends::types::{BatchKeccak256Backend, BatchRpx256Backend};
use crypto::merkle_tree::traits::IsMerkleTreeBackend;
use math::field::{element::FieldElement, traits::IsField};
use math::traits::AsBytes;

use crate::whir_commit::Commitment;

/// The digest a configuration grinds over: its transcript's hash, because the
/// grinding seed is `transcript.state()`.
pub type GrindingDigest<H> = <<H as WhirHash>::Transcript as TranscriptHash>::Digest;

/// ★ Which kernel family the device must run for a configuration's trees.
///
/// Mirrors `math_cuda::DeviceHash` and exists separately only so this trait
/// compiles without the optional `math-cuda` dependency. The two are kept in
/// step by [`DeviceHashKey::into_math_cuda`] plus the compile-time pairing
/// assertion beside it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeviceHashKey {
    /// Keccak-256 at both the leaf and the parent layer.
    Keccak256,
    /// RPX256 (XHash12) at both layers.
    Rpx256,
}

impl DeviceHashKey {
    /// The key `math-cuda` dispatches on.
    ///
    /// Total, and exhaustive in both directions: a variant added on either side
    /// without its twin is a compile error here rather than a silent
    /// fallthrough to whichever hash happened to be first.
    #[cfg(feature = "cuda")]
    pub const fn into_math_cuda(self) -> math_cuda::DeviceHash {
        match self {
            Self::Keccak256 => math_cuda::DeviceHash::Keccak256,
            Self::Rpx256 => math_cuda::DeviceHash::Rpx256,
        }
    }

    /// The name a tree built under this key may be called by.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Keccak256 => "keccak256",
            Self::Rpx256 => "rpx256",
        }
    }
}

/// ✓ The bridge is a bijection, checked at compile time rather than by reading
/// it: every key maps to the twin of the same name, and the names agree.
#[cfg(feature = "cuda")]
const _: () = {
    const fn paired(key: DeviceHashKey, twin: math_cuda::DeviceHash) -> bool {
        key.into_math_cuda() as u8 == twin as u8
    }
    assert!(paired(
        DeviceHashKey::Keccak256,
        math_cuda::DeviceHash::Keccak256
    ));
    assert!(paired(DeviceHashKey::Rpx256, math_cuda::DeviceHash::Rpx256));
};

/// ★ One WHIR hash configuration.
///
/// Implementing it is the whole of adding a hash to this path: a unit struct, a
/// Merkle backend and a Fiat-Shamir configuration. Nothing in `whir_commit`,
/// `whir_round`, `whir_chain` or `stacked_eval` knows which one it has.
pub trait WhirHash: Copy + Clone + Default + Send + Sync + 'static {
    /// The name a proof's roots may be called by — for banners, KATs and
    /// diagnostics. Never absorbed into the transcript: the sponge IS this
    /// hash, so two configurations' challenge streams diverge at the first
    /// squeeze and a tag would separate nothing that is not already separate.
    const NAME: &'static str;

    /// The kernel family the device must run for trees labelled `Self`.
    ///
    /// Not `#[cfg(feature = "cuda")]`: a configuration names its device hash on
    /// every build, so a non-cuda build cannot define a configuration that would
    /// have had nothing to dispatch on.
    const DEVICE: DeviceHashKey;

    /// The Fiat-Shamir configuration this commitment hash is paired with — the
    /// sponge's hash, and the one the proof-of-work grind computes over.
    type Transcript: TranscriptHash;

    /// The Merkle backend: one leaf per fold block, 32-byte nodes.
    type Backend<F>: IsMerkleTreeBackend<Node = Commitment, Data = Vec<FieldElement<F>>>
    where
        F: IsField + 'static,
        FieldElement<F>: AsBytes + Sync + Send;
}

/// The keccak-256 configuration — the default everywhere on this path, and
/// byte-for-byte what PR #988 produces.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct KeccakWhir;

impl WhirHash for KeccakWhir {
    const NAME: &'static str = "keccak256";

    const DEVICE: DeviceHashKey = DeviceHashKey::Keccak256;

    type Transcript = KeccakTranscriptHash;

    type Backend<F>
        = BatchKeccak256Backend<F>
    where
        F: IsField + 'static,
        FieldElement<F>: AsBytes + Sync + Send;
}

/// ★ The RPX256 configuration — the algebraic hash, and the only reason this
/// trait exists.
///
/// Slower than keccak on a host by a wide margin, and that is not a defect to
/// be fixed: the lever it pulls is elsewhere. A keccak-f[1600] costs roughly
/// 73,700 trace cells in a field-native verifier against RPX's 325, so a WHIR
/// proof verified inside another proof pays about 227x less for its hashing
/// under this configuration. A proof that will only ever be checked by a host
/// should use [`KeccakWhir`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct RpxWhir;

impl WhirHash for RpxWhir {
    const NAME: &'static str = "rpx256";

    const DEVICE: DeviceHashKey = DeviceHashKey::Rpx256;

    type Transcript = RpxTranscriptHash;

    type Backend<F>
        = BatchRpx256Backend<F>
    where
        F: IsField + 'static,
        FieldElement<F>: AsBytes + Sync + Send;
}

/// ✓ Both configurations are INHABITED at the fields the prover actually
/// commits over — the base field for traces, the cubic extension for the folded
/// codewords — within one proof.
///
/// A `WhirHash` impl that type-checks in isolation can still be unusable: the
/// associated type is generic over `F`, and the bound that matters is the one
/// the prover instantiates it at. This is that instantiation, as a compile-time
/// check rather than a comment claiming it holds.
const _: fn() = || {
    fn assert_usable<H: WhirHash>()
    where
        H::Backend<math::field::goldilocks::GoldilocksField>:
            IsMerkleTreeBackend<Node = Commitment>,
        H::Backend<math::field::extensions_goldilocks::Degree3GoldilocksExtensionField>:
            IsMerkleTreeBackend<Node = Commitment>,
    {
    }

    assert_usable::<KeccakWhir>();
    assert_usable::<RpxWhir>();
};

/// ✓ Each configuration's transcript is ITS OWN, not the other's.
///
/// The half-flip is unspellable because one trait supplies both halves — but
/// only if the two impls actually name different transcripts. A copy-paste that
/// left `RpxWhir` on `KeccakTranscriptHash` would be exactly the silent
/// configuration this design exists to rule out, so it is made a compile error.
const _: fn() = || {
    fn assert_same<T>(_: core::marker::PhantomData<(T, T)>) {}

    assert_same::<KeccakTranscriptHash>(
        core::marker::PhantomData::<(KeccakTranscriptHash, <KeccakWhir as WhirHash>::Transcript)>,
    );
    assert_same::<RpxTranscriptHash>(
        core::marker::PhantomData::<(RpxTranscriptHash, <RpxWhir as WhirHash>::Transcript)>,
    );
};

/// ✓ A configuration and its device key answer to the SAME name.
///
/// Both are string constants written by hand, so nothing but this stops
/// `RpxWhir::NAME` from saying `rpx256` while its kernels are filed under
/// `keccak256` — which is precisely the mislabelling the key exists to prevent,
/// reintroduced one level up.
const _: () = {
    const fn same(a: &str, b: &str) -> bool {
        let (a, b) = (a.as_bytes(), b.as_bytes());
        if a.len() != b.len() {
            return false;
        }
        let mut i = 0;
        while i < a.len() {
            if a[i] != b[i] {
                return false;
            }
            i += 1;
        }
        true
    }
    assert!(same(KeccakWhir::NAME, KeccakWhir::DEVICE.name()));
    assert!(same(RpxWhir::NAME, RpxWhir::DEVICE.name()));
};
