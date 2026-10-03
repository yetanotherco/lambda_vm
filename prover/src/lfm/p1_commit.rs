//! ZisK's Poseidon1 as a STARK commitment configuration — the `p1/*`
//! exploration branch's base hash. ⚠ Not a candidate for landing; nothing
//! proves with it unless a caller names [`P1StarkHash`].
//!
//! The primitives are `crypto::hash::poseidon1_stark` (checked against ZisK's
//! own code): ZisK's leaf hash, ARITY-4 trees padded with zero digests, ZisK's
//! transcript sponge and its width-8 grinding permutation. What this module
//! adds is the encoding at the STARK's seams, the same conventions
//! `algebraic_commit` and `algebraic_transcript` fix for RPX:
//!
//! | seam | encoding |
//! |---|---|
//! | leaf | the elements' base felts in order (`element_felts`), then `linear_hash` |
//! | node | four felts as 32 canonical big-endian bytes (`digest_to_commitment`) |
//! | padding child | the zero digest, i.e. 32 zero bytes |
//! | `append_bytes(b)` | `len(b)`, then `b` as 8-byte big-endian felts, the last zero-padded |
//! | `append_field_element(x)` | `x`'s three coefficients |
//! | `state()` | lanes 0..4 of the sponge after a flush, big-endian — the grinding seed |
//! | `sample_u64(n)` | one squeezed felt, canonical, masked to `n − 1` (one felt per draw) |
//! | grinding | one width-8 permutation of the input's 8-byte felts, zero-padded; digest = lanes 0..4 |
//!
//! Grinding keeps the STARK's two-level construction (`crypto::grinding`): the
//! per-nonce hash is `[inner0..3, nonce, 0, 0, 0]` through the width-8
//! permutation, ZisK's cost (one width-8 permutation per nonce) over our seed
//! layout (ZisK's is `[c0, c1, c2, nonce, 0, 0, 0, 0]`).

use core::marker::PhantomData;
use core::num::NonZeroUsize;

use math::field::element::FieldElement;
use math::field::traits::{IsField, IsPrimeField};
use math::traits::AsBytes;

use crypto::fiat_shamir::is_transcript::{IsStarkTranscript, IsTranscript};
use crypto::fiat_shamir::transcript_hash::TranscriptHash;
use crypto::hash::poseidon1_stark::{self as p1, linear_hash};
use crypto::hash::{poseidon1_w8, poseidon1_w16};
use crypto::merkle_tree::traits::{IsLeafHasher, IsMerkleTreeBackend, IsStreamingLeafBackend};
use stark::config::{Commitment, CommitmentHash, DeviceTreeBackend, StarkHash};

use super::algebraic_commit::{
    BYTES_PER_FELT, commitment_to_digest, digest_to_commitment, element_felts, felts_from_bytes,
};
use crate::tables::types::{FE, FEE, GoldilocksExtension, GoldilocksField};

/// ZisK's leaf hash over felts, as a commitment.
fn leaf(felts: &[FE]) -> Commitment {
    digest_to_commitment(&linear_hash(felts))
}

/// ZisK's 4-ary node over four commitments.
fn node4(children: &[Commitment; 4]) -> Commitment {
    digest_to_commitment(&poseidon1_w16::compress4(
        &children.map(|c| commitment_to_digest(&c)),
    ))
}

/// The batched leaf backend: one leaf per row group.
#[derive(Clone, Debug, Default)]
pub struct P1BatchBackend<F> {
    _marker: PhantomData<fn() -> F>,
}

/// The FRI-layer backend: one leaf per fixed pair.
#[derive(Clone, Debug, Default)]
pub struct P1PairBackend<F> {
    _marker: PhantomData<fn() -> F>,
}

impl<F> IsMerkleTreeBackend for P1BatchBackend<F>
where
    F: IsField + 'static,
    FieldElement<F>: AsBytes + Sync + Send,
    Vec<FieldElement<F>>: Sync + Send,
{
    type Node = Commitment;
    type Data = Vec<FieldElement<F>>;
    const ARITY: usize = 4;

    fn hash_data(input: &Vec<FieldElement<F>>) -> Commitment {
        <Self as IsStreamingLeafBackend<F>>::hash_data_from_slices(input, &[])
    }

    fn hash_new_parent(_: &Commitment, _: &Commitment) -> Commitment {
        unreachable!("a Poseidon1 tree is 4-ary: no binary parent exists")
    }

    fn hash_four(children: &[Commitment; 4]) -> Commitment {
        node4(children)
    }

    fn padding_node() -> Commitment {
        [0u8; 32]
    }
}

impl<F> IsStreamingLeafBackend<F> for P1BatchBackend<F>
where
    F: IsField + 'static,
    FieldElement<F>: AsBytes + Sync + Send,
    Vec<FieldElement<F>>: Sync + Send,
{
    /// Equals [`IsMerkleTreeBackend::hash_data`] on the elements `data`
    /// encodes (`tests::hash_bytes_agrees_with_hash_data`).
    fn hash_bytes(data: &[u8]) -> Commitment {
        leaf(&felts_from_bytes(data))
    }

    fn hash_data_from_slices(a: &[FieldElement<F>], b: &[FieldElement<F>]) -> Commitment {
        let mut felts = Vec::with_capacity(3 * (a.len() + b.len()));
        for e in a.iter().chain(b.iter()) {
            element_felts(e, &mut felts);
        }
        leaf(&felts)
    }

    type LeafHasher = P1LeafHasher<F>;

    fn leaf_hasher() -> Self::LeafHasher {
        P1LeafHasher {
            felts: Vec::new(),
            _marker: PhantomData,
        }
    }
}

impl<F> IsMerkleTreeBackend for P1PairBackend<F>
where
    F: IsField + 'static,
    FieldElement<F>: AsBytes + Sync + Send,
{
    type Node = Commitment;
    type Data = [FieldElement<F>; 2];
    const ARITY: usize = 4;

    fn hash_data(input: &[FieldElement<F>; 2]) -> Commitment {
        let mut felts = Vec::with_capacity(6);
        element_felts(&input[0], &mut felts);
        element_felts(&input[1], &mut felts);
        leaf(&felts)
    }

    fn hash_new_parent(_: &Commitment, _: &Commitment) -> Commitment {
        unreachable!("a Poseidon1 tree is 4-ary: no binary parent exists")
    }

    fn hash_four(children: &[Commitment; 4]) -> Commitment {
        node4(children)
    }

    fn padding_node() -> Commitment {
        [0u8; 32]
    }
}

impl<F> DeviceTreeBackend for P1BatchBackend<F>
where
    Self: IsMerkleTreeBackend<Node = Commitment>,
{
    const COMMITMENT_HASH: CommitmentHash = CommitmentHash::Poseidon1;
}
impl<F> DeviceTreeBackend for P1PairBackend<F>
where
    Self: IsMerkleTreeBackend<Node = Commitment>,
{
    const COMMITMENT_HASH: CommitmentHash = CommitmentHash::Poseidon1;
}

/// The incremental leaf hasher. ZisK's leaf needs no length up front, but the
/// row arrives in slices of field elements, so it buffers their felts.
pub struct P1LeafHasher<F> {
    felts: Vec<FE>,
    _marker: PhantomData<fn() -> F>,
}

impl<F> IsLeafHasher<F> for P1LeafHasher<F>
where
    F: IsField,
    FieldElement<F>: AsBytes,
{
    type Node = Commitment;

    fn update(&mut self, data: &[FieldElement<F>]) {
        for e in data {
            element_felts(e, &mut self.felts);
        }
    }

    fn finalize(self) -> Commitment {
        leaf(&self.felts)
    }
}

// =========================================================================
// Grinding
// =========================================================================

/// The grinding digest: one width-8 permutation of the absorbed bytes read
/// as 8-byte big-endian felts, zero-padded to eight lanes; the digest is
/// lanes 0..4. Both grinding hashes fit: the inner input is 41 bytes (6
/// felts), the per-nonce input 40 (5 felts). A longer input, which grinding
/// never produces, takes the width-16 leaf hash so the digest stays total.
#[derive(Clone, Default)]
pub struct P1GrindDigest {
    buf: Vec<u8>,
}

impl P1GrindDigest {
    /// The digest of everything absorbed so far.
    pub fn finalize_digest(&self) -> Commitment {
        let felts = felts_from_bytes(&self.buf);
        if felts.len() > poseidon1_w8::STATE_FELTS {
            return leaf(&felts);
        }
        let mut s = [FE::zero(); poseidon1_w8::STATE_FELTS];
        s[..felts.len()].copy_from_slice(&felts);
        let out = poseidon1_w8::permute(s);
        digest_to_commitment(&[out[0], out[1], out[2], out[3]])
    }
}

impl stark::grinding::GrindDigest for P1GrindDigest {
    /// `math_cuda::grinding::generate_nonce_p1_gpu` (`p1s_grind_w8`): one
    /// width-8 permutation of `[inner0..3, nonce, 0, 0, 0]` per nonce, the
    /// per-nonce half of [`Self::finalize_digest`] over 40 bytes.
    const DEVICE_GRIND: Option<stark::grinding::DeviceGrindKey> =
        Some(stark::grinding::DeviceGrindKey::Poseidon1);
}

impl digest::HashMarker for P1GrindDigest {}

impl digest::OutputSizeUser for P1GrindDigest {
    type OutputSize = digest::typenum::U32;
}

impl digest::Update for P1GrindDigest {
    fn update(&mut self, data: &[u8]) {
        self.buf.extend_from_slice(data);
    }
}

impl digest::FixedOutput for P1GrindDigest {
    fn finalize_into(self, out: &mut digest::Output<Self>) {
        out.copy_from_slice(&self.finalize_digest());
    }
}

impl digest::Reset for P1GrindDigest {
    fn reset(&mut self) {
        self.buf.clear();
    }
}

impl digest::FixedOutputReset for P1GrindDigest {
    fn finalize_into_reset(&mut self, out: &mut digest::Output<Self>) {
        out.copy_from_slice(&self.finalize_digest());
        self.buf.clear();
    }
}

/// The Fiat–Shamir configuration: what grinding computes over. The challenge
/// stream is [`P1Transcript`], which callers build and hand to the prover.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct P1TranscriptHash;

impl TranscriptHash for P1TranscriptHash {
    type Digest = P1GrindDigest;
    const REVERSES_SQUEEZE: bool = false;
    const CANDIDATES_PER_COORDINATE: Option<NonZeroUsize> = NonZeroUsize::new(1);
    const NAME: &'static str = "poseidon1-w16";
}

/// ★ ZisK's Poseidon1 as a STARK commitment configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct P1StarkHash;

impl StarkHash for P1StarkHash {
    type Batched<F>
        = P1BatchBackend<F>
    where
        F: IsField + 'static,
        FieldElement<F>: AsBytes + Sync + Send;

    type Pair<F>
        = P1PairBackend<F>
    where
        F: IsField + 'static,
        FieldElement<F>: AsBytes + Sync + Send;

    type Transcript = P1TranscriptHash;

    const COMMITMENT_HASH: CommitmentHash = CommitmentHash::Poseidon1;

    const ARITY: usize = 4;
}

// The configuration's arity is its backends'.
const _: () = {
    assert!(<P1BatchBackend<GoldilocksField> as IsMerkleTreeBackend>::ARITY == P1StarkHash::ARITY);
    assert!(
        <P1PairBackend<GoldilocksExtension> as IsMerkleTreeBackend>::ARITY == P1StarkHash::ARITY
    );
};

// =========================================================================
// The transcript
// =========================================================================

/// ZisK's transcript sponge (`poseidon1_stark::Transcript`) under the
/// encodings of the module table.
#[derive(Clone)]
pub struct P1Transcript {
    sponge: p1::Transcript,
}

impl Default for P1Transcript {
    fn default() -> Self {
        Self::new()
    }
}

impl P1Transcript {
    /// An empty transcript.
    pub fn new() -> Self {
        Self {
            sponge: p1::Transcript::new(),
        }
    }

    /// A fresh transcript with `seed` absorbed as its first `append_bytes`.
    pub fn with_seed(seed: &[u8]) -> Self {
        let mut t = Self::new();
        <Self as IsTranscript<GoldilocksExtension>>::append_bytes(&mut t, seed);
        t
    }

    /// The felts `append_bytes(bytes)` absorbs: the length, then the bytes in
    /// 8-byte big-endian groups, the last zero-padded.
    pub fn append_bytes_felts(bytes: &[u8]) -> Vec<FE> {
        let mut felts = Vec::with_capacity(1 + bytes.len().div_ceil(BYTES_PER_FELT));
        felts.push(FE::from(bytes.len() as u64));
        felts.extend(felts_from_bytes(bytes));
        felts
    }
}

impl IsTranscript<GoldilocksExtension> for P1Transcript {
    fn append_field_element(&mut self, element: &FEE) {
        self.sponge.put(element.value());
    }

    fn append_bytes(&mut self, new_bytes: &[u8]) {
        self.sponge.put(&Self::append_bytes_felts(new_bytes));
    }

    /// Lanes 0..4 after a flush, canonical big-endian. Flushes a copy, so the
    /// stream itself is unchanged: prover and verifier read it at the same
    /// points and agree.
    fn state(&self) -> [u8; 32] {
        let s = self.sponge.clone().state();
        digest_to_commitment(&[s[0], s[1], s[2], s[3]])
    }

    fn sample_field_element(&mut self) -> FEE {
        FEE::new(self.sponge.field())
    }

    /// Constant consumption: one felt per draw, masked to a power-of-two bound.
    fn sample_u64(&mut self, upper_bound: u64) -> u64 {
        debug_assert!(
            upper_bound.is_power_of_two(),
            "sample_u64 is masked, so a non-power-of-two bound ({upper_bound}) would be biased"
        );
        GoldilocksField::canonical(self.sponge.squeeze1().value()) & (upper_bound - 1)
    }
}

impl IsStarkTranscript<GoldilocksExtension, GoldilocksField> for P1Transcript {}

#[cfg(test)]
#[path = "p1_commit_tests.rs"]
mod tests;
