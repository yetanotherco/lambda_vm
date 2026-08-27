//! Merkle commitment backends for an ALGEBRAIC hash — item A.
//!
//! One implementation, generic over the permutation. RPO, RPX and the Poseidon
//! reference are **type tags**, not three code paths: that is the objective the
//! whole lane is organised around, and it is what makes a fifth candidate cost
//! a permutation plus a const.
//!
//! # What made this small
//!
//! ✓ VERIFIED, and each of these was checked rather than assumed:
//!
//! - `IsMerkleTreeBackend::Node` is bound only to
//!   `PartialEq + Eq + Clone + Sync + Send` — **the trait never became
//!   byte-only**, and the algebraic backends that shipped in January 2026
//!   (`BatchPoseidonTree`, `TreePoseidon`) are still in-tree as proof.
//! - `StarkHash` pins `Node = Commitment` (32 bytes) and **a four-felt
//!   Goldilocks digest is exactly 32 canonical bytes**, so the node type does
//!   not fight us.
//! - `IsStreamingLeafBackend::Data` is already `Vec<FieldElement<F>>` — felts,
//!   not bytes.
//!
//! **BLAKE3's path is untouched.** These are sibling types beside
//! `FieldElementVectorBackend`, not a reparameterisation of it, which is what
//! keeps the measured BLAKE3 record safe from this work.
//!
//! # The conventions, and what pins them
//!
//! **Felt↔byte is canonical BIG-endian**, one rule shared with
//! [`super::algebraic_transcript`] and with `ByteConversion::write_bytes_be`.
//!
//! **A parent is `compress(left, right)`** — one permutation of
//! `[left ‖ right ‖ capacity]` with the compress domain, which is zero, so a
//! parent is literally `Rpo256::merge` and externally checkable.
//!
//! **A leaf is the rate-8 OVERWRITE DUPLEX** (RPO spec §2.6), the convention
//! this lane priced and adopted: capacity lane 0 carries the padding flag
//! `len mod 8`, capacity lane 1 the LEAF domain tag, and each block overwrites
//! the eight rate lanes with no field arithmetic at all. It absorbs eight fresh
//! felts per permutation where the socket's as-built leaf chain absorbs four,
//! which is worth 25% of the aggregation program.
//!
//! ⚠ **`hash_bytes` must equal `hash_data` on the elements those bytes
//! encode** — that is the `IsStreamingLeafBackend` contract, and it is the one
//! place an algebraic backend could silently disagree with itself, because the
//! byte route has to rebuild the felts the felt route was handed.
//! [`tests::hash_bytes_agrees_with_hash_data`] is the gate.
//!
//! # A1 — the incremental leaf hasher buffers
//!
//! `IsStreamingLeafBackend` requires an incremental `LeafHasher`, but **every
//! capacity-flag padding rule needs the total length before the first
//! permutation**. So [`AlgebraicLeafHasher`] buffers the leaf's felts and
//! sponges at `finalize`. The buffer is bounded by the leaf's row width — no
//! worse than the `hash_data(Vec)` route that already exists — and the
//! alternative would be inventing a padding rule, which is not this lane's to
//! invent.

use core::marker::PhantomData;
use core::num::NonZeroUsize;

use math::field::element::FieldElement;
use math::field::traits::{IsField, IsPrimeField};
use math::traits::{AsBytes, ByteConversion};

use crypto::fiat_shamir::transcript_hash::TranscriptHash;
use crypto::merkle_tree::traits::{IsLeafHasher, IsMerkleTreeBackend, IsStreamingLeafBackend};
use stark::config::Commitment;

use super::hash::{HASH_STATE_FELTS, HasherKind, LfmHasher};
use super::rpo::{DOMAIN_LEAF, RATE_FELTS, domain_iv};
use super::word::LfmWord;
use crate::tables::types::{FE, GoldilocksField};

/// Bytes one Goldilocks felt serialises to — `ByteConversion::BYTE_LEN`.
pub const BYTES_PER_FELT: usize = 8;
/// Felts in a digest, hence `Commitment`'s 32 bytes.
pub const DIGEST_FELTS: usize = 4;

/// A type-level name for one algebraic permutation.
///
/// The whole reason the backends below are one implementation: a candidate
/// joins by adding a unit struct and a `KIND`, and nothing else here moves.
pub trait AlgebraicHasher: Clone + Copy + Default + Send + Sync + 'static {
    /// The permutation the `LFM_HASH` socket proves for this commitment.
    const KIND: HasherKind;
}

/// Rescue-Prime Optimized.
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub struct RpoCommit;
impl AlgebraicHasher for RpoCommit {
    const KIND: HasherKind = HasherKind::Rpo;
}

/// Rescue-Prime eXtended (XHash12).
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub struct RpxCommit;
impl AlgebraicHasher for RpxCommit {
    const KIND: HasherKind = HasherKind::Rpx;
}

/// ⚠ Poseidon-original — **UNSHIPPABLE** (broken family; eprint 2026/306 and
/// 2026/1692). Present so the comparison has its priced reference column and
/// its control, never as a candidate.
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub struct PoseidonCommit;
impl AlgebraicHasher for PoseidonCommit {
    const KIND: HasherKind = HasherKind::Poseidon;
}

/// Four felts as a 32-byte `Commitment`, canonical big-endian.
pub fn digest_to_commitment(d: &LfmWord) -> Commitment {
    let mut out = [0u8; 32];
    for (i, f) in d.iter().enumerate() {
        let v = GoldilocksField::canonical(f.value());
        out[i * BYTES_PER_FELT..(i + 1) * BYTES_PER_FELT].copy_from_slice(&v.to_be_bytes());
    }
    out
}

/// A 32-byte `Commitment` back to four felts.
pub fn commitment_to_digest(c: &Commitment) -> LfmWord {
    core::array::from_fn(|i| {
        let mut b = [0u8; BYTES_PER_FELT];
        b.copy_from_slice(&c[i * BYTES_PER_FELT..(i + 1) * BYTES_PER_FELT]);
        FE::from(u64::from_be_bytes(b))
    })
}

/// ★ The rate-8 overwrite duplex over a felt stream — the leaf construction.
///
/// Capacity lane 0 is the padding flag `len mod 8`, lane 1 the LEAF domain.
/// Each block OVERWRITES the eight rate lanes (spec §2.6), so absorption costs
/// no field arithmetic outside the permutation; the tail block is zero-padded.
/// ★ **THE LEAF CAPACITY RULE, stated once.** Lane 0 is the padding flag
/// `len mod 8` — zero when the length divides the rate, which is why no
/// trailing block is spent on an exact multiple — and lane 1 the LEAF domain.
///
/// ⚠ Exported so the MACHINE side derives it rather than restating it. Under
/// `MODE_P` the capacity is program data, so an emitter must supply exactly
/// this word; a second definition of it is a root nobody can reproduce.
pub fn leaf_capacity(num_felts: usize) -> LfmWord {
    let iv = domain_iv(DOMAIN_LEAF);
    let mut cap: LfmWord = core::array::from_fn(|k| FE::from(iv[k]));
    cap[0] = FE::from((num_felts % RATE_FELTS) as u64);
    cap
}

/// ★ **The three `MODE_P` input cells for a SINGLE-BLOCK leaf sponge** — the
/// rule, exported for the same reason [`leaf_capacity`] is.
///
/// Two rate cells carrying up to eight felts zero-padded, then the capacity.
/// This is the shape grinding takes (`state ‖ nonce` is 40 bytes, five felts,
/// one block), and the shape the duplex emitter's first block takes.
pub fn single_block_leaf_cells(felts: &[FE]) -> [LfmWord; 3] {
    debug_assert!(
        felts.len() <= RATE_FELTS,
        "a single block carries at most the rate ({RATE_FELTS} felts), got {}",
        felts.len()
    );
    let lane = |i: usize| felts.get(i).copied().unwrap_or_else(FE::zero);
    [
        [lane(0), lane(1), lane(2), lane(3)],
        [lane(4), lane(5), lane(6), lane(7)],
        leaf_capacity(felts.len()),
    ]
}

pub fn sponge_leaf(kind: HasherKind, felts: &[FE]) -> LfmWord {
    let mut state = [FE::zero(); HASH_STATE_FELTS];
    let cap = leaf_capacity(felts.len());
    state[RATE_FELTS..].copy_from_slice(&cap);

    if felts.is_empty() {
        return [state[0], state[1], state[2], state[3]];
    }
    for block in felts.chunks(RATE_FELTS) {
        for (lane, slot) in state.iter_mut().take(RATE_FELTS).enumerate() {
            *slot = block.get(lane).copied().unwrap_or_else(FE::zero);
        }
        state = kind.permute(state);
    }
    [state[0], state[1], state[2], state[3]]
}

/// Every 8-byte big-endian group of `bytes` as a felt.
///
/// The inverse of the serialisation `ByteConversion::write_bytes_be` performs,
/// which is how leaves reach a backend. A trailing partial group is
/// zero-extended on the LOW side, matching how a short write would land.
pub fn felts_from_bytes(bytes: &[u8]) -> Vec<FE> {
    bytes
        .chunks(BYTES_PER_FELT)
        .map(|c| {
            let mut b = [0u8; BYTES_PER_FELT];
            b[..c.len()].copy_from_slice(c);
            FE::from(u64::from_be_bytes(b))
        })
        .collect()
}

/// Decompose a field element — base or extension — into its base felts, by the
/// same serialisation the STARK uses.
fn element_felts<F>(e: &FieldElement<F>, out: &mut Vec<FE>)
where
    F: IsField,
    FieldElement<F>: AsBytes + ByteConversion,
{
    let mut buf = [0u8; 64];
    let len = <FieldElement<F> as ByteConversion>::BYTE_LEN;
    debug_assert!(len <= buf.len(), "a field element must fit the scratch");
    e.write_bytes_be(&mut buf[..len]);
    out.extend(felts_from_bytes(&buf[..len]));
}

/// The batched leaf backend — one leaf per row group.
#[derive(Clone, Debug, Default)]
pub struct AlgebraicBatchBackend<F, H> {
    _marker: PhantomData<fn() -> (F, H)>,
}

/// The FRI-layer backend — one leaf per fixed pair, no `Vec` per leaf.
#[derive(Clone, Debug, Default)]
pub struct AlgebraicPairBackend<F, H> {
    _marker: PhantomData<fn() -> (F, H)>,
}

/// A parent: `compress(left, right)`, one permutation, the compress domain.
fn parent<H: AlgebraicHasher>(left: &Commitment, right: &Commitment) -> Commitment {
    let l = commitment_to_digest(left);
    let r = commitment_to_digest(right);
    digest_to_commitment(&H::KIND.compress(&l, &r))
}

impl<F, H> IsMerkleTreeBackend for AlgebraicBatchBackend<F, H>
where
    F: IsField + 'static,
    H: AlgebraicHasher,
    FieldElement<F>: AsBytes + ByteConversion + Sync + Send,
    Vec<FieldElement<F>>: Sync + Send,
{
    type Node = Commitment;
    type Data = Vec<FieldElement<F>>;

    fn hash_data(input: &Vec<FieldElement<F>>) -> Commitment {
        <Self as IsStreamingLeafBackend<F>>::hash_data_from_slices(input, &[])
    }

    fn hash_new_parent(left: &Commitment, right: &Commitment) -> Commitment {
        parent::<H>(left, right)
    }
}

impl<F, H> IsStreamingLeafBackend<F> for AlgebraicBatchBackend<F, H>
where
    F: IsField + 'static,
    H: AlgebraicHasher,
    FieldElement<F>: AsBytes + ByteConversion + Sync + Send,
    Vec<FieldElement<F>>: Sync + Send,
{
    /// ⚠ Must equal [`IsMerkleTreeBackend::hash_data`] on the elements `data`
    /// encodes — the trait's contract, and the gate is
    /// `tests::hash_bytes_agrees_with_hash_data`.
    fn hash_bytes(data: &[u8]) -> Commitment {
        digest_to_commitment(&sponge_leaf(H::KIND, &felts_from_bytes(data)))
    }

    fn hash_data_from_slices(a: &[FieldElement<F>], b: &[FieldElement<F>]) -> Commitment {
        let mut felts = Vec::with_capacity((a.len() + b.len()) * DIGEST_FELTS);
        for e in a.iter().chain(b.iter()) {
            element_felts(e, &mut felts);
        }
        digest_to_commitment(&sponge_leaf(H::KIND, &felts))
    }

    type LeafHasher = AlgebraicLeafHasher<F, H>;

    fn leaf_hasher() -> Self::LeafHasher {
        AlgebraicLeafHasher {
            felts: Vec::new(),
            _marker: PhantomData,
        }
    }
}

impl<F, H> IsMerkleTreeBackend for AlgebraicPairBackend<F, H>
where
    F: IsField + 'static,
    H: AlgebraicHasher,
    FieldElement<F>: AsBytes + ByteConversion + Sync + Send,
{
    type Node = Commitment;
    type Data = [FieldElement<F>; 2];

    fn hash_data(input: &[FieldElement<F>; 2]) -> Commitment {
        let mut felts = Vec::with_capacity(2 * DIGEST_FELTS);
        element_felts(&input[0], &mut felts);
        element_felts(&input[1], &mut felts);
        digest_to_commitment(&sponge_leaf(H::KIND, &felts))
    }

    fn hash_new_parent(left: &Commitment, right: &Commitment) -> Commitment {
        parent::<H>(left, right)
    }
}

/// ★ A1 — the incremental leaf hasher, which BUFFERS.
///
/// The padding flag is `len mod 8` and the sponge needs it in the capacity
/// before the first permutation, so an incremental sponge cannot start until
/// the length is known. Buffering the leaf's felts is bounded by the row width
/// and is no worse than the `hash_data(Vec)` route; inventing a length-free
/// padding rule instead would be a cryptographic decision this lane does not
/// get to make.
pub struct AlgebraicLeafHasher<F, H> {
    felts: Vec<FE>,
    /// `fn() -> T` rather than `T`, so the marker is unconditionally `Send` and
    /// `Sync` without an `unsafe impl`. The trait requires `Send` because there
    /// is one hasher per leaf and a real epoch's base layer has millions of
    /// them, absorbed in parallel.
    _marker: PhantomData<fn() -> (F, H)>,
}

impl<F, H> IsLeafHasher<F> for AlgebraicLeafHasher<F, H>
where
    F: IsField,
    H: AlgebraicHasher,
    FieldElement<F>: AsBytes + ByteConversion,
{
    type Node = Commitment;

    fn update(&mut self, data: &[FieldElement<F>]) {
        for e in data {
            element_felts(e, &mut self.felts);
        }
    }

    fn finalize(self) -> Commitment {
        digest_to_commitment(&sponge_leaf(H::KIND, &self.felts))
    }
}

// =========================================================================
// The GRINDING hash — `StarkHash::Transcript`
// =========================================================================

/// A `digest::Digest` over an algebraic permutation, for the ONE thing a
/// configuration's `Transcript` associated type still decides: **grinding**.
///
/// # Why this exists at all
///
/// The CHALLENGE stream does not flow through here.
/// [`super::algebraic_transcript::AlgebraicTranscript`] carries that, because
/// `prove`/`verify` take the transcript as a parameter. What
/// `StarkHash::Transcript` forces is what the prover and verifier derive
/// *internally from the configuration* — grinding, whose seed is
/// `transcript.state()` and whose proof-of-work is `H(seed ‖ nonce)`.
///
/// ⚠ **And grinding is verified IN-VM** (`transcript_replay.rs` carries it; the
/// verifier absorbs `nonce_value.to_be_bytes()`). So whatever hash grinds, the
/// machine must be able to compute it. Leaving BLAKE3 here would keep the
/// 3,056-column slot-11 chip in an algebraic branch's AIR set — which would
/// make the four-branch comparison measure something other than the hash it
/// names. That is why this type exists rather than a `Blake3TranscriptHash`
/// alias.
///
/// # The construction is the LEAF one, deliberately
///
/// Grinding hashes a byte string — `state ‖ nonce`, 40 bytes, five felts — which
/// is DATA, exactly what a leaf is. It therefore reuses [`sponge_leaf`] and the
/// LEAF domain rather than inventing a fourth one.
///
/// ⚖ That is not only simplicity. The socket pins per-mode capacities for
/// `MODE_C`/`MODE_T`/`MODE_L`, so **a grinding-specific domain is not something
/// the in-VM verifier could emit with an existing mode.** Using the leaf
/// construction is what keeps the machine side expressible. The domain reuse is
/// not exploitable: the grinding check tests leading zeros of a hash whose
/// preimage is transcript-bound, so colliding it with some leaf digest buys an
/// adversary nothing.
///
/// # Buffering, for the third time and the same reason
///
/// `sponge_leaf`'s padding flag is `len mod 8`, needed in the capacity before
/// the first permutation, so this accumulates bytes and sponges at finalize —
/// the same resolution A1 and the transcript reached.
pub struct AlgebraicDigest<H> {
    buf: Vec<u8>,
    _marker: PhantomData<fn() -> H>,
}

impl<H> Default for AlgebraicDigest<H> {
    fn default() -> Self {
        Self {
            buf: Vec::new(),
            _marker: PhantomData,
        }
    }
}

impl<H> Clone for AlgebraicDigest<H> {
    fn clone(&self) -> Self {
        Self {
            buf: self.buf.clone(),
            _marker: PhantomData,
        }
    }
}

impl<H: AlgebraicHasher> AlgebraicDigest<H> {
    /// The digest of everything absorbed so far — the leaf construction over
    /// the buffered bytes.
    pub fn finalize_digest(&self) -> Commitment {
        digest_to_commitment(&sponge_leaf(H::KIND, &felts_from_bytes(&self.buf)))
    }
}

impl<H> digest::HashMarker for AlgebraicDigest<H> {}

impl<H> digest::OutputSizeUser for AlgebraicDigest<H> {
    type OutputSize = digest::typenum::U32;
}

impl<H: AlgebraicHasher> digest::Update for AlgebraicDigest<H> {
    fn update(&mut self, data: &[u8]) {
        self.buf.extend_from_slice(data);
    }
}

impl<H: AlgebraicHasher> digest::FixedOutput for AlgebraicDigest<H> {
    fn finalize_into(self, out: &mut digest::Output<Self>) {
        out.copy_from_slice(&self.finalize_digest());
    }
}

impl<H: AlgebraicHasher> digest::Reset for AlgebraicDigest<H> {
    fn reset(&mut self) {
        self.buf.clear();
    }
}

impl<H: AlgebraicHasher> digest::FixedOutputReset for AlgebraicDigest<H> {
    fn finalize_into_reset(&mut self, out: &mut digest::Output<Self>) {
        out.copy_from_slice(&self.finalize_digest());
        self.buf.clear();
    }
}

/// The Fiat–Shamir configuration for an algebraic hash.
///
/// `CANDIDATES_PER_COORDINATE` is **`Some(1)`, and guaranteed rather than
/// probabilistic** — the strongest schedule any configuration here has.
/// A squeeze yields four felts that are canonical BY CONSTRUCTION, so the
/// `u64`s carved out of their 32 canonical bytes are *always* in the field's
/// range. BLAKE3 needs two candidates because its bytes are arbitrary and a
/// single miss has nowhere to go; an algebraic squeeze cannot miss.
macro_rules! algebraic_transcript_hash {
    ($name:ident, $tag:ty, $label:literal, $doc:literal) => {
        #[doc = $doc]
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub struct $name;

        impl TranscriptHash for $name {
            type Digest = AlgebraicDigest<$tag>;
            const CANDIDATES_PER_COORDINATE: Option<NonZeroUsize> = NonZeroUsize::new(1);
            const NAME: &'static str = $label;
        }
    };
}

algebraic_transcript_hash!(
    RpoTranscriptHash,
    RpoCommit,
    "rpo256",
    "The RPO256 Fiat–Shamir configuration."
);
algebraic_transcript_hash!(
    RpxTranscriptHash,
    RpxCommit,
    "rpx256",
    "The RPX256 (XHash12) Fiat–Shamir configuration."
);
algebraic_transcript_hash!(
    PoseidonTranscriptHash,
    PoseidonCommit,
    "poseidon-goldilocks",
    "⚠ The Poseidon Fiat–Shamir configuration — UNSHIPPABLE, reference only."
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tables::types::{FEE, GoldilocksExtension};

    type Base = GoldilocksField;
    type Ext = GoldilocksExtension;

    /// Every tenant, so a break in the CONSTRUCTION shows as all-three-fail
    /// rather than being mistaken for something about one permutation.
    macro_rules! for_each_tenant {
        ($body:ident) => {
            $body::<RpoCommit>("Rpo");
            $body::<RpxCommit>("Rpx");
            $body::<PoseidonCommit>("Poseidon");
        };
    }

    fn base_leaf(n: usize) -> Vec<FieldElement<Base>> {
        (0..n as u64).map(|i| FE::from(i * 7 + 1)).collect()
    }

    fn ext_leaf(n: usize) -> Vec<FEE> {
        (0..n as u64)
            .map(|i| FEE::new([FE::from(i + 1), FE::from(i + 2), FE::from(i + 3)]))
            .collect()
    }

    /// ★★ **The `IsStreamingLeafBackend` contract**: `hash_bytes` must equal
    /// `hash_data` on the elements those bytes encode.
    ///
    /// This is the one place an algebraic backend can silently disagree with
    /// itself, because the byte route has to rebuild exactly the felts the felt
    /// route was handed. Checked over BASE and EXTENSION leaves — the extension
    /// is the interesting case, since one element is three felts.
    #[test]
    fn hash_bytes_agrees_with_hash_data() {
        fn check<H: AlgebraicHasher>(name: &str) {
            for n in [1usize, 2, 7, 8, 9, 16, 17, 33] {
                let leaf = base_leaf(n);
                let mut buf = Vec::new();
                for e in &leaf {
                    let mut b = [0u8; 8];
                    e.write_bytes_be(&mut b);
                    buf.extend_from_slice(&b);
                }
                assert_eq!(
                    <AlgebraicBatchBackend<Base, H> as IsStreamingLeafBackend<Base>>::hash_bytes(
                        &buf
                    ),
                    <AlgebraicBatchBackend<Base, H> as IsMerkleTreeBackend>::hash_data(&leaf),
                    "{name}: base leaf of {n}"
                );

                let leaf = ext_leaf(n);
                let mut buf = Vec::new();
                for e in &leaf {
                    let mut b = [0u8; 24];
                    e.write_bytes_be(&mut b);
                    buf.extend_from_slice(&b);
                }
                assert_eq!(
                    <AlgebraicBatchBackend<Ext, H> as IsStreamingLeafBackend<Ext>>::hash_bytes(
                        &buf
                    ),
                    <AlgebraicBatchBackend<Ext, H> as IsMerkleTreeBackend>::hash_data(&leaf),
                    "{name}: extension leaf of {n}"
                );
            }
        }
        for_each_tenant!(check);
    }

    /// ★ **A1's contract: splitting is free.** For any partition of a leaf's
    /// elements into consecutive chunks, updating with each in order and
    /// finalizing must equal `hash_data` over the whole — a backend whose
    /// framing depended on where the updates fell would produce leaves no
    /// verifier could re-derive.
    #[test]
    fn the_incremental_leaf_hasher_agrees_with_hash_data_under_every_split() {
        fn check<H: AlgebraicHasher>(name: &str) {
            let leaf = base_leaf(19);
            let want = <AlgebraicBatchBackend<Base, H> as IsMerkleTreeBackend>::hash_data(&leaf);
            for cut in 0..=leaf.len() {
                let mut h =
                    <AlgebraicBatchBackend<Base, H> as IsStreamingLeafBackend<Base>>::leaf_hasher();
                h.update(&leaf[..cut]);
                h.update(&leaf[cut..]);
                assert_eq!(h.finalize(), want, "{name}: split at {cut}");
            }
            // And a three-way split, so the property is not an artifact of two.
            let mut h =
                <AlgebraicBatchBackend<Base, H> as IsStreamingLeafBackend<Base>>::leaf_hasher();
            h.update(&leaf[..3]);
            h.update(&leaf[3..11]);
            h.update(&leaf[11..]);
            assert_eq!(h.finalize(), want, "{name}: three-way split");
        }
        for_each_tenant!(check);
    }

    /// A parent is the socket's `compress` — one permutation, the compress
    /// domain, which is zero. Under RPO that makes it literally
    /// `Rpo256::merge`, checkable against miden by someone who has never seen
    /// this codebase.
    #[test]
    fn a_parent_is_the_socket_compress() {
        fn check<H: AlgebraicHasher>(name: &str) {
            let l = digest_to_commitment(&[FE::from(1u64), FE::from(2), FE::from(3), FE::from(4)]);
            let r = digest_to_commitment(&[FE::from(5u64), FE::from(6), FE::from(7), FE::from(8)]);
            let got =
                <AlgebraicBatchBackend<Base, H> as IsMerkleTreeBackend>::hash_new_parent(&l, &r);
            let want = digest_to_commitment(
                &H::KIND.compress(&commitment_to_digest(&l), &commitment_to_digest(&r)),
            );
            assert_eq!(got, want, "{name}");
            // The pair backend must agree with the batched one on parents, or
            // one tree's internal nodes are not the other's.
            assert_eq!(
                <AlgebraicPairBackend<Base, H> as IsMerkleTreeBackend>::hash_new_parent(&l, &r),
                got,
                "{name}: pair and batched parents must be one function"
            );
        }
        for_each_tenant!(check);
    }

    /// A digest round-trips through its 32 canonical big-endian bytes, or
    /// `Commitment` does not name the digest.
    #[test]
    fn a_digest_round_trips_through_its_commitment_bytes() {
        let d: LfmWord = [
            FE::from(1u64),
            FE::from(0xFFFF_FFFF_0000_0000u64),
            FE::zero(),
            FE::from(0x0123_4567_89AB_CDEFu64),
        ];
        assert_eq!(commitment_to_digest(&digest_to_commitment(&d)), d);
    }

    /// ⚠ **The permutation count is `⌈felts/8⌉`, and an exact multiple spends
    /// NO trailing block.** The rate-8 census invariance rests on exactly this,
    /// so it is asserted against the same closed form the census uses rather
    /// than left to the reader.
    #[test]
    fn the_block_count_is_the_census_closed_form() {
        use crate::lfm::edsl::WrapHash;
        use crate::lfm::epoch_verify::blocks_for;
        for felts in 1..=512usize {
            assert_eq!(
                felts.div_ceil(RATE_FELTS),
                blocks_for(felts, WrapHash::Blake3),
                "{felts} felts must cost the census's block count"
            );
        }
        // The boundary the padding flag exists for: a rate multiple carries a
        // zero flag, a partial block carries its remainder.
        assert_eq!(16 % RATE_FELTS, 0);
        assert_eq!(17 % RATE_FELTS, 1);
    }

    /// The three tenants must be three different commitments — otherwise the
    /// generic backend has collapsed them.
    #[test]
    fn the_three_tenants_commit_differently() {
        let leaf = base_leaf(12);
        let rpo = <AlgebraicBatchBackend<Base, RpoCommit> as IsMerkleTreeBackend>::hash_data(&leaf);
        let rpx = <AlgebraicBatchBackend<Base, RpxCommit> as IsMerkleTreeBackend>::hash_data(&leaf);
        let pos =
            <AlgebraicBatchBackend<Base, PoseidonCommit> as IsMerkleTreeBackend>::hash_data(&leaf);
        assert_ne!(rpo, rpx);
        assert_ne!(rpo, pos);
        assert_ne!(rpx, pos);
    }

    /// ★★ **THE GRINDING GATE** — the host grinding digest and the in-VM
    /// computation of it must agree, for every tenant.
    ///
    /// ⚠ Grinding is verified INSIDE the machine (`transcript_replay.rs`), so
    /// this is not a host-only concern: if the two sides disagree, a proof-of-
    /// work the prover found is one the verifier cannot confirm, and that fails
    /// as an unprovable program rather than as a wrong answer.
    ///
    /// The preimage is `state ‖ nonce` — 32 + 8 bytes, five felts, **one rate
    /// block** — so the machine side is a single `MODE_P` row. Every cell it
    /// feeds is DERIVED from the exported rules ([`single_block_leaf_cells`]),
    /// never restated: this lane has now written a host↔machine encoding three
    /// times and twice a differential caught a machine side that had
    /// hand-written a constant agreeing with the rule only until the rule moved.
    #[test]
    fn the_host_grinding_digest_and_the_machine_agree() {
        use crate::lfm::builder::LfmBuilder;
        use crate::lfm::compiler::compile;
        use crate::lfm::proof::{lfm_prove_with_hasher, verify_against};
        use crate::lfm::registry::build_artifacts_with_hasher;
        use stark::proof::options::GoldilocksCubicProofOptions;

        const SEED: [u8; 32] = [
            0x9e, 0x37, 0x79, 0xb9, 0x7f, 0x4a, 0x7c, 0x00, 0xd1, 0xb5, 0x4a, 0x32, 0xd1, 0x92,
            0xed, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x00, 0x11, 0x22, 0x33, 0x44,
            0x55, 0x66, 0x77, 0x00,
        ];
        const NONCE: u64 = 0x0123_4567_89AB_CD00;

        fn check<H: AlgebraicHasher>(name: &str) {
            let opts = GoldilocksCubicProofOptions::with_blowup(2).expect("options");

            // HOST: the grinding digest through the public `digest` interface.
            let mut d = AlgebraicDigest::<H>::default();
            digest::Digest::update(&mut d, SEED);
            digest::Digest::update(&mut d, NONCE.to_be_bytes());
            let want = d.finalize_digest();
            // ⚠ The trait path must agree with the inherent one, or grinding —
            // which reaches this through `digest::Digest` — computes something
            // this test never checked.
            let via_trait: Commitment = digest::Digest::finalize(d.clone()).into();
            assert_eq!(
                via_trait, want,
                "{name}: Digest::finalize must be finalize_digest"
            );

            // The preimage's felts, and the MODE_P cells, both from the rules.
            let mut preimage = SEED.to_vec();
            preimage.extend_from_slice(&NONCE.to_be_bytes());
            let felts = felts_from_bytes(&preimage);
            assert_eq!(felts.len(), 5, "state ‖ nonce is five felts, one block");
            let cells = single_block_leaf_cells(&felts);

            // MACHINE: one permutation row, publishing the digest cell.
            let mut b = LfmBuilder::new();
            let arena = b.declare_arena(2);
            let rate0 = b.hint_word(arena, 0);
            let rate1 = b.hint_word(arena, 1);
            let cap = b.digest_const(cells[2]);
            let out = b.permute([rate0, rate1, cap.as_cell()]);
            b.public(out[0]);
            let program = compile(b.finish());

            let artifacts = build_artifacts_with_hasher(&program, &opts, H::KIND);
            let proved = lfm_prove_with_hasher(
                &program,
                &artifacts,
                &[vec![cells[0], cells[1]]],
                &opts,
                H::KIND,
            )
            .expect("the grinding row must prove");

            assert_eq!(
                digest_to_commitment(&proved.public_words[0].1),
                want,
                "{name}: the machine's grinding digest must be the host's"
            );
            assert!(
                verify_against(
                    &artifacts.roots,
                    &artifacts.program_id,
                    artifacts.keccak_rnd_chunks,
                    &proved.proof,
                    &proved.public_words,
                    &opts,
                    artifacts.hasher,
                    artifacts.chip_set,
                ),
                "{name}: the grinding proof must verify"
            );
        }
        for_each_tenant!(check);
    }

    /// The `digest` interface must agree with the direct construction — i.e.
    /// `update`-then-finalize is the leaf sponge over the concatenation, so a
    /// caller that splits its updates gets the same grinding digest.
    #[test]
    fn the_grinding_digest_is_split_invariant() {
        fn check<H: AlgebraicHasher>(name: &str) {
            let msg: Vec<u8> = (0..40u8).collect();
            let want = digest_to_commitment(&sponge_leaf(H::KIND, &felts_from_bytes(&msg)));
            for cut in [0usize, 1, 8, 17, 32, 40] {
                let mut d = AlgebraicDigest::<H>::default();
                digest::Digest::update(&mut d, &msg[..cut]);
                digest::Digest::update(&mut d, &msg[cut..]);
                assert_eq!(d.finalize_digest(), want, "{name}: split at {cut}");
            }
        }
        for_each_tenant!(check);
    }

    /// ★★★ **THE EMITTER GATE** — the in-VM leaf and parent must equal this
    /// module's host constructions, for every tenant.
    ///
    /// This is the pair the whole migration turns on: the host commits with
    /// `hash_data` / `hash_new_parent`, and the wrap program re-derives them
    /// with `WrapHash::Algebraic`'s `leaf_hash` / `hash_pair`. If they disagree
    /// the walk reconstructs nothing and the failure surfaces as a `DivByZero`
    /// deep in a query walk, naming neither the hash nor the site — so it is
    /// gated here, where a divergence is a failing test.
    ///
    /// The leaf is exercised at lengths that cross the rate boundary in both
    /// directions, because the padding flag (`len mod 8`) is the one part of
    /// the construction that is not the same on every block.
    #[test]
    fn the_emitted_leaf_and_parent_equal_the_host_backend() {
        use crate::lfm::builder::LfmBuilder;
        use crate::lfm::compiler::compile;
        use crate::lfm::edsl::{WrapDigest, WrapHash};
        use crate::lfm::proof::lfm_prove_with_hasher;
        use crate::lfm::registry::build_artifacts_with_hasher;
        use stark::proof::options::GoldilocksCubicProofOptions;

        fn check<H: AlgebraicHasher>(name: &str) {
            let opts = GoldilocksCubicProofOptions::with_blowup(2).expect("options");

            for n in [1usize, 4, 8, 9, 12, 16] {
                let leaf = base_leaf(n);
                let want =
                    <AlgebraicBatchBackend<Base, H> as IsMerkleTreeBackend>::hash_data(&leaf);

                // The program reads the leaf's felts from an arena, four per
                // word, and hashes them the way a wrap program would.
                let words = n.div_ceil(4);
                let mut b = LfmBuilder::new().with_wrap_hash(WrapHash::Algebraic);
                let arena = b.declare_arena(words as u32);
                let felts: Vec<_> = (0..words)
                    .flat_map(|w| {
                        let c = b.hint_word(arena, w as u32);
                        b.unpack(c).to_vec()
                    })
                    .take(n)
                    .collect();
                let d = WrapHash::Algebraic.leaf_hash(&mut b, &felts);
                assert_eq!(d.len(), 1, "{name}: an algebraic digest is ONE cell");
                b.public(d[0]);
                let program = compile(b.finish());

                let arena_words: Vec<LfmWord> = (0..words)
                    .map(|w| {
                        core::array::from_fn(|i| {
                            leaf.get(4 * w + i).copied().unwrap_or_else(FE::zero)
                        })
                    })
                    .collect();
                let artifacts = build_artifacts_with_hasher(&program, &opts, H::KIND);
                let proved =
                    lfm_prove_with_hasher(&program, &artifacts, &[arena_words], &opts, H::KIND)
                        .expect("the leaf program must prove");
                assert_eq!(
                    digest_to_commitment(&proved.public_words[0].1),
                    want,
                    "{name}: emitted leaf of {n} felts must equal the host's"
                );
            }

            // ★ EXTENSION leaves — the decomposition the call sites rely on.
            //
            // `sub_proof::emit_leaf_hash` and `batched_epoch_verify` absorb an
            // Fp3 value as `unpack(cell)[..3]`, deleting the byte serialisation.
            // That is correct only if the host's own decomposition agrees:
            // `write_bytes_be` for an Fp3 element writes components 0, 1, 2 in
            // order. Verified by reading, and gated here so it stays true.
            for n in [1usize, 2, 3, 5] {
                let leaf = ext_leaf(n);
                let want = <AlgebraicBatchBackend<Ext, H> as IsMerkleTreeBackend>::hash_data(&leaf);

                let mut b = LfmBuilder::new().with_wrap_hash(WrapHash::Algebraic);
                let arena = b.declare_arena(n as u32);
                let felts: Vec<_> = (0..n)
                    .flat_map(|i| {
                        let c = b.hint_word(arena, i as u32);
                        b.unpack(c)[..3].to_vec()
                    })
                    .collect();
                let d = WrapHash::Algebraic.leaf_hash(&mut b, &felts);
                b.public(d[0]);
                let program = compile(b.finish());

                let words: Vec<LfmWord> = leaf
                    .iter()
                    .map(|e| {
                        let v = e.value();
                        [v[0], v[1], v[2], FE::zero()]
                    })
                    .collect();
                let artifacts = build_artifacts_with_hasher(&program, &opts, H::KIND);
                let proved = lfm_prove_with_hasher(&program, &artifacts, &[words], &opts, H::KIND)
                    .expect("the extension leaf program must prove");
                assert_eq!(
                    digest_to_commitment(&proved.public_words[0].1),
                    want,
                    "{name}: emitted EXTENSION leaf of {n} elements must equal the host's"
                );
            }

            // And the parent.
            let l = digest_to_commitment(&[FE::from(3u64), FE::from(5), FE::from(7), FE::from(11)]);
            let r =
                digest_to_commitment(&[FE::from(13u64), FE::from(17), FE::from(19), FE::from(23)]);
            let want =
                <AlgebraicBatchBackend<Base, H> as IsMerkleTreeBackend>::hash_new_parent(&l, &r);

            let mut b = LfmBuilder::new().with_wrap_hash(WrapHash::Algebraic);
            let arena = b.declare_arena(2);
            let lc = WrapDigest::from_cell(b.hint_word(arena, 0));
            let rc = WrapDigest::from_cell(b.hint_word(arena, 1));
            let d = WrapHash::Algebraic.hash_pair(&mut b, lc, rc);
            b.public(d[0]);
            let program = compile(b.finish());
            let artifacts = build_artifacts_with_hasher(&program, &opts, H::KIND);
            let proved = lfm_prove_with_hasher(
                &program,
                &artifacts,
                &[vec![commitment_to_digest(&l), commitment_to_digest(&r)]],
                &opts,
                H::KIND,
            )
            .expect("the parent program must prove");
            assert_eq!(
                digest_to_commitment(&proved.public_words[0].1),
                want,
                "{name}: emitted parent must equal the host's"
            );
        }
        for_each_tenant!(check);
    }

    /// The leaf is DOMAIN-SEPARATED from a parent: hashing the eight felts of
    /// two digests as a LEAF must not equal compressing them as a PARENT.
    #[test]
    fn a_leaf_over_eight_felts_is_not_the_parent_of_those_two_digests() {
        fn check<H: AlgebraicHasher>(name: &str) {
            let felts: Vec<FieldElement<Base>> = (1..=8u64).map(FE::from).collect();
            let leaf = <AlgebraicBatchBackend<Base, H> as IsMerkleTreeBackend>::hash_data(&felts);
            let l = digest_to_commitment(&[felts[0], felts[1], felts[2], felts[3]]);
            let r = digest_to_commitment(&[felts[4], felts[5], felts[6], felts[7]]);
            let par =
                <AlgebraicBatchBackend<Base, H> as IsMerkleTreeBackend>::hash_new_parent(&l, &r);
            assert_ne!(leaf, par, "{name}: leaf and parent domains must differ");
        }
        for_each_tenant!(check);
    }
}
