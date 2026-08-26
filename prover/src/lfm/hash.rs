//! The LFM hash interface — the machine's swap surface.
//!
//! The ecosystem hash decision is open (Poseidon2 is broken; candidates are
//! Poseidon-original, RPO/XHash, Monolith and reduced-round Blake2s), so the
//! machine freezes only the *contract*: `Compress` and `Transcript` map two
//! digest cells to one — in different hash domains — `Permute` maps the
//! three-cell state to itself, and the `LFM_HASH`
//! bus tuples and opcode numbers are fixed. Whatever sits behind the trait is
//! the only thing a hash migration replaces.
//!
//! `TestPermutation` below is **NOT cryptographic**. It exists so the machine
//! can be built, executed and proved end-to-end before the hash decision
//! lands; it must never appear outside tests and pre-decision experiments.

use crate::tables::types::FE;

use super::instr::HashMode;
use super::word::LfmWord;

/// Felts in the sponge state (three machine cells).
pub const HASH_STATE_FELTS: usize = 12;
/// Felts in a digest (one machine cell).
pub const HASH_DIGEST_FELTS: usize = 4;

/// The machine's hash contract. `compress` has a default implementation as a
/// single permutation of `[a ‖ b ‖ IV]` truncated to the first cell, which is
/// the construction the chip's `Compress` mode implements; a real hash may
/// override it, but the bus contract (2 cells in, 1 cell out) is frozen.
///
/// ⚠ **`permute` is not total for every candidate.** It is typed over arbitrary
/// Goldilocks elements, but a hasher built on 32-bit words can only accept
/// lane-restricted state, and a hasher may implement one socket and not the
/// other. [`LfmHasher::admits`] is where such a restriction is *declared* and
/// rejected; silently reducing an out-of-range input instead is the bug that
/// would make a host-side assertion pass while the chip proved something else.
pub trait LfmHasher {
    /// The full state permutation (three cells → three cells).
    fn permute(&self, state: [FE; HASH_STATE_FELTS]) -> [FE; HASH_STATE_FELTS];

    /// The capacity cell injected into lanes 8–11 in `Compress` mode.
    fn compress_iv(&self) -> LfmWord;

    /// The capacity cell injected in `Transcript` mode.
    ///
    /// The default is [`LfmHasher::compress_iv`], which is the single-domain
    /// reading and carries the weakening [`LfmHasher::transcript_out`] records.
    /// A hasher that separates its domains through the CAPACITY — the natural
    /// hook for a sponge, and what RPO uses — overrides this instead of
    /// overriding `transcript_out`, because then the chip's `S8` copy
    /// constraint separates the domains too rather than the host doing it alone.
    fn transcript_iv(&self) -> LfmWord {
        self.compress_iv()
    }

    /// The capacity cell injected in `Leaf` mode. Same rule as
    /// [`LfmHasher::transcript_iv`].
    fn leaf_iv(&self) -> LfmWord {
        self.compress_iv()
    }

    /// The capacity cell a row of `mode` takes — the ONE rule the executor, the
    /// trace filler and the chip's constraints all read.
    ///
    /// `Permute` rows have no injected capacity (they carry their own third
    /// input cell), so this is never consulted for them; it answers with the
    /// compress capacity rather than panicking, and the callers gate on the
    /// mode before asking.
    fn mode_iv(&self, mode: HashMode) -> LfmWord {
        match mode {
            HashMode::Transcript => self.transcript_iv(),
            HashMode::Leaf => self.leaf_iv(),
            HashMode::Compress | HashMode::Permute => self.compress_iv(),
        }
    }

    /// The twelve `OUT` felts the chip writes on a `Compress` row.
    ///
    /// The default is the permute-and-truncate construction: all twelve lanes
    /// of `permute(a ‖ b ‖ IV)`, of which the low four are the digest. The
    /// executor records exactly this into the row's `OUT` columns, so a hasher
    /// that overrides [`LfmHasher::compress`] must override this too — or the
    /// trace would describe a permutation its own AIR does not constrain.
    fn compress_out(&self, a: &LfmWord, b: &LfmWord) -> [FE; HASH_STATE_FELTS] {
        self.permute_two_cells(a, b, &self.compress_iv())
    }

    /// The two-cells-plus-capacity permutation every two-cell mode's default is
    /// built from: `permute(a ‖ b ‖ iv)`.
    ///
    /// Factored out because the three modes differ ONLY in which capacity they
    /// inject, and writing that difference three times is how a mode ends up
    /// silently sharing another's domain — which is exactly what the trait's
    /// defaults used to do, `transcript_out` and `leaf_out` both routing through
    /// `compress_out` and picking up the compress IV on the way.
    fn permute_two_cells(&self, a: &LfmWord, b: &LfmWord, iv: &LfmWord) -> [FE; HASH_STATE_FELTS] {
        let mut state: [FE; HASH_STATE_FELTS] = core::array::from_fn(|_| FE::zero());
        state[0..4].clone_from_slice(a);
        state[4..8].clone_from_slice(b);
        state[8..12].clone_from_slice(iv);
        self.permute(state)
    }

    /// Two digest cells → one digest cell.
    fn compress(&self, a: &LfmWord, b: &LfmWord) -> LfmWord {
        let out = self.compress_out(a, b);
        [out[0], out[1], out[2], out[3]]
    }

    /// One Fiat–Shamir transcript step: the same two-cells-in, one-cell-out
    /// shape as [`LfmHasher::compress`], in the TRANSCRIPT hash domain.
    ///
    /// The default is `compress_out` — correct for a hasher with a single
    /// domain, which is what `TestPermutation` and Poseidon are here. A hasher
    /// that *has* domain separation overrides it, and BLAKE3 does: its socket
    /// carries the domain tag in a message word, so a transcript step
    /// and a Merkle parent over the same two cells are different digests.
    ///
    /// ⚠ The default is a real weakening for a hasher that ALSO leaves
    /// [`LfmHasher::transcript_iv`] at its default, and it is deliberate rather
    /// than overlooked: under `Test` and `Poseidon` a transcript step IS a
    /// Merkle parent, so those two hashers separate the domains not at all.
    /// Neither is a production hash — `TestPermutation` is explicitly
    /// non-cryptographic and Poseidon here is measurement-only.
    ///
    /// A production candidate must separate them, by ONE of two mechanisms:
    /// override this function (BLAKE3 does — its domain rides in a message
    /// word), or override `transcript_iv` and let this default carry it (RPO
    /// does — its domain rides in the capacity, which has the advantage that the
    /// chip's `S8` copy constraint separates the domains too).
    fn transcript_out(&self, a: &LfmWord, b: &LfmWord) -> [FE; HASH_STATE_FELTS] {
        self.permute_two_cells(a, b, &self.transcript_iv())
    }

    /// [`LfmHasher::transcript_out`] truncated to the digest cell.
    fn transcript(&self, a: &LfmWord, b: &LfmWord) -> LfmWord {
        let out = self.transcript_out(a, b);
        [out[0], out[1], out[2], out[3]]
    }

    /// A Merkle LEAF: a chaining accumulator and one cell read as four arbitrary
    /// FIELD ELEMENTS.
    ///
    /// **The accumulator is what makes the leaf a chain rather than a tree.** A
    /// wide leaf is an arbitrary-width row pair, so its felts arrive four at a
    /// time; carrying the running digest as this call's first operand absorbs
    /// four felts AND chains in ONE hash, where folding a felts-only leaf digest
    /// into the chain with a separate parent cost two (COMMIT.md §1.2). Leaf
    /// absorption is the dominant term of a recursion tower node, which is why
    /// the shape of this signature is worth the ripple.
    ///
    /// The default is a compress of the accumulator against the felts, which is
    /// the natural reading for a **field-native** hasher: `TestPermutation` and
    /// Poseidon take arbitrary Goldilocks elements directly, so a leaf needs no
    /// encoding from them.
    ///
    /// BLAKE3 overrides it, and the override is the point of the whole mode: its
    /// lanes must be `u32`, so each felt becomes a checked `lo`/`hi` pair inside
    /// the socket, under the `"LFML"` tag.
    ///
    /// ⚠ Same weakening as [`LfmHasher::transcript_out`], with the same two
    /// escapes: a hasher that overrides neither this nor
    /// [`LfmHasher::leaf_iv`] does not separate a leaf from a parent, so under
    /// `Test` and `Poseidon` the O5 second-preimage split is carried by fixed
    /// tree depth alone. BLAKE3 escapes by overriding this; RPO escapes by
    /// overriding `leaf_iv`.
    fn leaf_out(&self, acc: &LfmWord, felts: &LfmWord) -> [FE; HASH_STATE_FELTS] {
        self.permute_two_cells(acc, felts, &self.leaf_iv())
    }

    /// [`LfmHasher::leaf_out`] truncated to the digest cell.
    fn leaf(&self, acc: &LfmWord, felts: &LfmWord) -> LfmWord {
        let out = self.leaf_out(acc, felts);
        [out[0], out[1], out[2], out[3]]
    }

    /// Rejects a hash instruction this hasher's chip cannot prove, naming why.
    ///
    /// Total for every candidate whose domain is the whole state under both
    /// modes, which is why the default is `Ok`. It exists for the ones whose
    /// domain is smaller: `HasherKind::Blake3` uses it for both of its
    /// restrictions — it has no `permute` socket at all, and its `compress`
    /// lanes must be `u32`. Returning an error here is what turns "the AIR
    /// would reject this" into "the executor says so, with a reason".
    fn admits(&self, mode: HashMode, state: &[FE; HASH_STATE_FELTS]) -> Result<(), &'static str> {
        let _ = (mode, state);
        Ok(())
    }
}

/// A placeholder permutation: one round of `x ↦ (x + rc)³` followed by the
/// mixing matrix `M = I + J` (identity plus all-ones; eigenvalues 13 and 1,
/// so invertible over Goldilocks). Degree 3, one trace row per invocation.
///
/// **NOT CRYPTOGRAPHIC — wiring placeholder only.** No diffusion analysis, no
/// round count, nothing: it is a stand-in with the right shape and degree
/// while the ecosystem hash decision is open.
pub struct TestPermutation;

impl TestPermutation {
    /// Fixed round "constants" — an odd multiplier walk; arbitrary, public.
    pub fn round_constant(i: usize) -> FE {
        FE::from(0x9E37_79B9_7F4A_7C15u64.wrapping_mul(i as u64 + 1))
    }

    /// The compress-mode capacity constants as raw u64s (the chip bakes them
    /// into its constraints via `const_base`; `FE::from` reduces identically).
    pub fn compress_iv_raw() -> [u64; 4] {
        core::array::from_fn(|i| 0xC0DE_0000_0000_0001u64.wrapping_add(i as u64))
    }
}

impl LfmHasher for TestPermutation {
    fn permute(&self, state: [FE; HASH_STATE_FELTS]) -> [FE; HASH_STATE_FELTS] {
        // t_i = (s_i + rc_i)^3 ; out_j = t_j + Σ_i t_i   (M = I + J)
        let t: Vec<FE> = state
            .iter()
            .enumerate()
            .map(|(i, s)| {
                let x = s + Self::round_constant(i);
                &x * &x * x
            })
            .collect();
        let sum: FE = t.iter().fold(FE::zero(), |acc, x| acc + x);
        core::array::from_fn(|j| &t[j] + &sum)
    }

    fn compress_iv(&self) -> LfmWord {
        Self::compress_iv_raw().map(FE::from)
    }
}

/// Which permutation the `LFM_HASH` chip proves — a **construction-time**
/// choice, fixed before any trace exists.
///
/// The chips bake their hasher's round constants into their constraints, so
/// execution, trace generation and the AIR set must all agree (`proof.rs`
/// enforces that by construction: one kind reaches all three). This enum is
/// what carries the agreement, and it is threaded rather than global so a
/// single process can prove under both.
///
/// ⚠ **`Test` is the default and the machine's real hash is UNDECIDED.** The
/// default exists so every pre-decision call site keeps proving what it always
/// proved; it is not a statement that `TestPermutation` is the machine's hash.
/// The ecosystem hash decision is what the candidate columns feed.
///
/// The discriminants are written out and `#[repr(u8)]` because [`as_tag`] feeds
/// `lfm_program_id`'s preimage: the wire value must never follow declaration
/// order, or inserting a variant would silently move every program digest.
///
/// [`as_tag`]: HasherKind::as_tag
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
#[repr(u8)]
pub enum HasherKind {
    /// [`TestPermutation`] — NOT cryptographic. One degree-3 round.
    #[default]
    Test = 0,
    /// [`super::poseidon::PoseidonGoldilocks`] — Poseidon-original, width 12,
    /// `x^7`, 8 full + 22 partial rounds.
    Poseidon = 1,
    /// [`super::blake3_socket::Blake3Permutation`] — BLAKE3 behind the Option-A
    /// 2-to-1 compress socket, `compress` only.
    ///
    /// The one candidate here that is a real, standard, externally anchored
    /// hash: at the default `SOCKET_ROUNDS = 7` a compress is literally
    /// `blake3::hash(a ‖ b ‖ "LFMC")` truncated to 128 bits. It is also the one
    /// with a restricted domain — no `permute` socket, and `u32` lanes — which
    /// [`LfmHasher::admits`] enforces.
    Blake3 = 2,
    /// [`super::rpo::Rpo256`] — Rescue-Prime Optimized, width 12, rate 8,
    /// 7 rounds of `x^7` / `x^{1/7}`.
    ///
    /// The first candidate here that is BOTH field-native (so it needs no
    /// felt→`u32` encoding, and its digest cell is a full 4-felt ~128-bit
    /// digest rather than the socket's documented 64-bit one) and
    /// domain-separated (through the capacity, see [`LfmHasher::mode_iv`]).
    Rpo = 3,
}

impl HasherKind {
    /// The stable one-byte tag bound into `lfm_program_id`.
    ///
    /// A new candidate takes the next unused value and never reuses a retired
    /// one: a tag collision would give two different permutations one program
    /// identity, which is the whole thing this binding exists to prevent.
    pub const fn as_tag(self) -> u8 {
        self as u8
    }
}

impl LfmHasher for HasherKind {
    fn permute(&self, state: [FE; HASH_STATE_FELTS]) -> [FE; HASH_STATE_FELTS] {
        match self {
            HasherKind::Test => TestPermutation.permute(state),
            HasherKind::Poseidon => super::poseidon::PoseidonGoldilocks.permute(state),
            HasherKind::Blake3 => super::blake3_socket::Blake3Permutation.permute(state),
            HasherKind::Rpo => super::rpo::Rpo256.permute(state),
        }
    }

    fn compress_iv(&self) -> LfmWord {
        match self {
            HasherKind::Test => TestPermutation.compress_iv(),
            HasherKind::Poseidon => super::poseidon::PoseidonGoldilocks.compress_iv(),
            HasherKind::Blake3 => super::blake3_socket::Blake3Permutation.compress_iv(),
            HasherKind::Rpo => super::rpo::Rpo256.compress_iv(),
        }
    }

    /// Delegated explicitly, for the same reason `compress_out` is: RPO is the
    /// one candidate whose transcript CAPACITY differs from its compress one,
    /// and a dispatch that fell through to this enum's own `compress_iv` would
    /// hand the executor and the trace filler a capacity the chip does not
    /// constrain — a host/chip disagreement, not a wrong answer the chip
    /// catches.
    fn transcript_iv(&self) -> LfmWord {
        match self {
            HasherKind::Test => TestPermutation.transcript_iv(),
            HasherKind::Poseidon => super::poseidon::PoseidonGoldilocks.transcript_iv(),
            HasherKind::Blake3 => super::blake3_socket::Blake3Permutation.transcript_iv(),
            HasherKind::Rpo => super::rpo::Rpo256.transcript_iv(),
        }
    }

    /// Delegated explicitly, same reason as [`HasherKind::transcript_iv`].
    fn leaf_iv(&self) -> LfmWord {
        match self {
            HasherKind::Test => TestPermutation.leaf_iv(),
            HasherKind::Poseidon => super::poseidon::PoseidonGoldilocks.leaf_iv(),
            HasherKind::Blake3 => super::blake3_socket::Blake3Permutation.leaf_iv(),
            HasherKind::Rpo => super::rpo::Rpo256.leaf_iv(),
        }
    }

    /// Delegated explicitly rather than left to the trait default: a candidate
    /// that overrides `compress` must be honoured through this dispatch too.
    fn compress(&self, a: &LfmWord, b: &LfmWord) -> LfmWord {
        match self {
            HasherKind::Test => TestPermutation.compress(a, b),
            HasherKind::Poseidon => super::poseidon::PoseidonGoldilocks.compress(a, b),
            HasherKind::Blake3 => super::blake3_socket::Blake3Permutation.compress(a, b),
            HasherKind::Rpo => super::rpo::Rpo256.compress(a, b),
        }
    }

    /// Delegated explicitly, for the same reason `compress` is: BLAKE3
    /// overrides it, and a default that quietly permuted instead would write
    /// twelve felts its own AIR pins to four.
    fn compress_out(&self, a: &LfmWord, b: &LfmWord) -> [FE; HASH_STATE_FELTS] {
        match self {
            HasherKind::Test => TestPermutation.compress_out(a, b),
            HasherKind::Poseidon => super::poseidon::PoseidonGoldilocks.compress_out(a, b),
            HasherKind::Blake3 => super::blake3_socket::Blake3Permutation.compress_out(a, b),
            HasherKind::Rpo => super::rpo::Rpo256.compress_out(a, b),
        }
    }

    /// Delegated explicitly, third time for the same reason: BLAKE3 is the one
    /// candidate whose transcript domain differs from its compress domain, and
    /// a dispatch that fell through to the trait default would hash a
    /// transcript step under the MERKLE tag while its AIR proved the transcript
    /// one — a host/chip disagreement, not a wrong answer the chip catches.
    fn transcript_out(&self, a: &LfmWord, b: &LfmWord) -> [FE; HASH_STATE_FELTS] {
        match self {
            HasherKind::Test => TestPermutation.transcript_out(a, b),
            HasherKind::Poseidon => super::poseidon::PoseidonGoldilocks.transcript_out(a, b),
            HasherKind::Blake3 => super::blake3_socket::Blake3Permutation.transcript_out(a, b),
            HasherKind::Rpo => super::rpo::Rpo256.transcript_out(a, b),
        }
    }

    fn transcript(&self, a: &LfmWord, b: &LfmWord) -> LfmWord {
        match self {
            HasherKind::Test => TestPermutation.transcript(a, b),
            HasherKind::Poseidon => super::poseidon::PoseidonGoldilocks.transcript(a, b),
            HasherKind::Blake3 => super::blake3_socket::Blake3Permutation.transcript(a, b),
            HasherKind::Rpo => super::rpo::Rpo256.transcript(a, b),
        }
    }

    /// Delegated explicitly, fourth time for the same reason: BLAKE3's leaf mode
    /// is an ENCODING, not just a tag, so a dispatch that fell through to the
    /// trait default would hash four felts as a digest cell — a host answer no
    /// chip proves.
    fn leaf_out(&self, acc: &LfmWord, felts: &LfmWord) -> [FE; HASH_STATE_FELTS] {
        match self {
            HasherKind::Test => TestPermutation.leaf_out(acc, felts),
            HasherKind::Poseidon => super::poseidon::PoseidonGoldilocks.leaf_out(acc, felts),
            HasherKind::Blake3 => super::blake3_socket::Blake3Permutation.leaf_out(acc, felts),
            HasherKind::Rpo => super::rpo::Rpo256.leaf_out(acc, felts),
        }
    }

    fn leaf(&self, acc: &LfmWord, felts: &LfmWord) -> LfmWord {
        match self {
            HasherKind::Test => TestPermutation.leaf(acc, felts),
            HasherKind::Poseidon => super::poseidon::PoseidonGoldilocks.leaf(acc, felts),
            HasherKind::Blake3 => super::blake3_socket::Blake3Permutation.leaf(acc, felts),
            HasherKind::Rpo => super::rpo::Rpo256.leaf(acc, felts),
        }
    }

    fn admits(&self, mode: HashMode, state: &[FE; HASH_STATE_FELTS]) -> Result<(), &'static str> {
        match self {
            HasherKind::Test => TestPermutation.admits(mode, state),
            HasherKind::Poseidon => super::poseidon::PoseidonGoldilocks.admits(mode, state),
            HasherKind::Blake3 => super::blake3_socket::Blake3Permutation.admits(mode, state),
            HasherKind::Rpo => super::rpo::Rpo256.admits(mode, state),
        }
    }
}
