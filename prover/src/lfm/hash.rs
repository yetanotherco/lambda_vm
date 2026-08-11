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

    /// The twelve `OUT` felts the chip writes on a `Compress` row.
    ///
    /// The default is the permute-and-truncate construction: all twelve lanes
    /// of `permute(a ‖ b ‖ IV)`, of which the low four are the digest. The
    /// executor records exactly this into the row's `OUT` columns, so a hasher
    /// that overrides [`LfmHasher::compress`] must override this too — or the
    /// trace would describe a permutation its own AIR does not constrain.
    fn compress_out(&self, a: &LfmWord, b: &LfmWord) -> [FE; HASH_STATE_FELTS] {
        let iv = self.compress_iv();
        let mut state: [FE; HASH_STATE_FELTS] = core::array::from_fn(|_| FE::zero());
        state[0..4].clone_from_slice(a);
        state[4..8].clone_from_slice(b);
        state[8..12].clone_from_slice(&iv);
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
    /// carries the domain tag in the message word `m[8]`, so a transcript step
    /// and a Merkle parent over the same two cells are different digests.
    ///
    /// ⚠ The default is a real weakening for a single-domain hasher, and it is
    /// deliberate rather than overlooked: under `Test` and `Poseidon` a
    /// transcript step IS a Merkle parent, so those two hashers separate the
    /// domains not at all. Neither is a production hash — `TestPermutation` is
    /// explicitly non-cryptographic and Poseidon here is measurement-only — and
    /// the machine's real hash is the one that separates them. A future
    /// production candidate that reaches this default without overriding it is
    /// shipping a transcript with no domain separation.
    fn transcript_out(&self, a: &LfmWord, b: &LfmWord) -> [FE; HASH_STATE_FELTS] {
        self.compress_out(a, b)
    }

    /// [`LfmHasher::transcript_out`] truncated to the digest cell.
    fn transcript(&self, a: &LfmWord, b: &LfmWord) -> LfmWord {
        let out = self.transcript_out(a, b);
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
        }
    }

    fn compress_iv(&self) -> LfmWord {
        match self {
            HasherKind::Test => TestPermutation.compress_iv(),
            HasherKind::Poseidon => super::poseidon::PoseidonGoldilocks.compress_iv(),
            HasherKind::Blake3 => super::blake3_socket::Blake3Permutation.compress_iv(),
        }
    }

    /// Delegated explicitly rather than left to the trait default: a candidate
    /// that overrides `compress` must be honoured through this dispatch too.
    fn compress(&self, a: &LfmWord, b: &LfmWord) -> LfmWord {
        match self {
            HasherKind::Test => TestPermutation.compress(a, b),
            HasherKind::Poseidon => super::poseidon::PoseidonGoldilocks.compress(a, b),
            HasherKind::Blake3 => super::blake3_socket::Blake3Permutation.compress(a, b),
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
        }
    }

    fn transcript(&self, a: &LfmWord, b: &LfmWord) -> LfmWord {
        match self {
            HasherKind::Test => TestPermutation.transcript(a, b),
            HasherKind::Poseidon => super::poseidon::PoseidonGoldilocks.transcript(a, b),
            HasherKind::Blake3 => super::blake3_socket::Blake3Permutation.transcript(a, b),
        }
    }

    fn admits(&self, mode: HashMode, state: &[FE; HASH_STATE_FELTS]) -> Result<(), &'static str> {
        match self {
            HasherKind::Test => TestPermutation.admits(mode, state),
            HasherKind::Poseidon => super::poseidon::PoseidonGoldilocks.admits(mode, state),
            HasherKind::Blake3 => super::blake3_socket::Blake3Permutation.admits(mode, state),
        }
    }
}
