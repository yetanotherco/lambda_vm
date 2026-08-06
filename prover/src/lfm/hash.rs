//! The LFM hash interface — the machine's swap surface.
//!
//! The ecosystem hash decision is open (Poseidon2 is broken; candidates are
//! Poseidon-original, RPO/XHash, Monolith and reduced-round Blake2s), so the
//! machine freezes only the *contract*: `Compress` maps two digest cells to
//! one, `Permute` maps the three-cell state to itself, and the `LFM_HASH`
//! bus tuples and opcode numbers are fixed. Whatever sits behind the trait is
//! the only thing a hash migration replaces.
//!
//! `TestPermutation` below is **NOT cryptographic**. It exists so the machine
//! can be built, executed and proved end-to-end before the hash decision
//! lands; it must never appear outside tests and pre-decision experiments.

use crate::tables::types::FE;

use super::word::LfmWord;

/// Felts in the sponge state (three machine cells).
pub const HASH_STATE_FELTS: usize = 12;
/// Felts in a digest (one machine cell).
pub const HASH_DIGEST_FELTS: usize = 4;

/// The machine's hash contract. `compress` has a default implementation as a
/// single permutation of `[a ‖ b ‖ IV]` truncated to the first cell, which is
/// the construction the chip's `Compress` mode implements; a real hash may
/// override it, but the bus contract (2 cells in, 1 cell out) is frozen.
pub trait LfmHasher {
    /// The full state permutation (three cells → three cells).
    fn permute(&self, state: [FE; HASH_STATE_FELTS]) -> [FE; HASH_STATE_FELTS];

    /// The capacity cell injected into lanes 8–11 in `Compress` mode.
    fn compress_iv(&self) -> LfmWord;

    /// Two digest cells → one digest cell.
    fn compress(&self, a: &LfmWord, b: &LfmWord) -> LfmWord {
        let iv = self.compress_iv();
        let mut state: [FE; HASH_STATE_FELTS] = core::array::from_fn(|_| FE::zero());
        state[0..4].clone_from_slice(a);
        state[4..8].clone_from_slice(b);
        state[8..12].clone_from_slice(&iv);
        let out = self.permute(state);
        [out[0], out[1], out[2], out[3]]
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
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum HasherKind {
    /// [`TestPermutation`] — NOT cryptographic. One degree-3 round.
    #[default]
    Test,
    /// [`super::poseidon::PoseidonGoldilocks`] — Poseidon-original, width 12,
    /// `x^7`, 8 full + 22 partial rounds.
    Poseidon,
}

impl LfmHasher for HasherKind {
    fn permute(&self, state: [FE; HASH_STATE_FELTS]) -> [FE; HASH_STATE_FELTS] {
        match self {
            HasherKind::Test => TestPermutation.permute(state),
            HasherKind::Poseidon => super::poseidon::PoseidonGoldilocks.permute(state),
        }
    }

    fn compress_iv(&self) -> LfmWord {
        match self {
            HasherKind::Test => TestPermutation.compress_iv(),
            HasherKind::Poseidon => super::poseidon::PoseidonGoldilocks.compress_iv(),
        }
    }

    /// Delegated explicitly rather than left to the trait default: a candidate
    /// that overrides `compress` must be honoured through this dispatch too.
    fn compress(&self, a: &LfmWord, b: &LfmWord) -> LfmWord {
        match self {
            HasherKind::Test => TestPermutation.compress(a, b),
            HasherKind::Poseidon => super::poseidon::PoseidonGoldilocks.compress(a, b),
        }
    }
}
