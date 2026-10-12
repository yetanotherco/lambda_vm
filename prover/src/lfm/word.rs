//! The LFM machine word: `[F; 4]`, four Goldilocks elements.
//!
//! The word is digest-aligned, not extension-aligned: a Goldilocks-native hash
//! at a 128-bit target uses a 4-felt digest and a 12-felt state, so a digest is
//! exactly one cell, the sponge rate two cells and the state three cells. Base
//! values occupy lane 0 with lanes 1–3 zero; extension values (Fp3) occupy
//! lanes 0–2 with lane 3 zero. The zero lanes are enforced on the bus as
//! constant tuple entries, never as trace columns, so a base value cannot
//! smuggle a phantom extension element.

use crate::tables::types::{FE, FEE, GoldilocksField};
use math::field::traits::IsPrimeField;

/// One machine word / memory cell: four Goldilocks elements.
pub type LfmWord = [FE; 4];

/// Number of felt lanes in a word.
pub const WORD_LANES: usize = 4;

/// A base field value embedded as a word: `(v, 0, 0, 0)`.
pub fn base_word(v: FE) -> LfmWord {
    [v, FE::zero(), FE::zero(), FE::zero()]
}

/// An Fp3 extension value embedded as a word: `(a0, a1, a2, 0)`.
pub fn ext_word(e: &FEE) -> LfmWord {
    let [a0, a1, a2] = *e.value();
    [a0, a1, a2, FE::zero()]
}

/// Reads a word as a base value. `None` unless lanes 1–3 are zero — mirrors
/// the bus-level rule that a base receive carries constant zero high lanes.
pub fn word_as_base(w: &LfmWord) -> Option<FE> {
    (w[1] == FE::zero() && w[2] == FE::zero() && w[3] == FE::zero()).then(|| w[0])
}

/// Reads a word as an Fp3 value. `None` unless lane 3 is zero.
pub fn word_as_ext(w: &LfmWord) -> Option<FEE> {
    (w[3] == FE::zero()).then(|| FEE::new([w[0], w[1], w[2]]))
}

/// Packs a digest word into the 32-byte commitment format: four canonical
/// u64 lanes, little-endian, in lane order. Exact: 4 × 8 bytes.
pub fn pack_digest(w: &LfmWord) -> [u8; 32] {
    let mut out = [0u8; 32];
    for (lane, chunk) in w.iter().zip(out.chunks_exact_mut(8)) {
        chunk.copy_from_slice(&GoldilocksField::canonical(lane.value()).to_le_bytes());
    }
    out
}

/// Inverse of [`pack_digest`]. Lanes are reduced mod p on the way in.
pub fn unpack_digest(bytes: &[u8; 32]) -> LfmWord {
    let mut lanes = [FE::zero(), FE::zero(), FE::zero(), FE::zero()];
    for (lane, chunk) in lanes.iter_mut().zip(bytes.chunks_exact(8)) {
        let mut raw = [0u8; 8];
        raw.copy_from_slice(chunk);
        *lane = FE::from(u64::from_le_bytes(raw));
    }
    lanes
}
