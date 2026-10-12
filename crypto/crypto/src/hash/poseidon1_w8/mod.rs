//! Poseidon1 over Goldilocks at width 8 — the grinding permutation of the
//! Poseidon1 base STARK (pil2-proofman's `Poseidon1_8`).
//!
//! ⚠ **Not a shipped hash**: an exploration-branch instance, like
//! [`super::poseidon1_w16`].
//!
//! Same recipe as the width-16 instance: `x^7`, `R_F = 8`, `R_P = 22`, Grain
//! round constants (Plonky3's published width-8 table), and Plonky3's small
//! circulant `MATRIX_CIRC_MDS_8_SML_ROW`. ZisK's own implementation reproduces
//! the known answers ([`super::poseidon1_stark`]'s `PERM8` vectors).

pub mod constants;

use constants::{MDS_CIRC_ROW, ROUND_CONSTANTS};

use super::poseidon1_w16::{Fp, sbox};

/// Lanes in the permutation's state.
pub const STATE_FELTS: usize = 8;
/// Full rounds per half; `R_F = 8`.
pub const HALF_FULL_ROUNDS: usize = 4;
/// Partial rounds, S-box on lane 0 only.
pub const PARTIAL_ROUNDS: usize = 22;
/// All rounds.
pub const NUM_ROUNDS: usize = 2 * HALF_FULL_ROUNDS + PARTIAL_ROUNDS;

/// [`MDS_CIRC_ROW`] expanded at compile time: `MDS_MATRIX[i][j] = row[(j − i) mod 8]`.
const MDS_MATRIX: [[u64; STATE_FELTS]; STATE_FELTS] = {
    let mut m = [[0; STATE_FELTS]; STATE_FELTS];
    let mut i = 0;
    while i < STATE_FELTS {
        let mut j = 0;
        while j < STATE_FELTS {
            m[i][j] = MDS_CIRC_ROW[(j + STATE_FELTS - i) % STATE_FELTS];
            j += 1;
        }
        i += 1;
    }
    m
};

/// The circulant MDS product. The row sums to 43, so each lane's `u128` sum is
/// below `2^70` and `hi·EPSILON < 2^38` needs no reduction of its own.
fn mds(state: &[Fp; STATE_FELTS]) -> [Fp; STATE_FELTS] {
    /// `2^64 ≡ 2^32 − 1 (mod p)`.
    const EPSILON: u64 = 0xFFFF_FFFF;

    let mut out = [Fp::zero(); STATE_FELTS];
    for (row, o) in MDS_MATRIX.iter().zip(out.iter_mut()) {
        let mut acc: u128 = 0;
        for (c, s) in row.iter().zip(state) {
            acc += (*s.value() as u128) * (*c as u128);
        }
        let lo = acc as u64;
        let hi = (acc >> 64) as u64;
        *o = Fp::from(lo) + Fp::from(hi * EPSILON);
    }
    out
}

/// The permutation: 4 full, 22 partial, 4 full rounds (textbook form).
pub fn permute(state: [Fp; STATE_FELTS]) -> [Fp; STATE_FELTS] {
    let mut s = state;
    for (r, rc) in ROUND_CONSTANTS.iter().enumerate() {
        for (v, c) in s.iter_mut().zip(rc) {
            *v += Fp::from(*c);
        }
        if !(HALF_FULL_ROUNDS..HALF_FULL_ROUNDS + PARTIAL_ROUNDS).contains(&r) {
            for v in s.iter_mut() {
                *v = sbox(v);
            }
        } else {
            s[0] = sbox(&s[0]);
        }
        s = mds(&s);
    }
    s
}
