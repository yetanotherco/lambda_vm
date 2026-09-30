//! Poseidon1 over Goldilocks at width 16 — the MEASUREMENT instance for the
//! D-HASH track (rate 12, capacity 4, 4-ary Merkle nodes).
//!
//! ⚠ **Not a shipped hash.** Nothing in the prover commits with it. It exists
//! so the base-hash candidate can be measured (chip census, device kernels)
//! against RPX; choosing to ship it is a separate decision.
//!
//! # The instance, and where each number comes from
//!
//! Plonky3 publishes Goldilocks Poseidon1 constants and known answers only at
//! widths 8 and 12; at width 16 it has an MDS matrix but no round constants.
//! So the width-16 instance is DERIVED here by the Poseidon paper's procedure
//! (eprint 2019/458), and every step is checked by a second implementation:
//!
//! - **S-box** `x^7`: the smallest exponent coprime to `p − 1` for Goldilocks.
//! - **Rounds** `R_F = 8`, `R_P = 22`: the paper's inequalities at `t = 16`,
//!   `α = 7`, `M = 128` (statistical, interpolation, three Gröbner bounds, plus
//!   eprint 2023/537's binomial bound) give the minimum `R_F = 6`, `R_P = 20`;
//!   the paper's margin (`R_F + 2`, `⌈1.075·R_P⌉`) gives 8 and 22. The binding
//!   terms are the statistical bound (`R_F ≥ 6`) and interpolation
//!   (`R_F + R_P ≥ 1 + ⌈64·log_7 2⌉ + ⌈log_7 16⌉ = 26`). Same counts as Plonky3's
//!   widths 8 and 12. `scripts/poseidon1/p1_params.py rounds --width 16` prints
//!   every inequality.
//! - **Round constants**: the paper's Grain LFSR (Appendix E) with
//!   `field_type=1, sbox=0, n=64, t=16, R_F=8, R_P=22`. Our generator
//!   reproduces all 240/360 of Plonky3's width-8/12 constants and their known
//!   answers first; Plonky3's own generator (its width whitelist lifted)
//!   produces the same 480 width-16 constants.
//! - **MDS**: Plonky3's circulant `MATRIX_CIRC_MDS_16_SML_ROW` (small entries,
//!   so the product is shifts and adds). Checked MDS over Goldilocks
//!   exhaustively (all 3.0e8 square submatrices up to the circulant shift,
//!   `scripts/poseidon1/mds_check.rs`), and passes Algorithms 1–3 of eprint
//!   2020/500 (no invariant subspace trails) in Plonky3's implementation.
//!   ⚠ It is NOT the paper's Grain-derived Cauchy matrix; see the lane note
//!   for why that choice is open (eprint 2026/1760 attacks an MDS chosen after
//!   the constants).
//!
//! # Lane conventions (the measurement's, not a frozen format)
//!
//! Rate `0..12`, capacity `12..16`, digest `0..4` — RPX's rule at width 16.
//! A leaf is the overwrite duplex with the padding flag `len mod 12` in
//! capacity lane 0 and the domain [`DOMAIN_LEAF`] in lane 1. A 4-ary node
//! permutes the four child digests (all 16 lanes) and truncates to lanes
//! `0..4`: there is no capacity and hence no domain lane (D-HASH §5).

pub mod constants;
#[cfg(test)]
mod kat;
#[cfg(test)]
mod tests;

use math::field::element::FieldElement;
use math::field::goldilocks::GoldilocksField;

use constants::{MDS_CIRC_ROW, ROUND_CONSTANTS};

/// A Goldilocks field element.
pub type Fp = FieldElement<GoldilocksField>;

/// Lanes in the permutation's state.
pub const STATE_FELTS: usize = 16;
/// Lanes a block of input overwrites — the sponge's rate.
pub const RATE_FELTS: usize = 12;
/// Felts in a digest.
pub const DIGEST_FELTS: usize = 4;
/// Children per Merkle node.
pub const ARITY: usize = STATE_FELTS / DIGEST_FELTS;

/// The S-box exponent.
pub const ALPHA: u64 = 7;
/// Full rounds per half; `R_F = 8`.
pub const HALF_FULL_ROUNDS: usize = 4;
/// Partial rounds, S-box on lane 0 only.
pub const PARTIAL_ROUNDS: usize = 22;
/// All rounds.
pub const NUM_ROUNDS: usize = 2 * HALF_FULL_ROUNDS + PARTIAL_ROUNDS;

/// Capacity lane carrying the padding flag.
pub const CAPACITY_PAD_LANE: usize = 0;
/// Capacity lane carrying the domain tag.
pub const CAPACITY_DOMAIN_LANE: usize = 1;
/// The leaf domain, `"P1WL"` as a little-endian `u32` — distinct from RPX's.
pub const DOMAIN_LEAF: u64 = u32::from_le_bytes(*b"P1WL") as u64;

/// A four-felt digest.
pub type Digest = [Fp; DIGEST_FELTS];

/// Does round `r` S-box every lane?
pub const fn is_full_round(r: usize) -> bool {
    r < HALF_FULL_ROUNDS || r >= HALF_FULL_ROUNDS + PARTIAL_ROUNDS
}

/// `x^7` as `x²`, `x³ = x²·x`, `x^7 = (x³)²·x` — the AIR's association.
pub fn sbox(x: &Fp) -> Fp {
    let x2 = x * x;
    let x3 = &x2 * x;
    let x6 = &x3 * &x3;
    &x6 * x
}

/// [`MDS_CIRC_ROW`] expanded at compile time: `MDS_MATRIX[i][j] = row[(j − i) mod 16]`.
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

/// The circulant MDS product, one `u128` accumulation and one reduction per
/// lane. The row sums to 371, so each lane's sum is below `371·2^64 < 2^73`
/// and `hi < 2^9`; `hi·EPSILON < 2^41` needs no reduction of its own.
pub fn mds(state: &[Fp; STATE_FELTS]) -> [Fp; STATE_FELTS] {
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

/// One round: add constants, S-box (every lane in a full round, lane 0 in a
/// partial one), then the MDS.
pub fn round(state: &[Fp; STATE_FELTS], r: usize) -> [Fp; STATE_FELTS] {
    round_with(&ROUND_CONSTANTS, state, r)
}

fn round_with(
    rc: &[[u64; STATE_FELTS]; NUM_ROUNDS],
    state: &[Fp; STATE_FELTS],
    r: usize,
) -> [Fp; STATE_FELTS] {
    let mut s = *state;
    for (v, c) in s.iter_mut().zip(rc[r].iter()) {
        *v += Fp::from(*c);
    }
    if is_full_round(r) {
        for v in s.iter_mut() {
            *v = sbox(v);
        }
    } else {
        s[0] = sbox(&s[0]);
    }
    mds(&s)
}

/// ★ The permutation: 4 full, 22 partial, 4 full rounds.
///
/// The textbook form (a dense round each time), not Plonky3's sparse
/// partial-round decomposition — the known answers are reproduced by both.
pub fn permute(state: [Fp; STATE_FELTS]) -> [Fp; STATE_FELTS] {
    permute_with(&ROUND_CONSTANTS, state)
}

/// [`permute`] over a given constant table — the tests' mutation seam.
fn permute_with(
    rc: &[[u64; STATE_FELTS]; NUM_ROUNDS],
    state: [Fp; STATE_FELTS],
) -> [Fp; STATE_FELTS] {
    let mut s = state;
    for r in 0..NUM_ROUNDS {
        s = round_with(rc, &s, r);
    }
    s
}

/// The rate-12 overwrite duplex over a felt stream (a Merkle leaf).
pub fn sponge_leaf(felts: &[Fp]) -> Digest {
    let mut state = [Fp::zero(); STATE_FELTS];
    state[RATE_FELTS + CAPACITY_PAD_LANE] = Fp::from((felts.len() % RATE_FELTS) as u64);
    state[RATE_FELTS + CAPACITY_DOMAIN_LANE] = Fp::from(DOMAIN_LEAF);
    for block in felts.chunks(RATE_FELTS) {
        for (lane, slot) in state.iter_mut().take(RATE_FELTS).enumerate() {
            *slot = block.get(lane).copied().unwrap_or_else(Fp::zero);
        }
        state = permute(state);
    }
    [state[0], state[1], state[2], state[3]]
}

/// A 4-ary Merkle node: permute the four children, truncate to four felts.
pub fn compress4(children: &[Digest; ARITY]) -> Digest {
    let mut state = [Fp::zero(); STATE_FELTS];
    for (c, child) in children.iter().enumerate() {
        state[c * DIGEST_FELTS..(c + 1) * DIGEST_FELTS].copy_from_slice(child);
    }
    let out = permute(state);
    [out[0], out[1], out[2], out[3]]
}
