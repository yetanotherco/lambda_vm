//! Poseidon1 over Goldilocks at width 16 — the MEASUREMENT instance for the
//! D-HASH track (rate 12, capacity 4, 4-ary Merkle nodes).
//!
//! ⚠ **Not a shipped hash.** Nothing in the prover commits with it. It exists
//! so the base-hash candidate can be measured (chip census, device kernels)
//! against RPX; choosing to ship it is a separate decision.
//!
//! ZisK runs this same instance (pil2-proofman's `Poseidon1_16`): its own
//! implementation reproduces the known answers here, and [`super::poseidon1_stark`]
//! builds ZisK's leaf, tree, transcript and grinding on it.
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
//!   the constants). The Cauchy alternative is here too ([`permute_cauchy`],
//!   same rounds and constants), so both options can be measured.
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

use constants::{CAUCHY_MDS, MDS_CIRC_ROW, ROUND_CONSTANTS};

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

/// The dense product by the Grain Cauchy matrix [`CAUCHY_MDS`] — the MDS
/// alternative (full-field entries, so one field multiplication per entry).
pub fn mds_cauchy(state: &[Fp; STATE_FELTS]) -> [Fp; STATE_FELTS] {
    let mut out = [Fp::zero(); STATE_FELTS];
    for (row, o) in CAUCHY_MDS.iter().zip(out.iter_mut()) {
        for (c, s) in row.iter().zip(state) {
            *o += Fp::from(*c) * s;
        }
    }
    out
}

/// A linear layer: [`mds`] or [`mds_cauchy`].
type LinearLayer = fn(&[Fp; STATE_FELTS]) -> [Fp; STATE_FELTS];

/// One round: add constants, S-box (every lane in a full round, lane 0 in a
/// partial one), then the MDS.
pub fn round(state: &[Fp; STATE_FELTS], r: usize) -> [Fp; STATE_FELTS] {
    round_with(&ROUND_CONSTANTS, mds, state, r)
}

fn round_with(
    rc: &[[u64; STATE_FELTS]; NUM_ROUNDS],
    linear: LinearLayer,
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
    linear(&s)
}

/// ★ The permutation: 4 full, 22 partial, 4 full rounds.
///
/// With `std`, the partial rounds run in the sparse form ([`fast`]): the same
/// permutation by fewer multiplications, pinned to [`permute_textbook`] by
/// the known answers and by random-input equality. Without `std` (a guest),
/// the textbook form.
pub fn permute(state: [Fp; STATE_FELTS]) -> [Fp; STATE_FELTS] {
    #[cfg(feature = "std")]
    return fast::permute(fast::tables(), state);
    #[cfg(not(feature = "std"))]
    return permute_textbook(state);
}

/// The textbook form: every round dense (constants, S-box, the full MDS), the
/// definition the sparse form is checked against.
pub fn permute_textbook(state: [Fp; STATE_FELTS]) -> [Fp; STATE_FELTS] {
    permute_with(&ROUND_CONSTANTS, mds, state)
}

/// The partial rounds' sparse form (the Poseidon paper's Appendix B), derived
/// once from [`ROUND_CONSTANTS`] and the MDS — nothing is a new constant.
///
/// Two rewrites, each an identity on the permutation:
///
/// 1. **Constants.** A partial round adds `c` to every lane, S-boxes lane 0,
///    then multiplies by `M`. The lanes 1.. of `c` pass the S-box untouched, so
///    `M·(S(s + c₀e₀) + c_tail) = M·S(s + c₀e₀) + M·c_tail`: the round keeps
///    only its lane-0 constant and hands `M·c_tail` to the next round's. The
///    last partial round hands its carry to the first full round after.
/// 2. **The linear layers.** Write `N = M·diag(1, A_prev)` as
///    `diag(1, A)·B` with `A = N̂` (the lower-right 15 × 15 block) and `B`
///    sparse: `[[n₀₀, n_w], [A⁻¹·n_v, I]]`. A lane-0-only round commutes with
///    `diag(1, A)`, so that factor moves into the next round's `N`; each round
///    applies only its sparse `B` (≈ 2t multiplications instead of t²), and the
///    last round's `diag(1, A)` is applied once. `A` is invertible: it is a
///    product of square submatrices of an MDS matrix.
#[cfg(feature = "std")]
pub(crate) mod fast {
    use super::{
        Fp, HALF_FULL_ROUNDS, MDS_MATRIX, NUM_ROUNDS, PARTIAL_ROUNDS, ROUND_CONSTANTS, STATE_FELTS,
        mds, sbox,
    };

    const T1: usize = STATE_FELTS - 1;

    /// One partial round's sparse layer `B = [[m00, w], [v, I]]`.
    #[derive(Clone, Debug)]
    pub(crate) struct Sparse {
        pub(crate) m00: u64,
        pub(crate) w: [u64; T1],
        pub(crate) v: [u64; T1],
    }

    /// Everything the sparse form reads.
    #[derive(Clone, Debug)]
    pub(crate) struct Tables {
        /// Each partial round's lane-0 constant.
        pub(crate) lane0: [u64; PARTIAL_ROUNDS],
        pub(crate) sparse: Vec<Sparse>,
        /// The last partial round's `diag(1, A)`, lanes 1..
        pub(crate) tail: [[u64; T1]; T1],
        /// The full rounds' constants in round order, the first round after
        /// the partial ones carrying the partial rounds' constants.
        pub(crate) full: [[Fp; STATE_FELTS]; 2 * HALF_FULL_ROUNDS],
    }

    static TABLES: std::sync::OnceLock<Tables> = std::sync::OnceLock::new();

    /// The tables, derived on first use.
    pub(crate) fn tables() -> &'static Tables {
        TABLES.get_or_init(derive)
    }

    fn m(i: usize, j: usize) -> Fp {
        Fp::from(MDS_MATRIX[i][j])
    }

    /// `A·x = b` over the field by Gauss–Jordan elimination; `A` invertible.
    fn solve(a: &[[Fp; T1]; T1], b: &[Fp; T1]) -> [Fp; T1] {
        let mut aug: Vec<Vec<Fp>> = (0..T1)
            .map(|i| {
                let mut row = a[i].to_vec();
                row.push(b[i]);
                row
            })
            .collect();
        for col in 0..T1 {
            let pivot = (col..T1)
                .find(|&r| aug[r][col] != Fp::zero())
                .expect("a square submatrix of an MDS matrix is invertible");
            aug.swap(col, pivot);
            let inv = aug[col][col].inv().expect("a nonzero pivot");
            for x in aug[col].iter_mut() {
                *x *= inv;
            }
            let pivot_row = aug[col].clone();
            for (r, row) in aug.iter_mut().enumerate() {
                if r != col && row[col] != Fp::zero() {
                    let f = row[col];
                    for (x, p) in row.iter_mut().zip(&pivot_row).skip(col) {
                        *x = *x - f * p;
                    }
                }
            }
        }
        core::array::from_fn(|i| aug[i][T1])
    }

    pub(crate) fn derive() -> Tables {
        let first = HALF_FULL_ROUNDS;
        // 1. Constants, carried forward.
        let mut carry = [Fp::zero(); STATE_FELTS];
        let mut lane0 = [Fp::zero(); PARTIAL_ROUNDS];
        for (r, slot) in lane0.iter_mut().enumerate() {
            let e: [Fp; STATE_FELTS] =
                core::array::from_fn(|i| Fp::from(ROUND_CONSTANTS[first + r][i]) + carry[i]);
            *slot = e[0];
            let mut tail = e;
            tail[0] = Fp::zero();
            carry = mds(&tail);
        }
        let after: [Fp; STATE_FELTS] = core::array::from_fn(|i| {
            Fp::from(ROUND_CONSTANTS[first + PARTIAL_ROUNDS][i]) + carry[i]
        });
        // 2. The linear layers, factored.
        let mut prev = [[Fp::zero(); T1]; T1];
        for (i, row) in prev.iter_mut().enumerate() {
            row[i] = Fp::one();
        }
        let mut sparse = Vec::with_capacity(PARTIAL_ROUNDS);
        for _ in 0..PARTIAL_ROUNDS {
            // N = M·diag(1, prev).
            let n = |i: usize, j: usize| -> Fp {
                if j == 0 {
                    m(i, 0)
                } else {
                    (0..T1).fold(Fp::zero(), |acc, k| acc + m(i, 1 + k) * prev[k][j - 1])
                }
            };
            let a: [[Fp; T1]; T1] =
                core::array::from_fn(|i| core::array::from_fn(|j| n(1 + i, 1 + j)));
            let nv: [Fp; T1] = core::array::from_fn(|i| n(1 + i, 0));
            let v = solve(&a, &nv);
            sparse.push(Sparse {
                m00: n(0, 0).canonical(),
                w: core::array::from_fn(|j| n(0, 1 + j).canonical()),
                v: core::array::from_fn(|j| v[j].canonical()),
            });
            prev = a;
        }
        debug_assert_eq!(first + PARTIAL_ROUNDS + HALF_FULL_ROUNDS, NUM_ROUNDS);
        Tables {
            lane0: core::array::from_fn(|r| lane0[r].canonical()),
            sparse,
            tail: core::array::from_fn(|i| core::array::from_fn(|j| prev[i][j].canonical())),
            full: core::array::from_fn(|k| {
                if k == HALF_FULL_ROUNDS {
                    after
                } else {
                    let r = if k < HALF_FULL_ROUNDS {
                        k
                    } else {
                        k + PARTIAL_ROUNDS
                    };
                    core::array::from_fn(|i| Fp::from(ROUND_CONSTANTS[r][i]))
                }
            }),
        }
    }

    // ---- raw Goldilocks arithmetic on `u64` words (any `u64`, not only
    // canonical ones): one reduction per sum of products, not per product.

    /// `2^64 mod p`.
    const EPSILON: u64 = 0xFFFF_FFFF;

    /// A `u128` reduced to a word congruent mod `p` (not necessarily below
    /// `p`): `x_lo + x_hi·2^64`, `2^64 ≡ 2^32 − 1`, `2^96 ≡ −1`.
    #[inline(always)]
    fn reduce128(x: u128) -> u64 {
        let (lo, hi) = (x as u64, (x >> 64) as u64);
        let (hi_hi, hi_lo) = (hi >> 32, hi & EPSILON);
        let (mut t0, borrow) = lo.overflowing_sub(hi_hi);
        if borrow {
            t0 = t0.wrapping_sub(EPSILON);
        }
        let (r, carry) = t0.overflowing_add(hi_lo * EPSILON);
        r.wrapping_add(EPSILON * u64::from(carry))
    }

    /// `Σ a_j·b_j` over raw words with the products' low and high words summed
    /// apart, one reduction at the end.
    #[inline(always)]
    fn dot(a: &[u64], b: &[Fp]) -> Fp {
        let (mut lo, mut hi) = (0u128, 0u128);
        for (x, y) in a.iter().zip(b) {
            let p = u128::from(*x) * u128::from(*y.value());
            lo += u128::from(p as u64);
            hi += p >> 64;
        }
        // hi·2^64 ≡ hi·(2^32 − 1); hi < 2^68, so its reduction times ε < 2^96.
        Fp::from(reduce128(
            lo + u128::from(reduce128(hi)) * u128::from(EPSILON),
        ))
    }

    #[inline(always)]
    fn full_round(rc: &[Fp; STATE_FELTS], s: &mut [Fp; STATE_FELTS]) {
        for (v, c) in s.iter_mut().zip(rc) {
            *v = sbox(&(*v + c));
        }
        *s = mds(s);
    }

    /// ★ The permutation in the sparse form, over `t` (the tests' mutation
    /// seam: production passes [`tables`]).
    pub(crate) fn permute(t: &Tables, state: [Fp; STATE_FELTS]) -> [Fp; STATE_FELTS] {
        let mut s = state;
        for rc in &t.full[..HALF_FULL_ROUNDS] {
            full_round(rc, &mut s);
        }
        for (c0, b) in t.lane0.iter().zip(&t.sparse) {
            let x0 = sbox(&(s[0] + Fp::from(*c0)));
            let new0 = Fp::from(b.m00) * x0 + dot(&b.w, &s[1..]);
            for j in 0..T1 {
                s[1 + j] += Fp::from(b.v[j]) * x0;
            }
            s[0] = new0;
        }
        let rest: [Fp; T1] = core::array::from_fn(|i| dot(&t.tail[i], &s[1..]));
        s[1..].copy_from_slice(&rest);
        for rc in &t.full[HALF_FULL_ROUNDS..] {
            full_round(rc, &mut s);
        }
        s
    }
}

/// The same permutation with the Grain Cauchy MDS — the alternative instance
/// (same rounds, same round constants).
pub fn permute_cauchy(state: [Fp; STATE_FELTS]) -> [Fp; STATE_FELTS] {
    permute_with(&ROUND_CONSTANTS, mds_cauchy, state)
}

/// A permutation over a given constant table and linear layer — the tests'
/// mutation seam.
fn permute_with(
    rc: &[[u64; STATE_FELTS]; NUM_ROUNDS],
    linear: LinearLayer,
    state: [Fp; STATE_FELTS],
) -> [Fp; STATE_FELTS] {
    let mut s = state;
    for r in 0..NUM_ROUNDS {
        s = round_with(rc, linear, &s, r);
    }
    s
}

/// The rate-12 overwrite duplex over a felt stream (a Merkle leaf).
pub fn sponge_leaf(felts: &[Fp]) -> Digest {
    sponge_leaf_with(felts, permute)
}

/// [`sponge_leaf`] under [`permute_cauchy`].
pub fn sponge_leaf_cauchy(felts: &[Fp]) -> Digest {
    sponge_leaf_with(felts, permute_cauchy)
}

fn sponge_leaf_with(felts: &[Fp], perm: fn([Fp; STATE_FELTS]) -> [Fp; STATE_FELTS]) -> Digest {
    let mut state = [Fp::zero(); STATE_FELTS];
    state[RATE_FELTS + CAPACITY_PAD_LANE] = Fp::from((felts.len() % RATE_FELTS) as u64);
    state[RATE_FELTS + CAPACITY_DOMAIN_LANE] = Fp::from(DOMAIN_LEAF);
    for block in felts.chunks(RATE_FELTS) {
        for (lane, slot) in state.iter_mut().take(RATE_FELTS).enumerate() {
            *slot = block.get(lane).copied().unwrap_or_else(Fp::zero);
        }
        state = perm(state);
    }
    [state[0], state[1], state[2], state[3]]
}

/// A 4-ary Merkle node: permute the four children, truncate to four felts.
pub fn compress4(children: &[Digest; ARITY]) -> Digest {
    compress4_with(children, permute)
}

/// [`compress4`] under [`permute_cauchy`].
pub fn compress4_cauchy(children: &[Digest; ARITY]) -> Digest {
    compress4_with(children, permute_cauchy)
}

fn compress4_with(
    children: &[Digest; ARITY],
    perm: fn([Fp; STATE_FELTS]) -> [Fp; STATE_FELTS],
) -> Digest {
    let mut state = [Fp::zero(); STATE_FELTS];
    for (c, child) in children.iter().enumerate() {
        state[c * DIGEST_FELTS..(c + 1) * DIGEST_FELTS].copy_from_slice(child);
    }
    let out = perm(state);
    [out[0], out[1], out[2], out[3]]
}
