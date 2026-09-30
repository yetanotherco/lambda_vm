//! A Poseidon1 width-16 `LFM_HASH`-style chip — a MEASUREMENT chip for the
//! D-HASH track, not wired into the machine.
//!
//! The socket today is 12 felts wide (`HASH_STATE_FELTS`), so a width-16
//! permutation cannot be a `HasherKind` arm without a new tuple contract. What
//! the base-hash decision needs first is the chip's WIDTH (cells per
//! permutation), which decides most of the recursion side of D-HASH's model
//! (§2.3). This module builds the width-16 constraint set in two layouts,
//! fills rows for it, and pins both to the host reference
//! [`crypto::hash::poseidon1_w16`]:
//!
//! - [`Layout::Rule`] — our W12 chip's rule at width 16 (`chips::hash::
//!   poseidon_cols`): per round `x²`, `x³` for each S-boxed lane plus a fresh
//!   16-lane post-MDS output block (the last round's output is `OUT`).
//! - [`Layout::Compact`] — per S-boxed lane `x³ = a³` and `x⁷ = (x³)²·a`, and
//!   NO state columns: every round's input is an affine form in the earlier
//!   `x⁷` columns and the prefix, carried symbolically. Both constraints are
//!   degree 3. Plonky3's `poseidon1-air` commits a post-round state per full
//!   round on top of this (444 value columns at width 16); we skip it.
//!
//! # The socket prefix (width 16)
//!
//! Preprocessed ([`PREP_WIDTH`] = 16, not counted as cells): four input and
//! four output addresses, the four mode selectors, four multiplicities.
//! Value prefix (36): `IN0..16`, the capacity copies `S12..16`, `OUT0..16`.
//! The bus would carry four input and four output word tuples (8 interactions,
//! 4 LogUp aux columns); the aux columns are counted, not built.
//!
//! Modes: `MODE_P` (permutation) and `MODE_C` (4-ary node: the four child
//! digests fill all 16 lanes, so it takes its capacity lanes from `IN` like a
//! permutation) copy `S_k = IN_{12+k}`; `MODE_T` and `MODE_L` take a domain
//! capacity `[0, domain, 0, 0]`. The round constants are scaled by the mode
//! sum, so an all-zero padding row satisfies every constraint.

use stark::constraints::builder::{ConstraintBuilder, ConstraintSet};

use crate::tables::types::{FE, GoldilocksExtension, GoldilocksField};
use crypto::hash::poseidon1_w16::{
    self as p1, DOMAIN_LEAF, NUM_ROUNDS, RATE_FELTS, STATE_FELTS, constants::MDS_CIRC_ROW,
    constants::ROUND_CONSTANTS, is_full_round,
};
use math::field::traits::IsPrimeField;

type F = GoldilocksField;
type E = GoldilocksExtension;

/// The transcript domain of this measurement instance (`"P1WT"`).
pub const DOMAIN_TRANSCRIPT: u64 = u32::from_le_bytes(*b"P1WT") as u64;

/// Preprocessed columns.
pub const IN_ADDR0: usize = 0; // ..IN_ADDR3
pub const OUT_ADDR0: usize = 4; // ..OUT_ADDR3
pub const MODE_C: usize = 8;
pub const MODE_P: usize = 9;
pub const MODE_T: usize = 10;
pub const MODE_L: usize = 11;
pub const MULT0: usize = 12; // ..MULT3
pub const PREP_WIDTH: usize = 16;

/// Value prefix, shared by both layouts.
pub const IN0: usize = PREP_WIDTH;
pub const S12: usize = IN0 + STATE_FELTS;
pub const OUT0: usize = S12 + 4;
pub const PREFIX_VALUE_COLUMNS: usize = 2 * STATE_FELTS + 4;
/// First appended witness column.
pub const ROUNDS: usize = PREP_WIDTH + PREFIX_VALUE_COLUMNS;

/// Bus interactions the width-16 socket would carry: four word reads, four
/// word writes.
pub const BUS_INTERACTIONS: usize = 8;
/// LogUp aux columns: one per pair of interactions.
pub const AUX_COLUMNS: usize = BUS_INTERACTIONS.div_ceil(2);
/// Base-field cells per aux column (a cubic-extension accumulator), the census
/// convention `621 = 612 + 3·3` of the W12 chip.
pub const CELLS_PER_AUX_COLUMN: usize = 3;

/// S-boxed lanes in round `r`.
pub const fn sboxed_lanes(r: usize) -> usize {
    if is_full_round(r) { STATE_FELTS } else { 1 }
}

/// The two measured layouts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Layout {
    /// Our W12 rule at width 16: `x²`, `x³` per S-box plus a per-round state.
    Rule,
    /// `x³`, `x⁷` per S-box and no state columns.
    Compact,
}

impl Layout {
    /// Width of round `r`'s appended block.
    pub const fn block_width(self, r: usize) -> usize {
        match self {
            Layout::Rule => {
                let out = if r + 1 == NUM_ROUNDS { 0 } else { STATE_FELTS };
                2 * sboxed_lanes(r) + out
            }
            Layout::Compact => 2 * sboxed_lanes(r),
        }
    }

    /// First column of round `r`'s block.
    pub const fn block(self, r: usize) -> usize {
        let mut off = ROUNDS;
        let mut i = 0;
        while i < r {
            off += self.block_width(i);
            i += 1;
        }
        off
    }

    /// The first S-box witness of `lane` in round `r`: `x²` (Rule) or `x³` (Compact).
    pub const fn w1(self, r: usize, lane: usize) -> usize {
        self.block(r) + lane
    }

    /// The second S-box witness: `x³` (Rule) or `x⁷` (Compact).
    pub const fn w2(self, r: usize, lane: usize) -> usize {
        self.block(r) + sboxed_lanes(r) + lane
    }

    /// Rule only: round `r`'s post-MDS output lane `j` (`OUT` for the last round).
    pub const fn out(self, r: usize, j: usize) -> usize {
        if r + 1 == NUM_ROUNDS {
            OUT0 + j
        } else {
            self.block(r) + 2 * sboxed_lanes(r) + j
        }
    }

    /// The chip's total width, preprocessed prefix included.
    pub const fn num_columns(self) -> usize {
        self.block(NUM_ROUNDS)
    }

    /// Value columns: the census's `main_cols`.
    pub const fn value_columns(self) -> usize {
        self.num_columns() - PREP_WIDTH
    }

    /// Base-field cells per permutation: value columns plus the aux columns
    /// at three cells each.
    pub const fn cells_per_permutation(self) -> usize {
        self.value_columns() + CELLS_PER_AUX_COLUMN * AUX_COLUMNS
    }

    /// Constraints: 4 capacity copies, the mode-sum booleanity, then per round
    /// two per S-box; Rule adds 16 MDS outputs per round, Compact 16 at the end.
    pub const fn num_constraints(self) -> usize {
        let mut n = 5;
        let mut r = 0;
        while r < NUM_ROUNDS {
            n += 2 * sboxed_lanes(r);
            if let Layout::Rule = self {
                n += STATE_FELTS;
            }
            r += 1;
        }
        if let Layout::Compact = self {
            n += STATE_FELTS;
        }
        n
    }
}

/// The capacity cell for a domain: `[0, domain, 0, 0]`.
const fn domain_iv(domain: u64) -> [u64; 4] {
    [0, domain, 0, 0]
}

/// The mode a row is filled under.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// A 4-ary node: all 16 lanes from the input.
    Node,
    /// A full permutation.
    Permute,
    /// A transcript step: rate from the input, the transcript capacity.
    Transcript,
    /// A leaf block: rate from the input, the leaf capacity.
    Leaf,
}

/// A row for `input` under `mode`, witnessed in `layout`. Lanes the mode takes
/// from its capacity are overwritten; the filler walks the rounds itself
/// rather than delegating to [`p1::permute`], and the tests pin `OUT` to it.
pub fn fill_row(layout: Layout, mode: Mode, input: [FE; STATE_FELTS]) -> Vec<FE> {
    let mut row = vec![FE::zero(); layout.num_columns()];
    let sel = match mode {
        Mode::Node => MODE_C,
        Mode::Permute => MODE_P,
        Mode::Transcript => MODE_T,
        Mode::Leaf => MODE_L,
    };
    row[sel] = FE::one();
    let mut state = input;
    match mode {
        Mode::Node | Mode::Permute => {}
        Mode::Transcript | Mode::Leaf => {
            let domain = if mode == Mode::Leaf {
                DOMAIN_LEAF
            } else {
                DOMAIN_TRANSCRIPT
            };
            for (k, v) in domain_iv(domain).iter().enumerate() {
                state[RATE_FELTS + k] = FE::from(*v);
            }
        }
    }
    let read_lanes = match mode {
        Mode::Node | Mode::Permute => STATE_FELTS,
        Mode::Transcript | Mode::Leaf => RATE_FELTS,
    };
    row[IN0..IN0 + read_lanes].copy_from_slice(&state[..read_lanes]);
    for k in 0..4 {
        row[S12 + k] = state[RATE_FELTS + k];
    }

    let mut s = state;
    for r in 0..NUM_ROUNDS {
        let a: [FE; STATE_FELTS] = core::array::from_fn(|i| s[i] + FE::from(ROUND_CONSTANTS[r][i]));
        let mut f = a;
        for lane in 0..sboxed_lanes(r) {
            let x2 = a[lane] * a[lane];
            let x3 = x2 * a[lane];
            let x7 = x3 * x3 * a[lane];
            match layout {
                Layout::Rule => {
                    row[layout.w1(r, lane)] = x2;
                    row[layout.w2(r, lane)] = x3;
                }
                Layout::Compact => {
                    row[layout.w1(r, lane)] = x3;
                    row[layout.w2(r, lane)] = x7;
                }
            }
            f[lane] = x7;
        }
        s = p1::mds(&f);
        if layout == Layout::Rule && r + 1 < NUM_ROUNDS {
            for j in 0..STATE_FELTS {
                row[layout.out(r, j)] = s[j];
            }
        }
    }
    row[OUT0..OUT0 + STATE_FELTS].copy_from_slice(&s);
    row
}

/// An affine form over the row: `Σ col[c]·main(c) + m_coeff·m`, with `m` the
/// mode sum. Dense over the columns; the chip is under a thousand wide.
#[derive(Clone)]
struct Affine {
    col: Vec<FE>,
    m: FE,
}

impl Affine {
    fn zero(width: usize) -> Self {
        Self {
            col: vec![FE::zero(); width],
            m: FE::zero(),
        }
    }

    fn column(width: usize, c: usize) -> Self {
        let mut a = Self::zero(width);
        a.col[c] = FE::one();
        a
    }

    fn add_scaled(&mut self, other: &Affine, k: FE) {
        for (x, y) in self.col.iter_mut().zip(&other.col) {
            *x += k * y;
        }
        self.m += k * other.m;
    }

    fn expr<B: ConstraintBuilder<F, E>>(&self, b: &B, m: &B::Expr) -> B::Expr {
        let mut acc = b.const_base(canon(&self.m)) * m.clone();
        for (c, k) in self.col.iter().enumerate() {
            if *k != FE::zero() {
                acc = acc + b.const_base(canon(k)) * b.main(0, c);
            }
        }
        acc
    }
}

fn canon(x: &FE) -> u64 {
    GoldilocksField::canonical(x.value())
}

/// The MDS over affine forms: `out_o = Σ_i row[(i − o) mod 16]·f_i`.
fn mds_affine(f: &[Affine]) -> Vec<Affine> {
    let width = f[0].col.len();
    (0..STATE_FELTS)
        .map(|o| {
            let mut acc = Affine::zero(width);
            for (i, fi) in f.iter().enumerate() {
                acc.add_scaled(
                    fi,
                    FE::from(MDS_CIRC_ROW[(i + STATE_FELTS - o) % STATE_FELTS]),
                );
            }
            acc
        })
        .collect()
}

/// The width-16 constraint set under a layout.
pub struct P1W16Constraints {
    pub layout: Layout,
}

impl ConstraintSet<F, E> for P1W16Constraints {
    fn max_degree(&self) -> usize {
        3
    }

    fn eval<B: ConstraintBuilder<F, E>>(&self, b: &mut B) {
        let layout = self.layout;
        let width = layout.num_columns();
        let mode_c = b.main(0, MODE_C);
        let mode_p = b.main(0, MODE_P);
        let mode_t = b.main(0, MODE_T);
        let mode_l = b.main(0, MODE_L);
        let m = mode_c.clone() + mode_p.clone() + mode_t.clone() + mode_l.clone();

        // idx 0–3: S_k = (MODE_C + MODE_P)·IN_{12+k} + MODE_T·IVT_k + MODE_L·IVL_k.
        for k in 0..4 {
            let s = b.main(0, S12 + k);
            let in_k = b.main(0, IN0 + RATE_FELTS + k);
            let mut rhs = (mode_c.clone() + mode_p.clone()) * in_k;
            for (sel, domain) in [(&mode_t, DOMAIN_TRANSCRIPT), (&mode_l, DOMAIN_LEAF)] {
                let iv = domain_iv(domain)[k];
                if iv != 0 {
                    rhs = rhs + sel.clone() * b.const_base(iv);
                }
            }
            b.emit_base(k, s - rhs);
        }
        // idx 4: the mode sum is a bit.
        let one = b.one();
        b.emit_base(4, m.clone() * (one - m.clone()));

        let mut idx = 5;
        // The state entering round 0: IN for the rate, S for the capacity.
        let mut state: Vec<Affine> = (0..STATE_FELTS)
            .map(|i| {
                let c = if i < RATE_FELTS {
                    IN0 + i
                } else {
                    S12 + (i - RATE_FELTS)
                };
                Affine::column(width, c)
            })
            .collect();
        for (r, rc_row) in ROUND_CONSTANTS.iter().enumerate() {
            // a_i = state_i + rc[r][i]·m.
            let a: Vec<Affine> = state
                .iter()
                .zip(rc_row)
                .map(|(s, rc)| {
                    let mut x = s.clone();
                    x.m += FE::from(*rc);
                    x
                })
                .collect();
            let mut f: Vec<Affine> = a.clone();
            for lane in 0..sboxed_lanes(r) {
                let a_e = a[lane].expr(b, &m);
                let w1 = b.main(0, layout.w1(r, lane));
                let w2 = b.main(0, layout.w2(r, lane));
                match layout {
                    Layout::Rule => {
                        // x2 = a², x3 = x2·a; the S-box output (x3)²·a is an
                        // expression entering the MDS constraint below.
                        b.emit_base(idx, w1.clone() - a_e.clone() * a_e.clone());
                        b.emit_base(idx + 1, w2 - w1 * a_e);
                    }
                    Layout::Compact => {
                        // x3 = a³, x7 = (x3)²·a — both degree 3.
                        b.emit_base(idx, w1.clone() - a_e.clone() * a_e.clone() * a_e.clone());
                        b.emit_base(idx + 1, w2 - w1.clone() * w1 * a_e);
                        f[lane] = Affine::column(width, layout.w2(r, lane));
                    }
                }
                idx += 2;
            }
            match layout {
                Layout::Rule => {
                    // out_o = MDS(f)_o, with f_lane = (x3)²·a on S-boxed lanes.
                    for o in 0..STATE_FELTS {
                        let mut acc = b.zero();
                        for (i, fi) in f.iter().enumerate() {
                            let c = b.const_base(MDS_CIRC_ROW[(i + STATE_FELTS - o) % STATE_FELTS]);
                            let term = if i < sboxed_lanes(r) {
                                let x3 = b.main(0, layout.w2(r, i));
                                x3.clone() * x3 * a[i].expr(b, &m)
                            } else {
                                fi.expr(b, &m)
                            };
                            acc = acc + c * term;
                        }
                        let out = b.main(0, layout.out(r, o));
                        b.emit_base(idx, out - acc);
                        idx += 1;
                    }
                    state = (0..STATE_FELTS)
                        .map(|j| Affine::column(width, layout.out(r, j)))
                        .collect();
                }
                Layout::Compact => {
                    state = mds_affine(&f);
                }
            }
        }
        if layout == Layout::Compact {
            for (j, s) in state.iter().enumerate() {
                let out = b.main(0, OUT0 + j);
                b.emit_base(idx, out - s.expr(b, &m));
                idx += 1;
            }
        }
        debug_assert_eq!(idx, layout.num_constraints());
    }
}
