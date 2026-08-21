//! The LFM chips: bus interactions and constraint sets.
//!
//! Eleven chips live here. The other three slots of the fixed AIR set are the
//! production keccak family (`KECCAK_RND` / `KECCAK_RC` / `BITWISE`), hosted
//! unchanged from `tables/` and driven by `LFM_KECCAK` below.
//!
//! Shared conventions (see `SOUNDNESS.md` and the design doc):
//! - each chip's trace = its instruction column group (preprocessed, leading
//!   columns, layout in [`super::layout`]) followed by the value columns
//!   defined here;
//! - one sign convention machine-wide: writes are senders with
//!   `Multiplicity::Column(mult)` (a preprocessed column), reads are
//!   receivers gated by selectors / `is_real` (also preprocessed); no
//!   `Negated` forms anywhere;
//! - the `LfmMem` token is `(addr, v0, v1, v2, v3)`; base values carry
//!   constant-zero high lanes *in the tuple*, so a base cell cannot smuggle
//!   extension lanes;
//! - in-AIR checks are per-op algebra plus belt-over-suspenders booleanity;
//!   uniqueness/acyclicity/mult-equality/one-hot-ness are the registrar's
//!   (admission validator), per the soundness split.

use stark::constraints::builder::{ConstraintBuilder, ConstraintSet};
use stark::lookup::{BusInteraction, BusValue, LinearTerm, Multiplicity, Packing};

use crate::tables::types::{BusId, GoldilocksExtension, GoldilocksField};

type F = GoldilocksField;
type E = GoldilocksExtension;

fn direct(col: usize) -> BusValue {
    BusValue::Packed {
        start_column: col,
        packing: Packing::Direct,
    }
}

fn zero() -> BusValue {
    BusValue::constant(0)
}

/// A word value spread over four adjacent columns.
fn word(cols_start: usize) -> [BusValue; 4] {
    [
        direct(cols_start),
        direct(cols_start + 1),
        direct(cols_start + 2),
        direct(cols_start + 3),
    ]
}

/// `Σ` of a run of selector columns, as a LogUp multiplicity (= is_real).
fn selector_sum(first: usize, count: usize) -> Multiplicity {
    Multiplicity::Linear(
        (0..count)
            .map(|i| LinearTerm::ColumnUnsigned {
                coefficient: 1,
                column: first + i,
            })
            .collect(),
    )
}

/// A base-value memory token: `(addr, v, 0, 0, 0)`.
fn base_token(addr_col: usize, val_col: usize) -> Vec<BusValue> {
    vec![direct(addr_col), direct(val_col), zero(), zero(), zero()]
}

/// An ext-value memory token: `(addr, v0, v1, v2, 0)` — lane 3 is a tuple
/// constant, which is exactly what pins ext cells to lane-3-zero.
fn ext_token(addr_col: usize, lanes_start: usize) -> Vec<BusValue> {
    vec![
        direct(addr_col),
        direct(lanes_start),
        direct(lanes_start + 1),
        direct(lanes_start + 2),
        zero(),
    ]
}

/// A full-word memory token: `(addr, v0..v3)`.
fn word_token(addr_col: usize, lanes_start: usize) -> Vec<BusValue> {
    let mut v = vec![direct(addr_col)];
    v.extend(word(lanes_start));
    v
}

// =========================================================================
// LFM_CONST — pooled constants (all instruction data preprocessed)
// =========================================================================

pub mod const_ {
    use super::*;

    pub mod cols {
        pub use crate::lfm::layout::const_::*;
        /// All-zero main column: the commit path expects a non-empty
        /// non-preprocessed subset (KECCAK_RC precedent).
        pub const PAD: usize = PREP_WIDTH;
        pub const NUM_COLUMNS: usize = PREP_WIDTH + 1;
    }

    pub fn bus_interactions() -> Vec<BusInteraction> {
        vec![BusInteraction::sender(
            BusId::LfmMem,
            Multiplicity::Column(cols::MULT),
            word_token(cols::ADDR, cols::V0),
        )]
    }
}

// =========================================================================
// LFM_BALU — Goldilocks ALU
// =========================================================================

pub mod balu {
    use super::*;

    pub mod cols {
        pub use crate::lfm::layout::balu::*;
        pub const A: usize = PREP_WIDTH;
        pub const B: usize = PREP_WIDTH + 1;
        pub const C: usize = PREP_WIDTH + 2;
        pub const OUT: usize = PREP_WIDTH + 3;
        pub const NUM_COLUMNS: usize = PREP_WIDTH + 4;
    }

    pub fn bus_interactions() -> Vec<BusInteraction> {
        vec![
            BusInteraction::receiver(
                BusId::LfmMem,
                selector_sum(cols::SEL_ADD, cols::NUM_SELECTORS),
                base_token(cols::A_ADDR, cols::A),
            ),
            BusInteraction::receiver(
                BusId::LfmMem,
                selector_sum(cols::SEL_ADD, cols::NUM_SELECTORS),
                base_token(cols::B_ADDR, cols::B),
            ),
            BusInteraction::receiver(
                BusId::LfmMem,
                Multiplicity::Column(cols::SEL_MULADD),
                base_token(cols::C_ADDR, cols::C),
            ),
            BusInteraction::sender(
                BusId::LfmMem,
                Multiplicity::Column(cols::MULT),
                base_token(cols::OUT_ADDR, cols::OUT),
            ),
        ]
    }

    pub struct BaluConstraints;

    impl ConstraintSet<F, E> for BaluConstraints {
        fn max_degree(&self) -> usize {
            3
        }

        fn eval<B: ConstraintBuilder<F, E>>(&self, b: &mut B) {
            let a = b.main(0, cols::A);
            let bb = b.main(0, cols::B);
            let c = b.main(0, cols::C);
            let out = b.main(0, cols::OUT);
            let sel = |b: &B, i: usize| b.main(0, cols::SEL_ADD + i);

            // idx 0: add — sel·(a + b − out)
            b.emit_base(0, sel(b, 0) * (a.clone() + bb.clone() - out.clone()));
            // idx 1: sub — sel·(a − b − out)
            b.emit_base(1, sel(b, 1) * (a.clone() - bb.clone() - out.clone()));
            // idx 2: mul — sel·(a·b − out)
            b.emit_base(2, sel(b, 2) * (a.clone() * bb.clone() - out.clone()));
            // idx 3: div as reversed mul — sel·(b·out − a). With b = 0 this
            // forces a = 0 and leaves out free (the executor pins 0/0 = 1):
            // the assert-via-division mechanism.
            b.emit_base(3, sel(b, 3) * (bb.clone() * out.clone() - a.clone()));
            // idx 4: mul-add — sel·(a·b + c − out) (the Horner step)
            b.emit_base(4, sel(b, 4) * (a * bb + c - out));
            // idx 5: selector sum-boolean (belt; one-hot is the registrar's)
            let sum = (1..cols::NUM_SELECTORS).fold(sel(b, 0), |acc, i| acc + sel(b, i));
            let one = b.one();
            b.emit_base(5, sum.clone() * (one - sum));
        }
    }
}

// =========================================================================
// LFM_XALU — Fp3 ALU on word lanes 0–2 (w³ = 2)
// =========================================================================

pub mod xalu {
    use super::*;

    pub mod cols {
        pub use crate::lfm::layout::xalu::*;
        pub const A0: usize = PREP_WIDTH; // ..A2
        pub const B0: usize = PREP_WIDTH + 3; // ..B2
        pub const C0: usize = PREP_WIDTH + 6; // ..C2
        pub const OUT0: usize = PREP_WIDTH + 9; // ..OUT2
        pub const NUM_COLUMNS: usize = PREP_WIDTH + 12;
    }

    pub fn bus_interactions() -> Vec<BusInteraction> {
        vec![
            BusInteraction::receiver(
                BusId::LfmMem,
                selector_sum(cols::SEL_ADD, cols::NUM_SELECTORS),
                ext_token(cols::A_ADDR, cols::A0),
            ),
            BusInteraction::receiver(
                BusId::LfmMem,
                selector_sum(cols::SEL_ADD, cols::NUM_SELECTORS),
                ext_token(cols::B_ADDR, cols::B0),
            ),
            BusInteraction::receiver(
                BusId::LfmMem,
                Multiplicity::Column(cols::SEL_MULADD),
                ext_token(cols::C_ADDR, cols::C0),
            ),
            BusInteraction::sender(
                BusId::LfmMem,
                Multiplicity::Column(cols::MULT),
                ext_token(cols::OUT_ADDR, cols::OUT0),
            ),
        ]
    }

    pub struct XaluConstraints;

    impl XaluConstraints {
        /// The three product lanes of `X·Y` in `Fp[w]/(w³ − 2)`:
        /// `p0 = x0y0 + 2(x1y2 + x2y1)`, `p1 = x0y1 + x1y0 + 2·x2y2`,
        /// `p2 = x0y2 + x1y1 + x2y0` (matches `Degree3GoldilocksExtensionField::mul`).
        fn product<B: ConstraintBuilder<F, E>>(b: &B, x0: usize, y0: usize) -> [B::Expr; 3] {
            let x = |i: usize| b.main(0, x0 + i);
            let y = |i: usize| b.main(0, y0 + i);
            let two = b.const_base(2);
            [
                x(0) * y(0) + two.clone() * (x(1) * y(2) + x(2) * y(1)),
                x(0) * y(1) + x(1) * y(0) + two * x(2) * y(2),
                x(0) * y(2) + x(1) * y(1) + x(2) * y(0),
            ]
        }
    }

    impl ConstraintSet<F, E> for XaluConstraints {
        fn max_degree(&self) -> usize {
            3
        }

        fn eval<B: ConstraintBuilder<F, E>>(&self, b: &mut B) {
            let sel = |b: &B, i: usize| b.main(0, cols::SEL_ADD + i);
            let lane = |b: &B, base: usize, j: usize| b.main(0, base + j);

            // idx 0–2 add / 3–5 sub: componentwise.
            for j in 0..3 {
                let a = lane(b, cols::A0, j);
                let bb = lane(b, cols::B0, j);
                let out = lane(b, cols::OUT0, j);
                b.emit_base(j, sel(b, 0) * (a.clone() + bb.clone() - out.clone()));
                b.emit_base(3 + j, sel(b, 1) * (a - bb - out));
            }
            // idx 6–8 mul: P(A, B) = OUT.
            let p_ab = Self::product(b, cols::A0, cols::B0);
            for (j, p) in p_ab.into_iter().enumerate() {
                b.emit_base(6 + j, sel(b, 2) * (p - lane(b, cols::OUT0, j)));
            }
            // idx 9–11 div as reversed mul: P(B, OUT) = A (0/0 = (1,0,0) in
            // the executor; x/0 unprovable — the ext assert mechanism).
            let p_bout = Self::product(b, cols::B0, cols::OUT0);
            for (j, p) in p_bout.into_iter().enumerate() {
                b.emit_base(9 + j, sel(b, 3) * (p - lane(b, cols::A0, j)));
            }
            // idx 12–14 mul-add: P(A, B) + C = OUT.
            let p_ab = Self::product(b, cols::A0, cols::B0);
            for (j, p) in p_ab.into_iter().enumerate() {
                b.emit_base(
                    12 + j,
                    sel(b, 4) * (p + lane(b, cols::C0, j) - lane(b, cols::OUT0, j)),
                );
            }
            // idx 15–17 mul-base: OUT_j = A_j · B0.
            for j in 0..3 {
                b.emit_base(
                    15 + j,
                    sel(b, 5)
                        * (lane(b, cols::A0, j) * lane(b, cols::B0, 0) - lane(b, cols::OUT0, j)),
                );
            }
            // idx 18–19: MulBase's operand is a base word — the shared B
            // lanes 1–2 must vanish on its rows or the received token would
            // not match any base writer's.
            b.emit_base(18, sel(b, 5) * lane(b, cols::B0, 1));
            b.emit_base(19, sel(b, 5) * lane(b, cols::B0, 2));
            // idx 20: selector sum-boolean.
            let sum = (1..cols::NUM_SELECTORS).fold(sel(b, 0), |acc, i| acc + sel(b, i));
            let one = b.one();
            b.emit_base(20, sum.clone() * (one - sum));
        }
    }
}

// =========================================================================
// LFM_SELECT — conditional cell swap
// =========================================================================

pub mod select {
    use super::*;

    pub mod cols {
        pub use crate::lfm::layout::select::*;
        pub const BIT: usize = PREP_WIDTH;
        pub const INL0: usize = PREP_WIDTH + 1; // ..+4
        pub const INR0: usize = PREP_WIDTH + 5; // ..+8
        pub const OUTL0: usize = PREP_WIDTH + 9; // ..+12
        pub const OUTR0: usize = PREP_WIDTH + 13; // ..+16
        pub const NUM_COLUMNS: usize = PREP_WIDTH + 17;
    }

    pub fn bus_interactions() -> Vec<BusInteraction> {
        vec![
            BusInteraction::receiver(
                BusId::LfmMem,
                Multiplicity::Column(cols::IS_REAL),
                base_token(cols::BIT_ADDR, cols::BIT),
            ),
            BusInteraction::receiver(
                BusId::LfmMem,
                Multiplicity::Column(cols::IS_REAL),
                word_token(cols::INL_ADDR, cols::INL0),
            ),
            BusInteraction::receiver(
                BusId::LfmMem,
                Multiplicity::Column(cols::IS_REAL),
                word_token(cols::INR_ADDR, cols::INR0),
            ),
            BusInteraction::sender(
                BusId::LfmMem,
                Multiplicity::Column(cols::MULT_L),
                word_token(cols::OUTL_ADDR, cols::OUTL0),
            ),
            BusInteraction::sender(
                BusId::LfmMem,
                Multiplicity::Column(cols::MULT_R),
                word_token(cols::OUTR_ADDR, cols::OUTR0),
            ),
        ]
    }

    pub struct SelectConstraints;

    impl ConstraintSet<F, E> for SelectConstraints {
        fn max_degree(&self) -> usize {
            2
        }

        fn eval<B: ConstraintBuilder<F, E>>(&self, b: &mut B) {
            // idx 0: bit booleanity (belt over suspenders — a witness bit
            // exists, so it is constrained here and not only vouched).
            let bit = b.main(0, cols::BIT);
            let one = b.one();
            b.emit_base(0, bit.clone() * (one - bit));
            // idx 1–4 / 5–8: out_l = in_l + bit·(in_r − in_l); out_r mirrored.
            // Trivially satisfied on zero-filled padding rows.
            for j in 0..4 {
                let bit = b.main(0, cols::BIT);
                let inl = b.main(0, cols::INL0 + j);
                let inr = b.main(0, cols::INR0 + j);
                let outl = b.main(0, cols::OUTL0 + j);
                b.emit_base(
                    1 + j,
                    outl - (inl.clone() + bit.clone() * (inr.clone() - inl.clone())),
                );
                let outr = b.main(0, cols::OUTR0 + j);
                b.emit_base(5 + j, outr - (inr.clone() + bit * (inl - inr)));
            }
        }
    }
}

// =========================================================================
// LFM_BITDEC — canonical 64-bit decomposition over p = 2^64 − 2^32 + 1
// =========================================================================

pub mod bitdec {
    use super::*;

    pub mod cols {
        pub use crate::lfm::layout::bitdec::*;
        pub const BITS0: usize = PREP_WIDTH; // 64 bit columns, low-to-high
        pub const Z: usize = PREP_WIDTH + NUM_BITS;
        pub const GINV: usize = PREP_WIDTH + NUM_BITS + 1;
        pub const NUM_COLUMNS: usize = PREP_WIDTH + NUM_BITS + 2;
    }

    pub fn bus_interactions() -> Vec<BusInteraction> {
        // The received value is the recomposition Σ 2^i·B_i, expressed as a
        // linear bus value over the bit columns — no input column needed.
        let recomposition = BusValue::Linear(
            (0..cols::NUM_BITS)
                .map(|i| LinearTerm::ColumnUnsigned {
                    coefficient: 1u64 << i,
                    column: cols::BITS0 + i,
                })
                .collect(),
        );
        let mut interactions = vec![BusInteraction::receiver(
            BusId::LfmMem,
            Multiplicity::Column(cols::IS_REAL),
            vec![direct(cols::IN_ADDR), recomposition, zero(), zero(), zero()],
        )];
        for i in 0..cols::NUM_BITS {
            interactions.push(BusInteraction::sender(
                BusId::LfmMem,
                Multiplicity::Column(cols::bit_mult(i)),
                base_token(cols::bit_addr(i), cols::BITS0 + i),
            ));
        }
        // The two BIG-ENDIAN halves, as linear forms over the SAME bit
        // columns booleanity and canonicity already pin: bit `8k + j` of
        // half-word `h` (h = 0 is the value's HIGH word — it leads in
        // big-endian order) lands at byte `3 − k`, so its weight is
        // `2^(j + 8(3 − k))`. No value column, no constraint: the senders
        // are functions of already-constrained columns.
        for (h, first) in [(0usize, 32usize), (1, 0)] {
            let half = BusValue::Linear(
                (0..4)
                    .flat_map(|k| (0..8).map(move |j| (k, j)))
                    .map(|(k, j)| LinearTerm::ColumnUnsigned {
                        coefficient: 1u64 << (j + 8 * (3 - k)),
                        column: cols::BITS0 + first + 8 * k + j,
                    })
                    .collect(),
            );
            let (addr, mult) = if h == 0 {
                (cols::HALF0_ADDR, cols::HALF0_MULT)
            } else {
                (cols::HALF1_ADDR, cols::HALF1_MULT)
            };
            interactions.push(BusInteraction::sender(
                BusId::LfmMem,
                Multiplicity::Column(mult),
                vec![direct(addr), half, zero(), zero(), zero()],
            ));
        }
        interactions
    }

    pub struct BitDecConstraints;

    impl ConstraintSet<F, E> for BitDecConstraints {
        fn max_degree(&self) -> usize {
            3
        }

        fn eval<B: ConstraintBuilder<F, E>>(&self, b: &mut B) {
            // idx 0–63: booleanity.
            for i in 0..cols::NUM_BITS {
                let bit = b.main(0, cols::BITS0 + i);
                let one = b.one();
                b.emit_base(i, bit.clone() * (one - bit));
            }
            // Canonicity: p − 1 = (2^32 − 1)·2^32, i.e. 32 ones ‖ 32 zeros,
            // so value < p ⟺ (top 32 bits all ones ⇒ bottom 32 bits zero).
            // G = (2^32 − 1) − Σ_{i=32..63} 2^{i−32}·B_i; witnesses Z ("top
            // all ones"), GINV (= G⁻¹ when G ≠ 0).
            let top = (0..32).fold(None::<B::Expr>, |acc, k| {
                let term = b.const_base(1u64 << k) * b.main(0, cols::BITS0 + 32 + k);
                Some(match acc {
                    None => term,
                    Some(a) => a + term,
                })
            });
            let g = b.const_base(0xFFFF_FFFF) - top.expect("nonempty");
            let z = b.main(0, cols::Z);
            let ginv = b.main(0, cols::GINV);
            // idx 64: Z·G = 0 — G ≠ 0 forces Z = 0.
            b.emit_base(64, z.clone() * g.clone());
            // idx 65: IS_REAL·(1 − Z − G·GINV) = 0 — G = 0 forces Z = 1.
            // Gated by IS_REAL so zero-filled padding rows satisfy it.
            let is_real = b.main(0, cols::IS_REAL);
            let one = b.one();
            b.emit_base(65, is_real * (one - z.clone() - g * ginv));
            // idx 66: Z·(Σ_{i<32} 2^i·B_i) = 0 — top all ones ⇒ bottom zero.
            let low = (0..32).fold(None::<B::Expr>, |acc, k| {
                let term = b.const_base(1u64 << k) * b.main(0, cols::BITS0 + k);
                Some(match acc {
                    None => term,
                    Some(a) => a + term,
                })
            });
            b.emit_base(66, z * low.expect("nonempty"));
        }
    }
}

// =========================================================================
// LFM_HASH — the chiplet (frozen tuple contract; TestPermutation behind it)
// =========================================================================

pub mod hash {
    use super::*;
    use crate::lfm::hash::{HASH_STATE_FELTS, HasherKind, TestPermutation};
    use crate::lfm::instr::HashMode;
    use crate::tables::types::FE;
    use math::field::traits::IsPrimeField;

    pub mod cols {
        pub use crate::lfm::layout::hash::*;
        pub const IN0: usize = PREP_WIDTH; // ..IN11
        /// Materialized capacity-state columns for lanes 8–11:
        /// `S_i = MODE_P·IN_i + (MODE_C + MODE_T + MODE_L)·IV_i` (degree-2
        /// copy), so the permutation constraint stays at degree 3. Transcript
        /// and leaf rows are compresses in every structural respect, so they
        /// take the IV too.
        pub const S8: usize = PREP_WIDTH + 12; // ..S11
        pub const OUT0: usize = PREP_WIDTH + 16; // ..OUT11
        /// Value columns every hasher's layout shares: `IN`, `S`, `OUT`. The
        /// bus tuples read only these (`bus_interactions`), which is why they
        /// keep their offsets in EVERY layout — a candidate appends its
        /// witness columns after them rather than reflowing the prefix.
        pub const SHARED_VALUE_COLUMNS: usize = 28;
        /// Width of the [`HasherKind::Test`] layout. Use [`super::num_columns`]
        /// unless you specifically mean `TestPermutation`.
        pub const TEST_NUM_COLUMNS: usize = PREP_WIDTH + SHARED_VALUE_COLUMNS;
    }

    /// Column layout for the [`HasherKind::Poseidon`] configuration.
    ///
    /// The frozen prefix (`IN0..12`, `S8..12`, `OUT0..12`) keeps the offsets
    /// `cols` gives it, so [`bus_interactions`] is hasher-INDEPENDENT and the
    /// `LFM_HASH` tuple contract stays literally frozen. Everything Poseidon
    /// additionally witnesses is appended from [`ROUNDS`] on: per round, the
    /// `x²` and `x³` intermediates of its S-boxed lanes plus its post-MDS
    /// output — except the LAST round, whose output IS `OUT0..12`.
    ///
    /// Width: `28 + 7·36 + 24 + 22·14 = 612` value columns, one row per
    /// permutation.
    ///
    /// ⚠ This layout is a deliberate UPPER BOUND, roughly 2× a known-achievable
    /// one (Miden's measured Poseidon2 at the same width is 256 main cells via
    /// 16 columns × 16 rows, reusing state columns across rounds instead of
    /// allocating fresh ones). It is not optimised because the epoch verifier's
    /// already-measured non-hash residue dominates the total: halving the hash
    /// term moves the epoch bill by ~3%. Measure here, optimise elsewhere.
    pub mod poseidon_cols {
        use crate::lfm::hash::HASH_STATE_FELTS;
        use crate::lfm::poseidon::{NUM_ROUNDS, sboxed_lanes};

        pub use super::cols::{
            IN0, MODE_C, MODE_L, MODE_P, MODE_T, OUT0, PREP_WIDTH, S8, SHARED_VALUE_COLUMNS,
        };

        /// First appended witness column.
        pub const ROUNDS: usize = PREP_WIDTH + SHARED_VALUE_COLUMNS;

        /// Width of round `r`'s appended block: `x²` and `x³` for each S-boxed
        /// lane, plus 12 output columns — none for the last round, which writes
        /// its output into `OUT`.
        pub const fn block_width(r: usize) -> usize {
            let out = if r + 1 == NUM_ROUNDS {
                0
            } else {
                HASH_STATE_FELTS
            };
            2 * sboxed_lanes(r) + out
        }

        /// First column of round `r`'s appended block.
        pub const fn block(r: usize) -> usize {
            let mut off = ROUNDS;
            let mut i = 0;
            while i < r {
                off += block_width(i);
                i += 1;
            }
            off
        }

        /// `a_lane²` for round `r`. Only lanes `< sboxed_lanes(r)` exist.
        pub const fn x2(r: usize, lane: usize) -> usize {
            block(r) + lane
        }

        /// `a_lane³` for round `r`. Only lanes `< sboxed_lanes(r)` exist.
        pub const fn x3(r: usize, lane: usize) -> usize {
            block(r) + sboxed_lanes(r) + lane
        }

        /// Round `r`'s post-MDS output lane `j` — `OUT` for the final round.
        pub const fn out(r: usize, j: usize) -> usize {
            if r + 1 == NUM_ROUNDS {
                OUT0 + j
            } else {
                block(r) + 2 * sboxed_lanes(r) + j
            }
        }

        pub const NUM_COLUMNS: usize = block(NUM_ROUNDS);

        /// 4 capacity copies + 1 mode-boolean + per round (`2·sboxed` S-box
        /// steps and 12 MDS outputs).
        pub const NUM_CONSTRAINTS: usize = {
            let mut n = 5 + super::NUM_UNREAD_INPUT_PINS;
            let mut r = 0;
            while r < NUM_ROUNDS {
                n += 2 * sboxed_lanes(r) + HASH_STATE_FELTS;
                r += 1;
            }
            n
        };
    }

    /// The chip's total width under `kind` — the number the AIR is built with,
    /// the census reads, and the trace filler allocates.
    pub const fn num_columns(kind: HasherKind) -> usize {
        match kind {
            HasherKind::Test => cols::TEST_NUM_COLUMNS,
            HasherKind::Poseidon => poseidon_cols::NUM_COLUMNS,
            HasherKind::Blake3 => crate::lfm::blake3_socket::cols::NUM_COLUMNS,
        }
    }

    /// The chip's bus interactions under `kind`.
    ///
    /// **Hasher-DEPENDENT, and BLAKE3 is why.** The six `LfmMem` tuples below
    /// are the frozen `LFM_HASH` contract and are the same under every
    /// candidate; they read and write only the shared value prefix, whose
    /// offsets no layout moves. But a candidate built out of byte operations
    /// needs a lookup table, and BLAKE3 needs one per XOR byte and one per
    /// range-checked byte pair — over a thousand of them, none of which
    /// `TestPermutation` or Poseidon has, both being pure field arithmetic.
    ///
    /// Callers must thread the same `kind` they build the AIR's width and
    /// constraints with; `LfmAirs::new_with_hasher` is the one place that does.
    pub fn bus_interactions(kind: HasherKind) -> Vec<BusInteraction> {
        let mut interactions = lfm_mem_interactions();
        if kind == HasherKind::Blake3 {
            interactions.extend(crate::lfm::blake3_socket::bitwise_interactions());
        }
        interactions
    }

    /// The frozen `LFM_HASH` tuple contract: 2 (or 3) cells in, 1 (or 3) out.
    ///
    /// The FIRST TWO input cells are read in every mode, so their multiplicity
    /// is the row's is-real flag: the sum of all four mode selectors, which the
    /// AIR pins to a bit. The third is read only by a permutation.
    ///
    /// ⚠ The second cell's multiplicity used to EXCLUDE `MODE_L`, because a leaf
    /// row read one cell of four felts and receiving a second would have claimed
    /// a memory read it never made. Under the leaf RATE a leaf row reads two —
    /// a chaining accumulator and a felt cell — so that exclusion became the
    /// opposite bug: the felts would never be read from memory at all (COMMIT.md
    /// §1.4.4 **H3**). The bus ARITY does not move; this multiplicity is the one
    /// part of the frozen contract that the RATE does.
    fn lfm_mem_interactions() -> Vec<BusInteraction> {
        let is_real = || selector_sum(cols::MODE_C, cols::NUM_SELECTORS);
        vec![
            BusInteraction::receiver(
                BusId::LfmMem,
                is_real(),
                word_token(cols::IN_ADDR0, cols::IN0),
            ),
            BusInteraction::receiver(
                BusId::LfmMem,
                is_real(),
                word_token(cols::IN_ADDR1, cols::IN0 + 4),
            ),
            BusInteraction::receiver(
                BusId::LfmMem,
                Multiplicity::Column(cols::MODE_P),
                word_token(cols::IN_ADDR2, cols::IN0 + 8),
            ),
            BusInteraction::sender(
                BusId::LfmMem,
                Multiplicity::Column(cols::MULT0),
                word_token(cols::OUT_ADDR0, cols::OUT0),
            ),
            BusInteraction::sender(
                BusId::LfmMem,
                Multiplicity::Column(cols::MULT1),
                word_token(cols::OUT_ADDR1, cols::OUT0 + 4),
            ),
            BusInteraction::sender(
                BusId::LfmMem,
                Multiplicity::Column(cols::MULT2),
                word_token(cols::OUT_ADDR2, cols::OUT0 + 8),
            ),
        ]
    }

    fn canonical_u64(fe: &FE) -> u64 {
        GoldilocksField::canonical(fe.value())
    }

    /// Every mode selector paired with the mode it selects.
    ///
    /// One table, so the input pins below and anything else that reasons per
    /// mode read the same mapping rather than each carrying its own copy.
    pub(crate) const MODE_SELECTORS: [(usize, HashMode); 4] = [
        (cols::MODE_C, HashMode::Compress),
        (cols::MODE_T, HashMode::Transcript),
        (cols::MODE_L, HashMode::Leaf),
        (cols::MODE_P, HashMode::Permute),
    ];

    /// Input cell slots that SOME mode does not read, and which therefore need
    /// pinning. Cell 0 is read by every mode and is never a candidate.
    ///
    /// Derived rather than written down: the leaf RATE took `Leaf` from one
    /// input cell to two, which emptied slot 1's set. Left as a literal, the
    /// emitter's `.expect("some mode reads fewer than three input cells")` would
    /// have fired and AIR construction would have panicked (COMMIT.md §1.4.4
    /// **H2**).
    const fn unread_input_slots() -> usize {
        let mut slots = 0;
        let mut slot = 1;
        while slot < 3 {
            let mut i = 0;
            while i < MODE_SELECTORS.len() {
                if MODE_SELECTORS[i].1.num_input_cells() <= slot {
                    slots += 1;
                    break;
                }
                i += 1;
            }
            slot += 1;
        }
        slots
    }

    /// Constraints [`emit_unread_input_pins`] emits: four per input cell that
    /// some mode does not read.
    pub(crate) const NUM_UNREAD_INPUT_PINS: usize = 4 * unread_input_slots();

    /// The first constraint index each arm places the unread-`IN` pins at.
    ///
    /// Each arm chooses where in its own numbering they land, so the one place
    /// that knows all three is here, next to the emitter. The controls read it
    /// to assert that a forged row's violated set IS the pins.
    #[cfg(test)]
    pub(crate) const fn unread_input_pin_base(kind: HasherKind) -> usize {
        match kind {
            HasherKind::Test => 17,
            HasherKind::Poseidon => poseidon_cols::NUM_CONSTRAINTS - NUM_UNREAD_INPUT_PINS,
            HasherKind::Blake3 => crate::lfm::blake3_socket::UNREAD_IDX,
        }
    }

    /// ★ **Pins the `IN` columns of every input cell a mode does not read.**
    ///
    /// **This is load-bearing on any arm whose constraints READ `IN`, and that
    /// is not something to decide per arm.** A mode that reads fewer cells than
    /// the layout provides leaves the rest receiving nothing from `LfmMem` —
    /// their multiplicity excludes it — so if anything then reads those columns
    /// they are four free felts of prover choice and the row's output stops
    /// being a function of its input.
    ///
    /// That is not hypothetical: it shipped. `MODE_L` reads one cell, the bus
    /// and the validator were both taught so, the BLAKE3 arm pinned the unread
    /// columns — and the `Test` and `Poseidon` arms, whose round 0 reads
    /// `A_i = IN_i` for `i < 8`, were not. Under those two a leaf row carried
    /// four unconstrained felts that the permutation consumed, which is a
    /// Fiat–Shamir break for any program that absorbs data. Deriving the pins
    /// from [`HashMode::num_input_cells`] here, once, is what stops the next
    /// mode repeating it: an arm cannot forget a pin it does not write.
    ///
    /// Degree 2 (a selector sum times a column), so no arm's bound moves.
    ///
    /// Returns the next free constraint index.
    pub(crate) fn emit_unread_input_pins<B: ConstraintBuilder<F, E>>(
        b: &mut B,
        first_idx: usize,
    ) -> usize {
        let mut idx = first_idx;
        // Cell 0 is read by every mode, so it is never pinned. A slot EVERY mode
        // reads is skipped rather than pinned to nothing — which is the shape
        // slot 1 took when the leaf RATE gave `Leaf` a second input cell.
        for slot in 1..3usize {
            let Some(sel) = MODE_SELECTORS
                .iter()
                .filter(|(_, mode)| mode.num_input_cells() <= slot)
                .fold(None::<B::Expr>, |acc, (col, _)| {
                    let term = b.main(0, *col);
                    Some(match acc {
                        None => term,
                        Some(a) => a + term,
                    })
                })
            else {
                continue;
            };
            for j in 0..4 {
                let in_col = b.main(0, cols::IN0 + 4 * slot + j);
                b.emit_base(idx, sel.clone() * in_col);
                idx += 1;
            }
        }
        debug_assert_eq!(idx - first_idx, NUM_UNREAD_INPUT_PINS);
        idx
    }

    /// The permutation the chip proves, chosen at construction.
    ///
    /// One struct with a runtime discriminant rather than one type per hasher:
    /// `LfmAirs` holds `LfmAir<HashConstraints>` as a single field, so a
    /// per-hasher type would force a trait object or an enum there instead.
    pub struct HashConstraints {
        pub kind: HasherKind,
    }

    impl HashConstraints {
        /// The `TestPermutation` configuration — the machine's pre-decision
        /// default. `HashConstraints::default()` is the same thing.
        pub const TEST: Self = Self {
            kind: HasherKind::Test,
        };

        /// The Poseidon-original configuration.
        pub const POSEIDON: Self = Self {
            kind: HasherKind::Poseidon,
        };

        /// The BLAKE3 2-to-1 compress configuration.
        pub const BLAKE3: Self = Self {
            kind: HasherKind::Blake3,
        };

        /// Constraints emitted under `kind` — the count the framework's
        /// dense-index invariant requires `eval` to fill exactly.
        pub const fn num_constraints(kind: HasherKind) -> usize {
            match kind {
                HasherKind::Test => 17 + NUM_UNREAD_INPUT_PINS,
                HasherKind::Poseidon => poseidon_cols::NUM_CONSTRAINTS,
                HasherKind::Blake3 => crate::lfm::blake3_socket::NUM_CONSTRAINTS,
            }
        }
    }

    impl Default for HashConstraints {
        fn default() -> Self {
            Self::TEST
        }
    }

    impl ConstraintSet<F, E> for HashConstraints {
        fn max_degree(&self) -> usize {
            3
        }

        fn eval<B: ConstraintBuilder<F, E>>(&self, b: &mut B) {
            match self.kind {
                HasherKind::Test => Self::eval_test(b),
                HasherKind::Poseidon => Self::eval_poseidon(b),
                // The BLAKE3 arm lives in its own module: it shares the mixing
                // dataflow with `blake3_chip` rather than with anything here,
                // and putting it beside its column layout, its senders and its
                // trace filler is what keeps the four in step.
                HasherKind::Blake3 => crate::lfm::blake3_socket::eval(b),
            }
        }
    }

    impl HashConstraints {
        fn eval_test<B: ConstraintBuilder<F, E>>(b: &mut B) {
            let mode_c = b.main(0, cols::MODE_C);
            let mode_t = b.main(0, cols::MODE_T);
            let mode_l = b.main(0, cols::MODE_L);
            let mode_p = b.main(0, cols::MODE_P);

            // idx 0–3: capacity-state copy —
            // S_i = MODE_P·IN_i + (MODE_C + MODE_T + MODE_L)·IV_i. Transcript
            // and leaf rows are one-cell-out steps like a compress row, so they
            // take the same capacity; `TestPermutation` has one hash domain and
            // is field-native, so all three compute the same function (see
            // `LfmHasher::transcript_out` / `leaf_out` and their recorded
            // weakening).
            for (k, iv_raw) in TestPermutation::compress_iv_raw().into_iter().enumerate() {
                let s = b.main(0, cols::S8 + k);
                let in_i = b.main(0, cols::IN0 + 8 + k);
                let iv_i = b.const_base(iv_raw);
                let m = mode_c.clone() + mode_t.clone() + mode_l.clone();
                b.emit_base(k, s - (mode_p.clone() * in_i + m * iv_i));
            }

            // idx 4–15: the TestPermutation round — t_i = (A_i + rc_i·m)³
            // with A_i = IN_i (i < 8) or S_i (i ≥ 8) and m the mode sum;
            // OUT_j = t_j + Σ_i t_i (mixing matrix M = I + J). The round
            // constant is scaled by the mode sum so zero-filled padding rows
            // satisfy the constraint (0 = 0) without a degree-4 gate: on real
            // rows m = 1 and the permutation is unchanged.
            // NON-CRYPTOGRAPHIC — this block behind the bus contract above is
            // the hash-swap surface.
            let t: Vec<B::Expr> = (0..12)
                .map(|i| {
                    let a = if i < 8 {
                        b.main(0, cols::IN0 + i)
                    } else {
                        b.main(0, cols::S8 + (i - 8))
                    };
                    let rc = b.const_base(canonical_u64(&TestPermutation::round_constant(i)));
                    let m = b.main(0, cols::MODE_C)
                        + b.main(0, cols::MODE_T)
                        + b.main(0, cols::MODE_L)
                        + b.main(0, cols::MODE_P);
                    let x = a + rc * m;
                    x.clone() * x.clone() * x
                })
                .collect();
            let sum = t[1..].iter().fold(t[0].clone(), |acc, ti| acc + ti.clone());
            for (j, tj) in t.into_iter().enumerate() {
                let out = b.main(0, cols::OUT0 + j);
                b.emit_base(4 + j, out - (tj + sum.clone()));
            }

            // idx 16: mode sum-boolean (exactly-one-of is the registrar's).
            let mode_sum = mode_c + mode_t + mode_l + mode_p;
            let one = b.one();
            b.emit_base(16, mode_sum.clone() * (one - mode_sum));

            // idx 17–24: the unread input cells. ★ REQUIRED HERE, because the
            // round above reads `IN_i` for every `i < 8` — including the four a
            // leaf row does not read. See `emit_unread_input_pins`.
            emit_unread_input_pins(b, 17);
        }

        /// Poseidon-original at width 12: 30 rounds of `x ↦ x⁷` (all lanes on
        /// the 8 full rounds, lane 0 only on the 22 partial ones) followed by
        /// the circulant MDS.
        ///
        /// **Degree is exactly 3, by construction.** `x⁷` is lowered as
        /// `(x³)²·x` over the witnessed `x²`/`x³` columns, so the MDS output
        /// constraint — the highest-degree one — is `column² · (degree-1
        /// expression)`. That keeps `max_degree() = 3` and leaves the wrap's
        /// blowup 2 untouched, which is the whole reason the S-box is
        /// decomposed instead of written `a⁷`.
        ///
        /// **The round constant is scaled by the mode sum, and that is
        /// load-bearing.** With `m = MODE_C + MODE_P = 0` a zero-filled padding
        /// row gives `a = 0`, hence `x² = x³ = 0` and `out = MDS·0 = 0`,
        /// inductively through all 30 rounds — so padding satisfies every
        /// constraint without a degree-4 `IS_REAL` gate anywhere. On a real row
        /// `m = 1` and the permutation is unchanged.
        fn eval_poseidon<B: ConstraintBuilder<F, E>>(b: &mut B) {
            use crate::lfm::poseidon::{MDS_CIRC_ROW, ROUND_CONSTANTS, sboxed_lanes};
            use poseidon_cols as pc;

            let mode_c = b.main(0, pc::MODE_C);
            let mode_t = b.main(0, pc::MODE_T);
            let mode_l = b.main(0, pc::MODE_L);
            let mode_p = b.main(0, pc::MODE_P);
            let m = mode_c + mode_t + mode_l + mode_p.clone();

            // idx 0–3: capacity-state copy — S_i = MODE_P·IN_i.
            //
            // Poseidon's `compress_iv` is ZERO (plain sponge compression, no
            // domain separation invented here), so the `MODE_C·IV_i` term the
            // TestPermutation version carries vanishes: on a compress row
            // MODE_P = 0 forces S_i = 0, which IS the IV. Transcript and leaf
            // rows are the same shape and take the same zero capacity —
            // Poseidon has one domain here, so it separates none of them.
            for k in 0..4 {
                let s = b.main(0, pc::S8 + k);
                let in_i = b.main(0, pc::IN0 + 8 + k);
                b.emit_base(k, s - mode_p.clone() * in_i);
            }

            // idx 4: mode sum-boolean (exactly-one-of is the registrar's).
            let one = b.one();
            b.emit_base(4, m.clone() * (one - m.clone()));

            let mut idx = 5;
            for (r, rc_row) in ROUND_CONSTANTS.iter().enumerate() {
                let sboxed = sboxed_lanes(r);

                // a_i = state_i + rc[r][i]·m, degree 1. Round 0 reads IN/S;
                // later rounds read the previous round's MDS output.
                let a: Vec<B::Expr> = rc_row
                    .iter()
                    .enumerate()
                    .map(|(i, rc_i)| {
                        let state = if r == 0 {
                            if i < 8 {
                                b.main(0, pc::IN0 + i)
                            } else {
                                b.main(0, pc::S8 + (i - 8))
                            }
                        } else {
                            b.main(0, pc::out(r - 1, i))
                        };
                        let rc = b.const_base(*rc_i);
                        state + rc * m.clone()
                    })
                    .collect();

                // The two S-box steps per S-boxed lane, both degree 2.
                for (lane, a_lane) in a.iter().enumerate().take(sboxed) {
                    let x2 = b.main(0, pc::x2(r, lane));
                    let x3 = b.main(0, pc::x3(r, lane));
                    b.emit_base(idx, x2.clone() - a_lane.clone() * a_lane.clone());
                    b.emit_base(idx + 1, x3 - x2 * a_lane.clone());
                    idx += 2;
                }

                // What enters the MDS: a^7 = (x³)²·a on S-boxed lanes
                // (degree 3), the bare post-constant lane otherwise.
                let f: Vec<B::Expr> = (0..HASH_STATE_FELTS)
                    .map(|i| {
                        if i < sboxed {
                            let x3 = b.main(0, pc::x3(r, i));
                            x3.clone() * x3 * a[i].clone()
                        } else {
                            a[i].clone()
                        }
                    })
                    .collect();

                // out_o = Σ_i MDS_CIRC_ROW[(i − o) mod 12] · f_i — the same
                // orientation `poseidon::PoseidonGoldilocks::mds` uses, and one
                // of the three conventions the external KAT pins.
                for o in 0..HASH_STATE_FELTS {
                    let acc = f
                        .iter()
                        .enumerate()
                        .fold(None::<B::Expr>, |acc, (i, fi)| {
                            let c = b.const_base(
                                MDS_CIRC_ROW[(i + HASH_STATE_FELTS - o) % HASH_STATE_FELTS],
                            );
                            let term = c * fi.clone();
                            Some(match acc {
                                None => term,
                                Some(x) => x + term,
                            })
                        })
                        .expect("twelve lanes");
                    let out = b.main(0, pc::out(r, o));
                    b.emit_base(idx, out - acc);
                    idx += 1;
                }
            }
            idx = emit_unread_input_pins(b, idx);
            debug_assert_eq!(
                idx,
                poseidon_cols::NUM_CONSTRAINTS,
                "every declared constraint index must be emitted exactly once"
            );
        }
    }
}

// =========================================================================
// LFM_KECCAK — the keccak-f[1600] adapter
// =========================================================================
//
// Replaces the production `KECCAK` core chip, which is VM-coupled (it moves
// the state through timestamped `MEMW` tokens) and therefore unusable here.
// This chip owns exactly the core's two `Keccak` bus tokens and binds them to
// `LfmMem` words instead of memory. The permutation itself is proved by the
// UNCHANGED production `KECCAK_RND` / `KECCAK_RC` / `BITWISE` AIRs — see
// `keccak_adapter` and `keccak_probe`, which pin that contract standalone.
//
// CONSTRAINTS: none, and none are needed for the 400 state byte columns.
// Byte-ness is transitive: every IN byte is an operand of a `BYTE_ALU[XOR]`
// lookup in the round chip's θ column-parity chain (which covers all 25 lanes)
// and again in θ-final, and every OUT byte is the *result* of a `BYTE_ALU[XOR]`
// lookup (χ, or ι for lane 0). BYTE_ALU tokens carry the result as a tuple
// element, so a non-byte value finds no row in the 2^20 BITWISE table and the
// bus cannot balance. That in turn makes each `u32` half — a fixed linear
// combination of four such bytes — free of any separate range check: four
// values below 2^8 with coefficients 1, 2^8, 2^16, 2^24 cannot reach 2^32.

pub mod keccak {
    use super::*;
    use crate::lfm::layout::keccak::{
        BLOCK_HALVES, BLOCK_WORDS, NUM_HALVES, NUM_WORDS, RATE_BYTES, RATE_LANES,
    };
    use crate::tables::types::alu_op;

    pub mod cols {
        pub use crate::lfm::layout::keccak::*;
        /// The state as received from memory, 200 byte columns, lane-major:
        /// `STATE + lane * 8 + b`.
        pub const STATE: usize = PREP_WIDTH; // 56
        /// The rate block as received, 136 byte columns. Block byte `k` is byte
        /// `k % 8` of lane `k / 8` — rate bytes are lane-major and
        /// little-endian within a lane, exactly like the state columns, so
        /// block byte `k` pairs with state byte `k`. (The column-major traversal
        /// that bites elsewhere is a property of the *token element order*, not
        /// of this column layout — see `keccak_token`.)
        pub const BLOCK: usize = STATE + 200; // 256
        /// What enters the permutation: `STATE ⊕ BLOCK` over the rate region on
        /// absorb rows, `STATE` everywhere else.
        pub const PERM_IN: usize = BLOCK + RATE_BYTES; // 392
        /// The permuted state, 200 byte columns.
        pub const OUT: usize = PERM_IN + 200; // 592
        pub const NUM_COLUMNS: usize = OUT + 200; // 792

        pub const fn state_byte(lane: usize, b: usize) -> usize {
            STATE + lane * 8 + b
        }
        pub const fn perm_in_byte(lane: usize, b: usize) -> usize {
            PERM_IN + lane * 8 + b
        }
        pub const fn out_byte(lane: usize, b: usize) -> usize {
            OUT + lane * 8 + b
        }
    }

    /// The row's is-real flag: exactly one mode on a real row, neither on
    /// padding.
    fn is_real() -> Multiplicity {
        Multiplicity::Sum(cols::MODE_PERM, cols::MODE_ABSORB)
    }

    /// Half `h` of the byte family at `bytes_start`, recomposed from its four
    /// byte columns as `Σ byte_k · 256^k`.
    ///
    /// This is the trick the dropped core chip used to rebuild addresses from
    /// byte columns (`tables/keccak.rs`): the machine-side value never gets its
    /// own column, so there is nothing extra to keep consistent. Half slots at
    /// or above `num_halves` are the family's unused top lanes and become tuple
    /// constants — a nonzero value there cannot balance.
    ///
    /// Note `(h / 2) * 8 + 4 * (h % 2) == 4 * h`; the long form is kept because
    /// it names why: half `h` is the low or high 4 bytes of lane `h / 2`.
    fn half_value(bytes_start: usize, h: usize, num_halves: usize) -> BusValue {
        if h >= num_halves {
            return zero();
        }
        let byte0 = bytes_start + (h / 2) * 8 + 4 * (h % 2);
        BusValue::Linear(
            (0..4)
                .map(|k| LinearTerm::ColumnUnsigned {
                    coefficient: 1u64 << (8 * k),
                    column: byte0 + k,
                })
                .collect(),
        )
    }

    /// An `LfmMem` token for word `word` of a byte family: `(addr, h0..h3)`.
    fn word_token_from_bytes(
        addr_col: usize,
        bytes_start: usize,
        word: usize,
        num_halves: usize,
    ) -> Vec<BusValue> {
        let mut v = vec![direct(addr_col)];
        v.extend((0..4).map(|l| half_value(bytes_start, 4 * word + l, num_halves)));
        v
    }

    /// Half `h` of the byte-reversed digest: reversed byte `j` is digest byte
    /// `31 − j`, so this half's bytes are `OUT[31 − 4h − k]` for `k = 0..3` with
    /// the usual little-endian coefficients. Both the byte order WITHIN a half
    /// and the order OF the halves come out reversed, which is exactly what
    /// reversing all 32 bytes means.
    fn reversed_half_value(h: usize) -> BusValue {
        BusValue::Linear(
            (0..4)
                .map(|k| LinearTerm::ColumnUnsigned {
                    coefficient: 1u64 << (8 * k),
                    column: cols::OUT + 31 - 4 * h - k,
                })
                .collect(),
        )
    }

    /// An `LfmMem` token for word `w` of the reversed digest.
    fn reversed_digest_token(addr_col: usize, w: usize) -> Vec<BusValue> {
        let mut v = vec![direct(addr_col)];
        v.extend((0..4).map(|l| reversed_half_value(4 * w + l)));
        v
    }

    /// A `Keccak` bus token: `(tag_lo, tag_hi, round, state[200])`.
    ///
    /// The 200 state elements are traversed **column-major over lanes** —
    /// element `3 + 8·(5x + y) + b` is byte `b` of lane `x + 5y`, so lanes come
    /// in the order 0, 5, 10, 15, 20, 1, 6, … That asymmetry is inherited from
    /// the production sender's `for x { for y { … } }` loop over a
    /// `(x + 5y)·8 + b` column formula; emitting them in natural order instead
    /// leaves the bus unbalanced (falsification-verified in R1a).
    #[allow(clippy::needless_range_loop)]
    fn keccak_token(round: u64, bytes_start: usize) -> Vec<BusValue> {
        let mut values = vec![
            direct(cols::TAG_LO),
            direct(cols::TAG_HI),
            BusValue::constant(round),
        ];
        for x in 0..5 {
            for y in 0..5 {
                for b in 0..8 {
                    values.push(direct(bytes_start + (x + 5 * y) * 8 + b));
                }
            }
        }
        values
    }

    pub fn bus_interactions() -> Vec<BusInteraction> {
        let mut interactions = Vec::with_capacity(2 * NUM_WORDS + BLOCK_WORDS + RATE_BYTES + 2);
        // Reads: the 13 state words.
        for j in 0..NUM_WORDS {
            interactions.push(BusInteraction::receiver(
                BusId::LfmMem,
                is_real(),
                word_token_from_bytes(cols::in_addr(j), cols::STATE, j, NUM_HALVES),
            ));
        }
        // Reads: the 9 rate-block words — absorb rows only, so on a permute row
        // the BLOCK columns are read by nothing (no token, no lookup) and are
        // simply dead witness.
        for j in 0..BLOCK_WORDS {
            interactions.push(BusInteraction::receiver(
                BusId::LfmMem,
                Multiplicity::Column(cols::MODE_ABSORB),
                word_token_from_bytes(cols::block_addr(j), cols::BLOCK, j, BLOCK_HALVES),
            ));
        }
        // Writes: the 13 output words, each with its own read count.
        for j in 0..NUM_WORDS {
            interactions.push(BusInteraction::sender(
                BusId::LfmMem,
                Multiplicity::Column(cols::mult(j)),
                word_token_from_bytes(cols::out_addr(j), cols::OUT, j, NUM_HALVES),
            ));
        }
        // The absorb XOR, one BITWISE lookup per rate byte:
        // `PERM_IN[k] = STATE[k] ⊕ BLOCK[k]`.
        for k in 0..RATE_BYTES {
            interactions.push(BusInteraction::sender(
                BusId::ByteAlu,
                Multiplicity::Column(cols::MODE_ABSORB),
                vec![
                    BusValue::constant(alu_op::XOR as u64),
                    direct(cols::STATE + k),
                    direct(cols::BLOCK + k),
                    direct(cols::PERM_IN + k),
                ],
            ));
        }
        // The reversed digest: the first 32 output bytes read back-to-front,
        // as two words. This is the production transcript's `sample()` — it
        // finalizes, reverses the digest in place, absorbs the reversed bytes
        // and returns them, so one value serves as both the challenge and the
        // next segment's prefix.
        //
        // Reversal is FREE at the recomposition boundary: the bus already
        // rebuilds each `u32` half as a linear combination of four byte
        // columns, so flipping the coefficient order (and the half order) is a
        // different Linear over the SAME columns — no new value columns, no
        // BitDec, no extra permutation. Rows that need no reversed digest leave
        // `REV_MULT` at zero and these two sends are inert.
        for w in 0..cols::DIGEST_WORDS {
            interactions.push(BusInteraction::sender(
                BusId::LfmMem,
                Multiplicity::Column(cols::rev_mult(w)),
                reversed_digest_token(cols::rev_addr(w), w),
            ));
        }
        // The request/reply pair that drives the production keccak family.
        interactions.push(BusInteraction::sender(
            BusId::Keccak,
            is_real(),
            keccak_token(0, cols::PERM_IN),
        ));
        interactions.push(BusInteraction::receiver(
            BusId::Keccak,
            is_real(),
            keccak_token(24, cols::OUT),
        ));
        interactions
    }

    pub struct KeccakAdapterConstraints;

    impl ConstraintSet<F, E> for KeccakAdapterConstraints {
        fn max_degree(&self) -> usize {
            2
        }

        fn eval<B: ConstraintBuilder<F, E>>(&self, b: &mut B) {
            // idx 0..63: the capacity region never absorbs, in either mode —
            // `PERM_IN = STATE` for lanes 17..24, ungated (and trivially true on
            // zero-filled padding rows).
            for i in 0..(25 - RATE_LANES) * 8 {
                let k = RATE_BYTES + i;
                let s = b.main(0, cols::STATE + k);
                let p = b.main(0, cols::PERM_IN + k);
                b.emit_base(i, p - s);
            }
            // idx 64..199: on a permute row nothing is absorbed, so the rate
            // region passes through too. On an absorb row this is gated off and
            // the BYTE_ALU[XOR] lookups above pin PERM_IN instead. Without this,
            // a permute row could feed the family a state unrelated to the one
            // it read from memory.
            let base = (25 - RATE_LANES) * 8;
            for k in 0..RATE_BYTES {
                let mode_perm = b.main(0, cols::MODE_PERM);
                let s = b.main(0, cols::STATE + k);
                let p = b.main(0, cols::PERM_IN + k);
                b.emit_base(base + k, mode_perm * (p - s));
            }
            // idx 200: mode sum-boolean (exactly-one-of is the registrar's).
            let sum = b.main(0, cols::MODE_PERM) + b.main(0, cols::MODE_ABSORB);
            let one = b.one();
            b.emit_base(base + RATE_BYTES, sum.clone() * (one - sum));
        }
    }
}

// =========================================================================
// LFM_LANES — word ↔ lane conversion (Pack / Unpack)
// =========================================================================
//
// Discovered as a real ISA gap in Milestone C: challenges are squeezed from
// the sponge as *cells*, but the ALU consumes base/ext operands, and no
// composition of the original eight ops can cross that boundary. The chip
// has NO constraints — the shared value columns appearing in both the word
// token and the four lane tokens IS the semantics.

pub mod lanes {
    use super::*;

    pub mod cols {
        pub use crate::lfm::layout::lanes::*;
        pub const V0: usize = PREP_WIDTH; // ..V3
        pub const NUM_COLUMNS: usize = PREP_WIDTH + 4;
    }

    pub fn bus_interactions() -> Vec<BusInteraction> {
        let mut interactions = vec![
            // Pack rows write the assembled word; Unpack rows read one.
            BusInteraction::sender(
                BusId::LfmMem,
                Multiplicity::Column(cols::WORD_MULT),
                word_token(cols::WORD_ADDR, cols::V0),
            ),
            BusInteraction::receiver(
                BusId::LfmMem,
                Multiplicity::Column(cols::MODE_UNPACK),
                word_token(cols::WORD_ADDR, cols::V0),
            ),
        ];
        for i in 0..4 {
            // Unpack rows write the four lanes; Pack rows read them.
            interactions.push(BusInteraction::sender(
                BusId::LfmMem,
                Multiplicity::Column(cols::LANE_MULT0 + i),
                base_token(cols::LANE_ADDR0 + i, cols::V0 + i),
            ));
            interactions.push(BusInteraction::receiver(
                BusId::LfmMem,
                Multiplicity::Column(cols::MODE_PACK),
                base_token(cols::LANE_ADDR0 + i, cols::V0 + i),
            ));
        }
        interactions
    }
}

// =========================================================================
// LFM_HINT — arena ingestion (values unconstrained BY DESIGN; arena rule)
// =========================================================================

pub mod hint {
    use super::*;

    pub mod cols {
        pub use crate::lfm::layout::hint::*;
        pub const V0: usize = PREP_WIDTH; // ..V3
        pub const NUM_COLUMNS: usize = PREP_WIDTH + 4;
    }

    pub fn bus_interactions() -> Vec<BusInteraction> {
        vec![BusInteraction::sender(
            BusId::LfmMem,
            Multiplicity::Column(cols::MULT),
            word_token(cols::OUT_ADDR, cols::V0),
        )]
    }
}

// =========================================================================
// LFM_PUBLIC — attestation output (COMMIT-bus closure pattern)
// =========================================================================

pub mod public {
    use super::*;

    pub mod cols {
        pub use crate::lfm::layout::public::*;
        pub const V0: usize = PREP_WIDTH; // ..V3
        pub const NUM_COLUMNS: usize = PREP_WIDTH + 4;
    }

    pub fn bus_interactions() -> Vec<BusInteraction> {
        let mut send = vec![direct(cols::INDEX)];
        send.extend(word(cols::V0));
        vec![
            BusInteraction::receiver(
                BusId::LfmMem,
                Multiplicity::Column(cols::IS_REAL),
                word_token(cols::IN_ADDR, cols::V0),
            ),
            BusInteraction::sender(BusId::LfmPublic, Multiplicity::Column(cols::IS_REAL), send),
        ]
    }
}

// =========================================================================
// LFM_RANGE — fixed 2^16 lookup table (idle in v0; the future hash chip's
// byte/limb tables land here)
// =========================================================================

pub mod range {
    use super::*;

    pub mod cols {
        pub use crate::lfm::layout::range::*;
        pub const MU: usize = PREP_WIDTH;
        pub const NUM_COLUMNS: usize = PREP_WIDTH + 1;
    }

    pub fn bus_interactions() -> Vec<BusInteraction> {
        vec![BusInteraction::receiver(
            BusId::LfmRange,
            Multiplicity::Column(cols::MU),
            vec![direct(cols::VALUE)],
        )]
    }
}
