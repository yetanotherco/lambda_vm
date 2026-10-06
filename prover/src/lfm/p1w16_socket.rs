//! The width-16 `LFM_HASH` socket: ZisK's Poseidon1 permutation
//! ([`crypto::hash::poseidon1_w16`]) behind a four-cell-in, four-cell-out
//! memory contract.
//!
//! The machine's socket today is twelve felts (three cells), which a width-16
//! permutation does not fit. ZisK's constructions over this permutation need no
//! more than one shape:
//!
//! | construction | input cells | output cells read |
//! |---|---|---|
//! | 4-ary node (`compress4`) | the four children | cell 0 (the digest) |
//! | leaf block (`linear_hash`) | three rate cells + the carried digest (zero first) | cell 0 |
//! | transcript update | three rate cells + the previous state's cell 0 | all four (every lane is squeezed) |
//!
//! So the socket has ONE mode: a row reads four cells, permutes all sixteen
//! lanes, and offers each output cell to memory with its own multiplicity
//! (zero for an output nobody reads). No lane is a constant the chip injects —
//! ZisK's instance has no capacity IV and no domain tag — so there are no
//! capacity-copy columns and no unread input slots to pin.
//!
//! # Columns
//!
//! Preprocessed ([`cols::PREP_WIDTH`] = 13, RPX's width): four input and four
//! output addresses, `IS_REAL`, four output multiplicities.
//! Value: `IN0..16`, then per round the compact S-box witness — `x³ = a³` and
//! `x⁷ = (x³)²·a` for every S-boxed lane, both degree 3 — with every round's
//! S-box input carried as an affine form in the earlier `x⁷` columns and `IN`,
//! never committed. Under [`OutForm::Columns`] the sixteen output lanes are
//! committed as `OUT0..16`; under [`OutForm::OnBus`] they are not columns at
//! all: the last round is a full round followed by the MDS, so each output
//! lane is a fixed small-integer combination of that round's sixteen `x⁷`
//! columns, and the output tokens carry exactly that combination.
//!
//! The round constants are scaled by `IS_REAL`, so an all-zero padding row
//! satisfies every constraint and emits no bus token; an output multiplicity
//! on a padding row is rejected.

use stark::config::Commitment;
use stark::constraints::builder::{ConstraintBuilder, ConstraintSet};
use stark::lookup::{
    AirWithBuses, AuxiliaryTraceBuildData, BusInteraction, BusValue, LinearTerm, Multiplicity,
    Packing,
};
use stark::proof::options::ProofOptions;

use crate::tables::types::{BusId, FE, GoldilocksExtension, GoldilocksField};
use crypto::hash::poseidon1_w16::{
    self as p1, NUM_ROUNDS, STATE_FELTS, constants::MDS_CIRC_ROW, constants::ROUND_CONSTANTS,
    is_full_round,
};
use math::field::traits::IsPrimeField;

use super::airs::LfmAir;

type F = GoldilocksField;
type E = GoldilocksExtension;

/// Felts per memory cell.
pub const CELL_FELTS: usize = 4;
/// Cells a row reads, and cells it may write.
pub const CELLS: usize = STATE_FELTS / CELL_FELTS;

/// Column offsets.
pub mod cols {
    use super::{CELLS, STATE_FELTS};

    /// First input address; cell `k` is read at `IN_ADDR0 + k`.
    pub const IN_ADDR0: usize = 0;
    /// First output address; cell `k` is written at `OUT_ADDR0 + k`.
    pub const OUT_ADDR0: usize = IN_ADDR0 + CELLS;
    /// 1 on a permutation row, 0 on padding: the inputs' read multiplicity.
    pub const IS_REAL: usize = OUT_ADDR0 + CELLS;
    /// First output multiplicity: how many reads cell `k`'s write serves.
    pub const MULT0: usize = IS_REAL + 1;
    /// Preprocessed columns (the instruction column group).
    pub const PREP_WIDTH: usize = MULT0 + CELLS;
    /// The sixteen input lanes, cell-major.
    pub const IN0: usize = PREP_WIDTH;
    /// The value prefix every [`super::OutForm`] shares.
    pub const PREFIX_END: usize = IN0 + STATE_FELTS;
}

// The socket shares the twelve-felt socket's instruction column group width, so
// the compiler's one `LFM_HASH` group serves both.
const _: () = assert!(cols::PREP_WIDTH == super::layout::hash::PREP_WIDTH);

/// Where the sixteen output lanes live.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutForm {
    /// Committed as sixteen `OUT` columns, each constrained to its affine form.
    Columns,
    /// Not committed: each output token lane is the last round's MDS
    /// combination of its sixteen `x⁷` columns.
    OnBus,
}

/// The form the machine wires (`HasherKind::Poseidon1W16`'s `LFM_HASH`): the
/// outputs on the bus. BIG 664: 1.042–1.049× RPX's prove time per
/// permutation, against 1.067–1.094× with the outputs in columns.
pub const SOCKET_FORM: OutForm = OutForm::OnBus;

/// S-boxed lanes in round `r`.
pub const fn sboxed_lanes(r: usize) -> usize {
    if is_full_round(r) { STATE_FELTS } else { 1 }
}

/// LogUp aux columns: the eight interactions in pairs.
pub const AUX_COLUMNS: usize = (2 * CELLS).div_ceil(2);
/// Base-field cells per aux column (a cubic-extension accumulator).
pub const CELLS_PER_AUX_COLUMN: usize = 3;

impl OutForm {
    /// First `OUT` column ([`OutForm::Columns`] only).
    pub const fn out0(self) -> usize {
        cols::PREFIX_END
    }

    /// First round-witness column.
    pub const fn rounds(self) -> usize {
        match self {
            OutForm::Columns => cols::PREFIX_END + STATE_FELTS,
            OutForm::OnBus => cols::PREFIX_END,
        }
    }

    /// First column of round `r`'s witness block (`2·sboxed_lanes(r)` wide).
    pub const fn block(self, r: usize) -> usize {
        let mut off = self.rounds();
        let mut i = 0;
        while i < r {
            off += 2 * sboxed_lanes(i);
            i += 1;
        }
        off
    }

    /// `x³` of `lane` in round `r`.
    pub const fn x3(self, r: usize, lane: usize) -> usize {
        self.block(r) + lane
    }

    /// `x⁷` of `lane` in round `r`.
    pub const fn x7(self, r: usize, lane: usize) -> usize {
        self.block(r) + sboxed_lanes(r) + lane
    }

    /// The chip's width, preprocessed prefix included.
    pub const fn num_columns(self) -> usize {
        self.block(NUM_ROUNDS)
    }

    /// Value columns: the census's `main_cols`.
    pub const fn value_columns(self) -> usize {
        self.num_columns() - cols::PREP_WIDTH
    }

    /// Base-field cells per permutation: value columns plus the aux columns.
    pub const fn cells_per_permutation(self) -> usize {
        self.value_columns() + CELLS_PER_AUX_COLUMN * AUX_COLUMNS
    }

    /// Constraints: `IS_REAL` booleanity, one per output multiplicity (zero
    /// unless the row is real), two per S-box, and under [`OutForm::Columns`]
    /// one per `OUT` lane.
    pub const fn num_constraints(self) -> usize {
        let mut n = 1 + CELLS;
        let mut r = 0;
        while r < NUM_ROUNDS {
            n += 2 * sboxed_lanes(r);
            r += 1;
        }
        match self {
            OutForm::Columns => n + STATE_FELTS,
            OutForm::OnBus => n,
        }
    }
}

/// The MDS entry `M[o][i]` of Plonky3's circulant.
const fn mds_entry(o: usize, i: usize) -> u64 {
    MDS_CIRC_ROW[(i + STATE_FELTS - o) % STATE_FELTS]
}

/// The socket's eight interactions: the four input cells received with
/// multiplicity `IS_REAL`, the four output cells sent with their own
/// multiplicities. The `LfmMem` token is `(addr, v0, v1, v2, v3)`.
pub fn bus_interactions(form: OutForm) -> Vec<BusInteraction> {
    let direct = |c: usize| BusValue::Packed {
        start_column: c,
        packing: Packing::Direct,
    };
    let mut v = Vec::with_capacity(2 * CELLS);
    for k in 0..CELLS {
        let mut token = vec![direct(cols::IN_ADDR0 + k)];
        token.extend((0..CELL_FELTS).map(|j| direct(cols::IN0 + CELL_FELTS * k + j)));
        v.push(BusInteraction::receiver(
            BusId::LfmMem,
            Multiplicity::Column(cols::IS_REAL),
            token,
        ));
    }
    for k in 0..CELLS {
        let mut token = vec![direct(cols::OUT_ADDR0 + k)];
        for j in 0..CELL_FELTS {
            let lane = CELL_FELTS * k + j;
            token.push(match form {
                OutForm::Columns => direct(form.out0() + lane),
                OutForm::OnBus => BusValue::Linear(
                    (0..STATE_FELTS)
                        .map(|i| LinearTerm::Column {
                            coefficient: mds_entry(lane, i) as i64,
                            column: form.x7(NUM_ROUNDS - 1, i),
                        })
                        .collect(),
                ),
            });
        }
        v.push(BusInteraction::sender(
            BusId::LfmMem,
            Multiplicity::Column(cols::MULT0 + k),
            token,
        ));
    }
    v
}

/// The socket's AIR under `options`, its instruction column group committed
/// at `root`.
pub fn air(
    form: OutForm,
    options: &ProofOptions,
    root: Commitment,
) -> LfmAir<P1W16SocketConstraints> {
    AirWithBuses::new(
        form.num_columns(),
        AuxiliaryTraceBuildData {
            interactions: bus_interactions(form),
        },
        options,
        1,
        P1W16SocketConstraints { form },
    )
    .with_name("LFM_HASH16")
    .with_preprocessed(root, cols::PREP_WIDTH)
}

/// Writes a permutation row's value columns for `input` (the preprocessed
/// prefix is the caller's) and returns the permutation's output, which is the
/// host reference's [`p1::permute`].
pub fn fill_row(form: OutForm, input: [FE; STATE_FELTS], row: &mut [FE]) -> [FE; STATE_FELTS] {
    row[cols::IN0..cols::IN0 + STATE_FELTS].copy_from_slice(&input);
    let mut s = input;
    for (r, rc) in ROUND_CONSTANTS.iter().enumerate() {
        let mut f: [FE; STATE_FELTS] = core::array::from_fn(|i| s[i] + FE::from(rc[i]));
        for (lane, a) in f.iter_mut().enumerate().take(sboxed_lanes(r)) {
            let x3 = *a * *a * *a;
            let x7 = x3 * x3 * *a;
            row[form.x3(r, lane)] = x3;
            row[form.x7(r, lane)] = x7;
            *a = x7;
        }
        s = p1::mds(&f);
    }
    if form == OutForm::Columns {
        row[form.out0()..form.out0() + STATE_FELTS].copy_from_slice(&s);
    }
    s
}

/// An affine form over the row: `Σ col[c]·main(c) + real·IS_REAL`.
#[derive(Clone)]
struct Affine {
    col: Vec<FE>,
    real: FE,
}

impl Affine {
    fn zero(width: usize) -> Self {
        Self {
            col: vec![FE::zero(); width],
            real: FE::zero(),
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
        self.real += k * other.real;
    }

    fn expr<B: ConstraintBuilder<F, E>>(&self, b: &B, is_real: &B::Expr) -> B::Expr {
        let mut acc = b.const_base(canon(&self.real)) * is_real.clone();
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

/// The socket's constraint set.
pub struct P1W16SocketConstraints {
    pub form: OutForm,
}

impl ConstraintSet<F, E> for P1W16SocketConstraints {
    fn max_degree(&self) -> usize {
        3
    }

    fn eval<B: ConstraintBuilder<F, E>>(&self, b: &mut B) {
        let form = self.form;
        let width = form.num_columns();
        let is_real = b.main(0, cols::IS_REAL);
        let one = b.one();
        let not_real = one - is_real.clone();
        b.emit_base(0, is_real.clone() * not_real.clone());
        // An output is offered only by a real row. A padding row reads nothing
        // (its `IN` is free) and computes the permutation without round
        // constants, so a write from one would put a prover-chosen value in
        // memory. The program builder never emits one; this makes it
        // unprovable rather than merely unemitted.
        for k in 0..CELLS {
            b.emit_base(1 + k, b.main(0, cols::MULT0 + k) * not_real.clone());
        }
        let mut idx = 1 + CELLS;

        let mut state: Vec<Affine> = (0..STATE_FELTS)
            .map(|i| Affine::column(width, cols::IN0 + i))
            .collect();
        for (r, rc) in ROUND_CONSTANTS.iter().enumerate() {
            // a_i = state_i + rc[r][i]·IS_REAL; S-boxed lanes become their x⁷.
            let mut f: Vec<Affine> = state
                .iter()
                .zip(rc)
                .map(|(s, c)| {
                    let mut a = s.clone();
                    a.real += FE::from(*c);
                    a
                })
                .collect();
            for (lane, a) in f.iter_mut().enumerate().take(sboxed_lanes(r)) {
                let a_e = a.expr(b, &is_real);
                let x3 = b.main(0, form.x3(r, lane));
                let x7 = b.main(0, form.x7(r, lane));
                b.emit_base(idx, x3.clone() - a_e.clone() * a_e.clone() * a_e.clone());
                b.emit_base(idx + 1, x7 - x3.clone() * x3 * a_e);
                idx += 2;
                *a = Affine::column(width, form.x7(r, lane));
            }
            state = (0..STATE_FELTS)
                .map(|o| {
                    let mut acc = Affine::zero(width);
                    for (i, fi) in f.iter().enumerate() {
                        acc.add_scaled(fi, FE::from(mds_entry(o, i)));
                    }
                    acc
                })
                .collect();
        }
        if form == OutForm::Columns {
            for (j, s) in state.iter().enumerate() {
                let out = b.main(0, form.out0() + j);
                b.emit_base(idx, out - s.expr(b, &is_real));
                idx += 1;
            }
        }
        debug_assert_eq!(idx, form.num_constraints());
    }
}
