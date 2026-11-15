//! ★ The lean BLAKE3 tables ([`super::blake3_lean`]) as tables of a W-LFM
//! proof: a program's [`Instr::B3l`](super::instr::Instr::B3l) rows proved on
//! the real `LfmMem` bus.
//!
//! A program with `n > 0` compressions adds, after its LFM chips' tables
//! (`LfmAirs::air_refs` order), in this order:
//!
//! 1. `B3L_ROUND` chunks: the `7n` round rows (row `7k + r` is round `r` of
//!    compression `k`) cut by [`super::blake3_lean::chunk_rows`] at
//!    2^[`CHUNK_LOG`] rows, each chunk a power of two;
//! 2. `B3L_HEAD` chunks: the `n` head rows, cut the same way;
//! 3. `B3L_XOR`, the fixed 2^16-row byte table every lookup closes against.
//!
//! A program with no compression adds nothing ([`B3lShape`] is empty), so
//! every W-LFM proof of today's programs keeps its tables, its groups, its
//! identity and its bytes.
//!
//! # What binds what
//!
//! Every preprocessed column of the three tables goes through the W-LFM
//! prepared stacks, as every LFM chip's does: the head's instruction group
//! (the program's `B3L_HEAD` group, [`super::layout::b3l`]), the round rows'
//! `MU, TAG, RND` (a function of the compression count, derived here), and the
//! byte table's five fixed columns (as `LFM_RANGE`'s fixed group is). The
//! chunks are cut by a constant of the format ([`CHUNK_LOG`]), so a table's
//! height is the verifier's, never a proof's.
//!
//! The chain between a head row and its seven round rows is the round bus's
//! `(tag, round)` tokens (`BusId::B3lRound`); rows meet only through them, so
//! a compression may straddle two round chunks.

use stark::constraints::builder::EmptyConstraints;
use stark::proof::options::ProofOptions;
use stark::trace::TraceTable;

use crate::tables::types::{FE, GoldilocksExtension, GoldilocksField, zeroed_fe_vec};

use super::airs::{DynLfmAir, LfmAir, build_lfm_air};
use super::b3l::B3lValues;
use super::blake3_lean::{
    self as lean, HeadConstraints, RoundConstraints, TableHist, head, round, table,
};
use super::chips::mux::{MuxConstraints, cols as mux};
use super::compiler::{ColumnGroup, LfmProgram};
use super::executor::MuxRow;
use super::layout::b3l as l;

type F = GoldilocksField;
type E = GoldilocksExtension;

/// The head and round tables' chunk cap, 2^19 rows (`KECCAK_RND`'s chunk cap
/// in the machine): a round chunk is 2^19 × 376 ≈ 197 M main cells, a head
/// chunk 2^19 × 200 ≈ 105 M, so no table outgrows a group.
pub const CHUNK_LOG: u32 = lean::MAX_CHUNK_LOG;

/// The smallest chunk a remainder is padded up to.
pub const MIN_CHUNK_LOG: u32 = lean::MIN_CHUNK_LOG;

pub const ROUND_NAME: &str = "B3L_ROUND";
pub const HEAD_NAME: &str = "B3L_HEAD";
pub const TABLE_NAME: &str = "B3L_XOR";
pub const MUX_NAME: &str = "LEAN_MUX";

/// The largest `LEAN_MUX` chunk, in rows: the LFM chips' cut (`wlfm`'s
/// `WLFM_CHUNK_LOG`).
pub const MUX_CHUNK_LOG: u32 = 22;

/// The lean tables' shape: the padded rows of every round chunk and every
/// head chunk, in table order. Empty for a program with no compression, which
/// then has no lean table at all, not even the byte table.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct B3lShape {
    pub rounds: Vec<usize>,
    pub heads: Vec<usize>,
    /// The `LEAN_MUX` chunks (one-output selects): only in a program with
    /// compressions, after the byte table.
    pub muxes: Vec<usize>,
}

impl B3lShape {
    /// The shape of `compressions` compressions and no one-output select.
    pub fn of(compressions: usize) -> Self {
        if compressions == 0 {
            return Self::default();
        }
        Self {
            rounds: lean::chunk_rows(l::ROUNDS * compressions, CHUNK_LOG, MIN_CHUNK_LOG),
            heads: lean::chunk_rows(compressions, CHUNK_LOG, MIN_CHUNK_LOG),
            muxes: Vec::new(),
        }
    }

    /// A program's.
    pub fn of_program(program: &LfmProgram) -> Self {
        let mut shape = Self::of(program.groups.b3l.real_rows);
        if !shape.is_empty() {
            shape.muxes =
                lean::chunk_rows(program.groups.mux.real_rows, MUX_CHUNK_LOG, MIN_CHUNK_LOG);
        }
        shape
    }

    pub fn is_empty(&self) -> bool {
        self.heads.is_empty()
    }

    /// The tables this shape adds: the round chunks, the head chunks, the
    /// byte table and the mux chunks; none when empty.
    pub fn num_tables(&self) -> usize {
        if self.is_empty() {
            0
        } else {
            self.rounds.len() + self.heads.len() + 1 + self.muxes.len()
        }
    }

    /// Each added table's height in variables, in table order.
    pub fn table_num_vars(&self) -> Vec<u8> {
        if self.is_empty() {
            return Vec::new();
        }
        self.rounds
            .iter()
            .chain(&self.heads)
            .map(|rows| rows.trailing_zeros() as u8)
            .chain([table::ROWS.trailing_zeros() as u8])
            .chain(self.muxes.iter().map(|rows| rows.trailing_zeros() as u8))
            .collect()
    }
}

/// The lean tables' AIRs for a shape, with NO STARK preprocessed roots: the
/// W-LFM prepared stacks bind every preprocessed column (as
/// `wlfm::whir_lfm_airs` builds the LFM chips'), so this set must never reach
/// a STARK prover or verifier.
pub struct B3lAirs {
    rounds: Vec<LfmAir<RoundConstraints>>,
    heads: Vec<LfmAir<HeadConstraints>>,
    table: Option<LfmAir<EmptyConstraints>>,
    muxes: Vec<LfmAir<MuxConstraints>>,
}

impl B3lAirs {
    pub fn new(shape: &B3lShape, options: &ProofOptions) -> Self {
        #[cfg(test)]
        let (rounds, heads) = test_drops::get();
        #[cfg(not(test))]
        let (rounds, heads) = (RoundConstraints::default(), HeadConstraints::default());
        Self::with_constraints(shape, options, rounds, heads)
    }

    /// [`Self::new`] with the round and head constraint sets given: a test's
    /// mutation (a dropped constraint group).
    pub(crate) fn with_constraints(
        shape: &B3lShape,
        options: &ProofOptions,
        rounds: RoundConstraints,
        heads: HeadConstraints,
    ) -> Self {
        let zero = [0u8; 32];
        Self {
            rounds: shape
                .rounds
                .iter()
                .map(|_| {
                    build_lfm_air(
                        round::NUM_COLUMNS,
                        lean::round_interactions(),
                        options,
                        rounds,
                        ROUND_NAME,
                        zero,
                        round::PREP_WIDTH,
                    )
                })
                .collect(),
            heads: shape
                .heads
                .iter()
                .map(|_| {
                    build_lfm_air(
                        head::NUM_COLUMNS,
                        lean::head_interactions(),
                        options,
                        heads,
                        HEAD_NAME,
                        zero,
                        head::PREP_WIDTH,
                    )
                })
                .collect(),
            table: (!shape.is_empty()).then(|| {
                build_lfm_air(
                    table::NUM_COLUMNS,
                    lean::table_interactions(),
                    options,
                    EmptyConstraints,
                    TABLE_NAME,
                    zero,
                    table::PREP_WIDTH,
                )
            }),
            muxes: shape
                .muxes
                .iter()
                .map(|_| {
                    build_lfm_air(
                        mux::NUM_COLUMNS,
                        super::chips::mux::bus_interactions(),
                        options,
                        mux_constraints(),
                        MUX_NAME,
                        zero,
                        mux::PREP_WIDTH,
                    )
                })
                .collect(),
        }
    }

    /// The AIRs in table order.
    pub fn refs(&self) -> Vec<DynLfmAir<'_>> {
        let mut v: Vec<DynLfmAir<'_>> =
            Vec::with_capacity(self.rounds.len() + self.heads.len() + 1);
        v.extend(self.rounds.iter().map(|a| a as DynLfmAir<'_>));
        v.extend(self.heads.iter().map(|a| a as DynLfmAir<'_>));
        if let Some(t) = &self.table {
            v.push(t);
        }
        v.extend(self.muxes.iter().map(|a| a as DynLfmAir<'_>));
        v
    }

    /// Each AIR with its table, in table order.
    pub fn pairs<'a>(
        &'a self,
        traces: &'a mut B3lTraces,
    ) -> Result<Vec<(DynLfmAir<'a>, &'a mut HeldTable)>, String> {
        if traces.rounds.len() != self.rounds.len()
            || traces.heads.len() != self.heads.len()
            || traces.muxes.len() != self.muxes.len()
        {
            return Err(format!(
                "the lean traces are {} round, {} head and {} mux chunks, the AIR set {}, {} \
                 and {}",
                traces.rounds.len(),
                traces.heads.len(),
                traces.muxes.len(),
                self.rounds.len(),
                self.heads.len(),
                self.muxes.len()
            ));
        }
        let Some(table_air) = &self.table else {
            return Err("lean traces for an AIR set with no lean table".into());
        };
        let mut v: Vec<(DynLfmAir<'a>, &'a mut HeldTable)> = Vec::new();
        for (air, t) in self.rounds.iter().zip(traces.rounds.iter_mut()) {
            v.push((air, t));
        }
        for (air, t) in self.heads.iter().zip(traces.heads.iter_mut()) {
            v.push((air, t));
        }
        v.push((table_air, &mut traces.table));
        for (air, t) in self.muxes.iter().zip(traces.muxes.iter_mut()) {
            v.push((air, t));
        }
        Ok(v)
    }
}

/// How the host holds a lean table until the card takes it: a prover's
/// choice, never the proof's (the block prover commits the same columns
/// either way, `Narrowing`'s rule).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum B3lHold {
    /// As field elements, eight bytes a cell.
    Wide,
    /// Packed chunk by chunk as it is filled, each column at the bytes its
    /// largest value needs ([`multilinear::narrow::NarrowColumns`]); the card
    /// widens it. The production holding: a round chunk is ≈ 1.2 bytes a cell
    /// instead of 8.
    Narrow,
    /// The round chunks as the compressions their rows come from
    /// ([`B3lRoundRecords`], 112 bytes a compression against ≈ 3.2 KB of
    /// packed rows): the card fills them as it uploads them, a host reader
    /// gets this module's narrow fill. The head chunks and the byte table are
    /// held narrow.
    Records,
}

/// `LAMBDA_VM_B3L_RECORDS=1`: the round chunks held as their records and
/// filled on the card ([`B3lHold::Records`]). Off by default; a prover's
/// choice, never the proof's. Read once a process.
pub const RECORDS_ENV: &str = "LAMBDA_VM_B3L_RECORDS";

/// The holding a fill takes: [`B3lHold::Narrow`], [`B3lHold::Records`] under
/// [`RECORDS_ENV`], or a test's override.
pub fn fill_hold() -> B3lHold {
    #[cfg(test)]
    if let Some(hold) = test_hold::get() {
        return hold;
    }
    static RECORDS: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *RECORDS.get_or_init(|| std::env::var(RECORDS_ENV).is_ok_and(|v| v.trim() == "1")) {
        B3lHold::Records
    } else {
        B3lHold::Narrow
    }
}

/// A lean round chunk held as the compressions that reach into it: rows
/// `[skip, skip + rows)` of the table they make, 7 rows a compression, rows
/// past the last zero. The card fills its columns from them
/// (`math_cuda::fill`, `fill_b3l_round`); a host reader gets
/// [`fill_round_rows`]'s narrow fill of the same rows.
#[derive(Debug)]
pub struct B3lRoundRecords {
    ops: Vec<multilinear::records::B3lRoundOp>,
    skip: usize,
    rows: usize,
}

impl multilinear::records::TableRecords for B3lRoundRecords {
    fn rows(&self) -> usize {
        self.rows
    }

    fn cols(&self) -> usize {
        round::NUM_COLUMNS
    }

    fn fill(&self) -> multilinear::records::RecordsFill<'_> {
        multilinear::records::RecordsFill::B3lRound {
            ops: &self.ops,
            skip: self.skip,
        }
    }

    fn host_fill(&self) -> Option<multilinear::narrow::NarrowColumns> {
        let total = l::ROUNDS * self.ops.len();
        let (packed, _) = fill_narrow(self.rows, &round_widths(), |first, len, scratch, hists| {
            let lo = self.skip + first;
            let hi = (lo + len).min(total);
            fill_round_rows(lo, hi, scratch, hists, |k| {
                let op = &self.ops[k];
                (op_values(op), op.block_len, op.flags, op.tag)
            });
        })?;
        Some(packed)
    }
}

/// A record's chaining value and block, as the executor's values.
fn op_values(op: &multilinear::records::B3lRoundOp) -> B3lValues {
    let mut block = [0u8; 64];
    for (i, w) in op.m.iter().enumerate() {
        block[4 * i..4 * i + 4].copy_from_slice(&w.to_le_bytes());
    }
    B3lValues { h: op.h, block }
}

/// A fill worker's own buffers, kept across the blocks it fills: the byte
/// table's counts of the compressions it owns, a sink for the counts of those
/// it fills but does not own (their first round is another block's), and one
/// compression's round rows. One a worker, not one a block: a block of a few
/// dozen compressions must not zero and merge a 2^16-entry count of its own.
struct FillHists {
    counted: TableHist,
    skip: TableHist,
    rows: Vec<FE>,
}

impl FillHists {
    fn new() -> Self {
        FillHists {
            counted: TableHist::default(),
            skip: TableHist::default(),
            rows: zeroed_fe_vec(l::ROUNDS * round::NUM_COLUMNS),
        }
    }
}

/// Round rows `[lo, hi)` of the table compressions `op(k)` make (row
/// `7k + r` is round `r` of compression `k`), written into `scratch` from its
/// first row, every column of every row, and the scratch's rows past them
/// (the padding past the last compression) zeroed; each compression's lookups
/// counted into `hists.counted` only by the rows holding its first round.
fn fill_round_rows(
    lo: usize,
    hi: usize,
    scratch: &mut [FE],
    hists: &mut FillHists,
    op: impl Fn(usize) -> (B3lValues, u32, u32, u64),
) {
    let width = round::NUM_COLUMNS;
    let written = (hi.saturating_sub(lo) * width).min(scratch.len());
    scratch[written..].iter_mut().for_each(|v| *v = FE::zero());
    if lo >= hi {
        return;
    }
    let FillHists {
        counted: hist,
        skip,
        rows,
    } = hists;
    for k in lo / l::ROUNDS..hi.div_ceil(l::ROUNDS) {
        let (values, block_len, flags, tag) = op(k);
        rows.iter_mut().for_each(|v| *v = FE::zero());
        let counted = if (lo..hi).contains(&(l::ROUNDS * k)) {
            &mut *hist
        } else {
            &mut *skip
        };
        lean::fill_rounds(
            rows,
            tag,
            &values,
            block_len,
            flags,
            None,
            l::ROUNDS,
            counted,
        );
        for r in 0..l::ROUNDS {
            let global = l::ROUNDS * k + r;
            if (lo..hi).contains(&global) {
                let at = (global - lo) * width;
                scratch[at..at + width].copy_from_slice(&rows[r * width..(r + 1) * width]);
            }
        }
    }
}

/// One lean table as the host holds it.
pub enum HeldTable {
    Wide(Box<TraceTable<F, E>>),
    Narrow(multilinear::narrow::NarrowColumns),
    /// The rows' records ([`B3lRoundRecords`]): the card fills them.
    Records(multilinear::records::Records),
}

impl HeldTable {
    /// `rows` row-major rows of `width` cells, held as `hold` says.
    fn from_rows(data: Vec<FE>, width: usize, hold: B3lHold) -> Self {
        match hold {
            B3lHold::Wide => HeldTable::Wide(Box::new(TraceTable::new_main(data, width, 1))),
            B3lHold::Narrow | B3lHold::Records => {
                use math::field::traits::IsPrimeField;
                let words: Vec<u64> = data
                    .iter()
                    .map(|f| GoldilocksField::canonical(f.value()))
                    .collect();
                drop(data);
                HeldTable::Narrow(
                    multilinear::narrow::NarrowColumns::pack_row_major(&words, width)
                        .expect("whole rows"),
                )
            }
        }
    }

    pub fn rows(&self) -> usize {
        match self {
            HeldTable::Wide(t) => t.num_rows(),
            HeldTable::Narrow(p) => p.rows(),
            HeldTable::Records(r) => r.rows(),
        }
    }

    pub fn cols(&self) -> usize {
        match self {
            HeldTable::Wide(t) => t.main_table.width,
            HeldTable::Narrow(p) => p.cols(),
            HeldTable::Records(r) => r.cols(),
        }
    }

    /// Column `c` as field elements (a records table's from its host fill:
    /// slow, a reader's and a test's).
    pub fn column(&self, c: usize) -> Vec<FE> {
        match self {
            HeldTable::Wide(t) => (0..t.num_rows()).map(|r| *t.main_table.get(r, c)).collect(),
            HeldTable::Narrow(p) => p.column(c).into_iter().map(FE::from).collect(),
            HeldTable::Records(r) => r
                .host_fill()
                .map(|p| p.column(c).into_iter().map(FE::from).collect())
                .unwrap_or_default(),
        }
    }

    /// The cell at `(row, col)`.
    pub fn get(&self, row: usize, col: usize) -> FE {
        match self {
            HeldTable::Wide(t) => *t.main_table.get(row, col),
            HeldTable::Narrow(p) => FE::from(p.get(row, col)),
            HeldTable::Records(_) => self.column(col)[row],
        }
    }

    /// Whether the table is held as its records.
    pub fn is_records(&self) -> bool {
        matches!(self, HeldTable::Records(_))
    }

    /// The table moved out, an empty one left behind.
    pub fn take(&mut self) -> HeldTable {
        core::mem::replace(
            self,
            HeldTable::Wide(Box::new(TraceTable::new_main(Vec::new(), 1, 1))),
        )
    }

    /// A wide table's cells, for a test that corrupts one: `None` if narrow.
    #[cfg(test)]
    pub fn wide_mut(&mut self) -> Option<&mut TraceTable<F, E>> {
        match self {
            HeldTable::Wide(t) => Some(t.as_mut()),
            HeldTable::Narrow(_) | HeldTable::Records(_) => None,
        }
    }
}

/// A program's lean tables, filled.
pub struct B3lTraces {
    pub rounds: Vec<HeldTable>,
    pub heads: Vec<HeldTable>,
    pub table: HeldTable,
    /// The `LEAN_MUX` chunks ([`build_mux_tables`]).
    pub muxes: Vec<HeldTable>,
}

impl B3lTraces {
    /// The round chunk holding round-table row `i` (row `7k + r` is round `r`
    /// of compression `k`), and `i`'s row in it.
    pub fn round_at(&self, i: usize) -> Option<(&HeldTable, usize)> {
        let mut start = 0usize;
        for t in &self.rounds {
            let rows = t.rows();
            if i < start + rows {
                return Some((t, i - start));
            }
            start += rows;
        }
        None
    }

    /// Every table in table order.
    pub fn tables_mut(&mut self) -> impl Iterator<Item = &mut HeldTable> {
        self.rounds
            .iter_mut()
            .chain(self.heads.iter_mut())
            .chain(core::iter::once(&mut self.table))
            .chain(self.muxes.iter_mut())
    }
}

/// The round rows' instruction columns at round-table row `i` of `n`
/// compressions: `(MU, TAG, RND)`, zero past the last round.
fn round_prep(i: usize, n: usize) -> [u64; 3] {
    if i < l::ROUNDS * n {
        [1, l::tag_for_row(i / l::ROUNDS), (i % l::ROUNDS) as u64]
    } else {
        [0, 0, 0]
    }
}

/// Rows `[start, start + rows)` of a row-major `width`-wide matrix whose
/// first `real` rows are `data`'s, the rest zero.
fn chunk_of(data: &[FE], width: usize, real: usize, start: usize, rows: usize) -> ColumnGroup {
    let end = (start + rows).min(real.max(start));
    let mut out = data[start * width..end * width].to_vec();
    out.resize(rows * width, FE::zero());
    ColumnGroup {
        width,
        real_rows: end - start,
        padded_rows: rows,
        data: out,
    }
}

/// Every lean table's preprocessed prefix as the program commits it, in table
/// order: the round chunks' `MU, TAG, RND`, the head chunks' rows of the
/// program's `B3L_HEAD` group, the byte table's fixed five columns. Empty for
/// a program with no compression.
///
/// These are the SAME values [`build_b3l_traces`] writes into each table's
/// leading columns, so the prepared stack and the trace's prefix are one
/// object; the W-LFM prover checks it column by column before it commits.
pub fn b3l_prep_groups(program: &LfmProgram) -> Vec<ColumnGroup> {
    let shape = B3lShape::of_program(program);
    if shape.is_empty() {
        return Vec::new();
    }
    let n = program.groups.b3l.real_rows;
    let mut out = Vec::with_capacity(shape.num_tables());
    let mut start = 0usize;
    for &rows in &shape.rounds {
        let mut data = Vec::with_capacity(rows * round::PREP_WIDTH);
        for i in start..start + rows {
            data.extend(round_prep(i, n).map(FE::from));
        }
        let real = (l::ROUNDS * n).saturating_sub(start).min(rows);
        out.push(ColumnGroup {
            width: round::PREP_WIDTH,
            real_rows: real,
            padded_rows: rows,
            data,
        });
        start += rows;
    }
    let g = &program.groups.b3l;
    let mut start = 0usize;
    for &rows in &shape.heads {
        out.push(chunk_of(&g.data, g.width, g.real_rows, start, rows));
        start += rows;
    }
    let cols = lean::table_prep_columns();
    let mut data = Vec::with_capacity(table::ROWS * table::PREP_WIDTH);
    for r in 0..table::ROWS {
        data.extend(cols.iter().map(|c| c[r]));
    }
    out.push(ColumnGroup {
        width: table::PREP_WIDTH,
        real_rows: table::ROWS,
        padded_rows: table::ROWS,
        data,
    });
    let g = &program.groups.mux;
    let mut start = 0usize;
    for &rows in &shape.muxes {
        out.push(chunk_of(&g.data, g.width, g.real_rows, start, rows));
        start += rows;
    }
    out
}

/// The mux tables' constraint set: today's, or a test's mutant.
fn mux_constraints() -> MuxConstraints {
    #[cfg(test)]
    if let Some(c) = test_mux::get() {
        return c;
    }
    MuxConstraints::default()
}

/// A program's `LEAN_MUX` chunks from its mux records, in table order: each
/// row the program's instruction group, then the bit and the arms' and the
/// chosen word's lanes; the rows past the last instruction zero. Held wide.
pub fn build_mux_tables(program: &LfmProgram, records: &[MuxRow]) -> Vec<HeldTable> {
    let shape = B3lShape::of_program(program);
    let g = &program.groups.mux;
    assert_eq!(
        records.len(),
        g.real_rows,
        "one LEAN_MUX record a mux row: the executor and the compiler disagree"
    );
    let width = mux::NUM_COLUMNS;
    let mut start = 0usize;
    let mut out = Vec::with_capacity(shape.muxes.len());
    for &rows in &shape.muxes {
        let mut data = zeroed_fe_vec(rows * width);
        let fill = |(r, row): (usize, &mut [FE])| {
            let k = start + r;
            if k >= records.len() {
                return;
            }
            row[..mux::PREP_WIDTH].copy_from_slice(&g.data[k * g.width..(k + 1) * g.width]);
            let rec = &records[k];
            row[mux::BIT] = rec.bit;
            row[mux::INL0..mux::INL0 + 4].copy_from_slice(&rec.in_l);
            row[mux::INR0..mux::INR0 + 4].copy_from_slice(&rec.in_r);
            row[mux::OUT0..mux::OUT0 + 4].copy_from_slice(&rec.out);
        };
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            data.par_chunks_mut(width).enumerate().for_each(fill);
        }
        #[cfg(not(feature = "parallel"))]
        data.chunks_mut(width).enumerate().for_each(fill);
        out.push(HeldTable::from_rows(data, width, B3lHold::Wide));
        start += rows;
    }
    out
}

/// Fill a program's lean tables from its `B3L_HEAD` records (one a
/// compression, in program order; the executor's [`B3lValues`]), each chunk
/// filled and then held as [`fill_hold`] says. `None` for a program with no
/// compression.
///
/// A round chunk is filled from the compressions that reach into it, a
/// compression straddling two chunks filled for each; its lookups are counted
/// once, by the chunk that holds its first round.
pub fn build_b3l_traces(program: &LfmProgram, records: &[B3lValues]) -> Option<B3lTraces> {
    build_b3l_traces_held(program, records, fill_hold())
}

/// [`build_b3l_traces`] under `hold`.
pub fn build_b3l_traces_held(
    program: &LfmProgram,
    records: &[B3lValues],
    hold: B3lHold,
) -> Option<B3lTraces> {
    let g = &program.groups.b3l;
    let n = g.real_rows;
    assert_eq!(
        records.len(),
        n,
        "one B3L_HEAD record a compression: the executor and the compiler disagree"
    );
    if n == 0 {
        return None;
    }
    let shape = B3lShape::of(n);
    let consts = |k: usize| {
        use math::field::traits::IsPrimeField;
        let prep = &g.data[k * g.width..(k + 1) * g.width];
        let canon = |c: usize| GoldilocksField::canonical(prep[c].value()) as u32;
        (prep, canon(l::BLEN), canon(l::FLAGS))
    };
    // The card's path: the round chunks held as their records (the card fills
    // them), the heads and the byte table narrow.
    if hold == B3lHold::Records
        && let Some(traces) = fill_records(n, &shape, &consts, records)
    {
        return Some(traces);
    }
    // The production path: every chunk filled block by block straight into its
    // narrow columns, blocks in parallel. Falls back to the path below if a
    // value outgrows its column's width (never on an honest fill).
    if hold == B3lHold::Narrow
        && let Some(traces) = fill_narrow_direct(n, &shape, &consts, records)
    {
        return Some(traces);
    }
    let mut hist = TableHist::default();

    // The round chunks.
    let mut rounds = Vec::with_capacity(shape.rounds.len());
    let mut start = 0usize;
    for &size in &shape.rounds {
        let end = (start + size).min(l::ROUNDS * n);
        let mut data = zeroed_fe_vec(size * round::NUM_COLUMNS);
        if start < end {
            let (k_lo, k_hi) = (start / l::ROUNDS, end.div_ceil(l::ROUNDS));
            let fill = |k: usize, rows: &mut [FE], hist: &mut TableHist| {
                let (_, block_len, flags) = consts(k);
                let mut scratch = TableHist::default();
                // Count a compression's lookups in the chunk that holds its
                // first round only.
                let counted = if l::ROUNDS * k >= start {
                    hist
                } else {
                    &mut scratch
                };
                lean::fill_rounds(
                    rows,
                    l::tag_for_row(k),
                    &records[k],
                    block_len,
                    flags,
                    None,
                    l::ROUNDS,
                    counted,
                );
            };
            let mut buf = zeroed_fe_vec((k_hi - k_lo) * l::ROUNDS * round::NUM_COLUMNS);
            #[cfg(feature = "parallel")]
            let part = {
                use rayon::prelude::*;
                buf.par_chunks_mut(l::ROUNDS * round::NUM_COLUMNS)
                    .enumerate()
                    .fold(TableHist::default, |mut h, (i, rows)| {
                        fill(k_lo + i, rows, &mut h);
                        h
                    })
                    .reduce(TableHist::default, |a, b| a.merge(&b))
            };
            #[cfg(not(feature = "parallel"))]
            let part = {
                let mut h = TableHist::default();
                for (i, rows) in buf.chunks_mut(l::ROUNDS * round::NUM_COLUMNS).enumerate() {
                    fill(k_lo + i, rows, &mut h);
                }
                h
            };
            hist = hist.merge(&part);
            let skip = start - l::ROUNDS * k_lo;
            let rows = end - start;
            data[..rows * round::NUM_COLUMNS].copy_from_slice(
                &buf[skip * round::NUM_COLUMNS..(skip + rows) * round::NUM_COLUMNS],
            );
        }
        rounds.push(HeldTable::from_rows(data, round::NUM_COLUMNS, hold));
        start += size;
    }

    // The head chunks: one row a compression. Its final state is the round
    // chain's, recomputed (seven rounds on the host: the rows are not kept).
    let mut heads = Vec::with_capacity(shape.heads.len());
    let mut start = 0usize;
    for &size in &shape.heads {
        let end = (start + size).min(n);
        let mut data = zeroed_fe_vec(size * head::NUM_COLUMNS);
        let fill = |k: usize, hrow: &mut [FE], hist: &mut TableHist| {
            let (prep, block_len, flags) = consts(k);
            let fs = lean::final_state(&records[k], block_len, flags);
            lean::fill_head(hrow, prep, &records[k], &fs, hist);
        };
        let live = &mut data[..(end - start) * head::NUM_COLUMNS];
        #[cfg(feature = "parallel")]
        let part = {
            use rayon::prelude::*;
            live.par_chunks_mut(head::NUM_COLUMNS)
                .enumerate()
                .fold(TableHist::default, |mut h, (i, row)| {
                    fill(start + i, row, &mut h);
                    h
                })
                .reduce(TableHist::default, |a, b| a.merge(&b))
        };
        #[cfg(not(feature = "parallel"))]
        let part = {
            let mut h = TableHist::default();
            for (i, row) in live.chunks_mut(head::NUM_COLUMNS).enumerate() {
                fill(start + i, row, &mut h);
            }
            h
        };
        hist = hist.merge(&part);
        heads.push(HeldTable::from_rows(data, head::NUM_COLUMNS, hold));
        start += size;
    }

    let table = lean::table_trace(&hist);
    let table = match hold {
        B3lHold::Wide => HeldTable::Wide(Box::new(table)),
        B3lHold::Narrow | B3lHold::Records => {
            let width = table::NUM_COLUMNS;
            HeldTable::from_rows(table.main_table.row_major_data().to_vec(), width, hold)
        }
    };
    Some(B3lTraces {
        rounds,
        heads,
        table,
        muxes: Vec::new(),
    })
}

/// Rows a narrow fill writes at once: a block's wide scratch stays in cache.
const BLOCK_ROWS: usize = 1 << 9;

/// Rows of a block written to their columns at once (a tile of round rows is
/// 24 KB wide).
const WRITE_TILE: usize = 8;

/// The fewest blocks a worker fills in one turn: each turn starts a fresh
/// count of the byte table (2^16 entries an operation) and ends merging it,
/// so a turn must be many compressions long.
#[cfg_attr(not(feature = "parallel"), allow(dead_code))]
const BLOCKS_PER_WORKER_TURN: usize = 32;

/// The fewest compressions the records path counts in one turn, for the same
/// reason.
#[cfg_attr(not(feature = "parallel"), allow(dead_code))]
const COMPRESSIONS_PER_COUNT_TURN: usize = 1 << 12;

/// The bytes each round column's values need: tags and the word cells
/// (`a`, `c` inputs, the message) four, everything else one (bytes, nibbles,
/// bits, `MU`, `RND`).
fn round_widths() -> Vec<u8> {
    let mut w = vec![1u8; round::NUM_COLUMNS];
    w[round::TAG] = 4;
    for i in 0..4 {
        w[round::IA + i] = 4;
        w[round::IC + i] = 4;
    }
    for k in 0..16 {
        w[round::M + k] = 4;
    }
    w
}

/// The bytes each head column's values need: addresses eight, tags and read
/// counts four, the `h` limbs two, the canonicity inverses eight, the rest
/// one.
fn head_widths() -> Vec<u8> {
    let mut w = vec![1u8; head::NUM_COLUMNS];
    w[l::TAG] = 4;
    for j in 0..l::DIGEST_WORDS {
        w[l::haddr(j)] = 8;
        w[l::oaddr(j)] = 8;
        w[l::raddr(j)] = 8;
        w[l::ocnt(j)] = 4;
        w[l::rcnt(j)] = 4;
    }
    for j in 0..l::LANE_WORDS {
        w[l::maddr(j)] = 8;
    }
    for k in 0..8 {
        w[head::HL + k] = 2;
        w[head::CG + k] = 8;
    }
    w
}

/// `rows` rows of `widths.len()` columns, filled [`BLOCK_ROWS`] at a time by
/// `fill(first_row, rows, scratch, hists)` into a wide scratch and written
/// straight into narrow columns at `widths`, blocks in parallel. The scratch
/// and the counts are a worker's, reused block after block: `fill` must write
/// every cell of the scratch's first `rows` rows. Returns the columns and the
/// counts of every block (`hists.counted`, merged once a worker); `None` when
/// a value does not fit its column's width.
fn fill_narrow<Fill>(
    rows: usize,
    widths: &[u8],
    fill: Fill,
) -> Option<(multilinear::narrow::NarrowColumns, TableHist)>
where
    Fill: Fn(usize, usize, &mut [FE], &mut FillHists) + Sync,
{
    use math::field::traits::IsPrimeField;
    let cols = widths.len();
    let total: usize = widths.iter().map(|&w| rows * w as usize).sum();
    let mut data = vec![0u8; total];
    // Each column's bytes, cut into its blocks' segments; then by block.
    let num_blocks = rows.div_ceil(BLOCK_ROWS);
    let mut by_block: Vec<Vec<&mut [u8]>> =
        (0..num_blocks).map(|_| Vec::with_capacity(cols)).collect();
    let mut rest = data.as_mut_slice();
    for &w in widths {
        let (column, tail) = rest.split_at_mut(rows * w as usize);
        for (b, segment) in column.chunks_mut(BLOCK_ROWS * w as usize).enumerate() {
            by_block[b].push(segment);
        }
        rest = tail;
    }
    // A worker's state: its counts and its scratch, one block's rows wide.
    let fresh = || (FillHists::new(), zeroed_fe_vec(BLOCK_ROWS * cols));
    let one = |(mut hists, mut scratch): (FillHists, Vec<FE>),
               (b, mut segments): (usize, Vec<&mut [u8]>)|
     -> Option<(FillHists, Vec<FE>)> {
        let first = b * BLOCK_ROWS;
        let len = BLOCK_ROWS.min(rows - first);
        let scratch_rows = &mut scratch[..len * cols];
        fill(first, len, scratch_rows, &mut hists);
        // Tile by tile: a tile's rows stay in the first-level cache while
        // every column takes its share of them, each written at its width.
        let canon =
            |r: usize, c: usize| GoldilocksField::canonical(scratch_rows[r * cols + c].value());
        for t0 in (0..len).step_by(WRITE_TILE) {
            let t1 = (t0 + WRITE_TILE).min(len);
            for (c, (segment, &w)) in segments.iter_mut().zip(widths).enumerate() {
                match w {
                    1 => {
                        for r in t0..t1 {
                            let v = canon(r, c);
                            if v > u8::MAX as u64 {
                                return None;
                            }
                            segment[r] = v as u8;
                        }
                    }
                    2 => {
                        for r in t0..t1 {
                            let v = canon(r, c);
                            if v > u16::MAX as u64 {
                                return None;
                            }
                            segment[2 * r..2 * r + 2].copy_from_slice(&(v as u16).to_le_bytes());
                        }
                    }
                    4 => {
                        for r in t0..t1 {
                            let v = canon(r, c);
                            if v > u32::MAX as u64 {
                                return None;
                            }
                            segment[4 * r..4 * r + 4].copy_from_slice(&(v as u32).to_le_bytes());
                        }
                    }
                    8 => {
                        for r in t0..t1 {
                            segment[8 * r..8 * r + 8].copy_from_slice(&canon(r, c).to_le_bytes());
                        }
                    }
                    w => {
                        let w = w as usize;
                        for r in t0..t1 {
                            let v = canon(r, c);
                            if w < 8 && v >> (8 * w) != 0 {
                                return None;
                            }
                            segment[r * w..(r + 1) * w].copy_from_slice(&v.to_le_bytes()[..w]);
                        }
                    }
                }
            }
        }
        Some((hists, scratch))
    };
    #[cfg(feature = "parallel")]
    let hist = {
        use rayon::prelude::*;
        by_block
            .into_par_iter()
            .enumerate()
            .with_min_len(BLOCKS_PER_WORKER_TURN)
            .try_fold(fresh, one)
            .map(|state| state.map(|(hists, _)| hists.counted))
            .try_reduce(TableHist::default, |a, b| Some(a.merge(&b)))?
    };
    #[cfg(not(feature = "parallel"))]
    let hist = {
        let mut state = fresh();
        for item in by_block.into_iter().enumerate() {
            state = one(state, item)?;
        }
        state.0.counted
    };
    let packed = multilinear::narrow::NarrowColumns::from_parts(rows, widths.to_vec(), data)
        .expect("whole columns");
    Some((packed, hist))
}

/// [`build_b3l_traces_held`]'s narrow production path: each round and head
/// chunk filled with [`fill_narrow`]. A round block fills each compression
/// that reaches into it and keeps its rows there; a compression's lookups are
/// counted by the block holding its first round, a head row's by its own.
fn fill_narrow_direct<'p>(
    n: usize,
    shape: &B3lShape,
    consts: &(dyn Fn(usize) -> (&'p [FE], u32, u32) + Sync),
    records: &[B3lValues],
) -> Option<B3lTraces> {
    let total = l::ROUNDS * n;
    let widths = round_widths();
    let mut hist = TableHist::default();
    let mut rounds = Vec::with_capacity(shape.rounds.len());
    let mut start = 0usize;
    for &size in &shape.rounds {
        let chunk_start = start;
        let (packed, part) = fill_narrow(size, &widths, |first, len, scratch, hists| {
            let lo = chunk_start + first;
            let hi = (lo + len).min(total);
            fill_round_rows(lo, hi, scratch, hists, |k| {
                let (_, block_len, flags) = consts(k);
                (records[k], block_len, flags, l::tag_for_row(k))
            });
        })?;
        hist = hist.merge(&part);
        rounds.push(HeldTable::Narrow(packed));
        start += size;
    }
    let widths = head_widths();
    let width = head::NUM_COLUMNS;
    let mut heads = Vec::with_capacity(shape.heads.len());
    let mut start = 0usize;
    for &size in &shape.heads {
        let chunk_start = start;
        let (packed, part) = fill_narrow(size, &widths, |first, len, scratch, hists| {
            let lo = chunk_start + first;
            let hi = (lo + len).min(n);
            // The scratch is the worker's last block's: every row zeroed
            // before its fill, the padding rows past the last compression too.
            scratch.iter_mut().for_each(|v| *v = FE::zero());
            for (k, record) in records.iter().enumerate().take(hi).skip(lo) {
                let (prep, block_len, flags) = consts(k);
                let fs = lean::final_state(record, block_len, flags);
                let at = (k - lo) * width;
                lean::fill_head_uninverted(
                    &mut scratch[at..at + width],
                    prep,
                    record,
                    &fs,
                    &mut hists.counted,
                );
            }
            lean::invert_heads(scratch);
        })?;
        hist = hist.merge(&part);
        heads.push(HeldTable::Narrow(packed));
        start += size;
    }
    let table = lean::table_trace(&hist);
    let table = HeldTable::from_rows(
        table.main_table.row_major_data().to_vec(),
        table::NUM_COLUMNS,
        B3lHold::Narrow,
    );
    Some(B3lTraces {
        rounds,
        heads,
        table,
        muxes: Vec::new(),
    })
}

/// [`build_b3l_traces_held`]'s records path: each round chunk held as the
/// compressions that reach into it, the byte table's counts made on the host
/// from every compression once (its rows written to a scratch and dropped),
/// the head chunks filled narrow.
fn fill_records<'p>(
    n: usize,
    shape: &B3lShape,
    consts: &(dyn Fn(usize) -> (&'p [FE], u32, u32) + Sync),
    records: &[B3lValues],
) -> Option<B3lTraces> {
    let op_of = |k: usize| {
        let (_, block_len, flags) = consts(k);
        let values = &records[k];
        multilinear::records::B3lRoundOp {
            h: values.h,
            m: values.message(),
            block_len,
            flags,
            tag: l::tag_for_row(k),
        }
    };
    let total = l::ROUNDS * n;
    let mut rounds = Vec::with_capacity(shape.rounds.len());
    let mut start = 0usize;
    for &size in &shape.rounds {
        let end = (start + size).min(total);
        let (k_lo, k_hi) = if start < end {
            (start / l::ROUNDS, end.div_ceil(l::ROUNDS))
        } else {
            (n, n)
        };
        #[cfg(feature = "parallel")]
        let ops: Vec<_> = {
            use rayon::prelude::*;
            (k_lo..k_hi)
                .into_par_iter()
                .with_min_len(1 << 12)
                .map(op_of)
                .collect()
        };
        #[cfg(not(feature = "parallel"))]
        let ops: Vec<_> = (k_lo..k_hi).map(op_of).collect();
        let skip = start.saturating_sub(l::ROUNDS * k_lo);
        rounds.push(HeldTable::Records(multilinear::records::Records(
            std::sync::Arc::new(B3lRoundRecords {
                ops,
                skip,
                rows: size,
            }),
        )));
        start += size;
    }
    // The byte table counts every compression's round lookups once.
    let count = |k: usize, rows: &mut Vec<FE>, hist: &mut TableHist| {
        let (_, block_len, flags) = consts(k);
        lean::fill_rounds(
            rows,
            l::tag_for_row(k),
            &records[k],
            block_len,
            flags,
            None,
            l::ROUNDS,
            hist,
        );
    };
    #[cfg(feature = "parallel")]
    let mut hist = {
        use rayon::prelude::*;
        // A turn is many compressions long: each starts a fresh count of the
        // byte table and ends merging it ([`BLOCKS_PER_WORKER_TURN`]'s reason).
        (0..n)
            .into_par_iter()
            .with_min_len(COMPRESSIONS_PER_COUNT_TURN)
            .fold(
                || {
                    (
                        TableHist::default(),
                        zeroed_fe_vec(l::ROUNDS * round::NUM_COLUMNS),
                    )
                },
                |(mut hist, mut rows), k| {
                    count(k, &mut rows, &mut hist);
                    (hist, rows)
                },
            )
            .map(|(hist, _)| hist)
            .reduce(TableHist::default, |a, b| a.merge(&b))
    };
    #[cfg(not(feature = "parallel"))]
    let mut hist = {
        let mut hist = TableHist::default();
        let mut rows = zeroed_fe_vec(l::ROUNDS * round::NUM_COLUMNS);
        for k in 0..n {
            count(k, &mut rows, &mut hist);
        }
        hist
    };
    let widths = head_widths();
    let width = head::NUM_COLUMNS;
    let mut heads = Vec::with_capacity(shape.heads.len());
    let mut start = 0usize;
    for &size in &shape.heads {
        let chunk_start = start;
        let (packed, part) = fill_narrow(size, &widths, |first, len, scratch, hists| {
            let lo = chunk_start + first;
            let hi = (lo + len).min(n);
            // The scratch is the worker's last block's: every row zeroed
            // before its fill, the padding rows past the last compression too.
            scratch.iter_mut().for_each(|v| *v = FE::zero());
            for (k, record) in records.iter().enumerate().take(hi).skip(lo) {
                let (prep, block_len, flags) = consts(k);
                let fs = lean::final_state(record, block_len, flags);
                let at = (k - lo) * width;
                lean::fill_head_uninverted(
                    &mut scratch[at..at + width],
                    prep,
                    record,
                    &fs,
                    &mut hists.counted,
                );
            }
            lean::invert_heads(scratch);
        })?;
        hist = hist.merge(&part);
        heads.push(HeldTable::Narrow(packed));
        start += size;
    }
    let table = lean::table_trace(&hist);
    let table = HeldTable::from_rows(
        table.main_table.row_major_data().to_vec(),
        table::NUM_COLUMNS,
        B3lHold::Narrow,
    );
    Some(B3lTraces {
        rounds,
        heads,
        table,
        muxes: Vec::new(),
    })
}

/// A TEST-ONLY override of the holding every lean fill on this thread takes.
#[cfg(test)]
pub(crate) mod test_hold {
    use std::cell::Cell;

    use super::B3lHold;

    thread_local! {
        static HOLD: Cell<Option<B3lHold>> = const { Cell::new(None) };
    }

    /// The override, restored when dropped.
    pub(crate) struct Override(Option<B3lHold>);

    impl Drop for Override {
        fn drop(&mut self) {
            HOLD.with(|c| c.set(self.0));
        }
    }

    pub(crate) fn set(hold: B3lHold) -> Override {
        Override(HOLD.with(|c| c.replace(Some(hold))))
    }

    pub(super) fn get() -> Option<B3lHold> {
        HOLD.with(Cell::get)
    }
}

/// A TEST-ONLY override of the lean constraint sets every [`B3lAirs::new`]
/// on this thread builds (prover and verifier alike): a mutation, to show a
/// constraint group is what refuses its forgery. No production build reaches
/// it.
/// A TEST-ONLY override of the mux tables' constraint set (a mutant),
/// per thread, restored when the guard drops.
#[cfg(test)]
pub(crate) mod test_mux {
    use std::cell::Cell;

    use super::MuxConstraints;

    thread_local! {
        static MUX: Cell<Option<MuxConstraints>> = const { Cell::new(None) };
    }

    pub(crate) struct Override(Option<MuxConstraints>);

    impl Drop for Override {
        fn drop(&mut self) {
            MUX.with(|c| c.set(self.0));
        }
    }

    pub(crate) fn set(constraints: MuxConstraints) -> Override {
        Override(MUX.with(|c| c.replace(Some(constraints))))
    }

    pub(super) fn get() -> Option<MuxConstraints> {
        MUX.with(Cell::get)
    }
}

#[cfg(test)]
pub(crate) mod test_drops {
    use std::cell::Cell;

    use super::{HeadConstraints, RoundConstraints};

    thread_local! {
        static DROPS: Cell<(RoundConstraints, HeadConstraints)> =
            Cell::new((RoundConstraints::default(), HeadConstraints::default()));
    }

    /// The override, restored when dropped.
    pub(crate) struct Override((RoundConstraints, HeadConstraints));

    impl Drop for Override {
        fn drop(&mut self) {
            DROPS.with(|c| c.set(self.0));
        }
    }

    pub(crate) fn set(rounds: RoundConstraints, heads: HeadConstraints) -> Override {
        Override(DROPS.with(|c| c.replace((rounds, heads))))
    }

    pub(super) fn get() -> (RoundConstraints, HeadConstraints) {
        DROPS.with(Cell::get)
    }
}
