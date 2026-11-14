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
use super::compiler::{ColumnGroup, LfmProgram};
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

/// The lean tables' shape: the padded rows of every round chunk and every
/// head chunk, in table order. Empty for a program with no compression, which
/// then has no lean table at all, not even the byte table.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct B3lShape {
    pub rounds: Vec<usize>,
    pub heads: Vec<usize>,
}

impl B3lShape {
    /// The shape of `compressions` compressions.
    pub fn of(compressions: usize) -> Self {
        if compressions == 0 {
            return Self::default();
        }
        Self {
            rounds: lean::chunk_rows(l::ROUNDS * compressions, CHUNK_LOG, MIN_CHUNK_LOG),
            heads: lean::chunk_rows(compressions, CHUNK_LOG, MIN_CHUNK_LOG),
        }
    }

    /// A program's.
    pub fn of_program(program: &LfmProgram) -> Self {
        Self::of(program.groups.b3l.real_rows)
    }

    pub fn is_empty(&self) -> bool {
        self.heads.is_empty()
    }

    /// The tables this shape adds: the round chunks, the head chunks and the
    /// byte table; none when empty.
    pub fn num_tables(&self) -> usize {
        if self.is_empty() {
            0
        } else {
            self.rounds.len() + self.heads.len() + 1
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
        v
    }

    /// Each AIR with its trace, in table order.
    pub fn pairs<'a>(
        &'a self,
        traces: &'a mut B3lTraces,
    ) -> Result<Vec<(DynLfmAir<'a>, &'a mut TraceTable<F, E>)>, String> {
        if traces.rounds.len() != self.rounds.len() || traces.heads.len() != self.heads.len() {
            return Err(format!(
                "the lean traces are {} round and {} head chunks, the AIR set {} and {}",
                traces.rounds.len(),
                traces.heads.len(),
                self.rounds.len(),
                self.heads.len()
            ));
        }
        let Some(table_air) = &self.table else {
            return Err("lean traces for an AIR set with no lean table".into());
        };
        let mut v: Vec<(DynLfmAir<'a>, &'a mut TraceTable<F, E>)> = Vec::new();
        for (air, t) in self.rounds.iter().zip(traces.rounds.iter_mut()) {
            v.push((air, t));
        }
        for (air, t) in self.heads.iter().zip(traces.heads.iter_mut()) {
            v.push((air, t));
        }
        v.push((table_air, &mut traces.table));
        Ok(v)
    }
}

/// A program's lean tables, filled.
pub struct B3lTraces {
    pub rounds: Vec<TraceTable<F, E>>,
    pub heads: Vec<TraceTable<F, E>>,
    pub table: TraceTable<F, E>,
}

impl B3lTraces {
    /// The round chunk holding round-table row `i` (row `7k + r` is round `r`
    /// of compression `k`), and `i`'s row in it.
    pub fn round_at(&self, i: usize) -> Option<(&TraceTable<F, E>, usize)> {
        let mut start = 0usize;
        for t in &self.rounds {
            let rows = t.num_rows();
            if i < start + rows {
                return Some((t, i - start));
            }
            start += rows;
        }
        None
    }

    /// Every table in table order.
    pub fn tables_mut(&mut self) -> impl Iterator<Item = &mut TraceTable<F, E>> {
        self.rounds
            .iter_mut()
            .chain(self.heads.iter_mut())
            .chain(core::iter::once(&mut self.table))
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
    out
}

/// Fill a program's lean tables from its `B3L_HEAD` records (one a
/// compression, in program order; the executor's [`B3lValues`]). `None` for a
/// program with no compression.
pub fn build_b3l_traces(program: &LfmProgram, records: &[B3lValues]) -> Option<B3lTraces> {
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
    let total = l::ROUNDS * n;
    let mut round_data = zeroed_fe_vec(total * round::NUM_COLUMNS);
    let head_rows: usize = shape.heads.iter().sum();
    let mut head_data = zeroed_fe_vec(head_rows * head::NUM_COLUMNS);
    let fill_one = |k: usize, rows: &mut [FE], hrow: &mut [FE], hist: &mut TableHist| {
        let prep = &g.data[k * g.width..(k + 1) * g.width];
        let canon = |c: usize| {
            use math::field::traits::IsPrimeField;
            GoldilocksField::canonical(prep[c].value())
        };
        let (block_len, flags) = (canon(l::BLEN) as u32, canon(l::FLAGS) as u32);
        let tag = l::tag_for_row(k);
        let fs = lean::fill_rounds(
            rows,
            tag,
            &records[k],
            block_len,
            flags,
            None,
            l::ROUNDS,
            hist,
        );
        lean::fill_head(hrow, prep, &records[k], &fs, hist);
    };
    #[cfg(feature = "parallel")]
    let hist = {
        use rayon::prelude::*;
        round_data
            .par_chunks_mut(l::ROUNDS * round::NUM_COLUMNS)
            .zip(head_data.par_chunks_mut(head::NUM_COLUMNS))
            .enumerate()
            .fold(TableHist::default, |mut hist, (k, (rows, hrow))| {
                fill_one(k, rows, hrow, &mut hist);
                hist
            })
            .reduce(TableHist::default, |a, b| a.merge(&b))
    };
    #[cfg(not(feature = "parallel"))]
    let hist = {
        let mut hist = TableHist::default();
        for (k, (rows, hrow)) in round_data
            .chunks_mut(l::ROUNDS * round::NUM_COLUMNS)
            .zip(head_data.chunks_mut(head::NUM_COLUMNS))
            .enumerate()
        {
            fill_one(k, rows, hrow, &mut hist);
        }
        hist
    };
    let split = |data: &mut Vec<FE>, width: usize, sizes: &[usize]| -> Vec<TraceTable<F, E>> {
        let mut starts = Vec::with_capacity(sizes.len());
        let mut s = 0usize;
        for &size in sizes {
            starts.push(s);
            s += size;
        }
        // From the back, so each chunk's rows move once.
        let mut out = Vec::with_capacity(sizes.len());
        for (&start, &size) in starts.iter().zip(sizes).rev() {
            let mut chunk = data.split_off((start * width).min(data.len()));
            chunk.resize(size * width, FE::zero());
            out.push(TraceTable::new_main(chunk, width, 1));
        }
        out.reverse();
        out
    };
    let rounds = split(&mut round_data, round::NUM_COLUMNS, &shape.rounds);
    let heads = split(&mut head_data, head::NUM_COLUMNS, &shape.heads);
    Some(B3lTraces {
        rounds,
        heads,
        table: lean::table_trace(&hist),
    })
}

/// A TEST-ONLY override of the lean constraint sets every [`B3lAirs::new`]
/// on this thread builds (prover and verifier alike): a mutation, to show a
/// constraint group is what refuses its forgery. No production build reaches
/// it.
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
