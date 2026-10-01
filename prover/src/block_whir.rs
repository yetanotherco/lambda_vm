//! The block proved with the multilinear argument in ONE proof, no epochs:
//! prove-and-retire on WHIR.
//!
//! Same executor, traces and AIRs as [`crate::multilinear_prove`], at a table
//! height the card proves today (2^21 by default); every chunk of every table
//! type is a table of the proof — including KECCAK_RND, split at 2^16 rows so
//! no argument is taller than today's — and memory is the monolithic PAGE
//! argument (no local-to-global bookend, no cross-epoch proof).
//!
//! The proof is [`stark::multilinear_block`]'s: the tables split into groups a
//! card holds ([`block_groups`], a function of the shapes and of
//! [`BlockFormat`]); phase A commits each group and lets its codewords go;
//! every root goes into the transcript and `(z, α, β)` are drawn once; phase B
//! argues and opens each group on its own fork of that transcript. The bus
//! balance is checked once, over every table of the block.
//!
//! The columns an in-guest verifier cannot evaluate — DECODE's, and the dense
//! genesis pages' — are settled by PREPARED openings ([`prepared_tables`]):
//! commitments both sides derive from the program, absorbed after the groups'
//! roots and opened on the table's group fork, as an epoch settles DECODE.
//!
//! ★ #1010's epoch pipeline is untouched: this is a separate entry point with a
//! statement tag of its own ([`statement::MULTILINEAR_BLOCK_TAG`]).

use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use crypto::fiat_shamir::is_transcript::IsTranscript;
use executor::elf::Elf;
use executor::vm::execution::Executor;
use math::field::element::FieldElement;
use multilinear::mle::Mle;
use rayon::prelude::*;
use stark::multilinear_block::{self, BlockCommitted, GroupStamps, block_groups};
use stark::multilinear_table::{CommittedTable, MultiProof, TableLayout, TableStatement};
use stark::table::Table;
use stark::trace::TraceTable;
use std::time::Instant;

use crate::multilinear_prove::{
    absorb_tagged, chain_config_under, layout_of, preprocessed_mles, shapes_of, stacks,
};
use crate::statement::{self, MULTILINEAR_BLOCK_TAG};
use crate::tables::trace_builder::{
    ChunkJob, StreamTable, Traces, WindowStamps, WindowedTraceBuilder,
};
use crate::test_utils::{E, F};
use crate::zf_format::ZfFormat;
use crate::{
    Error, FIXED_TABLE_COUNT, MaxRowsConfig, ProofOptions, RuntimePageRange, TableCounts, VmAirs,
};

/// How many stacked polynomials a group may take. A format constant: both sides
/// derive the groups from it ([`block_groups`]). Three is today's heaviest
/// epoch (three 2^27 polynomials, epochs 3–6 and 13 of block 25368371), so no
/// group asks the card for more than an epoch does.
pub const BLOCK_GROUP_POLYS: usize = 3;

/// The most KECCAK_RND tables a block statement may declare: a bound on what
/// the verifier builds before anything is checked, far above any block.
pub const BLOCK_MAX_KECCAK_RND: usize = 1 << 12;

/// The rows each KECCAK_RND table holds at most — a prover's choice, not the
/// format's: 2^16 is today's tallest, so its argument's tree (the largest a
/// block has) is today's.
pub const BLOCK_KECCAK_RND_ROWS_LOG2: usize = 16;

/// Tree levels a group keeps OFF the host between its commit and its opening:
/// a query re-hashes `2^4` of its codeword's cosets to rebuild them. A
/// prover's choice; the proof does not depend on it.
pub const BLOCK_TREE_DROP_LEVELS: usize = 4;

/// The verifier-side constants of a block proof. [`BlockFormat::production`]
/// is what the block runs at; the tests shrink the stack so a small program
/// spans several groups.
#[derive(Clone, Copy, Debug)]
pub struct BlockFormat {
    pub zf: ZfFormat,
    /// The most stacked polynomials a group may take: the prover packs to it,
    /// and the verifier refuses a group whose stack needs more.
    pub group_polys: usize,
    /// The most groups a block statement may declare (the verifier refuses
    /// more). The proven bits are quoted at this count (I-NOEPOCH-W.md §8).
    pub max_groups: usize,
}

/// The most groups a block statement may declare. The block has 9; the bound
/// leaves room for larger blocks and caps what a statement can make the
/// verifier build.
pub const BLOCK_MAX_GROUPS: usize = 64;

impl BlockFormat {
    /// The process's WHIR format (as [`crate::multilinear_prove::chain_config`]
    /// reads it), [`BLOCK_GROUP_POLYS`] and [`BLOCK_MAX_GROUPS`].
    pub fn production() -> Self {
        Self {
            zf: *ZfFormat::global(),
            group_polys: BLOCK_GROUP_POLYS,
            max_groups: BLOCK_MAX_GROUPS,
        }
    }
}

/// The prover's own choices: none of them is in the proof.
#[derive(Clone, Debug)]
pub struct BlockOptions {
    pub max_rows: MaxRowsConfig,
    pub keccak_rnd_rows_log2: usize,
    pub drop_levels: usize,
    /// `Some(k)`: build the traces in windows of 2^k cycles and commit each
    /// table as its chunk completes ([`WindowedTraceBuilder`]); `None`: build
    /// the whole run, then commit.
    pub window_log2: Option<usize>,
}

impl BlockOptions {
    /// Every chunked table at 2^21 rows, KECCAK_RND at 2^16, windows of 2^20
    /// cycles.
    pub fn production() -> Self {
        Self {
            max_rows: MaxRowsConfig::uniform(1 << 21),
            keccak_rnd_rows_log2: BLOCK_KECCAK_RND_ROWS_LOG2,
            drop_levels: BLOCK_TREE_DROP_LEVELS,
            window_log2: Some(BLOCK_WINDOW_LOG2),
        }
    }
}

/// Cycles a streamed prove collects at a time: half a CPU instance at 2^21, so
/// every chunk is handed to phase A within a window of completing.
pub const BLOCK_WINDOW_LOG2: usize = 20;

/// A block proved in one proof.
///
/// The statement is [`crate::multilinear_prove::MultilinearVmProof`]'s; the
/// proof is read under the block's forked schedule, and only by
/// [`verify_block_whir`].
#[derive(Debug, Clone, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct BlockWhirProof {
    pub proof: MultiProof<F, E>,
    /// Each table's height in variables, in [`VmAirs::air_refs`] order.
    pub table_num_vars: Vec<u8>,
    pub runtime_page_ranges: Vec<RuntimePageRange>,
    pub table_counts: TableCounts,
    pub public_output: Vec<u8>,
    pub num_private_input_pages: usize,
    /// The groups, each its tables' indices in [`VmAirs::air_refs`] order, in
    /// the order the group stacks them. The prover's choice — a streamed prove
    /// packs tables as they arrive — bound in the statement before any root
    /// ([`absorb_block`]); the verifier checks it is a partition and rebuilds
    /// each group's stack from it under its own stack cap. The proof's tables
    /// are in this order.
    pub groups: Vec<Vec<u32>>,
    /// The prepared openings ([`prepared_tables`]), one per prepared table, in
    /// the proof's table order.
    pub prepared: Vec<multilinear::stacked_eval::StackedProof<F, E>>,
}

/// Where a block's prove spent its time, for the readout. Seconds.
#[derive(Clone, Debug, Default)]
pub struct BlockStamps {
    pub execute: f64,
    pub build: f64,
    /// KECCAK_RND split, AIRs, transposition and layouts.
    pub prep: f64,
    pub phase_a: f64,
    pub phase_b: f64,
    pub tables: usize,
    pub cells: usize,
    pub groups: Vec<GroupStamps>,
    /// The trace build's phase ends and each phase-5 generator's finish,
    /// seconds into the build.
    pub build_marks: Vec<(String, f64)>,
    /// Phase B's first-round trees: `open_many` calls served from the kept
    /// tops and the leaves re-hashed for them. A revived commitment never
    /// builds a device tree, so nothing re-commits.
    pub top_paths: (u64, u64),
    /// A streamed build: when its windows were all collected (seconds since
    /// the build started) and how many chunks they handed out.
    pub streamed: (f64, usize),
    /// A streamed build's windows: the walk, the routing and the streamed
    /// chunks' generation, summed.
    pub windows: WindowStamps,
    /// The prepared commitments: how many, and the seconds deriving them.
    pub prepared: (usize, f64),
}

impl BlockStamps {
    /// One line a group and a total, `BLOCK …` prefixed, for a box log.
    pub fn report(&self) -> String {
        let sum = |f: fn(&GroupStamps) -> f64| self.groups.iter().map(f).sum::<f64>();
        let polys: usize = self.groups.iter().map(|g| g.polys).sum();
        let tree: usize = self.groups.iter().map(|g| g.tree_bytes).sum();
        let mut out = format!(
            "BLOCK SHAPE: tables {} · groups {} · chains {} · cells {:.3} G · tree tops {:.2} GiB\n",
            self.tables,
            self.groups.len(),
            polys,
            self.cells as f64 / 1e9,
            tree as f64 / (1u64 << 30) as f64,
        );
        for (g, s) in self.groups.iter().enumerate() {
            out.push_str(&format!(
                "BLOCK GROUP {g}: tables {} · polys {} · cells {:.1} M || A wait {:.3} upload {:.3} commit {:.3} retire {:.3} done@{:.2} || B upload {:.3} argue {:.3} encode {:.3} open {:.3}\n",
                s.tables,
                s.polys,
                s.cells as f64 / 1e6,
                s.wait_a,
                s.upload_a,
                s.commit,
                s.retire,
                s.committed_at,
                s.upload_b,
                s.argue,
                s.encode,
                s.open,
            ));
        }
        if !self.build_marks.is_empty() {
            let marks: Vec<String> = self
                .build_marks
                .iter()
                .map(|(label, at)| format!("{label} {at:.2}"))
                .collect();
            out.push_str(&format!("BLOCK BUILD MARKS: {}\n", marks.join(" · ")));
        }
        if self.streamed.1 > 0 || self.streamed.0 > 0.0 {
            out.push_str(&format!(
                "BLOCK STREAM: windows collected at {:.2}s · run built at {:.2}s · {} chunks streamed · layout busy {:.2}s · phase A ended at {:.2}s (all since the streamed prove started, execution included)\n",
                self.streamed.0, self.build, self.streamed.1, self.prep, self.phase_a,
            ));
            out.push_str(&format!(
                "BLOCK WINDOWS: {} windows · walk {:.2}s (walker thread) · route+append {:.2}s · chunk handout {:.2}s (accumulator thread) · finish {:.2}s\n",
                self.windows.windows,
                self.windows.walk,
                self.windows.route,
                self.windows.generate,
                self.build - self.streamed.0,
            ));
        }
        out.push_str(&format!(
            "BLOCK RECOMMIT: 0 (a revived commitment builds no tree) · first-round paths from kept tops: {} calls · {} leaves re-hashed on the host · phase B on one thread + one upload helper\n",
            self.top_paths.0, self.top_paths.1,
        ));
        let argue = sum(|g| g.argue);
        let open = sum(|g| g.open);
        let tax = sum(|g| g.upload_b + g.encode);
        out.push_str(&format!(
            "BLOCK PREPARED: tables {} · derived in {:.3}\n",
            self.prepared.0, self.prepared.1
        ));
        out.push_str(&format!(
            "BLOCK PHASES: execute {:.2} · build {:.2} · prep {:.2} · A {:.2} (wait {:.2} upload {:.2} commit {:.2} retire {:.2}) · B {:.2} (argue {:.2} open {:.2} tax {:.2} = upload {:.2} + encode {:.2})\n",
            self.execute,
            self.build,
            self.prep,
            self.phase_a,
            sum(|g| g.wait_a),
            sum(|g| g.upload_a),
            sum(|g| g.commit),
            sum(|g| g.retire),
            self.phase_b,
            argue,
            open,
            tax,
            sum(|g| g.upload_b),
            sum(|g| g.encode),
        ));
        out
    }
}

/// The counts a block statement may declare: [`TableCounts::validate`], with
/// KECCAK_RND chunked (at most [`BLOCK_MAX_KECCAK_RND`] tables).
pub fn validate_block_counts(counts: &TableCounts) -> Result<(), Error> {
    if counts.keccak_rnd > BLOCK_MAX_KECCAK_RND {
        return Err(Error::InvalidTableCounts(format!(
            "keccak_rnd count is {} — a block takes at most {BLOCK_MAX_KECCAK_RND}",
            counts.keccak_rnd,
        )));
    }
    let mut rest = counts.clone();
    rest.keccak_rnd = rest.keccak_rnd.min(1);
    rest.validate()
}

/// The block statement: the monolithic multilinear one under
/// [`MULTILINEAR_BLOCK_TAG`], then the group partition — the count, then per
/// group its size and its tables' indices, every value a little-endian `u64`
/// (so the roots that follow stay on a field-element boundary).
#[allow(clippy::too_many_arguments)]
pub(crate) fn absorb_block(
    t: &mut impl IsTranscript<E>,
    elf_bytes: &[u8],
    public_output: &[u8],
    table_counts: &TableCounts,
    num_private_input_pages: usize,
    runtime_page_ranges: &[RuntimePageRange],
    table_num_vars: &[u8],
    config: &multilinear::whir_chain::ChainConfig,
    groups: &[Vec<u32>],
) {
    absorb_tagged(
        t,
        MULTILINEAR_BLOCK_TAG,
        "block",
        &statement::elf_digest(elf_bytes),
        public_output,
        table_counts,
        num_private_input_pages,
        runtime_page_ranges,
        table_num_vars,
        config,
    );
    t.append_bytes(&(groups.len() as u64).to_le_bytes());
    for group in groups {
        t.append_bytes(&(group.len() as u64).to_le_bytes());
        for &index in group {
            t.append_bytes(&u64::from(index).to_le_bytes());
        }
    }
}

/// A partition of `tables` tables into non-empty groups: every index below
/// `tables`, each exactly once. Returns the indices in proof order.
pub fn validate_groups(groups: &[Vec<u32>], tables: usize) -> Result<Vec<usize>, Error> {
    let mut seen = vec![false; tables];
    let mut order = Vec::with_capacity(tables);
    for group in groups {
        if group.is_empty() {
            return Err(Error::InvalidTableCounts("an empty group".to_string()));
        }
        for &index in group {
            let index = index as usize;
            match seen.get_mut(index) {
                Some(slot) if !*slot => *slot = true,
                Some(_) => {
                    return Err(Error::InvalidTableCounts(format!(
                        "table {index} is in two groups"
                    )));
                }
                None => {
                    return Err(Error::InvalidTableCounts(format!(
                        "group names table {index} of {tables}"
                    )));
                }
            }
            order.push(index);
        }
    }
    if order.len() != tables {
        return Err(Error::InvalidTableCounts(format!(
            "the groups cover {} of {tables} tables",
            order.len()
        )));
    }
    Ok(order)
}

/// A table cut into tables of `rows` rows each, in row order. Every VM table's
/// constraints read one row (no shifted reads), so each piece is a table of
/// the same AIR, and the bus sums over the pieces are the sum over the whole.
pub(crate) fn split_rows(table: TraceTable<F, E>, rows: usize) -> Vec<TraceTable<F, E>> {
    let height = table.main_table.height;
    if height <= rows {
        return vec![table];
    }
    let step = table.step_size;
    let width = table.main_table.width;
    // Row-major in, row-major out: each piece is a run of whole rows, copied
    // as they lie (a transposition of KECCAK_RND's 1,480 columns is what this
    // used to cost).
    (0..height / rows)
        .into_par_iter()
        .map(|k| {
            let mut data = Vec::with_capacity(rows * width);
            for row in k * rows..(k + 1) * rows {
                data.extend_from_slice(table.main_table.get_row(row));
            }
            TraceTable::new_main(data, width, step)
        })
        .collect()
}

/// One table of the proof: its layout and its columns, taken out of the trace
/// (whose row-major copy goes when `release_rows`), checked against the
/// columns the program implies.
fn table_of<'a>(
    air: &'a dyn stark::traits::AIR<Field = F, FieldExtension = E, PublicInputs = ()>,
    trace: &mut TraceTable<F, E>,
    (width, num_vars): (usize, usize),
    release_rows: bool,
) -> Result<CommittedTable<'a, F, E>, Error> {
    let layout = layout_of(air, width, num_vars)
        .map_err(|e| Error::Prover(format!("{}: {e:?}", air.name())))?;
    let mut columns = trace.main_table.columns_blocked();
    // The row-major copy goes as soon as the columns exist: a block's traces
    // are tens of GiB, and two copies of them at once is the peak this entry
    // point exists to avoid.
    if release_rows {
        trace.main_table = Table::new(Vec::new(), 0);
    }
    for (col, expected) in air.precomputed_columns().iter().enumerate() {
        if columns.get(col) != Some(expected) {
            return Err(Error::Prover(format!(
                "{}: preprocessed column {col} is not what the program implies",
                air.name(),
            )));
        }
    }
    CommittedTable::from_layout(layout, |col| core::mem::take(&mut columns[col as usize]))
        .map_err(|e| Error::Prover(format!("{}: {e:?}", air.name())))
}

/// A prepared table's index in [`VmAirs::air_refs`] order and the columns its
/// opening settles.
pub(crate) type PreparedColumns = (usize, Vec<Vec<FieldElement<F>>>);

/// One table's PREPARED commitment ([`multilinear_block::BlockPrepared`]): its
/// leading preprocessed columns, committed under the block's parameters. Both
/// sides derive it from the program — the prover opens it, the verifier absorbs
/// its roots and checks the opening — and no proof carries it.
pub(crate) struct TablePrepared<H: multilinear::whir_hash::WhirHash> {
    /// The table's index in [`VmAirs::air_refs`] order.
    pub(crate) table: usize,
    pub(crate) columns: Vec<Mle<F>>,
    pub(crate) roots: Vec<multilinear::whir_commit::Commitment>,
    pub(crate) commitment: multilinear::stacked_eval::StackedCommitment<F, H>,
}

/// The tables a block opens out of band, in [`VmAirs::air_refs`] order, with
/// the columns each opening settles:
/// - DECODE: its ELF-derived columns, all of them;
/// - every genesis page the cross-epoch rule stacks
///   ([`crate::continuation::genesis_stack_plan`]): both its columns.
///
/// These are the columns an in-guest verifier cannot evaluate: five columns of
/// 2^20, and a dense page whose sparse form is millions of rows. So the block
/// settles them with an opening, as an epoch settles DECODE. The result is a
/// function of the ELF and the page configs alone. The rule reads no private
/// page's bytes, so the prover's and the verifier's views of the configs give
/// the same set.
pub(crate) fn prepared_tables(
    airs: &VmAirs,
    page_configs: &[crate::tables::page::PageConfig],
) -> Result<Vec<PreparedColumns>, Error> {
    let refs = airs.air_refs();
    if page_configs.len() != airs.pages.len() {
        return Err(Error::Prover(format!(
            "{} page configs for {} page tables",
            page_configs.len(),
            airs.pages.len()
        )));
    }
    let first_page = match airs.pages.first() {
        Some(page) => refs
            .iter()
            .position(|air| {
                std::ptr::eq(
                    *air as *const _ as *const (),
                    page.as_ref() as *const _ as *const (),
                )
            })
            .ok_or_else(|| Error::Prover("the page tables are not in the AIR set".into()))?,
        None => refs.len(),
    };
    let decode = crate::multilinear_continuation::decode_table_index(&refs)?;
    let mut tables = vec![(decode, refs[decode].precomputed_columns())];
    let plan = crate::continuation::genesis_stack_plan(
        page_configs,
        first_page,
        crate::continuation::PAGE_NUM_VARS,
    );
    for route in plan.routes.iter().filter(|route| route.dense) {
        tables.push((
            route.table,
            crate::tables::page::preprocessed_columns(&page_configs[route.table - first_page]),
        ));
    }
    tables.sort_by_key(|&(table, _)| table);
    Ok(tables)
}

/// Commits one prepared table's columns under `config`.
pub(crate) fn commit_prepared<H: multilinear::whir_hash::WhirHash>(
    table: usize,
    columns: Vec<Vec<FieldElement<F>>>,
    config: &multilinear::whir_chain::ChainConfig,
) -> Result<TablePrepared<H>, Error> {
    let columns: Vec<Mle<F>> = columns
        .into_iter()
        .map(|values| Mle::new(values).map_err(|e| Error::Prover(format!("table {table}: {e:?}"))))
        .collect::<Result<_, _>>()?;
    let num_vars = columns
        .first()
        .ok_or_else(|| Error::Prover(format!("table {table} has no prepared column")))?
        .num_vars();
    if columns.iter().any(|c| c.num_vars() != num_vars) {
        return Err(Error::Prover(format!(
            "table {table}: prepared columns of different heights"
        )));
    }
    let layout =
        stark::multilinear_table::global_layout(&[(columns.len(), num_vars)], config.format.stack)
            .map_err(|e| Error::Prover(format!("{e:?}")))?;
    let commitment = multilinear::stacked_eval::StackedCommitment::<F, H>::commit(
        layout,
        &multilinear::stacking::borrow(&columns),
        None,
        config,
    )
    .map_err(|e| Error::Prover(format!("table {table}: {e:?}")))?;
    let roots = commitment.roots();
    Ok(TablePrepared {
        table,
        columns,
        roots,
        commitment,
    })
}

/// Each prepared table's position in the proof's table order (the groups
/// concatenated), with its index in `prepared`, sorted by position.
pub(crate) fn prepared_positions<H: multilinear::whir_hash::WhirHash>(
    prepared: &[TablePrepared<H>],
    groups: &[Vec<u32>],
) -> Result<Vec<(usize, usize)>, Error> {
    let mut out = prepared
        .iter()
        .enumerate()
        .map(|(k, p)| {
            groups
                .iter()
                .flatten()
                .position(|&t| t as usize == p.table)
                .map(|position| (position, k))
                .ok_or_else(|| Error::Prover(format!("prepared table {} is in no group", p.table)))
        })
        .collect::<Result<Vec<_>, _>>()?;
    out.sort_unstable();
    Ok(out)
}

/// The prover's prepared commitments over `columns` ([`prepared_tables`]),
/// with the deviation applied, and the seconds they took.
fn prover_prepared<H: multilinear::whir_hash::WhirHash>(
    columns: Vec<PreparedColumns>,
    config: &multilinear::whir_chain::ChainConfig,
    deviations: &Deviations,
) -> Result<(Vec<TablePrepared<H>>, f64), Error> {
    let t = Instant::now();
    let prepared = columns
        .into_iter()
        .enumerate()
        .map(|(k, (table, mut columns))| {
            if k == 0 && deviations.other_prepared {
                columns[0][0] += FieldElement::<F>::one();
            }
            commit_prepared::<H>(table, columns, config)
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok((prepared, t.elapsed().as_secs_f64()))
}

/// Proves a block in one proof.
pub fn prove_block_whir(
    elf_bytes: &[u8],
    private_inputs: &[u8],
    proof_options: &ProofOptions,
    format: &BlockFormat,
    options: &BlockOptions,
) -> Result<(BlockWhirProof, BlockStamps), Error> {
    prove_block_whir_with(
        elf_bytes,
        private_inputs,
        proof_options,
        format,
        options,
        &Deviations::default(),
    )
}

/// What a test makes the prover do wrong, for the verifier to refuse. The
/// default is an honest prover.
#[derive(Default)]
pub(crate) struct Deviations {
    /// Leave the first KECCAK_RND table out of the proof — its rows, its
    /// argument and its count — as a prover hiding an instance would. The
    /// first, because it holds real rounds: a table of padding rows alone
    /// carries nothing on the bus, and leaving one out is an honest proof.
    pub omit_first_keccak_rnd: bool,
    /// Prove group `g` on the fork of this index instead of `g`.
    pub fork_of: Option<fn(usize) -> usize>,
    /// Commit and open the first prepared table over columns that differ from
    /// the program's in one entry, as a prover of another program would.
    pub other_prepared: bool,
}

pub(crate) fn prove_block_whir_with(
    elf_bytes: &[u8],
    private_inputs: &[u8],
    proof_options: &ProofOptions,
    format: &BlockFormat,
    options: &BlockOptions,
    deviations: &Deviations,
) -> Result<(BlockWhirProof, BlockStamps), Error> {
    let mut stamps = BlockStamps::default();
    let program = Elf::load(elf_bytes).map_err(|e| Error::ElfLoad(format!("{e}")))?;

    if let Some(window_log2) = options.window_log2 {
        let proof = prove_streamed(
            &program,
            elf_bytes,
            private_inputs,
            1usize << window_log2,
            proof_options,
            format,
            options,
            deviations,
            &mut stamps,
        )?;
        return Ok((proof, stamps));
    }

    let t = Instant::now();
    let result = Executor::new(&program, private_inputs.to_vec())
        .map_err(|e| Error::Execution(format!("{e}")))?
        .run()
        .map_err(|e| Error::Execution(format!("{e}")))?;
    stamps.execute = t.elapsed().as_secs_f64();

    crate::tables::trace_builder::build_stamps::start();
    let t = Instant::now();
    let mut traces = Traces::from_elf_and_logs(
        &program,
        &result.logs,
        &options.max_rows,
        private_inputs,
        #[cfg(feature = "disk-spill")]
        stark::storage_mode::StorageMode::Ram,
    )?;
    drop(result);
    stamps.build = t.elapsed().as_secs_f64();
    stamps.build_marks = crate::tables::trace_builder::build_stamps::take();

    let t = Instant::now();
    split_keccak_rnd(&mut traces, options.keccak_rnd_rows_log2);
    if deviations.omit_first_keccak_rnd && !traces.keccak_rnds.is_empty() {
        traces.keccak_rnds.remove(0);
    }
    stamps.prep = t.elapsed().as_secs_f64();
    let proof = prove_traces(
        &program,
        elf_bytes,
        &mut traces,
        proof_options,
        format,
        options,
        deviations,
        true,
        &mut stamps,
    )?;
    Ok((proof, stamps))
}

/// Cuts every KECCAK_RND table into tables of at most `2^rows_log2` rows.
pub(crate) fn split_keccak_rnd(traces: &mut Traces, rows_log2: usize) {
    let rows = 1usize << rows_log2;
    traces.keccak_rnds = std::mem::take(&mut traces.keccak_rnds)
        .into_iter()
        .flat_map(|table| split_rows(table, rows))
        .collect();
}

/// The block's proof over traces already built (and KECCAK_RND already
/// split). `release_rows` lets each table's row-major copy go as its columns
/// are taken — what a prove does; a test proving the same traces twice keeps
/// them. Adds `prep`, `phase_a`, `phase_b` and the groups to `stamps`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn prove_traces(
    program: &Elf,
    elf_bytes: &[u8],
    traces: &mut Traces,
    proof_options: &ProofOptions,
    format: &BlockFormat,
    options: &BlockOptions,
    deviations: &Deviations,
    release_rows: bool,
    stamps: &mut BlockStamps,
) -> Result<BlockWhirProof, Error> {
    let t = Instant::now();
    let table_counts = traces.table_counts();
    validate_block_counts(&table_counts)?;
    let runtime_page_ranges = traces.runtime_page_ranges();
    let num_private_input_pages = traces
        .page_configs
        .iter()
        .filter(|c| c.is_private_input)
        .count();
    let public_output = traces.public_output_bytes.clone();

    let airs = VmAirs::new(
        program,
        proof_options,
        false,
        &traces.page_configs,
        &table_counts,
        None,
        true,
        None,
        None,
        None,
    );

    let prepared_columns = prepared_tables(&airs, &traces.page_configs)?;
    let pairs = airs.air_trace_pairs(traces);
    let shapes = shapes_of(&pairs)?;
    let table_num_vars: Vec<u8> = shapes.iter().map(|&(_, n)| n as u8).collect();
    let config = chain_config_under(&format.zf, &shapes);
    let sizes = block_groups(&shapes, config.format.stack, format.group_polys)
        .map_err(|e| Error::Prover(format!("{e:?}")))?;

    stamps.tables = pairs.len();
    stamps.cells = shapes.iter().map(|&(w, n)| w << n).sum();
    stamps.prep += t.elapsed().as_secs_f64();

    // The pairs, cut into the groups phase A commits: a producer lays each
    // group's tables out (its tables in parallel) while the card commits the
    // group before it.
    let mut groups_of_pairs: Vec<Vec<(crate::AirTracePair<'_>, (usize, usize))>> =
        Vec::with_capacity(sizes.len());
    {
        let mut pairs = pairs.into_iter().zip(shapes.iter().copied());
        for &size in &sizes {
            groups_of_pairs.push(pairs.by_ref().take(size).collect());
        }
    }
    // The groups are contiguous runs in table order here.
    let groups: Vec<Vec<u32>> = {
        let mut at = 0u32;
        sizes
            .iter()
            .map(|&size| {
                let group = (at..at + size as u32).collect();
                at += size as u32;
                group
            })
            .collect()
    };
    let proof = crate::with_whir_hash!(|H| {
        let mut transcript =
            DefaultTranscript::<E, <H as multilinear::whir_hash::WhirHash>::Transcript>::new(&[]);
        absorb_block(
            &mut transcript,
            elf_bytes,
            &public_output,
            &table_counts,
            num_private_input_pages,
            &runtime_page_ranges,
            &table_num_vars,
            &config,
            &groups,
        );
        let t = Instant::now();
        let (block, produced) = std::thread::scope(|scope| {
            let (tx, rx) = std::sync::mpsc::sync_channel(1);
            let producer = scope.spawn(move || -> Result<f64, Error> {
                let mut busy = 0.0;
                for group in groups_of_pairs {
                    let t = Instant::now();
                    let tables = group
                        .into_par_iter()
                        .map(|((air, trace, _), shape)| table_of(air, trace, shape, release_rows))
                        .collect::<Result<Vec<_>, Error>>()?;
                    busy += t.elapsed().as_secs_f64();
                    if tx.send(tables).is_err() {
                        break;
                    }
                }
                Ok(busy)
            });
            let block = BlockCommitted::commit_streamed::<H>(
                rx.iter(),
                &sizes,
                &config,
                options.drop_levels,
            );
            (block, producer.join())
        });
        // The producer's error names the cause; a commit that ran out of
        // groups only says that it did.
        let produced =
            produced.map_err(|_| Error::Prover("the block's table producer panicked".into()))??;
        let block = block.map_err(|e| Error::Prover(format!("{e:?}")))?;
        stamps.prep += produced;
        stamps.phase_a = t.elapsed().as_secs_f64();
        let (prepared, derive) = prover_prepared::<H>(prepared_columns, &config, deviations)?;
        stamps.prepared = (prepared.len(), derive);
        let positions = prepared_positions(&prepared, &groups)?;
        let borrowed: Vec<Vec<&Mle<F>>> = prepared
            .iter()
            .map(|p| multilinear::stacking::borrow(&p.columns))
            .collect();
        let openings: Vec<multilinear_block::BlockPrepared<'_, F, H>> = positions
            .iter()
            .map(|&(table, k)| multilinear_block::BlockPrepared {
                table,
                commitment: &prepared[k].commitment,
                columns: &borrowed[k],
            })
            .collect();
        let t = Instant::now();
        let paths_before = multilinear::whir_commit::top_path_counts();
        let identity = |g: usize| g;
        let fork_of: &dyn Fn(usize) -> usize = match &deviations.fork_of {
            Some(f) => f,
            None => &identity,
        };
        let (proof, prepared_openings, groups) =
            multilinear_block::block_prove_on_forks::<_, _, _, H>(
                block,
                &config,
                &mut transcript,
                &openings,
                fork_of,
            )
            .map_err(|e| Error::Prover(format!("{e:?}")))?;
        stamps.phase_b = t.elapsed().as_secs_f64();
        let paths_after = multilinear::whir_commit::top_path_counts();
        stamps.top_paths = (
            paths_after.0 - paths_before.0,
            paths_after.1 - paths_before.1,
        );
        stamps.groups = groups;
        (proof, prepared_openings)
    });

    Ok(BlockWhirProof {
        proof: proof.0,
        table_num_vars,
        runtime_page_ranges,
        table_counts,
        public_output,
        num_private_input_pages,
        groups,
        prepared: proof.1,
    })
}

/// One AIR per streamed table type, alive for the whole prove: a streamed
/// chunk is laid out (and its layout borrows its AIR) before the run's AIR set
/// exists. Every chunk of a type has the same AIR as the run's `TYPE[i]` — the
/// set builds them all with the same constructor — and the verifier checks the
/// result against its own set.
struct StreamAirs {
    cpu: crate::VmAir,
    memw_register: crate::VmAir,
    memw_aligned: crate::VmAir,
    memw: crate::VmAir,
    load: crate::VmAir,
    lt: crate::VmAir,
    shift: crate::VmAir,
    store: crate::VmAir,
}

type DynAir = dyn stark::traits::AIR<Field = F, FieldExtension = E, PublicInputs = ()>;

impl StreamAirs {
    fn new(opts: &ProofOptions) -> Self {
        use crate::test_utils::*;
        Self {
            cpu: Box::new(create_cpu_air(opts)),
            memw_register: Box::new(create_memw_register_air(opts)),
            memw_aligned: Box::new(create_memw_aligned_air(opts)),
            memw: Box::new(create_memw_air(opts)),
            load: Box::new(create_load_air(opts)),
            lt: Box::new(create_lt_air(opts)),
            shift: Box::new(create_shift_air(opts)),
            store: Box::new(create_store_air(opts)),
        }
    }

    fn of(&self, table: StreamTable) -> &DynAir {
        match table {
            StreamTable::Cpu => self.cpu.as_ref(),
            StreamTable::MemwRegister => self.memw_register.as_ref(),
            StreamTable::MemwAligned => self.memw_aligned.as_ref(),
            StreamTable::Memw => self.memw.as_ref(),
            StreamTable::Load => self.load.as_ref(),
            StreamTable::Lt => self.lt.as_ref(),
            StreamTable::Shift => self.shift.as_ref(),
            StreamTable::Store => self.store.as_ref(),
        }
    }
}

/// The name the run's AIR set gives chunk `index` of a streamed table.
fn stream_name(table: StreamTable, index: usize) -> String {
    let base = match table {
        StreamTable::Cpu => "CPU",
        StreamTable::MemwRegister => "MEMW_R",
        StreamTable::MemwAligned => "MEMW_A",
        StreamTable::Memw => "MEMW",
        StreamTable::Load => "LOAD",
        StreamTable::Lt => "LT",
        StreamTable::Shift => "SHIFT",
        StreamTable::Store => "STORE",
    };
    format!("{base}[{index}]")
}

/// What the builder thread reports when the run is built: when its windows
/// were all collected and when the run was built (seconds since the prove
/// started), how many chunks it handed out, the windows' stamps and `finish`'s
/// phase marks.
type BuilderReport = (f64, f64, usize, WindowStamps, Vec<(String, f64)>);

/// What the builder thread hands the layout thread.
enum Built {
    Job(Box<ChunkJob>),
    Rest(Box<Traces>),
}

/// A table before the run's AIR order exists: a streamed chunk, or a table of
/// the final build by its AIR index.
#[derive(Clone, Copy)]
enum Key {
    Streamed(StreamTable, usize),
    Air(usize),
}

/// A table of the final build, laid out: its AIR index, shape and table.
type Placed<'a> = (usize, (usize, usize), CommittedTable<'a, F, E>);

/// Packs tables, in the order they come, into groups of at most `max_polys`
/// stacked polynomials, and sends each group to phase A as it closes.
struct Packer<'a> {
    open: Vec<CommittedTable<'a, F, E>>,
    open_shapes: Vec<(usize, usize)>,
    open_keys: Vec<Key>,
    group_keys: Vec<Vec<Key>>,
    out: std::sync::mpsc::SyncSender<Vec<CommittedTable<'a, F, E>>>,
    cap: multilinear::whir_chain::StackVars,
    max_polys: usize,
}

impl<'a> Packer<'a> {
    /// A group closes, and goes to the card, when the next table would take
    /// it past its polynomial budget.
    fn place(
        &mut self,
        key: Key,
        shape: (usize, usize),
        table: CommittedTable<'a, F, E>,
    ) -> Result<(), Error> {
        self.open_shapes.push(shape);
        let polys = stark::multilinear_table::global_layout(&self.open_shapes, self.cap)
            .map_err(|e| Error::Prover(format!("{e:?}")))?
            .num_polys();
        if polys > self.max_polys && !self.open.is_empty() {
            self.open_shapes.clear();
            self.open_shapes.push(shape);
            self.close()?;
        }
        self.open.push(table);
        self.open_keys.push(key);
        Ok(())
    }

    fn close(&mut self) -> Result<(), Error> {
        self.group_keys.push(std::mem::take(&mut self.open_keys));
        self.out
            .send(std::mem::take(&mut self.open))
            .map_err(|_| Error::Prover("phase A stopped".into()))
    }

    /// Sends the last group; hands back every group's tables.
    fn finish(mut self) -> Result<Vec<Vec<Key>>, Error> {
        if !self.open.is_empty() {
            self.close()?;
        }
        Ok(self.group_keys)
    }
}

/// What the layout thread knows once the run is built: the statement, the AIR
/// order's shapes, and each group's tables as AIR indices.
struct Laid {
    table_counts: TableCounts,
    runtime_page_ranges: Vec<RuntimePageRange>,
    num_private_input_pages: usize,
    public_output: Vec<u8>,
    shapes: Vec<(usize, usize)>,
    groups: Vec<Vec<u32>>,
    /// The prepared tables' columns ([`prepared_tables`]).
    prepared: Vec<PreparedColumns>,
    busy: f64,
}

/// The block's proof with the build streamed into phase A: a builder thread
/// collects the run in windows of `window` cycles ([`WindowedTraceBuilder`])
/// and hands each full chunk over as it completes; a layout thread lays each
/// one out and packs the tables, in arrival order, into groups of at most
/// `format.group_polys` stacked polynomials; this thread commits each group as
/// it closes. When the run is built the rest of its tables join the packing.
/// The groups are the statement's ([`BlockWhirProof::groups`]).
#[allow(clippy::too_many_arguments)]
fn prove_streamed(
    program: &Elf,
    elf_bytes: &[u8],
    private_inputs: &[u8],
    window: usize,
    opts: &ProofOptions,
    format: &BlockFormat,
    options: &BlockOptions,
    deviations: &Deviations,
    stamps: &mut BlockStamps,
) -> Result<BlockWhirProof, Error> {
    let stream_airs = StreamAirs::new(opts);
    let run_airs: std::sync::OnceLock<VmAirs> = std::sync::OnceLock::new();
    // Phase A's commits read the blowup, the fold schedule and the format of
    // the config and nothing else; the full config (its query count reads every
    // shape) is built when the shapes are known, and must agree on those.
    let commit_config = chain_config_under(&format.zf, &[]);
    let cap = commit_config.format.stack;
    let start = Instant::now();

    crate::with_whir_hash!(|H| {
        let (block, built, laid, executed) = std::thread::scope(|scope| {
            // The executor, a window at a time, two windows ahead of the walk.
            let (ltx, lrx) = std::sync::mpsc::sync_channel::<Vec<executor::vm::logs::Log>>(2);
            let executor = scope.spawn(move || -> Result<f64, Error> {
                let mut executor = Executor::new(program, private_inputs.to_vec())
                    .map_err(|e| Error::Execution(format!("{e}")))?;
                while let Some(logs) = executor
                    .resume_with_limit(window)
                    .map_err(|e| Error::Execution(format!("{e}")))?
                {
                    if ltx.send(logs.to_vec()).is_err() {
                        break;
                    }
                }
                Ok(start.elapsed().as_secs_f64())
            });
            let (btx, brx) = std::sync::mpsc::sync_channel::<Built>(64);
            let builder = scope.spawn(move || -> Result<BuilderReport, Error> {
                let mut builder =
                    WindowedTraceBuilder::new(program, private_inputs, &options.max_rows)?;
                let mut streamed = 0usize;
                // The walk on its own thread, doing nothing but walk; this
                // thread appends each walked window, routes it and hands its
                // chunks out as jobs (the layout thread generates them).
                let last = {
                    let (mut walker, mut accumulator) = builder.split();
                    std::thread::scope(|inner| -> Result<Vec<executor::vm::logs::Log>, Error> {
                        let (wtx, wrx) = std::sync::mpsc::sync_channel(2);
                        let walking = inner.spawn(move || -> Result<_, Error> {
                            // One window held back: only the run's last window
                            // is `finish`'s, and it is the last only once the
                            // executor stops.
                            let mut held: Option<Vec<executor::vm::logs::Log>> = None;
                            for logs in lrx {
                                if let Some(w) = held.replace(logs)
                                    && wtx.send(walker.walk(&w)?).is_err()
                                {
                                    break;
                                }
                            }
                            held.ok_or_else(|| {
                                Error::Execution("the run executed no cycle".to_string())
                            })
                        });
                        for walked in wrx {
                            for job in accumulator.absorb(walked) {
                                streamed += 1;
                                if btx.send(Built::Job(Box::new(job))).is_err() {
                                    return Err(Error::Prover("the layout thread stopped".into()));
                                }
                            }
                        }
                        walking
                            .join()
                            .map_err(|_| Error::Prover("the block's walker panicked".into()))?
                    })?
                };
                let windows_done = start.elapsed().as_secs_f64();
                let window_stamps = builder.stamps();
                // The table phase's marks, for `finish` alone.
                crate::tables::trace_builder::build_stamps::start();
                let mut rest = builder.finish(&last)?;
                let finish_marks = crate::tables::trace_builder::build_stamps::take();
                split_keccak_rnd(&mut rest, options.keccak_rnd_rows_log2);
                if deviations.omit_first_keccak_rnd && !rest.keccak_rnds.is_empty() {
                    rest.keccak_rnds.remove(0);
                }
                let finished = start.elapsed().as_secs_f64();
                let _ = btx.send(Built::Rest(Box::new(rest)));
                Ok((
                    windows_done,
                    finished,
                    streamed,
                    window_stamps,
                    finish_marks,
                ))
            });

            let (gtx, grx) = std::sync::mpsc::sync_channel::<Vec<CommittedTable<'_, F, E>>>(1);
            let stream_airs = &stream_airs;
            let run_airs = &run_airs;
            let layout = scope.spawn(move || -> Result<Laid, Error> {
                let mut busy = 0.0;
                let mut packer = Packer {
                    open: Vec::new(),
                    open_shapes: Vec::new(),
                    open_keys: Vec::new(),
                    group_keys: Vec::new(),
                    out: gtx,
                    cap,
                    max_polys: format.group_polys,
                };
                let mut streamed_shapes: Vec<(StreamTable, usize, (usize, usize))> = Vec::new();
                let mut rest = None;
                for item in brx {
                    match item {
                        Built::Job(job) => {
                            let t = Instant::now();
                            let mut chunk = job.generate();
                            let height = chunk.trace.main_table.height;
                            let shape = (
                                chunk.trace.main_table.width,
                                height.trailing_zeros() as usize,
                            );
                            let table = table_of(
                                stream_airs.of(chunk.table),
                                &mut chunk.trace,
                                shape,
                                true,
                            )?;
                            streamed_shapes.push((chunk.table, chunk.index, shape));
                            packer.place(Key::Streamed(chunk.table, chunk.index), shape, table)?;
                            busy += t.elapsed().as_secs_f64();
                        }
                        Built::Rest(traces) => {
                            rest = Some(traces);
                            break;
                        }
                    }
                }
                let mut traces = *rest.ok_or_else(|| {
                    Error::Prover("the builder stopped before the run was built".into())
                })?;
                let t = Instant::now();
                let table_counts = traces.table_counts();
                validate_block_counts(&table_counts)?;
                let runtime_page_ranges = traces.runtime_page_ranges();
                let num_private_input_pages = traces
                    .page_configs
                    .iter()
                    .filter(|c| c.is_private_input)
                    .count();
                let public_output = traces.public_output_bytes.clone();
                let airs = run_airs.get_or_init(|| {
                    VmAirs::new(
                        program,
                        opts,
                        false,
                        &traces.page_configs,
                        &table_counts,
                        None,
                        true,
                        None,
                        None,
                        None,
                    )
                });
                let prepared = prepared_tables(airs, &traces.page_configs)?;
                let refs = airs.air_refs();
                let names: std::collections::HashMap<String, usize> = refs
                    .iter()
                    .enumerate()
                    .map(|(i, air)| (air.name().to_string(), i))
                    .collect();
                let mut shapes: Vec<Option<(usize, usize)>> = vec![None; refs.len()];
                for &(table, index, shape) in &streamed_shapes {
                    let at = *names.get(&stream_name(table, index)).ok_or_else(|| {
                        Error::Prover(format!("no AIR for {}", stream_name(table, index)))
                    })?;
                    shapes[at] = Some(shape);
                }
                // The tables the windows did not stream, laid out in parallel
                // and packed in AIR order.
                let pairs = airs.air_trace_pairs(&mut traces);
                let rest_tables: Vec<Placed<'_>> = pairs
                    .into_iter()
                    .zip(refs.iter().copied())
                    .enumerate()
                    .filter(|(_, ((_, trace, _), _))| trace.main_table.width != 0)
                    .collect::<Vec<_>>()
                    .into_par_iter()
                    .map(|(i, ((_, trace, _), air))| {
                        let shape = (
                            trace.main_table.width,
                            trace.main_table.height.trailing_zeros() as usize,
                        );
                        table_of(air, trace, shape, true).map(|table| (i, shape, table))
                    })
                    .collect::<Result<_, Error>>()?;
                for (i, shape, table) in rest_tables {
                    if shapes[i].replace(shape).is_some() {
                        return Err(Error::Prover(format!("table {i} built twice")));
                    }
                    packer.place(Key::Air(i), shape, table)?;
                }
                let group_keys = packer.finish()?;
                let shapes: Vec<(usize, usize)> = shapes
                    .into_iter()
                    .enumerate()
                    .map(|(i, s)| s.ok_or_else(|| Error::Prover(format!("table {i} never built"))))
                    .collect::<Result<_, _>>()?;
                for (air, &(width, _)) in refs.iter().zip(&shapes) {
                    if air.trace_layout().0 != width {
                        return Err(Error::Prover(format!(
                            "{}: {width} columns, the AIR declares {}",
                            air.name(),
                            air.trace_layout().0
                        )));
                    }
                }
                let groups = group_keys
                    .into_iter()
                    .map(|keys| {
                        keys.into_iter()
                            .map(|key| match key {
                                Key::Air(i) => Ok(i as u32),
                                Key::Streamed(table, index) => names
                                    .get(&stream_name(table, index))
                                    .map(|&i| i as u32)
                                    .ok_or_else(|| {
                                        Error::Prover(format!(
                                            "no AIR for {}",
                                            stream_name(table, index)
                                        ))
                                    }),
                            })
                            .collect::<Result<Vec<u32>, Error>>()
                    })
                    .collect::<Result<Vec<_>, Error>>()?;
                busy += t.elapsed().as_secs_f64();
                Ok(Laid {
                    table_counts,
                    runtime_page_ranges,
                    num_private_input_pages,
                    public_output,
                    shapes,
                    groups,
                    prepared,
                    busy,
                })
            });

            let block =
                BlockCommitted::commit_groups::<H>(grx.iter(), &commit_config, options.drop_levels);
            (block, builder.join(), layout.join(), executor.join())
        });
        stamps.execute =
            executed.map_err(|_| Error::Prover("the block's executor panicked".into()))??;
        // A thread's error names the cause; a commit that ran out of groups only
        // says that it did.
        let (windows_done, finished, streamed, window_stamps, finish_marks) =
            built.map_err(|_| Error::Prover("the block's builder panicked".into()))??;
        stamps.windows = window_stamps;
        stamps.build_marks = finish_marks;
        let laid =
            laid.map_err(|_| Error::Prover("the block's layout thread panicked".into()))??;
        let block = block.map_err(|e| Error::Prover(format!("{e:?}")))?;
        stamps.build = finished;
        stamps.streamed = (windows_done, streamed);
        stamps.prep = laid.busy;
        stamps.phase_a = start.elapsed().as_secs_f64();
        stamps.tables = laid.shapes.len();
        stamps.cells = laid.shapes.iter().map(|&(w, n)| w << n).sum();

        let config = chain_config_under(&format.zf, &laid.shapes);
        if config.log_blowup != commit_config.log_blowup
            || config.log_folding != commit_config.log_folding
            || config.format != commit_config.format
        {
            return Err(Error::Prover(
                "phase A committed under a config the block's shapes do not give".into(),
            ));
        }
        let table_num_vars: Vec<u8> = laid.shapes.iter().map(|&(_, n)| n as u8).collect();
        let mut transcript =
            DefaultTranscript::<E, <H as multilinear::whir_hash::WhirHash>::Transcript>::new(&[]);
        absorb_block(
            &mut transcript,
            elf_bytes,
            &laid.public_output,
            &laid.table_counts,
            laid.num_private_input_pages,
            &laid.runtime_page_ranges,
            &table_num_vars,
            &config,
            &laid.groups,
        );
        let (prepared, derive) = prover_prepared::<H>(laid.prepared, &config, deviations)?;
        stamps.prepared = (prepared.len(), derive);
        let positions = prepared_positions(&prepared, &laid.groups)?;
        let borrowed: Vec<Vec<&Mle<F>>> = prepared
            .iter()
            .map(|p| multilinear::stacking::borrow(&p.columns))
            .collect();
        let openings: Vec<multilinear_block::BlockPrepared<'_, F, H>> = positions
            .iter()
            .map(|&(table, k)| multilinear_block::BlockPrepared {
                table,
                commitment: &prepared[k].commitment,
                columns: &borrowed[k],
            })
            .collect();
        let t = Instant::now();
        let paths_before = multilinear::whir_commit::top_path_counts();
        let identity = |g: usize| g;
        let fork_of: &dyn Fn(usize) -> usize = match &deviations.fork_of {
            Some(f) => f,
            None => &identity,
        };
        let (proof, prepared_openings, groups) =
            multilinear_block::block_prove_on_forks::<_, _, _, H>(
                block,
                &config,
                &mut transcript,
                &openings,
                fork_of,
            )
            .map_err(|e| Error::Prover(format!("{e:?}")))?;
        stamps.phase_b = t.elapsed().as_secs_f64();
        let paths_after = multilinear::whir_commit::top_path_counts();
        stamps.top_paths = (
            paths_after.0 - paths_before.0,
            paths_after.1 - paths_before.1,
        );
        stamps.groups = groups;
        Ok(BlockWhirProof {
            proof,
            table_num_vars,
            runtime_page_ranges: laid.runtime_page_ranges,
            table_counts: laid.table_counts,
            public_output: laid.public_output,
            num_private_input_pages: laid.num_private_input_pages,
            groups: laid.groups,
            prepared: prepared_openings,
        })
    })
}

/// Verifies a proof from [`prove_block_whir`] under `format` — the verifier's
/// own constants, never the proof's.
pub fn verify_block_whir(
    proof: &BlockWhirProof,
    elf_bytes: &[u8],
    proof_options: &ProofOptions,
    format: &BlockFormat,
) -> Result<bool, Error> {
    let program = Elf::load(elf_bytes).map_err(|e| Error::ElfLoad(format!("{e}")))?;

    validate_block_counts(&proof.table_counts)?;
    let max_pages = crate::tables::page::max_private_input_pages();
    if proof.num_private_input_pages > max_pages {
        return Err(Error::InvalidTableCounts(format!(
            "num_private_input_pages ({}) exceeds max ({max_pages})",
            proof.num_private_input_pages,
        )));
    }
    let page_configs = Traces::page_configs_from_elf_and_runtime(
        &program,
        &proof.runtime_page_ranges,
        proof.num_private_input_pages,
        proof.proof.tables.len(),
    )?;
    let Some(expected) = proof
        .table_counts
        .total()
        .and_then(|t| t.checked_add(FIXED_TABLE_COUNT))
        .and_then(|t| t.checked_add(page_configs.len()))
    else {
        return Err(Error::InvalidTableCounts(
            "the table counts do not sum to a usize".to_string(),
        ));
    };
    if expected != proof.proof.tables.len() || proof.table_num_vars.len() != expected {
        return Err(Error::InvalidTableCounts(format!(
            "the statement implies {expected} tables; the proof carries {} and {} heights",
            proof.proof.tables.len(),
            proof.table_num_vars.len(),
        )));
    }

    let airs = VmAirs::new(
        &program,
        proof_options,
        false,
        &page_configs,
        &proof.table_counts,
        None,
        true,
        None,
        None,
        None,
    );
    let air_refs = airs.air_refs();
    if air_refs.len() != proof.proof.tables.len() {
        return Err(Error::InvalidTableCounts(format!(
            "the layout has {} tables, the proof carries {}",
            air_refs.len(),
            proof.proof.tables.len(),
        )));
    }
    // The width is the AIR's, never the proof's; only the height is stated.
    let shapes: Vec<(usize, usize)> = air_refs
        .iter()
        .zip(&proof.table_num_vars)
        .map(|(air, &num_vars)| (air.trace_layout().0, num_vars as usize))
        .collect();
    let config = chain_config_under(&format.zf, &shapes);
    let layouts: Vec<TableLayout<'_, F, E>> = air_refs
        .iter()
        .zip(&shapes)
        .map(|(air, &(width, num_vars))| {
            layout_of(*air, width, num_vars)
                .map_err(|e| Error::Prover(format!("{}: {e:?}", air.name())))
        })
        .collect::<Result<_, _>>()?;
    let preprocessed: Vec<Vec<Mle<F>>> = air_refs
        .iter()
        .map(|air| preprocessed_mles(*air))
        .collect::<Result<_, _>>()?;
    let statements: Vec<TableStatement<'_, F, E>> = layouts
        .iter()
        .zip(&preprocessed)
        .map(|(layout, cols)| layout.statement_with_preprocessed(cols))
        .collect();

    // The groups are the statement's; their stacks are built here, from the
    // shapes and the verifier's stack cap. The proof's tables are in group
    // order, so the statements are taken in that order too.
    if proof.groups.len() > format.max_groups {
        return Err(Error::InvalidTableCounts(format!(
            "{} groups — a block takes at most {}",
            proof.groups.len(),
            format.max_groups
        )));
    }
    let order = validate_groups(&proof.groups, shapes.len())?;
    let sizes: Vec<usize> = proof.groups.iter().map(Vec::len).collect();
    let group_shapes: Vec<(usize, usize)> = order.iter().map(|&i| shapes[i]).collect();
    let statements: Vec<TableStatement<'_, F, E>> = order.iter().map(|&i| statements[i]).collect();
    let (stack_layouts, domains) = stacks(&group_shapes, &sizes, &config)?;
    let prepared_columns = prepared_tables(&airs, &page_configs)?;
    // Every group within the stack budget — the verifier's constant, not the
    // prover's packing — except a table that alone needs more: it cannot be
    // split, so it is a group of its own (the packing's rule too).
    if let Some((g, layout)) = stack_layouts
        .iter()
        .enumerate()
        .find(|(g, layout)| layout.num_polys() > format.group_polys && sizes[*g] > 1)
    {
        return Err(Error::InvalidTableCounts(format!(
            "group {g} stacks into {} polynomials — a group takes at most {}",
            layout.num_polys(),
            format.group_polys
        )));
    }

    Ok(crate::with_whir_hash!(|H| {
        let mut transcript =
            DefaultTranscript::<E, <H as multilinear::whir_hash::WhirHash>::Transcript>::new(&[]);
        absorb_block(
            &mut transcript,
            elf_bytes,
            &proof.public_output,
            &proof.table_counts,
            proof.num_private_input_pages,
            &proof.runtime_page_ranges,
            &proof.table_num_vars,
            &config,
            &proof.groups,
        );
        // The prepared commitments, derived here from the program: their roots
        // are the verifier's, never the proof's.
        let prepared: Vec<TablePrepared<H>> = match prepared_columns
            .into_iter()
            .map(|(table, columns)| commit_prepared::<H>(table, columns, &config))
            .collect::<Result<Vec<_>, _>>()
        {
            Ok(prepared) => prepared,
            Err(_) => return Ok(false),
        };
        let Ok(positions) = prepared_positions(&prepared, &proof.groups) else {
            return Ok(false);
        };
        let checks: Vec<multilinear_block::BlockPreparedCheck<'_, F>> = positions
            .iter()
            .map(|&(table, k)| multilinear_block::BlockPreparedCheck {
                table,
                roots: &prepared[k].roots,
                layout: prepared[k].commitment.layout(),
                domain: prepared[k].commitment.domain(),
            })
            .collect();
        let derived: Vec<multilinear::whir_commit::Commitment> = checks
            .iter()
            .flat_map(|c| c.roots.iter().copied())
            .collect();
        // What the tables owe: the COMMIT bus's counterparty, at the block's
        // challenges — replayed on a fork through the roots block itself.
        let mut probe = transcript.clone();
        stark::multilinear_table::absorb_roots::<E, _>(&mut probe, &proof.proof.roots, &derived);
        let z: FieldElement<E> = probe.sample_field_element();
        let alpha: FieldElement<E> = probe.sample_field_element();
        let Some(owed) = crate::compute_commit_bus_offset(&proof.public_output, 0, &z, &alpha)
        else {
            return Ok(false);
        };
        multilinear_block::block_verify::<_, _, _, H>(
            &proof.proof,
            &proof.prepared,
            &checks,
            &statements,
            &stack_layouts,
            &domains,
            &sizes,
            &owed,
            &config,
            &mut transcript,
        )
        .is_ok()
    }))
}
