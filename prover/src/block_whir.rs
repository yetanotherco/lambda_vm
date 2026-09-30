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
use crate::tables::trace_builder::Traces;
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
    pub group_polys: usize,
}

impl BlockFormat {
    /// The process's WHIR format (as [`crate::multilinear_prove::chain_config`]
    /// reads it) and [`BLOCK_GROUP_POLYS`].
    pub fn production() -> Self {
        Self {
            zf: *ZfFormat::global(),
            group_polys: BLOCK_GROUP_POLYS,
        }
    }
}

/// The prover's own choices: none of them is in the proof.
#[derive(Clone, Debug)]
pub struct BlockOptions {
    pub max_rows: MaxRowsConfig,
    pub keccak_rnd_rows_log2: usize,
    pub drop_levels: usize,
}

impl BlockOptions {
    /// Every chunked table at 2^21 rows, KECCAK_RND at 2^16.
    pub fn production() -> Self {
        Self {
            max_rows: MaxRowsConfig::uniform(1 << 21),
            keccak_rnd_rows_log2: BLOCK_KECCAK_RND_ROWS_LOG2,
            drop_levels: BLOCK_TREE_DROP_LEVELS,
        }
    }
}

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
                "BLOCK GROUP {g}: tables {} · polys {} · cells {:.1} M || A wait {:.3} upload {:.3} commit {:.3} retire {:.3} || B upload {:.3} argue {:.3} encode {:.3} open {:.3}\n",
                s.tables,
                s.polys,
                s.cells as f64 / 1e6,
                s.wait_a,
                s.upload_a,
                s.commit,
                s.retire,
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
        out.push_str(&format!(
            "BLOCK RECOMMIT: 0 (a revived commitment builds no tree) · first-round paths from kept tops: {} calls · {} leaves re-hashed on the host · phase B on one thread + one upload helper\n",
            self.top_paths.0, self.top_paths.1,
        ));
        let argue = sum(|g| g.argue);
        let open = sum(|g| g.open);
        let tax = sum(|g| g.upload_b + g.encode);
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

/// A table cut into tables of `rows` rows each, in row order. Every VM table's
/// constraints read one row (no shifted reads), so each piece is a table of
/// the same AIR, and the bus sums over the pieces are the sum over the whole.
fn split_rows(table: TraceTable<F, E>, rows: usize) -> Vec<TraceTable<F, E>> {
    let height = table.main_table.height;
    if height <= rows {
        return vec![table];
    }
    let step = table.step_size;
    let columns = table.columns_main();
    drop(table);
    (0..height / rows)
        .map(|k| {
            TraceTable::from_columns_main(
                columns
                    .iter()
                    .map(|column| column[k * rows..(k + 1) * rows].to_vec())
                    .collect(),
                step,
            )
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
    let mut columns = trace.columns_main();
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
    let proof = crate::with_whir_hash!(|H| {
        let mut transcript =
            DefaultTranscript::<E, <H as multilinear::whir_hash::WhirHash>::Transcript>::new(&[]);
        absorb_tagged(
            &mut transcript,
            MULTILINEAR_BLOCK_TAG,
            "block",
            &statement::elf_digest(elf_bytes),
            &public_output,
            &table_counts,
            num_private_input_pages,
            &runtime_page_ranges,
            &table_num_vars,
            &config,
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
        let t = Instant::now();
        let paths_before = multilinear::whir_commit::top_path_counts();
        let identity = |g: usize| g;
        let fork_of: &dyn Fn(usize) -> usize = match &deviations.fork_of {
            Some(f) => f,
            None => &identity,
        };
        let (proof, groups) = multilinear_block::block_prove_on_forks::<_, _, _, H>(
            block,
            &config,
            &mut transcript,
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
        proof
    });

    Ok(BlockWhirProof {
        proof,
        table_num_vars,
        runtime_page_ranges,
        table_counts,
        public_output,
        num_private_input_pages,
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

    // The groups and their stacks, from the shapes and the format alone.
    let sizes = block_groups(&shapes, config.format.stack, format.group_polys)
        .map_err(|e| Error::Prover(format!("{e:?}")))?;
    let (stack_layouts, domains) = stacks(&shapes, &sizes, &config)?;

    Ok(crate::with_whir_hash!(|H| {
        let mut transcript =
            DefaultTranscript::<E, <H as multilinear::whir_hash::WhirHash>::Transcript>::new(&[]);
        absorb_tagged(
            &mut transcript,
            MULTILINEAR_BLOCK_TAG,
            "block",
            &statement::elf_digest(elf_bytes),
            &proof.public_output,
            &proof.table_counts,
            proof.num_private_input_pages,
            &proof.runtime_page_ranges,
            &proof.table_num_vars,
            &config,
        );
        // What the tables owe: the COMMIT bus's counterparty, at the block's
        // challenges — replayed on a fork through the roots block itself.
        let mut probe = transcript.clone();
        stark::multilinear_table::absorb_roots::<E, _>(&mut probe, &proof.proof.roots, &[]);
        let z: FieldElement<E> = probe.sample_field_element();
        let alpha: FieldElement<E> = probe.sample_field_element();
        let Some(owed) = crate::compute_commit_bus_offset(&proof.public_output, 0, &z, &alpha)
        else {
            return Ok(false);
        };
        multilinear_block::block_verify::<_, _, _, H>(
            &proof.proof,
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
