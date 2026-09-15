//! The VM proved with the multilinear argument, end to end.
//!
//! Same executor, same traces and the same AIRs as [`crate::prove`]; what
//! changes is how each table is argued. Instead of a composition polynomial
//! committed by FRI, every table's own constraints and every one of its buses
//! are discharged in **one** sumcheck against one WHIR commitment for its whole
//! trace, and the LogUp argument is a fraction tree rather than auxiliary
//! columns.
//!
//! The verifier here never sees a trace. It rebuilds each table's
//! [`TableLayout`] from the AIR and the table's shape, which is the same
//! construction the prover ran, and takes the commitment roots from the proof.
//!
//! # How the preprocessed tables are bound
//!
//! BITWISE, DECODE, KECCAK_RC, REGISTER and the PAGE tables have
//! **preprocessed** columns — columns `0..num_precomputed_columns()`, fully
//! determined by the program and known to both sides. Nothing in the argument
//! itself pins them: a commitment says the prover stayed consistent with what
//! it committed, not that it committed the right thing, and a forged bitwise
//! table would otherwise prove forged lookups.
//!
//! The univariate path pins them with a second Merkle root
//! ([`AIR::precomputed_commitment`]) the verifier recomputes from the ELF. Here
//! there is no second root to compare, so the verifier does it one step later:
//! it rebuilds the columns ([`AIR::precomputed_columns`]) and evaluates its own
//! copy at the point the proof settled on, demanding the value the proof claims
//! for that column. One pass per column, which is what recomputing that root
//! costs anyway. **This is what binds a proof to its program** — DECODE's
//! preprocessed columns *are* the instruction table, and REGISTER's carry the
//! entry point.
//!
//! **The bus balance is not zero.** The COMMIT table sends the program's public
//! output on a bus whose counterparty is the statement rather than another
//! table, so what the tables must sum to is
//! [`compute_commit_bus_offset`](crate::compute_commit_bus_offset) — computed
//! against the very challenges the verifier draws, which means replaying the
//! transcript up to that point.

use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use crypto::fiat_shamir::is_transcript::IsTranscript;
use executor::elf::Elf;
use executor::vm::execution::Executor;
use math::field::element::FieldElement;
use multilinear::mle::Mle;
use multilinear::whir_chain::{ChainConfig, GrindBits};
use stark::multilinear_air::Uniforms;
use stark::multilinear_table::{
    self, CommittedTable, CommittedTables, MultiProof, TableLayout, TableStatement,
};
use stark::traits::AIR;

use crate::statement::{self, MULTILINEAR_TAG};
use crate::tables::trace_builder::Traces;
use crate::test_utils::{E, F};
use crate::{
    Error, FIXED_TABLE_COUNT, MaxRowsConfig, ProofOptions, RuntimePageRange, TableCounts, VmAirs,
};

/// One table's height and main width — the shape the verifier needs to rebuild
/// its layout, and all it needs.
type Shape = (usize, usize);

/// The multilinear counterpart of [`crate::VmProof`].
///
/// Carries the same statement metadata, plus every table's height. The
/// univariate path reads heights off its FRI domains; here they are stated, and
/// bound into the transcript so restating them changes every challenge.
#[derive(Debug, Clone, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct MultilinearVmProof {
    /// Every table's argument and the one opening that settles all of them,
    /// in [`VmAirs::air_refs`] order.
    pub proof: MultiProof<F, E>,
    /// Each table's height in variables, same order.
    pub table_num_vars: Vec<u8>,
    pub runtime_page_ranges: Vec<RuntimePageRange>,
    pub table_counts: TableCounts,
    pub public_output: Vec<u8>,
    pub num_private_input_pages: usize,
}

/// The blowup, fold factor and grinding the whole VM proof runs at.
///
/// The query count comes from the tallest stacked polynomial in the proof, so
/// one config covers every table: a taller stack means more rounds, and more
/// rounds is what the union bound charges for.
pub fn chain_config(shapes: &[Shape]) -> ChainConfig {
    let tallest = shapes
        .iter()
        .map(|&(width, num_vars)| multilinear::constraint_argument::one_stack(num_vars, width))
        .max()
        .unwrap_or(1);
    ChainConfig::with_security(2, 4, tallest, 128, GrindBits::uniform(20))
}

/// Binds the statement into the transcript before any challenge is drawn.
///
/// The univariate encoding, under a tag of its own — a WHIR proof and a FRI
/// proof must never share a transcript prefix — followed by what only this path
/// states: the table heights and the parameters the argument runs at.
#[allow(clippy::too_many_arguments)]
fn absorb(
    t: &mut DefaultTranscript<E>,
    elf_digest: &[u8; 32],
    public_output: &[u8],
    table_counts: &TableCounts,
    num_private_input_pages: usize,
    runtime_page_ranges: &[RuntimePageRange],
    table_num_vars: &[u8],
    config: &ChainConfig,
) {
    t.append_bytes(MULTILINEAR_TAG);
    t.append_bytes(elf_digest);

    t.append_bytes(&(public_output.len() as u64).to_le_bytes());
    t.append_bytes(public_output);

    statement::absorb_table_counts(t, table_counts);
    t.append_bytes(&(num_private_input_pages as u64).to_le_bytes());

    t.append_bytes(&(runtime_page_ranges.len() as u64).to_le_bytes());
    for r in runtime_page_ranges {
        let &RuntimePageRange { base, count } = r;
        t.append_bytes(&base.to_le_bytes());
        t.append_bytes(&count.to_le_bytes());
    }

    // Every table's height. A prover who shrank a table would have to state the
    // smaller height here, which moves every challenge.
    t.append_bytes(&(table_num_vars.len() as u64).to_le_bytes());
    t.append_bytes(table_num_vars);

    // The parameters the argument runs at: derived from the heights above, but
    // absorbed rather than assumed, so the two sides agree in the transcript and
    // not only in the code.
    let &ChainConfig {
        log_blowup,
        log_folding,
        num_queries,
        grind,
    } = config;
    for value in [log_blowup as u64, log_folding as u64, num_queries as u64] {
        t.append_bytes(&value.to_le_bytes());
    }
    t.append_bytes(&[grind.folding, grind.ood, grind.query]);
}

/// Every table's `(main width, height in variables)`, checked to be what the
/// AIR says and a power of two tall.
fn shapes_of(pairs: &[crate::AirTracePair<'_>]) -> Result<Vec<Shape>, Error> {
    pairs
        .iter()
        .map(|(air, trace, _)| {
            // The table's own shape, not its columns: transposing the trace to
            // count it is the whole trace copied for two numbers.
            let width = trace.main_table.width;
            let rows = trace.main_table.height;
            if width == 0 || !rows.is_power_of_two() {
                return Err(Error::Prover(format!(
                    "{}: {width} columns of {rows} rows, which the hypercube cannot hold",
                    air.name(),
                )));
            }
            if width != air.trace_layout().0 {
                return Err(Error::Prover(format!(
                    "{}: trace has {width} main columns, the AIR declares {}",
                    air.name(),
                    air.trace_layout().0,
                )));
            }
            Ok((width, rows.trailing_zeros() as usize))
        })
        .collect()
}

/// Proves an ELF the multilinear way.
pub fn prove_with_options_and_inputs(
    elf_bytes: &[u8],
    private_inputs: &[u8],
    proof_options: &ProofOptions,
    max_rows: &MaxRowsConfig,
) -> Result<MultilinearVmProof, Error> {
    let program = Elf::load(elf_bytes).map_err(|e| Error::ElfLoad(format!("{e}")))?;
    let result = Executor::new(&program, private_inputs.to_vec())
        .map_err(|e| Error::Execution(format!("{e}")))?
        .run()
        .map_err(|e| Error::Execution(format!("{e}")))?;

    let mut traces = Traces::from_elf_and_logs(
        &program,
        &result.logs,
        max_rows,
        private_inputs,
        #[cfg(feature = "disk-spill")]
        stark::storage_mode::StorageMode::Ram,
    )?;
    drop(result);

    let table_counts = traces.table_counts();
    let runtime_page_ranges = traces.runtime_page_ranges();
    let num_private_input_pages = traces
        .page_configs
        .iter()
        .filter(|c| c.is_private_input)
        .count();
    let public_output = traces.public_output_bytes.clone();

    let airs = VmAirs::new(
        &program,
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

    let mut pairs = airs.air_trace_pairs(&mut traces);
    let shapes = shapes_of(&pairs)?;
    let table_num_vars: Vec<u8> = shapes.iter().map(|&(_, n)| n as u8).collect();
    let config = chain_config(&shapes);

    let mut transcript = DefaultTranscript::<E>::new(&[]);
    absorb(
        &mut transcript,
        &statement::elf_digest(elf_bytes),
        &public_output,
        &table_counts,
        num_private_input_pages,
        &runtime_page_ranges,
        &table_num_vars,
        &config,
    );

    // Commit every table against the layout the verifier will rebuild.
    let mut committed = Vec::with_capacity(pairs.len());
    for ((air, trace, _), &(width, num_vars)) in pairs.iter_mut().zip(&shapes) {
        let layout = layout_of(*air, width, num_vars)
            .map_err(|e| Error::Prover(format!("{}: {e:?}", air.name())))?;
        let mut columns = trace.columns_main();
        // The verifier will rebuild these and demand the proof open to them, so a
        // trace that disagrees produces a proof nobody can verify. Better to say
        // so here than to hand out that proof.
        for (col, expected) in air.precomputed_columns().iter().enumerate() {
            if columns.get(col) != Some(expected) {
                return Err(Error::Prover(format!(
                    "{}: preprocessed column {col} is not what the program implies",
                    air.name(),
                )));
            }
        }
        committed.push(
            // Moved, not cloned: `columns` is this iteration's own transpose
            // of the trace and nothing reads it afterwards.
            CommittedTable::from_layout(layout, |col| core::mem::take(&mut columns[col as usize]))
                .map_err(|e| Error::Prover(format!("{}: {e:?}", air.name())))?,
        );
    }

    // One commitment for every table in the proof: the opening is nearly all of
    // a proof's bytes, and one settles them all.
    let committed =
        CommittedTables::commit(committed, &config).map_err(|e| Error::Prover(format!("{e:?}")))?;
    let proof = multilinear_table::multi_prove(&committed, &config, &mut transcript)
        .map_err(|e| Error::Prover(format!("{e:?}")))?;

    Ok(MultilinearVmProof {
        proof,
        table_num_vars,
        runtime_page_ranges,
        table_counts,
        public_output,
        num_private_input_pages,
    })
}

/// [`prove_with_options_and_inputs`] with no private input.
pub fn prove_with_options(
    elf_bytes: &[u8],
    proof_options: &ProofOptions,
    max_rows: &MaxRowsConfig,
) -> Result<MultilinearVmProof, Error> {
    prove_with_options_and_inputs(elf_bytes, &[], proof_options, max_rows)
}

/// One table's layout, from the AIR and its shape alone. Both sides call this.
fn layout_of<'a>(
    air: &'a dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>,
    width: usize,
    num_vars: usize,
) -> Result<TableLayout<'a, F, E>, multilinear::Error> {
    TableLayout::<F, E>::new(
        air.constraint_program(),
        air.constraints_meta(),
        air.bus_interactions(),
        width,
        num_vars,
        Uniforms::default(),
    )
}

/// Each commitment group's stack and the domain it is committed over, rebuilt
/// from the shapes and the split alone — the verifier never takes either from
/// the proof.
pub(crate) fn stacks(
    shapes: &[Shape],
    sizes: &[usize],
    config: &ChainConfig,
) -> Result<
    (
        Vec<multilinear::stacking::StackedLayout>,
        Vec<multilinear::whir::Domain<F>>,
    ),
    Error,
> {
    let layouts = multilinear_table::global_layouts(shapes, sizes)
        .map_err(|e| Error::Prover(format!("{e:?}")))?;
    let domains = layouts
        .iter()
        .map(|layout| {
            multilinear::whir::Domain::<F>::new(layout.n_stack() + config.log_blowup)
                .map_err(|e| Error::Prover(format!("{e:?}")))
        })
        .collect::<Result<_, _>>()?;
    Ok((layouts, domains))
}

/// A table's preprocessed columns as MLEs, empty for a table that has none.
fn preprocessed_mles(
    air: &dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>,
) -> Result<Vec<Mle<F>>, Error> {
    air.precomputed_columns()
        .into_iter()
        .map(|values| Mle::new(values).map_err(|e| Error::Prover(format!("{}: {e:?}", air.name()))))
        .collect()
}

/// Verifies a proof from [`prove_with_options_and_inputs`].
pub fn verify_with_options(
    proof: &MultilinearVmProof,
    elf_bytes: &[u8],
    proof_options: &ProofOptions,
) -> Result<bool, Error> {
    let program = Elf::load(elf_bytes).map_err(|e| Error::ElfLoad(format!("{e}")))?;

    // A prover choosing the counts chooses the constraint sets, so they are
    // checked before an AIR is built off them.
    proof.table_counts.validate()?;
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

    let expected = proof.table_counts.total() + FIXED_TABLE_COUNT + page_configs.len();
    if expected != proof.proof.tables.len() {
        return Err(Error::InvalidTableCounts(format!(
            "table_counts total ({}) + {FIXED_TABLE_COUNT} fixed + {} pages = {expected}, but the proof carries {} tables",
            proof.table_counts.total(),
            page_configs.len(),
            proof.proof.tables.len(),
        )));
    }
    if proof.table_num_vars.len() != proof.proof.tables.len() {
        return Err(Error::InvalidTableCounts(format!(
            "the proof carries {} tables but {} heights",
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
    let shapes: Vec<Shape> = air_refs
        .iter()
        .zip(&proof.table_num_vars)
        .map(|(air, &num_vars)| (air.trace_layout().0, num_vars as usize))
        .collect();
    let config = chain_config(&shapes);

    let mut transcript = DefaultTranscript::<E>::new(&[]);
    absorb(
        &mut transcript,
        &statement::elf_digest(elf_bytes),
        &proof.public_output,
        &proof.table_counts,
        proof.num_private_input_pages,
        &proof.runtime_page_ranges,
        &proof.table_num_vars,
        &config,
    );

    let layouts: Vec<TableLayout<'_, F, E>> = air_refs
        .iter()
        .zip(&shapes)
        .map(|(air, &(width, num_vars))| {
            layout_of(*air, width, num_vars)
                .map_err(|e| Error::Prover(format!("{}: {e:?}", air.name())))
        })
        .collect::<Result<_, _>>()?;
    // The preprocessed columns, rebuilt from the ELF. Nothing else in the
    // argument says a preprocessed table is the one the program implies: the
    // commitment binds the prover to what it committed, not to the right thing.
    let preprocessed: Vec<Vec<Mle<F>>> = air_refs
        .iter()
        .map(|air| preprocessed_mles(*air))
        .collect::<Result<_, _>>()?;
    let statements: Vec<TableStatement<'_, F, E>> = layouts
        .iter()
        .zip(&preprocessed)
        .map(|(layout, cols)| layout.statement_with_preprocessed(cols))
        .collect();

    // What the tables owe: the COMMIT bus's counterparty is the statement, and
    // its offset depends on the very challenges `multi_verify` is about to draw
    // — so the transcript is replayed to that point on a fork.
    let mut probe = transcript.clone();
    for root in &proof.proof.roots {
        probe.append_bytes(root);
    }
    let z: FieldElement<E> = probe.sample_field_element();
    let alpha: FieldElement<E> = probe.sample_field_element();
    // `start_index` is the carried x254: zero for a monolithic proof.
    let Some(owed) = crate::compute_commit_bus_offset(&proof.public_output, 0, &z, &alpha) else {
        return Ok(false);
    };

    // The stack every table's columns share, rebuilt from the shapes alone. A
    // monolithic proof commits them all together, so there is one group.
    let sizes = [shapes.len()];
    let (layouts, domains) = stacks(&shapes, &sizes, &config)?;

    Ok(multilinear_table::multi_verify(
        &proof.proof,
        &statements,
        &layouts,
        &domains,
        &sizes,
        &owed,
        &config,
        &mut transcript,
    )
    .is_ok())
}
