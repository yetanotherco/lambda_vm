//! Continuations on the multilinear path.
//!
//! The split into epochs, the local-to-global bookend, the cross-epoch register
//! and commit-index carry — all of that is [`crate::continuation`]'s and none of
//! it depends on the commitment scheme. What changes here is only how an
//! epoch's tables are argued: one WHIR commitment over the whole epoch and one
//! opening, the way [`crate::multilinear_prove`] does it for a whole program.
//!
//! # The binding
//!
//! A continuation is two halves: every epoch, and one cross-epoch proof over
//! the bookends and the global-memory tables. What makes them one proof is that
//! an epoch's bookend is the table the cross-epoch proof chained — and a table
//! here has no root of its own, since the stack gives one root per *stacked
//! polynomial*, shared by whatever columns land in it.
//!
//! So each bookend is committed in a group by itself, whose layout follows from
//! the table's shape alone and is therefore the same on both sides, and the
//! binding is comparing those roots. That only holds because a table commits
//! the same columns in the same order wherever it is argued, which is
//! `LeafLayout::build_live_over`'s job, not this module's.

use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use crypto::fiat_shamir::is_transcript::IsTranscript;
use executor::elf::Elf;
use math::field::element::FieldElement;
use multilinear::mle::Mle;
use multilinear::whir_chain::ChainConfig;
use stark::config::Commitment;
use stark::multilinear_table::{
    self, CommittedTable, CommittedTables, MultiProof, TableLayout, TableStatement,
};
use stark::proof::options::ProofOptions;
use stark::traits::AIR;

use crate::multilinear_prove::chain_config;
use crate::statement;
use crate::tables::local_to_global::{self, CellBoundary};
use crate::tables::register;
use crate::tables::trace_builder::Traces;
use crate::test_utils::{E, F};
use crate::{Error, TableCounts};

/// Domain tag for a multilinear continuation epoch.
///
/// Distinct from both the univariate epoch tag and the monolithic multilinear
/// one: no two of the three may ever share a transcript prefix.
const MULTILINEAR_EPOCH_TAG: &[u8] = b"LAMBDAVM_MULTILINEAR_CONTINUATION_EPOCH_V1";

/// One epoch's proof and everything a standalone verifier re-binds.
///
/// Mirrors [`crate::continuation`]'s, minus the fields that only mean something
/// under FRI: there is no per-table root to carry, and the runtime page ranges
/// are always empty because a continuation epoch skips PAGE.
#[derive(Debug, Clone, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct EpochProof {
    /// The epoch's tables and the one opening that settles all of them, with
    /// the local-to-global table last.
    pub proof: MultiProof<F, E>,
    /// Each table's height in variables, same order.
    pub table_num_vars: Vec<u8>,
    pub table_counts: TableCounts,
    pub public_output: Vec<u8>,
    /// The epoch's final register file `R_{i+1}`, which the next epoch takes as
    /// its `INIT` — the cross-epoch register binding. x254 rides along.
    pub reg_fini: Vec<u32>,
}

impl EpochProof {
    /// The roots of the commitment the local-to-global bookend has to itself.
    ///
    /// This is what ties the epoch to the cross-epoch proof: the two commit the
    /// same table, and with the bookend in a commitment group of its own its
    /// roots say so. Every other table shares a group.
    ///
    /// `num_polys` is how many polynomials the stack split that group into —
    /// one for a bookend that fits in a stack, more for an epoch long enough
    /// that it does not. The group is the last, so its roots are the tail.
    pub fn l2g_roots(&self, num_polys: usize) -> Option<&[stark::config::Commitment]> {
        if num_polys == 0 {
            return None;
        }
        let start = self.proof.roots.len().checked_sub(num_polys)?;
        Some(&self.proof.roots[start..])
    }
}

/// The roots the local-to-global table commits to on its own — what an epoch
/// proof carries and the cross-epoch proof has to reproduce.
///
/// A function of the table, the blowup and the fold width, and of nothing else:
/// in particular not of the query count, which is what lets two proofs over
/// different table sets agree on them. One root per polynomial the stack split
/// the table into.
pub fn l2g_commitment(
    boundary: &[CellBoundary],
    config: &ChainConfig,
) -> Result<Vec<stark::config::Commitment>, Error> {
    let trace = local_to_global::generate_local_to_global_trace(boundary);
    let columns: Vec<Mle<F>> = trace
        .columns_main()
        .into_iter()
        .map(|values| Mle::new(values).map_err(|e| Error::Prover(format!("{e:?}"))))
        .collect::<Result<_, _>>()?;
    let shape = [(
        columns.len(),
        trace.main_table.height.trailing_zeros() as usize,
    )];
    let layout =
        multilinear_table::global_layout(&shape).map_err(|e| Error::Prover(format!("{e:?}")))?;
    let stacked = multilinear::stacked_eval::StackedCommitment::<F>::commit(
        layout,
        &multilinear::stacking::borrow(&columns),
        None,
        config,
    )
    .map_err(|e| Error::Prover(format!("{e:?}")))?;
    let roots = stacked.roots().to_vec();
    if roots.is_empty() {
        return Err(Error::Prover("the bookend commits to nothing".to_string()));
    }
    Ok(roots)
}

/// Binds an epoch's statement into the transcript before any challenge.
///
/// The monolithic multilinear statement plus the epoch's position. A
/// continuation epoch never has private-input pages (the bookend replaces
/// PAGE), so that count is not stated — it is zero by construction.
fn absorb_epoch(
    t: &mut DefaultTranscript<E>,
    elf_digest: &[u8; 32],
    public_output: &[u8],
    table_counts: &TableCounts,
    epoch_label: u64,
    table_num_vars: &[u8],
    config: &ChainConfig,
) {
    t.append_bytes(MULTILINEAR_EPOCH_TAG);
    t.append_bytes(elf_digest);
    t.append_bytes(&epoch_label.to_le_bytes());

    t.append_bytes(&(public_output.len() as u64).to_le_bytes());
    t.append_bytes(public_output);

    statement::absorb_table_counts(t, table_counts);

    t.append_bytes(&(table_num_vars.len() as u64).to_le_bytes());
    t.append_bytes(table_num_vars);

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

/// How an epoch's tables are split across commitments: everything together,
/// and the local-to-global bookend on its own.
///
/// The bookend needs a root of its own because the cross-epoch proof commits
/// the same table and the two have to be compared. A table that shares a stack
/// has no root — the stack gives one per stacked *polynomial* — so committing
/// it alone is the only way to say "this is that table".
pub(crate) fn epoch_groups(num_tables: usize) -> Vec<usize> {
    vec![num_tables - 1, 1]
}

/// A table's layout, from the AIR and the shape the verifier states.
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
        stark::multilinear_air::Uniforms::default(),
    )
}

/// What the epoch's tables owe the statement: the COMMIT bus's counterparty,
/// counted from the commit index this epoch carried in.
fn owed(
    public_output: &[u8],
    register_init: &[u32],
    roots: &[Commitment],
    transcript: &DefaultTranscript<E>,
) -> Option<FieldElement<E>> {
    let start_index = *register_init.get(register::X254_INDEX)? as u64;
    let mut probe = transcript.clone();
    for root in roots {
        probe.append_bytes(root);
    }
    let z: FieldElement<E> = probe.sample_field_element();
    let alpha: FieldElement<E> = probe.sample_field_element();
    crate::compute_commit_bus_offset(public_output, start_index, &z, &alpha)
}

/// Domain tag for the multilinear cross-epoch proof.
const MULTILINEAR_GLOBAL_TAG: &[u8] = b"LAMBDAVM_MULTILINEAR_CONTINUATION_GLOBAL_V1";

/// The one cross-epoch proof: every epoch's bookend and the global-memory
/// tables, in one transcript.
///
/// The bookends come first, one commitment group each, so root `k` is the one
/// epoch `k` carries — that comparison is what says the two proofs are about
/// the same table.
#[derive(Debug, Clone, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct GlobalProof {
    pub proof: MultiProof<F, E>,
    pub table_num_vars: Vec<u8>,
}

impl GlobalProof {
    /// The roots each epoch's bookend was committed under here, in epoch order.
    ///
    /// The bookends are the first commitment groups, one each, and the roots
    /// are flat — one per stacked polynomial — so a group's are a window.
    /// `polys` is how many polynomials each one stacked into.
    pub fn l2g_roots(&self, polys: &[usize]) -> Option<Vec<&[stark::config::Commitment]>> {
        let mut start = 0usize;
        let mut groups = Vec::with_capacity(polys.len());
        for &num_polys in polys {
            if num_polys == 0 {
                return None;
            }
            let end = start.checked_add(num_polys)?;
            groups.push(self.proof.roots.get(start..end)?);
            start = end;
        }
        Some(groups)
    }
}

/// Binds the cross-epoch statement: what the run was, not what any epoch was.
fn absorb_global(
    t: &mut DefaultTranscript<E>,
    elf_digest: &[u8; 32],
    num_epochs: usize,
    num_private_input_pages: usize,
    page_bases: &[u64],
    table_num_vars: &[u8],
    config: &ChainConfig,
) {
    t.append_bytes(MULTILINEAR_GLOBAL_TAG);
    t.append_bytes(elf_digest);
    t.append_bytes(&(num_epochs as u64).to_le_bytes());
    t.append_bytes(&(num_private_input_pages as u64).to_le_bytes());
    t.append_bytes(&(page_bases.len() as u64).to_le_bytes());
    for base in page_bases {
        t.append_bytes(&base.to_le_bytes());
    }
    t.append_bytes(&(table_num_vars.len() as u64).to_le_bytes());
    t.append_bytes(table_num_vars);
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

/// How the cross-epoch proof's tables are split: every bookend alone — so its
/// root can be compared against the epoch that committed it — and the
/// global-memory tables together.
pub(crate) fn global_groups(num_epochs: usize, num_pages: usize) -> Vec<usize> {
    let mut sizes = vec![1usize; num_epochs];
    // A run that touched no memory has no global-memory tables, and a group of
    // none is a commitment to nothing.
    if num_pages > 0 {
        sizes.push(num_pages);
    }
    sizes
}

/// Proves the cross-epoch memory chain: each epoch's bookend, and one
/// global-memory table per page the run touched.
pub fn prove_global(
    boundaries: &[std::sync::Arc<Vec<CellBoundary>>],
    elf_bytes: &[u8],
    init_page_data: &std::collections::HashMap<u64, Vec<u8>>,
    page_bases: &[u64],
    num_private_input_pages: usize,
    opts: &ProofOptions,
) -> Result<GlobalProof, Error> {
    // Each cell's final state; the boundaries are in epoch order, so the last
    // fini wins.
    let mut final_state: crate::tables::global_memory::FiniStateMap =
        std::collections::HashMap::new();
    for epoch in boundaries {
        for b in epoch.iter() {
            final_state.insert(
                b.address,
                crate::tables::global_memory::FiniState {
                    value: (b.fini.value & 0xFF) as u8,
                    epoch: b.fini.epoch,
                },
            );
        }
    }

    let gm_configs = crate::continuation::global_memory_configs_from_init_page_data(
        page_bases,
        init_page_data,
        num_private_input_pages,
        true,
    );

    let l2g_airs: Vec<_> = (0..boundaries.len())
        .map(|i| crate::continuation::l2g_global_air(opts, local_to_global::epoch_label(i as u64)))
        .collect();
    let gm_airs: Vec<_> = gm_configs
        .iter()
        .map(|config| crate::continuation::global_memory_air(opts, config, None))
        .collect();
    let mut l2g_traces: Vec<_> = boundaries
        .iter()
        .map(|epoch| local_to_global::generate_local_to_global_trace(epoch.as_slice()))
        .collect();
    let mut gm_traces: Vec<_> = gm_configs
        .iter()
        .map(|config| crate::tables::global_memory::generate_global_trace(config, &final_state))
        .collect();

    let mut pairs: Vec<crate::AirTracePair<'_>> = Vec::new();
    for (air, trace) in l2g_airs.iter().zip(l2g_traces.iter_mut()) {
        pairs.push((air, trace, &()));
    }
    for (air, trace) in gm_airs.iter().zip(gm_traces.iter_mut()) {
        pairs.push((air, trace, &()));
    }

    let shapes: Vec<(usize, usize)> = pairs
        .iter()
        .map(|(_, trace, _)| {
            (
                trace.main_table.width,
                trace.main_table.height.trailing_zeros() as usize,
            )
        })
        .collect();
    let table_num_vars: Vec<u8> = shapes.iter().map(|&(_, n)| n as u8).collect();
    let config = chain_config(&shapes);

    let mut transcript = DefaultTranscript::<E>::new(&[]);
    absorb_global(
        &mut transcript,
        &statement::elf_digest(elf_bytes),
        boundaries.len(),
        num_private_input_pages,
        page_bases,
        &table_num_vars,
        &config,
    );

    let mut committed = Vec::with_capacity(pairs.len());
    for ((air, trace, _), &(width, num_vars)) in pairs.iter_mut().zip(&shapes) {
        let layout = layout_of(*air, width, num_vars)
            .map_err(|e| Error::Prover(format!("{}: {e:?}", air.name())))?;
        let mut columns = trace.columns_main();
        for (col, expected) in air.precomputed_columns().iter().enumerate() {
            if columns.get(col) != Some(expected) {
                return Err(Error::Prover(format!(
                    "{}: preprocessed column {col} is not what the run implies",
                    air.name(),
                )));
            }
        }
        committed.push(
            CommittedTable::from_layout(layout, |col| core::mem::take(&mut columns[col as usize]))
                .map_err(|e| Error::Prover(format!("{}: {e:?}", air.name())))?,
        );
    }
    let sizes = global_groups(boundaries.len(), gm_configs.len());
    let committed = CommittedTables::commit_grouped(committed, &sizes, &config)
        .map_err(|e| Error::Prover(format!("{e:?}")))?;
    let proof = multilinear_table::multi_prove(&committed, &config, &mut transcript)
        .map_err(|e| Error::Prover(format!("{e:?}")))?;

    Ok(GlobalProof {
        proof,
        table_num_vars,
    })
}

/// Verifies the cross-epoch proof from the ELF and the run's public shape.
///
/// `page_bases` and `num_epochs` are the bundle's, and both are bound into the
/// transcript and pinned by the bus: a wrong set leaves the GlobalMemory bus
/// unbalanced or the AIR count mismatched.
pub fn verify_global(
    elf: &Elf,
    elf_bytes: &[u8],
    global: &GlobalProof,
    num_epochs: usize,
    page_bases: &[u64],
    num_private_input_pages: usize,
    opts: &ProofOptions,
) -> Result<bool, Error> {
    Ok(verify_global_bookends(
        elf,
        elf_bytes,
        global,
        num_epochs,
        page_bases,
        num_private_input_pages,
        opts,
    )?
    .is_some())
}

/// [`verify_global`], handing back the roots each epoch's bookend was
/// committed under — which is what the binding compares. `None` is a proof
/// that does not verify.
#[allow(clippy::too_many_arguments)]
fn verify_global_bookends(
    elf: &Elf,
    elf_bytes: &[u8],
    global: &GlobalProof,
    num_epochs: usize,
    page_bases: &[u64],
    num_private_input_pages: usize,
    opts: &ProofOptions,
) -> Result<Option<Vec<Vec<Commitment>>>, Error> {
    let l2g_airs: Vec<_> = (0..num_epochs)
        .map(|i| crate::continuation::l2g_global_air(opts, local_to_global::epoch_label(i as u64)))
        .collect();
    // Rebuilt from the ELF, never from the bundle: this is the genesis binding.
    let gm_configs =
        crate::continuation::global_memory_configs(page_bases, elf, num_private_input_pages);
    let gm_airs: Vec<_> = gm_configs
        .iter()
        .map(|config| crate::continuation::global_memory_air(opts, config, None))
        .collect();

    let mut air_refs: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>> =
        l2g_airs.iter().map(|a| a as _).collect();
    for air in &gm_airs {
        air_refs.push(air);
    }
    if air_refs.len() != global.proof.tables.len() || global.table_num_vars.len() != air_refs.len()
    {
        return Err(Error::InvalidTableCounts(format!(
            "the cross-epoch layout has {} tables, the proof carries {} and {} heights",
            air_refs.len(),
            global.proof.tables.len(),
            global.table_num_vars.len(),
        )));
    }

    let shapes: Vec<(usize, usize)> = air_refs
        .iter()
        .zip(&global.table_num_vars)
        .map(|(air, &num_vars)| (air.trace_layout().0, num_vars as usize))
        .collect();
    let config = chain_config(&shapes);

    let mut transcript = DefaultTranscript::<E>::new(&[]);
    absorb_global(
        &mut transcript,
        &statement::elf_digest(elf_bytes),
        num_epochs,
        num_private_input_pages,
        page_bases,
        &global.table_num_vars,
        &config,
    );

    let layouts: Vec<TableLayout<'_, F, E>> = air_refs
        .iter()
        .zip(&shapes)
        .map(|(air, &(width, num_vars))| {
            layout_of(*air, width, num_vars).map_err(|e| Error::Prover(format!("{e:?}")))
        })
        .collect::<Result<_, _>>()?;
    let preprocessed: Vec<Vec<Mle<F>>> = air_refs
        .iter()
        .map(|air| {
            air.precomputed_columns()
                .into_iter()
                .map(|values| Mle::new(values).map_err(|e| Error::Prover(format!("{e:?}"))))
                .collect::<Result<_, _>>()
        })
        .collect::<Result<_, _>>()?;
    let statements: Vec<TableStatement<'_, F, E>> = layouts
        .iter()
        .zip(&preprocessed)
        .map(|(layout, cols)| layout.statement_with_preprocessed(cols))
        .collect();

    let sizes = global_groups(num_epochs, gm_configs.len());
    let (stacks, domains) = crate::multilinear_prove::stacks(&shapes, &sizes, &config)?;
    // Each bookend is a group of its own, so its roots are the group's — as
    // many as the stack split it into.
    let polys: Vec<usize> = stacks[..num_epochs].iter().map(|l| l.num_polys()).collect();

    // The cross-epoch bus has no counterparty in the statement: it must vanish.
    if multilinear_table::multi_verify(
        &global.proof,
        &statements,
        &stacks,
        &domains,
        &sizes,
        &FieldElement::<E>::zero(),
        &config,
        &mut transcript,
    )
    .is_err()
    {
        return Ok(None);
    }
    Ok(global
        .l2g_roots(&polys)
        .map(|groups| groups.into_iter().map(<[_]>::to_vec).collect()))
}

/// Proves one epoch: its tables plus the local-to-global bookend, against one
/// commitment.
#[allow(clippy::too_many_arguments)]
pub fn prove_epoch(
    elf: &Elf,
    elf_bytes: &[u8],
    register_init: &[u32],
    label: u64,
    mut traces: Traces,
    is_final: bool,
    boundary: &[CellBoundary],
    opts: &ProofOptions,
    decode_commitment: Option<Commitment>,
) -> Result<EpochProof, Error> {
    // The bookend's range checks are lookups into BITWISE, so its
    // multiplicities have to carry them.
    crate::tables::bitwise::update_multiplicities(
        &mut traces.bitwise,
        &local_to_global::collect_bitwise_from_l2g(boundary),
    );
    if !traces.page_configs.is_empty() {
        return Err(Error::ContinuationInvariant(
            "continuation epoch must have no PAGE configs (L2G bookend replaces PAGE)".to_string(),
        ));
    }

    let reg_fini = register::fini_from_trace(&traces.register);
    let table_counts = traces.table_counts();
    let public_output = traces.public_output_bytes.clone();

    let airs = crate::continuation::build_epoch_airs(
        elf,
        opts,
        &[],
        &table_counts,
        register_init,
        &reg_fini,
        is_final,
        decode_commitment,
    );
    let l2g_air = crate::continuation::l2g_memory_air(opts, label);
    let mut l2g_trace = local_to_global::generate_local_to_global_trace(boundary);

    let mut pairs = airs.air_trace_pairs(&mut traces);
    pairs.push((&l2g_air, &mut l2g_trace, &()));

    let shapes: Vec<(usize, usize)> = pairs
        .iter()
        .map(|(_, trace, _)| {
            (
                trace.main_table.width,
                trace.main_table.height.trailing_zeros() as usize,
            )
        })
        .collect();
    let table_num_vars: Vec<u8> = shapes.iter().map(|&(_, n)| n as u8).collect();
    let config = chain_config(&shapes);

    let mut transcript = DefaultTranscript::<E>::new(&[]);
    absorb_epoch(
        &mut transcript,
        &statement::elf_digest(elf_bytes),
        &public_output,
        &table_counts,
        label,
        &table_num_vars,
        &config,
    );

    let mut committed = Vec::with_capacity(pairs.len());
    for ((air, trace, _), &(width, num_vars)) in pairs.iter_mut().zip(&shapes) {
        let layout = layout_of(*air, width, num_vars)
            .map_err(|e| Error::Prover(format!("{}: {e:?}", air.name())))?;
        let mut columns = trace.columns_main();
        // The verifier rebuilds these and demands the proof open to them, so a
        // trace that disagrees produces a proof nobody can verify.
        for (col, expected) in air.precomputed_columns().iter().enumerate() {
            if columns.get(col) != Some(expected) {
                return Err(Error::Prover(format!(
                    "{}: preprocessed column {col} is not what the program implies",
                    air.name(),
                )));
            }
        }
        committed.push(
            CommittedTable::from_layout(layout, |col| core::mem::take(&mut columns[col as usize]))
                .map_err(|e| Error::Prover(format!("{}: {e:?}", air.name())))?,
        );
    }
    let sizes = epoch_groups(committed.len());
    let committed = CommittedTables::commit_grouped(committed, &sizes, &config)
        .map_err(|e| Error::Prover(format!("{e:?}")))?;
    let proof = multilinear_table::multi_prove(&committed, &config, &mut transcript)
        .map_err(|e| Error::Prover(format!("{e:?}")))?;

    Ok(EpochProof {
        proof,
        table_num_vars,
        table_counts,
        public_output,
        reg_fini,
    })
}

/// A self-contained multilinear continuation proof.
///
/// Mirrors [`crate::continuation::ContinuationProof`]: the per-epoch proofs in
/// execution order, the one cross-epoch proof, and the two public values the
/// verifier rebuilds the cross-epoch tables from. **No cell values travel** —
/// the boundaries stay with the prover, because a boundary's init value is a
/// byte of the private input for a private read.
#[derive(Debug, Clone, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct ContinuationProof {
    pub epochs: Vec<EpochProof>,
    pub global: GlobalProof,
    pub num_private_input_pages: usize,
    /// Sorted, deduped page bases the run touched: page bases ONLY, so no
    /// private byte is in here. Prover-supplied but bus-enforced — a wrong set
    /// leaves the cross-epoch bus unbalanced or the table count mismatched, and
    /// it is bound into the cross-epoch statement.
    pub touched_page_bases: Vec<u64>,
}

impl ContinuationProof {
    pub fn num_epochs(&self) -> usize {
        self.epochs.len()
    }

    /// The run's committed output: every epoch's slice, in order.
    pub fn public_output(&self) -> Vec<u8> {
        self.epochs
            .iter()
            .flat_map(|e| e.public_output.iter().copied())
            .collect()
    }
}

/// Proves a whole run: every epoch, then the one cross-epoch proof that chains
/// their memory.
pub fn prove_continuation(
    elf_bytes: &[u8],
    private_inputs: &[u8],
    epoch_size_log2: u32,
    opts: &ProofOptions,
) -> Result<ContinuationProof, Error> {
    let elf = Elf::load(elf_bytes).map_err(|e| Error::ElfLoad(format!("{e}")))?;
    let decode_commitment = crate::tables::decode::commitment_from_elf(&elf, opts)
        .map_err(|e| Error::Recursion(format!("DECODE commitment from ELF: {e}")))?;
    let artifacts = crate::tables::trace_builder::DecodeArtifacts::from_elf(&elf)?;

    let mut epochs = Vec::new();
    let boundaries = crate::continuation::for_each_epoch_overlapped(
        &elf,
        private_inputs,
        epoch_size_log2,
        &artifacts,
        |prepared| {
            epochs.push(prove_epoch(
                &elf,
                elf_bytes,
                &prepared.register_init,
                prepared.label,
                prepared.traces,
                prepared.is_final,
                &prepared.boundary,
                opts,
                Some(decode_commitment),
            )?);
            Ok(())
        },
    )?;

    // The genesis image, which is the one the run started from — rebuilt here
    // rather than carried, because `for_each_epoch` advances its copy.
    let init_page_data = crate::tables::trace_builder::build_init_page_data(
        &crate::tables::trace_builder::build_initial_image_paged(&elf, private_inputs),
    );
    let num_private_input_pages = crate::tables::page::private_input_page_count(private_inputs);
    // One source of truth: the same list drives the committed tables and
    // travels in the bundle, so the two cannot diverge.
    let touched_page_bases = crate::continuation::touched_page_bases(&boundaries);
    let global = prove_global(
        &boundaries,
        elf_bytes,
        &init_page_data,
        &touched_page_bases,
        num_private_input_pages,
        opts,
    )?;

    Ok(ContinuationProof {
        epochs,
        global,
        num_private_input_pages,
        touched_page_bases,
    })
}

/// Verifies a whole run from the bundle and the ELF alone.
///
/// The verifier enumerates the epochs itself — `epoch_label` and `is_final` are
/// positions, not claims — derives each one's starting registers from the ELF
/// or the previous proof, closes the cross-epoch bus with genesis rebuilt from
/// the ELF, and **ties each epoch's bookend to the cross-epoch proof by its
/// root**. Without that last step the two halves are about unrelated tables.
pub fn verify_continuation(
    elf_bytes: &[u8],
    bundle: &ContinuationProof,
    opts: &ProofOptions,
) -> Result<bool, Error> {
    let Some(proved) = verify_epochs_bookends(elf_bytes, &bundle.epochs, opts)? else {
        return Ok(false);
    };
    let elf = Elf::load(elf_bytes).map_err(|e| Error::ElfLoad(format!("{e}")))?;
    let Some(chained) = verify_global_bookends(
        &elf,
        elf_bytes,
        &bundle.global,
        bundle.epochs.len(),
        &bundle.touched_page_bases,
        bundle.num_private_input_pages,
        opts,
    )?
    else {
        return Ok(false);
    };

    // The binding: epoch `k`'s bookend and the one the cross-epoch proof
    // chained are the same table, or neither half says anything about the
    // other. Comparing the groups whole is also what catches a bookend the two
    // sides stacked differently.
    Ok(proved == chained)
}

/// Proves every epoch of a run, in order, chaining the register file.
///
/// **Not a continuation proof yet.** Without the cross-epoch global-memory
/// proof nothing ties one epoch's *memory* to the next: what chains here is the
/// register file, which each epoch's REGISTER preprocessing binds at both ends.
/// The missing half, and why its binding needs a mechanism the univariate path
/// does not, is in this module's header.
pub fn prove_epochs(
    elf_bytes: &[u8],
    private_inputs: &[u8],
    epoch_size_log2: u32,
    opts: &ProofOptions,
) -> Result<Vec<EpochProof>, Error> {
    let elf = Elf::load(elf_bytes).map_err(|e| Error::ElfLoad(format!("{e}")))?;
    // A pure function of (ELF, opts), so it is computed once rather than once
    // per epoch inside the AIR build.
    let decode_commitment = crate::tables::decode::commitment_from_elf(&elf, opts)
        .map_err(|e| Error::Recursion(format!("DECODE commitment from ELF: {e}")))?;
    let artifacts = crate::tables::trace_builder::DecodeArtifacts::from_elf(&elf)?;

    let mut proofs = Vec::new();
    crate::continuation::for_each_epoch_overlapped(
        &elf,
        private_inputs,
        epoch_size_log2,
        &artifacts,
        |prepared| {
            proofs.push(prove_epoch(
                &elf,
                elf_bytes,
                &prepared.register_init,
                prepared.label,
                prepared.traces,
                prepared.is_final,
                &prepared.boundary,
                opts,
                Some(decode_commitment),
            )?);
            Ok(())
        },
    )?;
    Ok(proofs)
}

/// Verifies a run's epochs from the ELF alone, deriving each one's starting
/// registers from the last one's proof.
///
/// The verifier owns every value an epoch is checked against: the first
/// epoch's registers come from the ELF's entry point, the rest from the
/// previous epoch's proved `reg_fini`, and `is_final` is the position, not a
/// claim. Same caveat as [`prove_epochs`]: this checks the epochs, not that
/// their memory chains.
pub fn verify_epochs(
    elf_bytes: &[u8],
    epochs: &[EpochProof],
    opts: &ProofOptions,
) -> Result<bool, Error> {
    Ok(verify_epochs_bookends(elf_bytes, epochs, opts)?.is_some())
}

/// [`verify_epochs`], handing back each epoch's bookend roots in order — which
/// is what the binding compares. `None` is a run that does not verify.
fn verify_epochs_bookends(
    elf_bytes: &[u8],
    epochs: &[EpochProof],
    opts: &ProofOptions,
) -> Result<Option<Vec<Vec<Commitment>>>, Error> {
    if epochs.is_empty() {
        return Ok(None);
    }
    let elf = Elf::load(elf_bytes).map_err(|e| Error::ElfLoad(format!("{e}")))?;
    let mut carried = register::register_init_from_entry_point(elf.entry_point);
    let mut bookends = Vec::with_capacity(epochs.len());
    for (index, epoch) in epochs.iter().enumerate() {
        let label = local_to_global::epoch_label(index as u64);
        let is_final = index + 1 == epochs.len();
        let Some(roots) =
            verify_epoch_bookend(&elf, elf_bytes, epoch, &carried, is_final, label, opts)?
        else {
            return Ok(None);
        };
        bookends.push(roots);
        carried.clone_from(&epoch.reg_fini);
    }
    Ok(Some(bookends))
}

/// Verifies one epoch from the bundle and the ELF alone.
///
/// `register_init` is the verifier's, not the bundle's: the ELF's for epoch 0,
/// the previous epoch's `reg_fini` after that. That is the cross-epoch register
/// binding, and the commit index rides in it.
#[allow(clippy::too_many_arguments)]
pub fn verify_epoch(
    elf: &Elf,
    elf_bytes: &[u8],
    epoch: &EpochProof,
    register_init: &[u32],
    is_final: bool,
    label: u64,
    opts: &ProofOptions,
) -> Result<bool, Error> {
    Ok(
        verify_epoch_bookend(elf, elf_bytes, epoch, register_init, is_final, label, opts)?
            .is_some(),
    )
}

/// [`verify_epoch`], handing back the roots the epoch's bookend was committed
/// under — which is what the binding compares. `None` is a proof that does not
/// verify.
#[allow(clippy::too_many_arguments)]
fn verify_epoch_bookend(
    elf: &Elf,
    elf_bytes: &[u8],
    epoch: &EpochProof,
    register_init: &[u32],
    is_final: bool,
    label: u64,
    opts: &ProofOptions,
) -> Result<Option<Vec<Commitment>>, Error> {
    let airs = crate::continuation::build_epoch_airs(
        elf,
        opts,
        &[],
        &epoch.table_counts,
        register_init,
        &epoch.reg_fini,
        is_final,
        None,
    );
    let l2g_air = crate::continuation::l2g_memory_air(opts, label);
    let mut air_refs = airs.air_refs();
    air_refs.push(&l2g_air);

    if air_refs.len() != epoch.proof.tables.len() || epoch.table_num_vars.len() != air_refs.len() {
        return Err(Error::InvalidTableCounts(format!(
            "the epoch layout has {} tables, the proof carries {} and {} heights",
            air_refs.len(),
            epoch.proof.tables.len(),
            epoch.table_num_vars.len(),
        )));
    }

    // The width is the AIR's, never the proof's; only the height is stated.
    let shapes: Vec<(usize, usize)> = air_refs
        .iter()
        .zip(&epoch.table_num_vars)
        .map(|(air, &num_vars)| (air.trace_layout().0, num_vars as usize))
        .collect();
    let config = chain_config(&shapes);

    let mut transcript = DefaultTranscript::<E>::new(&[]);
    absorb_epoch(
        &mut transcript,
        &statement::elf_digest(elf_bytes),
        &epoch.public_output,
        &epoch.table_counts,
        label,
        &epoch.table_num_vars,
        &config,
    );

    let layouts: Vec<TableLayout<'_, F, E>> = air_refs
        .iter()
        .zip(&shapes)
        .map(|(air, &(width, num_vars))| {
            layout_of(*air, width, num_vars).map_err(|e| Error::Prover(format!("{e:?}")))
        })
        .collect::<Result<_, _>>()?;
    let preprocessed: Vec<Vec<Mle<F>>> = air_refs
        .iter()
        .map(|air| {
            air.precomputed_columns()
                .into_iter()
                .map(|values| Mle::new(values).map_err(|e| Error::Prover(format!("{e:?}"))))
                .collect::<Result<_, _>>()
        })
        .collect::<Result<_, _>>()?;
    let statements: Vec<TableStatement<'_, F, E>> = layouts
        .iter()
        .zip(&preprocessed)
        .map(|(layout, cols)| layout.statement_with_preprocessed(cols))
        .collect();

    let Some(owed) = owed(
        &epoch.public_output,
        register_init,
        &epoch.proof.roots,
        &transcript,
    ) else {
        return Ok(None);
    };

    let sizes = epoch_groups(shapes.len());
    let (layouts, domains) = crate::multilinear_prove::stacks(&shapes, &sizes, &config)?;
    // The bookend is committed in the last group, alone, so its roots are that
    // group's — as many as the stack split it into.
    let num_polys = layouts.last().map(|l| l.num_polys()).unwrap_or(0);

    if multilinear_table::multi_verify(
        &epoch.proof,
        &statements,
        &layouts,
        &domains,
        &sizes,
        &owed,
        &config,
        &mut transcript,
    )
    .is_err()
    {
        return Ok(None);
    }
    Ok(epoch.l2g_roots(num_polys).map(<[_]>::to_vec))
}
