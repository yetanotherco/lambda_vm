//! Continuations on the multilinear path.
//!
//! The split into epochs, the local-to-global bookend, the cross-epoch register
//! and commit-index carry — all of that is [`crate::continuation`]'s and none of
//! it depends on the commitment scheme. What changes here is only how an
//! epoch's tables are argued: one WHIR commitment over the whole epoch and one
//! opening, the way [`crate::multilinear_prove`] does it for a whole program.
//!
//! # What is not here yet
//!
//! The cross-epoch global-memory proof, and with it the binding that says an
//! epoch's local-to-global table is the one the global proof chains. The
//! univariate path compares that table's own Merkle root across the two proofs;
//! here a table has no root of its own — the stack gives one root per *stacked
//! polynomial*, shared by whatever columns land in it — so the binding needs
//! its own mechanism. See the module's notes in `thoughts/`.

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
    let committed =
        CommittedTables::commit(committed, &config).map_err(|e| Error::Prover(format!("{e:?}")))?;
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
    crate::continuation::for_each_epoch(
        &elf,
        private_inputs,
        epoch_size_log2,
        &artifacts,
        |prepared, _| {
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
    if epochs.is_empty() {
        return Ok(false);
    }
    let elf = Elf::load(elf_bytes).map_err(|e| Error::ElfLoad(format!("{e}")))?;
    let mut carried = register::register_init_from_entry_point(elf.entry_point);
    for (index, epoch) in epochs.iter().enumerate() {
        let label = local_to_global::epoch_label(index as u64);
        let is_final = index + 1 == epochs.len();
        if !verify_epoch(&elf, elf_bytes, epoch, &carried, is_final, label, opts)? {
            return Ok(false);
        }
        carried.clone_from(&epoch.reg_fini);
    }
    Ok(true)
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
        return Ok(false);
    };

    let stacked =
        multilinear_table::global_layout(&shapes).map_err(|e| Error::Prover(format!("{e:?}")))?;
    let domain = multilinear::whir::Domain::<F>::new(stacked.n_stack() + config.log_blowup)
        .map_err(|e| Error::Prover(format!("{e:?}")))?;

    Ok(multilinear_table::multi_verify(
        &epoch.proof,
        &statements,
        &stacked,
        &domain,
        &owed,
        &config,
        &mut transcript,
    )
    .is_ok())
}
