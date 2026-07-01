//! First production implementation of continuations (Approach 2).
//!
//! Splits an execution into fixed-size epochs, proves each epoch independently
//! (its memory is initialized/finalized by the per-epoch local-to-global table),
//! and proves one cross-epoch "global memory" LogUp that links every epoch's
//! `fini` to the next epoch's `init` (so `fini(epoch i) == init(epoch i+1)`).
//!
//! The global proof's genesis anchor is bound to the ELF: the verifier
//! recomputes the per-page preprocessed init commitment from the ELF in
//! `verify_global`, so the starting memory cannot be prover-supplied.
//!
//! The local-to-global columns are range-checked in the epoch proof (which
//! carries the BITWISE provider): values are bytes, and the cross-epoch-only
//! `init_epoch` is built from `IsHalfword`-checked halfwords. Address and
//! fini-timestamp need no extra check — they are matched against MEMW on the
//! epoch-local Memory bus, exactly as PAGE relies on MEMW. The global proof
//! commits the identical trace, so it inherits the guarantee via the commitment
//! binding. There is no cross-epoch timestamp; the chain is ordered by epoch.
//!
//! Cross-epoch registers are bound the same way: each continuation epoch
//! preprocesses its REGISTER `FINI` column to the epoch's final register file
//! `R_{i+1}` (alongside `INIT = R_i`), and the driver reuses the same `R_{i+1}`
//! as the next epoch's preprocessed `INIT` — so `init(epoch i+1) == fini(epoch i)`
//! by construction, with the REG-C2 Memory bus binding `FINI` to the true final
//! registers. No extra bus.
//!
//! The x254 commit index is carried across epochs by that same register binding, so the
//! COMMIT trace indexes its committed bytes from the carried global value (index 0 for the
//! first byte of the run). COMMIT correctness itself is a GLOBAL property: instead of each
//! epoch closing its own output slice, the COMMIT chip emits each byte as a Memory-bus token
//! in the `commit` domain (`domain = 2`), and — like the L2G table — each epoch's COMMIT
//! trace is re-committed in the global proof (root-bound by `verify_commit_commitment_binding`)
//! under a reduced air that carries only that emit. The verifier closes the output bus once,
//! over the whole run's output, via `compute_commit_bus_offset` (indices 0..N). The run-wide
//! output is absorbed into the global statement, so it is verifier-checked (length, order,
//! completeness, no-splice) rather than driver-trusted.
//!
//! The prover and verifier are split: `prove_continuation` emits a self-contained
//! `ContinuationProof` bundle and `verify_continuation` checks it from the bundle
//! and ELF alone (`prove_and_verify_continuation` is a thin wrapper over both).

use std::collections::HashMap;

use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use executor::elf::Elf;
use executor::vm::execution::Executor;
use executor::vm::memory::MAX_PRIVATE_INPUT_SIZE;
use math::field::element::FieldElement;
use stark::config::Commitment;
use stark::lookup::{AirWithBuses, AuxiliaryTraceBuildData, NullBoundaryConstraintBuilder};
use stark::proof::options::ProofOptions;
use stark::proof::stark::MultiProof;
use stark::prover::{IsStarkProver, Prover};
use stark::trace::TraceTable;
use stark::traits::AIR;
use stark::verifier::{IsStarkVerifier, Verifier};

use crate::statement::{StatementKind, absorb_continuation_global_statement, absorb_statement};
use crate::tables::local_to_global::{self, CellBoundary};
use crate::tables::page::{self, PageConfig};
use crate::tables::register;
use crate::tables::trace_builder::{Traces, build_init_page_data, build_initial_image_paged};
use crate::tables::types::{GoldilocksExtension, GoldilocksField};
use crate::tables::{MaxRowsConfig, global_memory};
use crate::{
    COMMIT_TABLE_INDEX, Error, FIXED_TABLE_COUNT, RuntimePageRange, TableCounts, VmAirs,
    compute_commit_bus_offset, replay_transcript_phase_a, verify_commit_commitment_binding,
    verify_l2g_commitment_binding,
};

type F = GoldilocksField;
type E = GoldilocksExtension;
type AirRef<'a> = &'a dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>;

fn empty_constraints()
-> Vec<Box<dyn stark::constraints::transition::TransitionConstraintEvaluator<F, E>>> {
    vec![]
}

/// Fresh transcript seeded with the epoch's statement (ELF, public output, table
/// layout) and `epoch_label` (its position). The epoch's prove, verify, and
/// bus-balance replay all seed via this so their challenges match; the seeding
/// pins each epoch proof to its program and position (replay protection).
fn epoch_transcript(
    elf_bytes: &[u8],
    public_output: &[u8],
    table_counts: &TableCounts,
    num_private_input_pages: usize,
    runtime_page_ranges: &[RuntimePageRange],
    epoch_label: u64,
) -> DefaultTranscript<E> {
    let mut transcript = DefaultTranscript::<E>::new(&[]);
    absorb_statement(
        &mut transcript,
        StatementKind::ContinuationEpoch { epoch_label },
        elf_bytes,
        public_output,
        table_counts,
        num_private_input_pages,
        runtime_page_ranges,
    );
    transcript
}

/// Fresh transcript seeded with the global proof's statement (ELF + epoch count + the
/// run-wide committed output). `prove_global` and `verify_global` both seed via this so their
/// challenges match; absorbing `full_output` binds it to the proof (tampering it diverges the
/// challenges and breaks the output-bus close).
fn global_transcript(
    elf_bytes: &[u8],
    num_epochs: usize,
    full_output: &[u8],
) -> DefaultTranscript<E> {
    let mut transcript = DefaultTranscript::<E>::new(&[]);
    absorb_continuation_global_statement(&mut transcript, elf_bytes, num_epochs, full_output);
    transcript
}

/// The L2G table's AIR constraint: the `MU` selector column is boolean.
///
/// The Memory bus already pins `MU = 1` on real rows and `MU = 0` on padding —
/// it's anchored to MEMW's own bit-constrained multiplicity, since a non-1 `MU`
/// leaves the cell's seed/fini tokens unmatched. This constraint makes
/// "`MU ∈ {0,1}`" explicit on the table itself rather than relying on that
/// cross-bus argument. Lives on the epoch-local air; the global proof commits the
/// identical trace (root-bound), so it inherits it.
fn l2g_constraints()
-> Vec<Box<dyn stark::constraints::transition::TransitionConstraintEvaluator<F, E>>> {
    use crate::constraints::templates::IsBitConstraint;
    use stark::constraints::transition::TransitionConstraint;
    vec![IsBitConstraint::unconditional(local_to_global::cols::MU, 0).boxed()]
}

/// Local-to-global AIR on the cross-epoch GlobalMemory bus (used in the global proof).
///
/// `epoch_label` is this epoch's 1-based label; it is the `fini_epoch` constant
/// the fini token carries (not a trace column, since it's the same for every row).
///
/// Uses `empty_constraints()` deliberately: the MU boolean (`MU·(1-MU)=0`), the
/// column range checks, and the `init_epoch < fini_epoch` ordering are NOT
/// re-asserted here. They are enforced once in the epoch proof's `l2g_memory_air`,
/// and `verify_l2g_commitment_binding` ties this global L2G sub-table to the *same*
/// committed trace (equal Merkle roots). So under collision resistance the trace the
/// global bus runs over already satisfies all those constraints — do not add them
/// here (it would be redundant, not a missing check).
fn l2g_global_air(
    opts: &ProofOptions,
    epoch_label: u64,
) -> AirWithBuses<F, E, NullBoundaryConstraintBuilder, ()> {
    AirWithBuses::new(
        local_to_global::cols::NUM_COLUMNS,
        AuxiliaryTraceBuildData {
            interactions: local_to_global::bus_interactions(epoch_label),
        },
        opts,
        1,
        empty_constraints(),
    )
}

/// Reduced COMMIT AIR for the global proof: carries ONLY the committed-output emit
/// ([`commit::output_bus_interaction`]) on the Memory bus in the `commit` domain.
///
/// The global proof re-commits each epoch's COMMIT trace under this air; the verifier
/// closes the emitted tokens once, over the whole run's output. Uses `empty_constraints()`
/// for the same reason as [`l2g_global_air`]: the COMMIT trace's `MU`/`END`/`FIRST` bits and
/// the recursion structure that pins the emitted `(index, value)` pairs are enforced in the
/// epoch proof's COMMIT air, and `verify_commit_commitment_binding` ties this sub-table to
/// that *same* committed trace (equal main-trace roots) — so re-asserting them here would be
/// redundant, not a missing check.
fn commit_global_air(opts: &ProofOptions) -> AirWithBuses<F, E, NullBoundaryConstraintBuilder, ()> {
    AirWithBuses::new(
        crate::tables::commit::cols::NUM_COLUMNS,
        AuxiliaryTraceBuildData {
            interactions: vec![crate::tables::commit::output_bus_interaction()],
        },
        opts,
        1,
        empty_constraints(),
    )
}

/// Local-to-global AIR on the epoch-local Memory bus (used inside an epoch proof).
///
/// Carries the column range checks and the `init_epoch < fini_epoch` ordering
/// check too: this proof has the BITWISE provider, and the global proof commits
/// the identical trace (the commitment binding compares roots), so checking here
/// covers both. `epoch_label` is the `fini_epoch` constant used by both.
fn l2g_memory_air(
    opts: &ProofOptions,
    epoch_label: u64,
) -> AirWithBuses<F, E, NullBoundaryConstraintBuilder, ()> {
    let interactions = [
        local_to_global::memory_bus_interactions(),
        local_to_global::range_check_interactions(epoch_label),
    ]
    .concat();
    AirWithBuses::new(
        local_to_global::cols::NUM_COLUMNS,
        AuxiliaryTraceBuildData { interactions },
        opts,
        1,
        l2g_constraints(),
    )
}

/// GLOBAL_MEMORY AIR for one touched page (the cross-epoch analog of PAGE).
///
/// It sends each cell's genesis init and receives its finalization on the
/// GlobalMemory bus. The genesis `init` column is preprocessed, so the verifier
/// recomputes its commitment from the ELF — exactly PAGE's binding mechanism:
/// ELF-data pages via `page::compute_precomputed_commitment`, zero-init pages
/// (stack/heap) via the static zero-page commitment. The prover cannot choose
/// the genesis values.
fn global_memory_air(
    opts: &ProofOptions,
    config: &PageConfig,
) -> AirWithBuses<F, E, NullBoundaryConstraintBuilder, ()> {
    let air = AirWithBuses::new(
        global_memory::cols::NUM_COLUMNS,
        AuxiliaryTraceBuildData {
            interactions: global_memory::bus_interactions(config.page_base),
        },
        opts,
        1,
        empty_constraints(),
    );
    let commitment = if config.init_values.is_some() {
        page::compute_precomputed_commitment(config, opts)
    } else {
        page::zero_init_preprocessed_commitment(opts)
    };
    air.with_preprocessed(commitment, global_memory::NUM_PREPROCESSED_COLS)
}

/// The touched pages (sorted) and their ELF-derived genesis configs, rebuilt
/// identically by prover and verifier from the ELF + private input. Each cell
/// the program touched lives on one of these pages; a page in the ELF/input
/// image carries its bytes as `init`, every other (stack/heap) page is zero-init.
fn global_memory_configs(
    boundaries: &[Vec<CellBoundary>],
    elf: &Elf,
    private_inputs: &[u8],
) -> Vec<PageConfig> {
    let image = build_initial_image_paged(elf, private_inputs);
    let init_page_data = build_init_page_data(&image);
    global_memory_configs_from_init_page_data(boundaries, &init_page_data)
}

fn global_memory_configs_from_init_page_data(
    boundaries: &[Vec<CellBoundary>],
    init_page_data: &HashMap<u64, Vec<u8>>,
) -> Vec<PageConfig> {
    let touched_pages: std::collections::BTreeSet<u64> = boundaries
        .iter()
        .flatten()
        .map(|b| page::page_base_for_address(b.address))
        .collect();
    touched_pages
        .into_iter()
        .map(|page_base| match init_page_data.get(&page_base) {
            Some(data) => PageConfig::with_data(page_base, data.clone()),
            None => PageConfig::zero_init(page_base),
        })
        .collect()
}

/// Per-epoch register state and label.
struct EpochStart<'a> {
    register_init: &'a [u32],
    /// This epoch's 1-based table label (the `fini_epoch` constant).
    label: u64,
}

/// One epoch's proof plus everything a standalone verifier needs to re-check it
/// using ONLY the bundle (never the prover's in-memory traces). Each field is a
/// public value the verifier re-binds: a wrong value either makes the proof's
/// transcript challenges diverge or the AIRs not match the committed trace, so the
/// proof fails to verify.
///
/// Note: continuation epochs use the L2G memory bookend, so PAGE is skipped and the
/// per-epoch page config set is empty — the verifier builds the AIRs with no PAGE
/// tables rather than trusting any prover-supplied page config.
#[derive(serde::Serialize, serde::Deserialize)]
struct EpochProof {
    /// The epoch's STARK proof (its tables + the epoch-local L2G sub-table last).
    proof: MultiProof<F, E, ()>,
    /// Bytes this epoch committed. Concatenated in order into the run-wide output, which the
    /// GLOBAL proof binds (output-bus close + COMMIT root binding) — so this slice is a data
    /// source, not a per-epoch trusted claim.
    public_output: Vec<u8>,
    /// Statement values the epoch transcript is seeded with (re-derived on verify).
    table_counts: TableCounts,
    /// Always zero for continuation epochs: PAGE is replaced by L2G, and private
    /// input genesis is carried by the continuation bundle for global verification.
    num_private_input_pages: usize,
    /// Always empty for continuation epochs: PAGE tables are skipped, so runtime
    /// pages are not part of the epoch AIR statement.
    runtime_page_ranges: Vec<RuntimePageRange>,
    /// The epoch's final register file `R_{i+1}` (its preprocessed FINI), which the
    /// driver/verifier reuses as the next epoch's derived INIT — the cross-epoch
    /// register binding. x254 (commit index) rides along at address 508.
    reg_fini: Vec<u32>,
    /// The committed L2G table root, tied to the global proof by
    /// [`verify_l2g_commitment_binding`].
    l2g_root: Commitment,
    /// The committed COMMIT table main-trace root, tied to the global proof by
    /// [`verify_commit_commitment_binding`]. The global proof re-commits this same COMMIT
    /// trace (root-bound here) and emits its output tokens there, so the run-wide output is
    /// closed once, globally, instead of per epoch.
    commit_root: Commitment,
    /// Touched-cell boundaries; the verifier rebuilds the global AIRs (touched-page
    /// set) from these. Values are redundant with the committed L2G trace.
    boundary: Vec<CellBoundary>,
}

/// A self-contained continuation proof: the per-epoch proofs in execution order,
/// the one cross-epoch global-memory proof, and the private inputs (needed to
/// rebuild the genesis image — bound by the global proof's genesis-from-ELF check).
///
/// `verify_continuation` checks this using only the bundle and the ELF. It derives
/// serde, so it round-trips through `bincode` exactly like a monolithic `VmProof`.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct ContinuationProof {
    epochs: Vec<EpochProof>,
    global: MultiProof<F, E, ()>,
    private_inputs: Vec<u8>,
}

impl ContinuationProof {
    /// Number of epochs the execution was split into.
    pub fn num_epochs(&self) -> usize {
        self.epochs.len()
    }
}

/// Build an epoch's AIRs identically on the prove and verify sides — the single
/// source of truth for the AIR set, so the two halves can never diverge. Mirrors
/// the old integrated path: `VmAirs` (HALT included iff `is_final`), with REGISTER
/// preprocessed to INIT = `register_init` and FINI = `reg_fini`. Continuation epochs
/// use the L2G bookend, so PAGE is skipped and `page_configs` is empty. The
/// epoch-local L2G air is built separately by the caller (it needs the `label`).
fn build_epoch_airs(
    elf: &Elf,
    opts: &ProofOptions,
    page_configs: &[PageConfig],
    table_counts: &TableCounts,
    register_init: &[u32],
    reg_fini: &[u32],
    is_final: bool,
) -> VmAirs {
    // Continuation epochs preprocess FINI = R_{i+1} too (not just INIT = R_i), so the
    // final register file is a verifier-known public value bound by the REG-C2
    // Memory-bus token; reusing the same R_{i+1} as the next epoch's INIT binds
    // init(epoch i+1) == fini(epoch i).
    let register_preprocessed = Some((
        register::compute_precomputed_commitment_with_fini(opts, register_init, reg_fini),
        register::NUM_PREPROCESSED_COLS_WITH_FINI,
    ));
    let mut airs = VmAirs::new(
        elf,
        opts,
        false,
        page_configs,
        table_counts,
        None,
        is_final,
        None,
        None,
        register_preprocessed,
    );
    // The committed output is closed once in the global proof, not per epoch: rebuild the
    // epoch's COMMIT air without the output emit (its base register/memory interactions still
    // commit the trace, whose root is re-committed and bound in the global proof). See
    // `commit_global_air`.
    airs.commit = crate::test_utils::create_commit_air(opts, false);
    airs
}

/// Prove one epoch (prove half only). Commits its local-to-global table (built from
/// `boundary`) on the epoch-local Memory bus and its REGISTER table with FINI
/// preprocessed to the epoch's final register file. Returns the [`EpochProof`] the
/// standalone verifier later re-checks (does NOT verify here) plus this epoch's COMMIT
/// trace, which the caller re-commits in the global proof to close the output bus globally.
#[allow(clippy::too_many_arguments)]
fn prove_epoch(
    elf: &Elf,
    elf_bytes: &[u8],
    start: &EpochStart,
    mut traces: Traces,
    is_final: bool,
    boundary: &[CellBoundary],
    opts: &ProofOptions,
) -> Result<(EpochProof, TraceTable<F, E>), Error> {
    // Count this L2G table's range-check lookups into the BITWISE table so its
    // AreBytes/IsHalfword multiplicities balance the range-check senders.
    crate::tables::bitwise::update_multiplicities(
        &mut traces.bitwise,
        &local_to_global::collect_bitwise_from_l2g(boundary),
    );

    // Continuation epochs use the L2G bookend, so PAGE is skipped: page_configs is
    // empty. The verifier hard-codes this (passes `&[]`); check the prover agrees so
    // the two sides build identical AIRs.
    if !traces.page_configs.is_empty() {
        return Err(Error::ContinuationInvariant(
            "continuation epoch must have no PAGE configs (L2G bookend replaces PAGE)".to_string(),
        ));
    }

    // R_{i+1}, read from the committed REGISTER trace (FINI, bound to the last write).
    let reg_fini = register::fini_from_trace(&traces.register);

    let table_counts = traces.table_counts();
    let public_output = traces.public_output_bytes.clone();
    let runtime_page_ranges = traces.runtime_page_ranges();
    let num_private_input_pages = traces
        .page_configs
        .iter()
        .filter(|c| c.is_private_input)
        .count();

    let airs = build_epoch_airs(
        elf,
        opts,
        &[],
        &table_counts,
        start.register_init,
        &reg_fini,
        is_final,
    );

    let label = start.label;
    let seed = || {
        epoch_transcript(
            elf_bytes,
            &public_output,
            &table_counts,
            num_private_input_pages,
            &runtime_page_ranges,
            label,
        )
    };

    let l2g_air = l2g_memory_air(opts, label);
    // Build this epoch's L2G table from the cross-epoch boundary so it is identical
    // to the one the global proof commits (the commitment binding compares their
    // roots). It is appended to the proof below, not through `air_trace_pairs`.
    let mut l2g_trace = local_to_global::generate_local_to_global_trace(boundary);

    // Snapshot the COMMIT main trace BEFORE proving: `multi_prove` appends this epoch's aux
    // columns to `traces.commit` in place, but the global proof must re-commit a MAIN-ONLY
    // trace (its reduced air generates its own single aux column). The main-trace data — and
    // therefore its Merkle root — is identical, so `verify_commit_commitment_binding` matches.
    let commit_trace = traces.commit.clone();

    let mut pairs = airs.air_trace_pairs(&mut traces);
    pairs.push((&l2g_air, &mut l2g_trace, &()));
    let proof = Prover::multi_prove(
        pairs,
        &mut seed(),
        #[cfg(feature = "disk-spill")]
        stark::storage_mode::StorageMode::Ram,
    )
    .map_err(|e| Error::Prover(format!("{e:?}")))?;

    let l2g_root = proof
        .proofs
        .last()
        .ok_or_else(|| {
            Error::ContinuationInvariant("epoch proof is missing the L2G sub-table".to_string())
        })?
        .lde_trace_main_merkle_root;

    // The COMMIT table sits at a fixed position; its main-trace root binds the global
    // proof's re-committed copy (whose reduced air emits the output tokens).
    let commit_root = proof
        .proofs
        .get(COMMIT_TABLE_INDEX)
        .ok_or_else(|| {
            Error::ContinuationInvariant("epoch proof is missing the COMMIT sub-table".to_string())
        })?
        .lde_trace_main_merkle_root;

    Ok((
        EpochProof {
            proof,
            public_output,
            table_counts,
            num_private_input_pages,
            runtime_page_ranges,
            reg_fini,
            l2g_root,
            commit_root,
            boundary: boundary.to_vec(),
        },
        commit_trace,
    ))
}

/// Verify one epoch using ONLY the [`EpochProof`] bundle plus the verifier-derived
/// `register_init` (epoch 0: from the ELF; epoch i>0: from the previous epoch's
/// `reg_fini`), `is_final`, and `label`. Rebuilds the AIRs and transcript
/// from the bundle's statement values and indexes commits from the carried x254
/// (`register_init[X254_INDEX]`), never from the prover's memory. PAGE is skipped for
/// continuation epochs, so the AIRs are built with no page configs (the bundle does
/// not get to supply any). Returns `true` iff the proof verifies and its committed
/// L2G root matches the claimed one.
fn verify_epoch(
    elf: &Elf,
    elf_bytes: &[u8],
    epoch: &EpochProof,
    register_init: &[u32],
    is_final: bool,
    label: u64,
    opts: &ProofOptions,
) -> bool {
    // Reject degenerate table counts (mirrors the monolithic verifier).
    if epoch.table_counts.validate().is_err() {
        return false;
    }

    // Cross-check table_counts before building AIRs from bundle data. Continuation
    // epochs have no PAGE proofs, and append one epoch-local L2G proof after the VM
    // tables. HALT is present only on the final epoch.
    let fixed_tables = if is_final {
        FIXED_TABLE_COUNT
    } else {
        FIXED_TABLE_COUNT - 1
    };
    let expected_proof_count = epoch.table_counts.total() + fixed_tables + 1;
    if expected_proof_count != epoch.proof.proofs.len() {
        return false;
    }

    let airs = build_epoch_airs(
        elf,
        opts,
        &[],
        &epoch.table_counts,
        register_init,
        &epoch.reg_fini,
        is_final,
    );
    let l2g_air = l2g_memory_air(opts, label);
    let mut refs = airs.air_refs();
    refs.push(&l2g_air);

    let seed = || {
        epoch_transcript(
            elf_bytes,
            &epoch.public_output,
            &epoch.table_counts,
            epoch.num_private_input_pages,
            &epoch.runtime_page_ranges,
            label,
        )
    };

    // The epoch's COMMIT air does not emit the output token (it is carried to the global
    // proof and closed there), so the epoch's bus closes to zero — no per-epoch commit
    // offset. The output's correctness is bound globally: `verify_commit_commitment_binding`
    // + the global output-bus close over the full run-wide output.
    if !Verifier::multi_verify(&refs, &epoch.proof, &mut seed(), &FieldElement::zero()) {
        return false;
    }

    // The claimed L2G root must be the one this proof actually committed (it is what
    // verify_l2g_commitment_binding later ties to the global proof).
    let l2g_root_ok = epoch
        .proof
        .proofs
        .last()
        .map(|p| p.lde_trace_main_merkle_root)
        == Some(epoch.l2g_root);

    // Likewise the claimed COMMIT root must be the one THIS epoch proof committed — the trace
    // whose `mu`/`end`/`index`/`value` columns the epoch's constraints pinned. Without this
    // anchor, `epoch.commit_root` is a free bundle value tied only to the (also prover-supplied)
    // global proof, so a prover could put a fabricated trace's root here, re-commit that fake
    // (unconstrained, `commit_global_air`) trace in the global proof, and forge the output while
    // every other check passes. Mirrors the `l2g_root` anchor above.
    let commit_root_ok = epoch
        .proof
        .proofs
        .get(COMMIT_TABLE_INDEX)
        .map(|p| p.lde_trace_main_merkle_root)
        == Some(epoch.commit_root);

    l2g_root_ok && commit_root_ok
}

/// Build the cross-epoch global proof. It commits, in this order:
/// 1. every epoch's L2G sub-table on the GlobalMemory bus (cross-epoch memory linkage);
/// 2. every epoch's COMMIT sub-table under the reduced [`commit_global_air`], which emits
///    the committed-output tokens (the run-wide output bus);
/// 3. one GLOBAL_MEMORY table per touched page, sending each cell's genesis init
///    (preprocessed from the ELF) and receiving its final value.
///
/// The GlobalMemory bus balances iff every `fini` matches the next epoch's `init` and every
/// genesis matches the ELF; the output bus is closed by the verifier's receiver over the
/// claimed `full_output` (which is absorbed into the global statement here).
fn prove_global(
    boundaries: &[Vec<CellBoundary>],
    commit_traces: &mut [TraceTable<F, E>],
    full_output: &[u8],
    elf_bytes: &[u8],
    init_page_data: &HashMap<u64, Vec<u8>>,
    opts: &ProofOptions,
) -> Result<MultiProof<F, E, ()>, Error> {
    // Each cell's final state (boundaries are in epoch order, so the last fini wins).
    let mut final_state: global_memory::FiniStateMap = HashMap::new();
    for epoch in boundaries {
        for b in epoch {
            final_state.insert(
                b.address,
                global_memory::FiniState {
                    value: (b.fini.value & 0xFF) as u8,
                    epoch: b.fini.epoch,
                },
            );
        }
    }

    let gm_configs = global_memory_configs_from_init_page_data(boundaries, init_page_data);

    let mut l2g_traces: Vec<TraceTable<F, E>> = boundaries
        .iter()
        .map(|epoch| local_to_global::generate_local_to_global_trace(epoch))
        .collect();
    let mut gm_traces: Vec<TraceTable<F, E>> = gm_configs
        .iter()
        .map(|config| global_memory::generate_global_trace(config, &final_state))
        .collect();

    // One L2G air per epoch, each carrying its own 1-based `fini_epoch` constant.
    let l2g_airs: Vec<_> = (0..boundaries.len())
        .map(|i| l2g_global_air(opts, local_to_global::epoch_label(i as u64)))
        .collect();
    // One reduced COMMIT air per epoch (all identical; the emitted tokens come from the
    // re-committed trace). Order/count must match the epochs so the roots bind.
    let commit_airs: Vec<_> = (0..commit_traces.len())
        .map(|_| commit_global_air(opts))
        .collect();
    let gm_airs: Vec<_> = gm_configs
        .iter()
        .map(|config| global_memory_air(opts, config))
        .collect();

    let mut pairs: Vec<(AirRef, &mut TraceTable<F, E>, &())> = l2g_airs
        .iter()
        .zip(l2g_traces.iter_mut())
        .map(|(air, t)| (air as AirRef, t, &()))
        .collect();
    for (air, trace) in commit_airs.iter().zip(commit_traces.iter_mut()) {
        pairs.push((air as AirRef, trace, &()));
    }
    for (air, trace) in gm_airs.iter().zip(gm_traces.iter_mut()) {
        pairs.push((air as AirRef, trace, &()));
    }

    Prover::multi_prove(
        pairs,
        &mut global_transcript(elf_bytes, boundaries.len(), full_output),
        #[cfg(feature = "disk-spill")]
        stark::storage_mode::StorageMode::Ram,
    )
    .map_err(|e| Error::Prover(format!("{e:?}")))
}

fn verify_global(
    boundaries: &[Vec<CellBoundary>],
    proof: &MultiProof<F, E, ()>,
    full_output: &[u8],
    elf: &Elf,
    elf_bytes: &[u8],
    private_inputs: &[u8],
    opts: &ProofOptions,
) -> bool {
    // One L2G air per epoch, each with its own 1-based `fini_epoch` constant —
    // must match the order/labels the global proof committed in `prove_global`.
    let l2g_airs: Vec<_> = (0..boundaries.len())
        .map(|i| l2g_global_air(opts, local_to_global::epoch_label(i as u64)))
        .collect();
    // One reduced COMMIT air per epoch — same order/count as `prove_global`.
    let commit_airs: Vec<_> = (0..boundaries.len())
        .map(|_| commit_global_air(opts))
        .collect();
    // Rebuild the genesis configs FROM THE ELF and recompute their commitments:
    // this is the binding — a prover that claimed different genesis values would
    // commit a different root and fail to verify.
    let gm_configs = global_memory_configs(boundaries, elf, private_inputs);
    let gm_airs: Vec<_> = gm_configs
        .iter()
        .map(|config| global_memory_air(opts, config))
        .collect();

    // Order must match `prove_global`: L2G, then COMMIT, then GLOBAL_MEMORY.
    let mut refs: Vec<AirRef> = l2g_airs.iter().map(|a| a as AirRef).collect();
    for air in &commit_airs {
        refs.push(air as AirRef);
    }
    for air in &gm_airs {
        refs.push(air as AirRef);
    }

    // The global proof closes two things via one scalar bus balance (all interactions fold
    // into one LogUp sum): the GlobalMemory bus (telescopes to 0) and the committed-output
    // bus (the verifier's receiver over the claimed full output). So the expected balance is
    // 0 + the output offset. Recover (z, alpha) by replaying the global proof's Phase A over
    // the same statement-seeded transcript the multi_verify below uses.
    let mut replay_transcript = global_transcript(elf_bytes, boundaries.len(), full_output);
    let expected = match compute_commit_bus_offset_via_replay(
        &refs,
        proof,
        full_output,
        &mut replay_transcript,
    ) {
        Some(expected) => expected,
        None => return false,
    };

    Verifier::multi_verify(
        &refs,
        proof,
        &mut global_transcript(elf_bytes, boundaries.len(), full_output),
        &expected,
    )
}

/// Recover (z, alpha) from the global proof's Phase-A replay and compute the committed-output
/// bus offset over `full_output`. Factored out so the transcript fork mirrors the monolithic
/// path (`compute_expected_commit_bus_balance`).
fn compute_commit_bus_offset_via_replay(
    refs: &[AirRef],
    proof: &MultiProof<F, E, ()>,
    full_output: &[u8],
    transcript: &mut DefaultTranscript<E>,
) -> Option<FieldElement<E>> {
    let (z, alpha) = replay_transcript_phase_a(refs, proof, transcript);
    compute_commit_bus_offset(full_output, &z, &alpha)
}

/// Prove a full continuation and return a self-contained [`ContinuationProof`]
/// (prove half only — no verification). Splits the execution into `2^epoch_size_log2`
/// cycle epochs, proves each, and proves the one cross-epoch global-memory linkage.
///
/// Intermediate epochs run exactly `2^epoch_size_log2` cycles, so their CPU tables
/// have power-of-two row counts and therefore zero padding rows — important because
/// CPU padding rows participate in the inline-PC `memory` chain (carrying pc=1)
/// which is only anchored by the HALT chip's emit_pc/consume_pc, and intermediate
/// epochs exclude HALT. With padding rows present and no HALT their pc=1 tokens
/// dangle and the Memory bus fails to balance; zero padding rows sidestep that. The
/// final epoch keeps its remainder and its HALT, so its padding chain is anchored as
/// usual. A program that fits in one epoch runs as a single final (monolithic-style)
/// epoch.
pub fn prove_continuation(
    elf_bytes: &[u8],
    private_inputs: &[u8],
    epoch_size_log2: u32,
    opts: &ProofOptions,
) -> Result<ContinuationProof, Error> {
    if epoch_size_log2 < 2 {
        return Err(Error::InvalidContinuationEpochSize(
            "epoch_size_log2 must be at least 2 (4 cycles)".to_string(),
        ));
    }
    let epoch_size = 1usize.checked_shl(epoch_size_log2).ok_or_else(|| {
        Error::InvalidContinuationEpochSize(format!(
            "epoch_size_log2 {epoch_size_log2} is too large for this platform"
        ))
    })?;

    let elf = Elf::load(elf_bytes).map_err(|e| Error::ElfLoad(format!("{e}")))?;
    let mut executor = Executor::new(&elf, private_inputs.to_vec())
        .map_err(|e| Error::Execution(format!("{e}")))?;

    // The cross-epoch memory image, carried forward: epoch i+1's init is epoch i's
    // fini, updated in place with each epoch's touched-cell final values.
    let mut image = build_initial_image_paged(&elf, private_inputs);
    let init_page_data = build_init_page_data(&image);
    let mut provenance =
        local_to_global::genesis_provenance(image.iter().map(|(a, v)| (a, v as u64)));

    let mut epochs: Vec<EpochProof> = Vec::new();
    // Each epoch's COMMIT trace, retained so the global proof can re-commit them and close
    // the output bus once, globally (see `prove_global`).
    let mut commit_traces: Vec<TraceTable<F, E>> = Vec::new();
    // The previous epoch's bound final register file R_{i+1}; epoch i+1's init is
    // derived from it (the cross-epoch register binding).
    let mut prev_fini: Option<Vec<u32>> = None;

    let mut index: u64 = 0;
    loop {
        if executor.pc() == 0 {
            break;
        }
        // The cross-epoch ordering check (IsB20 on `fini_epoch - 1 - init_epoch`)
        // only spans `local_to_global::MAX_EPOCHS` epochs. Beyond that the IsB20 bus
        // cannot balance, so an honest proof is impossible — fail fast with a clear
        // error instead of building an unprovable trace. The verifier already
        // rejects any such proof; this is a prover-side guard for a clean message.
        if index >= local_to_global::MAX_EPOCHS {
            return Err(Error::InvalidContinuationEpochSize(format!(
                "execution needs more than {} continuation epochs (the IsB20 cross-epoch \
                 ordering range); use a larger epoch size",
                local_to_global::MAX_EPOCHS
            )));
        }
        let register_init: Vec<u32> = if index == 0 {
            register::register_init_from_entry_point(elf.entry_point)
        } else {
            // Epoch i+1's init is epoch i's bound fini, reused directly (same
            // `register_word_address_list` order) — the cross-epoch register binding.
            prev_fini.clone().ok_or_else(|| {
                Error::ContinuationInvariant(
                    "previous epoch final registers are missing after the first epoch".to_string(),
                )
            })?
        };

        // Run one epoch; `logs` is this epoch's chunk only (the executor clears it).
        let logs = match executor
            .resume_with_limit(epoch_size)
            .map_err(|e| Error::Execution(format!("{e}")))?
        {
            Some(logs) => logs.to_vec(),
            None => break,
        };
        let is_final = executor.pc() == 0;

        // Invariant: a non-final epoch ran the full `epoch_size` (a power of two),
        // so its CPU table has no padding rows.
        if !is_final && logs.len() != epoch_size {
            return Err(Error::ContinuationInvariant(format!(
                "intermediate epoch ran {} cycles, expected {epoch_size}",
                logs.len()
            )));
        }

        let label = local_to_global::epoch_label(index);
        let traces = Traces::from_image_and_logs(
            &elf,
            &image,
            &register_init,
            &logs,
            &MaxRowsConfig::default(),
            private_inputs,
            is_final,
            true,
            #[cfg(feature = "disk-spill")]
            stark::storage_mode::StorageMode::Ram,
        )?;
        let boundary =
            local_to_global::epoch_boundary(&mut provenance, label, &traces.touched_memory_cells);

        let start = EpochStart {
            register_init: &register_init,
            label,
        };
        let (epoch, commit_trace) =
            prove_epoch(&elf, elf_bytes, &start, traces, is_final, &boundary, opts)?;
        prev_fini = Some(epoch.reg_fini.clone());

        // Carry the image forward: this epoch's fini is the next epoch's init.
        for cell in &boundary {
            image.set(cell.address, (cell.fini.value & 0xFF) as u8);
        }
        epochs.push(epoch);
        commit_traces.push(commit_trace);

        if is_final {
            break;
        }
        index += 1;
    }

    // The run-wide committed output, aggregated in epoch order. The global proof binds it
    // (both its value, absorbed into the global statement, and that it equals the union of
    // the re-committed COMMIT tables' emitted tokens), so it is verifier-checked rather than
    // driver-trusted.
    let full_output: Vec<u8> = epochs
        .iter()
        .flat_map(|e| e.public_output.iter().copied())
        .collect();

    // One global LogUp over all the (kept) local-to-global tables and COMMIT tables.
    let all_boundaries: Vec<Vec<CellBoundary>> =
        epochs.iter().map(|e| e.boundary.clone()).collect();
    let global = prove_global(
        &all_boundaries,
        &mut commit_traces,
        &full_output,
        elf_bytes,
        &init_page_data,
        opts,
    )?;

    Ok(ContinuationProof {
        epochs,
        global,
        private_inputs: private_inputs.to_vec(),
    })
}

/// Verify a [`ContinuationProof`] using ONLY the bundle and the ELF — nothing from
/// the prover's memory. Returns `Ok(Some(public_output))` (the run-wide committed
/// bytes, reconstructed from the per-epoch bound slices) iff every check holds, else
/// `Ok(None)`.
///
/// The verifier (1) enumerates epochs itself, assigning `epoch_label` and `is_final`
/// by position (a trusted enumeration); (2) verifies each epoch, deriving its
/// `register_init` from the ELF (epoch 0) or the previous epoch's bound `reg_fini`
/// (epoch i>0) — this is the cross-epoch register binding, and forces epoch 0 to start
/// at the genesis register file; (3) closes the cross-epoch GlobalMemory bus with
/// genesis rebuilt from the ELF; (4) ties each epoch's L2G root to the global proof;
/// (5) reconstructs the output by concatenating the per-epoch slices in order.
///
/// Completeness is forced by the enumeration: epoch 0's INIT must be the ELF genesis
/// (else its preprocessed-INIT commitment mismatches), and the last epoch must be
/// `is_final` (HALT included — so the program actually terminated); a truncated run
/// would have a non-halting last epoch built with HALT and fail.
pub fn verify_continuation(
    elf_bytes: &[u8],
    bundle: &ContinuationProof,
    opts: &ProofOptions,
) -> Result<Option<Vec<u8>>, Error> {
    if bundle.private_inputs.len() as u64 > MAX_PRIVATE_INPUT_SIZE {
        return Err(Error::InvalidTableCounts(format!(
            "private input size ({}) exceeds max ({MAX_PRIVATE_INPUT_SIZE})",
            bundle.private_inputs.len()
        )));
    }

    let elf = Elf::load(elf_bytes).map_err(|e| Error::ElfLoad(format!("{e}")))?;

    let n = bundle.epochs.len();
    if n == 0 {
        return Ok(None);
    }

    // Reject a malformed bundle up front. `reg_fini` is prover-supplied (deserialized,
    // untrusted) and is indexed by `NUM_REGISTER_ADDRESSES` when building each epoch's
    // preprocessed REGISTER commitment, so a wrong length would otherwise panic the
    // verifier instead of cleanly rejecting the proof.
    if bundle
        .epochs
        .iter()
        .any(|e| e.reg_fini.len() != register::NUM_REGISTER_ADDRESSES)
    {
        return Ok(None);
    }

    // Derived from the ELF for epoch 0, then from each epoch's bound fini.
    let mut register_init = register::register_init_from_entry_point(elf.entry_point);
    let mut epoch_roots: Vec<Commitment> = Vec::with_capacity(n);
    let mut commit_roots: Vec<Commitment> = Vec::with_capacity(n);
    let mut public_output: Vec<u8> = Vec::new();

    for (index, epoch) in bundle.epochs.iter().enumerate() {
        let is_final = index == n - 1;
        let label = local_to_global::epoch_label(index as u64);

        if !verify_epoch(
            &elf,
            elf_bytes,
            epoch,
            &register_init,
            is_final,
            label,
            opts,
        ) {
            return Ok(None);
        }

        epoch_roots.push(epoch.l2g_root);
        commit_roots.push(epoch.commit_root);
        // The aggregate output; its correctness is enforced globally below (the global
        // output-bus close over exactly these bytes + the COMMIT root binding), so this is
        // no longer a driver-trusted concatenation.
        public_output.extend_from_slice(&epoch.public_output);
        // Next epoch's init is this epoch's bound fini — the cross-epoch register
        // (and x254) binding. A mismatched fini desyncs the next epoch's AIRs.
        register_init = epoch.reg_fini.clone();
    }

    // Cross-epoch global memory: genesis rebuilt FROM THE ELF (+ private inputs), so the
    // starting memory cannot be prover-chosen (the bus telescopes fini→init). The same proof
    // also closes the committed-output bus against `public_output` (bound into its statement).
    let all_boundaries: Vec<Vec<CellBoundary>> =
        bundle.epochs.iter().map(|e| e.boundary.clone()).collect();
    if !verify_global(
        &all_boundaries,
        &bundle.global,
        &public_output,
        &elf,
        elf_bytes,
        &bundle.private_inputs,
        opts,
    ) {
        return Ok(None);
    }

    // Each epoch's committed L2G table is the same one the global proof used.
    if !verify_l2g_commitment_binding(&epoch_roots, &bundle.global) {
        return Ok(None);
    }

    // Each epoch's committed COMMIT table is the same one the global proof re-committed and
    // emitted output tokens from. The COMMIT sub-tables follow the `n` L2G sub-tables in the
    // global proof, so they start at offset `n`. This ties the globally-closed output to the
    // epochs' pinned COMMIT traces.
    if !verify_commit_commitment_binding(&commit_roots, &bundle.global, n) {
        return Ok(None);
    }

    Ok(Some(public_output))
}

/// Convenience wrapper: prove then verify in one call (the original integrated API).
/// Returns `Ok(Some(public_output))` iff the continuation proves and verifies.
pub fn prove_and_verify_continuation(
    elf_bytes: &[u8],
    private_inputs: &[u8],
    epoch_size_log2: u32,
    opts: &ProofOptions,
) -> Result<Option<Vec<u8>>, Error> {
    let bundle = prove_continuation(elf_bytes, private_inputs, epoch_size_log2, opts)?;
    verify_continuation(elf_bytes, &bundle, opts)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::asm_elf_bytes;

    // `test_commit_split` issues two Commit syscalls, one early and one late, so a
    // small epoch puts the second commit in a later epoch. That epoch starts with
    // x254 > 0 (the carried commit index), which exercises the cross-epoch commit
    // indexing: both the COMMIT trace and the verifier's `compute_commit_bus_offset`
    // index from the carried x254 rather than 0. Regression test for that fix.
    #[test]
    fn test_commit_across_epochs_verifies() {
        let _ = env_logger::builder().is_test(true).try_init();
        let elf_bytes = asm_elf_bytes("test_commit_split");
        let expected_output: [u8; 4] = [0xAA, 0xBB, 0xCC, 0xDD];

        let total = Executor::new(&Elf::load(&elf_bytes).unwrap(), vec![])
            .unwrap()
            .run()
            .unwrap()
            .logs
            .len();

        // Both commits in a single 64-cycle epoch (x254 starts at 0).
        let single = prove_and_verify_continuation(
            &elf_bytes,
            &[],
            6,
            &ProofOptions::default_test_options(),
        )
        .unwrap();
        assert_eq!(single.as_deref(), Some(&expected_output[..]));
        assert!(total <= (1 << 6), "single-epoch log2 must cover the run");

        // The late commit (only `halt` follows it) lands past the midpoint, so a
        // 16-cycle epoch forces it into a later epoch where x254 is already 2.
        // Prove first so we can assert the run actually split into >1 epoch — without
        // this the test would silently pass even if it degraded to a single epoch.
        let bundle =
            prove_continuation(&elf_bytes, &[], 4, &ProofOptions::default_test_options()).unwrap();
        assert!(
            bundle.num_epochs() > 1,
            "16-cycle epochs must split the run into multiple epochs"
        );
        let split = verify_continuation(&elf_bytes, &bundle, &ProofOptions::default_test_options())
            .unwrap();
        assert_eq!(
            split.as_deref(),
            Some(&expected_output[..]),
            "commit in a later epoch must verify and aggregate to the same output"
        );
    }

    // A memory-heavy multi-epoch continuation. `all_loadstore_32` is ~34 cycles, so
    // `epoch_size_log2 = 3` (8 cycles) yields several intermediate epochs (each an
    // exact power-of-two cycle count → no CPU padding rows) plus a final epoch.
    #[test]
    fn test_prove_and_verify_continuation() {
        let _ = env_logger::builder().is_test(true).try_init();
        let elf_bytes = asm_elf_bytes("all_loadstore_32");
        let epoch_size_log2 = 3;
        let epoch_size = 8;
        // Guard against silent degradation: the program must be longer than one
        // epoch, otherwise this collapses to a single final epoch and stops testing
        // the cross-epoch (intermediate-epoch) path.
        let total = Executor::new(&Elf::load(&elf_bytes).unwrap(), vec![])
            .unwrap()
            .run()
            .unwrap()
            .logs
            .len();
        assert!(
            total > epoch_size,
            "program too short ({total} cycles) to exercise intermediate epochs"
        );
        assert!(
            prove_and_verify_continuation(
                &elf_bytes,
                &[],
                epoch_size_log2,
                &ProofOptions::default_test_options()
            )
            .unwrap()
            .is_some()
        );
    }

    // Regression for touched-cell prediction from carried registers. A syscall
    // whose operand pointers live in registers (ECSM reads a0/a1/a2) can have those
    // registers set in an EARLIER epoch than the call. `test_ecsm_split` sets
    // a0/a1/a2 at the very start and runs the ECSM ~46 cycles later;
    // `epoch_size_log2 = 5` (32 cycles) puts the pointer setup in epoch 0 and the
    // ecall in epoch 1. The per-epoch touched-cell pass must carry registers across
    // the boundary — otherwise it reads the pointers as 0, mispredicts the touched
    // cells (and the ECSM operands), and the epoch cannot verify.
    #[test]
    fn test_ecsm_across_epochs_verifies() {
        let _ = env_logger::builder().is_test(true).try_init();
        let elf_bytes = asm_elf_bytes("test_ecsm_split");
        let total = Executor::new(&Elf::load(&elf_bytes).unwrap(), vec![])
            .unwrap()
            .run()
            .unwrap()
            .logs
            .len();
        assert!(total > 32, "the ECSM ecall must fall past the first epoch");
        let out = prove_and_verify_continuation(
            &elf_bytes,
            &[],
            5,
            &ProofOptions::default_test_options(),
        )
        .unwrap();
        assert!(
            out.is_some(),
            "an ECSM whose pointer registers were set in an earlier epoch must still verify"
        );
    }

    // Guards that the continuation API takes `epoch_size_log2` directly. A log2 of
    // 4 produces 16-cycle epochs over the 33-cycle `test_commit_split`, putting its
    // two commits in different epochs and exercising the cross-epoch x254 carry.
    #[test]
    fn test_continuation_epoch_size_log2() {
        let _ = env_logger::builder().is_test(true).try_init();
        let elf_bytes = asm_elf_bytes("test_commit_split");
        let out = prove_and_verify_continuation(
            &elf_bytes,
            &[],
            4,
            &ProofOptions::default_test_options(),
        )
        .unwrap();
        assert_eq!(out.as_deref(), Some(&[0xAA, 0xBB, 0xCC, 0xDD][..]));
    }

    #[test]
    fn test_continuation_rejects_too_small_epoch_size_log2() {
        assert!(matches!(
            prove_continuation(&[], &[], 1, &ProofOptions::default_test_options()),
            Err(Error::InvalidContinuationEpochSize(_))
        ));
    }

    // ---- Standalone (split) prover/verifier ----

    // Round-trip: a bundle from prove_continuation verifies on its own (only the
    // bundle + ELF) and reconstructs the exact run-wide output.
    #[test]
    fn test_split_verify_roundtrip() {
        let _ = env_logger::builder().is_test(true).try_init();
        let elf_bytes = asm_elf_bytes("test_commit_split");
        let bundle =
            prove_continuation(&elf_bytes, &[], 4, &ProofOptions::default_test_options()).unwrap();
        let out = verify_continuation(&elf_bytes, &bundle, &ProofOptions::default_test_options())
            .unwrap();
        assert_eq!(out.as_deref(), Some(&[0xAA, 0xBB, 0xCC, 0xDD][..]));
    }

    // A bundle survives a bincode round-trip and still verifies to the same output —
    // the serialization path the CLI's `prove`/`verify --continuations` relies on.
    #[test]
    fn test_continuation_bincode_roundtrip() {
        let _ = env_logger::builder().is_test(true).try_init();
        let elf_bytes = asm_elf_bytes("test_commit_split");
        let bundle =
            prove_continuation(&elf_bytes, &[], 4, &ProofOptions::default_test_options()).unwrap();

        let bytes = bincode::serialize(&bundle).unwrap();
        let restored: ContinuationProof = bincode::deserialize(&bytes).unwrap();

        let out = verify_continuation(&elf_bytes, &restored, &ProofOptions::default_test_options())
            .unwrap();
        assert_eq!(out.as_deref(), Some(&[0xAA, 0xBB, 0xCC, 0xDD][..]));
    }

    // Negative: dropping the final (halting) epoch must be rejected — the new last
    // epoch is non-halting but the verifier builds it as `is_final` (HALT included),
    // so it can't verify. Guards completeness / no-truncation.
    #[test]
    fn test_split_verify_rejects_dropped_last_epoch() {
        let _ = env_logger::builder().is_test(true).try_init();
        let elf_bytes = asm_elf_bytes("all_loadstore_32");
        let mut bundle =
            prove_continuation(&elf_bytes, &[], 3, &ProofOptions::default_test_options()).unwrap();
        assert!(bundle.epochs.len() >= 3, "need multiple epochs");
        bundle.epochs.pop();
        assert!(
            verify_continuation(&elf_bytes, &bundle, &ProofOptions::default_test_options())
                .unwrap()
                .is_none()
        );
    }

    // Negative: reordering epochs must be rejected — each epoch proof is bound to its
    // 1-based label (and register chain), so a swapped epoch fails to verify. Guards
    // the trusted-enumeration ordering.
    #[test]
    fn test_split_verify_rejects_reordered_epochs() {
        let _ = env_logger::builder().is_test(true).try_init();
        let elf_bytes = asm_elf_bytes("all_loadstore_32");
        let mut bundle =
            prove_continuation(&elf_bytes, &[], 3, &ProofOptions::default_test_options()).unwrap();
        assert!(bundle.epochs.len() >= 3, "need multiple epochs");
        bundle.epochs.swap(0, 1);
        assert!(
            verify_continuation(&elf_bytes, &bundle, &ProofOptions::default_test_options())
                .unwrap()
                .is_none()
        );
    }

    // Negative: corrupting an epoch's bound final register file (R_{i+1}) must be
    // rejected — the verifier derives the next epoch's INIT from it, so it no longer
    // matches that epoch's committed preprocessed INIT. Guards the cross-epoch
    // register binding (incl. x254).
    #[test]
    fn test_split_verify_rejects_tampered_register_fini() {
        let _ = env_logger::builder().is_test(true).try_init();
        let elf_bytes = asm_elf_bytes("all_loadstore_32");
        let mut bundle =
            prove_continuation(&elf_bytes, &[], 3, &ProofOptions::default_test_options()).unwrap();
        assert!(
            bundle.epochs.len() >= 2,
            "need a second epoch to chain into"
        );
        bundle.epochs[0].reg_fini[0] ^= 1;
        assert!(
            verify_continuation(&elf_bytes, &bundle, &ProofOptions::default_test_options())
                .unwrap()
                .is_none()
        );
    }

    // Negative: a malformed bundle whose `reg_fini` has the wrong length must be
    // rejected with `Ok(None)`, not panic the verifier. `reg_fini` is deserialized
    // (untrusted) and indexed by `NUM_REGISTER_ADDRESSES` when building the
    // preprocessed REGISTER commitment, so a short one would otherwise be an
    // out-of-bounds panic in release builds.
    #[test]
    fn test_split_verify_rejects_malformed_register_fini_length() {
        let _ = env_logger::builder().is_test(true).try_init();
        let elf_bytes = asm_elf_bytes("all_loadstore_32");
        let mut bundle =
            prove_continuation(&elf_bytes, &[], 3, &ProofOptions::default_test_options()).unwrap();
        assert!(!bundle.epochs.is_empty());
        bundle.epochs[0].reg_fini.pop();
        assert!(
            verify_continuation(&elf_bytes, &bundle, &ProofOptions::default_test_options())
                .unwrap()
                .is_none()
        );
    }

    // Negative: table_counts are bundle data. Inflating a positive count must be
    // rejected before the verifier builds AIRs from the malformed shape.
    #[test]
    fn test_split_verify_rejects_inflated_epoch_table_count() {
        let _ = env_logger::builder().is_test(true).try_init();
        let elf_bytes = asm_elf_bytes("all_loadstore_32");
        let mut bundle =
            prove_continuation(&elf_bytes, &[], 8, &ProofOptions::default_test_options()).unwrap();
        bundle.epochs[0].table_counts.cpu += 1;
        assert!(
            verify_continuation(&elf_bytes, &bundle, &ProofOptions::default_test_options())
                .unwrap()
                .is_none()
        );
    }

    // Negative: the verifier rebuilds private-input genesis from bundle bytes.
    // Changing those bytes after proving changes the global-memory preprocessed
    // genesis commitment, so the standalone verifier must reject.
    #[test]
    fn test_split_verify_rejects_tampered_private_input_genesis() {
        let _ = env_logger::builder().is_test(true).try_init();
        let elf_bytes = asm_elf_bytes("test_private_input_xpage");
        let private_inputs: Vec<u8> = (0u8..16).collect();
        let mut bundle = prove_continuation(
            &elf_bytes,
            &private_inputs,
            4,
            &ProofOptions::default_test_options(),
        )
        .unwrap();
        assert!(
            verify_continuation(&elf_bytes, &bundle, &ProofOptions::default_test_options())
                .unwrap()
                .is_some(),
            "baseline must verify before tampering"
        );

        bundle.private_inputs[4] ^= 0xFF;
        assert!(
            verify_continuation(&elf_bytes, &bundle, &ProofOptions::default_test_options())
                .unwrap()
                .is_none()
        );
    }

    // Negative: verifier-side private inputs are deserialized/untrusted, so reject
    // oversized bundles before rebuilding genesis page configs from them.
    #[test]
    fn test_split_verify_rejects_oversized_private_inputs() {
        let _ = env_logger::builder().is_test(true).try_init();
        let elf_bytes = asm_elf_bytes("all_loadstore_32");
        let mut bundle =
            prove_continuation(&elf_bytes, &[], 8, &ProofOptions::default_test_options()).unwrap();
        bundle.private_inputs = vec![0; MAX_PRIVATE_INPUT_SIZE as usize + 1];
        assert!(matches!(
            verify_continuation(&elf_bytes, &bundle, &ProofOptions::default_test_options()),
            Err(Error::InvalidTableCounts(_))
        ));
    }

    // The bundle's `boundary` field is used only to rebuild the global AIRs' touched-
    // PAGE set (genesis is recomputed from the ELF). The cross-epoch memory values
    // live in the committed L2G traces, tied to the epoch proofs by
    // `verify_l2g_commitment_binding` (exercised by test_split_verify_rejects_tampered_l2g_root
    // below). Tampering a boundary value is therefore inconsequential; omitting/adding
    // a touched page is caught by the GlobalMemory bus (unmatched fini / air count
    // mismatch). So there is no meaningful "tamper a boundary value" negative test.

    // Negative: corrupting an epoch's claimed L2G table root must be rejected —
    // `verify_l2g_commitment_binding` compares each epoch's `l2g_root` against the
    // corresponding sub-proof root in the global proof, so a mismatched root causes
    // the binding to fail. Guards the L2G root↔global commitment binding.
    #[test]
    fn test_split_verify_rejects_tampered_l2g_root() {
        let _ = env_logger::builder().is_test(true).try_init();
        let elf_bytes = asm_elf_bytes("all_loadstore_32");
        let mut bundle =
            prove_continuation(&elf_bytes, &[], 3, &ProofOptions::default_test_options()).unwrap();
        assert!(
            bundle.epochs.len() >= 2,
            "need multiple epochs to exercise the binding"
        );
        bundle.epochs[0].l2g_root[0] ^= 0xFF;
        assert!(
            verify_continuation(&elf_bytes, &bundle, &ProofOptions::default_test_options())
                .unwrap()
                .is_none()
        );
    }

    // Negative: corrupting an epoch's claimed COMMIT table root must be rejected —
    // `verify_commit_commitment_binding` compares each epoch's `commit_root` against the
    // corresponding re-committed COMMIT sub-proof root in the global proof (at offset `n`),
    // so a mismatch breaks the binding. Guards the COMMIT root↔global commitment binding,
    // which is what ties the globally-closed output to the epochs' pinned COMMIT traces.
    #[test]
    fn test_split_verify_rejects_tampered_commit_root() {
        let _ = env_logger::builder().is_test(true).try_init();
        // `test_commit_split` actually commits bytes, so the COMMIT trace is non-trivial.
        let elf_bytes = asm_elf_bytes("test_commit_split");
        let mut bundle =
            prove_continuation(&elf_bytes, &[], 4, &ProofOptions::default_test_options()).unwrap();
        assert!(bundle.num_epochs() > 1, "need multiple epochs");
        bundle.epochs[0].commit_root[0] ^= 0xFF;
        assert!(
            verify_continuation(&elf_bytes, &bundle, &ProofOptions::default_test_options())
                .unwrap()
                .is_none(),
            "a tampered COMMIT root must break verify_commit_commitment_binding"
        );
    }

    // Negative (the load-bearing one): `commit_root` must be anchored to the epoch proof's OWN
    // COMMIT trace, not merely be internally consistent with the global proof. Here we rebuild
    // the global proof over structurally-different COMMIT traces (built from the honest output
    // ops, so the global bus still closes against the true output) whose Merkle roots differ
    // from the epochs' real COMMIT traces, and repoint every `commit_root` at those. The global
    // binding (`verify_commit_commitment_binding`) is then internally consistent — so ONLY the
    // per-epoch `verify_epoch` anchor (`commit_root == proofs[COMMIT_TABLE_INDEX]`) can reject
    // it. If that anchor is removed, a prover could likewise re-commit *fabricated* traces
    // emitting a FORGED output and have it accepted; this test fails closed on that regression.
    #[test]
    fn test_split_verify_rejects_commit_root_not_anchored_to_epoch() {
        use crate::tables::commit::{CommitOperation, generate_commit_trace};

        let _ = env_logger::builder().is_test(true).try_init();
        let elf_bytes = asm_elf_bytes("test_commit_split");
        let opts = ProofOptions::default_test_options();
        let mut bundle = prove_continuation(&elf_bytes, &[], 4, &opts).unwrap();
        let n = bundle.epochs.len();
        assert!(n > 1, "need multiple epochs");

        // Baseline: honest bundle verifies to the true output.
        let full_output = verify_continuation(&elf_bytes, &bundle, &opts)
            .unwrap()
            .expect("honest bundle must verify");

        // Rebuild the global proof over alternate COMMIT traces that emit the SAME output tokens
        // (one op per committed byte at its true global index) but have a different layout →
        // different roots. `commit_global_air` has `empty_constraints`, so only the emitted
        // tokens matter for the global bus, and they still equal `{(i, full_output[i])}`.
        let elf = Elf::load(&elf_bytes).unwrap();
        let image = build_initial_image_paged(&elf, &[]);
        let init_page_data = build_init_page_data(&image);
        let boundaries: Vec<Vec<CellBoundary>> =
            bundle.epochs.iter().map(|e| e.boundary.clone()).collect();

        let mut global_index: u64 = 0;
        let mut alt_traces: Vec<TraceTable<F, E>> = Vec::with_capacity(n);
        for epoch in &bundle.epochs {
            let ops: Vec<CommitOperation> = epoch
                .public_output
                .iter()
                .enumerate()
                .map(|(k, &value)| CommitOperation {
                    timestamp: 0,
                    index: global_index + k as u64,
                    address: 0,
                    count: 1,
                    first: k == 0,
                    end: false,
                    value,
                })
                .collect();
            global_index += epoch.public_output.len() as u64;
            alt_traces.push(generate_commit_trace(&ops));
        }

        let alt_global = prove_global(
            &boundaries,
            &mut alt_traces,
            &full_output,
            &elf_bytes,
            &init_page_data,
            &opts,
        )
        .unwrap();

        // Repoint each commit_root at the alternate global's re-committed roots (offset n) and
        // swap in the alternate global proof, so the global binding stays internally consistent.
        for (i, epoch) in bundle.epochs.iter_mut().enumerate() {
            epoch.commit_root = alt_global.proofs[n + i].lde_trace_main_merkle_root;
        }
        bundle.global = alt_global;

        assert!(
            verify_continuation(&elf_bytes, &bundle, &opts)
                .unwrap()
                .is_none(),
            "a commit_root not anchored to the epoch's own COMMIT trace must be rejected"
        );
    }

    // Negative: tampering an epoch's committed output bytes must be rejected. The output is
    // only bound globally now (the epoch no longer closes its own commit slice), but the
    // tampered slice also diverges the epoch's own statement-seeded transcript — either way
    // the run must fail to verify, and the aggregate the verifier returns can never be a
    // forged output.
    #[test]
    fn test_split_verify_rejects_tampered_output() {
        let _ = env_logger::builder().is_test(true).try_init();
        let elf_bytes = asm_elf_bytes("test_commit_split");
        let mut bundle =
            prove_continuation(&elf_bytes, &[], 4, &ProofOptions::default_test_options()).unwrap();
        // Find an epoch that actually committed a byte and flip it.
        let epoch = bundle
            .epochs
            .iter_mut()
            .find(|e| !e.public_output.is_empty())
            .expect("some epoch must commit a byte in test_commit_split");
        epoch.public_output[0] ^= 0xFF;
        assert!(
            verify_continuation(&elf_bytes, &bundle, &ProofOptions::default_test_options())
                .unwrap()
                .is_none(),
            "a tampered committed byte must not verify to a forged output"
        );
    }
}
