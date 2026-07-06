//! First production implementation of continuations (Approach 2).
//!
//! Splits an execution into fixed-size epochs, proves each epoch independently
//! (its memory is initialized/finalized by the per-epoch local-to-global table),
//! and proves one cross-epoch "global memory" LogUp that links every epoch's
//! `fini` to the next epoch's `init` (so `fini(epoch i) == init(epoch i+1)`).
//!
//! The global proof's genesis anchor is bound to the ELF: for ELF/runtime pages the
//! verifier recomputes the per-page preprocessed init commitment from the ELF in
//! `verify_global`, so the starting memory cannot be prover-supplied. Private-input
//! pages are the one exception — their genesis is committed (non-preprocessed), exactly
//! as the monolithic prover does, with correctness enforced by the GlobalMemory bus
//! rather than ELF recomputation, so the raw private input is neither carried in the
//! proof bundle nor reconstructed by the verifier.
//!
//! Scope of the privacy guarantee: this is NOT zero-knowledge. Like every non-ZK STARK
//! column, the committed private genesis is opened at FRI query positions, so this does
//! not cryptographically hide the private input — it only guarantees the raw input is
//! not bundled and not recomputed by the verifier. Cryptographic hiding would require a
//! ZK/blinded proof system.
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
//! The x254 commit index is carried across epochs by that same register binding,
//! so a continuation epoch indexes its commits from the carried value: both the
//! COMMIT trace (`current_commit_index` seeded from x254) and the verifier's
//! `compute_commit_bus_offset` (a `start_index` parameter) count from it, and the
//! driver concatenates each epoch's committed bytes into the run-wide output.
//!
//! The prover and verifier are split: `prove_continuation` emits a self-contained
//! `ContinuationProof` bundle and `verify_continuation` checks it from the bundle
//! and ELF alone; `prove_and_verify_continuation` proves and verifies in one
//! streaming pass, verifying and dropping each epoch proof so its retained-proof
//! memory stays bounded to O(1) epochs (at most two are live across the one-epoch
//! `is_final` lookahead) instead of collecting all of them.

use std::collections::HashMap;

use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use executor::elf::Elf;
use executor::vm::execution::Executor;
use executor::vm::memory::MAX_PRIVATE_INPUT_SIZE;
use math::field::element::FieldElement;
use stark::config::Commitment;
use stark::constraints::builder::{ConstraintBuilder, ConstraintSet, EmptyConstraints};
use stark::lookup::{AirWithBuses, AuxiliaryTraceBuildData, NullBoundaryConstraintBuilder};
use stark::proof::options::ProofOptions;
use stark::proof::stark::MultiProof;
use stark::prover::{IsStarkProver, Prover};
use stark::trace::TraceTable;
use stark::traits::AIR;
use stark::verifier::{IsStarkVerifier, Verifier};

use crate::paged_mem::PagedMem;
use crate::statement::{StatementKind, absorb_continuation_global_statement, absorb_statement};
use crate::tables::local_to_global::{self, CellBoundary};
use crate::tables::page::{self, PageConfig};
use crate::tables::register;
use crate::tables::trace_builder::{Traces, build_init_page_data, build_initial_image_paged};
use crate::tables::types::{GoldilocksExtension, GoldilocksField};
use crate::tables::{MaxRowsConfig, global_memory};
use crate::{
    Error, FIXED_TABLE_COUNT, RuntimePageRange, TableCounts, VmAirs,
    compute_expected_commit_bus_balance, verify_l2g_commitment_binding,
};

type F = GoldilocksField;
type E = GoldilocksExtension;
type AirRef<'a> = &'a dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>;

/// Fresh transcript seeded with the epoch's statement (ELF, public output, table
/// layout) and `epoch_label` (its position). The epoch's prove, verify, and
/// bus-balance replay all seed via this so their challenges match; the seeding
/// pins each epoch proof to its program and position (replay protection).
fn epoch_transcript(
    elf_bytes: &[u8],
    public_output: &[u8],
    table_counts: &TableCounts,
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
        // Continuation epochs skip PAGE (the L2G bookend replaces it), so they never
        // have private-input pages — the private-input count is always 0 here.
        0,
        runtime_page_ranges,
    );
    transcript
}

/// Fresh transcript seeded with the global proof's statement (ELF + epoch count).
/// `prove_global` and `verify_global` both seed via this so their challenges match.
fn global_transcript(
    elf_bytes: &[u8],
    num_epochs: usize,
    num_private_input_pages: usize,
    touched_page_bases: &[u64],
) -> DefaultTranscript<E> {
    let mut transcript = DefaultTranscript::<E>::new(&[]);
    absorb_continuation_global_statement(
        &mut transcript,
        elf_bytes,
        num_epochs,
        num_private_input_pages,
        touched_page_bases,
    );
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
/// The L2G epoch-local table's single transition constraint: `MU ∈ {0,1}`
/// (`MU·(1−MU) = 0`) at constraint index 0.
struct L2gMemoryConstraints;

impl ConstraintSet<F, E> for L2gMemoryConstraints {
    fn eval<B: ConstraintBuilder<F, E>>(&self, b: &mut B) {
        crate::constraints::templates::emit_is_bit(b, 0, local_to_global::cols::MU, None);
    }
}

/// Local-to-global AIR on the cross-epoch GlobalMemory bus (used in the global proof).
///
/// `epoch_label` is this epoch's 1-based label; it is the `fini_epoch` constant
/// the fini token carries (not a trace column, since it's the same for every row).
///
/// Uses the `EmptyConstraints` set deliberately: the MU boolean (`MU·(1-MU)=0`), the
/// column range checks, and the `init_epoch < fini_epoch` ordering are NOT
/// re-asserted here. They are enforced once in the epoch proof's `l2g_memory_air`,
/// and `verify_l2g_commitment_binding` ties this global L2G sub-table to the *same*
/// committed trace (equal Merkle roots). So under collision resistance the trace the
/// global bus runs over already satisfies all those constraints — do not add them
/// here (it would be redundant, not a missing check).
fn l2g_global_air(
    opts: &ProofOptions,
    epoch_label: u64,
) -> AirWithBuses<F, E, NullBoundaryConstraintBuilder, (), EmptyConstraints> {
    AirWithBuses::new(
        local_to_global::cols::NUM_COLUMNS,
        AuxiliaryTraceBuildData {
            interactions: local_to_global::bus_interactions(epoch_label),
        },
        opts,
        1,
        EmptyConstraints,
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
) -> AirWithBuses<F, E, NullBoundaryConstraintBuilder, (), L2gMemoryConstraints> {
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
        L2gMemoryConstraints,
    )
}

/// GLOBAL_MEMORY AIR for one touched page (the cross-epoch analog of PAGE).
///
/// It sends each cell's genesis init and receives its finalization on the
/// GlobalMemory bus. For ELF/runtime pages the genesis `init` column is
/// preprocessed, so the verifier recomputes its commitment from the ELF — exactly
/// PAGE's binding mechanism: ELF-data pages via `page::compute_precomputed_commitment`,
/// zero-init pages (stack/heap) via the static zero-page commitment. The prover
/// cannot choose those genesis values.
///
/// Private-input pages are built NON-preprocessed (mirrors the monolithic PAGE in
/// `VmAirs::new`): INIT is a committed main-trace column the verifier never recomputes
/// from the ELF, so the raw private input is neither bundled nor reconstructed by the
/// verifier. Correctness is enforced by the GlobalMemory bus (the genesis token must
/// telescope into the epochs' reads), not by ELF recomputation. (Not a ZK/hiding claim —
/// the committed column is still opened at STARK query positions.)
fn global_memory_air(
    opts: &ProofOptions,
    config: &PageConfig,
) -> AirWithBuses<F, E, NullBoundaryConstraintBuilder, (), EmptyConstraints> {
    let air = AirWithBuses::new(
        global_memory::cols::NUM_COLUMNS,
        AuxiliaryTraceBuildData {
            interactions: global_memory::bus_interactions(config.page_base),
        },
        opts,
        1,
        EmptyConstraints,
    );
    if config.is_private_input {
        return air;
    }
    let commitment = if config.init_values.is_some() {
        page::compute_precomputed_commitment(config, opts)
    } else {
        page::zero_init_preprocessed_commitment(opts)
    };
    air.with_preprocessed(commitment, global_memory::NUM_PREPROCESSED_COLS)
}

/// The sorted, deduped set of page bases the touched cells fall on — the SINGLE source
/// of truth for which GLOBAL_MEMORY tables exist. The prover builds the committed tables
/// from this list, ships the identical list in the bundle (`ContinuationProof.touched_page_bases`),
/// and the verifier rebuilds the same tables from it. Sorted (BTreeSet order) so prover
/// and verifier iterate the identical sequence — `multi_verify` matches AIRs to sub-proofs
/// positionally. Carries page bases ONLY: no cell values, so private-input bytes never
/// enter the bundle (unlike the full `CellBoundary`, whose `init.value` is a private byte).
fn touched_page_bases(boundaries: &[Vec<CellBoundary>]) -> Vec<u64> {
    boundaries
        .iter()
        .flatten()
        .map(|b| page::page_base_for_address(b.address))
        .collect::<std::collections::BTreeSet<u64>>()
        .into_iter()
        .collect()
}

/// Canonicalize a possibly-untrusted, out-of-order page-base list to the same sorted,
/// deduped form the prover produces via [`touched_page_bases`], so the verifier rebuilds
/// tables in the committed order regardless of the wire order (a shuffled-but-same-set
/// list still verifies; a different set fails via bus imbalance / AIR-count mismatch).
fn canonical_page_bases(page_bases: &[u64]) -> Vec<u64> {
    page_bases
        .iter()
        .copied()
        .collect::<std::collections::BTreeSet<u64>>()
        .into_iter()
        .collect()
}

/// The touched pages' genesis configs, for the VERIFIER: built from the ELF alone (no
/// private bytes). `page_bases` is the canonical touched-page-base list. An ELF data page
/// carries its bytes as `init`, every other (stack/heap) page is zero-init.
///
/// Private-input pages are built NON-preprocessed, so the verifier never recomputes their
/// genesis from the ELF and never needs the raw private bytes. They are identified EXACTLY
/// as the monolithic verifier does — the first `num_private_input_pages` pages from
/// `PRIVATE_INPUT_START_INDEX` (see [`page::is_private_input_page`]).
fn global_memory_configs(
    page_bases: &[u64],
    elf: &Elf,
    num_private_input_pages: usize,
) -> Vec<PageConfig> {
    // No private bytes: the verifier only builds the AIRs, and private-input pages are
    // non-preprocessed (their INIT is never recomputed).
    let image = build_initial_image_paged(elf, &[]);
    let init_page_data = build_init_page_data(&image);
    global_memory_configs_from_init_page_data(
        page_bases,
        &init_page_data,
        num_private_input_pages,
        false,
    )
}

/// Shared genesis-config builder for prover and verifier, one `PageConfig` per page base
/// in `page_bases` (which must be canonical: sorted + deduped). `init_page_data` holds
/// each page's genesis bytes (ELF + private input on the prover side; ELF only on the
/// verifier side).
///
/// `include_private_genesis` — whether a private-input page's genesis bytes are loaded
/// from `init_page_data` into its config. The PROVER passes `true`: those bytes become
/// the committed INIT column. The VERIFIER passes `false`: its AIR for a private page is
/// non-preprocessed and never consults `init_values` (and its `init_page_data` is built
/// from the ELF alone, so there is nothing to load) — the config carries an explicitly
/// empty vec so no code path can silently start depending on verifier-side private data.
fn global_memory_configs_from_init_page_data(
    page_bases: &[u64],
    init_page_data: &HashMap<u64, Vec<u8>>,
    num_private_input_pages: usize,
    include_private_genesis: bool,
) -> Vec<PageConfig> {
    page_bases
        .iter()
        .map(|&page_base| {
            if page::is_private_input_page(page_base, num_private_input_pages) {
                let data = if include_private_genesis {
                    init_page_data.get(&page_base).cloned().unwrap_or_default()
                } else {
                    Vec::new()
                };
                PageConfig::with_private_input(page_base, data)
            } else {
                match init_page_data.get(&page_base) {
                    Some(data) => PageConfig::with_data(page_base, data.clone()),
                    None => PageConfig::zero_init(page_base),
                }
            }
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
    /// Bytes this epoch committed — the COMMIT-bus receiver reference.
    public_output: Vec<u8>,
    /// Statement values the epoch transcript is seeded with (re-derived on verify).
    table_counts: TableCounts,
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
}

/// A self-contained continuation proof: the per-epoch proofs in execution order, the one
/// cross-epoch global-memory proof, the number of private-input pages, and the touched
/// page-base set.
///
/// NO cell values are carried. The raw private input is not bundled (mirrors
/// `VmProof.num_private_input_pages`), and — since the per-epoch `CellBoundary` list
/// (whose `init.value` is a private-input byte for private reads) is NOT serialized —
/// touched-cell values never leave the prover either. The verifier only ever needed the
/// epoch count and the touched page-base set from those boundaries; both are preserved
/// (`epochs.len()` and `touched_page_bases`) at page granularity, value-free. Private-input
/// genesis lives in committed, bus-enforced GLOBAL_MEMORY columns the verifier never
/// recomputes. Both public values (`num_private_input_pages`, `touched_page_bases`) are
/// bound into the global Fiat-Shamir statement and pinned by the GlobalMemory bus /
/// AIR-count checks, so a wrong value is rejected; the count is also bound-checked up front.
///
/// `verify_continuation` checks this using only the bundle and the ELF. It derives
/// serde, so it round-trips through `bincode` exactly like a monolithic `VmProof`.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct ContinuationProof {
    epochs: Vec<EpochProof>,
    global: MultiProof<F, E, ()>,
    num_private_input_pages: usize,
    /// Sorted, deduped page bases the run touched — the verifier's minimal input for
    /// rebuilding the GLOBAL_MEMORY AIR set. Carries page bases ONLY (no cell values), so
    /// private-input bytes never appear in the bundle. Prover- supplied but bus-enforced:
    /// a wrong set imbalances the GlobalMemory bus / mismatches the AIR count, and it is
    /// bound into the global Fiat-Shamir statement (canonicalized on ingest).
    touched_page_bases: Vec<u64>,
}

impl ContinuationProof {
    /// Number of epochs the execution was split into.
    pub fn num_epochs(&self) -> usize {
        self.epochs.len()
    }
}

/// Build an epoch's AIRs identically on the prove and verify sides — the single
/// source of truth for the AIR set, so the two halves can never diverge. The set
/// is `VmAirs` (HALT included iff `is_final`), with REGISTER preprocessed to
/// INIT = `register_init` and FINI = `reg_fini`. Continuation epochs
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
    VmAirs::new(
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
    )
}

/// Prove one epoch (prove half only). Commits its local-to-global table (built from
/// `boundary`) on the epoch-local Memory bus and its REGISTER table with FINI
/// preprocessed to the epoch's final register file. Returns the [`EpochProof`] the
/// standalone verifier later re-checks; does NOT verify here.
#[allow(clippy::too_many_arguments)]
fn prove_epoch(
    elf: &Elf,
    elf_bytes: &[u8],
    start: &EpochStart,
    mut traces: Traces,
    is_final: bool,
    boundary: &[CellBoundary],
    opts: &ProofOptions,
) -> Result<EpochProof, Error> {
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
            &runtime_page_ranges,
            label,
        )
    };

    let l2g_air = l2g_memory_air(opts, label);
    // Build this epoch's L2G table from the cross-epoch boundary so it is identical
    // to the one the global proof commits (the commitment binding compares their
    // roots). It is appended to the proof below, not through `air_trace_pairs`.
    let mut l2g_trace = local_to_global::generate_local_to_global_trace(boundary);

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

    Ok(EpochProof {
        proof,
        public_output,
        table_counts,
        runtime_page_ranges,
        reg_fini,
        l2g_root,
    })
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
            &epoch.runtime_page_ranges,
            label,
        )
    };

    // Start the commit index from the carried x254 (the derived INIT), not a free
    // input — this is what binds the per-epoch commit slice to its global position.
    let commit_start_index = register_init
        .get(register::X254_INDEX)
        .copied()
        .unwrap_or(0) as u64;

    let expected = match compute_expected_commit_bus_balance(
        &refs,
        &epoch.proof,
        &epoch.public_output,
        commit_start_index,
        &mut seed(),
    ) {
        Some(expected) => expected,
        None => return false,
    };

    if !Verifier::multi_verify(&refs, &epoch.proof, &mut seed(), &expected) {
        return false;
    }

    // The claimed L2G root must be the one this proof actually committed (it is what
    // verify_l2g_commitment_binding later ties to the global proof).
    epoch
        .proof
        .proofs
        .last()
        .map(|p| p.lde_trace_main_merkle_root)
        == Some(epoch.l2g_root)
}

/// Build the cross-epoch global memory proof: every epoch's L2G sub-table on the
/// GlobalMemory bus, plus one GLOBAL_MEMORY table per touched page that sends each
/// cell's genesis init and receives its final value. For ELF/runtime pages the genesis
/// is preprocessed (the verifier recomputes it from the ELF); private-input pages are
/// non-preprocessed (committed, bus-enforced genesis — see `global_memory_air` / §3.6).
/// The bus balances iff every `fini` matches the next epoch's `init` and every genesis
/// matches its source (the ELF for ELF/runtime pages).
fn prove_global(
    boundaries: &[Vec<CellBoundary>],
    elf_bytes: &[u8],
    init_page_data: &HashMap<u64, Vec<u8>>,
    page_bases: &[u64],
    num_private_input_pages: usize,
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

    let gm_configs = global_memory_configs_from_init_page_data(
        page_bases,
        init_page_data,
        num_private_input_pages,
        true,
    );

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
    let gm_airs: Vec<_> = gm_configs
        .iter()
        .map(|config| global_memory_air(opts, config))
        .collect();

    let mut pairs: Vec<(AirRef, &mut TraceTable<F, E>, &())> = l2g_airs
        .iter()
        .zip(l2g_traces.iter_mut())
        .map(|(air, t)| (air as AirRef, t, &()))
        .collect();
    for (air, trace) in gm_airs.iter().zip(gm_traces.iter_mut()) {
        pairs.push((air as AirRef, trace, &()));
    }

    Prover::multi_prove(
        pairs,
        &mut global_transcript(
            elf_bytes,
            boundaries.len(),
            num_private_input_pages,
            page_bases,
        ),
        #[cfg(feature = "disk-spill")]
        stark::storage_mode::StorageMode::Ram,
    )
    .map_err(|e| Error::Prover(format!("{e:?}")))
}

fn verify_global(
    num_epochs: usize,
    page_bases: &[u64],
    proof: &MultiProof<F, E, ()>,
    elf: &Elf,
    elf_bytes: &[u8],
    num_private_input_pages: usize,
    opts: &ProofOptions,
) -> bool {
    // One L2G air per epoch, each with its own 1-based `fini_epoch` constant —
    // must match the order/labels the global proof committed in `prove_global`.
    let l2g_airs: Vec<_> = (0..num_epochs)
        .map(|i| l2g_global_air(opts, local_to_global::epoch_label(i as u64)))
        .collect();
    // Rebuild the genesis configs FROM THE ELF (no private bytes) and recompute their
    // commitments: this is the binding for ELF/runtime pages — a prover that claimed
    // different genesis values would commit a different root and fail to verify.
    // Private-input pages (the first `num_private_input_pages` from
    // PRIVATE_INPUT_START_INDEX) are built non-preprocessed, so the verifier never
    // recomputes their genesis from the ELF; the GlobalMemory bus enforces them. A
    // wrong `num_private_input_pages` flips a touched page's preprocessed mode, so the
    // rebuilt AIR no longer matches the committed trace and `multi_verify` rejects.
    let gm_configs = global_memory_configs(page_bases, elf, num_private_input_pages);
    let gm_airs: Vec<_> = gm_configs
        .iter()
        .map(|config| global_memory_air(opts, config))
        .collect();

    let mut refs: Vec<AirRef> = l2g_airs.iter().map(|a| a as AirRef).collect();
    for air in &gm_airs {
        refs.push(air as AirRef);
    }

    Verifier::multi_verify(
        &refs,
        proof,
        &mut global_transcript(elf_bytes, num_epochs, num_private_input_pages, page_bases),
        &FieldElement::zero(),
    )
}

/// Streaming driver over the epochs of one execution: owns the executor and the
/// carried cross-epoch state (`image`, `provenance`, `prev_fini`) and proves one epoch
/// per [`next`](EpochDriver::next) call. It is the single source of truth for the epoch
/// loop, so the bundle path ([`prove_continuation`], which keeps every proof) and the
/// streaming self-verify path ([`prove_and_verify_continuation`], which verifies and
/// drops each proof) cannot diverge in how epochs are produced.
///
/// `init_page_data` is snapshotted from the INITIAL image in [`new`](EpochDriver::new)
/// and never recomputed from the carried (mutated) image, so the global proof's genesis
/// matches the ELF the verifier rebuilds.
struct EpochDriver<'a> {
    elf: Elf,
    elf_bytes: &'a [u8],
    private_inputs: &'a [u8],
    opts: &'a ProofOptions,
    epoch_size: usize,
    executor: Executor,
    /// Cross-epoch memory image, carried forward: epoch i+1's init is epoch i's fini.
    image: PagedMem<u8>,
    provenance: PagedMem<(u64, u64, u64)>,
    /// Genesis page data snapshotted from the initial image (fed to `prove_global`).
    init_page_data: HashMap<u64, Vec<u8>>,
    /// The previous epoch's bound final register file `R_{i+1}` (next epoch's init).
    prev_fini: Option<Vec<u32>>,
    index: u64,
}

impl<'a> EpochDriver<'a> {
    fn new(
        elf_bytes: &'a [u8],
        private_inputs: &'a [u8],
        epoch_size_log2: u32,
        opts: &'a ProofOptions,
    ) -> Result<Self, Error> {
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
        let executor = Executor::new(&elf, private_inputs.to_vec())
            .map_err(|e| Error::Execution(format!("{e}")))?;

        let image = build_initial_image_paged(&elf, private_inputs);
        let init_page_data = build_init_page_data(&image);
        let provenance =
            local_to_global::genesis_provenance(image.iter().map(|(a, v)| (a, v as u64)));

        Ok(Self {
            elf,
            elf_bytes,
            private_inputs,
            opts,
            epoch_size,
            executor,
            image,
            provenance,
            init_page_data,
            prev_fini: None,
            index: 0,
        })
    }

    /// Prove the next epoch, or `Ok(None)` once the execution has halted. Advances the
    /// carried image/provenance/register state. This epoch's `is_final` is decided here
    /// by whether the program halted inside it (`executor.pc() == 0`) and baked into its
    /// AIRs (HALT is present only on the final epoch); the verifier re-derives `is_final`
    /// independently rather than trusting this.
    fn next(&mut self) -> Result<Option<(EpochProof, Vec<CellBoundary>)>, Error> {
        if self.executor.pc() == 0 {
            return Ok(None);
        }
        // The cross-epoch ordering check (IsB20 on `fini_epoch - 1 - init_epoch`) only
        // spans `local_to_global::MAX_EPOCHS` epochs. Beyond that an honest proof is
        // impossible — fail fast instead of building an unprovable trace.
        if self.index >= local_to_global::MAX_EPOCHS {
            return Err(Error::InvalidContinuationEpochSize(format!(
                "execution needs more than {} continuation epochs (the IsB20 cross-epoch \
                 ordering range); use a larger epoch size",
                local_to_global::MAX_EPOCHS
            )));
        }

        let register_init: Vec<u32> = if self.index == 0 {
            register::register_init_from_entry_point(self.elf.entry_point)
        } else {
            // Epoch i+1's init is epoch i's bound fini, reused directly — the
            // cross-epoch register binding.
            self.prev_fini.clone().ok_or_else(|| {
                Error::ContinuationInvariant(
                    "previous epoch final registers are missing after the first epoch".to_string(),
                )
            })?
        };

        // Run one epoch; `logs` is this epoch's chunk only (the executor clears it).
        let logs = match self
            .executor
            .resume_with_limit(self.epoch_size)
            .map_err(|e| Error::Execution(format!("{e}")))?
        {
            Some(logs) => logs.to_vec(),
            // Unreachable today: `resume_with_limit` returns `None` only when `pc == 0`
            // at entry, which the guard above already handles. If a future executor
            // change makes it stop mid-run for another reason, fail loudly at the cause
            // (still no panic) rather than ending the driver cleanly: a clean end here
            // would make `prove_continuation` emit a bundle whose last epoch lacks HALT
            // — never verifiable, with no diagnostic linking the later rejection back
            // to the early stop.
            None => {
                return Err(Error::ContinuationInvariant(
                    "executor stopped mid-run without halting (resume_with_limit \
                     returned None with pc != 0 at epoch entry)"
                        .to_string(),
                ));
            }
        };
        let is_final = self.executor.pc() == 0;

        // Invariant: a non-final epoch ran the full `epoch_size` (a power of two), so
        // its CPU table has no padding rows.
        if !is_final && logs.len() != self.epoch_size {
            return Err(Error::ContinuationInvariant(format!(
                "intermediate epoch ran {} cycles, expected {}",
                logs.len(),
                self.epoch_size
            )));
        }

        let label = local_to_global::epoch_label(self.index);
        let traces = Traces::from_image_and_logs(
            &self.elf,
            &self.image,
            &register_init,
            &logs,
            &MaxRowsConfig::default(),
            self.private_inputs,
            is_final,
            true,
            #[cfg(feature = "disk-spill")]
            stark::storage_mode::StorageMode::Ram,
        )?;
        let boundary = local_to_global::epoch_boundary(
            &mut self.provenance,
            label,
            &traces.touched_memory_cells,
        );

        let start = EpochStart {
            register_init: &register_init,
            label,
        };
        let epoch = prove_epoch(
            &self.elf,
            self.elf_bytes,
            &start,
            traces,
            is_final,
            &boundary,
            self.opts,
        )?;
        self.prev_fini = Some(epoch.reg_fini.clone());

        // Carry the image forward: this epoch's fini is the next epoch's init.
        for cell in &boundary {
            self.image.set(cell.address, (cell.fini.value & 0xFF) as u8);
        }
        self.index += 1;

        // `boundary` is kept prover-local (returned to the caller for the global prove);
        // it is deliberately NOT stored in `EpochProof`, so `CellBoundary`'s cell values
        // (private-input bytes) never reach the shipped bundle. Only the value-free
        // `touched_page_bases` derived from it is shipped.
        Ok(Some((epoch, boundary)))
    }
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
    let mut driver = EpochDriver::new(elf_bytes, private_inputs, epoch_size_log2, opts)?;

    let mut epochs: Vec<EpochProof> = Vec::new();
    // Full per-epoch boundaries, kept prover-local for `prove_global` (L2G traces +
    // final-state). The driver returns each epoch's boundary alongside its proof;
    // it is deliberately NOT stored in `EpochProof`/the bundle — `CellBoundary` holds
    // cell values (private-input bytes for private reads); only the value-free page-base
    // set is shipped (see `touched_page_bases`).
    let mut all_boundaries: Vec<Vec<CellBoundary>> = Vec::new();
    while let Some((epoch, boundary)) = driver.next()? {
        epochs.push(epoch);
        all_boundaries.push(boundary);
    }

    // One global LogUp over all the (kept) local-to-global tables. `all_boundaries` was
    // accumulated locally from the driver (never round-tripped through the bundle).
    let num_private_input_pages = page::private_input_page_count(private_inputs);
    // SINGLE source of truth: the same page-base list drives the committed GLOBAL_MEMORY
    // tables and is shipped in the bundle, so the two can never diverge in set or order.
    let touched_page_bases = touched_page_bases(&all_boundaries);
    let global = prove_global(
        &all_boundaries,
        elf_bytes,
        &driver.init_page_data,
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

/// Verify a [`ContinuationProof`] using ONLY the bundle and the ELF — nothing from
/// the prover's memory. Returns `Ok(Some(public_output))` (the run-wide committed
/// bytes, reconstructed from the per-epoch bound slices) iff every check holds,
/// `Ok(None)` if a well-formed proof fails verification, and `Err` if the bundle is
/// structurally malformed (fails validation before any proof is checked).
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
    // Bound the claimed private-input page count before using it to size/allocate AIRs
    // (mirrors `verify_with_options`). The count is also bound into the global proof's
    // Fiat-Shamir statement (`absorb_continuation_global_statement`), so any wrong value
    // diverges the verifier's challenges and `verify_global`'s `multi_verify` rejects —
    // on top of the committed-AIR-shape mismatch a wrong count causes on a touched page.
    let max_private_input_pages = page::max_private_input_pages();
    if bundle.num_private_input_pages > max_private_input_pages {
        return Err(Error::InvalidTableCounts(format!(
            "num_private_input_pages ({}) exceeds max ({max_private_input_pages})",
            bundle.num_private_input_pages
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
    let mut public_output: Vec<u8> = Vec::new();

    // NOTE: `prove_and_verify_continuation` mirrors this loop and the tail checks below
    // inline over streamed epochs. Any check added or changed here MUST be mirrored
    // there — the differential test only exercises honest proofs, so it cannot catch a
    // rejection check missing from one of the two copies.
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
        public_output.extend_from_slice(&epoch.public_output);
        // Next epoch's init is this epoch's bound fini — the cross-epoch register
        // (and x254) binding. A mismatched fini desyncs the next epoch's AIRs.
        register_init = epoch.reg_fini.clone();
    }

    // Cross-epoch global memory: genesis for ELF/runtime pages is rebuilt FROM THE ELF
    // (no private bytes), so the starting memory cannot be prover-chosen; the bus
    // telescopes fini→init. Private-input pages are committed, non-preprocessed (genesis
    // not bundled/ELF-recomputed), bus-enforced. The verifier needs only the epoch count and the
    // touched page-base set (never cell values); the bundle carries the latter directly.
    // Canonicalize the (untrusted) list so a shuffled-but-same-set list still verifies,
    // while a different set fails via GlobalMemory-bus imbalance / AIR-count mismatch.
    let page_bases = canonical_page_bases(&bundle.touched_page_bases);
    // Every honest base is produced by `page::page_base_for_address`, so it is page-aligned; a
    // non-aligned base is only reachable via a hand-crafted bundle. Left unchecked, such a base
    // still falls in the private-input range (`page::is_private_input_page`), so it would be
    // built NON-preprocessed with a prover-controlled genesis. The GlobalMemory bus already
    // prevents forging any real cell (no MEMW access exists at a non-aligned fake address, so no
    // L2G row consumes its genesis token), but a self-cancelling junk page could otherwise ride
    // along in an accepted proof. Reject here so the verifier's page set is exactly the aligned
    // set the prover could honestly derive. Like the count bound above, this is structural
    // validation of an untrusted bundle field, so it is an `Err` (malformed bundle), not
    // `Ok(None)` (well-formed proof that failed verification).
    if page_bases
        .iter()
        .any(|&b| b != page::page_base_for_address(b))
    {
        return Err(Error::MalformedContinuationBundle(
            "touched_page_bases contains a non-page-aligned entry".to_string(),
        ));
    }
    if !verify_global(
        n,
        &page_bases,
        &bundle.global,
        &elf,
        elf_bytes,
        bundle.num_private_input_pages,
        opts,
    ) {
        return Ok(None);
    }

    // Each epoch's committed L2G table is the same one the global proof used.
    if !verify_l2g_commitment_binding(&epoch_roots, &bundle.global) {
        return Ok(None);
    }

    Ok(Some(public_output))
}

/// Prove and verify a full continuation in one streaming pass: each epoch is proved,
/// verified inline, and its proof dropped — only `boundary`, `l2g_root`, and
/// `public_output` are retained — then the one cross-epoch global proof is built and
/// verified. This bounds the retained-proof memory to O(1) epochs instead of collecting
/// every epoch proof up front (as [`prove_continuation`] + [`verify_continuation`] do).
///
/// The verification is a faithful mirror of [`verify_continuation`]: `register_init` is
/// chained from each verified epoch's `reg_fini`, and `label`/`is_final` are derived
/// positionally here (one-epoch lookahead for `is_final`), never taken from the prover.
/// It is NOT a substitute for verifying an untrusted serialized `ContinuationProof` —
/// [`verify_continuation`] remains that. Returns `Ok(Some(public_output))` iff the
/// continuation proves and verifies.
pub fn prove_and_verify_continuation(
    elf_bytes: &[u8],
    private_inputs: &[u8],
    epoch_size_log2: u32,
    opts: &ProofOptions,
) -> Result<Option<Vec<u8>>, Error> {
    // Match `verify_continuation`'s up-front bound so this function and the standalone
    // verifier surface the same error variant for an oversized input, and fail fast
    // before the executor copies it. (The two-phase composition `prove_continuation` +
    // `verify_continuation` instead fails inside `Executor::new` with
    // `Error::Execution` — `prove_continuation` runs first, so the verifier's bound is
    // never reached on that path.)
    if private_inputs.len() as u64 > MAX_PRIVATE_INPUT_SIZE {
        return Err(Error::InvalidTableCounts(format!(
            "private input size ({}) exceeds max ({MAX_PRIVATE_INPUT_SIZE})",
            private_inputs.len()
        )));
    }

    let mut driver = EpochDriver::new(elf_bytes, private_inputs, epoch_size_log2, opts)?;

    // Verify-side state, rebuilt independently of the prover: `register_init` chains
    // from each verified epoch's bound `reg_fini` (epoch 0 from the ELF entry point).
    // NOTE: this loop and the tail checks below are an inline mirror of
    // `verify_continuation`'s — any check added or changed there MUST be mirrored here
    // (see the matching note on its epoch loop).
    let mut register_init = register::register_init_from_entry_point(driver.elf.entry_point);
    let mut all_boundaries: Vec<Vec<CellBoundary>> = Vec::new();
    let mut epoch_roots: Vec<Commitment> = Vec::new();
    let mut public_output: Vec<u8> = Vec::new();
    let mut index: u64 = 0;

    // One-epoch lookahead: `is_final` is derived from whether another epoch actually
    // follows (positional), not from a flag the prover stored, so a driver enumeration
    // bug (an off-by-one or reorder) cannot be silently self-consistent between the
    // prove and verify halves. Both halves ultimately observe the same in-process
    // executor, so this buys mirror fidelity with `verify_continuation`'s positional
    // derivation, not independence from the prover.
    let mut pending = driver.next()?;
    while let Some((epoch, boundary)) = pending {
        let next = driver.next()?;
        let is_final = next.is_none();
        let label = local_to_global::epoch_label(index);

        // Ordering is load-bearing: verify_epoch must pass BEFORE `l2g_root` is trusted
        // and the proof dropped. The global L2G air is constraint-free, so the epoch
        // proof (checked here) is the sole enforcer of the L2G column constraints, and
        // `l2g_root` is only bound to a verified trace once verify_epoch returns true.
        if !verify_epoch(
            &driver.elf,
            elf_bytes,
            &epoch,
            &register_init,
            is_final,
            label,
            opts,
        ) {
            return Ok(None);
        }

        // Keep only what the global step + final checks need; the MultiProof drops here.
        // `epoch` is owned, so `reg_fini` moves out (no clone) into the next epoch's
        // derived init — the cross-epoch register chain, checked independently here.
        // Unlike `verify_continuation` there is deliberately no `reg_fini` length guard:
        // there it is deserialized and untrusted, while here it comes from in-process
        // `prove_epoch` (`fini_from_trace` always yields exactly NUM_REGISTER_ADDRESSES
        // entries). A wrong length would be an internal invariant bug and should fail
        // loudly (the debug_assert in `compute_precomputed_commitment_with_fini`)
        // rather than be reported as a rejected proof.
        let EpochProof {
            l2g_root,
            public_output: out,
            reg_fini,
            ..
        } = epoch;
        register_init = reg_fini;
        public_output.extend_from_slice(&out);
        epoch_roots.push(l2g_root);
        all_boundaries.push(boundary);
        index += 1;

        pending = next;
    }

    // Mirror verify_continuation's n == 0 early-return (reachable via an `e_entry == 0`
    // ELF, which yields zero epochs): without it the empty global prove/verify would
    // trivially pass and accept a run the standalone verifier rejects.
    if all_boundaries.is_empty() {
        return Ok(None);
    }

    // Kept prover-local, exactly as the bundle path does: derive the value-free page-base
    // set and private-page count, then prove/verify the global step against them (never
    // shipping `all_boundaries`, which carries cell values).
    let num_private_input_pages = page::private_input_page_count(private_inputs);
    let touched_page_bases = touched_page_bases(&all_boundaries);
    let global = prove_global(
        &all_boundaries,
        elf_bytes,
        &driver.init_page_data,
        &touched_page_bases,
        num_private_input_pages,
        opts,
    )?;
    if !verify_global(
        all_boundaries.len(),
        &touched_page_bases,
        &global,
        &driver.elf,
        elf_bytes,
        num_private_input_pages,
        opts,
    ) {
        return Ok(None);
    }
    if !verify_l2g_commitment_binding(&epoch_roots, &global) {
        return Ok(None);
    }

    Ok(Some(public_output))
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

    // Anti-drift guard: the streaming self-verify path (`prove_and_verify_continuation`)
    // must accept/reject and aggregate output IDENTICALLY to the authoritative two-phase
    // path (`prove_continuation` + the untouched `verify_continuation`). The streaming
    // path derives `is_final`/`label` positionally and chains `register_init` itself, so
    // it must agree with the standalone verifier on both the single-epoch
    // (immediately-final) and multi-epoch (enumeration-boundary) shapes.
    #[test]
    fn test_streaming_matches_two_phase() {
        let _ = env_logger::builder().is_test(true).try_init();

        // `expected` pins an independent known-good output where we know it, so the
        // test is not purely differential over one shared prover (a shared-prover or
        // aggregation bug affecting both halves identically would otherwise pass).
        let commit_output: &[u8] = &[0xAA, 0xBB, 0xCC, 0xDD];
        for (name, log2, want_multi, expected) in [
            ("test_commit_split", 6u32, false, Some(commit_output)), // one final epoch
            ("test_commit_split", 4, true, Some(commit_output)),     // splits across epochs
            ("all_loadstore_32", 3, true, None),                     // several intermediate epochs
        ] {
            let elf_bytes = asm_elf_bytes(name);

            let bundle =
                prove_continuation(&elf_bytes, &[], log2, &ProofOptions::default_test_options())
                    .unwrap();
            if want_multi {
                assert!(
                    bundle.num_epochs() > 1,
                    "{name} @ log2={log2} must split into multiple epochs"
                );
            } else {
                assert_eq!(
                    bundle.num_epochs(),
                    1,
                    "{name} @ log2={log2} must be a single final epoch"
                );
            }
            let two_phase =
                verify_continuation(&elf_bytes, &bundle, &ProofOptions::default_test_options())
                    .unwrap();
            assert!(two_phase.is_some(), "{name} @ log2={log2} must verify");

            let streaming = prove_and_verify_continuation(
                &elf_bytes,
                &[],
                log2,
                &ProofOptions::default_test_options(),
            )
            .unwrap();

            assert_eq!(
                streaming, two_phase,
                "{name} @ log2={log2}: streaming self-verify diverged from prove+verify_continuation"
            );
            if let Some(expected) = expected {
                assert_eq!(
                    streaming.as_deref(),
                    Some(expected),
                    "{name} @ log2={log2}: streaming output must match the known-good value"
                );
            }
        }
    }

    // An ELF whose entry point is 0 yields zero epochs (the executor starts halted).
    // The streaming path must mirror `verify_continuation`'s `n == 0` early-return and
    // reject it with `Ok(None)` rather than trivially "verifying" an empty run.
    #[test]
    fn test_streaming_zero_epoch_returns_none() {
        let _ = env_logger::builder().is_test(true).try_init();
        let mut elf_bytes = asm_elf_bytes("all_loadstore_32");
        // ELF64 `e_entry` is an 8-byte little-endian field at offset 24; zero it so the
        // executor starts with pc == 0 and the driver produces no epochs.
        for b in &mut elf_bytes[24..32] {
            *b = 0;
        }
        assert_eq!(Elf::load(&elf_bytes).unwrap().entry_point, 0);
        assert!(
            prove_and_verify_continuation(
                &elf_bytes,
                &[],
                4,
                &ProofOptions::default_test_options()
            )
            .unwrap()
            .is_none(),
            "a zero-entry ELF (zero epochs) must be rejected, not accepted as empty"
        );
    }

    // Exactly-2-epoch discriminator for the `is_final` lookahead: the minimal case that
    // exercises the false→true transition exactly once. `epoch_size_log2` is derived from
    // the measured cycle count as the largest power of two strictly below `total`, so
    // `2^L < total <= 2^(L+1)` ⟹ exactly 2 epochs — pinning `num_epochs() == 2` (not just
    // `> 1`) and confirming the streaming path still equals the two-phase path. Guards
    // against a lookahead/label off-by-one at the tightest enumeration boundary.
    #[test]
    fn test_streaming_exactly_two_epochs() {
        let _ = env_logger::builder().is_test(true).try_init();
        let elf_bytes = asm_elf_bytes("all_loadstore_32");
        let total = Executor::new(&Elf::load(&elf_bytes).unwrap(), vec![])
            .unwrap()
            .run()
            .unwrap()
            .logs
            .len();
        // Guard before the arithmetic: `total - 1` underflows and `ilog2(0)` panics
        // for total <= 1, so check the precondition first rather than after.
        assert!(
            total >= 2,
            "program ({total} cycles) too short to derive a valid 2-epoch split"
        );
        let epoch_size_log2 = (total as u64 - 1).ilog2();
        assert!(
            epoch_size_log2 >= 2,
            "program ({total} cycles) too short to derive a valid 2-epoch split"
        );

        let bundle = prove_continuation(
            &elf_bytes,
            &[],
            epoch_size_log2,
            &ProofOptions::default_test_options(),
        )
        .unwrap();
        assert_eq!(
            bundle.num_epochs(),
            2,
            "epoch_size_log2={epoch_size_log2} over {total} cycles must yield exactly 2 epochs"
        );
        let two_phase =
            verify_continuation(&elf_bytes, &bundle, &ProofOptions::default_test_options())
                .unwrap();
        assert!(two_phase.is_some());

        let streaming = prove_and_verify_continuation(
            &elf_bytes,
            &[],
            epoch_size_log2,
            &ProofOptions::default_test_options(),
        )
        .unwrap();
        assert_eq!(
            streaming, two_phase,
            "streaming self-verify diverged from prove+verify_continuation at exactly 2 epochs"
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

    // The raw private input must not be bundled under continuations. The bundle carries no
    // raw private bytes (only `num_private_input_pages`), yet a multi-epoch continuation of
    // a program that reads private input verifies from the bundle + ELF ALONE and
    // reconstructs the committed output. Regression for the genesis leak: the global
    // proof's private-input genesis is a committed, bus-enforced column, not a
    // preprocessed value the verifier would have to recompute from the raw bytes.
    #[test]
    fn test_continuation_private_input_verifies_without_bytes() {
        let _ = env_logger::builder().is_test(true).try_init();
        let elf_bytes = asm_elf_bytes("test_private_input_xpage");
        let input: Vec<u8> = (0u8..16).collect();
        let expected = input[4..12].to_vec();

        // Smallest epochs (2^2 = 4 cycles) so the short program splits across epochs.
        let bundle =
            prove_continuation(&elf_bytes, &input, 2, &ProofOptions::default_test_options())
                .unwrap();
        assert!(
            bundle.num_epochs() > 1,
            "4-cycle epochs must split the run into multiple epochs"
        );
        assert!(
            bundle.num_private_input_pages > 0,
            "a program that reads private input must have a private-input page in the global proof"
        );

        // The serialized bundle must carry no raw private bytes: it survives a bincode
        // round-trip and still verifies using ONLY the bundle + ELF (no private input
        // is passed to `verify_continuation`).
        let bytes = bincode::serialize(&bundle).unwrap();
        let restored: ContinuationProof = bincode::deserialize(&bytes).unwrap();
        let out = verify_continuation(&elf_bytes, &restored, &ProofOptions::default_test_options())
            .unwrap();
        assert_eq!(
            out.as_deref(),
            Some(&expected[..]),
            "continuation with private input must verify from the bundle + ELF alone"
        );
    }

    // Negative: `num_private_input_pages` is pinned by the committed AIR shape for TOUCHED
    // pages. Deflating it to 0 for a program that reads private input makes the verifier
    // build that touched page preprocessed (ELF-recomputed → zero-init commitment) while
    // the prover committed it non-preprocessed, so the rebuilt AIR no longer matches the
    // committed trace and verification rejects. This is the replacement for the removed
    // tampered-genesis test: it guards the security claim that a wrong count flipping a
    // touched page's preprocessed mode cannot be accepted.
    #[test]
    fn test_split_verify_rejects_deflated_num_private_input_pages() {
        let _ = env_logger::builder().is_test(true).try_init();
        let elf_bytes = asm_elf_bytes("test_private_input_xpage");
        let input: Vec<u8> = (0u8..16).collect();
        let mut bundle =
            prove_continuation(&elf_bytes, &input, 2, &ProofOptions::default_test_options())
                .unwrap();
        assert!(
            bundle.num_private_input_pages > 0,
            "baseline must have a touched private-input page"
        );
        assert!(
            verify_continuation(&elf_bytes, &bundle, &ProofOptions::default_test_options())
                .unwrap()
                .is_some(),
            "baseline must verify before tampering"
        );

        bundle.num_private_input_pages = 0;
        assert!(
            verify_continuation(&elf_bytes, &bundle, &ProofOptions::default_test_options())
                .unwrap()
                .is_none(),
            "deflating the count flips a touched page's preprocessed mode → must reject"
        );
    }

    // Negative: inflating `num_private_input_pages` to an in-range but wrong value must also
    // reject. Inflation only enlarges the private-page *range* over untouched pages (no
    // touched page's preprocessed mode flips, so the committed-AIR-shape check alone would
    // NOT catch it) — the count is absorbed into the global proof's Fiat-Shamir statement, so
    // the verifier's challenges diverge from the prover's and `verify_global` rejects. Guards
    // the FS-binding of the count (complements the deflation test's AIR-shape-mismatch path).
    #[test]
    fn test_split_verify_rejects_inflated_num_private_input_pages() {
        let _ = env_logger::builder().is_test(true).try_init();
        let elf_bytes = asm_elf_bytes("test_private_input_xpage");
        let input: Vec<u8> = (0u8..16).collect();
        let mut bundle =
            prove_continuation(&elf_bytes, &input, 2, &ProofOptions::default_test_options())
                .unwrap();
        assert_eq!(
            bundle.num_private_input_pages, 1,
            "16 bytes of private input fits in one page"
        );
        assert!(
            verify_continuation(&elf_bytes, &bundle, &ProofOptions::default_test_options())
                .unwrap()
                .is_some(),
            "baseline must verify before tampering"
        );

        // In-range (well under the max bound) but one more than the true count.
        bundle.num_private_input_pages = 2;
        assert!(
            verify_continuation(&elf_bytes, &bundle, &ProofOptions::default_test_options())
                .unwrap()
                .is_none(),
            "an inflated count diverges the global Fiat-Shamir statement → must reject"
        );
    }

    // Private-input page classification is count-based (the first `n` pages from
    // PRIVATE_INPUT_START_INDEX), matching the monolithic verifier — the classification
    // depends ONLY on the count, never on the raw private-input byte range. So with no
    // private input, no page in the region is classified private (checked below), keeping
    // the continuation from ever marking more pages private than the monolithic path would.
    // (ELF data cannot be placed *inside* the reserved region: `Elf::load` rejects any
    // segment overlapping it, so a private page never holds ELF-bound data.)
    #[test]
    fn test_private_input_page_classification_is_count_based() {
        use executor::vm::memory::{MAX_PRIVATE_INPUT_SIZE, PRIVATE_INPUT_START_INDEX};
        let page_size = page::DEFAULT_PAGE_SIZE as u64;
        let start = PRIVATE_INPUT_START_INDEX;

        // With no private input, NO page in the private-input region is private — not the
        // first page, not the last. Classification is by count alone, not the region span.
        let region_pages = MAX_PRIVATE_INPUT_SIZE / page_size;
        let last_region_page = start + (region_pages - 1) * page_size;
        assert!(!page::is_private_input_page(start, 0));
        assert!(!page::is_private_input_page(last_region_page, 0));

        // Count n → exactly the first n pages from start.
        assert!(page::is_private_input_page(start, 1));
        assert!(!page::is_private_input_page(start + page_size, 1));
        assert!(page::is_private_input_page(start + page_size, 2));
        // Pages below the region are never private.
        assert!(!page::is_private_input_page(start - page_size, 10));

        // private_input_page_count: wire format is [len:4][data], region is page-aligned.
        assert_eq!(page::private_input_page_count(&[]), 0);
        assert_eq!(page::private_input_page_count(&[0u8; 16]), 1);
        // 4-byte prefix + (page_size - 4) data exactly fills one page.
        assert_eq!(
            page::private_input_page_count(&vec![0u8; page::DEFAULT_PAGE_SIZE - 4]),
            1
        );
        // One more byte spills into a second page.
        assert_eq!(
            page::private_input_page_count(&vec![0u8; page::DEFAULT_PAGE_SIZE - 3]),
            2
        );
    }

    // `private_input_page_bases` must enumerate exactly the aligned bases that
    // `is_private_input_page` classifies private, in ascending, page_size-spaced order.
    #[test]
    fn test_private_input_page_bases_enumeration() {
        use executor::vm::memory::PRIVATE_INPUT_START_INDEX;
        let page_size = page::DEFAULT_PAGE_SIZE as u64;
        let start = PRIVATE_INPUT_START_INDEX;

        // Count 0 yields nothing.
        assert_eq!(page::private_input_page_bases(0).count(), 0);

        // Ascending, exact page_size spacing from the region start.
        let bases: Vec<u64> = page::private_input_page_bases(3).collect();
        assert_eq!(bases, vec![start, start + page_size, start + 2 * page_size]);

        // The enumeration and the predicate agree: every yielded base classifies
        // private for that count, and the first base past them does not.
        for n in 0..4usize {
            for base in page::private_input_page_bases(n) {
                assert!(page::is_private_input_page(base, n));
            }
            assert!(!page::is_private_input_page(
                start + n as u64 * page_size,
                n
            ));
        }
    }

    // The deserialized-count bound is the tight honest max: exactly the pages a MAX-size
    // input occupies, with no slack. Pin the value and the tightness (checked via the byte
    // span so we don't allocate a 64 MiB test input).
    #[test]
    fn test_max_private_input_pages_is_tight() {
        use executor::vm::memory::{MAX_PRIVATE_INPUT_SIZE, PRIVATE_INPUT_LENGTH_PREFIX_BYTES};
        let page_size = page::DEFAULT_PAGE_SIZE;
        let max = page::max_private_input_pages();

        // (64 MiB + 4-byte prefix) / 256 KiB page = 257 pages (256 full data pages plus
        // the one page the length prefix spills into). Pinned so a size/page change is caught.
        assert_eq!(max, 257);

        // No slack: an honest MAX-size input needs the whole last page (the bound is not
        // padded), and never overflows into an extra one.
        let honest_bytes = MAX_PRIVATE_INPUT_SIZE as usize + PRIVATE_INPUT_LENGTH_PREFIX_BYTES;
        assert!((max - 1) * page_size < honest_bytes);
        assert!(honest_bytes <= max * page_size);
    }

    // The verifier builds private-page configs with `include_private_genesis=false`, which
    // must yield an explicitly empty genesis (never the looked-up bytes) so no verifier path
    // can start depending on private data; the prover's `true` still loads the committed bytes.
    #[test]
    fn test_global_memory_configs_private_genesis_inclusion() {
        use executor::vm::memory::PRIVATE_INPUT_START_INDEX;
        let private_base = PRIVATE_INPUT_START_INDEX;
        let genesis = vec![1u8, 2, 3, 4];
        let mut init_page_data = HashMap::new();
        init_page_data.insert(private_base, genesis.clone());

        // Verifier side: empty genesis even though bytes are present in the map.
        let verifier =
            global_memory_configs_from_init_page_data(&[private_base], &init_page_data, 1, false);
        assert_eq!(verifier.len(), 1);
        assert!(verifier[0].is_private_input);
        assert_eq!(verifier[0].init_values, Some(Vec::new()));

        // Prover side: the same call loads the genesis bytes into the committed config.
        let prover =
            global_memory_configs_from_init_page_data(&[private_base], &init_page_data, 1, true);
        assert!(prover[0].is_private_input);
        assert_eq!(prover[0].init_values, Some(genesis));
    }

    // Negative: `num_private_input_pages` is deserialized/untrusted, so reject a bundle
    // whose count exceeds the max before using it to size/build the global AIRs.
    #[test]
    fn test_split_verify_rejects_oversized_num_private_input_pages() {
        let _ = env_logger::builder().is_test(true).try_init();
        let elf_bytes = asm_elf_bytes("all_loadstore_32");
        let mut bundle =
            prove_continuation(&elf_bytes, &[], 3, &ProofOptions::default_test_options()).unwrap();
        bundle.num_private_input_pages = page::max_private_input_pages() + 1;
        assert!(matches!(
            verify_continuation(&elf_bytes, &bundle, &ProofOptions::default_test_options()),
            Err(Error::InvalidTableCounts(_))
        ));
    }

    // Negative: the streaming path's up-front private-input bound must reject an
    // oversized input with the same variant as `verify_continuation` (its documented
    // mirror). Pins the guard itself: without it the executor rejects the same input
    // later, but with `Error::Execution`. Fires before any proving, so the test only
    // costs the one oversized allocation.
    #[test]
    fn test_streaming_rejects_oversized_private_inputs() {
        let _ = env_logger::builder().is_test(true).try_init();
        let elf_bytes = asm_elf_bytes("all_loadstore_32");
        let oversized = vec![0; MAX_PRIVATE_INPUT_SIZE as usize + 1];
        assert!(matches!(
            prove_and_verify_continuation(
                &elf_bytes,
                &oversized,
                4,
                &ProofOptions::default_test_options()
            ),
            Err(Error::InvalidTableCounts(_))
        ));
    }

    // Privacy regression for the touched-cell value leak. Pre-fix, `EpochProof.boundary`
    // serialized each touched cell's `init.value`/`fini.value` as a u64, so a private
    // byte `b` appeared in the bundle as the 8-byte window `[b,0,0,0,0,0,0,0]`. The fix
    // drops `boundary` from the bundle entirely (only the value-free `touched_page_bases`
    // ships), so those windows must be gone. We mark distinctive TOUCHED byte values
    // (0xC7..) — their u64-LE encodings are astronomically unlikely to occur as any honest
    // field/count/root byte-run — and assert none appear in the serialized bundle. (The
    // committed `public_output` serializes bytes RAW, not as u64s, so it cannot produce
    // these windows even for the committed markers.)
    #[test]
    fn test_bundle_carries_no_touched_cell_values() {
        let _ = env_logger::builder().is_test(true).try_init();
        let elf_bytes = asm_elf_bytes("test_private_input_xpage");
        let mut input: Vec<u8> = (0u8..16).collect();
        let markers = [0xC7u8, 0xC8, 0xC9];
        input[4] = markers[0];
        input[5] = markers[1];
        input[6] = markers[2];

        let bundle =
            prove_continuation(&elf_bytes, &input, 2, &ProofOptions::default_test_options())
                .unwrap();
        let bytes = bincode::serialize(&bundle).unwrap();
        for m in markers {
            let needle = (m as u64).to_le_bytes(); // [m,0,0,0,0,0,0,0]
            assert!(
                !bytes.windows(8).any(|w| w == needle),
                "byte 0x{m:02X} appears as a u64 in the bundle — a touched-cell value leaked"
            );
        }
        // Sanity: still verifies from bundle + ELF alone.
        assert!(
            verify_continuation(&elf_bytes, &bundle, &ProofOptions::default_test_options())
                .unwrap()
                .is_some()
        );
    }

    // Multi-page private input: the program reads private input across TWO pages
    // (page 0 for the length, page 1 for the committed bytes), so the run touches two
    // private pages → `num_private_input_pages >= 2` and two NON-preprocessed
    // GLOBAL_MEMORY tables in the global proof. Verifies from bundle + ELF alone and the
    // output equals the page-1 bytes. Exercises the count-based classification and the
    // committed private genesis across more than one page.
    #[test]
    fn test_continuation_multipage_private_input() {
        use executor::vm::memory::PRIVATE_INPUT_START_INDEX;
        let _ = env_logger::builder().is_test(true).try_init();
        let elf_bytes = asm_elf_bytes("test_private_input_multipage");

        // Page 1 starts at memory address START + page_size = 0xFF040000, which is data
        // index `page_size - 4` (the 4-byte length prefix sits at START). The program
        // commits the 8 bytes there, so the input must extend through that.
        let page_size = page::DEFAULT_PAGE_SIZE;
        let commit_off = page_size - 4;
        let mut input = vec![0u8; commit_off + 8];
        let expected: [u8; 8] = [0xA1, 0xB2, 0xC3, 0xD4, 0xE5, 0xF6, 0x07, 0x18];
        input[commit_off..commit_off + 8].copy_from_slice(&expected);

        let bundle =
            prove_continuation(&elf_bytes, &input, 4, &ProofOptions::default_test_options())
                .unwrap();
        assert!(
            bundle.num_private_input_pages >= 2,
            "input spanning two pages must give >=2 private pages"
        );
        let start = PRIVATE_INPUT_START_INDEX;
        let ps = page_size as u64;
        assert!(
            bundle.touched_page_bases.contains(&start)
                && bundle.touched_page_bases.contains(&(start + ps)),
            "both private page 0 and page 1 must be touched (two GLOBAL_MEMORY tables)"
        );

        let out = verify_continuation(&elf_bytes, &bundle, &ProofOptions::default_test_options())
            .unwrap();
        assert_eq!(
            out.as_deref(),
            Some(&expected[..]),
            "committed output must be the 8 bytes read from private page 1"
        );
    }

    // The verifier canonicalizes (sorts/dedups) the shipped `touched_page_bases`, so a
    // list that is reordered AND has duplicates — but describes the same set — still
    // verifies. (Page-count-independent: duplicating then reversing exercises both dedup
    // and reordering even when the program touches a single page.)
    #[test]
    fn test_split_verify_tolerates_reordered_touched_page_bases() {
        let _ = env_logger::builder().is_test(true).try_init();
        let elf_bytes = asm_elf_bytes("all_loadstore_32");
        let mut bundle =
            prove_continuation(&elf_bytes, &[], 3, &ProofOptions::default_test_options()).unwrap();
        assert!(
            !bundle.touched_page_bases.is_empty(),
            "baseline must have touched pages"
        );
        assert!(
            verify_continuation(&elf_bytes, &bundle, &ProofOptions::default_test_options())
                .unwrap()
                .is_some(),
            "baseline must verify"
        );
        // Same set, but duplicated and reversed → canonicalization must recover it.
        let mut scrambled = bundle.touched_page_bases.clone();
        scrambled.extend(bundle.touched_page_bases.clone());
        scrambled.reverse();
        bundle.touched_page_bases = scrambled;
        assert!(
            verify_continuation(&elf_bytes, &bundle, &ProofOptions::default_test_options())
                .unwrap()
                .is_some(),
            "a reordered/duplicated same-set page-base list must still verify (canonicalized)"
        );
    }

    // Negative: dropping a genuinely-touched page base removes its GLOBAL_MEMORY table on
    // the verify side, so that page's L2G fini token has no receiver → GlobalMemory bus
    // imbalance (and the global Fiat-Shamir statement diverges) → reject.
    #[test]
    fn test_split_verify_rejects_dropped_touched_page_base() {
        let _ = env_logger::builder().is_test(true).try_init();
        let elf_bytes = asm_elf_bytes("all_loadstore_32");
        let mut bundle =
            prove_continuation(&elf_bytes, &[], 3, &ProofOptions::default_test_options()).unwrap();
        assert!(
            !bundle.touched_page_bases.is_empty(),
            "baseline must have touched pages"
        );
        assert!(
            verify_continuation(&elf_bytes, &bundle, &ProofOptions::default_test_options())
                .unwrap()
                .is_some(),
            "baseline must verify before tampering"
        );
        bundle.touched_page_bases.pop();
        assert!(
            verify_continuation(&elf_bytes, &bundle, &ProofOptions::default_test_options())
                .unwrap()
                .is_none(),
            "a missing touched page base must be rejected"
        );
    }

    // Negative: a non-page-aligned base is only reachable via a hand-crafted bundle (honest
    // bases come from `page_base_for_address`). The verifier rejects it up front so a base in
    // the private-input range can't be built NON-preprocessed with a prover-controlled genesis
    // and ride along as a self-cancelling junk page. Page-count-independent: perturbing any one
    // base by +1 makes it non-aligned.
    #[test]
    fn test_split_verify_rejects_non_page_aligned_touched_page_base() {
        let _ = env_logger::builder().is_test(true).try_init();
        let elf_bytes = asm_elf_bytes("all_loadstore_32");
        let mut bundle =
            prove_continuation(&elf_bytes, &[], 3, &ProofOptions::default_test_options()).unwrap();
        assert!(
            !bundle.touched_page_bases.is_empty(),
            "baseline must have touched pages"
        );
        assert!(
            verify_continuation(&elf_bytes, &bundle, &ProofOptions::default_test_options())
                .unwrap()
                .is_some(),
            "baseline must verify before tampering"
        );
        bundle.touched_page_bases[0] += 1;
        assert!(
            matches!(
                verify_continuation(&elf_bytes, &bundle, &ProofOptions::default_test_options()),
                Err(Error::MalformedContinuationBundle(_))
            ),
            "a non-page-aligned touched page base is a malformed bundle → must be an Err"
        );
    }

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
}
