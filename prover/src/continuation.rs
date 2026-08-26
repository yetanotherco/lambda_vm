//! First production implementation of continuations (Approach 2).
//!
//! Splits an execution into fixed-size epochs, proves each epoch independently
//! (its memory is initialized/finalized by the per-epoch local-to-global table),
//! and proves one cross-epoch "global memory" LogUp that links every epoch's
//! `fini` to the next epoch's `init` (so `fini(epoch i) == init(epoch i+1)`).
//!
//! The global proof's genesis anchor is bound to the ELF: for ELF/runtime pages the
//! verifier recomputes the per-page preprocessed init commitment from the ELF in
//! `verify_global` by default, so the starting memory cannot be prover-supplied.
//! `verify_continuation_with_roots` lets a caller supply these roots verbatim
//! instead, deferring binding to the caller's downstream recompute-and-compare
//! (like the monolithic prover's supplied-roots path). Private-input pages are the
//! one exception — their genesis is committed (non-preprocessed), exactly as the
//! monolithic prover does, with correctness enforced by the GlobalMemory bus rather
//! than ELF recomputation, so the raw private input is neither carried in the proof
//! bundle nor reconstructed by the verifier.
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
//! and ELF alone (`prove_and_verify_continuation` is a thin wrapper over both).

use std::collections::HashMap;
use std::sync::Arc;

use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use executor::elf::Elf;
use executor::vm::execution::Executor;
use math::field::element::FieldElement;
use stark::config::Commitment;
use stark::constraints::builder::{ConstraintBuilder, ConstraintSet, EmptyConstraints};
use stark::lookup::{AirWithBuses, AuxiliaryTraceBuildData, NullBoundaryConstraintBuilder};
use stark::proof::options::ProofOptions;
use stark::proof::stark::MultiProof;
use stark::proof::view::MultiProofView;
use stark::prover::{IsStarkProver, Prover};
use stark::trace::TraceTable;
use stark::traits::AIR;
use stark::verifier::{IsStarkVerifier, Verifier};

use crate::statement::{StatementKind, absorb_continuation_global_statement, absorb_statement};
use crate::tables::local_to_global::{self, CellBoundary};
use crate::tables::page::{self, PageConfig};
use crate::tables::register;
use crate::tables::trace_builder::{
    DecodeArtifacts, Traces, build_init_page_data, build_initial_image_paged,
};
use crate::tables::types::{GoldilocksExtension, GoldilocksField};
use crate::tables::{MaxRowsConfig, global_memory};
use crate::{
    Error, FIXED_TABLE_COUNT, RuntimePageRange, TableCounts, VmAirs,
    compute_expected_commit_bus_balance_view, verify_l2g_commitment_binding_view,
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
    fri_final_poly_log_degree: u8,
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
        fri_final_poly_log_degree,
    );
    transcript
}

/// Fresh transcript seeded with the global proof's statement (ELF + epoch count).
/// `prove_global` and `verify_global` both seed via this so their challenges match.
fn global_transcript(
    elf_bytes: &[u8],
    num_epochs: usize,
    num_private_input_pages: usize,
    fri_final_poly_log_degree: u8,
    touched_page_bases: &[u64],
) -> DefaultTranscript<E> {
    let mut transcript = DefaultTranscript::<E>::new(&[]);
    absorb_continuation_global_statement(
        &mut transcript,
        elf_bytes,
        num_epochs,
        num_private_input_pages,
        fri_final_poly_log_degree,
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
#[derive(Clone, Copy)]
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
/// and `verify_l2g_commitment_binding_view` ties this global L2G sub-table to the *same*
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
/// Private-input pages preprocess OFFSET **only** (mirrors the monolithic PAGE in
/// `VmAirs::new`): INIT is a committed main-trace column the verifier never recomputes
/// from the ELF, so the raw private input is neither bundled nor reconstructed by the
/// verifier. (Not a ZK/hiding claim — the committed column is still opened at STARK
/// query positions.) OFFSET, by contrast, is preprocessed like everywhere else: it is
/// program- and input-independent, and it is the row's address, so the GlobalMemory bus
/// alone cannot police it. Leaving it free was a soundness hole — the genesis token
/// could name any address in the page's high-limb space.
/// `preprocessed`, when `Some`, is used directly instead of recomputing the
/// genesis commitment from `config.init_values` — the recursion guest's
/// supplied roots skip the in-VM FFT + Merkle build (see `verify_global`).
/// `None` recomputes from `config` as before.
fn global_memory_air(
    opts: &ProofOptions,
    config: &PageConfig,
    preprocessed: Option<Commitment>,
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
        // OFFSET only — see the matching branch in `VmAirs::new`. INIT stays a
        // main-trace column (it is the private input); OFFSET must be committed or
        // `address_lo = page_base_lo + OFFSET` is prover-chosen and the genesis
        // token can name an arbitrary address. GLOBAL_MEMORY's OFFSET column is
        // identical to PAGE's, so the same commitment serves both.
        return air.with_preprocessed(
            page::private_page_preprocessed_commitment(opts),
            page::NUM_PREPROCESSED_COLS_PRIVATE,
        );
    }
    let commitment = preprocessed.unwrap_or_else(|| {
        if config.init_values.is_some() {
            page::compute_precomputed_commitment(config, opts)
        } else {
            page::zero_init_preprocessed_commitment(opts)
        }
    });
    air.with_preprocessed(commitment, global_memory::NUM_PREPROCESSED_COLS)
}

/// The sorted, deduped set of page bases the touched cells fall on — the SINGLE source
/// of truth for which GLOBAL_MEMORY tables exist. The prover builds the committed tables
/// from this list, ships the identical list in the bundle (`ContinuationProof.touched_page_bases`),
/// and the verifier rebuilds the same tables from it. Sorted (BTreeSet order) so prover
/// and verifier iterate the identical sequence — `multi_verify` matches AIRs to sub-proofs
/// positionally. Carries page bases ONLY: no cell values, so private-input bytes never
/// enter the bundle (unlike the full `CellBoundary`, whose `init.value` is a private byte).
fn touched_page_bases(boundaries: &[Arc<Vec<CellBoundary>>]) -> Vec<u64> {
    boundaries
        .iter()
        .flat_map(|epoch| epoch.iter())
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
    let image = build_initial_image_paged(elf, &[], &[]);
    let init_page_data = build_init_page_data(&image);
    global_memory_configs_from_init_page_data(
        page_bases,
        &init_page_data,
        num_private_input_pages,
        false,
    )
}

/// [`global_memory_configs`], but classification-only: whether each page is
/// ELF-backed (an address-range check against `elf.data` segments) or zero-init
/// — never materializing any byte. Correct ONLY when a supplied genesis root
/// covers every classified-ELF-backed page (see `verify_global`'s caller).
fn global_memory_configs_classify_only(
    page_bases: &[u64],
    elf: &Elf,
    num_private_input_pages: usize,
) -> Vec<PageConfig> {
    page_bases
        .iter()
        .map(|&page_base| {
            if page::is_private_input_page(page_base, num_private_input_pages) {
                PageConfig::with_private_input(page_base, Vec::new())
            } else if elf_page_has_data(elf, page_base) {
                PageConfig::with_data(page_base, Vec::new())
            } else {
                PageConfig::zero_init(page_base)
            }
        })
        .collect()
}

/// Whether any ELF segment overlaps the byte range `[page_base, page_base + DEFAULT_PAGE_SIZE)`.
/// `elf.data` is small (a handful of `PT_LOAD` segments) and sorted by `base_addr`, so this
/// is cheap without needing a full byte-level image.
fn elf_page_has_data(elf: &Elf, page_base: u64) -> bool {
    // Saturating: `page_base` can be the stack's page, right at `STACK_TOP =
    // 0xFFFFFFFFFFFFFFF0` — `page_base + DEFAULT_PAGE_SIZE` overflows there.
    let page_end = page_base.saturating_add(page::DEFAULT_PAGE_SIZE as u64);
    elf.data.iter().any(|segment| {
        let seg_start = segment.base_addr;
        // 4 bytes/word (`Segment::values: Vec<u32>`); `executor::elf::WORD_SIZE` is crate-private.
        let seg_end = seg_start.saturating_add(segment.values.len() as u64 * 4);
        seg_start < page_end && page_base < seg_end
    })
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

/// One epoch's proving inputs, fully derived from execution (register init,
/// traces, boundary — no dependency on any previous epoch's *proof*), so the
/// preparation of epoch i+1 can run on a producer thread while epoch i proves.
///
/// `boundary` is shared (`Arc`): the same per-epoch boundary feeds both this
/// epoch's prove and the cross-epoch global prove, which starts as soon as the
/// producer has prepared the last epoch (see `prove_continuation`).
struct PreparedEpoch {
    index: u64,
    register_init: Vec<u32>,
    label: u64,
    traces: Traces,
    boundary: Arc<Vec<CellBoundary>>,
    is_final: bool,
}

/// A collected-but-not-yet-built epoch, handed from the producer to the trace
/// builder pool. Everything sequential (execution, op collection over the
/// advancing memory image, boundary + register-fini derivation) already
/// happened on the producer; a builder turns `collected` into full trace
/// tables ([`Traces::build_from_collected`]) — pure epoch-local work — and
/// forwards the resulting [`PreparedEpoch`] to the epoch prover.
struct BuildJob {
    index: u64,
    register_init: Vec<u32>,
    label: u64,
    collected: crate::tables::trace_builder::CollectedEpoch,
    boundary: Arc<Vec<CellBoundary>>,
    is_final: bool,
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
#[derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
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
    /// [`verify_l2g_commitment_binding_view`].
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
/// rkyv, so it round-trips exactly like a monolithic `VmProof`.
#[derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
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

/// Borrowed view over an [`EpochProof`] (owned or archived-in-place). Lets
/// `verify_epoch` take a single argument again instead of the field-by-field
/// parameter list the owned/archived split used to force on every caller:
/// each accessor reads straight off whichever representation is behind it, a
/// plain field copy on the owned side and (for the small metadata fields) an
/// `rkyv::deserialize` on the archived side.
#[derive(Clone, Copy)]
enum EpochProofView<'a> {
    Owned(&'a EpochProof),
    Archived(&'a ArchivedEpochProof),
}

impl<'a> EpochProofView<'a> {
    /// The epoch's STARK proof (its tables + the epoch-local L2G sub-table
    /// last), as a [`MultiProofView`] — never materialized into an owned
    /// `MultiProof` on the archived side.
    fn proof(&self) -> MultiProofView<'a, F, E, ()> {
        match self {
            Self::Owned(e) => MultiProofView::Owned(&e.proof),
            Self::Archived(e) => MultiProofView::Archived(&e.proof),
        }
    }

    /// Bytes this epoch committed (zero-copy borrow either way).
    fn public_output(&self) -> &'a [u8] {
        match self {
            Self::Owned(e) => &e.public_output,
            Self::Archived(e) => e.public_output.as_slice(),
        }
    }

    fn table_counts(&self) -> Result<TableCounts, Error> {
        match self {
            Self::Owned(e) => Ok(e.table_counts.clone()),
            Self::Archived(e) => {
                rkyv::deserialize::<TableCounts, rkyv::rancor::Error>(&e.table_counts).map_err(
                    |err| Error::Execution(format!("rkyv deserialize table_counts failed: {err}")),
                )
            }
        }
    }

    /// Always empty for continuation epochs (PAGE is skipped); still routed
    /// through the archive rather than assumed, so a malformed non-empty
    /// bundle value surfaces instead of being silently ignored.
    fn runtime_page_ranges(&self) -> Result<Vec<RuntimePageRange>, Error> {
        match self {
            Self::Owned(e) => Ok(e.runtime_page_ranges.clone()),
            Self::Archived(e) => rkyv::deserialize::<Vec<RuntimePageRange>, rkyv::rancor::Error>(
                &e.runtime_page_ranges,
            )
            .map_err(|err| Error::Execution(format!("rkyv deserialize page ranges failed: {err}"))),
        }
    }

    /// Length of `reg_fini` without materializing it — used for the
    /// up-front malformed-bundle check, which only needs the count.
    fn reg_fini_len(&self) -> usize {
        match self {
            Self::Owned(e) => e.reg_fini.len(),
            Self::Archived(e) => e.reg_fini.len(),
        }
    }

    fn reg_fini(&self) -> Result<Vec<u32>, Error> {
        match self {
            Self::Owned(e) => Ok(e.reg_fini.clone()),
            Self::Archived(e) => rkyv::deserialize::<Vec<u32>, rkyv::rancor::Error>(&e.reg_fini)
                .map_err(|err| {
                    Error::Execution(format!("rkyv deserialize reg_fini failed: {err}"))
                }),
        }
    }

    fn l2g_root(&self) -> Commitment {
        match self {
            Self::Owned(e) => e.l2g_root,
            Self::Archived(e) => e.l2g_root,
        }
    }
}

/// Borrowed view over a [`ContinuationProof`] (owned or archived-in-place),
/// mirroring [`EpochProofView`] one level up. Lets
/// [`verify_continuation_with_roots`] and [`verify_continuation_archived`]
/// share one implementation ([`verify_continuation_view`]) instead of two
/// near-duplicate ~130-line bodies.
#[derive(Clone, Copy)]
enum ContinuationProofView<'a> {
    Owned(&'a ContinuationProof),
    Archived(&'a ArchivedContinuationProof),
}

impl<'a> ContinuationProofView<'a> {
    fn num_epochs(&self) -> usize {
        match self {
            Self::Owned(c) => c.epochs.len(),
            Self::Archived(c) => c.epochs.len(),
        }
    }

    fn epoch(&self, i: usize) -> EpochProofView<'a> {
        match self {
            Self::Owned(c) => EpochProofView::Owned(&c.epochs[i]),
            Self::Archived(c) => EpochProofView::Archived(&c.epochs.as_slice()[i]),
        }
    }

    fn epochs(&self) -> impl Iterator<Item = EpochProofView<'a>> {
        let this = *self;
        (0..this.num_epochs()).map(move |i| this.epoch(i))
    }

    /// The one cross-epoch global-memory proof, as a [`MultiProofView`].
    fn global(&self) -> MultiProofView<'a, F, E, ()> {
        match self {
            Self::Owned(c) => MultiProofView::Owned(&c.global),
            Self::Archived(c) => MultiProofView::Archived(&c.global),
        }
    }

    fn num_private_input_pages(&self) -> usize {
        match self {
            Self::Owned(c) => c.num_private_input_pages,
            Self::Archived(c) => c.num_private_input_pages.to_native() as usize,
        }
    }

    fn touched_page_bases(&self) -> Vec<u64> {
        match self {
            Self::Owned(c) => c.touched_page_bases.clone(),
            Self::Archived(c) => c.touched_page_bases.iter().map(|v| v.to_native()).collect(),
        }
    }
}

/// Build an epoch's AIRs identically on the prove and verify sides — the single
/// source of truth for the AIR set, so the two halves can never diverge. The set
/// is `VmAirs` (HALT included iff `is_final`), with REGISTER preprocessed to
/// INIT = `register_init` and FINI = `reg_fini`. Continuation epochs
/// use the L2G bookend, so PAGE is skipped and `page_configs` is empty. The
/// epoch-local L2G air is built separately by the caller (it needs the `label`).
#[allow(clippy::too_many_arguments)]
fn build_epoch_airs(
    elf: &Elf,
    opts: &ProofOptions,
    page_configs: &[PageConfig],
    table_counts: &TableCounts,
    register_init: &[u32],
    reg_fini: &[u32],
    is_final: bool,
    decode_commitment: Option<Commitment>,
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
        decode_commitment,
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
    decode_commitment: Commitment,
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
        // Computed once per prove_continuation — the DECODE commitment is a
        // function of (ELF, opts) only, identical for every epoch; passing
        // None here would rebuild the whole DECODE trace+LDE+tree per epoch.
        Some(decode_commitment),
    );

    let label = start.label;
    let seed = || {
        epoch_transcript(
            elf_bytes,
            &public_output,
            &table_counts,
            &runtime_page_ranges,
            label,
            opts.fri_final_poly_log_degree,
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

/// Verify one epoch using ONLY the epoch's public statement fields (via
/// [`EpochProofView`]) plus the verifier-derived `register_init` (epoch 0:
/// from the ELF; epoch i>0: from the previous epoch's `reg_fini`), `is_final`,
/// and `label`. Rebuilds the AIRs and transcript from the bundle's statement
/// values and indexes commits from the carried x254
/// (`register_init[X254_INDEX]`), never from the prover's memory. PAGE is
/// skipped for continuation epochs, so the AIRs are built with no page configs
/// (the bundle does not get to supply any). Returns `Ok(true)` iff the proof
/// verifies and its committed L2G root matches the claimed one; `Err` iff a
/// small metadata field failed to materialize off an archived bundle.
///
/// `epoch` is zero-copy either way: owned or archived (see the two callers).
#[allow(clippy::too_many_arguments)]
fn verify_epoch(
    elf: &Elf,
    elf_bytes: &[u8],
    epoch: EpochProofView<'_>,
    register_init: &[u32],
    is_final: bool,
    label: u64,
    opts: &ProofOptions,
    decode_commitment: Option<Commitment>,
) -> Result<bool, Error> {
    let table_counts = epoch.table_counts()?;
    // Reject degenerate table counts (mirrors the monolithic verifier).
    if table_counts.validate().is_err() {
        return Ok(false);
    }

    // Cross-check table_counts before building AIRs from bundle data. Continuation
    // epochs have no PAGE proofs, and append one epoch-local L2G proof after the VM
    // tables. HALT is present only on the final epoch.
    let fixed_tables = if is_final {
        FIXED_TABLE_COUNT
    } else {
        FIXED_TABLE_COUNT - 1
    };
    let proof = epoch.proof();
    let expected_proof_count = table_counts.total() + fixed_tables + 1;
    if expected_proof_count != proof.len() {
        return Ok(false);
    }

    let reg_fini = epoch.reg_fini()?;
    let runtime_page_ranges = epoch.runtime_page_ranges()?;
    let public_output = epoch.public_output();

    let airs = build_epoch_airs(
        elf,
        opts,
        &[],
        &table_counts,
        register_init,
        &reg_fini,
        is_final,
        decode_commitment,
    );
    let l2g_air = l2g_memory_air(opts, label);
    let mut refs = airs.air_refs();
    refs.push(&l2g_air);

    let seed = || {
        epoch_transcript(
            elf_bytes,
            public_output,
            &table_counts,
            &runtime_page_ranges,
            label,
            opts.fri_final_poly_log_degree,
        )
    };

    // Start the commit index from the carried x254 (the derived INIT), not a free
    // input — this is what binds the per-epoch commit slice to its global position.
    let commit_start_index = register_init
        .get(register::X254_INDEX)
        .copied()
        .unwrap_or(0) as u64;

    let expected = match compute_expected_commit_bus_balance_view(
        &refs,
        proof,
        public_output,
        commit_start_index,
        &mut seed(),
    ) {
        Some(expected) => expected,
        None => return Ok(false),
    };

    stark::profile_markers::step_marker::<{ stark::profile_markers::STEP_AIRS_AND_BUS_BALANCE_DONE }>(
    );

    if !Verifier::multi_verify_views(&refs, proof, &mut seed(), &expected) {
        return Ok(false);
    }

    // The claimed L2G root must be the one this proof actually committed (it is what
    // verify_l2g_commitment_binding_view later ties to the global proof).
    Ok(proof.last().map(|p| *p.lde_trace_main_merkle_root()) == Some(epoch.l2g_root()))
}

/// Build the cross-epoch global memory proof: every epoch's L2G sub-table on the
/// GlobalMemory bus, plus one GLOBAL_MEMORY table per touched page that sends each
/// cell's genesis init and receives its final value. For ELF/runtime pages the genesis
/// is preprocessed (the verifier recomputes it from the ELF); private-input pages are
/// non-preprocessed (committed, bus-enforced genesis — see `global_memory_air` / §3.6).
/// The bus balances iff every `fini` matches the next epoch's `init` and every genesis
/// matches its source (the ELF for ELF/runtime pages).
fn prove_global(
    boundaries: &[Arc<Vec<CellBoundary>>],
    elf_bytes: &[u8],
    init_page_data: &HashMap<u64, Vec<u8>>,
    page_bases: &[u64],
    num_private_input_pages: usize,
    opts: &ProofOptions,
) -> Result<MultiProof<F, E, ()>, Error> {
    // Each cell's final state (boundaries are in epoch order, so the last fini wins).
    let mut final_state: global_memory::FiniStateMap = HashMap::new();
    for epoch in boundaries {
        for b in epoch.iter() {
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
        .map(|epoch| local_to_global::generate_local_to_global_trace(epoch.as_slice()))
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
        .map(|config| global_memory_air(opts, config, None))
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
            opts.fri_final_poly_log_degree,
            page_bases,
        ),
        #[cfg(feature = "disk-spill")]
        stark::storage_mode::StorageMode::Ram,
    )
    .map_err(|e| Error::Prover(format!("{e:?}")))
}

#[allow(clippy::too_many_arguments)]
fn verify_global(
    num_epochs: usize,
    page_bases: &[u64],
    proof: MultiProofView<'_, F, E, ()>,
    elf: &Elf,
    elf_bytes: &[u8],
    num_private_input_pages: usize,
    opts: &ProofOptions,
    page_genesis_commitments: Option<&[(u64, Commitment)]>,
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
    //
    // `page_genesis_commitments` (the recursion guest's supplied roots) skips the
    // per-data-page recompute; a supplied root shifts the genesis binding to the
    // attestation fold + consumer recompute, exactly like the monolithic guest's
    // `page_commitments`. Zero-init pages always share one commitment, computed
    // once here rather than per touched page.
    let gm_configs = if page_genesis_commitments.is_some() {
        global_memory_configs_classify_only(page_bases, elf, num_private_input_pages)
    } else {
        global_memory_configs(page_bases, elf, num_private_input_pages)
    };
    // Keyed by raw page_base, same as the monolithic path's `page_commitments`
    // lookup (`lib.rs`).
    let supplied: HashMap<u64, Commitment> = page_genesis_commitments
        .map(|s| s.iter().copied().collect())
        .unwrap_or_default();
    // A missing entry here would leave `global_memory_air` to recompute over the
    // classify-only (empty) `init_values`, yielding the zero-init root instead of
    // the real genesis — an honest proof would then fail `multi_verify`, but
    // silently and confusingly. Reject explicitly instead.
    if page_genesis_commitments.is_some()
        && gm_configs
            .iter()
            .filter(|c| !c.is_private_input && c.init_values.is_some())
            .any(|c| !supplied.contains_key(&c.page_base))
    {
        return false;
    }
    let zero_init_root = page::zero_init_preprocessed_commitment(opts);
    let gm_airs: Vec<_> = gm_configs
        .iter()
        .map(|config| {
            let preprocessed = if config.is_private_input {
                None
            } else if config.init_values.is_some() {
                supplied.get(&config.page_base).copied()
            } else {
                Some(zero_init_root)
            };
            global_memory_air(opts, config, preprocessed)
        })
        .collect();

    let mut refs: Vec<AirRef> = l2g_airs.iter().map(|a| a as AirRef).collect();
    for air in &gm_airs {
        refs.push(air as AirRef);
    }

    stark::profile_markers::step_marker::<{ stark::profile_markers::STEP_AIRS_AND_BUS_BALANCE_DONE }>(
    );

    Verifier::multi_verify_views(
        &refs,
        proof,
        &mut global_transcript(
            elf_bytes,
            num_epochs,
            num_private_input_pages,
            opts.fri_final_poly_log_degree,
            page_bases,
        ),
        &FieldElement::zero(),
    )
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
    hints: &[[u8; 32]],
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

    // Root span for the profiling toolkit (scripts/profiling): the whole
    // continuation prove is one tree; per-stage spans below are recorded from
    // their worker threads and told apart by label + instance order.
    #[cfg(feature = "instruments")]
    stark::instruments::reset_timeline();
    #[cfg(feature = "instruments")]
    let __root = stark::instruments::span("prove_continuation_total");

    let elf = Elf::load(elf_bytes).map_err(|e| Error::ElfLoad(format!("{e}")))?;
    let resolved_hints = crate::resolve_hints(&elf, private_inputs, hints)?;
    let hints: &[[u8; 32]] = &resolved_hints;
    let mut executor = Executor::new(&elf, private_inputs.to_vec(), hints)
        .map_err(|e| Error::Execution(format!("{e}")))?;
    // The DECODE precomputed commitment depends only on (ELF, opts): compute
    // it once here instead of once per epoch inside `build_epoch_airs`.
    let decode_commitment = crate::tables::decode::commitment_from_elf(&elf, opts)
        .map_err(|e| Error::Recursion(format!("DECODE commitment from ELF: {e}")))?;
    // Same for the DECODE trace artifacts (instruction map + pristine trace):
    // a pure function of the ELF, built once and shared by every epoch's trace
    // build instead of re-parsed/regenerated inside the serial producer chain.
    let decode_artifacts = DecodeArtifacts::from_elf(&elf)?;

    // The cross-epoch memory image, carried forward: epoch i+1's init is epoch i's
    // fini, updated in place with each epoch's touched-cell final values.
    let mut image = build_initial_image_paged(&elf, private_inputs, hints);
    let init_page_data = build_init_page_data(&image);
    let mut provenance =
        local_to_global::genesis_provenance(image.iter().map(|(a, v)| (a, v as u64)));

    let mut epochs: Vec<EpochProof> = Vec::new();
    // Full per-epoch boundaries, kept prover-local for `prove_global` (L2G traces +
    // final-state). Deliberately NOT stored in `EpochProof`/the bundle — `CellBoundary`
    // holds cell values (private-input bytes for private reads); only the value-free
    // page-base set is shipped (see `touched_page_bases`).
    //
    // The producer publishes each epoch's boundary (an `Arc` share of the one it
    // sends to the epoch prover) on this dedicated channel, in epoch order. The
    // global-prove thread drains it until the producer hangs up (last epoch
    // prepared) — the global proof depends only on these execution artifacts,
    // never on an epoch *proof*, so it overlaps the epoch proves' tail instead
    // of serializing after them. Proof bytes are unchanged — only the schedule.
    let (boundary_tx, boundary_rx) = std::sync::mpsc::channel::<Arc<Vec<CellBoundary>>>();

    // Three-stage epoch pipeline: a producer thread runs the
    // sequential-critical work (execute + op collection over the advancing
    // memory image + boundary/fini derivation), a small pool of trace builders
    // turns collected epochs into trace tables, and a single prover proves
    // them. Everything the next epoch's preparation needs is derived from
    // execution, not from proofs or traces: `register_init` comes from the
    // collected register end state (`register_fini`, the same value the
    // generated REGISTER trace binds) and the memory image update comes from
    // the boundary — so the producer chains epochs without waiting for any
    // table to be built. Proof bytes are unchanged — only the schedule is.
    //
    // The bounded channels cap peak memory: at most one collected epoch
    // queued, `builders` building, one built epoch queued, one proving.
    let (tx, rx) = std::sync::mpsc::sync_channel::<Result<PreparedEpoch, Error>>(1);
    let (build_tx, build_rx) = std::sync::mpsc::sync_channel::<Result<BuildJob, Error>>(1);
    // Trace builders: each turns one collected epoch into full trace tables
    // (the bulk of the old per-epoch producer latency). 2 is enough to keep
    // the prove pipeline fed on the measured workloads; builds compete with
    // proves for CPU, so more builders mostly reshuffle the same cores.
    let builders = std::env::var("LAMBDA_VM_TRACE_BUILDERS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&b| b >= 1)
        .unwrap_or(2);
    let build_rx = std::sync::Mutex::new(build_rx);
    type EpochResult = (u64, EpochProof);
    let first_err: std::sync::Mutex<Option<Error>> = std::sync::Mutex::new(None);
    let decode_artifacts_ref = &decode_artifacts;
    let first_err_ref = &first_err;
    // On error the prover DRAINS the channel (discarding items) instead of
    // returning: the senders are bounded and can only unblock via a recv, so
    // an early return would leave a builder parked in `send` forever and the
    // scope would never join. Draining ends when every sender is dropped.
    let prove_worker = |rx: std::sync::mpsc::Receiver<Result<PreparedEpoch, Error>>| {
        let mut proved: Vec<EpochResult> = Vec::new();
        loop {
            let prepared = match rx.recv() {
                Ok(Ok(p)) => p,
                Ok(Err(e)) => {
                    first_err.lock().unwrap().get_or_insert(e);
                    continue;
                }
                Err(_) => return proved, // channel closed: no more epochs
            };
            if first_err.lock().unwrap().is_some() {
                continue; // an earlier failure is propagating; drain and discard
            }
            // Per-epoch identity on Nsight timelines (dynamic NVTX name); the
            // instruments span carries a static label and instances are told
            // apart by order (phase_table.py reports them per instance).
            #[cfg(feature = "nvtx")]
            let __nvtx =
                stark::instruments::nvtx_range_fmt(|| format!("epoch_prove[i={}]", prepared.index));
            #[cfg(feature = "instruments")]
            let __sp = stark::instruments::span("epoch_prove");
            let start = EpochStart {
                register_init: &prepared.register_init,
                label: prepared.label,
            };
            match prove_epoch(
                &elf,
                elf_bytes,
                &start,
                prepared.traces,
                prepared.is_final,
                &prepared.boundary,
                opts,
                decode_commitment,
            ) {
                Ok(epoch) => proved.push((prepared.index, epoch)),
                Err(e) => {
                    first_err.lock().unwrap().get_or_insert(e);
                    continue; // drain mode (see loop comment)
                }
            }
        }
    };
    // Trace-builder worker: drain collected epochs, build their trace tables
    // (pure epoch-local work) and forward the prepared epoch to the prover.
    // Errors propagate through the prove channel, exactly like producer errors.
    //
    // Test-only fault injection, keyed by a magic private input no real caller
    // passes (stateless, so concurrent tests can never trip it): exercises the
    // mid-pipeline error path, which must return `Err` instead of wedging the
    // bounded channels (see `test_fault`).
    let build_worker = |tx: std::sync::mpsc::SyncSender<Result<PreparedEpoch, Error>>| {
        loop {
            let msg = { build_rx.lock().unwrap().recv() };
            let job = match msg {
                Ok(Ok(j)) => j,
                Ok(Err(e)) => {
                    // Forward and keep draining (same reason as the
                    // prover: a return would strand the producer's send).
                    let _ = tx.send(Err(e));
                    continue;
                }
                Err(_) => return, // channel closed: no more epochs
            };
            if first_err.lock().unwrap().is_some() {
                continue; // the prover failed; drain and discard
            }
            #[cfg(test)]
            if job.index == test_fault::FAIL_INDEX && private_inputs == test_fault::MAGIC {
                let _ = tx.send(Err(Error::ContinuationInvariant(
                    "injected pipeline fault (test)".to_string(),
                )));
                continue;
            }
            #[cfg(feature = "nvtx")]
            let __nvtx = stark::instruments::nvtx_range_fmt(|| {
                format!("epoch_trace_build[i={}]", job.index)
            });
            #[cfg(feature = "instruments")]
            let __sp = stark::instruments::span("epoch_trace_build");
            let traces = Traces::build_from_collected(
                decode_artifacts_ref,
                job.collected,
                // Continuation epochs use the L2G bookend: PAGE tables (the
                // only image consumers in the build) are skipped.
                None::<&std::collections::HashMap<u64, u8>>,
                &job.register_init,
                &MaxRowsConfig::default(),
                private_inputs,
                hints,
                job.is_final,
                true,
                #[cfg(feature = "disk-spill")]
                stark::storage_mode::StorageMode::Ram,
            );
            // Close the build span BEFORE forwarding: the send below blocks
            // on prove-channel backpressure, which is waiting, not building.
            #[cfg(feature = "instruments")]
            drop(__sp);
            #[cfg(feature = "nvtx")]
            drop(__nvtx);
            match traces {
                Ok(traces) => {
                    // Pre-upload the big main traces from this builder thread
                    // (idle slack ahead of the prover), so the R1 main commits
                    // skip their H2D.
                    #[cfg(feature = "cuda")]
                    let traces = {
                        let mut traces = traces;
                        #[cfg(feature = "instruments")]
                        let __sp = stark::instruments::span("p6_trace_preupload");
                        traces.preupload_main_traces();
                        traces
                    };
                    let prepared = PreparedEpoch {
                        index: job.index,
                        register_init: job.register_init,
                        label: job.label,
                        traces,
                        boundary: job.boundary,
                        is_final: job.is_final,
                    };
                    // A send error means the prover side hung up (its error is
                    // already propagating) — stop quietly.
                    if tx.send(Ok(prepared)).is_err() {
                        return;
                    }
                }
                Err(e) => {
                    let _ = tx.send(Err(e));
                    continue; // drain mode
                }
            }
        }
    };

    // The global prove's result, produced by its own scoped thread. `None` only
    // if that thread never ran to completion (a panic — surfaced by the scope).
    type GlobalResult = (MultiProof<F, E, ()>, Vec<u64>, usize);
    let global_result: std::sync::Mutex<Option<Result<GlobalResult, Error>>> =
        std::sync::Mutex::new(None);
    let mut results = std::thread::scope(|scope| -> Result<Vec<EpochResult>, Error> {
        let elf_ref = &elf;
        let producer = scope.spawn(move || {
            let mut prepare_all = || -> Result<(), Error> {
                let mut prev_fini: Option<Vec<u32>> = None;
                let mut index: u64 = 0;
                loop {
                    if executor.pc() == 0 {
                        return Ok(());
                    }
                    // A downstream failure is already propagating: stop
                    // executing epochs so the pipeline can drain and shut down.
                    if first_err_ref.lock().unwrap().is_some() {
                        return Ok(());
                    }
                    // The cross-epoch ordering check (IsB20 on `fini_epoch - 1 -
                    // init_epoch`) only spans `local_to_global::MAX_EPOCHS` epochs.
                    // Beyond that the IsB20 bus cannot balance, so an honest proof
                    // is impossible — fail fast with a clear error instead of
                    // building an unprovable trace.
                    if index >= local_to_global::MAX_EPOCHS {
                        return Err(Error::InvalidContinuationEpochSize(format!(
                            "execution needs more than {} continuation epochs (the IsB20 \
                             cross-epoch ordering range); use a larger epoch size",
                            local_to_global::MAX_EPOCHS
                        )));
                    }
                    let register_init: Vec<u32> = match (index, prev_fini.take()) {
                        (0, _) => register::register_init_from_entry_point(elf_ref.entry_point),
                        // Epoch i+1's init is epoch i's bound fini, reused directly
                        // (same `register_word_address_list` order) — the cross-epoch
                        // register binding.
                        (_, Some(fini)) => fini,
                        (_, None) => {
                            return Err(Error::ContinuationInvariant(
                                "previous epoch final registers are missing after the first epoch"
                                    .to_string(),
                            ));
                        }
                    };

                    // Run one epoch; `logs` is this epoch's chunk only (the executor
                    // clears it).
                    #[cfg(feature = "instruments")]
                    let __sp = stark::instruments::span("epoch_execute");
                    let logs = match executor
                        .resume_with_limit(epoch_size)
                        .map_err(|e| Error::Execution(format!("{e}")))?
                    {
                        Some(logs) => logs.to_vec(),
                        None => return Ok(()),
                    };
                    #[cfg(feature = "instruments")]
                    drop(__sp);
                    let is_final = executor.pc() == 0;

                    // Invariant: a non-final epoch ran the full `epoch_size` (a power
                    // of two), so its CPU table has no padding rows.
                    if !is_final && logs.len() != epoch_size {
                        return Err(Error::ContinuationInvariant(format!(
                            "intermediate epoch ran {} cycles, expected {epoch_size}",
                            logs.len()
                        )));
                    }

                    let label = local_to_global::epoch_label(index);
                    // Sequential-critical half only (Phases 1-2): op collection
                    // over the pre-epoch image. The table build (Phases 3-5)
                    // happens on the builder pool — nothing below needs it.
                    #[cfg(feature = "nvtx")]
                    let __nvtx =
                        stark::instruments::nvtx_range_fmt(|| format!("epoch_collect[i={index}]"));
                    #[cfg(feature = "instruments")]
                    let __sp = stark::instruments::span("epoch_collect");
                    let collected = Traces::collect_epoch(
                        decode_artifacts_ref,
                        &image,
                        &register_init,
                        &logs,
                        is_final,
                    )?;
                    let boundary = Arc::new(local_to_global::epoch_boundary(
                        &mut provenance,
                        label,
                        &collected.touched_memory_cells(),
                    ));
                    // Publish this epoch's boundary for the global prove (in
                    // epoch order; the channel closes when the producer ends).
                    let _ = boundary_tx.send(Arc::clone(&boundary));

                    // R_{i+1} from the collected register end state — the exact
                    // value the generated REGISTER trace binds (`fini_from_trace`
                    // equivalence pinned by `fini_from_final_state_matches_trace`).
                    prev_fini = Some(collected.register_fini(&register_init));

                    // Carry the image forward: this epoch's fini is the next
                    // epoch's init.
                    for cell in boundary.iter() {
                        image.set(cell.address, (cell.fini.value & 0xFF) as u8);
                    }

                    // Close the collect span BEFORE handing off: the send below
                    // blocks on builder backpressure, which is waiting, not work.
                    #[cfg(feature = "instruments")]
                    drop(__sp);
                    #[cfg(feature = "nvtx")]
                    drop(__nvtx);
                    let job = BuildJob {
                        index,
                        register_init,
                        label,
                        collected,
                        boundary,
                        is_final,
                    };
                    // A send error means the builder side hung up (its error is
                    // already propagating) — stop preparing quietly.
                    if build_tx.send(Ok(job)).is_err() || is_final {
                        return Ok(());
                    }
                    index += 1;
                }
            };
            if let Err(e) = prepare_all() {
                // Surface preparation errors through the builder channel (a
                // builder forwards them to the prover); if the downstream side
                // is already gone the error there wins.
                let _ = build_tx.send(Err(e));
            }
        });

        // Trace-builder pool: collected epochs → trace tables → prove channel.
        // Each builder owns a clone of the prove sender; the original is
        // dropped below so the prover's channel closes once the producer and
        // every builder are done.
        for _ in 0..builders {
            let tx = tx.clone();
            scope.spawn(move || build_worker(tx));
        }
        drop(tx);

        // Global prove, overlapped: drain the boundary channel until the
        // producer hangs up (last epoch prepared), then prove the cross-epoch
        // global memory argument WHILE the tail epochs are still proving. The
        // global proof consumes only execution artifacts (boundaries, ELF,
        // genesis pages) — never an epoch proof — so this is pure schedule.
        let global_result_ref = &global_result;
        let init_page_data_ref = &init_page_data;
        scope.spawn(move || {
            let mut all: Vec<Arc<Vec<CellBoundary>>> = Vec::new();
            while let Ok(b) = boundary_rx.recv() {
                all.push(b);
            }
            // An epoch already failed: its error wins and the bundle is never
            // assembled — skip the (whole-prove-sized) global prove.
            if first_err_ref.lock().unwrap().is_some() {
                return;
            }
            let run = || -> Result<GlobalResult, Error> {
                #[cfg(feature = "instruments")]
                let __sp = stark::instruments::span("prove_global");
                let num_private_input_pages = page::private_input_page_count(private_inputs, hints);
                // SINGLE source of truth: the same page-base list drives the
                // committed GLOBAL_MEMORY tables and is shipped in the bundle,
                // so the two can never diverge in set or order.
                let touched = touched_page_bases(&all);
                let global = prove_global(
                    &all,
                    elf_bytes,
                    init_page_data_ref,
                    &touched,
                    num_private_input_pages,
                    opts,
                )?;
                Ok((global, touched, num_private_input_pages))
            };
            *global_result_ref.lock().unwrap() = Some(run());
        });

        // Prove epochs as the builders hand them over. Builders can finish
        // out of index order, so results are re-ordered by epoch index before
        // the bundle is assembled — proof bytes are identical to the
        // sequential schedule (each epoch is seeded by its own
        // label-domain-separated transcript and no epoch's proof feeds
        // another).
        let prover = scope.spawn(move || prove_worker(rx));
        let proved = prover.join().map_err(|_| {
            Error::ContinuationInvariant("epoch prover thread panicked".to_string())
        })?;
        producer.join().map_err(|_| {
            Error::ContinuationInvariant("epoch preparation thread panicked".to_string())
        })?;
        Ok(proved)
    })?;
    if let Some(e) = first_err.into_inner().unwrap() {
        return Err(e);
    }
    results.sort_by_key(|(index, _)| *index);
    for (_, epoch) in results {
        epochs.push(epoch);
    }

    // One global LogUp over all the (kept) local-to-global tables — proven
    // concurrently by the scoped thread above; collect its result here. The
    // scope guarantees the thread finished, so `None` is unreachable.
    let (global, touched_page_bases, num_private_input_pages) =
        global_result.into_inner().unwrap().ok_or_else(|| {
            Error::ContinuationInvariant("global prove thread produced no result".to_string())
        })??;

    // Same timeline output as the monolithic path (prover/src/lib.rs): print
    // the wall-clock span tree and honor LAMBDA_VM_TIMELINE_JSON. Without this,
    // continuation runs record spans that are never drained (the profiling
    // toolkit's phase_table.py consumes the JSON).
    #[cfg(feature = "instruments")]
    {
        drop(__root);
        let spans = stark::instruments::take_timeline();
        print!("{}", stark::instruments::format_timeline(&spans));
        if let Ok(path) = std::env::var("LAMBDA_VM_TIMELINE_JSON") {
            let _ = std::fs::write(&path, stark::instruments::timeline_json(&spans));
            println!("[timeline] wrote {path}");
        }
    }

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
    verify_continuation_with_roots(elf_bytes, bundle, opts, None, None)
}

/// [`verify_continuation`] with caller-supplied ELF-derived roots: the DECODE
/// preprocessed root (shared by every epoch) and the global-memory genesis
/// roots for touched data pages. Supplied roots are used VERBATIM — they are
/// NOT bound to `elf_bytes` here, exactly like `verify_with_options`' supplied
/// roots on the monolithic path. The recursion guest supplies them via private
/// input to skip the in-VM FFT + Merkle recomputes; on success it folds them
/// into the attestation's `program_id`, and the consumer's recompute+compare
/// is what restores the binding. `None` = recompute from the ELF (the
/// trustless host path).
pub fn verify_continuation_with_roots(
    elf_bytes: &[u8],
    bundle: &ContinuationProof,
    opts: &ProofOptions,
    decode_commitment: Option<Commitment>,
    page_genesis_commitments: Option<&[(u64, Commitment)]>,
) -> Result<Option<Vec<u8>>, Error> {
    let result = verify_continuation_view(
        ContinuationProofView::Owned(bundle),
        elf_bytes,
        opts,
        decode_commitment,
        page_genesis_commitments,
    )?;
    Ok(result.map(|(public_output, _entry_point)| public_output))
}

/// [`verify_continuation_with_roots`]'s zero-copy counterpart, for the
/// recursion `continuation` guest: reads every per-epoch/global proof in
/// place via [`ContinuationProofView::Archived`] instead of deserializing an
/// owned [`MultiProof`]. Only small per-epoch metadata is materialized. Roots
/// are always supplied here (the guest never recomputes from the ELF in-VM).
///
/// Also returns `entry_point` so callers can fold a `program_id` via
/// [`crate::recursion::program_id_from_digest`] without a second `Elf::load`.
pub(crate) fn verify_continuation_archived(
    archived: &ArchivedContinuationProof,
    elf_bytes: &[u8],
    opts: &ProofOptions,
    decode_commitment: Commitment,
    page_genesis_commitments: &[(u64, Commitment)],
) -> Result<Option<(Vec<u8>, u64)>, Error> {
    verify_continuation_view(
        ContinuationProofView::Archived(archived),
        elf_bytes,
        opts,
        Some(decode_commitment),
        Some(page_genesis_commitments),
    )
}

/// Shared implementation behind [`verify_continuation_with_roots`] (owned) and
/// [`verify_continuation_archived`] (archived), operating on a
/// [`ContinuationProofView`] rather than either's concrete type — the same
/// split [`crate::verify_recursion_blob`] uses for the monolithic path.
/// Returns the public output plus `entry_point` (see [`verify_continuation_archived`]).
fn verify_continuation_view(
    bundle: ContinuationProofView<'_>,
    elf_bytes: &[u8],
    opts: &ProofOptions,
    decode_commitment: Option<Commitment>,
    page_genesis_commitments: Option<&[(u64, Commitment)]>,
) -> Result<Option<(Vec<u8>, u64)>, Error> {
    // Bound the claimed private-input page count before using it to size/allocate AIRs
    // (mirrors `verify_with_options`). The count is also bound into the global proof's
    // Fiat-Shamir statement (`absorb_continuation_global_statement`), so any wrong value
    // diverges the verifier's challenges and `verify_global`'s `multi_verify` rejects —
    // on top of the committed-AIR-shape mismatch a wrong count causes on a touched page.
    let max_private_input_pages = page::max_private_input_pages();
    let num_private_input_pages = bundle.num_private_input_pages();
    if num_private_input_pages > max_private_input_pages {
        return Err(Error::InvalidTableCounts(format!(
            "num_private_input_pages ({num_private_input_pages}) exceeds max ({max_private_input_pages})",
        )));
    }

    let elf = Elf::load(elf_bytes).map_err(|e| Error::ElfLoad(format!("{e}")))?;

    let n = bundle.num_epochs();
    if n == 0 {
        return Ok(None);
    }

    // Reject a malformed bundle up front. `reg_fini` is prover-supplied (deserialized,
    // untrusted) and is indexed by `NUM_REGISTER_ADDRESSES` when building each epoch's
    // preprocessed REGISTER commitment, so a wrong length would otherwise panic the
    // verifier instead of cleanly rejecting the proof. Only the length is read here
    // (no materialization) — the values are only needed once we actually verify.
    if bundle
        .epochs()
        .any(|e| e.reg_fini_len() != register::NUM_REGISTER_ADDRESSES)
    {
        return Ok(None);
    }

    // Derived from the ELF for epoch 0, then from each epoch's bound fini.
    let mut register_init = register::register_init_from_entry_point(elf.entry_point);
    let mut epoch_roots: Vec<Commitment> = Vec::with_capacity(n);
    let mut public_output: Vec<u8> = Vec::new();

    for (index, epoch) in bundle.epochs().enumerate() {
        let is_final = index == n - 1;
        let label = local_to_global::epoch_label(index as u64);
        let l2g_root = epoch.l2g_root();
        let epoch_public_output = epoch.public_output();

        if !verify_epoch(
            &elf,
            elf_bytes,
            epoch,
            &register_init,
            is_final,
            label,
            opts,
            decode_commitment,
        )? {
            return Ok(None);
        }

        epoch_roots.push(l2g_root);
        public_output.extend_from_slice(epoch_public_output);
        // Next epoch's init is this epoch's bound fini — the cross-epoch register
        // (and x254) binding. A mismatched fini desyncs the next epoch's AIRs.
        register_init = epoch.reg_fini()?;
    }

    // Cross-epoch global memory: genesis for ELF/runtime pages is rebuilt FROM THE ELF
    // (no private bytes) by default, so the starting memory cannot be prover-chosen —
    // unless `page_genesis_commitments` supplies it verbatim, deferring binding to the
    // caller's recompute-and-compare. Either way the bus telescopes fini→init.
    // Private-input pages are committed, non-preprocessed (genesis not
    // bundled/ELF-recomputed), bus-enforced. The verifier needs only the epoch count and the
    // touched page-base set (never cell values); the bundle carries the latter directly.
    // Canonicalize the (untrusted) list so a shuffled-but-same-set list still verifies,
    // while a different set fails via GlobalMemory-bus imbalance / AIR-count mismatch.
    let touched_page_bases = bundle.touched_page_bases();
    let page_bases = canonical_page_bases(&touched_page_bases);
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
    // Caller-supplied (not bundle) bases feed the same raw-page_base matching;
    // an unaligned one needs the same rejection.
    if let Some(commitments) = page_genesis_commitments
        && commitments
            .iter()
            .any(|&(base, _)| base != page::page_base_for_address(base))
    {
        return Err(Error::MalformedContinuationBundle(
            "page_genesis_commitments contains a non-page-aligned entry".to_string(),
        ));
    }
    let global_proof = bundle.global();
    if !verify_global(
        n,
        &page_bases,
        global_proof,
        &elf,
        elf_bytes,
        num_private_input_pages,
        opts,
        page_genesis_commitments,
    ) {
        return Ok(None);
    }

    // Each epoch's committed L2G table is the same one the global proof used.
    if !verify_l2g_commitment_binding_view(&epoch_roots, global_proof) {
        return Ok(None);
    }

    Ok(Some((public_output, elf.entry_point)))
}

/// Precompute the ELF-derived roots [`verify_continuation_with_roots`] accepts:
/// the DECODE preprocessed root and one genesis root per touched non-private
/// data page (the same set `verify_global` would rebuild from the ELF). These
/// are what a caller packs as a continuation recursion guest's private input,
/// and what a consumer recomputes to re-bind the guest's attestation.
pub fn continuation_precomputed_commitments(
    elf_bytes: &[u8],
    bundle: &ContinuationProof,
    opts: &ProofOptions,
) -> Result<(Commitment, Vec<(u64, Commitment)>), Error> {
    // Same bound as `verify_continuation_with_roots`: `bundle` is untrusted
    // (rkyv-deserialized), and `num_private_input_pages` feeds a `* page_size`
    // multiplication downstream.
    let max_private_input_pages = page::max_private_input_pages();
    if bundle.num_private_input_pages > max_private_input_pages {
        return Err(Error::InvalidTableCounts(format!(
            "num_private_input_pages ({}) exceeds max ({max_private_input_pages})",
            bundle.num_private_input_pages
        )));
    }

    let elf = Elf::load(elf_bytes).map_err(|e| Error::ElfLoad(format!("{e}")))?;
    let decode_commitment = crate::tables::decode::commitment_from_elf(&elf, opts)
        .map_err(|e| Error::Recursion(format!("DECODE commitment from ELF: {e}")))?;
    let page_bases = canonical_page_bases(&bundle.touched_page_bases);
    let page_commitments = global_memory_configs(&page_bases, &elf, bundle.num_private_input_pages)
        .iter()
        .filter(|c| !c.is_private_input && c.init_values.is_some())
        .map(|c| (c.page_base, page::compute_precomputed_commitment(c, opts)))
        .collect();
    Ok((decode_commitment, page_commitments))
}

/// Convenience wrapper: prove then verify in one call (the original integrated API).
/// Returns `Ok(Some(public_output))` iff the continuation proves and verifies.
pub fn prove_and_verify_continuation(
    elf_bytes: &[u8],
    private_inputs: &[u8],
    epoch_size_log2: u32,
    opts: &ProofOptions,
) -> Result<Option<Vec<u8>>, Error> {
    let bundle = prove_continuation(elf_bytes, private_inputs, &[], epoch_size_log2, opts)?;
    verify_continuation(elf_bytes, &bundle, opts)
}

/// Stateless test-only fault trigger for the epoch pipeline: the builder
/// injects an error at epoch [`FAIL_INDEX`] when the prove's private input is
/// exactly [`MAGIC`]. Constants only — concurrent tests can never trip it.
#[cfg(test)]
pub(crate) mod test_fault {
    pub(crate) const MAGIC: &[u8] = b"__inject_pipeline_fault__";
    pub(crate) const FAIL_INDEX: u64 = 3;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::asm_elf_bytes;

    // Diagnostic (not a regression test): structurally diff two continuation
    // proof bundles of the same input. The prover is deterministic, so the
    // first differing field per table names the round where a corrupt run
    // diverged. Run with:
    //   PROOF_A=<good.bin> PROOF_B=<bad.bin> \
    //   cargo test -p prover --release proof_diff -- --ignored --nocapture
    #[test]
    #[ignore]
    fn proof_diff() {
        fn load(path: &str) -> ContinuationProof {
            use std::os::unix::fs::FileExt;
            let file = std::fs::File::open(path).unwrap();
            let len = file.metadata().unwrap().len() as usize;
            let mut aligned = rkyv::util::AlignedVec::<16>::with_capacity(len);
            aligned.resize(len, 0);
            file.read_exact_at(&mut aligned, 0).unwrap();
            rkyv::from_bytes::<ContinuationProof, rkyv::rancor::Error>(&aligned).unwrap()
        }
        fn table_eq(a: &stark::table::Table<E>, b: &stark::table::Table<E>) -> bool {
            if a.width != b.width || a.height != b.height {
                return false;
            }
            (0..a.height).all(|r| (0..a.width).all(|c| a.get(r, c) == b.get(r, c)))
        }
        fn diff_multi(label: &str, a: &MultiProof<F, E, ()>, b: &MultiProof<F, E, ()>) {
            assert_eq!(a.proofs.len(), b.proofs.len(), "{label}: table count");
            for (t, (pa, pb)) in a.proofs.iter().zip(b.proofs.iter()).enumerate() {
                let mut d = Vec::new();
                if pa.lde_trace_main_merkle_root != pb.lde_trace_main_merkle_root {
                    d.push("main_root");
                }
                if pa.lde_trace_aux_merkle_root != pb.lde_trace_aux_merkle_root {
                    d.push("aux_root");
                }
                if pa.lde_trace_precomputed_merkle_root != pb.lde_trace_precomputed_merkle_root {
                    d.push("preproc_root");
                }
                if pa.bus_public_inputs.as_ref().map(|x| &x.table_contribution)
                    != pb.bus_public_inputs.as_ref().map(|x| &x.table_contribution)
                {
                    d.push("bus_pi");
                }
                if pa.composition_poly_root != pb.composition_poly_root {
                    d.push("comp_root");
                }
                if !table_eq(&pa.trace_ood_evaluations, &pb.trace_ood_evaluations) {
                    d.push("trace_ood");
                }
                if !table_eq(
                    &pa.trace_ood_next_evaluations,
                    &pb.trace_ood_next_evaluations,
                ) {
                    d.push("trace_ood_next");
                }
                if pa.composition_poly_parts_ood_evaluation
                    != pb.composition_poly_parts_ood_evaluation
                {
                    d.push("parts_ood");
                }
                if pa.fri_layers_merkle_roots != pb.fri_layers_merkle_roots {
                    d.push("fri_roots");
                }
                if pa.fri_final_poly_coeffs != pb.fri_final_poly_coeffs {
                    d.push("fri_final");
                }
                if pa.nonce != pb.nonce {
                    d.push("nonce");
                }
                if !d.is_empty() {
                    println!(
                        "{label} table {t} (cols={} len={}): {d:?}",
                        pa.trace_ood_evaluations.width, pa.trace_length
                    );
                }
            }
        }
        let a = load(&std::env::var("PROOF_A").unwrap());
        let b = load(&std::env::var("PROOF_B").unwrap());
        assert_eq!(a.epochs.len(), b.epochs.len(), "epoch count");
        for (e, (ea, eb)) in a.epochs.iter().zip(b.epochs.iter()).enumerate() {
            diff_multi(&format!("epoch {e}"), &ea.proof, &eb.proof);
            if ea.public_output != eb.public_output {
                println!("epoch {e}: public_output differs");
            }
            if ea.reg_fini != eb.reg_fini {
                println!("epoch {e}: reg_fini differs");
            }
            if ea.l2g_root != eb.l2g_root {
                println!("epoch {e}: l2g_root differs");
            }
        }
        diff_multi("global", &a.global, &b.global);
        println!("diff complete");
    }

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

        let total = Executor::new(&Elf::load(&elf_bytes).unwrap(), vec![], &[])
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
        let bundle = prove_continuation(
            &elf_bytes,
            &[],
            &[],
            4,
            &ProofOptions::default_test_options(),
        )
        .unwrap();
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

    // The pipeline's error path: a mid-run failure with several epochs still
    // pending (past the bounded channels' slack) must surface as `Err` — the
    // regression this guards wedged every channel and hung `prove_continuation`
    // forever. Run under a timeout so a regression fails instead of hanging CI.
    #[test]
    fn test_prove_error_mid_pipeline_returns_err() {
        let _ = env_logger::builder().is_test(true).try_init();
        let elf_bytes = asm_elf_bytes("all_loadstore_32");
        // 4-cycle epochs over ~34 cycles → ~9 epochs; the injected failure at
        // epoch 3 leaves enough pending work to fill every bounded channel.
        let (done_tx, done_rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let r = prove_continuation(
                &elf_bytes,
                test_fault::MAGIC,
                &[],
                2,
                &ProofOptions::default_test_options(),
            );
            let _ = done_tx.send(r.map(|_| ()));
        });
        let result = done_rx
            .recv_timeout(std::time::Duration::from_secs(300))
            .expect("prove_continuation wedged: the pipeline did not shut down on error");
        let err = result.expect_err("the injected fault must surface as Err");
        assert!(
            format!("{err:?}").contains("injected pipeline fault"),
            "unexpected error: {err:?}"
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
        let total = Executor::new(&Elf::load(&elf_bytes).unwrap(), vec![], &[])
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

    // Supplied genesis roots must verify identically to the trustless recompute,
    // and a tampered root (DECODE or a page) must be rejected. `data_page_touch`
    // touches a real ELF `.data` page, unlike this file's stack-only fixtures.
    #[test]
    fn test_verify_continuation_with_supplied_roots() {
        let elf_bytes = asm_elf_bytes("data_page_touch");
        let opts = ProofOptions::default_test_options();
        let bundle = prove_continuation(&elf_bytes, &[], &[], 3, &opts).unwrap();

        let expected = verify_continuation(&elf_bytes, &bundle, &opts)
            .unwrap()
            .expect("trustless verify must accept an honest bundle");

        let (decode_commitment, page_commitments) =
            continuation_precomputed_commitments(&elf_bytes, &bundle, &opts).unwrap();
        assert!(
            !page_commitments.is_empty(),
            "fixture must touch at least one ELF data page"
        );
        let got = verify_continuation_with_roots(
            &elf_bytes,
            &bundle,
            &opts,
            Some(decode_commitment),
            Some(&page_commitments),
        )
        .unwrap()
        .expect("supplied-roots verify must accept the same honest bundle");
        assert_eq!(
            got, expected,
            "supplied-roots output must match the recompute path"
        );

        let mut tampered_page_commitments = page_commitments.clone();
        tampered_page_commitments[0].1[0] ^= 0xFF;
        let rejected = verify_continuation_with_roots(
            &elf_bytes,
            &bundle,
            &opts,
            Some(decode_commitment),
            Some(&tampered_page_commitments),
        )
        .unwrap();
        assert!(
            rejected.is_none(),
            "a tampered supplied page genesis root must be rejected"
        );

        let mut zeroed_page_commitments = page_commitments.clone();
        zeroed_page_commitments[0].1 = [0u8; 32];
        let rejected = verify_continuation_with_roots(
            &elf_bytes,
            &bundle,
            &opts,
            Some(decode_commitment),
            Some(&zeroed_page_commitments),
        )
        .unwrap();
        assert!(
            rejected.is_none(),
            "an all-zero supplied page genesis root must be rejected"
        );

        let mut tampered_decode = decode_commitment;
        tampered_decode[0] ^= 0xFF;
        let rejected = verify_continuation_with_roots(
            &elf_bytes,
            &bundle,
            &opts,
            Some(tampered_decode),
            Some(&page_commitments),
        )
        .unwrap();
        assert!(
            rejected.is_none(),
            "a tampered supplied DECODE root must be rejected"
        );
    }

    // Locks in the equivalence `verify_global`'s supplied-roots path relies on:
    // `global_memory_configs_classify_only` (range-overlap) must classify each page
    // identically (same private/data/zero-init kind) to `global_memory_configs`
    // (byte-level image), for both a data-touching and a stack-only fixture.
    #[test]
    fn test_classify_only_matches_byte_level_classification() {
        for name in ["data_page_touch", "all_loadstore_32"] {
            let elf_bytes = asm_elf_bytes(name);
            let opts = ProofOptions::default_test_options();
            let bundle = prove_continuation(&elf_bytes, &[], &[], 3, &opts).unwrap();
            let elf = Elf::load(&elf_bytes).unwrap();
            let page_bases = canonical_page_bases(&bundle.touched_page_bases);

            let byte_level =
                global_memory_configs(&page_bases, &elf, bundle.num_private_input_pages);
            let classify_only = global_memory_configs_classify_only(
                &page_bases,
                &elf,
                bundle.num_private_input_pages,
            );

            assert_eq!(byte_level.len(), classify_only.len(), "fixture: {name}");
            for (a, b) in byte_level.iter().zip(classify_only.iter()) {
                assert_eq!(a.page_base, b.page_base, "fixture: {name}");
                assert_eq!(a.is_private_input, b.is_private_input, "fixture: {name}");
                assert_eq!(
                    a.init_values.is_some(),
                    b.init_values.is_some(),
                    "fixture: {name}, page_base: {}",
                    a.page_base
                );
            }
        }
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
        let total = Executor::new(&Elf::load(&elf_bytes).unwrap(), vec![], &[])
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
            prove_continuation(&[], &[], &[], 1, &ProofOptions::default_test_options()),
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
        let bundle = prove_continuation(
            &elf_bytes,
            &[],
            &[],
            4,
            &ProofOptions::default_test_options(),
        )
        .unwrap();
        let out = verify_continuation(&elf_bytes, &bundle, &ProofOptions::default_test_options())
            .unwrap();
        assert_eq!(out.as_deref(), Some(&[0xAA, 0xBB, 0xCC, 0xDD][..]));
    }

    // A bundle survives an rkyv round-trip and still verifies to the same output —
    // the serialization path the CLI's `prove`/`verify --continuations` relies on.
    #[test]
    fn test_continuation_rkyv_roundtrip() {
        let _ = env_logger::builder().is_test(true).try_init();
        let elf_bytes = asm_elf_bytes("test_commit_split");
        let bundle = prove_continuation(
            &elf_bytes,
            &[],
            &[],
            4,
            &ProofOptions::default_test_options(),
        )
        .unwrap();

        let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&bundle).unwrap();
        let restored: ContinuationProof =
            rkyv::from_bytes::<_, rkyv::rancor::Error>(&bytes).unwrap();

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
        let mut bundle = prove_continuation(
            &elf_bytes,
            &[],
            &[],
            3,
            &ProofOptions::default_test_options(),
        )
        .unwrap();
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
        let mut bundle = prove_continuation(
            &elf_bytes,
            &[],
            &[],
            3,
            &ProofOptions::default_test_options(),
        )
        .unwrap();
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
        let mut bundle = prove_continuation(
            &elf_bytes,
            &[],
            &[],
            3,
            &ProofOptions::default_test_options(),
        )
        .unwrap();
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
        let mut bundle = prove_continuation(
            &elf_bytes,
            &[],
            &[],
            3,
            &ProofOptions::default_test_options(),
        )
        .unwrap();
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
        let mut bundle = prove_continuation(
            &elf_bytes,
            &[],
            &[],
            8,
            &ProofOptions::default_test_options(),
        )
        .unwrap();
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
        let bundle = prove_continuation(
            &elf_bytes,
            &input,
            &[],
            2,
            &ProofOptions::default_test_options(),
        )
        .unwrap();
        assert!(
            bundle.num_epochs() > 1,
            "4-cycle epochs must split the run into multiple epochs"
        );
        assert!(
            bundle.num_private_input_pages > 0,
            "a program that reads private input must have a private-input page in the global proof"
        );

        // The serialized bundle must carry no raw private bytes: it survives an rkyv
        // round-trip and still verifies using ONLY the bundle + ELF (no private input
        // is passed to `verify_continuation`).
        let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&bundle).unwrap();
        let restored: ContinuationProof =
            rkyv::from_bytes::<_, rkyv::rancor::Error>(&bytes).unwrap();
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
        let mut bundle = prove_continuation(
            &elf_bytes,
            &input,
            &[],
            2,
            &ProofOptions::default_test_options(),
        )
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
        let mut bundle = prove_continuation(
            &elf_bytes,
            &input,
            &[],
            2,
            &ProofOptions::default_test_options(),
        )
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

        // private_input_page_count: wire format is `[len:4][data][pad8][count:4][pad:4]`
        // plus 32-byte hint slots; the region is page-aligned. The always-written
        // 8-byte arena header shifts the old boundaries by +8 bytes.
        assert_eq!(page::private_input_page_count(&[], &[]), 0);
        assert_eq!(page::private_input_page_count(&[0u8; 16], &[]), 1);
        // 4-byte prefix + (page_size - 12) data pads to page_size - 8, plus the
        // 8-byte header exactly fills one page.
        assert_eq!(
            page::private_input_page_count(&vec![0u8; page::DEFAULT_PAGE_SIZE - 12], &[]),
            1
        );
        // One more byte pads up to page_size and the header spills into a second page.
        assert_eq!(
            page::private_input_page_count(&vec![0u8; page::DEFAULT_PAGE_SIZE - 11], &[]),
            2
        );
        // The old single-page boundary (page_size - 4) now needs two pages: the
        // data section alone fills the page and the +8 header spills over.
        assert_eq!(
            page::private_input_page_count(&vec![0u8; page::DEFAULT_PAGE_SIZE - 4], &[]),
            2
        );
        assert_eq!(
            page::private_input_page_count(&vec![0u8; page::DEFAULT_PAGE_SIZE - 3], &[]),
            2
        );
        // Hints extend the region: empty main + one hint = align8(4) + 8 + 32 = 48 bytes.
        assert_eq!(page::private_input_page_count(&[], &[[0u8; 32]]), 1);
        // The arena alone can push the span past a page boundary.
        let hints = vec![[0u8; 32]; page::DEFAULT_PAGE_SIZE / 32];
        assert_eq!(page::private_input_page_count(&[], &hints), 2);
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
    // span so we don't allocate a 512 MiB test input).
    #[test]
    fn test_max_private_input_pages_is_tight() {
        use executor::vm::memory::{MAX_PRIVATE_INPUT_SIZE, PRIVATE_INPUT_LENGTH_PREFIX_BYTES};
        let page_size = page::DEFAULT_PAGE_SIZE;
        let max = page::max_private_input_pages();

        // (512 MiB + 4-byte prefix) / 256 KiB page = 2049 pages (2048 full data pages plus
        // the one page the length prefix spills into). Pinned so a size/page change is caught.
        assert_eq!(max, 2049);

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
        let mut bundle = prove_continuation(
            &elf_bytes,
            &[],
            &[],
            3,
            &ProofOptions::default_test_options(),
        )
        .unwrap();
        bundle.num_private_input_pages = page::max_private_input_pages() + 1;
        assert!(matches!(
            verify_continuation(&elf_bytes, &bundle, &ProofOptions::default_test_options()),
            Err(Error::InvalidTableCounts(_))
        ));
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

        let bundle = prove_continuation(
            &elf_bytes,
            &input,
            &[],
            4,
            &ProofOptions::default_test_options(),
        )
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
        let mut bundle = prove_continuation(
            &elf_bytes,
            &[],
            &[],
            3,
            &ProofOptions::default_test_options(),
        )
        .unwrap();
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
        let mut bundle = prove_continuation(
            &elf_bytes,
            &[],
            &[],
            3,
            &ProofOptions::default_test_options(),
        )
        .unwrap();
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
        let mut bundle = prove_continuation(
            &elf_bytes,
            &[],
            &[],
            3,
            &ProofOptions::default_test_options(),
        )
        .unwrap();
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

    // Negative: corrupting an epoch's claimed L2G table root must be rejected. This
    // tamper is caught by `verify_epoch`'s own root-consistency check (the epoch's
    // claimed `l2g_root` no longer matches what its own proof committed) before the
    // cross-epoch `verify_l2g_commitment_binding_view` ever runs — see
    // `test_split_verify_rejects_global_proof_from_a_different_run` for that.
    #[test]
    fn test_split_verify_rejects_tampered_l2g_root() {
        let _ = env_logger::builder().is_test(true).try_init();
        let elf_bytes = asm_elf_bytes("all_loadstore_32");
        let mut bundle = prove_continuation(
            &elf_bytes,
            &[],
            &[],
            3,
            &ProofOptions::default_test_options(),
        )
        .unwrap();
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

    // Same tamper as `test_split_verify_rejects_tampered_l2g_root`, but through the
    // zero-copy blob path (`verify_continuation_and_attest`) rather than
    // `verify_continuation`. Guards the archived path's per-epoch root check against
    // the same corruption the owned path already catches.
    #[test]
    fn test_continuation_blob_rejects_tampered_l2g_root() {
        let _ = env_logger::builder().is_test(true).try_init();
        let elf_bytes = asm_elf_bytes("all_loadstore_32");
        let mut bundle = prove_continuation(
            &elf_bytes,
            &[],
            &[],
            3,
            &crate::recursion::MIN_PROOF_OPTIONS,
        )
        .unwrap();
        assert!(
            bundle.epochs.len() >= 2,
            "need multiple epochs to exercise the binding"
        );
        bundle.epochs[0].l2g_root[0] ^= 0xFF;

        let blob = crate::recursion::encode_continuation_guest_input(
            bundle,
            &elf_bytes,
            &crate::recursion::MIN_PROOF_OPTIONS,
        )
        .expect("encode_continuation_guest_input failed");

        let result = crate::recursion::verify_continuation_and_attest(
            &blob,
            &crate::recursion::MIN_PROOF_OPTIONS,
        )
        .expect("verify_continuation_and_attest errored");
        assert!(
            result.is_none(),
            "a tampered l2g_root must be rejected over the archived blob path too"
        );
    }

    // Negative: `verify_l2g_commitment_binding_view`'s own reject branch, which the two
    // tests above don't reach (they're caught earlier by `verify_epoch`'s per-epoch root
    // check). Two bundles proved from the same ELF/epoch size with different
    // same-length private inputs share every shape value (`n`, `table_counts`,
    // `touched_page_bases`, `num_private_input_pages`) but commit different actual L2G
    // data, so splicing one's `global` proof onto the other's epochs leaves every
    // per-epoch check and `verify_global`'s own `multi_verify` passing (each half is
    // independently valid for that exact shape) while the per-epoch claimed roots no
    // longer match what the spliced-in global proof's L2G sub-tables actually commit.
    #[test]
    fn test_split_verify_rejects_global_proof_from_a_different_run() {
        let _ = env_logger::builder().is_test(true).try_init();
        let elf_bytes = asm_elf_bytes("test_private_input_xpage");
        let opts = ProofOptions::default_test_options();

        let input_a: Vec<u8> = (0u8..16).collect();
        let input_b: Vec<u8> = (0u8..16).map(|b| b ^ 0xFF).collect();

        let mut bundle_a = prove_continuation(&elf_bytes, &input_a, &[], 2, &opts).unwrap();
        let bundle_b = prove_continuation(&elf_bytes, &input_b, &[], 2, &opts).unwrap();
        assert!(
            verify_continuation(&elf_bytes, &bundle_a, &opts)
                .unwrap()
                .is_some(),
            "bundle_a must verify standalone before splicing"
        );
        assert!(
            verify_continuation(&elf_bytes, &bundle_b, &opts)
                .unwrap()
                .is_some(),
            "bundle_b must verify standalone before splicing"
        );
        assert_eq!(
            bundle_a.epochs.len(),
            bundle_b.epochs.len(),
            "same ELF/epoch size/input length must yield the same epoch split"
        );
        assert_eq!(
            bundle_a.touched_page_bases, bundle_b.touched_page_bases,
            "same-length private inputs must touch the same pages"
        );
        assert_ne!(
            bundle_a.epochs[0].l2g_root, bundle_b.epochs[0].l2g_root,
            "different private-input bytes must commit different L2G data"
        );

        bundle_a.global = bundle_b.global;

        assert!(
            verify_continuation(&elf_bytes, &bundle_a, &opts)
                .unwrap()
                .is_none(),
            "a global proof spliced in from a different run must be rejected"
        );
    }

    // Same construction as `test_split_verify_rejects_global_proof_from_a_different_run`,
    // but through the zero-copy blob path — guards
    // `verify_l2g_commitment_binding_view`'s archived call site.
    #[test]
    fn test_continuation_blob_rejects_global_proof_from_a_different_run() {
        let _ = env_logger::builder().is_test(true).try_init();
        let elf_bytes = asm_elf_bytes("test_private_input_xpage");
        let opts = crate::recursion::MIN_PROOF_OPTIONS;

        let input_a: Vec<u8> = (0u8..16).collect();
        let input_b: Vec<u8> = (0u8..16).map(|b| b ^ 0xFF).collect();

        let mut bundle_a = prove_continuation(&elf_bytes, &input_a, &[], 2, &opts).unwrap();
        let bundle_b = prove_continuation(&elf_bytes, &input_b, &[], 2, &opts).unwrap();
        assert_eq!(bundle_a.epochs.len(), bundle_b.epochs.len());
        assert_eq!(bundle_a.touched_page_bases, bundle_b.touched_page_bases);
        assert_ne!(bundle_a.epochs[0].l2g_root, bundle_b.epochs[0].l2g_root);

        bundle_a.global = bundle_b.global;

        let blob = crate::recursion::encode_continuation_guest_input(bundle_a, &elf_bytes, &opts)
            .expect("encode_continuation_guest_input failed");
        let result = crate::recursion::verify_continuation_and_attest(&blob, &opts)
            .expect("verify_continuation_and_attest errored");
        assert!(
            result.is_none(),
            "a global proof spliced in from a different run must be rejected over the archived blob path too"
        );
    }
}
