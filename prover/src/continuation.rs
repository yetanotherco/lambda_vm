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

use executor::elf::Elf;
use executor::vm::execution::Executor;
use math::field::element::FieldElement;
use stark::config::Commitment;
use stark::constraints::builder::{ConstraintBuilder, ConstraintSet, EmptyConstraints};
use stark::lookup::{
    AirWithBuses, AuxiliaryTraceBuildData, LazyCommitment, NullBoundaryConstraintBuilder,
};
use stark::multilinear_table::PreparedColumn;
use stark::proof::options::ProofOptions;
use stark::proof::stark::MultiProof;
use stark::proof::view::MultiProofView;
use stark::prover::IsStarkProver;
use stark::trace::TraceTable;
use stark::traits::AIR;
use stark::verifier::IsStarkVerifier;

use crate::statement::{StatementKind, absorb_continuation_global_statement, absorb_statement};
use crate::tables::local_to_global::{self, CellBoundary};
use crate::tables::page::{self, PageConfig};
use crate::tables::register;
use crate::tables::trace_builder::{
    DecodeArtifacts, Traces, build_init_page_data, build_initial_image_paged,
};
use crate::tables::types::{FE, GoldilocksExtension, GoldilocksField};
use crate::tables::{MaxRowsConfig, global_memory};
use crate::{
    Error, FIXED_TABLE_COUNT, RuntimePageRange, TableCounts, VmAirs,
    compute_expected_commit_bus_balance_view, verify_l2g_commitment_binding_view,
};

type F = GoldilocksField;
type E = GoldilocksExtension;
type AirRef<'a> = &'a dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>;

/// Fresh transcript seeded with the epoch's statement (ELF, public output, table
/// layout), `epoch_label` (its position) and `is_final` (whether it carries
/// HALT). The epoch's prove, verify, and bus-balance replay all seed via this so
/// their challenges match; the seeding pins each epoch proof to its program,
/// position and role (replay protection).
fn epoch_transcript(
    elf_bytes: &[u8],
    public_output: &[u8],
    table_counts: &TableCounts,
    runtime_page_ranges: &[RuntimePageRange],
    epoch_label: u64,
    is_final: bool,
    fri_final_poly_log_degree: u8,
) -> crate::hash_pin::BlockTranscript {
    let mut transcript = crate::hash_pin::block_transcript(&[]);
    absorb_statement(
        &mut transcript,
        StatementKind::ContinuationEpoch {
            epoch_label,
            is_final,
        },
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
) -> crate::hash_pin::BlockTranscript {
    let mut transcript = crate::hash_pin::block_transcript(&[]);
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
pub(crate) struct L2gMemoryConstraints;

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
pub(crate) fn l2g_global_air(
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
pub(crate) fn l2g_memory_air(
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
pub(crate) fn global_memory_air(
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
        return air.with_lazy_preprocessed_columns(
            page::private_page_lazy_commitment(opts),
            page::NUM_PREPROCESSED_COLS_PRIVATE,
            Arc::new(|| vec![page::offset_column()]),
        );
    }
    // The columns as well as the root: the univariate path compares the root
    // the verifier recomputes, the multilinear one has no second root and
    // compares these instead. They are PAGE's — GLOBAL_MEMORY's preprocessed
    // prefix is the same OFFSET and INIT, which is why the same commitment
    // serves both.
    // Both leaf layouts (S2): a zero-init page's one-row root is the static
    // twin, a data page's is computed on first use; a supplied root is a
    // row-pair root and never stands in for the other layout.
    let commitment = if config.init_values.is_some() {
        page::data_page_lazy_commitment(config, opts, preprocessed)
    } else {
        match preprocessed {
            Some(c) => page::zero_init_lazy_commitment_from(c, opts),
            None => {
                let options = opts.clone();
                LazyCommitment::deferred(move || page::zero_init_preprocessed_commitment(&options))
                    .with_one_row({
                        let options = opts.clone();
                        move || {
                            page::zero_init_preprocessed_commitment_for(
                                &options,
                                stark::leaf_layout::LeafLayout::Row,
                            )
                        }
                    })
            }
        }
    };
    let config = config.clone();
    air.with_lazy_preprocessed_columns(
        commitment,
        global_memory::NUM_PREPROCESSED_COLS,
        Arc::new(move || page::preprocessed_columns(&config)),
    )
}

/// The sorted, deduped set of page bases the touched cells fall on — the SINGLE source
/// of truth for which GLOBAL_MEMORY tables exist. The prover builds the committed tables
/// from this list, ships the identical list in the bundle (`ContinuationProof.touched_page_bases`),
/// and the verifier rebuilds the same tables from it. Sorted (BTreeSet order) so prover
/// and verifier iterate the identical sequence — `multi_verify` matches AIRs to sub-proofs
/// positionally. Carries page bases ONLY: no cell values, so private-input bytes never
/// enter the bundle (unlike the full `CellBoundary`, whose `init.value` is a private byte).
pub(crate) fn touched_page_bases(boundaries: &[Arc<Vec<CellBoundary>>]) -> Vec<u64> {
    boundaries
        .iter()
        .flat_map(|epoch| epoch.iter())
        .map(|b| page::page_base_for_address(b.address))
        .collect::<std::collections::BTreeSet<u64>>()
        .into_iter()
        .collect()
}

/// The cross-epoch page census of an execution, WITHOUT proving anything: how
/// many epochs the guest runs, which page bases the epoch-crossing cells fall
/// on, and how many cells each page contributes. This is the shape data the
/// aggregation layer's census consumes — the global memory proof commits one
/// GLOBAL_MEMORY table per touched page, sized by that page's crossing cells,
/// and the aggregator's cost per page follows the table height — so pricing
/// the aggregation program requires exactly this and nothing heavier.
#[cfg(test)]
pub(crate) struct BlockPageCensus {
    pub num_epochs: usize,
    /// Sorted, deduped — [`touched_page_bases`]' own order.
    pub touched_page_bases: Vec<u64>,
    pub num_private_input_pages: usize,
    /// Epoch-crossing cells per touched page, in `touched_page_bases` order —
    /// the GLOBAL_MEMORY table height driver (rows before padding).
    pub page_cells: Vec<usize>,
    /// Cells in each epoch's L2G table (rows before padding), epoch order.
    pub l2g_cells: Vec<usize>,
}

/// Runs the guest to completion and collects [`BlockPageCensus`] — the
/// producer loop's sequential-critical half (execute, collect, boundary,
/// image carry) with every prove and trace build omitted.
#[cfg(test)]
pub(crate) fn block_page_census(
    elf_bytes: &[u8],
    private_inputs: &[u8],
    epoch_size_log2: u32,
) -> Result<BlockPageCensus, Error> {
    let epoch_size = 1usize
        .checked_shl(epoch_size_log2)
        .ok_or_else(|| Error::InvalidContinuationEpochSize("epoch size overflow".to_string()))?;
    let elf = Elf::load(elf_bytes).map_err(|e| Error::ElfLoad(format!("{e}")))?;
    let mut executor = Executor::new(&elf, private_inputs.to_vec())
        .map_err(|e| Error::Execution(format!("{e}")))?;
    let decode_artifacts = DecodeArtifacts::from_elf(&elf)?;
    let mut image = build_initial_image_paged(&elf, private_inputs);
    let mut provenance =
        local_to_global::genesis_provenance(image.iter().map(|(a, v)| (a, v as u64)));

    let mut boundaries: Vec<Arc<Vec<CellBoundary>>> = Vec::new();
    let mut prev_fini: Option<Vec<u32>> = None;
    let mut index: u64 = 0;
    loop {
        if executor.pc() == 0 {
            break;
        }
        if index >= local_to_global::MAX_EPOCHS {
            return Err(Error::InvalidContinuationEpochSize(
                "execution exceeds the IsB20 cross-epoch ordering range".to_string(),
            ));
        }
        let register_init: Vec<u32> = match (index, prev_fini.take()) {
            (0, _) => register::register_init_from_entry_point(elf.entry_point),
            (_, Some(fini)) => fini,
            (_, None) => {
                return Err(Error::ContinuationInvariant(
                    "previous epoch final registers are missing after the first epoch".to_string(),
                ));
            }
        };
        let logs = match executor
            .resume_with_limit(epoch_size)
            .map_err(|e| Error::Execution(format!("{e}")))?
        {
            Some(logs) => logs.to_vec(),
            None => break,
        };
        let is_final = executor.pc() == 0;
        let collected =
            Traces::collect_epoch(&decode_artifacts, &image, &register_init, &logs, is_final)?;
        let boundary = Arc::new(local_to_global::epoch_boundary(
            &mut provenance,
            local_to_global::epoch_label(index),
            &collected.touched_memory_cells(),
        ));
        prev_fini = Some(collected.register_fini(&register_init));
        for cell in boundary.iter() {
            image.set(cell.address, (cell.fini.value & 0xFF) as u8);
        }
        boundaries.push(boundary);
        if is_final {
            break;
        }
        index += 1;
    }

    let bases = touched_page_bases(&boundaries);
    let mut per_page: std::collections::BTreeMap<u64, usize> =
        bases.iter().map(|&b| (b, 0)).collect();
    for b in boundaries.iter().flat_map(|epoch| epoch.iter()) {
        *per_page
            .get_mut(&page::page_base_for_address(b.address))
            .expect("every crossing cell's page is in the census") += 1;
    }
    Ok(BlockPageCensus {
        num_epochs: boundaries.len(),
        page_cells: bases.iter().map(|b| per_page[b]).collect(),
        touched_page_bases: bases,
        num_private_input_pages: page::private_input_page_count(private_inputs),
        l2g_cells: boundaries.iter().map(|b| b.len()).collect(),
    })
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
pub(crate) fn global_memory_configs(
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
pub(crate) fn global_memory_configs_from_init_page_data(
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

// ===========================================================================
// THE GENESIS-STACK SPLIT: which cross-epoch pages a PREPARED OPENING carries,
// and which keep the sparse closed form.
//
// ⛔ IT LIVES HERE AND NOT IN `crate::lfm`, AND THAT IS A LAYERING FACT, NOT A
// PREFERENCE. Three parties evaluate this rule — the cross-epoch PROVER, its
// VERIFIER, and the in-guest EMITTER — and they must reach the SAME answer from
// data all three hold, because the stack's root is absorbed in the roots block:
// two sides that disagree about the set commit different stacks, absorb
// different roots, and diverge at `z` with nothing in the failure naming the
// cause. `crate::lfm` depends on this module and not the reverse, so a rule the
// prover and verifier evaluate cannot live there or call anything that does.
//
// # Why there is a split at all
//
// A page's INIT column is its genesis bytes, zero past `init_values.len()`, and
// the in-guest verifier discharges it by the sparse form — one term per NONZERO
// entry, so [`sparse_leg_rows`]. Free for a zero-init page, cheap for a page
// holding a few constants, and ruinous for a page that is genuinely full.
//
// Measured on block 25368371 (ELF sha256
// `8f826601776d4085cbb6fbf0302fe8d8d5d1be7940ac1aaca24899c6244ec80a`, 3,948,504
// B; input `573004e62e3680a00d3cdbae19dc4897e2ec60d6ec0c1d05d9ef118cb8aef17f`,
// 1,110,183 B; `LAMBDA_VM_MAX_ROWS_LOG2=21`), the run touches 35 pages — 30
// genesis and 5 private — and the genesis bytes are CONCENTRATED: `0x0` carries
// 116,692 nonzero of 262,144, `0x40000` 229,290, `0x280000` 223,380, and the
// other 27 are all-zero. Total 10,249,056 rows for INIT alone, against a 2-4 M
// band for the WHOLE cross-epoch program. Three pages of thirty carry all of it.
//
// ⛔ THE PRIVATE-INPUT PAGES ARE EXCLUDED FIRST, AND NOT AS AN OPTIMISATION.
// [`global_memory_configs_from_init_page_data`] is called with
// `include_private_genesis = true` by the PROVER and `false` by the VERIFIER
// (through [`global_memory_configs`]), so a private-input page carries
// `init_values = Some(<the private bytes>)` on one side and `Some(vec![])` on
// the other. A threshold evaluated over every config would read a different
// nonzero count for those pages on the two sides and select a different set.
// The filter is on `is_private_input`, which both derive from `page_base` and
// `num_private_input_pages` — two values the cross-epoch statement already
// binds — and the reason is not "private pages are not worth stacking" but that
// a private page presents no INIT column to settle at all.
//
// # The rule, in two parts
//
// 1. **Is this page a CANDIDATE?** Its sparse leg [`sparse_leg_rows`] against
//    what carrying it would ADD to the prepared leg,
//    [`marginal_stacked_rows`], evaluated at the fixed height
//    [`fixed_stack_vars`]. On the block that is `18 + 18*S > 111`, so
//    `S >= 6`: a page with six nonzero genesis bytes is already worth more
//    carried than kept, and the 27 all-zero pages are not. ⚠ BOTH SIDES MOVE
//    WITH THE RUN'S BRACKET — the marginal is a form evaluated at the height
//    this run's genesis page count puts the stack at, not a number quoted
//    once, and `τ` is 5 below nine genesis pages and 6 from there.
// 2. **Is the CHAIN worth paying for?** The candidates' total savings against
//    [`PREPARED_LEG_ROWS`], evaluated ONCE on the whole set. If it refuses,
//    every candidate stays sparse — there is no partial answer, because a
//    subset still pays the whole chain.
//
// ⛔ WHY IT TAKES TWO PARTS AND NOT ONE. The leg's cost is a fixed chain plus a
// per-page marginal, and no per-page rule can charge a fixed cost correctly:
// charge each page the whole chain (the rule this replaces) and two pages that
// would gladly share one are both refused; charge none of them and a lone page
// worth five rows drags in a chain costing 175,066. Splitting the question is
// what lets each term be charged to the thing that actually causes it.
//
// ⛔⛔ AND ALL THREE INPUTS ARE DATA ALL THREE PARTIES HOLD. Every `S` is a
// nonzero count over a genesis page's `init_values`, and `P_touched` is the
// genesis page count — both read off the very page configs the prover and the
// verifier each build from the ELF and the touched page list, and the emitter
// reads through the plan. Nothing in the rule reads a proof, a private byte, or
// the order the pages were considered in. `the_prover_and_verifier_views_of_a_
// private_page_plan_alike` is the assertion of it.
// ===========================================================================

/// A page table's height in variables.
///
/// ⚠ DERIVED FROM [`page::DEFAULT_PAGE_SIZE`] AND NOT FROM A PROOF. Every
/// GLOBAL_MEMORY table is one page tall and [`page::preprocessed_columns`]
/// builds columns of exactly that many rows, so this is a property of the page
/// and not a claim a bundle gets to make.
pub(crate) const PAGE_NUM_VARS: usize = page::DEFAULT_PAGE_SIZE.trailing_zeros() as usize;

/// How many preprocessed columns a genesis page presents: `[OFFSET, INIT]`.
///
/// ⚠ THE STACK TAKES BOTH, AND THAT IS A PRICE PAID TO A CONTRACT. Only INIT is
/// worth an opening — OFFSET is the identity ramp, whose extension is
/// `sum_k 2^k * r_k` and costs `num_vars - 1` rows to check outright. But
/// `check_preprocessed` skips a PREFIX of a table's preprocessed columns, and
/// `{1}` is not one, so settling INIT alone is inexpressible. Taking `{0, 1}`
/// keeps that contract unchanged at a cost of about 230 rows on the block: 51
/// lost ramps against 282 for three more stacked columns, on a hybrid costing
/// order `10^5`.
pub(crate) const PAGE_PREPROCESSED_COLUMNS: usize = 2;

/// The rows the in-guest verifier spends discharging one page's INIT by the
/// sparse closed form: `MLE(r) = sum_{v_i != 0} v_i * eq(r, i)`, which is
/// `num_vars` complements hoisted plus `num_vars` rows per nonzero entry.
///
/// An all-zero page costs `num_vars` — the interned zero — and not nothing,
/// which is why the block's 27 zero pages are 486 rows and not 0.
pub(crate) fn sparse_leg_rows(num_vars: usize, nonzero: usize) -> usize {
    num_vars + num_vars * nonzero
}

/// PART 2's term: the rows the prepared leg costs ONCE, however many pages it
/// carries — the stacked chain itself.
///
/// ⛔⛔ THIS IS AN UPPER BOUND, NOT THE STACK'S COST, and calling it the cost
/// would be a number answering a retired question. It is V1i's card-free sizing
/// of a stacked family polynomial at **24 variables over 35 columns**; the stack
/// this code actually builds is `2 * n_dense` columns at
/// `PAGE_NUM_VARS + ceil(log2(2 * n_dense))` variables — six columns at 21 on
/// the block. Fewer variables means fewer rounds means fewer rows, so the real
/// chain is CHEAPER than this and the rule is conservative in the direction that
/// matters: the carried set must be worth more than the chain could possibly
/// cost before anything joins it.
///
/// ⚠ IT PAYS FOR THE CHAIN AND FOR NOTHING ELSE. What a page adds to the leg by
/// being carried is [`marginal_stacked_rows`], and that is charged to the page
/// in part 1. Charging this constant per page — the rule this replaces — asked
/// each page to pay for a chain all of them share.
///
/// ⚠ CONFIGURATION IS PART OF THE NUMBER. Chain rows move with blowup, folding
/// and the query count. It is a constant here rather than a call because of the
/// layering note above, and `lfm::whir_chain_tests` is where it is PINNED
/// against `chain_shape_rows` at the block's shape — the assertion that keeps
/// the word "bound" honest.
pub(crate) const PREPARED_LEG_ROWS: usize = 175_066;

/// The stack height the rule CHARGES AT, which is deliberately not the height
/// the stack will have: `num_vars + ceil(log2(2 * genesis_pages))`.
///
/// ⛔⛔ WHY A FIXED HEIGHT AT ALL. The real stack stands at
/// `num_vars + ceil(log2(2 * n_dense))`, and `n_dense` is the ANSWER — a
/// marginal evaluated there would be defined in terms of its own output, and
/// prover, verifier and emitter would each have to guess where the recursion
/// settled. Evaluating at every touched genesis page instead is a FIXED
/// quantity all three read off the same page configs, before any routing
/// decision is made. `n_dense <= genesis_pages`, so it is at or above the real
/// height.
///
/// ★ AND THIS IS THE BRACKET THE MARGINAL IS EVALUATED AT, per run.
/// [`marginal_stacked_rows`] reads it, so a run standing taller is charged MORE
/// rather than charged too little — which is the whole gain over the literal
/// this replaces. The block's thirty pages give 24 ([`BLOCK_STACK_VARS`]); a
/// lone page gives 19 and is charged 101 rather than the block's 111.
pub(crate) const fn fixed_stack_vars(num_vars: usize, genesis_pages: usize) -> usize {
    num_vars + (2 * genesis_pages).next_power_of_two().trailing_zeros() as usize
}

/// The bracket the block's thirty genesis pages put the stack at, and the
/// bracket every consequence quoted in these docs is quoted AT.
///
/// ⛔⛔ A MARGINAL HAS NO MEANING WITHOUT ITS BRACKET, which is exactly what
/// the literal this replaces could not carry. [`marginal_stacked_rows`] is 101
/// at a one-page stack, 111 here and 113 one bracket up; `τ` is 5 below 23
/// variables and 6 from there. A number quoted for "the rule" is quoted at this
/// bracket unless it names another, and a run standing elsewhere is charged its
/// own.
///
/// ⚠ IT IS A READING OF [`fixed_stack_vars`], NOT AN INPUT TO ANYTHING. No
/// production path reads it — `genesis_stack_plan` computes the bracket from
/// the configs it was handed — and `the_form_and_the_bracket_it_is_quoted_at`
/// asserts the two agree.
pub(crate) const BLOCK_STACK_VARS: usize = 24;

/// The rows a stacked column costs OUTSIDE `weight_at_rows`: its
/// `absorb_unpack_rows()`, its share of `challenge_powers_rows(columns)` and its
/// entry in `columns_of(..)` — one row each.
///
/// ⚠ A LITERAL RATHER THAN THREE CALLS, for the layering reason
/// [`PREPARED_LEG_ROWS`] gives: this module does not depend on `lfm`'s cost
/// forms, and the pin is what ties this number to them.
pub(crate) const STACKED_ROWS_PER_COLUMN: usize = 3;

/// The most rows one more carried page can add to the THREADED sponge — the one
/// term of the marginal with no closed form, and a PROVEN BOUND rather than a
/// measurement.
///
/// `stacked_verify_cost` absorbs `COORDINATES_PER_EXT` = 3 coordinates per
/// column into ONE buffer and squeezes once, at
/// `squeeze_rows(f) = f.div_ceil(4) + f.div_ceil(8) + 1`. A page is
/// [`PAGE_PREPROCESSED_COLUMNS`] columns, so six more felts, and the `+ 1` is
/// per-squeeze and differences away:
///
/// ```text
/// Δ ceil(f/4)  over  f -> f + 6   ∈ {1, 2}
/// Δ ceil(f/8)  over  f -> f + 6   ∈ {0, 1}
/// ⇒ s ∈ {1, 2, 3}, and the bound is 3
/// ```
///
/// ⚠ THE BOUND IS EXACT AND THE VALUE IS NOT. `s` cycles `2, 3, 1, 3` with
/// period 4 as pages are added, so the marginal at ONE bracket is a spread of
/// three values and the form charges the dearest. Six felts divide neither 4
/// nor 8, which is the whole of why there is a spread to bound.
pub(crate) const MAX_SPONGE_MARGINAL: usize = 3;

/// PART 1's term: the rows ONE MORE carried page adds to the prepared leg, at a
/// stack standing at `n_fixed` variables.
///
/// ⛔⛔ THREE SHAPES WERE TRIED AND THE ARITHMETIC WAS NEVER THE PROBLEM:
///
/// 1. a FORMULA, wrong because it omitted terms — two careful readings of it
///    each found a term the other's lacked, one the three rows per column paid
///    outside `weight_at_rows`, the other a shared `Sub` that differences to
///    ZERO;
/// 2. a LITERAL, wrong because the quantity is not constant — it took three
///    values at one bracket and a different three one bracket up, and a literal
///    carries its bracket only in prose, where prose goes stale;
/// 3. a FORMULA again, whose one irreducible term is a proven BOUND.
///
/// ★ THE LESSON, which is the part worth carrying to the next cost like this:
/// when a cost splits into a deterministic part and a bounded nondeterministic
/// one, WRITE BOTH. A literal hides the split; a form that omits the bound
/// looks complete while being wrong.
///
/// **The terms**, each by its shape (`lfm::whir_stacked::weight_at_rows`,
/// `lfm::whir_stacked::stacked_verify_cost`):
///
/// 1. `5 * num_vars` — one `eq` over the page's own variables plus the `MulAdd`
///    that joins its group: `4n + (n - 1) + 1`. A page's two preprocessed
///    columns settle at ONE point, its own table's, so they are one group and
///    pay one `eq` between them.
/// 2. [`PAGE_PREPROCESSED_COLUMNS`] prefix indicators at
///    `max(n_fixed - num_vars, 1)` rows apiece. ⚠ THIS is the term that makes
///    the marginal a function of the bracket rather than a number: it grows two
///    rows per prefix bit, so the same page costs 101 in a one-page stack, 111
///    in the block's and 113 one bracket above it.
/// 3. [`STACKED_ROWS_PER_COLUMN`] per column, the rows paid outside
///    `weight_at_rows`.
/// 4. [`MAX_SPONGE_MARGINAL`], the threaded sponge's contribution — the term
///    with no closed form, bounded rather than computed.
///
/// ⛔ A FIFTH TERM IS NOT IN THE MARGINAL AT ALL: the shared `Sub`
/// `weight_at_rows` emits once per prefix POSITION any column of the polynomial
/// reads as a zero bit. It is per-POLYNOMIAL, so differenced across one more
/// page it contributes ZERO. It was carried here for a while as "≤ 1,
/// amortised"; differencing is what exposed that as a fudge, and it is why the
/// `+ 6` sits beside a 102 and not a 103.
///
/// ★ SET-INDEPENDENCE SURVIVES THE CHANGE, which is the property the single
/// literal was protecting. `n_fixed` is [`fixed_stack_vars`] of the GENESIS
/// PAGE COUNT, a quantity prover, verifier and emitter each derive from the ELF
/// and the touched page list before any routing decision exists. It is NOT
/// `n_dense`, which is this rule's own output — see [`fixed_stack_vars`].
///
/// ⚠ IT IS AN UPPER BOUND AT EVERY BRACKET, deliberately: part 1 asks a page to
/// beat the dearest position it could occupy, and part 2 understates the set's
/// savings, so both parts are conservative. The block is charged above its own
/// stack twice over — its three dense pages are six columns standing at
/// `n_stack` 21 with a prefix of 3, so its true weight marginal is 96 against
/// the 102 charged here, before `s`.
///
/// The pin
/// `lfm::whir_stacked_tests::the_marginal_the_routing_rule_charges_is_the_one_the_stack_bills`
/// is where this form is read off `stacked_verify_cost` rather than asserted
/// against itself.
pub(crate) const fn marginal_stacked_rows(num_vars: usize, n_fixed: usize) -> usize {
    // `max(n_fixed - num_vars, 1)` without `Ord::max`, which is not const: a
    // column filling the whole stack has no indicator and pays only its fold.
    let prefix = if n_fixed > num_vars + 1 {
        n_fixed - num_vars
    } else {
        1
    };
    5 * num_vars
        + PAGE_PREPROCESSED_COLUMNS * prefix
        + PAGE_PREPROCESSED_COLUMNS * STACKED_ROWS_PER_COLUMN
        + MAX_SPONGE_MARGINAL
}
/// PART 1: whether a page is even a CANDIDATE — whether keeping it sparse costs
/// more than carrying it would add to the leg.
///
/// ★ SET-INDEPENDENT, AND THE `n_fixed` ARGUMENT IS WHY IT STAYS SO. Both
/// sides of the comparison are this page's own nonzero count and a form
/// evaluated at the GENESIS PAGE COUNT's bracket — never at `n_dense`, which is
/// the answer. Nothing here depends on which other pages qualified or on the
/// order they were considered in, and prover, verifier and emitter reach the
/// same bracket from the same page configs.
pub(crate) fn is_candidate(num_vars: usize, n_fixed: usize, nonzero: usize) -> bool {
    nonzero >= candidate_threshold_entries(num_vars, n_fixed)
}

/// The fewest nonzero entries that make a page a candidate — `τ`, the number
/// part 1 is quoted by, and the form [`is_candidate`] is decided on.
///
/// The least `S` with
/// `num_vars + num_vars * S > marginal_stacked_rows(num_vars, n_fixed)`,
/// which is floor plus one and NOT `div_ceil`: were the division exact,
/// `div_ceil` would hand back an `S` whose leg EQUALS the marginal rather than
/// exceeding it.
///
/// ⚠ `τ` IS DERIVED FROM THE MARGINAL AND IS NOT ITSELF A RULED QUANTITY. It
/// is 5 at this literal, holding at 5 for a marginal in `[90, 107]`, at 6 on
/// `[108, 125]` and at 7 on `[126, 143]` — so the form's 111 will make it 6.
/// The only page that difference reclassifies is one carrying five nonzero
/// genesis bytes,
/// and neither the block (0, or 116,692 and up) nor any fixture (112, 65,652)
/// sits there; `the_ruled_boundaries_hold_for_every_marginal_in_band` is the
/// reading of that rather than the assurance.
///
/// ⚠ ONE FORM, NOT TWO. Part 1 could as easily be the comparison
/// `sparse_leg_rows(..) > marginal_stacked_rows(..)`, and the two agree for
/// every `S` — `the_threshold_and_its_entry_count_are_one_rule` is the reading
/// of that rather than the algebra taken on trust. Deciding through `τ` keeps
/// the number a reader is quoted and the number the code branches on one
/// object.
pub(crate) fn candidate_threshold_entries(num_vars: usize, n_fixed: usize) -> usize {
    if num_vars == 0 {
        // A page of no rows has no sparse leg to save, so nothing qualifies.
        return usize::MAX;
    }
    marginal_stacked_rows(num_vars, n_fixed).saturating_sub(num_vars) / num_vars + 1
}

/// What carrying one candidate SAVES: its sparse leg, less what it adds to the
/// prepared one. Zero for a page that is not a candidate.
pub(crate) fn page_savings(num_vars: usize, n_fixed: usize, nonzero: usize) -> usize {
    sparse_leg_rows(num_vars, nonzero).saturating_sub(marginal_stacked_rows(num_vars, n_fixed))
}

/// PART 2: whether the candidate set is worth paying the chain for at all.
///
/// ⛔ EVALUATED ONCE, ON THE WHOLE SET, AND THAT IS WHY IT IS A SEPARATE PART.
/// The chain is paid once however many pages ride it, so no per-page rule can
/// see it: a per-page test with a fixed-cost term in it charges every page for
/// the same chain and refuses two pages that would gladly share one. That is
/// the case the single-page rule got wrong — two pages of 5,000 entries apiece
/// were each worth less than the chain and jointly worth more, and both stayed
/// sparse at 180,036 rows.
///
/// If this refuses, EVERY candidate stays sparse. There is no partial answer:
/// carrying a subset still pays the whole chain.
pub(crate) fn chain_is_paid(total_savings: usize) -> bool {
    total_savings > PREPARED_LEG_ROWS
}

/// The most nonzero entries a page this rule leaves SPARSE can carry.
///
/// ⛔⛔ THE QUANTITY THE SPARSE-LEG CAP MUST CLEAR, and it is a property of the
/// two-part rule rather than of one page. A page is left sparse only when part
/// 2 refused, so the whole candidate set saved at most [`PREPARED_LEG_ROWS`];
/// the total is at least this page's own savings, so this page saved at most
/// that too. Invert [`page_savings`]:
///
/// ```text
/// num_vars + num_vars * S - MARGINAL <= PREPARED_LEG_ROWS
/// S <= (PREPARED_LEG_ROWS + MARGINAL - num_vars) / num_vars
/// ```
///
/// ⚠ IT BOUNDS A NON-CANDIDATE TOO, and by much more room: a page failing part
/// 1 has a sparse leg no larger than the marginal, which is three orders under
/// this. So the one number covers both ways a page can end up sparse.
///
/// ⛔ AND IT MOVES WITH THE BRACKET — one entry per `num_vars` rows of
/// marginal. At the block's 24 variables it is **9,731**, so the pair is
/// `(9,731 sparse, 9,732 dense)`, holding for a marginal in `[110, 127]`; at a
/// LONE page's 19 the marginal is 101 and the pair is `(9,730, 9,731)`, on
/// `[92, 109]`. ⚠ SO THE PAIR IS A PROPERTY OF THE RUN AND NOT OF THE RULE, and
/// no test hard-codes one: they assert this function and `+ 1` at a named
/// bracket, and the band assertion names the pair when it moves. The
/// cap-overlap conclusion is untouched at any of these values — the margin to
/// `MAX_SPARSE_INIT_ENTRIES` is six-fold.
pub(crate) fn densest_sparse_entries(num_vars: usize, n_fixed: usize) -> usize {
    if num_vars == 0 {
        return 0;
    }
    (PREPARED_LEG_ROWS + marginal_stacked_rows(num_vars, n_fixed)).saturating_sub(num_vars)
        / num_vars
}

/// How many of the pages are GENESIS pages — the count `n_fixed` is taken at.
///
/// ⚠ THE PRIVATE FILTER IS APPLIED HERE TOO, and for the same reason it is
/// applied in the plan: the prover's and the verifier's views of a private page
/// differ in their bytes, and a count that included them would still agree, but
/// a rule that read their bytes would not. Counting only what presents an INIT
/// column keeps every input to the routing decision one both sides derive from
/// the ELF.
pub(crate) fn genesis_page_count(configs: &[PageConfig]) -> usize {
    configs.iter().filter(|c| !c.is_private_input).count()
}

/// How many of a page's genesis bytes are nonzero — the only quantity the
/// threshold reads.
///
/// `init_values` is not padded to the page, so every offset at or past its
/// length is zero and costs nothing; a `None` page is zero to the last byte.
pub(crate) fn nonzero_entries(config: &PageConfig) -> usize {
    config
        .init_values
        .as_ref()
        .map(|values| values.iter().filter(|&&b| b != 0).count())
        .unwrap_or(0)
}

/// One page of the cross-epoch page family, as the routing decision sees it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PageRoute {
    /// Its index in the AIR set, bookends included — the index a
    /// `PreparedColumn` names.
    pub table: usize,
    pub page_base: u64,
    pub nonzero: usize,
    /// `false` for a private-input page, which presents no INIT column at all.
    pub has_init: bool,
    /// PART 1's answer: keeping this page sparse costs more than carrying it
    /// would add to the leg.
    ///
    /// ⚠ RECORDED SEPARATELY FROM `dense` ON PURPOSE. A candidate that is not
    /// dense is a page the chain could not be paid for, and that is a different
    /// state from a page nobody wanted — the one the fixtures are actually in.
    /// Without this field a reader meets an absence and has to infer which rule
    /// produced it.
    pub candidate: bool,
    /// Both parts' answer: the opening carries this page.
    pub dense: bool,
}

/// Which genesis pages the prepared opening carries, and what it costs the pages
/// it leaves behind.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct GenesisStackPlan {
    /// Every touched page, in the AIR set's own page order.
    pub routes: Vec<PageRoute>,
    /// Where each stacked column is settled, in stack order: both preprocessed
    /// columns of each dense page, dense pages in canonical page-base order.
    ///
    /// ⚠ THE ORDER IS A CONTRACT AND IS ASSERTED, NOT AGREED. Run `k` of the
    /// commitment settles table `k`'s claims, so a silent reorder would settle
    /// one page's values against another page's commitment while every value
    /// gate stayed green. `multilinear_table::prepared_runs` refuses a list that
    /// does not visit each table once in stack order.
    pub at: Vec<PreparedColumn>,
    /// The rows the pages NOT carried still cost by the sparse form.
    pub sparse_rows: usize,
    /// The height THIS run's stack would stand at, [`fixed_stack_vars`] of the
    /// genesis page count.
    ///
    /// ★ AND IT DECIDES THE MARGINAL. [`marginal_stacked_rows`] is evaluated
    /// here, so this field is the bracket every number in this plan is charged
    /// at — the block's 24, a lone page's 19. It is recorded rather than
    /// recomputed by readers for the reason the order in `at` is asserted: two
    /// spellings of one quantity is how they come to differ.
    pub n_fixed: usize,
    /// What the candidate set saves, the quantity part 2 weighs against
    /// [`PREPARED_LEG_ROWS`]. Nonzero even when nothing was carried: that is
    /// the reading that says the chain was refused rather than unwanted.
    pub savings: usize,
}

impl GenesisStackPlan {
    /// Whether anything is stacked at all. A run whose genesis is entirely
    /// sparse carries no prepared opening, and its cross-epoch proof is then
    /// byte-for-byte the one it was before this route existed.
    pub fn is_empty(&self) -> bool {
        self.at.is_empty()
    }

    /// The page-family indices the opening carries, in stack order.
    pub fn dense_pages(&self) -> Vec<usize> {
        self.routes
            .iter()
            .enumerate()
            .filter(|(_, r)| r.dense)
            .map(|(page, _)| page)
            .collect()
    }
}

/// The routing decision, from the page configs and where the page family starts
/// in the AIR set.
///
/// `num_bookends` is how many local-to-global tables precede the pages — the
/// cross-epoch AIR set is every bookend then every page, and `WhirGlobalAirs`
/// is the one place that order is written. The `PreparedColumn::table` indices
/// this produces are indices into THAT order, because that is the order
/// `multi_prove` and `multi_verify` match positionally.
///
/// ⛔ TWO PASSES, BECAUSE PART 2 IS A PROPERTY OF THE SET. The first decides
/// candidacy and totals the savings; only then can the second know whether the
/// chain is paid for, and therefore whether any candidate is carried. Doing it
/// in one pass would mean deciding a page's fate before the quantity that
/// decides it has been computed — which is the set-dependence this rule is
/// built to avoid, wearing the other hat.
pub(crate) fn genesis_stack_plan(
    configs: &[PageConfig],
    num_bookends: usize,
    num_vars: usize,
) -> GenesisStackPlan {
    // ⚠ COMPUTED ONCE AND CONSUMED BY BOTH PASSES. This is the height THIS
    // run's stack would stand at, and [`marginal_stacked_rows`] is evaluated at
    // it, so a run standing taller is charged MORE rather than charged too
    // little. It is derived from the genesis page COUNT — never from how many
    // pages turn out to be dense, which is this function's own output.
    let n_fixed = fixed_stack_vars(num_vars, genesis_page_count(configs));

    // PASS 1 — candidacy, and what the candidates would save between them.
    let mut routes: Vec<PageRoute> = Vec::with_capacity(configs.len());
    let mut savings = 0usize;
    for (page, config) in configs.iter().enumerate() {
        // ⛔ The private filter comes FIRST and is not the threshold's business;
        // see the section header for what a threshold reading those bytes does.
        let has_init = !config.is_private_input;
        let nonzero = if has_init { nonzero_entries(config) } else { 0 };
        let candidate = has_init && is_candidate(num_vars, n_fixed, nonzero);
        if candidate {
            savings += page_savings(num_vars, n_fixed, nonzero);
        }
        routes.push(PageRoute {
            table: num_bookends + page,
            page_base: config.page_base,
            nonzero,
            has_init,
            candidate,
            dense: false,
        });
    }

    // PASS 2 — the chain, paid for the whole set or for none of it.
    let paid = chain_is_paid(savings);
    let mut at = Vec::new();
    let mut sparse_rows = 0usize;
    for route in &mut routes {
        route.dense = route.candidate && paid;
        if route.dense {
            // BOTH preprocessed columns, in index order: the prefix
            // `check_preprocessed` skips. See `PAGE_PREPROCESSED_COLUMNS`.
            at.extend(stark::multilinear_table::leading_columns(
                route.table,
                PAGE_PREPROCESSED_COLUMNS,
            ));
        } else if route.has_init {
            sparse_rows += sparse_leg_rows(num_vars, route.nonzero);
        }
    }

    GenesisStackPlan {
        routes,
        at,
        sparse_rows,
        n_fixed,
        savings,
    }
}

/// The stacked columns themselves, in stack order: each dense page's OFFSET then
/// its INIT.
///
/// ⚠ TAKEN FROM [`page::preprocessed_columns`] AND NOT REBUILT. The columns the
/// opening commits must be the columns the page table's own argument claims, and
/// the page AIR's preprocessed columns come from that function; a second
/// spelling of "the genesis bytes as a column" is how two objects with the same
/// name come to hold different values.
///
/// ⚠ AND THE ORDER MATCHES [`GenesisStackPlan::at`] BY CONSTRUCTION — both walk
/// `dense_pages()` and then the page's own preprocessed order. The assertion
/// that they agree is in `genesis_prepared_for`, where the two meet.
pub(crate) fn genesis_stack_columns(
    configs: &[PageConfig],
    plan: &GenesisStackPlan,
) -> Vec<Vec<FE>> {
    plan.dense_pages()
        .into_iter()
        .flat_map(|page| page::preprocessed_columns(&configs[page]))
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
pub(crate) struct PreparedEpoch {
    pub index: u64,
    pub register_init: Vec<u32>,
    pub label: u64,
    pub traces: Traces,
    pub boundary: Arc<Vec<CellBoundary>>,
    pub is_final: bool,
}

/// Every epoch's proving inputs, prepared in order and handed over one at a
/// time.
///
/// This is the sequential half of [`prove_continuation`]'s pipeline — execution,
/// op collection over the advancing memory image, the boundary, and the register
/// carry — run to completion for one epoch before the next. **None of it depends
/// on a proof, or on how one is made**, so it is the same work whichever
/// commitment scheme argues the epochs; what differs is only what `each` does
/// with a [`PreparedEpoch`].
///
/// One epoch is alive at a time, which is the whole point of continuations.
/// `prove_continuation` runs the same steps across three threads instead, to
/// overlap them; this one is for callers that want them in order.
///
/// Returns every epoch's boundary, which is what the cross-epoch global proof
/// is made of. They stay prover-local: a boundary carries cell values, and for
/// a private read that value is a byte of the private input.
pub(crate) fn for_each_epoch(
    elf: &Elf,
    private_inputs: &[u8],
    epoch_size_log2: u32,
    artifacts: &DecodeArtifacts,
    mut each: impl FnMut(PreparedEpoch, &[Arc<Vec<CellBoundary>>]) -> Result<(), Error>,
) -> Result<Vec<Arc<Vec<CellBoundary>>>, Error> {
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

    let mut executor = Executor::new(elf, private_inputs.to_vec())
        .map_err(|e| Error::Execution(format!("{e}")))?;
    let mut image = build_initial_image_paged(elf, private_inputs);
    let mut provenance =
        local_to_global::genesis_provenance(image.iter().map(|(a, v)| (a, v as u64)));

    let mut boundaries: Vec<Arc<Vec<CellBoundary>>> = Vec::new();
    let mut prev_fini: Option<Vec<u32>> = None;
    let mut index: u64 = 0;
    while executor.pc() != 0 {
        if index >= local_to_global::MAX_EPOCHS {
            return Err(Error::InvalidContinuationEpochSize(format!(
                "execution needs more than {} continuation epochs (the IsB20 \
                 cross-epoch ordering range); use a larger epoch size",
                local_to_global::MAX_EPOCHS
            )));
        }
        let register_init: Vec<u32> = match (index, prev_fini.take()) {
            (0, _) => register::register_init_from_entry_point(elf.entry_point),
            (_, Some(fini)) => fini,
            (_, None) => {
                return Err(Error::ContinuationInvariant(
                    "previous epoch final registers are missing after the first epoch".to_string(),
                ));
            }
        };

        // ── the producer's four stages, under `LAMBDA_VM_BASE_SPLIT=1` ──
        // The wall is opened first and closed last, so the four stages
        // PARTITION it with no gap: `execute` runs to the cycle-count check,
        // `collect` from the label to the image advance, `build` over the trace
        // tables, `handoff` over the call to `each`. Σ(stages) = wall by
        // construction — which is the point: the only thing that can break the
        // identity is a stage whose timer is missing, and that is what the
        // harness's check is for.
        let __bs_epoch = multilinear::whir_split::mark();
        let __bs_exec = multilinear::whir_split::mark();
        let logs = match executor
            .resume_with_limit(epoch_size)
            .map_err(|e| Error::Execution(format!("{e}")))?
        {
            Some(logs) => logs.to_vec(),
            None => break,
        };
        let is_final = executor.pc() == 0;
        if !is_final && logs.len() != epoch_size {
            return Err(Error::ContinuationInvariant(format!(
                "intermediate epoch ran {} cycles, expected {epoch_size}",
                logs.len()
            )));
        }
        let __bs_exec_s = multilinear::whir_split::stage_done(index, "execute", __bs_exec);

        let __bs_collect = multilinear::whir_split::mark();
        let label = local_to_global::epoch_label(index);
        let collected = Traces::collect_epoch(artifacts, &image, &register_init, &logs, is_final)?;
        let boundary = Arc::new(local_to_global::epoch_boundary(
            &mut provenance,
            label,
            &collected.touched_memory_cells(),
        ));
        boundaries.push(Arc::clone(&boundary));
        prev_fini = Some(collected.register_fini(&register_init));
        for cell in boundary.iter() {
            image.set(cell.address, (cell.fini.value & 0xFF) as u8);
        }
        let __bs_collect_s = multilinear::whir_split::stage_done(index, "collect", __bs_collect);

        let __bs_build = multilinear::whir_split::mark();
        let traces = Traces::build_from_collected(
            artifacts,
            collected,
            // Continuation epochs use the L2G bookend: PAGE tables (the only
            // image consumers in the build) are skipped.
            None::<&HashMap<u64, u8>>,
            &register_init,
            &MaxRowsConfig::default(),
            private_inputs,
            is_final,
            true,
            #[cfg(feature = "disk-spill")]
            stark::storage_mode::StorageMode::Ram,
        )?;
        let __bs_build_s = multilinear::whir_split::stage_done(index, "build", __bs_build);

        // ★ THE BACKPRESSURE IS THE MEASUREMENT. Under
        // [`for_each_epoch_overlapped`] `each` is a closure whose whole body is
        // a blocking `send` on an UNBUFFERED channel, so this stage is the
        // producer waiting for the prover to take the epoch: `handoff` ≫ 0 says
        // the prover is the bottleneck and nothing spent on the three stages
        // above buys wall; `handoff` ≈ 0 says the producer is, and the prover
        // idled instead. No other stage on either thread can answer that.
        //
        // ⚠ Under a DIRECT `for_each_epoch` caller (the epoch-program and
        // bench tests) `each` is the whole prove, so `handoff` names that
        // instead. None of those sets the knob; the WHIR base is the only
        // caller that does, and it goes through the overlapped form.
        let __bs_handoff = multilinear::whir_split::mark();
        each(
            PreparedEpoch {
                index,
                register_init,
                label,
                traces,
                boundary,
                is_final,
            },
            &boundaries,
        )?;
        let __bs_handoff_s = multilinear::whir_split::stage_done(index, "handoff", __bs_handoff);
        let __bs_epoch_s = multilinear::whir_split::stage_done(index, "epoch", __bs_epoch);
        multilinear::whir_split::push_producer(multilinear::whir_split::ProducerSplit {
            index,
            execute: __bs_exec_s,
            collect: __bs_collect_s,
            build: __bs_build_s,
            handoff: __bs_handoff_s,
            wall: __bs_epoch_s,
        });
        index += 1;
    }

    Ok(boundaries)
}

/// [`for_each_epoch`], with the next epoch prepared while `each` still has the
/// current one.
///
/// Preparing an epoch is the executor and the trace builders — host work — and
/// for a prover `each` is a proof, which is mostly the device's. The two
/// overlap, so a run costs the proofs plus one preparation instead of both.
///
/// The channel has no buffer, so **two epochs are alive at most**: the one in
/// `each`'s hands and the one waiting to be taken. That bound is the point —
/// an epoch's traces are the biggest thing here, and a queue would trade the
/// memory continuations exist to save.
pub(crate) fn for_each_epoch_overlapped(
    elf: &Elf,
    private_inputs: &[u8],
    epoch_size_log2: u32,
    artifacts: &DecodeArtifacts,
    mut each: impl FnMut(PreparedEpoch) -> Result<(), Error>,
) -> Result<Vec<Arc<Vec<CellBoundary>>>, Error> {
    let (sender, receiver) = std::sync::mpsc::sync_channel::<PreparedEpoch>(0);
    std::thread::scope(|scope| {
        let producer = scope.spawn(move || {
            for_each_epoch(
                elf,
                private_inputs,
                epoch_size_log2,
                artifacts,
                |prepared, _| {
                    // The consumer stopping is not this side's failure to
                    // report: its error is the one that says why.
                    sender.send(prepared).map_err(|_| {
                        Error::ContinuationInvariant("the epoch consumer stopped".to_string())
                    })
                },
            )
        });
        // The receiver is consumed here, so it is dropped before the join
        // below — which is what unblocks a producer waiting to hand over the
        // epoch nobody is going to take.
        let used = receiver.into_iter().try_for_each(&mut each);
        let produced = producer
            .join()
            .map_err(|_| Error::ContinuationInvariant("the epoch producer panicked".to_string()))?;
        used?;
        produced
    })
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
pub(crate) struct EpochProof {
    /// The epoch's STARK proof: one `StarkProof` per table, the epoch-local L2G
    /// sub-table last. Its main root is the commitment
    /// `verify_l2g_commitment_binding_view` ties to the global proof.
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

    /// Epoch `i`'s bundle, as the same [`EpochProofView`] the verifier reads.
    #[cfg(test)]
    pub(crate) fn epoch_view(&self, i: usize) -> EpochProofView<'_> {
        EpochProofView::Owned(&self.epochs[i])
    }

    /// The global proof, as the same view the verifier reads.
    #[cfg(test)]
    pub(crate) fn global_proof_view(&self) -> MultiProofView<'_, F, E, ()> {
        MultiProofView::Owned(&self.global)
    }

    /// The shipped touched-page-base list (value-free consumer data).
    #[cfg(test)]
    pub(crate) fn touched_pages(&self) -> &[u64] {
        &self.touched_page_bases
    }

    /// The shipped private-input page count.
    #[cfg(test)]
    pub(crate) fn num_private_pages(&self) -> usize {
        self.num_private_input_pages
    }

    /// What each epoch declared it carries, for tests that have to show the
    /// epochs disagree. `epochs` itself stays private.
    #[cfg(test)]
    pub(crate) fn epoch_table_counts(&self) -> Vec<&TableCounts> {
        self.epochs.iter().map(|e| &e.table_counts).collect()
    }
}

#[cfg(test)]
impl ContinuationProof {
    /// Test-only: flip one bit of epoch `i`'s bound `reg_fini` — a
    /// prover-supplied bundle field the verifier re-binds through the REGISTER
    /// preprocessed commitment, so any consumer that rebuilds the epoch's AIRs
    /// from the bundle must reject the flip.
    pub(crate) fn corrupt_epoch_reg_fini_for_tests(&mut self, i: usize) {
        self.epochs[i].reg_fini[0] ^= 1;
    }

    /// Test-only: flip one bit of epoch `i`'s claimed L2G root — the value
    /// [`verify_epoch`] checks against the proof's own committed root.
    pub(crate) fn corrupt_epoch_l2g_root_for_tests(&mut self, i: usize) {
        self.epochs[i].l2g_root[0] ^= 1;
    }
}

/// The chain-derived inputs epoch `index` of a bundle verifies under, by the
/// same rule as [`verify_continuation_view`]'s loop: `register_init` starts
/// from the ELF entry point, each epoch's bound `reg_fini` becomes the next
/// epoch's INIT, the label is the epoch's position, and the last epoch is
/// final.
#[cfg(test)]
pub(crate) struct EpochChainPosition {
    pub(crate) register_init: Vec<u32>,
    pub(crate) is_final: bool,
    pub(crate) label: u64,
}

/// Derive [`EpochChainPosition`] for epoch `index` of `bundle`. `Ok(None)` if
/// `index` is out of range or an earlier epoch's `reg_fini` has the wrong
/// length (a malformed bundle the verifier rejects up front); `Err` iff a
/// metadata field fails to materialize.
#[cfg(test)]
pub(crate) fn epoch_chain_position(
    bundle: &ContinuationProof,
    elf: &Elf,
    index: usize,
) -> Result<Option<EpochChainPosition>, Error> {
    let n = bundle.num_epochs();
    if index >= n {
        return Ok(None);
    }
    let mut register_init = register::register_init_from_entry_point(elf.entry_point);
    for prior in 0..index {
        let view = bundle.epoch_view(prior);
        if view.reg_fini_len() != register::NUM_REGISTER_ADDRESSES {
            return Ok(None);
        }
        register_init = view.reg_fini()?;
    }
    Ok(Some(EpochChainPosition {
        register_init,
        is_final: index == n - 1,
        label: local_to_global::epoch_label(index as u64),
    }))
}

/// Zero-copy readers over an ARCHIVED bundle, for the LFM arena filler.
///
/// Deliberately on the archived type only. The recursion guest never holds a
/// `ContinuationProof` — it reads a blob from private input and verifies in
/// place ([`verify_continuation_archived`]) — so these expose a path production
/// actually traverses. The equivalent on the owned type would expose a structure
/// the real recursion path never sees, which is a weaker proposition.
///
/// Methods rather than relaxed field visibility because rkyv mirrors the source
/// field's visibility onto the archived struct: opening `epochs` would open the
/// owned type at the same time.
impl ArchivedContinuationProof {
    pub(crate) fn num_epochs(&self) -> usize {
        self.epochs.len()
    }

    /// Epoch `i`'s STARK proof (its tables, epoch-local L2G sub-table last),
    /// as the same view the verifier reads in place.
    pub(crate) fn epoch_proof(&self, i: usize) -> MultiProofView<'_, F, E, ()> {
        MultiProofView::Archived(&self.epochs[i].proof)
    }

    /// Bytes epoch `i` committed.
    pub(crate) fn epoch_public_output(&self, i: usize) -> &[u8] {
        self.epochs[i].public_output.as_slice()
    }

    /// Epoch `i`'s own committed L2G table root — the left-hand side of the
    /// cross-epoch binding [`crate::verify_l2g_commitment_binding_view`] checks
    /// against the global proof's `i`-th sub-proof.
    pub(crate) fn epoch_l2g_root(&self, i: usize) -> Commitment {
        self.epochs[i].l2g_root
    }

    /// Epoch `i`'s final register file `R_{i+1}`, the vector
    /// [`build_epoch_airs`] preprocesses as FINI and the chaining loop carries
    /// forward as epoch `i+1`'s INIT.
    pub(crate) fn epoch_reg_fini(&self, i: usize) -> Result<Vec<u32>, Error> {
        EpochProofView::Archived(&self.epochs[i]).reg_fini()
    }

    /// The one cross-epoch global-memory proof, as the same view the verifier
    /// reads in place. Its first `num_epochs()` sub-proofs are the per-epoch L2G
    /// tables the binding ties to.
    pub(crate) fn global_proof(&self) -> MultiProofView<'_, F, E, ()> {
        MultiProofView::Archived(&self.global)
    }
}

/// Borrowed view over an [`EpochProof`] (owned or archived-in-place). Lets
/// `verify_epoch` take a single argument again instead of the field-by-field
/// parameter list the owned/archived split used to force on every caller:
/// each accessor reads straight off whichever representation is behind it, a
/// plain field copy on the owned side and (for the small metadata fields) an
/// `rkyv::deserialize` on the archived side. The wrap flow's from-proof
/// constructor reads the same view (via [`ContinuationProof::epoch_view`] +
/// [`reconstruct_epoch_airs`]), so an epoch's verifier-side reconstruction has
/// exactly one reader-facing surface.
#[derive(Clone, Copy)]
pub(crate) enum EpochProofView<'a> {
    Owned(&'a EpochProof),
    Archived(&'a ArchivedEpochProof),
}

impl<'a> EpochProofView<'a> {
    /// The epoch's proof (its tables + the epoch-local L2G sub-table last), as
    /// a [`MultiProofView`] — never materialized into an owned `MultiProof` on
    /// the archived side.
    pub(crate) fn per_table_proof(&self) -> MultiProofView<'a, F, E, ()> {
        match self {
            Self::Owned(e) => MultiProofView::Owned(&e.proof),
            Self::Archived(e) => MultiProofView::Archived(&e.proof),
        }
    }

    /// Sub-proof count — one per table, which is what
    /// [`reconstruct_epoch_airs`]'s structural check needs.
    pub(crate) fn num_sub_proofs(&self) -> usize {
        match self {
            Self::Owned(e) => e.proof.proofs.len(),
            Self::Archived(e) => e.proof.proofs.len(),
        }
    }

    /// Bytes this epoch committed (zero-copy borrow either way).
    pub(crate) fn public_output(&self) -> &'a [u8] {
        match self {
            Self::Owned(e) => &e.public_output,
            Self::Archived(e) => e.public_output.as_slice(),
        }
    }

    pub(crate) fn table_counts(&self) -> Result<TableCounts, Error> {
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
    pub(crate) fn runtime_page_ranges(&self) -> Result<Vec<RuntimePageRange>, Error> {
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

    pub(crate) fn reg_fini(&self) -> Result<Vec<u32>, Error> {
        match self {
            Self::Owned(e) => Ok(e.reg_fini.clone()),
            Self::Archived(e) => rkyv::deserialize::<Vec<u32>, rkyv::rancor::Error>(&e.reg_fini)
                .map_err(|err| {
                    Error::Execution(format!("rkyv deserialize reg_fini failed: {err}"))
                }),
        }
    }

    pub(crate) fn l2g_root(&self) -> Commitment {
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
pub(crate) fn build_epoch_airs(
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
    let register_preprocessed = Some(crate::RegisterPreprocessed {
        commitment: register::compute_precomputed_commitment_with_fini(
            opts,
            register_init,
            reg_fini,
        ),
        init: register_init,
        fini: reg_fini,
    });
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
            is_final,
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

    #[cfg(feature = "shape-profile")]
    crate::shape_profile::capture(pairs.iter().map(|(air, trace, _)| (*air, trace.num_rows())));
    let proof = crate::hash_pin::BlockProver::<F, E, ()>::multi_prove(
        pairs,
        &mut seed(),
        #[cfg(feature = "disk-spill")]
        stark::storage_mode::StorageMode::Ram,
        stark::residency_mode::ResidencyMode::Retain,
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

/// An epoch's verifier-side reconstruction: the AIR set (VM tables + the
/// epoch-local L2G air) and the statement values, all rebuilt from the proof
/// bundle plus the chain-derived inputs (`register_init`, `is_final`,
/// `label`). Shared by [`verify_epoch`] and the wrap flow's
/// `RealEpoch`-from-proof constructor so the two reconstructions cannot
/// diverge.
pub(crate) struct EpochReconstruction {
    pub(crate) airs: VmAirs,
    pub(crate) l2g_air: Box<dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>>,
    pub(crate) table_counts: TableCounts,
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) reg_fini: Vec<u32>,
    pub(crate) runtime_page_ranges: Vec<RuntimePageRange>,
}

/// Rebuild epoch `epoch`'s AIR set from the bundle alone. `Ok(None)` = a
/// well-formed bundle that is structurally wrong (degenerate table counts,
/// sub-proof count mismatch) — the cases [`verify_epoch`] rejects with
/// `Ok(false)`; `Err` iff a small metadata field failed to materialize off an
/// archived bundle.
#[allow(clippy::too_many_arguments)]
pub(crate) fn reconstruct_epoch_airs(
    elf: &Elf,
    epoch: EpochProofView<'_>,
    register_init: &[u32],
    is_final: bool,
    label: u64,
    opts: &ProofOptions,
    decode_commitment: Option<Commitment>,
) -> Result<Option<EpochReconstruction>, Error> {
    let table_counts = epoch.table_counts()?;
    // Reject degenerate table counts (mirrors the monolithic verifier).
    if table_counts.validate().is_err() {
        return Ok(None);
    }

    // Cross-check table_counts before building AIRs from bundle data. Continuation
    // epochs have no PAGE proofs, and append one epoch-local L2G proof after the VM
    // tables. HALT is present only on the final epoch.
    let fixed_tables = if is_final {
        FIXED_TABLE_COUNT
    } else {
        FIXED_TABLE_COUNT - 1
    };
    // Checked: the counts are prover-supplied and release builds wrap, so a
    // plain sum would let one astronomically large field still match
    // `num_sub_proofs()` and reach `VmAirs::new`, which sizes a `Vec` from that
    // field directly.
    let Some(expected_proof_count) = table_counts
        .total()
        .and_then(|t| t.checked_add(fixed_tables))
        .and_then(|t| t.checked_add(1))
    else {
        return Ok(None);
    };
    if expected_proof_count != epoch.num_sub_proofs() {
        return Ok(None);
    }

    let reg_fini = epoch.reg_fini()?;
    let runtime_page_ranges = epoch.runtime_page_ranges()?;

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
    let l2g_air = Box::new(l2g_memory_air(opts, label))
        as Box<dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>>;
    Ok(Some(EpochReconstruction {
        airs,
        l2g_air,
        table_counts,
        reg_fini,
        runtime_page_ranges,
    }))
}

/// Verify one epoch using ONLY the epoch's public statement fields (via
/// [`EpochProofView`]) plus the verifier-derived `register_init` (epoch 0:
/// from the ELF; epoch i>0: from the previous epoch's `reg_fini`), `is_final`,
/// and `label`. Rebuilds the AIRs and transcript from the bundle's statement
/// values ([`reconstruct_epoch_airs`]) and indexes commits from the carried
/// x254 (`register_init[X254_INDEX]`), never from the prover's memory. PAGE is
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
    let Some(recon) = reconstruct_epoch_airs(
        elf,
        epoch,
        register_init,
        is_final,
        label,
        opts,
        decode_commitment,
    )?
    else {
        return Ok(false);
    };
    let EpochReconstruction {
        airs,
        l2g_air,
        table_counts,
        reg_fini: _,
        runtime_page_ranges,
    } = recon;

    let public_output = epoch.public_output();
    let mut refs = airs.air_refs();
    refs.push(&*l2g_air);

    let seed = || {
        epoch_transcript(
            elf_bytes,
            public_output,
            &table_counts,
            &runtime_page_ranges,
            label,
            is_final,
            opts.fri_final_poly_log_degree,
        )
    };

    // Start the commit index from the carried x254 (the derived INIT), not a free
    // input — this is what binds the per-epoch commit slice to its global position.
    let commit_start_index = register_init
        .get(register::X254_INDEX)
        .copied()
        .unwrap_or(0) as u64;

    let proof = epoch.per_table_proof();
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

    if !crate::hash_pin::BlockVerifier::<F, E, ()>::multi_verify_views(
        &refs,
        proof,
        &mut seed(),
        &expected,
    ) {
        return Ok(false);
    }

    // The claimed L2G root must be the one this proof actually committed (it is
    // what verify_l2g_commitment_binding_view later ties to the global proof).
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

    #[cfg(feature = "shape-profile")]
    crate::shape_profile::capture(pairs.iter().map(|(air, trace, _)| (*air, trace.num_rows())));

    crate::hash_pin::BlockProver::<F, E, ()>::multi_prove(
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
        stark::residency_mode::ResidencyMode::Retain,
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

    crate::hash_pin::BlockVerifier::<F, E, ()>::multi_verify_views(
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

/// The text of a panic payload, for the pipeline error that reports it in
/// place of the panic (`panic!` with a string literal or a formatted message
/// covers every abort the device layer raises).
fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "non-string panic payload".to_string()
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
/// `LAMBDA_VM_BASE_SPLIT=1` turns on one `BASE EPOCH` line per pipeline stage.
///
/// ★ WHY. The base prints ONE number for nineteen epochs — `base: 19 epochs in
/// 67.1s` — and behind it is a three-stage pipeline: a single-threaded producer
/// (execute + collect), a small pool of trace builders, and one prover. Those
/// are three different machines with three different levers, and an aggregate
/// cannot say which of them set the wall. The `recv` stage is the one the
/// aggregate hides hardest: it is the prover thread — the only stage that
/// reaches the card — sitting idle waiting for a builder.
///
/// Runtime-gated rather than `--features instruments` for the reason
/// [`stark::prove_split`] gives: the record is produced by a binary that does
/// not enable the feature, and a split taken from a different binary describes
/// a different run.
///
/// ★ ONE KNOB, BOTH PIPELINES. This name also turns on the WHIR base's stages
/// — the producer's four in [`for_each_epoch`] and the prover's in
/// [`crate::multilinear_continuation::prove_epoch`] — and it means the same
/// thing on both: one line per pipeline stage per epoch, in this format.
///
/// ⛔ IT DID NOT USED TO. The LFM tree launcher has exported
/// `LAMBDA_VM_BASE_SPLIT=1` on the WHIR arm since that arm existed, "byte
/// identical to the D-S exports" — and it reached nothing, because the WHIR
/// base does not go through the [`prove_continuation`] below. Fifteen epochs
/// and 57% of the block's wall printed as one number, under a knob the log
/// showed as set. A knob that names a measurement on one pipeline and nothing
/// at all on another is worse than a missing one: the export is the evidence a
/// reader uses to believe the breakdown was taken.
fn base_split_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| match std::env::var("LAMBDA_VM_BASE_SPLIT") {
        Ok(v) => !v.is_empty() && v != "0",
        Err(_) => false,
    })
}

/// Open a timed pipeline stage. `None`, and no clock read, when the knob is off.
fn base_stage() -> Option<(std::time::Instant, f64)> {
    base_split_enabled().then(|| (std::time::Instant::now(), stark::prove_split::epoch_secs()))
}

/// Close a stage opened by [`base_stage`] and print its line.
///
/// The two wall-clock stamps are what let an external GPU sampler be sliced by
/// stage; the duration alone cannot place the stage on the sampler's timeline.
fn base_stage_done(index: u64, stage: &str, open: Option<(std::time::Instant, f64)>) {
    if let Some((start, t0)) = open {
        let secs = start.elapsed().as_secs_f64();
        println!(
            "BASE EPOCH {index}: {stage} {secs:.2}s t=[{t0:.3},{:.3}]",
            stark::prove_split::epoch_secs()
        );
    }
}

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

    // Root span for the profiling toolkit (scripts/profiling): the whole
    // continuation prove is one tree; per-stage spans below are recorded from
    // their worker threads and told apart by label + instance order.
    #[cfg(feature = "instruments")]
    stark::instruments::reset_timeline();
    #[cfg(feature = "instruments")]
    let __root = stark::instruments::span("prove_continuation_total");

    let elf = Elf::load(elf_bytes).map_err(|e| Error::ElfLoad(format!("{e}")))?;
    let mut executor = Executor::new(&elf, private_inputs.to_vec())
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
    let mut image = build_initial_image_paged(&elf, private_inputs);
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
    // sends to the epoch prover) on this dedicated channel, in epoch order. It
    // is unbounded, so the producer never blocks on it and the epoch pipeline's
    // own bounded channels stay the only backpressure; it is drained once, after
    // the epoch scope joins, and the global proof is proven from it THERE — see
    // the ★ note at the drain site for why that is deliberately not overlapped.
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
            let __bs_recv = base_stage();
            let prepared = match rx.recv() {
                Ok(Ok(p)) => p,
                Ok(Err(e)) => {
                    first_err.lock().unwrap().get_or_insert(e);
                    continue;
                }
                Err(_) => return proved, // channel closed: no more epochs
            };
            base_stage_done(prepared.index, "recv", __bs_recv);
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
            // A PANIC in the prove — a loud device abort, or any bug — must take
            // the same drain path as an `Err`. If this thread simply died, `rx`
            // would close, every builder would stop on its dead sender, and the
            // producer (which watches `first_err`, never written by a panic)
            // would block forever in `build_tx.send` on the capacity-1 build
            // channel, whose receiver lives outside the scope: the prove would
            // hang instead of failing. Seen once: an aux-build abort slept for
            // 21 minutes under the CLI.
            let index = prepared.index;
            let __bs_prove = base_stage();
            let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                // Inside the guard on purpose: a panic raised on this thread
                // OUTSIDE it reproduces the original wedge (which is what the
                // test's first placement did, and the test caught it).
                #[cfg(test)]
                if index == test_fault::FAIL_INDEX && private_inputs == test_fault::PANIC_MAGIC {
                    panic!("injected prover panic (test)");
                }
                prove_epoch(
                    &elf,
                    elf_bytes,
                    &start,
                    prepared.traces,
                    prepared.is_final,
                    &prepared.boundary,
                    opts,
                    decode_commitment,
                )
            }));
            base_stage_done(index, "prove", __bs_prove);
            match outcome {
                Ok(Ok(epoch)) => proved.push((index, epoch)),
                Ok(Err(e)) => {
                    first_err.lock().unwrap().get_or_insert(e);
                    continue; // drain mode (see loop comment)
                }
                Err(payload) => {
                    first_err
                        .lock()
                        .unwrap()
                        .get_or_insert(Error::ContinuationInvariant(format!(
                            "epoch {index} prover panicked: {}",
                            panic_message(&*payload)
                        )));
                    continue; // drain mode
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
            // Same reason as the prover's guard: a builder that dies leaves the
            // producer blocked on the build channel once every builder is gone.
            let build_index = job.index;
            let __bs_build = base_stage();
            let traces = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                Traces::build_from_collected(
                    decode_artifacts_ref,
                    job.collected,
                    // Continuation epochs use the L2G bookend: PAGE tables (the
                    // only image consumers in the build) are skipped.
                    None::<&std::collections::HashMap<u64, u8>>,
                    &job.register_init,
                    &MaxRowsConfig::default(),
                    private_inputs,
                    job.is_final,
                    true,
                    #[cfg(feature = "disk-spill")]
                    stark::storage_mode::StorageMode::Ram,
                )
            })) {
                Ok(traces) => traces,
                Err(payload) => Err(Error::ContinuationInvariant(format!(
                    "epoch {build_index} trace build panicked: {}",
                    panic_message(&*payload)
                ))),
            };
            base_stage_done(build_index, "build", __bs_build);
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
                        let __bs_pre = base_stage();
                        traces.preupload_main_traces();
                        base_stage_done(job.index, "preupload", __bs_pre);
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
                    let __bs_exec = base_stage();
                    let logs = match executor
                        .resume_with_limit(epoch_size)
                        .map_err(|e| Error::Execution(format!("{e}")))?
                    {
                        Some(logs) => logs.to_vec(),
                        None => return Ok(()),
                    };
                    base_stage_done(index, "execute", __bs_exec);
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
                    let __bs_collect = base_stage();
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
                    base_stage_done(index, "collect", __bs_collect);
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
                    let __bs_send = base_stage();
                    let send_failed = build_tx.send(Ok(job)).is_err();
                    base_stage_done(index, "handoff", __bs_send);
                    if send_failed || is_final {
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

    // One global LogUp over all the (kept) local-to-global tables. The scope
    // above has joined, so every epoch prove has finished and released its
    // device memory before this one starts.
    //
    // ★ Deliberately NOT overlapped with the epoch proves, though nothing in the
    // ARGUMENT forbids it: this is a second `stark::prover::multi_prove`, and
    // each `multi_prove` builds its OWN `VramGate` from the whole device budget
    // (80% of total). Two running at once therefore put 2x the card's admission
    // budget on one GPU with nothing summing them, and each additionally holds
    // its round-1 working set — every table's LDE, trace snapshot and Merkle
    // tree stay device-resident from its commit until its own rounds task ends —
    // which no gate counts at all. Measured on an RTX 5090 (31.40 GiB): the
    // overlap window opened about `builders + 2` epochs before the end, and
    // inside it the card ran out with tables from BOTH proves aborting in the
    // same instant (`gpu_lde::refuse_host_recovery`).
    //
    // The cost of serialising is this proof's own wall time, which the overlap
    // used to hide behind the tail epochs. It changes no proof bytes: the global
    // proof consumes only execution artifacts (boundaries, ELF, genesis pages),
    // never an epoch proof, so the schedule was always free to choose.
    let all: Vec<Arc<Vec<CellBoundary>>> = boundary_rx.try_iter().collect();
    let num_private_input_pages = page::private_input_page_count(private_inputs);
    // SINGLE source of truth: the same page-base list drives the committed
    // GLOBAL_MEMORY tables and is shipped in the bundle, so the two can never
    // diverge in set or order.
    let touched_page_bases = touched_page_bases(&all);
    let global = {
        #[cfg(feature = "instruments")]
        let __sp = stark::instruments::span("prove_global");
        prove_global(
            &all,
            elf_bytes,
            &init_page_data,
            &touched_page_bases,
            num_private_input_pages,
            opts,
        )?
    };

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
    let bundle = prove_continuation(elf_bytes, private_inputs, epoch_size_log2, opts)?;
    verify_continuation(elf_bytes, &bundle, opts)
}

/// Stateless test-only fault trigger for the epoch pipeline: the builder
/// injects an error at epoch [`FAIL_INDEX`] when the prove's private input is
/// exactly [`MAGIC`]. Constants only — concurrent tests can never trip it.
#[cfg(test)]
pub(crate) mod test_fault {
    pub(crate) const MAGIC: &[u8] = b"__inject_pipeline_fault__";
    /// Same index, but the PROVER thread panics instead of returning `Err`:
    /// the shutdown path a loud device abort takes.
    pub(crate) const PANIC_MAGIC: &[u8] = b"__inject_pipeline_panic__";
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

    // The pipeline's PANIC path: a loud device abort is a panic on the prover
    // thread, not an `Err`. Before the guard it wedged the pipeline — the
    // builders stopped on their dead sender and the producer blocked forever
    // on the build channel — and a real prove slept for 21 minutes. It must
    // surface as `Err` carrying the panic's own message.
    #[test]
    fn test_prover_panic_mid_pipeline_returns_err() {
        let _ = env_logger::builder().is_test(true).try_init();
        let elf_bytes = asm_elf_bytes("all_loadstore_32");
        let (done_tx, done_rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let r = prove_continuation(
                &elf_bytes,
                test_fault::PANIC_MAGIC,
                2,
                &ProofOptions::default_test_options(),
            );
            let _ = done_tx.send(r.map(|_| ()));
        });
        let result = done_rx
            .recv_timeout(std::time::Duration::from_secs(300))
            .expect("prove_continuation wedged: a prover panic did not shut the pipeline down");
        let err = result.expect_err("the injected panic must surface as Err");
        assert!(
            format!("{err:?}").contains("injected prover panic"),
            "unexpected error: {err:?}"
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
                2,
                &ProofOptions::default_test_options(),
            );
            let _ = done_tx.send(r.map(|_| ()));
        });
        let result = done_rx
            // 1800 s of headroom rather than 300. This is a liveness guard — the
            // regression it catches wedges the pipeline FOREVER, so any finite
            // bound still catches it — and the bound has to clear an honest run
            // under load: an algebraic hash pin doubles this test's own proving
            // work (alone, three runs each: 7.3-7.6 s at the BLAKE3 default,
            // 15.7-15.9 s under RPX), and inside the full `--lib` suite's
            // parallel load the old 300 s fired while the test was still making
            // progress.
            .recv_timeout(std::time::Duration::from_secs(1800))
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

    /// Each epoch drops the chips it never reaches, and it decides that on its
    /// own: a table missing from one epoch still shows up in another that does
    /// use it. The skip is not a property of the run, it is a property of the
    /// epoch — computing it over the whole run instead would drag every table
    /// used anywhere into every epoch.
    #[test]
    fn table_presence_is_decided_per_epoch() {
        let _ = env_logger::builder().is_test(true).try_init();
        let elf_bytes = asm_elf_bytes("all_loadstore_32");
        let epoch_size_log2 = 3;
        let opts = ProofOptions::default_test_options();

        let bundle = prove_continuation(&elf_bytes, &[], epoch_size_log2, &opts).unwrap();
        // Guard against silent degradation: one epoch cannot disagree with another.
        assert!(
            bundle.epochs.len() >= 2,
            "need at least two epochs, got {}",
            bundle.epochs.len()
        );

        // `(name, count)` per epoch, in a fixed order so the rows line up.
        let per_epoch: Vec<Vec<(&str, usize)>> = bundle
            .epochs
            .iter()
            .map(|e| {
                let c = &e.table_counts;
                vec![
                    ("lt", c.lt),
                    ("memw", c.memw),
                    ("memw_aligned", c.memw_aligned),
                    ("load", c.load),
                    ("mul", c.mul),
                    ("dvrm", c.dvrm),
                    ("shift", c.shift),
                    ("branch", c.branch),
                    ("eq", c.eq),
                    ("bytewise", c.bytewise),
                    ("store", c.store),
                    ("cpu32", c.cpu32),
                    ("keccak", c.keccak),
                    ("keccak_rnd", c.keccak_rnd),
                    ("ecsm", c.ecsm),
                    ("ecdas", c.ecdas),
                    ("hint", c.hint),
                    ("commit", c.commit),
                ]
            })
            .collect();
        let layout = || {
            per_epoch
                .iter()
                .enumerate()
                .map(|(i, row)| {
                    let present: Vec<&str> = row
                        .iter()
                        .filter(|(_, n)| *n > 0)
                        .map(|(t, _)| *t)
                        .collect();
                    format!("epoch {i}: {present:?}")
                })
                .collect::<Vec<_>>()
                .join("\n  ")
        };

        // Something is actually being skipped, or the rest proves nothing.
        assert!(
            per_epoch.iter().any(|row| row.iter().any(|(_, n)| *n == 0)),
            "no epoch skipped any table:\n  {}",
            layout()
        );

        // And the epochs disagree: some table is in one and out of another.
        //
        // Among the *non-final* epochs only. The final epoch is the one that
        // carries HALT and the one where a program's output is committed, so a
        // difference that involves it can be structural — a whole-run table set
        // would still produce it, and this assertion would pass while measuring
        // nothing about per-epoch granularity.
        assert!(
            per_epoch.len() >= 3,
            "need at least two non-final epochs to compare, got {} epochs",
            per_epoch.len()
        );
        let non_final = &per_epoch[..per_epoch.len() - 1];
        let disagreeing: Vec<&str> = non_final[0]
            .iter()
            .enumerate()
            .filter(|(i, (_, first))| {
                non_final
                    .iter()
                    .any(|row| (row[*i].1 == 0) != (*first == 0))
            })
            .map(|(_, (name, _))| *name)
            .collect();
        assert!(
            !disagreeing.is_empty(),
            "the non-final epochs all carry the same tables, so per-epoch \
             granularity is untested here — pick a program or epoch size that \
             varies:\n  {}",
            layout()
        );

        // Sharper: a table that is present, goes away, and comes back cannot be
        // produced by any scheme that computes one set over the run or over a
        // prefix of it. Counting the blocks of consecutive epochs a table
        // appears in, more than one block is exactly that shape.
        let blocks = |i: usize| {
            let present: Vec<bool> = per_epoch.iter().map(|row| row[i].1 > 0).collect();
            present
                .iter()
                .enumerate()
                .filter(|(k, p)| **p && (*k == 0 || !present[k - 1]))
                .count()
        };
        let reappearing: Vec<&str> = per_epoch[0]
            .iter()
            .enumerate()
            .filter(|(i, _)| blocks(*i) > 1)
            .map(|(_, (name, _))| *name)
            .collect();
        assert!(
            !reappearing.is_empty(),
            "no table leaves and comes back, so a union or prefix scheme would \
             produce this same layout — the test cannot tell them apart:\n  {}",
            layout()
        );

        println!("tables present in some non-final epochs but not others: {disagreeing:?}");
        println!("tables that leave and come back: {reappearing:?}");
        println!("  {}", layout());

        // The mixed-shape bundle has to verify end to end.
        assert!(
            verify_continuation(&elf_bytes, &bundle, &opts)
                .unwrap()
                .is_some(),
            "a bundle whose epochs carry different table sets must still verify"
        );
    }

    // Supplied genesis roots must verify identically to the trustless recompute,
    // and a tampered root (DECODE or a page) must be rejected. `data_page_touch`
    // touches a real ELF `.data` page, unlike this file's stack-only fixtures.
    #[test]
    fn test_verify_continuation_with_supplied_roots() {
        let elf_bytes = asm_elf_bytes("data_page_touch");
        let opts = ProofOptions::default_test_options();
        let bundle = prove_continuation(&elf_bytes, &[], 3, &opts).unwrap();

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
            let bundle = prove_continuation(&elf_bytes, &[], 3, &opts).unwrap();
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

    // A bundle survives an rkyv round-trip and still verifies to the same output —
    // the serialization path the CLI's `prove`/`verify --continuations` relies on.
    #[test]
    fn test_continuation_rkyv_roundtrip() {
        let _ = env_logger::builder().is_test(true).try_init();
        let elf_bytes = asm_elf_bytes("test_commit_split");
        let bundle =
            prove_continuation(&elf_bytes, &[], 4, &ProofOptions::default_test_options()).unwrap();

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

    /// The epoch counterpart of the overflow branch: `verify_epoch` swallows a
    /// total that wraps as `Ok(false)` where the monolithic verifier returns
    /// `Err`, so a regression there is silent. Nothing reached that arm before.
    #[test]
    fn test_split_verify_rejects_an_epoch_whose_counts_overflow() {
        let _ = env_logger::builder().is_test(true).try_init();
        let elf_bytes = asm_elf_bytes("all_loadstore_32");
        let opts = ProofOptions::default_test_options();
        let mut bundle = prove_continuation(&elf_bytes, &[], 3, &opts).unwrap();
        assert!(!bundle.epochs.is_empty());

        bundle.epochs[0].table_counts.lt = usize::MAX;
        assert!(
            bundle.epochs[0].table_counts.total().is_none(),
            "the tampered counts must actually wrap, or this tests the wrong branch"
        );
        assert!(
            verify_continuation(&elf_bytes, &bundle, &opts)
                .unwrap()
                .is_none(),
            "an epoch whose declared counts have no total must be rejected"
        );
    }

    /// The continuation counterpart of the monolithic
    /// `test_verify_rejects_undercounted_table_count`: an epoch that declares
    /// away a table it actually carries. Both branches of the cross-check are
    /// exercised, because the fixed-table term differs between a non-final
    /// epoch (`FIXED_TABLE_COUNT - 1`, no HALT) and the final one — and
    /// `verify_epoch` swallows the mismatch as `Ok(false)` rather than an
    /// error, so a regression in that arithmetic would be silent.
    #[test]
    fn test_split_verify_rejects_undercounted_epoch_table_count() {
        let _ = env_logger::builder().is_test(true).try_init();
        let elf_bytes = asm_elf_bytes("all_loadstore_32");
        let opts = ProofOptions::default_test_options();
        let mut bundle = prove_continuation(&elf_bytes, &[], 3, &opts).unwrap();
        assert!(
            bundle.epochs.len() >= 2,
            "need a non-final and a final epoch, got {}",
            bundle.epochs.len()
        );
        let last = bundle.epochs.len() - 1;
        for epoch in [0, last] {
            // Whatever this epoch does carry: the point is declaring one of its
            // own tables away, not which one.
            let counts = &mut bundle.epochs[epoch].table_counts;
            let (name, restore) = if counts.load > 0 {
                ("load", std::mem::replace(&mut counts.load, 0))
            } else if counts.store > 0 {
                ("store", std::mem::replace(&mut counts.store, 0))
            } else {
                ("lt", std::mem::replace(&mut counts.lt, 0))
            };
            assert!(
                restore > 0,
                "epoch {epoch} carries no optional table to declare away"
            );
            assert!(
                verify_continuation(&elf_bytes, &bundle, &opts)
                    .unwrap()
                    .is_none(),
                "epoch {epoch} declaring away its {name} table must be rejected"
            );
            let counts = &mut bundle.epochs[epoch].table_counts;
            match name {
                "load" => counts.load = restore,
                "store" => counts.store = restore,
                _ => counts.lt = restore,
            }
        }

        // The control, and the reason the rejections above mean something: put
        // the counts back and the same bundle verifies. One verify, at the end,
        // rather than one before and one after — each costs a full pass.
        assert!(
            verify_continuation(&elf_bytes, &bundle, &opts)
                .unwrap()
                .is_some(),
            "restoring the counts must bring the bundle back"
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
        let mut bundle =
            prove_continuation(&elf_bytes, &[], 3, &ProofOptions::default_test_options()).unwrap();
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

    // Negative: corrupting an epoch's claimed L2G table root must be rejected. This
    // tamper is caught by `verify_epoch`'s own root-consistency check (the epoch's
    // claimed `l2g_root` no longer matches what its own proof committed) before the
    // cross-epoch `verify_l2g_commitment_binding_view` ever runs — see
    // `test_split_verify_rejects_global_proof_from_a_different_run` for that.
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

    // Same tamper as `test_split_verify_rejects_tampered_l2g_root`, but through the
    // zero-copy blob path (`verify_continuation_and_attest`) rather than
    // `verify_continuation`. Guards the archived path's per-epoch root check against
    // the same corruption the owned path already catches.
    #[test]
    fn test_continuation_blob_rejects_tampered_l2g_root() {
        let _ = env_logger::builder().is_test(true).try_init();
        let elf_bytes = asm_elf_bytes("all_loadstore_32");
        let mut bundle =
            prove_continuation(&elf_bytes, &[], 3, &crate::recursion::MIN_PROOF_OPTIONS).unwrap();
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

        let mut bundle_a = prove_continuation(&elf_bytes, &input_a, 2, &opts).unwrap();
        let bundle_b = prove_continuation(&elf_bytes, &input_b, 2, &opts).unwrap();
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

        let mut bundle_a = prove_continuation(&elf_bytes, &input_a, 2, &opts).unwrap();
        let bundle_b = prove_continuation(&elf_bytes, &input_b, 2, &opts).unwrap();
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

    // =======================================================================
    // THE GENESIS-STACK SPLIT
    // =======================================================================

    /// The block's measured census, reproduced by the forms — an F1 on the
    /// census rather than a restatement of it.
    ///
    /// Read on the box at 60d790310 (`cens2`, one execution of the guest, no
    /// proving, no card): `GENESIS CENSUS: 35 pages = 30 genesis + 5 private;
    /// 569362 nonzero entries, worst page 229290; 10249056 leg rows`.
    const BLOCK_DENSE: [(u64, usize, usize); 3] = [
        (0x0, 116_692, 2_100_474),
        (0x40000, 229_290, 4_127_238),
        (0x280000, 223_380, 4_020_858),
    ];
    const BLOCK_ZERO_PAGES: usize = 27;
    const BLOCK_PAGE_VARS: usize = 18;
    const BLOCK_CENSUS_LEG_ROWS: usize = 10_249_056;
    /// The census's genesis pages: 35 touched less the 5 private.
    const BLOCK_GENESIS_PAGES: usize = 30;
    /// The bracket the block stands at and what the form charges there — both
    /// named here so a reader of these numbers sees what they rest on.
    const BLOCK_N_FIXED: usize = BLOCK_STACK_VARS;
    const BLOCK_MARGINAL: usize = marginal_stacked_rows(BLOCK_PAGE_VARS, BLOCK_STACK_VARS);
    /// The bracket a LONE genesis page stands at — `18 + ceil(log2(2))` — and
    /// what the rule charges there. ⚠ IT IS NOT THE BLOCK'S, and the ten rows
    /// between them are the whole reason the marginal stopped being one number.
    const LONE_N_FIXED: usize = fixed_stack_vars(BLOCK_PAGE_VARS, 1);
    const LONE_MARGINAL: usize = marginal_stacked_rows(BLOCK_PAGE_VARS, LONE_N_FIXED);

    /// The census was read at one page size; every number below is evaluated at
    /// that height only while this holds.
    #[test]
    fn the_blocks_page_height_is_the_page_size_the_code_uses() {
        assert_eq!(PAGE_NUM_VARS, BLOCK_PAGE_VARS);
    }

    /// A genesis page presents exactly the columns the stack takes.
    #[test]
    fn a_genesis_pages_preprocessed_columns_are_the_pair_the_stack_carries() {
        let config = PageConfig::with_data(0x40000, vec![1u8; 8]);
        assert_eq!(
            page::preprocessed_columns(&config).len(),
            PAGE_PREPROCESSED_COLUMNS,
            "the stack takes a prefix of this list, so its length is the contract"
        );
    }

    #[test]
    fn the_sparse_form_reproduces_the_blocks_census_to_the_row() {
        let mut total = 0usize;
        for (base, nonzero, rows) in BLOCK_DENSE {
            let got = sparse_leg_rows(BLOCK_PAGE_VARS, nonzero);
            assert_eq!(got, rows, "page {base:#x}: {nonzero} nonzero entries");
            total += got;
        }
        // The 27 all-zero pages are not free: each costs the interned zero.
        total += BLOCK_ZERO_PAGES * sparse_leg_rows(BLOCK_PAGE_VARS, 0);
        assert_eq!(
            total, BLOCK_CENSUS_LEG_ROWS,
            "the closed form and the box census disagree about what INIT costs"
        );
    }

    /// A page set of [`BLOCK_GENESIS_PAGES`] genesis pages whose first ones
    /// carry the given nonzero counts and whose rest are all-zero.
    ///
    /// ⚠ THESE TESTS BUILD A SET AND RUN THE PLANNER, rather than calling the
    /// forms. Part 2 is a property of the whole set, so a two-page answer and a
    /// thirty-page answer are different questions, and only a real plan
    /// exercises the two passes that decide them.
    fn genesis_pages_with(counts: &[usize]) -> Vec<PageConfig> {
        let page = 1u64 << BLOCK_PAGE_VARS;
        let mut configs = Vec::with_capacity(BLOCK_GENESIS_PAGES);
        let mut base = 0u64;
        for &nonzero in counts {
            configs.push(data_page(base, vec![1u8; nonzero]));
            base += page;
        }
        while configs.len() < BLOCK_GENESIS_PAGES {
            configs.push(PageConfig::zero_init(base));
            base += page;
        }
        configs
    }

    /// The block's own thirty genesis pages: the three dense ones at their
    /// census bases and counts, and twenty-seven all-zero.
    fn block_genesis_configs() -> Vec<PageConfig> {
        assert_eq!(
            BLOCK_DENSE.len() + BLOCK_ZERO_PAGES,
            BLOCK_GENESIS_PAGES,
            "the census's page counts must add up, or this set is not the block's"
        );
        let page = 1u64 << BLOCK_PAGE_VARS;
        let mut configs: Vec<PageConfig> = BLOCK_DENSE
            .iter()
            .map(|&(base, nonzero, _)| data_page(base, vec![1u8; nonzero]))
            .collect();
        let mut base = 0x1000000u64;
        for _ in 0..BLOCK_ZERO_PAGES {
            configs.push(PageConfig::zero_init(base));
            base += page;
        }
        configs.sort_by_key(|config| config.page_base);
        configs
    }

    /// THE FORM, THE BRACKET IT IS QUOTED AT, AND WHAT IT DERIVES.
    ///
    /// ⚠ NEITHER `τ` NOR THE MARGINAL IS A RULED QUANTITY, and both move with
    /// the RUN's bracket rather than with a decision. `τ` is 5 below `n_fixed`
    /// 23 and 6 from there, so a run of nine or more genesis pages sits at 6
    /// and the block does; the only page that difference reclassifies carries
    /// five nonzero genesis bytes, and nothing in the block or the fixtures
    /// sits there. The band test below walks the brackets and is the reading of
    /// that.
    #[test]
    fn the_form_and_the_bracket_it_is_quoted_at() {
        // The bracket is the one the block's page count gives, which is what
        // makes 111 the right charge for a block-shaped run and the wrong one
        // for a lone page.
        assert_eq!(
            fixed_stack_vars(BLOCK_PAGE_VARS, BLOCK_GENESIS_PAGES),
            BLOCK_STACK_VARS,
            "18 + ceil(log2(60))"
        );
        assert_eq!(BLOCK_MARGINAL, 111);
        assert_eq!(LONE_N_FIXED, 19, "18 + ceil(log2(2))");
        assert_eq!(LONE_MARGINAL, 101);

        // ⚠ THIS IS THE FORM'S ARITHMETIC, NOT THE STACK'S BILL. The pin
        // `lfm::whir_stacked_tests::the_marginal_the_routing_rule_charges_is_the_one_the_stack_bills`
        // is the only place the form is compared against what
        // `stacked_verify_cost` actually charges; here the four terms are added
        // a second way so a mistyped one is caught where it is written.
        assert_eq!(
            BLOCK_MARGINAL,
            5 * BLOCK_PAGE_VARS
                + PAGE_PREPROCESSED_COLUMNS * (BLOCK_STACK_VARS - BLOCK_PAGE_VARS)
                + PAGE_PREPROCESSED_COLUMNS * STACKED_ROWS_PER_COLUMN
                + MAX_SPONGE_MARGINAL,
            "eq 90 + indicators 12 + per-column 6 + the sponge bound 3"
        );
        // Two rows a page per prefix bit, which is the term that makes the
        // marginal a function of the bracket at all.
        assert_eq!(
            marginal_stacked_rows(BLOCK_PAGE_VARS, BLOCK_STACK_VARS + 1) - BLOCK_MARGINAL,
            PAGE_PREPROCESSED_COLUMNS
        );
        assert_eq!(
            marginal_stacked_rows(BLOCK_PAGE_VARS, BLOCK_STACK_VARS + 1),
            113
        );

        // τ from the form at the block's bracket: `18 + 18*S > 111` ⇒ S >= 6.
        let tau = candidate_threshold_entries(BLOCK_PAGE_VARS, BLOCK_STACK_VARS);
        assert_eq!(tau, 6);
        assert!(!is_candidate(BLOCK_PAGE_VARS, BLOCK_STACK_VARS, 5));
        assert!(is_candidate(BLOCK_PAGE_VARS, BLOCK_STACK_VARS, 6));
        // ⚠ FLOOR PLUS ONE AND NOT `div_ceil`: τ's leg must EXCEED the
        // marginal, never meet it.
        assert!(sparse_leg_rows(BLOCK_PAGE_VARS, tau - 1) <= BLOCK_MARGINAL);
        assert!(sparse_leg_rows(BLOCK_PAGE_VARS, tau) > BLOCK_MARGINAL);
        // ⛔ AND τ IS 5 AT A LONE PAGE'S BRACKET — the same page, a different
        // run, a different answer. A reader quoting "τ = 6" without its bracket
        // is quoting the block's number for everyone.
        assert_eq!(
            candidate_threshold_entries(BLOCK_PAGE_VARS, LONE_N_FIXED),
            5
        );

        // ⛔ THE ONE BOUND THAT HOLDS AT EVERY BRACKET: the marginal must exceed
        // what `weight_at_rows` alone bills for a page at the charging height,
        // which is 102 at the block's (eq 90 + two indicators of six). The form
        // clears it by construction — the `+ 6` and the sponge bound are the
        // terms that closure cannot see — and a form that stopped would charge
        // less than a term it provably contains.
        //
        // ⚠ A `const` ASSERT, NOT A RUNTIME ONE: breaking it stops the tree
        // COMPILING rather than failing a test nobody ran.
        const {
            assert!(
                marginal_stacked_rows(BLOCK_PAGE_VARS, BLOCK_STACK_VARS) > 102,
                "the marginal must exceed the 102 rows the weight closure alone bills \
                 a page at the charging height; `whir_chain_tests` reads that 102 off \
                 the emitter rather than restating it"
            )
        };
        const {
            assert!(
                MAX_SPONGE_MARGINAL >= 1,
                "every carried page absorbs six more felts into the threaded sponge, so \
                 its marginal contribution is at least one row; a zero bound would \
                 undercharge every position"
            )
        };
    }

    /// ⚠ PART 1's TWO SPELLINGS ARE ONE RULE — read, not taken on trust.
    ///
    /// [`is_candidate`] branches on `nonzero >= τ`; the rule is stated as
    /// `sparse_leg_rows > marginal_stacked_rows(..)`. The equality of the two
    /// is a floor-versus-strict-inequality argument, which is exactly the kind
    /// that is right until the division comes out exact. So it is executed,
    /// over every `S` around the boundary, at every height AND at every
    /// bracket — including the pairs where `(marginal - num_vars)` divides
    /// evenly by `num_vars`.
    #[test]
    fn the_threshold_and_its_entry_count_are_one_rule() {
        let mut exact_divisions = 0usize;
        let mut div_ceil_would_differ = 0usize;
        // ⚠ THE HEIGHT AND THE BRACKET ARE BOTH SWEPT, AND BOTH HAVE TO BE.
        // `(marginal - n) % n` is zero only when `n` divides `2p + 9`, `p`
        // being the prefix width, and at the production `num_vars = 18` it
        // never does — a sweep held at 18 could not tell floor-plus-one from
        // `div_ceil` at all and would have been the check that cannot fail.
        for num_vars in 1..=40usize {
            for prefix in 1..=8usize {
                let n_fixed = num_vars + prefix;
                let marginal = marginal_stacked_rows(num_vars, n_fixed);
                if marginal > num_vars && (marginal - num_vars).is_multiple_of(num_vars) {
                    exact_divisions += 1;
                    if (marginal - num_vars).div_ceil(num_vars)
                        != candidate_threshold_entries(num_vars, n_fixed)
                    {
                        div_ceil_would_differ += 1;
                    }
                }
                for nonzero in 0..200 {
                    assert_eq!(
                        is_candidate(num_vars, n_fixed, nonzero),
                        sparse_leg_rows(num_vars, nonzero) > marginal,
                        "num_vars {num_vars}, bracket {n_fixed}, {nonzero} entries: the \
                         entry count and the row comparison disagree"
                    );
                }
            }
        }
        // The arms that make this a check rather than a restatement.
        //
        // ★ AND THE COVERAGE IS NOW A PROPERTY OF THE FORM RATHER THAN OF A
        // CHOSEN NUMBER, which is worth saying because it used to be the
        // opposite. `marginal - n = 4n + 2p + 9`, so an exact division is `n`
        // dividing `2p + 9`: over `p = 1..=8` that is the divisor count of
        // {11, 13, 15, 17, 19, 21, 23, 25} at heights under 41, which is 21
        // pairs. A literal gave three heights when it happened to be composite
        // and one when it happened to be prime — coverage that moved with a
        // number chosen for other reasons.
        println!("EXACT DIVISIONS in the sweep: {exact_divisions}");
        assert_eq!(
            exact_divisions, 21,
            "the divisor count of 2p + 9 over p = 1..=8 at heights under 41; if this \
             moved, a term of the form moved with it and the sweep is no longer \
             separating floor-plus-one from div_ceil where it thinks it is"
        );
        assert_eq!(
            div_ceil_would_differ, exact_divisions,
            "at every exact division `div_ceil` must give a DIFFERENT answer, or the \
             two forms were never actually separated"
        );
    }

    /// ★ CONSEQUENCE (a) — THE PRE-REGISTRATION, on a real plan over the
    /// block's thirty pages: exactly those three bases, in that order.
    #[test]
    fn the_threshold_selects_exactly_the_blocks_three_dense_pages() {
        let configs = block_genesis_configs();
        let plan = genesis_stack_plan(&configs, 15, BLOCK_PAGE_VARS);
        assert_eq!(plan.n_fixed, BLOCK_N_FIXED);

        // Part 1: the three, and only the three. The 27 zero pages cost 18 rows
        // sparse against a 103-row marginal, so carrying one would COST rows.
        assert_eq!(sparse_leg_rows(BLOCK_PAGE_VARS, 0), 18);
        assert!(!is_candidate(BLOCK_PAGE_VARS, plan.n_fixed, 0));
        assert_eq!(
            plan.routes.iter().filter(|route| route.candidate).count(),
            BLOCK_DENSE.len()
        );

        // Part 2: paid, by nearly two orders of magnitude.
        let expected: usize = BLOCK_DENSE
            .iter()
            .map(|&(_, _, rows)| rows - BLOCK_MARGINAL)
            .sum();
        assert_eq!(plan.savings, expected);
        assert_eq!(plan.savings, 10_248_237);
        assert!(chain_is_paid(plan.savings));

        // ⚠ THE SET AND ITS ORDER, not a count: the stack's column order IS
        // page-base order, so three of the wrong three would pass a count.
        let carried: Vec<u64> = plan
            .routes
            .iter()
            .filter(|route| route.dense)
            .map(|route| route.page_base)
            .collect();
        assert_eq!(carried, vec![0x0, 0x40000, 0x280000]);
        assert_eq!(plan.sparse_rows, BLOCK_ZERO_PAGES * 18);
    }

    /// The decision does not sit near either constant, which is what makes it a
    /// routing rule rather than a tuning knob.
    #[test]
    fn nothing_on_the_block_sits_near_the_threshold() {
        let least_dense = BLOCK_DENSE.iter().map(|&(_, _, r)| r).min().expect("three");
        // Part 1: the cheapest carried page clears the marginal 20,000-fold.
        assert!(least_dense > 10_000 * BLOCK_MARGINAL);

        // ⛔ AND THE ZERO PAGES' MARGIN IS A RATIO WITH A FLOOR, NOT A FIXED
        // FACTOR. This read `* 6 <`, which was true at 109, FALSE at 103 and
        // true again at the form's 111 — a margin stated as a multiple of a
        // number that moves is an assertion about a draft, and it reddened the
        // gate for exactly that reason. It is now the ratio it IS, printed,
        // floored at the weakest value any bracket gives: a LONE page's 101
        // still clears an all-zero page by five.
        let zero_page = sparse_leg_rows(BLOCK_PAGE_VARS, 0);
        let margin = BLOCK_MARGINAL / zero_page;
        println!(
            "ZERO-PAGE MARGIN: an all-zero page costs {zero_page} rows against a \
             {BLOCK_MARGINAL}-row marginal at the block's bracket ({LONE_MARGINAL} at a \
             lone page's) — a factor of {margin}"
        );
        assert!(
            margin >= 5,
            "an all-zero page costs {zero_page} rows against a marginal of \
             {BLOCK_MARGINAL}, within a factor of {margin} of qualifying — the 27 zero \
             pages are supposed to fail part 1 by a wide margin"
        );
        assert!(
            LONE_MARGINAL / zero_page >= 5,
            "the weakest bracket's marginal ({LONE_MARGINAL}) brings an all-zero page \
             within a factor of {} of qualifying",
            LONE_MARGINAL / zero_page
        );
        // Part 2: the CHEAPEST carried page is worth twelve chains on its own,
        // and the three together fifty-eight, so no plausible re-sizing of the
        // budget changes the block's answer.
        assert!(
            least_dense > 10 * PREPARED_LEG_ROWS,
            "the cheapest stacked page is {least_dense} rows against a \
             {PREPARED_LEG_ROWS} budget"
        );
    }

    /// ★ CONSEQUENCE (c) — A LONE PAGE, AT THE BOUNDARY BOTH WAYS.
    ///
    /// At one genesis page the two parts collapse to a single condition on that
    /// page: it is the only candidate, so the set's savings ARE its savings.
    /// That is the SHAPE of the rule this replaces — and not its value.
    ///
    /// ⛔⛔ THE BOUNDARY IS DERIVED FROM THE RUN'S OWN BRACKET AND IS NOT
    /// HARD-CODED HERE, and that is deliberate. It is
    /// `densest_sparse_entries()` and one more, and it MOVES one entry per
    /// `num_vars` rows of marginal. A LONE page stands at 19 variables and is
    /// charged 101, which puts it at (9,730 sparse, 9,731 dense) across a
    /// marginal of `[92, 109]`; a page inside the BLOCK's thirty stands at 24,
    /// is charged 111, and faces (9,731, 9,732) across `[110, 127]`. ⚠ BOTH
    /// PAIRS ARE REAL AND NEITHER IS "the" boundary — which is the reading the
    /// literal could not produce, since it charged a lone page the block's
    /// bracket. `the_ruled_boundaries_hold_for_every_marginal_in_band` states
    /// the bands.
    ///
    /// ⛔ THE RETIRED BREAK-EVEN WAS 9,725, and the entries between it and this
    /// boundary are why every bound derived from the old rule has to be
    /// re-derived rather than re-read: a test left at 9,724 stays green under
    /// both and means something under neither.
    #[test]
    fn a_lone_page_is_at_the_boundary_its_own_bracket_puts_it_at() {
        let sparse_at = densest_sparse_entries(BLOCK_PAGE_VARS, LONE_N_FIXED);
        let dense_at = sparse_at + 1;
        println!(
            "LONE BOUNDARY at bracket {LONE_N_FIXED} (marginal {LONE_MARGINAL}): \
             {sparse_at} sparse, {dense_at} dense; this pair holds across a marginal \
             of [92, 109], so there are {} rows of headroom above this bracket's \
             charge. At the block's bracket ({BLOCK_STACK_VARS}, marginal \
             {BLOCK_MARGINAL}) the same page would face ({}, {})",
            109 - LONE_MARGINAL,
            densest_sparse_entries(BLOCK_PAGE_VARS, BLOCK_STACK_VARS),
            densest_sparse_entries(BLOCK_PAGE_VARS, BLOCK_STACK_VARS) + 1
        );
        assert_eq!(
            (sparse_at, dense_at),
            (9_730, 9_731),
            "the pair at a LONE page's bracket. If it moved, read the marginal: this \
             pair holds across [92, 109], and the block's bracket puts it at \
             (9,731, 9,732) across [110, 127]"
        );
        // ⛔ THE CONTRAST, ASSERTED RATHER THAN DESCRIBED: the same entry count,
        // a different run, a different answer. A reader quoting one pair without
        // its bracket is quoting it for the wrong runs.
        assert_eq!(
            densest_sparse_entries(BLOCK_PAGE_VARS, BLOCK_STACK_VARS),
            9_731
        );

        for (nonzero, carried) in [(sparse_at, false), (dense_at, true)] {
            let configs = vec![data_page(0x40000, vec![1u8; nonzero])];
            let plan = genesis_stack_plan(&configs, 3, BLOCK_PAGE_VARS);
            // ★ At one page the plan stands at 19 variables and the marginal
            // is charged THERE — 101, not the block's 111. The form following
            // the run's shape is the whole of what it buys over a literal.
            assert_eq!(plan.n_fixed, LONE_N_FIXED);
            assert_eq!(plan.n_fixed, 19, "18 + ceil(log2(2))");
            assert_eq!(marginal_stacked_rows(BLOCK_PAGE_VARS, plan.n_fixed), 101);
            assert!(
                plan.routes[0].candidate,
                "a page of {nonzero} entries clears part 1 either way; only part 2 \
                 separates these two"
            );
            assert_eq!(
                plan.savings,
                page_savings(BLOCK_PAGE_VARS, plan.n_fixed, nonzero)
            );
            assert_eq!(
                !plan.is_empty(),
                carried,
                "a lone page of {nonzero} nonzero entries saves {} against a \
                 {PREPARED_LEG_ROWS}-row chain",
                plan.savings
            );
        }

        // ⛔ HOW MUCH MARGIN THE PAIR HAS, ASSERTED SO NOBODY REDISCOVERS IT.
        // The page just over the boundary clears the chain by NINE rows at this
        // bracket — so nine rows of marginal, half a prefix bit, would move it.
        assert_eq!(
            page_savings(BLOCK_PAGE_VARS, LONE_N_FIXED, dense_at),
            PREPARED_LEG_ROWS + 9
        );

        // The retired rule's break-even, kept here as the contrast and nowhere
        // else: `18 + 18*S > PREPARED_LEG_ROWS`, floor plus one.
        let retired = (PREPARED_LEG_ROWS - BLOCK_PAGE_VARS) / BLOCK_PAGE_VARS + 1;
        assert_eq!(retired, 9_725);
        assert!(
            retired < dense_at,
            "the two-part rule charges a lone page its own marginal on top of the \
             chain, so its boundary must sit ABOVE the retired one ({retired} vs \
             {dense_at})"
        );
    }

    /// ★ CONSEQUENCE (b) — A CANDIDATE THE CHAIN REFUSES TO PAY FOR, which is
    /// where every fixture in the tree sits.
    ///
    /// `data_page_touch`'s one data page carries 112 nonzero genesis bytes. It
    /// passes part 1 comfortably — 2,034 rows against a 111-row marginal — and
    /// part 2 refuses it, because 1,923 saved rows do not buy a 175,066-row
    /// chain. ⚠ THAT IS THE INTERESTING BRANCH AND IT IS WHY THE FIELD
    /// `candidate` EXISTS: under the retired rule this page failed the only
    /// test there was, and "no opening" meant "nobody wanted it".
    #[test]
    fn a_candidate_the_chain_cannot_be_paid_for_stays_sparse() {
        let configs = genesis_pages_with(&[112]);
        let plan = genesis_stack_plan(&configs, 3, BLOCK_PAGE_VARS);
        assert_eq!(plan.n_fixed, BLOCK_N_FIXED);
        assert_eq!(sparse_leg_rows(BLOCK_PAGE_VARS, 112), 2_034);
        assert!(plan.routes[0].candidate, "2,034 rows against the marginal");
        // ⛔ ONE DERIVATION, AND THE VALUE IN THE MESSAGE. This line used to be
        // followed by `assert_eq!(plan.savings, 1_931)` — the same quantity a
        // second time, as a bare literal computed under the retired 103. It
        // named no symbol, so the sweep that re-pointed every reference to the
        // marginal could not see it, and the form moved the savings to 1,923
        // underneath it. ⇒ a number a reader wants is a MESSAGE, never a second
        // assertion: the second one drifts, and it drifts silently until the
        // day the first one moves.
        assert_eq!(
            plan.savings,
            2_034 - BLOCK_MARGINAL,
            "the sparse leg pinned just above (2,034) less what the block's bracket \
             charges a carried page ({BLOCK_MARGINAL}), which is {} rows",
            plan.savings
        );
        assert!(!chain_is_paid(plan.savings));
        assert!(
            plan.is_empty(),
            "the chain is not paid, so nothing is carried"
        );
        assert!(!plan.routes[0].dense);
        assert_eq!(
            plan.sparse_rows,
            2_034 + (BLOCK_GENESIS_PAGES - 1) * 18,
            "a refused candidate still pays its sparse leg, and so do the zero pages"
        );

        // And the dense guest written for this route clears part 2 outright:
        // `dense_data_page_touch` surrounds its cell with 32 KiB either side.
        let dense = genesis_stack_plan(&genesis_pages_with(&[65_652]), 3, BLOCK_PAGE_VARS);
        assert_eq!(dense.savings, 18 + 18 * 65_652 - BLOCK_MARGINAL);
        assert!(chain_is_paid(dense.savings));
        assert_eq!(dense.dense_pages(), vec![0]);
    }

    /// ★ CONSEQUENCE (d) — TWO PAGES THAT SHARE ONE CHAIN, the case the retired
    /// rule got wrong.
    ///
    /// Neither page is worth a chain of its own; between them they are worth
    /// two. A rule that charges the chain per page refuses both and spends
    /// 180,036 rows keeping them sparse — more than the chain it declined to
    /// buy. That is not a tuning miss, it is a fixed cost charged to the wrong
    /// thing, and part 2 is where it moves.
    #[test]
    fn two_pages_worth_less_than_the_chain_apiece_share_one() {
        let configs = genesis_pages_with(&[5_000, 5_000]);
        let plan = genesis_stack_plan(&configs, 3, BLOCK_PAGE_VARS);
        assert_eq!(plan.n_fixed, BLOCK_N_FIXED);
        assert_eq!(sparse_leg_rows(BLOCK_PAGE_VARS, 5_000), 90_018);

        // Neither one pays for the chain alone.
        let alone = page_savings(BLOCK_PAGE_VARS, plan.n_fixed, 5_000);
        assert_eq!(alone, 89_907);
        assert!(!chain_is_paid(alone));
        // Together they do, and BOTH are carried — the set is paid for or none
        // of it is.
        //
        // ⛔ ONE DERIVATION, AND THE SUM IS THE SECOND PLACE THIS BIT. A bare
        // `assert_eq!(plan.savings, 179_830)` stood here — `2 × 89,915`, the
        // retired constant's `alone` DOUBLED. The by-value sweep that caught
        // the same shape at the refused candidate looked for the BASE
        // quantities and not their MULTIPLES, so re-pointing `alone` from
        // 89,915 to 89,907 left its double untouched one line below. ⇒ a sweep
        // over a moved form must cover SUMS AND PRODUCTS of what moved, not
        // only the terms themselves.
        assert_eq!(
            plan.savings,
            2 * alone,
            "two candidates of {} rows each between them save {} against a \
             {PREPARED_LEG_ROWS}-row chain",
            alone,
            plan.savings
        );
        assert!(chain_is_paid(plan.savings));
        assert_eq!(plan.dense_pages(), vec![0, 1]);

        // What the retired rule spent instead.
        assert_eq!(2 * sparse_leg_rows(BLOCK_PAGE_VARS, 5_000), 180_036);
    }

    fn data_page(base: u64, bytes: Vec<u8>) -> PageConfig {
        PageConfig::with_data(base, bytes)
    }

    /// ⛔ THE PRIVATE FILTER, adversarially: a private-input page whose genesis
    /// bytes WOULD cross the threshold is still not stacked.
    ///
    /// This is the arm that would have caught the prover and the verifier
    /// selecting different sets. On the prover a private page carries the
    /// private input; on the verifier it carries an empty vec.
    #[test]
    fn a_private_page_dense_enough_to_qualify_is_still_not_stacked() {
        let dense_bytes = vec![0xABu8; 20_000];
        // ⚠ THE PRECONDITION: these bytes must be carried when they are PUBLIC,
        // or the test passes on a page nobody would have stacked anyway.
        // ⚠ AT THIS RUN'S OWN BRACKET, which is a ONE-genesis-page stack: the
        // private page is filtered out before the count is taken, so the
        // precondition has to be evaluated where the plan evaluates it.
        assert!(is_candidate(
            BLOCK_PAGE_VARS,
            LONE_N_FIXED,
            dense_bytes.len()
        ));
        assert!(chain_is_paid(page_savings(
            BLOCK_PAGE_VARS,
            LONE_N_FIXED,
            dense_bytes.len()
        )));
        let mut private = data_page(0xff000000, dense_bytes.clone());
        private.is_private_input = true;
        let public = data_page(0x40000, dense_bytes);

        let plan = genesis_stack_plan(&[private, public], 3, BLOCK_PAGE_VARS);
        assert_eq!(
            plan.at,
            stark::multilinear_table::leading_columns(4, PAGE_PREPROCESSED_COLUMNS),
            "only the non-private page may be stacked, at its own AIR-set index, \
             and with BOTH of its preprocessed columns"
        );
        assert!(!plan.routes[0].dense);
        assert!(!plan.routes[0].has_init);
        assert!(plan.routes[1].dense);
    }

    /// The same page set with the private page's bytes REMOVED — the verifier's
    /// view — plans identically. That is the property the two sides need, and
    /// the reason the filter is on `is_private_input` and not on the bytes.
    #[test]
    fn the_prover_and_verifier_views_of_a_private_page_plan_alike() {
        let dense_bytes = vec![0xABu8; 20_000];
        let mut prover_side = data_page(0xff000000, dense_bytes.clone());
        prover_side.is_private_input = true;
        let mut verifier_side = data_page(0xff000000, Vec::new());
        verifier_side.is_private_input = true;
        let public = data_page(0x40000, dense_bytes);

        let from_prover = genesis_stack_plan(&[prover_side, public.clone()], 3, BLOCK_PAGE_VARS);
        let from_verifier = genesis_stack_plan(&[verifier_side, public], 3, BLOCK_PAGE_VARS);
        assert_eq!(
            from_prover.at, from_verifier.at,
            "prover and verifier must commit the same stack or the roots block diverges"
        );
        assert_eq!(from_prover.sparse_rows, from_verifier.sparse_rows);
        // ⛔ EVERY INPUT TO THE RULE, NOT JUST ITS ANSWER. The two-part rule
        // reads a page count and a savings total as well as each page's bytes,
        // and two sides that agreed on the set while disagreeing on either
        // would be agreeing by luck. `n_fixed` is the page count's fingerprint
        // and `savings` the candidate set's.
        assert_eq!(
            from_prover.n_fixed, from_verifier.n_fixed,
            "the fixed height is taken at the genesis page count, which the private \
             filter must make identical on both sides"
        );
        assert_eq!(from_prover.savings, from_verifier.savings);
        assert_eq!(
            from_prover.routes, from_verifier.routes,
            "including each page's candidacy, which is part 1's answer"
        );
    }

    /// The table index a stacked column names is the AIR set's, bookends
    /// included — not the page's index in its own family. And the columns come
    /// out in stack order: page by page, each page's prefix in index order.
    #[test]
    fn the_stacked_columns_name_their_index_in_the_whole_air_set_in_stack_order() {
        let dense = vec![0x01u8; 20_000];
        let configs = vec![
            PageConfig::zero_init(0x0),
            data_page(0x40000, dense.clone()),
            PageConfig::zero_init(0x80000),
            data_page(0x280000, dense),
        ];
        let plan = genesis_stack_plan(&configs, 15, BLOCK_PAGE_VARS);
        let mut want = stark::multilinear_table::leading_columns(16, PAGE_PREPROCESSED_COLUMNS);
        want.extend(stark::multilinear_table::leading_columns(
            18,
            PAGE_PREPROCESSED_COLUMNS,
        ));
        assert_eq!(
            plan.at, want,
            "fifteen bookends precede the pages, so pages 1 and 3 are tables 16 and 18"
        );
        assert_eq!(plan.dense_pages(), vec![1, 3]);
        // The two zero pages still cost the interned zero apiece.
        assert_eq!(plan.sparse_rows, 2 * sparse_leg_rows(BLOCK_PAGE_VARS, 0));

        // ⚠ AND THE COLUMNS AGREE WITH `at` IN LENGTH AND IN ORDER. A stack
        // whose columns and whose destinations disagreed would settle one page's
        // values against another's commitment.
        let columns = genesis_stack_columns(&configs, &plan);
        assert_eq!(columns.len(), plan.at.len());
        assert_eq!(
            columns[0],
            page::preprocessed_columns(&configs[1])[0],
            "stack column 0 is dense page 1's OFFSET"
        );
        assert_eq!(
            columns[1],
            page::preprocessed_columns(&configs[1])[1],
            "stack column 1 is dense page 1's INIT"
        );
        assert_eq!(
            columns[2],
            page::preprocessed_columns(&configs[3])[0],
            "stack column 2 is dense page 3's OFFSET"
        );
    }

    /// A run whose genesis is entirely sparse carries no opening at all, and its
    /// cross-epoch proof is the one it was before this route existed.
    ///
    /// ⚠ AND IT SAYS WHICH PART REFUSED. The data page IS a candidate here —
    /// part 2 is what leaves the set empty — so a green would otherwise be
    /// consistent with a part 1 that had stopped working.
    #[test]
    fn an_all_sparse_page_set_stacks_nothing() {
        let configs = vec![
            PageConfig::zero_init(0x0),
            data_page(0x40000, vec![1u8; 112]),
        ];
        let plan = genesis_stack_plan(&configs, 3, BLOCK_PAGE_VARS);
        assert!(plan.is_empty());
        assert!(plan.dense_pages().is_empty());
        assert!(genesis_stack_columns(&configs, &plan).is_empty());
        assert!(!plan.routes[0].candidate, "an all-zero page fails part 1");
        assert!(plan.routes[1].candidate, "112 entries clear part 1");
        assert!(
            !chain_is_paid(plan.savings),
            "part 2 is what empties this set, and its savings are {}",
            plan.savings
        );
    }

    /// ⛔⛔ HOW MUCH OF THIS RULE DEPENDS ON THE MARGINAL'S EXACT VALUE —
    /// EXECUTED, not asserted to be small.
    ///
    /// [`marginal_stacked_rows`] is a form now, so the question is no longer
    /// "what will the pin say" but "which bracket is this run in": the same
    /// form charges 101 at one genesis page, 111 at the block's thirty and 125
    /// at a run of 4,096. So "the exact value is immaterial" is a claim that
    /// has to be READ over the whole range a run can put it in, one consequence
    /// at a time.
    ///
    /// ★ AND IT IS HOW THE MOVING QUANTITIES ANNOUNCE THEMSELVES. Two of the
    /// five do move, and the bands are asserted rather than the values:
    ///
    /// | consequence | holds for a marginal in |
    /// |---|---|
    /// | the block's three pages, paid | `[19, 2_100_474]` |
    /// | the fixture (S = 112) refused by part 2 | every marginal |
    /// | two pages of 5,000 share one chain | `[1, 2_484]` |
    /// | the lone-page pair (9,730, 9,731) | `[92, 109]` ⚠ |
    /// | `τ = 5` | `[90, 107]` ⚠ — the block's bracket has 6 |
    ///
    /// ⚠ AND THE BRACKETS ARE WALKED THROUGH THOSE BANDS, which is the arm a
    /// literal could not have: `τ` is 5 up to `n_fixed` 22 and 6 from 23, so
    /// the rule's threshold depends on the RUN'S PAGE COUNT — nine genesis
    /// pages or more and a page needs six nonzero bytes, eight or fewer and it
    /// needs five. The day a bracket takes a quantity out of its band, this
    /// goes red naming the band and the quantity, and the doc at
    /// [`densest_sparse_entries`] says what to write instead.
    #[test]
    fn the_ruled_boundaries_hold_for_every_marginal_in_band() {
        // The forms with the marginal as a PARAMETER. ⚠ They are restated here
        // and nowhere else, and the assertions below tie them back to the live
        // functions at the shipped literal, so a restatement that drifted from
        // the code would be caught rather than swept.
        let savings = |m: usize, s: usize| sparse_leg_rows(BLOCK_PAGE_VARS, s).saturating_sub(m);
        let tau = |m: usize| m.saturating_sub(BLOCK_PAGE_VARS) / BLOCK_PAGE_VARS + 1;
        let densest = |m: usize| (PREPARED_LEG_ROWS + m - BLOCK_PAGE_VARS) / BLOCK_PAGE_VARS;
        let m0 = BLOCK_MARGINAL;
        assert_eq!(
            tau(m0),
            candidate_threshold_entries(BLOCK_PAGE_VARS, BLOCK_STACK_VARS)
        );
        assert_eq!(
            densest(m0),
            densest_sparse_entries(BLOCK_PAGE_VARS, BLOCK_STACK_VARS)
        );
        assert_eq!(
            savings(m0, 112),
            page_savings(BLOCK_PAGE_VARS, BLOCK_STACK_VARS, 112)
        );

        // THE TWO THAT DO NOT MOVE ANYWHERE NEAR HERE.
        for m in 90..=130usize {
            assert!(
                !chain_is_paid(savings(m, 112)),
                "at marginal {m} the fixture's 112-entry page would buy a chain"
            );
            assert!(
                chain_is_paid(2 * savings(m, 5_000)) && !chain_is_paid(savings(m, 5_000)),
                "at marginal {m} the two 5,000-entry pages no longer share exactly one chain"
            );
            let block: usize = BLOCK_DENSE.iter().map(|&(_, _, r)| r - m).sum();
            assert!(
                chain_is_paid(block),
                "at marginal {m} the block would not pay"
            );
            assert!(
                sparse_leg_rows(BLOCK_PAGE_VARS, 0) <= m,
                "at marginal {m} an all-zero page would become a candidate"
            );
        }

        // ★ AND THE BLOCK'S SET IS STABLE ACROSS FIVE ORDERS OF MAGNITUDE,
        // which is the argument that this is a routing rule and not a knob: any
        // marginal from the cost of one zero page up to the cheapest dense
        // page's own leg selects exactly those three.
        for m in [19usize, 109, 1_000, 100_000, 2_100_473] {
            let block: usize = BLOCK_DENSE.iter().map(|&(_, _, r)| r - m).sum();
            assert!(chain_is_paid(block) && sparse_leg_rows(BLOCK_PAGE_VARS, 0) <= m);
        }

        // ⚠ THE TWO THAT MOVE, asserted as BANDS so the move is legible.
        let band = |f: &dyn Fn(usize) -> bool| {
            let hits: Vec<usize> = (1..=3_000usize).filter(|&m| f(m)).collect();
            let (lo, hi) = (hits[0], hits[hits.len() - 1]);
            assert_eq!(
                hits,
                (lo..=hi).collect::<Vec<_>>(),
                "a band that is not contiguous is not a band"
            );
            (lo, hi)
        };
        let tau_is_five = band(&|m| tau(m) == 5);
        let lone_pair =
            band(&|m| !chain_is_paid(savings(m, 9_730)) && chain_is_paid(savings(m, 9_731)));
        println!(
            "MARGINAL BANDS at the block's bracket ({BLOCK_STACK_VARS}, marginal {m0}): \
             τ = 5 for {tau_is_five:?} (τ here is {}), the (9730, 9731) pair for \
             {lone_pair:?}, densest sparse {}",
            tau(m0),
            densest(m0)
        );
        // ⛔ THE BANDS ARE COMPLETE AT THE TOP, WHICH THEY WERE NOT. τ was
        // tabled as "5 on [90, 107], 6 from 108" and then asserted at m = 130,
        // where it is 7 — an open-ended "from" is not a band, and the sweep ran
        // past the end of the one that had been written down. All three
        // sub-bands are asserted here, with both of their edges.
        assert_eq!(tau_is_five, (90, 107));
        assert_eq!(band(&|m| tau(m) == 6), (108, 125));
        assert_eq!(band(&|m| tau(m) == 7), (126, 143));
        assert_eq!(lone_pair, (92, 109));
        assert_eq!(band(&|m| densest(m) == 9_731), (110, 127));
        assert_eq!(band(&|m| densest(m) == 9_732), (128, 145));

        // ⛔ AND WHERE THE LITERAL FALLS IS COMPUTED, NOT HARD-CODED. Every
        // earlier version named the band it expected, so each time the literal
        // moved this reddened on the naming rather than on the finding. It now
        // asserts only that the literal lies inside the band for ITS OWN value
        // — true at the current 103, at the coming form's 111, and at the 113 a
        // taller bracket costs — and PRINTS which bands those are.
        let my_tau = tau(m0);
        let my_pair = densest(m0);
        assert_eq!(
            my_tau,
            candidate_threshold_entries(BLOCK_PAGE_VARS, BLOCK_STACK_VARS)
        );
        assert_eq!(
            my_pair,
            densest_sparse_entries(BLOCK_PAGE_VARS, BLOCK_STACK_VARS)
        );
        let tau_band = band(&|m| tau(m) == my_tau);
        let pair_band = band(&|m| densest(m) == my_pair);
        println!(
            "THE FORM AT {BLOCK_STACK_VARS} CHARGES {m0}: τ = {my_tau} over \
             {tau_band:?}, pair ({my_pair}, {}) over {pair_band:?}",
            my_pair + 1
        );
        assert!(tau_band.0 <= m0 && m0 <= tau_band.1);
        assert!(pair_band.0 <= m0 && m0 <= pair_band.1);

        // ⛔⛔ THE WHOLE SPREAD AT ONE BRACKET, EXECUTED. The per-page marginal
        // is not one number even there: the threaded sponge makes it 109, 110
        // or 111 by position, and the form charges the dearest — which is why
        // `m0` is the TOP of that spread rather than a reading of one position.
        // Walking every member means the table in the form's doc is a reading
        // and not a claim.
        assert_eq!(
            m0, 111,
            "the form charges the dearest position in the block bracket's spread"
        );
        for m in [109usize, 110, 111] {
            assert_eq!(tau(m), 6, "τ is 6 across the whole spread");
        }
        assert_eq!(
            densest(109),
            9_730,
            "the cheapest position keeps today's pair"
        );
        for m in [110usize, 111] {
            assert_eq!(densest(m), 9_731);
        }
        // ⇒ AND THAT IS WHY THE FORM TAKES THE MAXIMUM RATHER THAN A MIDPOINT:
        // τ is invariant across the spread, so the only quantity the choice
        // decides is the pair, and charging the dearest position is
        // conservative everywhere for the price of one entry.

        // ★★ THE BRACKETS THEMSELVES, WALKED — the arm a literal could not
        // have. Every genesis page count from one page to four thousand, the
        // marginal the form charges there, and the two derived quantities read
        // at each. ⚠ τ IS NOT INVARIANT ACROSS THEM, and that is the finding
        // rather than a defect: it is 5 while the marginal is under 108 and 6
        // from there, so the crossing is a property of the RUN's page count.
        // (It reaches 7 at 126, which no reachable page count produces: 2^393
        // thousand pages.)
        let mut crossings = 0usize;
        let mut previous: Option<usize> = None;
        for pages in [1usize, 2, 4, 8, 9, 16, 30, 64, 256, 4_096] {
            let n_fixed = fixed_stack_vars(BLOCK_PAGE_VARS, pages);
            let m = marginal_stacked_rows(BLOCK_PAGE_VARS, n_fixed);
            let t = candidate_threshold_entries(BLOCK_PAGE_VARS, n_fixed);
            println!(
                "BRACKET {pages} pages -> n_fixed {n_fixed}, marginal {m}, τ {t}, \
                 densest sparse {}",
                densest_sparse_entries(BLOCK_PAGE_VARS, n_fixed)
            );
            assert_eq!(t, tau(m), "the closure and the live function must agree");
            assert!(
                (5..=6).contains(&t),
                "τ left the two values every bracket up to 4,096 genesis pages produces"
            );
            if previous.is_some_and(|p| p != t) {
                crossings += 1;
            }
            previous = Some(t);
        }
        assert_eq!(
            crossings, 1,
            "τ crosses once over this range, between eight and nine genesis pages; a \
             second crossing means a term of the form is not monotone in the bracket"
        );
        // The crossing, named: eight pages charge 107 and nine charge 109.
        assert_eq!(
            candidate_threshold_entries(BLOCK_PAGE_VARS, fixed_stack_vars(BLOCK_PAGE_VARS, 8)),
            5
        );
        assert_eq!(
            candidate_threshold_entries(BLOCK_PAGE_VARS, fixed_stack_vars(BLOCK_PAGE_VARS, 9)),
            6
        );
        // ⚠ AND THE PAIR MOVES WITH THE BRACKET TOO, one entry at a time: a
        // lone page's 9,730 against the block's 9,731. Both are inside the
        // bands above, which is why neither moves a ruled consequence.
        assert_eq!(densest_sparse_entries(BLOCK_PAGE_VARS, LONE_N_FIXED), 9_730);
        assert_eq!(
            densest_sparse_entries(BLOCK_PAGE_VARS, BLOCK_STACK_VARS),
            9_731
        );

        // ⚠ THE 113 A TALLER RUN COSTS, which the form charges rather than
        // understating: a stack at `n_stack` 25 reaches 113 a page. It moves
        // NEITHER quantity, both bands reaching past it.
        assert_eq!(
            marginal_stacked_rows(BLOCK_PAGE_VARS, BLOCK_STACK_VARS + 1),
            113
        );
        assert_eq!(tau(113), 6);
        assert_eq!(densest(113), 9_731);
    }
}
