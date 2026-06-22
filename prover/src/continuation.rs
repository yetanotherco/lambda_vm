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
//! quantities (epoch, init-timestamp) are built from `IsHalfword`-checked
//! halfwords. Address and fini-timestamp need no extra check — they are matched
//! against MEMW on the epoch-local Memory bus, exactly as PAGE relies on MEMW.
//! The global proof commits the identical trace, so it inherits the guarantee
//! via the commitment binding.
//!
//! Cross-epoch registers are bound the same way: each continuation epoch
//! preprocesses its REGISTER `FINI` column to the epoch's final register file
//! `R_{i+1}` (alongside `INIT = R_i`), and the driver reuses the same `R_{i+1}`
//! as the next epoch's preprocessed `INIT` — so `init(epoch i+1) == fini(epoch i)`
//! by construction, with the REG-C2 Memory bus binding `FINI` to the true final
//! registers. No extra bus. Still deferred: statement/Fiat-Shamir binding of the
//! per-epoch transcripts (only needed for a split prover/verifier) and the x254
//! commit-index across epoch boundaries.

use std::collections::HashMap;

use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use executor::elf::Elf;
use executor::vm::execution::Executor;
use executor::vm::logs::Log;
use math::field::element::FieldElement;
use stark::config::Commitment;
use stark::lookup::{AirWithBuses, AuxiliaryTraceBuildData, NullBoundaryConstraintBuilder};
use stark::proof::options::ProofOptions;
use stark::proof::stark::MultiProof;
use stark::prover::{IsStarkProver, Prover};
use stark::trace::TraceTable;
use stark::traits::AIR;
use stark::verifier::{IsStarkVerifier, Verifier};

use crate::paged_mem::PagedMem;
use crate::tables::local_to_global::{self, CellBoundary};
use crate::tables::page::{self, PageConfig};
use crate::tables::register;
use crate::tables::trace_builder::{
    Traces, build_init_page_data, build_initial_image, build_initial_image_paged,
    epoch_touched_cells,
};
use crate::tables::types::{GoldilocksExtension, GoldilocksField};
use crate::tables::{MaxRowsConfig, global_memory};
use crate::{Error, VmAirs, compute_expected_commit_bus_balance, verify_l2g_commitment_binding};

type F = GoldilocksField;
type E = GoldilocksExtension;
type AirRef<'a> = &'a dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>;

fn empty_constraints()
-> Vec<Box<dyn stark::constraints::transition::TransitionConstraintEvaluator<F, E>>> {
    vec![]
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
    let image = build_initial_image(elf, private_inputs);
    let init_page_data = build_init_page_data(&image);
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

/// Per-epoch starting state: the memory image and register image the epoch begins from.
/// `image` is borrowed from the persistent cross-epoch image (init = previous fini), so
/// it is not re-snapshotted or cloned per epoch.
struct EpochStart<'a> {
    image: &'a PagedMem<u8>,
    register_init: HashMap<u64, u32>,
    is_first: bool,
    /// This epoch's 1-based table label (the `fini_epoch` constant).
    label: u64,
}

/// A successful epoch proof's outputs:
/// - the L2G commitment root the global proof binds against (cross-epoch memory), and
/// - the epoch's final register file `R_{i+1}`: the 67 final register values in
///   `register_word_address_list` order (read from the committed REGISTER trace).
///
/// The driver feeds `R_{i+1}` to the next epoch as its preprocessed INIT — that,
/// plus this epoch's preprocessed FINI commitment over the same `R_{i+1}`, binds
/// `init(epoch i+1) == fini(epoch i)`.
type EpochOutput = (Commitment, Vec<u32>);

/// Prove and verify one epoch, committing its local-to-global table (built from
/// `boundary`) on the epoch-local Memory bus, and its REGISTER table with FINI
/// preprocessed to the epoch's final register file. Returns the L2G commitment
/// root and that final register file if the epoch verifies, or `None` if not.
fn prove_verify_epoch(
    elf: &Elf,
    start: &EpochStart,
    logs: &[Log],
    is_final: bool,
    boundary: &[CellBoundary],
    private_inputs: &[u8],
    opts: &ProofOptions,
) -> Result<Option<EpochOutput>, Error> {
    let mut traces = Traces::from_image_and_logs(
        elf,
        start.image,
        &start.register_init,
        logs,
        &MaxRowsConfig::default(),
        private_inputs,
        is_final,
        true,
        #[cfg(feature = "disk-spill")]
        stark::storage_mode::StorageMode::Ram,
    )?;

    // Use the cross-epoch boundary so this epoch's L2G table is identical to the
    // one the global proof commits (the commitment binding compares their roots).
    // Its init value equals the epoch-start value either way, so the epoch-local
    // Memory bus still balances.
    traces.local_to_global = local_to_global::generate_local_to_global_trace(boundary);

    // Count this L2G table's range-check lookups into the (full, untrimmed)
    // BITWISE table so its AreBytes/IsHalfword multiplicities balance the
    // range-check senders in `l2g_memory_air`. Must use the same `boundary` the
    // committed L2G trace was built from.
    crate::tables::bitwise::update_multiplicities(
        &mut traces.bitwise,
        &local_to_global::collect_bitwise_from_l2g(boundary),
    );

    // The epoch's final register file R_{i+1}, read from the committed REGISTER
    // trace (its FINI column, which the Memory bus binds to the true last write).
    let reg_fini = register::fini_from_trace(&traces.register);

    let table_counts = traces.table_counts();
    let register_init_arg = if start.is_first {
        None
    } else {
        Some(&start.register_init)
    };
    let mut airs = VmAirs::new(
        elf,
        opts,
        false,
        &traces.page_configs,
        &table_counts,
        None,
        is_final,
        register_init_arg,
        None,
    );

    // Continuation epochs preprocess FINI = R_{i+1} too (not just INIT = R_i), so
    // the epoch's final register file is a verifier-known public value bound by the
    // REG-C2 Memory-bus token. The driver reuses this same R_{i+1} as the next
    // epoch's preprocessed INIT, binding init(epoch i+1) == fini(epoch i) with no
    // extra bus. Built from the same (R_i, R_{i+1}) the REGISTER trace holds, so it
    // matches the committed preprocessed columns.
    airs.register = crate::test_utils::create_register_air(opts).with_preprocessed(
        register::compute_precomputed_commitment_with_fini(opts, &start.register_init, &reg_fini),
        register::NUM_PREPROCESSED_COLS_WITH_FINI,
    );

    let l2g_air = l2g_memory_air(opts, start.label);
    let mut l2g_trace = std::mem::replace(
        &mut traces.local_to_global,
        local_to_global::generate_local_to_global_trace(&[]),
    );

    let mut pairs = airs.air_trace_pairs(&mut traces);
    pairs.push((&l2g_air, &mut l2g_trace, &()));
    let proof = Prover::multi_prove(
        pairs,
        &mut DefaultTranscript::<E>::new(&[]),
        #[cfg(feature = "disk-spill")]
        stark::storage_mode::StorageMode::Ram,
    )
    .map_err(|e| Error::Prover(format!("{e:?}")))?;

    let mut refs = airs.air_refs();
    refs.push(&l2g_air);
    let mut replay = DefaultTranscript::<E>::new(&[]);
    let expected = compute_expected_commit_bus_balance(
        &refs,
        &proof,
        &traces.public_output_bytes,
        &mut replay,
    )
    .ok_or_else(|| Error::Prover("commit bus fingerprint collision".into()))?;

    if !Verifier::multi_verify(
        &refs,
        &proof,
        &mut DefaultTranscript::<E>::new(&[]),
        &expected,
    ) {
        return Ok(None);
    }
    Ok(Some((
        proof.proofs.last().unwrap().lde_trace_main_merkle_root,
        reg_fini,
    )))
}

/// Build the cross-epoch global memory proof: every epoch's L2G sub-table on the
/// GlobalMemory bus, plus one GLOBAL_MEMORY table per touched page that sends each
/// cell's genesis init (preprocessed from the ELF, so the verifier recomputes it)
/// and receives its final value. The bus balances iff every `fini` matches the next
/// epoch's `init` and every genesis value matches the ELF.
fn prove_global(
    boundaries: &[Vec<CellBoundary>],
    elf: &Elf,
    private_inputs: &[u8],
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
                    timestamp: b.fini.timestamp,
                },
            );
        }
    }

    let gm_configs = global_memory_configs(boundaries, elf, private_inputs);

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
        &mut DefaultTranscript::<E>::new(&[]),
        #[cfg(feature = "disk-spill")]
        stark::storage_mode::StorageMode::Ram,
    )
    .map_err(|e| Error::Prover(format!("{e:?}")))
}

fn verify_global(
    boundaries: &[Vec<CellBoundary>],
    proof: &MultiProof<F, E, ()>,
    elf: &Elf,
    private_inputs: &[u8],
    opts: &ProofOptions,
) -> bool {
    // One L2G air per epoch, each with its own 1-based `fini_epoch` constant —
    // must match the order/labels the global proof committed in `prove_global`.
    let l2g_airs: Vec<_> = (0..boundaries.len())
        .map(|i| l2g_global_air(opts, local_to_global::epoch_label(i as u64)))
        .collect();
    // Rebuild the genesis configs FROM THE ELF and recompute their commitments:
    // this is the binding — a prover that claimed different genesis values would
    // commit a different root and fail to verify.
    let gm_configs = global_memory_configs(boundaries, elf, private_inputs);
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
        &mut DefaultTranscript::<E>::new(&[]),
        &FieldElement::zero(),
    )
}

/// Prove and verify a full continuation: split the execution into epochs of
/// `epoch_size` cycles, prove+verify each epoch, prove+verify the cross-epoch
/// global memory linkage, and check that each epoch proof committed the same
/// local-to-global table the global proof used. Returns `Ok(true)` iff all hold.
pub fn prove_and_verify_continuation(
    elf_bytes: &[u8],
    private_inputs: &[u8],
    epoch_size: usize,
) -> Result<bool, Error> {
    let elf = Elf::load(elf_bytes).map_err(|e| Error::ElfLoad(format!("{e}")))?;
    let mut executor = Executor::new(&elf, private_inputs.to_vec())
        .map_err(|e| Error::Execution(format!("{e}")))?;

    // The cross-epoch memory image, carried forward across epochs: epoch i+1's init is
    // epoch i's fini, so it is updated in place with each epoch's touched-cell final
    // values rather than re-snapshotted from the executor every epoch.
    let mut image = build_initial_image_paged(&elf, private_inputs);
    let initial_memory: HashMap<u64, u64> = image.iter().map(|(a, v)| (a, v as u64)).collect();

    // Running cross-epoch provenance (the L2G init source). Only the sparse
    // boundaries and the per-epoch roots are kept — everything else is dropped
    // after each epoch is proven (the streaming/eviction the spec describes).
    let mut provenance = local_to_global::genesis_provenance(&initial_memory);
    let mut all_boundaries: Vec<Vec<CellBoundary>> = Vec::new();
    let mut epoch_roots: Vec<Commitment> = Vec::new();
    // The previous epoch's bound final register file (its REGISTER FINI, read back
    // via `fini_from_trace` as the 67 values in `register_word_address_list` order).
    // Epoch i+1's register init is sourced from it — and its preprocessed INIT
    // commitment is built from it — rather than from a trusted executor snapshot.
    // This is the cross-epoch register binding: the same R_{i+1} is epoch i's
    // preprocessed FINI and epoch i+1's preprocessed INIT.
    let mut prev_fini: Option<Vec<u32>> = None;
    let opts = ProofOptions::default_test_options();

    let mut index: u64 = 0;
    loop {
        let start_pc = executor.pc();
        if start_pc == 0 {
            break;
        }
        let register_init = if index == 0 {
            register::register_init_from_entry_point(elf.entry_point)
        } else {
            // Expand the previous epoch's bound fini vector into the address-keyed
            // init map the trace builder consumes (same R_{i+1} bytes).
            register::register_init_from_fini(
                prev_fini
                    .as_ref()
                    .expect("prev_fini is set after the first epoch"),
            )
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

        // `image` is this epoch's starting memory (the previous epoch's fini).
        // Epoch tables are labelled 1-based (genesis is 0), so the ordering check
        // `init_epoch < fini_epoch` holds for genesis-origin cells.
        let label = local_to_global::epoch_label(index);
        let touched = epoch_touched_cells(&elf, &image, &logs)?;
        let boundary = local_to_global::epoch_boundary(&mut provenance, label, &touched);

        let start = EpochStart {
            image: &image,
            register_init,
            is_first: index == 0,
            label,
        };
        match prove_verify_epoch(
            &elf,
            &start,
            &logs,
            is_final,
            &boundary,
            private_inputs,
            &opts,
        )? {
            Some((root, reg_fini)) => {
                epoch_roots.push(root);
                prev_fini = Some(reg_fini);
            }
            None => return Ok(false),
        }

        // Carry the image forward: this epoch's fini is the next epoch's init.
        for cell in &boundary {
            image.set(cell.address, (cell.fini.value & 0xFF) as u8);
        }
        all_boundaries.push(boundary);
        // `start`, `logs`, and this epoch's traces are dropped here.

        if is_final {
            break;
        }
        index += 1;
    }

    // One global LogUp over all the (kept) local-to-global tables.
    let global_proof = prove_global(&all_boundaries, &elf, private_inputs, &opts)?;
    if !verify_global(&all_boundaries, &global_proof, &elf, private_inputs, &opts) {
        return Ok(false);
    }

    Ok(verify_l2g_commitment_binding(&epoch_roots, &global_proof))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::asm_elf_bytes;

    #[test]
    fn test_prove_and_verify_continuation() {
        let _ = env_logger::builder().is_test(true).try_init();
        let elf_bytes = asm_elf_bytes("all_loadstore_32");
        let elf = Elf::load(&elf_bytes).unwrap();
        let total = Executor::new(&elf, vec![])
            .unwrap()
            .run()
            .unwrap()
            .logs
            .len();
        let epoch_size = (total / 3).max(1);
        assert!(prove_and_verify_continuation(&elf_bytes, &[], epoch_size).unwrap());
    }
}
