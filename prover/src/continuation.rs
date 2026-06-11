//! First production implementation of continuations (Approach 2).
//!
//! Splits an execution into fixed-size epochs, proves each epoch independently
//! (its memory is initialized/finalized by the per-epoch local-to-global table),
//! and proves one cross-epoch "global memory" LogUp that links every epoch's
//! `fini` to the next epoch's `init` (so `fini(epoch i) == init(epoch i+1)`).
//!
//! This is a FIRST implementation and is NOT fully sound: the global proof's
//! genesis/program-end anchors are prover-supplied (not yet bound to the ELF),
//! and the local-to-global columns are not range-checked. Those are deferred.

use std::collections::HashMap;

use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use executor::elf::Elf;
use executor::vm::execution::Executor;
use executor::vm::logs::Log;
use math::field::element::FieldElement;
use stark::config::Commitment;
use stark::lookup::{
    AirWithBuses, AuxiliaryTraceBuildData, BusInteraction, BusValue, Multiplicity,
    NullBoundaryConstraintBuilder, Packing,
};
use stark::proof::options::ProofOptions;
use stark::proof::stark::MultiProof;
use stark::prover::{IsStarkProver, Prover};
use stark::trace::TraceTable;
use stark::traits::AIR;
use stark::verifier::{IsStarkVerifier, Verifier};

use crate::tables::MaxRowsConfig;
use crate::tables::local_to_global::{self, CellBoundary, GENESIS_EPOCH};
use crate::tables::register;
use crate::tables::trace_builder::{Traces, build_initial_image, epoch_touched_cells};
use crate::tables::types::{BusId, FE, GoldilocksExtension, GoldilocksField};
use crate::{Error, VmAirs, compute_expected_commit_bus_balance, verify_l2g_commitment_binding};

type F = GoldilocksField;
type E = GoldilocksExtension;
type AirRef<'a> = &'a dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>;

/// One GlobalMemory token `(address, value, epoch, timestamp)`.
type Token = (u64, u64, u64, u64);

/// Anchor trace columns: one GlobalMemory token per row, in the same order as the
/// local-to-global table's GlobalMemory init/fini tokens.
mod anchor_cols {
    pub const ADDR_LO: usize = 0;
    pub const ADDR_HI: usize = 1;
    pub const VAL: usize = 2;
    pub const EPOCH: usize = 3;
    pub const TS_LO: usize = 4;
    pub const TS_HI: usize = 5;
    pub const NUM_COLUMNS: usize = 6;
}

fn empty_constraints()
-> Vec<Box<dyn stark::constraints::transition::TransitionConstraintEvaluator<F, E>>> {
    vec![]
}

/// Local-to-global AIR on the cross-epoch GlobalMemory bus (used in the global proof).
fn l2g_global_air(opts: &ProofOptions) -> AirWithBuses<F, E, NullBoundaryConstraintBuilder, ()> {
    AirWithBuses::new(
        local_to_global::cols::NUM_COLUMNS,
        AuxiliaryTraceBuildData {
            interactions: local_to_global::bus_interactions(),
        },
        opts,
        1,
        empty_constraints(),
    )
}

/// Local-to-global AIR on the epoch-local Memory bus (used inside an epoch proof).
fn l2g_memory_air(opts: &ProofOptions) -> AirWithBuses<F, E, NullBoundaryConstraintBuilder, ()> {
    AirWithBuses::new(
        local_to_global::cols::NUM_COLUMNS,
        AuxiliaryTraceBuildData {
            interactions: local_to_global::memory_bus_interactions(),
        },
        opts,
        1,
        empty_constraints(),
    )
}

/// Anchor AIR: sends (genesis) or receives (program-end) one GlobalMemory token per row.
fn anchor_air(
    opts: &ProofOptions,
    is_sender: bool,
) -> AirWithBuses<F, E, NullBoundaryConstraintBuilder, ()> {
    let values = vec![
        BusValue::Packed {
            start_column: anchor_cols::ADDR_LO,
            packing: Packing::Direct,
        },
        BusValue::Packed {
            start_column: anchor_cols::ADDR_HI,
            packing: Packing::Direct,
        },
        BusValue::Packed {
            start_column: anchor_cols::VAL,
            packing: Packing::Direct,
        },
        BusValue::Packed {
            start_column: anchor_cols::EPOCH,
            packing: Packing::Direct,
        },
        BusValue::Packed {
            start_column: anchor_cols::TS_LO,
            packing: Packing::Direct,
        },
        BusValue::Packed {
            start_column: anchor_cols::TS_HI,
            packing: Packing::Direct,
        },
    ];
    let interaction = if is_sender {
        BusInteraction::sender(BusId::GlobalMemory, Multiplicity::One, values)
    } else {
        BusInteraction::receiver(BusId::GlobalMemory, Multiplicity::One, values)
    };
    AirWithBuses::new(
        anchor_cols::NUM_COLUMNS,
        AuxiliaryTraceBuildData {
            interactions: vec![interaction],
        },
        opts,
        1,
        empty_constraints(),
    )
}

fn anchor_trace(tokens: &[Token]) -> TraceTable<F, E> {
    let num_rows = tokens.len().next_power_of_two().max(4);
    let mut data = vec![FE::zero(); num_rows * anchor_cols::NUM_COLUMNS];
    for (i, &(addr, value, epoch, ts)) in tokens.iter().enumerate() {
        let base = i * anchor_cols::NUM_COLUMNS;
        data[base + anchor_cols::ADDR_LO] = FE::from(addr & 0xFFFF_FFFF);
        data[base + anchor_cols::ADDR_HI] = FE::from(addr >> 32);
        data[base + anchor_cols::VAL] = FE::from(value & 0xFF);
        data[base + anchor_cols::EPOCH] = FE::from(epoch);
        data[base + anchor_cols::TS_LO] = FE::from(ts & 0xFFFF_FFFF);
        data[base + anchor_cols::TS_HI] = FE::from(ts >> 32);
    }
    TraceTable::new_main(data, anchor_cols::NUM_COLUMNS, 1)
}

/// Per-epoch starting state: the memory image and register image the epoch begins from.
struct EpochStart {
    image: HashMap<u64, u8>,
    register_init: HashMap<u64, u32>,
    is_first: bool,
}

/// Prove and verify one epoch, committing its local-to-global table (built from
/// `boundary`) on the epoch-local Memory bus. Returns the L2G commitment root if
/// the epoch verifies, or `None` if it does not.
fn prove_verify_epoch(
    elf: &Elf,
    start: &EpochStart,
    logs: &[Log],
    is_final: bool,
    boundary: &[CellBoundary],
    private_inputs: &[u8],
    opts: &ProofOptions,
) -> Result<Option<Commitment>, Error> {
    let mut traces = Traces::from_image_and_logs(
        elf,
        &start.image,
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

    let table_counts = traces.table_counts();
    let register_init_arg = if start.is_first {
        None
    } else {
        Some(&start.register_init)
    };
    let airs = VmAirs::new(
        elf,
        opts,
        true,
        &traces.page_configs,
        &table_counts,
        None,
        is_final,
        register_init_arg,
        None,
    );

    let l2g_air = l2g_memory_air(opts);
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
    Ok(Some(
        proof.proofs.last().unwrap().lde_trace_main_merkle_root,
    ))
}

/// Build the cross-epoch global memory proof: every epoch's L2G sub-table on the
/// GlobalMemory bus, plus a genesis sender (each cell's first init) and a
/// program-end receiver (each cell's final fini). The bus balances iff every
/// `fini` matches the next epoch's `init`.
fn prove_global(
    boundaries: &[Vec<CellBoundary>],
    opts: &ProofOptions,
) -> Result<MultiProof<F, E, ()>, Error> {
    let all: Vec<CellBoundary> = boundaries.iter().flatten().copied().collect();

    let genesis: Vec<Token> = all
        .iter()
        .filter(|b| b.init.originating_epoch == GENESIS_EPOCH)
        .map(|b| {
            (
                b.address,
                b.init.value,
                b.init.originating_epoch,
                b.init.timestamp,
            )
        })
        .collect();

    let mut final_fini: HashMap<u64, Token> = HashMap::new();
    for epoch in boundaries {
        for b in epoch {
            final_fini.insert(
                b.address,
                (b.address, b.fini.value, b.fini.epoch, b.fini.timestamp),
            );
        }
    }
    let program_end: Vec<Token> = final_fini.into_values().collect();

    let mut l2g_traces: Vec<TraceTable<F, E>> = boundaries
        .iter()
        .map(|epoch| local_to_global::generate_local_to_global_trace(epoch))
        .collect();
    let mut genesis_trace = anchor_trace(&genesis);
    let mut program_end_trace = anchor_trace(&program_end);

    let l2g = l2g_global_air(opts);
    let genesis_anchor = anchor_air(opts, true);
    let program_end_anchor = anchor_air(opts, false);

    let mut pairs: Vec<(AirRef, &mut TraceTable<F, E>, &())> = l2g_traces
        .iter_mut()
        .map(|t| (&l2g as AirRef, t, &()))
        .collect();
    pairs.push((&genesis_anchor, &mut genesis_trace, &()));
    pairs.push((&program_end_anchor, &mut program_end_trace, &()));

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
    opts: &ProofOptions,
) -> bool {
    let l2g = l2g_global_air(opts);
    let genesis_anchor = anchor_air(opts, true);
    let program_end_anchor = anchor_air(opts, false);

    let mut refs: Vec<AirRef> = vec![&l2g; boundaries.len()];
    refs.push(&genesis_anchor);
    refs.push(&program_end_anchor);

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

    let program_image = build_initial_image(&elf, private_inputs);
    let initial_memory: HashMap<u64, u64> =
        program_image.iter().map(|(&a, &v)| (a, v as u64)).collect();

    // Running cross-epoch provenance (the L2G init source). Only the sparse
    // boundaries and the per-epoch roots are kept — everything else is dropped
    // after each epoch is proven (the streaming/eviction the spec describes).
    let mut provenance = local_to_global::genesis_provenance(&initial_memory);
    let mut all_boundaries: Vec<Vec<CellBoundary>> = Vec::new();
    let mut epoch_roots: Vec<Commitment> = Vec::new();
    let opts = ProofOptions::default_test_options();

    let mut index: u64 = 0;
    loop {
        // Capture the epoch's starting state BEFORE running it.
        let start_pc = executor.pc();
        if start_pc == 0 {
            break;
        }
        let start_image: HashMap<u64, u8> = executor.memory().iter_bytes().collect();
        let register_init = if index == 0 {
            register::register_init_from_entry_point(elf.entry_point)
        } else {
            register::register_init_from_snapshot(executor.registers(), start_pc)
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

        let touched = epoch_touched_cells(&elf, &start_image, &logs)?;
        let boundary = local_to_global::epoch_boundary(&mut provenance, index, &touched);

        let start = EpochStart {
            image: start_image,
            register_init,
            is_first: index == 0,
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
            Some(root) => epoch_roots.push(root),
            None => return Ok(false),
        }
        all_boundaries.push(boundary);
        // `start`, `logs`, and this epoch's traces are dropped here.

        if is_final {
            break;
        }
        index += 1;
    }

    // One global LogUp over all the (kept) local-to-global tables.
    let global_proof = prove_global(&all_boundaries, &opts)?;
    if !verify_global(&all_boundaries, &global_proof, &opts) {
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
