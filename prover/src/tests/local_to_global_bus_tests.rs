//! Cross-epoch GlobalMemory bus tests for the local-to-global table.
//!
//! Proves+verifies that the `GlobalMemory` bus balances over the combined L2G
//! table plus two anchors: a genesis sender (program-start initial memory) and a
//! program-end receiver (final value of each cell). The bus balances iff every
//! epoch's `fini` matches the next epoch's `init` (the cross-epoch telescoping).

use std::collections::HashMap;

use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use math::field::element::FieldElement;

use stark::config::Commitment;
use stark::constraints::transition::TransitionConstraintEvaluator;
use stark::lookup::{
    AirWithBuses, AuxiliaryTraceBuildData, BusInteraction, BusValue, Multiplicity,
    NullBoundaryConstraintBuilder, Packing,
};
use stark::proof::options::ProofOptions;
use stark::proof::stark::MultiProof;
use stark::trace::TraceTable;
use stark::traits::AIR;
use stark::verifier::{IsStarkVerifier, Verifier};

use crate::tables::local_to_global::{
    self, CellBoundary, GENESIS_EPOCH, epoch_boundaries, generate_local_to_global_trace,
};
use crate::tables::types::{BusId, FE, GoldilocksExtension, GoldilocksField};
use crate::test_utils::multi_prove_ram;

type F = GoldilocksField;
type E = GoldilocksExtension;

/// Columns of an anchor trace: one GlobalMemory token `(address, value, epoch,
/// timestamp)` per row, packed in the same order as the L2G init/fini tokens.
mod anchor_cols {
    pub const ADDR_LO: usize = 0;
    pub const ADDR_HI: usize = 1;
    pub const VAL_LO: usize = 2;
    pub const VAL_HI: usize = 3;
    pub const EPOCH: usize = 4;
    pub const TS_LO: usize = 5;
    pub const TS_HI: usize = 6;
    pub const NUM_COLUMNS: usize = 7;
}

type Token = (u64, u64, u64, u64);

fn l2g_air(proof_options: &ProofOptions) -> AirWithBuses<F, E, NullBoundaryConstraintBuilder, ()> {
    let transition_constraints: Vec<Box<dyn TransitionConstraintEvaluator<F, E>>> = vec![];
    AirWithBuses::new(
        local_to_global::cols::NUM_COLUMNS,
        AuxiliaryTraceBuildData {
            interactions: local_to_global::bus_interactions(),
        },
        proof_options,
        1,
        transition_constraints,
    )
}

fn anchor_air(
    proof_options: &ProofOptions,
    is_sender: bool,
) -> AirWithBuses<F, E, NullBoundaryConstraintBuilder, ()> {
    let transition_constraints: Vec<Box<dyn TransitionConstraintEvaluator<F, E>>> = vec![];
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
            start_column: anchor_cols::VAL_LO,
            packing: Packing::Direct,
        },
        BusValue::Packed {
            start_column: anchor_cols::VAL_HI,
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
        proof_options,
        1,
        transition_constraints,
    )
}

fn anchor_trace(tokens: &[Token]) -> TraceTable<F, E> {
    let num_rows = tokens.len().next_power_of_two().max(4);
    let mut data = vec![FE::zero(); num_rows * anchor_cols::NUM_COLUMNS];
    for (i, &(addr, value, epoch, ts)) in tokens.iter().enumerate() {
        let base = i * anchor_cols::NUM_COLUMNS;
        data[base + anchor_cols::ADDR_LO] = FE::from(addr & 0xFFFF_FFFF);
        data[base + anchor_cols::ADDR_HI] = FE::from(addr >> 32);
        data[base + anchor_cols::VAL_LO] = FE::from(value & 0xFFFF_FFFF);
        data[base + anchor_cols::VAL_HI] = FE::from(value >> 32);
        data[base + anchor_cols::EPOCH] = FE::from(epoch);
        data[base + anchor_cols::TS_LO] = FE::from(ts & 0xFFFF_FFFF);
        data[base + anchor_cols::TS_HI] = FE::from(ts >> 32);
    }
    TraceTable::new_main(data, anchor_cols::NUM_COLUMNS, 1)
}

/// Inert L2G AIR: commits the trace columns with no bus and no constraints —
/// the deterministic commitment an epoch proof publishes for its L2G table. The
/// main-trace Merkle root is over the main columns only, so it matches the L2G
/// sub-table root committed in the bus proof.
fn inert_l2g_air(
    proof_options: &ProofOptions,
) -> AirWithBuses<F, E, NullBoundaryConstraintBuilder, ()> {
    let transition_constraints: Vec<Box<dyn TransitionConstraintEvaluator<F, E>>> = vec![];
    AirWithBuses::new(
        local_to_global::cols::NUM_COLUMNS,
        AuxiliaryTraceBuildData {
            interactions: vec![],
        },
        proof_options,
        1,
        transition_constraints,
    )
}

/// Commit one epoch's L2G trace in a minimal proof and return its Merkle root —
/// the `R_i` an epoch proof publishes for that epoch.
fn l2g_root(boundary: &[CellBoundary]) -> Commitment {
    let proof_options = ProofOptions::default_test_options();
    let air = inert_l2g_air(&proof_options);
    let mut trace = generate_local_to_global_trace(boundary);
    let pairs: Vec<(
        &dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>,
        _,
        _,
    )> = vec![(&air, &mut trace, &())];
    let proof = multi_prove_ram(pairs, &mut DefaultTranscript::<E>::new(&[])).unwrap();
    proof.proofs[0].lde_trace_main_merkle_root
}

/// Prove the cross-epoch GlobalMemory bus over one L2G sub-table per epoch plus
/// the genesis/program-end anchors. The first N sub-tables (epoch order) are the
/// per-epoch L2G tables.
pub(crate) fn prove_global(boundaries: &[Vec<CellBoundary>]) -> MultiProof<F, E, ()> {
    let all: Vec<CellBoundary> = boundaries.iter().flatten().copied().collect();

    // Genesis anchor: a SEND token for each cell first touched from program memory.
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

    // Program-end anchor: a RECEIVE token for each cell's final fini (epochs are
    // in order, so the last write wins).
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
        .map(|epoch| generate_local_to_global_trace(epoch))
        .collect();
    let mut genesis_trace = anchor_trace(&genesis);
    let mut program_end_trace = anchor_trace(&program_end);

    let proof_options = ProofOptions::default_test_options();
    let l2g = l2g_air(&proof_options);
    let genesis_anchor = anchor_air(&proof_options, true);
    let program_end_anchor = anchor_air(&proof_options, false);

    // Per-epoch L2G sub-tables (all sharing the one L2G air), then the anchors.
    let mut air_trace_pairs: Vec<(
        &dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>,
        _,
        _,
    )> = l2g_traces
        .iter_mut()
        .map(|trace| {
            (
                &l2g as &dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>,
                trace,
                &(),
            )
        })
        .collect();
    air_trace_pairs.push((&genesis_anchor, &mut genesis_trace, &()));
    air_trace_pairs.push((&program_end_anchor, &mut program_end_trace, &()));

    multi_prove_ram(air_trace_pairs, &mut DefaultTranscript::<E>::new(&[])).unwrap()
}

pub(crate) fn prove_and_verify(boundaries: &[Vec<CellBoundary>]) -> bool {
    let proof = prove_global(boundaries);

    let proof_options = ProofOptions::default_test_options();
    let l2g = l2g_air(&proof_options);
    let genesis_anchor = anchor_air(&proof_options, true);
    let program_end_anchor = anchor_air(&proof_options, false);

    // air_refs must match the air_trace_pairs order: one &l2g per epoch, then anchors.
    let mut airs: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>> =
        vec![&l2g; boundaries.len()];
    airs.push(&genesis_anchor);
    airs.push(&program_end_anchor);

    Verifier::multi_verify(
        &airs,
        &proof,
        &mut DefaultTranscript::<E>::new(&[]),
        &FieldElement::zero(),
    )
}

#[test]
fn test_global_memory_bus_balances() {
    // Cell 10 touched in epochs 0,1,2; cell 20 in epoch 0 then again epoch 2
    // (skipping 1); cell 30 once.
    let initial_memory = HashMap::from([(10u64, 5u64)]);
    let epochs = vec![
        vec![(10, 7, 3), (20, 9, 4)],
        vec![(10, 8, 10)],
        vec![(20, 9, 20), (30, 1, 21)],
    ];
    let boundaries = epoch_boundaries(&initial_memory, &epochs);
    assert!(prove_and_verify(&boundaries));
}

#[test]
fn test_global_memory_bus_rejects_tampered_boundary() {
    let initial_memory = HashMap::from([(10u64, 5u64)]);
    let epochs = vec![vec![(10, 7, 3)], vec![(10, 8, 10)]];
    let mut boundaries = epoch_boundaries(&initial_memory, &epochs);
    assert!(prove_and_verify(&boundaries));

    // Break the chain: epoch 0 now claims a different fini than epoch 1's init.
    boundaries[0][0].fini.value = 999;
    assert!(!prove_and_verify(&boundaries));
}

#[test]
fn test_l2g_binding_holds() {
    // Per-epoch L2G roots committed by the epoch proofs match the per-epoch L2G
    // sub-table roots in the final cross-epoch proof.
    let initial_memory = HashMap::from([(10u64, 5u64)]);
    let epochs = vec![
        vec![(10, 7, 3), (20, 9, 4)],
        vec![(10, 8, 10)],
        vec![(20, 9, 20), (30, 1, 21)],
    ];
    let boundaries = epoch_boundaries(&initial_memory, &epochs);

    let final_proof = prove_global(&boundaries);
    let roots: Vec<Commitment> = boundaries.iter().map(|b| l2g_root(b)).collect();

    assert!(crate::verify_l2g_commitment_binding(&roots, &final_proof));
}

#[test]
fn test_l2g_binding_rejects_mismatch() {
    // The final proof uses a DIFFERENT epoch-0 L2G table than the epoch proofs
    // committed, so the binding must reject it.
    let initial_memory = HashMap::from([(10u64, 5u64)]);
    let epochs = vec![
        vec![(10, 7, 3), (20, 9, 4)],
        vec![(10, 8, 10)],
        vec![(20, 9, 20), (30, 1, 21)],
    ];
    let boundaries = epoch_boundaries(&initial_memory, &epochs);

    // Honest per-epoch roots.
    let roots: Vec<Commitment> = boundaries.iter().map(|b| l2g_root(b)).collect();

    // Final proof built over a tampered epoch-0 L2G table.
    let mut tampered = boundaries.clone();
    tampered[0][0].fini.value = 999;
    let final_proof = prove_global(&tampered);

    assert!(!crate::verify_l2g_commitment_binding(&roots, &final_proof));
}
