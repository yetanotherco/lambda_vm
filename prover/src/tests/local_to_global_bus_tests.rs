//! Cross-epoch GlobalMemory bus tests for the local-to-global table.
//!
//! Proves+verifies that the `GlobalMemory` bus balances over the combined L2G
//! table plus two anchors: a genesis sender (program-start initial memory) and a
//! program-end receiver (final value of each cell). The bus balances iff every
//! epoch's `fini` matches the next epoch's `init` (the cross-epoch telescoping).

use stark::constraints::builder::EmptyConstraints;
use std::collections::HashMap;

use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use math::field::element::FieldElement;

use stark::config::Commitment;
use stark::lookup::{
    AirWithBuses, AuxiliaryTraceBuildData, BusInteraction, BusValue, Multiplicity,
    NullBoundaryConstraintBuilder, Packing,
};
use stark::proof::options::ProofOptions;
use stark::proof::stark::MultiProof;
use stark::trace::TraceTable;
use stark::traits::AIR;
use stark::verifier::{IsStarkVerifier, Verifier};

use crate::tables::bitwise::{BitwiseOperation, BitwiseOperationType};
use crate::tables::local_to_global::{
    self, CellBoundary, GENESIS_EPOCH, epoch_boundaries, generate_local_to_global_trace,
};
use crate::tables::types::{BusId, FE, GoldilocksExtension, GoldilocksField};
use crate::test_utils::multi_prove_ram;

type F = GoldilocksField;
type E = GoldilocksExtension;

/// Columns of an anchor trace: one GlobalMemory token `(address, value, epoch)`
/// per row, packed in the same order as the L2G init/fini tokens (no timestamp —
/// the cross-epoch chain is ordered by epoch).
mod anchor_cols {
    pub const ADDR_LO: usize = 0;
    pub const ADDR_HI: usize = 1;
    pub const VAL: usize = 2;
    pub const EPOCH: usize = 3;
    pub const NUM_COLUMNS: usize = 4;
}

type Token = (u64, u64, u64);

fn l2g_air(
    proof_options: &ProofOptions,
    epoch_label: u64,
) -> AirWithBuses<F, E, NullBoundaryConstraintBuilder, (), EmptyConstraints> {
    AirWithBuses::new(
        local_to_global::cols::NUM_COLUMNS,
        AuxiliaryTraceBuildData {
            interactions: local_to_global::bus_interactions(epoch_label),
        },
        proof_options,
        1,
        EmptyConstraints,
    )
}

fn anchor_air(
    proof_options: &ProofOptions,
    is_sender: bool,
) -> AirWithBuses<F, E, NullBoundaryConstraintBuilder, (), EmptyConstraints> {
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
        EmptyConstraints,
    )
}

fn anchor_trace(tokens: &[Token]) -> TraceTable<F, E> {
    let num_rows = tokens.len().next_power_of_two().max(4);
    let mut data = vec![FE::zero(); num_rows * anchor_cols::NUM_COLUMNS];
    for (i, &(addr, value, epoch)) in tokens.iter().enumerate() {
        let base = i * anchor_cols::NUM_COLUMNS;
        data[base + anchor_cols::ADDR_LO] = FE::from(addr & 0xFFFF_FFFF);
        data[base + anchor_cols::ADDR_HI] = FE::from(addr >> 32);
        data[base + anchor_cols::VAL] = FE::from(value & 0xFF);
        data[base + anchor_cols::EPOCH] = FE::from(epoch);
    }
    TraceTable::new_main(data, anchor_cols::NUM_COLUMNS, 1)
}

/// L2G air on the epoch-LOCAL `Memory` bus (uses `memory_bus_interactions`).
fn l2g_memory_air(
    proof_options: &ProofOptions,
) -> AirWithBuses<F, E, NullBoundaryConstraintBuilder, (), EmptyConstraints> {
    AirWithBuses::new(
        local_to_global::cols::NUM_COLUMNS,
        AuxiliaryTraceBuildData {
            interactions: local_to_global::memory_bus_interactions(),
        },
        proof_options,
        1,
        EmptyConstraints,
    )
}

/// Columns of a MEMW-substitute trace: per touched byte, the `Memory` tokens the
/// real access chain would emit — opposite polarity to L2G's bookend.
mod memw_sub_cols {
    pub const ADDR_LO: usize = 0;
    pub const ADDR_HI: usize = 1;
    pub const INIT_VAL: usize = 2;
    pub const FINI_TS_LO: usize = 3;
    pub const FINI_TS_HI: usize = 4;
    pub const FINI_VAL: usize = 5;
    pub const NUM_COLUMNS: usize = 6;
}

/// Minimal BITWISE-receiver substitute for the L2G range-check buses. It receives
/// the same AreBytes, IsHalfword, and IsB20 tokens that the real BITWISE table
/// would receive, but only for rows supplied by the test.
mod range_recv_cols {
    pub const X: usize = 0;
    pub const Y: usize = 1;
    pub const Z: usize = 2;
    pub const MU_ARE_BYTES: usize = 3;
    pub const MU_IS_HALF: usize = 4;
    pub const MU_IS_B20: usize = 5;
    pub const NUM_COLUMNS: usize = 6;
}

/// MEMW-substitute air: counterpart to `memory_bus_interactions`. Sends each
/// cell's init token at ts=0 (cancelling L2G's init-receive) and receives each
/// cell's fini token at the last timestamp (cancelling L2G's fini-send).
fn memw_sub_air(
    proof_options: &ProofOptions,
) -> AirWithBuses<F, E, NullBoundaryConstraintBuilder, (), EmptyConstraints> {
    let init_send = BusInteraction::sender(
        BusId::Memory,
        Multiplicity::One,
        vec![
            BusValue::constant(0),
            BusValue::Packed {
                start_column: memw_sub_cols::ADDR_LO,
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: memw_sub_cols::ADDR_HI,
                packing: Packing::Direct,
            },
            BusValue::constant(0),
            BusValue::constant(0),
            BusValue::Packed {
                start_column: memw_sub_cols::INIT_VAL,
                packing: Packing::Direct,
            },
        ],
    );
    let fini_recv = BusInteraction::receiver(
        BusId::Memory,
        Multiplicity::One,
        vec![
            BusValue::constant(0),
            BusValue::Packed {
                start_column: memw_sub_cols::ADDR_LO,
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: memw_sub_cols::ADDR_HI,
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: memw_sub_cols::FINI_TS_LO,
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: memw_sub_cols::FINI_TS_HI,
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: memw_sub_cols::FINI_VAL,
                packing: Packing::Direct,
            },
        ],
    );
    AirWithBuses::new(
        memw_sub_cols::NUM_COLUMNS,
        AuxiliaryTraceBuildData {
            interactions: vec![init_send, fini_recv],
        },
        proof_options,
        1,
        EmptyConstraints,
    )
}

fn l2g_range_air(
    proof_options: &ProofOptions,
    epoch_label: u64,
) -> AirWithBuses<F, E, NullBoundaryConstraintBuilder, (), EmptyConstraints> {
    AirWithBuses::new(
        local_to_global::cols::NUM_COLUMNS,
        AuxiliaryTraceBuildData {
            interactions: local_to_global::range_check_interactions(epoch_label),
        },
        proof_options,
        1,
        EmptyConstraints,
    )
}

fn range_receiver_air(
    proof_options: &ProofOptions,
) -> AirWithBuses<F, E, NullBoundaryConstraintBuilder, (), EmptyConstraints> {
    let interactions = vec![
        BusInteraction::receiver(
            BusId::AreBytes,
            Multiplicity::Column(range_recv_cols::MU_ARE_BYTES),
            vec![
                BusValue::Packed {
                    start_column: range_recv_cols::X,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: range_recv_cols::Y,
                    packing: Packing::Direct,
                },
            ],
        ),
        BusInteraction::receiver(
            BusId::IsHalfword,
            Multiplicity::Column(range_recv_cols::MU_IS_HALF),
            vec![BusValue::linear(vec![
                stark::lookup::LinearTerm::Column {
                    coefficient: 1,
                    column: range_recv_cols::X,
                },
                stark::lookup::LinearTerm::Column {
                    coefficient: 256,
                    column: range_recv_cols::Y,
                },
            ])],
        ),
        BusInteraction::receiver(
            BusId::IsB20,
            Multiplicity::Column(range_recv_cols::MU_IS_B20),
            vec![BusValue::linear(vec![
                stark::lookup::LinearTerm::Column {
                    coefficient: 1,
                    column: range_recv_cols::X,
                },
                stark::lookup::LinearTerm::Column {
                    coefficient: 256,
                    column: range_recv_cols::Y,
                },
                stark::lookup::LinearTerm::Column {
                    coefficient: 65536,
                    column: range_recv_cols::Z,
                },
            ])],
        ),
    ];
    AirWithBuses::new(
        range_recv_cols::NUM_COLUMNS,
        AuxiliaryTraceBuildData { interactions },
        proof_options,
        1,
        EmptyConstraints,
    )
}

fn range_receiver_trace(ops: &[BitwiseOperation]) -> TraceTable<F, E> {
    let num_rows = ops.len().next_power_of_two().max(4);
    let mut data = vec![FE::zero(); num_rows * range_recv_cols::NUM_COLUMNS];
    for (i, op) in ops.iter().enumerate() {
        let base = i * range_recv_cols::NUM_COLUMNS;
        data[base + range_recv_cols::X] = FE::from(op.x as u64);
        data[base + range_recv_cols::Y] = FE::from(op.y as u64);
        data[base + range_recv_cols::Z] = FE::from(op.z as u64);
        let mu_col = match op.lookup_type {
            BitwiseOperationType::AreBytes => range_recv_cols::MU_ARE_BYTES,
            BitwiseOperationType::IsHalf => range_recv_cols::MU_IS_HALF,
            BitwiseOperationType::IsB20 => range_recv_cols::MU_IS_B20,
            _ => panic!("unexpected L2G range-check lookup"),
        };
        data[base + mu_col] = FE::one();
    }
    TraceTable::new_main(data, range_recv_cols::NUM_COLUMNS, 1)
}

fn memw_sub_trace(boundary: &[CellBoundary]) -> TraceTable<F, E> {
    let num_rows = boundary.len().next_power_of_two().max(4);
    let mut data = vec![FE::zero(); num_rows * memw_sub_cols::NUM_COLUMNS];
    for (i, b) in boundary.iter().enumerate() {
        let base = i * memw_sub_cols::NUM_COLUMNS;
        data[base + memw_sub_cols::ADDR_LO] = FE::from(b.address & 0xFFFF_FFFF);
        data[base + memw_sub_cols::ADDR_HI] = FE::from(b.address >> 32);
        data[base + memw_sub_cols::INIT_VAL] = FE::from(b.init.value & 0xFF);
        data[base + memw_sub_cols::FINI_TS_LO] = FE::from(b.fini.timestamp & 0xFFFF_FFFF);
        data[base + memw_sub_cols::FINI_TS_HI] = FE::from(b.fini.timestamp >> 32);
        data[base + memw_sub_cols::FINI_VAL] = FE::from(b.fini.value & 0xFF);
    }
    TraceTable::new_main(data, memw_sub_cols::NUM_COLUMNS, 1)
}

/// Prove + verify the epoch-local `Memory` bus over L2G's bookend (built from
/// `l2g_boundary`) plus the MEMW-substitute chain (built from `memw_boundary`).
/// Equal boundaries balance; a mismatch leaves the bus unbalanced.
fn prove_verify_memory(l2g_boundary: &[CellBoundary], memw_boundary: &[CellBoundary]) -> bool {
    let proof_options = ProofOptions::default_test_options();
    let l2g = l2g_memory_air(&proof_options);
    let memw = memw_sub_air(&proof_options);
    let mut l2g_trace = generate_local_to_global_trace(l2g_boundary);
    let mut memw_trace = memw_sub_trace(memw_boundary);
    let pairs: Vec<(
        &dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>,
        _,
        _,
    )> = vec![(&l2g, &mut l2g_trace, &()), (&memw, &mut memw_trace, &())];
    let proof = multi_prove_ram(pairs, &mut DefaultTranscript::<E>::new(&[])).unwrap();
    let airs: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>> = vec![&l2g, &memw];
    Verifier::multi_verify(
        &airs,
        &proof,
        &mut DefaultTranscript::<E>::new(&[]),
        &FieldElement::zero(),
    )
}

fn prove_verify_l2g_range_with_trace(
    l2g_trace: &mut TraceTable<F, E>,
    range_ops: &[BitwiseOperation],
    epoch_label: u64,
) -> bool {
    let proof_options = ProofOptions::default_test_options();
    let l2g = l2g_range_air(&proof_options, epoch_label);
    let receiver = range_receiver_air(&proof_options);
    let mut receiver_trace = range_receiver_trace(range_ops);
    let pairs: Vec<(
        &dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>,
        _,
        _,
    )> = vec![
        (&l2g, l2g_trace, &()),
        (&receiver, &mut receiver_trace, &()),
    ];
    let proof = multi_prove_ram(pairs, &mut DefaultTranscript::<E>::new(&[])).unwrap();
    let airs: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>> =
        vec![&l2g, &receiver];
    Verifier::multi_verify(
        &airs,
        &proof,
        &mut DefaultTranscript::<E>::new(&[]),
        &FieldElement::zero(),
    )
}

/// Inert L2G AIR: commits the trace columns with no bus and no constraints —
/// the deterministic commitment an epoch proof publishes for its L2G table. The
/// main-trace Merkle root is over the main columns only, so it matches the L2G
/// sub-table root committed in the bus proof.
fn inert_l2g_air(
    proof_options: &ProofOptions,
) -> AirWithBuses<F, E, NullBoundaryConstraintBuilder, (), EmptyConstraints> {
    AirWithBuses::new(
        local_to_global::cols::NUM_COLUMNS,
        AuxiliaryTraceBuildData {
            interactions: vec![],
        },
        proof_options,
        1,
        EmptyConstraints,
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
        .map(|b| (b.address, b.init.value, b.init.originating_epoch))
        .collect();

    // Program-end anchor: a RECEIVE token for each cell's final fini (epochs are
    // in order, so the last write wins).
    let mut final_fini: HashMap<u64, Token> = HashMap::new();
    for epoch in boundaries {
        for b in epoch {
            final_fini.insert(b.address, (b.address, b.fini.value, b.fini.epoch));
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
    // One L2G air per epoch, each carrying its 1-based `fini_epoch` constant.
    let l2g_airs: Vec<_> = (0..boundaries.len())
        .map(|i| l2g_air(&proof_options, local_to_global::epoch_label(i as u64)))
        .collect();
    let genesis_anchor = anchor_air(&proof_options, true);
    let program_end_anchor = anchor_air(&proof_options, false);

    // Per-epoch L2G sub-tables (each with its own air), then the anchors.
    let mut air_trace_pairs: Vec<(
        &dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>,
        _,
        _,
    )> = l2g_airs
        .iter()
        .zip(l2g_traces.iter_mut())
        .map(|(air, trace)| {
            (
                air as &dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>,
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
    let l2g_airs: Vec<_> = (0..boundaries.len())
        .map(|i| l2g_air(&proof_options, local_to_global::epoch_label(i as u64)))
        .collect();
    let genesis_anchor = anchor_air(&proof_options, true);
    let program_end_anchor = anchor_air(&proof_options, false);

    // air_refs must match the air_trace_pairs order: one L2G air per epoch, then anchors.
    let mut airs: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>> = l2g_airs
        .iter()
        .map(|a| a as &dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>)
        .collect();
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

// =========================================================================
// Helpers for soundness regression tests
// =========================================================================

/// Like `prove_verify_memory` but accepts a pre-built (possibly mutated)
/// l2g trace instead of deriving it from a boundary slice.
///
/// Used by tests that forge individual columns (values, timestamps, epoch
/// halfwords) after trace generation — the mutation must survive into the proof so
/// the verifier sees the forged commitment.
fn prove_verify_memory_with_trace(
    l2g_trace: &mut TraceTable<F, E>,
    memw_boundary: &[CellBoundary],
) -> bool {
    let proof_options = ProofOptions::default_test_options();
    let l2g = l2g_memory_air(&proof_options);
    let memw = memw_sub_air(&proof_options);
    let mut memw_trace = memw_sub_trace(memw_boundary);
    let pairs: Vec<(
        &dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>,
        _,
        _,
    )> = vec![(&l2g, l2g_trace, &()), (&memw, &mut memw_trace, &())];
    let proof = multi_prove_ram(pairs, &mut DefaultTranscript::<E>::new(&[])).unwrap();
    let airs: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>> = vec![&l2g, &memw];
    Verifier::multi_verify(
        &airs,
        &proof,
        &mut DefaultTranscript::<E>::new(&[]),
        &FieldElement::zero(),
    )
}

/// Like `prove_global` (and `prove_and_verify`) but accepts pre-built l2g
/// traces (one per epoch) so that column mutations applied before this call
/// survive into the proof.
///
/// Returns `true` iff the multi-table verifier accepts the proof.
fn prove_and_verify_global_with_traces(
    boundaries: &[Vec<CellBoundary>],
    l2g_traces: &mut [TraceTable<F, E>],
) -> bool {
    let all: Vec<CellBoundary> = boundaries.iter().flatten().copied().collect();

    let genesis: Vec<Token> = all
        .iter()
        .filter(|b| b.init.originating_epoch == GENESIS_EPOCH)
        .map(|b| (b.address, b.init.value, b.init.originating_epoch))
        .collect();

    let mut final_fini: HashMap<u64, Token> = HashMap::new();
    for epoch in boundaries {
        for b in epoch {
            final_fini.insert(b.address, (b.address, b.fini.value, b.fini.epoch));
        }
    }
    let program_end: Vec<Token> = final_fini.into_values().collect();

    let mut genesis_trace = anchor_trace(&genesis);
    let mut program_end_trace = anchor_trace(&program_end);

    let proof_options = ProofOptions::default_test_options();
    let l2g_airs: Vec<_> = (0..boundaries.len())
        .map(|i| l2g_air(&proof_options, local_to_global::epoch_label(i as u64)))
        .collect();
    let genesis_anchor = anchor_air(&proof_options, true);
    let program_end_anchor = anchor_air(&proof_options, false);

    let mut air_trace_pairs: Vec<(
        &dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>,
        _,
        _,
    )> = l2g_airs
        .iter()
        .zip(l2g_traces.iter_mut())
        .map(|(air, trace)| {
            (
                air as &dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>,
                trace,
                &(),
            )
        })
        .collect();
    air_trace_pairs.push((&genesis_anchor, &mut genesis_trace, &()));
    air_trace_pairs.push((&program_end_anchor, &mut program_end_trace, &()));

    let proof = multi_prove_ram(air_trace_pairs, &mut DefaultTranscript::<E>::new(&[])).unwrap();

    let mut airs: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>> = l2g_airs
        .iter()
        .map(|a| a as &dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>)
        .collect();
    airs.push(&genesis_anchor);
    airs.push(&program_end_anchor);

    Verifier::multi_verify(
        &airs,
        &proof,
        &mut DefaultTranscript::<E>::new(&[]),
        &FieldElement::zero(),
    )
}

// =========================================================================
// Soundness regression tests: brought-forward filler rows
// =========================================================================
//
// The L2G table has no MU selector: padding rows are real brought-forward
// (filler) rows for untouched cells. A filler is a value-preserving no-op only
// because the epoch-local Memory bus forces it to be one — an untouched cell has
// no MEMW partner, so its init-receive and fini-send must be the identical token
// (`fini_ts = 0`, `init_value = fini_value`) or the bus dangles. These tests pin
// that: an honest filler balances, and any non-no-op filler is rejected.

/// Honest filler rows self-cancel on the epoch-local Memory bus: the MEMW
/// substitute only covers the genuinely-touched cell, and the filler's identical
/// init/fini tokens net to zero, so the bus balances.
#[test]
fn test_l2g_filler_self_cancels_on_memory_bus() {
    // 3 touched cells → padded to 4 rows with one brought-forward filler (row 3).
    let mut provenance = local_to_global::genesis_provenance([(10u64, 5u64)]);
    let mut l2g_boundary = local_to_global::epoch_boundary(
        &mut provenance,
        local_to_global::epoch_label(0),
        &[(10, 7, 3), (11, 8, 4), (12, 9, 5)],
    );
    local_to_global::append_bring_forward_fillers(
        &mut l2g_boundary,
        &mut provenance,
        &[0],
        local_to_global::epoch_label(0),
    )
    .unwrap();
    assert_eq!(
        l2g_boundary.len(),
        4,
        "padded to a power of two with one filler"
    );

    // The MEMW substitute covers only the real touched cells (rows 0..3); the filler
    // (row 3) has no MEMW partner and must self-cancel on its own.
    assert!(
        prove_verify_memory(&l2g_boundary, &l2g_boundary[..3]),
        "an honest brought-forward filler must self-cancel on the Memory bus"
    );
}

/// A filler with `init_value != fini_value` breaks self-cancellation: its
/// init-receive and fini-send become different Memory-bus tokens, and an untouched
/// cell has no MEMW partner for either → the bus dangles → reject.
#[test]
fn test_l2g_filler_value_mismatch_rejects() {
    // 3 touched cells → one filler at row 3.
    let mut provenance = local_to_global::genesis_provenance([(10u64, 5u64)]);
    let mut l2g_boundary = local_to_global::epoch_boundary(
        &mut provenance,
        local_to_global::epoch_label(0),
        &[(10, 7, 3), (11, 8, 4), (12, 9, 5)],
    );
    local_to_global::append_bring_forward_fillers(
        &mut l2g_boundary,
        &mut provenance,
        &[0],
        local_to_global::epoch_label(0),
    )
    .unwrap();

    // Forge the filler row (row 3) to change its value between init and fini.
    let mut forged = generate_local_to_global_trace(&l2g_boundary);
    forged
        .main_table
        .set(3, local_to_global::cols::FINI_VALUE, FE::from(0x33u64));

    assert!(
        !prove_verify_memory_with_trace(&mut forged, &l2g_boundary[..3]),
        "a filler with init_value != fini_value must dangle on the Memory bus"
    );
}

/// A filler with `fini_timestamp != 0` breaks self-cancellation: its fini-send
/// token carries a nonzero timestamp while its init-receive is at ts=0, so the two
/// differ and neither finds a MEMW partner → reject.
#[test]
fn test_l2g_filler_nonzero_fini_timestamp_rejects() {
    // 3 touched cells → one filler at row 3.
    let mut provenance = local_to_global::genesis_provenance([(10u64, 5u64)]);
    let mut l2g_boundary = local_to_global::epoch_boundary(
        &mut provenance,
        local_to_global::epoch_label(0),
        &[(10, 7, 3), (11, 8, 4), (12, 9, 5)],
    );
    local_to_global::append_bring_forward_fillers(
        &mut l2g_boundary,
        &mut provenance,
        &[0],
        local_to_global::epoch_label(0),
    )
    .unwrap();

    // Forge the filler row (row 3) to carry a nonzero fini timestamp.
    let mut forged = generate_local_to_global_trace(&l2g_boundary);
    forged
        .main_table
        .set(3, local_to_global::cols::FINI_TIMESTAMP_LO, FE::from(7u64));

    assert!(
        !prove_verify_memory_with_trace(&mut forged, &l2g_boundary[..3]),
        "a filler with fini_timestamp != 0 must dangle on the Memory bus"
    );
}

/// Honest brought-forward fillers telescope on the cross-epoch GlobalMemory bus:
/// an epoch with a non-power-of-two touched count is padded with fillers and still
/// balances end-to-end (genesis → epoch fini/filler → program-end).
#[test]
fn test_global_memory_bus_balances_with_fillers() {
    let mut provenance = local_to_global::genesis_provenance(std::iter::empty());
    let mut epoch0 = local_to_global::epoch_boundary(
        &mut provenance,
        local_to_global::epoch_label(0),
        &[(10, 7, 3), (20, 9, 4), (30, 1, 5)],
    );
    assert_eq!(epoch0.len(), 3, "3 touched cells (not a power of two)");
    local_to_global::append_bring_forward_fillers(
        &mut epoch0,
        &mut provenance,
        &[0],
        local_to_global::epoch_label(0),
    )
    .unwrap();
    assert_eq!(epoch0.len(), 4, "padded to a power of two with one filler");

    let boundaries = vec![epoch0];
    assert!(
        prove_and_verify(&boundaries),
        "brought-forward fillers must telescope on the GlobalMemory bus"
    );
}

/// Multiple fillers in one epoch (5 touched → padded to 8, i.e. 3 fillers) telescope
/// on the GlobalMemory bus. The single-filler tests never exercise more than one
/// filler row nor a wider power-of-two gap.
#[test]
fn test_global_memory_bus_balances_with_multiple_fillers() {
    let mut provenance = local_to_global::genesis_provenance(std::iter::empty());
    let mut epoch0 = local_to_global::epoch_boundary(
        &mut provenance,
        local_to_global::epoch_label(0),
        &[(10, 1, 3), (20, 2, 4), (30, 3, 5), (40, 4, 6), (50, 5, 7)],
    );
    assert_eq!(epoch0.len(), 5, "5 touched cells (not a power of two)");
    local_to_global::append_bring_forward_fillers(
        &mut epoch0,
        &mut provenance,
        &[0],
        local_to_global::epoch_label(0),
    )
    .unwrap();
    assert_eq!(
        epoch0.len(),
        8,
        "padded to a power of two with three fillers"
    );

    let boundaries = vec![epoch0];
    assert!(
        prove_and_verify(&boundaries),
        "multiple brought-forward fillers must telescope on the GlobalMemory bus"
    );
}

/// A cell brought forward as a filler in one epoch and then genuinely TOUCHED in a
/// later epoch must telescope correctly: the later epoch's init consumes the
/// filler's produced token (init_epoch = the filler epoch), and the value flows
/// through. Locks in the filler→real-touch chaining the security/spec reviews
/// verified only by inspection.
#[test]
fn test_filler_then_real_touch_telescopes() {
    let mut provenance = local_to_global::genesis_provenance(std::iter::empty());

    // Epoch 0 touches 3 cells (10, 11, 12) → padded to 4 with one filler. With
    // candidate page 0 and those cells occupied, the filler is address 0.
    let mut epoch0 = local_to_global::epoch_boundary(
        &mut provenance,
        local_to_global::epoch_label(0),
        &[(10, 7, 3), (11, 8, 4), (12, 9, 5)],
    );
    local_to_global::append_bring_forward_fillers(
        &mut epoch0,
        &mut provenance,
        &[0],
        local_to_global::epoch_label(0),
    )
    .unwrap();
    assert!(
        epoch0.iter().any(|b| b.address == 0),
        "the filler brought forward cell 0"
    );

    // Epoch 1 genuinely touches cell 0 (the brought-forward cell), writing 42. Its
    // init must link to the filler's fini (init_epoch = 1, the filler's epoch).
    let epoch1 = local_to_global::epoch_boundary(
        &mut provenance,
        local_to_global::epoch_label(1),
        &[(0, 42, 10)],
    );
    assert_eq!(
        epoch1[0].init.originating_epoch,
        local_to_global::epoch_label(0)
    );
    assert_eq!(
        epoch1[0].init.value, 0,
        "consumes the filler's preserved value"
    );

    let boundaries = vec![epoch0, epoch1];
    assert!(
        prove_and_verify(&boundaries),
        "a filler cell later touched for real must telescope (genesis → filler → touch → final)"
    );
}

// =========================================================================
// Soundness regression tests: init_epoch ordering (IsB20)
// =========================================================================

/// (2) Forged init_epoch violating the ordering constraint is rejected.
///
/// Property guarded: `init_epoch < fini_epoch` is enforced via an IsB20
/// lookup on `fini_epoch − 1 − init_epoch`. A forged row that claims
/// `init_epoch >= fini_epoch` causes the difference to underflow in the
/// field to a value far outside [0, 2^20); no matching IsB20 row exists in
/// the BITWISE table, so the range-check bus cannot balance and the proof
/// must fail.
///
/// The ordering check lives on `range_check_interactions`, which is wired to
/// the BITWISE table inside the epoch proof. The epoch-local `l2g_memory_air`
/// in this test file does NOT include `range_check_interactions` — it only
/// covers the Memory bus. The full range-check path (with a live BITWISE
/// table) is exercised inside the epoch prover in `continuation.rs`
/// (`l2g_memory_air` there concatenates both, see line 155-159). Wiring the
/// complete BITWISE sub-proof here would require replicating `prove_epoch`'s
/// full table set, which is out of scope for a unit bus test.
///
/// This test asserts the arithmetic property that makes the attack fail.
/// `test_ordering_rejects_future_reference` in
/// `local_to_global.rs::tests` (line 831) already verifies that the field
/// value `fini_epoch − 1 − init_epoch` wraps to a value ≥ 2^20 for both
/// self-references and future-references, so no IsB20 row matches. The
/// proof-level bus path is covered by
/// `test_l2g_init_epoch_ordering_live_is_b20_rejects` below.
///
/// Variants that ARE expressible without the full bitwise table:
///   - Self-reference (init_epoch == fini_epoch) and future-reference
///     (init_epoch > fini_epoch) are both covered by the arithmetic check.
///   - The GlobalMemory bus itself does NOT enforce the ordering; it only
///     checks that tokens match across epochs. The IsB20 sender is wired
///     exclusively on the epoch-local table (which carries the BITWISE provider).
///
/// The paired live-bus test wires an L2G range-check AIR to a minimal BITWISE
/// receiver table and proves that a self-reference rejects through IsB20.
#[test]
fn test_l2g_init_epoch_ordering_field_arithmetic() {
    // Verify the arithmetic property that underlies the IsB20 soundness argument
    // without running a full proof. The ordering sender computes:
    //   fini_epoch − 1 − init_epoch   (in the Goldilocks field)
    // For an honest row this is a small non-negative integer in [0, 2^20).
    // For a forged row it wraps to a huge field value outside [0, 2^20).

    let order_field_value = |fini_label: u64, init_epoch: u64| -> u64 {
        // Replicate the field arithmetic: FE::from(fini_label - 1) - FE::from(init_epoch).
        // The Goldilocks prime is 2^64 - 2^32 + 1.
        let result = FE::from(fini_label - 1) - FE::from(init_epoch);
        *result.value()
    };

    // Honest: epoch 2 consuming genesis (epoch 0) fini → 2 - 1 - 0 = 1.
    assert!(order_field_value(2, GENESIS_EPOCH) < (1 << 20));

    // Honest: epoch 5 consuming epoch 2's fini → 5 - 1 - 2 = 2.
    assert!(order_field_value(5, 2) < (1 << 20));

    // Forged self-reference: init_epoch == fini_epoch → 5 - 1 - 5 = -1 in field.
    let self_ref = order_field_value(5, 5);
    assert!(
        self_ref >= (1 << 20),
        "self-reference must produce a value outside the IsB20 range (got {self_ref})"
    );

    // Forged future-reference: init_epoch > fini_epoch → 5 - 1 - 9 < 0 in field.
    let future_ref = order_field_value(5, 9);
    assert!(
        future_ref >= (1 << 20),
        "future-reference must produce a value outside the IsB20 range (got {future_ref})"
    );
}

#[test]
fn test_l2g_init_epoch_ordering_live_is_b20_rejects() {
    // Epoch 1 consumes epoch 0's fini for cell 10. Honest ordering has
    // init_epoch=1, fini_epoch=2, so 2 - 1 - 1 = 0 is a valid IsB20 lookup.
    let initial_memory = HashMap::new();
    let epochs = vec![vec![(10, 7, 3)], vec![(10, 8, 10)]];
    let boundaries = epoch_boundaries(&initial_memory, &epochs);
    let boundary = &boundaries[1];
    let epoch_label = boundary[0].fini.epoch;
    assert_eq!(epoch_label, 2);

    let mut honest_trace = generate_local_to_global_trace(boundary);
    let honest_ops = local_to_global::collect_bitwise_from_l2g(boundary);
    assert!(
        prove_verify_l2g_range_with_trace(&mut honest_trace, &honest_ops, epoch_label),
        "honest L2G range checks must balance against BITWISE receivers"
    );

    // Forge a self-reference: init_epoch == fini_epoch. The halfword lookups are
    // still satisfiable, so the receiver table below includes them. The missing
    // piece is exactly IsB20[2 - 1 - 2], which underflows in the field and has no
    // valid 20-bit receiver row.
    let mut forged_trace = generate_local_to_global_trace(boundary);
    forged_trace.main_table.set(
        0,
        local_to_global::cols::INIT_EPOCH_0,
        FE::from(epoch_label),
    );
    forged_trace
        .main_table
        .set(0, local_to_global::cols::INIT_EPOCH_1, FE::zero());

    let cell = boundary[0];
    let forged_ops = vec![
        BitwiseOperation::byte_op(
            BitwiseOperationType::AreBytes,
            (cell.init.value & 0xFF) as u8,
            (cell.fini.value & 0xFF) as u8,
        ),
        BitwiseOperation::halfword(
            BitwiseOperationType::IsHalf,
            (epoch_label & 0xFF) as u8,
            ((epoch_label >> 8) & 0xFF) as u8,
        ),
        BitwiseOperation::halfword(BitwiseOperationType::IsHalf, 0, 0),
    ];
    assert!(
        !prove_verify_l2g_range_with_trace(&mut forged_trace, &forged_ops, epoch_label),
        "self-referential init_epoch must fail through the live IsB20 bus"
    );
}

// =========================================================================
// Soundness regression tests: chain truncation / forged fillers
// =========================================================================

/// Design-Y (orphan a touched epoch by silencing its L2G row with `MU=0`) is
/// STRUCTURALLY IMPOSSIBLE now: there is no MU column, so no row can be silenced —
/// every L2G row fires with multiplicity 1. Instead, the closest remaining lever is
/// forging a *filler* row's committed value so it does not telescope. This test
/// guards that such a forgery dangles on the cross-epoch GlobalMemory bus.
///
/// Cell 10 is touched in epochs 0 and 1; epoch 1 is padded with a brought-forward
/// filler for an untouched cell. Forging that filler's fini value makes the token it
/// produces (consumed by the program-end anchor) disagree with the token it consumed
/// (its init), so the GlobalMemory bus cannot balance. Uses
/// `prove_and_verify_global_with_traces` to inject the pre-forged epoch-1 trace.
#[test]
fn test_l2g_forged_filler_value_rejects_global_bus() {
    // Epoch 0 touches cells 10 and 20 (2 cells, already a power of two). Epoch 1
    // touches 3 cells, so it is padded with one brought-forward filler (row 3).
    let mut provenance = local_to_global::genesis_provenance([(10u64, 5u64)]);
    let epoch0 = local_to_global::epoch_boundary(
        &mut provenance,
        local_to_global::epoch_label(0),
        &[(10, 7, 3), (20, 9, 4)],
    );
    let mut epoch1 = local_to_global::epoch_boundary(
        &mut provenance,
        local_to_global::epoch_label(1),
        &[(10, 8, 10), (20, 5, 11), (30, 2, 12)],
    );
    local_to_global::append_bring_forward_fillers(
        &mut epoch1,
        &mut provenance,
        &[0],
        local_to_global::epoch_label(1),
    )
    .unwrap();
    assert_eq!(
        epoch1.len(),
        4,
        "epoch 1 padded to a power of two with one filler"
    );
    let boundaries = vec![epoch0, epoch1];

    // Honest baseline balances.
    assert!(
        prove_and_verify(&boundaries),
        "baseline must verify before forgery"
    );

    // Forge the filler row (epoch 1, row 3) to change its fini value. Its
    // GlobalMemory fini-send now produces a token no anchor / later init consumes,
    // while the value it consumed still expects the honest value → bus imbalance.
    let epoch0_trace = generate_local_to_global_trace(&boundaries[0]);
    let mut epoch1_trace = generate_local_to_global_trace(&boundaries[1]);
    epoch1_trace
        .main_table
        .set(3, local_to_global::cols::FINI_VALUE, FE::from(0x33u64));
    let mut l2g_traces = vec![epoch0_trace, epoch1_trace];

    assert!(
        !prove_and_verify_global_with_traces(&boundaries, &mut l2g_traces),
        "a forged filler value must break the GlobalMemory telescoping chain"
    );
}

/// A touched cell cannot be dropped from the L2G table without dangling its
/// MEMW-chain tokens: the epoch-local Memory bus rejects. This is the property that
/// replaces the old `MU`-gating — participation is no longer selectable, so omitting
/// a row (the only way to "drop" one) leaves the MEMW substitute's init-send and
/// fini-receive for that cell unmatched.
#[test]
fn test_l2g_dropped_touched_row_rejects_memory_bus() {
    let initial_memory = HashMap::from([(10u64, 5u64)]);
    let epochs = vec![vec![(10, 7, 3)]];
    let boundaries = epoch_boundaries(&initial_memory, &epochs);

    // Honest baseline: L2G bookend matches the MEMW substitute.
    assert!(prove_verify_memory(&boundaries[0], &boundaries[0]));

    // Drop the L2G row for cell 10 (empty L2G boundary) while the MEMW substitute
    // still accesses cell 10 → its tokens have no L2G partner → reject.
    assert!(
        !prove_verify_memory(&[], &boundaries[0]),
        "dropping a touched cell's L2G row must dangle its MEMW tokens"
    );
}

// =========================================================================
// Soundness regression tests: private-input continuation
// =========================================================================

/// (4) Private-input continuation: `test_private_input_xpage` spans multiple
/// epochs and verifies with non-empty private inputs.
///
/// Property guarded: the continuation prover correctly handles private-input
/// pages (which are touched in the first epoch and potentially persist across
/// epoch boundaries) and the resulting multi-epoch L2G chain verifies end-to-end.
///
/// The fixture reads 16 bytes of private input from 0xFF000000, then commits
/// bytes 4..12 (8 bytes after the 4-byte length prefix). With `epoch_size_log2=2`
/// (4 cycles) the 11-cycle program spans three epochs: epoch 0 reads the private-input
/// page (touching 0xFF000000..), epoch 1 performs the commit syscall, epoch 2
/// halts. The private-input page's L2G entry (epoch 0 fini → epoch 1+ init)
/// is the cross-epoch link under test.
///
/// Modelled on `continuation::tests::test_prove_and_verify_continuation`
/// (continuation.rs:896) and `prove_elfs_tests::test_prove_private_input_xpage`
/// (prove_elfs_tests.rs:2649).
#[test]
fn test_continuation_private_input_spans_epochs() {
    let elf_bytes = crate::test_utils::asm_elf_bytes("test_private_input_xpage");

    // 16-byte private input: 4-byte length prefix (=16) + 8 bytes of payload
    // that will be committed + 4 padding bytes (the fixture commits bytes 4..12).
    let mut input: Vec<u8> = Vec::with_capacity(16);
    // Length prefix: 16 as little-endian u32.
    input.extend_from_slice(&16u32.to_le_bytes());
    // 8-byte payload that will be committed.
    input.extend_from_slice(&[0x11u8, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88]);
    // 4 trailing padding bytes (not committed).
    input.extend_from_slice(&[0x00u8, 0x00, 0x00, 0x00]);
    assert_eq!(input.len(), 16);

    let result = crate::continuation::prove_and_verify_continuation(
        &elf_bytes,
        &input,
        2,
        &ProofOptions::default_test_options(),
    );

    // The continuation must prove and verify without error.
    let output = result.expect("prove_and_verify_continuation must not error");

    // The fixture commits bytes 4..12 of private input (the 8-byte payload).
    assert_eq!(
        output.as_deref(),
        Some(&input[4..12]),
        "committed output must equal private input bytes 4..12"
    );
}

#[test]
fn test_local_memory_bus_balances() {
    // For each touched byte, L2G's init-receive (ts=0) + fini-send cancel the
    // MEMW chain's init-send + fini-receive: the epoch-local Memory bus balances.
    let initial_memory = HashMap::from([(10u64, 5u64)]);
    let epochs = vec![vec![(10, 7, 3), (20, 9, 4)]];
    let boundaries = epoch_boundaries(&initial_memory, &epochs);
    assert!(prove_verify_memory(&boundaries[0], &boundaries[0]));
}

#[test]
fn test_local_memory_bus_rejects_tamper() {
    // L2G claims the real fini value but the access chain ends on a different
    // one — the Memory bus no longer balances.
    let initial_memory = HashMap::from([(10u64, 5u64)]);
    let epochs = vec![vec![(10, 7, 3)]];
    let boundaries = epoch_boundaries(&initial_memory, &epochs);
    assert!(prove_verify_memory(&boundaries[0], &boundaries[0]));

    let mut tampered = boundaries[0].clone();
    tampered[0].fini.value = 999;
    assert!(!prove_verify_memory(&boundaries[0], &tampered));
}
