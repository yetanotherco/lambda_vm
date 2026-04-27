//! Soundness and completeness tests for the unified BITWISE bus.
//!
//! These tests model the real `BusId::Bitwise` protocol: a 4-element token
//! `(op_id, X, Y, RESULT)` where `op_id = AND + 2*OR + 4*XOR` (disjoint-bit
//! encoding). Z values 1, 2, 4 carry the AND, OR, XOR results; other Z
//! values have RESULT = 0 and are absorbed only by accident on a malicious
//! trace that bypasses IS_BIT + DECODE in the real CPU.
//!
//! The receiver here mirrors a slice of the BITWISE table: a single
//! interaction on `BusId::Bitwise` with token `(Z, X, Y, RESULT)`.

use std::collections::HashMap;

use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use math::field::element::FieldElement;

use stark::constraints::transition::TransitionConstraintEvaluator;
use stark::lookup::{
    AirWithBuses, AuxiliaryTraceBuildData, BusInteraction, BusValue, Multiplicity,
    NullBoundaryConstraintBuilder, Packing,
};
use stark::proof::options::ProofOptions;
use stark::prover::{IsStarkProver, Prover};
use stark::trace::TraceTable;
use stark::traits::AIR;
use stark::verifier::{IsStarkVerifier, Verifier};

use crate::tables::types::{BusId, FE, GoldilocksExtension, GoldilocksField};

type F = GoldilocksField;
type E = GoldilocksExtension;

// =============================================================================
// Column indices (mirror the real 4-element token)
// =============================================================================

/// Sender table columns. Token sent: (OP_ID, X, Y, RESULT) with multiplicity MU.
mod sender_cols {
    pub const OP_ID: usize = 0;
    pub const X: usize = 1;
    pub const Y: usize = 2;
    pub const RESULT: usize = 3;
    /// Sender multiplicity (1 in honest traces; 0 on padding rows).
    pub const MU: usize = 4;
    pub const NUM_COLUMNS: usize = 5;
}

/// Receiver (BITWISE-like) table columns. Token received: (Z, X, Y, RESULT).
mod receiver_cols {
    pub const Z: usize = 0;
    pub const X: usize = 1;
    pub const Y: usize = 2;
    pub const RESULT: usize = 3;
    pub const MU_BITWISE: usize = 4;
    pub const NUM_COLUMNS: usize = 5;
}

// =============================================================================
// AIR definitions
// =============================================================================

fn new_sender_air(
    proof_options: &ProofOptions,
) -> AirWithBuses<F, E, NullBoundaryConstraintBuilder, ()> {
    let transition_constraints: Vec<Box<dyn TransitionConstraintEvaluator<F, E>>> = vec![];

    let auxiliary_trace_build_data = AuxiliaryTraceBuildData {
        interactions: vec![BusInteraction::sender(
            BusId::Bitwise,
            Multiplicity::Column(sender_cols::MU),
            vec![
                BusValue::Packed {
                    start_column: sender_cols::OP_ID,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: sender_cols::X,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: sender_cols::Y,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: sender_cols::RESULT,
                    packing: Packing::Direct,
                },
            ],
        )],
    };

    AirWithBuses::new(
        sender_cols::NUM_COLUMNS,
        auxiliary_trace_build_data,
        proof_options,
        1,
        transition_constraints,
    )
}

fn new_receiver_air(
    proof_options: &ProofOptions,
) -> AirWithBuses<F, E, NullBoundaryConstraintBuilder, ()> {
    let transition_constraints: Vec<Box<dyn TransitionConstraintEvaluator<F, E>>> = vec![];

    let auxiliary_trace_build_data = AuxiliaryTraceBuildData {
        interactions: vec![BusInteraction::receiver(
            BusId::Bitwise,
            Multiplicity::Column(receiver_cols::MU_BITWISE),
            vec![
                BusValue::Packed {
                    start_column: receiver_cols::Z,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: receiver_cols::X,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: receiver_cols::Y,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: receiver_cols::RESULT,
                    packing: Packing::Direct,
                },
            ],
        )],
    };

    AirWithBuses::new(
        receiver_cols::NUM_COLUMNS,
        auxiliary_trace_build_data,
        proof_options,
        1,
        transition_constraints,
    )
}

// =============================================================================
// Helpers
// =============================================================================

/// Honest result for a (op_id, x, y) triple.
///
/// Mirrors the precomputed RESULT in the BITWISE table:
/// op_id = 1 → x & y, op_id = 2 → x | y, op_id = 4 → x ^ y, otherwise 0.
fn honest_result(op_id: u8, x: u8, y: u8) -> u8 {
    match op_id {
        1 => x & y,
        2 => x | y,
        4 => x ^ y,
        _ => 0,
    }
}

/// Build a sender trace from `(op_id, x, y, result)` lookups, multiplicity 1.
fn create_sender_trace(lookups: &[(u8, u8, u8, u8)]) -> TraceTable<F, E> {
    let num_rows = lookups.len().next_power_of_two().max(4);
    let mut data = vec![FE::zero(); num_rows * sender_cols::NUM_COLUMNS];

    for (i, &(op_id, x, y, result)) in lookups.iter().enumerate() {
        let base = i * sender_cols::NUM_COLUMNS;
        data[base + sender_cols::OP_ID] = FE::from(op_id as u64);
        data[base + sender_cols::X] = FE::from(x as u64);
        data[base + sender_cols::Y] = FE::from(y as u64);
        data[base + sender_cols::RESULT] = FE::from(result as u64);
        data[base + sender_cols::MU] = FE::one();
    }

    TraceTable::new_main(data, sender_cols::NUM_COLUMNS, 1)
}

/// Build a receiver trace that auto-precomputes the honest RESULT for every
/// `(op_id, x, y)` triple sent and accumulates the matching multiplicities.
///
/// This mirrors the real BITWISE behaviour: the precomputed table fixes
/// RESULT for every (Z, X, Y), and only `MU_BITWISE` is witnessed.
fn create_honest_receiver_trace(lookups: &[(u8, u8, u8, u8)]) -> TraceTable<F, E> {
    let mut multiplicities: HashMap<(u8, u8, u8), u32> = HashMap::new();
    for &(op_id, x, y, _) in lookups {
        *multiplicities.entry((op_id, x, y)).or_insert(0) += 1;
    }

    let unique: Vec<((u8, u8, u8), u32)> = multiplicities.into_iter().collect();
    let num_rows = unique.len().next_power_of_two().max(4);
    let mut data = vec![FE::zero(); num_rows * receiver_cols::NUM_COLUMNS];

    for (i, &((op_id, x, y), mu)) in unique.iter().enumerate() {
        let base = i * receiver_cols::NUM_COLUMNS;
        data[base + receiver_cols::Z] = FE::from(op_id as u64);
        data[base + receiver_cols::X] = FE::from(x as u64);
        data[base + receiver_cols::Y] = FE::from(y as u64);
        data[base + receiver_cols::RESULT] = FE::from(honest_result(op_id, x, y) as u64);
        data[base + receiver_cols::MU_BITWISE] = FE::from(mu as u64);
    }

    TraceTable::new_main(data, receiver_cols::NUM_COLUMNS, 1)
}

/// Build a fully manual receiver trace from `(z, x, y, result, mu)` rows.
fn create_custom_receiver_trace(rows: &[(u8, u8, u8, u8, u32)]) -> TraceTable<F, E> {
    let num_rows = rows.len().next_power_of_two().max(4);
    let mut data = vec![FE::zero(); num_rows * receiver_cols::NUM_COLUMNS];

    for (i, &(z, x, y, result, mu)) in rows.iter().enumerate() {
        let base = i * receiver_cols::NUM_COLUMNS;
        data[base + receiver_cols::Z] = FE::from(z as u64);
        data[base + receiver_cols::X] = FE::from(x as u64);
        data[base + receiver_cols::Y] = FE::from(y as u64);
        data[base + receiver_cols::RESULT] = FE::from(result as u64);
        data[base + receiver_cols::MU_BITWISE] = FE::from(mu as u64);
    }

    TraceTable::new_main(data, receiver_cols::NUM_COLUMNS, 1)
}

fn run_proof(
    mut sender_trace: TraceTable<F, E>,
    mut receiver_trace: TraceTable<F, E>,
) -> bool {
    let proof_options = ProofOptions::default_test_options();
    let sender_air = new_sender_air(&proof_options);
    let receiver_air = new_receiver_air(&proof_options);

    let air_trace_pairs: Vec<(
        &dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>,
        _,
        _,
    )> = vec![
        (&sender_air, &mut sender_trace, &()),
        (&receiver_air, &mut receiver_trace, &()),
    ];

    let multi_proof =
        Prover::multi_prove(air_trace_pairs, &mut DefaultTranscript::<E>::new(&[])).unwrap();

    let airs: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>> =
        vec![&sender_air, &receiver_air];

    Verifier::multi_verify(
        &airs,
        &multi_proof,
        &mut DefaultTranscript::<E>::new(&[]),
        &FieldElement::zero(),
    )
}

/// Sender lookups + auto-precomputed receiver. The bus balances iff every
/// `(op_id, x, y, result)` matches the honest result for that op.
fn prove_and_verify(lookups: &[(u8, u8, u8, u8)]) -> bool {
    run_proof(
        create_sender_trace(lookups),
        create_honest_receiver_trace(lookups),
    )
}

/// Sender lookups + manual receiver rows. Used to inject malicious receiver
/// states (wrong RESULT, wrong multiplicity, missing rows, etc).
fn prove_and_verify_custom(
    sender_lookups: &[(u8, u8, u8, u8)],
    receiver_rows: &[(u8, u8, u8, u8, u32)],
) -> bool {
    run_proof(
        create_sender_trace(sender_lookups),
        create_custom_receiver_trace(receiver_rows),
    )
}

// =============================================================================
// Completeness — valid lookups should be accepted for AND, OR, XOR
// =============================================================================

#[test]
fn test_completeness_and_simple() {
    // op_id = 1 (AND): 5 & 3 = 1
    assert!(prove_and_verify(&[(1, 5, 3, 1)]));
}

#[test]
fn test_completeness_and_zero_result() {
    // 0xAA & 0x55 = 0
    assert!(prove_and_verify(&[(1, 0xAA, 0x55, 0x00)]));
}

#[test]
fn test_completeness_and_max() {
    assert!(prove_and_verify(&[(1, 0xFF, 0xFF, 0xFF)]));
}

#[test]
fn test_completeness_or_simple() {
    // op_id = 2 (OR): 5 | 3 = 7
    assert!(prove_and_verify(&[(2, 5, 3, 7)]));
}

#[test]
fn test_completeness_or_alternating_bits() {
    // 0xAA | 0x55 = 0xFF
    assert!(prove_and_verify(&[(2, 0xAA, 0x55, 0xFF)]));
}

#[test]
fn test_completeness_xor_simple() {
    // op_id = 4 (XOR): 5 ^ 3 = 6
    assert!(prove_and_verify(&[(4, 5, 3, 6)]));
}

#[test]
fn test_completeness_xor_alternating_bits() {
    // 0xAA ^ 0x55 = 0xFF
    assert!(prove_and_verify(&[(4, 0xAA, 0x55, 0xFF)]));
}

#[test]
fn test_completeness_mixed_ops() {
    // Mix the three ops in a single trace.
    let lookups = vec![
        (1u8, 0x12u8, 0x34u8, 0x12u8 & 0x34u8),
        (2u8, 0x12u8, 0x34u8, 0x12u8 | 0x34u8),
        (4u8, 0x12u8, 0x34u8, 0x12u8 ^ 0x34u8),
        (1u8, 0xFFu8, 0xFFu8, 0xFFu8),
    ];
    assert!(prove_and_verify(&lookups));
}

#[test]
fn test_completeness_duplicate_lookups() {
    // Same (op_id, x, y) repeated three times → multiplicity 3 on a single row.
    let lookups = vec![(1u8, 5u8, 3u8, 1u8); 3];
    assert!(prove_and_verify(&lookups));
}

#[test]
fn test_completeness_branch_style_constant_op_id() {
    // Mirrors branch.rs and shift.rs, which always emit op_id = 1 (AND) as a
    // literal constant. Honest result for `unmasked & 0xFE`.
    let lookups = vec![
        (1u8, 0x12u8, 0xFEu8, 0x12u8 & 0xFEu8),
        (1u8, 0xABu8, 0xFEu8, 0xABu8 & 0xFEu8),
    ];
    assert!(prove_and_verify(&lookups));
}

// =============================================================================
// Soundness — invalid lookups should be rejected
// =============================================================================

#[test]
fn test_soundness_wrong_and_result() {
    // 5 & 3 = 1, sender claims 99.
    assert!(!prove_and_verify(&[(1, 5, 3, 99)]));
}

#[test]
fn test_soundness_wrong_or_result() {
    // 5 | 3 = 7, sender claims 0.
    assert!(!prove_and_verify(&[(2, 5, 3, 0)]));
}

#[test]
fn test_soundness_wrong_xor_result() {
    // 5 ^ 3 = 6, sender claims 5.
    assert!(!prove_and_verify(&[(4, 5, 3, 5)]));
}

#[test]
fn test_soundness_op_id_swapped() {
    // Sender emits op_id = 2 (OR) but the receiver row is z = 1 (AND).
    let sender = vec![(2u8, 5u8, 3u8, 7u8)]; // 5 | 3 = 7
    let receiver = vec![(1u8, 5u8, 3u8, 5u8 & 3u8, 1u32)];
    assert!(!prove_and_verify_custom(&sender, &receiver));
}

#[test]
fn test_soundness_multiplicity_mismatch() {
    // Sender sends two AND lookups, receiver claims multiplicity 1.
    let sender = vec![(1u8, 5u8, 3u8, 1u8); 2];
    let receiver = vec![(1u8, 5u8, 3u8, 1u8, 1u32)];
    assert!(!prove_and_verify_custom(&sender, &receiver));
}

#[test]
fn test_soundness_swapped_inputs() {
    // Sender x=3, y=5; receiver row stores x=5, y=3 (different).
    let sender = vec![(1u8, 3u8, 5u8, 1u8)];
    let receiver = vec![(1u8, 5u8, 3u8, 1u8, 1u32)];
    assert!(!prove_and_verify_custom(&sender, &receiver));
}

#[test]
fn test_soundness_extra_sender_lookup() {
    let sender = vec![(1u8, 5u8, 3u8, 1u8), (1u8, 10u8, 6u8, 2u8)];
    let receiver = vec![(1u8, 5u8, 3u8, 1u8, 1u32)];
    assert!(!prove_and_verify_custom(&sender, &receiver));
}

#[test]
fn test_soundness_op_id_zero_with_nonzero_multiplicity() {
    // op_id = 0 corresponds to "no op": no honest CPU sender ever emits a
    // token with op_id = 0 and multiplicity > 0 (Sum3(AND, OR, XOR) = 0).
    // A standalone receiver row at z = 0 with MU > 0 must not balance.
    let sender: Vec<(u8, u8, u8, u8)> = vec![];
    let receiver = vec![(0u8, 5u8, 3u8, 0u8, 1u32)];
    assert!(!prove_and_verify_custom(&sender, &receiver));
}

// =============================================================================
// Multi-flag attack — what the bus *cannot* prevent on its own
// =============================================================================
//
// These tests document the soundness limit of the unified bus: in isolation
// (without IS_BIT + DECODE on the sender's flag columns), a malicious prover
// can balance the bus with op_id ∈ {3, 5, 6, 7} as long as RESULT = 0,
// because BITWISE has precomputed rows at those Z values with RESULT = 0.
// In production this is blocked by IS_BIT (force AND/OR/XOR ∈ {0,1}) and
// the DECODE lookup (force at most one flag set). See cpu.rs and bitwise.rs
// soundness comments.

#[test]
fn test_isolated_bus_accepts_op_id_3_with_zero_result() {
    // op_id = 3 (would correspond to AND=OR=1 in the sender's flag columns,
    // ruled out by IS_BIT + DECODE). RESULT = 0 matches the precomputed row
    // at Z = 3, so the bus balances on its own — this is the gap that
    // IS_BIT + DECODE close in production.
    let sender = vec![(3u8, 5u8, 3u8, 0u8)];
    let receiver = vec![(3u8, 5u8, 3u8, 0u8, 1u32)];
    assert!(prove_and_verify_custom(&sender, &receiver));
}

#[test]
fn test_isolated_bus_rejects_op_id_3_with_nonzero_result() {
    // The Z = 3 precomputed row has RESULT = 0. A sender that emits a
    // non-zero RESULT for op_id = 3 still cannot balance, because the
    // fingerprint depends on RESULT — it would require a receiver row
    // with that exact RESULT and matching multiplicity.
    let sender = vec![(3u8, 5u8, 3u8, 1u8)];
    let receiver = vec![(3u8, 5u8, 3u8, 0u8, 1u32)];
    assert!(!prove_and_verify_custom(&sender, &receiver));
}
