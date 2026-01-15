//! Tests for valid/honest LogUp bus interactions.
//!
//! These tests verify that the prover and verifier work correctly for legitimate use cases.

use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use math::field::element::FieldElement;
use math::field::fields::fft_friendly::{
    babybear::Babybear31PrimeField, quartic_babybear::Degree4BabyBearExtensionField,
};

use crate::examples::multi_table_lookup::{
    new_add_air_with_lookup, new_cpu_air_with_lookup, new_mul_air_with_lookup,
};
use crate::lookup::{
    AirWithBuses, AuxiliaryTraceBuildData, BusInteraction, NullBoundaryConstraintBuilder, Packing,
};
use crate::proof::options::ProofOptions;
use crate::prover::{IsStarkProver, Prover};
use crate::trace::TraceTable;
use crate::traits::AIR;
use crate::verifier::{IsStarkVerifier, Verifier};

type F = Babybear31PrimeField;
type E = Degree4BabyBearExtensionField;
type FE = FieldElement<F>;

// =============================================================================
// Packing combine unit tests
// =============================================================================

#[test]
fn test_packing_direct() {
    // Direct: 1 column -> 1 bus element, no combining
    let values = vec![FE::from(42u64)];
    let combined = Packing::Direct.combine(&values);
    assert_eq!(combined.len(), 1);
    assert_eq!(combined[0], FE::from(42u64));
}

#[test]
fn test_packing_word4l_combine() {
    // 4 bytes: [0x12, 0x34, 0x56, 0x78]
    // Expected: 0x78563412 (little-endian)
    let bytes = vec![
        FE::from(0x12u64),
        FE::from(0x34u64),
        FE::from(0x56u64),
        FE::from(0x78u64),
    ];
    let combined = Packing::Word4L.combine(&bytes);
    assert_eq!(combined.len(), 1);
    assert_eq!(combined[0], FE::from(0x78563412u64));
}

#[test]
fn test_packing_word2l_combine() {
    // 2 halves: [0x1234, 0x5678]
    // Expected: 0x56781234 (little-endian)
    let halves = vec![FE::from(0x1234u64), FE::from(0x5678u64)];
    let combined = Packing::Word2L.combine(&halves);
    assert_eq!(combined.len(), 1);
    assert_eq!(combined[0], FE::from(0x56781234u64));
}

#[test]
fn test_packing_dword_hl_combine() {
    // 4 halves: [0x1234, 0x5678, 0x9ABC, 0xDEF0]
    // Expected: [0x56781234, 0xDEF09ABC]
    let halves = vec![
        FE::from(0x1234u64),
        FE::from(0x5678u64),
        FE::from(0x9ABCu64),
        FE::from(0xDEF0u64),
    ];
    let combined = Packing::DWordHL.combine(&halves);
    assert_eq!(combined.len(), 2);
    assert_eq!(combined[0], FE::from(0x56781234u64));
    assert_eq!(combined[1], FE::from(0xDEF09ABCu64));
}

#[test]
fn test_packing_dword_wl_combine() {
    // 2 words: [0x12345678, 0xDEADBEEF]
    // Expected: [0x12345678, 0xDEADBEEF] (no combining, just 2 bus elements)
    let words = vec![FE::from(0x12345678u64), FE::from(0xDEADBEEFu64)];
    let combined = Packing::DWordWL.combine(&words);
    assert_eq!(combined.len(), 2);
    assert_eq!(combined[0], FE::from(0x12345678u64));
    assert_eq!(combined[1], FE::from(0xDEADBEEFu64));
}

#[test]
fn test_packing_dword_hhw_combine() {
    // [Word, Half, Half] where Word is LSB
    // columns: [0xAABBCCDD, 0x1234, 0x5678]
    // Expected: [0xAABBCCDD, 0x56781234]
    let cols = vec![
        FE::from(0xAABBCCDDu64),
        FE::from(0x1234u64),
        FE::from(0x5678u64),
    ];
    let combined = Packing::DWordHHW.combine(&cols);
    assert_eq!(combined.len(), 2);
    assert_eq!(combined[0], FE::from(0xAABBCCDDu64));
    assert_eq!(combined[1], FE::from(0x56781234u64));
}

// =============================================================================
// AirWithBuses setup tests
// =============================================================================

#[test]
fn test_air_with_buses_layout() {
    // Test that AirWithBuses correctly computes trace layout
    let interaction = BusInteraction::sender(Some(0), Packing::Direct.columns(&[1, 2, 3]));

    let build_data = AuxiliaryTraceBuildData {
        interactions: vec![interaction],
    };

    let proof_options = ProofOptions::default_test_options();
    let air = AirWithBuses::<F, E, NullBoundaryConstraintBuilder, ()>::new(
        4, // 4 main columns
        build_data,
        &proof_options,
        1,
        vec![],
    );

    // 4 main, 2 aux (1 term column + 1 accumulated column)
    assert_eq!(air.trace_layout(), (4, 2));
}

#[test]
fn test_air_with_buses_multiple_interactions() {
    // Multiple interactions should create multiple term columns + 1 accumulated
    let interaction1 = BusInteraction::sender(Some(0), Packing::Direct.columns(&[1, 2]));
    let interaction2 = BusInteraction::sender(Some(0), Packing::Direct.columns(&[3, 4]));

    let build_data = AuxiliaryTraceBuildData {
        interactions: vec![interaction1, interaction2],
    };

    let proof_options = ProofOptions::default_test_options();
    let air = AirWithBuses::<F, E, NullBoundaryConstraintBuilder, ()>::new(
        5, // 5 main columns
        build_data,
        &proof_options,
        1,
        vec![],
    );

    // 5 main, 3 aux (2 term columns + 1 accumulated column)
    assert_eq!(air.trace_layout(), (5, 3));
}

// =============================================================================
// Valid multi-table integration tests
// =============================================================================

/// Standard valid multi-table proof with CPU, ADD, and MUL tables.
#[test_log::test]
fn test_valid_multi_table_proof() {
    // CPU Trace (8 rows): dispatches operations to ADD and MUL tables
    let add_column = vec![
        FE::one(),
        FE::zero(),
        FE::one(),
        FE::zero(),
        FE::one(),
        FE::one(),
        FE::zero(),
        FE::zero(),
    ];
    let mul_column = vec![
        FE::zero(),
        FE::one(),
        FE::zero(),
        FE::one(),
        FE::zero(),
        FE::zero(),
        FE::one(),
        FE::one(),
    ];
    let a_column = vec![
        FE::from(1),
        FE::from(2),
        FE::from(3),
        FE::from(4),
        FE::from(5),
        FE::from(6),
        FE::from(7),
        FE::from(8),
    ];
    let b_column = vec![
        FE::from(10),
        FE::from(20),
        FE::from(30),
        FE::from(40),
        FE::from(50),
        FE::from(60),
        FE::from(70),
        FE::from(80),
    ];
    let c_column = vec![
        FE::from(11),  // 1 + 10
        FE::from(40),  // 2 * 20
        FE::from(33),  // 3 + 30
        FE::from(160), // 4 * 40
        FE::from(55),  // 5 + 50
        FE::from(66),  // 6 + 60
        FE::from(490), // 7 * 70
        FE::from(640), // 8 * 80
    ];
    let mut cpu_trace = TraceTable::from_columns_main(
        vec![add_column, mul_column, a_column, b_column, c_column],
        1,
    );

    // ADD Trace (4 rows): receives addition operations
    let mut add_trace = TraceTable::from_columns_main(
        vec![
            vec![FE::from(1), FE::from(3), FE::from(5), FE::from(6)],
            vec![FE::from(10), FE::from(30), FE::from(50), FE::from(60)],
            vec![FE::from(11), FE::from(33), FE::from(55), FE::from(66)],
            vec![FE::one(), FE::one(), FE::one(), FE::one()],
        ],
        1,
    );

    // MUL Trace (4 rows): receives multiplication operations
    let mut mul_trace = TraceTable::from_columns_main(
        vec![
            vec![FE::from(2), FE::from(4), FE::from(7), FE::from(8)],
            vec![FE::from(20), FE::from(40), FE::from(70), FE::from(80)],
            vec![FE::from(40), FE::from(160), FE::from(490), FE::from(640)],
            vec![FE::one(), FE::one(), FE::one(), FE::one()],
        ],
        1,
    );

    let proof_options = ProofOptions::default_test_options();
    let cpu_air = new_cpu_air_with_lookup(&proof_options);
    let add_air = new_add_air_with_lookup(&proof_options);
    let mul_air = new_mul_air_with_lookup(&proof_options);

    let air_trace_pairs: Vec<(
        &dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>,
        _,
        _,
    )> = vec![
        (&cpu_air, &mut cpu_trace, &()),
        (&add_air, &mut add_trace, &()),
        (&mul_air, &mut mul_trace, &()),
    ];

    let multi_proof =
        Prover::multi_prove(air_trace_pairs, &mut DefaultTranscript::<E>::new(&[])).unwrap();

    let airs: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>> =
        vec![&cpu_air, &add_air, &mul_air];

    assert!(Verifier::multi_verify(
        &airs,
        &multi_proof,
        &mut DefaultTranscript::<E>::new(&[]),
    ));
}

/// Test with all padding rows (multiplicity = 0 everywhere).
/// Bus should balance at zero.
#[test_log::test]
fn test_valid_all_padding() {
    // CPU Trace: all flags are 0, so nothing is sent
    let mut cpu_trace = TraceTable::from_columns_main(
        vec![
            vec![FE::zero(); 4], // add_flag = 0
            vec![FE::zero(); 4], // mul_flag = 0
            vec![FE::zero(); 4], // a
            vec![FE::zero(); 4], // b
            vec![FE::zero(); 4], // c
        ],
        1,
    );

    // ADD Trace: multiplicity = 0, nothing received
    let mut add_trace = TraceTable::from_columns_main(
        vec![
            vec![FE::zero(); 4], // a
            vec![FE::zero(); 4], // b
            vec![FE::zero(); 4], // c
            vec![FE::zero(); 4], // multiplicity = 0
        ],
        1,
    );

    // MUL Trace: multiplicity = 0, nothing received
    let mut mul_trace = TraceTable::from_columns_main(
        vec![
            vec![FE::zero(); 4], // a
            vec![FE::zero(); 4], // b
            vec![FE::zero(); 4], // c
            vec![FE::zero(); 4], // multiplicity = 0
        ],
        1,
    );

    let proof_options = ProofOptions::default_test_options();
    let cpu_air = new_cpu_air_with_lookup(&proof_options);
    let add_air = new_add_air_with_lookup(&proof_options);
    let mul_air = new_mul_air_with_lookup(&proof_options);

    let air_trace_pairs: Vec<(
        &dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>,
        _,
        _,
    )> = vec![
        (&cpu_air, &mut cpu_trace, &()),
        (&add_air, &mut add_trace, &()),
        (&mul_air, &mut mul_trace, &()),
    ];

    let multi_proof =
        Prover::multi_prove(air_trace_pairs, &mut DefaultTranscript::<E>::new(&[])).unwrap();

    let airs: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>> =
        vec![&cpu_air, &add_air, &mul_air];

    assert!(Verifier::multi_verify(
        &airs,
        &multi_proof,
        &mut DefaultTranscript::<E>::new(&[]),
    ));
}

/// Test with a single operation (minimal non-trivial case).
#[test_log::test]
fn test_valid_single_operation() {
    // CPU Trace: single ADD operation in first row, rest padding
    let mut cpu_trace = TraceTable::from_columns_main(
        vec![
            vec![FE::one(), FE::zero(), FE::zero(), FE::zero()], // add_flag
            vec![FE::zero(); 4],                                 // mul_flag
            vec![FE::from(5), FE::zero(), FE::zero(), FE::zero()], // a
            vec![FE::from(3), FE::zero(), FE::zero(), FE::zero()], // b
            vec![FE::from(8), FE::zero(), FE::zero(), FE::zero()], // c = 5 + 3
        ],
        1,
    );

    // ADD Trace: receives the single operation
    let mut add_trace = TraceTable::from_columns_main(
        vec![
            vec![FE::from(5), FE::zero(), FE::zero(), FE::zero()], // a
            vec![FE::from(3), FE::zero(), FE::zero(), FE::zero()], // b
            vec![FE::from(8), FE::zero(), FE::zero(), FE::zero()], // c
            vec![FE::one(), FE::zero(), FE::zero(), FE::zero()],   // multiplicity
        ],
        1,
    );

    // MUL Trace: empty (all padding)
    let mut mul_trace = TraceTable::from_columns_main(
        vec![
            vec![FE::zero(); 4],
            vec![FE::zero(); 4],
            vec![FE::zero(); 4],
            vec![FE::zero(); 4],
        ],
        1,
    );

    let proof_options = ProofOptions::default_test_options();
    let cpu_air = new_cpu_air_with_lookup(&proof_options);
    let add_air = new_add_air_with_lookup(&proof_options);
    let mul_air = new_mul_air_with_lookup(&proof_options);

    let air_trace_pairs: Vec<(
        &dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>,
        _,
        _,
    )> = vec![
        (&cpu_air, &mut cpu_trace, &()),
        (&add_air, &mut add_trace, &()),
        (&mul_air, &mut mul_trace, &()),
    ];

    let multi_proof =
        Prover::multi_prove(air_trace_pairs, &mut DefaultTranscript::<E>::new(&[])).unwrap();

    let airs: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>> =
        vec![&cpu_air, &add_air, &mul_air];

    assert!(Verifier::multi_verify(
        &airs,
        &multi_proof,
        &mut DefaultTranscript::<E>::new(&[]),
    ));
}

/// Test duplicate operations: same (a,b,c) sent twice, received twice.
#[test_log::test]
fn test_valid_duplicate_operations() {
    // CPU Trace: sends (5, 3, 8) twice to ADD
    let mut cpu_trace = TraceTable::from_columns_main(
        vec![
            vec![FE::one(), FE::one(), FE::zero(), FE::zero()], // add_flag (2 sends)
            vec![FE::zero(); 4],                                // mul_flag
            vec![FE::from(5), FE::from(5), FE::zero(), FE::zero()], // a (same value)
            vec![FE::from(3), FE::from(3), FE::zero(), FE::zero()], // b (same value)
            vec![FE::from(8), FE::from(8), FE::zero(), FE::zero()], // c (same value)
        ],
        1,
    );

    // ADD Trace: receives (5,3,8) with multiplicity=2 in one row
    let mut add_trace = TraceTable::from_columns_main(
        vec![
            vec![FE::from(5), FE::zero(), FE::zero(), FE::zero()],
            vec![FE::from(3), FE::zero(), FE::zero(), FE::zero()],
            vec![FE::from(8), FE::zero(), FE::zero(), FE::zero()],
            vec![FE::from(2), FE::zero(), FE::zero(), FE::zero()], // multiplicity = 2
        ],
        1,
    );

    // MUL Trace: empty
    let mut mul_trace = TraceTable::from_columns_main(
        vec![
            vec![FE::zero(); 4],
            vec![FE::zero(); 4],
            vec![FE::zero(); 4],
            vec![FE::zero(); 4],
        ],
        1,
    );

    let proof_options = ProofOptions::default_test_options();
    let cpu_air = new_cpu_air_with_lookup(&proof_options);
    let add_air = new_add_air_with_lookup(&proof_options);
    let mul_air = new_mul_air_with_lookup(&proof_options);

    let air_trace_pairs: Vec<(
        &dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>,
        _,
        _,
    )> = vec![
        (&cpu_air, &mut cpu_trace, &()),
        (&add_air, &mut add_trace, &()),
        (&mul_air, &mut mul_trace, &()),
    ];

    let multi_proof =
        Prover::multi_prove(air_trace_pairs, &mut DefaultTranscript::<E>::new(&[])).unwrap();

    let airs: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>> =
        vec![&cpu_air, &add_air, &mul_air];

    assert!(Verifier::multi_verify(
        &airs,
        &multi_proof,
        &mut DefaultTranscript::<E>::new(&[]),
    ));
}

/// Test serialization round-trip: proof survives serialize/deserialize.
#[test_log::test]
fn test_valid_proof_serialization_roundtrip() {
    // Simple valid trace
    let mut cpu_trace = TraceTable::from_columns_main(
        vec![
            vec![FE::one(), FE::zero(), FE::zero(), FE::zero()],
            vec![FE::zero(); 4],
            vec![FE::from(1), FE::zero(), FE::zero(), FE::zero()],
            vec![FE::from(2), FE::zero(), FE::zero(), FE::zero()],
            vec![FE::from(3), FE::zero(), FE::zero(), FE::zero()],
        ],
        1,
    );

    let mut add_trace = TraceTable::from_columns_main(
        vec![
            vec![FE::from(1), FE::zero(), FE::zero(), FE::zero()],
            vec![FE::from(2), FE::zero(), FE::zero(), FE::zero()],
            vec![FE::from(3), FE::zero(), FE::zero(), FE::zero()],
            vec![FE::one(), FE::zero(), FE::zero(), FE::zero()],
        ],
        1,
    );

    let mut mul_trace = TraceTable::from_columns_main(
        vec![
            vec![FE::zero(); 4],
            vec![FE::zero(); 4],
            vec![FE::zero(); 4],
            vec![FE::zero(); 4],
        ],
        1,
    );

    let proof_options = ProofOptions::default_test_options();
    let cpu_air = new_cpu_air_with_lookup(&proof_options);
    let add_air = new_add_air_with_lookup(&proof_options);
    let mul_air = new_mul_air_with_lookup(&proof_options);

    let air_trace_pairs: Vec<(
        &dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>,
        _,
        _,
    )> = vec![
        (&cpu_air, &mut cpu_trace, &()),
        (&add_air, &mut add_trace, &()),
        (&mul_air, &mut mul_trace, &()),
    ];

    let multi_proof =
        Prover::multi_prove(air_trace_pairs, &mut DefaultTranscript::<E>::new(&[])).unwrap();

    // Serialize and deserialize
    let serialized = serde_cbor::to_vec(&multi_proof).expect("serialization failed");
    let deserialized: crate::proof::stark::MultiProof<F, E, ()> =
        serde_cbor::from_slice(&serialized).expect("deserialization failed");

    let airs: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>> =
        vec![&cpu_air, &add_air, &mul_air];

    // Verify the deserialized proof
    assert!(Verifier::multi_verify(
        &airs,
        &deserialized,
        &mut DefaultTranscript::<E>::new(&[]),
    ));
}
