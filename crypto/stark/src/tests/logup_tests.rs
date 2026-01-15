//! Tests for the LogUp lookup mechanism (bus interactions, packing, multi-table proving).

use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use math::field::element::FieldElement;
use math::field::fields::fft_friendly::{
    babybear::Babybear31PrimeField, babybear_u32::Babybear31PrimeField as Babybear31PrimeFieldU32,
    quartic_babybear::Degree4BabyBearExtensionField,
    quartic_babybear_u32::Degree4BabyBearU32ExtensionField,
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

type FE = FieldElement<Babybear31PrimeField>;

// =============================================================================
// Unit tests for Packing combine logic
// =============================================================================

type F = Babybear31PrimeFieldU32;
type E = Degree4BabyBearU32ExtensionField;
type FEU32 = FieldElement<F>;

#[test]
fn test_word4l_combine() {
    // 4 bytes: [0x12, 0x34, 0x56, 0x78]
    // Expected: 0x78563412 (little-endian)
    let bytes = vec![
        FEU32::from(0x12u64),
        FEU32::from(0x34u64),
        FEU32::from(0x56u64),
        FEU32::from(0x78u64),
    ];
    let combined = Packing::Word4L.combine(&bytes);
    assert_eq!(combined.len(), 1);
    // 0x12 + 0x34*256 + 0x56*65536 + 0x78*16777216 = 2018915346
    assert_eq!(combined[0], FEU32::from(0x78563412u64));
}

#[test]
fn test_dword_hl_combine() {
    // 4 halves: [0x1234, 0x5678, 0x9ABC, 0xDEF0]
    // Expected: [0x56781234, 0xDEF09ABC]
    let halves = vec![
        FEU32::from(0x1234u64),
        FEU32::from(0x5678u64),
        FEU32::from(0x9ABCu64),
        FEU32::from(0xDEF0u64),
    ];
    let combined = Packing::DWordHL.combine(&halves);
    assert_eq!(combined.len(), 2);
    assert_eq!(combined[0], FEU32::from(0x56781234u64));
    assert_eq!(combined[1], FEU32::from(0xDEF09ABCu64));
}

#[test]
fn test_dword_hhw_combine() {
    // [Word, Half, Half] where Word is LSB
    // columns: [0xAABBCCDD, 0x1234, 0x5678]
    // Expected: [0xAABBCCDD, 0x56781234]
    let cols = vec![
        FEU32::from(0xAABBCCDDu64),
        FEU32::from(0x1234u64),
        FEU32::from(0x5678u64),
    ];
    let combined = Packing::DWordHHW.combine(&cols);
    assert_eq!(combined.len(), 2);
    assert_eq!(combined[0], FEU32::from(0xAABBCCDDu64));
    assert_eq!(combined[1], FEU32::from(0x56781234u64));
}

// =============================================================================
// Unit tests for AirWithBuses
// =============================================================================

/// Test that typed interactions work with a simple sender/receiver pair.
#[test]
fn test_typed_air_with_buses_simple() {
    // Create a simple trace with 4 rows
    // Columns: [mult, a, b, c] where c = a + b (conceptually)
    let num_rows = 4;
    let num_cols = 4;

    let mut columns: Vec<Vec<FEU32>> = vec![vec![FEU32::zero(); num_rows]; num_cols];

    // Fill with test data
    // Row 0: mult=1, a=10, b=20, c=30
    // Row 1: mult=1, a=5, b=15, c=20
    // Row 2: mult=0, a=0, b=0, c=0 (padding)
    // Row 3: mult=0, a=0, b=0, c=0 (padding)
    columns[0][0] = FEU32::from(1u64); // mult
    columns[1][0] = FEU32::from(10u64); // a
    columns[2][0] = FEU32::from(20u64); // b
    columns[3][0] = FEU32::from(30u64); // c

    columns[0][1] = FEU32::from(1u64);
    columns[1][1] = FEU32::from(5u64);
    columns[2][1] = FEU32::from(15u64);
    columns[3][1] = FEU32::from(20u64);

    // Create trace
    let mut trace: TraceTable<F, E> = TraceTable::from_columns_main(columns, 1);

    // Define a typed interaction using Direct (no combining)
    let interaction = BusInteraction::sender(
        Some(0),                             // multiplicity column
        Packing::Direct.columns(&[1, 2, 3]), // a, b, c
    );

    let build_data = AuxiliaryTraceBuildData {
        interactions: vec![interaction],
    };

    // Create AIR
    let proof_options = ProofOptions::default_test_options();
    let air = AirWithBuses::<F, E, NullBoundaryConstraintBuilder, ()>::new(
        num_cols,
        build_data,
        &proof_options,
        1,
        vec![],
    );

    // Verify layout
    assert_eq!(air.trace_layout(), (4, 2)); // 4 main, 2 aux (1 term + 1 acc)

    // Build auxiliary trace with dummy challenges
    let z = FieldElement::<E>::from(12345u64);
    let alpha = FieldElement::<E>::from(67890u64);
    let challenges = vec![z, alpha];

    let bus_public_inputs = air.build_auxiliary_trace(&mut trace, &challenges);

    // Should have bus public inputs
    assert!(bus_public_inputs.is_some());
    let bpi = bus_public_inputs.unwrap();

    // Final accumulated should be non-zero (we have 2 rows with mult=1)
    assert_ne!(bpi.final_accumulated, FieldElement::<E>::zero());
}

/// Test with DWordWL type (2 columns combined)
#[test]
fn test_typed_interaction_dword_wl() {
    // Columns: [mult, lhs_lo, lhs_hi, rhs_lo, rhs_hi, sum_lo, sum_hi]
    let num_rows = 4;
    let num_cols = 7;

    let mut columns: Vec<Vec<FEU32>> = vec![vec![FEU32::zero(); num_rows]; num_cols];

    // Row 0: 100 + 200 = 300 (as 64-bit split into two 32-bit words)
    columns[0][0] = FEU32::from(1u64); // mult
    columns[1][0] = FEU32::from(100u64); // lhs_lo
    columns[2][0] = FEU32::from(0u64); // lhs_hi
    columns[3][0] = FEU32::from(200u64); // rhs_lo
    columns[4][0] = FEU32::from(0u64); // rhs_hi
    columns[5][0] = FEU32::from(300u64); // sum_lo
    columns[6][0] = FEU32::from(0u64); // sum_hi

    let mut trace: TraceTable<F, E> = TraceTable::from_columns_main(columns, 1);

    // Define interaction with DWordWL types
    // lhs: columns 1,2 | rhs: columns 3,4 | sum: columns 5,6
    let interaction = BusInteraction::sender(Some(0), Packing::DWordWL.columns(&[1, 3, 5]));

    let build_data = AuxiliaryTraceBuildData {
        interactions: vec![interaction],
    };

    let proof_options = ProofOptions::default_test_options();
    let air = AirWithBuses::<F, E, NullBoundaryConstraintBuilder, ()>::new(
        num_cols,
        build_data,
        &proof_options,
        1,
        vec![],
    );

    let z = FieldElement::<E>::from(99999u64);
    let alpha = FieldElement::<E>::from(11111u64);
    let challenges = vec![z, alpha];

    let bus_public_inputs = air.build_auxiliary_trace(&mut trace, &challenges);
    assert!(bus_public_inputs.is_some());
}

// =============================================================================
// Integration tests for multi-table LogUp proving/verification
// =============================================================================

#[test_log::test]
fn test_multi_airs_log_up() {
    // CPU Trace
    // ADD | MUL | a | b  | c   | aux add | aux mul | aux total
    // 1   | 0   | 1 | 10 | 11  | 0       | 0       | 0
    // 0   | 1   | 2 | 20 | 40  | 0       | 0       | 0
    // 1   | 0   | 3 | 30 | 33  | 0       | 0       | 0
    // 0   | 1   | 4 | 40 | 160 | 0       | 0       | 0
    // 1   | 0   | 5 | 50 | 55  | 0       | 0       | 0
    // 1   | 0   | 6 | 60 | 66  | 0       | 0       | 0
    // 0   | 1   | 7 | 70 | 490 | 0       | 0       | 0
    // 0   | 1   | 8 | 80 | 640 | 0       | 0       | 0
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
        FE::from(11),
        FE::from(40),
        FE::from(33),
        FE::from(160),
        FE::from(55),
        FE::from(66),
        FE::from(490),
        FE::from(640),
    ];
    let main_columns = vec![add_column, mul_column, a_column, b_column, c_column];
    let mut cpu_trace = TraceTable::from_columns_main(main_columns, 1);

    // ADD Trace
    // a | b  | c  | m | aux cpu
    // 1 | 10 | 11 | 1 |  0
    // 3 | 30 | 33 | 1 |  0
    // 5 | 50 | 55 | 1 |  0
    // 6 | 60 | 66 | 1 |  0
    let a_column = vec![FE::from(1), FE::from(3), FE::from(5), FE::from(6)];
    let b_column = vec![FE::from(10), FE::from(30), FE::from(50), FE::from(60)];
    let c_column = vec![FE::from(11), FE::from(33), FE::from(55), FE::from(66)];
    let m_column = vec![FE::one(), FE::one(), FE::one(), FE::one()];
    let mut add_trace =
        TraceTable::from_columns_main(vec![a_column, b_column, c_column, m_column], 1);

    // MUL Trace
    // a | b  | c   | m | aux cpu
    // 2 | 20 | 40  | 1 |   0
    // 4 | 40 | 160 | 1 |   0
    // 7 | 70 | 490 | 1 |   0
    // 8 | 80 | 640 | 1 |   0
    let a_column = vec![FE::from(2), FE::from(4), FE::from(7), FE::from(8)];
    let b_column = vec![FE::from(20), FE::from(40), FE::from(70), FE::from(80)];
    let c_column = vec![FE::from(40), FE::from(160), FE::from(490), FE::from(640)];
    let m_column = vec![FE::one(), FE::one(), FE::one(), FE::one()];
    let mut mul_trace =
        TraceTable::from_columns_main(vec![a_column, b_column, c_column, m_column], 1);

    let proof_options = ProofOptions::default_test_options();

    let cpu_air = new_cpu_air_with_lookup(&proof_options);
    let add_air = new_add_air_with_lookup(&proof_options);
    let mul_air = new_mul_air_with_lookup(&proof_options);

    let air_trace_pairs: Vec<(
        &dyn AIR<
            Field = Babybear31PrimeField,
            FieldExtension = Degree4BabyBearExtensionField,
            PublicInputs = (),
        >,
        &mut TraceTable<Babybear31PrimeField, Degree4BabyBearExtensionField>,
        &(),
    )> = vec![
        (&cpu_air, &mut cpu_trace, &()),
        (&add_air, &mut add_trace, &()),
        (&mul_air, &mut mul_trace, &()),
    ];

    let multi_proof = Prover::multi_prove(
        air_trace_pairs,
        &mut DefaultTranscript::<Degree4BabyBearExtensionField>::new(&[]),
    )
    .unwrap();

    let airs: Vec<
        &dyn AIR<
            Field = Babybear31PrimeField,
            FieldExtension = Degree4BabyBearExtensionField,
            PublicInputs = (),
        >,
    > = vec![&cpu_air, &add_air, &mul_air];

    assert!(Verifier::multi_verify(
        &airs,
        &multi_proof,
        &mut DefaultTranscript::<Degree4BabyBearExtensionField>::new(&[]),
    ));
}

/// Test that detects when lookup values don't match between tables.
///
/// This simulates a cheating prover who tries to claim that the CPU performed
/// an addition (1 + 10 = 11) but the ADD table has a different result (1 + 10 = 99).
///
/// The verifier detects this because the LogUp bus does not balance.
#[test_log::test]
fn test_multi_airs_log_up_cheating_wrong_value_detected() {
    // CPU Trace - same as valid test
    // CPU claims: 1 + 10 = 11 (row 0, ADD operation)
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
        FE::from(11), // CPU claims 1 + 10 = 11
        FE::from(40),
        FE::from(33),
        FE::from(160),
        FE::from(55),
        FE::from(66),
        FE::from(490),
        FE::from(640),
    ];
    let main_columns = vec![add_column, mul_column, a_column, b_column, c_column];
    let mut cpu_trace = TraceTable::from_columns_main(main_columns, 1);

    // CHEATING ADD Trace - wrong value in first row!
    // CPU sent (1, 10, 11) but ADD table has (1, 10, 99) - MISMATCH!
    let a_column = vec![FE::from(1), FE::from(3), FE::from(5), FE::from(6)];
    let b_column = vec![FE::from(10), FE::from(30), FE::from(50), FE::from(60)];
    let c_column = vec![
        FE::from(99), // CHEAT: Wrong result! Should be 11
        FE::from(33),
        FE::from(55),
        FE::from(66),
    ];
    let m_column = vec![FE::one(), FE::one(), FE::one(), FE::one()];
    let mut add_trace =
        TraceTable::from_columns_main(vec![a_column, b_column, c_column, m_column], 1);

    // MUL Trace - correct values
    let a_column = vec![FE::from(2), FE::from(4), FE::from(7), FE::from(8)];
    let b_column = vec![FE::from(20), FE::from(40), FE::from(70), FE::from(80)];
    let c_column = vec![FE::from(40), FE::from(160), FE::from(490), FE::from(640)];
    let m_column = vec![FE::one(), FE::one(), FE::one(), FE::one()];
    let mut mul_trace =
        TraceTable::from_columns_main(vec![a_column, b_column, c_column, m_column], 1);

    let proof_options = ProofOptions::default_test_options();

    let cpu_air = new_cpu_air_with_lookup(&proof_options);
    let add_air = new_add_air_with_lookup(&proof_options);
    let mul_air = new_mul_air_with_lookup(&proof_options);

    let air_trace_pairs: Vec<(
        &dyn AIR<
            Field = Babybear31PrimeField,
            FieldExtension = Degree4BabyBearExtensionField,
            PublicInputs = (),
        >,
        &mut TraceTable<Babybear31PrimeField, Degree4BabyBearExtensionField>,
        &(),
    )> = vec![
        (&cpu_air, &mut cpu_trace, &()),
        (&add_air, &mut add_trace, &()),
        (&mul_air, &mut mul_trace, &()),
    ];

    let multi_proof = Prover::multi_prove(
        air_trace_pairs,
        &mut DefaultTranscript::<Degree4BabyBearExtensionField>::new(&[]),
    )
    .unwrap();

    let airs: Vec<
        &dyn AIR<
            Field = Babybear31PrimeField,
            FieldExtension = Degree4BabyBearExtensionField,
            PublicInputs = (),
        >,
    > = vec![&cpu_air, &add_air, &mul_air];

    // Verifier should reject because bus does not balance
    assert!(!Verifier::multi_verify(
        &airs,
        &multi_proof,
        &mut DefaultTranscript::<Degree4BabyBearExtensionField>::new(&[]),
    ));
}

/// Test that detects when multiplicity is wrong between tables.
///
/// This simulates a cheating prover who claims the ADD table processed
/// a row twice (multiplicity=2) when the CPU only sent it once.
///
/// The verifier detects this because the LogUp bus does not balance.
#[test_log::test]
fn test_multi_airs_log_up_cheating_wrong_multiplicity_detected() {
    // CPU Trace - sends (1, 10, 11) once via ADD flag
    let add_column = vec![
        FE::one(), // Send to ADD once
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
        FE::from(11),
        FE::from(40),
        FE::from(33),
        FE::from(160),
        FE::from(55),
        FE::from(66),
        FE::from(490),
        FE::from(640),
    ];
    let main_columns = vec![add_column, mul_column, a_column, b_column, c_column];
    let mut cpu_trace = TraceTable::from_columns_main(main_columns, 1);

    // CHEATING ADD Trace - wrong multiplicity!
    // First row claims multiplicity=2, but CPU only sent it once
    let a_column = vec![FE::from(1), FE::from(3), FE::from(5), FE::from(6)];
    let b_column = vec![FE::from(10), FE::from(30), FE::from(50), FE::from(60)];
    let c_column = vec![FE::from(11), FE::from(33), FE::from(55), FE::from(66)];
    let m_column = vec![
        FE::from(2), // CHEAT: Claims multiplicity 2, but CPU sent only 1
        FE::one(),
        FE::one(),
        FE::one(),
    ];
    let mut add_trace =
        TraceTable::from_columns_main(vec![a_column, b_column, c_column, m_column], 1);

    // MUL Trace - correct
    let a_column = vec![FE::from(2), FE::from(4), FE::from(7), FE::from(8)];
    let b_column = vec![FE::from(20), FE::from(40), FE::from(70), FE::from(80)];
    let c_column = vec![FE::from(40), FE::from(160), FE::from(490), FE::from(640)];
    let m_column = vec![FE::one(), FE::one(), FE::one(), FE::one()];
    let mut mul_trace =
        TraceTable::from_columns_main(vec![a_column, b_column, c_column, m_column], 1);

    let proof_options = ProofOptions::default_test_options();

    let cpu_air = new_cpu_air_with_lookup(&proof_options);
    let add_air = new_add_air_with_lookup(&proof_options);
    let mul_air = new_mul_air_with_lookup(&proof_options);

    let air_trace_pairs: Vec<(
        &dyn AIR<
            Field = Babybear31PrimeField,
            FieldExtension = Degree4BabyBearExtensionField,
            PublicInputs = (),
        >,
        &mut TraceTable<Babybear31PrimeField, Degree4BabyBearExtensionField>,
        &(),
    )> = vec![
        (&cpu_air, &mut cpu_trace, &()),
        (&add_air, &mut add_trace, &()),
        (&mul_air, &mut mul_trace, &()),
    ];

    let multi_proof = Prover::multi_prove(
        air_trace_pairs,
        &mut DefaultTranscript::<Degree4BabyBearExtensionField>::new(&[]),
    )
    .unwrap();

    let airs: Vec<
        &dyn AIR<
            Field = Babybear31PrimeField,
            FieldExtension = Degree4BabyBearExtensionField,
            PublicInputs = (),
        >,
    > = vec![&cpu_air, &add_air, &mul_air];

    // Verifier should reject because bus does not balance
    assert!(!Verifier::multi_verify(
        &airs,
        &multi_proof,
        &mut DefaultTranscript::<Degree4BabyBearExtensionField>::new(&[]),
    ));
}
