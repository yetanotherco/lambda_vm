#![allow(clippy::type_complexity)]

use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use math::field::element::FieldElement;
use math::field::fields::fft_friendly::{
    babybear::Babybear31PrimeField, quartic_babybear::Degree4BabyBearExtensionField,
};

use crate::examples::multi_table_lookup::{
    new_add_air_with_lookup, new_cpu_air_with_lookup, new_mul_air_with_lookup,
};
use crate::trace::TraceTable;
use crate::traits::AIR;
use crate::{
    proof::options::ProofOptions,
    prover::{IsStarkProver, Prover},
    verifier::{IsStarkVerifier, Verifier},
};

type F = Babybear31PrimeField;
type E = Degree4BabyBearExtensionField;
type FE = FieldElement<F>;

/// Creates the standard CPU trace columns for tests.
fn create_cpu_columns() -> (Vec<FE>, Vec<FE>, Vec<FE>, Vec<FE>, Vec<FE>) {
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
    (add_column, mul_column, a_column, b_column, c_column)
}

/// Creates the standard ADD trace columns.
fn create_add_columns() -> (Vec<FE>, Vec<FE>, Vec<FE>, Vec<FE>) {
    (
        vec![FE::from(1), FE::from(3), FE::from(5), FE::from(6)],
        vec![FE::from(10), FE::from(30), FE::from(50), FE::from(60)],
        vec![FE::from(11), FE::from(33), FE::from(55), FE::from(66)],
        vec![FE::one(); 4],
    )
}

/// Creates the standard MUL trace columns.
fn create_mul_columns() -> (Vec<FE>, Vec<FE>, Vec<FE>, Vec<FE>) {
    (
        vec![FE::from(2), FE::from(4), FE::from(7), FE::from(8)],
        vec![FE::from(20), FE::from(40), FE::from(70), FE::from(80)],
        vec![FE::from(40), FE::from(160), FE::from(490), FE::from(640)],
        vec![FE::one(); 4],
    )
}

/// Creates all traces for the multi-table test scenario.
fn create_test_traces() -> (TraceTable<F, E>, TraceTable<F, E>, TraceTable<F, E>) {
    let (add_col, mul_col, a_col, b_col, c_col) = create_cpu_columns();
    let cpu_trace =
        TraceTable::from_columns_main(vec![add_col, mul_col, a_col, b_col, c_col], 1);

    let (add_a, add_b, add_c, add_m) = create_add_columns();
    let add_trace = TraceTable::from_columns_main(vec![add_a, add_b, add_c, add_m], 1);

    let (mul_a, mul_b, mul_c, mul_m) = create_mul_columns();
    let mul_trace = TraceTable::from_columns_main(vec![mul_a, mul_b, mul_c, mul_m], 1);

    (cpu_trace, add_trace, mul_trace)
}

#[test_log::test]
fn test_multi_airs_log_up() {
    let (mut cpu_trace, mut add_trace, mut mul_trace) = create_test_traces();
    let proof_options = ProofOptions::default_test_options();

    let cpu_air = new_cpu_air_with_lookup(&proof_options);
    let add_air = new_add_air_with_lookup(&proof_options);
    let mul_air = new_mul_air_with_lookup(&proof_options);

    let air_trace_pairs: Vec<(
        &dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>,
        &mut TraceTable<F, E>,
        &(),
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

/// Test that detects when lookup values don't match between tables.
///
/// This simulates a cheating prover who tries to claim that the CPU performed
/// an addition (1 + 10 = 11) but the ADD table has a different result (1 + 10 = 99).
#[test_log::test]
fn test_multi_airs_log_up_cheating_wrong_value_detected() {
    let (mut cpu_trace, _, mut mul_trace) = create_test_traces();

    // Cheating ADD trace with wrong result in first row
    let (add_a, add_b, _, add_m) = create_add_columns();
    let cheating_add_c = vec![
        FE::from(99), // Wrong! Should be 11
        FE::from(33),
        FE::from(55),
        FE::from(66),
    ];
    let mut add_trace = TraceTable::from_columns_main(vec![add_a, add_b, cheating_add_c, add_m], 1);

    let proof_options = ProofOptions::default_test_options();
    let cpu_air = new_cpu_air_with_lookup(&proof_options);
    let add_air = new_add_air_with_lookup(&proof_options);
    let mul_air = new_mul_air_with_lookup(&proof_options);

    let air_trace_pairs: Vec<(
        &dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>,
        &mut TraceTable<F, E>,
        &(),
    )> = vec![
        (&cpu_air, &mut cpu_trace, &()),
        (&add_air, &mut add_trace, &()),
        (&mul_air, &mut mul_trace, &()),
    ];

    let multi_proof =
        Prover::multi_prove(air_trace_pairs, &mut DefaultTranscript::<E>::new(&[])).unwrap();

    let airs: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>> =
        vec![&cpu_air, &add_air, &mul_air];

    // Verifier should reject because bus does not balance
    assert!(!Verifier::multi_verify(
        &airs,
        &multi_proof,
        &mut DefaultTranscript::<E>::new(&[]),
    ));
}

/// Test that detects when multiplicity is wrong between tables.
///
/// This simulates a cheating prover who claims the ADD table processed
/// a row twice (multiplicity=2) when the CPU only sent it once.
#[test_log::test]
fn test_multi_airs_log_up_cheating_wrong_multiplicity_detected() {
    let (mut cpu_trace, _, mut mul_trace) = create_test_traces();

    // Cheating ADD trace with wrong multiplicity
    let (add_a, add_b, add_c, _) = create_add_columns();
    let cheating_add_m = vec![
        FE::from(2), // Wrong! CPU only sent once
        FE::one(),
        FE::one(),
        FE::one(),
    ];
    let mut add_trace = TraceTable::from_columns_main(vec![add_a, add_b, add_c, cheating_add_m], 1);

    let proof_options = ProofOptions::default_test_options();
    let cpu_air = new_cpu_air_with_lookup(&proof_options);
    let add_air = new_add_air_with_lookup(&proof_options);
    let mul_air = new_mul_air_with_lookup(&proof_options);

    let air_trace_pairs: Vec<(
        &dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>,
        &mut TraceTable<F, E>,
        &(),
    )> = vec![
        (&cpu_air, &mut cpu_trace, &()),
        (&add_air, &mut add_trace, &()),
        (&mul_air, &mut mul_trace, &()),
    ];

    let multi_proof =
        Prover::multi_prove(air_trace_pairs, &mut DefaultTranscript::<E>::new(&[])).unwrap();

    let airs: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>> =
        vec![&cpu_air, &add_air, &mul_air];

    // Verifier should reject because bus does not balance
    assert!(!Verifier::multi_verify(
        &airs,
        &multi_proof,
        &mut DefaultTranscript::<E>::new(&[]),
    ));
}
