use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use math::field::element::FieldElement;

use math::field::fields::fft_friendly::{
    babybear::Babybear31PrimeField, quartic_babybear::Degree4BabyBearExtensionField,
};

use crate::examples::multi_table_lookup::new_add_air_with_lookup;
use crate::examples::multi_table_lookup::new_cpu_air_with_lookup;
use crate::examples::multi_table_lookup::new_mul_air_with_lookup;
use crate::trace::TraceTable;
use crate::traits::AIR;
use crate::{
    proof::options::ProofOptions,
    prover::{IsStarkProver, Prover},
    verifier::{IsStarkVerifier, Verifier},
};

type FE = FieldElement<Babybear31PrimeField>;
type ExtFE = FieldElement<Degree4BabyBearExtensionField>;

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
    let aux_columns = vec![
        vec![ExtFE::zero(); 8],
        vec![ExtFE::zero(); 8],
        vec![ExtFE::zero(); 8],
    ];
    let mut cpu_trace = TraceTable::from_columns(main_columns, aux_columns, 1);

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
    let mut add_trace = TraceTable::from_columns(
        vec![a_column, b_column, c_column, m_column],
        vec![vec![ExtFE::zero(); 4]],
        1,
    );

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
    let mut mul_trace = TraceTable::from_columns(
        vec![a_column, b_column, c_column, m_column],
        vec![vec![ExtFE::zero(); 4]],
        1,
    );

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
    let aux_columns = vec![
        vec![ExtFE::zero(); 8],
        vec![ExtFE::zero(); 8],
        vec![ExtFE::zero(); 8],
    ];
    let mut cpu_trace = TraceTable::from_columns(main_columns, aux_columns, 1);

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
    let mut add_trace = TraceTable::from_columns(
        vec![a_column, b_column, c_column, m_column],
        vec![vec![ExtFE::zero(); 4]],
        1,
    );

    // MUL Trace - correct values
    let a_column = vec![FE::from(2), FE::from(4), FE::from(7), FE::from(8)];
    let b_column = vec![FE::from(20), FE::from(40), FE::from(70), FE::from(80)];
    let c_column = vec![FE::from(40), FE::from(160), FE::from(490), FE::from(640)];
    let m_column = vec![FE::one(), FE::one(), FE::one(), FE::one()];
    let mut mul_trace = TraceTable::from_columns(
        vec![a_column, b_column, c_column, m_column],
        vec![vec![ExtFE::zero(); 4]],
        1,
    );

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
    let aux_columns = vec![
        vec![ExtFE::zero(); 8],
        vec![ExtFE::zero(); 8],
        vec![ExtFE::zero(); 8],
    ];
    let mut cpu_trace = TraceTable::from_columns(main_columns, aux_columns, 1);

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
    let mut add_trace = TraceTable::from_columns(
        vec![a_column, b_column, c_column, m_column],
        vec![vec![ExtFE::zero(); 4]],
        1,
    );

    // MUL Trace - correct
    let a_column = vec![FE::from(2), FE::from(4), FE::from(7), FE::from(8)];
    let b_column = vec![FE::from(20), FE::from(40), FE::from(70), FE::from(80)];
    let c_column = vec![FE::from(40), FE::from(160), FE::from(490), FE::from(640)];
    let m_column = vec![FE::one(), FE::one(), FE::one(), FE::one()];
    let mut mul_trace = TraceTable::from_columns(
        vec![a_column, b_column, c_column, m_column],
        vec![vec![ExtFE::zero(); 4]],
        1,
    );

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
