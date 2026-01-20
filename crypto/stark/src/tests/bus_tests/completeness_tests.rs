//! Completeness tests: valid proofs are accepted.
//!
//! These tests verify that the prover and verifier work correctly for legitimate use cases.

use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use math::field::element::FieldElement;
use math::field::fields::fft_friendly::{
    babybear::Babybear31PrimeField, quartic_babybear::Degree4BabyBearExtensionField,
};

use crate::examples::multi_table_lookup::{
    generate_random_traces, new_add_air_with_lookup, new_cpu_air_with_lookup,
    new_mul_air_with_lookup,
};
use crate::proof::options::ProofOptions;
use crate::prover::{IsStarkProver, Prover};
use crate::trace::TraceTable;
use crate::traits::AIR;
use crate::verifier::{IsStarkVerifier, Verifier};

type F = Babybear31PrimeField;
type E = Degree4BabyBearExtensionField;
type FE = FieldElement<F>;

/// Standard valid multi-table proof with CPU, ADD, and MUL tables.
#[test_log::test]
fn test_multi_table_proof() {
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

/// All padding rows (multiplicity = 0 everywhere). Bus should balance at zero.
#[test_log::test]
fn test_all_padding() {
    let mut cpu_trace = TraceTable::from_columns_main(
        vec![
            vec![FE::zero(); 4], // add_flag = 0
            vec![FE::zero(); 4], // mul_flag = 0
            vec![FE::zero(); 4],
            vec![FE::zero(); 4],
            vec![FE::zero(); 4],
        ],
        1,
    );

    let mut add_trace = TraceTable::from_columns_main(
        vec![
            vec![FE::zero(); 4],
            vec![FE::zero(); 4],
            vec![FE::zero(); 4],
            vec![FE::zero(); 4], // multiplicity = 0
        ],
        1,
    );

    let mut mul_trace = TraceTable::from_columns_main(
        vec![
            vec![FE::zero(); 4],
            vec![FE::zero(); 4],
            vec![FE::zero(); 4],
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

/// Single operation (minimal non-trivial case).
#[test_log::test]
fn test_single_operation() {
    let mut cpu_trace = TraceTable::from_columns_main(
        vec![
            vec![FE::one(), FE::zero(), FE::zero(), FE::zero()], // add_flag
            vec![FE::zero(); 4],                                 // mul_flag
            vec![FE::from(5), FE::zero(), FE::zero(), FE::zero()],
            vec![FE::from(3), FE::zero(), FE::zero(), FE::zero()],
            vec![FE::from(8), FE::zero(), FE::zero(), FE::zero()], // 5 + 3
        ],
        1,
    );

    let mut add_trace = TraceTable::from_columns_main(
        vec![
            vec![FE::from(5), FE::zero(), FE::zero(), FE::zero()],
            vec![FE::from(3), FE::zero(), FE::zero(), FE::zero()],
            vec![FE::from(8), FE::zero(), FE::zero(), FE::zero()],
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

    let airs: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>> =
        vec![&cpu_air, &add_air, &mul_air];

    assert!(Verifier::multi_verify(
        &airs,
        &multi_proof,
        &mut DefaultTranscript::<E>::new(&[]),
    ));
}

/// Duplicate operations: same (a,b,c) sent twice, received with multiplicity=2.
#[test_log::test]
fn test_duplicate_operations() {
    let mut cpu_trace = TraceTable::from_columns_main(
        vec![
            vec![FE::one(), FE::one(), FE::zero(), FE::zero()], // sends twice
            vec![FE::zero(); 4],
            vec![FE::from(5), FE::from(5), FE::zero(), FE::zero()],
            vec![FE::from(3), FE::from(3), FE::zero(), FE::zero()],
            vec![FE::from(8), FE::from(8), FE::zero(), FE::zero()],
        ],
        1,
    );

    let mut add_trace = TraceTable::from_columns_main(
        vec![
            vec![FE::from(5), FE::zero(), FE::zero(), FE::zero()],
            vec![FE::from(3), FE::zero(), FE::zero(), FE::zero()],
            vec![FE::from(8), FE::zero(), FE::zero(), FE::zero()],
            vec![FE::from(2), FE::zero(), FE::zero(), FE::zero()], // multiplicity = 2
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

    let airs: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>> =
        vec![&cpu_air, &add_air, &mul_air];

    assert!(Verifier::multi_verify(
        &airs,
        &multi_proof,
        &mut DefaultTranscript::<E>::new(&[]),
    ));
}

/// Proof serialization round-trip.
#[test_log::test]
fn test_serialization_roundtrip() {
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

    assert!(Verifier::multi_verify(
        &airs,
        &deserialized,
        &mut DefaultTranscript::<E>::new(&[]),
    ));
}

/// Verification of a random CPU trace table of 16 rows and its respective ADD and MUL tables.
#[test_log::test]
fn test_random_traces_proof_verification_short_table() {
    // Generate random traces.
    // CPU number of rows: 16
    let (mut cpu_trace, mut add_trace, mut mul_trace) = generate_random_traces(16, None);

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

    let multi_proof = Prover::multi_prove(air_trace_pairs, &mut DefaultTranscript::<E>::new(&[]))
        .expect("Failed to generate proof");

    let airs: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>> =
        vec![&cpu_air, &add_air, &mul_air];

    assert!(
        Verifier::multi_verify(&airs, &multi_proof, &mut DefaultTranscript::<E>::new(&[]),),
        "Proof verification failed for random traces"
    );
}

/// Verification of a random CPU trace table of 2^16 rows and its respective ADD and MUL tables.
#[test_log::test]
fn test_random_traces_proof_verification_long_table() {
    // Generate random traces.
    // CPU number of rows: 2^16
    let (mut cpu_trace, mut add_trace, mut mul_trace) = generate_random_traces(65536, None);

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

    let multi_proof = Prover::multi_prove(air_trace_pairs, &mut DefaultTranscript::<E>::new(&[]))
        .expect("Failed to generate proof");

    let airs: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>> =
        vec![&cpu_air, &add_air, &mul_air];

    assert!(
        Verifier::multi_verify(&airs, &multi_proof, &mut DefaultTranscript::<E>::new(&[]),),
        "Proof verification failed for random traces"
    );
}
