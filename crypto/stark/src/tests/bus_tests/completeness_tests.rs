//! Completeness tests: valid proofs are accepted.
//!
//! These tests verify that the prover and verifier work correctly for legitimate use cases.

use crate::constraints::builder::EmptyConstraints;
use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use math::field::element::FieldElement;
use math::field::{
    extensions_goldilocks::Degree3GoldilocksExtensionField, goldilocks::GoldilocksField,
};

use crate::examples::multi_table_lookup::{
    new_add_air_with_lookup, new_cpu_air_with_lookup, new_mul_air_with_lookup,
};
use crate::lookup::{
    AirWithBuses, AuxiliaryTraceBuildData, BusInteraction, Multiplicity,
    NullBoundaryConstraintBuilder, Packing,
};
use crate::proof::options::ProofOptions;
use crate::test_utils::multi_prove_ram;
use crate::trace::TraceTable;
use crate::traits::AIR;
use crate::verifier::{IsStarkVerifier, Verifier};

type F = GoldilocksField;
type E = Degree3GoldilocksExtensionField;
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
        multi_prove_ram(air_trace_pairs, &mut DefaultTranscript::<E>::new(&[])).unwrap();

    let airs: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>> =
        vec![&cpu_air, &add_air, &mul_air];

    assert!(Verifier::multi_verify(
        &airs,
        &multi_proof,
        &mut DefaultTranscript::<E>::new(&[]),
        &FieldElement::zero(),
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
        multi_prove_ram(air_trace_pairs, &mut DefaultTranscript::<E>::new(&[])).unwrap();

    let airs: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>> =
        vec![&cpu_air, &add_air, &mul_air];

    assert!(Verifier::multi_verify(
        &airs,
        &multi_proof,
        &mut DefaultTranscript::<E>::new(&[]),
        &FieldElement::zero(),
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
        multi_prove_ram(air_trace_pairs, &mut DefaultTranscript::<E>::new(&[])).unwrap();

    let airs: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>> =
        vec![&cpu_air, &add_air, &mul_air];

    assert!(Verifier::multi_verify(
        &airs,
        &multi_proof,
        &mut DefaultTranscript::<E>::new(&[]),
        &FieldElement::zero(),
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
        multi_prove_ram(air_trace_pairs, &mut DefaultTranscript::<E>::new(&[])).unwrap();

    let airs: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>> =
        vec![&cpu_air, &add_air, &mul_air];

    assert!(Verifier::multi_verify(
        &airs,
        &multi_proof,
        &mut DefaultTranscript::<E>::new(&[]),
        &FieldElement::zero(),
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
        multi_prove_ram(air_trace_pairs, &mut DefaultTranscript::<E>::new(&[])).unwrap();

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
        &FieldElement::zero(),
    ));
}

/// Integration test for BusValue features: Packed, constant, column, and Linear.
///
/// Tests that the full prover/verifier pipeline works with:
/// - `BusValue::Packed` (via `Packing::columns()`)
/// - `BusValue::constant()` for fixed values (e.g., table IDs)
/// - `BusValue::column()` for single column values
/// - `BusValue::Linear()` for custom linear combinations
///
/// Fingerprint structure: constant(0x42) + α·Word2L(h0,h1) + α²·col[2] + α³·(3·col[3] + 5)
#[test_log::test]
fn test_bus_value_features() {
    use crate::lookup::{BusValue, LinearTerm};

    // Sender: [mult, h0, h1, a, b] - 5 columns
    // Fingerprint: 0x42 + α·Word2L(h0,h1) + α²·a + α³·(3·b + 5)
    let sender_air = {
        let mut values = vec![BusValue::constant(0x42)]; // constant table ID
        values.extend(Packing::Word2L.columns(&[1])); // cols 1,2 packed
        values.push(BusValue::column(3)); // col 3 directly
        values.push(BusValue::Linear(vec![
            // custom: 3*col[4] + 5
            LinearTerm::Column {
                coefficient: 3,
                column: 4,
            },
            LinearTerm::Constant(5),
        ]));
        let build_data = AuxiliaryTraceBuildData {
            interactions: vec![BusInteraction::sender(
                0u64,
                Multiplicity::Column(0),
                values,
            )],
        };
        let proof_options = ProofOptions::default_test_options();
        AirWithBuses::<F, E, NullBoundaryConstraintBuilder, (), _>::new(
            5,
            build_data,
            &proof_options,
            1,
            EmptyConstraints,
        )
    };

    // Receiver: [h0, h1, a, b, mult] - same fingerprint, different column layout
    let receiver_air = {
        let mut values = vec![BusValue::constant(0x42)];
        values.extend(Packing::Word2L.columns(&[0])); // cols 0,1 packed
        values.push(BusValue::column(2)); // col 2 directly
        values.push(BusValue::Linear(vec![
            LinearTerm::Column {
                coefficient: 3,
                column: 3,
            },
            LinearTerm::Constant(5),
        ]));
        let build_data = AuxiliaryTraceBuildData {
            interactions: vec![BusInteraction::receiver(
                0u64,
                Multiplicity::Column(4),
                values,
            )],
        };
        let proof_options = ProofOptions::default_test_options();
        AirWithBuses::<F, E, NullBoundaryConstraintBuilder, (), _>::new(
            5,
            build_data,
            &proof_options,
            1,
            EmptyConstraints,
        )
    };

    // Sender trace: [mult, h0, h1, a, b]
    let mut sender_trace = TraceTable::from_columns_main(
        vec![
            vec![FE::one(), FE::one(), FE::zero(), FE::zero()], // mult
            vec![
                FE::from(0x1234u64),
                FE::from(0xABCDu64),
                FE::zero(),
                FE::zero(),
            ], // h0
            vec![
                FE::from(0x5678u64),
                FE::from(0xEF01u64),
                FE::zero(),
                FE::zero(),
            ], // h1
            vec![FE::from(100u64), FE::from(200u64), FE::zero(), FE::zero()], // a
            vec![FE::from(10u64), FE::from(20u64), FE::zero(), FE::zero()], // b
        ],
        1,
    );

    // Receiver trace: [h0, h1, a, b, mult]
    let mut receiver_trace = TraceTable::from_columns_main(
        vec![
            vec![
                FE::from(0x1234u64),
                FE::from(0xABCDu64),
                FE::zero(),
                FE::zero(),
            ], // h0
            vec![
                FE::from(0x5678u64),
                FE::from(0xEF01u64),
                FE::zero(),
                FE::zero(),
            ], // h1
            vec![FE::from(100u64), FE::from(200u64), FE::zero(), FE::zero()], // a
            vec![FE::from(10u64), FE::from(20u64), FE::zero(), FE::zero()],   // b
            vec![FE::one(), FE::one(), FE::zero(), FE::zero()],               // mult
        ],
        1,
    );

    let air_trace_pairs: Vec<(
        &dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>,
        _,
        _,
    )> = vec![
        (&sender_air, &mut sender_trace, &()),
        (&receiver_air, &mut receiver_trace, &()),
    ];

    let multi_proof =
        multi_prove_ram(air_trace_pairs, &mut DefaultTranscript::<E>::new(&[])).unwrap();

    let airs: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>> =
        vec![&sender_air, &receiver_air];

    assert!(Verifier::multi_verify(
        &airs,
        &multi_proof,
        &mut DefaultTranscript::<E>::new(&[]),
        &FieldElement::zero(),
    ));
}
