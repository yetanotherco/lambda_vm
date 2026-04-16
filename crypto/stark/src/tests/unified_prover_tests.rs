//! Tests for `multi_prove_unified` — batched commitment + single FRI.
//!
//! Uses the same synthetic multi-table AIRs as bus_tests/completeness_tests
//! (CPU dispatching to ADD and MUL tables via LogUp bus).

use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use math::field::element::FieldElement;
use math::field::{
    extensions_goldilocks::Degree3GoldilocksExtensionField, goldilocks::GoldilocksField,
};

use crate::examples::multi_table_lookup::{
    new_add_air_with_lookup, new_cpu_air_with_lookup, new_mul_air_with_lookup,
};
use crate::proof::options::ProofOptions;
use crate::prover::{IsStarkProver, Prover};
use crate::trace::TraceTable;
use crate::traits::AIR;

type F = GoldilocksField;
type E = Degree3GoldilocksExtensionField;
type FE = FieldElement<F>;

/// Test multi_prove_unified with the standard CPU+ADD+MUL synthetic tables.
///
/// All three tables have the same trace length (8 rows padded to 8) and are
/// non-preprocessed, making them valid candidates for unified proving.
#[test_log::test]
fn test_unified_prove_basic() {
    // CPU Trace (8 rows)
    let mut cpu_trace = TraceTable::from_columns_main(
        vec![
            vec![
                FE::one(),
                FE::zero(),
                FE::one(),
                FE::zero(),
                FE::one(),
                FE::one(),
                FE::zero(),
                FE::zero(),
            ],
            vec![
                FE::zero(),
                FE::one(),
                FE::zero(),
                FE::one(),
                FE::zero(),
                FE::zero(),
                FE::one(),
                FE::one(),
            ],
            vec![
                FE::from(1),
                FE::from(2),
                FE::from(3),
                FE::from(4),
                FE::from(5),
                FE::from(6),
                FE::from(7),
                FE::from(8),
            ],
            vec![
                FE::from(10),
                FE::from(20),
                FE::from(30),
                FE::from(40),
                FE::from(50),
                FE::from(60),
                FE::from(70),
                FE::from(80),
            ],
            vec![
                FE::from(11),
                FE::from(40),
                FE::from(33),
                FE::from(160),
                FE::from(55),
                FE::from(66),
                FE::from(490),
                FE::from(640),
            ],
        ],
        1,
    );

    // ADD Trace (4 rows, padded to 8 to match CPU trace length)
    let mut add_trace = TraceTable::from_columns_main(
        vec![
            vec![
                FE::from(1),
                FE::from(3),
                FE::from(5),
                FE::from(6),
                FE::zero(),
                FE::zero(),
                FE::zero(),
                FE::zero(),
            ],
            vec![
                FE::from(10),
                FE::from(30),
                FE::from(50),
                FE::from(60),
                FE::zero(),
                FE::zero(),
                FE::zero(),
                FE::zero(),
            ],
            vec![
                FE::from(11),
                FE::from(33),
                FE::from(55),
                FE::from(66),
                FE::zero(),
                FE::zero(),
                FE::zero(),
                FE::zero(),
            ],
            vec![
                FE::one(),
                FE::one(),
                FE::one(),
                FE::one(),
                FE::zero(),
                FE::zero(),
                FE::zero(),
                FE::zero(),
            ],
        ],
        1,
    );

    // MUL Trace (4 rows, padded to 8)
    let mut mul_trace = TraceTable::from_columns_main(
        vec![
            vec![
                FE::from(2),
                FE::from(4),
                FE::from(7),
                FE::from(8),
                FE::zero(),
                FE::zero(),
                FE::zero(),
                FE::zero(),
            ],
            vec![
                FE::from(20),
                FE::from(40),
                FE::from(70),
                FE::from(80),
                FE::zero(),
                FE::zero(),
                FE::zero(),
                FE::zero(),
            ],
            vec![
                FE::from(40),
                FE::from(160),
                FE::from(490),
                FE::from(640),
                FE::zero(),
                FE::zero(),
                FE::zero(),
                FE::zero(),
            ],
            vec![
                FE::one(),
                FE::one(),
                FE::one(),
                FE::one(),
                FE::zero(),
                FE::zero(),
                FE::zero(),
                FE::zero(),
            ],
        ],
        1,
    );

    let proof_options = ProofOptions::default_test_options();
    let cpu_air = new_cpu_air_with_lookup(&proof_options);
    let add_air = new_add_air_with_lookup(&proof_options);
    let mul_air = new_mul_air_with_lookup(&proof_options);

    // Verify all tables have same trace length and are non-preprocessed
    assert_eq!(cpu_trace.num_rows(), add_trace.num_rows());
    assert_eq!(cpu_trace.num_rows(), mul_trace.num_rows());
    assert!(!cpu_air.is_preprocessed());
    assert!(!add_air.is_preprocessed());
    assert!(!mul_air.is_preprocessed());

    let air_trace_pairs: Vec<(
        &dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>,
        _,
        _,
    )> = vec![
        (&cpu_air, &mut cpu_trace, &()),
        (&add_air, &mut add_trace, &()),
        (&mul_air, &mut mul_trace, &()),
    ];

    let unified_proof = Prover::multi_prove_unified(
        air_trace_pairs,
        &mut DefaultTranscript::<E>::new(&[]),
    )
    .expect("multi_prove_unified failed");

    // Structural assertions
    assert_eq!(unified_proof.table_data.len(), 3, "should have 3 table OOD entries");
    assert_eq!(unified_proof.column_layout.len(), 3, "should have 3 column ranges");
    assert!(unified_proof.aux_trace_root.is_some(), "all tables have aux traces");
    assert!(!unified_proof.fri_layers_merkle_roots.is_empty(), "FRI should produce layers");
    assert!(!unified_proof.query_openings.is_empty(), "should have query openings");
    assert!(unified_proof.precomputed_roots.is_empty(), "no preprocessed tables");

    // Check column layout is consistent
    let layout = &unified_proof.column_layout;
    // CPU: 5 main cols, ADD: 4 main cols, MUL: 4 main cols → total 13
    assert_eq!(layout[0].main_start, 0);
    assert_eq!(layout[0].main_count, 5);
    assert_eq!(layout[1].main_start, 5);
    assert_eq!(layout[1].main_count, 4);
    assert_eq!(layout[2].main_start, 9);
    assert_eq!(layout[2].main_count, 4);

    // Verify each table's OOD data has correct dimensions
    for (idx, td) in unified_proof.table_data.iter().enumerate() {
        assert_eq!(td.trace_length, 8, "table {idx} trace_length");
        assert!(!td.composition_poly_parts_ood_evaluation.is_empty());
        assert!(td.bus_public_inputs.is_some(), "table {idx} should have bus inputs");
    }

    println!(
        "Unified proof OK: {} tables, {} FRI layers, {} queries, {} query openings",
        unified_proof.table_data.len(),
        unified_proof.fri_layers_merkle_roots.len(),
        unified_proof.fri_query_list.len(),
        unified_proof.query_openings.len(),
    );
}

/// All-padding variant: every multiplicity is 0.
#[test_log::test]
fn test_unified_prove_all_padding() {
    let mut cpu_trace = TraceTable::from_columns_main(
        vec![vec![FE::zero(); 4]; 5],
        1,
    );
    let mut add_trace = TraceTable::from_columns_main(
        vec![vec![FE::zero(); 4]; 4],
        1,
    );
    let mut mul_trace = TraceTable::from_columns_main(
        vec![vec![FE::zero(); 4]; 4],
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

    let unified_proof = Prover::multi_prove_unified(
        air_trace_pairs,
        &mut DefaultTranscript::<E>::new(&[]),
    )
    .expect("multi_prove_unified with all-padding should succeed");

    assert_eq!(unified_proof.table_data.len(), 3);
}
