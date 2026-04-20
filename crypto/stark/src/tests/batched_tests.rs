use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use math::field::element::FieldElement;
use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField;
use math::field::goldilocks::GoldilocksField;

use crate::config::BatchedMerkleTreeBackend;
use crate::examples::multi_table_lookup::{
    new_add_air_with_lookup, new_cpu_air_with_lookup, new_mul_air_with_lookup,
};
use crate::proof::options::ProofOptions;
use crate::prover::{IsStarkProver, Prover};
use crate::trace::TraceTable;
use crate::traits::AIR;
use crate::verifier::{IsStarkVerifier, Verifier};

type Felt = FieldElement<GoldilocksField>;

#[test]
fn test_batched_main_trace_commit_and_open() {
    // Create two "tables" with different column counts
    let lde_size = 8;
    let table0_cols: Vec<Vec<Felt>> = (0..3)
        .map(|c| (0..lde_size).map(|r| Felt::from((c * 100 + r) as u64)).collect())
        .collect();
    let table1_cols: Vec<Vec<Felt>> = (0..2)
        .map(|c| (0..lde_size).map(|r| Felt::from((c * 200 + r) as u64)).collect())
        .collect();

    let per_table = vec![table0_cols.clone(), table1_cols.clone()];
    let (tree, root, layout) =
        Prover::<GoldilocksField, GoldilocksField, ()>::commit_main_traces_batched(
            &per_table, lde_size,
        )
        .expect("Commit should succeed");

    // Layout should track 3 + 2 = 5 total columns
    assert_eq!(layout.total_columns, 5);
    assert_eq!(layout.table_ranges, vec![(0, 3), (3, 5)]);

    // Open at index 0 and verify the Merkle proof
    let proof = tree.get_proof_by_pos(0).expect("Should get proof at index 0");
    // The leaf at position 0 was built from column values at bit-reversed row 0
    let br_0 = math::fft::cpu::bit_reversing::reverse_index(0, lde_size as u64);
    let opened_row: Vec<Felt> = (0..5)
        .map(|c| {
            if c < 3 {
                table0_cols[c][br_0].clone()
            } else {
                table1_cols[c - 3][br_0].clone()
            }
        })
        .collect();
    assert!(proof.verify::<BatchedMerkleTreeBackend<GoldilocksField>>(&root, 0, &opened_row));
}

#[test]
fn test_batched_composition_commit_n_point() {
    use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField;
    type FeltExt = FieldElement<Degree3GoldilocksExtensionField>;

    // Create fake composition parts for 2 tables (each has 2 parts of N=8 evals)
    let n = 8usize;
    let table0_parts: Vec<Vec<FeltExt>> = (0..2)
        .map(|p| {
            (0..n)
                .map(|i| FeltExt::from((p * 100 + i) as u64))
                .collect()
        })
        .collect();
    let table1_parts: Vec<Vec<FeltExt>> = (0..2)
        .map(|p| {
            (0..n)
                .map(|i| FeltExt::from((p * 200 + i) as u64))
                .collect()
        })
        .collect();

    let per_table = vec![table0_parts.clone(), table1_parts.clone()];
    let (tree, root, layout) =
        Prover::<GoldilocksField, Degree3GoldilocksExtensionField, ()>::commit_composition_polys_batched(
            &per_table, n,
        )
        .expect("Commit should succeed");

    // Layout: 2 + 2 = 4 total columns, domain_size = N (not 2N)
    assert_eq!(layout.total_columns, 4);
    assert_eq!(layout.domain_size, n);
    assert_eq!(layout.table_ranges, vec![(0, 2), (2, 4)]);

    // Open at index 0 and verify the Merkle proof
    let proof = tree
        .get_proof_by_pos(0)
        .expect("Should get proof at index 0");
    let br_0 = math::fft::cpu::bit_reversing::reverse_index(0, n as u64);
    let opened_row: Vec<FeltExt> = vec![
        table0_parts[0][br_0].clone(),
        table0_parts[1][br_0].clone(),
        table1_parts[0][br_0].clone(),
        table1_parts[1][br_0].clone(),
    ];
    assert!(proof.verify::<BatchedMerkleTreeBackend<Degree3GoldilocksExtensionField>>(
        &root, 0, &opened_row
    ));
}

type F = GoldilocksField;
type E = Degree3GoldilocksExtensionField;
type FE = FieldElement<F>;

/// Full prove/verify roundtrip test using batched proving with shared trees and FRI.
///
/// Uses the multi-table lookup example (CPU, ADD, MUL tables linked via LogUp bus).
/// All tables have the same trace length (uniform sizing).
#[test_log::test]
fn test_multi_prove_verify_batched_roundtrip() {
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

    // All tables must have same trace length for batched proving (uniform sizing)
    let trace_len = 8;

    let mut cpu_trace = TraceTable::from_columns_main(
        vec![add_column, mul_column, a_column, b_column, c_column],
        1,
    );

    // ADD Trace (8 rows): receives addition operations, padded to match CPU trace length
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

    // MUL Trace (8 rows): receives multiplication operations, padded to match CPU trace length
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

    assert_eq!(cpu_trace.num_rows(), trace_len);
    assert_eq!(add_trace.num_rows(), trace_len);
    assert_eq!(mul_trace.num_rows(), trace_len);

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

    let batched_proof = Prover::multi_prove_batched(
        air_trace_pairs,
        &mut DefaultTranscript::<E>::new(&[]),
    )
    .expect("Batched proving should succeed");

    // Verify
    let airs: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>> =
        vec![&cpu_air, &add_air, &mul_air];

    assert!(Verifier::multi_verify_batched(
        &airs,
        &batched_proof,
        &mut DefaultTranscript::<E>::new(&[]),
        &FieldElement::zero(),
    ));
}

/// All padding rows should also verify correctly with batched proving.
#[test_log::test]
fn test_batched_all_padding() {
    let trace_len = 8;

    let mut cpu_trace = TraceTable::from_columns_main(
        vec![
            vec![FE::zero(); trace_len],
            vec![FE::zero(); trace_len],
            vec![FE::zero(); trace_len],
            vec![FE::zero(); trace_len],
            vec![FE::zero(); trace_len],
        ],
        1,
    );

    let mut add_trace = TraceTable::from_columns_main(
        vec![
            vec![FE::zero(); trace_len],
            vec![FE::zero(); trace_len],
            vec![FE::zero(); trace_len],
            vec![FE::zero(); trace_len],
        ],
        1,
    );

    let mut mul_trace = TraceTable::from_columns_main(
        vec![
            vec![FE::zero(); trace_len],
            vec![FE::zero(); trace_len],
            vec![FE::zero(); trace_len],
            vec![FE::zero(); trace_len],
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

    let batched_proof = Prover::multi_prove_batched(
        air_trace_pairs,
        &mut DefaultTranscript::<E>::new(&[]),
    )
    .expect("Batched proving should succeed");

    let airs: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>> =
        vec![&cpu_air, &add_air, &mul_air];

    assert!(Verifier::multi_verify_batched(
        &airs,
        &batched_proof,
        &mut DefaultTranscript::<E>::new(&[]),
        &FieldElement::zero(),
    ));
}
