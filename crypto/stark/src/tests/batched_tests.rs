use math::field::element::FieldElement;
use math::field::goldilocks::GoldilocksField;

use crate::config::BatchedMerkleTreeBackend;
use crate::prover::{IsStarkProver, Prover};

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
