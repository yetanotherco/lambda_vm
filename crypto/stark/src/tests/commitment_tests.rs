//! Unit tests for the Merkle commitment layer (`crate::commitment`): they pin
//! the bit-reversed, row-grouped leaf byte layout that the GPU kernels and the
//! verifier's `verify_opening_pair` must match. Previously this layout was only
//! covered transitively through full prove→verify.

use crate::commitment::{
    ROWS_PER_LEAF, commit_bit_reversed, keccak_leaves_bit_reversed,
    keccak_leaves_bit_reversed_grouped, keccak_leaves_row_pair_bit_reversed,
};
use crate::config::{BatchedMerkleTree, BatchedMerkleTreeBackend, Commitment};
use math::fft::bit_reversing::reverse_index;
use math::field::{element::FieldElement, goldilocks::GoldilocksField};
use math::traits::ByteConversion;

type F = GoldilocksField;
type Felt = FieldElement<F>;

/// 3 columns × 8 rows of distinct, nonzero values.
fn sample_columns() -> Vec<Vec<Felt>> {
    (0..3u64)
        .map(|c| (0..8u64).map(|r| Felt::from(100 * c + r + 1)).collect())
        .collect()
}

/// Independent reference for one leaf, written straight from the module-doc
/// layout (`rows_per_leaf` consecutive bit-reversed rows, column-major within
/// each row, big-endian), hashed once with the same backend the prover uses.
/// Structurally separate from the production `map_init` loop, so a transposed
/// row/column order or a wrong bit-reversal in production fails this check.
fn expected_leaf(columns: &[Vec<Felt>], rows_per_leaf: usize, leaf_idx: usize) -> Commitment {
    let num_rows = columns[0].len();
    let byte_len = <Felt as ByteConversion>::BYTE_LEN;
    let mut buf = vec![0u8; rows_per_leaf * columns.len() * byte_len];
    let mut offset = 0;
    for k in 0..rows_per_leaf {
        let br = reverse_index(rows_per_leaf * leaf_idx + k, num_rows as u64);
        for col in columns {
            col[br].write_bytes_be(&mut buf[offset..offset + byte_len]);
            offset += byte_len;
        }
    }
    BatchedMerkleTreeBackend::<F>::hash_bytes(&buf)
}

#[test]
fn grouped_leaves_match_documented_layout_for_r1_and_r2() {
    let columns = sample_columns();
    let num_rows = columns[0].len();
    for &rows_per_leaf in &[1usize, 2usize] {
        let leaves = keccak_leaves_bit_reversed_grouped(&columns, rows_per_leaf);
        assert_eq!(
            leaves.len(),
            num_rows / rows_per_leaf,
            "leaf count for rows_per_leaf={rows_per_leaf}"
        );
        for (i, leaf) in leaves.iter().enumerate() {
            assert_eq!(
                *leaf,
                expected_leaf(&columns, rows_per_leaf, i),
                "leaf {i} for rows_per_leaf={rows_per_leaf}"
            );
        }
    }
}

#[test]
fn wrappers_agree_with_grouped() {
    let columns = sample_columns();
    assert_eq!(
        keccak_leaves_bit_reversed(&columns),
        keccak_leaves_bit_reversed_grouped(&columns, 1)
    );
    assert_eq!(
        keccak_leaves_row_pair_bit_reversed(&columns),
        keccak_leaves_bit_reversed_grouped(&columns, ROWS_PER_LEAF)
    );
}

#[test]
fn commit_root_matches_tree_built_over_leaves() {
    let columns = sample_columns();
    let leaves = keccak_leaves_bit_reversed_grouped(&columns, ROWS_PER_LEAF);
    let tree = BatchedMerkleTree::<F>::build_from_hashed_leaves(leaves).unwrap();
    let (_, root) = commit_bit_reversed(&columns, ROWS_PER_LEAF).unwrap();
    assert_eq!(root, tree.root);
}

#[test]
fn empty_and_zero_row_inputs_short_circuit() {
    let empty: Vec<Vec<Felt>> = vec![];
    assert!(keccak_leaves_bit_reversed_grouped(&empty, ROWS_PER_LEAF).is_empty());
    assert!(commit_bit_reversed(&empty, ROWS_PER_LEAF).is_none());
    let zero_rows: Vec<Vec<Felt>> = vec![vec![]];
    assert!(keccak_leaves_bit_reversed_grouped(&zero_rows, ROWS_PER_LEAF).is_empty());
    assert!(commit_bit_reversed(&zero_rows, ROWS_PER_LEAF).is_none());
}

/// ★ The [`StarkHash`] two-element-leaf invariant, for the keccak configuration.
///
/// The prover commits FRI layers with `Pair` and the verifier authenticates
/// those openings with `Batched` (`verify_fri_layer_openings` builds a
/// two-element `Vec`). Nothing in the type system makes those agree — this is
/// what says they do, so a second configuration that breaks it fails here
/// rather than by rejecting every honest proof at its first FRI query.
#[test]
fn batched_and_pair_agree_on_a_two_element_leaf() {
    use crate::config::{KeccakStarkHash, StarkHash};
    use crypto::merkle_tree::traits::IsMerkleTreeBackend;

    type Batched = <KeccakStarkHash as StarkHash>::Batched<F>;
    type Pair = <KeccakStarkHash as StarkHash>::Pair<F>;

    for (a, b) in [(0u64, 1u64), (7, 7), (u64::MAX - 1, 12345)] {
        let (x, y) = (Felt::from(a), Felt::from(b));
        assert_eq!(
            <Batched as IsMerkleTreeBackend>::hash_data(&vec![x, y]),
            <Pair as IsMerkleTreeBackend>::hash_data(&[x, y]),
            "Batched and Pair must hash the pair ({a}, {b}) identically"
        );
    }
}

/// The streaming routes and the owned-`Data` route are the same leaf.
#[test]
fn streaming_leaf_routes_match_hash_data() {
    use crate::config::{KeccakStarkHash, StarkHash};
    use crypto::merkle_tree::traits::{IsMerkleTreeBackend, IsStreamingLeafBackend};

    type Batched = <KeccakStarkHash as StarkHash>::Batched<F>;

    let row: Vec<Felt> = (0..5u64).map(Felt::from).collect();
    let (left, right) = row.split_at(2);

    let owned = <Batched as IsMerkleTreeBackend>::hash_data(&row);
    assert_eq!(
        owned,
        <Batched as IsStreamingLeafBackend<F>>::hash_data_from_slices(left, right),
        "hash_data_from_slices must equal hash_data on the concatenation"
    );

    let mut buf = Vec::new();
    for e in &row {
        let mut b = [0u8; 8];
        e.write_bytes_be(&mut b);
        buf.extend_from_slice(&b);
    }
    assert_eq!(
        owned,
        <Batched as IsStreamingLeafBackend<F>>::hash_bytes(&buf),
        "hash_bytes must equal hash_data on the elements those bytes encode"
    );
}
