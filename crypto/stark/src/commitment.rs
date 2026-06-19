//! Merkle-tree commitment to bit-reversed, column-major LDE evaluations.
//!
//! This is the commitment layer the prover uses for the main/aux trace LDEs and
//! the composition-polynomial parts. It is decoupled from `IsStarkProver`: the
//! prover only orchestrates *when* to commit; the *how* (leaf layout, bit-reverse
//! permutation, Keccak hashing, tree build) lives here.
//!
//! ## Leaf layout
//!
//! For each leaf `i` we hash `rows_per_leaf` consecutive (bit-reversed) rows,
//! big-endian-concatenated column-by-column:
//!
//! ```text
//! leaf(i) = keccak( col_0[br(R·i)]‖col_1[br(R·i)]‖…  ‖  col_0[br(R·i+1)]‖…  ‖ … )
//!   where R = rows_per_leaf and br(j) = reverse_index(j, num_rows)
//! ```
//!
//! - `rows_per_leaf == 1`: one row per leaf — main/aux trace LDE columns.
//! - `rows_per_leaf == 2`: a row pair per leaf — composition-polynomial parts
//!   (Round 2/3), where leaf `i` hashes rows `2i` and `2i+1`.
//!
//! The field-element serialization (`write_bytes_be`) + `hash_bytes` path is kept
//! exactly as before, so commitments are byte-identical to the previous
//! per-row / per-row-pair implementations.

use math::fft::bit_reversing::reverse_index;
use math::field::element::FieldElement;
use math::field::traits::IsField;
use math::traits::{AsBytes, ByteConversion};

#[cfg(feature = "parallel")]
use rayon::prelude::{IntoParallelIterator, ParallelIterator};

use crate::config::{BatchedMerkleTree, BatchedMerkleTreeBackend, Commitment};

/// Computes the Keccak-256 leaf hashes for a bit-reversed, column-major commitment,
/// grouping `rows_per_leaf` consecutive bit-reversed rows into each leaf.
///
/// Returns one `Commitment` per leaf (`columns[0].len() / rows_per_leaf` leaves),
/// or an empty `Vec` when there is nothing to hash. See the module docs for the
/// exact leaf byte layout. This is the single code path behind both the per-row
/// ([`keccak_leaves_bit_reversed`]) and per-row-pair
/// ([`keccak_leaves_row_pair_bit_reversed`]) commitments.
pub fn keccak_leaves_bit_reversed_grouped<E>(
    columns: &[Vec<FieldElement<E>>],
    rows_per_leaf: usize,
) -> Vec<Commitment>
where
    E: IsField,
    FieldElement<E>: AsBytes + Sync + Send + ByteConversion,
{
    if columns.is_empty() || columns[0].is_empty() {
        return Vec::new();
    }

    let num_rows = columns[0].len();
    let byte_len = <FieldElement<E> as ByteConversion>::BYTE_LEN;

    debug_assert!(
        num_rows.is_power_of_two(),
        "num_rows must be a power of two for reverse_index"
    );
    debug_assert!(
        rows_per_leaf >= 1 && num_rows % rows_per_leaf == 0,
        "num_rows must be a multiple of rows_per_leaf"
    );

    let num_leaves = num_rows / rows_per_leaf;
    let total_bytes = rows_per_leaf * columns.len() * byte_len;

    // Leaf `i`: the `rows_per_leaf` bit-reversed rows starting at `R·i`, each row
    // written column-by-column in big-endian, then hashed once.
    let hash_leaf = |buf: &mut [u8], leaf_idx: usize| -> Commitment {
        let mut offset = 0;
        for k in 0..rows_per_leaf {
            let br = reverse_index(rows_per_leaf * leaf_idx + k, num_rows as u64);
            for col in columns {
                col[br].write_bytes_be(&mut buf[offset..offset + byte_len]);
                offset += byte_len;
            }
        }
        BatchedMerkleTreeBackend::<E>::hash_bytes(buf)
    };

    // Per-thread buffer reuse (map_init) avoids millions of small allocations.
    #[cfg(feature = "parallel")]
    let result: Vec<Commitment> = (0..num_leaves)
        .into_par_iter()
        .map_init(|| vec![0u8; total_bytes], |buf, i| hash_leaf(buf, i))
        .collect();

    #[cfg(not(feature = "parallel"))]
    let result: Vec<Commitment> = {
        let mut buf = vec![0u8; total_bytes];
        (0..num_leaves).map(|i| hash_leaf(&mut buf, i)).collect()
    };

    result
}

/// Per-row Keccak-256 leaf hashes (one leaf per bit-reversed row). Used for the
/// main/aux trace LDE commitments. Thin wrapper over
/// [`keccak_leaves_bit_reversed_grouped`] with `rows_per_leaf = 1`.
///
/// Kept as a named public function because the GPU parity tests in dependent
/// crates compare against this exact code path.
pub fn keccak_leaves_bit_reversed<E>(columns: &[Vec<FieldElement<E>>]) -> Vec<Commitment>
where
    E: IsField,
    FieldElement<E>: AsBytes + Sync + Send + ByteConversion,
{
    keccak_leaves_bit_reversed_grouped(columns, 1)
}

/// Per-row-pair Keccak-256 leaf hashes (leaf `i` hashes bit-reversed rows `2i`,
/// `2i+1`). Used for the composition-polynomial-parts commitment. Thin wrapper
/// over [`keccak_leaves_bit_reversed_grouped`] with `rows_per_leaf = 2`.
pub fn keccak_leaves_row_pair_bit_reversed<E>(parts: &[Vec<FieldElement<E>>]) -> Vec<Commitment>
where
    E: IsField,
    FieldElement<E>: AsBytes + Sync + Send + ByteConversion,
{
    keccak_leaves_bit_reversed_grouped(parts, 2)
}

/// Builds the Merkle tree committing to `columns`' bit-reversed, column-major LDE
/// evaluations, grouping `rows_per_leaf` rows per leaf, and returns the tree and
/// its root. `None` when there is nothing to commit.
///
/// Replaces the prover's former `commit_columns_bit_reversed` (`rows_per_leaf = 1`)
/// and `commit_composition_polynomial` (`rows_per_leaf = 2`).
pub fn commit_bit_reversed<E>(
    columns: &[Vec<FieldElement<E>>],
    rows_per_leaf: usize,
) -> Option<(BatchedMerkleTree<E>, Commitment)>
where
    E: IsField,
    FieldElement<E>: AsBytes + Sync + Send + ByteConversion,
{
    if columns.is_empty() || columns[0].is_empty() {
        return None;
    }
    let hashed_leaves = keccak_leaves_bit_reversed_grouped(columns, rows_per_leaf);
    let tree = BatchedMerkleTree::<E>::build_from_hashed_leaves(hashed_leaves)?;
    let root = tree.root;
    Some((tree, root))
}
