//! Single source of truth for the main-trace MMCS leaf hash format.
//!
//! Both the prover (when computing per-row leaves before MMCS build) and
//! the verifier (when re-hashing a per-row opening to compare against
//! `MmcsOpening::matrix_leaves`) must produce byte-identical digests for
//! the same `(MatrixTag, row_bytes)` pair. Centralising the format here
//! removes the risk of prover/verifier divergence.
//!
//! Leaf bytes layout:
//!
//! ```text
//! Keccak256( LEAF_DOMAIN_TAG || tag.0 (8 bytes) || row_bytes_be )
//! ```
//!
//! where `row_bytes_be` is every committed column's element written
//! big-endian, in column order. For preprocessed tables the precomputed
//! slice is NOT included here (those columns live in a separate
//! per-table Merkle tree).
//!
//! Bump `LEAF_DOMAIN_TAG` on any wire-incompatible change.

use crypto::merkle_tree::mmcs::MatrixTag;
use math::field::element::FieldElement;
use math::field::traits::IsField;
use math::traits::ByteConversion;
use sha3::{Digest, Keccak256};

use crate::config::Commitment;

/// Versioned domain separator for main-trace MMCS leaves. Bump suffix on
/// any encoding change so old proofs cannot be silently re-interpreted.
pub const LEAF_DOMAIN_TAG: &[u8] = b"LAMBDAVM_MAIN_MMCS_LEAF_V1";

/// Hash one row's worth of column bytes into a leaf digest using the
/// canonical tagged format. `row_bytes_be` is the concatenation of every
/// committed column's element written big-endian, in column order.
#[inline]
pub fn hash_tagged_row_bytes(tag: MatrixTag, row_bytes_be: &[u8]) -> Commitment {
    let mut h = Keccak256::new();
    h.update(LEAF_DOMAIN_TAG);
    h.update(tag.0);
    h.update(row_bytes_be);
    h.finalize().into()
}

/// Convenience: hash a row from individual field elements. Allocates a
/// stack-or-heap buffer for the row, suitable for verifier-side per-query
/// re-hashing (where allocation cost is dominated by FRI work anyway).
pub fn hash_tagged_row<E>(tag: MatrixTag, row: &[FieldElement<E>]) -> Commitment
where
    E: IsField,
    FieldElement<E>: ByteConversion,
{
    let byte_len = <FieldElement<E> as ByteConversion>::BYTE_LEN;
    let mut buf = vec![0u8; row.len() * byte_len];
    for (col_idx, fe) in row.iter().enumerate() {
        fe.write_bytes_be(&mut buf[col_idx * byte_len..(col_idx + 1) * byte_len]);
    }
    hash_tagged_row_bytes(tag, &buf)
}

#[cfg(test)]
mod tests {
    use super::*;
    use math::field::goldilocks::GoldilocksField;

    type FE = FieldElement<GoldilocksField>;

    #[test]
    fn tag_changes_digest() {
        let row = vec![FE::from(1u64), FE::from(2u64), FE::from(3u64)];
        let a = hash_tagged_row(MatrixTag::new([0; 8]), &row);
        let b = hash_tagged_row(MatrixTag::new([1, 0, 0, 0, 0, 0, 0, 0]), &row);
        assert_ne!(a, b);
    }

    #[test]
    fn row_change_changes_digest() {
        let tag = MatrixTag::new([7; 8]);
        let row_a = vec![FE::from(1u64), FE::from(2u64)];
        let row_b = vec![FE::from(1u64), FE::from(3u64)];
        assert_ne!(hash_tagged_row(tag, &row_a), hash_tagged_row(tag, &row_b));
    }
}
