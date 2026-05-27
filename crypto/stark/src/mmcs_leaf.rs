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

/// Aliased name for `LEAF_DOMAIN_TAG`. Use this in new code to make the
/// intent explicit when an MMCS-specific tag is needed alongside the aux
/// tag below.
pub const LEAF_DOMAIN_TAG_MAIN: &[u8] = LEAF_DOMAIN_TAG;

/// Versioned domain separator for AUX-trace MMCS leaves. Distinct from
/// `LEAF_DOMAIN_TAG_MAIN` so that an aux leaf and a main leaf with the
/// same `(MatrixTag, row_bytes)` produce different digests — i.e. neither
/// MMCS opening can authenticate a leaf that was committed against the
/// other.
pub const LEAF_DOMAIN_TAG_AUX: &[u8] = b"LAMBDAVM_AUX_MMCS_LEAF_V1";

/// Synthesize `n` distinct [`MatrixTag`]s derived from positional index.
/// Useful for generic stark tests where the caller does not own a stable
/// chip-type assignment. Production code in lambda-vm uses
/// `VmAirs::air_tags()` instead, which encodes chip type + chunk index.
pub fn synth_main_tags(n: usize) -> Vec<MatrixTag> {
    (0..n)
        .map(|i| MatrixTag::new((i as u64).to_le_bytes()))
        .collect()
}

/// Convenience: synthesize `MatrixTag`s sized to a slice. Equivalent to
/// `synth_main_tags(slice.len())`.
pub fn synth_main_tags_for<T>(slice: &[T]) -> Vec<MatrixTag> {
    synth_main_tags(slice.len())
}

/// Hash one row's worth of column bytes into a MAIN-trace MMCS leaf digest.
/// `row_bytes_be` is the concatenation of every committed column's element
/// written big-endian, in column order.
#[inline]
pub fn hash_tagged_row_bytes(tag: MatrixTag, row_bytes_be: &[u8]) -> Commitment {
    hash_with_domain(LEAF_DOMAIN_TAG_MAIN, tag, row_bytes_be)
}

/// Hash one row's worth of column bytes into an AUX-trace MMCS leaf digest.
/// Uses [`LEAF_DOMAIN_TAG_AUX`] so the digest cannot collide with a
/// main-trace leaf for the same `(tag, row_bytes)`.
#[inline]
pub fn hash_tagged_row_bytes_aux(tag: MatrixTag, row_bytes_be: &[u8]) -> Commitment {
    hash_with_domain(LEAF_DOMAIN_TAG_AUX, tag, row_bytes_be)
}

#[inline]
fn hash_with_domain(domain: &[u8], tag: MatrixTag, row_bytes_be: &[u8]) -> Commitment {
    let mut h = Keccak256::new();
    h.update(domain);
    h.update(tag.0);
    h.update(row_bytes_be);
    h.finalize().into()
}

/// Convenience: hash a MAIN-trace row from individual field elements.
/// Allocates a row-sized buffer; suitable for verifier-side per-query
/// re-hashing (where allocation cost is dominated by FRI work anyway).
pub fn hash_tagged_row<E>(tag: MatrixTag, row: &[FieldElement<E>]) -> Commitment
where
    E: IsField,
    FieldElement<E>: ByteConversion,
{
    hash_tagged_row_inner::<E>(LEAF_DOMAIN_TAG_MAIN, tag, row)
}

/// Convenience: hash an AUX-trace row from individual field elements. Same
/// allocation pattern as [`hash_tagged_row`].
pub fn hash_tagged_row_aux<E>(tag: MatrixTag, row: &[FieldElement<E>]) -> Commitment
where
    E: IsField,
    FieldElement<E>: ByteConversion,
{
    hash_tagged_row_inner::<E>(LEAF_DOMAIN_TAG_AUX, tag, row)
}

#[inline]
fn hash_tagged_row_inner<E>(
    domain: &[u8],
    tag: MatrixTag,
    row: &[FieldElement<E>],
) -> Commitment
where
    E: IsField,
    FieldElement<E>: ByteConversion,
{
    let byte_len = <FieldElement<E> as ByteConversion>::BYTE_LEN;
    let mut buf = vec![0u8; row.len() * byte_len];
    for (col_idx, fe) in row.iter().enumerate() {
        fe.write_bytes_be(&mut buf[col_idx * byte_len..(col_idx + 1) * byte_len]);
    }
    hash_with_domain(domain, tag, &buf)
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

    #[test]
    fn main_and_aux_domains_separate() {
        // Same (tag, row) under the two domains MUST produce distinct
        // digests; otherwise an aux opening could authenticate a main leaf
        // (or vice versa).
        let tag = MatrixTag::new([0xAB; 8]);
        let row = vec![FE::from(42u64), FE::from(7u64)];
        let main_digest = hash_tagged_row(tag, &row);
        let aux_digest = hash_tagged_row_aux(tag, &row);
        assert_ne!(main_digest, aux_digest);
    }

    #[test]
    fn aux_bytes_helper_matches_aux_element_helper() {
        // The bytes-flavoured helper and the element-flavoured helper must
        // agree on the same input — same domain separator, same hash.
        let tag = MatrixTag::new([3; 8]);
        let row = vec![FE::from(11u64), FE::from(13u64), FE::from(17u64)];
        let byte_len = <FE as ByteConversion>::BYTE_LEN;
        let mut buf = vec![0u8; row.len() * byte_len];
        for (i, fe) in row.iter().enumerate() {
            fe.write_bytes_be(&mut buf[i * byte_len..(i + 1) * byte_len]);
        }
        assert_eq!(hash_tagged_row_bytes_aux(tag, &buf), hash_tagged_row_aux(tag, &row));
    }
}
