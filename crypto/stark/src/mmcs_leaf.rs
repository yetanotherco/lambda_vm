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

/// Versioned domain separator for COMPOSITION-trace MMCS leaves.
/// Composition leaves hash a PAIR of rows (br_0 || br_1) instead of a
/// single row — the legacy `keccak_leaves_row_pair_bit_reversed` shape.
/// Distinct from main/aux so no composition opening can authenticate a
/// main or aux leaf.
pub const LEAF_DOMAIN_TAG_COMPOSITION: &[u8] = b"LAMBDAVM_COMP_MMCS_LEAF_V1";

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

/// Hash a COMPOSITION-trace MMCS leaf from a pre-concatenated `(br_0 ||
/// br_1)` byte buffer — i.e. the two row-pair rows written big-endian,
/// `part_0_row_0 || part_1_row_0 || ... || part_0_row_1 || part_1_row_1
/// || ...`. Uses [`LEAF_DOMAIN_TAG_COMPOSITION`].
#[inline]
pub fn hash_tagged_row_pair_bytes_composition(
    tag: MatrixTag,
    row_pair_bytes_be: &[u8],
) -> Commitment {
    hash_with_domain(LEAF_DOMAIN_TAG_COMPOSITION, tag, row_pair_bytes_be)
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

/// Convenience: hash a COMPOSITION-trace row-pair from two slices of
/// field elements (the parts evaluated at `br_0` and `br_1`), each
/// `num_parts` long.
pub fn hash_tagged_row_pair_composition<E>(
    tag: MatrixTag,
    parts_at_br_0: &[FieldElement<E>],
    parts_at_br_1: &[FieldElement<E>],
) -> Commitment
where
    E: IsField,
    FieldElement<E>: ByteConversion,
{
    debug_assert_eq!(parts_at_br_0.len(), parts_at_br_1.len());
    let byte_len = <FieldElement<E> as ByteConversion>::BYTE_LEN;
    let num_parts = parts_at_br_0.len();
    let mut buf = vec![0u8; 2 * num_parts * byte_len];
    let mut offset = 0;
    for fe in parts_at_br_0 {
        fe.write_bytes_be(&mut buf[offset..offset + byte_len]);
        offset += byte_len;
    }
    for fe in parts_at_br_1 {
        fe.write_bytes_be(&mut buf[offset..offset + byte_len]);
        offset += byte_len;
    }
    hash_tagged_row_pair_bytes_composition(tag, &buf)
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
    fn composition_domain_separates_from_main_and_aux() {
        // Same row-pair under composition MUST differ from main + aux
        // domains so a composition opening can't authenticate a main or
        // aux leaf.
        let tag = MatrixTag::new([0xCC; 8]);
        let row0 = vec![FE::from(1u64), FE::from(2u64)];
        let row1 = vec![FE::from(3u64), FE::from(4u64)];
        let comp_digest = hash_tagged_row_pair_composition(tag, &row0, &row1);

        // Build the equivalent flat byte buffer manually and run it
        // through the main + aux single-domain helpers.
        let byte_len = <FE as ByteConversion>::BYTE_LEN;
        let mut flat = vec![0u8; (row0.len() + row1.len()) * byte_len];
        let mut offset = 0;
        for fe in row0.iter().chain(row1.iter()) {
            fe.write_bytes_be(&mut flat[offset..offset + byte_len]);
            offset += byte_len;
        }
        let main_digest = hash_tagged_row_bytes(tag, &flat);
        let aux_digest = hash_tagged_row_bytes_aux(tag, &flat);
        assert_ne!(comp_digest, main_digest);
        assert_ne!(comp_digest, aux_digest);
    }

    #[test]
    fn composition_bytes_helper_matches_composition_element_helper() {
        let tag = MatrixTag::new([5; 8]);
        let row0 = vec![FE::from(10u64), FE::from(20u64)];
        let row1 = vec![FE::from(30u64), FE::from(40u64)];
        let from_elements = hash_tagged_row_pair_composition(tag, &row0, &row1);

        let byte_len = <FE as ByteConversion>::BYTE_LEN;
        let mut flat = vec![0u8; 2 * row0.len() * byte_len];
        let mut offset = 0;
        for fe in row0.iter().chain(row1.iter()) {
            fe.write_bytes_be(&mut flat[offset..offset + byte_len]);
            offset += byte_len;
        }
        let from_bytes = hash_tagged_row_pair_bytes_composition(tag, &flat);
        assert_eq!(from_elements, from_bytes);
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
