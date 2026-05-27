//! Helpers that bridge per-chip LDE columns to the unified MMCS over the
//! main trace (PR2 of the streaming-MMCS plan).
//!
//! This module is **not yet wired into `multi_prove`**. It exists so the
//! API + leaf-hash format can be reviewed and tested in isolation before
//! the hot-path change. The pattern PR2 will use:
//!
//! 1. For each chip-chunk: compute its tagged leaf-digest array via
//!    [`compute_chip_leaves_with_tag`]. The chip's LDE columns can be
//!    dropped immediately after.
//! 2. Once every chip has produced its leaves, call
//!    [`build_main_trace_mmcs`] with the `(MatrixTag, leaves)` pairs to
//!    get a single MMCS root + the prover-side tree for opens.
//! 3. Absorb that one root into the transcript instead of N per-chip roots.
//! 4. Per query: `mmcs.open(global_index)` returns one `MmcsOpening`
//!    covering every chip at the appropriate shifted indices.
//!
//! The leaf-hash format is deliberately **distinct** from
//! `stark::prover::keccak_leaves_bit_reversed` — that one omits the
//! per-chip tag, which is why N independent trees today are safe (each
//! root inherently binds its content). With a single shared root the tag
//! must move into the leaf, and feeding the old bytes into the MMCS would
//! be a silent soundness bug.

use crypto::merkle_tree::mmcs::{Mmcs, MmcsBuilder, MmcsError, MmcsOpening};
use math::fft::bit_reversing::reverse_index;
use math::field::element::FieldElement;
use math::field::traits::IsField;
use math::traits::{AsBytes, ByteConversion};
use sha3::{Digest, Keccak256};
use stark::config::{BatchedMerkleTreeBackend, Commitment};

pub use crate::tables::mmcs_tags as tags;
pub use crypto::merkle_tree::mmcs::MatrixTag;

/// Domain tag prepended to every main-trace MMCS leaf hash so that
/// (a) the bytes are clearly versioned against any future change and
/// (b) they cannot collide with leaves of a different MMCS (aux trace,
/// composition, ...). Bump the suffix on any encoding change.
const LEAF_DOMAIN_TAG: &[u8] = b"LAMBDAVM_MAIN_MMCS_LEAF_V1";

/// Compute the per-row leaf digests for a chip's main-trace LDE,
/// binding the chip's `MatrixTag` into every leaf so the MMCS can
/// authenticate (matrix, row) pairs uniquely.
///
/// Each row is laid out bit-reversed (matching the existing FRI / Merkle
/// layout). The leaf is `Keccak256(LEAF_DOMAIN_TAG || tag.0 || row_bytes)`
/// where `row_bytes` is every column's element written big-endian and
/// concatenated.
///
/// The input columns are read but never mutated; the caller can drop
/// them immediately after this returns — memory peak is one chip's LDE
/// at a time (same as today's per-chip Merkle build).
pub fn compute_chip_leaves_with_tag<E>(
    columns: &[Vec<FieldElement<E>>],
    tag: MatrixTag,
) -> Vec<Commitment>
where
    E: IsField + Send + Sync,
    FieldElement<E>: AsBytes + Sync + Send + ByteConversion,
{
    if columns.is_empty() || columns[0].is_empty() {
        return Vec::new();
    }
    let num_rows = columns[0].len();
    let num_cols = columns.len();
    let byte_len = <FieldElement<E> as ByteConversion>::BYTE_LEN;
    debug_assert!(
        num_rows.is_power_of_two(),
        "num_rows must be a power of two for reverse_index"
    );

    let total_bytes = num_cols * byte_len;

    let hash_leaf = |buf: &mut [u8], row_idx: usize| -> Commitment {
        let br_idx = reverse_index(row_idx, num_rows as u64);
        for (col_idx, col) in columns.iter().enumerate() {
            col[br_idx].write_bytes_be(&mut buf[col_idx * byte_len..(col_idx + 1) * byte_len]);
        }
        let mut h = Keccak256::new();
        h.update(LEAF_DOMAIN_TAG);
        h.update(tag.0);
        h.update(&buf[..]);
        h.finalize().into()
    };

    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        (0..num_rows)
            .into_par_iter()
            .map_init(
                || vec![0u8; total_bytes],
                |buf, row_idx| hash_leaf(buf, row_idx),
            )
            .collect()
    }
    #[cfg(not(feature = "parallel"))]
    {
        let mut buf = vec![0u8; total_bytes];
        (0..num_rows)
            .map(|row_idx| hash_leaf(&mut buf, row_idx))
            .collect()
    }
}

/// Convenience: build the unified main-trace MMCS from `(tag, leaves)`
/// pairs that the caller produced via [`compute_chip_leaves_with_tag`].
pub fn build_main_trace_mmcs<F>(
    entries: Vec<(MatrixTag, Vec<Commitment>)>,
) -> Result<Mmcs<BatchedMerkleTreeBackend<F>>, MmcsError>
where
    F: IsField + Send + Sync,
    FieldElement<F>: AsBytes + Sync + Send,
{
    let mut builder = MmcsBuilder::<BatchedMerkleTreeBackend<F>>::new();
    for (tag, leaves) in entries {
        builder.add_matrix(tag, leaves)?;
    }
    builder.finalize()
}

/// Convenience opening accessor for tests / callers that don't want to
/// import `Mmcs` directly.
pub fn open_main_trace_mmcs<F>(
    mmcs: &Mmcs<BatchedMerkleTreeBackend<F>>,
    global_index: usize,
) -> Result<MmcsOpening<Commitment>, MmcsError>
where
    F: IsField + Send + Sync,
    FieldElement<F>: AsBytes + Sync + Send,
{
    mmcs.open(global_index)
}

#[cfg(test)]
mod tests {
    use super::*;
    use math::field::goldilocks::GoldilocksField;

    type FE = FieldElement<GoldilocksField>;

    fn fake_columns(seed: u64, num_cols: usize, num_rows: usize) -> Vec<Vec<FE>> {
        (0..num_cols)
            .map(|c| {
                (0..num_rows)
                    .map(|r| FE::from((seed.wrapping_add(c as u64) * 31 + r as u64) % 1_000_003))
                    .collect()
            })
            .collect()
    }

    #[test]
    fn leaves_change_when_tag_changes() {
        let cols = fake_columns(42, 4, 8);
        let tag_a = tags::chip_tag(tags::CHIP_CPU, 0);
        let tag_b = tags::chip_tag(tags::CHIP_CPU, 1);
        let la = compute_chip_leaves_with_tag(&cols, tag_a);
        let lb = compute_chip_leaves_with_tag(&cols, tag_b);
        assert_eq!(la.len(), 8);
        assert_eq!(la.len(), lb.len());
        assert_ne!(la[0], lb[0], "tag must be in the leaf");
        // Every row should differ; collision at one row would be extreme.
        assert!(la.iter().zip(lb.iter()).any(|(a, b)| a != b));
    }

    #[test]
    fn leaves_differ_from_legacy_format() {
        // Sanity: our tagged leaves are NOT equal to a Keccak256 of just
        // the row bytes (i.e. the legacy non-tagged format). Feeding old
        // bytes into the MMCS would be a silent soundness bug.
        let cols = fake_columns(1, 2, 4);
        let tag = tags::chip_tag(tags::CHIP_BITWISE, 0);
        let tagged = compute_chip_leaves_with_tag(&cols, tag);
        let untagged: Commitment = {
            let mut buf = [0u8; 2 * 8];
            let br = reverse_index(0, 4);
            for (c, col) in cols.iter().enumerate() {
                col[br].write_bytes_be(&mut buf[c * 8..(c + 1) * 8]);
            }
            let mut h = Keccak256::new();
            h.update(&buf[..]);
            h.finalize().into()
        };
        assert_ne!(tagged[0], untagged);
    }

    #[test]
    fn build_main_trace_mmcs_round_trips() {
        // 3 chips at distinct heights — realistic small case.
        let cols_a = fake_columns(1, 6, 16);
        let cols_b = fake_columns(2, 4, 8);
        let cols_c = fake_columns(3, 2, 4);
        let tag_a = tags::chip_tag(tags::CHIP_CPU, 0);
        let tag_b = tags::chip_tag(tags::CHIP_MEMW, 0);
        let tag_c = tags::chip_tag(tags::CHIP_BITWISE, 0);
        let leaves_a = compute_chip_leaves_with_tag(&cols_a, tag_a);
        let leaves_b = compute_chip_leaves_with_tag(&cols_b, tag_b);
        let leaves_c = compute_chip_leaves_with_tag(&cols_c, tag_c);
        let entries = vec![(tag_a, leaves_a), (tag_b, leaves_b), (tag_c, leaves_c)];
        let mmcs = build_main_trace_mmcs::<GoldilocksField>(entries).expect("build mmcs");
        let spec = mmcs.spec();
        // 16 is the max; open at every row in that domain.
        for i in 0..16 {
            let opening = mmcs.open(i).expect("open");
            assert!(
                opening.verify::<BatchedMerkleTreeBackend<GoldilocksField>>(mmcs.root(), &spec),
                "round-trip failed at index {i}"
            );
        }
    }

    #[test]
    fn build_main_trace_mmcs_same_height_chunks() {
        // 3 chips at the SAME height — exercises the same-height combine
        // path with realistic lambda-vm-style data (CPU chunks).
        let cols_0 = fake_columns(10, 8, 16);
        let cols_1 = fake_columns(11, 8, 16);
        let cols_2 = fake_columns(12, 8, 16);
        let entries = vec![
            (
                tags::chip_tag(tags::CHIP_CPU, 0),
                compute_chip_leaves_with_tag(&cols_0, tags::chip_tag(tags::CHIP_CPU, 0)),
            ),
            (
                tags::chip_tag(tags::CHIP_CPU, 1),
                compute_chip_leaves_with_tag(&cols_1, tags::chip_tag(tags::CHIP_CPU, 1)),
            ),
            (
                tags::chip_tag(tags::CHIP_CPU, 2),
                compute_chip_leaves_with_tag(&cols_2, tags::chip_tag(tags::CHIP_CPU, 2)),
            ),
        ];
        let mmcs = build_main_trace_mmcs::<GoldilocksField>(entries).expect("build mmcs");
        let spec = mmcs.spec();
        for i in 0..16 {
            let opening = mmcs.open(i).expect("open");
            assert!(
                opening.verify::<BatchedMerkleTreeBackend<GoldilocksField>>(mmcs.root(), &spec)
            );
        }
    }

    #[test]
    fn duplicate_tag_caught_at_build() {
        // Two chips sharing a tag is a caller bug (e.g. forgot to bump
        // chunk_index). MMCS rejects at finalize time.
        let cols = fake_columns(7, 2, 4);
        let tag = tags::chip_tag(tags::CHIP_CPU, 0);
        let entries = vec![
            (tag, compute_chip_leaves_with_tag(&cols, tag)),
            (tag, compute_chip_leaves_with_tag(&cols, tag)),
        ];
        let err = build_main_trace_mmcs::<GoldilocksField>(entries);
        assert!(matches!(err, Err(MmcsError::DuplicateTag)));
    }
}
