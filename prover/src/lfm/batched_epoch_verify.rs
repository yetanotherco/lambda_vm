//! The batched epoch's verification legs — the mixed-height MMCS walk.
//!
//! The batched counterpart of [`super::sub_proof`]'s authentication half.
//! The order authority is `stark::fri::mmcs::MixedMmcs::verify_batch`
//! (fri/mmcs.rs' "Tree layout" section is the contract): ONE path
//! authenticates every matrix of a round — the tallest matrices batch into
//! the base leaf, and each shorter height group is INJECTED where the climb
//! reaches its layer, as one extra compression.
//!
//! Heights and widths are program shape, so the injection schedule UNROLLS at
//! emit time: the emitted walk is straight-line — per level one
//! compress-with-sibling (two `Select`s on the shared index bit) and, iff
//! some matrix sits at that level's injection height, one further compress
//! with that height group's leaf hash. No branch, no `Select` beyond the
//! sibling ordering, exactly as [`super::edsl::WrapHash::merkle_walk`]'s doc
//! anticipated ("a batched path that injects at mixed heights extends this
//! rather than replacing it").
//!
//! ## The index convention, in cells
//!
//! The machine's shared query index is a BIT VECTOR (low-to-high,
//! `h_max_fri − 1` bits, drawn once by the spine). `fri/mmcs.rs`' index
//! reduction — `iota_round = iota_fri >> (h_max_fri − h_max_round)` — is
//! [`reduce_iota_bits`]: DROP THE LOW BITS, keep the high ones. In LFM the
//! reduction is free (slicing a cell vector emits nothing), but the DIRECTION
//! is still the soundness-relevant choice: host-side a wrong shift is
//! self-consistent between prover and verifier and fails silently, which is
//! why `the_wrong_index_reduction_direction_fails_the_walk` ports the
//! `short_round_low_bit_convention_is_exercised` control to the machine.

use super::builder::{Bit, Cell, Felt, LfmBuilder};
use super::edsl::{self, WrapDigest};
use super::epoch::RootCells;
use super::sub_proof::GroupShape;

/// One matrix of a mixed round, as the walk consumes it — its shape (columns
/// and element kind), its height (the injection schedule's key), and its
/// opened row pair as the caller's CELLS. There is deliberately no
/// constructor that hints: the values are whatever the caller already holds,
/// which is what makes the authentication and the folds share them.
pub struct MixedMatrixOpening<'a> {
    pub shape: GroupShape,
    /// `log2` of the matrix's LDE height — where in the climb it enters.
    pub log_height: usize,
    /// `evaluations ‖ evaluations_sym` in leaf order — `2 · num_columns`
    /// cells.
    pub values: &'a [Cell],
}

/// The leaf hash of one HEIGHT GROUP's row pairs — `hash_group_openings`'
/// layout: every matrix's `evaluations ‖ evaluations_sym`, in round INPUT
/// order, flat, one hash. Each element renders exactly as the per-table leaf
/// does ([`super::sub_proof::emit_leaf_hash`]): a base element as its eight
/// big-endian bytes, an extension element as its three components, each eight
/// big-endian bytes. Lane 3 of an extension cell is NOT hashed — production
/// hashes three components — and the same caveat applies as there: every ext
/// value a query opens is also an ext operand of the DEEP crossing, which is
/// what pins lane 3 to zero.
pub fn emit_group_leaf_hash(b: &mut LfmBuilder, group: &[&MixedMatrixOpening<'_>]) -> WrapDigest {
    use super::keccak_host::BYTES_PER_HALF;
    use super::transcript_replay::felt_be_halves;

    assert!(!group.is_empty(), "a group leaf covers at least one matrix");
    let mut stream: Vec<Felt> = Vec::new();
    for m in group {
        assert_eq!(
            m.values.len(),
            m.shape.num_values(),
            "a matrix's opening covers its whole row pair"
        );
        for v in m.values {
            if m.shape.is_ext {
                let lanes = b.unpack(*v);
                for lane in lanes.iter().take(3) {
                    stream.extend(felt_be_halves(b, *lane));
                }
            } else {
                stream.extend(felt_be_halves(b, Felt(v.addr())));
            }
        }
    }
    let len_bytes = BYTES_PER_HALF * stream.len();
    edsl::wrap_hash_bytes(b, &stream, len_bytes)
}

/// Authenticate one mixed round's openings against its committed root — the
/// injecting walk, `MixedMmcs::verify_batch` emitted.
///
/// `matrices` in round INPUT order; `siblings` leaf level first,
/// `h_max − 1` of them; `bits` the REDUCED shared index, low-to-high,
/// `h_max − 1` of them ([`reduce_iota_bits`]). The final assert against the
/// root's lanes is the binding: the root cells are the SAME cells the spine
/// absorbed, so there is no second copy for a prover to disagree with.
pub fn emit_mixed_verify_batch(
    b: &mut LfmBuilder,
    root: &RootCells,
    matrices: &[MixedMatrixOpening<'_>],
    siblings: &[WrapDigest],
    bits: &[Bit],
) {
    let h_max = matrices
        .iter()
        .map(|m| m.log_height)
        .max()
        .expect("a round has at least one matrix");
    assert!(h_max >= 1, "a row-pair tree needs at least two rows");
    assert_eq!(siblings.len(), h_max - 1, "one sibling per level");
    assert_eq!(
        bits.len(),
        h_max - 1,
        "the reduced index has h_max − 1 bits"
    );
    for m in matrices {
        assert!(
            (1..=h_max).contains(&m.log_height),
            "a matrix's height sits inside its round's climb"
        );
    }

    // Base node: every tallest matrix's row pair, one leaf hash.
    let base: Vec<&MixedMatrixOpening<'_>> =
        matrices.iter().filter(|m| m.log_height == h_max).collect();
    let mut acc = emit_group_leaf_hash(b, &base);

    for (level, (bit, sibling)) in bits.iter().zip(siblings).enumerate() {
        // Both halves of the digest must swap on the SAME bit; bit = 0 means
        // the current node is the LEFT child, as in every walk here.
        let (l0, r0) = b.select(*bit, acc[0], sibling[0]);
        let (l1, r1) = b.select(*bit, acc[1], sibling[1]);
        let mut parent = edsl::wrap_hash_pair(b, [l0, l1], [r0, r1]);

        // The injection, unrolled: heights are shape, so whether a group
        // enters here is decided now, not by an emitted branch.
        let inject_h = h_max - 1 - level;
        let group: Vec<&MixedMatrixOpening<'_>> = matrices
            .iter()
            .filter(|m| m.log_height == inject_h)
            .collect();
        if !group.is_empty() {
            let inj = emit_group_leaf_hash(b, &group);
            parent = edsl::wrap_hash_pair(b, parent, inj);
        }
        acc = parent;
    }

    edsl::assert_word_eq_lanes(b, acc[0], &root.lanes[0]);
    edsl::assert_word_eq_lanes(b, acc[1], &root.lanes[1]);
}

/// Reduce the SHARED query-index bits to a round (or per-table tree) whose
/// own tallest height is `h_max_round` — `reduce_iota_to_round`'s
/// `iota >> (h_max_fri − h_max_round)`, on a low-to-high bit vector: drop
/// the LOW `h_max_fri − h_max_round` bits, keep the high `h_max_round − 1`.
///
/// Free — slicing emits nothing — but direction-critical; see the module doc.
pub fn reduce_iota_bits(bits: &[Bit], h_max_fri: usize, h_max_round: usize) -> &[Bit] {
    assert!(
        h_max_round <= h_max_fri,
        "no round is taller than the FRI's domain"
    );
    assert_eq!(
        bits.len(),
        h_max_fri - 1,
        "the shared index has h_max_fri − 1 bits"
    );
    &bits[(h_max_fri - h_max_round)..]
}
