//! Soundness negatives for the batched-commitment primitives — the mixed-height
//! MMCS ([`crate::fri::mmcs`]) and the batched-FRI transcript
//! ([`crate::fri::batched`]).
//!
//! Each test builds one honest commitment over a small mixed-height epoch, then
//! tampers a single component and asserts rejection. The honest opening is
//! re-asserted in every test, so a false-reject regression cannot make the
//! negatives pass vacuously.
//!
//! Scope: these reach only what the primitives decide. The forgeries that a
//! batched *proof* must also resist — a tampered per-query FRI layer evaluation,
//! an OOD value, the bus balance, the query count, the grinding nonce — need the
//! prover/verifier integration and belong with it.

use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use crypto::fiat_shamir::is_transcript::IsTranscript;
use math::fft::bit_reversing::reverse_index;
use math::field::element::FieldElement;
use math::field::goldilocks::GoldilocksField;

use crate::config::KeccakStarkHash;
use crate::fri::batched::{
    BatchedFriLayout, absorb_shape_histogram, derive_batched_fri_challenges,
};
use crate::fri::mmcs::{LeafSource, MixedMmcs, MixedOpening};

type F = GoldilocksField;
type FE = FieldElement<F>;
type Mmcs = MixedMmcs<F, KeccakStarkHash>;
type Transcript = DefaultTranscript<F>;

/// Bit-reversed row-major matrices, in the layout the MMCS commits.
struct Matrices {
    /// `(bit-reversed row-major data, log_height, width)`.
    mats: Vec<(Vec<FE>, usize, usize)>,
}

impl LeafSource<F> for Matrices {
    fn num_matrices(&self) -> usize {
        self.mats.len()
    }
    fn log_height(&self, m: usize) -> usize {
        self.mats[m].1
    }
    fn width(&self, m: usize) -> usize {
        self.mats[m].2
    }
    fn append_row(&self, m: usize, bitrev_row: usize, out: &mut Vec<FE>) {
        let (data, _, width) = &self.mats[m];
        out.extend_from_slice(&data[bitrev_row * width..(bitrev_row + 1) * width]);
    }
}

fn matrix(log_height: usize, width: usize, seed: u64) -> (Vec<FE>, usize, usize) {
    let num_rows = 1usize << log_height;
    let mut data = vec![FE::from(0u64); num_rows * width];
    for (r, chunk) in data.chunks_exact_mut(width).enumerate() {
        let br = reverse_index(r, num_rows as u64);
        for (c, slot) in chunk.iter_mut().enumerate() {
            *slot = FE::from(seed.wrapping_mul(31) + (c as u64) * 1009 + (br as u64) * 7 + 1);
        }
    }
    (data, log_height, width)
}

/// A four-matrix epoch: two tall (base group), one injected, one injected lower.
/// Heights {5, 5, 4, 2}, widths {3, 3, 2, 4}. Two of the tall matrices share a
/// width so the "swap two openings" forgery below is a pure reordering.
fn epoch() -> (Matrices, Vec<usize>, Vec<usize>) {
    let mats = Matrices {
        mats: vec![
            matrix(5, 3, 11),
            matrix(5, 3, 22),
            matrix(4, 2, 33),
            matrix(2, 4, 44),
        ],
    };
    let heights = vec![5, 5, 4, 2];
    let widths = vec![3, 3, 2, 4];
    (mats, heights, widths)
}

const IOTA: usize = 9;

fn honest() -> ([u8; 32], MixedOpening<F>, Vec<usize>, Vec<usize>) {
    let (mats, heights, widths) = epoch();
    let mmcs = Mmcs::commit(&mats);
    let opening = mmcs.open_batch(IOTA, &mats);
    (mmcs.root(), opening, heights, widths)
}

/// Sanity anchor: the untampered opening verifies.
#[test]
fn honest_batched_opening_verifies() {
    let (root, opening, heights, widths) = honest();
    assert!(
        Mmcs::verify_batch(&root, IOTA, &opening, &heights, &widths),
        "an honest mixed-height opening must verify"
    );
}

/// Tampering any matrix's opened row breaks the one shared authentication path —
/// including the SHORT matrices, which are bound through injection rather than
/// through the base leaf.
#[test]
fn rejects_a_tampered_row_in_every_height_group() {
    let (root, opening, heights, widths) = honest();
    for m in 0..opening.per_matrix.len() {
        let mut tampered = opening.clone();
        tampered.per_matrix[m].evaluations[0] =
            &tampered.per_matrix[m].evaluations[0] + &FE::from(1u64);
        assert!(
            !Mmcs::verify_batch(&root, IOTA, &tampered, &heights, &widths),
            "a tampered row of matrix {m} (height {}) must be rejected",
            heights[m]
        );

        let mut tampered_sym = opening.clone();
        tampered_sym.per_matrix[m].evaluations_sym[0] =
            &tampered_sym.per_matrix[m].evaluations_sym[0] + &FE::from(1u64);
        assert!(
            !Mmcs::verify_batch(&root, IOTA, &tampered_sym, &heights, &widths),
            "a tampered symmetric row of matrix {m} must be rejected"
        );
    }
}

/// Tampering the shared authentication path itself.
#[test]
fn rejects_a_tampered_authentication_path() {
    let (root, opening, heights, widths) = honest();
    for level in 0..opening.proof.merkle_path.len() {
        let mut tampered = opening.clone();
        tampered.proof.merkle_path[level][0] ^= 1;
        assert!(
            !Mmcs::verify_batch(&root, IOTA, &tampered, &heights, &widths),
            "a tampered sibling at level {level} must be rejected"
        );
    }
    // Truncating or padding the path is a shape error, not a hash mismatch.
    let mut short = opening.clone();
    short.proof.merkle_path.pop();
    assert!(!Mmcs::verify_batch(&root, IOTA, &short, &heights, &widths));
    let mut long = opening.clone();
    long.proof.merkle_path.push([0u8; 32]);
    assert!(!Mmcs::verify_batch(&root, IOTA, &long, &heights, &widths));
}

/// An honest opening replayed at a different query index must be rejected: the
/// path is position-dependent, so one opening does not authenticate every leaf.
#[test]
fn rejects_an_opening_replayed_at_another_index() {
    let (root, opening, heights, widths) = honest();
    let n0 = 1usize << (5 - 1);
    for iota in 0..n0 {
        let accepted = Mmcs::verify_batch(&root, iota, &opening, &heights, &widths);
        assert_eq!(
            accepted,
            iota == IOTA,
            "the opening at {IOTA} must verify at {IOTA} and nowhere else (index {iota})"
        );
    }
    // And past the tree's leaf range — the index-convention guard.
    assert!(!Mmcs::verify_batch(&root, n0, &opening, &heights, &widths));
}

/// INPUT ORDER is part of the commitment: swapping two same-height, same-width
/// matrices' openings changes the flat concatenation the group leaf hashes, so
/// the tree no longer reproduces. Without order-dependence a prover could serve
/// one table's rows in another's slot.
#[test]
fn rejects_swapped_openings_within_a_height_group() {
    let (root, opening, heights, widths) = honest();
    assert_eq!(
        (heights[0], widths[0]),
        (heights[1], widths[1]),
        "matrices 0 and 1 must share a shape for this to be a pure reordering"
    );
    let mut swapped = opening.clone();
    swapped.per_matrix.swap(0, 1);
    assert_ne!(
        swapped.per_matrix[0].evaluations, opening.per_matrix[0].evaluations,
        "the two matrices must carry different data"
    );
    assert!(
        !Mmcs::verify_batch(&root, IOTA, &swapped, &heights, &widths),
        "reordering two same-shape matrices must be rejected"
    );
}

/// The verifier's `heights` fix the injection schedule. Relabelling a matrix's
/// height — claiming the height-4 matrix is height 3, so it is injected a level
/// later — must not reproduce the root, or a prover could move a table to a
/// layer where its rows are checked against a different query position.
#[test]
fn rejects_a_relabelled_injection_height() {
    let (root, opening, heights, widths) = honest();
    let mut relabelled = heights.clone();
    relabelled[2] = 3;
    assert!(
        !Mmcs::verify_batch(&root, IOTA, &opening, &relabelled, &widths),
        "moving a matrix to another injection level must be rejected"
    );

    // Promoting a short matrix into the base group is likewise rejected.
    let mut promoted = heights.clone();
    promoted[3] = 5;
    assert!(!Mmcs::verify_batch(
        &root, IOTA, &opening, &promoted, &widths
    ));
}

/// Widths are verifier-supplied and length-checked, so a width that does not
/// match the opening is rejected before any hashing — the guard that closes the
/// leaf-boundary shift.
#[test]
fn rejects_widths_that_disagree_with_the_opening() {
    let (root, opening, heights, widths) = honest();
    for m in 0..widths.len() {
        let mut wrong = widths.clone();
        wrong[m] += 1;
        assert!(
            !Mmcs::verify_batch(&root, IOTA, &opening, &heights, &wrong),
            "a width disagreeing with matrix {m}'s opening must be rejected"
        );
    }
}

/// A root committed over a different epoch shape does not authenticate this
/// opening, even where the tree depth coincides.
#[test]
fn rejects_a_root_from_another_epoch_shape() {
    let (_, opening, heights, widths) = honest();
    let other = Matrices {
        mats: vec![
            matrix(5, 3, 11),
            matrix(5, 3, 22),
            matrix(4, 2, 33),
            // Same height and width, different data.
            matrix(2, 4, 99),
        ],
    };
    let other_root = Mmcs::commit(&other).root();
    assert!(
        !Mmcs::verify_batch(&other_root, IOTA, &opening, &heights, &widths),
        "an opening must not verify against another epoch's root"
    );
}

/// The round-4 transcript binds the shape and every committed FRI layer, so
/// tampering a layer root or a terminal coefficient moves the query indices the
/// prover must answer at. This is what stops a prover from choosing its FRI
/// commitments after seeing the queries.
#[test]
fn tampering_the_fri_transcript_moves_the_query_indices() {
    let heights = vec![10usize, 10, 8, 7];
    let widths = vec![4usize, 2, 3, 1];
    let (blowup_log, k) = (1u32, 5u32);
    let layout = BatchedFriLayout::new(10, 7, blowup_log, k);
    let roots: Vec<[u8; 32]> = (0u8..layout.num_committed as u8).map(|i| [i; 32]).collect();
    let coeffs: Vec<FE> = (0..(1u64 << layout.effective_k)).map(FE::from).collect();

    let derive = |roots: &[[u8; 32]], coeffs: &[FE], heights: &[usize], widths: &[usize]| {
        derive_batched_fri_challenges(
            &mut Transcript::new(b"batched_soundness"),
            heights,
            widths,
            roots,
            coeffs,
            blowup_log,
            k,
            0,
            None,
            16,
        )
        .expect("a well-formed layer-root and coefficient count")
        .iotas
    };

    let base = derive(&roots, &coeffs, &heights, &widths);
    assert!(!base.is_empty());

    let mut other_root = roots.clone();
    other_root[0][0] ^= 1;
    assert_ne!(
        base,
        derive(&other_root, &coeffs, &heights, &widths),
        "a tampered FRI layer root must move the query indices"
    );

    let mut other_coeffs = coeffs.clone();
    other_coeffs[0] = &other_coeffs[0] + &FE::from(1u64);
    assert_ne!(
        base,
        derive(&roots, &other_coeffs, &heights, &widths),
        "a tampered terminal coefficient must move the query indices"
    );

    let mut other_heights = heights.clone();
    other_heights[2] = 9;
    assert_ne!(
        base,
        derive(&roots, &coeffs, &other_heights, &widths),
        "a tampered height must move the query indices"
    );

    let mut other_widths = widths.clone();
    other_widths[2] = 4;
    assert_ne!(
        base,
        derive(&roots, &coeffs, &heights, &other_widths),
        "a tampered width must move the query indices"
    );
}

/// The shape histogram's encoding is injective: no two distinct epoch shapes
/// absorb the same bytes. A collision would let a prover present one shape to
/// the transcript and another to the opening parse.
#[test]
fn the_shape_encoding_separates_distinct_epochs() {
    let absorbed = |heights: &[usize], widths: &[usize]| {
        let mut t = Transcript::new(b"shape");
        absorb_shape_histogram(&mut t, heights, widths);
        t.state()
    };

    // The classic ambiguity a length prefix and fixed-width fields must close:
    // one table of shape (h, w) against two tables whose fields interleave to the
    // same sequence.
    let one = absorbed(&[3, 4], &[4, 5]);
    let two = absorbed(&[3], &[4]);
    let three = absorbed(&[3, 4, 5], &[4, 5, 6]);
    assert_ne!(one, two);
    assert_ne!(one, three);
    assert_ne!(two, three);

    // Swapping height and width within a table is a different epoch.
    assert_ne!(absorbed(&[3, 4], &[4, 3]), absorbed(&[4, 3], &[3, 4]));
}
