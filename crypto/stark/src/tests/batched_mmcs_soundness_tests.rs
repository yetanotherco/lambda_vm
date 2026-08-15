//! Soundness negatives for the batched-commitment primitives — the mixed-height
//! MMCS ([`crate::fri::mmcs`]) and the batched-FRI transcript
//! ([`crate::fri::batched`]).
//!
//! Each test builds one honest commitment over a small mixed-height epoch, then
//! tampers a single component and asserts rejection. The honest opening is
//! re-asserted in every test, so a false-reject regression cannot make the
//! negatives pass vacuously.
//!
//! Scope grows with the integration. The first section reaches only what the
//! primitives decide; the per-query batched-FRI section below arrived with the
//! round-4 wiring ([`crate::batched::round4`]), which is what made a tampered
//! layer evaluation, a mis-sized decommitment and a wrong injection expressible.
//! The forgeries that still need the full prover/verifier integration — an OOD
//! value, the bus balance, the query count, the grinding nonce — belong with it.

use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use crypto::fiat_shamir::is_transcript::IsTranscript;
use math::fft::bit_reversing::reverse_index;
use math::field::element::FieldElement;
use math::field::goldilocks::GoldilocksField;

use crate::batched::round4::BatchedFriCommit;
use crate::batched::round4::tests as round4_tests;
use crate::config::KeccakStarkHash;
use crate::fri::batched::{
    BatchedFriLayout, absorb_shape_histogram, derive_batched_fri_challenges,
};
use crate::fri::fri_decommit::FriDecommitment;
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

// ---------------------------------------------------------------------------
// Per-query batched FRI (M-3). These need the round-4 wiring, not only the
// primitives, so they were deferred when the primitives landed.
// ---------------------------------------------------------------------------

/// One honest batched round 4 plus everything a verifier needs to check a query.
struct Round4Fixture {
    tables: Vec<round4_tests::FakeTable>,
    commit: BatchedFriCommit<F, KeccakStarkHash>,
    betas: Vec<FE>,
    decommitments: Vec<FriDecommitment<F>>,
    alpha: FE,
    h_max: usize,
    plan: crate::fri::batched::FriInstancePlan,
}

impl Round4Fixture {
    fn build() -> Self {
        let tables = round4_tests::fixture();
        let mut transcript = round4_tests::Transcript::new(b"batched_soundness_r4");
        let commit = round4_tests::commit_fixture(&tables, &mut transcript, 0, 6);
        let decommitments =
            crate::fri::query_phase::<F, KeccakStarkHash>(&commit.layers, &commit.iotas);

        let mut verifier_transcript = round4_tests::Transcript::new(b"batched_soundness_r4");
        let replay = crate::batched::round4::replay_batched_fri::<F, round4_tests::Transcript>(
            &mut verifier_transcript,
            &round4_tests::heights_of(&tables),
            &round4_tests::widths_of(&tables),
            &commit.layer_roots,
            &commit.final_poly_coeffs,
            round4_tests::BLOWUP_LOG,
            round4_tests::FINAL_POLY_LOG_DEGREE,
            0,
            None,
            6,
        )
        .expect("an honest shape must derive");

        Self {
            tables,
            betas: replay.betas,
            alpha: replay.alpha,
            h_max: replay.plan.h_max,
            plan: replay.plan,
            decommitments,
            commit,
        }
    }

    /// Verify query `q` with every input honest except what `mutate` changes.
    fn check_query_with<M>(&self, q: usize, mutate: M) -> bool
    where
        M: FnOnce(&mut FriDecommitment<F>, &mut (FE, FE), &mut Vec<Option<FE>>, &mut Vec<FE>),
    {
        let iota = self.commit.iotas[q];
        let (mut p0, mut buckets) = round4_tests::query_inputs(&self.tables, &self.alpha, iota);
        let mut decommitment = self.decommitments[q].clone();
        let mut coeffs = self.commit.final_poly_coeffs.clone();
        mutate(&mut decommitment, &mut p0, &mut buckets, &mut coeffs);
        round4_tests::verify_one_query(
            &self.commit,
            &self.betas,
            self.h_max,
            iota,
            &decommitment,
            (&p0.0, &p0.1),
            &buckets,
            &self.commit.layer_roots,
            &coeffs,
        )
    }

    fn check_query(&self, q: usize) -> bool {
        self.check_query_with(q, |_, _, _, _| {})
    }
}

/// The honest-path control for every negative below. Also pins that the fixture
/// is not degenerate: it must actually commit layers, or the fold loop the
/// negatives target would never run.
#[test]
fn honest_batched_fri_queries_verify() {
    let f = Round4Fixture::build();
    assert!(
        f.commit.layout.num_committed >= 1,
        "the fixture must commit at least one FRI layer"
    );
    assert!(
        f.commit.layout.total_folds as usize > f.commit.layout.num_committed,
        "the fixture must exercise the final fold"
    );
    for q in 0..f.commit.iotas.len() {
        assert!(f.check_query(q), "honest query {q} must verify");
    }
}

/// A per-query FRI layer evaluation is prover-supplied and NOT in the
/// transcript; only the layer's Merkle root binds it.
#[test]
fn a_tampered_fri_layer_evaluation_is_rejected() {
    let f = Round4Fixture::build();
    for q in 0..f.commit.iotas.len() {
        assert!(f.check_query(q), "honest control for query {q}");
        for layer in 0..f.commit.layout.num_committed {
            assert!(
                !f.check_query_with(q, |d, _, _, _| {
                    d.layers_evaluations_sym[layer] =
                        &d.layers_evaluations_sym[layer] + &FE::from(1u64);
                }),
                "query {q}: a tampered evaluation at layer {layer} must be rejected"
            );
        }
    }
}

/// The authentication path is what carries the layer opening to the root.
#[test]
fn a_tampered_fri_layer_auth_path_is_rejected() {
    let f = Round4Fixture::build();
    for layer in 0..f.commit.layout.num_committed {
        assert!(f.check_query(0), "honest control");
        assert!(
            !f.check_query_with(0, |d, _, _, _| {
                d.layers_auth_paths[layer].merkle_path[0][0] ^= 1;
            }),
            "a tampered sibling at layer {layer} must be rejected"
        );
    }
}

/// The decommitment vectors are not bound by Fiat-Shamir, so their lengths have
/// to be pinned before anything iterates them: a short one would end the fold
/// early and accept without reaching the terminal, a long one would run past it.
#[test]
fn a_mis_sized_fri_decommitment_is_rejected() {
    let f = Round4Fixture::build();
    assert!(f.check_query(0), "honest control");

    assert!(
        !f.check_query_with(0, |d, _, _, _| {
            d.layers_auth_paths.pop();
            d.layers_evaluations_sym.pop();
        }),
        "a truncated decommitment must be rejected"
    );
    assert!(
        !f.check_query_with(0, |d, _, _, _| {
            let path = d.layers_auth_paths[0].clone();
            let evaluation = d.layers_evaluations_sym[0];
            d.layers_auth_paths.push(path);
            d.layers_evaluations_sym.push(evaluation);
        }),
        "a padded decommitment must be rejected"
    );
    assert!(
        !f.check_query_with(0, |d, _, _, _| {
            d.layers_auth_paths.clear();
            d.layers_evaluations_sym.clear();
        }),
        "an empty decommitment must be rejected, not accepted vacuously"
    );
}

/// The terminal polynomial is where FRI's low-degree claim is finally cashed in.
#[test]
fn a_tampered_terminal_coefficient_is_rejected() {
    let f = Round4Fixture::build();
    assert!(f.check_query(0), "honest control");
    for i in 0..f.commit.final_poly_coeffs.len() {
        assert!(
            !f.check_query_with(0, |_, _, _, coeffs| {
                coeffs[i] = &coeffs[i] + &FE::from(1u64);
            }),
            "a tampered terminal coefficient {i} must be rejected"
        );
    }
}

/// The tallest tables enter FRI as layer 0, which is never committed — the only
/// thing binding them is that the fold has to land on the terminal.
#[test]
fn a_tampered_layer_zero_value_is_rejected() {
    let f = Round4Fixture::build();
    assert!(f.check_query(0), "honest control");
    assert!(
        !f.check_query_with(0, |_, p0, _, _| { p0.0 = &p0.0 + &FE::from(1u64) }),
        "a tampered p0 must be rejected"
    );
    assert!(
        !f.check_query_with(0, |_, p0, _, _| { p0.1 = &p0.1 + &FE::from(1u64) }),
        "a tampered p0 symmetric value must be rejected"
    );
    assert!(
        !f.check_query_with(0, |_, p0, _, _| { core::mem::swap(&mut p0.0, &mut p0.1) }),
        "swapping the layer-0 pair must be rejected — the two are not interchangeable"
    );
}

/// The injected buckets are the whole point of a mixed-height batch: a short
/// table is bound ONLY by the value it contributes at its injection layer. Three
/// ways to get that wrong, all of which leave the tall tables untouched and so
/// would pass a control that only tampered the base group.
#[test]
fn a_wrong_injection_is_rejected() {
    let f = Round4Fixture::build();
    // The BATCHED class's short heights — the standalone class is not injected at
    // all, and asking for its bucket would be asking about a codeword that is not
    // in this instance.
    let injected_heights: Vec<usize> = f
        .plan
        .batched
        .iter()
        .map(|&t| f.tables[t].height)
        .filter(|h| *h < f.h_max)
        .collect();
    assert!(
        !injected_heights.is_empty(),
        "the fixture must have at least one injected height"
    );

    for q in 0..f.commit.iotas.len() {
        assert!(f.check_query(q), "honest control for query {q}");
        for &h in &injected_heights {
            assert!(
                !f.check_query_with(q, |_, _, buckets, _| {
                    let value = buckets[h].take().expect("the height is occupied");
                    buckets[h] = Some(&value + &FE::from(1u64));
                }),
                "query {q}: a tampered injection at height {h} must be rejected"
            );
            assert!(
                !f.check_query_with(q, |_, _, buckets, _| { buckets[h] = None }),
                "query {q}: dropping the injection at height {h} must be rejected"
            );
        }
    }
}

/// The injection position is derived, not sent, and prover and verifier derive it
/// separately — so a control that only tampers the VALUE would pass under a wrong
/// derivation. Reading the other row of the same opened pair is the mistake a
/// off-by-one in `injection_position` would make, so it is the one to pin.
#[test]
fn an_injection_read_at_the_sibling_row_is_rejected() {
    let f = Round4Fixture::build();
    let mut exercised = 0usize;
    for (q, &iota) in f.commit.iotas.iter().enumerate() {
        assert!(f.check_query(q), "honest control for query {q}");
        for &t in f.plan.batched.iter() {
            let h = f.tables[t].height;
            if h == f.h_max {
                continue;
            }
            let position = crate::batched::round4::injection_position(iota, f.h_max, h);
            let sibling = position ^ 1;
            let inputs: Vec<(Vec<FE>, usize)> = f
                .plan
                .batched
                .iter()
                .map(|&b| (f.tables[b].codeword.clone(), f.tables[b].height))
                .collect();
            let combined = crate::fri::batched::combine_by_height(&inputs, &f.alpha);
            let bucket = combined[h].as_ref().expect("the height is occupied");
            // A degenerate codeword whose two rows coincide would make this
            // vacuous; skip rather than assert a rejection that means nothing.
            if bucket[position] == bucket[sibling] {
                continue;
            }
            exercised += 1;
            let sibling_value = bucket[sibling];
            assert!(
                !f.check_query_with(q, |_, _, buckets, _| {
                    buckets[h] = Some(sibling_value);
                }),
                "query {q}: reading height {h}'s injection at the sibling row must be rejected"
            );
        }
    }
    assert!(
        exercised > 0,
        "no non-degenerate sibling pair was exercised — the test proved nothing"
    );
}

/// Everything on the verifier's path is prover-supplied, so it must fail closed
/// on shapes that cannot occur honestly rather than panic on them.
#[test]
fn malformed_batched_fri_inputs_are_rejected_without_panicking() {
    let f = Round4Fixture::build();
    let iota = f.commit.iotas[0];
    let (p0, buckets) = round4_tests::query_inputs(&f.tables, &f.alpha, iota);
    let terminal_offset =
        FE::from(round4_tests::COSET_OFFSET).pow(1u64 << f.commit.layout.total_folds);
    let terminal = crate::fri::terminal::terminal_codeword_from_coeffs::<F, F>(
        &f.commit.final_poly_coeffs,
        &terminal_offset,
        f.commit.layout.terminal_len,
    );
    let point_inv = round4_tests::evaluation_point_inv(iota, f.h_max);

    let run = |layer_roots: &[[u8; 32]],
               betas: &[FE],
               h_max: usize,
               iota: usize,
               buckets: &[Option<FE>],
               terminal: &[FE]| {
        crate::batched::round4::verify_batched_fri_query::<F, F, KeccakStarkHash>(
            layer_roots,
            betas,
            &f.commit.layout,
            h_max,
            iota,
            &f.decommitments[0],
            &point_inv,
            (&p0.0, &p0.1),
            buckets,
            terminal,
        )
    };

    assert!(
        run(
            &f.commit.layer_roots,
            &f.betas,
            f.h_max,
            iota,
            &buckets,
            &terminal
        ),
        "honest control"
    );
    assert!(
        !run(&[], &f.betas, f.h_max, iota, &buckets, &terminal),
        "a missing layer-root vector must be rejected"
    );
    assert!(
        !run(
            &f.commit.layer_roots,
            &[],
            f.h_max,
            iota,
            &buckets,
            &terminal
        ),
        "a missing beta vector must be rejected"
    );
    assert!(
        !run(
            &f.commit.layer_roots,
            &f.betas,
            0,
            iota,
            &buckets,
            &terminal
        ),
        "h_max = 0 must be rejected, not shifted by"
    );
    assert!(
        !run(
            &f.commit.layer_roots,
            &f.betas,
            f.h_max,
            1usize << (f.h_max - 1),
            &buckets,
            &terminal
        ),
        "an iota from a taller domain must be rejected"
    );
    assert!(
        !run(
            &f.commit.layer_roots,
            &f.betas,
            f.h_max,
            iota,
            &buckets[..1],
            &terminal
        ),
        "a bucket vector too short to cover every height must be rejected"
    );
    assert!(
        !run(
            &f.commit.layer_roots,
            &f.betas,
            f.h_max,
            iota,
            &buckets,
            &[]
        ),
        "an empty terminal codeword must be rejected"
    );
}

/// ★ The control the two-class split requires: a table of EACH class must be
/// tamper-checked.
///
/// The classes read the same query index in different spaces — the batched class
/// uses `iota` directly, a standalone table uses `iota >> (h_max - h)`. Prover
/// and verifier both derive that shift from the shape, so a wrong convention is
/// self-consistent and honest proofs keep verifying; the failure is that the
/// standalone tables end up checked at positions nothing else reaches. A control
/// that only tampered the batched class would pass under ANY convention for the
/// other, which is exactly how consolidating a per-table check loses coverage.
#[test]
fn each_instance_class_is_tamper_checked() {
    let f = Round4Fixture::build();
    assert!(
        !f.plan.standalone.is_empty(),
        "the fixture must exercise BOTH classes, or this control proves nothing \
         about the split"
    );
    assert!(
        f.plan.batched.len() > 1,
        "the batched class must carry more than the tallest table"
    );

    // --- Batched class: covered by the fold recursion. ---
    assert!(f.check_query(0), "honest control");
    assert!(
        !f.check_query_with(0, |_, p0, _, _| { p0.0 = &p0.0 + &FE::from(1u64) }),
        "a tampered batched-class value must be rejected"
    );

    // --- Standalone class: its own terminal-only instance. ---
    for &t in &f.plan.standalone {
        let table = &f.tables[t];
        // A zero-layer table's terminal codeword IS its deep-composition
        // codeword — nothing folds — so the honest terminal is the codeword.
        let terminal = &table.codeword;
        for (q, &iota) in f.commit.iotas.iter().enumerate() {
            let reduced = crate::batched::round4::reduce_iota_to_round(iota, f.h_max, table.height)
                .expect("a standalone table is never taller than the FRI");
            let honest = (terminal[reduced * 2], terminal[reduced * 2 + 1]);
            assert!(
                crate::batched::round4::verify_standalone_fri_query::<F>(
                    iota,
                    f.h_max,
                    table.height,
                    (&honest.0, &honest.1),
                    terminal,
                ),
                "query {q}: the honest standalone opening must verify"
            );
            let tampered = &honest.0 + &FE::from(1u64);
            assert!(
                !crate::batched::round4::verify_standalone_fri_query::<F>(
                    iota,
                    f.h_max,
                    table.height,
                    (&tampered, &honest.1),
                    terminal,
                ),
                "query {q}: a tampered standalone value must be rejected"
            );
            // The index rule itself: reading the table at the UNREDUCED batched
            // index is the mistake the two-class split makes possible, and it is
            // silent unless something rejects it.
            if iota != reduced && iota * 2 + 1 < terminal.len() {
                assert!(
                    !crate::batched::round4::verify_standalone_fri_query::<F>(
                        iota,
                        f.h_max,
                        table.height,
                        (&terminal[iota * 2], &terminal[iota * 2 + 1]),
                        terminal,
                    ),
                    "query {q}: the un-reduced index must not authenticate a \
                     standalone table"
                );
            }
        }
    }
}

/// A standalone table must not be reachable through the batched instance's
/// injection path: it contributes no bucket, so a prover that manufactured one
/// is claiming a codeword this instance never mixed.
#[test]
fn a_standalone_table_contributes_no_injection() {
    let f = Round4Fixture::build();
    assert!(
        !f.plan.standalone.is_empty(),
        "the fixture needs both classes"
    );
    for &t in &f.plan.standalone {
        let h = f.tables[t].height;
        assert!(
            !f.plan.batched.iter().any(|&b| f.tables[b].height == h),
            "the fixture's standalone height must be unique to that class"
        );
        for q in 0..f.commit.iotas.len() {
            assert!(f.check_query(q), "honest control for query {q}");
            assert!(
                !f.check_query_with(q, |_, _, buckets, _| {
                    buckets[h] = Some(FE::from(7u64));
                }),
                "query {q}: a bucket manufactured at a standalone height must be \
                 rejected"
            );
        }
    }
}
