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
use crate::config::DefaultStarkHash;
use crate::fri::batched::{
    BatchedFriLayout, absorb_shape_histogram, derive_batched_fri_challenges,
};
use crate::fri::fri_decommit::FriDecommitment;
use crate::fri::mmcs::{LeafSource, MixedMmcs, MixedOpening};

type F = GoldilocksField;
type FE = FieldElement<F>;
type Mmcs = MixedMmcs<F, DefaultStarkHash>;
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
    commit: BatchedFriCommit<F, DefaultStarkHash>,
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
            crate::fri::query_phase::<F, DefaultStarkHash>(&commit.layers, &commit.iotas);

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
        crate::batched::round4::verify_batched_fri_query::<F, F, DefaultStarkHash>(
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

// ===========================================================================
// EPOCH-LEVEL NEGATIVES — the items M-2 deferred "with the integration"
// ===========================================================================
//
// Everything above decides what the PRIMITIVES can decide: a tampered row, a
// mis-sized path, a replayed index, a swapped opening. These need a whole
// epoch, so they arrive with `multi_prove_batched` and
// `batched::verifier::replay_epoch_transcript`.
//
// ⚠ What is covered and what is not. Query count, the grinding nonce, the OOD
// values and the bus-contribution BINDING are covered. Bus BALANCE — that the
// per-table contributions sum to the expected value across the epoch — is NOT,
// and cannot be until the batched verifier grows the constraint half; see
// `batched/verifier.rs`'s header for why that is blocked and on what.
//
// Every negative below has an honest-path control beside it. Without one, a
// rejection proves only that the checker rejects, not that it discriminates.
mod epoch {
    use super::*;
    use crate::batched::verifier::{replay_epoch_transcript, verify_epoch_commitments};
    use crate::config::DefaultStarkHash;
    use crate::residency_mode::ResidencyMode;
    use crate::tests::batched_prover_tests::{Air, E, F, folding_options, prove_repeated};
    use crate::traits::AIR;
    use crypto::fiat_shamir::default_transcript::DefaultTranscript;
    use math::field::element::FieldElement;

    type Proof = crate::batched::proof::BatchedMultiProof<F, E, ()>;

    fn air_refs(airs: &[Air]) -> Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>> {
        airs.iter()
            .map(|a| a as &dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>)
            .collect()
    }

    /// Replay `proof` and run the commitment checks. `None` when the replay
    /// itself rejects, so a test can tell "rejected structurally" from
    /// "rejected on the openings".
    fn replay_and_check(airs: &[Air], proof: &Proof) -> Option<bool> {
        let refs = air_refs(airs);
        let (shape, params, challenges) =
            replay_epoch_transcript(&refs, proof, &mut DefaultTranscript::<E>::new(&[]))?;
        Some(verify_epoch_commitments::<F, E, (), DefaultStarkHash>(
            &refs,
            proof,
            &shape,
            &params,
            &challenges,
        ))
    }

    fn honest() -> (Vec<Air>, Proof) {
        let (airs, proof, _, _) = prove_repeated(1, &folding_options(), ResidencyMode::Retain);
        (airs, proof)
    }

    /// ★ The strongest oracle available for "the prover and the verifier are one
    /// protocol": not that some challenge agrees, but that the two transcripts
    /// END in the same state. A divergence anywhere in the sequence — a root
    /// absorbed in the wrong order, a challenge sampled that the other side does
    /// not sample, an OOD block walked differently — lands here, where comparing
    /// individual challenges would only catch it if you happened to compare the
    /// right one.
    #[test_log::test]
    fn replay_matches_the_provers_ending_state() {
        let mut prover_transcript = DefaultTranscript::<E>::new(&[]);
        let (airs, proof, _, _) = crate::tests::batched_prover_tests::prove_repeated_with(
            1,
            &folding_options(),
            ResidencyMode::Retain,
            &mut prover_transcript,
        );

        let mut verifier_transcript = DefaultTranscript::<E>::new(&[]);
        let refs = air_refs(&airs);
        replay_epoch_transcript(&refs, &proof, &mut verifier_transcript)
            .expect("an honest epoch must replay");

        assert_eq!(
            prover_transcript.state(),
            verifier_transcript.state(),
            "prover and verifier must end the epoch in the same transcript state"
        );
    }

    /// The honest-path control every negative below leans on.
    #[test_log::test]
    fn an_honest_epoch_passes_the_commitment_checks() {
        let (airs, proof) = honest();
        assert_eq!(
            replay_and_check(&airs, &proof),
            Some(true),
            "an honest epoch must replay and authenticate"
        );
    }

    /// Query count. Nothing the transcript has already checked implies it: a
    /// prover that sent fewer openings would simply be checked less often.
    #[test_log::test]
    fn a_short_query_list_is_rejected() {
        let (airs, mut proof) = honest();
        assert!(
            proof.queries.len() > 1,
            "the fixture must have queries to drop"
        );
        proof.queries.pop();
        assert_eq!(
            replay_and_check(&airs, &proof),
            Some(false),
            "dropping a query must be rejected"
        );
    }

    /// Grinding. The nonce is absorbed, so a forged one moves every later
    /// challenge AND fails its own proof-of-work check; either rejection is
    /// correct and the test asserts the outcome, not the route.
    #[test_log::test]
    fn a_forged_grinding_nonce_is_rejected() {
        let (airs, mut proof) = honest();
        let nonce = proof.nonce.expect("the fixture grinds");
        proof.nonce = Some(nonce.wrapping_add(1));
        assert_ne!(
            replay_and_check(&airs, &proof),
            Some(true),
            "a nonce the prover did not grind must be rejected"
        );
    }

    /// A missing nonce where the epoch's grinding factor demands one.
    #[test_log::test]
    fn an_absent_grinding_nonce_is_rejected() {
        let (airs, mut proof) = honest();
        proof.nonce = None;
        assert_ne!(
            replay_and_check(&airs, &proof),
            Some(true),
            "an epoch with a positive grinding factor must carry a nonce"
        );
    }

    /// OOD values. They are absorbed before alpha, so tampering one must move
    /// the query indices — which is what makes the openings, honestly produced
    /// at the honest indices, stop authenticating.
    #[test_log::test]
    fn a_tampered_ood_value_is_rejected() {
        let (airs, honest_proof) = honest();
        let refs = air_refs(&airs);
        let honest_iotas =
            replay_epoch_transcript(&refs, &honest_proof, &mut DefaultTranscript::<E>::new(&[]))
                .expect("honest replay")
                .2
                .fri
                .iotas;

        let mut proof = honest_proof.clone();
        proof.tables[0].composition_poly_parts_ood_evaluation[0] += FieldElement::<E>::one();
        let (_, _, tampered) =
            replay_epoch_transcript(&refs, &proof, &mut DefaultTranscript::<E>::new(&[]))
                .expect("the shape is still structurally consistent");
        assert_ne!(
            tampered.fri.iotas, honest_iotas,
            "an OOD value the prover did not commit must move the query indices"
        );
        assert_eq!(
            replay_and_check(&airs, &proof),
            Some(false),
            "and the openings must then fail to authenticate"
        );
    }

    /// A trace OOD value, tampered in the other block, must behave the same —
    /// the two blocks are absorbed separately and a control on only one would
    /// miss a verifier that walked just that one.
    #[test_log::test]
    fn a_tampered_trace_ood_value_is_rejected() {
        let (airs, mut proof) = honest();
        let table = &mut proof.tables[0];
        let value = *table.trace_ood_evaluations.get(0, 0);
        table
            .trace_ood_evaluations
            .set(0, 0, value + FieldElement::<E>::one());
        assert_eq!(
            replay_and_check(&airs, &proof),
            Some(false),
            "a trace OOD value the prover did not commit must be rejected"
        );
    }

    /// Bus-contribution BINDING (not balance): which tables carry one is a fact
    /// about the AIR set, so dropping one must be a structural rejection rather
    /// than a transcript that quietly absorbs one element fewer.
    #[test_log::test]
    fn a_dropped_bus_contribution_is_rejected() {
        let (airs, mut proof) = honest();
        assert!(
            proof.tables[0].bus_public_inputs.is_some(),
            "the fixture's tables all have a RAP"
        );
        proof.tables[0].bus_public_inputs = None;
        assert_eq!(
            replay_and_check(&airs, &proof),
            None,
            "a table whose AIR has a RAP must carry a bus contribution"
        );
    }

    /// A tampered bus contribution is absorbed, so it moves the challenges.
    #[test_log::test]
    fn a_tampered_bus_contribution_is_rejected() {
        let (airs, mut proof) = honest();
        if let Some(bpi) = proof.tables[0].bus_public_inputs.as_mut() {
            bpi.table_contribution += FieldElement::<E>::one();
        }
        assert_eq!(
            replay_and_check(&airs, &proof),
            Some(false),
            "a bus contribution the prover did not commit must be rejected"
        );
    }

    /// A whole round's root, dropped. The AIR set says the aux round exists, so
    /// its absence is a structural rejection — without this a prover could
    /// remove a round's binding entirely.
    #[test_log::test]
    fn a_dropped_round_root_is_rejected() {
        let (airs, mut proof) = honest();
        proof.aux_root = None;
        assert_eq!(
            replay_and_check(&airs, &proof),
            None,
            "the aux round exists in this epoch, so its root cannot be absent"
        );
    }

    /// An opening the epoch does not have. This fixture has no preprocessed
    /// table, so a query carrying a preprocessed opening must reject: the
    /// count is bound by the AIR set, never by what the proof sends.
    #[test_log::test]
    fn an_invented_prep_opening_is_rejected() {
        let (airs, mut proof) = honest();
        assert!(
            proof.queries[0].prep.is_empty(),
            "the fixture has no preprocessed table"
        );
        proof.queries[0].prep.push(crate::proof::stark::PolynomialOpenings {
            proof: crypto::merkle_tree::proof::Proof { merkle_path: Vec::new() },
            evaluations: Vec::new(),
            evaluations_sym: Vec::new(),
        });
        assert_eq!(
            replay_and_check(&airs, &proof),
            Some(false),
            "a query carrying a preprocessed opening the AIR set does not declare \
             must be rejected"
        );
    }

    /// The instance-class partition is derived from the shape and never sent, so
    /// a terminal polynomial present for a batched table — or of the wrong
    /// length for a standalone one — must be rejected.
    #[test_log::test]
    fn a_misplaced_standalone_terminal_polynomial_is_rejected() {
        let (airs, honest_proof) = honest();

        let mut invented = honest_proof.clone();
        let batched_table = invented
            .tables
            .iter()
            .position(|t| t.standalone_final_poly_coeffs.is_none())
            .expect("the tallest table is always batched");
        invented.tables[batched_table].standalone_final_poly_coeffs =
            Some(vec![FieldElement::<E>::one(); 2]);
        assert_eq!(
            replay_and_check(&airs, &invented),
            Some(false),
            "a batched table must not carry a terminal-only polynomial"
        );

        if let Some(standalone_table) = honest_proof
            .tables
            .iter()
            .position(|t| t.standalone_final_poly_coeffs.is_some())
        {
            let mut truncated = honest_proof.clone();
            let coeffs = truncated.tables[standalone_table]
                .standalone_final_poly_coeffs
                .as_mut()
                .expect("just checked");
            coeffs.pop();
            assert_eq!(
                replay_and_check(&airs, &truncated),
                Some(false),
                "a standalone terminal polynomial of the wrong degree bound must be rejected"
            );
        }
    }

    /// The width the openings are authenticated under is the verifier's, and a
    /// table whose declared trace length disagrees with the epoch it was proved
    /// for moves the whole shape — heights, histogram, every challenge.
    #[test_log::test]
    fn a_tampered_trace_length_is_rejected() {
        let (airs, mut proof) = honest();
        proof.tables[1].trace_length *= 2;
        assert_ne!(
            replay_and_check(&airs, &proof),
            Some(true),
            "a trace length the prover did not commit must be rejected"
        );
    }
}

// ===========================================================================
// The DEEP / FRI join (M-5 core)
// ===========================================================================
//
// These are the tests that give the authenticated openings meaning. Everything
// in `epoch` above shows the proof opened the rows its roots bind; these show
// those rows evaluate to a codeword the batched FRI folds to the terminal
// polynomial it sent.
//
// The honest path is unusually load-bearing here: it can only pass if the DEEP
// reconstruction, the alpha mixing in `plan.batched` order, the per-table index
// reduction, the injection convention (value chosen from the opened row pair by
// `injection_position`'s low bit) and the coset relabelling are ALL right at
// once. Any one of them wrong and the terminal check fails.
mod fri_join {
    use crate::batched::verifier::{replay_epoch_transcript, verify_epoch_fri};
    use crate::config::DefaultStarkHash;
    use crate::residency_mode::ResidencyMode;
    use crate::tests::batched_prover_tests::{Air, E, F, folding_options, prove_repeated};
    use crate::traits::AIR;
    use crate::verifier::GenericVerifier;
    use crypto::fiat_shamir::default_transcript::DefaultTranscript;
    use math::field::element::FieldElement;

    type Proof = crate::batched::proof::BatchedMultiProof<F, E, ()>;
    type V = GenericVerifier<F, E, (), DefaultStarkHash>;

    fn join_holds(airs: &[Air], proof: &Proof) -> Option<bool> {
        let refs: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>> = airs
            .iter()
            .map(|a| a as &dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>)
            .collect();
        let (shape, params, challenges) =
            replay_epoch_transcript(&refs, proof, &mut DefaultTranscript::<E>::new(&[]))?;
        Some(verify_epoch_fri::<F, E, (), DefaultStarkHash, V>(
            &refs,
            proof,
            &shape,
            &params,
            &challenges,
        ))
    }

    fn honest() -> (Vec<Air>, Proof) {
        let (airs, proof, _, _) = prove_repeated(1, &folding_options(), ResidencyMode::Retain);
        (airs, proof)
    }

    /// ★ The honest path — and passing it is the joint statement listed above.
    #[test_log::test]
    fn the_batched_fri_join_verifies_an_honest_epoch() {
        let (airs, proof) = honest();
        assert_eq!(
            join_holds(&airs, &proof),
            Some(true),
            "an honest epoch's opened rows must fold to the terminal polynomial it sent"
        );
    }

    /// It must also hold at the degenerate shape, where every table terminates
    /// immediately and the batched instance folds nothing — the branch
    /// `verify_batched_fri_query` handles with `total_folds == 0` and the one
    /// that puts every other table in the standalone class.
    #[test_log::test]
    fn the_join_verifies_a_no_fold_epoch() {
        let (airs, proof, _, _) = prove_repeated(
            1,
            &crate::proof::options::ProofOptions::default_test_options(),
            ResidencyMode::Retain,
        );
        assert_eq!(
            join_holds(&airs, &proof),
            Some(true),
            "an epoch whose tables all terminate immediately must still verify"
        );
    }

    /// The FRI layer openings are NOT absorbed into the transcript — the
    /// structural length check and this recursion are the only things pinning
    /// them. Tampering one leaves every Merkle root and every challenge intact,
    /// so it is caught here or nowhere.
    #[test_log::test]
    fn a_tampered_fri_layer_evaluation_breaks_the_join() {
        let (airs, honest_proof) = honest();
        assert_eq!(
            join_holds(&airs, &honest_proof),
            Some(true),
            "honest-path control"
        );

        let layers = honest_proof.queries[0].fri.layers_evaluations_sym.len();
        assert!(layers > 0, "the folding fixture must commit a layer");
        let mut proof = honest_proof.clone();
        proof.queries[0].fri.layers_evaluations_sym[0] += FieldElement::<E>::one();
        assert_eq!(
            join_holds(&airs, &proof),
            Some(false),
            "a FRI layer value the prover did not commit must break the fold"
        );
    }

    /// A truncated decommitment would make the fold loop run fewer rounds and
    /// accept the query without ever reaching the terminal. The length check
    /// runs before the loop for exactly that reason.
    #[test_log::test]
    fn a_truncated_fri_decommitment_is_rejected() {
        let (airs, mut proof) = honest();
        proof.queries[0].fri.layers_evaluations_sym.pop();
        assert_eq!(
            join_holds(&airs, &proof),
            Some(false),
            "a short decommitment must be rejected, not folded fewer times"
        );
    }

    /// An opened trace row that the MMCS would accept only if the roots moved
    /// with it: tampering one must break the DEEP value it feeds, so the join
    /// fails independently of the Merkle check.
    #[test_log::test]
    fn a_tampered_opened_row_breaks_the_join() {
        let (airs, mut proof) = honest();
        proof.queries[0].main.per_matrix[0].evaluations[0] += FieldElement::<F>::one();
        assert_eq!(
            join_holds(&airs, &proof),
            Some(false),
            "a tampered trace row must change the DEEP value and fail the fold"
        );
    }

    /// A standalone table's terminal polynomial is checked by the OTHER class's
    /// routine, so it needs its own control: a tampered coefficient must be
    /// caught even though the batched instance is untouched.
    #[test_log::test]
    fn a_tampered_standalone_terminal_polynomial_breaks_the_join() {
        let (airs, honest_proof) = honest();
        let Some(table) = honest_proof
            .tables
            .iter()
            .position(|t| t.standalone_final_poly_coeffs.is_some())
        else {
            // The fixture's shape put every table in the batched class; nothing
            // to test, and saying so beats a silently vacuous pass.
            return;
        };
        let mut proof = honest_proof.clone();
        proof.tables[table]
            .standalone_final_poly_coeffs
            .as_mut()
            .expect("just checked")[0] += FieldElement::<E>::one();
        assert_ne!(
            join_holds(&airs, &proof),
            Some(true),
            "a standalone terminal polynomial the prover did not commit must be rejected"
        );
    }
}

// ===========================================================================
// `multi_verify_batched` — the complete verification
// ===========================================================================
mod full_verify {
    use crate::batched::verifier::multi_verify_batched;
    use crate::config::DefaultStarkHash;
    use crate::residency_mode::ResidencyMode;
    use crate::tests::batched_prover_tests::{Air, E, F, folding_options, prove_repeated};
    use crate::traits::AIR;
    use crate::verifier::GenericVerifier;
    use crypto::fiat_shamir::default_transcript::DefaultTranscript;
    use math::field::element::FieldElement;

    type Proof = crate::batched::proof::BatchedMultiProof<F, E, ()>;
    type V = GenericVerifier<F, E, (), DefaultStarkHash>;

    fn verifies(airs: &[Air], proof: &Proof) -> bool {
        let refs: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>> = airs
            .iter()
            .map(|a| a as &dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>)
            .collect();
        multi_verify_batched::<F, E, (), DefaultStarkHash, V, _>(
            &refs,
            proof,
            &mut DefaultTranscript::<E>::new(&[]),
            &FieldElement::zero(),
        )
    }

    fn honest() -> (Vec<Air>, Proof) {
        let (airs, proof, _, _) = prove_repeated(1, &folding_options(), ResidencyMode::Retain);
        (airs, proof)
    }

    /// ★★ Completeness: a proof this repository's batched prover produced is
    /// accepted by this repository's batched verifier, end to end — replay,
    /// commitments, constraint identity at every `z`, bus balance, and the
    /// DEEP/FRI join across both instance classes.
    #[test_log::test]
    fn an_honest_batched_epoch_verifies_end_to_end() {
        let (airs, proof) = honest();
        assert!(
            verifies(&airs, &proof),
            "an honest batched epoch must verify"
        );
    }

    /// The same at the degenerate shape, where nothing folds.
    #[test_log::test]
    fn an_honest_no_fold_epoch_verifies_end_to_end() {
        let (airs, proof, _, _) = prove_repeated(
            1,
            &crate::proof::options::ProofOptions::default_test_options(),
            ResidencyMode::Retain,
        );
        assert!(
            verifies(&airs, &proof),
            "an epoch whose tables all terminate immediately must verify"
        );
    }

    /// Residency is a performance choice, so it must be invisible to a verifier
    /// as well as to the roots.
    #[test_log::test]
    fn both_residency_modes_produce_verifying_proofs() {
        for mode in [ResidencyMode::Retain, ResidencyMode::RecomputeLde] {
            let (airs, proof, _, _) = prove_repeated(1, &folding_options(), mode);
            assert!(
                verifies(&airs, &proof),
                "{mode:?} must produce a valid proof"
            );
        }
    }

    /// The bus balance is the one cross-table statement, and no per-table check
    /// can make it. Verifying against a balance the epoch does not have must
    /// fail — with the honest expectation as the control.
    #[test_log::test]
    fn a_wrong_expected_bus_balance_is_rejected() {
        let (airs, proof) = honest();
        let refs: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>> = airs
            .iter()
            .map(|a| a as &dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>)
            .collect();
        assert!(
            multi_verify_batched::<F, E, (), DefaultStarkHash, V, _>(
                &refs,
                &proof,
                &mut DefaultTranscript::<E>::new(&[]),
                &FieldElement::zero(),
            ),
            "honest-path control: the epoch balances at zero"
        );
        assert!(
            !multi_verify_batched::<F, E, (), DefaultStarkHash, V, _>(
                &refs,
                &proof,
                &mut DefaultTranscript::<E>::new(&[]),
                &FieldElement::one(),
            ),
            "an expected balance the epoch does not have must be rejected"
        );
    }

    /// The constraint identity. A composition-parts OOD value the trace does not
    /// justify must fail — this is the check whose absence would let the batched
    /// path accept a proof of a false statement while every root and every
    /// opening stayed consistent.
    #[test_log::test]
    fn a_claimed_composition_value_the_trace_does_not_justify_is_rejected() {
        let (airs, mut proof) = honest();
        proof.tables[0].composition_poly_parts_ood_evaluation[0] += FieldElement::<E>::one();
        assert!(
            !verifies(&airs, &proof),
            "a composition OOD value the trace does not justify must be rejected"
        );
    }

    /// Every negative already covered piecewise must also be rejected by the
    /// whole verifier — a check that exists but is never reached is not a check.
    #[test_log::test]
    fn the_whole_verifier_rejects_what_the_pieces_reject() {
        let (airs, honest_proof) = honest();
        assert!(verifies(&airs, &honest_proof), "honest-path control");

        let mut short_queries = honest_proof.clone();
        short_queries.queries.pop();
        assert!(!verifies(&airs, &short_queries), "short query list");

        let mut bad_nonce = honest_proof.clone();
        bad_nonce.nonce = Some(bad_nonce.nonce.expect("the fixture grinds").wrapping_add(1));
        assert!(!verifies(&airs, &bad_nonce), "forged grinding nonce");

        let mut bad_row = honest_proof.clone();
        bad_row.queries[0].main.per_matrix[0].evaluations[0] += FieldElement::<F>::one();
        assert!(!verifies(&airs, &bad_row), "tampered opened row");

        let mut bad_layer = honest_proof.clone();
        bad_layer.queries[0].fri.layers_evaluations_sym[0] += FieldElement::<E>::one();
        assert!(!verifies(&airs, &bad_layer), "tampered FRI layer value");

        let mut no_aux_root = honest_proof.clone();
        no_aux_root.aux_root = None;
        assert!(!verifies(&airs, &no_aux_root), "dropped aux root");
    }
}

// ===========================================================================
// The preprocessed binding, end to end
// ===========================================================================
//
// Preprocessed tables are bound PER TABLE: each root is
// `air.precomputed_commitment()`, absorbed by both sides from the AIR set and
// authenticated per query at the reduced per-table index — the per-table
// path's critical soundness check, unchanged in kind. There is no pinned
// fused round and no caller-side pin: the old `PinnedPrep` width tests have
// no analogue because widths come from the AIR set on both sides, and the
// fail-closed `None` arm has no analogue because there is nothing to omit.
// What this module still owes §3.3 is the per-matrix quantifier through the
// WHOLE verifier, and the wrong-root rejection — both kept below.
mod prep_binding {
    use crate::batched::verifier::multi_verify_batched;
    use crate::config::DefaultStarkHash;
    use crate::tests::batched_prover_tests::{Air, E, F, PREP_WIDTHS, prove_preprocessed};
    use crate::traits::AIR;
    use crate::verifier::GenericVerifier;
    use crypto::fiat_shamir::default_transcript::DefaultTranscript;
    use math::field::element::FieldElement;

    type Proof = crate::batched::proof::BatchedMultiProof<F, E, ()>;
    type V = GenericVerifier<F, E, (), DefaultStarkHash>;

    fn verifies(airs: &[Air], proof: &Proof) -> bool {
        let refs: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>> = airs
            .iter()
            .map(|a| a as &dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>)
            .collect();
        multi_verify_batched::<F, E, (), DefaultStarkHash, V, _>(
            &refs,
            proof,
            &mut DefaultTranscript::<E>::new(&[]),
            &FieldElement::zero(),
        )
    }

    fn honest() -> (Vec<Air>, Proof) {
        let (airs, proof, _) = prove_preprocessed().expect("an honest preprocessed epoch");
        (airs, proof)
    }

    /// ★★ The honest path. A preprocessed epoch verifies end to end against
    /// the AIR set's own pinned roots — every other test in this module is a
    /// rejection, and without this one they would all be satisfied by a
    /// verifier that rejected everything.
    #[test_log::test]
    fn an_honest_preprocessed_epoch_verifies_end_to_end() {
        let (airs, proof) = honest();
        assert!(
            verifies(&airs, &proof),
            "an honest preprocessed epoch must verify against the AIR set's roots"
        );
    }

    /// ★ The check the per-table binding exists for. A verifier whose AIR set
    /// pins a DIFFERENT preprocessed root must reject the proof: the roots are
    /// the verifier's own, so a prover cannot substitute preprocessed content
    /// and stay self-consistent.
    #[test_log::test]
    fn a_prep_root_the_program_does_not_pin_is_rejected() {
        use crate::examples::multi_table_lookup::{
            new_add_air_with_lookup, new_cpu_air_with_lookup, new_mul_air_with_lookup,
        };
        use crate::tests::batched_prover_tests::folding_options;

        let (airs, proof) = honest();
        let options = folding_options();
        let mut wrong_root = airs[1].precomputed_commitment();
        wrong_root[0] ^= 0xff;
        let wrong_airs = vec![
            new_cpu_air_with_lookup(&options),
            new_add_air_with_lookup(&options).with_preprocessed(wrong_root, PREP_WIDTHS[0]),
            new_mul_air_with_lookup(&options)
                .with_preprocessed(airs[2].precomputed_commitment(), PREP_WIDTHS[1]),
        ];
        assert!(
            !verifies(&wrong_airs, &proof),
            "a proof whose preprocessed content is not the verifier's pinned one \
             must be rejected"
        );
    }

    /// ★ The per-matrix quantifier, reached through the WHOLE verifier rather
    /// than through the opening check alone. §3.3's requirement survives the
    /// per-table layout: the verification must fail if ANY one table's
    /// preprocessed value is wrong.
    #[test_log::test]
    fn a_tampered_prep_matrix_is_rejected_per_matrix_end_to_end() {
        let (airs, honest_proof) = honest();
        assert!(verifies(&airs, &honest_proof), "honest-path control");

        let matrices = honest_proof.queries[0].prep.len();
        assert_eq!(
            matrices,
            PREP_WIDTHS.len(),
            "the fixture must contribute one opening per preprocessed table"
        );

        for matrix in 0..matrices {
            let mut tampered = honest_proof.clone();
            tampered.queries[0].prep[matrix].evaluations[0] += FieldElement::<F>::one();
            assert!(
                !verifies(&airs, &tampered),
                "prep table {matrix}: a tampered precomputed value must be rejected \
                 by the whole verifier"
            );
        }
    }
}
