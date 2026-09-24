//! S2 (one-row trace openings with a committed FRI input) on the CPU prover
//! and host verifier: design/FRI.md §7 and §10 — U6 at one_row, the tamper
//! tests T4–T6, the load-bearing mutation M3, the transcript-order KAT, the
//! per-table `auto` rule (RULINGS 6, REVIEW-FRI F5), the preprocessed-root
//! miss (RULINGS 14) and the cap × FRI × one-row matrix (REVIEW-FRI F9).

use std::sync::Mutex;

use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use crypto::fiat_shamir::is_transcript::IsTranscript;
use crypto::merkle_tree::cap::CapPolicy;
use math::field::element::FieldElement;
use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField;
use math::field::goldilocks::GoldilocksField;
use math::field::traits::IsFFTField;

use crate::config::{Blake3StarkHash, KeccakStarkHash};
use crate::examples::fibonacci_2_columns::compute_trace;
use crate::examples::simple_fibonacci::FibonacciPublicInputs;
use crate::fri::capture::{FriCapture, capture};
use crate::fri::fri_functions::compute_coset_twiddles_inv;
use crate::fri::terminal::FriFoldLayout;
use crate::fri::{commit_phase_with_layout, fold_times};
use crate::leaf_layout::{
    LeafLayout, M3_PAIR_BOUND_UNDER_ONE_ROW, TableWidths, resolve_leaf_layout, table_leaf_layout,
    table_openings_cost_q,
};
use crate::proof::options::{FriMode, FriScheduleOverride, OneRowMode, ProofFormat, ProofOptions};
use crate::proof::stark::MultiProof;
use crate::prover::{IsStarkProver, Prover, ProvingError};
use crate::tests::opening_width_tests::FibonacciSplitAIR;
use crate::traits::AIR;
use crate::verifier::{IsStarkVerifier, Verifier};

use super::zf_golden_tests::{
    golden_options, prove_logup, prove_multi, prove_simple_addition, verify_logup, verify_multi,
    verify_simple_addition,
};

type F = GoldilocksField;
type E = Degree3GoldilocksExtensionField;
type Felt = FieldElement<F>;
type Ext = FieldElement<E>;

/// Serialises the tests that flip the process-global M3 switch (see
/// `leaf_layout::M3_PAIR_BOUND_UNDER_ONE_ROW`).
static M3_LOCK: Mutex<()> = Mutex::new(());

fn fmt(one_row: OneRowMode, fri_mode: FriMode, schedule: Option<&[u8]>) -> ProofFormat {
    ProofFormat {
        one_row,
        fri_mode,
        fri_schedule_override: schedule.map(|s| FriScheduleOverride::new(s).unwrap()),
        ..ProofFormat::DEFAULT
    }
}

fn on(fri_mode: FriMode) -> ProofFormat {
    fmt(OneRowMode::On, fri_mode, None)
}

// ---------------------------------------------------------------------------
// The layout helper (REVIEW-FRI F7): one place a query becomes rows.
// ---------------------------------------------------------------------------

#[test]
fn query_rows_bounds_and_depths() {
    use math::fft::bit_reversing::reverse_index;
    for lde_log in 1..=12u32 {
        let n = 1usize << lde_log;
        assert_eq!(LeafLayout::RowPair.query_bound(n as u64), (n / 2) as u64);
        assert_eq!(LeafLayout::Row.query_bound(n as u64), n as u64);
        assert_eq!(
            LeafLayout::RowPair.tree_depth(lde_log as usize),
            lde_log as usize - 1
        );
        assert_eq!(
            LeafLayout::Row.tree_depth(lde_log as usize),
            lde_log as usize
        );
        for q in 0..n / 2 {
            assert_eq!(
                LeafLayout::RowPair.query_rows(q, n),
                (
                    reverse_index(2 * q, n as u64),
                    Some(reverse_index(2 * q + 1, n as u64))
                )
            );
        }
        for r in 0..n {
            assert_eq!(
                LeafLayout::Row.query_rows(r, n),
                (reverse_index(r, n as u64), None)
            );
        }
    }
}

/// REVIEW-FRI F7: no stray `2·iota(+1)` row arithmetic outside the helper in
/// the opening code of the prover and the verifier (the legacy FRI
/// zero-fold terminal check, which indexes the TERMINAL codeword by the pair,
/// is the one named exception).
#[test]
fn every_opening_site_goes_through_query_rows() {
    let prover = include_str!("../prover.rs");
    let verifier = include_str!("../verifier.rs");
    for (name, src) in [("prover.rs", prover), ("verifier.rs", verifier)] {
        for (i, line) in src.lines().enumerate() {
            let code = line.split("//").next().unwrap_or("");
            let pairish = code.contains("* 2 + 1") || code.contains("*2+1");
            let terminal = code.contains(".get(iota * 2 + 1)");
            let point_helper = code.contains("let raw = iota * 2");
            assert!(
                !pairish || terminal || point_helper,
                "{name}:{}: row-pair arithmetic outside LeafLayout::query_rows: {line}",
                i + 1
            );
        }
    }
}

// ---------------------------------------------------------------------------
// U6: round trips at one_row, every fold count, pair and dp FRI.
// ---------------------------------------------------------------------------

fn check_shape_simple(
    proof: &crate::proof::stark::StarkProof<
        F,
        F,
        crate::examples::simple_addition::SimpleAdditionPublicInputs<F>,
    >,
    lde_log: u32,
    layout: &FriFoldLayout,
) {
    assert_eq!(proof.fri_layers_merkle_roots.len(), layout.num_committed);
    for (q, dec) in proof.query_list.iter().zip(&proof.deep_poly_openings) {
        assert!(dec.main_trace_polys.evaluations_sym.is_empty());
        assert!(dec.composition_poly.evaluations_sym.is_empty());
        assert_eq!(
            dec.main_trace_polys.proof.merkle_path.len(),
            lde_log as usize
        );
        assert_eq!(
            dec.composition_poly.proof.merkle_path.len(),
            lde_log as usize
        );
        assert_eq!(
            q.layers_evaluations_sym.len(),
            layout.opened_values_per_query()
        );
    }
}

#[test]
fn one_row_round_trips_at_every_fold_count() {
    // k = 1: total_folds = log2(rows) + blowup_log − (blowup_log + 1).
    for blowup in [2u8, 4] {
        for log_rows in 1..=10u32 {
            for mode in [FriMode::Pair, FriMode::Dp] {
                let rows = 1usize << log_rows;
                let o = golden_options(blowup, 1, 9, on(mode));
                let (air, proof) = prove_simple_addition::<KeccakStarkHash>(rows, &o);
                assert!(
                    verify_simple_addition::<KeccakStarkHash>(&air, &proof),
                    "rows {rows} blowup {blowup} {mode:?}"
                );
                let lde_log = log_rows + blowup.trailing_zeros();
                let l =
                    FriFoldLayout::for_options(lde_log, blowup.trailing_zeros(), &o, true).unwrap();
                check_shape_simple(&proof, lde_log, &l);
                if l.total_folds > 0 {
                    // The input tree is layer 0: one more committed layer
                    // than the row-pair chain has under the pair schedule.
                    assert_eq!(
                        l.schedule.iter().map(|&d| u32::from(d)).sum::<u32>(),
                        l.total_folds
                    );
                }
            }
        }
    }
}

#[test]
fn one_row_round_trips_under_explicit_schedules() {
    // rows 2^9, blowup 4, k 1: lde_log 11, chain from 11 to T = 3: 8 bits.
    for sched in [
        &[1u8, 3, 4][..],
        &[3, 1, 3, 1],
        &[2, 1, 2, 2, 1],
        &[1, 1, 1, 1, 1, 1, 1, 1],
        &[6, 2],
        &[1, 6, 1],
        &[4, 4],
    ] {
        let o = golden_options(4, 1, 9, fmt(OneRowMode::On, FriMode::Dp, Some(sched)));
        let (air, proof) = prove_simple_addition::<Blake3StarkHash>(512, &o);
        assert!(
            verify_simple_addition::<Blake3StarkHash>(&air, &proof),
            "{sched:?}"
        );
        assert_eq!(proof.fri_layers_merkle_roots.len(), sched.len());
        assert_eq!(
            proof.query_list[0].layers_evaluations_sym.len(),
            sched.iter().map(|&d| 1usize << d).sum::<usize>()
        );
    }
    // An override that fits the row-pair chain (7 bits) but not the one-row
    // chain (8 bits) is a proving error under one row, never a fallback.
    let o = golden_options(4, 1, 9, fmt(OneRowMode::On, FriMode::Dp, Some(&[3, 4])));
    let air = crate::examples::simple_addition::SimpleAdditionAIR::<F>::new(&o);
    let mut trace = crate::examples::simple_addition::simple_addition_trace::<F>(512);
    let pi = crate::examples::simple_addition::SimpleAdditionPublicInputs {
        a: Felt::from(1u64),
        b: Felt::from(2u64),
    };
    assert!(
        crate::prover::GenericProver::<F, F, _, KeccakStarkHash>::prove(
            &air,
            &mut trace,
            &pi,
            &mut DefaultTranscript::<F>::new(&[]),
        )
        .is_err()
    );
}

#[test]
fn one_row_round_trips_ext3_aux_and_multi_table() {
    for (rows, blowup) in [(4usize, 2u8), (16, 2), (128, 4), (512, 2)] {
        for format in [on(FriMode::Pair), on(FriMode::Dp)] {
            let o = golden_options(blowup, 1, 7, format);
            let (air, proof, _) = prove_logup::<Blake3StarkHash>(rows, &o);
            assert!(verify_logup::<Blake3StarkHash>(&air, &proof), "rows {rows}");
            let lde_log = rows.trailing_zeros() + blowup.trailing_zeros();
            for dec in &proof.deep_poly_openings {
                let aux = dec.aux_trace_polys.as_ref().expect("aux opening");
                assert!(aux.evaluations_sym.is_empty());
                assert_eq!(aux.proof.merkle_path.len(), lde_log as usize);
            }
            let (air, proof, _) = prove_logup::<KeccakStarkHash>(rows, &o);
            assert!(
                verify_logup::<KeccakStarkHash>(&air, &proof),
                "keccak rows {rows}"
            );
        }
    }
    for format in [
        on(FriMode::Pair),
        on(FriMode::Dp),
        fmt(OneRowMode::Auto, FriMode::Dp, None),
    ] {
        let o = golden_options(2, 1, 6, format);
        let multi = prove_multi::<Blake3StarkHash>(&o);
        assert!(verify_multi::<Blake3StarkHash>(&o, &multi), "{format:?}");
    }
}

/// The archived (rkyv, read-in-place) verifier path verifies a one-row proof
/// too: the proof structs did not change, only the encoding of their vectors.
#[test]
fn one_row_verifies_archived() {
    let o = golden_options(4, 2, 5, on(FriMode::Dp));
    let (air, proof) = prove_simple_addition::<KeccakStarkHash>(256, &o);
    let multi = MultiProof {
        proofs: vec![proof.clone()],
    };
    let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&multi).unwrap();
    type Pi = crate::examples::simple_addition::SimpleAdditionPublicInputs<F>;
    let archived = rkyv::access::<
        crate::proof::stark::ArchivedMultiProof<F, F, Pi>,
        rkyv::rancor::Error,
    >(&bytes)
    .unwrap();
    let airs: Vec<&dyn AIR<Field = F, FieldExtension = F, PublicInputs = Pi>> = vec![&air];
    assert!(
        crate::verifier::GenericVerifier::<F, F, Pi, KeccakStarkHash>::multi_verify_archived(
            &airs,
            archived,
            &mut DefaultTranscript::<F>::new(&[]),
            &Felt::zero(),
        )
    );
}

/// The layout is a verifier-side constant: a one-row proof does not verify
/// under row-pair options, nor a row-pair proof under one-row options.
#[test]
fn the_layout_is_a_verifier_constant() {
    let one = golden_options(4, 1, 9, on(FriMode::Pair));
    let pair = golden_options(4, 1, 9, ProofFormat::DEFAULT);
    let (one_air, one_proof) = prove_simple_addition::<KeccakStarkHash>(1024, &one);
    let (pair_air, pair_proof) = prove_simple_addition::<KeccakStarkHash>(1024, &pair);
    assert!(verify_simple_addition::<KeccakStarkHash>(
        &one_air, &one_proof
    ));
    assert!(verify_simple_addition::<KeccakStarkHash>(
        &pair_air,
        &pair_proof
    ));
    assert!(!verify_simple_addition::<KeccakStarkHash>(
        &pair_air, &one_proof
    ));
    assert!(!verify_simple_addition::<KeccakStarkHash>(
        &one_air,
        &pair_proof
    ));
}

// ---------------------------------------------------------------------------
// T4–T6: tamper tests on a one-row proof.
// ---------------------------------------------------------------------------

#[test]
fn tampering_a_one_row_proof_is_rejected() {
    let o = golden_options(4, 1, 5, fmt(OneRowMode::On, FriMode::Dp, Some(&[3, 2, 3])));
    let (air, honest, _) = prove_logup::<KeccakStarkHash>(512, &o);
    assert!(verify_logup::<KeccakStarkHash>(&air, &honest));
    let bump = Ext::new([Felt::one(), Felt::zero(), Felt::zero()]);
    let values = honest.query_list[0].layers_evaluations_sym.len();
    assert_eq!(values, 8 + 4 + 8);

    // T4: every value of query 0's input group (layer 0), the slot included —
    // the input-slot check `group₀[slot] == DEEP(x_r)` and the group hash.
    for i in 0..8 {
        let mut p = honest.clone();
        p.query_list[0].layers_evaluations_sym[i] += bump;
        assert!(
            !verify_logup::<KeccakStarkHash>(&air, &p),
            "input group value {i}"
        );
    }
    // The input tree's root and a sibling of its path.
    let mut p = honest.clone();
    p.fri_layers_merkle_roots[0][3] ^= 1;
    assert!(!verify_logup::<KeccakStarkHash>(&air, &p), "input root");
    let mut p = honest.clone();
    p.query_list[0].layers_auth_paths[0].merkle_path[0][0] ^= 1;
    assert!(!verify_logup::<KeccakStarkHash>(&air, &p), "input path");

    // T5: a non-empty `evaluations_sym` under one row, for each tree — even
    // one holding the honest value of the symmetric row.
    let mut p = honest.clone();
    p.deep_poly_openings[0].main_trace_polys.evaluations_sym =
        p.deep_poly_openings[0].main_trace_polys.evaluations.clone();
    assert!(!verify_logup::<KeccakStarkHash>(&air, &p), "main sym");
    let mut p = honest.clone();
    p.deep_poly_openings[0].composition_poly.evaluations_sym =
        p.deep_poly_openings[0].composition_poly.evaluations.clone();
    assert!(
        !verify_logup::<KeccakStarkHash>(&air, &p),
        "composition sym"
    );
    let mut p = honest.clone();
    let aux = p.deep_poly_openings[1].aux_trace_polys.as_mut().unwrap();
    aux.evaluations_sym = aux.evaluations.clone();
    assert!(!verify_logup::<KeccakStarkHash>(&air, &p), "aux sym");

    // T6: a trace value, an aux value and a composition value of one opening
    // (each moves DEEP(x_r) and the leaf hash).
    let mut p = honest.clone();
    p.deep_poly_openings[2].main_trace_polys.evaluations[0] += Felt::one();
    assert!(!verify_logup::<KeccakStarkHash>(&air, &p), "main value");
    let mut p = honest.clone();
    p.deep_poly_openings[2]
        .aux_trace_polys
        .as_mut()
        .unwrap()
        .evaluations[0] += bump;
    assert!(!verify_logup::<KeccakStarkHash>(&air, &p), "aux value");
    let mut p = honest.clone();
    p.deep_poly_openings[2].composition_poly.evaluations[0] += bump;
    assert!(
        !verify_logup::<KeccakStarkHash>(&air, &p),
        "composition value"
    );
    // A trace path one level short (the row-pair depth) and one long.
    let mut p = honest.clone();
    p.deep_poly_openings[0]
        .main_trace_polys
        .proof
        .merkle_path
        .pop();
    assert!(
        !verify_logup::<KeccakStarkHash>(&air, &p),
        "short trace path"
    );
    let mut p = honest.clone();
    p.deep_poly_openings[0]
        .main_trace_polys
        .proof
        .merkle_path
        .push([0u8; 32]);
    assert!(
        !verify_logup::<KeccakStarkHash>(&air, &p),
        "long trace path"
    );
    // The flat group vector one short / one long.
    let mut p = honest.clone();
    p.query_list[0].layers_evaluations_sym.pop();
    assert!(!verify_logup::<KeccakStarkHash>(&air, &p));
    let mut p = honest.clone();
    p.query_list[0].layers_evaluations_sym.push(Ext::zero());
    assert!(!verify_logup::<KeccakStarkHash>(&air, &p));
    // A missing input layer.
    let mut p = honest.clone();
    p.fri_layers_merkle_roots.remove(0);
    assert!(!verify_logup::<KeccakStarkHash>(&air, &p));
}

/// Zero folds under one row (`B ≤ T`): no layer, no challenge; the terminal
/// codeword IS the DEEP codeword and `terminal[r] == DEEP(x_r)` is the check.
#[test]
fn one_row_zero_fold_case() {
    // rows 4, blowup 2, k 2: T = min(1 + 2, 3) = 3 = lde_log → no fold.
    let o = golden_options(2, 2, 5, on(FriMode::Pair));
    let (air, proof) = prove_simple_addition::<KeccakStarkHash>(4, &o);
    assert!(verify_simple_addition::<KeccakStarkHash>(&air, &proof));
    assert!(proof.fri_layers_merkle_roots.is_empty());
    assert_eq!(proof.fri_final_poly_coeffs.len(), 4);
    let mut p = proof.clone();
    p.fri_final_poly_coeffs[1] += Felt::one();
    assert!(!verify_simple_addition::<KeccakStarkHash>(&air, &p));
    let mut p = proof.clone();
    p.deep_poly_openings[0].main_trace_polys.evaluations[1] += Felt::one();
    assert!(!verify_simple_addition::<KeccakStarkHash>(&air, &p));
}

// ---------------------------------------------------------------------------
// FRI.md §7.7 (i): r is uniform over ALL of D₀. M3 shows the test that says so
// is load-bearing.
// ---------------------------------------------------------------------------

/// The query indexes the verifier draws for a one-row SimpleAddition proof of
/// `rows` rows at blowup 2 with `queries` queries (and the proof verifies).
fn one_row_iotas(rows: usize, queries: usize) -> (Vec<usize>, bool) {
    let o = golden_options(2, 1, queries, on(FriMode::Pair));
    let (air, proof) = prove_simple_addition::<KeccakStarkHash>(rows, &o);
    let (ok, records) = capture(|| verify_simple_addition::<KeccakStarkHash>(&air, &proof));
    let rec = FriCapture::<F>::from_any(records[0].as_ref()).expect("one record");
    (rec.iotas.clone(), ok)
}

/// With 64 queries over an LDE of 64 points, all 64 indexes below `N / 2`
/// has probability 2⁻⁶⁴ under the right bound; the pair bound makes it
/// certain.
fn upper_half_reached(iotas: &[usize], lde: usize) -> bool {
    iotas.iter().any(|&r| r >= lde / 2) && iotas.iter().all(|&r| r < lde)
}

#[test]
fn one_row_query_indexes_cover_the_whole_lde() {
    let _g = M3_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let (iotas, ok) = one_row_iotas(32, 64);
    assert!(ok);
    assert!(upper_half_reached(&iotas, 64), "iotas {iotas:?}");
}

/// M3: sample r over N/2 under one row. Prover and verifier agree on the
/// mutated bound, so the proof still VERIFIES — the bias is invisible to
/// verification, and only the bound test catches it.
#[test]
fn m3_the_query_bound_test_is_load_bearing() {
    let _g = M3_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    M3_PAIR_BOUND_UNDER_ONE_ROW.store(true, std::sync::atomic::Ordering::SeqCst);
    let (iotas, ok) = one_row_iotas(32, 64);
    M3_PAIR_BOUND_UNDER_ONE_ROW.store(false, std::sync::atomic::Ordering::SeqCst);
    assert!(ok, "the mutated proof still verifies (both sides mutated)");
    assert!(
        !upper_half_reached(&iotas, 64),
        "under the mutation the bound test must fail"
    );
}

// ---------------------------------------------------------------------------
// FRI.md §7.7 (ii): the input root is absorbed before ζ₀ (transcript KAT).
// ---------------------------------------------------------------------------

#[test]
fn input_root_is_absorbed_before_the_first_challenge() {
    use crate::fri::group::roots_of_unity_table;
    let o = Felt::from(3u64);
    let lde_log = 10u32;
    let n = 1usize << lde_log;
    // A low-degree ext3 codeword (256 coefficients, blowup 4), bit-reversed.
    let coeffs: Vec<Ext> = (0..256u64)
        .map(|i| Ext::new([Felt::from(i + 1), Felt::from(3 * i), Felt::from(7)]))
        .collect();
    let poly = math::polynomial::Polynomial::new(&coeffs);
    let mut cw =
        math::polynomial::Polynomial::evaluate_offset_fft::<F>(&poly, 4, Some(256), &o).unwrap();
    math::fft::bit_reversing::in_place_bit_reverse_permute(&mut cw);
    // One row, schedule [2, 3, 3] from 10 to T = 2 + 0 = 2.
    let layout = FriFoldLayout::from_schedule(lde_log, 2, 0, true, vec![2, 3, 3]).unwrap();
    let tw = compute_coset_twiddles_inv::<F>(&o, n);
    let mut t = DefaultTranscript::<E>::new(&[9]);
    let (_coeffs, layers) = commit_phase_with_layout::<F, E, _, KeccakStarkHash>(
        cw.clone(),
        &mut t,
        &o,
        n,
        2,
        0,
        &layout,
        &tw,
    );
    assert_eq!(layers.len(), 3);
    assert_eq!(
        layers[0].evaluation, cw,
        "layer 0 is the DEEP codeword itself"
    );

    // The right order: root₀, then ζ₀. Folding layer 0 with that ζ₀ gives
    // exactly the committed layer 1.
    let mut right = DefaultTranscript::<E>::new(&[9]);
    right.append_bytes(&layers[0].merkle_tree.root);
    let zeta0 = right.sample_field_element();
    let mut folded = cw.clone();
    let mut tw2 = tw.clone();
    fold_times(&mut folded, &zeta0, 2, &mut tw2);
    assert_eq!(folded, layers[1].evaluation, "ζ₀ was drawn after root₀");

    // The wrong order (ζ₀ before root₀) gives another challenge and another
    // layer 1.
    let mut wrong = DefaultTranscript::<E>::new(&[9]);
    let zeta_wrong = wrong.sample_field_element();
    assert_ne!(zeta_wrong, zeta0);
    let mut folded = cw.clone();
    let mut tw2 = tw.clone();
    fold_times(&mut folded, &zeta_wrong, 2, &mut tw2);
    assert_ne!(folded, layers[1].evaluation);
    let _ = roots_of_unity_table::<F>(1);
}

// ---------------------------------------------------------------------------
// Preprocessed tables: one-row roots, and RULINGS 14 (a miss is an error).
// ---------------------------------------------------------------------------

#[test]
fn one_row_preprocessed_table_and_a_missing_root() {
    let opts = golden_options(2, 1, 5, on(FriMode::Pair));
    let mut trace = compute_trace([Felt::one(), Felt::one()], 256);
    let reference = FibonacciSplitAIR::<F>::honest(&opts, None);
    let pair_root = Prover::compute_precomputed_commitment_for_testing(&trace, &reference, 1)
        .expect("row-pair root");
    let row_root = Prover::compute_precomputed_commitment_for_testing_with(
        &trace,
        &reference,
        1,
        LeafLayout::Row,
    )
    .expect("one-row root");
    assert_ne!(
        pair_root, row_root,
        "the two layouts commit different bytes"
    );
    let pi = FibonacciPublicInputs {
        a0: Felt::one(),
        a1: Felt::one(),
    };

    // With both roots: proves and verifies, and the proof carries the ROW root.
    let air = FibonacciSplitAIR::<F>::preprocessed_declaring(&opts, None, 1, pair_root)
        .with_one_row_commitment(row_root);
    let proof =
        Prover::prove(&air, &mut trace, &pi, &mut DefaultTranscript::<F>::new(&[])).expect("prove");
    assert_eq!(proof.lde_trace_precomputed_merkle_root, Some(row_root));
    assert!(Verifier::verify(
        &proof,
        &air,
        &mut DefaultTranscript::<F>::new(&[])
    ));

    // The same AIR without a one-row root: the prover refuses with an Err
    // (no panic, no recompute) and the verifier rejects the honest proof.
    let bare = FibonacciSplitAIR::<F>::preprocessed_declaring(&opts, None, 1, pair_root);
    let mut trace2 = compute_trace([Felt::one(), Felt::one()], 256);
    match Prover::prove(
        &bare,
        &mut trace2,
        &pi,
        &mut DefaultTranscript::<F>::new(&[]),
    ) {
        Err(ProvingError::PrecomputedCommitmentMissing(_)) => {}
        other => panic!(
            "expected PrecomputedCommitmentMissing, got {:?}",
            other.map(|_| ())
        ),
    }
    assert!(!Verifier::verify(
        &proof,
        &bare,
        &mut DefaultTranscript::<F>::new(&[])
    ));

    // A wrong one-row root: the prover's rebuilt tree disagrees.
    let wrong = FibonacciSplitAIR::<F>::preprocessed_declaring(&opts, None, 1, pair_root)
        .with_one_row_commitment(pair_root);
    let mut trace3 = compute_trace([Felt::one(), Felt::one()], 256);
    assert!(matches!(
        Prover::prove(
            &wrong,
            &mut trace3,
            &pi,
            &mut DefaultTranscript::<F>::new(&[])
        ),
        Err(ProvingError::PrecomputedCommitmentMismatch)
    ));
}

// ---------------------------------------------------------------------------
// RULINGS 6 / REVIEW-FRI F5: the per-table `auto` rule.
// ---------------------------------------------------------------------------

fn opts_q(q: usize, one_row: OneRowMode, fri: FriMode, cap: CapPolicy) -> ProofOptions {
    let mut o = golden_options(4, 7, q, fmt(one_row, fri, None));
    o.format.merkle_cap = cap;
    o
}

/// The rule is the cost comparison, strictly: one row iff cheaper.
#[test]
fn auto_is_the_strict_cost_comparison() {
    for fri in [FriMode::Pair, FriMode::Dp] {
        for cap in [CapPolicy::Off, CapPolicy::Auto] {
            let o = opts_q(110, OneRowMode::Auto, fri, cap);
            for lde_log in 4..=24u32 {
                for main in [1u64, 4, 8, 30, 120, 400] {
                    for aux in [0u64, 3, 30, 120] {
                        let w = TableWidths {
                            precomputed: 0,
                            main,
                            aux,
                            composition: 6,
                        };
                        let row = table_openings_cost_q(&w, &o, lde_log, 2, true);
                        let pair = table_openings_cost_q(&w, &o, lde_log, 2, false);
                        assert_eq!(
                            resolve_leaf_layout(&w, &o, lde_log, 2),
                            LeafLayout::from_one_row(row < pair),
                            "fri {fri:?} cap {cap:?} B {lde_log} main {main} aux {aux}"
                        );
                    }
                }
            }
        }
    }
    // Off and On ignore the costs.
    let w = TableWidths {
        precomputed: 0,
        main: 1,
        aux: 0,
        composition: 3,
    };
    for lde_log in 4..=24 {
        let off = opts_q(110, OneRowMode::Off, FriMode::Pair, CapPolicy::Off);
        let on = opts_q(110, OneRowMode::On, FriMode::Pair, CapPolicy::Off);
        assert_eq!(
            resolve_leaf_layout(&w, &off, lde_log, 2),
            LeafLayout::RowPair
        );
        assert_eq!(resolve_leaf_layout(&w, &on, lde_log, 2), LeafLayout::Row);
    }
}

/// ⚠ A FORMAT PIN: `auto`'s choice for a set of production-like shapes (Q =
/// 110, blowup 4, k = 7, cap auto, fri dp). Wide tables go one-row, narrow
/// tall ones stay row pairs. Any change to the cost function or its weights
/// that moves one of these is a format change. The widths are illustrative
/// (MEMW 49 main / 13 aux as REVIEW-FRI §C reads them; the others are round
/// numbers), not a census: at generation the MEMW-like and CPU-like cases sat
/// within 0.5% and 2% of the threshold (row 49,988,402 vs pair 49,741,452;
/// row 51,474,062 vs pair 52,465,162, ×Q ns), so they pin the rule's edge.
#[test]
fn auto_choices_are_pinned() {
    let o = opts_q(110, OneRowMode::Auto, FriMode::Dp, CapPolicy::Auto);
    // (name, B, precomputed, main, aux ext columns, composition parts, one row?)
    let cases: &[(&str, u32, u64, u64, u64, u64, bool)] = &[
        ("wide keccak-like", 16, 0, 2600, 40, 2, true),
        ("wide, short", 12, 0, 400, 20, 2, true),
        ("narrow tall, preprocessed", 22, 12, 4, 2, 2, false),
        ("narrow short, preprocessed", 7, 8, 1, 1, 2, false),
        ("memw-like", 21, 0, 49, 13, 2, false),
        ("cpu-like", 21, 0, 74, 20, 2, true),
    ];
    let mut got = Vec::new();
    for &(name, b, pre, main, aux, parts, _) in cases {
        let w = TableWidths {
            precomputed: pre,
            main,
            aux: aux * 3,
            composition: parts * 3,
        };
        got.push((name, resolve_leaf_layout(&w, &o, b, 2).is_one_row()));
    }
    let want: Vec<_> = cases.iter().map(|c| (c.0, c.6)).collect();
    assert_eq!(got, want);
}

/// `auto` resolves per table from the AIR, and the prover and the verifier
/// resolve identically (one function, `table_leaf_layout`); a multi-table
/// proof can mix layouts.
#[test]
fn auto_resolves_per_table_from_the_air() {
    let o = golden_options(2, 1, 6, fmt(OneRowMode::Auto, FriMode::Dp, None));
    let air = crate::examples::simple_addition::SimpleAdditionAIR::<F>::new(&o);
    for log_rows in 1..=20 {
        let rows = 1usize << log_rows;
        let w = TableWidths::of(&air, rows);
        assert_eq!(w.main, air.trace_layout().0 as u64);
        assert_eq!(
            table_leaf_layout(&air, rows),
            resolve_leaf_layout(&w, &o, log_rows + 1, 1)
        );
    }
    // At the default format every AIR is row pairs, whatever its widths.
    let d = golden_options(2, 1, 6, ProofFormat::DEFAULT);
    let air = crate::examples::simple_addition::SimpleAdditionAIR::<F>::new(&d);
    assert_eq!(table_leaf_layout(&air, 1 << 20), LeafLayout::RowPair);
}

// ---------------------------------------------------------------------------
// REVIEW-FRI F9: {cap off, auto} × {pair, dp} × {0, 1, auto}, Q ≥ 20.
// ---------------------------------------------------------------------------

#[test]
fn cap_fri_one_row_matrix_round_trips() {
    for cap in [CapPolicy::Off, CapPolicy::Auto] {
        for fri in [FriMode::Pair, FriMode::Dp] {
            for one_row in [OneRowMode::Off, OneRowMode::On, OneRowMode::Auto] {
                let mut o = golden_options(4, 1, 24, fmt(one_row, fri, None));
                o.format.merkle_cap = cap;
                let (air, proof, _) = prove_logup::<Blake3StarkHash>(256, &o);
                assert!(
                    verify_logup::<Blake3StarkHash>(&air, &proof),
                    "cap {cap:?} fri {fri:?} one_row {one_row:?}"
                );
                let (air, proof) = prove_simple_addition::<KeccakStarkHash>(1024, &o);
                assert!(
                    verify_simple_addition::<KeccakStarkHash>(&air, &proof),
                    "simple cap {cap:?} fri {fri:?} one_row {one_row:?}"
                );
                if cap == CapPolicy::Auto {
                    // Q = 24 ≥ 20: every tree deeper than 3 carries a cap of 3
                    // at the end of query 0's path.
                    let lde_log = 12usize;
                    let layout = table_leaf_layout(&air, 1024);
                    let d = layout.tree_depth(lde_log);
                    assert_eq!(
                        proof.deep_poly_openings[0]
                            .main_trace_polys
                            .proof
                            .merkle_path
                            .len(),
                        d - 3 + 8
                    );
                    assert_eq!(
                        proof.deep_poly_openings[1]
                            .main_trace_polys
                            .proof
                            .merkle_path
                            .len(),
                        d - 3
                    );
                }
            }
        }
    }
}

/// The FRI layout the verifier builds for a table resolves the SAME layout the
/// prover used, for the base-field and the extension-field AIRs alike.
#[test]
fn widths_of_an_extension_air() {
    let o = golden_options(2, 1, 6, on(FriMode::Pair));
    let air = crate::examples::read_only_memory_logup::LogReadOnlyRAP::<F, E>::new(&o);
    let w = TableWidths::of(&air, 64);
    assert_eq!(w.aux, 3 * air.num_auxiliary_rap_columns() as u64);
    assert_eq!(w.main, air.trace_layout().0 as u64);
    assert!(w.composition % 3 == 0 && w.composition > 0);
    let _ = <F as IsFFTField>::TWO_ADICITY;
}
