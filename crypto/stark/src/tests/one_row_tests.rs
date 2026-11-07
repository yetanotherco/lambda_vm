//! S2 (one-row trace openings with a committed FRI input) on the CPU prover
//! and host verifier: the round trip U6 at one_row, the tamper
//! tests T4–T6, the load-bearing mutation M3, the transcript-order KAT, the
//! per-table `auto` rule, the preprocessed-root
//! miss (a hard error, never a recompute) and the cap × FRI × one-row matrix.

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
use crate::fri::schedule::FRI_COST_WEIGHTS;
use crate::fri::terminal::FriFoldLayout;
use crate::fri::{commit_phase_with_layout, fold_times};
use crate::leaf_layout::{
    LeafLayout, M3_PAIR_BOUND_AT_LDE, TableWidths, deep_point_xalu_rows, resolve_leaf_layout,
    table_leaf_layout, table_openings_cost_q,
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
/// `leaf_layout::M3_PAIR_BOUND_AT_LDE`).
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
// The layout helper: one place a query becomes rows.
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

/// No stray `2·iota(+1)` row arithmetic outside the helper in
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
// Soundness: r is uniform over ALL of D₀. M3 shows the test that says so
// is load-bearing.
// ---------------------------------------------------------------------------

/// The query indexes the verifier draws for a one-row SimpleAddition proof of
/// `rows` rows at blowup 2 with `queries` queries (and whether it verifies).
fn one_row_iotas(rows: usize, queries: usize) -> (Vec<usize>, bool) {
    let o = golden_options(2, 1, queries, on(FriMode::Pair));
    let (air, proof) = prove_simple_addition::<KeccakStarkHash>(rows, &o);
    let (ok, records) = capture(|| verify_simple_addition::<KeccakStarkHash>(&air, &proof));
    let rec = FriCapture::<F>::from_any(records[0].as_ref()).expect("one record");
    (rec.iotas.clone(), ok)
}

/// The M3 shape: 4096 rows at blowup 2, an LDE of 8192 points no other
/// one-row test proves at (the mutation is keyed by it), and 64 queries: all
/// 64 indexes below `N / 2` has probability 2⁻⁶⁴ under the right bound; the
/// pair bound makes it certain.
const M3_ROWS: usize = 4096;
const M3_LDE: usize = 2 * M3_ROWS;

fn upper_half_reached(iotas: &[usize], lde: usize) -> bool {
    iotas.iter().any(|&r| r >= lde / 2) && iotas.iter().all(|&r| r < lde)
}

#[test]
fn one_row_query_indexes_cover_the_whole_lde() {
    let _g = M3_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let (iotas, ok) = one_row_iotas(M3_ROWS, 64);
    assert!(ok);
    assert!(upper_half_reached(&iotas, M3_LDE), "iotas {iotas:?}");
}

/// M3: sample r over N/2 under one row. Prover and verifier agree on the
/// mutated bound, so the proof still VERIFIES — the bias is invisible to
/// verification, and only the bound test catches it.
#[test]
fn m3_the_query_bound_test_is_load_bearing() {
    let _g = M3_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    M3_PAIR_BOUND_AT_LDE.store(M3_LDE as u64, std::sync::atomic::Ordering::SeqCst);
    let (iotas, ok) = one_row_iotas(M3_ROWS, 64);
    M3_PAIR_BOUND_AT_LDE.store(0, std::sync::atomic::Ordering::SeqCst);
    assert!(ok, "the mutated proof still verifies (both sides mutated)");
    assert!(
        !upper_half_reached(&iotas, M3_LDE),
        "under the mutation the bound test must fail"
    );
}

// ---------------------------------------------------------------------------
// Soundness: the input root is absorbed before ζ₀ (transcript KAT).
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

/// M1 at the input tree: the input-slot check `group₀[slot] == DEEP(x_r)` is
/// what ties FRI to the trace openings under one row. A prover commits (and
/// folds) the input codeword `p₀ + c` — still low degree, so every layer and
/// the terminal are consistent — while DEEP(x_r) from the openings is `p₀`.
/// With the check the forgery is rejected; with it skipped (the mutation) it
/// is ACCEPTED.
#[test]
fn m1_the_input_slot_check_is_load_bearing() {
    use crate::fri::group::{
        GROUP_MUTATION, GroupMutation, roots_of_unity_table, verify_query_groups,
    };
    use crate::fri::query_phase_with_layout;
    use crate::fri::terminal::terminal_codeword_from_coeffs;
    use crate::merkle_caps::TreeCheck;
    use math::fft::bit_reversing::{in_place_bit_reverse_permute, reverse_index};
    type H = KeccakStarkHash;

    let o = Felt::from(3u64);
    let lde_log = 10u32;
    let n = 1usize << lde_log;
    let coeffs: Vec<Ext> = (0..256u64)
        .map(|i| Ext::new([Felt::from(i + 5), Felt::from(i * i), Felt::from(11)]))
        .collect();
    let poly = math::polynomial::Polynomial::new(&coeffs);
    let mut p0 =
        math::polynomial::Polynomial::evaluate_offset_fft::<F>(&poly, 4, Some(256), &o).unwrap();
    in_place_bit_reverse_permute(&mut p0);
    let c = Ext::new([Felt::from(5u64), Felt::from(6u64), Felt::from(7u64)]);
    let shifted: Vec<Ext> = p0.iter().map(|v| v + &c).collect();

    // One row, schedule [3, 2, 3] from 10 to T = 2 + 0.
    let layout = FriFoldLayout::from_schedule(lde_log, 2, 0, true, vec![3, 2, 3]).unwrap();
    let tw = compute_coset_twiddles_inv::<F>(&o, n);
    let mut t = DefaultTranscript::<E>::new(&[5]);
    let (tcoeffs, layers) =
        commit_phase_with_layout::<F, E, _, H>(shifted, &mut t, &o, n, 2, 0, &layout, &tw);
    let roots: Vec<[u8; 32]> = layers.iter().map(|l| l.merkle_tree.root).collect();
    // Replay: root₀ first, then (ζ, root) per later layer, then the final ζ.
    let mut replay = DefaultTranscript::<E>::new(&[5]);
    let mut zetas = Vec::new();
    for (j, r) in roots.iter().enumerate() {
        if j > 0 {
            zetas.push(replay.sample_field_element());
        }
        replay.append_bytes(r);
    }
    zetas.push(replay.sample_field_element());
    assert_eq!(zetas.len(), layout.num_zetas());
    let queries: Vec<usize> = (0..n).step_by(53).collect();
    let decs = query_phase_with_layout::<E, H>(&layers, &queries, &layout);
    let terminal = terminal_codeword_from_coeffs::<F, E>(
        &tcoeffs,
        &o.pow(1u64 << layout.total_folds),
        layout.terminal_len,
    );
    let tables: Vec<Vec<Felt>> = (0..=6)
        .map(|d| roots_of_unity_table::<F>(d).unwrap())
        .collect();
    let checks: Vec<TreeCheck<'_>> = roots
        .iter()
        .enumerate()
        .map(|(j, root)| {
            TreeCheck::build::<<H as crate::config::StarkHash>::Batched<E>>(
                root,
                layout.layer_depth(lde_log, j) as usize,
                0,
                || None,
            )
            .unwrap()
        })
        .collect();
    let accepts = |deep: &[Ext]| {
        queries.iter().zip(&decs).all(|(&r, dec)| {
            let w = F::get_primitive_root_of_unity(u64::from(lde_log)).unwrap();
            let x_r = &o * w.pow(reverse_index(r, n as u64) as u64);
            verify_query_groups::<F, E, <H as crate::config::StarkHash>::Batched<E>>(
                &layout,
                &checks,
                1,
                |j| dec.layers_auth_paths[j].merkle_path.as_slice(),
                &dec.layers_evaluations_sym,
                &zetas,
                r,
                deep[r],
                x_r.inv().unwrap(),
                &terminal,
                &tables,
            )
        })
    };
    let shifted_again: Vec<Ext> = p0.iter().map(|v| v + &c).collect();
    assert!(accepts(&shifted_again), "control: honest for p0 + c");
    assert!(!accepts(&p0), "the input-slot check must reject");
    GROUP_MUTATION.with(|m| m.set(GroupMutation::SkipSlotCheck));
    let mutated = accepts(&p0);
    GROUP_MUTATION.with(|m| m.set(GroupMutation::None));
    assert!(
        mutated,
        "without the input-slot check the forgery is accepted (the check is load-bearing)"
    );
}

// ---------------------------------------------------------------------------
// Preprocessed tables: one-row roots, and a miss is an error (never a recompute).
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
// The per-table `auto` rule.
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
                            deep_point_rows: deep_point_xalu_rows(main + 2 * aux, 2, 2),
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
        deep_point_rows: deep_point_xalu_rows(1, 1, 1),
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

/// The DEEP term of the pinned cases: `E = 2` OOD rows (a current and a next
/// row; every case has aux columns, whose LogUp accumulators read the next
/// row), every column opened at the current row and the aux columns at the
/// next (an ASSUMED window: the real one is each AIR's
/// `trace_ood_next_row_columns`), `parts` composition parts.
fn pinned_deep_rows(pre: u64, main: u64, aux: u64, parts: u64) -> u64 {
    deep_point_xalu_rows(pre + main + aux + aux, 2, parts)
}

/// ⚠ A FORMAT PIN: `auto`'s choice for a set of production-like shapes (Q =
/// 110, blowup 4, k = 7, cap auto, fri dp). Wide tables go one-row, narrow
/// tall ones stay row pairs. Any change to the cost function or its weights
/// that moves one of these is a format change. The widths are illustrative
/// (MEMW 49 main / 13 aux; the others are round
/// numbers), not a census. With every emitted FRI row priced and DEEP at two
/// points vs one, two choices are one row only because of the DEEP term: the
/// MEMW-like case (one row by 5.5%, see `auto_choices_margins`) and the narrow short
/// preprocessed one (its LDE is already terminal, so no FRI layer separates
/// the layouts and the second DEEP point decides).
#[test]
fn auto_choices_are_pinned() {
    let o = opts_q(110, OneRowMode::Auto, FriMode::Dp, CapPolicy::Auto);
    // (name, B, precomputed, main, aux ext columns, composition parts, one row?)
    let cases: &[(&str, u32, u64, u64, u64, u64, bool)] = &[
        ("wide keccak-like", 16, 0, 2600, 40, 2, true),
        ("wide, short", 12, 0, 400, 20, 2, true),
        ("narrow tall, preprocessed", 22, 12, 4, 2, 2, false),
        ("narrow short, preprocessed", 7, 8, 1, 1, 2, true),
        ("memw-like", 21, 0, 49, 13, 2, MEMW_LIKE_ONE_ROW),
        ("cpu-like", 21, 0, 74, 20, 2, true),
    ];
    let mut got = Vec::new();
    for &(name, b, pre, main, aux, parts, _) in cases {
        got.push((
            name,
            resolve_leaf_layout(&pinned_widths(pre, main, aux, parts), &o, b, 2).is_one_row(),
        ));
    }
    let want: Vec<_> = cases.iter().map(|c| (c.0, c.6)).collect();
    assert_eq!(got, want);
}

/// The MEMW-like case's pinned choice (see `auto_choices_are_pinned`).
const MEMW_LIKE_ONE_ROW: bool = true;

fn pinned_widths(pre: u64, main: u64, aux: u64, parts: u64) -> TableWidths {
    TableWidths {
        precomputed: pre,
        main,
        aux: aux * 3,
        composition: parts * 3,
        deep_point_rows: pinned_deep_rows(pre, main, aux, parts),
    }
}

/// Prints each pinned case's two prices (`-- --nocapture`), and pins how much
/// of the one-row saving the DEEP term is at the two edge cases.
#[test]
fn auto_choices_margins() {
    let o = opts_q(110, OneRowMode::Auto, FriMode::Dp, CapPolicy::Auto);
    for (name, pre, main, aux, parts) in [
        ("memw-like", 0u64, 49u64, 13u64, 2u64),
        ("cpu-like", 0, 74, 20, 2),
    ] {
        let w = pinned_widths(pre, main, aux, parts);
        let row = table_openings_cost_q(&w, &o, 21, 2, true);
        let pair = table_openings_cost_q(&w, &o, 21, 2, false);
        let deep_point = 110 * w.deep_point_rows * FRI_COST_WEIGHTS.xalu;
        println!(
            "  {name}: row {row} pair {pair} (x Q ns; one DEEP point {deep_point}; row/pair {:.4})",
            row as f64 / pair as f64
        );
        assert!(row < pair, "{name} goes one row");
        // Without the DEEP point one row saves, the MEMW-like case would stay
        // row pairs: the term decides it.
        if name == "memw-like" {
            assert!(row + deep_point >= pair, "{name}: the DEEP term decides");
        }
    }
}

/// DEEP costs two points under row pairs and one under one row,
/// each [`TableWidths::deep_point_rows`] XALU rows per query — and nothing
/// else in the price depends on it.
#[test]
fn the_deep_term_is_two_points_vs_one() {
    for fri in [FriMode::Pair, FriMode::Dp] {
        for cap in [CapPolicy::Off, CapPolicy::Auto] {
            let o = opts_q(110, OneRowMode::Auto, fri, cap);
            let base = pinned_widths(0, 49, 13, 2);
            let none = TableWidths {
                deep_point_rows: 0,
                ..base
            };
            let point = 110 * base.deep_point_rows * FRI_COST_WEIGHTS.xalu;
            for lde_log in 8..=22u32 {
                for (one_row, points) in [(true, 1u64), (false, 2)] {
                    assert_eq!(
                        table_openings_cost_q(&base, &o, lde_log, 2, one_row),
                        table_openings_cost_q(&none, &o, lde_log, 2, one_row) + points * point,
                        "fri {fri:?} cap {cap:?} B {lde_log} one_row {one_row}"
                    );
                }
            }
        }
    }
    // The formula: one XALU row per surviving opening, plus 4 per OOD row,
    // one per part and 3.
    assert_eq!(deep_point_xalu_rows(100, 2, 2), 100 + 8 + 2 + 3);
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
// The format matrix: {cap off, auto} × {pair, dp} × {0, 1, auto}, Q ≥ 20.
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
    assert!(w.composition.is_multiple_of(3) && w.composition > 0);
    // The DEEP term from the AIR's own OOD layout (the verifier's reading).
    let e = air.context().transition_offsets.len() * air.step_size();
    let ood = crate::ood::OodLayout::new(
        air.context().trace_columns,
        e,
        air.step_size(),
        air.trace_ood_next_row_columns(),
    );
    assert_eq!(
        w.deep_point_rows,
        deep_point_xalu_rows(ood.num_surviving() as u64, e as u64, w.composition / 3)
    );
    assert!(w.deep_point_rows > ood.num_surviving() as u64);
    let _ = <F as IsFFTField>::TWO_ADICITY;
}
