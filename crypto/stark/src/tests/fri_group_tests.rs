//! S3 (group-leaf FRI layers) on the CPU prover and host verifier: FRI.md §10
//! U4–U6, the tamper tests T1–T3, the load-bearing mutations M1–M2 and the
//! differential of the group path at the all-ones schedule against the legacy
//! path (REVIEW-FRI F1.2).

use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use crypto::fiat_shamir::is_transcript::IsTranscript;
use math::fft::bit_reversing::{in_place_bit_reverse_permute, reverse_index};
use math::field::element::FieldElement;
use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField;
use math::field::goldilocks::GoldilocksField;
use math::field::traits::IsFFTField;
use math::polynomial::Polynomial;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

use crate::config::{Blake3StarkHash, KeccakStarkHash, StarkHash};
use crate::fri::fri_functions::compute_coset_twiddles_inv;
use crate::fri::group::{
    GROUP_MUTATION, GroupMutation, group_fold, roots_of_unity_table, verify_query_groups,
};
use crate::fri::terminal::{FriFoldLayout, terminal_codeword_from_coeffs};
use crate::fri::{commit_phase_with_layout, fold_times, query_phase_with_layout};
use crate::merkle_caps::TreeCheck;
use crate::proof::options::{FriMode, FriScheduleOverride, ProofFormat};
use crate::traits::AIR;

use super::zf_golden_tests::{
    fingerprint, golden_options, prove_logup, prove_multi, prove_simple_addition, verify_logup,
    verify_multi, verify_simple_addition,
};

type F = GoldilocksField;
type E = Degree3GoldilocksExtensionField;
type Felt = FieldElement<F>;
type Ext = FieldElement<E>;

fn rand_ext(rng: &mut ChaCha20Rng) -> Ext {
    Ext::new([
        Felt::from(rng.r#gen::<u64>()),
        Felt::from(rng.r#gen::<u64>()),
        Felt::from(rng.r#gen::<u64>()),
    ])
}

/// The point at position `p` of a bit-reversed layer of length `2^b` on the
/// coset `o·⟨ω_{2^b}⟩`.
fn point(o: &Felt, b: u32, p: usize) -> Felt {
    let w = F::get_primitive_root_of_unity(u64::from(b)).unwrap();
    o * w.pow(reverse_index(p, 1u64 << b) as u64)
}

/// A bit-reversed coset codeword of a random ext3 polynomial with `num_coeffs`
/// coefficients over `2^b` points; returns (codeword, coefficients).
fn random_codeword(
    rng: &mut ChaCha20Rng,
    b: u32,
    num_coeffs: usize,
    o: &Felt,
) -> (Vec<Ext>, Vec<Ext>) {
    let coeffs: Vec<Ext> = (0..num_coeffs).map(|_| rand_ext(rng)).collect();
    let poly = Polynomial::new(&coeffs);
    let n = 1usize << b;
    let mut cw = Polynomial::evaluate_offset_fft::<F>(
        &poly,
        n / num_coeffs.next_power_of_two(),
        Some(num_coeffs),
        o,
    )
    .expect("fft");
    assert_eq!(cw.len(), n);
    in_place_bit_reverse_permute(&mut cw);
    (cw, coeffs)
}

fn dp_with(schedule: Option<&[u8]>) -> ProofFormat {
    ProofFormat {
        fri_mode: FriMode::Dp,
        fri_schedule_override: schedule.map(|s| FriScheduleOverride::new(s).unwrap()),
        ..ProofFormat::DEFAULT
    }
}

// ---------------------------------------------------------------------------
// U5: a group leaf is a coset, and its base folds to the next layer's point.
// ---------------------------------------------------------------------------

#[test]
fn group_leaf_is_a_coset() {
    let o = Felt::from(3u64);
    for b in 1..=10u32 {
        for d in 1..=b.min(6) {
            let roots = roots_of_unity_table::<F>(d).unwrap();
            // The table's root is the layer domain's ω_{2^b}^{2^{b−d}}.
            let w_b = F::get_primitive_root_of_unity(u64::from(b)).unwrap();
            assert_eq!(roots[1], w_b.pow(1u64 << (b - d)), "b={b} d={d}");
            let o_next = o.pow(1u64 << d);
            for g in 0..(1usize << (b - d)) {
                let x_g = &o * w_b.pow(reverse_index(g, 1u64 << (b - d)) as u64);
                for t in 0..(1usize << d) {
                    let want = &x_g * &roots[reverse_index(t, 1u64 << d)];
                    assert_eq!(point(&o, b, (g << d) + t), want, "b={b} d={d} g={g} t={t}");
                }
                // x_g^{2^d} is position g of the layer folded d times.
                assert_eq!(
                    x_g.pow(1u64 << d),
                    point(&o_next, b - d, g),
                    "b={b} d={d} g={g}"
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// U4: the group fold = d binary folds with ζ, ζ², … = 2^d·Σ ζ^i f_i.
// ---------------------------------------------------------------------------

#[test]
fn group_fold_equals_d_binary_folds() {
    let mut rng = ChaCha20Rng::seed_from_u64(0x5334);
    let o = Felt::from(3u64);
    let b = 9u32;
    let n = 1usize << b;
    let (codeword, coeffs) = random_codeword(&mut rng, b, n, &o);
    for d in 1..=6u32 {
        let zeta = rand_ext(&mut rng);
        // Prover: d binary folds with ζ^{2^ℓ} (`fold_times`, the commit loop's).
        let mut folded = codeword.clone();
        let mut tw = compute_coset_twiddles_inv::<F>(&o, n);
        fold_times(&mut folded, &zeta, d, &mut tw);
        assert_eq!(folded.len(), n >> d);

        let roots = roots_of_unity_table::<F>(d).unwrap();
        let o_next = o.pow(1u64 << d);
        let two_d = Felt::from(1u64 << d);
        for g in 0..(n >> d) {
            let group = &codeword[g << d..(g + 1) << d];
            // Verifier: the group fold from ANY slot's point gives the same value.
            for s in 0..(1usize << d) {
                let y_inv = point(&o, b, (g << d) + s).inv().unwrap();
                let x_g_inv = &y_inv * &roots[reverse_index(s, 1u64 << d)];
                assert_eq!(
                    group_fold::<F, E>(group, &zeta, &x_g_inv, &roots),
                    folded[g],
                    "d={d} g={g} slot={s}"
                );
            }
            // The polynomial identity: 2^d · Σ_i ζ^i f_i(Y), f(X) = Σ X^i f_i(X^{2^d}).
            let y = point(&o_next, b - d, g);
            let mut acc = Ext::zero();
            let mut zp = Ext::one();
            for i in 0..(1usize << d) {
                let mut fi = Ext::zero();
                let mut yp = Felt::one();
                for k in (i..n).step_by(1 << d) {
                    fi += &yp * &coeffs[k];
                    yp = &yp * &y;
                }
                acc += &zp * &fi;
                zp = &zp * &zeta;
            }
            assert_eq!(folded[g], &two_d * &acc, "d={d} g={g}: 2^d·Σζ^i f_i");
        }
    }
}

// ---------------------------------------------------------------------------
// FRI-level harness (M1, M2): commit a codeword, open queries, verify with the
// same group checks the host verifier runs.
// ---------------------------------------------------------------------------

struct FriRun {
    layout: FriFoldLayout,
    lde_log: u32,
    roots: Vec<[u8; 32]>,
    zetas: Vec<Ext>,
    coeffs: Vec<Ext>,
    decommitments: Vec<crate::fri::fri_decommit::FriDecommitment<E>>,
    iotas: Vec<usize>,
    terminal_offset: Felt,
}

/// FRI over `committed` (what the prover commits); the transcript is replayed
/// to recover ζ. `lde_log` 10, blowup 4 (log 2), k 1: the chain covers 9 → 3,
/// schedule `[3, 1, 2]`.
fn fri_run<H: StarkHash>(committed: &[Ext], o: &Felt) -> FriRun {
    let lde_log = 10u32;
    let n = 1usize << lde_log;
    assert_eq!(committed.len(), n);
    let layout = FriFoldLayout::from_schedule(lde_log, 2, 1, false, vec![3, 1, 2]).unwrap();
    assert!(!layout.is_legacy());
    let tw = compute_coset_twiddles_inv::<F>(o, n);
    let mut transcript = DefaultTranscript::<E>::new(&[7]);
    let (coeffs, layers) = commit_phase_with_layout::<F, E, _, H>(
        committed.to_vec(),
        &mut transcript,
        o,
        n,
        2,
        1,
        &layout,
        &tw,
    );
    let roots: Vec<[u8; 32]> = layers.iter().map(|l| l.merkle_tree.root).collect();
    let mut replay = DefaultTranscript::<E>::new(&[7]);
    let mut zetas = Vec::new();
    for r in &roots {
        zetas.push(replay.sample_field_element());
        replay.append_bytes(r);
    }
    zetas.push(replay.sample_field_element());
    let iotas: Vec<usize> = (0..n / 2).step_by(37).collect();
    let decommitments = query_phase_with_layout::<E, H>(&layers, &iotas, &layout);
    FriRun {
        terminal_offset: o.pow(1u64 << layout.total_folds),
        layout,
        lde_log,
        roots,
        zetas,
        coeffs,
        decommitments,
        iotas,
    }
}

/// Verify every query of `run` with DEEP values read from `deep` (the
/// codeword the VERIFIER believes in, bit-reversed).
fn fri_accepts<H: StarkHash>(run: &FriRun, deep: &[Ext], o: &Felt) -> bool {
    let terminal = terminal_codeword_from_coeffs::<F, E>(
        &run.coeffs,
        &run.terminal_offset,
        run.layout.terminal_len,
    );
    let tables: Vec<Vec<Felt>> = (0..=6)
        .map(|d| roots_of_unity_table::<F>(d).unwrap())
        .collect();
    // One uncapped check per layer tree, at the layout's group-tree depth.
    let checks: Vec<TreeCheck<'_>> = run
        .roots
        .iter()
        .enumerate()
        .map(|(j, root)| {
            let depth = run.layout.layer_depth(run.lde_log, j) as usize;
            TreeCheck::build::<H::Batched<E>>(root, depth, 0, || None).unwrap()
        })
        .collect();
    run.iotas
        .iter()
        .zip(&run.decommitments)
        .all(|(&iota, dec)| {
            let x = point(o, run.lde_log, 2 * iota);
            let x_inv = x.inv().unwrap();
            let (p0, p0s) = (&deep[2 * iota], &deep[2 * iota + 1]);
            let v = (p0 + p0s) + &x_inv * &run.zetas[0] * (p0 - p0s);
            verify_query_groups::<F, E, H::Batched<E>>(
                &run.layout,
                &checks,
                0,
                |j| dec.layers_auth_paths[j].merkle_path.as_slice(),
                &dec.layers_evaluations_sym,
                &run.zetas,
                iota,
                v,
                x_inv.square(),
                &terminal,
                &tables,
            )
        })
}

fn with_mutation<T>(m: GroupMutation, f: impl FnOnce() -> T) -> T {
    GROUP_MUTATION.with(|c| c.set(m));
    let out = f();
    GROUP_MUTATION.with(|c| c.set(GroupMutation::None));
    out
}

/// A low-degree ext3 codeword on LDE 2^10, blowup 4 (256 coefficients).
fn low_degree(seed: u64, o: &Felt) -> Vec<Ext> {
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    random_codeword(&mut rng, 10, 256, o).0
}

#[test]
fn honest_fri_run_is_accepted() {
    let o = Felt::from(3u64);
    let p0 = low_degree(1, &o);
    let run = fri_run::<KeccakStarkHash>(&p0, &o);
    assert_eq!(run.roots.len(), 3);
    assert!(fri_accepts::<KeccakStarkHash>(&run, &p0, &o));
    let run = fri_run::<Blake3StarkHash>(&p0, &o);
    assert!(fri_accepts::<Blake3StarkHash>(&run, &p0, &o));
}

/// M1 — the slot check is load-bearing. A prover commits FRI for
/// `p₀ + c` (still low degree, so every layer and the terminal are
/// consistent) while the trace openings say `p₀`: only `group[slot] == v` at
/// the first committed layer ties FRI to the DEEP value. With it the forgery
/// is rejected; with it skipped (the mutation) it is ACCEPTED.
#[test]
fn m1_the_slot_check_is_load_bearing() {
    let o = Felt::from(3u64);
    let p0 = low_degree(2, &o);
    let c = Ext::new([Felt::from(5u64), Felt::from(6u64), Felt::from(7u64)]);
    let shifted: Vec<Ext> = p0.iter().map(|v| v + &c).collect();
    let run = fri_run::<KeccakStarkHash>(&shifted, &o);
    assert!(
        fri_accepts::<KeccakStarkHash>(&run, &shifted, &o),
        "control: FRI of p0 + c is honest for p0 + c"
    );
    assert!(
        !fri_accepts::<KeccakStarkHash>(&run, &p0, &o),
        "the slot check must reject"
    );
    assert!(
        with_mutation(GroupMutation::SkipSlotCheck, || fri_accepts::<
            KeccakStarkHash,
        >(&run, &p0, &o)),
        "without the slot check the forgery is accepted (the check is load-bearing)"
    );
}

/// M2 — the group's Merkle authentication is load-bearing. Replacing a layer
/// root (with the challenges kept) leaves the fold chain consistent; only the
/// authentication of the group against the root rejects it.
#[test]
fn m2_the_group_authentication_is_load_bearing() {
    let o = Felt::from(3u64);
    let p0 = low_degree(3, &o);
    let mut run = fri_run::<KeccakStarkHash>(&p0, &o);
    run.roots[1] = [0xAB; 32];
    assert!(
        !fri_accepts::<KeccakStarkHash>(&run, &p0, &o),
        "authentication must reject"
    );
    assert!(
        with_mutation(
            GroupMutation::SkipLeafAuth,
            || fri_accepts::<KeccakStarkHash>(&run, &p0, &o)
        ),
        "without authentication the foreign root is accepted (the check is load-bearing)"
    );
}

// ---------------------------------------------------------------------------
// U6: prove / verify round trips at fri = dp.
// ---------------------------------------------------------------------------

/// SimpleAddition at `rows`, blowup `blowup`, k = 1, under `format`: returns
/// the proof's committed-layer count and the values per query, after asserting
/// it verifies.
fn round_trip_simple<H: StarkHash>(rows: usize, blowup: u8, format: ProofFormat) -> (usize, usize) {
    let o = golden_options(blowup, 1, 9, format);
    let (air, proof) = prove_simple_addition::<H>(rows, &o);
    assert!(
        verify_simple_addition::<H>(&air, &proof),
        "rows {rows} blowup {blowup} {format:?}: an honest proof must verify"
    );
    let layers = proof.fri_layers_merkle_roots.len();
    let values = proof.query_list[0].layers_evaluations_sym.len();
    for q in &proof.query_list {
        assert_eq!(q.layers_auth_paths.len(), layers);
        assert_eq!(q.layers_evaluations_sym.len(), values);
    }
    (layers, values)
}

#[test]
fn dp_round_trips_at_every_fold_count() {
    // k = 1: total_folds = log2(rows) + blowup_log − (blowup_log + 1).
    for blowup in [2u8, 4] {
        for log_rows in 1..=10u32 {
            let rows = 1usize << log_rows;
            let (layers, values) =
                round_trip_simple::<KeccakStarkHash>(rows, blowup, dp_with(None));
            let lde_log = log_rows + blowup.trailing_zeros();
            let o = golden_options(blowup, 1, 9, dp_with(None));
            let l = FriFoldLayout::for_options(lde_log, blowup.trailing_zeros(), &o).unwrap();
            assert_eq!(layers, l.num_committed, "rows {rows}");
            assert_eq!(values, l.opened_values_per_query(), "rows {rows}");
        }
    }
    // A shape where the DP picks a non-trivial schedule is exercised.
    let o = golden_options(4, 1, 9, dp_with(None));
    let l = FriFoldLayout::for_options(12, 2, &o).unwrap();
    assert!(
        l.schedule.iter().any(|&d| d > 1),
        "schedule {:?}",
        l.schedule
    );
}

#[test]
fn dp_round_trips_under_explicit_schedules() {
    // rows 2^9, blowup 4, k 1: lde_log 11, chain from 10 to T = 3: 7 bits.
    for sched in [
        &[1u8, 3, 3][..],
        &[3, 1, 3],
        &[2, 1, 2, 2],
        &[1, 1, 1, 1, 1, 1, 1],
        &[6, 1],
        &[1, 6],
        &[4, 3],
    ] {
        for blake in [false, true] {
            let (layers, values) = if blake {
                round_trip_simple::<Blake3StarkHash>(512, 4, dp_with(Some(sched)))
            } else {
                round_trip_simple::<KeccakStarkHash>(512, 4, dp_with(Some(sched)))
            };
            assert_eq!(layers, sched.len(), "{sched:?}");
            assert_eq!(
                values,
                sched.iter().map(|&d| 1usize << d).sum::<usize>(),
                "{sched:?}"
            );
        }
    }
    // An override that does not fit is a proving error, never a fallback.
    let o = golden_options(4, 1, 9, dp_with(Some(&[3, 1])));
    let air = crate::examples::simple_addition::SimpleAdditionAIR::<F>::new(&o);
    let mut trace = crate::examples::simple_addition::simple_addition_trace::<F>(512);
    let pi = crate::examples::simple_addition::SimpleAdditionPublicInputs {
        a: Felt::from(1u64),
        b: Felt::from(2u64),
    };
    use crate::prover::IsStarkProver;
    let res = crate::prover::GenericProver::<F, F, _, KeccakStarkHash>::prove(
        &air,
        &mut trace,
        &pi,
        &mut DefaultTranscript::<F>::new(&[]),
    );
    assert!(
        res.is_err(),
        "a schedule that does not cover the folds must be refused"
    );
}

#[test]
fn dp_round_trips_ext3_aux_and_multi_table() {
    for (rows, blowup) in [(16usize, 2u8), (128, 4), (512, 2)] {
        let lde_log = rows.trailing_zeros() + blowup.trailing_zeros();
        // Committed chain: from lde_log − 1 down to T = blowup_log + k (k = 1).
        let span = (lde_log - 1 - (blowup.trailing_zeros() + 1)) as u8;
        // An uneven explicit schedule where there is room for one.
        let explicit = if span >= 3 {
            vec![2u8, span - 2]
        } else {
            vec![span]
        };
        for format in [dp_with(None), dp_with(Some(&explicit))] {
            let o = golden_options(blowup, 1, 7, format);
            let (air, proof, _) = prove_logup::<Blake3StarkHash>(rows, &o);
            assert!(
                verify_logup::<Blake3StarkHash>(&air, &proof),
                "logup rows {rows} {format:?}"
            );
            let (air, proof, _) = prove_logup::<KeccakStarkHash>(rows, &o);
            assert!(
                verify_logup::<KeccakStarkHash>(&air, &proof),
                "logup keccak rows {rows}"
            );
        }
    }
    let o = golden_options(2, 1, 6, dp_with(None));
    let multi = prove_multi::<Blake3StarkHash>(&o);
    assert!(verify_multi::<Blake3StarkHash>(&o, &multi));
    assert!(
        multi
            .proofs
            .iter()
            .any(|p| !p.fri_layers_merkle_roots.is_empty())
    );
}

/// The format is a verifier-side constant: a dp proof does not verify under
/// pair options, nor a pair proof under dp options.
#[test]
fn the_format_is_a_verifier_constant() {
    let dp = golden_options(4, 1, 9, dp_with(None));
    let pair = golden_options(4, 1, 9, ProofFormat::DEFAULT);
    let (_, dp_proof) = prove_simple_addition::<KeccakStarkHash>(1024, &dp);
    let (_, pair_proof) = prove_simple_addition::<KeccakStarkHash>(1024, &pair);
    let dp_air = crate::examples::simple_addition::SimpleAdditionAIR::<F>::new(&dp);
    let pair_air = crate::examples::simple_addition::SimpleAdditionAIR::<F>::new(&pair);
    assert!(verify_simple_addition::<KeccakStarkHash>(
        &dp_air, &dp_proof
    ));
    assert!(verify_simple_addition::<KeccakStarkHash>(
        &pair_air,
        &pair_proof
    ));
    assert!(!verify_simple_addition::<KeccakStarkHash>(
        &pair_air, &dp_proof
    ));
    assert!(!verify_simple_addition::<KeccakStarkHash>(
        &dp_air,
        &pair_proof
    ));
}

// ---------------------------------------------------------------------------
// F1.2: the group path at the all-ones schedule vs the legacy path.
// ---------------------------------------------------------------------------

/// Proving under `dp` with an all-ones schedule runs the GROUP code path (group
/// trees via `H::Batched`, full-group encoding, the group verifier) where the
/// legacy format runs the pair path. Every root, the terminal polynomial, every
/// trace/composition opening and every FRI path must be identical (so ζ and ι
/// are too), and each two-value group must be exactly the legacy pair: the
/// legacy sibling is the group entry that is not the query's own value.
#[test]
fn generic_path_at_all_ones_equals_legacy() {
    for blowup in [2u8, 4] {
        let rows = 256usize;
        let lde_log = rows.trailing_zeros() + blowup.trailing_zeros();
        let span = (lde_log - 1 - (blowup.trailing_zeros() + 1)) as usize;
        let ones = vec![1u8; span];
        let legacy_o = golden_options(blowup, 1, 9, ProofFormat::DEFAULT);
        let group_o = golden_options(blowup, 1, 9, dp_with(Some(&ones)));
        let (_, legacy, _) = prove_logup::<KeccakStarkHash>(rows, &legacy_o);
        let (air, group, _) = prove_logup::<KeccakStarkHash>(rows, &group_o);
        assert!(verify_logup::<KeccakStarkHash>(&air, &group));

        assert_eq!(
            legacy.fri_layers_merkle_roots,
            group.fri_layers_merkle_roots
        );
        assert_eq!(legacy.fri_final_poly_coeffs, group.fri_final_poly_coeffs);
        let a = fingerprint!(&legacy);
        let b = fingerprint!(&group);
        assert_eq!(
            a.openings, b.openings,
            "trace/composition openings (so every ι) equal"
        );
        assert_eq!(a.fri_roots, b.fri_roots);
        assert_ne!(
            a.proof, b.proof,
            "the encodings differ (full groups vs siblings)"
        );
        for (lq, gq) in legacy.query_list.iter().zip(&group.query_list) {
            assert_eq!(lq.layers_auth_paths.len(), span);
            for j in 0..span {
                assert_eq!(
                    lq.layers_auth_paths[j].merkle_path, gq.layers_auth_paths[j].merkle_path,
                    "layer {j}: same leaf, same path"
                );
                let pair = &gq.layers_evaluations_sym[2 * j..2 * j + 2];
                let sym = &lq.layers_evaluations_sym[j];
                assert!(
                    pair.contains(sym),
                    "layer {j}: the legacy sibling is in the group"
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// T1–T3: tamper tests on a dp proof with non-trivial groups.
// ---------------------------------------------------------------------------

#[test]
fn tampering_any_fri_value_path_or_root_is_rejected() {
    let format = dp_with(Some(&[3, 2, 2]));
    let o = golden_options(4, 1, 5, format);
    let (air, honest, _) = prove_logup::<KeccakStarkHash>(512, &o);
    assert!(verify_logup::<KeccakStarkHash>(&air, &honest));
    let values = honest.query_list[0].layers_evaluations_sym.len();
    assert_eq!(values, 8 + 4 + 4);
    let bump = Ext::new([Felt::one(), Felt::zero(), Felt::zero()]);

    // T1/T2: every value of query 0's groups (the slot value and every other).
    for i in 0..values {
        let mut p = honest.clone();
        p.query_list[0].layers_evaluations_sym[i] += bump;
        assert!(
            !verify_logup::<KeccakStarkHash>(&air, &p),
            "value {i} tampered"
        );
    }
    // A value of the LAST query too.
    let last = honest.query_list.len() - 1;
    let mut p = honest.clone();
    p.query_list[last].layers_evaluations_sym[values - 1] += bump;
    assert!(!verify_logup::<KeccakStarkHash>(&air, &p));
    // One path sibling per layer.
    for j in 0..3 {
        let mut p = honest.clone();
        p.query_list[0].layers_auth_paths[j].merkle_path[0][0] ^= 1;
        assert!(!verify_logup::<KeccakStarkHash>(&air, &p), "layer {j} path");
    }
    // A layer root.
    for j in 0..3 {
        let mut p = honest.clone();
        p.fri_layers_merkle_roots[j][5] ^= 1;
        assert!(!verify_logup::<KeccakStarkHash>(&air, &p), "layer {j} root");
    }
    // T3: the flat value vector one short / one long (checked before the loop,
    // so neither panics).
    let mut p = honest.clone();
    p.query_list[0].layers_evaluations_sym.pop();
    assert!(!verify_logup::<KeccakStarkHash>(&air, &p));
    let mut p = honest.clone();
    p.query_list[0].layers_evaluations_sym.push(Ext::zero());
    assert!(!verify_logup::<KeccakStarkHash>(&air, &p));
    // A path of the wrong length (the exact depth is a verifier constant).
    let mut p = honest.clone();
    p.query_list[0].layers_auth_paths[1].merkle_path.pop();
    assert!(!verify_logup::<KeccakStarkHash>(&air, &p));
    // One layer too few.
    let mut p = honest.clone();
    p.fri_layers_merkle_roots.pop();
    assert!(!verify_logup::<KeccakStarkHash>(&air, &p));
}
