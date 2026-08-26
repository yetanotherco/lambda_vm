//! The RPX256 `LFM_HASH` chip: its mixed-schedule layout, its degree bound, and
//! the prove+verify that turns a predicted cell count into a measured one.
//!
//! ## What pins what
//!
//! The permutation is pinned in [`super::rpx`], and ⚠ **more weakly than RPO's**
//! — miden publishes no RPX known-answer table, so the anchors are (a) the
//! shared constants, MDS and FB round, externally anchored through RPO's
//! nineteen vectors, and (b) the new extension arithmetic, pinned against naive
//! polynomial multiplication and generic exponentiation. Nothing here re-checks
//! the algebra; this module checks the *chip*.
//!
//! ## What is different from the RPO arm
//!
//! The schedule. RPX's seven rounds are three kinds — FB, E, M — so the layout's
//! block width depends on the round, the constraint count is not a multiple of
//! anything, and the padding-by-zero argument has to hold in all three. Those
//! are the properties worth testing, and they are what this file tests.

use math::field::element::FieldElement;
use stark::constraints::builder::{
    CaptureBuilder, ConstraintSet, ProverEvalFolder, RootKind, num_base_from_meta,
};
use stark::frame::Frame;
use stark::proof::options::{GoldilocksCubicProofOptions, ProofOptions};
use stark::table::TableView;
use stark::traits::TransitionEvaluationContext;

use crate::tables::types::{FE, GoldilocksExtension, GoldilocksField};

use super::airs::lfm_chip_census_with_hasher;
use super::chips::hash::{self, HashConstraints, rpx_cols as rc};
use super::fixture::fixture_prove_with_hasher;
use super::hash::{HASH_STATE_FELTS, HasherKind, LfmHasher};
use super::instr::HashMode;
use super::programs::{fri_toy_program, trivial_program};
use super::proof::{lfm_prove_with_hasher, verify_against};
use super::registry::build_artifacts_with_hasher;
use super::rpx::{EXT_DEGREE, EXT_ELEMENTS, NUM_ROUNDS, Rpx256, is_fb_round, is_final_round};
use super::trace::fill_rpx_witness;
use super::word::LfmWord;

type Gl = GoldilocksField;
type Gl3 = GoldilocksExtension;

/// The predicted layout width: `28 + 3·60 + 3·36`.
const PINNED_VALUE_COLUMNS: usize = 316;
/// The predicted constraint count: `4 + 1 + 3·60 + 3·36 + 12 + 4`.
const PINNED_CONSTRAINTS: usize = 309;
/// Base-equivalent cells per permutation: `316 + 3·3`.
const PINNED_CELLS_PER_PERMUTATION: u64 = 325;

fn options() -> ProofOptions {
    GoldilocksCubicProofOptions::with_blowup(2).expect("blowup=2 is valid")
}

fn arenas() -> Vec<Vec<LfmWord>> {
    vec![
        (0..4u64)
            .map(|i| core::array::from_fn(|j| FE::from(1_000 * (i + 1) + j as u64)))
            .collect(),
    ]
}

fn mode_selector(mode: HashMode) -> usize {
    match mode {
        HashMode::Compress => rc::MODE_C,
        HashMode::Transcript => rc::MODE_T,
        HashMode::Leaf => rc::MODE_L,
        HashMode::Permute => rc::MODE_P,
    }
}

fn hash_row(state: [FE; HASH_STATE_FELTS], mode: HashMode) -> Vec<FE> {
    let mut row = vec![FE::zero(); rc::NUM_COLUMNS];
    row[mode_selector(mode)] = FE::one();
    let mut permuted_input = state;
    if mode == HashMode::Permute {
        row[hash::cols::IN0..hash::cols::IN0 + HASH_STATE_FELTS].copy_from_slice(&state);
        row[hash::cols::S8..hash::cols::S8 + 4].copy_from_slice(&state[8..12]);
    } else {
        row[hash::cols::IN0..hash::cols::IN0 + 8].copy_from_slice(&state[0..8]);
        let iv = Rpx256.mode_iv(mode);
        row[hash::cols::S8..hash::cols::S8 + 4].copy_from_slice(&iv);
        permuted_input[8..12].copy_from_slice(&iv);
    }
    let permuted = Rpx256.permute(permuted_input);
    row[hash::cols::OUT0..hash::cols::OUT0 + HASH_STATE_FELTS].copy_from_slice(&permuted);
    fill_rpx_witness(&mut row);
    row
}

fn sample_row() -> Vec<FE> {
    hash_row(
        core::array::from_fn(|i| FE::from(0x9E37_79B9_7F4A_7C15u64.wrapping_mul(i as u64 + 1))),
        HashMode::Permute,
    )
}

fn evaluate(row: &[FE]) -> Vec<FE> {
    let set = HashConstraints::RPX;
    let n = ConstraintSet::<Gl, Gl3>::meta(&set).len();
    let no_ch: Vec<FieldElement<Gl3>> = vec![];
    let offset = FieldElement::<Gl3>::zero();
    let frame = Frame::<Gl, Gl3>::new(vec![TableView::new(vec![row.to_vec()], vec![vec![]])]);
    let ctx =
        TransitionEvaluationContext::new_prover(frame.as_row_frame(), &no_ch, &no_ch, &offset);
    let mut base_out = vec![FE::zero(); n];
    let mut ext_out = vec![FieldElement::<Gl3>::zero(); n];
    let mut folder = ProverEvalFolder::new(&ctx, &mut base_out, &mut ext_out);
    set.eval(&mut folder);
    folder.assert_all_emitted();
    base_out
}

fn violations(row: &[FE]) -> Vec<usize> {
    evaluate(row)
        .iter()
        .enumerate()
        .filter(|(_, v)| **v != FE::zero())
        .map(|(i, _)| i)
        .collect()
}

// =========================================================================
// The layout
// =========================================================================

/// ★ The width, and the headline against RPO: **316 against 436, 27% narrower.**
///
/// Both sides are stated independently — the left is the AIR's own width, the
/// right is the closed form the round schedule implies.
#[test]
fn the_rpx_layout_is_316_value_columns() {
    assert_eq!(rc::NUM_COLUMNS - rc::PREP_WIDTH, PINNED_VALUE_COLUMNS);
    // Three FB rounds at 48 ladder + 12 output, three E rounds at 24 extension
    // intermediates + 12 output, and the M round at nothing.
    assert_eq!(PINNED_VALUE_COLUMNS, 28 + 3 * (48 + 12) + 3 * (24 + 12));
    assert_eq!(rc::PREP_WIDTH, 13, "the preprocessed prefix does not move");
    // ★ The comparison this lane exists to make, asserted rather than narrated.
    const {
        assert!(
            rc::NUM_COLUMNS < hash::rpo_cols::NUM_COLUMNS,
            "RPX must be narrower than RPO — three inverse layers against seven"
        )
    };
    const {
        assert!(
            hash::rpo_cols::NUM_COLUMNS < hash::poseidon_cols::NUM_COLUMNS,
            "and RPO narrower than the Poseidon reference"
        )
    };
}

/// The layout is injective and gapless across a MIXED schedule — the property
/// most at risk when block width depends on the round kind.
#[test]
fn the_rpx_layout_assigns_every_column_exactly_once() {
    assert_eq!(
        (0..HASH_STATE_FELTS)
            .map(|j| rc::out(NUM_ROUNDS - 1, j))
            .collect::<Vec<_>>(),
        (0..HASH_STATE_FELTS)
            .map(|j| hash::cols::OUT0 + j)
            .collect::<Vec<_>>(),
        "the M round's output must BE the frozen OUT columns"
    );

    let mut seen = vec![0usize; rc::NUM_COLUMNS];
    let mut claim = |c: usize| seen[c] += 1;
    for i in 0..HASH_STATE_FELTS {
        claim(hash::cols::IN0 + i);
    }
    for k in 0..4 {
        claim(hash::cols::S8 + k);
    }
    for j in 0..HASH_STATE_FELTS {
        claim(hash::cols::OUT0 + j);
    }
    for r in 0..NUM_ROUNDS {
        if is_final_round(r) {
            continue;
        }
        if is_fb_round(r) {
            for lane in 0..HASH_STATE_FELTS {
                claim(rc::u2(r, lane));
                claim(rc::u3(r, lane));
                claim(rc::y2(r, lane));
                claim(rc::y3(r, lane));
            }
        } else {
            for e in 0..EXT_ELEMENTS {
                for k in 0..EXT_DEGREE {
                    claim(rc::t2(r, e, k));
                    claim(rc::t3(r, e, k));
                }
            }
        }
        for j in 0..HASH_STATE_FELTS {
            claim(rc::out(r, j));
        }
    }
    for (c, &n) in seen.iter().enumerate().skip(rc::PREP_WIDTH) {
        assert_eq!(
            n, 1,
            "value column {c} is claimed {n} times, want exactly 1"
        );
    }
    for (c, &n) in seen.iter().enumerate().take(rc::PREP_WIDTH) {
        assert_eq!(n, 0, "preprocessed column {c} must not be claimed");
    }
}

#[test]
fn rpx_adds_no_bus_interactions() {
    assert_eq!(hash::bus_interactions(HasherKind::Rpx).len(), 6);
    assert_eq!(hash::num_columns(HasherKind::Rpx), rc::NUM_COLUMNS);
}

// =========================================================================
// The degree bound
// =========================================================================

/// ★ Degree 3 across all three round kinds — including the EXTENSION seventh
/// power, whose lowering is the one genuinely new thing in this chip.
#[test]
fn every_rpx_constraint_is_degree_three_or_less() {
    let set = HashConstraints::RPX;
    let meta = ConstraintSet::<Gl, Gl3>::meta(&set);
    let n = meta.len();
    assert_eq!(n, PINNED_CONSTRAINTS);
    // 4 capacity + 1 mode + FB 5/lane + E 3 per extension op on 4 triples +
    // M 1/lane + the shared unread-IN pins.
    assert_eq!(
        PINNED_CONSTRAINTS,
        4 + 1 + 3 * (5 * 12) + 3 * (3 * 3 * 4) + 12 + super::chips::hash::NUM_UNREAD_INPUT_PINS
    );
    for (i, m) in meta.iter().enumerate() {
        assert_eq!(m.constraint_idx, i, "meta must be dense and idx-ordered");
        assert_eq!(m.kind, RootKind::Base);
    }

    let mut cb = CaptureBuilder::<Gl, Gl3>::new();
    set.eval(&mut cb);
    let (_prog, degrees) = cb.finish(num_base_from_meta(&meta));
    assert_eq!(degrees.len(), n, "one emit per constraint");
    let mut emitted: Vec<usize> = degrees.iter().map(|&(idx, _)| idx).collect();
    emitted.sort_unstable();
    assert!(emitted.iter().enumerate().all(|(i, &idx)| i == idx));

    let declared = ConstraintSet::<Gl, Gl3>::max_degree(&set);
    assert_eq!(declared, 3, "the wrap's blowup 2 depends on this staying 3");
    for &(idx, measured) in &degrees {
        assert!(
            measured <= declared,
            "constraint {idx}: measured degree {measured} EXCEEDS declared {declared}"
        );
    }
    assert_eq!(degrees.iter().map(|&(_, d)| d).max(), Some(3));
}

// =========================================================================
// Satisfaction and rejection
// =========================================================================

#[test]
fn a_real_rpx_row_satisfies_every_constraint() {
    for mode in [
        HashMode::Compress,
        HashMode::Transcript,
        HashMode::Leaf,
        HashMode::Permute,
    ] {
        let state: [FE; HASH_STATE_FELTS] = core::array::from_fn(|i| FE::from(7 * i as u64 + 1));
        let row = hash_row(state, mode);
        assert_eq!(
            violations(&row),
            Vec::<usize>::new(),
            "an honest row ({mode:?}) must satisfy every constraint"
        );
    }
}

/// Perturbing any single witness column fires — one per structural role, so
/// both round kinds and the chaining are covered.
#[test]
fn perturbing_one_column_is_rejected() {
    let base = sample_row();
    assert!(
        violations(&base).is_empty(),
        "the unperturbed row is honest"
    );

    let cases: [(&str, usize); 6] = [
        ("FB u2 (round 0, lane 5)", rc::u2(0, 5)),
        ("FB y3 (round 2, lane 9)", rc::y3(2, 9)),
        ("FB output (round 4, lane 3)", rc::out(4, 3)),
        ("E t2 (round 1, element 2, coeff 1)", rc::t2(1, 2, 1)),
        ("E t3 (round 3, element 0, coeff 2)", rc::t3(3, 0, 2)),
        ("E output (round 5, lane 7)", rc::out(5, 7)),
    ];
    for (label, col) in cases {
        let mut row = base.clone();
        row[col] = &row[col] + FE::one();
        assert!(
            !violations(&row).is_empty(),
            "perturbing {label} (column {col}) must fire at least one constraint"
        );
    }
}

/// ★ The EXTENSION fold is load-bearing: a consistent `t²`/`t³` ladder for the
/// wrong extension element must still be rejected, because `x ↦ x^7` permutes
/// `GF(p³)` and the output is pinned to the seventh power of the round input.
#[test]
fn a_consistent_extension_ladder_for_the_wrong_value_is_rejected() {
    use super::rpx::cubic_ext;
    let mut row = sample_row();
    let (r, e) = (1usize, 2usize);
    let wrong: cubic_ext::Ext =
        core::array::from_fn(|k| &row[rc::t2(r, e, k)] + FE::from(k as u64 + 1));
    let t3 = cubic_ext::mul(&wrong, &wrong);
    for k in 0..EXT_DEGREE {
        row[rc::t2(r, e, k)] = wrong[k];
        row[rc::t3(r, e, k)] = t3[k];
    }
    assert!(
        !violations(&row).is_empty(),
        "a consistent extension ladder for the wrong value must be rejected"
    );
}

/// ★ Padding by zero must hold in ALL THREE round kinds — the FB rounds' RPO
/// argument, the E rounds' extension products, and the M round's linear map.
#[test]
fn the_all_zero_padding_row_satisfies_every_constraint() {
    let row = vec![FE::zero(); rc::NUM_COLUMNS];
    assert_eq!(violations(&row), Vec::<usize>::new());
}

#[test]
fn a_padding_row_claiming_to_be_real_is_rejected() {
    for mode in [
        HashMode::Compress,
        HashMode::Transcript,
        HashMode::Leaf,
        HashMode::Permute,
    ] {
        let mut row = vec![FE::zero(); rc::NUM_COLUMNS];
        row[mode_selector(mode)] = FE::one();
        assert!(!violations(&row).is_empty(), "{mode:?}");
    }
}

/// The three domains separate under RPX exactly as under RPO — same prefix
/// emitter, so this is a check that sharing it did not lose anything.
#[test]
fn a_row_carrying_another_modes_capacity_is_rejected() {
    let two_cell = [HashMode::Compress, HashMode::Transcript, HashMode::Leaf];
    let state: [FE; HASH_STATE_FELTS] = core::array::from_fn(|i| {
        if i < 8 {
            FE::from(5 * i as u64 + 3)
        } else {
            FE::zero()
        }
    });
    for claimed in two_cell {
        for carried in two_cell {
            if claimed == carried {
                continue;
            }
            let mut row = hash_row(state, carried);
            row[mode_selector(carried)] = FE::zero();
            row[mode_selector(claimed)] = FE::one();
            assert!(
                !violations(&row).is_empty(),
                "a {claimed:?} row carrying the {carried:?} capacity must be rejected"
            );
        }
    }
}

// =========================================================================
// Prove, verify, and end to end
// =========================================================================

#[test]
fn the_rpx_chip_proves_and_verifies() {
    let opts = options();
    let program = trivial_program();
    let artifacts = build_artifacts_with_hasher(&program, &opts, HasherKind::Rpx);
    let proved = lfm_prove_with_hasher(&program, &artifacts, &arenas(), &opts, HasherKind::Rpx)
        .expect("proving under RPX must succeed");
    assert!(verify_against(
        &artifacts.roots,
        &artifacts.program_id,
        artifacts.keccak_rnd_chunks,
        &proved.proof,
        &proved.public_words,
        &opts,
        artifacts.hasher,
        artifacts.chip_set,
    ));
}

/// ⚠ RPO and RPX share constants, an MDS and three of seven rounds, so they are
/// the most confusable pair in the machine. A proof under one must not verify
/// under the other, in both directions.
#[test]
fn an_rpx_proof_does_not_verify_under_rpo_or_the_reverse() {
    let opts = options();
    let program = trivial_program();
    for (proved_under, verified_under) in [
        (HasherKind::Rpx, HasherKind::Rpo),
        (HasherKind::Rpo, HasherKind::Rpx),
    ] {
        let artifacts = build_artifacts_with_hasher(&program, &opts, proved_under);
        let proved = lfm_prove_with_hasher(&program, &artifacts, &arenas(), &opts, proved_under)
            .expect("prove");
        assert!(
            !verify_against(
                &artifacts.roots,
                &artifacts.program_id,
                artifacts.keccak_rnd_chunks,
                &proved.proof,
                &proved.public_words,
                &opts,
                verified_under,
                artifacts.chip_set,
            ),
            "a proof made under {proved_under:?} must not verify under {verified_under:?}"
        );
    }
}

/// ★★★ The same end-to-end that RPO passes, under RPX — and on the SAME
/// unmodified verifier program, which is the point of the exercise.
#[test]
fn the_machine_verifies_a_fixture_fri_proof_end_to_end_under_rpx() {
    let opts = options();
    let program = fri_toy_program();
    let artifacts = build_artifacts_with_hasher(&program, &opts, HasherKind::Rpx);
    let inner = fixture_prove_with_hasher(HasherKind::Rpx);
    let proved = lfm_prove_with_hasher(
        &program,
        &artifacts,
        &[inner.commitments.clone(), inner.openings.clone()],
        &opts,
        HasherKind::Rpx,
    )
    .expect("the machine must accept an RPX-committed inner proof");
    assert_eq!(proved.public_words[0].1, inner.commitments[0]);
    assert!(verify_against(
        &artifacts.roots,
        &artifacts.program_id,
        artifacts.keccak_rnd_chunks,
        &proved.proof,
        &proved.public_words,
        &opts,
        artifacts.hasher,
        artifacts.chip_set,
    ));
}

#[test]
fn the_hasher_tags_are_stable_and_distinct() {
    assert_eq!(HasherKind::Rpo.as_tag(), 3);
    assert_eq!(HasherKind::Rpx.as_tag(), 4);
}

// =========================================================================
// The measurement
// =========================================================================

/// ★★ Base-equivalent cells per permutation, on the same census instrument that
/// measured the RPO, Poseidon and BLAKE3 columns — which is what makes the
/// three-way comparison a comparison rather than three separate claims.
#[test]
fn the_measured_cells_per_permutation_match_the_pinned_prediction() {
    let program = trivial_program();
    let census = lfm_chip_census_with_hasher(&program, HasherKind::Rpx);
    let chip = census
        .iter()
        .find(|c| c.name == "LFM_HASH")
        .expect("LFM_HASH is slot-registered");
    assert_eq!(chip.main_cols, PINNED_VALUE_COLUMNS);
    assert_eq!(
        chip.aux_cols, 3,
        "six LfmMem interactions ⇒ three aux columns"
    );
    let per_permutation = chip.main_cols as u64 + 3 * chip.aux_cols as u64;
    assert_eq!(per_permutation, PINNED_CELLS_PER_PERMUTATION);

    // ★ The three-way ladder, measured on one instrument in one test.
    let cells = |k: HasherKind| -> u64 {
        let c = lfm_chip_census_with_hasher(&program, k);
        let h = c.iter().find(|c| c.name == "LFM_HASH").expect("LFM_HASH");
        h.main_cols as u64 + 3 * h.aux_cols as u64
    };
    let rpx = cells(HasherKind::Rpx);
    let rpo = cells(HasherKind::Rpo);
    let poseidon = cells(HasherKind::Poseidon);
    assert_eq!((rpx, rpo, poseidon), (325, 445, 621));
    assert!(rpx < rpo && rpo < poseidon);
    // And all three an order of magnitude under the slot-11 BLAKE3 incumbent,
    // whose 4,946 is quoted from the wrap census rather than recomputed here.
    const BLAKE3_CELLS_PER_COMPRESSION: u64 = 4_946;
    assert!(BLAKE3_CELLS_PER_COMPRESSION / rpx >= 15);
}
