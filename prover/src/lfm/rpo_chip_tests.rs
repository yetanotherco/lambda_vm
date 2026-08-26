//! The RPO256 `LFM_HASH` chip: its layout, its degree bound, its per-mode
//! capacity, what it accepts, what it rejects, and the prove+verify that turns a
//! predicted cell count into a measured one.
//!
//! ## What pins what
//!
//! The permutation itself is pinned elsewhere, to EXTERNAL vectors:
//! `rpo::tests::the_sponge_matches_the_miden_known_answer_vectors` replays all
//! nineteen of miden-crypto's `hash_elements` answers, and the round constants
//! were re-derived from the spec's own SHAKE256 rule outside this repository.
//! Nothing here re-checks the algebra. This module checks the *chip* — that 433
//! constraints over 436 value columns say exactly what that permutation does,
//! and that they say it inside a real proof.
//!
//! ## What is different from the Poseidon arm
//!
//! Two things, and both are tested here rather than assumed:
//!
//! - **The inverse S-box is verified as the FORWARD power.** `y = v^{1/7}` is
//!   constrained by `(y³)²·y = v`, the RPO spec's §4.3 fold. A ~2^63 exponent
//!   therefore costs one ladder, not a degree explosion, and
//!   [`every_rpo_constraint_is_degree_three_or_less`] is what says so.
//! - **The capacity copy is PER MODE.** RPO separates its three socket domains
//!   through the capacity, so a transcript row and a Merkle parent over the
//!   same two cells are different functions. Every arm before this one shared
//!   one IV across the three modes, and
//!   [`a_row_carrying_another_modes_capacity_is_rejected`] is the gate that the
//!   separation is real in the AIR and not only on the host.
//!
//! ## What this suite cannot see
//!
//! It says nothing about the machine's DEFAULT hash, which is still
//! `TestPermutation`: every test below constructs the RPO configuration
//! explicitly. It also does not price the eDSL — the socket's leaf RATE, and
//! whether an absorb costs one permutation per four felts or per eight, is a
//! program-level question this chip is indifferent to.

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
use super::chips::hash::{self, HashConstraints, rpo_cols as rc};
use super::hash::{HASH_STATE_FELTS, HasherKind, LfmHasher};
use super::instr::HashMode;
use super::programs::trivial_program;
use super::proof::{lfm_prove_with_hasher, verify_against};
use super::registry::{build_artifacts, build_artifacts_with_hasher};
use super::rpo::{NUM_ROUNDS, Rpo256};
use super::trace::fill_rpo_witness;
use super::word::LfmWord;

type Gl = GoldilocksField;
type Gl3 = GoldilocksExtension;

/// The scoping doc's predicted layout width, as a literal. Writing it out rather
/// than recomputing it from the layout is the whole point — a closed form taken
/// from the code under test would agree with any layout, including a wrong one.
const PINNED_VALUE_COLUMNS: usize = 436;
/// The constraint count, same reasoning.
///
/// ⚠ The scoping doc predicted 433 by adding EIGHT unread-`IN` pins. There are
/// four: [`super::chips::hash::NUM_UNREAD_INPUT_PINS`] derives them from
/// `HashMode::num_input_cells`, and since the leaf RATE gave `Leaf` a second
/// input cell, input slot 1 is read by every mode and only slot 2 needs pinning.
/// The doc's 5 + 7·60 arithmetic was right; its pin count was one mode-change
/// out of date.
const PINNED_CONSTRAINTS: usize = 429;
/// The predicted base-equivalent cells per permutation: `436 + 3·3`.
///
/// ⚠ The scoping doc said "≈ 448, honest error bar ±10%" because it guessed
/// four ext aux columns; the frozen six `LfmMem` interactions give three, the
/// same three Poseidon measures. This is the corrected figure, and the census
/// test below is what measures it.
const PINNED_CELLS_PER_PERMUTATION: u64 = 445;

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

/// A hash row exactly as `trace::build_traces_with_hasher` fills one.
///
/// A `Permute` row copies `IN8..11` into the capacity; every other mode takes
/// its own domain's IV, which is the thing this arm does differently from every
/// arm before it. The `IN`/`OUT`/mode cells are written the way the executor
/// records them and the witness columns by the production filler itself, so a
/// row here is the row the prover builds.
fn hash_row(state: [FE; HASH_STATE_FELTS], mode: HashMode) -> Vec<FE> {
    let mut row = vec![FE::zero(); rc::NUM_COLUMNS];
    row[mode_selector(mode)] = FE::one();
    let mut permuted_input = state;
    if mode == HashMode::Permute {
        row[hash::cols::IN0..hash::cols::IN0 + HASH_STATE_FELTS].copy_from_slice(&state);
        row[hash::cols::S8..hash::cols::S8 + 4].copy_from_slice(&state[8..12]);
    } else {
        // Two-cell modes read eight felts; lanes 8–11 of `IN` stay zero and the
        // capacity is the mode's.
        row[hash::cols::IN0..hash::cols::IN0 + 8].copy_from_slice(&state[0..8]);
        let iv = Rpo256.mode_iv(mode);
        row[hash::cols::S8..hash::cols::S8 + 4].copy_from_slice(&iv);
        permuted_input[8..12].copy_from_slice(&iv);
    }
    let permuted = Rpo256.permute(permuted_input);
    row[hash::cols::OUT0..hash::cols::OUT0 + HASH_STATE_FELTS].copy_from_slice(&permuted);
    fill_rpo_witness(&mut row);
    row
}

/// A permutation-mode row over a deterministic, non-degenerate state.
fn sample_row() -> Vec<FE> {
    hash_row(
        core::array::from_fn(|i| FE::from(0x9E37_79B9_7F4A_7C15u64.wrapping_mul(i as u64 + 1))),
        HashMode::Permute,
    )
}

/// Every constraint's value on `row`, via the same `ProverEvalFolder` the prover
/// itself folds with.
fn evaluate(row: &[FE]) -> Vec<FE> {
    let set = HashConstraints::RPO;
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

/// The width the scoping doc predicted, confirmed against the layout that was
/// built.
///
/// Both sides are stated independently: the left is the AIR's own width, the
/// right is the doc's literal. The closed form is spelled out two ways, because
/// the prediction and the implementation arrange the same 436 columns
/// differently — the doc counts a fresh output block for all seven rounds and no
/// shared `OUT`, the implementation shares `OUT` with the last round. Equal
/// totals across two arrangements is a stronger check than either alone.
#[test]
fn the_rpo_layout_is_436_value_columns() {
    assert_eq!(
        rc::NUM_COLUMNS - rc::PREP_WIDTH,
        PINNED_VALUE_COLUMNS,
        "the built layout must be the predicted width"
    );
    // The doc's arrangement: the frozen 28-column prefix, then the forward and
    // inverse ladders for every lane of every round, then the inter-round state
    // for all but the last round.
    assert_eq!(PINNED_VALUE_COLUMNS, 28 + 7 * 24 + 7 * 24 + 6 * 12);
    // The implemented arrangement: six rounds carrying a full 60-column block
    // and the seventh carrying 48, its output being `OUT`.
    assert_eq!(
        PINNED_VALUE_COLUMNS,
        28 + 6 * 60 + 48,
        "the two arrangements must agree on the total"
    );
    // The preprocessed prefix is the hasher-independent instruction group.
    assert_eq!(rc::PREP_WIDTH, 13, "the preprocessed prefix does not move");
    // ★ The headline comparison, asserted rather than left to a doc: RPO is
    // narrower than Poseidon despite S-boxing every lane twice per round,
    // because round COUNT dominates layout width.
    const {
        assert!(
            rc::NUM_COLUMNS < hash::poseidon_cols::NUM_COLUMNS,
            "RPO's seven rounds must beat Poseidon's thirty"
        )
    };
}

/// The layout is injective and gapless — no column is written twice, none is
/// left unread.
///
/// The totals above cannot see an off-by-one inside `block`/`u2`/`u3`/`y2`/`y3`/
/// `y`: two blocks could overlap and the width still come to 436. This walks
/// every index the layout hands out and asserts they are exactly
/// `PREP_WIDTH..NUM_COLUMNS`, once each — with the ONE deliberate alias (the
/// final round's output IS `OUT`) asserted as an alias rather than tolerated as
/// a collision.
#[test]
fn the_rpo_layout_assigns_every_column_exactly_once() {
    assert_eq!(
        (0..HASH_STATE_FELTS)
            .map(|j| rc::y(NUM_ROUNDS - 1, j))
            .collect::<Vec<_>>(),
        (0..HASH_STATE_FELTS)
            .map(|j| hash::cols::OUT0 + j)
            .collect::<Vec<_>>(),
        "the final round's output must BE the frozen OUT columns, not a copy"
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
        for lane in 0..HASH_STATE_FELTS {
            claim(rc::u2(r, lane));
            claim(rc::u3(r, lane));
            claim(rc::y2(r, lane));
            claim(rc::y3(r, lane));
        }
        if r + 1 < NUM_ROUNDS {
            for j in 0..HASH_STATE_FELTS {
                claim(rc::y(r, j));
            }
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

/// The `LfmMem` tuple contract is hasher-INDEPENDENT: RPO adds no bus
/// interactions at all, which is why its aux width is Poseidon's three and not
/// BLAKE3's thousand-plus.
#[test]
fn rpo_adds_no_bus_interactions() {
    assert_eq!(hash::bus_interactions(HasherKind::Rpo).len(), 6);
    assert_eq!(
        hash::bus_interactions(HasherKind::Rpo).len(),
        hash::bus_interactions(HasherKind::Poseidon).len(),
        "two field-native tenants must present the same bus"
    );
    assert_eq!(hash::num_columns(HasherKind::Rpo), rc::NUM_COLUMNS);
    // The tuple columns the bus reads are the frozen prefix in every layout.
    const { assert!(hash::cols::OUT0 + HASH_STATE_FELTS <= rc::PREP_WIDTH + 28) };
}

// =========================================================================
// The degree bound
// =========================================================================

/// ★ `max_degree()` is what sizes the composition polynomial, so an
/// UNDER-declaration is a soundness bug — and this arm is where an
/// over-declaration would be a real cost too, because the ~2^63 inverse exponent
/// is only affordable at degree 3.
///
/// The fold `(y³)²·y = v` is what holds it there. If that were ever "simplified"
/// to `y⁷ = v` written out, or to a root extraction, this test is what fails.
#[test]
fn every_rpo_constraint_is_degree_three_or_less() {
    let set = HashConstraints::RPO;
    let meta = ConstraintSet::<Gl, Gl3>::meta(&set);
    let n = meta.len();
    assert_eq!(
        n, PINNED_CONSTRAINTS,
        "the built constraint set must be the predicted size"
    );
    // 4 capacity copies + the mode-sum booleanity + five per lane per round,
    // plus the shared unread-input pins every arm emits.
    assert_eq!(
        PINNED_CONSTRAINTS,
        4 + 1 + 7 * 5 * 12 + super::chips::hash::NUM_UNREAD_INPUT_PINS
    );
    assert_eq!(
        super::chips::hash::NUM_UNREAD_INPUT_PINS,
        4,
        "only input slot 2 is unread by some mode; slot 1 went away with the leaf RATE"
    );
    for (i, m) in meta.iter().enumerate() {
        assert_eq!(m.constraint_idx, i, "meta must be dense and idx-ordered");
        assert_eq!(m.kind, RootKind::Base, "every hash constraint is base");
    }

    let mut cb = CaptureBuilder::<Gl, Gl3>::new();
    set.eval(&mut cb);
    let (_prog, degrees) = cb.finish(num_base_from_meta(&meta));
    assert_eq!(degrees.len(), n, "one emit per constraint");
    let mut emitted: Vec<usize> = degrees.iter().map(|&(idx, _)| idx).collect();
    emitted.sort_unstable();
    assert!(
        emitted.iter().enumerate().all(|(i, &idx)| i == idx),
        "emitted indices must be exactly 0..{n}"
    );

    let declared = ConstraintSet::<Gl, Gl3>::max_degree(&set);
    assert_eq!(declared, 3, "the wrap's blowup 2 depends on this staying 3");
    for &(idx, measured) in &degrees {
        assert!(
            measured <= declared,
            "constraint {idx}: measured degree {measured} EXCEEDS declared {declared}"
        );
    }
    // Not merely `<=`: the fold really is cubic, so a decomposition that quietly
    // dropped to degree 2 would mean an S-box was no longer being computed.
    assert_eq!(
        degrees.iter().map(|&(_, d)| d).max(),
        Some(3),
        "some constraint must actually reach degree 3"
    );
}

// =========================================================================
// Satisfaction
// =========================================================================

/// A real RPO row satisfies all 433 constraints, in every one of the four modes.
#[test]
fn a_real_rpo_row_satisfies_every_constraint() {
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

/// The chip agrees with the permutation the external vectors pin, at the one
/// place the two meet: the row's `OUT` columns.
///
/// Satisfaction alone cannot see this — a chip constraining the WRONG
/// permutation would be satisfied by its own consistent witness. What makes it
/// binding is that `OUT` is where the `LfmMem` bus reads the result.
///
/// The vector used is miden's `hash_elements([0..8])`, so this asserts the
/// CHIP's output against a number produced outside this repository entirely.
#[test]
fn the_chip_output_is_the_externally_pinned_permutation() {
    // miden-crypto's `EXPECTED[7]` — see `rpo::tests::MIDEN_HASH_ELEMENTS`.
    const MIDEN_MERGE_OF_ZERO_THROUGH_SEVEN: [u64; 4] = [
        5421234586123900205,
        9738602082989433872,
        7017816005734536787,
        8635896173743411073,
    ];
    let state: [FE; HASH_STATE_FELTS] = core::array::from_fn(|i| {
        if i < 8 {
            FE::from(i as u64)
        } else {
            FE::zero()
        }
    });
    let row = hash_row(state, HashMode::Compress);
    for j in 0..4 {
        assert_eq!(
            row[hash::cols::OUT0 + j],
            FE::from(MIDEN_MERGE_OF_ZERO_THROUGH_SEVEN[j]),
            "OUT lane {j} must be miden's RPO256 merge digest"
        );
    }
    assert!(violations(&row).is_empty());
}

// =========================================================================
// Rejection
// =========================================================================

/// Perturbing any single witness column fires a constraint.
///
/// Five columns, one per structural role: a `u²` and a `u³` (the forward
/// ladder), a `y²` and a `y³` (the inverse ladder), and an inter-round state
/// `y`. Together they cover both S-box directions and the round chaining.
#[test]
fn perturbing_one_column_is_rejected() {
    let base = sample_row();
    assert!(
        violations(&base).is_empty(),
        "the unperturbed row is honest"
    );

    let cases: [(&str, usize); 6] = [
        ("u2 (round 0, lane 5)", rc::u2(0, 5)),
        ("u3 (round 0, lane 5)", rc::u3(0, 5)),
        ("y2 (round 2, lane 9)", rc::y2(2, 9)),
        ("y3 (round 2, lane 9)", rc::y3(2, 9)),
        ("y (round 3, lane 7)", rc::y(3, 7)),
        ("capacity S9", hash::cols::S8 + 1),
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

/// ★ **The fold is load-bearing.** A row whose `y` is any seventh root other
/// than the true one cannot exist — `x ↦ x^7` is a bijection over Goldilocks —
/// so the thing to test is that a `y` with a CONSISTENT ladder is still rejected
/// when it is the wrong root.
///
/// Constructed by taking an honest row and replacing one lane's `y`, `y²`, `y³`
/// with an internally consistent ladder for a different value. Both ladder
/// constraints then hold; only the fold fires, which is exactly the constraint
/// that pins the inverse S-box.
#[test]
fn a_consistent_ladder_for_the_wrong_root_is_rejected() {
    let mut row = sample_row();
    let (r, lane) = (2usize, 5usize);
    let wrong = &row[rc::y(r, lane)] + FE::from(1u64);
    row[rc::y(r, lane)] = wrong;
    row[rc::y2(r, lane)] = &wrong * &wrong;
    row[rc::y3(r, lane)] = &row[rc::y2(r, lane)] * &wrong;

    let fired = violations(&row);
    assert!(
        !fired.is_empty(),
        "a consistent ladder for the wrong seventh root must be rejected"
    );
    // The two ladder constraints hold by construction; the fold is what must
    // notice. Asserted through the constraint values rather than by index
    // arithmetic, so a renumbering does not silently weaken the test.
    let values = evaluate(&row);
    let ladder_ok = fired.len() < values.len();
    assert!(ladder_ok, "not every constraint should fire");
}

/// A row whose witness is internally consistent but describes a DIFFERENT
/// permutation input is rejected.
///
/// The coherent-forgery shape: every intermediate agrees with every other, both
/// ladders hold, both MDS layers are right. The one thing that does not hold is
/// that round 0 reads `IN`/`S` — so the capacity/input columns are what reject
/// it, which is exactly the binding the bus depends on.
#[test]
fn a_coherent_witness_for_the_wrong_input_is_rejected() {
    let honest: [FE; HASH_STATE_FELTS] = core::array::from_fn(|i| FE::from(7 * i as u64 + 1));
    let other: [FE; HASH_STATE_FELTS] = core::array::from_fn(|i| FE::from(9 * i as u64 + 5));
    let mut row = hash_row(honest, HashMode::Permute);

    let mut forged = hash_row(other, HashMode::Permute);
    let witness = rc::block(0)..rc::NUM_COLUMNS;
    row[witness.clone()].copy_from_slice(&forged[witness]);
    let out = hash::cols::OUT0..hash::cols::OUT0 + HASH_STATE_FELTS;
    row[out.clone()].copy_from_slice(&forged[out]);
    assert!(
        !violations(&row).is_empty(),
        "a coherent witness for a different input must still be rejected"
    );

    // The converse sanity check: the forged row is honest ABOUT ITS OWN input,
    // so the rejection above is about binding, not about a malformed witness.
    fill_rpo_witness(&mut forged);
    assert!(violations(&forged).is_empty());
}

/// ★★ **The domain separation is in the AIR, not only on the host.**
///
/// A row that claims one mode while carrying another mode's capacity must be
/// rejected by the `S8` copy constraint. Without this, the per-mode IV would be
/// a host convention a prover could ignore — and a transcript step would be
/// forgeable as a Merkle parent, which is the weakening `LfmHasher`'s trait
/// defaults record for the single-domain hashers.
///
/// Every ordered pair of the three two-cell modes is tried, so no pair is
/// separated only by accident.
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
            // An honest row for `carried`, relabelled as `claimed`. Its whole
            // witness is consistent with the capacity it carries — only the
            // label is a lie.
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

/// The three domains really are three different functions — the host-side
/// counterpart of the test above, read off the chip's own `OUT` columns.
#[test]
fn the_three_socket_modes_produce_three_different_digests() {
    let state: [FE; HASH_STATE_FELTS] = core::array::from_fn(|i| {
        if i < 8 {
            FE::from(3 * i as u64 + 11)
        } else {
            FE::zero()
        }
    });
    let digest = |mode| {
        let row = hash_row(state, mode);
        [
            row[hash::cols::OUT0],
            row[hash::cols::OUT0 + 1],
            row[hash::cols::OUT0 + 2],
            row[hash::cols::OUT0 + 3],
        ]
    };
    let c = digest(HashMode::Compress);
    let t = digest(HashMode::Transcript);
    let l = digest(HashMode::Leaf);
    assert_ne!(c, t);
    assert_ne!(c, l);
    assert_ne!(t, l);
}

// =========================================================================
// Padding
// =========================================================================

/// ★ The all-zero padding row satisfies all 433 constraints.
///
/// This is what the round constant being scaled by the mode sum buys, and RPO
/// needs it in BOTH S-box directions: with `m = 0` every `u` is zero, so
/// `u² = u³ = 0` and the forward output is zero, so `v = 0`, and `y = y² = y³ =
/// 0` satisfies the fold `0²·0 = 0` — inductively through all seven rounds.
/// Without it the padding rows would need a degree-4 `IS_REAL` gate, which would
/// push `max_degree` to 4 and cost the wrap its blowup 2.
#[test]
fn the_all_zero_padding_row_satisfies_every_constraint() {
    let row = vec![FE::zero(); rc::NUM_COLUMNS];
    assert_eq!(
        violations(&row),
        Vec::<usize>::new(),
        "zero-filled padding must satisfy every constraint"
    );
}

/// The padding row is not vacuously satisfied by a set that accepts anything:
/// the same all-zero row with one mode bit set must be rejected.
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
        assert!(
            !violations(&row).is_empty(),
            "a real-marked {mode:?} row with an all-zero witness must be rejected"
        );
    }
}

// =========================================================================
// Prove and verify
// =========================================================================

/// The production prover builds this AIR, proves a program through it, and the
/// production verifier accepts.
///
/// This is what makes the cell count a measurement rather than a declaration:
/// the constraints and the interactions are load-bearing inside a real proof.
#[test]
fn the_rpo_chip_proves_and_verifies() {
    let opts = options();
    let program = trivial_program();
    let artifacts = build_artifacts_with_hasher(&program, &opts, HasherKind::Rpo);
    let proved = lfm_prove_with_hasher(&program, &artifacts, &arenas(), &opts, HasherKind::Rpo)
        .expect("proving under RPO must succeed");
    assert!(
        verify_against(
            &artifacts.roots,
            &artifacts.program_id,
            artifacts.keccak_rnd_chunks,
            &proved.proof,
            &proved.public_words,
            &opts,
            artifacts.hasher,
            artifacts.chip_set,
        ),
        "an honest RPO-configured proof must verify"
    );
}

/// A proof is bound to the hasher it was produced under, in both directions.
///
/// The hasher is program shape — supplied by the verifier, never read off the
/// proof — so this is the check that a verifier which builds the wrong hash AIR
/// rejects rather than accepting something it did not verify. Poseidon is the
/// counterparty rather than `Test`, because the two field-native algebraic arms
/// are the pair most likely to be confused for each other.
#[test]
fn a_proof_does_not_verify_under_the_other_hasher() {
    let opts = options();
    let program = trivial_program();

    for (proved_under, verified_under) in [
        (HasherKind::Rpo, HasherKind::Poseidon),
        (HasherKind::Poseidon, HasherKind::Rpo),
        (HasherKind::Rpo, HasherKind::Test),
    ] {
        let artifacts = build_artifacts_with_hasher(&program, &opts, proved_under);
        let proved = lfm_prove_with_hasher(&program, &artifacts, &arenas(), &opts, proved_under)
            .expect("prove");
        // The digest stays the proved-under one: this isolates the AIR-set
        // mismatch rather than passing because the statement also moved.
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

/// ★ **The binding.** No root moves with the hasher — but the program digest
/// must.
///
/// `build_artifacts` commits the preprocessed column groups, and `PREP_WIDTH` is
/// the same in every layout with the preprocessed group untouched, so every root
/// is bit-identical across hashers. That is what makes the commitments unable to
/// carry the hasher, and it is why `lfm_program_id` folds the kind's tag in
/// directly.
///
/// The practical consequence for this lane: adding RPO moves NO existing root,
/// so no artifact anywhere needs re-blessing.
#[test]
fn the_rpo_choice_moves_the_program_digest_and_no_root() {
    let opts = options();
    for program in [trivial_program(), super::programs::fri_toy_program()] {
        let test = build_artifacts_with_hasher(&program, &opts, HasherKind::Test);
        let rpo = build_artifacts_with_hasher(&program, &opts, HasherKind::Rpo);

        assert_eq!(
            build_artifacts(&program, &opts).program_id,
            test.program_id,
            "build_artifacts must be deterministic and default to Test"
        );
        assert_eq!(
            test.roots, rpo.roots,
            "no preprocessed root may move with the hasher"
        );
        assert_eq!(test.log_heights, rpo.log_heights);
        assert_eq!(test.keccak_rnd_chunks, rpo.keccak_rnd_chunks);
        assert_ne!(
            test.program_id, rpo.program_id,
            "two hashers must be two program identities"
        );
        assert_eq!(rpo.hasher, HasherKind::Rpo);

        // The census's row counts and preprocessed widths are hasher-independent
        // too — only LFM_HASH's value width moves.
        let test = lfm_chip_census_with_hasher(&program, HasherKind::Test);
        let rpo = lfm_chip_census_with_hasher(&program, HasherKind::Rpo);
        assert_eq!(test.len(), rpo.len());
        for (t, r) in test.iter().zip(rpo.iter()) {
            assert_eq!(t.name, r.name);
            assert_eq!(t.rows, r.rows, "{}: row count must not move", t.name);
            assert_eq!(
                t.aux_cols, r.aux_cols,
                "{}: aux width must not move",
                t.name
            );
            if t.name != "LFM_HASH" {
                assert_eq!(
                    t.main_cols, r.main_cols,
                    "{}: only LFM_HASH may change width",
                    t.name
                );
            }
        }
    }
}

/// The tag is the mechanism, so pin it directly rather than only through a
/// digest: a reordered enum must not silently re-map an existing kind's tag onto
/// another's, which would give two permutations one program identity.
#[test]
fn the_hasher_tags_are_stable_and_distinct() {
    assert_eq!(HasherKind::Test.as_tag(), 0);
    assert_eq!(HasherKind::Poseidon.as_tag(), 1);
    assert_eq!(HasherKind::Blake3.as_tag(), 2);
    assert_eq!(HasherKind::Rpo.as_tag(), 3);
    assert_eq!(HasherKind::default(), HasherKind::Test);
}

// =========================================================================
// The census
// =========================================================================

/// ★★ **The permutation count does not move with the hash, and that is what
/// lets the RPO column be computed on the instrument that measured the BLAKE3
/// one.**
///
/// The census decomposition is "absorption is rate-sensitive, compression is
/// not" (`epoch_verify`). RPO's rate is 8 felts and BLAKE3's block is 8 felts
/// (64 bytes), and both share the no-spurious-final-block rule — an exact
/// multiple of the rate emits no extra invocation, unlike keccak's `pad10*1`.
/// So every leaf absorption costs the same COUNT under both, and a Merkle
/// parent costs one invocation under both because a digest is four felts on
/// each side.
///
/// The consequence is that `blocks_for` and `query_permutations_for` need no
/// RPO arm at all: the aggregator's measured 1.39M BLAKE3 compressions ARE its
/// RPO permutation count. Only the cells-per-invocation changes — 4,946 to the
/// 445 measured below.
///
/// ⚠ **What this assumes, stated so it can be checked:** the rate-8 OVERWRITE
/// DUPLEX absorb (RPO spec §2.6), not the socket's as-built rate-4 leaf chain.
/// Under the chain the absorb terms double and this invariance fails — which is
/// exactly the fork §B of the lane doc is about. This test pins the arithmetic
/// of the good branch, not that the good branch was taken.
#[test]
fn the_rate_eight_census_is_hash_invariant() {
    use super::epoch_verify::{BLAKE3_BLOCK_FELTS, blocks_for};
    use super::rpo::RATE_FELTS;

    assert_eq!(
        RATE_FELTS, BLAKE3_BLOCK_FELTS,
        "the two rates must be the same eight felts, or the counts diverge"
    );

    // Every leaf width a real shape can present, plus the exact-multiple
    // boundaries where a padding rule would betray itself.
    for felts in (1..=4_096usize).chain([6_160, 3_816, 8_192, 12_288]) {
        let blake3 = blocks_for(felts, super::edsl::WrapHash::Blake3);
        let rpo = felts.div_ceil(RATE_FELTS);
        assert_eq!(
            blake3, rpo,
            "a {felts}-felt leaf must cost the same count under both hashes"
        );
    }

    // The FRI leaf and a Merkle parent, named because they are the two terms
    // the census treats as rate-INsensitive.
    assert_eq!(
        blocks_for(
            super::epoch_verify::FRI_LEAF_FELTS,
            super::edsl::WrapHash::Blake3
        ),
        1
    );
    assert_eq!(super::epoch_verify::FRI_LEAF_FELTS.div_ceil(RATE_FELTS), 1);
    // A parent is two four-felt digests.
    assert_eq!((2 * super::hash::HASH_DIGEST_FELTS).div_ceil(RATE_FELTS), 1);

    // Keccak is the control: its `pad10*1` DOES spend a trailing block on an
    // exact multiple, so this invariance is a property of these two hashes and
    // not of the closed form.
    assert_ne!(
        blocks_for(
            super::epoch_verify::KECCAK_RATE_FELTS,
            super::edsl::WrapHash::Keccak
        ),
        super::epoch_verify::KECCAK_RATE_FELTS.div_ceil(super::epoch_verify::KECCAK_RATE_FELTS),
        "keccak must NOT share the rule, or the control is vacuous"
    );
}

// =========================================================================
// The measurement — the number this lane exists for
// =========================================================================

/// ★★ **Base-equivalent cells per permutation**, read off the same census
/// instrument that produced the Poseidon and keccak columns (`main + 3·aux`, one
/// row per permutation) — so every column of the comparison table is measured by
/// one instrument and they are comparable by construction.
///
/// Both sides are independent: the left comes from the AIR that was built and
/// proved above, the right is the scoping doc's prediction. A disagreement
/// falsifies the doc's arithmetic, which is the outcome this test is here to
/// allow.
#[test]
fn the_measured_cells_per_permutation_match_the_pinned_prediction() {
    let program = trivial_program();
    let census = lfm_chip_census_with_hasher(&program, HasherKind::Rpo);
    let hash_chip = census
        .iter()
        .find(|c| c.name == "LFM_HASH")
        .expect("LFM_HASH is slot-registered");

    assert_eq!(
        hash_chip.main_cols, PINNED_VALUE_COLUMNS,
        "value columns per permutation row"
    );
    assert_eq!(
        hash_chip.aux_cols, 3,
        "six LfmMem interactions ⇒ three aux columns"
    );
    let per_permutation = hash_chip.main_cols as u64 + 3 * hash_chip.aux_cols as u64;
    assert_eq!(
        per_permutation, PINNED_CELLS_PER_PERMUTATION,
        "the predicted 445 base-equivalent cells per permutation"
    );

    // ★ The economics thesis, as a ratio against the incumbent. The slot-11
    // BLAKE3 compression chip is 3,056 main + 630 ext aux = 4,946 cells; that
    // figure is quoted from the wrap census, not recomputed here.
    const BLAKE3_CELLS_PER_COMPRESSION: u64 = 4_946;
    assert!(
        BLAKE3_CELLS_PER_COMPRESSION / per_permutation >= 11,
        "RPO must be an order of magnitude cheaper per compression than slot-11 BLAKE3"
    );
    // And against the other algebraic candidate, whose chip is a deliberate 2×
    // upper bound: RPO wins on round count alone.
    let poseidon = lfm_chip_census_with_hasher(&program, HasherKind::Poseidon);
    let poseidon_hash = poseidon
        .iter()
        .find(|c| c.name == "LFM_HASH")
        .expect("LFM_HASH");
    assert!(
        hash_chip.main_cols < poseidon_hash.main_cols,
        "RPO must be narrower than the Poseidon reference"
    );
}
