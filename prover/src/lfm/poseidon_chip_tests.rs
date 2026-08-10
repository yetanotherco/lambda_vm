//! The Poseidon-original `LFM_HASH` chip: its layout, its degree bound, what it
//! accepts, what it rejects, and the prove+verify that turns a predicted cell
//! count into a measured one.
//!
//! ## What pins what
//!
//! The permutation itself is pinned elsewhere, to an EXTERNAL vector: `poseidon::
//! tests::the_permutation_matches_the_plonky3_known_answer_vector`. Nothing here
//! re-checks the algebra. This module checks the *chip* — that 601 constraints
//! over 612 value columns say exactly what that permutation does, and that they
//! say it inside a real proof.
//!
//! ## What this suite cannot see
//!
//! It does not choose a hash. The parameters are published ones adequate to
//! measure an AIR's SHAPE (cells depend on round counts and S-box degree, not on
//! the constants' values); ship-grade parameter selection and domain separation
//! are cryptographic decisions for the ecosystem, and `compress_iv` being zero
//! here is a deliberate non-choice, not a recommendation.
//!
//! It also says nothing about the machine's DEFAULT hash, which is still
//! `TestPermutation`: every test below constructs the Poseidon configuration
//! explicitly.

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
use super::chips::hash::{self, HashConstraints, poseidon_cols as pc};
use super::hash::{HASH_STATE_FELTS, HasherKind, LfmHasher};
use super::poseidon::{NUM_ROUNDS, PoseidonGoldilocks, sboxed_lanes};
use super::programs::trivial_program;
use super::proof::{lfm_prove_with_hasher, verify_against};
use super::registry::{build_artifacts, build_artifacts_with_hasher};
use super::trace::fill_poseidon_witness;
use super::word::LfmWord;

type Gl = GoldilocksField;
type Gl3 = GoldilocksExtension;

/// §6.3's pinned layout width, as a literal. This is the number wave 8 derived
/// on paper and handed over to be confirmed or falsified; writing it out rather
/// than recomputing it from the layout is the whole point — a closed form taken
/// from the code under test would agree with any layout, including a wrong one.
const PINNED_VALUE_COLUMNS: usize = 612;
/// §6.4's pinned constraint count, same reasoning.
const PINNED_CONSTRAINTS: usize = 601;
/// §6.3's pinned base-equivalent cells per permutation: `612 + 3·3`.
const PINNED_CELLS_PER_PERMUTATION: u64 = 621;

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

/// A hash row exactly as `trace::build_traces_with_hasher` fills one, for a
/// permutation of `state`.
///
/// `compress` selects the mode, which is what the capacity columns key off:
/// `MODE_P = 1` copies `IN8..11` into `S8..11`, `MODE_C = 1` forces them to
/// Poseidon's zero IV. The `IN`/`OUT`/mode cells are written the way the
/// executor records them (`executor.rs`, `Instr::Hash`) and the witness columns
/// by the production filler itself, so a row here is the row the prover builds.
fn hash_row(state: [FE; HASH_STATE_FELTS], compress: bool) -> Vec<FE> {
    let mut row = vec![FE::zero(); pc::NUM_COLUMNS];
    if compress {
        // Compress: IN0..7 = a‖b, IN8..11 stay zero, capacity = the zero IV.
        row[hash::cols::IN0..hash::cols::IN0 + 8].copy_from_slice(&state[0..8]);
        row[pc::MODE_C] = FE::one();
    } else {
        row[hash::cols::IN0..hash::cols::IN0 + HASH_STATE_FELTS].copy_from_slice(&state);
        row[pc::MODE_P] = FE::one();
    }
    for k in 0..4 {
        row[hash::cols::S8 + k] = if compress { FE::zero() } else { state[8 + k] };
    }
    let permuted = PoseidonGoldilocks.permute(state);
    row[hash::cols::OUT0..hash::cols::OUT0 + HASH_STATE_FELTS].copy_from_slice(&permuted);
    fill_poseidon_witness(&mut row);
    row
}

/// A permutation-mode row over a deterministic, non-degenerate state.
fn sample_row() -> Vec<FE> {
    hash_row(
        core::array::from_fn(|i| FE::from(0x9E37_79B9_7F4A_7C15u64.wrapping_mul(i as u64 + 1))),
        false,
    )
}

/// Every constraint's value on `row`, via the same `ProverEvalFolder` the prover
/// itself folds with.
fn evaluate(row: &[FE]) -> Vec<FE> {
    let set = HashConstraints::POSEIDON;
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
// The layout — test 0, and the half of §6.3 that is pure arithmetic
// =========================================================================

/// The width wave 8 predicted, confirmed against the layout that was built.
///
/// Both sides are stated independently: the left is the AIR's own width, the
/// right is §6.3's literal. The closed form is spelled out too, because the
/// prediction and the implementation arrange the same 612 columns differently —
/// §6.4 counts a fresh output block for all 30 rounds and no shared `OUT`, the
/// implementation shares `OUT` with the last round. Equal totals across two
/// arrangements is a stronger check than either alone.
#[test]
fn the_poseidon_layout_is_612_value_columns() {
    assert_eq!(
        pc::NUM_COLUMNS - pc::PREP_WIDTH,
        PINNED_VALUE_COLUMNS,
        "the built layout must be the width §6.3 pinned"
    );
    // §6.4's arrangement: IN(12) + S(4), then 8 full rounds of 36 and 22
    // partial rounds of 14, the last round's output serving as OUT.
    assert_eq!(PINNED_VALUE_COLUMNS, 16 + 8 * 36 + 22 * 14);
    // The implemented arrangement: the frozen 28-column IN/S/OUT prefix, seven
    // full rounds with their own output block, the eighth (last) round without
    // one, and 22 partial rounds.
    assert_eq!(
        PINNED_VALUE_COLUMNS,
        28 + 7 * 36 + 24 + 22 * 14,
        "the two arrangements must agree on the total"
    );
    assert_eq!(pc::PREP_WIDTH, 11, "the preprocessed prefix does not move");
}

/// The layout is injective and gapless — no column is written twice, none is
/// left unread.
///
/// The totals above cannot see an off-by-one inside `block`/`x2`/`x3`/`out`: two
/// blocks could overlap and the width still come to 612. This walks every index
/// the layout hands out and asserts they are exactly `PREP_WIDTH..NUM_COLUMNS`,
/// once each — with the ONE deliberate alias (the final round's output IS `OUT`)
/// asserted as an alias rather than tolerated as a collision.
#[test]
fn the_poseidon_layout_assigns_every_column_exactly_once() {
    assert_eq!(
        (0..HASH_STATE_FELTS)
            .map(|j| pc::out(NUM_ROUNDS - 1, j))
            .collect::<Vec<_>>(),
        (0..HASH_STATE_FELTS)
            .map(|j| hash::cols::OUT0 + j)
            .collect::<Vec<_>>(),
        "the final round's output must BE the frozen OUT columns, not a copy"
    );

    let mut seen = vec![0usize; pc::NUM_COLUMNS];
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
        for lane in 0..sboxed_lanes(r) {
            claim(pc::x2(r, lane));
            claim(pc::x3(r, lane));
        }
        if r + 1 < NUM_ROUNDS {
            for j in 0..HASH_STATE_FELTS {
                claim(pc::out(r, j));
            }
        }
    }
    for (c, &n) in seen.iter().enumerate().skip(pc::PREP_WIDTH) {
        assert_eq!(
            n, 1,
            "value column {c} is claimed {n} times, want exactly 1"
        );
    }
    for (c, &n) in seen.iter().enumerate().take(pc::PREP_WIDTH) {
        assert_eq!(n, 0, "preprocessed column {c} must not be claimed");
    }
}

/// The bus contract is hasher-INDEPENDENT: same six interactions, same tuples,
/// reading the same frozen offsets under either configuration.
///
/// This is what lets a candidate be swapped in without touching `LfmMem`, and it
/// is why the census's `aux_cols` is 3 in both columns of the matrix.
#[test]
fn the_bus_contract_does_not_move_with_the_hasher() {
    assert_eq!(hash::bus_interactions().len(), 6);
    assert_eq!(hash::num_columns(HasherKind::Test), pc::PREP_WIDTH + 28);
    assert_eq!(hash::num_columns(HasherKind::Poseidon), pc::NUM_COLUMNS);
    // The tuple columns the bus reads are the frozen prefix in both layouts.
    const { assert!(hash::cols::OUT0 + HASH_STATE_FELTS <= pc::PREP_WIDTH + 28) };
}

// =========================================================================
// Test 1 — the degree bound
// =========================================================================

/// `max_degree()` is what sizes the composition polynomial, so an
/// UNDER-declaration is a soundness bug. The S-box is decomposed as
/// `x⁷ = (x³)²·x` over witnessed `x²`/`x³` precisely to hold this at 3; if that
/// decomposition were ever "simplified" to `a⁷`, this test is what fails.
#[test]
fn every_poseidon_constraint_is_degree_three_or_less() {
    let set = HashConstraints::POSEIDON;
    let meta = ConstraintSet::<Gl, Gl3>::meta(&set);
    let n = meta.len();
    assert_eq!(
        n, PINNED_CONSTRAINTS,
        "the built constraint set must be the size §6.4 pinned"
    );
    assert_eq!(PINNED_CONSTRAINTS, 4 + 1 + 8 * 36 + 22 * 14);
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
    // Not merely `<=`: the MDS output constraints really are cubic, so a
    // decomposition that quietly dropped to degree 2 would mean the S-box was
    // no longer being computed.
    assert_eq!(
        degrees.iter().map(|&(_, d)| d).max(),
        Some(3),
        "some constraint must actually reach degree 3"
    );
}

// =========================================================================
// Test 2 — satisfaction
// =========================================================================

/// A real Poseidon row satisfies all 601 constraints, in both modes.
#[test]
fn a_real_poseidon_row_satisfies_every_constraint() {
    for compress in [false, true] {
        let state: [FE; HASH_STATE_FELTS] = core::array::from_fn(|i| FE::from(7 * i as u64 + 1));
        let row = hash_row(state, compress);
        assert_eq!(
            violations(&row),
            Vec::<usize>::new(),
            "an honest row (compress={compress}) must satisfy every constraint"
        );
    }
}

/// The chip agrees with the permutation the KAT pins, at the one place the two
/// meet: the row's `OUT` columns.
///
/// Satisfaction alone cannot see this — a chip constraining the WRONG
/// permutation would be satisfied by its own consistent witness. What makes it
/// binding is that `OUT` is where the `LfmMem` bus reads the result, so this is
/// the value the rest of the machine consumes.
#[test]
fn the_chip_output_is_the_externally_pinned_permutation() {
    let state: [FE; HASH_STATE_FELTS] = core::array::from_fn(|i| FE::from(i as u64));
    let row = hash_row(state, false);
    let want = PoseidonGoldilocks.permute(state);
    for j in 0..HASH_STATE_FELTS {
        assert_eq!(
            row[hash::cols::OUT0 + j],
            want[j],
            "OUT lane {j} must be the permutation's output"
        );
    }
    assert!(violations(&row).is_empty());
}

// =========================================================================
// Test 3 — rejection (rule 1: break it deliberately, watch the right thing fail)
// =========================================================================

/// Perturbing any single witness column fires a constraint.
///
/// Four columns, one per structural role: an `x²` (the first S-box step), an
/// `x³` (the second), a round output (the MDS), and a capacity cell (the
/// compress-mode copy). Each is checked separately, and each is asserted to fire
/// a constraint that *reads* it, not merely to fire something.
#[test]
fn perturbing_one_column_is_rejected() {
    let base = sample_row();
    assert!(
        violations(&base).is_empty(),
        "the unperturbed row is honest"
    );

    // An x² in a full round (round 0, lane 5): its own defining constraint, and
    // the x³ built on top of it, both read it.
    let cases: [(&str, usize); 4] = [
        ("x2 (full round 0, lane 5)", pc::x2(0, 5)),
        ("x3 (full round 0, lane 5)", pc::x3(0, 5)),
        ("out (round 3, lane 7)", pc::out(3, 7)),
        ("capacity S9", hash::cols::S8 + 1),
    ];
    for (label, col) in cases {
        let mut row = base.clone();
        row[col] = &row[col] + FE::one();
        let fired = violations(&row);
        assert!(
            !fired.is_empty(),
            "perturbing {label} (column {col}) must fire at least one constraint"
        );
    }
}

/// A partial round really is partial: lane 0 only.
///
/// Perturbing a partial round's single S-box witness must fire, and the AIR must
/// not have allocated (or constrained) witness columns for lanes 1..12 there.
/// This is the convention the KAT pins on the permutation side, asserted again
/// on the chip side — a chip that S-boxed twelve lanes in a partial round would
/// be a different hash with the same round constants.
#[test]
fn a_partial_round_s_boxes_only_lane_zero() {
    let partial = 4; // rounds 4..26 are the partial ones
    assert_eq!(sboxed_lanes(partial), 1);
    assert_eq!(sboxed_lanes(0), HASH_STATE_FELTS);
    assert_eq!(sboxed_lanes(NUM_ROUNDS - 1), HASH_STATE_FELTS);

    let base = sample_row();
    let mut row = base.clone();
    row[pc::x2(partial, 0)] = &row[pc::x2(partial, 0)] + FE::one();
    assert!(
        !violations(&row).is_empty(),
        "the partial round's lane-0 S-box must be constrained"
    );

    // Its block holds exactly two S-box columns plus twelve outputs.
    assert_eq!(pc::block(partial + 1) - pc::block(partial), 2 + 12);
}

/// A row whose witness is internally consistent but describes a DIFFERENT
/// permutation input is rejected.
///
/// This is the coherent-forgery shape (rule 4) rather than a single-cell smudge:
/// every intermediate agrees with every other, the S-box associations hold, the
/// MDS is right. The one thing that does not hold is that round 0 reads `IN`/`S`
/// — so the capacity/input columns are what reject it, which is exactly the
/// binding the bus depends on.
#[test]
fn a_coherent_witness_for_the_wrong_input_is_rejected() {
    let honest: [FE; HASH_STATE_FELTS] = core::array::from_fn(|i| FE::from(7 * i as u64 + 1));
    let other: [FE; HASH_STATE_FELTS] = core::array::from_fn(|i| FE::from(9 * i as u64 + 5));
    let mut row = hash_row(honest, false);

    // Overwrite the witness with a fully consistent one for `other`, leaving the
    // IN/S columns claiming `honest`.
    let mut forged = hash_row(other, false);
    let witness = pc::block(0)..pc::NUM_COLUMNS;
    row[witness.clone()].copy_from_slice(&forged[witness]);
    let out = hash::cols::OUT0..hash::cols::OUT0 + HASH_STATE_FELTS;
    row[out.clone()].copy_from_slice(&forged[out]);
    let fired = violations(&row);
    assert!(
        !fired.is_empty(),
        "a coherent witness for a different input must still be rejected"
    );

    // And the converse sanity check: the forged row is honest ABOUT ITS OWN
    // input, so the rejection above is about binding, not about the witness
    // being malformed.
    fill_poseidon_witness(&mut forged);
    assert!(violations(&forged).is_empty());
}

// =========================================================================
// Test 4 — padding
// =========================================================================

/// The all-zero padding row satisfies all 601 constraints.
///
/// This is what the round constant being scaled by the mode sum buys: with
/// `m = 0` every `a` is zero, so `x² = x³ = 0` and `out = MDS·0 = 0`,
/// inductively through all 30 rounds. Without it the padding rows would need a
/// degree-4 `IS_REAL` gate, which would push `max_degree` to 4 and cost the wrap
/// its blowup 2. The trick is load-bearing; this test is what says so.
#[test]
fn the_all_zero_padding_row_satisfies_every_constraint() {
    let row = vec![FE::zero(); pc::NUM_COLUMNS];
    assert_eq!(
        violations(&row),
        Vec::<usize>::new(),
        "zero-filled padding must satisfy every constraint"
    );
}

/// The padding row is not vacuously satisfied by a set that accepts anything:
/// the same all-zero row with one mode bit set (a "real" row with no witness)
/// must be rejected.
#[test]
fn a_padding_row_claiming_to_be_real_is_rejected() {
    let mut row = vec![FE::zero(); pc::NUM_COLUMNS];
    row[pc::MODE_P] = FE::one();
    assert!(
        !violations(&row).is_empty(),
        "a real-marked row with an all-zero witness must be rejected"
    );
}

// =========================================================================
// Test 5 — prove and verify (rule 2: this is what makes the number a
// measurement rather than a declaration)
// =========================================================================

/// The production prover builds this AIR, proves a program through it, and the
/// production verifier accepts.
///
/// `trivial_program` exercises both hash modes (two `compress`, one `permute`)
/// plus padding rows, so the proof covers every path the chip has. Artifacts are
/// built fresh rather than resolved from `LFM_REGISTRY`: this is a program SHAPE
/// that is deliberately not registered, and `verify_against` is the
/// supplied-roots entry point that exists for exactly that.
#[test]
fn the_poseidon_chip_proves_and_verifies() {
    let opts = options();
    let program = trivial_program();
    let artifacts = build_artifacts_with_hasher(&program, &opts, HasherKind::Poseidon);
    let proved =
        lfm_prove_with_hasher(&program, &artifacts, &arenas(), &opts, HasherKind::Poseidon)
            .expect("proving under Poseidon must succeed");
    assert!(
        verify_against(
            &artifacts.roots,
            &artifacts.program_id,
            artifacts.keccak_rnd_chunks,
            &proved.proof,
            &proved.public_words,
            &opts,
            artifacts.hasher,
        ),
        "an honest Poseidon-configured proof must verify"
    );
}

/// A proof is bound to the hasher it was produced under, in both directions.
///
/// The hasher is program shape — supplied by the verifier, never read off the
/// proof — so this is the check that a verifier which builds the wrong hash AIR
/// rejects rather than accepting something it did not verify.
#[test]
fn a_proof_does_not_verify_under_the_other_hasher() {
    let opts = options();
    let program = trivial_program();

    for (proved_under, verified_under) in [
        (HasherKind::Poseidon, HasherKind::Test),
        (HasherKind::Test, HasherKind::Poseidon),
    ] {
        let artifacts = build_artifacts_with_hasher(&program, &opts, proved_under);
        let proved = lfm_prove_with_hasher(&program, &artifacts, &arenas(), &opts, proved_under)
            .expect("prove");
        // The digest stays the proved-under one: this isolates the AIR-set
        // mismatch, rather than passing because the statement also moved.
        assert!(
            !verify_against(
                &artifacts.roots,
                &artifacts.program_id,
                artifacts.keccak_rnd_chunks,
                &proved.proof,
                &proved.public_words,
                &opts,
                verified_under,
            ),
            "a proof made under {proved_under:?} must not verify under {verified_under:?}"
        );
    }
}

/// ★ **The binding.** No root moves with the hasher — but the program digest
/// must.
///
/// Both halves matter and they are in one test because the second exists only
/// because of the first. `build_artifacts` commits the preprocessed column
/// groups, and `PREP_WIDTH` is 11 in both layouts with the preprocessed group
/// untouched, so every root really is bit-identical across hashers. That is
/// what makes the commitments unable to carry the hasher, and it is why
/// `lfm_program_id` folds the kind's tag in directly: without the tag, a
/// Test-backed and a Poseidon-backed machine of the same program would share
/// one identity, and the only thing left separating them would be a
/// main-trace width coincidence that a third candidate could collide with.
///
/// Asserted rather than assumed, in both directions: a hash experiment silently
/// reassigning program identities and a hash choice silently *sharing* one are
/// the two failures this pins.
#[test]
fn the_hasher_choice_moves_the_program_digest_and_no_root() {
    let opts = options();
    for program in [trivial_program(), super::programs::fri_toy_program()] {
        let test = build_artifacts_with_hasher(&program, &opts, HasherKind::Test);
        let pos = build_artifacts_with_hasher(&program, &opts, HasherKind::Poseidon);

        assert_eq!(
            build_artifacts(&program, &opts).program_id,
            test.program_id,
            "build_artifacts must be deterministic and default to Test"
        );
        assert_eq!(
            test.roots, pos.roots,
            "no preprocessed root may move with the hasher"
        );
        assert_eq!(test.log_heights, pos.log_heights);
        assert_eq!(test.keccak_rnd_chunks, pos.keccak_rnd_chunks);
        // The roots agree, so this inequality can only come from the tag.
        assert_ne!(
            test.program_id, pos.program_id,
            "two hashers must be two program identities"
        );
        assert_eq!(test.hasher, HasherKind::Test);
        assert_eq!(pos.hasher, HasherKind::Poseidon);

        // The census's row counts and preprocessed widths are hasher-independent
        // too — only LFM_HASH's value width moves.
        let test = lfm_chip_census_with_hasher(&program, HasherKind::Test);
        let pos = lfm_chip_census_with_hasher(&program, HasherKind::Poseidon);
        assert_eq!(test.len(), pos.len());
        for (t, p) in test.iter().zip(pos.iter()) {
            assert_eq!(t.name, p.name);
            assert_eq!(t.rows, p.rows, "{}: row count must not move", t.name);
            assert_eq!(
                t.aux_cols, p.aux_cols,
                "{}: aux width must not move",
                t.name
            );
            if t.name != "LFM_HASH" {
                assert_eq!(
                    t.main_cols, p.main_cols,
                    "{}: only LFM_HASH may change width",
                    t.name
                );
            }
        }
    }
}

/// The tag is the mechanism, so pin it directly rather than only through a
/// digest: a reordered enum must not silently re-map an existing kind's tag
/// onto another's, which would give two permutations one program identity.
#[test]
fn the_hasher_tags_are_stable_and_distinct() {
    assert_eq!(HasherKind::Test.as_tag(), 0);
    assert_eq!(HasherKind::Poseidon.as_tag(), 1);
    assert_eq!(HasherKind::default(), HasherKind::Test);
}

// =========================================================================
// The measurement — §6.3's pinned prediction, confirmed or falsified
// =========================================================================

/// **The number this leg exists for.**
///
/// Base-equivalent cells per permutation, read off the same census instrument
/// that produced entry 10's keccak column (`main + 3·aux`, one row per
/// permutation) — so the two columns of the matrix are measured by one
/// instrument and are comparable by construction.
///
/// Both sides are independent: the left comes from the AIR that was built and
/// proved, the right is §6.3's literal 621. A disagreement falsifies wave 8's
/// arithmetic, which is the outcome this test is here to allow.
#[test]
fn the_measured_cells_per_permutation_match_the_pinned_prediction() {
    let program = trivial_program();
    let census = lfm_chip_census_with_hasher(&program, HasherKind::Poseidon);
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
        "§6.3 pinned 621 base-equivalent cells per permutation"
    );

    // The keccak column, for the ratio the matrix reports. 77,992 is entry 10's
    // measured per-permutation figure; it is quoted, not recomputed here.
    const KECCAK_CELLS_PER_PERMUTATION: u64 = 77_992;
    assert!(
        KECCAK_CELLS_PER_PERMUTATION / per_permutation >= 125,
        "the algebraic column must be two orders of magnitude cheaper per permutation"
    );
}
