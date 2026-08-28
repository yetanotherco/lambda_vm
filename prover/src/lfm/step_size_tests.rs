//! Assembly ledger entries 8 and 9 — the two OOD-grid blindnesses, witnessed.
//!
//! Both entries are members of the phase's degenerate-parameter family: a defect
//! that no test can see because every proof the phase has shares a parameter
//! value. Their shared cause is `num_eval_points = transition_offsets.len() ·
//! step_size` (`verifier.rs:179`, `prover.rs:1452`) being 2 for every production
//! AIR, which collapses the OOD grid to one row per block.
//!
//! - **Entry 8, the ABSORB ORDER.** Production absorbs both pruned OOD blocks
//!   COLUMN-major (`verifier.rs:1421-1431`). Row-major is indistinguishable while
//!   every block is one row tall. Witnessed here by a THREE-offset AIR: the
//!   next-row block's height is `num_eval_points − step_size`, so three offsets at
//!   `step_size = 1` make it two rows tall.
//! - **Entry 9, the FRAME-STEP VIEW.** `Op::Var{offset}` indexes the constraint
//!   frame's evaluation STEP, and a step is `step_size` grid rows. Witnessed here
//!   by comparing [`super::epoch_verify::frame_step_view`] against production's
//!   own `StarkTableView::into_frame` at `step_size = 2`.
//!
//! ## Two different witnesses, and why one AIR could not carry both
//!
//! The brief asked for ONE synthetic AIR with three transition offsets AND
//! `step_size > 1`, on the grounds that a witness for one entry closes the other
//! only if it exercises both. That is right about the requirement and wrong about
//! the vehicle, for two reasons found by reading and then measured:
//!
//! 1. **Three offsets and `step_size > 1` cannot coexist in a provable AIR.**
//!    `AirWithBuses::new` hardcodes `transition_offsets: vec![0, 1]`
//!    (`lookup.rs:922`), so three offsets means an `AIR` impl, and the only ones
//!    outside `crypto/**`'s example tree are the examples themselves — none of
//!    which has `step_size > 1`. Writing one means adding to `crypto/**`, which is
//!    on the standing always-stop list.
//! 2. **`step_size > 1` is not provable at all** — a framework ceiling, measured
//!    by [`the_prover_cannot_prove_a_step_size_two_air`] rather than argued (in
//!    debug the prover panics on the `RowFrame` shape assert; in release, with
//!    that assert compiled out, it emits a proof production's own verifier
//!    rejects). So entry 9 cannot be closed by a proof of any kind, from any AIR.
//!
//! What closes entry 9 instead is that it does not need a proof. The defect is in
//! how the machine maps a reconstructed grid onto frame steps, and production has
//! its own function for exactly that mapping (`into_frame`), which is a pure
//! function of a grid and a `step_size`. Differentialling against it needs no
//! prover, and it is a stronger oracle than a proof would have been: it is the
//! very code the real verifier runs.
//!
//! So the two entries get two witnesses, each with a production oracle, and
//! neither witness is of the other's defect. What is NOT covered, stated plainly:
//! no test here runs the ASSEMBLED verifier at `step_size > 1`, because nothing
//! can produce such a proof. Entry 9's closure is therefore about the emitter's
//! grid indexing, not about an end-to-end run.

use crypto::fiat_shamir::is_transcript::IsTranscript;
use math::field::traits::IsField;
use stark::config::DefaultStarkTranscript;
use stark::constraints::builder::{ConstraintBuilder, ConstraintSet};
use stark::examples::fibonacci_multi_column::{
    FibonacciMultiColumnAIR, FibonacciMultiColumnPublicInputs, compute_trace,
};
use stark::lookup::{AirWithBuses, AuxiliaryTraceBuildData, NullBoundaryConstraintBuilder};
use stark::proof::options::{GoldilocksCubicProofOptions, ProofOptions};
use stark::proof::view::{MultiProofView, StarkProofView, StarkTableView};
use stark::table::Table;
use stark::trace::TraceTable;
use stark::traits::AIR;
use stark::verifier::{IsStarkVerifier, Verifier};

use crate::tables::types::{FE, FEE, GoldilocksExtension, GoldilocksField};

use super::builder::LfmBuilder;
use super::compiler::compile;
use super::epoch::{
    RootCells, TableAbsorbs, TableChallengeShape, emit_table_challenges, fork_table,
};
use super::executor::execute;
use super::fri::FriShape;
use super::transcript_replay::TranscriptReplay;
use super::validator::validate;
use super::word::{base_word, ext_word, word_as_base, word_as_ext};

type Gl = GoldilocksField;
type Ext3 = GoldilocksExtension;

fn options() -> ProofOptions {
    GoldilocksCubicProofOptions::with_blowup(2).expect("fixture options")
}

// =============================================================================
// Entry 9 — the frame-step view, against production's own frame assembly
// =============================================================================

/// A `num_eval_points × width` grid of distinct extension values, so a
/// mis-indexed read is caught by VALUE and not merely by shape.
fn distinct_grid(rows: usize, width: usize) -> Vec<Vec<FEE>> {
    (0..rows)
        .map(|r| {
            (0..width)
                .map(|c| {
                    FEE::new([
                        FE::from((100 * r + c + 1) as u64),
                        FE::from((7 * r + 2 * c + 3) as u64),
                        FE::from((13 * r + 5 * c + 11) as u64),
                    ])
                })
                .collect()
        })
        .collect()
}

/// What production's constraint interpreter resolves `Op::Var{offset, col}` to,
/// for every offset and column of a grid — via its OWN frame assembly.
///
/// `into_frame` groups the grid into `step_size`-row steps
/// (`proof/view.rs:269-294`) and the interpreter takes row 0 of the step
/// (`constraint_ir/interp.rs:240-242`, which asserts `row == 0`). Nothing here is
/// our arithmetic: the grid goes in, production decides which value each offset
/// sees.
fn production_frame_values(
    grid: &[Vec<FEE>],
    main_width: usize,
    step_size: usize,
) -> Vec<Vec<FEE>> {
    let width = grid[0].len();
    let flat: Vec<FEE> = grid.iter().flat_map(|r| r.iter().cloned()).collect();
    let table = Table::new(flat, width);
    let frame = StarkTableView::Owned(&table).into_frame(main_width, step_size);
    (0..grid.len() / step_size)
        .map(|offset| {
            let step = frame.get_evaluation_step(offset);
            (0..width)
                .map(|col| {
                    if col < main_width {
                        *step.get_main_evaluation_element(0, col)
                    } else {
                        *step.get_aux_evaluation_element(0, col - main_width)
                    }
                })
                .collect()
        })
        .collect()
}

/// ★ ENTRY 9: the machine's frame-step view is production's, at a `step_size`
/// where the two possible answers differ.
///
/// The oracle is `StarkTableView::into_frame` — the function the real verifier
/// calls on the reconstructed grid (`verifier.rs:320-321`) — so this is not a
/// comparison of two of our own passes.
///
/// The `step_size = 1` case is included deliberately and it is the point of the
/// entry: there the strided view and the whole grid are the SAME vector, so the
/// test passes for a correct emitter and for the defective one alike. `step_size =
/// 2` separates them, and the negative half below is what shows it.
#[test]
fn the_frame_step_view_matches_productions_own_frame_assembly() {
    use super::epoch_verify::frame_step_view;

    let main_width = 3usize;
    let width = 4usize; // one aux column, so the aux branch is exercised too
    for (offsets, step_size) in [(2usize, 1usize), (3, 1), (2, 2), (3, 2), (2, 4)] {
        let rows = offsets * step_size;
        let grid = distinct_grid(rows, width);
        let expected = production_frame_values(&grid, main_width, step_size);
        let got = frame_step_view(&grid, step_size);
        assert_eq!(
            got.len(),
            offsets,
            "offsets {offsets}, step_size {step_size}: one view row per frame step"
        );
        assert_eq!(
            got, expected,
            "offsets {offsets}, step_size {step_size}: the machine's frame-step \
             view must be the values production's own frame assembly hands the \
             interpreter"
        );
    }

    // ---- ★ the negative half: the wave-5 defect, and the fact that only
    // step_size > 1 can see it.
    //
    // M2 passed the WHOLE grid to the constraint fold. At step_size 1 that is
    // literally the same vector, so the mutation was invisible to every test in
    // the suite. At step_size 2 it is a different vector, and this is the
    // comparison that says so.
    for step_size in [1usize, 2] {
        let grid = distinct_grid(3 * step_size, width);
        let expected = production_frame_values(&grid, main_width, step_size);
        let whole_grid = grid.clone();
        if step_size == 1 {
            assert_eq!(
                whole_grid, expected,
                "at step_size 1 the whole grid IS the frame view — this is the \
                 blindness the entry records, not a bug"
            );
        } else {
            assert_ne!(
                whole_grid, expected,
                "at step_size {step_size} the whole grid must NOT be the frame \
                 view, or this witness sees nothing"
            );
            // And precisely which rows differ: production sees rows 0, 2, 4.
            assert_eq!(
                expected,
                vec![grid[0].clone(), grid[2].clone(), grid[4].clone()],
                "production's frame reads every step_size-th row"
            );
        }
    }
}

// =============================================================================
// The framework ceiling: step_size > 1 is not provable
// =============================================================================

/// The parameter the ceiling test uses.
const STEP_SIZE: usize = 2;
const STRIDED_COLS: usize = 3;
const STRIDED_ROWS: usize = 64;

/// Reads both transition steps, so the AIR would have a non-empty next-row
/// column set if it could be proved.
struct StridedConstraints;

type StridedAir = AirWithBuses<Gl, Ext3, NullBoundaryConstraintBuilder, (), StridedConstraints>;

impl<F: IsField, E: IsField> ConstraintSet<F, E> for StridedConstraints {
    fn eval<B: ConstraintBuilder<F, E>>(&self, b: &mut B) {
        let here = b.main(0, 0);
        let there = b.main(1, 0);
        b.emit_base(0, there - here);
    }
}

/// The `step_size = 2` fixture, shared by both halves of the ceiling test.
///
/// Column 0 — the only column `StridedConstraints` reads — is CONSTANT, so
/// `main(1, 0) − main(0, 0)` is zero under ANY choice of which rows the two
/// transition offsets resolve to. That is what makes the release half below
/// meaningful: the proof it produces is rejected for a structural
/// prover/verifier disagreement, not because some other frame reading would
/// violate the constraint.
fn strided_fixture() -> (StridedAir, TraceTable<Gl, Ext3>) {
    let air = StridedAir::new(
        STRIDED_COLS,
        AuxiliaryTraceBuildData {
            interactions: vec![],
        },
        &options(),
        STEP_SIZE,
        StridedConstraints,
    )
    .with_name("STRIDED");

    let mut main = Vec::with_capacity(STRIDED_ROWS * STRIDED_COLS);
    for r in 0..STRIDED_ROWS as u64 {
        main.push(FE::from(7u64));
        main.push(FE::from(1_000 + r));
        main.push(FE::from(2_000 + 3 * r));
    }
    let trace = TraceTable::new_main(main, STRIDED_COLS, STEP_SIZE);

    assert_eq!(air.step_size(), STEP_SIZE, "the fixture's step size");
    assert_eq!(
        air.context().transition_offsets.len() * air.step_size(),
        4,
        "num_eval_points is offsets x step_size, so this AIR WOULD have two-row \
         blocks and a stride of two — the shape both entries want"
    );

    (air, trace)
}

/// ★ A FRAMEWORK CEILING, measured rather than asserted from reading, and
/// reported as a finding (standing decisions: report ceilings, do not work around
/// them silently).
///
/// The production prover cannot prove ANY AIR with `step_size > 1`, but it fails
/// in two DIFFERENT ways depending on the build profile, so the ceiling is
/// witnessed twice — once per profile — rather than in a single `should_panic`
/// that only holds in one of them:
///
/// - **Debug (this body).** The CPU transition evaluator borrows one row per
///   transition offset (`RowFrame::from_lde`, `evaluator.rs:72`) and asserts the
///   single-row shape outright: `debug_assert_eq!(lde_trace.lde_step_size,
///   lde_trace.blowup_factor, "RowFrame requires single-row steps (step_size 1)")`
///   — and `lde_step_size = trace_step_size · blowup_factor`, so the equality IS
///   `step_size == 1`. The prover panics before emitting anything.
/// - **Release (the sibling body below, selected by `cfg(not(debug_assertions))`).**
///   `debug_assert` is compiled out, so the prover runs to completion and returns
///   `Ok(proof)` — and production's own verifier REJECTS that proof. Measured, not
///   read: the ceiling is a completeness failure, not a soundness one, and it is
///   NOT the one assert. Relaxing the assert alone would not lift it.
///
/// Nothing production-reachable is affected either way: every AIR in the tree
/// reports `step_size = 1` — the VM tables and the LFM chips pass it through
/// their `build_air` helpers, the continuation AIRs pass it to
/// `AirWithBuses::new` directly, and every example AIR's `step_size` impl returns
/// the literal `1` — so this shape exists only in this fixture.
///
/// What this costs the ledger: entry 9 can have no end-to-end witness, from any
/// AIR, until the ceiling lifts —
/// [`the_frame_step_view_matches_productions_own_frame_assembly`] closes the
/// emitter's half against production's own frame assembly instead.
///
/// Both halves are self-updating: if the ceiling is ever lifted, the debug half
/// stops panicking and the release half starts verifying, and each fails saying
/// entry 9 became closeable end to end.
#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "RowFrame requires single-row steps")]
fn the_prover_cannot_prove_a_step_size_two_air() {
    use crate::test_utils::multi_prove_ram;

    let (air, mut trace) = strided_fixture();

    let pairs: Vec<(
        &dyn AIR<Field = Gl, FieldExtension = Ext3, PublicInputs = ()>,
        _,
        _,
    )> = vec![(&air, &mut trace, &())];
    let _ = multi_prove_ram(pairs, &mut DefaultStarkTranscript::<Ext3>::new(&[]));
}

/// ★ The same ceiling, as release actually reaches it — see the debug body above
/// for the full finding.
///
/// With the `RowFrame` `debug_assert` compiled out the prover does NOT stop: it
/// emits a proof. What still holds is the claim the test's name makes, one level
/// out — that proof does not round-trip, because production's own verifier
/// rejects it. Asserting the rejection (rather than skipping the test in release)
/// is what keeps the required release CI gate covering this path.
#[cfg(not(debug_assertions))]
#[test]
fn the_prover_cannot_prove_a_step_size_two_air() {
    use crate::test_utils::multi_prove_ram;

    let (air, mut trace) = strided_fixture();

    let pairs: Vec<(
        &dyn AIR<Field = Gl, FieldExtension = Ext3, PublicInputs = ()>,
        _,
        _,
    )> = vec![(&air, &mut trace, &())];
    let proof = multi_prove_ram(pairs, &mut DefaultStarkTranscript::<Ext3>::new(&[]))
        .expect("with the debug_assert compiled out the prover runs to completion");

    let refs: Vec<&dyn AIR<Field = Gl, FieldExtension = Ext3, PublicInputs = ()>> = vec![&air];
    assert!(
        !Verifier::multi_verify_views(
            &refs,
            MultiProofView::Owned(&proof),
            &mut DefaultStarkTranscript::<Ext3>::new(&[]),
            &FEE::zero(),
        ),
        "production accepted a step_size = 2 proof — the framework ceiling lifted, \
         so entry 9 is now closeable end to end and this test should be replaced by \
         the end-to-end witness"
    );
}

// =============================================================================
// Entry 8 — the absorb order, on a real proof with a multi-row OOD block
// =============================================================================

type FibAir = FibonacciMultiColumnAIR<Gl, Ext3>;
type FibPi = FibonacciMultiColumnPublicInputs<Gl>;

/// Columns the three-offset fixture carries. More than one, or a column-major and
/// a row-major absorb of the block coincide.
const FIB_COLS: usize = 3;
const FIB_ROWS: usize = 64;

fn fib_initial_values() -> Vec<(FE, FE)> {
    (0..FIB_COLS as u64)
        .map(|c| (FE::from(1 + c), FE::from(3 + 2 * c)))
        .collect()
}

/// A real proof of a real three-offset AIR, produced and accepted by production.
fn fib_proof() -> (
    FibAir,
    FibPi,
    stark::proof::stark::MultiProof<Gl, Ext3, FibPi>,
) {
    use crate::test_utils::multi_prove_ram;

    let opts = options();
    let air = FibAir::with_num_columns(&opts, FIB_COLS);
    let initial_values = fib_initial_values();
    let pi = FibPi {
        initial_values: initial_values.clone(),
    };
    let mut trace = compute_trace::<Gl, Ext3>(&initial_values, FIB_ROWS);

    let pairs: Vec<(
        &dyn AIR<Field = Gl, FieldExtension = Ext3, PublicInputs = FibPi>,
        _,
        _,
    )> = vec![(&air, &mut trace, &pi)];
    let proof = multi_prove_ram(pairs, &mut DefaultStarkTranscript::<Ext3>::new(&[]))
        .expect("the three-offset fixture must prove");
    (air, pi, proof)
}

/// The challenges production derives from this proof, and the shape the machine
/// must replay.
struct FibReplay {
    shape: TableChallengeShape,
    main_root: stark::config::Commitment,
    composition_root: stark::config::Commitment,
    ood_current: Vec<FEE>,
    ood_next: Vec<FEE>,
    parts: Vec<FEE>,
    fri_roots: Vec<stark::config::Commitment>,
    fri_coeffs: Vec<FEE>,
    nonce: Option<u64>,
    beta: FEE,
    z: FEE,
    gamma: FEE,
    zetas: Vec<FEE>,
    iotas: Vec<usize>,
}

fn fib_replay(
    air: &FibAir,
    pi: &FibPi,
    proof: &stark::proof::stark::MultiProof<Gl, Ext3, FibPi>,
) -> FibReplay {
    use stark::domain::new_verifier_domain;

    let view = StarkProofView::Owned(&proof.proofs[0]);
    let opts = air.options();
    let trace_length = view.trace_length();
    let log2_trace_length = trace_length.trailing_zeros();
    let log2_blowup = (opts.blowup_factor as usize).trailing_zeros();

    // Single-table Phase A, transcribed from `multi_verify_views`: this AIR is not
    // preprocessed and has no aux trace, so it is the main root and nothing else.
    assert!(!air.is_preprocessed(), "the fixture is not preprocessed");
    assert!(!air.has_aux_trace(), "the fixture has no aux trace");
    let mut transcript = DefaultStarkTranscript::<Ext3>::new(&[]);
    transcript.append_bytes(view.lde_trace_main_merkle_root());

    let domain = new_verifier_domain(air, trace_length);
    let layout = Verifier::<Gl, Ext3, FibPi>::ood_layout(air);
    let challenges = Verifier::<Gl, Ext3, FibPi>::replay_rounds_after_round_1(
        air,
        view,
        pi,
        &domain,
        &mut transcript,
        Vec::new(),
        &layout,
    );

    let nt = challenges.transition_coeffs.len();
    let beta = if nt > 1 {
        challenges.transition_coeffs[1]
    } else {
        challenges.boundary_coeffs[0]
    };
    let gamma = challenges.trace_term_coeffs[1][0];

    let ood_c = view.trace_ood_evaluations();
    let ood_n = view.trace_ood_next_evaluations();
    let shape = TableChallengeShape {
        index: 0,
        num_tables: 1,
        has_aux_root: view.lde_trace_aux_merkle_root().is_some(),
        has_contribution: view.bus_table_contribution().is_some(),
        log2_trace_length,
        log2_blowup,
        coset_offset: FE::from(opts.coset_offset),
        ood_current_dims: (ood_c.width(), ood_c.height()),
        ood_next_dims: (ood_n.width(), ood_n.height()),
        num_parts: view.composition_poly_parts_ood_evaluation().len(),
        fri: FriShape::from_options(opts, log2_trace_length + log2_blowup),
        grinding_factor: opts.grinding_factor,
        num_queries: opts.fri_number_of_queries,
    };

    FibReplay {
        shape,
        main_root: *view.lde_trace_main_merkle_root(),
        composition_root: *view.composition_poly_root(),
        ood_current: ood_c.row_major_data().to_vec(),
        ood_next: ood_n.row_major_data().to_vec(),
        parts: view.composition_poly_parts_ood_evaluation().to_vec(),
        fri_roots: view.fri_layers_merkle_roots().to_vec(),
        fri_coeffs: view.fri_final_poly_coeffs().to_vec(),
        nonce: view.nonce(),
        beta,
        z: challenges.z,
        gamma,
        zetas: challenges.zetas.clone(),
        iotas: challenges.iotas.clone(),
    }
}

/// The machine's replay of `r`'s rounds, publishing every challenge.
fn fib_challenge_program(
    r: &FibReplay,
) -> (super::compiler::LfmProgram, Vec<Vec<super::word::LfmWord>>) {
    let s = &r.shape;
    let mut b = LfmBuilder::new().with_wrap_hash(super::edsl::WrapHash::production());

    let a_main = b.declare_arena(super::proof_arena::words_per_root() as u32);
    let a_composition = b.declare_arena(super::proof_arena::words_per_root() as u32);
    let a_current = b.declare_arena((s.ood_current_dims.0 * s.ood_current_dims.1) as u32);
    let a_next = b.declare_arena((s.ood_next_dims.0 * s.ood_next_dims.1) as u32);
    let a_parts = b.declare_arena(s.num_parts as u32);
    let a_fri_roots =
        b.declare_arena(super::proof_arena::words_per_root() as u32 * s.fri.num_committed() as u32);
    let a_fri_coeffs = b.declare_arena(s.fri.num_terminal_coeffs() as u32);
    let a_nonce = (s.grinding_factor > 0).then(|| b.declare_arena(1));

    let mut t = TranscriptReplay::new(&[]);
    let main = RootCells::hint(&mut b, a_main, 0);
    main.absorb(&mut b, &mut t);

    let composition = RootCells::hint(&mut b, a_composition, 0);
    let current: Vec<_> = (0..(s.ood_current_dims.0 * s.ood_current_dims.1) as u32)
        .map(|i| b.hint_word(a_current, i).as_ext())
        .collect();
    let next: Vec<_> = (0..(s.ood_next_dims.0 * s.ood_next_dims.1) as u32)
        .map(|i| b.hint_word(a_next, i).as_ext())
        .collect();
    let parts: Vec<_> = (0..s.num_parts as u32)
        .map(|i| b.hint_word(a_parts, i).as_ext())
        .collect();
    let fri_roots: Vec<_> = (0..s.fri.num_committed())
        .map(|i| {
            RootCells::hint(
                &mut b,
                a_fri_roots,
                super::proof_arena::words_per_root() as u32 * i as u32,
            )
        })
        .collect();
    let fri_coeffs: Vec<_> = (0..s.fri.num_terminal_coeffs() as u32)
        .map(|i| b.hint_word(a_fri_coeffs, i).as_ext())
        .collect();
    let nonce = a_nonce.map(|id| b.hint_felt(id, 0));

    let mut fork = fork_table(&t, s.index, s.num_tables);
    let ch = emit_table_challenges(
        &mut b,
        &mut fork,
        s,
        &TableAbsorbs {
            aux_root: None,
            contribution: None,
            composition_root: &composition,
            ood_current: &current,
            ood_next: &next,
            parts: &parts,
            fri_roots: &fri_roots,
            fri_coeffs: &fri_coeffs,
            nonce,
        },
    );
    b.public(ch.beta.as_cell());
    b.public(ch.z.as_cell());
    b.public(ch.gamma.as_cell());
    for zeta in &ch.zetas {
        b.public(zeta.as_cell());
    }
    for bits in &ch.iota_bits {
        let felt = super::edsl::bits_to_felt(&mut b, bits);
        b.public(felt.as_cell());
    }

    let program = compile(b.finish());
    validate(&program).expect("the three-offset replay must be admissible");

    let mut arenas = vec![
        super::proof_arena::commitments_to_arena(&[r.main_root]),
        super::proof_arena::commitments_to_arena(&[r.composition_root]),
        r.ood_current.iter().map(ext_word).collect(),
        r.ood_next.iter().map(ext_word).collect(),
        r.parts.iter().map(ext_word).collect(),
        super::proof_arena::commitments_to_arena(&r.fri_roots),
        r.fri_coeffs.iter().map(ext_word).collect(),
    ];
    if let Some(n) = r.nonce {
        arenas.push(vec![base_word(FE::from(n))]);
    }
    (program, arenas)
}

/// ★ ENTRY 8: the machine absorbs a MULTI-ROW OOD block in production's order.
///
/// The fixture is `FibonacciMultiColumnAIR` with three columns — three transition
/// offsets at `step_size = 1`, so `num_eval_points = 3` and the next-row block is
/// 2 rows × 3 columns. That is the first block in this phase where column-major
/// and row-major absorbs differ.
///
/// The oracle is production's own `replay_rounds_after_round_1` on a proof the
/// production verifier accepts, so this is the same differential the epoch spine
/// runs — on a shape the epoch cannot produce.
///
/// The negative half is not optional: without it a green test here would be
/// consistent with the block still being one row tall. So the same proof's block
/// is absorbed ROW-major through production's own transcript, and the challenge
/// that follows must MOVE.
#[test]
fn the_machine_absorbs_a_multi_row_ood_block_in_productions_order() {
    let (air, pi, proof) = fib_proof();

    // Production must accept it, or the blocks below are not a real proof's.
    let refs: Vec<&dyn AIR<Field = Gl, FieldExtension = Ext3, PublicInputs = FibPi>> = vec![&air];
    assert!(
        Verifier::multi_verify_views(
            &refs,
            MultiProofView::Owned(&proof),
            &mut DefaultStarkTranscript::<Ext3>::new(&[]),
            &FEE::zero(),
        ),
        "production must accept the three-offset fixture"
    );

    let r = fib_replay(&air, &pi, &proof);

    // ---- ★ the blindness this fixture removes, ASSERTED before it is relied on.
    assert_eq!(
        air.context().transition_offsets.len(),
        3,
        "the fixture must have three transition offsets"
    );
    assert_eq!(
        air.context().transition_offsets.len() * air.step_size(),
        3,
        "num_eval_points"
    );
    println!(
        "  three-offset fixture: offsets {:?}, step_size {}, ood_current {:?}, \
         ood_next {:?}, next_row_cols {:?}, parts {}",
        air.context().transition_offsets,
        air.step_size(),
        r.shape.ood_current_dims,
        r.shape.ood_next_dims,
        air.trace_ood_next_row_columns(),
        r.shape.num_parts,
    );
    assert!(
        r.shape.ood_next_dims.1 > 1 && r.shape.ood_next_dims.0 > 1,
        "the next-row OOD block must be taller than one row AND wider than one \
         column, or a row-major absorb is indistinguishable: got {:?}",
        r.shape.ood_next_dims
    );

    // ---- the differential: every challenge, against production's own replay.
    let (program, arenas) = fib_challenge_program(&r);
    let exec = execute(&program, &arenas, &crate::hash_pin::BLOCK_HASHER)
        .expect("the three-offset replay must execute");

    let pub_ext = |i: usize| word_as_ext(&exec.public_words[i].1).expect("an ext challenge");
    assert_eq!(pub_ext(0), r.beta, "beta");
    assert_eq!(pub_ext(1), r.z, "z");
    assert_eq!(
        pub_ext(2),
        r.gamma,
        "gamma — the first challenge AFTER the OOD absorb"
    );
    let mut cursor = 3usize;
    for (k, want) in r.zetas.iter().enumerate() {
        assert_eq!(pub_ext(cursor + k), *want, "zeta {k}");
    }
    cursor += r.zetas.len();
    for q in 0..r.shape.num_queries {
        let got = word_as_base(&exec.public_words[cursor + q].1).expect("an index is a base felt");
        assert_eq!(got, FE::from(r.iotas[q] as u64), "iota {q}");
    }
    cursor += r.shape.num_queries;
    assert_eq!(
        cursor,
        exec.public_words.len(),
        "every published challenge must be checked"
    );

    // ---- ★ the negative half: INJECT the row-major absorb and watch it fail.
    //
    // Not a comparison of two of my own orders — a control program that replays
    // the same rounds with the blocks absorbed row-major, checked against the SAME
    // production challenge the positive half matched. One side is production's.
    //
    // The control stops at `gamma`, the first challenge drawn after the OOD
    // absorb: everything downstream of a wrong `gamma` is wrong for a derived
    // reason, and stopping here says the divergence begins exactly at the absorb.
    let control_gamma = row_major_control_gamma(&r);
    assert_ne!(
        control_gamma, r.gamma,
        "a row-major absorb of this proof's OOD blocks must move the first \
         challenge drawn after them — if it does not, the fixture is as blind as \
         the epoch and entry 8 stays open"
    );
    println!(
        "  row-major control: gamma moves ({} != production's), so the absorb \
         order is load-bearing on this fixture and the differential above covers it",
        control_gamma == r.gamma
    );
}

/// The DENIED absorb order, emitted: Phase A, round 2, `z`, both OOD blocks
/// ROW-major, the parts, then `γ`.
///
/// This is the mutation entry 8 says nothing can catch — deliberately built so it
/// CAN be caught, on a fixture whose blocks are more than one row tall. It stops
/// at `γ` because that is the first value the absorb order can move, so a
/// difference here is attributable to the order and to nothing else.
///
/// It duplicates the round structure of [`super::epoch::emit_table_challenges`]
/// rather than calling it with a flag: a production emitter should not carry a
/// switch for its own denied behaviour, and the duplication is bounded because
/// the control needs nothing past `γ`.
fn row_major_control_gamma(r: &FibReplay) -> FEE {
    let s = &r.shape;
    let mut b = LfmBuilder::new().with_wrap_hash(super::edsl::WrapHash::production());

    let a_main = b.declare_arena(super::proof_arena::words_per_root() as u32);
    let a_composition = b.declare_arena(super::proof_arena::words_per_root() as u32);
    let a_current = b.declare_arena((s.ood_current_dims.0 * s.ood_current_dims.1) as u32);
    let a_next = b.declare_arena((s.ood_next_dims.0 * s.ood_next_dims.1) as u32);
    let a_parts = b.declare_arena(s.num_parts as u32);

    let mut t = TranscriptReplay::new(&[]);
    let main = RootCells::hint(&mut b, a_main, 0);
    main.absorb(&mut b, &mut t);
    let mut fork = fork_table(&t, s.index, s.num_tables);

    let _beta = fork.sample_ext(&mut b);
    let composition = RootCells::hint(&mut b, a_composition, 0);
    composition.absorb(&mut b, &mut fork);

    let _z = super::epoch::emit_z_ood(&mut b, &mut fork, s);
    let current: Vec<_> = (0..(s.ood_current_dims.0 * s.ood_current_dims.1) as u32)
        .map(|i| b.hint_word(a_current, i).as_ext())
        .collect();
    let next: Vec<_> = (0..(s.ood_next_dims.0 * s.ood_next_dims.1) as u32)
        .map(|i| b.hint_word(a_next, i).as_ext())
        .collect();
    let parts: Vec<_> = (0..s.num_parts as u32)
        .map(|i| b.hint_word(a_parts, i).as_ext())
        .collect();

    // ★ THE MUTATION: rows outside, columns inside — production has it the other
    // way round (`verifier.rs:1421-1431`).
    for (dims, block) in [(s.ood_current_dims, &current), (s.ood_next_dims, &next)] {
        let (width, height) = dims;
        for row in 0..height {
            for col in 0..width {
                let coords = b.unpack(block[row * width + col].as_cell());
                fork.append_ext(&mut b, [coords[0], coords[1], coords[2]]);
            }
        }
    }
    for part in &parts {
        let coords = b.unpack(part.as_cell());
        fork.append_ext(&mut b, [coords[0], coords[1], coords[2]]);
    }
    let gamma = fork.sample_ext(&mut b);
    b.public(gamma.as_cell());

    let program = compile(b.finish());
    validate(&program).expect("the control must be admissible");
    let arenas = vec![
        super::proof_arena::commitments_to_arena(&[r.main_root]),
        super::proof_arena::commitments_to_arena(&[r.composition_root]),
        r.ood_current.iter().map(ext_word).collect(),
        r.ood_next.iter().map(ext_word).collect(),
        r.parts.iter().map(ext_word).collect(),
    ];
    let exec = execute(&program, &arenas, &crate::hash_pin::BLOCK_HASHER)
        .expect("the control must execute");
    word_as_ext(&exec.public_words[0].1).expect("gamma is ext")
}
