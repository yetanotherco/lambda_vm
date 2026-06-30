//! Differential tests for the explicit-builder constraint capture spike (Plan B).
//!
//! For each algebraic transition constraint, capture it into a flat IR via its
//! `Capture::capture` method (an explicit `IrBuilder`), then assert that
//! interpreting the IR reproduces the constraint's real
//! `evaluate::<GoldilocksField, GoldilocksExtension>` bit-for-bit over many
//! random main rows.

use crate::constraints::cpu::ProductZeroConstraint;
use crate::constraints::templates::{AddConstraint, AddOperand, IsBitConstraint};
use crate::tables::types::{FE, GoldilocksExtension, GoldilocksField};

use math::field::element::FieldElement;
use stark::constraint_ir::{Capture, IrBuilder, eval_program_base};
use stark::constraints::transition::TransitionConstraint;
use stark::table::TableView;

/// Number of random trials per constraint.
const TRIALS: usize = 1000;

/// Column count for the random frame; larger than any column index read by the
/// constraints under test (CPU columns go up to 37).
const NUM_COLS: usize = 64;

/// A tiny deterministic SplitMix64 PRNG so the test needs no `rand` dependency
/// and is fully reproducible.
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

/// Run the differential check: capture `c` via the builder, then for `TRIALS`
/// random rows compare the real `evaluate` against the IR interpreter,
/// bit-for-bit.
fn assert_ir_matches_evaluate<T>(c: &T, label: &str)
where
    T: TransitionConstraint<GoldilocksField, GoldilocksExtension> + Capture,
{
    let mut b = IrBuilder::new();
    c.capture(&mut b);
    let prog = b.finish(0);
    eprintln!("[{label}] captured {} IR nodes", prog.len());

    let mut rng = SplitMix64::new(0xDEAD_BEEF_CAFE_F00D ^ (label.len() as u64));

    for trial in 0..TRIALS {
        // Build a random main row.
        let row: Vec<FE> = (0..NUM_COLS).map(|_| FE::from(rng.next_u64())).collect();

        // Real evaluate: wrap the row in a base/ext TableView (1 row, no aux).
        let real_step: TableView<GoldilocksField, GoldilocksExtension> =
            TableView::new(vec![row.clone()], vec![Vec::new()]);
        let real: FieldElement<GoldilocksField> =
            c.evaluate::<GoldilocksField, GoldilocksExtension>(&real_step);

        // IR interpreter over the same row.
        let got = eval_program_base(&prog, c.constraint_idx(), &row);

        assert_eq!(
            real, got,
            "[{label}] mismatch at trial {trial}: real={real:?} got={got:?}"
        );
    }
}

#[test]
fn test_ir_matches_is_bit_unconditional() {
    // X * (1 - X), X at column 7.
    let c = IsBitConstraint::unconditional(7, 0);
    assert_ir_matches_evaluate(&c, "is_bit_unconditional");
}

#[test]
fn test_ir_matches_is_bit_conditional() {
    // cond * X * (1 - X), cond at column 3, X at column 5.
    let c = IsBitConstraint::new(3, 5, 0);
    assert_ir_matches_evaluate(&c, "is_bit_conditional");
}

#[test]
fn test_ir_matches_add_constraint_carries() {
    // 64-bit ADD with embedded carries, DWordWL operands.
    // cond at col 0; lhs=[1,2], rhs=[3,4], sum=[5,6].
    let (carry0, carry1) = AddConstraint::new_pair(
        vec![0],
        AddOperand::dword(1),
        AddOperand::dword(3),
        AddOperand::dword(5),
        0,
    );
    assert_ir_matches_evaluate(&carry0, "add_carry_0");
    assert_ir_matches_evaluate(&carry1, "add_carry_1");
}

#[test]
fn test_ir_matches_product_zero() {
    // col_a * col_b, columns 12 and 17.
    let c = ProductZeroConstraint::new(12, 17, 0);
    assert_ir_matches_evaluate(&c, "product_zero");
}

// =============================================================================
// Phase 1 GATE: full-table, full-program differential test (CPU, LogUp-heavy).
//
// `create_cpu_air` assembles every algebraic CPU constraint AND, via
// `AirWithBuses::new`, the 2 LogUp constraints for its bus interactions
// (DECODE/ALU/MEMORY/CPU32/MEMW/BRANCH/ECALL). Capturing its full program and
// interpreting it over a real LDE must reproduce `air.compute_transition_prover`
// (prover) and `air.compute_transition` (verifier, at the OOD point) bit-for-bit.
// =============================================================================
mod full_table_gate {
    use crate::tables::cpu::{CpuOperation, generate_cpu_trace};
    use crate::tables::eq::{EqOperation, generate_eq_trace};
    use crate::tables::types::DecodeEntry;
    use crate::test_utils::{VmAir, create_cpu_air, create_eq_air};

    use crypto::fiat_shamir::default_transcript::DefaultTranscript;
    use executor::vm::instruction::decoding::{ArithOp, Instruction};
    use executor::vm::logs::Log;
    use math::field::element::FieldElement;
    use stark::constraint_ir::{eval_program, eval_program_verifier};
    use stark::frame::Frame;
    use stark::proof::options::ProofOptions;
    use stark::table::TableView;
    use stark::trace::{LDETraceTable, TraceTable};
    use stark::traits::{AIR, TransitionEvaluationContext};

    use super::{GoldilocksExtension, GoldilocksField};

    const PC: u64 = 0x1000;

    /// Build a `CpuOperation` from an instruction + register values (mirrors
    /// `prover/src/tests/cpu_tests.rs::op_of`, duplicated here to keep this
    /// gate test self-contained).
    fn op_of(instr: Instruction, src1: u64, src2: u64, dst: u64, next_pc: u64) -> CpuOperation {
        let decode = DecodeEntry::from_instruction(PC, instr, 4);
        let log = Log {
            current_pc: PC,
            next_pc,
            src1_val: src1,
            src2_val: src2,
            dst_val: dst,
        };
        CpuOperation::from_log(&log, 4, decode)
    }

    /// A handful of real CPU operations exercising different bus interactions
    /// (ALU add, ALU sub) so the captured LogUp program sees non-trivial
    /// fingerprints/multiplicities on every row, not just padding zeros.
    fn sample_operations() -> Vec<CpuOperation> {
        vec![
            op_of(
                Instruction::Arith {
                    dst: 3,
                    src1: 1,
                    src2: 2,
                    op: ArithOp::Add,
                },
                10,
                20,
                30,
                PC + 4,
            ),
            op_of(
                Instruction::Arith {
                    dst: 3,
                    src1: 1,
                    src2: 2,
                    op: ArithOp::Sub,
                },
                50,
                20,
                30,
                PC + 4,
            ),
        ]
    }

    /// A handful of real EQ operations (BEQ-style and BNE-style) so the
    /// captured LogUp program sees non-trivial fingerprints on every row.
    ///
    /// Both this table and CPU (below) have an even `bus_interactions().len()`
    /// (EQ: 6, CPU: 20), so `split_interactions` gives both an `absorbed_count`
    /// of **2**, not 1 — there is in fact no in-repo production table whose
    /// real interaction count is odd-and->1, so nothing here exercises
    /// `LookupAccumulatedConstraint`'s 1-absorbed branch end-to-end. Both the
    /// 1- and 2-absorbed branches (plus `LookupBatchedTermConstraint`, and all
    /// 10 `Packing` variants) are covered by targeted, self-certifying
    /// differential tests in `crypto/stark/src/lookup.rs`'s
    /// `logup_capture_tests` module instead (each asserts `absorbed.len()`/
    /// `degree()` up front so the test can't silently degrade to the wrong
    /// branch).
    fn sample_eq_operations() -> Vec<EqOperation> {
        vec![
            EqOperation::new(42, 42, false),
            EqOperation::new(7, 9, true),
        ]
    }

    #[test]
    fn test_cpu_table_full_program_matches_boxed_path_prover_and_verifier() {
        let air = create_cpu_air(&ProofOptions::default_test_options());
        let trace = generate_cpu_trace(&sample_operations());
        assert_full_table_ir_matches_boxed_path(&air, trace, "cpu_full_table");
    }

    #[test]
    fn test_eq_table_full_program_matches_boxed_path_prover_and_verifier() {
        let air = create_eq_air(&ProofOptions::default_test_options());
        let trace = generate_eq_trace(&sample_eq_operations());
        assert_full_table_ir_matches_boxed_path(&air, trace, "eq_full_table");
    }

    /// Capture `air`'s full program, then for every prover-side LDE row and one
    /// verifier-side OOD point, assert the IR interpreter reproduces the boxed
    /// `compute_transition_prover`/`compute_transition` path bit-for-bit.
    fn assert_full_table_ir_matches_boxed_path(
        air: &VmAir,
        mut trace: TraceTable<GoldilocksField, GoldilocksExtension>,
        label: &str,
    ) {
        // Build the aux (LogUp) trace + rap challenges, exactly as the prover
        // pipeline would (minus the surrounding LDE/FRI machinery, which the
        // constraint evaluator doesn't touch).
        let mut transcript = DefaultTranscript::<GoldilocksExtension>::new(&[]);
        let rap_challenges = air.build_rap_challenges(&mut transcript);
        air.build_auxiliary_trace(&mut trace, &rap_challenges);

        let num_rows = trace.num_rows();
        assert!(num_rows >= 2, "need >=2 rows for the LogUp next-row read");

        let main_columns: Vec<Vec<FieldElement<GoldilocksField>>> = (0..trace.num_main_columns)
            .map(|col| {
                (0..num_rows)
                    .map(|row| *trace.main_table.get(row, col))
                    .collect()
            })
            .collect();
        let aux_columns: Vec<Vec<FieldElement<GoldilocksExtension>>> = (0..trace.num_aux_columns)
            .map(|col| {
                (0..num_rows)
                    .map(|row| *trace.aux_table.get(row, col))
                    .collect()
            })
            .collect();
        let lde_trace =
            LDETraceTable::from_columns(main_columns.clone(), aux_columns.clone(), 1, 1);

        let prog = air.constraint_program();
        eprintln!(
            "[{label}] captured {} IR nodes, {} constraints (num_base={})",
            prog.len(),
            prog.roots.len(),
            prog.num_base
        );
        assert_eq!(
            prog.roots.len(),
            air.num_transition_constraints(),
            "every constraint_idx must have been emitted"
        );
        assert_eq!(prog.num_base, air.num_base_transition_constraints());
        assert!(
            prog.complete,
            "[{label}] every production constraint must have a real Capture impl \
             (a constraint fell back to the default IrBuilder::mark_unsupported, \
             which would make ConstraintEvaluator skip the IR path entirely for this AIR)"
        );

        let num_base = air.num_base_transition_constraints();
        let num_transition = air.num_transition_constraints();
        let no_periodic: Vec<FieldElement<GoldilocksField>> = Vec::new();
        let logup_alpha_powers = {
            // Mirrors `ConstraintEvaluator::evaluate_transitions`'s alpha-power
            // precompute (`compute_alpha_powers`, crate-private in `stark`):
            // [1, alpha, alpha^2, ...], rap_challenges[1] is alpha.
            use stark::lookup::LOGUP_CHALLENGE_ALPHA;
            if rap_challenges.len() > LOGUP_CHALLENGE_ALPHA {
                let alpha = &rap_challenges[LOGUP_CHALLENGE_ALPHA];
                let count = air.max_bus_elements();
                let mut powers = Vec::with_capacity(count);
                let mut cur = FieldElement::<GoldilocksExtension>::one();
                for _ in 0..count {
                    powers.push(cur);
                    cur *= alpha;
                }
                powers
            } else {
                Vec::new()
            }
        };
        let logup_table_offset = FieldElement::<GoldilocksExtension>::zero();
        let packing_shifts = stark::lookup::PackingShifts::<GoldilocksField>::new();

        // --- Prover-side: every row, boxed path vs IR interpreter ---
        let offsets = &air.context().transition_offsets;
        for step in 0..lde_trace.num_steps() {
            let frame: Frame<GoldilocksField, GoldilocksExtension> =
                Frame::read_step_from_lde(&lde_trace, step, offsets);
            let ctx = TransitionEvaluationContext::new_prover(
                &frame,
                &no_periodic,
                &rap_challenges,
                &logup_alpha_powers,
                &logup_table_offset,
                &packing_shifts,
            );

            let mut boxed_base = vec![FieldElement::<GoldilocksField>::zero(); num_base];
            let mut boxed_ext = vec![FieldElement::<GoldilocksExtension>::zero(); num_transition];
            air.compute_transition_prover(&ctx, &mut boxed_base, &mut boxed_ext);

            let mut ir_base = vec![FieldElement::<GoldilocksField>::zero(); num_base];
            let mut ir_ext = vec![FieldElement::<GoldilocksExtension>::zero(); num_transition];
            eval_program(&prog, &ctx, &mut ir_base, &mut ir_ext);

            assert_eq!(boxed_base, ir_base, "base evals mismatch at step {step}");
            assert_eq!(
                boxed_ext[num_base..],
                ir_ext[num_base..],
                "ext (LogUp) evals mismatch at step {step}"
            );
        }

        // --- Verifier-side: at one "OOD" point, boxed path vs IR interpreter ---
        // The verifier frame holds only extension-field elements; embed the
        // same real row data (rows 0 and 1, matching transition_offsets=[0,1])
        // into GoldilocksExtension to build it, exactly as `evaluate_zerofier`'s
        // sibling machinery would after a real FRI opening.
        let embed_row = |row: usize| -> (
            Vec<FieldElement<GoldilocksExtension>>,
            Vec<FieldElement<GoldilocksExtension>>,
        ) {
            let main: Vec<_> = main_columns
                .iter()
                .map(|col| col[row].to_extension())
                .collect();
            let aux: Vec<_> = aux_columns.iter().map(|col| col[row]).collect();
            (main, aux)
        };
        let (main0, aux0) = embed_row(0);
        let (main1, aux1) = embed_row(1);
        let verifier_frame: Frame<GoldilocksExtension, GoldilocksExtension> = Frame::new(vec![
            TableView::new(vec![main0], vec![aux0]),
            TableView::new(vec![main1], vec![aux1]),
        ]);
        let no_periodic_ext: Vec<FieldElement<GoldilocksExtension>> = no_periodic
            .iter()
            .map(|x: &FieldElement<GoldilocksField>| (*x).to_extension())
            .collect();
        let verifier_packing_shifts = stark::lookup::PackingShifts::<GoldilocksExtension>::new();
        let verifier_ctx = TransitionEvaluationContext::new_verifier(
            &verifier_frame,
            &no_periodic_ext,
            &rap_challenges,
            &logup_alpha_powers,
            &logup_table_offset,
            &verifier_packing_shifts,
        );

        let boxed_verifier = air.compute_transition(&verifier_ctx);
        let mut ir_verifier = vec![FieldElement::<GoldilocksExtension>::zero(); num_transition];
        eval_program_verifier(&prog, &verifier_ctx, &mut ir_verifier);

        assert_eq!(
            boxed_verifier, ir_verifier,
            "verifier evals mismatch at OOD point"
        );
    }
}
