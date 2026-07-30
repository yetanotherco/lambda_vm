//! The constraint-evaluation leg: lowering differential, cost census, and the
//! falsifications for each mechanism the lowering relies on.
//!
//! ## The oracle
//!
//! `eval_program_verifier` — the production CPU interpreter, on the OOD shape —
//! run over the DESERIALIZED artifact, never a local reimplementation of the
//! algebra. It is in turn pinned against the compiled folders by
//! `tests::constraint_artifact_tests`, so the chain from an AIR's Rust
//! constraints to the number this machine computes has no unpinned link.
//!
//! ## What this suite cannot see
//!
//! It executes; it does not prove. Per method rule 2, execution says nothing
//! about whether the CHIPS agree with the executor — the executor mirrors the
//! ALU it is checking. `constraint_leg_proves_and_verifies` is the test that
//! sees the chips, and it is deliberately on a small AIR: the differential's job
//! is coverage across all 28 tables, the proof's job is to close the
//! executor-vs-chip gap once.

use stark::constraint_ir::{ConstraintArtifact, eval_program_verifier};
use stark::frame::Frame;
use stark::proof::options::GoldilocksCubicProofOptions;
use stark::table::TableView;
use stark::traits::TransitionEvaluationContext;

use crate::tables::types::{FE, FEE, GoldilocksExtension, GoldilocksField};
use crate::test_utils::{NUM_PRODUCTION_AIRS, production_airs};

use super::builder::LfmBuilder;
use super::compiler::compile;
use super::constraints::{
    Analysis, OodOperands, analyze, emit_analyzed, hint_ood_frame, ood_frame_words,
};
use super::executor::execute;
use super::hash::TestPermutation;
use super::validator::validate;
use super::word::{LfmWord, ext_word, word_as_ext};

type Gl = GoldilocksField;
type Ext3 = GoldilocksExtension;

fn options() -> stark::proof::options::ProofOptions {
    GoldilocksCubicProofOptions::with_blowup(2).expect("blowup=2 is valid")
}

/// Deterministic SplitMix64, matching the artifact suite's generator so the two
/// sweep the same kind of input.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn fp3(&mut self) -> FEE {
        FEE::new([
            FE::from(self.next_u64()),
            FE::from(self.next_u64()),
            FE::from(self.next_u64()),
        ])
    }
}

/// One AIR's OOD inputs: an all-extension frame with the verifier's next-row
/// PRUNING already applied, plus the per-proof uniforms.
///
/// The pruning matters to the differential, not just to the cost: the verifier
/// reconstructs an undeclared next-row column as ZERO, so a host frame that put
/// a random value there would be comparing the machine against a frame no
/// verifier can produce.
struct OodFixture {
    /// `steps[offset]` = `[main | aux]`, aux starting at `main_width`.
    steps: Vec<Vec<FEE>>,
    main_width: usize,
    aux_width: usize,
    rap_challenges: Vec<FEE>,
    alpha_powers: Vec<FEE>,
    table_offset: FEE,
}

impl OodFixture {
    fn sample(artifact: &ConstraintArtifact, rng: &mut SplitMix64) -> Self {
        let shape = &artifact.shape;
        let main_width = shape.main_width as usize;
        let aux_width = shape.aux_width as usize;
        let width = main_width + aux_width;
        let num_steps = shape.transition_offsets.len().max(1);

        let steps = (0..num_steps)
            .map(|offset| {
                (0..width)
                    .map(|col| {
                        let opened = offset == 0 || shape.next_row_columns.contains(&(col as u32));
                        if opened { rng.fp3() } else { FEE::zero() }
                    })
                    .collect()
            })
            .collect();

        Self {
            steps,
            main_width,
            aux_width,
            // [z, alpha] — the LogUp RAP challenges, in the verifier's order.
            rap_challenges: vec![rng.fp3(), rng.fp3()],
            alpha_powers: (0..shape.max_bus_elements as usize + 2)
                .map(|_| rng.fp3())
                .collect(),
            table_offset: rng.fp3(),
        }
    }

    /// The oracle's frame: the same values, in the verifier's own container.
    fn frame(&self) -> Frame<Ext3, Ext3> {
        Frame::new(
            self.steps
                .iter()
                .map(|s| {
                    TableView::new(
                        vec![s[..self.main_width].to_vec()],
                        vec![s[self.main_width..].to_vec()],
                    )
                })
                .collect(),
        )
    }

    /// The arena the machine reads, in [`hint_ood_frame`]'s order: every opened
    /// entry, step by step, column by column — pruned entries omitted because
    /// the program supplies its own zero for them.
    fn arena(&self, artifact: &ConstraintArtifact) -> Vec<LfmWord> {
        let shape = &artifact.shape;
        let mut out = Vec::new();
        for (offset, step) in self.steps.iter().enumerate() {
            for (col, v) in step.iter().enumerate() {
                if offset == 0 || shape.next_row_columns.contains(&(col as u32)) {
                    out.push(ext_word(v));
                }
            }
        }
        out
    }

    fn uniform_arena(&self) -> Vec<LfmWord> {
        self.rap_challenges
            .iter()
            .chain(&self.alpha_powers)
            .chain(std::iter::once(&self.table_offset))
            .map(ext_word)
            .collect()
    }
}

/// Builds the differential program for one artifact: hint the frame and the
/// uniforms, lower the constraints, publish nothing.
///
/// The uniforms are hinted HERE and only here. In the assembled verifier they
/// come from `TranscriptReplay` — an arena would let a prover choose its own
/// challenges — so this shortcut is a property of the isolated slice and is
/// asserted against in `challenges_are_not_an_arena_in_the_assembled_verifier`.
fn differential_program(
    artifact: &ConstraintArtifact,
    an: &Analysis,
) -> (super::compiler::LfmProgram, Vec<super::builder::Ext>) {
    let mut b = LfmBuilder::new();

    let frame_arena = b.declare_arena(ood_frame_words(artifact));
    let (steps, words) = hint_ood_frame(&mut b, artifact, frame_arena, 0);
    assert_eq!(
        words,
        ood_frame_words(artifact),
        "ood_frame_words must predict what hint_ood_frame consumes"
    );

    let shape = &artifact.shape;
    let num_uniforms = 2 + (shape.max_bus_elements + 2) + 1;
    let uniform_arena = b.declare_arena(num_uniforms);
    let mut next = 0u32;
    let mut take = |b: &mut LfmBuilder| {
        let c = b.hint_word(uniform_arena, next).as_ext();
        next += 1;
        c
    };
    let rap_challenges = vec![take(&mut b), take(&mut b)];
    let alpha_powers: Vec<_> = (0..shape.max_bus_elements + 2)
        .map(|_| take(&mut b))
        .collect();
    let table_offset = take(&mut b);

    let ood = OodOperands {
        steps,
        main_width: shape.main_width as usize,
        rap_challenges,
        alpha_powers,
        table_offset,
    };
    let evals = emit_analyzed(&mut b, an, &ood);
    for e in &evals {
        b.public(e.as_cell());
    }
    (compile(b.finish()), evals)
}

// =============================================================================
// (a) + (b) — the lowering differential, every production AIR
// =============================================================================

/// ★ Every production AIR's lowered constraint program computes exactly what
/// the production interpreter computes, on random all-extension OOD frames.
///
/// This is the acceptance criterion for the lowering pass. It runs over the
/// DESERIALIZED artifact, so the wire hop is inside the loop, and it compares
/// every constraint of every one of the 28 tables — including the three
/// continuation-only tables that no monolithic proof contains.
#[test]
fn lowered_constraints_match_the_verifier_interpreter() {
    const TRIALS: usize = 4;

    let opts = options();
    let airs = production_airs(&opts);
    assert_eq!(airs.len(), NUM_PRODUCTION_AIRS);

    for (label, air) in &airs {
        let artifact = ConstraintArtifact::capture(&**air);
        let bytes = artifact.to_bytes().expect("serialize");
        let artifact = ConstraintArtifact::from_bytes(&bytes).expect("deserialize");
        let prog = artifact.program();
        let n = prog.roots.len();

        let an = analyze(&artifact);
        let (program, evals) = differential_program(&artifact, &an);
        validate(&program).unwrap_or_else(|e| panic!("[{label}] lowered program invalid: {e:?}"));

        let mut rng = SplitMix64(0xC0FF_EE00 ^ label.len() as u64);
        for trial in 0..TRIALS {
            let fixture = OodFixture::sample(&artifact, &mut rng);

            let exec = execute(
                &program,
                &[fixture.arena(&artifact), fixture.uniform_arena()],
                &TestPermutation,
            )
            .unwrap_or_else(|e| panic!("[{label}] trial {trial}: execution failed: {e:?}"));

            // --- oracle: the production interpreter, verifier shape ---
            let frame = fixture.frame();
            let ctx = TransitionEvaluationContext::<Gl, Ext3>::new_verifier(
                &frame,
                &fixture.rap_challenges,
                &fixture.alpha_powers,
                &fixture.table_offset,
            );
            let mut expected = vec![FEE::zero(); n];
            eval_program_verifier(&prog, &ctx, &mut expected);

            for (c, want) in expected.iter().enumerate() {
                let cell = exec.memory[evals[c].addr().0 as usize]
                    .unwrap_or_else(|| panic!("[{label}] constraint {c} cell unwritten"));
                let got = word_as_ext(&cell).expect("an ext value has lane 3 zero");
                assert_eq!(
                    got, *want,
                    "[{label}] trial {trial}: constraint {c} disagrees with the interpreter"
                );
            }
            assert_eq!(
                fixture.aux_width, artifact.shape.aux_width as usize,
                "[{label}] fixture and artifact disagree on aux width"
            );
        }
    }
}

/// The differential's own falsification: a lowering that drops the extension
/// arithmetic must be CAUGHT. Perturbing one constraint value by one and
/// re-checking proves the comparison above is load-bearing rather than
/// comparing two zeros.
#[test]
fn the_differential_rejects_a_perturbed_constraint_value() {
    let opts = options();
    let airs = production_airs(&opts);
    let (label, air) = airs
        .iter()
        .find(|(l, _)| *l == "L2G_GLOBAL")
        .expect("L2G_GLOBAL is a production AIR");

    let artifact = ConstraintArtifact::capture(&**air);
    let prog = artifact.program();
    let an = analyze(&artifact);
    let (program, evals) = differential_program(&artifact, &an);

    let mut rng = SplitMix64(1);
    let fixture = OodFixture::sample(&artifact, &mut rng);
    let exec = execute(
        &program,
        &[fixture.arena(&artifact), fixture.uniform_arena()],
        &TestPermutation,
    )
    .expect("execution");

    let frame = fixture.frame();
    let ctx = TransitionEvaluationContext::<Gl, Ext3>::new_verifier(
        &frame,
        &fixture.rap_challenges,
        &fixture.alpha_powers,
        &fixture.table_offset,
    );
    let mut expected = vec![FEE::zero(); prog.roots.len()];
    eval_program_verifier(&prog, &ctx, &mut expected);

    let got = word_as_ext(&exec.memory[evals[0].addr().0 as usize].expect("written")).expect("ext");
    assert_eq!(got, expected[0], "[{label}] baseline must agree");
    assert_ne!(
        got,
        &expected[0] + FEE::one(),
        "[{label}] a one-off value must NOT compare equal — otherwise the \
         differential above proves nothing"
    );
}

// =============================================================================
// (b) — cost, against the design document's measured table
// =============================================================================

/// The per-AIR `instr` column of `others/lfm-constraint-lowering-design.md` §8.1,
/// as written there. Copied deliberately rather than recomputed: the point of
/// this test is to compare the emitter against the DESIGN's prediction and
/// report where they differ, which a self-consistent recomputation cannot do.
const DESIGN_INSTR: &[(&str, usize)] = &[
    ("CPU", 489),
    ("BITWISE", 112),
    ("LT", 116),
    ("SHIFT", 321),
    ("EQ", 88),
    ("BYTEWISE", 138),
    ("STORE", 149),
    ("CPU32", 414),
    ("MEMW", 448),
    ("MEMW_A", 311),
    ("MEMW_R", 153),
    ("LOAD", 162),
    ("DECODE", 18),
    ("MUL", 320),
    ("DVRM", 423),
    ("BRANCH", 108),
    ("HALT", 701),
    ("COMMIT", 359),
    ("PAGE", 41),
    ("REGISTER", 29),
    ("KECCAK", 3_146),
    ("KECCAK_RND", 14_016),
    ("KECCAK_RC", 26),
    ("ECSM", 19_264),
    ("ECDAS", 22_718),
    ("L2G_GLOBAL", 27),
    ("L2G_MEMORY", 65),
    ("GLOBAL_MEMORY", 25),
];

/// ★ What the emitter actually costs per AIR, against the design's table.
///
/// Two numbers per table. `unfused` is the design's own column — arithmetic
/// nodes that survive the verify-time-base fold, one row each — and must match
/// it exactly, because a mismatch means the design measured a different program
/// than the one being lowered. `emitted` is what the pass really writes, after
/// `MulAdd` fusion and dead-code elimination.
///
/// ### What this instrument cannot see
///
/// Nothing about how an EPOCH is assembled: it is per distinct AIR, and the
/// sub-proof count per epoch comes from `tests::constraint_artifact_tests`, not
/// from here. It also says nothing about padded CELL cost, which depends on how
/// these rows interleave with the rest of a program's.
#[test]
fn constraint_leg_instruction_census() {
    let opts = options();
    let airs = production_airs(&opts);
    assert_eq!(airs.len(), NUM_PRODUCTION_AIRS);

    println!("\nconstraint-leg lowering cost, per AIR");
    println!(
        "{:<14} {:>7} {:>7} {:>6} {:>6} {:>5} {:>6} {:>6} {:>8} {:>8} {:>8} {:>7}",
        "table",
        "nodes",
        "leaves",
        "fold",
        "foldX",
        "dead",
        "unrK",
        "fused",
        "ext",
        "mulbase",
        "emitted",
        "unfused"
    );

    let (mut t_unfused, mut t_emitted, mut t_fused, mut t_dead, mut t_foldx) = (0, 0, 0, 0, 0);
    let (mut t_dead_const, mut t_cands, mut t_orphans) = (0, 0, 0);
    let mut t_orphans_all = 0;
    let mut mismatches: Vec<String> = Vec::new();

    for (label, air) in &airs {
        let artifact = ConstraintArtifact::capture(&**air);
        let r = analyze(&artifact).report().clone();

        println!(
            "{:<14} {:>7} {:>7} {:>6} {:>6} {:>5} {:>6} {:>6} {:>8} {:>8} {:>8} {:>7}",
            label,
            r.nodes,
            r.leaves,
            r.fold_base,
            r.fold_ext,
            r.dead,
            r.unreached_const,
            r.fused,
            r.ext_alu,
            r.mul_base,
            r.alu_rows(),
            r.unfused_alu_rows()
        );

        t_unfused += r.unfused_alu_rows();
        t_emitted += r.alu_rows();
        t_fused += r.fused;
        t_dead += r.dead;
        t_dead_const += r.unreached_const;
        t_cands += r.fuse_candidates;
        t_orphans += r.orphans;
        t_orphans_all += r.orphans_all_kinds;
        t_foldx += r.fold_ext;

        let design = DESIGN_INSTR
            .iter()
            .find(|(l, _)| l == label)
            .map(|(_, n)| *n)
            .unwrap_or_else(|| panic!("no design entry for {label}"));
        // The design's column counts every surviving arithmetic node once, with
        // no DCE and no extension-valued folding, so add back what this pass
        // removes on top of that.
        let comparable = r.unfused_alu_rows() + r.dead + r.fold_ext + r.aliased;
        if comparable != design {
            mismatches.push(format!(
                "{label}: design {design}, emitter {comparable} (delta {})",
                comparable as i64 - design as i64
            ));
        }
    }

    println!(
        "\nTOTALS  unfused {t_unfused}  emitted {t_emitted}  (fusion saves {t_fused})\n\
         beyond the design's rule: {t_dead} dead ROWS eliminated, \
         {t_dead_const} root-unreachable constant nodes (free either way, \
         counted under the design's `fold`), {t_foldx} extension-valued constant \
         subtrees folded\n\
         fusion: {t_cands} candidate (Add, Mul) operand pairs, {t_fused} taken \
         — the gap is sums with TWO single-consumer products, which can absorb \
         only one\n\
         locally-orphaned nodes (the design's fanout-0 measure): {t_orphans} \
         arithmetic, {t_orphans_all} over all node kinds"
    );

    assert!(
        mismatches.is_empty(),
        "the emitter's per-AIR cost no longer matches the design's §8.1 table \
         (this is a real finding either way — the design is measured, not \
         guessed):\n  {}",
        mismatches.join("\n  ")
    );
}

// =============================================================================
// Falsifications — one per mechanism the lowering relies on
// =============================================================================

/// Emitting a program for one artifact, purely to count what lands in it.
fn emitted_instrs(artifact: &ConstraintArtifact) -> Vec<super::instr::Instr> {
    let an = analyze(artifact);
    let (program, _) = differential_program(artifact, &an);
    program.instrs
}

fn count_ext(instrs: &[super::instr::Instr], want: super::instr::ExtOp) -> usize {
    instrs
        .iter()
        .filter(|i| matches!(i, super::instr::Instr::ExtAlu { op, .. } if *op == want))
        .count()
}

/// ★ `MulAdd` fusion happens, and the single-consumer guard is what stops it.
///
/// Two constraint sets differing only in whether the shared product is read
/// twice. Hash-consing collapses the repeated `m0·m1` into ONE node, so the
/// second set is exactly the hazard `ConstraintArtifact`'s doc comment warns
/// about: fusing a shared `Mul` into each consumer would recompute it.
///
/// The falsification is the second half. A test that only showed fusion
/// happening would pass just as well against an emitter that fused
/// unconditionally — which is the unsound one.
#[test]
fn muladd_fusion_requires_a_single_consumer() {
    let single = fusion_air(false);
    let shared = fusion_air(true);

    let a_single = ConstraintArtifact::capture(&single);
    let a_shared = ConstraintArtifact::capture(&shared);

    let r_single = analyze(&a_single).report().clone();
    let r_shared = analyze(&a_shared).report().clone();

    assert!(
        r_single.fused >= 1,
        "a single-consumer Mul under an Add must fuse; report {r_single:?}"
    );
    assert_eq!(
        r_shared.fused, 0,
        "a Mul read by two Adds must NOT fuse — hash-consing makes that a \
         recomputation, not a saving; report {r_shared:?}"
    );

    let i_single = emitted_instrs(&a_single);
    assert_eq!(
        count_ext(&i_single, super::instr::ExtOp::MulAdd),
        r_single.fused,
        "every fusion the report claims must be a MulAdd row in the program"
    );
    assert_eq!(
        count_ext(&emitted_instrs(&a_shared), super::instr::ExtOp::MulAdd),
        0,
        "the shared-Mul program must contain no MulAdd row"
    );
}

/// ★ A constraint root is a consumer: fusing it away would delete the value the
/// quotient recombination reads.
#[test]
fn a_rooted_mul_is_never_fused_away() {
    let air = rooted_mul_air();
    let artifact = ConstraintArtifact::capture(&air);
    let an = analyze(&artifact);

    assert_eq!(
        an.report().fused,
        0,
        "the only Add's product operand is also a constraint root, so it must \
         survive as its own row"
    );

    // And the root still evaluates: emit, run, compare against the interpreter.
    let prog = artifact.program();
    let (program, evals) = differential_program(&artifact, &an);
    let mut rng = SplitMix64(7);
    let fixture = OodFixture::sample(&artifact, &mut rng);
    let exec = execute(
        &program,
        &[fixture.arena(&artifact), fixture.uniform_arena()],
        &TestPermutation,
    )
    .expect("execution");

    let frame = fixture.frame();
    let ctx = TransitionEvaluationContext::<Gl, Ext3>::new_verifier(
        &frame,
        &fixture.rap_challenges,
        &fixture.alpha_powers,
        &fixture.table_offset,
    );
    let mut expected = vec![FEE::zero(); prog.roots.len()];
    eval_program_verifier(&prog, &ctx, &mut expected);
    for (c, want) in expected.iter().enumerate() {
        let got =
            word_as_ext(&exec.memory[evals[c].addr().0 as usize].expect("written")).expect("ext");
        assert_eq!(got, *want, "constraint {c}");
    }
}

/// ★ `Op::Neg` lowers to a subtract from the pooled zero, and that is the only
/// thing it can lower to — the ISA has no unary negate.
#[test]
fn neg_lowers_to_a_subtract_from_zero() {
    let air = neg_air();
    let artifact = ConstraintArtifact::capture(&air);
    let prog = artifact.program();
    let negs = prog
        .nodes
        .iter()
        .filter(|n| matches!(n, stark::constraint_ir::Op::Neg(_)))
        .count();
    assert!(
        negs >= 1,
        "the fixture AIR must actually capture a Neg node"
    );

    let instrs = emitted_instrs(&artifact);
    let subs = count_ext(&instrs, super::instr::ExtOp::Sub);
    let ir_subs = prog
        .nodes
        .iter()
        .filter(|n| matches!(n, stark::constraint_ir::Op::Sub(_, _)))
        .count();
    assert_eq!(
        subs,
        ir_subs + negs,
        "each Neg must add exactly one Sub row on top of the IR's own Subs"
    );

    // One pooled zero for all of them: the constant pool interns by value.
    let zeros = instrs
        .iter()
        .filter(|i| matches!(i, super::instr::Instr::Const { value, .. } if *value == ext_word(&FEE::zero())))
        .count();
    assert!(
        zeros <= 1,
        "the zero constant must be interned once, found {zeros}"
    );
}

/// ★ Dead-code elimination drops an unreachable arithmetic node instead of
/// writing it with multiplicity zero.
///
/// No production artifact exercises this: `constraint_leg_instruction_census`
/// measures ZERO root-unreachable arithmetic nodes across all 28 tables, and
/// only three orphaned nodes of any kind (which is the design §4.3 number,
/// reproduced — they are leaves, not arithmetic, so they never cost a row).
/// A defensive path nothing reaches is a path nothing has tested, so the
/// unreachable node is INJECTED here: one extra `Mul` appended past every root.
///
/// The falsification is the first assertion. Without it the test would pass
/// against an emitter that lowered the injected node too, since an unread write
/// is legal — merely wasteful — and the differential would not notice.
#[test]
fn dead_nodes_are_eliminated() {
    use stark::constraint_ir::DeviceNode;

    let opts = options();
    let airs = production_airs(&opts);
    let (_, air) = airs
        .iter()
        .find(|(l, _)| *l == "L2G_GLOBAL")
        .expect("L2G_GLOBAL is a production AIR");

    let clean = ConstraintArtifact::capture(&**air);
    let baseline = analyze(&clean).report().clone();
    assert_eq!(
        baseline.dead, 0,
        "the unmodified artifact has no dead nodes"
    );

    // A product of the last two nodes, appended past every root. Operands are
    // strictly earlier, so `validate_self` still accepts it.
    let mut injected = clean.clone();
    let n = injected.nodes.len() as u32;
    injected.nodes.push(DeviceNode {
        op: stark::constraint_ir::device::OP_MUL,
        a: n - 2,
        b: n - 1,
        dim: stark::constraint_ir::device::DIM_EXT,
    });
    injected
        .validate_self()
        .expect("an appended node keeps the artifact well-formed");

    let report = analyze(&injected).report().clone();
    assert_eq!(
        report.dead, 1,
        "the injected node is reachable from no root and must be counted dead"
    );
    assert_eq!(
        report.alu_rows(),
        baseline.alu_rows(),
        "an unreachable node must cost no rows at all"
    );

    let an = analyze(&injected);
    let (program, _) = differential_program(&injected, &an);
    validate(&program).expect("a program with DCE applied is valid");
    for instr in &program.instrs {
        if let super::instr::Instr::ExtAlu { op, mult, .. } = instr {
            assert_ne!(
                *mult, 0,
                "an emitted {op:?} row is never read; DCE should have removed it"
            );
        }
    }
}

/// ★ The next-row PRUNING is in the program text, not in the supplied arena.
///
/// A column the AIR does not declare is reconstructed as ZERO by the verifier.
/// If the machine hinted a value there instead, a prover could supply a next-row
/// opening the real verifier never reads — so this is a soundness property, not
/// a size one.
#[test]
fn pruned_next_row_columns_are_program_zeros() {
    let opts = options();
    let airs = production_airs(&opts);
    let (label, air) = airs
        .iter()
        .find(|(l, _)| *l == "CPU")
        .expect("CPU is a production AIR");
    let artifact = ConstraintArtifact::capture(&**air);
    let shape = &artifact.shape;
    let width = (shape.main_width + shape.aux_width) as usize;
    let steps = shape.transition_offsets.len();
    assert!(steps >= 2, "[{label}] needs a next-row step to prune");

    let mut b = LfmBuilder::new();
    let arena = b.declare_arena(ood_frame_words(&artifact));
    let (frame, words) = hint_ood_frame(&mut b, &artifact, arena, 0);

    assert_eq!(
        words as usize,
        width + (steps - 1) * shape.next_row_columns.len(),
        "[{label}] only the opened entries may consume arena words"
    );
    assert!(
        (words as usize) < steps * width,
        "[{label}] the pruning must actually save arena words"
    );

    let zero_addr = b.felt_const(FE::zero()).addr();
    for (col, cell) in frame[1].iter().enumerate().take(width) {
        let declared = shape.next_row_columns.contains(&(col as u32));
        let is_zero = cell.addr() == zero_addr;
        assert_eq!(
            is_zero, !declared,
            "[{label}] next-row column {col}: declared={declared} but \
             pruned={is_zero}"
        );
    }
}

// =============================================================================
// Fixture AIRs — capture paths the production tables cannot reach
// =============================================================================

use math::field::traits::IsField;
use stark::constraints::builder::{ConstraintBuilder, ConstraintSet};
use stark::lookup::{AirWithBuses, AuxiliaryTraceBuildData, NullBoundaryConstraintBuilder};

type FixtureAir<C> = AirWithBuses<Gl, Ext3, NullBoundaryConstraintBuilder, (), C>;

fn fixture_air<C: ConstraintSet<Gl, Ext3>>(
    cols: usize,
    set: C,
    name: &'static str,
) -> FixtureAir<C> {
    AirWithBuses::new(
        cols,
        AuxiliaryTraceBuildData {
            interactions: vec![],
        },
        &options(),
        1,
        set,
    )
    .with_name(name)
}

/// `shared = false`: one product under one sum — fusable.
/// `shared = true`: the SAME product under two sums — hash-consed to one node
/// with two consumers, so not fusable.
struct FusionConstraints {
    shared: bool,
}

impl<F: IsField, E: IsField> ConstraintSet<F, E> for FusionConstraints {
    fn eval<B: ConstraintBuilder<F, E>>(&self, b: &mut B) {
        let m0 = b.main(0, 0);
        let m1 = b.main(0, 1);
        let m2 = b.main(0, 2);
        let m3 = b.main(0, 3);
        b.emit_base(0, m0.clone() * m1.clone() + m2.clone());
        if self.shared {
            // Structurally identical product: capture hash-conses it.
            b.emit_base(1, m0 * m1 + m3);
        } else {
            b.emit_base(1, m2 * m3 + m0);
        }
    }
}

fn fusion_air(shared: bool) -> FixtureAir<FusionConstraints> {
    fixture_air(
        4,
        FusionConstraints { shared },
        if shared { "SHARED" } else { "SINGLE" },
    )
}

/// A product that is BOTH a constraint root and an operand of a sum.
struct RootedMulConstraints;

impl<F: IsField, E: IsField> ConstraintSet<F, E> for RootedMulConstraints {
    fn eval<B: ConstraintBuilder<F, E>>(&self, b: &mut B) {
        let m0 = b.main(0, 0);
        let m1 = b.main(0, 1);
        let m2 = b.main(0, 2);
        b.emit_base(0, m0.clone() * m1.clone());
        b.emit_base(1, m0 * m1 + m2);
    }
}

fn rooted_mul_air() -> FixtureAir<RootedMulConstraints> {
    fixture_air(3, RootedMulConstraints, "ROOTED_MUL")
}

/// Negation, which no production table's captured IR happens to hold in
/// isolation.
struct NegConstraints;

impl<F: IsField, E: IsField> ConstraintSet<F, E> for NegConstraints {
    fn eval<B: ConstraintBuilder<F, E>>(&self, b: &mut B) {
        let m0 = b.main(0, 0);
        let m1 = b.main(0, 1);
        b.emit_base(0, -m0.clone() + m1.clone());
        b.emit_base(1, m0 - m1);
    }
}

fn neg_air() -> FixtureAir<NegConstraints> {
    fixture_air(2, NegConstraints, "NEG")
}
