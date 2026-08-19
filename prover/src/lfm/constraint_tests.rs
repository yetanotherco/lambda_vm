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
use super::word::{LfmWord, base_word, ext_word, word_as_ext};

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
/// challenges — so this shortcut is a property of the isolated slice. The
/// guard asserting it (`challenges_are_not_an_arena_in_the_assembled_verifier`)
/// cannot exist until the assembled verifier does; it is owed as an OPEN entry
/// in `others/lfm-assembly-obligations.md`, not by this file.
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
    // 14_016 → 12_998: main replaced the θ/ρ HWSL lookups with inline μ-gated
    // linear identities in `KeccakRndConstraints`, which nets −1018 constraint
    // arithmetic rows (the same change whose receiver-side multiplicity drop is
    // reconciled in `keccak_adapter::bitwise_ops_for`).
    ("KECCAK_RND", 12_998),
    ("KECCAK_RC", 26),
    ("ECSM", 19_264),
    ("ECDAS", 22_718),
    ("HINT", 418), // main's new receiver AIR for the `hint` ecall.
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
    use stark::constraint_ir::ArtifactNode;

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
    injected.nodes.push(ArtifactNode {
        op: stark::constraint_ir::device::OP_MUL,
        a: n - 2,
        b: n - 1,
        dim: stark::constraint_ir::artifact::DIM_EXT,
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

// =============================================================================
// (c) + (d) — the quotient recombination, against a REAL proof
// =============================================================================

use crypto::fiat_shamir::is_transcript::IsTranscript;
use stark::config::DefaultStarkTranscript;
use stark::domain::new_verifier_domain;
use stark::lookup::{BusPublicInputs, LOGUP_CHALLENGE_ALPHA, LOGUP_NUM_CHALLENGES};
use stark::proof::stark::MultiProof;
use stark::proof::view::StarkProofView;
use stark::table::Table;
use stark::traits::AIR;
use stark::verifier::{Challenges, IsStarkVerifier, Verifier};

use super::constraints::{BoundaryTerm, QuotientShape, emit_quotient};
use super::proof::{lfm_prove, verify_against};
use super::registry::build_artifacts;

/// A genuine STARK proof of a production AIR, opened up far enough that the
/// machine can be asked to redo the verifier's composition check on it.
///
/// Everything here is READ OFF a real proof or replayed from a real transcript.
/// Nothing is synthesized: the OOD frame is the prover's, the composition parts
/// are the prover's, and the challenges come out of the production verifier's
/// own `replay_rounds_after_round_1` rather than a local Fiat-Shamir model.
pub(super) struct RealSubProof {
    pub(super) artifact: ConstraintArtifact,
    pub(super) ood_full: Table<Ext3>,
    pub(super) main_width: usize,
    pub(super) num_steps: usize,
    pub(super) rap_challenges: Vec<FEE>,
    pub(super) alpha_powers: Vec<FEE>,
    pub(super) table_offset: FEE,
    /// The table's total bus contribution `L`, undivided. The machine derives
    /// `L/N` from THIS cell rather than reading a second arena word — see
    /// `constraints::emit_table_offset` for why that is a soundness
    /// requirement rather than a saving.
    pub(super) contribution: FEE,
    pub(super) zeta: FEE,
    pub(super) beta: FEE,
    pub(super) challenges: Challenges<Ext3>,
    pub(super) claimed_parts: Vec<FEE>,
    pub(super) quotient: QuotientShape,
}

/// Proves L2G_MEMORY — a real continuation table, and the only continuation AIR
/// with genuine constraints — over a real boundary-claim trace.
///
/// Returns the AIR alongside the proof because the DEEP differential needs both:
/// its oracle is the production reconstruction, which takes the AIR's layout and
/// the proof's own openings.
pub(super) fn real_fixture() -> (BoxedAir, MultiProof<Gl, Ext3, ()>) {
    use crate::tables::local_to_global::{
        CellBoundary, FiniClaim, InitClaim, generate_local_to_global_trace,
    };
    use crate::test_utils::{EPOCH_TEST_LABEL, multi_prove_ram};

    let opts = options();
    let air = crate::continuation::l2g_memory_air(&opts, EPOCH_TEST_LABEL);

    let boundaries: Vec<CellBoundary> = (0..4u64)
        .map(|i| CellBoundary {
            address: 0x1000 + 8 * i,
            init: InitClaim {
                value: i + 1,
                timestamp: 0,
                originating_epoch: 0,
            },
            fini: FiniClaim {
                value: 2 * i + 3,
                epoch: EPOCH_TEST_LABEL,
                timestamp: 17 + i,
            },
        })
        .collect();
    let mut trace = generate_local_to_global_trace(&boundaries);

    let pairs: Vec<(
        &dyn AIR<Field = Gl, FieldExtension = Ext3, PublicInputs = ()>,
        _,
        _,
    )> = vec![(&air, &mut trace, &())];
    let proof = multi_prove_ram(pairs, &mut DefaultStarkTranscript::<Ext3>::new(&[]))
        .expect("the L2G_MEMORY fixture must prove");

    (Box::new(air), proof)
}

pub(super) type BoxedAir = Box<dyn AIR<Field = Gl, FieldExtension = Ext3, PublicInputs = ()>>;

pub(super) fn real_sub_proof() -> RealSubProof {
    let (air, proof) = real_fixture();
    open_sub_proof(&*air, &proof)
}

/// Replays the production verifier's rounds over a real single-table proof and
/// packages everything the constraint leg needs.
pub(super) fn open_sub_proof(
    air: &dyn AIR<Field = Gl, FieldExtension = Ext3, PublicInputs = ()>,
    proof: &MultiProof<Gl, Ext3, ()>,
) -> RealSubProof {
    let view = StarkProofView::Owned(&proof.proofs[0]);

    // ---- Round 1, Phase A/B/C, transcribed from `multi_verify_views` for the
    // single-table case (no per-table domain separator).
    let mut transcript = DefaultStarkTranscript::<Ext3>::new(&[]);
    if air.is_preprocessed() {
        transcript.append_bytes(&air.precomputed_commitment());
    }
    transcript.append_bytes(view.lde_trace_main_merkle_root());
    let rap_challenges: Vec<FEE> = if air.has_aux_trace() {
        (0..LOGUP_NUM_CHALLENGES)
            .map(|_| transcript.sample_field_element())
            .collect()
    } else {
        Vec::new()
    };
    if let Some(root) = view.lde_trace_aux_merkle_root() {
        transcript.append_bytes(root);
    }
    if let Some(contribution) = view.bus_table_contribution() {
        transcript.append_field_element(&contribution);
    }

    let trace_length = view.trace_length();
    let domain = new_verifier_domain(air, trace_length);
    let layout = Verifier::ood_layout(air);
    let challenges: Challenges<Ext3> = Verifier::replay_rounds_after_round_1(
        air,
        view,
        &(),
        &domain,
        &mut transcript,
        rap_challenges.clone(),
        &layout,
    );

    // ---- β, recovered from the verifier's own coefficient run and CHECKED.
    //
    // `replay_rounds_after_round_1` expands one geometric run of β and splits it
    // into the transition coefficients then the boundary ones. That split is
    // exactly the term ordering `emit_quotient` folds, so asserting it here —
    // against the verifier's values, not a model — is what pins the Horner.
    let nt = challenges.transition_coeffs.len();
    assert_eq!(
        challenges.transition_coeffs[0],
        FEE::one(),
        "the coefficient run starts at beta^0"
    );
    let beta = challenges.transition_coeffs[1];
    for (c, coeff) in challenges.transition_coeffs.iter().enumerate() {
        assert_eq!(*coeff, beta.pow(c as u64), "transition coefficient {c}");
    }
    for (k, coeff) in challenges.boundary_coeffs.iter().enumerate() {
        assert_eq!(
            *coeff,
            beta.pow((nt + k) as u64),
            "boundary coefficient {k} must continue the same run past the \
             transition constraints"
        );
    }

    // ---- the OOD grid, reconstructed by the verifier's own layout so the
    // pruning is not modelled here.
    let ood_current = view.trace_ood_evaluations();
    let ood_next = view.trace_ood_next_evaluations();
    let ood_full = layout.reconstruct_full(
        ood_current.row_major_data(),
        ood_current.width(),
        ood_next.row_major_data(),
    );

    let (main_width, _) = air.trace_layout();
    let bus_public_inputs = view
        .bus_table_contribution()
        .map(BusPublicInputs::from_contribution);
    let logup_alpha_powers: Vec<FEE> = if rap_challenges.len() > LOGUP_CHALLENGE_ALPHA {
        let alpha = rap_challenges[LOGUP_CHALLENGE_ALPHA];
        (0..air.max_bus_elements())
            .map(|i| alpha.pow(i as u64))
            .collect()
    } else {
        Vec::new()
    };
    let contribution = view.bus_table_contribution().unwrap_or_else(FEE::zero);
    let table_offset = FE::from(trace_length as u64)
        .inv()
        .expect("a nonzero trace length")
        * contribution;

    let boundary_constraints = air.boundary_constraints(
        &(),
        &rap_challenges,
        bus_public_inputs.as_ref(),
        trace_length,
    );
    // `VerifierDomain::trace_primitive_root` is crate-private, so the generator
    // is rederived the same way `new_verifier_domain` does: the root of unity of
    // order `trace_length`.
    let generator = <Gl as math::field::traits::IsFFTField>::get_primitive_root_of_unity(
        trace_length.trailing_zeros() as u64,
    )
    .expect("a power-of-two trace length has a root of unity");
    let boundary: Vec<BoundaryTerm> = boundary_constraints
        .constraints
        .iter()
        .map(|c| BoundaryTerm {
            col: if c.is_aux { main_width + c.col } else { c.col },
            point: generator.pow(c.step as u64),
            value: c.value,
        })
        .collect();

    let claimed_parts: Vec<FEE> = view.composition_poly_parts_ood_evaluation().to_vec();
    let artifact = ConstraintArtifact::capture(air);

    RealSubProof {
        num_steps: artifact.shape.transition_offsets.len(),
        quotient: QuotientShape {
            log2_trace_length: trace_length.trailing_zeros(),
            num_composition_parts: claimed_parts.len(),
            boundary,
        },
        artifact,
        ood_full,
        main_width,
        rap_challenges,
        alpha_powers: logup_alpha_powers,
        table_offset,
        contribution,
        zeta: challenges.z,
        beta,
        challenges,
        claimed_parts,
    }
}

impl RealSubProof {
    /// The OOD frame as the machine's arena sees it: opened entries only, in
    /// [`hint_ood_frame`]'s order.
    fn frame_arena(&self) -> Vec<LfmWord> {
        let shape = &self.artifact.shape;
        let width = (shape.main_width + shape.aux_width) as usize;
        let mut out = Vec::new();
        for offset in 0..self.num_steps {
            let row = self.ood_full.get_row(offset);
            for (col, v) in row.iter().enumerate().take(width) {
                if offset == 0 || shape.next_row_columns.contains(&(col as u32)) {
                    out.push(ext_word(v));
                }
            }
        }
        out
    }

    fn uniform_arena(&self) -> Vec<LfmWord> {
        // `alpha_powers` are DERIVED in-machine from `rap_challenges[ALPHA]`
        // (`constraints::emit_alpha_powers`), so they are deliberately absent
        // here — a hinted power is a claim about alpha that nothing checks.
        self.rap_challenges
            .iter()
            .chain([&self.contribution, &self.zeta, &self.beta])
            .map(ext_word)
            .collect()
    }

    fn parts_arena(&self) -> Vec<LfmWord> {
        self.claimed_parts.iter().map(ext_word).collect()
    }

    pub(super) fn arenas(&self) -> Vec<Vec<LfmWord>> {
        vec![self.frame_arena(), self.uniform_arena(), self.parts_arena()]
    }
}

/// The full composition-check program for one sub-proof: lower the AIR's
/// transition constraints at ζ, recombine them against the shared zerofier and
/// the boundary quotient, and ASSERT the result equals the composition value the
/// proof claims.
///
/// The assert is the point. A program that merely computed the composition would
/// be a calculator; asserting it against the claimed parts is what makes the
/// machine's acceptance mean something, and it is what the tamper vectors below
/// have to break.
fn composition_program_source(sp: &RealSubProof) -> super::builder::LfmProgramSource {
    let mut b = LfmBuilder::new();

    let frame_arena = b.declare_arena(ood_frame_words(&sp.artifact));
    let (steps, _) = hint_ood_frame(&mut b, &sp.artifact, frame_arena, 0);

    // The alpha POWERS are no longer hinted: they are derived from the one
    // alpha challenge, so the uniform arena is that much shorter.
    let num_uniforms = (sp.rap_challenges.len() + 3) as u32;
    let uniform_arena = b.declare_arena(num_uniforms);
    let mut next = 0u32;
    let mut take = |b: &mut LfmBuilder| {
        let c = b.hint_word(uniform_arena, next).as_ext();
        next += 1;
        c
    };
    let rap_challenges: Vec<_> = (0..sp.rap_challenges.len()).map(|_| take(&mut b)).collect();
    let alpha_powers = super::constraints::emit_alpha_powers(
        &mut b,
        rap_challenges[stark::lookup::LOGUP_CHALLENGE_ALPHA],
        sp.alpha_powers.len(),
    );
    // `L`, undivided. The per-row offset is DERIVED from it so the constraint
    // leg and the LogUp closure consume one cell rather than two independently
    // hinted ones (`constraints::emit_table_offset`).
    let contribution = take(&mut b);
    let table_offset =
        super::constraints::emit_table_offset(&mut b, contribution, sp.quotient.log2_trace_length);
    let zeta = take(&mut b);
    let beta = take(&mut b);

    let parts_arena = b.declare_arena(sp.claimed_parts.len() as u32);
    let claimed_parts: Vec<_> = (0..sp.claimed_parts.len() as u32)
        .map(|i| b.hint_word(parts_arena, i).as_ext())
        .collect();

    let ood = OodOperands {
        steps,
        main_width: sp.main_width,
        rap_challenges,
        alpha_powers,
        table_offset,
    };
    let (evals, _) = super::constraints::emit_constraint_evals(&mut b, &sp.artifact, &ood);
    let q = emit_quotient(
        &mut b,
        &sp.quotient,
        &ood,
        zeta,
        beta,
        &evals,
        &claimed_parts,
    );

    b.assert_eq_ext(q.claimed, q.composition);
    b.public(q.composition.as_cell());
    b.finish()
}

/// ★ (c) The machine reproduces the verifier's composition check on a REAL
/// proof of a REAL production table.
///
/// The oracle is the proof itself: an honestly generated proof satisfies
/// `Σ_j part_j·ζ^j = boundary_quotient + Σ_c β^c·C_c/Z`, so a machine that
/// computes either side differently cannot execute the in-machine assert. That
/// makes this a differential against the production prover and verifier
/// together, not against a transcription of one formula.
#[test]
fn composition_check_matches_a_real_proof() {
    let sp = real_sub_proof();

    // Make the coverage legible, and fail rather than silently degrade if the
    // fixture ever stops exercising a term.
    assert_eq!(
        sp.quotient.boundary.len(),
        1,
        "L2G_MEMORY has bus interactions, so it carries the framework's \
         acc[0] = 0 boundary constraint — without it the boundary half of the \
         recombination would be untested"
    );
    assert!(
        sp.quotient.log2_trace_length >= 2,
        "the zerofier must cost more than a squaring or two"
    );
    assert!(
        !sp.alpha_powers.is_empty(),
        "the LogUp uniforms must be live"
    );

    let program = compile(composition_program_source(&sp));
    validate(&program).expect("the composition program is admissible");

    let leg = analyze(&sp.artifact).report().clone();
    println!(
        "L2G_MEMORY composition check: {} instructions total, of which {} are \
         the constraint leg ({} constraints, {} parts, log2(N) = {})",
        program.instrs.len(),
        leg.alu_rows(),
        sp.artifact.roots.len(),
        sp.quotient.num_composition_parts,
        sp.quotient.log2_trace_length,
    );

    let exec = execute(&program, &sp.arenas(), &TestPermutation)
        .expect("an honest proof's composition check must execute");

    // The published value is the recomputed composition; it must equal the
    // Horner fold of the parts the proof carries.
    let expected = sp
        .claimed_parts
        .iter()
        .rev()
        .fold(FEE::zero(), |acc, part| acc * sp.zeta + part);
    let (_, word) = exec.public_words[0];
    assert_eq!(
        word_as_ext(&word).expect("ext"),
        expected,
        "the machine's composition must equal the claimed composition"
    );
    assert!(
        expected != FEE::zero(),
        "a zero composition would make the assert vacuous"
    );
}

/// ★ (c) falsification: every input the check depends on, broken one at a time.
///
/// Each vector leaves a genuine proof's data in place and changes exactly one
/// word. The in-machine `assert_eq_ext` lowers to `diff / ZERO`, which under the
/// machine's `x/0 = error` convention makes a mismatching run UNEXECUTABLE — the
/// earliest and loudest failure, and the one that shows the assert is carrying
/// the check rather than decorating it.
#[test]
fn a_tampered_composition_input_cannot_execute() {
    let sp = real_sub_proof();
    let program = compile(composition_program_source(&sp));
    execute(&program, &sp.arenas(), &TestPermutation).expect("baseline honest run");

    /// One tamper: a name and the single word it corrupts.
    type Vector = (&'static str, Box<dyn Fn(&mut Vec<Vec<LfmWord>>)>);

    let vectors: Vec<Vector> = vec![
        (
            "a wrong OOD frame value",
            Box::new(|a: &mut Vec<Vec<LfmWord>>| a[0][0][0] += FE::one()),
        ),
        (
            "a wrong LogUp challenge",
            Box::new(|a: &mut Vec<Vec<LfmWord>>| a[1][0][0] += FE::one()),
        ),
        (
            "a wrong out-of-domain point zeta",
            Box::new(|a: &mut Vec<Vec<LfmWord>>| {
                let i = a[1].len() - 2;
                a[1][i][0] += FE::one();
            }),
        ),
        (
            "a wrong composition challenge beta",
            Box::new(|a: &mut Vec<Vec<LfmWord>>| {
                let i = a[1].len() - 1;
                a[1][i][0] += FE::one();
            }),
        ),
        (
            "a wrong claimed composition part",
            Box::new(|a: &mut Vec<Vec<LfmWord>>| a[2][0][0] += FE::one()),
        ),
    ];

    // A zeta ON the trace domain, which makes the zerofier vanish. The
    // out-of-domain sampler cannot produce it, but the constraint leg does not
    // contain the sampler, so the reciprocal guard rather than an argument about
    // a component elsewhere is what rules it out. `g` is chosen over `1` on
    // purpose: at zeta = 1 the BOUNDARY denominator vanishes too, and the run
    // would fail without saying which guard caught it.
    {
        let generator = <Gl as math::field::traits::IsFFTField>::get_primitive_root_of_unity(
            sp.quotient.log2_trace_length as u64,
        )
        .expect("root of unity");
        assert_eq!(
            generator.pow(1u64 << sp.quotient.log2_trace_length),
            FE::one(),
            "g^N = 1, so the zerofier vanishes at zeta = g"
        );
        assert_ne!(
            generator,
            FE::one(),
            "but the boundary denominator does not"
        );
        let mut arenas = sp.arenas();
        let i = arenas[1].len() - 2;
        arenas[1][i] = ext_word(&generator.to_extension::<Ext3>());
        assert!(
            execute(&program, &arenas, &TestPermutation).is_err(),
            "a zeta on the trace domain must be rejected by the zerofier's \
             reciprocal guard, not silently return 0/0 = 1"
        );
    }

    for (what, tamper) in vectors {
        let mut arenas = sp.arenas();
        tamper(&mut arenas);
        assert!(
            execute(&program, &arenas, &TestPermutation).is_err(),
            "{what} must make the composition check unexecutable"
        );
    }
}

/// ★ (d) The composition check PROVES and VERIFIES.
///
/// Per method rule 2 this is the only test in this file that says anything about
/// the chips: everything above runs the executor, which mirrors the very ALU it
/// is checking. Here the emitted rows are proved by `LFM_XALU` and friends and
/// the proof is verified against the program's own committed artifacts.
///
/// It uses `verify_against` rather than the registry. That is deliberate and is
/// the sanctioned path for a shape that is not registered: this program is one
/// AIR's leg, not the epoch verifier, and pinning its digest would pin a shape
/// that has to move once the DEEP and opening legs land.
#[test]
fn constraint_leg_proves_and_verifies() {
    let opts = options();
    let sp = real_sub_proof();
    let program = compile(composition_program_source(&sp));
    let artifacts = build_artifacts(&program, &opts);

    let proved = lfm_prove(&program, &artifacts, &sp.arenas(), &opts)
        .expect("the honest composition check must execute and prove");

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
        "the proved composition check must verify"
    );

    // A verifier that claims a different composition value must reject, even
    // though the proof itself is untouched: the claimed public words are what
    // bind the machine's output to the statement.
    let mut wrong = proved.public_words.clone();
    wrong[0].1[0] += FE::one();
    assert!(
        !verify_against(
            &artifacts.roots,
            &artifacts.program_id,
            artifacts.keccak_rnd_chunks,
            &proved.proof,
            &wrong,
            &opts,
            artifacts.hasher,
            artifacts.chip_set,
        ),
        "a mismatched claimed composition must be rejected"
    );
}

/// The emitted program is deterministic — same builder calls, same instructions,
/// same digest. That is the property registration would pin, asserted here for a
/// shape that is deliberately not in `LFM_REGISTRY`.
#[test]
fn composition_program_is_deterministic() {
    let sp = real_sub_proof();
    let a = compile(composition_program_source(&sp));
    let b = compile(composition_program_source(&sp));
    assert_eq!(a.instrs.len(), b.instrs.len());
    assert_eq!(a.num_addrs, b.num_addrs);
    let opts = options();
    assert_eq!(
        build_artifacts(&a, &opts).program_id,
        build_artifacts(&b, &opts).program_id,
        "the same source must produce the same program identity"
    );
}

// =============================================================================
// (b) — the per-epoch budget
// =============================================================================

/// Rows the recombination costs for one sub-proof, MEASURED by emitting it into
/// a throwaway builder rather than counted off the source by eye.
///
/// The operand plumbing (hints for the frame, the challenges, the constraint
/// values and the claimed parts) is built twice into two independent builders —
/// once alone and once followed by the quotient — and the difference is the
/// quotient's own rows. Emission is deterministic, so the two plumbings are
/// identical by construction.
fn quotient_rows(artifact: &ConstraintArtifact, log2_trace_length: u32) -> usize {
    let shape = &artifact.shape;
    let width = (shape.main_width + shape.aux_width) as usize;
    let num_steps = shape.transition_offsets.len().max(1);
    let num_parts = shape.composition_degree_multiplier as usize;
    let num_constraints = artifact.roots.len();

    let quotient = QuotientShape {
        log2_trace_length,
        num_composition_parts: num_parts,
        // Every table with bus interactions carries the framework's single
        // acc[0] = 0 constraint on the last aux column, and no production table
        // declares any other boundary constraint.
        boundary: if shape.has_trace_interaction {
            vec![BoundaryTerm {
                col: width - 1,
                point: FE::one(),
                value: FEE::zero(),
            }]
        } else {
            Vec::new()
        },
    };

    let plumbing = |b: &mut LfmBuilder| {
        let total = num_steps * width + 2 + num_constraints + num_parts + 1;
        let arena = b.declare_arena(total as u32);
        let mut idx = 0u32;
        let mut take = |b: &mut LfmBuilder| {
            let c = b.hint_word(arena, idx).as_ext();
            idx += 1;
            c
        };
        let steps: Vec<Vec<_>> = (0..num_steps)
            .map(|_| (0..width).map(|_| take(b)).collect())
            .collect();
        let ood = OodOperands {
            steps,
            main_width: shape.main_width as usize,
            rap_challenges: Vec::new(),
            alpha_powers: Vec::new(),
            table_offset: take(b),
        };
        let zeta = take(b);
        let beta = take(b);
        let evals: Vec<_> = (0..num_constraints).map(|_| take(b)).collect();
        let parts: Vec<_> = (0..num_parts).map(|_| take(b)).collect();
        (ood, zeta, beta, evals, parts)
    };

    let mut bare = LfmBuilder::new();
    let _ = plumbing(&mut bare);
    let baseline = bare.finish().instrs.len();

    let mut full = LfmBuilder::new();
    let (ood, zeta, beta, evals, parts) = plumbing(&mut full);
    let q = emit_quotient(&mut full, &quotient, &ood, zeta, beta, &evals, &parts);
    full.assert_eq_ext(q.claimed, q.composition);
    full.finish().instrs.len() - baseline
}

/// ★ The constraint leg for a CONTINUATION EPOCH, against the design's budget.
///
/// The composition is `others/lfm-constraint-lowering-design.md` §8.2.2's, which
/// `tests::constraint_artifact_tests::continuation_epoch_constraint_leg` derives
/// from the real epoch shape and pins against a measured 24/25 sub-proof count:
/// 14 split-table families at one chunk each, plus the nine fixed tables an
/// intermediate epoch carries (all ten on the final one), plus one L2G_MEMORY.
/// PAGE does not appear — epochs pass `page_configs = &[]`.
///
/// ### What this instrument cannot see
///
/// It assumes the MINIMUM epoch, one chunk per family. A larger epoch adds
/// chunks of the cheap tables, which the design measures at +642 instructions
/// per doubling past 2^19 cycles. It also fixes one trace length for the
/// zerofier across every sub-proof, so the recombination term is a
/// representative figure rather than a per-chunk one.
#[test]
fn continuation_epoch_constraint_leg_cost() {
    /// Trace length assumed for the zerofier's squaring chain.
    const LOG2_TRACE_LENGTH: u32 = 20;

    /// The 14 chunked split-table families.
    const SPLIT_FAMILIES: &[&str] = &[
        "CPU", "LT", "SHIFT", "EQ", "BYTEWISE", "STORE", "CPU32", "MEMW", "MEMW_A", "MEMW_R",
        "LOAD", "MUL", "DVRM", "BRANCH",
    ];
    /// `FIXED_TABLE_COUNT`'s ten, which contribute exactly one sub-proof each
    /// regardless of `TableCounts`. HALT is last: an intermediate epoch drops it.
    const FIXED: &[&str] = &[
        "BITWISE",
        "DECODE",
        "COMMIT",
        "KECCAK",
        "KECCAK_RND",
        "KECCAK_RC",
        "REGISTER",
        "ECSM",
        "ECDAS",
        "HALT",
    ];

    let opts = options();
    let airs = production_airs(&opts);
    let cost: std::collections::BTreeMap<&str, (usize, usize, usize)> = airs
        .iter()
        .map(|(label, air)| {
            let artifact = ConstraintArtifact::capture(&**air);
            let r = analyze(&artifact).report().clone();
            (
                *label,
                (
                    r.alu_rows(),
                    r.unfused_alu_rows(),
                    quotient_rows(&artifact, LOG2_TRACE_LENGTH),
                ),
            )
        })
        .collect();

    let sum = |labels: &[&str], pick: fn(&(usize, usize, usize)) -> usize| -> usize {
        labels.iter().map(|l| pick(&cost[l])).sum()
    };

    let families = sum(SPLIT_FAMILIES, |c| c.0);
    let fixed_no_halt = sum(&FIXED[..9], |c| c.0);
    let halt = cost["HALT"].0;
    let l2g = cost["L2G_MEMORY"].0;

    let families_unfused = sum(SPLIT_FAMILIES, |c| c.1);
    let fixed_unfused = sum(&FIXED[..9], |c| c.1);
    let l2g_unfused = cost["L2G_MEMORY"].1;

    let recombination =
        sum(SPLIT_FAMILIES, |c| c.2) + sum(&FIXED[..9], |c| c.2) + cost["L2G_MEMORY"].2;

    let intermediate = families + fixed_no_halt + l2g;
    let final_leg = intermediate + halt;
    let final_total = final_leg + recombination + cost["HALT"].2;
    let design_intermediate = families_unfused + fixed_unfused + l2g_unfused;

    println!(
        "\ncontinuation epoch, constraint leg (minimum shape, 26 sub-proofs)\n\
         \x20 14 split families        {families:>7}  (unfused {families_unfused})\n\
         \x20  9 fixed, no HALT        {fixed_no_halt:>7}  (unfused {fixed_unfused})\n\
         \x20  1 L2G_MEMORY            {l2g:>7}  (unfused {l2g_unfused})\n\
         \x20 INTERMEDIATE leg         {intermediate:>7}  vs the design's {design_intermediate}\n\
         \x20 + recombination @ log2(N) = {LOG2_TRACE_LENGTH}  {recombination:>7}  \
         (zerofier, beta-fold, one division, claimed-parts Horner, assert)\n\
         \x20 INTERMEDIATE total       {:>7}  over 26 sub-proofs\n\
         \x20 FINAL epoch (+HALT)      {final_leg:>7} leg, {final_total} total, \
         over 25 sub-proofs",
        intermediate + recombination
    );

    // The design's §8.2.2 arithmetic, reproduced from the emitter's own unfused
    // counts. A mismatch means the epoch composition changed, which is a finding
    // about the epoch, not about this pass.
    //
    // 63_393 → 62_375 (−1018): attributed in full to KECCAK_RND, which is in
    // `FIXED`. Main replaced its θ/ρ HWSL lookups with inline μ-gated linear
    // identities in `KeccakRndConstraints`, netting −1018 constraint arithmetic
    // rows — the exact same delta the per-AIR census records for KECCAK_RND
    // (14_016 → 12_998). No other table moved; this is not a blind re-bless.
    assert_eq!(
        design_intermediate, 62_375,
        "the design's intermediate-epoch budget no longer reproduces"
    );
    assert!(
        intermediate < design_intermediate,
        "fusion must not make the leg more expensive"
    );
}

// =============================================================================
// The DEEP leg — differential against the production reconstruction
// =============================================================================

use super::deep::{DeepOpening, DeepShape, emit_deep_invariants, emit_deep_point};

/// The DEEP shape and the γ challenge, read off a real proof's replayed
/// challenges rather than modelled.
pub(super) fn deep_shape(
    sp: &RealSubProof,
    air: &dyn AIR<Field = Gl, FieldExtension = Ext3, PublicInputs = ()>,
) -> (DeepShape, FEE) {
    let layout = Verifier::ood_layout(air);
    let (main_width, aux_width) = air.trace_layout();
    let num_total_cols = main_width + aux_width;

    let shape = DeepShape {
        step_size: layout.step_size(),
        num_eval_points: sp.num_steps * layout.step_size(),
        num_total_cols,
        next_row_cols: layout.next_row_cols().to_vec(),
        num_composition_parts: sp.claimed_parts.len(),
        log2_trace_length: sp.quotient.log2_trace_length,
    };

    // γ, recovered from the coefficient run and CHECKED against every entry the
    // verifier built: coeff[c][r] is γ raised to a position-determined exponent,
    // so if the emitter's exponent formula is wrong this assertion is what says
    // so — not the differential, which would only say the answer differs.
    let coeffs = &sp.challenges.trace_term_coeffs;
    let gamma = coeffs[1][0];
    #[allow(clippy::needless_range_loop)] // `row` is a column-index, not a row-index, into `coeffs`
    for row in 0..shape.num_eval_points {
        let (cols, start, stride) = shape.block_for_test(row);
        for (k, &c) in cols.iter().enumerate() {
            assert_eq!(
                coeffs[c][row],
                gamma.pow((start + k * stride) as u64),
                "trace_term_coeffs[{c}][{row}] disagrees with the emitter's \
                 exponent formula"
            );
        }
        // Every column OUTSIDE the block must carry a zero coefficient — that is
        // the pruning, and it is what makes folding the window alone exact.
        if row >= shape.step_size {
            for c in 0..num_total_cols {
                if !cols.contains(&c) {
                    assert_eq!(
                        coeffs[c][row],
                        FEE::zero(),
                        "column {c} is pruned at row {row}"
                    );
                }
            }
        }
    }
    for (j, g) in sp.challenges.gammas.iter().enumerate() {
        assert_eq!(
            *g,
            gamma.pow((shape.num_surviving() + j) as u64),
            "composition gamma {j} must continue the same geometric run"
        );
    }

    (shape, gamma)
}

/// ★ The machine's DEEP reconstruction equals the production verifier's, on a
/// real proof's real query openings.
///
/// The oracle is `reconstruct_deep_composition_poly_evaluation_pair` itself,
/// fed through `compute_query_invariant_deep_terms` — the exact pair of
/// functions `verify_rounds_2_to_4` calls, with the exact values a real proof
/// carries. Nothing about the algebra is transcribed into the test.
#[test]
fn deep_reconstruction_matches_the_production_verifier() {
    let (air, proof) = real_fixture();
    let sp = open_sub_proof(&*air, &proof);
    let (shape, gamma) = deep_shape(&sp, &*air);

    let view = StarkProofView::Owned(&proof.proofs[0]);
    let layout = Verifier::ood_layout(&*air);
    let invariants = Verifier::<Gl, Ext3, ()>::compute_query_invariant_deep_terms(
        &sp.challenges,
        view,
        &sp.ood_full,
        layout.next_row_cols(),
        layout.step_size(),
    )
    .expect("a real proof's invariant terms");

    let domain = new_verifier_domain(&*air, view.trace_length());
    let generator = <Gl as math::field::traits::IsFFTField>::get_primitive_root_of_unity(
        sp.quotient.log2_trace_length as u64,
    )
    .expect("root of unity");

    let mut checked = 0usize;
    for (q, iota) in sp.challenges.iotas.iter().enumerate() {
        let opening = view.deep_poly_opening(q);
        let precomputed: &[FE] = opening
            .precomputed_trace_polys()
            .map(|p| p.evaluations())
            .unwrap_or(&[]);
        let main = opening.main_trace_polys().evaluations();
        let aux: &[FEE] = opening
            .aux_trace_polys()
            .map(|a| a.evaluations())
            .unwrap_or(&[]);
        let precomputed_sym: &[FE] = opening
            .precomputed_trace_polys()
            .map(|p| p.evaluations_sym())
            .unwrap_or(&[]);
        let main_sym = opening.main_trace_polys().evaluations_sym();
        let aux_sym: &[FEE] = opening
            .aux_trace_polys()
            .map(|a| a.evaluations_sym())
            .unwrap_or(&[]);

        type V = Verifier<Gl, Ext3, ()>;
        let point = V::query_challenge_to_evaluation_point(*iota, false, &domain);
        let point_sym = V::query_challenge_to_evaluation_point(*iota, true, &domain);

        // --- oracle ---
        let (want, want_sym) = V::reconstruct_deep_composition_poly_evaluation_pair(
            &point,
            &point_sym,
            &generator,
            &sp.challenges,
            &invariants,
            layout.next_row_cols(),
            layout.step_size(),
            precomputed,
            main,
            aux,
            opening.composition_poly().evaluations(),
            precomputed_sym,
            main_sym,
            aux_sym,
            opening.composition_poly().evaluations_sym(),
        )
        .expect("a real proof reconstructs");

        // --- the machine ---
        let trace: Vec<FEE> = precomputed
            .iter()
            .chain(main.iter())
            .map(|v| v.to_extension::<Ext3>())
            .chain(aux.iter().copied())
            .collect();
        let trace_sym: Vec<FEE> = precomputed_sym
            .iter()
            .chain(main_sym.iter())
            .map(|v| v.to_extension::<Ext3>())
            .chain(aux_sym.iter().copied())
            .collect();
        assert_eq!(trace.len(), shape.num_total_cols);

        let ood_words: Vec<LfmWord> = (0..shape.num_eval_points)
            .flat_map(|r| {
                let row = sp.ood_full.get_row(r);
                (0..shape.num_total_cols)
                    .map(|c| ext_word(&row[c]))
                    .collect::<Vec<_>>()
            })
            .collect();

        let mut b = LfmBuilder::new();
        let words: Vec<LfmWord> = std::iter::once(ext_word(&gamma))
            .chain(std::iter::once(ext_word(&sp.zeta)))
            .chain(ood_words.iter().copied())
            .chain(sp.claimed_parts.iter().map(ext_word))
            .chain(std::iter::once(base_word(point)))
            .chain(trace.iter().map(ext_word))
            .chain(
                opening
                    .composition_poly()
                    .evaluations()
                    .iter()
                    .map(ext_word),
            )
            .chain(std::iter::once(base_word(point_sym)))
            .chain(trace_sym.iter().map(ext_word))
            .chain(
                opening
                    .composition_poly()
                    .evaluations_sym()
                    .iter()
                    .map(ext_word),
            )
            .collect();
        let arena = b.declare_arena(words.len() as u32);
        let mut idx = 0u32;
        let mut take = |b: &mut LfmBuilder| {
            let c = b.hint_word(arena, idx).as_ext();
            idx += 1;
            c
        };
        let g_cell = take(&mut b);
        let z_cell = take(&mut b);
        let ood_steps: Vec<Vec<_>> = (0..shape.num_eval_points)
            .map(|_| (0..shape.num_total_cols).map(|_| take(&mut b)).collect())
            .collect();
        let parts: Vec<_> = (0..shape.num_composition_parts)
            .map(|_| take(&mut b))
            .collect();
        let inv = emit_deep_invariants(&mut b, &shape, g_cell, z_cell, &ood_steps, &parts);

        let read_opening = |b: &mut LfmBuilder, idx: &mut u32| {
            let p = super::builder::Felt(b.hint_word(arena, *idx).addr());
            *idx += 1;
            let mut cells = Vec::with_capacity(shape.num_total_cols);
            for _ in 0..shape.num_total_cols {
                cells.push(b.hint_word(arena, *idx).as_ext());
                *idx += 1;
            }
            let mut ps = Vec::with_capacity(shape.num_composition_parts);
            for _ in 0..shape.num_composition_parts {
                ps.push(b.hint_word(arena, *idx).as_ext());
                *idx += 1;
            }
            DeepOpening {
                point: p,
                trace: cells,
                parts: ps,
            }
        };
        let regular = read_opening(&mut b, &mut idx);
        let symmetric = read_opening(&mut b, &mut idx);
        let got = emit_deep_point(&mut b, &shape, g_cell, &inv, &regular);
        let got_sym = emit_deep_point(&mut b, &shape, g_cell, &inv, &symmetric);
        b.public(got.as_cell());
        b.public(got_sym.as_cell());

        let program = compile(b.finish());
        validate(&program).expect("the DEEP program is admissible");
        let exec = execute(&program, &[words], &TestPermutation).expect("DEEP executes");
        assert_eq!(
            word_as_ext(&exec.public_words[0].1).expect("ext"),
            want,
            "query {q}: DEEP at the regular point"
        );
        assert_eq!(
            word_as_ext(&exec.public_words[1].1).expect("ext"),
            want_sym,
            "query {q}: DEEP at the symmetric point"
        );
        assert_ne!(want, FEE::zero(), "query {q} must not be vacuously zero");

        checked += 1;
        if checked == 3 {
            break;
        }
    }
    assert!(checked > 0, "the fixture must carry at least one query");
    println!("DEEP differential: {checked} queries, both points each");
}

/// ★ The coefficient-exponent formula holds where no production AIR reaches:
/// `step_size = 2` with two next rows.
///
/// The DEEP differential above runs on L2G_MEMORY, and every production AIR has
/// `step_size = 1` and a single next row — which collapses both strides to one.
/// A plain Horner in γ would therefore pass every test we have. This one builds
/// the verifier's own coefficient table at a wider step through
/// `build_pruned_trace_term_coeffs` and checks the emitter against it, then
/// shows the stride-1 reading DISAGREES. Without that second half the test would
/// pass against the wrong emitter.
#[test]
fn the_coefficient_exponent_formula_holds_at_a_wider_step() {
    use stark::ood::build_pruned_trace_term_coeffs;

    const COLS: usize = 5;
    const STEP: usize = 2;
    const EVAL_POINTS: usize = 4; // two offsets x step 2
    let next_row_cols = vec![1usize, 3];

    let shape = DeepShape {
        step_size: STEP,
        num_eval_points: EVAL_POINTS,
        num_total_cols: COLS,
        next_row_cols: next_row_cols.clone(),
        num_composition_parts: 2,
        log2_trace_length: 4,
    };
    let surviving = shape.num_surviving();
    assert_eq!(
        surviving,
        COLS * STEP + next_row_cols.len() * (EVAL_POINTS - STEP)
    );

    let gamma = FEE::new([FE::from(7u64), FE::from(11u64), FE::from(13u64)]);
    let powers: Vec<FEE> = (0..surviving).map(|p| gamma.pow(p as u64)).collect();
    let coeffs = build_pruned_trace_term_coeffs(&powers, COLS, EVAL_POINTS, STEP, &next_row_cols);

    let mut stride_ever_exceeds_one = false;
    let mut plain_horner_would_differ = false;

    #[allow(clippy::needless_range_loop)] // `row` is a column-index, not a row-index, into `coeffs`
    for row in 0..EVAL_POINTS {
        let (cols, start, stride) = shape.block_for_test(row);
        if stride > 1 {
            stride_ever_exceeds_one = true;
        }
        for (k, &c) in cols.iter().enumerate() {
            assert_eq!(
                coeffs[c][row],
                gamma.pow((start + k * stride) as u64),
                "coeffs[{c}][{row}] disagrees with the emitter's (start {start}, \
                 stride {stride}) formula"
            );
            // The falsification: what a stride-1 fold would have used.
            if coeffs[c][row] != gamma.pow((start + k) as u64) {
                plain_horner_would_differ = true;
            }
        }
        // Pruned columns carry a zero coefficient on next rows.
        if row >= STEP {
            for c in 0..COLS {
                if !cols.contains(&c) {
                    assert_eq!(coeffs[c][row], FEE::zero(), "column {c} at row {row}");
                }
            }
        }
    }

    assert!(
        stride_ever_exceeds_one,
        "the fixture must actually produce a stride above one"
    );
    assert!(
        plain_horner_would_differ,
        "a plain Horner in gamma must give a DIFFERENT coefficient here, or this \
         test does not show the stride is load-bearing"
    );
}

/// ★ What a DEEP query costs, per sub-proof and per epoch.
///
/// ### What this instrument cannot see
///
/// The query COUNT. It is a proof-options property (219 at blowup 2, 73 at
/// blowup 8), not an AIR property, so the per-epoch line below is parameterised
/// on it rather than measured. It also excludes the Merkle authentication of the
/// openings this leg consumes, which is the R1f leg's cost, and the FRI folding
/// that consumes this leg's output.
#[test]
fn deep_leg_cost() {
    /// Queries at blowup 2 — stated, not measured here.
    const QUERIES: usize = 219;

    let opts = options();
    let airs = production_airs(&opts);

    println!("\nDEEP cost per query point, by AIR");
    println!(
        "{:<14} {:>6} {:>7} {:>6} {:>9} {:>10}",
        "table", "cols", "window", "parts", "rows/pt", "rows/query"
    );

    let mut total_per_query = 0usize;
    for (label, air) in &airs {
        let artifact = ConstraintArtifact::capture(&**air);
        let layout = Verifier::<Gl, Ext3, ()>::ood_layout(&**air);
        let (main_width, aux_width) = air.trace_layout();
        let shape = DeepShape {
            step_size: layout.step_size(),
            num_eval_points: artifact.shape.transition_offsets.len() * layout.step_size(),
            num_total_cols: main_width + aux_width,
            next_row_cols: layout.next_row_cols().to_vec(),
            num_composition_parts: artifact.shape.composition_degree_multiplier as usize,
            log2_trace_length: 20,
        };

        // Measure by emitting, twice, and differencing out the plumbing.
        let plumb = |b: &mut LfmBuilder| {
            let n = 2
                + shape.num_eval_points * shape.num_total_cols
                + 2 * shape.num_composition_parts
                + shape.num_total_cols
                + 1;
            let arena = b.declare_arena(n as u32);
            let mut i = 0u32;
            let mut take = |b: &mut LfmBuilder| {
                let c = b.hint_word(arena, i).as_ext();
                i += 1;
                c
            };
            let g = take(b);
            let z = take(b);
            let steps: Vec<Vec<_>> = (0..shape.num_eval_points)
                .map(|_| (0..shape.num_total_cols).map(|_| take(b)).collect())
                .collect();
            let parts: Vec<_> = (0..shape.num_composition_parts).map(|_| take(b)).collect();
            let trace: Vec<_> = (0..shape.num_total_cols).map(|_| take(b)).collect();
            let qparts: Vec<_> = (0..shape.num_composition_parts).map(|_| take(b)).collect();
            let point = super::builder::Felt(take(b).addr());
            (g, z, steps, parts, trace, qparts, point)
        };

        let mut bare = LfmBuilder::new();
        let _ = plumb(&mut bare);
        let baseline = bare.finish().instrs.len();

        let mut inv_only = LfmBuilder::new();
        let (g, z, steps, parts, _, _, _) = plumb(&mut inv_only);
        let _ = emit_deep_invariants(&mut inv_only, &shape, g, z, &steps, &parts);
        let invariant_rows = inv_only.finish().instrs.len() - baseline;

        let mut full = LfmBuilder::new();
        let (g, z, steps, parts, trace, qparts, point) = plumb(&mut full);
        let inv = emit_deep_invariants(&mut full, &shape, g, z, &steps, &parts);
        emit_deep_point(
            &mut full,
            &shape,
            g,
            &inv,
            &DeepOpening {
                point,
                trace,
                parts: qparts,
            },
        );
        let point_rows = full.finish().instrs.len() - baseline - invariant_rows;

        let per_query = 2 * point_rows;
        total_per_query += per_query;
        println!(
            "{:<14} {:>6} {:>7} {:>6} {:>9} {:>10}",
            label,
            shape.num_total_cols,
            shape.next_row_cols.len(),
            shape.num_composition_parts,
            point_rows,
            per_query
        );
    }

    println!(
        "\nSum over all 28 AIRs, one query each (both points): {total_per_query} rows.\n\
         At {QUERIES} queries that is {} rows if every AIR appeared once — an\n\
         ORDER-OF-MAGNITUDE figure, not an epoch: an epoch's sub-proof set is not\n\
         the 28-AIR set, and this excludes the Merkle authentication of these same\n\
         openings and the FRI folding that consumes the result.",
        total_per_query * QUERIES
    );
}
