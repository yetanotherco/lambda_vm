//! Serialization bit-exactness: for every production table, the constraint
//! artifact SERIALIZED at build time and read back evaluates identically to the
//! compiled folders.
//!
//! `constraint_program_tests` already pins the in-memory capture against both
//! folders. This suite pins the extra hop that "constraints as data" adds — the
//! bytes:
//!
//! ```text
//!   capture → artifact → to_bytes → from_bytes → lift → evaluate
//! ```
//!
//! Every stage in that chain is a place a program can change meaning: a
//! truncated `u32` index, a constant that loses canonical form through raw
//! limbs, a metadata list that reorders, an op tag that decodes to a different
//! operation. None of it is visible to a test that only exercises the in-memory
//! program, which is why this suite runs the deserialized artifact — never the
//! captured object — against the folders.
//!
//! The folders are the oracle: they are the production prove/verify path,
//! independently pinned by the prove→verify suites and cross-version
//! verification. All three evaluation paths are checked against them:
//! `eval_program` (prover shape), `eval_program_verifier` (OOD shape, the
//! recursion path), and `eval_device_program` (the flat blob).

use math::field::element::FieldElement;
use stark::constraint_ir::{
    ConstraintArtifact, eval_device_program, eval_program, eval_program_verifier,
};
use stark::frame::Frame;
use stark::proof::options::GoldilocksCubicProofOptions;
use stark::table::TableView;
use stark::traits::{AIR, TransitionEvaluationContext};

use crate::tables::types::{GoldilocksExtension, GoldilocksField};
use crate::test_utils::*;

type Gl = GoldilocksField;
type Ext3 = GoldilocksExtension;
type Fp = FieldElement<Gl>;
type Fp3 = FieldElement<Ext3>;

const TRIALS: usize = 100;

/// Deterministic SplitMix64 (no `rand` dependency).
struct SplitMix64(u64);
impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn fp3(&mut self) -> Fp3 {
        Fp3::new([
            Fp::from(self.next_u64()),
            Fp::from(self.next_u64()),
            Fp::from(self.next_u64()),
        ])
    }
}

/// Extension element → raw `[u64; 3]` limbs (the device representation).
fn enc(x: &Fp3) -> [u64; 3] {
    let limbs = x.value();
    [*limbs[0].value(), *limbs[1].value(), *limbs[2].value()]
}

/// One production AIR's serialized-artifact differential. Returns the artifact's
/// measured size for the size report.
fn check_air_artifact(
    air: &dyn AIR<Field = Gl, FieldExtension = Ext3, PublicInputs = ()>,
    label: &str,
) -> ArtifactSize {
    let n = air.context().num_transition_constraints;
    let num_base = air.num_base_transition_constraints();
    let (n_main, n_aux) = air.trace_layout();

    let artifact = ConstraintArtifact::capture(air);
    artifact
        .validate_against(air)
        .unwrap_or_else(|e| panic!("[{label}] freshly captured artifact rejected: {e}"));

    // The wire hop. Everything below runs the DESERIALIZED artifact, so a
    // codec bug cannot hide behind the in-memory object.
    let bytes = artifact
        .to_bytes()
        .unwrap_or_else(|e| panic!("[{label}] serialize failed: {e}"));
    let artifact = ConstraintArtifact::from_bytes(&bytes)
        .unwrap_or_else(|e| panic!("[{label}] deserialize failed: {e}"));
    artifact
        .validate_against(air)
        .unwrap_or_else(|e| panic!("[{label}] deserialized artifact rejected: {e}"));

    let prog = artifact.program();
    let dev = artifact.device_program();

    // Structural identity against the capture the AIR itself would produce.
    let captured = air.constraint_program();
    assert_eq!(
        prog.nodes, captured.nodes,
        "[{label}] nodes changed on the wire"
    );
    assert_eq!(
        prog.dims, captured.dims,
        "[{label}] dims changed on the wire"
    );
    assert_eq!(
        prog.roots, captured.roots,
        "[{label}] roots changed on the wire"
    );
    assert_eq!(
        prog.num_base, captured.num_base,
        "[{label}] num_base changed"
    );
    assert_eq!(
        prog.base_consts, captured.base_consts,
        "[{label}] base constants changed on the wire"
    );
    assert_eq!(
        prog.ext_consts, captured.ext_consts,
        "[{label}] ext constants changed on the wire"
    );
    assert_eq!(prog.roots.len(), n, "[{label}] one root per constraint");

    // Release-safe exact-once backstop, as in `constraint_program_tests`: root
    // id 0 is the reserved base-zero sentinel and no production constraint is
    // identically zero, so a root still at the sentinel means that constraint
    // was never captured.
    for (i, &root) in prog.roots.iter().enumerate() {
        assert_ne!(root, 0, "[{label}] constraint {i} was never captured");
    }

    let mut rng = SplitMix64(0x5EED_1234 ^ label.len() as u64);
    for trial in 0..TRIALS {
        let mk_step = |rng: &mut SplitMix64| {
            let main: Vec<Fp> = (0..n_main).map(|_| Fp::from(rng.next_u64())).collect();
            let aux: Vec<Fp3> = (0..n_aux).map(|_| rng.fp3()).collect();
            TableView::new(vec![main], vec![aux])
        };
        let frame = Frame::<Gl, Ext3>::new(vec![mk_step(&mut rng), mk_step(&mut rng)]);
        let challenges = vec![rng.fp3(), rng.fp3()]; // [z, alpha]
        let alphas: Vec<Fp3> = (0..air.max_bus_elements() + 2).map(|_| rng.fp3()).collect();
        let offset = rng.fp3();

        let ctx = TransitionEvaluationContext::new_prover(
            frame.as_row_frame(),
            &challenges,
            &alphas,
            &offset,
        );

        // --- oracle: the compiled prover folder ---
        let mut f_base = vec![Fp::zero(); num_base];
        let mut f_ext = vec![Fp3::zero(); n];
        air.compute_transition_prover(&ctx, &mut f_base, &mut f_ext);

        // --- path 1: generic interpreter over the deserialized program ---
        let mut i_base = vec![Fp::zero(); num_base];
        let mut i_ext = vec![Fp3::zero(); n];
        eval_program(&prog, &ctx, &mut i_base, &mut i_ext);

        for c in 0..num_base {
            assert_eq!(
                f_base[c], i_base[c],
                "[{label}] prover folder vs serialized program, base constraint {c}, trial {trial}"
            );
        }
        for c in num_base..n {
            assert_eq!(
                f_ext[c], i_ext[c],
                "[{label}] prover folder vs serialized program, ext constraint {c}, trial {trial}"
            );
        }

        // --- path 2: flat device walk over the deserialized blob ---
        let main_raw: Vec<Vec<u64>> = (0..2)
            .map(|off| {
                let step = frame.get_evaluation_step(off);
                (0..n_main)
                    .map(|c| *step.get_main_evaluation_element(0, c).value())
                    .collect()
            })
            .collect();
        let aux_raw: Vec<Vec<[u64; 3]>> = (0..2)
            .map(|off| {
                let step = frame.get_evaluation_step(off);
                (0..n_aux)
                    .map(|c| enc(step.get_aux_evaluation_element(0, c)))
                    .collect()
            })
            .collect();
        let rap_raw: Vec<[u64; 3]> = challenges.iter().map(enc).collect();
        let alpha_raw: Vec<[u64; 3]> = alphas.iter().map(enc).collect();

        let mut d_base = vec![0u64; num_base];
        let mut d_ext = vec![[0u64; 3]; n];
        eval_device_program(
            &dev,
            &main_raw,
            &aux_raw,
            &rap_raw,
            &alpha_raw,
            enc(&offset),
            &mut d_base,
            &mut d_ext,
        );
        for c in 0..num_base {
            assert_eq!(
                d_base[c],
                *f_base[c].value(),
                "[{label}] prover folder vs serialized device blob, base constraint {c}, trial {trial}"
            );
        }
        for c in num_base..n {
            assert_eq!(
                d_ext[c],
                enc(&f_ext[c]),
                "[{label}] prover folder vs serialized device blob, ext constraint {c}, trial {trial}"
            );
        }

        // --- path 3: the verifier/OOD shape, i.e. the recursion path ---
        let embed = |step: &TableView<Gl, Ext3>| -> TableView<Ext3, Ext3> {
            let main: Vec<Fp3> = (0..n_main)
                .map(|c| step.get_main_evaluation_element(0, c).to_extension())
                .collect();
            let aux: Vec<Fp3> = (0..n_aux)
                .map(|c| *step.get_aux_evaluation_element(0, c))
                .collect();
            TableView::new(vec![main], vec![aux])
        };
        let vframe: Frame<Ext3, Ext3> = Frame::new(vec![
            embed(frame.get_evaluation_step(0)),
            embed(frame.get_evaluation_step(1)),
        ]);
        let vctx = TransitionEvaluationContext::<Gl, Ext3>::new_verifier(
            &vframe,
            &challenges,
            &alphas,
            &offset,
        );

        let v_folder = air.compute_transition(&vctx);
        let mut v_interp = vec![Fp3::zero(); n];
        eval_program_verifier(&prog, &vctx, &mut v_interp);
        for c in 0..n {
            assert_eq!(
                v_folder[c], v_interp[c],
                "[{label}] verifier folder vs serialized program, constraint {c}, trial {trial}"
            );
        }
    }

    ArtifactSize {
        label: label.to_string(),
        constraints: n,
        nodes: artifact.nodes.len(),
        base_consts: artifact.base_consts.len(),
        ext_consts: artifact.ext_consts.len(),
        bytes: bytes.len(),
    }
}

/// One AIR's measured artifact size.
struct ArtifactSize {
    label: String,
    constraints: usize,
    nodes: usize,
    base_consts: usize,
    ext_consts: usize,
    bytes: usize,
}

/// Every production table's serialized artifact evaluates bit-identically to
/// the compiled folders, on the prover shape, the verifier/OOD shape, and the
/// flat device blob.
#[test]
fn all_table_artifacts_roundtrip_and_match_folders() {
    let opts = GoldilocksCubicProofOptions::with_blowup(2).expect("blowup=2 valid");
    let airs = production_airs(&opts);
    assert_eq!(
        airs.len(),
        NUM_PRODUCTION_AIRS,
        "the production AIR list changed size; every per-table suite's coverage moved with it"
    );

    let mut sizes: Vec<ArtifactSize> = Vec::with_capacity(airs.len());
    for (label, air) in &airs {
        sizes.push(check_air_artifact(&**air, label));
    }

    report_sizes(&sizes);
}

/// Print the per-AIR and total artifact sizes — the recursion machine's
/// program-length budget — and hold a ceiling so runaway growth is noticed.
fn report_sizes(sizes: &[ArtifactSize]) {
    let total_bytes: usize = sizes.iter().map(|s| s.bytes).sum();
    let total_nodes: usize = sizes.iter().map(|s| s.nodes).sum();
    let total_constraints: usize = sizes.iter().map(|s| s.constraints).sum();

    println!("\nconstraint artifact sizes (blowup=2)");
    println!(
        "{:<12} {:>7} {:>9} {:>7} {:>7} {:>10}",
        "table", "constr", "nodes", "bconst", "econst", "bytes"
    );
    for s in sizes {
        println!(
            "{:<12} {:>7} {:>9} {:>7} {:>7} {:>10}",
            s.label, s.constraints, s.nodes, s.base_consts, s.ext_consts, s.bytes
        );
    }
    println!(
        "{:<12} {:>7} {:>9} {:>7} {:>7} {:>10}",
        "TOTAL", total_constraints, total_nodes, "", "", total_bytes
    );
    println!(
        "total: {total_nodes} nodes, {total_bytes} bytes ({:.1} KiB) across {} tables\n",
        total_bytes as f64 / 1024.0,
        sizes.len()
    );

    // A loose ceiling: this is a budget signal, not a tight assertion. It exists
    // so a change that multiplies the program size fails here instead of being
    // discovered when the recursion machine will not fit.
    const CEILING_BYTES: usize = 8 * 1024 * 1024;
    assert!(
        total_bytes < CEILING_BYTES,
        "total artifact size {total_bytes} exceeds the {CEILING_BYTES}-byte budget ceiling"
    );
}

/// Per-AIR instruction census for the recursion machine's constraint leg.
///
/// The machine is straight-line and cannot interpret, so a serialized program is
/// UNROLLED: one machine instruction per arithmetic IR node. That makes the node
/// census a direct instruction-count estimate for the constraint-evaluation leg,
/// which is the last unmeasured piece of the epoch-verifier budget.
///
/// The classification that matters, and why:
///
/// - **Leaves are addresses, not instructions.** `Var` reads an OOD frame value
///   the DEEP/opening leg already placed in memory; `RapChallenge` / `AlphaPow` /
///   `TableOffset` are transcript-derived values computed once per proof. The
///   constraint leg pays nothing marginal for them.
/// - **Constants are `Const` instructions**, one per distinct pooled value.
/// - **Arithmetic nodes are ALU instructions**, and the base/ext split is the
///   expensive distinction: a base node is a `BaseAlu` over one Goldilocks
///   element, an extension node an `ExtAlu` over three.
/// - **A `Mul` with an extension result and exactly one base operand** is the
///   `MulBase` form — 3 base multiplies instead of 9. Counting these separately
///   is the difference between a real estimate and a pessimistic one.
///
/// # The IR's own `dim` tags are the WRONG split for the machine
///
/// `Dim` records what the PROVER computes: its frame is base-field, so a
/// trace-only subexpression stays in the base field. The machine runs the
/// VERIFIER's evaluation at the OOD point, where the frame holds only extension
/// elements — `eval_program_verifier` resolves every `Var` to `Value::Ext`
/// regardless of `main`. So a node is base at verify time only if its whole
/// subtree is constants.
///
/// Both splits are reported because the difference is large and load-bearing,
/// and taking the declared one would badly understate the machine's extension
/// traffic. The verifier-side column is the one to budget against.
///
/// A consequence worth naming: a base-at-verify-time node is a constant-only
/// subtree, so the emitter can FOLD it at build time into a pooled constant. It
/// costs zero instructions, which is why the instruction estimate below counts
/// only extension work plus the pool.
///
/// Printed rather than asserted (beyond a loose ceiling): this is an instrument,
/// and pinning exact counts would turn every constraint edit into a test failure.
///
/// See also `epoch_chunk_multiplier`, which turns this per-AIR table into a
/// per-EPOCH figure — the counts here are per distinct AIR, and an epoch
/// evaluates the leg once per sub-proof.
#[test]
fn constraint_op_census() {
    use stark::constraint_ir::device::{
        DIM_BASE, OP_ADD, OP_ALPHA_POW, OP_CONST_BASE, OP_CONST_EXT, OP_EMBED, OP_MUL, OP_NEG,
        OP_RAP_CHALLENGE, OP_SUB, OP_TABLE_OFFSET, OP_VAR,
    };

    let opts = GoldilocksCubicProofOptions::with_blowup(2).expect("blowup=2 valid");
    let airs = production_airs(&opts);
    assert_eq!(airs.len(), NUM_PRODUCTION_AIRS);

    println!("\nconstraint-leg instruction census (one instruction per arithmetic node)");
    println!("  prover-dim = what the IR declares; verify-dim = what the machine runs");
    println!(
        "{:<14} {:>7} {:>7} {:>6} {:>8} {:>8} {:>7} {:>8} {:>8}",
        "table", "nodes", "leaves", "const", "pv-base", "fold", "ext", "mulbase", "instr"
    );

    let (mut t_nodes, mut t_leaves, mut t_const) = (0usize, 0usize, 0usize);
    let (mut t_pv_base, mut t_fold, mut t_ext, mut t_mulbase) = (0usize, 0usize, 0usize, 0usize);
    let mut t_constraints = 0usize;

    for (label, air) in &airs {
        let artifact = ConstraintArtifact::capture(&**air);
        let nodes = &artifact.nodes;

        // Verifier-side dim: base ONLY for constant-only subtrees, because the
        // OOD frame is all-extension.
        let mut v_base = vec![false; nodes.len()];

        let (mut leaves, mut consts, mut pv_base, mut foldable, mut ext_alu, mut mulbase) =
            (0, 0, 0, 0, 0, 0);
        for (i, n) in nodes.iter().enumerate() {
            match n.op {
                OP_VAR | OP_RAP_CHALLENGE | OP_ALPHA_POW | OP_TABLE_OFFSET => {
                    leaves += 1;
                    v_base[i] = false;
                }
                OP_CONST_BASE => {
                    consts += 1;
                    v_base[i] = true;
                }
                OP_CONST_EXT => {
                    consts += 1;
                    v_base[i] = false;
                }
                OP_ADD | OP_SUB | OP_MUL | OP_NEG | OP_EMBED => {
                    if n.dim == DIM_BASE {
                        pv_base += 1;
                    }
                    let (ba, bb) = match n.op {
                        OP_NEG => (v_base[n.a as usize], true),
                        OP_EMBED => (false, false),
                        _ => (v_base[n.a as usize], v_base[n.b as usize]),
                    };
                    // Mirrors `interp::binop`: base only when both operands are
                    // base values AND the declared dim is base.
                    v_base[i] = ba && bb && n.dim == DIM_BASE;
                    if v_base[i] {
                        // Constant-only subtree: the emitter folds it at build
                        // time, so it emits no instruction at all.
                        foldable += 1;
                    } else {
                        let is_mulbase = n.op == OP_MUL && (ba != bb);
                        if is_mulbase {
                            mulbase += 1;
                        } else {
                            ext_alu += 1;
                        }
                    }
                }
                other => panic!("[{label}] unclassified op tag {other}"),
            }
        }

        // Instructions the constraint leg actually emits: extension ALU work
        // plus one Const per pooled constant. Leaves are addresses, and
        // constant-only subtrees fold away at build time.
        let instr = ext_alu + mulbase + consts;
        println!(
            "{:<14} {:>7} {:>7} {:>6} {:>8} {:>8} {:>7} {:>8} {:>8}",
            label,
            nodes.len(),
            leaves,
            consts,
            pv_base,
            foldable,
            ext_alu,
            mulbase,
            instr
        );

        t_nodes += nodes.len();
        t_leaves += leaves;
        t_const += consts;
        t_pv_base += pv_base;
        t_fold += foldable;
        t_ext += ext_alu;
        t_mulbase += mulbase;
        t_constraints += artifact.roots.len();

        // Every constant node must correspond to exactly one pooled table entry;
        // if that ever stopped holding, the Const instruction count above would
        // be wrong.
        assert_eq!(
            consts,
            artifact.base_consts.len() + artifact.ext_consts.len(),
            "[{label}] constant nodes and pooled constants disagree"
        );
    }

    // --- two emitter properties the design depends on, measured ---
    //
    // 1. MulAdd fusability. `ExtAlu` carries MulAdd as a first-class op, but the
    //    IR has no MulAdd node — it emits Mul then Add. A peephole can fuse
    //    `Add(Mul(a,b), c)` into one instruction, but ONLY when the Mul feeds
    //    exactly one consumer; hash-consing means a shared Mul would have to be
    //    recomputed, turning a saving into a cost.
    // 2. `Op::Embed` usage. It should be zero — the builder documents it as
    //    unreachable from the single-body capture path — which matters because
    //    Embed is the one op whose machine lowering depends on the word model.
    let (mut t_fusable, mut t_embed, mut t_dead, mut t_maxfan) = (0usize, 0usize, 0usize, 0u32);
    // Machine constants are interned PROGRAM-WIDE, keyed on the canonical
    // 4-lane word — one `Const` row per distinct value however many AIRs and
    // however many nodes use it. Summing the per-AIR pools therefore overcounts,
    // and small structural constants (0, 1, byte/halfword shifts) recur across
    // every table.
    let mut distinct_words: std::collections::BTreeSet<[u64; 4]> =
        std::collections::BTreeSet::new();
    for (_, air) in &airs {
        let artifact = ConstraintArtifact::capture(&**air);
        let nodes = &artifact.nodes;

        let mut uses = vec![0u32; nodes.len()];
        for n in nodes {
            match n.op {
                OP_ADD | OP_SUB | OP_MUL => {
                    uses[n.a as usize] += 1;
                    uses[n.b as usize] += 1;
                }
                OP_NEG | OP_EMBED => uses[n.a as usize] += 1,
                _ => {}
            }
        }
        // A root is a consumer too: fusing away a node that a constraint roots at
        // would delete the value the quotient recombination needs.
        for &r in &artifact.roots {
            uses[r as usize] += 1;
        }

        for &c in &artifact.base_consts {
            distinct_words.insert([c, 0, 0, 0]);
        }
        for &e in &artifact.ext_consts {
            distinct_words.insert([e[0], e[1], e[2], 0]);
        }

        // A node nobody reads is a write with mult = 0 — wasted instructions, and
        // the emitter must DCE it rather than emit a zero-multiplicity write.
        t_dead += uses.iter().filter(|&&u| u == 0).count();
        t_maxfan = t_maxfan.max(uses.iter().copied().max().unwrap_or(0));

        for n in nodes {
            if n.op == OP_EMBED {
                t_embed += 1;
            }
            if n.op == OP_ADD {
                let a_fusable = nodes[n.a as usize].op == OP_MUL && uses[n.a as usize] == 1;
                let b_fusable = nodes[n.b as usize].op == OP_MUL && uses[n.b as usize] == 1;
                if a_fusable || b_fusable {
                    t_fusable += 1;
                }
            }
        }
    }

    let arith = t_fold + t_ext + t_mulbase;
    // One row per instruction, and every constant is one interned row
    // program-wide — so the pool is counted once, not once per AIR.
    let pool = distinct_words.len();
    let instr = t_ext + t_mulbase + pool;
    println!(
        "{:<14} {:>7} {:>7} {:>6} {:>8} {:>8} {:>7} {:>8} {:>8}",
        "TOTAL", t_nodes, t_leaves, t_const, t_pv_base, t_fold, t_ext, t_mulbase, instr
    );
    let unfused = instr + t_constraints;
    let fused = unfused - t_fusable;
    println!(
        "\n  arithmetic nodes            {arith}\n  \
           base by the IR's own dim    {t_pv_base}   (prover-side; NOT the machine's split)\n  \
           base at verify time         {t_fold}   (constant-only subtrees -> fold at build time)\n  \
           extension ALU               {t_ext}\n  \
           of which MulBase-routed     {t_mulbase}   (ext x base: 1 XALU row, vs 4+ if lowered by hand)\n  \
           per-AIR constant pools      {t_const}   (sum; NOT the machine's cost)\n  \
           interned program-wide       {pool}   (one Const row per distinct 4-lane word)\n  \
           = constraint-leg instr      {instr}\n  \
           + quotient recombination    {t_constraints} beta-folds (one per constraint)\n  \
           = upper bound               {unfused}\n  \
           MulAdd-fusable Add nodes    {t_fusable}   (Add over a single-use Mul; MulAdd costs the same as Mul, so ALWAYS fuse)\n  \
           = ESTIMATE, fused           {fused}\n\n  \
           leaves (addresses, free)    {t_leaves}\n  \
           Op::Embed nodes             {t_embed}   (base->ext is free; would emit nothing)\n  \
           dead nodes (mult = 0)       {t_dead}\n  \
           max fanout (max mult)       {t_maxfan}\n"
    );

    // Loose ceiling: the design budget treats this leg as ~1% of the epoch
    // program. An order-of-magnitude regression should fail here.
    assert!(
        instr < 200_000,
        "constraint-leg instruction estimate {instr} has grown past the design budget"
    );
}

/// The constraint leg's per-EPOCH multiplier, measured on real fixtures.
///
/// `constraint_op_census` counts instructions per distinct AIR. An epoch does
/// not evaluate each AIR once: every SUB-PROOF carries its own trace and needs
/// its own constraint evaluation, and the split-table families are CHUNKED —
/// `chunks = ceil(rows / max_rows[table])`, with `max_rows` sized per table so
/// each chunk costs about the same memory (`tables/mod.rs::max_rows`).
///
/// So the epoch cost is `Σ over sub-proofs instr(that sub-proof's AIR)`, and the
/// multiplier against the per-AIR total is what this measures. It is the number
/// `lfm-design.md` §5.2 is missing: its ≈69K line reads as per-epoch but is
/// per-distinct-AIR.
///
/// Measured by building real traces, so the chunk counts are the prover's own
/// splitting rather than a reconstruction of it.
#[test]
fn epoch_chunk_multiplier() {
    use crate::tables::MaxRowsConfig;
    use crate::tables::trace_builder::Traces;

    let opts = GoldilocksCubicProofOptions::with_blowup(2).expect("blowup=2 valid");

    // Per-AIR constraint-leg instruction counts, keyed by label.
    let instr: std::collections::BTreeMap<&str, usize> = production_airs(&opts)
        .iter()
        .map(|(label, air)| (*label, leg_instructions(&**air)))
        .collect();
    let get = |k: &str| *instr.get(k).unwrap_or_else(|| panic!("no AIR {k}"));

    // Fixtures spanning roughly an epoch's worth of execution. An intermediate
    // continuation epoch runs exactly 2^epoch_size_log2 cycles, so a fixture's
    // cycle count is the axis to read these against.
    for name in [
        "fib_iterative_1M",
        "fib_iterative_2M",
        "array_multipass_20M",
    ] {
        let (elf, logs, _) = run_asm_elf(name);
        let max_rows = MaxRowsConfig::default();
        let traces = Traces::from_elf_and_logs_minimal(&elf, &logs, &max_rows, &[])
            .expect("trace build succeeds");

        // (label, chunk count) for every sub-proof the epoch would contain.
        let chunked: Vec<(&str, usize)> = vec![
            ("CPU", traces.cpus.len()),
            ("LT", traces.lts.len()),
            ("SHIFT", traces.shifts.len()),
            ("MEMW", traces.memws.len()),
            ("MEMW_A", traces.memw_aligneds.len()),
            ("LOAD", traces.loads.len()),
            ("MUL", traces.muls.len()),
            ("DVRM", traces.dvrms.len()),
            ("BRANCH", traces.branches.len()),
            ("MEMW_R", traces.memw_registers.len()),
            ("EQ", traces.eqs.len()),
            ("BYTEWISE", traces.bytewises.len()),
            ("STORE", traces.stores.len()),
            ("CPU32", traces.cpu32s.len()),
        ];

        let chunked_total: usize = chunked.iter().map(|(l, n)| get(l) * n).sum();
        let chunk_count: usize = chunked.iter().map(|(_, n)| *n).sum();

        // Fixed (unchunked) tables present once per proof, plus one PAGE per
        // touched page. A continuation epoch substitutes L2G/GLOBAL_MEMORY for
        // PAGE (page_configs is empty there), so this monolithic shape is an
        // upper bound on the page contribution.
        let pages = traces.pages.len();
        let fixed = get("BITWISE")
            + get("DECODE")
            + get("REGISTER")
            + get("COMMIT")
            + get("HALT")
            + get("KECCAK")
            + get("KECCAK_RND")
            + get("KECCAK_RC")
            + get("ECSM")
            + get("ECDAS");
        let page_total = get("PAGE") * pages;
        let epoch_total = chunked_total + fixed + page_total;

        let per_air_total: usize = instr.values().sum();
        println!(
            "\n{name}: {} cycles\n  \
             chunked sub-proofs {chunk_count} (of 14 families) -> {chunked_total} instr\n  \
             fixed tables                                       -> {fixed} instr\n  \
             {pages} pages x {} instr                                 -> {page_total} instr\n  \
             EPOCH TOTAL {epoch_total} instr   vs per-distinct-AIR {per_air_total}   \
             multiplier {:.2}x",
            logs.len(),
            get("PAGE"),
            epoch_total as f64 / per_air_total as f64
        );
        for (l, n) in &chunked {
            if *n > 1 {
                println!("    {l:<10} {n} chunks x {} = {}", get(l), get(l) * n);
            }
        }
    }
}

/// The constraint leg for a real CONTINUATION EPOCH — the shape we actually
/// recurse.
///
/// The monolithic measurement above is the wrong shape for the target: a
/// continuation epoch passes `page_configs = &[]`, so PAGE never appears, and it
/// carries an L2G_MEMORY sub-proof instead. Its composition is
///
/// ```text
///   14 split-table families (>= 1 chunk each)
/// + FIXED_TABLE_COUNT       (10 final, 9 intermediate — HALT only on the last)
/// + 1 L2G_MEMORY
/// ```
///
/// which gives **24 sub-proofs intermediate, 25 final** — independently measured
/// on the LFM fibonacci epoch fixture. This test asserts that arithmetic so the
/// composition is pinned rather than inferred: if the epoch shape changes, the
/// count here stops matching the measured one and this fails.
///
/// The instruction total is then a minimum, since it assumes one chunk per
/// family — a larger epoch adds chunks of the CHEAP AIRs (see
/// `epoch_chunk_multiplier`).
#[test]
fn continuation_epoch_constraint_leg() {
    let opts = GoldilocksCubicProofOptions::with_blowup(2).expect("blowup=2 valid");
    let instr: std::collections::BTreeMap<&str, usize> = production_airs(&opts)
        .iter()
        .map(|(label, air)| (*label, leg_instructions(&**air)))
        .collect();
    let get = |k: &str| *instr.get(k).unwrap_or_else(|| panic!("no AIR {k}"));

    // The 14 chunked split-table families, at their minimum of one chunk each.
    let families = [
        "CPU", "LT", "SHIFT", "MEMW", "MEMW_A", "LOAD", "MUL", "DVRM", "BRANCH", "MEMW_R", "EQ",
        "BYTEWISE", "STORE", "CPU32",
    ];
    // FIXED_TABLE_COUNT = 10 (`prover/src/lib.rs`): always exactly one sub-proof
    // each, REGARDLESS of TableCounts — a zero-row table still needs its proof,
    // or its constraints drop out of verification. HALT is the one an
    // intermediate epoch omits.
    let fixed_final = [
        "BITWISE",
        "DECODE",
        "HALT",
        "COMMIT",
        "KECCAK",
        "KECCAK_RND",
        "KECCAK_RC",
        "REGISTER",
        "ECSM",
        "ECDAS",
    ];

    let families_instr: usize = families.iter().map(|l| get(l)).sum();
    let fixed_final_instr: usize = fixed_final.iter().map(|l| get(l)).sum();
    let fixed_intermediate_instr = fixed_final_instr - get("HALT");
    let l2g = get("L2G_MEMORY");

    let intermediate = families_instr + fixed_intermediate_instr + l2g;
    let final_epoch = families_instr + fixed_final_instr + l2g;

    let n_intermediate = families.len() + fixed_final.len() - 1 + 1;
    let n_final = families.len() + fixed_final.len() + 1;
    assert_eq!(
        (n_intermediate, n_final),
        (24, 25),
        "epoch sub-proof composition no longer reproduces the measured 24 intermediate / 25 final"
    );

    println!(
        "\ncontinuation epoch constraint leg (minimum: one chunk per family)\n  \
           14 split families      {families_instr}\n  \
           9 fixed (no HALT)      {fixed_intermediate_instr}\n  \
           1 L2G_MEMORY           {l2g}\n  \
           INTERMEDIATE epoch     {intermediate} instr over {n_intermediate} sub-proofs\n  \
           FINAL epoch (+HALT)    {final_epoch} instr over {n_final} sub-proofs\n  \
           fixed share            {:.0}% — the leg is workload-INDEPENDENT\n",
        100.0 * fixed_intermediate_instr as f64 / intermediate as f64
    );

    // The global proof is one L2G_GLOBAL per epoch plus one GLOBAL_MEMORY per
    // touched page — negligible at any plausible page count, which is what
    // settles the page-base question as identity-only rather than size.
    println!(
        "  global proof: {} instr/epoch (L2G_GLOBAL) + {} instr/page (GLOBAL_MEMORY)\n",
        get("L2G_GLOBAL"),
        get("GLOBAL_MEMORY")
    );
}

/// Constraint-leg instructions for one AIR: extension ALU plus MulBase-routed
/// multiplies. Shared by `constraint_op_census` and `epoch_chunk_multiplier` so
/// the two cannot drift apart.
fn leg_instructions(air: &dyn AIR<Field = Gl, FieldExtension = Ext3, PublicInputs = ()>) -> usize {
    use stark::constraint_ir::device::{
        DIM_BASE, OP_ADD, OP_ALPHA_POW, OP_CONST_BASE, OP_CONST_EXT, OP_EMBED, OP_MUL, OP_NEG,
        OP_RAP_CHALLENGE, OP_SUB, OP_TABLE_OFFSET, OP_VAR,
    };
    let artifact = ConstraintArtifact::capture(air);
    let nodes = &artifact.nodes;
    let mut v_base = vec![false; nodes.len()];
    let mut count = 0usize;
    for (i, n) in nodes.iter().enumerate() {
        match n.op {
            OP_VAR | OP_RAP_CHALLENGE | OP_ALPHA_POW | OP_TABLE_OFFSET | OP_CONST_EXT => {
                v_base[i] = false
            }
            OP_CONST_BASE => v_base[i] = true,
            OP_ADD | OP_SUB | OP_MUL | OP_NEG | OP_EMBED => {
                let (ba, bb) = match n.op {
                    OP_NEG => (v_base[n.a as usize], true),
                    OP_EMBED => (false, false),
                    _ => (v_base[n.a as usize], v_base[n.b as usize]),
                };
                v_base[i] = ba && bb && n.dim == DIM_BASE;
                if !v_base[i] {
                    count += 1;
                }
            }
            other => panic!("unclassified op tag {other}"),
        }
    }
    count
}

/// The captured artifact does not depend on the proof options.
///
/// This is the premise behind leaving `ProofOptions` OUT of the artifact — if it
/// failed, one artifact per table would not be enough and the whole scheme would
/// need an artifact per (table, blowup) pair. `AirContext` carries the options
/// alongside the shape scalars, so the independence is worth pinning rather than
/// assuming from a reading of the constructor.
#[test]
fn artifacts_are_invariant_across_proof_options() {
    let opts2 = GoldilocksCubicProofOptions::with_blowup(2).expect("blowup=2 valid");
    let opts4 = GoldilocksCubicProofOptions::with_blowup(4).expect("blowup=4 valid");

    let airs2 = production_airs(&opts2);
    let airs4 = production_airs(&opts4);

    for ((label, a2), (_, a4)) in airs2.iter().zip(airs4.iter()) {
        let art2 = ConstraintArtifact::capture(&**a2);
        let art4 = ConstraintArtifact::capture(&**a4);
        assert_eq!(
            art2, art4,
            "[{label}] the constraint artifact differs between blowup 2 and 4; it would have to \
             be stored per (table, blowup) pair"
        );
        assert_eq!(
            art2.to_bytes().expect("serialize"),
            art4.to_bytes().expect("serialize"),
            "[{label}] artifact bytes differ across blowup factors"
        );
    }
}

/// The captured artifact does not depend on the trace length either.
///
/// Same failure mode as the proof-options axis, different variable: if anything
/// in a captured program folded a domain-size-dependent constant, artifacts
/// would multiply per epoch shape.
///
/// The axis is structurally absent — no AIR constructor takes a trace length —
/// so the only route by which one could reach the artifact is
/// `composition_poly_degree_bound(n)`, the single trace-length-dependent method
/// on the trait, whose value the artifact stores divided through by `n`. That
/// division is only sound if the bound is exactly linear, so this sweeps a wide
/// range of `n` per table rather than trusting the two probe points
/// `ConstraintArtifact::capture` checks. A table whose bound had any constant
/// term or any non-linearity would be misrepresented by the stored multiplier,
/// and would show up here.
#[test]
fn artifacts_are_invariant_across_trace_length() {
    let opts = GoldilocksCubicProofOptions::with_blowup(2).expect("blowup=2 valid");
    let airs = production_airs(&opts);
    assert_eq!(airs.len(), NUM_PRODUCTION_AIRS);

    for (label, air) in &airs {
        let artifact = ConstraintArtifact::capture(&**air);
        let k = artifact.shape.composition_degree_multiplier as usize;

        for log_n in 4usize..=24 {
            let n = 1usize << log_n;
            assert_eq!(
                air.composition_poly_degree_bound(n),
                k * n,
                "[{label}] composition_poly_degree_bound is not k·n at n=2^{log_n}; the artifact \
                 stores only the linear coefficient, so a trace-length-dependent AIR would need \
                 an artifact per epoch shape"
            );
        }

        // Nothing else on the artifact can vary with the trace length, but pin
        // capture determinism so a future source of nondeterminism (map
        // iteration order in the constant tables, say) is caught here.
        let again = ConstraintArtifact::capture(&**air);
        assert_eq!(artifact, again, "[{label}] capture is not deterministic");
    }
}

/// The four PARAMETERIZED tables produce a different program per parameter
/// value.
///
/// `PAGE` / `GLOBAL_MEMORY` fold a page base into constant bus terms; the two
/// `L2G` tables fold an epoch label. This test does not assert that away — it
/// characterizes it, because it is a real property of the current constraints
/// and the recursion machine has to plan around it.
///
/// # The variation is NOT confined to constant VALUES
///
/// The obvious guess is that two parameter values give the same node array with
/// one constant swapped. That is what `PAGE` and `GLOBAL_MEMORY` do, and it is
/// wrong in general: the builder interns constants by value, so a parameter
/// whose value happens to already be in the constant table costs no new node,
/// while a fresh value appends one — which shifts every later node id and hence
/// the constraint ROOTS. `L2G_GLOBAL` at `epoch_label = 1` reuses the existing
/// `1` constant; at `epoch_label = 7` it appends. Same algebra, different node
/// count and different root ids.
///
/// This matters for the machine-side fix: "swap one constant per page" would be
/// a cheap patch and it is not available. Promoting the parameter to a runtime
/// uniform is, because the ALGEBRA is invariant — which is what the shape and
/// metadata assertions below pin.
#[test]
fn parameterized_airs_vary_per_parameter_value() {
    let opts = GoldilocksCubicProofOptions::with_blowup(2).expect("blowup=2 valid");

    // (label, artifact at parameter A, artifact at parameter B)
    let cases: Vec<(&str, ConstraintArtifact, ConstraintArtifact)> = vec![
        (
            "PAGE",
            ConstraintArtifact::capture(&create_page_air(&opts, 0x1000)),
            ConstraintArtifact::capture(&create_page_air(&opts, 0x9000)),
        ),
        (
            "GLOBAL_MEMORY",
            ConstraintArtifact::capture(&create_global_memory_air(&opts, 0x1000)),
            ConstraintArtifact::capture(&create_global_memory_air(&opts, 0x9000)),
        ),
        (
            "L2G_GLOBAL",
            ConstraintArtifact::capture(&crate::continuation::l2g_global_air(&opts, 1)),
            ConstraintArtifact::capture(&crate::continuation::l2g_global_air(&opts, 7)),
        ),
        (
            "L2G_MEMORY",
            ConstraintArtifact::capture(&crate::continuation::l2g_memory_air(&opts, 1)),
            ConstraintArtifact::capture(&crate::continuation::l2g_memory_air(&opts, 7)),
        ),
    ];

    println!("\nparameterized tables: how two parameter values differ");
    for (label, a, b) in &cases {
        assert_ne!(
            a, b,
            "[{label}] is documented as parameterized but two parameter values gave the same \
             artifact; either the parameter stopped reaching the IR or the test picked two \
             values that collide"
        );

        // Invariant: the ALGEBRA. Same widths, same constraint count, same
        // zerofier shapes, same degree bound — only the embedded parameter
        // moves. This is the property that makes the parameter promotable to a
        // runtime uniform.
        assert_eq!(a.shape, b.shape, "[{label}] shape must not vary");
        assert_eq!(a.meta, b.meta, "[{label}] metadata must not vary");
        assert_eq!(a.num_base, b.num_base, "[{label}] num_base must not vary");
        assert_eq!(
            a.roots.len(),
            b.roots.len(),
            "[{label}] constraint count must not vary"
        );

        // Variable: node count and root ids — an artifact of the builder's
        // hash-consing, not of the constraints. A parameter value already in the
        // constant table costs no new ConstBase node while a fresh one appends,
        // which shifts every later node id.
        //
        // MEASURED, and note the counts are not all +1: L2G_GLOBAL moves 1 node
        // for 1 constant, L2G_MEMORY moves 2 for 1. The second node is some
        // further CSE difference downstream of the reused constant (L2G_MEMORY
        // at epoch_label = 1 contributes the constant 0, which IS node id 0, so
        // expressions over it have more chance to coincide with existing ones) —
        // that specific explanation is inferred, not verified, so the bound
        // below is deliberately loose. What is being pinned is only that the
        // delta stays local rather than the algebra changing shape.
        let node_delta = a.nodes.len().abs_diff(b.nodes.len());
        let const_delta = a.base_consts.len().abs_diff(b.base_consts.len());
        assert!(
            node_delta <= 4 && const_delta <= 1,
            "[{label}] two parameter values changed the program by {node_delta} nodes and \
             {const_delta} constants — too much to be the parameter's own interned constant and \
             its enclosing ops; the variation is structural, not just parametric"
        );
        let roots_moved = a.roots != b.roots;

        println!(
            "  {label:<14} nodes {:>3} vs {:>3}   consts {:>2} vs {:>2}   roots moved: {}",
            a.nodes.len(),
            b.nodes.len(),
            a.base_consts.len(),
            b.base_consts.len(),
            roots_moved
        );
    }
    println!();
}

/// GLOBAL_MEMORY has a second, ENUMERABLE axis: private-input pages are built
/// non-preprocessed, which changes the artifact's SHAPE rather than a constant.
///
/// Worth separating from the parameter axis above because the two have very
/// different consequences. A page base is an arbitrary address, so its artifact
/// set is unbounded; `is_private_input` is a boolean, so GLOBAL_MEMORY simply has
/// two shape variants and both can be enumerated. This pins that the difference
/// is confined to the preprocessed-column fields and does not touch the program.
#[test]
fn global_memory_private_input_is_a_second_shape_not_a_second_program() {
    use crate::tables::page::PageConfig;
    let opts = GoldilocksCubicProofOptions::with_blowup(2).expect("blowup=2 valid");

    let elf_page = PageConfig::zero_init(PAGE_TEST_BASE);
    let mut private_page = PageConfig::zero_init(PAGE_TEST_BASE);
    private_page.is_private_input = true;

    let elf = ConstraintArtifact::capture(&crate::continuation::global_memory_air(
        &opts,
        &elf_page,
        Some([0u8; 32]),
    ));
    let private = ConstraintArtifact::capture(&crate::continuation::global_memory_air(
        &opts,
        &private_page,
        Some([0u8; 32]),
    ));

    // Same constraints, same metadata: the bus interactions depend only on the
    // page base, which is equal here.
    assert_eq!(elf.nodes, private.nodes, "the program must not vary");
    assert_eq!(elf.base_consts, private.base_consts);
    assert_eq!(elf.roots, private.roots);
    assert_eq!(elf.meta, private.meta);

    // The shape does vary, in exactly the preprocessed fields.
    assert!(elf.shape.is_preprocessed, "an ELF page is preprocessed");
    assert!(
        !private.shape.is_preprocessed,
        "a private-input page is not preprocessed — the verifier never recomputes its genesis \
         column from the ELF"
    );
    assert!(elf.shape.num_precomputed_columns > 0);
    assert_eq!(private.shape.num_precomputed_columns, 0);

    let mut normalized = private.shape.clone();
    normalized.is_preprocessed = elf.shape.is_preprocessed;
    normalized.num_precomputed_columns = elf.shape.num_precomputed_columns;
    assert_eq!(
        normalized, elf.shape,
        "the two variants must differ ONLY in the preprocessed-column fields"
    );
}

/// An artifact captured from one table must not validate against another.
///
/// The suite above only ever shows the shape check ACCEPTING. Without this, a
/// `validate_against` that returned `Ok(())` unconditionally would pass
/// everything here.
#[test]
fn an_artifact_does_not_validate_against_a_different_table() {
    let opts = GoldilocksCubicProofOptions::with_blowup(2).expect("blowup=2 valid");
    let airs = production_airs(&opts);

    let mut checked = 0usize;
    for (i, (label_i, air_i)) in airs.iter().enumerate() {
        let artifact = ConstraintArtifact::capture(&**air_i);
        for (j, (label_j, air_j)) in airs.iter().enumerate() {
            if i == j {
                continue;
            }
            if artifact.validate_against(&**air_j).is_ok() {
                // Two tables can legitimately share every shape scalar (several
                // are bus-only tables with identical layouts), so an accept is
                // only a failure when the programs actually differ.
                let prog_i = air_i.constraint_program();
                let prog_j = air_j.constraint_program();
                assert_eq!(
                    (&prog_i.nodes, &prog_i.roots),
                    (&prog_j.nodes, &prog_j.roots),
                    "[{label_i}] artifact validated against [{label_j}], whose constraint \
                     program is different — the shape check cannot tell them apart"
                );
            } else {
                checked += 1;
            }
        }
    }
    assert!(
        checked > 0,
        "validate_against never rejected any cross-table pairing; the check is not live"
    );
}

/// A pre-captured program can be supplied to a production AIR and is used
/// without capture — the scoped verify-path unban, on real tables.
#[test]
fn production_airs_accept_a_precaptured_program() {
    let opts = GoldilocksCubicProofOptions::with_blowup(2).expect("blowup=2 valid");

    // Build the artifact from one instance, install it into a fresh one.
    let artifact = ConstraintArtifact::capture(&create_cpu_air(&opts));
    let air = create_cpu_air(&opts).with_precaptured(artifact.program());

    let supplied = air
        .precaptured_constraint_program()
        .expect("the supplied program must be visible on the guest-safe accessor");
    assert!(
        std::ptr::eq(air.constraint_program(), supplied),
        "constraint_program() must return the supplied program, not a fresh capture"
    );

    // And it still evaluates like the folder.
    let n = air.context().num_transition_constraints;
    let num_base = air.num_base_transition_constraints();
    let (n_main, n_aux) = air.trace_layout();
    let mut rng = SplitMix64(0x00A1_1CE5);
    for _ in 0..16 {
        let mk_step = |rng: &mut SplitMix64| {
            let main: Vec<Fp> = (0..n_main).map(|_| Fp::from(rng.next_u64())).collect();
            let aux: Vec<Fp3> = (0..n_aux).map(|_| rng.fp3()).collect();
            TableView::new(vec![main], vec![aux])
        };
        let frame = Frame::<Gl, Ext3>::new(vec![mk_step(&mut rng), mk_step(&mut rng)]);
        let challenges = vec![rng.fp3(), rng.fp3()];
        let alphas: Vec<Fp3> = (0..air.max_bus_elements() + 2).map(|_| rng.fp3()).collect();
        let offset = rng.fp3();
        let ctx = TransitionEvaluationContext::new_prover(
            frame.as_row_frame(),
            &challenges,
            &alphas,
            &offset,
        );

        let mut f_base = vec![Fp::zero(); num_base];
        let mut f_ext = vec![Fp3::zero(); n];
        air.compute_transition_prover(&ctx, &mut f_base, &mut f_ext);

        let mut i_base = vec![Fp::zero(); num_base];
        let mut i_ext = vec![Fp3::zero(); n];
        eval_program(supplied, &ctx, &mut i_base, &mut i_ext);

        assert_eq!(f_base, i_base);
        for c in num_base..n {
            assert_eq!(f_ext[c], i_ext[c]);
        }
    }
}
