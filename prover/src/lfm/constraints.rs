//! Lowering a captured [`ConstraintArtifact`] into LFM instructions — the
//! constraint-evaluation leg of the epoch verifier.
//!
//! The pass runs on the HOST at registry-build time, so constant folding,
//! dead-code elimination, fanout analysis and peephole fusion are free: what
//! reaches the machine is fixed program text whose digest the registry pins.
//!
//! # The IR's `dim` tags describe the PROVER; the machine runs the VERIFIER
//!
//! [`Dim`] records what the prover computes over a base-field trace frame. The
//! machine evaluates at the out-of-domain point, where the frame holds only
//! extension elements — `eval_program_verifier` resolves every [`Op::Var`] to an
//! extension value regardless of `main`. Propagating that through the
//! interpreter's rule (base only when both operands are base *values* and the
//! declared dim is base), a node is base at verify time **only if its entire
//! subtree is constants**. Sizing this leg off the declared dims understates
//! extension traffic by roughly 14×.
//!
//! Those constant-only subtrees are exactly the nodes this pass folds, so they
//! cost no rows at all rather than costing base-ALU rows.
//!
//! # What costs a row and what does not
//!
//! | IR op | lowering | rows |
//! |---|---|---|
//! | [`Op::Var`] / [`Op::RapChallenge`] / [`Op::AlphaPow`] / [`Op::TableOffset`] | an address supplied by [`OodOperands`] | 0 |
//! | [`Op::ConstBase`] / [`Op::ConstExt`] | an interned `Instr::Const` word | 1 per distinct word, program-wide |
//! | [`Op::Embed`] | **nothing** — a base word already IS its extension embedding | 0 |
//! | [`Op::Neg`] | `ExtAlu{Sub}` against the pooled zero — the ISA has no unary negate | 1 |
//! | [`Op::Add`] | `ExtAlu{Add}`, or absorbed into a producer `Mul` as `MulAdd` | 1 or 0 |
//! | [`Op::Sub`] | `ExtAlu{Sub}` | 1 |
//! | [`Op::Mul`] | `ExtAlu{Mul}`, or `MulBase` when one operand is a base word | 1 |
//!
//! `MulAdd` costs the same single row as `Mul`, so fusing is not an optimization
//! — an unfused emitter simply pays two rows where one would do. It is sound
//! only under a single-consumer guard: the IR is HASH-CONSED, so a shared `Mul`
//! feeds several parents and fusing it into each would recompute it per parent.
//! The hazard is documented on [`ConstraintArtifact`] itself.
//!
//! # `MulBase` is cost-neutral here, not a saving
//!
//! `LFM_XALU` charges one row for `Mul` and one for `MulBase`, on the same chip
//! at the same width, and its `B` operand is received as an extension token
//! either way (`chips::xalu`). A base-valued word `(c, 0, 0, 0)` is therefore a
//! legal `Mul` operand and yields the same product. This pass routes the case
//! through `MulBase` because that states the intent and pins lanes 1–2 to zero
//! by constraint, but nothing breaks — and no row is added — if it does not.
//! See `others/lfm-constraint-lowering-design.md` §3, which overstates this as a
//! 4× obligation by comparing against a hand-lowering nobody would write.

use std::collections::HashSet;

use stark::constraint_ir::{ConstraintArtifact, ConstraintProgram, Dim, Op};

use crate::tables::types::{FE, FEE, GoldilocksExtension, GoldilocksField};

use super::builder::{Ext, Felt, LfmBuilder};

type Prog = ConstraintProgram<GoldilocksField, GoldilocksExtension>;

/// Where a lowered constraint program reads its per-proof operands.
///
/// The distinction between these four sources is a soundness boundary, not a
/// packaging convenience (`SOUNDNESS.md` §5): OOD frame values are arena-fed and
/// must be authenticated transitively by the DEEP/opening leg, whereas
/// challenges and alpha powers are computed in-machine by the transcript replay
/// and must NEVER come from an arena. This struct takes them as already-resolved
/// cells precisely so the lowering pass cannot invent either one.
pub struct OodOperands {
    /// `steps[offset][col]` — the full-width `[main | aux]` OOD frame at each
    /// transition offset, aux columns starting at `main_width`. This is the same
    /// concatenated indexing the verifier's reconstructed grid uses.
    ///
    /// Next-row entries outside the AIR's declared `next_row_columns` are
    /// reconstructed as ZERO by the verifier, so the caller supplies the pooled
    /// zero cell there — see [`hint_ood_frame`].
    pub steps: Vec<Vec<Ext>>,
    /// Where the aux columns start inside each step.
    pub main_width: usize,
    /// The LogUp RAP challenges, transcript-derived.
    pub rap_challenges: Vec<Ext>,
    /// Precomputed LogUp alpha powers.
    pub alpha_powers: Vec<Ext>,
    /// The LogUp table offset `L/N`.
    ///
    /// ⚠ In any program that ALSO runs the LogUp closure, this must be the cell
    /// [`emit_table_offset`] returns, not a separate hint. See that function for
    /// why — the two legs consuming `L` independently makes the bus-balance
    /// check vacuous. The synthetic IR-lowering differential is exempt: it has
    /// no trace length and no closure, so there is no second consumer to agree
    /// with.
    pub table_offset: Ext,
}

/// `L/N` — the LogUp per-row offset — DERIVED from the table's total bus
/// contribution `L` rather than hinted alongside it.
///
/// # Why this is a soundness requirement and not a convenience
///
/// `L` has two consumers that never meet. The circular accumulator constraint
/// (`lookup.rs`'s `emit_logup_accumulated`) enforces
/// `acc_next − acc_curr − Σterms + L/N = 0`, which together with the `acc[0] = 0`
/// boundary pins `L` to the aux trace: the accumulator wraps to zero after `N`
/// rows only if `L` really is that table's total. The bus-balance closure
/// separately checks `Σ_tables L = expected`. Production computes the offset
/// from the one `L` the proof carries (`verifier.rs`'s `logup_table_offset`), so
/// the two agree by construction.
///
/// A machine that hinted `L/N` for the constraint leg and `L` for the closure
/// would hand the prover two independent arena words. Supply a truthful `L₁/N`
/// so every accumulator wraps, and an arbitrary `L₂` so the sum hits the target:
/// both legs pass in isolation and the bus balance is a statement about numbers
/// attached to nothing. Deriving one from the other is what denies that, and it
/// costs a single `MulBase` against a program constant, since `N` is shape.
///
/// This is the same failure the DEEP/authentication join closes one leg over —
/// two consumers of one value, agreeing only because the host filling the arena
/// agreed with itself.
pub fn emit_table_offset(b: &mut LfmBuilder, contribution: Ext, log2_trace_length: u32) -> Ext {
    let n = FE::from(1u64 << log2_trace_length);
    let n_inv = n
        .inv()
        .expect("a power-of-two trace length is nonzero, so invertible");
    let c = b.felt_const(n_inv);
    b.emul_base(contribution, c)
}

/// `[α⁰, α¹, …, α^{n−1}]` — the LogUp alpha powers, DERIVED from the one α the
/// transcript produced rather than hinted one word each.
///
/// # Why these cannot be arena words
///
/// Same class as [`emit_table_offset`], and worse in degree. `Op::AlphaPow{idx}`
/// resolves to `alpha_powers[idx]`, and those powers are what build every LogUp
/// FINGERPRINT: a row's tuple `(v₀, v₁, …)` enters the constraint as
/// `z − Σ vⱼ·αʲ`. A prover who supplies the powers independently of α chooses
/// the fingerprints, and with the fingerprints goes the entire lookup argument —
/// any tuple can be made to match any other. The powers are not merely a second
/// consumer of α, they are a CLAIM about α that nothing else checks.
///
/// Deriving them costs one `ExtAlu{Mul}` per power beyond the first two (α⁰ is
/// the interned one and α¹ is α itself), and `n = AIR::max_bus_elements()` is
/// shape, so the chain length is program text.
///
/// The alpha this consumes must itself come from the transcript replay, never
/// from an arena — that part is an assembly obligation this function cannot
/// enforce, since it takes α as a cell and cannot see where the cell came from.
pub fn emit_alpha_powers(b: &mut LfmBuilder, alpha: Ext, n: usize) -> Vec<Ext> {
    let mut powers = Vec::with_capacity(n);
    for i in 0..n {
        powers.push(match i {
            0 => b.ext_const(&FEE::one()),
            1 => alpha,
            _ => {
                let prev = powers[i - 1];
                b.emul(prev, alpha)
            }
        });
    }
    powers
}

/// What one AIR's lowering cost, measured by the pass that emitted it.
///
/// Every field is a count of what the pass DID, not a prediction: [`analyze`]
/// and [`emit_constraint_evals`] share one analysis, so a report can never drift
/// from the program it describes.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LoweringReport {
    /// IR nodes in the artifact.
    pub nodes: usize,
    /// Leaf nodes (frame reads and uniforms) — addresses, never rows.
    pub leaves: usize,
    /// Arithmetic nodes unreachable from any root that would otherwise have
    /// cost a ROW — the rows dead-code elimination actually saves.
    pub dead: usize,
    /// Constant nodes no root reaches — overwhelmingly the INTERIOR of a folded
    /// subtree, whose value is absorbed into the fold rather than read. They
    /// cost no row either way, and the design's census reports them under
    /// `fold`, so a reader comparing the two totals must not add them again.
    pub unreached_const: usize,
    /// Constant-only subtrees folded to a BASE value at build time — the
    /// "verify-time base" set, and the column the design doc calls `fold`.
    pub fold_base: usize,
    /// Constant-only subtrees whose declared dim widens them to the extension.
    /// Folding these too is strictly cheaper and the design's census does not
    /// count them; reported separately so the two remain comparable.
    pub fold_ext: usize,
    /// [`Op::Embed`] nodes lowered to a pure address alias.
    pub aliased: usize,
    /// `Mul`/`Add` pairs collapsed into one `MulAdd`.
    pub fused: usize,
    /// Fusable `(Add, Mul)` OPERAND pairs, before the one-multiply-per-row
    /// limit is applied. An `Add` whose two operands are both single-consumer
    /// `Mul`s offers two candidates and can absorb only one, so this exceeds
    /// [`Self::fused`] by exactly the number of such sums — which is the whole
    /// gap between a candidate count and a saving.
    pub fuse_candidates: usize,
    /// Arithmetic nodes no other node references and no root names — "fanout 0"
    /// measured LOCALLY, which is the measure the design's §4.3 reports.
    pub orphans: usize,
    /// The same count over EVERY node kind, leaves and constants included.
    pub orphans_all_kinds: usize,
    /// Extension ALU rows: `Add`, `Sub`, `Mul`, `Neg` and the fused `MulAdd`s.
    pub ext_alu: usize,
    /// `MulBase`-routed multiplies (one base-valued operand).
    pub mul_base: usize,
    /// Distinct constant WORDS this AIR asks the builder for. The builder
    /// interns program-wide, so a program covering several AIRs emits fewer
    /// `Const` rows than the sum of these.
    pub constants: usize,
}

impl LoweringReport {
    /// ALU rows the pass emits for this AIR: `ext_alu + mul_base`. Constants are
    /// excluded because interning makes them a program-wide, not per-AIR, cost.
    pub fn alu_rows(&self) -> usize {
        self.ext_alu + self.mul_base
    }

    /// ALU rows this AIR would cost with fusion switched off — the design doc's
    /// per-AIR `instr` column, and an upper bound on [`Self::alu_rows`].
    pub fn unfused_alu_rows(&self) -> usize {
        self.alu_rows() + self.fused
    }
}

/// A node's compile-time value, when it has one.
#[derive(Clone, Copy, Debug)]
enum Konst {
    Base(FE),
    Ext(FEE),
}

impl Konst {
    fn to_ext(self) -> FEE {
        match self {
            Konst::Base(x) => x.to_extension::<GoldilocksExtension>(),
            Konst::Ext(x) => x,
        }
    }

    fn is_base(self) -> bool {
        matches!(self, Konst::Base(_))
    }
}

/// Host-side analysis of one artifact: which nodes fold, which are dead, which
/// `Mul`s are absorbed by a consumer `Add`.
///
/// Kept separate from emission so instruction counts can be measured without
/// building a program, and so the numbers reported are by construction the
/// numbers emitted.
pub struct Analysis {
    prog: Prog,
    konst: Vec<Option<Konst>>,
    live: Vec<bool>,
    /// On an `Add` that absorbs a producer `Mul`: that `Mul`'s node id.
    fuse_src: Vec<Option<u32>>,
    /// Set on a `Mul` absorbed by its single consumer.
    fused_away: Vec<bool>,
    report: LoweringReport,
}

impl Analysis {
    /// The measured cost of this lowering.
    pub fn report(&self) -> &LoweringReport {
        &self.report
    }

    /// The lifted program the analysis ran over.
    pub fn program(&self) -> &Prog {
        &self.prog
    }
}

/// Analyse an artifact without emitting anything.
///
/// # Panics
///
/// On a malformed artifact (out-of-range operand or constant index): the
/// artifact's own `validate_self` runs first and reports precisely what is
/// wrong. This is a build-time entry point, so failing loudly here is correct —
/// a silently mis-lowered constraint would be a wrong program with a valid
/// digest.
pub fn analyze(artifact: &ConstraintArtifact) -> Analysis {
    artifact
        .validate_self()
        .expect("constraint artifact failed its own consistency check");
    let prog = artifact.program();
    let n = prog.nodes.len();

    let mut report = LoweringReport {
        nodes: n,
        ..Default::default()
    };

    // ---- forward pass: compile-time values ----
    //
    // Mirrors `interp::run` exactly, including the dim-driven widening in
    // `binop`, so a folded value is bit-identical to what the interpreter would
    // have computed for that node.
    let mut konst: Vec<Option<Konst>> = vec![None; n];
    for i in 0..n {
        let k = match prog.nodes[i] {
            Op::ConstBase(idx) => Some(Konst::Base(prog.base_consts[idx as usize])),
            Op::ConstExt(idx) => Some(Konst::Ext(prog.ext_consts[idx as usize])),
            Op::Var { row, .. } => {
                assert_eq!(row, 0, "node {i}: the capture path only reads row 0");
                report.leaves += 1;
                None
            }
            Op::RapChallenge { .. } | Op::AlphaPow { .. } | Op::TableOffset => {
                report.leaves += 1;
                None
            }
            Op::Add(a, b) => fold_binop(&konst, a, b, prog.dims[i], |x, y| x + y, |x, y| x + y),
            Op::Sub(a, b) => fold_binop(&konst, a, b, prog.dims[i], |x, y| x - y, |x, y| x - y),
            Op::Mul(a, b) => fold_binop(&konst, a, b, prog.dims[i], |x, y| x * y, |x, y| x * y),
            Op::Neg(a) => match (konst[a as usize], prog.dims[i]) {
                (Some(Konst::Base(x)), Dim::Base) => Some(Konst::Base(-x)),
                (Some(v), _) => Some(Konst::Ext(-v.to_ext())),
                (None, _) => None,
            },
            Op::Embed(a) => konst[a as usize].map(|v| Konst::Ext(v.to_ext())),
        };
        konst[i] = k;
    }

    // ---- backward pass: liveness from the roots ----
    //
    // A folded node reads nothing at run time, so it does not keep its operands
    // alive; that is what lets a whole constant subtree disappear rather than
    // just its top node.
    let mut live = vec![false; n];
    for &r in &prog.roots {
        live[r as usize] = true;
    }
    for i in (0..n).rev() {
        if !live[i] || konst[i].is_some() {
            continue;
        }
        for a in operands(&prog.nodes[i]) {
            live[a as usize] = true;
        }
    }

    // ---- fanout over EMITTED consumers, then fusion selection ----
    let mut fanout = vec![0u32; n];
    for i in 0..n {
        if !live[i] || konst[i].is_some() {
            continue;
        }
        for a in operands(&prog.nodes[i]) {
            if konst[a as usize].is_none() {
                fanout[a as usize] += 1;
            }
        }
    }
    // A root is a consumer: the quotient recombination reads it.
    for &r in &prog.roots {
        if konst[r as usize].is_none() {
            fanout[r as usize] += 1;
        }
    }

    // Local fanout, the design's own measure: every reference from any node,
    // folded or not, plus roots. Distinct from `fanout` above, which counts only
    // the consumers that survive to read a cell.
    {
        let mut refs = vec![0u32; n];
        for i in 0..n {
            for a in operands(&prog.nodes[i]) {
                refs[a as usize] += 1;
            }
        }
        for &r in &prog.roots {
            refs[r as usize] += 1;
        }
        report.orphans = (0..n)
            .filter(|&i| refs[i] == 0 && is_arith(&prog.nodes[i]))
            .count();
        report.orphans_all_kinds = (0..n).filter(|&i| refs[i] == 0).count();
    }

    let mut fuse_src: Vec<Option<u32>> = vec![None; n];
    let mut fused_away = vec![false; n];
    for i in 0..n {
        if !live[i] || konst[i].is_some() {
            continue;
        }
        let Op::Add(a, b) = prog.nodes[i] else {
            continue;
        };
        // `a` first, then `b` — `Add` is commutative and `MulAdd` computes
        // `a·b + c`, so either side may supply the product. Only one can: the
        // instruction carries a single multiply.
        for cand in [a, b] {
            if fusable(&prog, &konst, &fanout, cand) {
                report.fuse_candidates += 1;
                if fuse_src[i].is_none() {
                    fuse_src[i] = Some(cand);
                    fused_away[cand as usize] = true;
                }
            }
        }
    }

    // ---- cost accounting ----
    let mut constants: HashSet<[u64; 4]> = HashSet::new();
    let want_const = |k: Konst, set: &mut HashSet<[u64; 4]>| {
        let w = match k {
            Konst::Base(v) => super::word::base_word(v),
            Konst::Ext(v) => super::word::ext_word(&v),
        };
        set.insert(core::array::from_fn(|l| {
            <GoldilocksField as math::field::traits::IsPrimeField>::canonical(w[l].value())
        }));
    };

    for i in 0..n {
        // Constants are classified BEFORE liveness, because a constant-only
        // subtree costs no row whether or not a root reaches it — and because
        // that is the split the design's census reports, so the two stay
        // comparable. `dead` is then the DCE that actually saves rows.
        if let Some(k) = konst[i] {
            if is_arith(&prog.nodes[i]) {
                if k.is_base() {
                    report.fold_base += 1;
                } else {
                    report.fold_ext += 1;
                }
                if !live[i] {
                    report.unreached_const += 1;
                }
            }
            continue;
        }
        if !live[i] {
            if is_arith(&prog.nodes[i]) {
                report.dead += 1;
            }
            continue;
        }
        if fused_away[i] {
            report.fused += 1;
            continue;
        }
        match prog.nodes[i] {
            Op::Embed(_) => report.aliased += 1,
            Op::Add(_, _) | Op::Sub(_, _) => report.ext_alu += 1,
            Op::Neg(_) => report.ext_alu += 1,
            Op::Mul(a, b) => {
                if is_base_konst(&konst, a) != is_base_konst(&konst, b) {
                    report.mul_base += 1;
                } else {
                    report.ext_alu += 1;
                }
            }
            _ => {}
        }
    }

    // Constants actually referenced by an emitted node or a root, plus the
    // pooled zero every `Neg` subtracts from.
    let mut needs_zero = false;
    for i in 0..n {
        if !live[i] || konst[i].is_some() || fused_away[i] {
            continue;
        }
        if matches!(prog.nodes[i], Op::Neg(_)) {
            needs_zero = true;
        }
        for a in operands(&prog.nodes[i]) {
            if let Some(k) = konst[a as usize] {
                want_const(k, &mut constants);
            }
        }
    }
    for &r in &prog.roots {
        if let Some(k) = konst[r as usize] {
            want_const(k, &mut constants);
        }
    }
    if needs_zero {
        want_const(Konst::Base(FE::zero()), &mut constants);
    }
    report.constants = constants.len();

    Analysis {
        prog,
        konst,
        live,
        fuse_src,
        fused_away,
        report,
    }
}

/// Lower an artifact's constraint program, returning one cell per constraint
/// root in `constraint_idx` order.
///
/// The returned values are the AIR's transition-constraint evaluations at the
/// OOD point — the input to the zerofier/quotient recombination, not the
/// quotient itself.
pub fn emit_constraint_evals(
    b: &mut LfmBuilder,
    artifact: &ConstraintArtifact,
    ood: &OodOperands,
) -> (Vec<Ext>, LoweringReport) {
    let analysis = analyze(artifact);
    let evals = emit_analyzed(b, &analysis, ood);
    (evals, analysis.report)
}

/// [`emit_constraint_evals`] over an analysis the caller already has.
pub fn emit_analyzed(b: &mut LfmBuilder, an: &Analysis, ood: &OodOperands) -> Vec<Ext> {
    let prog = &an.prog;
    let n = prog.nodes.len();
    let mut addr: Vec<Option<Ext>> = vec![None; n];

    for i in 0..n {
        if !an.live[i] || an.konst[i].is_some() || an.fused_away[i] {
            continue;
        }
        let out = match prog.nodes[i] {
            Op::Var {
                main, offset, col, ..
            } => {
                let step = ood
                    .steps
                    .get(offset as usize)
                    .unwrap_or_else(|| panic!("node {i}: frame has no offset {offset}"));
                let idx = if main {
                    col as usize
                } else {
                    ood.main_width + col as usize
                };
                *step
                    .get(idx)
                    .unwrap_or_else(|| panic!("node {i}: frame step {offset} has no column {idx}"))
            }
            Op::RapChallenge { idx } => ood.rap_challenges[idx as usize],
            Op::AlphaPow { idx } => ood.alpha_powers[idx as usize],
            Op::TableOffset => ood.table_offset,
            // A base word IS its own extension embedding, so this is an address
            // alias and not an instruction.
            Op::Embed(a) => operand(b, an, &addr, a),
            Op::Add(x, y) => match an.fuse_src[i] {
                Some(m) => {
                    let (p, q) = match prog.nodes[m as usize] {
                        Op::Mul(p, q) => (p, q),
                        _ => unreachable!("fusion source is always a Mul"),
                    };
                    let other = if m == x { y } else { x };
                    let (p, q, c) = (
                        operand(b, an, &addr, p),
                        operand(b, an, &addr, q),
                        operand(b, an, &addr, other),
                    );
                    b.emul_add(p, q, c)
                }
                None => {
                    let (x, y) = (operand(b, an, &addr, x), operand(b, an, &addr, y));
                    b.eadd(x, y)
                }
            },
            Op::Sub(x, y) => {
                let (x, y) = (operand(b, an, &addr, x), operand(b, an, &addr, y));
                b.esub(x, y)
            }
            Op::Mul(x, y) => {
                match (is_base_konst(&an.konst, x), is_base_konst(&an.konst, y)) {
                    (false, true) => {
                        let (a, s) = (operand(b, an, &addr, x), base_operand(b, an, y));
                        b.emul_base(a, s)
                    }
                    (true, false) => {
                        let (a, s) = (operand(b, an, &addr, y), base_operand(b, an, x));
                        b.emul_base(a, s)
                    }
                    // Both extension, or both base — a base word is a legal
                    // extension operand, so the plain product is correct.
                    _ => {
                        let (x, y) = (operand(b, an, &addr, x), operand(b, an, &addr, y));
                        b.emul(x, y)
                    }
                }
            }
            // The ISA has no unary negate: subtract from the pooled zero.
            Op::Neg(x) => {
                let zero = b.felt_const(FE::zero()).as_ext();
                let x = operand(b, an, &addr, x);
                b.esub(zero, x)
            }
            Op::ConstBase(_) | Op::ConstExt(_) => unreachable!("constants fold"),
        };
        addr[i] = Some(out);
    }

    prog.roots
        .iter()
        .map(|&r| operand(b, an, &addr, r))
        .collect()
}

// =============================================================================
// helpers
// =============================================================================

fn operands(op: &Op) -> Vec<u32> {
    match *op {
        Op::Add(a, b) | Op::Sub(a, b) | Op::Mul(a, b) => vec![a, b],
        Op::Neg(a) | Op::Embed(a) => vec![a],
        _ => Vec::new(),
    }
}

fn is_arith(op: &Op) -> bool {
    matches!(
        op,
        Op::Add(_, _) | Op::Sub(_, _) | Op::Mul(_, _) | Op::Neg(_) | Op::Embed(_)
    )
}

fn is_base_konst(konst: &[Option<Konst>], i: u32) -> bool {
    konst[i as usize].is_some_and(Konst::is_base)
}

fn fold_binop(
    konst: &[Option<Konst>],
    a: u32,
    b: u32,
    dim: Dim,
    base_op: impl Fn(FE, FE) -> FE,
    ext_op: impl Fn(FEE, FEE) -> FEE,
) -> Option<Konst> {
    let (ka, kb) = (konst[a as usize]?, konst[b as usize]?);
    Some(match (ka, kb, dim) {
        (Konst::Base(x), Konst::Base(y), Dim::Base) => Konst::Base(base_op(x, y)),
        _ => Konst::Ext(ext_op(ka.to_ext(), kb.to_ext())),
    })
}

/// Whether `cand` may be absorbed into a consumer `Add` as a `MulAdd`.
///
/// The single-consumer guard is what makes this sound: the IR is hash-consed, so
/// a shared `Mul` feeds several parents and fusing it into each would recompute
/// it per parent — a loss, not a saving. `fanout` counts roots as consumers, so
/// a constraint root is never fused away.
fn fusable(prog: &Prog, konst: &[Option<Konst>], fanout: &[u32], cand: u32) -> bool {
    let i = cand as usize;
    konst[i].is_none() && matches!(prog.nodes[i], Op::Mul(_, _)) && fanout[i] == 1
}

fn operand(b: &mut LfmBuilder, an: &Analysis, addr: &[Option<Ext>], i: u32) -> Ext {
    match an.konst[i as usize] {
        Some(Konst::Base(v)) => b.felt_const(v).as_ext(),
        Some(Konst::Ext(v)) => b.ext_const(&v),
        None => addr[i as usize].unwrap_or_else(|| {
            panic!("node {i} is read before it is emitted; the IR claims topological order")
        }),
    }
}

fn base_operand(b: &mut LfmBuilder, an: &Analysis, i: u32) -> Felt {
    match an.konst[i as usize] {
        Some(Konst::Base(v)) => b.felt_const(v),
        _ => unreachable!("base_operand is only called on a base-valued constant"),
    }
}

// =============================================================================
// frame supply
// =============================================================================

/// Hint one AIR's OOD frame into the machine, honouring the verifier's next-row
/// PRUNING: at frame offsets past the first, only the columns the AIR declares
/// in `next_row_columns` are opened, and every other column is reconstructed as
/// ZERO.
///
/// Getting that wrong in the permissive direction is a soundness bug rather than
/// a cost one — a column the AIR omits from its declaration is read as zero by
/// the real verifier, so a machine that hinted a value there would accept frames
/// the verifier rejects. Emitting the pooled zero constant makes the pruning
/// part of the program text instead of a property of the supplied arena.
///
/// Returns the frame and the number of arena words consumed.
pub fn hint_ood_frame(
    b: &mut LfmBuilder,
    artifact: &ConstraintArtifact,
    arena: super::instr::ArenaId,
    first_index: u32,
) -> (Vec<Vec<Ext>>, u32) {
    let shape = &artifact.shape;
    let width = (shape.main_width + shape.aux_width) as usize;
    let steps = shape.transition_offsets.len().max(1);
    let next_row: HashSet<u32> = shape.next_row_columns.iter().copied().collect();

    let zero = b.felt_const(FE::zero()).as_ext();
    let mut index = first_index;
    let mut out = Vec::with_capacity(steps);
    for offset in 0..steps {
        let mut step = Vec::with_capacity(width);
        for col in 0..width {
            let opened = offset == 0 || next_row.contains(&(col as u32));
            if opened {
                step.push(b.hint_word(arena, index).as_ext());
                index += 1;
            } else {
                step.push(zero);
            }
        }
        out.push(step);
    }
    (out, index - first_index)
}

/// Arena words [`hint_ood_frame`] consumes for this AIR — the frame's opened
/// entries, which is `width + (steps − 1) · |next_row_columns|`, not `steps ·
/// width`.
pub fn ood_frame_words(artifact: &ConstraintArtifact) -> u32 {
    let shape = &artifact.shape;
    let width = shape.main_width + shape.aux_width;
    let steps = shape.transition_offsets.len().max(1) as u32;
    width + (steps - 1) * shape.next_row_columns.len() as u32
}

// =============================================================================
// zerofier and quotient recombination
// =============================================================================

/// One boundary constraint, as program SHAPE.
///
/// Boundary constraints are deliberately NOT part of a [`ConstraintArtifact`] —
/// `AIR::boundary_constraints` is a function of the public inputs, so it is not
/// a static property of the AIR and serializing it is a separate problem. Every
/// production VM table uses `NullBoundaryConstraintBuilder`, whose only output is
/// the framework's `acc[0] = 0` on the last aux column, so in practice this is a
/// zero- or one-element list whose `point` is `g^0 = 1` and whose `value` is
/// zero. The general form is carried anyway: an emitter that silently assumed
/// the degenerate case would be wrong the first time an AIR grew a real one.
#[derive(Clone, Debug)]
pub struct BoundaryTerm {
    /// Full-width `[main | aux]` column index of the value opened at ζ.
    pub col: usize,
    /// The trace-domain point `g^step` the constraint is anchored at.
    pub point: FE,
    /// The value that column must take there.
    pub value: FEE,
}

/// Everything about the recombination that is compile-time for one sub-proof.
#[derive(Clone, Debug)]
pub struct QuotientShape {
    /// `log2(N)`. The zerofier costs exactly this many squarings, so the trace
    /// length is program SHAPE — a machine that read it from an arena would be
    /// letting the prover pick the domain it is checked against.
    pub log2_trace_length: u32,
    /// `composition_poly_degree_bound(N) / N`, i.e. how many parts the claimed
    /// composition evaluation is split into. Also shape: the verifier rejects a
    /// proof whose part count disagrees with the AIR.
    pub num_composition_parts: usize,
    /// The AIR's boundary constraints.
    pub boundary: Vec<BoundaryTerm>,
}

/// What the recombination computed.
pub struct QuotientEval {
    /// `ζ^N − 1`.
    pub zerofier: Ext,
    /// `Σ_c β^c·C_c / Z  +  Σ_k β^{n+k}·(t_k(ζ) − v_k)/(ζ − p_k)`.
    pub composition: Ext,
    /// `Σ_j part_j·ζ^j`, the claimed value the proof carries.
    pub claimed: Ext,
}

/// Emit the zerofier, the β-power fold and the claimed-composition Horner for
/// one sub-proof.
///
/// # One division, not one per constraint
///
/// Every production constraint applies to every row (`RowDomain::ALL`, measured
/// across all 28 tables), so `end_exemptions` is zero everywhere and all of an
/// AIR's constraints share the zerofier `Z = ζ^N − 1`. That lets the division
/// factor out of the β-power sum:
///
/// ```text
///   Σ_c β^c·C_c/Z  =  (Σ_c β^c·C_c)/Z
/// ```
///
/// one division per AIR rather than one per constraint. The boundary terms do
/// NOT share `Z` — they have their own denominators — so they are pre-scaled by
/// `Z` before entering the same fold, which keeps the single division while
/// still giving each boundary term its own `β` power. Naively recomputing `ζ^N`
/// and a full extension inversion per constraint, as the verifier does today,
/// would cost about 24 rows per constraint instead of per AIR.
///
/// # Why the reciprocal rather than a direct divide
///
/// `Z` and each `ζ − p_k` are inverted against the interned one, and the
/// quotient is then a multiply. That costs one extra row apiece and closes a
/// hole: the machine's convention is `0/0 = 1`, so a direct `Div` would silently
/// return 1 for a zero denominator with a zero numerator, whereas `1/0` has no
/// satisfying assignment (`B·OUT = A` becomes `0 = 1`) and is therefore
/// unprovable. `z` is sampled outside the trace domain so neither denominator
/// can vanish in an honest proof — but "the sampler prevents it" is a property
/// of a component this leg does not contain, and the guard costs two rows.
///
/// `constraint_evals` are in `constraint_idx` order, `claimed_parts` in the
/// order the proof carries them (part `j` multiplying `ζ^j`).
pub fn emit_quotient(
    b: &mut LfmBuilder,
    shape: &QuotientShape,
    ood: &OodOperands,
    zeta: Ext,
    beta: Ext,
    constraint_evals: &[Ext],
    claimed_parts: &[Ext],
) -> QuotientEval {
    assert!(
        !constraint_evals.is_empty() || !shape.boundary.is_empty(),
        "a sub-proof with neither transition nor boundary constraints has no \
         composition to check"
    );
    assert!(
        !claimed_parts.is_empty(),
        "the composition is claimed in at least one part"
    );
    assert_eq!(
        claimed_parts.len(),
        shape.num_composition_parts,
        "the supplied parts must match the AIR's part count, which is shape and \
         never read off the proof"
    );

    let one = b.ext_const(&FEE::one());

    // Z = ζ^N − 1, by repeated squaring. log2(N) rows, not N.
    let mut power = zeta;
    for _ in 0..shape.log2_trace_length {
        power = b.emul(power, power);
    }
    let zerofier = b.esub(power, one);
    let z_inv = b.ediv(one, zerofier);

    // Terms of Σ_k β^k·X_k, highest power first: the boundary terms occupy the
    // indices past the transition constraints, exactly as `replay_rounds_after_
    // round_1` splits one geometric run of β into transition then boundary
    // coefficients.
    let mut terms: Vec<Ext> = Vec::with_capacity(constraint_evals.len() + shape.boundary.len());
    for term in shape.boundary.iter().rev() {
        let opened = *ood.steps[0]
            .get(term.col)
            .unwrap_or_else(|| panic!("boundary column {} is outside the frame", term.col));
        let numerator = if term.value == FEE::zero() {
            opened
        } else {
            let v = b.ext_const(&term.value);
            b.esub(opened, v)
        };
        let point = b.felt_const(term.point).as_ext();
        let denominator = b.esub(zeta, point);
        let den_inv = b.ediv(one, denominator);
        let quotient = b.emul(numerator, den_inv);
        terms.push(b.emul(quotient, zerofier));
    }
    terms.extend(constraint_evals.iter().rev().copied());

    let mut acc = terms[0];
    for t in &terms[1..] {
        acc = b.emul_add(acc, beta, *t);
    }
    let composition = b.emul(acc, z_inv);

    // claimed = Σ_j part_j·ζ^j, the same Horner the verifier folds.
    let mut iter = claimed_parts.iter().rev();
    let mut claimed = *iter.next().expect("checked non-empty");
    for p in iter {
        claimed = b.emul_add(claimed, zeta, *p);
    }

    QuotientEval {
        zerofier,
        composition,
        claimed,
    }
}
