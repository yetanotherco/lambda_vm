//! Tests for the `ConstraintBuilder` framework: one sample [`ConstraintSet`]
//! (EqXor-shaped, IsBit-shaped and Add-carry-pair-shaped bodies, plus a
//! LogUp-shaped extension constraint) checked three ways on random rows:
//!
//! 1. `ProverEvalFolder` output == direct `FieldElement` arithmetic;
//! 2. `ProverEvalFolder` output == `eval_program` over the captured program;
//! 3. `VerifierEvalFolder` output == `eval_program_verifier` over the captured
//!    program;
//!
//! plus: capture-measured degrees == declared `meta.degree`, the meta
//! Base-prefix/density invariants, and the folders' debug-build
//! exactly-once/completeness asserts.

use math::field::element::FieldElement;
use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField as Ext;
use math::field::goldilocks::GoldilocksField as Fp;

use crate::constraint_ir::{Dim, eval_program, eval_program_verifier};
use crate::constraints::builder::{
    CaptureBuilder, ConstraintBuilder, ConstraintMeta, ConstraintSet, ProverEvalFolder, RootKind,
    RowDomain, VerifierEvalFolder, num_base_from_meta,
};
use crate::frame::Frame;
use crate::table::TableView;
use crate::traits::TransitionEvaluationContext;

type FpE = FieldElement<Fp>;
type ExtE = FieldElement<Ext>;

const TRIALS: usize = 1000;

/// Deterministic SplitMix64.
struct SplitMix64(u64);
impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn fp(&mut self) -> FpE {
        FpE::from(self.next_u64())
    }
    fn ext(&mut self) -> ExtE {
        ExtE::from_raw([self.fp(), self.fp(), self.fp()])
    }
}

// =============================================================================
// The sample table: local column layout + single body
// =============================================================================

mod cols {
    // EqXor: res = eq XOR invert.
    pub const RES: usize = 0;
    pub const EQ: usize = 1;
    pub const INVERT: usize = 2;
    // IsBit.
    pub const BIT: usize = 3;
    // Add carry pair (64-bit add split in 32-bit halves), gated by COND.
    pub const COND: usize = 4;
    pub const LHS_LO: usize = 5;
    pub const LHS_HI: usize = 6;
    pub const RHS_LO: usize = 7;
    pub const RHS_HI: usize = 8;
    pub const SUM_LO: usize = 9;
    pub const SUM_HI: usize = 10;
    pub const NUM_COLS: usize = 11;
}

/// `2^-32` as a canonical Goldilocks `u64` (the add-carry repack constant).
fn inv_shift_32() -> u64 {
    *FpE::from(1u64 << 32).inv().unwrap().value()
}

/// Sample table: 4 base constraints + 1 LogUp-shaped extension constraint.
struct SampleSet;

impl ConstraintSet<Fp, Ext> for SampleSet {
    // idx 2,3 are degree-3 carry constraints.
    fn max_degree(&self) -> usize {
        3
    }

    fn eval<B: ConstraintBuilder<Fp, Ext>>(&self, b: &mut B) {
        // idx 0 — EqXor (degree 2): res − (eq + invert − 2·eq·invert).
        let res = b.main(0, cols::RES);
        let eq = b.main(0, cols::EQ);
        let invert = b.main(0, cols::INVERT);
        let two = b.const_base(2);
        b.emit_base(0, res - (eq.clone() + invert.clone() - two * eq * invert));

        // idx 1 — IsBit (degree 2): x·(1 − x).
        let x = b.main(0, cols::BIT);
        let one = b.one();
        b.emit_base(1, x.clone() * (one - x));

        // idx 2, 3 — the add carry pair:
        //   carry_0 = (lhs.lo + rhs.lo − sum.lo)·2⁻³²
        //   carry_1 = (lhs.hi + rhs.hi + carry_0 − sum.hi)·2⁻³²
        //   emit cond·carry_i·(1 − carry_i).
        let inv_2_32 = b.const_base(inv_shift_32());
        let lhs_lo = b.main(0, cols::LHS_LO);
        let lhs_hi = b.main(0, cols::LHS_HI);
        let rhs_lo = b.main(0, cols::RHS_LO);
        let rhs_hi = b.main(0, cols::RHS_HI);
        let sum_lo = b.main(0, cols::SUM_LO);
        let sum_hi = b.main(0, cols::SUM_HI);
        let cond = b.main(0, cols::COND);
        let one = b.one();
        let carry_0 = (lhs_lo + rhs_lo - sum_lo) * inv_2_32.clone();
        let carry_1 = (lhs_hi + rhs_hi + carry_0.clone() - sum_hi) * inv_2_32;
        // idx 2, 3 — degree 3 (cond·carry·(1−carry)).
        b.emit_base(2, cond.clone() * carry_0.clone() * (one.clone() - carry_0));
        b.emit_base(3, cond * carry_1.clone() * (one - carry_1));

        // idx 4 — LogUp-shaped (degree 1): (challenge₀ + aux₀)·alpha₀ − L/N.
        let ch = b.challenge(0);
        let au = b.aux(0, 0);
        let alpha = b.alpha_pow(0);
        let off = b.table_offset();
        b.emit_ext(4, (ch + au) * alpha - off);
    }
}

const NUM_BASE: usize = 4;
const NUM_CONSTRAINTS: usize = 5;

/// Direct `FieldElement` arithmetic reference for the sample set's base
/// constraints on a main row.
fn direct_base(row: &[FpE]) -> [FpE; NUM_BASE] {
    let two = FpE::from(2u64);
    let one = FpE::one();
    let inv = FpE::from(1u64 << 32).inv().unwrap();

    let c0 = row[cols::RES]
        - (row[cols::EQ] + row[cols::INVERT] - two * row[cols::EQ] * row[cols::INVERT]);
    let c1 = row[cols::BIT] * (one - row[cols::BIT]);
    let carry_0 = (row[cols::LHS_LO] + row[cols::RHS_LO] - row[cols::SUM_LO]) * inv;
    let carry_1 = (row[cols::LHS_HI] + row[cols::RHS_HI] + carry_0 - row[cols::SUM_HI]) * inv;
    let c2 = row[cols::COND] * carry_0 * (one - carry_0);
    let c3 = row[cols::COND] * carry_1 * (one - carry_1);
    [c0, c1, c2, c3]
}

/// Direct reference for the extension constraint.
fn direct_ext(aux0: &ExtE, challenge0: &ExtE, alpha0: &ExtE, offset: &ExtE) -> ExtE {
    (*challenge0 + *aux0) * *alpha0 - *offset
}

/// One random trial's inputs.
struct TrialData {
    row: Vec<FpE>,
    aux0: ExtE,
    challenge0: ExtE,
    alpha0: ExtE,
    offset: ExtE,
}

fn random_trial(rng: &mut SplitMix64) -> TrialData {
    TrialData {
        row: (0..cols::NUM_COLS).map(|_| rng.fp()).collect(),
        aux0: rng.ext(),
        challenge0: rng.ext(),
        alpha0: rng.ext(),
        offset: rng.ext(),
    }
}

// =============================================================================
// The three-way differential checks
// =============================================================================

#[test]
fn prover_folder_matches_direct_arithmetic() {
    let mut rng = SplitMix64(0x0001_F01D_u64 ^ 0xABCD);
    for trial in 0..TRIALS {
        let t = random_trial(&mut rng);
        let step = TableView::<Fp, Ext>::new(vec![t.row.clone()], vec![vec![t.aux0]]);
        let frame = Frame::<Fp, Ext>::new(vec![step]);
        let challenges = vec![t.challenge0];
        let alphas = vec![t.alpha0];
        let ctx = TransitionEvaluationContext::new_prover(
            frame.as_row_frame(),
            &challenges,
            &alphas,
            &t.offset,
        );

        let mut base_out = vec![FpE::zero(); NUM_BASE];
        let mut ext_out = vec![ExtE::zero(); NUM_CONSTRAINTS];
        let mut folder = ProverEvalFolder::new(&ctx, &mut base_out, &mut ext_out);
        SampleSet.eval(&mut folder);
        folder.assert_all_emitted();

        let expected_base = direct_base(&t.row);
        for (i, expected) in expected_base.iter().enumerate() {
            assert_eq!(&base_out[i], expected, "base constraint {i}, trial {trial}");
        }
        let expected_ext = direct_ext(&t.aux0, &t.challenge0, &t.alpha0, &t.offset);
        assert_eq!(ext_out[4], expected_ext, "ext constraint, trial {trial}");
    }
}

#[test]
fn prover_folder_matches_interpreted_capture() {
    // Capture once (setup-time), interpret per row.
    let mut cb = CaptureBuilder::<Fp, Ext>::new();
    SampleSet.eval(&mut cb);
    let (prog, _degrees) = cb.finish(NUM_BASE);
    let mut rng = SplitMix64(0x0002_F01D_u64 ^ 0xABCD);
    for trial in 0..TRIALS {
        let t = random_trial(&mut rng);
        let step = TableView::<Fp, Ext>::new(vec![t.row.clone()], vec![vec![t.aux0]]);
        let frame = Frame::<Fp, Ext>::new(vec![step]);
        let challenges = vec![t.challenge0];
        let alphas = vec![t.alpha0];
        let ctx = TransitionEvaluationContext::new_prover(
            frame.as_row_frame(),
            &challenges,
            &alphas,
            &t.offset,
        );

        let mut folder_base = vec![FpE::zero(); NUM_BASE];
        let mut folder_ext = vec![ExtE::zero(); NUM_CONSTRAINTS];
        let mut folder = ProverEvalFolder::new(&ctx, &mut folder_base, &mut folder_ext);
        SampleSet.eval(&mut folder);
        folder.assert_all_emitted();

        let mut interp_base = vec![FpE::zero(); NUM_BASE];
        let mut interp_ext = vec![ExtE::zero(); NUM_CONSTRAINTS];
        eval_program(&prog, &ctx, &mut interp_base, &mut interp_ext);

        assert_eq!(folder_base, interp_base, "base evals, trial {trial}");
        assert_eq!(folder_ext[4], interp_ext[4], "ext eval, trial {trial}");
    }
}

#[test]
fn verifier_folder_matches_interpreted_capture() {
    let mut cb = CaptureBuilder::<Fp, Ext>::new();
    SampleSet.eval(&mut cb);
    let (prog, _degrees) = cb.finish(NUM_BASE);
    let mut rng = SplitMix64(0x0003_F01D_u64 ^ 0xABCD);
    for trial in 0..TRIALS {
        let t = random_trial(&mut rng);
        // The verifier frame holds only extension elements (OOD evaluations).
        let row_e: Vec<ExtE> = t.row.iter().map(|x| x.to_extension()).collect();
        let step = TableView::<Ext, Ext>::new(vec![row_e], vec![vec![t.aux0]]);
        let frame = Frame::<Ext, Ext>::new(vec![step]);
        let challenges = vec![t.challenge0];
        let alphas = vec![t.alpha0];
        let ctx = TransitionEvaluationContext::<Fp, Ext>::new_verifier(
            &frame,
            &challenges,
            &alphas,
            &t.offset,
        );

        let mut folder_ext = vec![ExtE::zero(); NUM_CONSTRAINTS];
        let mut folder = VerifierEvalFolder::new(&ctx, &mut folder_ext);
        SampleSet.eval(&mut folder);
        folder.assert_all_emitted();

        let mut interp_ext = vec![ExtE::zero(); NUM_CONSTRAINTS];
        eval_program_verifier(&prog, &ctx, &mut interp_ext);

        assert_eq!(folder_ext, interp_ext, "ood evals, trial {trial}");
    }
}

// =============================================================================
// Degree measurement + meta invariants
// =============================================================================

#[test]
fn capture_measured_degrees_match_declared_meta() {
    let mut cb = CaptureBuilder::<Fp, Ext>::new();
    SampleSet.eval(&mut cb);
    let (prog, degrees) = cb.finish(NUM_BASE);
    assert_eq!(prog.roots.len(), NUM_CONSTRAINTS);

    let meta = SampleSet.meta();
    assert_eq!(degrees.len(), meta.len());
    let max_degree = SampleSet.max_degree();
    for (i, &(idx, measured)) in degrees.iter().enumerate() {
        assert_eq!(idx, i, "emit order != idx order");
        assert!(
            measured <= max_degree,
            "constraint {idx}: tree-measured degree {measured} EXCEEDS max_degree() {max_degree}"
        );
    }
}

#[test]
fn meta_base_prefix_gives_num_base() {
    assert_eq!(num_base_from_meta(&SampleSet.meta()), NUM_BASE);

    // Pure-base and pure-ext lists.
    let pure_base = vec![ConstraintMeta::base(0), ConstraintMeta::base(1)];
    assert_eq!(num_base_from_meta(&pure_base), 2);
    let pure_ext = vec![ConstraintMeta::ext(0), ConstraintMeta::ext(1)];
    assert_eq!(num_base_from_meta(&pure_ext), 0);
    assert_eq!(num_base_from_meta(&[]), 0);

    // RootKind sanity on the sample.
    let meta = SampleSet.meta();
    assert!(meta[..NUM_BASE].iter().all(|m| m.kind == RootKind::Base));
    assert!(meta[NUM_BASE..].iter().all(|m| m.kind == RootKind::Ext));
}

#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "must form a prefix")]
fn meta_base_after_ext_panics() {
    let bad = vec![
        ConstraintMeta::base(0),
        ConstraintMeta::ext(1),
        ConstraintMeta::base(2),
    ];
    num_base_from_meta(&bad);
}

#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "dense and idx-ordered")]
fn meta_non_dense_panics() {
    let bad = vec![ConstraintMeta::base(0), ConstraintMeta::base(2)];
    num_base_from_meta(&bad);
}

// =============================================================================
// Folder completeness asserts (debug builds)
// =============================================================================

/// Run a body that emits only constraint 0 of 2, then check completeness.
#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "never emitted")]
fn prover_folder_missing_emit_asserts() {
    let step = TableView::<Fp, Ext>::new(vec![vec![FpE::zero(); cols::NUM_COLS]], vec![vec![]]);
    let frame = Frame::<Fp, Ext>::new(vec![step]);
    let challenges: Vec<ExtE> = vec![];
    let alphas: Vec<ExtE> = vec![];
    let offset = ExtE::zero();
    let ctx = TransitionEvaluationContext::new_prover(
        frame.as_row_frame(),
        &challenges,
        &alphas,
        &offset,
    );

    let mut base_out = vec![FpE::zero(); 2];
    let mut ext_out = vec![ExtE::zero(); 2];
    let mut folder = ProverEvalFolder::new(&ctx, &mut base_out, &mut ext_out);
    let x = folder.main(0, 0);
    folder.emit_base(0, x); // constraint 1 never emitted
    folder.assert_all_emitted();
}

#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "emitted twice")]
fn prover_folder_double_emit_asserts() {
    let step = TableView::<Fp, Ext>::new(vec![vec![FpE::zero(); cols::NUM_COLS]], vec![vec![]]);
    let frame = Frame::<Fp, Ext>::new(vec![step]);
    let challenges: Vec<ExtE> = vec![];
    let alphas: Vec<ExtE> = vec![];
    let offset = ExtE::zero();
    let ctx = TransitionEvaluationContext::new_prover(
        frame.as_row_frame(),
        &challenges,
        &alphas,
        &offset,
    );

    let mut base_out = vec![FpE::zero(); 2];
    let mut ext_out = vec![ExtE::zero(); 2];
    let mut folder = ProverEvalFolder::new(&ctx, &mut base_out, &mut ext_out);
    let x = folder.main(0, 0);
    folder.emit_base(0, x);
    let x = folder.main(0, 0);
    folder.emit_base(0, x);
}

#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "never emitted")]
fn verifier_folder_missing_emit_asserts() {
    let step = TableView::<Ext, Ext>::new(vec![vec![ExtE::zero(); cols::NUM_COLS]], vec![vec![]]);
    let frame = Frame::<Ext, Ext>::new(vec![step]);
    let challenges: Vec<ExtE> = vec![];
    let alphas: Vec<ExtE> = vec![];
    let offset = ExtE::zero();
    let ctx =
        TransitionEvaluationContext::<Fp, Ext>::new_verifier(&frame, &challenges, &alphas, &offset);

    let mut ext_out = vec![ExtE::zero(); 2];
    let mut folder = VerifierEvalFolder::new(&ctx, &mut ext_out);
    let x = folder.main(0, 0);
    folder.emit_base(1, x);
    folder.assert_all_emitted();
}

// =============================================================================
// PR-2 pre-flight: num_base alignment guard (release-checked)
// =============================================================================

/// A capture wrapper that records which `emit_*` sink each constraint index
/// used, so the meta-derived `num_base` can be checked against the body's
/// actual base-emit count (the folders route by the sink called; the
/// interpreter routes by `c < prog.num_base` — these must agree).
struct CountingCapture {
    inner: CaptureBuilder<Fp, Ext>,
    base_idxs: Vec<usize>,
    ext_idxs: Vec<usize>,
}

impl ConstraintBuilder<Fp, Ext> for CountingCapture {
    type Expr = crate::constraints::builder::IrExpr;
    type ExprE = crate::constraints::builder::IrExpr;

    fn main(&self, offset: usize, col: usize) -> Self::Expr {
        self.inner.main(offset, col)
    }
    fn aux(&self, offset: usize, col: usize) -> Self::ExprE {
        self.inner.aux(offset, col)
    }
    fn challenge(&self, idx: usize) -> Self::ExprE {
        self.inner.challenge(idx)
    }
    fn alpha_pow(&self, idx: usize) -> Self::ExprE {
        self.inner.alpha_pow(idx)
    }
    fn table_offset(&self) -> Self::ExprE {
        self.inner.table_offset()
    }
    fn const_base(&self, v: u64) -> Self::Expr {
        self.inner.const_base(v)
    }
    fn const_signed(&self, v: i64) -> Self::Expr {
        self.inner.const_signed(v)
    }
    fn emit_base_rows(&mut self, constraint_idx: usize, rows: RowDomain, e: Self::Expr) {
        self.base_idxs.push(constraint_idx);
        self.inner.emit_base_rows(constraint_idx, rows, e);
    }
    fn emit_ext_rows(&mut self, constraint_idx: usize, rows: RowDomain, e: Self::ExprE) {
        self.ext_idxs.push(constraint_idx);
        self.inner.emit_ext_rows(constraint_idx, rows, e);
    }
}

/// `num_base` has two independent sources of truth: the meta Base-prefix
/// (what the engine wires everywhere) and which `emit_*` sink the body
/// actually calls (what the folders route by; the interpreter panics via
/// `.as_base()` if `prog.num_base` disagrees with the root dims). This
/// asserts they all agree for the sample set — with plain (release-checked)
/// asserts, per plan §5.9.0.
#[test]
fn num_base_from_meta_matches_captured_base_emits() {
    let meta = SampleSet.meta();
    let num_base = num_base_from_meta(&meta);

    let mut counting = CountingCapture {
        inner: CaptureBuilder::new(),
        base_idxs: Vec::new(),
        ext_idxs: Vec::new(),
    };
    SampleSet.eval(&mut counting);
    let CountingCapture {
        inner,
        mut base_idxs,
        mut ext_idxs,
    } = counting;
    let (prog, _degrees) = inner.finish(num_base);

    // 1. The body's base-emit count equals the meta-derived num_base, and the
    //    emitted indices are exactly the meta prefix / suffix.
    base_idxs.sort_unstable();
    ext_idxs.sort_unstable();
    assert_eq!(base_idxs.len(), num_base);
    assert_eq!(base_idxs, (0..num_base).collect::<Vec<_>>());
    assert_eq!(ext_idxs, (num_base..meta.len()).collect::<Vec<_>>());

    // 2. The interpreter's routing criterion agrees: every base-prefix root is
    //    Dim::Base (otherwise eval_program's `.as_base()` would panic) and
    //    every remaining root is Dim::Ext.
    assert_eq!(prog.num_base, num_base);
    assert_eq!(prog.roots.len(), meta.len());
    for (c, &root) in prog.roots.iter().enumerate() {
        let dim = prog.dims[root as usize];
        if c < num_base {
            assert_eq!(dim, Dim::Base, "base-prefix constraint {c} has an ext root");
        } else {
            assert_eq!(dim, Dim::Ext, "ext constraint {c} has a base root");
        }
    }
}

// =============================================================================
// PR-2 pre-flight: next-row aux read + two alpha indices (LogUp shape)
// =============================================================================

/// LogUp-accumulator-shaped sample: the real 1-/2-absorbed LogUp bodies read
/// `aux(1, col)` (next-row accumulator) and use several alpha powers — the
/// primary sample covers neither.
struct NextRowLogUpSet;

mod lcols {
    /// A main witness column.
    pub const VAL: usize = 0;
    pub const NUM_MAIN: usize = 1;
    /// Aux: a term column and the accumulator.
    pub const TERM: usize = 0;
    pub const ACC: usize = 1;
    pub const NUM_AUX: usize = 2;
}

impl ConstraintSet<Fp, Ext> for NextRowLogUpSet {
    fn eval<B: ConstraintBuilder<Fp, Ext>>(&self, b: &mut B) {
        // idx 0 (base, degree 1): next-row main read — main(1, VAL) − main(0, VAL).
        let cur = b.main(0, lcols::VAL);
        let next = b.main(1, lcols::VAL);
        b.emit_base(0, next - cur);

        // idx 1 (ext, degree 1, 1 end exemption): acc' − acc − (challenge₀·α₀ + term·α₁) + L/N,
        // with acc' read from the NEXT row (aux offset 1).
        let acc = b.aux(0, lcols::ACC);
        let acc_next = b.aux(1, lcols::ACC);
        let term = b.aux(0, lcols::TERM);
        let ch = b.challenge(0);
        let a0 = b.alpha_pow(0);
        let a1 = b.alpha_pow(1);
        let off = b.table_offset();
        b.emit_ext_rows(
            1,
            RowDomain::except_last(1),
            acc_next - acc - (ch * a0 + term * a1) + off,
        );
    }
}

/// Three-way differential for [`NextRowLogUpSet`] on random two-step frames:
/// prover folder == direct arithmetic == interpreted capture, and verifier
/// folder == interpreted capture.
#[test]
fn next_row_aux_and_multi_alpha_folder_matches_capture() {
    let meta = NextRowLogUpSet.meta();
    let num_base = num_base_from_meta(&meta);
    let mut cb = CaptureBuilder::<Fp, Ext>::new();
    NextRowLogUpSet.eval(&mut cb);
    let (prog, degrees) = cb.finish(num_base);
    let max_degree = NextRowLogUpSet.max_degree();
    for &(idx, measured) in &degrees {
        assert!(
            measured <= max_degree,
            "constraint {idx}: tree degree {measured} EXCEEDS max_degree() {max_degree}"
        );
    }
    let mut rng = SplitMix64(0x0004_F01D_u64 ^ 0xABCD);
    for trial in 0..TRIALS {
        // Two frame steps with distinct main and aux rows.
        let rows: Vec<Vec<FpE>> = (0..2)
            .map(|_| (0..lcols::NUM_MAIN).map(|_| rng.fp()).collect())
            .collect();
        let auxs: Vec<Vec<ExtE>> = (0..2)
            .map(|_| (0..lcols::NUM_AUX).map(|_| rng.ext()).collect())
            .collect();
        let challenges = vec![rng.ext()];
        let alphas = vec![rng.ext(), rng.ext()];
        let offset = rng.ext();

        // --- prover folder vs direct arithmetic vs interpreter ---
        let steps: Vec<TableView<Fp, Ext>> = (0..2)
            .map(|s| TableView::new(vec![rows[s].clone()], vec![auxs[s].clone()]))
            .collect();
        let frame = Frame::<Fp, Ext>::new(steps);
        let ctx = TransitionEvaluationContext::new_prover(
            frame.as_row_frame(),
            &challenges,
            &alphas,
            &offset,
        );

        let mut folder_base = vec![FpE::zero(); num_base];
        let mut folder_ext = vec![ExtE::zero(); meta.len()];
        let mut folder = ProverEvalFolder::new(&ctx, &mut folder_base, &mut folder_ext);
        NextRowLogUpSet.eval(&mut folder);
        folder.assert_all_emitted();

        let direct_base = rows[1][lcols::VAL] - rows[0][lcols::VAL];
        let direct_ext = auxs[1][lcols::ACC]
            - auxs[0][lcols::ACC]
            - (challenges[0] * alphas[0] + auxs[0][lcols::TERM] * alphas[1])
            + offset;
        assert_eq!(folder_base[0], direct_base, "trial {trial} base direct");
        assert_eq!(folder_ext[1], direct_ext, "trial {trial} ext direct");

        let mut interp_base = vec![FpE::zero(); num_base];
        let mut interp_ext = vec![ExtE::zero(); meta.len()];
        eval_program(&prog, &ctx, &mut interp_base, &mut interp_ext);
        assert_eq!(folder_base, interp_base, "trial {trial} base interp");
        assert_eq!(folder_ext[1], interp_ext[1], "trial {trial} ext interp");

        // --- verifier folder vs interpreter ---
        let steps_e: Vec<TableView<Ext, Ext>> = (0..2)
            .map(|s| {
                TableView::new(
                    vec![rows[s].iter().map(|x| x.to_extension()).collect()],
                    vec![auxs[s].clone()],
                )
            })
            .collect();
        let frame_e = Frame::<Ext, Ext>::new(steps_e);
        let vctx = TransitionEvaluationContext::<Fp, Ext>::new_verifier(
            &frame_e,
            &challenges,
            &alphas,
            &offset,
        );

        let mut vfolder_ext = vec![ExtE::zero(); meta.len()];
        let mut vfolder = VerifierEvalFolder::new(&vctx, &mut vfolder_ext);
        NextRowLogUpSet.eval(&mut vfolder);
        vfolder.assert_all_emitted();

        let mut vinterp_ext = vec![ExtE::zero(); meta.len()];
        eval_program_verifier(&prog, &vctx, &mut vinterp_ext);
        assert_eq!(vfolder_ext, vinterp_ext, "trial {trial} verifier interp");
    }
}
