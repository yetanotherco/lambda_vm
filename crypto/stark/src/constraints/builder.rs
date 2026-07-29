//! The `ConstraintBuilder` single-body constraint front-end.
//!
//! One constraint body, written once against [`ConstraintBuilder`], is
//! interpreted three ways depending on the implementation it runs over:
//! - [`ProverEvalFolder`]: `Expr = FieldElement<F>` — compiled per-row prover
//!   evaluation (the CPU hot path).
//! - [`VerifierEvalFolder`]: `Expr = FieldElement<E>` — the same body at the
//!   OOD point (and, monomorphized into the guest binary, the recursion path;
//!   no capture, no hashing, no interpretation in-circuit).
//! - [`CaptureBuilder`]: `Expr` = an owned expression tree — one setup-time run
//!   that flattens into the flat [`ConstraintProgram`] IR for the CPU
//!   interpreter and the GPU, measuring constraint degrees along the way.
//!
//! A table's constraints are packaged as a [`ConstraintSet`]: idx-ordered
//! [`ConstraintMeta`] (plain data: kind, declared degree, zerofier shape) plus
//! THE single `eval` body that emits every constraint.
//!
//! Fixed packing-shift constants (`2^8`/`2^16`/`2^24`) have no dedicated leaf:
//! bodies lower them through `const_base`, like any other structural constant.

use std::marker::PhantomData;
use std::ops::{Add, Mul, Neg, Sub};
use std::rc::Rc;

use math::field::element::FieldElement;
use math::field::traits::{IsField, IsSubFieldOf};

use crate::constraint_ir::{ConstraintProgram, Dim, IrBuilder};
use crate::frame::{Frame, RowFrame};
use crate::traits::TransitionEvaluationContext;

// =============================================================================
// Operator-bound aliases
// =============================================================================

/// Base-field expression operations. `Ext` is the builder's extension
/// expression type; mixed ops keep the base operand on the LEFT (the field
/// tower only implements subfield ∘ superfield, not the reverse — see
/// `math::field::element` operator impls).
pub trait ExprOps<Ext>:
    Sized
    + Clone
    + Add<Self, Output = Self>
    + Sub<Self, Output = Self>
    + Mul<Self, Output = Self>
    + Neg<Output = Self>
    + Add<Ext, Output = Ext>
    + Sub<Ext, Output = Ext>
    + Mul<Ext, Output = Ext>
{
}
impl<T, Ext> ExprOps<Ext> for T where
    T: Sized
        + Clone
        + Add<T, Output = T>
        + Sub<T, Output = T>
        + Mul<T, Output = T>
        + Neg<Output = T>
        + Add<Ext, Output = Ext>
        + Sub<Ext, Output = Ext>
        + Mul<Ext, Output = Ext>
{
}

/// Extension-field expression operations (self ops only; base×ext lives on
/// [`ExprOps`] so the base operand stays on the left).
pub trait ExtExprOps:
    Sized
    + Clone
    + Add<Self, Output = Self>
    + Sub<Self, Output = Self>
    + Mul<Self, Output = Self>
    + Neg<Output = Self>
{
}
impl<T> ExtExprOps for T where
    T: Sized
        + Clone
        + Add<T, Output = T>
        + Sub<T, Output = T>
        + Mul<T, Output = T>
        + Neg<Output = T>
{
}

// =============================================================================
// The trait
// =============================================================================

/// The single-body constraint front-end: leaves + emit sinks. Constraint
/// bodies are generic over an implementation of this trait; the associated
/// `Expr`/`ExprE` types decide what a run of the body *means*.
///
/// `const_base`/`const_signed` are the ONLY constant path — there is no
/// `From<FieldElement<F>>` on `Expr` (it would be wrong for
/// [`VerifierEvalFolder`], where `Expr = FieldElement<E>`).
pub trait ConstraintBuilder<F: IsField, E: IsField> {
    /// Base-field expression.
    type Expr: ExprOps<Self::ExprE>;
    /// Extension-field expression.
    type ExprE: ExtExprOps;

    // ---- leaves ---------------------------------------------------------
    fn main(&self, offset: usize, col: usize) -> Self::Expr;
    fn aux(&self, offset: usize, col: usize) -> Self::ExprE;
    /// `rap_challenges[idx]`.
    fn challenge(&self, idx: usize) -> Self::ExprE;
    /// `logup_alpha_powers[idx]`.
    fn alpha_pow(&self, idx: usize) -> Self::ExprE;
    /// The LogUp table offset `L/N`.
    fn table_offset(&self) -> Self::ExprE;
    fn const_base(&self, v: u64) -> Self::Expr;
    fn const_signed(&self, v: i64) -> Self::Expr;
    fn one(&self) -> Self::Expr {
        self.const_base(1)
    }
    fn zero(&self) -> Self::Expr {
        self.const_base(0)
    }

    // ---- sinks ----------------------------------------------------------
    /// Record base-field constraint `constraint_idx`'s value over the trace
    /// `rows` it applies to (see [`RowDomain`]). Recording it here is what lets
    /// [`ConstraintSet::meta`] be *derived* from this single body (via
    /// [`MetaBuilder`]) instead of hand-maintained as a parallel list. The
    /// constraint's polynomial degree is NOT declared per-constraint — only the
    /// per-table max matters, declared once via [`ConstraintSet::max_degree`].
    fn emit_base_rows(&mut self, constraint_idx: usize, rows: RowDomain, e: Self::Expr);
    /// Extension-field (LogUp) counterpart of [`Self::emit_base_rows`].
    fn emit_ext_rows(&mut self, constraint_idx: usize, rows: RowDomain, e: Self::ExprE);
    /// Record a base-field constraint that applies to every row (common case).
    #[inline]
    fn emit_base(&mut self, constraint_idx: usize, e: Self::Expr) {
        self.emit_base_rows(constraint_idx, RowDomain::ALL, e);
    }
    /// Record an extension-field (LogUp) constraint that applies to every row.
    #[inline]
    fn emit_ext(&mut self, constraint_idx: usize, e: Self::ExprE) {
        self.emit_ext_rows(constraint_idx, RowDomain::ALL, e);
    }

    // ---- folds ----------------------------------------------------------
    /// Fold one α·value term into a running LogUp fingerprint:
    /// `fp − v·α[alpha_idx]`.
    ///
    /// This default emits the multiply unconditionally — the only option for
    /// capture (the IR has no data-dependent control flow) and correct for
    /// every builder. [`ProverEvalFolder`] overrides it with a zero-skip: a
    /// bus element that is zero on this row contributes nothing (`0·α = 0`),
    /// so the F×E multiply is skipped. That covers the constant-0 bus-width
    /// padding plus any variable element that is zero on the row, and it runs
    /// once per fingerprint element per LDE row — the hot path where the old
    /// runtime body had the same skip.
    fn fold_fingerprint_term(
        &self,
        fp: Self::ExprE,
        v: Self::Expr,
        alpha_idx: usize,
    ) -> Self::ExprE {
        fp - v * self.alpha_pow(alpha_idx)
    }
}

// =============================================================================
// Constraint metadata
// =============================================================================

/// Whether a constraint's root value lives in the base field or the extension.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RootKind {
    /// Base-field constraint (algebraic table constraints).
    Base,
    /// Extension-field constraint (LogUp).
    Ext,
}

/// Which trace rows a transition constraint applies to. `ALL` = every row;
/// `except_last(n)` skips the final `n` rows — used by constraints that read
/// `n` rows ahead (the last `n` rows have no valid "next" to check). Passed at
/// the emit site; degree is NOT here (it's a per-table property, see
/// [`ConstraintSet::max_degree`]) — the two are orthogonal.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct RowDomain {
    /// Number of exempted rows at the end of the trace.
    pub end_exemptions: usize,
}

impl RowDomain {
    /// Every row (no exemptions).
    pub const ALL: RowDomain = RowDomain { end_exemptions: 0 };
    /// Every row except the last `n`.
    pub const fn except_last(n: usize) -> RowDomain {
        RowDomain { end_exemptions: n }
    }
}

/// Per-constraint metadata, DERIVED from the body (via [`MetaBuilder`]). `Base`
/// entries MUST form a prefix of an idx-ordered, dense list — see
/// [`num_base_from_meta`]. Degree is intentionally absent: only the per-table
/// max is consumed (by `composition_poly_degree_bound`), declared once via
/// [`ConstraintSet::max_degree`].
#[derive(Clone, Debug)]
pub struct ConstraintMeta {
    pub constraint_idx: usize,
    /// Base | Ext; Base entries MUST be a prefix.
    pub kind: RootKind,
    /// Number of exempted rows at the end of the trace (default 0).
    pub end_exemptions: usize,
}

impl ConstraintMeta {
    /// A base-field constraint applying to every row.
    pub fn base(constraint_idx: usize) -> Self {
        Self {
            constraint_idx,
            kind: RootKind::Base,
            end_exemptions: 0,
        }
    }

    /// An extension-field (LogUp) constraint applying to every row.
    pub fn ext(constraint_idx: usize) -> Self {
        Self {
            kind: RootKind::Ext,
            ..Self::base(constraint_idx)
        }
    }

    pub fn with_end_exemptions(mut self, end_exemptions: usize) -> Self {
        self.end_exemptions = end_exemptions;
        self
    }
}

/// Compute `num_base` from a table's metadata, debug-asserting the invariants:
/// the list is dense and idx-ordered (`meta[i].constraint_idx == i`) and
/// `RootKind::Base` entries form a prefix — the prefix length IS `num_base`,
/// matching the engine's existing base/ext split convention.
pub fn num_base_from_meta(meta: &[ConstraintMeta]) -> usize {
    let num_base = meta.iter().take_while(|m| m.kind == RootKind::Base).count();
    #[cfg(debug_assertions)]
    for (i, m) in meta.iter().enumerate() {
        assert_eq!(
            m.constraint_idx, i,
            "constraint meta must be dense and idx-ordered: entry {i} has idx {}",
            m.constraint_idx
        );
        assert!(
            (m.kind == RootKind::Base) == (i < num_base),
            "RootKind::Base entries must form a prefix: entry {i} is {:?}",
            m.kind
        );
    }
    num_base
}

/// One table's constraints: THE single body.
///
/// `eval` is the sole source of truth — it emits every constraint once,
/// declaring each one's kind (via `emit_base`/`emit_ext`), degree, and
/// end-exemptions at the emit site. `meta()` is DERIVED from it by running the
/// same body through a [`MetaBuilder`], so there is no parallel list to keep in
/// sync. See [`num_base_from_meta`] for the invariants the derived metadata
/// upholds.
pub trait ConstraintSet<F: IsField, E: IsField>: Send + Sync {
    /// The single constraint body: emits every constraint exactly once.
    fn eval<B: ConstraintBuilder<F, E>>(&self, b: &mut B);

    /// The maximum multivariate degree over this set's base constraints — the
    /// only degree info the proof consumes (via `composition_poly_degree_bound`,
    /// which takes the per-table max). Declared once here instead of per
    /// constraint; default 2 covers most tables, override to 3 for the few that
    /// have a degree-3 constraint. Hand-declared, never auto-measured (that
    /// would change the composition bound); the capture path asserts every
    /// constraint's measured degree is `<=` this.
    fn max_degree(&self) -> usize {
        2
    }

    /// Main-trace columns this set reads on the NEXT row (via `main(1, col)`).
    /// The verifier opens OOD next-row evaluations only for these columns (unioned
    /// with the LogUp accumulator); an undeclared next-row read is pruned and
    /// reconstructed as zero, silently corrupting this table's transition eval.
    /// Statically declared so the verify/recursion path never materializes the IR;
    /// a test asserts the IR's actual next-row reads are a subset of this. Default
    /// empty (most tables read only the current row).
    fn next_row_columns(&self) -> Vec<usize> {
        Vec::new()
    }

    /// Idx-ordered metadata, derived by running [`Self::eval`] through a
    /// [`MetaBuilder`] (which records the `{kind, end_exemptions}` at each
    /// `emit_*`). Never overridden — the body is the source.
    fn meta(&self) -> Vec<ConstraintMeta> {
        let mut mb = MetaBuilder::new();
        self.eval(&mut mb);
        mb.into_meta()
    }
}

/// A [`ConstraintSet`] with no transition constraints — for tables whose
/// soundness rests entirely on their bus (LogUp) interactions (e.g. BITWISE,
/// PAGE, REGISTER, the continuation GLOBAL_MEMORY / global L2G sub-tables).
/// The framework still appends the LogUp constraints; this contributes nothing
/// before them.
pub struct EmptyConstraints;

impl<F: IsField, E: IsField> ConstraintSet<F, E> for EmptyConstraints {
    fn eval<B: ConstraintBuilder<F, E>>(&self, _b: &mut B) {}
}

// =============================================================================
// MetaBuilder — derive ConstraintMeta by running the body with no arithmetic
// =============================================================================

/// No-op expression for [`MetaBuilder`]: every leaf and operator yields `Nil`,
/// so running a constraint body over it does no field work — it only drives the
/// `emit_*` calls, which is all metadata derivation needs.
#[derive(Clone, Copy)]
pub struct Nil;

impl core::ops::Add for Nil {
    type Output = Nil;
    fn add(self, _rhs: Nil) -> Nil {
        Nil
    }
}
impl core::ops::Sub for Nil {
    type Output = Nil;
    fn sub(self, _rhs: Nil) -> Nil {
        Nil
    }
}
impl core::ops::Mul for Nil {
    type Output = Nil;
    fn mul(self, _rhs: Nil) -> Nil {
        Nil
    }
}
impl core::ops::Neg for Nil {
    type Output = Nil;
    fn neg(self) -> Nil {
        Nil
    }
}

/// Derives [`ConstraintMeta`] from a [`ConstraintSet`] body: a metadata-only
/// [`ConstraintBuilder`] whose leaves/operators are no-ops and whose `emit_*`
/// sinks record `{constraint_idx, kind, degree, end_exemptions}`. Runs once at
/// setup — never on the per-row prover path.
pub struct MetaBuilder {
    metas: Vec<ConstraintMeta>,
}

impl MetaBuilder {
    pub fn new() -> Self {
        Self { metas: Vec::new() }
    }

    /// The recorded metadata, sorted by `constraint_idx` (emission order need
    /// not match index order; the sort restores the dense idx-ordering
    /// [`num_base_from_meta`] expects).
    pub fn into_meta(mut self) -> Vec<ConstraintMeta> {
        self.metas.sort_by_key(|m| m.constraint_idx);
        self.metas
    }
}

impl Default for MetaBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: IsField, E: IsField> ConstraintBuilder<F, E> for MetaBuilder {
    type Expr = Nil;
    type ExprE = Nil;

    fn main(&self, _offset: usize, _col: usize) -> Nil {
        Nil
    }
    fn aux(&self, _offset: usize, _col: usize) -> Nil {
        Nil
    }
    fn challenge(&self, _idx: usize) -> Nil {
        Nil
    }
    fn alpha_pow(&self, _idx: usize) -> Nil {
        Nil
    }
    fn table_offset(&self) -> Nil {
        Nil
    }
    fn const_base(&self, _v: u64) -> Nil {
        Nil
    }
    fn const_signed(&self, _v: i64) -> Nil {
        Nil
    }

    fn emit_base_rows(&mut self, constraint_idx: usize, rows: RowDomain, _e: Nil) {
        self.metas.push(ConstraintMeta {
            constraint_idx,
            kind: RootKind::Base,
            end_exemptions: rows.end_exemptions,
        });
    }
    fn emit_ext_rows(&mut self, constraint_idx: usize, rows: RowDomain, _e: Nil) {
        self.metas.push(ConstraintMeta {
            constraint_idx,
            kind: RootKind::Ext,
            end_exemptions: rows.end_exemptions,
        });
    }
}

// =============================================================================
// Shared AIR plumbing: run a ConstraintSet through the folders
// =============================================================================

/// Run a [`ConstraintSet`] through the [`ProverEvalFolder`]: the body of an
/// `AIR::compute_transition_prover` override. `base_evals` must be sized
/// `num_base` (the Base-prefix length of the set's meta, see
/// [`num_base_from_meta`]) and `ext_evals` the total constraint count —
/// the engine's existing contract.
///
/// Panics if `ctx` is the Verifier variant (the engine only calls the
/// prover path with a prover frame).
pub fn run_transition_prover<F, E, CS>(
    cs: &CS,
    ctx: &TransitionEvaluationContext<'_, F, E>,
    base_evals: &mut [FieldElement<F>],
    ext_evals: &mut [FieldElement<E>],
) where
    F: IsSubFieldOf<E>,
    E: IsField,
    CS: ConstraintSet<F, E>,
{
    let mut folder = ProverEvalFolder::new(ctx, base_evals, ext_evals);
    cs.eval(&mut folder);
    folder.assert_all_emitted();
}

/// Run a [`ConstraintSet`] at a single point, returning all constraint
/// values in the extension field: the body of an `AIR::compute_transition`
/// override.
///
/// A Verifier context runs the [`VerifierEvalFolder`] (the OOD/recursion
/// path). A Prover context is also accepted — debug trace validation calls
/// this method with a prover frame — by running the [`ProverEvalFolder`]
/// and promoting the Base-prefix results into the extension.
pub fn run_transition_verifier<F, E, CS>(
    cs: &CS,
    ctx: &TransitionEvaluationContext<'_, F, E>,
    num_base: usize,
    num_constraints: usize,
) -> Vec<FieldElement<E>>
where
    F: IsSubFieldOf<E>,
    E: IsField,
    CS: ConstraintSet<F, E>,
{
    let mut ext_evals = vec![FieldElement::<E>::zero(); num_constraints];
    match ctx {
        TransitionEvaluationContext::Verifier { .. } => {
            let mut folder = VerifierEvalFolder::new(ctx, &mut ext_evals);
            cs.eval(&mut folder);
            folder.assert_all_emitted();
        }
        TransitionEvaluationContext::Prover { .. } => {
            let mut base_evals = vec![FieldElement::<F>::zero(); num_base];
            let mut folder = ProverEvalFolder::new(ctx, &mut base_evals, &mut ext_evals);
            cs.eval(&mut folder);
            folder.assert_all_emitted();
            for (slot, base) in ext_evals.iter_mut().zip(base_evals) {
                *slot = base.to_extension();
            }
        }
    }
    ext_evals
}

// =============================================================================
// Debug-build emit tracking (shared by the folders)
// =============================================================================

/// Debug-build bitset asserting every constraint index is emitted exactly
/// once. A zero-sized no-op in release builds.
struct EmitTracker {
    #[cfg(debug_assertions)]
    seen: Vec<bool>,
}

impl EmitTracker {
    fn new(_num_constraints: usize) -> Self {
        Self {
            #[cfg(debug_assertions)]
            seen: vec![false; _num_constraints],
        }
    }

    #[inline]
    fn mark(&mut self, _idx: usize) {
        #[cfg(debug_assertions)]
        {
            assert!(
                _idx < self.seen.len(),
                "constraint idx {_idx} out of range ({} constraints)",
                self.seen.len()
            );
            assert!(!self.seen[_idx], "constraint {_idx} emitted twice");
            self.seen[_idx] = true;
        }
    }

    fn assert_complete(&self) {
        #[cfg(debug_assertions)]
        for (i, emitted) in self.seen.iter().enumerate() {
            assert!(emitted, "constraint {i} was never emitted");
        }
    }
}

// =============================================================================
// 1. ProverEvalFolder — compiled per-row evaluation (base-field frame)
// =============================================================================

/// Direct evaluation over one prover row: `Expr = FieldElement<F>`,
/// `ExprE = FieldElement<E>`. Constructed per row from the Prover
/// [`TransitionEvaluationContext`] variant plus the output slices;
/// `emit_base` writes `base_evals[idx]`, `emit_ext` writes `ext_evals[idx]`
/// (ABSOLUTE constraint index — `ext_evals` is sized to the total constraint
/// count). This is the CPU hot path: after inlining, a body run is the same
/// machine code as a hand-written `evaluate`.
pub struct ProverEvalFolder<'a, F, E>
where
    F: IsSubFieldOf<E>,
    E: IsField,
{
    rows: RowFrame<'a, F, E>,
    challenges: &'a [FieldElement<E>],
    alphas: &'a [FieldElement<E>],
    logup_table_offset: &'a FieldElement<E>,
    base_out: &'a mut [FieldElement<F>],
    ext_out: &'a mut [FieldElement<E>],
    tracker: EmitTracker,
}

impl<'a, F, E> ProverEvalFolder<'a, F, E>
where
    F: IsSubFieldOf<E>,
    E: IsField,
{
    /// Build a folder from the Prover context variant. `base_out` must be
    /// sized `num_base`; `ext_out` must be sized to the total constraint
    /// count (matching the engine's `compute_transition_prover` contract).
    ///
    /// Panics if `ctx` is the Verifier variant.
    pub fn new(
        ctx: &TransitionEvaluationContext<'a, F, E>,
        base_out: &'a mut [FieldElement<F>],
        ext_out: &'a mut [FieldElement<E>],
    ) -> Self {
        let TransitionEvaluationContext::Prover {
            rows,
            rap_challenges,
            logup_alpha_powers,
            logup_table_offset,
            ..
        } = ctx
        else {
            unreachable!("ProverEvalFolder::new called with a Verifier context")
        };
        let num_constraints = base_out.len().max(ext_out.len());
        Self {
            rows: *rows,
            challenges: rap_challenges,
            alphas: logup_alpha_powers,
            logup_table_offset,
            base_out,
            ext_out,
            tracker: EmitTracker::new(num_constraints),
        }
    }

    /// Debug-build check that every constraint index was emitted exactly
    /// once (no-op in release builds). Call after running a body.
    pub fn assert_all_emitted(&self) {
        self.tracker.assert_complete();
    }
}

impl<F, E> ConstraintBuilder<F, E> for ProverEvalFolder<'_, F, E>
where
    F: IsSubFieldOf<E>,
    E: IsField,
{
    type Expr = FieldElement<F>;
    type ExprE = FieldElement<E>;

    fn main(&self, offset: usize, col: usize) -> FieldElement<F> {
        self.rows.main(offset, col).clone()
    }
    fn aux(&self, offset: usize, col: usize) -> FieldElement<E> {
        self.rows.aux(offset, col).clone()
    }
    fn challenge(&self, idx: usize) -> FieldElement<E> {
        self.challenges[idx].clone()
    }
    fn alpha_pow(&self, idx: usize) -> FieldElement<E> {
        self.alphas[idx].clone()
    }
    fn table_offset(&self) -> FieldElement<E> {
        self.logup_table_offset.clone()
    }
    fn const_base(&self, v: u64) -> FieldElement<F> {
        FieldElement::<F>::from(v)
    }
    fn const_signed(&self, v: i64) -> FieldElement<F> {
        FieldElement::<F>::from(v)
    }

    #[inline]
    fn emit_base_rows(&mut self, constraint_idx: usize, _rows: RowDomain, e: FieldElement<F>) {
        self.tracker.mark(constraint_idx);
        self.base_out[constraint_idx] = e;
    }
    #[inline]
    fn emit_ext_rows(&mut self, constraint_idx: usize, _rows: RowDomain, e: FieldElement<E>) {
        debug_assert!(
            constraint_idx >= self.base_out.len(),
            "emit_ext with a base-prefix index {constraint_idx}"
        );
        self.tracker.mark(constraint_idx);
        self.ext_out[constraint_idx] = e;
    }

    fn fold_fingerprint_term(
        &self,
        fp: FieldElement<E>,
        v: FieldElement<F>,
        alpha_idx: usize,
    ) -> FieldElement<E> {
        // Zero bus elements contribute nothing — skip the F×E multiply.
        if v == FieldElement::zero() {
            fp
        } else {
            fp - v * &self.alphas[alpha_idx]
        }
    }
}

// =============================================================================
// 2. VerifierEvalFolder — same body at the OOD point (all-extension frame)
// =============================================================================

/// Direct evaluation at the OOD point: the frame holds only extension
/// elements, so `Expr = FieldElement<E>` and base-constraint results are
/// already extension values. `const_base` embeds via
/// `FieldElement::<F>::from(v).to_extension::<E>()`; `emit_base` writes the
/// (already promoted) value into `ext_evals[idx]`, mirroring the old
/// adapter's `evaluate(..).to_extension()` promotion. Runs once per proof at
/// the OOD point; this exact monomorphization, compiled into the guest
/// binary, is the recursion-guest path.
pub struct VerifierEvalFolder<'a, F, E>
where
    F: IsSubFieldOf<E>,
    E: IsField,
{
    frame: &'a Frame<E, E>,
    challenges: &'a [FieldElement<E>],
    alphas: &'a [FieldElement<E>],
    logup_table_offset: &'a FieldElement<E>,
    ext_out: &'a mut [FieldElement<E>],
    tracker: EmitTracker,
    _base_field: PhantomData<F>,
}

impl<'a, F, E> VerifierEvalFolder<'a, F, E>
where
    F: IsSubFieldOf<E>,
    E: IsField,
{
    /// Build a folder from the Verifier context variant. `ext_out` must be
    /// sized to the total constraint count (matching the engine's
    /// `compute_transition` contract).
    ///
    /// Panics if `ctx` is the Prover variant.
    pub fn new(
        ctx: &TransitionEvaluationContext<'a, F, E>,
        ext_out: &'a mut [FieldElement<E>],
    ) -> Self {
        let TransitionEvaluationContext::Verifier {
            frame,
            rap_challenges,
            logup_alpha_powers,
            logup_table_offset,
            ..
        } = ctx
        else {
            unreachable!("VerifierEvalFolder::new called with a Prover context")
        };
        let num_constraints = ext_out.len();
        Self {
            frame,
            challenges: rap_challenges,
            alphas: logup_alpha_powers,
            logup_table_offset,
            ext_out,
            tracker: EmitTracker::new(num_constraints),
            _base_field: PhantomData,
        }
    }

    /// Debug-build check that every constraint index was emitted exactly
    /// once (no-op in release builds). Call after running a body.
    pub fn assert_all_emitted(&self) {
        self.tracker.assert_complete();
    }
}

impl<F, E> ConstraintBuilder<F, E> for VerifierEvalFolder<'_, F, E>
where
    F: IsSubFieldOf<E>,
    E: IsField,
{
    type Expr = FieldElement<E>;
    type ExprE = FieldElement<E>;

    fn main(&self, offset: usize, col: usize) -> FieldElement<E> {
        self.frame
            .get_evaluation_step(offset)
            .get_main_evaluation_element(0, col)
            .clone()
    }
    fn aux(&self, offset: usize, col: usize) -> FieldElement<E> {
        self.frame
            .get_evaluation_step(offset)
            .get_aux_evaluation_element(0, col)
            .clone()
    }
    fn challenge(&self, idx: usize) -> FieldElement<E> {
        self.challenges[idx].clone()
    }
    fn alpha_pow(&self, idx: usize) -> FieldElement<E> {
        self.alphas[idx].clone()
    }
    fn table_offset(&self) -> FieldElement<E> {
        self.logup_table_offset.clone()
    }
    fn const_base(&self, v: u64) -> FieldElement<E> {
        FieldElement::<F>::from(v).to_extension::<E>()
    }
    fn const_signed(&self, v: i64) -> FieldElement<E> {
        FieldElement::<F>::from(v).to_extension::<E>()
    }

    fn emit_base_rows(&mut self, constraint_idx: usize, _rows: RowDomain, e: FieldElement<E>) {
        self.tracker.mark(constraint_idx);
        self.ext_out[constraint_idx] = e;
    }
    fn emit_ext_rows(&mut self, constraint_idx: usize, _rows: RowDomain, e: FieldElement<E>) {
        self.tracker.mark(constraint_idx);
        self.ext_out[constraint_idx] = e;
    }
}

// =============================================================================
// 3. CaptureBuilder — owned expression tree, flattened into the flat IR
// =============================================================================

/// One node of the capture tree. `degree` is eager (leaf var = 1,
/// constants/uniforms = 0, mul sums, add/sub max, neg passthrough — p3's
/// `degree_multiple`).
struct TreeNode {
    kind: TreeKind,
    dim: Dim,
    degree: usize,
}

enum TreeKind {
    Main {
        offset: u8,
        col: u16,
    },
    Aux {
        offset: u8,
        col: u16,
    },
    Challenge(u16),
    AlphaPow(u16),
    TableOffset,
    /// Raw `u64` base-field constant; canonicalized (and value-deduplicated)
    /// by the [`IrBuilder`] at flatten time.
    ConstBase(u64),
    /// Raw `i64` base-field constant; negatives map to `p - |v|` at flatten
    /// time, exactly as `IrBuilder::const_signed`.
    ConstSigned(i64),
    Add(IrExpr, IrExpr),
    Sub(IrExpr, IrExpr),
    Mul(IrExpr, IrExpr),
    Neg(IrExpr),
}

/// Owned capture expression: `Rc` tree with operator overloading. Cloning is
/// a pointer bump; operators allocate nodes — no arena, no interior
/// mutability, no hashing (CSE happens at flatten time via [`IrBuilder`]).
/// Constants carry raw integers, so the tree needs no field type parameters.
#[derive(Clone)]
pub struct IrExpr(Rc<TreeNode>);

impl IrExpr {
    fn leaf(kind: TreeKind, dim: Dim, degree: usize) -> Self {
        IrExpr(Rc::new(TreeNode { kind, dim, degree }))
    }

    fn join(a: Dim, b: Dim) -> Dim {
        match (a, b) {
            (Dim::Base, Dim::Base) => Dim::Base,
            _ => Dim::Ext,
        }
    }

    fn binop(f: fn(IrExpr, IrExpr) -> TreeKind, degree: usize, a: IrExpr, b: IrExpr) -> Self {
        let dim = Self::join(a.0.dim, b.0.dim);
        IrExpr(Rc::new(TreeNode {
            kind: f(a, b),
            dim,
            degree,
        }))
    }

    /// The tree-measured constraint degree (multivariate, in trace columns).
    pub fn degree(&self) -> usize {
        self.0.degree
    }
}

impl Add for IrExpr {
    type Output = IrExpr;
    fn add(self, rhs: IrExpr) -> IrExpr {
        let d = self.0.degree.max(rhs.0.degree);
        IrExpr::binop(TreeKind::Add, d, self, rhs)
    }
}
impl Sub for IrExpr {
    type Output = IrExpr;
    fn sub(self, rhs: IrExpr) -> IrExpr {
        let d = self.0.degree.max(rhs.0.degree);
        IrExpr::binop(TreeKind::Sub, d, self, rhs)
    }
}
impl Mul for IrExpr {
    type Output = IrExpr;
    // The degree of a product is the SUM of the factor degrees.
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn mul(self, rhs: IrExpr) -> IrExpr {
        let d = self.0.degree + rhs.0.degree;
        IrExpr::binop(TreeKind::Mul, d, self, rhs)
    }
}
impl Neg for IrExpr {
    type Output = IrExpr;
    fn neg(self) -> IrExpr {
        let (dim, degree) = (self.0.dim, self.0.degree);
        IrExpr(Rc::new(TreeNode {
            kind: TreeKind::Neg(self),
            dim,
            degree,
        }))
    }
}

/// Captures every emitted constraint into a [`ConstraintProgram`] by
/// flattening the finished trees into an [`IrBuilder`] (whose hash-consing
/// provides structural CSE, host-side, once at setup). Also records each
/// root's tree-measured degree — the degree-measurement API backing the
/// declared-vs-measured gate.
pub struct CaptureBuilder<F: IsField, E: IsField> {
    ir: IrBuilder<F, E>,
    /// `(constraint_idx, tree-measured degree)` per emit.
    degrees: Vec<(usize, usize)>,
}

impl<F: IsField, E: IsField> Default for CaptureBuilder<F, E> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: IsField, E: IsField> CaptureBuilder<F, E> {
    pub fn new() -> Self {
        Self {
            ir: IrBuilder::new(),
            degrees: Vec::new(),
        }
    }

    fn flatten(&mut self, e: &IrExpr) -> crate::constraint_ir::Expr {
        match &e.0.kind {
            TreeKind::Main { offset, col } => self.ir.main(*offset, *col as usize),
            TreeKind::Aux { offset, col } => self.ir.aux(*offset, *col as usize),
            TreeKind::Challenge(idx) => self.ir.challenge(*idx as usize),
            TreeKind::AlphaPow(idx) => self.ir.alpha_power(*idx as usize),
            TreeKind::TableOffset => self.ir.table_offset(),
            TreeKind::ConstBase(v) => self.ir.const_base(*v),
            TreeKind::ConstSigned(v) => self.ir.const_signed(*v),
            TreeKind::Add(a, b) => {
                let (fa, fb) = (self.flatten(a), self.flatten(b));
                self.ir.add(fa, fb)
            }
            TreeKind::Sub(a, b) => {
                let (fa, fb) = (self.flatten(a), self.flatten(b));
                self.ir.sub(fa, fb)
            }
            TreeKind::Mul(a, b) => {
                let (fa, fb) = (self.flatten(a), self.flatten(b));
                self.ir.mul(fa, fb)
            }
            TreeKind::Neg(a) => {
                let fa = self.flatten(a);
                self.ir.neg(fa)
            }
        }
    }

    /// Finish capture: `(program, per-emit tree-measured degrees)`.
    pub fn finish(self, num_base: usize) -> (ConstraintProgram<F, E>, Vec<(usize, usize)>) {
        (self.ir.finish(num_base), self.degrees)
    }
}

impl<F: IsField, E: IsField> ConstraintBuilder<F, E> for CaptureBuilder<F, E> {
    type Expr = IrExpr;
    type ExprE = IrExpr;

    fn main(&self, offset: usize, col: usize) -> IrExpr {
        // Capture runs once at setup — assert the narrow IR encodings fit
        // rather than silently truncating into the GPU program.
        assert!(u8::try_from(offset).is_ok() && u16::try_from(col).is_ok());
        IrExpr::leaf(
            TreeKind::Main {
                offset: offset as u8,
                col: col as u16,
            },
            Dim::Base,
            1,
        )
    }
    fn aux(&self, offset: usize, col: usize) -> IrExpr {
        assert!(u8::try_from(offset).is_ok() && u16::try_from(col).is_ok());
        IrExpr::leaf(
            TreeKind::Aux {
                offset: offset as u8,
                col: col as u16,
            },
            Dim::Ext,
            1,
        )
    }
    fn challenge(&self, idx: usize) -> IrExpr {
        assert!(u16::try_from(idx).is_ok());
        IrExpr::leaf(TreeKind::Challenge(idx as u16), Dim::Ext, 0)
    }
    fn alpha_pow(&self, idx: usize) -> IrExpr {
        assert!(u16::try_from(idx).is_ok());
        IrExpr::leaf(TreeKind::AlphaPow(idx as u16), Dim::Ext, 0)
    }
    fn table_offset(&self) -> IrExpr {
        IrExpr::leaf(TreeKind::TableOffset, Dim::Ext, 0)
    }
    fn const_base(&self, v: u64) -> IrExpr {
        IrExpr::leaf(TreeKind::ConstBase(v), Dim::Base, 0)
    }
    fn const_signed(&self, v: i64) -> IrExpr {
        IrExpr::leaf(TreeKind::ConstSigned(v), Dim::Base, 0)
    }

    fn emit_base_rows(&mut self, constraint_idx: usize, _rows: RowDomain, e: IrExpr) {
        debug_assert_eq!(e.0.dim, Dim::Base, "emit_base on an extension expression");
        let root = self.flatten(&e);
        self.ir.emit(constraint_idx, root);
        // Record the TREE-MEASURED degree so the host-side test can assert
        // measured <= the table's declared max_degree().
        self.degrees.push((constraint_idx, e.degree()));
    }
    fn emit_ext_rows(&mut self, constraint_idx: usize, _rows: RowDomain, e: IrExpr) {
        let root = self.flatten(&e);
        self.ir.emit(constraint_idx, root);
        self.degrees.push((constraint_idx, e.degree()));
    }
}
