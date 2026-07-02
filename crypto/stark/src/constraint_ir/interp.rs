//! CPU interpreter for a captured [`ConstraintProgram`].
//!
//! A single forward pass over the topologically ordered nodes evaluates each
//! node into a [`Value`] (base [`Dim::Base`] or extension [`Dim::Ext`]), reusing
//! the real `FieldElement` arithmetic so per-op results are bit-identical to the
//! compiled constraint path. Mixed-dimension ops auto-embed the base operand
//! into the extension, mirroring the field tower's `F: IsSubFieldOf<E>`
//! arithmetic.
//!
//! [`eval_program`] / [`eval_program_verifier`] are the full entry points,
//! matching `AIR::compute_transition_prover` / `AIR::compute_transition`
//! respectively. [`eval_program_base`] is the minimal entry point (single root,
//! main-only, base-field result) kept for the per-constraint diff test.
//!
//! Every entry point is generic over the field tower `<F: IsSubFieldOf<E>, E>`;
//! for the Goldilocks tower these monomorphize to the same arithmetic the
//! compiled folder emits.

use math::field::element::FieldElement;
use math::field::traits::{IsField, IsSubFieldOf};

use super::ir::{ConstraintProgram, Dim, Op};
use crate::table::TableView;
use crate::traits::TransitionEvaluationContext;

/// A node's computed value: base field ([`Dim::Base`]) or extension
/// ([`Dim::Ext`]).
///
/// `Clone`, not `Copy` — `Copy` is not provable for a generic `FieldElement<F>`.
/// For the Goldilocks tower these clones compile to register copies.
#[derive(Clone, Debug)]
enum Value<F: IsField, E: IsField> {
    Base(FieldElement<F>),
    Ext(FieldElement<E>),
}

impl<F: IsSubFieldOf<E>, E: IsField> Value<F, E> {
    /// Promote to the extension field, embedding a base value if needed.
    fn to_ext(&self) -> FieldElement<E> {
        match self {
            Value::Base(x) => x.clone().to_extension::<E>(),
            Value::Ext(x) => x.clone(),
        }
    }

    fn as_base(&self) -> FieldElement<F> {
        match self {
            Value::Base(x) => x.clone(),
            Value::Ext(_) => {
                panic!("expected a base value but found an extension value")
            }
        }
    }
}

/// Shared forward pass: evaluate every node, then return the value array.
/// `resolve_var` resolves `Op::Var` leaves; `resolve_periodic` resolves
/// `Op::Periodic`; the remaining uniforms are read from field-agnostic closures
/// so prover/verifier share this one walk.
#[allow(clippy::too_many_arguments)]
fn run<F, E, FVar, FPeriodic, FChallenge, FAlpha, FOffset>(
    prog: &ConstraintProgram<F, E>,
    resolve_var: FVar,
    resolve_periodic: FPeriodic,
    resolve_challenge: FChallenge,
    resolve_alpha: FAlpha,
    resolve_offset: FOffset,
) -> Vec<Value<F, E>>
where
    F: IsSubFieldOf<E>,
    E: IsField,
    FVar: Fn(bool, u8, u8, u16) -> Value<F, E>,
    FPeriodic: Fn(u16) -> Value<F, E>,
    FChallenge: Fn(u16) -> FieldElement<E>,
    FAlpha: Fn(u16) -> FieldElement<E>,
    FOffset: Fn() -> FieldElement<E>,
{
    let mut values: Vec<Value<F, E>> = Vec::with_capacity(prog.nodes.len());

    for (i, op) in prog.nodes.iter().enumerate() {
        let v = match *op {
            Op::ConstBase(idx) => Value::Base(prog.base_consts[idx as usize].clone()),
            Op::ConstExt(idx) => Value::Ext(prog.ext_consts[idx as usize].clone()),
            Op::Var {
                main,
                offset,
                row,
                col,
            } => resolve_var(main, offset, row, col),
            Op::Periodic { idx } => resolve_periodic(idx),
            Op::RapChallenge { idx } => Value::Ext(resolve_challenge(idx)),
            Op::AlphaPow { idx } => Value::Ext(resolve_alpha(idx)),
            Op::TableOffset => Value::Ext(resolve_offset()),
            Op::Add(a, b) => binop(&values, a, b, prog.dims[i], |x, y| x + y, |x, y| x + y),
            Op::Sub(a, b) => binop(&values, a, b, prog.dims[i], |x, y| x - y, |x, y| x - y),
            Op::Mul(a, b) => binop(&values, a, b, prog.dims[i], |x, y| x * y, |x, y| x * y),
            Op::Neg(a) => match (&values[a as usize], prog.dims[i]) {
                (Value::Base(x), Dim::Base) => Value::Base(-x),
                (val, Dim::Ext) => Value::Ext(-val.to_ext()),
                // A base value tagged extension (or vice versa) is a dim
                // mismatch; keep it in the extension to stay well-typed.
                (Value::Ext(x), Dim::Base) => Value::Ext(-x.clone()),
            },
            Op::Embed(a) => Value::Ext(values[a as usize].to_ext()),
        };
        values.push(v);
    }

    values
}

/// Apply a binary op, auto-embedding to the extension field when the result
/// dimension is [`Dim::Ext`] (or either operand is already extension).
#[inline]
fn binop<F, E>(
    values: &[Value<F, E>],
    a: u32,
    b: u32,
    result_dim: Dim,
    base_op: impl Fn(FieldElement<F>, FieldElement<F>) -> FieldElement<F>,
    ext_op: impl Fn(FieldElement<E>, FieldElement<E>) -> FieldElement<E>,
) -> Value<F, E>
where
    F: IsSubFieldOf<E>,
    E: IsField,
{
    let va = &values[a as usize];
    let vb = &values[b as usize];
    match (va, vb, result_dim) {
        (Value::Base(x), Value::Base(y), Dim::Base) => Value::Base(base_op(x.clone(), y.clone())),
        _ => Value::Ext(ext_op(va.to_ext(), vb.to_ext())),
    }
}

/// Evaluate one constraint's root over a base-field main row.
///
/// `main_row[col]` resolves `Var { main: true, col, .. }` leaves. The minimal
/// algebraic constraint set only reads main columns at offset 0, row 0 and
/// returns a base-field value. `constraint_idx` selects which root to read.
///
/// Kept for the per-constraint diff test; [`eval_program`] is the full prover
/// entry point.
pub fn eval_program_base<F, E>(
    prog: &ConstraintProgram<F, E>,
    constraint_idx: usize,
    main_row: &[FieldElement<F>],
) -> FieldElement<F>
where
    F: IsSubFieldOf<E>,
    E: IsField,
{
    let values = run(
        prog,
        |main, _offset, row, col| {
            assert!(main, "aux leaves are not part of the minimal algebraic set");
            assert_eq!(row, 0, "minimal set reads row 0 only");
            Value::Base(main_row[col as usize].clone())
        },
        |_idx| panic!("periodic leaves are not part of the minimal algebraic set"),
        |_idx| panic!("challenge leaves are not part of the minimal algebraic set"),
        |_idx| panic!("alpha_power leaves are not part of the minimal algebraic set"),
        || panic!("table_offset leaves are not part of the minimal algebraic set"),
    );
    let root = prog.roots[constraint_idx];
    values[root as usize].as_base()
}

/// Full prover entry point: evaluate every constraint in `prog` against `ctx`
/// (must be [`TransitionEvaluationContext::Prover`]), writing base-field
/// ([`Dim::Base`]-rooted) constraints into `base_evals` and extension-field
/// ([`Dim::Ext`]-rooted) constraints into `ext_evals[prog.num_base..]` — the
/// same contract as `AIR::compute_transition_prover`.
pub fn eval_program<F, E>(
    prog: &ConstraintProgram<F, E>,
    ctx: &TransitionEvaluationContext<F, E>,
    base_evals: &mut [FieldElement<F>],
    ext_evals: &mut [FieldElement<E>],
) where
    F: IsSubFieldOf<E>,
    E: IsField,
{
    let TransitionEvaluationContext::Prover {
        frame,
        periodic_values,
        rap_challenges,
        logup_alpha_powers,
        logup_table_offset,
        ..
    } = ctx
    else {
        unreachable!("eval_program called with a Verifier context");
    };

    let values = run(
        prog,
        |main, offset, row, col| {
            let step: &TableView<F, E> = frame.get_evaluation_step(offset as usize);
            debug_assert_eq!(row, 0, "tables read row 0 of each frame step");
            if main {
                Value::Base(step.get_main_evaluation_element(0, col as usize).clone())
            } else {
                Value::Ext(step.get_aux_evaluation_element(0, col as usize).clone())
            }
        },
        |idx| Value::Base(periodic_values[idx as usize].clone()),
        |idx| rap_challenges[idx as usize].clone(),
        |idx| logup_alpha_powers[idx as usize].clone(),
        || (*logup_table_offset).clone(),
    );

    for (c, &root) in prog.roots.iter().enumerate() {
        let v = &values[root as usize];
        if c < prog.num_base {
            base_evals[c] = v.as_base();
        } else {
            ext_evals[c] = v.to_ext();
        }
    }
}

/// Full verifier entry point: evaluate every constraint in `prog` against `ctx`
/// (must be [`TransitionEvaluationContext::Verifier`]) at the out-of-domain
/// point, writing every constraint (base or LogUp) into `ext_evals` — the same
/// contract as `AIR::compute_transition`. The verifier frame holds only
/// extension-field elements, so base-rooted constraints are embedded into the
/// extension on write.
pub fn eval_program_verifier<F, E>(
    prog: &ConstraintProgram<F, E>,
    ctx: &TransitionEvaluationContext<F, E>,
    ext_evals: &mut [FieldElement<E>],
) where
    F: IsSubFieldOf<E>,
    E: IsField,
{
    let TransitionEvaluationContext::Verifier {
        frame,
        periodic_values,
        rap_challenges,
        logup_alpha_powers,
        logup_table_offset,
        ..
    } = ctx
    else {
        unreachable!("eval_program_verifier called with a Prover context");
    };

    let values = run(
        prog,
        |main, offset, row, col| {
            let step: &TableView<E, E> = frame.get_evaluation_step(offset as usize);
            debug_assert_eq!(row, 0, "tables read row 0 of each frame step");
            if main {
                Value::Ext(step.get_main_evaluation_element(0, col as usize).clone())
            } else {
                Value::Ext(step.get_aux_evaluation_element(0, col as usize).clone())
            }
        },
        |idx| Value::Ext(periodic_values[idx as usize].clone()),
        |idx| rap_challenges[idx as usize].clone(),
        |idx| logup_alpha_powers[idx as usize].clone(),
        || (*logup_table_offset).clone(),
    );

    for (c, &root) in prog.roots.iter().enumerate() {
        ext_evals[c] = values[root as usize].to_ext();
    }
}
