//! CPU interpreter for a captured [`ConstraintProgram`].
//!
//! A single forward pass over the topologically ordered nodes evaluates each
//! node into a [`Value`] (base `D1` or extension `D3`), reusing the real
//! `FieldElement` arithmetic so per-op results are bit-identical to the boxed
//! constraint path. Mixed-dimension ops auto-embed the `D1` operand into `D3`,
//! mirroring the field tower's `F: IsSubFieldOf<E>` arithmetic.
//!
//! [`eval_program`] / [`eval_program_verifier`] are the full entry points,
//! matching `AIR::compute_transition_prover` / `AIR::compute_transition`
//! respectively. [`eval_program_base`] is the minimal Phase-0 entry point
//! (single root, main-only, base-field result) kept for the original spike
//! diff test.

use math::field::element::FieldElement;
use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField as GoldilocksExtension;
use math::field::goldilocks::GoldilocksField;

use super::ir::{ConstraintProgram, Dim, Op};
use crate::table::TableView;
use crate::traits::TransitionEvaluationContext;

type Fp = FieldElement<GoldilocksField>;
type Fp3 = FieldElement<GoldilocksExtension>;

/// A node's computed value: base field (`D1`) or degree-3 extension (`D3`).
#[derive(Clone, Copy, Debug)]
enum Value {
    D1(Fp),
    D3(Fp3),
}

impl Value {
    /// Promote to the extension field, embedding a base value if needed.
    fn to_ext(self) -> Fp3 {
        match self {
            Value::D1(x) => x.to_extension::<GoldilocksExtension>(),
            Value::D3(x) => x,
        }
    }

    fn as_base(self) -> Fp {
        match self {
            Value::D1(x) => x,
            Value::D3(_) => {
                panic!("expected a base (D1) value but found an extension (D3) value")
            }
        }
    }
}

/// Shared forward pass: evaluate every node, then return the value array.
/// `resolve_var` resolves `Op::Var` leaves; `resolve_periodic` resolves
/// `Op::Periodic`; the rest of the uniforms are read directly from `inputs`-
/// agnostic closures so prover/verifier share this one walk.
#[allow(clippy::too_many_arguments)]
fn run<FVar, FPeriodic, FChallenge, FAlpha, FOffset>(
    prog: &ConstraintProgram,
    resolve_var: FVar,
    resolve_periodic: FPeriodic,
    resolve_challenge: FChallenge,
    resolve_alpha: FAlpha,
    resolve_offset: FOffset,
) -> Vec<Value>
where
    FVar: Fn(bool, u8, u8, u16) -> Value,
    FPeriodic: Fn(u16) -> Value,
    FChallenge: Fn(u16) -> Fp3,
    FAlpha: Fn(u16) -> Fp3,
    FOffset: Fn() -> Fp3,
{
    let mut values: Vec<Value> = Vec::with_capacity(prog.nodes.len());

    for (i, op) in prog.nodes.iter().enumerate() {
        let v = match *op {
            Op::Const1(c) => Value::D1(Fp::from(c)),
            Op::Const3([c0, c1, c2]) => {
                Value::D3(Fp3::from_raw([Fp::from(c0), Fp::from(c1), Fp::from(c2)]))
            }
            Op::Var {
                main,
                offset,
                row,
                col,
            } => resolve_var(main, offset, row, col),
            Op::Periodic { idx } => resolve_periodic(idx),
            Op::RapChallenge { idx } => Value::D3(resolve_challenge(idx)),
            Op::AlphaPow { idx } => Value::D3(resolve_alpha(idx)),
            Op::TableOffset => Value::D3(resolve_offset()),
            Op::Add(a, b) => binop(&values, a, b, prog.dims[i], |x, y| x + y, |x, y| x + y),
            Op::Sub(a, b) => binop(&values, a, b, prog.dims[i], |x, y| x - y, |x, y| x - y),
            Op::Mul(a, b) => binop(&values, a, b, prog.dims[i], |x, y| x * y, |x, y| x * y),
            Op::Neg(a) => match (values[a as usize], prog.dims[i]) {
                (Value::D1(x), Dim::D1) => Value::D1(-x),
                (val, Dim::D3) => Value::D3(-val.to_ext()),
                (Value::D3(x), Dim::D1) => Value::D3(-x), // dim mismatch, keep ext
            },
            Op::Embed(a) => Value::D3(values[a as usize].to_ext()),
        };
        values.push(v);
    }

    values
}

/// Apply a binary op, auto-embedding to the extension field when the result
/// dimension is `D3` (or either operand is already `D3`).
#[inline]
fn binop(
    values: &[Value],
    a: u32,
    b: u32,
    result_dim: Dim,
    base_op: impl Fn(Fp, Fp) -> Fp,
    ext_op: impl Fn(Fp3, Fp3) -> Fp3,
) -> Value {
    let va = values[a as usize];
    let vb = values[b as usize];
    match (va, vb, result_dim) {
        (Value::D1(x), Value::D1(y), Dim::D1) => Value::D1(base_op(x, y)),
        _ => Value::D3(ext_op(va.to_ext(), vb.to_ext())),
    }
}

/// Evaluate one constraint's root over a base-field main row.
///
/// `main_row[col]` resolves `Var { main: true, col, .. }` leaves. The minimal
/// algebraic constraint set only reads main columns at offset 0, row 0 and
/// returns a base-field (`D1`) value, so this returns a `FieldElement<F>`.
/// `constraint_idx` selects which root to read (a single-constraint capture
/// from the Phase-0 diff test always uses the constraint's own
/// `constraint_idx`, which `IrBuilder::emit` now indexes `roots` by).
///
/// Kept for the Phase-0 diff test (`prover/src/tests/constraint_ir_tests.rs`);
/// [`eval_program`] is the full prover entry point.
pub fn eval_program_base(prog: &ConstraintProgram, constraint_idx: usize, main_row: &[Fp]) -> Fp {
    let values = run(
        prog,
        |main, _offset, row, col| {
            assert!(main, "aux leaves are not part of the minimal algebraic set");
            assert_eq!(row, 0, "minimal set reads row 0 only");
            Value::D1(main_row[col as usize])
        },
        |_idx| panic!("periodic leaves are not part of the minimal algebraic set"),
        |_idx| panic!("challenge leaves are not part of the minimal algebraic set"),
        |_idx| panic!("alpha_power leaves are not part of the minimal algebraic set"),
        || panic!("table_offset leaves are not part of the minimal algebraic set"),
    );
    let root = prog.roots[constraint_idx];
    values[root as usize].as_base()
}

/// Full prover entry point: evaluate every constraint in `prog` against
/// `ctx` (must be [`TransitionEvaluationContext::Prover`]), writing base-field
/// (`D1`-rooted) constraints into `base_evals` and extension-field
/// (`D3`-rooted) constraints into `ext_evals[prog.num_base..]` — the same
/// contract as `AIR::compute_transition_prover`.
pub fn eval_program(
    prog: &ConstraintProgram,
    ctx: &TransitionEvaluationContext<GoldilocksField, GoldilocksExtension>,
    base_evals: &mut [FieldElement<GoldilocksField>],
    ext_evals: &mut [FieldElement<GoldilocksExtension>],
) {
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
            let step: &TableView<GoldilocksField, GoldilocksExtension> =
                frame.get_evaluation_step(offset as usize);
            debug_assert_eq!(row, 0, "tables read row 0 of each frame step");
            if main {
                Value::D1(*step.get_main_evaluation_element(0, col as usize))
            } else {
                Value::D3(*step.get_aux_evaluation_element(0, col as usize))
            }
        },
        |idx| Value::D1(periodic_values[idx as usize]),
        |idx| rap_challenges[idx as usize],
        |idx| logup_alpha_powers[idx as usize],
        || *(*logup_table_offset),
    );

    for (c, &root) in prog.roots.iter().enumerate() {
        let v = values[root as usize];
        if c < prog.num_base {
            base_evals[c] = v.as_base();
        } else {
            ext_evals[c] = v.to_ext();
        }
    }
}

/// Full verifier entry point: evaluate every constraint in `prog` against
/// `ctx` (must be [`TransitionEvaluationContext::Verifier`]) at the
/// out-of-domain point, writing every constraint (base or LogUp) into
/// `ext_evals` — the same contract as `AIR::compute_transition`. The verifier
/// frame holds only extension-field elements, so `D1`-rooted constraints are
/// embedded into `D3` on write (mirrors `evaluate(..).to_extension()` in
/// [`crate::constraints::transition::TransitionConstraintAdapter`]).
pub fn eval_program_verifier(
    prog: &ConstraintProgram,
    ctx: &TransitionEvaluationContext<GoldilocksField, GoldilocksExtension>,
    ext_evals: &mut [FieldElement<GoldilocksExtension>],
) {
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
            let step: &TableView<GoldilocksExtension, GoldilocksExtension> =
                frame.get_evaluation_step(offset as usize);
            debug_assert_eq!(row, 0, "tables read row 0 of each frame step");
            if main {
                Value::D3(*step.get_main_evaluation_element(0, col as usize))
            } else {
                Value::D3(*step.get_aux_evaluation_element(0, col as usize))
            }
        },
        |idx| Value::D3(periodic_values[idx as usize]),
        |idx| rap_challenges[idx as usize],
        |idx| logup_alpha_powers[idx as usize],
        || *(*logup_table_offset),
    );

    for (c, &root) in prog.roots.iter().enumerate() {
        ext_evals[c] = values[root as usize].to_ext();
    }
}
