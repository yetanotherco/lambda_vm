//! Generic-`Field`/`FieldExtension` → concrete-Goldilocks TypeId seam.
//!
//! `eval_program`/`eval_program_verifier` are concretely typed to
//! `GoldilocksField`/`Degree3GoldilocksExtensionField` (the IR is single-field,
//! see `crate::constraint_ir`), but the prover/verifier's evaluation loops
//! (`crate::constraints::evaluator::ConstraintEvaluator`,
//! `crate::verifier::verify`) are generic over `Field: IsSubFieldOf<FieldExtension>`.
//! `try_eval_program_prover`/`try_eval_program_verifier` bridge the two: a
//! `TypeId` check establishes `Field == GoldilocksField` and
//! `FieldExtension == Degree3GoldilocksExtensionField` exactly, after which a
//! `&TransitionEvaluationContext<Field, FieldExtension>` is reinterpreted as
//! `&TransitionEvaluationContext<GoldilocksField, Degree3GoldilocksExtensionField>`
//! (same layout — only the type parameters differ, and the check pins them to
//! be the same concrete type), mirroring the seam already used by
//! `crate::gpu_lde` (`TypeId::of::<F>()` guards + `transmute_copy`).
//!
//! Returns `false` (no-op on the caller's buffers) when the TypeId check
//! fails, so callers fall back to the boxed path unconditionally outside the
//! lambda_vm single-field (Goldilocks base + degree-3 extension) setup.

use std::any::TypeId;
use std::mem::transmute;

use math::field::element::FieldElement;
use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField as GoldilocksExtension;
use math::field::goldilocks::GoldilocksField;
use math::field::traits::{IsFFTField, IsField, IsSubFieldOf};

use super::interp::{eval_program, eval_program_verifier};
use super::ir::ConstraintProgram;
use crate::traits::TransitionEvaluationContext;

/// `true` iff `Field == GoldilocksField` and `FieldExtension == Degree3GoldilocksExtensionField`.
#[inline]
fn is_goldilocks_tower<Field: 'static, FieldExtension: 'static>() -> bool {
    TypeId::of::<Field>() == TypeId::of::<GoldilocksField>()
        && TypeId::of::<FieldExtension>() == TypeId::of::<GoldilocksExtension>()
}

/// Prover-side bridge: interpret `prog` via [`eval_program`] in place of the
/// boxed `air.compute_transition_prover(...)` call, writing the same
/// `base_evals`/`ext_evals` contract. Returns `true` if it ran (the type
/// tower matched Goldilocks); `false` otherwise (caller should fall back).
pub fn try_eval_program_prover<Field, FieldExtension>(
    prog: &ConstraintProgram,
    ctx: &TransitionEvaluationContext<Field, FieldExtension>,
    base_evals: &mut [FieldElement<Field>],
    ext_evals: &mut [FieldElement<FieldExtension>],
) -> bool
where
    Field: IsSubFieldOf<FieldExtension> + IsFFTField + Send + Sync + 'static,
    FieldExtension: IsField + Send + Sync + 'static,
{
    if !is_goldilocks_tower::<Field, FieldExtension>() {
        return false;
    }
    // SAFETY: the TypeId check above establishes `Field == GoldilocksField`
    // and `FieldExtension == Degree3GoldilocksExtensionField` exactly, so
    // `TransitionEvaluationContext<Field, FieldExtension>` and
    // `[FieldElement<Field>]`/`[FieldElement<FieldExtension>]` have the same
    // layout as their Goldilocks-concrete counterparts (same generic struct,
    // same — now proven identical — type arguments). Mirrors the
    // `transmute_copy` seam in `crate::gpu_lde`.
    let ctx_gl: &TransitionEvaluationContext<GoldilocksField, GoldilocksExtension> =
        unsafe { transmute(ctx) };
    let base_gl: &mut [FieldElement<GoldilocksField>] = unsafe { transmute(base_evals) };
    let ext_gl: &mut [FieldElement<GoldilocksExtension>] = unsafe { transmute(ext_evals) };
    eval_program(prog, ctx_gl, base_gl, ext_gl);
    true
}

/// Verifier-side bridge: interpret `prog` via [`eval_program_verifier`] in
/// place of the boxed `air.compute_transition(...)` call, writing the same
/// `ext_evals` contract. Returns `true` if it ran; `false` otherwise.
///
/// At the OOD point the verifier's `TransitionEvaluationContext` is always
/// `<E, E>` (`Field` and `FieldExtension` are the same type — see
/// `TransitionEvaluationContext::Verifier`, which has no base-field data), so
/// unlike [`try_eval_program_prover`] this only needs `Field: IsSubFieldOf<FieldExtension>`
/// (reflexive for any field) and not `IsFFTField`.
pub fn try_eval_program_verifier<Field, FieldExtension>(
    prog: &ConstraintProgram,
    ctx: &TransitionEvaluationContext<Field, FieldExtension>,
    ext_evals: &mut [FieldElement<FieldExtension>],
) -> bool
where
    Field: IsSubFieldOf<FieldExtension> + Send + Sync + 'static,
    FieldExtension: IsField + Send + Sync + 'static,
{
    if !is_goldilocks_tower::<Field, FieldExtension>() {
        return false;
    }
    // SAFETY: see `try_eval_program_prover`.
    let ctx_gl: &TransitionEvaluationContext<GoldilocksField, GoldilocksExtension> =
        unsafe { transmute(ctx) };
    let ext_gl: &mut [FieldElement<GoldilocksExtension>] = unsafe { transmute(ext_evals) };
    eval_program_verifier(prog, ctx_gl, ext_gl);
    true
}
