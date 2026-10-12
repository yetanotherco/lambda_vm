//! `IrShape::combine` as a machine leg — the zerocheck rule's value.
//!
//! The mirror is `stark::multilinear_air::IrShape::combine`
//! (`multilinear_air.rs:898-916`): the table's constraint DAG run over the
//! factor values, then one root per constraint batched by the powers of `beta`,
//! each root optionally gated by its selector.
//!
//! ```text
//!     combine(β, v) = Σ_i β^i · node[root_i] · (selector_i ? v[slot_i] : 1)
//! ```
//!
//! # ⛔ Why this is not `emit_program` pointed at `IrShape::program`
//!
//! `IrShape::program(&betas, weight)` builds the same expression as a
//! `Program`, and a `Program` is what [`super::whir_program::emit_program`]
//! already emits and is gated on. It cannot be used here, and the reason is the
//! whole shape of this leg: `Builder::weighted_sum` interns each batching
//! coefficient as a `Fixed` STEP (`program.rs:304-316`), and those coefficients
//! are the powers of `beta` — a challenge the verifier draws from the
//! transcript. `emit_program` interns every `Fixed` as an `LFM_CONST`, which is
//! right for the AIR's own constants and wrong for a value that does not exist
//! until the proof is being read.
//!
//! So the DAG half goes through `emit_steps` — the same walk, the same
//! constants, already gated — and the batching half is emitted here with the
//! beta powers as WIRES. The accessors that make the structure readable
//! (`steps_as_ops`, `root_steps`, `selector_of_root`) were added to `IrShape`
//! for exactly this and nothing else.
//!
//! # The factor ordering, and what `values` must hold
//!
//! `Var(i)` reads factor `i`, and a selector names a factor slot, so `values` is
//! the woven factor vector `constraint_argument::verify_core` builds: the
//! committed factors in kind order, then the public ones, then the two weight
//! tables (`multilinear_table.rs:616-618`). The rule's own value is this times
//! the weight at `weights(kinds.len()).0`, which the CALLER applies —
//! `combine` does not, and the gate is on `combine`.
//!
//! # ⚠ One commutative reordering, stated rather than hidden
//!
//! The host multiplies the node by `β^i` and then by the selector; this
//! multiplies by the selector and then folds `β^i` in with a `MulAdd`. Same
//! value, same row count — a field multiplication is commutative and the fold
//! is one row either way. It is a rewrite of the host's expression, not a
//! difference, and it is written down because "the operands are in the other
//! order" is exactly what a reader should be able to check rather than wonder
//! about.

use stark::multilinear_air::IrShape;

use crate::tables::types::{FEE, GoldilocksExtension, GoldilocksField};

use super::builder::{Ext, LfmBuilder};
use super::whir_program::{emit_steps, steps_rows};

type Shape = IrShape<GoldilocksField, GoldilocksExtension>;

/// INSTRUCTIONS [`emit_combine`] emits, constants included.
///
/// Every term by the shape it comes from:
///
/// - the DAG's own rows, `steps_rows` — one `LFM_XALU` per operation node, one
///   `LFM_CONST` per distinct `Fixed` value, and the zero a `Neg` subtracts
///   from;
/// - one `Mul` per root that carries a SELECTOR, and none for a root that
///   applies on every step;
/// - one row per root to batch it: a `Mul` for the first, which opens the
///   accumulator, and a `MulAdd` for every later one.
///
/// A shape with no roots emits one `LFM_CONST` zero and nothing else, which is
/// what `combine`'s fold from zero comes to.
pub fn combine_rows(shape: &Shape) -> usize {
    let roots = shape.root_steps().len();
    if roots == 0 {
        return 1;
    }
    let selectors = shape
        .selector_of_root()
        .iter()
        .filter(|selector| selector.is_some())
        .count();
    steps_rows(&shape.steps_as_ops()) + selectors + roots
}

/// ★ `IrShape::combine(beta_powers, values)`, emitted with the beta powers as
/// runtime wires.
///
/// Panics if the number of beta powers is not the number of roots, or if a step
/// or a selector reads a factor the caller did not supply — every one of those
/// is an emit-time length, so a mismatch is a bug in the caller rather than a
/// condition to carry at runtime. The host would return
/// `VariableCountMismatch` for the first and index out of bounds for the rest.
pub fn emit_combine(b: &mut LfmBuilder, shape: &Shape, betas: &[Ext], values: &[Ext]) -> Ext {
    assert_eq!(
        betas.len(),
        shape.num_roots(),
        "one beta power per batched root — the host's `beta_powers` length check"
    );
    let ops = shape.steps_as_ops();
    let nodes = emit_steps(b, &ops, values);

    let mut batched: Option<Ext> = None;
    for ((&root, &beta), selector) in shape
        .root_steps()
        .iter()
        .zip(betas)
        .zip(shape.selector_of_root())
    {
        let mut term = nodes[root as usize];
        if let Some(slot) = selector {
            let gate = *values.get(*slot).unwrap_or_else(|| {
                panic!(
                    "a root's selector reads factor {slot}, but only {} were supplied",
                    values.len()
                )
            });
            term = b.emul(term, gate);
        }
        batched = Some(match batched {
            None => b.emul(beta, term),
            Some(accumulated) => b.emul_add(beta, term, accumulated),
        });
    }

    batched.unwrap_or_else(|| b.ext_const(&FEE::zero()))
}
