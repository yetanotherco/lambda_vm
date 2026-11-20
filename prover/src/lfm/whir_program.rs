//! The constraint DAG as a machine leg: `multilinear::program::Program`,
//! emitted.
//!
//! Every table's constraint argument reduces, at the end, to evaluating one
//! straight-line program at the point its sumcheck left — `Program::eval`
//! (`crypto/multilinear/src/program.rs:52-71`). It is walked ONCE per table,
//! not once per round and not once per constraint, which is why the sizing
//! note's census puts the whole term at a few percent of a wrap.
//!
//! ⚠ **This is not `constraints::emit_analyzed` re-aimed**, which is what my
//! sizing note §5 recommended. That emitter consumes an `Analysis` over the
//! UNIVARIATE `Op`, whose `Var` carries `{main, offset, col}` and which arrives
//! with liveness, folding and `MulAdd` fusion already computed. The multilinear
//! `Program` is a different and much smaller type: six variants over a flat
//! `steps` vector with a `root`. So this walks it directly.
//!
//! ⚠ **One row per operation node, and the number that costs.** The univariate
//! emitter fuses a `Mul` feeding a single-use `Add` into one `MulAdd` and is
//! MEASURED at 0.75–0.87 rows per IR node with folding and DCE. This sits at
//! 1.0. The term is small — the note prices the whole constraint DAG at ~70k
//! rows for all 34 sub-proofs plus ~15k for the two bus programs, 1.6% of
//! today's wrap — so the fusion is worth roughly 0.3% of an epoch. It is left
//! as a named lever rather than taken here, because its closed form would have
//! to count fusions the same way the emitter does, which turns the row pin into
//! a round trip on exactly the term the optimisation moves. Doing it properly
//! means an independent fusion count, which is a leg of its own.

use multilinear::program::{Op, Program};

use crate::tables::types::{FEE, GoldilocksExtension};

use super::builder::{Ext, LfmBuilder};

/// INSTRUCTIONS [`emit_program`] emits for `program`.
///
/// One `LFM_XALU` row per operation node — `Add`, `Sub`, `Mul` and `Neg` — plus
/// one `LFM_CONST` row per DISTINCT `Fixed` value and one more for the zero a
/// `Neg` subtracts from, when the program has any. `Var` costs nothing: it is
/// already a wire.
///
/// Countable straight off `steps()` without simulating the walk, which is what
/// makes the pin independent of the emitter rather than a restatement of it.
pub fn program_rows(program: &Program<GoldilocksExtension>) -> usize {
    steps_rows(program.steps())
}

/// The same count over a DAG's steps alone.
///
/// Split out because a caller that reads SEVERAL roots out of one DAG — the
/// zerocheck rule batches one per constraint root — has no single `Program` to
/// hand over, and counting it a second way would be a second form to keep in
/// step.
pub fn steps_rows(steps: &[Op<GoldilocksExtension>]) -> usize {
    let mut operations = 0;
    let mut negates = false;
    let mut constants: Vec<FEE> = Vec::new();
    for step in steps {
        match step {
            Op::Fixed(value) => {
                if !constants.contains(value) {
                    constants.push(*value);
                }
            }
            Op::Var(_) => {}
            Op::Add(_, _) | Op::Sub(_, _) | Op::Mul(_, _) => operations += 1,
            Op::Neg(_) => {
                operations += 1;
                negates = true;
            }
        }
    }
    // The zero a `Neg` subtracts from is interned like any other constant, so a
    // program that already carries `Fixed(0)` would pay for it once.
    //
    // ⚠ MEASURED: that guard is unreachable with the host's own `Builder`, which
    // folds identities at CONSTRUCTION — `sub(x, zero)` returns `x` and
    // `add(x, zero)` returns `x`, so a `Fixed(0)` never becomes a step. It is
    // kept because the form should be right about the type rather than about
    // the one producer, and it is written down as unexercised rather than left
    // looking like a tested branch.
    //
    // ⚠ The same measurement retires a mutation this leg was pre-registered
    // with. Emitting `Neg` as a multiply by an interned `−1` instead of a
    // subtraction from an interned zero is EQUIVALENT in value and in cost —
    // one row against one constant either way — on every program this builder
    // can produce, because the only shape that could tell them apart is one
    // holding `Fixed(0)`, and it folds away. It is not a mutation; it is a
    // choice with no observable difference.
    let zero_for_neg = usize::from(negates && !constants.contains(&FEE::zero()));
    operations + constants.len() + zero_for_neg
}

/// ★ `Program::eval`, emitted.
///
/// The walk is the host's, step for step: each step's value is kept as a wire,
/// `Var(i)` reads `values[i]`, and the answer is the wire at `root()`. `Neg` is
/// `0 − a` rather than a multiply by `−1`, so it costs the same one row without
/// a second constant.
///
/// Panics if a step reads a slot the caller did not supply — the host indexes
/// `values` directly and would panic too, but here the lengths are emit-time
/// constants, so a mismatch is a bug in the caller rather than a condition to
/// carry at runtime.
pub fn emit_program(
    b: &mut LfmBuilder,
    program: &Program<GoldilocksExtension>,
    values: &[Ext],
) -> Ext {
    let wires = emit_steps(b, program.steps(), values);
    wires[program.root() as usize]
}

/// Every step's value as a wire, in step order.
///
/// [`emit_program`] is this and then one index. The zerocheck rule needs the
/// whole vector, because it reads one node per constraint root out of a single
/// DAG and batches them.
pub fn emit_steps(
    b: &mut LfmBuilder,
    steps: &[Op<GoldilocksExtension>],
    values: &[Ext],
) -> Vec<Ext> {
    let mut wires: Vec<Ext> = Vec::with_capacity(steps.len());
    let mut zero: Option<Ext> = None;
    for (index, step) in steps.iter().enumerate() {
        let wire = match *step {
            Op::Fixed(value) => b.ext_const(&value),
            Op::Var(slot) => *values.get(slot as usize).unwrap_or_else(|| {
                panic!(
                    "step {index} reads factor {slot}, but only {} were supplied",
                    values.len()
                )
            }),
            Op::Add(a, c) => b.eadd(wires[a as usize], wires[c as usize]),
            Op::Sub(a, c) => b.esub(wires[a as usize], wires[c as usize]),
            Op::Mul(a, c) => b.emul(wires[a as usize], wires[c as usize]),
            Op::Neg(a) => {
                let unit = *zero.get_or_insert_with(|| b.ext_const(&FEE::zero()));
                b.esub(unit, wires[a as usize])
            }
        };
        wires.push(wire);
    }
    wires
}
