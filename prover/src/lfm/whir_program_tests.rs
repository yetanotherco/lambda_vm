//! Gates for the constraint-DAG leg.

use multilinear::program::{Builder, Program};

use crate::tables::types::{FE, FEE, GoldilocksExtension};

use super::builder::{Ext, LfmBuilder};
use super::compiler::{LfmProgram, compile};
use super::executor::execute;
use super::validator::validate;
use super::whir_program::{emit_program, program_rows};
use super::word::{ext_word, word_as_ext};

fn fee(v: u64) -> FEE {
    FEE::new([
        FE::from(v.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 9),
        FE::from(v ^ 0x5A5A),
        FE::from(v.wrapping_add(11)),
    ])
}

/// One shape under test: a name, the program, and how many factors it reads.
struct Shape {
    name: &'static str,
    program: Program<GoldilocksExtension>,
    factors: usize,
}

/// Programs with the structure a real constraint DAG has: shared
/// subexpressions, a constant that appears twice, a negation, a degree
/// stack, and one that `simplify` has been through.
fn shapes() -> Vec<Shape> {
    let mut out = Vec::new();

    // A single factor read straight to the root: no operation at all.
    let mut b = Builder::<GoldilocksExtension>::new();
    let v = b.var(0);
    out.push(Shape {
        name: "one factor, no work",
        program: b.finish(v).expect("root is a step"),
        factors: 1,
    });

    // A constant used twice, a negation, and a shared subexpression.
    let mut b = Builder::<GoldilocksExtension>::new();
    let x = b.var(0);
    let y = b.var(1);
    let k = b.fixed(fee(7));
    let k_again = b.fixed(fee(7));
    let shared = b.mul(x, y);
    let left = b.add(shared, k);
    let right = b.sub(shared, k_again);
    let negated = b.neg(right);
    let root = b.mul(left, negated);
    out.push(Shape {
        name: "shared subexpression, repeated constant, a negation",
        program: b.finish(root).expect("root is a step"),
        factors: 2,
    });

    // A degree stack: the shape a high-degree rule takes.
    let mut b = Builder::<GoldilocksExtension>::new();
    let mut acc = b.var(0);
    for slot in 1..6 {
        let next = b.var(slot);
        let scaled = b.mul(acc, next);
        let shifted = b.fixed(fee(slot as u64 * 13));
        acc = b.add(scaled, shifted);
    }
    let root = b.neg(acc);
    out.push(Shape {
        name: "a degree stack with six factors",
        program: b.finish(root).expect("root is a step"),
        factors: 6,
    });

    // ★ An attempt to build a program holding `Fixed(0)` beside a `Neg` — the
    // only shape that could tell `0 − a` from `(−1)·a`.
    //
    // ⚠ It cannot be built: the printed step count is TWO, not five, because the
    // host's `Builder` folds identities at construction and the zero never
    // becomes a step. The shape is kept for exactly that reading — it pins that
    // the program the verifier evaluates carries no `Fixed(0)`, which is what
    // makes the two spellings of a negation indistinguishable and retires a
    // mutation this leg was pre-registered with.
    let mut b = Builder::<GoldilocksExtension>::new();
    let x = b.var(0);
    let z = b.fixed(FEE::zero());
    let shifted = b.sub(x, z);
    let negated = b.neg(shifted);
    let root = b.add(negated, z);
    out.push(Shape {
        name: "a zero the builder folds away, and a negation",
        program: b.finish(root).expect("root is a step"),
        factors: 1,
    });

    // The host's own simplification of each, which is the form the verifier
    // actually evaluates.
    //
    // ⚠ MEASURED, not assumed: `simplify` leaves BOTH of these at the step count
    // they already have — the builder interns a repeated `Fixed` before
    // `simplify` ever sees it, and neither program has a dead step, an add of
    // zero or a multiply by one. An earlier version of this comment claimed the
    // repeated constant was deduped here; the printed step counts say otherwise.
    // What these two shapes pin, then, is that an already-minimal program stays
    // minimal through `simplify` and that the leg costs the same either way —
    // which is worth pinning, because the verifier evaluates the simplified
    // form and the census is taken on the unsimplified one.
    let simplified: Vec<Shape> = out
        .iter()
        .skip(1)
        .map(|shape| Shape {
            name: shape.name,
            program: shape.program.simplify(),
            factors: shape.factors,
        })
        .collect();
    for (index, shape) in simplified.into_iter().enumerate() {
        out.push(Shape {
            name: match index {
                0 => "the repeated constant, simplified by the host",
                1 => "the stack, simplified by the host",
                _ => "the folded zero, simplified by the host",
            },
            program: shape.program,
            factors: shape.factors,
        });
    }

    out
}

/// The program's factors hinted, its root published.
fn program_of(shape: &Shape) -> LfmProgram {
    let mut b = LfmBuilder::new().with_wrap_hash(super::edsl::WrapHash::production());
    let arena = b.declare_arena(shape.factors as u32);
    let values: Vec<Ext> = (0..shape.factors)
        .map(|slot| b.hint_word(arena, slot as u32).as_ext())
        .collect();
    let root = emit_program(&mut b, &shape.program, &values);
    b.public(root.as_cell());
    let program = compile(b.finish());
    validate(&program).expect("the constraint leg must be admissible");
    program
}

/// ★ F1: every row named, and the prediction read off the HOST's program rather
/// than off the emitter's walk.
#[test]
fn the_constraint_leg_emits_its_closed_form() {
    for shape in shapes() {
        let emitted = program_of(&shape);
        let measured = emitted.instrs.len() - shape.factors - 1;
        let predicted = program_rows(&shape.program);
        println!(
            "constraint [{}]: {measured:>3} rows emitted, {predicted:>3} predicted \
             ({} steps)",
            shape.name,
            shape.program.steps().len()
        );
        assert_eq!(
            measured, predicted,
            "{}: the emitted row count must equal the closed form",
            shape.name
        );
    }
}

/// ★ The leg computes what `Program::eval` computes — the function the
/// constraint argument itself calls.
#[test]
fn the_constraint_leg_computes_what_the_host_computes() {
    for shape in shapes() {
        let emitted = program_of(&shape);
        for seed in [0x13u64, 0x27, 0x41] {
            let values: Vec<FEE> = (0..shape.factors)
                .map(|slot| fee(seed * 101 + slot as u64))
                .collect();
            let mut scratch = Vec::new();
            let want = shape.program.eval(&values, &mut scratch);

            let arena: Vec<_> = values.iter().map(ext_word).collect();
            let exec = execute(&emitted, &[arena], &crate::hash_pin::BLOCK_HASHER)
                .expect("the constraint leg executes");
            let got = word_as_ext(&exec.public_words[0].1).expect("a published value");
            assert_eq!(
                got, want,
                "{} at seed {seed:#x}: the emitted leg and the host disagree",
                shape.name
            );
        }
    }
}
