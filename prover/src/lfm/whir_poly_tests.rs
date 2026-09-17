//! Gates for the shared polynomial primitives.

use multilinear::eq::eq_eval;

use crate::tables::types::{FE, FEE};

use super::builder::LfmBuilder;
use super::compiler::{LfmProgram, compile};
use super::executor::execute;
use super::validator::validate;
use super::whir_poly::{emit_eq_eval, eq_eval_rows};
use super::word::{ext_word, word_as_ext};

fn sample(seed: u64, n: usize) -> Vec<FEE> {
    let mut state = seed | 1;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        FE::from(state >> 2)
    };
    (0..n).map(|_| FEE::new([next(), next(), next()])).collect()
}

/// `eq` over `n` variables, both points hinted, the result published.
fn eq_only_program(n: usize) -> LfmProgram {
    let mut b = LfmBuilder::new().with_wrap_hash(super::edsl::WrapHash::production());
    let arena = b.declare_arena(2 * n as u32);
    let r: Vec<_> = (0..n)
        .map(|i| b.hint_word(arena, i as u32).as_ext())
        .collect();
    let x: Vec<_> = (0..n)
        .map(|i| b.hint_word(arena, (n + i) as u32).as_ext())
        .collect();
    let v = emit_eq_eval(&mut b, &r, &x);
    b.public(v.as_cell());
    let program = compile(b.finish());
    validate(&program).expect("the eq leg must be admissible");
    program
}

/// The leg's marginal cost: the same program without it.
fn marginal_rows(n: usize) -> usize {
    let with = eq_only_program(n);
    let without = {
        let mut b = LfmBuilder::new().with_wrap_hash(super::edsl::WrapHash::production());
        let arena = b.declare_arena(2 * n as u32);
        let first = b.hint_word(arena, 0).as_ext();
        for i in 1..2 * n {
            let _ = b.hint_word(arena, i as u32);
        }
        b.public(first.as_cell());
        compile(b.finish())
    };
    with.instrs.len() - without.instrs.len()
}

/// ★ F1 for `eq`, across the widths the verifier actually reaches: a table's
/// height in variables, a GKR layer's, and the stack's 25.
#[test]
fn the_eq_leg_emits_its_closed_form() {
    for n in [1usize, 2, 4, 13, 20, 21, 25, 31] {
        let measured = marginal_rows(n);
        let predicted = eq_eval_rows(n);
        println!("eq over {n:>2} variables: {measured:>4} rows emitted, {predicted:>4} predicted");
        assert_eq!(
            measured, predicted,
            "eq over {n} variables: the emitted row count must equal the closed form"
        );
    }
}

/// ★ The leg computes what `multilinear::eq::eq_eval` computes — the function
/// the verifier calls, not a restatement of the emitter's own algebra.
#[test]
fn the_eq_leg_computes_what_the_host_computes() {
    for n in [1usize, 4, 20] {
        let program = eq_only_program(n);
        for seed in [0x11u64, 0x22, 0x33] {
            let r = sample(seed, n);
            let x = sample(seed ^ 0xFFFF, n);
            let arenas = vec![r.iter().chain(x.iter()).map(ext_word).collect::<Vec<_>>()];
            let exec = execute(&program, &arenas, &crate::hash_pin::BLOCK_HASHER)
                .expect("the eq leg executes");
            let got = word_as_ext(&exec.public_words[0].1).expect("a published extension value");
            let want = eq_eval(&r, &x).expect("the host agrees on the width");
            assert_eq!(
                got, want,
                "eq over {n} variables at seed {seed:#x}: the emitted leg and \
                 the host disagree"
            );
        }
    }
}

/// `eq` is an indicator on the cube: one when the two points are equal corners,
/// zero otherwise. A leg that got the factor backwards still agrees with a
/// host that got it backwards the same way; this does not.
#[test]
fn eq_is_one_on_the_diagonal_and_zero_off_it() {
    let n = 6;
    let program = eq_only_program(n);
    let corner = |mask: usize| -> Vec<FEE> {
        (0..n)
            .map(|i| {
                let bit = (mask >> i) & 1;
                FEE::new([FE::from(bit as u64), FE::zero(), FE::zero()])
            })
            .collect()
    };
    let run = |r: &[FEE], x: &[FEE]| -> FEE {
        let arenas = vec![r.iter().chain(x.iter()).map(ext_word).collect::<Vec<_>>()];
        let exec = execute(&program, &arenas, &crate::hash_pin::BLOCK_HASHER).expect("executes");
        word_as_ext(&exec.public_words[0].1).expect("a published extension value")
    };
    for mask in [0usize, 1, 0b101010, 0b111111] {
        let c = corner(mask);
        assert_eq!(
            run(&c, &c),
            FEE::one(),
            "eq must be one where the corners agree ({mask:#08b})"
        );
        let other = corner(mask ^ 0b000100);
        assert_eq!(
            run(&c, &other),
            FEE::zero(),
            "eq must be zero where the corners differ ({mask:#08b})"
        );
    }
}
