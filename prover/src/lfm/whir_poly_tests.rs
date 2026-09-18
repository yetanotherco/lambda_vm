//! Gates for the shared polynomial primitives.

use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use multilinear::eq::{eq_eval, rot_eval, shift_eval};
use multilinear::sumcheck::{RoundProof, verify_rounds};

use crate::tables::types::{FE, FEE, GoldilocksExtension};

use super::builder::LfmBuilder;
use super::compiler::{LfmProgram, compile};
use super::executor::execute;
use super::validator::validate;
use super::whir_poly::{
    emit_eq_eval, emit_shift_eval, emit_sumcheck_rounds, eq_eval_rows, eq_eval_rows_again,
    shift_eval_rows, sumcheck_round_consts, sumcheck_round_rows,
};
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

/// `legs` independent `eq` legs over their own hinted points, and the rows they
/// emit BETWEEN them — the program's total less the `2n + 1` hints and public
/// each leg's plumbing contributes, which is countable by construction rather
/// than by differencing against another program.
fn legs_rows(n: usize, legs: usize) -> usize {
    let mut b = LfmBuilder::new().with_wrap_hash(super::edsl::WrapHash::production());
    let arena = b.declare_arena((2 * n * legs) as u32);
    for leg in 0..legs {
        let base = (2 * n * leg) as u32;
        let r: Vec<_> = (0..n)
            .map(|i| b.hint_word(arena, base + i as u32).as_ext())
            .collect();
        let x: Vec<_> = (0..n)
            .map(|i| b.hint_word(arena, base + (n + i) as u32).as_ext())
            .collect();
        let v = emit_eq_eval(&mut b, &r, &x);
        b.public(v.as_cell());
    }
    let program = compile(b.finish());
    validate(&program).expect("a multi-leg eq program must be admissible");
    program.instrs.len() - legs * (2 * n + 1)
}

/// ★ The second pin on the cost, and the reason there are two.
///
/// `eq_eval_rows(n)` is `5n` at every width, so the single-leg measurement in
/// [`the_eq_leg_emits_its_closed_form`] forces only the SUM of the form's
/// terms. A second leg in the same program separates them: it pays no second
/// constant, so it costs `5n − 1`. Between the two measurements all three terms
/// are determined, and a form that moved a row from the constant into the
/// per-variable count would pass the first pin and fail this one.
///
/// It also carries a control on the first pin's method — `legs_rows(n, 1)`
/// counts the leg directly, where `marginal_rows(n)` differences two programs,
/// and the two agree.
#[test]
fn the_interned_one_is_paid_once_per_program() {
    for n in [1usize, 4, 20, 25] {
        let one_leg = legs_rows(n, 1);
        let two_legs = legs_rows(n, 2);
        let second = two_legs - one_leg;
        println!(
            "eq over {n:>2} variables: first leg {one_leg:>4} rows emitted, \
             {:>4} predicted; second leg {second:>4} rows emitted, {:>4} predicted",
            eq_eval_rows(n),
            eq_eval_rows_again(n)
        );
        assert_eq!(
            one_leg,
            eq_eval_rows(n),
            "eq over {n} variables: counted directly, the first leg must equal the closed form"
        );
        assert_eq!(
            one_leg,
            marginal_rows(n),
            "eq over {n} variables: the direct count and the differenced count must agree"
        );
        assert_eq!(
            second,
            eq_eval_rows_again(n),
            "eq over {n} variables: a second leg must pay every row but the interned constant"
        );
    }
}

/// ★ The zero-variable entry, which the assembled census REACHES rather than
/// avoids: `gkr_layer_rows(i)` adds `eq_eval_rows_again(i)`, so the ladder's
/// layer 0 asks for `eq` over no variables. Both cost branches (`= 1` for a
/// first leg, `= 0` for a further one) and the value are gated here; the loops
/// above start at one variable and could not have.
///
/// The value side is against `eq_eval(&[], &[])` — the host's own empty
/// product, not the emitter's claim about it.
#[test]
fn eq_over_no_variables_is_the_empty_product() {
    let one_leg = legs_rows(0, 1);
    let second = legs_rows(0, 2) - one_leg;
    println!(
        "eq over  0 variables: first leg {one_leg} rows emitted, {} predicted; \
         second leg {second} rows emitted, {} predicted",
        eq_eval_rows(0),
        eq_eval_rows_again(0)
    );
    assert_eq!(
        one_leg,
        eq_eval_rows(0),
        "the empty product costs exactly the constant it returns"
    );
    assert_eq!(
        second,
        eq_eval_rows_again(0),
        "a further empty product costs nothing: the constant is already interned"
    );

    let program = eq_only_program(0);
    let exec = execute(&program, &[Vec::new()], &crate::hash_pin::BLOCK_HASHER)
        .expect("the eq leg executes over an empty point");
    let got = word_as_ext(&exec.public_words[0].1).expect("a published extension value");
    let want = eq_eval::<GoldilocksExtension>(&[], &[]).expect("the host agrees on the width");
    assert_eq!(
        got, want,
        "eq over no variables: the emitted leg and the host disagree on the empty product"
    );
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

/// A group of `rounds` sumcheck rounds at `degree`, everything hinted: the
/// claim carried in at word 0, then each round's `degree` evaluations followed
/// by its challenge. The claim the group leaves is published.
///
/// ⚠ Hinting the challenges is what makes this a TEST and not a verifier — the
/// arena rule forbids deriving a challenge from an arena. The replay supplies
/// them in the assembled program.
fn sumcheck_program(degree: usize, rounds: usize) -> LfmProgram {
    let mut b = LfmBuilder::new().with_wrap_hash(super::edsl::WrapHash::production());
    let arena = b.declare_arena(sumcheck_arena_words(degree, rounds) as u32);
    let claim = b.hint_word(arena, 0).as_ext();
    let mut evaluations = Vec::with_capacity(rounds);
    let mut challenges = Vec::with_capacity(rounds);
    for round in 0..rounds {
        let base = (1 + round * (degree + 1)) as u32;
        evaluations.push(
            (0..degree)
                .map(|j| b.hint_word(arena, base + j as u32).as_ext())
                .collect::<Vec<_>>(),
        );
        challenges.push(b.hint_word(arena, base + degree as u32).as_ext());
    }
    let out = emit_sumcheck_rounds(&mut b, claim, &evaluations, &challenges);
    b.public(out.as_cell());
    let program = compile(b.finish());
    validate(&program).expect("the sumcheck leg must be admissible");
    program
}

/// The claim, then `degree` evaluations and one challenge per round.
fn sumcheck_arena_words(degree: usize, rounds: usize) -> usize {
    1 + rounds * (degree + 1)
}

/// The leg's rows: the program's total less the hints and the one public, which
/// are countable by construction.
fn sumcheck_leg_rows(degree: usize, rounds: usize) -> usize {
    let program = sumcheck_program(degree, rounds);
    program.instrs.len() - sumcheck_arena_words(degree, rounds) - 1
}

/// ★ F1 for the sumcheck round, as a SLOPE: one more round in the same program
/// costs exactly `sumcheck_round_rows(degree)`, with the interned constants
/// cancelling out of the difference. The intercept pins those constants.
#[test]
fn a_sumcheck_round_costs_its_closed_form() {
    for degree in [1usize, 2, 3, 5, 7] {
        for rounds in [1usize, 2, 5] {
            let slope = sumcheck_leg_rows(degree, rounds + 1) - sumcheck_leg_rows(degree, rounds);
            let predicted = sumcheck_round_rows(degree);
            println!(
                "sumcheck degree {degree}, round {} → {}: {slope:>3} rows emitted, \
                 {predicted:>3} predicted",
                rounds,
                rounds + 1
            );
            assert_eq!(
                slope, predicted,
                "degree {degree}: one more round must cost the closed form"
            );
        }
        let first = sumcheck_leg_rows(degree, 1);
        let predicted = sumcheck_round_rows(degree) + sumcheck_round_consts(degree);
        println!(
            "sumcheck degree {degree}, first round: {first:>3} rows emitted, \
             {predicted:>3} predicted (round + interned constants)"
        );
        assert_eq!(
            first, predicted,
            "degree {degree}: the first round pays for its Newton constants too"
        );
    }
}

/// ★ The degree-0 entry, which is the caller's degree and not a shape the
/// emitter can be handed.
///
/// `verify_rounds` clamps with `degree.max(1)` (`sumcheck.rs:366`), so a rule
/// whose degree is zero is verified at degree ONE and its round proof carries
/// one evaluation. A census adding up `sumcheck_round_rows(degree_of(rules))`
/// holds the RAW degree, so the cost form has to be callable there — and
/// `clamp_degree` is only right if it matches the host's clamp. Nothing above
/// can see that: the loops start at one and the emitter refuses zero
/// evaluations outright.
///
/// Three observations, each able to fail: the host ACCEPTS a one-evaluation
/// round at degree 0 (the clamp, watched rather than assumed); it REJECTS a
/// two-evaluation one, which is what says the clamp is a clamp and not an
/// unchecked pass; and the cost form at the raw 0 predicts the program emitted
/// at the width the clamp implies.
#[test]
fn a_degree_zero_rule_is_verified_at_degree_one() {
    let rounds = 3;
    let values = sample(0x0D, 1 + rounds);
    let claim = values[0];
    let proof: Vec<RoundProof<GoldilocksExtension>> = (0..rounds)
        .map(|round| RoundProof {
            evaluations: vec![values[1 + round]],
        })
        .collect();

    let mut transcript = DefaultTranscript::<GoldilocksExtension>::new(b"v1-sumcheck-degree0");
    let host = verify_rounds(&proof, claim, 0, &mut transcript)
        .expect("asked for degree 0, the host clamps to 1 and accepts one evaluation a round");
    assert_eq!(host.point.len(), rounds, "one challenge per round");

    let mut wider = DefaultTranscript::<GoldilocksExtension>::new(b"v1-sumcheck-degree0");
    let two: Vec<RoundProof<GoldilocksExtension>> = (0..rounds)
        .map(|round| RoundProof {
            evaluations: vec![values[1 + round], values[1 + round]],
        })
        .collect();
    assert!(
        verify_rounds(&two, claim, 0, &mut wider).is_err(),
        "the clamp is a clamp: at degree 0 a two-evaluation round must be rejected, or \
         the acceptance above says nothing about the width"
    );

    // The cost form at the RAW degree, against the program emitted at the width
    // the clamp implies.
    let slope = sumcheck_leg_rows(1, rounds + 1) - sumcheck_leg_rows(1, rounds);
    let first = sumcheck_leg_rows(1, 1);
    println!(
        "sumcheck degree 0 (clamped to 1): {slope} rows a round emitted, \
         {} predicted; first round {first}, {} predicted",
        sumcheck_round_rows(0),
        sumcheck_round_rows(0) + sumcheck_round_consts(0)
    );
    assert_eq!(
        slope,
        sumcheck_round_rows(0),
        "a degree-0 rule's round must cost what the clamped width emits"
    );
    assert_eq!(
        first,
        sumcheck_round_rows(0) + sumcheck_round_consts(0),
        "and its constants must be the clamped width's constants"
    );

    // The value, against the host's own claim.
    let mut arena = vec![ext_word(&claim)];
    for (round, r) in host.point.iter().enumerate() {
        arena.extend(proof[round].evaluations.iter().map(ext_word));
        arena.push(ext_word(r));
    }
    let program = sumcheck_program(1, rounds);
    let exec = execute(&program, &[arena], &crate::hash_pin::BLOCK_HASHER)
        .expect("the sumcheck leg executes at the clamped width");
    let got = word_as_ext(&exec.public_words[0].1).expect("a published claim");
    assert_eq!(
        got, host.expected_evaluation,
        "degree 0 over {rounds} rounds: the emitted leg and the host disagree"
    );
}

/// ★ The leg computes what `sumcheck::verify_rounds` computes, driven by that
/// function's OWN challenges.
///
/// The host is run first over a real transcript on arbitrary evaluations — it
/// never rejects, so any values are a legal input, and generic values exercise
/// the recursion harder than a satisfiable proof would. The challenges it
/// sampled come back as `SumcheckClaim::point`; those are hinted into the
/// machine, and the machine's final claim is compared against the host's
/// `expected_evaluation`. Nothing here compares the emitter against my own
/// interpolation.
#[test]
fn the_sumcheck_leg_computes_what_the_host_computes() {
    for degree in [1usize, 2, 3, 5, 7] {
        for rounds in [1usize, 4] {
            for seed in [0xA5u64, 0x5A, 0xC3] {
                let values = sample(seed, 1 + rounds * degree);
                let claim = values[0];
                let proof: Vec<RoundProof<GoldilocksExtension>> = (0..rounds)
                    .map(|round| RoundProof {
                        evaluations: values[1 + round * degree..1 + (round + 1) * degree].to_vec(),
                    })
                    .collect();

                let mut transcript =
                    DefaultTranscript::<GoldilocksExtension>::new(b"v1-sumcheck-gate");
                let host = verify_rounds(&proof, claim, degree, &mut transcript)
                    .expect("verify_rounds accepts a well-shaped group");
                assert_eq!(host.point.len(), rounds, "one challenge per round");

                let mut arena = vec![ext_word(&claim)];
                for (round, r) in host.point.iter().enumerate() {
                    arena.extend(proof[round].evaluations.iter().map(ext_word));
                    arena.push(ext_word(r));
                }
                let program = sumcheck_program(degree, rounds);
                let exec = execute(&program, &[arena], &crate::hash_pin::BLOCK_HASHER)
                    .expect("the sumcheck leg executes");
                let got = word_as_ext(&exec.public_words[0].1).expect("a published claim");
                assert_eq!(
                    got, host.expected_evaluation,
                    "degree {degree}, {rounds} rounds, seed {seed:#x}: the emitted leg and \
                     the host disagree on the claim the group leaves"
                );
            }
        }
    }
}

/// `shift_k` over `n` variables, both points hinted, the result published.
fn shift_only_program(n: usize, k: usize) -> LfmProgram {
    let mut b = LfmBuilder::new().with_wrap_hash(super::edsl::WrapHash::production());
    let arena = b.declare_arena(2 * n as u32);
    let x: Vec<_> = (0..n)
        .map(|i| b.hint_word(arena, i as u32).as_ext())
        .collect();
    let y: Vec<_> = (0..n)
        .map(|i| b.hint_word(arena, (n + i) as u32).as_ext())
        .collect();
    let v = emit_shift_eval(&mut b, &x, &y, k);
    b.public(v.as_cell());
    let program = compile(b.finish());
    validate(&program).expect("the shift leg must be admissible");
    program
}

/// The shift leg's marginal cost, measured the way the `eq` leg's is: the same
/// program without it.
fn shift_marginal_rows(n: usize, k: usize) -> usize {
    let with = shift_only_program(n, k);
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

/// The shifts `claim_reduce` reaches, and the ones that exercise the carry.
///
/// A VM table's factors are read at frame offsets — 0 and 1 in the tables this
/// campaign proves — but the kernel takes any `k`, and a form that only ever
/// saw 0 and 1 would never see a carry ripple past one bit.
const SHIFTS: [(usize, usize); 12] = [
    (1, 0),
    (1, 1),
    (4, 0),
    (4, 1),
    (4, 2),
    (4, 5),
    (4, 15),
    (7, 0),
    (7, 1),
    (7, 3),
    (7, 64),
    (7, 127),
];

/// ★ F1 for the shift kernel, against a form derived from the recursion rather
/// than from the emitter.
#[test]
fn the_shift_leg_emits_its_closed_form() {
    for (n, k) in SHIFTS {
        let measured = shift_marginal_rows(n, k);
        // The leg's own rows, plus the `LFM_CONST` holding `1` — the same
        // accounting `eq_eval_rows` uses, where the constant is paid once per
        // program and `shift_eval_rows` is the second-leg form.
        let predicted = shift_eval_rows(n, k) + 1;
        println!("shift n={n} k={k}: {measured} rows emitted, {predicted} predicted");
        assert_eq!(measured, predicted, "shift over {n} variables at k={k}");
    }
}

/// ★★ At `k = 0` the shift IS `eq`, in VALUE and in COST, and the two forms
/// were derived from different code.
///
/// The second carry state is never reached with every bit zero, so the
/// recursion collapses to `eq`'s: this pins `shift_eval_rows(n, 0)` against
/// `eq_eval_rows_again(n)`, a number this module pinned another way entirely
/// (against a second leg in one program, to fix the split of `5n`).
#[test]
fn the_shift_form_is_eqs_at_a_zero_shift() {
    for n in 1..=12 {
        assert_eq!(
            shift_eval_rows(n, 0),
            eq_eval_rows_again(n),
            "a zero shift is eq over {n} variables, and must cost exactly that"
        );
    }
}

/// ★ The shift leg against the host, at every shift in [`SHIFTS`].
///
/// `shift_eval` is the function `claim_reduce` settles a shifted column with,
/// and it is the only kernel in the per-table verify whose answer depends on a
/// constant the AIR chose rather than on a challenge.
#[test]
fn the_shift_leg_computes_what_the_host_computes() {
    for (n, k) in SHIFTS {
        let program = shift_only_program(n, k);
        for seed in [0x11u64, 0x5eed] {
            let x = sample(seed, n);
            let y = sample(seed ^ 0xFFFF, n);
            let arenas = vec![x.iter().chain(y.iter()).map(ext_word).collect::<Vec<_>>()];
            let exec = execute(&program, &arenas, &crate::hash_pin::BLOCK_HASHER)
                .expect("the shift leg executes");
            let got = word_as_ext(&exec.public_words[0].1).expect("a published extension value");
            let want = shift_eval(&x, &y, k).expect("the host agrees on the width");
            assert_eq!(got, want, "shift_{k} over {n} variables at seed {seed:#x}");
            if k == 0 {
                assert_eq!(want, eq_eval(&x, &y).expect("eq agrees"), "shift_0 is eq");
            }
            if k == 1 {
                assert_eq!(want, rot_eval(&x, &y).expect("rot agrees"), "shift_1 is rot");
            }
        }
    }
}

/// ★ The shift is an indicator on the cube, and that is what says the kernel
/// counts in the right DIRECTION.
///
/// A leg that added the constant the other way round agrees with the host's own
/// arithmetic at random points only if it is wrong in the same direction; the
/// corners say which way it goes. `index(y) = index(x) + k mod 2^n`, with the
/// index read big-endian — variable 0 is the most significant, which is why the
/// recursion walks the variables in reverse.
#[test]
fn the_shift_is_one_exactly_where_the_index_advances() {
    let n = 4;
    let size = 1usize << n;
    for k in [0usize, 1, 3, 13] {
        let program = shift_only_program(n, k);
        let corner = |index: usize| -> Vec<FEE> {
            (0..n)
                .map(|bit| {
                    if index >> (n - 1 - bit) & 1 == 1 {
                        FEE::one()
                    } else {
                        FEE::zero()
                    }
                })
                .collect()
        };
        for from in 0..size {
            for to in 0..size {
                let x = corner(from);
                let y = corner(to);
                let arenas = vec![x.iter().chain(y.iter()).map(ext_word).collect::<Vec<_>>()];
                let exec = execute(&program, &arenas, &crate::hash_pin::BLOCK_HASHER)
                    .expect("the shift leg executes");
                let got = word_as_ext(&exec.public_words[0].1).expect("a published value");
                let want = if to == (from + k) % size {
                    FEE::one()
                } else {
                    FEE::zero()
                };
                assert_eq!(got, want, "shift_{k}: corner {from} to corner {to}");
            }
        }
    }
}

/// Bits of `k` at or above `n` are never read, so the kernel is a function of
/// `k mod 2^n` — the host's loop only ever asks for `t < n` (`eq.rs:155`).
///
/// `claim_reduce` is handed a factor's raw offset, and `materialize` reduces it
/// on the prover's side (`claim_reduce.rs:357`); this is what says the two
/// agree without the verifier reducing anything.
#[test]
fn the_shift_reads_only_the_bits_it_has_variables_for() {
    let n = 4;
    let size = 1usize << n;
    for k in [0usize, 3, 9] {
        assert_eq!(shift_eval_rows(n, k), shift_eval_rows(n, k + size));
        let x = sample(0xA1, n);
        let y = sample(0xB2, n);
        assert_eq!(
            shift_eval(&x, &y, k).expect("the host agrees"),
            shift_eval(&x, &y, k + 3 * size).expect("the host agrees"),
        );
    }
}
