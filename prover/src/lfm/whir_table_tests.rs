//! Gates for the per-table verify's legs.
//!
//! The selector is gated against `Selector::evaluate` itself, at EVERY
//! exemption count rather than at a sample of them: the cost and the value both
//! turn on the bit pattern of `cutoff`, so a gate that picked a few counts
//! would be picking a few bit patterns.

use multilinear::selector::Selector;

use crate::tables::types::{FE, FEE};

use super::builder::LfmBuilder;
use super::compiler::{LfmProgram, compile};
use super::executor::execute;
use super::validator::validate;
use super::whir_bus::Cost;
use super::whir_table::{emit_selector, selector_cost};
use super::word::{LfmWord, ext_word, word_as_ext};

/// One `Public` instruction per published value (`builder.rs:637-642`), which
/// the marginal below has to take out to be about the leg.
const PUBLISH_ROW: usize = 1;

fn pseudo(seed: u64, count: usize) -> Vec<FEE> {
    let mut state = seed | 1;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        FE::from(state >> 2)
    };
    (0..count)
        .map(|_| FEE::new([next(), next(), next()]))
        .collect()
}

fn words(point: &[FEE]) -> Vec<Vec<LfmWord>> {
    vec![point.iter().map(ext_word).collect()]
}

/// Every exemption count a table of `2^num_vars` rows can carry, the trivial
/// and the all-exempt included.
fn every_selector(num_vars: usize) -> Vec<Selector> {
    (0..=(1usize << num_vars))
        .map(Selector::except_last)
        .collect()
}

/// The first `count` selectors emitted over one hinted point, each published.
fn selectors_program(num_vars: usize, count: usize) -> LfmProgram {
    let mut b = LfmBuilder::new().with_wrap_hash(super::edsl::WrapHash::production());
    let arena = b.declare_arena(num_vars.max(1) as u32);
    let point: Vec<_> = (0..num_vars)
        .map(|i| b.hint_word(arena, i as u32).as_ext())
        .collect();
    // A program with no variables still needs something in its arena to hint,
    // and something to publish.
    let anchor = if num_vars == 0 {
        b.hint_word(arena, 0).as_ext()
    } else {
        point[0]
    };
    b.public(anchor.as_cell());
    for selector in every_selector(num_vars).into_iter().take(count) {
        let value = emit_selector(&mut b, selector, &point);
        b.public(value.as_cell());
    }
    compile(b.finish())
}

/// ★ The selector leg computes what the host computes, at every cutoff.
#[test]
fn the_selector_is_the_hosts_indicator() {
    for num_vars in 1..=5usize {
        let selectors = every_selector(num_vars);
        let program = selectors_program(num_vars, selectors.len());
        validate(&program).expect("the selector leg must be admissible");
        for seed in [0x5E1EC7u64, 0xB17, 0xC0FFEE] {
            let point = pseudo(seed, num_vars);
            let exec = execute(&program, &words(&point), &crate::hash_pin::BLOCK_HASHER)
                .unwrap_or_else(|e| panic!("num_vars {num_vars}: must execute: {e:?}"));
            let got: Vec<FEE> = exec.public_words[1..]
                .iter()
                .map(|(_, word)| word_as_ext(word).expect("a published extension value"))
                .collect();
            assert_eq!(got.len(), selectors.len());
            for (selector, value) in selectors.iter().zip(&got) {
                let want = selector
                    .evaluate::<crate::tables::types::GoldilocksExtension>(&point)
                    .expect("the host evaluates it");
                assert_eq!(
                    *value, want,
                    "num_vars {num_vars}, end_exemptions {} at seed {seed:#x}",
                    selector.end_exemptions
                );
            }
        }
    }
}

/// The two cases the general form does not reach by arithmetic: a selector that
/// applies everywhere is the literal one, and one that applies nowhere is the
/// literal zero.
///
/// Named separately because they are the branches that RETURN before the loop,
/// and a gate that only swept the middle would leave both unexercised at the
/// value level.
#[test]
fn the_trivial_and_the_all_exempt_selectors_are_literals() {
    let num_vars = 3;
    let point = pseudo(0xAB1E, num_vars);
    for (selector, want) in [
        (Selector::ALL, FEE::one()),
        (Selector::except_last(1usize << num_vars), FEE::zero()),
    ] {
        let mut b = LfmBuilder::new().with_wrap_hash(super::edsl::WrapHash::production());
        let arena = b.declare_arena(num_vars as u32);
        let wires: Vec<_> = (0..num_vars)
            .map(|i| b.hint_word(arena, i as u32).as_ext())
            .collect();
        let value = emit_selector(&mut b, selector, &wires);
        b.public(value.as_cell());
        let program = compile(b.finish());
        let exec = execute(&program, &words(&point), &crate::hash_pin::BLOCK_HASHER)
            .expect("the literal cases must execute");
        let got = word_as_ext(&exec.public_words[0].1).expect("a published extension value");
        assert_eq!(got, want, "end_exemptions {}", selector.end_exemptions);
        assert_eq!(
            got,
            selector
                .evaluate::<crate::tables::types::GoldilocksExtension>(&point)
                .expect("the host evaluates it"),
            "and it must be the host's answer, not just the literal I expected"
        );
    }
}

/// ★ F1, incrementally: each selector's own marginal rows against its form.
///
/// Incremental rather than total, because the `1` is shared across the whole
/// set — a total would let one selector's over-count cancel another's, and the
/// first selector is the only one that pays for the constant.
#[test]
fn each_selector_emits_its_closed_form() {
    for num_vars in 1..=4usize {
        let selectors = every_selector(num_vars);
        let mut previous = selectors_program(num_vars, 0).instrs.len();
        let mut pool = Cost::default();
        for (index, selector) in selectors.iter().enumerate() {
            let before = pool.rows();
            selector_cost(*selector, num_vars, &mut pool);
            let predicted = pool.rows() - before;

            let now = selectors_program(num_vars, index + 1).instrs.len();
            let measured = now - previous - PUBLISH_ROW;
            previous = now;
            assert_eq!(
                measured, predicted,
                "num_vars {num_vars}, end_exemptions {}",
                selector.end_exemptions
            );
        }
        println!(
            "selectors num_vars={num_vars}: {} selectors, {} rows pooled ({} ops + {} constants)",
            selectors.len(),
            pool.rows(),
            pool.operations(),
            pool.constants()
        );
    }
}

/// The pooled total is what the incremental steps add up to, and the whole set
/// costs it once.
///
/// This is the statement the incremental test cannot make: that the ACCUMULATOR
/// shares its constants rather than charging each selector its own `1`.
#[test]
fn the_selectors_share_one_interned_constant() {
    let num_vars = 4;
    let selectors = every_selector(num_vars);
    let mut pool = Cost::default();
    for selector in &selectors {
        selector_cost(*selector, num_vars, &mut pool);
    }
    let mut separately = 0usize;
    for selector in &selectors {
        let mut alone = Cost::default();
        selector_cost(*selector, num_vars, &mut alone);
        separately += alone.rows();
    }
    // Every selector interns the `1`; the all-exempt one also interns a `0`.
    // Pooled, those are two rows for the whole set.
    let shared = separately - pool.rows();
    assert_eq!(
        shared,
        selectors.len() - 1,
        "the `1` is paid once and not {} times",
        selectors.len()
    );

    let measured = selectors_program(num_vars, selectors.len()).instrs.len()
        - selectors_program(num_vars, 0).instrs.len()
        - selectors.len() * PUBLISH_ROW;
    assert_eq!(measured, pool.rows(), "the pooled form is the emitted set");
}
