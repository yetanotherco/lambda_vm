//! Gates for the preprocessed leg.
//!
//! Two obligations, and they fail for different reasons. The closed form in
//! [`super::preprocessed::bitwise_preprocessed_rows`] is a claim about how much
//! the emitter costs; `emit_bitwise_preprocessed` is a claim about what it
//! computes. A leg can satisfy either alone: one that emits the predicted
//! number of rows and the wrong values, or the right values at a cost nobody
//! predicted. So the count is pinned against the emitter and the values are
//! pinned against the host's own fold over the real 2^20 columns — never
//! against each other.

use multilinear::mle::Mle;

use crate::tables::bitwise::{NUM_PRECOMPUTED_COLS, NUM_VARS, preprocessed_mle_at};
use crate::tables::types::{FE, FEE, GoldilocksExtension};

use super::builder::LfmBuilder;
use super::compiler::{LfmProgram, compile};
use super::executor::execute;
use super::preprocessed::{bitwise_preprocessed_rows, emit_bitwise_preprocessed};
use super::validator::validate;
use super::word::{ext_word, word_as_ext};

/// A fixed-seed xorshift, so a failure names one reproducible point.
fn sample_point(seed: u64) -> Vec<FEE> {
    let mut state = seed | 1;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        FE::from(state >> 2)
    };
    (0..NUM_VARS)
        .map(|_| FEE::new([next(), next(), next()]))
        .collect()
}

/// The leg alone: the point arrives by hint, the eleven columns are published.
///
/// Hinting the point is a TEST convenience, not the production shape — there
/// the point is the reduced claim the sumcheck left, and a challenge must never
/// come from an arena.
fn leg_only_program() -> LfmProgram {
    let mut b = LfmBuilder::new().with_wrap_hash(super::edsl::WrapHash::production());
    let arena = b.declare_arena(NUM_VARS as u32);
    let point: Vec<_> = (0..NUM_VARS)
        .map(|i| b.hint_word(arena, i as u32).as_ext())
        .collect();
    let columns = emit_bitwise_preprocessed(&mut b, &point);
    for c in columns {
        b.public(c.as_cell());
    }
    let program = compile(b.finish());
    validate(&program).expect("the preprocessed leg must be admissible");
    program
}

/// The rows the leg costs, measured as the MARGINAL cost of emitting it.
///
/// A whole-program count would carry the hints, the publishes and whatever the
/// builder interns; the difference between a program with the leg and the same
/// program without it is the leg and nothing else.
fn marginal_rows() -> usize {
    let with = leg_only_program();
    let without = {
        let mut b = LfmBuilder::new().with_wrap_hash(super::edsl::WrapHash::production());
        let arena = b.declare_arena(NUM_VARS as u32);
        let point: Vec<_> = (0..NUM_VARS)
            .map(|i| b.hint_word(arena, i as u32).as_ext())
            .collect();
        // Publish the inputs, so the hints and publishes cancel and only the
        // leg's own rows survive the subtraction.
        for p in point.iter().take(NUM_PRECOMPUTED_COLS) {
            b.public(p.as_cell());
        }
        compile(b.finish())
    };
    with.instrs.len() - without.instrs.len()
}

/// ★ F1. The emitted count equals the closed form.
#[test]
fn the_bitwise_leg_emits_its_closed_form() {
    let measured = marginal_rows();
    let predicted = bitwise_preprocessed_rows();
    println!(
        "BITWISE preprocessed leg: {measured} rows emitted, {predicted} predicted; \
         the fold it replaces is {} steps",
        11u64 << NUM_VARS
    );
    assert_eq!(
        measured, predicted,
        "the emitted row count must equal the closed form — if the emitter \
         changed, the closed form's named terms say which one"
    );
}

/// ★ The leg computes what the host's fold computes, over the real columns.
///
/// The right-hand side is `Mle::evaluate_in` over
/// `bitwise::preprocessed_columns()` — the 2^20-row fold this leg exists to
/// avoid — so the comparison is against the function being replaced rather than
/// against the closed form that replaced it.
#[test]
fn the_bitwise_leg_computes_what_the_host_fold_computes() {
    let program = leg_only_program();
    let mles: Vec<Mle<crate::tables::types::GoldilocksField>> =
        crate::tables::bitwise::preprocessed_columns()
            .into_iter()
            .map(|values| Mle::new(values).expect("a power-of-two column"))
            .collect();
    assert_eq!(mles.len(), NUM_PRECOMPUTED_COLS);

    for seed in [0x5eed_1001u64, 0x5eed_1002] {
        let point = sample_point(seed);
        let arenas = vec![point.iter().map(ext_word).collect::<Vec<_>>()];
        let exec = execute(&program, &arenas, &crate::hash_pin::BLOCK_HASHER)
            .expect("the preprocessed leg executes");
        let got: Vec<FEE> = exec
            .public_words
            .iter()
            .map(|(_, w)| word_as_ext(w).expect("a published extension value"))
            .collect();
        assert_eq!(got.len(), NUM_PRECOMPUTED_COLS);

        // The closed form the emitter mirrors, and the fold it replaces.
        let closed = preprocessed_mle_at(&point).expect("the point is NUM_VARS long");
        for (col, mle) in mles.iter().enumerate() {
            let folded = mle
                .evaluate_in::<GoldilocksExtension>(&point)
                .expect("the column has NUM_VARS variables");
            assert_eq!(
                got[col], folded,
                "column {col} at seed {seed:#x}: the EMITTED leg disagrees with \
                 the host's 2^20 fold"
            );
            assert_eq!(
                got[col], closed[col],
                "column {col} at seed {seed:#x}: the emitted leg disagrees with \
                 the host closed form it mirrors"
            );
        }
    }
}
