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
use super::preprocessed::{
    MAX_CONST_MLE_VARS, bitwise_preprocessed_rows, const_mle_constants, const_mle_rows,
    emit_bitwise_preprocessed, emit_const_mle_at, eq_table_rows,
};
use super::validator::validate;
use super::whir_chain_tests::const_rows;
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

// =============================================================================
// The columns no closed form covers: KECCAK_RC and REGISTER
// =============================================================================

/// A fixed-seed point of `n` coordinates, so a failure names one reproducible
/// point.
fn sample_point_n(seed: u64, n: usize) -> Vec<FEE> {
    let mut state = seed | 1;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        FE::from(state >> 2)
    };
    (0..n).map(|_| FEE::new([next(), next(), next()])).collect()
}

/// The two real continuation-epoch tables this route serves, with their columns
/// as the AIR hands them over.
///
/// REGISTER's `init` and `fini` are a register file, so they are chosen here
/// rather than zeroed: an all-zero column is the one shape whose fold cannot
/// distinguish a right answer from a wrong one.
fn const_mle_fixtures() -> Vec<(&'static str, Vec<Vec<FE>>)> {
    let init: Vec<u32> = (0..crate::tables::register::NUM_REGISTER_ADDRESSES)
        .map(|i| (i as u32).wrapping_mul(7).wrapping_add(3) % 251)
        .collect();
    let fini: Vec<u32> = (0..crate::tables::register::NUM_REGISTER_ADDRESSES)
        .map(|i| (i as u32).wrapping_mul(11).wrapping_add(5) % 241)
        .collect();
    vec![
        (
            "KECCAK_RC",
            crate::tables::keccak_rc::preprocessed_columns(),
        ),
        (
            "REGISTER",
            crate::tables::register::preprocessed_columns_with_fini(&init, &fini),
        ),
    ]
}

/// The leg alone: the point arrives by hint, the columns' values are published.
fn const_mle_program(columns: &[Vec<FE>], num_vars: usize, leg: bool) -> LfmProgram {
    let mut b = LfmBuilder::new().with_wrap_hash(super::edsl::WrapHash::production());
    let arena = b.declare_arena(num_vars as u32);
    let point: Vec<_> = (0..num_vars)
        .map(|i| b.hint_word(arena, i as u32).as_ext())
        .collect();
    if !leg {
        // The hints and the publishes cancel out of the subtraction, so what
        // survives it is the leg and nothing else.
        //
        // ⚠ ONE publish per COLUMN, not per coordinate. A table can have more
        // columns than variables (KECCAK_RC has nine of each five), and a
        // `take(columns.len())` over a shorter point silently publishes fewer
        // cells here than the leg does — which the subtraction then charges to
        // the leg. That is exactly how this came in four rows long.
        for index in 0..columns.len() {
            b.public(point[index % num_vars].as_cell());
        }
        return compile(b.finish());
    }
    let borrowed: Vec<&[FE]> = columns.iter().map(Vec::as_slice).collect();
    for value in emit_const_mle_at(&mut b, &borrowed, &point) {
        b.public(value.as_cell());
    }
    let program = compile(b.finish());
    validate(&program).expect("the constant-MLE leg must be admissible");
    program
}

/// The `LFM_CONST` words a program interns, in no particular order.
fn const_words_of(program: &LfmProgram) -> Vec<super::word::LfmWord> {
    program
        .instrs
        .iter()
        .filter_map(|instr| match instr {
            super::instr::Instr::Const { value, .. } => Some(*value),
            _ => None,
        })
        .collect()
}

/// ★ THE VALUE GATE: the leg computes what `check_preprocessed`'s own fold
/// computes, over the real columns of the two real tables.
///
/// The right-hand side is `Mle::evaluate_in`, which is the exact call
/// `check_preprocessed` (`multilinear_table.rs:1001`) makes — the host
/// function this leg replaces, not a model of it.
#[test]
fn the_constant_mle_leg_is_the_hosts_fold() {
    for (name, columns) in const_mle_fixtures() {
        let num_vars = columns[0].len().trailing_zeros() as usize;
        let program = const_mle_program(&columns, num_vars, true);
        for seed in [1u64, 0x5eed, 0xDEAD_BEEF] {
            let point = sample_point_n(seed, num_vars);
            let arena: Vec<_> = point.iter().map(ext_word).collect();
            let exec = execute(&program, &[arena], &crate::hash_pin::BLOCK_HASHER)
                .unwrap_or_else(|e| panic!("{name}: the leg must execute: {e:?}"));
            assert_eq!(
                exec.public_words.len(),
                columns.len(),
                "{name}: one published value per preprocessed column"
            );
            for (index, column) in columns.iter().enumerate() {
                let emitted = word_as_ext(&exec.public_words[index].1).expect("a published value");
                let host = Mle::new(column.clone())
                    .expect("a preprocessed column is a power of two long")
                    .evaluate_in::<GoldilocksExtension>(&point)
                    .expect("the point has the column's variables");
                assert_eq!(
                    emitted, host,
                    "{name} column {index} at seed {seed}: the machine must \
                     compute the fold `check_preprocessed` computes"
                );
            }
        }
    }
}

/// ★ F1. The emitted count equals the closed form, and the pool is NAMED.
///
/// The second half is the assertion instance 63 earned: the test ASKS which
/// constants the program interns that the form does not name, so the next
/// unnamed constant fails here instead of being absorbed into a fudge factor.
#[test]
fn the_constant_mle_leg_emits_its_closed_form() {
    for (name, columns) in const_mle_fixtures() {
        let num_vars = columns[0].len().trailing_zeros() as usize;
        let with = const_mle_program(&columns, num_vars, true);
        let without = const_mle_program(&columns, num_vars, false);
        let consts_with = const_words_of(&with);
        let consts_without = const_words_of(&without);
        let measured =
            (with.instrs.len() - without.instrs.len()) - (consts_with.len() - consts_without.len());
        let borrowed: Vec<&[FE]> = columns.iter().map(Vec::as_slice).collect();
        let predicted = const_mle_rows(&borrowed, num_vars);
        println!(
            "{name} constant-MLE leg at {num_vars} vars, {} columns: \
             {measured} rows emitted, {predicted} predicted \
             (eq table {}, folds {}); the host fold it replaces is {} steps",
            columns.len(),
            eq_table_rows(num_vars),
            predicted - eq_table_rows(num_vars),
            columns.len() << num_vars,
        );
        assert_eq!(
            measured, predicted,
            "{name}: the emitted operation count must equal the closed form"
        );

        // `LfmWord` is four field elements and field elements are not ordered,
        // so the two sets are compared by containment both ways rather than by
        // sorting — which also reports WHICH word is unnamed.
        let named = const_mle_constants(&borrowed);
        let mut interned: Vec<super::word::LfmWord> = Vec::new();
        for word in consts_with {
            if !consts_without.contains(&word) && !interned.contains(&word) {
                interned.push(word);
            }
        }
        for word in &interned {
            assert!(
                named.contains(word),
                "{name}: the program interns a constant the form does not name \
                 ({word:?}) — that is a term, not a fudge factor"
            );
        }
        for word in &named {
            assert!(
                interned.contains(word),
                "{name}: the form names a constant the program does not intern \
                 ({word:?})"
            );
        }
        assert_eq!(
            interned.len(),
            named.len(),
            "{name}: the interned pool and the named pool must be the same set"
        );
    }
}

/// ⛔ The cap refuses rather than emitting a program nobody can prove.
///
/// The failure this asserts is the one that would otherwise arrive as a prove
/// that does not finish: a preprocessed table at twenty variables costs `2^20`
/// rows a column on this route, and BITWISE and DECODE are precisely the tables
/// that must not take it.
#[test]
#[should_panic(expected = "needs a closed form or a prepared opening")]
fn a_table_above_the_cap_is_refused() {
    let num_vars = MAX_CONST_MLE_VARS + 1;
    let column: Vec<FE> = vec![FE::from(1u64); 1usize << num_vars];
    const_mle_program(std::slice::from_ref(&column), num_vars, true);
}

/// The cap's CONTROL: one variable below it emits and executes, so the refusal
/// above is the cap and not a leg that cannot serve any table.
#[test]
fn a_table_at_the_cap_still_emits() {
    let num_vars = MAX_CONST_MLE_VARS;
    let column: Vec<FE> = (0..1usize << num_vars)
        .map(|i| FE::from(i as u64))
        .collect();
    let columns = vec![column.clone()];
    let program = const_mle_program(&columns, num_vars, true);
    let point = sample_point_n(7, num_vars);
    let arena: Vec<_> = point.iter().map(ext_word).collect();
    let exec = execute(&program, &[arena], &crate::hash_pin::BLOCK_HASHER)
        .expect("a table at the cap must still execute");
    let emitted = word_as_ext(&exec.public_words[0].1).expect("a published value");
    let host = Mle::new(column)
        .expect("a power of two")
        .evaluate_in::<GoldilocksExtension>(&point)
        .expect("the point has the column's variables");
    assert_eq!(
        emitted, host,
        "the value at the cap is still the host's fold"
    );
}

// =============================================================================
// The OFFSET ramp — the cross-epoch proof's page tables
// =============================================================================

/// The ramp leg alone: the point arrives by hint, the one value is published.
fn ramp_only_program(num_vars: usize, leg: bool) -> LfmProgram {
    let mut b = LfmBuilder::new().with_wrap_hash(super::edsl::WrapHash::production());
    let arena = b.declare_arena(num_vars as u32);
    let point: Vec<_> = (0..num_vars)
        .map(|i| b.hint_word(arena, i as u32).as_ext())
        .collect();
    // Both arms publish ONE word, so the publish cancels out of the delta and
    // what is left is the ramp.
    let out = if leg {
        super::preprocessed::emit_offset_ramp(&mut b, &point)
    } else {
        point[0]
    };
    b.public(out.as_cell());
    let program = compile(b.finish());
    validate(&program).expect("the ramp leg must be admissible");
    program
}

/// ★★ THE GATE: the emitted ramp equals the host's `2^18` FOLD over
/// `page::offset_column()` — the very work this closed form exists to avoid.
///
/// Three derivations meet here and no two of them share an author: the EMITTED
/// leg, `Mle::evaluate_in` over the real column, and
/// `preprocessed::offset_ramp_at`. The fold is the one that matters; the third
/// is there so a disagreement says WHICH of the two claims broke.
#[test]
fn the_offset_ramp_computes_what_the_hosts_page_fold_computes() {
    let column = crate::tables::page::offset_column();
    let num_vars = column.len().trailing_zeros() as usize;
    assert_eq!(
        1usize << num_vars,
        column.len(),
        "the page size must be a power of two for the column to be an MLE"
    );
    let mle = Mle::new(column).expect("a power-of-two column");
    let program = ramp_only_program(num_vars, true);

    for seed in [0x0ff5_e701u64, 0x0ff5_e702, 0x0ff5_e703] {
        let point = sample_point_n(seed, num_vars);
        let arenas = vec![point.iter().map(ext_word).collect::<Vec<_>>()];
        let exec = execute(&program, &arenas, &crate::hash_pin::BLOCK_HASHER)
            .expect("the ramp leg executes");
        assert_eq!(exec.public_words.len(), 1, "the leg publishes one value");
        let got = word_as_ext(&exec.public_words[0].1).expect("an extension value");

        let folded = mle
            .evaluate_in::<GoldilocksExtension>(&point)
            .expect("the column has num_vars variables");
        let closed = super::preprocessed::offset_ramp_at(&point);

        // ⛔ ANTI-VACUITY ON THE ANSWER, not on the inputs (instance 72). The
        // trivial leg — the one a dropped Horner loop leaves behind — returns
        // `point[0]`, and at a random point the true value is not that.
        assert_ne!(
            got, point[0],
            "seed {seed:#x}: the ramp returned its leading coordinate, which is \
             what a leg that emitted nothing would return"
        );
        assert_eq!(
            got, folded,
            "seed {seed:#x}: the EMITTED ramp disagrees with the host's 2^{num_vars} fold"
        );
        assert_eq!(
            got, closed,
            "seed {seed:#x}: the emitted ramp disagrees with the host closed form it mirrors"
        );
    }
}

/// ⛔ THE BIT ORDER IS LOAD-BEARING, AND THIS IS WHAT SAYS SO.
///
/// The convention — index bit `k` carried by `point[num_vars − 1 − k]` — is
/// taken from `emit_bitwise_preprocessed`, not derived here. A test that only
/// compared the emitter against `offset_ramp_at` would agree with itself under
/// EITHER convention, because both are written by the same hand. So the
/// reversed point is evaluated against the REAL fold and must differ: that is
/// the observation which would fail if the convention were the other one.
#[test]
fn the_offset_ramp_is_wrong_under_the_reversed_bit_order() {
    let column = crate::tables::page::offset_column();
    let num_vars = column.len().trailing_zeros() as usize;
    let mle = Mle::new(column).expect("a power-of-two column");
    let point = sample_point_n(0x0ff5_e7a0, num_vars);
    let reversed: Vec<FEE> = point.iter().rev().cloned().collect();

    let folded = mle
        .evaluate_in::<GoldilocksExtension>(&point)
        .expect("the column has num_vars variables");
    assert_eq!(
        super::preprocessed::offset_ramp_at(&point),
        folded,
        "the convention this crate uses must reproduce the host's fold"
    );
    assert_ne!(
        super::preprocessed::offset_ramp_at(&reversed),
        folded,
        "a reversed point must give a DIFFERENT value, or this test cannot see \
         the convention at all"
    );
}

/// ★ F1: `num_vars − 1` rows and ONE interned constant, whatever the page size.
#[test]
fn the_offset_ramp_emits_its_closed_form() {
    for num_vars in [2usize, 5, 18] {
        let with = ramp_only_program(num_vars, true);
        let without = ramp_only_program(num_vars, false);
        let with_consts = const_rows(&with);
        let without_consts = const_rows(&without);
        let measured = (with.instrs.len() - without.instrs.len()) - (with_consts - without_consts);
        let predicted = super::preprocessed::offset_ramp_rows(num_vars);
        println!(
            "offset ramp at {num_vars} vars: {measured} rows emitted, {predicted} predicted, \
             {} constants interned",
            with_consts - without_consts
        );
        assert_eq!(
            measured, predicted,
            "{num_vars} vars: the emitted operation count must equal the closed form"
        );

        // The pool, by value and both ways.
        let named = super::preprocessed::offset_ramp_constants();
        let interned: Vec<super::word::LfmWord> = with
            .instrs
            .iter()
            .filter_map(|instr| match instr {
                super::instr::Instr::Const { value, .. } => Some(*value),
                _ => None,
            })
            .filter(|word| {
                !without.instrs.iter().any(|instr| {
                    matches!(instr, super::instr::Instr::Const { value, .. } if value == word)
                })
            })
            .collect();
        for word in &interned {
            assert!(
                named.contains(word),
                "{num_vars} vars: the ramp interns {word:?}, which the form does not name"
            );
        }
        assert_eq!(
            interned.len(),
            named.len(),
            "{num_vars} vars: the interned pool and the named pool must be the same set"
        );
    }
}

/// ★ THE SIZE OF THE PRIZE, asserted rather than asserted-in-a-comment: the
/// closed form is smaller than the fold it replaces by more than four orders of
/// magnitude at the real page size.
#[test]
fn the_offset_ramp_is_cheaper_than_the_fold_it_replaces() {
    let rows = crate::tables::page::DEFAULT_PAGE_SIZE;
    let num_vars = rows.trailing_zeros() as usize;
    let closed = super::preprocessed::offset_ramp_rows(num_vars);
    println!(
        "OFFSET at {rows} rows: {closed} rows closed-form vs {rows} for a fold \
         ({}x), and the cross-epoch proof carries one per page",
        rows / closed.max(1)
    );
    assert!(
        closed * 10_000 < rows,
        "the ramp costs {closed} rows against a {rows}-row fold, which is not the \
         saving this leg exists for"
    );
}

// =============================================================================
// The SPARSE leg — the cross-epoch proof's page INIT columns
// =============================================================================

/// The sparse leg alone, over one column, publishing its one value.
///
/// The control arm publishes the interned ZERO — which is exactly what the
/// emitter returns for a column with no surviving entry, and therefore exactly
/// what a leg whose coefficient loop emitted nothing would return. Both arms
/// publish ONE word, so the publish cancels out of the delta and what is left is
/// the leg.
fn sparse_only_program(column: &[FE], num_vars: usize, leg: bool) -> LfmProgram {
    let mut b = LfmBuilder::new().with_wrap_hash(super::edsl::WrapHash::production());
    let arena = b.declare_arena(num_vars as u32);
    let point: Vec<_> = (0..num_vars)
        .map(|i| b.hint_word(arena, i as u32).as_ext())
        .collect();
    let out = if leg {
        super::preprocessed::emit_sparse_mle_at(&mut b, &[column], &point)[0]
    } else {
        b.ext_const(&FEE::zero())
    };
    b.public(out.as_cell());
    let program = compile(b.finish());
    validate(&program).expect("the sparse leg must be admissible");
    program
}

/// A real page's genesis column: the bytes an ELF data page loads, zero to the
/// end of the page.
///
/// ★ THE REAL OBJECT, not a hand-built vector. `page::preprocessed_columns` is
/// what `global_memory_air` hands the verifier, column 1 is INIT, and the whole
/// claim of the sparse route is about THAT column's shape — mostly zero, with a
/// short nonzero prefix. A synthetic column could be sparse for reasons the real
/// one is not.
fn real_genesis_column(bytes: &[u8]) -> Vec<FE> {
    let config = crate::tables::page::PageConfig::with_data(0, bytes.to_vec());
    let mut columns = crate::tables::page::preprocessed_columns(&config);
    assert_eq!(columns.len(), 2, "a non-private page carries OFFSET and INIT");
    columns.remove(1)
}

/// ★★ THE SPARSE LEG AGAINST THE FOLD IT REPLACES, AND AGAINST A THIRD
/// DERIVATION — the ramp's own pattern, over a real page's INIT column.
///
/// Three derivations and no two share an author: `Mle::evaluate_in` over the
/// real `2^18` column (the host fold this leg exists to avoid), the host's own
/// [`super::preprocessed::sparse_mle_at`] (the arithmetic being claimed), and
/// the EMITTED program's published value.
#[test]
fn the_sparse_leg_computes_what_the_hosts_genesis_fold_computes() {
    // Eight nonzero bytes with no repeats, so the distinct-coefficient pool is
    // eight and a dropped entry moves the answer.
    let bytes = [0xF0u8, 0xDE, 0xBC, 0x9A, 0x78, 0x56, 0x34, 0x12];
    let column = real_genesis_column(&bytes);
    let num_vars = column.len().trailing_zeros() as usize;
    assert_eq!(1usize << num_vars, column.len(), "a page is a power of two tall");
    let entries = super::preprocessed::sparse_entries(&[column.as_slice()]);
    assert_eq!(
        entries,
        bytes.len(),
        "the support this arm is written against: every genesis byte is nonzero \
         and the rest of the page is zero"
    );

    let mle = Mle::new(column.clone()).expect("a power-of-two column");
    let program = sparse_only_program(&column, num_vars, true);

    for seed in [0x5a17_0001u64, 0x5a17_0002, 0x5a17_0003] {
        let point = sample_point_n(seed, num_vars);
        let arenas = vec![point.iter().map(ext_word).collect::<Vec<_>>()];
        let exec = execute(&program, &arenas, &crate::hash_pin::BLOCK_HASHER)
            .expect("the sparse leg executes");
        assert_eq!(exec.public_words.len(), 1, "the leg publishes one value");
        let got = word_as_ext(&exec.public_words[0].1).expect("an extension value");

        let folded = mle
            .evaluate_in::<GoldilocksExtension>(&point)
            .expect("the column has num_vars variables");
        let closed = super::preprocessed::sparse_mle_at(&column, &point);

        // ⛔ ANTI-VACUITY ON THE ANSWER (instance 72). A leg whose coefficient
        // loop emitted nothing returns the interned ZERO, and at a random point
        // over a column with support the true value is not zero.
        assert_ne!(
            got,
            FEE::zero(),
            "seed {seed:#x}: the sparse leg returned zero, which is what a leg that \
             emitted nothing over a column with {entries} surviving entries would return"
        );
        assert_eq!(
            got, folded,
            "seed {seed:#x}: the EMITTED sparse leg disagrees with the host's \
             2^{num_vars} fold over the same column"
        );
        assert_eq!(
            got, closed,
            "seed {seed:#x}: the emitted leg disagrees with the host closed form it mirrors"
        );
    }
}

/// ⛔ THE BIT ORDER IS OBSERVABLE, so the convention is pinned rather than
/// agreed with itself.
///
/// The emitter's order is taken from `emit_const_mle_at` — `point[0]` binds the
/// HIGH index bit. Reversing the point must give a DIFFERENT value against the
/// real fold; if it did not, no arm of this suite could see the convention
/// reversed and the comment claiming it would be the only thing holding it.
#[test]
fn the_sparse_legs_bit_order_is_observable() {
    let column = real_genesis_column(&[0x01u8, 0x02, 0x03, 0x04]);
    let num_vars = column.len().trailing_zeros() as usize;
    let mle = Mle::new(column.clone()).expect("a power-of-two column");
    let program = sparse_only_program(&column, num_vars, true);

    let point = sample_point_n(0x5a17_0b17, num_vars);
    let reversed: Vec<FEE> = point.iter().rev().cloned().collect();

    let run = |p: &[FEE]| {
        let arenas = vec![p.iter().map(ext_word).collect::<Vec<_>>()];
        let exec = execute(&program, &arenas, &crate::hash_pin::BLOCK_HASHER)
            .expect("the sparse leg executes");
        word_as_ext(&exec.public_words[0].1).expect("an extension value")
    };

    let forward = run(&point);
    let backward = run(&reversed);
    assert_eq!(
        forward,
        mle.evaluate_in::<GoldilocksExtension>(&point)
            .expect("the fold"),
        "the emitted leg is the fold at the point as given"
    );
    assert_ne!(
        forward, backward,
        "the two bit orders agree at this point, so a reversed convention would be \
         invisible here — pick another point"
    );
    assert_ne!(
        backward,
        mle.evaluate_in::<GoldilocksExtension>(&point)
            .expect("the fold"),
        "the REVERSED order must disagree with the fold, which is what makes the \
         convention a checkable fact rather than a shared assumption"
    );
}

/// ★ THE ROW FORM, against the emitter, as a DELTA between the two arms.
///
/// `sparse_mle_rows` is a claim about cost and the value gate above is a claim
/// about arithmetic; a leg can satisfy either alone. The delta cancels the
/// hints, the publish and the compiler's own overhead, so what is left is the
/// leg — and the constant pool is asserted BOTH ways, by count and by value.
#[test]
fn the_sparse_leg_emits_its_row_form() {
    for bytes in [vec![0xABu8, 0xCD, 0xEF], Vec::new(), vec![7u8; 5]] {
        let column = real_genesis_column(&bytes);
        let num_vars = column.len().trailing_zeros() as usize;
        let leg = sparse_only_program(&column, num_vars, true);
        let control = sparse_only_program(&column, num_vars, false);

        let emitted = (leg.instrs.len() - const_rows(&leg))
            - (control.instrs.len() - const_rows(&control));
        let predicted = super::preprocessed::sparse_mle_rows(&[column.as_slice()], num_vars);
        let constants = super::preprocessed::sparse_mle_constants(&[column.as_slice()]);
        println!(
            "sparse leg over {} genesis bytes at {num_vars} vars: {emitted} rows emitted, \
             {predicted} predicted, {} constants named, {} interned",
            bytes.len(),
            constants.len(),
            const_rows(&leg),
        );
        assert_eq!(
            emitted, predicted,
            "the sparse leg's row form missed the emitter over {} genesis bytes",
            bytes.len()
        );
        // ⚠ The control interns the ZERO it publishes, and the leg interns it
        // too when the column is empty — so the pools are compared by the form's
        // own list, which names that zero exactly when the emitter does.
        assert_eq!(
            const_rows(&leg),
            constants.len(),
            "the pool the form names is the pool the emitter interns"
        );
    }
}
