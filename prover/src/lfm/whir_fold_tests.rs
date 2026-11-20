//! Gates for the coset fold.

use multilinear::whir::Domain;
use multilinear::whir_commit::fold_coset;

use crate::tables::types::{FE, FEE, GoldilocksExtension, GoldilocksField};

use super::builder::{Bit, Ext, LfmBuilder};
use super::compiler::{LfmProgram, compile};
use super::executor::execute;
use super::validator::validate;
use super::whir_fold::{
    emit_fold_coset_lean, fold_coset_constants_lean, fold_coset_rows_lean, lean_prepare_rows,
    prepare_lean_alphas,
};
use super::word::{LfmWord, ext_word, word_as_ext};

type F = GoldilocksField;
type E = GoldilocksExtension;

fn fee(v: u64) -> FEE {
    FEE::new([
        FE::from(v.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 7),
        FE::from(v ^ 0x1234),
        FE::from(v.wrapping_add(5)),
    ])
}

fn fe(v: u64) -> FE {
    FE::from(v.wrapping_mul(0xD1B5_4A32_D192_ED03) >> 5)
}

/// Rows the plumbing contributes and the leg does not: the hints, the one
/// `BitDec`, and the public. Counted by construction.
fn fold_plumbing(block: usize, levels: usize) -> usize {
    (block + levels + 1) + 1 + 1
}

fn fold_arena(values: &[LfmWord], alphas: &[FEE], index: usize) -> Vec<LfmWord> {
    let mut arena = values.to_vec();
    arena.extend(alphas.iter().map(ext_word));
    arena.push([FE::from(index as u64), FE::zero(), FE::zero(), FE::zero()]);
    arena
}

fn const_rows(program: &LfmProgram) -> usize {
    program
        .instrs
        .iter()
        .filter(|instr| matches!(instr, super::instr::Instr::Const { .. }))
        .count()
}

/// `(log_domain, levels, index_bits)`.
///
/// The first four are the shapes a chain reaches: the index is the BLOCK index,
/// bounded by `domain.size() >> log_folding`, so `index_bits = log_domain −
/// levels` — the production fold of four at two tree sizes, the `S mod k` tail
/// round's fold of one, and a fold of two.
///
/// The fifth is deliberately off that relation. `2·index_bits` is the only term
/// that moves with the index width, so pinning it needs two widths at the SAME
/// block: `(8, 4, 4)` and `(8, 4, 6)` are that pair, and a form that folded the
/// index term into the block term would fit one and miss the other.
///
/// The last two are W2's first folds (`first5`, `first6`): blocks of 32 and 64,
/// on the chain relation, the widest folds the stack runs (`MAX_FOLD`).
const SHAPES: &[(usize, usize, usize)] = &[
    (5, 1, 4),
    (6, 2, 4),
    (8, 4, 4),
    (10, 4, 6),
    (8, 4, 6),
    (12, 5, 7),
    (13, 6, 7),
];

/// The block's values and the folding challenges hinted, the query index's bits
/// hinted as ONE word and decomposed in one `BitDec` — which is exactly what
/// `WhirTranscript::sample_u64_pow2` emits, so the assembled chain hands these
/// bits over rather than rebuilding them — then the round's `½·α` prepared
/// once and one lean fold.
fn lean_fold_program(log_domain: usize, levels: usize, index_bits: usize) -> LfmProgram {
    let block = 1usize << levels;
    let domain = Domain::<F>::new(log_domain).expect("a domain of that size");
    let mut b = LfmBuilder::new().with_wrap_hash(super::edsl::WrapHash::production());
    let arena = b.declare_arena((block + levels + 1) as u32);

    let values: Vec<Ext> = (0..block)
        .map(|i| b.hint_word(arena, i as u32).as_ext())
        .collect();
    let alphas: Vec<Ext> = (0..levels)
        .map(|i| b.hint_word(arena, (block + i) as u32).as_ext())
        .collect();
    let index = b.hint_felt(arena, (block + levels) as u32);
    let bits: Vec<Bit> = b.bit_dec(index, index_bits);

    let prepared = prepare_lean_alphas(&mut b, &alphas);
    let folded = emit_fold_coset_lean(&mut b, &values, &domain, &bits, &prepared);
    b.public(folded.as_cell());
    let program = compile(b.finish());
    validate(&program).expect("the lean fold leg must be admissible");
    program
}

/// ★ The lean fold's F1: rows and interned constants, by count and by value.
#[test]
fn the_lean_fold_emits_its_closed_form() {
    for &(log_domain, levels, index_bits) in SHAPES {
        let block = 1usize << levels;
        let program = lean_fold_program(log_domain, levels, index_bits);
        let constants = const_rows(&program);
        let measured = program.instrs.len()
            - fold_plumbing(block, levels)
            - constants
            - lean_prepare_rows(levels);
        assert_eq!(
            measured,
            fold_coset_rows_lean(block, index_bits),
            "a lean block of {block} on 2^{log_domain} at {index_bits} index bits"
        );
        let named = fold_coset_constants_lean(
            &Domain::<F>::new(log_domain).expect("a domain"),
            levels,
            index_bits,
        );
        let interned: Vec<LfmWord> = program
            .instrs
            .iter()
            .filter_map(|i| match i {
                super::instr::Instr::Const { value, .. } => Some(*value),
                _ => None,
            })
            .collect();
        assert_eq!(named.len(), interned.len(), "a lean block of {block}");
        for word in &named {
            assert!(
                interned.contains(word),
                "a lean block of {block}: named, not interned"
            );
        }
        for word in &interned {
            assert!(
                named.contains(word),
                "a lean block of {block}: interned, not named"
            );
        }
    }
}

/// ★ The lean leg computes what `whir_commit::fold_coset` computes — over
/// extension blocks, over round 0's base blocks, and at the widest indices.
#[test]
fn the_lean_fold_computes_what_the_host_computes() {
    for &(log_domain, levels, index_bits) in SHAPES {
        let block = 1usize << levels;
        let domain = Domain::<F>::new(log_domain).expect("a domain");
        let program = lean_fold_program(log_domain, levels, index_bits);
        for seed in [0x21u64, 0x63] {
            let alphas: Vec<FEE> = (0..levels).map(|i| fee(seed * 91 + 1 + i as u64)).collect();
            let ext_values: Vec<FEE> = (0..block).map(|i| fee(seed * 17 + i as u64)).collect();
            let base_values: Vec<FE> = (0..block).map(|i| fe(seed * 13 + i as u64)).collect();
            for index in [0usize, 1, 3, (1 << index_bits) - 1] {
                let want_ext = fold_coset::<F, E, E>(&ext_values, &domain, index, &alphas)
                    .expect("the host folds the block");
                let words: Vec<LfmWord> = ext_values.iter().map(ext_word).collect();
                let exec = execute(
                    &program,
                    &[fold_arena(&words, &alphas, index)],
                    &crate::hash_pin::BLOCK_HASHER,
                )
                .expect("the lean fold executes");
                assert_eq!(
                    word_as_ext(&exec.public_words[0].1).expect("a value"),
                    want_ext,
                    "ext block {block} on 2^{log_domain}, index {index}, seed {seed:#x}"
                );

                let want_base = fold_coset::<F, F, E>(&base_values, &domain, index, &alphas)
                    .expect("the host folds the base block");
                let words: Vec<LfmWord> = base_values
                    .iter()
                    .map(|v| [*v, FE::zero(), FE::zero(), FE::zero()])
                    .collect();
                let exec = execute(
                    &program,
                    &[fold_arena(&words, &alphas, index)],
                    &crate::hash_pin::BLOCK_HASHER,
                )
                .expect("the lean fold executes over a base block");
                assert_eq!(
                    word_as_ext(&exec.public_words[0].1).expect("a value"),
                    want_base,
                    "base block {block} on 2^{log_domain}, index {index}, seed {seed:#x}"
                );
            }
        }
    }
    // The deepest fold at indices spanning the whole domain.
    let (log_domain, levels, index_bits) = (6usize, 4usize, 6usize);
    let block = 1usize << levels;
    let domain = Domain::<F>::new(log_domain).expect("a domain");
    let program = lean_fold_program(log_domain, levels, index_bits);
    let values: Vec<FEE> = (0..block).map(|i| fee(0x99 + i as u64)).collect();
    let alphas: Vec<FEE> = (0..levels).map(|i| fee(0x5150 + i as u64)).collect();
    let words: Vec<LfmWord> = values.iter().map(ext_word).collect();
    let half = 1usize << (log_domain - 1);
    for index in [half - 1, half, half + 1, (1 << log_domain) - 1] {
        let want = fold_coset::<F, E, E>(&values, &domain, index, &alphas).expect("host");
        let exec = execute(
            &program,
            &[fold_arena(&words, &alphas, index)],
            &crate::hash_pin::BLOCK_HASHER,
        )
        .expect("the lean fold executes");
        assert_eq!(
            word_as_ext(&exec.public_words[0].1).expect("a value"),
            want,
            "the lean point chain at index {index}"
        );
    }
}

/// ★ Load-bearing, not incidental: one changed opened value or one changed
/// challenge moves the lean fold's output off the host's fold of the originals.
#[test]
fn the_lean_fold_moves_with_every_input() {
    let (log_domain, levels, index_bits) = (8usize, 4usize, 4usize);
    let block = 1usize << levels;
    let domain = Domain::<F>::new(log_domain).expect("a domain");
    let program = lean_fold_program(log_domain, levels, index_bits);
    let values: Vec<FEE> = (0..block).map(|i| fee(0x777 + i as u64)).collect();
    let alphas: Vec<FEE> = (0..levels).map(|i| fee(0x888 + i as u64)).collect();
    let index = 5usize;
    let honest = fold_coset::<F, E, E>(&values, &domain, index, &alphas).expect("host");
    let run = |values: &[FEE], alphas: &[FEE]| -> FEE {
        let words: Vec<LfmWord> = values.iter().map(ext_word).collect();
        let exec = execute(
            &program,
            &[fold_arena(&words, alphas, index)],
            &crate::hash_pin::BLOCK_HASHER,
        )
        .expect("the lean fold executes");
        word_as_ext(&exec.public_words[0].1).expect("a value")
    };
    assert_eq!(run(&values, &alphas), honest);
    for slot in 0..block {
        let mut tampered = values.clone();
        tampered[slot] += FEE::one();
        assert_ne!(
            run(&tampered, &alphas),
            honest,
            "value {slot} must reach the output"
        );
    }
    for level in 0..levels {
        let mut tampered = alphas.clone();
        tampered[level] += FEE::one();
        assert_ne!(
            run(&values, &tampered),
            honest,
            "alpha {level} must reach the output"
        );
    }
}
