//! Gates for the coset fold.

use multilinear::whir::Domain;
use multilinear::whir_commit::fold_coset;

use crate::tables::types::{FE, FEE, GoldilocksExtension, GoldilocksField};

use super::builder::{Bit, Ext, LfmBuilder};
use super::compiler::{LfmProgram, compile};
use super::executor::execute;
use super::validator::validate;
use super::whir_fold::{emit_fold_coset, fold_coset_consts, fold_coset_rows};
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

/// The block's values and the folding challenges hinted, the query index's bits
/// hinted as ONE word and decomposed in one `BitDec` — which is exactly what
/// `WhirTranscript::sample_u64_pow2` emits, so the assembled chain hands these
/// bits over rather than rebuilding them.
fn fold_program(log_domain: usize, levels: usize, index_bits: usize) -> LfmProgram {
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

    let folded = emit_fold_coset(&mut b, &values, &domain, &bits, &alphas);
    b.public(folded.as_cell());
    let program = compile(b.finish());
    validate(&program).expect("the fold leg must be admissible");
    program
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
const SHAPES: &[(usize, usize, usize)] = &[
    (5, 1, 4),
    (6, 2, 4),
    (8, 4, 4),
    (10, 4, 6),
    (8, 4, 6),
];

/// ★ F1 for the fold: every row named, with the interned constants counted
/// separately and pinned in their own right.
#[test]
fn the_fold_emits_its_closed_form() {
    for &(log_domain, levels, index_bits) in SHAPES {
        let block = 1usize << levels;
        let program = fold_program(log_domain, levels, index_bits);
        let constants = const_rows(&program);
        let measured = program.instrs.len() - fold_plumbing(block, levels) - constants;
        let predicted = fold_coset_rows(block, index_bits);
        println!(
            "fold of {block:>2} on 2^{log_domain} at {index_bits} index bits: \
             {measured:>3} rows emitted, {predicted:>3} predicted \
             ({constants} interned constants, {} predicted; {} named)",
            fold_coset_consts(log_domain, levels, index_bits),
            1 + index_bits + 1 + (levels - 1)
        );
        assert_eq!(
            constants,
            fold_coset_consts(log_domain, levels, index_bits),
            "a block of {block} on 2^{log_domain}: the interned constants must be the ones \
             the shape names, and none of them may collide"
        );
        assert_eq!(
            measured, predicted,
            "a block of {block} on 2^{log_domain} at {index_bits} index bits must emit its \
             closed form"
        );
    }
}

/// ★ The leg computes what `whir_commit::fold_coset` computes, over the
/// EXTENSION — every block but round 0's current one.
#[test]
fn the_fold_computes_what_the_host_computes() {
    for &(log_domain, levels, index_bits) in SHAPES {
        let block = 1usize << levels;
        let domain = Domain::<F>::new(log_domain).expect("a domain");
        let program = fold_program(log_domain, levels, index_bits);
        for seed in [0x21u64, 0x63] {
            let values: Vec<FEE> = (0..block).map(|i| fee(seed * 17 + i as u64)).collect();
            let alphas: Vec<FEE> = (0..levels).map(|i| fee(seed * 91 + 1 + i as u64)).collect();
            for index in [0usize, 1, 3, (1 << index_bits) - 1] {
                let want = fold_coset::<F, E, E>(&values, &domain, index, &alphas)
                    .expect("the host folds the block");

                let words: Vec<LfmWord> = values.iter().map(ext_word).collect();
                let exec = execute(
                    &program,
                    &[fold_arena(&words, &alphas, index)],
                    &crate::hash_pin::BLOCK_HASHER,
                )
                .expect("the fold leg executes");
                let got = word_as_ext(&exec.public_words[0].1).expect("a published value");
                assert_eq!(
                    got, want,
                    "block {block} on 2^{log_domain}, index {index}, seed {seed:#x}: \
                     the emitted leg and the host disagree"
                );
            }
        }
    }
}

/// ★ ROUND 0: the same leg over a BASE block, against the host's own base
/// instantiation.
///
/// `fold_coset::<F, F, E>` is what `whir_round::verify` calls for
/// `RoundOpenings::Base`, and it folds the first level in the base field before
/// the multiply by `α` lifts it. The machine folds every level in the
/// extension. This is what says those are the same value and not merely the
/// same shape — the one gate that a cost-only argument about lifting could not
/// stand in for.
#[test]
fn the_fold_over_a_base_block_agrees_with_the_hosts_base_instantiation() {
    for &(log_domain, levels, index_bits) in SHAPES {
        let block = 1usize << levels;
        let domain = Domain::<F>::new(log_domain).expect("a domain");
        let program = fold_program(log_domain, levels, index_bits);
        for seed in [0x11u64, 0x2F] {
            let values: Vec<FE> = (0..block).map(|i| fe(seed * 13 + i as u64)).collect();
            let alphas: Vec<FEE> = (0..levels).map(|i| fee(seed * 71 + 1 + i as u64)).collect();
            for index in [0usize, 2, (1 << index_bits) - 1] {
                let want = fold_coset::<F, F, E>(&values, &domain, index, &alphas)
                    .expect("the host folds the base block");

                // A base opening arrives as `(v, 0, 0, 0)`; the leaf's `Pack` is
                // what pins it there (see `whir_open`'s module doc).
                let words: Vec<LfmWord> = values
                    .iter()
                    .map(|v| [*v, FE::zero(), FE::zero(), FE::zero()])
                    .collect();
                let exec = execute(
                    &program,
                    &[fold_arena(&words, &alphas, index)],
                    &crate::hash_pin::BLOCK_HASHER,
                )
                .expect("the fold leg executes over a base block");
                let got = word_as_ext(&exec.public_words[0].1).expect("a published value");
                assert_eq!(
                    got, want,
                    "base block {block} on 2^{log_domain}, index {index}, seed {seed:#x}: \
                     the lifted leg and the host's base fold disagree"
                );
            }
        }
    }
}

/// ★ The point chain at the widest indices and the deepest fold: one
/// `pow_bits` and a squaring a level, driven where an exponent could wrap.
///
/// ⚠ What this does NOT test, corrected from the claim it was written under:
/// the host's `position %= current_domain.size()` is a no-op for the value,
/// because each level's generator is a primitive root of exactly that order, so
/// `g_l^p = g_l^(p mod n_l)` identically. There is no "reduction the emitter
/// must survive" — the squaring identity is just `(g²)^p = (g^p)²`. What is
/// left is still worth running, and is what the name should say: the deepest
/// fold this lane tests, at indices spanning the whole domain rather than the
/// small ones the loops above use, where `pow_bits` is widest and the squaring
/// chain is longest.
#[test]
fn the_point_chain_holds_at_the_widest_indices() {
    // 2^6 domain, four levels: the position is reduced at every level after the
    // first, and an index in the top half of the domain wraps at level one.
    let (log_domain, levels, index_bits) = (6usize, 4usize, 6usize);
    let block = 1usize << levels;
    let domain = Domain::<F>::new(log_domain).expect("a domain");
    let program = fold_program(log_domain, levels, index_bits);
    let values: Vec<FEE> = (0..block).map(|i| fee(0x99 + i as u64)).collect();
    let alphas: Vec<FEE> = (0..levels).map(|i| fee(0x5150 + i as u64)).collect();
    let words: Vec<LfmWord> = values.iter().map(ext_word).collect();

    let half = 1usize << (log_domain - 1);
    let mut top_half = 0;
    for index in [half - 1, half, half + 1, (1 << log_domain) - 1] {
        if index >= half {
            top_half += 1;
        }
        let want =
            fold_coset::<F, E, E>(&values, &domain, index, &alphas).expect("the host folds");
        let exec = execute(
            &program,
            &[fold_arena(&words, &alphas, index)],
            &crate::hash_pin::BLOCK_HASHER,
        )
        .expect("the fold leg executes");
        let got = word_as_ext(&exec.public_words[0].1).expect("a published value");
        assert_eq!(
            got, want,
            "index {index}, the deepest fold at the widest index: the squaring chain and \
             pow_bits must together reach the host's point"
        );
    }
    assert!(
        top_half >= 3,
        "this test is about indices in the domain's top half, where pow_bits uses its \
         highest factors; only {top_half} of them were"
    );
}
