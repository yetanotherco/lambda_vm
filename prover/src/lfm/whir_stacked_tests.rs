//! Gates for the stacked-evaluation weight.
//!
//! `weight_at` is private to the multilinear crate, so the gate is against the
//! public pair that DEFINES it — `weight_table`, the table the prover actually
//! commits, evaluated at the same point. That is a stronger comparison than the
//! host's own closed form would be: it catches an error the two closed forms
//! could share.

use multilinear::mle::Mle;
use multilinear::stacked_eval::{WeightShare, weight_table};
use multilinear::stacking::StackedLayout;

use crate::tables::types::{FE, FEE};

use super::builder::{Ext, LfmBuilder};
use super::compiler::{LfmProgram, compile};
use super::executor::execute;
use super::validator::validate;
use super::whir_stacked::{ColumnClaim, emit_weight_at, weight_at_consts, weight_at_rows};
use super::word::{LfmWord, ext_word, word_as_ext};

fn fee(v: u64) -> FEE {
    FEE::new([
        FE::from(v.wrapping_mul(6364136223846793005) >> 11),
        FE::from(v ^ 0xA5A5),
        FE::from(v.wrapping_add(7)),
    ])
}

/// One shape under test: a layout, which claimed point each column shares, and
/// the number of variables each of those points carries.
struct Shape {
    name: &'static str,
    layout: StackedLayout,
    group_of: Vec<usize>,
    group_vars: Vec<usize>,
}

fn shape(name: &'static str, heights: &[usize], n_stack: usize, group_of: Vec<usize>) -> Shape {
    let layout = StackedLayout::build(heights, n_stack).expect("the layout packs");
    let groups = group_of.iter().copied().max().map(|g| g + 1).unwrap_or(0);
    let mut group_vars = vec![usize::MAX; groups];
    for (column, &group) in group_of.iter().enumerate() {
        let vars = layout.placements()[column].num_vars;
        if group_vars[group] == usize::MAX {
            group_vars[group] = vars;
        }
        assert_eq!(
            group_vars[group], vars,
            "{name}: columns sharing a point must share a height"
        );
    }
    Shape {
        name,
        layout,
        group_of,
        group_vars,
    }
}

/// The shapes: one polynomial with every column the same height and two
/// distinct points; columns of MIXED heights, so prefix lengths differ inside
/// one polynomial; and a stack that spills into more than one polynomial.
fn shapes() -> Vec<Shape> {
    vec![
        shape(
            "eight equal columns, two points",
            &[3; 8],
            6,
            (0..8).map(|c| c % 2).collect(),
        ),
        shape(
            "mixed heights, one point each",
            &[4, 3, 3, 2, 2, 5],
            6,
            vec![0, 1, 1, 2, 2, 3],
        ),
        shape(
            "twelve columns spilling into two polynomials",
            &[3; 12],
            6,
            (0..12).map(|c| c % 3).collect(),
        ),
        shape("one column filling the stack", &[5, 5], 5, vec![0, 1]),
    ]
}

fn arena_words(shape: &Shape) -> usize {
    shape.layout.n_stack() + shape.group_vars.iter().sum::<usize>() + shape.group_of.len()
}

/// Hints `at`, then each group's point, then every column's weight — the order
/// [`weight_arena`] fills.
fn weight_program(shape: &Shape, poly: usize) -> LfmProgram {
    let mut b = LfmBuilder::new().with_wrap_hash(super::edsl::WrapHash::production());
    let arena = b.declare_arena(arena_words(shape) as u32);
    let mut idx = 0u32;
    let next = |b: &mut LfmBuilder, idx: &mut u32| -> Ext {
        let cell = b.hint_word(arena, *idx).as_ext();
        *idx += 1;
        cell
    };

    let mut at = Vec::with_capacity(shape.layout.n_stack());
    for _ in 0..shape.layout.n_stack() {
        at.push(next(&mut b, &mut idx));
    }
    let mut points: Vec<Vec<Ext>> = Vec::with_capacity(shape.group_vars.len());
    for &vars in &shape.group_vars {
        let mut point = Vec::with_capacity(vars);
        for _ in 0..vars {
            point.push(next(&mut b, &mut idx));
        }
        points.push(point);
    }
    let mut weights = Vec::with_capacity(shape.group_of.len());
    for _ in 0..shape.group_of.len() {
        weights.push(next(&mut b, &mut idx));
    }

    let claims: Vec<ColumnClaim<'_>> = shape
        .group_of
        .iter()
        .enumerate()
        .map(|(column, &group)| ColumnClaim {
            point: &points[group],
            weight: weights[column],
        })
        .collect();

    let weight = emit_weight_at(&mut b, &shape.layout, poly, &claims, &at);
    b.public(weight.as_cell());

    let program = compile(b.finish());
    validate(&program).expect("the weight leg must be admissible");
    program
}

fn weight_arena(at: &[FEE], points: &[Vec<FEE>], weights: &[FEE]) -> Vec<LfmWord> {
    let mut words: Vec<LfmWord> = at.iter().map(ext_word).collect();
    for point in points {
        words.extend(point.iter().map(ext_word));
    }
    words.extend(weights.iter().map(ext_word));
    words
}

/// ★ F1 for the weight, on every shape and every polynomial of it.
#[test]
fn the_weight_leg_emits_its_closed_form() {
    for shape in shapes() {
        for poly in 0..shape.layout.num_polys() {
            let program = weight_program(&shape, poly);
            let measured = program.instrs.len() - arena_words(&shape) - 1;
            let predicted =
                weight_at_rows(&shape.layout, poly, &shape.group_of) + weight_at_consts();
            println!(
                "weight [{}] poly {poly}: {measured:>4} rows emitted, {predicted:>4} predicted",
                shape.name
            );
            assert_eq!(
                measured, predicted,
                "{}: polynomial {poly} must emit its closed form",
                shape.name
            );
        }
    }
}

/// ★ The leg computes what the committed weight table evaluates to.
#[test]
fn the_weight_leg_computes_what_the_host_computes() {
    for shape in shapes() {
        let n_stack = shape.layout.n_stack();
        for seed in [0x31u64, 0x77] {
            let at: Vec<FEE> = (0..n_stack).map(|i| fee(seed + i as u64)).collect();
            let points: Vec<Vec<FEE>> = shape
                .group_vars
                .iter()
                .enumerate()
                .map(|(group, &vars)| {
                    (0..vars)
                        .map(|i| fee(seed * 13 + (group * 100 + i) as u64))
                        .collect()
                })
                .collect();
            let weights: Vec<FEE> = (0..shape.group_of.len())
                .map(|c| fee(seed * 29 + 1 + c as u64))
                .collect();

            for poly in 0..shape.layout.num_polys() {
                let shares: Vec<WeightShare<'_, _>> = shape
                    .layout
                    .placements()
                    .iter()
                    .enumerate()
                    .filter(|(_, place)| place.poly == poly)
                    .map(|(column, place)| WeightShare {
                        offset: place.offset,
                        point: &points[shape.group_of[column]],
                        scale: weights[column],
                    })
                    .collect();
                let table: Mle<_> =
                    weight_table(&shares, n_stack).expect("the weight table builds");
                let want = table
                    .evaluate(&at)
                    .expect("the table takes the stack's point");

                let program = weight_program(&shape, poly);
                let arena = weight_arena(&at, &points, &weights);
                let exec = execute(&program, &[arena], &crate::hash_pin::BLOCK_HASHER)
                    .expect("the weight leg executes");
                let got = word_as_ext(&exec.public_words[0].1).expect("a published weight");
                assert_eq!(
                    got, want,
                    "{}: polynomial {poly} at seed {seed:#x} disagrees with the committed \
                     weight table",
                    shape.name
                );
            }
        }
    }
}

/// ★ `multilinear::challenge_powers` is REACHABLE from this crate, and means
/// what the batching weights need it to mean.
///
/// Without this, item 0's visibility change is a check that cannot fail: a
/// `pub` item with no consumer draws no warning and no test, so "the emitter
/// can gate against it" would be a claim nothing stands behind until the
/// stacked wiring lands. This is the cheapest thing that fails if the
/// visibility is reverted — it would not compile — and it pins the semantics
/// the γ-powers leg will be gated against: `[1, γ, γ², …]`, the FIRST weight
/// one and not γ, which is the end that is easy to get wrong.
#[test]
fn challenge_powers_is_reachable_and_starts_at_one() {
    let gamma = fee(0x51);
    let powers = multilinear::challenge_powers::<
        math::field::extensions_goldilocks::Degree3GoldilocksExtensionField,
    >(&gamma, 4);
    let want = [FEE::one(), gamma, gamma * gamma, gamma * gamma * gamma];
    assert_eq!(powers.len(), want.len(), "one weight per source");
    for (i, (got, expected)) in powers.iter().zip(&want).enumerate() {
        assert_eq!(got, expected, "gamma^{i}");
    }
}
