//! Properties of a block tree's plan: the leaf partition's exact cover and its
//! doubling threshold, the tree shape's coverage, and what the published
//! state word is.

use crate::hash_pin::{BlockFamily, ProofFamily};
use crate::tables::types::{FE, FEE};

use super::builder::LfmBuilder;
use super::compiler::compile;
use super::executor::execute;
use super::per_table_aggregator::tree_shape;
use super::whir_block::{
    BlockPartition, LeafScale, P1_LEAF_HEIGHTS, P1_LEAF_HEIGHTS_2X, scaled_leaf_partition,
};
use super::whir_chain::ChipRows;
use super::whir_epoch::emit_roots_block;
use super::whir_transcript::WhirTranscript;
use super::word::{LfmWord, word_as_ext};

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 33
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n.max(1) as u64) as usize
    }
}

fn random_load(r: &mut Lcg, room: ChipRows, frac_den: usize) -> ChipRows {
    let a = room.to_array();
    ChipRows::from_array(core::array::from_fn(|c| {
        if a[c] == 0 {
            0
        } else if a[c] == usize::MAX {
            r.below(64)
        } else {
            r.below(a[c] / frac_den + 1)
        }
    }))
}

/// The partition under either scale is an exact cover of the groups, every
/// leaf (front included) under the heights it reports, the same on a re-run,
/// and doubled exactly when the single partition reaches the threshold.
#[test]
fn the_partition_is_an_exact_cover_and_doubles_on_an_integer_compare() {
    let mut r = Lcg(0x5eed_0606);
    let front = ChipRows {
        hash: 9_000,
        select: 20_000,
        balu: 30_000,
        xalu: 40_000,
        lanes: 10_000,
        hint: 5_000,
        bitdec: 100,
        consts: 500,
        public: 12,
        accel: 0,
    };
    let room = ChipRows::from_array(core::array::from_fn(|c| {
        P1_LEAF_HEIGHTS.to_array()[c].saturating_sub(front.to_array()[c])
    }));
    let mut doubled_seen = [false; 2];
    for case in 0..400 {
        let n = 1 + r.below(160);
        let den = 2 + r.below(20);
        let loads: Vec<ChipRows> = (0..n).map(|_| random_load(&mut r, room, den)).collect();
        let t = 1 + r.below(14);
        for scale in [LeafScale::Fixed, LeafScale::DoubleFrom(t)] {
            let got =
                scaled_leaf_partition(&loads, front, scale, P1_LEAF_HEIGHTS, P1_LEAF_HEIGHTS_2X);
            let (p, bound) = got.unwrap_or_else(|e| panic!("case {case}: {e}"));
            let again =
                scaled_leaf_partition(&loads, front, scale, P1_LEAF_HEIGHTS, P1_LEAF_HEIGHTS_2X)
                    .expect("re-run");
            assert_eq!(p, again.0, "case {case}: deterministic");
            assert_eq!(bound, again.1);
            BlockPartition::new(p.leaves().to_vec(), n).expect("an exact cover");
            assert_eq!(p.num_units(), n);
            for leaf in p.leaves() {
                let sum = leaf.iter().fold(front, |acc, &g| acc.plus(loads[g]));
                assert!(
                    sum.under(bound.heights),
                    "case {case}: leaf over its heights"
                );
            }
            let want_doubled =
                matches!(scale, LeafScale::DoubleFrom(t) if bound.single_leaves >= t);
            assert_eq!(bound.doubled, want_doubled, "case {case}");
            assert_eq!(
                bound.heights,
                if want_doubled {
                    P1_LEAF_HEIGHTS_2X
                } else {
                    P1_LEAF_HEIGHTS
                }
            );
            doubled_seen[usize::from(bound.doubled)] = true;
        }
    }
    assert_eq!(doubled_seen, [true, true], "both outcomes exercised");
}

/// Every level's arities consume exactly the level below, each in
/// `1..=fan_in`, and the last level is one node: every leaf reaches the top.
#[test]
fn the_tree_shape_covers_every_leaf_and_closes_to_one_node() {
    for n in 2..=400 {
        let shape = tree_shape(n, 3);
        let mut below = n;
        for level in &shape {
            assert!(level.arities.iter().all(|&a| (1..=3).contains(&a)));
            assert_eq!(level.arities.iter().sum::<usize>(), below, "n {n}");
            below = level.arities.len();
        }
        assert_eq!(below, 1, "n {n} closes to one node");
    }
    assert!(
        tree_shape(1, 3).is_empty(),
        "one leaf: the plan adds the top"
    );
}

/// The leaf's published state word is lanes 0..3 of the permutation output the
/// draws come from: its first three lanes are `z`, its fourth `α`'s first.
/// Equal states across leaves therefore pin `z` and `α₀` directly, and `α₁`,
/// `α₂`, `β` only through the sponge's 4-felt truncated-output collision
/// resistance. Two different carried roots give two different states.
#[test]
fn the_state_word_is_the_draw_permutations_first_four_lanes() {
    let run = |roots: &[LfmWord]| -> Vec<LfmWord> {
        let mut b = LfmBuilder::new().with_wrap_hash(<BlockFamily as ProofFamily>::WRAP);
        let arena = b.declare_arena(roots.len() as u32);
        let carried: Vec<_> = (0..roots.len() as u32)
            .map(|i| b.hint_word(arena, i))
            .collect();
        let mut t = WhirTranscript::for_builder(&mut b);
        t.absorb_const_bytes(b"a statement call");
        let (z, alpha, beta) = emit_roots_block(&mut b, &mut t, &carried, &[]);
        let state = t.state(&mut b);
        for e in [z, alpha, beta] {
            b.public(e.as_cell());
        }
        b.public(state);
        let program = compile(b.finish());
        execute(
            &program,
            &[roots.to_vec()],
            &program.hasher(<BlockFamily as ProofFamily>::SOCKET),
        )
        .expect("executes")
        .public_words
        .iter()
        .map(|(_, w)| *w)
        .collect()
    };
    let root = |s: u64| -> LfmWord { core::array::from_fn(|i| FE::from(s * 10 + i as u64)) };
    let a = run(&[root(1), root(2), root(3)]);
    let b = run(&[root(1), root(2), root(4)]);
    for words in [&a, &b] {
        let z: FEE = word_as_ext(&words[0]).expect("z");
        let alpha: FEE = word_as_ext(&words[1]).expect("alpha");
        let state = words[3];
        assert_eq!(&state[..3], z.value(), "state lanes 0..3 are z");
        assert_eq!(state[3], alpha.value()[0], "state lane 3 is alpha's first");
    }
    assert_ne!(a[3], b[3], "another carried root, another state");
    assert_ne!(a[2], b[2], "and another beta");
}
