//! A shape bench: one group's lookup argument for the Keccak rounds, at
//! today's table shape against a two-rounds-a-row layout, on the card.
//!
//! ```text
//! cargo test --release -p multilinear --features cuda,parallel --test argue_keccak_shape -- --ignored --nocapture --test-threads=1
//! ```
//!
//! Needs a GPU. The columns are random and the interactions synthetic (one
//! numerator term, three denominator terms each); only the shapes are real.
//! Both arms cover the same permutations, 2^16 rows a table:
//!
//! - `today`: four tables of 1,011 interactions (one tree of 1,024 slots each,
//!   at the batched argue's bin cap, so each table is a bin of its own) and
//!   1,480 columns;
//! - `sliced`: one table of 857 interactions (one tree of 1,024 slots) and
//!   5,461 columns.
//!
//! Each bin's tree is written from the table's resident columns and folded
//! (`trees`), then proved by the batched argue's lockstep ladder (`ladders`),
//! as the batched argue does it. Two warm-ups and seven timed passes an arm,
//! the arms interleaved; medians. `KSHAPE` lines.

#![cfg(feature = "cuda")]

use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use math::field::element::FieldElement;
use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField as Ext;
use math::field::goldilocks::GoldilocksField as F;
use multilinear::constraint_argument::FactorKind;
use multilinear::gkr::FractionTree;
use multilinear::gkr_lockstep::{self as lockstep, LadderTree};
use multilinear::gpu;
use multilinear::logup::{self, Affine, Interaction};
use multilinear::mle::Mle;

type FE = FieldElement<F>;
type EE = FieldElement<Ext>;

const ROWS_LOG2: usize = 16;

fn next(seed: &mut u64) -> u64 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 7;
    *seed ^= *seed << 17;
    *seed
}

/// A table resident on the card and its interactions.
struct Table {
    store: std::sync::Arc<gpu::ResidentColumns>,
    width: usize,
    interactions: Vec<Interaction<Ext>>,
}

fn table(width: usize, count: usize, seed: u64) -> Table {
    let mut seed = seed | 1;
    let rows = 1usize << ROWS_LOG2;
    let columns: Vec<Mle<F>> = (0..width)
        .map(|_| Mle::new((0..rows).map(|_| FE::from(next(&mut seed) >> 1)).collect()).unwrap())
        .collect();
    let ext = |seed: &mut u64| {
        EE::new([
            FE::from(next(seed)),
            FE::from(next(seed)),
            FE::from(next(seed)),
        ])
    };
    let affine = |seed: &mut u64, terms: usize| {
        let terms = (0..terms)
            .map(|_| ((next(seed) as usize) % width, ext(seed)))
            .collect();
        Affine::new(terms, ext(seed))
    };
    let interactions = (0..count)
        .map(|_| Interaction::new(affine(&mut seed, 1), affine(&mut seed, 3)))
        .collect();
    let refs: Vec<&Mle<F>> = columns.iter().collect();
    let store = std::sync::Arc::new(gpu::upload_columns(&refs).expect("the columns reside"));
    Table {
        store,
        width,
        interactions,
    }
}

/// One bin: the table's tree written from its resident columns, then its
/// ladder. Seconds of each.
fn bin(t: &Table) -> (f64, f64) {
    let rows = 1usize << ROWS_LOG2;
    let kinds: Vec<FactorKind> = (0..t.width).map(FactorKind::direct).collect();
    let sources: Vec<_> = kinds.iter().map(FactorKind::source).collect();
    let plan = logup::input_plan(&t.interactions, &sources, rows).expect("a plan");
    let from = std::time::Instant::now();
    let device = gpu::input_layer_tree_from_columns(t.store.clone(), 0, t.width, rows, &plan)
        .expect("the card builds the tree");
    let tree = FractionTree::from_device(device).expect("the tree's output");
    let trees = from.elapsed().as_secs_f64();
    let refs: Vec<&dyn LadderTree<Ext>> = vec![&tree as &dyn LadderTree<Ext>];
    let mut tr = DefaultTranscript::<Ext>::new(b"argue-keccak-shape");
    let from = std::time::Instant::now();
    let (_proof, claims) = lockstep::prove_trees(&refs, &mut tr).expect("the ladder proves");
    let ladders = from.elapsed().as_secs_f64();
    assert_eq!(claims.len(), 1);
    (trees, ladders)
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

#[test]
#[ignore = "a card shape bench: box only"]
fn keccak_round_group_lookup_argument_today_against_sliced() {
    const WARM: usize = 2;
    const TIMED: usize = 7;
    let built = std::time::Instant::now();
    let today: Vec<Table> = (0..4).map(|i| table(1480, 1011, 0x7EA + i)).collect();
    let sliced = vec![table(5461, 857, 0x511CE)];
    println!(
        "KSHAPE SETUP: today 4 tables x 1480 columns x 2^{ROWS_LOG2} rows x 1011 interactions · sliced 1 table x 5461 columns x 2^{ROWS_LOG2} rows x 857 interactions · built and uploaded in {:.1} s",
        built.elapsed().as_secs_f64()
    );
    let arms: [(&str, &[Table]); 2] = [("today", &today), ("sliced", &sliced)];
    let mut times: Vec<Vec<(f64, f64)>> = vec![Vec::new(), Vec::new()];
    for rep in 0..WARM + TIMED {
        // Interleaved, the order flipped every pass.
        let order: [usize; 2] = if rep % 2 == 0 { [0, 1] } else { [1, 0] };
        for &a in &order {
            let (name, tables) = arms[a];
            let (mut trees, mut ladders) = (0.0, 0.0);
            for t in tables {
                let (tr, la) = bin(t);
                trees += tr;
                ladders += la;
            }
            println!(
                "KSHAPE RUN arm={name} rep={rep} warm={} trees_ms={:.3} ladders_ms={:.3} total_ms={:.3}",
                rep < WARM,
                trees * 1e3,
                ladders * 1e3,
                (trees + ladders) * 1e3
            );
            if rep >= WARM {
                times[a].push((trees, ladders));
            }
        }
    }
    let mut totals = [0.0; 2];
    for (a, (name, _)) in arms.iter().enumerate() {
        let trees = median(times[a].iter().map(|t| t.0).collect());
        let ladders = median(times[a].iter().map(|t| t.1).collect());
        let total = median(times[a].iter().map(|t| t.0 + t.1).collect());
        totals[a] = total;
        println!(
            "KSHAPE MEDIAN arm={name} reps={TIMED} trees_ms={:.3} ladders_ms={:.3} total_ms={:.3}",
            trees * 1e3,
            ladders * 1e3,
            total * 1e3
        );
    }
    println!(
        "KSHAPE RATIO sliced/today={:.4} (one group's lookup argument at equal permutations)",
        totals[1] / totals[0]
    );
}
