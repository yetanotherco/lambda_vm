//! A table's GKR input layer written from its resident base columns against
//! today's, from the lifted factors, on the card (`LAMBDA_VM_ARGUE_GKR_INPUT`,
//! D-BATCH M1-2).
//!
//! ```text
//! cargo test --release -p multilinear --features cuda,parallel --test argue_gkr_input -- --test-threads=1
//! ```
//!
//! Needs a GPU. The tests switch process-wide overrides, so each takes one lock.
#![cfg(feature = "cuda")]

use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use crypto::fiat_shamir::is_transcript::IsTranscript;
use math::field::element::FieldElement;
use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField as Ext;
use math::field::goldilocks::GoldilocksField as F;
use multilinear::claim_reduce::FactorSource;
use multilinear::constraint_argument::FactorKind;
use multilinear::gkr::{self, FractionTree, GkrProof};
use multilinear::gpu;
use multilinear::logup::{self, Affine, Interaction};
use multilinear::mle::Mle;

type FE = FieldElement<F>;
type EE = FieldElement<Ext>;

static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

struct Restore {
    _serial: std::sync::MutexGuard<'static, ()>,
}

impl Drop for Restore {
    fn drop(&mut self) {
        gpu::force_hand_back_for_tests(false);
    }
}

fn serial() -> Restore {
    Restore {
        _serial: SERIAL.lock().unwrap_or_else(|e| e.into_inner()),
    }
}

fn next(seed: &mut u64) -> u64 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 7;
    *seed ^= *seed << 17;
    *seed
}

/// A table of `width` random columns over `2^num_vars` rows, its factors (every
/// column, then two shifted views), and `count` interactions of up to five
/// terms over them.
struct Table {
    columns: Vec<Mle<F>>,
    kinds: Vec<FactorKind>,
    interactions: Vec<Interaction<Ext>>,
}

fn table(num_vars: usize, width: usize, count: usize, seed: u64) -> Table {
    let mut seed = seed | 1;
    let rows = 1usize << num_vars;
    let columns: Vec<Mle<F>> = (0..width)
        .map(|_| Mle::new((0..rows).map(|_| FE::from(next(&mut seed) >> 1)).collect()).unwrap())
        .collect();
    let mut kinds: Vec<FactorKind> = (0..width).map(FactorKind::direct).collect();
    kinds.push(FactorKind::shifted(0, 1));
    kinds.push(FactorKind::shifted(width - 1, 5));
    let slots = kinds.len();
    let ext = |seed: &mut u64| {
        EE::new([
            FE::from(next(seed)),
            FE::from(next(seed)),
            FE::from(next(seed)),
        ])
    };
    let affine = |seed: &mut u64| {
        let terms = (0..1 + (next(seed) % 5) as usize)
            .map(|_| ((next(seed) as usize) % slots, ext(seed)))
            .collect();
        Affine::new(terms, ext(seed))
    };
    let interactions = (0..count)
        .map(|_| Interaction::new(affine(&mut seed), affine(&mut seed)))
        .collect();
    Table {
        columns,
        kinds,
        interactions,
    }
}

#[allow(clippy::type_complexity)]
fn layers(proof: &GkrProof<Ext>) -> Vec<(Vec<Vec<EE>>, [EE; 4])> {
    proof
        .layers
        .iter()
        .map(|layer| {
            (
                layer
                    .sumcheck
                    .rounds
                    .iter()
                    .map(|r| r.evaluations.clone())
                    .collect(),
                [layer.p_lo, layer.p_hi, layer.q_lo, layer.q_hi],
            )
        })
        .collect()
}

/// A proved tree: its output, its proof and the transcript after it.
type Proved = ((EE, EE), GkrProof<Ext>, DefaultTranscript<Ext>);

/// Proves a tree.
fn prove(tree: &FractionTree<Ext>) -> Proved {
    let mut t = DefaultTranscript::<Ext>::new(b"argue-gkr-input");
    let out = gkr::prove(tree, &mut t).expect("the tree proves");
    (tree.output(), out.proof, t)
}

/// Today's tree and the columns' tree over the same table, both proved.
fn both(t: &Table) -> (Proved, Proved) {
    let refs: Vec<&Mle<F>> = t.columns.iter().collect();
    let store = std::sync::Arc::new(gpu::upload_columns(&refs).expect("the columns reside"));
    let factors =
        gpu::upload_factors_from_columns::<F, Ext, _>(&t.columns, &t.kinds, &[], Some((&store, 0)))
            .expect("the factors lift");
    let today =
        logup::resident_tree(&t.interactions, std::sync::Arc::new(factors)).expect("today's tree");
    let rows = t.columns[0].len();
    let sources: Vec<Option<FactorSource>> = t.kinds.iter().map(FactorKind::source).collect();
    let plan = logup::input_plan(&t.interactions, &sources, rows).expect("a plan");
    let before = gpu::input_columns_trees();
    let device = gpu::input_layer_tree_from_columns(store, 0, t.columns.len(), rows, &plan)
        .expect("the columns' tree");
    assert_eq!(gpu::input_columns_trees(), before + 1, "counted");
    let columns = FractionTree::from_device(device).unwrap();
    (prove(&today), prove(&columns))
}

fn assert_same(t: &Table, what: &str) {
    let ((out_a, proof_a, mut ta), (out_b, proof_b, mut tb)) = both(t);
    assert_eq!(out_a, out_b, "{what}: the bus output moved");
    assert_eq!(layers(&proof_a), layers(&proof_b), "{what}: a layer moved");
    assert_eq!(ta.state(), tb.state(), "{what}: the transcripts parted");
    assert_eq!(
        ta.sample_field_element(),
        tb.sample_field_element(),
        "{what}: the next challenge moved"
    );
    eprintln!(
        "argue gkr input: {what}: {} layers, the same proof",
        proof_b.layers.len()
    );
}

/// ★ The stage's identity: the tree over the input layer written from the
/// columns proves the same output, rounds and transcript as today's, at
/// several shapes, padded and not, carried and handed back (its rewrite).
#[test]
fn the_columns_write_todays_tree() {
    let _restore = serial();
    for (num_vars, width, count, seed) in [
        (12usize, 6usize, 3usize, 1u64),
        (14, 9, 8, 2),
        (16, 20, 13, 3),
        (10, 40, 64, 4),
    ] {
        let t = table(num_vars, width, count, seed);
        assert_same(
            &t,
            &format!("2^{num_vars} rows, {width} columns, {count} interactions, carried"),
        );
        gpu::force_hand_back_for_tests(true);
        assert_same(
            &t,
            &format!("2^{num_vars} rows, {width} columns, {count} interactions, handed back"),
        );
        gpu::force_hand_back_for_tests(false);
    }
}

/// ⛔ The comparison can fail: one interaction's constant off by one on the
/// columns' side changes the output and the proof.
#[test]
fn a_wrong_plan_changes_the_tree() {
    let _restore = serial();
    let t = table(12, 6, 3, 9);
    let refs: Vec<&Mle<F>> = t.columns.iter().collect();
    let store = std::sync::Arc::new(gpu::upload_columns(&refs).expect("the columns reside"));
    let factors =
        gpu::upload_factors_from_columns::<F, Ext, _>(&t.columns, &t.kinds, &[], Some((&store, 0)))
            .expect("the factors lift");
    let today =
        logup::resident_tree(&t.interactions, std::sync::Arc::new(factors)).expect("today's tree");
    let rows = t.columns[0].len();
    let sources: Vec<Option<FactorSource>> = t.kinds.iter().map(FactorKind::source).collect();
    let mut plan = logup::input_plan(&t.interactions, &sources, rows).expect("a plan");
    plan.constants[3] = plan.constants[3].wrapping_add(1);
    let wrong = FractionTree::from_device(
        gpu::input_layer_tree_from_columns(store, 0, t.columns.len(), rows, &plan).unwrap(),
    )
    .unwrap();
    assert_ne!(
        today.output(),
        wrong.output(),
        "a wrong denominator constant must move the output"
    );
}
