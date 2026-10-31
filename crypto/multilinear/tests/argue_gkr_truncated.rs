//! Tables' GKR input layers kept on the card without their padding against
//! today's trees (`LAMBDA_VM_ARGUE_INPUT_REAL`, `multilinear::gkr_input_real`).
//!
//! ```text
//! cargo test --release -p multilinear --features cuda,parallel --test argue_gkr_truncated -- --test-threads=1
//! ```
//!
//! Needs a GPU. Each arm builds its trees from the same resident columns under
//! one setting and proves them — one table at a time (`gkr::prove`) and several
//! in lockstep (`gkr_lockstep::prove_trees`, the batched argue's ladder) — and
//! the proofs' bytes, the claims they leave and the transcripts after them are
//! compared with today's. The tests switch process-wide overrides and read
//! process-wide counters, so each takes one lock.
#![cfg(feature = "cuda")]

use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use crypto::fiat_shamir::is_transcript::IsTranscript;
use math::field::element::FieldElement;
use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField as Ext;
use math::field::goldilocks::GoldilocksField as F;
use multilinear::claim_reduce::FactorSource;
use multilinear::constraint_argument::FactorKind;
use multilinear::gkr::{self, FractionTree};
use multilinear::gkr_input_real::{self as input_real, InputReal};
use multilinear::gkr_lockstep::{self as lockstep, LadderTree};
use multilinear::gpu;
use multilinear::logup::{self, Affine, Interaction};
use multilinear::mle::Mle;

type FE = FieldElement<F>;
type EE = FieldElement<Ext>;

static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Every override this file sets, put back when the test ends — a failed
/// assertion included.
struct Restore {
    _serial: std::sync::MutexGuard<'static, ()>,
}

impl Drop for Restore {
    fn drop(&mut self) {
        input_real::force_argue_input_real(None);
        input_real::force_check_fault(false);
        multilinear::gkr_gruen::force_argue_gkr_gruen(None);
        gpu::force_hand_back_for_tests(false);
        gpu::set_hand_back_threshold_for_tests(None);
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

/// A table of `width` random columns over `2^num_vars` rows, and `count`
/// interactions of up to five terms over them (two shifted views too).
struct Table {
    columns: Vec<Mle<F>>,
    kinds: Vec<FactorKind>,
    interactions: Vec<Interaction<Ext>>,
}

impl Table {
    fn rows(&self) -> usize {
        self.columns[0].len()
    }

    fn real(&self) -> usize {
        self.interactions.len() * self.rows()
    }

    fn full(&self) -> usize {
        self.interactions.len().next_power_of_two() * self.rows()
    }
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

fn resident(t: &Table) -> std::sync::Arc<gpu::ResidentColumns> {
    let refs: Vec<&Mle<F>> = t.columns.iter().collect();
    std::sync::Arc::new(gpu::upload_columns(&refs).expect("the columns reside"))
}

/// The table's tree over its resident columns, its input layer written from
/// them under whatever setting is in force.
fn tree_over(t: &Table, store: std::sync::Arc<gpu::ResidentColumns>) -> FractionTree<Ext> {
    let sources: Vec<Option<FactorSource>> = t.kinds.iter().map(FactorKind::source).collect();
    let plan = logup::input_plan(&t.interactions, &sources, t.rows()).expect("a plan");
    let device = gpu::input_layer_tree_from_columns(store, 0, t.columns.len(), t.rows(), &plan)
        .expect("the card builds the tree");
    FractionTree::from_device(device).unwrap()
}

fn device_tree(t: &Table) -> FractionTree<Ext> {
    tree_over(t, resident(t))
}

fn transcript() -> DefaultTranscript<Ext> {
    DefaultTranscript::<Ext>::new(b"argue-gkr-truncated")
}

/// One table proved alone: the proof's bytes, the claim's values, and the
/// transcript's state and next challenge.
type Alone = (Vec<u8>, Vec<EE>, [u8; 32], EE);

fn prove_alone(t: &Table) -> Alone {
    let tree = device_tree(t);
    let mut tr = transcript();
    let out = gkr::prove(&tree, &mut tr).expect("the tree proves");
    let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&out.proof)
        .unwrap()
        .to_vec();
    let mut claim = out.claim.point.clone();
    claim.extend([out.claim.p, out.claim.q]);
    let state = tr.state();
    (bytes, claim, state, tr.sample_field_element())
}

/// Several tables proved in lockstep: the ladder's bytes, every input claim,
/// and the transcript's state and next challenge.
fn prove_ladder(tables: &[Table]) -> Alone {
    let trees: Vec<FractionTree<Ext>> = tables.iter().map(device_tree).collect();
    let ladder: Vec<&dyn LadderTree<Ext>> =
        trees.iter().map(|t| t as &dyn LadderTree<Ext>).collect();
    let mut tr = transcript();
    let (proof, claims) = lockstep::prove_trees(&ladder, &mut tr).expect("the ladder proves");
    let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&proof)
        .unwrap()
        .to_vec();
    let mut values = Vec::new();
    for claim in claims {
        values.extend(claim.point);
        values.extend([claim.p, claim.q]);
    }
    // The verifier takes it, from the outputs and heights alone.
    let outputs: Vec<(EE, EE)> = trees.iter().map(|t| t.output()).collect();
    let heights: Vec<usize> = trees.iter().map(|t| t.input_vars()).collect();
    lockstep::verify(&proof, &outputs, &heights, &mut transcript()).expect("the ladder verifies");
    let state = tr.state();
    (bytes, values, state, tr.sample_field_element())
}

fn assert_same(today: &Alone, arm: &Alone, what: &str) {
    assert_eq!(today.0, arm.0, "{what}: the proof's bytes moved");
    assert_eq!(today.1, arm.1, "{what}: a claim moved");
    assert_eq!(today.2, arm.2, "{what}: the transcripts parted");
    assert_eq!(today.3, arm.3, "{what}: the next challenge moved");
}

/// The tables the tests prove: padded at a quarter, three eighths, a CPU-like
/// twelve of 32 and a Keccak-like 31 of 64; and one with no padding.
fn tables() -> Vec<Table> {
    vec![
        table(12, 6, 3, 1),
        table(13, 9, 5, 2),
        table(11, 20, 20, 3),
        table(12, 10, 8, 4),
        table(10, 40, 33, 5),
    ]
}

/// ★ Each table proved alone, today's trees against kept ones: the same proof
/// bytes, claim and transcript, with Gruen's rounds (the default) and with
/// today's layer rounds (`LAMBDA_VM_ARGUE_GKR_GRUEN=0`), which write a kept
/// layer whole again. Every padded table was kept, and only those.
#[test]
fn kept_layers_prove_todays_tables() {
    let _restore = serial();
    for gruen in [true, false] {
        multilinear::gkr_gruen::force_argue_gkr_gruen(Some(gruen));
        for (n, t) in tables().iter().enumerate() {
            let what = format!(
                "table {n} ({} of {} slots × {} rows), gruen {gruen}",
                t.interactions.len(),
                t.interactions.len().next_power_of_two(),
                t.rows()
            );
            let padded = t.real() < t.full();
            input_real::force_argue_input_real(Some(InputReal::Off));
            let before = input_real::counts();
            let today = prove_alone(t);
            let off = input_real::counts();
            assert_eq!(
                off.kept_trees, before.kept_trees,
                "{what}: off keeps nothing"
            );
            assert_eq!(
                off.padded_trees - before.padded_trees,
                u64::from(padded),
                "{what}: counted padded"
            );

            input_real::force_argue_input_real(Some(InputReal::On));
            let kept = prove_alone(t);
            let on = input_real::counts();
            assert_eq!(
                on.kept_trees - off.kept_trees,
                u64::from(padded),
                "{what}: kept iff padded"
            );
            assert_eq!(
                on.cells_not_stored - off.cells_not_stored,
                (t.full() - t.real()) as u64,
                "{what}: the padding's cells not stored"
            );
            assert_eq!(on.declines, off.declines, "{what}: nothing declined");
            assert_eq!(
                on.rebuilds - off.rebuilds,
                u64::from(padded && !gruen),
                "{what}: today's layer rounds write the kept layer whole again, and only they"
            );
            assert_same(&today, &kept, &what);
            eprintln!(
                "argue gkr truncated: {what}: the same proof ({} bytes)",
                kept.0.len()
            );
        }
    }
}

/// ★ The batched argue's ladder over every table at once, today's trees
/// against kept ones: the same ladder bytes, input claims and transcript, and
/// the verifier takes both.
#[test]
fn kept_layers_prove_todays_ladder() {
    let _restore = serial();
    let tables = tables();
    let padded = tables.iter().filter(|t| t.real() < t.full()).count() as u64;
    input_real::force_argue_input_real(Some(InputReal::Off));
    let today = prove_ladder(&tables);
    input_real::force_argue_input_real(Some(InputReal::On));
    let before = input_real::counts();
    let kept = prove_ladder(&tables);
    let after = input_real::counts();
    assert_eq!(
        after.kept_trees - before.kept_trees,
        padded,
        "every padded tree kept"
    );
    assert_eq!(
        after.rebuilds, before.rebuilds,
        "no kept layer written whole again"
    );
    assert_same(&today, &kept, "the ladder");
    eprintln!(
        "argue gkr truncated: ladder of {} trees ({padded} kept): the same proof ({} bytes)",
        tables.len(),
        kept.0.len()
    );
}

/// `check` runs the whole layer beside every kept one and compares every
/// round and the halves, limb for limb: every kept layer compared, none
/// differing, and the ladder is today's.
#[test]
fn the_check_compares_every_kept_layer() {
    let _restore = serial();
    let tables = tables();
    let padded = tables.iter().filter(|t| t.real() < t.full()).count() as u64;
    input_real::force_argue_input_real(Some(InputReal::Off));
    let today = prove_ladder(&tables);
    input_real::force_argue_input_real(Some(InputReal::Check));
    let before = input_real::counts();
    let checked = prove_ladder(&tables);
    let after = input_real::counts();
    assert_eq!(
        after.kept_trees - before.kept_trees,
        padded,
        "every padded tree kept"
    );
    assert_eq!(
        after.checked - before.checked,
        padded,
        "every kept layer compared"
    );
    assert_eq!(after.differing, before.differing, "none differed");
    assert_eq!(after.check_skipped, before.check_skipped, "none skipped");
    assert_same(&today, &checked, "the checked ladder");
}

/// ⛔ The check can fail: with the fault armed each kept layer's first sums
/// are one limb off. Every kept layer is then counted as differing and
/// finished on the whole layer — and the ladder is still today's.
#[test]
fn a_kept_layer_that_differs_is_counted_and_finished_whole() {
    let _restore = serial();
    let tables = tables();
    let padded = tables.iter().filter(|t| t.real() < t.full()).count() as u64;
    input_real::force_argue_input_real(Some(InputReal::Off));
    let today = prove_ladder(&tables);
    input_real::force_argue_input_real(Some(InputReal::Check));
    input_real::force_check_fault(true);
    let before = input_real::counts();
    let faulted = prove_ladder(&tables);
    input_real::force_check_fault(false);
    let after = input_real::counts();
    assert_eq!(
        after.checked - before.checked,
        padded,
        "every kept layer compared"
    );
    assert_eq!(
        after.differing - before.differing,
        padded,
        "every one seen to differ"
    );
    assert_same(&today, &faulted, "the ladder finished on the whole layers");
}

/// The room a kept tree promises: its layer and the levels above it
/// (`math_cuda::gkr::truncated_peak_bytes`), against today's carry of four
/// times the whole layer — the promise the argue's high water is made of.
#[test]
fn a_kept_tree_promises_its_layer_and_the_levels_above() {
    let _restore = serial();
    let be = math_cuda::device::backend().expect("a device");
    let t = table(14, 12, 20, 7);
    let (real, full) = (t.real(), t.full());
    // The columns reside first, so the promise is the tree's alone.
    let store = resident(&t);
    let promised = |setting| {
        input_real::force_argue_input_real(Some(setting));
        let before = be.reserved_bytes();
        let tree = tree_over(&t, store.clone());
        let held = be.reserved_bytes() - before;
        drop(tree);
        held
    };
    let today = promised(InputReal::Off);
    let kept = promised(InputReal::On);
    assert_eq!(today, (4 * full) as u64 * 24, "today's carry");
    assert_eq!(
        kept,
        math_cuda::gkr::truncated_peak_bytes(real, full),
        "the kept tree's promise"
    );
    eprintln!(
        "argue gkr truncated: promise today {} MiB, kept {} MiB ({} of {} slots)",
        today >> 20,
        kept >> 20,
        t.interactions.len(),
        t.interactions.len().next_power_of_two()
    );
}

/// ⛔ When the card will not promise the kept tree it builds today's — here
/// the tree that hands its layer back, which promises less — counts the
/// decline, and the proof is today's.
#[test]
fn a_kept_tree_the_card_declines_builds_todays() {
    let _restore = serial();
    let be = math_cuda::device::backend().expect("a device");
    let t = table(14, 12, 20, 8);
    let (real, full) = (t.real(), t.full());
    input_real::force_argue_input_real(Some(InputReal::Off));
    let today = prove_alone(&t);

    // Room for the handed-back tree and not for the kept one: they differ by
    // `24·full` bytes.
    gpu::set_hand_back_threshold_for_tests(Some(0));
    input_real::force_argue_input_real(Some(InputReal::On));
    let room = math_cuda::gkr::padded_peak_bytes(real, full) + (1 << 20);
    assert!(room < math_cuda::gkr::truncated_peak_bytes(real, full));
    let store = resident(&t);
    let before = input_real::counts();
    let free = be.vram_budget_bytes().saturating_sub(be.reserved_bytes());
    let held = math_cuda::device::reserve(free - room).expect("the rest of the budget");
    let tree = tree_over(&t, store);
    drop(held);
    let after = input_real::counts();
    assert_eq!(after.kept_trees, before.kept_trees, "not kept");
    assert_eq!(after.declines - before.declines, 1, "the decline counted");
    let mut tr = transcript();
    let out = gkr::prove(&tree, &mut tr).expect("today's tree proves");
    let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&out.proof)
        .unwrap()
        .to_vec();
    assert_eq!(bytes, today.0, "the proof is today's");
    assert_eq!(tr.state(), today.2, "the transcript is today's");
}
