//! Tables' GKR trees proved with the rounds' sums through a slot against
//! today's (`LAMBDA_VM_ARGUE_ROUND_SLOT`, `multilinear::round_slot`), and a
//! bin's trees built without a wait each.
//!
//! ```text
//! cargo test --release -p multilinear --features cuda,parallel --test argue_round_slot -- --test-threads=1
//! ```
//!
//! Needs a GPU. Each arm builds its trees from the same resident columns and
//! proves them — one table at a time and several in lockstep (the batched
//! argue's ladder) — under one setting; the proofs' bytes, the claims and the
//! transcripts after them are compared with today's. The tests switch
//! process-wide overrides and read process-wide counters, so each takes one
//! lock.
#![cfg(feature = "cuda")]

use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use crypto::fiat_shamir::is_transcript::IsTranscript;
use math::field::element::FieldElement;
use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField as Ext;
use math::field::goldilocks::GoldilocksField as F;
use multilinear::claim_reduce::FactorSource;
use multilinear::constraint_argument::FactorKind;
use multilinear::gkr::{self, FractionTree};
use multilinear::gkr_lockstep::{self as lockstep, LadderTree};
use multilinear::gpu;
use multilinear::logup::{self, Affine, Interaction};
use multilinear::mle::Mle;
use multilinear::round_slot::{self as slot, RoundSlotSetting};

type FE = FieldElement<F>;
type EE = FieldElement<Ext>;

static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

struct Restore {
    _serial: std::sync::MutexGuard<'static, ()>,
}

impl Drop for Restore {
    fn drop(&mut self) {
        slot::force_argue_round_slot(None);
        multilinear::gkr_gruen::force_argue_gkr_gruen(None);
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

struct Table {
    columns: Vec<Mle<F>>,
    kinds: Vec<FactorKind>,
    interactions: Vec<Interaction<Ext>>,
}

impl Table {
    fn rows(&self) -> usize {
        self.columns[0].len()
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
    let slots = kinds.len();
    let ext = |seed: &mut u64| {
        EE::new([
            FE::from(next(seed)),
            FE::from(next(seed)),
            FE::from(next(seed)),
        ])
    };
    let affine = |seed: &mut u64| {
        let terms = (0..1 + (next(seed) % 4) as usize)
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

fn plan(t: &Table) -> logup::InputPlan {
    let sources: Vec<Option<FactorSource>> = t.kinds.iter().map(FactorKind::source).collect();
    logup::input_plan(&t.interactions, &sources, t.rows()).expect("a plan")
}

fn device_tree(t: &Table) -> FractionTree<Ext> {
    let device =
        gpu::input_layer_tree_from_columns(resident(t), 0, t.columns.len(), t.rows(), &plan(t))
            .expect("the card builds the tree");
    FractionTree::from_device(device).unwrap()
}

fn transcript() -> DefaultTranscript<Ext> {
    DefaultTranscript::<Ext>::new(b"argue-round-slot")
}

/// A proof's bytes, its claims, and the transcript's state and next challenge.
type Proved = (Vec<u8>, Vec<EE>, [u8; 32], EE);

fn prove_alone(t: &Table) -> Proved {
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

fn prove_ladder(tables: &[Table]) -> Proved {
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
    let outputs: Vec<(EE, EE)> = trees.iter().map(|t| t.output()).collect();
    let heights: Vec<usize> = trees.iter().map(|t| t.input_vars()).collect();
    lockstep::verify(&proof, &outputs, &heights, &mut transcript()).expect("the ladder verifies");
    let state = tr.state();
    (bytes, values, state, tr.sample_field_element())
}

fn assert_same(today: &Proved, arm: &Proved, what: &str) {
    assert_eq!(today.0, arm.0, "{what}: the proof's bytes moved");
    assert_eq!(today.1, arm.1, "{what}: a claim moved");
    assert_eq!(today.2, arm.2, "{what}: the transcripts parted");
    assert_eq!(today.3, arm.3, "{what}: the next challenge moved");
}

fn tables() -> Vec<Table> {
    vec![
        table(12, 6, 3, 1),
        table(13, 9, 5, 2),
        table(11, 20, 20, 3),
        table(12, 10, 8, 4),
    ]
}

/// ★ Proofs with the rounds' sums through a slot are today's, byte for byte —
/// each table alone and every table in lockstep — and under `check` every
/// round is compared and none differs. Under `on` the Gruen layers launch no
/// second launch.
#[test]
fn the_slot_proves_todays_bytes() {
    let _restore = serial();
    let tables = tables();
    slot::force_argue_round_slot(Some(RoundSlotSetting::Off));
    let ladder_today = prove_ladder(&tables);
    let alone_today: Vec<Proved> = tables.iter().map(prove_alone).collect();
    for setting in [RoundSlotSetting::On, RoundSlotSetting::Check] {
        slot::force_argue_round_slot(Some(setting));
        let before = math_cuda::round_slot::counts();
        let ladder = prove_ladder(&tables);
        let alone: Vec<Proved> = tables.iter().map(prove_alone).collect();
        let after = math_cuda::round_slot::counts();
        assert_same(&ladder_today, &ladder, &format!("{setting:?} ladder"));
        for (n, (a, b)) in alone_today.iter().zip(&alone).enumerate() {
            assert_same(a, b, &format!("{setting:?} table {n}"));
        }
        let rounds = after.slot_rounds - before.slot_rounds;
        assert!(rounds > 0, "{setting:?}: rounds went through slots");
        assert_eq!(
            after.declines, before.declines,
            "{setting:?}: no slot declined"
        );
        match setting {
            RoundSlotSetting::On => assert_eq!(
                after.second_launches, before.second_launches,
                "on: no second launch"
            ),
            _ => {
                assert_eq!(
                    after.checked - before.checked,
                    rounds,
                    "check: every round compared"
                );
                assert_eq!(after.mismatches, before.mismatches, "check: none differed");
            }
        }
        eprintln!("argue round slot: {setting:?}: {rounds} rounds through slots, today's bytes");
    }
}

/// The same with today's generic layer rounds (`LAMBDA_VM_ARGUE_GKR_GRUEN=0`):
/// no Gruen layer, so no slot; the bytes are today's.
#[test]
fn without_gruens_rounds_the_slot_changes_nothing() {
    let _restore = serial();
    multilinear::gkr_gruen::force_argue_gkr_gruen(Some(false));
    let tables = tables();
    slot::force_argue_round_slot(Some(RoundSlotSetting::Off));
    let today = prove_ladder(&tables);
    slot::force_argue_round_slot(Some(RoundSlotSetting::On));
    let arm = prove_ladder(&tables);
    assert_same(&today, &arm, "generic ladder");
}

/// A tree built with its output left unread is today's tree: the same output
/// once read, the same proof.
#[test]
fn a_deferred_tree_is_todays_tree() {
    let _restore = serial();
    for t in tables() {
        let eager = device_tree(&t);
        let deferred = gpu::input_layer_tree_from_columns_deferred(
            resident(&t),
            0,
            t.columns.len(),
            t.rows(),
            &plan(&t),
        )
        .expect("the deferred tree");
        let deferred = FractionTree::from_device(deferred).unwrap();
        assert_eq!(eager.output(), deferred.output(), "the output fraction");
        let (mut a, mut b) = (transcript(), transcript());
        let pa = gkr::prove(&eager, &mut a).unwrap();
        let pb = gkr::prove(&deferred, &mut b).unwrap();
        assert_eq!(
            rkyv::to_bytes::<rkyv::rancor::Error>(&pa.proof)
                .unwrap()
                .to_vec(),
            rkyv::to_bytes::<rkyv::rancor::Error>(&pb.proof)
                .unwrap()
                .to_vec(),
            "the proof"
        );
        assert_eq!(a.state(), b.state());
    }
}

/// ⛔ A deferred tree the card will not promise falls to the host and is
/// counted as a GKR tree refusal — unlike a prefetch's, it is the consume
/// path.
#[test]
fn a_deferred_trees_refusal_is_counted() {
    let _restore = serial();
    let be = math_cuda::device::backend().expect("a device");
    let t = table(14, 12, 20, 9);
    let store = resident(&t);
    let p = plan(&t);
    gpu::set_hand_back_threshold_for_tests(Some(0));
    let before = gpu::gkr_tree_refusals();
    let free = be.vram_budget_bytes().saturating_sub(be.reserved_bytes());
    let held = math_cuda::device::reserve(free).expect("the rest of the budget");
    let refused =
        gpu::input_layer_tree_from_columns_deferred(store, 0, t.columns.len(), t.rows(), &p);
    drop(held);
    assert!(refused.is_none(), "with no room the card refuses the tree");
    assert_eq!(gpu::gkr_tree_refusals() - before, 1, "the refusal counted");
}
