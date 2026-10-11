//! The two memory kernels end to end on the device: a stacked opening whose
//! first rounds run over its shares (the lean opening) and whose codeword
//! folds run every level in one launch proves the SAME bytes as today's
//! device path (factors materialised up front, one fold launch per level), and
//! the host verifier accepts both.
//!
//! ```text
//! cargo test --release -p multilinear --features cuda --test whir_lean_device
//! ```
//!
//! Four arms: the path before the kernels (lean rounds 0, folds level by
//! level), six lean rounds with the fused fold (`LAMBDA_VM_WHIR_LEAN_ROUNDS=6`,
//! the way back), the defaults (three lean rounds, the fused fold), and the
//! defaults with the first successor encoded from the bound message instead of
//! folded (`LAMBDA_VM_WHIR_ENCODE_SUCCESSOR=1`). Needs a GPU. Each new path is
//! asserted TAKEN in its arms and NOT in the others (the call counters), so a
//! card that declined could not turn this into a comparison of today's path
//! with itself. One `#[test]`: the arms switch process-wide overrides, and
//! parallel tests would race them.
#![cfg(feature = "cuda")]

use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use math::field::element::FieldElement;
use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField as Ext;
use math::field::goldilocks::GoldilocksField as F;
use multilinear::gpu;
use multilinear::mle::Mle;
use multilinear::stacked_eval::{self, Claimed, StackedCommitment, StackedProof};
use multilinear::stacking::StackedLayout;
use multilinear::whir_chain::{
    CapPolicy, ChainConfig, ChainFormat, FirstFold, GrindBits, WhirFolds,
};
use multilinear::whir_hash::{KeccakWhir, RpxWhir, WhirHash};

type FE = FieldElement<F>;
type EE = FieldElement<Ext>;

fn column(num_vars: usize, seed: u64) -> Mle<F> {
    Mle::new(
        (0..(1u64 << num_vars))
            .map(|i| {
                FE::from(
                    i.wrapping_add(seed)
                        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                        .wrapping_add(seed.wrapping_mul(6364136223846793005))
                        >> 11,
                )
            })
            .collect(),
    )
    .unwrap()
}

/// First6 then uniform4, caps on every tree, no grind: a ground proof's nonce
/// is not reproducible across two searches, and the grind is not what moved.
fn config() -> ChainConfig {
    ChainConfig {
        log_blowup: 2,
        log_folding: 4,
        num_queries: 20,
        grind: GrindBits::default(),
        format: ChainFormat {
            cap: CapPolicy::Auto,
            folds: WhirFolds::First(FirstFold::new(6).unwrap()),
            ..ChainFormat::DEFAULT
        },
    }
}

/// Columns of mixed heights in a `2^16` stack — above the device commit and
/// sumcheck thresholds — each claimed at its own point.
struct Case {
    layout: StackedLayout,
    columns: Vec<Mle<F>>,
    points: Vec<Vec<EE>>,
    values: Vec<EE>,
}

fn case() -> Case {
    let heights = [13usize, 13, 13, 12, 12, 10, 8, 6, 3];
    let columns: Vec<Mle<F>> = heights
        .iter()
        .enumerate()
        .map(|(i, &h)| column(h, 7 + 31 * i as u64))
        .collect();
    let layout = StackedLayout::build(&heights, 16).unwrap();
    assert_eq!(layout.num_polys(), 1);
    let points: Vec<Vec<EE>> = heights
        .iter()
        .enumerate()
        .map(|(i, &h)| {
            (0..h)
                .map(|j| EE::from(101 + 13 * i as u64 + j as u64))
                .collect()
        })
        .collect();
    let values = columns
        .iter()
        .zip(&points)
        .map(|(c, z)| c.evaluate_in(z).unwrap())
        .collect();
    Case {
        layout,
        columns,
        points,
        values,
    }
}

/// Which path an arm proves on.
#[derive(Clone, Copy, Debug)]
enum Arm {
    /// No lean rounds, every fold level by level: the path before the kernels.
    Materialised,
    /// Six lean rounds and the fused fold (`LAMBDA_VM_WHIR_LEAN_ROUNDS=6`).
    Six,
    /// Nothing forced: the defaults, three lean rounds and the fused fold.
    Defaults,
    /// The defaults, with each chain's first successor encoded from its bound
    /// message (`LAMBDA_VM_WHIR_ENCODE_SUCCESSOR=1`).
    Encoded,
}

/// One arm: the path before the kernels or the two kernels.
fn prove_arm<H: WhirHash>(
    c: &Case,
    arm: Arm,
    resident: Option<&gpu::ResidentColumns>,
) -> (StackedProof<F, Ext>, Vec<[u8; 32]>) {
    let lean = !matches!(arm, Arm::Materialised);
    gpu::force_encode_successor(Some(matches!(arm, Arm::Encoded)));
    match arm {
        Arm::Materialised => {
            gpu::force_lean_rounds(Some(0));
            gpu::force_fused_fold(Some(false));
        }
        Arm::Six => {
            gpu::force_lean_rounds(Some(6));
            gpu::force_fused_fold(Some(true));
        }
        Arm::Defaults | Arm::Encoded => {
            gpu::force_lean_rounds(None);
            gpu::force_fused_fold(None);
            if std::env::var_os("LAMBDA_VM_WHIR_LEAN_ROUNDS").is_none() {
                assert_eq!(
                    gpu::lean_rounds_in_effect(),
                    3,
                    "the default is three lean rounds"
                );
            }
        }
    }
    let columns = multilinear::stacking::borrow(&c.columns);
    let stacked = StackedCommitment::<F, H>::commit(
        c.layout.clone(),
        &columns,
        resident.map(|store| (store, 0)),
        &config(),
    )
    .unwrap();
    let (lean_before, fused_before) = (gpu::lean_open_calls(), gpu::fused_fold_calls());
    let encoded_before = gpu::encoded_successors();
    let (opens_before, folds_before) = (gpu::open_calls(), gpu::resident_fold_calls());
    let proof = stacked_eval::prove::<F, Ext, _, H, _>(
        &stacked,
        &columns,
        resident.map(|store| (store, 0)),
        &Claimed::PerColumn(&c.points),
        &c.values,
        &config(),
        &mut DefaultTranscript::<Ext, H::Transcript>::new(b"whir-lean-device"),
    )
    .unwrap();
    let (lean_opens, fused_folds) = (
        gpu::lean_open_calls() - lean_before,
        gpu::fused_fold_calls() - fused_before,
    );
    // The totals the production log reads each share against: both arms open
    // and fold on the device, so a share of zero is the path off, never a
    // path that did not run.
    let (opens, folds) = (
        gpu::open_calls() - opens_before,
        gpu::resident_fold_calls() - folds_before,
    );
    assert!(opens > 0, "{}: no opening ran on the device", H::NAME);
    assert!(folds > 0, "{}: no fold ran on the device", H::NAME);
    // Each chain's first successor, and only in its arm: one a chain, and the
    // chain's later folds still fold.
    assert_eq!(
        gpu::encoded_successors() - encoded_before,
        if matches!(arm, Arm::Encoded) {
            opens
        } else {
            0
        },
        "{} {arm:?}: successors encoded",
        H::NAME
    );
    if lean {
        assert!(
            lean_opens > 0,
            "{}: the lean opening was not taken",
            H::NAME
        );
        assert!(fused_folds > 0, "{}: the fused fold was not taken", H::NAME);
        assert_eq!(
            (lean_opens, fused_folds),
            (opens, folds),
            "{}: every opening lean and every fold fused",
            H::NAME
        );
    } else {
        assert_eq!(
            lean_opens,
            0,
            "{}: today's arm took the lean opening",
            H::NAME
        );
        assert_eq!(
            fused_folds,
            0,
            "{}: today's arm took the fused fold",
            H::NAME
        );
    }
    stacked_eval::verify::<F, Ext, _, H>(
        &proof,
        &c.layout,
        &stacked.roots(),
        &Claimed::PerColumn(&c.points),
        &c.values,
        stacked.domain(),
        &config(),
        &mut DefaultTranscript::<Ext, H::Transcript>::new(b"whir-lean-device"),
    )
    .unwrap_or_else(|e| panic!("{}: the host verifier refused the proof: {e:?}", H::NAME));
    (proof, stacked.roots())
}

fn both_arms<H: WhirHash>(c: &Case, resident: Option<&gpu::ResidentColumns>) {
    let (today, today_roots) = prove_arm::<H>(c, Arm::Materialised, resident);
    let today_bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&today).unwrap();
    for arm in [Arm::Six, Arm::Defaults, Arm::Encoded] {
        let (lean, lean_roots) = prove_arm::<H>(c, arm, resident);
        assert_eq!(today_roots, lean_roots, "{} {arm:?}: roots", H::NAME);
        assert_eq!(
            rkyv::to_bytes::<rkyv::rancor::Error>(&lean)
                .unwrap()
                .as_slice(),
            today_bytes.as_slice(),
            "{} ({}, {arm:?}): the kernels must prove today's bytes",
            H::NAME,
            if resident.is_some() {
                "resident"
            } else {
                "parts"
            },
        );
    }
}

#[test]
fn the_memory_kernels_prove_todays_bytes() {
    let c = case();
    both_arms::<RpxWhir>(&c, None);
    both_arms::<KeccakWhir>(&c, None);
    let store = gpu::upload_columns(&multilinear::stacking::borrow(&c.columns))
        .expect("the card took the columns (needs a GPU)");
    both_arms::<RpxWhir>(&c, Some(&store));
    gpu::force_lean_rounds(None);
    gpu::force_fused_fold(None);
    gpu::force_encode_successor(None);
}
