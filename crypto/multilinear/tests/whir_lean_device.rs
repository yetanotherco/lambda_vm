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
//! Needs a GPU. Both new paths are asserted TAKEN in the one arm and NOT in the
//! other (the call counters), so a card that declined could not turn this into
//! a comparison of today's path with itself. One `#[test]`: the arms switch a
//! process-wide override, and parallel tests would race it.
#![cfg(feature = "cuda")]

use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use math::field::element::FieldElement;
use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField as Ext;
use math::field::goldilocks::GoldilocksField as F;
use multilinear::gpu;
use multilinear::mle::Mle;
use multilinear::stacked_eval::{
    self, Claimed, ColumnsAt, CommitOod, StackedCommitment, StackedProof,
};
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
    case_of(&[13, 13, 13, 12, 12, 10, 8, 6, 3], 16)
}

/// A taller stack whose first run of columns is over the device evaluation's
/// threshold, so its commit-time answers are evaluated on the card in place.
fn ood_case() -> Case {
    case_of(&[15, 15, 15, 14, 12, 9], 17)
}

fn case_of(heights: &[usize], n_stack: usize) -> Case {
    let columns: Vec<Mle<F>> = heights
        .iter()
        .enumerate()
        .map(|(i, &h)| column(h, 7 + 31 * i as u64))
        .collect();
    let layout = StackedLayout::build(heights, n_stack).unwrap();
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

/// One arm: today's device path (`lean = false`) or the two kernels — with
/// the chains carrying commit-time out-of-domain answers when `ood` (I-WOOD):
/// the materialised arm adds the overlay with `add_scaled_eq`, the lean one
/// reads it as one more share row.
fn prove_arm_ood<H: WhirHash>(
    c: &Case,
    lean: bool,
    resident: Option<&gpu::ResidentColumns>,
    ood: bool,
) -> (StackedProof<F, Ext>, Vec<[u8; 32]>) {
    gpu::force_lean_rounds(Some(if lean { 6 } else { 0 }));
    gpu::force_fused_fold(Some(lean));
    let columns = multilinear::stacking::borrow(&c.columns);
    let stacked = StackedCommitment::<F, H>::commit(
        c.layout.clone(),
        &columns,
        resident.map(|store| (store, 0)),
        &config(),
    )
    .unwrap();
    let (lean_before, fused_before) = (gpu::lean_open_calls(), gpu::fused_fold_calls());
    let (opens_before, folds_before) = (gpu::open_calls(), gpu::resident_fold_calls());
    let z0 = EE::from(0x00d_ca11u64) + EE::new([FE::from(3), FE::from(5), FE::from(7)]);
    let answers = if ood {
        let (answers, counts) = stacked_eval::ood_answers::<F, Ext, _>(
            &c.layout,
            &columns,
            resident.map(|store| (store, ColumnsAt::From(0))),
            &z0,
        )
        .unwrap();
        let (host, _) =
            stacked_eval::ood_answers::<F, Ext, _>(&c.layout, &columns, None, &z0).unwrap();
        assert_eq!(
            answers,
            host,
            "{}: the card's answers are the host's",
            H::NAME
        );
        if resident.is_some() {
            assert!(
                counts.on_card > 0,
                "{}: no answer was evaluated on the card",
                H::NAME
            );
        }
        answers
    } else {
        Vec::new()
    };
    let commit_ood = ood.then_some(CommitOod {
        z0: &z0,
        values: &answers,
    });
    let proof = stacked_eval::prove_mapped_ood::<F, Ext, _, H, _>(
        &stacked,
        &columns,
        resident.map(|store| (store, ColumnsAt::From(0))),
        &Claimed::PerColumn(&c.points),
        &c.values,
        commit_ood,
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
    stacked_eval::verify_ood::<F, Ext, _, H>(
        &proof,
        &c.layout,
        &stacked.roots(),
        &Claimed::PerColumn(&c.points),
        &c.values,
        commit_ood,
        stacked.domain(),
        &config(),
        &mut DefaultTranscript::<Ext, H::Transcript>::new(b"whir-lean-device"),
    )
    .unwrap_or_else(|e| panic!("{}: the host verifier refused the proof: {e:?}", H::NAME));
    if ood {
        let mut wrong = answers.clone();
        wrong[0] += EE::one();
        assert!(
            stacked_eval::verify_ood::<F, Ext, _, H>(
                &proof,
                &c.layout,
                &stacked.roots(),
                &Claimed::PerColumn(&c.points),
                &c.values,
                Some(CommitOod {
                    z0: &z0,
                    values: &wrong,
                }),
                stacked.domain(),
                &config(),
                &mut DefaultTranscript::<Ext, H::Transcript>::new(b"whir-lean-device"),
            )
            .is_err(),
            "{}: a false answer verified",
            H::NAME
        );
    }
    (proof, stacked.roots())
}

fn both_arms<H: WhirHash>(c: &Case, resident: Option<&gpu::ResidentColumns>) {
    both_arms_ood::<H>(c, resident, false);
}

fn both_arms_ood<H: WhirHash>(c: &Case, resident: Option<&gpu::ResidentColumns>, ood: bool) {
    let (today, today_roots) = prove_arm_ood::<H>(c, false, resident, ood);
    let (lean, lean_roots) = prove_arm_ood::<H>(c, true, resident, ood);
    assert_eq!(today_roots, lean_roots, "{}: roots", H::NAME);
    assert_eq!(
        rkyv::to_bytes::<rkyv::rancor::Error>(&lean)
            .unwrap()
            .as_slice(),
        rkyv::to_bytes::<rkyv::rancor::Error>(&today)
            .unwrap()
            .as_slice(),
        "{} ({}): the lean opening and the fused fold must prove today's bytes",
        H::NAME,
        if resident.is_some() {
            "resident"
        } else {
            "parts"
        },
    );
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
}

/// The same with commit-time out-of-domain answers (I-WOOD): the lean opening
/// reading the overlay as its row past the columns', and the materialised one
/// adding it with `add_scaled_eq`, prove the same bytes, which the host
/// verifier accepts and refuses with a false answer; the answers evaluated on
/// the card in place equal the host's.
#[test]
fn the_memory_kernels_prove_the_same_bytes_with_commit_answers() {
    let c = ood_case();
    both_arms_ood::<RpxWhir>(&c, None, true);
    let store = gpu::upload_columns(&multilinear::stacking::borrow(&c.columns))
        .expect("the card took the columns (needs a GPU)");
    both_arms_ood::<RpxWhir>(&c, Some(&store), true);
    both_arms_ood::<KeccakWhir>(&c, Some(&store), true);
    gpu::force_lean_rounds(None);
    gpu::force_fused_fold(None);
}
