//! A ladder whose small device layers run their Gruen rounds in one launch a
//! round (`gkr_gruen::argue_gruen_batch`) against today's, one launch a tree a
//! round, on the card.
//!
//! ```text
//! cargo test --release -p multilinear --features cuda,parallel --test argue_gruen_batch -- --test-threads=1
//! ```
//!
//! Needs a GPU: only trees a device holds run their layers on the card. The
//! arms switch process-wide overrides and read process-wide counters, so every
//! test takes one lock.
#![cfg(feature = "cuda")]

use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use crypto::fiat_shamir::is_transcript::IsTranscript;
use math::field::element::FieldElement;
use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField as Ext;
use math::field::goldilocks::GoldilocksField as F;
use multilinear::gkr::{FractionLayer, FractionTree, GkrClaim};
use multilinear::gkr_gruen as gruen;
use multilinear::gkr_lockstep::{self, LadderTree, LockstepProof};
use multilinear::mle::Mle;

type FE = FieldElement<F>;
type EE = FieldElement<Ext>;

static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

struct Restore {
    _serial: std::sync::MutexGuard<'static, ()>,
}

impl Drop for Restore {
    fn drop(&mut self) {
        gruen::force_argue_gruen_batch(None);
    }
}

fn serial() -> Restore {
    Restore {
        _serial: SERIAL.lock().unwrap_or_else(|e| e.into_inner()),
    }
}

/// A fraction layer with no structure a kernel could accidentally satisfy.
fn input(num_vars: usize, seed: u64) -> FractionLayer<Ext> {
    let value = |i: u64, salt: u64| {
        let x = i
            .wrapping_add(seed)
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .wrapping_add(salt.wrapping_mul(6364136223846793005));
        EE::new([FE::from(x >> 11), FE::from(x >> 7), FE::from(x >> 3)])
    };
    let size = 1u64 << num_vars;
    let p = (0..size).map(|i| value(i, 1)).collect();
    let q = (0..size).map(|i| value(i, 2)).collect();
    FractionLayer::new(Mle::new(p).unwrap(), Mle::new(q).unwrap()).unwrap()
}

/// One ladder over fresh trees (a device tree spends its layers): the proof,
/// the input claims, the next challenge, and the batched launches it made.
#[allow(clippy::type_complexity)]
fn ladder(
    inputs: &[FractionLayer<Ext>],
    batch: bool,
) -> (
    LockstepProof<Ext>,
    Vec<GkrClaim<Ext>>,
    EE,
    (u64, u64),
    usize,
) {
    gruen::force_argue_gruen_batch(Some(batch));
    let trees: Vec<FractionTree<Ext>> = inputs
        .iter()
        .map(|layer| FractionTree::build(layer.clone()).unwrap())
        .collect();
    let on_card = trees.iter().filter(|t| t.device().is_some()).count();
    let refs: Vec<&dyn LadderTree<Ext>> = trees.iter().map(|t| t as &dyn LadderTree<Ext>).collect();
    let before = gruen::batch_counts();
    let mut t = DefaultTranscript::<Ext>::new(b"argue-gruen-batch");
    let (proof, claims) = gkr_lockstep::prove_trees(&refs, &mut t).unwrap();
    let after = gruen::batch_counts();
    let next: EE = t.sample_field_element();
    (
        proof,
        claims,
        next,
        (after.0 - before.0, after.1 - before.1),
        on_card,
    )
}

/// ★ Twelve small trees (2^14 to 2^16 input cells, the many-table group's
/// shape) and one tall one in one ladder: with the small layers' rounds
/// batched the proof, every input claim and the next challenge are today's,
/// field element for field element; the batched arm made batched launches and
/// today's made none.
#[test]
fn batched_small_layers_prove_todays_ladder() {
    let _restore = serial();
    let mut inputs: Vec<FractionLayer<Ext>> = (0..12)
        .map(|k| input(14 + k % 3, 0xBA7C + k as u64))
        .collect();
    inputs.push(input(19, 0x7A11));
    let (today, today_claims, today_next, today_batches, on_card) = ladder(&inputs, false);
    assert_eq!(on_card, inputs.len(), "every tree on the card");
    assert_eq!(today_batches, (0, 0), "today's arm batched");
    let (batched, claims, next, batches, _) = ladder(&inputs, true);
    assert!(
        batches.0 > 0 && batches.1 >= 2 * batches.0,
        "batched launches {batches:?}"
    );
    assert_eq!(format!("{batched:?}"), format!("{today:?}"), "the ladder");
    assert_eq!(claims.len(), today_claims.len());
    for (k, (a, b)) in claims.iter().zip(&today_claims).enumerate() {
        assert_eq!(
            (&a.point, &a.p, &a.q),
            (&b.point, &b.p, &b.q),
            "tree {k}'s input claim"
        );
    }
    assert_eq!(next, today_next, "the transcript after the ladder");
}
