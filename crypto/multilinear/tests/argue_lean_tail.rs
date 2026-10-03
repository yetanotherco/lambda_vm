//! A device GKR layer's host tail, lean against generic, on the card
//! (`LAMBDA_VM_ARGUE_LEAN_TAIL`).
//!
//! ```text
//! cargo test --release -p multilinear --features cuda,parallel --test argue_lean_tail -- --test-threads=1
//! ```
//!
//! Needs a GPU: only a layer a device handed back finishes lean. The arms
//! switch process-wide overrides and read process-wide counters, so every test
//! takes one lock as well.
#![cfg(feature = "cuda")]

use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use crypto::fiat_shamir::is_transcript::IsTranscript;
use math::field::element::FieldElement;
use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField as Ext;
use math::field::goldilocks::GoldilocksField as F;
use multilinear::Error;
use multilinear::gkr::{self, FractionLayer, FractionTree, GkrOutput, GkrProof};
use multilinear::gpu;
use multilinear::mle::Mle;

type FE = FieldElement<F>;
type EE = FieldElement<Ext>;

static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Every override this file sets, put back when the test ends — a failed
/// assertion included, so one red test cannot leave the next one armed.
struct Restore {
    _serial: std::sync::MutexGuard<'static, ()>,
}

impl Drop for Restore {
    fn drop(&mut self) {
        gpu::force_argue_lean_tail(None);
        gpu::force_argue_xcheck(None);
        gkr::force_lean_tail_fault(false);
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

fn transcript() -> DefaultTranscript<Ext> {
    DefaultTranscript::<Ext>::new(b"argue-lean-tail")
}

/// One prove of a fresh tree: the proof, the transcript after it, the layers a
/// device ran, and how many of them finished lean and were cross-checked.
struct Arm {
    output: Result<GkrOutput<Ext>, Error>,
    transcript: DefaultTranscript<Ext>,
    on_card: bool,
    card_layers: u64,
    lean: u64,
    xchecked: u64,
}

fn prove(layer: &FractionLayer<Ext>, lean: bool, xcheck: bool) -> Arm {
    gpu::force_argue_lean_tail(Some(lean));
    gpu::force_argue_xcheck(Some(xcheck));
    // A device tree spends its layers, so every arm builds its own.
    let tree = FractionTree::build(layer.clone()).unwrap();
    let (sessions, lean_before, xchecked_before) = (
        gpu::sumcheck_calls(),
        gpu::lean_tails(),
        gpu::lean_tail_xchecks(),
    );
    let mut t = transcript();
    let output = gkr::prove(&tree, &mut t);
    Arm {
        output,
        transcript: t,
        on_card: tree.device().is_some(),
        // Only the tree's layers run a device sumcheck here, one each.
        card_layers: gpu::sumcheck_calls() - sessions,
        lean: gpu::lean_tails() - lean_before,
        xchecked: gpu::lean_tail_xchecks() - xchecked_before,
    }
}

/// A proof's layers, comparable field element by field element: every round's
/// values and the four each layer is closed with.
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
                    .map(|round| round.evaluations.clone())
                    .collect(),
                [layer.p_lo, layer.p_hi, layer.q_lo, layer.q_hi],
            )
        })
        .collect()
}

fn verify(layer: &FractionLayer<Ext>, proof: &GkrProof<Ext>) -> Result<(), Error> {
    let output = FractionTree::build(layer.clone())?.output();
    gkr::verify(proof, output, &mut transcript()).map(|_| ())
}

/// ★ The stage's identity: a tree on the card proved with every device layer's
/// tail generic and lean gives the same proof — every round value, every
/// layer's four closing values, the claim it leaves — and the same transcript
/// after it. Each arm took the path it is named for: none lean with the knob
/// off, every device layer with it on.
#[test]
fn the_card_layers_finish_lean_to_the_same_proof() {
    let _restore = serial();
    let layer = input(15, 3);
    let mut off = prove(&layer, false, false);
    let mut on = prove(&layer, true, false);
    assert!(
        off.on_card && on.on_card,
        "a 2^15 tree is built on the card"
    );
    let (off_out, on_out) = (off.output.as_ref().unwrap(), on.output.as_ref().unwrap());
    let card = on.card_layers;
    assert!(card > 0, "the tree has layers the device ran");
    assert_eq!(
        off.card_layers, card,
        "both arms ran the same layers on the card"
    );
    assert_eq!(off.lean, 0, "knob off: no tail ran lean");
    assert_eq!(on.lean, card, "knob on: every device layer's tail ran lean");
    assert_eq!(
        layers(&off_out.proof),
        layers(&on_out.proof),
        "a layer's rounds or closing values moved"
    );
    assert_eq!(
        off_out.claim.point, on_out.claim.point,
        "the claim's point moved"
    );
    assert_eq!(
        (&off_out.claim.p, &off_out.claim.q),
        (&on_out.claim.p, &on_out.claim.q),
        "the claim moved"
    );
    assert_eq!(
        off.transcript.state(),
        on.transcript.state(),
        "the transcripts parted"
    );
    assert_eq!(
        off.transcript.sample_field_element(),
        on.transcript.sample_field_element(),
        "the next challenge moved"
    );
    verify(&layer, &off_out.proof).expect("the knob-off proof verifies");
    verify(&layer, &on_out.proof).expect("the knob-on proof verifies");
    eprintln!("argue lean tail: {card} card layers finished lean, the same proof");
}

/// Where the knob stops: a tree under the device's threshold is proved here
/// from the start, and its layers keep the generic rounds.
#[test]
fn the_knob_takes_only_layers_a_device_handed_back() {
    let _restore = serial();
    let layer = input(10, 5);
    let off = prove(&layer, false, false);
    let on = prove(&layer, true, false);
    assert!(!on.on_card, "a 2^10 tree stays on the host");
    assert_eq!(on.lean, 0, "no layer came off a device");
    assert_eq!(
        layers(&off.output.unwrap().proof),
        layers(&on.output.unwrap().proof),
        "the host tree's proof moved"
    );
}

/// ⛔ THE NEGATIVE CONTROLS: with the fault armed, every lean tail adds one to
/// its first round's `s(1)`. The proof must then differ from the generic one
/// and fail at GKR's own check — the first a verifier runs on it — and
/// `LAMBDA_VM_ARGUE_XCHECK` must refuse to prove at all; while without the
/// fault the cross-check proves and says how many layers it compared.
#[test]
fn a_wrong_lean_tail_is_seen_by_every_check() {
    let _restore = serial();
    let layer = input(15, 7);
    let off = prove(&layer, false, false);
    let generic = off.output.unwrap();

    gkr::force_lean_tail_fault(true);
    let faulted = prove(&layer, true, false);
    let crossed = prove(&layer, true, true);
    gkr::force_lean_tail_fault(false);

    let faulted_out = faulted.output.unwrap();
    assert!(faulted.lean > 0, "the fault is on the lean path");
    assert_ne!(
        layers(&faulted_out.proof),
        layers(&generic.proof),
        "a wrong lean round must change the proof"
    );
    let verdict = verify(&layer, &faulted_out.proof);
    assert!(
        matches!(verdict, Err(Error::LayerRelationMismatch { .. })),
        "GKR's layer check must refuse it, got {verdict:?}"
    );
    assert_eq!(
        crossed.output.err(),
        Some(Error::DeviceFailed { stage: "lean tail" }),
        "the cross-check must refuse to prove over a wrong lean round"
    );

    let clean = prove(&layer, true, true);
    let clean_out = clean.output.unwrap();
    assert_eq!(
        clean.xchecked, clean.lean,
        "without the fault the cross-check compares every lean tail"
    );
    assert!(clean.xchecked > 0, "and there is one to compare");
    assert_eq!(layers(&clean_out.proof), layers(&generic.proof));
    eprintln!(
        "argue lean tail: the wrong round was refused ({verdict:?}); {} tails cross-checked",
        clean.xchecked
    );
}
