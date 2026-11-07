//! A device GKR layer's rounds with Gruen's split against today's, on the card
//! (`LAMBDA_VM_ARGUE_GKR_GRUEN`, D-ARGUE S1-3).
//!
//! ```text
//! cargo test --release -p multilinear --features cuda,parallel --test argue_gkr_gruen -- --test-threads=1
//! ```
//!
//! Needs a GPU: only a layer a device holds runs the Gruen rounds. The arms
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
use multilinear::gkr_gruen as gruen;
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
        gruen::force_argue_gkr_gruen(None);
        gruen::force_argue_gkr_gruen_xcheck(None);
        gruen::force_gkr_gruen_tail(None);
        gruen::force_gkr_gruen_fault(false);
        gruen::force_gkr_gruen_direct_h0(false);
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
    DefaultTranscript::<Ext>::new(b"argue-gkr-gruen")
}

/// One prove of a fresh tree: the proof, the transcript after it, how long it
/// took, and how many layers ran Gruen's rounds and were cross-checked.
struct Arm {
    output: Result<GkrOutput<Ext>, Error>,
    transcript: DefaultTranscript<Ext>,
    on_card: bool,
    gruen: u64,
    xchecked: u64,
    secs: f64,
}

fn prove(layer: &FractionLayer<Ext>, on: bool, xcheck: bool, tail: usize) -> Arm {
    gruen::force_argue_gkr_gruen(Some(on));
    gruen::force_argue_gkr_gruen_xcheck(Some(xcheck));
    gruen::force_gkr_gruen_tail(Some(tail));
    // A device tree spends its layers, so every arm builds its own.
    let tree = FractionTree::build(layer.clone()).unwrap();
    let (gruen_before, xchecked_before) = (gruen::gruen_layers(), gruen::gruen_xchecked());
    let mut t = transcript();
    let start = std::time::Instant::now();
    let output = gkr::prove(&tree, &mut t);
    let secs = start.elapsed().as_secs_f64();
    Arm {
        output,
        transcript: t,
        on_card: tree.device().is_some(),
        gruen: gruen::gruen_layers() - gruen_before,
        xchecked: gruen::gruen_xchecked() - xchecked_before,
        secs,
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

/// The two arms' proofs, claims, transcripts and next challenges are equal.
fn assert_same_proof(today: &mut Arm, arm: &mut Arm, what: &str) {
    let (a, b) = (today.output.as_ref().unwrap(), arm.output.as_ref().unwrap());
    assert_eq!(
        layers(&a.proof),
        layers(&b.proof),
        "{what}: a layer's rounds or closing values moved"
    );
    assert_eq!(
        a.claim.point, b.claim.point,
        "{what}: the claim's point moved"
    );
    assert_eq!(
        (&a.claim.p, &a.claim.q),
        (&b.claim.p, &b.claim.q),
        "{what}: the claim moved"
    );
    assert_eq!(
        today.transcript.state(),
        arm.transcript.state(),
        "{what}: the transcripts parted"
    );
    assert_eq!(
        today.transcript.sample_field_element(),
        arm.transcript.sample_field_element(),
        "{what}: the next challenge moved"
    );
}

/// ★ The stage's identity: a tree on the card proved with today's layer rounds
/// and with Gruen's gives the same proof — every round value, every layer's
/// four closing values, the claim it leaves — and the same transcript after
/// it, at every tail cube and with `H(0)` summed on the card or taken from the
/// claim. The Gruen arm ran every device layer that way; today's none.
#[test]
fn the_card_layers_prove_todays_tree_with_gruens_rounds() {
    let _restore = serial();
    for (vars, seed) in [(14usize, 3u64), (16, 5), (20, 9)] {
        let layer = input(vars, seed);
        for tail in [512usize, 64, 32, 2] {
            for direct in [false, true] {
                if direct && tail != 64 {
                    continue;
                }
                let what = format!("2^{vars}, tail {tail}, direct H(0) {direct}");
                let mut today = prove(&layer, false, false, tail);
                gruen::force_gkr_gruen_direct_h0(direct);
                let mut arm = prove(&layer, true, false, tail);
                gruen::force_gkr_gruen_direct_h0(false);
                assert!(
                    today.on_card && arm.on_card,
                    "{what}: the tree is built on the card"
                );
                assert_eq!(today.gruen, 0, "{what}: today's arm ran no Gruen layer");
                assert!(
                    arm.gruen > 0,
                    "{what}: the Gruen arm ran its layers on the card"
                );
                assert_same_proof(&mut today, &mut arm, &what);
                verify(&layer, &arm.output.as_ref().unwrap().proof)
                    .unwrap_or_else(|e| panic!("{what}: {e:?}"));
                eprintln!(
                    "argue gkr gruen: {what}: {} layers, the same proof · today {:.1} ms, gruen {:.1} ms",
                    arm.gruen,
                    today.secs * 1e3,
                    arm.secs * 1e3
                );
            }
        }
    }
}

/// The cross-check compares every Gruen layer with today's rounds on the same
/// halves, and proves the same proof while it does.
#[test]
fn the_cross_check_compares_every_gruen_layer() {
    let _restore = serial();
    let layer = input(18, 11);
    let mut today = prove(&layer, false, false, 64);
    let mut crossed = prove(&layer, true, true, 64);
    assert!(crossed.gruen > 0, "there are Gruen layers to compare");
    assert_eq!(
        crossed.xchecked, crossed.gruen,
        "every Gruen layer was cross-checked"
    );
    assert_same_proof(&mut today, &mut crossed, "cross-checked");
    eprintln!(
        "argue gkr gruen: {} layers cross-checked, the same proof",
        crossed.xchecked
    );
}

/// ⛔ THE NEGATIVE CONTROLS: with the fault armed, every Gruen layer adds one
/// to its first round's `s(1)`. The proof must then differ from today's and
/// fail at GKR's own layer check, and the cross-check must refuse to prove at
/// all.
#[test]
fn a_wrong_gruen_round_is_seen_by_every_check() {
    let _restore = serial();
    let layer = input(16, 7);
    let today = prove(&layer, false, false, 64);
    let today_out = today.output.unwrap();

    gruen::force_gkr_gruen_fault(true);
    let faulted = prove(&layer, true, false, 64);
    let crossed = prove(&layer, true, true, 64);
    gruen::force_gkr_gruen_fault(false);

    let faulted_out = faulted.output.unwrap();
    assert!(faulted.gruen > 0, "the fault is on the Gruen path");
    assert_ne!(
        layers(&faulted_out.proof),
        layers(&today_out.proof),
        "a wrong Gruen round must change the proof"
    );
    let verdict = verify(&layer, &faulted_out.proof);
    assert!(
        matches!(verdict, Err(Error::LayerRelationMismatch { .. })),
        "GKR's layer check must refuse it, got {verdict:?}"
    );
    assert_eq!(
        crossed.output.err(),
        Some(Error::DeviceFailed {
            stage: "gkr gruen xcheck"
        }),
        "the cross-check must refuse to prove over a wrong Gruen round"
    );
    eprintln!("argue gkr gruen: the wrong round was refused ({verdict:?}) and by the cross-check");
}

/// Where the knob stops: a tree under the device's threshold is proved here
/// from the start, and runs no Gruen layer.
#[test]
fn the_knob_takes_only_layers_a_device_holds() {
    let _restore = serial();
    let layer = input(10, 5);
    let off = prove(&layer, false, false, 64);
    let on = prove(&layer, true, false, 64);
    assert!(!on.on_card, "a 2^10 tree stays on the host");
    assert_eq!(on.gruen, 0, "no layer ran on a device");
    assert_eq!(
        layers(&off.output.unwrap().proof),
        layers(&on.output.unwrap().proof)
    );
}
