//! Dropped main traces ([`crate::regen`]) change nothing a verifier can see:
//! a trace dropped after its Round-1 commit and deposited again by a
//! regenerator proves the resident bytes. A deposit that is not the dropped
//! trace, a regenerator that stops, or a trace dropped before its Round-1
//! commit refuses the proof with a typed error before that table's device
//! work, and nothing hangs.
//!
//! These prove: they run on the box (`-p stark`), not on the laptop.

use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use math::field::{
    extensions_goldilocks::Degree3GoldilocksExtensionField, goldilocks::GoldilocksField,
};

use super::residency_mode_tests::traces;
use crate::examples::multi_table_lookup::{
    new_add_air_with_lookup, new_cpu_air_with_lookup, new_mul_air_with_lookup,
};
use crate::narrow::NarrowMain;
use crate::proof::options::ProofOptions;
use crate::proof::stark::MultiProof;
use crate::prover::{IsStarkProver, Prover, ProvingError};
use crate::regen::{RegenSlot, RegenWindow};
use crate::residency_mode::ResidencyMode;
use crate::spill::{SpillOptions, SpillStore};
use crate::traits::AIR;

type F = GoldilocksField;
type E = Degree3GoldilocksExtensionField;

const RESIDENCIES: [ResidencyMode; 3] = [
    ResidencyMode::Retain,
    ResidencyMode::RecomputeLde,
    ResidencyMode::RecomputeLdeDevice,
];

/// What each of the three tables does between its Round-1 commit and its
/// fused task.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Held {
    Resident,
    Spilled,
    /// Dropped, ranked `n`.
    Dropped(u64),
    /// Dropped before Round 1 (never precommitted).
    DroppedEarly,
}

/// What the regenerator does.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Regenerator {
    /// Deposits every dropped trace, in rank order.
    Faithful,
    /// Deposits table `t`'s trace with one bit flipped.
    Bent(usize),
    /// Deposits every trace with one bit flipped.
    BentAll,
    /// Deposits table `t`'s trace with one bit flipped, the window's digest
    /// check off (test hook): only the checks behind the digest are left.
    BentNoDigest(usize),
    /// Dies before depositing anything.
    Dead,
}

fn test_options() -> ProofOptions {
    ProofOptions {
        grinding_factor: 0,
        ..ProofOptions::default_test_options()
    }
}

/// The CPU/ADD/MUL instance proved the block's way, each table held as `held`
/// says and its dropped traces deposited as `regenerator` says, on a thread of
/// its own within 120 s (a hang is a failure).
fn prove_held(
    residency: ResidencyMode,
    held: [Held; 3],
    regenerator: Regenerator,
) -> Result<MultiProof<F, E, ()>, ProvingError> {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(prove_held_here(residency, held, regenerator));
    });
    rx.recv_timeout(std::time::Duration::from_secs(120))
        .expect("the prove hung")
}

fn prove_held_here(
    residency: ResidencyMode,
    held: [Held; 3],
    regenerator: Regenerator,
) -> Result<MultiProof<F, E, ()>, ProvingError> {
    let (mut cpu_trace, mut add_trace, mut mul_trace) = traces();
    let proof_options = test_options();
    let cpu_air = new_cpu_air_with_lookup(&proof_options);
    let add_air = new_add_air_with_lookup(&proof_options);
    let mul_air = new_mul_air_with_lookup(&proof_options);
    let store = SpillStore::open(SpillOptions::default()).expect("a spill store");
    let (window, producer) = RegenWindow::new(1 << 30);
    let mut precommitted = Vec::new();
    // (rank, table, slot, the packed trace the regenerator rebuilds)
    let mut dropped: Vec<(u64, usize, RegenSlot, NarrowMain)> = Vec::new();
    for (t, (air, trace)) in [
        (
            &cpu_air as &dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>,
            &mut cpu_trace,
        ),
        (&add_air, &mut add_trace),
        (&mul_air, &mut mul_trace),
    ]
    .into_iter()
    .enumerate()
    {
        assert!(trace.pack_main_narrow());
        if held[t] == Held::DroppedEarly {
            precommitted.push(None);
        } else {
            precommitted.push(Some(Prover::precommit_main(
                air,
                trace,
                #[cfg(feature = "disk-spill")]
                crate::storage_mode::StorageMode::Ram,
                residency,
            )?));
        }
        let rank = match held[t] {
            Held::Dropped(rank) => rank,
            _ => t as u64,
        };
        match held[t] {
            Held::Resident => {}
            Held::Spilled => assert!(trace.spill_main(&store)),
            Held::Dropped(_) | Held::DroppedEarly => {
                let packed = trace.narrow_main().expect("packed").clone();
                let slot = trace
                    .drop_main_for_regen(&window, rank)
                    .expect("a packed trace drops");
                dropped.push((rank, t, slot, packed));
            }
        }
    }
    store.flush();
    dropped.sort_by_key(|d| d.0);
    if let Regenerator::BentNoDigest(_) = regenerator {
        window.set_verify(false);
    }
    let regen = std::thread::spawn(move || {
        let _producer = producer;
        if regenerator == Regenerator::Dead {
            return;
        }
        for (_, t, slot, mut packed) in dropped {
            if regenerator == Regenerator::Bent(t)
                || regenerator == Regenerator::BentAll
                || regenerator == Regenerator::BentNoDigest(t)
            {
                packed.flip_first_bit();
            }
            // A refused deposit fails its slot; the prove refuses the table.
            let _ = slot.deposit(packed);
        }
    });
    let pairs: Vec<(
        &dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>,
        _,
        _,
    )> = vec![
        (&cpu_air, &mut cpu_trace, &()),
        (&add_air, &mut add_trace, &()),
        (&mul_air, &mut mul_trace, &()),
    ];
    let out = Prover::multi_prove_precommitted(
        pairs,
        &mut DefaultTranscript::<E>::new(&[]),
        #[cfg(feature = "disk-spill")]
        crate::storage_mode::StorageMode::Ram,
        residency,
        precommitted,
    );
    // The prove's end closed the window: the regenerator has returned.
    regen.join().expect("the regenerator thread");
    out
}

fn bytes(proof: &MultiProof<F, E, ()>) -> Vec<u8> {
    bincode::serialize(proof).unwrap()
}

fn verifies(proof: &MultiProof<F, E, ()>) -> bool {
    use crate::verifier::{IsStarkVerifier, Verifier};
    let proof_options = test_options();
    let cpu_air = new_cpu_air_with_lookup(&proof_options);
    let add_air = new_add_air_with_lookup(&proof_options);
    let mul_air = new_mul_air_with_lookup(&proof_options);
    let airs: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>> =
        vec![&cpu_air, &add_air, &mul_air];
    Verifier::multi_verify(
        &airs,
        proof,
        &mut DefaultTranscript::<E>::new(&[]),
        &math::field::element::FieldElement::zero(),
    )
}

fn air_names() -> [String; 3] {
    let o = test_options();
    [
        new_cpu_air_with_lookup(&o).name().to_string(),
        new_add_air_with_lookup(&o).name().to_string(),
        new_mul_air_with_lookup(&o).name().to_string(),
    ]
}

const RESIDENT: [Held; 3] = [Held::Resident; 3];

/// ★ Every trace dropped (deposited in any rank order), and a mix of dropped,
/// spilled and resident tables, proves the resident bytes under every
/// residency.
#[test]
fn every_dropped_trace_proves_the_resident_bytes() {
    for residency in RESIDENCIES {
        let want = bytes(&prove_held(residency, RESIDENT, Regenerator::Faithful).unwrap());
        for held in [
            [Held::Dropped(0), Held::Dropped(1), Held::Dropped(2)],
            [Held::Dropped(2), Held::Dropped(0), Held::Dropped(1)],
            [Held::Dropped(0), Held::Spilled, Held::Resident],
            [Held::Spilled, Held::Resident, Held::Dropped(5)],
        ] {
            let got = bytes(&prove_held(residency, held, Regenerator::Faithful).unwrap());
            assert!(want == got, "{residency:?} {held:?}: proof bytes moved");
        }
    }
}

/// ★ A deposit one bit off the dropped trace refuses the proof with
/// `RegeneratedTraceMismatch` naming that table, under every residency.
#[test]
fn a_wrong_regenerated_trace_refuses_the_proof() {
    let names = air_names();
    let all = [Held::Dropped(0), Held::Dropped(1), Held::Dropped(2)];
    for residency in RESIDENCIES {
        for (table, want) in names.iter().enumerate() {
            match prove_held(residency, all, Regenerator::Bent(table)) {
                Err(ProvingError::RegeneratedTraceMismatch(name)) => {
                    assert_eq!(&name, want, "{residency:?}")
                }
                Err(e) => panic!("{residency:?} table {table}: wrong refusal {e:?}"),
                Ok(_) => panic!("{residency:?} table {table}: a wrong trace was proved"),
            }
        }
    }
}

/// ★ Negative, the digest off: a regenerated trace one bit off the dropped
/// one is never a proof the verifier accepts unless it is the resident proof
/// itself (a word no stage of phase B reads). Where phase B recomputes the
/// main LDE from the deposited words (`RecomputeLde`), every perturbation is
/// caught; on the card the kept-top check refuses it (the cuda twin below).
#[test]
fn a_wrong_regenerated_trace_without_the_digest_is_never_accepted() {
    let all = [Held::Dropped(0), Held::Dropped(1), Held::Dropped(2)];
    for residency in RESIDENCIES {
        let want = bytes(&prove_held(residency, RESIDENT, Regenerator::Faithful).unwrap());
        let (mut refused, mut rejected) = (0, 0);
        for table in 0..3 {
            match prove_held(residency, all, Regenerator::BentNoDigest(table)) {
                Err(_) => refused += 1,
                Ok(proof) if !verifies(&proof) => rejected += 1,
                Ok(proof) => assert!(
                    bytes(&proof) == want,
                    "{residency:?} table {table}: a wrong trace proved and verified"
                ),
            }
        }
        eprintln!("{residency:?}: refused {refused}, rejected {rejected} of 3");
        if residency == ResidencyMode::RecomputeLde {
            assert_eq!(refused + rejected, 3, "{residency:?}");
        }
    }
}

/// ★ On the card, the digest off: a regenerated trace one bit off is refused
/// by the kept-top check — the layer behind the digest — never proved.
#[cfg(feature = "cuda")]
#[test_log::test]
#[ignore = "requires a GPU, LAMBDA_VM_GPU_LDE_THRESHOLD=2 and LAMBDA_VM_RECOMMIT_TOP_LEVELS=2; run alone"]
fn a_wrong_regenerated_trace_without_the_digest_is_refused_on_the_card() {
    let all = [Held::Dropped(0), Held::Dropped(1), Held::Dropped(2)];
    for table in 0..3 {
        match prove_held(
            ResidencyMode::RecomputeLdeDevice,
            all,
            Regenerator::BentNoDigest(table),
        ) {
            Err(ProvingError::RecomputedCommitmentMismatch(msg)) => {
                assert!(
                    msg.contains("does not match the kept tree"),
                    "table {table}: {msg}"
                )
            }
            Ok(proof) => panic!(
                "table {table}: a bent trace was proved (verifies: {})",
                verifies(&proof)
            ),
            Err(e) => panic!("table {table}: unexpected refusal {e:?}"),
        }
    }
}

/// ★ On the card, the second layer behind a regenerated trace: every deposit
/// faithful (the digest passes), then one packed bit flipped right before the
/// device widens it, and the kept-top check refuses the proof with
/// `RecomputedCommitmentMismatch` — `a_bad_device_widen_is_refused` with the
/// regenerator as the source (R-REGEN N4).
#[cfg(feature = "cuda")]
#[test_log::test]
#[ignore = "requires a GPU, LAMBDA_VM_GPU_LDE_THRESHOLD=2 and LAMBDA_VM_RECOMMIT_TOP_LEVELS=2; run alone"]
fn a_regenerated_trace_bent_before_the_widen_is_refused_on_the_card() {
    let all = [Held::Dropped(0), Held::Dropped(1), Held::Dropped(2)];
    for table in 0..3 {
        crate::residency_mode::test_hooks::perturb_narrow_before_recommit(table);
        let out = prove_held(
            ResidencyMode::RecomputeLdeDevice,
            all,
            Regenerator::Faithful,
        );
        assert_eq!(
            crate::residency_mode::test_hooks::PERTURB_NARROW_BEFORE_RECOMMIT
                .load(std::sync::atomic::Ordering::SeqCst),
            0,
            "table {table}: the perturbation never fired"
        );
        match out {
            Err(ProvingError::RecomputedCommitmentMismatch(msg)) => {
                assert!(
                    msg.contains("does not match the kept tree"),
                    "table {table}: {msg}"
                )
            }
            Err(e) => panic!("table {table}: wrong refusal {e:?}"),
            Ok(_) => panic!("table {table}: a bent regenerated trace was proved"),
        }
    }
}

/// ★ A regenerator that dies before depositing refuses the proof with
/// `RegeneratedTraceFailed`, and the prove returns (no driver hangs).
#[test]
fn a_dead_regenerator_refuses_the_proof() {
    let all = [Held::Dropped(0), Held::Dropped(1), Held::Dropped(2)];
    for residency in RESIDENCIES {
        let out = prove_held(residency, all, Regenerator::Dead);
        assert!(
            matches!(out, Err(ProvingError::RegeneratedTraceFailed(_))),
            "{residency:?}: {out:?}"
        );
    }
}

/// ★ A trace dropped before its Round-1 commit has no words to commit: the
/// proof is refused with `RegeneratedTraceFailed`, not committed from an
/// empty table.
#[test]
fn a_trace_dropped_before_round_one_is_refused() {
    let out = prove_held(
        ResidencyMode::RecomputeLdeDevice,
        [Held::Resident, Held::DroppedEarly, Held::Resident],
        Regenerator::Faithful,
    );
    match out {
        Err(ProvingError::RegeneratedTraceFailed(why)) => {
            assert!(why.contains("before its Round-1 commit"), "{why}")
        }
        other => panic!("{other:?}"),
    }
}

/// ★ The fused walk takes the dropped tables last, by rank, and starts them
/// in that order; Round 1 is untouched.
#[test]
fn the_dropped_tables_are_proved_last_by_rank() {
    let overrides = crate::spill::test_hooks::ProveOverrides {
        drivers: Some(1),
        ..Default::default()
    };
    let held = [Held::Dropped(1), Held::Resident, Held::Dropped(0)];
    let (out, admissions) = crate::spill::test_hooks::with_prove_overrides(overrides, || {
        prove_held_here(
            ResidencyMode::RecomputeLdeDevice,
            held,
            Regenerator::Faithful,
        )
    });
    out.unwrap();
    assert_eq!(&admissions.fused_walk[1..], &[2, 0], "{admissions:?}");
    assert_eq!(admissions.fused_walk[0], 1);
    assert_eq!(admissions.fused_started, admissions.fused_walk);
}

/// ★ On the card: every table's regenerated trace wrong, and the prove is
/// refused with `RegeneratedTraceMismatch` before any table's phase-B device
/// work (no recompute, widen or recommit counted).
#[cfg(feature = "cuda")]
#[test_log::test]
#[ignore = "requires a GPU, LAMBDA_VM_GPU_LDE_THRESHOLD=2 and LAMBDA_VM_RECOMMIT_TOP_LEVELS=2; run alone"]
fn a_wrong_regenerated_trace_is_refused_before_device_work_on_the_card() {
    use std::sync::atomic::Ordering;
    let counters = || {
        (
            crate::prover::NARROW_DEVICE_WIDENS.load(Ordering::SeqCst),
            crate::prover::TOP_TREE_RECOMPUTES.load(Ordering::SeqCst),
            crate::residency_mode::DEVICE_RECOMMITS.load(Ordering::SeqCst),
        )
    };
    let all = [Held::Dropped(0), Held::Dropped(1), Held::Dropped(2)];
    // Round 1 is precommitted for every table, so the only phase-B device
    // work is the fused tasks', each refused before it starts.
    let before = counters();
    let out = prove_held(ResidencyMode::RecomputeLdeDevice, all, Regenerator::BentAll);
    assert!(
        matches!(out, Err(ProvingError::RegeneratedTraceMismatch(_))),
        "{out:?}"
    );
    assert_eq!(counters(), before, "no phase-B device work");
}
