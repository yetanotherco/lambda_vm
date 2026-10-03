//! A `Retain` prove under the shared VRAM gate keeps each table's resident
//! bytes (its main LDE, snapshot and tree) in the gate's account from its
//! Round-1 task until its fused task ends, and admits the fused task for the
//! rest of its set: the gate's running total is then what the prove holds on
//! the card. Proven here on the prove's own gate with the carry forced
//! (`ProveOverrides::carry_residents`), one driver, so every reading of the
//! account is exact. The proof does not move.

use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use math::field::{
    extensions_goldilocks::Degree3GoldilocksExtensionField, goldilocks::GoldilocksField,
};

use super::residency_mode_tests::traces;
use crate::examples::multi_table_lookup::{
    new_add_air_with_lookup, new_cpu_air_with_lookup, new_mul_air_with_lookup,
};
use crate::proof::options::ProofOptions;
use crate::proof::stark::MultiProof;
use crate::prover::{IsStarkProver, Prover};
use crate::residency_mode::ResidencyMode;
use crate::spill::test_hooks::{Admissions, ProveOverrides, with_prove_overrides};
use crate::traits::AIR;

type F = GoldilocksField;
type E = Degree3GoldilocksExtensionField;

/// Grinding off, so two proves of one instance are the same bytes.
fn test_options() -> ProofOptions {
    ProofOptions {
        grinding_factor: 0,
        ..ProofOptions::default_test_options()
    }
}

fn prove(overrides: Option<ProveOverrides>) -> (MultiProof<F, E, ()>, Option<Admissions>) {
    let run = || {
        let (mut cpu_trace, mut add_trace, mut mul_trace) = traces();
        let o = test_options();
        let cpu_air = new_cpu_air_with_lookup(&o);
        let add_air = new_add_air_with_lookup(&o);
        let mul_air = new_mul_air_with_lookup(&o);
        let pairs: Vec<(
            &dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>,
            _,
            _,
        )> = vec![
            (&cpu_air, &mut cpu_trace, &()),
            (&add_air, &mut add_trace, &()),
            (&mul_air, &mut mul_trace, &()),
        ];
        Prover::multi_prove(
            pairs,
            &mut DefaultTranscript::<E>::new(&[]),
            #[cfg(feature = "disk-spill")]
            crate::storage_mode::StorageMode::Ram,
            ResidencyMode::Retain,
        )
        .unwrap()
    };
    match overrides {
        Some(o) => {
            let (proof, admissions) = with_prove_overrides(o, run);
            (proof, Some(admissions))
        }
        None => (run(), None),
    }
}

fn bytes(proof: &MultiProof<F, E, ()>) -> Vec<u8> {
    bincode::serialize(proof).unwrap()
}

/// Each fused task starts with the gate holding exactly the residents of the
/// tables not yet proved (its own included) plus its own top-up.
fn assert_exact_account(a: &Admissions) {
    assert_eq!(a.fused_started, a.fused_walk, "one driver: walk order");
    assert_eq!(
        a.fused_gate_used.len(),
        a.fused_walk.len(),
        "one reading per task"
    );
    for (pos, &(idx, used)) in a.fused_gate_used.iter().enumerate() {
        assert_eq!(idx, a.fused_walk[pos]);
        let residents: u64 = a.fused_walk[pos..].iter().map(|&t| a.carried[t]).sum();
        assert_eq!(
            used,
            residents + a.fused_admitted[idx],
            "table {idx}: the gate's account at its fused start"
        );
    }
    assert_eq!(a.gate_used_after_fused, Some(0), "nothing left held");
}

/// ★ Carried: every table's resident bytes stay in the account from Round 1
/// to its fused task, which is admitted for the rest of its set; the claim
/// settles to the carried bytes plus the largest top-up; the gate ends empty;
/// the proof is the uncarried prove's, byte for byte.
#[test]
fn a_carrying_prove_keeps_the_account_exact_and_the_proof() {
    let want = bytes(&prove(None).0);
    let (proof, a) = prove(Some(ProveOverrides {
        drivers: Some(1),
        carry_residents: true,
        ..Default::default()
    }));
    let a = a.unwrap();
    assert!(want == bytes(&proof), "carrying moved the proof bytes");
    assert_eq!(a.carried.len(), 3);
    assert!(
        a.carried.iter().all(|&c| c > 0),
        "every table carries its residents: {:?}",
        a.carried
    );
    let claim = a.claim.expect("a carrying prove claims");
    let settled = a.settled_claim.expect("and settles");
    let held: u64 = a.carried.iter().sum();
    let top_up = a.fused_admitted.iter().copied().max().unwrap();
    assert_eq!(settled, held + top_up, "settled = carried + largest top-up");
    assert!(settled <= claim, "the claim only settles down");
    assert_exact_account(&a);
}

/// The control: not carrying, no claim, nothing carried, and each fused task
/// is admitted for its whole set alone.
#[test]
fn a_prove_that_does_not_carry_admits_whole_sets() {
    let (_, a) = prove(Some(ProveOverrides {
        drivers: Some(1),
        ..Default::default()
    }));
    let a = a.unwrap();
    assert_eq!(a.claim, None);
    assert_eq!(a.carried, vec![0; 3]);
    assert_exact_account(&a);
}
