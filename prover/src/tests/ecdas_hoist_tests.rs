//! Tests for hoisting the ECDAS double-and-add chip out of continuation epochs.
//!
//! ECDAS is a pure function — it takes an accumulator on a bus and returns the
//! next one, touching no memory and carrying no state between calls — so a
//! continuation proves ONE run-wide instance instead of one per epoch. Each
//! epoch's ECSM simply omits its delegation buses (`Ecdas` and `Bit`); the
//! global proof re-emits them over the SAME committed ECSM trace, where the
//! single chain answers them. The two are tied by comparing the ECSM table's
//! main-trace Merkle root.

use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use math::field::element::FieldElement;

use stark::traits::AIR;
use stark::verifier::{IsStarkVerifier, Verifier};

use crate::tables::ecdas::EcdasOperation;
use crate::tables::ecsm::{self, EcsmOperation};
use crate::tables::types::{BusId, GoldilocksExtension, GoldilocksField};
use crate::test_utils::multi_prove_ram;

type F = GoldilocksField;
type E = GoldilocksExtension;
type AirRef<'a> = &'a dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>;
type ProvePair<'a> = (AirRef<'a>, &'a mut stark::trace::TraceTable<F, E>, &'a ());

fn gx_le() -> [u8; 32] {
    let mut be = [
        0x79, 0xBE, 0x66, 0x7E, 0xF9, 0xDC, 0xBB, 0xAC, 0x55, 0xA0, 0x62, 0x95, 0xCE, 0x87, 0x0B,
        0x07, 0x02, 0x9B, 0xFC, 0xDB, 0x2D, 0xCE, 0x28, 0xD9, 0x59, 0xF2, 0x81, 0x5B, 0x16, 0xF8,
        0x17, 0x98,
    ];
    be.reverse();
    be
}

fn k_le(v: u64) -> [u8; 32] {
    let mut k = [0u8; 32];
    k[..8].copy_from_slice(&v.to_le_bytes());
    k
}

/// One scalar multiplication: the ECSM row and the ECDAS steps it drives. They
/// must come from the same witness, or the delegation bus cannot balance.
fn call(k: u64, timestamp: u64) -> (EcsmOperation, Vec<EcdasOperation>) {
    let witness = ::ecsm::compute_witness(&k_le(k), &gx_le()).unwrap();
    let steps = witness
        .steps
        .clone()
        .into_iter()
        .map(|step| EcdasOperation { timestamp, step })
        .collect();
    (
        EcsmOperation {
            timestamp,
            addr_xg: 0x2000,
            addr_k: 0x3000,
            addr_xr: 0x1000,
            witness,
        },
        steps,
    )
}

/// Hoisting removes exactly the ECDAS delegation and nothing else: the memory,
/// dispatch and range-check interactions must stay in the epoch, or the epoch's
/// other buses stop balancing.
#[test]
fn hoisting_removes_only_the_ecdas_delegation() {
    let full = ecsm::bus_interactions();
    let epoch = ecsm::bus_interactions_without_ecdas();
    let delegation = ecsm::ecdas_delegation_bus_interactions();

    let delegated = |v: &[stark::lookup::BusInteraction]| {
        v.iter()
            .filter(|i| i.bus_id == BusId::Ecdas as u64 || i.bus_id == BusId::Bit as u64)
            .count()
    };

    assert_eq!(delegated(&epoch), 0, "the epoch delegates nothing");
    assert_eq!(
        delegated(&full),
        delegation.len(),
        "the delegation is exactly the Ecdas + Bit interactions",
    );
    assert_eq!(
        epoch.len() + delegation.len(),
        full.len(),
        "the two halves must partition the monolithic set",
    );

    // What the global proof emits is byte-identical to what the monolithic ECSM
    // emits — same polarity, same tuples.
    let full_delegated: Vec<_> = full
        .iter()
        .filter(|i| i.bus_id == BusId::Ecdas as u64 || i.bus_id == BusId::Bit as u64)
        .collect();
    for (a, b) in full_delegated.iter().zip(delegation.iter()) {
        assert_eq!(a.bus_id, b.bus_id);
        assert_eq!(a.is_sender, b.is_sender, "polarity must match");
        assert_eq!(
            a.values.iter().map(|v| v.num_bus_elements()).sum::<usize>(),
            b.values.iter().map(|v| v.num_bus_elements()).sum::<usize>(),
            "tuple width must match",
        );
    }
}

/// The global half against the REAL chip: several epochs' ECSM tables on one
/// side, one run-wide ECDAS (with its BITWISE provider) on the other. The bus
/// balances only if that single chain serves every epoch's requests — which is
/// the whole claim of hoisting it out.
#[test]
fn one_ecdas_chip_serves_every_epochs_requests() {
    let opts = crate::recursion::MIN_PROOF_OPTIONS;

    // Three "epochs" with uneven usage, including one that does no scalar
    // multiplication — the case that must not disturb the shared chain.
    let (ecsm_a, ecdas_a) = call(5, 444);
    let (ecsm_b, ecdas_b) = call(1_000_003, 999);
    let ecsm_per_epoch: Vec<Vec<EcsmOperation>> = vec![vec![ecsm_a], Vec::new(), vec![ecsm_b]];
    let ecdas_per_epoch: Vec<Vec<EcdasOperation>> = vec![ecdas_a, Vec::new(), ecdas_b];

    let request_airs: Vec<_> = ecsm_per_epoch
        .iter()
        .filter(|ops| !ops.is_empty())
        .map(|_| crate::test_utils::create_ecsm_ecdas_request_air(&opts))
        .collect();
    let mut request_traces: Vec<_> = ecsm_per_epoch
        .iter()
        .filter(|ops| !ops.is_empty())
        .map(|ops| ecsm::generate_ecsm_trace(ops))
        .collect();

    let (ecdas_air, bw_air) = crate::continuation::global_ecdas_airs(&opts);
    let (mut ecdas_trace, mut bw_trace) =
        crate::continuation::global_ecdas_traces(&ecdas_per_epoch);

    let mut pairs: Vec<ProvePair<'_>> = request_airs
        .iter()
        .zip(request_traces.iter_mut())
        .map(|(a, t)| (a as AirRef<'_>, t, &()))
        .collect();
    pairs.push((&ecdas_air, &mut ecdas_trace, &()));
    pairs.push((&bw_air, &mut bw_trace, &()));

    let proof = multi_prove_ram(pairs, &mut DefaultTranscript::<E>::new(&[]))
        .expect("global ecdas proof should be provable");

    let mut refs: Vec<AirRef<'_>> = request_airs.iter().map(|a| a as _).collect();
    refs.push(&ecdas_air);
    refs.push(&bw_air);

    assert!(
        Verifier::multi_verify(
            &refs,
            &proof,
            &mut DefaultTranscript::<E>::new(&[]),
            &FieldElement::zero(),
        ),
        "one run-wide ECDAS must balance against every epoch's ECSM",
    );
}

/// A multi-epoch continuation over a program that actually does a scalar
/// multiplication. Requires the rust artifacts (`make compile-programs-rust`).
#[test]
fn continuation_with_ecsm_verifies_with_ecdas_hoisted() {
    let _ = env_logger::builder().is_test(true).try_init();

    let workspace_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .to_path_buf();
    let elf_bytes = std::fs::read(workspace_root.join("executor/program_artifacts/rust/ecsm.elf"))
        .expect("ecsm.elf not found — run `make compile-programs-rust`");
    let opts = crate::recursion::MIN_PROOF_OPTIONS;

    let bundle = crate::continuation::prove_continuation(&elf_bytes, &[], 12, &opts)
        .expect("continuation prove should succeed");
    assert!(
        bundle.num_epochs() > 1,
        "the program must split across epochs for this test to bite",
    );

    assert!(
        crate::continuation::verify_continuation(&elf_bytes, &bundle, &opts)
            .expect("bundle should be structurally well formed")
            .is_some(),
        "a continuation with ECDAS hoisted must verify",
    );
}

/// A program that never does EC work still has to produce a well-formed global
/// proof, whose ECDAS is entirely padding.
#[test]
fn continuation_without_any_ecsm_still_verifies() {
    let _ = env_logger::builder().is_test(true).try_init();

    let elf_bytes = crate::test_utils::asm_elf_bytes("all_loadstore_32");
    let opts = crate::recursion::MIN_PROOF_OPTIONS;

    let bundle = crate::continuation::prove_continuation(&elf_bytes, &[], 3, &opts)
        .expect("continuation prove should succeed");
    assert!(bundle.num_epochs() > 1);

    assert!(
        crate::continuation::verify_continuation(&elf_bytes, &bundle, &opts)
            .expect("bundle should be structurally well formed")
            .is_some(),
        "an all-padding global ECDAS must still verify",
    );
}
