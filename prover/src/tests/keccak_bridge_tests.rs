//! Tests for the KECCAK_BRIDGE table.
//!
//! The bridge's whole job is to be an exact inverse of itself across the two
//! proofs: the epoch-local polarity closes the `Keccak` bus against the KECCAK
//! core, and the global polarity re-opens the identical request against the
//! run-wide KECCAK_RND chain. These tests pin both halves of that claim —
//! that the two polarities cancel, and that the tuple the bridge speaks is
//! byte-for-byte the one the real keccak chips speak.

use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use math::field::element::FieldElement;

use stark::constraints::builder::EmptyConstraints;
use stark::lookup::{
    AirWithBuses, AuxiliaryTraceBuildData, BusInteraction, NullBoundaryConstraintBuilder,
};
use stark::proof::options::{GoldilocksCubicProofOptions, ProofOptions};
use stark::traits::AIR;
use stark::verifier::{IsStarkVerifier, Verifier};

use crate::tables::keccak::KeccakOperation;
use crate::tables::keccak_bridge::{
    self, epoch_bus_interactions, generate_keccak_bridge_trace, global_bus_interactions,
};
use crate::tables::types::{BusId, GoldilocksExtension, GoldilocksField};
use crate::test_utils::multi_prove_ram;

type F = GoldilocksField;
type E = GoldilocksExtension;
type AirRef<'a> = &'a dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>;
type ProvePair<'a> = (AirRef<'a>, &'a mut stark::trace::TraceTable<F, E>, &'a ());

fn bridge_air(proof_options: &ProofOptions, interactions: Vec<BusInteraction>) -> BridgeAir {
    AirWithBuses::new(
        keccak_bridge::cols::NUM_COLUMNS,
        AuxiliaryTraceBuildData { interactions },
        proof_options,
        1,
        EmptyConstraints,
    )
}

type BridgeAir = AirWithBuses<F, E, NullBoundaryConstraintBuilder, (), EmptyConstraints>;

fn airs<'a>(
    epoch: &'a BridgeAir,
    global: &'a BridgeAir,
) -> Vec<&'a dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>> {
    vec![epoch, global]
}

/// A few permutation calls with distinct states, including one repeated input
/// (two epochs legitimately hashing the same state) and one repeated timestamp,
/// which the multiset argument must handle without an epoch label.
fn ops() -> Vec<KeccakOperation> {
    let mut out = Vec::new();
    for (i, ts) in [7u64, 9, 9, 40_000_000_000].into_iter().enumerate() {
        let mut input = [0u64; 25];
        // Two of the four share an input state (i = 1 and i = 2).
        let seed = if i == 2 { 1 } else { i as u64 };
        for (lane, slot) in input.iter_mut().enumerate() {
            *slot = seed
                .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                .wrapping_add(lane as u64)
                .wrapping_mul(0xBF58_476D_1CE4_E5B9);
        }
        let mut output = input;
        executor::vm::instruction::execution::keccak_f1600(&mut output);
        out.push(KeccakOperation {
            timestamp: ts,
            state_addr: 0x1000 + 0x200 * i as u64,
            input,
            output,
        });
    }
    out
}

/// The two polarities are exact inverses: proving the SAME trace under both
/// AIRs must balance the `Keccak` bus to zero. This is the property the
/// epoch/global split rests on — whatever the epoch proof absorbs, the global
/// proof re-emits.
#[test]
fn polarities_cancel_on_the_same_trace() {
    let opts = GoldilocksCubicProofOptions::with_blowup(2).unwrap();
    let ops = ops();

    let epoch_air = bridge_air(&opts, epoch_bus_interactions());
    let global_air = bridge_air(&opts, global_bus_interactions());

    let mut epoch_trace = generate_keccak_bridge_trace(&ops);
    let mut global_trace = generate_keccak_bridge_trace(&ops);
    // Both proofs must commit the identical trace — that is what the root
    // binding ties together.
    for row in 0..epoch_trace.num_rows() {
        for col in 0..keccak_bridge::cols::NUM_COLUMNS {
            assert_eq!(
                epoch_trace.main_table.get(row, col),
                global_trace.main_table.get(row, col),
                "trace mismatch at ({row}, {col})",
            );
        }
    }

    let proof = multi_prove_ram(
        vec![
            (
                &epoch_air as &dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>,
                &mut epoch_trace,
                &(),
            ),
            (
                &global_air as &dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>,
                &mut global_trace,
                &(),
            ),
        ],
        &mut DefaultTranscript::<E>::new(&[]),
    )
    .expect("bridge proof");

    assert!(
        Verifier::multi_verify(
            &airs(&epoch_air, &global_air),
            &proof,
            &mut DefaultTranscript::<E>::new(&[]),
            &FieldElement::zero(),
        ),
        "the Keccak bus must balance to zero across the two polarities",
    );
}

/// Dropping a single request from the global side must break the bus. This is
/// what stops the global proof from serving fewer permutations than the epochs
/// asked for.
#[test]
fn dropping_a_request_globally_breaks_the_bus() {
    let opts = GoldilocksCubicProofOptions::with_blowup(2).unwrap();
    let ops = ops();

    let epoch_air = bridge_air(&opts, epoch_bus_interactions());
    let global_air = bridge_air(&opts, global_bus_interactions());

    let mut epoch_trace = generate_keccak_bridge_trace(&ops);
    // The global side serves one request fewer.
    let mut global_trace = generate_keccak_bridge_trace(&ops[..ops.len() - 1]);

    let proved = multi_prove_ram(
        vec![
            (
                &epoch_air as &dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>,
                &mut epoch_trace,
                &(),
            ),
            (
                &global_air as &dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>,
                &mut global_trace,
                &(),
            ),
        ],
        &mut DefaultTranscript::<E>::new(&[]),
    );

    let rejected = match proved {
        Err(_) => true,
        Ok(proof) => !Verifier::multi_verify(
            &airs(&epoch_air, &global_air),
            &proof,
            &mut DefaultTranscript::<E>::new(&[]),
            &FieldElement::zero(),
        ),
    };
    assert!(rejected, "a dropped global request must not verify");
}

/// The bridge must speak exactly the tuple the real chips speak, or it will
/// close the epoch bus against nothing. Pins the bus id, the sender/receiver
/// polarity and the bus-element count against `keccak::bus_interactions`.
#[test]
fn tuple_matches_the_keccak_core() {
    let core: Vec<_> = crate::tables::keccak::bus_interactions()
        .into_iter()
        .filter(|i| i.bus_id == BusId::Keccak as u64)
        .collect();
    assert_eq!(core.len(), 2, "the core has one send and one receive");
    let core_send = core.iter().find(|i| i.is_sender).unwrap();
    let core_recv = core.iter().find(|i| !i.is_sender).unwrap();

    let bridge = epoch_bus_interactions();
    let bridge_recv = bridge.iter().find(|i| !i.is_sender).unwrap();
    let bridge_send = bridge.iter().find(|i| i.is_sender).unwrap();

    // The bridge receives what the core sends, and sends what the core receives.
    let elems = |i: &BusInteraction| i.values.iter().map(|v| v.num_bus_elements()).sum::<usize>();
    assert_eq!(elems(bridge_recv), elems(core_send), "input tuple width");
    assert_eq!(elems(bridge_send), elems(core_recv), "output tuple width");
    assert_eq!(
        elems(bridge_recv),
        203,
        "ts_lo, ts_hi, round, 200 state bytes"
    );

    // And the global polarity is the mirror of the epoch one.
    let global = global_bus_interactions();
    for (e, g) in bridge.iter().zip(global.iter()) {
        assert_eq!(e.bus_id, g.bus_id);
        assert_ne!(e.is_sender, g.is_sender, "polarity must be flipped");
        assert_eq!(elems(e), elems(g), "same tuple, opposite direction");
    }
}

/// The real thing: a multi-epoch continuation over a program that actually
/// hashes. Each epoch commits only its KECCAK_BRIDGE; the single run-wide
/// KECCAK_RND in the global proof serves every epoch's requests. If the
/// hoisting were wrong in either direction — an epoch's `Keccak` bus left open,
/// or the global chain serving the wrong multiset — the bus balance fails and
/// verification rejects.
#[test]
fn continuation_with_keccak_verifies_with_the_round_chip_hoisted() {
    let _ = env_logger::builder().is_test(true).try_init();

    let elf_bytes = crate::test_utils::asm_elf_bytes("test_keccak_multi");
    let opts = crate::recursion::MIN_PROOF_OPTIONS;

    // Small epochs so the keccak calls land in more than one epoch — that is
    // what makes one run-wide round chip serve several epochs' requests.
    let bundle = crate::continuation::prove_continuation(&elf_bytes, &[], 6, &opts)
        .expect("continuation prove should succeed");
    assert!(
        bundle.num_epochs() > 1,
        "the program must split across epochs for this test to bite",
    );

    let verified = crate::continuation::verify_continuation(&elf_bytes, &bundle, &opts)
        .expect("bundle should be structurally well formed");
    assert!(
        verified.is_some(),
        "a continuation with the keccak round chip hoisted must verify",
    );
}

/// The zero-keccak case: a program that never hashes still has to produce a
/// well-formed global proof, whose KECCAK_RND is entirely padding. This is the
/// completeness trap the hoisting most easily breaks, because ECDAS/KECCAK
/// tables are emitted unconditionally.
#[test]
fn continuation_without_any_keccak_still_verifies() {
    let _ = env_logger::builder().is_test(true).try_init();

    let elf_bytes = crate::test_utils::asm_elf_bytes("all_loadstore_32");
    let opts = crate::recursion::MIN_PROOF_OPTIONS;

    let bundle = crate::continuation::prove_continuation(&elf_bytes, &[], 3, &opts)
        .expect("continuation prove should succeed");
    assert!(bundle.num_epochs() > 1);

    let verified = crate::continuation::verify_continuation(&elf_bytes, &bundle, &opts)
        .expect("bundle should be structurally well formed");
    assert!(
        verified.is_some(),
        "an all-padding global KECCAK_RND must still verify",
    );
}

/// The global half, against the REAL round chip: several epochs' worth of
/// KECCAK_BRIDGE traces on one side, and the single run-wide KECCAK_RND (with
/// its KECCAK_RC and BITWISE providers) on the other. The bus balances only if
/// that one chain actually serves every epoch's requests — which is the whole
/// claim of hoisting it out of the epochs.
///
/// This is the end-to-end mechanism minus the ELF: the asm/Rust guest programs
/// are not prebuilt in every environment, and this exercises the same buses.
#[test]
fn one_round_chip_serves_every_epochs_requests() {
    let opts = crate::recursion::MIN_PROOF_OPTIONS;

    // Three "epochs" with uneven keccak usage, including one that hashes
    // nothing — the case that must not disturb the shared chain.
    let all = ops();
    let per_epoch: Vec<Vec<KeccakOperation>> =
        vec![all[..2].to_vec(), Vec::new(), all[2..].to_vec()];

    let bridge_airs: Vec<_> = per_epoch
        .iter()
        .map(|_| crate::test_utils::create_keccak_bridge_global_air(&opts))
        .collect();
    let mut bridge_traces: Vec<_> = per_epoch
        .iter()
        .map(|ops| generate_keccak_bridge_trace(ops))
        .collect();

    let (rnd_air, rc_air, bw_air) = crate::continuation::global_keccak_airs(&opts);
    let (mut rnd_trace, mut rc_trace, mut bw_trace) =
        crate::continuation::global_keccak_traces(&per_epoch);

    let mut pairs: Vec<ProvePair<'_>> = bridge_airs
        .iter()
        .zip(bridge_traces.iter_mut())
        .map(|(a, t)| (a as AirRef<'_>, t, &()))
        .collect();
    pairs.push((&rnd_air, &mut rnd_trace, &()));
    pairs.push((&rc_air, &mut rc_trace, &()));
    pairs.push((&bw_air, &mut bw_trace, &()));

    let proof = multi_prove_ram(pairs, &mut DefaultTranscript::<E>::new(&[]))
        .expect("global keccak proof should be provable");

    let mut refs: Vec<AirRef<'_>> = bridge_airs.iter().map(|a| a as _).collect();
    refs.push(&rnd_air);
    refs.push(&rc_air);
    refs.push(&bw_air);

    assert!(
        Verifier::multi_verify(
            &refs,
            &proof,
            &mut DefaultTranscript::<E>::new(&[]),
            &FieldElement::zero(),
        ),
        "one run-wide KECCAK_RND must balance against every epoch's bridge",
    );
}
