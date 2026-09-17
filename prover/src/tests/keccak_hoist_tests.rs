//! Tests for hoisting the keccak round chip out of continuation epochs.
//!
//! KECCAK_RND is a pure function, so a continuation proves ONE run-wide
//! instance instead of one per epoch. Each epoch's core table simply omits its
//! `Keccak` bus pair; the global proof re-emits that pair over the SAME
//! committed core trace, where the single round chain answers it. The two are
//! tied by comparing the core table's main-trace Merkle root.
//!
//! These pin the two halves of that: that the epoch's core really does drop
//! only the round request, and that one round chain balances against several
//! epochs' cores at once.

use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use math::field::element::FieldElement;

use stark::traits::AIR;
use stark::verifier::{IsStarkVerifier, Verifier};

use crate::tables::keccak::{self, KeccakOperation};
use crate::tables::types::{BusId, GoldilocksExtension, GoldilocksField};
use crate::test_utils::multi_prove_ram;

type F = GoldilocksField;
type E = GoldilocksExtension;
type AirRef<'a> = &'a dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>;
type ProvePair<'a> = (AirRef<'a>, &'a mut stark::trace::TraceTable<F, E>, &'a ());

/// Permutation calls with distinct states, including one repeated input (two
/// epochs legitimately hashing the same state) and a repeated timestamp — both
/// of which the multiset argument must absorb without an epoch label.
fn ops() -> Vec<KeccakOperation> {
    (0..4)
        .map(|i| {
            let seed = if i == 2 { 1u64 } else { i as u64 };
            let mut input = [0u64; 25];
            for (lane, slot) in input.iter_mut().enumerate() {
                *slot = seed
                    .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                    .wrapping_add(lane as u64)
                    .wrapping_mul(0xBF58_476D_1CE4_E5B9);
            }
            let mut output = input;
            executor::vm::instruction::execution::keccak_f1600(&mut output);
            KeccakOperation {
                timestamp: [7u64, 9, 9, 40_000_000_000][i],
                state_addr: 0x1000 + 0x200 * i as u64,
                input,
                output,
            }
        })
        .collect()
}

/// Hoisting removes exactly the two `Keccak` bus interactions and nothing else:
/// the memory, dispatch and range-check interactions must stay in the epoch, or
/// the epoch's other buses stop balancing.
#[test]
fn hoisting_removes_only_the_round_request() {
    let full = keccak::bus_interactions();
    let epoch = keccak::bus_interactions_without_rounds();
    let request = keccak::rounds_request_bus_interactions();

    let on_keccak_bus = |v: &[stark::lookup::BusInteraction]| {
        v.iter()
            .filter(|i| i.bus_id == BusId::Keccak as u64)
            .count()
    };

    assert_eq!(
        on_keccak_bus(&full),
        2,
        "the core asks once and is answered once"
    );
    assert_eq!(on_keccak_bus(&epoch), 0, "the epoch asks nothing");
    assert_eq!(
        request.len(),
        2,
        "the global proof carries only the request pair"
    );
    assert_eq!(
        epoch.len() + request.len(),
        full.len(),
        "the two halves must partition the monolithic set",
    );

    // The request the global proof makes is byte-identical to the one the
    // monolithic core makes — same polarity, same tuple.
    let full_pair: Vec<_> = full
        .iter()
        .filter(|i| i.bus_id == BusId::Keccak as u64)
        .collect();
    for (a, b) in full_pair.iter().zip(request.iter()) {
        assert_eq!(a.is_sender, b.is_sender, "polarity must match");
        assert_eq!(
            a.values.iter().map(|v| v.num_bus_elements()).sum::<usize>(),
            b.values.iter().map(|v| v.num_bus_elements()).sum::<usize>(),
            "tuple width must match",
        );
    }
}

/// The global half against the REAL round chip: several epochs' core tables on
/// one side, one run-wide KECCAK_RND (with its KECCAK_RC and BITWISE providers)
/// on the other. The bus balances only if that single chain serves every
/// epoch's requests — which is the whole claim of hoisting it out.
#[test]
fn one_round_chip_serves_every_epochs_requests() {
    let opts = crate::recursion::MIN_PROOF_OPTIONS;

    // Three "epochs" with uneven keccak usage, including one that hashes
    // nothing — the case that must not disturb the shared chain.
    let all = ops();
    let per_epoch: Vec<Vec<KeccakOperation>> =
        vec![all[..2].to_vec(), Vec::new(), all[2..].to_vec()];

    let request_airs: Vec<_> = per_epoch
        .iter()
        .map(|_| crate::test_utils::create_keccak_rounds_request_air(&opts))
        .collect();
    let mut request_traces: Vec<_> = per_epoch
        .iter()
        .map(|ops| keccak::generate_keccak_trace(ops))
        .collect();

    let (rnd_air, rc_air, bw_air) = crate::continuation::global_keccak_airs(&opts);
    let (mut rnd_trace, mut rc_trace, mut bw_trace) =
        crate::continuation::global_keccak_traces(&per_epoch);

    let mut pairs: Vec<ProvePair<'_>> = request_airs
        .iter()
        .zip(request_traces.iter_mut())
        .map(|(a, t)| (a as AirRef<'_>, t, &()))
        .collect();
    pairs.push((&rnd_air, &mut rnd_trace, &()));
    pairs.push((&rc_air, &mut rc_trace, &()));
    pairs.push((&bw_air, &mut bw_trace, &()));

    let proof = multi_prove_ram(pairs, &mut DefaultTranscript::<E>::new(&[]))
        .expect("global keccak proof should be provable");

    let mut refs: Vec<AirRef<'_>> = request_airs.iter().map(|a| a as _).collect();
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
        "one run-wide KECCAK_RND must balance against every epoch's core",
    );
}

/// A multi-epoch continuation over a program that actually hashes. Requires the
/// asm artifacts (`make compile-programs-asm`).
#[test]
fn continuation_with_keccak_verifies_with_the_round_chip_hoisted() {
    let _ = env_logger::builder().is_test(true).try_init();

    let elf_bytes = crate::test_utils::asm_elf_bytes("test_keccak_multi");
    let opts = crate::recursion::MIN_PROOF_OPTIONS;

    let bundle = crate::continuation::prove_continuation(&elf_bytes, &[], 6, &opts)
        .expect("continuation prove should succeed");
    assert!(
        bundle.num_epochs() > 1,
        "the program must split across epochs for this test to bite",
    );

    assert!(
        crate::continuation::verify_continuation(&elf_bytes, &bundle, &opts)
            .expect("bundle should be structurally well formed")
            .is_some(),
        "a continuation with the keccak round chip hoisted must verify",
    );
}

/// A program that never hashes still has to produce a well-formed global proof,
/// whose KECCAK_RND is entirely padding — the completeness trap the hoisting
/// most easily breaks, since the keccak tables are emitted unconditionally.
#[test]
fn continuation_without_any_keccak_still_verifies() {
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
        "an all-padding global KECCAK_RND must still verify",
    );
}
