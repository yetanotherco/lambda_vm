//! The BLAKE3 Fiat-Shamir configuration, and rider 1's fixed consumption
//! schedule.
//!
//! Two things are under test and they are independent: that the transcript's
//! sponge is `Blake3Chain` in the framing the keccak one has always used, and
//! that a field draw under this configuration consumes a *fixed* number of
//! candidates. The second is what a straight-line machine needs — the LFM
//! transcript replay encodes one consumption schedule, so a draw whose count
//! varies with the bytes it happened to see is unprovable against it.

use alloc::vec::Vec;
use digest::Digest;
use math::field::{
    element::FieldElement, extensions_goldilocks::Degree3GoldilocksExtensionField,
    goldilocks::GoldilocksField, traits::HasDefaultTranscript,
};

use crate::fiat_shamir::default_transcript::{
    Blake3Transcript, DefaultTranscript, candidate_under_fixed_schedule,
};
use crate::fiat_shamir::is_transcript::IsTranscript;
use crate::fiat_shamir::transcript_hash::{
    Blake3TranscriptHash, KeccakTranscriptHash, TranscriptHash,
};
use crate::hash::blake3::chain::Blake3Chain;

type F = GoldilocksField;
type E = Degree3GoldilocksExtensionField;

const SEED: &[u8] = b"lambda-vm-blake3-transcript-kat-v1";

// =========================================================================
// The sponge is Blake3Chain, in the framing the keccak transcript uses.
// =========================================================================

/// ★ Anchor: the transcript's state is the chain hash of exactly what was
/// absorbed.
///
/// `state()` is not an internal detail — it is the grinding seed. Pinning it
/// against `Blake3Chain` computed directly is what says the transcript absorbs
/// what it claims to, and `Blake3Chain` is in turn anchored from outside (its
/// 7-round arm reproduces the `blake3` crate at all 1025 lengths ≤ 1 chunk, and
/// its 6-round arm has the committed `CHAIN_KAT_6ROUND` table).
#[test]
fn the_blake3_transcript_state_is_the_chain_of_what_was_absorbed() {
    let mut t = Blake3Transcript::<F>::new(SEED);
    assert_eq!(
        t.state(),
        <[u8; 32]>::from(Blake3Chain::digest(SEED)),
        "a fresh transcript's state must be the chain hash of its seed"
    );

    t.append_bytes(b"a-merkle-root");
    let mut expected = Vec::from(SEED);
    expected.extend_from_slice(b"a-merkle-root");
    assert_eq!(
        t.state(),
        <[u8; 32]>::from(Blake3Chain::digest(&expected)),
        "absorbing must concatenate into the same chain, not reset it"
    );
}

/// The duplex squeeze is the same construction under the new digest: finalize
/// **and reset**, reverse, absorb the reversed output.
///
/// Reimplemented here rather than compared against itself — this is the one
/// place where a hash swap could silently drop the reverse-and-reabsorb step,
/// and prover and verifier would still agree with each other while producing a
/// transcript nobody else can reproduce.
///
/// Note the reset: the squeeze is `finalize_reset`, so squeeze `k+1` hashes the
/// reversed output of squeeze `k` **alone**, not the whole absorbed history.
/// That is what makes the sponge a chain rather than a growing buffer, and it
/// is the detail this test exists to pin.
#[test]
fn the_blake3_squeeze_chain_matches_the_construction() {
    let mut t = Blake3Transcript::<F>::new(SEED);

    let mut pending = Vec::from(SEED);
    for k in 0..3 {
        let mut expected = <[u8; 32]>::from(Blake3Chain::digest(&pending));
        expected.reverse();
        assert_eq!(t.sample(), expected, "squeeze {k}");
        pending = Vec::from(expected);
    }
}

/// CONTROL: the two configurations are actually different transcripts.
///
/// Without this, every test here would pass just as well if `Blake3Transcript`
/// had been left resolving to keccak.
#[test]
fn the_blake3_and_keccak_transcripts_diverge() {
    let mut blake3 = Blake3Transcript::<F>::new(SEED);
    let mut keccak = DefaultTranscript::<F>::new(SEED);
    assert_ne!(blake3.state(), keccak.state());
    assert_ne!(blake3.sample(), keccak.sample());
}

// =========================================================================
// Rider 1 — the consumption schedule.
// =========================================================================

/// The configurations' schedules, as a fact rather than as prose.
///
/// The keccak arm MUST stay `None`. Rider 1 is adopted for BLAKE3 only, because
/// changing the keccak schedule would move every existing proof's challenges —
/// the one thing P-a's staging holds still until the flip.
#[test]
fn only_the_blake3_configuration_takes_the_fixed_schedule() {
    assert!(
        KeccakTranscriptHash::CANDIDATES_PER_COORDINATE.is_none(),
        "the keccak schedule must not move before the flip"
    );
    assert_eq!(
        Blake3TranscriptHash::CANDIDATES_PER_COORDINATE.map(|n| n.get()),
        Some(2)
    );
}

/// ★ Rider 1's whole content: the draw consumes exactly `n` candidates,
/// wherever the acceptable one sits — including when there is none.
///
/// Counting the calls is the only way to see this; the returned value cannot
/// distinguish "took the first and stopped" from "took the first and kept
/// drawing", and it is the *stopping* that a straight-line machine cannot
/// follow.
#[test]
fn a_fixed_schedule_draw_consumes_exactly_n_candidates() {
    // `p = 2^64 - 2^32 + 1`, so anything ≥ p is rejected. `u64::MAX` is.
    let out_of_range = u64::MAX;
    assert!(!F::candidate_in_range(out_of_range));
    let in_range = [7u64, 11, 13, 17];
    for c in in_range {
        assert!(F::candidate_in_range(c));
    }

    for n in 1..=4usize {
        for hit in 0..n {
            // A stream whose only in-range value sits at position `hit`.
            let stream: Vec<u64> = (0..n)
                .map(|i| if i == hit { in_range[0] } else { out_of_range })
                .collect();
            let mut calls = 0usize;
            let mut it = stream.iter();
            let got = candidate_under_fixed_schedule::<F>(n, || {
                calls += 1;
                *it.next().expect("the schedule must not overdraw")
            });
            assert_eq!(
                calls, n,
                "n={n}, acceptable candidate at {hit}: the draw must consume exactly n"
            );
            assert_eq!(got, in_range[0], "it must return the acceptable candidate");
        }

        // No acceptable candidate: still exactly `n`, and the value handed back
        // is one the field rejects, so its own loop draws another full `n`.
        let mut calls = 0usize;
        let got = candidate_under_fixed_schedule::<F>(n, || {
            calls += 1;
            out_of_range
        });
        assert_eq!(calls, n, "n={n}, no acceptable candidate: still exactly n");
        assert!(
            !F::candidate_in_range(got),
            "the fallback must NOT be reduced into range — a modular fallback \
             would bias challenges by ~2^-32, which at this system's security \
             level would dominate the soundness error"
        );
    }
}

/// ★ The schedule as the transcript actually runs it: an extension-field draw
/// takes SIX candidates from the squeeze stream, two per coordinate.
///
/// Reconstructed from the raw squeezes, so it distinguishes the fixed schedule
/// from the unbounded one: under `None` the coordinates would be candidates
/// 0, 1, 2 of the stream, and under `Some(2)` they are the first acceptable of
/// (0,1), (2,3), (4,5).
#[test]
fn an_extension_draw_consumes_two_candidates_per_coordinate() {
    // The raw candidate stream this transcript will hand out, taken from a
    // clone so the transcript under test is untouched.
    let candidates: Vec<u64> = {
        let mut probe = Blake3Transcript::<E>::new(SEED);
        let mut out = Vec::new();
        for _ in 0..2 {
            let squeeze = probe.sample();
            for chunk in squeeze.chunks_exact(8) {
                out.push(u64::from_be_bytes(chunk.try_into().unwrap()));
            }
        }
        out
    };
    assert_eq!(candidates.len(), 8);

    let pick = |a: u64, b: u64| {
        if F::candidate_in_range(a) { a } else { b }
    };
    let expected = [
        pick(candidates[0], candidates[1]),
        pick(candidates[2], candidates[3]),
        pick(candidates[4], candidates[5]),
    ];

    let mut t = Blake3Transcript::<E>::new(SEED);
    let drawn = t.sample_field_element();
    let coords: Vec<u64> = drawn.value().iter().map(|c| *c.value()).collect();
    assert_eq!(
        coords,
        expected.to_vec(),
        "each coordinate must be the first acceptable of its OWN pair"
    );

    // NEGATIVE CONTROL: it is not the unbounded schedule, which would take one
    // candidate per coordinate and so read 0, 1, 2.
    let unbounded = [candidates[0], candidates[1], candidates[2]];
    assert_ne!(
        coords,
        unbounded.to_vec(),
        "the fixed schedule must be distinguishable from the unbounded one"
    );
}

/// The keccak configuration still draws one candidate per coordinate.
///
/// The honest-path partner of the test above: it says the branch is a branch,
/// and that the default side of it did not move.
#[test]
fn the_keccak_extension_draw_still_takes_one_candidate_per_coordinate() {
    let candidates: Vec<u64> = {
        let mut probe = DefaultTranscript::<E>::new(SEED);
        let squeeze = probe.sample();
        squeeze
            .chunks_exact(8)
            .map(|c| u64::from_be_bytes(c.try_into().unwrap()))
            .collect()
    };

    let mut t = DefaultTranscript::<E>::new(SEED);
    let coords: Vec<u64> = t
        .sample_field_element()
        .value()
        .iter()
        .map(|c| *c.value())
        .collect();
    assert_eq!(
        coords,
        candidates[..3].to_vec(),
        "the keccak draw must still be one candidate per coordinate"
    );
}

/// A transcript is still deterministic under the fixed schedule — the property
/// every replaying verifier depends on.
#[test]
fn the_blake3_transcript_replays_identically() {
    let mut a = Blake3Transcript::<E>::new(SEED);
    let mut b = Blake3Transcript::<E>::new(SEED);
    a.append_bytes(b"round-1");
    b.append_bytes(b"round-1");

    let draw_a: Vec<FieldElement<E>> = (0..8).map(|_| a.sample_field_element()).collect();
    let draw_b: Vec<FieldElement<E>> = (0..8).map(|_| b.sample_field_element()).collect();
    assert_eq!(draw_a, draw_b);
    assert_eq!(a.sample_u64(1 << 20), b.sample_u64(1 << 20));
}
