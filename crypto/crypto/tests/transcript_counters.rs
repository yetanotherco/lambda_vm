//! ★★ The Fiat-Shamir counters answer "which sponge ran", on BOTH arms.
//!
//! ```text
//! cargo test -p crypto --features hash-metrics --test transcript_counters
//! ```
//!
//! # The trap these exist to close
//!
//! `hash_metrics`'s header already tells this story for Merkle: a counter that
//! compares `TypeId` against keccak and does nothing otherwise "became a check
//! that cannot fail the moment a second hash arrived — under RPX every Merkle
//! counter would have read ZERO, reporting *no hashing* for precisely the arm
//! whose purpose is to change the hashing".
//!
//! The transcript never got that treatment. `count_absorb` was bumped from one
//! place, the keccak wrapper's `update`; `Rpx256Digest::update` was a bare
//! `Vec::extend`; and `total`'s documentation claimed to count transcript
//! squeezes while nothing on the RPX side bumped it. So an RPX proof read zero
//! absorbs and zero transcript finalizes, and zero is exactly what a
//! *correctly instrumented* keccak-free run would also read. The measurement
//! could not distinguish "the other hash ran" from "nobody instrumented it".
//!
//! That is not hypothetical here: for four measured A/Bs the RPX arm ran a
//! KECCAK transcript, and no instrument disagreed.
//!
//! # So every assertion below is two-sided
//!
//! Each arm asserts both that its own counters MOVED and that the other arm's
//! are ZERO. One half alone is worthless: "rpx > 0, keccak 0" is equally true
//! of a run where the keccak transcript was never instrumented, which is the
//! state this file is about.
//!
//! # ⚠ Its own binary, AND a lock
//!
//! The counters are process-global, so both are needed and neither is enough.
//!
//! The binary, because `crypto`'s lib-test binary runs tests in parallel and
//! several of them hash: an exact assertion there is an assertion about
//! whatever else happened to be running — a keccak arm read 215 squeezes of
//! which 200 were keccak, purely from neighbours.
//!
//! The lock, because an integration test binary ALSO runs its own tests in
//! parallel. Moving the file and stopping there left four of five failing, one
//! reading `left: 2, right: 0` where a sibling had reset the counters between
//! this test's `reset` and its `snapshot`. Every test below takes
//! [`serialise`] for its whole reset-measure-assert window.

#![cfg(feature = "hash-metrics")]

use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use crypto::fiat_shamir::is_transcript::IsTranscript;
use crypto::fiat_shamir::transcript_hash::{KeccakTranscriptHash, RpxTranscriptHash};
use crypto::hash::rpx::Rpx256Digest;
use crypto::hash_metrics;
use digest::Update;
use math::field::element::FieldElement;
use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField as Ext;

/// ★ Taken by every test here: `reset` and `snapshot` address one global pair
/// of counters, so a measurement is only this test's while it holds this.
///
/// Poisoning is ignored so one failure does not cascade into unrelated tests.
static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn serialise() -> std::sync::MutexGuard<'static, ()> {
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

/// Four absorbs and two squeezes, driven identically on either configuration.
///
/// `new` absorbs once, so the count is `1 + 3`.
fn drive<T>(seed: &[u8]) -> (u64, u64)
where
    T: crypto::fiat_shamir::transcript_hash::TranscriptHash,
{
    let mut t = DefaultTranscript::<Ext, T>::new(seed);
    t.append_bytes(b"one");
    t.append_bytes(b"two");
    t.append_bytes(b"three");
    let _ = t.sample();
    let _ = t.sample();
    (4, 2)
}

#[test]
fn a_keccak_transcript_counts_as_keccak_and_nothing_else() {
    let _serialised = serialise();
    hash_metrics::reset();
    let (absorbs, squeezes) = drive::<KeccakTranscriptHash>(b"seed");
    let c = hash_metrics::snapshot();

    assert_eq!(c.transcript_absorbs_keccak, absorbs);
    assert_eq!(c.transcript_squeezes_keccak, squeezes);
    // The other arm must be silent — and it must be silent because nothing RPX
    // ran, which the totals below are what establish.
    assert_eq!(c.transcript_absorbs_rpx, 0);
    assert_eq!(c.transcript_squeezes_rpx, 0);
    assert_eq!(c.transcript_absorbs, absorbs);
    assert_eq!(c.transcript_squeezes, squeezes);
    assert_eq!(c.transcript_unattributed(), (0, 0, 0));
}

#[test]
fn an_rpx_transcript_counts_as_rpx_and_nothing_else() {
    let _serialised = serialise();
    hash_metrics::reset();
    let (absorbs, squeezes) = drive::<RpxTranscriptHash>(b"seed");
    let c = hash_metrics::snapshot();

    assert_eq!(
        c.transcript_absorbs_rpx, absorbs,
        "the RPX transcript absorbed {} times and the counter saw {} — an \
         un-instrumented sponge reads zero, which is indistinguishable from \
         one that never ran",
        absorbs, c.transcript_absorbs_rpx
    );
    assert_eq!(c.transcript_squeezes_rpx, squeezes);
    assert_eq!(
        c.transcript_absorbs_keccak, 0,
        "a keccak absorb during an RPX-only run: the transcript is not the \
         configuration's"
    );
    assert_eq!(c.transcript_squeezes_keccak, 0);
    assert_eq!(c.transcript_absorbs, absorbs);
    assert_eq!(c.transcript_squeezes, squeezes);
    assert_eq!(c.transcript_unattributed(), (0, 0, 0));
}

/// ★ The two configurations do the SAME amount of transcript work.
///
/// The counters must differ only in which bucket they land in. If a swap
/// changed the absorb count, the per-arm line would be reporting a protocol
/// difference as a hash difference — and a reader comparing arms would draw
/// the wrong conclusion about what the hash costs.
#[test]
fn the_two_configurations_do_the_same_transcript_work() {
    let _serialised = serialise();
    hash_metrics::reset();
    drive::<KeccakTranscriptHash>(b"same");
    let k = hash_metrics::snapshot();

    hash_metrics::reset();
    drive::<RpxTranscriptHash>(b"same");
    let r = hash_metrics::snapshot();

    assert_eq!(k.transcript_absorbs, r.transcript_absorbs);
    assert_eq!(k.transcript_squeezes, r.transcript_squeezes);
}

/// ★★ The transcript counter counts `Update::update` calls, tied to the
/// counter that already did.
///
/// On a keccak-only run `absorb_calls` sees every `update` the sponge receives
/// and this file's counter sees every one the TRANSCRIPT issues. They differ by
/// exactly the chaining re-absorb — `sample` feeds its own output back in, once
/// per squeeze, which is part of squeezing and not an absorb anyone asked for.
/// So:
///
/// ```text
/// absorb_calls == transcript_absorbs + transcript_squeezes
/// ```
///
/// This is the assertion that makes the new counters non-vacuous against
/// something that predates them, and it fails if either counter is moved,
/// double-counted, or attached to the wrong call.
///
/// ⚠ SCOPE: state reads are deliberately NOT a term here, and adding one would
/// be wrong rather than more complete. `state()` finalizes a clone; it issues
/// no `update`, so it cannot move `absorb_calls` and has no business in an
/// identity about absorbs. It is counted separately by
/// [`a_state_read_is_counted_separately_from_a_squeeze`], where its own control
/// is the grind count.
///
/// Lane V1's closed form for the block, in these terms and satisfying this
/// identity by construction: 582,703 absorbs + 182,734 squeezes == 765,437
/// `absorb_calls`, with the 2,996 state finalizes outside all three.
///
/// ⚠ An earlier version of this test asserted a cubic element absorbs in
/// "several chunks". It does not: `stream_bytes` for the degree-3 extension
/// writes one 24-byte buffer and calls the sink ONCE
/// (`extensions_goldilocks.rs:567-571`). The test failed, which is how the
/// claim — and a comment in `default_transcript.rs` repeating it — got fixed.
#[test]
fn the_transcript_absorbs_agree_with_the_generic_absorb_counter() {
    let _serialised = serialise();
    hash_metrics::reset();
    let (absorbs, squeezes) = drive::<KeccakTranscriptHash>(b"tie");
    let c = hash_metrics::snapshot();

    assert_eq!(
        c.absorb_calls,
        c.transcript_absorbs + c.transcript_squeezes,
        "the keccak sponge saw {} updates; the transcript issued {} absorbs and \
         {} squeezes, and a squeeze chains exactly one update",
        c.absorb_calls,
        c.transcript_absorbs,
        c.transcript_squeezes
    );
    assert_eq!(
        (c.transcript_absorbs, c.transcript_squeezes),
        (absorbs, squeezes)
    );
}

/// ★ A field element costs the same absorbs on either arm.
///
/// The counters must differ only in which bucket they land in: if a hash swap
/// changed the absorb count, the per-arm line would report a protocol
/// difference as a hash difference.
#[test]
fn a_field_element_absorb_costs_the_same_on_both_arms() {
    let _serialised = serialise();
    let element = FieldElement::<Ext>::from(7u64);

    hash_metrics::reset();
    let mut k = DefaultTranscript::<Ext, KeccakTranscriptHash>::new(&[]);
    k.append_field_element(&element);
    let k = hash_metrics::snapshot();

    hash_metrics::reset();
    let mut r = DefaultTranscript::<Ext, RpxTranscriptHash>::new(&[]);
    r.append_field_element(&element);
    let r = hash_metrics::snapshot();

    assert_eq!(
        k.transcript_absorbs_keccak, r.transcript_absorbs_rpx,
        "the two arms disagree on how much one element absorbs"
    );
    assert_eq!(k.transcript_absorbs_rpx, 0);
    assert_eq!(r.transcript_absorbs_keccak, 0);
}

/// ★★ …and the GENERIC counters are no longer keccak-only either.
///
/// `count_absorb` had exactly one call site — the keccak wrapper's `update` —
/// so `absorb_calls` and `absorb_bytes` read ZERO for an RPX proof, and
/// `total`'s documentation ("every finalize … transcript squeeze") was false
/// for this sponge. This drives the digest directly, so it fails if either call
/// is removed.
#[test]
fn the_rpx_digest_bumps_the_generic_counters() {
    let _serialised = serialise();
    hash_metrics::reset();
    let mut d = Rpx256Digest::default();
    Update::update(&mut d, b"twelve bytes");
    Update::update(&mut d, b"and more");
    let after_absorbs = hash_metrics::snapshot();

    assert_eq!(
        after_absorbs.absorb_calls, 2,
        "absorbing into an RPX digest moved `absorb_calls` to {} — it read 0 \
         before this was instrumented, for every RPX proof ever measured",
        after_absorbs.absorb_calls
    );
    assert_eq!(after_absorbs.absorb_bytes, 12 + 8);
    assert_eq!(
        after_absorbs.total, 0,
        "nothing has been finalized yet, so `total` must not have moved"
    );

    use digest::FixedOutputReset;
    let mut out = digest::Output::<Rpx256Digest>::default();
    d.finalize_into_reset(&mut out);
    assert_eq!(
        hash_metrics::snapshot().total,
        1,
        "an RPX finalize did not reach `total`, whose own doc says it counts \
         every finalize"
    );
}

/// ★★★ A `state()` IS COUNTED, AND IT IS NOT A SQUEEZE.
///
/// The distinction lane V1's closed form turns on. `sample` is a
/// `finalize_reset` whose output is chained back in, so it advances the
/// transcript; `state` finalizes a CLONE and changes nothing. On a block proof
/// they are 182,734 and 2,996 — a counter hooked only to `finalize_reset`
/// misses every one of the 2,996, which is precisely what this file's first
/// version did.
///
/// Both directions are asserted, because either conflation is a live failure
/// mode: a state must not appear as a squeeze, AND a squeeze must not appear as
/// a state. Counting their sum would satisfy neither of V1's two numbers.
#[test]
fn a_state_read_is_counted_separately_from_a_squeeze() {
    let _serialised = serialise();

    hash_metrics::reset();
    let mut t = DefaultTranscript::<Ext, RpxTranscriptHash>::new(b"state");
    let before = t.state();
    let c = hash_metrics::snapshot();
    assert_eq!(
        (c.transcript_states, c.transcript_states_rpx),
        (1, 1),
        "a state() read was not counted"
    );
    assert_eq!(
        c.transcript_squeezes, 0,
        "a state() read was counted as a squeeze — it finalizes a clone and \
         advances nothing, so a squeeze count including it cannot be checked \
         against a closed form"
    );

    // …and it really did not advance the chain, which is why it is a different
    // number rather than a different name for the same one.
    let again = t.state();
    assert_eq!(before, again, "state() advanced the transcript");
    assert_eq!(hash_metrics::snapshot().transcript_states, 2);

    // The converse: a squeeze is not counted as a state.
    hash_metrics::reset();
    let _ = t.sample();
    let c = hash_metrics::snapshot();
    assert_eq!(
        (c.transcript_squeezes, c.transcript_squeezes_rpx),
        (1, 1),
        "a squeeze was not counted"
    );
    assert_eq!(
        c.transcript_states, 0,
        "a squeeze was counted as a state read"
    );
    assert_eq!(c.transcript_unattributed(), (0, 0, 0));
}

/// ★ …and the keccak arm tags its state reads too.
///
/// One side is not evidence: "rpx states > 0, keccak 0" is equally true when
/// the keccak path was never instrumented.
#[test]
fn a_keccak_state_read_is_tagged_as_keccak() {
    let _serialised = serialise();

    hash_metrics::reset();
    let t = DefaultTranscript::<Ext, KeccakTranscriptHash>::new(b"state");
    let _ = t.state();
    let c = hash_metrics::snapshot();

    assert_eq!((c.transcript_states, c.transcript_states_keccak), (1, 1));
    assert_eq!(c.transcript_states_rpx, 0);
    assert_eq!(c.transcript_unattributed(), (0, 0, 0));
}

// =========================================================================
// W1-B: where the derived root goes
// =========================================================================

/// ★★★ THE ABSORB POSITION — the orderings are distinguishable, and the
/// challenge sees it.
///
/// W1-B adds one root the proof does not carry: the verifier derives it from
/// the ELF and absorbs it AFTER the roots the proof does carry, and BEFORE any
/// challenge. Two guards were proposed for that ordering and neither reaches
/// it:
///
/// * a transcript-count gate is **blind to position** — an absorb lands at
///   `out_pos == SQUEEZE_LEN` wherever it goes, so first and last cost an
///   identical `+1 absorb, +0 squeezes`. Demonstrated at the end of this test
///   rather than asserted in prose;
/// * the WHIR byte gate is **blind to the feature** — its fixture is a single
///   EQ air (`whir_byte_gate.rs:167,190`), so it commits no DECODE table, no
///   preprocessed group and no derived root at all.
///
/// Nothing downstream catches it either: prover and verifier share the
/// position, so a consistently wrong one still verifies. **A round-trip between
/// two halves that share code is not evidence about either.**
///
/// # WHAT THIS TEST DOES NOT DO
///
/// It does **not** call the production roots block, so it cannot tell you the
/// verifier absorbs anything at all. It pins that the specified order is
/// *distinguishable* from every wrong one — including from omitting the root
/// entirely — which is the property the count gate lacks and which has to hold
/// before a wiring guard can mean anything.
///
/// The wiring guard arrives at **step 3**, when `multi_verify`'s roots block
/// exists to be called; its mutation is deleting the absorb, and that is the
/// failure this branch has produced twice — the transcript defaulting to keccak
/// while everything else moved, and the grind dispatching keccak-only with a
/// correct RPX kernel sitting unused. **Both were the feature not wired, not
/// the feature wired wrongly.** Until step 3, this file pins the specification
/// and the design note's ordering section is the normative statement.
/// What one run of the roots-block script leaves behind: the sponge state, and
/// the first challenge drawn from it. Named because the two travel together —
/// a failure in the state and a failure in the challenge mean different things.
type Draw = ([u8; 32], FieldElement<Ext>);

#[test]
fn the_derived_root_is_absorbed_after_the_carried_ones() {
    let _serialised = serialise();

    // Stand-ins: what matters is the ORDER, not the values.
    let carried: [[u8; 32]; 3] = [[0x11; 32], [0x22; 32], [0x33; 32]];
    let derived = [0xAB; 32];

    // The script runs past the absorbs to the FIRST CHALLENGE, because that is
    // where the property has teeth: "every root is in the transcript before the
    // first challenge". Pinning the state alone would pin the absorbs' order
    // among themselves and leave "absorbed AFTER z" — the genuinely dangerous
    // variant — unrepresentable.
    //
    // `z` comes back beside the state so a failure localises: "the challenge
    // moved" reads differently from "the state moved".
    let script = |order: &[&[u8; 32]], after_z: Option<&[u8; 32]>| -> Draw {
        let mut t = DefaultTranscript::<Ext, RpxTranscriptHash>::new(b"w1b-roots");
        for root in order {
            t.append_bytes(*root);
        }
        let z: FieldElement<Ext> = t.sample_field_element();
        if let Some(late) = after_z {
            t.append_bytes(late);
        }
        (IsTranscript::<Ext>::state(&t), z)
    };

    let all = [&carried[0], &carried[1], &carried[2], &derived];
    let (specified, z_specified) = script(&all, None);

    // Every wrong placement, including the two that are not reorderings.
    let wrong: [(&str, Draw); 4] = [
        (
            "first",
            script(&[&derived, &carried[0], &carried[1], &carried[2]], None),
        ),
        (
            "middle",
            script(&[&carried[0], &derived, &carried[1], &carried[2]], None),
        ),
        // OMITTED — the failure mode that is not a reordering at all.
        (
            "omitted",
            script(&[&carried[0], &carried[1], &carried[2]], None),
        ),
        // AFTER the challenge — unrepresentable without the `z` draw above.
        (
            "after z",
            script(&[&carried[0], &carried[1], &carried[2]], Some(&derived)),
        ),
    ];

    for (name, (state, z)) in &wrong {
        assert_ne!(
            specified, *state,
            "placing the derived root {name} gives the same sponge state as the \
             specified order — the transcript cannot distinguish them, so this \
             test pins nothing"
        );
        if *name != "after z" {
            assert_ne!(
                z_specified, *z,
                "placing the derived root {name} draws the SAME challenge — the \
                 root is not binding z, which is the whole point of absorbing it \
                 before any challenge"
            );
        }
    }

    // ★★ THE DANGER, STATED AS AN EQUALITY: absorbing after the challenge is
    // indistinguishable IN THE CHALLENGE from not absorbing at all. Both draw
    // `z` from a transcript holding only the carried roots, so the derived root
    // binds nothing — while the counters happily see it and the sponge state
    // afterwards differs from the omitted case, which is what makes it look
    // wired when it is not.
    //
    // ⚠ An earlier draft asserted this against `z_specified` with the comment
    // "the same z BY CONSTRUCTION". That was wrong and the test said so: the
    // specified order absorbs FOUR roots before drawing, the late order absorbs
    // three, so the two draws differ. The equality that holds — and the one
    // worth pinning — is against the OMITTED case.
    assert_eq!(
        wrong[3].1.1, wrong[2].1.1,
        "absorbing after the challenge draws a different z from omitting the \
         root entirely — then a late absorb would be detectable in z, and the \
         reason this ordering is dangerous is not the one stated here"
    );

    // The count gate's blindness, demonstrated rather than asserted.
    hash_metrics::reset();
    let _ = script(&all, None);
    let last = hash_metrics::snapshot();
    hash_metrics::reset();
    let _ = script(&[&derived, &carried[0], &carried[1], &carried[2]], None);
    let first = hash_metrics::snapshot();
    assert_eq!(
        (last.transcript_absorbs, last.transcript_squeezes),
        (first.transcript_absorbs, first.transcript_squeezes),
        "the two orders differ in counters after all — then a count gate DOES see \
         position and this test's premise is wrong"
    );
}
