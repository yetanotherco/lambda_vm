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
    assert_eq!(c.transcript_unattributed(), (0, 0));
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
    assert_eq!(c.transcript_unattributed(), (0, 0));
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
