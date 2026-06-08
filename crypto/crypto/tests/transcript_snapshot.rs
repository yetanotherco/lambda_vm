//! `DefaultTranscript::clone` snapshot semantics. The GPU FRI fallback in
//! `stark::gpu_lde::try_fri_commit_gpu` relies on Clone being a true
//! byte-identical snapshot so a mid-loop cudarc failure can restore the
//! transcript to its pre-loop state and let the CPU path run as if the
//! GPU had never been touched. If this contract ever breaks, that
//! fallback silently produces a transcript-divergent proof.

use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use crypto::fiat_shamir::is_transcript::IsTranscript;
use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField;

type E = Degree3GoldilocksExtensionField;

/// Clone must be a true snapshot: restoring `*t = snap.clone()` after
/// arbitrary mutations must put the transcript in a state byte-identical
/// to a fresh clone of the snapshot.
#[test]
fn clone_then_restore_is_byte_identical_snapshot() {
    let mut t = DefaultTranscript::<E>::new(b"seed");
    let snap = t.clone();

    // Arbitrary mutation: sample + append.
    let _ = t.sample_field_element();
    t.append_bytes(b"some bytes");
    let _ = t.sample_field_element();
    t.append_bytes(b"more bytes");

    // Restore from the snapshot.
    t = snap.clone();

    // A fresh clone of the snapshot must produce the same outputs.
    let mut reference = snap.clone();
    assert_eq!(t.state(), reference.state(), "state diverged after restore");

    let a1 = t.sample_field_element();
    let b1 = reference.sample_field_element();
    assert_eq!(a1, b1, "sample_field_element diverged after restore");

    t.append_bytes(b"x");
    reference.append_bytes(b"x");
    assert_eq!(
        t.state(),
        reference.state(),
        "state diverged after parallel append"
    );

    let a2 = t.sample_field_element();
    let b2 = reference.sample_field_element();
    assert_eq!(a2, b2, "second sample diverged after restore");
}
