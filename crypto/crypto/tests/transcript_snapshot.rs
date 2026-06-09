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

/// Clone must be a true snapshot: it must be unaffected by subsequent
/// mutations of the source transcript, and restoring `*t = snap.clone()`
/// must put the transcript in a state byte-identical to the moment the
/// snapshot was taken.
#[test]
fn clone_then_restore_is_byte_identical_snapshot() {
    let mut t = DefaultTranscript::<E>::new(b"seed");
    let _ = t.sample_field_element();
    t.append_bytes(b"prelude");

    let entry = t.state();
    // Snapshot from a non-trivial state.
    let snap = t.clone();

    // Mutate the source after the snapshot.
    let _ = t.sample_field_element();
    t.append_bytes(b"more bytes");
    assert_eq!(snap.state(), entry, "snapshot disturbed by source mutation");

    // Restore from the snapshot.
    t = snap.clone();
    assert_eq!(t.state(), entry, "restore not byte-identical to entry");
}
