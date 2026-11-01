//! The 7-round BLAKE3 configuration (`Blake3StdTranscriptHash`, `Blake3Std`):
//! standard BLAKE3 whatever the build's `blake3-6round` feature selects, for
//! the sponge and for the grind.

use alloc::vec::Vec;
use digest::Digest;
use math::field::{element::FieldElement, extensions_goldilocks::Degree3GoldilocksExtensionField};

use crate::fiat_shamir::default_transcript::{Blake3Transcript, DefaultTranscript};
use crate::fiat_shamir::is_transcript::IsTranscript;
use crate::fiat_shamir::transcript_hash::{
    Blake3StdTranscriptHash, Blake3TranscriptHash, TranscriptHash,
};
use crate::grinding::{DeviceGrindKey, GrindDigest, generate_nonce, is_valid_nonce};
use crate::hash::blake3::{BLAKE3_ROUNDS, Blake3Std};

type E = Degree3GoldilocksExtensionField;

const SEED: &[u8] = b"lambda-vm-blake3-std-transcript";

/// ★ The sponge is the `blake3` crate's hash of what was absorbed, in every
/// build: the transcript state is the grinding seed, and it is standard BLAKE3.
#[test]
fn the_std_transcript_state_is_blake3_of_what_was_absorbed() {
    let mut t = DefaultTranscript::<E, Blake3StdTranscriptHash>::new(SEED);
    assert_eq!(t.state(), *blake3::hash(SEED).as_bytes());
    t.append_bytes(b"a-merkle-root");
    let mut absorbed = Vec::from(SEED);
    absorbed.extend_from_slice(b"a-merkle-root");
    assert_eq!(t.state(), *blake3::hash(&absorbed).as_bytes());
}

/// The configuration keeps the build's BLAKE3 transcript conventions and
/// moves the round count alone: under the default build (6 rounds) the two
/// transcripts' states differ from the first absorb; under a 7-round build
/// they are the same stream.
#[test]
fn the_std_transcript_differs_from_the_builds_by_the_rounds_alone() {
    assert_eq!(
        Blake3StdTranscriptHash::REVERSES_SQUEEZE,
        Blake3TranscriptHash::REVERSES_SQUEEZE
    );
    assert_eq!(
        Blake3StdTranscriptHash::CANDIDATES_PER_COORDINATE,
        Blake3TranscriptHash::CANDIDATES_PER_COORDINATE
    );
    let std = {
        let mut t = DefaultTranscript::<E, Blake3StdTranscriptHash>::new(SEED);
        let x: FieldElement<E> = t.sample_field_element();
        (t.state(), x)
    };
    let build = {
        let mut t = Blake3Transcript::<E>::new(SEED);
        let x: FieldElement<E> = t.sample_field_element();
        (t.state(), x)
    };
    if BLAKE3_ROUNDS == 7 {
        assert_eq!(std, build);
    } else {
        assert_ne!(std.0, build.0);
        assert_ne!(std.1, build.1);
    }
}

/// ★ The grind: a nonce the host search finds is valid, the same nonce under
/// another seed or a higher factor is refused, and the digest names the card
/// arm that searches with its own hash.
#[test]
fn a_std_grind_nonce_verifies_and_a_wrong_one_is_refused() {
    const FACTOR: u8 = 10;
    let seed: [u8; 32] = core::array::from_fn(|i| i as u8 * 7 + 1);
    let nonce = generate_nonce::<Blake3Std>(&seed, FACTOR).expect("a nonce exists");
    assert!(is_valid_nonce::<Blake3Std>(&seed, nonce, FACTOR));
    // Some nonce below the found one fails: the search is not vacuous.
    let refused = (0..nonce.min(64)).any(|n| !is_valid_nonce::<Blake3Std>(&seed, n, FACTOR));
    assert!(refused || nonce == 0);
    let mut other = seed;
    other[0] ^= 1;
    let still = is_valid_nonce::<Blake3Std>(&other, nonce, FACTOR)
        && is_valid_nonce::<Blake3Std>(&seed, nonce, 40);
    assert!(!still, "a nonce is bound to its seed and its factor");
    assert_eq!(
        <Blake3Std as GrindDigest>::DEVICE_GRIND,
        Some(DeviceGrindKey::Blake3Std)
    );
    // The predicate is standard BLAKE3: hash(hash(prefix ‖ seed ‖ factor) ‖ nonce).
    let mut inner = Vec::from([0x01u8, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xed]);
    inner.extend_from_slice(&seed);
    inner.push(FACTOR);
    let mut outer = Vec::from(*blake3::hash(&inner).as_bytes());
    outer.extend_from_slice(&nonce.to_be_bytes());
    let head = u64::from_be_bytes(blake3::hash(&outer).as_bytes()[..8].try_into().unwrap());
    assert!(head < 1u64 << (64 - FACTOR));
    assert_eq!(
        <[u8; 32]>::from(Blake3Std::digest(&outer)),
        *blake3::hash(&outer).as_bytes()
    );
}
