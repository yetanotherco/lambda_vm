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

// =========================================================================
// 64-byte squeezes (`Blake3StdSqueeze64TranscriptHash`)
// =========================================================================

mod squeeze64 {
    use alloc::vec::Vec;
    use math::field::{
        element::FieldElement, extensions_goldilocks::Degree3GoldilocksExtensionField,
        goldilocks::GoldilocksField, traits::HasDefaultTranscript,
    };

    use crate::fiat_shamir::default_transcript::DefaultTranscript;
    use crate::fiat_shamir::is_transcript::IsTranscript;
    use crate::fiat_shamir::transcript_hash::{
        Blake3StdSqueeze64TranscriptHash, Blake3StdTranscriptHash, TranscriptHash,
    };

    type F = GoldilocksField;
    type E = Degree3GoldilocksExtensionField;
    type Wide = DefaultTranscript<E, Blake3StdSqueeze64TranscriptHash>;
    type Narrow = DefaultTranscript<E, Blake3StdTranscriptHash>;

    const SEED: &[u8] = b"lambda-vm-blake3-squeeze64-kat";

    /// The `blake3` crate's first 64 bytes of extended output over `msg`.
    fn xof64(msg: &[u8]) -> [u8; 64] {
        let mut out = [0u8; 64];
        let mut hasher = blake3::Hasher::new();
        hasher.update(msg);
        hasher.finalize_xof().fill(&mut out);
        out
    }

    /// A squeeze of `msg` as the configuration hands it out, from the crate
    /// alone: each 32-byte half reversed, read as eight big-endian `u64`s;
    /// and the 32 bytes the chain re-absorbs.
    fn candidates_of(msg: &[u8]) -> ([u64; 8], [u8; 32]) {
        let mut out = xof64(msg);
        out[..32].reverse();
        out[32..].reverse();
        let c = core::array::from_fn(|k| {
            u64::from_be_bytes(out[8 * k..8 * k + 8].try_into().expect("8 bytes"))
        });
        (c, out[..32].try_into().expect("32 bytes"))
    }

    /// A coordinate under the two-candidate schedule: the first of the pair
    /// below `p`, else the second (which the field then rejects).
    fn coordinate(pair: [u64; 2]) -> u64 {
        if F::candidate_in_range(pair[0]) {
            pair[0]
        } else {
            pair[1]
        }
    }

    fn coords(x: &FieldElement<E>) -> [u64; 3] {
        let v = x.value();
        core::array::from_fn(|i| v[i].canonical())
    }

    /// ★ THE KAT: a wide transcript's challenges are the `blake3` crate's
    /// extended output, computed here from the crate alone. One squeeze of the
    /// seed hands out eight candidates: an extension challenge takes six, the
    /// next takes the last two before it squeezes again, and that squeeze
    /// hashes the first squeeze's 32 reversed digest bytes alone — its second
    /// half is never absorbed. The state (the grind's seed) is the crate's
    /// hash of exactly that.
    #[test]
    fn a_squeeze_hands_out_the_blake3_crates_64_byte_output() {
        let mut t = Wide::new(SEED);
        let (c0, absorbed0) = candidates_of(SEED);
        let x: FieldElement<E> = t.sample_field_element();
        assert_eq!(
            coords(&x),
            [
                coordinate([c0[0], c0[1]]),
                coordinate([c0[2], c0[3]]),
                coordinate([c0[4], c0[5]])
            ],
            "one squeeze's first six candidates"
        );
        assert_eq!(t.state(), *blake3::hash(&absorbed0).as_bytes());
        let (c1, _) = candidates_of(&absorbed0);
        let y: FieldElement<E> = t.sample_field_element();
        assert_eq!(
            coords(&y),
            [
                coordinate([c0[6], c0[7]]),
                coordinate([c1[0], c1[1]]),
                coordinate([c1[2], c1[3]])
            ],
            "the last two of the first squeeze, then the second's"
        );
    }

    /// The raw `sample()` and the grind's `state()` are the narrow
    /// configuration's: a wide squeeze adds output, never changes the chain.
    /// So the two transcripts agree on a squeeze's first four candidates and
    /// part at the fifth, where the narrow one squeezes again.
    #[test]
    fn the_chain_is_the_narrow_configurations_and_the_draws_part_at_the_fifth() {
        let (mut wide, mut narrow) = (Wide::new(SEED), Narrow::new(SEED));
        assert_eq!(wide.state(), narrow.state());
        assert_eq!(wide.sample(), narrow.sample());
        assert_eq!(wide.state(), narrow.state());
        wide.append_bytes(b"a-root");
        narrow.append_bytes(b"a-root");
        let draws = |t: &mut dyn FnMut() -> u64| -> Vec<u64> { (0..5).map(|_| t()).collect() };
        let w = draws(&mut || wide.sample_u64(1 << 40));
        let n = draws(&mut || narrow.sample_u64(1 << 40));
        assert_eq!(w[..4], n[..4], "the first four candidates are the same bytes");
        assert_ne!(w[4], n[4], "the fifth is the extended output, not a new squeeze");
    }

    /// The configuration is the narrow one but for the squeeze's width.
    #[test]
    fn only_the_squeeze_width_differs() {
        type W = Blake3StdSqueeze64TranscriptHash;
        type N = Blake3StdTranscriptHash;
        assert_eq!(W::REVERSES_SQUEEZE, N::REVERSES_SQUEEZE);
        assert_eq!(W::CANDIDATES_PER_COORDINATE, N::CANDIDATES_PER_COORDINATE);
        assert_eq!((W::SQUEEZE_LEN, N::SQUEEZE_LEN), (64, 32));
        assert_ne!(W::NAME, N::NAME);
    }
}
