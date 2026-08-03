use alloc::vec::Vec;
use math::field::{
    element::FieldElement, extensions_goldilocks::Degree3GoldilocksExtensionField,
    goldilocks::GoldilocksField,
};

use crate::fiat_shamir::default_transcript::DefaultTranscript;
use crate::fiat_shamir::is_transcript::IsTranscript;

#[test]
fn basic_challenge() {
    let mut transcript = DefaultTranscript::<Degree3GoldilocksExtensionField>::default();

    let point_a: Vec<u8> = vec![0xFF, 0xAB];
    let point_b: Vec<u8> = vec![0xDD, 0x8C, 0x9D];

    transcript.append_bytes(&point_a); // point_a
    transcript.append_bytes(&point_b); // point_a + point_b

    let challenge1 = transcript.sample(); // Hash(point_a  + point_b)

    assert_eq!(
        challenge1,
        [
            0x0c, 0x2b, 0xd8, 0xcf, 0x2d, 0x71, 0xe0, 0x0a, 0xce, 0xa3, 0xbd, 0x5d, 0xc7, 0x9f,
            0x4f, 0x93, 0xed, 0x57, 0x42, 0xd0, 0x23, 0xbd, 0x47, 0xc9, 0x04, 0xc2, 0x67, 0x9d,
            0xbc, 0xfa, 0x7c, 0xa7
        ]
    );

    let point_c: Vec<u8> = vec![0xFF, 0xAB];
    let point_d: Vec<u8> = vec![0xDD, 0x8C, 0x9D];

    transcript.append_bytes(&point_c); // Hash(point_a  + point_b) + point_c
    transcript.append_bytes(&point_d); // Hash(point_a  + point_b) + point_c + point_d

    let challenge2 = transcript.sample(); // Hash(Hash(point_a  + point_b) + point_c + point_d)
    assert_eq!(
        challenge2,
        [
            0x81, 0x61, 0x51, 0xc5, 0x7e, 0xcb, 0x45, 0xd5, 0x17, 0x1a, 0x3c, 0x2e, 0x38, 0x04,
            0x5d, 0xfb, 0x3a, 0x3d, 0x33, 0x8a, 0x22, 0xaf, 0xf8, 0x60, 0x85, 0xb9, 0x54, 0x3f,
            0xf8, 0x32, 0x32, 0xbc
        ]
    );
}

type GoldFE = FieldElement<GoldilocksField>;
type Ext3FE = FieldElement<Degree3GoldilocksExtensionField>;

#[test]
fn degree3_goldilocks_transcript_distinguish_different_fe() {
    let mut transcript_1 = DefaultTranscript::<Degree3GoldilocksExtensionField>::default();
    transcript_1.append_field_element(&Ext3FE::new([
        GoldFE::one(),
        GoldFE::zero(),
        GoldFE::zero(),
    ]));
    let sample_1 = transcript_1.sample_field_element();

    let mut transcript_2 = DefaultTranscript::<Degree3GoldilocksExtensionField>::default();
    transcript_2.append_field_element(&Ext3FE::new([
        GoldFE::zero(),
        GoldFE::zero(),
        GoldFE::one(),
    ]));
    let sample_2 = transcript_2.sample_field_element();

    let mut transcript_3 = DefaultTranscript::<Degree3GoldilocksExtensionField>::default();
    transcript_3.append_field_element(&Ext3FE::new([
        GoldFE::one(),
        GoldFE::zero(),
        GoldFE::zero(),
    ]));
    let sample_3 = transcript_3.sample_field_element();

    assert!(sample_1 != sample_2);
    assert!(sample_1 == sample_3);
}

#[test]
fn fork_determinism() {
    // Cloning a transcript twice and running the same operations must produce identical challenges.
    let mut base = DefaultTranscript::<GoldilocksField>::default();
    base.append_bytes(&[0x01, 0x02, 0x03]);
    let _ = base.sample();
    base.append_bytes(&[0xAA, 0xBB]);

    let mut fork_a = base.clone();
    let mut fork_b = base.clone();

    fork_a.append_bytes(&[0x00]);
    fork_b.append_bytes(&[0x00]);

    assert_eq!(fork_a.sample(), fork_b.sample());
    assert_eq!(fork_a.sample(), fork_b.sample());
}

#[test]
fn fork_domain_separator_differentiates() {
    // Two forks from the same base with different domain separators must produce different challenges.
    let mut base = DefaultTranscript::<GoldilocksField>::default();
    base.append_bytes(&[0x01, 0x02, 0x03]);
    let _ = base.sample();
    base.append_bytes(&[0xAA, 0xBB]);

    let mut fork_0 = base.clone();
    fork_0.append_bytes(&(0u64).to_le_bytes());

    let mut fork_1 = base.clone();
    fork_1.append_bytes(&(1u64).to_le_bytes());

    assert_ne!(fork_0.sample(), fork_1.sample());
}

#[test]
fn sample_u64_consecutive_calls_return_different_values() {
    let mut transcript = DefaultTranscript::<Degree3GoldilocksExtensionField>::default();
    transcript.append_bytes(&[0x01, 0x02, 0x03]);

    let sample1 = transcript.sample_u64(1000);
    let sample2 = transcript.sample_u64(1000);
    let sample3 = transcript.sample_u64(1000);

    assert!(sample1 < 1000);
    assert!(sample2 < 1000);
    assert!(sample3 < 1000);

    assert_ne!(
        sample1, sample2,
        "consecutive sample_u64 calls should return different values"
    );
    assert_ne!(
        sample2, sample3,
        "consecutive sample_u64 calls should return different values"
    );
}

#[test]
fn sample_u64_upper_bound_one_always_returns_zero() {
    let mut transcript = DefaultTranscript::<Degree3GoldilocksExtensionField>::default();
    transcript.append_bytes(&[0x01, 0x02, 0x03]);

    for _ in 0..10 {
        assert_eq!(transcript.sample_u64(1), 0);
    }
}

#[test]
fn fork_isolation() {
    // Appending data to one fork must not affect challenges sampled from another.
    let mut base = DefaultTranscript::<GoldilocksField>::default();
    base.append_bytes(&[0x01, 0x02, 0x03]);
    let _ = base.sample();

    let mut fork_a = base.clone();
    fork_a.append_bytes(&(0u64).to_le_bytes());

    let mut fork_b = base.clone();
    fork_b.append_bytes(&(1u64).to_le_bytes());

    // Pollute fork_b with extra data
    fork_b.append_bytes(&[0xFF; 64]);
    let _ = fork_b.sample();
    fork_b.append_bytes(&[0xEE; 128]);

    // fork_a should still produce the same challenge as a fresh identical fork
    let mut fork_a_fresh = base.clone();
    fork_a_fresh.append_bytes(&(0u64).to_le_bytes());

    assert_eq!(fork_a.sample(), fork_a_fresh.sample());
}

// =========================================================================
// Duplex output-buffer contract (the soundness-critical invalidation lines).
//
// The roundtrip suites structurally cannot catch a missing invalidation:
// prover and verifier would consume identical stale bytes in lockstep. Each
// test below fails if its invalidation is removed, because the "next" sample
// would then come from bytes squeezed BEFORE the interleaved absorb — i.e.
// a challenge that does not depend on the absorbed commitment.
// =========================================================================

#[test]
fn absorb_bytes_invalidates_buffered_squeeze_output() {
    let mut t1 = DefaultTranscript::<GoldilocksField>::new(b"seed");
    let mut t2 = DefaultTranscript::<GoldilocksField>::new(b"seed");
    // Fill the buffer and consume one candidate on both.
    assert_eq!(t1.sample_field_element(), t2.sample_field_element());
    // Diverge the absorbed input; the next challenge must depend on it.
    t1.append_bytes(b"root-A");
    t2.append_bytes(b"root-B");
    assert_ne!(
        t1.sample_field_element(),
        t2.sample_field_element(),
        "a challenge sampled after an absorb must depend on the absorbed bytes"
    );
}

#[test]
fn absorb_field_element_invalidates_buffered_squeeze_output() {
    let mut t1 = DefaultTranscript::<GoldilocksField>::new(b"seed");
    let mut t2 = DefaultTranscript::<GoldilocksField>::new(b"seed");
    assert_eq!(t1.sample_field_element(), t2.sample_field_element());
    t1.append_field_element(&FieldElement::from(1u64));
    t2.append_field_element(&FieldElement::from(2u64));
    assert_ne!(
        t1.sample_field_element(),
        t2.sample_field_element(),
        "a challenge sampled after absorbing a field element must depend on it"
    );
}

#[test]
fn raw_sample_invalidates_buffered_squeeze_output() {
    let mut t1 = DefaultTranscript::<GoldilocksField>::new(b"seed");
    let mut t2 = DefaultTranscript::<GoldilocksField>::new(b"seed");
    assert_eq!(t1.sample_field_element(), t2.sample_field_element());
    // Interleave a raw squeeze on t1 only (the grinding path does this).
    let _ = t1.sample();
    assert_ne!(
        t1.sample_field_element(),
        t2.sample_field_element(),
        "a raw sample() must invalidate buffered bytes, not hand them out again"
    );
}

/// The GPU-FRI fallback clones the transcript mid-buffer; a clone that loses
/// `out_buf`/`out_pos` would replay a different challenge sequence there.
#[test]
fn clone_replays_identically_mid_buffer() {
    let mut t = DefaultTranscript::<GoldilocksField>::new(b"snapshot");
    let _ = t.sample_field_element(); // leave the buffer partially consumed
    let mut snap = t.clone();
    let original: (Vec<FieldElement<GoldilocksField>>, u64) = (
        (0..6).map(|_| t.sample_field_element()).collect(),
        t.sample_u64(1 << 20),
    );
    let replay: (Vec<FieldElement<GoldilocksField>>, u64) = (
        (0..6).map(|_| snap.sample_field_element()).collect(),
        snap.sample_u64(1 << 20),
    );
    assert_eq!(
        original, replay,
        "a mid-buffer clone must replay identically"
    );
}

/// Known-answer pin of the duplex byte semantics: BE u64 candidates, 8 bytes
/// per candidate, refill after 4, absorb invalidation between phases. Any
/// accidental change to byte order, chunking or refill granularity is a
/// transcript hard-fork and must show up here, not in a red proof.
#[test]
fn pinned_duplex_sample_semantics_across_refill() {
    let mut t = DefaultTranscript::<GoldilocksField>::new(b"lambda-vm-kat-v1");
    // Five base samples: the fifth forces a refill (4 candidates per squeeze).
    let base: Vec<u64> = (0..5).map(|_| *t.sample_field_element().value()).collect();
    assert_eq!(base, KAT_BASE);
    // A bounded index draw from the same buffered stream.
    assert_eq!(t.sample_u64(1 << 20), KAT_U64);
    // An ext3 sample after an absorb (invalidation + coordinate order).
    let mut te = DefaultTranscript::<Degree3GoldilocksExtensionField>::new(b"lambda-vm-kat-v1");
    te.append_bytes(b"phase-2");
    let ext = te.sample_field_element();
    let coords: Vec<u64> = ext.value().iter().map(|c| *c.value()).collect();
    assert_eq!(coords, KAT_EXT3);
}

const KAT_BASE: [u64; 5] = [
    14480544354348864378,
    16386050731901120766,
    7548241632395108276,
    4782457473227177333,
    12741265158531607555,
];
const KAT_U64: u64 = 661275;
const KAT_EXT3: [u64; 3] = [
    1422269417846962659,
    13550644288133318291,
    8414859559479507538,
];
