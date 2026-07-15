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

#[test]
fn sample_bits_values_are_in_range() {
    let mut t = DefaultTranscript::<Degree3GoldilocksExtensionField>::new(&[0x11, 0x22, 0x33]);
    let bits = 24;
    for _ in 0..200 {
        let v = t.sample_bits(bits);
        assert!(v < (1u64 << bits), "sample_bits({bits}) out of range: {v}");
    }
}

#[test]
fn sample_bits_is_deterministic_across_clones() {
    let base = DefaultTranscript::<Degree3GoldilocksExtensionField>::new(&[0xAB; 8]);
    let mut a = base.clone();
    let mut b = base.clone();
    let seq_a: Vec<u64> = (0..50).map(|_| a.sample_bits(21)).collect();
    let seq_b: Vec<u64> = (0..50).map(|_| b.sample_bits(21)).collect();
    assert_eq!(
        seq_a, seq_b,
        "identical transcript state must yield identical query bits (prover/verifier agreement)"
    );
}

#[test]
fn sample_bits_covers_full_small_range() {
    let mut t = DefaultTranscript::<Degree3GoldilocksExtensionField>::new(&[0x42]);
    let bits = 3; // range [0, 8)
    let mut seen = [false; 8];
    for _ in 0..500 {
        seen[t.sample_bits(bits) as usize] = true;
    }
    assert!(
        seen.iter().all(|&s| s),
        "sample_bits(3) should reach all 8 values (unbiased masking)"
    );
}

#[test]
fn absorbing_clears_sample_bits_reservoir() {
    // No absorb between two samples: the second reuses buffered bytes, no new squeeze.
    let mut t = DefaultTranscript::<Degree3GoldilocksExtensionField>::new(&[0x09]);
    let _ = t.sample_bits(24);
    let mid = t.keccak_squeezes();
    let _ = t.sample_bits(24);
    assert_eq!(
        t.keccak_squeezes(),
        mid,
        "a second sample_bits should reuse the reservoir (no extra Keccak squeeze)"
    );

    // Absorb between: the reservoir is invalidated and the next sample re-squeezes.
    let mut t = DefaultTranscript::<Degree3GoldilocksExtensionField>::new(&[0x09]);
    let _ = t.sample_bits(24);
    t.append_bytes(&[0xEE]);
    let mid = t.keccak_squeezes();
    let _ = t.sample_bits(24);
    assert_eq!(
        t.keccak_squeezes(),
        mid + 1,
        "an absorb must clear the reservoir, forcing a fresh squeeze so bits stay bound to it"
    );
}

#[test]
fn sample_bits_amortizes_keccak_squeezes() {
    let num_queries = 80usize;
    let bits = 24usize; // 3 bytes per query

    // New path: sample_bits from the shared reservoir.
    let mut t_bits = DefaultTranscript::<Degree3GoldilocksExtensionField>::new(&[0x01, 0x02]);
    let before = t_bits.keccak_squeezes();
    for _ in 0..num_queries {
        let _ = t_bits.sample_bits(bits);
    }
    let bits_squeezes = t_bits.keccak_squeezes() - before;

    // Old path: one sample_u64 (= one Keccak squeeze) per query.
    let mut t_u64 = DefaultTranscript::<Degree3GoldilocksExtensionField>::new(&[0x01, 0x02]);
    let before = t_u64.keccak_squeezes();
    for _ in 0..num_queries {
        let _ = t_u64.sample_u64(1u64 << bits);
    }
    let u64_squeezes = t_u64.keccak_squeezes() - before;

    // 80 queries * 3 bytes = 240 bytes -> ceil(240 / 32) = 8 squeezes.
    let expected = (num_queries * bits.div_ceil(8)).div_ceil(32);
    assert_eq!(bits_squeezes, expected, "reservoir squeeze count");
    assert_eq!(
        u64_squeezes, num_queries,
        "sample_u64 does one squeeze per query"
    );
    assert!(
        bits_squeezes * 8 <= u64_squeezes,
        "sample_bits should cut query-phase Keccak permutations ~10x: {bits_squeezes} vs {u64_squeezes}"
    );
}
