//! Parity: the device field-element serialization and 64-byte block framing must
//! reproduce the bytes the CPU commit path hashes.
//!
//! The leaf byte encoding does not move under P-a: `leaves_bit_reversed_grouped`
//! (`crypto/stark/src/commitment.rs:55`) writes each element in canonical
//! big-endian form and concatenates, and `hash_bytes` hashes that buffer. BLAKE3
//! reads a 64-byte block as 16 little-endian u32 words, so the device has to
//! transpose: one element becomes the byte-reverse of its canonical high half,
//! then of its low half. This pins that transposition, the canonicalisation in
//! front of it, and the block boundaries and zero-padding around it — everything
//! a leaf kernel needs that does not depend on the still-open chaining
//! construction (PA-PLAN §1.6).

use math::field::element::FieldElement;
use math::field::goldilocks::GoldilocksField;
use math::traits::AsBytes;
use math_cuda::blake3::{blocks_of_felts, serialize_felts};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

type Fp = FieldElement<GoldilocksField>;

const PRIME: u64 = 0xFFFF_FFFF_0000_0001;

/// Raw values that include the ones canonicalisation is the only thing standing
/// between: `p` and above are representable in the prover's non-canonical u64
/// form and serialize as their reduced value, so a kernel that skipped the
/// reduction would differ from the CPU on exactly these.
fn raws(seed: u64, n: usize) -> Vec<u64> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut v = vec![
        0u64,
        1,
        PRIME - 1,
        PRIME,
        PRIME + 1,
        PRIME + 12345,
        u64::MAX,
    ];
    v.truncate(n.min(7));
    while v.len() < n {
        // Half in range, half deliberately non-canonical.
        let x = rng.r#gen::<u64>();
        v.push(if v.len().is_multiple_of(2) {
            x % PRIME
        } else {
            x
        });
    }
    v
}

/// The bytes the CPU hashes for these elements, via the same `AsBytes` route
/// `leaves_bit_reversed_grouped` serializes through.
fn cpu_bytes(raws: &[u64]) -> Vec<u8> {
    let mut out = Vec::with_capacity(raws.len() * 8);
    for &r in raws {
        out.extend_from_slice(&Fp::from_raw(r).as_bytes());
    }
    out
}

/// Those bytes as BLAKE3 message words, zero-padded to whole 64-byte blocks.
fn cpu_block_words(bytes: &[u8]) -> Vec<u32> {
    let n_blocks = bytes.len().div_ceil(64);
    let mut padded = bytes.to_vec();
    padded.resize(n_blocks * 64, 0);
    padded
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

#[test]
fn device_serialization_is_the_cpu_leaf_bytes() {
    for n in [1usize, 2, 7, 8, 9, 64, 1000] {
        let vals = raws(11 + n as u64, n);
        let device = serialize_felts(&vals).unwrap();
        let expected = cpu_block_words(&cpu_bytes(&vals));
        // `serialize_felts` emits exactly two words per element with no padding,
        // so compare against the unpadded prefix of the block view.
        assert_eq!(device.len(), 2 * n);
        assert_eq!(
            device[..],
            expected[..2 * n],
            "serialization mismatch at n = {n}"
        );
    }
}

/// A field element is 8 bytes = 2 words, and a block is 16 words, so elements
/// straddle a block boundary only when the count is not a multiple of 8 — but
/// ext3 elements are 6 words and straddle routinely, which is why the builder
/// works at word granularity. Both cases are covered by the counts below.
#[test]
fn device_block_framing_matches_the_cpu_byte_stream() {
    for n in [1usize, 3, 8, 9, 16, 17, 63, 64, 255] {
        let vals = raws(500 + n as u64, n);
        let device = blocks_of_felts(&vals).unwrap();
        let expected = cpu_block_words(&cpu_bytes(&vals));
        assert_eq!(
            device.len(),
            expected.len(),
            "block count mismatch at n = {n}"
        );
        assert_eq!(device, expected, "block words mismatch at n = {n}");
    }
}

/// The tail block must be zero-padded, not left holding stale words. The check
/// above would catch that only if the padding happened to differ from whatever
/// was there; asserting the padded region directly is what makes it a test of the
/// padding rather than of the allocator.
#[test]
fn the_tail_block_is_zero_padded() {
    // 9 elements = 18 words = one full block plus 2 words, leaving 14 to pad.
    let vals = raws(77, 9);
    let device = blocks_of_felts(&vals).unwrap();
    assert_eq!(device.len(), 32, "expected exactly two blocks");
    assert!(
        device[18..].iter().all(|&w| w == 0),
        "tail block not zero-padded: {:?}",
        &device[18..]
    );
}
