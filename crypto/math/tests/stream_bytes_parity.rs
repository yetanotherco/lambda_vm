//! `AsBytes::stream_bytes` must emit exactly the bytes `as_bytes` returns.
//!
//! Nothing in the type system enforces it: `stream_bytes` is a defaulted trait
//! method, so an override that disagrees with `as_bytes` compiles cleanly and
//! then silently changes every Merkle leaf hash and Fiat-Shamir challenge that
//! flows through it — the transcript and the Merkle backends stream their input
//! rather than calling `as_bytes`. A divergence would surface as proofs that no
//! longer verify against previously committed roots, not as a test failure, so
//! it is pinned here.
//!
//! `ext3_stream_bytes_matches_gpu_kernel_contract` additionally pins the ext3
//! byte layout that `crypto/math-cuda/src/merkle.rs` mirrors: the GPU
//! `keccak_leaves_ext3` kernel reads three canonical u64s per column in
//! component order 0,1,2 to match `write_bytes_be`. CPU/GPU leaf parity depends
//! on the two staying in agreement, and the GPU parity tests only run on a CUDA
//! host, so this keeps the CPU half honest on a GPU-less runner.

use math::field::element::FieldElement;
use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField;
use math::field::goldilocks::GoldilocksField;
use math::traits::{AsBytes, ByteConversion};

type Fp = FieldElement<GoldilocksField>;
type Fp3 = FieldElement<Degree3GoldilocksExtensionField>;

fn streamed<T: AsBytes>(e: &T) -> Vec<u8> {
    let mut out = Vec::new();
    e.stream_bytes(&mut |b| out.extend_from_slice(b));
    out
}

/// Deterministic LCG: keeps the sweep reproducible and dependency-free.
fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state
}

const GOLDILOCKS_P: u64 = 0xFFFF_FFFF_0000_0001;

/// Values around the modulus matter: both encodings reduce through
/// `canonical_u64`, so a non-canonical `u64` is where a raw-value override would
/// diverge from `as_bytes`.
fn edge_values() -> Vec<u64> {
    vec![
        0,
        1,
        2,
        u32::MAX as u64,
        1u64 << 32,
        GOLDILOCKS_P - 1,
        GOLDILOCKS_P,     // 0 in the field
        GOLDILOCKS_P + 1, // 1 in the field
        u64::MAX - 1,
        u64::MAX,
    ]
}

#[test]
fn goldilocks_stream_bytes_matches_as_bytes_and_to_bytes_be() {
    let mut state = 0x1234_5678_9abc_def0u64;
    let mut values = edge_values();
    values.extend((0..2000).map(|_| lcg(&mut state)));

    for v in values {
        let e = Fp::from(v);
        let s = streamed(&e);
        assert_eq!(s.len(), 8, "goldilocks stream must be 8 bytes (v={v:#x})");
        // The Merkle backends stream instead of calling `as_bytes`.
        assert_eq!(s, e.as_bytes(), "stream != as_bytes (v={v:#x})");
        // `DefaultTranscript::append_field_element` streams instead of
        // appending `to_bytes_be`.
        assert_eq!(
            s,
            ByteConversion::to_bytes_be(&e),
            "stream != to_bytes_be (v={v:#x})"
        );
    }
}

#[test]
fn ext3_stream_bytes_matches_as_bytes_and_to_bytes_be() {
    let mut state = 0x0fed_cba9_8765_4321u64;
    let mut triples: Vec<[u64; 3]> = Vec::new();
    for v in edge_values() {
        triples.push([v, v, v]);
        triples.push([v, 0, 1]);
    }
    triples.extend((0..2000).map(|_| [lcg(&mut state), lcg(&mut state), lcg(&mut state)]));

    for t in triples {
        let e = Fp3::new([Fp::from(t[0]), Fp::from(t[1]), Fp::from(t[2])]);
        let s = streamed(&e);
        assert_eq!(s.len(), 24, "ext3 stream must be 24 bytes (t={t:?})");
        assert_eq!(s, e.as_bytes(), "stream != as_bytes (t={t:?})");
        assert_eq!(
            s,
            ByteConversion::to_bytes_be(&e),
            "stream != to_bytes_be (t={t:?})"
        );
    }
}

#[test]
fn ext3_stream_bytes_matches_gpu_kernel_contract() {
    let mut state = 0xdead_beef_cafe_babeu64;

    for _ in 0..1000 {
        let e = Fp3::new([
            Fp::from(lcg(&mut state)),
            Fp::from(lcg(&mut state)),
            Fp::from(lcg(&mut state)),
        ]);

        // What the CUDA kernel builds: canonical u64 per component, big-endian,
        // component order 0,1,2.
        let mut expected = Vec::new();
        for component in e.value() {
            expected.extend_from_slice(&component.canonical_u64().to_be_bytes());
        }
        assert_eq!(streamed(&e), expected, "ext3 stream != canonical-BE 0,1,2");

        let mut buf = [0u8; 24];
        ByteConversion::write_bytes_be(&e, &mut buf);
        assert_eq!(streamed(&e), buf, "ext3 stream != write_bytes_be");
    }
}

/// Keccak absorption means `update(a); update(b)` == `update(a || b)`, so a
/// digest can only move if the concatenated stream moves. Pins the multi-element
/// hash paths (`hash_data`, `hash_data_from_slices`) against the old
/// `as_bytes`-per-element input.
#[test]
fn concatenated_stream_matches_concatenated_as_bytes() {
    let mut state = 0xa5a5_5a5a_a5a5_5a5au64;
    let elements: Vec<Fp3> = (0..256)
        .map(|_| {
            Fp3::new([
                Fp::from(lcg(&mut state)),
                Fp::from(lcg(&mut state)),
                Fp::from(lcg(&mut state)),
            ])
        })
        .collect();

    let mut via_as_bytes = Vec::new();
    let mut via_stream = Vec::new();
    for e in &elements {
        via_as_bytes.extend_from_slice(&e.as_bytes());
        e.stream_bytes(&mut |b| via_stream.extend_from_slice(b));
    }

    assert_eq!(
        via_as_bytes, via_stream,
        "concatenated hasher input stream changed"
    );
}

/// The default `stream_bytes` body forwards to `as_bytes`; a type that does not
/// override it must still round-trip identically.
#[test]
fn default_stream_bytes_impl_matches_as_bytes() {
    struct Unoverridden(Vec<u8>);
    impl AsBytes for Unoverridden {
        fn as_bytes(&self) -> Vec<u8> {
            self.0.clone()
        }
    }

    for bytes in [vec![], vec![0u8], vec![1, 2, 3, 4, 5], vec![0xff; 64]] {
        let v = Unoverridden(bytes.clone());
        assert_eq!(streamed(&v), bytes);
    }
}
