//! ★ The two RPX implementations this workspace now carries, asserted equal.
//!
//! The merge that built this branch brought RPX in twice. The per-table branch
//! has it at `prover::lfm::algebraic_commit` (generic over `HasherKind`, shared
//! with the LFM's `LFM_HASH` socket and the algebraic Merkle backends); the
//! WHIR branch ported it to `crypto::hash::rpx` (monomorphic, so `crypto` can
//! offer an algebraic sponge without depending on `prover`). Both are real and
//! both stay: neither crate can use the other's.
//!
//! ⚠ **What that costs, and what this file buys back.** Two implementations of
//! one primitive, in one tree, both answering to the name `rpx256`, is a
//! configuration where "which RPX produced this root" has no answer in the
//! type system. Reading the two and observing that they are line-for-line the
//! same function is not a test — it is a claim about a moment, and the next
//! edit to either side is free to break it silently, because nothing links
//! them. Every consumer would keep compiling and every proof would keep
//! verifying; only a root built by one and checked by the other would differ,
//! and nothing in this workspace does that in a test.
//!
//! So the equality is asserted, at the lengths where the two could plausibly
//! disagree — the partial trailing group, the exact block boundary, the empty
//! input, the 40-byte grinding preimage — and anchored, where the shared vector
//! table has a row, to the digests the CUDA kernel is itself pinned to. That
//! last part matters most: `crypto/math-cuda/tests/host_kat/rpx_kat_vectors.h`
//! is one file consumed by the device KAT, so a host that reproduces it is
//! byte-compatible with the kernel as well as with its twin.
//!
//! The type names were made distinct in the same change (`RpxTranscriptHash`
//! became `AlgebraicRpxTranscriptHash` on the LFM side). The names stop a
//! reader confusing them; this file is what stops them drifting.

use crate::lfm::algebraic_commit::{digest_to_commitment, sponge_leaf_bytes};
use crate::lfm::hash::HasherKind;
use crypto::hash::rpx;

/// Deterministic, non-degenerate bytes: no all-zero run that a broken padding
/// rule could accidentally agree on.
fn sample(len: usize) -> Vec<u8> {
    (0..len)
        .map(|i| (i as u8).wrapping_mul(31).wrapping_add(7))
        .collect()
}

/// The LFM side's byte digest, as `AlgebraicDigest<RpxCommit>` finalizes it.
fn lfm_digest(bytes: &[u8]) -> [u8; 32] {
    digest_to_commitment(&sponge_leaf_bytes(HasherKind::Rpx, bytes))
}

/// The WHIR side's byte digest, as `Rpx256Digest` finalizes it.
fn whir_digest(bytes: &[u8]) -> [u8; 32] {
    rpx::digest_to_commitment(&rpx::sponge_leaf_bytes(bytes))
}

/// The lengths that matter, and why each is here:
///
/// - `0` — the empty input returns the capacity without permuting at all, the
///   one path that skips the absorb loop entirely.
/// - `1`, `7` — a partial trailing felt. The low-side zero extension is the
///   detail `sponge_leaf_bytes`' own doc calls easy to get backwards.
/// - `8` — exactly one felt, no padding.
/// - `9` — one felt plus one byte, the first input that spans a felt boundary.
/// - `16`, `17` — inside the first rate block and just past a felt boundary.
/// - `40` — five felts: the grinding preimage, `state ‖ nonce`, the one length
///   whose digest a device kernel also computes.
/// - `64` — exactly one rate block, so the padding flag `len mod 8` is zero and
///   no trailing block is spent.
/// - `65`, `200` — past one block, which is where a two-block absorb that
///   reset the capacity instead of overwriting the rate would diverge.
const LENGTHS: [usize; 11] = [0, 1, 7, 8, 9, 16, 17, 40, 64, 65, 200];

#[test]
fn the_two_rpx_implementations_agree_on_the_same_bytes() {
    for len in LENGTHS {
        let bytes = sample(len);
        assert_eq!(
            lfm_digest(&bytes),
            whir_digest(&bytes),
            "the LFM and WHIR RPX implementations disagree at {len} bytes — one of them has \
             drifted, and every root either produced is now ambiguous"
        );
    }
}

/// ★ FALSIFICATION. The test above compares two functions; if both were the
/// same stub it would pass. This pins them to a value neither produced.
///
/// The vectors are the 8- and 16-felt rows of
/// `crypto/math-cuda/tests/host_kat/rpx_kat_vectors.h`, the table the CUDA
/// kernel's host KAT replays. Canonical felts serialized big-endian round-trip
/// through `felts_from_bytes`, so the byte path must reach the felt path's
/// answer — which makes this an anchor for the byte APIs above, not just for
/// the sponge underneath them.
#[test]
fn both_implementations_reproduce_the_shared_kernel_vectors() {
    // Table 3, len = 8 felts.
    let felts_8: [u64; 8] = [
        3521541860211663897,
        5585621328801039182,
        3314063895810834828,
        6286715337571703139,
        9272399501810688383,
        17378448552699642502,
        9663403628134293866,
        8225575178453385283,
    ];
    let digest_8: [u64; 4] = [
        14052993739410942603,
        8384701950754250190,
        11473922331550289114,
        16644313465254305812,
    ];

    // Table 3, len = 16 felts — two full rate blocks.
    let felts_16: [u64; 16] = [
        9660685076555889599,
        4027567791223379602,
        11432600011703367870,
        6441517771629429252,
        8272264386868866348,
        16565648022353132158,
        16844837242675693755,
        12942506659476152817,
        11839051358503478840,
        1846358602548732379,
        118703897581348635,
        14480592082795401517,
        12015885875590073011,
        7433808365622677077,
        13247077855319202624,
        17837888200692576115,
    ];
    let digest_16: [u64; 4] = [
        18135965004560326100,
        1948492279228612931,
        17772968542724134453,
        12116464713281646840,
    ];

    for (felts, expected) in [(&felts_8[..], digest_8), (&felts_16[..], digest_16)] {
        let bytes: Vec<u8> = felts.iter().flat_map(|f| f.to_be_bytes()).collect();
        let want: [u8; 32] = {
            let mut out = [0u8; 32];
            for (i, w) in expected.iter().enumerate() {
                out[i * 8..i * 8 + 8].copy_from_slice(&w.to_be_bytes());
            }
            out
        };
        assert_eq!(
            lfm_digest(&bytes),
            want,
            "the LFM RPX no longer reproduces the vector the CUDA kernel is pinned to \
             ({} felts)",
            felts.len()
        );
        assert_eq!(
            whir_digest(&bytes),
            want,
            "the WHIR RPX no longer reproduces the vector the CUDA kernel is pinned to \
             ({} felts)",
            felts.len()
        );
    }
}

/// The sampler itself has to be able to tell two inputs apart, or the agreement
/// test above compares one digest with itself eleven times.
#[test]
fn the_sample_inputs_are_distinct() {
    let digests: Vec<[u8; 32]> = LENGTHS.iter().map(|l| lfm_digest(&sample(*l))).collect();
    for (i, a) in digests.iter().enumerate() {
        for (j, b) in digests.iter().enumerate().skip(i + 1) {
            assert_ne!(
                a, b,
                "lengths {} and {} hash the same",
                LENGTHS[i], LENGTHS[j]
            );
        }
    }
}
