//! ★ Known-answer tests for the DEVICE `Blake3Chain`, against references that
//! are not this tree's Rust.
//!
//! # Why this file exists (risk R13)
//!
//! Every other parity test here asserts device == host. That is necessary and
//! not sufficient: the device kernels were transcribed from the host reference,
//! so a shared misreading of the construction passes all of them. R13 is exactly
//! that gap — "track G would then be checking a device port against the same
//! code path it was derived from".
//!
//! Two references close it, and neither is Rust in this repository:
//!
//! - **At 7 rounds, the official `blake3` crate.** `Blake3Chain` over any
//!   message of at most one chunk (1024 bytes) IS `blake3::hash`, because
//!   standard BLAKE3's first chunk is this chain and a one-chunk message has
//!   that chunk's output as its root (PA-PLAN §1.7.2, P1). So for the whole
//!   range the prover actually hashes in, the device is checked against a
//!   published, externally maintained implementation with nothing in between.
//! - **At 6 rounds, `CHAIN_KAT_6ROUND`.** Those digests came from #903's Python
//!   oracle, not from this code (`chain.rs:280-294`). Asserting the device
//!   against them is a check against an artifact this tree did not compute.
//!
//! # Why the coverage is at multiples of four bytes
//!
//! The device chain is word-granular, because that is all it ever hashes: every
//! production message is a whole number of 8-byte field elements. The KAT
//! lengths that are not multiples of 4 (1, 31, 63, 127) are unreachable from
//! device code by construction and are covered by the host tests in
//! `crypto/crypto/src/hash/blake3/chain.rs` instead. What remains — 0, 64, 128,
//! 192, 256, 1024, 1088 — still covers every structural case PA-PLAN §1.7.4
//! names except the partial-tail ones: the empty message is one block (0), a
//! 64-byte message is the parent form (64), an exact multiple of 64 emits no
//! spurious final block (128), interior blocks carry no flags (192, 256, 1024),
//! and 1088 is where this construction leaves standard BLAKE3.
//!
//! Needs a GPU.

mod blake3_reference;

use blake3_reference::{expected_device_rounds, merkle_parent};
use crypto::hash::blake3::BLAKE3_ROUNDS;
use crypto::hash::blake3::chain::{
    CHAIN_KAT_6ROUND, CHAIN_KAT_LENS, blake3_chain_rounds, kat_message_byte,
};
use math_cuda::blake3::chain_probe;

/// The KAT message of a given length: byte `i` is `37i + 11 (mod 256)`.
fn message(len: usize) -> Vec<u8> {
    (0..len).map(kat_message_byte).collect()
}

/// A byte message as the little-endian u32 words the device chain absorbs.
/// Panics on a length that is not a whole number of words — see the module docs
/// for why that case cannot arise on device.
fn words(msg: &[u8]) -> Vec<u32> {
    assert!(msg.len().is_multiple_of(4), "device chain is word-granular");
    msg.chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn assert_lockstep() {
    assert_eq!(
        BLAKE3_ROUNDS,
        expected_device_rounds(),
        "crypto's blake3-6round and math-cuda's are out of lockstep"
    );
}

/// The KAT lengths the device can hash, paired with their index into
/// `CHAIN_KAT_6ROUND`.
fn device_reachable_lengths() -> Vec<(usize, usize)> {
    CHAIN_KAT_LENS
        .iter()
        .enumerate()
        .filter(|&(_, &len)| len.is_multiple_of(4))
        .map(|(i, &len)| (i, len))
        .collect()
}

/// ★ THE EXTERNAL ANCHOR. At 7 rounds the device chain must be the `blake3`
/// crate's hash, for every reachable length up to one full chunk.
///
/// This is the strongest statement available about the device port: no oracle,
/// no table, no transcription — a published implementation computes the same
/// bytes. It pins the block splitting, the zero padding, the final block's
/// `block_len`, the CHUNK_START/CHUNK_END/ROOT schedule, `t = 0` throughout, and
/// the little-endian digest read-back, all at once.
///
/// Only meaningful when the cubin is built for 7 rounds; under `blake3-6round`
/// nothing external recomputes this, which is PA-PLAN §1.6's premise and why the
/// 6-round arm needs the committed table instead.
#[test]
fn device_chain_is_the_blake3_crate_at_seven_rounds() {
    assert_lockstep();
    if expected_device_rounds() != 7 {
        return;
    }
    for (_, len) in device_reachable_lengths() {
        if len > 1024 {
            continue;
        }
        let msg = message(len);
        let device = chain_probe(&words(&msg)).unwrap();
        assert_eq!(
            device,
            *blake3::hash(&msg).as_bytes(),
            "device chain must equal the blake3 crate at length {len}"
        );
    }
}

/// ★ P3, on device. Past one chunk the construction deliberately leaves standard
/// BLAKE3 — the standard would start chunk 1 with `t = 1` and a reset chaining
/// value, this keeps chaining. Without this the test above would pass
/// identically if the kernels had implemented the whole chunk tree, so "the
/// device implements the single-chunk chain" would be unfalsifiable.
#[test]
fn device_chain_leaves_the_blake3_crate_past_one_chunk() {
    assert_lockstep();
    if expected_device_rounds() != 7 {
        return;
    }
    // 1024 is the last length where they agree; 1088 the first reachable one
    // past it. Asserting both locates the divergence rather than just observing
    // one.
    let agreeing = message(1024);
    assert_eq!(
        chain_probe(&words(&agreeing)).unwrap(),
        *blake3::hash(&agreeing).as_bytes(),
        "1024 bytes is still one chunk and must agree"
    );
    let diverging = message(1088);
    assert_ne!(
        chain_probe(&words(&diverging)).unwrap(),
        *blake3::hash(&diverging).as_bytes(),
        "past one chunk the device must leave the standard"
    );
}

/// ★ THE 6-ROUND ANCHOR. The device must reproduce the committed KAT table,
/// whose digests came from a Python oracle rather than from this code.
///
/// This is the assertion R13 asks for: at the round count the campaign actually
/// ships, the device port is pinned by numbers no Rust in this tree produced.
#[test]
fn device_chain_matches_the_committed_table_at_six_rounds() {
    assert_lockstep();
    if expected_device_rounds() != 6 {
        return;
    }
    for (i, len) in device_reachable_lengths() {
        let device = chain_probe(&words(&message(len))).unwrap();
        assert_eq!(
            device, CHAIN_KAT_6ROUND[i],
            "device chain must match the committed 6-round KAT at length {len}"
        );
    }
}

/// The device chain against the host chain at whatever round count this build
/// uses. Weaker than the two anchors above — both sides are ours — but it is the
/// one that runs in every configuration, and it is the property the commitment
/// path actually needs: a GPU tree and a CPU tree over the same leaves must be
/// the same tree.
#[test]
fn device_chain_matches_the_host_chain() {
    assert_lockstep();
    let rounds = expected_device_rounds();
    for (_, len) in device_reachable_lengths() {
        let msg = message(len);
        assert_eq!(
            chain_probe(&words(&msg)).unwrap(),
            blake3_chain_rounds(&msg, rounds),
            "device/host chain mismatch at length {len}, {rounds} rounds"
        );
    }
    // Lengths off the KAT list, stepping through several block boundaries, so
    // the agreement is not an artifact of the seven lengths chosen above.
    for len in (0..=520usize).step_by(4) {
        let msg = message(len);
        assert_eq!(
            chain_probe(&words(&msg)).unwrap(),
            blake3_chain_rounds(&msg, rounds),
            "device/host chain mismatch at length {len}"
        );
    }
}

/// ★ P2, on device: a 64-byte message through the chain is exactly the Merkle
/// parent compression.
///
/// This is the invariant that lets the leaf and parent layers be one hash — and
/// the reason `blake3_hash_merkle_parent` can be a single compression with no
/// chaining at all. If the chain's flag schedule or `block_len` moved, the two
/// would part here while every leaf test still passed.
#[test]
fn a_sixty_four_byte_chain_is_the_parent_compression() {
    assert_lockstep();
    let left: [u8; 32] = core::array::from_fn(|i| (i as u8).wrapping_mul(7));
    let right: [u8; 32] = core::array::from_fn(|i| (i as u8).wrapping_mul(31).wrapping_add(3));
    let mut msg = [0u8; 64];
    msg[..32].copy_from_slice(&left);
    msg[32..].copy_from_slice(&right);

    assert_eq!(
        chain_probe(&words(&msg)).unwrap(),
        merkle_parent(&left, &right, expected_device_rounds()),
        "a 64-byte device chain must be the parent form"
    );
}

/// NEGATIVE CONTROL: distinct lengths must give distinct digests, or the tests
/// above would pass with a probe that ignored its input length. In particular a
/// chain that ignored `block_len` would collide 0 with nothing visible here, but
/// one that dropped the final partial block would collide 64 with 128.
#[test]
fn device_digests_are_distinct_across_lengths() {
    assert_lockstep();
    let mut seen: Vec<[u8; 32]> = Vec::new();
    for len in (0..=256usize).step_by(4) {
        let d = chain_probe(&words(&message(len))).unwrap();
        assert!(
            !seen.contains(&d),
            "length {len} collides with a shorter message"
        );
        seen.push(d);
    }
}

/// The cubin's compiled-in round count must be the one the Rust side thinks it
/// is. Reading it back is the only way to observe from host code which arm
/// `blake3_merkle_level` and the leaf kernels were built for; a mismatch here is
/// a GPU tree committing under a different hash with no other symptom.
#[test]
fn the_cubin_round_count_is_what_the_feature_selected() {
    assert_eq!(
        math_cuda::blake3::device_rounds().unwrap() as usize,
        expected_device_rounds(),
        "cubin round count disagrees with math-cuda's blake3-6round feature"
    );
}
