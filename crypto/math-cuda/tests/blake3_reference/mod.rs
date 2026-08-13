//! The host BLAKE3 compression reference the device kernels are checked against.
//!
//! ⚠ **This is a duplicate.** The reference lives at
//! `prover/src/lfm/blake3.rs:125` (`blake3_compress_rounds`), which `math-cuda`
//! must not depend on — `prover` depends on this crate, not the other way round.
//! P-a Stage 1 sinks the real one into `crypto/crypto` (`hash/blake3/`);
//! **TODO: when it lands, delete the body below and re-export
//! `crypto::hash::blake3::blake3_compress_rounds` here instead**, so the device,
//! the host backend and the in-circuit chip are all checked against one function
//! rather than three copies of it.
//!
//! Until then the copy is checked from outside: at 7 rounds
//! `blake3_compress_parity.rs` anchors it against the `blake3` crate, and the
//! only difference between the two round counts is the loop bound.
//!
//! Lives in a subdirectory of `tests/`, so cargo treats it as a shared module
//! the parity tests `mod blake3_reference;` rather than as a test binary of its
//! own. `#![allow(dead_code)]` because not every includer uses every item.

#![allow(dead_code)]

/// The BLAKE3 IV (= SHA-256's initial state). Mirror of `blake3.rs:46`.
pub const BLAKE3_IV: [u32; 8] = [
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
];

/// The message-schedule permutation. Mirror of `blake3.rs:52`.
pub const BLAKE3_MSG_PERMUTATION: [usize; 16] =
    [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8];

/// Rounds of standard BLAKE3 — the arm the `blake3` crate anchors.
pub const BLAKE3_STANDARD_ROUNDS: usize = 7;

/// Rounds of the internal variant P-a ships (PA-PLAN §1.5).
pub const BLAKE3_SIX_ROUNDS: usize = 6;

/// `CHUNK_START | CHUNK_END | ROOT` — the flags of a hash whose whole message is
/// one block of one chunk, and the framing a Merkle parent uses.
pub const FLAGS_ONE_BLOCK: u32 = 0x0B;

fn blake3_g(v: &mut [u32; 16], a: usize, b: usize, c: usize, d: usize, mx: u32, my: u32) {
    v[a] = v[a].wrapping_add(v[b]).wrapping_add(mx);
    v[d] = (v[d] ^ v[a]).rotate_right(16);
    v[c] = v[c].wrapping_add(v[d]);
    v[b] = (v[b] ^ v[c]).rotate_right(12);
    v[a] = v[a].wrapping_add(v[b]).wrapping_add(my);
    v[d] = (v[d] ^ v[a]).rotate_right(8);
    v[c] = v[c].wrapping_add(v[d]);
    v[b] = (v[b] ^ v[c]).rotate_right(7);
}

/// The BLAKE3 compression function at an explicit round count, full 16-word
/// output. Transcription of `prover/src/lfm/blake3.rs:125`.
pub fn blake3_compress_rounds(
    h: &[u32; 8],
    m: &[u32; 16],
    t: u64,
    block_len: u32,
    flags: u32,
    rounds: usize,
) -> [u32; 16] {
    let mut v: [u32; 16] = [
        h[0],
        h[1],
        h[2],
        h[3],
        h[4],
        h[5],
        h[6],
        h[7],
        BLAKE3_IV[0],
        BLAKE3_IV[1],
        BLAKE3_IV[2],
        BLAKE3_IV[3],
        t as u32,
        (t >> 32) as u32,
        block_len,
        flags,
    ];

    let mut m = *m;
    for r in 0..rounds {
        blake3_g(&mut v, 0, 4, 8, 12, m[0], m[1]);
        blake3_g(&mut v, 1, 5, 9, 13, m[2], m[3]);
        blake3_g(&mut v, 2, 6, 10, 14, m[4], m[5]);
        blake3_g(&mut v, 3, 7, 11, 15, m[6], m[7]);
        blake3_g(&mut v, 0, 5, 10, 15, m[8], m[9]);
        blake3_g(&mut v, 1, 6, 11, 12, m[10], m[11]);
        blake3_g(&mut v, 2, 7, 8, 13, m[12], m[13]);
        blake3_g(&mut v, 3, 4, 9, 14, m[14], m[15]);
        if r < rounds - 1 {
            let prev = m;
            for (i, &p) in BLAKE3_MSG_PERMUTATION.iter().enumerate() {
                m[i] = prev[p];
            }
        }
    }

    let mut out = [0u32; 16];
    for i in 0..8 {
        out[i] = v[i] ^ v[i + 8];
        out[i + 8] = v[i + 8] ^ h[i];
    }
    out
}

/// A Merkle parent: one compression over the 64 bytes of two child digests, with
/// the digest read back out little-endian. The host `hash_new_parent` for a
/// BLAKE3 backend, and the reference for `blake3_merkle_level`.
pub fn merkle_parent(left: &[u8; 32], right: &[u8; 32], rounds: usize) -> [u8; 32] {
    let mut m = [0u32; 16];
    for i in 0..8 {
        m[i] = u32::from_le_bytes(left[4 * i..4 * i + 4].try_into().unwrap());
        m[i + 8] = u32::from_le_bytes(right[4 * i..4 * i + 4].try_into().unwrap());
    }
    let out = blake3_compress_rounds(&BLAKE3_IV, &m, 0, 64, FLAGS_ONE_BLOCK, rounds);
    let mut digest = [0u8; 32];
    for i in 0..8 {
        digest[4 * i..4 * i + 4].copy_from_slice(&out[i].to_le_bytes());
    }
    digest
}

/// The round count the kernels are compiled for, as the Rust side of the
/// feature. Mirrors `math-cuda`'s `blake3-6round`, which build.rs turns into
/// `-DBLAKE3_ROUNDS=6`.
pub const fn expected_device_rounds() -> usize {
    if cfg!(feature = "blake3-6round") {
        BLAKE3_SIX_ROUNDS
    } else {
        BLAKE3_STANDARD_ROUNDS
    }
}

/// This module is a transcription, so it gets its own check that it did not
/// drift — otherwise a device-vs-host parity failure would be ambiguous between
/// "the kernel is wrong" and "the copy is wrong".
///
/// Host-only: no GPU, so these run wherever the suite compiles, including the
/// laptops where the kernels are stubbed out.
#[cfg(test)]
mod tests {
    use super::*;

    /// At 7 rounds the reference must be the `blake3` crate, over every length a
    /// single block can hold — `block_len` and the zero-padding both key off the
    /// length, and one length would not see a port that ignored either.
    #[test]
    fn the_reference_copy_is_the_blake3_crate_at_seven_rounds() {
        for len in 0..=64usize {
            let msg: Vec<u8> = (0..len)
                .map(|i| (i as u8).wrapping_mul(37).wrapping_add(11))
                .collect();
            let mut block = [0u8; 64];
            block[..len].copy_from_slice(&msg);
            let words: [u32; 16] = core::array::from_fn(|i| {
                u32::from_le_bytes(block[4 * i..4 * i + 4].try_into().unwrap())
            });
            let out = blake3_compress_rounds(
                &BLAKE3_IV,
                &words,
                0,
                len as u32,
                FLAGS_ONE_BLOCK,
                BLAKE3_STANDARD_ROUNDS,
            );
            let mut ours = [0u8; 32];
            for i in 0..8 {
                ours[4 * i..4 * i + 4].copy_from_slice(&out[i].to_le_bytes());
            }
            assert_eq!(
                ours,
                *blake3::hash(&msg).as_bytes(),
                "reference copy diverged from the blake3 crate at length {len}"
            );
        }
    }

    /// And the parent framing on top of it: `hash_new_parent(a, b)` is
    /// `blake3::hash(a ‖ b)` at 7 rounds. Pins `block_len = 64`, `t = 0`,
    /// `h = IV`, the flag set, and the little-endian digest read-back in one shot.
    #[test]
    fn the_reference_parent_is_the_blake3_crate_at_seven_rounds() {
        let left: [u8; 32] = core::array::from_fn(|i| (i as u8).wrapping_mul(7));
        let right: [u8; 32] = core::array::from_fn(|i| (i as u8).wrapping_mul(31).wrapping_add(3));
        let mut msg = Vec::with_capacity(64);
        msg.extend_from_slice(&left);
        msg.extend_from_slice(&right);
        assert_eq!(
            merkle_parent(&left, &right, BLAKE3_STANDARD_ROUNDS),
            *blake3::hash(&msg).as_bytes()
        );
    }

    /// NEGATIVE CONTROL: at 6 rounds neither must match, or the two checks above
    /// would pass with `rounds` ignored.
    #[test]
    fn six_rounds_is_not_the_blake3_crate() {
        let msg: [u8; 36] = core::array::from_fn(|i| i as u8);
        let mut block = [0u8; 64];
        block[..36].copy_from_slice(&msg);
        let words: [u32; 16] = core::array::from_fn(|i| {
            u32::from_le_bytes(block[4 * i..4 * i + 4].try_into().unwrap())
        });
        let out = blake3_compress_rounds(
            &BLAKE3_IV,
            &words,
            0,
            36,
            FLAGS_ONE_BLOCK,
            BLAKE3_SIX_ROUNDS,
        );
        let mut ours = [0u8; 32];
        for i in 0..8 {
            ours[4 * i..4 * i + 4].copy_from_slice(&out[i].to_le_bytes());
        }
        assert_ne!(ours, *blake3::hash(&msg).as_bytes());

        let left = [1u8; 32];
        let right = [2u8; 32];
        let mut pmsg = Vec::with_capacity(64);
        pmsg.extend_from_slice(&left);
        pmsg.extend_from_slice(&right);
        assert_ne!(
            merkle_parent(&left, &right, BLAKE3_SIX_ROUNDS),
            *blake3::hash(&pmsg).as_bytes()
        );
    }
}
