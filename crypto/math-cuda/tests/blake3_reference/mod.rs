//! The host BLAKE3 compression reference the device kernels are checked against.
//!
//! **Not a copy any more.** The compression function, the IV, the permutation
//! and the round-count constants are re-exported from `crypto::hash::blake3`,
//! which P-a Stage 1 made their single home — so the device kernels, the host
//! commitment backends and the in-circuit chip are now all checked against one
//! function rather than three transcriptions of it. `crypto` is a dev-dependency
//! of this crate, which is what makes the re-export legal: `prover` (the old
//! home) depends on this crate, so it could never have been imported here.
//!
//! What stays local is [`merkle_parent`] — the *framing* a device Merkle parent
//! uses, which is a property of the kernel, not of the primitive. It is checked
//! two ways: against the `blake3` crate at 7 rounds, and against the production
//! host backend at the build's round count, so the reference cannot drift from
//! either the standard or the thing the CPU prover actually commits with.
//!
//! Lives in a subdirectory of `tests/`, so cargo treats it as a shared module
//! the parity tests `mod blake3_reference;` rather than as a test binary of its
//! own. `#![allow(dead_code)]` because not every includer uses every item.

#![allow(dead_code)]

pub use crypto::hash::blake3::chain::FLAGS_ONE_BLOCK;
pub use crypto::hash::blake3::{
    BLAKE3_IV, BLAKE3_SIX_ROUNDS, BLAKE3_STANDARD_ROUNDS, blake3_compress_rounds,
};

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

/// The compression function is shared now, but [`merkle_parent`]'s framing is
/// still written here, so it gets its own checks that it did not drift —
/// otherwise a device-vs-host parity failure would be ambiguous between "the
/// kernel is wrong" and "the reference is wrong".
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

    /// ★ The reference parent is what the **host commitment backend** computes.
    ///
    /// The two checks above anchor the framing against the standard; this one
    /// anchors it against the thing the CPU prover actually commits with, so a
    /// GPU tree and a CPU tree over the same leaves are the same tree. Without
    /// it, the device could be faithful to `blake3::hash(a ‖ b)` and still
    /// disagree with the backend the proof is verified against.
    ///
    /// It runs at [`expected_device_rounds`], and so it doubles as the LOCKSTEP
    /// alarm for the two crates' `blake3-6round` features: they are separate
    /// features and nothing forces them equal, and a mismatch means a GPU tree
    /// committing under a different hash than the CPU one. If this fails with
    /// the round counts differing, set both features or neither — `make lint`
    /// has a combined pass that compiles them together for the same reason.
    #[test]
    fn the_reference_parent_is_the_host_commitment_backend() {
        use crypto::hash::blake3::BLAKE3_ROUNDS;
        use crypto::merkle_tree::backends::types::BatchBlake3Backend;
        use crypto::merkle_tree::traits::IsMerkleTreeBackend;
        use math::field::goldilocks::GoldilocksField;

        assert_eq!(
            BLAKE3_ROUNDS,
            expected_device_rounds(),
            "crypto's blake3-6round and math-cuda's are out of lockstep: the GPU \
             kernels would commit under a different hash than the CPU backend"
        );

        let left: [u8; 32] = core::array::from_fn(|i| (i as u8).wrapping_mul(11).wrapping_add(5));
        let right: [u8; 32] = core::array::from_fn(|i| (i as u8).wrapping_mul(23));
        assert_eq!(
            merkle_parent(&left, &right, expected_device_rounds()),
            <BatchBlake3Backend<GoldilocksField> as IsMerkleTreeBackend>::hash_new_parent(
                &left, &right
            ),
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
