//! Specialized single-block Keccak256 for short inputs (≤ one rate block).
//!
//! The Merkle backend hashes only short, fixed-shape leaves: two 32-byte child
//! nodes (64 bytes) for internal nodes, and a handful of field elements for
//! leaves. All of these fit in a single Keccak256 rate block (136 bytes), so the
//! full `sha3` streaming sponge — its generic `block_buffer` (partial-block
//! buffering, length tracking) wrapped around the permutation — is overkill.
//!
//! This module computes the **identical Keccak256 digest** with a hand-rolled
//! single-block absorb: lay the input into the 200-byte state, apply Keccak
//! pad10*1 (`0x01` after the message, `0x80` at the last rate byte), call
//! `keccak::f1600` once, and read the first 32 bytes of the squeezed state.
//!
//! `keccak::f1600` is the key: on the guest it resolves (via the recursion
//! crate's `[patch.crates-io] keccak`) to the `KeccakPermute` precompile syscall,
//! so this routine issues exactly one ecall and runs none of the permutation in
//! RISC-V; on the host it is the upstream software permutation. The output is
//! byte-for-byte the same Keccak256 as `sha3::Keccak256`, so this is a
//! transparent implementation swap — no protocol or proof-format change.

/// Keccak256 rate in bytes (1088-bit rate, 512-bit capacity).
const RATE: usize = 136;
/// Keccak256 digest length in bytes.
pub const OUTPUT_LEN: usize = 32;

/// Keccak256 of an input that fits in a single rate block with room for padding
/// (`input.len() < 136`).
///
/// # Panics
/// Debug-asserts `input.len() < RATE`. At exactly `RATE` bytes the pad10*1
/// padding would spill into a second block, which this single-block routine does
/// not handle. Callers in the Merkle backend only ever pass ≤64-byte node pairs
/// or short field-element leaves, well within bounds.
#[inline]
pub fn keccak256_single_block(input: &[u8]) -> [u8; OUTPUT_LEN] {
    debug_assert!(
        input.len() < RATE,
        "keccak256_single_block: input does not leave room for padding in one block"
    );

    // Absorb: XOR the message bytes into the rate region of a zeroed state, byte
    // addressed little-endian within each 64-bit lane (Keccak's convention).
    let mut state = [0u64; 25];
    let mut block = [0u8; RATE];
    block[..input.len()].copy_from_slice(input);
    // pad10*1 (Keccak, not SHA-3): 0x01 immediately after the message, 0x80 at
    // the final rate byte. When the message is exactly RATE-1 long these land on
    // the same byte as 0x81; here inputs are short so they are distinct.
    block[input.len()] ^= 0x01;
    block[RATE - 1] ^= 0x80;
    for (lane, chunk) in state.iter_mut().zip(block.chunks_exact(8)) {
        *lane = u64::from_le_bytes(chunk.try_into().unwrap());
    }

    keccak::f1600(&mut state);

    // Squeeze: the first 32 output bytes are the first four state lanes (LE).
    let mut out = [0u8; OUTPUT_LEN];
    for (chunk, lane) in out.chunks_exact_mut(8).zip(state.iter()) {
        chunk.copy_from_slice(&lane.to_le_bytes());
    }
    out
}

/// Keccak256 over an arbitrary-length byte slice, absorbing block by block and
/// running each permutation via `keccak::f1600` (the `KeccakPermute` precompile
/// on the guest). Byte-identical to `sha3::Keccak256`, but skips `sha3`'s
/// `block_buffer` streaming machinery. Use this when the input may exceed one
/// rate block (e.g. wide trace-leaf serializations); for guaranteed-short inputs
/// (64-byte node pairs) prefer [`keccak256_single_block`].
#[inline]
pub fn keccak256(input: &[u8]) -> [u8; OUTPUT_LEN] {
    let mut state = [0u64; 25];
    let mut chunks = input.chunks_exact(RATE);
    for block in chunks.by_ref() {
        absorb_block(&mut state, block);
        keccak::f1600(&mut state);
    }
    // Final (possibly empty) partial block: pad10*1 then permute.
    let rem = chunks.remainder();
    let mut last = [0u8; RATE];
    last[..rem.len()].copy_from_slice(rem);
    last[rem.len()] ^= 0x01;
    last[RATE - 1] ^= 0x80;
    absorb_block(&mut state, &last);
    keccak::f1600(&mut state);

    let mut out = [0u8; OUTPUT_LEN];
    for (chunk, lane) in out.chunks_exact_mut(8).zip(state.iter()) {
        chunk.copy_from_slice(&lane.to_le_bytes());
    }
    out
}

/// XOR a full `RATE`-byte block into the rate region of `state` (lanes 0..17),
/// little-endian within each lane.
#[inline]
fn absorb_block(state: &mut [u64; 25], block: &[u8]) {
    for (lane, chunk) in state.iter_mut().zip(block.chunks_exact(8)) {
        *lane ^= u64::from_le_bytes(chunk.try_into().unwrap());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha3::{Digest, Keccak256};

    fn reference(input: &[u8]) -> [u8; 32] {
        let mut h = Keccak256::new();
        h.update(input);
        h.finalize().into()
    }

    #[test]
    fn matches_sha3_keccak256_for_node_pairs() {
        // 64-byte internal-node inputs (the dominant Merkle case).
        for seed in 0u8..32 {
            let mut input = [0u8; 64];
            for (i, b) in input.iter_mut().enumerate() {
                *b = seed.wrapping_mul(31).wrapping_add(i as u8);
            }
            assert_eq!(keccak256_single_block(&input), reference(&input));
        }
    }

    #[test]
    fn matches_sha3_keccak256_for_various_lengths() {
        // Empty, short, and up-to-(rate-1) inputs all agree with the streaming
        // sponge. `RATE` itself (136) needs a second block for padding and is out
        // of scope for the single-block routine.
        for len in [0usize, 1, 8, 31, 32, 33, 64, 72, 135] {
            let input: alloc::vec::Vec<u8> = (0..len).map(|i| (i as u8).wrapping_mul(7)).collect();
            assert_eq!(
                keccak256_single_block(&input),
                reference(&input),
                "mismatch at len {len}"
            );
        }
    }

    #[test]
    fn multiblock_matches_sha3_keccak256() {
        // Cover one-block, exact-block-boundary, and many-block inputs — the wide
        // trace-leaf serializations the Merkle backend hashes (e.g. 1480 columns).
        for len in [
            0usize,
            1,
            64,
            135,
            136,
            137,
            200,
            271,
            272,
            273,
            600,
            1480 * 8,
            12000,
        ] {
            let input: alloc::vec::Vec<u8> = (0..len)
                .map(|i| (i as u8).wrapping_mul(13).wrapping_add(1))
                .collect();
            assert_eq!(
                keccak256(&input),
                reference(&input),
                "mismatch at len {len}"
            );
        }
    }
}
