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

/// Keccak256 of exactly 64 bytes (two 32-byte Merkle node hashes concatenated).
///
/// Builds the keccak state directly as u64 lanes — no intermediate byte buffer.
/// pad10*1: byte 64 → lane 8 byte 0 (XOR 0x01); byte 135 → lane 16 byte 7 (XOR 0x80).
///
/// This is the hot path for the binary (ARITY=2) FRI-layer Merkle trees, where
/// every internal node hashes exactly two 32-byte children.
#[inline]
pub fn keccak256_two_nodes(left: &[u8; 32], right: &[u8; 32]) -> [u8; OUTPUT_LEN] {
    let mut state = [0u64; 25];
    // Load left child (bytes 0..32 = lanes 0..4).
    for (i, chunk) in left.chunks_exact(8).enumerate() {
        state[i] = u64::from_le_bytes(chunk.try_into().unwrap());
    }
    // Load right child (bytes 32..64 = lanes 4..8).
    for (i, chunk) in right.chunks_exact(8).enumerate() {
        state[4 + i] = u64::from_le_bytes(chunk.try_into().unwrap());
    }
    // pad10*1 for a 64-byte message in a 136-byte rate block:
    //   byte 64  = lane 8,  byte offset 0 → XOR 0x01
    //   byte 135 = lane 16, byte offset 7 → XOR 0x80
    state[8] ^= 0x01;
    state[16] ^= 0x80u64 << 56;
    keccak::f1600(&mut state);
    let mut out = [0u8; OUTPUT_LEN];
    for (chunk, lane) in out.chunks_exact_mut(8).zip(state.iter()) {
        chunk.copy_from_slice(&lane.to_le_bytes());
    }
    out
}

/// Keccak256 of exactly 128 bytes (four 32-byte Merkle node hashes concatenated).
///
/// Builds the keccak state directly as u64 lanes — no intermediate byte buffer.
/// pad10*1: byte 128 → lane 16 byte 0 (XOR 0x01); byte 135 → lane 16 byte 7 (XOR 0x80).
/// Combined: state[16] ^= 0x8000_0000_0000_0001 (little-endian: 0x01 at byte 0, 0x80 at byte 7).
///
/// This is the hot path for the quaternary (ARITY=4) trace/composition Merkle
/// trees, where every internal node hashes exactly four 32-byte children.
#[inline]
pub fn keccak256_four_nodes(children: &[[u8; 32]; 4]) -> [u8; OUTPUT_LEN] {
    let mut state = [0u64; 25];
    // Load all four children (128 bytes = 16 lanes).
    for (child_idx, child) in children.iter().enumerate() {
        for (byte_idx, chunk) in child.chunks_exact(8).enumerate() {
            state[child_idx * 4 + byte_idx] = u64::from_le_bytes(chunk.try_into().unwrap());
        }
    }
    // pad10*1 for a 128-byte message in a 136-byte rate block:
    //   byte 128 = lane 16, byte offset 0 → XOR 0x01
    //   byte 135 = lane 16, byte offset 7 → XOR 0x80
    // Combined (LE ordering): 0x80 at the high byte (offset 7) and 0x01 at low (offset 0).
    state[16] ^= 0x8000_0000_0000_0001u64;
    keccak::f1600(&mut state);
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

/// Keccak256 of a short sequence of field elements without any intermediate byte
/// buffer. Each element is serialized as 8-byte-aligned big-endian chunks (via
/// `to_bytes_be()`), which are XORed directly into successive keccak state lanes
/// (little-endian within each lane). Padding is applied in-place to the state
/// without a `[u8; RATE]` intermediate.
///
/// **Contract**: The total serialized length of all elements must be `< RATE`
/// (136 bytes) — i.e. at most 17 Goldilocks elements or 5 Fp3 elements — and
/// each element's `BYTE_LEN` must be a multiple of 8 (lane-aligned). Debug
/// asserts enforce both. For wider leaves use the `leaf_scratch`-based path.
///
/// Identical output to `keccak256_single_block(concatenation of to_bytes_be())`.
#[inline]
pub fn keccak256_field_elements_direct<F>(elements: &[math::field::element::FieldElement<F>]) -> [u8; OUTPUT_LEN]
where
    F: math::field::traits::IsField,
    math::field::element::FieldElement<F>: math::traits::ByteConversion,
{
    use math::traits::ByteConversion;
    // Each element contributes BYTE_LEN bytes, lane-aligned (multiple of 8).
    let elem_bytes = <math::field::element::FieldElement<F>>::BYTE_LEN;
    debug_assert_eq!(elem_bytes % 8, 0, "element byte length must be lane-aligned");
    let total_bytes = elements.len() * elem_bytes;
    debug_assert!(total_bytes < RATE, "leaf too wide for single-block direct absorb");

    let lanes_per_elem = elem_bytes / 8;
    let mut state = [0u64; 25];
    let mut lane_idx = 0usize;
    for element in elements.iter() {
        let bytes = element.to_bytes_be();
        for chunk in bytes.as_ref().chunks_exact(8) {
            state[lane_idx] = u64::from_le_bytes(chunk.try_into().unwrap());
            lane_idx += 1;
        }
    }
    // pad10*1 directly into state lanes — no intermediate [u8; RATE] buffer.
    // Byte `total_bytes` is in lane `total_bytes/8` at byte offset `total_bytes%8`.
    let pad_lane = total_bytes / 8;
    let pad_shift = (total_bytes % 8) * 8;
    state[pad_lane] ^= 0x01u64 << pad_shift;
    // Last rate byte (byte 135) is in lane 16, byte offset 7 → shift 56.
    state[16] ^= 0x80u64 << 56;
    keccak::f1600(&mut state);
    let mut out = [0u8; OUTPUT_LEN];
    for (chunk, lane) in out.chunks_exact_mut(8).zip(state.iter()) {
        chunk.copy_from_slice(&lane.to_le_bytes());
    }
    let _ = lanes_per_elem; // suppress unused warning in release builds
    out
}

/// Streaming Keccak256 hasher, byte-identical to `sha3::Keccak256` but built on a
/// direct `keccak::f1600` (the `KeccakPermute` precompile on the guest) and a
/// fixed-rate buffer, skipping `sha3`'s generic `block_buffer`/`Digest` machinery.
///
/// Drop-in for the transcript's incremental absorb (`update`) + squeeze
/// (`finalize` / `finalize_reset`) usage. `update` XORs bytes into the rate and
/// permutes on each completed 136-byte block; `finalize` applies pad10*1 to the
/// partial block, permutes once, and returns the first 32 squeezed bytes.
#[derive(Clone)]
pub struct Keccak256Hasher {
    /// Sponge state.
    state: [u64; 25],
    /// Pending rate bytes not yet absorbed+permuted (length `< RATE`).
    buf: [u8; RATE],
    /// Number of valid bytes in `buf`.
    buf_len: usize,
}

impl Default for Keccak256Hasher {
    fn default() -> Self {
        Self::new()
    }
}

impl Keccak256Hasher {
    #[inline]
    pub fn new() -> Self {
        Self {
            state: [0u64; 25],
            buf: [0u8; RATE],
            buf_len: 0,
        }
    }

    /// Absorb `input`, permuting once per completed rate block. Equivalent to
    /// `sha3::Keccak256::update`.
    #[inline]
    pub fn update(&mut self, mut input: &[u8]) {
        // Fill the partial buffer first.
        if self.buf_len > 0 {
            let take = core::cmp::min(RATE - self.buf_len, input.len());
            self.buf[self.buf_len..self.buf_len + take].copy_from_slice(&input[..take]);
            self.buf_len += take;
            input = &input[take..];
            if self.buf_len == RATE {
                let block = self.buf;
                absorb_block(&mut self.state, &block);
                keccak::f1600(&mut self.state);
                self.buf_len = 0;
            } else {
                // Partial buffer still not full and the input is exhausted; the
                // already-buffered bytes must be kept (do NOT fall through to the
                // remainder stash, which would clobber `buf_len`).
                debug_assert!(input.is_empty());
                return;
            }
        }
        // At this point the buffer is empty. Absorb whole blocks straight from the
        // input, then stash the trailing partial block.
        let mut chunks = input.chunks_exact(RATE);
        for block in chunks.by_ref() {
            absorb_block(&mut self.state, block);
            keccak::f1600(&mut self.state);
        }
        let rem = chunks.remainder();
        self.buf[..rem.len()].copy_from_slice(rem);
        self.buf_len = rem.len();
    }

    /// Pad and squeeze the 32-byte digest WITHOUT consuming `self` — equivalent to
    /// `sha3::Keccak256::clone().finalize()`. Used by the transcript's `state()`.
    #[inline]
    pub fn finalize(&self) -> [u8; OUTPUT_LEN] {
        let mut state = self.state;
        let mut last = [0u8; RATE];
        last[..self.buf_len].copy_from_slice(&self.buf[..self.buf_len]);
        last[self.buf_len] ^= 0x01;
        last[RATE - 1] ^= 0x80;
        absorb_block(&mut state, &last);
        keccak::f1600(&mut state);

        let mut out = [0u8; OUTPUT_LEN];
        for (chunk, lane) in out.chunks_exact_mut(8).zip(state.iter()) {
            chunk.copy_from_slice(&lane.to_le_bytes());
        }
        out
    }

    /// Squeeze the digest and reset to a fresh state — equivalent to
    /// `sha3::Keccak256::finalize_reset`.
    #[inline]
    pub fn finalize_reset(&mut self) -> [u8; OUTPUT_LEN] {
        let out = self.finalize();
        *self = Self::new();
        out
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
    fn two_nodes_matches_keccak256_single_block() {
        for seed in 0u8..32 {
            let mut left = [0u8; 32];
            let mut right = [0u8; 32];
            for (i, b) in left.iter_mut().enumerate() {
                *b = seed.wrapping_add(i as u8);
            }
            for (i, b) in right.iter_mut().enumerate() {
                *b = seed.wrapping_add(32 + i as u8);
            }
            let mut input = [0u8; 64];
            input[..32].copy_from_slice(&left);
            input[32..].copy_from_slice(&right);
            assert_eq!(
                keccak256_two_nodes(&left, &right),
                reference(&input),
                "keccak256_two_nodes mismatch at seed={seed}"
            );
        }
    }

    #[test]
    fn four_nodes_matches_keccak256_single_block() {
        for seed in 0u8..32 {
            let mut children = [[0u8; 32]; 4];
            for (c, child) in children.iter_mut().enumerate() {
                for (i, b) in child.iter_mut().enumerate() {
                    *b = seed.wrapping_add((c * 32 + i) as u8);
                }
            }
            let mut input = [0u8; 128];
            for (c, child) in children.iter().enumerate() {
                input[c * 32..(c + 1) * 32].copy_from_slice(child);
            }
            assert_eq!(
                keccak256_four_nodes(&children),
                reference(&input),
                "keccak256_four_nodes mismatch at seed={seed}"
            );
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
    fn streaming_hasher_matches_sha3_incremental() {
        // Mirror the transcript's usage: a sequence of variably-sized updates
        // (spanning rate-block boundaries) interleaved with finalize_reset and a
        // non-consuming finalize, checked against sha3::Keccak256 step for step.
        let updates: &[&[u8]] = &[
            &[],
            &[0xAB],
            &[1u8; 32],
            &[2u8; 135],
            &[3u8; 136],
            &[4u8; 137],
            &[5u8; 300],
            &[6u8; 8],
        ];

        let mut mine = Keccak256Hasher::new();
        let mut theirs = Keccak256::new();
        for (i, u) in updates.iter().enumerate() {
            mine.update(u);
            theirs.update(u);
            // Non-consuming digest must match clone().finalize().
            assert_eq!(
                mine.finalize(),
                <[u8; 32]>::from(theirs.clone().finalize()),
                "finalize mismatch after update {i}"
            );
        }
        // Consuming reset must match finalize_reset, and the fresh state must keep
        // agreeing afterwards.
        assert_eq!(
            mine.finalize_reset(),
            <[u8; 32]>::from(theirs.finalize_reset())
        );
        mine.update(&[7u8; 50]);
        theirs.update(&[7u8; 50]);
        assert_eq!(mine.finalize(), <[u8; 32]>::from(theirs.clone().finalize()));
    }

    #[test]
    fn field_elements_direct_matches_scratch_path() {
        // Verify that `keccak256_field_elements_direct` produces byte-identical
        // output to the scratch-buffer path (serialize to bytes, then
        // `keccak256_single_block`) for the two field types used by the verifier:
        // Goldilocks (8 bytes/element) and Fp3 extension (24 bytes/element).
        use math::field::{
            element::FieldElement,
            goldilocks::GoldilocksField,
        };
        use math::traits::ByteConversion;
        type Fp = GoldilocksField;
        type FpE = FieldElement<Fp>;

        // --- Goldilocks (8 bytes/element) ---
        for n in [1usize, 3, 5, 8, 16] {
            let elements: alloc::vec::Vec<FpE> = (0..n)
                .map(|i| FpE::from(i as u64 * 0x9e3779b97f4a7c15 + 1))
                .collect();
            // Reference: serialize then hash.
            let mut bytes = alloc::vec::Vec::new();
            for e in &elements {
                bytes.extend_from_slice(e.to_bytes_be().as_ref());
            }
            let reference = if bytes.len() < RATE {
                keccak256_single_block(&bytes)
            } else {
                keccak256(&bytes)
            };
            let direct = keccak256_field_elements_direct::<Fp>(&elements);
            assert_eq!(
                direct, reference,
                "Goldilocks mismatch for n={n} elements"
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
