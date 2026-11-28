//! The joint leaf of a 7-round BLAKE3 round-0 tree: one leaf over every
//! polynomial of a grouped run at one fold block, the BLAKE3 twin of
//! [`super::p1::joint_leaf`].
//!
//! # The bytes
//!
//! A joint leaf over `blocks` blocks of `per_block` values of `width` bytes is
//! [`Blake3Std`] over the message
//!
//! ```text
//! prefix (64 bytes) ‖ block 0's values ‖ block 1's values ‖ …
//! ```
//!
//! each value as its canonical big-endian bytes (what the per-polynomial leaf,
//! `FieldElementVectorBackend<_, Blake3Std, 32>`, absorbs), the blocks in slot
//! order. The prefix is one whole BLAKE3 block ([`joint_leaf_prefix`]):
//!
//! | bytes | content |
//! |---|---|
//! | `0..8` | `0xFF` × 8: not the big-endian bytes of any canonical Goldilocks value (all are below `2^64 − 2^32 + 1`) |
//! | `8..32` | [`JOINT_LEAF_DOMAIN`], ASCII |
//! | `32..40` | the values' byte count, `blocks · per_block · width`, big-endian |
//! | `40..48` | `blocks`, big-endian |
//! | `48..56` | `per_block`, big-endian |
//! | `56..64` | `width`, big-endian |
//!
//! So the prefix carries the leaf's width (its byte count) and its block
//! count, as the Poseidon1 joint leaf's first capacity does
//! (`poseidon1_stark::joint_leaf_capacity`), plus the block's own shape.
//!
//! # Past one chunk
//!
//! A run of two blocks of 64 base values is already 1,088 bytes with the
//! prefix, past one BLAKE3 chunk. [`Blake3Std`] keeps its own framing there:
//! one chunk that never ends (`chain.rs`'s header), so the message is one
//! chain of 64-byte blocks from the IV, `CHUNK_START` on the first (the
//! prefix), `CHUNK_END | ROOT` and the true byte count on the last, the
//! counter 0 throughout. Up to 1,024 bytes this is `blake3::hash`; past it,
//! it is not. Its collision resistance reduces to the 7-round compression
//! function's: two colliding messages of one block count collide at some
//! compression; of different counts, walking back from the last block reaches
//! the shorter one's first block, which carries `CHUNK_START` where the
//! longer's block at that place does not, so two different inputs give one
//! chaining value there.
//!
//! # The prefix costs no compression where it matters
//!
//! The prefix is a verifier constant, so its compression is too: the chain
//! after it is [`joint_leaf_midstate`], and a card kernel or an in-guest
//! verifier starts there with `CHUNK_START` spent, paying exactly the
//! compressions of the values. The host here hashes the prefix like any other
//! block; the two are the same digest (`a_joint_leaf_from_its_midstate_is_the_leaf`).

use alloc::vec::Vec;

use digest::Digest;
use math::{
    field::{element::FieldElement, traits::IsField},
    traits::AsBytes,
};

use crate::hash::blake3::chain::{BLOCK_LEN, block_flags};
use crate::hash::blake3::{BLAKE3_IV, BLAKE3_STANDARD_ROUNDS, Blake3Std, blake3_compress_rounds};

/// The domain bytes of a joint leaf's prefix (bytes `8..32`).
pub const JOINT_LEAF_DOMAIN: [u8; 24] = *b"LAMBDAVM_WHIR_JOINT_LEAF";

/// Bytes of a joint leaf's prefix: one BLAKE3 block.
pub const JOINT_LEAF_PREFIX_LEN: usize = BLOCK_LEN;

/// ★ The prefix of a joint leaf of `blocks` blocks of `per_block` values of
/// `width` bytes each (see the module header for the layout).
pub fn joint_leaf_prefix(blocks: u64, per_block: u64, width: u64) -> [u8; JOINT_LEAF_PREFIX_LEN] {
    let mut prefix = [0u8; JOINT_LEAF_PREFIX_LEN];
    prefix[..8].fill(0xFF);
    prefix[8..32].copy_from_slice(&JOINT_LEAF_DOMAIN);
    let total = blocks.wrapping_mul(per_block).wrapping_mul(width);
    prefix[32..40].copy_from_slice(&total.to_be_bytes());
    prefix[40..48].copy_from_slice(&blocks.to_be_bytes());
    prefix[48..56].copy_from_slice(&per_block.to_be_bytes());
    prefix[56..64].copy_from_slice(&width.to_be_bytes());
    prefix
}

/// The chaining value after a joint leaf's prefix: the 7-round compression of
/// the IV over the prefix's 16 little-endian words, counter 0, `block_len`
/// 64, `CHUNK_START` (the first block of a message of more than one block).
/// A chain that starts here with `CHUNK_START` spent and absorbs the values
/// ends at the joint leaf.
pub fn joint_leaf_midstate(blocks: u64, per_block: u64, width: u64) -> [u32; 8] {
    let prefix = joint_leaf_prefix(blocks, per_block, width);
    let mut m = [0u32; 16];
    for (word, bytes) in m.iter_mut().zip(prefix.chunks_exact(4)) {
        *word = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    }
    let out = blake3_compress_rounds(
        &BLAKE3_IV,
        &m,
        0,
        BLOCK_LEN as u32,
        block_flags(0, 2),
        BLAKE3_STANDARD_ROUNDS,
    );
    let mut cv = [0u32; 8];
    cv.copy_from_slice(&out[..8]);
    cv
}

/// ★ The joint leaf of `blocks` equal blocks laid end to end in `values`, as a
/// node. `None` when `values` is empty, `blocks` is zero or does not divide
/// `values`, or the values are not all one width: such a leaf has no prefix.
pub fn joint_leaf<F>(values: &[FieldElement<F>], blocks: usize) -> Option<[u8; 32]>
where
    F: IsField,
    FieldElement<F>: AsBytes,
{
    if blocks == 0 || values.is_empty() || !values.len().is_multiple_of(blocks) {
        return None;
    }
    let width = values[0].as_bytes().len();
    let per_block = values.len() / blocks;
    let mut hasher = Blake3Std::new();
    Digest::update(
        &mut hasher,
        joint_leaf_prefix(blocks as u64, per_block as u64, width as u64),
    );
    let mut same_width = true;
    let mut buffer: Vec<u8> = Vec::with_capacity(width);
    for value in values {
        buffer.clear();
        value.stream_bytes(&mut |bytes| buffer.extend_from_slice(bytes));
        same_width &= buffer.len() == width;
        Digest::update(&mut hasher, &buffer);
    }
    same_width.then(|| hasher.finalize_digest())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hash::blake3::chain::{block_len_of, num_blocks};
    use math::field::goldilocks::GoldilocksField as F;

    type FE = FieldElement<F>;

    fn values(n: usize, seed: u64) -> Vec<FE> {
        (0..n as u64)
            .map(|i| {
                FE::from(
                    i.wrapping_add(seed)
                        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                        .wrapping_add(7),
                )
            })
            .collect()
    }

    fn plain(bytes: &[u8]) -> [u8; 32] {
        <Blake3Std as Digest>::digest(bytes).into()
    }

    fn be_bytes(values: &[FE]) -> Vec<u8> {
        let mut out = Vec::new();
        for v in values {
            v.stream_bytes(&mut |b| out.extend_from_slice(b));
        }
        out
    }

    /// The chain spelled out from the midstate, block by block: what a card
    /// kernel or an in-guest verifier replays. `CHUNK_START` is spent on the
    /// prefix, so block `i` of the values is block `i + 1` of the message.
    fn from_midstate(blocks: usize, values: &[FE]) -> [u8; 32] {
        let per_block = (values.len() / blocks) as u64;
        let bytes = be_bytes(values);
        let mut cv = joint_leaf_midstate(blocks as u64, per_block, 8);
        let len = JOINT_LEAF_PREFIX_LEN + bytes.len();
        let n = num_blocks(len);
        for i in 1..n {
            let mut block = [0u8; BLOCK_LEN];
            let start = (i - 1) * BLOCK_LEN;
            let end = (start + BLOCK_LEN).min(bytes.len());
            block[..end - start].copy_from_slice(&bytes[start..end]);
            let mut m = [0u32; 16];
            for (w, b) in m.iter_mut().zip(block.chunks_exact(4)) {
                *w = u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
            }
            let out = blake3_compress_rounds(
                &cv,
                &m,
                0,
                block_len_of(i, len),
                block_flags(i, n),
                BLAKE3_STANDARD_ROUNDS,
            );
            cv.copy_from_slice(&out[..8]);
        }
        let mut digest = [0u8; 32];
        for (i, w) in cv.iter().enumerate() {
            digest[4 * i..4 * i + 4].copy_from_slice(&w.to_le_bytes());
        }
        digest
    }

    /// ★ The joint leaf is `Blake3Std` over the prefix and the values' bytes,
    /// and up to one chunk that is the `blake3` crate's hash of those bytes:
    /// the external anchor for the prefix's placement and the values' order.
    #[test]
    fn a_joint_leaf_is_blake3_over_its_prefix_and_values() {
        for (blocks, per_block) in [(2usize, 4usize), (3, 8), (2, 32), (4, 16), (3, 64), (4, 64)] {
            let v = values(blocks * per_block, 5);
            let mut message = joint_leaf_prefix(blocks as u64, per_block as u64, 8).to_vec();
            message.extend_from_slice(&be_bytes(&v));
            let leaf = joint_leaf(&v, blocks).expect("a joint leaf");
            assert_eq!(leaf, plain(&message));
            if message.len() <= 1024 {
                assert_eq!(
                    leaf,
                    *blake3::hash(&message).as_bytes(),
                    "{blocks} x {per_block}"
                );
            } else {
                assert_ne!(
                    leaf,
                    *blake3::hash(&message).as_bytes(),
                    "past one chunk the chain is not the crate's tree"
                );
            }
        }
    }

    /// ★ The chain from the prefix's midstate is the leaf, at every shape a
    /// run takes (2–4 blocks of 64 values: 1,024–2,048 bytes of values) and
    /// at lengths that end mid-block.
    #[test]
    fn a_joint_leaf_from_its_midstate_is_the_leaf() {
        for (blocks, per_block) in [(2usize, 64usize), (3, 64), (4, 64), (2, 3), (3, 5), (4, 16)] {
            let v = values(blocks * per_block, 11);
            assert_eq!(
                from_midstate(blocks, &v),
                joint_leaf(&v, blocks).unwrap(),
                "{blocks} x {per_block}"
            );
        }
    }

    /// The prefix's layout, byte for byte, and the midstate is not the IV.
    #[test]
    fn the_prefix_is_its_spelled_out_layout() {
        let p = joint_leaf_prefix(3, 64, 8);
        assert_eq!(&p[..8], &[0xFF; 8]);
        assert_eq!(&p[8..32], b"LAMBDAVM_WHIR_JOINT_LEAF");
        assert_eq!(u64::from_be_bytes(p[32..40].try_into().unwrap()), 1536);
        assert_eq!(u64::from_be_bytes(p[40..48].try_into().unwrap()), 3);
        assert_eq!(u64::from_be_bytes(p[48..56].try_into().unwrap()), 64);
        assert_eq!(u64::from_be_bytes(p[56..64].try_into().unwrap()), 8);
        assert_ne!(joint_leaf_midstate(3, 64, 8), BLAKE3_IV);
        // Bytes 0..8 are no canonical Goldilocks value's big-endian bytes.
        assert!(u64::from_be_bytes(p[..8].try_into().unwrap()) >= 0xFFFF_FFFF_0000_0001);
    }

    /// ★ The leaf binds its shape: the same values under another block count,
    /// the plain per-polynomial leaf over the same values, and the values with
    /// two blocks swapped all hash apart.
    #[test]
    fn the_leaf_binds_its_block_count_its_tag_and_its_order() {
        let v = values(3 * 64, 23);
        let leaf = joint_leaf(&v, 3).unwrap();
        // 192 values read as 2 blocks of 96 or 4 of 48.
        assert_ne!(leaf, joint_leaf(&v, 2).unwrap());
        assert_ne!(leaf, joint_leaf(&v, 4).unwrap());
        // One block: a prefix with a block count of 1 is still not the plain leaf.
        assert_ne!(joint_leaf(&v, 1).unwrap(), plain(&be_bytes(&v)));
        assert_ne!(leaf, plain(&be_bytes(&v)));
        // Blocks 0 and 1 swapped.
        let mut swapped = v[64..128].to_vec();
        swapped.extend_from_slice(&v[..64]);
        swapped.extend_from_slice(&v[128..]);
        assert_ne!(leaf, joint_leaf(&swapped, 3).unwrap());
        // One value moved.
        let mut moved = v.clone();
        moved[150] += FE::one();
        assert_ne!(leaf, joint_leaf(&moved, 3).unwrap());
    }

    /// A shape with no prefix has no leaf.
    #[test]
    fn a_leaf_with_no_shape_is_none() {
        let v = values(10, 1);
        assert_eq!(joint_leaf::<F>(&[], 2), None);
        assert_eq!(joint_leaf(&v, 0), None);
        assert_eq!(joint_leaf(&v, 3), None);
        assert!(joint_leaf(&v, 2).is_some());
    }
}
