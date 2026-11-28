//! ★ The RPX grind's host/device contract, and the endianness that decides it.
//!
//! The device search takes the 32-byte inner hash as four `u64`s, and the two
//! arms read those bytes in OPPOSITE orders — little-endian lanes for keccak,
//! big-endian felts for RPX. Crossing them compiles, runs, and searches for a
//! nonce under a message the host never hashes; the host check then rejects
//! every nonce the device returns and the prover falls back to the CPU forever,
//! which is a performance cliff with no error attached to it.
//!
//! So the mapping is pinned here against the ORACLE TABLE the CUDA kernel is
//! itself checked against — `RPX_GRIND_VECTORS` in
//! `crypto/math-cuda/tests/host_kat/rpx_kat_vectors.h`, three rows printed by
//! the per-table branch's host implementation. Each row carries the four
//! big-endian `inner_felts`, the smallest valid nonce, and `le_nonce`: what the
//! same kernel answers on the LITTLE-endian reading, which the header records as
//! `u64::MAX` — nothing found. That last column is the endianness control, and
//! it is why these tests need no GPU: they check the HOST side of an agreement
//! whose device side is pinned to the same numbers.

use alloc::vec::Vec;
use digest::Digest;

use crate::grinding::{inner_hash_felts, inner_hash_lanes, is_valid_nonce};
use crate::hash::rpx::Rpx256Digest;

/// One row of `RPX_GRIND_VECTORS`: the seed byte (repeated 32 times), the
/// grinding factor, the four BIG-endian inner felts, and the smallest valid
/// nonce.
const RPX_GRIND_VECTORS: [(u8, u8, [u64; 4], u64); 3] = [
    (
        90,
        12,
        [
            17047917526726690733,
            2027278666509702433,
            4678289907902145381,
            4242003890993108442,
        ],
        1342,
    ),
    (
        17,
        13,
        [
            3807340077325453675,
            129745844021573959,
            15014385560057355003,
            944573484564438641,
        ],
        300,
    ),
    (
        32,
        14,
        [
            5597071933014793605,
            8702110216523445336,
            2882478612521280078,
            9429844132731097150,
        ],
        705,
    ),
];

fn seed_of(byte: u8) -> [u8; 32] {
    [byte; 32]
}

/// ★★ `inner_hash_felts` reproduces the oracle's four felts, on every row.
///
/// This is the mapping itself: the four `u64`s the device search is handed.
/// Nothing in this repository produced these numbers — they are the per-table
/// branch's host implementation, and the CUDA kernel is checked against the
/// same table.
#[test]
fn the_inner_felts_match_the_device_oracle_table() {
    for (byte, factor, want, _) in RPX_GRIND_VECTORS {
        let got = inner_hash_felts::<Rpx256Digest>(&seed_of(byte), factor);
        assert_eq!(got, want, "seed 0x{byte:02x}, factor {factor}");
    }
}

/// ★ And the oracle's nonce passes the HOST predicate — the agreement the
/// device dispatch rests on, checked without a device.
#[test]
fn the_oracle_nonce_passes_the_host_predicate() {
    for (byte, factor, _, nonce) in RPX_GRIND_VECTORS {
        assert!(
            is_valid_nonce::<Rpx256Digest>(&seed_of(byte), nonce, factor),
            "seed 0x{byte:02x}, factor {factor}: nonce {nonce} rejected"
        );
    }
}

/// ★ …and it really is the SMALLEST, by exhaustive scan — a different algorithm
/// from the one that produced it.
#[test]
fn the_oracle_nonce_is_the_smallest_valid_one() {
    for (byte, factor, _, nonce) in RPX_GRIND_VECTORS {
        let seed = seed_of(byte);
        assert!(nonce > 0, "the scan below is vacuous at nonce 0");
        assert!(
            (0..nonce).all(|n| !is_valid_nonce::<Rpx256Digest>(&seed, n, factor)),
            "seed 0x{byte:02x}, factor {factor}: a nonce below {nonce} is also valid"
        );
    }
}

/// ⚠⚠ **THE ENDIANNESS CONTROL.** The little-endian reading of the same inner
/// hash is a DIFFERENT message.
///
/// Without this, `inner_hash_felts` could be `inner_hash_lanes` with a new name
/// and every test above would still pass — they would simply all be about the
/// wrong four `u64`s together. The header records `le_nonce = u64::MAX` for all
/// three rows: on the little-endian reading the kernel finds nothing at all in
/// its scanned block.
#[test]
fn the_little_endian_reading_is_a_different_message() {
    for (byte, factor, felts, _) in RPX_GRIND_VECTORS {
        let seed = seed_of(byte);
        let lanes = inner_hash_lanes::<Rpx256Digest>(&seed, factor);
        assert_ne!(
            lanes, felts,
            "seed 0x{byte:02x}: the two readings must differ, or there is nothing to get wrong"
        );
    }
}

/// ✓ The two readings are byte-reversals of each other, lane for lane — so the
/// difference above is exactly the endianness and not a hash that moved.
#[test]
fn the_two_readings_are_byte_reversals_of_one_another() {
    for (byte, factor, _, _) in RPX_GRIND_VECTORS {
        let seed = seed_of(byte);
        let lanes = inner_hash_lanes::<Rpx256Digest>(&seed, factor);
        let felts = inner_hash_felts::<Rpx256Digest>(&seed, factor);
        for (i, (l, f)) in lanes.iter().zip(&felts).enumerate() {
            assert_eq!(l.swap_bytes(), *f, "lane {i} of seed 0x{byte:02x}");
        }
    }
}

/// ★ The preimage is ONE rate-8 block, which is what makes the kernel's
/// `init(5)` right: `inner_hash ‖ nonce` is 40 bytes, five felts, padding flag
/// `5 mod 8 = 5`.
///
/// Checked by computing the digest the long way — felts in, sponge out — and
/// requiring it to equal what the production predicate hashes from bytes.
#[test]
fn the_grind_preimage_is_five_felts_in_one_block() {
    use crate::hash::rpx::{Fp, RATE_FELTS, digest_to_commitment, sponge_leaf};

    for (byte, factor, felts, nonce) in RPX_GRIND_VECTORS {
        // What the kernel absorbs: the four inner felts, then the nonce.
        let mut block: Vec<Fp> = felts.iter().map(|v| Fp::from(*v)).collect();
        block.push(Fp::from(nonce));
        assert_eq!(block.len(), 5, "the grind preimage is five felts");
        assert!(block.len() <= RATE_FELTS, "…and therefore one rate block");

        let by_felts = digest_to_commitment(&sponge_leaf(&block));

        // What the host predicate hashes: the 32 inner bytes then the nonce,
        // big-endian, through the production digest.
        let mut inner = [0u8; 32];
        for (i, v) in felts.iter().enumerate() {
            inner[i * 8..(i + 1) * 8].copy_from_slice(&v.to_be_bytes());
        }
        let mut data = [0u8; 40];
        data[..32].copy_from_slice(&inner);
        data[32..].copy_from_slice(&nonce.to_be_bytes());
        let by_bytes: [u8; 32] = Rpx256Digest::digest(data).into();

        assert_eq!(
            by_felts, by_bytes,
            "seed 0x{byte:02x}: the felt form and the byte form must be one hash"
        );

        // And that digest's leading u64 is what `limit` is compared against.
        let head = u64::from_be_bytes(by_bytes[..8].try_into().unwrap());
        assert!(
            head < 1u64 << (64 - factor),
            "seed 0x{byte:02x}: the oracle nonce must clear its own limit"
        );
    }
}
