//! Parity: the device BLAKE3 compression function must equal the host reference
//! bit-for-bit, at both round counts.
//!
//! The reference is one function whose only parameter is the round count
//! (`crypto::hash::blake3::blake3_compress_rounds`, re-exported through
//! `blake3_reference` — the same function the host commitment backends and the
//! in-circuit chip use, not a copy of it). So the 7-round arm, where the `blake3` crate
//! is an external known-answer test, certifies the whole device code path — the
//! G function, the message schedule, the counter split, the feed-forward — and
//! the 6-round arm differs from it by a loop bound alone. That is why the
//! anchor below is worth more than a table of 6-round vectors would be.
//!
//! Every test here needs a GPU, like the rest of this crate's parity suite.

mod blake3_reference;

use blake3_reference::{
    BLAKE3_IV, BLAKE3_SIX_ROUNDS, BLAKE3_STANDARD_ROUNDS, FLAGS_ONE_BLOCK, blake3_compress_rounds,
    expected_device_rounds,
};
use math_cuda::blake3::{CompressInput, ProbeRounds, compress_probe, device_rounds};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// Inputs that between them move every field of the compression's framing.
///
/// `block_len` and `flags` are state words, not lengths the kernel loops over, so
/// a port that dropped either would still pass on a single value of it — hence
/// the spread, including the 18..=64 range the host's canonical vectors cover and
/// the 36 the LFM socket uses. `t` carries values whose halves differ, since the
/// counter split is a real way to be wrong and a symmetric `t` cannot see it.
fn vectors(seed: u64) -> Vec<CompressInput> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let block_lens = [0u32, 1, 4, 18, 36, 63, 64];
    let flags = [0u32, 1, 2, 8, FLAGS_ONE_BLOCK, 0x0C, 0xFFFF_FFFF];
    let counters = [
        0u64,
        1,
        0xFFFF_FFFF,
        0x1_0000_0000,
        0xB4E1_357D_4A84_EB03,
        u64::MAX,
    ];

    let mut out = Vec::new();
    // Walk the framing exhaustively over random h/m: 7 × 7 × 6 = 294, plus the
    // eight `h = IV` cases below for 302 — not a multiple of the kernel's block
    // width, so the `tid >= n` guard is exercised too.
    for &block_len in block_lens.iter() {
        for &fl in flags.iter() {
            for &t in counters.iter() {
                out.push(CompressInput {
                    h: core::array::from_fn(|_| rng.r#gen::<u32>()),
                    m: core::array::from_fn(|_| rng.r#gen::<u32>()),
                    t,
                    block_len,
                    flags: fl,
                });
            }
        }
    }
    // `h = IV` is the case every real call site uses, and a random h would never
    // hit it: the feed-forward `out[i+8] = v[i+8] ^ h[i]` reads h a second time.
    for _ in 0..8 {
        out.push(CompressInput {
            h: BLAKE3_IV,
            m: core::array::from_fn(|_| rng.r#gen::<u32>()),
            t: 0,
            block_len: 64,
            flags: FLAGS_ONE_BLOCK,
        });
    }
    out
}

fn host_outputs(inputs: &[CompressInput], rounds: usize) -> Vec<[u32; 16]> {
    inputs
        .iter()
        .map(|i| blake3_compress_rounds(&i.h, &i.m, i.t, i.block_len, i.flags, rounds))
        .collect()
}

fn assert_parity(rounds: usize, probe: ProbeRounds, seed: u64) {
    let inputs = vectors(seed);
    let device = compress_probe(&inputs, probe).unwrap();
    let host = host_outputs(&inputs, rounds);
    assert_eq!(device.len(), host.len());
    for (i, (d, h)) in device.iter().zip(host.iter()).enumerate() {
        assert_eq!(
            d, h,
            "vector {i} mismatch at {rounds} rounds: input {:?}",
            inputs[i]
        );
    }
}

#[test]
fn device_compression_matches_host_at_six_rounds() {
    assert_parity(BLAKE3_SIX_ROUNDS, ProbeRounds::Six, 6001);
}

#[test]
fn device_compression_matches_host_at_seven_rounds() {
    assert_parity(BLAKE3_STANDARD_ROUNDS, ProbeRounds::Seven, 7001);
}

/// ★ **The external anchor.** At 7 rounds a message of at most 64 bytes is one
/// chunk and one block, so the whole tree hasher collapses to a single `f`:
/// `h = IV`, the block zero-padded and read as 16 little-endian words, `t = 0`,
/// `block_len` the true length, `flags = CHUNK_START|CHUNK_END|ROOT`. The digest
/// is `out[0..8]` little-endian.
///
/// Mirrors `prover/src/lfm/blake3.rs`'s `seven_rounds_is_the_blake3_crate`, over
/// the same 65 lengths and for the same reason: the length keys both `block_len`
/// and the padding, and a port that ignored `block_len` would pass at one length.
#[test]
fn seven_rounds_on_device_is_the_blake3_crate() {
    let inputs: Vec<CompressInput> = (0..=64usize)
        .map(|len| {
            let mut block = [0u8; 64];
            for (i, b) in block.iter_mut().take(len).enumerate() {
                *b = (i as u8).wrapping_mul(37).wrapping_add(11);
            }
            CompressInput {
                h: BLAKE3_IV,
                m: core::array::from_fn(|i| {
                    u32::from_le_bytes(block[4 * i..4 * i + 4].try_into().unwrap())
                }),
                t: 0,
                block_len: len as u32,
                flags: FLAGS_ONE_BLOCK,
            }
        })
        .collect();

    let device = compress_probe(&inputs, ProbeRounds::Seven).unwrap();
    for (len, out) in device.iter().enumerate() {
        let msg: Vec<u8> = (0..len)
            .map(|i| (i as u8).wrapping_mul(37).wrapping_add(11))
            .collect();
        let mut ours = [0u8; 32];
        for i in 0..8 {
            ours[4 * i..4 * i + 4].copy_from_slice(&out[i].to_le_bytes());
        }
        assert_eq!(
            ours,
            *blake3::hash(&msg).as_bytes(),
            "device 7-round compression must equal the blake3 crate at length {len}"
        );
    }
}

/// NEGATIVE CONTROL for the anchor above: at 6 rounds the device must NOT match
/// the crate. Without it, the anchor would pass just as well if the round count
/// were being ignored on device — the one bug that makes the whole
/// external-anchor argument vacuous, since the 6-round arm's only defence is
/// "the same code path with the loop bound changed".
#[test]
fn six_rounds_on_device_is_not_the_blake3_crate() {
    let msg: [u8; 36] = core::array::from_fn(|i| i as u8);
    let mut block = [0u8; 64];
    block[..36].copy_from_slice(&msg);
    let input = CompressInput {
        h: BLAKE3_IV,
        m: core::array::from_fn(|i| {
            u32::from_le_bytes(block[4 * i..4 * i + 4].try_into().unwrap())
        }),
        t: 0,
        block_len: 36,
        flags: FLAGS_ONE_BLOCK,
    };
    let out = compress_probe(&[input], ProbeRounds::Six).unwrap()[0];
    let mut ours = [0u8; 32];
    for i in 0..8 {
        ours[4 * i..4 * i + 4].copy_from_slice(&out[i].to_le_bytes());
    }
    assert_ne!(ours, *blake3::hash(&msg).as_bytes());
}

/// The two round counts must actually differ on device. Guards the shape of the
/// port itself: `blake3_compress` is one template instantiated twice, and a
/// template that collapsed (or a `#pragma unroll` that outran the bound) would
/// leave both parity tests above passing against a single arm.
#[test]
fn the_two_device_round_counts_differ() {
    let inputs = vectors(4242);
    let six = compress_probe(&inputs, ProbeRounds::Six).unwrap();
    let seven = compress_probe(&inputs, ProbeRounds::Seven).unwrap();
    for (i, (a, b)) in six.iter().zip(seven.iter()).enumerate() {
        assert_ne!(a, b, "6r and 7r agree on vector {i}");
    }
}

/// The round count the production kernels are compiled for must be the one the
/// `blake3-6round` feature selects.
///
/// This is the tripwire for a cross-crate feature mismatch: math-cuda's feature
/// and the host tree's are separate, nothing forces them equal, and the symptom
/// of a mismatch is a GPU tree that commits under a different hash than the CPU
/// one — no panic, no log line, just a proof that fails to verify. Asserting the
/// cubin's own round count is what turns that into a test failure.
#[test]
fn the_compiled_in_round_count_is_the_feature() {
    assert_eq!(
        device_rounds().unwrap() as usize,
        expected_device_rounds(),
        "kernels/blake3.cu was compiled for a different round count than the \
         math-cuda `blake3-6round` feature selects — check build.rs's -D plumbing"
    );

    // And the default probe must be the corresponding explicit arm, which is what
    // ties `blake3_merkle_level`'s hash to the number reported above.
    let inputs = vectors(909);
    let default = compress_probe(&inputs, ProbeRounds::CompiledIn).unwrap();
    let host = host_outputs(&inputs, expected_device_rounds());
    assert_eq!(default, host);
}
