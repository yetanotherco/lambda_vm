//! High-level Keccak-256 hasher backed by the lambda-vm `keccak_permute`
//! precompile (ECALL `u64::MAX - 1`).
//!
//! The precompile applies one Keccak-f[1600] permutation per call. This module
//! wraps it with the absorption/padding/squeezing logic of Keccak-256 so guest
//! code can hash arbitrary byte slices with a familiar API:
//!
//! ```ignore
//! let digest = lambda_vm_syscalls::keccak::keccak256(b"hello");
//! ```
//!
//! The sponge absorbs IN PLACE: the `[u64; 25]` state is the only buffer, and
//! input bytes are XORed directly into its rate lanes at a running offset
//! (tiny-keccak's design). There is no staging block, no per-block copy, and
//! no lane re-extraction — the permutation cost is the ecall itself, so every
//! byte moved around it is pure overhead the VM retires as real cycles.
//!
//! The VM traps on unaligned doubleword loads, so the whole-lane fast path is
//! gated on the input pointer's runtime alignment; misaligned input falls back
//! to byte-wise absorption. Correctness never depends on alignment.
//!
//! On a non-`riscv64` host, `keccak_permute` panics — this module is only
//! meant to be used from guest programs (compiled to `riscv64im-lambda-vm-elf`).

#[cfg(not(all(test, not(target_arch = "riscv64"))))]
use crate::syscalls::keccak_permute;

/// Software Keccak-f[1600] so host `cargo test` can exercise the sponge's
/// absorption/padding/squeezing against a reference implementation — the real
/// permutation is the VM ecall, unavailable off-guest. Guest builds and host
/// non-test builds are untouched.
#[cfg(all(test, not(target_arch = "riscv64")))]
fn keccak_permute(state: &mut [u64; 25]) {
    keccak::f1600(state);
}

/// Keccak-256 sponge rate in bytes (1088 bits = 136 bytes; capacity = 512 bits).
const RATE_BYTES: usize = 136;

/// Rate lanes (17 for r=1088).
const RATE_LANES: usize = RATE_BYTES / 8;

/// Keccak-256 domain-separator byte (per FIPS 202 / Ethereum convention).
/// Note: this is plain Keccak (0x01), not SHA-3 (0x06).
const DELIMITER: u8 = 0x01;

/// Final padding bit (high bit of the last rate byte), pre-shifted to its
/// position in the last rate lane. When the delimiter also lands in byte 135
/// the two XORs combine to the single-byte `0x81` pad, per pad10*1.
const FINAL_PAD_LANE_BIT: u64 = (0x80u64) << 56;

/// Incremental Keccak-256 hasher; the state doubles as the absorption buffer.
#[derive(Clone)]
pub struct Keccak256 {
    state: [u64; 25],
    /// Byte offset into the rate region where the next input byte XORs in.
    /// Invariant between calls: `offset < RATE_BYTES`.
    offset: usize,
}

impl Default for Keccak256 {
    fn default() -> Self {
        Self::new()
    }
}

impl Keccak256 {
    pub fn new() -> Self {
        Self {
            state: [0; 25],
            offset: 0,
        }
    }

    /// XOR one byte into the rate region at `self.offset` (no offset advance).
    #[inline(always)]
    fn xor_byte_at_offset(&mut self, b: u8) {
        self.state[self.offset / 8] ^= u64::from(b) << ((self.offset % 8) * 8);
    }

    /// Absorb one byte, permuting when the rate fills.
    #[inline(always)]
    fn absorb_byte(&mut self, b: u8) {
        self.xor_byte_at_offset(b);
        self.offset += 1;
        if self.offset == RATE_BYTES {
            keccak_permute(&mut self.state);
            self.offset = 0;
        }
    }

    /// Absorb more input into the sponge.
    pub fn update(&mut self, mut input: &[u8]) {
        while !input.is_empty() {
            // Whole-lane fast path: sponge offset on a lane boundary AND input
            // pointer 8-aligned (the VM traps on unaligned doubleword loads).
            // LE-only by construction: the raw `*const u64` read below equals the
            // required little-endian lane value only on a little-endian target, so
            // gate on it. `cfg!(target_endian = ...)` is a compile-time constant
            // that folds away on every real target (riscv64im-lambda-vm and the
            // host CI targets are all little-endian), leaving codegen unchanged;
            // the byte-wise fallback is endian-correct everywhere.
            if cfg!(target_endian = "little")
                && self.offset % 8 == 0
                && (input.as_ptr() as usize) % 8 == 0
                && input.len() >= 8
            {
                let lanes_left = (RATE_BYTES - self.offset) / 8;
                let take = lanes_left.min(input.len() / 8);
                let base = self.offset / 8;
                for i in 0..take {
                    // SAFETY: `input.as_ptr()` is 8-aligned (checked above) and
                    // `(i + 1) * 8 <= input.len()`, so this reads 8 in-bounds
                    // bytes at an aligned address; any bit pattern is a valid u64.
                    let lane = unsafe { (input.as_ptr().add(i * 8) as *const u64).read() };
                    self.state[base + i] ^= lane;
                }
                self.offset += take * 8;
                input = &input[take * 8..];
                if self.offset == RATE_BYTES {
                    keccak_permute(&mut self.state);
                    self.offset = 0;
                }
            } else {
                // Byte-wise fallback for misaligned input. A middle path that
                // assembled each lane with `from_le_bytes` (three formulations,
                // dropped commit f6d575ed) measured +4.7% cycles: on this
                // 1-cycle-per-instruction VM the extra shifts/ORs cost more than
                // the byte loop they replace, so don't re-propose it.
                self.absorb_byte(input[0]);
                input = &input[1..];
            }
        }
    }

    /// Finalize the sponge and write the 32-byte digest into `output`.
    pub fn finalize(mut self, output: &mut [u8; 32]) {
        // Pad in place: delimiter at the current offset, final bit at the end
        // of the rate. `offset < RATE_BYTES` always holds here, so the padded
        // block is exactly the one permutation below.
        self.xor_byte_at_offset(DELIMITER);
        self.state[RATE_LANES - 1] ^= FINAL_PAD_LANE_BIT;
        keccak_permute(&mut self.state);

        squeeze32_into(&self.state, output);
    }
}

/// Squeeze the 32-byte Keccak-256 digest out of a permuted state into `out`:
/// rate lanes 0..4, little-endian. Shared by the streaming
/// [`Keccak256::finalize`] and the fixed-shape [`keccak256_pair`] so the squeeze
/// loop lives in exactly one place. Writes into a caller-owned buffer (rather
/// than returning `[u8; 32]`) so `finalize` fills its `output` reference in
/// place, keeping the guest codegen identical to the pre-dedup loop. Do NOT
/// "simplify" this to a return-value form: that shape was measured at +81k
/// guest cycles (min preset) from the extra stack temporary it introduces.
#[inline(always)]
fn squeeze32_into(state: &[u64; 25], out: &mut [u8; 32]) {
    for (i, chunk) in out.chunks_exact_mut(8).enumerate() {
        chunk.copy_from_slice(&state[i].to_le_bytes());
    }
}

/// One-shot Keccak-256 hash of a single byte slice.
pub fn keccak256(input: &[u8]) -> [u8; 32] {
    let mut hasher = Keccak256::new();
    hasher.update(input);
    let mut out = [0u8; 32];
    hasher.finalize(&mut out);
    out
}

/// Keccak-256 of exactly two concatenated 32-byte nodes (64 bytes) — the fixed
/// shape of every Merkle parent hash. 64 bytes fit the 136-byte rate in one
/// block, so this skips the incremental sponge entirely: load the eight data
/// lanes straight from `left`/`right`, XOR the `pad10*1` bits in place, run one
/// permutation, squeeze four lanes. Byte-identical to feeding `left` then
/// `right` through the streaming [`Keccak256`] and finalizing.
///
/// The nodes are only byte-aligned, so lanes are assembled with `from_le_bytes`
/// over owned arrays — never an aligned doubleword load, which the VM would trap
/// on at a misaligned address.
pub fn keccak256_pair(left: &[u8; 32], right: &[u8; 32]) -> [u8; 32] {
    let mut state = [0u64; 25];
    // Bytes 0..64 span rate lanes 0..8: lanes 0..4 from `left`, 4..8 from `right`.
    for i in 0..4 {
        let l: &[u8; 8] = left[i * 8..i * 8 + 8].try_into().unwrap();
        state[i] = u64::from_le_bytes(*l);
        let r: &[u8; 8] = right[i * 8..i * 8 + 8].try_into().unwrap();
        state[4 + i] = u64::from_le_bytes(*r);
    }
    // pad10*1 for a 64-byte message at rate 136: delimiter at byte 64 (lane 8,
    // low byte) and the final bit at the last rate byte (byte 135, lane 16 high
    // byte). Both target lanes are still zero, so XOR == assignment.
    state[8] ^= u64::from(DELIMITER);
    state[RATE_LANES - 1] ^= FINAL_PAD_LANE_BIT;
    keccak_permute(&mut state);

    let mut out = [0u8; 32];
    squeeze32_into(&state, &mut out);
    out
}

/// Host-only differential tests: the sponge (absorption chunking, padding,
/// squeezing, the fixed-shape pair path) must produce digests byte-identical
/// to the reference `sha3::Keccak256` for every input length and every way of
/// slicing the input across `update` calls. The permutation itself is the
/// software `keccak::f1600` here (see `keccak_permute` above); on-guest the
/// end-to-end oracle is proof-blob acceptance (any digest difference diverges
/// the Fiat-Shamir transcript and fails verification loudly).
#[cfg(all(test, not(target_arch = "riscv64")))]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng, rngs::StdRng};
    use sha3::{Digest, Keccak256 as RefKeccak256};

    fn reference(input: &[u8]) -> [u8; 32] {
        RefKeccak256::digest(input).into()
    }

    /// Every length from empty through three full rate blocks (+2), so every
    /// padding boundary (135/136/137, 271/272/273, …) is hit.
    #[test]
    fn one_shot_matches_reference_for_all_lengths() {
        let data: Vec<u8> = (0..3 * RATE_BYTES + 2)
            .map(|i| (i * 31 + 7) as u8)
            .collect();
        for len in 0..=data.len() {
            assert_eq!(
                keccak256(&data[..len]),
                reference(&data[..len]),
                "digest mismatch at len={len}"
            );
        }
    }

    /// Differential chunking fuzz: random sub-slices (random start => misaligned
    /// pointers exercising the byte fallback) fed through `update` in random
    /// pieces must match the one-shot reference digest.
    #[test]
    fn chunked_misaligned_updates_match_reference() {
        let data: Vec<u8> = (0..1500).map(|i| (i * 131 + 17) as u8).collect();
        let mut rng = StdRng::seed_from_u64(0x9E37_79B9_7F4A_7C15);
        for case in 0..300 {
            let len = rng.random_range(0..data.len());
            let start = rng.random_range(0..data.len() - len + 1);
            let slice = &data[start..start + len];

            let mut hasher = Keccak256::new();
            let mut fed = 0;
            while fed < slice.len() {
                let n = 1 + rng.random_range(0..(slice.len() - fed).min(200));
                hasher.update(&slice[fed..fed + n]);
                fed += n;
            }
            let mut out = [0u8; 32];
            hasher.finalize(&mut out);
            assert_eq!(
                out,
                reference(slice),
                "chunked digest mismatch: case={case} start={start} len={len}"
            );
        }
    }

    /// The fixed-shape parent path must equal hashing the 64-byte concatenation.
    #[test]
    fn pair_matches_reference() {
        let mut rng = StdRng::seed_from_u64(0xD1B5_4A32_D192_ED03);
        for case in 0..64 {
            let mut left = [0u8; 32];
            let mut right = [0u8; 32];
            for b in left.iter_mut().chain(right.iter_mut()) {
                *b = rng.random();
            }
            let mut concat = [0u8; 64];
            concat[..32].copy_from_slice(&left);
            concat[32..].copy_from_slice(&right);
            assert_eq!(
                keccak256_pair(&left, &right),
                reference(&concat),
                "pair digest mismatch: case={case}"
            );
            assert_eq!(
                keccak256_pair(&left, &right),
                keccak256(&concat),
                "pair vs streaming sponge mismatch: case={case}"
            );
        }
    }
}
