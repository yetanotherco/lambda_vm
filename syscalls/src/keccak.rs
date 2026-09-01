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

#[cfg(all(
    feature = "keccak-sponge-accel",
    not(all(test, not(target_arch = "riscv64")))
))]
use crate::syscalls::keccak_absorb_blocks;

/// Software mirror of the `ECALL -4` sponge-absorb accelerator, so host
/// `cargo test --features keccak-sponge-accel` exercises the *composition*
/// (which blocks go to the accelerator, where padding lands) against the
/// reference digest. Semantics copied from the executor arm
/// (`executor/src/vm/instruction/execution.rs`, `SyscallNumbers::KeccakAbsorbBlocks`):
/// per block, XOR 17 little-endian dword lanes into the state, then permute.
///
/// It counts the blocks it absorbs so the tests can assert the accelerated
/// path actually fired instead of passing vacuously.
#[cfg(all(feature = "keccak-sponge-accel", test, not(target_arch = "riscv64")))]
fn keccak_absorb_blocks(state: &mut [u64; 25], data: &[u8], n_blocks: usize) {
    assert_eq!(data.len(), n_blocks * RATE_BYTES);
    assert!(n_blocks > 0);
    assert_eq!(data.as_ptr() as usize % 8, 0);
    for k in 0..n_blocks {
        let block = &data[k * RATE_BYTES..(k + 1) * RATE_BYTES];
        for (j, lane) in state.iter_mut().take(RATE_LANES).enumerate() {
            let dword: &[u8; 8] = block[j * 8..j * 8 + 8].try_into().unwrap();
            *lane ^= u64::from_le_bytes(*dword);
        }
        keccak_permute(state);
    }
    tests::ABSORBED_BLOCKS.fetch_add(n_blocks, core::sync::atomic::Ordering::Relaxed);
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

/// Guest-side mirror of the executor's `addr_limb_ok`: an
/// accelerator operand's LAST byte must fit inside the pointer's low 32-bit
/// limb, because the chip addresses each dword as `base_lo + offset` with no
/// carry into the high limb. The executor *traps* when this fails, so check it
/// here and fall back to software absorption instead — the fallback yields the
/// same digest, so this only ever costs cycles, never correctness. A 200-byte
/// state or a message buffer straddling a 4 GiB boundary is the only way to
/// hit it.
#[cfg(feature = "keccak-sponge-accel")]
#[inline(always)]
fn accel_low_limb_ok(addr: usize, len: usize) -> bool {
    debug_assert!(len > 0);
    ((addr as u64) & 0xFFFF_FFFF) + (len as u64 - 1) < (1u64 << 32)
}

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
            // Sponge-absorb accelerator (ECALL -4): standing at a rate-block
            // boundary with an 8-aligned pointer and at least one whole
            // 136-byte block, hand ALL whole blocks to the chip in a single
            // ecall. It XORs each block into lanes 0..17 and permutes, exactly
            // what the two paths below do in guest software, so the sponge is
            // left at `offset == 0` with `len % 136` bytes still to absorb —
            // which the loop then feeds to the unchanged software path, and
            // `finalize` pads as always. Padding never reaches the chip.
            //
            // Preconditions the executor enforces, and why each holds or is
            // checked: 8-alignment of the state is guaranteed by `[u64; 25]`;
            // 8-alignment of the data is the branch condition; `n_blocks > 0`
            // follows from `len >= RATE_BYTES`; low-limb room is checked
            // explicitly (see `accel_low_limb_ok`); and the state/data regions
            // cannot overlap because `&mut self` is exclusive, so `input`
            // cannot alias `self.state`.
            #[cfg(feature = "keccak-sponge-accel")]
            {
                let n_blocks = input.len() / RATE_BYTES;
                if cfg!(target_endian = "little")
                    && self.offset == 0
                    && n_blocks > 0
                    && (input.as_ptr() as usize).is_multiple_of(8)
                    && accel_low_limb_ok(input.as_ptr() as usize, n_blocks * RATE_BYTES)
                    && accel_low_limb_ok(self.state.as_ptr() as usize, 25 * 8)
                {
                    let (blocks, rest) = input.split_at(n_blocks * RATE_BYTES);
                    keccak_absorb_blocks(&mut self.state, blocks, n_blocks);
                    input = rest;
                    continue;
                }
            }

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
                // assembled each lane with `from_le_bytes` was tried in three
                // formulations and measured +4.7% cycles at the blowup8 preset
                // (see PR #847's measurement notes): on this
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
///
/// What these tests do NOT cover: the `keccak_permute` ecall itself and the
/// `#[cfg(target_arch = "riscv64")]` specialized call sites (the Merkle
/// backends' TypeId branches) — those are validated only by the blob oracle.
/// The generic-vs-specialized equivalence additionally rests on
/// `PlatformKeccak256` staying a pure passthrough of this sponge; see the
/// INVARIANT note in crypto/crypto/src/hash/platform_keccak.rs.
#[cfg(all(test, not(target_arch = "riscv64")))]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;
    use sha3::{Digest, Keccak256 as RefKeccak256};

    fn reference(input: &[u8]) -> [u8; 32] {
        RefKeccak256::digest(input).into()
    }

    /// Whole rate blocks routed through the `ECALL -4` shim, process-wide.
    /// Only meaningful as a "did the accelerated path fire at all" witness —
    /// tests run in parallel, so compare against a snapshot with `>`, never `==`.
    #[cfg(feature = "keccak-sponge-accel")]
    pub(super) static ABSORBED_BLOCKS: core::sync::atomic::AtomicUsize =
        core::sync::atomic::AtomicUsize::new(0);

    /// The A/B differential for the sponge-absorb accelerator. Run this file's
    /// tests twice — with and without `--features keccak-sponge-accel` — and
    /// both builds must match `sha3::Keccak256`; matching the same reference is
    /// what makes the two guest variants digest-identical.
    ///
    /// The buffer is `repr(align(8))` so the accelerator's alignment
    /// precondition is guaranteed rather than left to the allocator, and the
    /// lengths straddle every interesting boundary: empty, sub-block,
    /// rate−1/rate/rate+1, the same around 2 and 3 blocks, and ≥3-block
    /// messages where the chip does real work. Each length is also fed as two
    /// `update` calls so the accelerator is entered mid-stream (a split at 1 or
    /// 8 leaves a non-zero sponge offset that the software path must drain
    /// before the chip can take over).
    #[test]
    fn accel_length_sweep_matches_reference() {
        #[repr(align(8))]
        struct Aligned([u8; 640]);

        let mut buf = Aligned([0u8; 640]);
        for (i, b) in buf.0.iter_mut().enumerate() {
            *b = (i * 97 + 13) as u8;
        }
        assert_eq!(buf.0.as_ptr() as usize % 8, 0, "repr(align(8)) must hold");

        #[cfg(feature = "keccak-sponge-accel")]
        let absorbed_before = ABSORBED_BLOCKS.load(core::sync::atomic::Ordering::Relaxed);

        for len in [
            0, 1, 8, 135, 136, 137, 271, 272, 273, 407, 408, 409, 500, 544, 640,
        ] {
            let msg = &buf.0[..len];
            assert_eq!(keccak256(msg), reference(msg), "one-shot len={len}");

            for split in [1usize, 8, 64, 135, 136, 137] {
                if split > len {
                    continue;
                }
                let mut hasher = Keccak256::new();
                hasher.update(&msg[..split]);
                hasher.update(&msg[split..]);
                let mut out = [0u8; 32];
                hasher.finalize(&mut out);
                assert_eq!(out, reference(msg), "streaming len={len} split={split}");
            }
        }

        // Without this the test would still pass if the accelerated branch were
        // never taken (e.g. a mis-stated precondition silently disabling it).
        #[cfg(feature = "keccak-sponge-accel")]
        assert!(
            ABSORBED_BLOCKS.load(core::sync::atomic::Ordering::Relaxed) > absorbed_before,
            "accelerated absorb path never fired — the sweep proves nothing"
        );
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
        let mut rng = ChaCha8Rng::seed_from_u64(0x9E37_79B9_7F4A_7C15);
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

    /// Structural (allocator-independent) coverage of BOTH absorb paths. The
    /// other tests feed `Vec<u8>` slices, whose base alignment is up to the
    /// allocator — on current platforms they happen to be 8-aligned, so the
    /// whole-lane fast path is exercised only by luck. Here a `repr(align(8))`
    /// buffer GUARANTEES the aligned fast path, and a +1-offset view of the
    /// same bytes GUARANTEES the byte-wise fallback, across all padding
    /// boundaries.
    #[test]
    fn aligned_and_misaligned_paths_match_reference() {
        #[repr(align(8))]
        struct Aligned([u8; 3 * RATE_BYTES + 9]);

        let mut buf = Aligned([0u8; 3 * RATE_BYTES + 9]);
        for (i, b) in buf.0.iter_mut().enumerate() {
            *b = (i * 131 + 17) as u8;
        }
        assert_eq!(buf.0.as_ptr() as usize % 8, 0, "repr(align(8)) must hold");

        for len in [0, 1, 7, 8, 9, 63, 64, 135, 136, 137, 271, 272, 273, 400] {
            let aligned = &buf.0[..len];
            assert_eq!(keccak256(aligned), reference(aligned), "aligned len={len}");
            let misaligned = &buf.0[1..1 + len];
            assert_eq!(
                misaligned.as_ptr() as usize % 8,
                1,
                "offset view must be misaligned"
            );
            assert_eq!(
                keccak256(misaligned),
                reference(misaligned),
                "misaligned len={len}"
            );
        }
    }

    /// The fixed-shape parent path must equal hashing the 64-byte concatenation.
    #[test]
    fn pair_matches_reference() {
        let mut rng = ChaCha8Rng::seed_from_u64(0xD1B5_4A32_D192_ED03);
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
