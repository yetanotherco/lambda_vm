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

use crate::syscalls::keccak_permute;

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
            // Keccak state lanes are little-endian, so an in-memory u64 load
            // equals the from_le_bytes lane value.
            if self.offset % 8 == 0 && (input.as_ptr() as usize) % 8 == 0 && input.len() >= 8 {
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

        // Squeeze the first 32 bytes (4 u64 lanes).
        for (i, chunk) in output.chunks_exact_mut(8).enumerate() {
            chunk.copy_from_slice(&self.state[i].to_le_bytes());
        }
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
