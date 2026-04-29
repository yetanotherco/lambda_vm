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
//! On a non-`riscv64` host, `keccak_permute` panics — this module is only
//! meant to be used from guest programs (compiled to `riscv64im-lambda-vm-elf`).

use crate::syscalls::keccak_permute;

/// Keccak-256 sponge rate in bytes (1088 bits = 136 bytes; capacity = 512 bits).
const RATE_BYTES: usize = 136;

/// Keccak-256 domain-separator byte (per FIPS 202 / Ethereum convention).
/// Note: this is plain Keccak (0x01), not SHA-3 (0x06).
const DELIMITER: u8 = 0x01;

/// Last padding byte (set high bit of final rate byte).
const FINAL_PAD_BIT: u8 = 0x80;

/// Incremental Keccak-256 hasher.
pub struct Keccak256 {
    state: [u64; 25],
    buf: [u8; RATE_BYTES],
    buf_len: usize,
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
            buf: [0; RATE_BYTES],
            buf_len: 0,
        }
    }

    /// Absorb more input into the sponge.
    pub fn update(&mut self, mut input: &[u8]) {
        if self.buf_len > 0 {
            let take = (RATE_BYTES - self.buf_len).min(input.len());
            self.buf[self.buf_len..self.buf_len + take].copy_from_slice(&input[..take]);
            self.buf_len += take;
            input = &input[take..];
            if self.buf_len == RATE_BYTES {
                let block = self.buf;
                self.absorb_block(&block);
                self.buf_len = 0;
            }
        }
        while input.len() >= RATE_BYTES {
            let mut block = [0u8; RATE_BYTES];
            block.copy_from_slice(&input[..RATE_BYTES]);
            self.absorb_block(&block);
            input = &input[RATE_BYTES..];
        }
        if !input.is_empty() {
            self.buf[..input.len()].copy_from_slice(input);
            self.buf_len = input.len();
        }
    }

    /// Finalize the sponge and write the 32-byte digest into `output`.
    pub fn finalize(mut self, output: &mut [u8; 32]) {
        // Pad: append delimiter byte, zeros, final pad bit at the rate boundary.
        let mut last = [0u8; RATE_BYTES];
        last[..self.buf_len].copy_from_slice(&self.buf[..self.buf_len]);
        last[self.buf_len] = DELIMITER;
        last[RATE_BYTES - 1] |= FINAL_PAD_BIT;
        self.absorb_block(&last);

        // Squeeze the first 32 bytes (4 u64 lanes).
        for (i, lane) in self.state[..4].iter().enumerate() {
            output[i * 8..(i + 1) * 8].copy_from_slice(&lane.to_le_bytes());
        }
    }

    fn absorb_block(&mut self, block: &[u8; RATE_BYTES]) {
        // XOR the block into the rate portion of the state (first 17 lanes for r=1088).
        for (i, lane) in self.state.iter_mut().take(RATE_BYTES / 8).enumerate() {
            let mut bytes = [0u8; 8];
            bytes.copy_from_slice(&block[i * 8..(i + 1) * 8]);
            *lane ^= u64::from_le_bytes(bytes);
        }
        keccak_permute(&mut self.state);
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
