//! A bounded cache of Keccak-256 digests for the short inputs the guest hashes again
//! and again.
//!
//! The ethrex guest hashes the same 20-byte addresses (account trie keys), 32-byte
//! storage slots (storage trie keys) and 64-byte mapping slots (the `KECCAK256`
//! opcode) many times in a block, and the empty input too. Each repeat costs a
//! `keccak_permute` call (a KECCAK row and 24 KECCAK_RND rows in the proof) and its
//! sponge. At the median block the cache saves 6,852 of the guest's 96,384
//! permutations.
//!
//! One direct-mapped table per length. A lookup compares the whole input with the
//! stored one, so a hit returns the digest of exactly that input; a different input
//! in the same slot is a miss and replaces it. The tables have a fixed number of
//! entries and are allocated once, zeroed (an entry is used only once its `full` word
//! is set), so the memory is bounded whatever the block does. A collision of the
//! slot function only costs a miss.

#[cfg(target_arch = "riscv64")]
use std::cell::UnsafeCell;

/// Keccak-256 of the empty input.
pub(crate) const EMPTY_DIGEST: [u8; 32] = [
    0xc5, 0xd2, 0x46, 0x01, 0x86, 0xf7, 0x23, 0x3c, 0x92, 0x7e, 0x7d, 0xb2, 0xdc, 0xc7, 0x03, 0xc0,
    0xe5, 0x00, 0xb6, 0x53, 0xca, 0x82, 0x27, 0x3b, 0x7b, 0xfa, 0xd8, 0x04, 0x5d, 0x85, 0xa4, 0x70,
];

/// log2 of the entries in each table.
const SLOT_BITS: u32 = 12;

#[derive(Clone, Copy)]
#[repr(C)]
struct Entry<const W: usize> {
    /// The input as little-endian doublewords, the last one zero-padded.
    key: [u64; W],
    digest: [u8; 32],
    /// 1 once `key` and `digest` hold a pair, 0 before.
    full: u64,
}

/// A direct-mapped table for inputs of exactly `N` bytes, held in `W` doublewords.
struct Table<const N: usize, const W: usize> {
    entries: Box<[Entry<W>]>,
}

impl<const N: usize, const W: usize> Table<N, W> {
    fn new() -> Self {
        let entries = Box::<[Entry<W>]>::new_zeroed_slice(1 << SLOT_BITS);
        // SAFETY: every field of `Entry` is an integer or an integer array, for which
        // all-zero bytes are a valid value.
        Self {
            entries: unsafe { entries.assume_init() },
        }
    }

    /// `input` (`input.len() == N`) as little-endian doublewords, the last one
    /// zero-padded.
    #[inline(always)]
    fn key(input: &[u8]) -> [u64; W] {
        let mut key = [0u64; W];
        for (i, word) in key.iter_mut().enumerate() {
            let mut bytes = [0u8; 8];
            let end = N.min(8 * i + 8);
            bytes[..end - 8 * i].copy_from_slice(&input[8 * i..end]);
            *word = u64::from_le_bytes(bytes);
        }
        key
    }

    /// The digest of `input` (`input.len() == N`), from the table or from `hash`.
    ///
    /// The hit path is inlined into the provider; the miss path is a call of its own,
    /// so that a hit pays for no register saves the sponge needs.
    #[inline(always)]
    fn digest(&mut self, input: &[u8], hash: impl FnOnce(&[u8]) -> [u8; 32]) -> [u8; 32] {
        let key = Self::key(input);
        let mut fold = 0u64;
        for word in key {
            fold ^= word;
        }
        let slot = (fold.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> (64 - SLOT_BITS)) as usize;
        let entry = &mut self.entries[slot];
        // Word by word into one register, rather than an array compare that goes
        // through memory: the hit path then needs no stack frame.
        let mut diff = entry.full ^ 1;
        for (stored, word) in entry.key.iter().zip(key) {
            diff |= stored ^ word;
        }
        if diff == 0 {
            return entry.digest;
        }
        Self::fill(entry, input, hash)
    }

    /// Hash `input` and make it the entry's pair.
    #[inline(never)]
    fn fill(entry: &mut Entry<W>, input: &[u8], hash: impl FnOnce(&[u8]) -> [u8; 32]) -> [u8; 32] {
        let digest = hash(input);
        entry.key = Self::key(input);
        entry.digest = digest;
        entry.full = 1;
        digest
    }
}

/// The cache: one table per cached length.
pub(crate) struct KeccakCache {
    t20: Table<20, 3>,
    t32: Table<32, 4>,
    t64: Table<64, 8>,
}

impl KeccakCache {
    pub(crate) fn new() -> Self {
        Self {
            t20: Table::new(),
            t32: Table::new(),
            t64: Table::new(),
        }
    }

    /// Keccak-256 of `input`: `hash(input)`, which must be Keccak-256, or the digest it
    /// returned earlier for the same input.
    #[inline(always)]
    pub(crate) fn keccak256(
        &mut self,
        input: &[u8],
        hash: impl FnOnce(&[u8]) -> [u8; 32],
    ) -> [u8; 32] {
        match input.len() {
            0 => EMPTY_DIGEST,
            20 => self.t20.digest(input, hash),
            32 => self.t32.digest(input, hash),
            64 => self.t64.digest(input, hash),
            _ => hash(input),
        }
    }
}

/// The guest's one cache, created on first use.
#[cfg(target_arch = "riscv64")]
///
/// The guest is a single hart with no threads (the target is `singlethread`) and
/// nothing here re-enters: `hash` is the sponge, which never calls back into the
/// provider. So at most one `&mut` to the cell's contents exists at a time.
pub(crate) struct GuestCache(UnsafeCell<Option<KeccakCache>>);

// SAFETY: see the type's documentation; the guest never shares it between threads.
#[cfg(target_arch = "riscv64")]
unsafe impl Sync for GuestCache {}

#[cfg(target_arch = "riscv64")]
impl GuestCache {
    pub(crate) const fn new() -> Self {
        Self(UnsafeCell::new(None))
    }

    #[inline(always)]
    pub(crate) fn keccak256(&self, input: &[u8], hash: impl FnOnce(&[u8]) -> [u8; 32]) -> [u8; 32] {
        // SAFETY: single hart, no re-entry (see the type's documentation).
        let cell = unsafe { &mut *self.0.get() };
        let cache = match cell {
            Some(cache) => cache,
            None => Self::init(cell),
        };
        cache.keccak256(input, hash)
    }

    #[cold]
    #[inline(never)]
    fn init(cell: &mut Option<KeccakCache>) -> &mut KeccakCache {
        cell.insert(KeccakCache::new())
    }
}
