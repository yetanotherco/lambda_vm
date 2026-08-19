use std::collections::HashMap;
use std::hash::{BuildHasher, Hasher};

/// Fast hasher for u64 keys - uses the key directly as the hash value.
/// This avoids the overhead of SipHash for integer keys.
#[derive(Default)]
pub struct U64Hasher(u64);

impl Hasher for U64Hasher {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = self.0.wrapping_shl(8).wrapping_add(b as u64);
        }
    }

    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.0 = i;
    }

    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }
}

#[derive(Default, Clone)]
pub struct U64BuildHasher;

impl BuildHasher for U64BuildHasher {
    type Hasher = U64Hasher;
    #[inline]
    fn build_hasher(&self) -> U64Hasher {
        U64Hasher(0)
    }
}

pub type U64HashMap<V> = HashMap<u64, V, U64BuildHasher>;

/// Total cap on public output bytes across all `commit_public_output` calls.
/// The COMMIT AIR concatenates calls via the running `x254` index, so this
/// is enforced as a running-total budget rather than a per-call limit.
pub const MAX_PUBLIC_OUTPUT_TOTAL_SIZE: u64 = 1024 * 1024;
/// Maximum size of the private input memory region (in bytes). 512 MiB so a
/// real proof (e.g. a continuation bundle) fits as private input.
pub const MAX_PRIVATE_INPUT_SIZE: u64 = 512 * 1024 * 1024;
/// Fixed high address where private input is mapped. Guest programs can read
/// directly from this address (ZisK-style memory-mapped input).
/// Layout: `[u32 LE main_len][main data][zero-pad to 8][u32 LE hint_count][u32 zero pad]`
/// then `hint_count` 32-byte hint slots (see [`encode_private_input_region`]).
/// Must match `PRIVATE_INPUT_START` in `syscalls/src/syscalls.rs`.
pub const PRIVATE_INPUT_START_INDEX: u64 = 0xFF000000;
/// Size in bytes of the private input's wire-format length prefix (the `u32` LE
/// written at `PRIVATE_INPUT_START_INDEX` by [`Memory::store_private_inputs`]; the
/// data follows at `+ PRIVATE_INPUT_LENGTH_PREFIX_BYTES`). Single source of truth
/// for every page-span computation over the private-input region.
pub const PRIVATE_INPUT_LENGTH_PREFIX_BYTES: usize = size_of::<u32>();
/// Size in bytes of one hint slot in the hint arena.
pub const HINT_SLOT_BYTES: u64 = 32;
/// Size in bytes of the hint-arena header (`[u32 LE hint_count][u32 zero pad]`),
/// always present once the region is written at all.
pub const HINT_ARENA_HEADER_BYTES: u64 = 8;

/// Start of the hint request log: just past the reserved private-input window,
/// rounded up to 8 (`0xFF000000 + 4 + 512 MiB + 4` = `0x11F000008`). The guest
/// appends `(hint_id, input)` entries here when the hint arena is exhausted —
/// the recording pass of the two-pass hint flow. The host reads them back with
/// [`Memory::hint_requests`]. Must match `HINT_LOG_START` in
/// `syscalls/src/syscalls.rs`.
pub const HINT_LOG_START_INDEX: u64 =
    0xFF000000 + PRIVATE_INPUT_LENGTH_PREFIX_BYTES as u64 + MAX_PRIVATE_INPUT_SIZE + 4;
/// Size in bytes of the request-log header (`[u32 LE count][u32 zero pad]`).
pub const HINT_LOG_HEADER_BYTES: u64 = 8;
/// Size in bytes of one request-log entry (`[u64 LE hint_id][32-byte input]`).
pub const HINT_LOG_ENTRY_BYTES: u64 = 40;

/// Byte offset from `PRIVATE_INPUT_START_INDEX` at which the hint-arena header
/// (count word) sits, for a given main-input length: the 4-byte length prefix
/// plus the main data, padded with zeros up to the next 8-byte boundary.
pub const fn hint_arena_header_offset(main_len: u64) -> u64 {
    (PRIVATE_INPUT_LENGTH_PREFIX_BYTES as u64 + main_len + 7) & !7
}

/// Canonical encoder for the private-input region:
/// `[len][data][pad8][count][pad][slots]`. Single source of truth for the wire
/// format; the prover's trace builder must call this instead of re-encoding.
///
/// The count word and its pad are ALWAYS written (even when `hints` is empty),
/// so the layout is uniform and old guests reading past their data see a zero
/// count. The whole region must fit the reserved window:
/// `align8(4 + main_len) + 8 + 32 * hint_count <= 4 + MAX_PRIVATE_INPUT_SIZE`,
/// which keeps the verifier's `max_private_input_pages()` bound valid.
pub fn encode_private_input_region(
    inputs: &[u8],
    hints: &[[u8; 32]],
) -> Result<Vec<u8>, MemoryError> {
    let main_len =
        u32::try_from(inputs.len()).map_err(|_| MemoryError::PrivateInputSizeExceeded)?;
    let header_offset = hint_arena_header_offset(inputs.len() as u64);
    let hints_bytes = (hints.len() as u64)
        .checked_mul(HINT_SLOT_BYTES)
        .ok_or(MemoryError::PrivateInputSizeExceeded)?;
    let total = header_offset
        .checked_add(HINT_ARENA_HEADER_BYTES)
        .and_then(|t| t.checked_add(hints_bytes))
        .ok_or(MemoryError::PrivateInputSizeExceeded)?;
    if total > PRIVATE_INPUT_LENGTH_PREFIX_BYTES as u64 + MAX_PRIVATE_INPUT_SIZE {
        return Err(MemoryError::PrivateInputSizeExceeded);
    }

    let mut region = Vec::with_capacity(total as usize);
    region.extend_from_slice(&main_len.to_le_bytes());
    region.extend_from_slice(inputs);
    region.resize(header_offset as usize, 0);
    region.extend_from_slice(&(hints.len() as u32).to_le_bytes());
    region.extend_from_slice(&[0u8; 4]);
    for hint in hints {
        region.extend_from_slice(hint);
    }
    debug_assert_eq!(region.len() as u64, total);
    Ok(region)
}

#[derive(Default, Debug, Clone)]
pub struct Memory {
    cells: U64HashMap<[u8; 4]>,
    /// Bytes committed to public output via `commit_public_output`. The
    /// COMMIT AIR doesn't write to a fixed memory region (it streams bytes
    /// onto the Commit bus by `index`), so this buffer is purely the
    /// executor's view used by `read_return_value` and CLI display.
    public_output: Vec<u8>,
}

impl Memory {
    pub fn load_byte(&self, address: u64) -> u8 {
        let aligned_address = address - address % 4;
        let value = self
            .cells
            .get(&aligned_address)
            .cloned()
            .unwrap_or_default();
        value[(address % 4) as usize]
    }

    pub fn store_byte(&mut self, address: u64, value: u8) {
        let aligned_address = address - address % 4;
        let entry = self
            .cells
            .entry(aligned_address)
            .or_insert_with(|| [0, 0, 0, 0]);
        entry[(address % 4) as usize] = value;
    }

    /// Iterate over all stored bytes as `(address, value)` pairs. Cells are
    /// stored as 4-byte words; each word expands into its four byte addresses.
    /// Used to snapshot memory at an epoch boundary.
    pub fn iter_bytes(&self) -> impl Iterator<Item = (u64, u8)> + '_ {
        self.cells.iter().flat_map(|(&addr, bytes)| {
            bytes
                .iter()
                .enumerate()
                .map(move |(i, &b)| (addr + i as u64, b))
        })
    }

    pub fn load_word(&self, address: u64) -> Result<u32, MemoryError> {
        if address.is_multiple_of(4) {
            let bytes = self.cells.get(&address).cloned().unwrap_or_default();
            Ok(u32::from_le_bytes(bytes))
        } else {
            address.checked_add(3).ok_or(MemoryError::AddressOverflow)?;
            Ok(u32::from_le_bytes([
                self.load_byte(address),
                self.load_byte(address + 1),
                self.load_byte(address + 2),
                self.load_byte(address + 3),
            ]))
        }
    }

    pub fn store_word(&mut self, address: u64, value: u32) -> Result<(), MemoryError> {
        let bytes = value.to_le_bytes();
        if address.is_multiple_of(4) {
            self.cells.insert(address, bytes);
        } else {
            address.checked_add(3).ok_or(MemoryError::AddressOverflow)?;
            for (i, b) in bytes.iter().enumerate() {
                self.store_byte(address + i as u64, *b);
            }
        }
        Ok(())
    }

    /// Load a doubleword (64-bit) from memory - for LD instruction
    pub fn load_doubleword(&self, address: u64) -> Result<u64, MemoryError> {
        if address.is_multiple_of(8) {
            // 8-alignment bounds `address` to `u64::MAX - 7`, so `address + 4` can't overflow.
            let low_bytes = self.cells.get(&address).cloned().unwrap_or_default();
            let high_bytes = self.cells.get(&(address + 4)).cloned().unwrap_or_default();
            let low = u32::from_le_bytes(low_bytes) as u64;
            let high = u32::from_le_bytes(high_bytes) as u64;
            Ok(low | (high << 32))
        } else {
            address.checked_add(7).ok_or(MemoryError::AddressOverflow)?;
            let mut bytes = [0u8; 8];
            for (i, b) in bytes.iter_mut().enumerate() {
                *b = self.load_byte(address + i as u64);
            }
            Ok(u64::from_le_bytes(bytes))
        }
    }

    /// Store a doubleword (64-bit) to memory - for SD instruction
    pub fn store_doubleword(&mut self, address: u64, value: u64) -> Result<(), MemoryError> {
        if address.is_multiple_of(8) {
            let low = (value & 0xFFFFFFFF) as u32;
            let high = (value >> 32) as u32;
            // 8-alignment bounds `address` to `u64::MAX - 7`, so `address + 4` can't overflow.
            self.cells.insert(address, low.to_le_bytes());
            self.cells.insert(address + 4, high.to_le_bytes());
        } else {
            address.checked_add(7).ok_or(MemoryError::AddressOverflow)?;
            let bytes = value.to_le_bytes();
            for (i, b) in bytes.iter().enumerate() {
                self.store_byte(address + i as u64, *b);
            }
        }
        Ok(())
    }

    pub fn load_half(&self, address: u64) -> Result<u16, MemoryError> {
        if address.is_multiple_of(2) {
            let aligned_address = address - address % 4;
            let bytes = self
                .cells
                .get(&aligned_address)
                .cloned()
                .unwrap_or_default();
            let offset = (address % 4) as usize;
            Ok(u16::from_le_bytes([bytes[offset], bytes[offset + 1]]))
        } else {
            address.checked_add(1).ok_or(MemoryError::AddressOverflow)?;
            Ok(u16::from_le_bytes([
                self.load_byte(address),
                self.load_byte(address + 1),
            ]))
        }
    }

    pub fn store_half(&mut self, address: u64, value: u16) -> Result<(), MemoryError> {
        let bytes = value.to_le_bytes();
        if address.is_multiple_of(2) {
            let aligned_address = address - address % 4;
            let entry = self
                .cells
                .entry(aligned_address)
                .or_insert_with(|| [0, 0, 0, 0]);
            let offset = (address % 4) as usize;
            entry[offset] = bytes[0];
            entry[offset + 1] = bytes[1];
        } else {
            address.checked_add(1).ok_or(MemoryError::AddressOverflow)?;
            self.store_byte(address, bytes[0]);
            self.store_byte(address + 1, bytes[1]);
        }
        Ok(())
    }

    /// Append `length` bytes from guest memory starting at `address` to the
    /// public output. The COMMIT AIR concatenates calls via the running
    /// `x254` index, and the trace builder accumulates `commit_ops` into
    /// `VmProof.public_output`; this method maintains the executor's view
    /// of the same byte stream so `read_return_value` matches.
    pub fn commit_public_output(&mut self, address: u64, length: u64) -> Result<(), MemoryError> {
        let new_total = (self.public_output.len() as u64)
            .checked_add(length)
            .ok_or(MemoryError::CommitSizeExceeded)?;
        if new_total > MAX_PUBLIC_OUTPUT_TOTAL_SIZE {
            return Err(MemoryError::CommitSizeExceeded);
        }
        let bytes = self.load_bytes(address, length)?;
        self.public_output.extend_from_slice(&bytes);
        Ok(())
    }

    pub fn read_return_value(&self) -> Result<Vec<u8>, MemoryError> {
        Ok(self.public_output.clone())
    }

    /// Read the hint request log the guest appended via `request_hint` (the
    /// recording pass of the two-pass hint flow): `(hint_id, input)` entries in
    /// request order. Empty when the arena covered every request.
    pub fn hint_requests(&self) -> Result<Vec<(u64, [u8; 32])>, MemoryError> {
        let count = self.load_word(HINT_LOG_START_INDEX)? as usize;
        let mut out = Vec::with_capacity(count);
        for i in 0..count {
            let entry =
                HINT_LOG_START_INDEX + HINT_LOG_HEADER_BYTES + i as u64 * HINT_LOG_ENTRY_BYTES;
            let hint_id = self.load_doubleword(entry)?;
            let bytes = self.load_bytes(entry + 8, 32)?;
            let mut input = [0u8; 32];
            input.copy_from_slice(&bytes);
            out.push((hint_id, input));
        }
        Ok(out)
    }

    /// Pre-loads private input bytes at `PRIVATE_INPUT_START_INDEX` in the
    /// canonical wire format ([`encode_private_input_region`]): a 4-byte LE
    /// length prefix, the main data zero-padded to an 8-byte boundary, then the
    /// always-present hint-arena header (`[u32 LE hint_count][u32 zero pad]`)
    /// followed by `hint_count` 32-byte hint slots. The guest reads these bytes
    /// directly via normal RISC-V loads (ZisK-style memory-mapped input).
    ///
    /// With no main input AND no hints nothing is written at all (the region
    /// reads back as all zeros, including a zero hint count) so a no-input
    /// program keeps zero private-input pages.
    pub fn store_private_inputs(
        &mut self,
        inputs: Vec<u8>,
        hints: &[[u8; 32]],
    ) -> Result<(), MemoryError> {
        if inputs.is_empty() && hints.is_empty() {
            return Ok(());
        }
        let region = encode_private_input_region(&inputs, hints)?;
        self.set_bytes_aligned(PRIVATE_INPUT_START_INDEX, &region)?;
        Ok(())
    }

    pub fn load_bytes(&self, mut addr: u64, len: u64) -> Result<Vec<u8>, MemoryError> {
        let end = addr.checked_add(len).ok_or(MemoryError::AddressOverflow)?;
        let len_usize = usize::try_from(len).map_err(|_| MemoryError::AllocationFailed)?;
        let mut result = Vec::new();
        result
            .try_reserve_exact(len_usize)
            .map_err(|_| MemoryError::AllocationFailed)?;
        while addr < end {
            let aligned = addr - (addr % 4);
            let bytes = self.cells.get(&aligned).cloned().unwrap_or_default();
            let offset = (addr % 4) as usize;
            let take = std::cmp::min(4 - offset, (end - addr) as usize);
            result.extend_from_slice(&bytes[offset..offset + take]);
            addr += take as u64;
        }
        Ok(result)
    }

    /// Helper method to store a given input at an aligned address. It may also overwrite existing bytes with zero if inputs is not divisible by 4
    /// Should only be used to write to public output and private input where these limitations are not a problem
    pub(crate) fn set_bytes_aligned(
        &mut self,
        mut addr: u64,
        inputs: &[u8],
    ) -> Result<(), MemoryError> {
        if !addr.is_multiple_of(4) {
            return Err(MemoryError::UnalignedAccess);
        }
        for chunk in inputs.chunks(4) {
            let mut bytes = [0u8; 4];
            bytes[..chunk.len()].copy_from_slice(chunk);
            self.cells.insert(addr, bytes);
            addr += 4;
        }
        Ok(())
    }
}

#[derive(thiserror::Error, Debug)]
pub enum MemoryError {
    #[error("Unaligned memory access")]
    UnalignedAccess,
    #[error("Public output commit size exceeded")]
    CommitSizeExceeded,
    #[error("Private input size exceeded")]
    PrivateInputSizeExceeded,
    #[error("Address range exceeds u64::MAX")]
    AddressOverflow,
    #[error("Failed to allocate memory for load_bytes")]
    AllocationFailed,
}

#[cfg(test)]
mod tests {
    use super::*;

    // The wire-format writer and every private-input page-span computation assume the
    // length prefix is exactly a 4-byte LE `u32`; pin that so a change to the constant
    // is caught rather than silently drifting from the page math.
    #[test]
    fn private_input_length_prefix_is_a_le_u32() {
        assert_eq!(PRIVATE_INPUT_LENGTH_PREFIX_BYTES, 4);
        assert_eq!(PRIVATE_INPUT_LENGTH_PREFIX_BYTES, size_of::<u32>());
    }

    // `store_private_inputs` must write a LE length prefix at the region base and the data
    // immediately after it, at `+ PRIVATE_INPUT_LENGTH_PREFIX_BYTES`.
    #[test]
    fn store_private_inputs_writes_le_length_prefix_then_data() {
        let mut memory = Memory::default();
        let inputs = vec![0xAAu8, 0xBB, 0xCC];
        memory.store_private_inputs(inputs.clone(), &[]).unwrap();

        assert_eq!(
            memory.load_word(PRIVATE_INPUT_START_INDEX).unwrap(),
            inputs.len() as u32
        );
        let data = memory
            .load_bytes(
                PRIVATE_INPUT_START_INDEX + PRIVATE_INPUT_LENGTH_PREFIX_BYTES as u64,
                inputs.len() as u64,
            )
            .unwrap();
        assert_eq!(data, inputs);
    }

    // Roundtrip with hints: the count word sits at `hint_arena_header_offset(main_len)`
    // (8-aligned), the slots land right after the 8-byte header, and the region's
    // extent is `header + 8 + 32 * hint_count`.
    #[test]
    fn store_private_inputs_roundtrip_with_hints() {
        let mut memory = Memory::default();
        let inputs = vec![0x11u8; 5]; // odd length exercises the pad-to-8
        let hints = [[0x22u8; 32], [0x33u8; 32]];
        memory.store_private_inputs(inputs.clone(), &hints).unwrap();

        // Legacy prefix: [len][data].
        assert_eq!(
            memory.load_word(PRIVATE_INPUT_START_INDEX).unwrap(),
            inputs.len() as u32
        );
        let data = memory
            .load_bytes(
                PRIVATE_INPUT_START_INDEX + PRIVATE_INPUT_LENGTH_PREFIX_BYTES as u64,
                inputs.len() as u64,
            )
            .unwrap();
        assert_eq!(data, inputs);

        // Hint-arena header: align8(4 + 5) = 16, count word at base + 16, pad zero.
        let header = hint_arena_header_offset(inputs.len() as u64);
        assert_eq!(header, 16);
        assert!(header.is_multiple_of(8));
        assert_eq!(
            memory
                .load_word(PRIVATE_INPUT_START_INDEX + header)
                .unwrap(),
            hints.len() as u32
        );
        assert_eq!(
            memory
                .load_word(PRIVATE_INPUT_START_INDEX + header + 4)
                .unwrap(),
            0
        );

        // Slots land at header + 8, 32 bytes each, in order.
        for (i, hint) in hints.iter().enumerate() {
            let slot = memory
                .load_bytes(
                    PRIVATE_INPUT_START_INDEX
                        + header
                        + HINT_ARENA_HEADER_BYTES
                        + i as u64 * HINT_SLOT_BYTES,
                    HINT_SLOT_BYTES,
                )
                .unwrap();
            assert_eq!(slot, hint);
        }

        // The encoder agrees with what was stored, byte for byte.
        let encoded = encode_private_input_region(&inputs, &hints).unwrap();
        let stored = memory
            .load_bytes(PRIVATE_INPUT_START_INDEX, encoded.len() as u64)
            .unwrap();
        assert_eq!(stored, encoded);
    }

    // Empty main input with hints: the `[0]` length prefix and the arena header are
    // still written (no early return), and the slots follow the header.
    #[test]
    fn store_private_inputs_empty_main_with_hints() {
        let mut memory = Memory::default();
        let hints = [[0xABu8; 32]];
        memory.store_private_inputs(vec![], &hints).unwrap();

        assert_eq!(memory.load_word(PRIVATE_INPUT_START_INDEX).unwrap(), 0);
        // align8(4 + 0) = 8: the count word is readable at base + 8.
        let header = hint_arena_header_offset(0);
        assert_eq!(header, 8);
        assert_eq!(
            memory
                .load_word(PRIVATE_INPUT_START_INDEX + header)
                .unwrap(),
            1
        );
        let slot = memory
            .load_bytes(
                PRIVATE_INPUT_START_INDEX + header + HINT_ARENA_HEADER_BYTES,
                HINT_SLOT_BYTES,
            )
            .unwrap();
        assert_eq!(slot, hints[0]);
    }

    // Zero hints produce the legacy prefix plus the always-written 8-byte header
    // (count = 0, pad = 0) — and nothing past it.
    #[test]
    fn store_private_inputs_zero_hints_writes_empty_header() {
        let mut memory = Memory::default();
        let inputs = vec![0x42u8; 16];
        memory.store_private_inputs(inputs.clone(), &[]).unwrap();

        let header = hint_arena_header_offset(inputs.len() as u64);
        assert_eq!(header, 24); // align8(4 + 16)
        assert_eq!(
            memory
                .load_word(PRIVATE_INPUT_START_INDEX + header)
                .unwrap(),
            0
        );
        assert_eq!(
            memory
                .load_word(PRIVATE_INPUT_START_INDEX + header + 4)
                .unwrap(),
            0
        );
        let encoded = encode_private_input_region(&inputs, &[]).unwrap();
        assert_eq!(encoded.len() as u64, header + HINT_ARENA_HEADER_BYTES);
    }

    // No main input and no hints: nothing is written (a zero input program keeps
    // zero private-input pages); the count word reads back as 0 regardless.
    #[test]
    fn store_private_inputs_empty_everything_writes_nothing() {
        let mut memory = Memory::default();
        memory.store_private_inputs(vec![], &[]).unwrap();
        assert_eq!(memory.load_word(PRIVATE_INPUT_START_INDEX).unwrap(), 0);
        assert_eq!(
            memory
                .load_word(PRIVATE_INPUT_START_INDEX + hint_arena_header_offset(0))
                .unwrap(),
            0
        );
    }

    // Size cap: `align8(4 + main_len) + 8 + 32 * hint_count` must stay within
    // `4 + MAX_PRIVATE_INPUT_SIZE`. A main input exactly at the old cap leaves no
    // room for the arena header, and even one hint on top must fail.
    #[test]
    fn encode_private_input_region_enforces_size_cap() {
        // Rejection cases first: the cap is checked before the region is
        // allocated, so these never touch the 512 MiB output buffer.
        //
        // The old single-section cap (main_len > MAX) still fails.
        let err = encode_private_input_region(&vec![0u8; MAX_PRIVATE_INPUT_SIZE as usize + 1], &[])
            .unwrap_err();
        assert!(matches!(err, MemoryError::PrivateInputSizeExceeded));

        // Largest main_len whose region (with the header, zero hints) still fits:
        // align8(4 + main_len) + 8 <= 4 + MAX. With MAX a multiple of 8, that is
        // 4 + main_len <= MAX - 8, i.e. main_len = MAX - 12.
        let ok_len = MAX_PRIVATE_INPUT_SIZE as usize - 12;
        let inputs = vec![0u8; ok_len];
        // The same main_len plus a single hint exceeds the reserved window.
        let err = encode_private_input_region(&inputs, &[[0u8; 32]]).unwrap_err();
        assert!(matches!(err, MemoryError::PrivateInputSizeExceeded));

        let encoded = encode_private_input_region(&inputs, &[]).unwrap();
        assert_eq!(encoded.len() as u64, MAX_PRIVATE_INPUT_SIZE);
    }
}
