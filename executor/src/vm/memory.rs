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
/// Maximum size of the private input memory region (in bytes).
pub const MAX_PRIVATE_INPUT_SIZE: u64 = 6700000;
/// Fixed high address where private input is mapped. Guest programs can read
/// directly from this address (ZisK-style memory-mapped input).
/// Layout: 4-byte LE length prefix at `PRIVATE_INPUT_START_INDEX`, then data at +4.
/// Must match `PRIVATE_INPUT_START` in `syscalls/src/syscalls.rs`.
pub const PRIVATE_INPUT_START_INDEX: u64 = 0xFF000000;

#[derive(Default, Debug)]
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

    pub fn load_word(&self, address: u64) -> Result<u32, MemoryError> {
        if !address.is_multiple_of(4) {
            return Err(MemoryError::UnalignedAccess);
        }
        let bytes = self.cells.get(&address).cloned().unwrap_or_default();
        Ok(u32::from_le_bytes(bytes))
    }

    pub fn store_word(&mut self, address: u64, value: u32) -> Result<(), MemoryError> {
        if !address.is_multiple_of(4) {
            return Err(MemoryError::UnalignedAccess);
        }
        let bytes = value.to_le_bytes();
        self.cells.insert(address, bytes);
        Ok(())
    }

    /// Load a doubleword (64-bit) from memory - for LD instruction
    pub fn load_doubleword(&self, address: u64) -> Result<u64, MemoryError> {
        if !address.is_multiple_of(8) {
            return Err(MemoryError::UnalignedAccess);
        }
        let low_bytes = self.cells.get(&address).cloned().unwrap_or_default();
        let high_bytes = self.cells.get(&(address + 4)).cloned().unwrap_or_default();
        let low = u32::from_le_bytes(low_bytes) as u64;
        let high = u32::from_le_bytes(high_bytes) as u64;
        Ok(low | (high << 32))
    }

    /// Store a doubleword (64-bit) to memory - for SD instruction
    pub fn store_doubleword(&mut self, address: u64, value: u64) -> Result<(), MemoryError> {
        if !address.is_multiple_of(8) {
            return Err(MemoryError::UnalignedAccess);
        }
        let low = (value & 0xFFFFFFFF) as u32;
        let high = (value >> 32) as u32;
        self.cells.insert(address, low.to_le_bytes());
        self.cells.insert(address + 4, high.to_le_bytes());
        Ok(())
    }

    pub fn load_half(&self, address: u64) -> Result<u16, MemoryError> {
        if !address.is_multiple_of(2) {
            unimplemented!(
                "Unaligned load half memory access at address 0x{:016x}",
                address
            );
        }
        let aligned_address = address - address % 4;
        let bytes = self
            .cells
            .get(&aligned_address)
            .cloned()
            .unwrap_or_default();
        let value = &bytes[(address % 4) as usize..(address % 4) as usize + 2];
        Ok(u16::from_le_bytes(
            value.try_into().map_err(|_| MemoryError::LoadHalf)?,
        ))
    }

    pub fn store_half(&mut self, address: u64, value: u16) -> Result<(), MemoryError> {
        if !address.is_multiple_of(2) {
            return Err(MemoryError::UnalignedAccess);
        }
        let aligned_address = address - address % 4;
        let entry = self
            .cells
            .entry(aligned_address)
            .or_insert_with(|| [0, 0, 0, 0]);
        let bytes = value.to_le_bytes();
        entry[(address % 4) as usize] = bytes[0];
        entry[(address % 4) as usize + 1] = bytes[1];
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

    /// Pre-loads private input bytes at `PRIVATE_INPUT_START_INDEX` as a
    /// 4-byte LE length prefix followed by the raw data. The guest reads these
    /// bytes directly via normal RISC-V loads (ZisK-style memory-mapped input).
    pub fn store_private_inputs(&mut self, inputs: Vec<u8>) -> Result<(), MemoryError> {
        if inputs.is_empty() {
            return Ok(());
        }
        if inputs.len() as u64 > MAX_PRIVATE_INPUT_SIZE {
            return Err(MemoryError::PrivateInputSizeExceeded);
        }
        let len_u32 =
            u32::try_from(inputs.len()).map_err(|_| MemoryError::PrivateInputSizeExceeded)?;
        self.store_word(PRIVATE_INPUT_START_INDEX, len_u32)?;
        self.set_bytes_aligned(PRIVATE_INPUT_START_INDEX + 4, &inputs)?;
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
    fn set_bytes_aligned(&mut self, mut addr: u64, inputs: &[u8]) -> Result<(), MemoryError> {
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
    #[error("Failed to convert bytes to u16")]
    LoadHalf,
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
    use super::Memory;

    #[test]
    fn test_commit_public_output_single() {
        let mut memory = Memory::default();
        memory.store_byte(0x100, b'a');
        memory.store_byte(0x101, b'b');

        memory
            .commit_public_output(0x100, 2)
            .expect("commit should succeed");

        assert_eq!(
            memory
                .read_return_value()
                .expect("public output should be readable"),
            b"ab".to_vec()
        );
    }

    #[test]
    fn test_commit_public_output_appends() {
        let mut memory = Memory::default();
        memory.store_byte(0x100, b'a');
        memory.store_byte(0x101, b'b');
        memory.store_byte(0x104, b'c');
        memory.store_byte(0x105, b'd');

        memory
            .commit_public_output(0x100, 2)
            .expect("first commit should succeed");
        memory
            .commit_public_output(0x104, 2)
            .expect("second commit should succeed");

        // Append semantics: calls concatenate (EF zkVM IO interface).
        assert_eq!(
            memory
                .read_return_value()
                .expect("public output should be readable"),
            b"abcd".to_vec()
        );
    }

    #[test]
    fn test_commit_public_output_empty_is_ok() {
        let mut memory = Memory::default();
        memory
            .commit_public_output(0, 0)
            .expect("zero-length commit should succeed");
        assert!(
            memory
                .read_return_value()
                .expect("public output should be readable")
                .is_empty()
        );
    }

    #[test]
    fn test_commit_public_output_address_overflow() {
        let mut memory = Memory::default();
        let err = memory
            .commit_public_output(u64::MAX, 2)
            .expect_err("address overflow must error, not panic");
        assert!(matches!(err, super::MemoryError::AddressOverflow));
    }

    #[test]
    fn test_load_bytes_huge_len_returns_alloc_error() {
        let memory = Memory::default();
        // A multi-petabyte allocation request from a guest must fail cleanly,
        // not abort the host process via OOM. `addr=0` and `len=1<<50` keep
        // `checked_add` happy so the path reaches the allocation.
        let huge = 1u64 << 50;
        let err = memory
            .load_bytes(0, huge)
            .expect_err("huge alloc must error, not abort");
        assert!(matches!(err, super::MemoryError::AllocationFailed));
    }

    #[test]
    fn test_load_bytes_overflow_errors() {
        let memory = Memory::default();
        let err = memory
            .load_bytes(u64::MAX, 2)
            .expect_err("address overflow must error, not panic");
        assert!(matches!(err, super::MemoryError::AddressOverflow));
    }

    #[test]
    fn test_commit_public_output_total_cap() {
        let mut memory = Memory::default();
        // Seed enough source bytes for two 512 KB writes.
        let chunk = vec![0xAB; 512 * 1024];
        memory
            .set_bytes_aligned(0x1_0000, &chunk)
            .expect("seed should succeed");

        memory
            .commit_public_output(0x1_0000, 512 * 1024)
            .expect("first 512 KB commit should succeed");
        memory
            .commit_public_output(0x1_0000, 512 * 1024)
            .expect("second 512 KB commit should succeed (total = 1 MB)");

        // One more byte exceeds the 1 MB total cap.
        let err = memory.commit_public_output(0x1_0000, 1).unwrap_err();
        assert!(matches!(err, super::MemoryError::CommitSizeExceeded));
    }

    #[test]
    fn test_commit_zero_length_after_data() {
        let mut memory = Memory::default();
        memory.store_byte(0x100, b'x');
        memory
            .commit_public_output(0x100, 1)
            .expect("first commit should succeed");
        // Zero-length commit must be a no-op.
        memory
            .commit_public_output(0x100, 0)
            .expect("zero-length commit should succeed");
        assert_eq!(
            memory
                .read_return_value()
                .expect("public output should be readable"),
            b"x".to_vec()
        );
    }

    #[test]
    fn test_commit_zero_length_at_cap() {
        let mut memory = Memory::default();
        let chunk = vec![0xAB; 512 * 1024];
        memory
            .set_bytes_aligned(0x1_0000, &chunk)
            .expect("seed should succeed");
        // Fill to exactly 1 MB.
        memory
            .commit_public_output(0x1_0000, 512 * 1024)
            .expect("first half should succeed");
        memory
            .commit_public_output(0x1_0000, 512 * 1024)
            .expect("second half should succeed");
        // Zero-length commit at the cap boundary must still succeed.
        memory
            .commit_public_output(0x1_0000, 0)
            .expect("zero-length commit at cap should succeed");
    }

    #[test]
    fn test_commit_cap_exceeded_does_not_modify_output() {
        let mut memory = Memory::default();
        memory.store_byte(0x100, b'a');
        memory.store_byte(0x101, b'b');
        memory
            .commit_public_output(0x100, 2)
            .expect("initial commit should succeed");

        let before = memory
            .read_return_value()
            .expect("public output should be readable");

        // Attempt a commit that exceeds the cap.
        let _ = memory.commit_public_output(0x100, super::MAX_PUBLIC_OUTPUT_TOTAL_SIZE);

        let after = memory
            .read_return_value()
            .expect("public output should be readable");

        // The failed commit must not have modified the output buffer.
        assert_eq!(before, after);
    }
}
