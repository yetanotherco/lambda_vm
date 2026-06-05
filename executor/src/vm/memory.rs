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
