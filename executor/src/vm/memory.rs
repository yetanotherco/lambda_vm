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

// TODO: Correctly define this
const MAX_PUBLIC_OUTPUT_COMMIT_SIZE: u64 = 1024;
const PUBLIC_OUTPUT_START_INDEX: u64 = 0;
// Ported from main: increased from 1024 to support larger inputs (ethrex)
const MAX_PRIVATE_INPUT_SIZE: u64 = 6700000;
// Ported from main: fixed high address to avoid overlap with program memory
const PRIVATE_INPUT_START_INDEX: u64 = 0xFF000000;

/// Pre-allocated capacity for the memory HashMap to avoid resizing during execution.
/// This is sized to handle typical program memory usage without reallocation.
const INITIAL_MEMORY_CAPACITY: usize = 64 * 1024; // 64K entries = 256KB of addressable memory

#[derive(Debug)]
pub struct Memory(U64HashMap<[u8; 4]>);

impl Default for Memory {
    fn default() -> Self {
        Self::new()
    }
}

impl Memory {
    /// Create a new Memory with pre-allocated capacity
    pub fn new() -> Self {
        Self(U64HashMap::with_capacity_and_hasher(
            INITIAL_MEMORY_CAPACITY,
            U64BuildHasher,
        ))
    }

    pub fn load_byte(&self, address: u64) -> u8 {
        let aligned_address = address - address % 4;
        let value = self.0.get(&aligned_address).cloned().unwrap_or_default();
        value[(address % 4) as usize]
    }

    pub fn store_byte(&mut self, address: u64, value: u8) {
        let aligned_address = address - address % 4;
        let entry = self
            .0
            .entry(aligned_address)
            .or_insert_with(|| [0, 0, 0, 0]);
        entry[(address % 4) as usize] = value;
    }

    pub fn load_word(&self, address: u64) -> Result<u32, MemoryError> {
        if !address.is_multiple_of(4) {
            return Err(MemoryError::UnalignedAccess);
        }
        let bytes = self.0.get(&address).cloned().unwrap_or_default();
        Ok(u32::from_le_bytes(bytes))
    }

    pub fn store_word(&mut self, address: u64, value: u32) -> Result<(), MemoryError> {
        if !address.is_multiple_of(4) {
            return Err(MemoryError::UnalignedAccess);
        }
        let bytes = value.to_le_bytes();
        self.0.insert(address, bytes);
        Ok(())
    }

    /// Load a doubleword (64-bit) from memory - for LD instruction
    pub fn load_doubleword(&self, address: u64) -> Result<u64, MemoryError> {
        if !address.is_multiple_of(8) {
            return Err(MemoryError::UnalignedAccess);
        }
        let low_bytes = self.0.get(&address).cloned().unwrap_or_default();
        let high_bytes = self.0.get(&(address + 4)).cloned().unwrap_or_default();
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
        self.0.insert(address, low.to_le_bytes());
        self.0.insert(address + 4, high.to_le_bytes());
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
        let bytes = self.0.get(&aligned_address).cloned().unwrap_or_default();
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
            .0
            .entry(aligned_address)
            .or_insert_with(|| [0, 0, 0, 0]);
        let bytes = value.to_le_bytes();
        entry[(address % 4) as usize] = bytes[0];
        entry[(address % 4) as usize + 1] = bytes[1];
        Ok(())
    }

    pub fn commit_public_output(&mut self, address: u64, length: u64) -> Result<(), MemoryError> {
        if length > MAX_PUBLIC_OUTPUT_COMMIT_SIZE {
            return Err(MemoryError::CommitSizeExceeded);
        }
        self.store_word(PUBLIC_OUTPUT_START_INDEX, length as u32)?;
        let inputs = self.load_bytes(address, length);
        self.set_bytes_aligned(PUBLIC_OUTPUT_START_INDEX + 4, &inputs)?;
        Ok(())
    }

    pub fn read_return_value(&self) -> Result<Vec<u8>, MemoryError> {
        let size = self.load_word(PUBLIC_OUTPUT_START_INDEX)?;
        Ok(self.load_bytes(PUBLIC_OUTPUT_START_INDEX + 4, size as u64))
    }

    pub fn store_private_inputs(&mut self, inputs: Vec<u8>) -> Result<(), MemoryError> {
        if inputs.len() as u64 > MAX_PRIVATE_INPUT_SIZE {
            return Err(MemoryError::PrivateInputSizeExceeded);
        }
        self.store_word(PRIVATE_INPUT_START_INDEX, inputs.len() as u32)?;
        self.set_bytes_aligned(PRIVATE_INPUT_START_INDEX + 4, &inputs)?;
        Ok(())
    }

    pub fn load_private_inputs(&self) -> Result<Vec<u8>, MemoryError> {
        let size = self.load_word(PRIVATE_INPUT_START_INDEX)?;
        let mut inputs = size.to_le_bytes().to_vec();
        inputs.extend_from_slice(&self.load_bytes(PRIVATE_INPUT_START_INDEX + 4, size as u64));
        Ok(inputs)
    }

    pub fn load_bytes(&self, mut addr: u64, len: u64) -> Vec<u8> {
        let mut result = Vec::with_capacity(len as usize);
        let end = addr + len;
        while addr < end {
            let aligned = addr - (addr % 4);
            let bytes = self.0.get(&aligned).cloned().unwrap_or_default();
            let offset = (addr % 4) as usize;
            let take = std::cmp::min(4 - offset, (end - addr) as usize);
            result.extend_from_slice(&bytes[offset..offset + take]);
            addr += take as u64;
        }
        result
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
            self.0.insert(addr, bytes);
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
}
