use std::collections::BTreeMap;

// TODO: Correctly define this
const MAX_PUBLIC_OUTPUT_COMMIT_SIZE: u32 = 1024;
const PUBLIC_OUTPUT_START_INDEX: u32 = 0;

#[derive(Default, Debug)]
pub struct Memory(BTreeMap<u32, [u8; 4]>);

impl Memory {
    pub fn load_byte(&self, address: u32) -> u8 {
        let aligned_address = address - address % 4;
        let value = self.0.get(&aligned_address).cloned().unwrap_or_default();
        value[(address % 4) as usize]
    }
    pub fn store_byte(&mut self, address: u32, value: u8) {
        let aligned_address = address - address % 4;
        let entry = self
            .0
            .entry(aligned_address)
            .or_insert_with(|| [0, 0, 0, 0]);
        entry[(address % 4) as usize] = value;
    }
    pub fn load_word(&self, address: u32) -> Result<u32, MemoryError> {
        if !address.is_multiple_of(4) {
            return Err(MemoryError::UnalignedAccess);
        }
        let bytes = self.0.get(&address).cloned().unwrap_or_default();
        Ok(u32::from_le_bytes(bytes))
    }
    pub fn store_word(&mut self, address: u32, value: u32) -> Result<(), MemoryError> {
        if !address.is_multiple_of(4) {
            return Err(MemoryError::UnalignedAccess);
        }
        let bytes = value.to_le_bytes();
        self.0.insert(address, bytes);
        Ok(())
    }
    pub fn load_half(&self, address: u32) -> Result<u16, MemoryError> {
        if !address.is_multiple_of(2) {
            unimplemented!(
                "Unaligned load half memory access at address 0x{:08x}",
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
    pub fn store_half(&mut self, address: u32, value: u16) -> Result<(), MemoryError> {
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

    pub fn commit_public_output(&mut self, address: u32, length: u32) -> Result<(), MemoryError> {
        if length > MAX_PUBLIC_OUTPUT_COMMIT_SIZE {
            return Err(MemoryError::CommitSizeExceeded);
        }
        self.store_word(PUBLIC_OUTPUT_START_INDEX, length)?;
        for i in 0..length {
            let byte = self.load_byte(address + i);
            self.store_byte(PUBLIC_OUTPUT_START_INDEX + 4 + i, byte);
        }
        Ok(())
    }

    pub fn read_return_value(&self) -> Result<Vec<u8>, MemoryError> {
        let size = self.load_word(PUBLIC_OUTPUT_START_INDEX)?;
        let mut return_values = Vec::new();
        for i in 0..size {
            let word = self.load_byte(PUBLIC_OUTPUT_START_INDEX + 4 + i);
            return_values.push(word);
        }
        Ok(return_values)
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
}
