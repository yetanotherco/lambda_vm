use std::collections::BTreeMap;

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
}

#[derive(thiserror::Error, Debug)]
pub enum MemoryError {
    #[error("Failed to convert bytes to u16")]
    LoadHalf,
    #[error("Unaligned memory access")]
    UnalignedAccess,
}
