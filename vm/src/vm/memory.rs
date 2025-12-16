#[derive(Default, Debug)]
pub struct Memory(Vec<u8>);

impl Memory {
    pub fn load_byte(&self, address: u32) -> u8 {
        let byte_index = address as usize;
        if byte_index >= self.0.len() {
            return 0;
        }
        self.0[byte_index]
    }
    pub fn store_byte(&mut self, address: u32, value: u8) {
        let byte_index = address as usize;
        self.expand_memory(byte_index + 1);
        self.0[byte_index] = value;
    }
    pub fn load_word(&self, address: u32) -> u32 {
        let byte_index = address as usize;
        if byte_index + 4 > self.0.len() {
            return 0;
        }
        let bytes = &self.0[byte_index..byte_index + 4];
        u32::from_be_bytes(bytes.try_into().unwrap())
    }
    pub fn store_word(&mut self, address: u32, value: u32) {
        let byte_index = address as usize;
        self.expand_memory(byte_index + 4);
        let bytes = value.to_be_bytes();
        self.0[byte_index..byte_index + 4].copy_from_slice(&bytes);
    }
    pub fn load_half(&self, address: u32) -> u16 {
        let byte_index = address as usize;
        if byte_index + 2 > self.0.len() {
            return 0;
        }
        let bytes = &self.0[byte_index..byte_index + 2];
        u16::from_be_bytes(bytes.try_into().unwrap())
    }
    pub fn store_half(&mut self, address: u32, value: u16) {
        let byte_index = address as usize;
        self.expand_memory(byte_index + 2);
        let bytes = value.to_be_bytes();
        self.0[byte_index..byte_index + 2].copy_from_slice(&bytes);
    }

    fn expand_memory(&mut self, required_size: usize) {
        if self.0.len() < required_size {
            self.0.resize(required_size, 0);
        }
    }
}
