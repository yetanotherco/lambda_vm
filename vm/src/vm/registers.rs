use std::fmt::Display;

const STACK_MEMORY_SIZE: u32 = 0xFFFFFFFC; // 4GB (Multiple of 4)

#[derive(Debug)]
/// Holds the current value of all 32 registers
/// Register zero is implicit as it cannot hold any value other than zero
pub struct Registers([u32; 31]);

impl Default for Registers {
    fn default() -> Self {
        let mut registers = Registers(Default::default());
        // Initialize stack pointer according to available memory size
        registers.0[1] = STACK_MEMORY_SIZE;
        registers
    }
}

impl Registers {
    /// Read the current value of the given register
    pub fn read(&self, register: u32) -> u32 {
        if register == 0 {
            0
        } else {
            self.0[register as usize - 1]
        }
    }

    /// Update the value of the given register
    /// Writes to register zero are a no-op
    pub fn write(&mut self, register: u32, value: u32) {
        if register != 0 {
            self.0[register as usize - 1] = value
        }
    }

    /// Read the return values (aka registers a0 & a1)
    pub fn read_return_values(&self) -> (u32, u32) {
        (self.0[9], self.0[10])
    }
}

impl Display for Registers {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        const REGISTER_NAMES: [&str; 31] = [
            "ra", "sp", "gp", "tp", "t0", "t1", "t2", "s0", "s1", "a0", "a1", "a2", "a3", "a4",
            "a5", "a6", "a7", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11", "t3",
            "t4", "t5", "t6",
        ];
        let registers = self
            .0
            .iter()
            .zip(REGISTER_NAMES)
            .map(|(reg, name)| format!("{name}: {reg}"))
            .collect::<Vec<String>>()
            .join(",");
        writeln!(f, "[{}]", registers)
    }
}
