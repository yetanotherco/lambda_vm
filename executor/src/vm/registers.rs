use alloc::vec::Vec;
use core::fmt::Display;

pub use crate::constants::STACK_TOP;

#[derive(Debug)]
/// Holds the current value of all 32 registers
/// Register zero is implicit as it cannot hold any value other than zero
pub struct Registers([u64; 31]);

impl Default for Registers {
    fn default() -> Self {
        let mut registers = Registers(Default::default());
        // Initialize stack pointer according to available memory size
        registers.0[1] = STACK_TOP;
        registers
    }
}

impl Registers {
    /// Read the current value of the given register
    pub fn read(&self, register: u32) -> Result<u64, RegisterError> {
        if register > 31 {
            return Err(RegisterError::InvalidRegister(register));
        }
        if register == 0 {
            Ok(0)
        } else {
            Ok(self.0[register as usize - 1])
        }
    }

    /// Update the value of the given register
    /// Writes to register zero are a no-op
    pub fn write(&mut self, register: u32, value: u64) -> Result<(), RegisterError> {
        if register > 31 {
            return Err(RegisterError::InvalidRegister(register));
        }
        if register != 0 {
            self.0[register as usize - 1] = value
        }
        Ok(())
    }

    /// Read the return values (aka registers a0 & a1)
    pub fn read_return_values(&self) -> (u64, u64) {
        (self.0[9], self.0[10])
    }
}

impl Display for Registers {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        const REGISTER_NAMES: [&str; 32] = [
            "zero", "ra", "sp", "gp", "tp", "t0", "t1", "t2", "s0", "s1", "a0", "a1", "a2", "a3",
            "a4", "a5", "a6", "a7", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11",
            "t3", "t4", "t5", "t6",
        ];
        let values = core::iter::once(0u64).chain(self.0.iter().copied());

        for (i, chunk) in REGISTER_NAMES
            .iter()
            .zip(values)
            .collect::<Vec<_>>()
            .chunks(4)
            .enumerate()
        {
            if i != 0 {
                writeln!(f)?;
            }

            for (name, value) in chunk {
                write!(f, "{name:>4}: {value:#018x}",)?;
            }
        }
        Ok(())
    }
}

#[derive(thiserror::Error, Debug)]
pub enum RegisterError {
    #[error("Invalid register number: {0}")]
    InvalidRegister(u32),
}
