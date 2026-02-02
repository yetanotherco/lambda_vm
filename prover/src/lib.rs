#[cfg(feature = "dhat-heap")]
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

pub mod constraints;
pub mod tables;
pub mod test_utils;
pub mod tests;
pub mod utils;

use std::fmt;

/// Error type for the prover crate.
#[derive(Debug)]
pub enum ProverError {
    /// Instruction not found for a given PC address
    MissingInstruction(u64),
    /// Program does not contain an ECALL (halt) instruction
    MissingEcall,
}

impl fmt::Display for ProverError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ProverError::MissingInstruction(pc) => {
                write!(f, "instruction not found for PC {pc:#x}")
            }
            ProverError::MissingEcall => {
                write!(f, "program does not contain an ECALL (halt) instruction")
            }
        }
    }
}

impl std::error::Error for ProverError {}
