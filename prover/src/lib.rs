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
}

impl fmt::Display for ProverError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ProverError::MissingInstruction(pc) => {
                write!(f, "instruction not found for PC {pc:#x}")
            }
        }
    }
}

impl std::error::Error for ProverError {}
