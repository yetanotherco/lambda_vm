pub mod constraints;
pub mod segment;
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
    /// Segment size is too small (must be >= 4)
    SegmentSizeTooSmall(usize),
    /// Segment size is not a power of 2
    SegmentSizeNotPowerOfTwo(usize),
    /// Log count is not divisible by segment size
    LogCountNotDivisible { log_count: usize, segment_size: usize },
}

impl fmt::Display for ProverError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ProverError::MissingInstruction(pc) => {
                write!(f, "instruction not found for PC {pc:#x}")
            }
            ProverError::SegmentSizeTooSmall(size) => {
                write!(f, "segment_size must be >= 4, got {size}")
            }
            ProverError::SegmentSizeNotPowerOfTwo(size) => {
                write!(f, "segment_size must be power of 2, got {size}")
            }
            ProverError::LogCountNotDivisible { log_count, segment_size } => {
                write!(f, "log count ({log_count}) must be divisible by segment_size ({segment_size})")
            }
        }
    }
}

impl std::error::Error for ProverError {}
