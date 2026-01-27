//! Trace segmentation for proving arbitrarily long programs.
//!
//! This module provides utilities for splitting execution logs into fixed-size
//! segments that can be proven independently. Each segment must have a power-of-2
//! number of rows to satisfy FRI requirements.
//!
//! ## Usage
//!
//! ```ignore
//! use prover::segment::split_into_segments;
//!
//! let segments = split_into_segments(&logs, 64)?;
//!
//! for segment_logs in segments {
//!     let traces = Traces::from_logs(segment_logs, instructions.clone())?;
//!     // Prove each segment independently
//! }
//! ```

use std::fmt;

use executor::vm::logs::Log;

/// Errors that can occur during segmentation.
#[derive(Debug)]
pub enum SegmentError {
    /// Segment size is too small (must be >= 4)
    SizeTooSmall(usize),
    /// Segment size is not a power of 2
    SizeNotPowerOfTwo(usize),
    /// Log count is not divisible by segment size
    LogCountNotDivisible {
        log_count: usize,
        segment_size: usize,
    },
}

impl fmt::Display for SegmentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SegmentError::SizeTooSmall(size) => {
                write!(f, "segment_size must be >= 4, got {size}")
            }
            SegmentError::SizeNotPowerOfTwo(size) => {
                write!(f, "segment_size must be power of 2, got {size}")
            }
            SegmentError::LogCountNotDivisible {
                log_count,
                segment_size,
            } => {
                write!(
                    f,
                    "log count ({log_count}) must be divisible by segment_size ({segment_size})"
                )
            }
        }
    }
}

impl std::error::Error for SegmentError {}

/// Split logs into segments of exactly `segment_size` instructions.
///
/// The segment_size must be a power of 2 >= 4.
/// Returns slices of logs for each segment.
///
/// # Errors
///
/// Returns an error if:
/// - `segment_size` is less than 4
/// - `segment_size` is not a power of 2
/// - The total number of logs is not exactly divisible by segment_size
pub fn split_into_segments(logs: &[Log], segment_size: usize) -> Result<Vec<&[Log]>, SegmentError> {
    if segment_size < 4 {
        return Err(SegmentError::SizeTooSmall(segment_size));
    }
    if !segment_size.is_power_of_two() {
        return Err(SegmentError::SizeNotPowerOfTwo(segment_size));
    }
    if !logs.len().is_multiple_of(segment_size) {
        return Err(SegmentError::LogCountNotDivisible {
            log_count: logs.len(),
            segment_size,
        });
    }
    Ok(logs.chunks(segment_size).collect())
}
