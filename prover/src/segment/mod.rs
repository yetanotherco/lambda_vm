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
//! let segments = split_into_segments(&logs, 64);
//!
//! for segment_logs in segments {
//!     let traces = Traces::from_logs(segment_logs, instructions.clone())?;
//!     // Prove each segment independently
//! }
//! ```

use executor::vm::logs::Log;

/// Split logs into segments of exactly `segment_size` instructions.
///
/// The segment_size must be a power of 2 >= 4.
/// Returns slices of logs for each segment.
///
/// # Panics
///
/// Panics if:
/// - `segment_size` is less than 4
/// - `segment_size` is not a power of 2
/// - The total number of logs is not exactly divisible by segment_size
pub fn split_into_segments(logs: &[Log], segment_size: usize) -> Vec<&[Log]> {
    assert!(segment_size >= 4, "segment_size must be >= 4");
    assert!(
        segment_size.is_power_of_two(),
        "segment_size must be power of 2"
    );
    assert!(
        logs.len().is_multiple_of(segment_size),
        "Total logs ({}) must be divisible by segment_size ({})",
        logs.len(),
        segment_size
    );
    logs.chunks(segment_size).collect()
}
