//! Trace segmentation for proving arbitrarily long programs.
//!
//! This module provides utilities for splitting execution logs into fixed-size
//! segments that can be proven independently. Each segment must have a power-of-2
//! number of rows to satisfy FRI requirements.
//!
//! ## Usage
//!
//! ```ignore
//! use prover::segment::{SegmentConfig, split_into_segments};
//!
//! let config = SegmentConfig::new(64);
//! let segments = split_into_segments(&logs, &config);
//!
//! for segment_logs in segments {
//!     let traces = Traces::from_logs(segment_logs, instructions.clone())?;
//!     // Prove each segment independently
//! }
//! ```

mod config;

pub use config::SegmentConfig;

use executor::vm::logs::Log;

/// Split logs into segments of exactly `segment_size` instructions.
///
/// The segment_size must be a power of 2 >= 4 (configured via `SegmentConfig`).
/// Returns slices of logs for each segment.
///
/// # Panics
///
/// Panics if the total number of logs is not exactly divisible by segment_size.
/// Padding support will be added in a future implementation.
pub fn split_into_segments<'a>(logs: &'a [Log], config: &SegmentConfig) -> Vec<&'a [Log]> {
    assert!(
        logs.len().is_multiple_of(config.segment_size),
        "Total logs ({}) must be divisible by segment_size ({})",
        logs.len(),
        config.segment_size
    );
    logs.chunks(config.segment_size).collect()
}
