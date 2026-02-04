//! Trace segmentation support for large executions.
//!
//! When trace length exceeds the FFT domain limit, execution must be divided
//! into segments. This module provides the data structures for segment boundaries
//! and configuration.
//!
//! ## FFT Constraint
//!
//! The Goldilocks field has TWO_ADICITY = 32, meaning the maximum FFT domain
//! size is 2^32 elements. With a blowup factor of 4, the LDE domain is 4× the
//! trace length, so:
//!
//! - **Max trace length = 2^30 rows** (LDE = 2^32 = field limit)
//! - Exceeding this triggers `FFTError::DomainSizeError`
//!
//! For safety margin, we use MAX_TRACE_SIZE = 2^23 (~8M rows) by default.
//!
//! ## Note on Table Growth Rates
//!
//! **This implementation segments based on CPU table size** (1 row per instruction)
//! as a starting point. Different tables have different growth rates:
//!
//! | Table | Rows per instruction | Notes |
//! |-------|---------------------|-------|
//! | CPU | 1 | One row per instruction |
//! | MEMW | 1-4+ | Up to 3 register ops + memory op per instruction |
//! | LT | Variable | Grows with comparisons + timestamp ordering checks |
//! | LOAD | 0-1 | Only for load instructions |
//! | MUL | 0-1 | Only for multiply instructions |
//! | BRANCH | 0-1 | Only for taken branches |
//!
//! Future improvement: Check which table reaches MAX_TRACE_SIZE first.

use super::trace_builder::Traces;

/// Maximum trace size before segmentation is required.
///
/// Can be overridden via `LAMBDA_VM_MAX_TRACE_SIZE` env var.
/// Must be <= 2^30 due to FFT domain limit (TWO_ADICITY=32, blowup=4).
///
/// NOTE: Currently we segment based on CPU table size (1 row per instruction).
/// A proper implementation should check which table reaches this limit first.
pub const MAX_TRACE_SIZE: usize = 1 << 23; // 8M rows

/// Get max trace size from env var or constant.
pub fn get_max_trace_size() -> usize {
    std::env::var("LAMBDA_VM_MAX_TRACE_SIZE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(MAX_TRACE_SIZE)
}

/// State captured at segment boundaries for continuity.
///
/// This struct captures all state needed to resume trace generation
/// from a previous segment, ensuring continuity of timestamps, memory,
/// and register values across segment boundaries.
#[derive(Debug, Clone, Default)]
pub struct SegmentBoundary {
    /// Segment index (0-based)
    pub segment_index: u32,

    /// Starting timestamp for this segment.
    /// Each instruction consumes 4 timestamps, so this should be
    /// the end timestamp of the previous segment.
    pub start_timestamp: u64,

    /// Memory state: (address, value, timestamp) tuples.
    /// Contains all memory cells that were written during previous segments.
    pub memory_state: Vec<(u64, u8, u64)>,

    /// Register state: [(value, timestamp); 31] for registers x1-x31.
    /// x0 is excluded as it's always zero.
    pub register_state: [(u64, u64); 31],

    /// PC at segment start (for validation/debugging).
    pub start_pc: u64,
}

impl SegmentBoundary {
    /// Create the initial boundary for the first segment.
    pub fn initial() -> Self {
        Self {
            segment_index: 0,
            start_timestamp: 4, // Timestamps start at 4 (not 0)
            memory_state: Vec::new(),
            register_state: [(0, 0); 31],
            start_pc: 0,
        }
    }

    /// Create a boundary for the next segment.
    pub fn next(
        &self,
        end_timestamp: u64,
        memory_state: Vec<(u64, u8, u64)>,
        register_state: [(u64, u64); 31],
        next_pc: u64,
    ) -> Self {
        Self {
            segment_index: self.segment_index + 1,
            start_timestamp: end_timestamp,
            memory_state,
            register_state,
            start_pc: next_pc,
        }
    }
}

/// Configuration for segment-based trace generation.
#[derive(Debug, Clone)]
pub struct SegmentConfig {
    /// Whether this is the final segment (has ECALL).
    /// Non-final segments use a dummy HALT trace with zero multiplicity.
    pub is_final: bool,
}

impl SegmentConfig {
    /// Create config for a non-final segment.
    pub fn intermediate() -> Self {
        Self { is_final: false }
    }

    /// Create config for the final segment.
    pub fn final_segment() -> Self {
        Self { is_final: true }
    }
}

/// Result of generating traces for a single segment.
pub struct SegmentResult {
    /// Generated traces for this segment.
    pub traces: Traces,

    /// Boundary state for the next segment (None if this is the final segment).
    pub next_boundary: Option<SegmentBoundary>,

    /// Whether this was the final segment.
    pub is_final: bool,
}
