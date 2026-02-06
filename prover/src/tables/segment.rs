//! Trace segmentation support for large executions.
//!
//! When trace length exceeds the FFT domain limit, execution must be divided
//! into segments. This module provides the data structures for segmentation.
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

/// Cumulative derived op counts after processing each CPU operation.
///
/// During phase 2, we track how many derived ops have been generated after
/// each CPU operation. This allows slicing derived ops per segment without
/// re-processing: `segment_ops = all_ops[boundaries[start]..boundaries[end]]`.
#[derive(Debug, Clone, Copy, Default)]
pub struct OpBoundaries {
    pub memw: usize,
    pub load: usize,
    pub lt: usize,
    pub mul: usize,
    pub branch: usize,
    /// Bitwise ops from phase 2 (CPU ops + load ops). Phase 4 adds more per-segment.
    pub bitwise: usize,
}

/// Result of generating traces for a single segment.
pub struct SegmentResult {
    /// Generated traces for this segment.
    pub traces: Traces,
    /// Whether this was the final segment.
    pub is_final: bool,
    /// Segment index (0-based).
    pub segment_index: usize,
}
