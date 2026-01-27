//! Segmentation tests for splitting logs into segments.
//!
//! These tests verify the segmentation logic:
//! - Configuration validation (power of 2, minimum size)
//! - Splitting logs into segments
//!
//! Proving tests are in prove_elfs_tests.rs.

use crate::segment::{SegmentError, split_into_segments};
use executor::vm::logs::Log;

// =============================================================================
// Test helpers
// =============================================================================

/// Create n dummy logs for testing segmentation logic.
fn make_dummy_logs(n: usize) -> Vec<Log> {
    (0..n)
        .map(|i| Log {
            current_pc: (i * 4) as u64,
            next_pc: ((i + 1) * 4) as u64,
            src1_val: 0,
            src2_val: 0,
            dst_val: 0,
        })
        .collect()
}

// =============================================================================
// Validation tests
// =============================================================================

#[test]
fn test_segment_size_min() {
    let logs = make_dummy_logs(8);
    let result = split_into_segments(&logs, 2);
    assert!(matches!(result, Err(SegmentError::SizeTooSmall(2))));
}

#[test]
fn test_segment_size_power_of_two() {
    let logs = make_dummy_logs(100);
    let result = split_into_segments(&logs, 100);
    assert!(matches!(result, Err(SegmentError::SizeNotPowerOfTwo(100))));
}

#[test]
fn test_split_not_divisible() {
    let logs = make_dummy_logs(100);
    let result = split_into_segments(&logs, 64);
    assert!(matches!(
        result,
        Err(SegmentError::LogCountNotDivisible {
            log_count: 100,
            segment_size: 64
        })
    ));
}

// =============================================================================
// Split function tests
// =============================================================================

#[test]
fn test_split_into_segments_basic() {
    let logs = make_dummy_logs(128);
    let segments = split_into_segments(&logs, 64).unwrap();

    assert_eq!(segments.len(), 2, "Expected 2 segments of 64 each");
    assert_eq!(segments[0].len(), 64);
    assert_eq!(segments[1].len(), 64);
}

#[test]
fn test_split_into_segments_single() {
    let logs = make_dummy_logs(64);
    let segments = split_into_segments(&logs, 64).unwrap();

    assert_eq!(segments.len(), 1, "Expected 1 segment of 64");
    assert_eq!(segments[0].len(), 64);
}

#[test]
fn test_split_into_segments_four() {
    let logs = make_dummy_logs(128);
    let segments = split_into_segments(&logs, 32).unwrap();

    assert_eq!(segments.len(), 4, "Expected 4 segments of 32 each");
    for segment in &segments {
        assert_eq!(segment.len(), 32);
    }
}

#[test]
fn test_split_preserves_log_order() {
    let logs = make_dummy_logs(8);
    let segments = split_into_segments(&logs, 4).unwrap();

    // First segment should have PCs 0, 4, 8, 12
    assert_eq!(segments[0][0].current_pc, 0);
    assert_eq!(segments[0][3].current_pc, 12);

    // Second segment should have PCs 16, 20, 24, 28
    assert_eq!(segments[1][0].current_pc, 16);
    assert_eq!(segments[1][3].current_pc, 28);
}
