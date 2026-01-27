//! Segmentation tests for proving long programs in segments.
//!
//! These tests verify the segmentation feature:
//! - Configuration validation (power of 2, minimum size)
//! - Splitting logs into segments
//! - Independent proving of each segment

use crate::segment::{SegmentError, split_into_segments};
use crate::tables::lt::generate_lt_trace;
use crate::tables::trace_builder::Traces;
use crate::test_utils::{
    collect_bitwise_lookups_from_logs, collect_bitwise_lookups_from_lt,
    collect_lt_lookups_from_logs, generate_minimal_bitwise_trace, prove_and_verify_vm_minimal,
    run_asm_elf,
};

// =============================================================================
// Validation tests
// =============================================================================

#[test]
fn test_segment_size_min() {
    let (logs, _) = run_asm_elf("arith_8");
    let result = split_into_segments(&logs, 2);
    assert!(matches!(result, Err(SegmentError::SizeTooSmall(2))));
}

#[test]
fn test_segment_size_power_of_two() {
    let (logs, _) = run_asm_elf("loop_128");
    let result = split_into_segments(&logs, 100);
    assert!(matches!(result, Err(SegmentError::SizeNotPowerOfTwo(100))));
}

#[test]
fn test_split_not_divisible() {
    let (logs, _) = run_asm_elf("arith_8");
    // arith_8 has 8 instructions, which is not divisible by 64
    let result = split_into_segments(&logs, 64);
    assert!(matches!(
        result,
        Err(SegmentError::LogCountNotDivisible {
            log_count: 8,
            segment_size: 64
        })
    ));
}

// =============================================================================
// Split function tests
// =============================================================================

#[test]
fn test_split_into_segments_basic() {
    let (logs, _) = run_asm_elf("loop_128");
    assert_eq!(logs.len(), 128, "loop_128.elf should have 128 instructions");

    let segments = split_into_segments(&logs, 64).unwrap();

    assert_eq!(segments.len(), 2, "Expected 2 segments of 64 each");
    assert_eq!(segments[0].len(), 64);
    assert_eq!(segments[1].len(), 64);
}

#[test]
fn test_split_into_segments_single() {
    let (logs, _) = run_asm_elf("all_instructions_64");
    assert_eq!(logs.len(), 64);

    let segments = split_into_segments(&logs, 64).unwrap();

    assert_eq!(segments.len(), 1, "Expected 1 segment of 64");
    assert_eq!(segments[0].len(), 64);
}

// =============================================================================
// Segmented proving tests
// =============================================================================

#[test]
fn test_segmented_proving() {
    let (logs, instructions) = run_asm_elf("loop_128");
    assert_eq!(logs.len(), 128);

    let segments = split_into_segments(&logs, 64).unwrap();
    assert_eq!(segments.len(), 2);

    for (i, segment_logs) in segments.iter().enumerate() {
        assert_eq!(segment_logs.len(), 64);

        let mut traces = Traces::from_logs(segment_logs, instructions.clone())
            .expect("Failed to generate traces");

        let lt_lookups = collect_lt_lookups_from_logs(segment_logs, &instructions);
        let mut lt_trace = generate_lt_trace(&lt_lookups);
        let mut bitwise_lookups = collect_bitwise_lookups_from_logs(segment_logs, &instructions);
        bitwise_lookups.extend(collect_bitwise_lookups_from_lt(&lt_lookups));
        let mut bitwise_trace = generate_minimal_bitwise_trace(&bitwise_lookups);

        let verified =
            prove_and_verify_vm_minimal(&mut traces.cpu, &mut bitwise_trace, &mut lt_trace);
        assert!(verified, "Segment {} verification failed", i);

        println!("Segment {} verified: {} rows", i, segment_logs.len());
    }
}

#[test]
fn test_segmented_proving_four_segments() {
    let (logs, instructions) = run_asm_elf("loop_128");
    assert_eq!(logs.len(), 128);

    let segments = split_into_segments(&logs, 32).unwrap();
    assert_eq!(segments.len(), 4);

    for (i, segment_logs) in segments.iter().enumerate() {
        assert_eq!(segment_logs.len(), 32);

        let mut traces = Traces::from_logs(segment_logs, instructions.clone())
            .expect("Failed to generate traces");

        let lt_lookups = collect_lt_lookups_from_logs(segment_logs, &instructions);
        let mut lt_trace = generate_lt_trace(&lt_lookups);
        let mut bitwise_lookups = collect_bitwise_lookups_from_logs(segment_logs, &instructions);
        bitwise_lookups.extend(collect_bitwise_lookups_from_lt(&lt_lookups));
        let mut bitwise_trace = generate_minimal_bitwise_trace(&bitwise_lookups);

        let verified =
            prove_and_verify_vm_minimal(&mut traces.cpu, &mut bitwise_trace, &mut lt_trace);
        assert!(verified, "Segment {} verification failed", i);

        println!("Segment {} verified: {} rows", i, segment_logs.len());
    }
}
