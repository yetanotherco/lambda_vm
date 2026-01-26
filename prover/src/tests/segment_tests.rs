//! Segmentation tests for proving long programs in segments.
//!
//! These tests verify the segmentation feature:
//! - Configuration validation (power of 2, minimum size)
//! - Splitting logs into segments
//! - Independent proving of each segment

use crypto::fiat_shamir::default_transcript::DefaultTranscript;

use stark::proof::options::ProofOptions;
use stark::prover::{IsStarkProver, Prover};
use stark::traits::AIR;
use stark::verifier::{IsStarkVerifier, Verifier};

use crate::segment::{SegmentConfig, split_into_segments};
use crate::tables::lt::generate_lt_trace;
use crate::tables::trace_builder::Traces;
use crate::tables::types::{GoldilocksExtension, GoldilocksField};
use crate::test_utils::{
    collect_bitwise_lookups_from_logs, collect_bitwise_lookups_from_lt, collect_lt_lookups_from_logs,
    create_bitwise_air, create_cpu_air, create_lt_air, generate_minimal_bitwise_trace, run_asm_elf,
};

type F = GoldilocksField;
type E = GoldilocksExtension;

/// Run multi_prove and multi_verify for all VM tables with MINIMAL bitwise.
fn prove_and_verify_vm_minimal(
    cpu_trace: &mut stark::trace::TraceTable<F, E>,
    bitwise_trace: &mut stark::trace::TraceTable<F, E>,
    lt_trace: &mut stark::trace::TraceTable<F, E>,
) -> bool {
    let proof_options = ProofOptions::default_test_options();

    let cpu_air = create_cpu_air(&proof_options);
    let bitwise_air = create_bitwise_air(&proof_options);
    let lt_air = create_lt_air(&proof_options);

    let air_trace_pairs: Vec<(
        &dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>,
        _,
        _,
    )> = vec![
        (&cpu_air, cpu_trace, &()),
        (&bitwise_air, bitwise_trace, &()),
        (&lt_air, lt_trace, &()),
    ];

    let multi_proof =
        match Prover::multi_prove(air_trace_pairs, &mut DefaultTranscript::<E>::new(&[])) {
            Ok(proof) => proof,
            Err(e) => {
                eprintln!("Prover error: {:?}", e);
                return false;
            }
        };

    let airs: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>> =
        vec![&cpu_air, &bitwise_air, &lt_air];

    Verifier::multi_verify(&airs, &multi_proof, &mut DefaultTranscript::<E>::new(&[]))
}

// =============================================================================
// Configuration tests
// =============================================================================

#[test]
fn test_segment_config_valid() {
    // Valid configs
    let _ = SegmentConfig::new(4);
    let _ = SegmentConfig::new(64);
    let _ = SegmentConfig::new(1024);
}

#[test]
fn test_segment_config_default() {
    let config = SegmentConfig::default();
    assert_eq!(config.segment_size, 64);
}

#[test]
#[should_panic(expected = "segment_size must be >= 4")]
fn test_segment_config_min_size() {
    let _ = SegmentConfig::new(2);
}

#[test]
#[should_panic(expected = "segment_size must be power of 2")]
fn test_segment_config_power_of_two() {
    let _ = SegmentConfig::new(100);
}

// =============================================================================
// Split function tests
// =============================================================================

#[test]
fn test_split_into_segments_basic() {
    let (logs, _instructions) = run_asm_elf("loop_128");
    assert_eq!(logs.len(), 128, "loop_128.elf should have 128 instructions");

    let config = SegmentConfig::new(64);
    let segments = split_into_segments(&logs, &config);

    assert_eq!(segments.len(), 2, "Expected 2 segments of 64 each");
    assert_eq!(segments[0].len(), 64);
    assert_eq!(segments[1].len(), 64);
}

#[test]
fn test_split_into_segments_single() {
    let (logs, _instructions) = run_asm_elf("all_instructions_64");
    assert_eq!(
        logs.len(),
        64,
        "all_instructions_64.elf should have 64 instructions"
    );

    let config = SegmentConfig::new(64);
    let segments = split_into_segments(&logs, &config);

    assert_eq!(segments.len(), 1, "Expected 1 segment of 64");
    assert_eq!(segments[0].len(), 64);
}

#[test]
#[should_panic(expected = "must be divisible by segment_size")]
fn test_split_into_segments_not_divisible() {
    let (logs, _instructions) = run_asm_elf("arith_8");
    // arith_8 has 8 instructions, which is not divisible by 64
    let config = SegmentConfig::new(64);
    let _ = split_into_segments(&logs, &config);
}

// =============================================================================
// Segmented proving tests
// =============================================================================

#[test]
fn test_segmented_proving() {
    let (logs, instructions) = run_asm_elf("loop_128");

    // Verify we have exactly 128 instructions
    assert_eq!(
        logs.len(),
        128,
        "Test program must have exactly 128 instructions"
    );

    let config = SegmentConfig::new(64); // 64 rows per segment
    let segments = split_into_segments(&logs, &config);

    assert_eq!(segments.len(), 2, "Expected 2 segments of 64 each");

    for (i, segment_logs) in segments.iter().enumerate() {
        assert_eq!(segment_logs.len(), 64, "Each segment should have 64 rows");

        let mut traces = Traces::from_logs(segment_logs, instructions.clone())
            .expect("Failed to generate traces");

        // Collect lookups for minimal bitwise trace
        let lt_lookups = collect_lt_lookups_from_logs(segment_logs, &instructions);
        let mut lt_trace = generate_lt_trace(&lt_lookups);
        let mut bitwise_lookups = collect_bitwise_lookups_from_logs(segment_logs, &instructions);
        bitwise_lookups.extend(collect_bitwise_lookups_from_lt(&lt_lookups));
        let mut bitwise_trace = generate_minimal_bitwise_trace(&bitwise_lookups);

        let verified = prove_and_verify_vm_minimal(
            &mut traces.cpu,
            &mut bitwise_trace,
            &mut lt_trace,
        );
        assert!(verified, "Segment {} verification failed", i);

        println!("Segment {} verified: {} rows", i, segment_logs.len());
    }
}

#[test]
fn test_segmented_proving_four_segments() {
    let (logs, instructions) = run_asm_elf("loop_128");

    // Verify we have exactly 128 instructions
    assert_eq!(
        logs.len(),
        128,
        "Test program must have exactly 128 instructions"
    );

    let config = SegmentConfig::new(32); // 32 rows per segment = 4 segments
    let segments = split_into_segments(&logs, &config);

    assert_eq!(segments.len(), 4, "Expected 4 segments of 32 each");

    for (i, segment_logs) in segments.iter().enumerate() {
        assert_eq!(segment_logs.len(), 32, "Each segment should have 32 rows");

        let mut traces = Traces::from_logs(segment_logs, instructions.clone())
            .expect("Failed to generate traces");

        // Collect lookups for minimal bitwise trace
        let lt_lookups = collect_lt_lookups_from_logs(segment_logs, &instructions);
        let mut lt_trace = generate_lt_trace(&lt_lookups);
        let mut bitwise_lookups = collect_bitwise_lookups_from_logs(segment_logs, &instructions);
        bitwise_lookups.extend(collect_bitwise_lookups_from_lt(&lt_lookups));
        let mut bitwise_trace = generate_minimal_bitwise_trace(&bitwise_lookups);

        let verified = prove_and_verify_vm_minimal(
            &mut traces.cpu,
            &mut bitwise_trace,
            &mut lt_trace,
        );
        assert!(verified, "Segment {} verification failed", i);

        println!("Segment {} verified: {} rows", i, segment_logs.len());
    }
}
