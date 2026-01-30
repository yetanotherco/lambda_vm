//! VM Prover integration tests using multi_prove.
//!
//! These tests verify the full prover pipeline:
//! - Run ELF through executor
//! - Generate traces for CPU and Bitwise tables
//! - Use multi_prove/multi_verify with bus interactions
//!
//! Wired buses:
//! - CPU sends AND_BYTE, OR_BYTE, XOR_BYTE to Bitwise (×8 each)
//! - CPU sends MSB16 to Bitwise (for rv1_sign_bit, arg2_sign_bit when word_instr=1)
//! - CPU sends MSB8 to Bitwise (for res_sign_bit when word_instr=1)
//! - CPU sends ZERO to Bitwise (for is_equal when BEQ=1)
//!
//! TODO: LT bus (needs LT table integration)

use crypto::fiat_shamir::default_transcript::DefaultTranscript;

use stark::constraints::transition::TransitionConstraint;
use stark::lookup::{AirWithBuses, AuxiliaryTraceBuildData};
use stark::proof::options::ProofOptions;
use stark::prover::{IsStarkProver, Prover};
use stark::trace::TraceTable;
use stark::traits::AIR;
use stark::verifier::{IsStarkVerifier, Verifier};

use executor::elf::Elf;

use crate::tables::bitwise;
use crate::tables::decode;
use crate::tables::trace_builder::Traces;
use crate::tables::types::{GoldilocksExtension, GoldilocksField};

// Import shared utilities
use crate::test_utils::{
    create_bitwise_air, create_branch_air, create_cpu_air, create_decode_air, create_load_air,
    create_lt_air, create_memw_air, run_asm_elf,
};

type F = GoldilocksField;
type E = GoldilocksExtension;

// =============================================================================
// Prover test helpers
// =============================================================================

/// Run multi_prove and multi_verify for all VM tables.
///
/// Uses the FULL 2^20 row bitwise table with preprocessed commitment.
/// Returns true if verification succeeds.
fn prove_and_verify_vm(
    cpu_trace: &mut TraceTable<F, E>,
    bitwise_trace: &mut TraceTable<F, E>,
    lt_trace: &mut TraceTable<F, E>,
    memw_trace: &mut TraceTable<F, E>,
    load_trace: &mut TraceTable<F, E>,
    decode_trace: &mut TraceTable<F, E>,
    branch_trace: &mut TraceTable<F, E>,
    elf: &Elf,
) -> bool {
    let proof_options = ProofOptions::default_test_options();

    let cpu_air = create_cpu_air(&proof_options);
    // Use preprocessed commitment for full bitwise table verification
    let bitwise_air = create_bitwise_air(&proof_options).with_preprocessed(
        bitwise::preprocessed_commitment(),
        bitwise::NUM_PRECOMPUTED_COLS,
    );
    let lt_air = create_lt_air(&proof_options);
    let memw_air = create_memw_air(&proof_options);
    let load_air = create_load_air(&proof_options);
    // Verifier computes DECODE commitment directly from ELF (no executor needed)
    let decode_air = create_decode_air(&proof_options).with_preprocessed(
        decode::commitment_from_elf(elf, &proof_options).expect("Failed to compute decode commitment"),
        decode::NUM_PRECOMPUTED_COLS,
    );
    let branch_air = create_branch_air(&proof_options);

    let air_trace_pairs: Vec<(
        &dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>,
        _,
        _,
    )> = vec![
        (&cpu_air, cpu_trace, &()),
        (&bitwise_air, bitwise_trace, &()),
        (&lt_air, lt_trace, &()),
        (&memw_air, memw_trace, &()),
        (&load_air, load_trace, &()),
        (&decode_air, decode_trace, &()),
        (&branch_air, branch_trace, &()),
    ];

    let multi_proof =
        match Prover::multi_prove(air_trace_pairs, &mut DefaultTranscript::<E>::new(&[])) {
            Ok(proof) => proof,
            Err(e) => {
                eprintln!("Prover error: {:?}", e);
                return false;
            }
        };

    let airs: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>> = vec![
        &cpu_air,
        &bitwise_air,
        &lt_air,
        &memw_air,
        &load_air,
        &decode_air,
        &branch_air,
    ];

    let result = Verifier::multi_verify(&airs, &multi_proof, &mut DefaultTranscript::<E>::new(&[]));
    if !result {
        eprintln!("Verifier failed!");
    }
    result
}

/// Run multi_prove and multi_verify for all VM tables.
///
/// Used for fast tests where the bitwise table is a dummy that only contains
/// the rows needed to balance the bus. NOT the full preprocessed table.
fn prove_and_verify_vm_minimal(
    cpu_trace: &mut TraceTable<F, E>,
    bitwise_trace: &mut TraceTable<F, E>,
    lt_trace: &mut TraceTable<F, E>,
    memw_trace: &mut TraceTable<F, E>,
    load_trace: &mut TraceTable<F, E>,
    decode_trace: &mut TraceTable<F, E>,
    branch_trace: &mut TraceTable<F, E>,
) -> bool {
    let proof_options = ProofOptions::default_test_options();

    let cpu_air = create_cpu_air(&proof_options);
    let bitwise_air = create_bitwise_air(&proof_options);
    let lt_air = create_lt_air(&proof_options);
    let memw_air = create_memw_air(&proof_options);
    let load_air = create_load_air(&proof_options);
    let decode_air = create_decode_air(&proof_options);
    let branch_air = create_branch_air(&proof_options);

    let air_trace_pairs: Vec<(
        &dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>,
        _,
        _,
    )> = vec![
        (&cpu_air, cpu_trace, &()),
        (&bitwise_air, bitwise_trace, &()),
        (&lt_air, lt_trace, &()),
        (&memw_air, memw_trace, &()),
        (&load_air, load_trace, &()),
        (&decode_air, decode_trace, &()),
        (&branch_air, branch_trace, &()),
    ];

    let multi_proof =
        match Prover::multi_prove(air_trace_pairs, &mut DefaultTranscript::<E>::new(&[])) {
            Ok(proof) => proof,
            Err(e) => {
                eprintln!("Prover error: {:?}", e);
                return false;
            }
        };

    let airs: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>> = vec![
        &cpu_air,
        &bitwise_air,
        &lt_air,
        &memw_air,
        &load_air,
        &decode_air,
        &branch_air,
    ];

    Verifier::multi_verify(&airs, &multi_proof, &mut DefaultTranscript::<E>::new(&[]))
}

// =============================================================================
// Integration tests
// =============================================================================

/// Test CPU table alone (no bus interactions) to verify basic prove/verify works.
#[test]
fn test_cpu_only_no_bus() {
    let (elf, logs, instructions) = run_asm_elf("sub");
    assert_eq!(logs.len(), 4);

    let mut cpu_trace = Traces::from_logs(&logs, instructions).unwrap().cpu;
    println!(
        "CPU trace: {} rows x {} cols",
        cpu_trace.main_table.height, cpu_trace.main_table.width
    );

    let proof_options = ProofOptions::default_test_options();

    // Create AIR with NO bus interactions
    let transition_constraints: Vec<Box<dyn TransitionConstraint<F, E>>> = vec![];
    let auxiliary_trace_build_data = AuxiliaryTraceBuildData {
        interactions: vec![], // NO bus interactions
    };
    let cpu_air: AirWithBuses<F, E, stark::lookup::NullBoundaryConstraintBuilder, ()> =
        AirWithBuses::new(
            crate::tables::cpu::cols::NUM_COLUMNS,
            auxiliary_trace_build_data,
            &proof_options,
            1,
            transition_constraints,
        );

    let air_trace_pairs: Vec<(
        &dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>,
        _,
        _,
    )> = vec![(&cpu_air, &mut cpu_trace, &())];

    let multi_proof = Prover::multi_prove(air_trace_pairs, &mut DefaultTranscript::<E>::new(&[]))
        .expect("Prover failed");

    let airs: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>> = vec![&cpu_air];
    assert!(
        Verifier::multi_verify(&airs, &multi_proof, &mut DefaultTranscript::<E>::new(&[])),
        "CPU-only verification failed"
    );
}

// =============================================================================
// Fast tests using minimal (dummy) bitwise table
// =============================================================================
//
// These tests use a minimal bitwise table that only contains rows for the
// actual lookups. This is ~1000x faster than the full 2^20 row table.
//
// **WARNING: The minimal table is NOT production-safe!**
// The verifier expects the full deterministic 2^20 row public table.
// A minimal table would require the prover to reveal all values,
// making the proof size unacceptably large.

#[test]
fn test_prove_elfs_sub_fast() {
    let _ = env_logger::builder().is_test(true).try_init();
    let (elf, logs, instructions) = run_asm_elf("sub");
    assert_eq!(logs.len(), 4, "sub.elf should have 4 steps");

    // Use full Traces to get real MEMW trace (includes register operations)
    let mut traces = Traces::from_logs_minimal(&logs, instructions.clone()).unwrap();

    assert!(
        prove_and_verify_vm_minimal(
            &mut traces.cpu,
            &mut traces.bitwise,
            &mut traces.lt,
            &mut traces.memw,
            &mut traces.load,
            &mut traces.decode,
            &mut traces.branch
        ),
        "Proof verification failed for sub program (fast)"
    );
}

#[test]
fn test_prove_elfs_sub_neg_result_fast() {
    let (elf, logs, instructions) = run_asm_elf("sub_neg_result");
    assert_eq!(logs.len(), 4, "sub_neg_result.elf should have 4 steps");

    let mut traces = Traces::from_logs_minimal(&logs, instructions.clone()).unwrap();

    println!(
        "Fast SUB_NEG: CPU {} rows, Bitwise {} rows (minimal)",
        traces.cpu.main_table.height, traces.bitwise.main_table.height,
    );

    assert!(
        prove_and_verify_vm_minimal(
            &mut traces.cpu,
            &mut traces.bitwise,
            &mut traces.lt,
            &mut traces.memw,
            &mut traces.load,
            &mut traces.decode,
            &mut traces.branch
        ),
        "Proof verification failed for sub_neg_result program (fast)"
    );
}

#[test]
fn test_prove_elfs_sub_underflow_fast() {
    let (elf, logs, instructions) = run_asm_elf("sub_underflow");
    assert_eq!(logs.len(), 4, "sub_underflow.elf should have 4 steps");

    let mut traces = Traces::from_logs_minimal(&logs, instructions.clone()).unwrap();

    println!(
        "Fast SUB_UNDERFLOW: CPU {} rows, Bitwise {} rows (minimal)",
        traces.cpu.main_table.height, traces.bitwise.main_table.height,
    );

    assert!(
        prove_and_verify_vm_minimal(
            &mut traces.cpu,
            &mut traces.bitwise,
            &mut traces.lt,
            &mut traces.memw,
            &mut traces.load,
            &mut traces.decode,
            &mut traces.branch
        ),
        "Proof verification failed for sub_underflow program (fast)"
    );
}

#[test]
fn test_prove_elfs_subw_fast() {
    let (elf, logs, instructions) = run_asm_elf("subw");
    assert_eq!(logs.len(), 4, "subw.elf should have 4 steps");

    let mut traces = Traces::from_logs_minimal(&logs, instructions.clone()).unwrap();

    println!(
        "Fast SUBW: CPU {} rows, Bitwise {} rows (minimal)",
        traces.cpu.main_table.height, traces.bitwise.main_table.height,
    );

    assert!(
        prove_and_verify_vm_minimal(
            &mut traces.cpu,
            &mut traces.bitwise,
            &mut traces.lt,
            &mut traces.memw,
            &mut traces.load,
            &mut traces.decode,
            &mut traces.branch
        ),
        "Proof verification failed for subw program (fast)"
    );
}

/// 8-instruction test with LUI
#[test]
fn test_prove_elfs_arith_lui_8() {
    let (elf, logs, instructions) = run_asm_elf("arith_lui_8");
    assert_eq!(logs.len(), 8, "arith_lui_8.elf should have 8 steps");

    let mut traces = Traces::from_logs_minimal(&logs, instructions.clone()).unwrap();

    println!(
        "ArithLUI8: CPU {} rows, Bitwise {} rows (minimal)",
        traces.cpu.main_table.height, traces.bitwise.main_table.height,
    );

    assert!(
        prove_and_verify_vm_minimal(
            &mut traces.cpu,
            &mut traces.bitwise,
            &mut traces.lt,
            &mut traces.memw,
            &mut traces.load,
            &mut traces.decode,
            &mut traces.branch
        ),
        "Proof verification failed for arith_lui_8 program"
    );
}

/// 8-instruction test with ADD, SUB, ADDW, SUBW
#[test]
fn test_prove_elfs_arith_8() {
    let (elf, logs, instructions) = run_asm_elf("arith_8");
    assert_eq!(logs.len(), 8, "arith_8.elf should have 8 steps");

    let mut traces = Traces::from_logs_minimal(&logs, instructions.clone()).unwrap();

    println!(
        "Arith8: CPU {} rows, Bitwise {} rows (minimal)",
        traces.cpu.main_table.height, traces.bitwise.main_table.height,
    );

    assert!(
        prove_and_verify_vm_minimal(
            &mut traces.cpu,
            &mut traces.bitwise,
            &mut traces.lt,
            &mut traces.memw,
            &mut traces.load,
            &mut traces.decode,
            &mut traces.branch
        ),
        "Proof verification failed for arith_8 program"
    );
}

/// Basic arithmetic test with 32 instructions covering:
/// - 64-bit ADD with positive, negative, and edge cases
/// - 64-bit SUB with underflow, negative results
/// - 32-bit ADDW/SUBW with sign extension
#[test]
fn test_prove_elfs_basic_arith_32() {
    let (elf, logs, instructions) = run_asm_elf("basic_arith_32");
    assert_eq!(logs.len(), 32, "basic_arith_32.elf should have 32 steps");

    let mut traces = Traces::from_logs_minimal(&logs, instructions.clone()).unwrap();

    println!(
        "BasicArith32: CPU {} rows, Bitwise {} rows (minimal)",
        traces.cpu.main_table.height, traces.bitwise.main_table.height,
    );

    assert!(
        prove_and_verify_vm_minimal(
            &mut traces.cpu,
            &mut traces.bitwise,
            &mut traces.lt,
            &mut traces.memw,
            &mut traces.load,
            &mut traces.decode,
            &mut traces.branch
        ),
        "Proof verification failed for basic_arith_32 program"
    );
}

/// Comprehensive test covering all basic VM operations:
/// - 64-bit arithmetic: ADD, SUB (positive, negative, overflow)
/// - 32-bit word arithmetic: ADDW, SUBW (with sign extension)
/// - Multiplication: MUL, MULW
/// - Division: DIV, REM
/// - Shifts: SLL, SLLI, SRLI, SRA
/// - Bitwise: ANDI, ORI, XORI
/// - Comparisons: SLTI, SLTIU
/// - Immediates: LUI, ADDI with edge cases
#[test]
fn test_prove_elfs_comprehensive() {
    let _ = env_logger::builder().is_test(true).try_init();

    let (elf, logs, instructions) = run_asm_elf("comprehensive_test");
    assert_eq!(
        logs.len(),
        32,
        "comprehensive_test.elf should have 32 steps"
    );

    let mut traces = Traces::from_logs_minimal(&logs, instructions.clone()).unwrap();

    // Collect LT lookups first (needed for both LT trace and bitwise lookups)

    println!(
        "Comprehensive: CPU {} rows, Bitwise {} rows",
        traces.cpu.main_table.height, traces.bitwise.main_table.height,
    );

    assert!(
        prove_and_verify_vm_minimal(
            &mut traces.cpu,
            &mut traces.bitwise,
            &mut traces.lt,
            &mut traces.memw,
            &mut traces.load,
            &mut traces.decode,
            &mut traces.branch
        ),
        "Proof verification failed for comprehensive_test program"
    );
}

// =============================================================================
// Instruction-specific 8-step tests
// =============================================================================

#[test]
fn test_prove_elfs_test_add_8() {
    let (elf, logs, instructions) = run_asm_elf("test_add_8");
    assert_eq!(logs.len(), 8);
    let mut traces = Traces::from_logs_minimal(&logs, instructions.clone()).unwrap();
    // Use traces.lt and traces.bitwise directly instead of generating separate ones
    // This includes MEMW timestamp ordering LT ops and their bitwise lookups
    assert!(
        prove_and_verify_vm_minimal(
            &mut traces.cpu,
            &mut traces.bitwise,
            &mut traces.lt,
            &mut traces.memw,
            &mut traces.load,
            &mut traces.decode,
            &mut traces.branch
        ),
        "test_add_8 failed"
    );
}

#[test]
fn test_prove_elfs_test_sub_8() {
    let (elf, logs, instructions) = run_asm_elf("test_sub_8");
    assert_eq!(logs.len(), 8);
    let mut traces = Traces::from_logs_minimal(&logs, instructions.clone()).unwrap();
    assert!(
        prove_and_verify_vm_minimal(
            &mut traces.cpu,
            &mut traces.bitwise,
            &mut traces.lt,
            &mut traces.memw,
            &mut traces.load,
            &mut traces.decode,
            &mut traces.branch
        ),
        "test_sub_8 failed"
    );
}

#[test]
fn test_prove_elfs_test_addw_8() {
    let (elf, logs, instructions) = run_asm_elf("test_addw_8");
    assert_eq!(logs.len(), 8);
    let mut traces = Traces::from_logs_minimal(&logs, instructions.clone()).unwrap();
    assert!(
        prove_and_verify_vm_minimal(
            &mut traces.cpu,
            &mut traces.bitwise,
            &mut traces.lt,
            &mut traces.memw,
            &mut traces.load,
            &mut traces.decode,
            &mut traces.branch
        ),
        "test_addw_8 failed"
    );
}

#[test]
fn test_prove_elfs_test_subw_8() {
    let (elf, logs, instructions) = run_asm_elf("test_subw_8");
    assert_eq!(logs.len(), 8);
    let mut traces = Traces::from_logs_minimal(&logs, instructions.clone()).unwrap();
    // Using traces from Traces::from_logs() which includes MEMW LT ops
    assert!(
        prove_and_verify_vm_minimal(
            &mut traces.cpu,
            &mut traces.bitwise,
            &mut traces.lt,
            &mut traces.memw,
            &mut traces.load,
            &mut traces.decode,
            &mut traces.branch
        ),
        "test_subw_8 failed"
    );
}

#[test]
fn test_prove_elfs_test_addw_lui_8() {
    let (elf, logs, instructions) = run_asm_elf("test_addw_lui_8");
    assert_eq!(logs.len(), 8);
    let mut traces = Traces::from_logs_minimal(&logs, instructions.clone()).unwrap();
    // Using traces from Traces::from_logs() which includes MEMW LT ops
    assert!(
        prove_and_verify_vm_minimal(
            &mut traces.cpu,
            &mut traces.bitwise,
            &mut traces.lt,
            &mut traces.memw,
            &mut traces.load,
            &mut traces.decode,
            &mut traces.branch
        ),
        "test_addw_lui_8 failed"
    );
}

#[test]
fn test_prove_elfs_test_subw_lui_8() {
    let (elf, logs, instructions) = run_asm_elf("test_subw_lui_8");
    assert_eq!(logs.len(), 8);
    let mut traces = Traces::from_logs_minimal(&logs, instructions.clone()).unwrap();
    // Using traces from Traces::from_logs() which includes MEMW LT ops
    assert!(
        prove_and_verify_vm_minimal(
            &mut traces.cpu,
            &mut traces.bitwise,
            &mut traces.lt,
            &mut traces.memw,
            &mut traces.load,
            &mut traces.decode,
            &mut traces.branch
        ),
        "test_subw_lui_8 failed"
    );
}

#[test]
fn test_prove_elfs_test_add_neg_8() {
    let (elf, logs, instructions) = run_asm_elf("test_add_neg_8");
    assert_eq!(logs.len(), 8);
    let mut traces = Traces::from_logs_minimal(&logs, instructions.clone()).unwrap();
    // Using traces from Traces::from_logs() which includes MEMW LT ops
    assert!(
        prove_and_verify_vm_minimal(
            &mut traces.cpu,
            &mut traces.bitwise,
            &mut traces.lt,
            &mut traces.memw,
            &mut traces.load,
            &mut traces.decode,
            &mut traces.branch
        ),
        "test_add_neg_8 failed"
    );
}

#[test]
fn test_prove_elfs_test_sub_neg_8() {
    let (elf, logs, instructions) = run_asm_elf("test_sub_neg_8");
    assert_eq!(logs.len(), 8);
    let mut traces = Traces::from_logs_minimal(&logs, instructions.clone()).unwrap();
    // Using traces from Traces::from_logs() which includes MEMW LT ops
    assert!(
        prove_and_verify_vm_minimal(
            &mut traces.cpu,
            &mut traces.bitwise,
            &mut traces.lt,
            &mut traces.memw,
            &mut traces.load,
            &mut traces.decode,
            &mut traces.branch
        ),
        "test_sub_neg_8 failed"
    );
}

#[test]
fn test_prove_elfs_test_mul_8() {
    let (elf, logs, instructions) = run_asm_elf("test_mul_8");
    assert_eq!(logs.len(), 8);
    let mut traces = Traces::from_logs_minimal(&logs, instructions.clone()).unwrap();
    // Using traces from Traces::from_logs() which includes MEMW LT ops
    assert!(
        prove_and_verify_vm_minimal(
            &mut traces.cpu,
            &mut traces.bitwise,
            &mut traces.lt,
            &mut traces.memw,
            &mut traces.load,
            &mut traces.decode,
            &mut traces.branch
        ),
        "test_mul_8 failed"
    );
}

#[test]
fn test_prove_elfs_test_div_8() {
    let (elf, logs, instructions) = run_asm_elf("test_div_8");
    assert_eq!(logs.len(), 8);
    let mut traces = Traces::from_logs_minimal(&logs, instructions.clone()).unwrap();
    // Using traces from Traces::from_logs() which includes MEMW LT ops
    assert!(
        prove_and_verify_vm_minimal(
            &mut traces.cpu,
            &mut traces.bitwise,
            &mut traces.lt,
            &mut traces.memw,
            &mut traces.load,
            &mut traces.decode,
            &mut traces.branch
        ),
        "test_div_8 failed"
    );
}

#[test]
fn test_prove_elfs_test_shift_8() {
    let (elf, logs, instructions) = run_asm_elf("test_shift_8");
    assert_eq!(logs.len(), 8);
    let mut traces = Traces::from_logs_minimal(&logs, instructions.clone()).unwrap();
    // Using traces from Traces::from_logs() which includes MEMW LT ops
    assert!(
        prove_and_verify_vm_minimal(
            &mut traces.cpu,
            &mut traces.bitwise,
            &mut traces.lt,
            &mut traces.memw,
            &mut traces.load,
            &mut traces.decode,
            &mut traces.branch
        ),
        "test_shift_8 failed"
    );
}

#[test]
fn test_prove_elfs_test_bitwise_8() {
    let (elf, logs, instructions) = run_asm_elf("test_bitwise_8");
    assert_eq!(logs.len(), 8);
    let mut traces = Traces::from_logs_minimal(&logs, instructions.clone()).unwrap();
    // Using traces from Traces::from_logs() which includes MEMW LT ops
    assert!(
        prove_and_verify_vm_minimal(
            &mut traces.cpu,
            &mut traces.bitwise,
            &mut traces.lt,
            &mut traces.memw,
            &mut traces.load,
            &mut traces.decode,
            &mut traces.branch
        ),
        "test_bitwise_8 failed"
    );
}

#[test]
fn test_prove_elfs_test_slt_8() {
    // Initialize logger to see debug constraint validation output
    let _ = env_logger::builder().is_test(true).try_init();

    let (elf, logs, instructions) = run_asm_elf("test_slt_8");
    assert_eq!(logs.len(), 8);
    let mut traces = Traces::from_logs_minimal(&logs, instructions.clone()).unwrap();

    // Collect LT lookups first (needed for both LT trace and bitwise lookups)

    println!(
        "test_slt_8: CPU {} rows, Bitwise {} rows",
        traces.cpu.main_table.height, traces.bitwise.main_table.height,
    );
    assert!(
        prove_and_verify_vm_minimal(
            &mut traces.cpu,
            &mut traces.bitwise,
            &mut traces.lt,
            &mut traces.memw,
            &mut traces.load,
            &mut traces.decode,
            &mut traces.branch
        ),
        "test_slt_8 failed"
    );
}

// =========================================================================
// Comprehensive tests for all instructions
// =========================================================================

#[test]
fn test_prove_elfs_test_xor_8() {
    let (elf, logs, instructions) = run_asm_elf("test_xor_8");
    assert_eq!(logs.len(), 8);
    let mut traces = Traces::from_logs_minimal(&logs, instructions.clone()).unwrap();
    // Using traces from Traces::from_logs() which includes MEMW LT ops
    assert!(
        prove_and_verify_vm_minimal(
            &mut traces.cpu,
            &mut traces.bitwise,
            &mut traces.lt,
            &mut traces.memw,
            &mut traces.load,
            &mut traces.decode,
            &mut traces.branch
        ),
        "test_xor_8 failed"
    );
}

#[test]
fn test_prove_elfs_test_lb_lh_8() {
    let (elf, logs, instructions) = run_asm_elf("test_lb_lh_8");
    assert_eq!(logs.len(), 8);
    let mut traces = Traces::from_logs_minimal(&logs, instructions.clone()).unwrap();
    assert!(
        prove_and_verify_vm_minimal(
            &mut traces.cpu,
            &mut traces.bitwise,
            &mut traces.lt,
            &mut traces.memw,
            &mut traces.load,
            &mut traces.decode,
            &mut traces.branch
        ),
        "test_lb_lh_8 failed"
    );
}

#[test]
fn test_prove_elfs_test_sb_sh_8() {
    let (elf, logs, instructions) = run_asm_elf("test_sb_sh_8");
    assert_eq!(logs.len(), 8);
    let mut traces = Traces::from_logs_minimal(&logs, instructions.clone()).unwrap();
    // Using traces from Traces::from_logs() which includes MEMW LT ops
    assert!(
        prove_and_verify_vm_minimal(
            &mut traces.cpu,
            &mut traces.bitwise,
            &mut traces.lt,
            &mut traces.memw,
            &mut traces.load,
            &mut traces.decode,
            &mut traces.branch
        ),
        "test_sb_sh_8 failed"
    );
}

#[test]
fn test_prove_elfs_all_branches_16() {
    // Initialize logger to see debug constraint validation output
    let _ = env_logger::builder().is_test(true).try_init();

    let (elf, logs, instructions) = run_asm_elf("all_branches_16");
    assert_eq!(logs.len(), 16);
    let mut traces = Traces::from_logs_minimal(&logs, instructions.clone()).unwrap();

    // BLT instructions need LT table (like SLT)

    println!(
        "all_branches_16: CPU {} rows, Bitwise {} rows",
        traces.cpu.main_table.height, traces.bitwise.main_table.height,
    );
    assert!(
        prove_and_verify_vm_minimal(
            &mut traces.cpu,
            &mut traces.bitwise,
            &mut traces.lt,
            &mut traces.memw,
            &mut traces.load,
            &mut traces.decode,
            &mut traces.branch
        ),
        "all_branches_16 failed"
    );
}

#[test]
fn test_prove_elfs_all_loadstore_32() {
    let (elf, logs, instructions) = run_asm_elf("all_loadstore_32");
    assert_eq!(logs.len(), 32);
    // Use full Traces to get real MEMW and LOAD traces
    let mut traces = Traces::from_logs_minimal(&logs, instructions.clone()).unwrap();
    assert!(
        prove_and_verify_vm_minimal(
            &mut traces.cpu,
            &mut traces.bitwise,
            &mut traces.lt,
            &mut traces.memw,
            &mut traces.load,
            &mut traces.decode,
            &mut traces.branch
        ),
        "all_loadstore_32 failed"
    );
}

/// Fast version using minimal bitwise table for debugging
#[test]
fn test_prove_elfs_all_instructions_64() {
    let _ = env_logger::builder().is_test(true).try_init();

    let (elf, logs, instructions) = run_asm_elf("all_instructions_64");
    assert_eq!(logs.len(), 64);
    // Use full Traces to get real MEMW and LOAD traces
    let mut traces = Traces::from_logs_minimal(&logs, instructions.clone()).unwrap();

    // Includes SLT/SLTU instructions - need LT table

    println!(
        "all_instructions_64 (fast): CPU {} rows, Bitwise {} rows, MEMW {} rows, LOAD {} rows",
        traces.cpu.main_table.height,
        traces.bitwise.main_table.height,
        traces.memw.main_table.height,
        traces.load.main_table.height
    );
    assert!(
        prove_and_verify_vm_minimal(
            &mut traces.cpu,
            &mut traces.bitwise,
            &mut traces.lt,
            &mut traces.memw,
            &mut traces.load,
            &mut traces.decode,
            &mut traces.branch
        ),
        "all_instructions_64 failed"
    );
}

/// Slow version using full bitwise table (2^20 rows) - production-safe
///
/// This is the most comprehensive test covering all RV64IM instructions:
/// - 64-bit arithmetic: ADD, SUB, SLT, SLTU, SLTI, SLTIU, ADDI
/// - 32-bit word arithmetic: ADDW, SUBW, ADDIW
/// - 64-bit shifts: SLL, SRL, SRA, SLLI, SRLI, SRAI
/// - 32-bit word shifts: SLLW, SRLW, SRAW, SLLIW, SRLIW, SRAIW
/// - Bitwise: AND, OR, XOR, ANDI, ORI, XORI
/// - Multiplication: MUL, MULW, MULH, MULHU, MULHSU
/// - Division: DIV, DIVU, REM, REMU
/// - Control: LUI, AUIPC, JALR
#[test]
#[ignore] // Slow: run with `cargo test --ignored` or `make test-prover-all`
fn test_prove_elfs_all_instructions_64_full() {
    let _ = env_logger::builder().is_test(true).try_init();

    let (elf, logs, instructions) = run_asm_elf("all_instructions_64");
    assert_eq!(logs.len(), 64);
    // Use FULL bitwise table (2^20 rows) - this is the comprehensive test
    let mut traces = Traces::from_logs(&logs, instructions.clone()).unwrap();

    println!(
        "all_instructions_64_full: CPU {} rows, Bitwise {} rows (FULL)",
        traces.cpu.main_table.height, traces.bitwise.main_table.height,
    );

    assert!(
        prove_and_verify_vm(
            &mut traces.cpu,
            &mut traces.bitwise,
            &mut traces.lt,
            &mut traces.memw,
            &mut traces.load,
            &mut traces.decode,
            &mut traces.branch,
            &elf,
        ),
        "all_instructions_64_full failed - comprehensive test with full bitwise table"
    );
}

/// Memory profiling test using dhat.
///
/// Run with:
/// ```
/// cargo test -p prover --release --features dhat-heap test_dhat_memory_profile -- --ignored --nocapture
/// ```
///
/// This generates `dhat-heap.json` which can be viewed with:
/// https://nnethercote.github.io/dh_view/dh_view.html
#[test]
#[ignore]
fn test_dhat_memory_profile() {
    #[cfg(feature = "dhat-heap")]
    let _profiler = dhat::Profiler::new_heap();

    let program_name = "loop_4096";
    let (elf, logs, instructions) = run_asm_elf(program_name);

    // Output metadata for CI parsing
    println!("MEMORY_PROFILE_PROGRAM={}", program_name);
    println!("MEMORY_PROFILE_INSTRUCTIONS={}", logs.len());

    assert_eq!(logs.len(), 4096, "Expected 2^12 instructions");

    let mut traces = Traces::from_logs_minimal(&logs, instructions.clone()).unwrap();

    assert!(
        prove_and_verify_vm_minimal(
            &mut traces.cpu,
            &mut traces.bitwise,
            &mut traces.lt,
            &mut traces.memw,
            &mut traces.load,
            &mut traces.decode,
            &mut traces.branch
        ),
        "verification failed"
    );
}
