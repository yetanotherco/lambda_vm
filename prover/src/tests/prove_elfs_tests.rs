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

use crate::tables::bitwise::{generate_bitwise_trace, update_multiplicities};
use crate::tables::lt::generate_lt_trace;
use crate::tables::trace_builder::Traces;
use crate::tables::types::{GoldilocksExtension, GoldilocksField};

// Import shared utilities
use crate::test_utils::{
    collect_bitwise_lookups_from_logs, collect_bitwise_lookups_from_lt,
    collect_lt_lookups_from_logs, create_bitwise_air, create_cpu_air, create_lt_air,
    generate_minimal_bitwise_trace, run_asm_elf,
};

type F = GoldilocksField;
type E = GoldilocksExtension;

/// Alias for compatibility with existing test code.
fn collect_bitwise_lookups(
    logs: &[executor::vm::logs::Log],
    instructions: &executor::vm::memory::U64HashMap<
        executor::vm::instruction::decoding::Instruction,
    >,
) -> Vec<(crate::tables::bitwise::BitwiseLookup, u8, u8, u8)> {
    collect_bitwise_lookups_from_logs(logs, instructions)
}

// AIR creation helpers and lookup collection functions are now in test_utils module

// =============================================================================
// Prover test helpers
// =============================================================================

/// Run multi_prove and multi_verify for CPU + Bitwise + LT tables.
///
/// Returns true if verification succeeds.
fn prove_and_verify_vm_with_lt(
    cpu_trace: &mut TraceTable<F, E>,
    bitwise_trace: &mut TraceTable<F, E>,
    lt_trace: &mut TraceTable<F, E>,
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

/// Run multi_prove and multi_verify for CPU + Bitwise tables.
///
/// Returns true if verification succeeds.
fn prove_and_verify_vm(
    cpu_trace: &mut TraceTable<F, E>,
    bitwise_trace: &mut TraceTable<F, E>,
) -> bool {
    let proof_options = ProofOptions::default_test_options();

    let cpu_air = create_cpu_air(&proof_options);
    let bitwise_air = create_bitwise_air(&proof_options);

    let air_trace_pairs: Vec<(
        &dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>,
        _,
        _,
    )> = vec![
        (&cpu_air, cpu_trace, &()),
        (&bitwise_air, bitwise_trace, &()),
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
        vec![&cpu_air, &bitwise_air];

    Verifier::multi_verify(&airs, &multi_proof, &mut DefaultTranscript::<E>::new(&[]))
}

// =============================================================================
// Integration tests
// =============================================================================

/// Test CPU table alone (no bus interactions) to verify basic prove/verify works.
#[test]
fn test_cpu_only_no_bus() {
    let (logs, instructions) = run_asm_elf("sub");
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
    let (logs, instructions) = run_asm_elf("sub");
    assert_eq!(logs.len(), 4, "sub.elf should have 4 steps");

    let mut cpu_trace = Traces::from_logs(&logs, instructions.clone()).unwrap().cpu;
    let bitwise_lookups = collect_bitwise_lookups(&logs, &instructions);
    let mut bitwise_trace = generate_minimal_bitwise_trace(&bitwise_lookups);

    println!(
        "Fast SUB: CPU {} rows, Bitwise {} rows (minimal), {} lookups",
        cpu_trace.main_table.height,
        bitwise_trace.main_table.height,
        bitwise_lookups.len()
    );

    assert!(
        prove_and_verify_vm(&mut cpu_trace, &mut bitwise_trace),
        "Proof verification failed for sub program (fast)"
    );
}

#[test]
fn test_prove_elfs_sub_neg_result_fast() {
    let (logs, instructions) = run_asm_elf("sub_neg_result");
    assert_eq!(logs.len(), 4, "sub_neg_result.elf should have 4 steps");

    let mut cpu_trace = Traces::from_logs(&logs, instructions.clone()).unwrap().cpu;
    let bitwise_lookups = collect_bitwise_lookups(&logs, &instructions);
    let mut bitwise_trace = generate_minimal_bitwise_trace(&bitwise_lookups);

    println!(
        "Fast SUB_NEG: CPU {} rows, Bitwise {} rows (minimal), {} lookups",
        cpu_trace.main_table.height,
        bitwise_trace.main_table.height,
        bitwise_lookups.len()
    );

    assert!(
        prove_and_verify_vm(&mut cpu_trace, &mut bitwise_trace),
        "Proof verification failed for sub_neg_result program (fast)"
    );
}

#[test]
fn test_prove_elfs_sub_underflow_fast() {
    let (logs, instructions) = run_asm_elf("sub_underflow");
    assert_eq!(logs.len(), 4, "sub_underflow.elf should have 4 steps");

    let mut cpu_trace = Traces::from_logs(&logs, instructions.clone()).unwrap().cpu;
    let bitwise_lookups = collect_bitwise_lookups(&logs, &instructions);
    let mut bitwise_trace = generate_minimal_bitwise_trace(&bitwise_lookups);

    println!(
        "Fast SUB_UNDERFLOW: CPU {} rows, Bitwise {} rows (minimal), {} lookups",
        cpu_trace.main_table.height,
        bitwise_trace.main_table.height,
        bitwise_lookups.len()
    );

    assert!(
        prove_and_verify_vm(&mut cpu_trace, &mut bitwise_trace),
        "Proof verification failed for sub_underflow program (fast)"
    );
}

#[test]
fn test_prove_elfs_subw_fast() {
    let (logs, instructions) = run_asm_elf("subw");
    assert_eq!(logs.len(), 4, "subw.elf should have 4 steps");

    let mut cpu_trace = Traces::from_logs(&logs, instructions.clone()).unwrap().cpu;
    let bitwise_lookups = collect_bitwise_lookups(&logs, &instructions);
    let mut bitwise_trace = generate_minimal_bitwise_trace(&bitwise_lookups);

    println!(
        "Fast SUBW: CPU {} rows, Bitwise {} rows (minimal), {} lookups",
        cpu_trace.main_table.height,
        bitwise_trace.main_table.height,
        bitwise_lookups.len()
    );

    assert!(
        prove_and_verify_vm(&mut cpu_trace, &mut bitwise_trace),
        "Proof verification failed for subw program (fast)"
    );
}

/// 8-instruction test with LUI
#[test]
fn test_prove_elfs_arith_lui_8() {
    let (logs, instructions) = run_asm_elf("arith_lui_8");
    assert_eq!(logs.len(), 8, "arith_lui_8.elf should have 8 steps");

    let mut cpu_trace = Traces::from_logs(&logs, instructions.clone()).unwrap().cpu;
    let bitwise_lookups = collect_bitwise_lookups(&logs, &instructions);
    let mut bitwise_trace = generate_minimal_bitwise_trace(&bitwise_lookups);

    println!(
        "ArithLUI8: CPU {} rows, Bitwise {} rows (minimal), {} lookups",
        cpu_trace.main_table.height,
        bitwise_trace.main_table.height,
        bitwise_lookups.len()
    );

    assert!(
        prove_and_verify_vm(&mut cpu_trace, &mut bitwise_trace),
        "Proof verification failed for arith_lui_8 program"
    );
}

/// 8-instruction test with ADD, SUB, ADDW, SUBW
#[test]
fn test_prove_elfs_arith_8() {
    let (logs, instructions) = run_asm_elf("arith_8");
    assert_eq!(logs.len(), 8, "arith_8.elf should have 8 steps");

    let mut cpu_trace = Traces::from_logs(&logs, instructions.clone()).unwrap().cpu;
    let bitwise_lookups = collect_bitwise_lookups(&logs, &instructions);
    let mut bitwise_trace = generate_minimal_bitwise_trace(&bitwise_lookups);

    println!(
        "Arith8: CPU {} rows, Bitwise {} rows (minimal), {} lookups",
        cpu_trace.main_table.height,
        bitwise_trace.main_table.height,
        bitwise_lookups.len()
    );

    assert!(
        prove_and_verify_vm(&mut cpu_trace, &mut bitwise_trace),
        "Proof verification failed for arith_8 program"
    );
}

/// Basic arithmetic test with 32 instructions covering:
/// - 64-bit ADD with positive, negative, and edge cases
/// - 64-bit SUB with underflow, negative results
/// - 32-bit ADDW/SUBW with sign extension
#[test]
fn test_prove_elfs_basic_arith_32() {
    let (logs, instructions) = run_asm_elf("basic_arith_32");
    assert_eq!(logs.len(), 32, "basic_arith_32.elf should have 32 steps");

    let mut cpu_trace = Traces::from_logs(&logs, instructions.clone()).unwrap().cpu;
    let bitwise_lookups = collect_bitwise_lookups(&logs, &instructions);
    let mut bitwise_trace = generate_minimal_bitwise_trace(&bitwise_lookups);

    println!(
        "BasicArith32: CPU {} rows, Bitwise {} rows (minimal), {} lookups",
        cpu_trace.main_table.height,
        bitwise_trace.main_table.height,
        bitwise_lookups.len()
    );

    assert!(
        prove_and_verify_vm(&mut cpu_trace, &mut bitwise_trace),
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

    let (logs, instructions) = run_asm_elf("comprehensive_test");
    assert_eq!(
        logs.len(),
        32,
        "comprehensive_test.elf should have 32 steps"
    );

    let mut cpu_trace = Traces::from_logs(&logs, instructions.clone()).unwrap().cpu;

    // Collect LT lookups first (needed for both LT trace and bitwise lookups)
    let lt_lookups = collect_lt_lookups_from_logs(&logs, &instructions);
    let mut lt_trace = generate_lt_trace(&lt_lookups);

    // Collect ALL bitwise lookups: from CPU + from LT table
    let mut bitwise_lookups = collect_bitwise_lookups(&logs, &instructions);
    let lt_bitwise_lookups = collect_bitwise_lookups_from_lt(&lt_lookups);
    bitwise_lookups.extend(lt_bitwise_lookups);
    let mut bitwise_trace = generate_minimal_bitwise_trace(&bitwise_lookups);

    println!(
        "Comprehensive: CPU {} rows, Bitwise {} rows (minimal), {} bitwise lookups, {} lt lookups",
        cpu_trace.main_table.height,
        bitwise_trace.main_table.height,
        bitwise_lookups.len(),
        lt_lookups.len()
    );

    assert!(
        prove_and_verify_vm_with_lt(&mut cpu_trace, &mut bitwise_trace, &mut lt_trace),
        "Proof verification failed for comprehensive_test program"
    );
}

// =============================================================================
// Instruction-specific 8-step tests
// =============================================================================

#[test]
fn test_prove_elfs_test_add_8() {
    let (logs, instructions) = run_asm_elf("test_add_8");
    assert_eq!(logs.len(), 8);
    let mut cpu_trace = Traces::from_logs(&logs, instructions.clone()).unwrap().cpu;
    let bitwise_lookups = collect_bitwise_lookups(&logs, &instructions);
    let mut bitwise_trace = generate_minimal_bitwise_trace(&bitwise_lookups);
    println!("test_add_8: {} lookups", bitwise_lookups.len());
    assert!(
        prove_and_verify_vm(&mut cpu_trace, &mut bitwise_trace),
        "test_add_8 failed"
    );
}

#[test]
fn test_prove_elfs_test_sub_8() {
    let (logs, instructions) = run_asm_elf("test_sub_8");
    assert_eq!(logs.len(), 8);
    let mut cpu_trace = Traces::from_logs(&logs, instructions.clone()).unwrap().cpu;
    let bitwise_lookups = collect_bitwise_lookups(&logs, &instructions);
    let mut bitwise_trace = generate_minimal_bitwise_trace(&bitwise_lookups);
    println!("test_sub_8: {} lookups", bitwise_lookups.len());
    assert!(
        prove_and_verify_vm(&mut cpu_trace, &mut bitwise_trace),
        "test_sub_8 failed"
    );
}

#[test]
fn test_prove_elfs_test_addw_8() {
    let (logs, instructions) = run_asm_elf("test_addw_8");
    assert_eq!(logs.len(), 8);
    let mut cpu_trace = Traces::from_logs(&logs, instructions.clone()).unwrap().cpu;
    let bitwise_lookups = collect_bitwise_lookups(&logs, &instructions);
    let mut bitwise_trace = generate_minimal_bitwise_trace(&bitwise_lookups);
    println!("test_addw_8: {} lookups", bitwise_lookups.len());
    assert!(
        prove_and_verify_vm(&mut cpu_trace, &mut bitwise_trace),
        "test_addw_8 failed"
    );
}

#[test]
fn test_prove_elfs_test_subw_8() {
    let (logs, instructions) = run_asm_elf("test_subw_8");
    assert_eq!(logs.len(), 8);
    let mut cpu_trace = Traces::from_logs(&logs, instructions.clone()).unwrap().cpu;
    let bitwise_lookups = collect_bitwise_lookups(&logs, &instructions);
    let mut bitwise_trace = generate_minimal_bitwise_trace(&bitwise_lookups);
    println!("test_subw_8: {} lookups", bitwise_lookups.len());
    assert!(
        prove_and_verify_vm(&mut cpu_trace, &mut bitwise_trace),
        "test_subw_8 failed"
    );
}

#[test]
fn test_prove_elfs_test_addw_lui_8() {
    let (logs, instructions) = run_asm_elf("test_addw_lui_8");
    assert_eq!(logs.len(), 8);
    let mut cpu_trace = Traces::from_logs(&logs, instructions.clone()).unwrap().cpu;
    let bitwise_lookups = collect_bitwise_lookups(&logs, &instructions);
    let mut bitwise_trace = generate_minimal_bitwise_trace(&bitwise_lookups);
    println!("test_addw_lui_8: {} lookups", bitwise_lookups.len());
    assert!(
        prove_and_verify_vm(&mut cpu_trace, &mut bitwise_trace),
        "test_addw_lui_8 failed"
    );
}

#[test]
fn test_prove_elfs_test_subw_lui_8() {
    let (logs, instructions) = run_asm_elf("test_subw_lui_8");
    assert_eq!(logs.len(), 8);
    let mut cpu_trace = Traces::from_logs(&logs, instructions.clone()).unwrap().cpu;
    let bitwise_lookups = collect_bitwise_lookups(&logs, &instructions);
    let mut bitwise_trace = generate_minimal_bitwise_trace(&bitwise_lookups);
    println!("test_subw_lui_8: {} lookups", bitwise_lookups.len());
    assert!(
        prove_and_verify_vm(&mut cpu_trace, &mut bitwise_trace),
        "test_subw_lui_8 failed"
    );
}

#[test]
fn test_prove_elfs_test_add_neg_8() {
    let (logs, instructions) = run_asm_elf("test_add_neg_8");
    assert_eq!(logs.len(), 8);
    let mut cpu_trace = Traces::from_logs(&logs, instructions.clone()).unwrap().cpu;
    let bitwise_lookups = collect_bitwise_lookups(&logs, &instructions);
    let mut bitwise_trace = generate_minimal_bitwise_trace(&bitwise_lookups);
    println!("test_add_neg_8: {} lookups", bitwise_lookups.len());
    assert!(
        prove_and_verify_vm(&mut cpu_trace, &mut bitwise_trace),
        "test_add_neg_8 failed"
    );
}

#[test]
fn test_prove_elfs_test_sub_neg_8() {
    let (logs, instructions) = run_asm_elf("test_sub_neg_8");
    assert_eq!(logs.len(), 8);
    let mut cpu_trace = Traces::from_logs(&logs, instructions.clone()).unwrap().cpu;
    let bitwise_lookups = collect_bitwise_lookups(&logs, &instructions);
    let mut bitwise_trace = generate_minimal_bitwise_trace(&bitwise_lookups);
    println!("test_sub_neg_8: {} lookups", bitwise_lookups.len());
    assert!(
        prove_and_verify_vm(&mut cpu_trace, &mut bitwise_trace),
        "test_sub_neg_8 failed"
    );
}

#[test]
fn test_prove_elfs_test_mul_8() {
    let (logs, instructions) = run_asm_elf("test_mul_8");
    assert_eq!(logs.len(), 8);
    let mut cpu_trace = Traces::from_logs(&logs, instructions.clone()).unwrap().cpu;
    let bitwise_lookups = collect_bitwise_lookups(&logs, &instructions);
    let mut bitwise_trace = generate_minimal_bitwise_trace(&bitwise_lookups);
    println!("test_mul_8: {} lookups", bitwise_lookups.len());
    assert!(
        prove_and_verify_vm(&mut cpu_trace, &mut bitwise_trace),
        "test_mul_8 failed"
    );
}

#[test]
fn test_prove_elfs_test_div_8() {
    let (logs, instructions) = run_asm_elf("test_div_8");
    assert_eq!(logs.len(), 8);
    let mut cpu_trace = Traces::from_logs(&logs, instructions.clone()).unwrap().cpu;
    let bitwise_lookups = collect_bitwise_lookups(&logs, &instructions);
    let mut bitwise_trace = generate_minimal_bitwise_trace(&bitwise_lookups);
    println!("test_div_8: {} lookups", bitwise_lookups.len());
    assert!(
        prove_and_verify_vm(&mut cpu_trace, &mut bitwise_trace),
        "test_div_8 failed"
    );
}

#[test]
fn test_prove_elfs_test_shift_8() {
    let (logs, instructions) = run_asm_elf("test_shift_8");
    assert_eq!(logs.len(), 8);
    let mut cpu_trace = Traces::from_logs(&logs, instructions.clone()).unwrap().cpu;
    let bitwise_lookups = collect_bitwise_lookups(&logs, &instructions);
    let mut bitwise_trace = generate_minimal_bitwise_trace(&bitwise_lookups);
    println!("test_shift_8: {} lookups", bitwise_lookups.len());
    assert!(
        prove_and_verify_vm(&mut cpu_trace, &mut bitwise_trace),
        "test_shift_8 failed"
    );
}

#[test]
fn test_prove_elfs_test_bitwise_8() {
    let (logs, instructions) = run_asm_elf("test_bitwise_8");
    assert_eq!(logs.len(), 8);
    let mut cpu_trace = Traces::from_logs(&logs, instructions.clone()).unwrap().cpu;
    let bitwise_lookups = collect_bitwise_lookups(&logs, &instructions);
    let mut bitwise_trace = generate_minimal_bitwise_trace(&bitwise_lookups);
    println!("test_bitwise_8: {} lookups", bitwise_lookups.len());
    assert!(
        prove_and_verify_vm(&mut cpu_trace, &mut bitwise_trace),
        "test_bitwise_8 failed"
    );
}

#[test]
fn test_prove_elfs_test_slt_8() {
    // Initialize logger to see debug constraint validation output
    let _ = env_logger::builder().is_test(true).try_init();

    let (logs, instructions) = run_asm_elf("test_slt_8");
    assert_eq!(logs.len(), 8);
    let mut cpu_trace = Traces::from_logs(&logs, instructions.clone()).unwrap().cpu;

    // Collect LT lookups first (needed for both LT trace and bitwise lookups)
    let lt_lookups = collect_lt_lookups_from_logs(&logs, &instructions);
    let mut lt_trace = generate_lt_trace(&lt_lookups);

    // Collect ALL bitwise lookups: from CPU + from LT table
    let mut bitwise_lookups = collect_bitwise_lookups(&logs, &instructions);
    let lt_bitwise_lookups = collect_bitwise_lookups_from_lt(&lt_lookups);
    bitwise_lookups.extend(lt_bitwise_lookups);
    let mut bitwise_trace = generate_minimal_bitwise_trace(&bitwise_lookups);

    println!(
        "test_slt_8: {} bitwise lookups, {} lt lookups",
        bitwise_lookups.len(),
        lt_lookups.len()
    );
    assert!(
        prove_and_verify_vm_with_lt(&mut cpu_trace, &mut bitwise_trace, &mut lt_trace),
        "test_slt_8 failed"
    );
}

// =========================================================================
// Comprehensive tests for all instructions
// =========================================================================

#[test]
fn test_prove_elfs_test_xor_8() {
    let (logs, instructions) = run_asm_elf("test_xor_8");
    assert_eq!(logs.len(), 8);
    let mut cpu_trace = Traces::from_logs(&logs, instructions.clone()).unwrap().cpu;
    let bitwise_lookups = collect_bitwise_lookups(&logs, &instructions);
    let mut bitwise_trace = generate_minimal_bitwise_trace(&bitwise_lookups);
    println!("test_xor_8: {} lookups", bitwise_lookups.len());
    assert!(
        prove_and_verify_vm(&mut cpu_trace, &mut bitwise_trace),
        "test_xor_8 failed"
    );
}

#[test]
fn test_prove_elfs_test_lb_lh_8() {
    let (logs, instructions) = run_asm_elf("test_lb_lh_8");
    assert_eq!(logs.len(), 8);
    let mut cpu_trace = Traces::from_logs(&logs, instructions.clone()).unwrap().cpu;
    let bitwise_lookups = collect_bitwise_lookups(&logs, &instructions);
    let mut bitwise_trace = generate_minimal_bitwise_trace(&bitwise_lookups);
    println!("test_lb_lh_8: {} lookups", bitwise_lookups.len());
    assert!(
        prove_and_verify_vm(&mut cpu_trace, &mut bitwise_trace),
        "test_lb_lh_8 failed"
    );
}

#[test]
fn test_prove_elfs_test_sb_sh_8() {
    let (logs, instructions) = run_asm_elf("test_sb_sh_8");
    assert_eq!(logs.len(), 8);
    let mut cpu_trace = Traces::from_logs(&logs, instructions.clone()).unwrap().cpu;
    let bitwise_lookups = collect_bitwise_lookups(&logs, &instructions);
    let mut bitwise_trace = generate_minimal_bitwise_trace(&bitwise_lookups);
    println!("test_sb_sh_8: {} lookups", bitwise_lookups.len());
    assert!(
        prove_and_verify_vm(&mut cpu_trace, &mut bitwise_trace),
        "test_sb_sh_8 failed"
    );
}

#[test]
fn test_prove_elfs_all_branches_16() {
    // Initialize logger to see debug constraint validation output
    let _ = env_logger::builder().is_test(true).try_init();

    let (logs, instructions) = run_asm_elf("all_branches_16");
    assert_eq!(logs.len(), 16);
    let mut cpu_trace = Traces::from_logs(&logs, instructions.clone()).unwrap().cpu;

    // BLT instructions need LT table (like SLT)
    let lt_lookups = collect_lt_lookups_from_logs(&logs, &instructions);
    let mut lt_trace = generate_lt_trace(&lt_lookups);

    // Collect ALL bitwise lookups: from CPU + from LT table
    let mut bitwise_lookups = collect_bitwise_lookups(&logs, &instructions);
    let lt_bitwise_lookups = collect_bitwise_lookups_from_lt(&lt_lookups);
    bitwise_lookups.extend(lt_bitwise_lookups);
    let mut bitwise_trace = generate_minimal_bitwise_trace(&bitwise_lookups);

    println!(
        "all_branches_16: {} bitwise lookups, {} lt lookups",
        bitwise_lookups.len(),
        lt_lookups.len()
    );
    assert!(
        prove_and_verify_vm_with_lt(&mut cpu_trace, &mut bitwise_trace, &mut lt_trace),
        "all_branches_16 failed"
    );
}

#[test]
fn test_prove_elfs_all_loadstore_32() {
    let (logs, instructions) = run_asm_elf("all_loadstore_32");
    assert_eq!(logs.len(), 32);
    let mut cpu_trace = Traces::from_logs(&logs, instructions.clone()).unwrap().cpu;
    let bitwise_lookups = collect_bitwise_lookups(&logs, &instructions);
    let mut bitwise_trace = generate_minimal_bitwise_trace(&bitwise_lookups);
    println!("all_loadstore_32: {} lookups", bitwise_lookups.len());
    assert!(
        prove_and_verify_vm(&mut cpu_trace, &mut bitwise_trace),
        "all_loadstore_32 failed"
    );
}

/// Fast version using minimal bitwise table for debugging
#[test]
fn test_prove_elfs_all_instructions_64() {
    let _ = env_logger::builder().is_test(true).try_init();

    let (logs, instructions) = run_asm_elf("all_instructions_64");
    assert_eq!(logs.len(), 64);
    let mut cpu_trace = Traces::from_logs(&logs, instructions.clone()).unwrap().cpu;

    // Includes SLT/SLTU instructions - need LT table
    let lt_lookups = collect_lt_lookups_from_logs(&logs, &instructions);
    let mut lt_trace = generate_lt_trace(&lt_lookups);

    // Collect ALL bitwise lookups: from CPU + from LT table
    // Using minimal bitwise trace for fast debugging
    let mut bitwise_lookups = collect_bitwise_lookups(&logs, &instructions);
    let lt_bitwise_lookups = collect_bitwise_lookups_from_lt(&lt_lookups);
    bitwise_lookups.extend(lt_bitwise_lookups);
    let mut bitwise_trace = generate_minimal_bitwise_trace(&bitwise_lookups);

    println!(
        "all_instructions_64 (fast): {} bitwise lookups, {} lt lookups",
        bitwise_lookups.len(),
        lt_lookups.len()
    );
    assert!(
        prove_and_verify_vm_with_lt(&mut cpu_trace, &mut bitwise_trace, &mut lt_trace),
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

    let (logs, instructions) = run_asm_elf("all_instructions_64");
    assert_eq!(logs.len(), 64);
    let mut cpu_trace = Traces::from_logs(&logs, instructions.clone()).unwrap().cpu;

    // Includes SLT/SLTU instructions - need LT table
    let lt_lookups = collect_lt_lookups_from_logs(&logs, &instructions);
    let mut lt_trace = generate_lt_trace(&lt_lookups);

    // Collect ALL bitwise lookups: from CPU + from LT table
    let mut bitwise_lookups = collect_bitwise_lookups(&logs, &instructions);
    let lt_bitwise_lookups = collect_bitwise_lookups_from_lt(&lt_lookups);
    bitwise_lookups.extend(lt_bitwise_lookups);

    // Use FULL bitwise table (2^20 rows) for production-safe proof
    let mut bitwise_trace = generate_bitwise_trace();
    update_multiplicities(&mut bitwise_trace, &bitwise_lookups);

    println!(
        "all_instructions_64 (full): CPU {} rows, Bitwise {} rows, {} lookups, {} lt lookups",
        cpu_trace.main_table.height,
        bitwise_trace.main_table.height,
        bitwise_lookups.len(),
        lt_lookups.len()
    );

    assert!(
        prove_and_verify_vm_with_lt(&mut cpu_trace, &mut bitwise_trace, &mut lt_trace),
        "all_instructions_64 (full) failed"
    );
}

/// Edge case test: arg2 sign extension for signed word instructions (DIVW, REMW)
///
/// Tests the Arg2UpperConstraint:
///   arg2[4:] = (1-STORE-LOAD) * ((1-word_instr)*rv2[2] + signed*arg2_sign_bit*(2^32-1)) + ...
///
/// When word_instr=1, signed=1, and arg2_sign_bit=1 (bit 31 of arg2 is set),
/// arg2[4:7] should be 0xFFFFFFFF (sign-extended).
///
/// This case was NOT triggered by all_instructions_64.s because:
/// - SRAW/SRAIW use small positive shift amounts (arg2 bit 31 = 0)
/// - This test uses DIVW/REMW with a negative divisor (arg2 bit 31 = 1)
#[test]
fn test_prove_elfs_sign_ext_edge_cases_8() {
    let (logs, instructions) = run_asm_elf("sign_ext_edge_cases_8");
    assert_eq!(
        logs.len(),
        8,
        "sign_ext_edge_cases_8.elf should have 8 steps"
    );

    let mut cpu_trace = Traces::from_logs(&logs, instructions.clone()).unwrap().cpu;
    let bitwise_lookups = collect_bitwise_lookups(&logs, &instructions);
    let mut bitwise_trace = generate_minimal_bitwise_trace(&bitwise_lookups);

    println!(
        "sign_ext_edge_cases_8: CPU {} rows, {} bitwise lookups",
        cpu_trace.main_table.height,
        bitwise_lookups.len()
    );

    assert!(
        prove_and_verify_vm(&mut cpu_trace, &mut bitwise_trace),
        "sign_ext_edge_cases_8 failed - arg2 sign extension may be broken"
    );
}
