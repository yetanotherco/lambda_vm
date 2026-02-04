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
use crate::tables::register::{self, FinalRegisterStateMap};
use crate::tables::trace_builder::Traces;
use crate::tables::types::{GoldilocksExtension, GoldilocksField};

// Import shared utilities
use crate::test_utils::{
    create_bitwise_air, create_branch_air, create_cpu_air, create_decode_air, create_halt_air,
    create_load_air, create_lt_air, create_memw_air, create_page_air, create_register_air,
    run_asm_elf,
};
use crate::tables::page::PageConfig;

type F = GoldilocksField;
type E = GoldilocksExtension;

// =============================================================================
// Prover test helpers
// =============================================================================

/// Run multi_prove and multi_verify for all VM tables.
/// Run multi_prove and multi_verify for all VM tables (CPU + Bitwise + LT + MEMW + LOAD + DECODE + HALT).
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
    halt_trace: &mut TraceTable<F, E>,
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
        decode::commitment_from_elf(elf, &proof_options)
            .expect("Failed to compute decode commitment"),
        decode::NUM_PRECOMPUTED_COLS,
    );
    let branch_air = create_branch_air(&proof_options);
    let halt_air = create_halt_air(&proof_options);

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
        (&halt_air, halt_trace, &()),
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
        &halt_air,
    ];

    let result = Verifier::multi_verify(&airs, &multi_proof, &mut DefaultTranscript::<E>::new(&[]));
    if !result {
        eprintln!("Verifier failed!");
    }
    result
}

/// Run multi_prove and multi_verify for all VM tables.
/// Run multi_prove and multi_verify for all VM tables (CPU + Bitwise + LT + MEMW + LOAD + DECODE + HALT + REGISTER).
///
/// Used for fast tests where the bitwise table is a dummy that only contains
/// the rows needed to balance the bus. NOT the full preprocessed table.
///
/// Now includes REGISTER table for Memory bus register token support.
fn prove_and_verify_vm_minimal(
    cpu_trace: &mut TraceTable<F, E>,
    bitwise_trace: &mut TraceTable<F, E>,
    lt_trace: &mut TraceTable<F, E>,
    memw_trace: &mut TraceTable<F, E>,
    load_trace: &mut TraceTable<F, E>,
    decode_trace: &mut TraceTable<F, E>,
    branch_trace: &mut TraceTable<F, E>,
    halt_trace: &mut TraceTable<F, E>,
    register_trace: &mut TraceTable<F, E>,
) -> bool {
    // Call the version with empty PAGE tables (for tests without memory PAGE bus)
    prove_and_verify_vm_with_pages(
        cpu_trace,
        bitwise_trace,
        lt_trace,
        memw_trace,
        load_trace,
        decode_trace,
        branch_trace,
        halt_trace,
        register_trace,
        &mut Vec::new(),
        &[],
    )
}

/// Run multi_prove and multi_verify including PAGE and REGISTER tables for Memory bus.
///
/// This version accepts PAGE traces and configs, plus REGISTER trace for full Memory bus support.
fn prove_and_verify_vm_with_pages(
    cpu_trace: &mut TraceTable<F, E>,
    bitwise_trace: &mut TraceTable<F, E>,
    lt_trace: &mut TraceTable<F, E>,
    memw_trace: &mut TraceTable<F, E>,
    load_trace: &mut TraceTable<F, E>,
    decode_trace: &mut TraceTable<F, E>,
    branch_trace: &mut TraceTable<F, E>,
    halt_trace: &mut TraceTable<F, E>,
    register_trace: &mut TraceTable<F, E>,
    page_traces: &mut Vec<TraceTable<F, E>>,
    page_configs: &[PageConfig],
) -> bool {
    let proof_options = ProofOptions::default_test_options();

    let cpu_air = create_cpu_air(&proof_options);
    let bitwise_air = create_bitwise_air(&proof_options);
    let lt_air = create_lt_air(&proof_options);
    let memw_air = create_memw_air(&proof_options);
    let load_air = create_load_air(&proof_options);
    let decode_air = create_decode_air(&proof_options);
    let branch_air = create_branch_air(&proof_options);
    let halt_air = create_halt_air(&proof_options);
    let register_air = create_register_air(&proof_options);

    // Create PAGE AIRs (one per page, each with its own page_base)
    let page_airs: Vec<_> = page_configs
        .iter()
        .map(|config| create_page_air(&proof_options, config.page_base))
        .collect();

    // Build air_trace_pairs for core tables
    let mut air_trace_pairs: Vec<(
        &dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>,
        &mut TraceTable<F, E>,
        &(),
    )> = vec![
        (&cpu_air, cpu_trace, &()),
        (&bitwise_air, bitwise_trace, &()),
        (&lt_air, lt_trace, &()),
        (&memw_air, memw_trace, &()),
        (&load_air, load_trace, &()),
        (&decode_air, decode_trace, &()),
        (&branch_air, branch_trace, &()),
        (&halt_air, halt_trace, &()),
        (&register_air, register_trace, &()),
    ];

    // Add PAGE table pairs
    for (i, page_trace) in page_traces.iter_mut().enumerate() {
        air_trace_pairs.push((&page_airs[i], page_trace, &()));
    }

    let multi_proof =
        match Prover::multi_prove(air_trace_pairs, &mut DefaultTranscript::<E>::new(&[])) {
            Ok(proof) => proof,
            Err(e) => {
                eprintln!("Prover error: {:?}", e);
                return false;
            }
        };

    // Debug: Print bus_public_inputs for each table
    let table_names = ["CPU", "Bitwise", "LT", "MEMW", "LOAD", "DECODE", "BRANCH", "HALT", "REGISTER"];
    println!("\n=== Bus Public Inputs (final_accumulated values) ===");
    let mut total = math::field::element::FieldElement::<E>::zero();
    for (i, proof) in multi_proof.proofs.iter().enumerate() {
        let name = if i < table_names.len() {
            table_names[i]
        } else {
            "PAGE"
        };
        if let Some(bus_inputs) = &proof.bus_public_inputs {
            println!("{:8}: final_accumulated = {:?}", name, bus_inputs.final_accumulated);
            total = total + &bus_inputs.final_accumulated;
        } else {
            println!("{:8}: no bus interactions", name);
        }
    }
    println!("TOTAL: {:?}", total);
    println!("=== End Bus Public Inputs ===\n");

    // Build airs list for verification
    let mut airs: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>> = vec![
        &cpu_air,
        &bitwise_air,
        &lt_air,
        &memw_air,
        &load_air,
        &decode_air,
        &branch_air,
        &halt_air,
        &register_air,
    ];

    // Add PAGE AIRs
    for page_air in &page_airs {
        airs.push(page_air);
    }

    Verifier::multi_verify(&airs, &multi_proof, &mut DefaultTranscript::<E>::new(&[]))
}

// =============================================================================
// Integration tests
// =============================================================================

/// Test CPU table alone (no bus interactions) to verify basic prove/verify works.
#[test]
fn test_cpu_only_no_bus() {
    let (_elf, logs, instructions) = run_asm_elf("sub");

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
#[ignore] // TODO: Re-enable when Memory bus (REGISTER + PAGE) is fully implemented
fn test_prove_elfs_sub_fast() {
    let _ = env_logger::builder().is_test(true).try_init();
    let (elf, logs, _instructions) = run_asm_elf("sub");
    // Use from_elf_and_logs to get PAGE and REGISTER tables for Memory bus
    let mut traces = Traces::from_elf_and_logs(&elf, &logs).unwrap();

    assert!(
        prove_and_verify_vm_with_pages(
            &mut traces.cpu,
            &mut traces.bitwise,
            &mut traces.lt,
            &mut traces.memw,
            &mut traces.load,
            &mut traces.decode,
            &mut traces.branch,
            &mut traces.halt,
            &mut traces.register,
            &mut traces.pages,
            &traces.page_configs,
        ),
        "Proof verification failed for sub program (fast)"
    );
}

#[test]
fn test_prove_elfs_sub_neg_result_fast() {
    let (_elf, logs, instructions) = run_asm_elf("sub_neg_result");
    let mut traces = Traces::from_logs_minimal(&logs, instructions.clone()).unwrap();

    println!(
        "Fast SUB_NEG: CPU {} rows, Bitwise {} rows, MEMW {} rows, REGISTER {} rows",
        traces.cpu.main_table.height, traces.bitwise.main_table.height,
        traces.memw.main_table.height, traces.register.main_table.height,
    );

    assert!(
        prove_and_verify_vm_minimal(
            &mut traces.cpu,
            &mut traces.bitwise,
            &mut traces.lt,
            &mut traces.memw,
            &mut traces.load,
            &mut traces.decode,
            &mut traces.branch,
            &mut traces.halt,
            &mut traces.register
        ),
        "Proof verification failed for sub_neg_result program (fast)"
    );
}

#[test]
fn test_prove_elfs_sub_underflow_fast() {
    let (_elf, logs, instructions) = run_asm_elf("sub_underflow");
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
            &mut traces.branch,
            &mut traces.halt,
            &mut traces.register
        ),
        "Proof verification failed for sub_underflow program (fast)"
    );
}

#[test]
fn test_prove_elfs_subw_fast() {
    let (_elf, logs, instructions) = run_asm_elf("subw");
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
            &mut traces.branch,
            &mut traces.halt,
            &mut traces.register
        ),
        "Proof verification failed for subw program (fast)"
    );
}

/// 8-instruction test with LUI
#[test]
fn test_prove_elfs_arith_lui_8() {
    let (_elf, logs, instructions) = run_asm_elf("arith_lui_8");
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
            &mut traces.branch,
            &mut traces.halt,
            &mut traces.register
        ),
        "Proof verification failed for arith_lui_8 program"
    );
}

/// 8-instruction test with ADD, SUB, ADDW, SUBW
#[test]
fn test_prove_elfs_arith_8() {
    let (_elf, logs, instructions) = run_asm_elf("arith_8");
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
            &mut traces.branch,
            &mut traces.halt,
            &mut traces.register
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
    let (_elf, logs, instructions) = run_asm_elf("basic_arith_32");
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
            &mut traces.branch,
            &mut traces.halt,
            &mut traces.register
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

    let (_elf, logs, instructions) = run_asm_elf("comprehensive_test");
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
            &mut traces.branch,
            &mut traces.halt,
            &mut traces.register
        ),
        "Proof verification failed for comprehensive_test program"
    );
}

// =============================================================================
// Instruction-specific 8-step tests
// =============================================================================

#[test]
fn test_prove_elfs_test_add_8() {
    let (_elf, logs, instructions) = run_asm_elf("test_add_8");
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
            &mut traces.branch,
            &mut traces.halt,
            &mut traces.register
        ),
        "test_add_8 failed"
    );
}

#[test]
fn test_prove_elfs_test_sub_8() {
    let (_elf, logs, instructions) = run_asm_elf("test_sub_8");
    let mut traces = Traces::from_logs_minimal(&logs, instructions.clone()).unwrap();
    assert!(
        prove_and_verify_vm_minimal(
            &mut traces.cpu,
            &mut traces.bitwise,
            &mut traces.lt,
            &mut traces.memw,
            &mut traces.load,
            &mut traces.decode,
            &mut traces.branch,
            &mut traces.halt,
            &mut traces.register
        ),
        "test_sub_8 failed"
    );
}

#[test]
fn test_prove_elfs_test_addw_8() {
    let (_elf, logs, instructions) = run_asm_elf("test_addw_8");
    let mut traces = Traces::from_logs_minimal(&logs, instructions.clone()).unwrap();
    assert!(
        prove_and_verify_vm_minimal(
            &mut traces.cpu,
            &mut traces.bitwise,
            &mut traces.lt,
            &mut traces.memw,
            &mut traces.load,
            &mut traces.decode,
            &mut traces.branch,
            &mut traces.halt,
            &mut traces.register
        ),
        "test_addw_8 failed"
    );
}

#[test]
fn test_prove_elfs_test_subw_8() {
    let (_elf, logs, instructions) = run_asm_elf("test_subw_8");
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
            &mut traces.branch,
            &mut traces.halt,
            &mut traces.register
        ),
        "test_subw_8 failed"
    );
}

#[test]
fn test_prove_elfs_test_addw_lui_8() {
    let (_elf, logs, instructions) = run_asm_elf("test_addw_lui_8");
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
            &mut traces.branch,
            &mut traces.halt,
            &mut traces.register
        ),
        "test_addw_lui_8 failed"
    );
}

#[test]
fn test_prove_elfs_test_subw_lui_8() {
    let (_elf, logs, instructions) = run_asm_elf("test_subw_lui_8");
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
            &mut traces.branch,
            &mut traces.halt,
            &mut traces.register
        ),
        "test_subw_lui_8 failed"
    );
}

#[test]
fn test_prove_elfs_test_add_neg_8() {
    let (_elf, logs, instructions) = run_asm_elf("test_add_neg_8");
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
            &mut traces.branch,
            &mut traces.halt,
            &mut traces.register
        ),
        "test_add_neg_8 failed"
    );
}

#[test]
fn test_prove_elfs_test_sub_neg_8() {
    let (_elf, logs, instructions) = run_asm_elf("test_sub_neg_8");
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
            &mut traces.branch,
            &mut traces.halt,
            &mut traces.register
        ),
        "test_sub_neg_8 failed"
    );
}

#[test]
fn test_prove_elfs_test_mul_8() {
    let (_elf, logs, instructions) = run_asm_elf("test_mul_8");
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
            &mut traces.branch,
            &mut traces.halt,
            &mut traces.register
        ),
        "test_mul_8 failed"
    );
}

#[test]
fn test_prove_elfs_test_div_8() {
    let (_elf, logs, instructions) = run_asm_elf("test_div_8");
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
            &mut traces.branch,
            &mut traces.halt,
            &mut traces.register
        ),
        "test_div_8 failed"
    );
}

#[test]
fn test_prove_elfs_test_shift_8() {
    let (_elf, logs, instructions) = run_asm_elf("test_shift_8");
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
            &mut traces.branch,
            &mut traces.halt,
            &mut traces.register
        ),
        "test_shift_8 failed"
    );
}

#[test]
fn test_prove_elfs_test_bitwise_8() {
    let (_elf, logs, instructions) = run_asm_elf("test_bitwise_8");
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
            &mut traces.branch,
            &mut traces.halt,
            &mut traces.register
        ),
        "test_bitwise_8 failed"
    );
}

#[test]
fn test_prove_elfs_test_slt_8() {
    // Initialize logger to see debug constraint validation output
    let _ = env_logger::builder().is_test(true).try_init();

    let (_elf, logs, instructions) = run_asm_elf("test_slt_8");
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
            &mut traces.branch,
            &mut traces.halt,
            &mut traces.register
        ),
        "test_slt_8 failed"
    );
}

// =========================================================================
// Comprehensive tests for all instructions
// =========================================================================

#[test]
fn test_prove_elfs_test_xor_8() {
    let (_elf, logs, instructions) = run_asm_elf("test_xor_8");
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
            &mut traces.branch,
            &mut traces.halt,
            &mut traces.register
        ),
        "test_xor_8 failed"
    );
}

#[test]
fn test_prove_elfs_test_lb_lh_8() {
    let (_elf, logs, instructions) = run_asm_elf("test_lb_lh_8");
    let mut traces = Traces::from_logs_minimal(&logs, instructions.clone()).unwrap();
    assert!(
        prove_and_verify_vm_minimal(
            &mut traces.cpu,
            &mut traces.bitwise,
            &mut traces.lt,
            &mut traces.memw,
            &mut traces.load,
            &mut traces.decode,
            &mut traces.branch,
            &mut traces.halt,
            &mut traces.register
        ),
        "test_lb_lh_8 failed"
    );
}

#[test]
fn test_prove_elfs_test_sb_sh_8() {
    let (_elf, logs, instructions) = run_asm_elf("test_sb_sh_8");
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
            &mut traces.branch,
            &mut traces.halt,
            &mut traces.register
        ),
        "test_sb_sh_8 failed"
    );
}

#[test]
fn test_prove_elfs_all_branches_16() {
    // Initialize logger to see debug constraint validation output
    let _ = env_logger::builder().is_test(true).try_init();

    let (_elf, logs, instructions) = run_asm_elf("all_branches_16");
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
            &mut traces.branch,
            &mut traces.halt,
            &mut traces.register
        ),
        "all_branches_16 failed"
    );
}

#[test]
fn test_prove_elfs_all_loadstore_32() {
    let (_elf, logs, instructions) = run_asm_elf("all_loadstore_32");
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
            &mut traces.branch,
            &mut traces.halt,
            &mut traces.register
        ),
        "all_loadstore_32 failed"
    );
}

/// Fast version using minimal bitwise table for debugging
#[test]
fn test_prove_elfs_all_instructions_64() {
    let _ = env_logger::builder().is_test(true).try_init();

    let (_elf, logs, instructions) = run_asm_elf("all_instructions_64");
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
            &mut traces.branch,
            &mut traces.halt,
            &mut traces.register
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
            &mut traces.halt,
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
    let (_elf, logs, instructions) = run_asm_elf(program_name);

    // Output metadata for CI parsing
    println!("MEMORY_PROFILE_PROGRAM={}", program_name);
    println!("MEMORY_PROFILE_INSTRUCTIONS={}", logs.len());

    let mut traces = Traces::from_logs_minimal(&logs, instructions.clone()).unwrap();

    assert!(
        prove_and_verify_vm_minimal(
            &mut traces.cpu,
            &mut traces.bitwise,
            &mut traces.lt,
            &mut traces.memw,
            &mut traces.load,
            &mut traces.decode,
            &mut traces.branch,
            &mut traces.halt,
            &mut traces.register
        ),
        "verification failed"
    );
}

/// Debug test that manually computes Memory bus balance with fixed challenges.
///
/// Uses z=1, α=2 to make manual verification easy.
/// fingerprint = z - (bus_id + is_reg*α + addr_lo*α² + addr_hi*α³ + ts_lo*α⁴ + ts_hi*α⁵ + value*α⁶)
/// term = sign * mult / fingerprint
/// Bus balances when sum of all terms = 0
#[test]
fn test_debug_memory_bus_tokens() {
    use crate::tables::memw::cols as memw_cols;
    use crate::tables::register::cols as reg_cols;
    use std::collections::HashMap;

    let (_elf, logs, instructions) = run_asm_elf("sub_neg_result");
    let traces = Traces::from_logs_minimal(&logs, instructions.clone()).unwrap();

    println!("DEBUG TABLE SIZES:");
    println!("  MEMW: {} rows", traces.memw.num_rows());
    println!("  REGISTER: {} rows", traces.register.num_rows());

    // Collect all Memory bus tokens
    // Token = (is_reg, addr_lo, addr_hi, ts_lo, ts_hi, value)
    type Token = (u64, u64, u64, u64, u64, u64);

    // Track sends (+1) and receives (-1) with their sources
    let mut token_balance: HashMap<Token, (i64, Vec<String>)> = HashMap::new();

    // === MEMW tokens (for register rows only) ===
    println!("\n=== MEMW Memory Bus Tokens (register rows) ===");
    for row in 0..traces.memw.num_rows() {
        let is_reg = traces.memw.main_table.get(row, memw_cols::IS_REGISTER).to_raw();
        if is_reg == 0 {
            continue; // Skip memory rows (multiplicity = 0)
        }

        let base_lo = traces.memw.main_table.get(row, memw_cols::BASE_ADDRESS_0).to_raw();
        let base_hi = traces.memw.main_table.get(row, memw_cols::BASE_ADDRESS_1).to_raw();
        let ts_lo = traces.memw.main_table.get(row, memw_cols::TIMESTAMP_0).to_raw();
        let ts_hi = traces.memw.main_table.get(row, memw_cols::TIMESTAMP_1).to_raw();
        let old_ts0_lo = traces.memw.main_table.get(row, memw_cols::old_timestamp(0)[0]).to_raw();
        let old_ts0_hi = traces.memw.main_table.get(row, memw_cols::old_timestamp(0)[1]).to_raw();
        let old_ts1_lo = traces.memw.main_table.get(row, memw_cols::old_timestamp(1)[0]).to_raw();
        let old_ts1_hi = traces.memw.main_table.get(row, memw_cols::old_timestamp(1)[1]).to_raw();
        let val0 = traces.memw.main_table.get(row, memw_cols::VALUE[0]).to_raw();
        let val1 = traces.memw.main_table.get(row, memw_cols::VALUE[1]).to_raw();
        let old0 = traces.memw.main_table.get(row, memw_cols::OLD[0]).to_raw();
        let old1 = traces.memw.main_table.get(row, memw_cols::OLD[1]).to_raw();

        // M1: SEND old token for Word 0 (is_reg, base, old_ts[0], old[0])
        let m1_token: Token = (is_reg, base_lo, base_hi, old_ts0_lo, old_ts0_hi, old0);
        println!("MEMW row {} M1 SEND: {:?}", row, m1_token);
        let entry = token_balance.entry(m1_token).or_insert((0, vec![]));
        entry.0 += 1; // sender = +1
        entry.1.push(format!("MEMW[{}] M1 SEND", row));

        // M2: RECV new token for Word 0 (is_reg, base, ts, value[0])
        let m2_token: Token = (is_reg, base_lo, base_hi, ts_lo, ts_hi, val0);
        println!("MEMW row {} M2 RECV: {:?}", row, m2_token);
        let entry = token_balance.entry(m2_token).or_insert((0, vec![]));
        entry.0 -= 1; // receiver = -1
        entry.1.push(format!("MEMW[{}] M2 RECV", row));

        // M3: SEND old token for Word 1 (is_reg, base+1, old_ts[1], old[1])
        let m3_token: Token = (is_reg, base_lo + 1, base_hi, old_ts1_lo, old_ts1_hi, old1);
        println!("MEMW row {} M3 SEND: {:?}", row, m3_token);
        let entry = token_balance.entry(m3_token).or_insert((0, vec![]));
        entry.0 += 1;
        entry.1.push(format!("MEMW[{}] M3 SEND", row));

        // M4: RECV new token for Word 1 (is_reg, base+1, ts, value[1])
        let m4_token: Token = (is_reg, base_lo + 1, base_hi, ts_lo, ts_hi, val1);
        println!("MEMW row {} M4 RECV: {:?}", row, m4_token);
        let entry = token_balance.entry(m4_token).or_insert((0, vec![]));
        entry.0 -= 1;
        entry.1.push(format!("MEMW[{}] M4 RECV", row));
    }

    // === REGISTER tokens (all 64 rows participate) ===
    println!("\n=== REGISTER Memory Bus Tokens ===");
    for row in 0..traces.register.num_rows().min(64) {
        let offset = traces.register.main_table.get(row, reg_cols::OFFSET).to_raw();
        let init = traces.register.main_table.get(row, reg_cols::INIT).to_raw();
        let fini = traces.register.main_table.get(row, reg_cols::FINI).to_raw();
        let ts_lo = traces.register.main_table.get(row, reg_cols::TIMESTAMP_LO).to_raw();
        let ts_hi = traces.register.main_table.get(row, reg_cols::TIMESTAMP_HI).to_raw();

        // REG-C1: RECV init token (1, offset, 0, 0, 0, init)
        let c1_token: Token = (1, offset, 0, 0, 0, init);
        println!("REG row {} C1 RECV: {:?}", row, c1_token);
        let entry = token_balance.entry(c1_token).or_insert((0, vec![]));
        entry.0 -= 1; // receiver = -1
        entry.1.push(format!("REG[{}] C1 RECV", row));

        // REG-C2: SEND final token (1, offset, 0, ts_lo, ts_hi, fini)
        let c2_token: Token = (1, offset, 0, ts_lo, ts_hi, fini);
        println!("REG row {} C2 SEND: {:?}", row, c2_token);
        let entry = token_balance.entry(c2_token).or_insert((0, vec![]));
        entry.0 += 1; // sender = +1
        entry.1.push(format!("REG[{}] C2 SEND", row));
    }

    // === Check for imbalanced tokens ===
    println!("\n=== IMBALANCED TOKENS (should be empty if bus balances) ===");
    let mut imbalanced = 0;
    for (token, (balance, sources)) in &token_balance {
        if *balance != 0 {
            println!("IMBALANCED: {:?} balance={} sources={:?}", token, balance, sources);
            imbalanced += 1;
        }
    }
    if imbalanced == 0 {
        println!("All Memory bus tokens balance!");
    } else {
        println!("Found {} imbalanced tokens", imbalanced);
    }

    // === Compute LogUp balance with fixed challenges z=1000, α=2 ===
    // Using z=1000 to avoid division by zero (fingerprint = z - linear_comb)
    // fingerprint = z - (bus_id + is_reg*α + addr_lo*α² + addr_hi*α³ + ts_lo*α⁴ + ts_hi*α⁵ + value*α⁶)
    // term = sign * mult / fingerprint
    println!("\n=== LogUp Balance with z=1000, α=2 ===");

    let z: i128 = 1000;
    let alpha: i128 = 2;
    let bus_id: i128 = 16; // BusId::Memory

    // Compute fingerprint for a token
    let fingerprint = |is_reg: u64, addr_lo: u64, addr_hi: u64, ts_lo: u64, ts_hi: u64, value: u64| -> i128 {
        let linear_comb = bus_id
            + (is_reg as i128) * alpha
            + (addr_lo as i128) * alpha.pow(2)
            + (addr_hi as i128) * alpha.pow(3)
            + (ts_lo as i128) * alpha.pow(4)
            + (ts_hi as i128) * alpha.pow(5)
            + (value as i128) * alpha.pow(6);
        z - linear_comb
    };

    let mut total_sum: f64 = 0.0;

    // MEMW tokens
    for row in 0..traces.memw.num_rows() {
        let is_reg = traces.memw.main_table.get(row, memw_cols::IS_REGISTER).to_raw();
        if is_reg == 0 {
            continue;
        }

        let base_lo = traces.memw.main_table.get(row, memw_cols::BASE_ADDRESS_0).to_raw();
        let base_hi = traces.memw.main_table.get(row, memw_cols::BASE_ADDRESS_1).to_raw();
        let ts_lo = traces.memw.main_table.get(row, memw_cols::TIMESTAMP_0).to_raw();
        let ts_hi = traces.memw.main_table.get(row, memw_cols::TIMESTAMP_1).to_raw();
        let old_ts0_lo = traces.memw.main_table.get(row, memw_cols::old_timestamp(0)[0]).to_raw();
        let old_ts0_hi = traces.memw.main_table.get(row, memw_cols::old_timestamp(0)[1]).to_raw();
        let old_ts1_lo = traces.memw.main_table.get(row, memw_cols::old_timestamp(1)[0]).to_raw();
        let old_ts1_hi = traces.memw.main_table.get(row, memw_cols::old_timestamp(1)[1]).to_raw();
        let val0 = traces.memw.main_table.get(row, memw_cols::VALUE[0]).to_raw();
        let val1 = traces.memw.main_table.get(row, memw_cols::VALUE[1]).to_raw();
        let old0 = traces.memw.main_table.get(row, memw_cols::OLD[0]).to_raw();
        let old1 = traces.memw.main_table.get(row, memw_cols::OLD[1]).to_raw();

        // M1: SEND (+1) old token for Word 0
        let fp = fingerprint(is_reg, base_lo, base_hi, old_ts0_lo, old_ts0_hi, old0);
        let term = 1.0 / (fp as f64);
        total_sum += term;

        // M2: RECV (-1) new token for Word 0
        let fp = fingerprint(is_reg, base_lo, base_hi, ts_lo, ts_hi, val0);
        let term = -1.0 / (fp as f64);
        total_sum += term;

        // M3: SEND (+1) old token for Word 1
        let fp = fingerprint(is_reg, base_lo + 1, base_hi, old_ts1_lo, old_ts1_hi, old1);
        let term = 1.0 / (fp as f64);
        total_sum += term;

        // M4: RECV (-1) new token for Word 1
        let fp = fingerprint(is_reg, base_lo + 1, base_hi, ts_lo, ts_hi, val1);
        let term = -1.0 / (fp as f64);
        total_sum += term;
    }
    println!("After MEMW: total_sum = {}", total_sum);

    // REGISTER tokens
    for row in 0..traces.register.num_rows().min(64) {
        let offset = traces.register.main_table.get(row, reg_cols::OFFSET).to_raw();
        let init = traces.register.main_table.get(row, reg_cols::INIT).to_raw();
        let fini = traces.register.main_table.get(row, reg_cols::FINI).to_raw();
        let ts_lo = traces.register.main_table.get(row, reg_cols::TIMESTAMP_LO).to_raw();
        let ts_hi = traces.register.main_table.get(row, reg_cols::TIMESTAMP_HI).to_raw();

        // REG-C1: RECV (-1) init token
        let fp = fingerprint(1, offset, 0, 0, 0, init);
        let term = -1.0 / (fp as f64);
        total_sum += term;

        // REG-C2: SEND (+1) final token
        let fp = fingerprint(1, offset, 0, ts_lo, ts_hi, fini);
        let term = 1.0 / (fp as f64);
        total_sum += term;
    }
    println!("After REGISTER: total_sum = {}", total_sum);
    println!("Bus {} (should be ~0 if balanced)", if total_sum.abs() < 1e-10 { "BALANCES" } else { "DOES NOT BALANCE" });
}
