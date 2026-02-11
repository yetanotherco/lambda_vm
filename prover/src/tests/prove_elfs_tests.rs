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
use stark::traits::AIR;
use stark::verifier::{IsStarkVerifier, Verifier};

use crate::tables::trace_builder::Traces;
use crate::tables::types::{GoldilocksExtension, GoldilocksField};

use executor::elf::Elf;

// Import shared utilities
use crate::VmAirs;
use crate::test_utils::run_asm_elf;

type F = GoldilocksField;
type E = GoldilocksExtension;

// =============================================================================
// Prover test helpers
// =============================================================================

/// Run multi_prove and multi_verify for all VM tables.
///
/// Includes: CPU + Bitwise + LT + MEMW + LOAD + DECODE + MUL + BRANCH + HALT + REGISTER + PAGEs
///
/// Uses minimal bitwise (no full 2^20 preprocessed table) but DECODE is always preprocessed.
fn prove_and_verify_vm_minimal(elf: &Elf, traces: &mut Traces) -> bool {
    let proof_options = ProofOptions::default_test_options();

    // Create all AIRs including PAGE and REGISTER tables
    let airs = VmAirs::new(elf, &proof_options, true, &traces.page_configs);

    // Build air_trace_pairs for all tables
    let air_trace_pairs = airs.air_trace_pairs(traces);

    let multi_proof =
        match Prover::multi_prove(air_trace_pairs, &mut DefaultTranscript::<E>::new(&[])) {
            Ok(proof) => proof,
            Err(e) => {
                eprintln!("Prover error: {:?}", e);
                return false;
            }
        };

    // Verify using centralized air_refs() which includes all tables
    Verifier::multi_verify(
        &airs.air_refs(),
        &multi_proof,
        &mut DefaultTranscript::<E>::new(&[]),
    )
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
    let mut traces =
        Traces::from_elf_and_logs(&elf, &logs, crate::tables::page::DEFAULT_STACK_SIZE).unwrap();

    assert!(
        prove_and_verify_vm_minimal(&elf, &mut traces),
        "Proof verification failed for sub program (fast)"
    );
}

#[test]
fn test_prove_elfs_sub_neg_result_fast() {
    let (elf, logs, instructions) = run_asm_elf("sub_neg_result");
    let mut traces = Traces::from_logs_minimal(&logs, instructions.clone()).unwrap();

    println!(
        "Fast SUB_NEG: CPU {} rows, Bitwise {} rows, MEMW {} rows, REGISTER {} rows",
        traces.cpu.main_table.height,
        traces.bitwise.main_table.height,
        traces.memw.main_table.height,
        traces.register.main_table.height,
    );

    assert!(
        prove_and_verify_vm_minimal(&elf, &mut traces),
        "Proof verification failed for sub_neg_result program (fast)"
    );
}

#[test]
fn test_prove_elfs_sub_underflow_fast() {
    let (elf, logs, instructions) = run_asm_elf("sub_underflow");
    let mut traces = Traces::from_logs_minimal(&logs, instructions.clone()).unwrap();

    println!(
        "Fast SUB_UNDERFLOW: CPU {} rows, Bitwise {} rows (minimal)",
        traces.cpu.main_table.height, traces.bitwise.main_table.height,
    );

    assert!(
        prove_and_verify_vm_minimal(&elf, &mut traces),
        "Proof verification failed for sub_underflow program (fast)"
    );
}

#[test]
fn test_prove_elfs_subw_fast() {
    let (elf, logs, instructions) = run_asm_elf("subw");
    let mut traces = Traces::from_logs_minimal(&logs, instructions.clone()).unwrap();

    println!(
        "Fast SUBW: CPU {} rows, Bitwise {} rows (minimal)",
        traces.cpu.main_table.height, traces.bitwise.main_table.height,
    );

    assert!(
        prove_and_verify_vm_minimal(&elf, &mut traces),
        "Proof verification failed for subw program (fast)"
    );
}

/// 8-instruction test with LUI
#[test]
fn test_prove_elfs_arith_lui_8() {
    let (elf, logs, instructions) = run_asm_elf("arith_lui_8");
    let mut traces = Traces::from_logs_minimal(&logs, instructions.clone()).unwrap();

    println!(
        "ArithLUI8: CPU {} rows, Bitwise {} rows (minimal)",
        traces.cpu.main_table.height, traces.bitwise.main_table.height,
    );

    assert!(
        prove_and_verify_vm_minimal(&elf, &mut traces),
        "Proof verification failed for arith_lui_8 program"
    );
}

/// 8-instruction test with ADD, SUB, ADDW, SUBW
#[test]
fn test_prove_elfs_arith_8() {
    let (elf, logs, instructions) = run_asm_elf("arith_8");
    let mut traces = Traces::from_logs_minimal(&logs, instructions.clone()).unwrap();

    println!(
        "Arith8: CPU {} rows, Bitwise {} rows (minimal)",
        traces.cpu.main_table.height, traces.bitwise.main_table.height,
    );

    assert!(
        prove_and_verify_vm_minimal(&elf, &mut traces),
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
    let mut traces = Traces::from_logs_minimal(&logs, instructions.clone()).unwrap();

    println!(
        "BasicArith32: CPU {} rows, Bitwise {} rows (minimal)",
        traces.cpu.main_table.height, traces.bitwise.main_table.height,
    );

    assert!(
        prove_and_verify_vm_minimal(&elf, &mut traces),
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
    let mut traces = Traces::from_logs_minimal(&logs, instructions.clone()).unwrap();

    // Collect LT lookups first (needed for both LT trace and bitwise lookups)

    println!(
        "Comprehensive: CPU {} rows, Bitwise {} rows",
        traces.cpu.main_table.height, traces.bitwise.main_table.height,
    );

    assert!(
        prove_and_verify_vm_minimal(&elf, &mut traces),
        "Proof verification failed for comprehensive_test program"
    );
}

// =============================================================================
// Instruction-specific 8-step tests
// =============================================================================

#[test]
fn test_prove_elfs_test_add_8() {
    let (elf, logs, instructions) = run_asm_elf("test_add_8");
    let mut traces = Traces::from_logs_minimal(&logs, instructions.clone()).unwrap();
    // Use traces.lt and traces.bitwise directly instead of generating separate ones
    // This includes MEMW timestamp ordering LT ops and their bitwise lookups
    assert!(
        prove_and_verify_vm_minimal(&elf, &mut traces),
        "test_add_8 failed"
    );
}

#[test]
fn test_prove_elfs_test_sub_8() {
    let (elf, logs, instructions) = run_asm_elf("test_sub_8");
    let mut traces = Traces::from_logs_minimal(&logs, instructions.clone()).unwrap();
    assert!(
        prove_and_verify_vm_minimal(&elf, &mut traces),
        "test_sub_8 failed"
    );
}

#[test]
fn test_prove_elfs_test_addw_8() {
    let (elf, logs, instructions) = run_asm_elf("test_addw_8");
    let mut traces = Traces::from_logs_minimal(&logs, instructions.clone()).unwrap();
    assert!(
        prove_and_verify_vm_minimal(&elf, &mut traces),
        "test_addw_8 failed"
    );
}

#[test]
fn test_prove_elfs_test_subw_8() {
    let (elf, logs, instructions) = run_asm_elf("test_subw_8");
    let mut traces = Traces::from_logs_minimal(&logs, instructions.clone()).unwrap();
    // Using traces from Traces::from_logs() which includes MEMW LT ops
    assert!(
        prove_and_verify_vm_minimal(&elf, &mut traces),
        "test_subw_8 failed"
    );
}

#[test]
fn test_prove_elfs_test_addw_lui_8() {
    let (elf, logs, instructions) = run_asm_elf("test_addw_lui_8");
    let mut traces = Traces::from_logs_minimal(&logs, instructions.clone()).unwrap();
    // Using traces from Traces::from_logs() which includes MEMW LT ops
    assert!(
        prove_and_verify_vm_minimal(&elf, &mut traces),
        "test_addw_lui_8 failed"
    );
}

#[test]
fn test_prove_elfs_test_subw_lui_8() {
    let (elf, logs, instructions) = run_asm_elf("test_subw_lui_8");
    let mut traces = Traces::from_logs_minimal(&logs, instructions.clone()).unwrap();
    // Using traces from Traces::from_logs() which includes MEMW LT ops
    assert!(
        prove_and_verify_vm_minimal(&elf, &mut traces),
        "test_subw_lui_8 failed"
    );
}

#[test]
fn test_prove_elfs_test_add_neg_8() {
    let (elf, logs, instructions) = run_asm_elf("test_add_neg_8");
    let mut traces = Traces::from_logs_minimal(&logs, instructions.clone()).unwrap();
    // Using traces from Traces::from_logs() which includes MEMW LT ops
    assert!(
        prove_and_verify_vm_minimal(&elf, &mut traces),
        "test_add_neg_8 failed"
    );
}

#[test]
fn test_prove_elfs_test_sub_neg_8() {
    let (elf, logs, instructions) = run_asm_elf("test_sub_neg_8");
    let mut traces = Traces::from_logs_minimal(&logs, instructions.clone()).unwrap();
    // Using traces from Traces::from_logs() which includes MEMW LT ops
    assert!(
        prove_and_verify_vm_minimal(&elf, &mut traces),
        "test_sub_neg_8 failed"
    );
}

#[test]
fn test_prove_elfs_test_mul_8() {
    let (elf, logs, instructions) = run_asm_elf("test_mul_8");
    let mut traces = Traces::from_logs_minimal(&logs, instructions.clone()).unwrap();
    // Using traces from Traces::from_logs() which includes MEMW LT ops
    assert!(
        prove_and_verify_vm_minimal(&elf, &mut traces),
        "test_mul_8 failed"
    );
}

#[test]
fn test_prove_elfs_test_div_8() {
    let (elf, logs, instructions) = run_asm_elf("test_div_8");
    let mut traces = Traces::from_logs_minimal(&logs, instructions.clone()).unwrap();
    // Using traces from Traces::from_logs() which includes MEMW LT ops
    assert!(
        prove_and_verify_vm_minimal(&elf, &mut traces),
        "test_div_8 failed"
    );
}

#[test]
fn test_prove_elfs_test_shift_8() {
    let (elf, logs, instructions) = run_asm_elf("test_shift_8");
    let mut traces = Traces::from_logs_minimal(&logs, instructions.clone()).unwrap();
    // Using traces from Traces::from_logs() which includes MEMW LT ops
    assert!(
        prove_and_verify_vm_minimal(&elf, &mut traces),
        "test_shift_8 failed"
    );
}

#[test]
fn test_prove_elfs_test_bitwise_8() {
    let (elf, logs, instructions) = run_asm_elf("test_bitwise_8");
    let mut traces = Traces::from_logs_minimal(&logs, instructions.clone()).unwrap();
    // Using traces from Traces::from_logs() which includes MEMW LT ops
    assert!(
        prove_and_verify_vm_minimal(&elf, &mut traces),
        "test_bitwise_8 failed"
    );
}

#[test]
fn test_prove_elfs_test_slt_8() {
    // Initialize logger to see debug constraint validation output
    let _ = env_logger::builder().is_test(true).try_init();

    let (elf, logs, instructions) = run_asm_elf("test_slt_8");
    let mut traces = Traces::from_logs_minimal(&logs, instructions.clone()).unwrap();

    // Collect LT lookups first (needed for both LT trace and bitwise lookups)

    println!(
        "test_slt_8: CPU {} rows, Bitwise {} rows",
        traces.cpu.main_table.height, traces.bitwise.main_table.height,
    );
    assert!(
        prove_and_verify_vm_minimal(&elf, &mut traces),
        "test_slt_8 failed"
    );
}

// =========================================================================
// Comprehensive tests for all instructions
// =========================================================================

#[test]
fn test_prove_elfs_test_xor_8() {
    let (elf, logs, instructions) = run_asm_elf("test_xor_8");
    let mut traces = Traces::from_logs_minimal(&logs, instructions.clone()).unwrap();
    // Using traces from Traces::from_logs() which includes MEMW LT ops
    assert!(
        prove_and_verify_vm_minimal(&elf, &mut traces),
        "test_xor_8 failed"
    );
}

#[test]
fn test_prove_elfs_test_lb_lh_8() {
    let (elf, logs, _instructions) = run_asm_elf("test_lb_lh_8");
    let mut traces =
        Traces::from_elf_and_logs(&elf, &logs, crate::tables::page::DEFAULT_STACK_SIZE).unwrap();
    assert!(
        prove_and_verify_vm_minimal(&elf, &mut traces),
        "test_lb_lh_8 failed"
    );
}

#[test]
fn test_prove_elfs_test_sb_sh_8() {
    let (elf, logs, _instructions) = run_asm_elf("test_sb_sh_8");
    let mut traces =
        Traces::from_elf_and_logs(&elf, &logs, crate::tables::page::DEFAULT_STACK_SIZE).unwrap();
    assert!(
        prove_and_verify_vm_minimal(&elf, &mut traces),
        "test_sb_sh_8 failed"
    );
}

#[test]
fn test_prove_elfs_all_branches_16() {
    // Initialize logger to see debug constraint validation output
    let _ = env_logger::builder().is_test(true).try_init();

    let (elf, logs, instructions) = run_asm_elf("all_branches_16");
    let mut traces = Traces::from_logs_minimal(&logs, instructions.clone()).unwrap();

    // BLT instructions need LT table (like SLT)

    println!(
        "all_branches_16: CPU {} rows, Bitwise {} rows",
        traces.cpu.main_table.height, traces.bitwise.main_table.height,
    );
    assert!(
        prove_and_verify_vm_minimal(&elf, &mut traces),
        "all_branches_16 failed"
    );
}

#[test]
fn test_prove_elfs_all_loadstore_32() {
    let (elf, logs, _instructions) = run_asm_elf("all_loadstore_32");
    let mut traces =
        Traces::from_elf_and_logs(&elf, &logs, crate::tables::page::DEFAULT_STACK_SIZE).unwrap();
    assert!(
        prove_and_verify_vm_minimal(&elf, &mut traces),
        "all_loadstore_32 failed"
    );
}

/// Fast version using minimal bitwise table for debugging
#[test]
fn test_prove_elfs_all_instructions_64() {
    let _ = env_logger::builder().is_test(true).try_init();

    let (elf, logs, instructions) = run_asm_elf("all_instructions_64");
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
        prove_and_verify_vm_minimal(&elf, &mut traces),
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

    let elf_bytes = crate::test_utils::asm_elf_bytes("all_instructions_64");
    let result = crate::prove_and_verify(&elf_bytes).expect("prove_and_verify failed");
    assert!(
        result,
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

    let mut traces = Traces::from_logs_minimal(&logs, instructions.clone()).unwrap();

    assert!(
        prove_and_verify_vm_minimal(&elf, &mut traces),
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
        let is_reg = traces
            .memw
            .main_table
            .get(row, memw_cols::IS_REGISTER)
            .to_raw();
        if is_reg == 0 {
            continue; // Skip memory rows (multiplicity = 0)
        }

        let base_lo = traces
            .memw
            .main_table
            .get(row, memw_cols::BASE_ADDRESS_0)
            .to_raw();
        let base_hi = traces
            .memw
            .main_table
            .get(row, memw_cols::BASE_ADDRESS_1)
            .to_raw();
        let ts_lo = traces
            .memw
            .main_table
            .get(row, memw_cols::TIMESTAMP_0)
            .to_raw();
        let ts_hi = traces
            .memw
            .main_table
            .get(row, memw_cols::TIMESTAMP_1)
            .to_raw();
        let old_ts0_lo = traces
            .memw
            .main_table
            .get(row, memw_cols::old_timestamp(0)[0])
            .to_raw();
        let old_ts0_hi = traces
            .memw
            .main_table
            .get(row, memw_cols::old_timestamp(0)[1])
            .to_raw();
        let old_ts1_lo = traces
            .memw
            .main_table
            .get(row, memw_cols::old_timestamp(1)[0])
            .to_raw();
        let old_ts1_hi = traces
            .memw
            .main_table
            .get(row, memw_cols::old_timestamp(1)[1])
            .to_raw();
        let val0 = traces
            .memw
            .main_table
            .get(row, memw_cols::VALUE[0])
            .to_raw();
        let val1 = traces
            .memw
            .main_table
            .get(row, memw_cols::VALUE[1])
            .to_raw();
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
        let offset = traces
            .register
            .main_table
            .get(row, reg_cols::OFFSET)
            .to_raw();
        let init = traces.register.main_table.get(row, reg_cols::INIT).to_raw();
        let fini = traces.register.main_table.get(row, reg_cols::FINI).to_raw();
        let ts_lo = traces
            .register
            .main_table
            .get(row, reg_cols::TIMESTAMP_LO)
            .to_raw();
        let ts_hi = traces
            .register
            .main_table
            .get(row, reg_cols::TIMESTAMP_HI)
            .to_raw();

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
            println!(
                "IMBALANCED: {:?} balance={} sources={:?}",
                token, balance, sources
            );
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
    let fingerprint =
        |is_reg: u64, addr_lo: u64, addr_hi: u64, ts_lo: u64, ts_hi: u64, value: u64| -> i128 {
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
        let is_reg = traces
            .memw
            .main_table
            .get(row, memw_cols::IS_REGISTER)
            .to_raw();
        if is_reg == 0 {
            continue;
        }

        let base_lo = traces
            .memw
            .main_table
            .get(row, memw_cols::BASE_ADDRESS_0)
            .to_raw();
        let base_hi = traces
            .memw
            .main_table
            .get(row, memw_cols::BASE_ADDRESS_1)
            .to_raw();
        let ts_lo = traces
            .memw
            .main_table
            .get(row, memw_cols::TIMESTAMP_0)
            .to_raw();
        let ts_hi = traces
            .memw
            .main_table
            .get(row, memw_cols::TIMESTAMP_1)
            .to_raw();
        let old_ts0_lo = traces
            .memw
            .main_table
            .get(row, memw_cols::old_timestamp(0)[0])
            .to_raw();
        let old_ts0_hi = traces
            .memw
            .main_table
            .get(row, memw_cols::old_timestamp(0)[1])
            .to_raw();
        let old_ts1_lo = traces
            .memw
            .main_table
            .get(row, memw_cols::old_timestamp(1)[0])
            .to_raw();
        let old_ts1_hi = traces
            .memw
            .main_table
            .get(row, memw_cols::old_timestamp(1)[1])
            .to_raw();
        let val0 = traces
            .memw
            .main_table
            .get(row, memw_cols::VALUE[0])
            .to_raw();
        let val1 = traces
            .memw
            .main_table
            .get(row, memw_cols::VALUE[1])
            .to_raw();
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
        let offset = traces
            .register
            .main_table
            .get(row, reg_cols::OFFSET)
            .to_raw();
        let init = traces.register.main_table.get(row, reg_cols::INIT).to_raw();
        let fini = traces.register.main_table.get(row, reg_cols::FINI).to_raw();
        let ts_lo = traces
            .register
            .main_table
            .get(row, reg_cols::TIMESTAMP_LO)
            .to_raw();
        let ts_hi = traces
            .register
            .main_table
            .get(row, reg_cols::TIMESTAMP_HI)
            .to_raw();

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
    println!(
        "Bus {} (should be ~0 if balanced)",
        if total_sum.abs() < 1e-10 {
            "BALANCES"
        } else {
            "DOES NOT BALANCE"
        }
    );
}

/// Debug test to trace ALL Memory bus tokens (registers + memory).
/// This helps identify mismatches in the full Memory bus.
#[test]
fn test_debug_memory_tokens_sb_sh() {
    use crate::tables::memw::cols as memw_cols;
    use crate::tables::page::cols as page_cols;
    use crate::tables::register::cols as reg_cols;
    use std::collections::HashMap;

    let (elf, logs, _instructions) = run_asm_elf("test_sb_sh_8");
    let traces =
        Traces::from_elf_and_logs(&elf, &logs, crate::tables::page::DEFAULT_STACK_SIZE).unwrap();

    println!("DEBUG: test_sb_sh_8 Memory bus tokens (FULL)");
    println!("  MEMW rows: {}", traces.memw.num_rows());
    println!("  REGISTER rows: {}", traces.register.num_rows());
    println!("  PAGE tables: {}", traces.pages.len());

    // Token = (is_reg, addr_lo, addr_hi, ts_lo, ts_hi, value)
    type Token = (u64, u64, u64, u64, u64, u64);

    // Track sends (+1) and receives (-1)
    let mut token_balance: HashMap<Token, (i64, Vec<String>)> = HashMap::new();

    // === REGISTER Memory tokens ===
    println!("\n=== REGISTER Memory Bus Tokens (is_register=1) ===");
    for row in 0..traces.register.num_rows().min(64) {
        let offset = traces
            .register
            .main_table
            .get(row, reg_cols::OFFSET)
            .to_raw();
        let init = traces.register.main_table.get(row, reg_cols::INIT).to_raw();
        let fini = traces.register.main_table.get(row, reg_cols::FINI).to_raw();
        let ts_lo = traces
            .register
            .main_table
            .get(row, reg_cols::TIMESTAMP_LO)
            .to_raw();
        let ts_hi = traces
            .register
            .main_table
            .get(row, reg_cols::TIMESTAMP_HI)
            .to_raw();

        // REG-C1: RECV init token (1, offset, 0, 0, 0, init)
        let c1_token: Token = (1, offset, 0, 0, 0, init);
        let entry = token_balance.entry(c1_token).or_insert((0, vec![]));
        entry.0 -= 1; // receiver
        entry.1.push(format!("REG[{}] C1 RECV", row));

        // REG-C2: SEND final token (1, offset, 0, ts_lo, ts_hi, fini)
        let c2_token: Token = (1, offset, 0, ts_lo, ts_hi, fini);
        let entry = token_balance.entry(c2_token).or_insert((0, vec![]));
        entry.0 += 1; // sender
        entry.1.push(format!("REG[{}] C2 SEND", row));

        // Only print changed registers
        if ts_lo != 0 || ts_hi != 0 || init != fini {
            println!(
                "  row {} offset={} init={} fini={} ts={}:",
                row,
                offset,
                init,
                fini,
                ts_lo | (ts_hi << 32)
            );
            println!("    C1 RECV: {:?}", c1_token);
            println!("    C2 SEND: {:?}", c2_token);
        }
    }

    // === MEMW Memory tokens (ALL rows, both register and memory) ===
    println!("\n=== MEMW Memory Bus Tokens (ALL rows) ===");
    let mut memw_register_rows = 0;
    let mut memw_memory_rows = 0;
    for row in 0..traces.memw.num_rows() {
        let is_reg = traces
            .memw
            .main_table
            .get(row, memw_cols::IS_REGISTER)
            .to_raw();

        // Count row types
        if is_reg == 1 {
            memw_register_rows += 1;
        } else {
            memw_memory_rows += 1;
        }

        let mu_read = traces.memw.main_table.get(row, memw_cols::MU_READ).to_raw();
        let mu_write = traces
            .memw
            .main_table
            .get(row, memw_cols::MU_WRITE)
            .to_raw();
        let mu_sum = mu_read + mu_write;
        if mu_sum == 0 {
            continue; // Padding row
        }

        let base_lo = traces
            .memw
            .main_table
            .get(row, memw_cols::BASE_ADDRESS_0)
            .to_raw();
        let base_hi = traces
            .memw
            .main_table
            .get(row, memw_cols::BASE_ADDRESS_1)
            .to_raw();
        let ts_lo = traces
            .memw
            .main_table
            .get(row, memw_cols::TIMESTAMP_0)
            .to_raw();
        let ts_hi = traces
            .memw
            .main_table
            .get(row, memw_cols::TIMESTAMP_1)
            .to_raw();
        let old_ts0_lo = traces
            .memw
            .main_table
            .get(row, memw_cols::old_timestamp(0)[0])
            .to_raw();
        let old_ts0_hi = traces
            .memw
            .main_table
            .get(row, memw_cols::old_timestamp(0)[1])
            .to_raw();
        let val0 = traces
            .memw
            .main_table
            .get(row, memw_cols::VALUE[0])
            .to_raw();
        let old0 = traces.memw.main_table.get(row, memw_cols::OLD[0]).to_raw();

        let write2 = traces.memw.main_table.get(row, memw_cols::WRITE2).to_raw();
        let write4 = traces.memw.main_table.get(row, memw_cols::WRITE4).to_raw();
        let write8 = traces.memw.main_table.get(row, memw_cols::WRITE8).to_raw();

        println!(
            "MEMW row {} [is_read={}, is_reg={}, w2={}, w4={}, w8={}]:",
            row, mu_read, is_reg, write2, write4, write8
        );
        println!(
            "  base=0x{:08x}_{:08x}, ts={}, old_ts0={}",
            base_hi,
            base_lo,
            ts_lo | (ts_hi << 32),
            old_ts0_lo | (old_ts0_hi << 32)
        );

        // CM14: SEND old token for byte 0
        let send_token: Token = (is_reg, base_lo, base_hi, old_ts0_lo, old_ts0_hi, old0);
        println!("  CM14 SEND: {:?}", send_token);
        let entry = token_balance.entry(send_token).or_insert((0, vec![]));
        entry.0 += mu_sum as i64;
        entry.1.push(format!("MEMW[{}] CM14 SEND", row));

        // CM15: RECV new token for byte 0
        let recv_token: Token = (is_reg, base_lo, base_hi, ts_lo, ts_hi, val0);
        println!("  CM15 RECV: {:?}", recv_token);
        let entry = token_balance.entry(recv_token).or_insert((0, vec![]));
        entry.0 -= mu_sum as i64;
        entry.1.push(format!("MEMW[{}] CM15 RECV", row));

        // For multi-byte accesses (w2 = write2+write4+write8 > 0)
        let w2 = write2 + write4 + write8;
        if w2 > 0 {
            let old_ts1_lo = traces
                .memw
                .main_table
                .get(row, memw_cols::old_timestamp(1)[0])
                .to_raw();
            let old_ts1_hi = traces
                .memw
                .main_table
                .get(row, memw_cols::old_timestamp(1)[1])
                .to_raw();
            let val1 = traces
                .memw
                .main_table
                .get(row, memw_cols::VALUE[1])
                .to_raw();
            let old1 = traces.memw.main_table.get(row, memw_cols::OLD[1]).to_raw();

            // address_add(0) = base + 1, stored as DWordHL
            let addr1_lo0 = traces
                .memw
                .main_table
                .get(row, memw_cols::address_add(0)[0])
                .to_raw();
            let addr1_lo1 = traces
                .memw
                .main_table
                .get(row, memw_cols::address_add(0)[1])
                .to_raw();
            let addr1_hi0 = traces
                .memw
                .main_table
                .get(row, memw_cols::address_add(0)[2])
                .to_raw();
            let addr1_hi1 = traces
                .memw
                .main_table
                .get(row, memw_cols::address_add(0)[3])
                .to_raw();
            let addr1_lo = addr1_lo0 + (addr1_lo1 << 16);
            let addr1_hi = addr1_hi0 + (addr1_hi1 << 16);

            // CM16: SEND old token for byte 1
            let send_token1: Token = (is_reg, addr1_lo, addr1_hi, old_ts1_lo, old_ts1_hi, old1);
            println!("  CM16 SEND (byte 1): {:?}", send_token1);
            let entry = token_balance.entry(send_token1).or_insert((0, vec![]));
            entry.0 += w2 as i64;
            entry.1.push(format!("MEMW[{}] CM16 SEND", row));

            // CM17: RECV new token for byte 1
            let recv_token1: Token = (is_reg, addr1_lo, addr1_hi, ts_lo, ts_hi, val1);
            println!("  CM17 RECV (byte 1): {:?}", recv_token1);
            let entry = token_balance.entry(recv_token1).or_insert((0, vec![]));
            entry.0 -= w2 as i64;
            entry.1.push(format!("MEMW[{}] CM17 RECV", row));
        }
    }

    println!(
        "\n  MEMW summary: {} register rows, {} memory rows",
        memw_register_rows, memw_memory_rows
    );

    // === PAGE Memory tokens (only for accessed addresses) ===
    println!("\n=== PAGE Memory Bus Tokens ===");
    for (page_idx, (page_trace, page_config)) in traces
        .pages
        .iter()
        .zip(traces.page_configs.iter())
        .enumerate()
    {
        let page_base = page_config.page_base;
        let page_size = page_config.page_size;
        let page_lo = page_base & 0xFFFF_FFFF;
        let page_hi = page_base >> 32;
        let trace_rows = page_trace.num_rows();

        println!(
            "PAGE {} [base=0x{:016x}, size={}, trace_rows={}]:",
            page_idx, page_base, page_size, trace_rows
        );

        // Only show rows with non-zero timestamps (accessed addresses)
        for row in 0..page_trace.num_rows().min(page_size) {
            let offset = page_trace.main_table.get(row, page_cols::OFFSET).to_raw();
            let init = page_trace.main_table.get(row, page_cols::INIT).to_raw();
            let fini = page_trace.main_table.get(row, page_cols::FINI).to_raw();
            let ts_lo = page_trace
                .main_table
                .get(row, page_cols::TIMESTAMP_LO)
                .to_raw();
            let ts_hi = page_trace
                .main_table
                .get(row, page_cols::TIMESTAMP_HI)
                .to_raw();

            // Compute full address
            let addr_lo = page_lo + offset;
            let addr_hi = page_hi;

            // C3: RECV init token (all rows)
            let c3_token: Token = (0, addr_lo, addr_hi, 0, 0, init);
            let entry = token_balance.entry(c3_token).or_insert((0, vec![]));
            entry.0 -= 1; // receiver
            entry.1.push(format!("PAGE[{}][{}] C3 RECV", page_idx, row));

            // C4: SEND final token (all rows)
            let c4_token: Token = (0, addr_lo, addr_hi, ts_lo, ts_hi, fini);
            let entry = token_balance.entry(c4_token).or_insert((0, vec![]));
            entry.0 += 1; // sender
            entry.1.push(format!("PAGE[{}][{}] C4 SEND", page_idx, row));

            // Only print accessed addresses (non-zero timestamp or changed value)
            if ts_lo != 0 || ts_hi != 0 || init != fini {
                println!(
                    "  row {} addr=0x{:08x}_{:08x} init={} fini={} ts={}:",
                    row,
                    addr_hi,
                    addr_lo,
                    init,
                    fini,
                    ts_lo | (ts_hi << 32)
                );
                println!("    C3 RECV init: {:?}", c3_token);
                println!("    C4 SEND final: {:?}", c4_token);
            }
        }
    }

    // === Check for imbalanced memory tokens ===
    println!("\n=== IMBALANCED MEMORY TOKENS (should be empty) ===");
    let mut imbalanced = 0;
    for (token, (balance, sources)) in &token_balance {
        if *balance != 0 {
            println!(
                "IMBALANCED: {:?} balance={} sources={:?}",
                token, balance, sources
            );
            imbalanced += 1;
        }
    }
    if imbalanced == 0 {
        println!("All Memory bus tokens balance!");
    } else {
        println!("Found {} imbalanced memory tokens", imbalanced);
    }

    // === Count IS_BYTE lookups from PAGE (C1 init + C2 fini) ===
    println!("\n=== IS_BYTE Lookup Counts (from PAGE tables) ===");
    let mut is_byte_from_page = [0u64; 256];
    let total_page_rows: usize = traces.pages.iter().map(|p| p.num_rows()).sum();
    for (page_idx, page_trace) in traces.pages.iter().enumerate() {
        let page_size = traces.page_configs[page_idx].page_size;
        for row in 0..page_trace.num_rows().min(page_size) {
            let init = page_trace.main_table.get(row, page_cols::INIT).to_raw() as usize;
            let fini = page_trace.main_table.get(row, page_cols::FINI).to_raw() as usize;
            is_byte_from_page[init] += 1; // C1
            is_byte_from_page[fini] += 1; // C2
        }
    }
    println!(
        "Total PAGE rows: {}, Expected IS_BYTE: {} (2 per row)",
        total_page_rows,
        total_page_rows * 2
    );
    println!(
        "IS_BYTE[0]: {} lookups (most bytes are 0)",
        is_byte_from_page[0]
    );

    // Check if bitwise table has matching multiplicities for IS_BYTE
    // IS_BYTE uses rows 0..255 for each byte value, MU_IS_BYTE is column 17
    use crate::tables::bitwise::cols as bitwise_cols;
    let bitwise_is_byte_mult: u64 = (0..256usize)
        .map(|byte_val| {
            // IS_BYTE lookup is at row with X=byte_val, Y=0, Z=0
            // Row index = X + Y*256 + Z*256*256 = X (for Y=0, Z=0)
            traces
                .bitwise
                .main_table
                .get(byte_val, bitwise_cols::MU_IS_BYTE)
                .to_raw()
        })
        .sum();
    println!(
        "Bitwise IS_BYTE total multiplicity: {}",
        bitwise_is_byte_mult
    );

    // Also count total IS_BYTE lookups expected from collect_bitwise_from_page
    let is_byte_total_from_page: u64 = is_byte_from_page.iter().sum();
    println!(
        "Total IS_BYTE lookups from PAGE (counted): {}",
        is_byte_total_from_page
    );
    println!(
        "Difference: {} (should be 0 if PAGE IS_BYTE matches Bitwise)",
        bitwise_is_byte_mult as i64 - is_byte_total_from_page as i64
    );

    // === Verify PAGE AIR uses correct page_base ===
    println!("\n=== PAGE Configuration Check ===");
    for (idx, config) in traces.page_configs.iter().enumerate() {
        let page_lo = config.page_base & 0xFFFF_FFFF;
        let page_hi = config.page_base >> 32;
        println!(
            "PAGE {}: base=0x{:016x}, page_lo={}, page_hi={}",
            idx, config.page_base, page_lo, page_hi
        );
    }
}

#[test]
fn test_page_trace_values_debug() {
    use crate::tables::page::cols as page_cols;

    let (elf, logs, _instructions) = run_asm_elf("test_sb_sh_8");
    let traces =
        Traces::from_elf_and_logs(&elf, &logs, crate::tables::page::DEFAULT_STACK_SIZE).unwrap();

    println!("=== Checking PAGE trace values for stack addresses ===");

    // Find the stack page (PAGE 3)
    for (i, (trace, config)) in traces
        .pages
        .iter()
        .zip(traces.page_configs.iter())
        .enumerate()
    {
        let page_base = config.page_base;
        if page_base == 0xFFFF_FFFF_FFFF_F000 {
            println!(
                "Found stack page at index {} with base 0x{:016x}",
                i, page_base
            );

            // Check row 4064 (offset 0xFE0, address 0xFFFF_FFFF_FFFF_FFE0)
            let row = 4064;
            let offset = trace.main_table.get(row, page_cols::OFFSET).to_raw();
            let init = trace.main_table.get(row, page_cols::INIT).to_raw();
            let fini = trace.main_table.get(row, page_cols::FINI).to_raw();
            let ts_lo = trace.main_table.get(row, page_cols::TIMESTAMP_LO).to_raw();
            let ts_hi = trace.main_table.get(row, page_cols::TIMESTAMP_HI).to_raw();

            println!("Row {} (addr 0xFFFF_FFFF_FFFF_FFE0):", row);
            println!("  offset = {} (expected 4064)", offset);
            println!("  init = {} (expected 0)", init);
            println!("  fini = {} (expected 66 = 0x42)", fini);
            println!("  ts_lo = {} (expected 24)", ts_lo);
            println!("  ts_hi = {} (expected 0)", ts_hi);

            // Check row 4066 (offset 0xFE2, address 0xFFFF_FFFF_FFFF_FFE2)
            let row = 4066;
            let offset = trace.main_table.get(row, page_cols::OFFSET).to_raw();
            let init = trace.main_table.get(row, page_cols::INIT).to_raw();
            let fini = trace.main_table.get(row, page_cols::FINI).to_raw();
            let ts_lo = trace.main_table.get(row, page_cols::TIMESTAMP_LO).to_raw();

            println!("Row {} (addr 0xFFFF_FFFF_FFFF_FFE2):", row);
            println!("  offset = {} (expected 4066)", offset);
            println!("  init = {} (expected 0)", init);
            println!("  fini = {} (expected 137 = 0x89)", fini);
            println!("  ts_lo = {} (expected 28)", ts_lo);

            // Compute what the AIR would see for address_lo
            let page_lo = page_base & 0xFFFF_FFFF;
            let page_hi = page_base >> 32;
            println!("\nAIR would compute:");
            println!("  page_lo = {} (0x{:08x})", page_lo, page_lo);
            println!("  page_hi = {} (0x{:08x})", page_hi, page_hi);
            println!(
                "  For row 4064: addr_lo = page_lo + offset = {} + 4064 = {}",
                page_lo,
                page_lo + 4064
            );
            println!("  Expected: 0xFFFF_FFE0 = {}", 0xFFFF_FFE0u64);
        }
    }
}

/// Test with a single PAGE table to isolate the issue
#[test]
#[ignore] // Intentionally removes 3 of 4 PAGE tables, so Memory bus won't balance
fn test_single_page_table_balance() {
    let (elf, logs, _instructions) = run_asm_elf("test_sb_sh_8");
    let mut traces =
        Traces::from_elf_and_logs(&elf, &logs, crate::tables::page::DEFAULT_STACK_SIZE).unwrap();

    println!("Original PAGE tables: {}", traces.pages.len());
    println!("PAGE configs:");
    for (i, config) in traces.page_configs.iter().enumerate() {
        println!("  {}: base=0x{:016x}", i, config.page_base);
    }

    // Keep only the stack page (PAGE 3, which contains the accessed addresses)
    let stack_page_idx = traces
        .page_configs
        .iter()
        .position(|c| c.page_base == 0xFFFF_FFFF_FFFF_F000)
        .expect("Stack page not found");

    println!(
        "Using only PAGE {} (stack page at 0xFFFF_FFFF_FFFF_F000)",
        stack_page_idx
    );

    let single_page_trace = traces.pages.remove(stack_page_idx);
    let single_page_config = traces.page_configs.remove(stack_page_idx);

    traces.pages = vec![single_page_trace];
    traces.page_configs = vec![single_page_config];

    // Run the prover
    let result = prove_and_verify_vm_minimal(&elf, &mut traces);

    println!(
        "Single PAGE table test: {}",
        if result { "PASSED" } else { "FAILED" }
    );
    assert!(result, "Single PAGE table test failed");
}

// =============================================================================
// Deep stack tests (page coverage)
// =============================================================================

/// deep_stack allocates 8192 bytes, writing at SP = 0x...DFF0 (page D000).
/// Default stack_size=4096 only creates pages E000+F000, so page D000 is
/// missing and the memory bus cannot balance → verification must fail.
#[test]
fn test_deep_stack_default_stack_size_fails() {
    let (elf, logs, _instructions) = run_asm_elf("deep_stack");
    let mut traces = Traces::from_elf_and_logs(
        &elf,
        &logs,
        crate::tables::page::DEFAULT_STACK_SIZE,
    )
    .unwrap();

    assert!(
        !prove_and_verify_vm_minimal(&elf, &mut traces),
        "deep_stack should FAIL with default stack_size (page D000 not initialized)"
    );
}

/// Same program but with stack_size=8192, which adds page D000.
/// All accessed addresses are now covered → verification must succeed.
#[test]
fn test_deep_stack_large_stack_size_passes() {
    let (elf, logs, _instructions) = run_asm_elf("deep_stack");
    let mut traces = Traces::from_elf_and_logs(&elf, &logs, 8192).unwrap();

    assert!(
        prove_and_verify_vm_minimal(&elf, &mut traces),
        "deep_stack should PASS with stack_size=8192 (page D000 initialized)"
    );
}
