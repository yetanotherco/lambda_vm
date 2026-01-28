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
    collect_lt_lookups_from_logs, create_bitwise_air, create_cpu_air, create_load_air,
    create_lt_air, create_memw_air, generate_minimal_bitwise_trace, run_asm_elf,
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

/// Run multi_prove and multi_verify for all VM tables (CPU + Bitwise + LT + MEMW + LOAD).
///
/// Uses the FULL 2^20 row bitwise table with preprocessed commitment.
/// Returns true if verification succeeds.
fn prove_and_verify_vm(
    cpu_trace: &mut TraceTable<F, E>,
    bitwise_trace: &mut TraceTable<F, E>,
    lt_trace: &mut TraceTable<F, E>,
    memw_trace: &mut TraceTable<F, E>,
    load_trace: &mut TraceTable<F, E>,
) -> bool {
    let proof_options = ProofOptions::default_test_options();

    let cpu_air = create_cpu_air(&proof_options);
    let bitwise_air = create_bitwise_air(&proof_options);
    let lt_air = create_lt_air(&proof_options);
    let memw_air = create_memw_air(&proof_options);
    let load_air = create_load_air(&proof_options);

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
        vec![&cpu_air, &bitwise_air, &lt_air, &memw_air, &load_air];

    Verifier::multi_verify(&airs, &multi_proof, &mut DefaultTranscript::<E>::new(&[]))
}

/// Run multi_prove and multi_verify for all VM tables (CPU + Bitwise + LT + MEMW + LOAD).
///
/// Used for fast tests where the bitwise table is a dummy that only contains
/// the rows needed to balance the bus. NOT the full preprocessed table.
fn prove_and_verify_vm_minimal(
    cpu_trace: &mut TraceTable<F, E>,
    bitwise_trace: &mut TraceTable<F, E>,
    lt_trace: &mut TraceTable<F, E>,
    memw_trace: &mut TraceTable<F, E>,
    load_trace: &mut TraceTable<F, E>,
) -> bool {
    let proof_options = ProofOptions::default_test_options();

    let cpu_air = create_cpu_air(&proof_options);
    let bitwise_air = create_bitwise_air(&proof_options);
    let lt_air = create_lt_air(&proof_options);
    let memw_air = create_memw_air(&proof_options);
    let load_air = create_load_air(&proof_options);

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
    ];

    let multi_proof =
        match Prover::multi_prove(air_trace_pairs, &mut DefaultTranscript::<E>::new(&[])) {
            Ok(proof) => proof,
            Err(e) => {
                println!("=== PROVER ERROR: {:?} ===", e);
                panic!("Prover failed: {:?}", e);
            }
        };

    let airs: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>> =
        vec![&cpu_air, &bitwise_air, &lt_air, &memw_air, &load_air];

    let result = Verifier::multi_verify(&airs, &multi_proof, &mut DefaultTranscript::<E>::new(&[]));
    if !result {
        println!("=== VERIFIER FAILED ===");
    }
    result
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
    let _ = env_logger::builder().is_test(true).try_init();
    let (logs, instructions) = run_asm_elf("sub");
    assert_eq!(logs.len(), 4, "sub.elf should have 4 steps");

    // Use full Traces to get real MEMW trace (includes register operations)
    let mut traces = Traces::from_logs(&logs, instructions.clone()).unwrap();
    let lt_lookups = collect_lt_lookups_from_logs(&logs, &instructions);
    let mut lt_trace = generate_lt_trace(&lt_lookups);
    let mut bitwise_lookups = collect_bitwise_lookups(&logs, &instructions);
    bitwise_lookups.extend(collect_bitwise_lookups_from_lt(&lt_lookups));
    let mut bitwise_trace = generate_minimal_bitwise_trace(&bitwise_lookups);

    println!(
        "Fast SUB: CPU {} rows, Bitwise {} rows (minimal), {} lookups, MEMW {} rows",
        traces.cpu.main_table.height,
        bitwise_trace.main_table.height,
        bitwise_lookups.len(),
        traces.memw.main_table.height
    );

    // Debug: Print CPU register-related columns for each row
    use crate::tables::cpu::cols as cpu_cols;
    use crate::tables::memw::cols as memw_cols;
    for row in 0..traces.cpu.main_table.height {
        let base = row * cpu_cols::NUM_COLUMNS;
        let ts = traces.cpu.main_table.data[base + cpu_cols::TIMESTAMP];
        let rs1 = traces.cpu.main_table.data[base + cpu_cols::RS1];
        let rs2 = traces.cpu.main_table.data[base + cpu_cols::RS2];
        let rd = traces.cpu.main_table.data[base + cpu_cols::RD];
        let rr1 = traces.cpu.main_table.data[base + cpu_cols::READ_REGISTER1];
        let rr2 = traces.cpu.main_table.data[base + cpu_cols::READ_REGISTER2];
        let wr = traces.cpu.main_table.data[base + cpu_cols::WRITE_REGISTER];
        let rv1_0 = traces.cpu.main_table.data[base + cpu_cols::RV1_0];
        let rv1_1 = traces.cpu.main_table.data[base + cpu_cols::RV1_1];
        let rv1_2 = traces.cpu.main_table.data[base + cpu_cols::RV1_2];
        let rvd_0 = traces.cpu.main_table.data[base + cpu_cols::RVD_0];
        let rvd_1 = traces.cpu.main_table.data[base + cpu_cols::RVD_1];
        let m2 = traces.cpu.main_table.data[base + cpu_cols::MEMORY_2BYTES];
        let m4 = traces.cpu.main_table.data[base + cpu_cols::MEMORY_4BYTES];
        let m8 = traces.cpu.main_table.data[base + cpu_cols::MEMORY_8BYTES];
        println!(
            "CPU row {}: ts={:?} rs1={:?} rs2={:?} rd={:?} RR1={:?} RR2={:?} WR={:?} rv1=[{:?},{:?},{:?}] rvd=[{:?},{:?}] mem=[{:?},{:?},{:?}]",
            row, ts, rs1, rs2, rd, rr1, rr2, wr, rv1_0, rv1_1, rv1_2, rvd_0, rvd_1, m2, m4, m8
        );
    }

    // Debug: Print MEMW rows with full VALUE and OLD arrays
    for row in 0..traces.memw.main_table.height {
        let base = row * memw_cols::NUM_COLUMNS;
        let is_reg = traces.memw.main_table.data[base + memw_cols::IS_REGISTER];
        let addr0 = traces.memw.main_table.data[base + memw_cols::BASE_ADDRESS_0];
        let addr1 = traces.memw.main_table.data[base + memw_cols::BASE_ADDRESS_1];
        let ts0 = traces.memw.main_table.data[base + memw_cols::TIMESTAMP_0];
        let ts1 = traces.memw.main_table.data[base + memw_cols::TIMESTAMP_1];
        let mu_r = traces.memw.main_table.data[base + memw_cols::MU_READ];
        let mu_w = traces.memw.main_table.data[base + memw_cols::MU_WRITE];
        // Get all 8 VALUE elements
        let v: Vec<_> = (0..8).map(|i| traces.memw.main_table.data[base + memw_cols::VALUE[i]]).collect();
        // Get all 8 OLD elements
        let old: Vec<_> = (0..8).map(|i| traces.memw.main_table.data[base + memw_cols::OLD[i]]).collect();
        let w2 = traces.memw.main_table.data[base + memw_cols::WRITE2];
        let w4 = traces.memw.main_table.data[base + memw_cols::WRITE4];
        let w8 = traces.memw.main_table.data[base + memw_cols::WRITE8];
        // Get old_timestamp[0] (first 2 elements)
        let old_ts_0_0 = traces.memw.main_table.data[base + memw_cols::OLD_TIMESTAMP_START];
        let old_ts_0_1 = traces.memw.main_table.data[base + memw_cols::OLD_TIMESTAMP_START + 1];
        println!(
            "MEMW row {}: is_reg={:?} addr=[{:?},{:?}] ts=[{:?},{:?}] old_ts0=[{:?},{:?}] mu_r={:?} mu_w={:?}",
            row, is_reg, addr0, addr1, ts0, ts1, old_ts_0_0, old_ts_0_1, mu_r, mu_w
        );
        println!("  VALUE={:?}", v);
        println!("  OLD={:?}", old);
        println!("  w=[{:?},{:?},{:?}]", w2, w4, w8);
    }

    // Debug: Print LOAD trace size
    println!("LOAD trace: {} rows", traces.load.main_table.height);

    assert!(
        prove_and_verify_vm_minimal(
            &mut traces.cpu,
            &mut bitwise_trace,
            &mut lt_trace,
            &mut traces.memw,
            &mut traces.load
        ),
        "Proof verification failed for sub program (fast)"
    );
}

#[test]
fn test_prove_elfs_sub_neg_result_fast() {
    let (logs, instructions) = run_asm_elf("sub_neg_result");
    assert_eq!(logs.len(), 4, "sub_neg_result.elf should have 4 steps");

    let mut traces = Traces::from_logs(&logs, instructions.clone()).unwrap();
    let lt_lookups = collect_lt_lookups_from_logs(&logs, &instructions);
    let mut lt_trace = generate_lt_trace(&lt_lookups);
    let mut bitwise_lookups = collect_bitwise_lookups(&logs, &instructions);
    bitwise_lookups.extend(collect_bitwise_lookups_from_lt(&lt_lookups));
    let mut bitwise_trace = generate_minimal_bitwise_trace(&bitwise_lookups);

    println!(
        "Fast SUB_NEG: CPU {} rows, Bitwise {} rows (minimal), {} lookups",
        traces.cpu.main_table.height,
        bitwise_trace.main_table.height,
        bitwise_lookups.len()
    );

    assert!(
        prove_and_verify_vm_minimal(
            &mut traces.cpu,
            &mut bitwise_trace,
            &mut lt_trace,
            &mut traces.memw,
            &mut traces.load
        ),
        "Proof verification failed for sub_neg_result program (fast)"
    );
}

#[test]
fn test_prove_elfs_sub_underflow_fast() {
    let (logs, instructions) = run_asm_elf("sub_underflow");
    assert_eq!(logs.len(), 4, "sub_underflow.elf should have 4 steps");

    let mut traces = Traces::from_logs(&logs, instructions.clone()).unwrap();
    let lt_lookups = collect_lt_lookups_from_logs(&logs, &instructions);
    let mut lt_trace = generate_lt_trace(&lt_lookups);
    let mut bitwise_lookups = collect_bitwise_lookups(&logs, &instructions);
    bitwise_lookups.extend(collect_bitwise_lookups_from_lt(&lt_lookups));
    let mut bitwise_trace = generate_minimal_bitwise_trace(&bitwise_lookups);

    println!(
        "Fast SUB_UNDERFLOW: CPU {} rows, Bitwise {} rows (minimal), {} lookups",
        traces.cpu.main_table.height,
        bitwise_trace.main_table.height,
        bitwise_lookups.len()
    );

    assert!(
        prove_and_verify_vm_minimal(
            &mut traces.cpu,
            &mut bitwise_trace,
            &mut lt_trace,
            &mut traces.memw,
            &mut traces.load
        ),
        "Proof verification failed for sub_underflow program (fast)"
    );
}

#[test]
fn test_prove_elfs_subw_fast() {
    let (logs, instructions) = run_asm_elf("subw");
    assert_eq!(logs.len(), 4, "subw.elf should have 4 steps");

    let mut traces = Traces::from_logs(&logs, instructions.clone()).unwrap();
    let lt_lookups = collect_lt_lookups_from_logs(&logs, &instructions);
    let mut lt_trace = generate_lt_trace(&lt_lookups);
    let mut bitwise_lookups = collect_bitwise_lookups(&logs, &instructions);
    bitwise_lookups.extend(collect_bitwise_lookups_from_lt(&lt_lookups));
    let mut bitwise_trace = generate_minimal_bitwise_trace(&bitwise_lookups);

    println!(
        "Fast SUBW: CPU {} rows, Bitwise {} rows (minimal), {} lookups",
        traces.cpu.main_table.height,
        bitwise_trace.main_table.height,
        bitwise_lookups.len()
    );

    assert!(
        prove_and_verify_vm_minimal(
            &mut traces.cpu,
            &mut bitwise_trace,
            &mut lt_trace,
            &mut traces.memw,
            &mut traces.load
        ),
        "Proof verification failed for subw program (fast)"
    );
}

/// 8-instruction test with LUI
#[test]
fn test_prove_elfs_arith_lui_8() {
    let (logs, instructions) = run_asm_elf("arith_lui_8");
    assert_eq!(logs.len(), 8, "arith_lui_8.elf should have 8 steps");

    let mut traces = Traces::from_logs(&logs, instructions.clone()).unwrap();
    let lt_lookups = collect_lt_lookups_from_logs(&logs, &instructions);
    let mut lt_trace = generate_lt_trace(&lt_lookups);
    let mut bitwise_lookups = collect_bitwise_lookups(&logs, &instructions);
    bitwise_lookups.extend(collect_bitwise_lookups_from_lt(&lt_lookups));
    let mut bitwise_trace = generate_minimal_bitwise_trace(&bitwise_lookups);

    println!(
        "ArithLUI8: CPU {} rows, Bitwise {} rows (minimal), {} lookups",
        traces.cpu.main_table.height,
        bitwise_trace.main_table.height,
        bitwise_lookups.len()
    );

    assert!(
        prove_and_verify_vm_minimal(
            &mut traces.cpu,
            &mut bitwise_trace,
            &mut lt_trace,
            &mut traces.memw,
            &mut traces.load
        ),
        "Proof verification failed for arith_lui_8 program"
    );
}

/// 8-instruction test with ADD, SUB, ADDW, SUBW
#[test]
fn test_prove_elfs_arith_8() {
    let (logs, instructions) = run_asm_elf("arith_8");
    assert_eq!(logs.len(), 8, "arith_8.elf should have 8 steps");

    let mut traces = Traces::from_logs(&logs, instructions.clone()).unwrap();
    let lt_lookups = collect_lt_lookups_from_logs(&logs, &instructions);
    let mut lt_trace = generate_lt_trace(&lt_lookups);
    let mut bitwise_lookups = collect_bitwise_lookups(&logs, &instructions);
    bitwise_lookups.extend(collect_bitwise_lookups_from_lt(&lt_lookups));
    let mut bitwise_trace = generate_minimal_bitwise_trace(&bitwise_lookups);

    println!(
        "Arith8: CPU {} rows, Bitwise {} rows (minimal), {} lookups",
        traces.cpu.main_table.height,
        bitwise_trace.main_table.height,
        bitwise_lookups.len()
    );

    assert!(
        prove_and_verify_vm_minimal(
            &mut traces.cpu,
            &mut bitwise_trace,
            &mut lt_trace,
            &mut traces.memw,
            &mut traces.load
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
    let (logs, instructions) = run_asm_elf("basic_arith_32");
    assert_eq!(logs.len(), 32, "basic_arith_32.elf should have 32 steps");

    let mut traces = Traces::from_logs(&logs, instructions.clone()).unwrap();
    let lt_lookups = collect_lt_lookups_from_logs(&logs, &instructions);
    let mut lt_trace = generate_lt_trace(&lt_lookups);
    let mut bitwise_lookups = collect_bitwise_lookups(&logs, &instructions);
    bitwise_lookups.extend(collect_bitwise_lookups_from_lt(&lt_lookups));
    let mut bitwise_trace = generate_minimal_bitwise_trace(&bitwise_lookups);

    println!(
        "BasicArith32: CPU {} rows, Bitwise {} rows (minimal), {} lookups",
        traces.cpu.main_table.height,
        bitwise_trace.main_table.height,
        bitwise_lookups.len()
    );

    assert!(
        prove_and_verify_vm_minimal(
            &mut traces.cpu,
            &mut bitwise_trace,
            &mut lt_trace,
            &mut traces.memw,
            &mut traces.load
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

    let (logs, instructions) = run_asm_elf("comprehensive_test");
    assert_eq!(
        logs.len(),
        32,
        "comprehensive_test.elf should have 32 steps"
    );

    let mut traces = Traces::from_logs(&logs, instructions.clone()).unwrap();

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
        traces.cpu.main_table.height,
        bitwise_trace.main_table.height,
        bitwise_lookups.len(),
        lt_lookups.len()
    );

    assert!(
        prove_and_verify_vm_minimal(
            &mut traces.cpu,
            &mut bitwise_trace,
            &mut lt_trace,
            &mut traces.memw,
            &mut traces.load
        ),
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
    let mut traces = Traces::from_logs(&logs, instructions.clone()).unwrap();
    let lt_lookups = collect_lt_lookups_from_logs(&logs, &instructions);
    let mut lt_trace = generate_lt_trace(&lt_lookups);
    let mut bitwise_lookups = collect_bitwise_lookups(&logs, &instructions);
    bitwise_lookups.extend(collect_bitwise_lookups_from_lt(&lt_lookups));
    let mut bitwise_trace = generate_minimal_bitwise_trace(&bitwise_lookups);
    println!("test_add_8: {} lookups", bitwise_lookups.len());
    assert!(
        prove_and_verify_vm_minimal(
            &mut traces.cpu,
            &mut bitwise_trace,
            &mut lt_trace,
            &mut traces.memw,
            &mut traces.load
        ),
        "test_add_8 failed"
    );
}

#[test]
fn test_prove_elfs_test_sub_8() {
    let (logs, instructions) = run_asm_elf("test_sub_8");
    assert_eq!(logs.len(), 8);
    let mut traces = Traces::from_logs(&logs, instructions.clone()).unwrap();
    let lt_lookups = collect_lt_lookups_from_logs(&logs, &instructions);
    let mut lt_trace = generate_lt_trace(&lt_lookups);
    let mut bitwise_lookups = collect_bitwise_lookups(&logs, &instructions);
    bitwise_lookups.extend(collect_bitwise_lookups_from_lt(&lt_lookups));
    let mut bitwise_trace = generate_minimal_bitwise_trace(&bitwise_lookups);
    println!("test_sub_8: {} lookups", bitwise_lookups.len());
    assert!(
        prove_and_verify_vm_minimal(
            &mut traces.cpu,
            &mut bitwise_trace,
            &mut lt_trace,
            &mut traces.memw,
            &mut traces.load
        ),
        "test_sub_8 failed"
    );
}

#[test]
fn test_prove_elfs_test_addw_8() {
    let (logs, instructions) = run_asm_elf("test_addw_8");
    assert_eq!(logs.len(), 8);
    let mut traces = Traces::from_logs(&logs, instructions.clone()).unwrap();
    let lt_lookups = collect_lt_lookups_from_logs(&logs, &instructions);
    let mut lt_trace = generate_lt_trace(&lt_lookups);
    let mut bitwise_lookups = collect_bitwise_lookups(&logs, &instructions);
    bitwise_lookups.extend(collect_bitwise_lookups_from_lt(&lt_lookups));
    let mut bitwise_trace = generate_minimal_bitwise_trace(&bitwise_lookups);
    println!("test_addw_8: {} lookups", bitwise_lookups.len());
    assert!(
        prove_and_verify_vm_minimal(
            &mut traces.cpu,
            &mut bitwise_trace,
            &mut lt_trace,
            &mut traces.memw,
            &mut traces.load
        ),
        "test_addw_8 failed"
    );
}

#[test]
fn test_prove_elfs_test_subw_8() {
    let (logs, instructions) = run_asm_elf("test_subw_8");
    assert_eq!(logs.len(), 8);
    let mut traces = Traces::from_logs(&logs, instructions.clone()).unwrap();
    let lt_lookups = collect_lt_lookups_from_logs(&logs, &instructions);
    let mut lt_trace = generate_lt_trace(&lt_lookups);
    let mut bitwise_lookups = collect_bitwise_lookups(&logs, &instructions);
    bitwise_lookups.extend(collect_bitwise_lookups_from_lt(&lt_lookups));
    let mut bitwise_trace = generate_minimal_bitwise_trace(&bitwise_lookups);
    println!("test_subw_8: {} lookups", bitwise_lookups.len());
    assert!(
        prove_and_verify_vm_minimal(
            &mut traces.cpu,
            &mut bitwise_trace,
            &mut lt_trace,
            &mut traces.memw,
            &mut traces.load
        ),
        "test_subw_8 failed"
    );
}

#[test]
fn test_prove_elfs_test_addw_lui_8() {
    let (logs, instructions) = run_asm_elf("test_addw_lui_8");
    assert_eq!(logs.len(), 8);
    let mut traces = Traces::from_logs(&logs, instructions.clone()).unwrap();
    let lt_lookups = collect_lt_lookups_from_logs(&logs, &instructions);
    let mut lt_trace = generate_lt_trace(&lt_lookups);
    let mut bitwise_lookups = collect_bitwise_lookups(&logs, &instructions);
    bitwise_lookups.extend(collect_bitwise_lookups_from_lt(&lt_lookups));
    let mut bitwise_trace = generate_minimal_bitwise_trace(&bitwise_lookups);
    println!("test_addw_lui_8: {} lookups", bitwise_lookups.len());
    assert!(
        prove_and_verify_vm_minimal(
            &mut traces.cpu,
            &mut bitwise_trace,
            &mut lt_trace,
            &mut traces.memw,
            &mut traces.load
        ),
        "test_addw_lui_8 failed"
    );
}

#[test]
fn test_prove_elfs_test_subw_lui_8() {
    let (logs, instructions) = run_asm_elf("test_subw_lui_8");
    assert_eq!(logs.len(), 8);
    let mut traces = Traces::from_logs(&logs, instructions.clone()).unwrap();
    let lt_lookups = collect_lt_lookups_from_logs(&logs, &instructions);
    let mut lt_trace = generate_lt_trace(&lt_lookups);
    let mut bitwise_lookups = collect_bitwise_lookups(&logs, &instructions);
    bitwise_lookups.extend(collect_bitwise_lookups_from_lt(&lt_lookups));
    let mut bitwise_trace = generate_minimal_bitwise_trace(&bitwise_lookups);
    println!("test_subw_lui_8: {} lookups", bitwise_lookups.len());
    assert!(
        prove_and_verify_vm_minimal(
            &mut traces.cpu,
            &mut bitwise_trace,
            &mut lt_trace,
            &mut traces.memw,
            &mut traces.load
        ),
        "test_subw_lui_8 failed"
    );
}

#[test]
fn test_prove_elfs_test_add_neg_8() {
    let (logs, instructions) = run_asm_elf("test_add_neg_8");
    assert_eq!(logs.len(), 8);
    let mut traces = Traces::from_logs(&logs, instructions.clone()).unwrap();
    let lt_lookups = collect_lt_lookups_from_logs(&logs, &instructions);
    let mut lt_trace = generate_lt_trace(&lt_lookups);
    let mut bitwise_lookups = collect_bitwise_lookups(&logs, &instructions);
    bitwise_lookups.extend(collect_bitwise_lookups_from_lt(&lt_lookups));
    let mut bitwise_trace = generate_minimal_bitwise_trace(&bitwise_lookups);
    println!("test_add_neg_8: {} lookups", bitwise_lookups.len());
    assert!(
        prove_and_verify_vm_minimal(
            &mut traces.cpu,
            &mut bitwise_trace,
            &mut lt_trace,
            &mut traces.memw,
            &mut traces.load
        ),
        "test_add_neg_8 failed"
    );
}

#[test]
fn test_prove_elfs_test_sub_neg_8() {
    let (logs, instructions) = run_asm_elf("test_sub_neg_8");
    assert_eq!(logs.len(), 8);
    let mut traces = Traces::from_logs(&logs, instructions.clone()).unwrap();
    let lt_lookups = collect_lt_lookups_from_logs(&logs, &instructions);
    let mut lt_trace = generate_lt_trace(&lt_lookups);
    let mut bitwise_lookups = collect_bitwise_lookups(&logs, &instructions);
    bitwise_lookups.extend(collect_bitwise_lookups_from_lt(&lt_lookups));
    let mut bitwise_trace = generate_minimal_bitwise_trace(&bitwise_lookups);
    println!("test_sub_neg_8: {} lookups", bitwise_lookups.len());
    assert!(
        prove_and_verify_vm_minimal(
            &mut traces.cpu,
            &mut bitwise_trace,
            &mut lt_trace,
            &mut traces.memw,
            &mut traces.load
        ),
        "test_sub_neg_8 failed"
    );
}

#[test]
fn test_prove_elfs_test_mul_8() {
    let (logs, instructions) = run_asm_elf("test_mul_8");
    assert_eq!(logs.len(), 8);
    let mut traces = Traces::from_logs(&logs, instructions.clone()).unwrap();
    let lt_lookups = collect_lt_lookups_from_logs(&logs, &instructions);
    let mut lt_trace = generate_lt_trace(&lt_lookups);
    let mut bitwise_lookups = collect_bitwise_lookups(&logs, &instructions);
    bitwise_lookups.extend(collect_bitwise_lookups_from_lt(&lt_lookups));
    let mut bitwise_trace = generate_minimal_bitwise_trace(&bitwise_lookups);
    println!("test_mul_8: {} lookups", bitwise_lookups.len());
    assert!(
        prove_and_verify_vm_minimal(
            &mut traces.cpu,
            &mut bitwise_trace,
            &mut lt_trace,
            &mut traces.memw,
            &mut traces.load
        ),
        "test_mul_8 failed"
    );
}

#[test]
fn test_prove_elfs_test_div_8() {
    let (logs, instructions) = run_asm_elf("test_div_8");
    assert_eq!(logs.len(), 8);
    let mut traces = Traces::from_logs(&logs, instructions.clone()).unwrap();
    let lt_lookups = collect_lt_lookups_from_logs(&logs, &instructions);
    let mut lt_trace = generate_lt_trace(&lt_lookups);
    let mut bitwise_lookups = collect_bitwise_lookups(&logs, &instructions);
    bitwise_lookups.extend(collect_bitwise_lookups_from_lt(&lt_lookups));
    let mut bitwise_trace = generate_minimal_bitwise_trace(&bitwise_lookups);
    println!("test_div_8: {} lookups", bitwise_lookups.len());
    assert!(
        prove_and_verify_vm_minimal(
            &mut traces.cpu,
            &mut bitwise_trace,
            &mut lt_trace,
            &mut traces.memw,
            &mut traces.load
        ),
        "test_div_8 failed"
    );
}

#[test]
fn test_prove_elfs_test_shift_8() {
    let (logs, instructions) = run_asm_elf("test_shift_8");
    assert_eq!(logs.len(), 8);
    let mut traces = Traces::from_logs(&logs, instructions.clone()).unwrap();
    let lt_lookups = collect_lt_lookups_from_logs(&logs, &instructions);
    let mut lt_trace = generate_lt_trace(&lt_lookups);
    let mut bitwise_lookups = collect_bitwise_lookups(&logs, &instructions);
    bitwise_lookups.extend(collect_bitwise_lookups_from_lt(&lt_lookups));
    let mut bitwise_trace = generate_minimal_bitwise_trace(&bitwise_lookups);
    println!("test_shift_8: {} lookups", bitwise_lookups.len());
    assert!(
        prove_and_verify_vm_minimal(
            &mut traces.cpu,
            &mut bitwise_trace,
            &mut lt_trace,
            &mut traces.memw,
            &mut traces.load
        ),
        "test_shift_8 failed"
    );
}

#[test]
fn test_prove_elfs_test_bitwise_8() {
    let (logs, instructions) = run_asm_elf("test_bitwise_8");
    assert_eq!(logs.len(), 8);
    let mut traces = Traces::from_logs(&logs, instructions.clone()).unwrap();
    let lt_lookups = collect_lt_lookups_from_logs(&logs, &instructions);
    let mut lt_trace = generate_lt_trace(&lt_lookups);
    let mut bitwise_lookups = collect_bitwise_lookups(&logs, &instructions);
    bitwise_lookups.extend(collect_bitwise_lookups_from_lt(&lt_lookups));
    let mut bitwise_trace = generate_minimal_bitwise_trace(&bitwise_lookups);
    println!("test_bitwise_8: {} lookups", bitwise_lookups.len());
    assert!(
        prove_and_verify_vm_minimal(
            &mut traces.cpu,
            &mut bitwise_trace,
            &mut lt_trace,
            &mut traces.memw,
            &mut traces.load
        ),
        "test_bitwise_8 failed"
    );
}

#[test]
fn test_prove_elfs_test_slt_8() {
    // Initialize logger to see debug constraint validation output
    let _ = env_logger::builder().is_test(true).try_init();

    let (logs, instructions) = run_asm_elf("test_slt_8");
    assert_eq!(logs.len(), 8);
    let mut traces = Traces::from_logs(&logs, instructions.clone()).unwrap();

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
        prove_and_verify_vm_minimal(
            &mut traces.cpu,
            &mut bitwise_trace,
            &mut lt_trace,
            &mut traces.memw,
            &mut traces.load
        ),
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
    let mut traces = Traces::from_logs(&logs, instructions.clone()).unwrap();
    let lt_lookups = collect_lt_lookups_from_logs(&logs, &instructions);
    let mut lt_trace = generate_lt_trace(&lt_lookups);
    let mut bitwise_lookups = collect_bitwise_lookups(&logs, &instructions);
    bitwise_lookups.extend(collect_bitwise_lookups_from_lt(&lt_lookups));
    let mut bitwise_trace = generate_minimal_bitwise_trace(&bitwise_lookups);
    println!("test_xor_8: {} lookups", bitwise_lookups.len());
    assert!(
        prove_and_verify_vm_minimal(
            &mut traces.cpu,
            &mut bitwise_trace,
            &mut lt_trace,
            &mut traces.memw,
            &mut traces.load
        ),
        "test_xor_8 failed"
    );
}

#[test]
fn test_prove_elfs_test_lb_lh_8() {
    let (logs, instructions) = run_asm_elf("test_lb_lh_8");
    assert_eq!(logs.len(), 8);
    let mut traces = Traces::from_logs(&logs, instructions.clone()).unwrap();
    let lt_lookups = collect_lt_lookups_from_logs(&logs, &instructions);
    let mut lt_trace = generate_lt_trace(&lt_lookups);
    let mut bitwise_lookups = collect_bitwise_lookups(&logs, &instructions);
    bitwise_lookups.extend(collect_bitwise_lookups_from_lt(&lt_lookups));
    let mut bitwise_trace = generate_minimal_bitwise_trace(&bitwise_lookups);
    println!("test_lb_lh_8: {} lookups", bitwise_lookups.len());
    assert!(
        prove_and_verify_vm_minimal(
            &mut traces.cpu,
            &mut bitwise_trace,
            &mut lt_trace,
            &mut traces.memw,
            &mut traces.load
        ),
        "test_lb_lh_8 failed"
    );
}

#[test]
fn test_prove_elfs_test_sb_sh_8() {
    let (logs, instructions) = run_asm_elf("test_sb_sh_8");
    assert_eq!(logs.len(), 8);
    let mut traces = Traces::from_logs(&logs, instructions.clone()).unwrap();
    let lt_lookups = collect_lt_lookups_from_logs(&logs, &instructions);
    let mut lt_trace = generate_lt_trace(&lt_lookups);
    let mut bitwise_lookups = collect_bitwise_lookups(&logs, &instructions);
    bitwise_lookups.extend(collect_bitwise_lookups_from_lt(&lt_lookups));
    let mut bitwise_trace = generate_minimal_bitwise_trace(&bitwise_lookups);
    println!("test_sb_sh_8: {} lookups", bitwise_lookups.len());
    assert!(
        prove_and_verify_vm_minimal(
            &mut traces.cpu,
            &mut bitwise_trace,
            &mut lt_trace,
            &mut traces.memw,
            &mut traces.load
        ),
        "test_sb_sh_8 failed"
    );
}

#[test]
fn test_prove_elfs_all_branches_16() {
    // Initialize logger to see debug constraint validation output
    let _ = env_logger::builder().is_test(true).try_init();

    let (logs, instructions) = run_asm_elf("all_branches_16");
    assert_eq!(logs.len(), 16);
    let mut traces = Traces::from_logs(&logs, instructions.clone()).unwrap();

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
        prove_and_verify_vm_minimal(
            &mut traces.cpu,
            &mut bitwise_trace,
            &mut lt_trace,
            &mut traces.memw,
            &mut traces.load
        ),
        "all_branches_16 failed"
    );
}

#[test]
fn test_prove_elfs_all_loadstore_32() {
    let (logs, instructions) = run_asm_elf("all_loadstore_32");
    assert_eq!(logs.len(), 32);
    // Use full Traces to get real MEMW and LOAD traces
    let mut traces = Traces::from_logs(&logs, instructions.clone()).unwrap();
    let lt_lookups = collect_lt_lookups_from_logs(&logs, &instructions);
    let mut lt_trace = generate_lt_trace(&lt_lookups);
    let mut bitwise_lookups = collect_bitwise_lookups(&logs, &instructions);
    bitwise_lookups.extend(collect_bitwise_lookups_from_lt(&lt_lookups));
    let mut bitwise_trace = generate_minimal_bitwise_trace(&bitwise_lookups);
    println!(
        "all_loadstore_32: {} CPU rows, {} MEMW rows, {} LOAD rows, {} bitwise lookups",
        traces.cpu.main_table.height,
        traces.memw.main_table.height,
        traces.load.main_table.height,
        bitwise_lookups.len()
    );

    // Debug: Compare CPU LOAD sends vs LOAD receives
    use crate::tables::cpu::cols as cpu_cols;
    use crate::tables::load::cols as load_cols;
    let mut load_idx = 0;
    for row in 0..traces.cpu.main_table.height {
        let base = row * cpu_cols::NUM_COLUMNS;
        let is_load = traces.cpu.main_table.data[base + cpu_cols::LOAD];
        if is_load == crate::tables::types::FE::one() {
            // CPU row with LOAD=1
            let rvd_0 = traces.cpu.main_table.data[base + cpu_cols::RVD_0];
            let rvd_1 = traces.cpu.main_table.data[base + cpu_cols::RVD_1];
            let ts = traces.cpu.main_table.data[base + cpu_cols::TIMESTAMP];
            let m2 = traces.cpu.main_table.data[base + cpu_cols::MEMORY_2BYTES];
            let m4 = traces.cpu.main_table.data[base + cpu_cols::MEMORY_4BYTES];
            let m8 = traces.cpu.main_table.data[base + cpu_cols::MEMORY_8BYTES];
            // CPU RES (base_address) as bytes
            let cpu_res: Vec<_> = (0..8).map(|i| traces.cpu.main_table.data[base + cpu_cols::RES[i]]).collect();

            // Corresponding LOAD row
            let load_base = load_idx * load_cols::NUM_COLUMNS;
            let load_res: Vec<_> = (0..8).map(|i| traces.load.main_table.data[load_base + load_cols::RES[i]]).collect();
            let load_addr_0 = traces.load.main_table.data[load_base + load_cols::BASE_ADDRESS_0];
            let load_addr_1 = traces.load.main_table.data[load_base + load_cols::BASE_ADDRESS_1];
            let load_ts_0 = traces.load.main_table.data[load_base + load_cols::TIMESTAMP_0];
            let load_ts_1 = traces.load.main_table.data[load_base + load_cols::TIMESTAMP_1];
            let load_r2 = traces.load.main_table.data[load_base + load_cols::READ2];
            let load_r4 = traces.load.main_table.data[load_base + load_cols::READ4];
            let load_r8 = traces.load.main_table.data[load_base + load_cols::READ8];
            let load_mu = traces.load.main_table.data[load_base + load_cols::MU];

            println!("CPU row {} (LOAD): rvd=[{:?},{:?}] ts={:?} mem=[{:?},{:?},{:?}]",
                row, rvd_0, rvd_1, ts, m2, m4, m8);
            println!("  CPU res (addr bytes)={:?}", cpu_res);
            println!("LOAD row {}: mu={:?} res={:?}", load_idx, load_mu, load_res);
            println!("  addr=[{:?},{:?}] ts=[{:?},{:?}] read=[{:?},{:?},{:?}]",
                load_addr_0, load_addr_1, load_ts_0, load_ts_1, load_r2, load_r4, load_r8);
            load_idx += 1;
        }
    }
    println!("Total LOAD operations: {}", load_idx);

    // Validate LOAD extension constraints on trace data
    println!("\n=== Validating LOAD extension constraints ===");
    use crate::tables::types::FE;
    let ff = FE::from(255u64);
    for row in 0..traces.load.main_table.height {
        let base = row * load_cols::NUM_COLUMNS;
        let mu = traces.load.main_table.data[base + load_cols::MU];
        let read2 = traces.load.main_table.data[base + load_cols::READ2];
        let read4 = traces.load.main_table.data[base + load_cols::READ4];
        let read8 = traces.load.main_table.data[base + load_cols::READ8];
        let signed = traces.load.main_table.data[base + load_cols::SIGNED];
        let sign_bit = traces.load.main_table.data[base + load_cols::SIGN_BIT];
        let res: Vec<_> = (0..8).map(|i| traces.load.main_table.data[base + load_cols::RES[i]]).collect();

        if mu == FE::one() {
            // Check extension constraints
            let expected_fill = signed * sign_bit * ff;

            // ExtensionHigh: if !read8, res[4..8] must be sign-extended
            if read8 == FE::zero() {
                for i in 4..8 {
                    if res[i] != expected_fill {
                        println!("LOAD row {} FAIL ExtensionHigh({}): res[{}]={:?} expected {:?} (signed={:?} sign_bit={:?})",
                            row, i, i, res[i], expected_fill, signed, sign_bit);
                    }
                }
            }
            // ExtensionMid: if !read4 && !read8, res[2..4] must be sign-extended
            if read4 == FE::zero() && read8 == FE::zero() {
                for i in 2..4 {
                    if res[i] != expected_fill {
                        println!("LOAD row {} FAIL ExtensionMid({}): res[{}]={:?} expected {:?}",
                            row, i, i, res[i], expected_fill);
                    }
                }
            }
            // ExtensionLow: if !read2 && !read4 && !read8 (1-byte load), res[1] must be sign-extended
            if read2 == FE::zero() && read4 == FE::zero() && read8 == FE::zero() {
                if res[1] != expected_fill {
                    println!("LOAD row {} FAIL ExtensionLow: res[1]={:?} expected {:?}",
                        row, res[1], expected_fill);
                }
            }
        }
    }
    println!("=== LOAD constraint validation complete ===\n");

    assert!(
        prove_and_verify_vm_minimal(
            &mut traces.cpu,
            &mut bitwise_trace,
            &mut lt_trace,
            &mut traces.memw,
            &mut traces.load
        ),
        "all_loadstore_32 failed"
    );
}

/// Fast version using minimal bitwise table for debugging
#[test]
fn test_prove_elfs_all_instructions_64() {
    let _ = env_logger::builder().is_test(true).try_init();

    let (logs, instructions) = run_asm_elf("all_instructions_64");
    assert_eq!(logs.len(), 64);
    // Use full Traces to get real MEMW and LOAD traces
    let mut traces = Traces::from_logs(&logs, instructions.clone()).unwrap();

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
        "all_instructions_64 (fast): {} bitwise lookups, {} lt lookups, {} MEMW rows, {} LOAD rows",
        bitwise_lookups.len(),
        lt_lookups.len(),
        traces.memw.main_table.height,
        traces.load.main_table.height
    );
    assert!(
        prove_and_verify_vm_minimal(
            &mut traces.cpu,
            &mut bitwise_trace,
            &mut lt_trace,
            &mut traces.memw,
            &mut traces.load
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

    let (logs, instructions) = run_asm_elf("all_instructions_64");
    assert_eq!(logs.len(), 64);
    let mut traces = Traces::from_logs(&logs, instructions.clone()).unwrap();

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
        traces.cpu.main_table.height,
        bitwise_trace.main_table.height,
        bitwise_lookups.len(),
        lt_lookups.len()
    );

    assert!(
        prove_and_verify_vm(
            &mut traces.cpu,
            &mut bitwise_trace,
            &mut lt_trace,
            &mut traces.memw,
            &mut traces.load
        ),
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

    let mut traces = Traces::from_logs(&logs, instructions.clone()).unwrap();
    let lt_lookups = collect_lt_lookups_from_logs(&logs, &instructions);
    let mut lt_trace = generate_lt_trace(&lt_lookups);
    let mut bitwise_lookups = collect_bitwise_lookups(&logs, &instructions);
    bitwise_lookups.extend(collect_bitwise_lookups_from_lt(&lt_lookups));
    let mut bitwise_trace = generate_minimal_bitwise_trace(&bitwise_lookups);

    println!(
        "sign_ext_edge_cases_8: CPU {} rows, {} bitwise lookups",
        traces.cpu.main_table.height,
        bitwise_lookups.len()
    );

    assert!(
        prove_and_verify_vm_minimal(
            &mut traces.cpu,
            &mut bitwise_trace,
            &mut lt_trace,
            &mut traces.memw,
            &mut traces.load
        ),
        "sign_ext_edge_cases_8 failed - arg2 sign extension may be broken"
    );
}
