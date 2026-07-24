//! VM Prover integration tests using multi_prove.
//!
//! These tests verify the full prover pipeline:
//! - Run ELF through executor
//! - Generate traces for CPU and Bitwise tables
//! - Use multi_prove/multi_verify with bus interactions
//!
//! Wired buses:
//! - Byte-level AND/OR/XOR lookups are routed through BYTE_ALU
//! - CPU sends MSB16 to Bitwise (for rv1_sign_bit, arg2_sign_bit when word_instr=1)
//! - CPU sends MSB8 to Bitwise (for res_sign_bit when word_instr=1)
//! - CPU sends ZERO to Bitwise (for is_equal when BEQ=1)
//!
//! TODO: LT bus (needs LT table integration)

use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use math::field::element::FieldElement;
use stark::constraints::builder::EmptyConstraints;
use stark::lookup::{AirWithBuses, AuxiliaryTraceBuildData};
use stark::proof::options::ProofOptions;
use stark::proof::view::{MultiProofView, StarkProofView};
use stark::traits::AIR;
use stark::verifier::{IsStarkVerifier, Verifier};

use crate::VmProof;
use crate::tables::MaxRowsConfig;
use crate::tables::trace_builder::Traces;
use crate::tables::types::{GoldilocksExtension, GoldilocksField};

use executor::elf::Elf;
use executor::vm::execution::Executor;

// Import shared utilities
use crate::VmAirs;
use crate::test_utils::multi_prove_ram;
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
    let _ = env_logger::builder().is_test(true).try_init();
    let proof_options = ProofOptions::default_test_options();

    // Create all AIRs including PAGE and REGISTER tables
    let table_counts = traces.table_counts();
    let airs = VmAirs::new(
        elf,
        &proof_options,
        true,
        &traces.page_configs,
        &table_counts,
        None,
        true,
        None,
        None,
        None,
    );

    // Build air_trace_pairs for all tables
    let air_trace_pairs = airs.air_trace_pairs(traces);

    let multi_proof = match multi_prove_ram(air_trace_pairs, &mut DefaultTranscript::<E>::new(&[]))
    {
        Ok(proof) => proof,
        Err(_) => return false,
    };

    // Compute the verifier-side expected COMMIT bus balance from public output bytes
    let views: Vec<StarkProofView<F, E, ()>> = multi_proof
        .proofs
        .iter()
        .map(StarkProofView::Owned)
        .collect();
    let mut replay_transcript = DefaultTranscript::<E>::new(&[]);
    let expected_bus_balance = crate::compute_expected_commit_bus_balance_view(
        &airs.air_refs(),
        &views,
        &traces.public_output_bytes,
        0,
        &mut replay_transcript,
    )
    .expect("fingerprint collision in test");

    // Verify using centralized air_refs() which includes all tables
    Verifier::multi_verify_views(
        &airs.air_refs(),
        &views,
        &mut DefaultTranscript::<E>::new(&[]),
        &expected_bus_balance,
    )
}

/// Like [`crate::prove_with_options_and_inputs`] but trims the bitwise table to the
/// rows the program uses instead of proving the full 2^20-row table (TEST ONLY).
///
/// Same unsoundness caveats as [`Traces::from_elf_and_logs_minimal`]. The full
/// preprocessed bitwise path is covered by `test_prove_elfs_all_instructions_64_full`.
fn prove_vm_minimal(elf_bytes: &[u8], private_inputs: &[u8], max_rows: &MaxRowsConfig) -> VmProof {
    let proof_options = ProofOptions::default_test_options();
    let elf = Elf::load(elf_bytes).expect("ELF load");
    let executor = Executor::new(&elf, private_inputs.to_vec()).expect("executor");
    let result = executor.run().expect("execution");
    let mut traces =
        Traces::from_elf_and_logs_minimal(&elf, &result.logs, max_rows, private_inputs).unwrap();
    let table_counts = traces.table_counts();
    let airs = VmAirs::new(
        &elf,
        &proof_options,
        true,
        &traces.page_configs,
        &table_counts,
        None,
        true,
        None,
        None,
        None,
    );
    let runtime_page_ranges = traces.runtime_page_ranges();
    let proof = multi_prove_ram(
        airs.air_trace_pairs(&mut traces),
        &mut DefaultTranscript::<E>::new(&[]),
    )
    .expect("prove");
    let num_private_input_pages = traces
        .page_configs
        .iter()
        .filter(|c| c.is_private_input)
        .count();
    VmProof {
        proof,
        runtime_page_ranges,
        table_counts,
        public_output: traces.public_output_bytes.clone(),
        num_private_input_pages,
    }
}

/// Like [`crate::verify_with_options`] but matches the minimal bitwise AIR.
///
/// Must be used to verify proofs from [`prove_vm_minimal`].
fn verify_vm_minimal(vm_proof: &VmProof, elf_bytes: &[u8]) -> bool {
    let proof_options = ProofOptions::default_test_options();
    let elf = Elf::load(elf_bytes).expect("ELF load");
    let page_configs = Traces::page_configs_from_elf_and_runtime(
        &elf,
        &vm_proof.runtime_page_ranges,
        vm_proof.num_private_input_pages,
    );
    let airs = VmAirs::new(
        &elf,
        &proof_options,
        true,
        &page_configs,
        &vm_proof.table_counts,
        None,
        true,
        None,
        None,
        None,
    );
    let air_refs = airs.air_refs();
    let views: Vec<StarkProofView<F, E, ()>> = vm_proof
        .proof
        .proofs
        .iter()
        .map(StarkProofView::Owned)
        .collect();
    let mut replay_transcript = DefaultTranscript::<E>::new(&[]);
    let expected_bus_balance = crate::compute_expected_commit_bus_balance_view(
        &air_refs,
        &views,
        &vm_proof.public_output,
        0,
        &mut replay_transcript,
    )
    .expect("fingerprint collision in test");
    Verifier::multi_verify_views(
        &air_refs,
        &views,
        &mut DefaultTranscript::<E>::new(&[]),
        &expected_bus_balance,
    )
}

// =============================================================================
// Integration tests
// =============================================================================

/// Test CPU table alone (no bus interactions) to verify basic prove/verify works.
#[test]
fn test_cpu_only_no_bus() {
    let (_elf, logs, instructions) = run_asm_elf("sub");

    let mut cpu_trace = Traces::from_logs(&logs, instructions, &Default::default())
        .unwrap()
        .cpus
        .into_iter()
        .next()
        .unwrap();
    println!(
        "CPU trace: {} rows x {} cols",
        cpu_trace.main_table.height, cpu_trace.main_table.width
    );

    let proof_options = ProofOptions::default_test_options();

    // Create AIR with NO bus interactions
    let auxiliary_trace_build_data = AuxiliaryTraceBuildData {
        interactions: vec![], // NO bus interactions
    };
    let cpu_air: AirWithBuses<
        F,
        E,
        stark::lookup::NullBoundaryConstraintBuilder,
        (),
        EmptyConstraints,
    > = AirWithBuses::new(
        crate::tables::cpu::cols::NUM_COLUMNS,
        auxiliary_trace_build_data,
        &proof_options,
        1,
        EmptyConstraints,
    );

    let air_trace_pairs: Vec<(
        &dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>,
        _,
        _,
    )> = vec![(&cpu_air, &mut cpu_trace, &())];

    let multi_proof = multi_prove_ram(air_trace_pairs, &mut DefaultTranscript::<E>::new(&[]))
        .expect("Prover failed");

    let airs: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>> = vec![&cpu_air];
    assert!(
        Verifier::multi_verify(
            &airs,
            &multi_proof,
            &mut DefaultTranscript::<E>::new(&[]),
            &FieldElement::zero(),
        ),
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
    // Use from_elf_and_logs_minimal to get PAGE and REGISTER tables for Memory bus
    let mut traces =
        Traces::from_elf_and_logs_minimal(&elf, &logs, &Default::default(), &[]).unwrap();

    assert!(
        prove_and_verify_vm_minimal(&elf, &mut traces),
        "Proof verification failed for sub program (fast)"
    );
}

#[test]
fn test_prove_elfs_sub_neg_result_fast() {
    let (elf, logs, instructions) = run_asm_elf("sub_neg_result");
    let mut traces =
        Traces::from_logs_minimal(&logs, instructions.clone(), &Default::default()).unwrap();

    println!(
        "Fast SUB_NEG: CPU {} rows, Bitwise {} rows, MEMW {} tables ({} rows in first), REGISTER {} rows",
        traces.cpus[0].main_table.height,
        traces.bitwise.main_table.height,
        traces.memws.len(),
        traces.memws[0].main_table.height,
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
    let mut traces =
        Traces::from_logs_minimal(&logs, instructions.clone(), &Default::default()).unwrap();

    println!(
        "Fast SUB_UNDERFLOW: CPU {} rows, Bitwise {} rows (minimal)",
        traces.cpus[0].main_table.height, traces.bitwise.main_table.height,
    );

    assert!(
        prove_and_verify_vm_minimal(&elf, &mut traces),
        "Proof verification failed for sub_underflow program (fast)"
    );
}

#[test]
fn test_prove_elfs_subw_fast() {
    let (elf, logs, instructions) = run_asm_elf("subw");
    let mut traces =
        Traces::from_logs_minimal(&logs, instructions.clone(), &Default::default()).unwrap();

    println!(
        "Fast SUBW: CPU {} rows, Bitwise {} rows (minimal)",
        traces.cpus[0].main_table.height, traces.bitwise.main_table.height,
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
    let mut traces =
        Traces::from_logs_minimal(&logs, instructions.clone(), &Default::default()).unwrap();

    println!(
        "ArithLUI8: CPU {} rows, Bitwise {} rows (minimal)",
        traces.cpus[0].main_table.height, traces.bitwise.main_table.height,
    );

    assert!(
        prove_and_verify_vm_minimal(&elf, &mut traces),
        "Proof verification failed for arith_lui_8 program"
    );
}

// Test AUIPC.
// AUIPC uses rs1=x255 (PC register).
// read_register1 must be 1 for rs1≠0, triggering the MEMW M1 interaction.
#[test]
fn test_prove_elfs_auipc() {
    let (elf, logs, instructions) = run_asm_elf("auipc");
    let mut traces =
        Traces::from_logs_minimal(&logs, instructions.clone(), &Default::default()).unwrap();
    assert!(
        prove_and_verify_vm_minimal(&elf, &mut traces),
        "auipc failed"
    );
}

/// 8-instruction test with ADD, SUB, ADDW, SUBW
#[test]
fn test_prove_elfs_arith_8() {
    let (elf, logs, instructions) = run_asm_elf("arith_8");
    let mut traces =
        Traces::from_logs_minimal(&logs, instructions.clone(), &Default::default()).unwrap();

    println!(
        "Arith8: CPU {} rows, Bitwise {} rows (minimal)",
        traces.cpus[0].main_table.height, traces.bitwise.main_table.height,
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
    let mut traces =
        Traces::from_logs_minimal(&logs, instructions.clone(), &Default::default()).unwrap();

    println!(
        "BasicArith32: CPU {} rows, Bitwise {} rows (minimal)",
        traces.cpus[0].main_table.height, traces.bitwise.main_table.height,
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
    let mut traces =
        Traces::from_logs_minimal(&logs, instructions.clone(), &Default::default()).unwrap();

    // Collect LT lookups first (needed for both LT trace and bitwise lookups)

    println!(
        "Comprehensive: CPU {} rows, Bitwise {} rows",
        traces.cpus[0].main_table.height, traces.bitwise.main_table.height,
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
    let mut traces =
        Traces::from_logs_minimal(&logs, instructions.clone(), &Default::default()).unwrap();
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
    let mut traces =
        Traces::from_logs_minimal(&logs, instructions.clone(), &Default::default()).unwrap();
    assert!(
        prove_and_verify_vm_minimal(&elf, &mut traces),
        "test_sub_8 failed"
    );
}

#[test]
fn test_prove_elfs_test_addw_8() {
    let (elf, logs, instructions) = run_asm_elf("test_addw_8");
    let mut traces =
        Traces::from_logs_minimal(&logs, instructions.clone(), &Default::default()).unwrap();
    assert!(
        prove_and_verify_vm_minimal(&elf, &mut traces),
        "test_addw_8 failed"
    );
}

#[test]
fn test_prove_elfs_test_subw_8() {
    let (elf, logs, instructions) = run_asm_elf("test_subw_8");
    let mut traces =
        Traces::from_logs_minimal(&logs, instructions.clone(), &Default::default()).unwrap();
    // Using traces from Traces::from_logs() which includes MEMW LT ops
    assert!(
        prove_and_verify_vm_minimal(&elf, &mut traces),
        "test_subw_8 failed"
    );
}

#[test]
fn test_prove_elfs_test_addw_lui_8() {
    let (elf, logs, instructions) = run_asm_elf("test_addw_lui_8");
    let mut traces =
        Traces::from_logs_minimal(&logs, instructions.clone(), &Default::default()).unwrap();
    // Using traces from Traces::from_logs() which includes MEMW LT ops
    assert!(
        prove_and_verify_vm_minimal(&elf, &mut traces),
        "test_addw_lui_8 failed"
    );
}

#[test]
fn test_prove_elfs_test_subw_lui_8() {
    let (elf, logs, instructions) = run_asm_elf("test_subw_lui_8");
    let mut traces =
        Traces::from_logs_minimal(&logs, instructions.clone(), &Default::default()).unwrap();
    // Using traces from Traces::from_logs() which includes MEMW LT ops
    assert!(
        prove_and_verify_vm_minimal(&elf, &mut traces),
        "test_subw_lui_8 failed"
    );
}

#[test]
fn test_prove_elfs_test_add_neg_8() {
    let (elf, logs, instructions) = run_asm_elf("test_add_neg_8");
    let mut traces =
        Traces::from_logs_minimal(&logs, instructions.clone(), &Default::default()).unwrap();
    // Using traces from Traces::from_logs() which includes MEMW LT ops
    assert!(
        prove_and_verify_vm_minimal(&elf, &mut traces),
        "test_add_neg_8 failed"
    );
}

#[test]
fn test_prove_elfs_test_sub_neg_8() {
    let (elf, logs, instructions) = run_asm_elf("test_sub_neg_8");
    let mut traces =
        Traces::from_logs_minimal(&logs, instructions.clone(), &Default::default()).unwrap();
    // Using traces from Traces::from_logs() which includes MEMW LT ops
    assert!(
        prove_and_verify_vm_minimal(&elf, &mut traces),
        "test_sub_neg_8 failed"
    );
}

#[test]
fn test_prove_elfs_test_mul_8() {
    let (elf, logs, instructions) = run_asm_elf("test_mul_8");
    let mut traces =
        Traces::from_logs_minimal(&logs, instructions.clone(), &Default::default()).unwrap();
    // Using traces from Traces::from_logs() which includes MEMW LT ops
    assert!(
        prove_and_verify_vm_minimal(&elf, &mut traces),
        "test_mul_8 failed"
    );
}

#[test]
fn test_prove_elfs_mulw_neg() {
    let (elf, logs, instructions) = run_asm_elf("mulw_neg");
    let mut traces =
        Traces::from_logs_minimal(&logs, instructions.clone(), &Default::default()).unwrap();
    assert!(
        prove_and_verify_vm_minimal(&elf, &mut traces),
        "mulw_neg failed"
    );
}

#[test]
fn test_prove_elfs_test_div_8() {
    let (elf, logs, instructions) = run_asm_elf("test_div_8");
    let mut traces =
        Traces::from_logs_minimal(&logs, instructions.clone(), &Default::default()).unwrap();
    // Using traces from Traces::from_logs() which includes MEMW LT ops
    assert!(
        prove_and_verify_vm_minimal(&elf, &mut traces),
        "test_div_8 failed"
    );
}

// Test DIVW with negative operands (−100 / 7 = −14).
// Result bit 31 is set, so rvd ≠ res. The DVRM bus must send res (CPU-CA46), not rvd.
#[test]
fn test_prove_elfs_divw() {
    let (elf, logs, instructions) = run_asm_elf("divw");
    let mut traces =
        Traces::from_logs_minimal(&logs, instructions.clone(), &Default::default()).unwrap();
    assert!(
        prove_and_verify_vm_minimal(&elf, &mut traces),
        "divw failed"
    );
}

// Test REMW with negative operands (−100 % 7 = −2).
// Result bit 31 is set, so rvd ≠ res. The DVRM bus must send res (CPU-CA46), not rvd.
#[test]
fn test_prove_elfs_remw() {
    let (elf, logs, instructions) = run_asm_elf("remw");
    let mut traces =
        Traces::from_logs_minimal(&logs, instructions.clone(), &Default::default()).unwrap();
    assert!(
        prove_and_verify_vm_minimal(&elf, &mut traces),
        "remw failed"
    );
}

// Test DIVW/REMW with a negative divisor (arg2 bit 31 set).
// Exercises arg2 sign extension via CPU-CE63 (signed * arg2_sign_bit).
#[test]
fn test_prove_elfs_sign_ext_edge_cases_8() {
    let (elf, logs, instructions) = run_asm_elf("sign_ext_edge_cases_8");
    let mut traces =
        Traces::from_logs_minimal(&logs, instructions.clone(), &Default::default()).unwrap();
    assert!(
        prove_and_verify_vm_minimal(&elf, &mut traces),
        "sign_ext_edge_cases_8 failed"
    );
}

// Misaligned load/store regression tests. Each program issues one load or
// store whose effective address is not naturally aligned to the access width,
// crossing one or more 4-byte cell boundaries in the executor's memory map.
#[test]
fn test_prove_elfs_misalign_lh() {
    let (elf, logs, _instructions) = run_asm_elf("misalign_lh");
    let mut traces =
        Traces::from_elf_and_logs_minimal(&elf, &logs, &Default::default(), &[]).unwrap();
    assert!(
        prove_and_verify_vm_minimal(&elf, &mut traces),
        "misalign_lh failed"
    );
}

#[test]
fn test_prove_elfs_misalign_lhu() {
    let (elf, logs, _instructions) = run_asm_elf("misalign_lhu");
    let mut traces =
        Traces::from_elf_and_logs_minimal(&elf, &logs, &Default::default(), &[]).unwrap();
    assert!(
        prove_and_verify_vm_minimal(&elf, &mut traces),
        "misalign_lhu failed"
    );
}

#[test]
fn test_prove_elfs_misalign_lw() {
    let (elf, logs, _instructions) = run_asm_elf("misalign_lw");
    let mut traces =
        Traces::from_elf_and_logs_minimal(&elf, &logs, &Default::default(), &[]).unwrap();
    assert!(
        prove_and_verify_vm_minimal(&elf, &mut traces),
        "misalign_lw failed"
    );
}

#[test]
fn test_prove_elfs_misalign_lwu() {
    let (elf, logs, _instructions) = run_asm_elf("misalign_lwu");
    let mut traces =
        Traces::from_elf_and_logs_minimal(&elf, &logs, &Default::default(), &[]).unwrap();
    assert!(
        prove_and_verify_vm_minimal(&elf, &mut traces),
        "misalign_lwu failed"
    );
}

#[test]
fn test_prove_elfs_misalign_ld() {
    let (elf, logs, _instructions) = run_asm_elf("misalign_ld");
    let mut traces =
        Traces::from_elf_and_logs_minimal(&elf, &logs, &Default::default(), &[]).unwrap();
    assert!(
        prove_and_verify_vm_minimal(&elf, &mut traces),
        "misalign_ld failed"
    );
}

#[test]
fn test_prove_elfs_misalign_sh() {
    let (elf, logs, _instructions) = run_asm_elf("misalign_sh");
    let mut traces =
        Traces::from_elf_and_logs_minimal(&elf, &logs, &Default::default(), &[]).unwrap();
    assert!(
        prove_and_verify_vm_minimal(&elf, &mut traces),
        "misalign_sh failed"
    );
}

#[test]
fn test_prove_elfs_misalign_sw() {
    let (elf, logs, _instructions) = run_asm_elf("misalign_sw");
    let mut traces =
        Traces::from_elf_and_logs_minimal(&elf, &logs, &Default::default(), &[]).unwrap();
    assert!(
        prove_and_verify_vm_minimal(&elf, &mut traces),
        "misalign_sw failed"
    );
}

#[test]
fn test_prove_elfs_misalign_sd() {
    let (elf, logs, _instructions) = run_asm_elf("misalign_sd");
    let mut traces =
        Traces::from_elf_and_logs_minimal(&elf, &logs, &Default::default(), &[]).unwrap();
    assert!(
        prove_and_verify_vm_minimal(&elf, &mut traces),
        "misalign_sd failed"
    );
}

// MULW where the 32-bit product overflows past bit 31.
#[test]
fn test_prove_elfs_mulw_overflow() {
    let (elf, logs, instructions) = run_asm_elf("mulw_overflow");
    let mut traces =
        Traces::from_logs_minimal(&logs, instructions.clone(), &Default::default()).unwrap();
    assert!(
        prove_and_verify_vm_minimal(&elf, &mut traces),
        "mulw_overflow failed"
    );
}

// DIVUW where the 32-bit unsigned quotient has bit 31 set.
#[test]
fn test_prove_elfs_divuw_high_bit() {
    let (elf, logs, instructions) = run_asm_elf("divuw_high_bit");
    let mut traces =
        Traces::from_logs_minimal(&logs, instructions.clone(), &Default::default()).unwrap();
    assert!(
        prove_and_verify_vm_minimal(&elf, &mut traces),
        "divuw_high_bit failed"
    );
}

// REMUW where the 32-bit unsigned remainder has bit 31 set.
#[test]
fn test_prove_elfs_remuw_high_bit() {
    let (elf, logs, instructions) = run_asm_elf("remuw_high_bit");
    let mut traces =
        Traces::from_logs_minimal(&logs, instructions.clone(), &Default::default()).unwrap();
    assert!(
        prove_and_verify_vm_minimal(&elf, &mut traces),
        "remuw_high_bit failed"
    );
}

// MULW base case (no 32-bit overflow).
#[test]
fn test_prove_elfs_mulw() {
    let (elf, logs, instructions) = run_asm_elf("mulw");
    let mut traces =
        Traces::from_logs_minimal(&logs, instructions.clone(), &Default::default()).unwrap();
    assert!(
        prove_and_verify_vm_minimal(&elf, &mut traces),
        "mulw failed"
    );
}

// DIVW signed-overflow edge case: i32::MIN / -1 returns i32::MIN per RISC-V spec.
#[test]
fn test_prove_elfs_divw_overflow() {
    let (elf, logs, instructions) = run_asm_elf("divw_overflow");
    let mut traces =
        Traces::from_logs_minimal(&logs, instructions.clone(), &Default::default()).unwrap();
    assert!(
        prove_and_verify_vm_minimal(&elf, &mut traces),
        "divw_overflow failed"
    );
}

// DIVW divide-by-zero: quotient = -1 (all ones sign-extended).
#[test]
fn test_prove_elfs_divw_zero() {
    let (elf, logs, instructions) = run_asm_elf("divw_zero");
    let mut traces =
        Traces::from_logs_minimal(&logs, instructions.clone(), &Default::default()).unwrap();
    assert!(
        prove_and_verify_vm_minimal(&elf, &mut traces),
        "divw_zero failed"
    );
}

// REMW signed-overflow edge case: i32::MIN % -1 returns 0 per RISC-V spec.
#[test]
fn test_prove_elfs_remw_overflow() {
    let (elf, logs, instructions) = run_asm_elf("remw_overflow");
    let mut traces =
        Traces::from_logs_minimal(&logs, instructions.clone(), &Default::default()).unwrap();
    assert!(
        prove_and_verify_vm_minimal(&elf, &mut traces),
        "remw_overflow failed"
    );
}

// REMW divide-by-zero: remainder = dividend.
#[test]
fn test_prove_elfs_remw_zero() {
    let (elf, logs, instructions) = run_asm_elf("remw_zero");
    let mut traces =
        Traces::from_logs_minimal(&logs, instructions.clone(), &Default::default()).unwrap();
    assert!(
        prove_and_verify_vm_minimal(&elf, &mut traces),
        "remw_zero failed"
    );
}

// DIVUW base case (no high-bit set in quotient).
#[test]
fn test_prove_elfs_divuw() {
    let (elf, logs, instructions) = run_asm_elf("divuw");
    let mut traces =
        Traces::from_logs_minimal(&logs, instructions.clone(), &Default::default()).unwrap();
    assert!(
        prove_and_verify_vm_minimal(&elf, &mut traces),
        "divuw failed"
    );
}

// REMUW base case (no high-bit set in remainder).
#[test]
fn test_prove_elfs_remuw() {
    let (elf, logs, instructions) = run_asm_elf("remuw");
    let mut traces =
        Traces::from_logs_minimal(&logs, instructions.clone(), &Default::default()).unwrap();
    assert!(
        prove_and_verify_vm_minimal(&elf, &mut traces),
        "remuw failed"
    );
}

#[test]
fn test_prove_elfs_test_shift_8() {
    let (elf, logs, instructions) = run_asm_elf("test_shift_8");
    let mut traces =
        Traces::from_logs_minimal(&logs, instructions.clone(), &Default::default()).unwrap();
    // Using traces from Traces::from_logs() which includes MEMW LT ops
    assert!(
        prove_and_verify_vm_minimal(&elf, &mut traces),
        "test_shift_8 failed"
    );
}

// Tests that right shift by 0 bits (srli a0, a2, 0) is provable.
// Regression test for SHIFT-C4: previously the shift mask lookup could send 256
// as a byte input when shift=0, making the proof fail.
#[test]
fn test_prove_elfs_srli_one_zero() {
    let (elf, logs, instructions) = run_asm_elf("srli_one_zero");
    let mut traces =
        Traces::from_logs_minimal(&logs, instructions.clone(), &Default::default()).unwrap();
    assert!(
        prove_and_verify_vm_minimal(&elf, &mut traces),
        "srli_one_zero failed"
    );
}

#[test]
fn test_prove_elfs_sllw() {
    let (elf, logs, instructions) = run_asm_elf("sllw");
    let mut traces =
        Traces::from_logs_minimal(&logs, instructions.clone(), &Default::default()).unwrap();
    assert!(
        prove_and_verify_vm_minimal(&elf, &mut traces),
        "sllw failed"
    );
}

/// Proves and verifies a program containing a FENCE instruction.
/// FENCE is mapped to a no-op ADDI x0, x0, 0.
#[test]
fn test_prove_elfs_fence() {
    let (elf, logs, instructions) = run_asm_elf("fence");
    let mut traces =
        Traces::from_logs_minimal(&logs, instructions.clone(), &Default::default()).unwrap();
    assert!(
        prove_and_verify_vm_minimal(&elf, &mut traces),
        "fence failed"
    );
}

#[test]
fn test_prove_elfs_test_bitwise_8() {
    let (elf, logs, instructions) = run_asm_elf("test_bitwise_8");
    let mut traces =
        Traces::from_logs_minimal(&logs, instructions.clone(), &Default::default()).unwrap();
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
    let mut traces =
        Traces::from_logs_minimal(&logs, instructions.clone(), &Default::default()).unwrap();

    // Collect LT lookups first (needed for both LT trace and bitwise lookups)

    println!(
        "test_slt_8: CPU {} rows, Bitwise {} rows",
        traces.cpus[0].main_table.height, traces.bitwise.main_table.height,
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
    let mut traces =
        Traces::from_logs_minimal(&logs, instructions.clone(), &Default::default()).unwrap();
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
        Traces::from_elf_and_logs_minimal(&elf, &logs, &Default::default(), &[]).unwrap();
    assert!(
        prove_and_verify_vm_minimal(&elf, &mut traces),
        "test_lb_lh_8 failed"
    );
}

#[test]
fn test_prove_elfs_test_sb_sh_8() {
    let (elf, logs, _instructions) = run_asm_elf("test_sb_sh_8");
    let mut traces =
        Traces::from_elf_and_logs_minimal(&elf, &logs, &Default::default(), &[]).unwrap();
    assert!(
        !traces.memws.is_empty(),
        "test_sb_sh_8 should produce MEMW rows for byte/halfword memory accesses"
    );
    assert!(
        prove_and_verify_vm_minimal(&elf, &mut traces),
        "test_sb_sh_8 failed"
    );
}

/// Exercises the MEMW_A (aligned fast path) table.
/// lw_sw stores a word (4 bytes) at aligned address 20 and loads it back,
/// routing both operations through MEMW_A instead of the unaligned MEMW table.
#[test]
fn test_prove_elfs_lw_sw() {
    let (elf, logs, _instructions) = run_asm_elf("lw_sw");
    let mut traces =
        Traces::from_elf_and_logs_minimal(&elf, &logs, &Default::default(), &[]).unwrap();
    assert!(
        !traces.memw_aligneds.is_empty(),
        "lw_sw should produce MEMW_A rows for aligned word accesses"
    );
    assert!(
        prove_and_verify_vm_minimal(&elf, &mut traces),
        "lw_sw failed"
    );
}

/// Exercises both MEMW and MEMW_A in the same program.
///
/// Two separate `sb` instructions write to adjacent bytes (sp+0 and sp+1) at
/// different timestamps. The subsequent `lh` read spans both bytes and sees
/// mismatched old_timestamps, routing to MEMW. Register ops and the aligned
/// `sw`/`lw` pair route to MEMW_A.
#[test]
fn test_prove_elfs_test_memw_split_ts() {
    let (elf, logs, _instructions) = run_asm_elf("test_memw_split_ts");
    let mut traces =
        Traces::from_elf_and_logs_minimal(&elf, &logs, &Default::default(), &[]).unwrap();
    assert!(
        !traces.memws.is_empty(),
        "test_memw_split_ts should produce MEMW rows (split old_timestamps from sb+sb+lh)"
    );
    assert!(
        !traces.memw_aligneds.is_empty(),
        "test_memw_split_ts should produce MEMW_A rows (register ops and aligned sw/lw)"
    );
    assert!(
        prove_and_verify_vm_minimal(&elf, &mut traces),
        "test_memw_split_ts failed"
    );
}

#[test]
fn test_prove_elfs_all_branches_16() {
    // Initialize logger to see debug constraint validation output
    let _ = env_logger::builder().is_test(true).try_init();

    let (elf, logs, instructions) = run_asm_elf("all_branches_16");
    let mut traces =
        Traces::from_logs_minimal(&logs, instructions.clone(), &Default::default()).unwrap();

    // BLT instructions need LT table (like SLT)

    println!(
        "all_branches_16: CPU {} rows, Bitwise {} rows",
        traces.cpus[0].main_table.height, traces.bitwise.main_table.height,
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
        Traces::from_elf_and_logs_minimal(&elf, &logs, &Default::default(), &[]).unwrap();
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
    let mut traces =
        Traces::from_logs_minimal(&logs, instructions.clone(), &Default::default()).unwrap();

    // Includes SLT/SLTU instructions - need LT table

    println!(
        "all_instructions_64 (fast): CPU {} rows, Bitwise {} rows, MEMW {} tables ({} rows in first), LOAD {} rows",
        traces.cpus[0].main_table.height,
        traces.bitwise.main_table.height,
        traces.memws.len(),
        traces.memws[0].main_table.height,
        traces.loads[0].main_table.height
    );
    assert!(
        prove_and_verify_vm_minimal(&elf, &mut traces),
        "all_instructions_64 failed"
    );
}

#[test]
fn test_prove_elfs_keccak() {
    let _ = env_logger::builder().is_test(true).try_init();

    let (elf, logs, _instructions) = run_asm_elf("test_keccak");
    // Must use from_elf_and_logs (not from_logs_minimal) because keccak accesses
    // RAM (stack memory), which requires PAGE tables for Memory bus balance.
    let mut traces =
        Traces::from_elf_and_logs_minimal(&elf, &logs, &Default::default(), &[]).unwrap();

    assert!(
        prove_and_verify_vm_minimal(&elf, &mut traces),
        "keccak prove/verify failed"
    );
}

#[test]
fn test_prove_elfs_keccak_multi_call() {
    let _ = env_logger::builder().is_test(true).try_init();

    let elf_bytes = crate::test_utils::asm_elf_bytes("test_keccak_multi");
    let elf = Elf::load(&elf_bytes).expect("Failed to load ELF");
    let executor =
        executor::vm::execution::Executor::new(&elf, vec![]).expect("Failed to create executor");
    let result = executor.run().expect("Failed to run program");

    // The guest initializes lane[i] = i + 1 and applies keccak-f[1600] three times.
    // Cross-check the committed output against tiny-keccak's independent
    // implementation of the permutation.
    let mut expected_state: [u64; 25] = core::array::from_fn(|i| (i + 1) as u64);
    for _ in 0..3 {
        tiny_keccak::keccakf(&mut expected_state);
    }
    let mut expected_bytes = Vec::with_capacity(200);
    for lane in expected_state {
        expected_bytes.extend_from_slice(&lane.to_le_bytes());
    }

    assert_eq!(
        result.return_values.memory_values, expected_bytes,
        "committed state must match tiny-keccak after 3 keccak-f[1600] calls"
    );

    let mut traces =
        Traces::from_elf_and_logs_minimal(&elf, &result.logs, &Default::default(), &[]).unwrap();
    assert_eq!(
        traces.public_output_bytes,
        result.return_values.memory_values
    );

    assert!(
        prove_and_verify_vm_minimal(&elf, &mut traces),
        "keccak multi-call prove/verify failed"
    );
}

#[test]
fn test_prove_elfs_ecsm() {
    let _ = env_logger::builder().is_test(true).try_init();

    let elf_bytes = crate::test_utils::asm_elf_bytes("test_ecsm");
    let elf = Elf::load(&elf_bytes).expect("Failed to load ELF");
    let executor =
        executor::vm::execution::Executor::new(&elf, vec![]).expect("Failed to create executor");
    let result = executor.run().expect("Failed to run program");

    // The guest computes 5·G and commits the 32-byte x-coordinate; cross-check it against
    // the reference scalar multiplication. Gx, little-endian:
    let mut gx = [
        0x79u8, 0xBE, 0x66, 0x7E, 0xF9, 0xDC, 0xBB, 0xAC, 0x55, 0xA0, 0x62, 0x95, 0xCE, 0x87, 0x0B,
        0x07, 0x02, 0x9B, 0xFC, 0xDB, 0x2D, 0xCE, 0x28, 0xD9, 0x59, 0xF2, 0x81, 0x5B, 0x16, 0xF8,
        0x17, 0x98,
    ];
    gx.reverse();
    let mut k = [0u8; 32];
    k[0] = 5;
    let expected_xr = ecsm::scalar_mul_x(&k, &gx).unwrap();
    assert_eq!(
        result.return_values.memory_values,
        expected_xr.to_vec(),
        "committed xR must equal x(5G)"
    );

    let mut traces =
        Traces::from_elf_and_logs_minimal(&elf, &result.logs, &Default::default(), &[]).unwrap();
    assert!(
        prove_and_verify_vm_minimal(&elf, &mut traces),
        "ECSM prove/verify failed"
    );
}

#[test]
fn test_prove_elfs_ecsm_multi() {
    let _ = env_logger::builder().is_test(true).try_init();

    let elf_bytes = crate::test_utils::asm_elf_bytes("test_ecsm_multi");
    let elf = Elf::load(&elf_bytes).expect("Failed to load ELF");
    let executor =
        executor::vm::execution::Executor::new(&elf, vec![]).expect("Failed to create executor");
    let result = executor.run().expect("Failed to run program");

    // Gx little-endian.
    let mut gx = [
        0x79u8, 0xBE, 0x66, 0x7E, 0xF9, 0xDC, 0xBB, 0xAC, 0x55, 0xA0, 0x62, 0x95, 0xCE, 0x87, 0x0B,
        0x07, 0x02, 0x9B, 0xFC, 0xDB, 0x2D, 0xCE, 0x28, 0xD9, 0x59, 0xF2, 0x81, 0x5B, 0x16, 0xF8,
        0x17, 0x98,
    ];
    gx.reverse();

    // The guest commits x(1·G) || x(5·G) || x(0xABCDEF·G); cross-check each 32-byte chunk.
    // k=1 exercises the zero-ECDAS-steps edge; 0xABCDEF exercises many doubles + adds.
    let mut expected = Vec::new();
    for kv in [1u64, 5, 0xABCDEF] {
        let mut k = [0u8; 32];
        k[..8].copy_from_slice(&kv.to_le_bytes());
        expected.extend_from_slice(&ecsm::scalar_mul_x(&k, &gx).unwrap());
    }
    assert_eq!(
        result.return_values.memory_values, expected,
        "committed outputs must equal x(1G) || x(5G) || x(0xABCDEF·G)"
    );

    let mut traces =
        Traces::from_elf_and_logs_minimal(&elf, &result.logs, &Default::default(), &[]).unwrap();
    assert!(
        prove_and_verify_vm_minimal(&elf, &mut traces),
        "ECSM multi-call prove/verify failed"
    );
}

/// End-to-end via the **Rust-guest path**: the `syscalls::ecsm_mul` wrapper computes 5·G and
/// commits its x-coordinate. Verifies the wrapper works end-to-end (parity with the asm guest).
#[test]
fn test_prove_ecsm_rust_guest() {
    let _ = env_logger::builder().is_test(true).try_init();

    let workspace_root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root")
        .to_path_buf();
    let elf_bytes = std::fs::read(workspace_root.join("executor/program_artifacts/rust/ecsm.elf"))
        .expect("ecsm.elf not found — run `make compile-programs-rust`");

    let proof = prove_vm_minimal(&elf_bytes, &[], &Default::default());
    assert!(
        verify_vm_minimal(&proof, &elf_bytes),
        "ecsm rust guest should verify"
    );

    // Committed output must equal x(5·G).
    let mut gx = [
        0x79u8, 0xBE, 0x66, 0x7E, 0xF9, 0xDC, 0xBB, 0xAC, 0x55, 0xA0, 0x62, 0x95, 0xCE, 0x87, 0x0B,
        0x07, 0x02, 0x9B, 0xFC, 0xDB, 0x2D, 0xCE, 0x28, 0xD9, 0x59, 0xF2, 0x81, 0x5B, 0x16, 0xF8,
        0x17, 0x98,
    ];
    gx.reverse();
    let mut k = [0u8; 32];
    k[0] = 5;
    assert_eq!(
        proof.public_output,
        ecsm::scalar_mul_x(&k, &gx).unwrap().to_vec()
    );
}

/// Soundness: the verifier REJECTS a forged ECSM result.
///
/// A malicious prover must not be able to claim a wrong `k·G`. We tamper the result
/// x-coordinate `xR` in the ECSM trace (to a different valid byte). `xR` is bound by the
/// final ECDAS-bus tuple (the constrained double-and-add output) and by the `xR < p`
/// carry-chain check, so the forgery unbalances the buses / breaks the constraints and the
/// proof must fail to verify.
#[test]
fn test_prove_elfs_ecsm_forged_result_rejected() {
    use crate::tables::ecsm::cols as ecsm_cols;

    let _ = env_logger::builder().is_test(true).try_init();

    let elf_bytes = crate::test_utils::asm_elf_bytes("test_ecsm");
    let elf = Elf::load(&elf_bytes).expect("Failed to load ELF");
    let executor =
        executor::vm::execution::Executor::new(&elf, vec![]).expect("Failed to create executor");
    let result = executor.run().expect("Failed to run program");
    let mut traces =
        Traces::from_elf_and_logs_minimal(&elf, &result.logs, &Default::default(), &[]).unwrap();

    // Forge the low byte of xR on the (single) real ECSM row.
    let orig = *traces.ecsm.main_table.get(0, ecsm_cols::xr(0));
    let forged = orig + FieldElement::<GoldilocksField>::one();
    traces.ecsm.main_table.set(0, ecsm_cols::xr(0), forged);

    assert!(
        !prove_and_verify_vm_minimal(&elf, &mut traces),
        "Verifier must reject a forged ECSM result xR"
    );
}

/// Regression test: `µ` is the multiplicity of every ECDAS bus interaction, so it must remain
/// boolean. Forge a non-boolean `µ` on a real ECDAS row and assert the verifier rejects.
/// (k=5 produces 3 ECDAS rows.)
#[test]
fn test_prove_elfs_ecsm_forged_ecdas_mu_rejected() {
    use crate::tables::ecdas::cols as ecdas_cols;

    let _ = env_logger::builder().is_test(true).try_init();

    let elf_bytes = crate::test_utils::asm_elf_bytes("test_ecsm");
    let elf = Elf::load(&elf_bytes).expect("Failed to load ELF");
    let executor =
        executor::vm::execution::Executor::new(&elf, vec![]).expect("Failed to create executor");
    let result = executor.run().expect("Failed to run program");
    let mut traces =
        Traces::from_elf_and_logs_minimal(&elf, &result.logs, &Default::default(), &[]).unwrap();

    // Row 0 is a real ECDAS step (µ=1); forge µ to a non-boolean value.
    traces.ecdas.main_table.set(
        0,
        ecdas_cols::MU,
        FieldElement::<GoldilocksField>::from(2u64),
    );

    assert!(
        !prove_and_verify_vm_minimal(&elf, &mut traces),
        "Verifier must reject a non-boolean ECDAS multiplicity"
    );
}

/// Verifier REJECTS a forged trace where an addr byte cell is set to a
/// non-byte field element.
///
/// Without the ARE_BYTES range checks on addr(0..7), an attacker could keep
/// `addr_lo = b0 + 256·b1 + 65536·b2 + 2^24·b3` equal to an unaligned target
/// address as a field element while setting addr(0)=0 (passing the BYTE_ALU
/// alignment check) and folding the carry into addr(1) as a non-byte
/// FE-element. This test asserts that mutating addr(1) to a non-byte value
/// unbalances the verifier's bus checks and the proof is rejected.
#[test]
fn test_prove_elfs_keccak_unaligned_state_addr() {
    use crate::tables::keccak::cols as keccak_cols;

    let _ = env_logger::builder().is_test(true).try_init();

    let elf_bytes = crate::test_utils::asm_elf_bytes("test_keccak_multi");
    let elf = Elf::load(&elf_bytes).expect("Failed to load ELF");
    let executor =
        executor::vm::execution::Executor::new(&elf, vec![]).expect("Failed to create executor");
    let result = executor.run().expect("Failed to run program");
    let mut traces =
        Traces::from_elf_and_logs_minimal(&elf, &result.logs, &Default::default(), &[]).unwrap();

    // Tamper the first real keccak row: replace addr(1) (a byte cell) with a
    // value outside [0, 256). The new ARE_BYTES bus sender will emit this
    // value with multiplicity MU=1; the ARE_BYTES preprocessed table only
    // contains 0..256, so the bus cannot balance.
    traces.keccak.main_table.set(
        0,
        keccak_cols::addr(1),
        FieldElement::<GoldilocksField>::from(257u64),
    );

    assert!(
        !prove_and_verify_vm_minimal(&elf, &mut traces),
        "Verifier must reject a keccak proof whose addr cells are not bytes"
    );
}

#[test]
fn test_prove_elfs_test_commit_4() {
    let elf_bytes = crate::test_utils::asm_elf_bytes("test_commit_4");
    let elf = Elf::load(&elf_bytes).expect("Failed to load ELF");
    let executor =
        executor::vm::execution::Executor::new(&elf, vec![]).expect("Failed to create executor");
    let result = executor.run().expect("Failed to run program");

    // Verify public output matches the committed bytes [0xAA, 0xBB, 0xCC, 0xDD]
    assert_eq!(
        result.return_values.memory_values,
        vec![0xAA, 0xBB, 0xCC, 0xDD],
        "Public output should match committed bytes"
    );

    let mut traces =
        Traces::from_elf_and_logs_minimal(&elf, &result.logs, &Default::default(), &[]).unwrap();
    assert_eq!(
        traces.public_output_bytes,
        result.return_values.memory_values
    );
    assert!(
        prove_and_verify_vm_minimal(&elf, &mut traces),
        "test_commit_4 failed"
    );
}

/// Verifier REJECTS when page configs don't match the proven commit trace.
///
/// The prover generates a valid proof for test_commit_4 (which writes to page 0).
/// The verifier uses only ELF pages (no runtime pages) → page mismatch →
/// verification must fail.
#[test]
fn test_prove_elfs_test_commit_4_wrong_pages_rejected() {
    let elf_bytes = crate::test_utils::asm_elf_bytes("test_commit_4");
    let elf = Elf::load(&elf_bytes).expect("Failed to load ELF");

    let proof_options = ProofOptions::default_test_options();
    let executor =
        executor::vm::execution::Executor::new(&elf, vec![]).expect("Failed to create executor");
    let result = executor.run().expect("Failed to run program");
    let mut traces =
        Traces::from_elf_and_logs_minimal(&elf, &result.logs, &Default::default(), &[]).unwrap();

    // Prover uses correct page configs
    let table_counts = traces.table_counts();
    let prover_airs = crate::VmAirs::new(
        &elf,
        &proof_options,
        true,
        &traces.page_configs,
        &table_counts,
        None,
        true,
        None,
        None,
        None,
    );
    let proof = multi_prove_ram(
        prover_airs.air_trace_pairs(&mut traces),
        &mut DefaultTranscript::<E>::new(&[]),
    )
    .expect("Prover failed");

    // Verifier uses EMPTY runtime pages → missing stack/public-output pages
    let wrong_configs = Traces::page_configs_from_elf_and_runtime(&elf, &[], 0);
    let verifier_airs = crate::VmAirs::new(
        &elf,
        &proof_options,
        true,
        &wrong_configs,
        &table_counts,
        None,
        true,
        None,
        None,
        None,
    );
    let verifier_air_refs = verifier_airs.air_refs();
    let views: Vec<StarkProofView<F, E, ()>> =
        proof.proofs.iter().map(StarkProofView::Owned).collect();
    let mut replay_transcript = DefaultTranscript::<E>::new(&[]);
    let expected_bus_balance = crate::compute_expected_commit_bus_balance_view(
        &verifier_air_refs,
        &views,
        &traces.public_output_bytes,
        0,
        &mut replay_transcript,
    )
    .expect("fingerprint collision in test");

    let verified = Verifier::multi_verify_views(
        &verifier_air_refs,
        &views,
        &mut DefaultTranscript::<E>::new(&[]),
        &expected_bus_balance,
    );
    assert!(
        !verified,
        "Verifier should REJECT when runtime pages are missing (commit public output page)"
    );
}

#[test]
fn test_verify_rejects_tampered_public_output() {
    let elf_bytes = crate::test_utils::asm_elf_bytes("test_commit_4");
    let proof_options = ProofOptions::default_test_options();
    let vm_proof = crate::prove_with_options(&elf_bytes, &proof_options, &Default::default())
        .expect("Prover should succeed for test_commit_4");
    assert!(
        crate::verify_with_options(&vm_proof, &elf_bytes, &proof_options, None, None)
            .expect("Valid commit proof should verify"),
        "Baseline proof should verify before tampering"
    );
    let mut tampered_output = vm_proof.public_output.clone();
    tampered_output[0] ^= 0x01;

    let tampered_proof = crate::VmProof {
        public_output: tampered_output,
        ..vm_proof
    };

    let verified =
        crate::verify_with_options(&tampered_proof, &elf_bytes, &proof_options, None, None)
            .expect("Verifier should not error on tampered public output");
    assert!(
        !verified,
        "Verifier should reject proof when VmProof.public_output is tampered"
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
#[ignore] // Slow: run with `cargo test -- --ignored` or `make test-prover-all`
fn test_prove_elfs_all_instructions_64_full() {
    let _ = env_logger::builder().is_test(true).try_init();

    let elf_bytes = crate::test_utils::asm_elf_bytes("all_instructions_64");
    let result = crate::prove_and_verify(&elf_bytes).expect("prove_and_verify failed");
    assert!(
        result,
        "all_instructions_64_full failed - comprehensive test with full bitwise table"
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
    let traces =
        Traces::from_logs_minimal(&logs, instructions.clone(), &Default::default()).unwrap();

    let memw = &traces.memws[0]; // Small test: single MEMW chunk
    println!("DEBUG TABLE SIZES:");
    println!(
        "  MEMW: {} rows ({} tables)",
        memw.num_rows(),
        traces.memws.len()
    );
    println!("  REGISTER: {} rows", traces.register.num_rows());

    // Collect all Memory bus tokens
    // Token = (is_reg, addr_lo, addr_hi, ts_lo, ts_hi, value)
    type Token = (u64, u64, u64, u64, u64, u64);

    // Track sends (+1) and receives (-1) with their sources
    let mut token_balance: HashMap<Token, (i64, Vec<String>)> = HashMap::new();

    // === MEMW tokens (for register rows only) ===
    println!("\n=== MEMW Memory Bus Tokens (register rows) ===");
    for row in 0..memw.num_rows() {
        let is_reg = memw.main_table.get(row, memw_cols::IS_REGISTER).to_raw();
        if is_reg == 0 {
            continue; // Skip memory rows (multiplicity = 0)
        }

        let base_lo = memw.main_table.get(row, memw_cols::BASE_ADDRESS_0).to_raw();
        let base_hi = memw.main_table.get(row, memw_cols::BASE_ADDRESS_1).to_raw();
        let ts_lo = memw.main_table.get(row, memw_cols::TIMESTAMP_0).to_raw();
        let ts_hi = memw.main_table.get(row, memw_cols::TIMESTAMP_1).to_raw();
        let old_ts0_lo = memw
            .main_table
            .get(row, memw_cols::old_timestamp(0)[0])
            .to_raw();
        let old_ts0_hi = memw
            .main_table
            .get(row, memw_cols::old_timestamp(0)[1])
            .to_raw();
        let old_ts1_lo = memw
            .main_table
            .get(row, memw_cols::old_timestamp(1)[0])
            .to_raw();
        let old_ts1_hi = memw
            .main_table
            .get(row, memw_cols::old_timestamp(1)[1])
            .to_raw();
        let val0 = memw.main_table.get(row, memw_cols::VALUE[0]).to_raw();
        let val1 = memw.main_table.get(row, memw_cols::VALUE[1]).to_raw();
        let old0 = memw.main_table.get(row, memw_cols::OLD[0]).to_raw();
        let old1 = memw.main_table.get(row, memw_cols::OLD[1]).to_raw();

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
    for row in 0..memw.num_rows() {
        let is_reg = memw.main_table.get(row, memw_cols::IS_REGISTER).to_raw();
        if is_reg == 0 {
            continue;
        }

        let base_lo = memw.main_table.get(row, memw_cols::BASE_ADDRESS_0).to_raw();
        let base_hi = memw.main_table.get(row, memw_cols::BASE_ADDRESS_1).to_raw();
        let ts_lo = memw.main_table.get(row, memw_cols::TIMESTAMP_0).to_raw();
        let ts_hi = memw.main_table.get(row, memw_cols::TIMESTAMP_1).to_raw();
        let old_ts0_lo = memw
            .main_table
            .get(row, memw_cols::old_timestamp(0)[0])
            .to_raw();
        let old_ts0_hi = memw
            .main_table
            .get(row, memw_cols::old_timestamp(0)[1])
            .to_raw();
        let old_ts1_lo = memw
            .main_table
            .get(row, memw_cols::old_timestamp(1)[0])
            .to_raw();
        let old_ts1_hi = memw
            .main_table
            .get(row, memw_cols::old_timestamp(1)[1])
            .to_raw();
        let val0 = memw.main_table.get(row, memw_cols::VALUE[0]).to_raw();
        let val1 = memw.main_table.get(row, memw_cols::VALUE[1]).to_raw();
        let old0 = memw.main_table.get(row, memw_cols::OLD[0]).to_raw();
        let old1 = memw.main_table.get(row, memw_cols::OLD[1]).to_raw();

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
    let traces = Traces::from_elf_and_logs(
        &elf,
        &logs,
        &Default::default(),
        &[],
        #[cfg(feature = "disk-spill")]
        stark::storage_mode::StorageMode::Ram,
    )
    .unwrap();

    let memw = &traces.memws[0]; // Small test: single MEMW chunk
    println!("DEBUG: test_sb_sh_8 Memory bus tokens (FULL)");
    println!(
        "  MEMW rows: {} ({} tables)",
        memw.num_rows(),
        traces.memws.len()
    );
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
    for row in 0..memw.num_rows() {
        let is_reg = memw.main_table.get(row, memw_cols::IS_REGISTER).to_raw();

        // Count row types
        if is_reg == 1 {
            memw_register_rows += 1;
        } else {
            memw_memory_rows += 1;
        }

        let mu_read = memw.main_table.get(row, memw_cols::MU_READ).to_raw();
        let mu_write = memw.main_table.get(row, memw_cols::MU_WRITE).to_raw();
        let mu_sum = mu_read + mu_write;
        if mu_sum == 0 {
            continue; // Padding row
        }

        let base_lo = memw.main_table.get(row, memw_cols::BASE_ADDRESS_0).to_raw();
        let base_hi = memw.main_table.get(row, memw_cols::BASE_ADDRESS_1).to_raw();
        let ts_lo = memw.main_table.get(row, memw_cols::TIMESTAMP_0).to_raw();
        let ts_hi = memw.main_table.get(row, memw_cols::TIMESTAMP_1).to_raw();
        let old_ts0_lo = memw
            .main_table
            .get(row, memw_cols::old_timestamp(0)[0])
            .to_raw();
        let old_ts0_hi = memw
            .main_table
            .get(row, memw_cols::old_timestamp(0)[1])
            .to_raw();
        let val0 = memw.main_table.get(row, memw_cols::VALUE[0]).to_raw();
        let old0 = memw.main_table.get(row, memw_cols::OLD[0]).to_raw();

        let write2 = memw.main_table.get(row, memw_cols::WRITE2).to_raw();
        let write4 = memw.main_table.get(row, memw_cols::WRITE4).to_raw();
        let write8 = memw.main_table.get(row, memw_cols::WRITE8).to_raw();

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
            let old_ts1_lo = memw
                .main_table
                .get(row, memw_cols::old_timestamp(1)[0])
                .to_raw();
            let old_ts1_hi = memw
                .main_table
                .get(row, memw_cols::old_timestamp(1)[1])
                .to_raw();
            let val1 = memw.main_table.get(row, memw_cols::VALUE[1]).to_raw();
            let old1 = memw.main_table.get(row, memw_cols::OLD[1]).to_raw();

            // address_add(0) = base + 1, now virtual (computed from base + carry)
            let carry0 = memw.main_table.get(row, memw_cols::CARRY[0]).to_raw();
            let addr1_lo = base_lo + 1 - carry0 * (1u64 << 32);
            let addr1_hi = base_hi + carry0;

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
        let page_size = crate::tables::page::DEFAULT_PAGE_SIZE;
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

    // === Count ARE_BYTES lookups from PAGE (batched [init, fini] per row) ===
    println!("\n=== ARE_BYTES Lookup Counts (from PAGE tables) ===");
    let mut page_pair_counts: HashMap<(u8, u8), u64> = HashMap::new();
    let total_page_rows: usize = traces.pages.iter().map(|p| p.num_rows()).sum();
    for page_trace in traces.pages.iter() {
        let page_size = crate::tables::page::DEFAULT_PAGE_SIZE;
        for row in 0..page_trace.num_rows().min(page_size) {
            let init = page_trace.main_table.get(row, page_cols::INIT).to_raw() as u8;
            let fini = page_trace.main_table.get(row, page_cols::FINI).to_raw() as u8;
            *page_pair_counts.entry((init, fini)).or_insert(0) += 1;
        }
    }
    let page_are_bytes_total: u64 = page_pair_counts.values().sum();
    println!(
        "Total PAGE rows: {}, Expected ARE_BYTES (1 per row): {}",
        total_page_rows, total_page_rows,
    );
    println!(
        "ARE_BYTES[0, 0] from PAGE: {} lookups (most rows are (0,0))",
        page_pair_counts.get(&(0, 0)).copied().unwrap_or(0)
    );

    // BITWISE row for ARE_BYTES[X, Y] at Z=0 is X + 256*Y. We only sum
    // multiplicity at the (X, Y) pairs PAGE actually touches. Other senders
    // (e.g. CPU's paired ARE_BYTES checks) also bump this same MU_ARE_BYTES
    // column and may hit the same (X, Y) rows, so this is a coarse sanity
    // check (BITWISE mult >= PAGE's contribution), not an exact balance.
    use crate::tables::bitwise::cols as bitwise_cols;
    let bitwise_are_bytes_mult_over_page_pairs: u64 = page_pair_counts
        .keys()
        .map(|&(x, y)| {
            let row = x as usize + 256 * y as usize;
            traces
                .bitwise
                .main_table
                .get(row, bitwise_cols::MU_ARE_BYTES)
                .to_raw()
        })
        .sum();
    println!(
        "Bitwise ARE_BYTES mult summed over PAGE (init, fini) rows: {}",
        bitwise_are_bytes_mult_over_page_pairs
    );
    println!(
        "Total ARE_BYTES lookups from PAGE (counted): {}",
        page_are_bytes_total
    );
    // Note: this can be >= 0 because CPU byte-pair ARE_BYTES senders may also
    // hit some of the same (init, fini) rows. It should never be negative.
    println!(
        "Difference: {} (>= 0 expected; PAGE pairs may also receive from CPU)",
        bitwise_are_bytes_mult_over_page_pairs as i64 - page_are_bytes_total as i64
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

// =============================================================================
// Deep stack tests (page coverage)
// =============================================================================

/// deep_stack allocates 8192 bytes, writing at SP = 0x...DFF0 (page D000).
/// MemoryState-based page detection auto-discovers all accessed pages.
#[test]
fn test_deep_stack_passes() {
    let (elf, logs, _instructions) = run_asm_elf("deep_stack");
    let mut traces =
        Traces::from_elf_and_logs_minimal(&elf, &logs, &Default::default(), &[]).unwrap();

    assert!(
        prove_and_verify_vm_minimal(&elf, &mut traces),
        "deep_stack should PASS (MemoryState auto-discovers all accessed pages)"
    );
}

/// Tests the full prove → VmProof → verify roundtrip for deep_stack.
///
/// Verifies that `runtime_page_ranges` is correctly extracted by the prover
/// and used by the verifier to reconstruct page configs for non-ELF pages.
#[test]
fn test_deep_stack_runtime_pages_roundtrip() {
    let elf_bytes = crate::test_utils::asm_elf_bytes("deep_stack");
    let elf = Elf::load(&elf_bytes).expect("Failed to load ELF");

    let proof_options = ProofOptions::default_test_options();
    let executor =
        executor::vm::execution::Executor::new(&elf, vec![]).expect("Failed to create executor");
    let result = executor.run().expect("Failed to run program");
    let mut traces =
        Traces::from_elf_and_logs_minimal(&elf, &result.logs, &Default::default(), &[]).unwrap();

    let runtime_page_ranges = traces.runtime_page_ranges();
    let table_counts = traces.table_counts();
    assert!(
        !runtime_page_ranges.is_empty(),
        "deep_stack should have runtime page ranges beyond ELF (stack pages)"
    );

    let prover_airs = crate::VmAirs::new(
        &elf,
        &proof_options,
        true,
        &traces.page_configs,
        &table_counts,
        None,
        true,
        None,
        None,
        None,
    );
    let proof = multi_prove_ram(
        prover_airs.air_trace_pairs(&mut traces),
        &mut DefaultTranscript::<E>::new(&[]),
    )
    .expect("Prover failed");
    // Verifier reconstructs from ELF + runtime_page_ranges hint
    let verifier_configs = Traces::page_configs_from_elf_and_runtime(&elf, &runtime_page_ranges, 0);
    let verifier_airs = crate::VmAirs::new(
        &elf,
        &proof_options,
        true,
        &verifier_configs,
        &table_counts,
        None,
        true,
        None,
        None,
        None,
    );
    let verifier_air_refs = verifier_airs.air_refs();
    let views: Vec<StarkProofView<F, E, ()>> =
        proof.proofs.iter().map(StarkProofView::Owned).collect();
    let mut replay_transcript = DefaultTranscript::<E>::new(&[]);
    let expected_bus_balance = crate::compute_expected_commit_bus_balance_view(
        &verifier_air_refs,
        &views,
        &traces.public_output_bytes,
        0,
        &mut replay_transcript,
    )
    .expect("fingerprint collision in test");

    let verified = Verifier::multi_verify_views(
        &verifier_air_refs,
        &views,
        &mut DefaultTranscript::<E>::new(&[]),
        &expected_bus_balance,
    );
    assert!(
        verified,
        "Verifier should accept proof when using runtime_page_ranges hint"
    );
}

/// Tests that the verifier REJECTS when runtime_page_ranges hint is incomplete.
///
/// The prover generates a proof for deep_stack (which needs page D000).
/// The verifier is given an empty hint (no runtime pages) → commitment
/// mismatch → verification must fail.
#[test]
fn test_deep_stack_missing_pages_rejected() {
    let elf_bytes = crate::test_utils::asm_elf_bytes("deep_stack");
    let elf = Elf::load(&elf_bytes).expect("Failed to load ELF");

    let proof_options = ProofOptions::default_test_options();
    let executor =
        executor::vm::execution::Executor::new(&elf, vec![]).expect("Failed to create executor");
    let result = executor.run().expect("Failed to run program");
    let mut traces =
        Traces::from_elf_and_logs_minimal(&elf, &result.logs, &Default::default(), &[]).unwrap();

    // Prover uses correct page configs (auto-detected from MemoryState)
    let table_counts = traces.table_counts();
    let prover_airs = crate::VmAirs::new(
        &elf,
        &proof_options,
        true,
        &traces.page_configs,
        &table_counts,
        None,
        true,
        None,
        None,
        None,
    );
    let proof = multi_prove_ram(
        prover_airs.air_trace_pairs(&mut traces),
        &mut DefaultTranscript::<E>::new(&[]),
    )
    .expect("Prover failed");
    // Verifier uses EMPTY runtime_page_ranges → missing stack/heap pages
    let wrong_configs = Traces::page_configs_from_elf_and_runtime(&elf, &[], 0);
    let verifier_airs = crate::VmAirs::new(
        &elf,
        &proof_options,
        true,
        &wrong_configs,
        &table_counts,
        None,
        true,
        None,
        None,
        None,
    );
    let verifier_air_refs = verifier_airs.air_refs();
    let views: Vec<StarkProofView<F, E, ()>> =
        proof.proofs.iter().map(StarkProofView::Owned).collect();
    let mut replay_transcript = DefaultTranscript::<E>::new(&[]);
    let expected_bus_balance = crate::compute_expected_commit_bus_balance_view(
        &verifier_air_refs,
        &views,
        &traces.public_output_bytes,
        0,
        &mut replay_transcript,
    )
    .expect("fingerprint collision in test");

    let verified = Verifier::multi_verify_views(
        &verifier_air_refs,
        &views,
        &mut DefaultTranscript::<E>::new(&[]),
        &expected_bus_balance,
    );
    assert!(
        !verified,
        "Verifier should REJECT when runtime_page_ranges is incomplete (missing pages)"
    );
}

// =============================================================================
// Heap allocation tests (runtime page detection)
// =============================================================================

/// heap_alloc writes to addresses 0x80000..0x83000 far from ELF segments and
/// stack, plus a stack write. Tests that MemoryState-based page detection
/// discovers all heap and stack pages, and run-length encodes them.
/// With 256KB pages, all 4 writes (0x80000..0x83000) fit in a single page.
#[test]
fn test_heap_alloc_passes() {
    let (elf, logs, _instructions) = run_asm_elf("heap_alloc");
    let mut traces =
        Traces::from_elf_and_logs_minimal(&elf, &logs, &Default::default(), &[]).unwrap();

    // Verify runtime_page_ranges includes the heap page
    let ranges = traces.runtime_page_ranges();
    // With 256KB pages, all 4 writes land on one page containing 0x80000
    assert!(
        ranges.iter().any(|r| r.base == 0x80000 && r.count == 1),
        "Expected heap range (0x80000, 1), got {:?}",
        ranges
    );

    assert!(
        prove_and_verify_vm_minimal(&elf, &mut traces),
        "heap_alloc should PASS (MemoryState detects heap + stack pages)"
    );
}

/// Full prove → VmProof → verify roundtrip for heap_alloc.
/// Verifies the hint correctly conveys heap page ranges to the verifier.
#[test]
fn test_heap_alloc_runtime_pages_roundtrip() {
    let elf_bytes = crate::test_utils::asm_elf_bytes("heap_alloc");
    let elf = Elf::load(&elf_bytes).expect("Failed to load ELF");

    let proof_options = ProofOptions::default_test_options();
    let executor =
        executor::vm::execution::Executor::new(&elf, vec![]).expect("Failed to create executor");
    let result = executor.run().expect("Failed to run program");
    let mut traces =
        Traces::from_elf_and_logs_minimal(&elf, &result.logs, &Default::default(), &[]).unwrap();

    let runtime_page_ranges = traces.runtime_page_ranges();
    let table_counts = traces.table_counts();

    // With 256KB pages, all heap writes fit in 1 page + 1 stack page
    let total_pages: u64 = runtime_page_ranges.iter().map(|r| r.count).sum();
    assert!(
        total_pages >= 2,
        "Expected at least 2 runtime pages (1 heap + 1 stack), got {}",
        total_pages
    );

    let prover_airs = crate::VmAirs::new(
        &elf,
        &proof_options,
        true,
        &traces.page_configs,
        &table_counts,
        None,
        true,
        None,
        None,
        None,
    );
    let proof = multi_prove_ram(
        prover_airs.air_trace_pairs(&mut traces),
        &mut DefaultTranscript::<E>::new(&[]),
    )
    .expect("Prover failed");
    // Verifier reconstructs from ELF + runtime hint (ranges decoded to pages)
    let verifier_configs = Traces::page_configs_from_elf_and_runtime(&elf, &runtime_page_ranges, 0);
    let verifier_airs = crate::VmAirs::new(
        &elf,
        &proof_options,
        true,
        &verifier_configs,
        &table_counts,
        None,
        true,
        None,
        None,
        None,
    );
    let verifier_air_refs = verifier_airs.air_refs();
    let views: Vec<StarkProofView<F, E, ()>> =
        proof.proofs.iter().map(StarkProofView::Owned).collect();
    let mut replay_transcript = DefaultTranscript::<E>::new(&[]);
    let expected_bus_balance = crate::compute_expected_commit_bus_balance_view(
        &verifier_air_refs,
        &views,
        &traces.public_output_bytes,
        0,
        &mut replay_transcript,
    )
    .expect("fingerprint collision in test");

    let verified = Verifier::multi_verify_views(
        &verifier_air_refs,
        &views,
        &mut DefaultTranscript::<E>::new(&[]),
        &expected_bus_balance,
    );
    assert!(
        verified,
        "Verifier should accept heap_alloc proof with correct runtime_page_ranges"
    );
}

/// Verify that register ops route to MEMW_R and a full prove/verify roundtrip
/// succeeds. Uses `test_add_8` which exercises register reads and writes.
#[test]
fn test_prove_verify_with_memw_register() {
    let (elf, logs, instructions) = run_asm_elf("test_add_8");
    let mut traces =
        Traces::from_logs_minimal(&logs, instructions.clone(), &Default::default()).unwrap();

    // Register ops must go to MEMW_R, not to MEMW_A.
    assert!(
        !traces.memw_registers.is_empty(),
        "register ops should route to MEMW_R: memw_registers must be non-empty"
    );

    // MEMW_A should still have non-register aligned ops (e.g. stack stores).
    assert!(
        !traces.memw_aligneds.is_empty(),
        "MEMW_A should still have aligned non-register ops"
    );

    // Full prove + verify roundtrip.
    assert!(
        prove_and_verify_vm_minimal(&elf, &mut traces),
        "prove/verify should succeed when MEMW_R handles register ops"
    );
}

/// Verify rejects table_counts with all zeros.
#[test]
fn test_verify_rejects_zero_table_counts() {
    let elf_bytes = crate::test_utils::asm_elf_bytes("sub");
    let proof_options = ProofOptions::default_test_options();

    let vm_proof = crate::prove_with_options(&elf_bytes, &proof_options, &Default::default())
        .expect("Prover should succeed on valid program");

    assert!(
        crate::verify_with_options(&vm_proof, &elf_bytes, &proof_options, None, None)
            .expect("Verification should not error on valid proof"),
        "Valid proof should verify"
    );

    let tampered_proof = crate::VmProof {
        table_counts: crate::TableCounts {
            cpu: 0,
            lt: 0,
            memw: 0,
            memw_aligned: 0,
            load: 0,
            mul: 0,
            dvrm: 0,
            shift: 0,
            branch: 0,
            memw_register: 0,
            eq: 0,
            bytewise: 0,
            store: 0,
            cpu32: 0,
        },
        ..vm_proof
    };

    let result =
        crate::verify_with_options(&tampered_proof, &elf_bytes, &proof_options, None, None);
    assert!(result.is_err(), "Got {:?}", result);
}

/// Verify rejects table_counts with cpu=0.
#[test]
fn test_verify_rejects_zero_cpu_count() {
    let elf_bytes = crate::test_utils::asm_elf_bytes("sub");
    let proof_options = ProofOptions::default_test_options();

    let vm_proof = crate::prove_with_options(&elf_bytes, &proof_options, &Default::default())
        .expect("Prover should succeed on valid program");

    let tampered_proof = crate::VmProof {
        table_counts: crate::TableCounts {
            cpu: 0,
            ..vm_proof.table_counts.clone()
        },
        ..vm_proof
    };

    let result =
        crate::verify_with_options(&tampered_proof, &elf_bytes, &proof_options, None, None);
    assert!(result.is_err(), "Got {:?}", result);
}

/// Verify rejects table_counts with memw=0.
#[test]
fn test_verify_rejects_zero_memw_count() {
    let elf_bytes = crate::test_utils::asm_elf_bytes("sub");
    let proof_options = ProofOptions::default_test_options();

    let vm_proof = crate::prove_with_options(&elf_bytes, &proof_options, &Default::default())
        .expect("Prover should succeed on valid program");

    let tampered_proof = crate::VmProof {
        table_counts: crate::TableCounts {
            memw: 0,
            ..vm_proof.table_counts.clone()
        },
        ..vm_proof
    };

    let result =
        crate::verify_with_options(&tampered_proof, &elf_bytes, &proof_options, None, None);
    assert!(result.is_err(), "Got {:?}", result);
}

/// Verify rejects a proof with fewer proofs than AIRs.
#[test]
fn test_crafted_zero_count_proof_must_not_verify() {
    let elf_bytes = crate::test_utils::asm_elf_bytes("sub");
    let elf = Elf::load(&elf_bytes).expect("Failed to load ELF");
    let proof_options = ProofOptions::default_test_options();

    let zero_counts = crate::TableCounts {
        cpu: 0,
        lt: 0,
        memw: 0,
        memw_aligned: 0,
        load: 0,
        mul: 0,
        dvrm: 0,
        shift: 0,
        branch: 0,
        memw_register: 0,
        eq: 0,
        bytewise: 0,
        store: 0,
        cpu32: 0,
    };
    let airs = VmAirs::new(
        &elf,
        &proof_options,
        true,
        &[],
        &zero_counts,
        None,
        true,
        None,
        None,
        None,
    );

    let verifier_air_refs = airs.air_refs();
    assert_eq!(verifier_air_refs.len(), crate::FIXED_TABLE_COUNT);

    let mut bitwise_trace = crate::tables::bitwise::generate_bitwise_trace();

    let instructions = crate::tables::decode::instructions_from_elf(&elf)
        .expect("Failed to parse instructions from ELF");
    let (mut decode_trace, _) = crate::tables::decode::generate_decode_trace(&instructions);

    let pairs: Vec<(
        &dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>,
        _,
        _,
    )> = vec![
        (airs.bitwise.as_ref(), &mut bitwise_trace, &()),
        (airs.decode.as_ref(), &mut decode_trace, &()),
    ];

    let proof = multi_prove_ram(pairs, &mut DefaultTranscript::<E>::new(&[]))
        .expect("Proof generation should succeed");

    assert_eq!(proof.proofs.len(), 2);

    let verified = Verifier::multi_verify(
        &verifier_air_refs,
        &proof,
        &mut DefaultTranscript::<E>::new(&[]),
        &FieldElement::zero(),
    );

    assert!(!verified);
}

/// Prove and verify with small max_rows to exercise table splitting.
#[test]
fn test_small_max_rows_splits_tables() {
    let elf_bytes = crate::test_utils::asm_elf_bytes("all_instructions_64");
    let max_rows = crate::tables::MaxRowsConfig::small();

    let vm_proof = prove_vm_minimal(&elf_bytes, &[], &max_rows);

    // With 2^5 max rows and 64+ instructions, tables should have multiple chunks.
    assert!(
        vm_proof.table_counts.cpu > 1,
        "CPU should have multiple chunks, got {}",
        vm_proof.table_counts.cpu
    );

    assert!(
        verify_vm_minimal(&vm_proof, &elf_bytes),
        "Proof with small max_rows should verify"
    );
}

// =============================================================================
// Soundness tests: tampered traces must be rejected
// =============================================================================

/// Tamper with NEXT_PC on a CPU row and verify the proof is rejected.
///
/// The PC linkage works via MEMW bus (CM54): each CPU row sends a read-write
/// for register x255 with old=pc, value=next_pc. If next_pc is wrong in the
/// CPU trace, the CM54 bus value won't match the MEMW table → bus imbalance
/// → verification failure.
#[test]
fn test_tampered_next_pc_rejected() {
    use crate::tables::cpu::cols;
    use math::field::element::FieldElement;

    let (elf, logs, instructions) = run_asm_elf("sub");
    let mut traces =
        Traces::from_logs_minimal(&logs, instructions.clone(), &Default::default()).unwrap();

    // Tamper: change NEXT_PC_0 on row 0 to a wrong value
    let original = *traces.cpus[0].get_main(0, cols::NEXT_PC_0);
    let tampered = original + FieldElement::<F>::from(42u64);
    traces.cpus[0].set_main(0, cols::NEXT_PC_0, tampered);

    assert!(
        !prove_and_verify_vm_minimal(&elf, &mut traces),
        "Tampered next_pc should cause verification failure"
    );
}

/// Verify rejects inflated table_counts that don't match proof sub-proof count.
#[test]
fn test_verify_rejects_inflated_table_counts() {
    let elf_bytes = crate::test_utils::asm_elf_bytes("sub");
    let proof_options = ProofOptions::default_test_options();

    let vm_proof = crate::prove_with_options(&elf_bytes, &proof_options, &Default::default())
        .expect("Prover should succeed on valid program");

    // Inflate cpu count — total won't match proof.proofs.len()
    let tampered_proof = crate::VmProof {
        table_counts: crate::TableCounts {
            cpu: 10000,
            ..vm_proof.table_counts.clone()
        },
        ..vm_proof
    };

    let result =
        crate::verify_with_options(&tampered_proof, &elf_bytes, &proof_options, None, None);
    assert!(
        result.is_err(),
        "Inflated table_counts should be rejected, got {:?}",
        result
    );
}

/// Proves a program that uses W-suffix instructions (ADDIW, SRLIW) on a
/// register holding a 64-bit value with non-zero upper 32 bits.
/// Verifies that the full 64-bit value is preserved in the MEMW_R chain.
#[test]
fn test_prove_wsuffix_64bit() {
    let elf_bytes = crate::test_utils::asm_elf_bytes("test_wsuffix_64bit");
    let vm_proof = prove_vm_minimal(&elf_bytes, &[], &Default::default());
    assert!(
        verify_vm_minimal(&vm_proof, &elf_bytes),
        "W-suffix 64-bit register test should verify"
    );
}

/// Proves a minimal Rust std program that uses `init_allocator()` and
/// `String::from("Hello World") + commit`. Exercises the full Rust-std stack:
/// TLSF heap init (SRL on high-bit values), CSR instructions injected by
/// the Rust toolchain, and the allocator's memory access patterns.
#[test]
fn test_prove_allocator_minimal_reproducer() {
    let _ = env_logger::builder().is_test(true).try_init();
    let workspace_root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .to_path_buf();
    let elf_bytes =
        std::fs::read(workspace_root.join("executor/program_artifacts/rust/allocator.elf"))
            .expect("allocator.elf not found — run `make compile-programs-rust`");
    let proof = prove_vm_minimal(&elf_bytes, &[], &Default::default());
    assert!(
        verify_vm_minimal(&proof, &elf_bytes),
        "allocator.elf should verify"
    );
    assert_eq!(proof.public_output, b"Hello World");
}

/// Minimal Rust program that proves: no_std, no_main, no allocator, no
/// syscalls crate. Only Commit + Halt ecalls (both have receivers).
#[test]
fn test_pure_commit_rust() {
    let workspace_root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .to_path_buf();
    let elf_bytes =
        std::fs::read(workspace_root.join("executor/program_artifacts/rust/pure_commit.elf"))
            .expect("pure_commit.elf not found — run `make compile-programs-rust`");
    let proof = prove_vm_minimal(&elf_bytes, &[], &Default::default());
    assert!(
        verify_vm_minimal(&proof, &elf_bytes),
        "pure_commit.elf should verify"
    );
    assert_eq!(proof.public_output, vec![0xAA, 0xBB, 0xCC, 0xDD]);
}

/// Backward-compatibility: `prove_with_inputs` with empty input must match `prove`.
#[test]
fn test_prove_with_input_empty() {
    let elf_bytes = crate::test_utils::asm_elf_bytes("sub");
    let result =
        crate::prove_with_inputs(&elf_bytes, &[]).expect("prove_with_inputs should succeed on sub");
    assert!(
        crate::verify(&result, &elf_bytes).expect("verify should not error"),
        "prove_with_inputs(empty) proof should verify"
    );
}

/// ASM test: reads private input from 0xFF000000, commits 8 bytes.
#[test]
fn test_prove_private_input_xpage() {
    let elf_bytes = crate::test_utils::asm_elf_bytes("test_private_input_xpage");
    let input: Vec<u8> = (0u8..16).collect();
    let proof = prove_vm_minimal(&elf_bytes, &input, &Default::default());
    assert!(verify_vm_minimal(&proof, &elf_bytes), "proof should verify");
    assert_eq!(proof.public_output, input[4..12].to_vec());
}

/// Same ASM, different input values — output depends on input.
#[test]
fn test_prove_private_input_different_values() {
    let elf_bytes = crate::test_utils::asm_elf_bytes("test_private_input_xpage");
    let input: Vec<u8> = vec![
        0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99,
        0x00,
    ];
    let proof = prove_vm_minimal(&elf_bytes, &input, &Default::default());
    assert!(verify_vm_minimal(&proof, &elf_bytes), "proof should verify");
    assert_eq!(proof.public_output, input[4..12].to_vec());
}

/// End-to-end: EF zkVM IO interface — demo guest reads its private input via
/// `read_input` and emits it back through TWO `write_output` calls. The
/// COMMIT AIR's running `x254` index concatenates them; the resulting proof's
/// `public_output` must equal the original input.
#[test]
fn test_prove_ef_io_demo_concatenates() {
    let workspace_root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root")
        .to_path_buf();
    let elf_bytes =
        std::fs::read(workspace_root.join("executor/program_artifacts/rust/ef_io_demo.elf"))
            .expect("ef_io_demo.elf not found — run `make compile-programs-rust`");
    let input: &[u8] = b"hello world!";
    let proof = crate::prove_with_inputs(&elf_bytes, input).expect("prove should succeed");
    assert!(
        crate::verify(&proof, &elf_bytes).expect("verify should not error"),
        "ef_io_demo should verify"
    );
    assert_eq!(
        proof.public_output, input,
        "two write_output calls must concatenate"
    );
}

/// End-to-end: Rust std program with private input.
#[test]
fn test_prove_commit_sum() {
    let workspace_root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root")
        .to_path_buf();
    let elf_bytes =
        std::fs::read(workspace_root.join("executor/program_artifacts/rust/commit_sum.elf"))
            .expect("commit_sum.elf not found — run `make compile-programs-rust`");
    let input = &[3u8, 5u8];
    let proof = prove_vm_minimal(&elf_bytes, input, &Default::default());
    assert!(
        verify_vm_minimal(&proof, &elf_bytes),
        "commit_sum should verify"
    );
    assert_eq!(proof.public_output, vec![8u8]);
}

#[test]
#[ignore = "takes too long"]
fn test_prove_ethrex_5_transfers() {
    let _ = env_logger::builder().is_test(true).try_init();
    let workspace_root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .to_path_buf();
    let elf_bytes =
        std::fs::read(workspace_root.join("executor/program_artifacts/rust/ethrex.elf"))
            .expect("need ethrex.elf");
    let input =
        std::fs::read(workspace_root.join("executor/tests/ethrex_5_transfers.bin")).unwrap();
    let proof = crate::prove_with_inputs(&elf_bytes, &input).expect("prove");
    assert!(
        crate::verify(&proof, &elf_bytes).expect("verify"),
        "ethrex 5-transfer block should verify"
    );
}

#[test]
#[ignore = "takes too long"]
fn test_prove_ethrex_20_transfers() {
    let _ = env_logger::builder().is_test(true).try_init();
    let workspace_root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .to_path_buf();
    let elf_bytes =
        std::fs::read(workspace_root.join("executor/program_artifacts/rust/ethrex.elf"))
            .expect("need ethrex.elf");
    let input =
        std::fs::read(workspace_root.join("executor/tests/ethrex_20_transfers.bin")).unwrap();
    let proof = crate::prove_with_inputs(&elf_bytes, &input).expect("prove");
    assert!(
        crate::verify(&proof, &elf_bytes).expect("verify"),
        "ethrex 20-transfer block should verify"
    );
}

#[test]
#[ignore = "takes too long"]
fn test_prove_ethrex_empty_block() {
    let _ = env_logger::builder().is_test(true).try_init();
    let workspace_root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .to_path_buf();
    let elf_bytes =
        std::fs::read(workspace_root.join("executor/program_artifacts/rust/ethrex.elf"))
            .expect("need ethrex.elf");
    let input =
        std::fs::read(workspace_root.join("executor/tests/ethrex_empty_block.bin")).unwrap();
    let proof = crate::prove_with_inputs(&elf_bytes, &input).expect("prove");
    assert!(
        crate::verify(&proof, &elf_bytes).expect("verify"),
        "ethrex empty block should verify"
    );
    assert_eq!(proof.public_output.len(), 160);
}

// =============================================================================
// Security: private-input tamper tests
// =============================================================================

/// Verifier must reject when num_private_input_pages is zeroed out.
/// The proof contains a non-preprocessed PAGE sub-proof for the private input,
/// but the verifier expects 0 such pages → proof count mismatch.
#[test]
fn test_verify_rejects_tampered_num_private_input_pages_zero() {
    let elf_bytes = crate::test_utils::asm_elf_bytes("test_private_input_xpage");
    let input: Vec<u8> = (0u8..16).collect();
    let vm_proof = crate::prove_with_inputs(&elf_bytes, &input).expect("prove should succeed");

    // Baseline: untampered proof must verify.
    assert!(
        crate::verify(&vm_proof, &elf_bytes).expect("verify should not error"),
        "Baseline proof must verify before tampering"
    );
    assert!(
        vm_proof.num_private_input_pages > 0,
        "proof should have private pages"
    );

    // Tamper: zero out private input pages.
    let tampered = crate::VmProof {
        num_private_input_pages: 0,
        ..vm_proof
    };

    let result = crate::verify(&tampered, &elf_bytes);
    assert!(
        result.is_err() || !result.unwrap(),
        "Verifier must reject proof with num_private_input_pages zeroed out"
    );
}

/// Verifier must reject when num_private_input_pages is inflated beyond actual.
/// The proof has 1 private page but we claim 2 → proof count mismatch.
#[test]
fn test_verify_rejects_inflated_num_private_input_pages() {
    let elf_bytes = crate::test_utils::asm_elf_bytes("test_private_input_xpage");
    let input: Vec<u8> = (0u8..16).collect();
    let vm_proof = crate::prove_with_inputs(&elf_bytes, &input).expect("prove should succeed");

    assert_eq!(
        vm_proof.num_private_input_pages, 1,
        "16 bytes fits in 1 page"
    );

    let tampered = crate::VmProof {
        num_private_input_pages: 2,
        ..vm_proof
    };

    let result = crate::verify(&tampered, &elf_bytes);
    assert!(
        result.is_err() || !result.unwrap(),
        "Verifier must reject proof with inflated num_private_input_pages"
    );
}

/// Verifier must reject num_private_input_pages that exceeds the max bound.
/// The early bounds check should catch this before constructing AIRs.
#[test]
fn test_verify_rejects_num_private_input_pages_exceeds_max() {
    let elf_bytes = crate::test_utils::asm_elf_bytes("test_private_input_xpage");
    let input: Vec<u8> = (0u8..16).collect();
    let vm_proof = crate::prove_with_inputs(&elf_bytes, &input).expect("prove should succeed");

    let tampered = crate::VmProof {
        num_private_input_pages: crate::tables::page::max_private_input_pages() + 1,
        ..vm_proof
    };

    assert!(
        crate::verify(&tampered, &elf_bytes).is_err(),
        "Verifier must error on num_private_input_pages exceeding max"
    );
}

/// Verifier must reject tampered public_output when private input is used.
/// Ensures the COMMIT bus balance check still works with non-preprocessed pages.
#[test]
fn test_verify_rejects_private_input_with_tampered_public_output() {
    let elf_bytes = crate::test_utils::asm_elf_bytes("test_private_input_xpage");
    let input: Vec<u8> = (0u8..16).collect();
    let vm_proof = crate::prove_with_inputs(&elf_bytes, &input).expect("prove should succeed");

    assert!(
        crate::verify(&vm_proof, &elf_bytes).expect("verify should not error"),
        "Baseline must verify"
    );

    let mut tampered_output = vm_proof.public_output.clone();
    tampered_output[0] ^= 0x01;
    let tampered = crate::VmProof {
        public_output: tampered_output,
        ..vm_proof
    };

    let verified =
        crate::verify(&tampered, &elf_bytes).expect("verify should not error on tampered output");
    assert!(
        !verified,
        "Verifier must reject proof with tampered public_output (private input present)"
    );
}

/// VmProof must not contain a field that stores the raw private input bytes.
/// This is a structural check: the proof struct should only carry
/// `num_private_input_pages`, not the actual input data.
#[test]
fn test_proof_does_not_contain_private_input_field() {
    let elf_bytes = crate::test_utils::asm_elf_bytes("test_private_input_xpage");
    let input: Vec<u8> = (0xA0u8..0xB0).collect();
    let vm_proof = crate::prove_with_inputs(&elf_bytes, &input).expect("prove should succeed");

    // The VmProof struct should only contain num_private_input_pages (a count),
    // not the actual bytes. Verify the proof's public fields don't contain them.
    assert_eq!(vm_proof.num_private_input_pages, 1);
    // public_output is the committed output, NOT the private input.
    // It should contain bytes [4..12] of the input (what the ASM program commits).
    assert_eq!(vm_proof.public_output, input[4..12].to_vec());
    // No `private_input` field exists — this is enforced by the type system,
    // but explicitly document that the proof carries only the page count.
    assert!(
        vm_proof.num_private_input_pages <= 1,
        "Only the page count is stored, not the bytes"
    );
}

/// Regression test: addiw with negative immediate must verify.
/// arg2_sign_bit is the sign bit of rv2 (bit 31), not of arg2, per spec
/// constraint CPU-CE61: MSB16[arg2_sign_bit; rv2[1]].
/// For I-type word instructions (rs2=x0, rv2=0), arg2_sign_bit must be 0.
#[test]
fn test_addiw_neg_immediate() {
    let elf_bytes = crate::test_utils::asm_elf_bytes("test_addiw_neg");
    let proof = prove_vm_minimal(&elf_bytes, &[], &Default::default());
    assert!(
        verify_vm_minimal(&proof, &elf_bytes),
        "addiw with negative immediate should verify"
    );
}

/// Regression test: both main and aux field element counts must be nonzero for any real ELF.
/// Guards against silent under-count if a table is added to `Traces` but not enumerated in
/// `total_field_elements` or `total_auxiliary_field_elements`.
#[test]
fn test_count_elements_nonzero() {
    let elf_bytes = crate::test_utils::asm_elf_bytes("addi_one");
    let (main, aux) = crate::count_elements(&elf_bytes, &[]).expect("count_elements failed");
    assert!(
        main > 0,
        "total_field_elements should be nonzero (got {main})"
    );
    assert!(
        aux > 0,
        "total_auxiliary_field_elements should be nonzero (got {aux})"
    );
}

/// Prove and verify the FIRST continuation epoch in isolation. Epoch 0 starts
/// from the program's initial memory/registers (so its init is correct) and does
/// not terminate, so it is proven with the HALT table excluded (`include_halt = false`).
#[test]
fn test_prove_first_epoch_without_halt() {
    use crate::compute_expected_commit_bus_balance_view;
    use crate::tables::trace_builder::build_initial_image;
    use crate::test_utils::asm_elf_bytes;

    let _ = env_logger::builder().is_test(true).try_init();
    let elf_bytes = asm_elf_bytes("arith_8");
    let elf = Elf::load(&elf_bytes).unwrap();

    // arith_8 is ~10 cycles; a power-of-two epoch_size of 4 makes epoch 0 an
    // intermediate epoch (4 cycles → no CPU padding rows) with the program
    // continuing past it.
    let epoch_size = 4;
    let epochs = Executor::new(&elf, vec![])
        .unwrap()
        .run_epochs(epoch_size)
        .unwrap();
    assert!(epochs.len() >= 2);

    // Epoch 0's starting memory/registers are the program-start image; it does
    // not halt (is_final=false).
    let image = build_initial_image(&elf, &[]);
    let register_init = crate::tables::register::register_init_from_entry_point(elf.entry_point);
    let mut traces = Traces::from_image_and_logs(
        &elf,
        &image,
        &register_init,
        &epochs[0].logs,
        &MaxRowsConfig::default(),
        &[],
        false,
        false,
        #[cfg(feature = "disk-spill")]
        stark::storage_mode::StorageMode::Ram,
    )
    .unwrap();

    let proof_options = ProofOptions::default_test_options();
    let table_counts = traces.table_counts();
    let airs = VmAirs::new(
        &elf,
        &proof_options,
        true,
        &traces.page_configs,
        &table_counts,
        None,
        false,
        None,
        None,
        None,
    );

    let multi_proof = multi_prove_ram(
        airs.air_trace_pairs(&mut traces),
        &mut DefaultTranscript::<E>::new(&[]),
    )
    .expect("first epoch failed to prove");

    let views: Vec<StarkProofView<F, E, ()>> = multi_proof
        .proofs
        .iter()
        .map(StarkProofView::Owned)
        .collect();
    let mut replay = DefaultTranscript::<E>::new(&[]);
    let expected_bus_balance = compute_expected_commit_bus_balance_view(
        &airs.air_refs(),
        &views,
        &traces.public_output_bytes,
        0,
        &mut replay,
    )
    .expect("fingerprint collision in test");

    assert!(
        Verifier::multi_verify_views(
            &airs.air_refs(),
            &views,
            &mut DefaultTranscript::<E>::new(&[]),
            &expected_bus_balance,
        ),
        "first epoch (HALT excluded) failed to verify"
    );
}

/// Prove and verify a NON-first continuation epoch (epoch 1) in isolation. Its
/// starting memory and registers come from epoch 0's boundary snapshot, and it
/// does not terminate (HALT excluded).
#[test]
fn test_prove_second_epoch_from_snapshot() {
    use crate::compute_expected_commit_bus_balance_view;
    use crate::tables::register;
    use crate::test_utils::asm_elf_bytes;

    let _ = env_logger::builder().is_test(true).try_init();
    let elf_bytes = asm_elf_bytes("arith_8");
    let elf = Elf::load(&elf_bytes).unwrap();

    // arith_8 is ~10 cycles; epoch_size 4 (power of two) yields epochs 4/4/2, so
    // epoch 1 is intermediate (4 cycles → no CPU padding rows).
    let epoch_size = 4;
    let epochs = Executor::new(&elf, vec![])
        .unwrap()
        .run_epochs(epoch_size)
        .unwrap();
    assert!(epochs.len() >= 3, "need an intermediate epoch 1");

    // Epoch 1 starts from epoch 0's ending memory + register snapshot.
    let image: std::collections::HashMap<u64, u8> = epochs[0].end_memory.iter_bytes().collect();
    let register_init =
        register::register_init_from_snapshot(&epochs[0].end_registers, epochs[0].end_pc);

    let mut traces = Traces::from_image_and_logs(
        &elf,
        &image,
        &register_init,
        &epochs[1].logs,
        &MaxRowsConfig::default(),
        &[],
        false,
        false,
        #[cfg(feature = "disk-spill")]
        stark::storage_mode::StorageMode::Ram,
    )
    .unwrap();

    let proof_options = ProofOptions::default_test_options();
    let table_counts = traces.table_counts();
    // The REGISTER commitment is built from this epoch's boundary register init.
    let airs = VmAirs::new(
        &elf,
        &proof_options,
        true,
        &traces.page_configs,
        &table_counts,
        None,
        false,
        Some(&register_init),
        None,
        None,
    );

    let multi_proof = multi_prove_ram(
        airs.air_trace_pairs(&mut traces),
        &mut DefaultTranscript::<E>::new(&[]),
    )
    .expect("second epoch failed to prove");

    let views: Vec<StarkProofView<F, E, ()>> = multi_proof
        .proofs
        .iter()
        .map(StarkProofView::Owned)
        .collect();
    let mut replay = DefaultTranscript::<E>::new(&[]);
    let expected_bus_balance = compute_expected_commit_bus_balance_view(
        &airs.air_refs(),
        &views,
        &traces.public_output_bytes,
        0,
        &mut replay,
    )
    .expect("fingerprint collision in test");

    assert!(
        Verifier::multi_verify_views(
            &airs.air_refs(),
            &views,
            &mut DefaultTranscript::<E>::new(&[]),
            &expected_bus_balance,
        ),
        "second epoch (register init from snapshot) failed to verify"
    );
}

/// An epoch proof can COMMIT the local-to-global table inertly — committed
/// columns, but no GlobalMemory bus and no constraints in the epoch proof — and
/// still verify, exposing the L2G commitment root that the final proof (Step 4)
/// will bind to. The cross-epoch GlobalMemory matching is proven separately.
#[test]
fn test_epoch_proof_commits_l2g() {
    use crate::compute_expected_commit_bus_balance_view;
    use crate::tables::local_to_global;
    use crate::tables::register;
    use crate::tables::trace_builder::{build_initial_image, epoch_touched_cells};
    use crate::test_utils::asm_elf_bytes;
    use std::collections::HashMap;

    let _ = env_logger::builder().is_test(true).try_init();
    let elf_bytes = asm_elf_bytes("all_loadstore_32");
    let elf = Elf::load(&elf_bytes).unwrap();

    // Power-of-two epoch size: all_loadstore_32 is ~34 cycles, so epoch_size 8
    // makes epoch 0 an intermediate epoch with no CPU padding rows.
    let epoch_size = 8;
    let epochs = Executor::new(&elf, vec![])
        .unwrap()
        .run_epochs(epoch_size)
        .unwrap();
    assert!(epochs.len() >= 2);

    let image = build_initial_image(&elf, &[]);
    let register_init = register::register_init_from_entry_point(elf.entry_point);
    let mut traces = Traces::from_image_and_logs(
        &elf,
        &image,
        &register_init,
        &epochs[0].logs,
        &MaxRowsConfig::default(),
        &[],
        false,
        false,
        #[cfg(feature = "disk-spill")]
        stark::storage_mode::StorageMode::Ram,
    )
    .unwrap();

    // Epoch 0's local-to-global trace, committed inertly below.
    let register_init0 = register::register_init_from_entry_point(elf.entry_point);
    let touched = epoch_touched_cells(&elf, &image, &register_init0, &epochs[0].logs).unwrap();
    let initial_memory: HashMap<u64, u64> = image.iter().map(|(&a, &v)| (a, v as u64)).collect();
    let boundaries = local_to_global::epoch_boundaries(&initial_memory, &[touched]);
    let mut l2g_trace = local_to_global::generate_local_to_global_trace(&boundaries[0]);

    let proof_options = ProofOptions::default_test_options();
    let table_counts = traces.table_counts();
    let airs = VmAirs::new(
        &elf,
        &proof_options,
        true,
        &traces.page_configs,
        &table_counts,
        None,
        false,
        None,
        None,
        None,
    );

    // Inert L2G AIR: commits the trace columns, but no bus and no constraints.
    let inert_l2g_air: AirWithBuses<
        F,
        E,
        stark::lookup::NullBoundaryConstraintBuilder,
        (),
        EmptyConstraints,
    > = AirWithBuses::new(
        local_to_global::cols::NUM_COLUMNS,
        AuxiliaryTraceBuildData {
            interactions: vec![],
        },
        &proof_options,
        1,
        EmptyConstraints,
    );

    let mut pairs = airs.air_trace_pairs(&mut traces);
    pairs.push((&inert_l2g_air, &mut l2g_trace, &()));

    let multi_proof = multi_prove_ram(pairs, &mut DefaultTranscript::<E>::new(&[]))
        .expect("epoch proof with inert L2G failed to prove");

    let mut refs = airs.air_refs();
    refs.push(&inert_l2g_air);

    let views: Vec<StarkProofView<F, E, ()>> = multi_proof
        .proofs
        .iter()
        .map(StarkProofView::Owned)
        .collect();
    let mut replay = DefaultTranscript::<E>::new(&[]);
    let expected_bus_balance = compute_expected_commit_bus_balance_view(
        &refs,
        &views,
        &traces.public_output_bytes,
        0,
        &mut replay,
    )
    .expect("fingerprint collision in test");

    assert!(
        Verifier::multi_verify_views(
            &refs,
            &views,
            &mut DefaultTranscript::<E>::new(&[]),
            &expected_bus_balance,
        ),
        "epoch proof with inert L2G failed to verify"
    );

    // The L2G table (pushed last) is committed: its Merkle root is exposed and
    // non-zero — this is the `R_i` the final proof will be bound to in Step 4.
    let l2g_root = multi_proof
        .proofs
        .last()
        .unwrap()
        .lde_trace_main_merkle_root;
    assert_ne!(
        l2g_root, [0u8; 32],
        "L2G commitment root should be non-zero"
    );
}

/// End-to-end continuation pipeline over a real ELF: split execution into epochs,
/// prove+verify each epoch (each committing its local-to-global table inertly and
/// exposing a root R_i), prove the cross-epoch GlobalMemory bus balances over the
/// real per-epoch boundaries, and finally bind the cross-epoch proof to the REAL
/// per-epoch roots. The R_i collected from the independent epoch proofs equal the
/// per-epoch L2G sub-table roots in the cross-epoch proof — that root equality is
/// the shared-commitment linkage between the epoch proofs and the global memory
/// argument.
#[test]
fn test_continuation_pipeline_end_to_end() {
    use crate::compute_expected_commit_bus_balance_view;
    use crate::tables::local_to_global;
    use crate::tables::register;
    use crate::tables::trace_builder::{build_initial_image, epoch_touched_cells};
    use crate::test_utils::asm_elf_bytes;
    use std::collections::HashMap;

    let _ = env_logger::builder().is_test(true).try_init();
    let elf_bytes = asm_elf_bytes("all_loadstore_32");
    let elf = Elf::load(&elf_bytes).unwrap();

    // Split execution into power-of-two epochs (all_loadstore_32 is ~34 cycles, so
    // epoch_size 8 gives intermediate epochs with no CPU padding rows).
    let epoch_size = 8;
    let epochs = Executor::new(&elf, vec![])
        .unwrap()
        .run_epochs(epoch_size)
        .unwrap();
    assert!(epochs.len() >= 2);

    let image0 = build_initial_image(&elf, &[]);
    let initial_memory: HashMap<u64, u64> = image0.iter().map(|(&a, &v)| (a, v as u64)).collect();

    // Pass 1: each epoch's starting state + the cells it touches. Epoch 0 starts
    // from the program image; epoch i>0 from epoch i-1's boundary snapshot.
    let mut images: Vec<HashMap<u64, u8>> = Vec::with_capacity(epochs.len());
    let mut register_inits: Vec<Vec<u32>> = Vec::with_capacity(epochs.len());
    let mut all_touched: Vec<Vec<(u64, u64, u64)>> = Vec::with_capacity(epochs.len());
    for (i, epoch) in epochs.iter().enumerate() {
        let (image_i, register_init_i) = if i == 0 {
            (
                image0.clone(),
                register::register_init_from_entry_point(elf.entry_point),
            )
        } else {
            let image_i: HashMap<u64, u8> = epochs[i - 1].end_memory.iter_bytes().collect();
            let register_init_i = register::register_init_from_snapshot(
                &epochs[i - 1].end_registers,
                epochs[i - 1].end_pc,
            );
            (image_i, register_init_i)
        };
        let touched_i = epoch_touched_cells(&elf, &image_i, &register_init_i, &epoch.logs).unwrap();
        images.push(image_i);
        register_inits.push(register_init_i);
        all_touched.push(touched_i);
    }
    let boundaries = local_to_global::epoch_boundaries(&initial_memory, &all_touched);

    let proof_options = ProofOptions::default_test_options();

    // Pass 2: prove+verify each epoch, committing boundaries[i] inertly, and
    // collect the L2G commitment root each epoch proof exposes.
    let mut epoch_roots = Vec::with_capacity(epochs.len());
    for (i, epoch) in epochs.iter().enumerate() {
        let is_final = i == epochs.len() - 1;
        let mut traces = Traces::from_image_and_logs(
            &elf,
            &images[i],
            &register_inits[i],
            &epoch.logs,
            &MaxRowsConfig::default(),
            &[],
            is_final,
            false,
            #[cfg(feature = "disk-spill")]
            stark::storage_mode::StorageMode::Ram,
        )
        .unwrap();

        let table_counts = traces.table_counts();
        let register_init_arg = if i == 0 {
            None
        } else {
            Some(register_inits[i].as_slice())
        };
        let airs = VmAirs::new(
            &elf,
            &proof_options,
            true,
            &traces.page_configs,
            &table_counts,
            None,
            is_final,
            register_init_arg,
            None,
            None,
        );

        let mut l2g_trace = local_to_global::generate_local_to_global_trace(&boundaries[i]);
        let inert_l2g_air: AirWithBuses<
            F,
            E,
            stark::lookup::NullBoundaryConstraintBuilder,
            (),
            EmptyConstraints,
        > = AirWithBuses::new(
            local_to_global::cols::NUM_COLUMNS,
            AuxiliaryTraceBuildData {
                interactions: vec![],
            },
            &proof_options,
            1,
            EmptyConstraints,
        );

        let mut pairs = airs.air_trace_pairs(&mut traces);
        pairs.push((&inert_l2g_air, &mut l2g_trace, &()));
        let multi_proof = multi_prove_ram(pairs, &mut DefaultTranscript::<E>::new(&[]))
            .expect("epoch proof failed to prove");

        let mut refs = airs.air_refs();
        refs.push(&inert_l2g_air);
        let views: Vec<StarkProofView<F, E, ()>> = multi_proof
            .proofs
            .iter()
            .map(StarkProofView::Owned)
            .collect();
        let mut replay = DefaultTranscript::<E>::new(&[]);
        let expected_bus_balance = compute_expected_commit_bus_balance_view(
            &refs,
            &views,
            &traces.public_output_bytes,
            0,
            &mut replay,
        )
        .expect("fingerprint collision in test");
        assert!(
            Verifier::multi_verify_views(
                &refs,
                &views,
                &mut DefaultTranscript::<E>::new(&[]),
                &expected_bus_balance,
            ),
            "epoch {i} failed to verify"
        );

        epoch_roots.push(
            multi_proof
                .proofs
                .last()
                .unwrap()
                .lde_trace_main_merkle_root,
        );
    }

    // The cross-epoch GlobalMemory bus balances over the real per-epoch boundaries.
    assert!(
        crate::tests::local_to_global_bus_tests::prove_and_verify(&boundaries),
        "final GlobalMemory bus must balance over real epoch data"
    );

    // The cross-epoch proof is bound to the REAL per-epoch roots: the L2G root each
    // epoch proof exposed equals the per-epoch L2G sub-table root in the final proof.
    let final_proof = crate::tests::local_to_global_bus_tests::prove_global(&boundaries);
    assert!(
        crate::verify_l2g_commitment_binding_view(
            &epoch_roots,
            MultiProofView::Owned(&final_proof)
        ),
        "final proof must be bound to the real per-epoch L2G roots"
    );
}

/// A continuation epoch built with `l2g_memory_bookend = true` proves and verifies:
/// PAGE no longer bookends the touched RAM bytes (they self-cancel), and the
/// local-to-global table provides their `Memory`-bus init/fini instead. The epoch
/// `Memory` bus still nets to zero — L2G has replaced PAGE as the bookend.
#[test]
fn test_epoch_memory_bus_with_l2g_bookend() {
    use crate::compute_expected_commit_bus_balance_view;
    use crate::tables::local_to_global;
    use crate::tables::register;
    use crate::tables::trace_builder::build_initial_image;
    use crate::test_utils::asm_elf_bytes;
    use std::collections::HashMap;

    let _ = env_logger::builder().is_test(true).try_init();
    let elf_bytes = asm_elf_bytes("all_loadstore_32");
    let elf = Elf::load(&elf_bytes).unwrap();

    // Power-of-two epoch size: all_loadstore_32 is ~34 cycles, so epoch_size 8
    // makes epoch 0 an intermediate epoch with no CPU padding rows.
    let epoch_size = 8;
    let epochs = Executor::new(&elf, vec![])
        .unwrap()
        .run_epochs(epoch_size)
        .unwrap();
    assert!(epochs.len() >= 2);

    // Epoch 0 starts from the program image; build it with the L2G memory bookend.
    let image = build_initial_image(&elf, &[]);
    let register_init = register::register_init_from_entry_point(elf.entry_point);
    let mut traces = Traces::from_image_and_logs(
        &elf,
        &image,
        &register_init,
        &epochs[0].logs,
        &MaxRowsConfig::default(),
        &[],
        false,
        true,
        #[cfg(feature = "disk-spill")]
        stark::storage_mode::StorageMode::Ram,
    )
    .unwrap();
    let initial_memory: HashMap<u64, u64> = image.iter().map(|(&a, &v)| (a, v as u64)).collect();
    let boundaries =
        local_to_global::epoch_boundaries(&initial_memory, &[traces.touched_memory_cells.clone()]);
    traces.local_to_global = local_to_global::generate_local_to_global_trace(&boundaries[0]);

    let proof_options = ProofOptions::default_test_options();
    let table_counts = traces.table_counts();
    let airs = VmAirs::new(
        &elf,
        &proof_options,
        true,
        &traces.page_configs,
        &table_counts,
        None,
        false,
        None,
        None,
        None,
    );

    // L2G air on the epoch-local Memory bus (the bookend that replaces PAGE).
    let l2g_air: AirWithBuses<
        F,
        E,
        stark::lookup::NullBoundaryConstraintBuilder,
        (),
        EmptyConstraints,
    > = AirWithBuses::new(
        local_to_global::cols::NUM_COLUMNS,
        AuxiliaryTraceBuildData {
            interactions: local_to_global::memory_bus_interactions(),
        },
        &proof_options,
        1,
        EmptyConstraints,
    );

    // Take the L2G trace out of `traces` so `air_trace_pairs` can borrow the rest.
    let mut l2g_trace = std::mem::replace(
        &mut traces.local_to_global,
        local_to_global::generate_local_to_global_trace(&[]),
    );

    let mut pairs = airs.air_trace_pairs(&mut traces);
    pairs.push((&l2g_air, &mut l2g_trace, &()));
    let multi_proof = multi_prove_ram(pairs, &mut DefaultTranscript::<E>::new(&[]))
        .expect("epoch with L2G memory bookend failed to prove");

    let mut refs = airs.air_refs();
    refs.push(&l2g_air);
    let views: Vec<StarkProofView<F, E, ()>> = multi_proof
        .proofs
        .iter()
        .map(StarkProofView::Owned)
        .collect();
    let mut replay = DefaultTranscript::<E>::new(&[]);
    let expected_bus_balance = compute_expected_commit_bus_balance_view(
        &refs,
        &views,
        &traces.public_output_bytes,
        0,
        &mut replay,
    )
    .expect("fingerprint collision in test");

    assert!(
        Verifier::multi_verify_views(
            &refs,
            &views,
            &mut DefaultTranscript::<E>::new(&[]),
            &expected_bus_balance,
        ),
        "epoch Memory bus must balance with L2G bookend + PAGE excluding touched cells"
    );
}
