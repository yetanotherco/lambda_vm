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

use executor::elf::Elf;
use executor::vm::execution::Executor;
use executor::vm::instruction::decoding::Instruction;
use executor::vm::memory::U64HashMap;

use stark::constraints::transition::TransitionConstraint;
use stark::lookup::{AirWithBuses, AuxiliaryTraceBuildData};
use stark::proof::options::ProofOptions;
use stark::prover::{IsStarkProver, Prover};
use stark::trace::TraceTable;
use stark::traits::AIR;
use stark::verifier::{IsStarkVerifier, Verifier};

use crate::constraints::cpu::create_all_cpu_constraints;
use crate::tables::bitwise::{
    BitwiseLookup, bus_interactions as bitwise_bus_interactions, cols as bitwise_cols,
    generate_bitwise_trace, update_multiplicities,
};
use crate::tables::cpu::{CpuOperation, bus_interactions as cpu_bus_interactions};
use crate::tables::lt::{LtOperation, bus_interactions as lt_bus_interactions, generate_lt_trace};
use crate::tables::trace_builder::Traces;
use crate::tables::types::{GoldilocksExtension, GoldilocksField};

type F = GoldilocksField;
type E = GoldilocksExtension;

/// Helper to run an ELF from the program_artifacts directory
fn run_asm_elf(name: &str) -> (Vec<executor::vm::logs::Log>, U64HashMap<Instruction>) {
    let path = format!(
        "{}/executor/program_artifacts/asm/{}.elf",
        env!("CARGO_MANIFEST_DIR").replace("/prover", ""),
        name
    );
    let elf_data = std::fs::read(&path).expect("Failed to read ELF");
    let program = Elf::load(&elf_data).expect("Failed to load ELF");
    let executor = Executor::new(program.image, program.entry_point, vec![])
        .expect("Failed to create executor");
    let result = executor.run().expect("Failed to run program");
    (result.logs, result.instructions)
}

/// Test helper: collect bitwise lookups from logs for minimal table tests.
fn collect_bitwise_lookups(
    logs: &[executor::vm::logs::Log],
    instructions: &U64HashMap<Instruction>,
) -> Vec<(BitwiseLookup, u8, u8, u8)> {
    logs.iter()
        .enumerate()
        .flat_map(|(i, log)| {
            let instruction = *instructions.get(&log.current_pc).unwrap();
            let op = CpuOperation::from_log(log, (i as u64) * 4, instruction);
            op.collect_bitwise_lookups()
        })
        .collect()
}

// =============================================================================
// AIR creation helpers
// =============================================================================

fn create_cpu_air(
    proof_options: &ProofOptions,
) -> AirWithBuses<F, E, stark::lookup::NullBoundaryConstraintBuilder, ()> {
    // Get all CPU constraints
    let (is_bit, add, other, _) = create_all_cpu_constraints();

    // All CPU constraints: IS_BIT + ADD + other (Branch, Arg1, Arg2, etc.)
    let mut transition_constraints: Vec<Box<dyn TransitionConstraint<F, E>>> = Vec::new();
    for c in is_bit {
        transition_constraints.push(Box::new(c));
    }
    for c in add {
        transition_constraints.push(Box::new(c));
    }
    for c in other {
        transition_constraints.push(c);
    }

    let auxiliary_trace_build_data = AuxiliaryTraceBuildData {
        interactions: cpu_bus_interactions(),
    };

    AirWithBuses::new(
        crate::tables::cpu::cols::NUM_COLUMNS,
        auxiliary_trace_build_data,
        proof_options,
        1,
        transition_constraints,
    )
}

fn create_bitwise_air(
    proof_options: &ProofOptions,
) -> AirWithBuses<F, E, stark::lookup::NullBoundaryConstraintBuilder, ()> {
    let transition_constraints: Vec<Box<dyn TransitionConstraint<F, E>>> = vec![];

    let auxiliary_trace_build_data = AuxiliaryTraceBuildData {
        interactions: bitwise_bus_interactions(),
    };

    AirWithBuses::new(
        crate::tables::bitwise::cols::NUM_COLUMNS,
        auxiliary_trace_build_data,
        proof_options,
        1,
        transition_constraints,
    )
}

fn create_lt_air(
    proof_options: &ProofOptions,
) -> AirWithBuses<F, E, stark::lookup::NullBoundaryConstraintBuilder, ()> {
    let transition_constraints: Vec<Box<dyn TransitionConstraint<F, E>>> = vec![];

    let auxiliary_trace_build_data = AuxiliaryTraceBuildData {
        interactions: lt_bus_interactions(),
    };

    AirWithBuses::new(
        crate::tables::lt::cols::NUM_COLUMNS,
        auxiliary_trace_build_data,
        proof_options,
        1,
        transition_constraints,
    )
}

/// Collect bitwise lookups from LT operations.
///
/// The LT table sends:
/// - MSB16 lookups (×2 per row: for lhs_msb and rhs_msb)
/// - IS_HALFWORD lookups (×4 per row: for lhs_sub_rhs range checks)
fn collect_bitwise_lookups_from_lt(lt_ops: &[LtOperation]) -> Vec<(BitwiseLookup, u8, u8, u8)> {
    let mut lookups = Vec::new();

    for op in lt_ops {
        // MSB16 for lhs_msb (bits 48-63 of lhs)
        let lhs_2 = ((op.lhs >> 48) & 0xFFFF) as u16;
        let x = (lhs_2 & 0xFF) as u8;
        let y = ((lhs_2 >> 8) & 0xFF) as u8;
        lookups.push((BitwiseLookup::Msb16, x, y, 0));

        // MSB16 for rhs_msb (bits 48-63 of rhs)
        let rhs_2 = ((op.rhs >> 48) & 0xFFFF) as u16;
        let x = (rhs_2 & 0xFF) as u8;
        let y = ((rhs_2 >> 8) & 0xFF) as u8;
        lookups.push((BitwiseLookup::Msb16, x, y, 0));

        // IS_HALFWORD for lhs_sub_rhs (4 halfwords)
        let lhs_sub_rhs = op.lhs.wrapping_sub(op.rhs);
        let sub_0 = (lhs_sub_rhs & 0xFFFF) as u16;
        let sub_1 = ((lhs_sub_rhs >> 16) & 0xFFFF) as u16;
        let sub_2 = ((lhs_sub_rhs >> 32) & 0xFFFF) as u16;
        let sub_3 = ((lhs_sub_rhs >> 48) & 0xFFFF) as u16;

        lookups.push((
            BitwiseLookup::IsHalf,
            (sub_0 & 0xFF) as u8,
            ((sub_0 >> 8) & 0xFF) as u8,
            0,
        ));
        lookups.push((
            BitwiseLookup::IsHalf,
            (sub_1 & 0xFF) as u8,
            ((sub_1 >> 8) & 0xFF) as u8,
            0,
        ));
        lookups.push((
            BitwiseLookup::IsHalf,
            (sub_2 & 0xFF) as u8,
            ((sub_2 >> 8) & 0xFF) as u8,
            0,
        ));
        lookups.push((
            BitwiseLookup::IsHalf,
            (sub_3 & 0xFF) as u8,
            ((sub_3 >> 8) & 0xFF) as u8,
            0,
        ));
    }

    lookups
}

/// Collect LT lookups from executor logs.
///
/// For each instruction that triggers an SLT or BLT operation, creates an LtOperation
/// with the arg1, arg2, and signed values.
fn collect_lt_lookups_from_logs(
    logs: &[executor::vm::logs::Log],
    instructions: &U64HashMap<Instruction>,
) -> Vec<LtOperation> {
    use executor::vm::instruction::decoding::{ArithOp, Comparison};

    let mut lookups = Vec::new();

    for log in logs {
        let instruction = *instructions.get(&log.current_pc).unwrap();

        let is_slt = matches!(
            &instruction,
            Instruction::Arith {
                op: ArithOp::SetLessThan,
                ..
            } | Instruction::Arith {
                op: ArithOp::SetLessThanU,
                ..
            } | Instruction::ArithImm {
                op: ArithOp::SetLessThan,
                ..
            } | Instruction::ArithImm {
                op: ArithOp::SetLessThanU,
                ..
            } | Instruction::ArithW {
                op: ArithOp::SetLessThan,
                ..
            } | Instruction::ArithW {
                op: ArithOp::SetLessThanU,
                ..
            } | Instruction::ArithImmW {
                op: ArithOp::SetLessThan,
                ..
            } | Instruction::ArithImmW {
                op: ArithOp::SetLessThanU,
                ..
            }
        );

        let is_blt = matches!(
            &instruction,
            Instruction::Branch {
                cond: Comparison::LessThan,
                ..
            } | Instruction::Branch {
                cond: Comparison::LessThanUnsigned,
                ..
            } | Instruction::Branch {
                cond: Comparison::GreaterOrEqual,
                ..
            } | Instruction::Branch {
                cond: Comparison::GreaterOrEqualUnsigned,
                ..
            }
        );

        if is_slt || is_blt {
            // Determine signed flag
            let signed = matches!(
                &instruction,
                Instruction::Arith {
                    op: ArithOp::SetLessThan,
                    ..
                } | Instruction::ArithImm {
                    op: ArithOp::SetLessThan,
                    ..
                } | Instruction::ArithW {
                    op: ArithOp::SetLessThan,
                    ..
                } | Instruction::ArithImmW {
                    op: ArithOp::SetLessThan,
                    ..
                } | Instruction::Branch {
                    cond: Comparison::LessThan,
                    ..
                } | Instruction::Branch {
                    cond: Comparison::GreaterOrEqual,
                    ..
                }
            );

            // Get arg1 and arg2 values
            // For SLT: arg1 = rv1, arg2 = rv2 or imm
            // For BLT: arg1 = rv1, arg2 = rv2
            let arg1 = log.src1_val;
            let arg2 = match &instruction {
                Instruction::ArithImm { imm, .. } | Instruction::ArithImmW { imm, .. } => {
                    *imm as i64 as u64
                }
                _ => log.src2_val,
            };

            lookups.push(LtOperation::new(arg1, arg2, signed));
        }
    }

    lookups
}

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

use std::collections::HashMap;

/// Generates a minimal bitwise trace containing only the rows needed for the given lookups.
///
/// **WARNING: FOR TESTING ONLY - NOT PRODUCTION SAFE!**
fn generate_minimal_bitwise_trace(lookups: &[(BitwiseLookup, u8, u8, u8)]) -> TraceTable<F, E> {
    // Collect unique (x, y, z) tuples and count multiplicities per lookup type
    let mut row_data: HashMap<(u8, u8, u8), [u64; 11]> = HashMap::new();

    for (lookup_type, x, y, z) in lookups {
        let key = (*x, *y, *z);
        let mu_idx = match lookup_type {
            BitwiseLookup::AndByte => 0,
            BitwiseLookup::OrByte => 1,
            BitwiseLookup::XorByte => 2,
            BitwiseLookup::Msb8 => 3,
            BitwiseLookup::Msb16 => 4,
            BitwiseLookup::Zero => 5,
            BitwiseLookup::IsByte => 6,
            BitwiseLookup::IsHalf => 7,
            BitwiseLookup::IsB20 => 8,
            BitwiseLookup::Hwsl => 9,
            BitwiseLookup::Hwslc => 10,
        };
        row_data.entry(key).or_insert([0; 11])[mu_idx] += 1;
    }

    // Need at least 4 rows for FRI, pad to power of 2
    let unique_rows: Vec<_> = row_data.keys().cloned().collect();
    let num_rows = unique_rows.len().max(4).next_power_of_two();

    type FE = math::field::element::FieldElement<F>;
    let mut data = vec![FE::zero(); num_rows * bitwise_cols::NUM_COLUMNS];

    for (row_idx, (x, y, z)) in unique_rows.iter().enumerate() {
        let base = row_idx * bitwise_cols::NUM_COLUMNS;
        let x = *x as u32;
        let y = *y as u32;
        let z = *z as u32;

        // Input columns
        data[base + bitwise_cols::X] = FE::from(x as u64);
        data[base + bitwise_cols::Y] = FE::from(y as u64);
        data[base + bitwise_cols::Z] = FE::from(z as u64);

        // Bitwise operation results
        data[base + bitwise_cols::AND] = FE::from((x & y) as u64);
        data[base + bitwise_cols::OR] = FE::from((x | y) as u64);
        data[base + bitwise_cols::XOR] = FE::from((x ^ y) as u64);

        // MSB extractions
        let msb8 = (x >> 7) & 1;
        let halfword = x + y * 256;
        let msb16 = (halfword >> 15) & 1;
        data[base + bitwise_cols::MSB8] = FE::from(msb8 as u64);
        data[base + bitwise_cols::MSB16] = FE::from(msb16 as u64);

        // Zero check
        let is_zero = if x == 0 && y == 0 { 1u64 } else { 0u64 };
        data[base + bitwise_cols::ZERO] = FE::from(is_zero);

        // Shift operations
        let sll = if z == 0 {
            halfword
        } else {
            (halfword << z) & 0xFFFF
        };
        let sllc = if z == 0 { 0 } else { halfword >> (16 - z) };
        data[base + bitwise_cols::SLL] = FE::from(sll as u64);
        data[base + bitwise_cols::SLLC] = FE::from(sllc as u64);

        // Multiplicity columns
        let mus = &row_data[&(x as u8, y as u8, z as u8)];
        data[base + bitwise_cols::MU_AND] = FE::from(mus[0]);
        data[base + bitwise_cols::MU_OR] = FE::from(mus[1]);
        data[base + bitwise_cols::MU_XOR] = FE::from(mus[2]);
        data[base + bitwise_cols::MU_MSB8] = FE::from(mus[3]);
        data[base + bitwise_cols::MU_MSB16] = FE::from(mus[4]);
        data[base + bitwise_cols::MU_ZERO] = FE::from(mus[5]);
        data[base + bitwise_cols::MU_IS_BYTE] = FE::from(mus[6]);
        data[base + bitwise_cols::MU_IS_HALF] = FE::from(mus[7]);
        data[base + bitwise_cols::MU_IS_B20] = FE::from(mus[8]);
        data[base + bitwise_cols::MU_HWSL] = FE::from(mus[9]);
        data[base + bitwise_cols::MU_HWSLC] = FE::from(mus[10]);
    }

    TraceTable::new_main(data, bitwise_cols::NUM_COLUMNS, 1)
}

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
