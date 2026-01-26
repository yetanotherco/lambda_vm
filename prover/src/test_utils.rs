//! Shared utilities for tests and benchmarks.
//!
//! This module contains common helper functions used across:
//! - Integration tests (`src/tests/prove_elfs_tests.rs`)
//! - Benchmarks (`benches/vm_prover_benchmark.rs`)
//!
//! Functions include:
//! - ELF execution helpers
//! - Lookup collection from executor logs
//! - Minimal trace generation for testing
//! - AIR creation helpers

use std::path::PathBuf;

use executor::elf::Elf;
use executor::vm::execution::Executor;
use executor::vm::instruction::decoding::Instruction;
use executor::vm::logs::Log;
use executor::vm::memory::U64HashMap;
use math::field::element::FieldElement;
use stark::constraints::transition::TransitionConstraint;
use stark::lookup::{AirWithBuses, AuxiliaryTraceBuildData, NullBoundaryConstraintBuilder};
use stark::proof::options::ProofOptions;
use stark::trace::TraceTable;

use crate::constraints::cpu::create_all_cpu_constraints;
use crate::tables::bitwise::{
    BitwiseLookup, bus_interactions as bitwise_bus_interactions, cols as bitwise_cols,
};
use crate::tables::cpu::{
    CpuOperation, bus_interactions as cpu_bus_interactions, cols as cpu_cols,
};
use crate::tables::lt::{LtOperation, bus_interactions as lt_bus_interactions, cols as lt_cols};
use crate::tables::types::{GoldilocksExtension, GoldilocksField};

pub type F = GoldilocksField;
pub type E = GoldilocksExtension;
pub type FE = FieldElement<F>;

pub type VmAir = AirWithBuses<F, E, NullBoundaryConstraintBuilder, ()>;

// =============================================================================
// ELF Execution Helpers
// =============================================================================

/// Helper to run an ELF from the program_artifacts directory.
///
/// Returns the execution logs and instruction map.
pub fn run_asm_elf(name: &str) -> (Vec<Log>, U64HashMap<Instruction>) {
    // Get workspace root by going up one level from prover directory
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest_dir
        .parent()
        .expect("Failed to get workspace root from CARGO_MANIFEST_DIR");

    // Build path to ELF file
    let path = workspace_root
        .join("executor")
        .join("program_artifacts")
        .join("asm")
        .join(format!("{}.elf", name));

    let elf_data =
        std::fs::read(&path).unwrap_or_else(|_| panic!("Failed to read ELF: {}", path.display()));
    let program = Elf::load(&elf_data).expect("Failed to load ELF");
    let executor = Executor::new(&program, vec![]).expect("Failed to create executor");
    let result = executor.run().expect("Failed to run program");
    (result.logs, result.instructions.into_instruction_map())
}

// =============================================================================
// Lookup Collection Functions
// =============================================================================

/// Collect bitwise lookups from executor logs for minimal table generation.
pub fn collect_bitwise_lookups_from_logs(
    logs: &[Log],
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

/// Collect LT lookups from executor logs.
///
/// For each instruction that triggers an SLT or BLT operation, creates an LtOperation
/// with the arg1, arg2, and signed values.
pub fn collect_lt_lookups_from_logs(
    logs: &[Log],
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

/// Collect bitwise lookups from LT operations (MSB16 and IS_HALFWORD).
///
/// The LT table sends:
/// - MSB16 lookups (×2 per row: for lhs_msb and rhs_msb)
/// - IS_HALFWORD lookups (×4 per row: for lhs_sub_rhs range checks)
pub fn collect_bitwise_lookups_from_lt(lt_ops: &[LtOperation]) -> Vec<(BitwiseLookup, u8, u8, u8)> {
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
        for shift in [0, 16, 32, 48] {
            let half = ((lhs_sub_rhs >> shift) & 0xFFFF) as u16;
            lookups.push((
                BitwiseLookup::IsHalf,
                (half & 0xFF) as u8,
                ((half >> 8) & 0xFF) as u8,
                0,
            ));
        }
    }

    lookups
}

// =============================================================================
// Minimal Trace Generation (for testing/benchmarking only)
// =============================================================================

/// Generate a minimal bitwise trace containing only the rows needed for the given lookups.
///
/// This is much faster than the full 2^20 row table for benchmarking/testing.
///
/// **WARNING: FOR TESTING/BENCHMARKING ONLY - NOT PRODUCTION SAFE!**
/// The verifier expects the full deterministic 2^20 row public table.
pub fn generate_minimal_bitwise_trace(lookups: &[(BitwiseLookup, u8, u8, u8)]) -> TraceTable<F, E> {
    use std::collections::HashMap;

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

// =============================================================================
// AIR Creation Helpers
// =============================================================================

/// Create CPU AIR with all constraints and bus interactions.
pub fn create_cpu_air(proof_options: &ProofOptions) -> VmAir {
    // Get all CPU constraints
    let (is_bit, add, other, _) = create_all_cpu_constraints();

    // All CPU constraints
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
        cpu_cols::NUM_COLUMNS,
        auxiliary_trace_build_data,
        proof_options,
        1,
        transition_constraints,
    )
}

/// Create Bitwise AIR with bus interactions.
pub fn create_bitwise_air(proof_options: &ProofOptions) -> VmAir {
    let transition_constraints: Vec<Box<dyn TransitionConstraint<F, E>>> = vec![];

    let auxiliary_trace_build_data = AuxiliaryTraceBuildData {
        interactions: bitwise_bus_interactions(),
    };

    AirWithBuses::new(
        bitwise_cols::NUM_COLUMNS,
        auxiliary_trace_build_data,
        proof_options,
        1,
        transition_constraints,
    )
}

/// Create LT AIR with bus interactions.
pub fn create_lt_air(proof_options: &ProofOptions) -> VmAir {
    let transition_constraints: Vec<Box<dyn TransitionConstraint<F, E>>> = vec![];

    let auxiliary_trace_build_data = AuxiliaryTraceBuildData {
        interactions: lt_bus_interactions(),
    };

    AirWithBuses::new(
        lt_cols::NUM_COLUMNS,
        auxiliary_trace_build_data,
        proof_options,
        1,
        transition_constraints,
    )
}
