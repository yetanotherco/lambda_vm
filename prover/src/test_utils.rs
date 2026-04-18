#![allow(clippy::needless_range_loop, clippy::assign_op_pattern)]
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
    BitwiseOperation, BitwiseOperationType, bus_interactions as bitwise_bus_interactions,
    cols as bitwise_cols,
};
use crate::tables::branch::{
    branch_constraints, bus_interactions as branch_bus_interactions, cols as branch_cols,
};
use crate::tables::commit::{
    bus_interactions as commit_bus_interactions, cols as commit_cols,
    create_constraints as commit_constraints,
};
use crate::tables::cpu::{
    CpuOperation, bus_interactions as cpu_bus_interactions, cols as cpu_cols,
};
use crate::tables::decode::{bus_interactions as decode_bus_interactions, cols as decode_cols};
use crate::tables::dvrm::{
    bus_interactions as dvrm_bus_interactions, cols as dvrm_cols, dvrm_constraints,
};
use crate::tables::halt::{bus_interactions as halt_bus_interactions, cols as halt_cols};
use crate::tables::load::{
    bus_interactions as load_bus_interactions, cols as load_cols, constraints as load_constraints,
};
use crate::tables::lt::{LtOperation, bus_interactions as lt_bus_interactions, cols as lt_cols};
use crate::tables::memw::{
    bus_interactions as memw_bus_interactions, cols as memw_cols, constraints as memw_constraints,
};
use crate::tables::memw_aligned::{
    bus_interactions as memw_aligned_bus_interactions, cols as memw_aligned_cols,
    constraints as memw_aligned_constraints,
};
use crate::tables::memw_register::{
    bus_interactions as memw_register_bus_interactions, cols as memw_register_cols,
    constraints as memw_register_constraints,
};
use crate::tables::mul::{bus_interactions as mul_bus_interactions, cols as mul_cols};
use crate::tables::page::{bus_interactions as page_bus_interactions, cols as page_cols};
use crate::tables::register::{
    bus_interactions as register_bus_interactions, cols as register_cols,
};
use crate::tables::shift::{
    bus_interactions as shift_bus_interactions, cols as shift_cols, shift_constraints,
};
use crate::tables::types::{GoldilocksExtension, GoldilocksField};

pub type F = GoldilocksField;
pub type E = GoldilocksExtension;
pub type FE = FieldElement<F>;

pub type VmAir = AirWithBuses<F, E, NullBoundaryConstraintBuilder, ()>;

// =============================================================================
// ELF Execution Helpers
// =============================================================================

/// Returns the raw ELF bytes for an assembly test program.
pub fn asm_elf_bytes(name: &str) -> Vec<u8> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest_dir
        .parent()
        .expect("Failed to get workspace root from CARGO_MANIFEST_DIR");

    let path = workspace_root
        .join("executor")
        .join("program_artifacts")
        .join("asm")
        .join(format!("{}.elf", name));

    std::fs::read(&path).unwrap_or_else(|_| panic!("Failed to read ELF: {}", path.display()))
}

/// Helper to run an ELF from the program_artifacts directory.
///
/// Returns the ELF, execution logs, and instruction map.
pub fn run_asm_elf(name: &str) -> (Elf, Vec<Log>, U64HashMap<Instruction>) {
    let elf_data = asm_elf_bytes(name);
    let elf = Elf::load(&elf_data).expect("Failed to load ELF");
    let executor = Executor::new(&elf, vec![]).expect("Failed to create executor");
    let result = executor.run().expect("Failed to run program");
    (elf, result.logs, result.instructions)
}

// =============================================================================
// Lookup Collection Functions
// =============================================================================

/// Collect bitwise lookups from executor logs for minimal table generation.
pub fn collect_bitwise_ops_from_logs(
    logs: &[Log],
    instructions: &U64HashMap<Instruction>,
) -> Vec<BitwiseOperation> {
    logs.iter()
        .enumerate()
        .flat_map(|(i, log)| {
            let instruction = *instructions.get(&log.current_pc).unwrap();
            let op = CpuOperation::from_log_and_instruction(log, (i as u64) * 4, instruction);
            op.collect_bitwise_ops()
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

/// Collect LOAD operations from executor logs.
///
/// Creates LoadOperation objects for each Load instruction in the logs.
pub fn collect_load_ops_from_logs(
    logs: &[Log],
    instructions: &U64HashMap<Instruction>,
) -> Vec<crate::tables::load::LoadOperation> {
    use executor::vm::instruction::decoding::LoadStoreWidth;

    let mut load_ops = Vec::new();

    for log in logs {
        let instruction = *instructions.get(&log.current_pc).unwrap();

        if let Instruction::Load { width, .. } = instruction {
            let base_address = log.src1_val.wrapping_add(match instruction {
                Instruction::Load { offset, .. } => offset as i64 as u64,
                _ => 0,
            });

            let (byte_count, signed) = match width {
                LoadStoreWidth::Byte => (1, true),
                LoadStoreWidth::ByteUnsigned => (1, false),
                LoadStoreWidth::Half => (2, true),
                LoadStoreWidth::HalfUnsigned => (2, false),
                LoadStoreWidth::Word => (4, true),
                LoadStoreWidth::WordUnsigned => (4, false),
                LoadStoreWidth::DoubleWord => (8, false),
            };

            let loaded_value = log.dst_val;

            // Extract individual bytes from loaded value
            let mut res_bytes = [0u64; 8];
            for (j, byte) in res_bytes.iter_mut().take(byte_count).enumerate() {
                *byte = (loaded_value >> (j * 8)) & 0xFF;
            }

            // Sign/zero extend the upper bytes
            if byte_count < 8 {
                let msb = res_bytes[byte_count - 1];
                let sign_bit = (msb >> 7) & 1;
                let fill = if signed && sign_bit == 1 { 0xFF } else { 0 };
                for byte in res_bytes.iter_mut().skip(byte_count) {
                    *byte = fill;
                }
            }

            // Use a dummy timestamp (not used for bitwise lookups)
            let timestamp = 0;

            load_ops.push(crate::tables::load::LoadOperation::new(
                base_address,
                timestamp,
                byte_count as u8,
                signed,
                res_bytes,
            ));
        }
    }

    load_ops
}

/// Collect bitwise lookups from LT operations (MSB16 and IS_HALFWORD).
///
/// The LT table sends:
/// - MSB16 lookups (×2 per row: for lhs_msb and rhs_msb)
/// - IS_HALFWORD lookups (×6 per row: ×4 for lhs_sub_rhs, ×1 for lhs[1], ×1 for rhs[1])
pub fn collect_bitwise_ops_from_lt(lt_ops: &[LtOperation]) -> Vec<BitwiseOperation> {
    let mut lookups = Vec::new();

    for op in lt_ops {
        // MSB16 for lhs_msb (bits 48-63 of lhs)
        let lhs_2 = ((op.lhs >> 48) & 0xFFFF) as u16;
        lookups.push(BitwiseOperation::halfword(
            BitwiseOperationType::Msb16,
            (lhs_2 & 0xFF) as u8,
            ((lhs_2 >> 8) & 0xFF) as u8,
        ));

        // MSB16 for rhs_msb (bits 48-63 of rhs)
        let rhs_2 = ((op.rhs >> 48) & 0xFFFF) as u16;
        lookups.push(BitwiseOperation::halfword(
            BitwiseOperationType::Msb16,
            (rhs_2 & 0xFF) as u8,
            ((rhs_2 >> 8) & 0xFF) as u8,
        ));

        // IS_HALFWORD for lhs_sub_rhs (4 halfwords)
        let lhs_sub_rhs = op.lhs.wrapping_sub(op.rhs);
        for shift in [0, 16, 32, 48] {
            let half = ((lhs_sub_rhs >> shift) & 0xFFFF) as u16;
            lookups.push(BitwiseOperation::halfword(
                BitwiseOperationType::IsHalf,
                (half & 0xFF) as u8,
                ((half >> 8) & 0xFF) as u8,
            ));
        }

        // IS_HALFWORD for lhs[1] (bits 32-47 of lhs)
        let lhs_1 = ((op.lhs >> 32) & 0xFFFF) as u16;
        lookups.push(BitwiseOperation::halfword(
            BitwiseOperationType::IsHalf,
            (lhs_1 & 0xFF) as u8,
            ((lhs_1 >> 8) & 0xFF) as u8,
        ));

        // IS_HALFWORD for rhs[1] (bits 32-47 of rhs)
        let rhs_1 = ((op.rhs >> 32) & 0xFFFF) as u16;
        lookups.push(BitwiseOperation::halfword(
            BitwiseOperationType::IsHalf,
            (rhs_1 & 0xFF) as u8,
            ((rhs_1 >> 8) & 0xFF) as u8,
        ));
    }

    lookups
}

/// Collect bitwise lookups from LOAD operations.
///
/// The LOAD table sends MSB8 lookups for sign bit extraction:
/// - read1: MSB8[res[0]] -> sign_bit
/// - read2: MSB8[res[1]] -> sign_bit
/// - read4: MSB8[res[3]] -> sign_bit
/// - read8: no MSB8 lookup (all 8 bytes are used)
pub fn collect_bitwise_ops_from_load(
    load_ops: &[crate::tables::load::LoadOperation],
) -> Vec<BitwiseOperation> {
    load_ops
        .iter()
        .flat_map(|op| op.collect_bitwise_ops())
        .collect()
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
pub fn generate_minimal_bitwise_trace(ops: &[BitwiseOperation]) -> TraceTable<F, E> {
    use std::collections::HashMap;

    // Collect unique (lo_byte, hi_byte, shift) tuples and count multiplicities per lookup type
    let mut row_data: HashMap<(u8, u8, u8), [u64; 10]> = HashMap::new();

    for op in ops {
        let key = (op.x, op.y, op.z);
        let mu_idx = match op.lookup_type {
            BitwiseOperationType::AndByte => 0,
            BitwiseOperationType::OrByte => 1,
            BitwiseOperationType::XorByte => 2,
            BitwiseOperationType::Msb8 => 3,
            BitwiseOperationType::Msb16 => 4,
            BitwiseOperationType::Zero => 5,
            BitwiseOperationType::IsByte => 6,
            BitwiseOperationType::IsHalf => 7,
            BitwiseOperationType::IsB20 => 8,
            BitwiseOperationType::Hwsl => 9,
        };
        row_data.entry(key).or_insert([0; 10])[mu_idx] += 1;
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
    .with_main_builder(cpu_main_builder_fn)
    .with_builder(cpu_builder_fn)
    .with_name("CPU")
}

/// AirBuilder constraint evaluator for the CPU table (66 constraints).
///
/// Constraints are emitted in the exact same order as create_all_cpu_constraints():
/// 1. IS_BIT (32): one per BIT_FLAG_COLUMNS entry
/// 2. ADD carries (8): ADD+LOAD(2), STORE(2), SUB+BEQ(2), JALR(2)
/// 3. OTHER (26): BranchCond, Ebreak, RegNotRead(6), Arg1(2), Arg2(2),
///    Rvd(2), SltResZero(7), ExtBitZero(3), NextPcAdd(2)
fn cpu_builder_fn(builder: &mut dyn stark::air_builder::AirBuilder<GoldilocksExtension>) {
    use crate::constraints::cpu::BIT_FLAG_COLUMNS;
    use crate::constraints::helpers::assert_is_bit;
    use crate::constraints::templates::INV_SHIFT_32;
    use crate::tables::cpu::cols;
    use math::field::element::FieldElement;

    type FE = FieldElement<GoldilocksExtension>;

    let one = FE::one();
    let two = FE::from(2u64);
    let inv_2_32 = FE::from(INV_SHIFT_32);
    let shift_8 = FE::from(1u64 << 8);
    let shift_16 = FE::from(1u64 << 16);
    let shift_24 = FE::from(1u64 << 24);
    let mask_32 = FE::from((1u64 << 32) - 1);

    // Helper: pack 4 byte values into a 32-bit word (inner fn, no builder capture).
    #[inline]
    fn pack(b0: FE, b1: FE, b2: FE, b3: FE, s8: &FE, s16: &FE, s24: &FE) -> FE {
        b0 + b1 * s8 + b2 * s16 + b3 * s24
    }

    // =================================================================
    // IS_BIT constraints (32)
    // =================================================================
    for &col in BIT_FLAG_COLUMNS {
        let x = builder.main(0, col);
        assert_is_bit(builder, x);
    }

    // =================================================================
    // ADD constraints (8)
    // =================================================================

    // ADD + LOAD (2): arg1 + arg2 = res (all DWordBL)
    {
        let ll = pack(
            builder.main(0, cols::ARG1_0),
            builder.main(0, cols::ARG1_1),
            builder.main(0, cols::ARG1_2),
            builder.main(0, cols::ARG1_3),
            &shift_8,
            &shift_16,
            &shift_24,
        );
        let rl = pack(
            builder.main(0, cols::ARG2_0),
            builder.main(0, cols::ARG2_1),
            builder.main(0, cols::ARG2_2),
            builder.main(0, cols::ARG2_3),
            &shift_8,
            &shift_16,
            &shift_24,
        );
        let sl = pack(
            builder.main(0, cols::RES_0),
            builder.main(0, cols::RES_0 + 1),
            builder.main(0, cols::RES_0 + 2),
            builder.main(0, cols::RES_0 + 3),
            &shift_8,
            &shift_16,
            &shift_24,
        );
        let lh = pack(
            builder.main(0, cols::ARG1_4),
            builder.main(0, cols::ARG1_5),
            builder.main(0, cols::ARG1_6),
            builder.main(0, cols::ARG1_7),
            &shift_8,
            &shift_16,
            &shift_24,
        );
        let rh = pack(
            builder.main(0, cols::ARG2_4),
            builder.main(0, cols::ARG2_5),
            builder.main(0, cols::ARG2_6),
            builder.main(0, cols::ARG2_7),
            &shift_8,
            &shift_16,
            &shift_24,
        );
        let sh = pack(
            builder.main(0, cols::RES_0 + 4),
            builder.main(0, cols::RES_0 + 5),
            builder.main(0, cols::RES_0 + 6),
            builder.main(0, cols::RES_0 + 7),
            &shift_8,
            &shift_16,
            &shift_24,
        );
        let cond = builder.main(0, cols::ADD) + builder.main(0, cols::LOAD);
        let c0 = (ll + rl - sl) * inv_2_32;
        builder.assert_zero(cond * c0 * (one - c0));
        let c1 = (lh + rh + c0 - sh) * inv_2_32;
        builder.assert_zero(cond * c1 * (one - c1));
    }

    // STORE (2): arg1 + imm = res
    {
        let ll = pack(
            builder.main(0, cols::ARG1_0),
            builder.main(0, cols::ARG1_1),
            builder.main(0, cols::ARG1_2),
            builder.main(0, cols::ARG1_3),
            &shift_8,
            &shift_16,
            &shift_24,
        );
        let lh = pack(
            builder.main(0, cols::ARG1_4),
            builder.main(0, cols::ARG1_5),
            builder.main(0, cols::ARG1_6),
            builder.main(0, cols::ARG1_7),
            &shift_8,
            &shift_16,
            &shift_24,
        );
        let rl = builder.main(0, cols::IMM_0);
        let rh = builder.main(0, cols::IMM_1);
        let sl = pack(
            builder.main(0, cols::RES_0),
            builder.main(0, cols::RES_0 + 1),
            builder.main(0, cols::RES_0 + 2),
            builder.main(0, cols::RES_0 + 3),
            &shift_8,
            &shift_16,
            &shift_24,
        );
        let sh = pack(
            builder.main(0, cols::RES_0 + 4),
            builder.main(0, cols::RES_0 + 5),
            builder.main(0, cols::RES_0 + 6),
            builder.main(0, cols::RES_0 + 7),
            &shift_8,
            &shift_16,
            &shift_24,
        );
        let cond = builder.main(0, cols::STORE);
        let c0 = (ll + rl - sl) * inv_2_32;
        builder.assert_zero(cond * c0 * (one - c0));
        let c1 = (lh + rh + c0 - sh) * inv_2_32;
        builder.assert_zero(cond * c1 * (one - c1));
    }

    // SUB + BEQ (2): arg2 + res = arg1
    {
        let ll = pack(
            builder.main(0, cols::ARG2_0),
            builder.main(0, cols::ARG2_1),
            builder.main(0, cols::ARG2_2),
            builder.main(0, cols::ARG2_3),
            &shift_8,
            &shift_16,
            &shift_24,
        );
        let lh = pack(
            builder.main(0, cols::ARG2_4),
            builder.main(0, cols::ARG2_5),
            builder.main(0, cols::ARG2_6),
            builder.main(0, cols::ARG2_7),
            &shift_8,
            &shift_16,
            &shift_24,
        );
        let rl = pack(
            builder.main(0, cols::RES_0),
            builder.main(0, cols::RES_0 + 1),
            builder.main(0, cols::RES_0 + 2),
            builder.main(0, cols::RES_0 + 3),
            &shift_8,
            &shift_16,
            &shift_24,
        );
        let rh = pack(
            builder.main(0, cols::RES_0 + 4),
            builder.main(0, cols::RES_0 + 5),
            builder.main(0, cols::RES_0 + 6),
            builder.main(0, cols::RES_0 + 7),
            &shift_8,
            &shift_16,
            &shift_24,
        );
        let sl = pack(
            builder.main(0, cols::ARG1_0),
            builder.main(0, cols::ARG1_1),
            builder.main(0, cols::ARG1_2),
            builder.main(0, cols::ARG1_3),
            &shift_8,
            &shift_16,
            &shift_24,
        );
        let sh = pack(
            builder.main(0, cols::ARG1_4),
            builder.main(0, cols::ARG1_5),
            builder.main(0, cols::ARG1_6),
            builder.main(0, cols::ARG1_7),
            &shift_8,
            &shift_16,
            &shift_24,
        );
        let cond = builder.main(0, cols::SUB) + builder.main(0, cols::BEQ);
        let c0 = (ll + rl - sl) * inv_2_32;
        builder.assert_zero(cond * c0 * (one - c0));
        let c1 = (lh + rh + c0 - sh) * inv_2_32;
        builder.assert_zero(cond * c1 * (one - c1));
    }

    // JALR (2): pc + instr_size = res; instr_size = 4 - 2*c_type
    {
        let ll = builder.main(0, cols::PC_0);
        let lh = builder.main(0, cols::PC_1);
        let rl = FE::from(4u64) - two * builder.main(0, cols::C_TYPE_INSTRUCTION);
        let sl = pack(
            builder.main(0, cols::RES_0),
            builder.main(0, cols::RES_0 + 1),
            builder.main(0, cols::RES_0 + 2),
            builder.main(0, cols::RES_0 + 3),
            &shift_8,
            &shift_16,
            &shift_24,
        );
        let sh = pack(
            builder.main(0, cols::RES_0 + 4),
            builder.main(0, cols::RES_0 + 5),
            builder.main(0, cols::RES_0 + 6),
            builder.main(0, cols::RES_0 + 7),
            &shift_8,
            &shift_16,
            &shift_24,
        );
        let cond = builder.main(0, cols::JALR);
        let c0 = (ll + rl - sl) * inv_2_32;
        builder.assert_zero(cond * c0 * (one - c0));
        let c1 = (lh + c0 - sh) * inv_2_32;
        builder.assert_zero(cond * c1 * (one - c1));
    }

    // =================================================================
    // OTHER constraints (26)
    // =================================================================

    // BranchCondConstraint (1)
    {
        let jalr = builder.main(0, cols::JALR);
        let blt = builder.main(0, cols::BLT);
        let beq = builder.main(0, cols::BEQ);
        let mp = builder.main(0, cols::MP_SELECTOR);
        let r0 = builder.main(0, cols::RES_0);
        let ieq = builder.main(0, cols::IS_EQUAL);
        let bc = builder.main(0, cols::BRANCH_COND);
        let res_xor_mp = r0 + mp - two * r0 * mp;
        let eq_xor_mp = ieq + mp - two * ieq * mp;
        builder.assert_zero(bc - (jalr + blt * res_xor_mp + beq * eq_xor_mp));
    }

    // EbreakConstraint (1)
    builder.assert_zero(builder.main(0, cols::EBREAK));

    // RegNotReadIsZero rv1 (3)
    {
        let rr = builder.main(0, cols::READ_REGISTER1);
        for &c in &[cols::RV1_0, cols::RV1_1, cols::RV1_2] {
            builder.assert_zero((one - rr) * builder.main(0, c));
        }
    }

    // RegNotReadIsZero rv2 (3)
    {
        let rr = builder.main(0, cols::READ_REGISTER2);
        for &c in &[cols::RV2_0, cols::RV2_1, cols::RV2_2] {
            builder.assert_zero((one - rr) * builder.main(0, c));
        }
    }

    // Arg1LowerConstraint (1)
    {
        let a1lo = pack(
            builder.main(0, cols::ARG1_0),
            builder.main(0, cols::ARG1_1),
            builder.main(0, cols::ARG1_2),
            builder.main(0, cols::ARG1_3),
            &shift_8,
            &shift_16,
            &shift_24,
        );
        let rv1lo = builder.main(0, cols::RV1_0) + builder.main(0, cols::RV1_1) * shift_16;
        builder.assert_zero(a1lo - rv1lo);
    }

    // Arg1UpperConstraint (1)
    {
        let a1hi = pack(
            builder.main(0, cols::ARG1_4),
            builder.main(0, cols::ARG1_5),
            builder.main(0, cols::ARG1_6),
            builder.main(0, cols::ARG1_7),
            &shift_8,
            &shift_16,
            &shift_24,
        );
        let rv1u = builder.main(0, cols::RV1_2);
        let wi = builder.main(0, cols::WORD_INSTR);
        let si = builder.main(0, cols::SIGNED);
        let eb = builder.main(0, cols::RV1_EXT_BIT);
        builder.assert_zero(a1hi - (rv1u * (one - wi) + mask_32 * eb * si));
    }

    // Arg2LowerConstraint (1)
    {
        let a2lo = pack(
            builder.main(0, cols::ARG2[0]),
            builder.main(0, cols::ARG2[1]),
            builder.main(0, cols::ARG2[2]),
            builder.main(0, cols::ARG2[3]),
            &shift_8,
            &shift_16,
            &shift_24,
        );
        let rv2lo = builder.main(0, cols::RV2_0) + builder.main(0, cols::RV2_1) * shift_16;
        let imm0 = builder.main(0, cols::IMM_0);
        let ld = builder.main(0, cols::LOAD);
        let st = builder.main(0, cols::STORE);
        let beq = builder.main(0, cols::BEQ);
        let blt = builder.main(0, cols::BLT);
        let expected = (one - ld) * rv2lo + (one - beq - blt - st) * imm0;
        builder.assert_zero(a2lo - expected);
    }

    // Arg2UpperConstraint (1)
    {
        let a2hi = pack(
            builder.main(0, cols::ARG2[4]),
            builder.main(0, cols::ARG2[5]),
            builder.main(0, cols::ARG2[6]),
            builder.main(0, cols::ARG2[7]),
            &shift_8,
            &shift_16,
            &shift_24,
        );
        let rv2u = builder.main(0, cols::RV2_2);
        let imm1 = builder.main(0, cols::IMM_1);
        let ld = builder.main(0, cols::LOAD);
        let st = builder.main(0, cols::STORE);
        let beq = builder.main(0, cols::BEQ);
        let blt = builder.main(0, cols::BLT);
        let wi = builder.main(0, cols::WORD_INSTR);
        let si = builder.main(0, cols::SIGNED);
        let eb = builder.main(0, cols::RV2_EXT_BIT);
        let rv2_term = (one - wi) * rv2u + si * eb * mask_32;
        let expected = (one - ld) * rv2_term + (one - beq - blt - st) * imm1;
        builder.assert_zero(a2hi - expected);
    }

    // RvdLowerConstraint (1)
    {
        let rvd0 = builder.main(0, cols::RVD_0);
        let rlo = pack(
            builder.main(0, cols::RES[0]),
            builder.main(0, cols::RES[1]),
            builder.main(0, cols::RES[2]),
            builder.main(0, cols::RES[3]),
            &shift_8,
            &shift_16,
            &shift_24,
        );
        let ld = builder.main(0, cols::LOAD);
        builder.assert_zero((one - ld) * (rvd0 - rlo));
    }

    // RvdUpperConstraint (1)
    {
        let rvd1 = builder.main(0, cols::RVD_1);
        let rhi = pack(
            builder.main(0, cols::RES[4]),
            builder.main(0, cols::RES[5]),
            builder.main(0, cols::RES[6]),
            builder.main(0, cols::RES[7]),
            &shift_8,
            &shift_16,
            &shift_24,
        );
        let ld = builder.main(0, cols::LOAD);
        let wi = builder.main(0, cols::WORD_INSTR);
        let reb = builder.main(0, cols::RES_EXT_BIT);
        let expected = (one - wi) * rhi + reb * mask_32;
        builder.assert_zero((one - ld) * (rvd1 - expected));
    }

    // SltResZeroConstraint (7)
    {
        let sb = builder.main(0, cols::SLT) + builder.main(0, cols::BLT);
        for i in 1..8usize {
            builder.assert_zero(sb * builder.main(0, cols::RES[i]));
        }
    }

    // ExtBitZeroConstraint (3)
    {
        let nw = one - builder.main(0, cols::WORD_INSTR);
        builder.assert_zero(nw * builder.main(0, cols::RV1_EXT_BIT));
        builder.assert_zero(nw * builder.main(0, cols::RV2_EXT_BIT));
        builder.assert_zero(nw * builder.main(0, cols::RES_EXT_BIT));
    }

    // NextPcAddConstraint (2)
    {
        let plo = builder.main(0, cols::PC_0);
        let phi = builder.main(0, cols::PC_1);
        let nlo = builder.main(0, cols::NEXT_PC_0);
        let nhi = builder.main(0, cols::NEXT_PC_1);
        let ct = builder.main(0, cols::C_TYPE_INSTRUCTION);
        let bc = builder.main(0, cols::BRANCH_COND);
        let isz = FE::from(4u64) - two * ct;
        let nb = one - bc;
        let c0 = (plo + isz - nlo) * inv_2_32;
        builder.assert_zero(nb * c0 * (one - c0));
        let c1 = (phi + c0 - nhi) * inv_2_32;
        builder.assert_zero(nb * c1 * (one - c1));
    }
}

/// MainAirBuilder (base-field) constraint evaluator for the CPU table (66 constraints).
///
/// Same constraint order as `cpu_builder_fn`, but uses base-field types for 2x speedup.
fn cpu_main_builder_fn(
    builder: &mut dyn stark::air_builder::MainAirBuilder<GoldilocksField, GoldilocksExtension>,
) {
    use crate::constraints::cpu::BIT_FLAG_COLUMNS;
    use crate::constraints::helpers::assert_is_bit_base;
    use crate::constraints::templates::INV_SHIFT_32;
    use crate::tables::cpu::cols;

    type FE = FieldElement<GoldilocksField>;

    let one = FE::one();
    let two = FE::from(2u64);
    let inv_2_32 = FE::from(INV_SHIFT_32);
    let shift_8 = FE::from(1u64 << 8);
    let shift_16 = FE::from(1u64 << 16);
    let shift_24 = FE::from(1u64 << 24);
    let mask_32 = FE::from((1u64 << 32) - 1);

    #[inline]
    fn pack(b0: FE, b1: FE, b2: FE, b3: FE, s8: &FE, s16: &FE, s24: &FE) -> FE {
        b0 + b1 * s8 + b2 * s16 + b3 * s24
    }

    // =================================================================
    // IS_BIT constraints (32)
    // =================================================================
    for &col in BIT_FLAG_COLUMNS {
        let x = builder.main_base(col);
        assert_is_bit_base(builder, x);
    }

    // =================================================================
    // ADD constraints (8)
    // =================================================================

    // ADD + LOAD (2)
    {
        let ll = pack(
            builder.main_base(cols::ARG1_0),
            builder.main_base(cols::ARG1_1),
            builder.main_base(cols::ARG1_2),
            builder.main_base(cols::ARG1_3),
            &shift_8,
            &shift_16,
            &shift_24,
        );
        let rl = pack(
            builder.main_base(cols::ARG2_0),
            builder.main_base(cols::ARG2_1),
            builder.main_base(cols::ARG2_2),
            builder.main_base(cols::ARG2_3),
            &shift_8,
            &shift_16,
            &shift_24,
        );
        let sl = pack(
            builder.main_base(cols::RES_0),
            builder.main_base(cols::RES_0 + 1),
            builder.main_base(cols::RES_0 + 2),
            builder.main_base(cols::RES_0 + 3),
            &shift_8,
            &shift_16,
            &shift_24,
        );
        let lh = pack(
            builder.main_base(cols::ARG1_4),
            builder.main_base(cols::ARG1_5),
            builder.main_base(cols::ARG1_6),
            builder.main_base(cols::ARG1_7),
            &shift_8,
            &shift_16,
            &shift_24,
        );
        let rh = pack(
            builder.main_base(cols::ARG2_4),
            builder.main_base(cols::ARG2_5),
            builder.main_base(cols::ARG2_6),
            builder.main_base(cols::ARG2_7),
            &shift_8,
            &shift_16,
            &shift_24,
        );
        let sh = pack(
            builder.main_base(cols::RES_0 + 4),
            builder.main_base(cols::RES_0 + 5),
            builder.main_base(cols::RES_0 + 6),
            builder.main_base(cols::RES_0 + 7),
            &shift_8,
            &shift_16,
            &shift_24,
        );
        let cond = builder.main_base(cols::ADD) + builder.main_base(cols::LOAD);
        let c0 = (ll + rl - sl) * inv_2_32;
        builder.assert_zero_base(cond * c0 * (one - c0));
        let c1 = (lh + rh + c0 - sh) * inv_2_32;
        builder.assert_zero_base(cond * c1 * (one - c1));
    }

    // STORE (2)
    {
        let ll = pack(
            builder.main_base(cols::ARG1_0),
            builder.main_base(cols::ARG1_1),
            builder.main_base(cols::ARG1_2),
            builder.main_base(cols::ARG1_3),
            &shift_8,
            &shift_16,
            &shift_24,
        );
        let lh = pack(
            builder.main_base(cols::ARG1_4),
            builder.main_base(cols::ARG1_5),
            builder.main_base(cols::ARG1_6),
            builder.main_base(cols::ARG1_7),
            &shift_8,
            &shift_16,
            &shift_24,
        );
        let rl = builder.main_base(cols::IMM_0);
        let rh = builder.main_base(cols::IMM_1);
        let sl = pack(
            builder.main_base(cols::RES_0),
            builder.main_base(cols::RES_0 + 1),
            builder.main_base(cols::RES_0 + 2),
            builder.main_base(cols::RES_0 + 3),
            &shift_8,
            &shift_16,
            &shift_24,
        );
        let sh = pack(
            builder.main_base(cols::RES_0 + 4),
            builder.main_base(cols::RES_0 + 5),
            builder.main_base(cols::RES_0 + 6),
            builder.main_base(cols::RES_0 + 7),
            &shift_8,
            &shift_16,
            &shift_24,
        );
        let cond = builder.main_base(cols::STORE);
        let c0 = (ll + rl - sl) * inv_2_32;
        builder.assert_zero_base(cond * c0 * (one - c0));
        let c1 = (lh + rh + c0 - sh) * inv_2_32;
        builder.assert_zero_base(cond * c1 * (one - c1));
    }

    // SUB + BEQ (2)
    {
        let ll = pack(
            builder.main_base(cols::ARG2_0),
            builder.main_base(cols::ARG2_1),
            builder.main_base(cols::ARG2_2),
            builder.main_base(cols::ARG2_3),
            &shift_8,
            &shift_16,
            &shift_24,
        );
        let lh = pack(
            builder.main_base(cols::ARG2_4),
            builder.main_base(cols::ARG2_5),
            builder.main_base(cols::ARG2_6),
            builder.main_base(cols::ARG2_7),
            &shift_8,
            &shift_16,
            &shift_24,
        );
        let rl = pack(
            builder.main_base(cols::RES_0),
            builder.main_base(cols::RES_0 + 1),
            builder.main_base(cols::RES_0 + 2),
            builder.main_base(cols::RES_0 + 3),
            &shift_8,
            &shift_16,
            &shift_24,
        );
        let rh = pack(
            builder.main_base(cols::RES_0 + 4),
            builder.main_base(cols::RES_0 + 5),
            builder.main_base(cols::RES_0 + 6),
            builder.main_base(cols::RES_0 + 7),
            &shift_8,
            &shift_16,
            &shift_24,
        );
        let sl = pack(
            builder.main_base(cols::ARG1_0),
            builder.main_base(cols::ARG1_1),
            builder.main_base(cols::ARG1_2),
            builder.main_base(cols::ARG1_3),
            &shift_8,
            &shift_16,
            &shift_24,
        );
        let sh = pack(
            builder.main_base(cols::ARG1_4),
            builder.main_base(cols::ARG1_5),
            builder.main_base(cols::ARG1_6),
            builder.main_base(cols::ARG1_7),
            &shift_8,
            &shift_16,
            &shift_24,
        );
        let cond = builder.main_base(cols::SUB) + builder.main_base(cols::BEQ);
        let c0 = (ll + rl - sl) * inv_2_32;
        builder.assert_zero_base(cond * c0 * (one - c0));
        let c1 = (lh + rh + c0 - sh) * inv_2_32;
        builder.assert_zero_base(cond * c1 * (one - c1));
    }

    // JALR (2)
    {
        let ll = builder.main_base(cols::PC_0);
        let lh = builder.main_base(cols::PC_1);
        let rl = FE::from(4u64) - two * builder.main_base(cols::C_TYPE_INSTRUCTION);
        let sl = pack(
            builder.main_base(cols::RES_0),
            builder.main_base(cols::RES_0 + 1),
            builder.main_base(cols::RES_0 + 2),
            builder.main_base(cols::RES_0 + 3),
            &shift_8,
            &shift_16,
            &shift_24,
        );
        let sh = pack(
            builder.main_base(cols::RES_0 + 4),
            builder.main_base(cols::RES_0 + 5),
            builder.main_base(cols::RES_0 + 6),
            builder.main_base(cols::RES_0 + 7),
            &shift_8,
            &shift_16,
            &shift_24,
        );
        let cond = builder.main_base(cols::JALR);
        let c0 = (ll + rl - sl) * inv_2_32;
        builder.assert_zero_base(cond * c0 * (one - c0));
        let c1 = (lh + c0 - sh) * inv_2_32;
        builder.assert_zero_base(cond * c1 * (one - c1));
    }

    // =================================================================
    // OTHER constraints (26)
    // =================================================================

    // BranchCondConstraint (1)
    {
        let jalr = builder.main_base(cols::JALR);
        let blt = builder.main_base(cols::BLT);
        let beq = builder.main_base(cols::BEQ);
        let mp = builder.main_base(cols::MP_SELECTOR);
        let r0 = builder.main_base(cols::RES_0);
        let ieq = builder.main_base(cols::IS_EQUAL);
        let bc = builder.main_base(cols::BRANCH_COND);
        let res_xor_mp = r0 + mp - two * r0 * mp;
        let eq_xor_mp = ieq + mp - two * ieq * mp;
        builder.assert_zero_base(bc - (jalr + blt * res_xor_mp + beq * eq_xor_mp));
    }

    // EbreakConstraint (1)
    builder.assert_zero_base(builder.main_base(cols::EBREAK));

    // RegNotReadIsZero rv1 (3)
    {
        let rr = builder.main_base(cols::READ_REGISTER1);
        for &c in &[cols::RV1_0, cols::RV1_1, cols::RV1_2] {
            builder.assert_zero_base((one - rr) * builder.main_base(c));
        }
    }

    // RegNotReadIsZero rv2 (3)
    {
        let rr = builder.main_base(cols::READ_REGISTER2);
        for &c in &[cols::RV2_0, cols::RV2_1, cols::RV2_2] {
            builder.assert_zero_base((one - rr) * builder.main_base(c));
        }
    }

    // Arg1LowerConstraint (1)
    {
        let a1lo = pack(
            builder.main_base(cols::ARG1_0),
            builder.main_base(cols::ARG1_1),
            builder.main_base(cols::ARG1_2),
            builder.main_base(cols::ARG1_3),
            &shift_8,
            &shift_16,
            &shift_24,
        );
        let rv1lo = builder.main_base(cols::RV1_0) + builder.main_base(cols::RV1_1) * shift_16;
        builder.assert_zero_base(a1lo - rv1lo);
    }

    // Arg1UpperConstraint (1)
    {
        let a1hi = pack(
            builder.main_base(cols::ARG1_4),
            builder.main_base(cols::ARG1_5),
            builder.main_base(cols::ARG1_6),
            builder.main_base(cols::ARG1_7),
            &shift_8,
            &shift_16,
            &shift_24,
        );
        let rv1u = builder.main_base(cols::RV1_2);
        let wi = builder.main_base(cols::WORD_INSTR);
        let si = builder.main_base(cols::SIGNED);
        let eb = builder.main_base(cols::RV1_EXT_BIT);
        builder.assert_zero_base(a1hi - (rv1u * (one - wi) + mask_32 * eb * si));
    }

    // Arg2LowerConstraint (1)
    {
        let a2lo = pack(
            builder.main_base(cols::ARG2[0]),
            builder.main_base(cols::ARG2[1]),
            builder.main_base(cols::ARG2[2]),
            builder.main_base(cols::ARG2[3]),
            &shift_8,
            &shift_16,
            &shift_24,
        );
        let rv2lo = builder.main_base(cols::RV2_0) + builder.main_base(cols::RV2_1) * shift_16;
        let imm0 = builder.main_base(cols::IMM_0);
        let ld = builder.main_base(cols::LOAD);
        let st = builder.main_base(cols::STORE);
        let beq = builder.main_base(cols::BEQ);
        let blt = builder.main_base(cols::BLT);
        let expected = (one - ld) * rv2lo + (one - beq - blt - st) * imm0;
        builder.assert_zero_base(a2lo - expected);
    }

    // Arg2UpperConstraint (1)
    {
        let a2hi = pack(
            builder.main_base(cols::ARG2[4]),
            builder.main_base(cols::ARG2[5]),
            builder.main_base(cols::ARG2[6]),
            builder.main_base(cols::ARG2[7]),
            &shift_8,
            &shift_16,
            &shift_24,
        );
        let rv2u = builder.main_base(cols::RV2_2);
        let imm1 = builder.main_base(cols::IMM_1);
        let ld = builder.main_base(cols::LOAD);
        let st = builder.main_base(cols::STORE);
        let beq = builder.main_base(cols::BEQ);
        let blt = builder.main_base(cols::BLT);
        let wi = builder.main_base(cols::WORD_INSTR);
        let si = builder.main_base(cols::SIGNED);
        let eb = builder.main_base(cols::RV2_EXT_BIT);
        let rv2_term = (one - wi) * rv2u + si * eb * mask_32;
        let expected = (one - ld) * rv2_term + (one - beq - blt - st) * imm1;
        builder.assert_zero_base(a2hi - expected);
    }

    // RvdLowerConstraint (1)
    {
        let rvd0 = builder.main_base(cols::RVD_0);
        let rlo = pack(
            builder.main_base(cols::RES[0]),
            builder.main_base(cols::RES[1]),
            builder.main_base(cols::RES[2]),
            builder.main_base(cols::RES[3]),
            &shift_8,
            &shift_16,
            &shift_24,
        );
        let ld = builder.main_base(cols::LOAD);
        builder.assert_zero_base((one - ld) * (rvd0 - rlo));
    }

    // RvdUpperConstraint (1)
    {
        let rvd1 = builder.main_base(cols::RVD_1);
        let rhi = pack(
            builder.main_base(cols::RES[4]),
            builder.main_base(cols::RES[5]),
            builder.main_base(cols::RES[6]),
            builder.main_base(cols::RES[7]),
            &shift_8,
            &shift_16,
            &shift_24,
        );
        let ld = builder.main_base(cols::LOAD);
        let wi = builder.main_base(cols::WORD_INSTR);
        let reb = builder.main_base(cols::RES_EXT_BIT);
        let expected = (one - wi) * rhi + reb * mask_32;
        builder.assert_zero_base((one - ld) * (rvd1 - expected));
    }

    // SltResZeroConstraint (7)
    {
        let sb = builder.main_base(cols::SLT) + builder.main_base(cols::BLT);
        for i in 1..8usize {
            builder.assert_zero_base(sb * builder.main_base(cols::RES[i]));
        }
    }

    // ExtBitZeroConstraint (3)
    {
        let nw = one - builder.main_base(cols::WORD_INSTR);
        builder.assert_zero_base(nw * builder.main_base(cols::RV1_EXT_BIT));
        builder.assert_zero_base(nw * builder.main_base(cols::RV2_EXT_BIT));
        builder.assert_zero_base(nw * builder.main_base(cols::RES_EXT_BIT));
    }

    // NextPcAddConstraint (2)
    {
        let plo = builder.main_base(cols::PC_0);
        let phi = builder.main_base(cols::PC_1);
        let nlo = builder.main_base(cols::NEXT_PC_0);
        let nhi = builder.main_base(cols::NEXT_PC_1);
        let ct = builder.main_base(cols::C_TYPE_INSTRUCTION);
        let bc = builder.main_base(cols::BRANCH_COND);
        let isz = FE::from(4u64) - two * ct;
        let nb = one - bc;
        let c0 = (plo + isz - nlo) * inv_2_32;
        builder.assert_zero_base(nb * c0 * (one - c0));
        let c1 = (phi + c0 - nhi) * inv_2_32;
        builder.assert_zero_base(nb * c1 * (one - c1));
    }
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
    .with_name("BITWISE")
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
    .with_name("LT")
}

/// Create SHIFT AIR with constraints and bus interactions.
pub fn create_shift_air(proof_options: &ProofOptions) -> VmAir {
    let (constraints, _) = shift_constraints(0);
    let transition_constraints: Vec<Box<dyn TransitionConstraint<F, E>>> =
        constraints.into_iter().map(|c| Box::new(c) as _).collect();

    let auxiliary_trace_build_data = AuxiliaryTraceBuildData {
        interactions: shift_bus_interactions(),
    };

    AirWithBuses::new(
        shift_cols::NUM_COLUMNS,
        auxiliary_trace_build_data,
        proof_options,
        1,
        transition_constraints,
    )
    .with_main_builder(|builder| {
        use crate::constraints::helpers::assert_is_bit_base;
        use crate::tables::shift::cols;
        use crate::tables::types::SHIFT_16;

        let one = FieldElement::<GoldilocksField>::one();
        let shift_16_val = FieldElement::<GoldilocksField>::from(SHIFT_16);

        let mu = builder.main_base(cols::MU);
        let direction = builder.main_base(cols::DIRECTION);
        let zbs = builder.main_base(cols::ZBS);
        let is_negative = builder.main_base(cols::IS_NEGATIVE);
        let left = mu - direction;

        // Constraint 0: DirectionImpliesMu
        builder.assert_zero_base(direction * (one - mu));

        // Constraints 1-4: ZbsOverrideX(0..4)
        for i in 0..4 {
            let x_i = builder.main_base(cols::X[i]);
            let in_i = builder.main_base(cols::IN[i]);
            builder.assert_zero_base(zbs * (x_i - in_i * left));
        }

        // Constraint 5: ZbsOverrideX4
        let x4 = builder.main_base(cols::X_4);
        builder.assert_zero_base(zbs * x4);

        // Constraints 6-9: ZbsOverrideY(0..4)
        for i in 0..4 {
            let y_i = builder.main_base(cols::Y[i]);
            let in_i = builder.main_base(cols::IN[i]);
            builder.assert_zero_base(zbs * (y_i - in_i * direction));
        }

        // Constraints 10-13: LimbShiftIsBit(0..4)
        for i in 0..3 {
            let ls = builder.main_base(cols::LIMB_SHIFT_RAW[i]);
            assert_is_bit_base(builder, ls);
        }
        let ls_raw_0 = builder.main_base(cols::LIMB_SHIFT_RAW[0]);
        let ls_raw_1 = builder.main_base(cols::LIMB_SHIFT_RAW[1]);
        let ls_raw_2 = builder.main_base(cols::LIMB_SHIFT_RAW[2]);
        let ls_3 = one - ls_raw_0 - ls_raw_1 - ls_raw_2;
        assert_is_bit_base(builder, ls_3);

        // Constraints 14-15: OutputMatchesShifted(0..2)
        let extension = FieldElement::<GoldilocksField>::from(65535u64) * is_negative;

        let x = [
            builder.main_base(cols::X[0]),
            builder.main_base(cols::X[1]),
            builder.main_base(cols::X[2]),
            builder.main_base(cols::X[3]),
            builder.main_base(cols::X_4),
        ];
        let y = [
            builder.main_base(cols::Y[0]),
            builder.main_base(cols::Y[1]),
            builder.main_base(cols::Y[2]),
            builder.main_base(cols::Y[3]),
        ];

        let ls = [ls_raw_0, ls_raw_1, ls_raw_2, ls_3];

        let mut shifted = Vec::with_capacity(4);
        for i in 0..4usize {
            let mut left_part = FieldElement::<GoldilocksField>::zero();
            for j in 0..=i {
                let k = i - j;
                let intra_left_k = if k == 0 { x[0] } else { x[k] + y[k - 1] };
                left_part += ls[j] * intra_left_k;
            }
            left_part = left * left_part;

            let mut right_shift_part = FieldElement::<GoldilocksField>::zero();
            for j in 0..=(3 - i) {
                let k = i + j;
                let intra_right_k = y[k] + x[k + 1];
                right_shift_part += ls[j] * intra_right_k;
            }

            let mut ext_sum = FieldElement::<GoldilocksField>::zero();
            for j in (4 - i)..4 {
                ext_sum += ls[j];
            }
            let right_ext_part = extension * ext_sum;

            let right_part = direction * (right_shift_part + right_ext_part);

            shifted.push(left_part + right_part);
        }

        // OutputMatchesShifted(0)
        let out_0 = builder.main_base(cols::OUT_0);
        builder.assert_zero_base(out_0 - shifted[0] - shifted[1] * shift_16_val);

        // OutputMatchesShifted(1)
        let out_1 = builder.main_base(cols::OUT_1);
        builder.assert_zero_base(out_1 - shifted[2] - shifted[3] * shift_16_val);
    })
    .with_builder(|builder| {
        use crate::constraints::helpers::assert_is_bit;
        use crate::tables::shift::cols;
        use crate::tables::types::{GoldilocksExtension, SHIFT_16};
        use math::field::element::FieldElement;

        let one = FieldElement::<GoldilocksExtension>::one();
        let shift_16_val = FieldElement::<GoldilocksExtension>::from(SHIFT_16);

        let mu = builder.main(0, cols::MU);
        let direction = builder.main(0, cols::DIRECTION);
        let zbs = builder.main(0, cols::ZBS);
        let is_negative = builder.main(0, cols::IS_NEGATIVE);
        let left = mu - direction;

        // Constraint 0: DirectionImpliesMu -- direction * (1 - mu) = 0
        builder.assert_zero(direction * (one - mu));

        // Constraints 1-4: ZbsOverrideX(0..4) -- zbs * (X[i] - in[i] * left) = 0
        for i in 0..4 {
            let x_i = builder.main(0, cols::X[i]);
            let in_i = builder.main(0, cols::IN[i]);
            builder.assert_zero(zbs * (x_i - in_i * left));
        }

        // Constraint 5: ZbsOverrideX4 -- zbs * X[4] = 0
        let x4 = builder.main(0, cols::X_4);
        builder.assert_zero(zbs * x4);

        // Constraints 6-9: ZbsOverrideY(0..4) -- zbs * (Y[i] - in[i] * right) = 0
        for i in 0..4 {
            let y_i = builder.main(0, cols::Y[i]);
            let in_i = builder.main(0, cols::IN[i]);
            builder.assert_zero(zbs * (y_i - in_i * direction));
        }

        // Constraints 10-13: LimbShiftIsBit(0..4)
        for i in 0..3 {
            let ls = builder.main(0, cols::LIMB_SHIFT_RAW[i]);
            assert_is_bit(builder, ls);
        }
        // limb_shift[3] is virtual: 1 - ls_raw[0] - ls_raw[1] - ls_raw[2]
        let ls_raw_0 = builder.main(0, cols::LIMB_SHIFT_RAW[0]);
        let ls_raw_1 = builder.main(0, cols::LIMB_SHIFT_RAW[1]);
        let ls_raw_2 = builder.main(0, cols::LIMB_SHIFT_RAW[2]);
        let ls_3 = one - ls_raw_0 - ls_raw_1 - ls_raw_2;
        assert_is_bit(builder, ls_3);

        // Constraints 14-15: OutputMatchesShifted(0..2)
        // Extension = 65535 * is_negative
        let extension = FieldElement::<GoldilocksExtension>::from(65535u64) * is_negative;

        let x = [
            builder.main(0, cols::X[0]),
            builder.main(0, cols::X[1]),
            builder.main(0, cols::X[2]),
            builder.main(0, cols::X[3]),
            builder.main(0, cols::X[4]),
        ];
        let y = [
            builder.main(0, cols::Y[0]),
            builder.main(0, cols::Y[1]),
            builder.main(0, cols::Y[2]),
            builder.main(0, cols::Y[3]),
        ];

        let ls = [ls_raw_0, ls_raw_1, ls_raw_2, ls_3];

        let mut shifted = Vec::with_capacity(4);
        for i in 0..4usize {
            let mut left_part = FieldElement::<GoldilocksExtension>::zero();
            for j in 0..=i {
                let k = i - j;
                let intra_left_k = if k == 0 { x[0] } else { x[k] + y[k - 1] };
                left_part += ls[j] * intra_left_k;
            }
            left_part = left * left_part;

            let mut right_shift_part = FieldElement::<GoldilocksExtension>::zero();
            for j in 0..=(3 - i) {
                let k = i + j;
                let intra_right_k = y[k] + x[k + 1];
                right_shift_part += ls[j] * intra_right_k;
            }

            let mut ext_sum = FieldElement::<GoldilocksExtension>::zero();
            for j in (4 - i)..4 {
                ext_sum += ls[j];
            }
            let right_ext_part = extension * ext_sum;

            let right_part = direction * (right_shift_part + right_ext_part);

            shifted.push(left_part + right_part);
        }

        // OutputMatchesShifted(0): out[0] - (shifted[0] + shifted[1] * 2^16) = 0
        let out_0 = builder.main(0, cols::OUT_0);
        builder.assert_zero(out_0 - shifted[0] - shifted[1] * shift_16_val);

        // OutputMatchesShifted(1): out[1] - (shifted[2] + shifted[3] * 2^16) = 0
        let out_1 = builder.main(0, cols::OUT_1);
        builder.assert_zero(out_1 - shifted[2] - shifted[3] * shift_16_val);
    })
    .with_name("SHIFT")
}

/// Create MEMW AIR with constraints and bus interactions.
pub fn create_memw_air(proof_options: &ProofOptions) -> VmAir {
    let transition_constraints = memw_constraints();

    let auxiliary_trace_build_data = AuxiliaryTraceBuildData {
        interactions: memw_bus_interactions(),
    };

    AirWithBuses::new(
        memw_cols::NUM_COLUMNS,
        auxiliary_trace_build_data,
        proof_options,
        1,
        transition_constraints,
    )
    .with_main_builder(|builder| {
        use crate::constraints::helpers::assert_is_bit_base;
        use crate::tables::memw::cols;

        let one = FieldElement::<GoldilocksField>::one();

        let mu_read = builder.main_base(cols::MU_READ);
        let mu_write = builder.main_base(cols::MU_WRITE);
        let mu_sum = mu_read + mu_write;

        // Constraint 0: IS_BIT<mu_sum>
        builder.assert_zero_base(mu_sum * (one - mu_sum));

        // Constraint 1: w2 => mu_sum
        let write2 = builder.main_base(cols::WRITE2);
        let write4 = builder.main_base(cols::WRITE4);
        let write8 = builder.main_base(cols::WRITE8);
        let w2 = write2 + write4 + write8;
        builder.assert_zero_base(w2 * (one - mu_sum));

        // Constraint 2: IS_BIT<mu_read>
        assert_is_bit_base(builder, mu_read);

        // Constraint 3: IS_BIT<mu_write>
        assert_is_bit_base(builder, mu_write);

        // Constraints 4-10: IS_BIT for carry[0..6]
        for &col in &cols::CARRY {
            let carry = builder.main_base(col);
            assert_is_bit_base(builder, carry);
        }
    })
    .with_builder(|builder| {
        use crate::constraints::helpers::assert_is_bit;
        use crate::tables::memw::cols;
        use crate::tables::types::GoldilocksExtension;
        use math::field::element::FieldElement;

        let one = FieldElement::<GoldilocksExtension>::one();

        let mu_read = builder.main(0, cols::MU_READ);
        let mu_write = builder.main(0, cols::MU_WRITE);
        let mu_sum = mu_read + mu_write;

        // Constraint 0: IS_BIT<μ_sum> — mu_sum * (1 - mu_sum) = 0
        builder.assert_zero(mu_sum * (one - mu_sum));

        // Constraint 1: w2 => μ_sum — (write2 + write4 + write8) * (1 - mu_sum) = 0
        let write2 = builder.main(0, cols::WRITE2);
        let write4 = builder.main(0, cols::WRITE4);
        let write8 = builder.main(0, cols::WRITE8);
        let w2 = write2 + write4 + write8;
        builder.assert_zero(w2 * (one - mu_sum));

        // Constraint 2: IS_BIT<μ_read>
        assert_is_bit(builder, mu_read);

        // Constraint 3: IS_BIT<μ_write>
        assert_is_bit(builder, mu_write);

        // Constraints 4-10: IS_BIT for carry[0..6]
        for &col in &cols::CARRY {
            let carry = builder.main(0, col);
            assert_is_bit(builder, carry);
        }
    })
    .with_name("MEMW")
}

/// Create MEMW_A (aligned) AIR with constraints and bus interactions.
pub fn create_memw_aligned_air(proof_options: &ProofOptions) -> VmAir {
    let transition_constraints = memw_aligned_constraints();

    let auxiliary_trace_build_data = AuxiliaryTraceBuildData {
        interactions: memw_aligned_bus_interactions(),
    };

    AirWithBuses::new(
        memw_aligned_cols::NUM_COLUMNS,
        auxiliary_trace_build_data,
        proof_options,
        1,
        transition_constraints,
    )
    .with_main_builder(|builder| {
        use crate::constraints::helpers::assert_is_bit_base;

        let mu_read = builder.main_base(memw_aligned_cols::MU_READ);
        let mu_write = builder.main_base(memw_aligned_cols::MU_WRITE);
        let mu_sum = mu_read + mu_write;

        // Constraint 0: IS_BIT<mu_sum>
        builder.assert_zero_base(mu_sum * (FieldElement::<GoldilocksField>::one() - mu_sum));

        // Constraint 1: w2 => mu_sum
        let write2 = builder.main_base(memw_aligned_cols::WRITE2);
        let write4 = builder.main_base(memw_aligned_cols::WRITE4);
        let write8 = builder.main_base(memw_aligned_cols::WRITE8);
        let w2 = write2 + write4 + write8;
        builder.assert_zero_base(w2 * (FieldElement::<GoldilocksField>::one() - mu_sum));

        // Constraint 2: IS_BIT<mu_read>
        assert_is_bit_base(builder, mu_read);

        // Constraint 3: IS_BIT<mu_write>
        assert_is_bit_base(builder, mu_write);
    })
    .with_builder(|builder| {
        use crate::constraints::helpers::assert_is_bit;

        let mu_read = builder.main(0, memw_aligned_cols::MU_READ);
        let mu_write = builder.main(0, memw_aligned_cols::MU_WRITE);
        let mu_sum = mu_read + mu_write;

        // Constraint 0: IS_BIT<mu_sum>: mu_sum * (1 - mu_sum) == 0
        builder.assert_zero(
            mu_sum
                * (math::field::element::FieldElement::<
                    crate::tables::types::GoldilocksExtension,
                >::one()
                    - mu_sum),
        );

        // Constraint 1: w2 => mu_sum: (write2 + write4 + write8) * (1 - mu_sum) == 0
        let write2 = builder.main(0, memw_aligned_cols::WRITE2);
        let write4 = builder.main(0, memw_aligned_cols::WRITE4);
        let write8 = builder.main(0, memw_aligned_cols::WRITE8);
        let w2 = write2 + write4 + write8;
        builder.assert_zero(
            w2 * (math::field::element::FieldElement::<
                crate::tables::types::GoldilocksExtension,
            >::one()
                - mu_sum),
        );

        // Constraint 2: IS_BIT<mu_read>: mu_read * (1 - mu_read) == 0
        assert_is_bit(builder, mu_read);

        // Constraint 3: IS_BIT<mu_write>: mu_write * (1 - mu_write) == 0
        assert_is_bit(builder, mu_write);
    })
    .with_name("MEMW_A")
}

/// Create MEMW_R (register) AIR with constraints and bus interactions.
pub fn create_memw_register_air(proof_options: &ProofOptions) -> VmAir {
    let transition_constraints = memw_register_constraints();

    let auxiliary_trace_build_data = AuxiliaryTraceBuildData {
        interactions: memw_register_bus_interactions(),
    };

    AirWithBuses::new(
        memw_register_cols::NUM_COLUMNS,
        auxiliary_trace_build_data,
        proof_options,
        1,
        transition_constraints,
    )
    .with_main_builder(|builder| {
        use crate::constraints::helpers::assert_is_bit_base;

        let mu_read = builder.main_base(memw_register_cols::MU_READ);
        let mu_write = builder.main_base(memw_register_cols::MU_WRITE);

        // Constraint 0: IS_BIT(mu_read)
        assert_is_bit_base(builder, mu_read);
        // Constraint 1: IS_BIT(mu_write)
        assert_is_bit_base(builder, mu_write);
        // Constraint 2: (mu_read + mu_write) * (1 - mu_read - mu_write) == 0
        let mu_sum = mu_read + mu_write;
        builder.assert_zero_base(
            mu_sum
                * (FieldElement::<GoldilocksField>::one() - mu_sum),
        );
    })
    .with_builder(|builder| {
        use crate::constraints::helpers::assert_is_bit;

        let mu_read = builder.main(0, memw_register_cols::MU_READ);
        let mu_write = builder.main(0, memw_register_cols::MU_WRITE);

        // Constraint 0: IS_BIT(mu_read)
        assert_is_bit(builder, mu_read);
        // Constraint 1: IS_BIT(mu_write)
        assert_is_bit(builder, mu_write);
        // Constraint 2: (mu_read + mu_write) * (1 - mu_read - mu_write) == 0
        let mu_sum = mu_read + mu_write;
        builder.assert_zero(
            mu_sum
                * (math::field::element::FieldElement::<crate::tables::types::GoldilocksExtension>::one() - mu_sum),
        );
    })
    .with_name("MEMW_R")
}

/// Create LOAD AIR with constraints and bus interactions.
pub fn create_load_air(proof_options: &ProofOptions) -> VmAir {
    let transition_constraints = load_constraints();

    let auxiliary_trace_build_data = AuxiliaryTraceBuildData {
        interactions: load_bus_interactions(),
    };

    AirWithBuses::new(
        load_cols::NUM_COLUMNS,
        auxiliary_trace_build_data,
        proof_options,
        1,
        transition_constraints,
    )
    .with_main_builder(|builder| {
        use crate::tables::load::cols;

        let one = FieldElement::<GoldilocksField>::one();
        let ff = FieldElement::<GoldilocksField>::from(255u64);

        let mu = builder.main_base(cols::MU);
        let read2 = builder.main_base(cols::READ2);
        let read4 = builder.main_base(cols::READ4);
        let read8 = builder.main_base(cols::READ8);
        let signed = builder.main_base(cols::SIGNED);
        let sign_bit = builder.main_base(cols::SIGN_BIT);

        // Constraint 0: ReadImpliesMu
        let read_sum = read2 + read4 + read8;
        builder.assert_zero_base(read_sum * (one - mu));

        let expected = signed * sign_bit * ff;

        // Constraints 1-4: ExtensionHigh(4..8)
        let not_read8 = one - read8;
        for i in 4..8 {
            let res_i = builder.main_base(cols::RES[i]);
            builder.assert_zero_base(not_read8 * (res_i - expected));
        }

        // Constraints 5-6: ExtensionMid(2..4)
        let not_read4_8 = one - read4 - read8;
        for i in 2..4 {
            let res_i = builder.main_base(cols::RES[i]);
            builder.assert_zero_base(not_read4_8 * (res_i - expected));
        }

        // Constraint 7: ExtensionLow
        let not_read2_4_8 = one - read2 - read4 - read8;
        let res_1 = builder.main_base(cols::RES[1]);
        builder.assert_zero_base(not_read2_4_8 * (res_1 - expected));
    })
    .with_builder(|builder| {
        use crate::tables::load::cols;
        use crate::tables::types::GoldilocksExtension;
        use math::field::element::FieldElement;

        let one = FieldElement::<GoldilocksExtension>::one();
        let ff = FieldElement::<GoldilocksExtension>::from(255u64);

        let mu = builder.main(0, cols::MU);
        let read2 = builder.main(0, cols::READ2);
        let read4 = builder.main(0, cols::READ4);
        let read8 = builder.main(0, cols::READ8);
        let signed = builder.main(0, cols::SIGNED);
        let sign_bit = builder.main(0, cols::SIGN_BIT);

        // Constraint 0: ReadImpliesMu — (read2 + read4 + read8) * (1 - μ) = 0
        let read_sum = read2 + read4 + read8;
        builder.assert_zero(read_sum * (one - mu));

        // Expected extension value
        let expected = signed * sign_bit * ff;

        // Constraints 1-4: ExtensionHigh(4..8) — (1 - read8) * (res[i] - expected) = 0
        let not_read8 = one - read8;
        for i in 4..8 {
            let res_i = builder.main(0, cols::RES[i]);
            builder.assert_zero(not_read8 * (res_i - expected));
        }

        // Constraints 5-6: ExtensionMid(2..4) — (1 - read4 - read8) * (res[i] - expected) = 0
        let not_read4_8 = one - read4 - read8;
        for i in 2..4 {
            let res_i = builder.main(0, cols::RES[i]);
            builder.assert_zero(not_read4_8 * (res_i - expected));
        }

        // Constraint 7: ExtensionLow — (1 - read2 - read4 - read8) * (res[1] - expected) = 0
        let not_read2_4_8 = one - read2 - read4 - read8;
        let res_1 = builder.main(0, cols::RES[1]);
        builder.assert_zero(not_read2_4_8 * (res_1 - expected));
    })
    .with_name("LOAD")
}

/// Create DECODE AIR with bus interactions.
///
/// The DECODE table has no transition constraints (it's a pure lookup table).
/// It receives lookups from the CPU table via the DECODE bus.
///
/// For production use with preprocessed verification, chain with `.with_preprocessed()`:
/// ```ignore
/// let decode_air = create_decode_air(&opts)
///     .with_preprocessed(
///         decode::compute_precomputed_commitment(&instructions, &opts),
///         decode::NUM_PRECOMPUTED_COLS,
///     );
/// ```
pub fn create_decode_air(proof_options: &ProofOptions) -> VmAir {
    let transition_constraints: Vec<Box<dyn TransitionConstraint<F, E>>> = vec![];

    let auxiliary_trace_build_data = AuxiliaryTraceBuildData {
        interactions: decode_bus_interactions(),
    };

    AirWithBuses::new(
        decode_cols::NUM_COLUMNS,
        auxiliary_trace_build_data,
        proof_options,
        1,
        transition_constraints,
    )
    .with_name("DECODE")
}

/// Create MUL AIR with bus interactions.
pub fn create_mul_air(proof_options: &ProofOptions) -> VmAir {
    let transition_constraints: Vec<Box<dyn TransitionConstraint<F, E>>> = vec![];

    let auxiliary_trace_build_data = AuxiliaryTraceBuildData {
        interactions: mul_bus_interactions(),
    };

    AirWithBuses::new(
        mul_cols::NUM_COLUMNS,
        auxiliary_trace_build_data,
        proof_options,
        1,
        transition_constraints,
    )
    .with_main_builder(|builder| {
        use crate::tables::mul::cols;

        let one = FieldElement::<GoldilocksField>::one();
        let sign_fill = FieldElement::<GoldilocksField>::from(0xFFFFu64);
        let shift_16 = FieldElement::<GoldilocksField>::from(65536u64);

        let lhs_signed = builder.main_base(cols::LHS_SIGNED);
        let rhs_signed = builder.main_base(cols::RHS_SIGNED);
        let lhs_is_neg = builder.main_base(cols::LHS_IS_NEGATIVE);
        let rhs_is_neg = builder.main_base(cols::RHS_IS_NEGATIVE);

        // Constraint 0: LhsSign
        builder.assert_zero_base((one - lhs_signed) * lhs_is_neg);

        // Constraint 1: RhsSign
        builder.assert_zero_base((one - rhs_signed) * rhs_is_neg);

        let lhs = [
            builder.main_base(cols::LHS_0),
            builder.main_base(cols::LHS_1),
            builder.main_base(cols::LHS_2),
            builder.main_base(cols::LHS_3),
        ];
        let rhs = [
            builder.main_base(cols::RHS_0),
            builder.main_base(cols::RHS_1),
            builder.main_base(cols::RHS_2),
            builder.main_base(cols::RHS_3),
        ];

        let lhs_sign_ext = sign_fill * lhs_is_neg;
        let lhs_ext = [
            lhs[0],
            lhs[1],
            lhs[2],
            lhs[3],
            lhs_sign_ext,
            lhs_sign_ext,
            lhs_sign_ext,
            lhs_sign_ext,
        ];

        let rhs_sign_ext = sign_fill * rhs_is_neg;
        let rhs_ext = [
            rhs[0],
            rhs[1],
            rhs[2],
            rhs[3],
            rhs_sign_ext,
            rhs_sign_ext,
            rhs_sign_ext,
            rhs_sign_ext,
        ];

        let raw_product_cols = [
            cols::RAW_PRODUCT_0,
            cols::RAW_PRODUCT_1,
            cols::RAW_PRODUCT_2,
            cols::RAW_PRODUCT_3,
        ];

        // Constraints 2-5: RawProduct(i) for i in 0..4
        for i in 0..4 {
            let mut sum = FieldElement::<GoldilocksField>::zero();

            for k in 0usize..=1 {
                let idx = 2 * i + k;
                if idx < 8 {
                    let mut inner_sum = FieldElement::<GoldilocksField>::zero();
                    for j in 0..=idx {
                        if j < 8 && (idx - j) < 8 {
                            inner_sum = inner_sum + (lhs_ext[j] * rhs_ext[idx - j]);
                        }
                    }
                    if k == 0 {
                        sum += inner_sum;
                    } else {
                        sum = sum + (inner_sum * shift_16);
                    }
                }
            }

            let raw_product = builder.main_base(raw_product_cols[i]);
            builder.assert_zero_base(raw_product - sum);
        }
    })
    .with_builder(|builder| {
        use crate::tables::mul::cols;
        use crate::tables::types::GoldilocksExtension;
        use math::field::element::FieldElement;

        let one = FieldElement::<GoldilocksExtension>::one();
        let sign_fill = FieldElement::<GoldilocksExtension>::from(0xFFFFu64);
        let shift_16 = FieldElement::<GoldilocksExtension>::from(65536u64);

        let lhs_signed = builder.main(0, cols::LHS_SIGNED);
        let rhs_signed = builder.main(0, cols::RHS_SIGNED);
        let lhs_is_neg = builder.main(0, cols::LHS_IS_NEGATIVE);
        let rhs_is_neg = builder.main(0, cols::RHS_IS_NEGATIVE);

        // Constraint 0: LhsSign -- (1 - lhs_signed) * lhs_is_negative == 0
        builder.assert_zero((one - lhs_signed) * lhs_is_neg);

        // Constraint 1: RhsSign -- (1 - rhs_signed) * rhs_is_negative == 0
        builder.assert_zero((one - rhs_signed) * rhs_is_neg);

        // Build lhs_ext[0..8] and rhs_ext[0..8]
        let lhs = [
            builder.main(0, cols::LHS_0),
            builder.main(0, cols::LHS_1),
            builder.main(0, cols::LHS_2),
            builder.main(0, cols::LHS_3),
        ];
        let rhs = [
            builder.main(0, cols::RHS_0),
            builder.main(0, cols::RHS_1),
            builder.main(0, cols::RHS_2),
            builder.main(0, cols::RHS_3),
        ];

        // lhs_ext[0..4] = lhs halfwords, lhs_ext[4..8] = SIGN_FILL * lhs_is_negative
        let lhs_sign_ext = sign_fill * lhs_is_neg;
        let lhs_ext = [
            lhs[0],
            lhs[1],
            lhs[2],
            lhs[3],
            lhs_sign_ext,
            lhs_sign_ext,
            lhs_sign_ext,
            lhs_sign_ext,
        ];

        // rhs_ext[0..4] = rhs halfwords, rhs_ext[4..8] = SIGN_FILL * rhs_is_negative
        let rhs_sign_ext = sign_fill * rhs_is_neg;
        let rhs_ext = [
            rhs[0],
            rhs[1],
            rhs[2],
            rhs[3],
            rhs_sign_ext,
            rhs_sign_ext,
            rhs_sign_ext,
            rhs_sign_ext,
        ];

        let raw_product_cols = [
            cols::RAW_PRODUCT_0,
            cols::RAW_PRODUCT_1,
            cols::RAW_PRODUCT_2,
            cols::RAW_PRODUCT_3,
        ];

        // Constraints 2-5: RawProduct(i) for i in 0..4
        // raw_product[i] = sum_{k=0}^{1} 2^(16k) * sum_{j=0}^{2i+k} lhs_ext[j] * rhs_ext[2i+k-j]
        for i in 0..4 {
            let mut sum = FieldElement::<GoldilocksExtension>::zero();

            for k in 0usize..=1 {
                let idx = 2 * i + k;
                if idx < 8 {
                    let mut inner_sum = FieldElement::<GoldilocksExtension>::zero();
                    for j in 0..=idx {
                        if j < 8 && (idx - j) < 8 {
                            inner_sum = inner_sum + (lhs_ext[j] * rhs_ext[idx - j]);
                        }
                    }
                    if k == 0 {
                        sum += inner_sum;
                    } else {
                        sum = sum + (inner_sum * shift_16);
                    }
                }
            }

            let raw_product = builder.main(0, raw_product_cols[i]);
            builder.assert_zero(raw_product - sum);
        }
    })
    .with_name("MUL")
}

/// Create DVRM AIR with constraints and bus interactions.
pub fn create_dvrm_air(proof_options: &ProofOptions) -> VmAir {
    let (constraints, _) = dvrm_constraints(0);
    let transition_constraints: Vec<Box<dyn TransitionConstraint<F, E>>> =
        constraints.into_iter().map(|c| Box::new(c) as _).collect();

    let auxiliary_trace_build_data = AuxiliaryTraceBuildData {
        interactions: dvrm_bus_interactions(),
    };

    AirWithBuses::new(
        dvrm_cols::NUM_COLUMNS,
        auxiliary_trace_build_data,
        proof_options,
        1,
        transition_constraints,
    )
    .with_main_builder(|builder| {
        use crate::constraints::helpers::assert_is_bit_base;
        use crate::constraints::templates::INV_SHIFT_32;
        use crate::tables::dvrm::cols;
        use crate::tables::types::SHIFT_16;

        let one = FieldElement::<GoldilocksField>::one();
        let shift_16 = FieldElement::<GoldilocksField>::from(SHIFT_16);
        let inv_2_32 = FieldElement::<GoldilocksField>::from(INV_SHIFT_32);
        let sign_fill = FieldElement::<GoldilocksField>::from(0xFFFFu64);

        let signed = builder.main_base(cols::SIGNED);
        let sign_n = builder.main_base(cols::SIGN_N);
        let sign_d = builder.main_base(cols::SIGN_D);
        let sign_r = builder.main_base(cols::SIGN_R);
        let sign_q = builder.main_base(cols::SIGN_Q);
        let sign_nsr = builder.main_base(cols::SIGN_N_SUB_R);
        let overflow = builder.main_base(cols::OVERFLOW);
        let div_by_zero = builder.main_base(cols::DIV_BY_ZERO);

        let r0 = builder.main_base(cols::R_0);
        let r1 = builder.main_base(cols::R_1);
        let r2 = builder.main_base(cols::R_2);
        let r3 = builder.main_base(cols::R_3);

        let d0 = builder.main_base(cols::D_0);
        let d1 = builder.main_base(cols::D_1);
        let d2 = builder.main_base(cols::D_2);
        let d3 = builder.main_base(cols::D_3);

        let n0 = builder.main_base(cols::N_0);
        let n1 = builder.main_base(cols::N_1);
        let n2 = builder.main_base(cols::N_2);
        let n3 = builder.main_base(cols::N_3);

        let q0 = builder.main_base(cols::Q_0);
        let q1 = builder.main_base(cols::Q_1);
        let q2 = builder.main_base(cols::Q_2);
        let q3 = builder.main_base(cols::Q_3);

        let nsr0 = builder.main_base(cols::N_SUB_R_0);
        let nsr1 = builder.main_base(cols::N_SUB_R_1);
        let nsr2 = builder.main_base(cols::N_SUB_R_2);
        let nsr3 = builder.main_base(cols::N_SUB_R_3);

        let abs_r0 = builder.main_base(cols::ABS_R_0);
        let abs_r1 = builder.main_base(cols::ABS_R_1);
        let abs_d0 = builder.main_base(cols::ABS_D_0);
        let abs_d1 = builder.main_base(cols::ABS_D_1);

        // Constraint 0: DVRM-A3
        assert_is_bit_base(builder, signed);

        // Constraint 1: DVRM-C1
        let r_sum = r0 + r1 + r2 + r3;
        builder.assert_zero_base(r_sum * (sign_r - sign_n));

        // Constraint 2: DVRM-C4.0
        let r_wl_0 = r0 + r1 * shift_16;
        builder.assert_zero_base((one - sign_r) * (abs_r0 - r_wl_0));

        // Constraint 3: DVRM-C4.1
        let r_wl_1 = r2 + r3 * shift_16;
        builder.assert_zero_base((one - sign_r) * (abs_r1 - r_wl_1));

        // Constraint 4: DVRM-C6.0
        let d_wl_0 = d0 + d1 * shift_16;
        builder.assert_zero_base((one - sign_d) * (abs_d0 - d_wl_0));

        // Constraint 5: DVRM-C6.1
        let d_wl_1 = d2 + d3 * shift_16;
        builder.assert_zero_base((one - sign_d) * (abs_d1 - d_wl_1));

        // Constraint 6: DVRM-C7
        builder.assert_zero_base(signed * (one - overflow) - sign_q);

        // Constraints 7-10: carries
        let sign_fill_word = sign_fill + sign_fill * shift_16;

        let ext_n = [
            n0 + n1 * shift_16,
            n2 + n3 * shift_16,
            sign_n * sign_fill_word,
            sign_n * sign_fill_word,
        ];

        let ext_r = [
            r_wl_0,
            r_wl_1,
            sign_r * sign_fill_word,
            sign_r * sign_fill_word,
        ];

        let ext_nsr = [
            nsr0 + nsr1 * shift_16,
            nsr2 + nsr3 * shift_16,
            sign_nsr * sign_fill_word,
            sign_nsr * sign_fill_word,
        ];

        let carry_0 = (ext_nsr[0] + ext_r[0] - ext_n[0]) * inv_2_32;
        builder.assert_zero_base(carry_0 * (one - carry_0));

        let carry_1 = (ext_nsr[1] + ext_r[1] + carry_0 - ext_n[1]) * inv_2_32;
        builder.assert_zero_base(carry_1 * (one - carry_1));

        let carry_2 = (ext_nsr[2] + ext_r[2] + carry_1 - ext_n[2]) * inv_2_32;
        builder.assert_zero_base(carry_2 * (one - carry_2));

        let carry_3 = (ext_nsr[3] + ext_r[3] + carry_2 - ext_n[3]) * inv_2_32;
        builder.assert_zero_base(carry_3 * (one - carry_3));

        // Constraint 11: DVRM-C15
        assert_is_bit_base(builder, sign_nsr);

        // Constraint 12: DVRM-C18b
        builder.assert_zero_base((one - signed) * sign_n);

        // Constraint 13: DVRM-C19b
        builder.assert_zero_base((one - signed) * sign_r);

        // Constraint 14: DVRM-C20b
        builder.assert_zero_base((one - signed) * sign_d);

        // Constraints 15-18: DVRM-C16.i
        let q_cols = [q0, q1, q2, q3];
        for q_val in &q_cols {
            builder.assert_zero_base(div_by_zero * (q_val - sign_fill));
        }
    })
    .with_builder(|builder| {
        use crate::constraints::helpers::assert_is_bit;
        use crate::constraints::templates::INV_SHIFT_32;
        use crate::tables::dvrm::cols;
        use crate::tables::types::{GoldilocksExtension, SHIFT_16};
        use math::field::element::FieldElement;

        let one = FieldElement::<GoldilocksExtension>::one();
        let shift_16 = FieldElement::<GoldilocksExtension>::from(SHIFT_16);
        let inv_2_32 = FieldElement::<GoldilocksExtension>::from(INV_SHIFT_32);
        let sign_fill = FieldElement::<GoldilocksExtension>::from(0xFFFFu64);

        // Load all columns used across constraints
        let signed = builder.main(0, cols::SIGNED);
        let sign_n = builder.main(0, cols::SIGN_N);
        let sign_d = builder.main(0, cols::SIGN_D);
        let sign_r = builder.main(0, cols::SIGN_R);
        let sign_q = builder.main(0, cols::SIGN_Q);
        let sign_nsr = builder.main(0, cols::SIGN_N_SUB_R);
        let overflow = builder.main(0, cols::OVERFLOW);
        let div_by_zero = builder.main(0, cols::DIV_BY_ZERO);

        let r0 = builder.main(0, cols::R_0);
        let r1 = builder.main(0, cols::R_1);
        let r2 = builder.main(0, cols::R_2);
        let r3 = builder.main(0, cols::R_3);

        let d0 = builder.main(0, cols::D_0);
        let d1 = builder.main(0, cols::D_1);
        let d2 = builder.main(0, cols::D_2);
        let d3 = builder.main(0, cols::D_3);

        let n0 = builder.main(0, cols::N_0);
        let n1 = builder.main(0, cols::N_1);
        let n2 = builder.main(0, cols::N_2);
        let n3 = builder.main(0, cols::N_3);

        let q0 = builder.main(0, cols::Q_0);
        let q1 = builder.main(0, cols::Q_1);
        let q2 = builder.main(0, cols::Q_2);
        let q3 = builder.main(0, cols::Q_3);

        let nsr0 = builder.main(0, cols::N_SUB_R_0);
        let nsr1 = builder.main(0, cols::N_SUB_R_1);
        let nsr2 = builder.main(0, cols::N_SUB_R_2);
        let nsr3 = builder.main(0, cols::N_SUB_R_3);

        let abs_r0 = builder.main(0, cols::ABS_R_0);
        let abs_r1 = builder.main(0, cols::ABS_R_1);
        let abs_d0 = builder.main(0, cols::ABS_D_0);
        let abs_d1 = builder.main(0, cols::ABS_D_1);

        // Constraint 0: DVRM-A3: signed * (1 - signed) = 0
        assert_is_bit(builder, signed);

        // Constraint 1: DVRM-C1: (r[0]+r[1]+r[2]+r[3]) * (sign_r - sign_n) = 0
        let r_sum = r0 + r1 + r2 + r3;
        builder.assert_zero(r_sum * (sign_r - sign_n));

        // Constraint 2: DVRM-C4.0: (1-sign_r) * (abs_r[0] - r::DWordWL[0]) = 0
        // r::DWordWL[0] = r[0] + r[1]*2^16
        let r_wl_0 = r0 + r1 * shift_16;
        builder.assert_zero((one - sign_r) * (abs_r0 - r_wl_0));

        // Constraint 3: DVRM-C4.1: (1-sign_r) * (abs_r[1] - r::DWordWL[1]) = 0
        // r::DWordWL[1] = r[2] + r[3]*2^16
        let r_wl_1 = r2 + r3 * shift_16;
        builder.assert_zero((one - sign_r) * (abs_r1 - r_wl_1));

        // Constraint 4: DVRM-C6.0: (1-sign_d) * (abs_d[0] - d::DWordWL[0]) = 0
        let d_wl_0 = d0 + d1 * shift_16;
        builder.assert_zero((one - sign_d) * (abs_d0 - d_wl_0));

        // Constraint 5: DVRM-C6.1: (1-sign_d) * (abs_d[1] - d::DWordWL[1]) = 0
        let d_wl_1 = d2 + d3 * shift_16;
        builder.assert_zero((one - sign_d) * (abs_d1 - d_wl_1));

        // Constraint 6: DVRM-C7: signed * (1-overflow) - sign_q = 0
        builder.assert_zero(signed * (one - overflow) - sign_q);

        // Constraints 7-10: DVRM-C12.i: carry[i] * (1 - carry[i]) = 0
        // Build sign-extended QuadWL representations for n, r, n_sub_r
        let sign_fill_word = sign_fill + sign_fill * shift_16;

        // ext_n: [n0+n1*2^16, n2+n3*2^16, sign_n*0xFFFFFFFF, sign_n*0xFFFFFFFF]
        let ext_n = [
            n0 + n1 * shift_16,
            n2 + n3 * shift_16,
            sign_n * sign_fill_word,
            sign_n * sign_fill_word,
        ];

        // ext_r: [r0+r1*2^16, r2+r3*2^16, sign_r*0xFFFFFFFF, sign_r*0xFFFFFFFF]
        let ext_r = [
            r_wl_0,
            r_wl_1,
            sign_r * sign_fill_word,
            sign_r * sign_fill_word,
        ];

        // ext_nsr: [nsr0+nsr1*2^16, nsr2+nsr3*2^16, sign_nsr*0xFFFFFFFF, sign_nsr*0xFFFFFFFF]
        let ext_nsr = [
            nsr0 + nsr1 * shift_16,
            nsr2 + nsr3 * shift_16,
            sign_nsr * sign_fill_word,
            sign_nsr * sign_fill_word,
        ];

        // carry[0] = (ext_nsr[0] + ext_r[0] - ext_n[0]) * inv_2_32
        let carry_0 = (ext_nsr[0] + ext_r[0] - ext_n[0]) * inv_2_32;
        // Constraint 7: carry[0] * (1 - carry[0]) = 0
        builder.assert_zero(carry_0 * (one - carry_0));

        // carry[1] = (ext_nsr[1] + ext_r[1] + carry[0] - ext_n[1]) * inv_2_32
        let carry_1 = (ext_nsr[1] + ext_r[1] + carry_0 - ext_n[1]) * inv_2_32;
        // Constraint 8: carry[1] * (1 - carry[1]) = 0
        builder.assert_zero(carry_1 * (one - carry_1));

        // carry[2] = (ext_nsr[2] + ext_r[2] + carry[1] - ext_n[2]) * inv_2_32
        let carry_2 = (ext_nsr[2] + ext_r[2] + carry_1 - ext_n[2]) * inv_2_32;
        // Constraint 9: carry[2] * (1 - carry[2]) = 0
        builder.assert_zero(carry_2 * (one - carry_2));

        // carry[3] = (ext_nsr[3] + ext_r[3] + carry[2] - ext_n[3]) * inv_2_32
        let carry_3 = (ext_nsr[3] + ext_r[3] + carry_2 - ext_n[3]) * inv_2_32;
        // Constraint 10: carry[3] * (1 - carry[3]) = 0
        builder.assert_zero(carry_3 * (one - carry_3));

        // Constraint 11: DVRM-C15: sign_n_sub_r * (1 - sign_n_sub_r) = 0
        assert_is_bit(builder, sign_nsr);

        // Constraint 12: DVRM-C18b: (1-signed) * sign_n = 0
        builder.assert_zero((one - signed) * sign_n);

        // Constraint 13: DVRM-C19b: (1-signed) * sign_r = 0
        builder.assert_zero((one - signed) * sign_r);

        // Constraint 14: DVRM-C20b: (1-signed) * sign_d = 0
        builder.assert_zero((one - signed) * sign_d);

        // Constraints 15-18: DVRM-C16.i: div_by_zero * (q[i] - 65535) = 0
        let q_cols = [q0, q1, q2, q3];
        for q_val in &q_cols {
            builder.assert_zero(div_by_zero * (q_val - sign_fill));
        }
    })
    .with_name("DVRM")
}

/// Create BRANCH AIR with constraints and bus interactions.
///
/// The BRANCH table computes next_pc for branch/jump instructions:
/// - For branches (BEQ, BLT, JAL): next_pc = pc + sign_extend(offset)
/// - For JALR: next_pc = (register + sign_extend(offset)) & ~1
pub fn create_branch_air(proof_options: &ProofOptions) -> VmAir {
    let (constraints, _) = branch_constraints(0);
    let transition_constraints: Vec<Box<dyn TransitionConstraint<F, E>>> =
        constraints.into_iter().map(|c| Box::new(c) as _).collect();

    let auxiliary_trace_build_data = AuxiliaryTraceBuildData {
        interactions: branch_bus_interactions(),
    };

    AirWithBuses::new(
        branch_cols::NUM_COLUMNS,
        auxiliary_trace_build_data,
        proof_options,
        1,
        transition_constraints,
    )
    .with_main_builder(|builder| {
        use crate::constraints::templates::INV_SHIFT_32;
        use crate::tables::branch::cols;
        use crate::tables::types::SHIFT_16;

        let shift_8 = FieldElement::<GoldilocksField>::from(1u64 << 8);
        let shift_16 = FieldElement::<GoldilocksField>::from(SHIFT_16);
        let inv_2_32 = FieldElement::<GoldilocksField>::from(INV_SHIFT_32);
        let one = FieldElement::<GoldilocksField>::one();

        let jalr = builder.main_base(cols::JALR);
        let pc_0 = builder.main_base(cols::PC_0);
        let pc_1 = builder.main_base(cols::PC_1);
        let offset_0 = builder.main_base(cols::OFFSET_0);
        let offset_1 = builder.main_base(cols::OFFSET_1);
        let register_0 = builder.main_base(cols::REGISTER_0);
        let register_1 = builder.main_base(cols::REGISTER_1);
        let unmasked_low_byte = builder.main_base(cols::UNMASKED_LOW_BYTE);
        let next_pc_low_1 = builder.main_base(cols::NEXT_PC_LOW_1);
        let next_pc_high_0 = builder.main_base(cols::NEXT_PC_HIGH_0);
        let next_pc_high_1 = builder.main_base(cols::NEXT_PC_HIGH_1);
        let next_pc_high_2 = builder.main_base(cols::NEXT_PC_HIGH_2);

        let unmasked_0 = unmasked_low_byte + next_pc_low_1 * shift_8 + next_pc_high_0 * shift_16;
        let unmasked_1 = next_pc_high_1 + next_pc_high_2 * shift_16;

        // Constraint 0: PcCarry0IsBit
        let carry_0_pc = (pc_0 + offset_0 - unmasked_0) * inv_2_32;
        let cond_pc = one - jalr;
        builder.assert_zero_base(cond_pc * carry_0_pc * (one - carry_0_pc));

        // Constraint 1: PcCarry1IsBit
        let carry_1_pc = (pc_1 + offset_1 + carry_0_pc - unmasked_1) * inv_2_32;
        builder.assert_zero_base(cond_pc * carry_1_pc * (one - carry_1_pc));

        // Constraint 2: RegCarry0IsBit
        let carry_0_reg = (register_0 + offset_0 - unmasked_0) * inv_2_32;
        builder.assert_zero_base(jalr * carry_0_reg * (one - carry_0_reg));

        // Constraint 3: RegCarry1IsBit
        let carry_1_reg = (register_1 + offset_1 + carry_0_reg - unmasked_1) * inv_2_32;
        builder.assert_zero_base(jalr * carry_1_reg * (one - carry_1_reg));
    })
    .with_builder(|builder| {
        use crate::constraints::templates::INV_SHIFT_32;
        use crate::tables::branch::cols;
        use crate::tables::types::{GoldilocksExtension, SHIFT_16};
        use math::field::element::FieldElement;

        let shift_8 = FieldElement::<GoldilocksExtension>::from(1u64 << 8);
        let shift_16 = FieldElement::<GoldilocksExtension>::from(SHIFT_16);
        let inv_2_32 = FieldElement::<GoldilocksExtension>::from(INV_SHIFT_32);
        let one = FieldElement::<GoldilocksExtension>::one();

        let jalr = builder.main(0, cols::JALR);
        let pc_0 = builder.main(0, cols::PC_0);
        let pc_1 = builder.main(0, cols::PC_1);
        let offset_0 = builder.main(0, cols::OFFSET_0);
        let offset_1 = builder.main(0, cols::OFFSET_1);
        let register_0 = builder.main(0, cols::REGISTER_0);
        let register_1 = builder.main(0, cols::REGISTER_1);
        let unmasked_low_byte = builder.main(0, cols::UNMASKED_LOW_BYTE);
        let next_pc_low_1 = builder.main(0, cols::NEXT_PC_LOW_1);
        let next_pc_high_0 = builder.main(0, cols::NEXT_PC_HIGH_0);
        let next_pc_high_1 = builder.main(0, cols::NEXT_PC_HIGH_1);
        let next_pc_high_2 = builder.main(0, cols::NEXT_PC_HIGH_2);

        // Reconstruct next_pc_unmasked as DWordWL:
        // unmasked_0 = unmasked_low_byte + 2^8 * next_pc_low_1 + 2^16 * next_pc_high_0
        // unmasked_1 = next_pc_high_1 + 2^16 * next_pc_high_2
        let unmasked_0 = unmasked_low_byte + next_pc_low_1 * shift_8 + next_pc_high_0 * shift_16;
        let unmasked_1 = next_pc_high_1 + next_pc_high_2 * shift_16;

        // Constraint 0: PcCarry0IsBit — (1 - JALR) * carry_0_pc * (1 - carry_0_pc) = 0
        let carry_0_pc = (pc_0 + offset_0 - unmasked_0) * inv_2_32;
        let cond_pc = one - jalr;
        builder.assert_zero(cond_pc * carry_0_pc * (one - carry_0_pc));

        // Constraint 1: PcCarry1IsBit — (1 - JALR) * carry_1_pc * (1 - carry_1_pc) = 0
        let carry_1_pc = (pc_1 + offset_1 + carry_0_pc - unmasked_1) * inv_2_32;
        builder.assert_zero(cond_pc * carry_1_pc * (one - carry_1_pc));

        // Constraint 2: RegCarry0IsBit — JALR * carry_0_reg * (1 - carry_0_reg) = 0
        let carry_0_reg = (register_0 + offset_0 - unmasked_0) * inv_2_32;
        builder.assert_zero(jalr * carry_0_reg * (one - carry_0_reg));

        // Constraint 3: RegCarry1IsBit — JALR * carry_1_reg * (1 - carry_1_reg) = 0
        let carry_1_reg = (register_1 + offset_1 + carry_0_reg - unmasked_1) * inv_2_32;
        builder.assert_zero(jalr * carry_1_reg * (one - carry_1_reg));
    })
    .with_name("BRANCH")
}

/// Create HALT AIR with bus interactions (no transition constraints).
pub fn create_halt_air(proof_options: &ProofOptions) -> VmAir {
    let transition_constraints: Vec<Box<dyn TransitionConstraint<F, E>>> = vec![];

    let auxiliary_trace_build_data = AuxiliaryTraceBuildData {
        interactions: halt_bus_interactions(),
    };

    AirWithBuses::new(
        halt_cols::NUM_COLUMNS,
        auxiliary_trace_build_data,
        proof_options,
        1,
        transition_constraints,
    )
    .with_name("HALT")
}

/// Create COMMIT AIR with constraints and bus interactions.
pub fn create_commit_air(proof_options: &ProofOptions) -> VmAir {
    let (transition_constraints, _) = commit_constraints(0);

    let auxiliary_trace_build_data = AuxiliaryTraceBuildData {
        interactions: commit_bus_interactions(),
    };

    AirWithBuses::new(
        commit_cols::NUM_COLUMNS,
        auxiliary_trace_build_data,
        proof_options,
        1,
        transition_constraints,
    )
    .with_main_builder(|builder| {
        use crate::constraints::helpers::assert_is_bit_base;
        use crate::constraints::templates::INV_SHIFT_32;
        use crate::tables::commit::cols;

        let one = FieldElement::<GoldilocksField>::one();
        let inv_2_32 = FieldElement::<GoldilocksField>::from(INV_SHIFT_32);
        let shift_16 = FieldElement::<GoldilocksField>::from(65536u64);

        let first = builder.main_base(cols::FIRST);
        let end = builder.main_base(cols::END);
        let mu = builder.main_base(cols::MU);

        // Constraint 0: IS_BIT(FIRST)
        assert_is_bit_base(builder, first);

        // Constraint 1: IS_BIT(END)
        assert_is_bit_base(builder, end);

        // Constraint 2: IS_BIT(MU)
        assert_is_bit_base(builder, mu);

        // Constraint 3: (first + end) * (1 - mu) = 0
        builder.assert_zero_base((first + end) * (one - mu));

        // Constraint 4-5: ADD for address + 1 = address_incr
        let addr_lo = builder.main_base(cols::ADDRESS_0);
        let addr_hi = builder.main_base(cols::ADDRESS_1);
        let rhs_lo = one;
        let addr_incr_0 = builder.main_base(cols::ADDRESS_INCR_0);
        let addr_incr_1 = builder.main_base(cols::ADDRESS_INCR_1);
        let addr_incr_2 = builder.main_base(cols::ADDRESS_INCR_2);
        let addr_incr_3 = builder.main_base(cols::ADDRESS_INCR_3);
        let sum_lo = addr_incr_0 + addr_incr_1 * shift_16;
        let sum_hi = addr_incr_2 + addr_incr_3 * shift_16;

        let carry_0 = (addr_lo + rhs_lo - sum_lo) * inv_2_32;
        // Constraint 4
        builder.assert_zero_base(carry_0 * (one - carry_0));

        let carry_1 = (addr_hi + carry_0 - sum_hi) * inv_2_32;
        // Constraint 5
        builder.assert_zero_base(carry_1 * (one - carry_1));

        // Constraint 6-7: SUB for count_decr + 1 = count
        let cd_0 = builder.main_base(cols::COUNT_DECR_0);
        let cd_1 = builder.main_base(cols::COUNT_DECR_1);
        let cd_2 = builder.main_base(cols::COUNT_DECR_2);
        let cd_3 = builder.main_base(cols::COUNT_DECR_3);
        let lhs_lo = cd_0 + cd_1 * shift_16;
        let lhs_hi = cd_2 + cd_3 * shift_16;
        let count_lo = builder.main_base(cols::COUNT_0);
        let count_hi = builder.main_base(cols::COUNT_1);

        let carry_0_sub = (lhs_lo + one - count_lo) * inv_2_32;
        // Constraint 6
        builder.assert_zero_base(carry_0_sub * (one - carry_0_sub));

        let carry_1_sub = (lhs_hi + carry_0_sub - count_hi) * inv_2_32;
        // Constraint 7
        builder.assert_zero_base(carry_1_sub * (one - carry_1_sub));
    })
    .with_builder(|builder| {
        use crate::constraints::helpers::assert_is_bit;
        use crate::constraints::templates::INV_SHIFT_32;
        use crate::tables::commit::cols;
        use crate::tables::types::GoldilocksExtension;
        use math::field::element::FieldElement;

        let one = FieldElement::<GoldilocksExtension>::one();
        let inv_2_32 = FieldElement::<GoldilocksExtension>::from(INV_SHIFT_32);
        let shift_16 = FieldElement::<GoldilocksExtension>::from(65536u64);

        let first = builder.main(0, cols::FIRST);
        let end = builder.main(0, cols::END);
        let mu = builder.main(0, cols::MU);

        // Constraint 0: IS_BIT(FIRST)
        assert_is_bit(builder, first);

        // Constraint 1: IS_BIT(END)
        assert_is_bit(builder, end);

        // Constraint 2: IS_BIT(MU)
        assert_is_bit(builder, mu);

        // Constraint 3: (first + end) * (1 - mu) = 0
        builder.assert_zero((first + end) * (one - mu));

        // Constraint 4-5: ADD for address + 1 = address_incr (unconditional)
        // lhs = address (DWordWL: cols 3,4)
        // rhs = constant(1): lo=1, hi=0
        // sum = address_incr (DWordHL: cols 5,6,7,8 -> DWordWL)
        let addr_lo = builder.main(0, cols::ADDRESS_0);
        let addr_hi = builder.main(0, cols::ADDRESS_1);
        let rhs_lo = one;
        // sum_lo = addr_incr[0] + 2^16 * addr_incr[1]
        let addr_incr_0 = builder.main(0, cols::ADDRESS_INCR_0);
        let addr_incr_1 = builder.main(0, cols::ADDRESS_INCR_1);
        let addr_incr_2 = builder.main(0, cols::ADDRESS_INCR_2);
        let addr_incr_3 = builder.main(0, cols::ADDRESS_INCR_3);
        let sum_lo = addr_incr_0 + addr_incr_1 * shift_16;
        let sum_hi = addr_incr_2 + addr_incr_3 * shift_16;

        // carry_0 = (addr_lo + 1 - sum_lo) * 2^(-32)
        let carry_0 = (addr_lo + rhs_lo - sum_lo) * inv_2_32;
        // Constraint 4: carry_0 * (1 - carry_0) = 0
        builder.assert_zero(carry_0 * (one - carry_0));

        // carry_1 = (addr_hi + 0 + carry_0 - sum_hi) * 2^(-32)
        let carry_1 = (addr_hi + carry_0 - sum_hi) * inv_2_32;
        // Constraint 5: carry_1 * (1 - carry_1) = 0
        builder.assert_zero(carry_1 * (one - carry_1));

        // Constraint 6-7: SUB for count_decr + 1 = count (unconditional)
        // lhs = count_decr (DWordHL: cols 11,12,13,14 -> DWordWL)
        // rhs = constant(1): lo=1, hi=0
        // sum = count (DWordWL: cols 9,10)
        let cd_0 = builder.main(0, cols::COUNT_DECR_0);
        let cd_1 = builder.main(0, cols::COUNT_DECR_1);
        let cd_2 = builder.main(0, cols::COUNT_DECR_2);
        let cd_3 = builder.main(0, cols::COUNT_DECR_3);
        let lhs_lo = cd_0 + cd_1 * shift_16;
        let lhs_hi = cd_2 + cd_3 * shift_16;
        let count_lo = builder.main(0, cols::COUNT_0);
        let count_hi = builder.main(0, cols::COUNT_1);

        // carry_0 = (lhs_lo + 1 - count_lo) * 2^(-32)
        let carry_0_sub = (lhs_lo + one - count_lo) * inv_2_32;
        // Constraint 6: carry_0 * (1 - carry_0) = 0
        builder.assert_zero(carry_0_sub * (one - carry_0_sub));

        // carry_1 = (lhs_hi + carry_0 - count_hi) * 2^(-32)
        let carry_1_sub = (lhs_hi + carry_0_sub - count_hi) * inv_2_32;
        // Constraint 7: carry_1 * (1 - carry_1) = 0
        builder.assert_zero(carry_1_sub * (one - carry_1_sub));
    })
    .with_name("COMMIT")
}

/// Create PAGE AIR with bus interactions for a specific page.
///
/// Each PAGE table instance has its own AIR because the bus interactions
/// include the page_base as a constant. The `page_base` parameter specifies
/// the base address of this page.
///
/// The PAGE table has no transition constraints (it's a pure lookup table).
/// It interacts with:
/// - IS_BYTE bus: range checks for init/fini values
/// - Memory bus: provides initial and final memory tokens
pub fn create_page_air(proof_options: &ProofOptions, page_base: u64) -> VmAir {
    let transition_constraints: Vec<Box<dyn TransitionConstraint<F, E>>> = vec![];

    let auxiliary_trace_build_data = AuxiliaryTraceBuildData {
        interactions: page_bus_interactions(page_base),
    };

    AirWithBuses::new(
        page_cols::NUM_COLUMNS,
        auxiliary_trace_build_data,
        proof_options,
        1,
        transition_constraints,
    )
    .with_name(&format!("PAGE:0x{:x}", page_base))
}

/// Create REGISTER AIR with bus interactions.
///
/// The REGISTER table provides initial and final tokens for register accesses
/// on the Memory bus (is_register=1).
pub fn create_register_air(proof_options: &ProofOptions) -> VmAir {
    let transition_constraints: Vec<Box<dyn TransitionConstraint<F, E>>> = vec![];

    let auxiliary_trace_build_data = AuxiliaryTraceBuildData {
        interactions: register_bus_interactions(),
    };

    AirWithBuses::new(
        register_cols::NUM_COLUMNS,
        auxiliary_trace_build_data,
        proof_options,
        1,
        transition_constraints,
    )
    .with_name("REGISTER")
}
