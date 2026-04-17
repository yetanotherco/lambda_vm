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
    .with_name("CPU")
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
    .with_builder(|builder| {
        use crate::constraints::helpers::assert_is_bit;

        let mu_read = builder.main(0, memw_aligned_cols::MU_READ);
        let mu_write = builder.main(0, memw_aligned_cols::MU_WRITE);
        let mu_sum = &mu_read + &mu_write;

        // Constraint 0: IS_BIT<mu_sum>: mu_sum * (1 - mu_sum) == 0
        builder.assert_zero(
            mu_sum.clone()
                * (math::field::element::FieldElement::<
                    crate::tables::types::GoldilocksExtension,
                >::one()
                    - mu_sum.clone()),
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
    .with_builder(|builder| {
        use crate::constraints::helpers::assert_is_bit;

        let mu_read = builder.main(0, memw_register_cols::MU_READ);
        let mu_write = builder.main(0, memw_register_cols::MU_WRITE);

        // Constraint 0: IS_BIT(mu_read)
        assert_is_bit(builder, mu_read.clone());
        // Constraint 1: IS_BIT(mu_write)
        assert_is_bit(builder, mu_write.clone());
        // Constraint 2: (mu_read + mu_write) * (1 - mu_read - mu_write) == 0
        let mu_sum = &mu_read + &mu_write;
        builder.assert_zero(
            mu_sum.clone()
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
        builder.assert_zero((&one - &lhs_signed) * &lhs_is_neg);

        // Constraint 1: RhsSign -- (1 - rhs_signed) * rhs_is_negative == 0
        builder.assert_zero((&one - &rhs_signed) * &rhs_is_neg);

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
        let lhs_sign_ext = &sign_fill * &lhs_is_neg;
        let lhs_ext = [
            lhs[0].clone(),
            lhs[1].clone(),
            lhs[2].clone(),
            lhs[3].clone(),
            lhs_sign_ext.clone(),
            lhs_sign_ext.clone(),
            lhs_sign_ext.clone(),
            lhs_sign_ext.clone(),
        ];

        // rhs_ext[0..4] = rhs halfwords, rhs_ext[4..8] = SIGN_FILL * rhs_is_negative
        let rhs_sign_ext = &sign_fill * &rhs_is_neg;
        let rhs_ext = [
            rhs[0].clone(),
            rhs[1].clone(),
            rhs[2].clone(),
            rhs[3].clone(),
            rhs_sign_ext.clone(),
            rhs_sign_ext.clone(),
            rhs_sign_ext.clone(),
            rhs_sign_ext.clone(),
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
                            inner_sum = &inner_sum + &(&lhs_ext[j] * &rhs_ext[idx - j]);
                        }
                    }
                    if k == 0 {
                        sum = &sum + &inner_sum;
                    } else {
                        sum = &sum + &(&inner_sum * &shift_16);
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
        let unmasked_0 =
            &unmasked_low_byte + &next_pc_low_1 * &shift_8 + &next_pc_high_0 * &shift_16;
        let unmasked_1 = &next_pc_high_1 + &next_pc_high_2 * &shift_16;

        // Constraint 0: PcCarry0IsBit — (1 - JALR) * carry_0_pc * (1 - carry_0_pc) = 0
        let carry_0_pc = (&pc_0 + &offset_0 - &unmasked_0) * &inv_2_32;
        let cond_pc = &one - &jalr;
        builder.assert_zero(&cond_pc * &carry_0_pc * (&one - &carry_0_pc));

        // Constraint 1: PcCarry1IsBit — (1 - JALR) * carry_1_pc * (1 - carry_1_pc) = 0
        let carry_1_pc = (&pc_1 + &offset_1 + &carry_0_pc - &unmasked_1) * &inv_2_32;
        builder.assert_zero(&cond_pc * &carry_1_pc * (&one - &carry_1_pc));

        // Constraint 2: RegCarry0IsBit — JALR * carry_0_reg * (1 - carry_0_reg) = 0
        let carry_0_reg = (&register_0 + &offset_0 - &unmasked_0) * &inv_2_32;
        builder.assert_zero(&jalr * &carry_0_reg * (&one - &carry_0_reg));

        // Constraint 3: RegCarry1IsBit — JALR * carry_1_reg * (1 - carry_1_reg) = 0
        let carry_1_reg = (&register_1 + &offset_1 + &carry_0_reg - &unmasked_1) * &inv_2_32;
        builder.assert_zero(&jalr * &carry_1_reg * (&one - &carry_1_reg));
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
