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

use alloc::format;
use alloc::boxed::Box;
use alloc::vec::Vec;

#[cfg(feature = "prove")]
use std::path::PathBuf;

use crypto::fiat_shamir::is_transcript::IsStarkTranscript;
#[cfg(feature = "prove")]
use executor::elf::Elf;
#[cfg(feature = "prove")]
use executor::vm::execution::Executor;
#[cfg(feature = "prove")]
use executor::vm::instruction::decoding::Instruction;
#[cfg(feature = "prove")]
use executor::vm::logs::Log;
#[cfg(feature = "prove")]
use executor::vm::memory::U64HashMap;
use math::field::element::FieldElement;
use stark::constraints::transition::{TransitionConstraint, TransitionConstraintEvaluator};
use stark::debug::validate_trace;
use stark::domain::Domain;
use stark::lookup::{
    AirWithBuses, AuxiliaryTraceBuildData, BusInteraction, BusValue, NullBoundaryConstraintBuilder,
};
use stark::proof::options::ProofOptions;
use stark::proof::stark::MultiProof;
use stark::prover::{IsStarkProver, Prover, ProvingError};
#[cfg(feature = "disk-spill")]
use stark::storage_mode::StorageMode;
use stark::trace::TraceTable;
use stark::traits::AIR;

use crate::constraints::cpu::create_all_cpu_constraints;
use crate::tables::bitwise::{
    BitwiseOperation, BitwiseOperationType, bus_interactions as bitwise_bus_interactions,
    cols as bitwise_cols,
};
use crate::tables::branch::{
    branch_constraints, bus_interactions as branch_bus_interactions, cols as branch_cols,
};
use crate::tables::bytewise::{
    bus_interactions as bytewise_bus_interactions, cols as bytewise_cols,
};
use crate::tables::commit::{
    bus_interactions as commit_bus_interactions, cols as commit_cols,
    create_constraints as commit_constraints,
};
use crate::tables::cpu::{
    CpuOperation, bus_interactions as cpu_bus_interactions, cols as cpu_cols,
};
use crate::tables::cpu32::{
    bus_interactions as cpu32_bus_interactions, cols as cpu32_cols, cpu32_constraints,
};
use crate::tables::decode::{bus_interactions as decode_bus_interactions, cols as decode_cols};
use crate::tables::dvrm::{
    bus_interactions as dvrm_bus_interactions, cols as dvrm_cols, dvrm_constraints,
};
use crate::tables::ec_scalar::{
    bus_interactions as ec_scalar_bus_interactions, cols as ec_scalar_cols,
};
use crate::tables::ecdas::{bus_interactions as ecdas_bus_interactions, cols as ecdas_cols};
use crate::tables::ecsm::{bus_interactions as ecsm_bus_interactions, cols as ecsm_cols};
use crate::tables::eq::{bus_interactions as eq_bus_interactions, cols as eq_cols, eq_constraints};
use crate::tables::halt::{bus_interactions as halt_bus_interactions, cols as halt_cols};
use crate::tables::keccak::{bus_interactions as keccak_bus_interactions, cols as keccak_cols};
use crate::tables::keccak_rc::{
    bus_interactions as keccak_rc_bus_interactions, cols as keccak_rc_cols,
};
use crate::tables::keccak_rnd::{
    bus_interactions as keccak_rnd_bus_interactions, cols as keccak_rnd_cols,
};
use crate::tables::load::{
    bus_interactions as load_bus_interactions, cols as load_cols, constraints as load_constraints,
};
use crate::tables::lt::{
    LtOperation, bus_interactions as lt_bus_interactions, cols as lt_cols, lt_constraints,
};
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
use crate::tables::mul::{
    bus_interactions as mul_bus_interactions, cols as mul_cols, mul_constraints,
};
use crate::tables::page::{bus_interactions as page_bus_interactions, cols as page_cols};
use crate::tables::register::{
    bus_interactions as register_bus_interactions, cols as register_cols,
};
use crate::tables::shift::{
    bus_interactions as shift_bus_interactions, cols as shift_cols, shift_constraints,
};
use crate::tables::store::{
    bus_interactions as store_bus_interactions, cols as store_cols, store_constraints,
};
use crate::tables::types::{BusId, GoldilocksExtension, GoldilocksField};

pub type F = GoldilocksField;
pub type E = GoldilocksExtension;
pub type FE = FieldElement<F>;

pub type VmAir = AirWithBuses<F, E, NullBoundaryConstraintBuilder, ()>;

type GoldilocksPair<'a, PI> = (
    &'a dyn AIR<Field = F, FieldExtension = E, PublicInputs = PI>,
    &'a mut TraceTable<F, E>,
    &'a PI,
);

pub fn multi_prove_ram<PI>(
    air_trace_pairs: Vec<GoldilocksPair<'_, PI>>,
    transcript: &mut (impl IsStarkTranscript<E, F> + Clone + Send),
) -> Result<MultiProof<F, E, PI>, ProvingError>
where
    PI: Send + Sync + Clone,
{
    Prover::<F, E, PI>::multi_prove(
        air_trace_pairs,
        transcript,
        #[cfg(feature = "disk-spill")]
        StorageMode::Ram,
    )
}

// =============================================================================
// Soundness regression helpers (negative AIR tests)
// =============================================================================

/// Build a bus-less AIR carrying only the given in-chip transition constraints.
/// With zero bus interactions, `AirWithBuses::new` appends no LogUp constraints
/// and allocates no aux columns, so `validate_trace` evaluates exactly the chip's
/// transition constraints over a main-only trace.
pub fn busless_air<C: TransitionConstraint<F, E> + 'static>(
    num_columns: usize,
    constraints: Vec<C>,
) -> VmAir {
    let transition_constraints = constraints.into_iter().map(|c| c.boxed()).collect();
    AirWithBuses::new(
        num_columns,
        AuxiliaryTraceBuildData {
            interactions: vec![],
        },
        &ProofOptions::default_test_options(),
        1,
        transition_constraints,
    )
}

/// Run `validate_trace` for a bus-less chip AIR over a main-only trace.
/// Returns `true` iff every transition constraint holds on every row.
pub fn validate_busless(air: &VmAir, trace: &TraceTable<F, E>) -> bool {
    let domain = Domain::new(air, trace.num_rows());
    validate_trace(air, &(), trace, &domain, &[], None)
}

/// Number of transition constraints a production builder registers on top of its
/// bus constraints, as a delta against a bus-only AIR with the same interactions
/// but no in-chip constraints. Isolates the in-chip count even though
/// `AirWithBuses::new` also appends LogUp constraints, so a plain count cannot.
pub fn in_chip_constraint_count(
    wired: usize,
    num_columns: usize,
    buses: Vec<BusInteraction>,
) -> usize {
    let bus_only = AirWithBuses::<F, E, NullBoundaryConstraintBuilder, ()>::new(
        num_columns,
        AuxiliaryTraceBuildData {
            interactions: buses,
        },
        &ProofOptions::default_test_options(),
        1,
        vec![],
    )
    .num_transition_constraints();
    wired
        .checked_sub(bus_only)
        .expect("wired (in-chip + bus constraints) must be >= bus-only constraint count")
}

/// Collect the `start_column`s of every `IS_HALFWORD` sender in `interactions`.
/// Used to assert input/operand half-limbs are range-checked. Scope: only
/// single-column `Packed` senders (which is how every current IS_HALFWORD sender is
/// declared); it does not inspect `Linear` senders or sender multiplicities.
pub fn is_halfword_sender_columns(interactions: &[BusInteraction]) -> Vec<usize> {
    let id: u64 = BusId::IsHalfword.into();
    interactions
        .iter()
        .filter(|i| i.is_sender && i.bus_id == id)
        .flat_map(|i| {
            i.values.iter().filter_map(|v| match v {
                BusValue::Packed { start_column, .. } => Some(*start_column),
                BusValue::Linear(_) => None,
            })
        })
        .collect()
}

// =============================================================================
// ELF Execution Helpers
// =============================================================================

/// Returns the raw ELF bytes for an assembly test program.
#[cfg(feature = "prove")]
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
#[cfg(feature = "prove")]
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
#[cfg(feature = "prove")]
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
#[cfg(feature = "prove")]
pub fn collect_lt_lookups_from_logs(
    logs: &[Log],
    instructions: &U64HashMap<Instruction>,
) -> Vec<LtOperation> {
    #[cfg(feature = "prove")]
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
#[cfg(feature = "prove")]
pub fn collect_load_ops_from_logs(
    logs: &[Log],
    instructions: &U64HashMap<Instruction>,
) -> Vec<crate::tables::load::LoadOperation> {
    #[cfg(feature = "prove")]
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
#[cfg(feature = "prove")]
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
#[cfg(feature = "prove")]
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
#[cfg(feature = "prove")]
pub fn generate_minimal_bitwise_trace(ops: &[BitwiseOperation]) -> TraceTable<F, E> {
    #[cfg(feature = "prove")]
    use std::collections::HashMap;

    // Collect unique (lo_byte, hi_byte, shift) tuples and count multiplicities per lookup type
    let mut row_data: HashMap<(u8, u8, u8), [u64; 10]> = HashMap::new();

    for op in ops {
        let key = (op.x, op.y, op.z);
        let mu_idx = match op.lookup_type {
            BitwiseOperationType::Msb8 => 0,
            BitwiseOperationType::Msb16 => 1,
            BitwiseOperationType::Zero => 2,
            BitwiseOperationType::AreBytes => 3,
            BitwiseOperationType::IsHalf => 4,
            BitwiseOperationType::IsB20 => 5,
            BitwiseOperationType::Hwsl => 6,
            BitwiseOperationType::ByteAluAnd => 7,
            BitwiseOperationType::ByteAluOr => 8,
            BitwiseOperationType::ByteAluXor => 9,
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
        data[base + bitwise_cols::MU_MSB8] = FE::from(mus[0]);
        data[base + bitwise_cols::MU_MSB16] = FE::from(mus[1]);
        data[base + bitwise_cols::MU_ZERO] = FE::from(mus[2]);
        data[base + bitwise_cols::MU_ARE_BYTES] = FE::from(mus[3]);
        data[base + bitwise_cols::MU_IS_HALF] = FE::from(mus[4]);
        data[base + bitwise_cols::MU_IS_B20] = FE::from(mus[5]);
        data[base + bitwise_cols::MU_HWSL] = FE::from(mus[6]);
        data[base + bitwise_cols::MU_BYTE_ALU_AND] = FE::from(mus[7]);
        data[base + bitwise_cols::MU_BYTE_ALU_OR] = FE::from(mus[8]);
        data[base + bitwise_cols::MU_BYTE_ALU_XOR] = FE::from(mus[9]);
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
    let mut transition_constraints: Vec<Box<dyn TransitionConstraintEvaluator<F, E>>> = Vec::new();
    for c in is_bit {
        transition_constraints.push(c.boxed());
    }
    for c in add {
        transition_constraints.push(c.boxed());
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
    let transition_constraints: Vec<Box<dyn TransitionConstraintEvaluator<F, E>>> = vec![];

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

/// Create LT AIR with constraints and bus interactions.
pub fn create_lt_air(proof_options: &ProofOptions) -> VmAir {
    let (constraints, _) = lt_constraints(0);
    let transition_constraints: Vec<Box<dyn TransitionConstraintEvaluator<F, E>>> =
        constraints.into_iter().map(|c| c.boxed()).collect();

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
    let transition_constraints: Vec<Box<dyn TransitionConstraintEvaluator<F, E>>> =
        constraints.into_iter().map(|c| c.boxed()).collect();

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

/// Create the EQ AIR.
pub fn create_eq_air(proof_options: &ProofOptions) -> VmAir {
    let (transition_constraints, _) = eq_constraints(0);
    let auxiliary_trace_build_data = AuxiliaryTraceBuildData {
        interactions: eq_bus_interactions(),
    };
    AirWithBuses::new(
        eq_cols::NUM_COLUMNS,
        auxiliary_trace_build_data,
        proof_options,
        1,
        transition_constraints,
    )
    .with_name("EQ")
}

/// Create the BYTEWISE AIR. No polynomial constraints.
pub fn create_bytewise_air(proof_options: &ProofOptions) -> VmAir {
    let transition_constraints: Vec<Box<dyn TransitionConstraintEvaluator<F, E>>> = vec![];
    let auxiliary_trace_build_data = AuxiliaryTraceBuildData {
        interactions: bytewise_bus_interactions(),
    };
    AirWithBuses::new(
        bytewise_cols::NUM_COLUMNS,
        auxiliary_trace_build_data,
        proof_options,
        1,
        transition_constraints,
    )
    .with_name("BYTEWISE")
}

/// Create the STORE AIR.
pub fn create_store_air(proof_options: &ProofOptions) -> VmAir {
    let (transition_constraints, _) = store_constraints(0);
    let auxiliary_trace_build_data = AuxiliaryTraceBuildData {
        interactions: store_bus_interactions(),
    };
    AirWithBuses::new(
        store_cols::NUM_COLUMNS,
        auxiliary_trace_build_data,
        proof_options,
        1,
        transition_constraints,
    )
    .with_name("STORE")
}

/// Create the CPU32 AIR.
pub fn create_cpu32_air(proof_options: &ProofOptions) -> VmAir {
    let (transition_constraints, _) = cpu32_constraints(0);
    let auxiliary_trace_build_data = AuxiliaryTraceBuildData {
        interactions: cpu32_bus_interactions(),
    };
    AirWithBuses::new(
        cpu32_cols::NUM_COLUMNS,
        auxiliary_trace_build_data,
        proof_options,
        1,
        transition_constraints,
    )
    .with_name("CPU32")
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
    let transition_constraints: Vec<Box<dyn TransitionConstraintEvaluator<F, E>>> = vec![];

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

/// Create MUL AIR with constraints and bus interactions.
pub fn create_mul_air(proof_options: &ProofOptions) -> VmAir {
    let (constraints, _) = mul_constraints(0);
    let transition_constraints: Vec<Box<dyn TransitionConstraintEvaluator<F, E>>> =
        constraints.into_iter().map(|c| c.boxed()).collect();

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
    .with_name("MUL")
}

/// Create DVRM AIR with constraints and bus interactions.
pub fn create_dvrm_air(proof_options: &ProofOptions) -> VmAir {
    let (constraints, _) = dvrm_constraints(0);
    let transition_constraints: Vec<Box<dyn TransitionConstraintEvaluator<F, E>>> =
        constraints.into_iter().map(|c| c.boxed()).collect();

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
    let transition_constraints: Vec<Box<dyn TransitionConstraintEvaluator<F, E>>> =
        constraints.into_iter().map(|c| c.boxed()).collect();

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
    .with_name("BRANCH")
}

/// Create HALT AIR with bus interactions (no transition constraints).
pub fn create_halt_air(proof_options: &ProofOptions) -> VmAir {
    let transition_constraints: Vec<Box<dyn TransitionConstraintEvaluator<F, E>>> = vec![];

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
/// - ARE_BYTES bus: range checks for init/fini values
/// - Memory bus: provides initial and final memory tokens
pub fn create_page_air(proof_options: &ProofOptions, page_base: u64) -> VmAir {
    let transition_constraints: Vec<Box<dyn TransitionConstraintEvaluator<F, E>>> = vec![];

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
    let transition_constraints: Vec<Box<dyn TransitionConstraintEvaluator<F, E>>> = vec![];

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

/// Create KECCAK core AIR with ADD constraints and bus interactions.
pub fn create_keccak_air(proof_options: &ProofOptions) -> VmAir {
    let (constraints, _) = crate::tables::keccak::create_constraints(0);
    let transition_constraints: Vec<Box<dyn TransitionConstraintEvaluator<F, E>>> = constraints;

    let auxiliary_trace_build_data = AuxiliaryTraceBuildData {
        interactions: keccak_bus_interactions(),
    };

    AirWithBuses::new(
        keccak_cols::NUM_COLUMNS,
        auxiliary_trace_build_data,
        proof_options,
        1,
        transition_constraints,
    )
    .with_name("KECCAK")
}

/// Create KECCAK_RND AIR with pi constraints and bus interactions.
pub fn create_keccak_rnd_air(proof_options: &ProofOptions) -> VmAir {
    let (constraints, _) = crate::tables::keccak_rnd::create_constraints(0);
    let transition_constraints: Vec<Box<dyn TransitionConstraintEvaluator<F, E>>> = constraints;

    let auxiliary_trace_build_data = AuxiliaryTraceBuildData {
        interactions: keccak_rnd_bus_interactions(),
    };

    AirWithBuses::new(
        keccak_rnd_cols::NUM_COLUMNS,
        auxiliary_trace_build_data,
        proof_options,
        1,
        transition_constraints,
    )
    .with_name("KECCAK_RND")
}

/// Create KECCAK_RC AIR with bus interactions (preprocessed table).
pub fn create_keccak_rc_air(proof_options: &ProofOptions) -> VmAir {
    let transition_constraints: Vec<Box<dyn TransitionConstraintEvaluator<F, E>>> = vec![];

    let auxiliary_trace_build_data = AuxiliaryTraceBuildData {
        interactions: keccak_rc_bus_interactions(),
    };

    AirWithBuses::new(
        keccak_rc_cols::NUM_COLUMNS,
        auxiliary_trace_build_data,
        proof_options,
        1,
        transition_constraints,
    )
    .with_name("KECCAK_RC")
}

/// Create ECSM core AIR (secp256k1 scalar-multiplication orchestrator).
pub fn create_ecsm_air(proof_options: &ProofOptions) -> VmAir {
    let (transition_constraints, _) = crate::tables::ecsm::create_constraints(0);
    let auxiliary_trace_build_data = AuxiliaryTraceBuildData {
        interactions: ecsm_bus_interactions(),
    };
    AirWithBuses::new(
        ecsm_cols::NUM_COLUMNS,
        auxiliary_trace_build_data,
        proof_options,
        1,
        transition_constraints,
    )
    .with_name("ECSM")
}

/// Create EC_SCALAR AIR (serves the scalar bit-by-bit to ECDAS).
pub fn create_ec_scalar_air(proof_options: &ProofOptions) -> VmAir {
    let (transition_constraints, _) = crate::tables::ec_scalar::create_constraints(0);
    let auxiliary_trace_build_data = AuxiliaryTraceBuildData {
        interactions: ec_scalar_bus_interactions(),
    };
    AirWithBuses::new(
        ec_scalar_cols::NUM_COLUMNS,
        auxiliary_trace_build_data,
        proof_options,
        1,
        transition_constraints,
    )
    .with_name("EC_SCALAR")
}

/// Create ECDAS AIR (per-step double/add of the scalar-multiplication sequence).
pub fn create_ecdas_air(proof_options: &ProofOptions) -> VmAir {
    let (transition_constraints, _) = crate::tables::ecdas::create_constraints(0);
    let auxiliary_trace_build_data = AuxiliaryTraceBuildData {
        interactions: ecdas_bus_interactions(),
    };
    AirWithBuses::new(
        ecdas_cols::NUM_COLUMNS,
        auxiliary_trace_build_data,
        proof_options,
        1,
        transition_constraints,
    )
    .with_name("ECDAS")
}
