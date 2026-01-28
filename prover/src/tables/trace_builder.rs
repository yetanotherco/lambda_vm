//! Trace generation from execution logs.
//!
//! This module uses a phased collection approach where each table's operations
//! are collected in explicit phases based on their dependencies.
//!
//! ## Architecture
//!
//! The trace generation follows this dependency graph:
//!
//! ```text
//! PHASE 1: Logs → CPU ops
//! PHASE 2: CPU → LT ops (SLT/BLT), Bitwise lookups
//! PHASE 3: LT → Bitwise lookups
//! ```
//!
//! Future phases (to be added):
//! - PHASE 2+: CPU → MEMW, LOAD, SHIFT, MUL, BRANCH
//! - PHASE 3+: LOAD → MEMW
//! - PHASE 4+: MEMW → LT (timestamp ordering)
//! - PHASE 5+: All tables → Bitwise lookups
//!
//! ## Usage
//!
//! ```ignore
//! use prover::tables::trace_builder::Traces;
//!
//! let traces = Traces::from_logs(&logs, instructions)?;
//! // Use traces.cpu, traces.bitwise, traces.lt
//! ```

use executor::vm::instruction::decoding::Instruction;
use executor::vm::logs::Log;
use executor::vm::memory::U64HashMap;
use stark::trace::TraceTable;

use super::bitwise::{self, BitwiseLookup};
use super::cpu::{self, CpuOperation};
use super::lt::{self, LtOperation};
use super::types::{GoldilocksExtension, GoldilocksField};
use crate::ProverError;

// =============================================================================
// Collection Context
// =============================================================================

/// Central context for collecting all operations during trace generation.
///
/// This struct accumulates operations from all phases. Add new fields as you
/// implement new tables.
pub struct CollectionContext {
    // =========================================================================
    // Primary operations (from logs)
    // =========================================================================
    /// CPU operations - one per instruction
    pub cpu_ops: Vec<CpuOperation>,

    // =========================================================================
    // Secondary operations (from CPU)
    // =========================================================================
    /// LT operations for comparisons (SLT, BLT instructions)
    pub lt_ops: Vec<LtOperation>,

    // TODO: Add these fields when implementing the tables:
    // pub memw_ops: Vec<MemwOperation>,
    // pub load_ops: Vec<LoadOperation>,
    // pub shift_ops: Vec<ShiftOperation>,
    // pub mul_ops: Vec<MulOperation>,
    // pub branch_ops: Vec<BranchOperation>,
    // pub divrem_ops: Vec<DivRemOperation>,

    // =========================================================================
    // State trackers (for memory consistency)
    // =========================================================================
    // TODO: Add these when implementing MEMW:
    // pub memory_state: MemoryState,
    // pub register_state: RegisterState,

    // =========================================================================
    // Lookups (accumulated from all tables)
    // =========================================================================
    /// Bitwise lookups accumulated from all tables
    pub bitwise_lookups: Vec<(BitwiseLookup, u8, u8, u8)>,
}

impl CollectionContext {
    /// Creates a new empty collection context with pre-allocated capacity.
    pub fn new(estimated_instructions: usize) -> Self {
        Self {
            cpu_ops: Vec::with_capacity(estimated_instructions),
            lt_ops: Vec::with_capacity(estimated_instructions / 10 + 1),
            bitwise_lookups: Vec::with_capacity(estimated_instructions * 4),
        }
    }
}

// =============================================================================
// Phase 1: Logs → CPU
// =============================================================================

/// Processes execution logs into CPU operations.
///
/// This is the entry point - all other operations derive from CPU ops.
pub fn collect_cpu_from_logs(
    ctx: &mut CollectionContext,
    logs: &[Log],
    instructions: &U64HashMap<Instruction>,
) -> Result<(), ProverError> {
    for (i, log) in logs.iter().enumerate() {
        // Timestamp: 4 units per instruction, starting at 4 (not 0)
        // Starting at 4 ensures first memory access has old_timestamp(0) < timestamp(4)
        let timestamp = (i as u64 + 1) * 4;

        let instruction = instructions
            .get(&log.current_pc)
            .copied()
            .ok_or(ProverError::MissingInstruction(log.current_pc))?;

        let cpu_op = CpuOperation::from_log(log, timestamp, instruction);
        ctx.cpu_ops.push(cpu_op);
    }
    Ok(())
}

// =============================================================================
// Phase 2: CPU → Secondary tables
// =============================================================================

/// Collects LT operations from CPU (SLT and BLT instructions).
///
/// From spec cpu.md constraint A3:
/// `LT[res[0]; arg1::DWordWL, arg2::DWordWL, signed]` with multiplicity SLT + BLT
pub fn collect_lt_from_cpu(ctx: &mut CollectionContext) {
    for cpu_op in &ctx.cpu_ops {
        if cpu_op.op_slt || cpu_op.op_blt {
            let arg1 = cpu_op.compute_arg1();
            let arg2 = cpu_op.compute_arg2();
            ctx.lt_ops.push(LtOperation::new(arg1, arg2, cpu_op.signed));
        }
    }
}

/// Collects bitwise lookups directly from CPU operations.
///
/// From spec cpu.md:
/// - A5: AND_BYTE[res[i]; arg1[i], arg2[i]] × 8 (multiplicity: AND)
/// - A6: OR_BYTE[res[i]; arg1[i], arg2[i]] × 8 (multiplicity: OR)
/// - A7: XOR_BYTE[res[i]; arg1[i], arg2[i]] × 8 (multiplicity: XOR)
/// - E2: MSB16[rv1_sign_bit; rv1[1]] (multiplicity: word_instr)
/// - E5: MSB16[arg2_sign_bit; rv2[1]] (multiplicity: word_instr)
/// - E8: MSB8[res_sign_bit; res[3]] (multiplicity: word_instr)
/// - is_equal: ZERO[is_equal; sum(res)] (multiplicity: BEQ)
/// - R28-R33: IS_BYTE range checks
pub fn collect_bitwise_from_cpu(ctx: &mut CollectionContext) {
    for cpu_op in &ctx.cpu_ops {
        ctx.bitwise_lookups.extend(cpu_op.collect_bitwise_lookups());
    }
}

// TODO: Add these functions when implementing the tables:
//
// /// Collects MEMW operations from CPU (register reads/writes, stores, PC).
// /// From spec cpu.md: M1, M3, M5, M7, M8
// pub fn collect_memw_from_cpu(ctx: &mut CollectionContext) { ... }
//
// /// Collects LOAD operations from CPU.
// /// From spec cpu.md: M6
// pub fn collect_load_from_cpu(ctx: &mut CollectionContext) { ... }
//
// /// Collects SHIFT operations from CPU.
// /// From spec cpu.md: A8
// pub fn collect_shift_from_cpu(ctx: &mut CollectionContext) { ... }
//
// /// Collects MUL operations from CPU.
// /// From spec cpu.md: A10
// pub fn collect_mul_from_cpu(ctx: &mut CollectionContext) { ... }
//
// /// Collects BRANCH operations from CPU.
// /// From spec cpu.md: O3
// pub fn collect_branch_from_cpu(ctx: &mut CollectionContext) { ... }

// =============================================================================
// Phase 3: LOAD → MEMW (to be implemented)
// =============================================================================

// TODO: Add when implementing LOAD:
// /// Collects MEMW operations from LOAD (memory reads).
// /// From spec load.md constraint 2: MEMW[res; 0, base_address, ...]
// pub fn collect_memw_from_load(ctx: &mut CollectionContext) { ... }

// =============================================================================
// Phase 4: MEMW → LT (to be implemented)
// =============================================================================

// TODO: Add when implementing MEMW:
// /// Collects LT operations from MEMW (timestamp ordering).
// /// From spec memw.md constraints 7-10: LT[1; old_timestamp[i], timestamp, 0]
// /// From spec memw.md constraints R1-R3: overflow checks
// pub fn collect_lt_from_memw(ctx: &mut CollectionContext) { ... }

// =============================================================================
// Phase 5: All tables → Bitwise lookups
// =============================================================================

/// Collects bitwise lookups from LT operations.
///
/// From spec lt.md:
/// - lt:c:lhs_msb: MSB16[lhs_msb; lhs[2]] (multiplicity: μ)
/// - lt:c:rhs_msb: MSB16[rhs_msb; rhs[2]] (multiplicity: μ)
/// - lt:c:range_lhs: IS_HALFWORD[lhs[1]] (multiplicity: μ)
/// - lt:c:range_rhs: IS_HALFWORD[rhs[1]] (multiplicity: μ)
/// - lt:c:lhs_sub_rhs_range: IS_HALFWORD[lhs_sub_rhs[i]] × 4 (multiplicity: μ)
pub fn collect_bitwise_from_lt(ctx: &mut CollectionContext) {
    for lt_op in &ctx.lt_ops {
        ctx.bitwise_lookups.extend(lt_op.collect_bitwise_lookups());
    }
}

// TODO: Add these functions when implementing the tables:
//
// /// Collects bitwise lookups from MEMW (IS_HALFWORD for address_add).
// /// From spec memw.md constraint 6: IS_HALFWORD[address_add[i][j]]
// pub fn collect_bitwise_from_memw(ctx: &mut CollectionContext) { ... }
//
// /// Collects bitwise lookups from LOAD (MSB8 for sign bit).
// /// From spec load.md constraints 3-5: MSB8[sign_bit; res[k]]
// pub fn collect_bitwise_from_load(ctx: &mut CollectionContext) { ... }
//
// /// Collects bitwise lookups from SHIFT.
// /// From spec shift.md: HWSL, HWSLC, AND_BYTE, MSB16
// pub fn collect_bitwise_from_shift(ctx: &mut CollectionContext) { ... }
//
// /// Collects bitwise lookups from MUL.
// /// From spec mul.md: IS_B20 for carry
// pub fn collect_bitwise_from_mul(ctx: &mut CollectionContext) { ... }
//
// /// Collects bitwise lookups from BRANCH.
// /// From spec branch.md: IS_BYTE, AND_BYTE, IS_HALFWORD
// pub fn collect_bitwise_from_branch(ctx: &mut CollectionContext) { ... }

// =============================================================================
// Trace Generation
// =============================================================================

/// All generated trace tables.
pub struct Traces {
    /// CPU execution trace (one row per instruction)
    pub cpu: TraceTable<GoldilocksField, GoldilocksExtension>,

    /// BITWISE precomputed lookup table (2^20 rows)
    pub bitwise: TraceTable<GoldilocksField, GoldilocksExtension>,

    /// LT comparison trace (deduplicated operations)
    pub lt: TraceTable<GoldilocksField, GoldilocksExtension>,

    // TODO: Add these fields when implementing the tables:
    // pub memw: TraceTable<GoldilocksField, GoldilocksExtension>,
    // pub load: TraceTable<GoldilocksField, GoldilocksExtension>,
    // pub shift: TraceTable<GoldilocksField, GoldilocksExtension>,
    // pub mul: TraceTable<GoldilocksField, GoldilocksExtension>,
    // pub branch: TraceTable<GoldilocksField, GoldilocksExtension>,
}

impl Traces {
    /// Generates all traces from execution logs using phased collection.
    ///
    /// The phases are:
    /// 1. Logs → CPU operations
    /// 2. CPU → LT operations, Bitwise lookups
    /// 3. LT → Bitwise lookups
    ///
    /// Future phases will add MEMW, LOAD, SHIFT, MUL, BRANCH tables.
    pub fn from_logs(
        logs: &[Log],
        instructions: U64HashMap<Instruction>,
    ) -> Result<Self, ProverError> {
        let mut ctx = CollectionContext::new(logs.len());

        // =====================================================================
        // PHASE 1: Logs → CPU
        // =====================================================================
        collect_cpu_from_logs(&mut ctx, logs, &instructions)?;

        // =====================================================================
        // PHASE 2: CPU → Secondary tables
        // Order within phase doesn't matter (all read from cpu_ops)
        // =====================================================================
        collect_lt_from_cpu(&mut ctx);
        collect_bitwise_from_cpu(&mut ctx);

        // TODO: Uncomment as you implement:
        // collect_memw_from_cpu(&mut ctx);
        // collect_load_from_cpu(&mut ctx);
        // collect_shift_from_cpu(&mut ctx);
        // collect_mul_from_cpu(&mut ctx);
        // collect_branch_from_cpu(&mut ctx);

        // =====================================================================
        // PHASE 3: LOAD → MEMW (must be after LOAD, before MEMW→LT)
        // =====================================================================
        // TODO: Uncomment when implementing LOAD:
        // collect_memw_from_load(&mut ctx);

        // =====================================================================
        // PHASE 4: MEMW → LT (must be after all MEMW collected)
        // =====================================================================
        // TODO: Uncomment when implementing MEMW:
        // collect_lt_from_memw(&mut ctx);

        // =====================================================================
        // PHASE 5: All tables → Bitwise lookups
        // =====================================================================
        collect_bitwise_from_lt(&mut ctx);

        // TODO: Uncomment as you implement:
        // collect_bitwise_from_memw(&mut ctx);
        // collect_bitwise_from_load(&mut ctx);
        // collect_bitwise_from_shift(&mut ctx);
        // collect_bitwise_from_mul(&mut ctx);
        // collect_bitwise_from_branch(&mut ctx);

        // =====================================================================
        // Generate final traces from collected operations
        // =====================================================================
        ctx.build_traces()
    }
}

impl CollectionContext {
    /// Converts collected operations into final trace tables.
    pub fn build_traces(self) -> Result<Traces, ProverError> {
        // Generate CPU trace
        let cpu = cpu::generate_cpu_trace(&self.cpu_ops);

        // Generate LT trace (handles deduplication internally)
        let lt = lt::generate_lt_trace(&self.lt_ops);

        // Generate BITWISE trace and update multiplicities
        let mut bitwise = bitwise::generate_bitwise_trace();
        bitwise::update_multiplicities(&mut bitwise, &self.bitwise_lookups);

        Ok(Traces { cpu, bitwise, lt })
    }
}
