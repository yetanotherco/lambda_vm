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
//! // Use traces.cpu, traces.bitwise, traces.lt, traces.memw, traces.load
//! ```

use std::collections::HashMap;

use executor::vm::instruction::decoding::{Instruction, LoadStoreWidth};
use executor::vm::logs::Log;
use executor::vm::memory::U64HashMap;
use stark::trace::TraceTable;

use super::bitwise::{self, BitwiseLookup};
use super::cpu::{self, CpuOperation};
use super::load::{self, LoadOperation};
use super::lt::{self, LtOperation};
use super::memw::{self, MemwOperation};
use super::types::{GoldilocksExtension, GoldilocksField};
use crate::ProverError;

// =============================================================================
// Memory and Register State Tracking
// =============================================================================

/// Memory cell state: (value_byte, last_write_timestamp)
type MemoryCell = (u8, u64);

/// Register state: (value, last_write_timestamp)
type RegisterCell = (u64, u64);

/// Memory state tracker for generating MEMW/LOAD traces.
struct MemoryState {
    /// Map from byte address to (value, timestamp)
    cells: HashMap<u64, MemoryCell>,
}

impl MemoryState {
    fn new() -> Self {
        Self {
            cells: HashMap::new(),
        }
    }

    /// Read a byte from memory. Returns (value, timestamp) or (0, 0) if never written.
    fn read_byte(&self, address: u64) -> MemoryCell {
        self.cells.get(&address).copied().unwrap_or((0, 0))
    }

    /// Write a byte to memory with the given timestamp.
    fn write_byte(&mut self, address: u64, value: u8, timestamp: u64) {
        self.cells.insert(address, (value, timestamp));
    }

    /// Read multiple bytes. Returns arrays of values and timestamps.
    fn read_bytes(&self, base_address: u64, count: usize) -> ([u64; 8], [u64; 8]) {
        let mut values = [0u64; 8];
        let mut timestamps = [0u64; 8];
        for i in 0..count {
            let (val, ts) = self.read_byte(base_address.wrapping_add(i as u64));
            values[i] = val as u64;
            timestamps[i] = ts;
        }
        (values, timestamps)
    }

    /// Write multiple bytes from a value.
    fn write_bytes(&mut self, base_address: u64, value: u64, count: usize, timestamp: u64) {
        for i in 0..count {
            let byte = ((value >> (i * 8)) & 0xFF) as u8;
            self.write_byte(base_address.wrapping_add(i as u64), byte, timestamp);
        }
    }
}

/// Register state tracker for generating MEMW register traces.
struct RegisterState {
    /// Register file: (value, last_write_timestamp)
    regs: [RegisterCell; 32],
}

impl RegisterState {
    fn new() -> Self {
        Self {
            // All registers start at (0, 0) - value 0 at timestamp 0
            regs: [(0, 0); 32],
        }
    }

    /// Read a register. Returns (value, last_write_timestamp).
    fn read(&self, reg: u8) -> RegisterCell {
        self.regs[reg as usize]
    }

    /// Write a register with the given timestamp.
    fn write(&mut self, reg: u8, value: u64, timestamp: u64) {
        if reg != 0 {
            // x0 is always 0 and never written
            self.regs[reg as usize] = (value, timestamp);
        }
    }
}

/// Convert LoadStoreWidth to byte count and signed flag.
fn width_to_bytes_and_signed(width: LoadStoreWidth) -> (usize, bool) {
    match width {
        LoadStoreWidth::Byte => (1, true),
        LoadStoreWidth::ByteUnsigned => (1, false),
        LoadStoreWidth::Half => (2, true),
        LoadStoreWidth::HalfUnsigned => (2, false),
        LoadStoreWidth::Word => (4, true),
        LoadStoreWidth::WordUnsigned => (4, false),
        LoadStoreWidth::DoubleWord => (8, false),
    }
}

/// Pack a 64-bit register value into the MEMW value format.
///
/// For register operations, values are packed as [lo32, hi32, 0, 0, 0, 0, 0, 0].
fn pack_register_value(value: u64) -> [u64; 8] {
    [
        value & 0xFFFF_FFFF,
        value >> 32,
        0, 0, 0, 0, 0, 0,
    ]
}

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

    /// MEMW operations for memory/register reads and writes
    pub memw_ops: Vec<MemwOperation>,

    /// LOAD operations for memory loads with sign/zero extension
    pub load_ops: Vec<LoadOperation>,

    // TODO: Add these fields when implementing the tables:
    // pub shift_ops: Vec<ShiftOperation>,
    // pub mul_ops: Vec<MulOperation>,
    // pub branch_ops: Vec<BranchOperation>,
    // pub divrem_ops: Vec<DivRemOperation>,

    // =========================================================================
    // State trackers (for memory consistency)
    // =========================================================================
    /// Memory state: tracks (value, timestamp) for each byte address
    memory_state: MemoryState,
    /// Register state: tracks (value, timestamp) for each register
    register_state: RegisterState,

    // =========================================================================
    // Lookups (accumulated from all tables)
    // =========================================================================
    /// Bitwise lookups accumulated from all tables
    pub bitwise_lookups: Vec<(BitwiseLookup, u8, u8, u8)>,

    // =========================================================================
    // Instruction cache (for MEMW/LOAD collection)
    // =========================================================================
    /// Instructions indexed by PC (needed for MEMW/LOAD collection)
    instructions: U64HashMap<Instruction>,
    /// Logs (needed for MEMW/LOAD collection)
    logs: Vec<Log>,
}

impl CollectionContext {
    /// Creates a new empty collection context with pre-allocated capacity.
    pub fn new(
        estimated_instructions: usize,
        instructions: U64HashMap<Instruction>,
        logs: Vec<Log>,
    ) -> Self {
        Self {
            cpu_ops: Vec::with_capacity(estimated_instructions),
            lt_ops: Vec::with_capacity(estimated_instructions / 10 + 1),
            memw_ops: Vec::with_capacity(estimated_instructions * 3), // ~3 ops per instruction (M1, M3, M5)
            load_ops: Vec::with_capacity(estimated_instructions / 8 + 1),
            memory_state: MemoryState::new(),
            register_state: RegisterState::new(),
            bitwise_lookups: Vec::with_capacity(estimated_instructions * 4),
            instructions,
            logs,
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

/// Collects MEMW operations from CPU for register reads/writes and stores.
///
/// From spec cpu.md:
/// - M1: Register read rs1 at timestamp+0
/// - M3: Register read rs2 at timestamp+1
/// - M5: Register write rd at timestamp+2
/// - M7: Memory store at timestamp+1
pub fn collect_memw_from_cpu(ctx: &mut CollectionContext) {
    for (i, cpu_op) in ctx.cpu_ops.iter().enumerate() {
        let log = &ctx.logs[i];
        let timestamp = cpu_op.timestamp;

        // M1: Read rs1 register at timestamp+0
        if cpu_op.read_register1 && cpu_op.rs1 != 0 {
            let reg_value = pack_register_value(log.src1_val);
            let reg_addr = 2 * cpu_op.rs1 as u64;
            let (_old_val, old_ts) = ctx.register_state.read(cpu_op.rs1);
            let old_timestamps = [old_ts; 8];

            let memw_op = MemwOperation::new(
                true,       // is_register
                reg_addr,
                reg_value,
                timestamp,  // timestamp + 0
                8,          // width = 8 (full register)
                true,       // is_read
            ).with_old(reg_value, old_timestamps);

            ctx.memw_ops.push(memw_op);
            ctx.register_state.write(cpu_op.rs1, log.src1_val, timestamp);
        }

        // M3: Read rs2 register at timestamp+1
        if cpu_op.read_register2 && cpu_op.rs2 != 0 {
            let reg_value = pack_register_value(log.src2_val);
            let reg_addr = 2 * cpu_op.rs2 as u64;
            let (_old_val, old_ts) = ctx.register_state.read(cpu_op.rs2);
            let old_timestamps = [old_ts; 8];

            let memw_op = MemwOperation::new(
                true,           // is_register
                reg_addr,
                reg_value,
                timestamp + 1,  // timestamp + 1
                8,
                true,           // is_read
            ).with_old(reg_value, old_timestamps);

            ctx.memw_ops.push(memw_op);
            ctx.register_state.write(cpu_op.rs2, log.src2_val, timestamp + 1);
        }

        // M5: Write rd register at timestamp+2
        if cpu_op.write_register && cpu_op.rd != 0 {
            let reg_value = pack_register_value(log.dst_val);
            let reg_addr = 2 * cpu_op.rd as u64;
            let (old_val, old_ts) = ctx.register_state.read(cpu_op.rd);
            let old_value = pack_register_value(old_val);
            let old_timestamps = [old_ts; 8];

            let memw_op = MemwOperation::new(
                true,           // is_register
                reg_addr,
                reg_value,
                timestamp + 2,  // timestamp + 2
                8,
                false,          // is_read = false (write)
            ).with_old(old_value, old_timestamps);

            ctx.memw_ops.push(memw_op);
            ctx.register_state.write(cpu_op.rd, log.dst_val, timestamp + 2);
        }

        // M7: Memory store at timestamp+1
        if cpu_op.op_store {
            let instruction = ctx.instructions.get(&log.current_pc).copied().unwrap();
            if let Instruction::Store { offset, width, .. } = instruction {
                let base_address = log.src1_val.wrapping_add(offset as i64 as u64);
                let (byte_count, _) = width_to_bytes_and_signed(width);
                let store_value = log.src2_val;

                let (old_values, old_timestamps) = ctx.memory_state.read_bytes(base_address, 8);

                // Pack store value as [lo32, hi32, 0, 0, 0, 0, 0, 0]
                let value_bytes = [
                    store_value & 0xFFFF_FFFF,
                    store_value >> 32,
                    0, 0, 0, 0, 0, 0,
                ];

                let memw_op = MemwOperation::new(
                    false,          // is_register = false (memory)
                    base_address,
                    value_bytes,
                    timestamp + 1,  // M7 uses timestamp+1
                    byte_count as u8,
                    false,          // is_read = false (write)
                ).with_old(old_values, old_timestamps);

                ctx.memw_ops.push(memw_op);
                ctx.memory_state.write_bytes(base_address, store_value, byte_count, timestamp + 1);
            }
        }
    }
}

/// Collects LOAD operations from CPU and creates corresponding MEMW reads.
///
/// From spec cpu.md M6 and load.md:
/// - Creates LOAD operation for the load instruction
/// - Creates MEMW operation for the memory read
pub fn collect_load_from_cpu(ctx: &mut CollectionContext) {
    for (i, cpu_op) in ctx.cpu_ops.iter().enumerate() {
        if !cpu_op.op_load {
            continue;
        }

        let log = &ctx.logs[i];
        let timestamp = cpu_op.timestamp;
        let instruction = ctx.instructions.get(&log.current_pc).copied().unwrap();

        if let Instruction::Load { offset, width, .. } = instruction {
            let base_address = log.src1_val.wrapping_add(offset as i64 as u64);
            let (byte_count, signed) = width_to_bytes_and_signed(width);
            let loaded_value = log.dst_val;

            // Read old timestamps from memory state
            let (_old_values, old_timestamps) = ctx.memory_state.read_bytes(base_address, 8);

            // Extract individual bytes from loaded value
            let mut value_bytes = [0u64; 8];
            for (j, byte) in value_bytes.iter_mut().take(byte_count).enumerate() {
                *byte = (loaded_value >> (j * 8)) & 0xFF;
            }

            // Sign/zero extend the upper bytes
            let mut res_bytes = value_bytes;
            if byte_count < 8 {
                let msb = value_bytes[byte_count - 1];
                let sign_bit = (msb >> 7) & 1;
                let fill = if signed && sign_bit == 1 { 0xFF } else { 0 };
                for byte in res_bytes.iter_mut().skip(byte_count) {
                    *byte = fill;
                }
            }

            // Create MEMW operation (memory read)
            let memw_op = MemwOperation::new(
                false,      // is_register = false (memory)
                base_address,
                res_bytes,
                timestamp,
                byte_count as u8,
                true,       // is_read = true
            ).with_old(res_bytes, old_timestamps);
            ctx.memw_ops.push(memw_op);

            // Create LOAD operation
            let load_op = LoadOperation::new(
                base_address,
                timestamp,
                byte_count as u8,
                signed,
                res_bytes,
            );
            ctx.load_ops.push(load_op);

            // Update memory state
            ctx.memory_state.write_bytes(base_address, loaded_value, byte_count, timestamp);
        }
    }
}

// TODO: Add these functions when implementing the tables:
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
// Phase 3: LOAD → MEMW
// =============================================================================
// Note: LOAD→MEMW is handled within collect_load_from_cpu() since we create
// the MEMW read operation at the same time as the LOAD operation.

// =============================================================================
// Phase 4: MEMW → LT (timestamp ordering)
// =============================================================================

/// Collects LT operations from MEMW for timestamp ordering verification.
///
/// From spec memw.md constraints 7-10:
/// - Constraint 7: LT[1; old_timestamp[0], timestamp, 0] with μ_sum
/// - Constraint 8: LT[1; old_timestamp[1], timestamp, 0] with w2
/// - Constraint 9: LT[1; old_timestamp[i], timestamp, 0] for i ∈ [2,3] with w4
/// - Constraint 10: LT[1; old_timestamp[i], timestamp, 0] for i ∈ [4,7] with write8
///
/// Each LT operation verifies old_timestamp < timestamp (result must be 1).
pub fn collect_lt_from_memw(ctx: &mut CollectionContext) {
    for memw_op in &ctx.memw_ops {
        let timestamp = memw_op.timestamp;

        // All MEMW operations are active (either read or write)
        // Determine write flags based on width
        let width = memw_op.width;
        let w2 = width >= 2; // write2 + write4 + write8
        let w4 = width >= 4; // write4 + write8
        let w8 = width >= 8; // write8

        // Constraint 7: LT for old_timestamp[0] (always, with μ_sum)
        ctx.lt_ops.push(LtOperation::new(
            memw_op.old_timestamp[0],
            timestamp,
            false, // unsigned
        ));

        // Constraint 8: LT for old_timestamp[1] (with w2)
        if w2 {
            ctx.lt_ops.push(LtOperation::new(
                memw_op.old_timestamp[1],
                timestamp,
                false,
            ));
        }

        // Constraint 9: LT for old_timestamp[2..4] (with w4)
        if w4 {
            for i in 2..4 {
                ctx.lt_ops.push(LtOperation::new(
                    memw_op.old_timestamp[i],
                    timestamp,
                    false,
                ));
            }
        }

        // Constraint 10: LT for old_timestamp[4..8] (with w8)
        if w8 {
            for i in 4..8 {
                ctx.lt_ops.push(LtOperation::new(
                    memw_op.old_timestamp[i],
                    timestamp,
                    false,
                ));
            }
        }
    }
}

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

/// Collects bitwise lookups from LOAD (MSB8 for sign bit extraction).
///
/// From spec load.md constraints 3-5: MSB8[sign_bit; res[k]]
pub fn collect_bitwise_from_load(ctx: &mut CollectionContext) {
    for load_op in &ctx.load_ops {
        ctx.bitwise_lookups.extend(load_op.collect_bitwise_lookups());
    }
}

/// Collects bitwise lookups from MEMW (IS_HALFWORD for address_add).
///
/// From spec memw.md constraint 6: IS_HALFWORD[address_add[i][j]]
/// Each address_add[i] (i=0..7) is a 64-bit address stored as 4 halfwords.
/// We need to range check each halfword to be in [0, 2^16).
pub fn collect_bitwise_from_memw(ctx: &mut CollectionContext) {
    for memw_op in &ctx.memw_ops {
        // Compute address_add[i] = base_address + (i+1) for i in 0..7
        for i in 0..7 {
            let addr = memw_op.base_address.wrapping_add((i + 1) as u64);

            // Split into 4 halfwords (DWordHL format)
            let h0 = (addr & 0xFFFF) as u16;
            let h1 = ((addr >> 16) & 0xFFFF) as u16;
            let h2 = ((addr >> 32) & 0xFFFF) as u16;
            let h3 = ((addr >> 48) & 0xFFFF) as u16;

            // Add IS_HALFWORD lookups for each halfword
            for h in [h0, h1, h2, h3] {
                ctx.bitwise_lookups.push((
                    BitwiseLookup::IsHalf,
                    (h & 0xFF) as u8,
                    ((h >> 8) & 0xFF) as u8,
                    0,
                ));
            }
        }
    }
}
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

    /// MEMW memory/register read/write trace
    pub memw: TraceTable<GoldilocksField, GoldilocksExtension>,

    /// LOAD memory load with extension trace
    pub load: TraceTable<GoldilocksField, GoldilocksExtension>,

    // TODO: Add these fields when implementing the tables:
    // pub shift: TraceTable<GoldilocksField, GoldilocksExtension>,
    // pub mul: TraceTable<GoldilocksField, GoldilocksExtension>,
    // pub branch: TraceTable<GoldilocksField, GoldilocksExtension>,
}

impl Traces {
    /// Generates all traces from execution logs using phased collection.
    ///
    /// The phases are:
    /// 1. Logs → CPU operations
    /// 2. CPU → LT operations, Bitwise lookups, MEMW ops, LOAD ops
    /// 3. LT → Bitwise lookups
    /// 4. Generate all traces: CPU, LT, MEMW, LOAD, BITWISE
    ///
    /// Future phases will add SHIFT, MUL, BRANCH tables.
    pub fn from_logs(
        logs: &[Log],
        instructions: U64HashMap<Instruction>,
    ) -> Result<Self, ProverError> {
        // Clone instructions for CPU collection (needed before ctx takes ownership)
        let instructions_for_cpu = instructions.clone();
        let mut ctx = CollectionContext::new(logs.len(), instructions, logs.to_vec());

        // =====================================================================
        // PHASE 1: Logs → CPU
        // =====================================================================
        collect_cpu_from_logs(&mut ctx, logs, &instructions_for_cpu)?;

        // =====================================================================
        // PHASE 2: CPU → Secondary tables
        // Order within phase doesn't matter (all read from cpu_ops)
        // =====================================================================
        collect_lt_from_cpu(&mut ctx);
        collect_bitwise_from_cpu(&mut ctx);
        collect_memw_from_cpu(&mut ctx);
        collect_load_from_cpu(&mut ctx);

        // TODO: Uncomment as you implement:
        // collect_shift_from_cpu(&mut ctx);
        // collect_mul_from_cpu(&mut ctx);
        // collect_branch_from_cpu(&mut ctx);

        // =====================================================================
        // PHASE 3: LOAD → MEMW
        // =====================================================================
        // Note: LOAD→MEMW is handled within collect_load_from_cpu()

        // =====================================================================
        // PHASE 4: MEMW → LT (timestamp ordering)
        // =====================================================================
        collect_lt_from_memw(&mut ctx);

        // =====================================================================
        // PHASE 5: All tables → Bitwise lookups
        // =====================================================================
        collect_bitwise_from_lt(&mut ctx);
        collect_bitwise_from_load(&mut ctx);
        collect_bitwise_from_memw(&mut ctx);
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

        // Generate MEMW trace
        let memw = memw::generate_memw_trace(&self.memw_ops);

        // Generate LOAD trace
        let load = load::generate_load_trace(&self.load_ops);

        // Generate BITWISE trace and update multiplicities
        let mut bitwise = bitwise::generate_bitwise_trace();
        bitwise::update_multiplicities(&mut bitwise, &self.bitwise_lookups);

        Ok(Traces {
            cpu,
            bitwise,
            lt,
            memw,
            load,
        })
    }
}
