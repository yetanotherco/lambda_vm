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
//! PHASE 2: CPU ops → MEMW, LOAD, LT, Bitwise (with state tracking for MEMW/LOAD)
//! PHASE 3: MEMW → LT ops (timestamp ordering, overflow checks)
//! PHASE 4: LT, MEMW → Bitwise lookups
//! PHASE 5: Generate all traces
//! ```
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

use executor::vm::instruction::decoding::Instruction;
use executor::vm::logs::Log;
use executor::vm::memory::U64HashMap;
use stark::trace::TraceTable;

use super::bitwise::{self, BitwiseOperation, BitwiseOperationType};
use super::cpu::{self, CpuOperation};
use super::decode;
use super::halt;
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

// =============================================================================
// Helper Functions
// =============================================================================

/// Get byte count and signed flag from CpuOperation memory flags.
fn cpu_op_to_bytes_and_signed(op: &CpuOperation) -> (usize, bool) {
    let byte_count = if op.decode.memory_8bytes {
        8
    } else if op.decode.memory_4bytes {
        4
    } else if op.decode.memory_2bytes {
        2
    } else {
        1
    };
    (byte_count, op.decode.signed)
}

/// Pack a 64-bit register value into the MEMW value format.
///
/// For register operations, values are packed as [lo32, hi32, 0, 0, 0, 0, 0, 0].
fn pack_register_value(value: u64) -> [u64; 8] {
    [value & 0xFFFF_FFFF, value >> 32, 0, 0, 0, 0, 0, 0]
}

// =============================================================================
// Phase 1: Logs → CPU ops
// =============================================================================

/// Collects CPU operations from execution logs.
///
/// Returns a vector of CpuOperation, one per log entry.
fn collect_cpu_ops(
    logs: &[Log],
    instructions: &U64HashMap<Instruction>,
) -> Result<Vec<CpuOperation>, ProverError> {
    let mut cpu_ops = Vec::with_capacity(logs.len());

    // Timestamps start at 4 (not 0) to ensure old_timestamp < timestamp holds
    // for the first access to any register/memory location (where old_timestamp=0).
    for (i, log) in logs.iter().enumerate() {
        let timestamp = (i as u64) * 4 + 4;
        let instruction = instructions
            .get(&log.current_pc)
            .copied()
            .ok_or(ProverError::MissingInstruction(log.current_pc))?;

        let op = CpuOperation::from_log_and_instruction(log, timestamp, instruction);
        cpu_ops.push(op);
    }
    Ok(cpu_ops)
}

// =============================================================================
// Phase 2: CPU ops → MEMW, LOAD, LT, Bitwise
// =============================================================================

/// Collects all derived operations from CPU operations in a single pass.
///
/// This includes:
/// - MEMW ops (register reads/writes M1/M3/M5, memory loads/stores M6/M7)
/// - LOAD ops (memory loads with sign/zero extension)
/// - LT ops (from SLT/BLT instructions)
/// - Bitwise lookups (from CPU operations)
///
/// MEMW and LOAD collection requires sequential processing with state tracking.
///
/// Returns: (memw_ops, load_ops, lt_ops, bitwise_ops)
fn collect_ops_from_cpu(
    cpu_ops: &[CpuOperation],
    memory_state: &mut MemoryState,
    register_state: &mut RegisterState,
) -> (
    Vec<MemwOperation>,
    Vec<LoadOperation>,
    Vec<LtOperation>,
    Vec<BitwiseOperation>,
) {
    let mut memw_ops = Vec::with_capacity(cpu_ops.len() * 3);
    let mut load_ops = Vec::with_capacity(cpu_ops.len() / 8 + 1);
    let mut lt_ops = Vec::with_capacity(cpu_ops.len() / 10 + 1);
    let mut bitwise_ops = Vec::with_capacity(cpu_ops.len() * 4);

    for op in cpu_ops {
        // --- MEMW and LOAD (require state tracking, order matters) ---

        // Collect memory operations for Load/Store instructions
        if op.decode.op_load {
            let (memw_op, load_op, lookups) = collect_load_op_from_cpu(op, memory_state);
            memw_ops.push(memw_op);
            load_ops.push(load_op);
            bitwise_ops.extend(lookups);
        } else if op.decode.op_store {
            let memw_op = collect_store_op_from_cpu(op, memory_state);
            memw_ops.push(memw_op);
        }

        // Collect register operations (M1, M3, M5)
        let reg_memw_ops = collect_register_ops_from_cpu(op, register_state);
        memw_ops.extend(reg_memw_ops);

        // --- LT and Bitwise (no state tracking needed) ---

        // Collect LT operations from SLT/BLT instructions
        if op.decode.op_slt || op.decode.op_blt {
            let arg1 = op.compute_arg1();
            let arg2 = op.compute_arg2();
            lt_ops.push(LtOperation::new(arg1, arg2, op.decode.signed));
        }

        // Collect bitwise lookups
        bitwise_ops.extend(op.collect_bitwise_ops());
    }

    (memw_ops, load_ops, lt_ops, bitwise_ops)
}

/// Collects a LOAD operation and corresponding MEMW read from CpuOperation.
///
/// Returns: (memw_op, load_op, bitwise_ops)
fn collect_load_op_from_cpu(
    op: &CpuOperation,
    memory_state: &mut MemoryState,
) -> (MemwOperation, LoadOperation, Vec<BitwiseOperation>) {
    // res contains the effective address (base + offset)
    let base_address = op.res;
    let (byte_count, signed) = cpu_op_to_bytes_and_signed(op);
    // rvd contains the loaded value
    let loaded_value = op.rvd;

    // Read old timestamps from memory state
    let (_old_values, old_timestamps) = memory_state.read_bytes(base_address, 8);

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

    // Create MEMW operation (read)
    let memw_op = MemwOperation::new(
        false, // is_register = false
        base_address,
        res_bytes,
        op.timestamp,
        byte_count as u8,
        true, // is_read = true
    )
    .with_old(res_bytes, old_timestamps);

    // Create LOAD operation
    let load_op = LoadOperation::new(
        base_address,
        op.timestamp,
        byte_count as u8,
        signed,
        res_bytes,
    );

    // Collect MSB8 lookups for sign bit extraction
    let bitwise_ops = load_op.collect_bitwise_ops();

    // Update memory state
    memory_state.write_bytes(base_address, loaded_value, byte_count, op.timestamp);

    (memw_op, load_op, bitwise_ops)
}

/// Collects a STORE operation as a MEMW write from CpuOperation.
///
/// Returns: memw_op
fn collect_store_op_from_cpu(op: &CpuOperation, memory_state: &mut MemoryState) -> MemwOperation {
    // res contains the effective address (base + offset)
    let base_address = op.res;
    let (byte_count, _) = cpu_op_to_bytes_and_signed(op);
    // rv2 contains the store value
    let store_value = op.rv2;

    // Read old values and timestamps
    let (old_values, old_timestamps) = memory_state.read_bytes(base_address, 8);

    // Pack store value as [lo32, hi32, 0, 0, 0, 0, 0, 0] to match CPU M7
    let value_bytes = [
        store_value & 0xFFFF_FFFF,
        store_value >> 32,
        0,
        0,
        0,
        0,
        0,
        0,
    ];

    // Create MEMW operation (write) - M7 uses timestamp+1
    let memw_op = MemwOperation::new(
        false, // is_register = false
        base_address,
        value_bytes,
        op.timestamp + 1,
        byte_count as u8,
        false, // is_read = false (write)
    )
    .with_old(old_values, old_timestamps);

    // Update memory state (using timestamp+1 to match M7)
    memory_state.write_bytes(base_address, store_value, byte_count, op.timestamp + 1);

    memw_op
}

/// Collects register read/write operations (M1, M3, M5) from CpuOperation.
///
/// Returns: Vec of MEMW operations for register accesses
fn collect_register_ops_from_cpu(
    op: &CpuOperation,
    register_state: &mut RegisterState,
) -> Vec<MemwOperation> {
    let mut memw_ops = Vec::with_capacity(3);
    let d = &op.decode;

    // M1: Read rs1 register at timestamp+0
    // Skip x0 (hardwired zero) and x255 (virtual PC register for AUIPC/JAL)
    if d.read_register1 && d.rs1 != 0 && d.rs1 != 255 {
        let reg_value = pack_register_value(op.rv1);
        let reg_addr = 2 * d.rs1 as u64;
        let (_old_val, old_ts) = register_state.read(d.rs1);
        let old_timestamps = [old_ts; 8];

        let memw_op = MemwOperation::new(true, reg_addr, reg_value, op.timestamp, 8, true)
            .with_old(reg_value, old_timestamps);
        memw_ops.push(memw_op);
        register_state.write(d.rs1, op.rv1, op.timestamp);
    }

    // M3: Read rs2 register at timestamp+1
    if d.read_register2 && d.rs2 != 0 {
        let reg_value = pack_register_value(op.rv2);
        let reg_addr = 2 * d.rs2 as u64;
        let (_old_val, old_ts) = register_state.read(d.rs2);
        let old_timestamps = [old_ts; 8];

        let memw_op = MemwOperation::new(true, reg_addr, reg_value, op.timestamp + 1, 8, true)
            .with_old(reg_value, old_timestamps);
        memw_ops.push(memw_op);
        register_state.write(d.rs2, op.rv2, op.timestamp + 1);
    }

    // M5: Write rd register at timestamp+2
    if d.write_register && d.rd != 0 {
        let reg_value = pack_register_value(op.rvd);
        let reg_addr = 2 * d.rd as u64;
        let (old_val, old_ts) = register_state.read(d.rd);
        let old_value = pack_register_value(old_val);
        let old_timestamps = [old_ts; 8];

        let memw_op = MemwOperation::new(true, reg_addr, reg_value, op.timestamp + 2, 8, false)
            .with_old(old_value, old_timestamps);
        memw_ops.push(memw_op);
        register_state.write(d.rd, op.rvd, op.timestamp + 2);
    }

    memw_ops
}

// =============================================================================
// Phase 3: MEMW → LT
// =============================================================================

/// Collects LT operations from MEMW for timestamp ordering and overflow checks.
///
/// From spec memw.md:
/// - C7-C10: old_timestamp[i] < timestamp (based on width)
/// - R1-R3: base_address < base_address + offset (overflow checks)
///
/// Returns: Vec of LT operations
fn collect_lt_from_memw(memw_ops: &[MemwOperation]) -> Vec<LtOperation> {
    let mut lt_ops = Vec::with_capacity(memw_ops.len() * 8);

    for memw_op in memw_ops {
        // C7: old_timestamp[0] < timestamp (all accesses)
        lt_ops.push(LtOperation::new(
            memw_op.old_timestamp[0],
            memw_op.timestamp,
            false,
        ));

        // C8: old_timestamp[1] < timestamp (width >= 2)
        if memw_op.width >= 2 {
            lt_ops.push(LtOperation::new(
                memw_op.old_timestamp[1],
                memw_op.timestamp,
                false,
            ));
        }

        // C9: old_timestamp[2,3] < timestamp (width >= 4)
        if memw_op.width >= 4 {
            lt_ops.push(LtOperation::new(
                memw_op.old_timestamp[2],
                memw_op.timestamp,
                false,
            ));
            lt_ops.push(LtOperation::new(
                memw_op.old_timestamp[3],
                memw_op.timestamp,
                false,
            ));
        }

        // C10: old_timestamp[4..7] < timestamp (width == 8)
        if memw_op.width == 8 {
            for i in 4..8 {
                lt_ops.push(LtOperation::new(
                    memw_op.old_timestamp[i],
                    memw_op.timestamp,
                    false,
                ));
            }
        }

        // R1-R3: Address overflow checks
        if memw_op.width == 2 {
            let addr_plus_1 = memw_op.base_address.wrapping_add(1);
            if addr_plus_1 > memw_op.base_address {
                lt_ops.push(LtOperation::new(memw_op.base_address, addr_plus_1, false));
            }
        }
        if memw_op.width == 4 {
            let addr_plus_3 = memw_op.base_address.wrapping_add(3);
            if addr_plus_3 > memw_op.base_address {
                lt_ops.push(LtOperation::new(memw_op.base_address, addr_plus_3, false));
            }
        }
        if memw_op.width == 8 {
            let addr_plus_7 = memw_op.base_address.wrapping_add(7);
            if addr_plus_7 > memw_op.base_address {
                lt_ops.push(LtOperation::new(memw_op.base_address, addr_plus_7, false));
            }
        }
    }

    lt_ops
}

// =============================================================================
// Phase 4: All → Bitwise lookups
// =============================================================================

/// Collects bitwise lookups from LT operations (MSB16 and IS_HALFWORD).
///
/// Returns: Vec of bitwise lookups
fn collect_bitwise_from_lt(lt_ops: &[LtOperation]) -> Vec<BitwiseOperation> {
    let mut bitwise_ops = Vec::with_capacity(lt_ops.len() * 8);

    for op in lt_ops {
        // MSB16 lookups for lhs[2] and rhs[2]
        let lhs_2 = ((op.lhs >> 48) & 0xFFFF) as u16;
        let rhs_2 = ((op.rhs >> 48) & 0xFFFF) as u16;

        bitwise_ops.push(BitwiseOperation::halfword(
            BitwiseOperationType::Msb16,
            (lhs_2 & 0xFF) as u8,
            (lhs_2 >> 8) as u8,
        ));
        bitwise_ops.push(BitwiseOperation::halfword(
            BitwiseOperationType::Msb16,
            (rhs_2 & 0xFF) as u8,
            (rhs_2 >> 8) as u8,
        ));

        // IS_HALFWORD lookups for lhs_sub_rhs[0..4]
        let lhs_sub_rhs = op.lhs.wrapping_sub(op.rhs);
        for shift in [0, 16, 32, 48] {
            let half = ((lhs_sub_rhs >> shift) & 0xFFFF) as u16;
            bitwise_ops.push(BitwiseOperation::halfword(
                BitwiseOperationType::IsHalf,
                (half & 0xFF) as u8,
                (half >> 8) as u8,
            ));
        }

        // IS_HALFWORD lookups for lhs[1] and rhs[1]
        let lhs_1 = ((op.lhs >> 32) & 0xFFFF) as u16;
        let rhs_1 = ((op.rhs >> 32) & 0xFFFF) as u16;
        bitwise_ops.push(BitwiseOperation::halfword(
            BitwiseOperationType::IsHalf,
            (lhs_1 & 0xFF) as u8,
            (lhs_1 >> 8) as u8,
        ));
        bitwise_ops.push(BitwiseOperation::halfword(
            BitwiseOperationType::IsHalf,
            (rhs_1 & 0xFF) as u8,
            (rhs_1 >> 8) as u8,
        ));
    }

    bitwise_ops
}

/// Collects IS_HALFWORD lookups from MEMW address_add columns.
///
/// Returns: Vec of bitwise lookups
fn collect_bitwise_from_memw(memw_ops: &[MemwOperation]) -> Vec<BitwiseOperation> {
    let mut bitwise_ops = Vec::with_capacity(memw_ops.len() * 28); // 7 addresses * 4 halfwords

    for memw_op in memw_ops {
        for i in 0..7u64 {
            let addr_add = memw_op.base_address.wrapping_add(i + 1);
            // Extract 4 halfwords (DWordHL packing)
            for shift in [0, 16, 32, 48] {
                let half = ((addr_add >> shift) & 0xFFFF) as u16;
                bitwise_ops.push(BitwiseOperation::halfword(
                    BitwiseOperationType::IsHalf,
                    (half & 0xFF) as u8,
                    (half >> 8) as u8,
                ));
            }
        }
    }

    bitwise_ops
}

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

    /// DECODE instruction decoding table
    pub decode: TraceTable<GoldilocksField, GoldilocksExtension>,

    /// HALT single-row table for program termination
    pub halt: TraceTable<GoldilocksField, GoldilocksExtension>,
}

impl Traces {
    /// Generates all traces from execution logs using phased collection.
    ///
    /// The phases are:
    /// 1. Logs → CPU operations
    /// 2. CPU ops → MEMW, LOAD, LT, Bitwise (state tracking for MEMW/LOAD)
    /// 3. MEMW → LT operations (timestamp ordering)
    /// 4. LT, MEMW → Bitwise lookups
    /// 5. Generate all traces
    pub fn from_logs(
        logs: &[Log],
        instructions: U64HashMap<Instruction>,
    ) -> Result<Self, ProverError> {
        // =====================================================================
        // PHASE 1: Logs → CPU operations
        // =====================================================================
        let cpu_ops = collect_cpu_ops(logs, &instructions)?;

        // =====================================================================
        // PHASE 2: CPU ops → MEMW, LOAD, LT, Bitwise
        // =====================================================================
        // Processes cpu_ops in order. MEMW/LOAD need state tracking, LT/Bitwise don't.
        let mut memory_state = MemoryState::new();
        let mut register_state = RegisterState::new();
        let (memw_ops, load_ops, mut lt_ops, mut bitwise_ops) =
            collect_ops_from_cpu(&cpu_ops, &mut memory_state, &mut register_state);

        // =====================================================================
        // PHASE 3: MEMW → LT (timestamp ordering and overflow checks)
        // =====================================================================
        lt_ops.extend(collect_lt_from_memw(&memw_ops));

        // =====================================================================
        // PHASE 4: All → Bitwise lookups
        // =====================================================================
        bitwise_ops.extend(collect_bitwise_from_lt(&lt_ops));
        bitwise_ops.extend(collect_bitwise_from_memw(&memw_ops));

        // =====================================================================
        // PHASE 5: Generate final traces
        // =====================================================================

        // Extract halt timestamp from the last ECALL instruction
        let halt_op = cpu_ops
            .iter()
            .rev()
            .find(|op| op.decode.op_ecall)
            .ok_or(ProverError::MissingEcall)?;
        let halt_trace = halt::generate_halt_trace(halt_op.timestamp);

        let cpu = cpu::generate_cpu_trace(&cpu_ops);
        let lt = lt::generate_lt_trace(&lt_ops);
        let memw = memw::generate_memw_trace(&memw_ops);
        let load = load::generate_load_trace(&load_ops);

        let mut bitwise = bitwise::generate_bitwise_trace();
        bitwise::update_multiplicities(&mut bitwise, &bitwise_ops);

        // Generate DECODE trace and update multiplicities
        // Each CPU operation looks up the DECODE table once
        // Padding rows also look up pc=1 (the CPU padding entry)
        let (mut decode, pc_to_row) = decode::generate_decode_trace(&instructions);
        let num_padding_rows = cpu_ops.len().next_power_of_two() - cpu_ops.len();
        let mut decode_lookups: Vec<u64> = cpu_ops.iter().map(|op| op.decode.pc).collect();
        decode_lookups.extend(std::iter::repeat_n(cpu::CPU_PADDING_PC, num_padding_rows));
        decode::update_multiplicities(&mut decode, &pc_to_row, &decode_lookups);

        Ok(Traces {
            cpu,
            bitwise,
            lt,
            memw,
            load,
            decode,
            halt: halt_trace,
        })
    }

    /// Generates all traces with a trimmed bitwise table (TEST ONLY).
    ///
    /// # WARNING: UNSOUND FOR PRODUCTION
    ///
    /// This function generates the full 2^20 row bitwise table, updates multiplicities,
    /// then removes rows where all multiplicity columns are zero. This is **unsound**
    /// because:
    ///
    /// 1. The bitwise table is NOT preprocessed - the verifier checks the prover's
    ///    commitment instead of a hardcoded trusted commitment
    /// 2. A malicious prover could provide incorrect bitwise results and the
    ///    verifier would accept them (e.g., claim 5 AND 3 = 7)
    /// 3. The table structure differs from production (row indices don't match)
    ///
    /// This is acceptable for tests because we're testing:
    /// - Bus interaction balancing (sends = receives)
    /// - Constraint satisfaction
    /// - LogUp protocol correctness
    ///
    /// The full preprocessed bitwise verification is tested separately in the
    /// comprehensive `test_prove_elfs_all_instructions_64_full` test.
    #[cfg(test)]
    pub fn from_logs_trimmed(
        logs: &[Log],
        instructions: U64HashMap<Instruction>,
    ) -> Result<Self, ProverError> {
        // Generate full traces (including full 2^20 bitwise table with multiplicities)
        let mut traces = Self::from_logs(logs, instructions)?;

        // Trim the bitwise table to only rows with non-zero multiplicities
        traces.bitwise = bitwise::trim_zero_rows(traces.bitwise);

        Ok(traces)
    }

    /// Generates all traces with a minimal bitwise table (TEST ONLY).
    ///
    /// Alias for `from_logs_trimmed` for backwards compatibility.
    #[cfg(test)]
    pub fn from_logs_minimal(
        logs: &[Log],
        instructions: U64HashMap<Instruction>,
    ) -> Result<Self, ProverError> {
        Self::from_logs_trimmed(logs, instructions)
    }
}
