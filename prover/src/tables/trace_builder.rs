//! Trace generation from execution logs.
//!
//! This module provides a single entry point for generating all trace tables
//! from execution logs in a single pass.
//!
//! ## Usage
//!
//! ```ignore
//! use prover::tables::trace_builder::Traces;
//!
//! let traces = Traces::from_logs(&logs)?;
//! // Use traces.cpu, traces.bitwise, traces.lt, traces.memw, traces.load
//! ```

use std::collections::HashMap;

use executor::vm::instruction::decoding::{Instruction, LoadStoreWidth};
use executor::vm::logs::Log;
use executor::vm::memory::U64HashMap;
use stark::trace::TraceTable;

use super::bitwise;
use super::cpu::{self, CpuOperation};
use super::load::{self, LoadOperation};
use super::lt::{self, LtOperation};
use super::memw::{self, MemwOperation};
use super::types::{GoldilocksExtension, GoldilocksField};
use crate::ProverError;

/// Memory cell state: (value_byte, last_write_timestamp)
type MemoryCell = (u8, u64);

/// Register state: (value, last_write_timestamp)
/// Registers are 64-bit values, timestamp is for the whole register.
type RegisterCell = (u64, u64);

/// Memory state tracker for generating MEMW/LOAD traces.
struct MemoryState {
    /// Map from byte address to (value, timestamp)
    cells: HashMap<u64, MemoryCell>,
}

/// Register state tracker for generating MEMW register traces.
/// Tracks the last write timestamp for each register (0-31).
struct RegisterState {
    /// Register file: (value, last_write_timestamp)
    /// Index by register number (0-31)
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

/// Convert LoadStoreWidth to byte count and signed flag.
fn width_to_bytes_and_signed(width: LoadStoreWidth) -> (usize, bool) {
    match width {
        LoadStoreWidth::Byte => (1, true),
        LoadStoreWidth::ByteUnsigned => (1, false),
        LoadStoreWidth::Half => (2, true),
        LoadStoreWidth::HalfUnsigned => (2, false),
        LoadStoreWidth::Word => (4, true),
        LoadStoreWidth::WordUnsigned => (4, false),
        LoadStoreWidth::DoubleWord => (8, false), // 64-bit loads are never sign-extended
    }
}

/// Pack a 64-bit register value into the MEMW value format.
///
/// For register operations, values are packed as [lo32, hi32, 0, 0, 0, 0, 0, 0].
/// This matches the CPU bus interaction which sends rv1/rv2/rvd as 2 words + 6 zeros.
fn pack_register_value(value: u64) -> [u64; 8] {
    [
        value & 0xFFFF_FFFF,  // lo32
        value >> 32,          // hi32
        0, 0, 0, 0, 0, 0,     // remaining 6 elements are unconstrained (zeros)
    ]
}

/// All generated trace tables.
pub struct Traces {
    /// CPU execution trace (one row per instruction)
    pub cpu: TraceTable<GoldilocksField, GoldilocksExtension>,

    /// BITWISE precomputed lookup table (2^20 rows)
    pub bitwise: TraceTable<GoldilocksField, GoldilocksExtension>,

    /// LT comparison trace (deduplicated operations)
    pub lt: TraceTable<GoldilocksField, GoldilocksExtension>,

    /// MEMW memory read/write trace
    pub memw: TraceTable<GoldilocksField, GoldilocksExtension>,

    /// LOAD memory load with extension trace
    pub load: TraceTable<GoldilocksField, GoldilocksExtension>,
}

impl Traces {
    /// Generates all traces from execution logs in a single pass.
    pub fn from_logs(
        logs: &[Log],
        instructions: U64HashMap<Instruction>,
    ) -> Result<Self, ProverError> {
        // Pre-allocate collectors
        let mut cpu_ops = Vec::with_capacity(logs.len());
        let mut bitwise_lookups = Vec::with_capacity(logs.len() * 4);
        let mut lt_ops = Vec::with_capacity(logs.len() / 10 + 1);
        let mut memw_ops = Vec::with_capacity(logs.len() / 4 + 1);
        let mut load_ops = Vec::with_capacity(logs.len() / 8 + 1);

        // Memory state tracker
        let mut memory_state = MemoryState::new();

        // Register state tracker for old_timestamp values
        let mut register_state = RegisterState::new();

        // Single pass over logs
        for (i, log) in logs.iter().enumerate() {
            let timestamp = (i as u64) * 4;
            let instruction = instructions
                .get(&log.current_pc)
                .copied()
                .ok_or(ProverError::MissingInstruction(log.current_pc))?;
            let op = CpuOperation::from_log(log, timestamp, instruction);

            // Collect bitwise lookups from this operation
            bitwise_lookups.extend(op.collect_bitwise_lookups());

            // Collect LT operations for SLT and BLT instructions
            if op.op_slt || op.op_blt {
                let arg1 = op.compute_arg1();
                let arg2 = op.compute_arg2();
                lt_ops.push(LtOperation::new(arg1, arg2, op.signed));
            }

            // Collect memory operations for Load/Store instructions
            match instruction {
                Instruction::Load { offset, width, .. } => {
                    // effective_address = base_register_value + offset
                    let base_address = log.src1_val.wrapping_add(offset as i64 as u64);
                    let (byte_count, signed) = width_to_bytes_and_signed(width);
                    let loaded_value = log.dst_val;

                    // Read old values and timestamps from memory state
                    let (old_values, old_timestamps) = memory_state.read_bytes(base_address, 8);

                    // Extract individual bytes from loaded value
                    let mut value_bytes = [0u64; 8];
                    for (j, byte) in value_bytes.iter_mut().take(byte_count).enumerate() {
                        *byte = (loaded_value >> (j * 8)) & 0xFF;
                    }

                    // Sign/zero extend the upper bytes for LOAD result
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
                    // Use res_bytes (sign-extended) to match what LOAD→MEMW sends
                    let memw_op = MemwOperation::new(
                        false, // is_register = false (memory access)
                        base_address,
                        res_bytes,
                        timestamp,
                        byte_count as u8,
                        true, // is_read = true
                    )
                    .with_old(old_values, old_timestamps);
                    memw_ops.push(memw_op);

                    // Create LOAD operation
                    let load_op = LoadOperation::new(
                        base_address,
                        timestamp,
                        byte_count as u8,
                        signed,
                        res_bytes,
                    );
                    load_ops.push(load_op);

                    // Update memory state (reads still update timestamp for consistency)
                    memory_state.write_bytes(base_address, loaded_value, byte_count, timestamp);
                }
                Instruction::Store { offset, width, .. } => {
                    // effective_address = base_register_value + offset
                    let base_address = log.src1_val.wrapping_add(offset as i64 as u64);
                    let (byte_count, _) = width_to_bytes_and_signed(width);
                    let store_value = log.src2_val;

                    // Read old values and timestamps from memory state
                    let (old_values, old_timestamps) = memory_state.read_bytes(base_address, 8);

                    // Pack store value as [lo32, hi32, 0, 0, 0, 0, 0, 0] to match CPU M7
                    // CPU sends rv2 as packed words (can't decompose to bytes in bus interaction)
                    let value_bytes = [
                        store_value & 0xFFFF_FFFF,  // lo32
                        store_value >> 32,          // hi32
                        0, 0, 0, 0, 0, 0,
                    ];

                    // Create MEMW operation (write)
                    // M7 uses timestamp+1 (M1=ts+0, M3=ts+1, M5=ts+2)
                    let memw_op = MemwOperation::new(
                        false, // is_register = false (memory access)
                        base_address,
                        value_bytes,
                        timestamp + 1,
                        byte_count as u8,
                        false, // is_read = false (write)
                    )
                    .with_old(old_values, old_timestamps);
                    memw_ops.push(memw_op);

                    // Update memory state (using timestamp+1 to match M7)
                    memory_state.write_bytes(base_address, store_value, byte_count, timestamp + 1);
                }
                _ => {}
            }

            // =========================================================================
            // Register operations (M1, M3, M5)
            // =========================================================================
            // These MEMW operations handle register file access as part of the memory model.
            // Register addresses are 2 * register_index (0-31 → 0-62).
            // Values are packed as [lo32, hi32, 0, 0, 0, 0, 0, 0].
            //
            // The register_state tracks (value, last_timestamp) for each register.
            // For the MEMW internal Memory bus to balance:
            // - old_timestamp must be the timestamp of the last operation on this register
            // - old_value must be the value at that timestamp

            // M1: Read rs1 register at timestamp+0
            if op.read_register1 && op.rs1 != 0 {
                let reg_value = pack_register_value(log.src1_val);
                let reg_addr = 2 * op.rs1 as u64;
                // Get old_timestamp from register state (when this register was last accessed)
                let (old_val, old_ts) = register_state.read(op.rs1);
                let old_value = pack_register_value(old_val);
                let old_timestamps = [old_ts; 8];
                let memw_op = MemwOperation::new(
                    true,        // is_register = true
                    reg_addr,    // base_address = 2 * rs1
                    reg_value,   // value
                    timestamp,   // timestamp + 0
                    8,           // width = 8 (full 64-bit register)
                    true,        // is_read = true
                )
                .with_old(old_value, old_timestamps);
                memw_ops.push(memw_op);
                // Update register state: value stays same, timestamp updates to current
                register_state.write(op.rs1, log.src1_val, timestamp);
            }

            // M3: Read rs2 register at timestamp+1
            if op.read_register2 && op.rs2 != 0 {
                let reg_value = pack_register_value(log.src2_val);
                let reg_addr = 2 * op.rs2 as u64;
                // Get old_timestamp from register state
                let (old_val, old_ts) = register_state.read(op.rs2);
                let old_value = pack_register_value(old_val);
                let old_timestamps = [old_ts; 8];
                let memw_op = MemwOperation::new(
                    true,            // is_register = true
                    reg_addr,        // base_address = 2 * rs2
                    reg_value,       // value
                    timestamp + 1,   // timestamp + 1
                    8,               // width = 8
                    true,            // is_read = true
                )
                .with_old(old_value, old_timestamps);
                memw_ops.push(memw_op);
                // Update register state: value stays same, timestamp updates to current
                register_state.write(op.rs2, log.src2_val, timestamp + 1);
            }

            // M5: Write rd register at timestamp+2
            if op.write_register && op.rd != 0 {
                let reg_value = pack_register_value(log.dst_val);
                let reg_addr = 2 * op.rd as u64;
                // Get old value and timestamp from register state
                let (old_val, old_ts) = register_state.read(op.rd);
                let old_value = pack_register_value(old_val);
                let old_timestamps = [old_ts; 8];
                let memw_op = MemwOperation::new(
                    true,            // is_register = true
                    reg_addr,        // base_address = 2 * rd
                    reg_value,       // value = rvd
                    timestamp + 2,   // timestamp + 2
                    8,               // width = 8
                    false,           // is_read = false (write)
                )
                .with_old(old_value, old_timestamps);
                memw_ops.push(memw_op);
                // Update register state with new value and timestamp
                register_state.write(op.rd, log.dst_val, timestamp + 2);
            }

            cpu_ops.push(op);
        }

        // Generate CPU trace (handles padding internally)
        let cpu = cpu::generate_cpu_trace(&cpu_ops);

        // Generate BITWISE trace and update multiplicities
        let mut bitwise = bitwise::generate_bitwise_trace();
        bitwise::update_multiplicities(&mut bitwise, &bitwise_lookups);

        // Generate LT trace (handles deduplication and padding internally)
        let lt = lt::generate_lt_trace(&lt_ops);

        // Generate MEMW and LOAD traces
        let memw = memw::generate_memw_trace(&memw_ops);
        let load = load::generate_load_trace(&load_ops);

        Ok(Traces {
            cpu,
            bitwise,
            lt,
            memw,
            load,
        })
    }
}
