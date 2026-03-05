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
//! PHASE 0: ELF → DECODE, MEMORY_INIT (preprocessed tables)
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
//! use lambda_vm_prover::tables::trace_builder::Traces;
//!
//! let traces = Traces::from_elf_and_logs(&elf, &logs)?;
//! // Use traces.cpus, traces.bitwise, traces.lts, traces.memws, traces.loads
//! ```

use std::collections::HashMap;

use executor::elf::Elf;
use executor::vm::instruction::decoding::Instruction;
use executor::vm::logs::Log;
use executor::vm::memory::U64HashMap;
use stark::trace::TraceTable;

use super::bitwise::{self, BitwiseOperation, BitwiseOperationType};
use super::branch::{self, BranchOperation};
use super::commit::{self, CommitOperation};
use super::cpu::{self, CpuOperation};
use super::decode;
use super::dvrm::{self, DvrmOperation};
use super::halt;
use super::load::{self, LoadOperation};
use super::lt::{self, LtOperation};
use super::memw::{self, MemwOperation};
use super::mul::{self, MulOperation};
use super::page::{self, FinalByteState, FinalStateMap, PageConfig};
use super::register::{self, FinalRegisterStateMap, FinalRegisterWordState};
use super::types::{GoldilocksExtension, GoldilocksField};
use crate::Error;

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

    /// Initialize memory state from ELF segments.
    ///
    /// Pre-populates all ELF bytes with timestamp=0 so that when MEMW first
    /// accesses an address, it gets the correct initial value for `old_value`.
    /// This is required for the Memory bus to balance (MEMW-M1 must match PAGE-C3).
    fn from_elf(elf: &Elf) -> Self {
        let mut cells = HashMap::new();
        for segment in &elf.data {
            for (i, &word) in segment.values.iter().enumerate() {
                let word_addr = segment.base_addr.wrapping_add(i as u64 * 4);
                // Split 32-bit word into 4 bytes (little-endian)
                for byte_offset in 0..4u64 {
                    let byte_addr = word_addr.wrapping_add(byte_offset);
                    let byte_value = ((word >> (byte_offset * 8)) & 0xFF) as u8;
                    // Initial state: value from ELF, timestamp=0
                    cells.insert(byte_addr, (byte_value, 0));
                }
            }
        }
        Self { cells }
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
        let mut regs = [(0u64, 0u64); 32];
        // SP (x2) starts at STACK_TOP
        regs[2] = (page::STACK_TOP, 0);
        Self { regs }
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

    /// Generate the final register state map for the REGISTER table.
    ///
    /// Returns a map from register Word address to final (timestamp, value).
    /// Each register uses 2 Word addresses (reg_addr = 2 * reg_idx, then +0, +1).
    fn to_final_state_map(&self) -> FinalRegisterStateMap {
        let mut map = FinalRegisterStateMap::new();

        for reg_idx in 0..32u8 {
            let (value, timestamp) = self.regs[reg_idx as usize];
            let base_addr = register::register_base_address(reg_idx);

            // Each register is stored as 2 Words (32-bit each) in little-endian order
            let value_lo = (value & 0xFFFF_FFFF) as u32;
            let value_hi = (value >> 32) as u32;

            map.insert(
                base_addr,
                FinalRegisterWordState {
                    timestamp,
                    value: value_lo,
                },
            );
            map.insert(
                base_addr + 1,
                FinalRegisterWordState {
                    timestamp,
                    value: value_hi,
                },
            );
        }

        map
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
) -> Result<Vec<CpuOperation>, Error> {
    let mut cpu_ops = Vec::with_capacity(logs.len());

    // Timestamps start at 4 (not 0) to ensure old_timestamp < timestamp holds
    // for the first access to any register/memory location (where old_timestamp=0).
    for (i, log) in logs.iter().enumerate() {
        let timestamp = (i as u64) * 4 + 4;
        let instruction = instructions
            .get(&log.current_pc)
            .copied()
            .ok_or(Error::MissingInstruction(log.current_pc))?;

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

        // Collect COMMIT ECALL memory operations (register reads/writes + byte reads)
        if op.ecall_commit {
            let commit_ops = collect_commit_memw_ops(op, register_state, memory_state);
            memw_ops.extend(commit_ops);
        }

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

    // Pack ALL 8 bytes of store_value into value_bytes.
    // Bus 14: MEMW Memory Write receiver reconstructs lo32/hi32 via linear combination
    //   of all 8 bytes. Must match CPU M7 which sends full rv2 as [lo32, hi32].
    // Bus 16: only positions 0..byte_count participate (controlled by w2/w4/write8
    //   multiplicities), so extra bytes don't affect memory consistency.
    let mut value_bytes = [0u64; 8];
    for (j, byte) in value_bytes.iter_mut().enumerate() {
        *byte = (store_value >> (j * 8)) & 0xFF;
    }

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
        // old_timestamps array is 8 elements but only first 2 are used for registers
        let old_timestamps = [old_ts, old_ts, 0, 0, 0, 0, 0, 0];

        let memw_op = MemwOperation::new(true, reg_addr, reg_value, op.timestamp, 2, true)
            .with_old(reg_value, old_timestamps);
        memw_ops.push(memw_op);
        register_state.write(d.rs1, op.rv1, op.timestamp);
    }

    // M3: Read rs2 register at timestamp+1
    if d.read_register2 && d.rs2 != 0 {
        let reg_value = pack_register_value(op.rv2);
        let reg_addr = 2 * d.rs2 as u64;
        let (_old_val, old_ts) = register_state.read(d.rs2);
        // old_timestamps array is 8 elements but only first 2 are used for registers
        let old_timestamps = [old_ts, old_ts, 0, 0, 0, 0, 0, 0];

        let memw_op = MemwOperation::new(true, reg_addr, reg_value, op.timestamp + 1, 2, true)
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
        // old_timestamps array is 8 elements but only first 2 are used for registers
        let old_timestamps = [old_ts, old_ts, 0, 0, 0, 0, 0, 0];

        let memw_op = MemwOperation::new(true, reg_addr, reg_value, op.timestamp + 2, 2, false)
            .with_old(old_value, old_timestamps);
        memw_ops.push(memw_op);
        register_state.write(d.rd, op.rvd, op.timestamp + 2);
    }

    memw_ops
}

/// Collects MEMW operations for a COMMIT ECALL from CpuOperation.
///
/// All operations use the raw ECALL timestamp (no offsets). Per the spec,
/// independent accesses at different addresses can share a timestamp.
///
/// Operations:
/// - Read+write x10 at ts: asserts fd=1 (old), writes count (new)
/// - Read x11 at ts: reads buf_addr
/// - Read x12 at ts: reads count
/// - Read bytes at ts: reads committed bytes from memory
///
/// Returns: Vec of MEMW operations
fn collect_commit_memw_ops(
    op: &CpuOperation,
    register_state: &mut RegisterState,
    memory_state: &mut MemoryState,
) -> Vec<MemwOperation> {
    debug_assert!(
        !op.decode.read_register1 && !op.decode.read_register2 && !op.decode.write_register,
        "ECALL decode must not set register read/write flags"
    );

    let ts = op.timestamp;
    let buf_addr = op.commit_buf_addr;
    let count = op.commit_count;

    let mut memw_ops = Vec::with_capacity(3 + count as usize);

    // Combined read+write x10 at ts: old=fd=1, new=count
    // This atomically asserts x10 held fd=1 and writes count as return value.
    {
        let old_value = pack_register_value(1); // fd = 1
        let new_value = pack_register_value(count);
        let reg_addr = 2 * 10u64; // x10 → addr 20
        let (old_val, old_ts) = register_state.read(10);
        debug_assert_eq!(
            old_val, 1,
            "ECALL commit: x10 (fd) must be 1, got {old_val}"
        );
        let old_timestamps = [old_ts, old_ts, 0, 0, 0, 0, 0, 0];
        let memw_op = MemwOperation::new(true, reg_addr, new_value, ts, 2, false)
            .with_old(old_value, old_timestamps);
        memw_ops.push(memw_op);
        register_state.write(10, count, ts);
    }

    // Read x11 (buf_addr) at ts
    {
        let reg_value = pack_register_value(buf_addr);
        let reg_addr = 2 * 11u64; // x11 → addr 22
        let (_old_val, old_ts) = register_state.read(11);
        let old_timestamps = [old_ts, old_ts, 0, 0, 0, 0, 0, 0];
        let memw_op = MemwOperation::new(true, reg_addr, reg_value, ts, 2, true)
            .with_old(reg_value, old_timestamps);
        memw_ops.push(memw_op);
        register_state.write(11, buf_addr, ts);
    }

    // Read x12 (count) at ts
    {
        let reg_value = pack_register_value(count);
        let reg_addr = 2 * 12u64; // x12 → addr 24
        let (_old_val, old_ts) = register_state.read(12);
        let old_timestamps = [old_ts, old_ts, 0, 0, 0, 0, 0, 0];
        let memw_op = MemwOperation::new(true, reg_addr, reg_value, ts, 2, true)
            .with_old(reg_value, old_timestamps);
        memw_ops.push(memw_op);
        register_state.write(12, count, ts);
    }

    // Memory byte reads at ts
    for i in 0..count {
        let addr = buf_addr.wrapping_add(i);
        let (byte_val, old_ts) = memory_state.read_byte(addr);
        let value = [byte_val as u64, 0, 0, 0, 0, 0, 0, 0];
        let old_timestamps = [old_ts, 0, 0, 0, 0, 0, 0, 0];
        let memw_op =
            MemwOperation::new(false, addr, value, ts, 1, true).with_old(value, old_timestamps);
        memw_ops.push(memw_op);
        memory_state.write_byte(addr, byte_val, ts);
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

        // R1-R3: Address overflow checks (unconditional per MEMW-CR13/14/15)
        // If overflow occurs, LT returns lt=0 and the constraint (expecting lt=1)
        // rejects the proof via value mismatch.
        if memw_op.width == 2 {
            let addr_plus_1 = memw_op.base_address.wrapping_add(1);
            lt_ops.push(LtOperation::new(memw_op.base_address, addr_plus_1, false));
        }
        if memw_op.width == 4 {
            let addr_plus_3 = memw_op.base_address.wrapping_add(3);
            lt_ops.push(LtOperation::new(memw_op.base_address, addr_plus_3, false));
        }
        if memw_op.width == 8 {
            let addr_plus_7 = memw_op.base_address.wrapping_add(7);
            lt_ops.push(LtOperation::new(memw_op.base_address, addr_plus_7, false));
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

/// Collects bitwise lookups from MUL operations (MSB16 for sign bits).
///
/// MUL sends MSB16 lookups when signed=1 to extract sign bits,
/// IS_HALF lookups for lo/hi range checks, and IS_B20 lookups for carry range checks.
///
/// Returns: Vec of bitwise lookups
fn collect_bitwise_from_mul(mul_ops: &[(MulOperation, bool)]) -> Vec<BitwiseOperation> {
    let mut bitwise_ops = Vec::with_capacity(mul_ops.len() * 14);

    // IS_HALF and IS_B20: one set per raw op (multiplicity Sum(MU_LO, MU_HI))
    for (op, _wants_hi) in mul_ops {
        let (lo, hi) = op.compute_product();

        // IS_HALF for lo halfwords
        for shift in [0, 16, 32, 48] {
            let half = ((lo >> shift) & 0xFFFF) as u16;
            bitwise_ops.push(BitwiseOperation::halfword(
                BitwiseOperationType::IsHalf,
                (half & 0xFF) as u8,
                (half >> 8) as u8,
            ));
        }

        // IS_HALF for hi halfwords
        for shift in [0, 16, 32, 48] {
            let half = ((hi >> shift) & 0xFFFF) as u16;
            bitwise_ops.push(BitwiseOperation::halfword(
                BitwiseOperationType::IsHalf,
                (half & 0xFF) as u8,
                (half >> 8) as u8,
            ));
        }

        // IS_B20 for carry[0..4] range checks
        let raw_products = op.compute_raw_products();
        let carries = mul::compute_carries(lo, hi, &raw_products);
        for carry in carries {
            let x = (carry & 0xFF) as u8;
            let y = ((carry >> 8) & 0xFF) as u8;
            let z = ((carry >> 16) & 0xF) as u8;
            bitwise_ops.push(BitwiseOperation::b20(x, y, z));
        }
    }

    // MSB16: one per unique signed op (multiplicity Column(LHS_SIGNED) / Column(RHS_SIGNED))
    // The MUL table sends MSB16 with multiplicity = value of the SIGNED column (0 or 1)
    // per unique row, so we must generate exactly one MSB16 per unique MUL operation,
    // not per raw op.
    let mut msb16_seen = std::collections::HashSet::new();
    for (op, _wants_hi) in mul_ops {
        if msb16_seen.insert((op.lhs, op.lhs_signed, op.rhs, op.rhs_signed)) {
            if op.lhs_signed {
                let lhs_3 = ((op.lhs >> 48) & 0xFFFF) as u16;
                bitwise_ops.push(BitwiseOperation::halfword(
                    BitwiseOperationType::Msb16,
                    (lhs_3 & 0xFF) as u8,
                    (lhs_3 >> 8) as u8,
                ));
            }
            if op.rhs_signed {
                let rhs_3 = ((op.rhs >> 48) & 0xFFFF) as u16;
                bitwise_ops.push(BitwiseOperation::halfword(
                    BitwiseOperationType::Msb16,
                    (rhs_3 & 0xFF) as u8,
                    (rhs_3 >> 8) as u8,
                ));
            }
        }
    }

    bitwise_ops
}

/// Collects bitwise lookups from DVRM operations.
///
/// Generates: IS_HALF (×20), MSB16 (×3), ZERO (×2 per raw op + up to ×4 per unique signed op).
///
/// Returns: Vec of bitwise lookups
fn collect_bitwise_from_dvrm(dvrm_ops: &[(DvrmOperation, bool)]) -> Vec<BitwiseOperation> {
    let mut bitwise_ops = Vec::with_capacity(dvrm_ops.len() * 24);

    for (op, _wants_remainder) in dvrm_ops {
        // IS_HALF for n[0..4] (DVRM-A1)
        for shift in [0, 16, 32, 48] {
            let half = ((op.n >> shift) & 0xFFFF) as u16;
            bitwise_ops.push(BitwiseOperation::halfword(
                BitwiseOperationType::IsHalf,
                (half & 0xFF) as u8,
                (half >> 8) as u8,
            ));
        }

        // IS_HALF for d[0..4] (DVRM-A2)
        for shift in [0, 16, 32, 48] {
            let half = ((op.d >> shift) & 0xFFFF) as u16;
            bitwise_ops.push(BitwiseOperation::halfword(
                BitwiseOperationType::IsHalf,
                (half & 0xFF) as u8,
                (half >> 8) as u8,
            ));
        }

        // IS_HALF for r[0..4] (DVRM-C10)
        let r = op.compute_remainder();
        for shift in [0, 16, 32, 48] {
            let half = ((r >> shift) & 0xFFFF) as u16;
            bitwise_ops.push(BitwiseOperation::halfword(
                BitwiseOperationType::IsHalf,
                (half & 0xFF) as u8,
                (half >> 8) as u8,
            ));
        }

        // IS_HALF for n_sub_r[0..4] (DVRM-C11)
        let n_sub_r = op.n.wrapping_sub(r);
        for shift in [0, 16, 32, 48] {
            let half = ((n_sub_r >> shift) & 0xFFFF) as u16;
            bitwise_ops.push(BitwiseOperation::halfword(
                BitwiseOperationType::IsHalf,
                (half & 0xFF) as u8,
                (half >> 8) as u8,
            ));
        }

        // IS_HALF for q[0..4] (DVRM-C15)
        let q = op.compute_quotient();
        for shift in [0, 16, 32, 48] {
            let half = ((q >> shift) & 0xFFFF) as u16;
            bitwise_ops.push(BitwiseOperation::halfword(
                BitwiseOperationType::IsHalf,
                (half & 0xFF) as u8,
                (half >> 8) as u8,
            ));
        }

        // ZERO lookups per raw op (multiplicity = μ_sum = μ_q + μ_r)

        // C8: ZERO[overflow; overflow_sum]
        // overflow_sum = n[0]+n[1]+n[2]+n[3] - 32769*sign_n + 262141 - d[0]-d[1]-d[2]-d[3]
        let n_halves: [u32; 4] = [
            (op.n & 0xFFFF) as u32,
            ((op.n >> 16) & 0xFFFF) as u32,
            ((op.n >> 32) & 0xFFFF) as u32,
            ((op.n >> 48) & 0xFFFF) as u32,
        ];
        let d_halves: [u32; 4] = [
            (op.d & 0xFFFF) as u32,
            ((op.d >> 16) & 0xFFFF) as u32,
            ((op.d >> 32) & 0xFFFF) as u32,
            ((op.d >> 48) & 0xFFFF) as u32,
        ];
        let sign_n: u32 = if op.sign_n() { 1 } else { 0 };
        let overflow_sum = n_halves[0] + n_halves[1] + n_halves[2] + n_halves[3] + 262141
            - 32769 * sign_n
            - d_halves[0]
            - d_halves[1]
            - d_halves[2]
            - d_halves[3];
        bitwise_ops.push(BitwiseOperation::zero(overflow_sum));

        // C20: ZERO[div_by_zero; d[0]+d[1]+d[2]+d[3]]
        let d_sum = d_halves[0] + d_halves[1] + d_halves[2] + d_halves[3];
        bitwise_ops.push(BitwiseOperation::zero(d_sum));
    }

    // MSB16 lookups: one per unique signed op (not per raw op).
    // The DVRM bus interaction uses Multiplicity::Column(SIGNED)=1, so we
    // must emit exactly one MSB16 per unique signed (n, d) combo.
    let mut msb16_seen = std::collections::HashSet::new();
    for (op, _wants_remainder) in dvrm_ops {
        if op.signed && msb16_seen.insert(op.clone()) {
            let r = op.compute_remainder();

            // MSB16[n[3]]
            let n_3 = ((op.n >> 48) & 0xFFFF) as u16;
            bitwise_ops.push(BitwiseOperation::halfword(
                BitwiseOperationType::Msb16,
                (n_3 & 0xFF) as u8,
                (n_3 >> 8) as u8,
            ));

            // MSB16[r[3]]
            let r_3 = ((r >> 48) & 0xFFFF) as u16;
            bitwise_ops.push(BitwiseOperation::halfword(
                BitwiseOperationType::Msb16,
                (r_3 & 0xFF) as u8,
                (r_3 >> 8) as u8,
            ));

            // MSB16[d[3]]
            let d_3 = ((op.d >> 48) & 0xFFFF) as u16;
            bitwise_ops.push(BitwiseOperation::halfword(
                BitwiseOperationType::Msb16,
                (d_3 & 0xFF) as u8,
                (d_3 >> 8) as u8,
            ));
        }
    }

    // ZERO lookups for NEG template: one per unique op where sign is set.
    // C3 uses Multiplicity::Column(SIGN_R) = 1 per unique row where sign_r = 1.
    // C5 uses Multiplicity::Column(SIGN_D) = 1 per unique row where sign_d = 1.
    let mut zero_seen = std::collections::HashSet::new();
    for (op, _wants_remainder) in dvrm_ops {
        if zero_seen.insert(op.clone()) {
            // C3: NEG for r (when sign_r = 1)
            if op.sign_r() {
                let r = op.compute_remainder();
                let r_halves: [u32; 4] = [
                    (r & 0xFFFF) as u32,
                    ((r >> 16) & 0xFFFF) as u32,
                    ((r >> 32) & 0xFFFF) as u32,
                    ((r >> 48) & 0xFFFF) as u32,
                ];
                // C3a: ZERO[1-carry_r[0]; r[0]+r[1]]
                bitwise_ops.push(BitwiseOperation::zero(r_halves[0] + r_halves[1]));
                // C3b: ZERO[1-carry_r[1]; r[0]+r[1]+r[2]+r[3]]
                bitwise_ops.push(BitwiseOperation::zero(
                    r_halves[0] + r_halves[1] + r_halves[2] + r_halves[3],
                ));
            }

            // C5: NEG for d (when sign_d = 1)
            if op.sign_d() {
                let d_halves: [u32; 4] = [
                    (op.d & 0xFFFF) as u32,
                    ((op.d >> 16) & 0xFFFF) as u32,
                    ((op.d >> 32) & 0xFFFF) as u32,
                    ((op.d >> 48) & 0xFFFF) as u32,
                ];
                // C5a: ZERO[1-carry_d[0]; d[0]+d[1]]
                bitwise_ops.push(BitwiseOperation::zero(d_halves[0] + d_halves[1]));
                // C5b: ZERO[1-carry_d[1]; d[0]+d[1]+d[2]+d[3]]
                bitwise_ops.push(BitwiseOperation::zero(
                    d_halves[0] + d_halves[1] + d_halves[2] + d_halves[3],
                ));
            }
        }
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

/// Collects bitwise lookups from BRANCH operations.
///
/// BRANCH sends:
/// - IS_BYTE[next_pc_low[1]] - range check bits 8-15
/// - AND_BYTE[unmasked_low_byte, 254, next_pc_low[0]] - LSB masking
/// - IS_HALFWORD[next_pc_high[0..3]] - range checks for bits 16-63
///
/// Returns: Vec of bitwise lookups
fn collect_bitwise_from_branch(branch_ops: &[BranchOperation]) -> Vec<BitwiseOperation> {
    let mut bitwise_ops = Vec::with_capacity(branch_ops.len() * 5);

    for op in branch_ops {
        let next_pc = op.compute_next_pc();
        let next_pc_unmasked = op.compute_next_pc_unmasked();

        // Extract next_pc components
        let _next_pc_low_0 = (next_pc & 0xFF) as u8; // Used by constraint, not lookup
        let next_pc_low_1 = ((next_pc >> 8) & 0xFF) as u8;
        let next_pc_high_0 = ((next_pc >> 16) & 0xFFFF) as u16;
        let next_pc_high_1 = ((next_pc >> 32) & 0xFFFF) as u16;
        let next_pc_high_2 = ((next_pc >> 48) & 0xFFFF) as u16;
        let unmasked_low_byte = (next_pc_unmasked & 0xFF) as u8;

        // IS_BYTE[next_pc_low[1]] - range check for byte value
        bitwise_ops.push(BitwiseOperation::single_byte(
            BitwiseOperationType::IsByte,
            next_pc_low_1,
        ));

        // AND_BYTE[unmasked_low_byte, 254] → next_pc_low[0]
        // Verifies: next_pc_low[0] = unmasked_low_byte & 0xFE
        bitwise_ops.push(BitwiseOperation::byte_op(
            BitwiseOperationType::AndByte,
            unmasked_low_byte,
            254, // 0xFE mask
        ));

        // IS_HALFWORD[next_pc_high[0]]
        bitwise_ops.push(BitwiseOperation::halfword(
            BitwiseOperationType::IsHalf,
            (next_pc_high_0 & 0xFF) as u8,
            (next_pc_high_0 >> 8) as u8,
        ));

        // IS_HALFWORD[next_pc_high[1]]
        bitwise_ops.push(BitwiseOperation::halfword(
            BitwiseOperationType::IsHalf,
            (next_pc_high_1 & 0xFF) as u8,
            (next_pc_high_1 >> 8) as u8,
        ));

        // IS_HALFWORD[next_pc_high[2]]
        bitwise_ops.push(BitwiseOperation::halfword(
            BitwiseOperationType::IsHalf,
            (next_pc_high_2 & 0xFF) as u8,
            (next_pc_high_2 >> 8) as u8,
        ));
    }

    bitwise_ops
}

/// Collects IS_BYTE lookups from PAGE data (init and fini values).
///
/// Each PAGE byte generates 2 IS_BYTE lookups:
/// - C1: IS_BYTE[init] for initialization range check
/// - C2: IS_BYTE[fini] for finalization range check
///
/// This must be called BEFORE bitwise multiplicities are updated.
fn collect_bitwise_from_page(elf: &Elf, memory_state: &MemoryState) -> Vec<BitwiseOperation> {
    use std::collections::BTreeSet;

    let page_size = page::DEFAULT_PAGE_SIZE;
    let mut bitwise_ops = Vec::new();

    // Collect ELF page init data
    let mut elf_page_data: HashMap<u64, Vec<u8>> = HashMap::new();

    for segment in &elf.data {
        for (i, &word) in segment.values.iter().enumerate() {
            let word_addr = segment.base_addr + (i as u64 * 4);
            for byte_offset in 0..4u64 {
                let byte_addr = word_addr + byte_offset;
                let byte_value = ((word >> (byte_offset * 8)) & 0xFF) as u8;

                let page_base = page::page_base_for_address(byte_addr, page_size);
                let offset = page::offset_in_page(byte_addr, page_size);

                let page_data = elf_page_data
                    .entry(page_base)
                    .or_insert_with(|| vec![0u8; page_size]);
                page_data[offset] = byte_value;
            }
        }
    }

    // Derive ALL page bases from memory_state (includes ELF + runtime pages)
    let mut page_bases: BTreeSet<u64> = BTreeSet::new();
    for &addr in memory_state.cells.keys() {
        page_bases.insert(page::page_base_for_address(addr, page_size));
    }

    // Build final state map from memory_state
    let final_state: FinalStateMap = memory_state
        .cells
        .iter()
        .map(|(&addr, &(value, timestamp))| (addr, FinalByteState { timestamp, value }))
        .collect();

    // For each page and each byte, add IS_BYTE lookups for init and fini
    for &page_base in &page_bases {
        let init_data = elf_page_data.get(&page_base);

        for offset in 0..page_size {
            let addr = page_base + offset as u64;

            // Get init value (from ELF or 0)
            let init = init_data.map_or(0u8, |data| data[offset]);

            // Get fini value (from final_state or init if never accessed)
            let fini = final_state.get(&addr).map_or(init, |state| state.value);

            // C1: IS_BYTE[init]
            bitwise_ops.push(BitwiseOperation::single_byte(
                BitwiseOperationType::IsByte,
                init,
            ));

            // C2: IS_BYTE[fini]
            bitwise_ops.push(BitwiseOperation::single_byte(
                BitwiseOperationType::IsByte,
                fini,
            ));
        }
    }

    bitwise_ops
}

// =============================================================================
// COMMIT Operation Expansion
// =============================================================================

/// Expand Commit ECALLs from CPU operations into individual byte rows.
///
/// For each Commit ECALL, generates count+1 rows:
/// - Row 0: first=1, reads byte at buf_addr
/// - Row i (0 < i < count): first=0, reads byte at buf_addr+i
/// - Row count: end=1, count=0, no byte read
fn expand_commit_operations(
    cpu_ops: &[CpuOperation],
    memory_state: &MemoryState,
) -> Vec<CommitOperation> {
    let mut ops = Vec::new();

    for ecall in cpu_ops.iter().filter(|op| op.ecall_commit) {
        let timestamp = ecall.timestamp;
        let buf_addr = ecall.commit_buf_addr;
        let count = ecall.commit_count;

        for i in 0..=count {
            let remaining = count - i;
            let is_end = remaining == 0;
            let value = if !is_end {
                let (byte_val, _ts) = memory_state.read_byte(buf_addr + i);
                byte_val
            } else {
                0
            };
            ops.push(CommitOperation {
                timestamp,
                address: buf_addr + i,
                count: remaining,
                first: i == 0,
                end: is_end,
                value,
            });
        }
    }
    ops
}

/// Collect bitwise lookups from COMMIT operations.
///
/// The COMMIT table sends:
/// - IsByte for value (1 per non-end row, mult = mu - end)
/// - IsHalfword for count_decr components (4 per real row, mult = mu)
/// - IsHalfword for address_incr halfwords (4 per real row, mult = mu)
/// - Zero for end detection (1 per real row, mult = mu)
fn collect_bitwise_from_commit(commit_ops: &[CommitOperation]) -> Vec<BitwiseOperation> {
    let mut lookups = Vec::new();

    for op in commit_ops {
        // IsByte for value (mult = mu - end, skip end rows)
        if !op.end {
            lookups.push(BitwiseOperation::single_byte(
                BitwiseOperationType::IsByte,
                op.value,
            ));
        }

        // IsHalfword for count_decr components (4 halfwords, mult = mu)
        let count_decr = if op.count == 0 {
            u64::MAX
        } else {
            op.count - 1
        };
        for shift in [0, 16, 32, 48] {
            let half = ((count_decr >> shift) & 0xFFFF) as u16;
            lookups.push(BitwiseOperation::halfword(
                BitwiseOperationType::IsHalf,
                (half & 0xFF) as u8,
                ((half >> 8) & 0xFF) as u8,
            ));
        }

        // IsHalfword for address_incr halfwords (4 halfwords, mult = mu)
        // All real rows send these, matching the spec's unconditional mult = mu.
        let address_incr = op.address.wrapping_add(1);
        for shift in [0, 16, 32, 48] {
            let half = ((address_incr >> shift) & 0xFFFF) as u16;
            lookups.push(BitwiseOperation::halfword(
                BitwiseOperationType::IsHalf,
                (half & 0xFF) as u8,
                ((half >> 8) & 0xFF) as u8,
            ));
        }

        // Zero bus for end detection (mult = mu)
        // Input: (65535 - cd_0) + (65535 - cd_1) + (65535 - cd_2) + (65535 - cd_3)
        // When count_decr = 0xFFFF_FFFF_FFFF_FFFF (count=0), sum = 0 → end=1
        let cd_0 = (count_decr & 0xFFFF) as u32;
        let cd_1 = ((count_decr >> 16) & 0xFFFF) as u32;
        let cd_2 = ((count_decr >> 32) & 0xFFFF) as u32;
        let cd_3 = ((count_decr >> 48) & 0xFFFF) as u32;
        let zero_input = (65535 - cd_0) + (65535 - cd_1) + (65535 - cd_2) + (65535 - cd_3);
        lookups.push(BitwiseOperation::zero(zero_input));
    }

    lookups
}

// =============================================================================
// PAGE Table Generation
// =============================================================================

/// Generates PAGE tables for memory initialization and finalization.
///
/// Derives all page bases from `memory_state.cells.keys()` — this includes
/// every address accessed during execution (ELF init + runtime stores/loads).
/// ELF pages get their init data from the binary; all others are zero-init.
fn generate_page_tables(
    elf: &Elf,
    memory_state: &MemoryState,
) -> (
    Vec<TraceTable<GoldilocksField, GoldilocksExtension>>,
    Vec<PageConfig>,
) {
    use std::collections::BTreeSet;

    let page_size = page::DEFAULT_PAGE_SIZE;

    // Collect ELF page init data (needed for PageConfig::with_data)
    let mut elf_page_data: HashMap<u64, Vec<u8>> = HashMap::new();

    for segment in &elf.data {
        for (i, &word) in segment.values.iter().enumerate() {
            let word_addr = segment.base_addr + (i as u64 * 4);

            for byte_offset in 0..4u64 {
                let byte_addr = word_addr + byte_offset;
                let byte_value = ((word >> (byte_offset * 8)) & 0xFF) as u8;

                let page_base = page::page_base_for_address(byte_addr, page_size);
                let offset = page::offset_in_page(byte_addr, page_size);

                let page_data = elf_page_data
                    .entry(page_base)
                    .or_insert_with(|| vec![0u8; page_size]);
                page_data[offset] = byte_value;
            }
        }
    }

    // Derive ALL page bases from memory_state (includes ELF + runtime pages)
    let mut page_bases: BTreeSet<u64> = BTreeSet::new();
    for &addr in memory_state.cells.keys() {
        page_bases.insert(page::page_base_for_address(addr, page_size));
    }

    // Build final state map from memory_state
    let final_state: FinalStateMap = memory_state
        .cells
        .iter()
        .map(|(&addr, &(value, timestamp))| (addr, FinalByteState { timestamp, value }))
        .collect();

    // Generate PAGE tables and configs
    let mut pages = Vec::new();
    let mut page_configs = Vec::new();

    for &page_base in &page_bases {
        let config = if let Some(init_data) = elf_page_data.get(&page_base) {
            PageConfig::with_data(page_base, page_size, init_data.clone())
        } else {
            PageConfig::zero_init(page_base, page_size)
        };

        let trace = page::generate_page_trace(&config, &final_state);
        pages.push(trace);
        page_configs.push(config);
    }

    (pages, page_configs)
}

// =============================================================================
// Trace Generation
// =============================================================================

/// All generated trace tables.
pub struct Traces {
    /// CPU execution traces (split into chunks of max_rows::CPU)
    pub cpus: Vec<TraceTable<GoldilocksField, GoldilocksExtension>>,

    /// BITWISE precomputed lookup table (2^20 rows)
    pub bitwise: TraceTable<GoldilocksField, GoldilocksExtension>,

    /// LT comparison traces (split into chunks of max_rows::LT)
    pub lts: Vec<TraceTable<GoldilocksField, GoldilocksExtension>>,

    /// MEMW memory/register read/write traces (split into chunks of max_rows::MEMW)
    pub memws: Vec<TraceTable<GoldilocksField, GoldilocksExtension>>,

    /// LOAD memory load with extension traces (split into chunks of max_rows::LOAD)
    pub loads: Vec<TraceTable<GoldilocksField, GoldilocksExtension>>,

    /// DECODE instruction decoding table (preprocessed from ELF)
    pub decode: TraceTable<GoldilocksField, GoldilocksExtension>,

    /// MUL multiplication traces (wrapped in Vec for uniform architecture)
    pub muls: Vec<TraceTable<GoldilocksField, GoldilocksExtension>>,

    /// DVRM division/remainder traces (wrapped in Vec for uniform architecture)
    pub dvrms: Vec<TraceTable<GoldilocksField, GoldilocksExtension>>,

    /// PAGE tables for memory initialization/finalization (one per page)
    pub pages: Vec<TraceTable<GoldilocksField, GoldilocksExtension>>,

    /// Page configurations (for bus interactions)
    pub page_configs: Vec<PageConfig>,

    /// REGISTER table for register initialization/finalization
    pub register: TraceTable<GoldilocksField, GoldilocksExtension>,

    /// BRANCH target calculation traces (wrapped in Vec for uniform architecture)
    pub branches: Vec<TraceTable<GoldilocksField, GoldilocksExtension>>,

    /// HALT single-row table for program termination
    pub halt: TraceTable<GoldilocksField, GoldilocksExtension>,

    /// COMMIT table for write syscall (byte-by-byte commit with recursive bus)
    pub commit: TraceTable<GoldilocksField, GoldilocksExtension>,
}

/// Chunk raw ops and generate one trace table per chunk.
fn chunk_and_generate<T>(
    ops: &[T],
    max_rows: usize,
    generate: impl Fn(&[T]) -> TraceTable<GoldilocksField, GoldilocksExtension>,
) -> Vec<TraceTable<GoldilocksField, GoldilocksExtension>> {
    if ops.is_empty() {
        vec![generate(&[])]
    } else {
        ops.chunks(max_rows).map(generate).collect()
    }
}

impl Traces {
    /// Returns the number of chunks for each split table.
    pub fn table_counts(&self) -> crate::TableCounts {
        crate::TableCounts {
            cpu: self.cpus.len(),
            lt: self.lts.len(),
            memw: self.memws.len(),
            load: self.loads.len(),
            mul: self.muls.len(),
            dvrm: self.dvrms.len(),
            branch: self.branches.len(),
        }
    }

    /// Extract page configurations from ELF only (deterministic from binary).
    ///
    /// Returns PageConfigs for pages covered by ELF segments, with their
    /// init data populated. Used by the verifier to reconstruct the ELF
    /// portion of the PAGE table layout.
    pub fn page_configs_from_elf(elf: &Elf) -> Vec<PageConfig> {
        use std::collections::BTreeSet;

        let page_size = page::DEFAULT_PAGE_SIZE;
        let mut page_bases: BTreeSet<u64> = BTreeSet::new();
        let mut elf_page_data: HashMap<u64, Vec<u8>> = HashMap::new();

        for segment in &elf.data {
            for (i, &word) in segment.values.iter().enumerate() {
                let word_addr = segment.base_addr + (i as u64 * 4);
                for byte_offset in 0..4u64 {
                    let byte_addr = word_addr + byte_offset;
                    let byte_value = ((word >> (byte_offset * 8)) & 0xFF) as u8;
                    let page_base = page::page_base_for_address(byte_addr, page_size);
                    let offset = page::offset_in_page(byte_addr, page_size);
                    page_bases.insert(page_base);

                    let page_data = elf_page_data
                        .entry(page_base)
                        .or_insert_with(|| vec![0u8; page_size]);
                    page_data[offset] = byte_value;
                }
            }
        }

        page_bases
            .into_iter()
            .map(|base| {
                if let Some(init_data) = elf_page_data.get(&base) {
                    PageConfig::with_data(base, page_size, init_data.clone())
                } else {
                    PageConfig::zero_init(base, page_size)
                }
            })
            .collect()
    }

    /// Reconstruct page configs from ELF and runtime page ranges.
    ///
    /// Used by the verifier to reconstruct the full PAGE table layout.
    /// Combines deterministic ELF pages with prover-provided runtime
    /// page ranges (zero-initialized: stack, heap, etc.).
    /// Each range is `(base, count)` — `count` contiguous 4KB pages from `base`.
    pub fn page_configs_from_elf_and_runtime(
        elf: &Elf,
        runtime_page_ranges: &[crate::RuntimePageRange],
    ) -> Vec<PageConfig> {
        let mut configs = Self::page_configs_from_elf(elf);
        let page_size = page::DEFAULT_PAGE_SIZE;
        for r in runtime_page_ranges {
            let (base, count) = (r.base, r.count);
            for i in 0..count {
                configs.push(PageConfig::zero_init(
                    base + i * page_size as u64,
                    page_size,
                ));
            }
        }
        configs.sort_by_key(|c| c.page_base);
        configs
    }

    /// Extracts runtime page ranges from the generated page configs.
    ///
    /// Returns run-length encoded `(base, count)` pairs for page bases not
    /// covered by ELF segments. Contiguous pages are merged into a single range.
    ///
    /// Runtime (non-ELF) pages are identified by `init_values == None`
    /// (zero-init), avoiding a redundant ELF segment scan.
    pub fn runtime_page_ranges(&self) -> Vec<crate::RuntimePageRange> {
        let page_size = page::DEFAULT_PAGE_SIZE as u64;

        // Collect sorted non-ELF page bases (zero-init pages are runtime pages)
        let runtime_bases: Vec<u64> = self
            .page_configs
            .iter()
            .filter(|config| config.init_values.is_none())
            .map(|config| config.page_base)
            .collect();

        // Run-length encode contiguous pages into (base, count) ranges
        let mut ranges = Vec::new();
        if runtime_bases.is_empty() {
            return ranges;
        }

        let mut start = runtime_bases[0];
        let mut count = 1u64;

        for &base in &runtime_bases[1..] {
            if base == start + count * page_size {
                count += 1;
            } else {
                ranges.push(crate::RuntimePageRange { base: start, count });
                start = base;
                count = 1;
            }
        }
        ranges.push(crate::RuntimePageRange { base: start, count });

        ranges
    }

    /// Generates all traces from ELF and execution logs using phased collection.
    ///
    /// The phases are:
    /// 0. ELF → DECODE (preprocessed table)
    /// 1. Logs → CPU operations
    /// 2. CPU ops → MEMW, LOAD, LT, Bitwise, Branch (state tracking for MEMW/LOAD)
    /// 3. MEMW → LT operations (timestamp ordering)
    /// 4. LT, MEMW, Branch → Bitwise lookups
    /// 5. Generate all traces including PAGE tables
    pub fn from_elf_and_logs(
        elf: &Elf,
        logs: &[Log],
        max_rows: &super::MaxRowsConfig,
    ) -> Result<Self, Error> {
        // =====================================================================
        // PHASE 0: ELF → DECODE + instructions
        // =====================================================================
        // IMPORTANT: Use generate_decode_trace (same as compute_precomputed_commitment)
        // so the DECODE trace row ordering matches the AIR's hardcoded commitment.
        // tables_from_elf iterates ELF segments sequentially, but the commitment
        // is computed via HashMap iteration which may have different ordering.
        let instructions = decode::instructions_from_elf(elf)
            .map_err(|e| Error::Execution(format!("Failed to parse instructions: {e}")))?;
        let (decode_trace, decode_pc_to_row) = decode::generate_decode_trace(&instructions);

        // =====================================================================
        // PHASE 1: Logs → CPU operations
        // =====================================================================
        let cpu_ops = collect_cpu_ops(logs, &instructions)?;

        // =====================================================================
        // PHASE 2: CPU ops → MEMW, LOAD, LT, Bitwise, Branch
        // =====================================================================
        // Processes cpu_ops in order. MEMW/LOAD need state tracking, LT/Bitwise don't.
        // Initialize memory state from ELF so first accesses get correct old_value.
        let mut memory_state = MemoryState::from_elf(elf);
        let mut register_state = RegisterState::new();
        let (memw_ops, load_ops, mut lt_ops, mut bitwise_ops) =
            collect_ops_from_cpu(&cpu_ops, &mut memory_state, &mut register_state);

        // Collect BRANCH operations from CPU ops where branch_cond = true
        let branch_ops: Vec<BranchOperation> = cpu_ops
            .iter()
            .filter(|op| op.branch_cond)
            .map(|op| {
                BranchOperation::new(
                    op.decode.pc,
                    op.decode.imm, // offset as full 64-bit DWordWL (already sign-extended)
                    op.compute_arg1(), // register value must match CPU's arg1 for bus signature
                    op.decode.op_jalr,
                )
            })
            .collect();

        // Collect MUL operations from CPU ops where op_mul = true
        let mut mul_ops: Vec<(MulOperation, bool)> = cpu_ops
            .iter()
            .filter(|op| op.decode.op_mul)
            .map(|op| {
                let lhs = op.compute_arg1();
                let lhs_signed = op.decode.signed;
                // rhs_signed = mp_selector per spec CPU-CA44:
                // MUL/MULH have mp_selector=1 (both signed), MULHU/MULHSU have mp_selector=0 (rhs unsigned)
                let rhs_signed = op.decode.mp_selector;
                let rhs = op.compute_arg2();
                let wants_hi = op.decode.muldiv_selector;
                (
                    MulOperation::new(lhs, lhs_signed, rhs, rhs_signed),
                    wants_hi,
                )
            })
            .collect();

        // Collect DVRM operations from CPU ops where op_divrem = true
        let dvrm_ops: Vec<(DvrmOperation, bool)> = cpu_ops
            .iter()
            .filter(|op| op.decode.op_divrem)
            .map(|op| {
                let n = op.compute_arg1();
                let d = op.compute_arg2();
                let signed = op.decode.signed;
                let wants_remainder = op.decode.muldiv_selector;
                (DvrmOperation::new(n, d, signed), wants_remainder)
            })
            .collect();

        // Collect LT operations from DVRM: |r| < |d| (unsigned comparison)
        for (op, _wants_remainder) in &dvrm_ops {
            lt_ops.push(LtOperation::new(op.abs_r(), op.abs_d(), false));
        }

        // Collect MUL operations from DVRM: d * q = n_sub_r (C13 lo, C14 hi)
        for (op, _wants_remainder) in &dvrm_ops {
            let d = op.d;
            let d_signed = op.signed;
            let q = op.compute_quotient();
            let q_signed = op.sign_q();
            let mul_op = MulOperation::new(d, d_signed, q, q_signed);
            mul_ops.push((mul_op.clone(), false)); // C13: lo (muldiv_selector=0)
            mul_ops.push((mul_op, true)); // C14: hi (muldiv_selector=1)
        }

        // =====================================================================
        // PHASE 3: MEMW → LT (timestamp ordering and overflow checks)
        // =====================================================================
        lt_ops.extend(collect_lt_from_memw(&memw_ops));

        // =====================================================================
        // PHASE 4: All → Bitwise lookups
        // =====================================================================
        bitwise_ops.extend(collect_bitwise_from_lt(&lt_ops));
        bitwise_ops.extend(collect_bitwise_from_memw(&memw_ops));
        bitwise_ops.extend(collect_bitwise_from_mul(&mul_ops));
        bitwise_ops.extend(collect_bitwise_from_dvrm(&dvrm_ops));
        bitwise_ops.extend(collect_bitwise_from_branch(&branch_ops));
        // PAGE tables do IS_BYTE lookups for init and fini values (C1, C2)
        bitwise_ops.extend(collect_bitwise_from_page(elf, &memory_state));

        // Expand COMMIT ECALL operations into per-byte rows
        let commit_ops = expand_commit_operations(&cpu_ops, &memory_state);
        // COMMIT table sends IsByte and IsHalfword lookups
        bitwise_ops.extend(collect_bitwise_from_commit(&commit_ops));

        // =====================================================================
        // PHASE 5: Generate final traces (parallelized)
        // =====================================================================

        // Extract halt timestamp from the last ECALL instruction
        let halt_op = cpu_ops
            .iter()
            .rev()
            .find(|op| op.decode.op_ecall)
            .ok_or(Error::MissingHaltEcall)?;
        let halt_timestamp = halt_op.timestamp;

        let cpus = chunk_and_generate(&cpu_ops, max_rows.cpu, cpu::generate_cpu_trace);
        let memws = chunk_and_generate(&memw_ops, max_rows.memw, memw::generate_memw_trace);
        let loads = chunk_and_generate(&load_ops, max_rows.load, load::generate_load_trace);
        let lts = chunk_and_generate(&lt_ops, max_rows.lt, lt::generate_lt_trace);
        let muls = chunk_and_generate(&mul_ops, max_rows.mul, mul::generate_mul_trace);
        let dvrms = chunk_and_generate(&dvrm_ops, max_rows.dvrm, dvrm::generate_dvrm_trace);
        let branches =
            chunk_and_generate(&branch_ops, max_rows.branch, branch::generate_branch_trace);

        let mut bitwise = bitwise::generate_bitwise_trace();
        bitwise::update_multiplicities(&mut bitwise, &bitwise_ops);

        // Update DECODE multiplicities
        // Each CPU operation looks up the DECODE table once
        // Padding rows also look up pc=1 (the CPU padding entry)
        // When CPU is split, each chunk pads independently
        let mut decode = decode_trace;
        let pc_to_row = decode_pc_to_row;
        let num_padding_rows: usize = cpu_ops
            .chunks(max_rows.cpu)
            .map(|chunk| chunk.len().next_power_of_two().max(4) - chunk.len())
            .sum();
        let mut decode_lookups: Vec<u64> = cpu_ops.iter().map(|op| op.decode.pc).collect();
        decode_lookups.extend(std::iter::repeat_n(cpu::CPU_PADDING_PC, num_padding_rows));
        decode::update_multiplicities(&mut decode, &pc_to_row, &decode_lookups);

        // Prepare register final state before scope (needs register_state ownership)
        let register_final_state = register_state.to_final_state_map();

        // Generate remaining traces in parallel (page, register, halt, commit).
        // chunk_and_generate already handled cpu, lt, memw, load, mul, dvrm, branch above.
        let commit_trace = commit::generate_commit_trace(&commit_ops);
        let (pages, page_configs, register_trace, halt_trace);
        #[cfg(feature = "parallel")]
        {
            let ((pages_val, register_val), halt_val) = rayon::join(
                || {
                    rayon::join(
                        || generate_page_tables(elf, &memory_state),
                        || register::generate_register_trace(&register_final_state),
                    )
                },
                || halt::generate_halt_trace(halt_timestamp),
            );
            let (pages_v, page_configs_v) = pages_val;
            pages = pages_v;
            page_configs = page_configs_v;
            register_trace = register_val;
            halt_trace = halt_val;
        }
        #[cfg(not(feature = "parallel"))]
        {
            let (pages_v, page_configs_v) = generate_page_tables(elf, &memory_state);
            pages = pages_v;
            page_configs = page_configs_v;
            register_trace = register::generate_register_trace(&register_final_state);
            halt_trace = halt::generate_halt_trace(halt_timestamp);
        }

        Ok(Traces {
            cpus,
            bitwise,
            lts,
            memws,
            loads,
            decode,
            muls,
            dvrms,
            pages,
            page_configs,
            register: register_trace,
            branches,
            halt: halt_trace,
            commit: commit_trace,
        })
    }

    /// Generates all traces from execution logs (legacy API).
    ///
    /// This is a compatibility wrapper. Prefer `from_elf_and_logs` for new code
    /// as it generates PAGE tables from ELF data.
    ///
    /// Note: This creates empty PAGE tables since no ELF is provided.
    pub fn from_logs(
        logs: &[Log],
        instructions: U64HashMap<Instruction>,
        max_rows: &super::MaxRowsConfig,
    ) -> Result<Self, Error> {
        // =====================================================================
        // PHASE 1: Logs → CPU operations
        // =====================================================================
        let cpu_ops = collect_cpu_ops(logs, &instructions)?;

        // =====================================================================
        // PHASE 2: CPU ops → MEMW, LOAD, LT, Bitwise, Branch
        // =====================================================================
        // Processes cpu_ops in order. MEMW/LOAD need state tracking, LT/Bitwise don't.
        let mut memory_state = MemoryState::new();
        let mut register_state = RegisterState::new();
        let (memw_ops, load_ops, mut lt_ops, mut bitwise_ops) =
            collect_ops_from_cpu(&cpu_ops, &mut memory_state, &mut register_state);

        // Collect MUL operations from CPU ops where op_mul = true
        let mut mul_ops: Vec<(MulOperation, bool)> = cpu_ops
            .iter()
            .filter(|op| op.decode.op_mul)
            .map(|op| {
                let lhs = op.compute_arg1();
                let lhs_signed = op.decode.signed;
                // rhs_signed = mp_selector per spec CPU-CA44:
                // MUL/MULH have mp_selector=1 (both signed), MULHU/MULHSU have mp_selector=0 (rhs unsigned)
                let rhs_signed = op.decode.mp_selector;
                let rhs = op.compute_arg2();
                let wants_hi = op.decode.muldiv_selector;
                (
                    MulOperation::new(lhs, lhs_signed, rhs, rhs_signed),
                    wants_hi,
                )
            })
            .collect();

        // Collect DVRM operations from CPU ops where op_divrem = true
        let dvrm_ops: Vec<(DvrmOperation, bool)> = cpu_ops
            .iter()
            .filter(|op| op.decode.op_divrem)
            .map(|op| {
                let n = op.compute_arg1();
                let d = op.compute_arg2();
                let signed = op.decode.signed;
                let wants_remainder = op.decode.muldiv_selector;
                (DvrmOperation::new(n, d, signed), wants_remainder)
            })
            .collect();

        // Collect BRANCH operations from CPU ops where branch_cond = true
        let branch_ops: Vec<BranchOperation> = cpu_ops
            .iter()
            .filter(|op| op.branch_cond)
            .map(|op| {
                BranchOperation::new(
                    op.decode.pc,
                    op.decode.imm, // offset as full 64-bit DWordWL (already sign-extended)
                    op.compute_arg1(), // register value must match CPU's arg1 for bus signature
                    op.decode.op_jalr,
                )
            })
            .collect();

        // Collect LT operations from DVRM: |r| < |d| (unsigned comparison)
        for (op, _wants_remainder) in &dvrm_ops {
            lt_ops.push(LtOperation::new(op.abs_r(), op.abs_d(), false));
        }

        // Collect MUL operations from DVRM: d * q = n_sub_r (C13 lo, C14 hi)
        for (op, _wants_remainder) in &dvrm_ops {
            let d = op.d;
            let d_signed = op.signed;
            let q = op.compute_quotient();
            let q_signed = op.sign_q();
            let mul_op = MulOperation::new(d, d_signed, q, q_signed);
            mul_ops.push((mul_op.clone(), false)); // C13: lo (muldiv_selector=0)
            mul_ops.push((mul_op, true)); // C14: hi (muldiv_selector=1)
        }

        // =====================================================================
        // PHASE 3: MEMW → LT (timestamp ordering and overflow checks)
        // =====================================================================
        lt_ops.extend(collect_lt_from_memw(&memw_ops));

        // =====================================================================
        // PHASE 4: All → Bitwise lookups
        // =====================================================================
        bitwise_ops.extend(collect_bitwise_from_lt(&lt_ops));
        bitwise_ops.extend(collect_bitwise_from_memw(&memw_ops));
        bitwise_ops.extend(collect_bitwise_from_mul(&mul_ops));
        bitwise_ops.extend(collect_bitwise_from_dvrm(&dvrm_ops));
        bitwise_ops.extend(collect_bitwise_from_branch(&branch_ops));

        // Expand COMMIT ECALL operations into per-byte rows
        let commit_ops = expand_commit_operations(&cpu_ops, &memory_state);
        // COMMIT table sends IsByte and IsHalfword lookups
        bitwise_ops.extend(collect_bitwise_from_commit(&commit_ops));

        // =====================================================================
        // PHASE 5: Generate final traces (parallelized)
        // =====================================================================

        // Extract halt timestamp from the last ECALL instruction
        let halt_op = cpu_ops
            .iter()
            .rev()
            .find(|op| op.decode.op_ecall)
            .ok_or(Error::MissingHaltEcall)?;
        let halt_timestamp = halt_op.timestamp;

        let cpus = chunk_and_generate(&cpu_ops, max_rows.cpu, cpu::generate_cpu_trace);
        let memws = chunk_and_generate(&memw_ops, max_rows.memw, memw::generate_memw_trace);
        let loads = chunk_and_generate(&load_ops, max_rows.load, load::generate_load_trace);
        let lts = chunk_and_generate(&lt_ops, max_rows.lt, lt::generate_lt_trace);
        let muls = chunk_and_generate(&mul_ops, max_rows.mul, mul::generate_mul_trace);
        let dvrms = chunk_and_generate(&dvrm_ops, max_rows.dvrm, dvrm::generate_dvrm_trace);
        let branches =
            chunk_and_generate(&branch_ops, max_rows.branch, branch::generate_branch_trace);

        let mut bitwise = bitwise::generate_bitwise_trace();
        bitwise::update_multiplicities(&mut bitwise, &bitwise_ops);

        // Generate DECODE trace and update multiplicities
        // Each CPU operation looks up the DECODE table once
        // Padding rows also look up pc=1 (the CPU padding entry)
        // When CPU is split, each chunk pads independently
        let (mut decode, pc_to_row) = decode::generate_decode_trace(&instructions);
        let num_padding_rows: usize = cpu_ops
            .chunks(max_rows.cpu)
            .map(|chunk| chunk.len().next_power_of_two().max(4) - chunk.len())
            .sum();
        let mut decode_lookups: Vec<u64> = cpu_ops.iter().map(|op| op.decode.pc).collect();
        decode_lookups.extend(std::iter::repeat_n(cpu::CPU_PADDING_PC, num_padding_rows));
        decode::update_multiplicities(&mut decode, &pc_to_row, &decode_lookups);
        let register_final_state = register_state.to_final_state_map();

        let commit_trace = commit::generate_commit_trace(&commit_ops);

        // Generate remaining traces in parallel (register, halt).
        // chunk_and_generate already handled cpu, lt, memw, load, mul, dvrm, branch above.
        let (register_trace, halt_trace);
        #[cfg(feature = "parallel")]
        {
            let (register_val, halt_val) = rayon::join(
                || register::generate_register_trace(&register_final_state),
                || halt::generate_halt_trace(halt_timestamp),
            );
            register_trace = register_val;
            halt_trace = halt_val;
        }
        #[cfg(not(feature = "parallel"))]
        {
            register_trace = register::generate_register_trace(&register_final_state);
            halt_trace = halt::generate_halt_trace(halt_timestamp);
        }

        // Create empty PAGE tables for legacy API
        // (caller should use from_elf_and_logs for proper PAGE table support)
        let pages = Vec::new();
        let page_configs = Vec::new();

        Ok(Traces {
            cpus,
            bitwise,
            lts,
            memws,
            loads,
            decode,
            muls,
            dvrms,
            pages,
            page_configs,
            register: register_trace,
            branches,
            halt: halt_trace,
            commit: commit_trace,
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
        max_rows: &super::MaxRowsConfig,
    ) -> Result<Self, Error> {
        // Generate full traces (including full 2^20 bitwise table with multiplicities)
        let mut traces = Self::from_logs(logs, instructions, max_rows)?;

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
        max_rows: &super::MaxRowsConfig,
    ) -> Result<Self, Error> {
        Self::from_logs_trimmed(logs, instructions, max_rows)
    }
}
