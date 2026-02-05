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
//! // Use traces.cpu, traces.bitwise, traces.lt, traces.memw, traces.load, traces.memory_init
//! ```

use std::collections::HashMap;

use executor::elf::Elf;
use executor::vm::instruction::decoding::Instruction;
use executor::vm::logs::Log;
use executor::vm::memory::U64HashMap;
use stark::trace::TraceTable;

use super::bitwise::{self, BitwiseOperation, BitwiseOperationType};
use super::branch::{self, BranchOperation};
use super::cpu::{self, CpuOperation};
use super::decode::{self, PcToRow};
use super::halt;
use super::load::{self, LoadOperation};
use super::lt::{self, LtOperation};
use super::memw::{self, MemwOperation};
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
                let word_addr = segment.base_addr + (i as u64 * 4);
                // Split 32-bit word into 4 bytes (little-endian)
                for byte_offset in 0..4u64 {
                    let byte_addr = word_addr + byte_offset;
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

    // Pack store value as individual bytes (per spec: memory uses 8 range-checked Bytes)
    let mut value_bytes = [0u64; 8];
    for (j, byte) in value_bytes.iter_mut().take(byte_count).enumerate() {
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

    // Collect all pages needed from ELF segments and build init data
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

                let page_data =
                    elf_page_data.entry(page_base).or_insert_with(|| vec![0u8; page_size]);
                page_data[offset] = byte_value;
            }
        }
    }

    // Add stack pages covering from STACK_TOP down to stack_bottom
    // (same as in generate_page_tables)
    let stack_size = 4096u64;
    let stack_bottom = page::STACK_TOP - stack_size;
    let stack_top_page = page::page_base_for_address(page::STACK_TOP, page_size);
    page_bases.insert(stack_top_page);
    let stack_bottom_page = page::page_base_for_address(stack_bottom, page_size);
    if stack_bottom_page != stack_top_page {
        page_bases.insert(stack_bottom_page);
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
            bitwise_ops.push(BitwiseOperation::single_byte(BitwiseOperationType::IsByte, init));

            // C2: IS_BYTE[fini]
            bitwise_ops.push(BitwiseOperation::single_byte(BitwiseOperationType::IsByte, fini));
        }
    }

    bitwise_ops
}

// =============================================================================
// Memory Coverage Debug
// =============================================================================

/// Debug function to verify memory bus token balance.
/// Traces MEMW operations and compares with PAGE table expected values.
fn debug_memory_coverage(
    memw_ops: &[MemwOperation],
    page_configs: &[PageConfig],
    memory_state: &MemoryState,
    elf: &Elf,
) {
    use std::collections::{BTreeMap, BTreeSet};

    // Build ELF init data map (address -> init_value)
    let mut elf_init: HashMap<u64, u8> = HashMap::new();
    for segment in &elf.data {
        for (i, &word) in segment.values.iter().enumerate() {
            let word_addr = segment.base_addr + (i as u64 * 4);
            for byte_offset in 0..4u64 {
                let byte_addr = word_addr + byte_offset;
                let byte_value = ((word >> (byte_offset * 8)) & 0xFF) as u8;
                elf_init.insert(byte_addr, byte_value);
            }
        }
    }

    // Track per-address: first access (old_ts, old_val) and last access (ts, val)
    #[derive(Debug, Default)]
    struct AddrInfo {
        first_old_ts: u64,
        first_old_val: u64,
        last_ts: u64,
        last_val: u64,
        access_count: u32,
    }
    let mut addr_info: BTreeMap<u64, AddrInfo> = BTreeMap::new();

    // Process MEMW operations to find first and last access per byte address
    for op in memw_ops {
        if op.is_register {
            continue; // Skip registers
        }
        for i in 0..op.width as usize {
            let addr = op.base_address + i as u64;
            let entry = addr_info.entry(addr).or_default();
            if entry.access_count == 0 {
                // First access
                entry.first_old_ts = op.old_timestamp[i];
                entry.first_old_val = op.old[i];
            }
            // Always update last access
            entry.last_ts = op.timestamp;
            entry.last_val = op.value[i];
            entry.access_count += 1;
        }
    }

    eprintln!("=== Memory Token Debug ===");

    // First, print the ENTIRE memory_state to see what PAGE tables will use
    eprintln!("\n=== memory_state contents (what PAGE uses for final values) ===");
    let mut mem_entries: Vec<_> = memory_state.cells.iter().collect();
    mem_entries.sort_by_key(|(addr, _)| *addr);
    for (addr, (value, timestamp)) in &mem_entries {
        eprintln!("  addr=0x{:016x}: final_val={}, final_ts={}", addr, value, timestamp);
    }
    eprintln!("memory_state total entries: {}", mem_entries.len());
    eprintln!("=== end memory_state ===\n");

    eprintln!("MEMW memory addresses accessed: {}", addr_info.len());

    // Count MEMW memory operations and bytes
    let mut memw_mem_ops = 0;
    let mut memw_mem_bytes = 0;
    for op in memw_ops {
        if !op.is_register {
            memw_mem_ops += 1;
            memw_mem_bytes += op.width as usize;
            eprintln!("  MEMW mem op: addr=0x{:016x} width={} ts={} is_read={}",
                op.base_address, op.width, op.timestamp, op.is_read);
            for i in 0..op.width as usize {
                eprintln!("    byte[{}]: old_ts={} old_val={} new_val={}",
                    i, op.old_timestamp[i], op.old[i], op.value[i]);
            }
        }
    }
    eprintln!("MEMW memory ops: {}, total bytes: {}", memw_mem_ops, memw_mem_bytes);

    // Also show MEMW register operations
    eprintln!("\n=== MEMW Register Operations ===");
    let mut memw_reg_ops = 0;
    for op in memw_ops {
        if op.is_register {
            memw_reg_ops += 1;
            eprintln!("  MEMW reg op: addr={} width={} ts={} is_read={}",
                op.base_address, op.width, op.timestamp, op.is_read);
            for i in 0..op.width as usize {
                // For registers, old[i] and value[i] are 32-bit words
                eprintln!("    word[{}]: old_ts={} old_val=0x{:08x} new_val=0x{:08x}",
                    i, op.old_timestamp[i], op.old[i] as u32, op.value[i] as u32);
            }
        }
    }
    eprintln!("MEMW register ops: {}", memw_reg_ops);

    // Check SP (register x2): should have init=STACK_TOP
    eprintln!("\n=== SP Init Check ===");
    eprintln!("Expected SP init: 0x{:016x}", page::STACK_TOP);
    // Find first access to SP (address 4 for low word, 5 for high word)
    let sp_lo_addr = register::register_base_address(2);  // x2 = SP
    let sp_hi_addr = sp_lo_addr + 1;
    for op in memw_ops {
        if op.is_register && op.base_address == sp_lo_addr {
            let sp_old = (op.old[0] as u64) | ((op.old[1] as u64) << 32);
            eprintln!("First SP access: old_val=0x{:016x} (lo=0x{:08x}, hi=0x{:08x})",
                sp_old, op.old[0] as u32, op.old[1] as u32);
            if sp_old != page::STACK_TOP {
                eprintln!("!!! SP INIT MISMATCH! Expected 0x{:016x}, got 0x{:016x}",
                    page::STACK_TOP, sp_old);
            }
            break;
        }
    }
    eprintln!("=== End SP Check ===\n");

    // Check register token balance for t0 (x5, addresses 10-11)
    eprintln!("=== Register Token Debug (t0/x5) ===");
    let t0_addr = register::register_base_address(5);  // x5 = t0
    eprintln!("t0 addresses: {} (lo), {} (hi)", t0_addr, t0_addr + 1);
    eprintln!("Expected init: 0x00000000");

    // Find all MEMW accesses to t0
    let mut t0_ops: Vec<_> = memw_ops.iter()
        .filter(|op| op.is_register && op.base_address == t0_addr)
        .collect();
    eprintln!("t0 MEMW ops: {}", t0_ops.len());
    for op in &t0_ops {
        eprintln!("  ts={} is_read={} old_ts={} old_val_lo=0x{:08x} val_lo=0x{:08x}",
            op.timestamp, op.is_read, op.old_timestamp[0], op.old[0] as u32, op.value[0] as u32);
    }

    // What should REGISTER table have?
    // REG-C1 receives: (1, 10, 0, 0, 0), (1, 11, 0, 0, 0)
    // REG-C2 sends: (1, 10, final_ts, 0, final_val_lo), (1, 11, final_ts, 0, final_val_hi)
    // First MEMW should send: (1, 10, 0, 0, 0), (1, 11, 0, 0, 0) to cancel REG-C1
    // Last MEMW should receive: (1, 10, final_ts, 0, final_val_lo) to cancel REG-C2
    if let Some(first) = t0_ops.first() {
        eprintln!("First t0 access: old_ts={} old_val=0x{:08x}", first.old_timestamp[0], first.old[0] as u32);
        if first.old_timestamp[0] != 0 || first.old[0] != 0 {
            eprintln!("!!! FIRST ACCESS MISMATCH! Expected old_ts=0, old_val=0");
        }
    }
    if let Some(last) = t0_ops.last() {
        eprintln!("Last t0 access: ts={} val=0x{:08x}", last.timestamp, last.value[0] as u32);
    }
    eprintln!("=== End Register Token Debug ===\n");

    // Count PAGE table bytes
    let total_page_bytes: usize = page_configs.iter().map(|c| c.page_size).sum();
    eprintln!("PAGE total bytes: {}", total_page_bytes);

    // Check each address
    let mut mismatches = Vec::new();
    for (&addr, info) in &addr_info {
        // Expected init value from ELF (or 0)
        let expected_init = elf_init.get(&addr).copied().unwrap_or(0) as u64;

        // Expected final value/timestamp from memory_state
        let (expected_final_val, expected_final_ts) = memory_state
            .cells
            .get(&addr)
            .map(|&(val, ts)| (val as u64, ts))
            .unwrap_or((expected_init, 0));

        // Check init: MEMW first old should match PAGE init (ts=0, val=init)
        let init_ts_ok = info.first_old_ts == 0;
        let init_val_ok = info.first_old_val == expected_init;

        // Check fini: MEMW last should match PAGE fini
        let fini_ts_ok = info.last_ts == expected_final_ts;
        let fini_val_ok = info.last_val == expected_final_val;

        if !init_ts_ok || !init_val_ok || !fini_ts_ok || !fini_val_ok {
            mismatches.push((
                addr,
                info,
                expected_init,
                expected_final_val,
                expected_final_ts,
                init_ts_ok,
                init_val_ok,
                fini_ts_ok,
                fini_val_ok,
            ));
        }
    }

    if mismatches.is_empty() {
        eprintln!("All token values match ✓");
    } else {
        eprintln!("TOKEN MISMATCHES ({}):", mismatches.len());
        for (addr, info, exp_init, exp_fini_val, exp_fini_ts, init_ts_ok, init_val_ok, fini_ts_ok, fini_val_ok) in &mismatches {
            eprintln!("  addr=0x{:016x}:", addr);
            eprintln!("    MEMW first: old_ts={}, old_val={}", info.first_old_ts, info.first_old_val);
            eprintln!("    PAGE init:  ts=0, val={}", exp_init);
            if !init_ts_ok { eprintln!("      ^ init_ts MISMATCH!"); }
            if !init_val_ok { eprintln!("      ^ init_val MISMATCH!"); }
            eprintln!("    MEMW last:  ts={}, val={}", info.last_ts, info.last_val);
            eprintln!("    PAGE fini:  ts={}, val={}", exp_fini_ts, exp_fini_val);
            if !fini_ts_ok { eprintln!("      ^ fini_ts MISMATCH!"); }
            if !fini_val_ok { eprintln!("      ^ fini_val MISMATCH!"); }
        }
    }

    // Also show page coverage
    let mem_addrs: BTreeSet<u64> = addr_info.keys().copied().collect();
    let mut uncovered = Vec::new();
    for &addr in &mem_addrs {
        let covered = page_configs.iter().any(|cfg| {
            addr >= cfg.page_base && (addr - cfg.page_base) < cfg.page_size as u64
        });
        if !covered {
            uncovered.push(addr);
        }
    }

    if !uncovered.is_empty() {
        eprintln!("UNCOVERED addresses ({}):", uncovered.len());
        for addr in uncovered.iter().take(20) {
            eprintln!("  0x{:016x}", addr);
        }
    } else {
        eprintln!("All memory addresses covered by PAGE tables ✓");
    }

    eprintln!("PAGE tables ({}):", page_configs.len());
    for cfg in page_configs {
        eprintln!("  0x{:016x} - 0x{:016x} (size={})",
            cfg.page_base, cfg.page_base.wrapping_add(cfg.page_size as u64 - 1), cfg.page_size);
    }

    // Detailed signature comparison for first accessed address
    if let Some((&addr, info)) = addr_info.iter().next() {
        eprintln!("\n=== Signature Comparison for addr=0x{:016x} ===", addr);

        // Compute what PAGE would use
        let page_base = page_configs.iter()
            .find(|cfg| addr >= cfg.page_base && (addr - cfg.page_base) < cfg.page_size as u64)
            .map(|cfg| cfg.page_base)
            .unwrap_or(0);
        let offset = addr - page_base;
        let page_base_lo = (page_base & 0xFFFF_FFFF) as u64;
        let page_base_hi = (page_base >> 32) as u64;

        // Expected init from ELF
        let expected_init = elf_init.get(&addr).copied().unwrap_or(0);

        // Expected final from memory_state
        let (expected_final_val, expected_final_ts) = memory_state
            .cells
            .get(&addr)
            .map(|&(val, ts)| (val as u64, ts))
            .unwrap_or((expected_init as u64, 0));

        eprintln!("PAGE C3 (recv init):");
        eprintln!("  is_reg=0, addr_lo={} (base_lo {} + offset {}), addr_hi={}, ts=0, val={}",
            page_base_lo + offset, page_base_lo, offset, page_base_hi, expected_init);

        eprintln!("PAGE C4 (send fini):");
        eprintln!("  is_reg=0, addr_lo={}, addr_hi={}, ts_lo={}, ts_hi={}, val={}",
            page_base_lo + offset, page_base_hi,
            expected_final_ts & 0xFFFF_FFFF, expected_final_ts >> 32, expected_final_val);

        eprintln!("MEMW first access (send old):");
        eprintln!("  is_reg=0, addr_lo={}, addr_hi={}, old_ts_lo={}, old_ts_hi={}, old_val={}",
            addr & 0xFFFF_FFFF, addr >> 32,
            info.first_old_ts & 0xFFFF_FFFF, info.first_old_ts >> 32, info.first_old_val);

        eprintln!("MEMW last access (recv new):");
        eprintln!("  is_reg=0, addr_lo={}, addr_hi={}, ts_lo={}, ts_hi={}, val={}",
            addr & 0xFFFF_FFFF, addr >> 32,
            info.last_ts & 0xFFFF_FFFF, info.last_ts >> 32, info.last_val);

        // Check if addr_lo matches
        let page_addr_lo = page_base_lo + offset;
        let memw_addr_lo = addr & 0xFFFF_FFFF;
        if page_addr_lo != memw_addr_lo {
            eprintln!("!!! ADDR_LO MISMATCH: PAGE={}, MEMW={}", page_addr_lo, memw_addr_lo);
        }
        eprintln!("=== End Signature Comparison ===");
    }

    eprintln!("=== End Memory Token Debug ===");

    // Count expected IsByte operations from PAGE tables
    eprintln!("\n=== IsByte Bus Debug ===");
    let total_page_bytes: usize = page_configs.iter().map(|c| c.page_size).sum();
    let expected_isbyte_sends = total_page_bytes * 2;  // C1 (init) + C2 (fini) per byte
    eprintln!("PAGE IsByte sends expected: {} (from {} bytes × 2)", expected_isbyte_sends, total_page_bytes);
    eprintln!("=== End IsByte Debug ===");
}

// =============================================================================
// PAGE Table Generation
// =============================================================================

/// Generates PAGE tables for memory initialization and finalization.
///
/// Creates one PAGE table per memory page covering:
/// 1. ELF segments (code, data, BSS)
/// 2. Stack region (from STACK_TOP - stack_size to STACK_TOP)
///
/// Each PAGE table contains initial values from ELF and final state from execution.
fn generate_page_tables(
    elf: &Elf,
    memory_state: &MemoryState,
) -> (
    Vec<TraceTable<GoldilocksField, GoldilocksExtension>>,
    Vec<PageConfig>,
) {
    use std::collections::BTreeSet;

    let page_size = page::DEFAULT_PAGE_SIZE;

    // Collect all pages needed from ELF segments
    let mut page_bases: BTreeSet<u64> = BTreeSet::new();
    let mut elf_page_data: HashMap<u64, Vec<u8>> = HashMap::new();

    for segment in &elf.data {
        for (i, &word) in segment.values.iter().enumerate() {
            let word_addr = segment.base_addr + (i as u64 * 4);

            // For each byte in the 32-bit word
            for byte_offset in 0..4u64 {
                let byte_addr = word_addr + byte_offset;
                let byte_value = ((word >> (byte_offset * 8)) & 0xFF) as u8;

                let page_base = page::page_base_for_address(byte_addr, page_size);
                let offset = page::offset_in_page(byte_addr, page_size);

                page_bases.insert(page_base);

                // Store initial values for this page
                let page_data = elf_page_data.entry(page_base).or_insert_with(|| vec![0u8; page_size]);
                page_data[offset] = byte_value;
            }
        }
    }

    // Add stack pages covering from STACK_TOP down to stack_bottom
    // Stack grows downward from STACK_TOP, so we need pages for both ends
    // TODO: Make this configurable via MemoryInitConfig
    let stack_size = 4096u64; // 1 page for now
    let stack_bottom = page::STACK_TOP - stack_size;
    // Add page containing STACK_TOP (where SP starts and first accesses happen)
    let stack_top_page = page::page_base_for_address(page::STACK_TOP, page_size);
    page_bases.insert(stack_top_page);
    // Also add page containing stack_bottom (in case stack grows that far)
    let stack_bottom_page = page::page_base_for_address(stack_bottom, page_size);
    if stack_bottom_page != stack_top_page {
        page_bases.insert(stack_bottom_page);
    }

    // Build final state map from memory_state
    let final_state: FinalStateMap = memory_state
        .cells
        .iter()
        .map(|(&addr, &(value, timestamp))| {
            (addr, FinalByteState { timestamp, value })
        })
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

    /// DECODE instruction decoding table (preprocessed from ELF)
    pub decode: TraceTable<GoldilocksField, GoldilocksExtension>,

    /// PAGE tables for memory initialization/finalization (one per page)
    pub pages: Vec<TraceTable<GoldilocksField, GoldilocksExtension>>,

    /// Page configurations (for bus interactions)
    pub page_configs: Vec<PageConfig>,

    /// REGISTER table for register initialization/finalization
    pub register: TraceTable<GoldilocksField, GoldilocksExtension>,

    /// BRANCH target calculation trace
    pub branch: TraceTable<GoldilocksField, GoldilocksExtension>,

    /// HALT single-row table for program termination
    pub halt: TraceTable<GoldilocksField, GoldilocksExtension>,

    /// PC to row mapping for DECODE multiplicities (internal use)
    pc_to_row: PcToRow,
}

impl Traces {
    /// Generates all traces from ELF and execution logs using phased collection.
    ///
    /// The phases are:
    /// 0. ELF → DECODE (preprocessed table)
    /// 1. Logs → CPU operations
    /// 2. CPU ops → MEMW, LOAD, LT, Bitwise, Branch (state tracking for MEMW/LOAD)
    /// 3. MEMW → LT operations (timestamp ordering)
    /// 4. LT, MEMW, Branch → Bitwise lookups
    /// 5. Generate all traces including PAGE tables
    pub fn from_elf_and_logs(elf: &Elf, logs: &[Log]) -> Result<Self, Error> {
        // =====================================================================
        // PHASE 0: ELF → DECODE
        // =====================================================================
        let elf_tables = decode::tables_from_elf(elf)
            .map_err(|e| Error::Execution(format!("Failed to process ELF: {e}")))?;

        // Extract instructions map for CPU ops collection
        let instructions = decode::instructions_from_elf(elf)
            .map_err(|e| Error::Execution(format!("Failed to parse instructions: {e}")))?;

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

        // =====================================================================
        // PHASE 3: MEMW → LT (timestamp ordering and overflow checks)
        // =====================================================================
        lt_ops.extend(collect_lt_from_memw(&memw_ops));

        // =====================================================================
        // PHASE 4: All → Bitwise lookups
        // =====================================================================
        bitwise_ops.extend(collect_bitwise_from_lt(&lt_ops));
        bitwise_ops.extend(collect_bitwise_from_memw(&memw_ops));
        bitwise_ops.extend(collect_bitwise_from_branch(&branch_ops));
        // PAGE tables do IS_BYTE lookups for init and fini values (C1, C2)
        let before_page = bitwise_ops.len();
        bitwise_ops.extend(collect_bitwise_from_page(elf, &memory_state));
        let after_page = bitwise_ops.len();
        let page_isbyte_ops = after_page - before_page;
        eprintln!("collect_bitwise_from_page added {} IsByte operations", page_isbyte_ops);

        // =====================================================================
        // PHASE 5: Generate final traces
        // =====================================================================

        // Extract halt timestamp from the last ECALL instruction
        let halt_op = cpu_ops
            .iter()
            .rev()
            .find(|op| op.decode.op_ecall)
            .ok_or(Error::MissingHaltEcall)?;
        let halt_trace = halt::generate_halt_trace(halt_op.timestamp);

        let cpu = cpu::generate_cpu_trace(&cpu_ops);
        let lt = lt::generate_lt_trace(&lt_ops);
        let memw = memw::generate_memw_trace(&memw_ops);
        let load = load::generate_load_trace(&load_ops);
        let branch = branch::generate_branch_trace(&branch_ops);

        let mut bitwise = bitwise::generate_bitwise_trace();
        bitwise::update_multiplicities(&mut bitwise, &bitwise_ops);

        // Update DECODE multiplicities
        // Each CPU operation looks up the DECODE table once
        // Padding rows also look up pc=1 (the CPU padding entry)
        let mut decode = elf_tables.decode;
        let pc_to_row = elf_tables.pc_to_row;
        let num_padding_rows = cpu_ops.len().next_power_of_two() - cpu_ops.len();
        let mut decode_lookups: Vec<u64> = cpu_ops.iter().map(|op| op.decode.pc).collect();
        decode_lookups.extend(std::iter::repeat_n(cpu::CPU_PADDING_PC, num_padding_rows));
        decode::update_multiplicities(&mut decode, &pc_to_row, &decode_lookups);

        // Generate PAGE tables from ELF and final memory state
        let (pages, page_configs) = generate_page_tables(elf, &memory_state);

        // Debug: Check memory coverage and token values
        debug_memory_coverage(&memw_ops, &page_configs, &memory_state, elf);

        // Generate REGISTER table from final register state
        let register_final_state = register_state.to_final_state_map();
        let register_trace = register::generate_register_trace(&register_final_state);

        Ok(Traces {
            cpu,
            bitwise,
            lt,
            memw,
            load,
            decode,
            pages,
            page_configs,
            register: register_trace,
            branch,
            halt: halt_trace,
            pc_to_row,
        })
    }

    /// Generates all traces from execution logs (legacy API).
    ///
    /// This is a compatibility wrapper. Prefer `from_elf_and_logs` for new code
    /// as it generates PAGE tables from ELF data.
    ///
    /// Note: This creates empty PAGE tables since no ELF is provided.
    pub fn from_logs(logs: &[Log], instructions: U64HashMap<Instruction>) -> Result<Self, Error> {
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

        // =====================================================================
        // PHASE 3: MEMW → LT (timestamp ordering and overflow checks)
        // =====================================================================
        lt_ops.extend(collect_lt_from_memw(&memw_ops));

        // =====================================================================
        // PHASE 4: All → Bitwise lookups
        // =====================================================================
        bitwise_ops.extend(collect_bitwise_from_lt(&lt_ops));
        bitwise_ops.extend(collect_bitwise_from_memw(&memw_ops));
        bitwise_ops.extend(collect_bitwise_from_branch(&branch_ops));

        // =====================================================================
        // PHASE 5: Generate final traces
        // =====================================================================

        // Extract halt timestamp from the last ECALL instruction
        let halt_op = cpu_ops
            .iter()
            .rev()
            .find(|op| op.decode.op_ecall)
            .ok_or(Error::MissingHaltEcall)?;
        let halt_trace = halt::generate_halt_trace(halt_op.timestamp);

        let cpu = cpu::generate_cpu_trace(&cpu_ops);
        let lt = lt::generate_lt_trace(&lt_ops);
        let memw = memw::generate_memw_trace(&memw_ops);
        let load = load::generate_load_trace(&load_ops);
        let branch = branch::generate_branch_trace(&branch_ops);

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

        // Create empty PAGE tables for legacy API
        // (caller should use from_elf_and_logs for proper PAGE table support)
        let pages = Vec::new();
        let page_configs = Vec::new();

        // Generate REGISTER table from final register state
        let register_final_state = register_state.to_final_state_map();
        let register_trace = register::generate_register_trace(&register_final_state);

        Ok(Traces {
            cpu,
            bitwise,
            lt,
            memw,
            load,
            decode,
            pages,
            page_configs,
            register: register_trace,
            branch,
            halt: halt_trace,
            pc_to_row,
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
    ) -> Result<Self, Error> {
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
    ) -> Result<Self, Error> {
        Self::from_logs_trimmed(logs, instructions)
    }
}
