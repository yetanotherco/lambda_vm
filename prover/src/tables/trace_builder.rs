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
//! PHASE 2: CPU ops → MEMW, MEMW_A, MEMW_R, LOAD, LT, Bitwise, KECCAK (with state tracking for MEMW/LOAD/ECALL)
//! PHASE 3: MEMW/MEMW_A → LT ops (timestamp ordering); MEMW_R uses IS_HALFWORD instead
//! PHASE 4: LT, MEMW_A, MEMW_R, KECCAK → Bitwise lookups
//! PHASE 5: Generate all traces (including KECCAK core, KECCAK_RND, KECCAK_RC)
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
#[cfg(feature = "disk-spill")]
use std::collections::HashSet;

use executor::elf::Elf;
use executor::vm::instruction::decoding::Instruction;
use executor::vm::logs::Log;
use executor::vm::memory::U64HashMap;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
#[cfg(feature = "disk-spill")]
use stark::storage_mode::StorageMode;
use stark::trace::TraceTable;

use super::bitwise::{self, BitwiseOperation, BitwiseOperationType};
use super::branch::{self, BranchOperation};
use super::bytewise;
use super::commit::{self, CommitOperation};
use super::cpu::{self, CpuOperation};
use super::cpu32;
use super::decode;
use super::dvrm::{self, DvrmOperation};
use super::ec_scalar;
use super::ecdas;
use super::ecsm;
use super::eq;
use super::halt;
use super::keccak::{self, KeccakOperation};
use super::keccak_rc;
use super::keccak_rnd::{self, KeccakRoundOperation};
use super::load::{self, LoadOperation};
use super::local_to_global;
use super::lt::{self, LtOperation};
use super::memw::{self, MemwOperation};
use super::memw_aligned;
use super::memw_register::{self, RegRow};
use super::mul::{self, MulOperation};
use super::page::{self, FinalByteState, FinalStateMap, PageConfig};
use super::register::{self, FinalRegisterStateMap, FinalRegisterWordState};
use super::shift::{self, ShiftOperation};
use super::store;
use super::types::{GoldilocksExtension, GoldilocksField};
use crate::Error;
use crate::paged_mem::{ImageSource, PagedMem};

// =============================================================================
// Memory and Register State Tracking
// =============================================================================

/// Memory cell state: (value_byte, last_write_timestamp)
type MemoryCell = (u8, u64);

/// Register state: (value, last_write_timestamp)
type RegisterCell = (u64, u64);

/// Memory state tracker for generating MEMW/LOAD traces.
struct MemoryState {
    /// Per byte-address `(value, timestamp)`, as a dense per-page store. This is
    /// the hot structure — `read_byte`/`write_byte` hit it on every memory access
    /// during the replay, and it's rebuilt each epoch — so a per-page array (small
    /// page-map lookup + dense indexing, no per-cell hashing or rehash-on-grow)
    /// is both lighter and faster than a per-cell `HashMap`.
    cells: PagedMem<MemoryCell>,
}

impl MemoryState {
    fn new() -> Self {
        Self {
            cells: PagedMem::new((0, 0)),
        }
    }

    /// Initialize memory state from a pre-built initial-memory image.
    ///
    /// Pre-populates all starting bytes with timestamp=0 so that when MEMW first
    /// accesses an address, it gets the correct initial value for `old_value`.
    /// This is required for the Memory bus to balance (MEMW-M1 must match PAGE-C3).
    fn from_image<I: ImageSource>(image: &I) -> Self {
        let mut cells = PagedMem::new((0, 0));
        for (addr, value) in image.image_iter() {
            cells.set(addr, (value, 0));
        }
        Self { cells }
    }

    /// Number of distinct pages that contain at least one cell.
    #[cfg(feature = "disk-spill")]
    fn unique_page_count(&self, page_size: u64) -> u64 {
        debug_assert!(
            page_size.is_power_of_two(),
            "page_size must be a power of two for the bitmask to work"
        );
        let mask = !(page_size - 1);
        let pages: HashSet<u64> = self.cells.iter().map(|(a, _)| a & mask).collect();
        pages.len() as u64
    }

    /// Read a byte from memory. Returns (value, timestamp) or (0, 0) if never written.
    fn read_byte(&self, address: u64) -> MemoryCell {
        self.cells.get(address)
    }

    /// Write a byte to memory with the given timestamp.
    fn write_byte(&mut self, address: u64, value: u8, timestamp: u64) {
        self.cells.set(address, (value, timestamp));
    }

    /// Read multiple bytes. Returns arrays of values and timestamps.
    fn read_bytes(&self, base_address: u64, count: usize) -> ([u32; 8], [u64; 8]) {
        let mut values = [0u32; 8];
        let mut timestamps = [0u64; 8];
        for i in 0..count {
            let (val, ts) = self.read_byte(base_address.wrapping_add(i as u64));
            values[i] = val as u32;
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
    /// Synthetic x254 commit index register: (value, last_write_timestamp)
    index_register: (u32, u64),
    /// PC register x255: (value, last_write_timestamp)
    pc_register: RegisterCell,
}

impl RegisterState {
    fn new(entry_point: u64) -> Self {
        // Per spec/memory.typ: "register initialization happens at timestamp 1"
        // to enable loading of the PC via the CPU memory argument.
        let mut regs = [(0u64, 1u64); 32];
        // SP (x2) starts at STACK_TOP
        regs[2] = (page::STACK_TOP, 1);
        Self {
            regs,
            index_register: (0, 1),
            // PC register (x255) starts at entry_point, timestamp 1
            pc_register: (entry_point, 1),
        }
    }

    /// Seed register state from a register init vector (one value per row, in
    /// `register_word_address_list` order), so the first access in a continuation
    /// epoch reads the epoch's boundary register values as `old_value`. All initial
    /// timestamps are 1, matching the REGISTER table's init token. Mirrors
    /// `MemoryState::from_image`.
    fn from_init(init: &[u32]) -> Self {
        let word = |pos: usize| init.get(pos).copied().unwrap_or(0) as u64;
        let mut regs = [(0u64, 1u64); 32];
        for (reg, slot) in regs.iter_mut().enumerate() {
            let base = reg * 2;
            *slot = (word(base) | (word(base + 1) << 32), 1);
        }
        Self {
            regs,
            index_register: (init.get(register::X254_INDEX).copied().unwrap_or(0), 1),
            pc_register: (
                word(register::PC_LO_INDEX) | (word(register::PC_HI_INDEX) << 32),
                1,
            ),
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

    /// Read the PC register (x255). Returns (value, last_write_timestamp).
    fn read_pc(&self) -> RegisterCell {
        self.pc_register
    }

    /// Write the PC register (x255) with the given timestamp.
    fn write_pc(&mut self, value: u64, timestamp: u64) {
        self.pc_register = (value, timestamp);
    }

    /// Read the synthetic x254 commit index register.
    fn read_index(&self) -> (u32, u64) {
        self.index_register
    }

    /// Write the synthetic x254 commit index register.
    fn write_index(&mut self, value: u32, timestamp: u64) {
        self.index_register = (value, timestamp);
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

        // Synthetic x254 commit index at address 508 (single-word per spec).
        {
            let (value, timestamp) = self.index_register;
            map.insert(
                register::register_base_address(254),
                FinalRegisterWordState { timestamp, value },
            );
        }

        // PC register (x255) at addresses 510, 511
        {
            let (value, timestamp) = self.pc_register;
            let base_addr = register::register_base_address(255);
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
    let f = &op.decode.fields;
    (f.mem_bytes(), f.mem_signed())
}

/// Pack a 64-bit register value into the MEMW value format.
///
/// For register operations, values are packed as [lo32, hi32, 0, 0, 0, 0, 0, 0].
fn pack_register_value(value: u64) -> [u32; 8] {
    [
        (value & 0xFFFF_FFFF) as u32,
        (value >> 32) as u32,
        0,
        0,
        0,
        0,
        0,
        0,
    ]
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
    // for the first access to any register/memory location. The +4 stride reserves
    // per-cycle slots for M1/M3/M5 register accesses and the inline PC read.
    // Exactly 4 so that inline PC's prev_ts = timestamp - 3 = 1 on the first row,
    // matching the REGISTER table's initial PC token at timestamp 1 (per spec/memory.typ).
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

/// Destination table for a `MemwOperation`.
///
/// The order of the checks matters and must never change: register ops would
/// also pass `is_aligned_op`, so MEMW_R is decided first, then MEMW_A, and the
/// rest goes to the general MEMW table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MemwRoute {
    Register,
    Aligned,
    General,
}

/// The single classification used everywhere a `MemwOperation` is routed to a
/// table — the walk's [`MemwBuckets`] and the sizing pass (`count_table_lengths`)
/// share it, so their routing cannot drift.
#[inline]
fn classify_memw(op: &MemwOperation) -> MemwRoute {
    if is_register_op(op) {
        MemwRoute::Register
    } else if is_aligned_op(op) {
        MemwRoute::Aligned
    } else {
        MemwRoute::General
    }
}

/// Routes each `MemwOperation` into its destination table bucket at CREATION time
/// (register fast-path / aligned / general), so the walk fills the three buckets directly
/// and no separate routing pass is needed downstream. Classification order is register
/// first, then aligned (see [`classify_memw`]), and push order within each bucket is the
/// walk's insertion order — the buckets are fully deterministic, which the per-cell
/// multiplicity counts rely on.
///
/// ## Direct-to-column register fill
///
/// For the register fast path we do NOT materialize a `Vec<MemwOperation>`. Ops that route
/// to MEMW_R are stored as compact [`RegRow`]s (`register_rows`) and later filled directly
/// into the MEMW_R columns. The `aligned` / `general` buckets hold `MemwOperation`s — an op
/// that FAILS `is_register_op` is routed there (aligned if `is_aligned_op`, else general).
#[derive(Default)]
struct MemwBuckets {
    /// Compact register rows (filled directly into the MEMW_R columns).
    register_rows: Vec<RegRow>,
    aligned: Vec<MemwOperation>,
    general: Vec<MemwOperation>,
}

impl MemwBuckets {
    fn with_register_capacity(n: usize) -> Self {
        Self {
            register_rows: Vec::with_capacity(n),
            aligned: Vec::new(),
            general: Vec::new(),
        }
    }

    #[inline]
    fn push(&mut self, op: MemwOperation) {
        match classify_memw(&op) {
            MemwRoute::Register => self.register_rows.push(RegRow::from_memw(&op)),
            MemwRoute::Aligned => self.aligned.push(op),
            MemwRoute::General => self.general.push(op),
        }
    }
    fn extend_ops(&mut self, ops: impl IntoIterator<Item = MemwOperation>) {
        for op in ops {
            self.push(op);
        }
    }
}

/// Sink for `MemwOperation`s so `collect_register_ops_from_cpu` can feed either a plain
/// `Vec` (the `count_table_lengths` trace-sizing pass) or the classifying
/// [`MemwBuckets`] (the walk).
trait MemwSink {
    fn push_op(&mut self, op: MemwOperation);

    /// Fast path for a 2-word register access (M1/M3/M5 and precompile register I/O).
    ///
    /// The caller passes the compact, pre-decomposed fields. The sink decides routing
    /// (via the same predicate as `is_register_op`): if the timestamp delta admits the
    /// op into MEMW_R it fills a compact [`RegRow`] DIRECTLY — no `MemwOperation` is
    /// built. Only on the (rare) fallback (delta out of IS_HALF range, or upper-limb
    /// mismatch) does it build the `MemwOperation` (via [`build_reg_fallback`]) and
    /// route it to the aligned/general bucket exactly as before.
    ///
    /// `reg_addr` is `2 * reg_index`; `val`/`old` are the two 32-bit halves of the new
    /// and previous register words; `old_ts` is the (shared) old_timestamp of both words.
    #[inline]
    fn push_reg_access(
        &mut self,
        reg_addr: u64,
        val: [u32; 2],
        old: [u32; 2],
        timestamp: u64,
        old_ts: u64,
        is_read: bool,
    ) {
        // Default impl (plain Vec): register accesses are still ordinary MemwOperations.
        self.push_op(build_reg_fallback(
            reg_addr, val, old, timestamp, old_ts, is_read,
        ));
    }
}
impl MemwSink for Vec<MemwOperation> {
    #[inline]
    fn push_op(&mut self, op: MemwOperation) {
        self.push(op);
    }
}
impl MemwSink for MemwBuckets {
    #[inline]
    fn push_op(&mut self, op: MemwOperation) {
        self.push(op);
    }

    #[inline]
    fn push_reg_access(
        &mut self,
        reg_addr: u64,
        val: [u32; 2],
        old: [u32; 2],
        timestamp: u64,
        old_ts: u64,
        is_read: bool,
    ) {
        // Mirror `is_register_op` for a width-2 register access whose two words share
        // `old_ts` (always true here by construction). If it passes, fill a RegRow
        // directly; otherwise fall back to the general/aligned MemwOperation path.
        if reg_ts_delta_in_range(timestamp, old_ts) {
            self.register_rows.push(RegRow::new(
                reg_addr, timestamp, val[0], val[1], old[0], old[1], old_ts, is_read,
            ));
        } else {
            let op = build_reg_fallback(reg_addr, val, old, timestamp, old_ts, is_read);
            debug_assert!(!is_register_op(&op), "reg fallback must not be MEMW_R");
            self.push(op);
        }
    }
}

/// Materialize the aligned/general `MemwOperation` for a register access that does
/// NOT fit the MEMW_R fast path. Register values pack as `[lo, hi, 0, …]` (see
/// [`pack_register_value`]) and both words share `old_ts`, so this rebuilds exactly
/// the op the fast-path callers would otherwise have routed to the buckets.
fn build_reg_fallback(
    reg_addr: u64,
    val: [u32; 2],
    old: [u32; 2],
    timestamp: u64,
    old_ts: u64,
    is_read: bool,
) -> MemwOperation {
    let value = [val[0], val[1], 0, 0, 0, 0, 0, 0];
    let old_value = [old[0], old[1], 0, 0, 0, 0, 0, 0];
    let old_timestamps = [old_ts, old_ts, 0, 0, 0, 0, 0, 0];
    MemwOperation::new(true, reg_addr, value, timestamp, 2, is_read)
        .with_old(old_value, old_timestamps)
}

/// Collects all derived operations from CPU operations in a single pass.
///
/// This includes:
/// - MEMW ops (register reads/writes M1/M3/M5, memory loads/stores M6/M7),
///   already routed into their MEMW_R / MEMW_A / MEMW buckets (see [`MemwBuckets`])
/// - LOAD ops (memory loads with sign/zero extension)
/// - LT ops (from SLT/BLT instructions)
/// - Bitwise lookups (from CPU operations)
///
/// MEMW and LOAD collection requires sequential processing with state tracking.
///
/// Returns: (memw_buckets, load_ops, lt_ops, shift_ops, bitwise_ops, commit_ops,
/// keccak_ops, cpu32_ops, ecsm_ops, ec_scalar_ops, ecdas_ops)
#[allow(clippy::type_complexity)]
fn collect_ops_from_cpu(
    cpu_ops: &[CpuOperation],
    memory_state: &mut MemoryState,
    register_state: &mut RegisterState,
) -> (
    MemwBuckets,
    Vec<LoadOperation>,
    Vec<LtOperation>,
    Vec<ShiftOperation>,
    Vec<BitwiseOperation>,
    Vec<CommitOperation>,
    Vec<KeccakOperation>,
    Vec<cpu32::Cpu32Operation>,
    Vec<ecsm::EcsmOperation>,
    Vec<ec_scalar::EcScalarOperation>,
    Vec<ecdas::EcdasOperation>,
) {
    let mut memw = MemwBuckets::with_register_capacity(cpu_ops.len() * 3);
    let mut load_ops = Vec::with_capacity(cpu_ops.len() / 8 + 1);
    let mut lt_ops = Vec::with_capacity(cpu_ops.len() / 10 + 1);
    let mut shift_ops = Vec::with_capacity(cpu_ops.len() / 10 + 1);
    let mut bitwise_ops = Vec::with_capacity(cpu_ops.len() * 4);
    let mut commit_ops = Vec::new();
    let mut keccak_ops = Vec::new();
    let mut cpu32_ops = Vec::new();
    let mut ecsm_ops = Vec::new();
    let mut ec_scalar_ops = Vec::new();
    let mut ecdas_ops = Vec::new();
    // Seed from the carried x254 (0 for a monolithic run or the first epoch) so a
    // continuation epoch indexes its commits globally, matching the x254 the
    // register binding transports across epochs. Resetting to 0 here would drift
    // from x254 and break the COMMIT chip's Memw token (see the drift assert below).
    let start_commit_index = register_state.read_index().0;
    let mut current_commit_index = start_commit_index;
    let mut commit_ecall_count = 0u32;

    for op in cpu_ops {
        // Word (`*W`) instructions delegate to the CPU32 table (built in program
        // order; its register accesses are still emitted via the shared register
        // collector below so the MEMW table balances).
        if op.decode.fields.word_instr {
            cpu32_ops.push(build_cpu32_op(op));
        }

        // --- MEMW and LOAD (require state tracking, order matters) ---

        // Collect memory operations for Load/Store instructions
        if op.decode.fields.is_load() {
            let (memw_op, load_op, lookups) = collect_load_op_from_cpu(op, memory_state);
            memw.push(memw_op);
            load_ops.push(load_op);
            bitwise_ops.extend(lookups);
        } else if op.decode.fields.is_store() {
            let memw_op = collect_store_op_from_cpu(op, memory_state);
            memw.push(memw_op);
        }

        // Collect register operations (M1, M3, M5)
        collect_register_ops_from_cpu(op, register_state, &mut memw);

        // Collect COMMIT ECALL memory operations (register reads/writes + byte reads)
        if op.ecall_commit {
            commit_ops.extend(expand_commit_operations_for_ecall(
                op,
                memory_state,
                current_commit_index as u64,
            ));
            let reg_commit_ops = collect_commit_memw_ops(op, register_state, memory_state);
            memw.extend_ops(reg_commit_ops);
            let count = u32::try_from(op.commit_count).expect("commit_count exceeds u32 range");
            current_commit_index = current_commit_index
                .checked_add(count)
                .expect("commit index exceeds u32 range");
            debug_assert_eq!(
                current_commit_index,
                register_state.read_index().0,
                "commit index drift: current_commit_index and register_state.index_register must stay in sync"
            );
            commit_ecall_count += 1;
        }

        // Collect KeccakPermute ECALL operations
        if op.ecall_keccak {
            let state_addr = op.keccak_state_addr;
            let mut input = [0u64; 25];
            for (i, lane) in input.iter_mut().enumerate() {
                let addr = state_addr
                    .checked_add(i as u64 * 8)
                    .expect("keccak state address range must be validated by the executor");
                let mut val = 0u64;
                for b in 0..8 {
                    let byte_addr = addr
                        .checked_add(b as u64)
                        .expect("keccak state address range must be validated by the executor");
                    let (byte_val, _ts) = memory_state.read_byte(byte_addr);
                    val |= (byte_val as u64) << (b * 8);
                }
                *lane = val;
            }
            let mut output = input;
            executor::vm::instruction::execution::keccak_f1600(&mut output);
            // collect_keccak_memw_ops handles memory_state + register_state updates
            let keccak_memw_ops =
                collect_keccak_memw_ops(op, &input, &output, memory_state, register_state);
            memw.extend_ops(keccak_memw_ops);
            keccak_ops.push(KeccakOperation {
                timestamp: op.timestamp,
                state_addr,
                input,
                output,
            });
        }

        // Collect ECSM ecall operations (memory I/O + the three table row sets)
        if op.ecall_ecsm {
            let (ecsm_memw, ecsm_op, ec_scalar_rows, ecdas_rows) =
                collect_ecsm_ops(op, memory_state, register_state);
            memw.extend_ops(ecsm_memw);
            ecsm_ops.push(ecsm_op);
            ec_scalar_ops.extend(ec_scalar_rows);
            ecdas_ops.extend(ecdas_rows);
        }

        // --- ALU chip dispatch (no state tracking) ---
        // Word (`*W`) instructions are delegated to CPU32 (which itself drives
        // the ALU chips); the main CPU does not send the ALU bus for them, so we
        // must not emit chip ops here. CPU32 op-generation is B5b.
        let f = op.decode.fields;
        if !f.word_instr {
            // LT: SLT / BLT / BGE, dispatched on the unified ALU bus. `invert`
            // (BGE/BGEU) is applied inside the LT chip (`out = lt XOR invert`).
            if f.is_lt() {
                lt_ops.push(LtOperation::new_with_invert(
                    op.rv1,
                    op.arg2,
                    f.alu_signed(),
                    f.alu_signed2_or_invert(),
                ));
            }
            // SHIFT: SLL/SRL/SRA. direction = invert bit (0 = left, 1 = right).
            // The full arg2 goes on the ALU bus as in2; the chip uses its low
            // byte for the (mod 32/64) computation.
            if f.is_shift() {
                shift_ops.push(ShiftOperation::new(
                    op.rv1,
                    op.arg2,
                    f.alu_signed2_or_invert(),
                    f.alu_signed(),
                    f.word_instr,
                ));
            }
        }

        // Collect CPU range-check bitwise lookups (ARE_BYTES + IS_HALF). Kept serial here:
        // it's only ~110 ms (a serial `.extend` into one growing Vec), and moving it to a
        // rayon `flat_map`-collect over 6.8 M per-op Vecs regressed p4 ~4× (alloc + merge).
        bitwise_ops.extend(op.collect_bitwise_ops());
    }

    // Each ecall generates count+1 operations (count real rows + 1 end row).
    // Count only this epoch's rows, so subtract the carried start index.
    debug_assert_eq!(
        commit_ops.len(),
        (current_commit_index - start_commit_index) as usize + commit_ecall_count as usize,
        "commit_ops count should match accumulated commit index plus end rows"
    );

    (
        memw,
        load_ops,
        lt_ops,
        shift_ops,
        bitwise_ops,
        commit_ops,
        keccak_ops,
        cpu32_ops,
        ecsm_ops,
        ec_scalar_ops,
        ecdas_ops,
    )
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
    let mut value_bytes = [0u32; 8];
    for (j, byte) in value_bytes.iter_mut().take(byte_count).enumerate() {
        *byte = ((loaded_value >> (j * 8)) & 0xFF) as u32;
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
        res_bytes.map(u64::from),
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
    // Bus 14: the MEMW Memory Write receiver reconstructs lo32/hi32 via a linear
    //   combination of all 8 bytes, so it must match the store value the CPU sends
    //   as [lo32, hi32] on the MEMORY bus (MEMOP) and that this STORE chip forwards
    //   as the MEMW write (the CPU no longer emits an inline store MEMW — see below).
    // Bus 16: only positions 0..byte_count participate (controlled by w2/w4/write8
    //   multiplicities), so extra bytes don't affect memory consistency.
    let mut value_bytes = [0u32; 8];
    for (j, byte) in value_bytes.iter_mut().enumerate() {
        *byte = ((store_value >> (j * 8)) & 0xFF) as u32;
    }

    // The STORE chip now owns this MEMW write (the CPU sends MEMORY instead of
    // the old inline M7). It uses the base timestamp — the same the CPU sends on
    // the MEMORY bus — per spec store.toml.
    let memw_op = MemwOperation::new(
        false, // is_register = false
        base_address,
        value_bytes,
        op.timestamp,
        byte_count as u8,
        false, // is_read = false (write)
    )
    .with_old(old_values, old_timestamps);

    // Update memory state at the base timestamp (matches the STORE MEMW write).
    memory_state.write_bytes(base_address, store_value, byte_count, op.timestamp);

    memw_op
}

/// Collects all MEMW ops and the ECSM / EC_SCALAR / ECDAS table ops for one ECSM ecall.
///
/// Timestamp scheme (within the instruction's 4-wide budget): the `x11`/`x12` register reads
/// and the `xG`/`k` memory reads happen at `T`; the `x10` register read and the EC_SCALAR
/// byte reads at `T + 1`; the `xR` memory writes at `T + 2`. Every read advances
/// `memory_state` / `register_state` (the offline read-old + write-new model), so later
/// accesses always observe a strictly smaller old timestamp.
#[allow(clippy::needless_range_loop)]
fn collect_ecsm_ops(
    op: &CpuOperation,
    memory_state: &mut MemoryState,
    register_state: &mut RegisterState,
) -> (
    Vec<MemwOperation>,
    ecsm::EcsmOperation,
    Vec<ec_scalar::EcScalarOperation>,
    Vec<ecdas::EcdasOperation>,
) {
    let t = op.timestamp;
    let addr_xr = register_state.read(10).0;
    let addr_xg = register_state.read(11).0;
    let addr_k = register_state.read(12).0;

    // Read the xG and k operands (32 little-endian bytes each) from memory.
    let mut xg = [0u8; 32];
    let mut k = [0u8; 32];
    for i in 0..32 {
        xg[i] = memory_state.read_byte(addr_xg.wrapping_add(i as u64)).0;
        k[i] = memory_state.read_byte(addr_k.wrapping_add(i as u64)).0;
    }

    let witness = ::ecsm::compute_witness(&k, &xg)
        .expect("ECSM witness: executor validates 0 < k < N and xG on curve");

    let mut memw_ops = Vec::with_capacity(47);

    // x11 -> addr_xG, x12 -> addr_k (register reads at T).
    for reg in [11u8, 12u8] {
        let (val, old_ts) = register_state.read(reg);
        let value = pack_register_value(val);
        memw_ops.push(
            MemwOperation::new(true, 2 * reg as u64, value, t, 2, true)
                .with_old(value, [old_ts, old_ts, 0, 0, 0, 0, 0, 0]),
        );
        register_state.write(reg, val, t);
    }

    // xG and k: 4 doubleword reads each at T.
    for (base, bytes) in [(addr_xg, &witness.x_g), (addr_k, &witness.k)] {
        for i in 0..4 {
            let addr = base.wrapping_add((8 * i) as u64);
            let mut value = [0u32; 8];
            let mut dword = 0u64;
            for j in 0..8 {
                value[j] = bytes[8 * i + j] as u32;
                dword |= (bytes[8 * i + j] as u64) << (8 * j);
            }
            let (_old, old_ts) = memory_state.read_bytes(addr, 8);
            memw_ops
                .push(MemwOperation::new(false, addr, value, t, 8, true).with_old(value, old_ts));
            memory_state.write_bytes(addr, dword, 8, t);
        }
    }

    // x10 -> addr_xR (register read at T + 1).
    {
        let (val, old_ts) = register_state.read(10);
        let value = pack_register_value(val);
        memw_ops.push(
            MemwOperation::new(true, 2 * 10, value, t + 1, 2, true)
                .with_old(value, [old_ts, old_ts, 0, 0, 0, 0, 0, 0]),
        );
        register_state.write(10, val, t + 1);
    }

    // EC_SCALAR byte reads of k at T + 1 (one per scalar byte).
    for offset in 0..32u64 {
        let addr = addr_k.wrapping_add(offset);
        let byte = k[offset as usize];
        let value = [byte as u32, 0, 0, 0, 0, 0, 0, 0];
        let (_v, old_ts) = memory_state.read_byte(addr);
        memw_ops.push(
            MemwOperation::new(false, addr, value, t + 1, 1, true)
                .with_old(value, [old_ts, 0, 0, 0, 0, 0, 0, 0]),
        );
        memory_state.write_byte(addr, byte, t + 1);
    }

    // xR writes at T + 2 (4 doublewords).
    for i in 0..4 {
        let addr = addr_xr.wrapping_add((8 * i) as u64);
        let mut value = [0u32; 8];
        let mut dword = 0u64;
        for j in 0..8 {
            value[j] = witness.x_r[8 * i + j] as u32;
            dword |= (witness.x_r[8 * i + j] as u64) << (8 * j);
        }
        let (old_vals, old_ts) = memory_state.read_bytes(addr, 8);
        memw_ops.push(
            MemwOperation::new(false, addr, value, t + 2, 8, false).with_old(old_vals, old_ts),
        );
        memory_state.write_bytes(addr, dword, 8, t + 2);
    }

    let ec_scalar_ops = ec_scalar::rows_for_scalar(t, addr_k, &witness.k);
    let ecdas_ops = witness
        .steps
        .iter()
        .cloned()
        .map(|step| ecdas::EcdasOperation { timestamp: t, step })
        .collect();
    let ecsm_op = ecsm::EcsmOperation {
        timestamp: t,
        addr_xg,
        addr_k,
        addr_xr,
        witness,
    };

    (memw_ops, ecsm_op, ec_scalar_ops, ecdas_ops)
}

/// Collects register read/write operations (M1, M3, M5) from CpuOperation,
/// pushing them into `memw_ops`.
fn collect_register_ops_from_cpu<S: MemwSink>(
    op: &CpuOperation,
    register_state: &mut RegisterState,
    memw_ops: &mut S,
) {
    let d = &op.decode.fields;
    // These register accesses happen for every real instruction. For non-word
    // rows the main CPU sends the MEMW lookups; for word (`*W`) rows the CPU32
    // table sends them. Either way the MEMW *table* receives the same record, so
    // we generate it here (in program order, for register-state timestamps).

    // M1: Read rs1 register at timestamp+0
    // Skip x0 (hardwired zero). x255 (the register where the pc is stored) is handled
    // via read_pc/write_pc since regs[] only covers indices 0..31.
    if d.read_register1 && d.rs1 != 0 {
        let reg_value = pack_register_value(op.rv1);
        let reg_addr = 2 * d.rs1 as u64;
        let (_old_val, old_ts) = if d.rs1 == 255 {
            register_state.read_pc()
        } else {
            register_state.read(d.rs1)
        };
        let ts = op.timestamp;
        // Direct fast path: fill a RegRow when routing to MEMW_R; push_reg_access
        // rebuilds the identical MemwOperation only on the (rare) general/aligned
        // fallback. Reads leave the value unchanged, so old == new here.
        memw_ops.push_reg_access(
            reg_addr,
            [reg_value[0], reg_value[1]],
            [reg_value[0], reg_value[1]],
            ts,
            old_ts,
            true,
        );
        if d.rs1 == 255 {
            register_state.write_pc(op.rv1, op.timestamp);
        } else {
            register_state.write(d.rs1, op.rv1, op.timestamp);
        }
    }

    // M3: Read rs2 register at timestamp+1
    if d.read_register2 && d.rs2 != 0 {
        let reg_value = pack_register_value(op.rv2);
        let reg_addr = 2 * d.rs2 as u64;
        let (_old_val, old_ts) = register_state.read(d.rs2);
        let ts = op.timestamp + 1;
        memw_ops.push_reg_access(
            reg_addr,
            [reg_value[0], reg_value[1]],
            [reg_value[0], reg_value[1]],
            ts,
            old_ts,
            true,
        );
        register_state.write(d.rs2, op.rv2, op.timestamp + 1);
    }

    // M5: Write rd register at timestamp+2
    if d.write_register && d.rd != 0 {
        let reg_value = pack_register_value(op.rvd);
        let reg_addr = 2 * d.rd as u64;
        let (old_val, old_ts) = register_state.read(d.rd);
        let old_value = pack_register_value(old_val);
        let ts = op.timestamp + 2;
        memw_ops.push_reg_access(
            reg_addr,
            [reg_value[0], reg_value[1]],
            [old_value[0], old_value[1]],
            ts,
            old_ts,
            false,
        );
        register_state.write(d.rd, op.rvd, op.timestamp + 2);
    }

    // PC register state update (needed for M1 reads when rs1=255, i.e. AUIPC/JAL).
    // The actual PC read/write is now inline in the CPU via memory bus interactions.
    register_state.write_pc(op.next_pc, op.timestamp + 1);
}

// =============================================================================
// CPU32 (word `*W` instruction) op-generation
// =============================================================================

/// The raw ALU result `res` for a CPU32 row, matching what the dispatched chip
/// (or the ADD/SUB fast-path) computes from the sign-extended `arg1`/`arg2`.
fn cpu32_res(c: &cpu32::Cpu32Operation, arg1: u64, arg2: u64) -> u64 {
    use crate::tables::types::alu_op;
    if c.add {
        return arg1.wrapping_add(arg2);
    }
    if c.sub {
        return arg1.wrapping_sub(arg2);
    }
    if !c.alu {
        return 0;
    }
    let op = c.alu_flags & 0x1F;
    let signed = (c.alu_flags >> 5) & 1 == 1;
    let s2_or_inv = (c.alu_flags >> 6) & 1 == 1;
    let muldiv = (c.alu_flags >> 7) & 1 == 1;
    if op == alu_op::SHIFT || op == alu_op::SHIFTW {
        // The ALU bus carries the chip's raw OUT (not the sign-extended value);
        // CPU32 sign-extends it to rvd.
        ShiftOperation::new(arg1, arg2, s2_or_inv, signed, true).compute_out()
    } else if op == alu_op::MUL {
        MulOperation::new(arg1, signed, arg2, s2_or_inv)
            .compute_product()
            .0
    } else if op == alu_op::DIVREM {
        let d = DvrmOperation::new(arg1, arg2, signed);
        if muldiv {
            d.compute_remainder()
        } else {
            d.compute_quotient()
        }
    } else {
        0
    }
}

/// Builds the CPU32 row for a word (`*W`) instruction. `op.rv1/rv2/rvd` carry the
/// real register values (the main CPU delegate row zeroes its own columns).
fn build_cpu32_op(op: &CpuOperation) -> cpu32::Cpu32Operation {
    let f = &op.decode.fields;
    let mut c = cpu32::Cpu32Operation {
        timestamp: op.timestamp,
        pc: op.decode.pc,
        rs1: f.rs1,
        read_register1: f.read_register1,
        rv1: op.rv1,
        rs2: f.rs2,
        read_register2: f.read_register2,
        rv2: op.rv2,
        imm: op.decode.imm,
        res: 0,
        rd: f.rd,
        write_register: f.write_register,
        alu: f.alu,
        alu_flags: f.alu_flags,
        add: f.add,
        sub: f.sub,
        half_instruction_length: f.half_instruction_length,
    };
    let aux = c.compute_aux();
    c.res = cpu32_res(&c, aux.arg1, aux.arg2);
    c
}

/// The BITWISE-table lookups a CPU32 row sends: 5×ARE_BYTES (byte fields),
/// 8×IS_HALF (rv1/rv2 low-word halves + the 4 res halves), 1×BYTE_ALU (extracts
/// the signed bit from `alu_flags`), and the MSB16 sign bits: `res` always, plus
/// `rv1`/`rv2` only when `signed` (their MSB16 is gated by the `signed` column).
fn collect_cpu32_bitwise(c: &cpu32::Cpu32Operation) -> Vec<BitwiseOperation> {
    let mut ops = Vec::with_capacity(17);
    let half = |v: u64, sh: u32| ((v >> sh) & 0xFFFF) as u16;
    let push_half = |ops: &mut Vec<BitwiseOperation>, kind, h: u16| {
        ops.push(BitwiseOperation::halfword(
            kind,
            (h & 0xFF) as u8,
            (h >> 8) as u8,
        ));
    };

    for b in [c.half_instruction_length, c.alu_flags, c.rs1, c.rs2, c.rd] {
        ops.push(BitwiseOperation::single_byte(
            BitwiseOperationType::AreBytes,
            b,
        ));
    }
    // IS_HALF: rv1[0],rv1[1],rv2[0],rv2[1],res[0..3]
    let rv1_h0 = half(c.rv1, 0);
    let rv1_h1 = half(c.rv1, 16);
    let rv2_h0 = half(c.rv2, 0);
    let rv2_h1 = half(c.rv2, 16);
    for h in [rv1_h0, rv1_h1, rv2_h0, rv2_h1] {
        push_half(&mut ops, BitwiseOperationType::IsHalf, h);
    }
    for i in 0..4 {
        push_half(&mut ops, BitwiseOperationType::IsHalf, half(c.res, i * 16));
    }
    // BYTE_ALU[AND, X=32, Y=alu_flags] -> 32*signed (extract signed bit).
    ops.push(BitwiseOperation::byte_op(
        BitwiseOperationType::ByteAluAnd,
        32,
        c.alu_flags,
    ));
    // MSB16 on the high half of each low word. `rv1`/`rv2` are gated by `signed`
    // (the SIGN template's `signed` multiplicity — no lookup when zero-extending);
    // `res` is always sent (μ), since the `*W` result is always sign-extended.
    if c.signed() {
        push_half(&mut ops, BitwiseOperationType::Msb16, rv1_h1);
        push_half(&mut ops, BitwiseOperationType::Msb16, rv2_h1);
    }
    push_half(&mut ops, BitwiseOperationType::Msb16, half(c.res, 16));
    ops
}

/// The ALU-chip op a word ALU instruction dispatches (SHIFT/MUL/DVRM). ADDW/SUBW
/// are the CPU32 ADD/SUB fast-path (no external chip), returning `None`.
#[allow(clippy::type_complexity)]
fn cpu32_chip_op(
    c: &cpu32::Cpu32Operation,
    shift_ops: &mut Vec<ShiftOperation>,
    mul_ops: &mut Vec<(MulOperation, bool)>,
    dvrm_ops: &mut Vec<(DvrmOperation, bool)>,
) {
    use crate::tables::types::alu_op;
    if c.add || c.sub || !c.alu {
        return;
    }
    let aux = c.compute_aux();
    let op = c.alu_flags & 0x1F;
    let signed = aux.signed;
    let s2_or_inv = (c.alu_flags >> 6) & 1 == 1;
    let muldiv = (c.alu_flags >> 7) & 1 == 1;
    if op == alu_op::SHIFT || op == alu_op::SHIFTW {
        shift_ops.push(ShiftOperation::new(
            aux.arg1, aux.arg2, s2_or_inv, signed, true,
        ));
    } else if op == alu_op::MUL {
        mul_ops.push((
            MulOperation::new(aux.arg1, signed, aux.arg2, s2_or_inv),
            muldiv,
        ));
    } else if op == alu_op::DIVREM {
        dvrm_ops.push((DvrmOperation::new(aux.arg1, aux.arg2, signed), muldiv));
    }
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
/// - Read+write x254 at ts: updates the global commit index
/// - Read bytes at ts: reads committed bytes from memory
///
/// Note: x17 (syscall number) is read by CPU's M1 interaction (read_register1=true, rs1=17).
///
/// Returns: Vec of MEMW operations
fn collect_commit_memw_ops(
    op: &CpuOperation,
    register_state: &mut RegisterState,
    memory_state: &mut MemoryState,
) -> Vec<MemwOperation> {
    let ts = op.timestamp;
    let buf_addr = op.commit_buf_addr;
    let count = op.commit_count;

    let mut memw_ops = Vec::with_capacity(5 + count as usize);

    // Combined read+write x10 at ts: old=fd=1, new=count
    // This atomically asserts x10 held fd=1 and writes count as return value.
    // Uses is_read=true so MEMW activates the CO24 receiver (24 elements with old[]),
    // matching the COMMIT chip's CO24 bus send format.
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
        let memw_op = MemwOperation::new(true, reg_addr, new_value, ts, 2, true)
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

    // Read+write x254 (global commit index) at ts
    {
        let (old_index, old_ts) = register_state.read_index();
        let new_index = old_index
            .checked_add(u32::try_from(count).expect("commit_count exceeds u32 range"))
            .expect("commit index exceeds u32 range");
        let old_value = [old_index, 0, 0, 0, 0, 0, 0, 0];
        let new_value = [new_index, 0, 0, 0, 0, 0, 0, 0];
        let old_timestamps = [old_ts, 0, 0, 0, 0, 0, 0, 0];
        let memw_op = MemwOperation::new(
            true,
            register::register_base_address(254),
            new_value,
            ts,
            1,
            true,
        )
        .with_old(old_value, old_timestamps);
        memw_ops.push(memw_op);
        register_state.write_index(new_index, ts);
    }

    // Memory byte reads at ts
    for i in 0..count {
        let addr = buf_addr.wrapping_add(i);
        let (byte_val, old_ts) = memory_state.read_byte(addr);
        let value = [byte_val as u32, 0, 0, 0, 0, 0, 0, 0];
        let old_timestamps = [old_ts, 0, 0, 0, 0, 0, 0, 0];
        let memw_op =
            MemwOperation::new(false, addr, value, ts, 1, true).with_old(value, old_timestamps);
        memw_ops.push(memw_op);
        memory_state.write_byte(addr, byte_val, ts);
    }

    memw_ops
}

/// Collects HALT finalization MEMW operations for all 33 registers.
///
/// Per spec (halt.toml): at timestamp 2^64-1, HALT finalizes the GP registers:
/// - x1-x9, x11-x31: write 0 (zeroize)
/// - x10: read (verify exit code = 0; if x10 ≠ 0, proof fails via bus mismatch)
///
/// The PC (x255) is NOT finalized here — it is handled on the inline-PC `memory`
/// bus by the HALT chip's consume_pc/emit_pc plus the CPU padding chain (its
/// REGISTER final token is set separately by the caller, at the last padding
/// timestamp). Also updates `register_state` so `to_final_state_map()` reflects
/// the finalized GP register values.
fn collect_halt_ops(register_state: &mut RegisterState) -> Vec<MemwOperation> {
    let mut ops = Vec::with_capacity(32);
    let ts = u64::MAX;

    // x1-x9: write 0
    for i in 1..=9u8 {
        let (old_val, old_ts) = register_state.read(i);
        let old_value = pack_register_value(old_val);
        let old_timestamps = [old_ts, old_ts, 0, 0, 0, 0, 0, 0];
        let memw_op = MemwOperation::new(true, 2 * i as u64, [0; 8], ts, 2, false)
            .with_old(old_value, old_timestamps);
        ops.push(memw_op);
        register_state.write(i, 0, ts);
    }

    // x10: read with old=0 at ts=2^64-1 (enforce exit_code=0)
    // Per spec halt:c:read_zero_exit_code: old=0 enforces x10 was 0 at halt.
    // Non-zero exit code → bus imbalance → proof failure.
    {
        let (old_val, old_ts) = register_state.read(10);
        let old_value = pack_register_value(old_val);
        let old_timestamps = [old_ts, old_ts, 0, 0, 0, 0, 0, 0];
        let memw_op =
            MemwOperation::new(true, 20, [0; 8], ts, 2, true).with_old(old_value, old_timestamps);
        ops.push(memw_op);
        register_state.write(10, 0, ts);
    }

    // x11-x31: write 0
    for i in 11..=31u8 {
        let (old_val, old_ts) = register_state.read(i);
        let old_value = pack_register_value(old_val);
        let old_timestamps = [old_ts, old_ts, 0, 0, 0, 0, 0, 0];
        let memw_op = MemwOperation::new(true, 2 * i as u64, [0; 8], ts, 2, false)
            .with_old(old_value, old_timestamps);
        ops.push(memw_op);
        register_state.write(i, 0, ts);
    }

    // x255 (PC) is finalized via the inline-PC `memory` bus + REGISTER table, not
    // via a MEMW write at 2^64-1. See `collect_halt_ops` doc and the PC finalization
    // in the caller.

    ops
}

// =============================================================================
// Phase 3: MEMW → LT
// =============================================================================

/// Collects LT operations from MEMW for timestamp ordering.
/// Collect MEMW operations for a KeccakPermute ECALL.
///
/// Generates 25 read operations (input lanes at timestamp) and 25 write
/// operations (output lanes at timestamp+1). Each operation is 8 bytes wide.
fn collect_keccak_memw_ops(
    op: &CpuOperation,
    input: &[u64; 25],
    output: &[u64; 25],
    memory_state: &mut MemoryState,
    register_state: &mut RegisterState,
) -> Vec<MemwOperation> {
    let ts = op.timestamp;
    let state_addr = op.keccak_state_addr;
    let mut memw_ops = Vec::with_capacity(26); // 1 register read + 25 lane ops

    // Per spec (keccak:c:read_addr): read register x10 to get state_addr
    {
        let reg_value = pack_register_value(state_addr);
        let reg_addr = 2 * 10u64; // x10 → address 20
        let (_old_val, old_ts) = register_state.read(10);
        let old_timestamps = [old_ts, old_ts, 0, 0, 0, 0, 0, 0];
        let memw_op = MemwOperation::new(true, reg_addr, reg_value, ts, 2, true)
            .with_old(reg_value, old_timestamps);
        memw_ops.push(memw_op);
        register_state.write(10, state_addr, ts);
    }

    // Per spec (keccak:c:load_store_state): single combined read+write MEMW per lane.
    // input = [0, state_ptr, output_state, timestamp, 0, 0, 1], output = input_state
    // The MEMW table sees: old=input_state, value=output_state, is_read=true.
    for (lane_idx, (&in_lane, &out_lane)) in input.iter().zip(output.iter()).enumerate() {
        let lane_addr = state_addr
            .checked_add(lane_idx as u64 * 8)
            .expect("keccak state address range must be validated by the executor");

        let mut old_bytes = [0u32; 8];
        let mut old_timestamps = [0u64; 8];
        for b in 0..8 {
            old_bytes[b] = ((in_lane >> (b * 8)) & 0xFF) as u32;
            let byte_addr = lane_addr
                .checked_add(b as u64)
                .expect("keccak state address range must be validated by the executor");
            let (_old_val, old_ts) = memory_state.read_byte(byte_addr);
            old_timestamps[b] = old_ts;
        }

        let mut value_bytes = [0u32; 8];
        for (b, byte) in value_bytes.iter_mut().enumerate() {
            *byte = ((out_lane >> (b * 8)) & 0xFF) as u32;
        }

        let memw_op = MemwOperation::new(false, lane_addr, value_bytes, ts, 8, true)
            .with_old(old_bytes, old_timestamps);
        memw_ops.push(memw_op);

        // Update memory state
        for (b, &val) in value_bytes.iter().enumerate() {
            let byte_addr = lane_addr
                .checked_add(b as u64)
                .expect("keccak state address range must be validated by the executor");
            memory_state.write_byte(byte_addr, val as u8, ts);
        }
    }

    memw_ops
}

///
/// From spec memw.md:
/// - MEMW-C4 through MEMW-C7: old_timestamp[i] < timestamp (based on width)
///
/// Returns: Vec of LT operations
fn collect_lt_from_memw(memw_ops: &[MemwOperation]) -> Vec<LtOperation> {
    let mut lt_ops = Vec::with_capacity(memw_ops.len() * 8);

    for memw_op in memw_ops {
        // MEMW-C4: old_timestamp[0] < timestamp (all accesses)
        lt_ops.push(LtOperation::new(
            memw_op.old_timestamp[0],
            memw_op.timestamp,
            false,
        ));

        // MEMW-C5: old_timestamp[1] < timestamp (width >= 2)
        if memw_op.width >= 2 {
            lt_ops.push(LtOperation::new(
                memw_op.old_timestamp[1],
                memw_op.timestamp,
                false,
            ));
        }

        // MEMW-C6: old_timestamp[2,3] < timestamp (width >= 4)
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

        // MEMW-C7: old_timestamp[4..7] < timestamp (width == 8)
        if memw_op.width == 8 {
            for i in 4..8 {
                lt_ops.push(LtOperation::new(
                    memw_op.old_timestamp[i],
                    memw_op.timestamp,
                    false,
                ));
            }
        }
    }

    lt_ops
}

/// Collects LT operations from MEMW_A for timestamp ordering.
///
/// Each aligned operation has a single old_timestamp < timestamp check.
fn collect_lt_from_memw_aligned(memw_aligned_ops: &[MemwOperation]) -> Vec<LtOperation> {
    // Address overflow LT checks (R1-R3 in MEMW) are intentionally absent.
    // Alignment guarantees addr + (width-1) never wraps: the largest width-N
    // aligned address is 2^64-N, and 2^64-N+(N-1) = 2^64-1, so no u64 overflow.
    memw_aligned_ops
        .iter()
        .map(|op| LtOperation::new(op.old_timestamp[0], op.timestamp, false))
        .collect()
}

/// Checks whether a MEMW operation qualifies for the aligned fast path (MEMW_A).
///
/// An operation is aligned if:
/// 1. For width > 1: base_address is aligned to width (low bits are zero)
/// 2. All accessed bytes share the same old_timestamp
fn is_aligned_op(op: &MemwOperation) -> bool {
    let low = (op.base_address & 0xFFFF_FFFF) as u32;
    let width = op.width as u32;

    // Check alignment (trivially true for width=1)
    if width > 1 && (low & (width - 1)) != 0 {
        return false;
    }

    // Check uniform old_timestamp
    for i in 1..op.width as usize {
        if op.old_timestamp[i] != op.old_timestamp[0] {
            return false;
        }
    }

    true
}

/// Collects bitwise lookups from MEMW_A operations.
///
/// Per operation:
/// - 1 IS_HALF for alignment check: IS_HALF[base_address[0] + mask]
///
/// IS_HALF[base_address[i]] for i ∈ [0, 1] and IS_WORD[base_address[2]] are
/// assumptions — the caller's (CPU's) responsibility.
fn collect_bitwise_from_memw_aligned(ops: &[MemwOperation]) -> Vec<BitwiseOperation> {
    let mut bitwise_ops = Vec::with_capacity(ops.len());

    for op in ops {
        let low_half = (op.base_address & 0xFFFF) as u32;
        let mask: u32 = match op.width {
            2 => 1,
            4 => 3,
            8 => 7,
            _ => 0,
        };

        // IS_HALF[base_address[0] + mask]
        let value = low_half + mask;
        debug_assert!(
            value <= 0xFFFF,
            "misaligned: base_address[0] + mask overflows halfword"
        );
        let x = (value & 0xFF) as u8;
        let y = ((value >> 8) & 0xFF) as u8;
        bitwise_ops.push(BitwiseOperation::halfword(
            BitwiseOperationType::IsHalf,
            x,
            y,
        ));
    }

    bitwise_ops
}

// =============================================================================
// Routing predicates (MEMW_R register fast path)
// =============================================================================

/// An operation routes to MEMW_R if:
/// 1. It's a 2-word register access (is_register = true, width = 2)
/// 2. Both words share the same old_timestamp (atomic register write)
/// 3. timestamp[1] == old_timestamp[1] (upper limbs match)
/// 4. timestamp[0] > old_timestamp[0] (lower limb ordering)
/// 5. timestamp[0] - old_timestamp[0] <= 0x10000 (delta fits in IS_HALF range [1, 2^16])
///
/// Width-1 register ops (e.g. COMMIT x254) stay in MEMW, which has
/// dynamic write flags. MEMW_R hardcodes write2=1.
pub(crate) fn is_register_op(op: &MemwOperation) -> bool {
    if !op.is_register || op.width != 2 {
        return false;
    }
    // Both words must share old_timestamp (atomic register write assumption)
    if op.old_timestamp[0] != op.old_timestamp[1] {
        return false;
    }
    reg_ts_delta_in_range(op.timestamp, op.old_timestamp[0])
}

/// The timestamp-delta admission test for MEMW_R (conditions 3-5 of `is_register_op`),
/// factored out so the direct fast path (`push_reg_access`) and the `MemwOperation`
/// classifier (`is_register_op`) share EXACTLY the same routing logic:
/// - `ts_hi == old_ts_hi` (upper limbs match)
/// - `ts_lo > old_ts_lo` (lower-limb ordering)
/// - `ts_lo - old_ts_lo <= 2^16` (delta fits the IS_HALF range [1, 2^16])
///
/// The fast path only calls this for width-2 register accesses whose two words share
/// `old_ts` by construction, so conditions 1-2 of `is_register_op` always hold there.
#[inline]
fn reg_ts_delta_in_range(timestamp: u64, old_ts: u64) -> bool {
    let ts_lo = timestamp & 0xFFFF_FFFF;
    let old_ts_lo = old_ts & 0xFFFF_FFFF;
    let ts_hi = timestamp >> 32;
    let old_ts_hi = old_ts >> 32;
    ts_hi == old_ts_hi && ts_lo > old_ts_lo && (ts_lo - old_ts_lo) <= 0x10000
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
/// IS_HALF lookups for lhs/rhs input and lo/hi output range checks,
/// and IS_B20 lookups for carry range checks.
///
/// IS_HALF and IS_B20 are emitted once per raw op. MSB16 is deduplicated
/// per `max_rows_mul` chunk, mirroring `chunk_and_generate` — a unique signed
/// op that spans two instances is sent twice and must be tallied twice.
///
/// Returns: Vec of bitwise lookups
pub(crate) fn collect_bitwise_from_mul(
    mul_ops: &[(MulOperation, bool)],
    max_rows_mul: usize,
) -> Vec<BitwiseOperation> {
    let mut bitwise_ops = Vec::with_capacity(mul_ops.len() * 20);

    // IS_HALF and IS_B20: one set per raw op (multiplicity Sum(MU_LO, MU_HI))
    for (op, _wants_hi) in mul_ops {
        let (lo, hi) = op.compute_product();

        // IS_HALF for lhs/rhs INPUT halfwords (matches the lhs/rhs IS_HALF senders
        // in mul::bus_interactions).
        for word in [op.lhs, op.rhs] {
            for shift in [0, 16, 32, 48] {
                let half = ((word >> shift) & 0xFFFF) as u16;
                bitwise_ops.push(BitwiseOperation::halfword(
                    BitwiseOperationType::IsHalf,
                    (half & 0xFF) as u8,
                    (half >> 8) as u8,
                ));
            }
        }

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

    // MSB16: dedup per chunk — the MUL AIR sends Msb16 once per unique signed row
    // per instance, so the collector must mirror the same chunk boundary.
    for chunk in mul_ops.chunks(max_rows_mul) {
        let mut msb16_seen = std::collections::HashSet::new();
        for (op, _wants_hi) in chunk {
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
    }

    bitwise_ops
}

/// Collects bitwise lookups from DVRM operations.
///
/// Generates: IS_HALF (×20: n, d, r, n_sub_r, q) and ZERO (×2) per raw op, plus
/// MSB16 (up to ×3) and NEG ZERO (up to ×4) per unique signed op per chunk.
///
/// DVRM-A1 (IS_HALF[n]) and DVRM-A2 (IS_HALF[d]) are range-checked by the DVRM
/// table itself (n/d IS_HALF senders in dvrm::bus_interactions), so their lookups
/// are collected here alongside the constraint-level ones.
///
/// IS_HALF and ZERO (C8/C20) are emitted once per raw op. MSB16 and the
/// NEG-template ZERO lookups (C3/C5) are deduplicated per `max_rows_dvrm`
/// chunk, mirroring `chunk_and_generate`.
///
/// Returns: Vec of bitwise lookups
pub(crate) fn collect_bitwise_from_dvrm(
    dvrm_ops: &[(DvrmOperation, bool)],
    max_rows_dvrm: usize,
) -> Vec<BitwiseOperation> {
    let mut bitwise_ops = Vec::with_capacity(dvrm_ops.len() * 24);

    for (op, _wants_remainder) in dvrm_ops {
        // IS_HALF for n[0..4] and d[0..4] (DVRM-A1/A2): range-check the input
        // half-limbs so a prover cannot supply non-canonical halves (matches the
        // n/d IS_HALF senders in dvrm::bus_interactions).
        for word in [op.n, op.d] {
            for shift in [0, 16, 32, 48] {
                let half = ((word >> shift) & 0xFFFF) as u16;
                bitwise_ops.push(BitwiseOperation::halfword(
                    BitwiseOperationType::IsHalf,
                    (half & 0xFF) as u8,
                    (half >> 8) as u8,
                ));
            }
        }

        // IS_HALF for r[0..4] (DVRM-C13)
        let r = op.compute_remainder();
        for shift in [0, 16, 32, 48] {
            let half = ((r >> shift) & 0xFFFF) as u16;
            bitwise_ops.push(BitwiseOperation::halfword(
                BitwiseOperationType::IsHalf,
                (half & 0xFF) as u8,
                (half >> 8) as u8,
            ));
        }

        // IS_HALF for n_sub_r[0..4] (DVRM-C14)
        let n_sub_r = op.n.wrapping_sub(r);
        for shift in [0, 16, 32, 48] {
            let half = ((n_sub_r >> shift) & 0xFFFF) as u16;
            bitwise_ops.push(BitwiseOperation::halfword(
                BitwiseOperationType::IsHalf,
                (half & 0xFF) as u8,
                (half >> 8) as u8,
            ));
        }

        // IS_HALF for q[0..4] (DVRM-C11)
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

    // MSB16: same per-chunk dedup as MUL (Column(SIGNED) is a bit, not a count).
    for chunk in dvrm_ops.chunks(max_rows_dvrm) {
        let mut msb16_seen = std::collections::HashSet::new();
        for (op, _wants_remainder) in chunk {
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
    }

    // ZERO (NEG template): same — SIGN_R/SIGN_D are bits, dedup per chunk.
    for chunk in dvrm_ops.chunks(max_rows_dvrm) {
        let mut zero_seen = std::collections::HashSet::new();
        for (op, _wants_remainder) in chunk {
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
    }

    bitwise_ops
}

/// Collects bitwise lookups from BRANCH operations.
///
/// BRANCH sends:
/// - ARE_BYTES[next_pc_low[1], 0] - range check bits 8-15
/// - BYTE_ALU[AND, unmasked_low_byte, 254, next_pc_low[0]] - LSB masking
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

        // ARE_BYTES[next_pc_low[1], 0] - range check for byte value
        bitwise_ops.push(BitwiseOperation::single_byte(
            BitwiseOperationType::AreBytes,
            next_pc_low_1,
        ));

        // BYTE_ALU[AND, unmasked_low_byte, 254] → next_pc_low[0]
        // Verifies: next_pc_low[0] = unmasked_low_byte & 0xFE
        bitwise_ops.push(BitwiseOperation::byte_op(
            BitwiseOperationType::ByteAluAnd,
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

/// Generates ARE_BYTES ops for CPU padding rows.
///
/// CPU padding rows have all byte columns = 0 (RS1=0, RS2=0, RD=0, etc.).
/// Since the CPU bus interactions use Multiplicity::One for range checks,
/// padding rows also send, so we need matching bitwise ops.
///
/// Add the BITWISE lookups every CPU padding row sends. A padding row has all
/// values zero, so per row it sends 3× ARE_BYTES[0,0] (rs1/rs2,
/// rd/instruction_length, alu_flags/mem_flags) and 4× IS_HALF[0] (the four `res`
/// halves). Every padding row sends the same lookups, so their whole
/// contribution is a per-cell count: two `bump_n`s, no per-row work.
fn add_padding_byte_checks(hist: &mut bitwise::BitwiseHistogram, num_padding_rows: usize) {
    let n = num_padding_rows as u64;
    hist.bump_n(
        BitwiseOperation::byte_op(BitwiseOperationType::AreBytes, 0, 0),
        3 * n,
    );
    hist.bump_n(
        BitwiseOperation::halfword(BitwiseOperationType::IsHalf, 0, 0),
        4 * n,
    );
}

/// Collects ARE_BYTES lookups from PAGE data (init and fini values).
///
/// Each PAGE row generates 1 batched ARE_BYTES lookup:
/// - C1+C2: ARE_BYTES[init, fini] — range-checks both bytes in one interaction
///
/// This must be called BEFORE bitwise multiplicities are updated.
///
/// Encode private input as `[len_u32_LE][data]` — the canonical wire format.
/// Must match `executor::vm::memory::Memory::store_private_inputs`.
fn private_input_bytes(private_input: &[u8]) -> Vec<u8> {
    let len_bytes = (private_input.len() as u32).to_le_bytes();
    len_bytes
        .iter()
        .chain(private_input.iter())
        .copied()
        .collect()
}

/// Build the initial-memory image (byte address -> value) from the ELF segments
/// and the private-input region. Single source of "what memory starts as", read
/// by both `MemoryState` seeding and PAGE/bitwise init.
pub(crate) fn build_initial_image(elf: &Elf, private_input: &[u8]) -> HashMap<u64, u8> {
    let mut image: HashMap<u64, u8> = HashMap::new();
    for segment in &elf.data {
        for (i, &word) in segment.values.iter().enumerate() {
            let word_addr = segment.base_addr.wrapping_add(i as u64 * 4);
            for byte_offset in 0..4u64 {
                let byte_addr = word_addr.wrapping_add(byte_offset);
                let byte_value = ((word >> (byte_offset * 8)) & 0xFF) as u8;
                image.insert(byte_addr, byte_value);
            }
        }
    }
    if !private_input.is_empty() {
        use executor::vm::memory::PRIVATE_INPUT_START_INDEX;
        for (i, &b) in private_input_bytes(private_input).iter().enumerate() {
            image.insert(PRIVATE_INPUT_START_INDEX + i as u64, b);
        }
    }
    image
}

/// Build the initial-memory image as a dense per-page store instead of a
/// per-cell `HashMap`. Used by the streaming continuation, which carries the
/// image across all epochs (so its size matters); the byte values are identical
/// to [`build_initial_image`]. Unset cells read back as 0.
pub(crate) fn build_initial_image_paged(elf: &Elf, private_input: &[u8]) -> PagedMem<u8> {
    let mut image = PagedMem::new(0u8);
    for segment in &elf.data {
        for (i, &word) in segment.values.iter().enumerate() {
            let word_addr = segment.base_addr.wrapping_add(i as u64 * 4);
            for byte_offset in 0..4u64 {
                let byte_addr = word_addr.wrapping_add(byte_offset);
                let byte_value = ((word >> (byte_offset * 8)) & 0xFF) as u8;
                image.set(byte_addr, byte_value);
            }
        }
    }
    if !private_input.is_empty() {
        use executor::vm::memory::PRIVATE_INPUT_START_INDEX;
        for (i, &b) in private_input_bytes(private_input).iter().enumerate() {
            image.set(PRIVATE_INPUT_START_INDEX + i as u64, b);
        }
    }
    image
}

/// Test helper for computing one epoch's local-to-global touched cells without
/// building every trace table.
#[cfg(test)]
pub(crate) fn epoch_touched_cells<I: ImageSource>(
    elf: &Elf,
    initial_image: &I,
    register_init: &[u32],
    logs: &[Log],
) -> Result<Vec<(u64, u64, u64)>, Error> {
    let instructions = decode::instructions_from_elf(elf)
        .map_err(|e| Error::Execution(format!("Failed to parse instructions: {e}")))?;
    let cpu_ops = collect_cpu_ops(logs, &instructions)?;

    let mut memory_state = MemoryState::from_image(initial_image);
    let mut register_state = RegisterState::from_init(register_init);
    let _ = collect_ops_from_cpu(&cpu_ops, &mut memory_state, &mut register_state);

    Ok(touched_cells_from_memory_state(&memory_state))
}

fn touched_cells_from_memory_state(memory_state: &MemoryState) -> local_to_global::EpochTouches {
    let mut touched: Vec<(u64, u64, u64)> = memory_state
        .cells
        .iter()
        .filter(|(_, cell)| cell.1 > 0)
        .map(|(addr, cell)| (addr, cell.0 as u64, cell.1))
        .collect();
    touched.sort_by_key(|&(addr, _, _)| addr);
    touched
}

/// Bucket an initial-memory image into per-page byte arrays for PAGE init columns.
pub(crate) fn build_init_page_data<I: ImageSource>(image: &I) -> HashMap<u64, Vec<u8>> {
    let page_size = page::DEFAULT_PAGE_SIZE;
    let mut init_page_data: HashMap<u64, Vec<u8>> = HashMap::new();
    for (addr, value) in image.image_iter() {
        let page_base = page::page_base_for_address(addr);
        let offset = page::offset_in_page(addr);
        let page_data = init_page_data
            .entry(page_base)
            .or_insert_with(|| vec![0u8; page_size]);
        page_data[offset] = value;
    }
    init_page_data
}

// EXPERIMENT (bench/page-drop-arebytes): no longer called — PAGE's ARE_BYTES range
// check was removed. Kept (dead) so the change is a trivial one-line revert.
#[allow(dead_code)]
fn collect_bitwise_from_page<I: ImageSource>(
    image: &I,
    memory_state: &MemoryState,
    exclude_touched: bool,
    hist: &mut bitwise::BitwiseHistogram,
) {
    use std::collections::BTreeSet;

    let page_size = page::DEFAULT_PAGE_SIZE;

    let init_page_data = build_init_page_data(image);

    // Derive ALL page bases from memory_state (includes ELF + runtime pages)
    let page_bases: BTreeSet<u64> = memory_state.cells.page_bases().collect();

    // Build final state map from memory_state, matching `generate_page_tables`:
    // when `exclude_touched`, touched cells (timestamp > 0) are dropped so PAGE
    // emits `fini == init` for them, and the ARE_BYTES multiplicities here must
    // agree (otherwise the AreBytes bus would not balance).
    let final_state: FinalStateMap = memory_state
        .cells
        .iter()
        .filter(|(_, cell)| !exclude_touched || cell.1 == 0)
        .map(|(addr, (value, timestamp))| (addr, FinalByteState { timestamp, value }))
        .collect();

    // For each page and each byte, add ARE_BYTES lookups for init and fini
    for &page_base in &page_bases {
        let init_data = init_page_data.get(&page_base);

        for offset in 0..page_size {
            let addr = page_base + offset as u64;

            // Get init value (from ELF or 0). `.get().unwrap_or(0)` to match the
            // relaxed `init_values` contract: a shorter vec reads as trailing zeros.
            let init = init_data.map_or(0u8, |data| data.get(offset).copied().unwrap_or(0));

            // Get fini value (from final_state or init if never accessed)
            let fini = final_state.get(&addr).map_or(init, |state| state.value);

            // C1+C2: ARE_BYTES[init, fini] — batched range check for both bytes.
            // Bumped straight into the histogram: this loop visits every byte of
            // every touched page, and the histogram is the only consumer.
            hist.bump(BitwiseOperation::byte_op(
                BitwiseOperationType::AreBytes,
                init,
                fini,
            ));
        }
    }
}

// =============================================================================
// COMMIT Operation Expansion
// =============================================================================

/// Expand one Commit ECALL into its per-byte COMMIT rows using the memory state
/// at the moment the ECALL executes.
fn expand_commit_operations_for_ecall(
    ecall: &CpuOperation,
    memory_state: &MemoryState,
    start_index: u64,
) -> Vec<CommitOperation> {
    let mut ops = Vec::new();

    let timestamp = ecall.timestamp;
    let buf_addr = ecall.commit_buf_addr;
    let count = ecall.commit_count;

    for i in 0..=count {
        let remaining = count - i;
        let is_end = remaining == 0;
        let value = if !is_end {
            let (byte_val, _ts) = memory_state.read_byte(buf_addr.wrapping_add(i));
            byte_val
        } else {
            0
        };
        ops.push(CommitOperation {
            timestamp,
            index: start_index.wrapping_add(i),
            address: buf_addr.wrapping_add(i),
            count: remaining,
            first: i == 0,
            end: is_end,
            value,
        });
    }

    ops
}

/// Collect bitwise lookups from COMMIT operations.
///
/// The COMMIT table sends:
/// - IsHalfword for count_decr components (4 per real row, mult = mu)
/// - IsHalfword for address_incr halfwords (4 per real row, mult = mu)
/// - Zero for end detection (1 per real row, mult = mu)
///
/// Note: AreBytes for value is intentionally omitted per spec.
fn collect_bitwise_from_commit(commit_ops: &[CommitOperation]) -> Vec<BitwiseOperation> {
    let mut lookups = Vec::new();

    for op in commit_ops {
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
// BITWISE lookup helpers
// =============================================================================

/// IS_HALF lookup for a value `v in [0, 2^16)` (split into low/high bytes).
fn is_half_op(v: u16) -> BitwiseOperation {
    BitwiseOperation::halfword(
        BitwiseOperationType::IsHalf,
        (v & 0xFF) as u8,
        (v >> 8) as u8,
    )
}

/// IS_BYTE lookup for a single byte (sent as `AreBytes[byte, 0]`).
fn is_byte_op(b: u8) -> BitwiseOperation {
    BitwiseOperation::byte_op(BitwiseOperationType::AreBytes, b, 0)
}

/// BITWISE lookups sent by the ECSM core table (range checks + the `k != 0` ZERO check),
/// so the BITWISE receiver multiplicities account for them.
#[allow(clippy::needless_range_loop)]
pub(crate) fn collect_bitwise_from_ecsm(ops: &[ecsm::EcsmOperation]) -> Vec<BitwiseOperation> {
    let mut out = Vec::new();
    for op in ops {
        let w = &op.witness;
        // IS_BYTE on x2, q0, yG, q1[0..31].
        for i in 0..32 {
            out.push(is_byte_op(w.x2[i]));
            out.push(is_byte_op(w.q0[i]));
            out.push(is_byte_op(w.y_g[i]));
            out.push(is_byte_op(w.q1[i]));
        }
        // IS_HALF on the shifted carries (i = 0..62).
        for i in 0..63 {
            out.push(is_half_op((w.c0[i] + ecsm::CARRY_OFFSET_X2) as u16));
            out.push(is_half_op((w.c1[i] + ecsm::CARRY_OFFSET_YG) as u16));
        }
        // IS_HALF on the U256HL limbs of k_sub_N and xR_sub_p.
        for i in 0..16 {
            out.push(is_half_op(
                w.k_sub_n[2 * i] as u16 + ((w.k_sub_n[2 * i + 1] as u16) << 8),
            ));
            out.push(is_half_op(
                w.x_r_sub_p[2 * i] as u16 + ((w.x_r_sub_p[2 * i + 1] as u16) << 8),
            ));
        }
        // ZERO: assert k != 0 (sum of k's bytes).
        let sum: u32 = w.k.iter().map(|&b| b as u32).sum();
        out.push(BitwiseOperation::zero(sum));
    }
    out
}

/// BITWISE lookups sent by every ECDAS row (range checks on the byte limbs + carries).
#[allow(clippy::needless_range_loop)]
pub(crate) fn collect_bitwise_from_ecdas(ops: &[ecdas::EcdasOperation]) -> Vec<BitwiseOperation> {
    let mut out = Vec::new();
    for op in ops {
        let s = &op.step;
        out.push(is_byte_op(s.round));
        for i in 0..32 {
            out.push(is_byte_op(s.lambda[i]));
            out.push(is_byte_op(s.x_r[i]));
            out.push(is_byte_op(s.y_r[i]));
        }
        for i in 0..33 {
            out.push(is_byte_op(s.q0[i]));
            out.push(is_byte_op(s.q1[i]));
            out.push(is_byte_op(s.q2[i]));
        }
        for i in 0..63 {
            out.push(is_half_op((s.c0[i] + ecdas::CARRY_OFFSET_LAMBDA) as u16));
            out.push(is_half_op((s.c1[i] + ecdas::CARRY_OFFSET_XR) as u16));
            out.push(is_half_op((s.c2[i] + ecdas::CARRY_OFFSET_YR) as u16));
        }
    }
    out
}

/// Collect BITWISE lookups generated by the keccak chips.
///
/// The keccak round chip sends BYTE_ALU, HWSL, and ARE_BYTES
/// interactions; the keccak core chip sends IS_HALF interactions.
/// All of these must be registered so the BITWISE table's multiplicities are correct.
#[allow(clippy::needless_range_loop)]
pub(crate) fn collect_bitwise_from_keccak(keccak_ops: &[KeccakOperation]) -> Vec<BitwiseOperation> {
    use executor::vm::instruction::execution::{KECCAK_RC, KECCAK_RHO};

    let mut ops = Vec::new();

    for kop in keccak_ops {
        let state_addr = kop.state_addr;

        ops.push(BitwiseOperation::byte_op(
            BitwiseOperationType::ByteAluAnd,
            (state_addr & 0xFF) as u8,
            7,
        ));

        // Range-check addr bytes (paired with the ARE_BYTES sends in
        // keccak::bus_interactions): without this the field-element value of
        // the addr_lo / addr_hi linear combinations is unconstrained per byte.
        // 4 paired ops matching the (addr[2i], addr[2i+1]) sender pairing.
        for i in 0..4 {
            let lo = ((state_addr >> (2 * i * 8)) & 0xFF) as u8;
            let hi = ((state_addr >> ((2 * i + 1) * 8)) & 0xFF) as u8;
            ops.push(BitwiseOperation::byte_op(
                BitwiseOperationType::AreBytes,
                lo,
                hi,
            ));
        }

        // IS_HALF for state_ptr halfwords (100 per call)
        for lane_idx in 0..25 {
            let ptr = state_addr
                .checked_add(lane_idx as u64 * 8)
                .expect("keccak state address range must be validated by the executor");
            for shift in [0, 16, 32, 48] {
                let half = ((ptr >> shift) & 0xFFFF) as u16;
                ops.push(BitwiseOperation::halfword(
                    BitwiseOperationType::IsHalf,
                    (half & 0xFF) as u8,
                    ((half >> 8) & 0xFF) as u8,
                ));
            }
        }

        // Replay keccak round computation to extract bitwise lookups
        let mut state = kop.input;
        for round in 0..24 {
            // --- theta: Cxz chain BYTE_ALU[XOR] (160) ---
            let mut cxz = [[[0u8; 8]; 4]; 5];
            for x in 0..5 {
                for b in 0..8 {
                    let v0 = ((state[x] >> (b * 8)) & 0xFF) as u8;
                    let v1 = ((state[x + 5] >> (b * 8)) & 0xFF) as u8;
                    cxz[x][0][b] = v0 ^ v1;
                    ops.push(BitwiseOperation::byte_op(
                        BitwiseOperationType::ByteAluXor,
                        v0,
                        v1,
                    ));
                }
                for stage in 1..4usize {
                    let y = stage + 1;
                    for b in 0..8 {
                        let prev = cxz[x][stage - 1][b];
                        let sv = ((state[x + 5 * y] >> (b * 8)) & 0xFF) as u8;
                        cxz[x][stage][b] = prev ^ sv;
                        ops.push(BitwiseOperation::byte_op(
                            BitwiseOperationType::ByteAluXor,
                            prev,
                            sv,
                        ));
                    }
                }
            }

            // theta: HWSL for rotated C (20) + ARE_BYTES on Cxz_left (20 pairs).
            // Cxz_right is range-checked via IS_BIT polynomial constraints
            // on the keccak_rnd chip, not via lookups (spec d75944ee).
            let mut rotated_c = [[0u8; 8]; 5];
            for x in 0..5 {
                let c = cxz[x][3];
                for hw in 0..4 {
                    let halfword = (c[hw * 2] as u16) | ((c[hw * 2 + 1] as u16) << 8);
                    let shifted = halfword << 1; // u16 wraps
                    ops.push(BitwiseOperation::new(
                        BitwiseOperationType::Hwsl,
                        (halfword & 0xFF) as u8,
                        ((halfword >> 8) & 0xFF) as u8,
                        1,
                    ));
                    // ARE_BYTES for cxz_left bytes: paired (low, high) of the halfword,
                    // matching `(cxz_left[x][2i], cxz_left[x][2i+1])` sender pairing.
                    ops.push(BitwiseOperation::byte_op(
                        BitwiseOperationType::AreBytes,
                        (shifted & 0xFF) as u8,
                        ((shifted >> 8) & 0xFF) as u8,
                    ));
                }
                // Reconstruct rotated_c using the bit-typed Cxz_right.
                let mut left_bytes = [0u8; 8];
                let mut right_bits = [0u8; 4];
                for hw in 0..4 {
                    let halfword = (c[hw * 2] as u16) | ((c[hw * 2 + 1] as u16) << 8);
                    let shifted = halfword << 1;
                    left_bytes[hw * 2] = (shifted & 0xFF) as u8;
                    left_bytes[hw * 2 + 1] = ((shifted >> 8) & 0xFF) as u8;
                    right_bits[hw] = (halfword >> 15) as u8;
                }
                for b in 0usize..8 {
                    let right_contribution = if b.is_multiple_of(2) {
                        right_bits[(b / 2 + 3) % 4]
                    } else {
                        0
                    };
                    rotated_c[x][b] = left_bytes[b].wrapping_add(right_contribution);
                }
            }

            // theta: Dxz BYTE_ALU[XOR] (40)
            let mut d_bytes = [[0u8; 8]; 5];
            for x in 0..5 {
                for b in 0..8 {
                    let a = cxz[(x + 4) % 5][3][b];
                    let rb = rotated_c[(x + 1) % 5][b];
                    d_bytes[x][b] = a ^ rb;
                    ops.push(BitwiseOperation::byte_op(
                        BitwiseOperationType::ByteAluXor,
                        a,
                        rb,
                    ));
                }
            }

            // theta final: BYTE_ALU[XOR] (200)
            let mut theta_lanes = [0u64; 25];
            for x in 0..5 {
                for y in 0..5 {
                    let lane = state[x + 5 * y];
                    let mut d_lane = 0u64;
                    for b in 0..8 {
                        d_lane |= (d_bytes[x][b] as u64) << (b * 8);
                    }
                    theta_lanes[x + 5 * y] = lane ^ d_lane;
                    for b in 0..8 {
                        let s = ((lane >> (b * 8)) & 0xFF) as u8;
                        ops.push(BitwiseOperation::byte_op(
                            BitwiseOperationType::ByteAluXor,
                            s,
                            d_bytes[x][b],
                        ));
                    }
                }
            }

            // rho: HWSL (100) + ARE_BYTES (200 pairs)
            for x in 0..5 {
                for y in 0..5 {
                    let rho_offset = KECCAK_RHO[x][y] as usize;
                    let rnc_val = (rho_offset % 16) as u8;
                    let theta_lane = theta_lanes[x + 5 * y];
                    for hw in 0..4 {
                        let halfword = ((theta_lane >> (hw * 16)) & 0xFFFF) as u16;
                        let (shifted, carry) = if rnc_val == 0 {
                            (halfword, 0u16)
                        } else {
                            (halfword << rnc_val, halfword >> (16 - rnc_val))
                        };
                        ops.push(BitwiseOperation::new(
                            BitwiseOperationType::Hwsl,
                            (halfword & 0xFF) as u8,
                            ((halfword >> 8) & 0xFF) as u8,
                            rnc_val,
                        ));
                        // ARE_BYTES paired as (rot_left[b], rot_right[b]) for
                        // each byte of the halfword, matching the sender pairing
                        // in keccak_rnd::bus_interactions.
                        ops.push(BitwiseOperation::byte_op(
                            BitwiseOperationType::AreBytes,
                            (shifted & 0xFF) as u8,
                            (carry & 0xFF) as u8,
                        ));
                        ops.push(BitwiseOperation::byte_op(
                            BitwiseOperationType::AreBytes,
                            ((shifted >> 8) & 0xFF) as u8,
                            ((carry >> 8) & 0xFF) as u8,
                        ));
                    }
                }
            }

            // pi: compute pi_lanes
            let mut pi_lanes = [0u64; 25];
            for x in 0..5 {
                for y in 0..5 {
                    let rotated = theta_lanes[x + 5 * y].rotate_left(KECCAK_RHO[x][y]);
                    let dst_x = y;
                    let dst_y = (2 * x + 3 * y) % 5;
                    pi_lanes[dst_x + 5 * dst_y] = rotated;
                }
            }

            // chi: BYTE_ALU[AND] (200) + BYTE_ALU[XOR] (200)
            let mut chi_lanes = [0u64; 25];
            for x in 0..5 {
                for y in 0..5 {
                    let not_next = !pi_lanes[(x + 1) % 5 + 5 * y];
                    let next2 = pi_lanes[(x + 2) % 5 + 5 * y];
                    let and_val = not_next & next2;
                    chi_lanes[x + 5 * y] = pi_lanes[x + 5 * y] ^ and_val;
                    for b in 0..8 {
                        let not_byte = ((not_next >> (b * 8)) & 0xFF) as u8;
                        let n2_byte = ((next2 >> (b * 8)) & 0xFF) as u8;
                        ops.push(BitwiseOperation::byte_op(
                            BitwiseOperationType::ByteAluAnd,
                            not_byte,
                            n2_byte,
                        ));
                        let pi_byte = ((pi_lanes[x + 5 * y] >> (b * 8)) & 0xFF) as u8;
                        let and_byte = ((and_val >> (b * 8)) & 0xFF) as u8;
                        ops.push(BitwiseOperation::byte_op(
                            BitwiseOperationType::ByteAluXor,
                            pi_byte,
                            and_byte,
                        ));
                    }
                }
            }

            // iota: BYTE_ALU[XOR] (8)
            let rc_val = KECCAK_RC[round];
            for b in 0..8 {
                let chi_byte = ((chi_lanes[0] >> (b * 8)) & 0xFF) as u8;
                let rc_byte = ((rc_val >> (b * 8)) & 0xFF) as u8;
                ops.push(BitwiseOperation::byte_op(
                    BitwiseOperationType::ByteAluXor,
                    chi_byte,
                    rc_byte,
                ));
            }

            // Update state
            chi_lanes[0] ^= rc_val;
            state = chi_lanes;
        }
    }

    ops
}

/// every address accessed during execution (ELF init + runtime stores/loads).
/// ELF pages get their init data from the binary; all others are zero-init.
fn generate_page_tables<I: ImageSource>(
    image: &I,
    memory_state: &MemoryState,
    private_input: &[u8],
    exclude_touched: bool,
) -> (
    Vec<TraceTable<GoldilocksField, GoldilocksExtension>>,
    Vec<PageConfig>,
) {
    use std::collections::BTreeSet;

    // Per-page init bytes from the initial-memory image.
    let init_page_data = build_init_page_data(image);

    // Derive ALL page bases from memory_state (includes ELF + runtime pages)
    let page_bases: BTreeSet<u64> = memory_state.cells.page_bases().collect();

    // INSTRUMENTATION (bench/page-drop-arebytes): measure how many of the full
    // 2^18-row PAGE tables monolithic builds are loaded-but-never-touched (no cell
    // accessed by a load/store, i.e. all timestamps 0). `touched` here should equal
    // the continuation's `touched_page_bases` count; `untouched` is the tables
    // monolithic commits for nothing.
    {
        let touched: BTreeSet<u64> = memory_state
            .cells
            .iter()
            .filter(|(_, cell)| cell.1 > 0)
            .map(|(addr, _)| page::page_base_for_address(addr))
            .collect();
        eprintln!(
            "[PAGE-COUNT] populated={} touched={} untouched={}",
            page_bases.len(),
            touched.len(),
            page_bases.len() - touched.len(),
        );
        // Print each untouched page's base address (hex) so it can be mapped to
        // ELF sections (`readelf -S`) — i.e. is the wasted data .text / .rodata / .data?
        for base in page_bases.difference(&touched) {
            eprintln!("[PAGE-UNTOUCHED] 0x{base:x}");
        }
    }

    // Build final state map from memory_state. When `exclude_touched` (continuation
    // epoch with L2G bookend), drop touched cells (timestamp > 0) so PAGE self-
    // cancels them (init == fini, ts == 0) and the local-to-global table owns their
    // Memory-bus init/fini instead.
    let final_state: FinalStateMap = memory_state
        .cells
        .iter()
        .filter(|(_, cell)| !exclude_touched || cell.1 == 0)
        .map(|(addr, (value, timestamp))| (addr, FinalByteState { timestamp, value }))
        .collect();

    // Generate PAGE tables and configs
    let mut pages = Vec::new();
    let mut page_configs = Vec::new();

    // Determine which page bases hold private input data — count-based, via the
    // shared helpers (single source of truth with the continuation path).
    let num_private_input_pages = page::private_input_page_count(private_input);

    for &page_base in &page_bases {
        let config = if page::is_private_input_page(page_base, num_private_input_pages) {
            let init_data = init_page_data.get(&page_base).cloned().unwrap_or_default();
            PageConfig::with_private_input(page_base, init_data)
        } else if let Some(init_data) = init_page_data.get(&page_base) {
            PageConfig::with_data(page_base, init_data.clone())
        } else {
            PageConfig::zero_init(page_base)
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

    /// SHIFT shift operation traces (split into chunks of max_rows::SHIFT)
    pub shifts: Vec<TraceTable<GoldilocksField, GoldilocksExtension>>,

    /// MEMW memory/register read/write traces (split into chunks of max_rows::MEMW)
    pub memws: Vec<TraceTable<GoldilocksField, GoldilocksExtension>>,

    /// MEMW_A aligned memory/register read/write traces (split into chunks of max_rows::MEMW_A)
    pub memw_aligneds: Vec<TraceTable<GoldilocksField, GoldilocksExtension>>,

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

    /// Committed public output bytes recovered during trace generation.
    pub public_output_bytes: Vec<u8>,

    /// BRANCH target calculation traces (wrapped in Vec for uniform architecture)
    pub branches: Vec<TraceTable<GoldilocksField, GoldilocksExtension>>,

    /// HALT single-row table for program termination
    pub halt: TraceTable<GoldilocksField, GoldilocksExtension>,

    /// COMMIT table for write syscall (byte-by-byte commit with recursive bus)
    pub commit: TraceTable<GoldilocksField, GoldilocksExtension>,

    /// KECCAK core table (one row per keccak permutation call)
    pub keccak: TraceTable<GoldilocksField, GoldilocksExtension>,

    /// KECCAK_RND round table (24 rows per keccak call)
    pub keccak_rnd: TraceTable<GoldilocksField, GoldilocksExtension>,

    /// KECCAK_RC precomputed round constant table (32 rows)
    pub keccak_rc: TraceTable<GoldilocksField, GoldilocksExtension>,

    /// ECSM core table (one row per scalar-multiplication ecall)
    pub ecsm: TraceTable<GoldilocksField, GoldilocksExtension>,

    /// EC_SCALAR table (32 rows per ecall)
    pub ec_scalar: TraceTable<GoldilocksField, GoldilocksExtension>,

    /// ECDAS double/add table (variable rows per ecall)
    pub ecdas: TraceTable<GoldilocksField, GoldilocksExtension>,

    /// MEMW_R register-only fast-path traces (split into chunks of max_rows::MEMW_R)
    pub memw_registers: Vec<TraceTable<GoldilocksField, GoldilocksExtension>>,
    /// Local-to-global boundary table for continuation epochs. Empty unless the
    /// continuation driver fills it with the boundary derived from
    /// `touched_memory_cells`.
    pub local_to_global: TraceTable<GoldilocksField, GoldilocksExtension>,
    /// Touched cells observed while replaying this epoch's logs, each as
    /// `(address, end_value, end_timestamp)`. Populated only for continuation
    /// epochs that use the L2G memory bookend.
    pub touched_memory_cells: local_to_global::EpochTouches,
    // Auxiliary ALU / memory / CPU32 dispatch chips (split into chunks of their max_rows)
    pub eqs: Vec<TraceTable<GoldilocksField, GoldilocksExtension>>,
    pub bytewises: Vec<TraceTable<GoldilocksField, GoldilocksExtension>>,
    pub stores: Vec<TraceTable<GoldilocksField, GoldilocksExtension>>,
    pub cpu32s: Vec<TraceTable<GoldilocksField, GoldilocksExtension>>,
}

/// Intermediate state from Phase 2: all ops collected from CPU, ready for
/// Phases 3-5 (LT extension, bitwise, trace generation).
struct CollectedOps {
    cpu_ops: Vec<CpuOperation>,
    memw_ops: Vec<MemwOperation>,
    memw_aligned_ops: Vec<MemwOperation>,
    /// Direct-fill MEMW_R rows (register fast path).
    memw_register_rows: Vec<RegRow>,
    load_ops: Vec<LoadOperation>,
    lt_ops: Vec<LtOperation>,
    shift_ops: Vec<ShiftOperation>,
    bitwise_ops: Vec<BitwiseOperation>,
    branch_ops: Vec<BranchOperation>,
    mul_ops: Vec<(MulOperation, bool)>,
    dvrm_ops: Vec<(DvrmOperation, bool)>,
    commit_ops: Vec<CommitOperation>,
    keccak_ops: Vec<KeccakOperation>,
    // Auxiliary ALU / memory / CPU32 dispatch chips (driven by the CPU ALU/MEMORY dispatch).
    eq_ops: Vec<eq::EqOperation>,
    bytewise_ops: Vec<bytewise::BytewiseOperation>,
    store_ops: Vec<store::StoreOperation>,
    cpu32_ops: Vec<cpu32::Cpu32Operation>,
    // EC scalar-multiplication accelerator chips.
    ecsm_ops: Vec<ecsm::EcsmOperation>,
    ec_scalar_ops: Vec<ec_scalar::EcScalarOperation>,
    ecdas_ops: Vec<ecdas::EcdasOperation>,
}

/// Chunk raw ops and generate one trace table per chunk. When `storage_mode`
/// is `Disk`, each chunk's main table is spilled to mmap before the next chunk
/// is built so peak heap usage stays bounded.
fn chunk_and_generate<T: Sync>(
    ops: &[T],
    max_rows: usize,
    generate: impl Fn(&[T]) -> TraceTable<GoldilocksField, GoldilocksExtension> + Send + Sync,
    #[cfg(feature = "disk-spill")] storage_mode: StorageMode,
) -> Result<Vec<TraceTable<GoldilocksField, GoldilocksExtension>>, Error> {
    let op_chunks: Vec<&[T]> = if ops.is_empty() {
        vec![&[][..]]
    } else {
        ops.chunks(max_rows).collect()
    };
    // Disk mode generates one chunk at a time so each spills before the next
    // allocates, keeping trace memory bounded.
    #[cfg(feature = "disk-spill")]
    if storage_mode == StorageMode::Disk {
        let mut tables = Vec::with_capacity(op_chunks.len());
        for chunk in op_chunks {
            let mut t = generate(chunk);
            t.main_table
                .spill_to_disk()
                .map_err(|e| Error::Prover(format!("disk-spill trace: {e}")))?;
            tables.push(t);
        }
        return Ok(tables);
    }
    #[cfg(feature = "parallel")]
    let tables = op_chunks.into_par_iter().map(generate).collect();
    #[cfg(not(feature = "parallel"))]
    let tables = op_chunks.into_iter().map(generate).collect();
    Ok(tables)
}

/// Phase 2: Collect and route all operations from CPU ops.
///
/// Takes the raw output of `collect_ops_from_cpu` plus `register_state`
/// (for HALT finalization), and returns fully-routed ops ready for Phase 3+.
#[allow(clippy::too_many_arguments)]
fn collect_all_ops(
    cpu_ops: Vec<CpuOperation>,
    mut memw: MemwBuckets,
    load_ops: Vec<LoadOperation>,
    mut lt_ops: Vec<LtOperation>,
    mut shift_ops: Vec<ShiftOperation>,
    mut bitwise_ops: Vec<BitwiseOperation>,
    commit_ops: Vec<CommitOperation>,
    keccak_ops: Vec<KeccakOperation>,
    cpu32_ops: Vec<cpu32::Cpu32Operation>,
    ecsm_ops: Vec<ecsm::EcsmOperation>,
    ec_scalar_ops: Vec<ec_scalar::EcScalarOperation>,
    ecdas_ops: Vec<ecdas::EcdasOperation>,
    register_state: &mut RegisterState,
    is_final: bool,
) -> CollectedOps {
    // HALT finalization: 33 register MEMW operations at timestamp u64::MAX.
    // Must come before Phase 3 (LT from MEMW) so HALT ops get timestamp checks.
    // Only the final epoch terminates; intermediate epochs keep their boundary
    // register state (no zeroizing) so it can seed the next epoch.
    if is_final {
        // Route halt ops through the same classifier; they append to the end of their
        // buckets.
        memw.extend_ops(collect_halt_ops(register_state));
    }

    // The walk (`collect_ops_from_cpu`) already routed every MemwOperation into its bucket at
    // creation via `MemwBuckets`, so there is no separate routing pass here: the ops are not
    // moved a second time. Order within each bucket is the walk's insertion order, which the
    // multiplicity counts depend on being deterministic.
    let MemwBuckets {
        register_rows: memw_register_rows,
        aligned: memw_aligned_ops,
        general: memw_ops,
    } = memw;

    // Collect BRANCH operations from CPU ops where branch_cond = true
    let branch_ops: Vec<BranchOperation> = cpu_ops
        .iter()
        .filter(|op| op.branch_cond)
        .map(|op| {
            BranchOperation::new(
                op.decode.pc,
                op.decode.imm, // offset as full 64-bit DWordWL (already sign-extended)
                op.rv1,        // register value must match the CPU's BRANCH bus signature
                op.decode.fields.jalr(),
            )
        })
        .collect();

    // Collect MUL operations from non-word MUL instructions. lhs_signed = `signed`
    // (alu_flags bit 5); rhs_signed = `signed2` (bit 6); wants_hi = `muldiv` (bit 7).
    let mut mul_ops: Vec<(MulOperation, bool)> = cpu_ops
        .iter()
        .filter(|op| !op.decode.fields.word_instr && op.decode.fields.is_mul())
        .map(|op| {
            let f = op.decode.fields;
            (
                MulOperation::new(op.rv1, f.alu_signed(), op.arg2, f.alu_signed2_or_invert()),
                f.alu_muldiv(),
            )
        })
        .collect();

    // Collect DVRM operations from non-word DIV/REM instructions.
    let mut dvrm_ops: Vec<(DvrmOperation, bool)> = cpu_ops
        .iter()
        .filter(|op| !op.decode.fields.word_instr && op.decode.fields.is_divrem())
        .map(|op| {
            let f = op.decode.fields;
            (
                DvrmOperation::new(op.rv1, op.arg2, f.alu_signed()),
                f.alu_muldiv(),
            )
        })
        .collect();

    // Collect the ALU/MEMORY chip ops (non-word rows).
    // EQ: BEQ/BNE (invert = alu_flags bit 6). BYTEWISE: AND/OR/XOR (op = alu_op).
    let eq_ops: Vec<eq::EqOperation> = cpu_ops
        .iter()
        .filter(|op| !op.decode.fields.word_instr && op.decode.fields.is_eq())
        .map(|op| eq::EqOperation::new(op.rv1, op.arg2, op.decode.fields.alu_signed2_or_invert()))
        .collect();
    let bytewise_ops: Vec<bytewise::BytewiseOperation> = cpu_ops
        .iter()
        .filter(|op| {
            let f = &op.decode.fields;
            !f.word_instr && (f.is_and() || f.is_or() || f.is_xor())
        })
        .map(|op| bytewise::BytewiseOperation::new(op.rv1, op.arg2, op.decode.fields.alu_op()))
        .collect();
    // STORE: receives MEMORY(memory_op=1) from the CPU and sends the MEMW write
    // at timestamp+1 (mirrors `collect_store_op_from_cpu`, which records the MEMW
    // table row).
    let store_ops: Vec<store::StoreOperation> = cpu_ops
        .iter()
        .filter(|op| op.decode.fields.is_store())
        .map(|op| {
            // The MEMORY bus and the STORE chip's MEMW write share the base
            // timestamp (spec store.toml uses one `timestamp` for both).
            store::StoreOperation::new(
                op.res,
                op.timestamp,
                op.rv2,
                op.decode.fields.mem_bytes() as u8,
            )
        })
        .collect();

    // CPU32 (word `*W`) dispatch: each CPU32 row that uses the full ALU sends to
    // the SHIFT/MUL/DVRM chips (ADDW/SUBW are the CPU32 ADD/SUB fast-path). These
    // word DVRM ops are added before the DVRM→LT/MUL loops so they get their own
    // internal consistency lookups. CPU32 also sends its own BITWISE range checks.
    for c in &cpu32_ops {
        cpu32_chip_op(c, &mut shift_ops, &mut mul_ops, &mut dvrm_ops);
        bitwise_ops.extend(collect_cpu32_bitwise(c));
    }

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

    CollectedOps {
        cpu_ops,
        memw_ops,
        memw_aligned_ops,
        memw_register_rows,
        load_ops,
        lt_ops,
        shift_ops,
        bitwise_ops,
        branch_ops,
        mul_ops,
        dvrm_ops,
        commit_ops,
        keccak_ops,
        eq_ops,
        bytewise_ops,
        store_ops,
        cpu32_ops,
        ecsm_ops,
        ec_scalar_ops,
        ecdas_ops,
    }
}

/// Phases 3-5: From routed ops, produce all traces and assemble `Traces`.
///
/// `initial_image` controls PAGE table generation: `Some(image)` generates real
/// PAGE tables and PAGE bitwise lookups seeded from the initial-memory image;
/// `None` produces empty page tables.
#[allow(clippy::too_many_arguments)]
fn build_traces<I: ImageSource + Sync>(
    ops: CollectedOps,
    initial_image: Option<&I>,
    memory_state: &MemoryState,
    register_init: &[u32],
    decode_trace: TraceTable<GoldilocksField, GoldilocksExtension>,
    decode_pc_to_row: decode::PcToRow,
    mut register_state: RegisterState,
    max_rows: &super::MaxRowsConfig,
    #[cfg(feature = "disk-spill")] storage_mode: StorageMode,
    private_input: &[u8],
    is_final: bool,
    l2g_memory_bookend: bool,
) -> Result<Traces, Error> {
    let CollectedOps {
        cpu_ops,
        memw_ops,
        memw_aligned_ops,
        memw_register_rows,
        load_ops,
        mut lt_ops,
        shift_ops,
        bitwise_ops,
        branch_ops,
        mul_ops,
        dvrm_ops,
        commit_ops,
        keccak_ops,
        eq_ops,
        bytewise_ops,
        store_ops,
        cpu32_ops,
        ecsm_ops,
        ec_scalar_ops,
        ecdas_ops,
    } = ops;

    // =====================================================================
    // PHASE 3: MEMW → LT (timestamp ordering and overflow checks)
    // =====================================================================
    lt_ops.extend(collect_lt_from_memw(&memw_ops));
    lt_ops.extend(collect_lt_from_memw_aligned(&memw_aligned_ops));

    // =====================================================================
    // PHASE 4: All → Bitwise lookups
    // =====================================================================
    #[cfg(feature = "instruments")]
    let __sp = stark::instruments::span("p4_bitwise_collect");

    let public_output_bytes: Vec<u8> = commit_ops
        .iter()
        .filter(|op| !op.end)
        .map(|op| op.value)
        .collect();

    // CPU padding rows send ARE_BYTES with all-zero values.
    // Add corresponding ops so the bitwise table multiplicities balance.
    let num_padding_rows: usize = cpu_ops
        .chunks(max_rows.cpu)
        .map(|chunk| chunk.len().next_power_of_two().max(4) - chunk.len())
        .sum();

    // The per-source bitwise collectors are all pure functions of their inputs, and the
    // BITWISE multiplicities are order-independent (they ride a permutation-invariant bus),
    // so every source can be collected in parallel and the per-worker histograms summed in
    // any order.
    //
    // MUL/DVRM dedup their per-unique bit-gated lookups PER CHIP INSTANCE, so pass the same
    // chunk size used to split them into instances so multiplicities match the per-instance
    // sends. MEMW_R sends IS_HALFWORD[timestamp_0 - old_timestamp_lo - 1]. PAGE does a
    // batched ARE_BYTES[init, fini] per row (skipped in continuation epochs, which the L2G
    // table owns). COMMIT sends AreBytes+IsHalfword; KECCAK_RND sends XOR/AND/ARE_BYTES/HWSL.
    // We never concatenate the lookups into one giant `Vec<BitwiseOperation>` (~140 M ops /
    // ~560 MB at 10-tx whose only consumer is the multiplicity count). Each collector bumps
    // the `BitwiseHistogram` it is handed: the heavy sources (MEMW_R one-per-row, PAGE
    // one-per-byte, padding) count directly with no per-source Vec at all, and the small
    // sources fold their transient `collect_*` Vec in and drop it. The histogram is a
    // commutative monoid, so per-worker histograms tree-reduce to multiplicities that are
    // independent of accumulation order.
    type Collector<'a> = Box<dyn Fn(&mut bitwise::BitwiseHistogram) + Sync + 'a>;
    let mul_chunk = max_rows.mul;
    let dvrm_chunk = max_rows.dvrm;
    // Every source except the two dominant ones (the in-walk lookups and MEMW_R, which are
    // split into row-ranges in the parallel path below) stays a single whole-source collector.
    let collectors: Vec<Collector> = vec![
        Box::new(|h| h.add_ops(&collect_bitwise_from_lt(&lt_ops))),
        Box::new(|h| h.add_ops(&collect_bitwise_from_mul(&mul_ops, mul_chunk))),
        Box::new(|h| h.add_ops(&collect_bitwise_from_dvrm(&dvrm_ops, dvrm_chunk))),
        Box::new(|h| h.add_ops(&collect_bitwise_from_branch(&branch_ops))),
        Box::new(|h| h.add_ops(&shift::collect_bitwise_from_shift(&shift_ops))),
        Box::new(|h| {
            for op in &bytewise_ops {
                h.add_ops(&op.collect_bitwise_ops());
            }
        }),
        Box::new(|h| {
            for op in &eq_ops {
                h.add_ops(&op.collect_bitwise_ops());
            }
        }),
        Box::new(|h| {
            for op in &store_ops {
                h.add_ops(&op.collect_bitwise_ops());
            }
        }),
        Box::new(|h| h.add_ops(&collect_bitwise_from_memw_aligned(&memw_aligned_ops))),
        Box::new(|h| h.add_ops(&collect_bitwise_from_commit(&commit_ops))),
        Box::new(|h| h.add_ops(&collect_bitwise_from_keccak(&keccak_ops))),
        Box::new(|h| h.add_ops(&collect_bitwise_from_ecsm(&ecsm_ops))),
        Box::new(|h| h.add_ops(&collect_bitwise_from_ecdas(&ecdas_ops))),
        Box::new(|h| add_padding_byte_checks(h, num_padding_rows)),
    ];
    // EXPERIMENT (bench/page-drop-arebytes): PAGE's per-row ARE_BYTES[init, fini] range
    // check is removed on this branch (see `page::bus_interactions`), so its multiplicity
    // must NOT be registered here or the AreBytes bus would over-count on the receiver
    // side and fail to balance. The `collect_bitwise_from_page` collector is therefore
    // no longer pushed. (`initial_image`/`l2g_memory_bookend` remain used elsewhere.)

    let mut base = bitwise::BitwiseHistogram::new();

    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        // Cap concurrent 80 MiB histograms at `cap` to bound peak memory. The two dominant
        // sources — the in-walk lookups and MEMW_R (each tens of millions of items) — are
        // split into ~`cap` row-range slices so they parallelize INTERNALLY instead of each
        // pinning one core while the rest idle. Every unit (whole collectors + the heavy
        // slices) is round-robined into exactly `cap` buckets, one histogram each, so the
        // split heavy work is spread across buckets rather than piled into one.
        // add_ops/bump/merge form a commutative monoid, so any partition yields
        // byte-identical multiplicities (same as the serial fallback below).
        let cap = rayon::current_num_threads().clamp(1, 8);
        let mut units: Vec<Collector> = Vec::with_capacity(collectors.len() + 2 * cap);
        let iw_chunk = bitwise_ops.len().div_ceil(cap).max(1);
        for slice in bitwise_ops.chunks(iw_chunk) {
            units.push(Box::new(move |h| h.add_ops(slice)));
        }
        let reg_chunk = memw_register_rows.len().div_ceil(cap).max(1);
        for slice in memw_register_rows.chunks(reg_chunk) {
            units.push(Box::new(move |h| {
                memw_register::collect_bitwise_from_memw_register(slice, h)
            }));
        }
        units.extend(collectors);

        let mut buckets: Vec<Vec<Collector>> = (0..cap).map(|_| Vec::new()).collect();
        for (i, unit) in units.into_iter().enumerate() {
            buckets[i % cap].push(unit);
        }
        if let Some(reduced) = buckets
            .par_iter()
            .map(|bucket| {
                let mut h = bitwise::BitwiseHistogram::new();
                for f in bucket {
                    f(&mut h);
                }
                h
            })
            .reduce_with(|mut a, b| {
                a.merge(&b);
                a
            })
        {
            base.merge(&reduced);
        }
    }
    #[cfg(not(feature = "parallel"))]
    {
        base.add_ops(&bitwise_ops);
        memw_register::collect_bitwise_from_memw_register(&memw_register_rows, &mut base);
        for f in &collectors {
            f(&mut base);
        }
    }
    let bitwise_histogram = base;
    // The in-walk lookup Vec has been counted into the histogram; free it now.
    drop(bitwise_ops);
    #[cfg(feature = "instruments")]
    drop(__sp);

    // =====================================================================
    // PHASE 5: Generate final traces (parallelized)
    // =====================================================================
    #[cfg(feature = "instruments")]
    let __sp = stark::instruments::span("p5_generate_tables");

    // A monolithic run or the final continuation epoch terminates on the program's
    // halt ECALL. Intermediate continuation epochs do not halt, so fall back to the
    // last cycle's timestamp and skip HALT-based PC finalization — the PC is carried
    // to the next epoch via the register snapshot, and HALT is excluded from the
    // proof (`include_halt = false`) so `halt_trace` is unused there.
    let (halt_timestamp, halt_next_pc) = if is_final {
        let halt_op = cpu_ops
            .iter()
            .rev()
            .find(|op| op.decode.fields.ecall)
            .ok_or(Error::MissingHaltEcall)?;
        // Finalize the PC (x255) on the REGISTER table. The CPU padding rows carry
        // pc=1 and chain the inline-PC `memory` tokens with a +4 timestamp cadence
        // starting from the HALT chip's emit_pc at `halt_timestamp + 1`; the last
        // padding write therefore lands at `halt_timestamp + 4*num_padding_rows + 1`
        // (= `halt_timestamp + 1` when there is no padding). The REGISTER final token
        // must match that last write to balance the memory argument.
        register_state.write_pc(1, halt_op.timestamp + 4 * num_padding_rows as u64 + 1);
        (halt_op.timestamp, halt_op.next_pc)
    } else {
        (cpu_ops.last().map(|op| op.timestamp).unwrap_or(0), 0)
    };

    let register_final_state = register_state.to_final_state_map();

    // Each build below reads disjoint op lists and writes its own table, so
    // they all run in one rayon scope. Disk-spill stays sequential: its
    // generate→spill order keeps trace memory bounded.
    let cpu_ops_ref = &cpu_ops;
    let gen_cpus = || {
        chunk_and_generate(
            cpu_ops_ref,
            max_rows.cpu,
            cpu::generate_cpu_trace,
            #[cfg(feature = "disk-spill")]
            storage_mode,
        )
    };
    let gen_memws = || {
        chunk_and_generate(
            &memw_ops,
            max_rows.memw,
            memw::generate_memw_trace,
            #[cfg(feature = "disk-spill")]
            storage_mode,
        )
    };
    let gen_memw_aligneds = || {
        chunk_and_generate(
            &memw_aligned_ops,
            max_rows.memw_aligned,
            memw_aligned::generate_memw_aligned_trace,
            #[cfg(feature = "disk-spill")]
            storage_mode,
        )
    };
    let gen_memw_registers = || {
        // Direct-to-column fill from compact RegRows — the register fast path never
        // materializes a `Vec<MemwOperation>`.
        chunk_and_generate(
            &memw_register_rows,
            max_rows.memw_register,
            memw_register::generate_memw_register_trace_from_rows,
            #[cfg(feature = "disk-spill")]
            storage_mode,
        )
    };
    let gen_loads = || {
        chunk_and_generate(
            &load_ops,
            max_rows.load,
            load::generate_load_trace,
            #[cfg(feature = "disk-spill")]
            storage_mode,
        )
    };
    let gen_lts = || {
        chunk_and_generate(
            &lt_ops,
            max_rows.lt,
            lt::generate_lt_trace,
            #[cfg(feature = "disk-spill")]
            storage_mode,
        )
    };
    let gen_shifts = || {
        chunk_and_generate(
            &shift_ops,
            max_rows.shift,
            shift::generate_shift_trace,
            #[cfg(feature = "disk-spill")]
            storage_mode,
        )
    };
    let gen_muls = || {
        chunk_and_generate(
            &mul_ops,
            max_rows.mul,
            mul::generate_mul_trace,
            #[cfg(feature = "disk-spill")]
            storage_mode,
        )
    };
    let gen_dvrms = || {
        chunk_and_generate(
            &dvrm_ops,
            max_rows.dvrm,
            dvrm::generate_dvrm_trace,
            #[cfg(feature = "disk-spill")]
            storage_mode,
        )
    };
    let gen_branches = || {
        chunk_and_generate(
            &branch_ops,
            max_rows.branch,
            branch::generate_branch_trace,
            #[cfg(feature = "disk-spill")]
            storage_mode,
        )
    };
    // Auxiliary ALU / memory / CPU32 dispatch chips. Not yet driven by the CPU
    // dispatch, so they are generated empty — one padded (μ=0) chunk each, which
    // contributes nothing to any bus.
    let gen_eqs = || {
        chunk_and_generate::<eq::EqOperation>(
            &eq_ops,
            max_rows.eq,
            eq::generate_eq_trace,
            #[cfg(feature = "disk-spill")]
            storage_mode,
        )
    };
    let gen_bytewises = || {
        chunk_and_generate::<bytewise::BytewiseOperation>(
            &bytewise_ops,
            max_rows.bytewise,
            bytewise::generate_bytewise_trace,
            #[cfg(feature = "disk-spill")]
            storage_mode,
        )
    };
    let gen_stores = || {
        chunk_and_generate::<store::StoreOperation>(
            &store_ops,
            max_rows.store,
            store::generate_store_trace,
            #[cfg(feature = "disk-spill")]
            storage_mode,
        )
    };
    let gen_cpu32s = || {
        chunk_and_generate::<cpu32::Cpu32Operation>(
            &cpu32_ops,
            max_rows.cpu32,
            cpu32::generate_cpu32_trace,
            #[cfg(feature = "disk-spill")]
            storage_mode,
        )
    };
    let gen_bitwise = || {
        let mut bitwise = bitwise::generate_bitwise_trace();
        // Fill the MU columns (11..=20) from the accumulated histogram.
        bitwise_histogram.fill_multiplicities(&mut bitwise);
        bitwise
    };
    // Each CPU operation looks up the DECODE table once; padding rows look up
    // pc=1 (the CPU padding entry). When CPU is split, each chunk pads
    // independently.
    let gen_decode = move || {
        let mut decode = decode_trace;
        let mut decode_lookups: Vec<u64> = cpu_ops_ref.iter().map(|op| op.decode.pc).collect();
        decode_lookups.extend(std::iter::repeat_n(cpu::CPU_PADDING_PC, num_padding_rows));
        decode::update_multiplicities(&mut decode, &decode_pc_to_row, &decode_lookups);
        decode
    };
    let gen_commit = || commit::generate_commit_trace(&commit_ops);
    let gen_keccak = || keccak::generate_keccak_trace(&keccak_ops);
    let gen_keccak_rnd = || {
        let keccak_rnd_ops: Vec<KeccakRoundOperation> = keccak_ops
            .iter()
            .map(|op| KeccakRoundOperation {
                timestamp: op.timestamp,
                input: op.input,
                output: op.output,
            })
            .collect();
        keccak_rnd::generate_keccak_rnd_trace(&keccak_rnd_ops)
    };
    let gen_keccak_rc = || {
        let mut keccak_rc_trace = keccak_rc::generate_keccak_rc_trace();
        keccak_rc::update_multiplicities(&mut keccak_rc_trace, keccak_ops.len());
        keccak_rc_trace
    };
    let gen_pages = || match initial_image {
        // Continuation epochs (l2g_memory_bookend) skip PAGE: the L2G table owns
        // every touched cell's Memory init/fini, and every untouched PAGE row
        // self-cancels (init==fini, ts=0), so PAGE contributes nothing here.
        Some(image) if !l2g_memory_bookend => {
            generate_page_tables(image, memory_state, private_input, l2g_memory_bookend)
        }
        _ => (Vec::new(), Vec::new()),
    };
    let gen_register = || register::generate_register_trace(&register_final_state, register_init);
    let gen_halt = || halt::generate_halt_trace(halt_timestamp, halt_next_pc);
    // ECSM accelerator traces (empty/all-padding for programs that do not use ECSM).
    let gen_ecsm = || ecsm::generate_ecsm_trace(&ecsm_ops);
    let gen_ec_scalar = || ec_scalar::generate_ec_scalar_trace(&ec_scalar_ops);
    let gen_ecdas = || ecdas::generate_ecdas_trace(&ecdas_ops);

    let (mut cpus_slot, mut memws_slot, mut memw_aligneds_slot, mut memw_registers_slot) =
        (None, None, None, None);
    let (mut loads_slot, mut lts_slot, mut shifts_slot, mut muls_slot) = (None, None, None, None);
    let (mut dvrms_slot, mut branches_slot, mut bitwise_slot, mut decode_slot) =
        (None, None, None, None);
    let (mut commit_slot, mut keccak_slot, mut keccak_rnd_slot, mut keccak_rc_slot) =
        (None, None, None, None);
    let (mut pages_slot, mut register_slot, mut halt_slot) = (None, None, None);
    let (mut eqs_slot, mut bytewises_slot, mut stores_slot, mut cpu32s_slot) =
        (None, None, None, None);
    let (mut ecsm_slot, mut ec_scalar_slot, mut ecdas_slot) = (None, None, None);

    #[cfg(feature = "disk-spill")]
    let sequential = storage_mode == StorageMode::Disk || cfg!(not(feature = "parallel"));
    #[cfg(not(feature = "disk-spill"))]
    let sequential = cfg!(not(feature = "parallel"));

    if !sequential {
        #[cfg(feature = "parallel")]
        rayon::scope(|s| {
            macro_rules! spawn_into {
                ($slot:ident, $gen:ident) => {{
                    let slot = &mut $slot;
                    s.spawn(move |_| *slot = Some($gen()));
                }};
            }
            // Heaviest builds first so the scheduler overlaps them with the rest.
            spawn_into!(memw_registers_slot, gen_memw_registers);
            spawn_into!(cpus_slot, gen_cpus);
            spawn_into!(memws_slot, gen_memws);
            spawn_into!(lts_slot, gen_lts);
            spawn_into!(decode_slot, gen_decode);
            spawn_into!(branches_slot, gen_branches);
            spawn_into!(bitwise_slot, gen_bitwise);
            spawn_into!(muls_slot, gen_muls);
            spawn_into!(memw_aligneds_slot, gen_memw_aligneds);
            spawn_into!(loads_slot, gen_loads);
            spawn_into!(shifts_slot, gen_shifts);
            spawn_into!(dvrms_slot, gen_dvrms);
            spawn_into!(pages_slot, gen_pages);
            spawn_into!(keccak_slot, gen_keccak);
            spawn_into!(keccak_rnd_slot, gen_keccak_rnd);
            spawn_into!(keccak_rc_slot, gen_keccak_rc);
            spawn_into!(commit_slot, gen_commit);
            spawn_into!(register_slot, gen_register);
            spawn_into!(halt_slot, gen_halt);
            spawn_into!(eqs_slot, gen_eqs);
            spawn_into!(bytewises_slot, gen_bytewises);
            spawn_into!(stores_slot, gen_stores);
            spawn_into!(cpu32s_slot, gen_cpu32s);
            spawn_into!(ecsm_slot, gen_ecsm);
            spawn_into!(ec_scalar_slot, gen_ec_scalar);
            spawn_into!(ecdas_slot, gen_ecdas);
        });
    } else {
        cpus_slot = Some(gen_cpus());
        memws_slot = Some(gen_memws());
        memw_aligneds_slot = Some(gen_memw_aligneds());
        memw_registers_slot = Some(gen_memw_registers());
        loads_slot = Some(gen_loads());
        lts_slot = Some(gen_lts());
        shifts_slot = Some(gen_shifts());
        muls_slot = Some(gen_muls());
        dvrms_slot = Some(gen_dvrms());
        branches_slot = Some(gen_branches());
        bitwise_slot = Some(gen_bitwise());
        decode_slot = Some(gen_decode());
        commit_slot = Some(gen_commit());
        keccak_slot = Some(gen_keccak());
        keccak_rnd_slot = Some(gen_keccak_rnd());
        keccak_rc_slot = Some(gen_keccak_rc());
        pages_slot = Some(gen_pages());
        register_slot = Some(gen_register());
        halt_slot = Some(gen_halt());
        eqs_slot = Some(gen_eqs());
        bytewises_slot = Some(gen_bytewises());
        stores_slot = Some(gen_stores());
        cpu32s_slot = Some(gen_cpu32s());
        ecsm_slot = Some(gen_ecsm());
        ec_scalar_slot = Some(gen_ec_scalar());
        ecdas_slot = Some(gen_ecdas());
    }

    const PHASE5_RAN: &str = "phase 5 generation ran in one of the branches above";
    let cpus = cpus_slot.expect(PHASE5_RAN)?;
    let memws = memws_slot.expect(PHASE5_RAN)?;
    let memw_aligneds = memw_aligneds_slot.expect(PHASE5_RAN)?;
    let memw_registers = memw_registers_slot.expect(PHASE5_RAN)?;
    let loads = loads_slot.expect(PHASE5_RAN)?;
    let lts = lts_slot.expect(PHASE5_RAN)?;
    let shifts = shifts_slot.expect(PHASE5_RAN)?;
    let muls = muls_slot.expect(PHASE5_RAN)?;
    let dvrms = dvrms_slot.expect(PHASE5_RAN)?;
    let branches = branches_slot.expect(PHASE5_RAN)?;
    let eqs = eqs_slot.expect(PHASE5_RAN)?;
    let bytewises = bytewises_slot.expect(PHASE5_RAN)?;
    let stores = stores_slot.expect(PHASE5_RAN)?;
    let cpu32s = cpu32s_slot.expect(PHASE5_RAN)?;
    #[allow(unused_mut)]
    let mut bitwise = bitwise_slot.expect(PHASE5_RAN);
    #[allow(unused_mut)]
    let mut decode = decode_slot.expect(PHASE5_RAN);
    #[allow(unused_mut)]
    let mut commit_trace = commit_slot.expect(PHASE5_RAN);
    let keccak_trace = keccak_slot.expect(PHASE5_RAN);
    let keccak_rnd_trace = keccak_rnd_slot.expect(PHASE5_RAN);
    let keccak_rc_trace = keccak_rc_slot.expect(PHASE5_RAN);
    #[allow(unused_mut)]
    let (mut pages, page_configs) = pages_slot.expect(PHASE5_RAN);
    #[allow(unused_mut)]
    let mut register_trace = register_slot.expect(PHASE5_RAN);
    #[allow(unused_mut)]
    let mut halt_trace = halt_slot.expect(PHASE5_RAN);
    let ecsm_trace = ecsm_slot.expect(PHASE5_RAN);
    let ec_scalar_trace = ec_scalar_slot.expect(PHASE5_RAN);
    let ecdas_trace = ecdas_slot.expect(PHASE5_RAN);

    // Fixed-size and per-page tables aren't built through `chunk_and_generate`,
    // so spill them here before returning.
    #[cfg(feature = "disk-spill")]
    if storage_mode == StorageMode::Disk {
        bitwise
            .main_table
            .spill_to_disk()
            .map_err(|e| Error::Prover(format!("disk-spill bitwise: {e}")))?;
        decode
            .main_table
            .spill_to_disk()
            .map_err(|e| Error::Prover(format!("disk-spill decode: {e}")))?;
        commit_trace
            .main_table
            .spill_to_disk()
            .map_err(|e| Error::Prover(format!("disk-spill commit: {e}")))?;
        register_trace
            .main_table
            .spill_to_disk()
            .map_err(|e| Error::Prover(format!("disk-spill register: {e}")))?;
        halt_trace
            .main_table
            .spill_to_disk()
            .map_err(|e| Error::Prover(format!("disk-spill halt: {e}")))?;
        for page in &mut pages {
            page.main_table
                .spill_to_disk()
                .map_err(|e| Error::Prover(format!("disk-spill page: {e}")))?;
        }
    }

    // Continuation callers derive the real cross-epoch boundary from this set and
    // install its L2G trace after provenance is applied. Avoid building a
    // throwaway genesis-only L2G trace here.
    let touched_memory_cells = if l2g_memory_bookend {
        touched_cells_from_memory_state(memory_state)
    } else {
        Vec::new()
    };
    let local_to_global = local_to_global::generate_local_to_global_trace(&[]);

    #[cfg(feature = "instruments")]
    drop(__sp);
    Ok(Traces {
        cpus,
        bitwise,
        lts,
        shifts,
        memws,
        memw_aligneds,
        loads,
        decode,
        muls,
        dvrms,
        pages,
        page_configs,
        register: register_trace,
        public_output_bytes,
        branches,
        halt: halt_trace,
        commit: commit_trace,
        keccak: keccak_trace,
        keccak_rnd: keccak_rnd_trace,
        keccak_rc: keccak_rc_trace,
        ecsm: ecsm_trace,
        ec_scalar: ec_scalar_trace,
        ecdas: ecdas_trace,
        memw_registers,
        local_to_global,
        touched_memory_cells,
        eqs,
        bytewises,
        stores,
        cpu32s,
    })
}

/// Padded row count after chunking.
#[cfg(feature = "disk-spill")]
fn padded_chunked_rows(ops_count: usize, max_rows: usize) -> u64 {
    // `max_rows <= 0` would loop forever. Called internally with const values > 0.
    assert!(max_rows > 0, "max_rows must be positive");
    if ops_count == 0 {
        return 4; // empty-chunk tables still allocate one 4-row padded chunk
    }
    let mut total: u64 = 0;
    let mut remaining = ops_count;
    while remaining > 0 {
        let chunk_size = remaining.min(max_rows);
        total += chunk_size.next_power_of_two().max(4) as u64;
        remaining -= chunk_size;
    }
    total
}

/// Per-table padded row counts plus auxiliary metrics for peak-heap estimation.
#[cfg(feature = "disk-spill")]
#[derive(Debug, Default, Clone)]
pub struct TableLengths {
    pub cpu_padded_rows: u64,
    pub memw_padded_rows: u64,
    pub memw_aligned_padded_rows: u64,
    pub memw_register_padded_rows: u64,
    pub load_padded_rows: u64,
    pub lt_padded_rows: u64,
    pub shift_padded_rows: u64,
    pub mul_padded_rows: u64,
    pub dvrm_padded_rows: u64,
    pub branch_padded_rows: u64,
    pub commit_padded_rows: u64,
    pub decode_rows: u64,
    pub unique_page_count: u64,
    pub cycle_count: u64,
    pub unique_byte_count: u64,
}

/// Per-table row counts from `logs`, without building op vectors.
/// Exact for tables that don't dedup; upper bound for LT, MUL, DVRM, BRANCH.
/// Must stay in sync with `Traces::from_elf_and_logs`.
#[cfg(feature = "disk-spill")]
pub fn count_table_lengths(
    elf: &Elf,
    logs: &[Log],
    max_rows: &super::MaxRowsConfig,
    private_input: &[u8],
) -> Result<TableLengths, Error> {
    // Phase 0: ELF → instructions + DECODE row count.
    let instructions = decode::instructions_from_elf(elf)
        .map_err(|e| Error::Execution(format!("Failed to parse instructions: {e}")))?;
    // Mirrors the padding inside `generate_decode_trace`.
    let decode_rows = (instructions.len() as u64 + 1).next_power_of_two().max(2);

    // Memory + register state for partition predicates that need timestamps.
    let mut memory_state = MemoryState::from_image(&build_initial_image(elf, private_input));
    let mut register_state = RegisterState::new(elf.entry_point);

    // Raw counts (pre-chunking + pre-padding).
    let mut cpu_count = 0usize;
    // Wide-MEMW counts bucketed by width, used by the LT-from-MEMW derivation.
    let mut memw_by_width: [usize; 4] = [0; 4];
    let mut memw_aligned_count = 0usize;
    let mut memw_register_count = 0usize;
    let mut load_count = 0usize;
    let mut lt_count = 0usize;
    let mut shift_count = 0usize;
    let mut mul_count = 0usize;
    let mut dvrm_count = 0usize;
    let mut branch_count = 0usize;
    let mut commit_count = 0usize;
    let mut current_commit_index = 0u32;

    let partition_memw = |op: &MemwOperation,
                          by_width: &mut [usize; 4],
                          aligned: &mut usize,
                          register: &mut usize| {
        // Same classifier as the walk's MemwBuckets, so the sizing pass counts
        // exactly the rows trace generation will produce.
        match classify_memw(op) {
            MemwRoute::Register => *register += 1,
            MemwRoute::Aligned => *aligned += 1,
            MemwRoute::General => {
                let idx = match op.width {
                    1 => 0,
                    2 => 1,
                    4 => 2,
                    8 => 3,
                    _ => return,
                };
                by_width[idx] += 1;
            }
        }
    };

    let mut reg_memw_scratch: Vec<MemwOperation> = Vec::with_capacity(4);
    for (i, log) in logs.iter().enumerate() {
        let timestamp = (i as u64) * 4 + 4;
        let instruction = instructions
            .get(&log.current_pc)
            .copied()
            .ok_or(Error::MissingInstruction(log.current_pc))?;
        let cpu_op = CpuOperation::from_log_and_instruction(log, timestamp, instruction);
        cpu_count += 1;

        // Memory ops from load/store
        if cpu_op.decode.fields.is_load() {
            let (memw_op, _load_op, _bitwise) =
                collect_load_op_from_cpu(&cpu_op, &mut memory_state);
            partition_memw(
                &memw_op,
                &mut memw_by_width,
                &mut memw_aligned_count,
                &mut memw_register_count,
            );
            load_count += 1;
        } else if cpu_op.decode.fields.is_store() {
            let memw_op = collect_store_op_from_cpu(&cpu_op, &mut memory_state);
            partition_memw(
                &memw_op,
                &mut memw_by_width,
                &mut memw_aligned_count,
                &mut memw_register_count,
            );
        }

        // Register accesses.
        reg_memw_scratch.clear();
        collect_register_ops_from_cpu(&cpu_op, &mut register_state, &mut reg_memw_scratch);
        for memw_op in &reg_memw_scratch {
            partition_memw(
                memw_op,
                &mut memw_by_width,
                &mut memw_aligned_count,
                &mut memw_register_count,
            );
        }

        // ECALL Commit
        if cpu_op.ecall_commit {
            // Match `expand_commit_operations_for_ecall`'s `0..=count` loop
            // without building the op vector.
            commit_count += (cpu_op.commit_count as usize)
                .checked_add(1)
                .ok_or_else(|| Error::Execution("commit_count overflows usize".into()))?;
            let reg_commit_ops =
                collect_commit_memw_ops(&cpu_op, &mut register_state, &mut memory_state);
            for memw_op in &reg_commit_ops {
                partition_memw(
                    memw_op,
                    &mut memw_by_width,
                    &mut memw_aligned_count,
                    &mut memw_register_count,
                );
            }
            let count = u32::try_from(cpu_op.commit_count)
                .map_err(|_| Error::Execution("commit_count exceeds u32 range".into()))?;
            current_commit_index = current_commit_index
                .checked_add(count)
                .ok_or_else(|| Error::Execution("commit index exceeds u32 range".into()))?;
        }

        // CPU-side per-instruction-kind counters (non-word; word → CPU32, B5b)
        let f = &cpu_op.decode.fields;
        if !f.word_instr && f.is_lt() {
            lt_count += 1;
        }
        if !f.word_instr && f.is_shift() {
            shift_count += 1;
        }
        if !f.word_instr && f.is_mul() {
            mul_count += 1;
        }
        if !f.word_instr && f.is_divrem() {
            dvrm_count += 1;
        }
        if cpu_op.branch_cond {
            branch_count += 1;
        }
    }

    // HALT finalization. Halt ops fall through to wide MEMW.
    let halt_memw_ops = collect_halt_ops(&mut register_state);
    for memw_op in &halt_memw_ops {
        partition_memw(
            memw_op,
            &mut memw_by_width,
            &mut memw_aligned_count,
            &mut memw_register_count,
        );
    }

    // LT ops derived from wide-MEMW and memw_aligned ops.
    let memw_count = memw_by_width.iter().sum::<usize>();
    let lt_from_memw =
        memw_by_width[0] + 2 * memw_by_width[1] + 4 * memw_by_width[2] + 8 * memw_by_width[3];
    lt_count += lt_from_memw + memw_aligned_count;

    // DVRM derives mul and lt ops.
    mul_count += 2 * dvrm_count;
    lt_count += dvrm_count;

    let unique_page_count = memory_state.unique_page_count(page::DEFAULT_PAGE_SIZE as u64);
    let unique_byte_count = memory_state.cells.len() as u64;
    let cycle_count = logs.len() as u64;

    Ok(TableLengths {
        cpu_padded_rows: padded_chunked_rows(cpu_count, max_rows.cpu),
        memw_padded_rows: padded_chunked_rows(memw_count, max_rows.memw),
        memw_aligned_padded_rows: padded_chunked_rows(memw_aligned_count, max_rows.memw_aligned),
        memw_register_padded_rows: padded_chunked_rows(memw_register_count, max_rows.memw_register),
        load_padded_rows: padded_chunked_rows(load_count, max_rows.load),
        lt_padded_rows: padded_chunked_rows(lt_count, max_rows.lt),
        shift_padded_rows: padded_chunked_rows(shift_count, max_rows.shift),
        mul_padded_rows: padded_chunked_rows(mul_count, max_rows.mul),
        dvrm_padded_rows: padded_chunked_rows(dvrm_count, max_rows.dvrm),
        branch_padded_rows: padded_chunked_rows(branch_count, max_rows.branch),
        commit_padded_rows: commit_count
            .checked_next_power_of_two()
            .unwrap_or(usize::MAX)
            .max(4) as u64,
        decode_rows,
        unique_page_count,
        cycle_count,
        unique_byte_count,
    })
}

impl Traces {
    /// Returns the total number of main-trace field elements across all tables.
    ///
    /// Counts only the main (base-field) trace columns — equivalent to SP1's
    /// `main_area` — for apples-to-apples comparison with other zkVMs.
    ///
    /// Preprocessed columns (committed in a separate PCS round during setup, not at
    /// proving time) are excluded: BITWISE (11), DECODE (5), REGISTER (2), PAGE (2).
    pub fn total_field_elements(&self) -> u64 {
        use super::bitwise::NUM_PRECOMPUTED_COLS as BITWISE_PRECOMPUTED;
        use super::bitwise::cols::NUM_COLUMNS as BITWISE_COLS;
        use super::branch::cols::NUM_COLUMNS as BRANCH_COLS;
        use super::bytewise::cols::NUM_COLUMNS as BYTEWISE_COLS;
        use super::commit::cols::NUM_COLUMNS as COMMIT_COLS;
        use super::cpu::cols::NUM_COLUMNS as CPU_COLS;
        use super::cpu32::cols::NUM_COLUMNS as CPU32_COLS;
        use super::decode::NUM_PRECOMPUTED_COLS as DECODE_PRECOMPUTED;
        use super::decode::cols::NUM_COLUMNS as DECODE_COLS;
        use super::dvrm::cols::NUM_COLUMNS as DVRM_COLS;
        use super::ec_scalar::cols::NUM_COLUMNS as EC_SCALAR_COLS;
        use super::ecdas::cols::NUM_COLUMNS as ECDAS_COLS;
        use super::ecsm::cols::NUM_COLUMNS as ECSM_COLS;
        use super::eq::cols::NUM_COLUMNS as EQ_COLS;
        use super::halt::cols::NUM_COLUMNS as HALT_COLS;
        use super::keccak::cols::NUM_COLUMNS as KECCAK_COLS;
        use super::keccak_rc::NUM_PRECOMPUTED_COLS as KECCAK_RC_PRECOMPUTED;
        use super::keccak_rc::cols::NUM_COLUMNS as KECCAK_RC_COLS;
        use super::keccak_rnd::cols::NUM_COLUMNS as KECCAK_RND_COLS;
        use super::load::cols::NUM_COLUMNS as LOAD_COLS;
        use super::lt::cols::NUM_COLUMNS as LT_COLS;
        use super::memw::cols::NUM_COLUMNS as MEMW_COLS;
        use super::memw_aligned::cols::NUM_COLUMNS as MEMW_A_COLS;
        use super::memw_register::cols::NUM_COLUMNS as MEMW_R_COLS;
        use super::mul::cols::NUM_COLUMNS as MUL_COLS;
        use super::page::NUM_PREPROCESSED_COLS as PAGE_PREPROCESSED;
        use super::page::cols::NUM_COLUMNS as PAGE_COLS;
        use super::register::NUM_PREPROCESSED_COLS as REGISTER_PREPROCESSED;
        use super::register::cols::NUM_COLUMNS as REGISTER_COLS;
        use super::shift::cols::NUM_COLUMNS as SHIFT_COLS;
        use super::store::cols::NUM_COLUMNS as STORE_COLS;

        let Traces {
            cpus,
            bitwise,
            lts,
            shifts,
            memws,
            memw_aligneds,
            loads,
            decode,
            muls,
            dvrms,
            pages,
            register,
            branches,
            halt,
            commit,
            keccak,
            keccak_rnd,
            keccak_rc,
            ecsm,
            ec_scalar,
            ecdas,
            memw_registers,
            eqs,
            bytewises,
            stores,
            cpu32s,
            page_configs: _,
            public_output_bytes: _,
            local_to_global: _,
            touched_memory_cells: _,
        } = self;

        let mut total: u64 = 0;
        for t in cpus {
            total += (t.num_rows() * CPU_COLS) as u64;
        }
        total += (bitwise.num_rows() * (BITWISE_COLS - BITWISE_PRECOMPUTED)) as u64;
        for t in lts {
            total += (t.num_rows() * LT_COLS) as u64;
        }
        for t in shifts {
            total += (t.num_rows() * SHIFT_COLS) as u64;
        }
        for t in memws {
            total += (t.num_rows() * MEMW_COLS) as u64;
        }
        for t in memw_aligneds {
            total += (t.num_rows() * MEMW_A_COLS) as u64;
        }
        for t in loads {
            total += (t.num_rows() * LOAD_COLS) as u64;
        }
        total += (decode.num_rows() * (DECODE_COLS - DECODE_PRECOMPUTED)) as u64;
        for t in muls {
            total += (t.num_rows() * MUL_COLS) as u64;
        }
        for t in dvrms {
            total += (t.num_rows() * DVRM_COLS) as u64;
        }
        for t in branches {
            total += (t.num_rows() * BRANCH_COLS) as u64;
        }
        total += (halt.num_rows() * HALT_COLS) as u64;
        total += (commit.num_rows() * COMMIT_COLS) as u64;
        total += (register.num_rows() * (REGISTER_COLS - REGISTER_PREPROCESSED)) as u64;
        for t in pages {
            total += (t.num_rows() * (PAGE_COLS - PAGE_PREPROCESSED)) as u64;
        }
        for t in memw_registers {
            total += (t.num_rows() * MEMW_R_COLS) as u64;
        }
        total += (keccak.num_rows() * KECCAK_COLS) as u64;
        total += (keccak_rnd.num_rows() * KECCAK_RND_COLS) as u64;
        total += (keccak_rc.num_rows() * (KECCAK_RC_COLS - KECCAK_RC_PRECOMPUTED)) as u64;
        for t in eqs {
            total += (t.num_rows() * EQ_COLS) as u64;
        }
        for t in bytewises {
            total += (t.num_rows() * BYTEWISE_COLS) as u64;
        }
        for t in stores {
            total += (t.num_rows() * STORE_COLS) as u64;
        }
        for t in cpu32s {
            total += (t.num_rows() * CPU32_COLS) as u64;
        }
        total += (ecsm.num_rows() * ECSM_COLS) as u64;
        total += (ec_scalar.num_rows() * EC_SCALAR_COLS) as u64;
        total += (ecdas.num_rows() * ECDAS_COLS) as u64;
        total
    }

    /// Returns the total number of auxiliary-trace field elements (extension field)
    /// across all tables.
    ///
    /// The LogUp layout packs N bus interactions into ⌈N/2⌉ EF columns
    /// (`num_committed_pairs + 1` accumulated column). Each EF column costs one
    /// extension-field element per row.
    pub fn total_auxiliary_field_elements(&self) -> u64 {
        // ⌈N/2⌉ = number of aux EF columns for a table with N bus interactions.
        fn aux_cols(n: usize) -> usize {
            n.div_ceil(2)
        }

        let n_cpu = aux_cols(super::cpu::bus_interactions().len());
        let n_bitwise = aux_cols(super::bitwise::bus_interactions().len());
        let n_lt = aux_cols(super::lt::bus_interactions().len());
        let n_shift = aux_cols(super::shift::bus_interactions().len());
        let n_memw = aux_cols(super::memw::bus_interactions().len());
        let n_memw_a = aux_cols(super::memw_aligned::bus_interactions().len());
        let n_load = aux_cols(super::load::bus_interactions().len());
        let n_decode = aux_cols(super::decode::bus_interactions().len());
        let n_mul = aux_cols(super::mul::bus_interactions().len());
        let n_dvrm = aux_cols(super::dvrm::bus_interactions().len());
        let n_branch = aux_cols(super::branch::bus_interactions().len());
        let n_halt = aux_cols(super::halt::bus_interactions().len());
        let n_commit = aux_cols(super::commit::bus_interactions().len());
        let n_register = aux_cols(super::register::bus_interactions().len());
        // page::bus_interactions count is constant regardless of page_base.
        let n_page = aux_cols(super::page::bus_interactions(0).len());
        let n_memw_r = aux_cols(super::memw_register::bus_interactions().len());
        let n_keccak = aux_cols(super::keccak::bus_interactions().len());
        let n_keccak_rnd = aux_cols(super::keccak_rnd::bus_interactions().len());
        let n_keccak_rc = aux_cols(super::keccak_rc::bus_interactions().len());
        let n_eq = aux_cols(super::eq::bus_interactions().len());
        let n_bytewise = aux_cols(super::bytewise::bus_interactions().len());
        let n_store = aux_cols(super::store::bus_interactions().len());
        let n_cpu32 = aux_cols(super::cpu32::bus_interactions().len());
        let n_ecsm = aux_cols(super::ecsm::bus_interactions().len());
        let n_ec_scalar = aux_cols(super::ec_scalar::bus_interactions().len());
        let n_ecdas = aux_cols(super::ecdas::bus_interactions().len());

        let Traces {
            cpus,
            bitwise,
            lts,
            shifts,
            memws,
            memw_aligneds,
            loads,
            decode,
            muls,
            dvrms,
            pages,
            register,
            branches,
            halt,
            commit,
            keccak,
            keccak_rnd,
            keccak_rc,
            ecsm,
            ec_scalar,
            ecdas,
            memw_registers,
            eqs,
            bytewises,
            stores,
            cpu32s,
            page_configs: _,
            public_output_bytes: _,
            local_to_global: _,
            touched_memory_cells: _,
        } = self;

        let mut total: u64 = 0;
        for t in cpus {
            total += (t.num_rows() * n_cpu) as u64;
        }
        total += (bitwise.num_rows() * n_bitwise) as u64;
        for t in lts {
            total += (t.num_rows() * n_lt) as u64;
        }
        for t in shifts {
            total += (t.num_rows() * n_shift) as u64;
        }
        for t in memws {
            total += (t.num_rows() * n_memw) as u64;
        }
        for t in memw_aligneds {
            total += (t.num_rows() * n_memw_a) as u64;
        }
        for t in loads {
            total += (t.num_rows() * n_load) as u64;
        }
        total += (decode.num_rows() * n_decode) as u64;
        for t in muls {
            total += (t.num_rows() * n_mul) as u64;
        }
        for t in dvrms {
            total += (t.num_rows() * n_dvrm) as u64;
        }
        for t in branches {
            total += (t.num_rows() * n_branch) as u64;
        }
        total += (halt.num_rows() * n_halt) as u64;
        total += (commit.num_rows() * n_commit) as u64;
        total += (register.num_rows() * n_register) as u64;
        for t in pages {
            total += (t.num_rows() * n_page) as u64;
        }
        for t in memw_registers {
            total += (t.num_rows() * n_memw_r) as u64;
        }
        total += (keccak.num_rows() * n_keccak) as u64;
        total += (keccak_rnd.num_rows() * n_keccak_rnd) as u64;
        total += (keccak_rc.num_rows() * n_keccak_rc) as u64;
        for t in eqs {
            total += (t.num_rows() * n_eq) as u64;
        }
        for t in bytewises {
            total += (t.num_rows() * n_bytewise) as u64;
        }
        for t in stores {
            total += (t.num_rows() * n_store) as u64;
        }
        for t in cpu32s {
            total += (t.num_rows() * n_cpu32) as u64;
        }
        total += (ecsm.num_rows() * n_ecsm) as u64;
        total += (ec_scalar.num_rows() * n_ec_scalar) as u64;
        total += (ecdas.num_rows() * n_ecdas) as u64;
        total
    }

    /// Returns the number of chunks for each split table.
    pub fn table_counts(&self) -> crate::TableCounts {
        crate::TableCounts {
            cpu: self.cpus.len(),
            lt: self.lts.len(),
            memw: self.memws.len(),
            memw_aligned: self.memw_aligneds.len(),
            load: self.loads.len(),
            mul: self.muls.len(),
            dvrm: self.dvrms.len(),
            shift: self.shifts.len(),
            branch: self.branches.len(),
            memw_register: self.memw_registers.len(),
            eq: self.eqs.len(),
            bytewise: self.bytewises.len(),
            store: self.stores.len(),
            cpu32: self.cpu32s.len(),
        }
    }

    /// Extract page configurations from ELF only (deterministic from binary).
    ///
    /// Returns PageConfigs for pages covered by ELF segments, with their
    /// init data populated. Used by the verifier to reconstruct the ELF
    /// portion of the PAGE table layout.
    pub fn page_configs_from_elf(elf: &Elf) -> Vec<PageConfig> {
        use std::collections::BTreeSet;

        let init_page_data = build_init_page_data(&build_initial_image(elf, &[]));

        let page_bases: BTreeSet<u64> = init_page_data.keys().copied().collect();

        page_bases
            .into_iter()
            .map(|base| {
                if let Some(init_data) = init_page_data.get(&base) {
                    PageConfig::with_data(base, init_data.clone())
                } else {
                    PageConfig::zero_init(base)
                }
            })
            .collect()
    }

    /// Reconstruct page configs from ELF, runtime page ranges, and private-input page count.
    ///
    /// Used by the verifier to reconstruct the full PAGE table layout.
    /// Combines:
    /// - Deterministic ELF pages (preprocessed, init from binary)
    /// - Runtime pages from prover hints (preprocessed, zero-init)
    /// - Private-input pages (NOT preprocessed, verifier doesn't see init values)
    pub fn page_configs_from_elf_and_runtime(
        elf: &Elf,
        runtime_page_ranges: &[crate::RuntimePageRange],
        num_private_input_pages: usize,
    ) -> Vec<PageConfig> {
        let mut configs = Self::page_configs_from_elf(elf);
        let page_size = page::DEFAULT_PAGE_SIZE;

        // Add zero-init runtime pages (stack, heap)
        for r in runtime_page_ranges {
            let (base, count) = (r.base, r.count);
            for i in 0..count {
                configs.push(PageConfig::zero_init(base + i * page_size as u64));
            }
        }

        // Add private-input pages (non-preprocessed, verifier doesn't know init values)
        for page_base in page::private_input_page_bases(num_private_input_pages) {
            configs.push(PageConfig {
                page_base,
                init_values: None, // Verifier doesn't know these
                is_private_input: true,
            });
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
        private_input: &[u8],
        #[cfg(feature = "disk-spill")] storage_mode: StorageMode,
    ) -> Result<Self, Error> {
        let initial_image = build_initial_image(elf, private_input);
        let register_init = register::register_init_from_entry_point(elf.entry_point);
        Self::from_image_and_logs(
            elf,
            &initial_image,
            &register_init,
            logs,
            max_rows,
            private_input,
            true,
            false,
            #[cfg(feature = "disk-spill")]
            storage_mode,
        )
    }

    /// Build traces for one execution epoch starting from an explicit
    /// initial-memory image (the epoch's starting memory) rather than the ELF
    /// image. `elf` is still used for the program code (DECODE) and entry point.
    ///
    /// `register_init` is the epoch's starting register image (word address ->
    /// value): the program-start image for the first epoch, or an epoch's boundary
    /// register snapshot for later epochs. It seeds both `RegisterState` (for
    /// first-access old values) and the REGISTER table's init column.
    ///
    /// `is_final` marks the last epoch: it applies HALT finalization (zeroize
    /// registers, require the terminating ECALL). Intermediate epochs (`false`)
    /// skip HALT and keep their boundary register/memory state.
    #[allow(clippy::too_many_arguments)]
    pub fn from_image_and_logs<I: ImageSource + Sync>(
        elf: &Elf,
        initial_image: &I,
        register_init: &[u32],
        logs: &[Log],
        max_rows: &super::MaxRowsConfig,
        private_input: &[u8],
        is_final: bool,
        l2g_memory_bookend: bool,
        #[cfg(feature = "disk-spill")] storage_mode: StorageMode,
    ) -> Result<Self, Error> {
        // A non-final epoch must not contain the program-terminating instruction
        // (next_pc == 0). Otherwise the CPU sends an ECALL bus token with no HALT
        // table to receive it (HALT is excluded when !is_final), producing an
        // unverifiable proof. Fail explicitly instead.
        if !is_final && logs.iter().any(|log| log.next_pc == 0) {
            return Err(Error::HaltInNonFinalEpoch);
        }

        // Phase 0: ELF → DECODE + instructions
        // IMPORTANT: Use generate_decode_trace (same as compute_precomputed_commitment)
        // so the DECODE trace row ordering matches the AIR's hardcoded commitment.
        #[cfg(feature = "instruments")]
        let __sp = stark::instruments::span("p0_decode");
        let instructions = decode::instructions_from_elf(elf)
            .map_err(|e| Error::Execution(format!("Failed to parse instructions: {e}")))?;
        let (decode_trace, decode_pc_to_row) = decode::generate_decode_trace(&instructions);
        #[cfg(feature = "instruments")]
        drop(__sp);

        // Phase 1: Logs → CPU operations
        #[cfg(feature = "instruments")]
        let __sp = stark::instruments::span("p1_cpu_ops");
        let cpu_ops = collect_cpu_ops(logs, &instructions)?;
        #[cfg(feature = "instruments")]
        drop(__sp);

        // Phase 2: Collect + route all ops
        let mut memory_state = MemoryState::from_image(initial_image);
        let mut register_state = RegisterState::from_init(register_init);
        #[cfg(feature = "instruments")]
        let __sp = stark::instruments::span("p2a_collect_cpu");
        let (
            memw_ops,
            load_ops,
            lt_ops,
            shift_ops,
            bitwise_ops,
            commit_ops,
            keccak_ops,
            cpu32_ops,
            ecsm_ops,
            ec_scalar_ops,
            ecdas_ops,
        ) = collect_ops_from_cpu(&cpu_ops, &mut memory_state, &mut register_state);
        #[cfg(feature = "instruments")]
        drop(__sp);

        #[cfg(feature = "instruments")]
        let __sp = stark::instruments::span("p2b_collect_all");
        let ops = collect_all_ops(
            cpu_ops,
            memw_ops,
            load_ops,
            lt_ops,
            shift_ops,
            bitwise_ops,
            commit_ops,
            keccak_ops,
            cpu32_ops,
            ecsm_ops,
            ec_scalar_ops,
            ecdas_ops,
            &mut register_state,
            is_final,
        );
        #[cfg(feature = "instruments")]
        drop(__sp);

        // Phases 3-5
        #[cfg(feature = "instruments")]
        let __sp = stark::instruments::span("p3to5_build_traces");
        let result = build_traces(
            ops,
            Some(initial_image),
            &memory_state,
            register_init,
            decode_trace,
            decode_pc_to_row,
            register_state,
            max_rows,
            #[cfg(feature = "disk-spill")]
            storage_mode,
            private_input,
            is_final,
            l2g_memory_bookend,
        );
        #[cfg(feature = "instruments")]
        drop(__sp);
        result
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
        // Phase 1: Logs → CPU operations
        let cpu_ops = collect_cpu_ops(logs, &instructions)?;

        // Phase 2: Collect + route all ops
        let mut memory_state = MemoryState::new();
        let entry_point = cpu_ops.first().map_or(0, |op| op.decode.pc);
        let register_init = register::register_init_from_entry_point(entry_point);
        let mut register_state = RegisterState::new(entry_point);
        let (
            memw_ops,
            load_ops,
            lt_ops,
            shift_ops,
            bitwise_ops,
            commit_ops,
            keccak_ops,
            cpu32_ops,
            ecsm_ops,
            ec_scalar_ops,
            ecdas_ops,
        ) = collect_ops_from_cpu(&cpu_ops, &mut memory_state, &mut register_state);

        let ops = collect_all_ops(
            cpu_ops,
            memw_ops,
            load_ops,
            lt_ops,
            shift_ops,
            bitwise_ops,
            commit_ops,
            keccak_ops,
            cpu32_ops,
            ecsm_ops,
            ec_scalar_ops,
            ecdas_ops,
            &mut register_state,
            true,
        );

        // DECODE (from_elf_and_logs does this in Phase 0; same result either way)
        let (decode_trace, decode_pc_to_row) = decode::generate_decode_trace(&instructions);

        // Phases 3-5 (elf=None → empty PAGE tables)
        build_traces(
            ops,
            None::<&HashMap<u64, u8>>,
            &memory_state,
            &register_init,
            decode_trace,
            decode_pc_to_row,
            register_state,
            max_rows,
            #[cfg(feature = "disk-spill")]
            StorageMode::Ram,
            &[],
            true,
            false,
        )
    }
}
