//! CPU table for the 64-bit VM.
//!
//! The CPU table is the central execution table. Following `spec/src/cpu.toml`
//! it is narrow (37 columns): there are no per-opcode one-hot ALU selectors and
//! no `*_ext_bit`/`arg1` columns. Instead each row carries:
//! - top-level flags `ALU/ADD/SUB/MEMORY/BRANCH/ECALL` (+ `word_instr`),
//! - the packed `alu_flags`/`mem_flags` bytes (the chips unpack them), and
//! - register indices + read/write flags.
//!
//! Dispatch happens over a small set of buses:
//! - `DECODE[pc, imm, packed_decode]` (mult `1 - word_instr`): instruction fetch.
//! - `ALU[rv1, arg2, alu_flags] -> res` (mult `ALU`): unified ALU lookup; the
//!   lt/mul/dvrm/shift/eq/bytewise chips receive on it, keyed by `alu_flags`.
//! - `MEMORY[timestamp, address, rv2, mem_flags] -> rvd` (mult `MEMORY`): high
//!   level LOAD/STORE dispatch (the LOAD/STORE chips receive on it).
//! - `CPU32[timestamp, pc, half_instruction_length]` (mult `word_instr`): every word
//!   (`*W`) instruction is delegated to the CPU32 table, which does its own
//!   register I/O and sign-extension. On a `word_instr` row the main CPU is a
//!   pure delegate: all operational flags are 0 and only the PC advances.
//! - `MEMW` register read/write (×3), `BRANCH`, `ECALL`, inline-PC `memory`
//!   tokens, and `ARE_BYTES`/`IS_HALF` range checks.
//!
//! `JALR` is virtual: under `BRANCH` the `mem_flags` byte only ever holds the
//! JALR bit (the memory-width bits are 0), so `mem_flags ∈ {0,1} = JALR` and the
//! `mem_flags` column is used directly as `JALR` wherever it is gated by `BRANCH`.

use super::types::{BusId, DecodeEntry, FE, GoldilocksExtension, GoldilocksField, VmTable, alu_op};
use crate::Error;
use executor::vm::{
    instruction::{decoding::Instruction, execution::SyscallNumbers},
    logs::Log,
    memory::U64HashMap,
};
use math::polynomial::Polynomial;
use stark::commitment::{ROWS_PER_LEAF, commit_bit_reversed};
use stark::config::Commitment;
use stark::lookup::{BusInteraction, BusValue, LinearTerm, Multiplicity, Packing};
use stark::proof::options::ProofOptions;
use stark::prover::evaluate_polynomial_on_lde_domain;
use stark::trace::TraceTable;

/// PC value used for CPU padding rows. Per spec this is an odd address
/// (unreachable during normal execution); the DECODE table contains a matching
/// padding entry at this PC (all flags 0, `half_instruction_length = 0`).
pub const CPU_PADDING_PC: u64 = 1;

/// Number of preprocessed columns: `TIMESTAMP` (column 0).
///
/// Row `i` of every CPU table holds `4·(i + 1)`, as spec `config.toml` defines
/// the `Timestamp` type. An epoch (or a monolithic proof) splits its CPU rows
/// over several tables, so table `k` adds the constant `4·S_k` to every
/// timestamp it sends, where `S_k` is the total height of tables `0..k` (see
/// [`timestamp_bases`]). The verifier reads the heights off the proof, so every
/// CPU timestamp is verifier-known.
pub const NUM_PRECOMPUTED_COLS: usize = 1;

/// Timestamp slots per CPU row (spec `memory.typ`, granularity aside).
pub const TIMESTAMP_STRIDE: u64 = 4;

/// Maximum number of CPU rows (padding included) in an epoch or a monolithic
/// proof. The last row's accesses reach `4·N + 2`, which must stay below the
/// HALT timestamp `2^32 - 1` (spec `streaming.typ`: an epoch should not exceed
/// `2^30` cycles; `N = 2^30` would put the last timestamp at `2^32`).
pub const MAX_CPU_ROWS: u64 = (1 << 30) - 1;

/// Largest continuation epoch, `log2` of its cycle count: `2^29` is the largest
/// power of two within [`MAX_CPU_ROWS`].
pub const MAX_EPOCH_SIZE_LOG2: u32 = 29;

/// Timestamp the final PC token is written at: the last CPU row (`4·N`) emits
/// `next_pc` at `4·N + 1`. Spec `halt.typ`: the verifier knows it.
pub fn final_pc_timestamp(total_rows: u64) -> u64 {
    TIMESTAMP_STRIDE * total_rows + 1
}

/// The constant each CPU table adds to its preprocessed timestamps:
/// `4·(h_0 + … + h_{k-1})` for table `k`.
pub fn timestamp_bases(heights: &[usize]) -> Vec<u64> {
    heights
        .iter()
        .scan(0u64, |rows, &h| {
            let base = TIMESTAMP_STRIDE * *rows;
            *rows += h as u64;
            Some(base)
        })
        .collect()
}

/// Checks the CPU table heights of an epoch or monolithic proof: each a power
/// of two in `[4, max_rows::CPU]` (the range the static timestamp commitments
/// cover) and at most [`MAX_CPU_ROWS`] in total.
pub fn validate_heights(heights: &[usize]) -> Result<(), Error> {
    let mut total: u64 = 0;
    for &h in heights {
        if !h.is_power_of_two() || !(4..=super::max_rows::CPU).contains(&h) {
            return Err(Error::InvalidTableCounts(format!(
                "CPU table height must be a power of two in [4, {}], got {h}",
                super::max_rows::CPU
            )));
        }
        total += h as u64;
    }
    if total > MAX_CPU_ROWS {
        return Err(Error::InvalidTableCounts(format!(
            "{total} CPU rows exceed the per-epoch maximum {MAX_CPU_ROWS}"
        )));
    }
    Ok(())
}

/// The CPU table heights the trace builder produces for `rows` executed cycles
/// split into chunks of `max_rows_cpu`: full chunks, then the last one padded to
/// a power of two (at least 4). Lets the prover check [`validate_heights`] right
/// after execution, before building any trace.
pub fn table_heights(rows: usize, max_rows_cpu: usize) -> Vec<usize> {
    (0..rows)
        .step_by(max_rows_cpu.max(1))
        .map(|start| (rows - start).min(max_rows_cpu).next_power_of_two().max(4))
        .collect()
}

/// Checks a prover's CPU chunk size before execution: every CPU table but the
/// last has exactly `max_rows_cpu` rows, so it must be a height
/// [`validate_heights`] accepts. Fails before the run instead of after the
/// trace build.
pub fn validate_max_rows(max_rows_cpu: usize) -> Result<(), Error> {
    validate_heights(&[max_rows_cpu])
}

/// The preprocessed `TIMESTAMP` commitment of a CPU table of `height` rows.
/// Static for the standard options (see [`static_timestamp_commitment`]),
/// recomputed otherwise.
pub fn timestamp_commitment(options: &ProofOptions, height: usize) -> Commitment {
    debug_assert!(
        height.is_power_of_two(),
        "CPU table height {height} must be checked by validate_heights first"
    );
    if options.coset_offset == 3
        && let Some(commitment) =
            static_timestamp_commitment(options.blowup_factor, height.trailing_zeros())
    {
        return commitment;
    }
    // Once per process: the verifier asks for one commitment per CPU table.
    static WARNED: std::sync::Once = std::sync::Once::new();
    WARNED.call_once(|| {
        log::warn!(
            "CPU timestamp commitment not static for (blowup={}, coset={}); \
             falling back to recompute. Add it to `CPU_TIMESTAMP_COMMITMENTS` by running \
             `cargo run --bin compute_static_commitments --release`.",
            options.blowup_factor,
            options.coset_offset,
        );
    });
    compute_timestamp_commitment(options, height)
}

/// Recomputes the preprocessed `TIMESTAMP` commitment of a CPU table of `height`
/// rows: the column `4·(i + 1)`, low-degree extended and committed exactly as
/// the prover commits a table's precomputed columns.
pub fn compute_timestamp_commitment(options: &ProofOptions, height: usize) -> Commitment {
    let column: Vec<FE> = (0..height as u64)
        .map(|i| FE::from(TIMESTAMP_STRIDE * (i + 1)))
        .collect();
    commit_timestamp_column(options, &column)
}

/// Commits a CPU `TIMESTAMP` column as the prover commits a table's precomputed
/// columns. Soundness tests use it to commit a forged column.
pub(crate) fn commit_timestamp_column(options: &ProofOptions, column: &[FE]) -> Commitment {
    let poly = Polynomial::interpolate_fft::<GoldilocksField>(column)
        .expect("FFT interpolation failed for CPU timestamp column");
    let lde = evaluate_polynomial_on_lde_domain(
        &poly,
        options.blowup_factor as usize,
        column.len(),
        &FE::from(options.coset_offset),
    )
    .expect("LDE evaluation failed for CPU timestamp column");
    let (_, root) = commit_bit_reversed(&[lde], ROWS_PER_LEAF)
        .expect("Failed to build Merkle tree for CPU timestamp LDE");
    root
}

/// Static CPU timestamp commitments for `log2(height)` in
/// `2..=log2(max_rows::CPU)`, per standard blowup factor (coset offset 3).
///
/// # Regenerating
///
/// Only after a deliberate change to the timestamp column or the FFT / LDE /
/// Merkle pipeline: `cargo run --bin compute_static_commitments --release`,
/// then paste the printed table over [`CPU_TIMESTAMP_COMMITMENTS`]. A failing
/// drift test (`static_commitments_tests`) is a reason to ask why the bytes
/// changed, not to regenerate them.
pub fn static_timestamp_commitment(blowup_factor: u8, log_height: u32) -> Option<Commitment> {
    let (_, by_height) = CPU_TIMESTAMP_COMMITMENTS
        .iter()
        .find(|(blowup, _)| *blowup == blowup_factor)?;
    by_height
        .get(log_height.checked_sub(MIN_LOG_HEIGHT)? as usize)
        .copied()
}

/// Smallest CPU table height is `2^2` (`generate_cpu_trace` pads to at least 4).
pub const MIN_LOG_HEIGHT: u32 = 2;

include!("cpu_timestamp_commitments.rs");

// =========================================================================
// Column indices for the CPU table
// =========================================================================

/// Column definitions for the CPU table.
pub mod cols {
    // -------------------------------------------------------------------------
    // Input columns (from DECODE)
    // -------------------------------------------------------------------------

    /// timestamp: Timestamp for memory argument coordination. Preprocessed: row
    /// `i` holds `4·(i + 1)` (spec type `Timestamp`); see [`super::NUM_PRECOMPUTED_COLS`].
    pub const TIMESTAMP: usize = 0;

    /// pc: program counter (DWordWL, 2 words).
    pub const PC_0: usize = 1;
    pub const PC_1: usize = 2;

    /// rs1/rs2/rd: register indices (Byte).
    pub const RS1: usize = 3;
    pub const RS2: usize = 4;
    pub const RD: usize = 5;

    /// read_register1/2, write_register (Bit).
    pub const READ_REGISTER1: usize = 6;
    pub const READ_REGISTER2: usize = 7;
    pub const WRITE_REGISTER: usize = 8;

    /// imm: fully extended immediate (DWordWL, 2 words).
    pub const IMM_0: usize = 9;
    pub const IMM_1: usize = 10;

    /// half_instruction_length: half the bytes consumed (Byte; 1 or 2). The real
    /// length is `2 * half_instruction_length`.
    pub const HALF_INSTRUCTION_LENGTH: usize = 11;
    /// word_instr: `*W` instruction (delegated to CPU32) (Bit).
    pub const WORD_INSTR: usize = 12;

    /// ALU: use the unified ALU for this instruction (Bit).
    pub const ALU: usize = 13;
    /// alu_flags: packed ALU op + flags byte (Byte).
    pub const ALU_FLAGS: usize = 14;
    /// ADD/SUB: arithmetic fast-paths bypassing the ALU (Bit).
    pub const ADD: usize = 15;
    pub const SUB: usize = 16;
    /// MEMORY: touches memory (LOAD/STORE) (Bit).
    pub const MEMORY: usize = 17;
    /// mem_flags: packed memory op + width + signed byte (Byte). Under BRANCH
    /// this column doubles as the virtual `JALR` bit.
    pub const MEM_FLAGS: usize = 18;
    /// BRANCH: conditional branch or jump (Bit).
    pub const BRANCH: usize = 19;
    /// ECALL: environment call (Bit).
    pub const ECALL: usize = 20;

    // -------------------------------------------------------------------------
    // Output columns
    // -------------------------------------------------------------------------

    /// next_pc: program counter for the next instruction (DWordWL, 2 words).
    pub const NEXT_PC_0: usize = 21;
    pub const NEXT_PC_1: usize = 22;

    /// rvd: value to (maybe) write back to rd (DWordWL, 2 words).
    pub const RVD_0: usize = 23;
    pub const RVD_1: usize = 24;

    // -------------------------------------------------------------------------
    // Auxiliary columns
    // -------------------------------------------------------------------------

    /// pc_double_read: PC is read as a general register (`rs1 = 255`) this cycle
    /// (AUIPC/JAL) (Bit).
    pub const PC_DOUBLE_READ: usize = 25;

    /// rv1: value of register rs1 (DWordWL, 2 words).
    pub const RV1_0: usize = 26;
    pub const RV1_1: usize = 27;

    /// rv2: value of register rs2 (DWordWL, 2 words).
    pub const RV2_0: usize = 28;
    pub const RV2_1: usize = 29;

    /// arg2: multiplexed second ALU argument (DWordWL, 2 words).
    pub const ARG2_0: usize = 30;
    pub const ARG2_1: usize = 31;

    /// res: ALU result (DWordHL, 4 halves → 2 words via `cast`).
    pub const RES_0: usize = 32;
    pub const RES_1: usize = 33;
    pub const RES_2: usize = 34;
    pub const RES_3: usize = 35;

    /// branch_cond: whether the branch/jump is taken (Bit).
    pub const BRANCH_COND: usize = 36;

    /// Total number of columns.
    pub const NUM_COLUMNS: usize = 37;

    /// res half columns as an array (DWordHL).
    pub const RES: [usize; 4] = [RES_0, RES_1, RES_2, RES_3];
}

// =========================================================================
// CPU Operation (for trace generation)
// =========================================================================

/// A single CPU cycle to be added to the trace.
///
/// Holds the decoded instruction (`DecodeEntry`) plus the runtime values needed
/// to fill a row: register values, the multiplexed `arg2`, the ALU result, and
/// the branch decision. For `word_instr` rows all operational values are 0 (the
/// row is a pure CPU32 delegate).
#[derive(Debug, Clone, Default)]
pub struct CpuOperation {
    /// Static decode information (shared with the DECODE table).
    pub decode: DecodeEntry,
    /// Timestamp for memory argument coordination.
    pub timestamp: u64,
    /// Next program counter.
    pub next_pc: u64,
    /// Value to write back to rd.
    pub rvd: u64,
    /// Value of register rs1.
    pub rv1: u64,
    /// Value of register rs2.
    pub rv2: u64,
    /// Multiplexed second ALU argument.
    pub arg2: u64,
    /// ALU result (or memory address for LOAD/STORE).
    pub res: u64,
    /// Whether the branch/jump is taken.
    pub branch_cond: bool,

    /// Whether this ECALL is a Commit syscall.
    pub ecall_commit: bool,
    /// For Commit ECALLs: buffer address from x11.
    pub commit_buf_addr: u64,
    /// For Commit ECALLs: byte count from x12.
    pub commit_count: u64,
    /// Whether this ECALL is a KeccakPermute syscall.
    pub ecall_keccak: bool,
    /// For KeccakPermute ECALLs: state address from x10.
    pub keccak_state_addr: u64,

    /// Whether this ECALL is an ECSM (elliptic-curve scalar multiply) syscall
    pub ecall_ecsm: bool,

    /// Whether this ECALL is a non-constraining Hint syscall. The hint operand
    /// addresses (x10/x11/x12) are recovered from the register state in the trace
    /// builder, exactly like ECSM.
    pub ecall_hint: bool,

    /// Whether this ECALL is a DMA memcpy. Operands are recovered from x10/x11/x12.
    pub ecall_dma_memcpy: bool,

    /// Whether this ECALL is a DMA memset. Operands are recovered from x10/x11/x12.
    pub ecall_dma_memset: bool,
}

impl CpuOperation {
    /// Creates a new CPU operation with defaults.
    pub fn new() -> Self {
        Self::default()
    }

    // ------- convenience accessors -------
    #[inline]
    pub fn pc(&self) -> u64 {
        self.decode.pc
    }
    #[inline]
    pub fn imm(&self) -> u64 {
        self.decode.imm
    }
    #[inline]
    pub fn word_instr(&self) -> bool {
        self.decode.fields.word_instr
    }
    /// Virtual `JALR` bit: bit 0 of `mem_flags` (only meaningful under BRANCH).
    #[inline]
    pub fn jalr(&self) -> bool {
        self.decode.fields.mem_flags & 1 == 1
    }

    /// Creates a CpuOperation from an executor Log and a DecodeEntry.
    pub fn from_log(log: &Log, timestamp: u64, decode: DecodeEntry) -> Self {
        let f = decode.fields;
        // Real byte length: the column stores half.
        let instruction_length = 2 * f.half_instruction_length as u64;

        // ECALL syscall classification (rv1 = a7 = syscall number).
        let ecall_commit = f.ecall && log.src1_val == SyscallNumbers::Commit as u64;
        let (commit_buf_addr, commit_count) = if ecall_commit {
            (log.src2_val, log.dst_val)
        } else {
            (0, 0)
        };
        let ecall_keccak =
            f.ecall && log.src1_val == executor::vm::instruction::execution::KECCAK_SYSCALL_NUMBER;
        let keccak_state_addr = if ecall_keccak { log.src2_val } else { 0 };
        // The ECSM operand addresses (x10/x11/x12) are recovered from the register state
        // in the trace builder.
        let ecall_ecsm =
            f.ecall && log.src1_val == executor::vm::instruction::execution::ECSM_SYSCALL_NUMBER;
        let ecall_hint =
            f.ecall && log.src1_val == executor::vm::instruction::execution::HINT_SYSCALL_NUMBER;
        let ecall_dma_memcpy = f.ecall
            && log.src1_val == executor::vm::instruction::execution::DMA_MEMCPY_SYSCALL_NUMBER;
        let ecall_dma_memset = f.ecall
            && log.src1_val == executor::vm::instruction::execution::DMA_MEMSET_SYSCALL_NUMBER;

        // Word instructions are fully handled by CPU32; the main CPU row is a
        // delegate that only advances the PC and sends the CPU32 lookup. We still
        // carry the real register values (rv1/rv2/rvd) so the CPU32 op-generation
        // and its register MEMW accesses can use them — `generate_cpu_trace`
        // zeroes the operational columns on the delegate row.
        if f.word_instr {
            return Self {
                next_pc: decode.pc.wrapping_add(instruction_length),
                rv1: log.src1_val,
                rv2: if f.read_register2 { log.src2_val } else { 0 },
                rvd: log.dst_val,
                ecall_commit,
                commit_buf_addr,
                commit_count,
                ecall_keccak,
                keccak_state_addr,
                decode,
                timestamp,
                ..Default::default()
            };
        }

        // Register values. x255 is the PC register (read by AUIPC/JAL via rs1).
        let rv1 = if f.rs1 == 255 {
            log.current_pc
        } else if f.read_register1 {
            log.src1_val
        } else {
            0
        };
        let rv2 = if f.read_register2 { log.src2_val } else { 0 };

        let jalr = f.mem_flags & 1 == 1;

        // arg2 multiplex (CPU-A1), matching `cpu.toml`:
        //   MEMORY -> imm
        //   BRANCH -> rv2                 (JAL/JALR read no rs2, so rv2 = 0)
        //   else   -> rv2 + imm           (≤1 nonzero by decode A2)
        let arg2 = if f.memory {
            decode.imm
        } else if f.branch {
            rv2
        } else {
            rv2.wrapping_add(decode.imm)
        };

        // Branch decision. JAL/JALR always jump; conditional branches evaluate
        // the EQ/LT comparison (with invert) encoded in `alu_flags`.
        let branch_cond = if f.branch {
            if jalr {
                true
            } else {
                Self::branch_taken(&f, rv1, rv2)
            }
        } else {
            false
        };

        // res = ALU result / address. ADD covers add/load/store/JAL(R); SUB the
        // subtraction fast-path; ALU the comparison (branch) or the chip result.
        let res = if f.add {
            rv1.wrapping_add(arg2)
        } else if f.sub {
            rv1.wrapping_sub(arg2)
        } else if f.alu {
            if f.branch {
                branch_cond as u64
            } else {
                log.dst_val
            }
        } else {
            0
        };

        // rvd: loaded value for LOAD; 0 for STORE (output unused); the return
        // address `pc + instruction_length` on every BRANCH row (written to `rd`
        // only by JAL/JALR — `cpu.toml` branch group); `res`
        // otherwise. The spec computes this `pc + len` via the ADD chip gated on
        // `BRANCH`; we pin it with `emit_branch_rvd_pair` (carry-omitting, like
        // `next_pc`). For conditional branches `rvd` is computed but never
        // written (`write_register = 0`).
        let store = f.memory && jalr; // under MEMORY, mem_flags bit 0 = memory_op (1 = store)
        let rvd = if f.memory {
            if store { 0 } else { log.dst_val }
        } else if f.branch {
            decode.pc.wrapping_add(instruction_length)
        } else {
            res
        };

        // next_pc: branch target for taken branches/jumps; otherwise pc + len.
        // ECALL keeps next_pc = pc + len (CO69) even though the executor sets 0
        // to signal halt; the HALT table proves termination separately.
        let next_pc = if f.ecall {
            decode.pc.wrapping_add(instruction_length)
        } else if branch_cond {
            log.next_pc
        } else {
            decode.pc.wrapping_add(instruction_length)
        };

        Self {
            decode,
            timestamp,
            next_pc,
            rvd,
            rv1,
            rv2,
            arg2,
            res,
            branch_cond,
            ecall_commit,
            commit_buf_addr,
            commit_count,
            ecall_keccak,
            keccak_state_addr,
            ecall_ecsm,
            ecall_hint,
            ecall_dma_memcpy,
            ecall_dma_memset,
        }
    }

    /// Evaluate a conditional-branch comparison `(rv1 ? rv2)` from `alu_flags`.
    /// `alu_flags = alu_op + 32·signed + 64·invert` for branches.
    fn branch_taken(f: &super::types::ShrunkDecode, rv1: u64, rv2: u64) -> bool {
        let op = f.alu_flags & 0x1F;
        let signed = (f.alu_flags >> 5) & 1 == 1;
        let invert = (f.alu_flags >> 6) & 1 == 1;
        let cmp = match op {
            x if x == alu_op::EQ => rv1 == rv2,
            x if x == alu_op::LT => {
                if signed {
                    (rv1 as i64) < (rv2 as i64)
                } else {
                    rv1 < rv2
                }
            }
            _ => false,
        };
        cmp ^ invert
    }

    /// Creates a CpuOperation from Log and Instruction (convenience).
    pub fn from_log_and_instruction(log: &Log, timestamp: u64, instruction: Instruction) -> Self {
        let decode = DecodeEntry::from_instruction(log.current_pc, instruction, 4);
        Self::from_log(log, timestamp, decode)
    }

    /// Collects the BITWISE-table range-check lookups generated by this row, so
    /// the BITWISE table can account for the matching multiplicities:
    /// 3 `ARE_BYTES` (rs1/rs2, rd/half_instruction_length, alu_flags/mem_flags) and
    /// 4 `IS_HALF` (the four halves of `res`).
    pub fn collect_bitwise_ops(&self) -> Vec<super::bitwise::BitwiseOperation> {
        use super::bitwise::{BitwiseOperation, BitwiseOperationType};
        let f = self.decode.fields;
        let mut ops = Vec::with_capacity(7);

        // Must mirror the trace columns exactly. On word delegate rows the CPU
        // zeroes rs1/rs2/rd/alu_flags/mem_flags and res (half_instruction_length stays);
        // CPU32 emits its own range checks for the real decoded values.
        let word = f.word_instr;
        let z = |v: u8| if word { 0 } else { v };
        let res = if word { 0 } else { self.res };

        ops.push(BitwiseOperation::byte_op(
            BitwiseOperationType::AreBytes,
            z(f.rs1),
            z(f.rs2),
        ));
        ops.push(BitwiseOperation::byte_op(
            BitwiseOperationType::AreBytes,
            z(f.rd),
            f.half_instruction_length,
        ));
        ops.push(BitwiseOperation::byte_op(
            BitwiseOperationType::AreBytes,
            z(f.alu_flags),
            z(f.mem_flags),
        ));

        for i in 0..4 {
            let half = ((res >> (i * 16)) & 0xFFFF) as u16;
            ops.push(BitwiseOperation::halfword(
                BitwiseOperationType::IsHalf,
                (half & 0xFF) as u8,
                (half >> 8) as u8,
            ));
        }

        ops
    }
}

// =========================================================================
// Trace generation
// =========================================================================

/// Preprocessed `TIMESTAMP` of `row`: `4·(row + 1)`.
fn local_timestamp(row: usize) -> u64 {
    TIMESTAMP_STRIDE * (row as u64 + 1)
}

/// Generates the CPU trace table from a list of operations.
///
/// Each operation becomes one row; the table is padded to the next power of 2.
pub fn generate_cpu_trace(
    operations: &[CpuOperation],
) -> TraceTable<GoldilocksField, GoldilocksExtension> {
    let n = operations.len();
    let num_rows = n.next_power_of_two().max(4);
    let mut trace = TraceTable::new_main(
        crate::tables::types::zeroed_fe_vec(num_rows * cols::NUM_COLUMNS),
        cols::NUM_COLUMNS,
        1,
    );
    let table = &mut trace.main_table;

    for (row_idx, op) in operations.iter().enumerate() {
        let f = &op.decode.fields;
        let word = f.word_instr;

        // For a word_instr delegate row the operational flags/register I/O are
        // suppressed (CPU32 owns them); only the PC-advancing columns are set.
        let effective = |flag: bool| !word && flag;

        // Preprocessed: the table-local `4·(row + 1)`, whatever `op.timestamp` is.
        // The rest of the trace builder sends `op.timestamp`, which an honest
        // build makes `base + 4·(row + 1)` (see `timestamp_bases`).
        table.set_u64(row_idx, cols::TIMESTAMP, local_timestamp(row_idx));
        table.set_dword_wl(row_idx, cols::PC_0, op.decode.pc);

        // rs1/rs2/rd and read/write flags are only present on non-word rows.
        let (rs1, rs2, rd) = if word {
            (0, 0, 0)
        } else {
            (f.rs1, f.rs2, f.rd)
        };
        table.set_byte(row_idx, cols::RS1, rs1);
        table.set_byte(row_idx, cols::RS2, rs2);
        table.set_byte(row_idx, cols::RD, rd);

        // x0 is hardwired zero (never read/written); x255 is the PC register and
        // must be read (read_register1=1) so its MEMW interaction fires.
        table.set_bool(
            row_idx,
            cols::READ_REGISTER1,
            effective(f.read_register1 && f.rs1 != 0),
        );
        table.set_bool(
            row_idx,
            cols::READ_REGISTER2,
            effective(f.read_register2 && f.rs2 != 0),
        );
        table.set_bool(
            row_idx,
            cols::WRITE_REGISTER,
            effective(f.write_register && f.rd != 0),
        );

        // On word delegate rows, all operational data columns are 0 (CPU32 owns
        // the real values); the register-zero / arg2 / rvd=res constraints all
        // hold with read flags = 0. `op` still carries the real rv1/rv2/rvd for
        // the CPU32 op-generation, so we mask the columns here.
        let (imm, rvd, rv1, rv2, arg2, res) = if word {
            (0, 0, 0, 0, 0, 0)
        } else {
            (op.decode.imm, op.rvd, op.rv1, op.rv2, op.arg2, op.res)
        };

        table.set_dword_wl(row_idx, cols::IMM_0, imm);

        table.set_byte(
            row_idx,
            cols::HALF_INSTRUCTION_LENGTH,
            f.half_instruction_length,
        );
        table.set_bool(row_idx, cols::WORD_INSTR, word);

        table.set_bool(row_idx, cols::ALU, effective(f.alu));
        table.set_byte(row_idx, cols::ALU_FLAGS, if word { 0 } else { f.alu_flags });
        table.set_bool(row_idx, cols::ADD, effective(f.add));
        table.set_bool(row_idx, cols::SUB, effective(f.sub));
        table.set_bool(row_idx, cols::MEMORY, effective(f.memory));
        table.set_byte(row_idx, cols::MEM_FLAGS, if word { 0 } else { f.mem_flags });
        table.set_bool(row_idx, cols::BRANCH, effective(f.branch));
        table.set_bool(row_idx, cols::ECALL, effective(f.ecall));

        table.set_dword_wl(row_idx, cols::NEXT_PC_0, op.next_pc);

        table.set_dword_wl(row_idx, cols::RVD_0, rvd);

        // rv1/rv2/arg2 as DWordWL (2 × 32-bit words).
        table.set_dword_wl(row_idx, cols::RV1_0, rv1);
        table.set_dword_wl(row_idx, cols::RV2_0, rv2);
        table.set_dword_wl(row_idx, cols::ARG2_0, arg2);

        // res as DWordHL (4 × 16-bit halves).
        table.set_dword_hl(row_idx, cols::RES_0, res);

        table.set_bool(row_idx, cols::BRANCH_COND, op.branch_cond);

        // Inline-PC coordination column.
        let pc_double_read = !word && f.read_register1 && f.rs1 == 255;
        table.set_bool(row_idx, cols::PC_DOUBLE_READ, pc_double_read);
    }

    // Padding rows: pc = next_pc = 1 (odd, unreachable), half_instruction_length = 0 so
    // next_pc = pc + 0 = pc, all flags 0. The DECODE table has the matching padding
    // entry at pc = 1. Per spec, padding rows participate in the inline-PC `memory`
    // chain: each reads pc=1 at `timestamp - 3` and writes pc=1 at `timestamp + 1`,
    // continuing the preprocessed +4 cadence from the last real row (the halting
    // ECALL). pc_double_read stays 0, giving prev_ts = timestamp - 3. The first
    // padding read then lands on the halting row's timestamp + 1, where the HALT
    // chip's emit_pc deposited pc = 1.
    for row_idx in n..num_rows {
        table.set_u64(row_idx, cols::TIMESTAMP, local_timestamp(row_idx));
        table.set_u64(row_idx, cols::PC_0, CPU_PADDING_PC);
        table.set_u64(row_idx, cols::NEXT_PC_0, CPU_PADDING_PC);
    }

    trace
}

/// Generates the CPU trace table directly from executor logs.
pub fn generate_cpu_trace_from_logs(
    logs: &[Log],
    instructions: &U64HashMap<Instruction>,
) -> Result<TraceTable<GoldilocksField, GoldilocksExtension>, Error> {
    let mut operations = Vec::with_capacity(logs.len());
    for (i, log) in logs.iter().enumerate() {
        let instruction = *instructions
            .get(&log.current_pc)
            .ok_or(Error::MissingInstruction(log.current_pc))?;
        operations.push(CpuOperation::from_log_and_instruction(
            log,
            (i as u64) * 4 + 4,
            instruction,
        ));
    }
    Ok(generate_cpu_trace(&operations))
}

/// Collects all BITWISE lookups generated by these CPU operations.
pub fn collect_bitwise_ops(operations: &[CpuOperation]) -> Vec<super::bitwise::BitwiseOperation> {
    operations
        .iter()
        .flat_map(|op| op.collect_bitwise_ops())
        .collect()
}

/// Collects all BITWISE lookups from executor logs.
pub fn collect_bitwise_ops_from_logs(
    logs: &[Log],
    instructions: &U64HashMap<Instruction>,
) -> Result<Vec<super::bitwise::BitwiseOperation>, Error> {
    let mut operations = Vec::with_capacity(logs.len());
    for (i, log) in logs.iter().enumerate() {
        let instruction = *instructions
            .get(&log.current_pc)
            .ok_or(Error::MissingInstruction(log.current_pc))?;
        operations.push(CpuOperation::from_log_and_instruction(
            log,
            (i as u64) * 4 + 4,
            instruction,
        ));
    }
    Ok(collect_bitwise_ops(&operations))
}

// =========================================================================
// Bus interactions
// =========================================================================

/// LinearTerm with coefficient 2^bit for a column (packed_decode reconstruction).
fn pow2_term(bit: u32, column: usize) -> LinearTerm {
    LinearTerm::Column {
        coefficient: 1i64 << bit,
        column,
    }
}

/// `BusValue` for the low 32-bit word and high 32-bit word of `res` (DWordHL),
/// i.e. `cast(res, DWordWL)` as 2 bus elements.
fn res_cast_wl() -> BusValue {
    BusValue::Packed {
        start_column: cols::RES_0,
        packing: Packing::DWordHL,
    }
}

/// `timestamp + ts_base + offset` as linear terms.
fn timestamp_terms(ts_base: u64, offset: i64) -> Vec<LinearTerm> {
    let base = i64::try_from(ts_base).expect("CPU timestamp base fits in i64");
    vec![
        LinearTerm::Column {
            coefficient: 1,
            column: cols::TIMESTAMP,
        },
        LinearTerm::Constant(base + offset),
    ]
}

/// `timestamp + ts_base + offset`: a timestamp slot of this row as one bus
/// element (the low word; the high word is 0, CPU timestamps fit in 32 bits).
fn timestamp_slot(ts_base: u64, offset: i64) -> BusValue {
    BusValue::linear(timestamp_terms(ts_base, offset))
}

/// Returns the bus interactions for a CPU table whose timestamps start at
/// `ts_base` (`4·S_k` for table `k`, see [`timestamp_bases`]).
pub fn bus_interactions(ts_base: u64) -> Vec<BusInteraction> {
    use super::types::packed_decode_shrunk as pd;

    let mut interactions = Vec::with_capacity(24);

    // -------------------------------------------------------------------------
    // DECODE: instruction fetch (mult = 1 - word_instr; word rows go to CPU32).
    // -------------------------------------------------------------------------
    interactions.push(BusInteraction::sender(
        BusId::Decode,
        Multiplicity::Negated(cols::WORD_INSTR),
        vec![
            BusValue::Packed {
                start_column: cols::PC_0,
                packing: Packing::DWordWL,
            },
            BusValue::Packed {
                start_column: cols::IMM_0,
                packing: Packing::DWordWL,
            },
            BusValue::linear(vec![
                pow2_term(pd::READ_REG1, cols::READ_REGISTER1),
                pow2_term(pd::READ_REG2, cols::READ_REGISTER2),
                pow2_term(pd::WRITE_REG, cols::WRITE_REGISTER),
                pow2_term(pd::WORD_INSTR, cols::WORD_INSTR),
                pow2_term(pd::ALU, cols::ALU),
                pow2_term(pd::ADD, cols::ADD),
                pow2_term(pd::SUB, cols::SUB),
                pow2_term(pd::MEMORY, cols::MEMORY),
                pow2_term(pd::BRANCH, cols::BRANCH),
                pow2_term(pd::ECALL, cols::ECALL),
                pow2_term(pd::RS1, cols::RS1),
                pow2_term(pd::RS2, cols::RS2),
                pow2_term(pd::RD, cols::RD),
                pow2_term(pd::HALF_INSTRUCTION_LENGTH, cols::HALF_INSTRUCTION_LENGTH),
                pow2_term(pd::ALU_FLAGS, cols::ALU_FLAGS),
                pow2_term(pd::MEM_FLAGS, cols::MEM_FLAGS),
            ]),
        ],
    ));

    // -------------------------------------------------------------------------
    // ALU: unified dispatch ALU[rv1, arg2, alu_flags] -> cast(res, WL).
    // -------------------------------------------------------------------------
    interactions.push(BusInteraction::sender(
        BusId::Alu,
        Multiplicity::Column(cols::ALU),
        vec![
            BusValue::Packed {
                start_column: cols::RV1_0,
                packing: Packing::DWordWL,
            },
            BusValue::Packed {
                start_column: cols::ARG2_0,
                packing: Packing::DWordWL,
            },
            BusValue::Packed {
                start_column: cols::ALU_FLAGS,
                packing: Packing::Direct,
            },
            res_cast_wl(),
        ],
    ));

    // -------------------------------------------------------------------------
    // CPU32: delegate word (`*W`) instructions (mult = word_instr).
    // CPU32[timestamp::DWordWL, pc::DWordWL, half_instruction_length].
    // -------------------------------------------------------------------------
    interactions.push(BusInteraction::sender(
        BusId::Cpu32,
        Multiplicity::Column(cols::WORD_INSTR),
        vec![
            timestamp_slot(ts_base, 0),
            BusValue::constant(0), // timestamp_hi (CPU timestamps fit in 32 bits)
            BusValue::Packed {
                start_column: cols::PC_0,
                packing: Packing::DWordWL,
            },
            BusValue::Packed {
                start_column: cols::HALF_INSTRUCTION_LENGTH,
                packing: Packing::Direct,
            },
        ],
    ));

    // -------------------------------------------------------------------------
    // Register reads/writes via MEMW (24-element read, 16-element write).
    // rv1/rv2/rvd are DWordWL, so the value words are emitted directly.
    // -------------------------------------------------------------------------
    interactions.push(memw_register_read(
        cols::READ_REGISTER1,
        cols::RS1,
        cols::RV1_0,
        cols::RV1_1,
        timestamp_slot(ts_base, 0),
    ));
    interactions.push(memw_register_read(
        cols::READ_REGISTER2,
        cols::RS2,
        cols::RV2_0,
        cols::RV2_1,
        timestamp_slot(ts_base, 1),
    ));
    // Register write of rvd at timestamp+2 (16 elements, no `old`).
    interactions.push(BusInteraction::sender(
        BusId::Memw,
        Multiplicity::Column(cols::WRITE_REGISTER),
        vec![
            BusValue::constant(1), // is_register
            BusValue::linear(vec![LinearTerm::Column {
                coefficient: 2,
                column: cols::RD,
            }]), // base_address[0] = 2*rd
            BusValue::constant(0), // base_address[1]
            BusValue::Packed {
                start_column: cols::RVD_0,
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::RVD_1,
                packing: Packing::Direct,
            },
            BusValue::constant(0),
            BusValue::constant(0),
            BusValue::constant(0),
            BusValue::constant(0),
            BusValue::constant(0),
            BusValue::constant(0),
            timestamp_slot(ts_base, 2),
            BusValue::constant(0),
            BusValue::constant(1), // write2 (register access = 2 words)
            BusValue::constant(0),
            BusValue::constant(0),
        ],
    ));

    // -------------------------------------------------------------------------
    // MEMORY: high-level LOAD/STORE dispatch (mult = MEMORY).
    // MEMORY[timestamp, cast(res, WL) = address, rv2, mem_flags] -> rvd.
    // -------------------------------------------------------------------------
    interactions.push(BusInteraction::sender(
        BusId::MemoryOp,
        Multiplicity::Column(cols::MEMORY),
        vec![
            timestamp_slot(ts_base, 0),
            BusValue::constant(0), // timestamp_hi
            res_cast_wl(),         // address (2 words)
            BusValue::Packed {
                start_column: cols::RV2_0,
                packing: Packing::DWordWL,
            }, // value to store (2 words)
            BusValue::Packed {
                start_column: cols::MEM_FLAGS,
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::RVD_0,
                packing: Packing::DWordWL,
            }, // loaded value (output)
        ],
    ));

    // -------------------------------------------------------------------------
    // Inline PC memory tokens (mult = 1, per spec): read PC at the coordinated
    // previous timestamp, write next_pc at timestamp+1. x255 lives at addresses
    // 510/511. Padding rows participate too (they carry PC=1 and chain their
    // timestamps); the HALT chip's consume_pc/emit_pc bridges the last real write
    // to the padding chain.
    //
    // The timestamp is preprocessed, so `timestamp - 3` never underflows (the
    // first row of an epoch reads the REGISTER init token at 1) and
    // `pc_double_read` needs no constraint beyond IS_BIT (spec `cpu.typ`): with
    // it set, the read lands on this row's own `rs1 = x255` read at `timestamp`,
    // and nothing else writes x255 there.
    // -------------------------------------------------------------------------
    let pc_mult = Multiplicity::One;
    // prev_ts = ts_base + timestamp - 3*(1 - pc_double_read)
    let mut prev_ts_terms = timestamp_terms(ts_base, -3);
    prev_ts_terms.push(LinearTerm::Column {
        coefficient: 3,
        column: cols::PC_DOUBLE_READ,
    });
    let prev_ts = BusValue::linear(prev_ts_terms);
    for i in 0..2u64 {
        let pc_col = if i == 0 { cols::PC_0 } else { cols::PC_1 };
        let next_pc_col = if i == 0 {
            cols::NEXT_PC_0
        } else {
            cols::NEXT_PC_1
        };
        // PC read (sender): consume the existing token.
        interactions.push(BusInteraction::sender(
            BusId::Memory,
            pc_mult.clone(),
            vec![
                BusValue::constant(1),
                BusValue::constant(510 + i),
                BusValue::constant(0),
                prev_ts.clone(),
                BusValue::constant(0),
                BusValue::Packed {
                    start_column: pc_col,
                    packing: Packing::Direct,
                },
            ],
        ));
        // PC write (receiver): emit the next token at timestamp+1.
        interactions.push(BusInteraction::receiver(
            BusId::Memory,
            pc_mult.clone(),
            vec![
                BusValue::constant(1),
                BusValue::constant(510 + i),
                BusValue::constant(0),
                timestamp_slot(ts_base, 1),
                BusValue::constant(0),
                BusValue::Packed {
                    start_column: next_pc_col,
                    packing: Packing::Direct,
                },
            ],
        ));
    }

    // -------------------------------------------------------------------------
    // BRANCH: target computation (mult = branch_cond).
    // BRANCH[pc, imm, rv1, JALR] -> next_pc. JALR ≡ mem_flags under BRANCH.
    // Order matches the BRANCH table receiver: [next_pc, pc, imm, register, JALR].
    // -------------------------------------------------------------------------
    interactions.push(BusInteraction::sender(
        BusId::Branch,
        Multiplicity::Column(cols::BRANCH_COND),
        vec![
            BusValue::Packed {
                start_column: cols::NEXT_PC_0,
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::NEXT_PC_1,
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::PC_0,
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::PC_1,
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::IMM_0,
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::IMM_1,
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::RV1_0,
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::RV1_1,
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::MEM_FLAGS,
                packing: Packing::Direct,
            }, // JALR
        ],
    ));

    // -------------------------------------------------------------------------
    // Range checks: ARE_BYTES (rs1/rs2, rd/half_instruction_length, alu_flags/mem_flags)
    // and IS_HALF on each `res` half. Every row sends (incl. padding: all 0).
    // -------------------------------------------------------------------------
    for (a, b) in [
        (cols::RS1, cols::RS2),
        (cols::RD, cols::HALF_INSTRUCTION_LENGTH),
        (cols::ALU_FLAGS, cols::MEM_FLAGS),
    ] {
        interactions.push(BusInteraction::sender(
            BusId::AreBytes,
            Multiplicity::One,
            vec![
                BusValue::Packed {
                    start_column: a,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: b,
                    packing: Packing::Direct,
                },
            ],
        ));
    }
    for &res_col in &cols::RES {
        interactions.push(BusInteraction::sender(
            BusId::IsHalfword,
            Multiplicity::One,
            vec![BusValue::Packed {
                start_column: res_col,
                packing: Packing::Direct,
            }],
        ));
    }

    // -------------------------------------------------------------------------
    // ECALL: system-call bus (HALT/COPY/KECCAK/ECSM/HINT receive). mult = ECALL.
    // ECALL[timestamp, rv1].
    // -------------------------------------------------------------------------
    interactions.push(BusInteraction::sender(
        BusId::Ecall,
        Multiplicity::Column(cols::ECALL),
        vec![
            timestamp_slot(ts_base, 0),
            BusValue::constant(0),
            BusValue::Packed {
                start_column: cols::RV1_0,
                packing: Packing::DWordWL,
            },
        ],
    ));

    interactions
}

/// MEMW register-read interaction (24 elements: `old(8), is_register, base(2),
/// value(8), timestamp(2), w2, w4, w8`). Register values are DWordWL (the two
/// value words are read directly; the remaining 6 byte slots are 0).
fn memw_register_read(
    read_flag_col: usize,
    rs_col: usize,
    rv_lo_col: usize,
    rv_hi_col: usize,
    ts: BusValue,
) -> BusInteraction {
    let value_lo = || BusValue::Packed {
        start_column: rv_lo_col,
        packing: Packing::Direct,
    };
    let value_hi = || BusValue::Packed {
        start_column: rv_hi_col,
        packing: Packing::Direct,
    };
    BusInteraction::sender(
        BusId::Memw,
        Multiplicity::Column(read_flag_col),
        vec![
            // old[0..8] = rv (2 words) + 6 zeros
            value_lo(),
            value_hi(),
            BusValue::constant(0),
            BusValue::constant(0),
            BusValue::constant(0),
            BusValue::constant(0),
            BusValue::constant(0),
            BusValue::constant(0),
            // is_register = 1
            BusValue::constant(1),
            // base_address[0] = 2*rs, base_address[1] = 0
            BusValue::linear(vec![LinearTerm::Column {
                coefficient: 2,
                column: rs_col,
            }]),
            BusValue::constant(0),
            // value[0..8] = rv (2 words) + 6 zeros
            value_lo(),
            value_hi(),
            BusValue::constant(0),
            BusValue::constant(0),
            BusValue::constant(0),
            BusValue::constant(0),
            BusValue::constant(0),
            BusValue::constant(0),
            // timestamp[0..2]
            ts,
            BusValue::constant(0),
            // write2 = 1, write4 = 0, write8 = 0 (register = 2 words)
            BusValue::constant(1),
            BusValue::constant(0),
            BusValue::constant(0),
        ],
    )
}
