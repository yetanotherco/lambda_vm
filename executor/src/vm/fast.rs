//! The fast engine: the same machine as [`Executor`](super::execution::Executor),
//! interpreted tightly.
//!
//! - Every instruction of the program's executable segments is decoded once
//!   (by [`Instruction::parse`], the only decoder) into an 8-byte [`Op`], found
//!   by `(pc - base) / 4`; one flat `match` on its [`Kind`] runs it.
//! - The registers are a `[u64; 32]` with `x0` put back to zero after every
//!   instruction; the memory is a [`FlatMemory`].
//! - The records are the executor's: one [`Log`] per cycle, written into the
//!   window buffer [`FastExecutor::resume_owned`] hands out.
//!
//! One semantics, two dispatchers: every arm computes with the functions the
//! reference executor calls ([`ArithOp::apply`], [`ArithOp::apply_word`],
//! [`Comparison::apply`], the load and store widths, the jump targets), each
//! with a constant operation, and every instruction it does not run itself
//! (ecalls, CSR, FENCE, a `pc` outside the decoded segments, misaligned `pc`)
//! goes through the reference's own cycle ([`step`]) on the same memory. A
//! load or store that fails is run again through that cycle too, so the error
//! is the reference's own (a failing access changes nothing before it fails).
use super::{
    execution::{ExecutorError, InstructionCache, ReturnValues, load_program, step},
    flat_memory::{FlatLayout, FlatMemory},
    history::{
        HISTORY_MAX_CYCLES, HistExtra, History, HistoryFinal, NO_ACCESS, RegUse, RegisterCells,
        WindowHistory, cycle_timestamp, step_registers,
    },
    instruction::execution::{BLAKE3_SYSCALL_NUMBER, KECCAK_SYSCALL_NUMBER},
    instruction::{
        decoding::{ArithOp, Comparison, Instruction, LoadStoreWidth},
        execution::{
            auipc_value, effective_address, jalr_target, load_value, lui_value, pc_relative,
            store_value,
        },
    },
    logs::Log,
    memory::{GuestMemory, MemorySnapshot},
    registers::{Registers, STACK_TOP},
};
use crate::elf::Elf;

/// The most logs [`FastExecutor::resume_owned`] reserves for its next call,
/// as [`Executor::resume_owned`](super::execution::Executor::resume_owned).
const MAX_RESERVED_LOGS: usize = 1 << 22;

/// The most ops the decoded table spans (64 MiB of ops); a program whose
/// executable segments span more is decoded over its largest segment only,
/// the rest running through the reference cycle.
const MAX_OPS: u64 = 1 << 23;

/// What an [`Op`] does. The arms of [`FastExecutor::run`] are one per kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
enum Kind {
    // Register-register, 64-bit.
    Add,
    Sub,
    Xor,
    Or,
    And,
    Sll,
    Srl,
    Sra,
    Slt,
    Sltu,
    Mul,
    Mulh,
    Mulhsu,
    Mulhu,
    Div,
    Divu,
    Rem,
    Remu,
    // Register-immediate, 64-bit (the ones the decoder produces).
    Addi,
    Xori,
    Ori,
    Andi,
    Slli,
    Srli,
    Srai,
    Slti,
    Sltiu,
    // Register-register, 32-bit.
    Addw,
    Subw,
    Sllw,
    Srlw,
    Sraw,
    Mulw,
    Divw,
    Divuw,
    Remw,
    Remuw,
    // Register-immediate, 32-bit.
    Addiw,
    Slliw,
    Srliw,
    Sraiw,
    // Loads and stores.
    Lb,
    Lh,
    Lw,
    Ld,
    Lbu,
    Lhu,
    Lwu,
    Sb,
    Sh,
    Sw,
    Sd,
    // Control flow.
    Beq,
    Bne,
    Blt,
    Bge,
    Bltu,
    Bgeu,
    Jal,
    Jalr,
    Lui,
    Auipc,
    /// Through the reference cycle: ecalls, CSR, FENCE, anything else.
    Step,
}

/// One decoded instruction: its kind, registers and immediate (the offset of
/// a load, store, branch or jump; the raw 32-bit value of LUI and AUIPC).
#[derive(Clone, Copy, Debug)]
#[repr(C)]
struct Op {
    kind: Kind,
    rd: u8,
    rs1: u8,
    rs2: u8,
    imm: i32,
}

const _: () = assert!(std::mem::size_of::<Op>() == 8);

const STEP: Op = Op {
    kind: Kind::Step,
    rd: 0,
    rs1: 0,
    rs2: 0,
    imm: 0,
};

impl Op {
    fn new(kind: Kind, rd: u32, rs1: u32, rs2: u32, imm: i32) -> Self {
        // The decoder's register fields are five bits.
        let reg = |r: u32| (r & 31) as u8;
        Self {
            kind,
            rd: reg(rd),
            rs1: reg(rs1),
            rs2: reg(rs2),
            imm,
        }
    }

    /// The op an instruction runs as.
    fn encode(instruction: &Instruction) -> Self {
        use ArithOp as A;
        match *instruction {
            Instruction::Arith {
                dst,
                src1,
                src2,
                op,
            } => {
                let kind = match op {
                    A::Add => Kind::Add,
                    A::Sub => Kind::Sub,
                    A::Xor => Kind::Xor,
                    A::Or => Kind::Or,
                    A::And => Kind::And,
                    A::ShiftLeftLogical => Kind::Sll,
                    A::ShiftRightLogical => Kind::Srl,
                    A::ShiftRightArith => Kind::Sra,
                    A::SetLessThan => Kind::Slt,
                    A::SetLessThanU => Kind::Sltu,
                    A::Mul => Kind::Mul,
                    A::MulHigh => Kind::Mulh,
                    A::MulHighSignedUnsigned => Kind::Mulhsu,
                    A::MulHighUnsigned => Kind::Mulhu,
                    A::Div => Kind::Div,
                    A::DivUnsigned => Kind::Divu,
                    A::Remainder => Kind::Rem,
                    A::RemainderUnsigned => Kind::Remu,
                };
                Self::new(kind, dst, src1, src2, 0)
            }
            Instruction::ArithImm { dst, src, imm, op } => {
                let kind = match op {
                    A::Add => Kind::Addi,
                    A::Xor => Kind::Xori,
                    A::Or => Kind::Ori,
                    A::And => Kind::Andi,
                    A::ShiftLeftLogical => Kind::Slli,
                    A::ShiftRightLogical => Kind::Srli,
                    A::ShiftRightArith => Kind::Srai,
                    A::SetLessThan => Kind::Slti,
                    A::SetLessThanU => Kind::Sltiu,
                    _ => return STEP,
                };
                Self::new(kind, dst, src, 0, imm)
            }
            Instruction::ArithW {
                dst,
                src1,
                src2,
                op,
            } => {
                let kind = match op {
                    A::Add => Kind::Addw,
                    A::Sub => Kind::Subw,
                    A::ShiftLeftLogical => Kind::Sllw,
                    A::ShiftRightLogical => Kind::Srlw,
                    A::ShiftRightArith => Kind::Sraw,
                    A::Mul => Kind::Mulw,
                    A::Div => Kind::Divw,
                    A::DivUnsigned => Kind::Divuw,
                    A::Remainder => Kind::Remw,
                    A::RemainderUnsigned => Kind::Remuw,
                    _ => return STEP,
                };
                Self::new(kind, dst, src1, src2, 0)
            }
            Instruction::ArithImmW { dst, src, imm, op } => {
                let kind = match op {
                    A::Add => Kind::Addiw,
                    A::ShiftLeftLogical => Kind::Slliw,
                    A::ShiftRightLogical => Kind::Srliw,
                    A::ShiftRightArith => Kind::Sraiw,
                    _ => return STEP,
                };
                Self::new(kind, dst, src, 0, imm)
            }
            Instruction::Load {
                dst,
                offset,
                base,
                width,
            } => {
                let kind = match width {
                    LoadStoreWidth::Byte => Kind::Lb,
                    LoadStoreWidth::Half => Kind::Lh,
                    LoadStoreWidth::Word => Kind::Lw,
                    LoadStoreWidth::DoubleWord => Kind::Ld,
                    LoadStoreWidth::ByteUnsigned => Kind::Lbu,
                    LoadStoreWidth::HalfUnsigned => Kind::Lhu,
                    LoadStoreWidth::WordUnsigned => Kind::Lwu,
                };
                Self::new(kind, dst, base, 0, offset)
            }
            Instruction::Store {
                src,
                offset,
                base,
                width,
            } => {
                let kind = match width {
                    LoadStoreWidth::Byte => Kind::Sb,
                    LoadStoreWidth::Half => Kind::Sh,
                    LoadStoreWidth::Word => Kind::Sw,
                    LoadStoreWidth::DoubleWord => Kind::Sd,
                    _ => return STEP,
                };
                Self::new(kind, 0, base, src, offset)
            }
            Instruction::Branch {
                src1,
                src2,
                cond,
                offset,
            } => {
                let kind = match cond {
                    Comparison::Equal => Kind::Beq,
                    Comparison::NotEqual => Kind::Bne,
                    Comparison::LessThan => Kind::Blt,
                    Comparison::GreaterOrEqual => Kind::Bge,
                    Comparison::LessThanUnsigned => Kind::Bltu,
                    Comparison::GreaterOrEqualUnsigned => Kind::Bgeu,
                };
                Self::new(kind, 0, src1, src2, offset)
            }
            Instruction::JumpAndLink { dst, offset } => Self::new(Kind::Jal, dst, 0, 0, offset),
            Instruction::JumpAndLinkRegister { base, dst, offset } => {
                Self::new(Kind::Jalr, dst, base, 0, offset)
            }
            Instruction::LoadUpperImm { dst, imm } => Self::new(Kind::Lui, dst, 0, 0, imm as i32),
            Instruction::AddUpperImmToPc { dst, imm } => {
                Self::new(Kind::Auipc, dst, 0, 0, imm as i32)
            }
            Instruction::CSR { .. } | Instruction::EcallEbreak | Instruction::Fence => STEP,
        }
    }
}

/// The ops of a program: `ops[i]` runs the instruction at `base + 4·i`.
fn decode_ops(instructions: &InstructionCache) -> (u64, Box<[Op]>) {
    let segments: Vec<(u64, &[Instruction])> = instructions.segments().collect();
    let Some(start) = segments.iter().map(|(base, _)| *base).min() else {
        return (0, Box::new([]));
    };
    let end = segments
        .iter()
        .map(|(base, ins)| base.saturating_add(4 * ins.len() as u64))
        .max()
        .unwrap_or(start);
    let (start, end) = if (end - start) / 4 <= MAX_OPS {
        (start, end)
    } else {
        // Only the largest segment; the rest through the reference cycle.
        let (base, ins) = segments
            .iter()
            .max_by_key(|(_, ins)| ins.len())
            .copied()
            .unwrap_or((start, &[]));
        (base, base + 4 * (ins.len() as u64).min(MAX_OPS))
    };
    let mut ops = vec![STEP; ((end - start) / 4) as usize];
    for (base, ins) in segments {
        for (i, instruction) in ins.iter().enumerate() {
            let pc = base + 4 * i as u64;
            if (start..end).contains(&pc) && (pc - start).is_multiple_of(4) {
                ops[((pc - start) / 4) as usize] = Op::encode(instruction);
            }
        }
    }
    (start, ops.into_boxed_slice())
}

/// The fast engine's state: the reference executor's, held for speed.
pub struct FastExecutor {
    ops: Box<[Op]>,
    ops_base: u64,
    regs: [u64; 32],
    pc: u64,
    memory: FlatMemory,
    /// The decoded segments, for the reference cycle.
    instructions: InstructionCache,
    logs: Vec<Log>,
    /// A handed-out window's buffer given back ([`FastExecutor::recycle`]).
    spare: Option<Vec<Log>>,
    /// The cycles run so far.
    cycles: u64,
    /// History mode ([`FastExecutor::enable_history`]): the history, and
    /// each op's register and memory use (`None` where no instruction is
    /// decoded: found as the reference cycle finds it).
    history: Option<Box<History>>,
    uses: Box<[Option<RegUse>]>,
}

impl FastExecutor {
    /// A run of `program` on `private_inputs`, set up as
    /// [`Executor::new`](super::execution::Executor::new) sets one up, with
    /// the guest's flat layout.
    pub fn new(program: &Elf, private_inputs: Vec<u8>) -> Result<Self, ExecutorError> {
        Self::with_layout(program, private_inputs, FlatLayout::GUEST)
    }

    /// [`FastExecutor::new`] on a given flat layout (a test puts the regions'
    /// edges where its accesses are).
    pub fn with_layout(
        program: &Elf,
        private_inputs: Vec<u8>,
        layout: FlatLayout,
    ) -> Result<Self, ExecutorError> {
        let mut memory = FlatMemory::new(layout)?;
        // In the reference's order: the input, the decode, the program.
        GuestMemory::store_private_inputs(&mut memory, private_inputs)?;
        let instructions = InstructionCache::new(&program.data)?;
        load_program(&program.data, &mut memory)?;
        let (ops_base, ops) = decode_ops(&instructions);
        let mut regs = [0u64; 32];
        regs[2] = STACK_TOP;
        debug_assert_eq!(Registers::default().to_array(), regs);
        Ok(Self {
            ops,
            ops_base,
            regs,
            pc: program.entry_point,
            memory,
            instructions,
            logs: Vec::with_capacity(super::execution::CHUNK_SIZE),
            spare: None,
            cycles: 0,
            history: None,
            uses: Box::new([]),
        })
    }

    /// Turns history mode on: from now on the engine keeps every byte's and
    /// register's last-access timestamp by the walk's rules and writes the
    /// history beside the logs ([`FastExecutor::resume_history`]). Before the
    /// first cycle only.
    pub fn enable_history(&mut self) -> Result<(), ExecutorError> {
        if self.cycles > 0 || self.history.is_some() {
            return Err(ExecutorError::History(
                "history mode starts before the first cycle".into(),
            ));
        }
        self.history = Some(Box::new(History::new(self.memory.layout(), self.pc)?));
        self.uses = (0..self.ops.len() as u64)
            .map(|i| self.instructions.get(self.ops_base + 4 * i).map(RegUse::of))
            .collect();
        Ok(())
    }

    /// [`FastExecutor::resume_owned`] in history mode: the window's logs and
    /// its [`WindowHistory`].
    pub fn resume_history(
        &mut self,
        limit: usize,
    ) -> Result<Option<(Vec<Log>, WindowHistory)>, ExecutorError> {
        let window = limit.min(MAX_RESERVED_LOGS);
        if self.pc == 0 {
            return Ok(None);
        }
        if self.history.is_none() {
            return Err(ExecutorError::History("history mode is off".into()));
        }
        if self.logs.capacity() < window {
            self.logs.clear();
            self.logs.reserve_exact(window);
        }
        self.logs.clear();
        let first = self.cycles;
        if let Some(history) = &mut self.history {
            history.begin(first, window);
        }
        self.run_history(limit)?;
        if self.logs.is_empty() {
            return Ok(None);
        }
        let next = super::logs::next_buffer(&mut self.spare, window);
        let mut logs = std::mem::replace(&mut self.logs, next);
        logs.shrink_to_fit();
        let history = match &mut self.history {
            Some(h) => h.take_window(),
            None => WindowHistory::default(),
        };
        Ok(Some((logs, history)))
    }

    /// History mode's end: the memory and every byte's timestamp after the
    /// last cycle (history mode ends with it).
    pub fn take_history_final(&mut self) -> Option<HistoryFinal> {
        let history = self.history.take()?;
        Some(HistoryFinal {
            memory: self.memory.snapshot(),
            shadow: history.shadow,
            registers: history.cells,
        })
    }

    /// Sets the cycles run so far (a test reaches the history's clock limit
    /// without running to it).
    #[cfg(test)]
    pub(crate) fn set_cycles(&mut self, cycles: u64) {
        self.cycles = cycles;
    }

    /// The register cells now, as the walk holds them (history mode).
    pub fn history_registers(&self) -> Option<RegisterCells> {
        self.history.as_ref().map(|h| h.cells)
    }

    /// [`FastExecutor::run`] keeping the history.
    #[inline(never)]
    fn run_history(&mut self, limit: usize) -> Result<(), ExecutorError> {
        let Some(history) = self.history.as_deref_mut() else {
            return Err(ExecutorError::History("history mode is off".into()));
        };
        let ops = &*self.ops;
        let uses = &*self.uses;
        let base = self.ops_base;
        let regs = &mut self.regs;
        let memory = &mut self.memory;
        let instructions = &self.instructions;
        let logs = &mut self.logs;
        let mut pc = self.pc;
        let mut cycle = self.cycles;
        let result = loop {
            if pc == 0 || RecordSink::cycles(logs) >= limit {
                break Ok(());
            }
            if cycle > HISTORY_MAX_CYCLES {
                break Err(ExecutorError::History(format!(
                    "cycle {cycle} is past the {HISTORY_MAX_CYCLES} a history's timestamps hold"
                )));
            }
            let ts = cycle_timestamp(cycle);
            let index = pc.wrapping_sub(base).rotate_right(2);
            let (op, uses, address) = if index < ops.len() as u64
                && let Some(u) = uses[index as usize]
            {
                let op = ops[index as usize];
                let address = effective_address(regs[op.rs1 as usize & 31], op.imm);
                (op, u, address)
            } else {
                let (u, address) = fetched_use(pc, regs, memory, instructions);
                (STEP, u, address)
            };
            // Only an ECALL needs the cells before it (a 0.5 KiB copy; an
            // eager copy every cycle cost ≈ 0.9 s on the median block).
            let cells_before = if uses.ecall {
                Some(history.cells)
            } else {
                None
            };
            let extra = if uses.mem > 0 {
                history.before_access(memory, address)
            } else if uses.ecall {
                let a7 = regs[17];
                let provisional = Log {
                    current_pc: pc,
                    next_pc: 0,
                    src1_val: a7,
                    src2_val: match a7 {
                        64 => regs[11],
                        n if n == KECCAK_SYSCALL_NUMBER || n == BLAKE3_SYSCALL_NUMBER => regs[10],
                        _ => 0,
                    },
                    dst_val: if a7 == 64 { regs[12] } else { 0 },
                };
                history.before_ecall(memory, &provisional, ts)
            } else {
                HistExtra {
                    side: NO_ACCESS,
                    ..HistExtra::default()
                }
            };
            let log = match fast_cycle(op, pc, regs, memory) {
                Some(log) => log,
                None => match reference_cycle(pc, regs, memory, instructions) {
                    Ok(log) => log,
                    Err(e) => break Err(e),
                },
            };
            if uses.mem > 0 {
                history.after_access(address, uses.mem, ts);
            }
            if let Some(before) = &cells_before {
                history.after_ecall(&log, before, ts);
            }
            let mut extra = extra;
            extra.regs_ts = step_registers(&mut history.cells, uses, &log, ts);
            extra.reg_side = if uses.ecall {
                history.ecall_registers(&log, ts)
            } else {
                NO_ACCESS
            };
            history.window.extras.push(extra);
            pc = log.next_pc;
            logs.record(log);
            cycle += 1;
        };
        self.pc = pc;
        self.cycles = cycle;
        result
    }

    /// Current program counter (0 once the program has halted).
    pub fn pc(&self) -> u64 {
        self.pc
    }

    /// Current register state.
    pub fn registers(&self) -> Registers {
        Registers::from_array(&self.regs)
    }

    /// Current memory.
    pub fn memory(&self) -> &FlatMemory {
        &self.memory
    }

    /// The memory as it is now ([`FlatMemory::snapshot`]).
    pub fn snapshot(&mut self) -> MemorySnapshot {
        self.memory.snapshot()
    }

    /// [`Executor::resume_with_limit`](super::execution::Executor::resume_with_limit):
    /// at most `limit` cycles, their logs lent; `None` once the program has
    /// finished.
    pub fn resume_with_limit(&mut self, limit: usize) -> Result<Option<&[Log]>, ExecutorError> {
        if self.pc == 0 {
            return Ok(None);
        }
        self.logs.clear();
        self.run(limit)?;
        if self.logs.is_empty() {
            Ok(None)
        } else {
            Ok(Some(&self.logs))
        }
    }

    /// [`Executor::resume_owned`](super::execution::Executor::resume_owned):
    /// the window's logs handed out as they lie, a new buffer reserved for
    /// the next call.
    pub fn resume_owned(&mut self, limit: usize) -> Result<Option<Vec<Log>>, ExecutorError> {
        let window = limit.min(MAX_RESERVED_LOGS);
        if self.logs.capacity() < window {
            self.logs.clear();
            self.logs.reserve_exact(window);
        }
        if self.resume_with_limit(limit)?.is_none() {
            return Ok(None);
        }
        let next = super::logs::next_buffer(&mut self.spare, window);
        let mut logs = std::mem::replace(&mut self.logs, next);
        logs.shrink_to_fit();
        Ok(Some(logs))
    }

    /// [`Executor::recycle`](super::execution::Executor::recycle): the next
    /// window written into a buffer given back.
    pub fn recycle(&mut self, buffer: Vec<Log>) {
        self.spare = Some(buffer);
    }

    /// A buffer for a later window's history extras (history mode): used
    /// when it holds exactly one window, as [`FastExecutor::recycle`] for the
    /// logs.
    pub fn recycle_extras(&mut self, buffer: Vec<HistExtra>) {
        if let Some(history) = &mut self.history {
            history.spare_extras = Some(buffer);
        }
    }

    /// The return values, as [`Executor::finish`](super::execution::Executor::finish).
    pub fn finish(self) -> Result<ReturnValues, ExecutorError> {
        Ok(ReturnValues {
            memory_values: self.memory.public_output_bytes().to_vec(),
            register_values: (self.regs[10] as i64, self.regs[11] as i64),
        })
    }

    /// Runs until the program halts or `limit` logs are written, appending a
    /// log per cycle to `self.logs`.
    #[inline(never)]
    fn run(&mut self, limit: usize) -> Result<(), ExecutorError> {
        let ops = &*self.ops;
        let base = self.ops_base;
        let regs = &mut self.regs;
        let memory = &mut self.memory;
        let instructions = &self.instructions;
        let logs = &mut self.logs;
        let before = RecordSink::cycles(logs);
        let mut pc = self.pc;
        let result = loop {
            if pc == 0 || RecordSink::cycles(logs) >= limit {
                break Ok(());
            }
            // A misaligned `pc` rotates its low bits to the top: out of range,
            // to the reference cycle, which refuses it.
            let index = pc.wrapping_sub(base).rotate_right(2);
            let op = if index < ops.len() as u64 {
                ops[index as usize]
            } else {
                STEP
            };
            match fast_cycle(op, pc, regs, memory) {
                Some(log) => {
                    pc = log.next_pc;
                    logs.record(log);
                }
                None => match reference_cycle(pc, regs, memory, instructions) {
                    Ok(log) => {
                        pc = log.next_pc;
                        logs.record(log);
                    }
                    Err(e) => break Err(e),
                },
            }
        };
        self.cycles += (RecordSink::cycles(logs) - before) as u64;
        self.pc = pc;
        result
    }
}

/// Where the run's per-cycle records go: the one place the loop writes a
/// record, apart from the dispatch arms (which only compute the cycle's
/// [`Log`]). Today a window buffer of 40-byte logs; a wider record (the log
/// plus what the cycle overwrote) would be another sink fed from here.
trait RecordSink {
    /// Records one cycle.
    fn record(&mut self, log: Log);
    /// The cycles recorded so far in this window.
    fn cycles(&self) -> usize;
}

impl RecordSink for Vec<Log> {
    #[inline(always)]
    fn record(&mut self, log: Log) {
        self.push(log);
    }

    #[inline(always)]
    fn cycles(&self) -> usize {
        self.len()
    }
}

/// The register and memory use of the instruction at `pc` outside the
/// decoded ops, found as the reference cycle finds it, and a load's or
/// store's address; the default (no use) where the cycle will fail.
#[cold]
#[inline(never)]
fn fetched_use(
    pc: u64,
    regs: &[u64; 32],
    memory: &FlatMemory,
    instructions: &InstructionCache,
) -> (RegUse, u64) {
    if !pc.is_multiple_of(4) {
        return (RegUse::default(), 0);
    }
    let instruction = match instructions.get(pc) {
        Some(&i) => i,
        None => match GuestMemory::load_word(memory, pc)
            .ok()
            .and_then(|w| Instruction::parse(w).ok())
        {
            Some(i) => i,
            None => return (RegUse::default(), 0),
        },
    };
    let address = match instruction {
        Instruction::Load { base, offset, .. } | Instruction::Store { base, offset, .. } => {
            effective_address(regs[base as usize & 31], offset)
        }
        _ => 0,
    };
    (RegUse::of(&instruction), address)
}

/// One cycle through the reference's [`step`], on these registers and
/// memory.
#[cold]
#[inline(never)]
fn reference_cycle(
    pc: u64,
    regs: &mut [u64; 32],
    memory: &mut FlatMemory,
    instructions: &InstructionCache,
) -> Result<Log, ExecutorError> {
    let mut registers = Registers::from_array(regs);
    let mut next = pc;
    let log = step(&mut next, &mut registers, memory, instructions)?;
    *regs = registers.to_array();
    Ok(log)
}

/// One cycle of `op` at `pc`, or `None` for the reference cycle to run it
/// (an op of kind [`Kind::Step`], or an access that fails).
#[inline(always)]
fn fast_cycle(op: Op, pc: u64, regs: &mut [u64; 32], memory: &mut FlatMemory) -> Option<Log> {
    let rd = op.rd as usize & 31;
    let a = regs[op.rs1 as usize & 31];
    let b = regs[op.rs2 as usize & 31];
    let imm = op.imm;
    let mut next = pc.wrapping_add(4);
    let (s1, s2, d);

    macro_rules! reg {
        ($op:ident) => {{
            d = ArithOp::$op.apply(a as i64, b as i64) as u64;
            (s1, s2) = (a, b);
        }};
    }
    macro_rules! imm {
        ($op:ident) => {{
            d = ArithOp::$op.apply(a as i64, imm as i64) as u64;
            (s1, s2) = (a, 0);
        }};
    }
    macro_rules! regw {
        ($op:ident) => {{
            d = ArithOp::$op.apply_word(a as i32, b as i32).ok()? as i64 as u64;
            (s1, s2) = (a, b);
        }};
    }
    macro_rules! immw {
        ($op:ident) => {{
            d = ArithOp::$op.apply_word(a as i32, imm).ok()? as i64 as u64;
            (s1, s2) = (a, 0);
        }};
    }
    macro_rules! load {
        ($width:ident) => {{
            d = load_value(memory, LoadStoreWidth::$width, effective_address(a, imm)).ok()?;
            (s1, s2) = (a, 0);
        }};
    }
    macro_rules! store {
        ($width:ident) => {{
            store_value(memory, LoadStoreWidth::$width, effective_address(a, imm), b).ok()?;
            (s1, s2, d) = (a, b, 0);
        }};
    }
    macro_rules! branch {
        ($cond:ident) => {{
            if Comparison::$cond.apply(a, b) {
                next = pc_relative(pc, imm);
            }
            (s1, s2, d) = (a, b, 0);
        }};
    }

    match op.kind {
        Kind::Add => reg!(Add),
        Kind::Sub => reg!(Sub),
        Kind::Xor => reg!(Xor),
        Kind::Or => reg!(Or),
        Kind::And => reg!(And),
        Kind::Sll => reg!(ShiftLeftLogical),
        Kind::Srl => reg!(ShiftRightLogical),
        Kind::Sra => reg!(ShiftRightArith),
        Kind::Slt => reg!(SetLessThan),
        Kind::Sltu => reg!(SetLessThanU),
        Kind::Mul => reg!(Mul),
        Kind::Mulh => reg!(MulHigh),
        Kind::Mulhsu => reg!(MulHighSignedUnsigned),
        Kind::Mulhu => reg!(MulHighUnsigned),
        Kind::Div => reg!(Div),
        Kind::Divu => reg!(DivUnsigned),
        Kind::Rem => reg!(Remainder),
        Kind::Remu => reg!(RemainderUnsigned),
        Kind::Addi => imm!(Add),
        Kind::Xori => imm!(Xor),
        Kind::Ori => imm!(Or),
        Kind::Andi => imm!(And),
        Kind::Slli => imm!(ShiftLeftLogical),
        Kind::Srli => imm!(ShiftRightLogical),
        Kind::Srai => imm!(ShiftRightArith),
        Kind::Slti => imm!(SetLessThan),
        Kind::Sltiu => imm!(SetLessThanU),
        Kind::Addw => regw!(Add),
        Kind::Subw => regw!(Sub),
        Kind::Sllw => regw!(ShiftLeftLogical),
        Kind::Srlw => regw!(ShiftRightLogical),
        Kind::Sraw => regw!(ShiftRightArith),
        Kind::Mulw => regw!(Mul),
        Kind::Divw => regw!(Div),
        Kind::Divuw => regw!(DivUnsigned),
        Kind::Remw => regw!(Remainder),
        Kind::Remuw => regw!(RemainderUnsigned),
        Kind::Addiw => immw!(Add),
        Kind::Slliw => immw!(ShiftLeftLogical),
        Kind::Srliw => immw!(ShiftRightLogical),
        Kind::Sraiw => immw!(ShiftRightArith),
        Kind::Lb => load!(Byte),
        Kind::Lh => load!(Half),
        Kind::Lw => load!(Word),
        Kind::Ld => load!(DoubleWord),
        Kind::Lbu => load!(ByteUnsigned),
        Kind::Lhu => load!(HalfUnsigned),
        Kind::Lwu => load!(WordUnsigned),
        Kind::Sb => store!(Byte),
        Kind::Sh => store!(Half),
        Kind::Sw => store!(Word),
        Kind::Sd => store!(DoubleWord),
        Kind::Beq => branch!(Equal),
        Kind::Bne => branch!(NotEqual),
        Kind::Blt => branch!(LessThan),
        Kind::Bge => branch!(GreaterOrEqual),
        Kind::Bltu => branch!(LessThanUnsigned),
        Kind::Bgeu => branch!(GreaterOrEqualUnsigned),
        Kind::Jal => {
            d = pc.wrapping_add(4);
            next = pc_relative(pc, imm);
            (s1, s2) = (0, 0);
        }
        Kind::Jalr => {
            d = pc.wrapping_add(4);
            next = jalr_target(a, imm);
            (s1, s2) = (a, 0);
        }
        Kind::Lui => {
            d = lui_value(imm as u32);
            (s1, s2) = (0, 0);
        }
        Kind::Auipc => {
            d = auipc_value(pc, imm as u32);
            (s1, s2) = (0, 0);
        }
        Kind::Step => return None,
    }
    // Stores and branches write no register: their `rd` is 0, and `x0`
    // takes the write and is put back to zero.
    regs[rd] = d;
    regs[0] = 0;
    Some(Log {
        current_pc: pc,
        next_pc: next,
        src1_val: s1,
        src2_val: s2,
        dst_val: d,
    })
}
