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
        HISTORY_MAX_CYCLES, HistExtra, History, HistoryFinal, Keeper, LeanWords, NO_ACCESS, RegUse,
        RegisterCells, WORD_BYTES, WindowHistory, WordHistory, WordHistoryFinal, cycle_timestamp,
        step_registers,
    },
    instruction::execution::{BLAKE3_SYSCALL_NUMBER, KECCAK_SYSCALL_NUMBER},
    instruction::{
        decoding::{ArithOp, Comparison, Instruction, LoadStoreWidth},
        execution::{
            auipc_value, ecall_log, effective_address, jalr_target, load_value, lui_value,
            pc_relative, store_value,
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

/// The capacity of the log buffer a window of `limit` cycles is written into
/// ([`FastExecutor::resume_owned`], [`FastExecutor::resume_history`],
/// [`FastExecutor::resume_word_history`]): a buffer given back through
/// [`FastExecutor::recycle`] is taken for the next window only when it has
/// exactly this capacity.
pub fn log_window_capacity(limit: usize) -> usize {
    limit.min(MAX_RESERVED_LOGS)
}

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
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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
    /// The word history ([`FastExecutor::enable_word_history`]): history
    /// mode with a timestamp a word; never on beside `history`.
    word_history: Option<Box<WordHistory>>,
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
            word_history: None,
            uses: Box::new([]),
        })
    }

    /// Turns history mode on: from now on the engine keeps every byte's and
    /// register's last-access timestamp by the walk's rules and writes the
    /// history beside the logs ([`FastExecutor::resume_history`]). Before the
    /// first cycle only.
    pub fn enable_history(&mut self) -> Result<(), ExecutorError> {
        self.refuse_second_history()?;
        self.history = Some(Box::new(History::new(self.memory.layout(), self.pc)?));
        self.decode_uses();
        Ok(())
    }

    /// Turns the word history on: history mode with a timestamp an aligned
    /// 8-byte word instead of a byte ([`WordHistory`]), for a run whose memory
    /// is checked by word. Its windows come from
    /// [`FastExecutor::resume_word_history`]. Before the first cycle only, and
    /// never beside [`FastExecutor::enable_history`].
    pub fn enable_word_history(&mut self) -> Result<(), ExecutorError> {
        self.refuse_second_history()?;
        self.word_history = Some(Box::new(WordHistory::new(self.memory.layout(), self.pc)?));
        self.decode_uses();
        Ok(())
    }

    /// Refuses a history turned on after the first cycle or beside another.
    fn refuse_second_history(&self) -> Result<(), ExecutorError> {
        if self.cycles > 0 || self.history.is_some() || self.word_history.is_some() {
            return Err(ExecutorError::History(
                "history mode starts before the first cycle, once".into(),
            ));
        }
        Ok(())
    }

    /// Each op's register and memory use, for history mode.
    fn decode_uses(&mut self) {
        self.uses = (0..self.ops.len() as u64)
            .map(|i| self.instructions.get(self.ops_base + 4 * i).map(RegUse::of))
            .collect();
    }

    /// [`FastExecutor::resume_owned`] in history mode: the window's logs and
    /// its [`WindowHistory`].
    pub fn resume_history(
        &mut self,
        limit: usize,
    ) -> Result<Option<(Vec<Log>, WindowHistory)>, ExecutorError> {
        let window = log_window_capacity(limit);
        if self.pc == 0 {
            return Ok(None);
        }
        if self.history.is_none() {
            return Err(ExecutorError::History("history mode is off".into()));
        }
        self.prepare_logs(window);
        let first = self.cycles;
        if let Some(history) = &mut self.history {
            history.begin(first, window);
        }
        self.run_history(limit)?;
        if self.logs.is_empty() {
            return Ok(None);
        }
        let logs = self.hand_out_logs(window);
        let history = match &mut self.history {
            Some(h) => h.take_window(),
            None => WindowHistory::default(),
        };
        Ok(Some((logs, history)))
    }

    /// [`FastExecutor::resume_history`] under the word history: the window's
    /// logs and its [`WindowHistory`], whose memory cells are whole words
    /// ([`WindowHistory::words`]).
    pub fn resume_word_history(
        &mut self,
        limit: usize,
    ) -> Result<Option<(Vec<Log>, WindowHistory)>, ExecutorError> {
        let window = log_window_capacity(limit);
        if self.pc == 0 {
            return Ok(None);
        }
        if self.word_history.is_none() {
            return Err(ExecutorError::History("the word history is off".into()));
        }
        self.prepare_logs(window);
        let first = self.cycles;
        if let Some(history) = &mut self.word_history {
            history.begin(first, window);
        }
        self.run_word_history(limit)?;
        if self.logs.is_empty() {
            return Ok(None);
        }
        let logs = self.hand_out_logs(window);
        let history = match &mut self.word_history {
            Some(h) => h.take_window(),
            None => WindowHistory::default(),
        };
        Ok(Some((logs, history)))
    }

    /// Turns the lean stream on, over the word history: from now on a window
    /// is taken with [`FastExecutor::resume_word_lean`], which writes no logs
    /// and no per-cycle record, only the access records, the ECALLs' cells
    /// and places, and the window's starting cells; a walker re-runs the
    /// window from them ([`Replay`]). Before the first cycle, after
    /// [`FastExecutor::enable_word_history`].
    pub fn enable_lean_stream(&mut self) -> Result<(), ExecutorError> {
        match &mut self.word_history {
            Some(history) if self.cycles == 0 => {
                history.lean = true;
                Ok(())
            }
            Some(_) => Err(ExecutorError::History(
                "the lean stream starts before the first cycle".into(),
            )),
            None => Err(ExecutorError::History(
                "the lean stream needs the word history".into(),
            )),
        }
    }

    /// [`FastExecutor::resume_word_history`] under the lean stream: the
    /// window's cycle count and its [`WindowHistory`] (with
    /// [`WindowHistory::lean`] set), no logs. A cycle whose instruction the
    /// engine would fetch from memory (outside the program's decoded
    /// segments, which the prover's DECODE table also is) is refused.
    pub fn resume_word_lean(
        &mut self,
        limit: usize,
    ) -> Result<Option<(usize, WindowHistory)>, ExecutorError> {
        let window = log_window_capacity(limit);
        if self.pc == 0 {
            return Ok(None);
        }
        match &mut self.word_history {
            Some(history) if history.lean => history.begin(self.cycles, window),
            _ => return Err(ExecutorError::History("the lean stream is off".into())),
        }
        let first = self.cycles;
        self.run_word_lean(limit)?;
        let cycles = (self.cycles - first) as usize;
        if cycles == 0 {
            return Ok(None);
        }
        let mut history = match &mut self.word_history {
            Some(h) => h.take_window(),
            None => WindowHistory::default(),
        };
        history.cycles = cycles;
        Ok(Some((cycles, history)))
    }

    /// The log buffer cleared, with room for a `window` of logs.
    fn prepare_logs(&mut self, window: usize) {
        if self.logs.capacity() < window {
            self.logs.clear();
            self.logs.reserve_exact(window);
        }
        self.logs.clear();
    }

    /// The window's logs handed out, a buffer for the next one in place.
    fn hand_out_logs(&mut self, window: usize) -> Vec<Log> {
        let next = super::logs::next_buffer(&mut self.spare, window);
        let mut logs = std::mem::replace(&mut self.logs, next);
        logs.shrink_to_fit();
        logs
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

    /// The word history's end: the memory and every word's timestamp after
    /// the last cycle (the word history ends with it).
    pub fn take_word_history_final(&mut self) -> Option<WordHistoryFinal> {
        let history = self.word_history.take()?;
        Some(WordHistoryFinal {
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

    /// The byte history's timestamps now (a test compares the word history
    /// with them).
    #[cfg(test)]
    pub(crate) fn byte_stamps(&self) -> Option<&super::history::TsShadow> {
        self.history.as_ref().map(|h| &h.shadow)
    }

    /// The word history's timestamps now.
    #[cfg(test)]
    pub(crate) fn word_stamps(&self) -> Option<&super::history::WordShadow> {
        self.word_history.as_ref().map(|h| &h.shadow)
    }

    /// The register cells now, as the walk holds them (history mode, either
    /// history).
    pub fn history_registers(&self) -> Option<RegisterCells> {
        match (&self.history, &self.word_history) {
            (Some(h), _) => Some(h.cells),
            (None, Some(h)) => Some(h.cells),
            (None, None) => None,
        }
    }

    /// [`FastExecutor::run`] keeping the history.
    #[inline(never)]
    fn run_history(&mut self, limit: usize) -> Result<(), ExecutorError> {
        let Some(history) = self.history.as_deref_mut() else {
            return Err(ExecutorError::History("history mode is off".into()));
        };
        let (pc, cycle, result) = run_kept(
            history,
            &self.ops,
            &self.uses,
            self.ops_base,
            &mut self.regs,
            &mut self.memory,
            &self.instructions,
            &mut self.logs,
            (self.pc, self.cycles),
            limit,
        );
        self.pc = pc;
        self.cycles = cycle;
        result
    }

    /// [`FastExecutor::run`] keeping the word history.
    #[inline(never)]
    fn run_word_history(&mut self, limit: usize) -> Result<(), ExecutorError> {
        let Some(history) = self.word_history.as_deref_mut() else {
            return Err(ExecutorError::History("the word history is off".into()));
        };
        let (pc, cycle, result) = run_kept(
            history,
            &self.ops,
            &self.uses,
            self.ops_base,
            &mut self.regs,
            &mut self.memory,
            &self.instructions,
            &mut self.logs,
            (self.pc, self.cycles),
            limit,
        );
        self.pc = pc;
        self.cycles = cycle;
        result
    }

    /// [`FastExecutor::run_word_history`] under the lean stream: the cycles
    /// counted, no log written.
    #[inline(never)]
    fn run_word_lean(&mut self, limit: usize) -> Result<(), ExecutorError> {
        let Some(history) = self.word_history.as_deref_mut() else {
            return Err(ExecutorError::History("the word history is off".into()));
        };
        let mut count = CycleCount(0);
        let (pc, cycle, result) = run_kept(
            &mut LeanWords(history),
            &self.ops,
            &self.uses,
            self.ops_base,
            &mut self.regs,
            &mut self.memory,
            &self.instructions,
            &mut count,
            (self.pc, self.cycles),
            limit,
        );
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
        let window = log_window_capacity(limit);
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
        } else if let Some(history) = &mut self.word_history {
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

/// The lean stream's sink: the cycles counted, no log kept.
struct CycleCount(usize);

impl RecordSink for CycleCount {
    #[inline(always)]
    fn record(&mut self, _log: Log) {
        self.0 += 1;
    }

    #[inline(always)]
    fn cycles(&self) -> usize {
        self.0
    }
}

/// History mode's run loop over either history ([`Keeper`]): runs from
/// `(pc, cycle)` until the program halts or `limit` logs are written, and
/// returns where it stopped. Inlined into each history's own entry
/// ([`FastExecutor::run_history`], [`FastExecutor::run_word_history`]), so
/// each is compiled for its history alone.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn run_kept<K: Keeper, S: RecordSink>(
    history: &mut K,
    ops: &[Op],
    uses: &[Option<RegUse>],
    base: u64,
    regs: &mut [u64; 32],
    memory: &mut FlatMemory,
    instructions: &InstructionCache,
    logs: &mut S,
    (mut pc, mut cycle): (u64, u64),
    limit: usize,
) -> (u64, u64, Result<(), ExecutorError>) {
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
            // The lean stream's replay has no memory to fetch an instruction
            // from; the prover's DECODE table holds no such instruction either.
            if K::LEAN && pc.is_multiple_of(4) && instructions.get(pc).is_none() {
                break Err(ExecutorError::History(format!(
                    "lean stream: the instruction at {pc:#x} is outside the program's decoded \
                     segments"
                )));
            }
            let (u, address) = fetched_use(pc, regs, memory, instructions);
            (STEP, u, address)
        };
        // Only an ECALL needs the cells before it (a 0.5 KiB copy; an
        // eager copy every cycle cost ≈ 0.9 s on the median block).
        let cells_before = if uses.ecall {
            Some(*history.cells())
        } else {
            None
        };
        let side = if uses.mem > 0 {
            history.before_access(memory, address, uses.mem)
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
            NO_ACCESS
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
        let regs_ts = step_registers(history.cells(), uses, &log, ts);
        if uses.ecall {
            history.ecall_registers(&log, ts);
        }
        history.push_extra(HistExtra { regs_ts, side });
        pc = log.next_pc;
        logs.record(log);
        cycle += 1;
    };
    (pc, cycle, result)
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
fn reference_cycle<M: GuestMemory>(
    pc: u64,
    regs: &mut [u64; 32],
    memory: &mut M,
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
fn fast_cycle<M: GuestMemory>(
    op: Op,
    pc: u64,
    regs: &mut [u64; 32],
    memory: &mut M,
) -> Option<Log> {
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

/// A walker's re-run of a lean window ([`FastExecutor::resume_word_lean`]):
/// the engine's own cycle ([`fast_cycle`], the reference's for anything
/// else) on registers alone, each load answered from the window's access
/// record and each ECALL's log made from the registers ([`ecall_log`]). The
/// logs it writes are the ones the engine wrote, byte for byte: the same
/// function of the same registers and the same loaded words. The op table is
/// its own, decoded from the program as the engine decodes its.
pub struct Replay {
    ops: Box<[Op]>,
    ops_base: u64,
    uses: Box<[Option<RegUse>]>,
    instructions: InstructionCache,
}

impl std::fmt::Debug for Replay {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Replay")
            .field("ops", &self.ops.len())
            .field("ops_base", &self.ops_base)
            .finish()
    }
}

/// Where a replay of a window stands: the pc and registers after the cycles
/// re-run so far, and how many of the window's cycles, access records and
/// ECALLs they took.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReplayCursor {
    pub pc: u64,
    pub regs: [u64; 32],
    /// The cycles re-run, from the window's first.
    pub cycle: usize,
    /// The access records ([`WindowHistory::word_accesses`]) taken.
    pub access: usize,
    /// The ECALLs ([`WindowHistory::groups`]) taken.
    pub ecall: usize,
}

/// A replay that cannot go on: the window's records are not the ones its
/// cycles make (or the window is not a lean one).
fn replay_error(why: String) -> ExecutorError {
    ExecutorError::History(format!("replay: {why}"))
}

impl Replay {
    /// The replay of `program`'s windows.
    pub fn new(program: &Elf) -> Result<Self, ExecutorError> {
        let instructions = InstructionCache::new(&program.data)?;
        let (ops_base, ops) = decode_ops(&instructions);
        let uses = (0..ops.len() as u64)
            .map(|i| instructions.get(ops_base + 4 * i).map(RegUse::of))
            .collect();
        Ok(Self {
            ops,
            ops_base,
            uses,
            instructions,
        })
    }

    /// Whether its op table (ops, base, register uses) is `engine`'s, after
    /// `engine`'s history is on: the replay decodes its own from the program
    /// with the engine's [`decode_ops`], and a test pins the two equal.
    #[cfg(test)]
    pub(crate) fn same_table(&self, engine: &FastExecutor) -> bool {
        self.ops_base == engine.ops_base && self.ops == engine.ops && self.uses == engine.uses
    }

    /// The cursor at the start of the lean window `history`: its starting
    /// pc and register values.
    pub fn start(history: &WindowHistory) -> Result<ReplayCursor, ExecutorError> {
        if !(history.lean && history.words) {
            return Err(replay_error("the window is not a lean one".into()));
        }
        let Some(cells) = &history.registers else {
            return Err(replay_error("the window has no starting cells".into()));
        };
        Ok(ReplayCursor {
            pc: cells.pc.0,
            regs: std::array::from_fn(|r| if r == 0 { 0 } else { cells.regs[r].0 }),
            cycle: 0,
            access: 0,
            ecall: 0,
        })
    }

    /// The next `n` cycles of the lean window `history` from `at`, their
    /// logs appended to `out`. Refuses a cycle past the window's end, a load
    /// or store whose record is missing or names another address, a load
    /// that reads a byte its record does not hold, and an ECALL that is not
    /// the window's next at its cycle.
    pub fn run(
        &self,
        history: &WindowHistory,
        at: &mut ReplayCursor,
        n: usize,
        out: &mut Vec<Log>,
    ) -> Result<(), ExecutorError> {
        if at.cycle + n > history.cycles {
            return Err(replay_error(format!(
                "{} cycles asked from cycle {} of a window of {}",
                n, at.cycle, history.cycles
            )));
        }
        out.reserve(n);
        for _ in 0..n {
            let log = self.cycle(history, at)?;
            at.pc = log.next_pc;
            at.cycle += 1;
            out.push(log);
        }
        Ok(())
    }

    /// The window's end: every cycle, access record and ECALL taken.
    pub fn finish(history: &WindowHistory, at: &ReplayCursor) -> Result<(), ExecutorError> {
        if at.cycle != history.cycles
            || at.access != history.word_accesses.len()
            || at.ecall != history.groups.len()
        {
            return Err(replay_error(format!(
                "the window ends after {} of {} cycles, {} of {} access records, {} of {} ECALLs",
                at.cycle,
                history.cycles,
                at.access,
                history.word_accesses.len(),
                at.ecall,
                history.groups.len()
            )));
        }
        Ok(())
    }

    /// One cycle, as [`run_kept`] runs it.
    #[inline(always)]
    fn cycle(&self, history: &WindowHistory, at: &mut ReplayCursor) -> Result<Log, ExecutorError> {
        let pc = at.pc;
        let index = pc.wrapping_sub(self.ops_base).rotate_right(2);
        let (op, uses, address) = if index < self.ops.len() as u64
            && let Some(u) = self.uses[index as usize]
        {
            let op = self.ops[index as usize];
            let address = effective_address(at.regs[op.rs1 as usize & 31], op.imm);
            (op, u, address)
        } else {
            match self.instructions.get(pc) {
                Some(instruction) if pc.is_multiple_of(4) => {
                    let address = match *instruction {
                        Instruction::Load { base, offset, .. }
                        | Instruction::Store { base, offset, .. } => {
                            effective_address(at.regs[base as usize & 31], offset)
                        }
                        _ => 0,
                    };
                    (STEP, RegUse::of(instruction), address)
                }
                _ => {
                    return Err(replay_error(format!(
                        "cycle {}: no decoded instruction at {pc:#x}",
                        at.cycle
                    )));
                }
            }
        };
        if uses.ecall {
            return self.ecall(history, at, pc);
        }
        let mut memory = ReplayMemory::default();
        if uses.mem > 0 {
            let Some(record) = history.word_accesses.get(at.access) else {
                return Err(replay_error(format!(
                    "cycle {}: a load or store past the window's {} access records",
                    at.cycle,
                    history.word_accesses.len()
                )));
            };
            if record.addr != address {
                return Err(replay_error(format!(
                    "cycle {}: the access at {address:#x} finds the record of {:#x}",
                    at.cycle, record.addr
                )));
            }
            let next = match record.next {
                NO_ACCESS => None,
                k => match history.next_words.get(k as usize) {
                    Some(cell) => Some(cell.word),
                    None => {
                        return Err(replay_error(format!(
                            "cycle {}: its second word is missing",
                            at.cycle
                        )));
                    }
                },
            };
            memory = ReplayMemory {
                base: record.addr & !(WORD_BYTES - 1),
                word: record.word,
                next,
                held: if next.is_some() { 16 } else { 8 },
                fault: std::cell::Cell::new(false),
            };
            at.access += 1;
        }
        let log = match fast_cycle(op, pc, &mut at.regs, &mut memory) {
            Some(log) => log,
            None => reference_cycle(pc, &mut at.regs, &mut memory, &self.instructions)?,
        };
        if memory.fault.get() {
            return Err(replay_error(format!(
                "cycle {}: a byte outside its access record",
                at.cycle
            )));
        }
        Ok(log)
    }

    /// An ECALL's cycle: the window's next ECALL must be at this cycle, after
    /// as many access records as taken; its log from the registers, the
    /// commit's `x10` written.
    #[cold]
    fn ecall(
        &self,
        history: &WindowHistory,
        at: &mut ReplayCursor,
        pc: u64,
    ) -> Result<Log, ExecutorError> {
        let k = at.ecall;
        let placed = history.ecall_at.get(k).map(|&c| c as usize) == Some(at.cycle)
            && history.accesses_before.get(k).map(|&a| a as usize) == Some(at.access)
            && k < history.groups.len();
        if !placed {
            return Err(replay_error(format!(
                "cycle {}: an ECALL the window does not place here (its ECALL {k})",
                at.cycle
            )));
        }
        let Some((log, x10)) = ecall_log(pc, &at.regs) else {
            return Err(replay_error(format!(
                "cycle {}: ECALL {:#x} has no log on these registers",
                at.cycle, at.regs[17]
            )));
        };
        if let Some(value) = x10 {
            at.regs[10] = value;
        }
        at.ecall += 1;
        Ok(log)
    }
}

/// A replayed cycle's memory: the words of its access record (the word
/// holding its first byte and, when it crosses, the next), read only. A write
/// is dropped (a later load reads its own record); a read of any other byte
/// is marked a fault and reads 0. The default holds no byte (a cycle with
/// no access record).
#[derive(Default)]
struct ReplayMemory {
    base: u64,
    word: u64,
    next: Option<u64>,
    /// The bytes held from `base`: 0, 8 or 16.
    held: u64,
    fault: std::cell::Cell<bool>,
}

impl GuestMemory for ReplayMemory {
    #[inline(always)]
    fn read_n<const N: usize>(&self, address: u64) -> [u8; N] {
        let mut out = [0u8; N];
        let off = address.wrapping_sub(self.base);
        let held = self.held;
        if N as u64 > held || off > held - N as u64 {
            self.fault.set(true);
            return out;
        }
        let both = (u128::from(self.next.unwrap_or(0)) << 64) | u128::from(self.word);
        let bytes = (both >> (8 * off)).to_le_bytes();
        out.copy_from_slice(&bytes[..N]);
        out
    }

    #[inline(always)]
    fn write_n<const N: usize>(&mut self, _address: u64, _data: [u8; N]) {}

    fn read_range(&self, _address: u64, out: &mut [u8]) {
        if !out.is_empty() {
            self.fault.set(true);
        }
    }

    fn write_range(&mut self, _address: u64, _data: &[u8]) {}

    fn public_output(&self) -> &[u8] {
        &[]
    }

    fn extend_public_output(&mut self, _bytes: &[u8]) {}

    fn dma_bulk(&self) -> bool {
        false
    }
}
