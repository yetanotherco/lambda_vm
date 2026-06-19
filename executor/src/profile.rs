//! Dynamic instruction-class profiling for guest RISC-V programs.
//!
//! Bins executed instructions by class (and ECALLs by syscall) to show what
//! the guest spends its cycles on. This is an *exact dynamic count* over the
//! execution logs — it is not a proving-cost estimate. For the trace-side
//! breakdown that drives proving cost, see the prover's per-table report
//! (`lambda_vm_prover::table_report`).

use std::collections::BTreeMap;
use std::io::{self, Write};

use crate::vm::execution::InstructionCache;
use crate::vm::instruction::decoding::{ArithOp, Instruction};
use crate::vm::instruction::execution::KECCAK_SYSCALL_NUMBER;
use crate::vm::logs::Log;

/// A coarse instruction class, chosen to line up with how the prover groups
/// work into chips/tables (ALU mul vs div vs shift vs compare, memory loads vs
/// stores, control flow, and the individual syscalls).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum InstrClass {
    /// ADD/SUB/AND/OR/XOR (reg-reg or reg-imm), incl. their `*W` forms and
    /// LUI/AUIPC — the cheap "add path" CPU rows.
    AluBasic,
    /// SLT/SLTU and their immediate forms (LT chip).
    Compare,
    /// SLL/SRL/SRA and their immediate / `*W` forms (SHIFT chip).
    Shift,
    /// MUL/MULH/MULHU/MULHSU (MUL chip).
    Mul,
    /// DIV/DIVU/REM/REMU (DVRM chip).
    DivRem,
    /// Memory loads (LOAD + MEMW chips).
    Load,
    /// Memory stores (STORE + MEMW chips).
    Store,
    /// Conditional branches (BRANCH + EQ/LT chips).
    Branch,
    /// JAL/JALR (jumps and calls/returns).
    Jump,
    /// FENCE / CSR (treated as no-ops by the VM).
    Fence,
    /// ECALL: keccak permute syscall.
    EcallKeccak,
    /// ECALL: elliptic-curve scalar-multiply syscall.
    EcallEcsm,
    /// ECALL: commit (public output) syscall.
    EcallCommit,
    /// ECALL: halt syscall.
    EcallHalt,
    /// ECALL: any other syscall (print, panic, unknown).
    EcallOther,
}

impl InstrClass {
    /// Stable human-readable label for reports.
    pub fn label(self) -> &'static str {
        match self {
            InstrClass::AluBasic => "alu (add/sub/bitwise)",
            InstrClass::Compare => "compare (slt)",
            InstrClass::Shift => "shift",
            InstrClass::Mul => "mul",
            InstrClass::DivRem => "div/rem",
            InstrClass::Load => "load",
            InstrClass::Store => "store",
            InstrClass::Branch => "branch",
            InstrClass::Jump => "jump (jal/jalr)",
            InstrClass::Fence => "fence/csr",
            InstrClass::EcallKeccak => "ecall:keccak",
            InstrClass::EcallEcsm => "ecall:ecsm",
            InstrClass::EcallCommit => "ecall:commit",
            InstrClass::EcallHalt => "ecall:halt",
            InstrClass::EcallOther => "ecall:other",
        }
    }
}

/// Map an `ArithOp` to a class. Shared by reg-reg and reg-imm (and `*W`) forms
/// because the chip selection depends only on the operation.
fn arith_class(op: ArithOp) -> InstrClass {
    match op {
        ArithOp::Add | ArithOp::Sub | ArithOp::Xor | ArithOp::Or | ArithOp::And => {
            InstrClass::AluBasic
        }
        ArithOp::SetLessThan | ArithOp::SetLessThanU => InstrClass::Compare,
        ArithOp::ShiftLeftLogical | ArithOp::ShiftRightLogical | ArithOp::ShiftRightArith => {
            InstrClass::Shift
        }
        ArithOp::Mul
        | ArithOp::MulHigh
        | ArithOp::MulHighSignedUnsigned
        | ArithOp::MulHighUnsigned => InstrClass::Mul,
        ArithOp::Div | ArithOp::DivUnsigned | ArithOp::Remainder | ArithOp::RemainderUnsigned => {
            InstrClass::DivRem
        }
    }
}

/// Classify a single executed instruction. For ECALLs the class is refined by
/// the syscall number, which `Log` records in `src1_val` (the guest's x17).
fn classify(instruction: Instruction, log: &Log) -> InstrClass {
    match instruction {
        Instruction::Arith { op, .. }
        | Instruction::ArithImm { op, .. }
        | Instruction::ArithW { op, .. }
        | Instruction::ArithImmW { op, .. } => arith_class(op),
        Instruction::LoadUpperImm { .. } | Instruction::AddUpperImmToPc { .. } => {
            InstrClass::AluBasic
        }
        Instruction::Load { .. } => InstrClass::Load,
        Instruction::Store { .. } => InstrClass::Store,
        Instruction::Branch { .. } => InstrClass::Branch,
        Instruction::JumpAndLink { .. } | Instruction::JumpAndLinkRegister { .. } => {
            InstrClass::Jump
        }
        Instruction::Fence | Instruction::CSR { .. } => InstrClass::Fence,
        // This branch's executor has no ECSM syscall (it predates that work),
        // so `EcallEcsm` is never produced here — an ECSM ecall, if present,
        // would fall through to `EcallOther`.
        Instruction::EcallEbreak => match log.src1_val {
            v if v == KECCAK_SYSCALL_NUMBER => InstrClass::EcallKeccak,
            64 => InstrClass::EcallCommit,
            93 => InstrClass::EcallHalt,
            _ => InstrClass::EcallOther,
        },
    }
}

/// Accumulates a dynamic instruction-class histogram across execution logs.
#[derive(Default)]
pub struct InstrHistogram {
    counts: BTreeMap<InstrClass, u64>,
    total: u64,
}

/// Errors that can occur while profiling logs.
#[derive(Debug)]
pub enum ProfileError {
    /// Instruction not found for a given program counter.
    InstructionNotFound,
}

impl InstrHistogram {
    pub fn new() -> Self {
        Self::default()
    }

    /// Process a batch of execution logs, accumulating per-class counts.
    pub fn process_logs(
        &mut self,
        logs: &[Log],
        instructions: &InstructionCache,
    ) -> Result<(), ProfileError> {
        for log in logs {
            let instruction = instructions
                .get(log.current_pc)
                .copied()
                .ok_or(ProfileError::InstructionNotFound)?;
            let class = classify(instruction, log);
            *self.counts.entry(class).or_insert(0) += 1;
            self.total += 1;
        }
        Ok(())
    }

    /// Total instructions counted.
    pub fn total(&self) -> u64 {
        self.total
    }

    /// Class counts sorted by descending count (ties broken by class order).
    pub fn sorted(&self) -> Vec<(InstrClass, u64)> {
        let mut v: Vec<_> = self.counts.iter().map(|(&c, &n)| (c, n)).collect();
        v.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        v
    }

    /// Write a human-readable histogram to `writer`, sorted by count, with a
    /// percentage-of-total column.
    pub fn write_report<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        writeln!(writer, "=== INSTRUCTION CLASS HISTOGRAM ===")?;
        writeln!(writer, "  {:<24} {:>14} {:>7}", "Class", "Count", "%")?;
        writeln!(writer, "  {}", "-".repeat(48))?;
        for (class, count) in self.sorted() {
            let pct = if self.total > 0 {
                count as f64 / self.total as f64 * 100.0
            } else {
                0.0
            };
            writeln!(
                writer,
                "  {:<24} {:>14} {:>6.2}%",
                class.label(),
                count,
                pct
            )?;
        }
        writeln!(writer, "  {}", "-".repeat(48))?;
        writeln!(writer, "  {:<24} {:>14}", "TOTAL", self.total)?;
        Ok(())
    }
}
