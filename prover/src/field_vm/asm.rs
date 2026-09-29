//! Assembler for the Field VM, with the pseudoinstructions of the spec's
//! "Instruction notation" table and its calling convention.

use super::isa::{Arg, ENTRY_PC, Instr, Program, REG_PC, REG_ZERO, gpr_index};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Label(usize);

#[derive(Clone, Copy, Debug)]
enum Fixup {
    /// `FMA PC == target`.
    Jump,
    /// `FMA PC == ZERO * (-PC + (target - 1)) + (PC + 1)`.
    JumpZero,
    /// `FMA PC == ZERO * (PC - target) + (ZERO + target)`.
    JumpNotZero,
}

#[derive(Default)]
pub struct Asm {
    body: Vec<Instr>,
    labels: Vec<Option<u64>>,
    fixups: Vec<(usize, Label, Fixup)>,
}

#[derive(Debug, PartialEq, Eq)]
pub enum AsmError {
    UnboundLabel(Label),
    Invalid { pc: usize, reason: &'static str },
}

impl Asm {
    pub fn new() -> Self {
        Self::default()
    }

    /// Address of the next emitted instruction.
    pub fn here(&self) -> u64 {
        ENTRY_PC + self.body.len() as u64
    }

    pub fn label(&mut self) -> Label {
        self.labels.push(None);
        Label(self.labels.len() - 1)
    }

    pub fn bind(&mut self, l: Label) {
        assert!(self.labels[l.0].is_none(), "label bound twice");
        self.labels[l.0] = Some(self.here());
    }

    pub fn emit(&mut self, instr: Instr) -> &mut Self {
        self.body.push(instr);
        self
    }

    /// Marks general-purpose registers as input-hinted by the last instruction.
    pub fn hint_in(&mut self, regs: &[u8]) -> &mut Self {
        let last = self.body.last_mut().expect("no instruction to hint");
        for &r in regs {
            let g = gpr_index(r).expect("only general-purpose registers can be input-hinted");
            last.hint_in[g] = true;
        }
        self
    }

    /// `FMA d == a * b + c`.
    pub fn fma(&mut self, d: Arg, a: Arg, b: Arg, c: Arg) -> &mut Self {
        self.emit(Instr::new(d, a, b, c, false))
    }

    /// `FMA d == a * b + c, hint out`.
    pub fn fma_out(&mut self, d: Arg, a: Arg, b: Arg, c: Arg) -> &mut Self {
        self.emit(Instr::new(d, a, b, c, true))
    }

    pub fn add(&mut self, d: Arg, a: Arg, b: Arg) -> &mut Self {
        self.fma_out(d, Arg::imm(1), a, b)
    }

    /// `d = a - b`.
    pub fn sub(&mut self, d: Arg, a: Arg, b: Arg) -> &mut Self {
        self.fma_out(d, Arg::imm(-1), b, a)
    }

    pub fn mul(&mut self, d: Arg, a: Arg, b: Arg) -> &mut Self {
        self.fma_out(d, a, b, Arg::imm(0))
    }

    pub fn mov(&mut self, d: Arg, a: Arg) -> &mut Self {
        self.fma_out(d, Arg::imm(0), Arg::imm(0), a)
    }

    /// Asserts `MEM[addr] == v`, fixing the cell if it is still free.
    pub fn store(&mut self, addr: Arg, v: Arg) -> &mut Self {
        assert!(addr.mem);
        self.fma(addr, Arg::imm(0), Arg::imm(0), v)
    }

    /// `reg = a^{-1}` as `FMA reg == (a + 1) * reg - 1, hint reg`.
    /// `a` must not dereference memory nor use `reg`.
    pub fn inv(&mut self, reg: u8, a: Arg) -> &mut Self {
        assert!(
            !a.mem && a.reg != reg,
            "INV: `a` must be a register expression not using `reg`"
        );
        let a1 = Arg {
            offset: a.offset + math::field::element::FieldElement::one(),
            ..a
        };
        self.fma(Arg::reg(reg), a1, Arg::reg(reg), Arg::imm(-1))
            .hint_in(&[reg])
    }

    fn jump_kind(&mut self, target: Label, kind: Fixup) -> &mut Self {
        self.fixups.push((self.body.len(), target, kind));
        self.fma_out(Arg::reg(REG_PC), Arg::imm(0), Arg::imm(0), Arg::imm(0))
    }

    pub fn jump(&mut self, target: Label) -> &mut Self {
        self.jump_kind(target, Fixup::Jump)
    }

    /// Jumps when the previous instruction's `d` was zero.
    pub fn jump_zero(&mut self, target: Label) -> &mut Self {
        self.jump_kind(target, Fixup::JumpZero)
    }

    /// Jumps when the previous instruction's `d` was nonzero.
    pub fn jump_not_zero(&mut self, target: Label) -> &mut Self {
        self.jump_kind(target, Fixup::JumpNotZero)
    }

    /// `FMA PC == 0, hint out`: jump into the halt loop.
    pub fn halt(&mut self) -> &mut Self {
        self.fma_out(Arg::reg(REG_PC), Arg::imm(0), Arg::imm(0), Arg::imm(0))
    }

    /// The spec's `CALL`. The new frame pointer is an output hint the oracle
    /// answers. `saved` registers go to `MEM[fp - 1 - i]`, `args` (evaluated
    /// with the new `fp`) to `MEM[fp + 2 + j]`; the callee leaves its result at
    /// `MEM[fp + 2 + args.len()]`, which lands in `ret` before the caller's
    /// registers and `fp` are restored.
    pub fn call(
        &mut self,
        target: Label,
        fp: u8,
        saved: &[u8],
        args: &[Arg],
        ret: Option<u8>,
    ) -> &mut Self {
        self.fma_out(Arg::mem(fp, 0), Arg::imm(0), Arg::imm(0), Arg::reg(fp));
        for (i, &r) in saved.iter().enumerate() {
            self.store(Arg::mem(fp, -1 - i as i64), Arg::reg(r));
        }
        for (j, &a) in args.iter().enumerate() {
            self.store(Arg::mem(fp, 2 + j as i64), a);
        }
        self.store(Arg::mem(fp, 1), Arg::lin(REG_PC, 1, 2));
        self.jump(target);
        if let Some(r) = ret {
            self.mov(Arg::reg(r), Arg::mem(fp, 2 + args.len() as i64));
        }
        for (i, &r) in saved.iter().enumerate() {
            self.mov(Arg::reg(r), Arg::mem(fp, -1 - i as i64));
        }
        self.mov(Arg::reg(fp), Arg::mem(fp, 0))
    }

    /// `FMA PC == MEM[fp + 1], hint out`.
    pub fn ret(&mut self, fp: u8) -> &mut Self {
        self.fma_out(Arg::reg(REG_PC), Arg::imm(0), Arg::imm(0), Arg::mem(fp, 1))
    }

    pub fn finish(mut self) -> Result<Program, AsmError> {
        for &(idx, label, kind) in &self.fixups {
            let t = self.labels[label.0].ok_or(AsmError::UnboundLabel(label))? as i64;
            let instr = &mut self.body[idx];
            match kind {
                Fixup::Jump => instr.c = Arg::imm(t),
                Fixup::JumpZero => {
                    instr.a = Arg::reg(REG_ZERO);
                    instr.b = Arg::lin(REG_PC, -1, t - 1);
                    instr.c = Arg::lin(REG_PC, 1, 1);
                }
                Fixup::JumpNotZero => {
                    instr.a = Arg::reg(REG_ZERO);
                    instr.b = Arg::lin(REG_PC, 1, -t);
                    instr.c = Arg::lin(REG_ZERO, 1, t);
                }
            }
        }
        Program::new(self.body).map_err(|(pc, reason)| AsmError::Invalid { pc, reason })
    }
}
