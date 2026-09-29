//! Field VM instruction set.
//!
//! Spec: `spec/chapters/field_vm.typ` (#971, `spec/recursion.vm` @ `d55a0f122`).
//!
//! The only instruction is the constraint `FMA d == a * b + c` over the cubic
//! extension. Each argument is `imm0 * reg + imm1`, optionally dereferenced as
//! `MEM[imm0 * reg + imm1]`; `d` reads its register from the future state.
//! Registers are indexed `ZERO = 0`, `PC = 1`, then `N` general-purpose ones.

use crate::tables::types::{FE, FEE};

/// Number of general-purpose registers.
pub const N: usize = 5;
/// `ZERO`, `PC` and the `N` general-purpose registers.
pub const NUM_REGS: usize = N + 2;

pub const REG_ZERO: u8 = 0;
pub const REG_PC: u8 = 1;

/// Index of general-purpose register `i`.
pub const fn gpr(i: usize) -> u8 {
    assert!(i < N);
    (i + 2) as u8
}

/// `Some(i)` when `reg` is general-purpose register `i`.
pub const fn gpr_index(reg: u8) -> Option<usize> {
    if reg >= 2 && (reg as usize) < NUM_REGS {
        Some(reg as usize - 2)
    } else {
        None
    }
}

/// The halting self-loop lives at `PC = 0`.
pub const HALT_PC: u64 = 0;
/// Execution starts at `PC = 1`, which holds a hint-free `FMA 0 == 0`.
pub const START_PC: u64 = 1;
/// First address of the user program.
pub const ENTRY_PC: u64 = 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Arg {
    pub reg: u8,
    pub scale: FE,
    pub offset: FE,
    pub mem: bool,
}

impl Arg {
    /// `scale * reg + offset`.
    pub fn lin(reg: u8, scale: i64, offset: i64) -> Self {
        Self {
            reg,
            scale: FE::from(scale),
            offset: FE::from(offset),
            mem: false,
        }
    }

    pub fn reg(reg: u8) -> Self {
        Self::lin(reg, 1, 0)
    }

    /// The base-field immediate `v`.
    pub fn imm(v: i64) -> Self {
        Self::lin(REG_ZERO, 0, v)
    }

    pub fn imm_fe(v: FE) -> Self {
        Self {
            reg: REG_ZERO,
            scale: FE::zero(),
            offset: v,
            mem: false,
        }
    }

    /// `MEM[addr]`.
    pub fn mem_abs(addr: u64) -> Self {
        Self::imm_fe(FE::from(addr)).deref()
    }

    /// `MEM[reg + offset]`.
    pub fn mem(reg: u8, offset: i64) -> Self {
        Self::lin(reg, 1, offset).deref()
    }

    /// The same address expression, dereferenced.
    pub fn deref(self) -> Self {
        Self { mem: true, ..self }
    }

    /// The value before a potential memory lookup, given all register values.
    pub fn premem(&self, regs: &[FEE; NUM_REGS]) -> FEE {
        self.scale * regs[self.reg as usize] + self.offset.to_extension()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Instr {
    pub d: Arg,
    pub a: Arg,
    pub b: Arg,
    pub c: Arg,
    pub hint_out: bool,
    pub hint_in: [bool; N],
}

impl Instr {
    pub fn new(d: Arg, a: Arg, b: Arg, c: Arg, hint_out: bool) -> Self {
        Self {
            d,
            a,
            b,
            c,
            hint_out,
            hint_in: [false; N],
        }
    }

    pub fn args(&self) -> [Arg; 4] {
        [self.d, self.a, self.b, self.c]
    }

    /// `FMA PC == PC, hint out, hint 2, ..., hint (N + 1)`.
    pub fn halt() -> Self {
        Self {
            d: Arg::reg(REG_PC),
            a: Arg::imm(0),
            b: Arg::imm(0),
            c: Arg::reg(REG_PC),
            hint_out: true,
            hint_in: [true; N],
        }
    }

    /// `FMA 0 == 0`, no hinting.
    pub fn nop() -> Self {
        Self::new(Arg::imm(0), Arg::imm(0), Arg::imm(0), Arg::imm(0), false)
    }

    pub fn validate(&self) -> Result<(), &'static str> {
        if self.args().iter().any(|a| a.reg as usize >= NUM_REGS) {
            return Err("register index out of range");
        }
        if self.hint_out && self.d.reg == REG_ZERO {
            return Err("the ZERO register cannot be output-hinted");
        }
        Ok(())
    }
}

/// A program: the halt loop at `PC = 0`, the start `nop` at `PC = 1`, then the body.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Program {
    pub instrs: Vec<Instr>,
}

impl Program {
    pub fn new(body: Vec<Instr>) -> Result<Self, (usize, &'static str)> {
        let mut instrs = Vec::with_capacity(body.len() + 2);
        instrs.push(Instr::halt());
        instrs.push(Instr::nop());
        instrs.extend(body);
        for (pc, instr) in instrs.iter().enumerate() {
            instr.validate().map_err(|e| (pc, e))?;
        }
        Ok(Self { instrs })
    }

    pub fn len(&self) -> usize {
        self.instrs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.instrs.is_empty()
    }
}
