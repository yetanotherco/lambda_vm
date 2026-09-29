//! Field VM executor.
//!
//! Produces one [`Step`] per trace row. Row `i` holds the state instruction
//! `pc_i` acts on, *after* that instruction's input hints were applied; the
//! instruction's `d` argument reads row `i + 1`.
//!
//! `MEM` is read-only for the proof, but its contents are chosen by the prover:
//! a `d` argument that dereferences a cell nobody has fixed yet defines it as
//! `a * b + c`. Any other read of an unfixed cell is an error.
//!
//! Hints are resolved in this order:
//! - output hint, `d` a register expression with `imm0 != 0`: solved for the
//!   register; the halt loop's input hints are all zero;
//! - a single input hint whose instruction is linear in it (e.g. `INV`): solved;
//! - anything else: asked to the [`HintOracle`].

use super::isa::{
    Arg, HALT_PC, Instr, N, NUM_REGS, Program, REG_PC, REG_ZERO, START_PC, gpr, gpr_index,
};
use crate::tables::types::{FE, FEE};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct State {
    pub zero: bool,
    pub pc: u64,
    pub regs: [FEE; N],
}

impl State {
    pub fn initial() -> Self {
        Self {
            zero: false,
            pc: START_PC,
            regs: [FEE::zero(); N],
        }
    }

    /// All registers in register-index order.
    pub fn all_regs(&self) -> [FEE; NUM_REGS] {
        let mut all = [FEE::zero(); NUM_REGS];
        all[REG_ZERO as usize] = if self.zero { FEE::one() } else { FEE::zero() };
        all[REG_PC as usize] = FEE::from(self.pc);
        all[2..].copy_from_slice(&self.regs);
        all
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Step {
    pub state: State,
    pub args_premem: [FEE; 4],
    pub args: [FEE; 4],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HintKind {
    Output,
    Input,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HintRequest {
    pub kind: HintKind,
    pub pc: u64,
    pub reg: u8,
    /// For an output hint, the value `a * b + c` that `d` must equal.
    pub target: Option<FEE>,
}

pub trait HintOracle {
    fn hint(&mut self, req: &HintRequest) -> Option<FEE>;
}

impl<F: FnMut(&HintRequest) -> Option<FEE>> HintOracle for F {
    fn hint(&mut self, req: &HintRequest) -> Option<FEE> {
        self(req)
    }
}

pub struct NoHints;

impl HintOracle for NoHints {
    fn hint(&mut self, _: &HintRequest) -> Option<FEE> {
        None
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExecError {
    PcOutOfRange { pc: u64, target: FEE },
    BadAddress { pc: u64, addr: FEE },
    UninitializedRead { pc: u64, addr: u64 },
    FmaViolation { pc: u64 },
    HintCollision { pc: u64, reg: u8 },
    UnresolvedHint { pc: u64, reg: u8 },
    StepLimit,
}

#[derive(Clone, Debug)]
pub struct Memory {
    cells: Vec<Option<FEE>>,
    mult: Vec<u64>,
}

impl Memory {
    pub fn new(size: usize) -> Self {
        Self {
            cells: vec![None; size],
            mult: vec![0; size],
        }
    }

    /// A memory whose first cells are `init` (e.g. the proof being verified).
    pub fn with_init(size: usize, init: &[FEE]) -> Self {
        assert!(init.len() <= size);
        let mut mem = Self::new(size);
        for (cell, v) in mem.cells.iter_mut().zip(init) {
            *cell = Some(*v);
        }
        mem
    }

    pub fn len(&self) -> usize {
        self.cells.len()
    }

    pub fn is_empty(&self) -> bool {
        self.cells.is_empty()
    }

    pub fn get(&self, addr: usize) -> Option<FEE> {
        self.cells[addr]
    }

    fn addr(&self, pc: u64, v: &FEE) -> Result<usize, ExecError> {
        let [c0, c1, c2] = v.value();
        let a = c0.canonical();
        if *c1 != FE::zero() || *c2 != FE::zero() || a >= self.cells.len() as u64 {
            return Err(ExecError::BadAddress { pc, addr: *v });
        }
        Ok(a as usize)
    }

    fn peek(&self, pc: u64, addr: &FEE) -> Result<Option<FEE>, ExecError> {
        Ok(self.cells[self.addr(pc, addr)?])
    }

    fn read(&mut self, pc: u64, addr: &FEE) -> Result<FEE, ExecError> {
        let a = self.addr(pc, addr)?;
        let v = self.cells[a].ok_or(ExecError::UninitializedRead { pc, addr: a as u64 })?;
        self.mult[a] += 1;
        Ok(v)
    }

    /// Reads `MEM[addr]`, defining it as `v` if nothing fixed it yet.
    fn read_or_define(&mut self, pc: u64, addr: &FEE, v: FEE) -> Result<FEE, ExecError> {
        let a = self.addr(pc, addr)?;
        let cell = *self.cells[a].get_or_insert(v);
        self.mult[a] += 1;
        Ok(cell)
    }
}

#[derive(Clone, Debug)]
pub struct Execution {
    /// One step per executed instruction, ending with the first halt step.
    pub steps: Vec<Step>,
    /// Final memory contents; cells never fixed read as zero.
    pub mem: Vec<FEE>,
    /// Number of lookups into each memory cell.
    pub mem_mult: Vec<u64>,
    /// Number of executed steps per program address.
    pub decode_mult: Vec<u64>,
}

fn arg_value(
    pc: u64,
    arg: &Arg,
    regs: &[FEE; NUM_REGS],
    mem: &mut Memory,
) -> Result<(FEE, FEE), ExecError> {
    let premem = arg.premem(regs);
    let v = if arg.mem {
        mem.read(pc, &premem)?
    } else {
        premem
    };
    Ok((premem, v))
}

/// `d - (a * b + c)` for `instr` with general-purpose register `g` set to `x`,
/// assuming no output hint (the future state keeps `x`). `None` when a memory
/// argument is not fixed, or its address depends on `x`.
fn residual_at(instr: &Instr, state: &State, g: usize, x: FEE, mem: &Memory) -> Option<FEE> {
    let mut cur = *state;
    cur.regs[g] = x;
    let regs = cur.all_regs();
    let mut next = cur;
    next.pc += 1;
    let next_regs = next.all_regs();
    let value = |arg: &Arg, regs: &[FEE; NUM_REGS]| -> Option<FEE> {
        let premem = arg.premem(regs);
        if !arg.mem {
            return Some(premem);
        }
        if gpr_index(arg.reg) == Some(g) && arg.scale != FE::zero() {
            return None;
        }
        mem.peek(state.pc, &premem).ok().flatten()
    };
    if instr.d.reg == REG_ZERO && instr.d.scale != FE::zero() {
        return None;
    }
    let d = value(&instr.d, &next_regs)?;
    let a = value(&instr.a, &regs)?;
    let b = value(&instr.b, &regs)?;
    let c = value(&instr.c, &regs)?;
    Some(d - (a * b + c))
}

/// Solves the single input-hinted register `g` of `instr` when the
/// constraint is linear in it.
fn solve_input_hint(instr: &Instr, state: &State, g: usize, mem: &Memory) -> Option<FEE> {
    if instr.hint_out {
        return None;
    }
    let f0 = residual_at(instr, state, g, FEE::zero(), mem)?;
    let f1 = residual_at(instr, state, g, FEE::one(), mem)?;
    let f2 = residual_at(instr, state, g, FEE::from(2u64), mem)?;
    let two = FEE::from(2u64);
    let quad = (f2 - f1 * two + f0) * two.inv().ok()?;
    if quad != FEE::zero() {
        return None;
    }
    let lin = f1 - f0;
    Some(-f0 * lin.inv().ok()?)
}

fn instr_at(program: &Program, pc: u64, target: FEE) -> Result<&Instr, ExecError> {
    program
        .instrs
        .get(pc as usize)
        .ok_or(ExecError::PcOutOfRange { pc, target })
}

/// Applies `instr`'s input hints to `state`, which it is about to act on.
fn apply_input_hints(
    instr: &Instr,
    state: &mut State,
    mem: &Memory,
    oracle: &mut dyn HintOracle,
) -> Result<(), ExecError> {
    let hinted: Vec<usize> = (0..N).filter(|&g| instr.hint_in[g]).collect();
    if state.pc == HALT_PC {
        state.regs = [FEE::zero(); N];
        return Ok(());
    }
    let solved = match hinted.as_slice() {
        [g] => solve_input_hint(instr, state, *g, mem).map(|x| (*g, x)),
        _ => None,
    };
    for g in hinted {
        let x = match solved {
            Some((sg, x)) if sg == g => x,
            _ => oracle
                .hint(&HintRequest {
                    kind: HintKind::Input,
                    pc: state.pc,
                    reg: gpr(g),
                    target: None,
                })
                .ok_or(ExecError::UnresolvedHint {
                    pc: state.pc,
                    reg: gpr(g),
                })?,
        };
        state.regs[g] = x;
    }
    Ok(())
}

pub fn execute(
    program: &Program,
    mut mem: Memory,
    oracle: &mut dyn HintOracle,
    max_steps: usize,
) -> Result<Execution, ExecError> {
    let mut decode_mult = vec![0u64; program.len()];
    let mut steps = Vec::new();
    let mut cur = State::initial();

    loop {
        if steps.len() >= max_steps {
            return Err(ExecError::StepLimit);
        }
        let pc = cur.pc;
        let instr = *instr_at(program, pc, FEE::from(pc))?;
        let regs = cur.all_regs();

        let (a_pre, a) = arg_value(pc, &instr.a, &regs, &mut mem)?;
        let (b_pre, b) = arg_value(pc, &instr.b, &regs, &mut mem)?;
        let (c_pre, c) = arg_value(pc, &instr.c, &regs, &mut mem)?;
        let target = a * b + c;

        let mut next = cur;
        next.pc = pc + 1;
        let d = instr.d;
        if instr.hint_out {
            let solved = if !d.mem && d.scale != FE::zero() {
                Some(
                    (target - d.offset.to_extension())
                        * d.scale.inv().expect("nonzero").to_extension(),
                )
            } else {
                oracle.hint(&HintRequest {
                    kind: HintKind::Output,
                    pc,
                    reg: d.reg,
                    target: Some(target),
                })
            }
            .ok_or(ExecError::UnresolvedHint { pc, reg: d.reg })?;
            match gpr_index(d.reg) {
                Some(g) => next.regs[g] = solved,
                None => {
                    let [c0, c1, c2] = solved.value();
                    if *c1 != FE::zero() || *c2 != FE::zero() {
                        return Err(ExecError::PcOutOfRange { pc, target: solved });
                    }
                    next.pc = c0.canonical();
                }
            }
        }

        let next_instr = *instr_at(program, next.pc, FEE::from(next.pc))?;
        if let Some(g) = gpr_index(d.reg)
            && next_instr.hint_in[g]
        {
            return Err(ExecError::HintCollision { pc, reg: d.reg });
        }
        apply_input_hints(&next_instr, &mut next, &mem, oracle)?;

        let d_pre = d.premem(&next.all_regs());
        let d_val = if d.mem {
            mem.read_or_define(pc, &d_pre, target)?
        } else {
            d_pre
        };
        if d_val != target {
            return Err(ExecError::FmaViolation { pc });
        }
        next.zero = d_val == FEE::zero();

        decode_mult[pc as usize] += 1;
        steps.push(Step {
            state: cur,
            args_premem: [d_pre, a_pre, b_pre, c_pre],
            args: [d_val, a, b, c],
        });
        if pc == HALT_PC {
            break;
        }
        cur = next;
    }

    let mem_vals = mem.cells.iter().map(|c| c.unwrap_or(FEE::zero())).collect();
    Ok(Execution {
        steps,
        mem: mem_vals,
        mem_mult: mem.mult,
        decode_mult,
    })
}
