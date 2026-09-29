//! Translates an executed LFM program into a Field VM program over the same
//! memory.
//!
//! `MEM` holds the LFM's final memory (lanes 0..3 of each word), so every
//! translated instruction is an `FMA` assertion over cells that are already
//! fixed; executing the result checks the translation. What the 3MI split
//! gives to the RISC-V half (hashing and the byte glue around it) is skipped
//! and counted: its outputs are in `MEM` like any other cell.

use std::collections::BTreeMap;

use super::asm::Asm;
use super::isa::{Arg, Program, REG_ZERO, gpr};
use crate::lfm::compiler::LfmProgram;
use crate::lfm::executor::LfmExecution;
use crate::lfm::instr::{Addr, BaseOp, ExtOp, Instr};
use crate::tables::types::{FE, FEE};

pub struct Translation {
    pub program: Program,
    pub mem: Vec<FEE>,
    /// Addresses the LFM program publishes and its extension constants.
    pub public: Vec<u64>,
    /// LFM instructions per kind, and how many were skipped.
    pub lfm_counts: BTreeMap<&'static str, usize>,
    pub skipped: BTreeMap<&'static str, usize>,
}

fn cell(a: &Addr) -> Arg {
    Arg::mem_abs(a.0)
}

fn word_to_ext(w: [FE; 4]) -> FEE {
    FEE::new([w[0], w[1], w[2]])
}

/// `bit_dec` decomposes with FMA rows (a bit check and an accumulation per
/// bit); otherwise it is skipped as RISC-V-half work.
pub fn translate(program: &LfmProgram, exec: &LfmExecution, bit_dec: bool) -> Translation {
    let n = program.num_addrs;
    let mut mem: Vec<FEE> = (0..n)
        .map(|a| {
            exec.memory
                .get(Addr(a))
                .map(word_to_ext)
                .unwrap_or(FEE::zero())
        })
        .collect();
    let scratch = |mem: &mut Vec<FEE>, v: FEE| {
        mem.push(v);
        mem.len() as u64 - 1
    };
    let (acc, tmp) = (gpr(0), gpr(1));
    let mut asm = Asm::new();
    let mut public = Vec::new();
    let mut lfm_counts = BTreeMap::new();
    let mut skipped = BTreeMap::new();

    for instr in &program.instrs {
        let kind = match instr {
            Instr::Const { .. } => "const",
            Instr::BaseAlu { .. } => "base_alu",
            Instr::ExtAlu { .. } => "ext_alu",
            Instr::Select { .. } => "select",
            Instr::BitDec { .. } => "bit_dec",
            Instr::Hash { .. } => "hash",
            Instr::Hint { .. } => "hint",
            Instr::Pack { .. } => "pack",
            Instr::Unpack { .. } => "unpack",
            Instr::KeccakF(_) => "keccak",
            Instr::Blake3(_) => "blake3",
            Instr::Public { .. } => "public",
        };
        *lfm_counts.entry(kind).or_insert(0) += 1;
        match instr {
            Instr::Const { out, value, .. } => {
                if value[1] == FE::zero() && value[2] == FE::zero() {
                    asm.store(cell(out), Arg::imm_fe(value[0]));
                } else {
                    public.push(out.0);
                }
            }
            Instr::BaseAlu {
                op, out, a, b, c, ..
            } => {
                let one = Arg::imm(1);
                match op {
                    BaseOp::Add => asm.fma(cell(out), one, cell(a), cell(b)),
                    BaseOp::Sub => asm.fma(cell(out), Arg::imm(-1), cell(b), cell(a)),
                    BaseOp::Mul => asm.fma(cell(out), cell(a), cell(b), Arg::imm(0)),
                    BaseOp::MulAdd => asm.fma(cell(out), cell(a), cell(b), cell(c)),
                    BaseOp::Div => asm.fma(cell(a), cell(b), cell(out), Arg::imm(0)),
                };
            }
            Instr::ExtAlu {
                op, out, a, b, c, ..
            } => {
                let one = Arg::imm(1);
                match op {
                    ExtOp::Add => asm.fma(cell(out), one, cell(a), cell(b)),
                    ExtOp::Sub => asm.fma(cell(out), Arg::imm(-1), cell(b), cell(a)),
                    ExtOp::Mul | ExtOp::MulBase => {
                        asm.fma(cell(out), cell(a), cell(b), Arg::imm(0))
                    }
                    ExtOp::MulAdd => asm.fma(cell(out), cell(a), cell(b), cell(c)),
                    ExtOp::Div => asm.fma(cell(a), cell(b), cell(out), Arg::imm(0)),
                };
            }
            Instr::Select {
                bit,
                out_l,
                out_r,
                in_l,
                in_r,
                ..
            } => {
                // r = in_l − in_r; out_l = in_l − bit·r; out_r = in_r + bit·r.
                asm.fma_out(Arg::reg(tmp), Arg::imm(-1), cell(in_r), cell(in_l))
                    .fma(cell(out_l), cell(bit), Arg::lin(tmp, -1, 0), cell(in_l))
                    .fma(cell(out_r), cell(bit), Arg::reg(tmp), cell(in_r));
            }
            Instr::BitDec {
                input,
                bits,
                halves,
            } => {
                if !bit_dec {
                    *skipped.entry(kind).or_insert(0) += 1;
                    continue;
                }
                let x = mem[input.0 as usize].value()[0].canonical();
                let lo_cell = scratch(&mut mem, FEE::from(x & 0xFFFF_FFFF));
                for half in 0..2 {
                    asm.mov(Arg::reg(acc), Arg::imm(0));
                    for i in 32 * half..32 * (half + 1) {
                        let b = match bits.get(i) {
                            Some((a, _)) => a.0,
                            None => scratch(&mut mem, FEE::from((x >> i) & 1)),
                        };
                        asm.fma(
                            Arg::mem_abs(b),
                            Arg::mem_abs(b),
                            Arg::mem_abs(b),
                            Arg::imm(0),
                        )
                        .fma_out(
                            Arg::reg(acc),
                            Arg::imm_fe(FE::from(1u64 << (i - 32 * half))),
                            Arg::mem_abs(b),
                            Arg::reg(acc),
                        );
                    }
                    if half == 0 {
                        asm.store(Arg::mem_abs(lo_cell), Arg::reg(acc));
                    }
                }
                asm.fma(
                    cell(input),
                    Arg::imm_fe(FE::from(1u64 << 32)),
                    Arg::reg(acc),
                    Arg::mem_abs(lo_cell),
                );
                // The halves are byte-swapped `u32`s for the transcript: byte
                // glue, which the 3MI split leaves to the RISC-V half.
                if halves.is_some() {
                    *skipped.entry("bit_dec_halves").or_insert(0) += 1;
                }
            }
            Instr::Public { addr, .. } => public.push(addr.0),
            Instr::Hint { .. } => {}
            _ => *skipped.entry(kind).or_insert(0) += 1,
        }
    }
    asm.halt();
    public.sort_unstable();
    public.dedup();
    let mut program = asm.finish().expect("translated program");
    if std::env::var_os("FVM_MEM_COMPACT").is_some() {
        (mem, public) = compact(&mut program, &mem, &public);
    }
    Translation {
        program,
        mem,
        public,
        lfm_counts,
        skipped,
    }
}

/// Keeps only the cells the program reads or publishes, renumbered densely in
/// first-use order. Every memory argument of a translation is an absolute
/// address, so renumbering is rewriting offsets.
fn compact(program: &mut Program, mem: &[FEE], public: &[u64]) -> (Vec<FEE>, Vec<u64>) {
    let mut remap: std::collections::HashMap<u64, u64> = std::collections::HashMap::new();
    let mut dense = Vec::new();
    let mut slot = |old: u64, dense: &mut Vec<FEE>| {
        *remap.entry(old).or_insert_with(|| {
            dense.push(mem[old as usize]);
            dense.len() as u64 - 1
        })
    };
    for instr in program.instrs.iter_mut() {
        for arg in [&mut instr.d, &mut instr.a, &mut instr.b, &mut instr.c] {
            if arg.mem {
                assert!(
                    arg.reg == REG_ZERO && arg.scale == FE::zero(),
                    "translations address memory absolutely"
                );
                arg.offset = FE::from(slot(arg.offset.canonical(), &mut dense));
            }
        }
    }
    let mut public: Vec<u64> = public.iter().map(|&a| slot(a, &mut dense)).collect();
    public.sort_unstable();
    (dense, public)
}
