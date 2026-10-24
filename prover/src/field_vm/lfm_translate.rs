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
use super::isa::{Arg, ENTRY_PC, Instr as FInstr, Program, REG_ZERO, gpr};
use crate::lfm::compiler::LfmProgram;
use crate::lfm::executor::LfmExecution;
pub use crate::lfm::fn_scope::Scope;
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
    /// The memory address of each cell asked to be kept, in request order.
    pub kept: Vec<u64>,
    /// `(publish index, memory address)` of each word the program publishes,
    /// in program order.
    #[cfg_attr(not(feature = "parallel"), allow(dead_code))]
    pub published: Vec<(u32, u64)>,
    /// The public cells that are the program's constants, with their values.
    #[cfg(all(test, feature = "parallel"))]
    pub consts: Vec<(u64, FEE)>,
    /// Lane 3 of constant cells, which the Field VM's extension memory omits.
    pub const_lane3: Vec<(u64, FE)>,
    /// The LFM instruction each program address was translated from;
    /// `usize::MAX` for the halt loop, the start `nop` and loop control.
    pub origin: Vec<usize>,
}

fn cell(a: &Addr) -> Arg {
    Arg::mem_abs(a.0)
}

fn word_to_ext(w: [FE; 4]) -> FEE {
    FEE::new([w[0], w[1], w[2]])
}

/// An ALU instruction as `(is_ext, op, out, a, b, c)`.
fn alu(i: &Instr) -> Option<(bool, AluOp, u64, u64, u64, u64)> {
    match i {
        Instr::BaseAlu {
            op, out, a, b, c, ..
        } => Some((false, AluOp::from_base(*op), out.0, a.0, b.0, c.0)),
        Instr::ExtAlu {
            op, out, a, b, c, ..
        } => Some((true, AluOp::from_ext(*op), out.0, a.0, b.0, c.0)),
        _ => None,
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum AluOp {
    Add,
    Sub,
    Mul,
    MulAdd,
    Div,
}

impl AluOp {
    fn from_base(op: BaseOp) -> Self {
        match op {
            BaseOp::Add => Self::Add,
            BaseOp::Sub => Self::Sub,
            BaseOp::Mul => Self::Mul,
            BaseOp::MulAdd => Self::MulAdd,
            BaseOp::Div => Self::Div,
        }
    }

    fn from_ext(op: ExtOp) -> Self {
        match op {
            ExtOp::Add => Self::Add,
            ExtOp::Sub => Self::Sub,
            ExtOp::Mul | ExtOp::MulBase => Self::Mul,
            ExtOp::MulAdd => Self::MulAdd,
            ExtOp::Div => Self::Div,
        }
    }
}

/// How an ALU instruction's result is produced.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Plan {
    Plain,
    /// Folded into its single consumer's row.
    Fused,
    /// Written to the accumulator register for the next instruction.
    Chained,
}

/// Single-use results that fold into their consumer's FMA, or stay in a
/// register when the consumer comes right after.
fn plan(program: &LfmProgram) -> Vec<Plan> {
    let mut produced: std::collections::HashMap<u64, (usize, u64)> =
        std::collections::HashMap::new();
    for (i, instr) in program.instrs.iter().enumerate() {
        match instr {
            Instr::BaseAlu { out, mult, .. } | Instr::ExtAlu { out, mult, .. } => {
                produced.insert(out.0, (i, *mult));
            }
            _ => {}
        }
    }
    let single = |addr: u64| {
        produced
            .get(&addr)
            .filter(|(_, m)| *m == 1)
            .map(|(i, _)| *i)
    };
    let mut plan = vec![Plan::Plain; program.instrs.len()];
    // Rows whose result lands outside the `d` slot (a Div, a fused Sub).
    let mut out_not_d = vec![false; program.instrs.len()];
    for (j, instr) in program.instrs.iter().enumerate() {
        let Some((ext, op, _, a, b, _)) = alu(instr) else {
            continue;
        };
        let is = |p: usize, want: AluOp| {
            alu(&program.instrs[p]).is_some_and(|(pe, pop, ..)| pe == ext && pop == want)
        };
        match op {
            // (x − y) / c  ⇒  [x] == [c]·[q] + [y]
            AluOp::Div => {
                out_not_d[j] = true;
                if let Some(p) = single(a).filter(|&p| is(p, AluOp::Sub)) {
                    plan[p] = Plan::Fused;
                }
            }
            // z − x·y  ⇒  [z] == [x]·[y] + [q]
            AluOp::Sub => {
                if let Some(p) = single(b).filter(|&p| is(p, AluOp::Mul)) {
                    plan[p] = Plan::Fused;
                    out_not_d[j] = true;
                }
            }
            // x·y + z  ⇒  [q] == [x]·[y] + [z]
            AluOp::Add => {
                if let Some(p) = single(a).filter(|&p| is(p, AluOp::Mul)) {
                    plan[p] = Plan::Fused;
                } else if let Some(p) = single(b).filter(|&p| is(p, AluOp::Mul)) {
                    plan[p] = Plan::Fused;
                }
            }
            _ => {}
        }
    }
    for c in 1..program.instrs.len() {
        let p = c - 1;
        let (Some((_, _, out, ..)), Some((_, cop, _, a, b, cc))) =
            (alu(&program.instrs[p]), alu(&program.instrs[c]))
        else {
            continue;
        };
        let reads = a == out || b == out || (cop == AluOp::MulAdd && cc == out);
        if plan[p] == Plan::Plain
            && plan[c] != Plan::Fused
            && !out_not_d[p]
            && reads
            && single(out) == Some(p)
        {
            plan[p] = Plan::Chained;
        }
    }
    plan
}

/// `bit_dec` decomposes with FMA rows (a bit check and an accumulation per
/// bit); otherwise it is skipped as RISC-V-half work. `FVM_TRANSLATE_OPT`
/// turns on the planner above and moves base constants to public cells.
pub fn translate(program: &LfmProgram, exec: &LfmExecution, bit_dec: bool) -> Translation {
    translate_keeping(program, exec, bit_dec, &[], &|_, _| false)
}

/// [`translate`] keeping the cells at `keep` in memory whether or not the
/// program reads them, and skipping the instructions `skip` names by index.
pub fn translate_keeping(
    program: &LfmProgram,
    exec: &LfmExecution,
    bit_dec: bool,
    keep: &[u64],
    skip: &dyn Fn(usize, &Instr) -> bool,
) -> Translation {
    translate_outlined(program, exec, bit_dec, keep, skip, &[])
}

/// [`translate_keeping`] with the regions `scopes` names outlined
/// (`FVM_OUTLINE`): each function's body is emitted once and every region
/// that matches it becomes a call ([`outline`]).
pub fn translate_outlined(
    program: &LfmProgram,
    exec: &LfmExecution,
    bit_dec: bool,
    keep: &[u64],
    skip: &dyn Fn(usize, &Instr) -> bool,
    scopes: &[Scope],
) -> Translation {
    translate_outlined_mode(
        program,
        exec,
        bit_dec,
        keep,
        skip,
        scopes,
        std::env::var_os("FVM_CANONICAL_PUBLICS").is_some(),
    )
}

/// [`translate_outlined`] with deterministic public ordering: `Public`
/// instructions first in publish order, followed by constant inputs.
pub fn translate_outlined_canonical(
    program: &LfmProgram,
    exec: &LfmExecution,
    bit_dec: bool,
    keep: &[u64],
    skip: &dyn Fn(usize, &Instr) -> bool,
    scopes: &[Scope],
) -> Translation {
    translate_outlined_mode(program, exec, bit_dec, keep, skip, scopes, true)
}

fn translate_outlined_mode(
    program: &LfmProgram,
    exec: &LfmExecution,
    bit_dec: bool,
    keep: &[u64],
    skip: &dyn Fn(usize, &Instr) -> bool,
    scopes: &[Scope],
    canonical: bool,
) -> Translation {
    let program_lfm = program;
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
    // A select never meets a chain (chains feed only the next ALU
    // instruction), so it borrows the accumulator; the second register is
    // left free for loop indices (`FVM_REROLL`).
    let (acc, tmp) = (gpr(0), gpr(0));
    let opt = std::env::var_os("FVM_TRANSLATE_OPT").is_some();
    let mut plan = if opt {
        plan(program)
    } else {
        vec![Plan::Plain; program.instrs.len()]
    };
    // An outlined region keeps the accumulator dead at both ends: no chain
    // into its first instruction or out of its last.
    if std::env::var_os("FVM_OUTLINE").is_some() {
        for sc in scopes {
            for edge in [sc.start, sc.end] {
                if let Some(p) = edge.checked_sub(1)
                    && plan.get(p) == Some(&Plan::Chained)
                {
                    plan[p] = Plan::Plain;
                }
            }
        }
    }
    let producer: std::collections::HashMap<u64, usize> = program
        .instrs
        .iter()
        .enumerate()
        .filter_map(|(i, instr)| alu(instr).map(|(_, _, out, ..)| (out, i)))
        .collect();
    let fused = |x: u64| {
        producer
            .get(&x)
            .filter(|&&p| plan[p] == Plan::Fused)
            .and_then(|&p| alu(&program.instrs[p]))
    };
    // The address the accumulator holds for the current instruction.
    let mut in_acc: Option<u64> = None;
    let mut asm = Asm::new();
    let mut public = Vec::new();
    let mut consts = Vec::new();
    let mut const_lane3 = Vec::new();
    // `FVM_CANONICAL_PUBLICS`: every constant is program text (a base one by
    // immediate, an extension one from the public cell `w = (0, 1, 0)`), and
    // the program's own public cells come first in memory, in publish order.
    let mut published = Vec::new();
    let mut publish_index = Vec::new();
    let mut w_cells: Option<(u64, u64)> = None;
    let mut lfm_counts = BTreeMap::new();
    let mut skipped = BTreeMap::new();

    let mut origin: Vec<usize> = Vec::new();
    for (idx, instr) in program.instrs.iter().enumerate() {
        origin.resize((asm.here() - ENTRY_PC) as usize, idx.wrapping_sub(1));
        let held = in_acc.take();
        let arg = |x: u64| {
            if held == Some(x) {
                Arg::reg(acc)
            } else {
                Arg::mem_abs(x)
            }
        };
        let kind = match instr {
            Instr::Const { .. } => "const",
            Instr::BaseAlu { .. } => "base_alu",
            Instr::ExtAlu { .. } => "ext_alu",
            Instr::Select { .. } => "select",
            Instr::BitDec { .. } => "bit_dec",
            Instr::Hash { .. } => "hash",
            Instr::Hash16(_) => "hash16",
            Instr::Hint { .. } => "hint",
            Instr::Pack { .. } => "pack",
            Instr::Unpack { .. } => "unpack",
            Instr::KeccakF(_) => "keccak",
            Instr::Blake3(_) => "blake3",
            Instr::Public { .. } => "public",
        };
        *lfm_counts.entry(kind).or_insert(0) += 1;
        if skip(idx, instr) {
            *skipped.entry(kind).or_insert(0) += 1;
            continue;
        }
        match instr {
            Instr::Const { out, value, .. } if canonical => {
                const_lane3.push((out.0, value[3]));
                if value[1] == FE::zero() && value[2] == FE::zero() {
                    asm.store(cell(out), Arg::imm_fe(value[0]));
                } else {
                    let (w, w2) = *w_cells.get_or_insert_with(|| {
                        let w = scratch(&mut mem, FEE::new([FE::zero(), FE::one(), FE::zero()]));
                        let w2 = scratch(&mut mem, FEE::new([FE::zero(), FE::zero(), FE::one()]));
                        public.push(w);
                        consts.push((w, mem[w as usize]));
                        asm.fma(
                            Arg::mem_abs(w2),
                            Arg::mem_abs(w),
                            Arg::mem_abs(w),
                            Arg::imm(0),
                        );
                        (w, w2)
                    });
                    let t = scratch(&mut mem, FEE::new([value[0], FE::zero(), value[2]]));
                    asm.fma(
                        Arg::mem_abs(t),
                        Arg::imm_fe(value[2]),
                        Arg::mem_abs(w2),
                        Arg::imm_fe(value[0]),
                    )
                    .fma(
                        cell(out),
                        Arg::imm_fe(value[1]),
                        Arg::mem_abs(w),
                        Arg::mem_abs(t),
                    );
                }
            }
            Instr::Const { out, value, .. } => {
                const_lane3.push((out.0, value[3]));
                if value[1] == FE::zero() && value[2] == FE::zero() && !opt {
                    asm.store(cell(out), Arg::imm_fe(value[0]));
                } else {
                    public.push(out.0);
                    consts.push((out.0, word_to_ext(*value)));
                }
            }
            Instr::BaseAlu { .. } | Instr::ExtAlu { .. } => {
                let (_, op, out, a, b, c) = alu(instr).expect("an ALU instruction");
                if plan[idx] == Plan::Fused {
                    *skipped.entry("fused").or_insert(0) += 1;
                    continue;
                }
                let (d, hint) = if plan[idx] == Plan::Chained {
                    in_acc = Some(out);
                    (Arg::reg(acc), true)
                } else {
                    (Arg::mem_abs(out), false)
                };
                let (fd, fa, fb, fc) = match op {
                    AluOp::Add => match (fused(a), fused(b)) {
                        (Some((.., x, y, _)), _) => (d, arg(x), arg(y), arg(b)),
                        (_, Some((.., x, y, _))) => (d, arg(x), arg(y), arg(a)),
                        _ => (d, Arg::imm(1), arg(a), arg(b)),
                    },
                    AluOp::Sub => match fused(b) {
                        Some((.., x, y, _)) => (arg(a), arg(x), arg(y), Arg::mem_abs(out)),
                        None => (d, Arg::imm(-1), arg(b), arg(a)),
                    },
                    AluOp::Mul => (d, arg(a), arg(b), Arg::imm(0)),
                    AluOp::MulAdd => (d, arg(a), arg(b), arg(c)),
                    AluOp::Div => match fused(a) {
                        Some((.., x, y, _)) => (arg(x), arg(b), Arg::mem_abs(out), arg(y)),
                        None => (arg(a), arg(b), Arg::mem_abs(out), Arg::imm(0)),
                    },
                };
                if hint {
                    asm.fma_out(fd, fa, fb, fc);
                } else {
                    asm.fma(fd, fa, fb, fc);
                }
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
                let cells: [u64; 64] = core::array::from_fn(|i| match bits.get(i) {
                    Some((a, _)) => a.0,
                    None => scratch(&mut mem, FEE::from((x >> i) & 1)),
                });
                if std::env::var_os("FVM_BITDEC_HORNER").is_some() {
                    canonical_bit_dec_horner(&mut asm, &mut mem, cell(input), &cells, acc);
                } else {
                    canonical_bit_dec(&mut asm, &mut mem, cell(input), &cells, acc);
                }
                // The halves are byte-swapped `u32`s for the transcript: byte
                // glue, which the 3MI split leaves to the RISC-V half.
                if halves.is_some() {
                    *skipped.entry("bit_dec_halves").or_insert(0) += 1;
                }
            }
            Instr::Public { addr, index } => {
                public.push(addr.0);
                published.push(addr.0);
                publish_index.push(*index);
            }
            Instr::Hint { .. } => {}
            _ => *skipped.entry(kind).or_insert(0) += 1,
        }
    }
    origin.resize((asm.here() - ENTRY_PC) as usize, program.instrs.len() - 1);
    asm.halt();
    origin.push(usize::MAX);
    let mut origin: Vec<usize> = [usize::MAX; ENTRY_PC as usize]
        .into_iter()
        .chain(origin)
        .collect();
    if canonical {
        let mut ordered = Vec::with_capacity(public.len());
        let mut seen = std::collections::HashSet::with_capacity(public.len());
        for a in published.iter().copied().chain(public.iter().copied()) {
            if seen.insert(a) {
                ordered.push(a);
            }
        }
        public = ordered;
    } else {
        public.sort_unstable();
        public.dedup();
    }
    let mut program = asm.finish().expect("translated program");
    let mut kept = keep.to_vec();
    let mut frames: Option<(u64, usize, u64)> = None;
    if !scopes.is_empty() && std::env::var_os("FVM_OUTLINE").is_some() {
        let protected: std::collections::HashSet<u64> =
            public.iter().chain(keep).copied().collect();
        let o = outline(&program, &origin, &mut mem, scopes, &protected);
        eprintln!(
            "  outline: {} functions, {} calls, {} regions left inline, body {} instrs, program {} -> {}",
            o.functions,
            o.calls,
            o.inline,
            o.body_len,
            program.len(),
            o.program.len()
        );
        program = o.program;
        origin = o.origin;
        frames = Some((o.frames_from, o.main_start, o.frame_cells));
    }
    if std::env::var_os("FVM_MEM_COMPACT").is_some() {
        let first: Vec<u64> = if canonical {
            published
                .iter()
                .copied()
                .chain(w_cells.map(|(w, _)| w))
                .collect()
        } else {
            Vec::new()
        };
        // `FVM_COMPACT_BITDEC_LAST`: bit decompositions place their cells
        // last, so the bits the per-query code reads land in its order.
        let late: Vec<bool> = if std::env::var_os("FVM_COMPACT_BITDEC_LAST").is_some() {
            origin
                .iter()
                .map(|&o| matches!(program_lfm.instrs.get(o), Some(Instr::BitDec { .. })))
                .collect()
        } else {
            Vec::new()
        };
        let kept_and_published: Vec<u64> = keep.iter().chain(&published).copied().collect();
        (mem, public, kept, consts, const_lane3) = compact(
            &mut program,
            &mem,
            &public,
            &kept_and_published,
            &consts,
            &const_lane3,
            &first,
            frames.map(|(f, ..)| f),
            &late,
            canonical,
        );
        published = kept.split_off(keep.len());
    }
    if std::env::var_os("FVM_REROLL").is_some() {
        let from;
        (program, from) = reroll(&program, frames.map_or(ENTRY_PC as usize, |(_, m, _)| m));
        origin = from
            .into_iter()
            .map(|i| origin.get(i).copied().unwrap_or(usize::MAX))
            .collect();
    }
    if let Some((_, main_start, frame_cells)) = frames {
        set_return_addresses(&program, &mut mem, frame_cells, main_start);
    }
    Translation {
        program,
        mem,
        public,
        lfm_counts,
        origin,
        skipped,
        kept,
        published: publish_index.into_iter().zip(published).collect(),
        #[cfg(all(test, feature = "parallel"))]
        consts,
        const_lane3,
    }
}

/// Asserts `input == Σ 2^i · MEM[bits[i]]` over boolean bits, with the
/// LFM's canonicality rule: when the high 32 bits are all ones the low 32 are
/// zero, so a value has one decomposition (not also that of `x + p`). The
/// witness (the low half and `w`) is derived from the bits in `mem`. Clobbers
/// `acc`.
pub(super) fn canonical_bit_dec(
    asm: &mut Asm,
    mem: &mut Vec<FEE>,
    input: Arg,
    bits: &[u64; 64],
    acc: u8,
) {
    let bit = |i: usize| mem[bits[i] as usize].value()[0].canonical();
    let half = |h: usize| (0..32).fold(0u64, |v, i| v | (bit(32 * h + i) << i));
    let (lo, hi) = (half(0), half(1));
    let w = (FE::from(u32::MAX as u64) - FE::from(hi))
        .inv()
        .map_or(FE::zero(), |d| FE::from(lo) * d);
    mem.push(FEE::from(lo));
    let lo_cell = mem.len() as u64 - 1;
    mem.push(w.to_extension());
    let w_cell = mem.len() as u64 - 1;
    for half in 0..2 {
        asm.mov(Arg::reg(acc), Arg::imm(0));
        for (j, &a) in bits[32 * half..32 * (half + 1)].iter().enumerate() {
            let b = Arg::mem_abs(a);
            asm.fma(b, b, b, Arg::imm(0)).fma_out(
                Arg::reg(acc),
                Arg::imm_fe(FE::from(1u64 << j)),
                b,
                Arg::reg(acc),
            );
        }
        if half == 0 {
            asm.store(Arg::mem_abs(lo_cell), Arg::reg(acc));
        }
    }
    // lo == ((2^32 − 1) − hi) · w: lo = 0 when hi = 2^32 − 1.
    asm.fma(
        input,
        Arg::imm_fe(FE::from(1u64 << 32)),
        Arg::reg(acc),
        Arg::mem_abs(lo_cell),
    )
    .fma(
        Arg::mem_abs(lo_cell),
        Arg::lin(acc, -1, u32::MAX as i64),
        Arg::mem_abs(w_cell),
        Arg::imm(0),
    );
}

/// [`canonical_bit_dec`] with every bit's row the same instruction, so
/// `FVM_REROLL` rolls it: Horner from the top bit, `acc = 2·acc + b`, after
/// each bit's booleanity. The high half is stored on the way down, `x` is
/// checked whole, and `lo = x − 2^32·hi` meets the same canonicality rule:
/// `lo = (2^32 − 1)·w − hi·w`.
pub(super) fn canonical_bit_dec_horner(
    asm: &mut Asm,
    mem: &mut Vec<FEE>,
    input: Arg,
    bits: &[u64; 64],
    acc: u8,
) {
    let bit = |i: usize| mem[bits[i] as usize].value()[0].canonical();
    let half = |h: usize| (0..32).fold(0u64, |v, i| v | (bit(32 * h + i) << i));
    let (lo, hi) = (half(0), half(1));
    let w = (FE::from(u32::MAX as u64) - FE::from(hi))
        .inv()
        .map_or(FE::zero(), |d| FE::from(lo) * d);
    let mut cell = |v: FE| {
        mem.push(v.to_extension());
        mem.len() as u64 - 1
    };
    let hi_cell = cell(FE::from(hi));
    let lo_cell = cell(FE::from(lo));
    let w_cell = cell(w);
    let hw_cell = cell(FE::from(hi) * w);
    let ww_cell = cell(FE::from(u32::MAX as u64) * w);
    asm.mov(Arg::reg(acc), Arg::imm(0));
    for i in (0..64).rev() {
        let b = Arg::mem_abs(bits[i]);
        asm.fma(b, b, b, Arg::imm(0))
            .fma_out(Arg::reg(acc), Arg::imm(2), Arg::reg(acc), b);
        if i == 32 {
            asm.store(Arg::mem_abs(hi_cell), Arg::reg(acc));
        }
    }
    // x == acc; lo = x − 2^32·hi; (2^32 − 1)·w = lo + hi·w.
    asm.fma(input, Arg::imm(1), Arg::reg(acc), Arg::imm(0))
        .fma(
            Arg::mem_abs(lo_cell),
            Arg::imm_fe(-FE::from(1u64 << 32)),
            Arg::mem_abs(hi_cell),
            input,
        )
        .fma(
            Arg::mem_abs(hw_cell),
            Arg::mem_abs(hi_cell),
            Arg::mem_abs(w_cell),
            Arg::imm(0),
        )
        .fma(
            Arg::mem_abs(ww_cell),
            Arg::imm_fe(FE::from(u32::MAX as u64)),
            Arg::mem_abs(w_cell),
            Arg::imm(0),
        )
        .fma(
            Arg::mem_abs(ww_cell),
            Arg::imm(1),
            Arg::mem_abs(lo_cell),
            Arg::mem_abs(hw_cell),
        );
}

/// Keeps only the cells the program reads or publishes, renumbered densely in
/// first-use order. Every memory argument of a translation is an absolute
/// address, so renumbering is rewriting offsets.
type Compacted = (
    Vec<FEE>,
    Vec<u64>,
    Vec<u64>,
    Vec<(u64, FEE)>,
    Vec<(u64, FE)>,
);

#[allow(clippy::too_many_arguments)]
fn compact(
    program: &mut Program,
    mem: &[FEE],
    public: &[u64],
    keep: &[u64],
    consts: &[(u64, FEE)],
    const_lane3: &[(u64, FE)],
    first: &[u64],
    frames_from: Option<u64>,
    late: &[bool],
    canonical: bool,
) -> Compacted {
    let mut remap: std::collections::HashMap<u64, u64> = std::collections::HashMap::new();
    let mut dense = Vec::new();
    let mut slot = |old: u64, dense: &mut Vec<FEE>| {
        *remap.entry(old).or_insert_with(|| {
            dense.push(mem[old as usize]);
            dense.len() as u64 - 1
        })
    };
    for &a in first {
        slot(a, &mut dense);
    }
    // Two passes: the rows `late` names take their cells only after every
    // other row has, so a cell they share with the rest is placed where the
    // rest first reads it.
    for pass in [false, true] {
        for (pc, instr) in program.instrs.iter_mut().enumerate() {
            if late.get(pc).copied().unwrap_or(false) != pass {
                continue;
            }
            for arg in [&mut instr.d, &mut instr.a, &mut instr.b, &mut instr.c] {
                if arg.mem && arg.reg == FP {
                    continue;
                }
                if arg.mem {
                    assert!(
                        arg.reg == REG_ZERO && arg.scale == FE::zero(),
                        "translations address memory absolutely"
                    );
                    arg.offset = FE::from(slot(arg.offset.canonical(), &mut dense));
                }
            }
        }
    }
    let mut public: Vec<u64> = public.iter().map(|&a| slot(a, &mut dense)).collect();
    if !canonical {
        public.sort_unstable();
    }
    let kept = keep.iter().map(|&a| slot(a, &mut dense)).collect();
    let consts = consts
        .iter()
        .map(|&(a, v)| (slot(a, &mut dense), v))
        .collect();
    let const_lane3 = const_lane3
        .iter()
        .map(|&(a, v)| (slot(a, &mut dense), v))
        .collect();
    if let Some(from) = frames_from {
        // Frames stay one block, after every other cell; each call's frame
        // pointer moves with it.
        let to = dense.len() as u64;
        dense.extend_from_slice(&mem[from as usize..]);
        for instr in program.instrs.iter_mut() {
            if let Some(base) = frame_base(instr) {
                instr.c = Arg::imm_fe(FE::from(base - from + to));
            }
        }
    }
    (dense, public, kept, consts, const_lane3)
}

/// Shortest loop body worth its two rows of per-iteration overhead.
const MIN_LOOP_BODY: usize = 32;

/// Re-rolls repeated straight-line blocks into loops over the second
/// register. A block qualifies when it repeats at least three times with the
/// same instruction shapes and every memory offset moving by one common stride
/// per repetition, or not at all. Each loop is
///
/// ```text
///     j = −reps·s
/// top: body, strided offsets as MEM[j + base + reps·s]
///     j = j + s          // ZERO once j reaches 0
///     JNZ top
/// ```
///
/// Registers are only ever the accumulator inside a body and the index `j`,
/// which the body does not touch, so a chain that crosses from one repetition
/// into the next still finds the accumulator it left.
// The register check is on a per-build constant, but only fires when loops
// are asked for.
#[allow(clippy::assertions_on_constants)]
/// Returns the program and, for each of its addresses, the source address it
/// came from (`usize::MAX` for loop control).
fn reroll(program: &Program, skip: usize) -> (Program, Vec<usize>) {
    use super::isa::{Instr, REG_PC};
    use std::collections::HashMap;
    assert!(
        super::isa::N >= 2,
        "loops need a second register for their index"
    );
    let j = gpr(1);
    let src = &program.instrs;
    // `FVM_REROLL_MIN`: a shorter body (a bit decomposition's eight bits are
    // sixteen rows) for two more rows of loop control a repetition.
    let min_body = std::env::var("FVM_REROLL_MIN")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(MIN_LOOP_BODY);
    let shape = |i: &Instr| {
        let mut h = format!("{}{:?}", i.hint_out as u8, i.hint_in);
        for a in i.args() {
            h.push_str(&format!(
                "|{}:{}:{}",
                a.reg,
                a.scale.canonical(),
                a.mem as u8
            ));
            if !a.mem {
                h.push_str(&format!(":{}", a.offset.canonical()));
            }
        }
        h
    };
    let ids: Vec<u32> = {
        let mut map: HashMap<String, u32> = HashMap::new();
        src.iter()
            .map(|i| {
                let n = map.len() as u32;
                *map.entry(shape(i)).or_insert(n)
            })
            .collect()
    };
    // Candidate periods: the commonest distances between repeats of a window.
    const K: usize = 32;
    let mut last: HashMap<&[u32], usize> = HashMap::new();
    let mut dist: HashMap<usize, usize> = HashMap::new();
    for i in 0..src.len().saturating_sub(K) {
        if let Some(&q) = last.get(&ids[i..i + K]) {
            *dist.entry(i - q).or_insert(0) += 1;
        }
        last.insert(&ids[i..i + K], i);
    }
    // A short period (a uniform run of rows) counts for its first multiple
    // long enough to be a body: the run's nearest repeat is always the short one.
    let mut widened: HashMap<usize, usize> = HashMap::new();
    for (p, c) in dist {
        *widened.entry(p * min_body.div_ceil(p)).or_insert(0) += c;
    }
    let mut periods: Vec<(usize, usize)> = widened
        .into_iter()
        .filter(|&(p, _)| p >= min_body)
        .collect();
    periods.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    periods.truncate(16);
    let offsets =
        |i: &Instr| -> [Option<u64>; 4] { i.args().map(|a| a.mem.then(|| a.offset.canonical())) };
    // The common stride between repetition 0 and repetition `k` at `i`.
    let stride = |i: usize, p: usize, k: usize| -> Option<i128> {
        let mut s: Option<i128> = None;
        // An outlined call's rows address its frame through the index
        // register: a loop over them would overwrite the frame pointer.
        if (i..i + (k + 1) * p).any(|q| src[q].args().iter().any(|a| a.reg == j)) {
            return None;
        }
        for t in 0..p {
            if ids[i + t] != ids[i + t + k * p] {
                return None;
            }
            for (x, y) in offsets(&src[i + t])
                .into_iter()
                .zip(offsets(&src[i + t + k * p]))
            {
                if let (Some(x), Some(y)) = (x, y) {
                    let d = y as i128 - x as i128;
                    if d != 0 {
                        match s {
                            None => s = Some(d),
                            Some(v) if v == d => {}
                            _ => return None,
                        }
                    }
                }
            }
        }
        Some(s.unwrap_or(0))
    };
    let mut out: Vec<Instr> = src[..skip].to_vec();
    let mut from: Vec<usize> = (0..skip).collect();
    let mut i = skip;
    while i < src.len() {
        let mut best: Option<(usize, usize, i128)> = None;
        for &(p, _) in &periods {
            if i + 3 * p > src.len() {
                continue;
            }
            let Some(s) = stride(i, p, 1).filter(|&s| s != 0) else {
                continue;
            };
            let mut reps = 2;
            while i + (reps + 1) * p <= src.len() && stride(i, p, reps) == Some(s * reps as i128) {
                reps += 1;
            }
            if reps >= 3 && best.is_none_or(|(bp, br, _)| reps * p > bp * br) {
                best = Some((p, reps, s));
            }
        }
        let Some((p, reps, s)) = best else {
            out.push(src[i]);
            from.push(i);
            i += 1;
            continue;
        };
        let span = s * reps as i128;
        out.push(Instr::new(
            Arg::reg(j),
            Arg::imm(0),
            Arg::imm(0),
            Arg::imm_fe(FE::from(-span as i64)),
            true,
        ));
        from.push(usize::MAX);
        let top = out.len() as i64;
        for t in 0..p {
            let mut instr = src[i + t];
            let moved = offsets(&src[i + t + p]);
            for (k, arg) in [&mut instr.d, &mut instr.a, &mut instr.b, &mut instr.c]
                .into_iter()
                .enumerate()
            {
                if let (Some(x), Some(y)) = (offsets(&src[i + t])[k], moved[k])
                    && y != x
                {
                    *arg = Arg::mem(j, x as i64 + span as i64);
                }
            }
            out.push(instr);
            from.push(i + t);
        }
        out.push(Instr::new(
            Arg::reg(j),
            Arg::imm(1),
            Arg::reg(j),
            Arg::imm_fe(FE::from(s as i64)),
            true,
        ));
        // JNZ top: PC' = ZERO·(PC − top) + (ZERO + top).
        out.push(Instr::new(
            Arg::reg(REG_PC),
            Arg::reg(REG_ZERO),
            Arg::lin(REG_PC, 1, -top),
            Arg::lin(REG_ZERO, 1, top),
            true,
        ));
        from.extend([usize::MAX; 2]);
        i += reps * p;
    }
    debug_assert_eq!(from.len(), out.len());
    (Program { instrs: out }, from)
}

/// Frame pointer of outlined calls. Translations use only the accumulator, and
/// `FVM_REROLL` takes this register for its loop index after outlining, where
/// no call sits in a loop: a call's frame base is an immediate of its own.
const FP: u8 = 3; // gpr(1)

/// `fp = base`, a call's first instruction: its frame base.
fn frame_base(i: &FInstr) -> Option<u64> {
    let imm = |a: &Arg| !a.mem && a.reg == REG_ZERO && a.scale == FE::zero();
    (i.hint_out && !i.d.mem && i.d.reg == FP && i.d.scale == FE::one() && imm(&i.a) && imm(&i.c))
        .then(|| i.c.offset.canonical())
}

/// Writes each call's return address into its frame, from the call's final
/// position: `MEM[fp + ra] == PC + 2` stores it at run time. A call's frame
/// lies in the last `frame_cells` cells and its store is followed by the jump
/// into a body, below `main_start`; `FVM_REROLL`'s loop control also sets
/// this register and is not a call.
fn set_return_addresses(program: &Program, mem: &mut [FEE], frame_cells: u64, main_start: usize) {
    use super::isa::REG_PC;
    let frames = mem.len() as u64 - frame_cells..mem.len() as u64;
    let into_body = |i: &FInstr| {
        i.hint_out
            && !i.d.mem
            && i.d.reg == REG_PC
            && !i.c.mem
            && i.c.reg == REG_ZERO
            && i.c.offset.canonical() < main_start as u64
    };
    let mut base = None;
    for (pc, i) in program.instrs.iter().enumerate() {
        if let Some(b) = frame_base(i) {
            base = frames.contains(&b).then_some(b);
        } else if let Some(b) = base
            && program.instrs.get(pc + 1).is_some_and(into_body)
            && i.d.mem
            && i.d.reg == FP
            && !i.c.mem
            && i.c.reg == REG_PC
            && i.c.offset == FE::from(2u64)
        {
            let ra = b + i.d.offset.canonical();
            mem[ra as usize] = FEE::from(pc as u64 + 2);
        }
    }
}

struct Outlined {
    program: Program,
    origin: Vec<usize>,
    /// First memory cell of the frames, which come after every other cell.
    frames_from: u64,
    /// First address after the bodies, where the main code starts.
    main_start: usize,
    /// Cells the frames take, at the end of memory.
    frame_cells: u64,
    functions: usize,
    calls: usize,
    inline: usize,
    body_len: usize,
}

/// Outlines the regions `scopes` names: a function is the first region of a
/// name, its memory arguments rewritten as `MEM[fp + slot]`, and every region
/// of that name with the same instruction shapes and the same pattern of
/// addresses becomes a call
///
/// ```text
///     fp = frame                     // a fresh block of cells
///     MEM[fp + s] == MEM[x_s]        // each slot whose cell lives outside
///     MEM[fp + ra] == PC + 2
///     PC = body                      // ... body ...  PC = MEM[fp + ra]
/// ```
///
/// An address the same in every region stays absolute (constants). The
/// equalities tie each frame cell to the cell it stands for, so the body
/// constrains the same values the region did; cells only the region touches
/// move into the frame. A region whose accumulator is live across either end,
/// or whose shapes differ, stays inline. Bodies sit after the start `nop`,
/// behind one jump to the main code.
#[allow(clippy::type_complexity)]
fn outline(
    program: &Program,
    origin: &[usize],
    mem: &mut Vec<FEE>,
    scopes: &[Scope],
    protected: &std::collections::HashSet<u64>,
) -> Outlined {
    use super::isa::{REG_PC, gpr_index};
    use std::collections::{BTreeMap, HashMap};
    let src = &program.instrs;
    let frames_from = mem.len() as u64;
    let acc = gpr(0);
    let entry = ENTRY_PC as usize;
    let last = src.len() - 1; // the closing halt
    // Program range of each scope: origins are nondecreasing over the body.
    let range = |sc: &Scope| -> (usize, usize) {
        let body = &origin[entry..last];
        let lo = entry + body.partition_point(|&o| o < sc.start);
        let hi = entry + body.partition_point(|&o| o < sc.end);
        (lo, hi)
    };
    // Whether the accumulator's value on entry to `pc` is read before written.
    let mut live = vec![false; src.len() + 1];
    for pc in (entry..src.len()).rev() {
        let i = &src[pc];
        let reads = [i.a, i.b, i.c]
            .iter()
            .any(|a| a.reg == acc && a.scale != FE::zero())
            || (i.d.reg == acc && i.d.scale != FE::zero() && (i.d.mem || !i.hint_out));
        let writes = i.hint_out && !i.d.mem && i.d.reg == acc;
        let hinted = gpr_index(acc).is_some_and(|g| i.hint_in[g]);
        live[pc] = !hinted && (reads || (live[pc + 1] && !writes));
    }
    let shape_eq = |x: &FInstr, y: &FInstr| {
        x.hint_out == y.hint_out
            && x.hint_in == y.hint_in
            && x.args().iter().zip(y.args()).all(|(a, b)| {
                a.reg == b.reg
                    && a.scale == b.scale
                    && a.mem == b.mem
                    && (a.mem || a.offset == b.offset)
            })
    };
    let addrs = |i: &FInstr| -> [Option<u64>; 4] {
        i.args().map(|a| {
            a.mem.then(|| {
                assert!(
                    a.reg == REG_ZERO && a.scale == FE::zero(),
                    "absolute before outlining"
                );
                a.offset.canonical()
            })
        })
    };
    let mut access: HashMap<u64, u32> = HashMap::new();
    for i in &src[entry..] {
        for a in addrs(i).into_iter().flatten() {
            *access.entry(a).or_default() += 1;
        }
    }

    // Regions by name, valid ones only.
    let mut by_name: BTreeMap<&str, Vec<(usize, usize)>> = BTreeMap::new();
    let mut inline = 0;
    for sc in scopes {
        let (lo, hi) = range(sc);
        if hi <= lo {
            continue;
        }
        let clean = !live[lo]
            && !live[hi]
            && src[lo..hi]
                .iter()
                .all(|i| i.args().iter().all(|a| a.reg != FP && a.reg != REG_PC));
        if clean {
            by_name.entry(sc.name.as_str()).or_default().push((lo, hi));
        } else {
            if std::env::var_os("FVM_OUTLINE_DEBUG").is_some() {
                eprintln!(
                    "    outline: {} at {lo}..{hi} inline (live in {}, live out {}, regs {})",
                    sc.name,
                    live[lo],
                    live[hi],
                    !src[lo..hi]
                        .iter()
                        .all(|i| i.args().iter().all(|a| a.reg != FP && a.reg != REG_PC))
                );
            }
            inline += 1;
        }
    }

    struct Func {
        body: Vec<FInstr>,
        body_origin: Vec<usize>,
        slots: usize,
    }
    // A call: (region start, region end, function, frame base, copies (slot, cell)).
    let mut calls: Vec<(usize, usize, usize, u64, Vec<(usize, u64)>)> = Vec::new();
    let mut funcs: Vec<Func> = Vec::new();
    // Regions of one name split into classes of equal shapes: the first
    // region of a name can differ from the rest (its results are shared
    // with them, so the translator keeps them in memory).
    let mut classes: Vec<Vec<(usize, usize)>> = Vec::new();
    for regions in by_name.values() {
        let mut mine: Vec<Vec<(usize, usize)>> = Vec::new();
        for &(lo, hi) in regions {
            let same = |&(r0, r1): &(usize, usize)| {
                r1 - r0 == hi - lo && (0..hi - lo).all(|t| shape_eq(&src[r0 + t], &src[lo + t]))
            };
            match mine.iter_mut().find(|c| same(&c[0])) {
                Some(c) => c.push((lo, hi)),
                None => mine.push(vec![(lo, hi)]),
            }
        }
        classes.extend(mine);
    }
    for regions in &classes {
        let (d0, d1) = regions[0];
        let len = d1 - d0;
        let mut members: Vec<(usize, usize)> = regions
            .iter()
            .copied()
            .filter(|&(lo, hi)| {
                hi - lo == len && (0..len).all(|t| shape_eq(&src[d0 + t], &src[lo + t]))
            })
            .collect();
        inline += regions.len() - members.len();
        if std::env::var_os("FVM_OUTLINE_DEBUG").is_some() {
            let lens: Vec<usize> = regions.iter().map(|(lo, hi)| hi - lo).collect();
            eprintln!(
                "    outline: {} regions, lengths {:?}, {} match the first",
                regions.len(),
                &lens[..lens.len().min(8)],
                members.len()
            );
            if let Some(&(lo, _)) = regions.get(1) {
                let (d0, _) = regions[0];
                let diffs: Vec<usize> = (0..lens[0].min(lens[1]))
                    .filter(|&t| !shape_eq(&src[d0 + t], &src[lo + t]))
                    .collect();
                eprintln!(
                    "    outline: {} differing places, first {:?}",
                    diffs.len(),
                    &diffs[..diffs.len().min(5)]
                );
                for &t in diffs.iter().take(3) {
                    eprintln!("      {:?}\n      {:?}", src[d0 + t], src[lo + t]);
                }
            }
        }
        loop {
            if members.len() < 2 {
                inline += members.len();
                break;
            }
            // An address of the definition stays absolute when every member
            // has it at every place the definition does.
            let mut places: HashMap<u64, Vec<(usize, usize)>> = HashMap::new();
            for t in 0..len {
                for (k, a) in addrs(&src[d0 + t]).into_iter().enumerate() {
                    if let Some(a) = a {
                        places.entry(a).or_default().push((t, k));
                    }
                }
            }
            let absolute: std::collections::HashSet<u64> = places
                .iter()
                .filter(|(a, ps)| {
                    members.iter().all(|&(lo, _)| {
                        ps.iter().all(|&(t, k)| addrs(&src[lo + t])[k] == Some(**a))
                    })
                })
                .map(|(a, _)| *a)
                .collect();
            let mut slot_of: HashMap<u64, usize> = HashMap::new();
            for t in 0..len {
                for a in addrs(&src[d0 + t]).into_iter().flatten() {
                    if !absolute.contains(&a) {
                        let n = slot_of.len();
                        slot_of.entry(a).or_insert(n);
                    }
                }
            }
            let slots = slot_of.len();
            // Each member's cell per slot, or None when its pattern differs.
            let cells = |lo: usize| -> Option<Vec<u64>> {
                let mut cell: Vec<Option<u64>> = vec![None; slots];
                for t in 0..len {
                    for (a, b) in addrs(&src[d0 + t]).into_iter().zip(addrs(&src[lo + t])) {
                        let (Some(a), Some(b)) = (a, b) else { continue };
                        match slot_of.get(&a) {
                            None if a != b => return None,
                            None => {}
                            Some(&s) => match cell[s] {
                                None => cell[s] = Some(b),
                                Some(c) if c != b => return None,
                                _ => {}
                            },
                        }
                    }
                }
                cell.into_iter().collect()
            };
            let resolved: Vec<_> = members.iter().map(|&(lo, _)| cells(lo)).collect();
            if resolved.iter().any(Option::is_none) {
                let before = members.len();
                members = members
                    .into_iter()
                    .zip(&resolved)
                    .filter(|(_, c)| c.is_some())
                    .map(|(m, _)| m)
                    .collect();
                inline += before - members.len();
                if members.first() != Some(&(d0, d1)) {
                    inline += members.len();
                    break;
                }
                continue;
            }
            let f = funcs.len();
            let body = src[d0..d1]
                .iter()
                .map(|i| {
                    let mut i = *i;
                    for arg in [&mut i.d, &mut i.a, &mut i.b, &mut i.c] {
                        if arg.mem
                            && let Some(&s) = slot_of.get(&arg.offset.canonical())
                        {
                            *arg = Arg::mem(FP, s as i64);
                        }
                    }
                    i
                })
                .collect();
            funcs.push(Func {
                body,
                body_origin: origin[d0..d1].to_vec(),
                slots,
            });
            for (&(lo, hi), cells) in members.iter().zip(resolved) {
                let cells = cells.expect("resolved");
                let mut inside: HashMap<u64, u32> = HashMap::new();
                for i in &src[lo..hi] {
                    for a in addrs(i).into_iter().flatten() {
                        *inside.entry(a).or_default() += 1;
                    }
                }
                let base = mem.len() as u64;
                let values: Vec<FEE> = cells.iter().map(|&x| mem[x as usize]).collect();
                mem.extend(values);
                mem.push(FEE::zero()); // the return address, set once placed
                let copies = cells
                    .iter()
                    .enumerate()
                    .filter(|&(_, x)| protected.contains(x) || access[x] > inside[x])
                    .map(|(s, &x)| (s, x))
                    .collect();
                calls.push((lo, hi, f, base, copies));
            }
            break;
        }
    }
    calls.sort_unstable_by_key(|c| c.0);

    let mut out: Vec<FInstr> = src[..entry].to_vec();
    let mut from: Vec<usize> = vec![usize::MAX; entry];
    let jump = |target: usize| {
        FInstr::new(
            Arg::reg(REG_PC),
            Arg::imm(0),
            Arg::imm(0),
            Arg::imm(target as i64),
            true,
        )
    };
    out.push(jump(0)); // patched below
    from.push(usize::MAX);
    let mut starts = Vec::with_capacity(funcs.len());
    let mut body_len = 0;
    for f in &funcs {
        starts.push(out.len());
        out.extend_from_slice(&f.body);
        from.extend_from_slice(&f.body_origin);
        out.push(FInstr::new(
            Arg::reg(REG_PC),
            Arg::imm(0),
            Arg::imm(0),
            Arg::mem(FP, f.slots as i64),
            true,
        ));
        from.push(usize::MAX);
        body_len += f.body.len() + 1;
    }
    let main_start = out.len();
    out[entry] = jump(main_start);
    let mut next_call = 0;
    let mut pc = entry;
    while pc < src.len() {
        if next_call < calls.len() && calls[next_call].0 == pc {
            let (_, hi, f, base, ref copies) = calls[next_call];
            out.push(FInstr::new(
                Arg::reg(FP),
                Arg::imm(0),
                Arg::imm(0),
                Arg::imm(base as i64),
                true,
            ));
            for &(s, x) in copies {
                out.push(FInstr::new(
                    Arg::mem(FP, s as i64),
                    Arg::imm(0),
                    Arg::imm(0),
                    Arg::mem_abs(x),
                    false,
                ));
            }
            out.push(FInstr::new(
                Arg::mem(FP, funcs[f].slots as i64),
                Arg::imm(0),
                Arg::imm(0),
                Arg::lin(REG_PC, 1, 2),
                false,
            ));
            out.push(jump(starts[f]));
            from.resize(out.len(), usize::MAX);
            next_call += 1;
            pc = hi;
            continue;
        }
        out.push(src[pc]);
        from.push(origin[pc]);
        pc += 1;
    }
    Outlined {
        program: Program { instrs: out },
        origin: from,
        frame_cells: mem.len() as u64 - frames_from,
        frames_from,
        main_start,
        functions: funcs.len(),
        calls: calls.len(),
        inline,
        body_len,
    }
}

#[cfg(test)]
mod reroll_tests {
    use super::*;

    /// One Horner bit decomposition over fresh cells rolls into loops: its
    /// sixty-four bit rows are one instruction shape over strided cells.
    #[test]
    fn a_horner_bit_decomposition_rerolls() {
        let x: u64 = 0x0123_4567_89AB_CDEF;
        let mut mem: Vec<FEE> = vec![FEE::from(x)];
        let bits: [u64; 64] = core::array::from_fn(|i| {
            mem.push(FEE::from((x >> i) & 1));
            mem.len() as u64 - 1
        });
        let mut asm = Asm::new();
        for _ in 0..3 {
            canonical_bit_dec_horner(&mut asm, &mut mem, Arg::mem_abs(0), &bits, gpr(0));
        }
        asm.halt();
        let program = asm.finish().expect("a program");
        // SAFETY: test-local flag, read once by `reroll`.
        unsafe { std::env::set_var("FVM_REROLL_MIN", "16") };
        let (rolled, _) = reroll(&program, ENTRY_PC as usize);
        eprintln!(
            "bit decomposition: {} -> {} instructions",
            program.len(),
            rolled.len()
        );
        assert!(
            rolled.len() < program.len() / 2,
            "{} -> {}",
            program.len(),
            rolled.len()
        );
    }
}
