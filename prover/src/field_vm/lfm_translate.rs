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
    let plan = if opt {
        plan(program)
    } else {
        vec![Plan::Plain; program.instrs.len()]
    };
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
    let mut lfm_counts = BTreeMap::new();
    let mut skipped = BTreeMap::new();

    for (idx, instr) in program.instrs.iter().enumerate() {
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
                if value[1] == FE::zero() && value[2] == FE::zero() && !opt {
                    asm.store(cell(out), Arg::imm_fe(value[0]));
                } else {
                    public.push(out.0);
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
    if std::env::var_os("FVM_REROLL").is_some() {
        program = reroll(&program);
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
fn reroll(program: &Program) -> Program {
    use super::isa::{ENTRY_PC, Instr, REG_PC};
    use std::collections::HashMap;
    assert!(
        super::isa::N >= 2,
        "loops need a second register for their index"
    );
    let j = gpr(1);
    let src = &program.instrs;
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
    let mut periods: Vec<(usize, usize)> = dist
        .into_iter()
        .filter(|&(p, _)| p >= MIN_LOOP_BODY)
        .collect();
    periods.sort_by(|a, b| b.1.cmp(&a.1));
    periods.truncate(16);
    let offsets =
        |i: &Instr| -> [Option<u64>; 4] { i.args().map(|a| a.mem.then(|| a.offset.canonical())) };
    // The common stride between repetition 0 and repetition `k` at `i`.
    let stride = |i: usize, p: usize, k: usize| -> Option<i128> {
        let mut s: Option<i128> = None;
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
    let mut out: Vec<Instr> = src[..ENTRY_PC as usize].to_vec();
    let mut i = ENTRY_PC as usize;
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
        i += reps * p;
    }
    Program { instrs: out }
}
