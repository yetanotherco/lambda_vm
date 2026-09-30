//! The 3MI split of an epoch verifier: `verify_f` on the Field VM, `verify_b`
//! on the RISC-V VM, as the guest `executor/programs/rust/split_verify_b`.
//!
//! [`cut`] gives the RISC-V half the hashing and byte glue
//! ([`is_hash_side`]) but for selects between field constants, plus the
//! assertions on hash outputs (Merkle roots against the commitments, proof of
//! work): `x − y` over hash outputs, hints and constants, then `d / 0`, when
//! only the RISC-V half reads the result. Hints, their lanes and constants go
//! to both halves; so do the bits of hints (query indices) with `fvm_bits`,
//! which the Field VM then decomposes itself.
//!
//! What crosses, lanes 0..3 of each cell:
//! - the input: hints, and the lanes of hints, both halves read (the spec's
//!   shared PAGE input);
//! - the record: `out`, Field VM results the RISC-V half reads, then `back`,
//!   RISC-V results the Field VM reads or publishes (the Fiat–Shamir
//!   challenges and query bits).
//!
//! The Field VM proof publishes every crossing cell; the guest commits
//! `keccak256(program) ‖ keccak256(input) ‖ keccak256(record)`. [`coupled`] is
//! the check the next layer makes: both describe one input and one record.

use std::collections::{BTreeMap, HashMap};

use tiny_keccak::{Hasher, Keccak};

use super::hash_side::is_hash_side;
use super::prove::PublicCell;
use crate::lfm::compiler::LfmProgram;
use crate::lfm::executor::LfmExecution;
use crate::lfm::instr::{Addr, BaseOp, ExtOp, HashMode, Instr, KeccakMode};
use crate::lfm::word::LfmWord;
use crate::tables::types::FE;

const VERSION: u32 = 2;

pub struct Cut {
    /// Per program instruction: whether the RISC-V half runs it (the input
    /// both halves hold excluded).
    pub riscv: Vec<bool>,
    /// Field instructions moved to the RISC-V half.
    pub moved: usize,
    pub out: Vec<u64>,
    pub back: Vec<u64>,
    pub input: Vec<u64>,
    /// Each `input` cell's hint in `hints` and its lane there.
    pub input_src: Vec<(usize, usize)>,
    /// The guest's program words, and the hint values it reads, in order.
    pub words: Vec<u32>,
    pub hints: Vec<LfmWord>,
    /// The values of `out`, `back` and `input` in the LFM's execution.
    pub values: HashMap<u64, LfmWord>,
    /// Guest instructions by kind, and where each `back` cell comes from.
    pub counts: BTreeMap<&'static str, usize>,
    pub back_origin: BTreeMap<String, usize>,
    pub out_origin: BTreeMap<String, usize>,
    pub cells: usize,
}

fn kind(i: &Instr) -> &'static str {
    match i {
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
    }
}

pub fn cut(
    program: &LfmProgram,
    exec: &LfmExecution,
    arenas: &[Vec<LfmWord>],
    fvm_bits: bool,
) -> Cut {
    let n = program.num_addrs as usize;
    let instrs = &program.instrs;
    let mut buf = Vec::new();
    let mut writer: Vec<usize> = vec![usize::MAX; n];
    let mut readers: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (k, i) in instrs.iter().enumerate() {
        buf.clear();
        i.writes_into(&mut buf);
        for a in &buf {
            writer[a.0 as usize] = k;
        }
        buf.clear();
        i.reads_into(&mut buf);
        for a in &buf {
            readers[a.0 as usize].push(k);
        }
    }
    // Hints, constants and the unpacking of hints are input: both halves
    // hold them. With `fvm_bits` so are the bits of hints (query indices),
    // which the Field VM then decomposes itself; the byte-swapped halves stay
    // the RISC-V half's.
    let mut copied = vec![false; instrs.len()];
    for (k, i) in instrs.iter().enumerate() {
        let of_input = |a: &Addr| {
            let w = writer[a.0 as usize];
            copied[w] && !matches!(instrs[w], Instr::Const { .. })
        };
        copied[k] = match i {
            Instr::Hint { .. } | Instr::Const { .. } => true,
            Instr::Unpack { input, .. } => of_input(input),
            Instr::BitDec { input, halves, .. } => {
                fvm_bits
                    && of_input(input)
                    && halves.iter().flatten().all(|(a, _)| {
                        readers[a.0 as usize]
                            .iter()
                            .all(|&r| is_hash_side(&instrs[r]))
                    })
            }
            _ => false,
        };
    }
    // A select between two field constants is a field select (x = g or 1 by
    // a query bit), which the Field VM does in three rows.
    let field_const = |a: &Addr| {
        matches!(&instrs[writer[a.0 as usize]], Instr::Const { value, .. } if value[3] == FE::zero())
    };
    let mut riscv: Vec<bool> = instrs
        .iter()
        .zip(&copied)
        .map(|(i, &c)| {
            !c && is_hash_side(i)
                && !matches!(i, Instr::Select { in_l, in_r, .. } if field_const(in_l) && field_const(in_r))
        })
        .collect();
    // Candidates: assertions on hash outputs, `x − y` with `x` or `y` a hash
    // output and the other a hash output or input, then `d / 0`.
    let mut cand = vec![false; instrs.len()];
    let zero_const = |a: &Addr| {
        matches!(&instrs[writer[a.0 as usize]], Instr::Const { value, .. }
            if value.iter().all(|x| *x == FE::zero()))
    };
    for (k, i) in instrs.iter().enumerate() {
        let (sub, div) = match i {
            Instr::BaseAlu { op, a, b, .. } => (
                (*op == BaseOp::Sub).then_some((a, b)),
                (*op == BaseOp::Div).then_some((a, b)),
            ),
            Instr::ExtAlu { op, a, b, .. } => (
                (*op == ExtOp::Sub).then_some((a, b)),
                (*op == ExtOp::Div).then_some((a, b)),
            ),
            _ => (None, None),
        };
        let hashed = |a: &Addr| riscv[writer[a.0 as usize]];
        let known = |a: &Addr| hashed(a) || copied[writer[a.0 as usize]];
        if let Some((a, b)) = sub {
            cand[k] = known(a) && known(b) && (hashed(a) || hashed(b));
        } else if let Some((a, b)) = div {
            cand[k] = zero_const(b) && (hashed(a) || cand[writer[a.0 as usize]]);
        }
    }
    // Kept only when every reader of the result runs on the RISC-V half.
    let mut moved = 0;
    for k in (0..instrs.len()).rev() {
        if !cand[k] {
            continue;
        }
        buf.clear();
        instrs[k].writes_into(&mut buf);
        if buf
            .iter()
            .all(|a| readers[a.0 as usize].iter().all(|&r| riscv[r]))
        {
            riscv[k] = true;
            moved += 1;
        }
    }

    let mut riscv_reads = vec![false; n];
    let mut field_reads = vec![false; n];
    for (k, i) in instrs.iter().enumerate() {
        if copied[k] {
            continue;
        }
        buf.clear();
        i.reads_into(&mut buf);
        for a in &buf {
            if riscv[k] {
                riscv_reads[a.0 as usize] = true;
            } else {
                field_reads[a.0 as usize] = true;
            }
        }
    }
    // The input the guest holds: what it reads, and the hints that unpack.
    let mut emit = vec![false; instrs.len()];
    for k in (0..instrs.len()).rev() {
        if !copied[k] {
            continue;
        }
        buf.clear();
        instrs[k].writes_into(&mut buf);
        emit[k] |= buf.iter().any(|a| riscv_reads[a.0 as usize]);
        if let (true, Instr::Unpack { input, .. } | Instr::BitDec { input, .. }) =
            (emit[k], &instrs[k])
        {
            emit[writer[input.0 as usize]] = true;
        }
    }
    let (mut out, mut back, mut input) = (Vec::new(), Vec::new(), Vec::new());
    for a in 0..n {
        let k = writer[a];
        if k >= instrs.len() {
            continue;
        }
        if copied[k] {
            if emit[k]
                && field_reads[a]
                && matches!(instrs[k], Instr::Hint { .. } | Instr::Unpack { .. })
            {
                input.push(a as u64);
            }
        } else if riscv[k] {
            if field_reads[a] {
                back.push(a as u64);
            }
        } else if riscv_reads[a] {
            out.push(a as u64);
        }
    }

    let mut dense: HashMap<u64, u32> = HashMap::new();
    for (k, &a) in out.iter().enumerate() {
        dense.insert(a, k as u32);
    }
    let read = |dense: &HashMap<u64, u32>, a: &Addr| -> u32 {
        *dense
            .get(&a.0)
            .unwrap_or_else(|| panic!("read before write at {}", a.0))
    };
    let write = |dense: &mut HashMap<u64, u32>, a: &Addr| -> u32 {
        let n = dense.len() as u32;
        assert!(dense.insert(a.0, n).is_none(), "double write at {}", a.0);
        n
    };
    let mut body: Vec<u32> = Vec::new();
    let mut hints = Vec::new();
    let mut counts = BTreeMap::new();
    let mut n_instr = 0u32;
    let mut shared_order = Vec::new();
    let mut input_src = Vec::new();
    let mut hint_at: HashMap<u64, usize> = HashMap::new();
    for (k, i) in instrs.iter().enumerate() {
        if !riscv[k] && !emit[k] {
            continue;
        }
        n_instr += 1;
        *counts.entry(kind(i)).or_insert(0) += 1;
        match i {
            Instr::Hash { mode, ins, outs, .. } => {
                body.push(1);
                body.push(match mode {
                    HashMode::Compress => 0,
                    HashMode::Transcript => 1,
                    HashMode::Leaf => 2,
                    HashMode::Permute => 3,
                });
                for a in &ins[..mode.num_input_cells()] {
                    body.push(read(&dense, a));
                }
                for a in &outs[..mode.num_output_cells()] {
                    body.push(write(&mut dense, a));
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
                body.push(2);
                for a in [bit, in_l, in_r] {
                    body.push(read(&dense, a));
                }
                for a in [out_l, out_r] {
                    body.push(write(&mut dense, a));
                }
            }
            Instr::Pack { lanes, out, .. } => {
                body.push(3);
                for a in lanes {
                    body.push(read(&dense, a));
                }
                body.push(write(&mut dense, out));
            }
            Instr::Unpack { input, outs, .. } => {
                body.push(4);
                body.push(read(&dense, input));
                let mut mask = 0;
                for (lane, a) in outs.iter().enumerate() {
                    body.push(write(&mut dense, a));
                    if copied[k] && field_reads[a.0 as usize] {
                        mask |= 1 << lane;
                        shared_order.push(a.0);
                        input_src.push((hint_at[&input.0], lane));
                    }
                }
                body.push(mask);
            }
            Instr::BitDec {
                input,
                bits,
                halves,
            } => {
                body.push(5);
                body.push(read(&dense, input));
                body.push(bits.len() as u32);
                for (a, _) in bits {
                    body.push(write(&mut dense, a));
                }
                body.push(halves.is_some() as u32);
                if let Some(hs) = halves {
                    for (a, _) in hs {
                        body.push(write(&mut dense, a));
                    }
                }
            }
            Instr::KeccakF(op) => {
                body.push(6);
                let absorb = op.mode == KeccakMode::Absorb;
                body.push(absorb as u32);
                for a in &op.ins {
                    body.push(read(&dense, a));
                }
                if absorb {
                    for a in &op.block {
                        body.push(read(&dense, a));
                    }
                }
                for a in &op.outs {
                    body.push(write(&mut dense, a));
                }
                body.push(op.rev.is_some() as u32);
                if let Some(rev) = &op.rev {
                    for a in &rev.outs {
                        body.push(write(&mut dense, a));
                    }
                }
            }
            Instr::Hint {
                arena, index, out, ..
            } => {
                body.push(7);
                body.push(write(&mut dense, out));
                let shared = field_reads[out.0 as usize];
                body.push(shared as u32);
                if shared {
                    shared_order.push(out.0);
                    input_src.push((hints.len(), 0));
                }
                hint_at.insert(out.0, hints.len());
                hints.push(arenas[*arena as usize][*index as usize]);
            }
            Instr::Const { out, value, .. } => {
                body.push(8);
                body.push(write(&mut dense, out));
                for x in value {
                    let v = x.canonical();
                    body.push(v as u32);
                    body.push((v >> 32) as u32);
                }
            }
            Instr::BaseAlu {
                op, out, a, b, c, ..
            } => {
                body.push(9);
                body.push(match op {
                    BaseOp::Add => 0,
                    BaseOp::Sub => 1,
                    BaseOp::Mul => 2,
                    BaseOp::Div => 3,
                    BaseOp::MulAdd => 4,
                });
                body.push(read(&dense, a));
                body.push(read(&dense, b));
                if *op == BaseOp::MulAdd {
                    body.push(read(&dense, c));
                }
                body.push(write(&mut dense, out));
            }
            Instr::ExtAlu {
                op, out, a, b, c, ..
            } => {
                body.push(10);
                body.push(match op {
                    ExtOp::Add => 0,
                    ExtOp::Sub => 1,
                    ExtOp::Mul => 2,
                    ExtOp::Div => 3,
                    ExtOp::MulAdd => 4,
                    ExtOp::MulBase => 5,
                });
                body.push(read(&dense, a));
                body.push(read(&dense, b));
                if *op == ExtOp::MulAdd {
                    body.push(read(&dense, c));
                }
                body.push(write(&mut dense, out));
            }
            other => panic!("the guest has no {}", kind(other)),
        }
    }
    let mut sorted = shared_order.clone();
    sorted.sort_unstable();
    assert_eq!(sorted, input, "the guest holds every shared input cell");
    let input = shared_order;
    let mut words = vec![
        VERSION,
        dense.len() as u32,
        out.len() as u32,
        back.len() as u32,
        n_instr,
    ];
    for a in &back {
        words.push(read(&dense, &Addr(*a)));
    }
    words.extend(body);

    let word = |a: u64| exec.memory.get(Addr(a)).expect("an executed cell");
    let values = out
        .iter()
        .chain(&back)
        .chain(&input)
        .map(|&a| (a, word(a)))
        .collect();
    let mut back_origin = BTreeMap::new();
    for &a in &back {
        let w = &instrs[writer[a as usize]];
        let key = format!("{}<{}", kind(w), origin(instrs, &writer, a));
        *back_origin.entry(key).or_insert(0) += 1;
    }
    let mut out_origin = BTreeMap::new();
    for &a in &out {
        let r = readers[a as usize]
            .iter()
            .find(|&&r| riscv[r])
            .map_or("none", |&r| kind(&instrs[r]));
        let key = format!("{}>{}", kind(&instrs[writer[a as usize]]), r);
        *out_origin.entry(key).or_insert(0) += 1;
    }
    Cut {
        out_origin,
        riscv,
        moved,
        out,
        back,
        input,
        input_src,
        words,
        hints,
        values,
        counts,
        back_origin,
        cells: dense.len(),
    }
}

/// The nearest hash above a cell, following first operands.
fn origin(instrs: &[Instr], writer: &[usize], mut a: u64) -> &'static str {
    for _ in 0..64 {
        let next = match &instrs[writer[a as usize]] {
            Instr::Hash { mode, .. } => {
                return match mode {
                    HashMode::Compress => "compress",
                    HashMode::Transcript => "transcript",
                    HashMode::Leaf => "leaf",
                    HashMode::Permute => "permute",
                };
            }
            Instr::KeccakF(_) => return "keccak",
            Instr::Hint { .. } => return "hint",
            Instr::Const { .. } => return "const",
            Instr::Unpack { input, .. } | Instr::BitDec { input, .. } => input,
            Instr::Select { in_l, .. } => in_l,
            Instr::Pack { lanes, .. } => &lanes[0],
            Instr::BaseAlu { a, .. } | Instr::ExtAlu { a, .. } => a,
            _ => return "other",
        };
        a = next.0;
    }
    "deep"
}

impl Cut {
    /// `out`, `back`, then `input`: the cells the Field VM publishes.
    pub fn crossing(&self) -> Vec<u64> {
        self.out
            .iter()
            .chain(&self.back)
            .chain(&self.input)
            .copied()
            .collect()
    }

    /// The guest's private input, the guest assuming `out` for the `out`
    /// cells; `reps` > 1 re-runs the program, for measuring.
    pub fn guest_input(&self, out: &[LfmWord], reps: u32) -> Vec<u8> {
        self.guest_input_with(out, &self.hints, reps)
    }

    /// [`Cut::guest_input`] with other hint values.
    pub fn guest_input_with(&self, out: &[LfmWord], hints: &[LfmWord], reps: u32) -> Vec<u8> {
        let mut v = Vec::with_capacity(8 + 4 * self.words.len() + 32 * (out.len() + hints.len()));
        v.extend((self.words.len() as u32).to_le_bytes());
        v.extend(reps.to_le_bytes());
        for w in &self.words {
            v.extend(w.to_le_bytes());
        }
        for w in out.iter().chain(hints) {
            for x in w {
                v.extend(x.canonical().to_le_bytes());
            }
        }
        v
    }

    pub fn out_values(&self) -> Vec<LfmWord> {
        self.out.iter().map(|a| self.values[a]).collect()
    }

    pub fn program_digest(&self) -> [u8; 32] {
        let bytes: Vec<u8> = self.words.iter().flat_map(|w| w.to_le_bytes()).collect();
        keccak(&bytes)
    }

    /// The guest's public output for the LFM's values.
    pub fn expected_output(&self) -> Vec<u8> {
        let lanes = |cells: &[u64]| -> Vec<[FE; 3]> {
            cells
                .iter()
                .map(|a| {
                    let w = self.values[a];
                    [w[0], w[1], w[2]]
                })
                .collect()
        };
        let record: Vec<_> = lanes(&self.out)
            .into_iter()
            .chain(lanes(&self.back))
            .collect();
        [
            self.program_digest(),
            digest(&lanes(&self.input)),
            digest(&record),
        ]
        .concat()
    }
}

fn keccak(bytes: &[u8]) -> [u8; 32] {
    let mut k = Keccak::v256();
    k.update(bytes);
    let mut out = [0u8; 32];
    k.finalize(&mut out);
    out
}

/// Lanes 0..3 of each cell, canonical little-endian `u64`s, as the guest
/// hashes them.
pub fn digest(cells: &[[FE; 3]]) -> [u8; 32] {
    let bytes: Vec<u8> = cells
        .iter()
        .flat_map(|c| c.iter().flat_map(|x| x.canonical().to_le_bytes()))
        .collect();
    keccak(&bytes)
}

/// Lanes 0..3 of the Field VM public cells at `faddr`.
fn published(public: &[PublicCell], faddr: &[u64]) -> Option<Vec<[FE; 3]>> {
    let by_addr: HashMap<u64, [FE; 3]> = public.iter().map(|(a, v)| (*a, *v.value())).collect();
    faddr.iter().map(|a| by_addr.get(a).copied()).collect()
}

/// The coupling: the guest ran the expected program over the input and
/// record the Field VM proof publishes. `faddr` are the Field VM addresses of
/// [`Cut::crossing`], in order. Both proofs must verify on their own.
pub fn coupled(
    cut: &Cut,
    program: &[u8; 32],
    fvm_public: &[PublicCell],
    faddr: &[u64],
    riscv_output: &[u8],
) -> bool {
    let records = cut.out.len() + cut.back.len();
    let (Some(record), Some(input)) = (
        published(fvm_public, &faddr[..records]),
        published(fvm_public, &faddr[records..]),
    ) else {
        return false;
    };
    riscv_output.len() == 96
        && riscv_output[..32] == program[..]
        && riscv_output[32..64] == digest(&input)[..]
        && riscv_output[64..] == digest(&record)[..]
}
