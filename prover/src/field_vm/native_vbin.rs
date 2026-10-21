//! A native `verify_b`: the hash half of a 3MI split compiled to RV64IM from
//! the program words the `split_verify_b` interpreter reads (`field_vm::split`
//! encoding, version 3), instead of interpreted.
//!
//! [`build`] is a pure function of those words, so a verifier recomputes the
//! guest ELF from the cut, and the ELF digest the proof binds is the binding
//! of the program. The guest keeps the interpreter's semantics, checks and
//! public output (`keccak256(words) ‖ in_base ‖ back_base ‖ n_in ‖ n_back ‖
//! publics`, the digest a constant of the ELF), over the region input of
//! `Cut::guest_input_regions` (the program words in it are skipped).
//!
//! Repeated blocks whose cell indices move by a constant per repetition are
//! re-rolled into loops, nested, with one pointer register per class of
//! operands that move together. Failed checks accumulate into two registers
//! that are tested before the commit, so check code has no branches.

use std::collections::HashMap;
use std::rc::Rc;

use crypto::hash::platform_keccak::PlatformKeccak256 as Keccak256;
use digest::Digest;

use super::rv64::*;

const VERSION: u32 = 3;
const CODE_BASE: u64 = 0x1_0000;
const SCRATCH: u64 = 0x1000_0000;
const STATE: u64 = SCRATCH;
const KSTATE: u64 = SCRATCH + 0x100;
const ARGS: u64 = SCRATCH + 0x200;
const FTMP: u64 = SCRATCH + 0x400;
/// Loop pointers kept in memory when the registers run out: the first
/// `NEAR_SLOTS` below `a0` (the RPX state, `SCRATCH`), the rest from `FAR`.
const SPILL: i64 = -0x800;
const NEAR_SLOTS: usize = 256;
const FAR: u64 = SCRATCH - 2 * PAGE;
const SPILL_SLOTS: usize = NEAR_SLOTS + (PAGE / 8) as usize;
const PUB: u64 = SCRATCH + 0x800;
const PAGE: u64 = 1 << 18;
const INPUT: u64 = 0xFF00_0004;
const P: u64 = 0xFFFF_FFFF_0000_0001;
const RPX_SYSCALL: i64 = -6;
const DOMAIN_TRANSCRIPT: i64 = u32::from_le_bytes(*b"LFMT") as i64;
const DOMAIN_LEAF: i64 = u32::from_le_bytes(*b"LFML") as i64;

/// Failed must-be-zero checks are OR-ed into `ERR`, must-be-one bits AND-ed
/// into `OK`.
const ERR: Reg = SP;
const OK: Reg = TP;
const PR: Reg = GP;
const HINTS: Reg = A6;
const INC: Reg = S0;
const PUBC: Reg = S1;
/// The pointer of the operands that do not move (the static class).
const FP: Reg = 27;
const POOL: [Reg; 9] = [18, 19, 20, 21, 22, 23, 24, 25, 26];
const ADDR_TMP: [Reg; 4] = [T3, T4, T5, T6];
/// Byte offset of a freshly slid pointer from the cell it was slid to: room
/// for the cells written after it.
const SLIDE: i64 = 2016;
/// Repetitions per trip of an innermost loop of at most `UNROLL_OPS` ops.
const UNROLL: u32 = 4;
const UNROLL_OPS: usize = 4;

/// Lanes 1..4 known zero.
pub(crate) const UPPER_ZERO: u8 = 1;
/// Lane 3 known zero.
pub(crate) const L3_ZERO: u8 = 2;
/// The zero word, a constant.
pub(crate) const ZERO_CONST: u8 = 4;

#[derive(Clone, Debug)]
pub(crate) enum Op {
    Hash {
        mode: u32,
    },
    Select,
    Pack,
    Unpack {
        mask: u32,
    },
    BitDec {
        n: u32,
        halves: bool,
    },
    Keccak {
        absorb: bool,
        rev: bool,
    },
    Hint {
        flags: u32,
    },
    BaseAlu {
        op: u32,
    },
    ExtAlu {
        op: u32,
    },
    Public,
    /// One level of a 4-ary Merkle walk (`lfm::p1w16_emit::walk4_child`):
    /// five selects placing the current digest at position `b0 + 2·b1` among
    /// its three siblings, then the width-16 permutation of the four.
    /// Reads `[b0, b1, cur, s0, s1, s2]`, writes the permutation's four words.
    Walk4,
}

/// Whether `op` runs the permutation, clobbering the state slots.
fn permutes(op: &Op) -> bool {
    matches!(op, Op::Hash { .. } | Op::Walk4)
}

pub(crate) struct Program {
    pub(crate) n_cells: u32,
    pub(crate) n_out: u32,
    pub(crate) back: Vec<u32>,
    pub(crate) consts: Vec<(u32, [u64; 4])>,
    pub(crate) ops: Vec<Op>,
    /// Per op: its cells, reads before writes, and the static facts of what
    /// it reads.
    pub(crate) cells: Vec<Vec<u32>>,
    pub(crate) facts: Vec<Vec<u8>>,
    /// Per op: how many of its cells it reads.
    pub(crate) reads: Vec<usize>,
    /// Per op cell, 0, or `1 + s` when the value passes through RPX state
    /// slot `s` (lanes `4s..4s + 4`) instead of its cell: a read finds it
    /// there, a write leaves it there (`forward`).
    fwd: Vec<Vec<u8>>,
    /// Reads of a shared hint's cell served from its IN cell instead
    /// (`alias_in`): per (op, cell), the byte offset from the IN cursor.
    in_alias: HashMap<(usize, usize), i64>,
    /// Reads of a hint's cell served from the hint itself in the input
    /// (`alias_in`): per (op, cell), the byte offset from the hint cursor.
    hint_alias: HashMap<(usize, usize), i64>,
    /// Per cell: every read of it passes its lanes unchanged (through
    /// selects and packs) into RPX inputs, whose table constrains them
    /// canonical (`rpx_only`).
    pub(crate) rpx_only: Vec<bool>,
}

struct Words<'a> {
    w: &'a [u32],
    at: usize,
}

impl Words<'_> {
    fn u32(&mut self) -> Option<u32> {
        let v = *self.w.get(self.at)?;
        self.at += 1;
        Some(v)
    }

    fn u64(&mut self) -> Option<u64> {
        Some(self.u32()? as u64 | (self.u32()? as u64) << 32)
    }
}

fn decode_plain(words: &[u32]) -> Option<Program> {
    let mut w = Words { w: words, at: 0 };
    if w.u32()? != VERSION {
        return None;
    }
    let n_cells = w.u32()?;
    let n_out = w.u32()?;
    let n_back = w.u32()?;
    let n_instr = w.u32()?;
    let back = (0..n_back).map(|_| w.u32()).collect::<Option<Vec<_>>>()?;
    let mut fact = vec![0u8; n_cells as usize];
    for f in fact.iter_mut().take(n_out as usize) {
        *f = L3_ZERO;
    }
    let mut prog = Program {
        n_cells,
        n_out,
        back,
        consts: Vec::new(),
        ops: Vec::new(),
        cells: Vec::new(),
        facts: Vec::new(),
        reads: Vec::new(),
        fwd: Vec::new(),
        in_alias: HashMap::new(),
        hint_alias: HashMap::new(),
        rpx_only: Vec::new(),
    };
    let cell = |w: &mut Words| -> Option<u32> { w.u32().filter(|&c| c < n_cells) };
    for _ in 0..n_instr {
        let mut reads = Vec::new();
        let mut writes = Vec::new();
        let (op, out_fact) = match w.u32()? {
            1 => {
                let mode = w.u32()?;
                let (ni, no) = match mode {
                    0..=2 => (2, 1),
                    3 => (3, 3),
                    4 => (4, 4),
                    _ => return None,
                };
                for _ in 0..ni {
                    reads.push(cell(&mut w)?);
                }
                for _ in 0..no {
                    writes.push(cell(&mut w)?);
                }
                (Op::Hash { mode }, 0)
            }
            2 => {
                for _ in 0..3 {
                    reads.push(cell(&mut w)?);
                }
                writes.push(cell(&mut w)?);
                writes.push(cell(&mut w)?);
                let f = fact[reads[1] as usize] & fact[reads[2] as usize];
                (Op::Select, f)
            }
            3 => {
                for _ in 0..4 {
                    reads.push(cell(&mut w)?);
                }
                writes.push(cell(&mut w)?);
                (Op::Pack, 0)
            }
            4 => {
                reads.push(cell(&mut w)?);
                for _ in 0..4 {
                    writes.push(cell(&mut w)?);
                }
                let mask = w.u32()?;
                (Op::Unpack { mask }, UPPER_ZERO | L3_ZERO)
            }
            5 => {
                reads.push(cell(&mut w)?);
                let n = w.u32()?;
                if n > 64 {
                    return None;
                }
                for _ in 0..n {
                    writes.push(cell(&mut w)?);
                }
                let halves = w.u32()? != 0;
                if halves {
                    writes.push(cell(&mut w)?);
                    writes.push(cell(&mut w)?);
                }
                (Op::BitDec { n, halves }, UPPER_ZERO | L3_ZERO)
            }
            6 => {
                let absorb = w.u32()? != 0;
                for _ in 0..13 + if absorb { 9 } else { 0 } {
                    reads.push(cell(&mut w)?);
                }
                for _ in 0..13 {
                    writes.push(cell(&mut w)?);
                }
                let rev = w.u32()? != 0;
                if rev {
                    writes.push(cell(&mut w)?);
                    writes.push(cell(&mut w)?);
                }
                (Op::Keccak { absorb, rev }, 0)
            }
            7 => {
                writes.push(cell(&mut w)?);
                let flags = w.u32()?;
                let f = match flags >> 1 {
                    1 => L3_ZERO,
                    2 => UPPER_ZERO | L3_ZERO,
                    _ => 0,
                };
                (Op::Hint { flags }, f)
            }
            8 => {
                let out = cell(&mut w)?;
                let value = [w.u64()?, w.u64()?, w.u64()?, w.u64()?];
                fact[out as usize] = if value == [0; 4] {
                    UPPER_ZERO | L3_ZERO | ZERO_CONST
                } else if value[1..] == [0; 3] {
                    UPPER_ZERO | L3_ZERO
                } else if value[3] == 0 {
                    L3_ZERO
                } else {
                    0
                };
                prog.consts.push((out, value));
                continue;
            }
            9 => {
                let op = w.u32()?;
                if op > 4 {
                    return None;
                }
                for _ in 0..2 + (op == 4) as usize {
                    reads.push(cell(&mut w)?);
                }
                writes.push(cell(&mut w)?);
                (Op::BaseAlu { op }, UPPER_ZERO | L3_ZERO)
            }
            10 => {
                let op = w.u32()?;
                if op > 5 {
                    return None;
                }
                for _ in 0..2 + (op == 4) as usize {
                    reads.push(cell(&mut w)?);
                }
                writes.push(cell(&mut w)?);
                (Op::ExtAlu { op }, L3_ZERO)
            }
            11 => {
                reads.push(cell(&mut w)?);
                (Op::Public, 0)
            }
            _ => return None,
        };
        let facts = reads.iter().map(|&c| fact[c as usize]).collect();
        for &c in &writes {
            fact[c as usize] = out_fact;
        }
        if let Op::Keccak { .. } = op {
            fact[writes[12] as usize] = L3_ZERO;
        }
        prog.reads.push(reads.len());
        reads.extend(writes);
        prog.ops.push(op);
        prog.cells.push(reads);
        prog.facts.push(facts);
    }
    if w.at != words.len() {
        return None;
    }
    if std::env::var_os("NATIVE_VBIN_NO_WALK4").is_none() {
        fuse_walk4(&mut prog);
    }
    Some(prog)
}

fn decode(words: &[u32]) -> Option<Program> {
    let mut prog = decode_plain(words)?;
    forward(&mut prog);
    alias_in(&mut prog);
    rpx_only(&mut prog);
    Some(prog)
}

/// The program with every value in its cell (no forwarding or aliasing),
/// `rpx_only` computed.
pub(crate) fn decode_generic(words: &[u32]) -> Option<Program> {
    let mut prog = decode_plain(words)?;
    prog.rpx_only = vec![false; prog.n_cells as usize];
    rpx_only(&mut prog);
    Some(prog)
}

/// Fuses each 4-ary Merkle level, five selects and the permutation of the
/// four words they place, into one [`Op::Walk4`] when the selects' outputs
/// have no other reader. The fused op computes the same words: the current
/// digest at position `b0 + 2·b1`, the siblings in order around it.
fn fuse_walk4(prog: &mut Program) {
    let n = prog.ops.len();
    let mut readers = vec![0u32; prog.n_cells as usize];
    for k in 0..n {
        for &c in &prog.cells[k][..prog.reads[k]] {
            readers[c as usize] += 1;
        }
    }
    for e in &prog.back {
        readers[(e & ((1 << 30) - 1)) as usize] += 1;
    }
    let mut keep = vec![true; n];
    let mut fused: Vec<Option<(Vec<u32>, Vec<u8>)>> = vec![None; n];
    let mut k = 0;
    while k + 6 <= n {
        let sel = |q: usize| matches!(prog.ops[k + q], Op::Select);
        if !((0..5).all(sel) && matches!(prog.ops[k + 5], Op::Hash { mode: 4 })) {
            k += 1;
            continue;
        }
        // Select cells: [bit, in_l, in_r, out_l, out_r].
        let c = |q: usize| &prog.cells[k + q];
        let (s1, s2, s3, s4, s5, h) = (c(0), c(1), c(2), c(3), c(4), c(5));
        let b1 = s1[0];
        let b0 = s3[0];
        let (partner, y) = (s1[3], s1[4]);
        let (u0, u1) = (s2[3], s2[4]);
        let (l, r) = (s3[3], s3[4]);
        let shape = s2[0] == b1
            && s2[2] == y
            && s3[2] == partner
            && s4[0] == b1
            && s4[1] == l
            && s4[2] == u0
            && s5[0] == b1
            && s5[1] == r
            && s5[2] == u1
            && h[..4] == [s4[3], s5[3], s4[4], s5[4]];
        let private = [partner, y, u0, u1, l, r, s4[3], s4[4], s5[3], s5[4]]
            .iter()
            .all(|&x| readers[x as usize] == 1);
        if !(shape && private) {
            k += 1;
            continue;
        }
        let f = |q: usize, j: usize| prog.facts[k + q][j];
        let cells = vec![b0, b1, s3[1], s1[1], s2[1], s1[2], h[4], h[5], h[6], h[7]];
        let facts = vec![f(2, 0), f(0, 0), f(2, 1), f(0, 1), f(1, 1), f(0, 2)];
        fused[k] = Some((cells, facts));
        for q in 1..6 {
            keep[k + q] = false;
        }
        k += 6;
    }
    let mut ops = Vec::with_capacity(n);
    let (mut cells, mut facts, mut reads) = (Vec::new(), Vec::new(), Vec::new());
    for k in 0..n {
        if !keep[k] {
            continue;
        }
        match fused[k].take() {
            Some((c, f)) => {
                ops.push(Op::Walk4);
                cells.push(c);
                facts.push(f);
                reads.push(6);
            }
            None => {
                ops.push(prog.ops[k].clone());
                cells.push(std::mem::take(&mut prog.cells[k]));
                facts.push(std::mem::take(&mut prog.facts[k]));
                reads.push(prog.reads[k]);
            }
        }
    }
    prog.ops = ops;
    prog.cells = cells;
    prog.facts = facts;
    prog.reads = reads;
}

/// A hash output no one reads, left in the RPX state (`Program::fwd`).
const DEAD: u8 = u8::MAX;
/// A shared extension hint whose cell is never written: its readers read
/// its IN cell (`alias_in`).
const IN_CELL: u8 = u8::MAX - 1;
/// A hint whose cell is never written: its readers read the hint in the
/// input (`alias_in`).
const IN_STREAM: u8 = u8::MAX - 2;

/// Which cells only ever reach RPX inputs, lanes unchanged: read by a hash,
/// or as a select's word or a pack's lane whose outputs only do. The RPX
/// table constrains every input felt below p (`tables::rpx`, the canonicity
/// constraints on the 12 input lanes), so a hint lane that only goes there
/// needs no check of its own.
fn rpx_only(prog: &mut Program) {
    let n_cells = prog.n_cells as usize;
    let mut flows = vec![true; n_cells];
    let mut read = vec![false; n_cells];
    for e in &prog.back {
        flows[(e & ((1 << 30) - 1)) as usize] = false;
    }
    let mut done = vec![false; n_cells];
    for k in (0..prog.ops.len()).rev() {
        let nr = prog.reads[k];
        for &c in &prog.cells[k][nr..] {
            let c = c as usize;
            done[c] = read[c] && flows[c];
        }
        let outs = &prog.cells[k][nr..];
        for (j, &c) in prog.cells[k][..nr].iter().enumerate() {
            let ok = match prog.ops[k] {
                Op::Hash { .. } => true,
                Op::Walk4 => j >= 2,
                Op::Select if j > 0 => outs.iter().all(|&o| done[o as usize]),
                Op::Pack => done[outs[0] as usize],
                _ => false,
            };
            read[c as usize] = true;
            flows[c as usize] &= ok;
        }
    }
    prog.rpx_only = done;
}

/// A shared extension hint (lane 3 zero, lanes 0..3 in IN) all of whose
/// readers take at most lanes 0..3, as base or extension reads, leaves its
/// cell unwritten: they read its IN cell, at a fixed distance behind the IN
/// cursor, which only hints and unpacks move.
fn alias_in(prog: &mut Program) {
    let n = prog.ops.len();
    let mut before = vec![0i64; n];
    let mut pos = prog.n_out as i64;
    for (k, op) in prog.ops.iter().enumerate() {
        before[k] = pos;
        pos += match *op {
            Op::Hint { flags } => (flags & 1) as i64,
            Op::Unpack { mask } => (mask & 0xf).count_ones() as i64,
            _ => 0,
        };
    }
    let mut hints_before = vec![0i64; n];
    let mut h = 0;
    for (k, op) in prog.ops.iter().enumerate() {
        hints_before[k] = h;
        if let Op::Hint { flags } = *op {
            h += 8 * stream_lanes(flags).len() as i64;
        }
    }
    let mut hint_of: HashMap<u32, usize> = HashMap::new();
    for (k, op) in prog.ops.iter().enumerate() {
        if matches!(op, Op::Hint { flags } if *flags == 3 || flags & 1 == 0) && prog.fwd[k][0] == 0
        {
            hint_of.insert(prog.cells[k][0], k);
        }
    }
    let mut uses: HashMap<u32, Vec<(usize, usize)>> = HashMap::new();
    let mut refused: std::collections::HashSet<u32> = Default::default();
    for e in &prog.back {
        refused.insert(e & ((1 << 30) - 1));
    }
    for k in 0..n {
        let base_reader = matches!(
            prog.ops[k],
            Op::Pack | Op::BitDec { .. } | Op::BaseAlu { .. } | Op::ExtAlu { .. }
        );
        for j in 0..prog.reads[k] {
            let c = prog.cells[k][j];
            if let Some(&h) = hint_of.get(&c) {
                let shared = matches!(prog.ops[h], Op::Hint { flags: 3 });
                let off = if shared {
                    24 * (before[h] - before[k])
                } else {
                    hints_before[h] - hints_before[k]
                };
                // A packed hint is as long as its type: only readers of no
                // more lanes than it holds read it in place.
                let fits = match prog.ops[h] {
                    Op::Hint { flags } if !shared && packed_hints() => match flags >> 1 {
                        1 => base_reader,
                        2 => matches!(
                            prog.ops[k],
                            Op::Pack | Op::BitDec { .. } | Op::BaseAlu { .. }
                        ),
                        _ => true,
                    },
                    _ => true,
                };
                if fits
                    && (base_reader || !shared)
                    && prog.fwd[k][j] == 0
                    && (-2048..=2047 - 24).contains(&off)
                {
                    uses.entry(c).or_default().push((k, j));
                } else {
                    refused.insert(c);
                }
            }
        }
    }
    for (c, h) in hint_of {
        if refused.contains(&c) {
            continue;
        }
        let shared = matches!(prog.ops[h], Op::Hint { flags: 3 });
        prog.fwd[h][0] = if shared { IN_CELL } else { IN_STREAM };
        for (k, j) in uses.remove(&c).unwrap_or_default() {
            if shared {
                prog.in_alias.insert((k, j), 24 * (before[h] - before[k]));
            } else {
                prog.hint_alias
                    .insert((k, j), hints_before[h] - hints_before[k]);
            }
        }
    }
}

/// Values that skip their cell, passed through the RPX state: a hash output
/// read once, before the next hash, stays in its slot (a permutation reading
/// it must read it at the same position); one no one reads is not copied
/// out; a word that a select, pack or hint writes for the next hash only,
/// before any other hash, is written into the slot of the input it is.
fn forward(prog: &mut Program) {
    let n = prog.ops.len();
    let mut readers = vec![0u32; prog.n_cells as usize];
    let mut reader_at = vec![u32::MAX; prog.n_cells as usize];
    for k in 0..n {
        for &c in &prog.cells[k][..prog.reads[k]] {
            readers[c as usize] += 1;
            reader_at[c as usize] = k as u32;
        }
    }
    for e in &prog.back {
        readers[(e & ((1 << 30) - 1)) as usize] += 1;
    }
    let inputs = |op: &Op| match op {
        Op::Hash { mode } if *mode < 3 => 2,
        Op::Hash { mode } => *mode as usize,
        _ => 0,
    };
    let mut occ: [Option<u32>; 4] = [None; 4];
    prog.fwd = prog.cells.iter().map(|c| vec![0; c.len()]).collect();
    for k in 0..n {
        let nr = prog.reads[k];
        for j in 0..nr {
            let c = prog.cells[k][j];
            if let Some(s) = occ.iter().position(|&o| o == Some(c)) {
                prog.fwd[k][j] = 1 + s as u8;
                occ[s] = None;
            }
        }
        match prog.ops[k] {
            Op::Hash { .. } | Op::Walk4 => {
                occ = [None; 4];
                for j in nr..prog.cells[k].len() {
                    let slot = j - nr;
                    let c = prog.cells[k][j] as usize;
                    if readers[c] == 0 {
                        prog.fwd[k][j] = DEAD;
                        continue;
                    }
                    let r = reader_at[c] as usize;
                    let same_slot = match prog.ops[r] {
                        Op::Hash { mode: 3 | 4 } => prog.cells[r][slot] as usize == c,
                        // A fused Merkle level reads only its current digest
                        // from a slot: its siblings' stores overwrite them.
                        Op::Walk4 => prog.cells[r][2] as usize == c,
                        _ => true,
                    };
                    if readers[c] == 1
                        && r > k
                        && r - k <= 64
                        && same_slot
                        && (k + 1..r).all(|q| !permutes(&prog.ops[q]))
                    {
                        prog.fwd[k][j] = 1 + slot as u8;
                        occ[slot] = Some(c as u32);
                    }
                }
            }
            Op::Select | Op::Pack | Op::Hint { .. } => {
                // The word's one reader is the next hash, within reach.
                for j in nr..prog.cells[k].len() {
                    let c = prog.cells[k][j];
                    let r = reader_at[c as usize] as usize;
                    if readers[c as usize] != 1
                        || r <= k
                        || r - k > 64
                        || inputs(&prog.ops[r]) == 0
                        || (k + 1..r).any(|q| permutes(&prog.ops[q]))
                    {
                        continue;
                    }
                    let ins = &prog.cells[r][..inputs(&prog.ops[r])];
                    let Some(q) = ins.iter().position(|&x| x == c) else {
                        continue;
                    };
                    if occ[q].is_none() {
                        prog.fwd[k][j] = 1 + q as u8;
                        occ[q] = Some(c);
                    }
                }
            }
            _ => {}
        }
    }
}

fn shape_key(op: &Op, facts: &[u8]) -> Vec<u64> {
    let mut k = match *op {
        Op::Hash { mode } => vec![1, mode as u64],
        Op::Select => vec![2],
        Op::Walk4 => vec![12],
        Op::Pack => vec![3],
        Op::Unpack { mask } => vec![4, mask as u64],
        Op::BitDec { n, halves } => vec![5, n as u64, halves as u64],
        Op::Keccak { absorb, rev } => vec![6, absorb as u64, rev as u64],
        Op::Hint { flags } => vec![7, flags as u64],
        Op::BaseAlu { op } => vec![9, op as u64],
        Op::ExtAlu { op } => vec![10, op as u64],
        Op::Public => vec![11],
    };
    k.extend(facts.iter().map(|&f| f as u64));
    k
}

#[derive(Clone)]
enum Item {
    Op(u32),
    Loop(Rc<Loop>),
}

struct Loop {
    id: u32,
    reps: u32,
    body: Vec<Item>,
    /// Per operand of the body's first repetition, its move per repetition.
    delta: Vec<i64>,
    shape: u32,
    cells: Vec<i64>,
    /// Ops in the body's code, nested loops counted once.
    size: usize,
    depth: usize,
}

struct Roller<'a> {
    prog: &'a Program,
    shapes: HashMap<Vec<u64>, u32>,
    op_shape: Vec<u32>,
    op_cells: Vec<Vec<i64>>,
    next_id: u32,
    refused: usize,
    /// Whether a nest of a shape fits its registers and spill slots.
    fits: HashMap<u32, bool>,
}

const MARK_LOOP: u64 = u64::MAX;

impl<'a> Roller<'a> {
    fn new(prog: &'a Program) -> Self {
        let mut r = Roller {
            prog,
            shapes: HashMap::new(),
            op_shape: Vec::new(),
            op_cells: Vec::new(),
            next_id: 0,
            refused: 0,
            fits: HashMap::new(),
        };
        for k in 0..prog.ops.len() {
            let mut key = shape_key(&prog.ops[k], &prog.facts[k]);
            key.extend(prog.fwd[k].iter().map(|&f| f as u64));
            for j in 0..prog.reads[k] {
                if let Some(&off) = prog.in_alias.get(&(k, j)) {
                    key.extend([j as u64, off as u64]);
                }
                if let Some(&off) = prog.hint_alias.get(&(k, j)) {
                    key.extend([100 + j as u64, off as u64]);
                }
            }
            if let Op::Hint { .. } = prog.ops[k] {
                key.push(prog.rpx_only[prog.cells[k][0] as usize] as u64);
            }
            let s = r.intern(key);
            r.op_shape.push(s);
            r.op_cells
                .push(prog.cells[k].iter().map(|&c| c as i64).collect());
        }
        r
    }

    fn intern(&mut self, key: Vec<u64>) -> u32 {
        let n = self.shapes.len() as u32;
        *self.shapes.entry(key).or_insert(n)
    }

    fn shape(&self, it: &Item) -> u32 {
        match it {
            Item::Op(k) => self.op_shape[*k as usize],
            Item::Loop(l) => l.shape,
        }
    }

    fn cells<'b>(&'b self, it: &'b Item) -> &'b [i64] {
        match it {
            Item::Op(k) => &self.op_cells[*k as usize],
            Item::Loop(l) => &l.cells,
        }
    }

    fn size(it: &Item) -> usize {
        match it {
            Item::Op(_) => 1,
            Item::Loop(l) => l.size + 2,
        }
    }

    fn make_loop(&mut self, body: Vec<Item>, reps: u32, delta: Vec<i64>) -> Rc<Loop> {
        let mut key = vec![MARK_LOOP, reps as u64, body.len() as u64];
        key.extend(body.iter().map(|it| self.shape(it) as u64));
        key.extend(delta.iter().map(|&d| d as u64));
        let shape = self.intern(key);
        let cells = body.iter().flat_map(|it| self.cells(it)).copied().collect();
        let size = body.iter().map(Self::size).sum();
        let depth = 1 + body
            .iter()
            .map(|it| match it {
                Item::Loop(l) => l.depth,
                Item::Op(_) => 0,
            })
            .max()
            .unwrap_or(0);
        self.next_id += 1;
        Rc::new(Loop {
            id: self.next_id,
            reps,
            body,
            delta,
            shape,
            cells,
            size,
            depth,
        })
    }

    /// One pass: runs of at least two repetitions of a block of at most
    /// `max_p` items become loops, the run saving the most code first.
    fn pass(&mut self, items: Vec<Item>, max_p: usize) -> Vec<Item> {
        let n = items.len();
        let ids: Vec<u64> = items.iter().map(|it| self.shape(it) as u64 + 1).collect();
        const B: u64 = 0x100_0000_01b3;
        let mut h = vec![0u64; n + 1];
        let mut pw = vec![1u64; n + 1];
        for i in 0..n {
            h[i + 1] = h[i].wrapping_mul(B).wrapping_add(ids[i]);
            pw[i + 1] = pw[i].wrapping_mul(B);
        }
        let hash = |a: usize, len: usize| h[a + len].wrapping_sub(h[a].wrapping_mul(pw[len]));
        let periods: Vec<usize> = if max_p <= 64 {
            (1..=max_p).collect()
        } else {
            const K: usize = 8;
            let mut last: HashMap<u64, usize> = HashMap::new();
            let mut dist: HashMap<usize, usize> = HashMap::new();
            for i in 0..n.saturating_sub(K) {
                let k = hash(i, K);
                if let Some(&q) = last.get(&k) {
                    *dist.entry(i - q).or_insert(0) += 1;
                }
                last.insert(k, i);
            }
            let mut d: Vec<(usize, usize)> = dist
                .into_iter()
                .filter(|&(p, _)| p <= max_p && p > 64)
                .collect();
            d.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
            d.truncate(32);
            d.into_iter().map(|(p, _)| p).collect()
        };
        let mut out = Vec::with_capacity(n);
        let mut i = 0;
        while i < n {
            let mut cands: Vec<(usize, usize, Vec<i64>, usize)> = Vec::new();
            for &p in &periods {
                if i + 2 * p > n || hash(i, p) != hash(i + p, p) {
                    continue;
                }
                let base: Vec<i64> = items[i..i + p]
                    .iter()
                    .flat_map(|it| self.cells(it))
                    .copied()
                    .collect();
                let delta: Vec<i64> = items[i + p..i + 2 * p]
                    .iter()
                    .flat_map(|it| self.cells(it))
                    .zip(&base)
                    .map(|(y, x)| y - x)
                    .collect();
                let mut reps = 2;
                while i + (reps + 1) * p <= n && hash(i, p) == hash(i + reps * p, p) && {
                    let r = reps as i64;
                    items[i + reps * p..i + (reps + 1) * p]
                        .iter()
                        .flat_map(|it| self.cells(it))
                        .zip(base.iter().zip(&delta))
                        .all(|(y, (x, d))| *y == x + r * d)
                } {
                    reps += 1;
                }
                let size: usize = items[i..i + p].iter().map(Self::size).sum();
                let saved = (reps - 1) * size;
                if saved >= 2 {
                    cands.push((p, reps, delta, saved));
                }
            }
            // The run saving the most code that fits its registers.
            cands.sort_by(|a, b| b.3.cmp(&a.3).then(a.0.cmp(&b.0)));
            let mut taken = None;
            for (p, reps, delta, _) in cands {
                let body = items[i..i + p].to_vec();
                let l = self.make_loop(body, reps as u32, delta);
                let fits = match self.fits.get(&l.shape) {
                    Some(&f) => f,
                    None => {
                        let f = nest_fits(&self.op_cells, &l);
                        self.fits.insert(l.shape, f);
                        f
                    }
                };
                if fits {
                    taken = Some((l, reps * p));
                    break;
                }
                self.refused += 1;
            }
            match taken {
                Some((l, span)) => {
                    out.push(Item::Loop(l));
                    i += span;
                }
                None => {
                    out.push(items[i].clone());
                    i += 1;
                }
            }
        }
        out
    }

    fn roll(&mut self) -> Vec<Item> {
        let mut items: Vec<Item> = (0..self.prog.ops.len() as u32).map(Item::Op).collect();
        for max_p in [4, 16, 64, 256, 1024, 4096, 1 << 16] {
            items = self.pass(items, max_p);
        }
        items
    }
}

/// A class of operands that move together: per enclosing loop, from the
/// outermost, its id and the operands' move per repetition, trailing zero
/// moves dropped. The empty class does not move.
type ClassKey = Vec<(u32, i64)>;

fn extend_key(outer: &ClassKey, id: u32, d: i64) -> ClassKey {
    let mut k = outer.clone();
    k.push((id, d));
    while k.last().is_some_and(|&(_, d)| d == 0) {
        k.pop();
    }
    k
}

/// Every non-static class of the operands under `items`, given their classes
/// from the enclosing loops: its first cell and how many operands it has.
fn classes(
    op_cells: &[Vec<i64>],
    items: &[Item],
    keys: &[ClassKey],
    out: &mut Vec<(ClassKey, i64, usize)>,
    index: &mut HashMap<ClassKey, usize>,
    weight: usize,
) {
    let mut at = 0;
    for it in items {
        match it {
            Item::Op(k) => {
                let cells = &op_cells[*k as usize];
                for (key, &c) in keys[at..at + cells.len()].iter().zip(cells) {
                    if key.is_empty() {
                        continue;
                    }
                    match index.get(key) {
                        Some(&i) => out[i].2 += weight,
                        None => {
                            index.insert(key.clone(), out.len());
                            out.push((key.clone(), c, weight));
                        }
                    }
                }
                at += cells.len();
            }
            Item::Loop(l) => {
                let n = l.cells.len();
                let inner: Vec<ClassKey> = keys[at..at + n]
                    .iter()
                    .zip(&l.delta)
                    .map(|(k, &d)| extend_key(k, l.id, d))
                    .collect();
                classes(
                    op_cells,
                    &l.body,
                    &inner,
                    out,
                    index,
                    weight.saturating_mul(l.reps as usize),
                );
                at += n;
            }
        }
    }
}

fn nest_classes(op_cells: &[Vec<i64>], l: &Rc<Loop>) -> Vec<(ClassKey, i64, usize)> {
    let mut out = Vec::new();
    classes(
        op_cells,
        &[Item::Loop(l.clone())],
        &vec![Vec::new(); l.cells.len()],
        &mut out,
        &mut HashMap::new(),
        1,
    );
    out
}

/// Whether the nest rooted at `l` has registers for its counters and room
/// for its pointers, in registers or spilled.
fn nest_fits(op_cells: &[Vec<i64>], l: &Rc<Loop>) -> bool {
    l.depth < POOL.len()
        && (l.cells.len() <= SPILL_SLOTS || nest_classes(op_cells, l).len() <= SPILL_SLOTS)
}

#[derive(Clone, Copy)]
enum Loc {
    Reg(Reg),
    /// A spill slot.
    Mem(usize),
}

struct Class {
    loc: Loc,
    /// `reg = MEM + 32·Σ k·d + off` over the enclosing loops' counters `k`.
    off: i64,
}

#[derive(Default)]
struct Runtime {
    fadd: Option<Label>,
    fsub: Option<Label>,
    fmul: Option<Label>,
    finv: Option<Label>,
    bswap: Option<Label>,
    base: [Option<Label>; 5],
    ext: [Option<Label>; 6],
    keccak: [Option<Label>; 4],
}

const DATA: [Reg; 4] = [T0, T1, T2, A1];

struct Gen<'a> {
    a: Asm,
    prog: &'a Program,
    op_cells: &'a [Vec<i64>],
    rc_addr: i64,
    keys: Vec<ClassKey>,
    key_ids: HashMap<ClassKey, usize>,
    cls: Vec<Option<Class>>,
    /// Per class, cells to add to its operands' (a copy of an unrolled body).
    shift: Vec<i64>,
    free: Vec<Reg>,
    tmp: usize,
    depth: usize,
    rt: Runtime,
    loops: usize,
    max_depth: usize,
    spilled: usize,
    /// The next free spill slot.
    spill_next: usize,
    /// Advances of the hint and IN cursors not emitted yet, in bytes.
    pend_hints: i64,
    pend_in: i64,
    /// Where each part of the code starts, by what it does.
    marks: Vec<(usize, &'static str)>,
    /// Per op kind, outside and inside loops: ops emitted and their words.
    census: std::collections::BTreeMap<&'static str, [(usize, usize); 2]>,
}

fn kind(op: &Op) -> &'static str {
    match op {
        Op::Hash { .. } => "hash",
        Op::Select => "select",
        Op::Walk4 => "walk4",
        Op::Pack => "pack",
        Op::Unpack { .. } => "unpack",
        Op::BitDec { .. } => "bit_dec",
        Op::Keccak { .. } => "keccak",
        Op::Hint { .. } => "hint",
        Op::BaseAlu { .. } => "base_alu",
        Op::ExtAlu { .. } => "ext_alu",
        Op::Public => "public",
    }
}

fn rt_label(a: &mut Asm, slot: &mut Option<Label>) -> Label {
    *slot.get_or_insert_with(|| a.label())
}

impl Gen<'_> {
    fn class_id(&mut self, key: ClassKey) -> usize {
        if let Some(&id) = self.key_ids.get(&key) {
            return id;
        }
        let id = self.keys.len();
        self.keys.push(key.clone());
        self.key_ids.insert(key, id);
        self.cls.push(None);
        self.shift.push(0);
        id
    }

    /// The address of spill slot `slot`, as `(base, offset)`, through `t`
    /// for a far one.
    fn spill_at(&mut self, slot: usize, t: Reg) -> (Reg, i64) {
        if slot < NEAR_SLOTS {
            return (A0, SPILL + 8 * slot as i64);
        }
        let addr = (FAR + 8 * (slot - NEAR_SLOTS) as u64) as i64;
        let lo = (addr << 52) >> 52;
        self.a.li(t, addr - lo);
        (t, lo)
    }

    fn spill_ld(&mut self, rd: Reg, slot: usize) {
        let (b, o) = self.spill_at(slot, rd);
        self.a.ld(rd, b, o);
    }

    fn spill_sd(&mut self, rs: Reg, slot: usize, t: Reg) {
        let (b, o) = self.spill_at(slot, t);
        self.a.sd(rs, b, o);
    }

    fn mark(&mut self, what: &'static str) {
        if self.marks.last().is_none_or(|m| m.1 != what) {
            self.marks.push((self.a.pc(), what));
        }
    }

    fn next_tmp(&mut self) -> Reg {
        self.tmp = (self.tmp + 1) % ADDR_TMP.len();
        ADDR_TMP[self.tmp]
    }

    /// A register and offset addressing lanes 0..4 of `cell` (its value with
    /// every enclosing loop at its first repetition). A write slides the
    /// class pointer when the cell is out of reach; a read goes through a
    /// temporary.
    fn base(&mut self, class: usize, cell: i64, write: bool) -> (Reg, i64) {
        let c = self.cls[class].as_ref().expect("a live class");
        let (loc, off) = (c.loc, c.off);
        let cell = cell + self.shift[class];
        let imm = 32 * cell - off;
        let in_reach = (-2048..=2047 - 24).contains(&imm);
        let owner = self.marks.last().map_or("?", |m| m.1);
        let reg = match loc {
            Loc::Reg(r) => r,
            Loc::Mem(o) => {
                self.mark("addr_spilled");
                let t = self.next_tmp();
                self.spill_ld(t, o);
                t
            }
        };
        if in_reach {
            self.mark(owner);
            return (reg, imm);
        }
        let t = self.next_tmp();
        if write {
            self.mark("addr_slide");
            let new_off = 32 * cell + SLIDE;
            self.a.add_imm(reg, reg, new_off - off, t);
            if let Loc::Mem(o) = loc {
                self.spill_sd(reg, o, t);
            }
            self.cls[class].as_mut().unwrap().off = new_off;
            self.mark(owner);
            return (reg, -SLIDE);
        }
        self.mark("addr_far");
        let mut lo = ((imm + 2048) & 4095) - 2048;
        if lo > 2047 - 24 {
            lo -= 4096;
        }
        self.a.li(t, imm - lo);
        self.a.add(t, t, reg);
        self.mark(owner);
        (t, lo)
    }

    /// `(cursor, off)` addressing lanes 0..4 at `off` from where the cursor
    /// stands at this point of the program, its pending advance applied.
    fn cursor(&mut self, cur: Reg, off: i64) -> (Reg, i64) {
        let pend = if cur == INC {
            self.pend_in
        } else {
            self.pend_hints
        };
        if !(-2048..=2047 - 24).contains(&(off + pend)) {
            self.flush_cursors();
            return (cur, off);
        }
        (cur, off + pend)
    }

    /// Advances the hint and IN cursors by `hints` and `ins` bytes, at
    /// compile time until the pending advance would leave the immediates.
    fn advance(&mut self, hints: i64, ins: i64) {
        self.pend_hints += hints;
        self.pend_in += ins;
        if self.pend_hints > 1024 || self.pend_in > 1024 {
            self.flush_cursors();
        }
    }

    fn flush_cursors(&mut self) {
        if self.pend_hints != 0 {
            self.a.addi(HINTS, HINTS, self.pend_hints);
            self.pend_hints = 0;
        }
        if self.pend_in != 0 {
            self.a.addi(INC, INC, self.pend_in);
            self.pend_in = 0;
        }
    }

    /// [`Gen::base`] for cell `j` of op `k`, or its RPX state slot.
    fn opnd(&mut self, k: usize, keys: &[usize], j: usize, write: bool) -> (Reg, i64) {
        if let Some(&off) = self.prog.in_alias.get(&(k, j)) {
            return self.cursor(INC, off);
        }
        if let Some(&off) = self.prog.hint_alias.get(&(k, j)) {
            return self.cursor(HINTS, off);
        }
        match self.prog.fwd[k][j] {
            0 => self.base(keys[j], self.op_cells[k][j], write),
            f => (A0, 32 * (f as i64 - 1)),
        }
    }

    /// Lanes 1..4 of the word at `(r, i)` zero, lane 3 skipped when `fact`
    /// knows it is.
    fn check_upper_zero(&mut self, r: Reg, i: i64, fact: u8) {
        for l in 1..if fact & L3_ZERO != 0 { 3 } else { 4 } {
            self.a.ld(A5, r, i + 8 * l);
            self.a.or(ERR, ERR, A5);
        }
    }

    fn items(&mut self, items: &[Item], keys: &[usize]) {
        let mut at = 0;
        for it in items {
            match it {
                Item::Op(k) => {
                    let n = self.op_cells[*k as usize].len();
                    self.op(*k as usize, &keys[at..at + n]);
                    at += n;
                }
                Item::Loop(l) => {
                    let n = l.cells.len();
                    self.lp(l, &keys[at..at + n]);
                    at += n;
                }
            }
        }
    }

    fn lp(&mut self, l: &Rc<Loop>, outer: &[usize]) {
        self.mark("loop");
        self.loops += 1;
        self.max_depth = self.max_depth.max(self.depth + l.depth);
        let inner: Vec<usize> = outer
            .iter()
            .zip(&l.delta)
            .map(|(&k, &d)| {
                let key = extend_key(&self.keys[k], l.id, d);
                self.class_id(key)
            })
            .collect();
        // The classes this loop owns: those moving with it and no deeper
        // loop, and the prefixes deeper loops start their own from. Each
        // starts from its parent (the class without this loop) at entry.
        let mut found = Vec::new();
        let outer_keys: Vec<ClassKey> = outer.iter().map(|&k| self.keys[k].clone()).collect();
        classes(
            self.op_cells,
            &[Item::Loop(l.clone())],
            &outer_keys,
            &mut found,
            &mut HashMap::new(),
            1,
        );
        let mut owned: Vec<(ClassKey, i64, usize)> = Vec::new();
        for (key, first, w) in found {
            let Some(i) = key.iter().position(|&(id, _)| id == l.id) else {
                continue;
            };
            let mut own = key[..=i].to_vec();
            while own.last().is_some_and(|&(_, d)| d == 0) {
                own.pop();
            }
            if own.last().is_none_or(|&(id, _)| id != l.id) {
                continue;
            }
            match owned.iter_mut().find(|(k, _, _)| *k == own) {
                Some(o) => o.2 += w,
                None => owned.push((own, first, w)),
            }
        }
        owned.sort_by(|a, b| b.2.cmp(&a.2));
        let in_regs = self.free.len().saturating_sub(l.depth);
        let spill_at = self.spill_next;
        let mut allocated = Vec::new();
        for (n, (key, first, _)) in owned.into_iter().enumerate() {
            let mut parent = key[..key.len() - 1].to_vec();
            while parent.last().is_some_and(|&(_, d)| d == 0) {
                parent.pop();
            }
            let pid = self.class_id(parent);
            let p = self.cls[pid].as_ref().expect("a live parent class");
            let (ploc, poff) = (p.loc, p.off);
            let preg = match ploc {
                Loc::Reg(r) => r,
                Loc::Mem(o) => {
                    self.spill_ld(T3, o);
                    T3
                }
            };
            let id = self.class_id(key);
            let off = 32 * first + SLIDE;
            let loc = if n < in_regs {
                let reg = self.free.pop().expect("a pointer register");
                self.a.add_imm(reg, preg, off - poff, T4);
                Loc::Reg(reg)
            } else {
                assert!(self.spill_next < SPILL_SLOTS, "spill slots");
                let o = self.spill_next;
                self.spill_next += 1;
                self.a.add_imm(T5, preg, off - poff, T4);
                self.spill_sd(T5, o, T4);
                self.spilled += 1;
                Loc::Mem(o)
            };
            self.cls[id] = Some(Class { loc, off });
            allocated.push(id);
        }
        // Small innermost bodies run `UNROLL` repetitions per trip, the rest
        // after the loop, each copy's cells shifted at compile time.
        // `NATIVE_VBIN_UNROLL` overrides the repetitions per trip.
        let unroll_by = std::env::var("NATIVE_VBIN_UNROLL")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(UNROLL);
        let unroll = if l.reps >= 2 * unroll_by
            && l.body.len() <= UNROLL_OPS
            && l.body.iter().all(|it| matches!(it, Item::Op(_)))
        {
            unroll_by
        } else {
            1
        };
        let trips = l.reps / unroll;
        let step = |keys: &[ClassKey], id: usize| -> i64 {
            keys[id]
                .iter()
                .find(|(lid, _)| *lid == l.id)
                .map_or(0, |&(_, d)| d)
        };
        let cnt = self.free.pop().expect("a counter register");
        self.a.li(cnt, trips as i64);
        self.flush_cursors();
        let top = self.a.here();
        let live: Vec<(usize, i64)> = self
            .cls
            .iter()
            .enumerate()
            .filter_map(|(id, c)| c.as_ref().map(|c| (id, c.off)))
            .collect();
        self.depth += 1;
        for copy in 0..unroll {
            for &(id, _) in &live {
                self.shift[id] = copy as i64 * step(&self.keys, id);
            }
            self.items(&l.body, &inner);
        }
        for &(id, _) in &live {
            self.shift[id] = 0;
        }
        self.depth -= 1;
        self.mark("loop");
        self.flush_cursors();
        for &(id, top_off) in &live {
            let c = self.cls[id].as_ref().unwrap();
            let (loc, off) = (c.loc, c.off);
            let corr = 32 * unroll as i64 * step(&self.keys, id) + top_off - off;
            match loc {
                Loc::Reg(reg) => self.a.add_imm(reg, reg, corr, T3),
                Loc::Mem(o) if corr != 0 => {
                    self.spill_ld(T3, o);
                    self.a.add_imm(T3, T3, corr, T4);
                    self.spill_sd(T3, o, T4);
                }
                Loc::Mem(_) => {}
            }
            self.cls[id].as_mut().unwrap().off = top_off;
        }
        self.a.addi(cnt, cnt, -1);
        self.a.bnez_back(cnt, top);
        let done = (trips * unroll) as i64;
        for &(id, top_off) in &live {
            let d = step(&self.keys, id);
            self.cls[id].as_mut().unwrap().off = top_off + 32 * done * d;
        }
        self.free.push(cnt);
        if unroll > 1 {
            self.depth += 1;
            for copy in 0..(l.reps % unroll) as i64 {
                for &(id, _) in &live {
                    self.shift[id] = (done + copy) * step(&self.keys, id);
                }
                self.items(&l.body, &inner);
            }
            for &(id, _) in &live {
                self.shift[id] = 0;
            }
            self.depth -= 1;
        }
        for id in allocated {
            if let Loc::Reg(r) = self.cls[id].take().unwrap().loc {
                self.free.push(r);
            }
        }
        self.spill_next = spill_at;
    }

    fn op(&mut self, k: usize, keys: &[usize]) {
        let at = self.a.pc();
        self.mark(kind(&self.prog.ops[k]));
        self.op_body(k, keys);
        self.mark("loop");
        let e = self.census.entry(kind(&self.prog.ops[k])).or_default();
        let c = &mut e[(self.depth > 0) as usize];
        c.0 += 1;
        c.1 += self.a.pc() - at;
    }

    fn op_body(&mut self, k: usize, keys: &[usize]) {
        let (op_cells, prog) = (self.op_cells, self.prog);
        let cells = &op_cells[k];
        let facts = &prog.facts[k];
        match prog.ops[k] {
            Op::Hash { mode } => {
                let (ni, no) = match mode {
                    3 => (3, 3),
                    4 => (4, 4),
                    _ => (2, 1),
                };
                let fwd = &prog.fwd[k];
                let regs = [DATA, [A2, A3, A4, A5]];
                // Inputs in the other slot first, before a copy overwrites it.
                for j in 0..ni {
                    if fwd[j] != 0 && fwd[j] as usize - 1 != j {
                        assert!(mode < 3, "a permutation reads its inputs in place");
                        for (l, &d) in regs[j].iter().enumerate() {
                            self.a.ld(d, A0, 32 * (fwd[j] as i64 - 1) + 8 * l as i64);
                        }
                    }
                }
                for j in 0..ni {
                    if fwd[j] as usize == j + 1 {
                        continue;
                    }
                    let rj = if mode >= 3 { DATA } else { regs[j] };
                    if fwd[j] == 0 {
                        let (r, i) = self.opnd(k, keys, j, false);
                        for (l, &d) in rj.iter().enumerate() {
                            self.a.ld(d, r, i + 8 * l as i64);
                        }
                    }
                    for (l, &d) in rj.iter().enumerate() {
                        self.a.sd(d, A0, 8 * (4 * j + l) as i64);
                    }
                }
                if mode < 3 {
                    self.a.sd(ZERO, A0, 64);
                    if mode == 0 {
                        self.a.sd(ZERO, A0, 72);
                    } else {
                        let dom = if mode == 1 {
                            DOMAIN_TRANSCRIPT
                        } else {
                            DOMAIN_LEAF
                        };
                        self.a.li(T0, dom);
                        self.a.sd(T0, A0, 72);
                    }
                    self.a.sd(ZERO, A0, 80);
                    self.a.sd(ZERO, A0, 88);
                }
                self.a.ecall();
                for j in (0..no).filter(|&j| fwd[ni + j] == 0) {
                    for (l, &d) in DATA.iter().enumerate() {
                        self.a.ld(d, A0, 8 * (4 * j + l) as i64);
                    }
                    let (r, i) = self.opnd(k, keys, ni + j, true);
                    for (l, &d) in DATA.iter().enumerate() {
                        self.a.sd(d, r, i + 8 * l as i64);
                    }
                }
            }
            Op::Walk4 => {
                // The bits, each a checked boolean.
                for (j, d) in [(0, T0), (1, T1)] {
                    let (r, i) = self.opnd(k, keys, j, false);
                    self.a.ld(d, r, i);
                    if facts[j] & UPPER_ZERO == 0 {
                        self.check_upper_zero(r, i, facts[j]);
                    }
                    self.a.sltiu(A5, d, 2);
                    self.a.and(OK, OK, A5);
                }
                // p = b0 + 2·b1.
                self.a.slli(T1, T1, 1);
                self.a.add(T0, T0, T1);
                // The current digest first: it may sit in a slot a sibling
                // is about to take.
                let cur = [A2, A3, A4, A5];
                let (r, i) = self.opnd(k, keys, 2, false);
                for (l, &d) in cur.iter().enumerate() {
                    self.a.ld(d, r, i + 8 * l as i64);
                }
                // Sibling `q` goes to slot `q + (p ≤ q)`, branch-free.
                let sib = [T1, T2, A1, RA];
                for q in 0..3 {
                    let (r, i) = self.opnd(k, keys, 3 + q, false);
                    for (l, &d) in sib.iter().enumerate() {
                        self.a.ld(d, r, i + 8 * l as i64);
                    }
                    let t = self.next_tmp();
                    self.a.sltiu(t, T0, q as i64 + 1);
                    self.a.slli(t, t, 5);
                    self.a.add(t, t, A0);
                    for (l, &d) in sib.iter().enumerate() {
                        self.a.sd(d, t, 32 * q as i64 + 8 * l as i64);
                    }
                }
                let t = self.next_tmp();
                self.a.slli(t, T0, 5);
                self.a.add(t, t, A0);
                for (l, &d) in cur.iter().enumerate() {
                    self.a.sd(d, t, 8 * l as i64);
                }
                self.a.ecall();
                let fwd = &prog.fwd[k];
                for j in (0..4).filter(|&j| fwd[6 + j] == 0) {
                    for (l, &d) in DATA.iter().enumerate() {
                        self.a.ld(d, A0, 8 * (4 * j + l) as i64);
                    }
                    let (r, i) = self.opnd(k, keys, 6 + j, true);
                    for (l, &d) in DATA.iter().enumerate() {
                        self.a.sd(d, r, i + 8 * l as i64);
                    }
                }
            }
            Op::Select => {
                let (r, i) = self.opnd(k, keys, 0, false);
                self.a.ld(T0, r, i);
                if facts[0] & UPPER_ZERO == 0 {
                    self.check_upper_zero(r, i, facts[0]);
                }
                self.a.sltiu(A5, T0, 2);
                self.a.and(OK, OK, A5);
                if prog.fwd[k] == [0, 1, 0, 1, 2] {
                    // A Merkle step: the left word already in slot 0 stays
                    // there unless the bit swaps it.
                    let right = [A2, A3, A4, A5];
                    let left = [T1, T2, A1, RA];
                    let (r, i) = self.opnd(k, keys, 2, false);
                    for (l, &d) in right.iter().enumerate() {
                        self.a.ld(d, r, i + 8 * l as i64);
                    }
                    let (keep, end) = (self.a.label(), self.a.label());
                    self.a.beq(T0, ZERO, keep);
                    for (l, &d) in left.iter().enumerate() {
                        self.a.ld(d, A0, 8 * l as i64);
                    }
                    for l in 0..4 {
                        self.a.sd(right[l], A0, 8 * l as i64);
                        self.a.sd(left[l], A0, 32 + 8 * l as i64);
                    }
                    self.a.jal(ZERO, end);
                    self.a.bind(keep);
                    if same_shape_paths() {
                        // The swap path's loads, stores and jump, so the
                        // tables' heights do not depend on the bits.
                        for (l, &d) in left.iter().enumerate() {
                            self.a.ld(d, A0, 8 * l as i64);
                        }
                        for (l, &d) in left.iter().enumerate() {
                            self.a.sd(d, A0, 8 * l as i64);
                        }
                    }
                    for (l, &d) in right.iter().enumerate() {
                        self.a.sd(d, A0, 32 + 8 * l as i64);
                    }
                    if same_shape_paths() {
                        self.a.jal(ZERO, end);
                    }
                    self.a.bind(end);
                    return;
                }
                self.a.mv(RA, T0);
                let left = [T0, T1, T2, A1];
                let right = [A2, A3, A4, A5];
                for (j, regs) in [(1, left), (2, right)] {
                    let (r, i) = self.opnd(k, keys, j, false);
                    for (l, &d) in regs.iter().enumerate() {
                        self.a.ld(d, r, i + 8 * l as i64);
                    }
                }
                // `out_l` by address, so that a slide for `out_r` keeps it.
                let (r, i) = self.opnd(k, keys, 3, true);
                let ol = self.next_tmp();
                self.a.addi(ol, r, i);
                let (or, oi) = self.opnd(k, keys, 4, true);
                let (straight, end) = (self.a.label(), self.a.label());
                self.a.beq(RA, ZERO, straight);
                for (x, y) in [(right, left), (left, right)] {
                    for l in 0..4 {
                        self.a.sd(x[l], ol, 8 * l as i64);
                        self.a.sd(y[l], or, oi + 8 * l as i64);
                    }
                    if x == right {
                        self.a.jal(ZERO, end);
                        self.a.bind(straight);
                    }
                }
                if same_shape_paths() {
                    self.a.jal(ZERO, end);
                }
                self.a.bind(end);
            }
            Op::Pack => {
                for (j, &d) in DATA.iter().enumerate() {
                    let (r, i) = self.opnd(k, keys, j, false);
                    self.a.ld(d, r, i);
                    if facts[j] & UPPER_ZERO == 0 {
                        self.check_upper_zero(r, i, facts[j]);
                    }
                }
                let (r, i) = self.opnd(k, keys, 4, true);
                for (l, &d) in DATA.iter().enumerate() {
                    self.a.sd(d, r, i + 8 * l as i64);
                }
            }
            Op::Unpack { mask } => {
                let (r, i) = self.opnd(k, keys, 0, false);
                for (l, &d) in DATA.iter().enumerate() {
                    self.a.ld(d, r, i + 8 * l as i64);
                }
                for (l, &d) in DATA.iter().enumerate() {
                    let (r, i) = self.opnd(k, keys, 1 + l, true);
                    self.a.sd(d, r, i);
                }
                for (l, &d) in DATA.iter().enumerate() {
                    if mask >> l & 1 != 0 {
                        self.push_input(&[d, ZERO, ZERO]);
                    }
                }
            }
            Op::BitDec { n, halves } => {
                let (r, i) = self.opnd(k, keys, 0, false);
                self.a.ld(T0, r, i);
                if facts[0] & UPPER_ZERO == 0 {
                    self.check_upper_zero(r, i, facts[0]);
                }
                for b in 0..n as usize {
                    if b == 0 {
                        self.a.andi(T1, T0, 1);
                    } else {
                        self.a.srli(T1, T0, b as u32);
                        self.a.andi(T1, T1, 1);
                    }
                    let (r, i) = self.opnd(k, keys, 1 + b, true);
                    self.a.sd(T1, r, i);
                }
                if halves {
                    let bswap = self.rt.bswap.expect("an emitted routine");
                    for h in 0..2 {
                        if h == 0 {
                            self.a.srli(A1, T0, 32);
                        } else {
                            self.a.slli(A1, T0, 32);
                            self.a.srli(A1, A1, 32);
                        }
                        self.a.call(T6, bswap);
                        let j = 1 + n as usize + h;
                        let (r, i) = self.opnd(k, keys, j, true);
                        self.a.sd(A1, r, i);
                    }
                }
            }
            Op::Keccak { absorb, rev } => {
                self.a.li(A1, ARGS as i64);
                let reads = 13 + if absorb { 9 } else { 0 };
                for j in 0..cells.len() {
                    let (r, i) = self.opnd(k, keys, j, j >= reads);
                    self.a.addi(T0, r, i);
                    self.a.sd(T0, A1, 8 * j as i64);
                }
                let v = absorb as usize * 2 + rev as usize;
                let l = self.rt.keccak[v].expect("an emitted routine");
                self.a.call(RA, l);
            }
            Op::Hint { flags } => {
                // Lanes a type requires zero are checked zero, which makes
                // them canonical and leaves them unwritten (the cell is
                // fresh), or the guest aborts. A hint both halves read takes
                // lanes 0..3 from its IN cell, which the pseudo-PAGE binds
                // to the Field VM's and keeps canonical, instead of reading
                // and comparing the hint's: lane 3 still comes from the hint.
                // A hint whose lanes only reach RPX inputs is not checked
                // canonical here (the RPX table does), and one whose readers
                // read it in place is not copied.
                let shared = flags & 1 != 0;
                let ty = flags >> 1;
                let free = match ty {
                    1 => 3,
                    2 => 1,
                    _ => 4,
                };
                let to_rpx = prog.rpx_only[prog.cells[k][0] as usize];
                let fw = prog.fwd[k][0];
                let in_place = fw == IN_CELL || fw == IN_STREAM;
                let zero_lanes: &[usize] = match ty {
                    1 => &[3],
                    2 => &[1, 2, 3],
                    _ => &[],
                };
                let lanes_in = stream_lanes(flags);
                let packed = packed_hints();
                for (l, &d) in DATA.iter().enumerate() {
                    let from_in = shared && l < 3;
                    let canon = l < free && !to_rpx && !from_in;
                    if packed && !from_in && zero_lanes.contains(&l) {
                        if !in_place && fw != 0 {
                            self.a.li(d, 0);
                        }
                        continue;
                    }
                    if in_place && !zero_lanes.contains(&l) && !canon {
                        continue;
                    }
                    let (r, i) = if from_in {
                        self.cursor(INC, 8 * l as i64)
                    } else {
                        let p = lanes_in
                            .iter()
                            .position(|&x| x == l)
                            .expect("a stream lane");
                        self.cursor(HINTS, 8 * p as i64)
                    };
                    self.a.ld(d, r, i);
                    if canon {
                        self.a.sltu(A2, d, PR);
                        self.a.and(OK, OK, A2);
                    }
                }
                if !in_place {
                    let (r, i) = self.opnd(k, keys, 0, true);
                    let lanes = if fw != 0 { 4 } else { free };
                    for (l, &d) in DATA.iter().enumerate().take(lanes) {
                        self.a.sd(d, r, i + 8 * l as i64);
                    }
                }
                match ty {
                    0 => {}
                    1 | 2 if packed => {
                        if shared && ty == 2 {
                            self.a.or(ERR, ERR, T1);
                            self.a.or(ERR, ERR, T2);
                        }
                    }
                    1 => self.a.or(ERR, ERR, A1),
                    2 => {
                        for d in [T1, T2, A1] {
                            self.a.or(ERR, ERR, d);
                        }
                    }
                    _ => self.a.ori(ERR, ERR, 1),
                }
                self.advance(8 * lanes_in.len() as i64, if shared { 24 } else { 0 });
            }
            Op::BaseAlu { op: 1 } => {
                for (j, d) in [(0, A1), (1, A2)] {
                    let (r, i) = self.opnd(k, keys, j, false);
                    self.a.ld(d, r, i);
                    if facts[j] & UPPER_ZERO == 0 {
                        self.check_upper_zero(r, i, facts[j]);
                    }
                }
                self.a.sltu(T0, A1, A2);
                self.a.sub(A1, A1, A2);
                self.a.neg(T0, T0);
                self.a.and(T0, T0, PR);
                self.a.add(A1, A1, T0);
                let (r, i) = self.opnd(k, keys, 2, true);
                self.a.sd(A1, r, i);
            }
            Op::BaseAlu { op: 3 } if facts[1] & ZERO_CONST != 0 => {
                let (r, i) = self.opnd(k, keys, 0, false);
                self.a.ld(A1, r, i);
                if facts[0] & UPPER_ZERO == 0 {
                    self.check_upper_zero(r, i, facts[0]);
                }
                self.a.or(ERR, ERR, A1);
                self.a.li(A1, 1);
                let (r, i) = self.opnd(k, keys, 2, true);
                self.a.sd(A1, r, i);
            }
            Op::BaseAlu { op } => {
                let n = cells.len() - 1;
                for (j, &f) in facts.iter().enumerate().take(n) {
                    let (r, i) = self.opnd(k, keys, j, false);
                    if f & UPPER_ZERO == 0 {
                        self.check_upper_zero(r, i, f);
                    }
                    self.a.addi(A1 + j as Reg, r, i);
                }
                let (r, i) = self.opnd(k, keys, n, true);
                self.a.addi(A4, r, i);
                let l = self.rt.base[op as usize].expect("an emitted routine");
                self.a.call(RA, l);
            }
            Op::ExtAlu { op } => {
                let n = cells.len() - 1;
                for (j, &f) in facts.iter().enumerate().take(n) {
                    let (r, i) = self.opnd(k, keys, j, false);
                    if j == 1 && op == 5 {
                        if f & UPPER_ZERO == 0 {
                            self.check_upper_zero(r, i, f);
                        }
                    } else if f & L3_ZERO == 0 {
                        self.a.ld(A5, r, i + 24);
                        self.a.or(ERR, ERR, A5);
                    }
                    self.a.addi(A1 + j as Reg, r, i);
                }
                let (r, i) = self.opnd(k, keys, n, true);
                self.a.addi(A4, r, i);
                let l = self.rt.ext[op as usize].expect("an emitted routine");
                self.a.call(RA, l);
            }
            Op::Public => {
                let (r, i) = self.opnd(k, keys, 0, false);
                for (l, &d) in DATA.iter().enumerate() {
                    self.a.ld(d, r, i + 8 * l as i64);
                    self.a.sd(d, PUBC, 8 * l as i64);
                }
                self.a.addi(PUBC, PUBC, 32);
            }
        }
    }

    /// The next IN cell must hold `lanes` (`ZERO` for a zero lane).
    fn push_input(&mut self, lanes: &[Reg; 3]) {
        for (l, &d) in lanes.iter().enumerate() {
            let (r, i) = self.cursor(INC, 8 * l as i64);
            self.a.ld(A5, r, i);
            if d != ZERO {
                self.a.xor(A5, A5, d);
            }
            self.a.or(ERR, ERR, A5);
        }
        self.advance(0, 24);
    }
}

const KECCAK_RC: [u64; 24] = [
    0x0000000000000001,
    0x0000000000008082,
    0x800000000000808a,
    0x8000000080008000,
    0x000000000000808b,
    0x0000000080000001,
    0x8000000080008081,
    0x8000000000008009,
    0x000000000000008a,
    0x0000000000000088,
    0x0000000080008009,
    0x000000008000000a,
    0x000000008000808b,
    0x800000000000008b,
    0x8000000000008089,
    0x8000000000008003,
    0x8000000000008002,
    0x8000000000000080,
    0x000000000000800a,
    0x800000008000000a,
    0x8000000080008081,
    0x8000000000008080,
    0x0000000080000001,
    0x8000000080008008,
];

/// Rotation of lane `x + 5y`, `KECCAK_RHO[x][y]`.
const KECCAK_RHO: [[u32; 5]; 5] = [
    [0, 36, 3, 41, 18],
    [1, 44, 10, 45, 2],
    [62, 6, 43, 15, 61],
    [28, 55, 25, 21, 56],
    [27, 20, 39, 8, 14],
];

/// `rd = rs <<< n`, through `T1` and `T2`.
fn rotl(a: &mut Asm, rd: Reg, rs: Reg, n: u32) {
    if n == 0 {
        a.mv(rd, rs);
    } else {
        a.slli(T1, rs, n);
        a.srli(T2, rs, 64 - n);
        a.or(rd, T1, T2);
    }
}

/// Keccak-f[1600] in software on the 25 lanes at `T3`, the round constants
/// at `rc` (no accelerator, so the proof carries no keccak tables); `FTMP`
/// holds the permuted lanes. Clobbers `T0..T2`, `T4..T6`, `A1..A5`.
fn keccak_f(a: &mut Asm, rc: i64) {
    let lane = |x: usize, y: usize| 8 * (x + 5 * y) as i64;
    let c = [A1, A2, A3, A4, T5];
    a.li(T4, FTMP as i64);
    a.li(T6, rc);
    a.li(A5, 24);
    let top = a.here();
    for (x, &cx) in c.iter().enumerate() {
        a.ld(cx, T3, lane(x, 0));
        for y in 1..5 {
            a.ld(T0, T3, lane(x, y));
            a.xor(cx, cx, T0);
        }
    }
    for x in 0..5 {
        rotl(a, T0, c[(x + 1) % 5], 1);
        a.xor(T0, T0, c[(x + 4) % 5]);
        for y in 0..5 {
            a.ld(T1, T3, lane(x, y));
            a.xor(T1, T1, T0);
            a.sd(T1, T3, lane(x, y));
        }
    }
    for (x, rho) in KECCAK_RHO.iter().enumerate() {
        for (y, &r) in rho.iter().enumerate() {
            a.ld(T0, T3, lane(x, y));
            rotl(a, T0, T0, r);
            a.sd(T0, T4, lane(y, (2 * x + 3 * y) % 5));
        }
    }
    for y in 0..5 {
        for (x, &cx) in c.iter().enumerate() {
            a.ld(cx, T4, lane(x, y));
        }
        for x in 0..5 {
            a.xori(T0, c[(x + 1) % 5], -1);
            a.and(T0, T0, c[(x + 2) % 5]);
            a.xor(T0, T0, c[x]);
            a.sd(T0, T3, lane(x, y));
        }
    }
    a.ld(T0, T3, 0);
    a.ld(T1, T6, 0);
    a.xor(T0, T0, T1);
    a.sd(T0, T3, 0);
    a.addi(T6, T6, 8);
    a.addi(A5, A5, -1);
    a.bnez_back(A5, top);
}

/// `x` canonical, through `t`.
fn canon(a: &mut Asm, x: Reg, t: Reg) {
    a.sltu(t, x, PR);
    a.addi(t, t, -1);
    a.and(t, t, PR);
    a.sub(x, x, t);
}

/// Scratch slots of the extension routines, 8 bytes each from `FTMP` (in `T3`).
/// `NATIVE_VBIN_SAME_SHAPE`: both arms of a select run the same loads,
/// stores and jumps, so how many of each the guest executes, hence its
/// tables' heights, is a function of the program alone.
fn same_shape_paths() -> bool {
    std::env::var_os("NATIVE_VBIN_SAME_SHAPE").is_some()
}

/// The hint stream holds only the lanes the guest takes from it
/// ([`stream_lanes`]), not four per hint; `NATIVE_VBIN_WIDE_HINTS` turns it off.
pub fn packed_hints() -> bool {
    std::env::var_os("NATIVE_VBIN_WIDE_HINTS").is_none()
}

/// The lanes of a hint the guest reads from the stream, in stream order.
/// Packed: a type's zero lanes are absent (zero by construction), and a
/// shared hint keeps only what its IN cell does not hold (lane 3 of a word).
fn stream_lanes(flags: u32) -> &'static [usize] {
    if !packed_hints() {
        return &[0, 1, 2, 3];
    }
    match (flags & 1 != 0, flags >> 1) {
        (true, 0) => &[3],
        (true, 1 | 2) => &[],
        (false, 1) => &[0, 1, 2],
        (false, 2) => &[0],
        _ => &[0, 1, 2, 3],
    }
}

/// Per hint of the program `words`, in order, the lanes its stream entry
/// holds.
pub fn hint_stream_lanes(words: &[u32]) -> Option<Vec<&'static [usize]>> {
    let prog = decode(words)?;
    Some(
        prog.ops
            .iter()
            .filter_map(|op| match *op {
                Op::Hint { flags } => Some(stream_lanes(flags)),
                _ => None,
            })
            .collect(),
    )
}

/// Cells share memory slots ([`reuse`]), so the guest touches memory in
/// proportion to its live cells, not to all of them; `NATIVE_VBIN_NO_REUSE`
/// turns it off.
fn reuse_slots() -> bool {
    std::env::var_os("NATIVE_VBIN_NO_REUSE").is_none()
}

/// Lanes the code of `op` stores for each write, a prefix of the word
/// (`None`: not known, the cell keeps a slot of its own).
fn lanes_written(op: &Op) -> Option<usize> {
    match *op {
        Op::Hash { .. } | Op::Select | Op::Pack | Op::Walk4 => Some(4),
        Op::Unpack { .. } | Op::BaseAlu { .. } | Op::BitDec { .. } => Some(1),
        Op::ExtAlu { .. } => Some(3),
        Op::Hint { flags } => Some(match flags >> 1 {
            1 => 3,
            2 => 1,
            _ => 4,
        }),
        Op::Keccak { .. } | Op::Public => None,
    }
}

/// Memory slots for the cells: a ring of `2^log2` slots per number of lanes
/// written, cell `c` at `c mod 2^log2` of its ring when the cell that held
/// that slot before was last read no later than `c` is written (the cell
/// indices keep their per-repetition moves, so re-rolled loops stay loops);
/// IN cells, constants, cells read before any write, BACK cells and cells
/// a ring cannot hold keep slots of their own. A slot's writers all store
/// the same lanes, so the lanes they leave unwritten stay zero, as with one
/// slot per cell. Per op its slots, per cell its slot, and the slot count.
fn reuse(prog: &Program, log2: u32) -> (Vec<Vec<i64>>, Vec<u32>, u32) {
    const NONE: u32 = u32::MAX;
    let n = prog.n_cells as usize;
    let w = 1usize << log2;
    let mut write = vec![usize::MAX; n];
    let mut class = vec![None; n];
    let mut last = vec![0usize; n];
    let mut fixed = vec![false; n];
    for c in fixed.iter_mut().take(prog.n_out as usize) {
        *c = true;
    }
    for (c, _) in &prog.consts {
        fixed[*c as usize] = true;
    }
    for e in &prog.back {
        fixed[(e & ((1 << 30) - 1)) as usize] = true;
    }
    for (k, cells) in prog.cells.iter().enumerate() {
        let nr = prog.reads[k];
        for &c in &cells[..nr] {
            let c = c as usize;
            if write[c] == usize::MAX {
                fixed[c] = true;
            }
            last[c] = k;
        }
        for &c in &cells[nr..] {
            let c = c as usize;
            write[c] = k;
            last[c] = last[c].max(k);
            match lanes_written(&prog.ops[k]) {
                Some(l) => class[c] = Some(l),
                None => fixed[c] = true,
            }
        }
    }
    // Ring placement in cell order: the previous holder must be dead.
    let mut holder: HashMap<(usize, usize), usize> = HashMap::new();
    let mut ring = vec![false; n];
    for c in 0..n {
        let Some(l) = class[c] else { continue };
        if fixed[c] || write[c] == usize::MAX {
            continue;
        }
        let key = (l, c % w);
        let ok = match holder.get(&key) {
            None => true,
            Some(&p) => {
                last[p] < write[c]
                    || (last[p] == write[c]
                        && write[p] < write[c]
                        && !matches!(prog.ops[write[c]], Op::Keccak { .. }))
            }
        };
        if ok {
            holder.insert(key, c);
            ring[c] = true;
        }
    }
    let mut slot = vec![NONE; n];
    let mut next = prog.n_out;
    for (c, s) in slot.iter_mut().enumerate().take(prog.n_out as usize) {
        *s = c as u32;
    }
    for c in prog.n_out as usize..n {
        if !ring[c] {
            slot[c] = next;
            next += 1;
        }
    }
    let rings: Vec<usize> = {
        let mut v: Vec<usize> = class.iter().flatten().copied().collect();
        v.sort_unstable();
        v.dedup();
        v
    };
    for c in 0..n {
        if ring[c] {
            let r = rings.iter().position(|&l| Some(l) == class[c]).unwrap();
            slot[c] = next + (r * w + c % w) as u32;
        }
    }
    let total = next + (rings.len() * w) as u32;
    let out = prog
        .cells
        .iter()
        .map(|cells| cells.iter().map(|&c| slot[c as usize] as i64).collect())
        .collect();
    (out, slot, total)
}

fn slot(s: usize) -> i64 {
    8 * s as i64
}

impl Gen<'_> {
    /// `slot d = f(slot x, slot y)` for a leaf field helper `f`.
    fn f2(&mut self, f: Label, d: usize, x: usize, y: usize) {
        self.a.ld(A1, T3, slot(x));
        self.a.ld(A2, T3, slot(y));
        self.a.jal(T6, f);
        self.a.sd(A1, T3, slot(d));
    }

    fn fp3_mul(&mut self, out: [usize; 3], x: [usize; 3], y: [usize; 3], t: usize) {
        let (fadd, fmul) = (self.rt.fadd.unwrap(), self.rt.fmul.unwrap());
        let (b12, b22, m1, m2, m3, s) = (t, t + 1, t + 2, t + 3, t + 4, t + 5);
        self.f2(fadd, b12, y[1], y[1]);
        self.f2(fadd, b22, y[2], y[2]);
        let terms = [
            [(x[0], y[0]), (x[1], b22), (x[2], b12)],
            [(x[0], y[1]), (x[1], y[0]), (x[2], b22)],
            [(x[0], y[2]), (x[1], y[1]), (x[2], y[0])],
        ];
        for (o, [p1, p2, p3]) in out.into_iter().zip(terms) {
            self.f2(fmul, m1, p1.0, p1.1);
            self.f2(fmul, m2, p2.0, p2.1);
            self.f2(fmul, m3, p3.0, p3.1);
            self.f2(fadd, s, m2, m3);
            self.f2(fadd, o, m1, s);
        }
    }

    /// The routines the ops use, before the code that calls them.
    fn runtime(&mut self) {
        self.mark("rt_field");
        for op in &self.prog.ops {
            match *op {
                Op::BaseAlu { op } => {
                    rt_label(&mut self.a, &mut self.rt.base[op as usize]);
                }
                Op::ExtAlu { op } => {
                    rt_label(&mut self.a, &mut self.rt.ext[op as usize]);
                }
                Op::Keccak { absorb, rev } => {
                    rt_label(
                        &mut self.a,
                        &mut self.rt.keccak[absorb as usize * 2 + rev as usize],
                    );
                }
                _ => {}
            }
        }
        // Leaf helpers, link T6: A1 = f(A1, A2).
        let fadd = rt_label(&mut self.a, &mut self.rt.fadd);
        let fsub = rt_label(&mut self.a, &mut self.rt.fsub);
        let fmul = rt_label(&mut self.a, &mut self.rt.fmul);
        let finv = rt_label(&mut self.a, &mut self.rt.finv);
        let a = &mut self.a;
        a.bind(fadd);
        a.add(T0, A1, A2);
        a.sltu(T1, T0, A1);
        a.neg(T1, T1);
        a.srli(T1, T1, 32);
        a.add(T0, T0, T1);
        a.sltu(T1, T0, T1);
        a.neg(T1, T1);
        a.srli(T1, T1, 32);
        a.add(A1, T0, T1);
        canon(a, A1, T0);
        a.ret_via(T6);

        a.bind(fsub);
        a.sltu(T0, A1, A2);
        a.sub(A1, A1, A2);
        a.neg(T0, T0);
        a.and(T0, T0, PR);
        a.add(A1, A1, T0);
        a.ret_via(T6);

        a.bind(fmul);
        a.mul(T0, A1, A2);
        a.mulhu(T1, A1, A2);
        a.srli(T2, T1, 32);
        a.sltu(A1, T0, T2);
        a.sub(T0, T0, T2);
        a.neg(A1, A1);
        a.srli(A1, A1, 32);
        a.sub(T0, T0, A1);
        a.slli(T1, T1, 32);
        a.srli(T1, T1, 32);
        a.slli(T2, T1, 32);
        a.sub(T2, T2, T1);
        a.add(A1, T0, T2);
        a.sltu(T0, A1, T0);
        a.neg(T0, T0);
        a.srli(T0, T0, 32);
        a.add(A1, A1, T0);
        canon(a, A1, T0);
        a.ret_via(T6);

        // finv, link T5: A1 = A1^(p − 2); clobbers T0..T2, T4, A2, A3, A5.
        a.bind(finv);
        a.mv(T4, A1);
        a.li(A3, (P - 2) as i64);
        a.li(A1, 1);
        let top = a.here();
        let sq = a.label();
        a.andi(T0, A3, 1);
        a.beq(T0, ZERO, sq);
        a.mv(A2, T4);
        a.jal(T6, fmul);
        a.bind(sq);
        a.mv(A5, A1);
        a.mv(A1, T4);
        a.mv(A2, T4);
        a.jal(T6, fmul);
        a.mv(T4, A1);
        a.mv(A1, A5);
        a.srli(A3, A3, 1);
        a.bne(A3, ZERO, top);
        a.ret_via(T5);

        // bswap, link T6: A1 = the byte swap of the u32 in A1; clobbers T1, T2.
        let bswap = rt_label(a, &mut self.rt.bswap);
        a.bind(bswap);
        a.andi(T1, A1, 0xff);
        a.slli(T1, T1, 24);
        for (sh, to) in [(8, 16), (16, 8), (24, 0)] {
            a.srli(T2, A1, sh);
            a.andi(T2, T2, 0xff);
            if to > 0 {
                a.slli(T2, T2, to);
            }
            a.or(T1, T1, T2);
        }
        a.mv(A1, T1);
        a.ret_via(T6);

        // Moved base instructions, link RA: A1 = &a, A2 = &b, A3 = &c, A4 = &out.
        for op in 0..5 {
            let Some(l) = self.rt.base[op] else { continue };
            let a = &mut self.a;
            a.bind(l);
            a.ld(A1, A1, 0);
            a.ld(A2, A2, 0);
            match op {
                0 => a.jal(T6, fadd),
                1 => a.jal(T6, fsub),
                2 => a.jal(T6, fmul),
                3 => {
                    let (zc, st) = (a.label(), a.label());
                    a.beq(A2, ZERO, zc);
                    a.mv(T3, A1);
                    a.mv(A1, A2);
                    a.jal(T5, finv);
                    a.mv(A2, T3);
                    a.jal(T6, fmul);
                    a.jal(ZERO, st);
                    a.bind(zc);
                    a.or(ERR, ERR, A1);
                    a.li(A1, 1);
                    a.bind(st);
                }
                _ => {
                    a.jal(T6, fmul);
                    a.ld(A2, A3, 0);
                    a.jal(T6, fadd);
                }
            }
            a.sd(A1, A4, 0);
            a.ret_via(RA);
        }

        // Moved extension instructions, over scratch slots: a 0..3, b 3..6,
        // c 6..9, out 9..12, temporaries from 12.
        for op in 0..6 {
            let Some(l) = self.rt.ext[op] else { continue };
            self.a.bind(l);
            self.a.li(T3, FTMP as i64);
            let b_lanes = if op == 5 { 1 } else { 3 };
            for (src, first, n) in [(A1, 0, 3), (A2, 3, b_lanes), (A3, 6, 3)] {
                if src == A3 && op != 4 {
                    continue;
                }
                for l in 0..n {
                    self.a.ld(T0, src, 8 * l as i64);
                    self.a.sd(T0, T3, slot(first + l));
                }
            }
            let (av, bv, cv, out) = ([0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]);
            match op {
                0 | 1 => {
                    let f = if op == 0 { fadd } else { fsub };
                    for l in 0..3 {
                        self.f2(f, out[l], av[l], bv[l]);
                    }
                }
                2 => self.fp3_mul(out, av, bv, 12),
                3 => self.fp3_div(out, av, bv),
                4 => {
                    let m = [12, 13, 14];
                    self.fp3_mul(m, av, bv, 15);
                    for l in 0..3 {
                        self.f2(fadd, out[l], m[l], cv[l]);
                    }
                }
                _ => {
                    for l in 0..3 {
                        self.f2(fmul, out[l], av[l], 3);
                    }
                }
            }
            for (l, &o) in out.iter().enumerate() {
                self.a.ld(T0, T3, slot(o));
                self.a.sd(T0, A4, 8 * l as i64);
            }
            self.a.ret_via(RA);
        }

        for v in 0..4 {
            if let Some(l) = self.rt.keccak[v] {
                self.keccak(l, v >= 2, v % 2 == 1);
            }
        }
    }

    /// The extension division `a / b` over slots, the LFM's `0/0 = 1`.
    fn fp3_div(&mut self, out: [usize; 3], av: [usize; 3], bv: [usize; 3]) {
        let (fadd, fsub, fmul, finv) = (
            self.rt.fadd.unwrap(),
            self.rt.fsub.unwrap(),
            self.rt.fmul.unwrap(),
            self.rt.finv.unwrap(),
        );
        let (nz, done) = (self.a.label(), self.a.label());
        self.a.ld(T0, T3, slot(bv[0]));
        for &b in &bv[1..] {
            self.a.ld(T1, T3, slot(b));
            self.a.or(T0, T0, T1);
        }
        self.a.bne(T0, ZERO, nz);
        for &x in &av {
            self.a.ld(T1, T3, slot(x));
            self.a.or(ERR, ERR, T1);
        }
        self.a.li(T0, 1);
        self.a.sd(T0, T3, slot(out[0]));
        self.a.sd(ZERO, T3, slot(out[1]));
        self.a.sd(ZERO, T3, slot(out[2]));
        self.a.jal(ZERO, done);
        self.a.bind(nz);
        let [b0, b1, b2] = bv;
        let mut next = 12;
        let mut fresh = || {
            next += 1;
            next - 1
        };
        let (s0, s1, s2, t01, t, four, six) = (
            fresh(),
            fresh(),
            fresh(),
            fresh(),
            fresh(),
            fresh(),
            fresh(),
        );
        self.f2(fmul, s0, b0, b0);
        self.f2(fmul, s1, b1, b1);
        self.f2(fmul, s2, b2, b2);
        self.f2(fmul, t01, b0, b1);
        self.f2(fmul, t, t01, b2);
        self.a.li(T0, 4);
        self.a.sd(T0, T3, slot(four));
        self.a.li(T0, 6);
        self.a.sd(T0, T3, slot(six));
        // norm = s0·b0 + (s1·b1 + s1·b1) + 4·(s2·b2) − 6·t
        let (x0, x1, x2, x3, x4, x5, x6, x7) = (
            fresh(),
            fresh(),
            fresh(),
            fresh(),
            fresh(),
            fresh(),
            fresh(),
            fresh(),
        );
        self.f2(fmul, x0, s0, b0);
        self.f2(fmul, x1, s1, b1);
        self.f2(fadd, x2, x1, x1);
        self.f2(fmul, x3, s2, b2);
        self.f2(fmul, x4, four, x3);
        self.f2(fadd, x5, x2, x4);
        self.f2(fadd, x6, x0, x5);
        self.f2(fmul, x7, six, t);
        let norm = fresh();
        self.f2(fsub, norm, x6, x7);
        let ni = fresh();
        self.a.ld(A1, T3, slot(norm));
        self.a.jal(T5, finv);
        self.a.sd(A1, T3, slot(ni));
        let (y0, y1, y2, inv0, y3, y4, inv1, y5, inv2) = (
            fresh(),
            fresh(),
            fresh(),
            fresh(),
            fresh(),
            fresh(),
            fresh(),
            fresh(),
            fresh(),
        );
        // inv0 = (s0 − (b1·b2 + b1·b2))·ni
        self.f2(fmul, y0, b1, b2);
        self.f2(fadd, y1, y0, y0);
        self.f2(fsub, y2, s0, y1);
        self.f2(fmul, inv0, y2, ni);
        // inv1 = ((s2 + s2) − b0·b1)·ni
        self.f2(fadd, y3, s2, s2);
        self.f2(fsub, y4, y3, t01);
        self.f2(fmul, inv1, y4, ni);
        // inv2 = (s1 − b0·b2)·ni
        self.f2(fmul, y5, b0, b2);
        let y6 = fresh();
        self.f2(fsub, y6, s1, y5);
        self.f2(fmul, inv2, y6, ni);
        let t = fresh();
        self.fp3_mul(out, av, [inv0, inv1, inv2], t);
        self.a.bind(done);
    }

    /// The `keccak` instruction, link RA, over the cell addresses in `ARGS`:
    /// the 13 state cells, the 9 block cells when absorbing, the 13 output
    /// cells, then the 2 digest cells.
    fn keccak(&mut self, l: Label, absorb: bool, rev: bool) {
        self.mark("rt_keccak");
        let bswap = rt_label(&mut self.a, &mut self.rt.bswap);
        let a = &mut self.a;
        a.bind(l);
        a.li(T3, KSTATE as i64);
        a.li(T4, ARGS as i64);
        for q in 0..25 {
            a.sd(ZERO, T3, 8 * q);
        }
        let groups: &[(i64, i64, bool)] = if absorb {
            &[(0, 50, false), (13, 34, true)]
        } else {
            &[(0, 50, false)]
        };
        for &(first, live, xor) in groups {
            let cells = if xor { 9 } else { 13 };
            for h in 0..4 * cells {
                a.ld(T5, T4, 8 * (first + h / 4));
                a.ld(T0, T5, 8 * (h % 4));
                if h >= live {
                    a.or(ERR, ERR, T0);
                    continue;
                }
                a.srli(T1, T0, 32);
                a.or(ERR, ERR, T1);
                if h % 2 == 1 {
                    a.slli(T0, T0, 32);
                }
                a.ld(T1, T3, 8 * (h / 2));
                if xor {
                    a.xor(T1, T1, T0);
                } else {
                    a.or(T1, T1, T0);
                }
                a.sd(T1, T3, 8 * (h / 2));
            }
        }
        keccak_f(a, self.rc_addr);
        a.li(T4, ARGS as i64);
        let outs = 13 + if absorb { 9 } else { 0 };
        let half = |a: &mut Asm, rd: Reg, q: i64| {
            a.ld(rd, T3, 8 * (q / 2));
            if q % 2 == 1 {
                a.srli(rd, rd, 32);
            } else {
                a.slli(rd, rd, 32);
                a.srli(rd, rd, 32);
            }
        };
        for j in 0..13 {
            a.ld(T5, T4, 8 * (outs + j));
            for l in 0..4 {
                let h = 4 * j + l;
                if h < 50 {
                    half(a, T0, h);
                    a.sd(T0, T5, 8 * l);
                } else {
                    a.sd(ZERO, T5, 8 * l);
                }
            }
        }
        if rev {
            for w in 0..2 {
                a.ld(T5, T4, 8 * (outs + 13 + w));
                for l in 0..4 {
                    half(a, A1, 7 - (4 * w + l));
                    a.jal(T6, bswap);
                    a.sd(A1, T5, 8 * l);
                }
            }
        }
        a.ret_via(RA);
    }
}

/// The native guest of a cut and what it is made of.
pub struct NativeGuest {
    pub elf: Vec<u8>,
    pub code_words: usize,
    pub data_bytes: usize,
    pub ops: usize,
    pub top_items: usize,
    pub loops: usize,
    pub max_depth: usize,
    /// Per op kind: ops and code words outside loops, then inside loop
    /// bodies (emitted once per loop), and loops refused for registers.
    pub report: String,
    /// Code addresses where each part starts, by what it does (an op kind,
    /// `loop` overhead, the runtime, prologue, epilogue), ascending.
    pub profile: Vec<(u64, &'static str)>,
}

/// The native guest of the hash-half program `words` (the interpreter's
/// encoding), `None` when they are not a program. Deterministic: the verifier
/// rebuilds the ELF from the cut.
pub fn build(words: &[u32]) -> Option<NativeGuest> {
    build_committing(words, &program_digest(words))
}

/// `keccak256` of the words, little-endian: what the guest output starts with.
pub fn program_digest(words: &[u32]) -> [u8; 32] {
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    Keccak256::digest(&bytes).into()
}

/// [`build`], the output starting with `digest` instead.
pub(crate) fn build_committing(words: &[u32], digest: &[u8; 32]) -> Option<NativeGuest> {
    let prog = decode(words)?;
    let mut roller = Roller::new(&prog);
    let (slot_of, n_slots) = if reuse_slots() {
        let log2 = std::env::var("NATIVE_VBIN_RING_LOG2")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(14);
        let (cells, slot_of, n) = reuse(&prog, log2);
        roller.op_cells = cells;
        (slot_of, n)
    } else {
        ((0..prog.n_cells).collect(), prog.n_cells)
    };
    let items = roller.roll();
    let op_cells = std::mem::take(&mut roller.op_cells);

    let mut data = digest.to_vec();
    let const_at = data.len() as u64;
    for (c, v) in &prog.consts {
        data.extend((slot_of[*c as usize] as u64).to_le_bytes());
        for x in v {
            data.extend(x.to_le_bytes());
        }
    }
    let rc_at = data.len() as u64;
    for c in KECCAK_RC {
        data.extend(c.to_le_bytes());
    }
    let back_at = data.len() as u64;
    for e in &prog.back {
        let cell = e & ((1 << 30) - 1);
        data.extend((e - cell + slot_of[cell as usize]).to_le_bytes());
    }
    let data_base = CODE_BASE;
    let code_base = (data_base + data.len() as u64).next_multiple_of(0x1000);
    let n_pub = prog.ops.iter().filter(|o| matches!(o, Op::Public)).count() as u64;
    let bind = bind_enabled();
    let bind_bytes = if bind { 80 } else { 0 };
    let mem = (PUB + 48 + 32 * n_pub + bind_bytes).next_multiple_of(PAGE) as i64;
    if mem as u64 + 32 * n_slots as u64 > 0xC000_0000 {
        return None;
    }

    let mut g = Gen {
        a: Asm::default(),
        prog: &prog,
        op_cells: &op_cells,
        rc_addr: (data_base + rc_at) as i64,
        keys: Vec::new(),
        key_ids: HashMap::new(),
        cls: Vec::new(),
        shift: Vec::new(),
        free: POOL.iter().rev().copied().collect(),
        tmp: 0,
        depth: 0,
        rt: Runtime::default(),
        loops: 0,
        max_depth: 0,
        spilled: 0,
        spill_next: 0,
        pend_hints: 0,
        pend_in: 0,
        marks: Vec::new(),
        census: Default::default(),
    };
    let fail = g.a.label();
    let main = g.a.label();
    g.a.jal(ZERO, main);
    g.runtime();
    g.mark("prologue");
    let a = &mut g.a;
    a.bind(main);
    a.li(PR, P as i64);
    a.li(A0, STATE as i64);
    a.li(A7, RPX_SYSCALL);
    a.li(ERR, 0);
    a.li(OK, 1);
    a.li(T3, INPUT as i64);
    a.lwu(T0, T3, 0);
    a.slli(T0, T0, 2);
    a.add(HINTS, T3, T0);
    a.addi(HINTS, HINTS, 16);
    a.lwu(T0, T3, 8);
    a.sltiu(T1, T0, 1);
    a.or(ERR, ERR, T1);
    a.add(INC, T3, T0);
    a.li(PUBC, (PUB + 48) as i64);
    if prog.n_out > 0 {
        a.li(T4, mem);
        a.li(T5, prog.n_out as i64);
        let top = a.here();
        for l in 0..3 {
            a.ld(T0, INC, 8 * l);
            a.sd(T0, T4, 8 * l);
        }
        a.addi(INC, INC, 24);
        a.addi(T4, T4, 32);
        a.addi(T5, T5, -1);
        a.bnez_back(T5, top);
    }
    if !prog.consts.is_empty() {
        a.li(T4, (data_base + const_at) as i64);
        a.li(T5, prog.consts.len() as i64);
        a.li(T6, mem);
        let top = a.here();
        a.ld(T0, T4, 0);
        a.slli(T0, T0, 5);
        a.add(T0, T0, T6);
        for l in 0..4 {
            a.ld(T1, T4, 8 + 8 * l);
            a.sd(T1, T0, 8 * l);
        }
        a.addi(T4, T4, 40);
        a.addi(T5, T5, -1);
        a.bnez_back(T5, top);
    }
    let fixed = g.class_id(Vec::new());
    g.a.li(FP, mem + SLIDE);
    g.cls[fixed] = Some(Class {
        loc: Loc::Reg(FP),
        off: SLIDE,
    });
    let n_opnds: usize = items
        .iter()
        .map(|it| match it {
            Item::Op(k) => op_cells[*k as usize].len(),
            Item::Loop(l) => l.cells.len(),
        })
        .sum();
    g.items(&items, &vec![fixed; n_opnds]);
    g.flush_cursors();
    g.mark("epilogue");

    let a = &mut g.a;
    if !prog.back.is_empty() {
        a.li(T3, INPUT as i64);
        a.lwu(T0, T3, 12);
        a.add(T3, T3, T0);
        a.li(T4, (data_base + back_at) as i64);
        a.li(T5, prog.back.len() as i64);
        a.li(A5, mem);
        let top = a.here();
        let store = a.label();
        a.lwu(T0, T4, 0);
        a.srli(T1, T0, 30);
        a.slli(T0, T0, 34);
        a.srli(T0, T0, 29);
        a.add(T0, T0, A5);
        for (l, r) in [A1, A2, A3, A4].into_iter().enumerate() {
            a.ld(r, T0, 8 * l as i64);
        }
        a.beq(T1, ZERO, store);
        a.or(ERR, ERR, A4);
        a.addi(T2, T1, -1);
        a.beq(T2, ZERO, store);
        a.or(ERR, ERR, A2);
        a.or(ERR, ERR, A3);
        a.addi(T2, T1, -2);
        a.beq(T2, ZERO, store);
        a.ori(ERR, ERR, 1);
        a.bind(store);
        for (l, r) in [A1, A2, A3].into_iter().enumerate() {
            a.sd(r, T3, 8 * l as i64);
        }
        a.addi(T3, T3, 24);
        a.addi(T4, T4, 4);
        a.addi(T5, T5, -1);
        a.bnez_back(T5, top);
    }
    if bind {
        assert!(
            !prog.ops.iter().any(|o| matches!(o, Op::Keccak { .. })),
            "the binding's scratch is the keccak state's"
        );
        let n_in = crossing_counts_of(&prog).0;
        emit_binding(a, n_in, prog.back.len());
        // (c_bin, α, s) after the published words.
        a.li(T3, KSTATE as i64);
        for (i, off) in [0, 8, 16, 24, 88, 96, 104, 64, 72, 80]
            .into_iter()
            .enumerate()
        {
            a.ld(T0, T3, off);
            a.sd(T0, PUBC, 8 * i as i64);
        }
        a.addi(PUBC, PUBC, 80);
    }
    a.bne(ERR, ZERO, fail);
    a.beq(OK, ZERO, fail);
    a.li(T3, PUB as i64);
    a.li(T4, data_base as i64);
    for l in 0..4 {
        a.ld(T0, T4, 8 * l);
        a.sd(T0, T3, 8 * l);
    }
    a.li(T5, INPUT as i64);
    a.lwu(T0, T5, 8);
    a.add(T0, T0, T5);
    a.sw(T0, T3, 32);
    a.lwu(T1, T5, 12);
    a.add(T1, T1, T5);
    a.sw(T1, T3, 36);
    a.sub(T2, INC, T0);
    a.li(T6, 24);
    a.divu(T2, T2, T6);
    a.sw(T2, T3, 40);
    a.li(T2, prog.back.len() as i64);
    a.sw(T2, T3, 44);
    a.li(A0, 1);
    a.mv(A1, T3);
    a.sub(A2, PUBC, T3);
    a.li(A7, 64);
    a.ecall();
    a.li(A0, 0);
    a.li(A7, 93);
    a.ecall();
    a.bind(fail);
    a.li(A0, PUB as i64);
    a.li(A1, 0);
    a.li(A7, 2);
    a.ecall();

    let (loops, max_depth, top_items) = (g.loops, g.max_depth, items.len());
    let mut report = format!(
        "loops refused {}, pointers spilled {}, {} slots for {} cells",
        roller.refused, g.spilled, n_slots, prog.n_cells
    );
    let mut shapes: std::collections::BTreeMap<String, usize> = Default::default();
    for (k, op) in prog.ops.iter().enumerate() {
        if matches!(
            op,
            Op::Pack
                | Op::Select
                | Op::Hash { .. }
                | Op::Hint { .. }
                | Op::Unpack { .. }
                | Op::Walk4
        ) {
            let key = format!("{op:?} facts {:?} fwd {:?}", prog.facts[k], prog.fwd[k]);
            *shapes.entry(key).or_insert(0) += 1;
        }
    }
    let mut shapes: Vec<_> = shapes.into_iter().collect();
    shapes.sort_by(|a, b| b.1.cmp(&a.1));
    for (k, n) in shapes.iter().take(25) {
        report.push_str(&format!("\n    {n} x {k}"));
    }
    for (k, [top, inner]) in &g.census {
        report.push_str(&format!(
            "; {k}: top {} ops {} words, loops {} ops {} words",
            top.0, top.1, inner.0, inner.1
        ));
    }
    let profile = g
        .marks
        .iter()
        .map(|&(pc, what)| (code_base + 4 * pc as u64, what))
        .collect();
    let code = g.a.finish();
    assert!(
        code_base + 4 * code.len() as u64 <= SCRATCH,
        "the guest code fits below its scratch"
    );
    Some(NativeGuest {
        elf: elf(code_base, &code, data_base, &data),
        code_words: code.len(),
        data_bytes: data.len(),
        ops: prog.ops.len() + prog.consts.len(),
        top_items,
        loops,
        max_depth,
        report,
        profile,
    })
}

/// How many cells the program `words` reads from IN (its `out` cells, then
/// the input it checks) and writes to BACK.
pub fn crossing_counts(words: &[u32]) -> Option<(usize, usize)> {
    Some(crossing_counts_of(&decode(words)?))
}

/// IN's and BACK's cell counts of a decoded program.
fn crossing_counts_of(prog: &Program) -> (usize, usize) {
    let pushes: usize = prog
        .ops
        .iter()
        .map(|op| match *op {
            Op::Hint { flags } => (flags & 1) as usize,
            Op::Unpack { mask } => (mask & 0xf).count_ones() as usize,
            _ => 0,
        })
        .sum();
    (prog.n_out as usize + pushes, prog.back.len())
}

/// `NATIVE_VBIN_BIND`: the guest binds its crossing cells for a two-proof
/// split (`field_vm::cksum`).
pub fn bind_enabled() -> bool {
    std::env::var_os("NATIVE_VBIN_BIND").is_some()
}

/// The guest's side of the binding, at the end of the run, BACK written:
/// `c_bin` (`cksum::crossing_digest` over IN then BACK, through the Poseidon1
/// ecall), then `s = Σ x_i·α^{n−1−i}` over the same cells through the Fp³
/// multiply-add ecall, `α` read right after BACK's cells. Leaves `c_bin` at
/// `KSTATE`, `[s, α]` at `KSTATE + 64`; restores `a0` and `a7`.
const MAC_UNROLL: usize = 8;

fn emit_binding(a: &mut Asm, n_in: usize, n_back: usize) {
    let bind = KSTATE as i64;
    for l in 0..16 {
        a.sd(ZERO, A0, 8 * l);
    }
    a.li(T0, n_in as i64);
    a.sd(T0, A0, 96);
    a.li(T0, n_back as i64);
    a.sd(T0, A0, 104);
    let regions = [(8, n_in), (12, n_back)];
    for (off, n) in regions {
        if n == 0 {
            continue;
        }
        a.li(T3, INPUT as i64);
        a.lwu(T4, T3, off);
        a.add(T4, T3, T4);
        let (full, rem) = (n / 4, n % 4);
        if full > 0 {
            a.li(T5, full as i64);
            let top = a.here();
            for w in 0..12 {
                a.ld(T0, T4, 8 * w);
                a.sd(T0, A0, 8 * w);
            }
            a.ecall();
            a.addi(T4, T4, 96);
            a.addi(T5, T5, -1);
            a.bnez_back(T5, top);
        }
        if rem > 0 {
            for w in 0..12 {
                if w < 3 * rem as i64 {
                    a.ld(T0, T4, 8 * w);
                    a.sd(T0, A0, 8 * w);
                } else {
                    a.sd(ZERO, A0, 8 * w);
                }
            }
            a.ecall();
        }
    }
    a.li(T3, bind);
    for l in 0..4 {
        a.ld(T0, A0, 8 * l);
        a.sd(T0, T3, 8 * l);
    }
    // [acc, α] = [0, α], α after BACK's cells.
    for l in 0..3 {
        a.sd(ZERO, T3, 64 + 8 * l);
    }
    a.li(T4, INPUT as i64);
    a.lwu(T5, T4, 12);
    a.add(T5, T4, T5);
    a.li(T6, 24 * n_back as i64);
    a.add(T5, T5, T6);
    for l in 0..3 {
        a.ld(T0, T5, 8 * l);
        a.sd(T0, T3, 88 + 8 * l);
    }
    a.addi(A0, T3, 64);
    a.li(
        A7,
        executor::vm::instruction::execution::FP3MAC_SYSCALL_NUMBER as i64,
    );
    for (off, n) in regions {
        if n == 0 {
            continue;
        }
        a.li(T4, INPUT as i64);
        a.lwu(T5, T4, off);
        a.add(A1, T4, T5);
        let (trips, rem) = (n / MAC_UNROLL, n % MAC_UNROLL);
        if trips > 0 {
            a.li(T6, trips as i64);
            let top = a.here();
            for _ in 0..MAC_UNROLL {
                a.ecall();
                a.addi(A1, A1, 24);
            }
            a.addi(T6, T6, -1);
            a.bnez_back(T6, top);
        }
        for _ in 0..rem {
            a.ecall();
            a.addi(A1, A1, 24);
        }
    }
    a.li(A0, STATE as i64);
    a.li(A7, RPX_SYSCALL);
}

/// The field routines of the native guest, every base and extension op:
/// link `RA`, `A1 = &a`, `A2 = &b`, `A3 = &c`, `A4 = &out`; `ERR` in `SP`, `p`
/// in `GP`. Clobbers `T0..T6`, `A1..A5`.
pub(crate) struct FieldRuntime {
    pub(crate) base: [Label; 5],
    pub(crate) ext: [Label; 6],
    pub(crate) bswap: Label,
}

pub(crate) fn field_runtime(a: Asm) -> (Asm, FieldRuntime) {
    let mut ops: Vec<Op> = (0..5).map(|op| Op::BaseAlu { op }).collect();
    ops.extend((0..6).map(|op| Op::ExtAlu { op }));
    let prog = Program {
        n_cells: 0,
        n_out: 0,
        back: Vec::new(),
        consts: Vec::new(),
        ops,
        cells: Vec::new(),
        facts: Vec::new(),
        reads: Vec::new(),
        fwd: Vec::new(),
        in_alias: HashMap::new(),
        hint_alias: HashMap::new(),
        rpx_only: Vec::new(),
    };
    let mut g = Gen {
        a,
        prog: &prog,
        op_cells: &[],
        rc_addr: 0,
        keys: Vec::new(),
        key_ids: HashMap::new(),
        cls: Vec::new(),
        shift: Vec::new(),
        free: Vec::new(),
        tmp: 0,
        depth: 0,
        rt: Runtime::default(),
        loops: 0,
        max_depth: 0,
        spilled: 0,
        spill_next: 0,
        pend_hints: 0,
        pend_in: 0,
        marks: Vec::new(),
        census: Default::default(),
    };
    g.runtime();
    let rt = FieldRuntime {
        base: g.rt.base.map(|l| l.expect("every base routine")),
        ext: g.rt.ext.map(|l| l.expect("every extension routine")),
        bswap: g.rt.bswap.expect("the byte swap"),
    };
    (g.a, rt)
}
