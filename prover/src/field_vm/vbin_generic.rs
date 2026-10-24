//! The generic native v_bin: one guest for every leaf. The leaf's hash-half
//! program (the cut's words) is compiled to a schedule the guest reads as
//! data, hashes (its digest is the first word the guest publishes, where the
//! per-leaf guest publishes `keccak256` of the words) and runs. Every value
//! keeps the per-leaf guest's semantics: the same BACK cells in the same
//! order, the same published words, the same binding.
//!
//! Memory: hints are read where the input holds them (a word hint in the
//! stream, a shared hint in its IN cell), constants in the schedule, BACK
//! cells in the BACK region; every other cell in a slot of `TMP`. Cells are
//! four lanes; IN and BACK cells too, lane 3 zero for IN.

use std::collections::HashMap;
use std::sync::OnceLock;

use super::native_vbin::{self, L3_ZERO, Op, Program, UPPER_ZERO, ZERO_CONST};
use super::rv64::*;

const VERSION: u32 = 1;
const P: u64 = 0xFFFF_FFFF_0000_0001;
const INPUT: u64 = 0xFF00_0004;
const CODE_BASE: u64 = 0x1_0000;
const SCRATCH: u64 = 0x1000_0000;
const STATE: u64 = SCRATCH;
const BIND: u64 = SCRATCH + 0x100;
const PUB: u64 = SCRATCH + 0x800;
const CTL: u64 = SCRATCH + 0x1_0000;
const FRAMES: u64 = 0x2000_0000;
const TMP: u64 = 0xE000_0000;
/// Addresses in the schedule are offsets from here, so that they fit a word
/// (the input ends past 2^32).
const OPND_BASE: u64 = 0xC000_0000;
const BIAS: u64 = 1 << 31;
const RING_LOG2: u32 = 14;
const P1W16_SYSCALL: i64 = -6;

pub(crate) const OP_HALT: u32 = 0;
pub(crate) const OP_END: u32 = 1;
pub(crate) const OP_LOOP: u32 = 2;
pub(crate) const OP_HASH: u32 = 3;
pub(crate) const OP_WALK4: u32 = 4;
pub(crate) const OP_SELECT: u32 = 5;
pub(crate) const OP_PACK: u32 = 6;
pub(crate) const OP_UNPACK: u32 = 7;
pub(crate) const OP_BITDEC: u32 = 8;
pub(crate) const OP_BASE: u32 = 9;
pub(crate) const OP_EXT: u32 = 10;
pub(crate) const OP_PUBLIC: u32 = 11;
pub(crate) const OP_MUX: u32 = 12;
pub(crate) const OP_ASSERT: u32 = 13;
pub(crate) const OP_CHAIN: u32 = 14;
pub(crate) const OP_PATH: u32 = 15;
pub(crate) const OP_ASSERT4: u32 = 16;
const N_OPS: usize = 17;

/// Where the regions of a leaf's input land, absolute.
#[derive(Clone, Copy, Debug)]
pub struct GenericLayout {
    pub stream_base: u64,
    pub in_base: u64,
    pub back_base: u64,
    pub alpha: u64,
    pub sched_base: u64,
}

/// A leaf's compiled schedule and the shape of its input.
pub struct Schedule {
    pub words: Vec<u32>,
    pub layout: GenericLayout,
    pub n_in: usize,
    pub n_back: usize,
    /// Per hint, the lanes its stream entry holds.
    pub stream_lanes: Vec<&'static [usize]>,
    pub ops: usize,
    pub items: usize,
    pub loops: usize,
    pub tmp_slots: usize,
}

/// One compiled op: its op word and operand addresses.
#[derive(Clone)]
struct GOp {
    word: u32,
    opnds: Vec<i64>,
}

fn op_word(code: u32, flags: u32) -> u32 {
    code | flags << 8
}

fn rel(a: u64) -> u32 {
    let r = a - OPND_BASE;
    assert!(r < 1 << 32, "an address the schedule can name");
    r as u32
}

/// Input offsets relative to `INPUT`: the header, the hint stream, IN
/// (page-aligned), BACK, `α`, then the schedule.
fn layout_for(stream_bytes: usize, n_in: usize, n_back: usize) -> GenericLayout {
    let stream_base = INPUT + 12;
    let page = crate::tables::page::DEFAULT_PAGE_SIZE as u64;
    let in_base = (stream_base + stream_bytes as u64).next_multiple_of(page);
    let back_base = in_base + 32 * n_in as u64;
    let alpha = back_base + 32 * n_back as u64;
    let sched_base = (alpha + 24).next_multiple_of(8);
    GenericLayout {
        stream_base,
        in_base,
        back_base,
        alpha,
        sched_base,
    }
}

/// The lanes of a hint's stream entry, as the per-leaf guest packs them.
fn stream_lanes(flags: u32) -> &'static [usize] {
    match (flags & 1 != 0, flags >> 1) {
        (true, 0) => &[3],
        (true, 1 | 2) => &[],
        (false, 1) => &[0, 1, 2],
        (false, 2) => &[0],
        _ => &[0, 1, 2, 3],
    }
}

/// One op of the lowered program, over cells.
#[derive(Clone, Debug)]
enum LOp {
    Hash {
        ins: [HIn; 4],
        outs: [u32; 4],
    },
    Walk4 {
        ins: [u32; 6],
        outs: [u32; 4],
        checks: u32,
    },
    Select {
        bit: u32,
        l: u32,
        r: u32,
        outs: [u32; 2],
        check: bool,
    },
    Mux {
        bits: Vec<u32>,
        leaves: Vec<u32>,
        out: u32,
        checks: u32,
        bsrc: Option<(u32, u32)>,
    },
    Path {
        cur: u32,
        sibs: Vec<u32>,
        bits: Vec<u32>,
        outs: [u32; 4],
        checks: u32,
        bsrc: Option<(u32, u32)>,
    },
    Pack {
        ins: [u32; 4],
        out: u32,
        checks: u32,
    },
    Unpack {
        src: u32,
        outs: [u32; 4],
    },
    BitDec {
        src: u32,
        n: u32,
        halves: bool,
        outs: Vec<u32>,
        check: bool,
    },
    Base {
        op: u32,
        ins: Vec<u32>,
        out: u32,
        checks: u32,
    },
    Ext {
        op: u32,
        ins: Vec<u32>,
        out: u32,
        checks: u32,
    },
    Assert {
        x: u32,
        y: u32,
        checks: u32,
    },
    Public {
        src: u32,
    },
}

/// A permutation input: a cell, or four lanes packed (a pack fused in), each
/// lane with its upper-zero check.
#[derive(Clone, Copy, Debug)]
enum HIn {
    Cell(u32),
    Pack([u32; 4], u32),
}

const NONE: u32 = u32::MAX;

impl LOp {
    /// The cells it reads.
    fn reads(&self, out: &mut Vec<u32>) {
        match self {
            LOp::Hash { ins, .. } => {
                for i in ins {
                    match i {
                        HIn::Cell(c) => out.push(*c),
                        HIn::Pack(l, _) => out.extend(l),
                    }
                }
            }
            LOp::Walk4 { ins, .. } => out.extend(ins),
            LOp::Select { bit, l, r, .. } => out.extend([*bit, *l, *r]),
            LOp::Mux {
                bits, leaves, bsrc, ..
            } => {
                match bsrc {
                    Some((src, _)) => out.push(*src),
                    None => out.extend(bits),
                }
                out.extend(leaves);
            }
            LOp::Path {
                cur,
                sibs,
                bits,
                bsrc,
                ..
            } => {
                out.push(*cur);
                out.extend(sibs);
                match bsrc {
                    Some((src, _)) => out.push(*src),
                    None => out.extend(bits),
                }
            }
            LOp::Pack { ins, .. } => out.extend(ins),
            LOp::Unpack { src, .. } => out.push(*src),
            LOp::BitDec { src, .. } => out.push(*src),
            LOp::Base { ins, .. } | LOp::Ext { ins, .. } => out.extend(ins),
            LOp::Assert { x, y, .. } => out.extend([*x, *y]),
            LOp::Public { src } => out.push(*src),
        }
    }

    /// The cells it writes (`NONE` for a discarded output).
    fn writes(&self, out: &mut Vec<u32>) {
        match self {
            LOp::Hash { outs, .. }
            | LOp::Walk4 { outs, .. }
            | LOp::Unpack { outs, .. }
            | LOp::Path { outs, .. } => out.extend(outs),
            LOp::Select { outs, .. } => out.extend(outs),
            LOp::Mux { out: o, .. } | LOp::Pack { out: o, .. } => out.push(*o),
            LOp::BitDec { outs, .. } => out.extend(outs),
            LOp::Base { out: o, .. } | LOp::Ext { out: o, .. } => out.push(*o),
            LOp::Assert { .. } | LOp::Public { .. } => {}
        }
    }
}

/// Compiles the hash-half program `words` (the cut's) for the generic guest.
pub fn compile(words: &[u32]) -> Result<Schedule, String> {
    let prog = native_vbin::decode_generic(words).ok_or("the words are not a program")?;
    let n = prog.n_cells as usize;
    let nops = prog.ops.len();
    let back_of: HashMap<u32, usize> = prog
        .back
        .iter()
        .enumerate()
        .map(|(i, e)| (e & ((1 << 30) - 1), i))
        .collect();
    let is_back = |c: u32| back_of.contains_key(&c);

    // The stream and IN, in program order.
    let mut stream_bytes = 0usize;
    let mut n_in = prog.n_out as usize;
    let mut lanes_of = Vec::new();
    for op in &prog.ops {
        match *op {
            Op::Hint { flags } => {
                let l = stream_lanes(flags);
                stream_bytes += 8 * l.len();
                lanes_of.push(l);
                n_in += (flags & 1) as usize;
            }
            Op::Unpack { mask } => n_in += (mask & 0xf).count_ones() as usize,
            Op::Keccak { .. } => return Err("keccak is not supported".into()),
            _ => {}
        }
    }
    let n_back = prog.back.len();
    let layout = layout_for(stream_bytes, n_in, n_back);

    // Fixed homes: OUT and shared hints in IN, word hints in the stream,
    // BACK in BACK, constants in the schedule (below).
    let mut home: Vec<Option<u64>> = vec![None; n];
    for (c, h) in home.iter_mut().enumerate().take(prog.n_out as usize) {
        *h = Some(layout.in_base + 32 * c as u64);
    }
    for (&c, &i) in &back_of {
        home[c as usize] = Some(layout.back_base + 32 * i as u64);
    }
    let mut hint_lanes: Vec<(u32, u64, &'static [usize], bool, usize)> = Vec::new();
    let mut base_checks: Vec<u64> = Vec::new();
    let (mut s_at, mut in_at) = (layout.stream_base, prog.n_out as u64);
    for (k, op) in prog.ops.iter().enumerate() {
        match *op {
            Op::Hint { flags } => {
                let c = prog.cells[k][0] as usize;
                let shared = flags & 1 != 0;
                let ty = flags >> 1;
                let free = match ty {
                    1 => 3,
                    2 => 1,
                    _ => 4,
                };
                let lanes = stream_lanes(flags);
                hint_lanes.push((c as u32, s_at, lanes, shared, free));
                if is_back(c as u32) {
                    return Err("a hint crosses back".into());
                }
                if shared {
                    let at = layout.in_base + 32 * in_at;
                    if ty == 0 {
                        return Err("a shared word hint needs a copy (not supported)".into());
                    }
                    home[c] = Some(at);
                    if ty == 2 {
                        base_checks.push(at);
                    }
                    in_at += 1;
                } else if ty == 0 {
                    home[c] = Some(s_at);
                } else {
                    return Err("a typed word hint needs a copy (not supported)".into());
                }
                s_at += 8 * lanes.len() as u64;
            }
            Op::Unpack { mask } => {
                if mask & 0xf != 0 {
                    return Err("unpacks into IN are not supported".into());
                }
            }
            _ => {}
        }
    }

    // A BACK cell the guest writes with its zero lanes itself needs no
    // type check.
    let mut writer_op = vec![usize::MAX; n];
    for k in 0..nops {
        for &c in &prog.cells[k][prog.reads[k]..] {
            writer_op[c as usize] = k;
        }
    }
    let mut type_runs: Vec<(u32, u32)> = Vec::new();
    for e in &prog.back {
        let c = e & ((1 << 30) - 1);
        let t = match prog.ops.get(writer_op[c as usize]) {
            Some(Op::BitDec { .. } | Op::Unpack { .. } | Op::BaseAlu { .. }) => 0,
            Some(Op::ExtAlu { .. }) if e >> 30 == 1 => 0,
            _ => e >> 30,
        };
        match type_runs.last_mut() {
            Some((u, m)) if *u == t => *m += 1,
            _ => type_runs.push((t, 1)),
        }
    }
    // The header, then the constants (their cells), then (once the program
    // is lowered) the hint lanes to check, the shared base hints, the types.
    let mut head = vec![
        VERSION,
        0,
        n_in as u32,
        n_back as u32,
        rel(layout.in_base),
        rel(layout.back_base),
        0,
        base_checks.len() as u32,
        2 * type_runs.len() as u32,
        prog.consts.len() as u32,
    ];
    let const_base = layout.sched_base + 4 * head.len() as u64;
    for (i, (c, v)) in prog.consts.iter().enumerate() {
        home[*c as usize] = Some(const_base + 32 * i as u64);
        for x in v {
            head.extend([*x as u32, (*x >> 32) as u32]);
        }
    }

    // Readers of each cell in the decoded program.
    let mut readers: Vec<Vec<(usize, usize)>> = vec![Vec::new(); n];
    for k in 0..nops {
        if matches!(prog.ops[k], Op::Hint { .. }) {
            continue;
        }
        for (j, &c) in prog.cells[k][..prog.reads[k]].iter().enumerate() {
            readers[c as usize].push((k, j));
        }
    }
    let out_of = |k: usize, j: usize| prog.cells[k][prog.reads[k] + j];
    let unread = |c: u32| !is_back(c) && readers[c as usize].is_empty();
    let fuse = std::env::var_os("VBG_NO_FUSE").is_none();
    let mut dead = vec![false; nops];

    // Asserts: `x − y` read only by a division by the zero constant whose
    // result no one reads.
    let mut assert_of = vec![false; nops];
    if fuse {
        for (k, is_assert) in assert_of.iter_mut().enumerate() {
            if !matches!(prog.ops[k], Op::BaseAlu { op: 1 }) {
                continue;
            }
            let d = out_of(k, 0);
            if is_back(d) {
                continue;
            }
            if let [(m, 0)] = readers[d as usize][..]
                && matches!(prog.ops[m], Op::BaseAlu { op: 3 })
                && prog.facts[m][1] & ZERO_CONST != 0
                && prog.facts[m][0] & UPPER_ZERO != 0
                && unread(out_of(m, 0))
            {
                *is_assert = true;
                dead[m] = true;
            }
        }
    }
    // Lanes of an unpack read in place by readers of lane 0 only.
    let lane0_reader = |k: usize, j: usize| -> bool {
        !dead[k]
            && match prog.ops[k] {
                Op::Pack => true,
                Op::BitDec { .. } => j == 0,
                Op::BaseAlu { .. } => true,
                _ => false,
            }
    };
    let mut alias: HashMap<u32, (u32, u64)> = HashMap::new();
    if fuse {
        for k in 0..nops {
            if !matches!(prog.ops[k], Op::Unpack { .. }) {
                continue;
            }
            let src = prog.cells[k][0];
            for l in 0..4 {
                let o = out_of(k, l);
                if is_back(o) || readers[o as usize].is_empty() {
                    continue;
                }
                if readers[o as usize].iter().all(|&(r, j)| lane0_reader(r, j)) {
                    alias.insert(o, (src, 8 * l as u64));
                }
            }
        }
    }
    // Packs fused into the permutation that reads them: read once, by the
    // next permutation, with none in between.
    let mut merged: HashMap<(usize, usize), usize> = HashMap::new();
    let mut is_merged = vec![false; nops];
    if fuse {
        let mut next_perm = vec![usize::MAX; nops + 1];
        for k in (0..nops).rev() {
            next_perm[k] = if matches!(prog.ops[k], Op::Hash { .. } | Op::Walk4) {
                k
            } else {
                next_perm[k + 1]
            };
        }
        for k in 0..nops {
            if !matches!(prog.ops[k], Op::Pack) {
                continue;
            }
            let p = out_of(k, 0);
            if is_back(p) {
                continue;
            }
            if let [(h, j)] = readers[p as usize][..]
                && matches!(prog.ops[h], Op::Hash { mode: 4 })
                && next_perm[k + 1] == h
            {
                merged.insert((h, j), k);
                is_merged[k] = true;
            }
        }
    }
    // Select trees whose every inner output has one reader, its parent:
    // a mux of the leaves by the levels' bits.
    let mut writer = vec![usize::MAX; n];
    for k in 0..nops {
        for &c in &prog.cells[k][prog.reads[k]..] {
            writer[c as usize] = k;
        }
    }
    // (bits by level, leaves) of the tree rooted at select `k`.
    fn tree(
        k: usize,
        prog: &Program,
        writer: &[usize],
        readers: &[Vec<(usize, usize)>],
        is_back: &dyn Fn(u32) -> bool,
        depth: usize,
    ) -> (Vec<u32>, Vec<u32>, Vec<usize>) {
        let c = &prog.cells[k];
        let (bit, l, r) = (c[0], c[1], c[2]);
        let child = |x: u32| -> Option<usize> {
            let w = writer[x as usize];
            if w == usize::MAX || !matches!(prog.ops[w], Op::Select) {
                return None;
            }
            let wc = &prog.cells[w];
            (wc[3] == x
                && !is_back(x)
                && readers[x as usize].len() == 1
                && !is_back(wc[4])
                && readers[wc[4] as usize].is_empty())
            .then_some(w)
        };
        if depth < 7
            && let (Some(a), Some(b)) = (child(l), child(r))
        {
            let (ba, la, da) = tree(a, prog, writer, readers, is_back, depth + 1);
            let (bb, lb, db) = tree(b, prog, writer, readers, is_back, depth + 1);
            if ba == bb && !ba.is_empty() {
                let mut bits = ba;
                bits.push(bit);
                let mut leaves = la;
                leaves.extend(lb);
                let mut del = da;
                del.extend(db);
                del.extend([a, b]);
                return (bits, leaves, del);
            }
        }
        (vec![bit], vec![l, r], Vec::new())
    }
    let mut mux_at: HashMap<usize, (Vec<u32>, Vec<u32>)> = HashMap::new();
    if fuse {
        for k in (0..nops).rev() {
            if dead[k] || !matches!(prog.ops[k], Op::Select) {
                continue;
            }
            if !unread(prog.cells[k][4]) {
                continue;
            }
            let (bits, leaves, del) = tree(k, &prog, &writer, &readers, &is_back, 0);
            if bits.len() < 2 {
                continue;
            }
            for d in del {
                dead[d] = true;
            }
            mux_at.insert(k, (bits, leaves));
        }
    }

    // The lowered program, in order.
    let check = |k: usize, j: usize| (prog.facts[k][j] & UPPER_ZERO == 0) as u32;
    let mut lops: Vec<LOp> = Vec::new();
    for k in 0..nops {
        if dead[k] || is_merged[k] || matches!(prog.ops[k], Op::Hint { .. }) {
            continue;
        }
        let cells = &prog.cells[k];
        let nr = prog.reads[k];
        let lop = match prog.ops[k] {
            Op::Hash { mode: 4 } => {
                let ins = core::array::from_fn(|j| match merged.get(&(k, j)) {
                    Some(&p) => {
                        let pc = &prog.cells[p];
                        HIn::Pack(
                            [pc[0], pc[1], pc[2], pc[3]],
                            (0..4).map(|i| check(p, i) << i).sum(),
                        )
                    }
                    None => HIn::Cell(cells[j]),
                });
                LOp::Hash {
                    ins,
                    outs: core::array::from_fn(|j| cells[4 + j]),
                }
            }
            Op::Hash { .. } => return Err("only width-16 permutations are supported".into()),
            Op::Walk4 => LOp::Walk4 {
                ins: core::array::from_fn(|j| cells[j]),
                outs: core::array::from_fn(|j| cells[6 + j]),
                checks: check(k, 0) | check(k, 1) << 1,
            },
            Op::Select => match mux_at.remove(&k) {
                Some((bits, leaves)) => {
                    // Every select of a level reads the same bit, with the
                    // same facts as the root's chain.
                    let mut checks = 0;
                    for (lvl, &b) in bits.iter().enumerate() {
                        let w = readers[b as usize]
                            .iter()
                            .find(|&&(r, j)| j == 0 && matches!(prog.ops[r], Op::Select))
                            .map(|&(r, _)| r)
                            .unwrap_or(k);
                        checks |= check(w, 0) << lvl;
                    }
                    LOp::Mux {
                        bits,
                        leaves,
                        out: cells[3],
                        checks,
                        bsrc: None,
                    }
                }
                None => LOp::Select {
                    bit: cells[0],
                    l: cells[1],
                    r: cells[2],
                    outs: [cells[3], cells[4]],
                    check: check(k, 0) != 0,
                },
            },
            Op::Pack => LOp::Pack {
                ins: core::array::from_fn(|j| cells[j]),
                out: cells[4],
                checks: (0..4).map(|j| check(k, j) << j).sum(),
            },
            Op::Unpack { .. } => LOp::Unpack {
                src: cells[0],
                outs: core::array::from_fn(|j| cells[1 + j]),
            },
            Op::BitDec { n: nb, halves } => LOp::BitDec {
                src: cells[0],
                n: nb,
                halves,
                outs: cells[1..].to_vec(),
                check: check(k, 0) != 0,
            },
            Op::BaseAlu { op: 1 } if assert_of[k] => LOp::Assert {
                x: cells[0],
                y: cells[1],
                checks: check(k, 0) | check(k, 1) << 1,
            },
            Op::BaseAlu { op } => LOp::Base {
                op,
                ins: cells[..nr].to_vec(),
                out: cells[nr],
                checks: (0..nr).map(|j| check(k, j) << j).sum(),
            },
            Op::ExtAlu { op } => LOp::Ext {
                op,
                ins: cells[..nr].to_vec(),
                out: cells[nr],
                checks: (0..nr)
                    .map(|j| {
                        let c = if j == 1 && op == 5 {
                            prog.facts[k][j] & UPPER_ZERO == 0
                        } else {
                            prog.facts[k][j] & L3_ZERO == 0
                        };
                        (c as u32) << j
                    })
                    .sum(),
            },
            Op::Public => LOp::Public { src: cells[0] },
            Op::Hint { .. } | Op::Keccak { .. } => unreachable!("skipped or refused"),
        };
        lops.push(lop);
    }

    // Paths and bit sources.
    if fuse && std::env::var_os("VBG_NO_MACROS").is_none() {
        let mut reads0 = vec![0u32; n];
        let mut b = Vec::new();
        for op in &lops {
            b.clear();
            op.reads(&mut b);
            for &c in &b {
                reads0[c as usize] += 1;
            }
        }
        let once = |c: u32| !is_back(c) && reads0[c as usize] == 1;
        let none = |c: u32| !is_back(c) && reads0[c as usize] == 0;
        let at = |c: u32| home[c as usize];
        let sib_at = |ins: &[u32; 6]| -> Option<u64> {
            let s = at(ins[3])?;
            (at(ins[4])? == s + 32 && at(ins[5])? == s + 64).then_some(s)
        };
        let mut out: Vec<LOp> = Vec::with_capacity(lops.len());
        let mut i = 0;
        while i < lops.len() {
            if let LOp::Walk4 { ins, outs, checks } = &lops[i]
                && let Some(s0) = sib_at(ins)
            {
                let mut last = i;
                while let Some(LOp::Walk4 {
                    ins: ni,
                    checks: nc,
                    ..
                }) = lops.get(last + 1)
                {
                    let LOp::Walk4 { outs: po, .. } = &lops[last] else {
                        unreachable!()
                    };
                    if !(ni[2] == po[0]
                        && once(po[0])
                        && po[1..].iter().all(|&o| none(o))
                        && nc == checks
                        && sib_at(ni) == Some(s0 + 96 * (last + 1 - i) as u64))
                    {
                        break;
                    }
                    last += 1;
                }
                if last > i {
                    let mut sibs = Vec::new();
                    let mut bits = Vec::new();
                    for w in &lops[i..=last] {
                        let LOp::Walk4 { ins: wi, .. } = w else {
                            unreachable!()
                        };
                        sibs.extend(&wi[3..6]);
                        bits.extend(&wi[0..2]);
                    }
                    let LOp::Walk4 { outs: lo, .. } = &lops[last] else {
                        unreachable!()
                    };
                    let _ = outs;
                    out.push(LOp::Path {
                        cur: ins[2],
                        sibs,
                        bits,
                        outs: *lo,
                        checks: *checks,
                        bsrc: None,
                    });
                    i = last + 1;
                    continue;
                }
            }
            out.push(lops[i].clone());
            i += 1;
        }
        lops = out;
        // Bits that are consecutive bits of one decomposition: read from
        // its input instead.
        let mut bitpos: HashMap<u32, (u32, u32)> = HashMap::new();
        for k in 0..nops {
            if let Op::BitDec { n: nb, .. } = prog.ops[k] {
                for j in 0..nb {
                    bitpos.insert(prog.cells[k][1 + j as usize], (prog.cells[k][0], j));
                }
            }
        }
        let consecutive = |bits: &[u32]| -> Option<(u32, u32)> {
            let (src, j0) = *bitpos.get(bits.first()?)?;
            bits.iter()
                .enumerate()
                .all(|(t, b)| bitpos.get(b) == Some(&(src, j0 + t as u32)))
                .then_some((src, j0))
        };
        for op in lops.iter_mut() {
            match op {
                LOp::Path {
                    bits,
                    checks: 0,
                    bsrc,
                    ..
                }
                | LOp::Mux {
                    bits,
                    checks: 0,
                    bsrc,
                    ..
                } => {
                    *bsrc = consecutive(bits);
                }
                _ => {}
            }
        }
    }

    // Liveness over the lowered program: a read of an aliased lane is a read
    // of its source.
    let root = |c: u32| alias.get(&c).map_or(c, |&(s, _)| s);
    let mut write = vec![usize::MAX; n];
    let mut last = vec![0usize; n];
    let mut live_read = vec![false; n];
    let mut buf = Vec::new();
    for (i, op) in lops.iter().enumerate() {
        buf.clear();
        op.reads(&mut buf);
        for &c in &buf {
            let r = root(c) as usize;
            last[r] = last[r].max(i);
            live_read[c as usize] = true;
            live_read[r] = true;
        }
        buf.clear();
        op.writes(&mut buf);
        for &c in &buf {
            if c != NONE {
                write[c as usize] = i;
                last[c as usize] = last[c as usize].max(i);
            }
        }
    }
    // An output is kept when read (not through an alias) or crossing back.
    let kept =
        |c: u32| c != NONE && (is_back(c) || (live_read[c as usize] && !alias.contains_key(&c)));
    let w = 1usize << RING_LOG2;
    let mut holder: HashMap<usize, usize> = HashMap::new();
    let ring_base = TMP + 32;
    let mut ring_slot: Vec<Option<u64>> = vec![None; n];
    let needs_slot = |c: usize, home: &[Option<u64>]| {
        home[c].is_none() && write[c] != usize::MAX && kept(c as u32)
    };
    for c in 0..n {
        if !needs_slot(c, &home) {
            continue;
        }
        let key = c % w;
        let ok = match holder.get(&key) {
            None => true,
            Some(&p) => last[p] < write[c] || (last[p] == write[c] && write[p] < write[c]),
        };
        if ok {
            holder.insert(key, c);
            ring_slot[c] = Some(ring_base + 32 * key as u64);
        }
    }
    let own_base = ring_base + 32 * w as u64;
    let mut own = 0u64;
    for c in 0..n {
        if !needs_slot(c, &home) {
            continue;
        }
        home[c] = Some(match ring_slot[c] {
            Some(a) => a,
            None => {
                own += 1;
                own_base + 32 * (own - 1)
            }
        });
    }
    let tmp_slots = w + own as usize + 1;
    let sink = TMP as i64;
    let raddr = |c: u32| -> Result<i64, String> {
        match alias.get(&c) {
            Some(&(src, off)) => home[src as usize]
                .map(|a| (a + off) as i64)
                .ok_or_else(|| format!("alias source {src} has no home")),
            None => home[c as usize]
                .map(|a| a as i64)
                .ok_or_else(|| format!("cell {c} has no home")),
        }
    };
    let waddr = |c: u32| -> i64 {
        if kept(c) {
            home[c as usize].map_or(sink, |a| a as i64)
        } else {
            sink
        }
    };

    // Reads of each cell over the lowered program.
    let mut lreads = vec![0u32; n];
    for op in &lops {
        buf.clear();
        op.reads(&mut buf);
        for &c in &buf {
            lreads[c as usize] += 1;
        }
    }
    let only_next = |c: u32| c != NONE && !is_back(c) && lreads[c as usize] == 1;
    // The run of a permutation's first three inputs: `(base, flags)`.
    let run_of = |ins: &[HIn; 4]| -> Result<Option<(i64, u32)>, String> {
        let mut v = Vec::new();
        for i in &ins[..3] {
            match i {
                HIn::Pack(l, ch) => {
                    for (q, &c) in l.iter().enumerate() {
                        v.push((raddr(c)?, ch >> q & 1));
                    }
                }
                HIn::Cell(_) => return Ok(None),
            }
        }
        let x = v[0].0;
        let ext = v
            .iter()
            .enumerate()
            .all(|(i, &(a, ch))| ch == 0 && a == x + 32 * (i / 3) as i64 + 8 * (i % 3) as i64);
        let base = v
            .iter()
            .enumerate()
            .all(|(i, &(a, ch))| ch == v[0].1 && a == x + 32 * i as i64);
        Ok(if ext {
            Some((x, 1u32 << 8))
        } else if base {
            Some((x, 1 << 9 | v[0].1 << 10))
        } else {
            None
        })
    };
    let fuse_macros = fuse && std::env::var_os("VBG_NO_MACROS").is_none();

    // The ops.
    let mut gops: Vec<GOp> = Vec::new();
    let mut li = 0;
    while li < lops.len() {
        let op = &lops[li];
        li += 1;
        let mut opnds = Vec::new();
        // A chain: permutations over consecutive runs, each capacity the
        // previous digest, read by it alone.
        if fuse_macros
            && let LOp::Hash { ins, .. } = op
            && let HIn::Cell(cap) = ins[3]
            && let Some((x, kind)) = run_of(ins)?
        {
            let stride = if kind & (1 << 8) != 0 { 128 } else { 384 };
            let mut last = li - 1;
            loop {
                let LOp::Hash { outs: po, .. } = &lops[last] else {
                    unreachable!()
                };
                let Some(LOp::Hash { ins: ni, .. }) = lops.get(last + 1) else {
                    break;
                };
                if !(matches!(ni[3], HIn::Cell(c) if c == po[0])
                    && only_next(po[0])
                    && po[1..].iter().all(|&o| !kept(o))
                    && run_of(ni)? == Some((x + stride * (last + 1 - (li - 1)) as i64, kind)))
                {
                    break;
                }
                last += 1;
            }
            if last > li - 1 {
                let LOp::Hash { outs, .. } = &lops[last] else {
                    unreachable!()
                };
                let count = (last + 1 - (li - 1)) as u32;
                if count > 255 {
                    return Err("a chain too long".into());
                }
                let mut flags = kind | count << 11;
                opnds.extend([x, raddr(cap)?]);
                for (j, &o) in outs.iter().enumerate() {
                    if kept(o) {
                        flags |= 1 << (4 + j);
                        opnds.push(waddr(o));
                    }
                }
                gops.push(GOp {
                    word: op_word(OP_CHAIN, flags),
                    opnds,
                });
                li = last + 1;
                continue;
            }
        }
        // Four asserts over the lanes of two cells.
        if fuse_macros
            && let LOp::Assert { x, y, checks: 0 } = op
            && li + 3 <= lops.len()
        {
            let (x0, y0) = (raddr(*x)?, raddr(*y)?);
            let mut all = true;
            for q in 1..4 {
                match &lops[li - 1 + q] {
                    LOp::Assert { x, y, checks: 0 }
                        if raddr(*x)? == x0 + 8 * q as i64 && raddr(*y)? == y0 + 8 * q as i64 => {}
                    _ => all = false,
                }
            }
            if all {
                gops.push(GOp {
                    word: op_word(OP_ASSERT4, 0),
                    opnds: vec![x0, y0],
                });
                li += 3;
                continue;
            }
        }
        let word = match op {
            LOp::Hash { ins, outs } => {
                let mut flags = 0;
                let lane_addrs: Option<Vec<(i64, u32)>> = {
                    let mut v = Vec::new();
                    let mut all = true;
                    for i in &ins[..3] {
                        match i {
                            HIn::Pack(l, ch) => {
                                for (q, &c) in l.iter().enumerate() {
                                    v.push((raddr(c)?, ch >> q & 1));
                                }
                            }
                            HIn::Cell(_) => all = false,
                        }
                    }
                    all.then_some(v)
                };
                let run = lane_addrs.as_ref().and_then(|v| {
                    let x = v[0].0;
                    let ext = v.iter().enumerate().all(|(i, &(a, ch))| {
                        ch == 0 && a == x + 32 * (i / 3) as i64 + 8 * (i % 3) as i64
                    });
                    let base = v
                        .iter()
                        .enumerate()
                        .all(|(i, &(a, ch))| ch == v[0].1 && a == x + 32 * i as i64);
                    if ext {
                        Some((x, 1u32 << 8))
                    } else if base {
                        Some((x, 1 << 9 | v[0].1 << 10))
                    } else {
                        None
                    }
                });
                match run {
                    Some((x, f)) => {
                        flags |= f;
                        opnds.push(x);
                        match ins[3] {
                            HIn::Cell(c) => opnds.push(raddr(c)?),
                            HIn::Pack(l, ch) => {
                                flags |= 1 << 3;
                                for (q, &c) in l.iter().enumerate() {
                                    opnds.push(raddr(c)? | (ch >> q & 1) as i64);
                                }
                            }
                        }
                    }
                    None => {
                        for (j, i) in ins.iter().enumerate() {
                            match i {
                                HIn::Cell(c) => opnds.push(raddr(*c)?),
                                HIn::Pack(l, ch) => {
                                    // The four lanes of one cell, in order: the cell.
                                    let a0 = raddr(l[0])?;
                                    let whole = *ch == 0
                                        && (1..4)
                                            .all(|q| raddr(l[q]).ok() == Some(a0 + 8 * q as i64));
                                    if whole {
                                        opnds.push(a0);
                                        continue;
                                    }
                                    flags |= 1 << j;
                                    for (q, &c) in l.iter().enumerate() {
                                        opnds.push(raddr(c)? | (ch >> q & 1) as i64);
                                    }
                                }
                            }
                        }
                    }
                }
                for (j, &o) in outs.iter().enumerate() {
                    if kept(o) {
                        flags |= 1 << (4 + j);
                        opnds.push(waddr(o));
                    }
                }
                op_word(OP_HASH, flags)
            }
            LOp::Walk4 { ins, outs, checks } => {
                for &c in ins {
                    opnds.push(raddr(c)?);
                }
                let mut flags = *checks;
                for (j, &o) in outs.iter().enumerate() {
                    if kept(o) {
                        flags |= 1 << (2 + j);
                        opnds.push(waddr(o));
                    }
                }
                op_word(OP_WALK4, flags)
            }
            LOp::Select {
                bit,
                l,
                r,
                outs,
                check,
            } => {
                opnds.extend([raddr(*bit)?, raddr(*l)?, raddr(*r)?]);
                let mut flags = *check as u32;
                for (j, &o) in outs.iter().enumerate() {
                    if kept(o) {
                        flags |= 1 << (1 + j);
                        opnds.push(waddr(o));
                    }
                }
                op_word(OP_SELECT, flags)
            }
            LOp::Mux {
                bits,
                leaves,
                out,
                checks,
                bsrc,
            } => {
                let mut flags = bits.len() as u32 | checks << 4;
                match bsrc {
                    Some((src, j0)) => {
                        flags |= 1 << 12 | j0 << 13;
                        opnds.push(raddr(*src)?);
                    }
                    None => {
                        for &b in bits {
                            opnds.push(raddr(b)?);
                        }
                    }
                }
                let addrs: Vec<i64> = leaves.iter().map(|&l| raddr(l)).collect::<Result<_, _>>()?;
                if addrs
                    .iter()
                    .enumerate()
                    .all(|(i, &x)| x == addrs[0] + 32 * i as i64)
                {
                    flags |= 1 << 19;
                    opnds.push(addrs[0]);
                } else {
                    opnds.extend(addrs);
                }
                opnds.push(waddr(*out));
                op_word(OP_MUX, flags)
            }
            LOp::Path {
                cur,
                sibs,
                bits,
                outs,
                checks,
                bsrc,
            } => {
                let levels = sibs.len() as u32 / 3;
                if levels > 255 {
                    return Err("a path too long".into());
                }
                let mut flags = *checks | levels << 8;
                opnds.extend([raddr(*cur)?, raddr(sibs[0])?]);
                match bsrc {
                    Some((src, j0)) => {
                        flags |= 1 << 16 | j0 << 17;
                        opnds.push(raddr(*src)?);
                    }
                    None => {
                        for &b in bits {
                            opnds.push(raddr(b)?);
                        }
                    }
                }
                for (j, &o) in outs.iter().enumerate() {
                    if kept(o) {
                        flags |= 1 << (2 + j);
                        opnds.push(waddr(o));
                    }
                }
                op_word(OP_PATH, flags)
            }
            LOp::Pack { ins, out, checks } => {
                for &c in ins {
                    opnds.push(raddr(c)?);
                }
                opnds.push(waddr(*out));
                op_word(OP_PACK, *checks)
            }
            LOp::Unpack { src, outs } => {
                if !outs.iter().any(|&o| kept(o)) {
                    continue;
                }
                opnds.push(raddr(*src)?);
                let mut flags = 0;
                for (j, &o) in outs.iter().enumerate() {
                    if kept(o) {
                        flags |= 1 << j;
                        opnds.push(waddr(o));
                    }
                }
                op_word(OP_UNPACK, flags)
            }
            LOp::BitDec {
                src,
                n: nb,
                halves,
                outs,
                check,
            } => {
                opnds.push(raddr(*src)?);
                let bits: Vec<i64> = outs[..*nb as usize].iter().map(|&o| waddr(o)).collect();
                let mut flags = nb | (*halves as u32) << 7 | (*check as u32) << 8;
                if !bits.is_empty()
                    && bits
                        .iter()
                        .enumerate()
                        .all(|(i, &x)| x == bits[0] + 32 * i as i64)
                {
                    flags |= 1 << 9;
                    opnds.push(bits[0]);
                } else {
                    opnds.extend(bits);
                }
                for &o in &outs[*nb as usize..] {
                    opnds.push(waddr(o));
                }
                op_word(OP_BITDEC, flags)
            }
            LOp::Base {
                op,
                ins,
                out,
                checks,
            } => {
                for &c in ins {
                    opnds.push(raddr(c)?);
                }
                opnds.push(waddr(*out));
                op_word(OP_BASE, op | checks << 4)
            }
            LOp::Ext {
                op,
                ins,
                out,
                checks,
            } => {
                for &c in ins {
                    opnds.push(raddr(c)?);
                }
                opnds.push(waddr(*out));
                op_word(OP_EXT, op | checks << 4)
            }
            LOp::Assert { x, y, checks } => {
                opnds.extend([raddr(*x)?, raddr(*y)?]);
                op_word(OP_ASSERT, *checks)
            }
            LOp::Public { src } => {
                opnds.push(raddr(*src)?);
                op_word(OP_PUBLIC, 0)
            }
        };
        gops.push(GOp { word, opnds });
    }

    let ops = gops.len();
    let mut roller = Roller::new(&gops);
    let items = roller.roll();
    let loops = roller.next_id as usize;
    let mut code = Vec::new();
    for it in &items {
        encode(&gops, it, &mut code);
    }
    code.push(OP_HALT);
    if std::env::var_os("VBG_STATS").is_some() {
        let mut by: std::collections::BTreeMap<u32, (usize, usize)> = Default::default();
        for g in &gops {
            let e = by.entry(g.word & 0xff).or_default();
            e.0 += 1;
            e.1 += g.opnds.len();
        }
        eprintln!("  generic ops by code (count, operands): {by:?}");
        let mut modes: std::collections::BTreeMap<u32, usize> = Default::default();
        for g in &gops {
            if g.word & 0xff == OP_HASH {
                *modes.entry((g.word >> 8) & 0x70f).or_default() += 1;
            }
        }
        eprintln!("  hash modes (run bits 0x700, packed inputs 0xf): {modes:?}");
        // Executed iterations and patched words, per loop shape.
        fn walk(
            it: &Item,
            mult: u64,
            acc: &mut (u64, u64, u64),
            hist: &mut std::collections::BTreeMap<(u32, usize, usize), (u64, u64)>,
        ) {
            if let Item::Loop(l) = it {
                let nnz = l.delta.iter().filter(|d| **d != 0).count();
                acc.0 += mult * l.reps as u64;
                acc.1 += mult * l.reps as u64 * nnz as u64;
                acc.2 += mult;
                let e = hist
                    .entry((l.reps.min(1000), l.body.len().min(1000), nnz.min(1000)))
                    .or_default();
                e.0 += mult;
                e.1 += mult * l.reps as u64 * nnz as u64;
                for b in &l.body {
                    walk(b, mult * l.reps as u64, acc, hist);
                }
            }
        }
        let mut acc = (0, 0, 0);
        let mut hist = Default::default();
        for it in &items {
            walk(it, 1, &mut acc, &mut hist);
        }
        eprintln!(
            "  generic loops: {} entries, {} iterations, {} patched words",
            acc.2, acc.0, acc.1
        );
        let mut v: Vec<_> = hist.into_iter().collect();
        v.sort_by(|a, b| b.1.1.cmp(&a.1.1));
        for ((reps, body, nnz), (entries, patched)) in v.iter().take(15) {
            eprintln!(
                "    reps {reps} body {body} nnz {nnz}: {entries} entries, {patched} patched"
            );
        }
    }
    // A hint lane is checked canonical unless every use of its value is
    // one a non-canonical value fails or cannot change: a permutation input
    // (the table bounds its lanes), an assert, BACK (hashed into `c_bin`),
    // or a select's, mux's, pack's or unpack's input whose outputs are.
    let mut need = vec![false; n];
    for op in lops.iter().rev() {
        let mark = |need: &mut Vec<bool>, c: u32| need[root(c) as usize] = true;
        let needed = |need: &Vec<bool>, c: u32| c != NONE && need[c as usize];
        match op {
            LOp::Hash { .. } | LOp::Assert { .. } => {}
            LOp::Walk4 { ins, .. } => {
                mark(&mut need, ins[0]);
                mark(&mut need, ins[1]);
            }
            LOp::Path { bits, bsrc, .. } => match bsrc {
                Some((src, _)) => mark(&mut need, *src),
                None => bits.iter().for_each(|&b| mark(&mut need, b)),
            },
            LOp::Select {
                bit, l, r, outs, ..
            } => {
                mark(&mut need, *bit);
                if outs.iter().any(|&o| needed(&need, o)) {
                    mark(&mut need, *l);
                    mark(&mut need, *r);
                }
            }
            LOp::Mux {
                bits,
                leaves,
                out,
                bsrc,
                ..
            } => {
                match bsrc {
                    Some((src, _)) => mark(&mut need, *src),
                    None => bits.iter().for_each(|&b| mark(&mut need, b)),
                }
                if needed(&need, *out) {
                    leaves.iter().for_each(|&l| mark(&mut need, l));
                }
            }
            LOp::Pack { ins, out, .. } => {
                if needed(&need, *out) {
                    ins.iter().for_each(|&c| mark(&mut need, c));
                }
            }
            LOp::Unpack { src, outs } => {
                if outs.iter().any(|&o| needed(&need, o)) {
                    mark(&mut need, *src);
                }
            }
            LOp::BitDec { src, .. } => mark(&mut need, *src),
            LOp::Base { ins, .. } | LOp::Ext { ins, .. } => {
                ins.iter().for_each(|&c| mark(&mut need, c))
            }
            LOp::Public { src } => mark(&mut need, *src),
        }
    }
    let strict = std::env::var_os("VBG_CANON_ALL").is_some();
    let mut canon: Vec<(u64, u64)> = Vec::new();
    for &(c, at, lanes, shared, free) in &hint_lanes {
        let check = if strict {
            !prog.rpx_only[c as usize]
        } else {
            need[c as usize]
        };
        for (p, &l) in lanes.iter().enumerate() {
            if l < free && !(shared && l < 3) && check {
                let a = at + 8 * p as u64;
                match canon.last_mut() {
                    Some((s, m)) if *s + 8 * *m == a => *m += 1,
                    _ => canon.push((a, 1)),
                }
            }
        }
    }
    head[6] = canon.len() as u32;
    for (a, m) in &canon {
        head.extend([rel(*a), *m as u32]);
    }
    head.extend(base_checks.iter().map(|&a| rel(a)));
    for (t, m) in &type_runs {
        head.extend([*t, *m]);
    }
    if head.len() % 2 == 1 {
        head.push(0);
    }
    let mut words = head;
    words.extend(code);
    words[1] = words.len() as u32;
    while !words.len().is_multiple_of(24) {
        words.push(0);
    }
    if !words.chunks(2).all(|p| p[1] != u32::MAX || p[0] == 0) {
        return Err("a felt of the schedule is not canonical".into());
    }
    Ok(Schedule {
        words,
        layout,
        n_in,
        n_back,
        stream_lanes: lanes_of,
        ops,
        items: items.len(),
        loops,
        tmp_slots,
    })
}

#[derive(Clone)]
enum Item {
    Op(u32),
    Loop(std::rc::Rc<Loop>),
}

struct Loop {
    reps: u32,
    body: Vec<Item>,
    delta: Vec<i64>,
    shape: u32,
    opnds: Vec<i64>,
    size: usize,
}

struct Roller<'a> {
    gops: &'a [GOp],
    h: f64,
    shapes: HashMap<Vec<u64>, u32>,
    op_shape: Vec<u32>,
    next_id: u32,
}

impl<'a> Roller<'a> {
    fn new(gops: &'a [GOp]) -> Self {
        let mut r = Roller {
            gops,
            h: std::env::var("VBG_H")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(1.5),
            shapes: HashMap::new(),
            op_shape: Vec::new(),
            next_id: 0,
        };
        for g in gops {
            let s = r.intern(vec![g.word as u64, g.opnds.len() as u64]);
            r.op_shape.push(s);
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

    fn opnds<'b>(&'b self, it: &'b Item) -> &'b [i64] {
        match it {
            Item::Op(k) => &self.gops[*k as usize].opnds,
            Item::Loop(l) => &l.opnds,
        }
    }

    fn size(it: &Item) -> usize {
        match it {
            Item::Op(_) => 1,
            Item::Loop(l) => l.size + 2,
        }
    }

    /// Schedule words `it` encodes to.
    fn words(&self, it: &Item) -> usize {
        match it {
            Item::Op(k) => 1 + self.gops[*k as usize].opnds.len(),
            Item::Loop(l) => {
                let nnz = l.delta.iter().filter(|d| **d != 0).count();
                5 + 2 * nnz + l.body.iter().map(|b| self.words(b)).sum::<usize>()
            }
        }
    }

    fn make_loop(&mut self, body: Vec<Item>, reps: u32, delta: Vec<i64>) -> std::rc::Rc<Loop> {
        let mut key = vec![u64::MAX, reps as u64, body.len() as u64];
        key.extend(body.iter().map(|it| self.shape(it) as u64));
        key.extend(delta.iter().map(|&d| d as u64));
        let shape = self.intern(key);
        let opnds = body.iter().flat_map(|it| self.opnds(it)).copied().collect();
        let size = body.iter().map(Self::size).sum();
        self.next_id += 1;
        std::rc::Rc::new(Loop {
            reps,
            body,
            delta,
            shape,
            opnds,
            size,
        })
    }

    /// Runs of at least two repetitions of a block of at most `max_p` items,
    /// the operands of each moving by a constant per repetition, become loops,
    /// the run saving the most first.
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
            let mut best: Option<(usize, usize, Vec<i64>, usize)> = None;
            for &p in &periods {
                if i + 2 * p > n || hash(i, p) != hash(i + p, p) {
                    continue;
                }
                let base: Vec<i64> = items[i..i + p]
                    .iter()
                    .flat_map(|it| self.opnds(it))
                    .copied()
                    .collect();
                let delta: Vec<i64> = items[i + p..i + 2 * p]
                    .iter()
                    .flat_map(|it| self.opnds(it))
                    .zip(&base)
                    .map(|(y, x)| y - x)
                    .collect();
                if delta.len() != base.len() {
                    continue;
                }
                let mut reps = 2;
                while i + (reps + 1) * p <= n && hash(i, p) == hash(i + reps * p, p) && {
                    let r = reps as i64;
                    items[i + reps * p..i + (reps + 1) * p]
                        .iter()
                        .flat_map(|it| self.opnds(it))
                        .zip(base.iter().zip(&delta))
                        .all(|(y, (x, d))| *y == x + r * d)
                } {
                    reps += 1;
                }
                // Cycles, a schedule word costing `H` to hash: a loop pays
                // its frame copy and a patch per moving operand a repetition.
                let bw: usize = items[i..i + p]
                    .iter()
                    .map(|it| self.words(it))
                    .sum::<usize>()
                    + 1;
                let nnz = delta.iter().filter(|d| **d != 0).count();
                let h = self.h;
                let rolled = 40.0
                    + 6.0 * bw as f64
                    + reps as f64 * (12.0 + 7.0 * nnz as f64)
                    + h * (4 + 2 * nnz + bw) as f64;
                let unrolled = h * (reps * (bw - 1)) as f64;
                let saved = (unrolled - rolled) as usize;
                if unrolled > rolled && best.as_ref().is_none_or(|b| saved > b.3) {
                    best = Some((p, reps, delta, saved));
                }
            }
            match best {
                Some((p, reps, delta, _)) => {
                    let body = items[i..i + p].to_vec();
                    out.push(Item::Loop(self.make_loop(body, reps as u32, delta)));
                    i += reps * p;
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
        let mut items: Vec<Item> = (0..self.gops.len() as u32).map(Item::Op).collect();
        let max = std::env::var("VBG_MAX_PERIOD")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1usize << 16);
        for max_p in [4, 16, 64, 256, 1024, 4096, 1 << 16]
            .into_iter()
            .filter(|&p| p <= max)
        {
            items = self.pass(items, max_p);
        }
        items
    }
}

/// Encodes `it`, returning the word positions of its operands (relative to
/// where it starts), in operand order.
fn encode(gops: &[GOp], it: &Item, out: &mut Vec<u32>) -> Vec<usize> {
    let start = out.len();
    match it {
        Item::Op(k) => {
            let g = &gops[*k as usize];
            out.push(g.word);
            let mut pos = Vec::with_capacity(g.opnds.len());
            for &a in &g.opnds {
                pos.push(out.len() - start);
                out.push(rel(a as u64));
            }
            pos
        }
        Item::Loop(l) => {
            let mut body = Vec::new();
            let mut bpos = Vec::new();
            for b in &l.body {
                let at = body.len();
                bpos.extend(encode(gops, b, &mut body).into_iter().map(|p| p + at));
            }
            body.push(OP_END);
            assert_eq!(bpos.len(), l.delta.len(), "one move per operand");
            let pairs: Vec<(u32, u32)> = bpos
                .iter()
                .zip(&l.delta)
                .filter(|(_, d)| **d != 0)
                .map(|(&p, &d)| (p as u32, (d as u32 as i32 as i64 + BIAS as i64) as u32))
                .collect();
            out.extend([
                op_word(OP_LOOP, 0),
                l.reps,
                body.len() as u32,
                pairs.len() as u32,
            ]);
            for (p, d) in &pairs {
                out.extend([*p, *d]);
            }
            let body_at = out.len() - start;
            out.extend(body);
            bpos.into_iter().map(|p| p + body_at).collect()
        }
    }
}

/// The digest the guest publishes for a schedule: width-16 Poseidon1 as an
/// overwrite sponge over the words two to a felt (little-endian), twelve felts
/// a permutation, the word count in lane 12; lanes 0..4.
pub fn schedule_digest(words: &[u32]) -> [u64; 4] {
    use crypto::hash::poseidon1_w16::{Fp, permute};
    let mut state = [Fp::zero(); 16];
    state[12] = Fp::from(words.len() as u64);
    for chunk in words.chunks(24) {
        for (i, lane) in state[..12].iter_mut().enumerate() {
            let lo = chunk.get(2 * i).copied().unwrap_or(0) as u64;
            let hi = chunk.get(2 * i + 1).copied().unwrap_or(0) as u64;
            *lane = Fp::from(lo | hi << 32);
        }
        state = permute(state);
    }
    core::array::from_fn(|i| state[i].canonical())
}

/// The guest's input: the hint stream (each hint's stream lanes), IN's cells,
/// BACK zero, `α`, the schedule.
pub fn input(s: &Schedule, hints: &[[u64; 4]], in_cells: &[[u64; 3]], alpha: [u64; 3]) -> Vec<u8> {
    assert_eq!(
        hints.len(),
        s.stream_lanes.len(),
        "one stream entry per hint"
    );
    assert_eq!(in_cells.len(), s.n_in, "IN's cells");
    let off = |a: u64| (a - INPUT) as usize;
    let end = off(s.layout.sched_base) + 4 * s.words.len();
    let mut v = vec![0u8; end];
    v[0..4].copy_from_slice(&(s.words.len() as u32).to_le_bytes());
    v[4..8].copy_from_slice(&(off(s.layout.sched_base) as u32).to_le_bytes());
    let mut at = off(s.layout.stream_base);
    for (h, lanes) in hints.iter().zip(&s.stream_lanes) {
        for &l in lanes.iter() {
            v[at..at + 8].copy_from_slice(&h[l].to_le_bytes());
            at += 8;
        }
    }
    for (k, c) in in_cells.iter().enumerate() {
        let at = off(s.layout.in_base) + 32 * k;
        for (l, x) in c.iter().enumerate() {
            v[at + 8 * l..at + 8 * l + 8].copy_from_slice(&x.to_le_bytes());
        }
    }
    let at = off(s.layout.alpha);
    for (l, x) in alpha.iter().enumerate() {
        v[at + 8 * l..at + 8 * l + 8].copy_from_slice(&x.to_le_bytes());
    }
    let at = off(s.layout.sched_base);
    for (i, w) in s.words.iter().enumerate() {
        v[at + 4 * i..at + 4 * i + 4].copy_from_slice(&w.to_le_bytes());
    }
    v
}

/// The guest's public output: the schedule's digest, IN's and BACK's bases
/// and counts, the published words, then `c_bin`, `α` and `s`.
pub fn expected_output(
    s: &Schedule,
    publics: &[[u64; 4]],
    c_bin: [u64; 4],
    alpha: [u64; 3],
    sum: [u64; 3],
) -> Vec<u8> {
    let mut out = Vec::new();
    for x in schedule_digest(&s.words) {
        out.extend(x.to_le_bytes());
    }
    for v in [
        rel(s.layout.in_base),
        rel(s.layout.back_base),
        s.n_in as u32,
        s.n_back as u32,
    ] {
        out.extend(v.to_le_bytes());
    }
    for w in publics {
        for x in w {
            out.extend(x.to_le_bytes());
        }
    }
    for x in c_bin.iter().chain(&alpha).chain(&sum) {
        out.extend(x.to_le_bytes());
    }
    out
}

/// BACK's cells in the guest's memory after a run: lanes 0..3 of each.
pub fn back_cells(s: &Schedule, read: impl Fn(u64) -> u64) -> Vec<[u64; 3]> {
    (0..s.n_back)
        .map(|i| core::array::from_fn(|l| read(s.layout.back_base + 32 * i as u64 + 8 * l as u64)))
        .collect()
}

/// The guest, the same for every leaf.
pub fn guest() -> &'static [u8] {
    &built().0
}

/// Where each part of the guest's code starts, ascending.
pub fn guest_profile() -> &'static [(u64, &'static str)] {
    &built().1
}

/// The guest's ELF and where each part of its code starts.
type Built = (Vec<u8>, Vec<(u64, &'static str)>);

fn built() -> &'static Built {
    static ELF: OnceLock<Built> = OnceLock::new();
    ELF.get_or_init(build_guest)
}

const IP: Reg = S0;
const PUBC: Reg = S1;
const FTOP: Reg = 18;
const CTLP: Reg = 19;
const JT: Reg = 20;
const OB: Reg = 21;
const BI: Reg = 22;
const SB: Reg = 23;
const N_IN: Reg = 24;
const N_BACK: Reg = 25;
const IN_B: Reg = 26;
const BACK_B: Reg = 27;
const ERR: Reg = SP;
const OK: Reg = TP;
const PR: Reg = GP;
const DATA_BASE: u64 = 0x0800_0000;
/// Where the epilogue finds the BACK types.
const TYPES_AT: u64 = SCRATCH + 0x200;

/// `rd = OB + word at off(IP)`: an operand's address.
fn opnd(a: &mut Asm, rd: Reg, off: i64) {
    a.lwu(rd, IP, off);
    a.add(rd, rd, OB);
}

/// `ERR |= lanes 1..4 of the cell at r`, through `t`.
fn upper_zero(a: &mut Asm, r: Reg, t: Reg) {
    for l in 1..4 {
        a.ld(t, r, 8 * l);
        a.or(ERR, ERR, t);
    }
}

/// Runs `body` when flag bit `bit` of the op word in `T0` is set, through `t`.
fn if_flag(a: &mut Asm, bit: u32, t: Reg, body: impl FnOnce(&mut Asm)) {
    let skip = a.label();
    a.srli(t, T0, 8 + bit);
    a.andi(t, t, 1);
    a.beq(t, ZERO, skip);
    body(a);
    a.bind(skip);
}

/// One packed lane: the operand word at `A5` (its address, bit 0 its
/// upper-zero check) into `off(A0)`; advances `A5`. Clobbers `T1..T3`.
fn lane(a: &mut Asm, off: i64) {
    let skip = a.label();
    a.lwu(T1, A5, 0);
    a.addi(A5, A5, 4);
    a.andi(T2, T1, 1);
    a.andi(T1, T1, -8);
    a.add(T1, T1, OB);
    a.ld(T3, T1, 0);
    a.sd(T3, A0, off);
    a.beq(T2, ZERO, skip);
    upper_zero(a, T1, T3);
    a.bind(skip);
}

fn copy_cell(a: &mut Asm, from: Reg, from_off: i64, to: Reg, to_off: i64, regs: [Reg; 4]) {
    for (l, &r) in regs.iter().enumerate() {
        a.ld(r, from, from_off + 8 * l as i64);
    }
    for (l, &r) in regs.iter().enumerate() {
        a.sd(r, to, to_off + 8 * l as i64);
    }
}

fn build_guest() -> Built {
    let mut a = Asm::default();
    let main = a.label();
    a.jal(ZERO, main);
    let (mut a, rt) = native_vbin::field_runtime(a);
    let next = a.label();
    let fail = a.label();
    let epilogue = a.label();
    let handlers: Vec<Label> = (0..N_OPS).map(|_| a.label()).collect();
    let mut marks: Vec<(usize, &'static str)> = vec![(0, "runtime")];

    // Prologue.
    a.bind(main);
    marks.push((a.pc(), "prologue"));
    a.li(PR, P as i64);
    a.li(ERR, 0);
    a.li(OK, 1);
    a.li(A0, STATE as i64);
    a.li(A7, P1W16_SYSCALL);
    a.li(OB, OPND_BASE as i64);
    a.li(BI, BIAS as i64);
    a.li(JT, DATA_BASE as i64);
    a.li(T3, INPUT as i64);
    a.lwu(T4, T3, 0);
    a.lwu(SB, T3, 4);
    a.add(SB, SB, T3);
    // The schedule's digest: twelve felts (24 words) a permutation.
    for l in 0..16 {
        a.sd(ZERO, A0, 8 * l);
    }
    a.sd(T4, A0, 96);
    a.li(T5, 24);
    a.divu(T6, T4, T5);
    a.mul(T5, T6, T5);
    a.xor(T5, T5, T4);
    a.or(ERR, ERR, T5);
    a.mv(T3, SB);
    {
        let done = a.label();
        a.beq(T6, ZERO, done);
        let top = a.here();
        for l in 0..12 {
            a.ld(T5, T3, 8 * l);
            a.sd(T5, A0, 8 * l);
        }
        a.ecall();
        a.addi(T3, T3, 96);
        a.addi(T6, T6, -1);
        a.bnez_back(T6, top);
        a.bind(done);
    }
    a.li(T3, PUB as i64);
    copy_cell(&mut a, A0, 0, T3, 0, [T4, T5, T6, A1]);
    // The header.
    a.lwu(T1, SB, 0);
    a.xori(T1, T1, VERSION as i64);
    a.or(ERR, ERR, T1);
    a.lwu(N_IN, SB, 8);
    a.lwu(N_BACK, SB, 12);
    a.lwu(IN_B, SB, 16);
    a.add(IN_B, IN_B, OB);
    a.lwu(BACK_B, SB, 20);
    a.add(BACK_B, BACK_B, OB);
    a.addi(IP, SB, 40);
    a.lwu(T1, SB, 36);
    a.slli(T1, T1, 5);
    a.add(IP, IP, T1);
    // Canonical hint lanes, four at a time.
    {
        a.lwu(T1, SB, 24);
        let done = a.label();
        a.beq(T1, ZERO, done);
        let run = a.here();
        a.lwu(T2, IP, 0);
        a.add(T2, T2, OB);
        a.lwu(T3, IP, 4);
        a.addi(IP, IP, 8);
        a.li(T5, 4);
        let (quad, single, run_done) = (a.label(), a.label(), a.label());
        a.bind(quad);
        a.bltu(T3, T5, single);
        for l in 0..4 {
            a.ld(T4, T2, 8 * l);
            a.sltu(T4, T4, PR);
            a.and(OK, OK, T4);
        }
        a.addi(T2, T2, 32);
        a.addi(T3, T3, -4);
        a.jal(ZERO, quad);
        a.bind(single);
        a.beq(T3, ZERO, run_done);
        a.ld(T4, T2, 0);
        a.sltu(T4, T4, PR);
        a.and(OK, OK, T4);
        a.addi(T2, T2, 8);
        a.addi(T3, T3, -1);
        a.jal(ZERO, single);
        a.bind(run_done);
        a.addi(T1, T1, -1);
        a.bnez_back(T1, run);
        a.bind(done);
    }
    // Shared base hints: lanes 1 and 2 zero.
    {
        a.lwu(T1, SB, 28);
        let done = a.label();
        a.beq(T1, ZERO, done);
        let top = a.here();
        a.lwu(T2, IP, 0);
        a.add(T2, T2, OB);
        a.ld(T3, T2, 8);
        a.or(ERR, ERR, T3);
        a.ld(T3, T2, 16);
        a.or(ERR, ERR, T3);
        a.addi(IP, IP, 4);
        a.addi(T1, T1, -1);
        a.bnez_back(T1, top);
        a.bind(done);
    }
    // IN's lane 3, four cells at a time.
    {
        a.mv(T1, N_IN);
        a.mv(T2, IN_B);
        a.li(T5, 4);
        let (quad, single, done) = (a.label(), a.label(), a.label());
        a.bind(quad);
        a.bltu(T1, T5, single);
        for c in 0..4 {
            a.ld(T3, T2, 32 * c + 24);
            a.or(ERR, ERR, T3);
        }
        a.addi(T2, T2, 128);
        a.addi(T1, T1, -4);
        a.jal(ZERO, quad);
        a.bind(single);
        a.beq(T1, ZERO, done);
        a.ld(T3, T2, 24);
        a.or(ERR, ERR, T3);
        a.addi(T2, T2, 32);
        a.addi(T1, T1, -1);
        a.jal(ZERO, single);
        a.bind(done);
    }
    // The types, then the constants.
    a.li(T3, TYPES_AT as i64);
    a.sd(IP, T3, 0);
    a.lwu(T1, SB, 32);
    a.slli(T1, T1, 2);
    a.add(IP, IP, T1);
    a.addi(IP, IP, 7);
    a.andi(IP, IP, -8);
    a.li(CTLP, CTL as i64);
    a.li(FTOP, FRAMES as i64);
    a.li(PUBC, (PUB + 48) as i64);

    // Dispatch.
    a.bind(next);
    marks.push((a.pc(), "dispatch"));
    a.lwu(T0, IP, 0);
    a.andi(T1, T0, 0xff);
    a.slli(T1, T1, 3);
    a.add(T1, T1, JT);
    a.ld(T1, T1, 0);
    a.jalr(ZERO, T1, 0);

    // HALT.
    a.bind(handlers[OP_HALT as usize]);
    marks.push((a.pc(), "halt"));
    a.jal(ZERO, epilogue);

    // END: the next repetition, the operands moved; or back to the parent.
    a.bind(handlers[OP_END as usize]);
    marks.push((a.pc(), "end"));
    {
        let done = a.label();
        a.ld(T1, CTLP, 0);
        a.addi(T1, T1, -1);
        a.beq(T1, ZERO, done);
        a.sd(T1, CTLP, 0);
        a.ld(T2, CTLP, 8);
        a.ld(T3, CTLP, 16);
        a.ld(T4, CTLP, 24);
        let restart = a.label();
        a.beq(T4, ZERO, restart);
        let top = a.here();
        a.lwu(T5, T3, 0);
        a.lwu(T6, T3, 4);
        a.sub(T6, T6, BI);
        a.slli(T5, T5, 2);
        a.add(T5, T5, T2);
        a.lwu(A1, T5, 0);
        a.add(A1, A1, T6);
        a.sw(A1, T5, 0);
        a.addi(T3, T3, 8);
        a.addi(T4, T4, -1);
        a.bnez_back(T4, top);
        a.bind(restart);
        a.mv(IP, T2);
        a.jal(ZERO, next);
        a.bind(done);
        a.ld(IP, CTLP, 32);
        a.ld(FTOP, CTLP, 40);
        a.addi(CTLP, CTLP, -48);
        a.jal(ZERO, next);
    }

    // LOOP: the body copied to a frame, the repetitions counted.
    a.bind(handlers[OP_LOOP as usize]);
    marks.push((a.pc(), "loop"));
    {
        a.lwu(T1, IP, 4);
        a.lwu(T2, IP, 8);
        a.lwu(T3, IP, 12);
        a.addi(T4, IP, 16);
        a.slli(T5, T3, 3);
        a.add(T5, T5, T4);
        a.slli(T6, T2, 2);
        a.add(A1, T5, T6);
        a.addi(CTLP, CTLP, 48);
        a.sd(T1, CTLP, 0);
        a.sd(FTOP, CTLP, 8);
        a.sd(T4, CTLP, 16);
        a.sd(T3, CTLP, 24);
        a.sd(A1, CTLP, 32);
        a.sd(FTOP, CTLP, 40);
        a.mv(A2, FTOP);
        a.mv(A3, T5);
        a.mv(A4, T2);
        let top = a.here();
        a.lwu(A5, A3, 0);
        a.sw(A5, A2, 0);
        a.addi(A3, A3, 4);
        a.addi(A2, A2, 4);
        a.addi(A4, A4, -1);
        a.bnez_back(A4, top);
        a.mv(IP, FTOP);
        a.add(FTOP, FTOP, T6);
        a.addi(FTOP, FTOP, 7);
        a.andi(FTOP, FTOP, -8);
        a.jal(ZERO, next);
    }

    // HASH: inputs (a cell, four packed lanes, or a run of twelve lanes and
    // input 3), then the kept outputs.
    a.bind(handlers[OP_HASH as usize]);
    marks.push((a.pc(), "hash"));
    {
        let cp = [T3, T4, T5, A1];
        let (generic, hashit) = (a.label(), a.label());
        a.srli(T6, T0, 8);
        a.addi(A5, IP, 4);
        a.andi(T1, T6, 0x300);
        a.beq(T1, ZERO, generic);
        {
            let (base_run, after_run) = (a.label(), a.label());
            a.lwu(T1, A5, 0);
            a.add(T1, T1, OB);
            a.addi(A5, A5, 4);
            a.andi(T2, T6, 0x100);
            a.beq(T2, ZERO, base_run);
            for i in 0..4 {
                for l in 0..3 {
                    a.ld(T3, T1, 32 * i + 8 * l);
                    a.sd(T3, A0, 8 * (3 * i + l));
                }
            }
            a.jal(ZERO, after_run);
            a.bind(base_run);
            for i in 0..12 {
                a.ld(T3, T1, 32 * i);
                a.sd(T3, A0, 8 * i);
            }
            a.andi(T2, T6, 0x400);
            a.beq(T2, ZERO, after_run);
            for i in 0..12 {
                for l in 1..4 {
                    a.ld(T3, T1, 32 * i + 8 * l);
                    a.or(ERR, ERR, T3);
                }
            }
            a.bind(after_run);
            let packed = a.label();
            a.andi(T2, T6, 8);
            a.bne(T2, ZERO, packed);
            a.lwu(T1, A5, 0);
            a.add(T1, T1, OB);
            a.addi(A5, A5, 4);
            copy_cell(&mut a, T1, 0, A0, 96, cp);
            a.jal(ZERO, hashit);
            a.bind(packed);
            for q in 0..4 {
                lane(&mut a, 96 + 8 * q);
            }
            a.jal(ZERO, hashit);
        }
        a.bind(generic);
        for j in 0..4 {
            let (packed, done) = (a.label(), a.label());
            a.andi(T2, T6, 1 << j);
            a.bne(T2, ZERO, packed);
            a.lwu(T1, A5, 0);
            a.add(T1, T1, OB);
            a.addi(A5, A5, 4);
            copy_cell(&mut a, T1, 0, A0, 32 * j, cp);
            a.jal(ZERO, done);
            a.bind(packed);
            for q in 0..4 {
                lane(&mut a, 32 * j + 8 * q);
            }
            a.bind(done);
        }
        a.bind(hashit);
        a.ecall();
        for j in 0..4 {
            let skip = a.label();
            a.andi(T2, T6, 1 << (4 + j));
            a.beq(T2, ZERO, skip);
            a.lwu(T1, A5, 0);
            a.add(T1, T1, OB);
            a.addi(A5, A5, 4);
            copy_cell(&mut a, A0, 32 * j, T1, 0, cp);
            a.bind(skip);
        }
        a.mv(IP, A5);
        a.jal(ZERO, next);
    }

    // WALK4: b0, b1, cur, s0, s1, s2, then the kept outputs.
    a.bind(handlers[OP_WALK4 as usize]);
    marks.push((a.pc(), "walk4"));
    {
        opnd(&mut a, T1, 4);
        a.ld(T2, T1, 0);
        if_flag(&mut a, 0, T6, |a| upper_zero(a, T1, T3));
        a.sltiu(T3, T2, 2);
        a.and(OK, OK, T3);
        opnd(&mut a, T1, 8);
        a.ld(T4, T1, 0);
        if_flag(&mut a, 1, T6, |a| upper_zero(a, T1, T3));
        a.sltiu(T3, T4, 2);
        a.and(OK, OK, T3);
        a.slli(T4, T4, 1);
        a.add(T2, T2, T4);
        a.andi(T2, T2, 3);
        opnd(&mut a, T1, 12);
        for l in 0..4 {
            a.ld(A1 + l as Reg, T1, 8 * l as i64);
        }
        for q in 0..3 {
            opnd(&mut a, T1, 16 + 4 * q);
            a.sltiu(A5, T2, q + 1);
            a.slli(A5, A5, 5);
            a.add(A5, A5, A0);
            copy_cell(&mut a, T1, 0, A5, 32 * q, [T3, T4, T5, T6]);
        }
        a.slli(T1, T2, 5);
        a.add(T1, T1, A0);
        for l in 0..4 {
            a.sd(A1 + l as Reg, T1, 8 * l as i64);
        }
        a.ecall();
        a.addi(A5, IP, 28);
        for j in 0..4 {
            let skip = a.label();
            a.srli(T2, T0, 10 + j);
            a.andi(T2, T2, 1);
            a.beq(T2, ZERO, skip);
            a.lwu(T1, A5, 0);
            a.add(T1, T1, OB);
            a.addi(A5, A5, 4);
            copy_cell(&mut a, A0, 32 * j as i64, T1, 0, [T3, T4, T5, T6]);
            a.bind(skip);
        }
        a.mv(IP, A5);
        a.jal(ZERO, next);
    }

    // SELECT: bit, l, r, then out_l and out_r when kept.
    a.bind(handlers[OP_SELECT as usize]);
    marks.push((a.pc(), "select"));
    {
        opnd(&mut a, T1, 4);
        a.ld(T2, T1, 0);
        if_flag(&mut a, 0, T6, |a| upper_zero(a, T1, T3));
        a.sltiu(T3, T2, 2);
        a.and(OK, OK, T3);
        let l = [T3, T4, T5, T6];
        let r = [A1, A2, A3, A4];
        opnd(&mut a, T1, 8);
        for (k, &x) in l.iter().enumerate() {
            a.ld(x, T1, 8 * k as i64);
        }
        opnd(&mut a, T1, 12);
        for (k, &x) in r.iter().enumerate() {
            a.ld(x, T1, 8 * k as i64);
        }
        a.addi(A5, IP, 16);
        let (swap, done) = (a.label(), a.label());
        a.bne(T2, ZERO, swap);
        for order in [[l, r], [r, l]] {
            for (j, regs) in order.iter().enumerate() {
                let skip = a.label();
                a.srli(T1, T0, 9 + j as u32);
                a.andi(T1, T1, 1);
                a.beq(T1, ZERO, skip);
                a.lwu(T1, A5, 0);
                a.add(T1, T1, OB);
                a.addi(A5, A5, 4);
                for (k, &x) in regs.iter().enumerate() {
                    a.sd(x, T1, 8 * k as i64);
                }
                a.bind(skip);
            }
            if order[0] == l {
                a.jal(ZERO, done);
                a.bind(swap);
            }
        }
        a.bind(done);
        a.mv(IP, A5);
        a.jal(ZERO, next);
    }

    // MUX: k bits (or their source felt, bit 12, from bit 13..), 2^k
    // leaves, out: the leaf the bits name.
    a.bind(handlers[OP_MUX as usize]);
    marks.push((a.pc(), "mux"));
    {
        let (explicit, select) = (a.label(), a.label());
        a.srli(T6, T0, 8);
        a.andi(T6, T6, 0xf);
        a.srli(T1, T0, 20);
        a.andi(T1, T1, 1);
        a.beq(T1, ZERO, explicit);
        opnd(&mut a, T1, 4);
        a.ld(T2, T1, 0);
        a.srli(T5, T0, 21);
        a.andi(T5, T5, 63);
        a.srl(T2, T2, T5);
        a.li(A2, 1);
        a.sll(A2, A2, T6);
        a.addi(T5, A2, -1);
        a.and(T2, T2, T5);
        a.addi(A5, IP, 8);
        a.jal(ZERO, select);
        a.bind(explicit);
        a.srli(T5, T0, 12);
        a.li(T2, 0);
        a.li(A2, 1);
        a.addi(A5, IP, 4);
        a.mv(T3, T6);
        let top = a.here();
        a.lwu(T1, A5, 0);
        a.add(T1, T1, OB);
        a.ld(T4, T1, 0);
        let nochk = a.label();
        a.andi(A1, T5, 1);
        a.beq(A1, ZERO, nochk);
        upper_zero(&mut a, T1, A1);
        a.bind(nochk);
        a.srli(T5, T5, 1);
        a.sltiu(A1, T4, 2);
        a.and(OK, OK, A1);
        a.sltu(A1, ZERO, T4);
        a.mul(A1, A1, A2);
        a.add(T2, T2, A1);
        a.slli(A2, A2, 1);
        a.addi(A5, A5, 4);
        a.addi(T3, T3, -1);
        a.bnez_back(T3, top);
        a.bind(select);
        let (table, got) = (a.label(), a.label());
        a.srli(T1, T0, 27);
        a.andi(T1, T1, 1);
        a.beq(T1, ZERO, table);
        a.lwu(T1, A5, 0);
        a.add(T1, T1, OB);
        a.slli(A1, T2, 5);
        a.add(T1, T1, A1);
        a.addi(A3, A5, 4);
        a.jal(ZERO, got);
        a.bind(table);
        a.slli(A1, T2, 2);
        a.add(A1, A1, A5);
        a.lwu(T1, A1, 0);
        a.add(T1, T1, OB);
        a.slli(A3, A2, 2);
        a.add(A3, A3, A5);
        a.bind(got);
        a.lwu(T4, A3, 0);
        a.add(T4, T4, OB);
        copy_cell(&mut a, T1, 0, T4, 0, [T3, T5, A1, A2]);
        a.addi(IP, A3, 4);
        a.jal(ZERO, next);
    }

    // ASSERT: x, y (lane 0 each): x − y must be zero.
    a.bind(handlers[OP_ASSERT as usize]);
    marks.push((a.pc(), "assert"));
    {
        opnd(&mut a, T1, 4);
        a.ld(A1, T1, 0);
        if_flag(&mut a, 0, T6, |a| upper_zero(a, T1, T3));
        opnd(&mut a, T2, 8);
        a.ld(A2, T2, 0);
        if_flag(&mut a, 1, T6, |a| upper_zero(a, T2, T3));
        a.sltu(T3, A1, A2);
        a.sub(A1, A1, A2);
        a.neg(T3, T3);
        a.and(T3, T3, PR);
        a.add(A1, A1, T3);
        a.or(ERR, ERR, A1);
        a.addi(IP, IP, 12);
        a.jal(ZERO, next);
    }

    // CHAIN: run base, first capacity, then the last permutation's kept
    // outputs; n permutations over consecutive runs.
    a.bind(handlers[OP_CHAIN as usize]);
    marks.push((a.pc(), "chain"));
    {
        let cp = [T3, T4, T5, A1];
        a.addi(A5, IP, 4);
        a.lwu(T1, A5, 0);
        a.add(T1, T1, OB);
        a.lwu(T2, A5, 4);
        a.add(T2, T2, OB);
        a.addi(A5, A5, 8);
        copy_cell(&mut a, T2, 0, A0, 96, cp);
        a.srli(T6, T0, 19);
        a.andi(T6, T6, 0xff);
        let (top, base, perm, done) = (a.label(), a.label(), a.label(), a.label());
        a.bind(top);
        a.srli(T2, T0, 16);
        a.andi(T2, T2, 1);
        a.beq(T2, ZERO, base);
        for i in 0..4 {
            for l in 0..3 {
                a.ld(T3, T1, 32 * i + 8 * l);
                a.sd(T3, A0, 8 * (3 * i + l));
            }
        }
        a.addi(T1, T1, 128);
        a.jal(ZERO, perm);
        a.bind(base);
        for i in 0..12 {
            a.ld(T3, T1, 32 * i);
            a.sd(T3, A0, 8 * i);
        }
        {
            let nochk = a.label();
            a.srli(T2, T0, 18);
            a.andi(T2, T2, 1);
            a.beq(T2, ZERO, nochk);
            for i in 0..12 {
                for l in 1..4 {
                    a.ld(T3, T1, 32 * i + 8 * l);
                    a.or(ERR, ERR, T3);
                }
            }
            a.bind(nochk);
        }
        a.addi(T1, T1, 384);
        a.bind(perm);
        a.ecall();
        a.addi(T6, T6, -1);
        a.beq(T6, ZERO, done);
        copy_cell(&mut a, A0, 0, A0, 96, cp);
        a.jal(ZERO, top);
        a.bind(done);
        for j in 0..4 {
            let skip = a.label();
            a.srli(T2, T0, 12 + j);
            a.andi(T2, T2, 1);
            a.beq(T2, ZERO, skip);
            a.lwu(T1, A5, 0);
            a.add(T1, T1, OB);
            a.addi(A5, A5, 4);
            copy_cell(&mut a, A0, 32 * j as i64, T1, 0, cp);
            a.bind(skip);
        }
        a.mv(IP, A5);
        a.jal(ZERO, next);
    }

    // PATH: current digest, first sibling, the levels' bits (or their
    // source felt, bit 24, from bit 25..), then the last level's kept
    // outputs; n levels, three consecutive siblings a level.
    a.bind(handlers[OP_PATH as usize]);
    marks.push((a.pc(), "path"));
    {
        let cur = [A1, A2, A3, A4];
        let cp = [T1, T6, A6, RA];
        opnd(&mut a, T1, 4);
        for (l, &r) in cur.iter().enumerate() {
            a.ld(r, T1, 8 * l as i64);
        }
        opnd(&mut a, T2, 8);
        a.srli(T3, T0, 16);
        a.andi(T3, T3, 0xff);
        let level = |a: &mut Asm| {
            for q in 0..3 {
                a.sltiu(T5, T4, q + 1);
                a.slli(T5, T5, 5);
                a.add(T5, T5, A0);
                copy_cell(a, T2, 32 * q, T5, 32 * q, cp);
            }
            a.slli(T5, T4, 5);
            a.add(T5, T5, A0);
            for (l, &r) in cur.iter().enumerate() {
                a.sd(r, T5, 8 * l as i64);
            }
            a.ecall();
            a.addi(T2, T2, 96);
            for (l, &r) in cur.iter().enumerate() {
                a.ld(r, A0, 8 * l as i64);
            }
            a.addi(T3, T3, -1);
        };
        let (explicit, outputs) = (a.label(), a.label());
        a.srli(T6, T0, 24);
        a.andi(T6, T6, 1);
        a.beq(T6, ZERO, explicit);
        {
            opnd(&mut a, T1, 12);
            a.ld(A5, T1, 0);
            a.srli(T6, T0, 25);
            a.andi(T6, T6, 63);
            a.srl(A5, A5, T6);
            let top = a.here();
            a.andi(T4, A5, 3);
            a.srli(A5, A5, 2);
            level(&mut a);
            a.bnez_back(T3, top);
            a.addi(A5, IP, 16);
            a.jal(ZERO, outputs);
        }
        a.bind(explicit);
        {
            a.addi(A5, IP, 12);
            let top = a.here();
            a.lwu(T1, A5, 0);
            a.add(T1, T1, OB);
            a.ld(T4, T1, 0);
            if_flag(&mut a, 0, T5, |a| upper_zero(a, T1, T5));
            a.sltiu(T5, T4, 2);
            a.and(OK, OK, T5);
            a.lwu(T1, A5, 4);
            a.add(T1, T1, OB);
            a.ld(T5, T1, 0);
            if_flag(&mut a, 1, T6, |a| upper_zero(a, T1, T6));
            a.sltiu(T6, T5, 2);
            a.and(OK, OK, T6);
            a.addi(A5, A5, 8);
            a.slli(T5, T5, 1);
            a.add(T4, T4, T5);
            a.andi(T4, T4, 3);
            level(&mut a);
            a.bnez_back(T3, top);
        }
        a.bind(outputs);
        for j in 0..4 {
            let skip = a.label();
            a.srli(T2, T0, 10 + j);
            a.andi(T2, T2, 1);
            a.beq(T2, ZERO, skip);
            a.lwu(T1, A5, 0);
            a.add(T1, T1, OB);
            a.addi(A5, A5, 4);
            copy_cell(&mut a, A0, 32 * j as i64, T1, 0, [T3, T4, T5, T6]);
            a.bind(skip);
        }
        a.mv(IP, A5);
        a.jal(ZERO, next);
    }

    // ASSERT4: x, y: the four lanes of x and y agree.
    a.bind(handlers[OP_ASSERT4 as usize]);
    marks.push((a.pc(), "assert4"));
    {
        opnd(&mut a, T1, 4);
        opnd(&mut a, T2, 8);
        for l in 0..4 {
            a.ld(A1, T1, 8 * l);
            a.ld(A2, T2, 8 * l);
            a.sltu(T3, A1, A2);
            a.sub(A1, A1, A2);
            a.neg(T3, T3);
            a.and(T3, T3, PR);
            a.add(A1, A1, T3);
            a.or(ERR, ERR, A1);
        }
        a.addi(IP, IP, 12);
        a.jal(ZERO, next);
    }

    // PACK: x0..x3 (lane 0 each), out.
    a.bind(handlers[OP_PACK as usize]);
    marks.push((a.pc(), "pack"));
    {
        let v = [A1, A2, A3, A4];
        for (j, &x) in v.iter().enumerate() {
            opnd(&mut a, T1, 4 + 4 * j as i64);
            a.ld(x, T1, 0);
            if_flag(&mut a, j as u32, T6, |a| upper_zero(a, T1, T3));
        }
        opnd(&mut a, T1, 20);
        for (l, &x) in v.iter().enumerate() {
            a.sd(x, T1, 8 * l as i64);
        }
        a.addi(IP, IP, 24);
        a.jal(ZERO, next);
    }

    // UNPACK: src, then the kept outputs (lane l of src in lane 0).
    a.bind(handlers[OP_UNPACK as usize]);
    marks.push((a.pc(), "unpack"));
    {
        let v = [A1, A2, A3, A4];
        opnd(&mut a, T1, 4);
        for (l, &x) in v.iter().enumerate() {
            a.ld(x, T1, 8 * l as i64);
        }
        a.addi(A5, IP, 8);
        for (l, &x) in v.iter().enumerate() {
            let skip = a.label();
            a.srli(T2, T0, 8 + l as u32);
            a.andi(T2, T2, 1);
            a.beq(T2, ZERO, skip);
            a.lwu(T1, A5, 0);
            a.add(T1, T1, OB);
            a.addi(A5, A5, 4);
            a.sd(x, T1, 0);
            for z in 1..4 {
                a.sd(ZERO, T1, 8 * z);
            }
            a.bind(skip);
        }
        a.mv(IP, A5);
        a.jal(ZERO, next);
    }

    // BITDEC: src, bits (flags & 0x7f), then two byte-swapped halves (bit 7).
    a.bind(handlers[OP_BITDEC as usize]);
    marks.push((a.pc(), "bitdec"));
    {
        opnd(&mut a, T1, 4);
        a.ld(A2, T1, 0);
        if_flag(&mut a, 8, T6, |a| upper_zero(a, T1, T3));
        a.srli(T3, T0, 8);
        a.andi(T3, T3, 0x7f);
        a.addi(T4, IP, 8);
        a.mv(T6, A2);
        a.li(A6, TMP as i64);
        let done = a.label();
        a.beq(T3, ZERO, done);
        let explicit = a.label();
        a.srli(T1, T0, 17);
        a.andi(T1, T1, 1);
        a.beq(T1, ZERO, explicit);
        {
            a.lwu(T1, T4, 0);
            a.add(T1, T1, OB);
            a.addi(T4, T4, 4);
            let top = a.here();
            a.andi(T5, T6, 1);
            a.srli(T6, T6, 1);
            a.sd(T5, T1, 0);
            for z in 1..4 {
                a.sd(ZERO, T1, 8 * z);
            }
            a.addi(T1, T1, 32);
            a.addi(T3, T3, -1);
            a.bnez_back(T3, top);
            a.jal(ZERO, done);
        }
        a.bind(explicit);
        let top = a.here();
        let unread = a.label();
        a.andi(T5, T6, 1);
        a.srli(T6, T6, 1);
        a.lwu(T1, T4, 0);
        a.add(T1, T1, OB);
        a.beq(T1, A6, unread);
        a.sd(T5, T1, 0);
        for z in 1..4 {
            a.sd(ZERO, T1, 8 * z);
        }
        a.bind(unread);
        a.addi(T4, T4, 4);
        a.addi(T3, T3, -1);
        a.bnez_back(T3, top);
        a.bind(done);
        let no_halves = a.label();
        a.srli(T3, T0, 15);
        a.andi(T3, T3, 1);
        a.beq(T3, ZERO, no_halves);
        a.mv(A3, T4);
        for h in 0..2 {
            if h == 0 {
                a.srli(A1, A2, 32);
            } else {
                a.slli(A1, A2, 32);
                a.srli(A1, A1, 32);
            }
            a.call(T6, rt.bswap);
            a.lwu(T1, A3, 4 * h);
            a.add(T1, T1, OB);
            a.sd(A1, T1, 0);
            for z in 1..4 {
                a.sd(ZERO, T1, 8 * z);
            }
        }
        a.addi(T4, A3, 8);
        a.bind(no_halves);
        a.mv(IP, T4);
        a.jal(ZERO, next);
    }

    // BASE and EXT: a, b, (c for op 4), out; the routine's address from the
    // data after the jump table.
    for (code, n_rt) in [(OP_BASE, 5usize), (OP_EXT, 6usize)] {
        a.bind(handlers[code as usize]);
        marks.push((a.pc(), if code == OP_BASE { "base" } else { "ext" }));
        let tbl = if code == OP_BASE { N_OPS } else { N_OPS + 5 };
        let three = a.label();
        let call = a.label();
        a.andi(T6, T0, 0x700);
        a.srli(T6, T6, 8);
        a.li(T5, 4);
        a.beq(T6, T5, three);
        for (j, r) in [A1, A2].into_iter().enumerate() {
            opnd(&mut a, r, 4 + 4 * j as i64);
        }
        opnd(&mut a, A4, 12);
        a.addi(IP, IP, 16);
        a.jal(ZERO, call);
        a.bind(three);
        for (j, r) in [A1, A2, A3].into_iter().enumerate() {
            opnd(&mut a, r, 4 + 4 * j as i64);
        }
        opnd(&mut a, A4, 16);
        a.addi(IP, IP, 20);
        a.bind(call);
        for (j, r) in [A1, A2, A3].into_iter().enumerate() {
            let is_ext = code == OP_EXT;
            if_flag(&mut a, 4 + j as u32, T5, |a| {
                if is_ext && j == 1 {
                    // `mul_base`'s scalar: lanes 1..4; any other: lane 3.
                    let (lane3, done) = (a.label(), a.label());
                    a.li(T1, 5);
                    a.bne(T6, T1, lane3);
                    upper_zero(a, r, T2);
                    a.jal(ZERO, done);
                    a.bind(lane3);
                    a.ld(T2, r, 24);
                    a.or(ERR, ERR, T2);
                    a.bind(done);
                } else if is_ext {
                    a.ld(T2, r, 24);
                    a.or(ERR, ERR, T2);
                } else {
                    upper_zero(a, r, T2);
                }
            });
        }
        a.slli(T6, T6, 3);
        a.addi(T6, T6, 8 * tbl as i64);
        a.add(T6, T6, JT);
        a.ld(T6, T6, 0);
        a.jalr(RA, T6, 0);
        let first_zero = if code == OP_BASE { 1 } else { 3 };
        for z in first_zero..4 {
            a.sd(ZERO, A4, 8 * z);
        }
        a.jal(ZERO, next);
        let _ = n_rt;
    }

    // PUBLIC: src, its four lanes published.
    a.bind(handlers[OP_PUBLIC as usize]);
    marks.push((a.pc(), "public"));
    opnd(&mut a, T1, 4);
    copy_cell(&mut a, T1, 0, PUBC, 0, [T2, T3, T4, T5]);
    a.addi(PUBC, PUBC, 32);
    a.addi(IP, IP, 8);
    a.jal(ZERO, next);

    // Epilogue: BACK's types, the binding, the output.
    a.bind(epilogue);
    marks.push((a.pc(), "epilogue"));
    {
        // Type runs: (type, count); 1: lane 3 zero, 2: lanes 1..4 zero,
        // 3: refused.
        a.li(T1, TYPES_AT as i64);
        a.ld(T1, T1, 0);
        a.mv(T2, BACK_B);
        a.lwu(T3, SB, 32);
        a.srli(T3, T3, 1);
        let done = a.label();
        a.beq(T3, ZERO, done);
        let run = a.here();
        a.lwu(T4, T1, 0);
        a.lwu(T5, T1, 4);
        a.addi(T1, T1, 8);
        let (t1, t2, t3, next_run) = (a.label(), a.label(), a.label(), a.label());
        a.li(T6, 1);
        a.beq(T4, T6, t1);
        a.li(T6, 2);
        a.beq(T4, T6, t2);
        a.bne(T4, ZERO, t3);
        a.slli(T6, T5, 5);
        a.add(T2, T2, T6);
        a.jal(ZERO, next_run);
        a.bind(t1);
        {
            let top = a.here();
            a.ld(A1, T2, 24);
            a.or(ERR, ERR, A1);
            a.addi(T2, T2, 32);
            a.addi(T5, T5, -1);
            a.bnez_back(T5, top);
            a.jal(ZERO, next_run);
        }
        a.bind(t2);
        {
            let top = a.here();
            upper_zero(&mut a, T2, A1);
            a.addi(T2, T2, 32);
            a.addi(T5, T5, -1);
            a.bnez_back(T5, top);
            a.jal(ZERO, next_run);
        }
        a.bind(t3);
        a.ori(ERR, ERR, 1);
        a.slli(T6, T5, 5);
        a.add(T2, T2, T6);
        a.bind(next_run);
        a.addi(T3, T3, -1);
        a.bnez_back(T3, run);
        a.bind(done);
    }
    // c_bin: IN's cells then BACK's, four a permutation, the counts in
    // lanes 12 and 13.
    for l in 0..16 {
        a.sd(ZERO, A0, 8 * l);
    }
    a.sd(N_IN, A0, 96);
    a.sd(N_BACK, A0, 104);
    for (base, count) in [(IN_B, N_IN), (BACK_B, N_BACK)] {
        let (full, rest, done) = (a.label(), a.label(), a.label());
        a.mv(T1, base);
        a.mv(T2, count);
        a.li(T6, 4);
        a.bind(full);
        a.bltu(T2, T6, rest);
        for c in 0..4 {
            for l in 0..3 {
                a.ld(T5, T1, 32 * c + 8 * l);
                a.sd(T5, A0, 8 * (3 * c + l));
            }
        }
        a.ecall();
        a.addi(T1, T1, 128);
        a.addi(T2, T2, -4);
        a.jal(ZERO, full);
        a.bind(rest);
        a.beq(T2, ZERO, done);
        for l in 0..12 {
            a.sd(ZERO, A0, 8 * l);
        }
        a.mv(T4, A0);
        let cell = a.here();
        for l in 0..3 {
            a.ld(T5, T1, 8 * l);
            a.sd(T5, T4, 8 * l);
        }
        a.addi(T1, T1, 32);
        a.addi(T4, T4, 24);
        a.addi(T2, T2, -1);
        a.bnez_back(T2, cell);
        a.ecall();
        a.bind(done);
    }
    a.li(T1, BIND as i64);
    copy_cell(&mut a, A0, 0, T1, 0, [T2, T3, T4, T5]);
    for l in 0..3 {
        a.sd(ZERO, T1, 64 + 8 * l);
    }
    a.slli(T2, N_BACK, 5);
    a.add(T2, T2, BACK_B);
    for l in 0..3 {
        a.ld(T3, T2, 8 * l);
        a.sd(T3, T1, 88 + 8 * l);
    }
    a.addi(A0, T1, 64);
    a.li(
        A7,
        executor::vm::instruction::execution::FP3MAC_SYSCALL_NUMBER as i64,
    );
    for (base, count) in [(IN_B, N_IN), (BACK_B, N_BACK)] {
        let (eight, single, done) = (a.label(), a.label(), a.label());
        a.mv(A1, base);
        a.mv(T2, count);
        a.li(T6, 8);
        a.bind(eight);
        a.bltu(T2, T6, single);
        for _ in 0..8 {
            a.ecall();
            a.addi(A1, A1, 32);
        }
        a.addi(T2, T2, -8);
        a.jal(ZERO, eight);
        a.bind(single);
        a.beq(T2, ZERO, done);
        a.ecall();
        a.addi(A1, A1, 32);
        a.addi(T2, T2, -1);
        a.jal(ZERO, single);
        a.bind(done);
    }
    a.li(T1, BIND as i64);
    for (i, off) in [0, 8, 16, 24, 88, 96, 104, 64, 72, 80]
        .into_iter()
        .enumerate()
    {
        a.ld(T2, T1, off);
        a.sd(T2, PUBC, 8 * i as i64);
    }
    a.addi(PUBC, PUBC, 80);
    a.bne(ERR, ZERO, fail);
    a.beq(OK, ZERO, fail);
    a.li(T3, PUB as i64);
    for (k, off) in [16i64, 20, 8, 12].into_iter().enumerate() {
        a.lwu(T1, SB, off);
        a.sw(T1, T3, 32 + 4 * k as i64);
    }
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

    // The jump table, then the field routines' addresses.
    let at = |a: &Asm, l: Label| CODE_BASE + 4 * (a.pc() as i64 + a.distance(l)) as u64;
    let mut data = Vec::new();
    for &h in &handlers {
        data.extend(at(&a, h).to_le_bytes());
    }
    for &l in rt.base.iter().chain(&rt.ext) {
        data.extend(at(&a, l).to_le_bytes());
    }
    let code = a.finish();
    assert!(
        CODE_BASE + 4 * code.len() as u64 <= DATA_BASE,
        "the code fits"
    );
    let mut marks: Vec<(u64, &'static str)> = marks
        .into_iter()
        .map(|(pc, w)| (CODE_BASE + 4 * pc as u64, w))
        .collect();
    marks.sort_by_key(|m| m.0);
    (elf(CODE_BASE, &code, DATA_BASE, &data), marks)
}

#[cfg(test)]
pub(crate) mod reference {
    //! The guest's semantics on the host: runs a schedule over its input.

    use super::*;

    pub struct Run {
        pub back: Vec<[u64; 4]>,
        pub publics: Vec<[u64; 4]>,
        pub err: bool,
        pub ops: usize,
    }

    struct Mem {
        input: Vec<u8>,
        other: HashMap<u64, u64>,
    }

    impl Mem {
        fn ld(&self, a: u64) -> u64 {
            assert_eq!(a % 8, 0, "aligned load at {a:#x}");
            if a >= INPUT && ((a - INPUT) as usize) + 8 <= self.input.len() {
                let o = (a - INPUT) as usize;
                u64::from_le_bytes(self.input[o..o + 8].try_into().unwrap())
            } else {
                *self.other.get(&a).unwrap_or(&0)
            }
        }
        fn sd(&mut self, a: u64, v: u64) {
            assert_eq!(a % 8, 0, "aligned store at {a:#x}");
            if a >= INPUT && ((a - INPUT) as usize) + 8 <= self.input.len() {
                let o = (a - INPUT) as usize;
                self.input[o..o + 8].copy_from_slice(&v.to_le_bytes());
            } else {
                self.other.insert(a, v);
            }
        }
        fn cell(&self, a: u64) -> [u64; 4] {
            core::array::from_fn(|l| self.ld(a + 8 * l as u64))
        }
        fn put(&mut self, a: u64, v: [u64; 4]) {
            for (l, x) in v.iter().enumerate() {
                self.sd(a + 8 * l as u64, *x);
            }
        }
    }

    fn fe(x: u64) -> crate::tables::types::FE {
        crate::tables::types::FE::from(x)
    }

    fn permute(st: [u64; 16]) -> [u64; 16] {
        use crypto::hash::poseidon1_w16::{Fp, permute};
        let o = permute(core::array::from_fn(|i| Fp::from(st[i])));
        core::array::from_fn(|i| o[i].canonical())
    }

    fn fadd(a: u64, b: u64) -> u64 {
        (fe(a) + fe(b)).canonical()
    }
    fn fsub(a: u64, b: u64) -> u64 {
        // The guest's: `a − b`, plus p when it borrows.
        let (d, borrow) = a.overflowing_sub(b);
        if borrow { d.wrapping_add(P) } else { d }
    }
    fn fmul(a: u64, b: u64) -> u64 {
        (fe(a) * fe(b)).canonical()
    }

    pub fn run(s: &Schedule, input: Vec<u8>) -> Run {
        let mut m = Mem {
            input,
            other: HashMap::new(),
        };
        let w = &s.words;
        let mut err = false;
        let mut ok = true;
        // Header checks.
        let n_canon = w[6] as usize;
        let n_base = w[7] as usize;
        let mut at = 10 + 8 * w[9] as usize;
        for _ in 0..n_canon {
            let (a, cnt) = (w[at] as u64 + OPND_BASE, w[at + 1] as u64);
            for i in 0..cnt {
                ok &= m.ld(a + 8 * i) < P;
            }
            at += 2;
        }
        for _ in 0..n_base {
            let a = w[at] as u64 + OPND_BASE;
            err |= m.ld(a + 8) != 0 || m.ld(a + 16) != 0;
            at += 1;
        }
        for i in 0..s.n_in as u64 {
            err |= m.ld(s.layout.in_base + 32 * i + 24) != 0;
        }
        let n_types = w[8] as usize;
        let types_at = at;
        at += n_types;
        if at % 2 == 1 {
            at += 1;
        }
        let mut publics = Vec::new();
        let mut ops = 0;
        struct Frame {
            code: Vec<u32>,
            ip: usize,
            reps: u32,
            pairs: Vec<(u32, u32)>,
            body: Vec<u32>,
        }
        let mut frames = vec![Frame {
            code: w.clone(),
            ip: at,
            reps: 1,
            pairs: Vec::new(),
            body: Vec::new(),
        }];
        loop {
            let f = frames.last().unwrap();
            let ip = f.ip;
            let word = f.code[ip];
            let (op, fl) = (word & 0xff, word >> 8);
            let opv: Vec<u64> = f.code[ip + 1..(ip + 80).min(f.code.len())]
                .iter()
                .map(|&x| x as u64)
                .collect();
            let a = |j: usize| opv[j] + OPND_BASE;
            let mut next = ip + 1;
            ops += 1;
            let check_upper =
                |m: &Mem, addr: u64| m.ld(addr + 8) | m.ld(addr + 16) | m.ld(addr + 24) != 0;
            match op {
                OP_HALT => break,
                OP_END => {
                    let top = frames.last_mut().unwrap();
                    top.reps -= 1;
                    if top.reps == 0 {
                        frames.pop();
                        continue;
                    }
                    for &(p, d) in &top.pairs {
                        let x = &mut top.body[p as usize];
                        *x = (*x as i64 + d as i64 - BIAS as i64) as u32;
                    }
                    top.code = top.body.clone();
                    top.ip = 0;
                    continue;
                }
                OP_LOOP => {
                    let code = &frames.last().unwrap().code;
                    let reps = code[ip + 1];
                    let bw = code[ip + 2] as usize;
                    let nnz = code[ip + 3] as usize;
                    let pairs: Vec<(u32, u32)> = (0..nnz)
                        .map(|i| (code[ip + 4 + 2 * i], code[ip + 5 + 2 * i]))
                        .collect();
                    let body_at = ip + 4 + 2 * nnz;
                    let body = code[body_at..body_at + bw].to_vec();
                    frames.last_mut().unwrap().ip = body_at + bw;
                    frames.push(Frame {
                        code: body.clone(),
                        ip: 0,
                        reps,
                        pairs,
                        body,
                    });
                    continue;
                }
                OP_HASH => {
                    let mut st = [0u64; 16];
                    let mut at = 0;
                    let lanes =
                        |m: &Mem, at: &mut usize, st: &mut [u64; 16], j: usize, err: &mut bool| {
                            for q in 0..4 {
                                let w = opv[*at];
                                *at += 1;
                                let addr = (w & !7) + OPND_BASE;
                                st[4 * j + q] = m.ld(addr);
                                if w & 1 != 0 {
                                    *err |= check_upper(m, addr);
                                }
                            }
                        };
                    if fl & 0x300 != 0 {
                        let x = a(at);
                        at += 1;
                        for (i, lane) in st.iter_mut().enumerate().take(12) {
                            let addr = if fl & 0x100 != 0 {
                                x + 32 * (i / 3) as u64 + 8 * (i % 3) as u64
                            } else {
                                x + 32 * i as u64
                            };
                            *lane = m.ld(addr);
                            if fl & 0x400 != 0 {
                                err |= check_upper(&m, addr);
                            }
                        }
                        if fl & 8 != 0 {
                            lanes(&m, &mut at, &mut st, 3, &mut err);
                        } else {
                            let c = m.cell(a(at));
                            at += 1;
                            st[12..].copy_from_slice(&c);
                        }
                    } else {
                        for j in 0..4 {
                            if fl >> j & 1 != 0 {
                                lanes(&m, &mut at, &mut st, j, &mut err);
                            } else {
                                let c = m.cell(a(at));
                                at += 1;
                                st[4 * j..4 * j + 4].copy_from_slice(&c);
                            }
                        }
                    }
                    let o = permute(st);
                    for j in 0..4 {
                        if fl >> (4 + j) & 1 != 0 {
                            m.put(a(at), core::array::from_fn(|l| o[4 * j + l]));
                            at += 1;
                        }
                    }
                    next += at;
                }
                OP_WALK4 => {
                    let b0 = m.ld(a(0));
                    let b1 = m.ld(a(1));
                    if fl & 1 != 0 {
                        err |= check_upper(&m, a(0));
                    }
                    if fl & 2 != 0 {
                        err |= check_upper(&m, a(1));
                    }
                    ok &= b0 < 2 && b1 < 2;
                    let p = ((b0 + 2 * b1) & 3) as usize;
                    let mut slots = [[0u64; 4]; 4];
                    for q in 0..3 {
                        slots[q + (p <= q) as usize] = m.cell(a(3 + q));
                    }
                    slots[p] = m.cell(a(2));
                    let st: [u64; 16] = core::array::from_fn(|i| slots[i / 4][i % 4]);
                    let o = permute(st);
                    let mut at = 6;
                    for j in 0..4 {
                        if fl >> (2 + j) & 1 != 0 {
                            m.put(a(at), core::array::from_fn(|l| o[4 * j + l]));
                            at += 1;
                        }
                    }
                    next += at;
                }
                OP_SELECT => {
                    let bit = m.ld(a(0));
                    if fl & 1 != 0 {
                        err |= check_upper(&m, a(0));
                    }
                    ok &= bit < 2;
                    let (l, r) = (m.cell(a(1)), m.cell(a(2)));
                    let (x, y) = if bit != 0 { (r, l) } else { (l, r) };
                    let mut at = 3;
                    for (j, v) in [x, y].into_iter().enumerate() {
                        if fl >> (1 + j) & 1 != 0 {
                            m.put(a(at), v);
                            at += 1;
                        }
                    }
                    next += at;
                }
                OP_MUX => {
                    let k = (fl & 0xf) as usize;
                    let mut idx = 0usize;
                    let nb = if fl & 0x1000 != 0 {
                        let x = m.ld(a(0)) >> ((fl >> 13) & 63);
                        idx = (x & ((1 << k) - 1)) as usize;
                        1
                    } else {
                        for t in 0..k {
                            let b = m.ld(a(t));
                            if fl >> (4 + t) & 1 != 0 {
                                err |= check_upper(&m, a(t));
                            }
                            ok &= b < 2;
                            idx |= ((b != 0) as usize) << t;
                        }
                        k
                    };
                    let nl = if fl & (1 << 19) != 0 { 1 } else { 1 << k };
                    let leaf = if nl == 1 {
                        a(nb) + 32 * idx as u64
                    } else {
                        a(nb + idx)
                    };
                    let v = m.cell(leaf);
                    m.put(a(nb + nl), v);
                    next += nb + nl + 1;
                }
                OP_CHAIN => {
                    let n = ((fl >> 11) & 0xff) as usize;
                    let mut st = [0u64; 16];
                    st[12..].copy_from_slice(&m.cell(a(1)));
                    let mut x = a(0);
                    for i in 0..n {
                        for (f, lane) in st.iter_mut().enumerate().take(12) {
                            let addr = if fl & 0x100 != 0 {
                                x + 32 * (f / 3) as u64 + 8 * (f % 3) as u64
                            } else {
                                x + 32 * f as u64
                            };
                            *lane = m.ld(addr);
                            if fl & 0x400 != 0 {
                                err |= check_upper(&m, addr);
                            }
                        }
                        x += if fl & 0x100 != 0 { 128 } else { 384 };
                        st = permute(st);
                        if i + 1 < n {
                            let d: [u64; 4] = st[..4].try_into().unwrap();
                            st[12..].copy_from_slice(&d);
                        }
                    }
                    let mut at = 2;
                    for j in 0..4 {
                        if fl >> (4 + j) & 1 != 0 {
                            m.put(a(at), core::array::from_fn(|l| st[4 * j + l]));
                            at += 1;
                        }
                    }
                    next += at;
                }
                OP_PATH => {
                    let n = ((fl >> 8) & 0xff) as usize;
                    let mut cur = m.cell(a(0));
                    let sib = a(1);
                    let mut st = [0u64; 16];
                    let src = fl & 0x10000 != 0;
                    let mut x = if src {
                        m.ld(a(2)) >> ((fl >> 17) & 63)
                    } else {
                        0
                    };
                    for i in 0..n {
                        let p = if src {
                            let p = (x & 3) as usize;
                            x >>= 2;
                            p
                        } else {
                            let (ba, bb) = (a(2 + 2 * i), a(3 + 2 * i));
                            let (b0, b1) = (m.ld(ba), m.ld(bb));
                            if fl & 1 != 0 {
                                err |= check_upper(&m, ba);
                            }
                            if fl & 2 != 0 {
                                err |= check_upper(&m, bb);
                            }
                            ok &= b0 < 2 && b1 < 2;
                            ((b0 + 2 * b1) & 3) as usize
                        };
                        let mut slots = [[0u64; 4]; 4];
                        for q in 0..3 {
                            slots[q + (p <= q) as usize] =
                                m.cell(sib + 96 * i as u64 + 32 * q as u64);
                        }
                        slots[p] = cur;
                        st = permute(core::array::from_fn(|f| slots[f / 4][f % 4]));
                        cur = st[..4].try_into().unwrap();
                    }
                    let mut at = if src { 3 } else { 2 + 2 * n };
                    for j in 0..4 {
                        if fl >> (2 + j) & 1 != 0 {
                            m.put(a(at), core::array::from_fn(|l| st[4 * j + l]));
                            at += 1;
                        }
                    }
                    next += at;
                }
                OP_ASSERT4 => {
                    for l in 0..4u64 {
                        err |= fsub(m.ld(a(0) + 8 * l), m.ld(a(1) + 8 * l)) != 0;
                    }
                    next += 2;
                }
                OP_ASSERT => {
                    let (x, y) = (m.ld(a(0)), m.ld(a(1)));
                    if fl & 1 != 0 {
                        err |= check_upper(&m, a(0));
                    }
                    if fl & 2 != 0 {
                        err |= check_upper(&m, a(1));
                    }
                    err |= fsub(x, y) != 0;
                    next += 2;
                }
                OP_PACK => {
                    let mut v = [0u64; 4];
                    for (j, x) in v.iter_mut().enumerate() {
                        *x = m.ld(a(j));
                        if fl >> j & 1 != 0 {
                            err |= check_upper(&m, a(j));
                        }
                    }
                    m.put(a(4), v);
                    next += 5;
                }
                OP_UNPACK => {
                    let c = m.cell(a(0));
                    let mut at = 1;
                    for (l, x) in c.iter().enumerate() {
                        if fl >> l & 1 != 0 {
                            m.put(a(at), [*x, 0, 0, 0]);
                            at += 1;
                        }
                    }
                    next += at;
                }
                OP_BITDEC => {
                    let nb = (fl & 0x7f) as usize;
                    let halves = fl >> 7 & 1 != 0;
                    let x = m.ld(a(0));
                    if fl >> 8 & 1 != 0 {
                        err |= check_upper(&m, a(0));
                    }
                    let run = fl >> 9 & 1 != 0;
                    for b in 0..nb {
                        let at = if run { a(1) + 32 * b as u64 } else { a(1 + b) };
                        m.put(at, [x >> b & 1, 0, 0, 0]);
                    }
                    let used = if run { 1 } else { nb };
                    next += 1 + used;
                    if halves {
                        let hi = ((x >> 32) as u32).swap_bytes() as u64;
                        let lo = (x as u32).swap_bytes() as u64;
                        m.put(a(1 + used), [hi, 0, 0, 0]);
                        m.put(a(2 + used), [lo, 0, 0, 0]);
                        next += 2;
                    }
                }
                OP_BASE => {
                    let bop = fl & 7;
                    let checks = fl >> 4;
                    let nin = if bop == 4 { 3 } else { 2 };
                    for j in 0..nin {
                        if checks >> j & 1 != 0 {
                            err |= check_upper(&m, a(j));
                        }
                    }
                    let (x, y) = (m.ld(a(0)), m.ld(a(1)));
                    let r = match bop {
                        0 => fadd(x, y),
                        1 => fsub(x, y),
                        2 => fmul(x, y),
                        3 if y == 0 => {
                            err |= x != 0;
                            1
                        }
                        3 => (fe(x) * fe(y).inv().unwrap()).canonical(),
                        _ => fadd(fmul(x, y), m.ld(a(2))),
                    };
                    m.put(a(nin), [r, 0, 0, 0]);
                    next += nin + 1;
                }
                OP_EXT => {
                    let eop = fl & 7;
                    let checks = fl >> 4;
                    let nin = if eop == 4 { 3 } else { 2 };
                    for j in 0..nin {
                        if checks >> j & 1 != 0 {
                            if j == 1 && eop == 5 {
                                err |= check_upper(&m, a(j));
                            } else {
                                err |= m.ld(a(j) + 24) != 0;
                            }
                        }
                    }
                    let ext = |m: &Mem, addr: u64| {
                        crate::tables::types::FEE::new(core::array::from_fn(|l| {
                            fe(m.ld(addr + 8 * l as u64))
                        }))
                    };
                    let (x, y) = (ext(&m, a(0)), ext(&m, a(1)));
                    let r = match eop {
                        0 => x + y,
                        1 => x - y,
                        2 => x * y,
                        3 if y == crate::tables::types::FEE::zero() => {
                            err |= x != crate::tables::types::FEE::zero();
                            crate::tables::types::FEE::one()
                        }
                        3 => x * y.inv().unwrap(),
                        4 => x * y + ext(&m, a(2)),
                        _ => x * crate::tables::types::FEE::new([fe(m.ld(a(1))), fe(0), fe(0)]),
                    };
                    let v = r.value();
                    m.put(
                        a(nin),
                        [v[0].canonical(), v[1].canonical(), v[2].canonical(), 0],
                    );
                    next += nin + 1;
                }
                OP_PUBLIC => {
                    publics.push(m.cell(a(0)));
                    next += 1;
                }
                _ => panic!("op {op}"),
            }
            frames.last_mut().unwrap().ip = next;
        }
        let mut back = Vec::new();
        let mut types = Vec::new();
        for r in 0..n_types / 2 {
            let (t, cnt) = (w[types_at + 2 * r], w[types_at + 2 * r + 1]);
            types.extend(std::iter::repeat_n(t, cnt as usize));
        }
        for (i, &t) in types.iter().enumerate().take(s.n_back) {
            let c = m.cell(s.layout.back_base + 32 * i as u64);
            if t >= 1 {
                err |= c[3] != 0;
            }
            if t >= 2 {
                err |= c[1] != 0 || c[2] != 0;
            }
            if t == 3 {
                err = true;
            }
            back.push(c);
        }
        Run {
            back,
            publics,
            err: err || !ok,
            ops,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tables::types::{FE, FEE};

    /// A small hash-half program over every op the guest runs but the walk:
    /// word and shared hints, unpacks, packs, a two-permutation chain, a bit
    /// decomposition of a squeezed felt, a two-level mux by its bits, an
    /// assert, a published word, and three cells crossing back.
    fn program() -> Vec<u32> {
        let mut w = vec![3u32, 45, 0, 3, 0];
        w.extend([33 | 2 << 30, 41, 26]);
        let mut n = 0;
        let mut op = |w: &mut Vec<u32>, words: &[u32]| {
            w.extend(words);
            n += 1;
        };
        op(&mut w, &[8, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        for c in 1..=8 {
            op(&mut w, &[7, c, 0]);
        }
        op(&mut w, &[7, 9, 3]);
        op(&mut w, &[7, 10, 3]);
        op(&mut w, &[4, 9, 11, 12, 13, 14, 0]);
        op(&mut w, &[4, 10, 15, 16, 17, 18, 0]);
        op(&mut w, &[3, 11, 12, 13, 15, 19]);
        op(&mut w, &[3, 16, 17, 0, 0, 20]);
        op(&mut w, &[1, 4, 19, 20, 1, 0, 21, 22, 23, 24]);
        op(&mut w, &[1, 4, 2, 3, 4, 21, 25, 26, 27, 28]);
        op(&mut w, &[4, 25, 29, 30, 31, 32, 0]);
        op(&mut w, &[5, 29, 4, 33, 34, 35, 36, 0]);
        op(&mut w, &[2, 33, 5, 6, 37, 38]);
        op(&mut w, &[2, 33, 7, 8, 39, 40]);
        op(&mut w, &[2, 34, 37, 39, 41, 42]);
        op(&mut w, &[9, 1, 30, 30, 43]);
        op(&mut w, &[9, 3, 43, 0, 44]);
        op(&mut w, &[11, 41]);
        w[4] = n;
        w
    }

    fn felt(seed: u64) -> u64 {
        let x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).rotate_left(17);
        x % P
    }

    #[test]
    fn the_guest_runs_a_program_as_the_reference_does() {
        let words = program();
        let sched = compile(&words).expect("the program compiles");
        let hints: Vec<[u64; 4]> = (0..10u64)
            .map(|h| core::array::from_fn(|l| felt(4 * h + l as u64)))
            .collect();
        let in_cells: Vec<[u64; 3]> = hints[8..].iter().map(|h| [h[0], h[1], h[2]]).collect();
        let alpha = [felt(100), felt(101), felt(102)];
        let input = input(&sched, &hints, &in_cells, alpha);

        let r = reference::run(&sched, input.clone());
        assert!(!r.err, "the reference's checks pass");
        // The second permutation's lane-word 1, independently.
        let p = |st: [u64; 16]| -> [u64; 16] {
            use crypto::hash::poseidon1_w16::{Fp, permute};
            let o = permute(st.map(Fp::from));
            core::array::from_fn(|i| o[i].canonical())
        };
        let (a, b) = (in_cells[0], in_cells[1]);
        let mut st = [0u64; 16];
        st[..4].copy_from_slice(&[a[0], a[1], a[2], b[0]]);
        st[4..8].copy_from_slice(&[b[1], b[2], 0, 0]);
        st[8..12].copy_from_slice(&hints[0]);
        let first = p(st);
        let mut st = [0u64; 16];
        st[..4].copy_from_slice(&hints[1]);
        st[4..8].copy_from_slice(&hints[2]);
        st[8..12].copy_from_slice(&hints[3]);
        st[12..].copy_from_slice(&first[..4]);
        let second = p(st);
        assert_eq!(r.back[2], [second[4], second[5], second[6], second[7]]);
        let x = second[0];
        let (b0, b1) = (x & 1, x >> 1 & 1);
        assert_eq!(r.back[0], [b0, 0, 0, 0], "bit 0 crosses back");
        let leaf = hints[4 + (b0 + 2 * b1) as usize];
        assert_eq!(r.back[1], leaf, "the mux's leaf crosses back");
        assert_eq!(r.publics, vec![leaf], "the mux's leaf is published");

        let back: Vec<[u64; 3]> = r.back.iter().map(|c| [c[0], c[1], c[2]]).collect();
        let c_bin = super::super::cksum::crossing_digest(&in_cells, &back);
        let fee = |c: &[u64; 3]| FEE::new(c.map(FE::from));
        let values: Vec<FEE> = in_cells.iter().chain(&back).map(fee).collect();
        let sum = super::super::cksum::horner(&values, fee(&alpha));
        let s = sum.value().map(|x| x.canonical());
        let expected = expected_output(&sched, &r.publics, c_bin, alpha, s);

        let program = executor::elf::Elf::load(guest()).expect("the guest ELF loads");
        let mut vm =
            executor::vm::execution::Executor::new(&program, input).expect("the guest starts");
        while vm.resume().expect("the guest runs").is_some() {}
        let got_back = back_cells(&sched, |a| {
            vm.memory().load_doubleword(a).expect("BACK reads")
        });
        assert_eq!(got_back, back, "the guest's BACK");
        let out = vm.finish().expect("the guest finishes").memory_values;
        assert_eq!(out, expected, "the guest's output");
    }

    #[test]
    fn a_tampered_schedule_changes_the_published_digest() {
        let sched = compile(&program()).expect("the program compiles");
        let mut words = sched.words.clone();
        let last = words.iter().rposition(|&w| w != 0).unwrap();
        words[last] ^= 1;
        assert_ne!(schedule_digest(&words), schedule_digest(&sched.words));
    }
}
