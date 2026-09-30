//! The Field VM with the LFM's hash-side chips: `LFM_HASH`, `LFM_LANES` and
//! `LFM_BITDEC` proved in the same `multi_prove`, joined to MEM by
//! `FIELD_VM_BRIDGE`.
//!
//! The LFM program splits in two. The field half (constants, ALU, selects,
//! hints, publics) is translated to the Field VM as before; the hash half
//! (hash, pack, unpack, bit decomposition) is compiled as an LFM program of
//! its own, where every cell it reads from the field half is a hint
//! placeholder. The placeholders are not proved: `FIELD_VM_BRIDGE` sends
//! those cells from MEM instead, with the multiplicities the compiler gave
//! the placeholders, and receives the hash half's writes that the field half
//! reads or publishes (the compiler counts one extra read of each).

use stark::lookup::{AirWithBuses, AuxiliaryTraceBuildData, BusInteraction};
use stark::proof::options::ProofOptions;
use stark::trace::TraceTable;

use super::air::FvmPublicInputs;
use super::bridge::{self, BridgeConstraints, BridgeRow};
use super::prove::ExtraAir;
use crate::lfm::builder::{ArenaSchema, LfmProgramSource};
use crate::lfm::chips::{balu, bitdec, const_, hash, keccak, lanes, select, xalu};
use crate::lfm::compiler::{LfmProgram, compile};
use crate::lfm::executor::LfmExecution;
use crate::lfm::hash::HasherKind;
use crate::lfm::instr::{Addr, Instr};
use crate::lfm::layout;
use crate::lfm::registry::LfmArtifacts;
use crate::lfm::word::LfmWord;
use crate::tables::types::{FE, GoldilocksExtension, GoldilocksField};
use crate::tables::{bitwise, keccak_rc, keccak_rnd};
use stark::constraints::builder::{ConstraintSet, EmptyConstraints};
use stark::lookup::NullBoundaryConstraintBuilder;

type F = GoldilocksField;
type E = GoldilocksExtension;

/// Selects move words (Merkle path order), whose fourth lane the Field VM
/// does not hold, so they go with the hash half too.
/// `FVM_HYBRID_ALL` moves everything but publics: the LFM's chips alone,
/// as a control.
pub fn is_hash_side(i: &Instr) -> bool {
    if std::env::var_os("FVM_HYBRID_ALL").is_some() {
        return !matches!(i, Instr::Public { .. });
    }
    matches!(
        i,
        Instr::Hash { .. }
            | Instr::Select { .. }
            | Instr::Pack { .. }
            | Instr::Unpack { .. }
            | Instr::BitDec { .. }
            | Instr::KeccakF(_)
            | Instr::Blake3(_)
    )
}

/// A cell the field half hands to the hash half.
#[derive(Clone, Debug)]
pub struct OutCell {
    pub addr: u64,
    pub is_word: bool,
    pub l3c: FE,
}

pub struct HashSide {
    /// The hash half, placeholders first.
    pub sub: LfmProgram,
    /// The program's arenas, then the placeholders' values.
    pub arenas: Vec<Vec<LfmWord>>,
    /// In placeholder order.
    pub out: Vec<OutCell>,
    /// Hash-half writes the field half reads or publishes.
    pub back: Vec<u64>,
    /// Whether [`renumber`] moved the hash half onto MEM's addresses, which
    /// retires BRIDGE: MEM itself sends and receives the crossing cells.
    pub merged: bool,
}

impl HashSide {
    /// The LFM addresses the Field VM's memory must keep.
    pub fn crossing(&self) -> Vec<u64> {
        self.out
            .iter()
            .map(|c| c.addr)
            .chain(self.back.iter().copied())
            .collect()
    }
}

/// With `FVM_HYBRID_HINTS`, hints the hash half reads stay LFM hints
/// (`LFM_HINT`) instead of crossing from MEM.
pub fn split(program: &LfmProgram, exec: &LfmExecution, arenas: &[Vec<LfmWord>]) -> HashSide {
    let n = program.num_addrs as usize;
    let mut hash_reads = vec![0u64; n];
    let mut buf = Vec::new();
    for i in program.instrs.iter().filter(|i| is_hash_side(i)) {
        buf.clear();
        i.reads_into(&mut buf);
        for a in &buf {
            hash_reads[a.0 as usize] += 1;
        }
    }
    let move_hints = std::env::var_os("FVM_HYBRID_HINTS").is_some();
    let moves = |i: &Instr| {
        is_hash_side(i)
            || (move_hints
                && match i {
                    Instr::Hint { out, .. } => hash_reads[out.0 as usize] > 0,
                    Instr::Const { out, value, .. } => {
                        hash_reads[out.0 as usize] > 0 && value[3] != FE::zero()
                    }
                    _ => false,
                })
    };
    let mut hash_written = vec![false; n];
    let mut field_writer: Vec<Option<&Instr>> = vec![None; n];
    let mut field_read = vec![false; n];
    for i in &program.instrs {
        let hs = moves(i);
        buf.clear();
        i.writes_into(&mut buf);
        for a in &buf {
            if hs {
                hash_written[a.0 as usize] = true;
            } else {
                field_writer[a.0 as usize] = Some(i);
            }
        }
        if !hs {
            buf.clear();
            i.reads_into(&mut buf);
            for a in &buf {
                field_read[a.0 as usize] = true;
            }
        }
    }
    let word = |a: u64| exec.memory.get(Addr(a)).expect("an executed cell");
    let mut out = Vec::new();
    let mut back = Vec::new();
    for a in 0..n {
        if hash_written[a] {
            if field_read[a] {
                back.push(a as u64);
            }
        } else if hash_reads[a] > 0 {
            let (is_word, l3c) = match field_writer[a] {
                Some(Instr::Hint { .. }) => (true, FE::zero()),
                Some(Instr::Const { value, .. }) => (false, value[3]),
                Some(_) => (false, FE::zero()),
                None => panic!("hash half reads unwritten cell {a}"),
            };
            out.push(OutCell {
                addr: a as u64,
                is_word,
                l3c,
            });
        }
    }
    let mut read_counts = hash_reads.clone();
    for &a in &back {
        read_counts[a as usize] += 1;
    }
    let placeholder_arena = arenas.len() as u32;
    let mut instrs: Vec<Instr> = out
        .iter()
        .enumerate()
        .map(|(k, c)| Instr::Hint {
            arena: placeholder_arena,
            index: k as u32,
            out: Addr(c.addr),
            mult: 0,
        })
        .collect();
    instrs.extend(program.instrs.iter().filter(|i| moves(i)).cloned());
    let mut arenas = arenas.to_vec();
    arenas.push(out.iter().map(|c| word(c.addr)).collect());
    let mut lens = program.arena_schema.lens.clone();
    lens.push(out.len() as u32);
    let sub = compile(LfmProgramSource {
        instrs,
        num_addrs: program.num_addrs,
        read_counts,
        arena_schema: ArenaSchema { lens },
        public_len: 0,
    });
    HashSide {
        sub,
        arenas,
        out,
        back,
        merged: false,
    }
}

fn remap(i: &mut Instr, f: &dyn Fn(Addr) -> Addr) {
    let m = |a: &mut Addr| *a = f(*a);
    match i {
        Instr::Const { out, .. } | Instr::Hint { out, .. } => m(out),
        Instr::BaseAlu { out, a, b, c, .. } | Instr::ExtAlu { out, a, b, c, .. } => {
            [out, a, b, c].into_iter().for_each(m)
        }
        Instr::Select {
            bit,
            out_l,
            out_r,
            in_l,
            in_r,
            ..
        } => [bit, out_l, out_r, in_l, in_r].into_iter().for_each(m),
        Instr::BitDec {
            input,
            bits,
            halves,
        } => {
            m(input);
            bits.iter_mut().for_each(|(a, _)| m(a));
            if let Some(hs) = halves {
                hs.iter_mut().for_each(|(a, _)| m(a));
            }
        }
        Instr::Hash { ins, outs, .. } => ins.iter_mut().chain(outs.iter_mut()).for_each(m),
        Instr::Pack { lanes, out, .. } => {
            lanes.iter_mut().for_each(m);
            m(out)
        }
        Instr::Unpack { input, outs, .. } => {
            m(input);
            outs.iter_mut().for_each(m)
        }
        Instr::KeccakF(k) => {
            k.ins
                .iter_mut()
                .chain(k.block.iter_mut())
                .chain(k.outs.iter_mut())
                .for_each(m);
            if let Some(rev) = &mut k.rev {
                rev.outs.iter_mut().for_each(m);
            }
        }
        Instr::Blake3(k) => {
            k.ins.iter_mut().chain(k.outs.iter_mut()).for_each(m);
            if let Some(rev) = &mut k.rev {
                rev.outs.iter_mut().for_each(m);
            }
        }
        Instr::Public { addr, .. } => m(addr),
    }
}

/// Renumbers the hash half so every crossing cell takes its Field VM address
/// and every other cell an address past MEM's `mem_len`.
pub fn renumber(side: &mut HashSide, faddr: &dyn Fn(u64) -> u64, mem_len: u64) {
    assert!(
        side.out.iter().all(|c| !c.is_word && c.l3c == FE::zero()),
        "MEM sends a zero fourth lane"
    );
    let n = side.sub.num_addrs as usize;
    let mut crossing = vec![false; n];
    for a in side.crossing() {
        crossing[a as usize] = true;
    }
    let mut next = mem_len;
    let map: Vec<u64> = (0..n)
        .map(|a| {
            if crossing[a] {
                faddr(a as u64)
            } else {
                next += 1;
                next - 1
            }
        })
        .collect();
    let f = |a: Addr| Addr(map[a.0 as usize]);
    let mut read_counts = vec![0u64; next as usize];
    let mut buf = Vec::new();
    let mut instrs = side.sub.instrs.clone();
    for i in instrs.iter_mut() {
        buf.clear();
        i.reads_into(&mut buf);
        for a in &buf {
            read_counts[map[a.0 as usize] as usize] += 1;
        }
        remap(i, &f);
    }
    for &a in &side.back {
        read_counts[map[a as usize] as usize] += 1;
    }
    side.sub = compile(LfmProgramSource {
        instrs,
        num_addrs: next,
        read_counts,
        arena_schema: side.sub.arena_schema.clone(),
        public_len: 0,
    });
    for c in side.out.iter_mut() {
        c.addr = map[c.addr as usize];
    }
    for a in side.back.iter_mut() {
        *a = map[*a as usize];
    }
    side.merged = true;
}

/// The bridge rows, with each cell's Field VM address from `faddr`.
pub fn bridge_rows(side: &HashSide, faddr: impl Fn(u64) -> u64) -> Vec<BridgeRow> {
    let mults: Vec<u64> = side.sub.instrs[..side.out.len()]
        .iter()
        .map(|i| match i {
            Instr::Hint { mult, .. } => *mult,
            _ => unreachable!("placeholders lead the hash half"),
        })
        .collect();
    side.out
        .iter()
        .zip(mults)
        .map(|(c, m)| BridgeRow {
            faddr: faddr(c.addr),
            laddr: c.addr,
            mult_out: m,
            is_in: false,
            is_word: c.is_word,
            l3c: c.l3c,
        })
        .chain(side.back.iter().map(|&a| BridgeRow {
            faddr: faddr(a),
            laddr: a,
            mult_out: 0,
            is_in: true,
            is_word: true,
            l3c: FE::zero(),
        }))
        .collect()
}

/// `LFM_HINT` over the hints the hash half keeps (placeholders excluded),
/// values from `values` or zero.
fn hint_table(side: &HashSide, values: Option<&[LfmWord]>) -> TraceTable<F, E> {
    use crate::lfm::chips::hint::cols;
    let hints: Vec<(u64, u64)> = side.sub.instrs[side.out.len()..]
        .iter()
        .filter_map(|i| match i {
            Instr::Hint { out, mult, .. } => Some((out.0, *mult)),
            _ => None,
        })
        .collect();
    let height = hints.len().next_power_of_two().max(super::prove::MIN_ROWS);
    let mut trace = TraceTable::new_main(
        crate::tables::types::zeroed_fe_vec(height * cols::NUM_COLUMNS),
        cols::NUM_COLUMNS,
        1,
    );
    let t = &mut trace.main_table;
    for (r, (addr, mult)) in hints.iter().enumerate() {
        t.set(r, cols::OUT_ADDR, FE::from(*addr));
        t.set(r, cols::MULT, FE::from(*mult));
        if let Some(v) = values {
            for (k, x) in v[side.out.len() + r].iter().enumerate() {
                t.set(r, cols::V0 + k, *x);
            }
        }
    }
    trace
}

fn air<CS: ConstraintSet<F, E> + 'static>(
    name: &str,
    num_columns: usize,
    interactions: Vec<BusInteraction>,
    cs: CS,
    root: stark::config::Commitment,
    prep: usize,
    options: &ProofOptions,
) -> ExtraAir {
    Box::new(
        AirWithBuses::<F, E, NullBoundaryConstraintBuilder, FvmPublicInputs, CS>::new(
            num_columns,
            AuxiliaryTraceBuildData { interactions },
            options,
            1,
            cs,
        )
        .with_name(name)
        .with_preprocessed(root, prep),
    )
}

/// `(first row, rows)` per chunk of a table with `real` rows: with
/// `FVM_HYBRID_CHUNK`, the largest power of two and the rest when that saves
/// an eighth of the rows. The chips split have no row-to-row constraints.
fn chunk_plan(real: usize, height: usize) -> Vec<(usize, usize)> {
    let first = height / 2;
    let min = super::prove::MIN_ROWS;
    if std::env::var_os("FVM_HYBRID_CHUNK").is_some() && real > first && first >= min {
        let rest = (real - first).next_power_of_two().max(min);
        if first + rest + height / 8 <= height {
            return vec![(0, first), (first, rest)];
        }
    }
    vec![(0, height)]
}

/// `t` cut per [`chunk_plan`]; a chunk's padding repeats `t`'s first padding row.
fn chunks(t: &TraceTable<F, E>, real: usize) -> Vec<TraceTable<F, E>> {
    let plan = chunk_plan(real, t.num_rows());
    if plan.len() == 1 {
        return vec![t.clone()];
    }
    let w = t.main_table.width;
    plan.into_iter()
        .map(|(start, rows)| {
            let mut data = crate::tables::types::zeroed_fe_vec(rows * w);
            for r in 0..rows {
                let src = if start + r < real { start + r } else { real };
                if src < t.num_rows() {
                    for c in 0..w {
                        data[r * w + c] = *t.main_table.get(src, c);
                    }
                }
            }
            TraceTable::new_main(data, w, 1)
        })
        .collect()
}

fn prep_roots(
    chunks: &[TraceTable<F, E>],
    prep: usize,
    options: &ProofOptions,
) -> Vec<stark::config::Commitment> {
    chunks
        .iter()
        .map(|t| {
            let columns: Vec<Vec<FE>> = (0..prep)
                .map(|c| (0..t.num_rows()).map(|r| *t.main_table.get(r, c)).collect())
                .collect();
            crate::lfm::commit::commit_columns(&columns, options)
        })
        .collect()
}

fn group_roots(
    g: &crate::lfm::compiler::ColumnGroup,
    options: &ProofOptions,
) -> Vec<stark::config::Commitment> {
    if g.real_rows == 0 {
        return Vec::new();
    }
    let t = TraceTable::new_main(g.data.clone(), g.width, 1);
    prep_roots(&chunks(&t, g.real_rows), g.width, options)
}

/// The verifier's half of the statement: the preprocessed roots.
pub struct HashSideId {
    pub bridge: Option<stark::config::Commitment>,
    pub hash: Vec<stark::config::Commitment>,
    /// One root per chunk ([`chunk_plan`]); empty when the chip has no rows.
    pub lanes: Vec<stark::config::Commitment>,
    pub bitdec: Option<stark::config::Commitment>,
    pub select: Vec<stark::config::Commitment>,
    pub hint: Vec<stark::config::Commitment>,
    pub const_: Option<stark::config::Commitment>,
    pub balu: Vec<stark::config::Commitment>,
    pub xalu: Vec<stark::config::Commitment>,
    /// `LFM_KECCAK` and `KECCAK_RC` roots and the `KECCAK_RND` chunk count.
    pub keccak: Option<(stark::config::Commitment, stark::config::Commitment, usize)>,
    pub bitwise: Option<stark::config::Commitment>,
    pub hasher: HasherKind,
}

pub fn side_id(
    side: &HashSide,
    rows: &[BridgeRow],
    artifacts: &LfmArtifacts,
    options: &ProofOptions,
) -> HashSideId {
    let g = &side.sub.groups;
    assert_eq!(g.blake3.real_rows, 0, "no BLAKE3 on the hash half");
    let hash = if g.hash.real_rows == 0 {
        Vec::new()
    } else if artifacts.hash_chunk_roots.is_empty() {
        vec![artifacts.roots[crate::lfm::airs::HASH_SLOT]]
    } else {
        artifacts.hash_chunk_roots.clone()
    };
    HashSideId {
        bridge: (!side.merged).then(|| bridge::commitment(rows, options, super::prove::MIN_ROWS)),
        hash,
        lanes: group_roots(&g.lanes, options),
        bitdec: (g.bitdec.real_rows > 0).then_some(artifacts.roots[4]),
        select: group_roots(&g.select, options),
        const_: (g.const_.real_rows > 0).then_some(artifacts.roots[0]),
        balu: group_roots(&g.balu, options),
        xalu: group_roots(&g.xalu, options),
        hint: if g.hint.real_rows > side.out.len() {
            prep_roots(
                &chunks(&hint_table(side, None), g.hint.real_rows - side.out.len()),
                layout::hint::PREP_WIDTH,
                options,
            )
        } else {
            Vec::new()
        },
        keccak: artifacts.chip_set.keccak.then_some((
            artifacts.roots[crate::lfm::airs::KECCAK_SLOT],
            artifacts.roots[crate::lfm::airs::KECCAK_RC_SLOT],
            artifacts.keccak_rnd_chunks,
        )),
        bitwise: artifacts
            .chip_set
            .bitwise
            .then_some(artifacts.roots[crate::lfm::airs::BITWISE_SLOT]),
        hasher: artifacts.hasher,
    }
}

/// BRIDGE, the `LFM_HASH` chunks, LANES, BITDEC, SELECT, CONST, HINT, then the keccak family
/// (`LFM_KECCAK`, the `KECCAK_RND` chunks, `KECCAK_RC`) and `BITWISE`.
pub fn airs(id: &HashSideId, options: &ProofOptions) -> Vec<ExtraAir> {
    let mut v = Vec::new();
    if let Some(root) = id.bridge {
        v.push(air(
            "FIELD_VM_BRIDGE",
            bridge::cols::NUM_COLUMNS,
            bridge::bus_interactions(),
            BridgeConstraints,
            root,
            bridge::cols::NUM_PRECOMPUTED_COLS,
            options,
        ));
    }
    for root in &id.hash {
        v.push(air(
            "LFM_HASH",
            hash::num_columns(id.hasher),
            hash::bus_interactions(id.hasher),
            hash::HashConstraints { kind: id.hasher },
            *root,
            layout::hash::PREP_WIDTH,
            options,
        ));
    }
    for &root in &id.lanes {
        v.push(air(
            "LFM_LANES",
            lanes::cols::NUM_COLUMNS,
            lanes::bus_interactions(),
            EmptyConstraints,
            root,
            layout::lanes::PREP_WIDTH,
            options,
        ));
    }
    if let Some(root) = id.bitdec {
        v.push(air(
            "LFM_BITDEC",
            bitdec::cols::NUM_COLUMNS,
            bitdec::bus_interactions(),
            bitdec::BitDecConstraints,
            root,
            layout::bitdec::PREP_WIDTH,
            options,
        ));
    }
    for &root in &id.select {
        v.push(air(
            "LFM_SELECT",
            select::cols::NUM_COLUMNS,
            select::bus_interactions(),
            select::SelectConstraints,
            root,
            layout::select::PREP_WIDTH,
            options,
        ));
    }
    if let Some(root) = id.const_ {
        v.push(air(
            "LFM_CONST",
            const_::cols::NUM_COLUMNS,
            const_::bus_interactions(),
            EmptyConstraints,
            root,
            layout::const_::PREP_WIDTH,
            options,
        ));
    }
    for &root in &id.balu {
        v.push(air(
            "LFM_BALU",
            balu::cols::NUM_COLUMNS,
            balu::bus_interactions(),
            balu::BaluConstraints,
            root,
            layout::balu::PREP_WIDTH,
            options,
        ));
    }
    for &root in &id.xalu {
        v.push(air(
            "LFM_XALU",
            xalu::cols::NUM_COLUMNS,
            xalu::bus_interactions(),
            xalu::XaluConstraints,
            root,
            layout::xalu::PREP_WIDTH,
            options,
        ));
    }
    for &root in &id.hint {
        v.push(air(
            "LFM_HINT",
            crate::lfm::chips::hint::cols::NUM_COLUMNS,
            crate::lfm::chips::hint::bus_interactions(),
            EmptyConstraints,
            root,
            layout::hint::PREP_WIDTH,
            options,
        ));
    }
    if let Some((adapter, rc, chunks)) = id.keccak {
        v.push(air(
            "LFM_KECCAK",
            keccak::cols::NUM_COLUMNS,
            keccak::bus_interactions(),
            keccak::KeccakAdapterConstraints,
            adapter,
            layout::keccak::PREP_WIDTH,
            options,
        ));
        for _ in 0..chunks {
            v.push(Box::new(
                AirWithBuses::<F, E, NullBoundaryConstraintBuilder, FvmPublicInputs, _>::new(
                    keccak_rnd::cols::NUM_COLUMNS,
                    AuxiliaryTraceBuildData {
                        interactions: keccak_rnd::bus_interactions(),
                    },
                    options,
                    1,
                    keccak_rnd::KeccakRndConstraints,
                )
                .with_name("KECCAK_RND"),
            ));
        }
        v.push(air(
            "KECCAK_RC",
            keccak_rc::cols::NUM_COLUMNS,
            keccak_rc::bus_interactions(),
            EmptyConstraints,
            rc,
            keccak_rc::NUM_PRECOMPUTED_COLS,
            options,
        ));
    }
    if let Some(root) = id.bitwise {
        v.push(air(
            "BITWISE",
            bitwise::cols::NUM_COLUMNS,
            bitwise::bus_interactions(),
            EmptyConstraints,
            root,
            bitwise::NUM_PRECOMPUTED_COLS,
            options,
        ));
    }
    v
}

/// The traces for [`airs`], and the cells MEM shares on `LfmMem` when merged;
/// `mem` is the Field VM's final memory.
pub fn traces(
    side: &HashSide,
    rows: &[BridgeRow],
    id: &HashSideId,
    mem: &[crate::tables::types::FEE],
) -> (Vec<TraceTable<F, E>>, Vec<super::mem::LfmCell>) {
    let exec = crate::lfm::executor::execute(&side.sub, &side.arenas, &id.hasher)
        .expect("the hash half executes");
    let word = |i: usize| -> [FE; 4] {
        let r = &rows[i];
        let lfm = exec.memory.get(Addr(r.laddr)).expect("a crossing cell");
        let fvm = mem[r.faddr as usize].value();
        debug_assert_eq!(&lfm[..3], fvm, "cell {} disagrees across the bridge", r.laddr);
        [fvm[0], fvm[1], fvm[2], lfm[3]]
    };
    let mut t = crate::lfm::trace::build_traces_with_hasher(&side.sub, &exec.records, id.hasher);
    let mut v = Vec::new();
    let mut cells = Vec::new();
    if id.bridge.is_some() {
        v.push(bridge::generate_trace(rows, word, super::prove::MIN_ROWS));
    } else {
        cells = rows
            .iter()
            .enumerate()
            .map(|(i, r)| (r.faddr, r.mult_out, r.is_in as u64, word(i)[3]))
            .collect();
    }
    if !id.hash.is_empty() {
        v.push(std::mem::replace(&mut t.hash, TraceTable::new_main(Vec::new(), 1, 1)));
        v.extend(t.hash_tail.drain(..));
    }
    let g = &side.sub.groups;
    if !id.lanes.is_empty() {
        v.extend(chunks(&t.lanes, g.lanes.real_rows));
    }
    if id.bitdec.is_some() {
        v.push(std::mem::replace(&mut t.bitdec, TraceTable::new_main(Vec::new(), 1, 1)));
    }
    if !id.select.is_empty() {
        v.extend(chunks(&t.select, g.select.real_rows));
    }
    if id.const_.is_some() {
        v.push(std::mem::replace(&mut t.const_, TraceTable::new_main(Vec::new(), 1, 1)));
    }
    if !id.balu.is_empty() {
        v.extend(chunks(&t.balu, g.balu.real_rows));
    }
    if !id.xalu.is_empty() {
        v.extend(chunks(&t.xalu, g.xalu.real_rows));
    }
    if !id.hint.is_empty() {
        v.extend(chunks(
            &hint_table(side, Some(&exec.records.hint)),
            g.hint.real_rows - side.out.len(),
        ));
    }
    if let Some((.., chunks)) = id.keccak {
        assert_eq!(t.keccak_rnd.len(), chunks, "KECCAK_RND chunking");
        v.push(std::mem::replace(&mut t.keccak, TraceTable::new_main(Vec::new(), 1, 1)));
        v.extend(t.keccak_rnd.drain(..));
        v.push(std::mem::replace(&mut t.keccak_rc, TraceTable::new_main(Vec::new(), 1, 1)));
    }
    if id.bitwise.is_some() {
        if t.bitwise.num_rows() == 0 {
            t.bitwise = bitwise::generate_bitwise_trace();
        }
        v.push(std::mem::replace(&mut t.bitwise, TraceTable::new_main(Vec::new(), 1, 1)));
    }
    (v, cells)
}
