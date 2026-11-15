//! The host's floor under the in-row register format
//! (`LAMBDA_VM_EXEC_HISTORY_FLOOR`): a timing arm of the host-only build in
//! record mode. The walkers make no side record, no MEMW_A row, no LOAD op and
//! no residual register row; they count those rows and their lookups, and the
//! build hands the families' full chunks out as placeholders of their shape,
//! for a card that would derive the rows from the executor's records. Nothing
//! fills a placeholder: the arm refuses a prove (`block_whir`'s
//! `exec_history_floor`), and a placeholder's host fill is none.
//!
//! Per window, on any walker, nothing carried in ([`light_window`]):
//! - in batches, the CPU ops collected and their DECODE rows counted; CPU32's
//!   ops and the walk's LT and SHIFT ops made, SHIFT's lookups counted;
//! - from the history record of each cycle: each register slot's `IS_HALF`
//!   (its step after [`in_row_prev`]), and each keep-alive and word-instruction
//!   register access as a residual row's lookups; each load's and store's
//!   class from its access record: MEMW_A's lookups and LT pair, or MEMW's op
//!   (kept as today); each load's lookups;
//! - each ECALL cycle walked as today, from a register state seeded from the
//!   cycle's recorded cells: its residual and aligned rows counted, its MEMW
//!   and accelerator ops kept;
//! - with [`FloorRoute::Walkers`], the batch's CPU ops routed and their
//!   lookups counted where they are, the window's segments handed over sized
//!   from the walker's last window; with [`FloorRoute::Accumulator`] the
//!   accumulator routes the window from its logs, as today.
//!
//! The accumulator ([`super::Accumulator::absorb`]) appends what the window
//! keeps, gives the CPU chunks the in-row shape with zero side records (one
//! shared buffer, so no byte is written a cycle), hands MEMW_A's LT pairs to
//! the hand-out's counter and cuts the families' chunks from the counts.
//!
//! What the arm does not build: the families' tails (the finish's tables hold
//! the last window's rows only), the CPU chunks' side columns (zero), anything
//! a prove reads from a placeholder. Every count is today's: BITWISE, DECODE,
//! LT and every table but the families' are the build's without the arm
//! (`floor/tests.rs`).
#![cfg_attr(not(feature = "parallel"), allow(dead_code))]

#[cfg(all(test, feature = "parallel"))]
mod tests;

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Instant;

use executor::vm::history::{ECALL, NO_ACCESS, WindowHistory};
use executor::vm::logs::Log;
use multilinear::narrow::NarrowColumns;
use multilinear::records::{Records, RecordsFill, TableRecords};

use super::super::{
    CpuRouter, MemoryState, MemwRoute, RegisterState, RoutedSegments, WalkOutputs, WalkSizes,
    build_cpu32_op, classify_memw, collect_cpu_ops_into, collect_window_ops_into, in_row_prev,
    memw_aligned_lookups,
};
use super::card_records::RowChunk;
use super::{
    ChunkJob, ChunkOps, HandoutStamps, Kept, StreamTable, Table, WalkCounts, WalkedWindow,
    WindowMade, WindowedTraceBuilder,
};
use crate::Error;
use crate::tables::bitwise::{BitwiseHistogram, BitwiseOperation, BitwiseOperationType};
use crate::tables::cpu::{CpuOperation, CpuSide};
use crate::tables::decode::DecodeTable;
use crate::tables::gpack::TraceForm;
use crate::tables::load::LoadOperation;
use crate::tables::lt::LtOperation;
use crate::tables::memw::MemwOperation;
use crate::tables::memw_aligned::{AlignedRow, MemwAlignedOrder};
use crate::tables::memw_register::RegRow;
use crate::tables::register::RegisterFormat;
use crate::tables::shift::{self, ShiftOperation};
use crate::tables::{load, memw_aligned, memw_register, register_residual};

/// Where the arm's windows are routed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FloorRoute {
    /// By the accumulator, from each window's logs, as today.
    Accumulator,
    /// By each window's walker, in its batches.
    Walkers,
}

/// The families the arm makes no row of, in this order everywhere: their
/// tables and the counts' places. Under the in-row register format MEMW_R's
/// table is the residual register table.
const FAMILIES: [StreamTable; 3] = [
    StreamTable::MemwRegister,
    StreamTable::MemwAligned,
    StreamTable::Load,
];
const RESIDUAL: usize = 0;
const MEMW_A: usize = 1;
const LOAD: usize = 2;

/// What a window's light pass hands over besides the lists the build still
/// keeps ([`WalkedWindow::walk`]: MEMW's ops, the walk's LT and SHIFT ops,
/// CPU32's ops, the accelerators').
pub(super) struct FloorWindow {
    /// With [`FloorRoute::Walkers`], the window's routing.
    pub(super) segments: Option<RoutedSegments>,
    /// The rows each of [`FAMILIES`] gets from the window, its ECALLs' included.
    rows: [usize; 3],
    /// MEMW_A's `(old timestamp, timestamp)` LT pairs, in its rows' order.
    pairs: Vec<(u64, u64)>,
    ecalls: usize,
    /// The light pass's seconds, and of them the ECALL cycles'.
    secs: f64,
    ecall_secs: f64,
}

/// One walker's buffers for the arm, kept from window to window.
pub(super) struct FloorWorker {
    /// A batch's CPU ops.
    cpu_ops: Vec<CpuOperation>,
    /// An ECALL cycle's walk: its lists and the register state it starts from.
    ecall: WalkOutputs,
    registers: RegisterState,
    /// The last window's kept lists' lengths, to size the next window's.
    sizes: WalkSizes,
    /// The last window's routed segments' lengths (CPU's six), and its cycles.
    routed: ([usize; 6], usize),
}

impl FloorWorker {
    pub(super) fn new(order: MemwAlignedOrder) -> Self {
        let mut ecall = WalkOutputs::for_walk(0, false, order);
        ecall.memw.registers = RegisterFormat::InRow;
        Self {
            cpu_ops: Vec::new(),
            ecall,
            registers: RegisterState::new(0),
            sizes: WalkSizes::default(),
            routed: ([0; 6], 0),
        }
    }
}

/// The lengths of the six segments the CPU ops route to.
fn routed_lens(s: &RoutedSegments) -> [usize; 6] {
    [
        s.branch_ops.len(),
        s.mul_filter.len(),
        s.dvrm_filter.len(),
        s.eq_ops.len(),
        s.bytewise_ops.len(),
        s.store_ops.len(),
    ]
}

/// The `IS_HALF` lookup of a halfword `d`.
#[inline]
fn is_half(d: u16) -> BitwiseOperation {
    BitwiseOperation::halfword(BitwiseOperationType::IsHalf, d as u8, (d >> 8) as u8)
}

/// A residual register row's lookups (its step's two limbs), as the residual
/// table counts them.
#[inline]
fn count_residual(row: RegRow, histogram: &mut BitwiseHistogram) {
    register_residual::count_bitwise_from_residual(std::slice::from_ref(&row), histogram);
}

/// The loaded bytes of a `width`-byte load of `value`, sign- or zero-filled
/// past the width, as the walk's LOAD op holds them.
#[inline]
fn loaded_bytes(value: u64, width: usize, signed: bool) -> [u32; 8] {
    let mut bytes = [0u32; 8];
    for (k, byte) in bytes.iter_mut().take(width).enumerate() {
        *byte = ((value >> (k * 8)) & 0xFF) as u32;
    }
    if width < 8 {
        let fill = if signed && (bytes[width - 1] >> 7) & 1 == 1 {
            0xFF
        } else {
            0
        };
        bytes[width..].fill(fill);
    }
    bytes
}

/// Where a chunk of `table` goes in a window's hand-out: [`super::tail_jobs`]'s
/// order, which the arm's placeholders join.
pub(super) fn hand_out_rank(table: StreamTable) -> usize {
    match table {
        StreamTable::Cpu => 0,
        StreamTable::MemwRegister => 1,
        StreamTable::MemwAligned => 2,
        StreamTable::Memw => 3,
        StreamTable::Load => 4,
        StreamTable::Lt => 5,
        StreamTable::Shift => 6,
        StreamTable::KeccakRnd => 7,
        StreamTable::Store => 8,
    }
}

/// The window from cycle `first` of record mode (`logs`, `history`), its
/// light pass: the window the walk makes, less the families' rows, which are
/// counted into `counts`. `record` is the walker's record-mode state, its
/// overlay on `history`: only ECALL cycles walk it, an ECREC call's witness
/// deferred with `defer_ecrec` as the walk defers it. Refuses a window whose
/// cycle halts (the finish's), or whose load or store has no access in the
/// history.
#[allow(clippy::too_many_arguments)]
pub(super) fn light_window(
    worker: &mut FloorWorker,
    counts: &mut WalkCounts,
    decode: &DecodeTable,
    order: MemwAlignedOrder,
    batch: usize,
    defer_ecrec: bool,
    route: FloorRoute,
    record: &mut MemoryState,
    first: usize,
    logs: &[Log],
    history: &WindowHistory,
) -> Result<WalkedWindow, Error> {
    let start = Instant::now();
    if logs.iter().any(|log| log.next_pc == 0) {
        return Err(Error::HaltInNonFinalEpoch);
    }
    let n = logs.len();
    if history.extras.len() != n {
        return Err(Error::Prover(format!(
            "the floor's pass: a window of {n} cycles with {} history records",
            history.extras.len()
        )));
    }
    let refuse = |cycle: usize, why: &str| {
        Err(Error::Prover(format!(
            "the floor's pass: cycle {}: {why}",
            first + cycle
        )))
    };
    // The lists the build keeps, sized from the last window's.
    let scaled = |len: usize, cycles: usize| {
        if cycles == 0 {
            return 0;
        }
        let s = (len as u128 * n as u128 / cycles as u128) as usize;
        s + s / 8 + 16
    };
    let sizes = worker.sizes;
    let mut walk = WalkOutputs::for_walk(0, false, order);
    walk.memw.registers = RegisterFormat::InRow;
    walk.memw
        .general
        .reserve(scaled(sizes.general, sizes.cycles));
    walk.lt_ops.reserve(scaled(sizes.lt, sizes.cycles));
    walk.shift_ops.reserve(scaled(sizes.shift, sizes.cycles));
    let mut router = (route == FloorRoute::Walkers).then(|| {
        let (lens, cycles) = worker.routed;
        CpuRouter::with_capacities(lens.map(|len| scaled(len, cycles)))
    });
    let mut rows = [0usize; 3];
    let mut pairs: Vec<(u64, u64)> = Vec::new();
    let (mut last_ecall, mut ecalls, mut ecall_secs) = (None, 0usize, 0.0f64);
    let histogram = counts.histogram.get_or_insert_with(BitwiseHistogram::new);
    let decode_rows = &mut counts.decode_rows;
    let batch = if batch == 0 { n.max(1) } else { batch };
    let FloorWorker {
        cpu_ops,
        ecall,
        registers,
        ..
    } = &mut *worker;
    for (b, part) in logs.chunks(batch).enumerate() {
        let from = b * batch;
        cpu_ops.clear();
        collect_cpu_ops_into(part, decode, first + from, cpu_ops)?;
        let shift_mark = walk.shift_ops.len();
        for (j, raw) in cpu_ops.iter().enumerate() {
            let i = from + j;
            let op = decode.op(raw);
            let f = op.decode.fields;
            let ts = op.timestamp;
            let row = raw.decode_row as usize;
            if row >= decode_rows.len() {
                decode_rows.resize(row + 1, 0);
            }
            decode_rows[row] += 1;
            if let Some(router) = router.as_mut() {
                router.push(op);
                op.count_bitwise_into(histogram);
            }
            let extra = history.extras[i];
            if f.ecall {
                // Today's walk of the cycle, from the cells its history gives:
                // its own register accesses, then the chip's.
                let t = Instant::now();
                ecalls += 1;
                last_ecall = Some((raw.timestamp, raw.next_pc));
                *registers = RegisterState {
                    regs: [(0, 0); 32],
                    index_register: (0, 0),
                    pc_register: (0, 0),
                };
                let seeds = [
                    (f.read_register1, f.rs1, op.rv1, 0),
                    (f.read_register2, f.rs2, op.rv2, 1),
                    (f.write_register, f.rd, 0, 2),
                ];
                for (on, reg, value, k) in seeds.into_iter().rev() {
                    let at = u64::from(extra.regs_ts[k]);
                    match reg {
                        0 => {}
                        _ if !on => {}
                        255 => registers.pc_register = (value, at),
                        r => registers.regs[usize::from(r & 31)] = (value, at),
                    }
                }
                for c in history.cycle_registers(i) {
                    if c.reg == 254 {
                        registers.index_register = (c.value as u32, u64::from(c.ts));
                    } else {
                        registers.regs[usize::from(c.reg & 31)] = (c.value, u64::from(c.ts));
                    }
                }
                collect_window_ops_into(
                    std::slice::from_ref(raw),
                    decode,
                    record,
                    registers,
                    ecall,
                    false,
                    defer_ecrec,
                )?;
                // The side rows, counted as the walk counts them.
                memw_register::collect_bitwise_from_memw_register(
                    &ecall.memw.register_rows,
                    histogram,
                );
                rows[RESIDUAL] += ecall.memw.register_rows.len();
                ecall.memw.register_rows.clear();
                for aligned in &ecall.memw.aligned {
                    memw_aligned_lookups(aligned, |b| histogram.bump(b));
                    pairs.extend(aligned.lt_pair());
                }
                rows[MEMW_A] += ecall.memw.aligned.len();
                ecall.memw.aligned.clear();
                histogram.add_ops(&ecall.bitwise_ops);
                ecall.bitwise_ops.clear();
                ecall.cpu_side.clear();
                rows[LOAD] += ecall.load_ops.len();
                for load in ecall.load_ops.drain(..) {
                    load.count_bitwise_into(histogram);
                }
                // MEMW's ops and the accelerators', kept as today's.
                walk.memw.general.append(&mut ecall.memw.general);
                walk.lt_ops.append(&mut ecall.lt_ops);
                walk.shift_ops.append(&mut ecall.shift_ops);
                walk.keccak_ops.append(&mut ecall.keccak_ops);
                walk.blake3_ops.append(&mut ecall.blake3_ops);
                walk.blake3_absorb_ops.append(&mut ecall.blake3_absorb_ops);
                walk.cpu32_ops.append(&mut ecall.cpu32_ops);
                walk.ecsm_ops.append(&mut ecall.ecsm_ops);
                walk.ecdas_ops.append(&mut ecall.ecdas_ops);
                walk.copy_ops.append(&mut ecall.copy_ops);
                walk.hint_ops.append(&mut ecall.hint_ops);
                walk.ecrec_ops.append(&mut ecall.ecrec_ops);
                ecall_secs += t.elapsed().as_secs_f64();
                continue;
            }
            if f.word_instr {
                walk.cpu32_ops.push(build_cpu32_op(op));
            }
            // The load or store first, as the walk makes it.
            let load = f.is_load();
            if load || f.is_store() {
                let Some(a) = (extra.side != NO_ACCESS && extra.side & ECALL == 0)
                    .then(|| history.accesses.get(extra.side as usize))
                    .flatten()
                else {
                    return refuse(i, "a load or store with no access in its history");
                };
                let width = f.mem_bytes();
                let base = op.res;
                let old_ts: [u64; 8] = if a.mixed == NO_ACCESS {
                    [u64::from(a.ts); 8]
                } else {
                    let Some(mixed) = history.mixed.get(a.mixed as usize) else {
                        return refuse(i, "an access with no mixed timestamps");
                    };
                    mixed.map(u64::from)
                };
                let memw_op = if load {
                    let bytes = loaded_bytes(op.rvd, width, f.mem_signed());
                    LoadOperation::new(
                        base,
                        ts,
                        width as u8,
                        f.mem_signed(),
                        bytes.map(|b| b as u8),
                    )
                    .count_bitwise_into(histogram);
                    rows[LOAD] += 1;
                    MemwOperation::new(false, base, bytes, ts, width as u8, true)
                        .with_old(bytes, old_ts)
                } else {
                    let value = std::array::from_fn(|k| ((op.rv2 >> (k * 8)) & 0xFF) as u32);
                    let old = a.old.to_le_bytes().map(u32::from);
                    MemwOperation::new(false, base, value, ts, width as u8, false)
                        .with_old(old, old_ts)
                };
                match classify_memw(&memw_op, order) {
                    MemwRoute::Aligned => {
                        let aligned = AlignedRow::from_memw(&memw_op, order)
                            .map_err(|why| Error::Prover(format!("MEMW_A: {why}")))?;
                        memw_aligned_lookups(&aligned, |b| histogram.bump(b));
                        pairs.extend(aligned.lt_pair());
                        rows[MEMW_A] += 1;
                    }
                    MemwRoute::General => walk.memw.general.push(memw_op),
                    MemwRoute::Register => {
                        return refuse(i, "a memory access the routing takes for a register's");
                    }
                }
            }
            // Then `rs1` at the cycle's timestamp, `rs2` one later, `rd` two:
            // a word instruction's as residual rows, any other row's in its
            // slots, past the slot's reach behind a keep-alive.
            let slots = [
                (f.read_register1, f.rs1),
                (f.read_register2, f.rs2),
                (f.write_register, f.rd),
            ];
            for (k, (on, reg)) in slots.into_iter().enumerate() {
                if !on || reg == 0 {
                    continue;
                }
                let at = ts + k as u64;
                let old = u64::from(extra.regs_ts[k]);
                if old >= at {
                    return refuse(i, "a register access no later than the one before it");
                }
                if f.word_instr {
                    let address = 2 * u64::from(reg);
                    count_residual(
                        RegRow::residual(address, at, [0; 2], [0; 2], old, k < 2),
                        histogram,
                    );
                    rows[RESIDUAL] += 1;
                    continue;
                }
                let prev = in_row_prev(at, old);
                if prev != old {
                    count_residual(
                        RegRow::keep_alive(2 * u64::from(reg), prev, [0; 2], old),
                        histogram,
                    );
                    rows[RESIDUAL] += 1;
                }
                histogram.bump(is_half((at - prev - 1) as u16));
            }
            // The ALU chips' ops the walk makes.
            if !f.word_instr {
                if f.is_lt() {
                    walk.lt_ops.push(LtOperation::new_with_invert(
                        op.rv1,
                        op.arg2(),
                        f.alu_signed(),
                        f.alu_signed2_or_invert(),
                    ));
                }
                if f.is_shift() {
                    walk.shift_ops.push(ShiftOperation::new(
                        op.rv1,
                        op.arg2(),
                        f.alu_signed2_or_invert(),
                        f.alu_signed(),
                        f.word_instr,
                    ));
                }
            }
        }
        histogram.add_ops(&shift::collect_bitwise_from_shift(
            &walk.shift_ops[shift_mark..],
        ));
    }
    let segments = router.map(|router| router.finish(&walk.cpu32_ops));
    if let Some(segments) = segments.as_ref() {
        worker.routed = (routed_lens(segments), n);
    }
    worker.sizes = WalkSizes::of(&walk, n);
    let secs = start.elapsed().as_secs_f64();
    Ok(WalkedWindow {
        cpu_ops: Vec::new(),
        first,
        cycles: n,
        walk,
        logs: None,
        made: Some(WindowMade { last_ecall }),
        floor: Some(Box::new(FloorWindow {
            segments,
            rows,
            pairs,
            ecalls,
            secs,
            ecall_secs,
        })),
    })
}

/// The arm's state in the build: its route, what the windows counted so far
/// and its zero side records.
pub(super) struct FloorState {
    pub(super) route: FloorRoute,
    rows: [usize; 3],
    /// One buffer of zero side records a window length, shared by every
    /// window of that length.
    zeros: BTreeMap<usize, Arc<Vec<CpuSide>>>,
    stamps: FloorStamps,
}

/// What the arm did, for its `BLOCK FLOOR` line. A measurement.
#[derive(Clone, Copy, Debug, Default)]
pub struct FloorStamps {
    pub windows: usize,
    /// The light passes' seconds, summed over the walkers, and of them the
    /// ECALL cycles' walks.
    pub light: f64,
    pub ecall: f64,
    pub ecalls: usize,
    /// The slowest window's light pass, and which window.
    pub slowest: f64,
    pub slowest_window: usize,
    /// The rows each family got (residual, MEMW_A, LOAD), and the
    /// placeholders handed out for them.
    pub rows: [usize; 3],
    pub placeholders: [usize; 3],
    /// Whether the walkers routed.
    pub walkers_route: bool,
}

/// A full chunk of one of [`FAMILIES`] with no row in it: its shape and
/// nothing else. A card's fill would read no record; a host reader gets no
/// columns. Only a host-only build, which commits nothing, hands one out.
#[derive(Debug)]
struct Placeholder {
    table: StreamTable,
    rows: usize,
}

impl Placeholder {
    fn columns(&self) -> usize {
        match self.table {
            StreamTable::MemwRegister => register_residual::cols::NUM_COLUMNS,
            StreamTable::MemwAligned => memw_aligned::cols::NUM_COLUMNS,
            _ => load::cols::NUM_COLUMNS,
        }
    }
}

impl TableRecords for Placeholder {
    fn rows(&self) -> usize {
        self.rows
    }

    fn cols(&self) -> usize {
        self.columns()
    }

    fn fill(&self) -> RecordsFill<'_> {
        match self.table {
            StreamTable::MemwRegister => RecordsFill::MemwR { rows: Vec::new() },
            StreamTable::MemwAligned => RecordsFill::MemwA { rows: Vec::new() },
            _ => RecordsFill::Load { ops: Vec::new() },
        }
    }

    fn bytes(&self) -> usize {
        0
    }

    fn host_fill(&self) -> Option<NarrowColumns> {
        None
    }
}

impl RowChunk for Placeholder {
    fn op_bytes(&self) -> usize {
        0
    }

    fn op_count(&self) -> usize {
        self.rows
    }

    fn into_trace(self: Box<Self>) -> Table {
        held_placeholder(*self)
    }

    fn regenerate(self: Box<Self>, _form: TraceForm, _digest: bool) -> (Option<[u64; 2]>, Table) {
        (None, held_placeholder(*self))
    }
}

/// The table a placeholder is: its shape, held as its (empty) records.
fn held_placeholder(p: Placeholder) -> Table {
    let (rows, cols) = (p.rows, p.columns());
    Table::try_from_records_main(Records(Arc::new(p)), 1)
        .unwrap_or_else(|_| Table::new_main(vec![Default::default(); rows * cols], cols, 1))
}

impl FloorState {
    /// A light window's part of the absorb, before the window's lists join
    /// the run's: its counts added, its CPU chunks' side records (zeros)
    /// queued with its logs, its MEMW_A LT pairs returned for the hand-out's
    /// counter.
    pub(super) fn absorb(
        &mut self,
        window: FloorWindow,
        cycles: usize,
        kept: &mut Kept,
    ) -> Vec<(u64, u64)> {
        let FloorWindow {
            rows,
            pairs,
            ecalls,
            secs,
            ecall_secs,
            ..
        } = window;
        for (sum, n) in self.rows.iter_mut().zip(rows) {
            *sum += n;
        }
        let zeros = self
            .zeros
            .entry(cycles)
            .or_insert_with(|| Arc::new(vec![CpuSide::default(); cycles]));
        kept.cpu_side.push_shared(Arc::clone(zeros));
        let s = &mut self.stamps;
        if secs > s.slowest {
            (s.slowest, s.slowest_window) = (secs, s.windows);
        }
        s.windows += 1;
        s.light += secs;
        s.ecall += ecall_secs;
        s.ecalls += ecalls;
        pairs
    }

    /// Each of [`FAMILIES`]' full chunks the counts complete, as placeholders;
    /// `done` the chunks of each handed out so far, `max` their heights.
    pub(super) fn jobs(
        &mut self,
        max: [usize; 3],
        done: [&mut usize; 3],
        stamps: &mut HandoutStamps,
    ) -> Vec<ChunkJob> {
        let mut jobs = Vec::new();
        for (f, (max, done)) in max.into_iter().zip(done).enumerate() {
            let table = FAMILIES[f];
            while self.rows[f] >= (*done + 1) * max {
                jobs.push(ChunkJob {
                    table,
                    index: *done,
                    ops: ChunkOps::Rows(Box::new(Placeholder { table, rows: max })),
                });
                stamps.note(table, 0.0, 0.0, 0);
                self.stamps.placeholders[f] += 1;
                *done += 1;
            }
        }
        self.stamps.rows = self.rows;
        jobs
    }
}

impl WindowedTraceBuilder<'_> {
    /// The floor arm (see the module docs): the walkers' light pass and the
    /// hand-out above, routed as `route` says. A timing arm only: the
    /// families' chunks are placeholders that nothing fills and the CPU
    /// chunks' side columns are zero, so the build must never be proved (the
    /// block's prove refuses the knob outside its host-only build). Needs the
    /// in-row register format, [`Self::drop_streamed_ops`],
    /// [`Self::card_fill_streamed`] with CPU (the CPU chunks held as their
    /// logs), the walkers counting where they make the rows, chunk heights of
    /// whole tables, and not [`Self::stream_memw_lt`]. Before the first window
    /// only.
    pub fn history_floor(mut self, route: FloorRoute) -> Result<Self, Error> {
        let refuse = |why: &str| Err(Error::Prover(format!("the floor arm: {why}")));
        if self.cycles > 0 {
            return refuse("after a window was walked");
        }
        if self.emitted.register_format != RegisterFormat::InRow {
            return refuse("it needs the in-row register format");
        }
        if self.kept.is_none() {
            return refuse("it needs a build that drops its streamed ops");
        }
        if !self.keeps_logs() || !self.cpu_listless() {
            return refuse("it needs the CPU chunks held as their logs, with no CPU list");
        }
        if !self.counts_where_made() {
            return refuse("it needs the walkers to count where they make the rows");
        }
        if self.stream_memw_lt {
            return refuse("the MEMW-derived LT ops streamed");
        }
        if self.writes_words().is_some() {
            return refuse("register words");
        }
        let m = &self.max_rows;
        if [m.cpu, m.memw_register, m.memw_aligned, m.load]
            .iter()
            .any(|&rows| !rows.is_power_of_two() || rows < 4)
        {
            return refuse("a chunk height that is not a whole table");
        }
        self.floor = Some(FloorState {
            route,
            rows: [0; 3],
            zeros: BTreeMap::new(),
            stamps: FloorStamps {
                walkers_route: route == FloorRoute::Walkers,
                ..FloorStamps::default()
            },
        });
        Ok(self)
    }

    /// The floor arm's route, if this is its build ([`Self::history_floor`]).
    pub fn floor_route(&self) -> Option<FloorRoute> {
        self.floor.as_ref().map(|state| state.route)
    }

    /// The arm's stamps.
    pub fn floor_stamps(&self) -> Option<FloorStamps> {
        self.floor.as_ref().map(|state| state.stamps)
    }
}

impl FloorStamps {
    /// The `BLOCK FLOOR` line.
    pub fn line(&self) -> String {
        let [r, a, l] = self.rows;
        let [pr, pa, pl] = self.placeholders;
        format!(
            "BLOCK FLOOR: light pass, no side record, MEMW_A row, LOAD op or residual register \
             row made (a timing arm; nothing proved) · routed by the {} · {} windows · light pass \
             {:.2}s summed, of it ECALL cycles {:.2}s ({} ECALLs) · slowest window {:.3}s (window \
             {}) · rows counted residual {r} MEMW_A {a} LOAD {l} · placeholders residual {pr} \
             MEMW_A {pa} LOAD {pl}",
            if self.walkers_route {
                "walkers"
            } else {
                "accumulator"
            },
            self.windows,
            self.light,
            self.ecall,
            self.ecalls,
            self.slowest,
            self.slowest_window,
        )
    }
}
