//! The hand-out's counting on a thread of its own
//! (`LAMBDA_VM_HANDOUT_COUNT_THREAD`, [`WindowedTraceBuilder::count_handout_on_thread`]).
//!
//! Under [`WindowedTraceBuilder::drop_streamed_ops`] each full chunk leaves the
//! run's tails with what the table phase would have read from its ops counted:
//! DECODE's lookups and the last ECALL from CPU, BITWISE's lookups from MEMW_R,
//! MEMW_A, LT and SHIFT, and the phase-3 LT ops of MEMW and MEMW_A. That
//! counting is the larger part of the accumulator's work, and the accumulator
//! absorbs the windows strictly in order, so it paces the windows. Every count
//! is either a sum (the BITWISE histogram, the DECODE rows) or a list appended
//! chunk by chunk in order (the derived LT ops, the last ECALL): a thread fed
//! the same chunks in the same order arrives at the same counts, and the
//! builder adds them to its own once the windows are walked
//! ([`HandoutCounter::finish`]). The accumulator then only takes the chunk out
//! of its tail and sends the counter the chunk's parts (shared, not copied).
//!
//! The functions below are the counts themselves, called on the accumulator's
//! thread (the knob at `0`) and on the counter's alike.
//!
//! [`WindowedTraceBuilder::drop_streamed_ops`]: super::WindowedTraceBuilder::drop_streamed_ops
//! [`WindowedTraceBuilder::count_handout_on_thread`]: super::WindowedTraceBuilder::count_handout_on_thread

use std::sync::Arc;
use std::sync::mpsc::{Sender, channel};
use std::time::Instant;

use super::super::{
    CompactLt, CpuOperation, LtOperation, MemwOperation, RegRow, ShiftOperation, bitwise,
    collect_bitwise_from_lt_ops, collect_bitwise_from_memw_aligned, collect_lt_from_memw,
    collect_lt_from_memw_aligned, lt_row_lookups,
};
use super::{Segments, StreamTable};
use crate::Error;
use crate::tables::decode::DecodeTable;
use crate::tables::{memw_aligned, memw_register, shift};

/// `LAMBDA_VM_HANDOUT_COUNT_THREAD`: unset or `1` counts on a thread of its
/// own (with the `parallel` feature; without it, on the accumulator's), `0`
/// on the accumulator's thread (the rollback). Anything else is an error. The
/// tables are the same either way.
pub(super) fn count_thread_from_env() -> Result<bool, Error> {
    match std::env::var("LAMBDA_VM_HANDOUT_COUNT_THREAD")
        .ok()
        .as_deref()
        .map(str::trim)
    {
        None | Some("") | Some("1") => Ok(cfg!(feature = "parallel")),
        Some("0") => Ok(false),
        Some(v) => Err(Error::Prover(format!(
            "LAMBDA_VM_HANDOUT_COUNT_THREAD={v}: 1 (a thread of its own) or 0 (the accumulator's)"
        ))),
    }
}

/// The (timestamp, next pc) of the last ECALL among `ops`.
pub(super) fn last_ecall(ops: &Segments<CpuOperation>, decode: &DecodeTable) -> Option<(u64, u64)> {
    ops.iter()
        .rev()
        .find(|op| decode.entry(op.decode_row).fields.ecall)
        .map(|op| (op.timestamp, op.next_pc))
}

/// A MEMW_R chunk's BITWISE lookups.
pub(super) fn count_memw_register(
    rows: &Segments<RegRow>,
    histogram: &mut bitwise::BitwiseHistogram,
) {
    for part in rows.slices() {
        memw_register::collect_bitwise_from_memw_register(part, histogram)
    }
}

/// A MEMW_A chunk's BITWISE lookups, and its phase-3 LT ops into `lt` (none
/// when the MEMW-derived LT ops are streamed).
pub(super) fn count_memw_aligned(
    ops: &Segments<memw_aligned::AlignedRow>,
    histogram: &mut bitwise::BitwiseHistogram,
    mut lt: Option<&mut CompactLt>,
) {
    for rows in ops.slices() {
        let lookups = collect_bitwise_from_memw_aligned(rows);
        histogram.add_ops(&lookups);
        if let Some(lt) = lt.as_deref_mut() {
            let derived = collect_lt_from_memw_aligned(rows);
            lt.extend(&derived);
        }
    }
}

/// A MEMW chunk's phase-3 LT ops into `lt` (none when they are streamed).
pub(super) fn count_memw(ops: &Segments<MemwOperation>, lt: Option<&mut CompactLt>) {
    if let Some(lt) = lt {
        for part in ops.slices() {
            lt.extend(&collect_lt_from_memw(part));
        }
    }
}

/// An LT chunk's BITWISE lookups: its ops' and its rows' μ bounds (the chunk
/// is one LT table, `types::MU_MAX`).
pub(super) fn count_lt(ops: &Segments<LtOperation>, histogram: &mut bitwise::BitwiseHistogram) {
    for part in ops.slices() {
        histogram.add_ops(&collect_bitwise_from_lt_ops(part))
    }
    histogram.add_ops(&lt_row_lookups(
        ops.slices().into_iter().flat_map(|part| part.iter()),
    ));
}

/// A SHIFT chunk's BITWISE lookups.
pub(super) fn count_shift(
    ops: &Segments<ShiftOperation>,
    histogram: &mut bitwise::BitwiseHistogram,
) {
    for part in ops.slices() {
        histogram.add_ops(&shift::collect_bitwise_from_shift(part))
    }
}

/// A chunk handed to the counter: its ops as the parts it lies in, shared
/// with the chunk the layout generates.
pub(super) enum CountWork {
    Cpu(Segments<CpuOperation>),
    MemwRegister(Segments<RegRow>),
    MemwAligned(Segments<memw_aligned::AlignedRow>),
    Memw(Segments<MemwOperation>),
    Lt(Segments<LtOperation>),
    Shift(Segments<ShiftOperation>),
}

/// Where the counter's thread spent its time, seconds, and what the builder
/// waited for it. A measurement.
#[derive(Clone, Copy, Debug, Default)]
pub struct CountThreadStamps {
    /// Counting, per streamed table in [`StreamTable::ALL`] order.
    pub busy: [f64; 9],
    /// The chunks counted, per streamed table.
    pub chunks: [usize; 9],
    /// Waiting for a chunk.
    pub idle: f64,
    /// The builder's wait, once the windows were walked, for the chunks the
    /// thread had still to count.
    pub drain: f64,
    /// The builder's adding the thread's counts to its own.
    pub merge: f64,
}

/// What the counter counted: the builder adds it to its own
/// ([`super::WindowedTraceBuilder::join_handout_count`]).
pub(super) struct Counts {
    pub(super) histogram: bitwise::BitwiseHistogram,
    /// DECODE's lookups, by row.
    pub(super) decode_rows: Vec<u64>,
    pub(super) last_ecall: Option<(u64, u64)>,
    pub(super) memw_lt: CompactLt,
    pub(super) memw_aligned_lt: CompactLt,
    pub(super) stamps: CountThreadStamps,
}

impl Counts {
    /// One chunk counted, as the accumulator's thread counts it; DECODE's
    /// lookups into `decode_rows` instead of DECODE's trace. Its table.
    fn count(
        &mut self,
        work: CountWork,
        decode: &DecodeTable,
        stream_memw_lt: bool,
    ) -> StreamTable {
        match work {
            CountWork::Cpu(ops) => {
                for op in ops.iter() {
                    let row = op.decode_row as usize;
                    if row >= self.decode_rows.len() {
                        self.decode_rows.resize(row + 1, 0);
                    }
                    self.decode_rows[row] += 1;
                }
                if let Some(ecall) = last_ecall(&ops, decode) {
                    self.last_ecall = Some(ecall);
                }
                StreamTable::Cpu
            }
            CountWork::MemwRegister(rows) => {
                count_memw_register(&rows, &mut self.histogram);
                StreamTable::MemwRegister
            }
            CountWork::MemwAligned(ops) => {
                count_memw_aligned(
                    &ops,
                    &mut self.histogram,
                    (!stream_memw_lt).then_some(&mut self.memw_aligned_lt),
                );
                StreamTable::MemwAligned
            }
            CountWork::Memw(ops) => {
                count_memw(&ops, (!stream_memw_lt).then_some(&mut self.memw_lt));
                StreamTable::Memw
            }
            CountWork::Lt(ops) => {
                count_lt(&ops, &mut self.histogram);
                StreamTable::Lt
            }
            CountWork::Shift(ops) => {
                count_shift(&ops, &mut self.histogram);
                StreamTable::Shift
            }
        }
    }
}

/// The counter's thread and the channel that feeds it, in run order.
pub(super) struct HandoutCounter {
    tx: Sender<CountWork>,
    thread: std::thread::JoinHandle<Counts>,
}

impl HandoutCounter {
    /// The thread, up and waiting for chunks. `memw_lt` and `memw_aligned_lt`
    /// are the empty lists it derives LT ops into (kept as the builder keeps
    /// its own). An error when the thread does not start.
    pub(super) fn spawn(
        decode: Arc<DecodeTable>,
        stream_memw_lt: bool,
        memw_lt: CompactLt,
        memw_aligned_lt: CompactLt,
    ) -> std::io::Result<Self> {
        let (tx, rx) = channel::<CountWork>();
        let thread = std::thread::Builder::new()
            .name("handout-count".to_string())
            .spawn(move || {
                let mut counts = Counts {
                    histogram: bitwise::BitwiseHistogram::new(),
                    decode_rows: vec![0; decode.padding_row() + 1],
                    last_ecall: None,
                    memw_lt,
                    memw_aligned_lt,
                    stamps: CountThreadStamps::default(),
                };
                loop {
                    let t = Instant::now();
                    let Ok(work) = rx.recv() else {
                        break;
                    };
                    counts.stamps.idle += t.elapsed().as_secs_f64();
                    let t = Instant::now();
                    let table = counts.count(work, &decode, stream_memw_lt);
                    let i = table.position();
                    counts.stamps.busy[i] += t.elapsed().as_secs_f64();
                    counts.stamps.chunks[i] += 1;
                }
                counts
            })?;
        Ok(Self { tx, thread })
    }

    /// Hands the thread a chunk to count. A thread that stopped (it panicked)
    /// is reported by [`Self::finish`].
    pub(super) fn send(&self, work: CountWork) {
        let _ = self.tx.send(work);
    }

    /// No more chunks: what the thread counted, once it has counted every
    /// chunk sent (`stamps.drain` the wait for it).
    pub(super) fn finish(self) -> Result<Counts, Error> {
        let t = Instant::now();
        let Self { tx, thread } = self;
        drop(tx);
        let mut counts = thread
            .join()
            .map_err(|_| Error::Prover("the hand-out's count thread panicked".to_string()))?;
        counts.stamps.drain = t.elapsed().as_secs_f64();
        Ok(counts)
    }
}
