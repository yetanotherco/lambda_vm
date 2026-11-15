//! The in-row register format's side records ([`CpuSide`]: each register
//! slot's previous timestamp and `rd`'s old value) derived from the executor's
//! history, instead of written by the walk: the host reference the card's
//! `derive_cpu_sides` (`crypto/math-cuda/kernels/fill.cu`) restates rule for
//! rule, and that every host reader of a CPU chunk held this way goes through.
//!
//! A chunk carries, beside its logs, each cycle's [`HistExtra`] (the old
//! timestamps of its `rs1`, `rs2` and `rd` accesses, as the history writes
//! them) and a short list of [`SideException`]s: the rows whose `rd` old value
//! its own logs do not give. For the row of cycle `c` (timestamp
//! `ts = 4c + 4`):
//! - a word instruction's row is all zeros (the walk's `CpuSide::default()`);
//! - each slot the row makes (`rs1` at `ts` when read and not x0, `rs2` at
//!   `ts + 1` likewise, `rd` at `ts + 2` when written and not x0) takes
//!   [`in_row_prev`] of its old timestamp;
//! - `rd`'s old value is the row's exception if the list names its cycle,
//!   else the value of the access at `o = regs_ts[2]`: cycle `j = (o − 4) / 4`,
//!   slot `(o − 4) % 4`, from `j`'s log (`src1_val`, `src2_val` or `dst_val`
//!   by slot), where `j` lies in the chunk at or before `c`, is no ECALL, and
//!   accesses `rd` at that slot.
//!
//! Who lists an exception ([`needs_exception`]): every row with an `rd` slot
//! whose old access lies before its window's first cycle or its chunk's, or
//! in an ECALL cycle. That holds everything the rule above cannot read, so a
//! list may name more rows than it must; the derivation refuses a row it can
//! neither read nor find.

use std::ops::Range;
use std::sync::{Arc, OnceLock};

use executor::vm::history::{HistExtra, WindowHistory};
use executor::vm::logs::Log;
use multilinear::records::{RegisterStamps, SideException};

use super::super::in_row_prev;
use super::Segments;
use crate::tables::cpu::CpuSide;
use crate::tables::decode::DecodeTable;

/// The knob of the trace's parts the card builds from the executor's
/// records: unset, empty, `0` or `off` builds none of them here (today's
/// trace); `rows` derives the in-row CPU rows' side records on the card from
/// the history ([`super::WindowedTraceBuilder::card_derives_sides`]), and
/// MEMW_A's rows from the windows' accesses
/// ([`super::WindowedTraceBuilder::card_derives_memw_a`]). The tables are the
/// same either way.
pub const CARD_TRACE_KNOB: &str = "LAMBDA_VM_CARD_TRACE";

/// [`CARD_TRACE_KNOB`]'s reading of `word`: whether `rows` is on. Any other
/// word is refused.
pub(crate) fn card_trace_setting(word: Option<&str>) -> Result<bool, String> {
    match word.map(str::trim) {
        None | Some("") | Some("0") | Some("off") => Ok(false),
        Some("rows") => Ok(true),
        Some(other) => Err(format!("{CARD_TRACE_KNOB}={other}: expected off or rows")),
    }
}

/// [`card_trace_setting`] of the process's environment.
pub(crate) fn card_trace_rows() -> Result<bool, String> {
    card_trace_setting(std::env::var(CARD_TRACE_KNOB).ok().as_deref())
}

/// A CPU chunk's history records ([`HistExtra`], one a cycle): pieces of the
/// histories of the windows it was cut from, shared and never copied.
#[derive(Clone, Default)]
pub(crate) struct HistoryParts {
    parts: Vec<(Arc<WindowHistory>, Range<usize>)>,
}

impl HistoryParts {
    /// Cycles `range` of `history`'s window (indices into its records).
    pub(crate) fn push(&mut self, history: Arc<WindowHistory>, range: Range<usize>) {
        if !range.is_empty() {
            self.parts.push((history, range));
        }
    }

    /// The records of each part, in order.
    pub(crate) fn slices(&self) -> Vec<&[HistExtra]> {
        self.parts
            .iter()
            .map(|(history, range)| &history.extras[range.clone()])
            .collect()
    }

    pub(crate) fn len(&self) -> usize {
        self.parts.iter().map(|(_, range)| range.len()).sum()
    }

    /// The bytes they take as the card reads them.
    pub(crate) fn bytes(&self) -> usize {
        self.len() * std::mem::size_of::<HistExtra>()
    }
}

/// A CPU chunk's side records as the card derives them: its history records,
/// its exceptions (ascending by cycle), and the side records the host
/// derives from them for every host reader, once, on the first ask.
pub(crate) struct HistorySides {
    pub(crate) extras: HistoryParts,
    pub(crate) exceptions: Vec<SideException>,
    derived: OnceLock<Option<Segments<CpuSide>>>,
}

impl HistorySides {
    pub(crate) fn new(extras: HistoryParts, exceptions: Vec<SideException>) -> Self {
        Self {
            extras,
            exceptions,
            derived: OnceLock::new(),
        }
    }

    /// The chunk's side records as the host derives them
    /// ([`cpu_sides_from_history`]) from its logs, from `first_cycle` on;
    /// `None` where a row breaks a rule (the chunk then reads as no table its
    /// AIR takes, which the verifier refuses).
    pub(crate) fn derived(
        &self,
        logs: &Segments<Log>,
        first_cycle: u64,
        decode: &DecodeTable,
    ) -> Option<&Segments<CpuSide>> {
        self.derived
            .get_or_init(|| {
                cpu_sides_from_history(
                    &logs.slices(),
                    &self.extras.slices(),
                    first_cycle,
                    decode,
                    &[&self.exceptions],
                )
                .map(Segments::one)
                .ok()
            })
            .as_ref()
    }
}

/// [`HistExtra`]s as the card reads them ([`RegisterStamps`]): the same
/// bytes.
pub(crate) fn card_stamps(extras: &[HistExtra]) -> &[RegisterStamps] {
    use std::mem::{align_of, offset_of, size_of};
    const _: () = {
        assert!(size_of::<HistExtra>() == size_of::<RegisterStamps>());
        assert!(align_of::<HistExtra>() == align_of::<RegisterStamps>());
        assert!(offset_of!(HistExtra, regs_ts) == offset_of!(RegisterStamps, regs_ts));
        assert!(offset_of!(HistExtra, side) == offset_of!(RegisterStamps, side));
    };
    // SAFETY: both are `#[repr(C)]` with the same `u32` fields in the same
    // order (asserted above).
    unsafe { std::slice::from_raw_parts(extras.as_ptr().cast(), extras.len()) }
}

/// The access at timestamp `o` (cycle `j`, slot `sj`, `sj < 3`) as the rule
/// reads it: the register it touches at that slot, and its value there; or
/// `None` where the cycle makes no such access.
fn access_at(decode: &DecodeTable, log: &Log, sj: u64) -> Option<(u8, u64)> {
    let row = decode.row(log.current_pc)?;
    let f = &decode.entry(row).fields;
    if f.ecall {
        return None;
    }
    match sj {
        0 if f.read_register1 => Some((f.rs1, log.src1_val)),
        1 if f.read_register2 => Some((f.rs2, log.src2_val)),
        2 if f.write_register => Some((f.rd, log.dst_val)),
        _ => None,
    }
}

/// [`in_row_prev`] with its preconditions held: `old < slot_ts`, both below
/// 2^32 (the clock's bound in history mode).
fn prev(slot_ts: u64, old: u32) -> Result<u32, String> {
    if u64::from(old) >= slot_ts || slot_ts >> 32 != 0 {
        return Err(format!(
            "a slot at timestamp {slot_ts} with an old timestamp {old}"
        ));
    }
    Ok(in_row_prev(slot_ts, u64::from(old)) as u32)
}

/// The side records of the CPU rows of `logs` (cycles `first_cycle…`, in
/// pieces), from their history `extras` (one a log, in pieces of their own)
/// and `exceptions` (ascending by cycle): what the walk writes for the same
/// rows. Refused where a row breaks a rule (see the module docs).
pub(crate) fn cpu_sides_from_history(
    logs: &[&[Log]],
    extras: &[&[HistExtra]],
    first_cycle: u64,
    decode: &DecodeTable,
    exceptions: &[&[SideException]],
) -> Result<Vec<CpuSide>, String> {
    let logs: Vec<&Log> = logs.iter().flat_map(|piece| piece.iter()).collect();
    let extras: Vec<&HistExtra> = extras.iter().flat_map(|piece| piece.iter()).collect();
    let exceptions: Vec<&SideException> =
        exceptions.iter().flat_map(|piece| piece.iter()).collect();
    if extras.len() != logs.len() {
        return Err(format!(
            "{} logs and {} history records",
            logs.len(),
            extras.len()
        ));
    }
    if exceptions.windows(2).any(|w| w[0].cycle >= w[1].cycle) {
        return Err("the side exceptions are not ascending by cycle".to_string());
    }
    let mut sides = Vec::with_capacity(logs.len());
    for (r, (log, extra)) in logs.iter().zip(&extras).enumerate() {
        let c = first_cycle + r as u64;
        let refuse = |why: &str| format!("cycle {c}: {why}");
        let row = decode
            .row(log.current_pc)
            .ok_or_else(|| refuse("no DECODE row"))?;
        let f = &decode.entry(row).fields;
        let mut side = CpuSide::default();
        if f.word_instr {
            sides.push(side);
            continue;
        }
        let ts = 4 * c + 4;
        if f.read_register1 && f.rs1 != 0 {
            side.prev_ts[0] = prev(ts, extra.regs_ts[0]).map_err(|e| refuse(&e))?;
        }
        if f.read_register2 && f.rs2 != 0 {
            side.prev_ts[1] = prev(ts + 1, extra.regs_ts[1]).map_err(|e| refuse(&e))?;
        }
        if f.write_register && f.rd != 0 {
            side.prev_ts[2] = prev(ts + 2, extra.regs_ts[2]).map_err(|e| refuse(&e))?;
            let listed = u32::try_from(c)
                .ok()
                .and_then(|cycle| exceptions.binary_search_by_key(&cycle, |x| x.cycle).ok())
                .map(|at| exceptions[at].old_rd);
            side.old_rd = match listed {
                Some(old) => old,
                None => {
                    let o = u64::from(extra.regs_ts[2]);
                    if o < 4 * first_cycle + 4 {
                        return Err(refuse("rd's old access lies before the chunk, unlisted"));
                    }
                    let (j, sj) = ((o - 4) / 4, (o - 4) % 4);
                    if j > c || sj >= 3 {
                        return Err(refuse("rd's old timestamp names no earlier slot"));
                    }
                    let lj = logs[(j - first_cycle) as usize];
                    match access_at(decode, lj, sj) {
                        Some((reg, value)) if reg == f.rd => [value as u32, (value >> 32) as u32],
                        _ => {
                            return Err(refuse(
                                "rd's old access is not rd's in its cycle's log, unlisted",
                            ));
                        }
                    }
                }
            };
        }
        sides.push(side);
    }
    Ok(sides)
}

/// Whether the walk must list the row of cycle `cycle` (an `rd` write whose
/// old access is at `old_ts`) as a [`SideException`]: its old access lies
/// before `floor` (the first cycle of its window, or of its CPU chunk if
/// later), or in a cycle of `window_logs` (the window's logs, from
/// `window_first`) that is an ECALL. What the derivation cannot read is always
/// listed; a few rows it could read may be too.
pub(crate) fn needs_exception(
    cycle: u64,
    old_ts: u32,
    floor: u64,
    window_first: u64,
    window_logs: &[Log],
    decode: &DecodeTable,
) -> bool {
    let o = u64::from(old_ts);
    if o < 4 * floor.max(window_first) + 4 {
        return true;
    }
    let (j, sj) = ((o - 4) / 4, (o - 4) % 4);
    if j > cycle || sj >= 3 {
        return true;
    }
    let Some(lj) = window_logs.get((j - window_first) as usize) else {
        return true;
    };
    decode
        .row(lj.current_pc)
        .is_none_or(|row| decode.entry(row).fields.ecall)
}

/// The first cycle of the CPU chunk of `chunk_rows` cycles that holds
/// `cycle` (CPU chunks are cut at multiples of their height).
pub(crate) fn chunk_first(cycle: u64, chunk_rows: u64) -> u64 {
    cycle - cycle % chunk_rows.max(1)
}

/// The exceptions a window's walk would list, from what it wrote: each row
/// of `sides` (the walk's, one a log of `logs`, the window's from
/// `window_first`) with an `rd` slot that [`needs_exception`] names, with the
/// walk's own old value. Ascending by cycle. (The walk lists them batch by
/// batch, [`list_exceptions`]; the tests, from a whole window.)
#[cfg(test)]
pub(crate) fn exceptions_of_walk(
    logs: &[Log],
    extras: &[HistExtra],
    sides: &[CpuSide],
    window_first: u64,
    chunk_rows: u64,
    decode: &DecodeTable,
) -> Result<Vec<SideException>, String> {
    let mut out = Vec::new();
    list_exceptions(
        logs,
        extras,
        sides,
        (window_first, 0),
        chunk_rows,
        decode,
        &mut out,
    )?;
    Ok(out)
}

/// [`exceptions_of_walk`] of the side records of a part of a window, its
/// cycles `window_first + from…` (`from` into the window's `logs` and
/// `extras`), appended to `out`: what the walk lists batch by batch, writing
/// no side record (the walk's hook, [`super::WindowedTraceBuilder::card_derives_sides`]).
pub(crate) fn list_exceptions(
    logs: &[Log],
    extras: &[HistExtra],
    sides: &[CpuSide],
    (window_first, from): (u64, usize),
    chunk_rows: u64,
    decode: &DecodeTable,
    out: &mut Vec<SideException>,
) -> Result<(), String> {
    if logs.len() != extras.len() || from + sides.len() > logs.len() {
        return Err(format!(
            "{} logs, {} history records, {} side records from {from}",
            logs.len(),
            extras.len(),
            sides.len()
        ));
    }
    for (k, side) in sides.iter().enumerate() {
        let i = from + k;
        let (log, extra) = (&logs[i], &extras[i]);
        let cycle = window_first + i as u64;
        let Some(row) = decode.row(log.current_pc) else {
            continue;
        };
        let f = &decode.entry(row).fields;
        if f.word_instr || !f.write_register || f.rd == 0 {
            continue;
        }
        let floor = window_first.max(chunk_first(cycle, chunk_rows));
        if needs_exception(cycle, extra.regs_ts[2], floor, window_first, logs, decode) {
            let cycle = u32::try_from(cycle).map_err(|_| format!("cycle {cycle} past u32"))?;
            out.push(SideException {
                cycle,
                old_rd: side.old_rd,
            });
        }
    }
    Ok(())
}

/// The walk's hook under [`super::WindowedTraceBuilder::card_derives_sides`]:
/// the window's history records, the CPU chunks' height, and where the
/// window's exceptions go. With it the walk lists each batch's exceptions and
/// keeps no side record.
pub(crate) struct SideHook<'a> {
    pub(crate) extras: &'a [HistExtra],
    pub(crate) chunk_rows: u64,
    pub(crate) out: &'a mut Vec<SideException>,
}

/// The windows' history records not yet in a handed-out CPU chunk, in step
/// with the logs' tail ([`super::WindowedTraceBuilder::card_derives_sides`]):
/// each window's history (shared, the front's first `start` records gone) and
/// the exceptions its walk listed, ascending by cycle.
#[derive(Default)]
pub(crate) struct HistoryTail {
    parts: std::collections::VecDeque<Arc<WindowHistory>>,
    start: usize,
    len: usize,
    exceptions: std::collections::VecDeque<SideException>,
    /// The first cycle not handed out.
    next: u64,
}

impl HistoryTail {
    /// A window's history and its exceptions, after the windows before it.
    pub(crate) fn push(&mut self, history: Arc<WindowHistory>, exceptions: Vec<SideException>) {
        self.len += history.extras.len();
        self.exceptions.extend(exceptions);
        if !history.extras.is_empty() {
            self.parts.push_back(history);
        }
    }

    /// The cycles not handed out, and the first of them.
    pub(crate) fn pending(&self) -> (usize, u64) {
        (self.len, self.next)
    }

    /// The next `n` cycles' history records (at most `len`), as the parts
    /// they lie in, and their exceptions.
    pub(crate) fn take(&mut self, n: usize) -> (HistoryParts, Vec<SideException>) {
        let n = n.min(self.len);
        self.len -= n;
        let mut parts = HistoryParts::default();
        let mut want = n;
        while want > 0 {
            let Some(front) = self.parts.front() else {
                break;
            };
            let available = front.extras.len() - self.start;
            if available <= want {
                let range = self.start..front.extras.len();
                if let Some(part) = self.parts.pop_front() {
                    parts.push(part, range);
                }
                self.start = 0;
                want -= available;
            } else {
                parts.push(Arc::clone(front), self.start..self.start + want);
                self.start += want;
                want = 0;
            }
        }
        let end = self.next + n as u64;
        let mut exceptions = Vec::new();
        while self
            .exceptions
            .front()
            .is_some_and(|x| u64::from(x.cycle) < end)
        {
            exceptions.extend(self.exceptions.pop_front());
        }
        self.next = end;
        (parts, exceptions)
    }
}
