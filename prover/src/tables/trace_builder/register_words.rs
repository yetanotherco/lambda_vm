//! MEMW_R's register rows as the 8-byte words the card builds them from
//! (`LAMBDA_VM_CARD_ROWS=words`): what a walker writes for each row, and the
//! host's decode of words back to the walk's rows from the logs. Every host
//! reader of a chunk's rows (its host fill, its records fill, its digest, the
//! finish's last rows, a regenerated chunk's check) goes through the decode,
//! so it reads the rows the walk would have written.
//!
//! A word is `timestamp | old_timestamp << 32` for a row the logs of its own
//! window give the rest of: its cycle is not an ECALL, its slot (0 `rs1`,
//! 1 `rs2`, 2 `rd`) matches read or write, and for a write the access its old
//! timestamp names lies in the same window, at slot 0, 1 or 2, in no ECALL
//! cycle. Every other row crosses whole, and its word is a marker: low half 0
//! (no timestamp is), high half the row's index among the run's whole rows,
//! set in run order by the accumulator ([`set_markers`]).

use std::sync::Arc;

use executor::vm::logs::Log;

use super::super::bitwise::BitwiseHistogram;
use super::super::cpu::CpuOperation;
use super::super::decode::DecodeTable;
use super::super::memw_register::{self, RegRow};
use crate::Error;

/// A window as its walker sees it for words: its first cycle, which of its
/// cycles are ECALLs (from its history), and the CPU tables' length in cycles
/// (a write's old access must lie in its row's window and CPU table, the logs
/// the card holds).
pub(crate) struct WordWindow {
    first: u64,
    len: u64,
    table: u64,
    ecall: Vec<u64>,
}

impl WordWindow {
    pub(crate) fn new(
        first: u64,
        len: usize,
        table: usize,
        ecalls: impl IntoIterator<Item = u64>,
    ) -> Self {
        let mut ecall = vec![0u64; len.div_ceil(64)];
        for c in ecalls {
            if let Some(at) = c.checked_sub(first).filter(|&at| at < len as u64) {
                ecall[(at / 64) as usize] |= 1 << (at % 64);
            }
        }
        Self {
            first,
            len: len as u64,
            table: table.max(1) as u64,
            ecall,
        }
    }

    /// The first cycle a write at `cycle` may read its old value from: its
    /// window's, or its CPU table's if later.
    fn floor(&self, cycle: u64) -> u64 {
        self.first.max(cycle - cycle % self.table)
    }

    /// Whether `cycle` is an ECALL; `None` outside the window.
    fn is_ecall(&self, cycle: u64) -> Option<bool> {
        let at = cycle.checked_sub(self.first).filter(|&at| at < self.len)?;
        Some(self.ecall[(at / 64) as usize] >> (at % 64) & 1 == 1)
    }
}

/// A test's mutants of the walker's rule and the decode, one bit each (0:
/// none); each must change a decoded row.
#[cfg(test)]
pub(crate) static FAULT_WORDS: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
/// The walker sends an ECALL cycle's rows as words.
pub(crate) const WORDS_FAULT_ECALL: u32 = 1;
/// The walker sends a write whose old timestamp lies before its window as a
/// word.
pub(crate) const WORDS_FAULT_BEFORE: u32 = 2;
/// The decode reads a write's old value as the written value of its old
/// timestamp's cycle, whatever the slot.
pub(crate) const WORDS_FAULT_DECODE_SLOT: u32 = 4;

#[cfg(test)]
fn fault(bit: u32) -> bool {
    FAULT_WORDS.load(std::sync::atomic::Ordering::Relaxed) & bit != 0
}
#[cfg(not(test))]
#[inline(always)]
fn fault(_: u32) -> bool {
    false
}

/// A window's register rows as a walker writes them in words mode.
pub(crate) struct WordState {
    window: WordWindow,
    pub(crate) words: Vec<u64>,
    pub(crate) whole: Vec<RegRow>,
    /// The markers' places in `words`, ascending.
    pub(crate) markers: Vec<u32>,
}

impl WordState {
    pub(crate) fn new(window: WordWindow, capacity: usize) -> Self {
        Self {
            window,
            words: Vec::with_capacity(capacity),
            whole: Vec::new(),
            markers: Vec::new(),
        }
    }

    /// Whether the logs of the row's own window give the rest of the row.
    #[inline]
    fn as_word(&self, timestamp: u64, old_ts: u64, is_read: bool) -> bool {
        if timestamp < 4 || timestamp > u64::from(u32::MAX) || old_ts >= timestamp {
            return false;
        }
        if timestamp - old_ts > 1 << 16 {
            return false;
        }
        let (cycle, slot) = ((timestamp - 4) / 4, (timestamp - 4) % 4);
        if slot == 3 || is_read == (slot == 2) {
            return false;
        }
        if self.window.is_ecall(cycle) != Some(false) && !fault(WORDS_FAULT_ECALL) {
            return false;
        }
        if !is_read {
            if old_ts < 4 * self.window.floor(cycle) + 4 {
                return fault(WORDS_FAULT_BEFORE) && old_ts >= 4;
            }
            if (old_ts - 4) % 4 == 3 {
                return false;
            }
            if self.window.is_ecall((old_ts - 4) / 4) != Some(false) && !fault(WORDS_FAULT_ECALL) {
                return false;
            }
        }
        true
    }

    /// The walk's row `row`, as a word or whole. Its old timestamp's high word
    /// is its timestamp's (MEMW_R's rows), so a timestamp under 2^32 gives the
    /// whole old timestamp.
    #[inline]
    pub(crate) fn push(&mut self, row: RegRow) {
        let r = &memw_register::card_rows(std::slice::from_ref(&row))[0];
        let (timestamp, old_ts) = (
            r.timestamp,
            u64::from(r.old_ts_lo) | (r.timestamp >> 32) << 32,
        );
        if self.as_word(timestamp, old_ts, r.is_read != 0) {
            self.words.push(timestamp | (old_ts & 0xFFFF_FFFF) << 32);
        } else {
            self.markers.push(self.words.len() as u32);
            self.words.push(0);
            self.whole.push(row);
        }
    }

    /// A row that crosses whole whatever the rule says.
    pub(crate) fn push_whole(&mut self, row: RegRow) {
        self.markers.push(self.words.len() as u32);
        self.words.push(0);
        self.whole.push(row);
    }

    /// The IsHalfword lookups of the rows from word `from` and whole row
    /// `from_whole` on (the ones a batch added), into `histogram`.
    pub(crate) fn count_bitwise_from(
        &self,
        from: usize,
        from_whole: usize,
        histogram: &mut BitwiseHistogram,
    ) {
        count_bitwise(&self.words[from..], &self.whole[from_whole..], histogram);
    }
}

/// The IsHalfword lookups of `words` (markers' rows from `whole`, in order).
pub(crate) fn count_bitwise(words: &[u64], whole: &[RegRow], histogram: &mut BitwiseHistogram) {
    let mut next = whole.iter();
    for &w in words {
        if w as u32 == 0 {
            if let Some(row) = next.next() {
                memw_register::collect_bitwise_from_memw_register(
                    std::slice::from_ref(row),
                    histogram,
                );
            }
        } else {
            memw_register::count_bitwise_lookup(w as u32, (w >> 32) as u32, histogram);
        }
    }
}

/// Sets a walked window's markers to their whole rows' run indices, from
/// `base` (the run's whole rows before this window's), in run order.
pub(crate) fn set_markers(state: &mut WordState, base: u64) {
    for (k, &at) in state.markers.iter().enumerate() {
        state.words[at as usize] = (base + k as u64) << 32;
    }
}

/// The walk's rows back from `words` (pieces in order), `whole` the rows its
/// markers name (pieces in order, the first at run index `whole_base`), from
/// the logs of the windows the rows lie in and the program's DECODE rows.
pub(crate) fn decode_rows(
    words: &[&[u64]],
    whole: &[&[RegRow]],
    whole_base: u64,
    logs: &[(u64, Arc<Vec<Log>>)],
    decode: &DecodeTable,
) -> Result<Vec<RegRow>, Error> {
    let whole: Vec<&RegRow> = whole.iter().flat_map(|p| p.iter()).collect();
    let n: usize = words.iter().map(|p| p.len()).sum();
    let mut rows = Vec::with_capacity(n);
    let mut ops: Vec<CpuOperation> = Vec::with_capacity(1);
    let mut cached: Option<u64> = None;
    let mut at = 0usize;
    let bad = |what: String| Error::Prover(format!("register words: {what}"));
    let log_of = |cycle: u64, at: &mut usize| -> Option<&Log> {
        if !logs
            .get(*at)
            .is_some_and(|(f, l)| *f <= cycle && cycle < f + l.len() as u64)
        {
            *at = logs
                .iter()
                .position(|(f, l)| *f <= cycle && cycle < f + l.len() as u64)?;
        }
        let (f, l) = &logs[*at];
        l.get((cycle - f) as usize)
    };
    for &w in words.iter().flat_map(|p| p.iter()) {
        if w as u32 == 0 {
            let k = (w >> 32)
                .checked_sub(whole_base)
                .ok_or_else(|| bad(format!("marker {} before the chunk's", w >> 32)))?;
            let row = whole
                .get(k as usize)
                .ok_or_else(|| bad(format!("marker {} past the chunk's whole rows", w >> 32)))?;
            rows.push(**row);
            continue;
        }
        let (ts, old) = (w & 0xFFFF_FFFF, w >> 32);
        let (cycle, slot) = ((ts - 4) / 4, (ts - 4) % 4);
        if cached != Some(cycle) {
            let log = log_of(cycle, &mut at)
                .ok_or_else(|| bad(format!("cycle {cycle} not held")))?
                .clone();
            ops.clear();
            super::collect_cpu_ops_into(
                std::slice::from_ref(&log),
                decode,
                cycle as usize,
                &mut ops,
            )?;
            cached = Some(cycle);
        }
        let op = &ops[0];
        let f = decode.op(op).decode.fields;
        let (reg, value) = match slot {
            0 => (f.rs1, op.rv1),
            1 => (f.rs2, op.rv2),
            2 => (f.rd, op.rvd),
            _ => return Err(bad(format!("timestamp {ts} at slot 3"))),
        };
        let before = if slot < 2 {
            value
        } else {
            let (j, sj) = ((old - 4) / 4, (old - 4) % 4);
            let log = log_of(j, &mut at).ok_or_else(|| bad(format!("old cycle {j} not held")))?;
            if fault(WORDS_FAULT_DECODE_SLOT) {
                log.dst_val
            } else {
                match sj {
                    0 => log.src1_val,
                    1 => log.src2_val,
                    2 => log.dst_val,
                    _ => return Err(bad(format!("old timestamp {old} at slot 3"))),
                }
            }
        };
        rows.push(RegRow::new(
            2 * u64::from(reg),
            ts,
            value as u32,
            (value >> 32) as u32,
            before as u32,
            (before >> 32) as u32,
            old,
            slot < 2,
        ));
    }
    Ok(rows)
}

/// The first and the last row's cycles of a chunk's words (markers' from
/// their whole rows, the first at run index `whole_base`).
pub(crate) fn cycle_span(words: &[&[u64]], whole: &[&[RegRow]], whole_base: u64) -> (u64, u64) {
    let whole: Vec<&RegRow> = whole.iter().flat_map(|p| p.iter()).collect();
    let cycle = |w: u64| -> u64 {
        let ts = if w as u32 == 0 {
            (w >> 32)
                .checked_sub(whole_base)
                .and_then(|k| whole.get(k as usize))
                .map_or(4, |r| {
                    memw_register::card_rows(std::slice::from_ref(*r))[0].timestamp
                })
        } else {
            w & 0xFFFF_FFFF
        };
        ts.saturating_sub(4) / 4
    };
    let first = words
        .iter()
        .find_map(|p| p.first())
        .copied()
        .map_or(0, cycle);
    let last = words
        .iter()
        .rev()
        .find_map(|p| p.last())
        .copied()
        .map_or(0, cycle);
    (first, last)
}
