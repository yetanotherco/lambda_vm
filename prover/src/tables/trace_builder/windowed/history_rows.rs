//! MEMW_A's rows derived from the executor's history instead of made by the
//! walk: the host reference the card's `derive_memw_a_from_cache`
//! (`crypto/math-cuda/kernels/card_rows.cu`) restates rule for rule, and that
//! every host reader of a MEMW_A chunk held this way goes through.
//!
//! A chunk is held as its rows' [`AlignedRef`]s, in walk order, each window's
//! in a part of its own beside that window's logs, history and whole rows
//! ([`RefWindow`]). A ref is one of:
//! - a load's or store's row: its cycle `c` and `at`, the index of the
//!   cycle's access in the window's [`WindowHistory::accesses`] (the cycle's
//!   [`HistExtra::side`](executor::vm::history::HistExtra)); the row is
//!   [`aligned_row_from_access`] of `c`'s log and that access;
//! - an ECALL cycle's row: [`ALIGNED_ECALL`] and the cycle, and `at`, the
//!   row's index in the window's whole rows (the walk's, copied).
//!
//! The rule for the row of cycle `c` (timestamp `ts = 4c + 4`), from the walk
//! (`collect_load_op_from_cpu`, `collect_store_op_from_cpu`, `classify_memw`,
//! `AlignedRow::from_memw`):
//! - the log's pc has a DECODE row, a load or a store, no ECALL;
//! - the address is the CPU op's `res` and the access's;
//! - the old timestamps are the access's eight bytes': its shared one, or its
//!   entry of the mixed list;
//! - a load: value and old are the loaded `width` bytes (`rvd`'s) then the
//!   sign or zero fill to eight; a store: value is `rv2`'s eight bytes, old
//!   the access's bytes;
//! - the access routes to MEMW_A ([`classify_memw`]: aligned to its width,
//!   one old timestamp over its width's bytes, and in the limbs' form a step
//!   that fits them). A row that does not is refused: a ref names only MEMW_A
//!   rows.

use std::sync::Arc;

use executor::vm::history::{MemAccess, NO_ACCESS, WindowHistory};
use executor::vm::logs::Log;
use multilinear::records::{ALIGNED_ECALL, AlignedRef, MemoryAccess};

use super::super::{MemwRoute, classify_memw};
use crate::tables::cpu::CpuOperation;
use crate::tables::decode::DecodeTable;
use crate::tables::memw::MemwOperation;
use crate::tables::memw_aligned::{AlignedRow, MemwAlignedOrder};

/// [`MemAccess`]es as the card reads them ([`MemoryAccess`]): the same bytes.
pub(crate) fn card_accesses(accesses: &[MemAccess]) -> &[MemoryAccess] {
    use std::mem::{align_of, offset_of, size_of};
    const _: () = {
        assert!(size_of::<MemAccess>() == size_of::<MemoryAccess>());
        assert!(align_of::<MemAccess>() == align_of::<MemoryAccess>());
        assert!(offset_of!(MemAccess, addr) == offset_of!(MemoryAccess, addr));
        assert!(offset_of!(MemAccess, old) == offset_of!(MemoryAccess, old));
        assert!(offset_of!(MemAccess, ts) == offset_of!(MemoryAccess, ts));
        assert!(offset_of!(MemAccess, mixed) == offset_of!(MemoryAccess, mixed));
        assert!(NO_ACCESS == multilinear::records::NO_MIXED);
    };
    // SAFETY: both are `#[repr(C)]` with the same fields in the same order
    // (asserted above).
    unsafe { std::slice::from_raw_parts(accesses.as_ptr().cast(), accesses.len()) }
}

/// The MEMW_A row of cycle `cycle`'s load or store, from its `log` and its
/// `access` (with `mixed`, its window's mixed list), in the run's `order`:
/// the row the walk makes (see the module docs). Refused where the cycle is
/// no load or store, its access is not its address's, or the access does not
/// route to MEMW_A.
pub(crate) fn aligned_row_from_access(
    cycle: u64,
    log: &Log,
    decode: &DecodeTable,
    access: &MemAccess,
    mixed: &[[u32; 8]],
    order: MemwAlignedOrder,
) -> Result<AlignedRow, String> {
    let refuse = |why: &str| format!("cycle {cycle}: {why}");
    // The clock's bound in history mode: every timestamp fits a `u32`.
    if cycle > executor::vm::history::HISTORY_MAX_CYCLES {
        return Err(refuse("past the history's clock"));
    }
    let row = decode
        .row(log.current_pc)
        .ok_or_else(|| refuse("no DECODE row"))?;
    let entry = decode.entry(row);
    let f = &entry.fields;
    if f.ecall || !(f.is_load() || f.is_store()) {
        return Err(refuse("no load or store"));
    }
    let ts = 4 * cycle + 4;
    let op = CpuOperation::from_log(log, ts, entry, row);
    let op = decode.op(&op);
    let address = op.res;
    if access.addr != address {
        return Err(refuse("the access is not at the op's address"));
    }
    let old_ts: [u64; 8] = if access.mixed == NO_ACCESS {
        [u64::from(access.ts); 8]
    } else {
        let stamps = mixed
            .get(access.mixed as usize)
            .ok_or_else(|| refuse("the access's mixed entry is not in its window's list"))?;
        stamps.map(u64::from)
    };
    let width = f.mem_bytes();
    let bytes = |word: u64| -> [u32; 8] { std::array::from_fn(|k| (word >> (8 * k)) as u8 as u32) };
    let memw = if f.is_load() {
        let mut res = bytes(op.rvd);
        if width < 8 {
            let sign = (res[width - 1] >> 7) & 1;
            let fill = if f.mem_signed() && sign == 1 { 0xFF } else { 0 };
            res[width..].iter_mut().for_each(|b| *b = fill);
        }
        MemwOperation::new(false, address, res, ts, width as u8, true).with_old(res, old_ts)
    } else {
        MemwOperation::new(false, address, bytes(op.rv2), ts, width as u8, false)
            .with_old(bytes(access.old), old_ts)
    };
    if classify_memw(&memw, order) != MemwRoute::Aligned {
        return Err(refuse("the access is not MEMW_A's"));
    }
    AlignedRow::from_memw(&memw, order).map_err(|e| refuse(&e))
}

/// A window as a MEMW_A chunk's rows read it: its first cycle, its logs, its
/// history (its accesses and mixed list), and the rows its ECALL cycles
/// made, whole.
pub(crate) struct RefWindow {
    pub(crate) first: u64,
    pub(crate) logs: Arc<Vec<Log>>,
    pub(crate) history: Arc<WindowHistory>,
    pub(crate) whole: Vec<AlignedRow>,
}

impl RefWindow {
    /// Its cycles.
    pub(crate) fn cycles(&self) -> std::ops::Range<u64> {
        self.first..self.first + self.logs.len() as u64
    }
}

/// The row `r` names in `window`, in the run's `order` (see the module
/// docs).
pub(crate) fn aligned_row_of_ref(
    r: AlignedRef,
    window: &RefWindow,
    decode: &DecodeTable,
    order: MemwAlignedOrder,
) -> Result<AlignedRow, String> {
    let cycle = u64::from(r.cycle & !ALIGNED_ECALL);
    if !window.cycles().contains(&cycle) {
        return Err(format!(
            "a MEMW_A row of cycle {cycle} outside its window from {}",
            window.first
        ));
    }
    if r.cycle & ALIGNED_ECALL != 0 {
        return window
            .whole
            .get(r.at as usize)
            .copied()
            .ok_or_else(|| format!("cycle {cycle}: ECALL row {} not in its window's", r.at));
    }
    let access = window
        .history
        .accesses
        .get(r.at as usize)
        .ok_or_else(|| format!("cycle {cycle}: access {} not in its window's", r.at))?;
    let log = &window.logs[(cycle - window.first) as usize];
    aligned_row_from_access(cycle, log, decode, access, &window.history.mixed, order)
}

/// The rows of a chunk's parts (each its refs and the window they lie in),
/// in order. Refused where a row breaks a rule.
pub(crate) fn aligned_rows_from_refs(
    parts: &[(&[AlignedRef], &RefWindow)],
    decode: &DecodeTable,
    order: MemwAlignedOrder,
) -> Result<Vec<AlignedRow>, String> {
    let n = parts.iter().map(|(refs, _)| refs.len()).sum();
    let mut rows = Vec::with_capacity(n);
    for (refs, window) in parts {
        for &r in refs.iter() {
            rows.push(aligned_row_of_ref(r, window, decode, order)?);
        }
    }
    Ok(rows)
}

/// A window's MEMW_A rows, as the walk made them, as refs and whole rows:
/// each row of a load or store as its cycle and access, each row of an
/// ECALL cycle whole. Refused where a row is of no cycle of the window, of a
/// cycle with no access, or two rows share a load's or store's cycle, or a
/// row is a register access's (no ref names one: the in-row register format
/// sends none to MEMW_A).
pub(crate) fn aligned_refs_of_rows(
    rows: &[AlignedRow],
    first: u64,
    history: &WindowHistory,
) -> Result<(Vec<AlignedRef>, Vec<AlignedRow>), String> {
    use executor::vm::history::ECALL;
    let mut refs = Vec::with_capacity(rows.len());
    let mut whole = Vec::new();
    let mut last_regular = None;
    for row in rows {
        let ts = row.timestamp();
        let cycle = (ts.checked_sub(4).ok_or("a MEMW_A row at timestamp 0")?) / 4;
        let i = cycle
            .checked_sub(first)
            .map(|i| i as usize)
            .filter(|&i| i < history.extras.len())
            .ok_or_else(|| format!("a MEMW_A row of cycle {cycle} outside its window"))?;
        let c = u32::try_from(cycle).map_err(|_| format!("cycle {cycle} past u32"))?;
        if c & ALIGNED_ECALL != 0 {
            return Err(format!("cycle {cycle} past the refs' range"));
        }
        let side = history.extras[i].side;
        if side != NO_ACCESS && side & ECALL != 0 {
            let at = u32::try_from(whole.len()).map_err(|_| "too many ECALL rows")?;
            refs.push(AlignedRef {
                cycle: c | ALIGNED_ECALL,
                at,
            });
            whole.push(*row);
            continue;
        }
        if row.is_register() {
            return Err(format!("cycle {cycle}: a register access's MEMW_A row"));
        }
        if side == NO_ACCESS {
            return Err(format!("cycle {cycle}: a MEMW_A row with no access"));
        }
        if last_regular == Some(cycle) || ts != 4 * cycle + 4 {
            return Err(format!("cycle {cycle}: a second or a later MEMW_A row"));
        }
        last_regular = Some(cycle);
        refs.push(AlignedRef { cycle: c, at: side });
    }
    Ok((refs, whole))
}

/// The windows' MEMW_A refs not yet in a handed-out chunk
/// ([`super::WindowedTraceBuilder::card_derives_memw_a`]), each window's
/// (shared) beside the window it lies in, in order: the registry a chunk's
/// host readers rebuild its rows from.
pub(crate) struct RefTail {
    parts: std::collections::VecDeque<(Arc<Vec<AlignedRef>>, Arc<RefWindow>)>,
    /// The front part's refs already handed out.
    start: usize,
    len: usize,
    pub(crate) decode: Arc<DecodeTable>,
    pub(crate) order: MemwAlignedOrder,
}

impl RefTail {
    pub(crate) fn new(decode: Arc<DecodeTable>, order: MemwAlignedOrder) -> Self {
        Self {
            parts: std::collections::VecDeque::new(),
            start: 0,
            len: 0,
            decode,
            order,
        }
    }

    /// A window's refs, after the windows before it.
    pub(crate) fn push(&mut self, refs: Vec<AlignedRef>, window: Arc<RefWindow>) {
        if !refs.is_empty() {
            self.len += refs.len();
            self.parts.push_back((Arc::new(refs), window));
        }
    }

    /// The refs not handed out.
    pub(crate) fn len(&self) -> usize {
        self.len
    }

    /// The next `n` refs (at most `len`), as the window parts they lie in,
    /// and each part's window.
    pub(crate) fn take(&mut self, n: usize) -> (super::Segments<AlignedRef>, Vec<Arc<RefWindow>>) {
        let mut want = n.min(self.len);
        self.len -= want;
        let (mut parts, mut windows) = (Vec::new(), Vec::new());
        while want > 0 {
            let Some((refs, window)) = self.parts.front() else {
                break;
            };
            let available = refs.len() - self.start;
            if available <= want {
                let range = self.start..refs.len();
                if let Some((refs, window)) = self.parts.pop_front() {
                    parts.push((refs, range));
                    windows.push(window);
                }
                self.start = 0;
                want -= available;
            } else {
                parts.push((Arc::clone(refs), self.start..self.start + want));
                windows.push(Arc::clone(window));
                self.start += want;
                want = 0;
            }
        }
        (super::Segments { parts }, windows)
    }

    /// Every ref not handed out, as its row derived on the host (the run's
    /// MEMW_A tail, or the rows the mode leaves behind as it ends), in order.
    pub(crate) fn take_rows(&mut self) -> Result<Vec<AlignedRow>, String> {
        let (refs, windows) = self.take(self.len);
        let slices = refs.slices();
        let parts: Vec<(&[AlignedRef], &RefWindow)> = slices
            .iter()
            .zip(&windows)
            .map(|(refs, window)| (*refs, &**window))
            .collect();
        aligned_rows_from_refs(&parts, &self.decode, self.order)
    }

    /// The bytes its refs take on the heap (the windows they lie in are the
    /// logs' and the history's, held elsewhere too).
    pub(crate) fn heap_bytes(&self) -> usize {
        self.parts
            .iter()
            .map(|(refs, window)| {
                refs.capacity() * std::mem::size_of::<AlignedRef>()
                    + window.whole.capacity() * std::mem::size_of::<AlignedRow>()
            })
            .sum()
    }
}
