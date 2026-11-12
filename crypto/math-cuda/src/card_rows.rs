//! Rank 1, stage S1 (`thoughts/zf/gap3/D-CARDROWS-FABLE.md` §7, with probe
//! K0b's lessons, `I-CARDPROBE.md`): a group's upload under
//! `LAMBDA_VM_CARD_ROWS` (read once; unset, `0` or `off` is today's path,
//! byte for byte and call for call).
//!
//! - `order`: every copy of the group first, then every kernel, then the one
//!   sync and the one check readback the upload already makes. Today each
//!   records table's copy waits, in stream order, for the table before it to
//!   be filled; beside a running commit each such wait costs up to what the
//!   commit has queued ahead (k002: today's upload took 2.03 commit periods).
//! - `derive`: `order`, and the card keeps the CPU chunks' logs it receives
//!   (an LRU cache of whole CPU tables, `LAMBDA_VM_CARD_ROWS_CACHE_MIB`,
//!   default 1024, each entry promised to the backend's ledger), fills a CPU
//!   table it already holds without copying its logs again, and derives a
//!   LOAD or STORE table from the cached logs (`derive_rows_from_cache`,
//!   `kernels/card_rows.cu`) instead of copying its records: the table's rows'
//!   cycles go up instead (4 bytes a row against 32), read from the records'
//!   timestamps. One launch for a group's LOAD tables and one for its STORE
//!   tables. A table whose cycles the cache does not hold, after this group's
//!   CPU tables are in it, goes up as records, as today.
//!
//! Nothing the verifier reads moves: the same tables, the same words. A row
//! whose cycle is not held, or whose log is not that family's op, is counted
//! into the upload's check, which declines the group (its readers take the
//! host's columns, as for a CPU log with no DECODE row). The cache is
//! emptied at a block's start and end ([`clear`]) and only ever holds a
//! table's logs while every upload that read them has synchronized: uploads
//! under this knob take the cache's lock for their whole length.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use cudarc::driver::{
    CudaFunction, CudaModule, CudaSlice, DevicePtr, DeviceRepr, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::Ptx;

use crate::columns::{DeviceColumns, TableUpload};
use crate::device::{DeviceReservation, alloc_or_trim, backend};
use crate::fill::{
    CardDecode, CpuLogsOnCard, FillChecks, LoadOpC, LogC, RecordsInput, Staged, StoreOpC,
};

const CARD_ROWS_CUBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/card_rows.cubin"));
const ROW_BLOCK: u32 = 256;

/// The knob's word.
pub const CARD_ROWS_ENV: &str = "LAMBDA_VM_CARD_ROWS";
/// The cache's budget in MiB.
pub const CARD_ROWS_CACHE_ENV: &str = "LAMBDA_VM_CARD_ROWS_CACHE_MIB";
const DEFAULT_CACHE_MIB: u64 = 1024;

/// What a group's upload does ([`CARD_ROWS_ENV`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// Today's upload.
    Off,
    /// Every copy, then every kernel.
    Order,
    /// [`Mode::Order`], CPU logs kept on the card, LOAD and STORE derived.
    Derive,
}

impl Mode {
    /// The knob's reading: unset, `0` or `off` is [`Mode::Off`]; `order`;
    /// `derive` or `1`. Anything else is refused.
    pub fn parse(word: Option<&str>) -> Result<Self, String> {
        match word.map(str::trim) {
            None | Some("") | Some("0") | Some("off") => Ok(Self::Off),
            Some("order") => Ok(Self::Order),
            Some("derive") | Some("1") => Ok(Self::Derive),
            Some(other) => Err(format!(
                "{CARD_ROWS_ENV}={other}: expected off, order or derive"
            )),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Order => "order",
            Self::Derive => "derive",
        }
    }
}

/// A test's override of the knob (0: none; 1 + the mode's index).
static OVERRIDE: AtomicU8 = AtomicU8::new(0);

/// The mode this process uploads in. An unknown word is reported once and
/// read as [`Mode::Off`]; the block's report line names the mode in force.
pub fn mode() -> Mode {
    match OVERRIDE.load(Ordering::Relaxed) {
        1 => return Mode::Off,
        2 => return Mode::Order,
        3 => return Mode::Derive,
        _ => {}
    }
    static FROM_ENV: OnceLock<Mode> = OnceLock::new();
    *FROM_ENV.get_or_init(
        || match Mode::parse(std::env::var(CARD_ROWS_ENV).ok().as_deref()) {
            Ok(mode) => mode,
            Err(e) => {
                eprintln!("[gpu] {e}; the upload stays off");
                Mode::Off
            }
        },
    )
}

/// A test's mode, over the knob's (`None` gives the knob back).
#[doc(hidden)]
pub fn set_mode_override(mode: Option<Mode>) {
    let v = match mode {
        None => 0,
        Some(Mode::Off) => 1,
        Some(Mode::Order) => 2,
        Some(Mode::Derive) => 3,
    };
    OVERRIDE.store(v, Ordering::Relaxed);
}

/// A card test's mutant: each derived row's cycle read this many cycles late
/// (0: none). The test must see other words, or the group declined.
#[doc(hidden)]
pub static FAULT_CYCLE_SHIFT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

fn budget_bytes() -> u64 {
    static BYTES: OnceLock<u64> = OnceLock::new();
    *BYTES.get_or_init(|| {
        std::env::var(CARD_ROWS_CACHE_ENV)
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .unwrap_or(DEFAULT_CACHE_MIB)
            << 20
    })
}

// ---------------------------------------------------------------------------
// The counters a block's report reads.
// ---------------------------------------------------------------------------

#[derive(Default)]
struct Counters {
    groups: AtomicU64,
    order_refused: AtomicU64,
    declined: AtomicU64,
    cpu_hits: AtomicU64,
    cpu_copies: AtomicU64,
    cpu_kept: AtomicU64,
    evicted: AtomicU64,
    derived_load: AtomicU64,
    derived_store: AtomicU64,
    records_load: AtomicU64,
    records_store: AtomicU64,
    list_bytes: AtomicU64,
    records_bytes_spared: AtomicU64,
    log_bytes_spared: AtomicU64,
}

static COUNTERS: OnceLock<Counters> = OnceLock::new();

fn counters() -> &'static Counters {
    COUNTERS.get_or_init(Counters::default)
}

/// What the uploads under the knob did since the process started.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Stats {
    /// Groups uploaded in the knob's order; refused that order (no room to
    /// hold every copy at once: uploaded as today); declined by a check.
    pub groups: u64,
    pub order_refused: u64,
    pub declined: u64,
    /// CPU tables filled from the cache's logs; whose logs were copied; kept
    /// after; entries let go to make room.
    pub cpu_hits: u64,
    pub cpu_copies: u64,
    pub cpu_kept: u64,
    pub evicted: u64,
    /// LOAD and STORE tables derived; sent as records.
    pub derived_load: u64,
    pub derived_store: u64,
    pub records_load: u64,
    pub records_store: u64,
    /// Bytes of cycle lists sent; of records not sent; of CPU logs not sent.
    pub list_bytes: u64,
    pub records_bytes_spared: u64,
    pub log_bytes_spared: u64,
}

pub fn stats() -> Stats {
    let c = counters();
    let l = |a: &AtomicU64| a.load(Ordering::Relaxed);
    Stats {
        groups: l(&c.groups),
        order_refused: l(&c.order_refused),
        declined: l(&c.declined),
        cpu_hits: l(&c.cpu_hits),
        cpu_copies: l(&c.cpu_copies),
        cpu_kept: l(&c.cpu_kept),
        evicted: l(&c.evicted),
        derived_load: l(&c.derived_load),
        derived_store: l(&c.derived_store),
        records_load: l(&c.records_load),
        records_store: l(&c.records_store),
        list_bytes: l(&c.list_bytes),
        records_bytes_spared: l(&c.records_bytes_spared),
        log_bytes_spared: l(&c.log_bytes_spared),
    }
}

fn add(counter: &AtomicU64, n: u64) {
    counter.fetch_add(n, Ordering::Relaxed);
}

// ---------------------------------------------------------------------------
// The cache of CPU tables' logs.
// ---------------------------------------------------------------------------

/// A CPU table's logs on the card: cycles `[first, first + n)`, copied from
/// the host buffer at `host` (its first piece's address): a table is served
/// from the cache only when its own logs lie there, so a copy of them that
/// differs (a test's fault, a regenerated window) is never taken for them.
struct Entry {
    n: usize,
    host: usize,
    logs: CudaSlice<LogC>,
    decode: Arc<CardDecode>,
    /// The upload that last read it.
    tick: u64,
    _room: DeviceReservation,
}

impl Entry {
    fn bytes(&self) -> u64 {
        (self.n * std::mem::size_of::<LogC>()) as u64
    }
}

#[derive(Default)]
struct Cache {
    /// By first cycle; ranges disjoint.
    entries: BTreeMap<u64, Entry>,
    tick: u64,
    bytes: u64,
}

static CACHE: Mutex<Option<Cache>> = Mutex::new(None);

/// Whether uploads keep CPU logs past their own group: only between a
/// block's [`begin`] and [`end`]. Outside one, a group still derives its
/// LOAD and STORE tables from its own CPU tables' logs, and keeps nothing.
static KEEPING: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// A block's prove starts: the cache emptied, the counters zeroed, and logs
/// kept from here to [`end`].
pub fn begin() {
    clear();
    KEEPING.store(true, Ordering::Relaxed);
}

/// A block's prove ended: nothing more is kept, and the cache is emptied.
pub fn end() {
    KEEPING.store(false, Ordering::Relaxed);
    let mut cache = lock();
    *cache = None;
}

fn lock() -> std::sync::MutexGuard<'static, Option<Cache>> {
    CACHE
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// The cache emptied and the counters zeroed: at a block's start and end, so
/// no prove reads another's logs and each block's report is its own. Its
/// entries' memory goes back in stream order.
pub fn clear() {
    let mut cache = lock();
    *cache = None;
    let c = counters();
    for counter in [
        &c.groups,
        &c.order_refused,
        &c.declined,
        &c.cpu_hits,
        &c.cpu_copies,
        &c.cpu_kept,
        &c.evicted,
        &c.derived_load,
        &c.derived_store,
        &c.records_load,
        &c.records_store,
        &c.list_bytes,
        &c.records_bytes_spared,
        &c.log_bytes_spared,
    ] {
        counter.store(0, Ordering::Relaxed);
    }
}

/// The block's report line (`BLOCK CARD ROWS: …`), from the counters and the
/// cache as they stand.
pub fn report_line() -> String {
    let s = stats();
    let (entries, bytes) = held();
    format!(
        "BLOCK CARD ROWS: mode {} · groups {} (order refused {}, declined {}) · CPU tables from the cache {}, copied {}, kept {}, let go {} · LOAD derived {} of {} · STORE derived {} of {} · cycle lists {} B · records not sent {} B · CPU logs not sent {} B · cache {} entries, {} B, budget {} B",
        mode().name(),
        s.groups,
        s.order_refused,
        s.declined,
        s.cpu_hits,
        s.cpu_copies,
        s.cpu_kept,
        s.evicted,
        s.derived_load,
        s.derived_load + s.records_load,
        s.derived_store,
        s.derived_store + s.records_store,
        s.list_bytes,
        s.records_bytes_spared,
        s.log_bytes_spared,
        entries,
        bytes,
        budget_bytes()
    )
}

/// Entries held and their bytes.
pub fn held() -> (usize, u64) {
    let cache = lock();
    cache
        .as_ref()
        .map_or((0, 0), |c| (c.entries.len(), c.bytes))
}

// ---------------------------------------------------------------------------
// Pure parts, tested on the host.
// ---------------------------------------------------------------------------

/// A record with the timestamp of the cycle that made it.
pub trait Timestamped {
    fn timestamp(&self) -> u64;
}

impl Timestamped for LoadOpC {
    fn timestamp(&self) -> u64 {
        self.timestamp
    }
}

impl Timestamped for StoreOpC {
    fn timestamp(&self) -> u64 {
        self.timestamp
    }
}

/// Each record's cycle (`timestamp = 4·cycle + 4`), in order, and the least
/// and the greatest; `None` for no record or a timestamp no cycle has.
pub fn cycles_of<T: Timestamped>(pieces: &[&[T]]) -> Option<(Vec<u32>, u64, u64)> {
    let n: usize = pieces.iter().map(|p| p.len()).sum();
    if n == 0 {
        return None;
    }
    let mut out = Vec::with_capacity(n);
    let (mut lo, mut hi) = (u64::MAX, 0u64);
    for piece in pieces {
        for record in piece.iter() {
            let ts = record.timestamp();
            if ts < 4 || ts % 4 != 0 {
                return None;
            }
            let cycle = ts / 4 - 1;
            out.push(u32::try_from(cycle).ok()?);
            lo = lo.min(cycle);
            hi = hi.max(cycle);
        }
    }
    Some((out, lo, hi))
}

/// Whether the ranges `held` (first cycle → length, disjoint) hold every
/// cycle of `[lo, hi]`.
pub fn covers(held: &BTreeMap<u64, u64>, lo: u64, hi: u64) -> bool {
    let mut next = lo;
    let start = held.range(..=lo).next_back().map_or(lo, |(&f, _)| f);
    for (&first, &n) in held.range(start..) {
        if first > next {
            return false;
        }
        next = next.max(first + n);
        if next > hi {
            return true;
        }
    }
    false
}

// ---------------------------------------------------------------------------
// The kernel.
// ---------------------------------------------------------------------------

/// `card_rows.cu`'s `CachedLogsC`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct CachedLogsC {
    first: u64,
    n: u64,
    logs: u64,
}

/// `card_rows.cu`'s `RowsTableC`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct RowsTableC {
    cycles: u64,
    rows: u64,
    height: u64,
    out: u64,
    before: u64,
}

const _: () = {
    assert!(std::mem::size_of::<CachedLogsC>() == 24);
    assert!(std::mem::size_of::<RowsTableC>() == 40);
};

// SAFETY: plain `#[repr(C)]` data of `u64`s, as the kernel reads them.
unsafe impl DeviceRepr for CachedLogsC {}
unsafe impl DeviceRepr for RowsTableC {}

struct Module {
    _module: Arc<CudaModule>,
    rows: CudaFunction,
}

fn module() -> crate::Result<&'static Module> {
    static MODULE: OnceLock<Module> = OnceLock::new();
    if let Some(m) = MODULE.get() {
        return Ok(m);
    }
    let be = backend()?;
    let module = be
        .ctx
        .load_module(Ptx::from_binary(CARD_ROWS_CUBIN.to_vec()))?;
    let m = Module {
        rows: module.load_function("derive_rows_from_cache")?,
        _module: module,
    };
    Ok(MODULE.get_or_init(|| m))
}

// ---------------------------------------------------------------------------
// The upload.
// ---------------------------------------------------------------------------

/// What a table of the group is, once planned.
enum Plan {
    Wide,
    Narrow,
    /// Records copied and filled (CPU's too, its logs then perhaps kept).
    Records,
    /// A CPU table the cache holds: filled from its logs, nothing copied.
    CpuHit(u64),
    /// A LOAD (`false`) or STORE table derived from the cached logs.
    Derived {
        store: bool,
        cycles: Vec<u32>,
    },
}

/// A group's upload under [`Mode::Order`] or [`Mode::Derive`]
/// ([`DeviceColumns::upload_tables`] calls it). `None` as today's: the card
/// declines and every reader takes the host's columns.
pub(crate) fn upload_tables(tables: &[TableUpload<'_>]) -> Option<DeviceColumns> {
    let mode = mode();
    let total: usize = tables.iter().map(TableUpload::cells).sum();
    if total == 0
        || tables.iter().any(|table| {
            matches!(table, TableUpload::Narrow(_)) && u32::try_from(table.cells()).is_err()
        })
    {
        return None;
    }
    let mut guard = lock();
    let cache = guard.get_or_insert_with(Cache::default);
    cache.tick += 1;
    let tick = cache.tick;
    let derive = mode == Mode::Derive;

    // ---- the plan: what each table sends ----
    let mut plans: Vec<Plan> = Vec::with_capacity(tables.len());
    // This group's CPU tables' cycles and decode, for coverage below.
    let mut group_cpu: BTreeMap<u64, u64> = BTreeMap::new();
    let mut group_decode: Option<Arc<CardDecode>> = None;
    let mut decode_mixed = false;
    for table in tables {
        let plan = match table {
            TableUpload::Wide(_) => Plan::Wide,
            TableUpload::Narrow(_) => Plan::Narrow,
            TableUpload::Records(RecordsInput::Cpu {
                logs,
                decode,
                first_cycle,
                rows,
            }) if derive => {
                let n: usize = logs.iter().map(|p| p.len()).sum();
                let host = host_of(logs);
                group_cpu.insert(*first_cycle, n as u64);
                match &group_decode {
                    None => group_decode = Some(Arc::clone(decode)),
                    Some(d) => decode_mixed |= !Arc::ptr_eq(d, decode),
                }
                match cache.entries.get_mut(first_cycle) {
                    Some(e)
                        if e.n == n
                            && e.host == host
                            && Arc::ptr_eq(&e.decode, decode)
                            && n <= *rows =>
                    {
                        e.tick = tick;
                        Plan::CpuHit(*first_cycle)
                    }
                    _ => Plan::Records,
                }
            }
            _ => Plan::Records,
        };
        plans.push(plan);
    }
    if derive && !decode_mixed {
        // The cache's ranges as they will be once this group's CPU tables are
        // in it, all read with one program's DECODE rows.
        let mut held: BTreeMap<u64, u64> = group_cpu.clone();
        for (&first, e) in &cache.entries {
            let same = group_decode
                .as_ref()
                .is_none_or(|d| Arc::ptr_eq(d, &e.decode));
            if same {
                held.entry(first).or_insert(e.n as u64);
            }
        }
        let one_decode = group_decode
            .clone()
            .or_else(|| cache.entries.values().next().map(|e| Arc::clone(&e.decode)));
        let all_one = cache.entries.values().all(|e| {
            one_decode
                .as_ref()
                .is_some_and(|d| Arc::ptr_eq(d, &e.decode))
        });
        for (plan, table) in plans.iter_mut().zip(tables) {
            let TableUpload::Records(records) = table else {
                continue;
            };
            let (store, cycles) = match records {
                RecordsInput::Load { records, rows } if records_len(records) == *rows => {
                    (false, cycles_of(records))
                }
                RecordsInput::Store { records, rows } if records_len(records) == *rows => {
                    (true, cycles_of(records))
                }
                _ => continue,
            };
            let counter = if store {
                &counters().records_store
            } else {
                &counters().records_load
            };
            let Some((mut cycles, lo, hi)) = cycles else {
                add(counter, 1);
                continue;
            };
            let shift = FAULT_CYCLE_SHIFT.load(Ordering::Relaxed);
            if shift != 0 {
                cycles.iter_mut().for_each(|c| *c = c.wrapping_add(shift));
            }
            if !all_one || !covers(&held, lo, hi) {
                add(counter, 1);
                continue;
            }
            // Keep the entries it reads: nothing evicts them in this upload.
            for (&first, e) in cache.entries.range_mut(..=hi) {
                if first + e.n as u64 > lo {
                    e.tick = tick;
                }
            }
            *plan = Plan::Derived { store, cycles };
        }
    }

    // ---- room: the table store, every copy held at once, kept logs ----
    let be = backend().ok()?;
    let Some(room) = be.reserve(total as u64 * 8) else {
        crate::device::note_device_fallback();
        return None;
    };
    let staging: u64 = tables
        .iter()
        .zip(&plans)
        .map(|(table, plan)| match (table, plan) {
            (TableUpload::Records(records), Plan::Records) => records.bytes() as u64,
            (_, Plan::Derived { cycles, .. }) => (cycles.len() * 4) as u64,
            (TableUpload::Narrow(packed), Plan::Narrow) => packed.bytes() as u64,
            _ => 0,
        })
        .sum();
    let Some(_staging) = (staging == 0)
        .then_some(None)
        .or_else(|| be.reserve(staging).map(Some))
    else {
        // Not room for every copy at once: today's upload, one table at a
        // time. The cache is left as it is.
        add(&counters().order_refused, 1);
        drop(guard);
        return DeviceColumns::upload_tables_today(tables);
    };
    crate::argue_probe::note_device(crate::argue_probe::Surface::Columns, total as u64 * 8);
    add(&counters().groups, 1);
    let stream = be.next_stream();
    // SAFETY: every element is written by the copies and kernels below.
    let mut buffer = unsafe { alloc_or_trim::<u64>(&stream, total) }.ok()?;
    let mut spans = Vec::new();
    let mut checks = FillChecks::default();

    // ---- pass 1: every copy ----
    let mut staged: Vec<Option<Staged<'_>>> = Vec::with_capacity(tables.len());
    let mut lists: Vec<Option<CudaSlice<u32>>> = Vec::with_capacity(tables.len());
    let mut at = 0usize;
    let mut offsets = Vec::with_capacity(tables.len());
    for (table, plan) in tables.iter().zip(&plans) {
        offsets.push(at);
        let (mut s, mut l) = (None, None);
        match (table, plan) {
            (TableUpload::Wide(columns), _) => {
                for column in columns.iter() {
                    let mut slab = buffer.slice_mut(at..at + column.len());
                    crate::pinned::note_upload(column.as_ptr().cast(), column.len() * 8);
                    stream.memcpy_htod(*column, &mut slab).ok()?;
                    spans.push((at, column.len()));
                    at += column.len();
                }
                staged.push(None);
                lists.push(None);
                continue;
            }
            (TableUpload::Narrow(_), _) => {}
            (TableUpload::Records(RecordsInput::Cpu { decode, logs, .. }), Plan::CpuHit(key)) => {
                let e = cache.entries.get(key)?;
                let (ptr, _record) = e.logs.device_ptr(&stream);
                add(&counters().cpu_hits, 1);
                add(
                    &counters().log_bytes_spared,
                    logs.iter()
                        .map(|p| std::mem::size_of_val(*p))
                        .sum::<usize>() as u64,
                );
                s = Some(Staged::Cpu {
                    logs: CpuLogsOnCard::Cached(ptr),
                    n: e.n,
                    decode: decode.as_ref(),
                });
            }
            (TableUpload::Records(records), Plan::Derived { store, cycles }) => {
                crate::pinned::note_upload(cycles.as_ptr().cast(), cycles.len() * 4);
                l = Some(stream.clone_htod(cycles).ok()?);
                add(&counters().list_bytes, (cycles.len() * 4) as u64);
                add(&counters().records_bytes_spared, records.bytes() as u64);
                add(
                    if *store {
                        &counters().derived_store
                    } else {
                        &counters().derived_load
                    },
                    1,
                );
                s = Some(Staged::Elsewhere);
            }
            (TableUpload::Records(records), _) => {
                if matches!(records, RecordsInput::Cpu { .. }) && derive {
                    add(&counters().cpu_copies, 1);
                }
                s = Some(crate::fill::stage(&stream, *records).ok()?);
            }
        }
        let (rows, cols) = match table {
            TableUpload::Narrow(packed) => (packed.rows(), packed.cols()),
            TableUpload::Records(records) => (records.rows(), records.cols()),
            TableUpload::Wide(_) => (0, 0),
        };
        spans.extend((0..cols).map(|k| (at + k * rows, rows)));
        at += rows * cols;
        staged.push(s);
        lists.push(l);
    }

    // The derived launches' descriptors: the cache's entries and this group's
    // CPU tables' logs, ascending; each family's tables.
    let mut kernels_ok = true;
    let derived_any = plans.iter().any(|p| matches!(p, Plan::Derived { .. }));
    let mut entry_descs: Vec<CachedLogsC> = Vec::new();
    let mut table_descs: [Vec<RowsTableC>; 2] = [Vec::new(), Vec::new()];
    let mut decode_for: Option<Arc<CardDecode>> = None;
    if derived_any {
        let mut by_first: BTreeMap<u64, CachedLogsC> = BTreeMap::new();
        for (first, e) in &cache.entries {
            let (ptr, _record) = e.logs.device_ptr(&stream);
            by_first.insert(
                *first,
                CachedLogsC {
                    first: *first,
                    n: e.n as u64,
                    logs: ptr,
                },
            );
            decode_for.get_or_insert_with(|| Arc::clone(&e.decode));
        }
        for (table, s) in tables.iter().zip(&staged) {
            if let (
                TableUpload::Records(RecordsInput::Cpu {
                    first_cycle,
                    decode,
                    ..
                }),
                Some(Staged::Cpu {
                    logs: CpuLogsOnCard::Owned(slice),
                    n,
                    ..
                }),
            ) = (table, s)
            {
                let (ptr, _record) = slice.device_ptr(&stream);
                by_first.insert(
                    *first_cycle,
                    CachedLogsC {
                        first: *first_cycle,
                        n: *n as u64,
                        logs: ptr,
                    },
                );
                decode_for.get_or_insert_with(|| Arc::clone(decode));
            }
        }
        entry_descs = by_first.into_values().collect();
        let (base, _record) = buffer.device_ptr(&stream);
        for ((table, plan), (list, &offset)) in
            tables.iter().zip(&plans).zip(lists.iter().zip(&offsets))
        {
            if let (Plan::Derived { store, cycles }, Some(list), TableUpload::Records(records)) =
                (plan, list, table)
            {
                let (cycles_at, _record) = list.device_ptr(&stream);
                let family = &mut table_descs[usize::from(*store)];
                let before = family.last().map_or(0, |t| t.before + t.rows);
                family.push(RowsTableC {
                    cycles: cycles_at,
                    rows: cycles.len() as u64,
                    height: records.rows() as u64,
                    out: base + (offset * 8) as u64,
                    before,
                });
            }
        }
    }
    let entries_dev = (!entry_descs.is_empty())
        .then(|| stream.clone_htod(&entry_descs).ok())
        .flatten();
    let tables_dev: Vec<Option<CudaSlice<RowsTableC>>> = table_descs
        .iter()
        .map(|t| (!t.is_empty()).then(|| stream.clone_htod(t).ok()).flatten())
        .collect();

    // ---- pass 2: every kernel ----
    for ((table, s), &offset) in tables.iter().zip(&staged).zip(&offsets) {
        match table {
            TableUpload::Wide(_) => {}
            TableUpload::Narrow(packed) => {
                let (rows, cols) = (packed.rows(), packed.cols());
                let mut slab = buffer.slice_mut(offset..offset + rows * cols);
                crate::narrow::widen_col_major_into(&stream, be, *packed, rows, cols, &mut slab)
                    .ok()?;
            }
            TableUpload::Records(records) => {
                let Some(s) = s else { continue };
                let (rows, cols) = (records.rows(), records.cols());
                let mut slab = buffer.slice_mut(offset..offset + rows * cols);
                crate::fill::launch_staged(&stream, s, rows, &mut slab, &mut checks).ok()?;
            }
        }
    }
    if let (Some(entries), Some(decode)) = (&entries_dev, &decode_for) {
        let m = module().ok()?;
        let (dec, runs, nruns) = decode.parts();
        let nentries = u32::try_from(entry_descs.len()).ok()?;
        for (k, (descs, dev)) in table_descs.iter().zip(&tables_dev).enumerate() {
            let Some(dev) = dev else { continue };
            let total_rows = descs.last().map_or(0, |t| t.before + t.rows);
            let ntables = u32::try_from(descs.len()).ok()?;
            let store = k as u32;
            let missing = checks.missing(&stream).ok()?;
            let config = LaunchConfig {
                grid_dim: (
                    u32::try_from(total_rows.div_ceil(u64::from(ROW_BLOCK))).ok()?,
                    1,
                    1,
                ),
                block_dim: (ROW_BLOCK, 1, 1),
                shared_mem_bytes: 0,
            };
            // SAFETY: the descriptors name buffers alive until the stream
            // is synchronized below: the cache's entries (kept by the lock
            // and this upload's tick), this group's copies, the store.
            let launched = unsafe {
                stream
                    .launch_builder(&m.rows)
                    .arg(entries)
                    .arg(&nentries)
                    .arg(dev)
                    .arg(&ntables)
                    .arg(&total_rows)
                    .arg(&store)
                    .arg(dec)
                    .arg(runs)
                    .arg(&nruns)
                    .arg(missing)
                    .launch(config)
            };
            kernels_ok &= launched.is_ok();
        }
    } else if derived_any {
        kernels_ok = false;
    }
    stream.synchronize().ok()?;
    if !kernels_ok || !checks.passed(&stream).ok()? {
        add(&counters().declined, 1);
        crate::device::note_device_fallback();
        return None;
    }

    // ---- keep this group's CPU logs ----
    if derive {
        for (table, s) in tables.iter().zip(staged.iter_mut()) {
            let TableUpload::Records(RecordsInput::Cpu {
                first_cycle,
                decode,
                logs,
                ..
            }) = table
            else {
                continue;
            };
            let Some(Staged::Cpu {
                logs: CpuLogsOnCard::Owned(_),
                ..
            }) = s
            else {
                continue;
            };
            let Some(Staged::Cpu {
                logs: CpuLogsOnCard::Owned(slice),
                n,
                ..
            }) = s.take()
            else {
                continue;
            };
            let host = host_of(logs);
            keep(
                cache,
                be,
                *first_cycle,
                n,
                host,
                slice,
                Arc::clone(decode),
                tick,
            );
        }
    }
    drop(guard);
    Some(DeviceColumns::from_parts(stream, buffer, spans, room))
}

fn records_len<T>(pieces: &[&[T]]) -> usize {
    pieces.iter().map(|p| p.len()).sum()
}

/// The address of a table's first log on the host (0 for none).
fn host_of(logs: &[&[LogC]]) -> usize {
    logs.iter()
        .find(|p| !p.is_empty())
        .map_or(0, |p| p.as_ptr() as usize)
}

/// `logs` (cycles `[first, first + n)`, from the host buffer at `host`)
/// kept in the cache, entries this upload did not read let go, least recently
/// read first, to make room within the budget and the ledger. Not kept
/// outside a block ([`begin`]), nor when no room is made.
#[allow(clippy::too_many_arguments)]
fn keep(
    cache: &mut Cache,
    be: &crate::device::Backend,
    first: u64,
    n: usize,
    host: usize,
    logs: CudaSlice<LogC>,
    decode: Arc<CardDecode>,
    tick: u64,
) {
    let bytes = (n * std::mem::size_of::<LogC>()) as u64;
    if !KEEPING.load(Ordering::Relaxed)
        || bytes > budget_bytes()
        || cache.entries.contains_key(&first)
    {
        return;
    }
    while cache.bytes + bytes > budget_bytes() {
        let Some((&victim, _)) = cache
            .entries
            .iter()
            .filter(|(_, e)| e.tick < tick)
            .min_by_key(|(_, e)| e.tick)
        else {
            return;
        };
        if let Some(e) = cache.entries.remove(&victim) {
            cache.bytes -= e.bytes();
            add(&counters().evicted, 1);
        }
    }
    let Some(room) = be.reserve(bytes) else {
        return;
    };
    cache.bytes += bytes;
    add(&counters().cpu_kept, 1);
    cache.entries.insert(
        first,
        Entry {
            n,
            host,
            logs,
            decode,
            tick,
            _room: room,
        },
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn op(ts: u64) -> LoadOpC {
        LoadOpC {
            timestamp: ts,
            ..LoadOpC::default()
        }
    }

    #[test]
    fn the_knob_reads_off_order_and_derive_and_refuses_the_rest() {
        assert_eq!(Mode::parse(None), Ok(Mode::Off));
        assert_eq!(Mode::parse(Some("")), Ok(Mode::Off));
        assert_eq!(Mode::parse(Some("0")), Ok(Mode::Off));
        assert_eq!(Mode::parse(Some("off")), Ok(Mode::Off));
        assert_eq!(Mode::parse(Some("order")), Ok(Mode::Order));
        assert_eq!(Mode::parse(Some(" derive ")), Ok(Mode::Derive));
        assert_eq!(Mode::parse(Some("1")), Ok(Mode::Derive));
        assert!(Mode::parse(Some("yes")).is_err());
        assert!(Mode::parse(Some("Derive")).is_err());
    }

    #[test]
    fn a_records_cycle_is_its_timestamp_less_four_over_four() {
        let a = [op(4), op(4 * 9 + 4)];
        let b = [op(4 * 7 + 4)];
        let (cycles, lo, hi) = cycles_of(&[&a[..], &b[..]]).expect("cycles");
        assert_eq!(cycles, vec![0, 9, 7]);
        assert_eq!((lo, hi), (0, 9));
        // No cycle has these timestamps.
        assert!(cycles_of(&[&[op(0)][..]]).is_none());
        assert!(cycles_of(&[&[op(6)][..]]).is_none());
        assert!(cycles_of(&[&[op(4 * (1u64 << 32) + 4)][..]]).is_none());
        assert!(cycles_of::<LoadOpC>(&[]).is_none());
    }

    #[test]
    fn coverage_needs_every_cycle_between_the_ends() {
        let held: BTreeMap<u64, u64> = [(0, 10), (10, 10), (30, 10)].into_iter().collect();
        assert!(covers(&held, 0, 19));
        assert!(covers(&held, 3, 12));
        assert!(covers(&held, 30, 39));
        assert!(!covers(&held, 15, 25), "the gap 20..30");
        assert!(!covers(&held, 35, 40), "past the last");
        assert!(!covers(&BTreeMap::new(), 0, 0));
        let one: BTreeMap<u64, u64> = [(5, 1)].into_iter().collect();
        assert!(covers(&one, 5, 5));
        assert!(!covers(&one, 4, 5));
    }
}
