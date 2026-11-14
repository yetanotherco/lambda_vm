//! A group's upload under
//! `LAMBDA_VM_CARD_ROWS` (read once). Unset is `words`, the default (below);
//! `order` is the next item; `0` or `off` is the upload as it was before
//! (`DeviceColumns::upload_tables_today`), byte for byte and call for call.
//! The knob takes no other word. Every mode writes the same words.
//!
//! - `order`: every copy of the group first, then every kernel, then the one
//!   sync and the one check readback the upload already makes. Today each
//!   records table's copy waits, in stream order, for the table before it to
//!   be filled; beside a running commit each such wait costs up to what the
//!   commit has queued ahead (a probe beside a looping commit measured the
//!   table-by-table upload at about two commit periods, this order at 1.3).
//! - `words` (the default): `order`, and the card keeps the CPU chunks' logs
//!   it receives (an LRU cache of whole CPU tables,
//!   `LAMBDA_VM_CARD_ROWS_CACHE_MIB`, default 768, each entry promised to the
//!   backend's ledger), fills a CPU table it already holds without copying
//!   its logs again, and derives a LOAD or STORE table from the cached logs
//!   (`derive_rows_from_cache`, `kernels/card_rows.cu`) instead of copying
//!   its records: the table's rows' cycles go up instead (4 bytes a row
//!   against 32), read from the records' timestamps. One launch for a
//!   group's LOAD tables and one for its STORE tables. A table whose cycles
//!   the cache does not hold, after this group's CPU tables are in it, goes
//!   up as records, as today. And MEMW_R is built on the card from the same
//!   cached logs (`derive_registers_from_cache`) and the 8-byte words the
//!   walk writes in place of the 32-byte rows (`prover`'s `register_words`),
//!   with no host pass at the upload; the rows of cycles the card will not
//!   hold are decoded on the host and sent whole. A build that cannot write
//!   words (no record mode, no CPU logs kept) sends MEMW_R as rows, and LOAD
//!   and STORE are still derived. Measured on whole blocks against `order`:
//!   −0.42 s on a median one, −1.66 s on a large one (the trace built −0.28 /
//!   −0.62 s).
//!
//! Nothing the verifier reads moves: the same tables, the same words. A row
//! whose cycle is not held, or whose log is not that family's op, is counted
//! into the upload's check, which declines the group (its readers take the
//! host's columns, as for a CPU log with no DECODE row). The cache is
//! emptied at a block's start and end ([`begin`], [`end`]) and only ever
//! holds a table's logs while every upload that read them has synchronized:
//! `words`' uploads take the cache's lock for their whole length (`order`'s
//! take none).

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use cudarc::driver::{
    CudaFunction, CudaModule, CudaSlice, DevicePtr, DeviceRepr, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::Ptx;

use crate::columns::{DeviceColumns, TableClock, TableUpload};
use crate::device::{DeviceReservation, alloc_or_trim, backend};
use crate::fill::{
    CardDecode, CpuLogsOnCard, FillChecks, LoadOpC, LogC, RecordsInput, RegRowC, Staged, StoreOpC,
};

const CARD_ROWS_CUBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/card_rows.cubin"));
const ROW_BLOCK: u32 = 256;

/// The knob's word.
pub const CARD_ROWS_ENV: &str = "LAMBDA_VM_CARD_ROWS";
/// The cache's budget in MiB.
pub const CARD_ROWS_CACHE_ENV: &str = "LAMBDA_VM_CARD_ROWS_CACHE_MIB";
const DEFAULT_CACHE_MIB: u64 = 768;

/// What a group's upload does ([`CARD_ROWS_ENV`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// The upload as it was before this knob: each table's copy, then its
    /// kernel, table by table.
    Off,
    /// Every copy, then every kernel.
    Order,
    /// [`Mode::Order`], CPU logs kept on the card, LOAD and STORE derived,
    /// and MEMW_R's register rows built on the card from the words the walk
    /// writes in their place: the default.
    Words,
}

impl Mode {
    /// The knob's reading: unset (or empty) or `words` is [`Mode::Words`],
    /// the default (measured on whole blocks against `order`: −0.42 s on a
    /// median one, −1.66 s on a large one); `order` is [`Mode::Order`];
    /// `0` or `off` is [`Mode::Off`]. Anything else (`derive`, `1` and
    /// `registers` among them) is refused.
    pub fn parse(word: Option<&str>) -> Result<Self, String> {
        match word.map(str::trim) {
            None | Some("") | Some("words") => Ok(Self::Words),
            Some("order") => Ok(Self::Order),
            Some("0") | Some("off") => Ok(Self::Off),
            Some(other) => Err(format!(
                "{CARD_ROWS_ENV}={other}: expected off, order or words"
            )),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Order => "order",
            Self::Words => "words",
        }
    }

    /// Whether the card keeps CPU logs and derives tables from them.
    pub fn derives(self) -> bool {
        matches!(self, Self::Words)
    }
}

/// Whether the walk writes MEMW_R's register rows as words for the card to
/// build them from ([`Mode::Words`]).
pub fn builds_words() -> bool {
    mode() == Mode::Words
}

/// A test's override of the knob (0: none; 1 + the mode's index).
static OVERRIDE: AtomicU8 = AtomicU8::new(0);

/// The knob's reading, once a process ([`Mode::parse`]): its mode, or the
/// error naming the values it takes. A block prove refuses on the error
/// ([`setting`]), before any upload reads the mode.
fn from_env() -> &'static Result<Mode, String> {
    static FROM_ENV: OnceLock<Result<Mode, String>> = OnceLock::new();
    FROM_ENV.get_or_init(|| Mode::parse(std::env::var(CARD_ROWS_ENV).ok().as_deref()))
}

/// [`CARD_ROWS_ENV`]'s setting: the mode it names, or an error naming the
/// values it takes (`off`, `order`, `words`) for any other word, a stale
/// `derive`, `1` or `registers` among them. A block prove calls this at its
/// start and refuses on the error, so a stale word never runs as another mode.
pub fn setting() -> Result<Mode, String> {
    from_env().clone()
}

/// The mode this process uploads in. A word the knob does not take reads as
/// [`Mode::Off`] here, but a block prove has refused it first ([`setting`]);
/// the block's report line names the mode in force.
pub fn mode() -> Mode {
    match OVERRIDE.load(Ordering::Relaxed) {
        1 => return Mode::Off,
        2 => return Mode::Order,
        3 => return Mode::Words,
        _ => {}
    }
    from_env().clone().unwrap_or(Mode::Off)
}

/// A test's mode, over the knob's (`None` gives the knob back).
#[doc(hidden)]
pub fn set_mode_override(mode: Option<Mode>) {
    let v = match mode {
        None => 0,
        Some(Mode::Off) => 1,
        Some(Mode::Order) => 2,
        Some(Mode::Words) => 3,
    };
    OVERRIDE.store(v, Ordering::Relaxed);
}

/// A card test's mutant: each derived row's cycle read this many cycles late
/// (0: none), in the families [`FAULT_FAMILIES`] names. The test must see
/// other words, or the group declined.
#[doc(hidden)]
pub static FAULT_CYCLE_SHIFT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// The families [`FAULT_CYCLE_SHIFT`] moves: bit 0 LOAD, bit 1 STORE (0:
/// both).
#[doc(hidden)]
pub static FAULT_FAMILIES: AtomicU8 = AtomicU8::new(0);

/// A card test's mutants of the card-built register rows' kernel, one bit
/// each (0: none). The test must see other words, or the group declined.
#[doc(hidden)]
pub static FAULT_REGISTERS: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
/// The kernel reads `rs2`'s row as `rs1`'s: its register and its value.
#[doc(hidden)]
pub const REGISTER_FAULT_SLOT: u32 = 1;
/// The kernel reads a write's old value as the written value of the cycle
/// its old timestamp names, whatever the slot.
#[doc(hidden)]
pub const REGISTER_FAULT_VALUE_AT: u32 = 2;
/// The kernel's faults (no other bit is read).
const KERNEL_FAULTS: u32 = REGISTER_FAULT_SLOT | REGISTER_FAULT_VALUE_AT;

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
    derived_memw_r: AtomicU64,
    register_words: AtomicU64,
    whole_not_held: AtomicU64,
    whole_other: AtomicU64,
    register_word_bytes: AtomicU64,
    register_whole_bytes: AtomicU64,
    register_records_spared: AtomicU64,
    plan_ns: AtomicU64,
    plan_wall_ns: AtomicU64,
}

static COUNTERS: OnceLock<Counters> = OnceLock::new();

fn counters() -> &'static Counters {
    COUNTERS.get_or_init(Counters::default)
}

/// What the uploads under the knob did since the process started.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
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
    /// Bytes of cycle lists sent; of records not sent (MEMW_R's included);
    /// of CPU logs not sent.
    pub list_bytes: u64,
    pub records_bytes_spared: u64,
    pub log_bytes_spared: u64,
    /// MEMW_R tables built on the card.
    pub derived_memw_r: u64,
    /// Register rows sent as their words; sent whole, by why
    /// ([`WholeRows`]).
    pub register_words: u64,
    pub whole: WholeRows,
    /// Bytes of the register rows' words sent; of the rows sent whole; of
    /// MEMW_R's records not sent (in `records_bytes_spared` too).
    pub register_word_bytes: u64,
    pub register_whole_bytes: u64,
    pub register_records_spared: u64,
    /// Seconds the host spent planning MEMW_R's tables ([`plan_words`]),
    /// summed over the tables; and the uploads' own wait for those plans,
    /// summed.
    pub plan_secs: f64,
    pub plan_wall_secs: f64,
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
        derived_memw_r: l(&c.derived_memw_r),
        register_words: l(&c.register_words),
        whole: WholeRows {
            not_held: l(&c.whole_not_held),
            other: l(&c.whole_other),
        },
        register_word_bytes: l(&c.register_word_bytes),
        register_whole_bytes: l(&c.register_whole_bytes),
        register_records_spared: l(&c.register_records_spared),
        plan_secs: l(&c.plan_ns) as f64 * 1e-9,
        plan_wall_secs: l(&c.plan_wall_ns) as f64 * 1e-9,
    }
}

fn add(counter: &AtomicU64, n: u64) {
    counter.fetch_add(n, Ordering::Relaxed);
}

// ---------------------------------------------------------------------------
// The cache of CPU tables' logs.
// ---------------------------------------------------------------------------

/// A CPU table's logs on the card: cycles `[first, first + n)`, copied from
/// the logs whose [`Identity`] it records, with those logs' records held
/// alive (`hold`) for as long as the entry lives.
struct Entry {
    n: usize,
    identity: Identity,
    /// The records the logs were copied from: while the entry lives, their
    /// buffers can be neither freed nor handed back to the executor
    /// (`Arc::try_unwrap` fails on a shared buffer), so no other logs can
    /// come to lie at the address the identity names.
    _hold: Arc<dyn std::any::Any + Send + Sync>,
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
        &c.derived_memw_r,
        &c.register_words,
        &c.whole_not_held,
        &c.whole_other,
        &c.register_word_bytes,
        &c.register_whole_bytes,
        &c.register_records_spared,
        &c.plan_ns,
        &c.plan_wall_ns,
    ] {
        counter.store(0, Ordering::Relaxed);
    }
}

/// The block's report line (`BLOCK CARD ROWS: …`), from the counters and the
/// cache as they stand. Under [`Mode::Words`] it ends with MEMW_R's part
/// (` · MEMW_R derived …`).
pub fn report_line() -> String {
    let s = stats();
    let (entries, bytes) = held();
    let registers = if mode() == Mode::Words {
        let w = s.whole;
        format!(
            " · MEMW_R derived {} · register rows as words {} · whole {} (not held {}, other {}) · MEMW_R records not sent {} B · words {} B · whole rows {} B · planned in {:.3} s summed, {:.3} s waited",
            s.derived_memw_r,
            s.register_words,
            w.total(),
            w.not_held,
            w.other,
            s.register_records_spared,
            s.register_word_bytes,
            s.register_whole_bytes,
            s.plan_secs,
            s.plan_wall_secs,
        )
    } else {
        String::new()
    };
    format!(
        "BLOCK CARD ROWS: mode {} · groups {} (order refused {}, declined {}) · CPU tables from the cache {}, copied {}, kept {}, let go {} · LOAD derived {} of {} · STORE derived {} of {} · cycle lists {} B · records not sent {} B · CPU logs not sent {} B · cache {} entries, {} B, budget {} B{registers}",
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
// The register rows: which cross as their words, which whole.
// ---------------------------------------------------------------------------

/// Why register rows cross whole instead of as their words.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WholeRows {
    /// A row whose cycle's logs the card does not hold (decoded on the host).
    pub not_held: u64,
    /// A row the walk wrote whole (its marker's).
    pub other: u64,
}

impl WholeRows {
    pub fn total(&self) -> u64 {
        self.not_held + self.other
    }
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

/// `card_rows.cu`'s `RegistersTableC`: a MEMW_R table built on the card,
/// its words and whole rows ([`WordsPlan`]).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct RegistersTableC {
    words: u64,
    whole: u64,
    rows: u64,
    height: u64,
    out: u64,
    before: u64,
    /// The run index of `whole`'s first row (a marker's high half less this
    /// indexes `whole`).
    whole_base: u64,
}

const _: () = {
    assert!(std::mem::size_of::<CachedLogsC>() == 24);
    assert!(std::mem::size_of::<RowsTableC>() == 40);
    assert!(std::mem::size_of::<RegistersTableC>() == 56);
};

// SAFETY: plain `#[repr(C)]` data of `u64`s, as the kernel reads them.
unsafe impl DeviceRepr for CachedLogsC {}
unsafe impl DeviceRepr for RowsTableC {}
unsafe impl DeviceRepr for RegistersTableC {}

struct Module {
    _module: Arc<CudaModule>,
    rows: CudaFunction,
    registers: CudaFunction,
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
        registers: module.load_function("derive_registers_from_cache")?,
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
    /// A MEMW_R table built from the walk's words and the cached logs.
    Words(WordsPlan),
}

/// A MEMW_R table in words mode as it goes up: the word ranges whose cycles'
/// logs the card does not hold, decoded on the host into `extra` (sent whole,
/// after the table's own whole rows, their words patched to markers).
struct WordsPlan {
    ranges: Vec<(usize, usize)>,
    extra: Vec<RegRowC>,
    n_whole: usize,
    whole_base: u64,
    n: usize,
    span: (u64, u64),
    /// Markers among the decoded ranges (their rows were whole already).
    markers_decoded: usize,
}

/// [`WordsPlan`] of a table's `words` (pieces in order) against the CPU
/// tables `held` (first cycle, count; ascending): the cycles not held form
/// index ranges, found by binary search (a table's words are in cycle order),
/// and decoded with `decode`. `None` (the table goes up as its decoded rows,
/// today's fill) when no row's cycle is held: the card then holds no log the
/// kernel could read, and a launch with no logs would decline the group.
fn plan_words(
    words: &[&[u64]],
    whole: &[&[RegRowC]],
    whole_base: u64,
    held: &[(u64, u64)],
    decode: &(dyn Fn(usize, usize) -> Vec<RegRowC> + Sync),
) -> Option<WordsPlan> {
    let starts: Vec<usize> = words
        .iter()
        .scan(0usize, |at, p| {
            let s = *at;
            *at += p.len();
            Some(s)
        })
        .collect();
    let n: usize = words.iter().map(|p| p.len()).sum();
    if n == 0 {
        return None;
    }
    let whole_flat: Vec<&RegRowC> = whole.iter().flat_map(|p| p.iter()).collect();
    let word_at = |i: usize| -> u64 {
        let k = starts.partition_point(|&s| s <= i) - 1;
        words[k][i - starts[k]]
    };
    let cycle_of = |i: usize| -> Option<u64> {
        let w = word_at(i);
        let ts = if w as u32 == 0 {
            whole_flat
                .get(usize::try_from((w >> 32).checked_sub(whole_base)?).ok()?)?
                .timestamp
        } else {
            w & 0xFFFF_FFFF
        };
        Some(ts.saturating_sub(4) / 4)
    };
    let (lo, hi) = (cycle_of(0)?, cycle_of(n - 1)?);
    // The cycles of [lo, hi] no held CPU table covers.
    let mut gaps = Vec::new();
    let mut c = lo;
    for &(f, len) in held {
        if f + len <= c {
            continue;
        }
        if f > hi {
            break;
        }
        if f > c {
            gaps.push((c, f - 1));
        }
        c = c.max(f + len);
        if c > hi {
            break;
        }
    }
    if c <= hi {
        gaps.push((c, hi));
    }
    let lower = |pred: &dyn Fn(u64) -> bool| -> Option<usize> {
        // The first index whose cycle fails `pred` (cycles ascend).
        let (mut a, mut b) = (0usize, n);
        while a < b {
            let m = (a + b) / 2;
            if pred(cycle_of(m)?) {
                a = m + 1;
            } else {
                b = m;
            }
        }
        Some(a)
    };
    let mut ranges = Vec::new();
    for (a, b) in gaps {
        let s = lower(&|c| c < a)?;
        let e = lower(&|c| c <= b)?;
        if s < e {
            ranges.push((s, e));
        }
    }
    if ranges.iter().map(|&(s, e)| e - s).sum::<usize>() == n {
        return None;
    }
    let mut extra = Vec::new();
    let mut markers_decoded = 0usize;
    for &(s, e) in &ranges {
        markers_decoded += (s..e).filter(|&i| word_at(i) as u32 == 0).count();
        let rows = decode(s, e);
        if rows.len() != e - s {
            return None;
        }
        extra.extend(rows);
    }
    Some(WordsPlan {
        ranges,
        extra,
        n_whole: whole_flat.len(),
        whole_base,
        n,
        span: (lo, hi),
        markers_decoded,
    })
}

/// Whether a group's plans build any table on the card from the logs it
/// holds (LOAD or STORE derived, MEMW_R built from words):
/// then its cached logs go up as descriptors and the derive kernels launch.
/// A table planned so and never launched would keep whatever its store held.
fn launches_from_logs(plans: &[Plan]) -> bool {
    plans.iter().any(|p| match p {
        Plan::Derived { .. } | Plan::Words(_) => true,
        Plan::Wide | Plan::Narrow | Plan::Records | Plan::CpuHit(_) => false,
    })
}

/// A group's upload under [`Mode::Order`] or [`Mode::Words`]
/// ([`DeviceColumns::upload_tables`] calls it). `None` as
/// today's: the card declines and every reader takes the host's columns.
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
    // Only `words` reads and writes the cache: `order` takes
    // no lock, so its uploads overlap as today's do.
    let mut local = Cache::default();
    let mut guard = mode.derives().then(lock);
    let cache = match guard.as_mut() {
        Some(held) => held.get_or_insert_with(Cache::default),
        None => &mut local,
    };
    cache.tick += 1;
    let tick = cache.tick;
    let derive = mode.derives();

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
                ..
            }) if derive => {
                let n: usize = logs.iter().map(|p| p.len()).sum();
                let identity = Identity::of(logs);
                group_cpu.insert(*first_cycle, n as u64);
                match &group_decode {
                    None => group_decode = Some(Arc::clone(decode)),
                    Some(d) => decode_mixed |= !Arc::ptr_eq(d, decode),
                }
                match cache.entries.get_mut(first_cycle) {
                    Some(e)
                        if e.n == n
                            && e.identity == identity
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
            let families = FAULT_FAMILIES.load(Ordering::Relaxed);
            let this = if store { 2 } else { 1 };
            if shift != 0 && (families == 0 || families & this != 0) {
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
        // Words mode: MEMW_R's tables as the walk wrote them; the rows of
        // cycles the card will not hold decoded on the host.
        if mode == Mode::Words && all_one {
            let held: Vec<(u64, u64)> = held.iter().map(|(&first, &n)| (first, n)).collect();
            for (i, table) in tables.iter().enumerate() {
                let TableUpload::Records(RecordsInput::MemwRWords {
                    words,
                    whole,
                    whole_base,
                    rows,
                    decode,
                }) = table
                else {
                    continue;
                };
                if words.iter().map(|p| p.len()).sum::<usize>() != *rows {
                    continue;
                }
                let clock = std::time::Instant::now();
                let plan = plan_words(words, whole, *whole_base, &held, *decode);
                let ns = clock.elapsed().as_nanos() as u64;
                add(&counters().plan_ns, ns);
                add(&counters().plan_wall_ns, ns);
                let Some(plan) = plan else { continue };
                let (lo, hi) = plan.span;
                for (&first, e) in cache.entries.range_mut(..=hi) {
                    if first + e.n as u64 > lo {
                        e.tick = tick;
                    }
                }
                plans[i] = Plan::Words(plan);
            }
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
            (_, Plan::Words(plan)) => {
                (plan.n * 8 + (plan.n_whole + plan.extra.len()) * std::mem::size_of::<RegRowC>())
                    as u64
            }
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
    // A MEMW_R table built on the card: its words and whole rows up.
    // (words, whole rows, the first whole row's run index, rows)
    type RegistersUp = (CudaSlice<u64>, Option<CudaSlice<RegRowC>>, u64, u64);
    let mut registers_up: Vec<Option<RegistersUp>> = Vec::with_capacity(tables.len());
    let mut at = 0usize;
    let mut offsets = Vec::with_capacity(tables.len());
    for (table, plan) in tables.iter().zip(&plans) {
        offsets.push(at);
        let (mut s, mut l, mut r) = (None, None, None);
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
                registers_up.push(None);
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
            (
                TableUpload::Records(
                    records @ RecordsInput::MemwRWords {
                        words: pieces,
                        whole,
                        ..
                    },
                ),
                Plan::Words(plan),
            ) => {
                let mut words = crate::fill::upload_pieces(&stream, pieces).ok()?;
                let mut k = plan.whole_base + plan.n_whole as u64;
                for &(s, e) in &plan.ranges {
                    let patch: Vec<u64> = (0..(e - s) as u64).map(|j| (k + j) << 32).collect();
                    k += (e - s) as u64;
                    stream
                        .memcpy_htod(&patch, &mut words.slice_mut(s..e))
                        .ok()?;
                }
                let mut wp: Vec<&[RegRowC]> = whole.to_vec();
                if !plan.extra.is_empty() {
                    wp.push(&plan.extra);
                }
                let total_whole = plan.n_whole + plan.extra.len();
                let whole_dev = if total_whole == 0 {
                    None
                } else {
                    Some(crate::fill::upload_pieces(&stream, &wp).ok()?)
                };
                r = Some((words, whole_dev, plan.whole_base, plan.n as u64));
                let c = counters();
                add(&c.derived_memw_r, 1);
                let decoded: usize = plan.ranges.iter().map(|&(s, e)| e - s).sum();
                add(
                    &c.register_words,
                    (plan.n - plan.n_whole.min(plan.n) - decoded + plan.markers_decoded) as u64,
                );
                add(&c.whole_other, (plan.n_whole - plan.markers_decoded) as u64);
                add(&c.whole_not_held, decoded as u64);
                add(&c.register_word_bytes, (plan.n * 8) as u64);
                add(
                    &c.register_whole_bytes,
                    (total_whole * std::mem::size_of::<RegRowC>()) as u64,
                );
                let spared = (plan.n * std::mem::size_of::<RegRowC>()) as u64;
                add(&c.records_bytes_spared, spared);
                add(&c.register_records_spared, spared);
                let _ = records;
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
        registers_up.push(r);
    }

    // The derived launches' descriptors: the cache's entries and this group's
    // CPU tables' logs, ascending; each family's tables.
    let mut kernels_ok = true;
    let derived_any = launches_from_logs(&plans);
    let mut entry_descs: Vec<CachedLogsC> = Vec::new();
    let mut table_descs: [Vec<RowsTableC>; 2] = [Vec::new(), Vec::new()];
    let mut register_descs: Vec<RegistersTableC> = Vec::new();
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
        for ((table, plan), (up, &offset)) in tables
            .iter()
            .zip(&plans)
            .zip(registers_up.iter().zip(&offsets))
        {
            if let (
                Plan::Words(_),
                Some((words, whole, whole_base, rows)),
                TableUpload::Records(records),
            ) = (plan, up, table)
            {
                let (words_at, _record) = words.device_ptr(&stream);
                let whole_at = whole.as_ref().map_or(0, |w| w.device_ptr(&stream).0);
                let before = register_descs.last().map_or(0, |t| t.before + t.rows);
                register_descs.push(RegistersTableC {
                    words: words_at,
                    whole: whole_at,
                    rows: *rows,
                    height: records.rows() as u64,
                    out: base + (offset * 8) as u64,
                    before,
                    whole_base: *whole_base,
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
    let registers_dev = (!register_descs.is_empty())
        .then(|| stream.clone_htod(&register_descs).ok())
        .flatten();

    // ---- pass 2: every kernel ----
    // `LAMBDA_VM_CARD_UPLOAD_TIMES=1` times each kernel here (its copy was
    // made in pass 1), and a derived family's launch as one table.
    let mut clocks = Vec::new();
    for ((table, s), &offset) in tables.iter().zip(&staged).zip(&offsets) {
        match table {
            TableUpload::Wide(_) => {}
            TableUpload::Narrow(packed) => {
                let (rows, cols) = (packed.rows(), packed.cols());
                let mut slab = buffer.slice_mut(offset..offset + rows * cols);
                let clock = TableClock::start(&stream, false);
                crate::narrow::widen_col_major_into(&stream, be, *packed, rows, cols, &mut slab)
                    .ok()?;
                clocks.extend(clock.filter(|c| c.done.record(&stream).is_ok()));
            }
            TableUpload::Records(records) => {
                let Some(s) = s else { continue };
                if matches!(s, Staged::Elsewhere) {
                    continue;
                }
                let (rows, cols) = (records.rows(), records.cols());
                let mut slab = buffer.slice_mut(offset..offset + rows * cols);
                let clock = TableClock::start(&stream, true);
                crate::fill::launch_staged(&stream, s, rows, &mut slab, &mut checks).ok()?;
                clocks.extend(clock.filter(|c| c.done.record(&stream).is_ok()));
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
            let clock = TableClock::start(&stream, true);
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
            clocks.extend(clock.filter(|c| c.done.record(&stream).is_ok()));
        }
        // The MEMW_R tables built on the card, one launch: a thread a row.
        if let Some(last) = register_descs.last() {
            let total_rows = last.before + last.rows;
            let ntables = u32::try_from(register_descs.len()).ok()?;
            let fault = FAULT_REGISTERS.load(Ordering::Relaxed) & KERNEL_FAULTS;
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
            match &registers_dev {
                None => kernels_ok = false,
                Some(dev) => {
                    let clock = TableClock::start(&stream, true);
                    // SAFETY: as the launches above; the words and whole rows
                    // are this group's copies, alive until the stream is
                    // synchronized below.
                    let launched = unsafe {
                        stream
                            .launch_builder(&m.registers)
                            .arg(entries)
                            .arg(&nentries)
                            .arg(dev)
                            .arg(&ntables)
                            .arg(&total_rows)
                            .arg(dec)
                            .arg(runs)
                            .arg(&nruns)
                            .arg(&fault)
                            .arg(missing)
                            .launch(config)
                    };
                    kernels_ok &= launched.is_ok();
                    clocks.extend(clock.filter(|c| c.done.record(&stream).is_ok()));
                }
            }
        }
    } else if derived_any {
        kernels_ok = false;
    }
    stream.synchronize().ok()?;
    clocks.into_iter().for_each(TableClock::add);
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
                hold: Some(hold),
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
            let kept = Kept {
                identity: Identity::of(logs),
                hold: Arc::clone(hold),
                decode: Arc::clone(decode),
            };
            keep(cache, be, *first_cycle, n, slice, kept, tick);
        }
    }
    drop(guard);
    Some(DeviceColumns::from_parts(stream, buffer, spans, room))
}

fn records_len<T>(pieces: &[&[T]]) -> usize {
    pieces.iter().map(|p| p.len()).sum()
}

/// What a CPU table's logs are taken to be when the cache is asked for them:
/// how many, where on the host each piece starts and ends, and a fingerprint
/// of their contents. A held entry serves a table only when all of it is
/// equal ([`Identity::of`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Identity {
    pub n: usize,
    /// Each piece's host address and length.
    pub pieces: Vec<(usize, usize)>,
    pub print: u64,
}

/// Logs a fingerprint reads: every log's first and last 64, and at most
/// this many more, evenly spaced, five words each.
pub const PRINT_SAMPLES: usize = 4096;

impl Identity {
    pub fn of(logs: &[&[LogC]]) -> Self {
        let n: usize = logs.iter().map(|p| p.len()).sum();
        let pieces = logs
            .iter()
            .map(|p| (p.as_ptr() as usize, p.len()))
            .collect();
        Self {
            n,
            pieces,
            print: fingerprint(logs, n),
        }
    }
}

/// A sampled fingerprint of `n` logs in `logs`' pieces (a mix of the count,
/// the first and last 64 logs, and up to [`PRINT_SAMPLES`] logs evenly
/// spaced between, every word of each). Not a hash of every byte: the
/// identity's addresses and the entry's held records are what make a stale
/// hit impossible; the fingerprint catches logs rewritten in place.
pub fn fingerprint(logs: &[&[LogC]], n: usize) -> u64 {
    let at = |mut i: usize| -> Option<&LogC> {
        for p in logs {
            if i < p.len() {
                return p.get(i);
            }
            i -= p.len();
        }
        None
    };
    let mut h = 0x9e37_79b9_7f4a_7c15u64 ^ n as u64;
    let mut mix = |w: u64| {
        h = (h ^ w).wrapping_mul(0xff51_afd7_ed55_8ccd);
        h ^= h >> 33;
    };
    let stride = (n / PRINT_SAMPLES).max(1);
    let edges = (0..n.min(64)).chain(n.saturating_sub(64)..n);
    for i in edges.chain((64..n.saturating_sub(64)).step_by(stride)) {
        if let Some(l) = at(i) {
            mix(i as u64);
            for w in [l.current_pc, l.next_pc, l.src1_val, l.src2_val, l.dst_val] {
                mix(w);
            }
        }
    }
    h
}

/// What an entry keeps besides the logs on the card.
struct Kept {
    identity: Identity,
    hold: Arc<dyn std::any::Any + Send + Sync>,
    decode: Arc<CardDecode>,
}

/// `logs` (cycles `[first, first + n)`) kept in the cache, entries this
/// upload did not read let go, least recently read first, to make room within
/// the budget and the ledger. Not kept outside a block ([`begin`]), nor when
/// no room is made.
fn keep(
    cache: &mut Cache,
    be: &crate::device::Backend,
    first: u64,
    n: usize,
    logs: CudaSlice<LogC>,
    kept: Kept,
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
            identity: kept.identity,
            _hold: kept.hold,
            logs,
            decode: kept.decode,
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
    fn the_knob_reads_its_words_and_refuses_the_rest() {
        // Unset is the default, words.
        assert_eq!(Mode::parse(None), Ok(Mode::Words));
        assert_eq!(Mode::parse(Some("")), Ok(Mode::Words));
        assert_eq!(Mode::parse(Some(" words ")), Ok(Mode::Words));
        assert_eq!(Mode::parse(Some("0")), Ok(Mode::Off));
        assert_eq!(Mode::parse(Some("off")), Ok(Mode::Off));
        assert_eq!(Mode::parse(Some("order")), Ok(Mode::Order));
        // The removed settings are refused like any other word, by a message
        // that names the knob's words.
        for bad in ["derive", " derive ", "1", "registers", "yes", "Derive", "2"] {
            let err = Mode::parse(Some(bad)).expect_err(bad);
            assert!(err.contains(CARD_ROWS_ENV), "{err}");
            assert!(err.contains("off, order or words"), "{err}");
        }
        assert!(!Mode::Order.derives() && !Mode::Off.derives());
        assert!(Mode::Words.derives());
        assert_eq!(DEFAULT_CACHE_MIB, 768);
    }

    /// A row `(cycle, slot)`'s word, its old timestamp the slot before.
    fn word(cycle: u64, slot: u64) -> u64 {
        let ts = 4 * cycle + 4 + slot;
        ts | (ts - 1) << 32
    }

    /// The words' plan: the rows of cycles no held CPU table covers are
    /// decoded and sent whole, found by binary search over pieces; a table
    /// none of whose cycles is held is not planned (it goes up as its decoded
    /// rows), so no launch is made without logs.
    #[test]
    fn a_words_tables_unheld_rows_are_decoded_and_a_table_with_none_held_is_not_planned() {
        // Cycles 0..20, two rows a cycle, one marker (run index 9) at cycle 5.
        let mut words: Vec<u64> = (0..20).flat_map(|c| [word(c, 0), word(c, 1)]).collect();
        words[10] = 9 << 32;
        let whole = [RegRowC {
            timestamp: 4 * 5 + 4,
            ..RegRowC::default()
        }];
        let decoded = std::sync::Mutex::new(Vec::new());
        let decode = |from: usize, to: usize| {
            decoded.lock().expect("the calls").push((from, to));
            vec![RegRowC::default(); to - from]
        };
        let (a, b) = words.split_at(7);
        let plan = plan_words(&[a, b], &[&whole], 9, &[(8, 4)], &decode).expect("planned");
        // Cycles 0..8 and 12..20 unheld: rows 0..16 and 24..40.
        assert_eq!(plan.ranges, vec![(0, 16), (24, 40)]);
        assert_eq!(plan.extra.len(), 32);
        assert_eq!(plan.markers_decoded, 1);
        assert_eq!((plan.n, plan.n_whole, plan.span), (40, 1, (0, 19)));
        assert_eq!(*decoded.lock().expect("the calls"), vec![(0, 16), (24, 40)]);
        // Every cycle held: nothing decoded.
        let all = plan_words(&[&words], &[&whole], 9, &[(0, 32)], &decode).expect("planned");
        assert!(all.ranges.is_empty() && all.extra.is_empty());
        // No cycle held (a group before any CPU table's logs): not planned.
        assert!(plan_words(&[&words], &[&whole], 9, &[(40, 8)], &decode).is_none());
        assert!(plan_words(&[&words], &[&whole], 9, &[], &decode).is_none());
    }

    /// Every plan that builds a table from the logs launches the kernels (a
    /// MEMW_R table built from words alone in its group included), and no
    /// other plan does.
    #[test]
    fn every_table_built_from_the_logs_launches_the_kernels() {
        let words = || {
            Plan::Words(WordsPlan {
                ranges: Vec::new(),
                extra: Vec::new(),
                n_whole: 0,
                whole_base: 7,
                n: 4,
                span: (0, 0),
                markers_decoded: 0,
            })
        };
        assert!(launches_from_logs(&[Plan::Records, words()]));
        assert!(launches_from_logs(&[words()]));
        assert!(launches_from_logs(&[Plan::Derived {
            store: false,
            cycles: Vec::new()
        }]));
        assert!(!launches_from_logs(&[
            Plan::Wide,
            Plan::Narrow,
            Plan::Records,
            Plan::CpuHit(0)
        ]));
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

    fn logs(n: u64) -> Vec<LogC> {
        (0..n)
            .map(|i| LogC {
                current_pc: 4 * i,
                next_pc: 4 * i + 4,
                src1_val: i,
                src2_val: 2 * i,
                dst_val: 3 * i,
            })
            .collect()
    }

    /// The cache's mutant at the level of its key: the same buffer, at the
    /// same address with the same length, holding other logs, is another
    /// identity, so a held entry cannot serve it. (The cache itself cannot be
    /// handed that case without writing through a shared buffer: an entry
    /// holds its records alive, so nothing else can lie at its address.)
    #[test]
    fn the_same_address_with_other_logs_is_another_identity() {
        let n = 3 * PRINT_SAMPLES as u64 + 777;
        let mut v = logs(n);
        let held = Identity::of(&[&v[..]]);
        let addr = v.as_ptr();
        assert_eq!(Identity::of(&[&v[..]]), held, "the same logs: a hit");
        // The first and the last log are always read; so is every
        // `stride`-th between (the sampled ones).
        let stride = (v.len() / PRINT_SAMPLES).max(1);
        for at in [0, v.len() - 1, 64 + 5 * stride] {
            v[at].dst_val ^= 1;
            let now = Identity::of(&[&v[..]]);
            assert_eq!(v.as_ptr(), addr);
            assert_eq!(now.pieces, held.pieces, "same address, same length");
            assert_ne!(now, held, "log {at} rewritten in place must miss");
            v[at].dst_val ^= 1;
        }
        // Every log rewritten (a test's fault on a CPU chunk's values).
        for l in v.iter_mut() {
            l.src1_val ^= 0xdead;
        }
        assert_ne!(Identity::of(&[&v[..]]), held);
        // The same logs elsewhere (a regenerated window): another address.
        let copy = logs(n);
        assert_ne!(Identity::of(&[&copy[..]]), held);
        // The same logs cut into other pieces: another identity too.
        let (a, b) = copy.split_at(1000);
        assert_ne!(Identity::of(&[a, b]), Identity::of(&[&copy[..]]));
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
