//! A group's upload under
//! `LAMBDA_VM_CARD_ROWS` (read once). Unset is `words`, the default (below);
//! `0` or `off` is the upload as it was before
//! (`DeviceColumns::upload_tables_today`), byte for byte and call for call.
//! Every mode writes the same words.
//!
//! - `order`: every copy of the group first, then every kernel, then the one
//!   sync and the one check readback the upload already makes. Today each
//!   records table's copy waits, in stream order, for the table before it to
//!   be filled; beside a running commit each such wait costs up to what the
//!   commit has queued ahead (a probe beside a looping commit measured the
//!   table-by-table upload at about two commit periods, this order at 1.3).
//! - `derive`: `order`, and the card keeps the CPU chunks' logs it receives
//!   (an LRU cache of whole CPU tables, `LAMBDA_VM_CARD_ROWS_CACHE_MIB`,
//!   default 768, each entry promised to the backend's ledger), fills a CPU
//!   table it already holds without copying its logs again, and derives a
//!   LOAD or STORE table from the cached logs (`derive_rows_from_cache`,
//!   `kernels/card_rows.cu`) instead of copying its records: the table's rows'
//!   cycles go up instead (4 bytes a row against 32), read from the records'
//!   timestamps. One launch for a group's LOAD tables and one for its STORE
//!   tables. A table whose cycles the cache does not hold, after this group's
//!   CPU tables are in it, goes up as records, as today.
//! - `registers` (opt-in): `derive`, and MEMW_R's register rows built on the
//!   card from the same cached logs (`derive_registers_from_cache`). Each row
//!   crosses as one word, its timestamp and its old timestamp (8 bytes against
//!   32), read from the rows the host holds ([`plan_registers`]). The card
//!   reads the register, the value, read or write and, for a write, the value
//!   before it from the logs. The rows it cannot read from them cross whole:
//!   an ECALL cycle's (its handler's rows; the ECALL cycles come from the
//!   run's history, so a run without it sends MEMW_R as records), a write
//!   whose register was last touched by an ECALL or before its CPU table, and
//!   a row whose cycle's logs the card does not hold. A table more than half
//!   of whose rows would cross whole goes up as records.
//! - `words` (the default): `derive`, and MEMW_R built on the card from the
//!   8-byte words the walk writes in place of the 32-byte rows
//!   (`prover`'s `register_words`), with no host pass at the upload; the
//!   rows of cycles the card will not hold are decoded on the host and sent
//!   whole. A build that cannot write words (no record mode, no CPU logs
//!   kept) sends MEMW_R as rows, and LOAD and STORE are still derived.
//!   Measured on whole blocks against `order`: −0.42 s on a median one,
//!   −1.66 s on a large one (the trace built −0.28 / −0.62 s).
//!
//! Nothing the verifier reads moves: the same tables, the same words. A row
//! whose cycle is not held, or whose log is not that family's op, is counted
//! into the upload's check, which declines the group (its readers take the
//! host's columns, as for a CPU log with no DECODE row). The cache is
//! emptied at a block's start and end ([`begin`], [`end`]) and only ever
//! holds a table's logs while every upload that read them has synchronized:
//! `derive`'s uploads take the cache's lock for their whole length (`order`'s
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
    /// [`Mode::Order`], CPU logs kept on the card, LOAD and STORE derived.
    Derive,
    /// [`Mode::Derive`], and MEMW_R's register rows built on the card.
    Registers,
    /// [`Mode::Derive`], and MEMW_R's register rows built on the card from
    /// the words the walk writes in their place (no host plan): the default.
    Words,
}

impl Mode {
    /// The knob's reading: unset (or empty) or `words` is [`Mode::Words`],
    /// the default (measured on whole blocks against `order`: −0.42 s on a
    /// median one, −1.66 s on a large one); `order` is [`Mode::Order`];
    /// `0` or `off` is [`Mode::Off`]; `derive` or `1` is [`Mode::Derive`];
    /// `registers` is [`Mode::Registers`]. Anything else is refused.
    pub fn parse(word: Option<&str>) -> Result<Self, String> {
        match word.map(str::trim) {
            None | Some("") | Some("words") => Ok(Self::Words),
            Some("order") => Ok(Self::Order),
            Some("0") | Some("off") => Ok(Self::Off),
            Some("derive") | Some("1") => Ok(Self::Derive),
            Some("registers") => Ok(Self::Registers),
            Some(other) => Err(format!(
                "{CARD_ROWS_ENV}={other}: expected off, order, derive, registers or words"
            )),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Order => "order",
            Self::Derive => "derive",
            Self::Registers => "registers",
            Self::Words => "words",
        }
    }

    /// Whether the card keeps CPU logs and derives tables from them.
    pub fn derives(self) -> bool {
        matches!(self, Self::Derive | Self::Registers | Self::Words)
    }
}

/// Whether the uploads build MEMW_R's register rows on the card
/// ([`Mode::Registers`]): a build then hands its MEMW_R chunks out with their
/// ECALL cycles, when its run has a history.
pub fn builds_registers() -> bool {
    mode() == Mode::Registers
}

/// Whether the walk writes MEMW_R's register rows as words for the card to
/// build them from ([`Mode::Words`]).
pub fn builds_words() -> bool {
    mode() == Mode::Words
}

/// A test's override of the knob (0: none; 1 + the mode's index).
static OVERRIDE: AtomicU8 = AtomicU8::new(0);

/// The mode this process uploads in. An unknown word is reported once and
/// read as [`Mode::Off`], the upload as it was; the block's report line
/// names the mode in force.
pub fn mode() -> Mode {
    match OVERRIDE.load(Ordering::Relaxed) {
        1 => return Mode::Off,
        2 => return Mode::Order,
        3 => return Mode::Derive,
        4 => return Mode::Registers,
        5 => return Mode::Words,
        _ => {}
    }
    static FROM_ENV: OnceLock<Mode> = OnceLock::new();
    *FROM_ENV.get_or_init(
        || match Mode::parse(std::env::var(CARD_ROWS_ENV).ok().as_deref()) {
            Ok(mode) => mode,
            Err(e) => {
                eprintln!("[gpu] {e}; the upload is as it was (off)");
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
        Some(Mode::Registers) => 4,
        Some(Mode::Words) => 5,
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

/// A card test's mutants of the card-built register rows, one bit each (0:
/// none). The test must see other words, or the group declined.
#[doc(hidden)]
pub static FAULT_REGISTERS: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
/// The kernel reads `rs2`'s row as `rs1`'s: its register and its value.
#[doc(hidden)]
pub const REGISTER_FAULT_SLOT: u32 = 1;
/// The kernel reads a write's old value as the written value of the cycle
/// its old timestamp names, whatever the slot.
#[doc(hidden)]
pub const REGISTER_FAULT_VALUE_AT: u32 = 2;
/// The host sends a write whose register was last touched before its CPU
/// table as its timestamps (the card holds no value for it).
#[doc(hidden)]
pub const REGISTER_FAULT_START: u32 = 4;
/// The host sends an ECALL cycle's rows, and the writes whose register an
/// ECALL touched last, as their timestamps.
#[doc(hidden)]
pub const REGISTER_FAULT_ECALL: u32 = 8;
/// The kernel's faults (the rest are the host's).
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
    records_memw_r: AtomicU64,
    register_words: AtomicU64,
    whole_ecall: AtomicU64,
    whole_after_ecall: AtomicU64,
    whole_before_table: AtomicU64,
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
    /// MEMW_R tables built on the card; sent as records though they had
    /// the run's ECALL cycles.
    pub derived_memw_r: u64,
    pub records_memw_r: u64,
    /// Register rows sent as their timestamps; sent whole, by why
    /// ([`WholeRows`]).
    pub register_words: u64,
    pub whole: WholeRows,
    /// Bytes of the register rows' words sent; of the rows sent whole; of
    /// MEMW_R's records not sent (in `records_bytes_spared` too).
    pub register_word_bytes: u64,
    pub register_whole_bytes: u64,
    pub register_records_spared: u64,
    /// Seconds the host spent planning the register rows, summed over the
    /// tables; and the uploads' own wait for those plans, summed.
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
        records_memw_r: l(&c.records_memw_r),
        register_words: l(&c.register_words),
        whole: WholeRows {
            ecall: l(&c.whole_ecall),
            after_ecall: l(&c.whole_after_ecall),
            before_table: l(&c.whole_before_table),
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
        &c.records_memw_r,
        &c.register_words,
        &c.whole_ecall,
        &c.whole_after_ecall,
        &c.whole_before_table,
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
/// cache as they stand. Under [`Mode::Registers`] it ends with MEMW_R's part
/// (` · MEMW_R derived …`).
pub fn report_line() -> String {
    let s = stats();
    let (entries, bytes) = held();
    let registers = if matches!(mode(), Mode::Registers | Mode::Words) {
        let w = s.whole;
        format!(
            " · MEMW_R derived {} of {} · register rows as words {} · whole {} (ECALL {}, after one {}, before their table {}, not held {}, other {}) · MEMW_R records not sent {} B · words {} B · whole rows {} B · planned in {:.3} s summed, {:.3} s waited",
            s.derived_memw_r,
            s.derived_memw_r + s.records_memw_r,
            s.register_words,
            w.total(),
            w.ecall,
            w.after_ecall,
            w.before_table,
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
// The register rows' plan: which rows cross as their timestamps, which whole.
// ---------------------------------------------------------------------------

/// Cycles before a MEMW_R chunk's first row's cycle whose ECALLs its rows may
/// name: a row's old timestamp is at most 2^16 before its own (MEMW_R's step,
/// `0 < ts − old ≤ 2^16`), so at most 2^14 cycles and a slot before it.
pub const ECALL_SPAN_BEFORE: u64 = (1 << 14) + 1;

/// Why register rows cross whole instead of as their timestamps.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WholeRows {
    /// An ECALL cycle's rows (its handler's; no log holds them).
    pub ecall: u64,
    /// A write whose register an ECALL touched last (the value it left is in
    /// no log).
    pub after_ecall: u64,
    /// A write whose register was last touched before its cycle's CPU table
    /// (the card holds no value for it).
    pub before_table: u64,
    /// A row whose cycle's logs the card does not hold.
    pub not_held: u64,
    /// A timestamp the rule does not read (no walk makes one).
    pub other: u64,
}

impl WholeRows {
    pub fn total(&self) -> u64 {
        self.ecall + self.after_ecall + self.before_table + self.not_held + self.other
    }

    fn add(&mut self, other: &Self) {
        self.ecall += other.ecall;
        self.after_ecall += other.after_ecall;
        self.before_table += other.before_table;
        self.not_held += other.not_held;
        self.other += other.other;
    }

    fn count(&mut self, why: Whole) {
        match why {
            Whole::Ecall => self.ecall += 1,
            Whole::AfterEcall => self.after_ecall += 1,
            Whole::BeforeTable => self.before_table += 1,
            Whole::NotHeld => self.not_held += 1,
            Whole::Other => self.other += 1,
        }
    }
}

/// Why one row crosses whole ([`WholeRows`]' fields).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Whole {
    Ecall,
    AfterEcall,
    BeforeTable,
    NotHeld,
    Other,
}

/// A MEMW_R table as it crosses when the card builds its rows
/// ([`plan_registers`]): a word a row, and the rows that cross whole.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RegisterPlan {
    /// Row `r`'s word: `timestamp | old_timestamp << 32` for a row the card
    /// builds from the logs; `k << 32` (its low word 0, which no timestamp
    /// is) for the `k`-th row of `whole`.
    pub words: Vec<u64>,
    pub whole: Vec<RegRowC>,
    pub why: WholeRows,
    /// The first and the last row's cycles.
    pub span: (u64, u64),
}

/// A word's low half for a row that crosses whole.
pub const WHOLE_ROW: u32 = 0;

/// Rows a part of a table's plan: the parts are planned on
/// [`PLAN_THREADS`] threads of their own.
const PLAN_PART: usize = 1 << 18;
/// The planning pool's threads. They take no lock of this module, so an
/// upload holding the cache's lock may wait on them.
const PLAN_THREADS: usize = 4;

/// The ECALL cycles of `[base, hi]`, a bit each.
struct EcallBits {
    base: u64,
    len: u64,
    words: Vec<u64>,
}

impl EcallBits {
    /// From `ecalls`, ascending; those outside `[base, hi]` are left out.
    fn new(base: u64, hi: u64, ecalls: &[u32]) -> Self {
        let len = hi - base + 1;
        let mut words = vec![0u64; len.div_ceil(64) as usize];
        let from = ecalls.partition_point(|&e| u64::from(e) < base);
        for &e in &ecalls[from..] {
            let at = u64::from(e) - base;
            if at >= len {
                break;
            }
            words[(at / 64) as usize] |= 1 << (at % 64);
        }
        Self { base, len, words }
    }

    /// Whether `cycle` is an ECALL's; `None` outside the bits' cycles.
    fn get(&self, cycle: u64) -> Option<bool> {
        let at = cycle.checked_sub(self.base).filter(|&at| at < self.len)?;
        Some(self.words[(at / 64) as usize] >> (at % 64) & 1 == 1)
    }
}

/// What a table's plan reads besides its rows.
struct Rule<'a> {
    held: &'a [(u64, u64)],
    bits: EcallBits,
    lo: u64,
    hi: u64,
    faults: u32,
}

impl Rule<'_> {
    /// The first cycle of the held range holding `cycle`, from the range
    /// last found (`at`) when it still holds it.
    fn table_of(&self, cycle: u64, at: &mut usize) -> Option<u64> {
        if let Some(&(first, n)) = self.held.get(*at)
            && first <= cycle
            && cycle < first + n
        {
            return Some(first);
        }
        let k = self.held.partition_point(|&(first, _)| first <= cycle);
        let (first, n) = *self.held.get(k.checked_sub(1)?)?;
        (cycle < first + n).then(|| {
            *at = k - 1;
            first
        })
    }

    /// Row `row`'s word, or why it crosses whole.
    fn word(&self, row: &RegRowC, at: &mut usize) -> Result<u64, Whole> {
        let (ts, old) = (row.timestamp, u64::from(row.old_ts_lo));
        // MEMW_R's step, and a timestamp the word holds.
        if ts < 4 || ts > u64::from(u32::MAX) || old >= ts || ts - old > 1 << 16 {
            return Err(Whole::Other);
        }
        let (cycle, slot) = ((ts - 4) / 4, (ts - 4) % 4);
        if slot == 3 || cycle < self.lo || cycle > self.hi {
            return Err(Whole::Other);
        }
        // Outside the bits (never, for a row in MEMW_R's step) reads as an
        // ECALL: the row crosses whole.
        let ecalls = self.faults & REGISTER_FAULT_ECALL == 0;
        if ecalls && self.bits.get(cycle) != Some(false) {
            return Err(Whole::Ecall);
        }
        let first = self.table_of(cycle, at).ok_or(Whole::NotHeld)?;
        let write = row.is_read == 0;
        if write != (slot == 2) {
            return Err(Whole::Other);
        }
        if write {
            if old < 4 * first + 4 {
                if self.faults & REGISTER_FAULT_START == 0 {
                    return Err(Whole::BeforeTable);
                }
            } else {
                if (old - 4) % 4 == 3 {
                    return Err(Whole::Other);
                }
                if ecalls && self.bits.get((old - 4) / 4) != Some(false) {
                    return Err(Whole::AfterEcall);
                }
            }
        }
        Ok(ts | old << 32)
    }

    /// The words of `rows` into `out` (one a row; [`WHOLE_ROW`] for a row
    /// that crosses whole), and the rows that cross whole with their places
    /// in `rows`.
    fn plan(
        &self,
        rows: &[RegRowC],
        out: &mut [std::mem::MaybeUninit<u64>],
    ) -> (Vec<(usize, RegRowC)>, WholeRows) {
        let (mut whole, mut why, mut at) = (Vec::new(), WholeRows::default(), 0usize);
        for (i, (row, word)) in rows.iter().zip(out.iter_mut()).enumerate() {
            match self.word(row, &mut at) {
                Ok(w) => {
                    word.write(w);
                }
                Err(reason) => {
                    word.write(u64::from(WHOLE_ROW));
                    why.count(reason);
                    whole.push((i, *row));
                }
            }
        }
        (whole, why)
    }
}

fn plan_pool() -> Option<&'static rayon::ThreadPool> {
    static POOL: OnceLock<Option<rayon::ThreadPool>> = OnceLock::new();
    POOL.get_or_init(|| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(PLAN_THREADS)
            .thread_name(|i| format!("card-rows-plan-{i}"))
            .build()
            .ok()
    })
    .as_ref()
}

/// A MEMW_R table's plan when the card builds its rows from the CPU logs of
/// the cycles `held` names (first cycle, count; ascending, disjoint, each a
/// CPU table), `ecalls` the ECALL cycles (ascending) from
/// [`ECALL_SPAN_BEFORE`] cycles before its first row's to its last row's,
/// and `faults` a test's host mutants ([`REGISTER_FAULT_START`],
/// [`REGISTER_FAULT_ECALL`]; 0 in a prove).
///
/// A row crosses as its timestamp and old timestamp when the card can read
/// the rest from the logs: its cycle's logs are held, the cycle is no ECALL,
/// and for a write, the access its old timestamp names lies in the same CPU
/// table and in no ECALL cycle (the value then is in that cycle's log, by
/// its slot). Any other row crosses whole ([`WholeRows`] says why). `None`
/// (the table goes up as records) when fewer than half of the span's cycles
/// are held, or more than half of the rows would cross whole.
pub fn plan_registers(
    pieces: &[&[RegRowC]],
    held: &[(u64, u64)],
    ecalls: &[u32],
    faults: u32,
) -> Option<RegisterPlan> {
    plan_registers_many(&[(pieces, ecalls)], held, faults)
        .0
        .pop()
        .flatten()
}

/// A MEMW_R table as [`plan_registers_many`] takes it: its rows' pieces in
/// order, and its ECALL cycles.
pub type RegisterTable<'a> = (&'a [&'a [RegRowC]], &'a [u32]);

/// [`plan_registers`] of each table, their parts planned on the planning
/// pool's threads together; and the seconds those threads spent, summed.
pub fn plan_registers_many(
    tables: &[RegisterTable<'_>],
    held: &[(u64, u64)],
    faults: u32,
) -> (Vec<Option<RegisterPlan>>, f64) {
    use rayon::prelude::*;
    use std::mem::MaybeUninit;
    // Each table's rule, when at least half of its span is held.
    let rules: Vec<Option<(Rule<'_>, usize)>> = tables
        .iter()
        .map(|&(pieces, ecalls)| {
            let n: usize = pieces.iter().map(|p| p.len()).sum();
            let first = pieces.iter().find_map(|p| p.first())?;
            let last = pieces.iter().rev().find_map(|p| p.last())?;
            let cycle = |ts: u64| ts.checked_sub(4).map(|t| t / 4);
            let (lo, hi) = (cycle(first.timestamp)?, cycle(last.timestamp)?);
            if lo > hi {
                return None;
            }
            let covered: u64 = held
                .iter()
                .map(|&(f, len)| (f + len).min(hi + 1).saturating_sub(f.max(lo)))
                .sum();
            if covered * 2 < hi - lo + 1 {
                return None;
            }
            let bits = EcallBits::new(lo.saturating_sub(ECALL_SPAN_BEFORE), hi, ecalls);
            Some((
                Rule {
                    held,
                    bits,
                    lo,
                    hi,
                    faults,
                },
                n,
            ))
        })
        .collect();
    let mut words: Vec<Vec<u64>> = rules
        .iter()
        .map(|r| Vec::with_capacity(r.as_ref().map_or(0, |r| r.1)))
        .collect();
    // The parts: (table, first row, rows, their words).
    type Job<'r, 'w> = (usize, usize, &'r [RegRowC], &'w mut [MaybeUninit<u64>]);
    let mut jobs: Vec<Job<'_, '_>> = Vec::new();
    for (t, (rule, out)) in rules.iter().zip(words.iter_mut()).enumerate() {
        let Some((_, n)) = rule else { continue };
        let mut rest = &mut out.spare_capacity_mut()[..*n];
        let mut at = 0usize;
        for piece in tables[t].0 {
            for part in piece.chunks(PLAN_PART) {
                let (mine, others) = std::mem::take(&mut rest).split_at_mut(part.len());
                jobs.push((t, at, part, mine));
                rest = others;
                at += part.len();
            }
        }
    }
    let run = |(t, at, rows, out): Job<'_, '_>| {
        let clock = std::time::Instant::now();
        let rule = &rules[t].as_ref().expect("a planned table").0;
        let (whole, why) = rule.plan(rows, out);
        (t, at, whole, why, clock.elapsed().as_nanos() as u64)
    };
    let done: Vec<_> = match plan_pool() {
        Some(pool) => pool.install(|| jobs.into_par_iter().map(run).collect()),
        None => jobs.into_iter().map(run).collect(),
    };
    let mut plans: Vec<Option<RegisterPlan>> = Vec::with_capacity(tables.len());
    for (t, (rule, mut w)) in rules.iter().zip(words).enumerate() {
        let Some((rule, n)) = rule else {
            plans.push(None);
            continue;
        };
        // SAFETY: the table's parts cover its `n` words, one after another
        // (`rest` above), and `Rule::plan` wrote each word of its part.
        unsafe { w.set_len(*n) };
        let mut plan = RegisterPlan {
            words: w,
            span: (rule.lo, rule.hi),
            ..RegisterPlan::default()
        };
        for (_, at, whole, why, _) in done.iter().filter(|d| d.0 == t) {
            plan.why.add(why);
            for &(i, row) in whole {
                plan.words[at + i] = (plan.whole.len() as u64) << 32;
                plan.whole.push(row);
            }
        }
        plans.push((plan.whole.len() * 2 <= *n).then_some(plan));
    }
    let secs = done.iter().map(|d| d.4).sum::<u64>() as f64 * 1e-9;
    (plans, secs)
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
/// its words and whole rows ([`RegisterPlan`]).
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
    /// indexes `whole`; 0 for a planned table).
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
    /// A MEMW_R table built from the cached logs.
    Registers(RegisterPlan),
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
/// holds (LOAD or STORE derived, MEMW_R built from a plan or from words):
/// then its cached logs go up as descriptors and the derive kernels launch.
/// A table planned so and never launched would keep whatever its store held.
fn launches_from_logs(plans: &[Plan]) -> bool {
    plans.iter().any(|p| match p {
        Plan::Derived { .. } | Plan::Registers(_) | Plan::Words(_) => true,
        Plan::Wide | Plan::Narrow | Plan::Records | Plan::CpuHit(_) => false,
    })
}

/// A group's upload under [`Mode::Order`], [`Mode::Derive`] or
/// [`Mode::Registers`] ([`DeviceColumns::upload_tables`] calls it). `None` as
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
    // Only `derive` and `registers` read and write the cache: `order` takes
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
    let registers = mode == Mode::Registers;

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
        // MEMW_R's tables that came with their ECALL cycles, planned together
        // on the planning pool: each row as its timestamps where the logs the
        // card will hold give the rest, else whole ([`plan_registers`]).
        let wanted: Vec<(usize, RegisterTable<'_>)> = if registers && all_one {
            tables
                .iter()
                .enumerate()
                .filter_map(|(i, table)| match table {
                    TableUpload::Records(RecordsInput::MemwR {
                        records,
                        rows,
                        ecall_cycles: Some(ecalls),
                    }) if records_len(records) == *rows => Some((i, (*records, *ecalls))),
                    _ => None,
                })
                .collect()
        } else {
            Vec::new()
        };
        if !wanted.is_empty() {
            let held: Vec<(u64, u64)> = held.iter().map(|(&first, &n)| (first, n)).collect();
            let faults = FAULT_REGISTERS.load(Ordering::Relaxed) & !KERNEL_FAULTS;
            let clock = std::time::Instant::now();
            let inputs: Vec<RegisterTable<'_>> = wanted.iter().map(|&(_, table)| table).collect();
            let (planned, secs) = plan_registers_many(&inputs, &held, faults);
            add(&counters().plan_ns, (secs * 1e9) as u64);
            add(&counters().plan_wall_ns, clock.elapsed().as_nanos() as u64);
            for (&(i, _), plan) in wanted.iter().zip(planned) {
                let Some(plan) = plan else { continue };
                let (lo, hi) = plan.span;
                for (&first, e) in cache.entries.range_mut(..=hi) {
                    if first + e.n as u64 > lo {
                        e.tick = tick;
                    }
                }
                plans[i] = Plan::Registers(plan);
            }
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
    // A MEMW_R table with its ECALL cycles that goes up as records.
    if registers {
        for (table, plan) in tables.iter().zip(&plans) {
            if let (
                TableUpload::Records(RecordsInput::MemwR {
                    ecall_cycles: Some(_),
                    ..
                }),
                Plan::Records,
            ) = (table, plan)
            {
                add(&counters().records_memw_r, 1);
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
            (_, Plan::Registers(plan)) => {
                (plan.words.len() * 8 + plan.whole.len() * std::mem::size_of::<RegRowC>()) as u64
            }
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
            (TableUpload::Records(records), Plan::Registers(plan)) => {
                let word_bytes = plan.words.len() * 8;
                let whole_bytes = std::mem::size_of_val(plan.whole.as_slice());
                crate::pinned::note_upload(plan.words.as_ptr().cast(), word_bytes);
                let words = stream.clone_htod(&plan.words).ok()?;
                let whole = if plan.whole.is_empty() {
                    None
                } else {
                    crate::pinned::note_upload(plan.whole.as_ptr().cast(), whole_bytes);
                    Some(stream.clone_htod(&plan.whole).ok()?)
                };
                r = Some((words, whole, 0, plan.words.len() as u64));
                let c = counters();
                add(&c.derived_memw_r, 1);
                add(
                    &c.register_words,
                    (plan.words.len() - plan.whole.len()) as u64,
                );
                let w = plan.why;
                add(&c.whole_ecall, w.ecall);
                add(&c.whole_after_ecall, w.after_ecall);
                add(&c.whole_before_table, w.before_table);
                add(&c.whole_not_held, w.not_held);
                add(&c.whole_other, w.other);
                add(&c.register_word_bytes, word_bytes as u64);
                add(&c.register_whole_bytes, whole_bytes as u64);
                add(&c.records_bytes_spared, records.bytes() as u64);
                add(&c.register_records_spared, records.bytes() as u64);
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
                Plan::Registers(_) | Plan::Words(_),
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
        assert_eq!(Mode::parse(Some(" derive ")), Ok(Mode::Derive));
        assert_eq!(Mode::parse(Some("1")), Ok(Mode::Derive));
        assert_eq!(Mode::parse(Some("registers")), Ok(Mode::Registers));
        assert!(Mode::parse(Some("yes")).is_err());
        assert!(Mode::parse(Some("Derive")).is_err());
        assert!(Mode::parse(Some("2")).is_err());
        assert!(Mode::Registers.derives() && Mode::Derive.derives());
        assert!(!Mode::Order.derives() && !Mode::Off.derives());
        assert!(Mode::Words.derives());
        assert_eq!(DEFAULT_CACHE_MIB, 768);
    }

    /// A register row at `ts` (cycle `(ts − 4) / 4`, slot `(ts − 4) % 4`)
    /// whose old timestamp is `old`; a write when its slot is 2.
    fn reg(ts: u64, old: u64) -> RegRowC {
        RegRowC {
            timestamp: ts,
            val0: (ts * 7) as u32,
            val1: 1,
            old0: (old * 3) as u32,
            old1: 2,
            old_ts_lo: old as u32,
            address: 5,
            is_read: u8::from((ts - 4) % 4 != 2),
            pad: 0,
        }
    }

    fn ts(cycle: u64, slot: u64) -> u64 {
        4 * cycle + 4 + slot
    }

    /// Every row as the plan sends it: the words of the rows it builds
    /// carry their timestamps, and a whole row's word names it, in order.
    fn check_words(rows: &[RegRowC], plan: &RegisterPlan) {
        assert_eq!(plan.words.len(), rows.len());
        let mut next = 0u64;
        for (r, (row, &word)) in rows.iter().zip(&plan.words).enumerate() {
            if word as u32 == WHOLE_ROW {
                assert_eq!(word >> 32, next, "row {r}: whole rows in order");
                assert_eq!(plan.whole[next as usize], *row, "row {r}");
                next += 1;
            } else {
                assert_eq!(
                    word,
                    row.timestamp | u64::from(row.old_ts_lo) << 32,
                    "row {r}"
                );
            }
        }
        assert_eq!(next as usize, plan.whole.len());
        assert_eq!(plan.why.total(), next);
    }

    /// The rule, row by row: CPU tables of 8 cycles, an ECALL at cycle 10.
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
        assert!(launches_from_logs(&[Plan::Registers(
            RegisterPlan::default()
        )]));
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
    fn a_register_row_crosses_as_its_timestamps_only_where_the_logs_give_the_rest() {
        let held = [(0u64, 8u64), (8, 8), (16, 8)];
        let ecalls = [10u32];
        let rows = [
            reg(ts(2, 0), ts(1, 2)),  // a read: word
            reg(ts(2, 2), ts(2, 0)),  // a write, old this cycle's rs1: word
            reg(ts(5, 2), ts(3, 1)),  // a write, old in its table: word
            reg(ts(9, 2), ts(6, 0)),  // a write, old before its table (8)
            reg(ts(10, 0), ts(9, 2)), // the ECALL's own row
            reg(ts(10, 0), ts(4, 1)), // its handler's rows
            RegRowC {
                is_read: 0,
                ..reg(ts(10, 1), ts(10, 0))
            }, // and one written at the `rs2` slot
            reg(ts(11, 0), ts(10, 0)), // a read after it: word
            reg(ts(12, 2), ts(10, 1)), // a write whose register the ECALL touched
            reg(ts(13, 2), ts(12, 2)), // a write, old the last write: word
            reg(ts(14, 1), ts(14, 1)), // an old timestamp not before it: whole
            reg(ts(15, 0), ts(14, 1)), // reads: words
            reg(ts(16, 0), ts(15, 0)), //
            reg(ts(20, 2), ts(17, 1)), // the next table: word
            reg(ts(22, 3), ts(21, 0)), // slot 3: no cycle access is there
        ];
        let plan = plan_registers(&[&rows[..]], &held, &ecalls, 0).expect("planned");
        check_words(&rows, &plan);
        let whole: Vec<usize> = (0..rows.len())
            .filter(|&r| plan.words[r] as u32 == WHOLE_ROW)
            .collect();
        assert_eq!(whole, vec![3, 4, 5, 6, 8, 10, 14]);
        assert_eq!(
            plan.why,
            WholeRows {
                ecall: 3,
                after_ecall: 1,
                before_table: 1,
                not_held: 0,
                other: 2,
            }
        );
        assert_eq!(plan.span, (2, 22));

        // The host mutants: the rows the card cannot read go as words.
        let start =
            plan_registers(&[&rows[..]], &held, &ecalls, REGISTER_FAULT_START).expect("planned");
        assert_ne!(
            start.words[3] as u32, WHOLE_ROW,
            "before its table, as a word"
        );
        assert_eq!(start.why.before_table, 0);
        let ecall =
            plan_registers(&[&rows[..]], &held, &ecalls, REGISTER_FAULT_ECALL).expect("planned");
        for r in [4, 5, 8] {
            assert_ne!(ecall.words[r] as u32, WHOLE_ROW, "row {r} as a word");
        }
        assert_eq!((ecall.why.ecall, ecall.why.after_ecall), (0, 0));
        // Row 6 (a write at the `rs2` slot) stays whole.
        assert_eq!(ecall.words[6] as u32, WHOLE_ROW);
        assert_eq!(ecall.why.other, 3);
    }

    #[test]
    fn rows_whose_cycles_are_not_held_cross_whole_and_too_many_send_records() {
        let rows: Vec<RegRowC> = (2..22).map(|c| reg(ts(c, 0), ts(c - 1, 1))).collect();
        // The middle table missing: its rows whole.
        let plan = plan_registers(&[&rows[..]], &[(0, 8), (16, 8)], &[], 0).expect("planned");
        check_words(&rows, &plan);
        assert_eq!(plan.why.not_held, 8);
        // Under half the span held: records.
        assert!(plan_registers(&[&rows[..]], &[(0, 8)], &[], 0).is_none());
        // Half the span held, more than half the rows whole: records.
        let late: Vec<RegRowC> = (2..22)
            .map(|c| reg(ts(c, 0), ts(c - 1, 1)))
            .chain((14..16).map(|c| reg(ts(c, 0), ts(c - 1, 1))))
            .collect();
        assert!(plan_registers(&[&late[..]], &[(0, 12)], &[], 0).is_none());
        // Nothing held, no rows: records.
        assert!(plan_registers(&[&rows[..]], &[], &[], 0).is_none());
        assert!(plan_registers(&[], &[(0, 8)], &[], 0).is_none());
    }

    /// A table larger than a planning part, in pieces, its whole rows spread
    /// over the parts: every word and whole row at its place, and the same
    /// plan however the rows are cut into pieces.
    #[test]
    fn a_large_tables_parts_are_planned_apart_and_put_back_in_order() {
        let cycles = (PLAN_PART as u64) * 3 / 2 + 77;
        let ecalls: Vec<u32> = (0..cycles as u32).filter(|c| c % 1001 == 7).collect();
        let rows: Vec<RegRowC> = (1..cycles)
            .flat_map(|c| [reg(ts(c, 0), ts(c - 1, 0)), reg(ts(c, 2), ts(c, 0))])
            .collect();
        let held = [(0u64, cycles / 2), (cycles / 2, cycles)];
        let one = plan_registers(&[&rows[..]], &held, &ecalls, 0).expect("planned");
        check_words(&rows, &one);
        // Each ECALL cycle's two rows; the write after it reads its own
        // cycle's `rs1`, a read the ECALL did not make.
        assert_eq!(
            one.why.ecall,
            2 * ecalls.iter().filter(|&&c| c >= 1).count() as u64
        );
        // The first write of the second table reads the cycle before it.
        assert_eq!(one.why.before_table, 0);
        let (a, b) = rows.split_at(PLAN_PART + 12345);
        let (b, c) = b.split_at(999);
        let cut = plan_registers(&[a, b, c], &held, &ecalls, 0).expect("planned");
        assert_eq!(cut, one);
        // The planning pool and one thread agree.
        let (many, _) = plan_registers_many(&[(&[&rows[..]][..], &ecalls[..])], &held, 0);
        assert_eq!(many[0].as_ref(), Some(&one));
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
