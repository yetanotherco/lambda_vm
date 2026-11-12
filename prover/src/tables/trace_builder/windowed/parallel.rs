//! A run's windows walked from their entries, on several threads
//! ([`walk_windows`], `LAMBDA_VM_BLOCK_WALKERS`).
//!
//! The lean pass ([`super::lean_pass::LeanPass`]) gives each window, before it
//! is walked, its first cycle, its starting register cells and its entries
//! (each byte it touches, with the byte's timestamp before it); the executor's
//! snapshot at the window's start gives the entries' values. A
//! [`WindowWalker`] seeds a window's state from them and walks the window with
//! the walk's own code ([`super::walk_window`]), so the window is the one the
//! sequential walk walks. Three checks keep a mistake from becoming a wrong
//! trace:
//! - a byte the window reads that its entries do not hold reads [`ABSENT`],
//!   and the window is refused;
//! - after the window, its registers must be the next window's starting
//!   registers, and every byte it holds that the next window touches must hold
//!   the next window's entry (the executor's value, the pass's timestamp);
//! - the run's last walked window must agree, byte for byte, with the state
//!   handed to the finish ([`super::lean_pass::LeanPass::walked_state`]);
//! - the windows' boundaries must add up ([`cell_hash`]): each window's
//!   seeded bytes A and the same bytes at its end B, hashed and summed, give
//!   Σ (B − A) over the run equal to the sum, over every byte a window
//!   touched, of its cell handed to the finish less its cell at the run's
//!   start. That catches what the exit checks cannot see: an entry's
//!   timestamp or value wrong on a byte the window then writes, whose end is
//!   right. (An entry wrong on a byte the window only reads ends as it was
//!   seeded and adds nothing to the sums: the exit checks see it where the
//!   next window touches the byte, and the verifier refuses its rows where
//!   none does.) An entry stamped 0 must carry the image's value.
//!
//! The threads: the executor (the caller's) sends each window with its
//! snapshot; one thread runs the pass and hands out each window as a job as it
//! arrives, but the one that halts, which is the run's last and the finish's;
//! `walkers` threads walk the jobs; the calling thread absorbs the walked
//! windows strictly in order. At most `walkers + 2` windows are handed out and
//! not yet absorbed.

use std::collections::BTreeMap;
use std::hash::BuildHasher;
use std::sync::mpsc::{Receiver, SyncSender, sync_channel};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use executor::vm::history::{HistoryFinal, RegisterCells, WindowHistory};
use executor::vm::logs::Log;
use executor::vm::memory::MemorySnapshot;

use super::super::{ABSENT, HistoryOverlay, MemoryState, NO_PAGE, RegisterState, WalkSizes};
use super::head_log;
use super::lean_pass::LeanPass;
use super::{Accumulator, WalkedWindow, WindowedTraceBuilder, walk_window};
use crate::Error;
use crate::paged_mem::PagedMem;
use crate::tables::decode::DecodeTable;
use crate::tables::page::DEFAULT_PAGE_SIZE;

/// A window as the executor sends it: its logs and either the memory at its
/// start (walked from the lean pass's entries) or its history (record mode,
/// [`walk_windows_from_history`]); the run's last window in record mode also
/// carries the run's end.
pub struct ExecWindow {
    pub logs: Arc<Vec<Log>>,
    pub start: Option<MemorySnapshot>,
    pub history: Option<Box<WindowHistory>>,
    pub last: Option<Box<HistoryFinal>>,
}

impl<L: Into<Arc<Vec<Log>>>> From<(L, Option<MemorySnapshot>)> for ExecWindow {
    fn from((logs, start): (L, Option<MemorySnapshot>)) -> Self {
        Self {
            logs: logs.into(),
            start,
            history: None,
            last: None,
        }
    }
}

/// The walk's register cells, as the executor's history holds them.
fn register_state(cells: &RegisterCells) -> RegisterState {
    RegisterState {
        regs: cells.regs,
        index_register: cells.index,
        pc_register: cells.pc,
    }
}

/// A window to walk from its entries.
struct WindowJob {
    /// Its number in the run, from 0.
    window: u32,
    /// Its first cycle.
    first: usize,
    logs: Arc<Vec<Log>>,
    /// The executor's memory at its start.
    start: Arc<MemorySnapshot>,
    /// The register cells at its start.
    registers: RegisterState,
    /// Each byte it touches and the byte's timestamp before it.
    entries: Arc<Vec<(u64, u64)>>,
    /// Record mode: the window's history (no entries, no start memory), and
    /// whether the check thread re-walks it (its walker then sends the
    /// digest of what it emitted, [`WalkSettings::digests`]).
    history: Option<Arc<WindowHistory>>,
    recheck: bool,
}

/// The walk's settings a window is walked with.
struct WalkSettings {
    decode: Arc<DecodeTable>,
    lookups: bool,
    batch: usize,
    /// [`WindowedTraceBuilder::defer_ecrec_witnesses`].
    defer_ecrec: bool,
    /// The run's key for [`cell_hash`].
    key: u64,
    /// Keep each window's logs with it ([`WindowedTraceBuilder::keeps_logs`]).
    keep_logs: bool,
    /// Record mode's check: where a walker sends the digest of a rechecked
    /// window's lists ([`check_thread`]).
    digests: Option<Mutex<std::sync::mpsc::Sender<(u32, u64)>>>,
}

/// One byte's cell hashed for the windows' boundary sums, under the run's
/// `key` so that no mistake with a structure cancels out in them; one-to-one
/// in `(timestamp, value)` for a given address. Timestamps stay below 2^40.
#[inline]
pub(super) fn cell_hash(key: u64, address: u64, timestamp: u64, value: u8) -> u64 {
    // splitmix64's finalizer.
    fn mix(mut x: u64) -> u64 {
        x ^= x >> 30;
        x = x.wrapping_mul(0xbf58_476d_1ce4_e5b9);
        x ^= x >> 27;
        x = x.wrapping_mul(0x94d0_49bb_1331_11eb);
        x ^ (x >> 31)
    }
    mix(mix(address ^ key) ^ ((timestamp << 8) | u64::from(value)))
}

/// The windows' boundary sums against the run's: refused unless they are the
/// same ([`cell_hash`]).
fn check_boundaries(windows: u64, run: u64) -> Result<(), Error> {
    if windows != run {
        return Err(Error::Prover(format!(
            "the windows' boundaries do not add up: the walked windows give {windows:#018x}, the \
             bytes they touched {run:#018x}"
        )));
    }
    Ok(())
}

/// One walker: its window state, kept from window to window so its pages are
/// reused, and what it walked last.
struct WindowWalker {
    settings: Arc<WalkSettings>,
    sizes: WalkSizes,
    /// The window's state: every byte it does not hold reads `(0, ABSENT)`.
    memory: MemoryState,
    /// The registers at the end of the window walked last.
    registers: RegisterState,
    /// The bytes the window walked last touched (reset to [`ABSENT`] before
    /// the next is seeded), and the state pages it touched.
    touched: Vec<u64>,
    pages: Vec<u64>,
    /// The seconds the window walked last took to seed.
    seeding: f64,
    /// Record mode's state: cells from the history only, no pages.
    record: MemoryState,
    /// Whether `memory` holds the last window's walk (a record-mode window
    /// rechecked, or any window walked from its entries).
    seeded: bool,
}

/// The state page holding `address`.
fn page_of(address: u64) -> u64 {
    address & !(DEFAULT_PAGE_SIZE as u64 - 1)
}

impl WindowWalker {
    fn new(settings: Arc<WalkSettings>) -> Self {
        let mut memory = MemoryState::new();
        memory.cells = PagedMem::new((0, ABSENT));
        memory.lean = true;
        Self {
            settings,
            sizes: WalkSizes::default(),
            memory,
            registers: RegisterState::new(0),
            touched: Vec::new(),
            pages: Vec::new(),
            seeding: 0.0,
            record: MemoryState::new(),
            seeded: false,
        }
    }

    /// Walks a record-mode `job` from its history: no seeding, no window
    /// state. Refuses the window if a cycle read a byte, or found a register
    /// cell, other than its history gives. A rechecked job's lists go, as a
    /// digest, to the check thread, which walks the window again from an
    /// independent start state ([`check_thread`]).
    fn walk_record(
        &mut self,
        job: &WindowJob,
        history: &Arc<WindowHistory>,
    ) -> Result<(WalkedWindow, u64), Error> {
        let Some(cells) = &history.registers else {
            return Err(Error::Prover(format!(
                "window {}: its history has no register cells",
                job.window
            )));
        };
        self.seeding = 0.0;
        self.registers = register_state(cells);
        self.record.overlay = Some(Box::new(HistoryOverlay::new(Arc::clone(history))));
        let settings = Arc::clone(&self.settings);
        let s = &*settings;
        let walked = walk_window(
            &s.decode,
            s.lookups,
            s.batch,
            s.defer_ecrec,
            &mut self.sizes,
            &mut self.record,
            &mut self.registers,
            job.first,
            &job.logs,
        );
        self.record.overlay = None;
        let mut walked = walked?;
        if s.keep_logs {
            walked.keep_logs(Arc::clone(&job.logs));
        }
        if self.record.take_missing() {
            return Err(Error::Prover(format!(
                "window {}: the walk read a byte, or found a register cell, other than the \
                 executor's history gives",
                job.window
            )));
        }
        self.seeded = false;
        if job.recheck
            && let Some(digests) = &s.digests
            && let Ok(digests) = digests.lock()
        {
            let _ = digests.send((job.window, walked.emitted_digest()));
        }
        Ok((walked, 0))
    }

    /// Walks `job` from its entries into fresh lists, with its boundary sum:
    /// its bytes at the end, hashed and summed, less the same at its start
    /// ([`cell_hash`]). Refuses the window if it read a byte its entries do
    /// not hold, or if an entry stamped 0 does not carry `image`'s value.
    fn walk(
        &mut self,
        job: &WindowJob,
        image: &PagedMem<u8>,
    ) -> Result<(WalkedWindow, u64), Error> {
        if let Some(history) = &job.history {
            return self.walk_record(job, history);
        }
        self.seeded = true;
        let t = Instant::now();
        let seeded = self.seed(job, image)?;
        self.seeding = t.elapsed().as_secs_f64();
        self.registers = job.registers.clone();
        let s = &*self.settings;
        let mut walked = walk_window(
            &s.decode,
            s.lookups,
            s.batch,
            s.defer_ecrec,
            &mut self.sizes,
            &mut self.memory,
            &mut self.registers,
            job.first,
            &job.logs,
        )?;
        if s.keep_logs {
            walked.keep_logs(Arc::clone(&job.logs));
        }
        if self.memory.take_missing() {
            return Err(Error::Prover(format!(
                "window {}: the walk read a byte its entries do not hold",
                job.window
            )));
        }
        let key = self.settings.key;
        let ended = job.entries.iter().fold(0u64, |sum, &(address, _)| {
            let (value, ts) = self.memory.cells.get(address);
            sum.wrapping_add(cell_hash(key, address, ts, value))
        });
        Ok((walked, ended.wrapping_sub(seeded)))
    }

    /// The state cleared of the last window's bytes, its pages kept if this
    /// window or the last touched them, then seeded with `job`'s entries; the
    /// seeded cells hashed and summed ([`cell_hash`]). Refuses an entry
    /// stamped 0 (no window wrote the byte before) whose value is not
    /// `image`'s.
    fn seed(&mut self, job: &WindowJob, image: &PagedMem<u8>) -> Result<u64, Error> {
        for &address in &self.touched {
            self.memory.write_byte(address, 0, ABSENT);
        }
        let mut pages: Vec<u64> = job.entries.iter().map(|&(a, _)| page_of(a)).collect();
        pages.sort_unstable();
        pages.dedup();
        let last = std::mem::replace(&mut self.pages, pages);
        let now = &self.pages;
        self.memory.cells.retain_pages(|base| {
            now.binary_search(&base).is_ok() || last.binary_search(&base).is_ok()
        });
        self.memory.hot = NO_PAGE;
        self.touched.clear();
        let (key, mut seeded) = (self.settings.key, 0u64);
        for &(address, ts) in job.entries.iter() {
            let value = job.start.load_byte(address);
            if ts == 0 && value != image.get(address) {
                return Err(Error::Prover(format!(
                    "window {}: the byte at {address:#x} has timestamp 0 but holds {value}, not the \
                     image's {}",
                    job.window,
                    image.get(address)
                )));
            }
            self.memory.write_byte(address, value, ts);
            self.touched.push(address);
            seeded = seeded.wrapping_add(cell_hash(key, address, ts, value));
        }
        self.memory.take_missing();
        Ok(seeded)
    }

    /// After walking window `job`: its registers must be `next_registers`,
    /// and each byte of `next_entries` it holds must hold the entry's
    /// timestamp and its value in `next_start`.
    fn check_exit(
        &self,
        job: &WindowJob,
        next_registers: &RegisterState,
        next_entries: &[(u64, u64)],
        next_start: &MemorySnapshot,
    ) -> Result<(), Error> {
        self.check_registers(job, next_registers)?;
        for &(address, ts) in next_entries {
            let held = self.memory.cells.get(address);
            if held.1 != ABSENT && held != (next_start.load_byte(address), ts) {
                return Err(Error::Prover(format!(
                    "window {}: it ends with {held:?} at {address:#x}, the next window starts from \
                     ({}, {ts})",
                    job.window,
                    next_start.load_byte(address)
                )));
            }
        }
        Ok(())
    }

    /// After walking the run's last walked window `job`: its registers must be
    /// `registers`, and every byte it touched must hold what `state` holds.
    fn check_exit_into(
        &self,
        job: &WindowJob,
        registers: &RegisterState,
        state: &MemoryState,
    ) -> Result<(), Error> {
        self.check_registers(job, registers)?;
        for &address in &self.touched {
            let (held, there) = (self.memory.cells.get(address), state.cells.get(address));
            if held != there {
                return Err(Error::Prover(format!(
                    "window {}: it ends with {held:?} at {address:#x}, the finish starts from \
                     {there:?}",
                    job.window
                )));
            }
        }
        Ok(())
    }

    fn check_registers(&self, job: &WindowJob, next: &RegisterState) -> Result<(), Error> {
        let (a, b) = (&self.registers, next);
        if a.regs != b.regs
            || a.index_register != b.index_register
            || a.pc_register != b.pc_register
        {
            return Err(Error::Prover(format!(
                "window {}: its registers at the end are not the next window's at its start",
                job.window
            )));
        }
        Ok(())
    }
}

/// A thread's error that only says another thread stopped first: never the
/// one [`walk_windows`] returns while there is another.
const WALKERS_STOPPED: &str = "the walkers stopped";
const PASS_STOPPED: &str = "the pass stopped before the window after it";

fn is_knock_on(e: &Error) -> bool {
    matches!(e, Error::Prover(m) if m == WALKERS_STOPPED || m.ends_with(PASS_STOPPED))
}

/// What follows a walked window, for its exit check.
enum After {
    /// The next window's start.
    Next {
        registers: RegisterState,
        entries: Arc<Vec<(u64, u64)>>,
        start: Arc<MemorySnapshot>,
    },
    /// The state handed to the finish (the window is the run's last walked).
    Finish {
        registers: RegisterState,
        state: Arc<MemoryState>,
    },
    /// Record mode: the next window's starting registers.
    NextHistory { registers: RegisterState },
}

impl WindowWalker {
    /// Walks `job`, then waits for what follows it and checks its exit
    /// against that. The busy seconds leave the wait out.
    fn walk_and_check(
        &mut self,
        job: &WindowJob,
        after: &Receiver<After>,
        image: &PagedMem<u8>,
    ) -> Result<(WalkedWindow, f64, u64), Error> {
        let t = Instant::now();
        let (walked, boundary) = self.walk(job, image)?;
        let mut busy = t.elapsed().as_secs_f64();
        head_log::window_walked(job.window as usize, t, self.seeding);
        let after = after
            .recv()
            .map_err(|_| Error::Prover(format!("window {}: {PASS_STOPPED}", job.window)))?;
        let t = Instant::now();
        match after {
            After::Next {
                registers,
                entries,
                start,
            } => self.check_exit(job, &registers, &entries, &start)?,
            After::Finish { registers, state } if self.seeded => {
                self.check_exit_into(job, &registers, &state)?
            }
            After::Finish { registers, .. } => self.check_registers(job, &registers)?,
            After::NextHistory { registers } => self.check_registers(job, &registers)?,
        }
        busy += t.elapsed().as_secs_f64();
        Ok((walked, busy, boundary))
    }

    /// The bytes its window state takes on the heap.
    fn state_bytes(&self) -> usize {
        self.memory.heap_bytes() + self.record.heap_bytes()
    }
}

/// A job, and where what follows it arrives.
struct Ticket {
    job: WindowJob,
    after: Receiver<After>,
}

/// What the pass thread ends with.
struct PassEnd {
    last: Arc<Vec<Log>>,
    state: Arc<MemoryState>,
    registers: RegisterState,
    /// The last window's first cycle, and the windows handed out.
    first: usize,
    windows: usize,
    busy: f64,
    /// Of `busy`, the finish's state.
    state_secs: f64,
    entries: usize,
    most_entries: usize,
    /// The bytes the windows touched: their cells handed to the finish less
    /// their cells at the run's start, hashed and summed ([`cell_hash`]).
    boundaries: u64,
}

/// What a run walked on several threads reports (the `BLOCK WALKERS` line).
pub struct ParallelWalk {
    /// The run's last window, the finish's.
    pub last: Vec<Log>,
    pub walkers: usize,
    pub windows: usize,
    pub pass_secs: f64,
    /// Of the pass's, the seconds building the finish's state.
    pub finish_state_secs: f64,
    /// The walkers' busy seconds, summed.
    pub walk_secs: f64,
    pub entries: usize,
    pub most_entries: usize,
    /// The most walked windows held, waiting for an earlier one.
    pub held_most: usize,
    /// The most bytes the walkers' window states took at once.
    pub state_bytes: usize,
    /// Walked from the executor's history ([`walk_windows_from_history`]).
    pub from_history: bool,
}

impl ParallelWalk {
    pub fn line(&self) -> String {
        format!(
            "BLOCK WALKERS: {} · pass busy {:.2}s · entries {} (most {} in one window) · walkers busy {:.2}s (summed) · reorder held at most {} · window states {:.2} GiB · {} windows walked from their entries · the finish's state built in {:.2}s",
            self.walkers,
            self.pass_secs,
            self.entries,
            self.most_entries,
            self.walk_secs,
            self.held_most,
            self.state_bytes as f64 / (1u64 << 30) as f64,
            self.windows,
            self.finish_state_secs,
        ) + if self.from_history {
            " · record mode: from the executor's history"
        } else {
            ""
        }
    }
}

/// The run's windows, as the executor sends them (each with its memory at the
/// window's start), walked on `walkers` threads from the lean pass's entries
/// and absorbed strictly in window order by `absorb`, which is also given the
/// bytes the walkers' window states take then. The builder then holds the
/// state the walk holds before the run's last window, which is returned (the
/// finish walks it as usual). The builder must not have walked a window yet.
/// The first error anywhere stops every thread and is returned; a walker that
/// panics is an error. A window's logs come owned (`Vec<Log>`) or shared
/// (`Arc<Vec<Log>>`, e.g. by an executor that takes the buffer back once
/// the walk lets it go).
#[cfg(test)]
pub fn walk_windows<W: Into<ExecWindow> + Send>(
    builder: &mut WindowedTraceBuilder<'_>,
    windows: Receiver<W>,
    walkers: usize,
    absorb: impl FnMut(&mut Accumulator<'_>, WalkedWindow, usize) -> Result<(), Error>,
) -> Result<ParallelWalk, Error> {
    walk_numbered(builder, windows, walkers, |k| k, None, absorb)
}

/// [`walk_windows`] (`record` `None`) or [`walk_windows_from_history`]
/// (`Some(recheck)`), as a caller's setting picks.
pub fn walk_windows_with<W: Into<ExecWindow> + Send>(
    builder: &mut WindowedTraceBuilder<'_>,
    windows: Receiver<W>,
    walkers: usize,
    record: Option<usize>,
    absorb: impl FnMut(&mut Accumulator<'_>, WalkedWindow, usize) -> Result<(), Error>,
) -> Result<ParallelWalk, Error> {
    walk_numbered(builder, windows, walkers, |k| k, record, absorb)
}

/// [`walk_windows`] in record mode: each window comes with the executor's
/// history ([`ExecWindow::history`]) and is walked from it, with no lean
/// pass, no entries, no start memory and no window state; the run's last
/// window brings the run's end ([`ExecWindow::last`]), from which the
/// finish's state is built. Every window's end registers are checked against
/// the next one's start, and each cycle's register accesses against the
/// walk's cells. With `recheck` > 0 a check thread re-derives the history
/// independently ([`check_thread`]): the lean pass over every window, and
/// every `recheck`-th window (which must come with its start memory)
/// re-walked from the pass's entries, its lists compared with its walker's.
#[cfg(test)]
pub fn walk_windows_from_history<W: Into<ExecWindow> + Send>(
    builder: &mut WindowedTraceBuilder<'_>,
    windows: Receiver<W>,
    walkers: usize,
    recheck: usize,
    absorb: impl FnMut(&mut Accumulator<'_>, WalkedWindow, usize) -> Result<(), Error>,
) -> Result<ParallelWalk, Error> {
    walk_numbered(builder, windows, walkers, |k| k, Some(recheck), absorb)
}

/// [`walk_windows`], each job numbered `number(k)` for the run's window `k`:
/// a test numbers them wrong to show the absorbing side refuses it.
fn walk_numbered<W: Into<ExecWindow> + Send>(
    builder: &mut WindowedTraceBuilder<'_>,
    windows: Receiver<W>,
    walkers: usize,
    number: fn(u32) -> u32,
    record: Option<usize>,
    mut absorb: impl FnMut(&mut Accumulator<'_>, WalkedWindow, usize) -> Result<(), Error>,
) -> Result<ParallelWalk, Error> {
    assert!(walkers >= 1, "walk_windows needs a walker");
    debug_assert_eq!(builder.cycles, 0, "walk_windows starts the run");
    // Record mode's check: the walkers' digests of the rechecked windows.
    let recheck = record.unwrap_or(0);
    let (digest_tx, digest_rx) = std::sync::mpsc::channel::<(u32, u64)>();
    let settings = Arc::new(WalkSettings {
        decode: Arc::clone(&builder.decode),
        lookups: !builder.lean.lookups,
        batch: builder.walk_batch,
        defer_ecrec: builder.defer_ecrec,
        key: std::collections::hash_map::RandomState::new().hash_one(0u64),
        keep_logs: builder.keeps_logs(),
        digests: (recheck > 0).then(|| Mutex::new(digest_tx)),
    });
    let key = settings.key;
    let start_registers = builder.register_state.clone();
    let lean = builder.lean.memory;
    let (end, walk_secs, held_most, state_bytes) = {
        let (mut accumulator, image) = builder.accumulator_and_image();
        let (job_tx, job_rx) = sync_channel::<Ticket>(walkers);
        let job_rx = Mutex::new(job_rx);
        std::thread::scope(|scope| -> Result<_, Error> {
            #[allow(clippy::type_complexity)]
            let (done_tx, done_rx) =
                sync_channel::<(u32, usize, Result<(WalkedWindow, f64, usize, u64), Error>)>(
                    walkers + 2,
                );
            let (permit_tx, permit_rx) = sync_channel::<()>(walkers + 2);
            let decode = Arc::clone(&settings.decode);
            // Record mode's check thread, fed every window in order.
            let (check_tx, check_rx) = sync_channel::<CheckWindow>(4);
            let check = (recheck > 0).then(|| {
                let (settings, registers) = (Arc::clone(&settings), start_registers.clone());
                scope.spawn(move || check_thread(settings, &registers, image, check_rx))
            });
            let check_tx = check.is_some().then_some(check_tx);
            let pass = scope.spawn(move || match record {
                Some(recheck) => record_thread(
                    &start_registers,
                    image,
                    lean,
                    windows,
                    number,
                    recheck,
                    check_tx,
                    job_tx,
                    permit_tx,
                ),
                None => pass_thread(
                    decode,
                    &start_registers,
                    image,
                    lean,
                    key,
                    windows,
                    number,
                    job_tx,
                    permit_tx,
                ),
            });
            let mut threads = Vec::with_capacity(walkers);
            for w in 0..walkers {
                let (job_rx, done_tx, settings) = (&job_rx, done_tx.clone(), Arc::clone(&settings));
                threads.push(scope.spawn(move || {
                    let mut walker = WindowWalker::new(settings);
                    // After an error (its own or the absorbing side's), the
                    // walker takes the jobs still coming and drops them, so the
                    // pass never waits on a full queue nobody reads; the
                    // closed permits stop the pass.
                    let mut stopped = false;
                    loop {
                        let ticket = match job_rx.lock() {
                            Ok(rx) => rx.recv(),
                            Err(_) => break,
                        };
                        let Ok(ticket) = ticket else {
                            break;
                        };
                        if stopped {
                            continue;
                        }
                        let window = ticket.job.window;
                        let outcome =
                            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                walker.walk_and_check(&ticket.job, &ticket.after, image)
                            }))
                            .unwrap_or_else(|_| {
                                Err(Error::Prover(format!(
                                    "window {window}: its walker panicked"
                                )))
                            })
                            .map(|(walked, busy, boundary)| {
                                (walked, busy, walker.state_bytes(), boundary)
                            });
                        let failed = outcome.is_err();
                        if done_tx.send((window, w, outcome)).is_err() || failed {
                            stopped = true;
                        }
                    }
                }));
            }
            drop(done_tx);
            let (mut held, mut next, mut held_most) = (BTreeMap::new(), 0u32, 0usize);
            // Each walker's window state as of its last window, and the most
            // they took at once.
            let (mut walk_secs, mut states, mut state_bytes) = (0.0, vec![0usize; walkers], 0);
            // The walked windows' boundary sums, added as they come.
            let mut boundaries = 0u64;
            // The first error from a walker or from absorbing; an absorbing
            // error is this thread's own and always wins.
            let (mut failure, mut absorbing) = (None, false);
            'windows: for (window, w, outcome) in &done_rx {
                let (walked, busy, bytes, boundary) = match outcome {
                    Ok(walked) => walked,
                    Err(e) => {
                        failure = Some(e);
                        break;
                    }
                };
                walk_secs += busy;
                boundaries = boundaries.wrapping_add(boundary);
                states[w] = bytes;
                state_bytes = state_bytes.max(states.iter().sum());
                if window < next || held.insert(window, walked).is_some() {
                    failure = Some(Error::Prover(format!("window {window} was walked twice")));
                    break;
                }
                held_most = held_most.max(held.len());
                while let Some(walked) = held.remove(&next) {
                    if let Err(e) = absorb(&mut accumulator, walked, states.iter().sum()) {
                        (failure, absorbing) = (Some(e), true);
                        break 'windows;
                    }
                    head_log::window_absorbed(next as usize);
                    // One window fewer handed out and not absorbed.
                    let _ = permit_rx.recv();
                    next += 1;
                }
                // On a correct run window `next` is handed out before every
                // window held, so the permits leave room for at most
                // `walkers + 1` held. With `walkers + 2` the pass waits for a
                // permit only window `next` frees: it was never handed out.
                if held.len() >= walkers + 2 {
                    failure = Some(Error::Prover(format!("window {next} was never walked")));
                    break;
                }
            }
            // On a failure the other threads see their channels close.
            drop(done_rx);
            drop(permit_rx);
            for thread in threads {
                if thread.join().is_err() {
                    failure.get_or_insert(Error::Prover("a walker thread panicked".into()));
                }
            }
            let end = pass
                .join()
                .map_err(|_| Error::Prover("the lean pass panicked".into()))?;
            // Record mode's check: its re-walks against the walkers' digests.
            if let Some(check) = check {
                let checked = check
                    .join()
                    .map_err(|_| Error::Prover("the history check panicked".into()))
                    .and_then(|walked| walked);
                let compared = checked.and_then(|walked| {
                    let mut got: std::collections::HashMap<u32, u64> =
                        digest_rx.try_iter().collect();
                    for (window, digest) in walked {
                        match got.remove(&window) {
                            Some(d) if d == digest => {}
                            Some(_) => {
                                return Err(Error::Prover(format!(
                                    "window {window}: walked from the executor's history it emits \
                                     other lists than walked from the lean pass's entries"
                                )));
                            }
                            None => {
                                return Err(Error::Prover(format!(
                                    "window {window}: re-walked, but its walker sent no digest"
                                )));
                            }
                        }
                    }
                    Ok(())
                });
                if let Err(e) = compared {
                    failure.get_or_insert(e);
                }
            }
            // The cause, not its knock-on: this thread's own error first, then
            // the pass's, then a walker's.
            match (failure, end) {
                (Some(e), _) if absorbing => Err(e),
                (Some(e), Err(p)) => Err(if is_knock_on(&p) { e } else { p }),
                (Some(e), Ok(_)) => Err(e),
                (None, Err(p)) => Err(p),
                (None, Ok(end)) if next as usize != end.windows || !held.is_empty() => {
                    Err(Error::Prover(format!(
                        "{} windows handed out, {next} absorbed in order",
                        end.windows
                    )))
                }
                (None, Ok(end)) => {
                    check_boundaries(boundaries, end.boundaries)?;
                    Ok((end, walk_secs, held_most, state_bytes))
                }
            }
        })?
    };
    let PassEnd {
        last,
        state,
        registers,
        first,
        windows,
        busy,
        state_secs,
        entries,
        most_entries,
        boundaries: _,
    } = end;
    let state = Arc::try_unwrap(state)
        .map_err(|_| Error::Prover("the finish's state is still shared".into()))?;
    builder.install_walked(state, registers, first, windows, walk_secs);
    Ok(ParallelWalk {
        last: Arc::try_unwrap(last).unwrap_or_else(|last| (*last).clone()),
        walkers,
        windows,
        pass_secs: busy,
        finish_state_secs: state_secs,
        walk_secs,
        entries,
        most_entries,
        held_most,
        state_bytes,
        from_history: record.is_some(),
    })
}

/// The pass's thread: each window passed and handed out as a job (numbered
/// `number(k)`) as it arrives, a permit taken for each; what follows each job
/// sent once the next window is passed. The window that halts is the run's
/// last: it is not handed out, and the job before it is followed by the
/// finish's state.
#[allow(clippy::too_many_arguments)]
fn pass_thread<W: Into<ExecWindow>>(
    decode: Arc<DecodeTable>,
    start_registers: &RegisterState,
    image: &PagedMem<u8>,
    lean: bool,
    key: u64,
    windows: Receiver<W>,
    number: fn(u32) -> u32,
    jobs: SyncSender<Ticket>,
    permits: SyncSender<()>,
) -> Result<PassEnd, Error> {
    fn stopped<T>(_: T) -> Error {
        Error::Prover(WALKERS_STOPPED.into())
    }
    let mut pass = LeanPass::new(decode, start_registers)?;
    let mut last: Option<(Arc<Vec<Log>>, Arc<MemoryState>, RegisterState)> = None;
    let mut waiting: Option<SyncSender<After>> = None;
    let (mut k, mut first, mut busy, mut entries, mut most) = (0u32, 0usize, 0.0, 0usize, 0usize);
    let (mut sent, mut state_secs, mut boundaries) = (false, 0.0, 0u64);
    for window in windows {
        let ExecWindow { logs, start, .. } = window.into();
        sent = true;
        let start = Arc::new(start.ok_or_else(|| {
            Error::Prover("the executor sent a window without its memory".into())
        })?);
        if last.is_some() {
            return Err(Error::Prover(
                "the executor sent a window after the one that halts".into(),
            ));
        }
        let t = Instant::now();
        let registers = pass.registers();
        // The run's last window: the walk's state at its start is the
        // finish's.
        if logs.last().is_some_and(|log| log.next_pc == 0) {
            let state = Arc::new(pass.walked_state(&start, image, lean));
            boundaries = pass.boundaries(&state, image, key);
            state_secs = t.elapsed().as_secs_f64();
            busy += state_secs;
            if let Some(after) = waiting.take() {
                let _ = after.send(After::Finish {
                    registers: registers.clone(),
                    state: Arc::clone(&state),
                });
            }
            last = Some((logs, state, registers));
            continue;
        }
        pass.begin(k)?;
        pass.pass(&logs, first)?;
        let window_entries = Arc::new(pass.take_entries());
        let passed = Instant::now();
        busy += t.elapsed().as_secs_f64();
        entries += window_entries.len();
        most = most.max(window_entries.len());
        if let Some(after) = waiting.take() {
            // A walker gone (an error elsewhere) does not stop the pass here:
            // the permits or the jobs channel do.
            let _ = after.send(After::Next {
                registers: registers.clone(),
                entries: Arc::clone(&window_entries),
                start: Arc::clone(&start),
            });
        }
        let (after_tx, after_rx) = sync_channel(1);
        permits.send(()).map_err(stopped)?;
        let (cycles, handed) = (logs.len(), window_entries.len());
        jobs.send(Ticket {
            job: WindowJob {
                window: number(k),
                first,
                logs,
                start,
                registers,
                entries: window_entries,
                history: None,
                recheck: false,
            },
            after: after_rx,
        })
        .map_err(stopped)?;
        head_log::window_handed(k as usize, handed, t, passed);
        waiting = Some(after_tx);
        first += cycles;
        k += 1;
    }
    drop(jobs);
    let Some((last, state, registers)) = last else {
        return Err(if sent {
            Error::Prover("the executor stopped before the run halted".into())
        } else {
            Error::Execution("the run executed no cycle".to_string())
        });
    };
    Ok(PassEnd {
        last,
        state,
        registers,
        first,
        windows: k as usize,
        busy,
        state_secs,
        entries,
        most_entries: most,
        boundaries,
    })
}

/// Record mode's thread in place of the pass ([`walk_windows_from_history`]):
/// each window handed out as a job with its history as it arrives, and (with
/// a check) passed on to the check thread ([`check_thread`]); what follows
/// each job, the next window's registers, sent once that window arrives. The
/// window that halts is the run's last: the finish's state is built from the
/// run's end it brings, undoing what the window itself touched.
#[allow(clippy::too_many_arguments)]
fn record_thread<W: Into<ExecWindow>>(
    start_registers: &RegisterState,
    image: &PagedMem<u8>,
    lean: bool,
    windows: Receiver<W>,
    number: fn(u32) -> u32,
    recheck: usize,
    check: Option<SyncSender<CheckWindow>>,
    jobs: SyncSender<Ticket>,
    permits: SyncSender<()>,
) -> Result<PassEnd, Error> {
    fn stopped<T>(_: T) -> Error {
        Error::Prover(WALKERS_STOPPED.into())
    }
    let mut last: Option<(Arc<Vec<Log>>, Arc<MemoryState>, RegisterState)> = None;
    // The previous job's follow-up.
    let mut waiting: Option<SyncSender<After>> = None;
    let (mut k, mut first, mut busy, mut state_secs) = (0u32, 0usize, 0.0, 0.0);
    let no_memory = Arc::new(executor::vm::memory::Memory::default().snapshot());
    let mut sent = false;
    for window in windows {
        let ExecWindow {
            logs,
            start,
            history,
            last: end,
        } = window.into();
        sent = true;
        if last.is_some() {
            return Err(Error::Prover(
                "the executor sent a window after the one that halts".into(),
            ));
        }
        let history: Arc<WindowHistory> = Arc::from(history.ok_or_else(|| {
            Error::Prover("record mode: the executor sent a window without its history".into())
        })?);
        if history.first != first as u64 || history.extras.len() != logs.len() {
            return Err(Error::Prover(format!(
                "record mode: window {k}'s history covers {} cycles from {}, its logs {} from {first}",
                history.extras.len(),
                history.first,
                logs.len()
            )));
        }
        let registers = register_state(history.registers.as_ref().ok_or_else(|| {
            Error::Prover(format!(
                "record mode: window {k}'s history has no registers"
            ))
        })?);
        if k == 0
            && (registers.regs != start_registers.regs
                || registers.index_register != start_registers.index_register
                || registers.pc_register != start_registers.pc_register)
        {
            return Err(Error::Prover(
                "record mode: the executor's register cells at the start are not the walk's".into(),
            ));
        }
        let t = Instant::now();
        let halts = logs.last().is_some_and(|log| log.next_pc == 0);
        let rechecked = recheck > 0 && !halts && (k as usize).is_multiple_of(recheck);
        if let Some(check) = &check {
            let start = match (rechecked, start) {
                (true, Some(start)) => Some(Arc::new(start)),
                (true, None) => {
                    return Err(Error::Prover(format!(
                        "record mode: window {k} is re-walked but came without its start memory"
                    )));
                }
                (false, _) => None,
            };
            // The check thread gone (an error there) does not stop the walk:
            // its join reports it.
            let _ = check.send(CheckWindow {
                window: k,
                first,
                logs: Arc::clone(&logs),
                start,
                history: Arc::clone(&history),
                last: halts,
            });
        }
        if !halts && let Some(after) = waiting.take() {
            let _ = after.send(After::NextHistory {
                registers: registers.clone(),
            });
        }
        if halts {
            let end = end.ok_or_else(|| {
                Error::Prover("record mode: the run's last window came without its end".into())
            })?;
            let state = Arc::new(finish_state(&end, &history, image, lean));
            if let Some(after) = waiting.take() {
                let _ = after.send(After::Finish {
                    registers: registers.clone(),
                    state: Arc::clone(&state),
                });
            }
            state_secs = t.elapsed().as_secs_f64();
            busy += state_secs;
            last = Some((logs, state, registers));
            continue;
        }
        busy += t.elapsed().as_secs_f64();
        let (after_tx, after_rx) = sync_channel(1);
        permits.send(()).map_err(stopped)?;
        let cycles = logs.len();
        jobs.send(Ticket {
            job: WindowJob {
                window: number(k),
                first,
                logs,
                start: Arc::clone(&no_memory),
                registers,
                entries: Arc::new(Vec::new()),
                history: Some(history),
                recheck: rechecked,
            },
            after: after_rx,
        })
        .map_err(stopped)?;
        waiting = Some(after_tx);
        first += cycles;
        k += 1;
    }
    drop(jobs);
    let Some((last, state, registers)) = last else {
        return Err(if sent {
            Error::Prover("the executor stopped before the run halted".into())
        } else {
            Error::Execution("the run executed no cycle".to_string())
        });
    };
    Ok(PassEnd {
        last,
        state,
        registers,
        first,
        windows: k as usize,
        busy,
        state_secs,
        entries: 0,
        most_entries: 0,
        boundaries: 0,
    })
}

/// A window as record mode's check thread takes it.
struct CheckWindow {
    window: u32,
    first: usize,
    logs: Arc<Vec<Log>>,
    /// The executor's memory at the window's start, for a re-walked window.
    start: Option<Arc<MemorySnapshot>>,
    history: Arc<WindowHistory>,
    /// The run's last window (the finish's): only its registers are checked.
    last: bool,
}

/// Record mode's check: a history re-derived independently of the
/// executor's. Today's lean pass runs over every window in order (its own
/// timestamps and register cells, from the logs alone); every window's
/// starting registers must be the history's. A window that comes with its
/// start memory is re-walked as today's walkers walk it — from the pass's
/// entries and that memory, on a seeded window state — after its entries
/// (addresses and timestamps) and their values are compared with the
/// history's first cells; returns each re-walked window's digest
/// ([`WalkedWindow::emitted_digest`]) for the caller to compare with what its
/// walker emitted from the history.
fn check_thread(
    settings: Arc<WalkSettings>,
    start_registers: &RegisterState,
    image: &PagedMem<u8>,
    windows: Receiver<CheckWindow>,
) -> Result<Vec<(u32, u64)>, Error> {
    let mut pass = LeanPass::new(Arc::clone(&settings.decode), start_registers)?;
    let mut walker = WindowWalker::new(settings);
    let mut walked = Vec::new();
    for w in windows {
        let registers = pass.registers();
        let history = w.history.registers.as_ref().map(register_state);
        if !history.is_some_and(|h| {
            h.regs == registers.regs
                && h.index_register == registers.index_register
                && h.pc_register == registers.pc_register
        }) {
            return Err(Error::Prover(format!(
                "window {}: the history's starting registers are not the lean pass's",
                w.window
            )));
        }
        if w.last {
            break;
        }
        pass.begin(w.window)?;
        pass.pass(&w.logs, w.first)?;
        let entries = pass.take_entries();
        let Some(start) = w.start else {
            continue;
        };
        let mut mine: Vec<(u64, u64)> = entries.clone();
        let theirs = w.history.first_touches();
        for &(address, value, _) in &theirs {
            if value != start.load_byte(address) {
                return Err(Error::Prover(format!(
                    "window {}: the history's value before it at {address:#x} is not the \
                     executor's memory's",
                    w.window
                )));
            }
        }
        let mut theirs: Vec<(u64, u64)> = theirs.into_iter().map(|(a, _, t)| (a, t)).collect();
        mine.sort_unstable();
        theirs.sort_unstable();
        if mine != theirs {
            return Err(Error::Prover(format!(
                "window {}: the history's first cells are not the lean pass's entries",
                w.window
            )));
        }
        let job = WindowJob {
            window: w.window,
            first: w.first,
            logs: w.logs,
            start,
            registers,
            entries: Arc::new(entries),
            history: None,
            recheck: false,
        };
        let (window, _) = walker.walk(&job, image)?;
        if walker.registers.regs != pass.registers().regs
            || walker.registers.index_register != pass.registers().index_register
            || walker.registers.pc_register != pass.registers().pc_register
        {
            return Err(Error::Prover(format!(
                "window {}: re-walked, it ends with other registers than the lean pass's",
                w.window
            )));
        }
        walked.push((w.window, window.emitted_digest()));
    }
    Ok(walked)
}

/// The walk's memory state at the start of the run's last window, from the
/// run's end and that window's history: the image's pages and cells (each
/// byte's value, timestamp 0), then every byte stamped before the window with
/// its value and timestamp then. A byte the window touched holds its cell
/// from before the window (its first cell in the history); every other byte
/// its cell at the end. As [`super::lean_pass::LeanPass::walked_state`], the
/// state's pages are those holding such a byte, filled at once.
fn finish_state(
    end: &HistoryFinal,
    last: &WindowHistory,
    image: &PagedMem<u8>,
    lean: bool,
) -> MemoryState {
    use crate::paged_mem::PageMut;
    use rayon::prelude::*;
    const PAGE: usize = 1 << 16;
    // The window's first cells, by 64 KiB page.
    let mut undo: BTreeMap<u64, Vec<(usize, u8, u64)>> = BTreeMap::new();
    for (address, value, ts) in last.first_touches() {
        undo.entry(address & !(PAGE as u64 - 1)).or_default().push((
            (address as usize) & (PAGE - 1),
            value,
            ts,
        ));
    }
    // Each shadow page's cells at the window's start; a page holding none
    // stamped is left out.
    let mut pages: Vec<(u64, Vec<u64>, Vec<u8>)> = end
        .shadow
        .pages()
        .into_par_iter()
        .filter_map(|(base, ts)| {
            let mut ts: Vec<u64> = ts.iter().map(|&t| u64::from(t)).collect();
            let mut values = vec![0u8; PAGE];
            end.memory.read(base, &mut values);
            if let Some(cells) = undo.get(&base) {
                for &(off, value, before) in cells {
                    ts[off] = before;
                    values[off] = value;
                }
            }
            ts.iter().any(|&t| t > 0).then_some((base, ts, values))
        })
        .collect();
    pages.sort_unstable_by_key(|(base, _, _)| *base);
    let mut memory = MemoryState::from_pages(image);
    memory.lean = lean;
    for (base, _, _) in &pages {
        memory
            .cells
            .page_index_or_insert(PagedMem::<(u8, u64)>::page_of(*base).0);
    }
    let fill = |mut state: PageMut<'_, (u8, u64)>| {
        let from = pages.partition_point(|(base, _, _)| *base < state.base());
        let to = from
            + pages[from..]
                .partition_point(|(base, _, _)| base - state.base() < DEFAULT_PAGE_SIZE as u64);
        for (base, ts, values) in &pages[from..to] {
            let at = (base - state.base()) as usize;
            for (off, (&t, &v)) in ts.iter().zip(values.iter()).enumerate() {
                if t > 0 {
                    state.set(at + off, (v, t));
                }
            }
        }
    };
    let state_pages: Vec<_> = memory.cells.pages_mut().collect();
    state_pages.into_par_iter().for_each(fill);
    memory
}

#[cfg(test)]
mod history_tests;
#[cfg(test)]
mod tests;
