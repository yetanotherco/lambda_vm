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

use executor::vm::logs::Log;
use executor::vm::memory::MemorySnapshot;

use super::super::{ABSENT, MemoryState, NO_PAGE, RegisterState, WalkSizes};
use super::lean_pass::LeanPass;
use super::{Accumulator, WalkedWindow, WindowedTraceBuilder, walk_window};
use crate::Error;
use crate::paged_mem::PagedMem;
use crate::tables::decode::DecodeTable;
use crate::tables::page::DEFAULT_PAGE_SIZE;

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
}

/// The walk's settings a window is walked with.
struct WalkSettings {
    decode: Arc<DecodeTable>,
    lookups: bool,
    batch: usize,
    /// The run's key for [`cell_hash`].
    key: u64,
    /// Keep each window's logs with it ([`WindowedTraceBuilder::keeps_logs`]).
    keep_logs: bool,
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
        }
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
        let seeded = self.seed(job, image)?;
        self.registers = job.registers.clone();
        let s = &*self.settings;
        let mut walked = walk_window(
            &s.decode,
            s.lookups,
            s.batch,
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
            After::Finish { registers, state } => self.check_exit_into(job, &registers, &state)?,
        }
        busy += t.elapsed().as_secs_f64();
        Ok((walked, busy, boundary))
    }

    /// The bytes its window state takes on the heap.
    fn state_bytes(&self) -> usize {
        self.memory.heap_bytes()
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
        )
    }
}

/// The run's windows, as the executor sends them (each with its memory at the
/// window's start), walked on `walkers` threads from the lean pass's entries
/// and absorbed strictly in window order by `absorb`, which is also given the
/// bytes the walkers' window states take then. The builder then holds the
/// state the walk holds before the run's last window, which is returned (the
/// finish walks it as usual). The builder must not have walked a window yet.
/// The first error anywhere stops every thread and is returned; a walker that
/// panics is an error.
pub fn walk_windows(
    builder: &mut WindowedTraceBuilder<'_>,
    windows: Receiver<(Vec<Log>, Option<MemorySnapshot>)>,
    walkers: usize,
    absorb: impl FnMut(&mut Accumulator<'_>, WalkedWindow, usize) -> Result<(), Error>,
) -> Result<ParallelWalk, Error> {
    walk_numbered(builder, windows, walkers, |k| k, absorb)
}

/// [`walk_windows`], each job numbered `number(k)` for the run's window `k`:
/// a test numbers them wrong to show the absorbing side refuses it.
fn walk_numbered(
    builder: &mut WindowedTraceBuilder<'_>,
    windows: Receiver<(Vec<Log>, Option<MemorySnapshot>)>,
    walkers: usize,
    number: fn(u32) -> u32,
    mut absorb: impl FnMut(&mut Accumulator<'_>, WalkedWindow, usize) -> Result<(), Error>,
) -> Result<ParallelWalk, Error> {
    assert!(walkers >= 1, "walk_windows needs a walker");
    debug_assert_eq!(builder.cycles, 0, "walk_windows starts the run");
    let settings = Arc::new(WalkSettings {
        decode: Arc::clone(&builder.decode),
        lookups: !builder.lean.lookups,
        batch: builder.walk_batch,
        key: std::collections::hash_map::RandomState::new().hash_one(0u64),
        keep_logs: builder.keeps_logs(),
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
            let pass = scope.spawn(move || {
                pass_thread(
                    decode,
                    &start_registers,
                    image,
                    lean,
                    key,
                    windows,
                    number,
                    job_tx,
                    permit_tx,
                )
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
    })
}

/// The pass's thread: each window passed and handed out as a job (numbered
/// `number(k)`) as it arrives, a permit taken for each; what follows each job
/// sent once the next window is passed. The window that halts is the run's
/// last: it is not handed out, and the job before it is followed by the
/// finish's state.
#[allow(clippy::too_many_arguments)]
fn pass_thread(
    decode: Arc<DecodeTable>,
    start_registers: &RegisterState,
    image: &PagedMem<u8>,
    lean: bool,
    key: u64,
    windows: Receiver<(Vec<Log>, Option<MemorySnapshot>)>,
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
    for (logs, start) in windows {
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
            last = Some((Arc::new(logs), state, registers));
            continue;
        }
        pass.begin(k)?;
        pass.pass(&logs, first)?;
        let window_entries = Arc::new(pass.take_entries());
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
        let cycles = logs.len();
        jobs.send(Ticket {
            job: WindowJob {
                window: number(k),
                first,
                logs: Arc::new(logs),
                start,
                registers,
                entries: window_entries,
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
        entries,
        most_entries: most,
        boundaries,
    })
}

#[cfg(test)]
mod tests;
