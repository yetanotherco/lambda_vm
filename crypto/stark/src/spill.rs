//! Packed main traces spilled to a local drive between their Round-1 commit
//! and phase B (D-ANYBLOCK §6.2).
//!
//! From each instance's Round-1 commit to its fused task, a block holds one
//! thing it could keep elsewhere: the packed main trace ([`NarrowMain`]).
//! Phase B reads each one once (twice when it was spilled before its Round-1
//! commit), in walk orders it knows before it starts. So a [`SpillStore`]
//! writes the packed bytes to one file, a prefetcher reads them back ahead of
//! the walks, and a table's driver waits for its bytes before it takes a VRAM
//! permit, never while holding one.
//!
//! The words that come back are the words that went out, checked against the
//! digest the writer takes before it writes them ([`NarrowMain::digest`]), so
//! no proof byte can depend on the spill. The digest is not cryptographic: the threat is a bug
//! or the disk, not an adversary, and the kept-top check and the verifier
//! stay behind it.
//!
//! - One unnamed file (`O_TMPFILE`, else created and unlinked at once) in
//!   `LAMBDA_VM_BLOCK_SPILL_DIR`, else `$TMPDIR`, else `/var/tmp`. A tmpfs or
//!   ramfs directory is refused: it would be RAM.
//! - `O_DIRECT` when an aligned probe write and read succeed; otherwise
//!   buffered, each slot synced and dropped from the page cache after its I/O
//!   (a cgroup counts the page cache).
//! - Slots are append-only and 4 KiB-aligned. Writer threads drain a queue
//!   bounded in bytes; a full queue blocks the spiller (back-pressure).
//! - A write error (ENOSPC, EIO) fails the store: the bytes it could not write
//!   stay in memory, later spills are declined, and the prove goes on resident.
//!   A read error or a digest mismatch refuses the proof.
//!
//! Nothing spills unless a caller opens a store and hands it traces
//! ([`crate::trace::TraceTable::spill_main`]).

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fs::File;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, Weak};
use std::time::Instant;

use crate::narrow::{Bytes, Digester, NarrowMain};

/// Alignment of every slot, I/O buffer and I/O length under `O_DIRECT`.
const ALIGN: usize = 4096;

/// Bytes moved per bounce-buffer copy.
const BOUNCE: usize = 4 << 20;

/// The probe block heads the file; slots start after it.
const HEAD: u64 = ALIGN as u64;

const GIB: f64 = (1u64 << 30) as f64;

fn round_up(n: usize) -> usize {
    n.div_ceil(ALIGN) * ALIGN
}

/// A poisoned lock here is a panic already being reported elsewhere; the
/// state behind it stays consistent (every transition is one assignment).
fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

fn wait<'a, T>(cv: &Condvar, guard: MutexGuard<'a, T>) -> MutexGuard<'a, T> {
    cv.wait(guard).unwrap_or_else(|e| e.into_inner())
}

/// Whether a store tries `O_DIRECT`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DirectIo {
    /// When the probe write and read succeed (Linux only).
    #[default]
    Auto,
    /// Never: buffered I/O, each slot dropped from the page cache after it.
    Off,
}

/// How a [`SpillStore`] is opened.
#[derive(Clone, Debug)]
pub struct SpillOptions {
    /// The directory of the spill file; `None` is [`default_dir`].
    pub dir: Option<PathBuf>,
    /// Whether to try `O_DIRECT`.
    pub direct: DirectIo,
    /// Bytes queued for the writers before a spill waits for room. A trace
    /// larger than this is queued alone.
    pub queue_bytes: u64,
    /// Writer threads.
    pub writers: usize,
    /// Diagnostic: a written slot keeps its bytes in memory. The writers pay
    /// their whole cost (digest, copies, writes) and nothing is freed; reads
    /// are served from memory.
    pub write_through: bool,
    /// Test only: treat the directory as RAM-backed, so the refusal can be
    /// checked where no tmpfs exists.
    #[cfg(any(test, feature = "test-utils"))]
    pub treat_dir_as_ram: bool,
}

impl Default for SpillOptions {
    fn default() -> Self {
        Self {
            dir: None,
            direct: DirectIo::Auto,
            queue_bytes: 2 << 30,
            writers: 1,
            write_through: false,
            #[cfg(any(test, feature = "test-utils"))]
            treat_dir_as_ram: false,
        }
    }
}

/// The spill directory when the caller names none: `LAMBDA_VM_BLOCK_SPILL_DIR`,
/// else `$TMPDIR`, else `/var/tmp` (`/tmp` is often a tmpfs).
pub fn default_dir() -> PathBuf {
    ["LAMBDA_VM_BLOCK_SPILL_DIR", "TMPDIR"]
        .iter()
        .find_map(|var| std::env::var_os(var).filter(|v| !v.is_empty()))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/var/tmp"))
}

/// Why a spilled trace could not be read back.
#[derive(Debug)]
pub enum SpillError {
    /// The bytes read back are not the ones written: another digest, or
    /// another length.
    Mismatch,
    /// The read failed.
    Io(io::Error),
}

impl std::fmt::Display for SpillError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SpillError::Mismatch => write!(f, "the bytes read back do not match their digest"),
            SpillError::Io(e) => write!(f, "reading the spill file: {e}"),
        }
    }
}

/// What a store has done so far ([`SpillStore::stats`]).
#[derive(Clone, Debug, Default)]
pub struct SpillStats {
    /// Whether the file is written and read with `O_DIRECT`.
    pub direct: bool,
    /// Traces spilled, and their packed bytes.
    pub slots: u64,
    pub bytes: u64,
    /// Slots written to the file, their bytes, and the writers' seconds.
    pub written: u64,
    pub bytes_written: u64,
    pub write_secs: f64,
    /// The writers' seconds by step: digesting the bytes, copying them into
    /// the aligned buffer (`O_DIRECT` only), and the write calls.
    pub digest_secs: f64,
    pub copy_secs: f64,
    pub pwrite_secs: f64,
    /// Slots written straight from their page-aligned bytes, each chunk
    /// digested just before its write, with no aligned copy.
    pub written_from_pages: u64,
    /// Reads from the file, their bytes, and the readers' seconds.
    pub reads: u64,
    pub bytes_read: u64,
    pub read_secs: f64,
    /// Reads served from memory: a slot not yet written, or kept after a
    /// failed write.
    pub memory_reads: u64,
    /// The most bytes queued for the writers at once.
    pub queue_high_water: u64,
    /// Reads refused for a digest mismatch.
    pub mismatches: u64,
    /// Why the store stopped spilling, if it did.
    pub failure: Option<String>,
    /// Whether written slots keep their bytes ([`SpillOptions::write_through`]).
    pub write_through: bool,
}

impl std::fmt::Display for SpillStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} slots {:.2} GiB · written {} {:.2} GiB in {:.2} s (digest {:.2} · copy {:.2} · \
             pwrite {:.2}) · from pages {} · read {} {:.2} GiB in {:.2} s · from memory {} · \
             queue high-water {:.2} GiB · mismatches {} · {}",
            self.slots,
            self.bytes as f64 / GIB,
            self.written,
            self.bytes_written as f64 / GIB,
            self.write_secs,
            self.digest_secs,
            self.copy_secs,
            self.pwrite_secs,
            self.written_from_pages,
            self.reads,
            self.bytes_read as f64 / GIB,
            self.read_secs,
            self.memory_reads,
            self.queue_high_water as f64 / GIB,
            self.mismatches,
            if self.direct { "O_DIRECT" } else { "buffered" },
        )?;
        if self.write_through {
            write!(f, " · write-through")?;
        }
        if let Some(why) = &self.failure {
            write!(f, " · FAILED ({why})")?;
        }
        Ok(())
    }
}

#[derive(Default)]
struct Counters {
    slots: AtomicU64,
    bytes: AtomicU64,
    written: AtomicU64,
    bytes_written: AtomicU64,
    write_ns: AtomicU64,
    digest_ns: AtomicU64,
    copy_ns: AtomicU64,
    pwrite_ns: AtomicU64,
    from_pages: AtomicU64,
    reads: AtomicU64,
    bytes_read: AtomicU64,
    read_ns: AtomicU64,
    memory_reads: AtomicU64,
    mismatches: AtomicU64,
}

impl Counters {
    fn add(counter: &AtomicU64, n: u64) {
        counter.fetch_add(n, Ordering::Relaxed);
    }
}

/// The writers' queue. It holds weak references: a slot whose every holder
/// is gone is never read, so it is never written.
struct Queue {
    jobs: VecDeque<(Weak<Slot>, u64)>,
    bytes: u64,
    high_water: u64,
    in_flight: usize,
    next_offset: u64,
    closed: bool,
}

/// Test-only faults.
#[cfg(any(test, feature = "test-utils"))]
struct Hooks {
    /// Writes after this many succeed fail with ENOSPC (`u64::MAX`: never).
    fail_after: AtomicU64,
    /// Milliseconds each read from the file sleeps first.
    read_delay_ms: AtomicU64,
}

struct Inner {
    file: File,
    direct: bool,
    dir: PathBuf,
    queue_bytes: u64,
    queue: Mutex<Queue>,
    /// A job was queued, or the queue closed.
    work: Condvar,
    /// The queue lost bytes, or the store failed.
    room: Condvar,
    /// The queue is empty and no write is in flight.
    idle: Condvar,
    failed: AtomicBool,
    failure: Mutex<Option<String>>,
    /// Whether reads check their digest (always, outside the tests).
    verify: AtomicBool,
    write_through: bool,
    counters: Counters,
    #[cfg(any(test, feature = "test-utils"))]
    hooks: Hooks,
}

/// Where one spilled trace's bytes are.
enum SlotState {
    /// In memory: queued for a writer, kept after a failed write, or kept
    /// after its write under [`SpillOptions::write_through`].
    Memory(Bytes),
    /// A writer holds the bytes.
    Writing,
    /// In the file.
    Disk,
    /// Moved out of memory by the trace's last reader.
    Taken,
}

pub(crate) struct Slot {
    store: Arc<Inner>,
    offset: u64,
    len: usize,
    /// The packed trace's shape, and its digest, taken by the writer before
    /// it writes the bytes (never by the spiller: phase A's committers).
    rows: usize,
    widths: Vec<u8>,
    digest: std::sync::OnceLock<[u64; 2]>,
    state: Mutex<SlotState>,
    changed: Condvar,
}

impl Inner {
    fn fail(&self, why: String) {
        lock(&self.failure).get_or_insert(why);
        self.failed.store(true, Ordering::Release);
        // Under the queue's lock, so a spiller between its check and its wait
        // cannot miss the wake-up.
        let _queue = lock(&self.queue);
        self.room.notify_all();
    }

    /// Test only: the write that [`SpillStore::fail_writes_after`] fails.
    fn injected_failure(&self) -> io::Result<()> {
        #[cfg(any(test, feature = "test-utils"))]
        if self.counters.written.load(Ordering::Relaxed)
            >= self.hooks.fail_after.load(Ordering::Relaxed)
        {
            return Err(io::Error::from_raw_os_error(libc::ENOSPC));
        }
        Ok(())
    }

    fn write_at(
        &self,
        offset: u64,
        bytes: &[u8],
        bounce: Option<&mut AlignedBuf>,
    ) -> io::Result<()> {
        self.injected_failure()?;
        match bounce {
            // O_DIRECT: aligned copies, the last one zero-padded to a block.
            Some(bounce) => {
                let buf = bounce.as_mut_slice();
                let mut done = 0;
                while done < bytes.len() {
                    let n = (bytes.len() - done).min(buf.len());
                    let t = Instant::now();
                    buf[..n].copy_from_slice(&bytes[done..done + n]);
                    let span = round_up(n);
                    buf[n..span].fill(0);
                    let copied = Instant::now();
                    Counters::add(&self.counters.copy_ns, (copied - t).as_nanos() as u64);
                    let wrote = os::pwrite_all(&self.file, &buf[..span], offset + done as u64);
                    Counters::add(&self.counters.pwrite_ns, copied.elapsed().as_nanos() as u64);
                    wrote?;
                    done += n;
                }
                Ok(())
            }
            None => {
                let t = Instant::now();
                let wrote = os::pwrite_all(&self.file, bytes, offset)
                    .and_then(|()| os::settle(&self.file, offset, bytes.len() as u64, true));
                Counters::add(&self.counters.pwrite_ns, t.elapsed().as_nanos() as u64);
                wrote
            }
        }
    }

    /// Write `padded` (page-aligned bytes and the zeros after them to a
    /// block, [`Bytes::direct`]) as it is, a chunk at a time, each chunk
    /// digested into `digest` just before its write, so the write reads it
    /// from cache: one pass over the bytes from memory instead of the digest's
    /// and an aligned copy's. Only `len` bytes are digested.
    fn write_from_pages(
        &self,
        offset: u64,
        padded: &[u8],
        len: usize,
        digest: &mut Digester,
    ) -> io::Result<()> {
        self.injected_failure()?;
        let mut done = 0;
        while done < padded.len() {
            let end = (done + BOUNCE).min(padded.len());
            let t = Instant::now();
            digest.update(&padded[done.min(len)..end.min(len)]);
            let digested = Instant::now();
            Counters::add(&self.counters.digest_ns, (digested - t).as_nanos() as u64);
            let wrote = os::pwrite_all(&self.file, &padded[done..end], offset + done as u64);
            Counters::add(
                &self.counters.pwrite_ns,
                digested.elapsed().as_nanos() as u64,
            );
            wrote?;
            done = end;
        }
        if self.direct {
            Ok(())
        } else {
            let t = Instant::now();
            let settled = os::settle(&self.file, offset, padded.len() as u64, true);
            Counters::add(&self.counters.pwrite_ns, t.elapsed().as_nanos() as u64);
            settled
        }
    }

    fn read_at(&self, offset: u64, len: usize) -> io::Result<Vec<u8>> {
        #[cfg(any(test, feature = "test-utils"))]
        {
            let ms = self.hooks.read_delay_ms.load(Ordering::Relaxed);
            if ms > 0 {
                std::thread::sleep(std::time::Duration::from_millis(ms));
            }
        }
        let t = Instant::now();
        let mut bounce = AlignedBuf::new(round_up(len.min(BOUNCE)).max(ALIGN));
        let buf = bounce.as_mut_slice();
        let mut out = Vec::with_capacity(len);
        while out.len() < len {
            let n = (len - out.len()).min(buf.len());
            let span = if self.direct { round_up(n) } else { n };
            os::pread_exact(&self.file, &mut buf[..span], offset + out.len() as u64)?;
            out.extend_from_slice(&buf[..n]);
        }
        if !self.direct {
            os::settle(&self.file, offset, len as u64, false)?;
        }
        Counters::add(&self.counters.reads, 1);
        Counters::add(&self.counters.bytes_read, len as u64);
        Counters::add(&self.counters.read_ns, t.elapsed().as_nanos() as u64);
        Ok(out)
    }
}

impl Slot {
    /// The packed trace from `bytes` read back, checked against the digest.
    /// Every slot in the file has one (the writer takes it first); bytes that
    /// never left memory before a writer reached them have none to check.
    fn check(&self, bytes: Bytes) -> Result<NarrowMain, SpillError> {
        let narrow = NarrowMain::from_bytes(self.rows, self.widths.clone(), bytes)
            .ok_or(SpillError::Mismatch)?;
        let store = &*self.store;
        if let Some(&digest) = self.digest.get()
            && store.verify.load(Ordering::Relaxed)
            && narrow.digest() != digest
        {
            Counters::add(&store.counters.mismatches, 1);
            return Err(SpillError::Mismatch);
        }
        Ok(narrow)
    }

    /// A copy of the bytes, from memory or the file; waits out a write in
    /// flight.
    fn read(&self) -> io::Result<Bytes> {
        let mut state = lock(&self.state);
        loop {
            match &*state {
                SlotState::Memory(bytes) => {
                    Counters::add(&self.store.counters.memory_reads, 1);
                    return Ok(bytes.clone());
                }
                SlotState::Writing => state = wait(&self.changed, state),
                SlotState::Disk => break,
                SlotState::Taken => return Err(taken()),
            }
        }
        drop(state);
        self.store.read_at(self.offset, self.len).map(Bytes::Heap)
    }

    /// The bytes for their last reader: moved out of memory when they are
    /// still there and nothing but the caller's `others` holders can read the
    /// slot again; otherwise a copy, as [`Self::read`].
    fn read_last(self: &Arc<Self>, others: usize) -> io::Result<Bytes> {
        let mut state = lock(&self.state);
        loop {
            match &*state {
                SlotState::Memory(_) if Arc::strong_count(self) <= others + 1 => {
                    Counters::add(&self.store.counters.memory_reads, 1);
                    match std::mem::replace(&mut *state, SlotState::Taken) {
                        SlotState::Memory(bytes) => return Ok(bytes),
                        _ => unreachable!("matched as Memory under the lock"),
                    }
                }
                SlotState::Memory(_) | SlotState::Disk | SlotState::Taken => break,
                SlotState::Writing => state = wait(&self.changed, state),
            }
        }
        drop(state);
        self.read()
    }

    fn write(&self, bounce: Option<&mut AlignedBuf>) {
        let store = &*self.store;
        if store.failed.load(Ordering::Acquire) {
            // The bytes stay in memory: the trace stays resident.
            return;
        }
        let bytes = {
            let mut state = lock(&self.state);
            match std::mem::replace(&mut *state, SlotState::Writing) {
                SlotState::Memory(bytes) => bytes,
                other => {
                    *state = other;
                    return;
                }
            }
        };
        let t = Instant::now();
        let result = match bytes.direct() {
            // Page-aligned bytes: written as they are, digested chunk by chunk.
            Some(padded) => {
                let mut digest = Digester::new(self.rows, &self.widths, bytes.len());
                let wrote = store.write_from_pages(self.offset, padded, bytes.len(), &mut digest);
                if wrote.is_ok() {
                    let _ = self.digest.set(digest.finish());
                    Counters::add(&store.counters.from_pages, 1);
                }
                wrote
            }
            // Heap bytes: digested whole, then copied into the aligned buffer
            // (`O_DIRECT`) or written as they are (buffered).
            None => {
                let _ =
                    self.digest
                        .set(crate::narrow::digest_parts(self.rows, &self.widths, &bytes));
                Counters::add(&store.counters.digest_ns, t.elapsed().as_nanos() as u64);
                store.write_at(self.offset, &bytes, bounce)
            }
        };
        let mut state = lock(&self.state);
        match result {
            Ok(()) => {
                let freed = if store.write_through {
                    *state = SlotState::Memory(bytes);
                    None
                } else {
                    *state = SlotState::Disk;
                    Some(bytes)
                };
                drop(state);
                self.changed.notify_all();
                drop(freed);
                Counters::add(&store.counters.written, 1);
                Counters::add(&store.counters.bytes_written, self.len as u64);
                Counters::add(&store.counters.write_ns, t.elapsed().as_nanos() as u64);
            }
            Err(e) => {
                *state = SlotState::Memory(bytes);
                drop(state);
                self.changed.notify_all();
                store.fail(format!(
                    "writing {} bytes at offset {} in {}: {e}",
                    self.len,
                    self.offset,
                    store.dir.display()
                ));
            }
        }
    }

    fn in_memory(&self) -> bool {
        matches!(&*lock(&self.state), SlotState::Memory(_))
    }
}

fn taken() -> io::Error {
    io::Error::other("the slot's bytes were moved out by the trace's last reader")
}

/// A spilled trace's shape and where its bytes are
/// ([`crate::trace::TraceTable::spill_main`]). One pointer, so a trace pays
/// eight bytes for it; clones share the slot.
#[derive(Clone)]
pub struct SpilledMain {
    slot: Arc<Slot>,
}

impl std::fmt::Debug for SpilledMain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SpilledMain")
            .field("rows", &self.slot.rows)
            .field("cols", &self.slot.widths.len())
            .field("bytes", &self.slot.len)
            .field("offset", &self.slot.offset)
            .finish_non_exhaustive()
    }
}

/// Two spilled traces are equal when they are one slot, or two whose packed
/// traces have the same shape, length and digest (taken by their writers).
impl PartialEq for SpilledMain {
    fn eq(&self, other: &Self) -> bool {
        let (a, b) = (&*self.slot, &*other.slot);
        Arc::ptr_eq(&self.slot, &other.slot)
            || (a.rows == b.rows
                && a.widths == b.widths
                && a.len == b.len
                && a.digest.get().is_some()
                && a.digest.get() == b.digest.get())
    }
}

impl Eq for SpilledMain {}

impl SpilledMain {
    /// Rows of the trace.
    pub fn rows(&self) -> usize {
        self.slot.rows
    }

    /// Bytes per cell of each column.
    pub fn widths(&self) -> &[u8] {
        &self.slot.widths
    }

    /// Packed bytes.
    pub fn len(&self) -> usize {
        self.slot.len
    }

    /// Whether the packed trace has no bytes.
    pub fn is_empty(&self) -> bool {
        self.slot.len == 0
    }

    /// Whether the bytes are in memory: not written yet, kept after a failed
    /// write, or kept under [`SpillOptions::write_through`].
    pub fn is_resident(&self) -> bool {
        self.slot.in_memory()
    }

    /// The digest the writer took before writing the bytes
    /// ([`NarrowMain::digest`]); `None` until a writer has reached them.
    pub fn digest(&self) -> Option<[u64; 2]> {
        self.slot.digest.get().copied()
    }

    /// A copy of the packed trace, checked against its digest; the slot
    /// stays readable.
    pub fn load(&self) -> Result<NarrowMain, SpillError> {
        self.slot.check(self.slot.read().map_err(SpillError::Io)?)
    }

    /// The packed trace for its last reader, checked against its digest:
    /// bytes still in memory move out unless a clone of this handle could
    /// read them again.
    pub(crate) fn into_narrow(self) -> Result<NarrowMain, SpillError> {
        self.slot
            .check(self.slot.read_last(0).map_err(SpillError::Io)?)
    }
}

/// One file of spilled traces and its writer threads (see the module docs).
/// Dropping it writes what is queued and joins the writers; the file closes
/// when the last [`SpilledMain`] in it is dropped.
pub struct SpillStore {
    inner: Arc<Inner>,
    writers: Vec<std::thread::JoinHandle<()>>,
}

impl SpillStore {
    /// Open a store: refuses a RAM-backed directory, creates the unnamed file,
    /// probes `O_DIRECT` and starts the writers.
    pub fn open(opts: SpillOptions) -> io::Result<Self> {
        let dir = opts.dir.clone().unwrap_or_else(default_dir);
        let ram = os::is_ram_fs(&dir)?;
        #[cfg(any(test, feature = "test-utils"))]
        let ram = ram || opts.treat_dir_as_ram;
        if ram {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                format!(
                    "{}: a tmpfs or ramfs directory is RAM, not a spill disk",
                    dir.display()
                ),
            ));
        }
        let file = os::open_unnamed(&dir)?;
        let direct = opts.direct == DirectIo::Auto && probe_direct(&file);
        let inner = Arc::new(Inner {
            file,
            direct,
            dir,
            queue_bytes: opts.queue_bytes,
            queue: Mutex::new(Queue {
                jobs: VecDeque::new(),
                bytes: 0,
                high_water: 0,
                in_flight: 0,
                next_offset: HEAD,
                closed: false,
            }),
            work: Condvar::new(),
            room: Condvar::new(),
            idle: Condvar::new(),
            failed: AtomicBool::new(false),
            failure: Mutex::new(None),
            verify: AtomicBool::new(true),
            write_through: opts.write_through,
            counters: Counters::default(),
            #[cfg(any(test, feature = "test-utils"))]
            hooks: Hooks {
                fail_after: AtomicU64::new(u64::MAX),
                read_delay_ms: AtomicU64::new(0),
            },
        });
        let mut store = Self {
            inner,
            writers: Vec::new(),
        };
        for n in 0..opts.writers.max(1) {
            let inner = Arc::clone(&store.inner);
            // On an error the store drops here, which closes the queue and
            // joins the writers already started.
            let writer = std::thread::Builder::new()
                .name(format!("spill-writer-{n}"))
                .spawn(move || write_loop(&inner))?;
            store.writers.push(writer);
        }
        Ok(store)
    }

    /// The directory of the file.
    pub fn dir(&self) -> &Path {
        &self.inner.dir
    }

    /// Whether the file is written and read with `O_DIRECT`.
    pub fn is_direct(&self) -> bool {
        self.inner.direct
    }

    /// Why the store stopped spilling, if a write failed.
    pub fn failure(&self) -> Option<String> {
        lock(&self.inner.failure).clone()
    }

    /// The counters so far.
    pub fn stats(&self) -> SpillStats {
        let c = &self.inner.counters;
        let get = |a: &AtomicU64| a.load(Ordering::Relaxed);
        SpillStats {
            direct: self.inner.direct,
            slots: get(&c.slots),
            bytes: get(&c.bytes),
            written: get(&c.written),
            bytes_written: get(&c.bytes_written),
            write_secs: get(&c.write_ns) as f64 / 1e9,
            digest_secs: get(&c.digest_ns) as f64 / 1e9,
            copy_secs: get(&c.copy_ns) as f64 / 1e9,
            pwrite_secs: get(&c.pwrite_ns) as f64 / 1e9,
            written_from_pages: get(&c.from_pages),
            reads: get(&c.reads),
            bytes_read: get(&c.bytes_read),
            read_secs: get(&c.read_ns) as f64 / 1e9,
            memory_reads: get(&c.memory_reads),
            queue_high_water: lock(&self.inner.queue).high_water,
            mismatches: get(&c.mismatches),
            failure: self.failure(),
            write_through: self.inner.write_through,
        }
    }

    /// Wait until every queued trace is written (or kept, after a failure).
    pub fn flush(&self) {
        let mut queue = lock(&self.inner.queue);
        while !queue.jobs.is_empty() || queue.in_flight > 0 {
            queue = wait(&self.inner.idle, queue);
        }
    }

    /// Queue `narrow`'s bytes for the writers and return where they will be;
    /// `narrow` back, untouched, once the store has failed. Blocks while the
    /// queue is full.
    pub(crate) fn spill(&self, narrow: NarrowMain) -> Result<SpilledMain, NarrowMain> {
        let inner = &self.inner;
        if inner.failed.load(Ordering::Acquire) {
            return Err(narrow);
        }
        let len = narrow.data().len() as u64;
        let mut queue = lock(&inner.queue);
        while queue.bytes > 0
            && queue.bytes + len > inner.queue_bytes
            && !inner.failed.load(Ordering::Acquire)
        {
            queue = wait(&inner.room, queue);
        }
        if inner.failed.load(Ordering::Acquire) {
            return Err(narrow);
        }
        let offset = queue.next_offset;
        queue.next_offset += round_up(len as usize) as u64;
        let (rows, widths, data) = narrow.into_parts();
        let slot = Arc::new(Slot {
            store: Arc::clone(inner),
            offset,
            len: data.len(),
            rows,
            widths,
            digest: std::sync::OnceLock::new(),
            state: Mutex::new(SlotState::Memory(data)),
            changed: Condvar::new(),
        });
        queue.jobs.push_back((Arc::downgrade(&slot), len));
        queue.bytes += len;
        queue.high_water = queue.high_water.max(queue.bytes);
        drop(queue);
        inner.work.notify_one();
        Counters::add(&inner.counters.slots, 1);
        Counters::add(&inner.counters.bytes, len);
        Ok(SpilledMain { slot })
    }

    /// Test only: whether reads check their digest.
    #[cfg(any(test, feature = "test-utils"))]
    pub fn set_verify(&self, on: bool) {
        self.inner.verify.store(on, Ordering::Relaxed);
    }

    /// Test only: every write after the next `n` fails with ENOSPC.
    #[cfg(any(test, feature = "test-utils"))]
    pub fn fail_writes_after(&self, n: u64) {
        let written = self.inner.counters.written.load(Ordering::Relaxed);
        self.inner
            .hooks
            .fail_after
            .store(written.saturating_add(n), Ordering::Relaxed);
    }

    /// Test only: each read from the file first sleeps `ms` milliseconds.
    #[cfg(any(test, feature = "test-utils"))]
    pub fn set_read_delay_ms(&self, ms: u64) {
        self.inner.hooks.read_delay_ms.store(ms, Ordering::Relaxed);
    }

    /// Test only: flip the low bit of byte `at` of `spilled`'s bytes in the
    /// file, after waiting for it to be written. `false` when the slot is not
    /// in this store's file.
    #[cfg(any(test, feature = "test-utils"))]
    pub fn corrupt_on_disk(&self, spilled: &SpilledMain, at: usize) -> io::Result<bool> {
        self.flush();
        let slot = &spilled.slot;
        if !Arc::ptr_eq(&slot.store, &self.inner)
            || !matches!(&*lock(&slot.state), SlotState::Disk)
            || at >= slot.len
        {
            return Ok(false);
        }
        let inner = &self.inner;
        let pos = slot.offset + at as u64;
        if inner.direct {
            let block = pos & !(ALIGN as u64 - 1);
            let mut buf = AlignedBuf::new(ALIGN);
            os::pread_exact(&inner.file, buf.as_mut_slice(), block)?;
            buf.as_mut_slice()[(pos - block) as usize] ^= 1;
            os::pwrite_all(&inner.file, buf.as_slice(), block)?;
        } else {
            let mut byte = [0u8];
            os::pread_exact(&inner.file, &mut byte, pos)?;
            byte[0] ^= 1;
            os::pwrite_all(&inner.file, &byte, pos)?;
            os::settle(&inner.file, pos, 1, true)?;
        }
        Ok(true)
    }
}

impl Drop for SpillStore {
    fn drop(&mut self) {
        lock(&self.inner.queue).closed = true;
        self.inner.work.notify_all();
        for writer in self.writers.drain(..) {
            let _ = writer.join();
        }
    }
}

fn write_loop(inner: &Inner) {
    let mut bounce = inner.direct.then(|| AlignedBuf::new(BOUNCE));
    loop {
        let job = {
            let mut queue = lock(&inner.queue);
            loop {
                if let Some((job, len)) = queue.jobs.pop_front() {
                    queue.bytes -= len;
                    queue.in_flight += 1;
                    break job;
                }
                if queue.closed {
                    return;
                }
                queue = wait(&inner.work, queue);
            }
        };
        inner.room.notify_all();
        if let Some(slot) = job.upgrade() {
            slot.write(bounce.as_mut());
        }
        let mut queue = lock(&inner.queue);
        queue.in_flight -= 1;
        if queue.jobs.is_empty() && queue.in_flight == 0 {
            inner.idle.notify_all();
        }
    }
}

fn probe_direct(file: &File) -> bool {
    if os::set_direct(file, true).is_err() {
        return false;
    }
    let mut out = AlignedBuf::new(ALIGN);
    for (i, b) in out.as_mut_slice().iter_mut().enumerate() {
        *b = (i % 251) as u8;
    }
    let mut back = AlignedBuf::new(ALIGN);
    let ok = os::pwrite_all(file, out.as_slice(), 0).is_ok()
        && os::pread_exact(file, back.as_mut_slice(), 0).is_ok()
        && back.as_slice() == out.as_slice();
    if !ok {
        let _ = os::set_direct(file, false);
    }
    ok
}

/// A zeroed heap buffer at [`ALIGN`], a multiple of it long, for `O_DIRECT`.
struct AlignedBuf {
    ptr: std::ptr::NonNull<u8>,
    len: usize,
}

// SAFETY: the buffer is owned memory with no thread affinity.
unsafe impl Send for AlignedBuf {}

impl AlignedBuf {
    fn layout(len: usize) -> std::alloc::Layout {
        std::alloc::Layout::from_size_align(len, ALIGN).expect("a power-of-two alignment")
    }

    fn new(len: usize) -> Self {
        let len = round_up(len.max(1));
        let layout = Self::layout(len);
        // SAFETY: `layout` has a non-zero size.
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        let ptr =
            std::ptr::NonNull::new(ptr).unwrap_or_else(|| std::alloc::handle_alloc_error(layout));
        Self { ptr, len }
    }

    fn as_slice(&self) -> &[u8] {
        // SAFETY: `ptr` owns `len` initialized (zeroed) bytes.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: as `as_slice`, borrowed mutably once.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

impl Drop for AlignedBuf {
    fn drop(&mut self) {
        // SAFETY: allocated in `new` with this layout.
        unsafe { std::alloc::dealloc(self.ptr.as_ptr(), Self::layout(self.len)) }
    }
}

/// Which read of a table a prefetched entry is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum ReadPhase {
    /// The Round-1 commit of a trace spilled before it (a copy; the trace
    /// stays spilled).
    Round1,
    /// The fused task (the last read: the trace is unspilled).
    Fused,
}

type ReadKey = (ReadPhase, usize);

#[derive(Default)]
struct PrefetchState {
    planned: BTreeSet<ReadKey>,
    /// Read and checked, waiting for their driver.
    passed: BTreeMap<ReadKey, Result<NarrowMain, SpillError>>,
    /// Taken by a driver before the reader got to them: skipped.
    claimed: BTreeSet<ReadKey>,
    reading: Option<ReadKey>,
    /// Bytes read or being read and not yet taken.
    parked: u64,
    high_water: u64,
    stop: bool,
    done: bool,
    reads: u64,
    bytes: u64,
    wait_ns: u64,
}

struct PrefetchShared {
    state: Mutex<PrefetchState>,
    changed: Condvar,
    window: u64,
}

/// The read-back ahead of phase B: one thread reads the planned slots in
/// order, at most `window` bytes ahead of their drivers, checks each against
/// its digest and parks it. Dropping it stops and joins the thread.
pub(crate) struct Prefetch {
    shared: Arc<PrefetchShared>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Prefetch {
    /// Read `reads` back in their order, `window` bytes ahead (one read
    /// larger than the window goes alone).
    pub(crate) fn start(reads: Vec<(ReadPhase, usize, SpilledMain)>, window: u64) -> Self {
        let shared = Arc::new(PrefetchShared {
            state: Mutex::new(PrefetchState {
                planned: reads.iter().map(|(p, i, _)| (*p, *i)).collect(),
                ..PrefetchState::default()
            }),
            changed: Condvar::new(),
            window,
        });
        let reader = Arc::clone(&shared);
        let thread = std::thread::Builder::new()
            .name("spill-prefetch".to_string())
            .spawn(move || {
                let body = std::panic::AssertUnwindSafe(|| read_ahead(&reader, reads));
                let _ = std::panic::catch_unwind(body);
                // Done, or dead: every driver still waiting reads for itself.
                lock(&reader.state).done = true;
                reader.changed.notify_all();
            });
        let thread = match thread {
            Ok(t) => Some(t),
            Err(_) => {
                lock(&shared.state).done = true;
                None
            }
        };
        Self { shared, thread }
    }

    /// Block until `(phase, idx)`'s bytes are parked, or it is not planned.
    pub(crate) fn wait(&self, phase: ReadPhase, idx: usize) {
        let key = (phase, idx);
        let t = Instant::now();
        let mut state = lock(&self.shared.state);
        while !Self::ready_in(&state, key) {
            state = wait(&self.shared.changed, state);
        }
        state.wait_ns += t.elapsed().as_nanos() as u64;
    }

    /// Whether [`Self::wait`] would return at once.
    pub(crate) fn is_ready(&self, phase: ReadPhase, idx: usize) -> bool {
        Self::ready_in(&lock(&self.shared.state), (phase, idx))
    }

    fn ready_in(state: &PrefetchState, key: ReadKey) -> bool {
        !state.planned.contains(&key)
            || state.passed.contains_key(&key)
            || state.claimed.contains(&key)
            || state.stop
            || state.done
    }

    /// `(phase, idx)`'s parked bytes; `None` when the driver reads them
    /// itself (not planned, or not reached yet, which then skips them).
    pub(crate) fn take(
        &self,
        phase: ReadPhase,
        idx: usize,
    ) -> Option<Result<NarrowMain, SpillError>> {
        let key = (phase, idx);
        let mut state = lock(&self.shared.state);
        loop {
            if let Some(read) = state.passed.remove(&key) {
                if let Ok(narrow) = &read {
                    state.parked -= narrow.data().len() as u64;
                    self.shared.changed.notify_all();
                }
                return Some(read);
            }
            if state.reading == Some(key) {
                state = wait(&self.shared.changed, state);
                continue;
            }
            if state.planned.contains(&key) {
                state.claimed.insert(key);
                self.shared.changed.notify_all();
            }
            return None;
        }
    }

    /// Stop: every read not handed over yet is its driver's to make, and
    /// every wait returns. For a prove whose tasks stopped running (a panic):
    /// the reads parked for them would never be taken.
    pub(crate) fn close(&self) {
        lock(&self.shared.state).stop = true;
        self.shared.changed.notify_all();
    }

    /// One line for the prove's log.
    pub(crate) fn report(&self) -> String {
        let state = lock(&self.shared.state);
        format!(
            "{} planned · read ahead {} ({:.2} GiB) · window {:.2} GiB, high-water {:.2} GiB \
             · drivers waited {:.2} s",
            state.planned.len(),
            state.reads,
            state.bytes as f64 / GIB,
            self.shared.window as f64 / GIB,
            state.high_water as f64 / GIB,
            state.wait_ns as f64 / 1e9,
        )
    }
}

impl Drop for Prefetch {
    fn drop(&mut self) {
        lock(&self.shared.state).stop = true;
        self.shared.changed.notify_all();
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

fn read_ahead(shared: &PrefetchShared, reads: Vec<(ReadPhase, usize, SpilledMain)>) {
    // Reads still planned of the same slot.
    fn later(slot: &Arc<Slot>, pending: &VecDeque<(ReadPhase, usize, SpilledMain)>) -> usize {
        pending
            .iter()
            .filter(|(_, _, s)| Arc::ptr_eq(&s.slot, slot))
            .count()
    }
    let mut pending: VecDeque<_> = reads.into();
    while let Some((phase, idx, spilled)) = pending.pop_front() {
        let key = (phase, idx);
        let len = spilled.len() as u64;
        {
            let mut state = lock(&shared.state);
            loop {
                if state.stop || state.claimed.contains(&key) {
                    break;
                }
                if state.parked == 0 || state.parked + len <= shared.window {
                    break;
                }
                state = wait(&shared.changed, state);
            }
            if state.stop {
                return;
            }
            if state.claimed.contains(&key) {
                continue;
            }
            state.reading = Some(key);
            state.parked += len;
            state.high_water = state.high_water.max(state.parked);
        }
        // The fused read is the trace's last: bytes still in memory move out
        // when only the trace and this plan hold the slot.
        let from_disk = !spilled.slot.in_memory();
        let read = match phase {
            ReadPhase::Round1 => spilled.load(),
            ReadPhase::Fused => {
                let others = 1 + later(&spilled.slot, &pending);
                let slot = spilled.slot;
                slot.read_last(others)
                    .map_err(SpillError::Io)
                    .and_then(|bytes| slot.check(bytes))
            }
        };
        let mut state = lock(&shared.state);
        state.reading = None;
        match &read {
            Ok(_) if from_disk => {
                state.reads += 1;
                state.bytes += len;
            }
            Ok(_) => {}
            Err(_) => state.parked -= len,
        }
        state.passed.insert(key, read);
        drop(state);
        shared.changed.notify_all();
    }
}

#[cfg(target_os = "linux")]
mod os {
    use std::ffi::CString;
    use std::fs::{File, OpenOptions};
    use std::io;
    use std::os::fd::AsRawFd;
    use std::os::unix::ffi::OsStrExt;
    use std::os::unix::fs::{FileExt, OpenOptionsExt};
    use std::path::Path;

    const TMPFS_MAGIC: i64 = 0x0102_1994;
    const RAMFS_MAGIC: i64 = 0x8584_58f6;

    pub(super) fn is_ram_fs(dir: &Path) -> io::Result<bool> {
        let path = CString::new(dir.as_os_str().as_bytes())?;
        // SAFETY: `statfs` writes the struct it is given; `path` is a valid
        // C string.
        let mut fs: libc::statfs = unsafe { std::mem::zeroed() };
        if unsafe { libc::statfs(path.as_ptr(), &mut fs) } != 0 {
            return Err(io::Error::last_os_error());
        }
        #[allow(clippy::unnecessary_cast)]
        let kind = fs.f_type as i64;
        Ok(kind == TMPFS_MAGIC || kind == RAMFS_MAGIC)
    }

    pub(super) fn open_unnamed(dir: &Path) -> io::Result<File> {
        OpenOptions::new()
            .read(true)
            .write(true)
            .mode(0o600)
            .custom_flags(libc::O_TMPFILE)
            .open(dir)
            .or_else(|_| super::named_then_unlinked(dir))
    }

    pub(super) fn set_direct(file: &File, on: bool) -> io::Result<()> {
        let fd = file.as_raw_fd();
        // SAFETY: plain fcntl calls on an open descriptor.
        let flags = unsafe { libc::fcntl(fd, libc::F_GETFL) };
        if flags < 0 {
            return Err(io::Error::last_os_error());
        }
        let flags = if on {
            flags | libc::O_DIRECT
        } else {
            flags & !libc::O_DIRECT
        };
        if unsafe { libc::fcntl(fd, libc::F_SETFL, flags) } < 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(())
    }

    /// Buffered I/O only: a written range is synced, then the range leaves
    /// the page cache.
    pub(super) fn settle(file: &File, offset: u64, len: u64, written: bool) -> io::Result<()> {
        if len == 0 {
            return Ok(());
        }
        let fd = file.as_raw_fd();
        if written {
            let how = libc::SYNC_FILE_RANGE_WAIT_BEFORE
                | libc::SYNC_FILE_RANGE_WRITE
                | libc::SYNC_FILE_RANGE_WAIT_AFTER;
            // SAFETY: plain syscall on an open descriptor.
            if unsafe { libc::sync_file_range(fd, offset as i64, len as i64, how) } != 0 {
                return Err(io::Error::last_os_error());
            }
        }
        // Advisory: a failure leaves pages cached, nothing else.
        // SAFETY: plain syscall on an open descriptor.
        unsafe {
            libc::posix_fadvise(fd, offset as i64, len as i64, libc::POSIX_FADV_DONTNEED);
        }
        Ok(())
    }

    pub(super) fn pwrite_all(file: &File, buf: &[u8], offset: u64) -> io::Result<()> {
        file.write_all_at(buf, offset)
    }

    pub(super) fn pread_exact(file: &File, buf: &mut [u8], offset: u64) -> io::Result<()> {
        file.read_exact_at(buf, offset)
    }
}

#[cfg(all(unix, not(target_os = "linux")))]
mod os {
    use std::fs::File;
    use std::io;
    use std::os::unix::fs::FileExt;
    use std::path::Path;

    #[cfg(target_os = "macos")]
    pub(super) fn is_ram_fs(dir: &Path) -> io::Result<bool> {
        use std::os::unix::ffi::OsStrExt;
        let path = std::ffi::CString::new(dir.as_os_str().as_bytes())?;
        // SAFETY: `statfs` writes the struct it is given; `path` is a valid
        // C string.
        let mut fs: libc::statfs = unsafe { std::mem::zeroed() };
        if unsafe { libc::statfs(path.as_ptr(), &mut fs) } != 0 {
            return Err(io::Error::last_os_error());
        }
        // SAFETY: the kernel NUL-terminates the type name.
        let name = unsafe { std::ffi::CStr::from_ptr(fs.f_fstypename.as_ptr()) };
        Ok(name.to_bytes() == b"tmpfs")
    }

    #[cfg(not(target_os = "macos"))]
    pub(super) fn is_ram_fs(dir: &Path) -> io::Result<bool> {
        std::fs::metadata(dir).map(|_| false)
    }

    pub(super) fn open_unnamed(dir: &Path) -> io::Result<File> {
        super::named_then_unlinked(dir)
    }

    pub(super) fn set_direct(_: &File, _: bool) -> io::Result<()> {
        Err(io::Error::from(io::ErrorKind::Unsupported))
    }

    pub(super) fn settle(_: &File, _: u64, _: u64, _: bool) -> io::Result<()> {
        Ok(())
    }

    pub(super) fn pwrite_all(file: &File, buf: &[u8], offset: u64) -> io::Result<()> {
        file.write_all_at(buf, offset)
    }

    pub(super) fn pread_exact(file: &File, buf: &mut [u8], offset: u64) -> io::Result<()> {
        file.read_exact_at(buf, offset)
    }
}

#[cfg(not(unix))]
mod os {
    use std::fs::File;
    use std::io;
    use std::path::Path;

    fn unsupported<T>() -> io::Result<T> {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "spilling traces needs a unix host",
        ))
    }

    pub(super) fn is_ram_fs(_: &Path) -> io::Result<bool> {
        unsupported()
    }

    pub(super) fn open_unnamed(_: &Path) -> io::Result<File> {
        unsupported()
    }

    pub(super) fn set_direct(_: &File, _: bool) -> io::Result<()> {
        unsupported()
    }

    pub(super) fn settle(_: &File, _: u64, _: u64, _: bool) -> io::Result<()> {
        unsupported()
    }

    pub(super) fn pwrite_all(_: &File, _: &[u8], _: u64) -> io::Result<()> {
        unsupported()
    }

    pub(super) fn pread_exact(_: &File, _: &mut [u8], _: u64) -> io::Result<()> {
        unsupported()
    }
}

/// A file created under a fresh name and unlinked at once: unnamed from then
/// on, gone when closed.
#[cfg(unix)]
fn named_then_unlinked(dir: &Path) -> io::Result<File> {
    use std::os::unix::fs::OpenOptionsExt;
    static N: AtomicU64 = AtomicU64::new(0);
    let path = dir.join(format!(
        ".lambda-vm-spill-{}-{}",
        std::process::id(),
        N.fetch_add(1, Ordering::Relaxed)
    ));
    let file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(&path)?;
    std::fs::remove_file(&path)?;
    Ok(file)
}

/// Test-only overrides of the next proves on the calling thread, and the
/// order their admitted tasks start in.
#[cfg(any(test, feature = "test-utils"))]
pub mod test_hooks {
    use std::cell::RefCell;
    use std::sync::{Arc, Mutex};

    /// What [`with_prove_overrides`] changes.
    #[derive(Clone, Copy, Debug, Default)]
    pub struct ProveOverrides {
        /// The VRAM gate's budget, in bytes.
        pub vram_budget: Option<u64>,
        /// The spill read-back's window, in bytes.
        pub window: Option<u64>,
        /// Driver threads per admitted phase.
        pub drivers: Option<usize>,
    }

    /// Per admitted phase: its walk order, and the order its tasks started in.
    #[derive(Clone, Debug, Default, PartialEq, Eq)]
    pub struct Admissions {
        pub r1_walk: Vec<usize>,
        pub r1_started: Vec<usize>,
        pub fused_walk: Vec<usize>,
        pub fused_started: Vec<usize>,
    }

    type Current = Option<(ProveOverrides, Arc<Mutex<Admissions>>)>;

    thread_local! {
        static CURRENT: RefCell<Current> = const { RefCell::new(None) };
    }

    struct Reset;

    impl Drop for Reset {
        fn drop(&mut self) {
            CURRENT.with(|c| *c.borrow_mut() = None);
        }
    }

    /// Run `f` (proves on this thread) under `overrides`; what it returns,
    /// and the admissions its proves logged.
    pub fn with_prove_overrides<R>(
        overrides: ProveOverrides,
        f: impl FnOnce() -> R,
    ) -> (R, Admissions) {
        let log = Arc::new(Mutex::new(Admissions::default()));
        CURRENT.with(|c| *c.borrow_mut() = Some((overrides, Arc::clone(&log))));
        let reset = Reset;
        let out = f();
        drop(reset);
        let admissions = log.lock().unwrap_or_else(|e| e.into_inner()).clone();
        (out, admissions)
    }

    pub(crate) fn current() -> Current {
        CURRENT.with(|c| c.borrow().clone())
    }
}
