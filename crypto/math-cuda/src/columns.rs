//! The epoch's trace columns, on the card once.
//!
//! Four things read the same columns — the commitment, the sumcheck's factors,
//! the evaluation at the reduction point, and the opening's message — and each
//! used to upload its own copy. That is the same forty gigabytes crossing the
//! bus four times, and at the speed pageable host memory gives it is a fifth of
//! the prove. They are put here once and read where they lie.
//!
//! A table's columns are a contiguous run of equal height, which is the layout
//! the factor gather and the batched evaluation already want; the commitment and
//! the opening scatter theirs, which on the card is a copy at device bandwidth.

use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream, DevicePtr, DevicePtrMut};

use crate::Result;
use crate::device::{DeviceReservation, alloc_or_trim, backend};

/// `LAMBDA_VM_CARD_UPLOAD_TIMES=1`: each packed table's copy and widen, and
/// each records table's copy and fill, in a group's upload is timed by events
/// on its stream and summed ([`upload_kernel_times`]). Read once; unset, no
/// event is made. A measurement; the store is the same.
fn timed_uploads() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("LAMBDA_VM_CARD_UPLOAD_TIMES").is_ok_and(|v| v.trim() == "1"))
}

/// The tables timed and their summed nanoseconds: packed (copy and widen),
/// records (copy and fill).
static WIDEN_TABLES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static WIDEN_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static FILL_TABLES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static FILL_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Under `LAMBDA_VM_CARD_UPLOAD_TIMES=1`, the tables whose upload was timed
/// since the process started and their card seconds:
/// `((packed tables, copy and widen s), (records tables, copy and fill s))`.
/// Zeros otherwise.
pub fn upload_kernel_times() -> ((u64, f64), (u64, f64)) {
    use std::sync::atomic::Ordering::Relaxed;
    let s = |ns: &std::sync::atomic::AtomicU64| ns.load(Relaxed) as f64 * 1e-9;
    (
        (WIDEN_TABLES.load(Relaxed), s(&WIDEN_NS)),
        (FILL_TABLES.load(Relaxed), s(&FILL_NS)),
    )
}

/// One table's upload timed on its stream: the events either side of it.
pub(crate) struct TableClock {
    records: bool,
    start: cudarc::driver::CudaEvent,
    pub(crate) done: cudarc::driver::CudaEvent,
}

impl TableClock {
    /// Events on `stream`, the first recorded now; `None` when uploads are not
    /// timed or an event cannot be made.
    pub(crate) fn start(stream: &Arc<CudaStream>, records: bool) -> Option<Self> {
        if !timed_uploads() {
            return None;
        }
        let flags = Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT);
        let ctx = stream.context();
        let clock = Self {
            records,
            start: ctx.new_event(flags).ok()?,
            done: ctx.new_event(flags).ok()?,
        };
        clock.start.record(stream).ok()?;
        Some(clock)
    }

    /// Adds the span to the totals, once the stream ran it.
    pub(crate) fn add(self) {
        use std::sync::atomic::Ordering::Relaxed;
        let Ok(ms) = self.start.elapsed_ms(&self.done) else {
            return;
        };
        let ns = (f64::from(ms) * 1e6) as u64;
        let (tables, total) = if self.records {
            (&FILL_TABLES, &FILL_NS)
        } else {
            (&WIDEN_TABLES, &WIDEN_NS)
        };
        tables.fetch_add(1, Relaxed);
        total.fetch_add(ns, Relaxed);
    }
}

/// Where a set of columns is: still here, or already there.
pub enum Columns<'a> {
    Host(&'a [&'a [u64]]),
    /// A run of `width` columns of `rows` each, starting at column `first`.
    Device {
        store: &'a DeviceColumns,
        first: usize,
        width: usize,
    },
}

impl Columns<'_> {
    pub fn width(&self) -> usize {
        match self {
            Self::Host(columns) => columns.len(),
            Self::Device { width, .. } => *width,
        }
    }

    pub fn rows(&self) -> usize {
        match self {
            Self::Host(columns) => columns.first().map_or(0, |c| c.len()),
            Self::Device { store, first, .. } => store.spans[*first].1,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.width() == 0
    }
}

/// A table's columns as they go up: as they are, packed (each column at the
/// bytes its words need, [`crate::narrow`]) and widened on the card, or as the
/// records its rows come from and filled on the card ([`crate::fill`]).
#[derive(Clone, Copy)]
pub enum TableUpload<'a> {
    /// The columns, every one the table's height.
    Wide(&'a [&'a [u64]]),
    Narrow(crate::narrow::NarrowInput<'a>),
    Records(crate::fill::RecordsInput<'a>),
}

impl TableUpload<'_> {
    /// Words the table takes on the card.
    pub fn cells(&self) -> usize {
        match self {
            Self::Wide(columns) => columns.iter().map(|c| c.len()).sum(),
            Self::Narrow(packed) => packed.rows() * packed.cols(),
            Self::Records(records) => records.rows() * records.cols(),
        }
    }
}

/// The columns themselves, laid end to end in one allocation.
pub struct DeviceColumns {
    stream: Arc<CudaStream>,
    /// Shared so a reader that outlives a borrow of the store — a WHIR opening's
    /// first rounds read the message where it lies — keeps it alive.
    buffer: Arc<CudaSlice<u64>>,
    /// `(offset, len)` in elements, per column, in the order uploaded.
    spans: Vec<(usize, usize)>,
    _room: DeviceReservation,
}

std::thread_local! {
    /// The copy lanes this thread's wide uploads take ([`with_copy_lanes`]).
    static COPY_LANES: std::cell::Cell<usize> = const { std::cell::Cell::new(1) };
}

/// Runs `f` with this thread's table uploads copying their wide columns on
/// `lanes` streams at once (`1`, the default: one stream, one copy after
/// another). Each lane is a thread of its own, since a copy from pageable
/// memory returns only once the driver has staged it. The same words at the
/// same places.
pub fn with_copy_lanes<R>(lanes: usize, f: impl FnOnce() -> R) -> R {
    let before = COPY_LANES.with(|c| c.replace(lanes.max(1)));
    let out = f();
    COPY_LANES.with(|c| c.set(before));
    out
}

/// This thread's copy lanes ([`with_copy_lanes`]).
pub(crate) fn copy_lanes() -> usize {
    COPY_LANES.with(std::cell::Cell::get)
}

/// Wide columns' copies, `(word offset in the buffer, the column)`, deferred
/// by an upload under more than one copy lane.
pub(crate) type WideCopies<'a> = Vec<(usize, &'a [u64])>;

/// Queues `copies` into `buffer` on `lanes` streams at once: the first lane on
/// `stream` from this thread, every other on a stream and a thread of its
/// own that waits for what `stream` had queued before (the buffer's
/// allocation), and `stream` then waits for every lane, so what it runs next
/// sees every word. The columns split by their words, the largest first.
/// `None` when a copy fails.
pub(crate) fn copy_wide_lanes(
    stream: &Arc<CudaStream>,
    buffer: &mut CudaSlice<u64>,
    copies: WideCopies<'_>,
    lanes: usize,
) -> Option<()> {
    if copies.is_empty() {
        return Some(());
    }
    let lanes = lanes.clamp(1, copies.len());
    let mut copies = copies;
    copies.sort_by_key(|&(_, column)| std::cmp::Reverse(column.len()));
    let mut parts: Vec<WideCopies<'_>> = (0..lanes).map(|_| Vec::new()).collect();
    let mut load = vec![0usize; lanes];
    for copy in copies {
        let k = (0..lanes).min_by_key(|&k| load[k]).unwrap_or(0);
        load[k] += copy.1.len();
        parts[k].push(copy);
    }
    let be = backend().ok()?;
    let ready = stream.record_event(None).ok()?;
    let (base, _record) = buffer.device_ptr_mut(stream);
    let copy = |on: &Arc<CudaStream>, part: &WideCopies<'_>| -> Option<()> {
        for &(at, column) in part {
            crate::pinned::note_upload(column.as_ptr().cast(), column.len() * 8);
            // SAFETY: `at .. at + column.len()` lies inside the buffer (the
            // caller's spans), and the buffer outlives the copy: the caller
            // synchronizes `stream`, which waits for this lane, before it
            // returns or drops the buffer.
            unsafe {
                cudarc::driver::result::memcpy_htod_async(
                    base + (at * 8) as u64,
                    column,
                    on.cu_stream(),
                )
            }
            .ok()?;
        }
        Some(())
    };
    let done: Option<Vec<cudarc::driver::CudaEvent>> = std::thread::scope(|scope| {
        let others: Vec<_> = parts[1..]
            .iter()
            .map(|part| {
                let ready = &ready;
                let copy = &copy;
                scope.spawn(move || -> Option<cudarc::driver::CudaEvent> {
                    be.ctx.bind_to_thread().ok()?;
                    let lane = be.next_stream();
                    lane.wait(ready).ok()?;
                    copy(&lane, part)?;
                    lane.record_event(None).ok()
                })
            })
            .collect();
        let first = copy(stream, &parts[0]);
        let events: Option<Vec<_>> = others
            .into_iter()
            .map(|h| h.join().ok().flatten())
            .collect();
        first.and(events)
    });
    for event in done? {
        stream.wait(&event).ok()?;
    }
    Some(())
}

impl DeviceColumns {
    /// `None` when the card will not promise the room, in which case every
    /// caller uploads its own copy as before.
    pub fn upload(columns: &[&[u64]]) -> Option<Self> {
        Self::upload_on(columns, true, None)
    }

    /// [`Self::upload`] for a store off the argue surface (a retired prepared
    /// stack's columns, for its revive and its opening): a refused room moves
    /// no work to the host (the reader copies twice on the card instead), so
    /// it is not a device fallback and is not counted as one, nor as argue
    /// bytes.
    pub fn upload_store(columns: &[&[u64]]) -> Option<Self> {
        Self::upload_on(columns, false, None)
    }

    /// [`Self::upload_store`] into a promise taken before (`room`, at least
    /// the columns' bytes): no reservation is asked for here, so the ledger
    /// cannot refuse it. What the promise holds past the columns' bytes is
    /// given back. `None` (the promise given back with it) when the room is
    /// short or the card fails the copy.
    pub fn upload_store_in(columns: &[&[u64]], room: DeviceReservation) -> Option<Self> {
        Self::upload_on(columns, false, Some(room))
    }

    fn upload_on(
        columns: &[&[u64]],
        argue: bool,
        prepaid: Option<DeviceReservation>,
    ) -> Option<Self> {
        if columns.is_empty() {
            return None;
        }
        let total: usize = columns.iter().map(|c| c.len()).sum();
        let be = backend().ok()?;
        let words = total as u64 * 8;
        let room = match prepaid {
            Some(room) if room.bytes() >= words => {
                room.shrink(room.bytes() - words);
                room
            }
            Some(_) => return None,
            None => {
                let Some(room) = be.reserve(words) else {
                    if argue {
                        crate::device::note_device_fallback();
                    }
                    return None;
                };
                room
            }
        };
        if argue {
            crate::argue_probe::note_device(crate::argue_probe::Surface::Columns, total as u64 * 8);
        }
        let stream = be.next_stream();
        // SAFETY: every element is written by the copies below.
        let mut buffer = unsafe { alloc_or_trim::<u64>(&stream, total) }.ok()?;
        let mut spans = Vec::with_capacity(columns.len());
        let mut at = 0usize;
        for column in columns {
            let mut slab = buffer.slice_mut(at..at + column.len());
            stream.memcpy_htod(*column, &mut slab).ok()?;
            spans.push((at, column.len()));
            at += column.len();
        }
        stream.synchronize().ok()?;
        Some(Self {
            stream,
            buffer: Arc::new(buffer),
            spans,
            _room: room,
        })
    }

    /// A new store of `columns` of `src` (copied on the card, in that order)
    /// followed by `extra` (copied from the host): a small store for a reader
    /// that needs a few of a store's columns beside columns of its own. `None`
    /// as for [`Self::upload`].
    pub fn gather(src: &DeviceColumns, columns: &[usize], extra: &[&[u64]]) -> Option<Self> {
        if columns.is_empty() && extra.is_empty() {
            return None;
        }
        if columns.iter().any(|&k| k >= src.spans.len()) {
            return None;
        }
        let total: usize = columns.iter().map(|&k| src.spans[k].1).sum::<usize>()
            + extra.iter().map(|c| c.len()).sum::<usize>();
        let be = backend().ok()?;
        // Not a device fallback: a refusal here takes the reader's old path,
        // which counts its own refusal where its work moves to the host
        // ([`crate::device::note_device_fallback`]'s site list).
        let room = be.reserve(total as u64 * 8)?;
        let stream = be.next_stream();
        // SAFETY: every element is written by the copies below.
        let mut buffer = unsafe { alloc_or_trim::<u64>(&stream, total) }.ok()?;
        let mut spans = Vec::with_capacity(columns.len() + extra.len());
        let mut at = 0usize;
        // The source's own uploads ended before it was shared (it is
        // read-only after), so copies issued on this stream race with nothing.
        for &k in columns {
            let len = src.spans[k].1;
            src.copy_into(k, &mut buffer, at, &stream).ok()?;
            spans.push((at, len));
            at += len;
        }
        for column in extra {
            let mut slab = buffer.slice_mut(at..at + column.len());
            stream.memcpy_htod(*column, &mut slab).ok()?;
            spans.push((at, column.len()));
            at += column.len();
        }
        stream.synchronize().ok()?;
        Some(Self {
            stream,
            buffer: Arc::new(buffer),
            spans,
            _room: room,
        })
    }

    /// [`Self::upload`] of whole tables, each as it is held: a packed one
    /// crosses the bus at its packed bytes and is widened on the card, column
    /// by column, into the run its columns take — the same words, at the same
    /// offsets, as its widened columns uploaded; one held as records crosses it
    /// at their bytes and is filled there, into the same run. `None` as for
    /// [`Self::upload`], or when a packed table is too big for one launch.
    pub fn upload_tables(tables: &[TableUpload<'_>]) -> Option<Self> {
        // `LAMBDA_VM_CARD_ROWS`: `off` is the upload below, untouched; unset
        // is `words` (`crate::card_rows`).
        if crate::card_rows::mode() != crate::card_rows::Mode::Off {
            return crate::card_rows::upload_tables(tables);
        }
        Self::upload_tables_today(tables)
    }

    /// The store of what [`crate::card_rows`]'s upload wrote.
    pub(crate) fn from_parts(
        stream: Arc<CudaStream>,
        buffer: CudaSlice<u64>,
        spans: Vec<(usize, usize)>,
        room: DeviceReservation,
    ) -> Self {
        Self {
            stream,
            buffer: Arc::new(buffer),
            spans,
            _room: room,
        }
    }

    /// [`Self::upload_tables`] as it is with `LAMBDA_VM_CARD_ROWS` off.
    pub(crate) fn upload_tables_today(tables: &[TableUpload<'_>]) -> Option<Self> {
        Self::upload_tables_on(tables, true, None, false)
    }

    /// [`Self::upload_tables`] for a store off the argue surface, as
    /// [`Self::upload_store`] is for [`Self::upload`]: a retired prepared
    /// stack's columns, packed or wide, for its revive and its opening. A
    /// refused room moves no work to the host, so it is not counted as a
    /// device fallback, nor its bytes as argue bytes. Records are not a
    /// store's (`None`).
    pub fn upload_store_tables(tables: &[TableUpload<'_>]) -> Option<Self> {
        if tables
            .iter()
            .any(|table| matches!(table, TableUpload::Records(_)))
        {
            return None;
        }
        Self::upload_tables_on(tables, false, None, false)
    }

    /// [`Self::upload_store_tables`] with each packed table's bytes copied
    /// through the backend's double-buffered pinned staging
    /// ([`crate::device::htod_staged`]) instead of one pageable copy a table.
    /// The same store.
    pub fn upload_store_tables_staged(tables: &[TableUpload<'_>]) -> Option<Self> {
        if tables
            .iter()
            .any(|table| matches!(table, TableUpload::Records(_)))
        {
            return None;
        }
        Self::upload_tables_on(tables, false, None, true)
    }

    /// [`Self::upload_store_tables`] into a promise taken before (`room`, at
    /// least the tables' words plus the largest packed table's bytes, which
    /// sit on the card while it widens): nothing is reserved here, and the
    /// staging's part of the promise is given back once the tables are up.
    /// `None` (the promise given back with it) when the room is short or the
    /// card fails the upload.
    pub fn upload_store_tables_in(
        tables: &[TableUpload<'_>],
        room: DeviceReservation,
    ) -> Option<Self> {
        if tables
            .iter()
            .any(|table| matches!(table, TableUpload::Records(_)))
        {
            return None;
        }
        Self::upload_tables_on(tables, false, Some(room), false)
    }

    fn upload_tables_on(
        tables: &[TableUpload<'_>],
        argue: bool,
        prepaid: Option<DeviceReservation>,
        staged: bool,
    ) -> Option<Self> {
        let total: usize = tables.iter().map(TableUpload::cells).sum();
        if total == 0
            || tables.iter().any(|table| {
                matches!(table, TableUpload::Narrow(_)) && u32::try_from(table.cells()).is_err()
            })
        {
            return None;
        }
        let be = backend().ok()?;
        let words = total as u64 * 8;
        // The packed bytes (or records) sit on the card only while their
        // table widens (or fills), one table at a time in stream order:
        // promised for the largest of them.
        let staging = tables
            .iter()
            .map(|table| match table {
                TableUpload::Wide(_) => 0,
                TableUpload::Narrow(packed) => packed.bytes() as u64,
                TableUpload::Records(records) => records.bytes() as u64,
            })
            .max()
            .unwrap_or(0);
        // A promise taken before covers the words and the staging; the
        // staging's part goes back once the tables are up.
        let (room, prepaid) = match prepaid {
            Some(room) if room.bytes() >= words.saturating_add(staging) => (room, true),
            Some(_) => return None,
            None => {
                let Some(room) = be.reserve(words) else {
                    if argue {
                        crate::device::note_device_fallback();
                    }
                    return None;
                };
                (room, false)
            }
        };
        let _staging = match staging {
            _ if prepaid => None,
            0 => None,
            bytes => {
                let Some(promise) = be.reserve(bytes) else {
                    if argue {
                        crate::device::note_device_fallback();
                    }
                    return None;
                };
                Some(promise)
            }
        };
        if argue {
            crate::argue_probe::note_device(crate::argue_probe::Surface::Columns, total as u64 * 8);
        }
        let stream = be.next_stream();
        // SAFETY: every element is written by the copies and widens below.
        let mut buffer = unsafe { alloc_or_trim::<u64>(&stream, total) }.ok()?;
        let mut spans = Vec::new();
        let mut at = 0usize;
        let mut checks = crate::fill::FillChecks::default();
        let mut clocks = Vec::new();
        let lanes = copy_lanes();
        let mut deferred: WideCopies<'_> = Vec::new();
        for table in tables {
            match table {
                TableUpload::Wide(columns) => {
                    for column in columns.iter() {
                        if lanes > 1 {
                            deferred.push((at, *column));
                        } else {
                            let mut slab = buffer.slice_mut(at..at + column.len());
                            crate::pinned::note_upload(column.as_ptr().cast(), column.len() * 8);
                            stream.memcpy_htod(*column, &mut slab).ok()?;
                        }
                        spans.push((at, column.len()));
                        at += column.len();
                    }
                }
                TableUpload::Narrow(packed) => {
                    let (rows, cols) = (packed.rows(), packed.cols());
                    let mut slab = buffer.slice_mut(at..at + rows * cols);
                    let clock = TableClock::start(&stream, false);
                    if staged {
                        crate::narrow::widen_col_major_into_staged(
                            &stream, be, *packed, rows, cols, &mut slab,
                        )
                        .ok()?;
                    } else {
                        crate::narrow::widen_col_major_into(
                            &stream, be, *packed, rows, cols, &mut slab,
                        )
                        .ok()?;
                    }
                    clocks.extend(clock.filter(|c| c.done.record(&stream).is_ok()));
                    spans.extend((0..cols).map(|k| (at + k * rows, rows)));
                    at += rows * cols;
                }
                TableUpload::Records(records) => {
                    let (rows, cols) = (records.rows(), records.cols());
                    let mut slab = buffer.slice_mut(at..at + rows * cols);
                    let clock = TableClock::start(&stream, true);
                    crate::fill::fill_col_major_into(&stream, *records, &mut slab, &mut checks)
                        .ok()?;
                    clocks.extend(clock.filter(|c| c.done.record(&stream).is_ok()));
                    spans.extend((0..cols).map(|k| (at + k * rows, rows)));
                    at += rows * cols;
                }
            }
        }
        copy_wide_lanes(&stream, &mut buffer, deferred, lanes)?;
        stream.synchronize().ok()?;
        if prepaid {
            room.shrink(room.bytes() - words);
        }
        clocks.into_iter().for_each(TableClock::add);
        // A fill whose records it refused (a CPU log at a pc with no DECODE
        // row): the card declines the group, and its readers take the host's
        // columns. The host fill refuses the same log, so that table reads as
        // no columns and the proof is one the verifier refuses. An honest run
        // never gets here: the walk refused such a log before any chunk.
        if !checks.passed(&stream).ok()? {
            crate::device::note_device_fallback();
            return None;
        }
        Some(Self {
            stream,
            buffer: Arc::new(buffer),
            spans,
            _room: room,
        })
    }

    /// The run of `width` columns from `first`, packed on the card (each
    /// column at the bytes its words need, [`crate::narrow`]): its widths and
    /// its packed bytes, the only bytes that come back. `None` when the columns
    /// are not a run or are too many words for one launch.
    pub fn pack_run(&self, first: usize, width: usize) -> Result<Option<(Vec<u8>, Vec<u8>)>> {
        if !self.is_run(first, width) {
            return Ok(None);
        }
        let rows = self.spans[first].1;
        if u32::try_from(rows * width).is_err() {
            return Ok(None);
        }
        let be = backend()?;
        let stream = be.next_stream();
        crate::narrow::pack_col_major_on_stream(&stream, be, &self.view(first, width), rows, width)
            .map(Some)
    }

    pub fn num_columns(&self) -> usize {
        self.spans.len()
    }

    /// Every column back on the host, in upload order: what a test compares.
    pub fn download(&self) -> Result<Vec<Vec<u64>>> {
        let all = crate::readback::clone_dtoh(&self.stream, &*self.buffer)?;
        self.stream.synchronize()?;
        Ok(self
            .spans
            .iter()
            .map(|&(at, len)| all[at..at + len].to_vec())
            .collect())
    }

    /// Whether `width` columns from `first` are a run of equal height — which
    /// is what the kernels that read a table's columns in place need.
    pub fn is_run(&self, first: usize, width: usize) -> bool {
        if width == 0 || first + width > self.spans.len() {
            return false;
        }
        let (start, rows) = self.spans[first];
        (0..width).all(|k| self.spans[first + k] == (start + k * rows, rows))
    }

    /// The one allocation every column lives in, shared.
    pub fn buffer(&self) -> Arc<CudaSlice<u64>> {
        self.buffer.clone()
    }

    /// Column `k`'s `(offset, len)` in elements, within [`buffer`](Self::buffer).
    pub fn span(&self, k: usize) -> (usize, usize) {
        self.spans[k]
    }

    /// The device address of column `k` and its length in elements.
    pub fn at(&self, k: usize) -> (u64, usize) {
        let (offset, len) = self.spans[k];
        let (base, _guard) = self.buffer.device_ptr(&self.stream);
        (base + (offset * 8) as u64, len)
    }

    /// A view over `width` columns from `first`, which must be a run.
    pub fn view(&self, first: usize, width: usize) -> cudarc::driver::CudaView<'_, u64> {
        let (offset, rows) = self.spans[first];
        self.buffer.slice(offset..offset + width * rows)
    }

    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    /// Copies column `k` into `dst` at `offset`, on the card.
    ///
    /// Issued on the **destination's** stream, so whatever reads `dst` next is
    /// ordered behind it. The store is written once and synchronized at upload
    /// and read-only after, so another stream reading it races with nothing.
    pub fn copy_into(
        &self,
        k: usize,
        dst: &mut CudaSlice<u64>,
        offset: usize,
        stream: &Arc<CudaStream>,
    ) -> Result<()> {
        let (src, len) = self.at(k);
        let (base, _guard) = dst.device_ptr_mut(stream);
        // SAFETY: both ranges are inside allocations this call holds a handle
        // to, and the caller checked `offset + len` against `dst`.
        unsafe {
            cudarc::driver::sys::cuMemcpyDtoDAsync_v2(
                base + (offset * 8) as u64,
                src,
                len * 8,
                stream.cu_stream(),
            )
            .result()?;
        }
        Ok(())
    }
}
