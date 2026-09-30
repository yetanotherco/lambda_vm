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

use std::cell::Cell;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use cudarc::driver::{CudaSlice, CudaStream, DevicePtr, DevicePtrMut};

use crate::Result;
use crate::device::{DeviceReservation, alloc_or_trim, backend};

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

/// The columns themselves, laid end to end in one allocation.
pub struct DeviceColumns {
    stream: Arc<CudaStream>,
    /// Shared so a reader that outlives a borrow of the store — a WHIR opening's
    /// first rounds read the message where it lies — keeps it alive.
    buffer: Arc<CudaSlice<u64>>,
    /// `(offset, len)` in elements, per column, in the order uploaded.
    spans: Vec<(usize, usize)>,
    upload: UploadRecord,
    _room: DeviceReservation,
}

impl DeviceColumns {
    /// `None` when the card will not promise the room, in which case every
    /// caller uploads its own copy as before.
    pub fn upload(columns: &[&[u64]]) -> Option<Self> {
        if columns.is_empty() {
            return None;
        }
        let total: usize = columns.iter().map(|c| c.len()).sum();
        let be = backend().ok()?;
        let Some(room) = be.reserve(total as u64 * 8) else {
            crate::device::note_device_fallback();
            return None;
        };
        crate::argue_probe::note_device(crate::argue_probe::Surface::Columns, total as u64 * 8);
        let stream = be.next_stream();
        // SAFETY: every element is written by the copies below.
        let mut buffer = unsafe { alloc_or_trim::<u64>(&stream, total) }.ok()?;
        let mut spans = Vec::with_capacity(columns.len());
        let mut at = 0usize;
        for column in columns {
            spans.push((at, column.len()));
            at += column.len();
        }
        let started = std::time::Instant::now();
        let mode = UploadMode::get();
        let skipped = match mode {
            UploadMode::Pageable => {
                for (column, &(at, len)) in columns.iter().zip(&spans) {
                    let mut slab = buffer.slice_mut(at..at + len);
                    stream.memcpy_htod(*column, &mut slab).ok()?;
                }
                0
            }
            UploadMode::Staged { threads } => {
                upload_staged(&stream, &mut buffer, columns, &spans, threads).ok()?
            }
        };
        stream.synchronize().ok()?;
        let upload = UploadRecord {
            bytes: total as u64 * 8,
            skipped,
            secs: started.elapsed().as_secs_f64(),
            threads: match mode {
                UploadMode::Pageable => 0,
                UploadMode::Staged { threads } => threads,
            },
        };
        UPLOADS.note(&upload);
        Some(Self {
            stream,
            buffer: Arc::new(buffer),
            spans,
            upload,
            _room: room,
        })
    }

    /// What this store's upload moved, skipped and took.
    pub fn upload_record(&self) -> UploadRecord {
        self.upload
    }

    pub fn num_columns(&self) -> usize {
        self.spans.len()
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

// ── The upload's path ────────────────────────────────────────────────────────

/// The knob for the base's trace upload: `1` (both parts) or `columns` sends the
/// epoch's columns through staging pairs from several threads, each column's
/// all-zero tail left behind and zeroed on the card; unset, empty or `0` (the
/// default) is one stream of pageable copies. `head` is the prover's part only
/// (`decode_prepared_for` beside epoch 0), which this crate ignores. The card
/// ends up holding the same values either way, so the proofs are identical.
pub const TRACE_UPLOAD_ENV: &str = "LAMBDA_VM_TRACE_UPLOAD";

/// Threads the staged upload fills pairs from (default 4, at most
/// [`MAX_STAGING_PAIRS`](crate::device::MAX_STAGING_PAIRS): one pair each).
pub const TRACE_UPLOAD_THREADS_ENV: &str = "LAMBDA_VM_TRACE_UPLOAD_THREADS";

/// A tail shorter than this is uploaded with its column: a memset per column is
/// a launch, and below this the bytes it saves are worth less than the launch.
pub const ZERO_TAIL_MIN_BYTES: usize = 64 << 10;

/// Whether [`TRACE_UPLOAD_ENV`]'s raw value turns the staged column upload on.
/// Anything unrecognised is off: a typo measures the default, never a crash.
pub fn columns_staged_setting(raw: Option<&str>) -> bool {
    matches!(raw.map(str::trim), Some("1" | "columns"))
}

/// [`TRACE_UPLOAD_THREADS_ENV`]'s raw value as a thread count.
pub fn upload_threads_setting(raw: Option<&str>) -> usize {
    raw.and_then(|v| v.trim().parse::<usize>().ok())
        .unwrap_or(4)
        .clamp(1, crate::device::MAX_STAGING_PAIRS)
}

thread_local! {
    static STAGED_OVERRIDE: Cell<Option<bool>> = const { Cell::new(None) };
}

/// Pin this thread's column uploads to one path (`Some`), or back to the
/// environment's answer (`None`). For tests that upload the same columns both
/// ways in one process.
pub fn set_trace_upload_override(on: Option<bool>) {
    STAGED_OVERRIDE.with(|o| o.set(on));
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum UploadMode {
    Pageable,
    Staged { threads: usize },
}

impl UploadMode {
    fn get() -> Self {
        static FROM_ENV: std::sync::OnceLock<(bool, usize)> = std::sync::OnceLock::new();
        let (on, threads) = *FROM_ENV.get_or_init(|| {
            (
                columns_staged_setting(std::env::var(TRACE_UPLOAD_ENV).ok().as_deref()),
                upload_threads_setting(std::env::var(TRACE_UPLOAD_THREADS_ENV).ok().as_deref()),
            )
        });
        let on = STAGED_OVERRIDE.with(Cell::get).unwrap_or(on);
        if on {
            Self::Staged { threads }
        } else {
            Self::Pageable
        }
    }
}

/// The length of `column` once its all-zero tail is cut, or the whole column
/// when that tail is shorter than [`ZERO_TAIL_MIN_BYTES`].
pub fn sent_len(column: &[u64]) -> usize {
    let live = column.iter().rposition(|&v| v != 0).map_or(0, |i| i + 1);
    if (column.len() - live) * 8 < ZERO_TAIL_MIN_BYTES {
        column.len()
    } else {
        live
    }
}

/// Fills `buffer` with `columns` at `spans` from `threads` threads, each taking
/// the next column, staging what [`sent_len`] keeps through a pair of its own
/// and zeroing the rest on the card. Every copy and memset is queued on
/// `stream`; the caller synchronizes it. Returns the bytes left behind.
fn upload_staged(
    stream: &Arc<CudaStream>,
    buffer: &mut CudaSlice<u64>,
    columns: &[&[u64]],
    spans: &[(usize, usize)],
    threads: usize,
) -> Result<u64> {
    let be = backend()?;
    // Held across the scope: it orders the writes on `stream` for cudarc's
    // event tracking, and the buffer outlives every copy queued here because
    // the caller synchronizes before it can drop.
    let (base, _record) = buffer.device_ptr_mut(stream);
    let next = AtomicUsize::new(0);
    let skipped = AtomicU64::new(0);
    let fill = || -> Result<()> {
        be.ctx.bind_to_thread()?;
        loop {
            let k = next.fetch_add(1, Ordering::Relaxed);
            let (Some(column), Some(&(at, len))) = (columns.get(k), spans.get(k)) else {
                return Ok(());
            };
            let sent = sent_len(column);
            let dst = base + (at * 8) as u64;
            // SAFETY: `column[..sent]` is initialised host memory; the spans
            // are disjoint and inside `buffer` (laid end to end over `total`),
            // so each thread writes a range no other copy touches.
            unsafe {
                let bytes = std::slice::from_raw_parts(column.as_ptr() as *const u8, sent * 8);
                crate::device::htod_staged_raw(stream, bytes, dst)?;
                if sent < len {
                    cudarc::driver::sys::cuMemsetD8Async(
                        dst + (sent * 8) as u64,
                        0,
                        (len - sent) * 8,
                        stream.cu_stream(),
                    )
                    .result()?;
                    skipped.fetch_add(((len - sent) * 8) as u64, Ordering::Relaxed);
                }
            }
        }
    };
    let results: Vec<Result<()>> = std::thread::scope(|s| {
        let workers: Vec<_> = (0..threads.min(columns.len()).max(1))
            .map(|_| s.spawn(fill))
            .collect();
        workers
            .into_iter()
            .map(|w| {
                w.join().unwrap_or(Err(cudarc::driver::DriverError(
                    cudarc::driver::sys::CUresult::CUDA_ERROR_UNKNOWN,
                )))
            })
            .collect()
    });
    for result in results {
        result?;
    }
    Ok(skipped.load(Ordering::Relaxed))
}

/// One column upload: bytes the columns hold, bytes left behind as zero tails,
/// wall seconds to the stream's synchronize, and threads (0 = pageable).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct UploadRecord {
    pub bytes: u64,
    pub skipped: u64,
    pub secs: f64,
    pub threads: usize,
}

impl UploadRecord {
    /// One line for the log.
    pub fn line(&self) -> String {
        let path = if self.threads == 0 {
            "pageable".to_string()
        } else {
            format!("staged x{}", self.threads)
        };
        let rate = if self.secs > 0.0 {
            (self.bytes - self.skipped) as f64 / 1e9 / self.secs
        } else {
            0.0
        };
        format!(
            "{:.3} GB in {:.3}s, {:.3} GB zero tails not sent, {rate:.1} GB/s sent ({path})",
            self.bytes as f64 / 1e9,
            self.secs,
            self.skipped as f64 / 1e9,
        )
    }
}

struct UploadTotals {
    calls: AtomicU64,
    bytes: AtomicU64,
    skipped: AtomicU64,
    nanos: AtomicU64,
    staged_calls: AtomicU64,
}

static UPLOADS: UploadTotals = UploadTotals {
    calls: AtomicU64::new(0),
    bytes: AtomicU64::new(0),
    skipped: AtomicU64::new(0),
    nanos: AtomicU64::new(0),
    staged_calls: AtomicU64::new(0),
};

impl UploadTotals {
    fn note(&self, record: &UploadRecord) {
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.bytes.fetch_add(record.bytes, Ordering::Relaxed);
        self.skipped.fetch_add(record.skipped, Ordering::Relaxed);
        self.nanos
            .fetch_add((record.secs * 1e9) as u64, Ordering::Relaxed);
        if record.threads > 0 {
            self.staged_calls.fetch_add(1, Ordering::Relaxed);
        }
    }
}

/// `(uploads, of which staged, bytes, bytes skipped, seconds)` since the
/// process started, for a test that asserts which path ran.
pub fn upload_totals() -> (u64, u64, u64, u64, f64) {
    (
        UPLOADS.calls.load(Ordering::Relaxed),
        UPLOADS.staged_calls.load(Ordering::Relaxed),
        UPLOADS.bytes.load(Ordering::Relaxed),
        UPLOADS.skipped.load(Ordering::Relaxed),
        UPLOADS.nanos.load(Ordering::Relaxed) as f64 / 1e9,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_knob_is_off_unless_asked() {
        for raw in [
            None,
            Some(""),
            Some("0"),
            Some("head"),
            Some("yes"),
            Some("2"),
        ] {
            assert!(!columns_staged_setting(raw), "{raw:?}");
        }
        for raw in [Some("1"), Some("columns"), Some(" 1 ")] {
            assert!(columns_staged_setting(raw), "{raw:?}");
        }
    }

    #[test]
    fn threads_default_to_four_within_the_pairs() {
        assert_eq!(upload_threads_setting(None), 4);
        assert_eq!(upload_threads_setting(Some("x")), 4);
        assert_eq!(upload_threads_setting(Some("0")), 1);
        assert_eq!(upload_threads_setting(Some("3")), 3);
        assert_eq!(
            upload_threads_setting(Some("99")),
            crate::device::MAX_STAGING_PAIRS
        );
    }

    #[test]
    fn the_override_beats_the_environment_on_its_thread_only() {
        set_trace_upload_override(Some(true));
        assert!(matches!(UploadMode::get(), UploadMode::Staged { .. }));
        std::thread::spawn(|| {
            // Another thread sees the environment's answer, which a test run
            // leaves unset.
            if std::env::var_os(TRACE_UPLOAD_ENV).is_none() {
                assert_eq!(UploadMode::get(), UploadMode::Pageable);
            }
        })
        .join()
        .unwrap();
        set_trace_upload_override(Some(false));
        assert_eq!(UploadMode::get(), UploadMode::Pageable);
        set_trace_upload_override(None);
    }

    /// Only a zero tail of at least [`ZERO_TAIL_MIN_BYTES`] is cut, and it is cut
    /// at the last nonzero value exactly: the card must end up with every value
    /// the column holds.
    #[test]
    fn a_column_keeps_everything_up_to_its_last_nonzero_value() {
        let min = ZERO_TAIL_MIN_BYTES / 8;
        // No tail, a short tail, an exact-threshold tail, a long one.
        let mut c = vec![7u64; 4 * min];
        assert_eq!(sent_len(&c), 4 * min);
        c[4 * min - 1] = 0;
        assert_eq!(sent_len(&c), 4 * min, "a one-value tail is sent");
        for v in &mut c[3 * min..] {
            *v = 0;
        }
        assert_eq!(
            sent_len(&c),
            3 * min,
            "a tail of exactly the threshold is cut"
        );
        c[3 * min + 5] = 1;
        assert_eq!(
            sent_len(&c),
            4 * min,
            "a nonzero value inside the tail is kept"
        );
        c[3 * min + 5] = 0;
        c[..].iter_mut().skip(min).for_each(|v| *v = 0);
        assert_eq!(sent_len(&c), min);
        // An all-zero column sends nothing once it is long enough, and a zero
        // inside the live part never shortens it.
        assert_eq!(sent_len(&vec![0u64; 2 * min]), 0);
        assert_eq!(sent_len(&vec![0u64; min - 1]), min - 1);
        let mut holes = vec![0u64; 4 * min];
        holes[0] = 1;
        holes[2 * min] = 1;
        assert_eq!(sent_len(&holes), 2 * min + 1);
        assert_eq!(sent_len(&[]), 0);
    }

    #[test]
    fn the_line_reads_the_record() {
        let r = UploadRecord {
            bytes: 2_000_000_000,
            skipped: 500_000_000,
            secs: 0.05,
            threads: 4,
        };
        assert_eq!(
            r.line(),
            "2.000 GB in 0.050s, 0.500 GB zero tails not sent, 30.0 GB/s sent (staged x4)"
        );
        let p = UploadRecord {
            threads: 0,
            skipped: 0,
            ..r
        };
        assert!(
            p.line().ends_with("40.0 GB/s sent (pageable)"),
            "{}",
            p.line()
        );
    }
}
