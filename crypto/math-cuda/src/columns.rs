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
        let zero_tails = zero_tails_on();
        let (skipped, scan_on_path) = if zero_tails {
            upload_zero_tails(&stream, &mut buffer, columns, &spans).ok()?
        } else {
            for (column, &(at, len)) in columns.iter().zip(&spans) {
                let mut slab = buffer.slice_mut(at..at + len);
                stream.memcpy_htod(*column, &mut slab).ok()?;
            }
            (0, 0.0)
        };
        stream.synchronize().ok()?;
        let upload = UploadRecord {
            bytes: total as u64 * 8,
            skipped,
            secs: started.elapsed().as_secs_f64(),
            zero_tails,
            scan_on_path,
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

    /// What this store's upload moved, left behind and took.
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

// ── The upload's zero tails ──────────────────────────────────────────────────

/// The knob for the columns' upload: `zerotail` leaves each column's all-zero
/// tail of at least [`ZERO_TAIL_MIN_BYTES`] behind and zeroes it on the card,
/// the rest still one stream of pageable copies; anything else, unset included
/// (the default), sends every byte. The card ends up holding the same values
/// either way, so the proofs are identical.
pub const TRACE_UPLOAD_ENV: &str = "LAMBDA_VM_TRACE_UPLOAD";

/// A tail shorter than this is sent with its column: each tail left behind is a
/// memset, and below this the bytes it saves are worth less than the call.
pub const ZERO_TAIL_MIN_BYTES: usize = 64 << 10;

/// Threads finding the tails ahead of the copies. The scan reads only the
/// tails, backwards, so two stay ahead of one stream of pageable copies.
const SCANNERS: usize = 2;

/// Whether [`TRACE_UPLOAD_ENV`]'s raw value turns the zero-tail upload on.
/// Anything unrecognised is off: a typo measures the default, never a crash.
pub fn zero_tails_setting(raw: Option<&str>) -> bool {
    matches!(raw.map(str::trim), Some("zerotail"))
}

thread_local! {
    static ZERO_TAILS_OVERRIDE: Cell<Option<bool>> = const { Cell::new(None) };
}

/// Pin this thread's column uploads to one path (`Some`), or back to the
/// environment's answer (`None`). For tests that upload the same columns both
/// ways in one process.
pub fn set_trace_upload_override(on: Option<bool>) {
    ZERO_TAILS_OVERRIDE.with(|o| o.set(on));
}

/// This thread's override, else the environment's answer, read once.
fn zero_tails_on() -> bool {
    static FROM_ENV: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    ZERO_TAILS_OVERRIDE.with(Cell::get).unwrap_or_else(|| {
        *FROM_ENV
            .get_or_init(|| zero_tails_setting(std::env::var(TRACE_UPLOAD_ENV).ok().as_deref()))
    })
}

/// One past `column`'s last nonzero value, found from the end: eight values at
/// a time while they are all zero, then one at a time.
fn live_len(column: &[u64]) -> usize {
    let mut end = column.len();
    while end >= 8 && column[end - 8..end].iter().fold(0, |acc, &v| acc | v) == 0 {
        end -= 8;
    }
    while end > 0 && column[end - 1] == 0 {
        end -= 1;
    }
    end
}

/// The length of `column` once its all-zero tail is cut, or the whole column
/// when that tail is shorter than [`ZERO_TAIL_MIN_BYTES`].
pub fn sent_len(column: &[u64]) -> usize {
    let live = live_len(column);
    if (column.len() - live) * 8 < ZERO_TAIL_MIN_BYTES {
        column.len()
    } else {
        live
    }
}

/// Fills `buffer` with `columns` at `spans` over `stream` in pageable copies,
/// each column's [`sent_len`] prefix sent and the rest zeroed on the card.
///
/// [`SCANNERS`] threads find the lengths ahead, in column order; the copies
/// take each one as it is published, and a column whose length is not there
/// yet is scanned on this thread instead of waited for, so a slow or failed
/// scanner costs time, never a hang. Returns the bytes left behind and the
/// seconds this thread spent scanning.
fn upload_zero_tails(
    stream: &Arc<CudaStream>,
    buffer: &mut CudaSlice<u64>,
    columns: &[&[u64]],
    spans: &[(usize, usize)],
) -> Result<(u64, f64)> {
    const UNSCANNED: usize = usize::MAX;
    let found: Vec<AtomicUsize> = columns
        .iter()
        .map(|_| AtomicUsize::new(UNSCANNED))
        .collect();
    let next = AtomicUsize::new(0);
    std::thread::scope(|s| {
        for _ in 0..SCANNERS.min(columns.len()) {
            s.spawn(|| {
                loop {
                    let k = next.fetch_add(1, Ordering::Relaxed);
                    let Some(column) = columns.get(k) else {
                        return;
                    };
                    found[k].store(sent_len(column), Ordering::Release);
                }
            });
        }
        let (mut skipped, mut on_path) = (0u64, std::time::Duration::ZERO);
        for (k, (column, &(at, len))) in columns.iter().zip(spans).enumerate() {
            let mut sent = found[k].load(Ordering::Acquire);
            if sent == UNSCANNED {
                let scan = std::time::Instant::now();
                sent = sent_len(column);
                on_path += scan.elapsed();
            }
            if sent > 0 {
                let mut slab = buffer.slice_mut(at..at + sent);
                stream.memcpy_htod(&column[..sent], &mut slab)?;
            }
            if sent < len {
                let mut tail = buffer.slice_mut(at + sent..at + len);
                stream.memset_zeros(&mut tail)?;
                skipped += ((len - sent) * 8) as u64;
            }
        }
        // The scanners may still be claiming columns this thread already
        // scanned; nothing waits on them, and the scope joins them.
        next.store(columns.len(), Ordering::Relaxed);
        Ok((skipped, on_path.as_secs_f64()))
    })
}

/// One column upload: bytes the columns hold, bytes left behind as zero tails,
/// wall seconds to the stream's synchronize, the path, and the seconds the
/// uploading thread spent scanning because no scanner was ahead of it.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct UploadRecord {
    pub bytes: u64,
    pub skipped: u64,
    pub secs: f64,
    pub zero_tails: bool,
    pub scan_on_path: f64,
}

impl UploadRecord {
    /// One line for the log.
    pub fn line(&self) -> String {
        let path = if self.zero_tails {
            format!("zerotail, scan on the path {:.3}s", self.scan_on_path)
        } else {
            "pageable".to_string()
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
    zero_tail_calls: AtomicU64,
    bytes: AtomicU64,
    skipped: AtomicU64,
}

static UPLOADS: UploadTotals = UploadTotals {
    calls: AtomicU64::new(0),
    zero_tail_calls: AtomicU64::new(0),
    bytes: AtomicU64::new(0),
    skipped: AtomicU64::new(0),
};

impl UploadTotals {
    fn note(&self, record: &UploadRecord) {
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.bytes.fetch_add(record.bytes, Ordering::Relaxed);
        self.skipped.fetch_add(record.skipped, Ordering::Relaxed);
        if record.zero_tails {
            self.zero_tail_calls.fetch_add(1, Ordering::Relaxed);
        }
    }
}

/// `(uploads, of which with zero tails left behind, bytes, bytes left behind)`
/// since the process started, for a test that asserts which path ran.
pub fn upload_totals() -> (u64, u64, u64, u64) {
    (
        UPLOADS.calls.load(Ordering::Relaxed),
        UPLOADS.zero_tail_calls.load(Ordering::Relaxed),
        UPLOADS.bytes.load(Ordering::Relaxed),
        UPLOADS.skipped.load(Ordering::Relaxed),
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
            Some("1"),
            Some("columns"),
            Some("head"),
            Some("zero"),
        ] {
            assert!(!zero_tails_setting(raw), "{raw:?}");
        }
        for raw in [Some("zerotail"), Some(" zerotail ")] {
            assert!(zero_tails_setting(raw), "{raw:?}");
        }
    }

    #[test]
    fn the_override_beats_the_environment_on_its_thread_only() {
        set_trace_upload_override(Some(true));
        assert!(zero_tails_on());
        std::thread::spawn(|| {
            // Another thread sees the environment's answer, which a test run
            // leaves unset.
            if std::env::var_os(TRACE_UPLOAD_ENV).is_none() {
                assert!(!zero_tails_on());
            }
        })
        .join()
        .unwrap();
        set_trace_upload_override(Some(false));
        assert!(!zero_tails_on());
        set_trace_upload_override(None);
    }

    /// Only a zero tail of at least [`ZERO_TAIL_MIN_BYTES`] is cut, and it is cut
    /// at the last nonzero value exactly: the card must end up with every value
    /// the column holds.
    #[test]
    fn a_column_keeps_everything_up_to_its_last_nonzero_value() {
        let min = ZERO_TAIL_MIN_BYTES / 8;
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
        assert_eq!(sent_len(&vec![0u64; 2 * min]), 0);
        assert_eq!(sent_len(&vec![0u64; min - 1]), min - 1);
        let mut holes = vec![0u64; 4 * min];
        holes[0] = 1;
        holes[2 * min] = 1;
        assert_eq!(sent_len(&holes), 2 * min + 1);
        assert_eq!(sent_len(&[]), 0);
    }

    /// The eight-at-a-time scan finds the same last nonzero value as a plain
    /// one-at-a-time scan, for every length mod 8 and every position of that
    /// value, the high bit included.
    #[test]
    fn the_wide_scan_agrees_with_the_plain_one() {
        let plain = |c: &[u64]| c.iter().rposition(|&v| v != 0).map_or(0, |i| i + 1);
        for len in 0..40usize {
            let zeros = vec![0u64; len];
            assert_eq!(live_len(&zeros), 0, "all zero, len {len}");
            for at in 0..len {
                for value in [1u64, 1 << 63, u64::MAX] {
                    let mut c = zeros.clone();
                    c[at] = value;
                    assert_eq!(live_len(&c), plain(&c), "len {len}, value at {at}");
                    if at > 0 {
                        c[0] = 3;
                        assert_eq!(
                            live_len(&c),
                            plain(&c),
                            "len {len}, value at {at}, head set"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn the_line_reads_the_record() {
        let r = UploadRecord {
            bytes: 2_000_000_000,
            skipped: 500_000_000,
            secs: 0.05,
            zero_tails: true,
            scan_on_path: 0.0015,
        };
        assert_eq!(
            r.line(),
            "2.000 GB in 0.050s, 0.500 GB zero tails not sent, 30.0 GB/s sent (zerotail, scan on the path \
             0.002s)"
        );
        let p = UploadRecord {
            zero_tails: false,
            skipped: 0,
            scan_on_path: 0.0,
            ..r
        };
        assert_eq!(
            p.line(),
            "2.000 GB in 0.050s, 0.000 GB zero tails not sent, 40.0 GB/s sent (pageable)"
        );
    }
}
