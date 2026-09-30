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
//!
//! The columns may come from page-locked memory ([`DeviceColumns::upload_parts`],
//! D-TRACE stage 1b), whose copies return before they finish. Every upload
//! synchronizes its stream before it returns, on every path, so no copy reads
//! the host side after the borrow of it ends.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

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
        let parts: Vec<(&[u64], usize)> = columns.iter().map(|c| (*c, c.len())).collect();
        Self::upload_parts(&parts, false)
    }

    /// [`Self::upload`], sending only the first `sent` values of each
    /// `(column, sent)` and leaving the rest zero on the card: when any column
    /// leaves a tail, one memset over the whole buffer goes first, then one
    /// copy per nonempty prefix, on one stream. With every `sent` the column's
    /// length that is [`Self::upload`] exactly: no memset, the same copies.
    ///
    /// `pinned` says the columns are page-locked, which only labels the record:
    /// the driver sees it from the addresses. From page-locked memory the
    /// copies are DMA and return before they finish, so the stream is
    /// synchronized before this returns, whatever happens.
    pub fn upload_parts(parts: &[(&[u64], usize)], pinned: bool) -> Option<Self> {
        if parts.is_empty() {
            return None;
        }
        let total: usize = parts.iter().map(|(column, _)| column.len()).sum();
        let be = backend().ok()?;
        let Some(room) = be.reserve(total as u64 * 8) else {
            crate::device::note_device_fallback();
            return None;
        };
        crate::argue_probe::note_device(crate::argue_probe::Surface::Columns, total as u64 * 8);
        let stream = be.next_stream();
        // SAFETY: every element is written below, by a copy or by the memset.
        let mut buffer = unsafe { alloc_or_trim::<u64>(&stream, total) }.ok()?;
        let started = std::time::Instant::now();
        let sent_of = |&(column, sent): &(&[u64], usize)| sent.min(column.len());
        let mut spans = Vec::with_capacity(parts.len());
        let mut zero_tails = 0u64;
        let issued = (|| -> Result<()> {
            if parts.iter().any(|part| sent_of(part) < part.0.len()) {
                stream.memset_zeros(&mut buffer)?;
            }
            let mut at = 0usize;
            for part in parts {
                let (column, sent) = (part.0, sent_of(part));
                if sent > 0 {
                    let mut slab = buffer.slice_mut(at..at + sent);
                    stream.memcpy_htod(&column[..sent], &mut slab)?;
                }
                zero_tails += ((column.len() - sent) * 8) as u64;
                spans.push((at, column.len()));
                at += column.len();
            }
            Ok(())
        })();
        // Before any return: a copy from page-locked memory may still be
        // reading it.
        let synced = stream.synchronize();
        issued.ok()?;
        synced.ok()?;
        let upload = UploadRecord {
            bytes: total as u64 * 8,
            zero_tails,
            secs: started.elapsed().as_secs_f64(),
            pinned,
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

    /// What this store's upload held, left behind and took.
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

/// One column upload: the bytes the columns hold, the bytes left behind as
/// zero tails (zeroed on the card instead of sent), the wall seconds from the
/// first write to the stream's synchronize, and whether the host side was
/// page-locked.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct UploadRecord {
    pub bytes: u64,
    pub zero_tails: u64,
    pub secs: f64,
    pub pinned: bool,
}

impl UploadRecord {
    /// One line for the log:
    /// `<GB> sent · <GB> zero tails · <s> · <GB/s> (pinned|pageable)`, the rate
    /// over the bytes sent.
    pub fn line(&self) -> String {
        let sent = self.bytes.saturating_sub(self.zero_tails);
        let rate = if self.secs > 0.0 {
            sent as f64 / 1e9 / self.secs
        } else {
            0.0
        };
        format!(
            "{:.3} GB sent · {:.3} GB zero tails · {:.3}s · {rate:.1} GB/s ({})",
            sent as f64 / 1e9,
            self.zero_tails as f64 / 1e9,
            self.secs,
            if self.pinned { "pinned" } else { "pageable" },
        )
    }
}

struct UploadTotals {
    calls: AtomicU64,
    pinned_calls: AtomicU64,
    bytes: AtomicU64,
    zero_tails: AtomicU64,
}

static UPLOADS: UploadTotals = UploadTotals {
    calls: AtomicU64::new(0),
    pinned_calls: AtomicU64::new(0),
    bytes: AtomicU64::new(0),
    zero_tails: AtomicU64::new(0),
};

impl UploadTotals {
    fn note(&self, record: &UploadRecord) {
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.bytes.fetch_add(record.bytes, Ordering::Relaxed);
        self.zero_tails
            .fetch_add(record.zero_tails, Ordering::Relaxed);
        if record.pinned {
            self.pinned_calls.fetch_add(1, Ordering::Relaxed);
        }
    }
}

/// Column uploads since the process started: `(uploads, of which pinned,
/// bytes, bytes left behind as zero tails)`, for a test that asserts which
/// path ran.
pub fn upload_totals() -> (u64, u64, u64, u64) {
    (
        UPLOADS.calls.load(Ordering::Relaxed),
        UPLOADS.pinned_calls.load(Ordering::Relaxed),
        UPLOADS.bytes.load(Ordering::Relaxed),
        UPLOADS.zero_tails.load(Ordering::Relaxed),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_line_reads_the_record() {
        let pinned = UploadRecord {
            bytes: 2_500_000_000,
            zero_tails: 500_000_000,
            secs: 0.04,
            pinned: true,
        };
        assert_eq!(
            pinned.line(),
            "2.000 GB sent · 0.500 GB zero tails · 0.040s · 50.0 GB/s (pinned)"
        );
        let pageable = UploadRecord {
            zero_tails: 0,
            pinned: false,
            ..pinned
        };
        assert_eq!(
            pageable.line(),
            "2.500 GB sent · 0.000 GB zero tails · 0.040s · 62.5 GB/s (pageable)"
        );
        assert_eq!(
            UploadRecord::default().line(),
            "0.000 GB sent · 0.000 GB zero tails · 0.000s · 0.0 GB/s (pageable)"
        );
    }
}
