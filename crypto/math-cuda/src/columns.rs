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

/// A table's columns as they go up: as they are, or packed (each column at the
/// bytes its words need, [`crate::narrow`]) and widened on the card.
#[derive(Clone, Copy)]
pub enum TableUpload<'a> {
    /// The columns, every one the table's height.
    Wide(&'a [&'a [u64]]),
    Narrow(crate::narrow::NarrowInput<'a>),
}

impl TableUpload<'_> {
    /// Words the table takes on the card.
    pub fn cells(&self) -> usize {
        match self {
            Self::Wide(columns) => columns.iter().map(|c| c.len()).sum(),
            Self::Narrow(packed) => packed.rows() * packed.cols(),
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
    /// offsets, as its widened columns uploaded. `None` as for [`Self::upload`],
    /// or when a packed table is too big for one launch.
    pub fn upload_tables(tables: &[TableUpload<'_>]) -> Option<Self> {
        let total: usize = tables.iter().map(TableUpload::cells).sum();
        if total == 0
            || tables.iter().any(|table| {
                matches!(table, TableUpload::Narrow(_)) && u32::try_from(table.cells()).is_err()
            })
        {
            return None;
        }
        let be = backend().ok()?;
        let Some(room) = be.reserve(total as u64 * 8) else {
            crate::device::note_device_fallback();
            return None;
        };
        // The packed bytes sit on the card only while their table widens, one
        // table at a time in stream order: promised for the largest of them.
        let staging = tables
            .iter()
            .map(|table| match table {
                TableUpload::Wide(_) => 0,
                TableUpload::Narrow(packed) => packed.bytes() as u64,
            })
            .max()
            .unwrap_or(0);
        let _staging = match staging {
            0 => None,
            bytes => {
                let Some(promise) = be.reserve(bytes) else {
                    crate::device::note_device_fallback();
                    return None;
                };
                Some(promise)
            }
        };
        crate::argue_probe::note_device(crate::argue_probe::Surface::Columns, total as u64 * 8);
        let stream = be.next_stream();
        // SAFETY: every element is written by the copies and widens below.
        let mut buffer = unsafe { alloc_or_trim::<u64>(&stream, total) }.ok()?;
        let mut spans = Vec::new();
        let mut at = 0usize;
        for table in tables {
            match table {
                TableUpload::Wide(columns) => {
                    for column in columns.iter() {
                        let mut slab = buffer.slice_mut(at..at + column.len());
                        stream.memcpy_htod(*column, &mut slab).ok()?;
                        spans.push((at, column.len()));
                        at += column.len();
                    }
                }
                TableUpload::Narrow(packed) => {
                    let (rows, cols) = (packed.rows(), packed.cols());
                    let mut slab = buffer.slice_mut(at..at + rows * cols);
                    crate::narrow::widen_col_major_into(
                        &stream, be, *packed, rows, cols, &mut slab,
                    )
                    .ok()?;
                    spans.extend((0..cols).map(|k| (at + k * rows, rows)));
                    at += rows * cols;
                }
            }
        }
        stream.synchronize().ok()?;
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

    /// [`Self::pack_run`], downloading into pages of their own when `pages`
    /// is set ([`crate::narrow::PackedData`]).
    pub fn pack_run_to(
        &self,
        first: usize,
        width: usize,
        pages: bool,
    ) -> Result<Option<(Vec<u8>, crate::narrow::PackedData)>> {
        if !self.is_run(first, width) {
            return Ok(None);
        }
        let rows = self.spans[first].1;
        if u32::try_from(rows * width).is_err() {
            return Ok(None);
        }
        let be = backend()?;
        let stream = be.next_stream();
        crate::narrow::pack_col_major_on_stream_to(
            &stream,
            be,
            &self.view(first, width),
            rows,
            width,
            pages,
        )
        .map(Some)
    }

    pub fn num_columns(&self) -> usize {
        self.spans.len()
    }

    /// Every column back on the host, in upload order: what a test compares.
    pub fn download(&self) -> Result<Vec<Vec<u64>>> {
        let all = self.stream.clone_dtoh(&*self.buffer)?;
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
