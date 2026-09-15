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

/// The columns themselves, laid end to end in one allocation.
pub struct DeviceColumns {
    stream: Arc<CudaStream>,
    buffer: CudaSlice<u64>,
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
        let room = be.reserve(total as u64 * 8)?;
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
            buffer,
            spans,
            _room: room,
        })
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
