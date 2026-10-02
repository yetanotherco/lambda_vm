//! Widening a packed main trace on the device (the stark crate's `NarrowMain`
//! layout): column `c` holds `rows` little-endian words of `widths[c]` bytes
//! (1, 2, 4 or 8) from byte `offsets[c]` of the packed data. The words come
//! back bit for bit, row-major, as the LDE reads its input.

use std::sync::Arc;

use cudarc::driver::{CudaStream, CudaViewMut, LaunchConfig, PushKernelArg};

use crate::Result;
use crate::device::{Backend, backend};

/// A packed trace on the host (see the module docs).
#[derive(Clone, Copy, Debug)]
pub struct NarrowInput<'a> {
    data: &'a [u8],
    offsets: &'a [u64],
    widths: &'a [u8],
    rows: usize,
}

impl<'a> NarrowInput<'a> {
    /// `rows × cols` words packed as the module docs say.
    pub fn new(
        data: &'a [u8],
        offsets: &'a [u64],
        widths: &'a [u8],
        rows: usize,
        cols: usize,
    ) -> Self {
        assert_eq!(offsets.len(), cols, "packed trace: one offset per column");
        assert_eq!(widths.len(), cols, "packed trace: one width per column");
        Self {
            data,
            offsets,
            widths,
            rows,
        }
    }

    /// Rows of the trace.
    pub fn rows(&self) -> usize {
        self.rows
    }
}

/// Upload `t`'s packed columns and widen them into `dst` (`rows × cols`
/// row-major words), all on `stream`: a consumer on the same stream needs no
/// further ordering, and the packed copy is freed in stream order.
pub(crate) fn widen_into(
    stream: &Arc<CudaStream>,
    be: &Backend,
    t: NarrowInput,
    rows: usize,
    cols: usize,
    dst: &mut CudaViewMut<u64>,
) -> Result<()> {
    assert_eq!(t.rows, rows, "packed trace: rows");
    assert_eq!(t.widths.len(), cols, "packed trace: columns");
    let cells = rows * cols;
    assert_eq!(dst.len(), cells, "packed trace: destination size");
    if cells == 0 {
        return Ok(());
    }
    let data_dev = stream.clone_htod(t.data)?;
    let offsets_dev = stream.clone_htod(t.offsets)?;
    let widths_dev = stream.clone_htod(t.widths)?;
    let cfg =
        LaunchConfig::for_num_elems(u32::try_from(cells).expect("packed trace: cells fit u32"));
    let (rows_u64, cols_u64) = (rows as u64, cols as u64);
    unsafe {
        stream
            .launch_builder(&be.widen_narrow_row_major)
            .arg(&data_dev)
            .arg(&offsets_dev)
            .arg(&widths_dev)
            .arg(dst)
            .arg(&rows_u64)
            .arg(&cols_u64)
            .launch(cfg)?;
    }
    Ok(())
}

/// [`widen_into`] on a stream of its own, downloaded: the device widen's words,
/// for the round-trip tests.
pub fn widen_to_host(
    data: &[u8],
    offsets: &[u64],
    widths: &[u8],
    rows: usize,
    cols: usize,
) -> Result<Vec<u64>> {
    let be = backend()?;
    let stream = be.next_stream();
    // SAFETY: `widen_into` writes all `rows × cols` words.
    let mut out = unsafe { stream.alloc::<u64>(rows * cols) }?;
    widen_into(
        &stream,
        be,
        NarrowInput::new(data, offsets, widths, rows, cols),
        rows,
        cols,
        &mut out.slice_mut(..),
    )?;
    let words = stream.clone_dtoh(&out)?;
    stream.synchronize()?;
    Ok(words)
}
