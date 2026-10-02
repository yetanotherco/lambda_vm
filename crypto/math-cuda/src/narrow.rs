//! Widening a packed main trace on the device (the stark crate's `NarrowMain`
//! layout): column `c` holds `rows` little-endian words of `widths[c]` bytes
//! (1, 2, 4 or 8) from byte `offsets[c]` of the packed data. The words come
//! back bit for bit, row-major, as the LDE reads its input.

use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream, CudaViewMut, LaunchConfig, PushKernelArg};

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

/// The bytes a word needs: 1, 2, 4 or 8 (the stark crate's `NarrowMain` rule).
pub fn width_of(max: u64) -> u8 {
    match max {
        0..=0xff => 1,
        0x100..=0xffff => 2,
        0x1_0000..=0xffff_ffff => 4,
        _ => 8,
    }
}

/// Pack the trace snapshot a commit left on the device (`handle.trace_dev`,
/// column-major) the way the stark crate's `NarrowMain` packs a host trace:
/// `(widths, data)`, column `c` from byte `Σ rows · widths[c' < c]` of `data`.
/// `None` when the handle kept no snapshot. Runs on a stream of its own after
/// the commit's `ready` event; only the packed bytes cross the bus.
pub fn pack_trace_snapshot(handle: &crate::lde::GpuLdeBase) -> Result<Option<(Vec<u8>, Vec<u8>)>> {
    let Some(src) = handle.trace_dev.as_deref() else {
        return Ok(None);
    };
    let (rows, cols) = (handle.trace_rows, handle.m);
    if rows == 0 || cols == 0 || src.len() < rows * cols {
        return Ok(None);
    }
    let be = backend()?;
    let stream = be.next_stream();
    handle.wait_ready_on(&stream)?;
    pack_col_major_on_stream(&stream, be, src, rows, cols).map(Some)
}

fn pack_col_major_on_stream(
    stream: &Arc<CudaStream>,
    be: &Backend,
    src: &CudaSlice<u64>,
    rows: usize,
    cols: usize,
) -> Result<(Vec<u8>, Vec<u8>)> {
    let mut max_dev = stream.alloc_zeros::<u64>(cols)?;
    let rows_u64 = rows as u64;
    let cfg = LaunchConfig {
        grid_dim: (
            u32::try_from(rows.div_ceil(256).min(64)).expect("at most 64"),
            u32::try_from(cols).expect("packed trace: columns fit u32"),
            1,
        ),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&be.column_max_col_major)
            .arg(src)
            .arg(&rows_u64)
            .arg(&mut max_dev)
            .launch(cfg)?;
    }
    let maxes = stream.clone_dtoh(&max_dev)?;
    stream.synchronize()?;
    let widths: Vec<u8> = maxes.into_iter().map(width_of).collect();
    let mut offsets = Vec::with_capacity(cols);
    let mut total = 0u64;
    for &w in &widths {
        offsets.push(total);
        total += rows_u64 * u64::from(w);
    }
    let offsets_dev = stream.clone_htod(&offsets)?;
    let widths_dev = stream.clone_htod(&widths)?;
    // SAFETY: the kernel writes every byte: each column's `rows × width`.
    let mut out = unsafe { stream.alloc::<u8>(total as usize) }?;
    let cells = rows * cols;
    let cfg =
        LaunchConfig::for_num_elems(u32::try_from(cells).expect("packed trace: cells fit u32"));
    let cols_u64 = cols as u64;
    unsafe {
        stream
            .launch_builder(&be.pack_col_major)
            .arg(src)
            .arg(&rows_u64)
            .arg(&cols_u64)
            .arg(&offsets_dev)
            .arg(&widths_dev)
            .arg(&mut out)
            .launch(cfg)?;
    }
    let data = stream.clone_dtoh(&out)?;
    stream.synchronize()?;
    Ok((widths, data))
}

/// [`pack_col_major_on_stream`] of a host column-major trace: the device pack,
/// for the tests that compare it with the host pack.
pub fn pack_col_major_from_host(
    col_major: &[u64],
    rows: usize,
    cols: usize,
) -> Result<(Vec<u8>, Vec<u8>)> {
    let be = backend()?;
    let stream = be.next_stream();
    let src = stream.clone_htod(col_major)?;
    pack_col_major_on_stream(&stream, be, &src, rows, cols)
}
