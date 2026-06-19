//! GPU page-table main-column generation. Mirrors `generate_page_trace`
//! in `prover/src/tables/page.rs`: produces `page_size * num_cols` u64s in
//! row-major canonical Goldilocks layout.
//!
//! The caller flattens its `FinalStateMap` HashMap into the three parallel
//! `Vec<u64>` arrays before calling — that step is cheap O(page_size) CPU
//! work. The kernel handles the actual row layout for every byte in the
//! page, which scales linearly with `page_size` (typically 2^18 = 262144).

use std::sync::Arc;

use cudarc::driver::{CudaStream, LaunchConfig, PushKernelArg};

use crate::Result;
use crate::device::backend;

const BLOCK_SIZE: u32 = 256;

/// Test-only fault injection. Same shape as `inverse.rs`'s
/// `FAULT_INVERSE_REMAINING_UNTIL_ERR`: when `test-faults` is on, setting
/// this to a finite value forces the next `generate_page_trace_dev` call
/// to return Err, exercising the prover's CPU fallback.
#[cfg(feature = "test-faults")]
pub static FAULT_PAGE_TRACE_REMAINING_UNTIL_ERR: std::sync::atomic::AtomicI64 =
    std::sync::atomic::AtomicI64::new(-1);

#[cfg(feature = "test-faults")]
fn check_page_trace_fault_injection() -> Result<()> {
    use std::sync::atomic::Ordering;
    let v = FAULT_PAGE_TRACE_REMAINING_UNTIL_ERR.load(Ordering::Relaxed);
    if v < 0 {
        return Ok(());
    }
    let new = FAULT_PAGE_TRACE_REMAINING_UNTIL_ERR.fetch_sub(1, Ordering::Relaxed);
    if new == 0 {
        return Err(cudarc::driver::DriverError(
            cudarc::driver::sys::CUresult::CUDA_ERROR_UNKNOWN,
        ));
    }
    Ok(())
}

/// Generate one page table's main columns on device. Inputs are length
/// `page_size` each; output is `page_size * num_cols` u64s row-major.
///
/// `num_cols` is the page table's `cols::NUM_COLUMNS` (currently 5, but
/// passed in so a stark-side schema bump doesn't silently desync).
pub fn generate_page_trace_dev(
    page_size: usize,
    init_values: &[u64],
    final_values: &[u64],
    final_timestamps: &[u64],
    num_cols: usize,
) -> Result<Vec<u64>> {
    #[cfg(feature = "test-faults")]
    check_page_trace_fault_injection()?;
    assert_eq!(init_values.len(), page_size);
    assert_eq!(final_values.len(), page_size);
    assert_eq!(final_timestamps.len(), page_size);
    assert!(num_cols >= 5, "page table needs at least 5 columns");
    // u32 grid bound — page_size at the current default (2^18) sits well below.
    assert!(
        page_size <= u32::MAX as usize / BLOCK_SIZE as usize,
        "page_size {page_size} would truncate u32 grid_dim",
    );

    let be = backend()?;
    let stream = be.next_stream();

    let init_dev = stream.clone_htod(init_values)?;
    let fini_dev = stream.clone_htod(final_values)?;
    let ts_dev = stream.clone_htod(final_timestamps)?;

    // SAFETY: kernel writes every row × every column.
    let mut table_dev = unsafe { stream.alloc::<u64>(page_size * num_cols) }?;

    let page_size_u = page_size as u64;
    let num_cols_u = num_cols as u64;
    let cfg = LaunchConfig {
        grid_dim: ((page_size as u32).div_ceil(BLOCK_SIZE), 1, 1),
        block_dim: (BLOCK_SIZE, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&be.generate_page_trace_rows)
            .arg(&page_size_u)
            .arg(&init_dev)
            .arg(&fini_dev)
            .arg(&ts_dev)
            .arg(&mut table_dev)
            .arg(&num_cols_u)
            .launch(cfg)?;
    }

    let out = stream.clone_dtoh(&table_dev)?;
    stream.synchronize()?;
    Ok(out)
}

/// Same as [`generate_page_trace_dev`] but on a caller-supplied stream
/// (the rest of the math-cuda layer prefers this shape; PR-6+ per-table
/// dispatchers thread their own stream so producer+consumer kernels
/// serialize naturally).
#[allow(clippy::too_many_arguments)]
pub fn generate_page_trace_dev_with_stream(
    stream: &Arc<CudaStream>,
    page_size: usize,
    init_values: &[u64],
    final_values: &[u64],
    final_timestamps: &[u64],
    num_cols: usize,
) -> Result<Vec<u64>> {
    assert_eq!(init_values.len(), page_size);
    assert_eq!(final_values.len(), page_size);
    assert_eq!(final_timestamps.len(), page_size);
    assert!(num_cols >= 5);
    assert!(
        page_size <= u32::MAX as usize / BLOCK_SIZE as usize,
        "page_size {page_size} would truncate u32 grid_dim",
    );

    let be = backend()?;

    let init_dev = stream.clone_htod(init_values)?;
    let fini_dev = stream.clone_htod(final_values)?;
    let ts_dev = stream.clone_htod(final_timestamps)?;
    let mut table_dev = unsafe { stream.alloc::<u64>(page_size * num_cols) }?;

    let page_size_u = page_size as u64;
    let num_cols_u = num_cols as u64;
    let cfg = LaunchConfig {
        grid_dim: ((page_size as u32).div_ceil(BLOCK_SIZE), 1, 1),
        block_dim: (BLOCK_SIZE, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&be.generate_page_trace_rows)
            .arg(&page_size_u)
            .arg(&init_dev)
            .arg(&fini_dev)
            .arg(&ts_dev)
            .arg(&mut table_dev)
            .arg(&num_cols_u)
            .launch(cfg)?;
    }

    let out = stream.clone_dtoh(&table_dev)?;
    stream.synchronize()?;
    Ok(out)
}
