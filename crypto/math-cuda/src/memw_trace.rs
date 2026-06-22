//! GPU MEMW (general unaligned / split-timestamp path) main-column
//! generation. Caller pre-builds seven parallel host buffers; the
//! `values`/`olds`/`old_timestamps` arrays are length `8 * num_rows`
//! (row-major 8-stride interleaving).
//!
//! `flags[row]` bit packing:
//!   bit 0: is_register
//!   bit 1: is_read (mu_read; mu_write = 1 - is_read when active)
//!   bit 2: write2
//!   bit 3: write4
//!   bit 4: write8
//!   bit 5: active (0 = padding row → all zeros)

use cudarc::driver::{LaunchConfig, PushKernelArg};

use crate::Result;
use crate::device::backend;

const BLOCK_SIZE: u32 = 256;

#[cfg(feature = "test-faults")]
pub static FAULT_MEMW_TRACE_REMAINING_UNTIL_ERR: std::sync::atomic::AtomicI64 =
    std::sync::atomic::AtomicI64::new(-1);

#[cfg(feature = "test-faults")]
fn check_memw_trace_fault_injection() -> Result<()> {
    use std::sync::atomic::Ordering;
    let v = FAULT_MEMW_TRACE_REMAINING_UNTIL_ERR.load(Ordering::Relaxed);
    if v < 0 {
        return Ok(());
    }
    let prev = FAULT_MEMW_TRACE_REMAINING_UNTIL_ERR.fetch_sub(1, Ordering::Relaxed);
    if prev == 1 {
        return Err(cudarc::driver::DriverError(
            cudarc::driver::sys::CUresult::CUDA_ERROR_UNKNOWN,
        ));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn generate_memw_trace_dev(
    num_rows: usize,
    base_addresses: &[u64],
    timestamps: &[u64],
    values: &[u64],
    olds: &[u64],
    old_timestamps: &[u64],
    flags: &[u64],
    num_cols: usize,
) -> Result<Vec<u64>> {
    #[cfg(feature = "test-faults")]
    check_memw_trace_fault_injection()?;
    assert_eq!(base_addresses.len(), num_rows);
    assert_eq!(timestamps.len(), num_rows);
    assert_eq!(values.len(), num_rows * 8);
    assert_eq!(olds.len(), num_rows * 8);
    assert_eq!(old_timestamps.len(), num_rows * 8);
    assert_eq!(flags.len(), num_rows);
    assert!(num_cols >= 49, "memw table needs at least 49 columns");
    assert!(
        num_rows <= u32::MAX as usize / BLOCK_SIZE as usize,
        "num_rows {num_rows} would truncate u32 grid_dim",
    );

    let be = backend()?;
    let stream = be.next_stream();
    let addr_dev = stream.clone_htod(base_addresses)?;
    let ts_dev = stream.clone_htod(timestamps)?;
    let values_dev = stream.clone_htod(values)?;
    let olds_dev = stream.clone_htod(olds)?;
    let ots_dev = stream.clone_htod(old_timestamps)?;
    let flags_dev = stream.clone_htod(flags)?;
    let mut table_dev = unsafe { stream.alloc::<u64>(num_rows * num_cols) }?;

    let num_rows_u = num_rows as u64;
    let num_cols_u = num_cols as u64;
    let cfg = LaunchConfig {
        grid_dim: ((num_rows as u32).div_ceil(BLOCK_SIZE), 1, 1),
        block_dim: (BLOCK_SIZE, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&be.generate_memw_trace_rows)
            .arg(&num_rows_u)
            .arg(&addr_dev)
            .arg(&ts_dev)
            .arg(&values_dev)
            .arg(&olds_dev)
            .arg(&ots_dev)
            .arg(&flags_dev)
            .arg(&mut table_dev)
            .arg(&num_cols_u)
            .launch(cfg)?;
    }
    let out = stream.clone_dtoh(&table_dev)?;
    stream.synchronize()?;
    Ok(out)
}
