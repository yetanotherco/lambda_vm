//! GPU LOAD-table main-column generation. Caller pre-bakes four parallel
//! `Vec<u64>` arrays of length `num_rows` (already padded for the trailing
//! rows), kernel lays out the 18-column row-major buffer.
//!
//! `flags[row]` bit packing:
//!   bit 0: READ2     bit 1: READ4     bit 2: READ8
//!   bit 3: SIGNED    bit 4: SIGN_BIT  bit 5: MU
//!
//! `res_bytes` is 8 u64s per row, interleaved (row 0 bytes 0..7, row 1
//! bytes 0..7, ...).

use cudarc::driver::{LaunchConfig, PushKernelArg};

use crate::Result;
use crate::device::backend;

const BLOCK_SIZE: u32 = 256;

#[cfg(feature = "test-faults")]
pub static FAULT_LOAD_TRACE_REMAINING_UNTIL_ERR: std::sync::atomic::AtomicI64 =
    std::sync::atomic::AtomicI64::new(-1);

#[cfg(feature = "test-faults")]
fn check_load_trace_fault_injection() -> Result<()> {
    use std::sync::atomic::Ordering;
    let v = FAULT_LOAD_TRACE_REMAINING_UNTIL_ERR.load(Ordering::Relaxed);
    if v < 0 {
        return Ok(());
    }
    let prev = FAULT_LOAD_TRACE_REMAINING_UNTIL_ERR.fetch_sub(1, Ordering::Relaxed);
    if prev == 1 {
        return Err(cudarc::driver::DriverError(
            cudarc::driver::sys::CUresult::CUDA_ERROR_UNKNOWN,
        ));
    }
    Ok(())
}

pub fn generate_load_trace_dev(
    num_rows: usize,
    base_addresses: &[u64],
    timestamps: &[u64],
    flags: &[u64],
    res_bytes: &[u64],
    num_cols: usize,
) -> Result<Vec<u64>> {
    #[cfg(feature = "test-faults")]
    check_load_trace_fault_injection()?;
    assert_eq!(base_addresses.len(), num_rows);
    assert_eq!(timestamps.len(), num_rows);
    assert_eq!(flags.len(), num_rows);
    assert_eq!(res_bytes.len(), 8 * num_rows);
    assert!(num_cols >= 18, "load table needs at least 18 columns");
    assert!(
        num_rows <= u32::MAX as usize / BLOCK_SIZE as usize,
        "num_rows {num_rows} would truncate u32 grid_dim",
    );

    let be = backend()?;
    let stream = be.next_stream();
    let addr_dev = stream.clone_htod(base_addresses)?;
    let ts_dev = stream.clone_htod(timestamps)?;
    let flags_dev = stream.clone_htod(flags)?;
    let res_dev = stream.clone_htod(res_bytes)?;
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
            .launch_builder(&be.generate_load_trace_rows)
            .arg(&num_rows_u)
            .arg(&addr_dev)
            .arg(&ts_dev)
            .arg(&flags_dev)
            .arg(&res_dev)
            .arg(&mut table_dev)
            .arg(&num_cols_u)
            .launch(cfg)?;
    }
    let out = stream.clone_dtoh(&table_dev)?;
    stream.synchronize()?;
    Ok(out)
}
