//! GPU BITWISE-table generation. Preprocessed table; each row is a
//! function of its index. No execution-dependent input — kernel produces
//! the full 21-column row-major buffer in one launch.
//!
//! Multiplicity columns (11..20) are left at zero by the kernel; the
//! prover-side caller runs `update_multiplicities` on the host TraceTable
//! afterwards.

use cudarc::driver::{LaunchConfig, PushKernelArg};

use crate::Result;
use crate::device::backend;

const BLOCK_SIZE: u32 = 256;

/// Test-only fault injection for the bitwise generator.
#[cfg(feature = "test-faults")]
pub static FAULT_BITWISE_TRACE_REMAINING_UNTIL_ERR: std::sync::atomic::AtomicI64 =
    std::sync::atomic::AtomicI64::new(-1);

#[cfg(feature = "test-faults")]
fn check_bitwise_trace_fault_injection() -> Result<()> {
    use std::sync::atomic::Ordering;
    let v = FAULT_BITWISE_TRACE_REMAINING_UNTIL_ERR.load(Ordering::Relaxed);
    if v < 0 {
        return Ok(());
    }
    let prev = FAULT_BITWISE_TRACE_REMAINING_UNTIL_ERR.fetch_sub(1, Ordering::Relaxed);
    if prev == 1 {
        return Err(cudarc::driver::DriverError(
            cudarc::driver::sys::CUresult::CUDA_ERROR_UNKNOWN,
        ));
    }
    Ok(())
}

/// Generate the BITWISE main-column trace on device.
///
/// `num_rows` is the table's `NUM_ROWS` constant (currently 2^20 = 1_048_576).
/// `num_cols` is `cols::NUM_COLUMNS` (currently 21).
pub fn generate_bitwise_trace_dev(num_rows: usize, num_cols: usize) -> Result<Vec<u64>> {
    #[cfg(feature = "test-faults")]
    check_bitwise_trace_fault_injection()?;
    assert!(num_cols >= 11, "bitwise table needs at least 11 columns");
    assert!(
        num_rows <= u32::MAX as usize / BLOCK_SIZE as usize,
        "num_rows {num_rows} would truncate u32 grid_dim",
    );

    let be = backend()?;
    let stream = be.next_stream();
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
            .launch_builder(&be.generate_bitwise_trace_rows)
            .arg(&num_rows_u)
            .arg(&mut table_dev)
            .arg(&num_cols_u)
            .launch(cfg)?;
    }
    let out = stream.clone_dtoh(&table_dev)?;
    stream.synchronize()?;
    Ok(out)
}
