//! GPU BYTEWISE-table main-column generation. CPU side dedupes operations
//! (HashMap merge with summed multiplicities) and pre-computes `res = a OP b`
//! before calling; kernel does the byte breakdowns and row layout.

use cudarc::driver::{LaunchConfig, PushKernelArg};

use crate::Result;
use crate::device::backend;

const BLOCK_SIZE: u32 = 256;

#[cfg(feature = "test-faults")]
pub static FAULT_BYTEWISE_TRACE_REMAINING_UNTIL_ERR: std::sync::atomic::AtomicI64 =
    std::sync::atomic::AtomicI64::new(-1);

#[cfg(feature = "test-faults")]
fn check_bytewise_trace_fault_injection() -> Result<()> {
    use std::sync::atomic::Ordering;
    let v = FAULT_BYTEWISE_TRACE_REMAINING_UNTIL_ERR.load(Ordering::Relaxed);
    if v < 0 {
        return Ok(());
    }
    let prev = FAULT_BYTEWISE_TRACE_REMAINING_UNTIL_ERR.fetch_sub(1, Ordering::Relaxed);
    if prev == 1 {
        return Err(cudarc::driver::DriverError(
            cudarc::driver::sys::CUresult::CUDA_ERROR_UNKNOWN,
        ));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn generate_bytewise_trace_dev(
    num_rows: usize,
    a_values: &[u64],
    b_values: &[u64],
    res_values: &[u64],
    ops: &[u64],
    multiplicities: &[u64],
    num_cols: usize,
) -> Result<Vec<u64>> {
    #[cfg(feature = "test-faults")]
    check_bytewise_trace_fault_injection()?;
    assert_eq!(a_values.len(), num_rows);
    assert_eq!(b_values.len(), num_rows);
    assert_eq!(res_values.len(), num_rows);
    assert_eq!(ops.len(), num_rows);
    assert_eq!(multiplicities.len(), num_rows);
    assert!(num_cols >= 26, "bytewise table needs at least 26 columns");
    assert!(
        num_rows <= u32::MAX as usize / BLOCK_SIZE as usize,
        "num_rows {num_rows} would truncate u32 grid_dim",
    );

    let be = backend()?;
    let stream = be.next_stream();
    let a_dev = stream.clone_htod(a_values)?;
    let b_dev = stream.clone_htod(b_values)?;
    let r_dev = stream.clone_htod(res_values)?;
    let op_dev = stream.clone_htod(ops)?;
    let mu_dev = stream.clone_htod(multiplicities)?;
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
            .launch_builder(&be.generate_bytewise_trace_rows)
            .arg(&num_rows_u)
            .arg(&a_dev)
            .arg(&b_dev)
            .arg(&r_dev)
            .arg(&op_dev)
            .arg(&mu_dev)
            .arg(&mut table_dev)
            .arg(&num_cols_u)
            .launch(cfg)?;
    }
    let out = stream.clone_dtoh(&table_dev)?;
    stream.synchronize()?;
    Ok(out)
}
