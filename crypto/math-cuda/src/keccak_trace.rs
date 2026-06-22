//! GPU KECCAK core-chip main-column generation. Caller pre-builds five
//! parallel host buffers; `inputs`/`outputs` are length `25 * num_rows`
//! (row-major 25-stride interleaving — one u64 per lane).
//!
//! `flags[row]` bit packing:
//!   bit 0: active (0 = padding row → only state_ptr[lane][0] = 8*lane_idx)

use cudarc::driver::{LaunchConfig, PushKernelArg};

use crate::Result;
use crate::device::backend;

const BLOCK_SIZE: u32 = 64;

#[cfg(feature = "test-faults")]
pub static FAULT_KECCAK_TRACE_REMAINING_UNTIL_ERR: std::sync::atomic::AtomicI64 =
    std::sync::atomic::AtomicI64::new(-1);

#[cfg(feature = "test-faults")]
fn check_keccak_trace_fault_injection() -> Result<()> {
    use std::sync::atomic::Ordering;
    let v = FAULT_KECCAK_TRACE_REMAINING_UNTIL_ERR.load(Ordering::Relaxed);
    if v < 0 {
        return Ok(());
    }
    let prev = FAULT_KECCAK_TRACE_REMAINING_UNTIL_ERR.fetch_sub(1, Ordering::Relaxed);
    if prev == 1 {
        return Err(cudarc::driver::DriverError(
            cudarc::driver::sys::CUresult::CUDA_ERROR_UNKNOWN,
        ));
    }
    Ok(())
}

pub fn generate_keccak_trace_dev(
    num_rows: usize,
    timestamps: &[u64],
    state_addrs: &[u64],
    inputs: &[u64],
    outputs: &[u64],
    flags: &[u64],
    num_cols: usize,
) -> Result<Vec<u64>> {
    #[cfg(feature = "test-faults")]
    check_keccak_trace_fault_injection()?;
    assert_eq!(timestamps.len(), num_rows);
    assert_eq!(state_addrs.len(), num_rows);
    assert_eq!(inputs.len(), num_rows * 25);
    assert_eq!(outputs.len(), num_rows * 25);
    assert_eq!(flags.len(), num_rows);
    assert!(num_cols >= 511, "keccak table needs at least 511 columns");
    assert!(
        num_rows <= u32::MAX as usize / BLOCK_SIZE as usize,
        "num_rows {num_rows} would truncate u32 grid_dim",
    );

    let be = backend()?;
    let stream = be.next_stream();
    let ts_dev = stream.clone_htod(timestamps)?;
    let addr_dev = stream.clone_htod(state_addrs)?;
    let in_dev = stream.clone_htod(inputs)?;
    let out_dev = stream.clone_htod(outputs)?;
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
            .launch_builder(&be.generate_keccak_trace_rows)
            .arg(&num_rows_u)
            .arg(&ts_dev)
            .arg(&addr_dev)
            .arg(&in_dev)
            .arg(&out_dev)
            .arg(&flags_dev)
            .arg(&mut table_dev)
            .arg(&num_cols_u)
            .launch(cfg)?;
    }
    let out = stream.clone_dtoh(&table_dev)?;
    stream.synchronize()?;
    Ok(out)
}
