//! GPU DVRM main-column generation. Caller pre-builds five parallel
//! `Vec<u64>` arrays (length `num_rows`, padded); kernel computes
//! quotient/remainder + sign/abs aux per the RISC-V spec.
//!
//! Dedup happens CPU-side (HashMap-keyed `(n, d, signed)` is 129 bits —
//! wider than the u128 multifield-multiplicity primitive supports), and
//! each unique row tracks separate `mu_q`/`mu_r`.
//!
//! `flags[row]` bit packing:
//!   bit 0: signed
//!   bit 1: active (0 = padding row → all zeros)

use cudarc::driver::{LaunchConfig, PushKernelArg};

use crate::Result;
use crate::device::backend;

const BLOCK_SIZE: u32 = 256;

#[cfg(feature = "test-faults")]
pub static FAULT_DVRM_TRACE_REMAINING_UNTIL_ERR: std::sync::atomic::AtomicI64 =
    std::sync::atomic::AtomicI64::new(-1);

#[cfg(feature = "test-faults")]
fn check_dvrm_trace_fault_injection() -> Result<()> {
    use std::sync::atomic::Ordering;
    let v = FAULT_DVRM_TRACE_REMAINING_UNTIL_ERR.load(Ordering::Relaxed);
    if v < 0 {
        return Ok(());
    }
    let prev = FAULT_DVRM_TRACE_REMAINING_UNTIL_ERR.fetch_sub(1, Ordering::Relaxed);
    if prev == 1 {
        return Err(cudarc::driver::DriverError(
            cudarc::driver::sys::CUresult::CUDA_ERROR_UNKNOWN,
        ));
    }
    Ok(())
}

pub fn generate_dvrm_trace_dev(
    num_rows: usize,
    ns: &[u64],
    ds: &[u64],
    flags: &[u64],
    mu_qs: &[u64],
    mu_rs: &[u64],
    num_cols: usize,
) -> Result<Vec<u64>> {
    #[cfg(feature = "test-faults")]
    check_dvrm_trace_fault_injection()?;
    assert_eq!(ns.len(), num_rows);
    assert_eq!(ds.len(), num_rows);
    assert_eq!(flags.len(), num_rows);
    assert_eq!(mu_qs.len(), num_rows);
    assert_eq!(mu_rs.len(), num_rows);
    assert!(num_cols >= 34, "dvrm table needs at least 34 columns");
    assert!(
        num_rows <= u32::MAX as usize / BLOCK_SIZE as usize,
        "num_rows {num_rows} would truncate u32 grid_dim",
    );

    let be = backend()?;
    let stream = be.next_stream();
    let n_dev = stream.clone_htod(ns)?;
    let d_dev = stream.clone_htod(ds)?;
    let flags_dev = stream.clone_htod(flags)?;
    let mu_q_dev = stream.clone_htod(mu_qs)?;
    let mu_r_dev = stream.clone_htod(mu_rs)?;
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
            .launch_builder(&be.generate_dvrm_trace_rows)
            .arg(&num_rows_u)
            .arg(&n_dev)
            .arg(&d_dev)
            .arg(&flags_dev)
            .arg(&mu_q_dev)
            .arg(&mu_r_dev)
            .arg(&mut table_dev)
            .arg(&num_cols_u)
            .launch(cfg)?;
    }
    let out = stream.clone_dtoh(&table_dev)?;
    stream.synchronize()?;
    Ok(out)
}
