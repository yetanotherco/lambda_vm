//! GPU MUL main-column generation. Caller pre-builds five parallel `Vec<u64>`
//! arrays (length `num_rows`, padded). MUL dedup happens CPU-side
//! (HashMap-keyed `(lhs, lhs_signed, rhs, rhs_signed)` is 130 bits → wider
//! than the u128 multifield-multiplicity primitive supports), and each
//! unique row tracks separate `mu_lo`/`mu_hi`.
//!
//! `flags[row]` bit packing:
//!   bit 0: lhs_signed
//!   bit 1: rhs_signed
//!   bit 2: active (0 = padding row → all zeros)

use cudarc::driver::{LaunchConfig, PushKernelArg};

use crate::Result;
use crate::device::backend;

const BLOCK_SIZE: u32 = 256;

#[cfg(feature = "test-faults")]
pub static FAULT_MUL_TRACE_REMAINING_UNTIL_ERR: std::sync::atomic::AtomicI64 =
    std::sync::atomic::AtomicI64::new(-1);

#[cfg(feature = "test-faults")]
fn check_mul_trace_fault_injection() -> Result<()> {
    use std::sync::atomic::Ordering;
    let v = FAULT_MUL_TRACE_REMAINING_UNTIL_ERR.load(Ordering::Relaxed);
    if v < 0 {
        return Ok(());
    }
    let prev = FAULT_MUL_TRACE_REMAINING_UNTIL_ERR.fetch_sub(1, Ordering::Relaxed);
    if prev == 1 {
        return Err(cudarc::driver::DriverError(
            cudarc::driver::sys::CUresult::CUDA_ERROR_UNKNOWN,
        ));
    }
    Ok(())
}

pub fn generate_mul_trace_dev(
    num_rows: usize,
    lhs_values: &[u64],
    rhs_values: &[u64],
    flags: &[u64],
    mu_lo: &[u64],
    mu_hi: &[u64],
    num_cols: usize,
) -> Result<Vec<u64>> {
    #[cfg(feature = "test-faults")]
    check_mul_trace_fault_injection()?;
    assert_eq!(lhs_values.len(), num_rows);
    assert_eq!(rhs_values.len(), num_rows);
    assert_eq!(flags.len(), num_rows);
    assert_eq!(mu_lo.len(), num_rows);
    assert_eq!(mu_hi.len(), num_rows);
    assert!(num_cols >= 26, "mul table needs at least 26 columns");
    assert!(
        num_rows <= u32::MAX as usize / BLOCK_SIZE as usize,
        "num_rows {num_rows} would truncate u32 grid_dim",
    );

    let be = backend()?;
    let stream = be.next_stream();
    let lhs_dev = stream.clone_htod(lhs_values)?;
    let rhs_dev = stream.clone_htod(rhs_values)?;
    let flags_dev = stream.clone_htod(flags)?;
    let mu_lo_dev = stream.clone_htod(mu_lo)?;
    let mu_hi_dev = stream.clone_htod(mu_hi)?;
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
            .launch_builder(&be.generate_mul_trace_rows)
            .arg(&num_rows_u)
            .arg(&lhs_dev)
            .arg(&rhs_dev)
            .arg(&flags_dev)
            .arg(&mu_lo_dev)
            .arg(&mu_hi_dev)
            .arg(&mut table_dev)
            .arg(&num_cols_u)
            .launch(cfg)?;
    }
    let out = stream.clone_dtoh(&table_dev)?;
    stream.synchronize()?;
    Ok(out)
}
