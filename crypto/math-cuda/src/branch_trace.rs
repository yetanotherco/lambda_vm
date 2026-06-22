//! GPU BRANCH main-column generation. Caller pre-builds five parallel
//! `Vec<u64>` arrays (length `num_rows`, padded). Dedup happens CPU-side
//! (HashMap-keyed `(pc, offset, register, jalr)` is 193 bits — wider than
//! the u128 multifield-multiplicity primitive supports).
//!
//! `flags[row]` bit packing:
//!   bit 0: jalr (1 = register base, 0 = pc base)
//!   bit 1: active (0 = padding row → all zeros)

use cudarc::driver::{LaunchConfig, PushKernelArg};

use crate::Result;
use crate::device::backend;

const BLOCK_SIZE: u32 = 256;

#[cfg(feature = "test-faults")]
pub static FAULT_BRANCH_TRACE_REMAINING_UNTIL_ERR: std::sync::atomic::AtomicI64 =
    std::sync::atomic::AtomicI64::new(-1);

#[cfg(feature = "test-faults")]
fn check_branch_trace_fault_injection() -> Result<()> {
    use std::sync::atomic::Ordering;
    let v = FAULT_BRANCH_TRACE_REMAINING_UNTIL_ERR.load(Ordering::Relaxed);
    if v < 0 {
        return Ok(());
    }
    let prev = FAULT_BRANCH_TRACE_REMAINING_UNTIL_ERR.fetch_sub(1, Ordering::Relaxed);
    if prev == 1 {
        return Err(cudarc::driver::DriverError(
            cudarc::driver::sys::CUresult::CUDA_ERROR_UNKNOWN,
        ));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn generate_branch_trace_dev(
    num_rows: usize,
    pcs: &[u64],
    offsets: &[u64],
    registers: &[u64],
    flags: &[u64],
    multiplicities: &[u64],
    num_cols: usize,
) -> Result<Vec<u64>> {
    #[cfg(feature = "test-faults")]
    check_branch_trace_fault_injection()?;
    assert_eq!(pcs.len(), num_rows);
    assert_eq!(offsets.len(), num_rows);
    assert_eq!(registers.len(), num_rows);
    assert_eq!(flags.len(), num_rows);
    assert_eq!(multiplicities.len(), num_rows);
    assert!(num_cols >= 14, "branch table needs at least 14 columns");
    assert!(
        num_rows <= u32::MAX as usize / BLOCK_SIZE as usize,
        "num_rows {num_rows} would truncate u32 grid_dim",
    );

    let be = backend()?;
    let stream = be.next_stream();
    let pc_dev = stream.clone_htod(pcs)?;
    let off_dev = stream.clone_htod(offsets)?;
    let reg_dev = stream.clone_htod(registers)?;
    let flags_dev = stream.clone_htod(flags)?;
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
            .launch_builder(&be.generate_branch_trace_rows)
            .arg(&num_rows_u)
            .arg(&pc_dev)
            .arg(&off_dev)
            .arg(&reg_dev)
            .arg(&flags_dev)
            .arg(&mu_dev)
            .arg(&mut table_dev)
            .arg(&num_cols_u)
            .launch(cfg)?;
    }
    let out = stream.clone_dtoh(&table_dev)?;
    stream.synchronize()?;
    Ok(out)
}
