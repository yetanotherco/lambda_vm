//! GPU DECODE-table main-column generation. Mirrors `generate_decode_trace`
//! in `prover/src/tables/decode.rs`: produces `num_rows * num_cols` u64s in
//! row-major canonical Goldilocks layout. Multiplicity (MU) column is left
//! at zero; the prover-crate caller runs `update_multiplicities` on the
//! host-side TraceTable as before.
//!
//! The caller pre-fills three parallel u64 arrays of length `num_rows`:
//! actual instruction entries, then the CPU-padding row, then any trailing
//! padding rows. Pre-bake on CPU is cheap (~num_rows u64 writes) vs the
//! row-layout kernel that does 6 u64 writes per row.

use cudarc::driver::{LaunchConfig, PushKernelArg};

use crate::Result;
use crate::device::backend;

const BLOCK_SIZE: u32 = 256;

/// Test-only fault injection. Same shape as `page_trace.rs`.
#[cfg(feature = "test-faults")]
pub static FAULT_DECODE_TRACE_REMAINING_UNTIL_ERR: std::sync::atomic::AtomicI64 =
    std::sync::atomic::AtomicI64::new(-1);

#[cfg(feature = "test-faults")]
fn check_decode_trace_fault_injection() -> Result<()> {
    use std::sync::atomic::Ordering;
    let v = FAULT_DECODE_TRACE_REMAINING_UNTIL_ERR.load(Ordering::Relaxed);
    if v < 0 {
        return Ok(());
    }
    let new = FAULT_DECODE_TRACE_REMAINING_UNTIL_ERR.fetch_sub(1, Ordering::Relaxed);
    if new == 0 {
        return Err(cudarc::driver::DriverError(
            cudarc::driver::sys::CUresult::CUDA_ERROR_UNKNOWN,
        ));
    }
    Ok(())
}

/// Generate the DECODE table's main columns on device.
///
/// `num_cols` is the table's `cols::NUM_COLUMNS` (currently 6).
pub fn generate_decode_trace_dev(
    num_rows: usize,
    pcs: &[u64],
    packed_decodes: &[u64],
    imms: &[u64],
    num_cols: usize,
) -> Result<Vec<u64>> {
    #[cfg(feature = "test-faults")]
    check_decode_trace_fault_injection()?;
    assert_eq!(pcs.len(), num_rows);
    assert_eq!(packed_decodes.len(), num_rows);
    assert_eq!(imms.len(), num_rows);
    assert!(num_cols >= 6, "decode table needs at least 6 columns");
    assert!(
        num_rows <= u32::MAX as usize / BLOCK_SIZE as usize,
        "num_rows {num_rows} would truncate u32 grid_dim",
    );

    let be = backend()?;
    let stream = be.next_stream();

    let pcs_dev = stream.clone_htod(pcs)?;
    let packed_dev = stream.clone_htod(packed_decodes)?;
    let imms_dev = stream.clone_htod(imms)?;

    // SAFETY: kernel writes every column for every row.
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
            .launch_builder(&be.generate_decode_trace_rows)
            .arg(&num_rows_u)
            .arg(&pcs_dev)
            .arg(&packed_dev)
            .arg(&imms_dev)
            .arg(&mut table_dev)
            .arg(&num_cols_u)
            .launch(cfg)?;
    }

    let out = stream.clone_dtoh(&table_dev)?;
    stream.synchronize()?;
    Ok(out)
}
