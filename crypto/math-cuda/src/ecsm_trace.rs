//! GPU ECSM core-chip main-column generation. ECSM rows hold dense
//! byte/halfword/carry witness data already computed by the CPU. The host
//! flattens that data into three flat blobs (byte/halfword/carry) +
//! per-row scalar arrays; the kernel is a pure column-layout splay.
//!
//! `flags_len_k[row]` bit packing:
//!   bits 0..7  len_k (Byte)
//!   bit  8     active (mu = 1; 0 = padding row)
//!
//! Buffer layouts (per row):
//!   byte_blob: 257 u64 cells:
//!     [0..32) x_r | [32..64) y_r | [64..96) k | [96..128) x_g
//!     [128..160) y_g | [160..192) x2 | [192..224) q0 | [224..257) q1
//!     Padding rows must hold P_BYTES at offset 224..256.
//!   hw_blob: 32 u64 cells: [0..16) k_sub_n | [16..32) xr_sub_p
//!   c_blob:  128 u64 cells: [0..64) c0 | [64..128) c1 (already
//!     CPU-converted from signed i64 to Goldilocks field rep)

use cudarc::driver::{LaunchConfig, PushKernelArg};

use crate::Result;
use crate::device::backend;

const BLOCK_SIZE: u32 = 64;

#[cfg(feature = "test-faults")]
pub static FAULT_ECSM_TRACE_REMAINING_UNTIL_ERR: std::sync::atomic::AtomicI64 =
    std::sync::atomic::AtomicI64::new(-1);

#[cfg(feature = "test-faults")]
fn check_ecsm_trace_fault_injection() -> Result<()> {
    use std::sync::atomic::Ordering;
    let v = FAULT_ECSM_TRACE_REMAINING_UNTIL_ERR.load(Ordering::Relaxed);
    if v < 0 {
        return Ok(());
    }
    let prev = FAULT_ECSM_TRACE_REMAINING_UNTIL_ERR.fetch_sub(1, Ordering::Relaxed);
    if prev == 1 {
        return Err(cudarc::driver::DriverError(
            cudarc::driver::sys::CUresult::CUDA_ERROR_UNKNOWN,
        ));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn generate_ecsm_trace_dev(
    num_rows: usize,
    timestamps: &[u64],
    addr_xgs: &[u64],
    addr_ks: &[u64],
    addr_xrs: &[u64],
    flags_len_k: &[u64],
    byte_blob: &[u64],
    hw_blob: &[u64],
    c_blob: &[u64],
    num_cols: usize,
) -> Result<Vec<u64>> {
    #[cfg(feature = "test-faults")]
    check_ecsm_trace_fault_injection()?;
    assert_eq!(timestamps.len(), num_rows);
    assert_eq!(addr_xgs.len(), num_rows);
    assert_eq!(addr_ks.len(), num_rows);
    assert_eq!(addr_xrs.len(), num_rows);
    assert_eq!(flags_len_k.len(), num_rows);
    assert_eq!(byte_blob.len(), num_rows * 257);
    assert_eq!(hw_blob.len(), num_rows * 32);
    assert_eq!(c_blob.len(), num_rows * 128);
    assert!(num_cols >= 427, "ecsm table needs at least 427 columns");
    assert!(
        num_rows <= u32::MAX as usize / BLOCK_SIZE as usize,
        "num_rows {num_rows} would truncate u32 grid_dim",
    );

    let be = backend()?;
    let stream = be.next_stream();
    let ts_dev = stream.clone_htod(timestamps)?;
    let axg_dev = stream.clone_htod(addr_xgs)?;
    let ak_dev = stream.clone_htod(addr_ks)?;
    let axr_dev = stream.clone_htod(addr_xrs)?;
    let flk_dev = stream.clone_htod(flags_len_k)?;
    let bb_dev = stream.clone_htod(byte_blob)?;
    let hw_dev = stream.clone_htod(hw_blob)?;
    let cb_dev = stream.clone_htod(c_blob)?;
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
            .launch_builder(&be.generate_ecsm_trace_rows)
            .arg(&num_rows_u)
            .arg(&ts_dev)
            .arg(&axg_dev)
            .arg(&ak_dev)
            .arg(&axr_dev)
            .arg(&flk_dev)
            .arg(&bb_dev)
            .arg(&hw_dev)
            .arg(&cb_dev)
            .arg(&mut table_dev)
            .arg(&num_cols_u)
            .launch(cfg)?;
    }
    let out = stream.clone_dtoh(&table_dev)?;
    stream.synchronize()?;
    Ok(out)
}
