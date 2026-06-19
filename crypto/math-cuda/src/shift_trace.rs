//! GPU SHIFT-table main-column generation. Caller pre-builds three
//! parallel `Vec<u64>` arrays (length num_rows, padded); kernel does the
//! full aux compute (intra-limb HWSL, limb-shift one-hot, output
//! reassembly) on-device and lays out the 29-column row-major buffer.
//!
//! `flags[row]` bit packing:
//!   bit 0: direction (0 = left, 1 = right)
//!   bit 1: signed
//!   bit 2: word_instr
//!   bit 3: active (mu = 1; 0 = pure padding row → only ZBS=1)

use cudarc::driver::{LaunchConfig, PushKernelArg};

use crate::Result;
use crate::device::backend;

const BLOCK_SIZE: u32 = 256;

#[cfg(feature = "test-faults")]
pub static FAULT_SHIFT_TRACE_REMAINING_UNTIL_ERR: std::sync::atomic::AtomicI64 =
    std::sync::atomic::AtomicI64::new(-1);

#[cfg(feature = "test-faults")]
fn check_shift_trace_fault_injection() -> Result<()> {
    use std::sync::atomic::Ordering;
    let v = FAULT_SHIFT_TRACE_REMAINING_UNTIL_ERR.load(Ordering::Relaxed);
    if v < 0 {
        return Ok(());
    }
    let prev = FAULT_SHIFT_TRACE_REMAINING_UNTIL_ERR.fetch_sub(1, Ordering::Relaxed);
    if prev == 1 {
        return Err(cudarc::driver::DriverError(
            cudarc::driver::sys::CUresult::CUDA_ERROR_UNKNOWN,
        ));
    }
    Ok(())
}

pub fn generate_shift_trace_dev(
    num_rows: usize,
    in_values: &[u64],
    shift_amounts: &[u64],
    flags: &[u64],
    num_cols: usize,
) -> Result<Vec<u64>> {
    #[cfg(feature = "test-faults")]
    check_shift_trace_fault_injection()?;
    assert_eq!(in_values.len(), num_rows);
    assert_eq!(shift_amounts.len(), num_rows);
    assert_eq!(flags.len(), num_rows);
    assert!(num_cols >= 29, "shift table needs at least 29 columns");
    assert!(
        num_rows <= u32::MAX as usize / BLOCK_SIZE as usize,
        "num_rows {num_rows} would truncate u32 grid_dim",
    );

    let be = backend()?;
    let stream = be.next_stream();
    let in_dev = stream.clone_htod(in_values)?;
    let sa_dev = stream.clone_htod(shift_amounts)?;
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
            .launch_builder(&be.generate_shift_trace_rows)
            .arg(&num_rows_u)
            .arg(&in_dev)
            .arg(&sa_dev)
            .arg(&flags_dev)
            .arg(&mut table_dev)
            .arg(&num_cols_u)
            .launch(cfg)?;
    }
    let out = stream.clone_dtoh(&table_dev)?;
    stream.synchronize()?;
    Ok(out)
}
