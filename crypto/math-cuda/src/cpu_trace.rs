//! GPU CPU-table main-column generation. Caller pre-builds ten parallel
//! `Vec<u64>` arrays (length `num_rows`, padded). The CPU pre-masks all
//! operational values for word/padding rows so the kernel does pure
//! row-layout splits with no branching.
//!
//! `flags[row]` bit packing (see `kernels/cpu_trace.cu` for the full map):
//!   bits  0..  7  rs1     |  bit 48  word_instr   |  bit 55  read_register1
//!   bits  8.. 15  rs2     |  bit 49  alu          |  bit 56  read_register2
//!   bits 16.. 23  rd      |  bit 50  add          |  bit 57  write_register
//!   bits 24.. 31  half_il |  bit 51  sub          |  bit 58  branch_cond
//!   bits 32.. 39  alu_fl  |  bit 52  memory       |  bit 59  pc_double_read
//!   bits 40.. 47  mem_fl  |  bit 53  branch       |  bit 60  prev_pc_ts_borrow
//!                         |  bit 54  ecall        |  bit 61  active (informational)

use cudarc::driver::{LaunchConfig, PushKernelArg};

use crate::Result;
use crate::device::backend;

const BLOCK_SIZE: u32 = 256;

#[cfg(feature = "test-faults")]
pub static FAULT_CPU_TRACE_REMAINING_UNTIL_ERR: std::sync::atomic::AtomicI64 =
    std::sync::atomic::AtomicI64::new(-1);

#[cfg(feature = "test-faults")]
fn check_cpu_trace_fault_injection() -> Result<()> {
    use std::sync::atomic::Ordering;
    let v = FAULT_CPU_TRACE_REMAINING_UNTIL_ERR.load(Ordering::Relaxed);
    if v < 0 {
        return Ok(());
    }
    let prev = FAULT_CPU_TRACE_REMAINING_UNTIL_ERR.fetch_sub(1, Ordering::Relaxed);
    if prev == 1 {
        return Err(cudarc::driver::DriverError(
            cudarc::driver::sys::CUresult::CUDA_ERROR_UNKNOWN,
        ));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn generate_cpu_trace_dev(
    num_rows: usize,
    timestamps: &[u64],
    pcs: &[u64],
    next_pcs: &[u64],
    imms: &[u64],
    rvds: &[u64],
    rv1s: &[u64],
    rv2s: &[u64],
    arg2s: &[u64],
    ress: &[u64],
    flags: &[u64],
    num_cols: usize,
) -> Result<Vec<u64>> {
    #[cfg(feature = "test-faults")]
    check_cpu_trace_fault_injection()?;
    assert_eq!(timestamps.len(), num_rows);
    assert_eq!(pcs.len(), num_rows);
    assert_eq!(next_pcs.len(), num_rows);
    assert_eq!(imms.len(), num_rows);
    assert_eq!(rvds.len(), num_rows);
    assert_eq!(rv1s.len(), num_rows);
    assert_eq!(rv2s.len(), num_rows);
    assert_eq!(arg2s.len(), num_rows);
    assert_eq!(ress.len(), num_rows);
    assert_eq!(flags.len(), num_rows);
    assert!(num_cols >= 38, "cpu table needs at least 38 columns");
    assert!(
        num_rows <= u32::MAX as usize / BLOCK_SIZE as usize,
        "num_rows {num_rows} would truncate u32 grid_dim",
    );

    let be = backend()?;
    let stream = be.next_stream();
    let ts_dev = stream.clone_htod(timestamps)?;
    let pc_dev = stream.clone_htod(pcs)?;
    let npc_dev = stream.clone_htod(next_pcs)?;
    let imm_dev = stream.clone_htod(imms)?;
    let rvd_dev = stream.clone_htod(rvds)?;
    let rv1_dev = stream.clone_htod(rv1s)?;
    let rv2_dev = stream.clone_htod(rv2s)?;
    let arg2_dev = stream.clone_htod(arg2s)?;
    let res_dev = stream.clone_htod(ress)?;
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
            .launch_builder(&be.generate_cpu_trace_rows)
            .arg(&num_rows_u)
            .arg(&ts_dev)
            .arg(&pc_dev)
            .arg(&npc_dev)
            .arg(&imm_dev)
            .arg(&rvd_dev)
            .arg(&rv1_dev)
            .arg(&rv2_dev)
            .arg(&arg2_dev)
            .arg(&res_dev)
            .arg(&flags_dev)
            .arg(&mut table_dev)
            .arg(&num_cols_u)
            .launch(cfg)?;
    }
    let out = stream.clone_dtoh(&table_dev)?;
    stream.synchronize()?;
    Ok(out)
}
