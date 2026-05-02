//! LogUp aux-trace-build term-column compute on device.
//!
//! For one interaction pair (a, b):
//!   1. logup_pair_fingerprint — reads main trace columns from host buffer
//!      (H2D once per call), interprets a bytecode per pair that encodes
//!      the BusValue/Packing/LinearTerm structure, emits 2n ext3 fingerprints.
//!   2. batch_inverse_ext3_dev — reuses the existing parallel Montgomery
//!      scan on the fingerprint buffer in place.
//!   3. logup_pair_term_assembly — reads inverted fingerprints + evaluates
//!      Multiplicity descriptors (from bytecode) to emit n ext3 term values.
//!
//! The bytecode format is shared between the CPU-side serializer (in
//! crypto/stark/src/lookup.rs) and the CUDA kernels (in
//! crypto/math-cuda/kernels/logup.cu). Keep them in lock-step.

use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};

use crate::Result;
use crate::device::backend;

// Op kinds — mirror the CUDA #defines in logup.cu
pub const OP_PACK_DIRECT: u8 = 0;
pub const OP_PACK_WORD2L: u8 = 1;
pub const OP_PACK_WORD4L: u8 = 2;
pub const OP_PACK_DWORDWL: u8 = 3;
pub const OP_PACK_DWORDHHW: u8 = 4;
pub const OP_PACK_DWORDWHH: u8 = 5;
pub const OP_PACK_DWORDHL: u8 = 6;
pub const OP_PACK_DWORDBL: u8 = 7;
pub const OP_PACK_QUADHL: u8 = 8;
pub const OP_PACK_QUADWL: u8 = 9;
pub const OP_LINEAR: u8 = 10;

pub const MULT_ONE: u8 = 0;
pub const MULT_COLUMN: u8 = 1;
pub const MULT_SUM: u8 = 2;
pub const MULT_NEGATED: u8 = 3;
pub const MULT_DIFF: u8 = 4;
pub const MULT_SUM3: u8 = 5;
pub const MULT_LINEAR: u8 = 6;

/// 32-byte packed op — `#[repr(C)]` must match `FingerprintOp` in logup.cu.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct FingerprintOp {
    pub kind: u8,
    pub pad0: [u8; 3],
    pub alpha_offset: u32,
    pub start_col: u32,
    pub num_linear_terms: u32,
    pub linear_term_offset: u32,
    pub pad1: [u32; 2],
}

/// 16-byte linear term. `value` is a **canonical** Goldilocks field element
/// in `[0, p)` — the serializer handles the conversion from signed i64 or
/// unsigned u64 (including large values that exceed i64::MAX).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct LinearTerm {
    pub kind: u8, // 0 = Column, 2 = Constant
    pub pad: [u8; 3],
    pub column: u32,
    pub value: u64,
}

pub const LT_KIND_COLUMN: u8 = 0;
pub const LT_KIND_CONSTANT: u8 = 2;

/// 24-byte multiplicity descriptor.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct MultiplicityDesc {
    pub kind: u8,
    pub pad: [u8; 3],
    pub cols: [u32; 3],
    pub num_linear_terms: u32,
    pub linear_term_offset: u32,
}

impl Default for MultiplicityDesc {
    fn default() -> Self {
        Self {
            kind: MULT_ONE,
            pad: [0; 3],
            cols: [0; 3],
            num_linear_terms: 0,
            linear_term_offset: 0,
        }
    }
}

/// Device-resident main columns — hold this once per aux-build so every
/// pair reuses the same H2D copy instead of re-uploading the ~240 MB
/// main trace for every interaction pair.
pub struct DeviceMainCols {
    pub dev: CudaSlice<u64>,
    pub num_cols: usize,
    pub n: usize,
}

/// Upload the column-major main trace (`num_main_cols * n` u64s) to the
/// device once. Pair kernels then reference it via `&DeviceMainCols`.
pub fn upload_main_cols(main_cols_host: &[u64], num_main_cols: usize, n: usize)
    -> Result<DeviceMainCols>
{
    assert_eq!(main_cols_host.len(), num_main_cols * n);
    let be = backend();
    let stream = be.next_stream();
    let dev = stream.clone_htod(main_cols_host)?;
    stream.synchronize()?;
    Ok(DeviceMainCols {
        dev,
        num_cols: num_main_cols,
        n,
    })
}

/// Variant of `logup_pair_term_column` that reuses a pre-uploaded
/// `DeviceMainCols`. This is the fast path for aux-build, where 30+
/// pairs all share the same main trace.
#[allow(clippy::too_many_arguments)]
pub fn logup_pair_term_column_on_device(
    main: &DeviceMainCols,
    bus_id_a: u64,
    bus_id_b: u64,
    ops_a: &[FingerprintOp],
    ops_b: &[FingerprintOp],
    linear_terms: &[LinearTerm],
    alpha_powers: &[u64],
    z: &[u64; 3],
    mult_a: &MultiplicityDesc,
    mult_b: &MultiplicityDesc,
    negate_a: bool,
    negate_b: bool,
) -> Result<Vec<u64>> {
    let n = main.n;
    let be = backend();
    let stream = be.next_stream();

    let ops_a_dev: CudaSlice<u8> = upload_ops(&stream, ops_a)?;
    let ops_b_dev: CudaSlice<u8> = upload_ops(&stream, ops_b)?;
    let lt_dev: CudaSlice<u8> = upload_linear_terms(&stream, linear_terms)?;
    let mult_a_dev: CudaSlice<u8> = upload_mult(&stream, mult_a)?;
    let mult_b_dev: CudaSlice<u8> = upload_mult(&stream, mult_b)?;
    let alpha_dev = stream.clone_htod(alpha_powers)?;
    let z_dev = stream.clone_htod(z)?;

    let mut fp_dev = stream.alloc_zeros::<u64>(2 * n * 3)?;

    let col_stride = n as u64;
    let n_u64 = n as u64;
    let ops_a_count = ops_a.len() as u32;
    let ops_b_count = ops_b.len() as u32;

    let cfg = LaunchConfig {
        grid_dim: (((n as u32) + 255) / 256, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&be.logup_pair_fingerprint)
            .arg(&main.dev)
            .arg(&col_stride)
            .arg(&n_u64)
            .arg(&bus_id_a)
            .arg(&bus_id_b)
            .arg(&ops_a_dev)
            .arg(&ops_a_count)
            .arg(&ops_b_dev)
            .arg(&ops_b_count)
            .arg(&lt_dev)
            .arg(&alpha_dev)
            .arg(&z_dev)
            .arg(&mut fp_dev)
            .launch(cfg)?;
    }

    let inv_fp_dev = run_batch_inverse_on_device(&stream, &fp_dev, 2 * n)?;

    let mut term_dev = stream.alloc_zeros::<u64>(n * 3)?;
    let neg_a: u8 = negate_a as u8;
    let neg_b: u8 = negate_b as u8;
    unsafe {
        stream
            .launch_builder(&be.logup_pair_term_assembly)
            .arg(&inv_fp_dev)
            .arg(&main.dev)
            .arg(&col_stride)
            .arg(&n_u64)
            .arg(&lt_dev)
            .arg(&mult_a_dev)
            .arg(&mult_b_dev)
            .arg(&neg_a)
            .arg(&neg_b)
            .arg(&mut term_dev)
            .launch(cfg)?;
    }

    let out = stream.clone_dtoh(&term_dev)?;
    stream.synchronize()?;
    Ok(out)
}

/// Single-interaction variant using a shared `DeviceMainCols`.
#[allow(clippy::too_many_arguments)]
pub fn logup_single_term_column_on_device(
    main: &DeviceMainCols,
    bus_id: u64,
    ops: &[FingerprintOp],
    linear_terms: &[LinearTerm],
    alpha_powers: &[u64],
    z: &[u64; 3],
    mult: &MultiplicityDesc,
    negate: bool,
) -> Result<Vec<u64>> {
    let n = main.n;
    let be = backend();
    let stream = be.next_stream();

    let ops_dev: CudaSlice<u8> = upload_ops(&stream, ops)?;
    let lt_dev: CudaSlice<u8> = upload_linear_terms(&stream, linear_terms)?;
    let mult_dev: CudaSlice<u8> = upload_mult(&stream, mult)?;
    let alpha_dev = stream.clone_htod(alpha_powers)?;
    let z_dev = stream.clone_htod(z)?;

    let mut fp_dev = stream.alloc_zeros::<u64>(n * 3)?;

    let col_stride = n as u64;
    let n_u64 = n as u64;
    let ops_count = ops.len() as u32;

    let cfg = LaunchConfig {
        grid_dim: (((n as u32) + 255) / 256, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&be.logup_single_fingerprint)
            .arg(&main.dev)
            .arg(&col_stride)
            .arg(&n_u64)
            .arg(&bus_id)
            .arg(&ops_dev)
            .arg(&ops_count)
            .arg(&lt_dev)
            .arg(&alpha_dev)
            .arg(&z_dev)
            .arg(&mut fp_dev)
            .launch(cfg)?;
    }

    let inv_fp_dev = run_batch_inverse_on_device(&stream, &fp_dev, n)?;

    let mut term_dev = stream.alloc_zeros::<u64>(n * 3)?;
    let neg: u8 = negate as u8;
    unsafe {
        stream
            .launch_builder(&be.logup_single_term_assembly)
            .arg(&inv_fp_dev)
            .arg(&main.dev)
            .arg(&col_stride)
            .arg(&n_u64)
            .arg(&lt_dev)
            .arg(&mult_dev)
            .arg(&neg)
            .arg(&mut term_dev)
            .launch(cfg)?;
    }

    let out = stream.clone_dtoh(&term_dev)?;
    stream.synchronize()?;
    Ok(out)
}

/// Run the fingerprint + batch-inverse + term-assembly pipeline for ONE
/// interaction pair. Produces an `Vec<u64>` of size `3 * n` (ext3
/// interleaved) representing the term column.
///
/// `main_cols_host`: column-major, `num_main_cols * n` u64s.
/// `ops_a / ops_b`: serialised FingerprintOp slices for each side.
/// `linear_terms`: shared pool indexed by op.linear_term_offset +
/// multiplicity.linear_term_offset.
/// `alpha_powers`: `3 * max_bus_elements` u64 (ext3 interleaved).
/// `z`: 3 u64.
#[allow(clippy::too_many_arguments)]
pub fn logup_pair_term_column(
    main_cols_host: &[u64],
    num_main_cols: usize,
    n: usize,
    bus_id_a: u64,
    bus_id_b: u64,
    ops_a: &[FingerprintOp],
    ops_b: &[FingerprintOp],
    linear_terms: &[LinearTerm],
    alpha_powers: &[u64],
    z: &[u64; 3],
    mult_a: &MultiplicityDesc,
    mult_b: &MultiplicityDesc,
    negate_a: bool,
    negate_b: bool,
) -> Result<Vec<u64>> {
    assert_eq!(main_cols_host.len(), num_main_cols * n);

    let be = backend();
    let stream = be.next_stream();

    // H2D main cols + bytecode.
    let main_dev = stream.clone_htod(main_cols_host)?;
    let ops_a_dev: CudaSlice<u8> = upload_ops(&stream, ops_a)?;
    let ops_b_dev: CudaSlice<u8> = upload_ops(&stream, ops_b)?;
    let lt_dev: CudaSlice<u8> = upload_linear_terms(&stream, linear_terms)?;
    let mult_a_dev: CudaSlice<u8> = upload_mult(&stream, mult_a)?;
    let mult_b_dev: CudaSlice<u8> = upload_mult(&stream, mult_b)?;
    let alpha_dev = stream.clone_htod(alpha_powers)?;
    let z_dev = stream.clone_htod(z)?;

    // Fingerprint buffer: 2n ext3.
    let mut fp_dev = stream.alloc_zeros::<u64>(2 * n * 3)?;

    let col_stride = n as u64;
    let n_u64 = n as u64;
    let bus_a = bus_id_a;
    let bus_b = bus_id_b;
    let ops_a_count = ops_a.len() as u32;
    let ops_b_count = ops_b.len() as u32;

    let cfg = LaunchConfig {
        grid_dim: (((n as u32) + 255) / 256, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&be.logup_pair_fingerprint)
            .arg(&main_dev)
            .arg(&col_stride)
            .arg(&n_u64)
            .arg(&bus_a)
            .arg(&bus_b)
            .arg(&ops_a_dev)
            .arg(&ops_a_count)
            .arg(&ops_b_dev)
            .arg(&ops_b_count)
            .arg(&lt_dev)
            .arg(&alpha_dev)
            .arg(&z_dev)
            .arg(&mut fp_dev)
            .launch(cfg)?;
    }

    // Batch-invert the 2n fingerprints in place using our parallel scan.
    // The existing `batch_inverse_ext3_dev` expects a &CudaSlice<u64> and
    // returns a new Vec<u64> (host). For the fused flow we want to keep
    // the inverted fingerprints on device; reuse the lower-level ops.
    // Simplest: run it host-side (it D2H'd and we'd H2D back — wasteful).
    //
    // Better: replicate the scan-phase launches here, writing back to
    // `fp_dev`. Avoids the round-trip entirely.
    let inv_fp_dev = run_batch_inverse_on_device(&stream, &fp_dev, 2 * n)?;

    // Term assembly.
    let mut term_dev = stream.alloc_zeros::<u64>(n * 3)?;
    let neg_a: u8 = negate_a as u8;
    let neg_b: u8 = negate_b as u8;
    unsafe {
        stream
            .launch_builder(&be.logup_pair_term_assembly)
            .arg(&inv_fp_dev)
            .arg(&main_dev)
            .arg(&col_stride)
            .arg(&n_u64)
            .arg(&lt_dev)
            .arg(&mult_a_dev)
            .arg(&mult_b_dev)
            .arg(&neg_a)
            .arg(&neg_b)
            .arg(&mut term_dev)
            .launch(cfg)?;
    }

    let out = stream.clone_dtoh(&term_dev)?;
    stream.synchronize()?;
    Ok(out)
}

/// Single-interaction variant (used for the absorbed odd interaction).
#[allow(clippy::too_many_arguments)]
pub fn logup_single_term_column(
    main_cols_host: &[u64],
    num_main_cols: usize,
    n: usize,
    bus_id: u64,
    ops: &[FingerprintOp],
    linear_terms: &[LinearTerm],
    alpha_powers: &[u64],
    z: &[u64; 3],
    mult: &MultiplicityDesc,
    negate: bool,
) -> Result<Vec<u64>> {
    assert_eq!(main_cols_host.len(), num_main_cols * n);

    let be = backend();
    let stream = be.next_stream();

    let main_dev = stream.clone_htod(main_cols_host)?;
    let ops_dev: CudaSlice<u8> = upload_ops(&stream, ops)?;
    let lt_dev: CudaSlice<u8> = upload_linear_terms(&stream, linear_terms)?;
    let mult_dev: CudaSlice<u8> = upload_mult(&stream, mult)?;
    let alpha_dev = stream.clone_htod(alpha_powers)?;
    let z_dev = stream.clone_htod(z)?;

    let mut fp_dev = stream.alloc_zeros::<u64>(n * 3)?;

    let col_stride = n as u64;
    let n_u64 = n as u64;
    let ops_count = ops.len() as u32;

    let cfg = LaunchConfig {
        grid_dim: (((n as u32) + 255) / 256, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&be.logup_single_fingerprint)
            .arg(&main_dev)
            .arg(&col_stride)
            .arg(&n_u64)
            .arg(&bus_id)
            .arg(&ops_dev)
            .arg(&ops_count)
            .arg(&lt_dev)
            .arg(&alpha_dev)
            .arg(&z_dev)
            .arg(&mut fp_dev)
            .launch(cfg)?;
    }

    let inv_fp_dev = run_batch_inverse_on_device(&stream, &fp_dev, n)?;

    let mut term_dev = stream.alloc_zeros::<u64>(n * 3)?;
    let neg: u8 = negate as u8;
    unsafe {
        stream
            .launch_builder(&be.logup_single_term_assembly)
            .arg(&inv_fp_dev)
            .arg(&main_dev)
            .arg(&col_stride)
            .arg(&n_u64)
            .arg(&lt_dev)
            .arg(&mult_dev)
            .arg(&neg)
            .arg(&mut term_dev)
            .launch(cfg)?;
    }

    let out = stream.clone_dtoh(&term_dev)?;
    stream.synchronize()?;
    Ok(out)
}

// =============================================================================
// Internals: upload helpers + re-runnable batch inverse on device
// =============================================================================

fn upload_ops(
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    ops: &[FingerprintOp],
) -> Result<CudaSlice<u8>> {
    let bytes = unsafe {
        core::slice::from_raw_parts(
            ops.as_ptr() as *const u8,
            ops.len() * core::mem::size_of::<FingerprintOp>(),
        )
    };
    if bytes.is_empty() {
        // cudarc disallows zero-length allocs; use a 1-byte dummy.
        let dummy = [0u8; 1];
        return Ok(stream.clone_htod(&dummy)?);
    }
    Ok(stream.clone_htod(bytes)?)
}

fn upload_linear_terms(
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    terms: &[LinearTerm],
) -> Result<CudaSlice<u8>> {
    let bytes = unsafe {
        core::slice::from_raw_parts(
            terms.as_ptr() as *const u8,
            terms.len() * core::mem::size_of::<LinearTerm>(),
        )
    };
    if bytes.is_empty() {
        let dummy = [0u8; 1];
        return Ok(stream.clone_htod(&dummy)?);
    }
    Ok(stream.clone_htod(bytes)?)
}

fn upload_mult(
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    m: &MultiplicityDesc,
) -> Result<CudaSlice<u8>> {
    let bytes = unsafe {
        core::slice::from_raw_parts(
            m as *const MultiplicityDesc as *const u8,
            core::mem::size_of::<MultiplicityDesc>(),
        )
    };
    Ok(stream.clone_htod(bytes)?)
}

/// Inline version of parallel Montgomery batch inverse that runs entirely
/// on device without D2H'ing the scan result. Mirrors the logic in
/// `crate::inverse::batch_inverse_ext3_dev` but is duplicated here to keep
/// the fingerprint buffer on the same stream.
fn run_batch_inverse_on_device(
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    a_dev: &CudaSlice<u64>,
    n: usize,
) -> Result<CudaSlice<u64>> {
    let be = backend();
    let mut prefix_dev = stream.alloc_zeros::<u64>(n * 3)?;
    let mut suffix_dev = stream.alloc_zeros::<u64>(n * 3)?;

    let k: u32 = 256;
    let c_per_thread: u64 = ((n as u64) + (k as u64) - 1) / (k as u64);
    let mut chunk_totals = stream.alloc_zeros::<u64>((k as usize) * 3)?;
    let mut chunk_offsets = stream.alloc_zeros::<u64>((k as usize) * 3)?;
    let n_u64 = n as u64;
    let k_u64 = k as u64;

    let cfg_scan = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (k, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        stream
            .launch_builder(&be.chunk_prefix_scan_ext3)
            .arg(a_dev)
            .arg(&n_u64)
            .arg(&c_per_thread)
            .arg(&mut prefix_dev)
            .arg(&mut chunk_totals)
            .launch(cfg_scan)?;
    }
    unsafe {
        stream
            .launch_builder(&be.exclusive_scan_of_totals_ext3)
            .arg(&chunk_totals)
            .arg(&k_u64)
            .arg(&mut chunk_offsets)
            .launch(LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (1, 1, 1),
                shared_mem_bytes: 0,
            })?;
    }
    unsafe {
        stream
            .launch_builder(&be.apply_scan_offsets_ext3)
            .arg(&mut prefix_dev)
            .arg(&n_u64)
            .arg(&c_per_thread)
            .arg(&chunk_offsets)
            .launch(cfg_scan)?;
    }

    let mut suf_ct = stream.alloc_zeros::<u64>((k as usize) * 3)?;
    let mut suf_off = stream.alloc_zeros::<u64>((k as usize) * 3)?;
    unsafe {
        stream
            .launch_builder(&be.chunk_suffix_scan_ext3)
            .arg(a_dev)
            .arg(&n_u64)
            .arg(&c_per_thread)
            .arg(&mut suffix_dev)
            .arg(&mut suf_ct)
            .launch(cfg_scan)?;
    }
    unsafe {
        stream
            .launch_builder(&be.exclusive_reverse_scan_of_totals_ext3)
            .arg(&suf_ct)
            .arg(&k_u64)
            .arg(&mut suf_off)
            .launch(LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (1, 1, 1),
                shared_mem_bytes: 0,
            })?;
    }
    unsafe {
        stream
            .launch_builder(&be.apply_reverse_scan_offsets_ext3)
            .arg(&mut suffix_dev)
            .arg(&n_u64)
            .arg(&c_per_thread)
            .arg(&suf_off)
            .launch(cfg_scan)?;
    }

    // D2H last prefix element, invert on host, H2D inv_total.
    let total = {
        let last_view = prefix_dev.slice((n - 1) * 3..n * 3);
        let last_host: Vec<u64> = stream.clone_dtoh(&last_view)?;
        stream.synchronize()?;
        crate::inverse::invert_ext3_host_pub([last_host[0], last_host[1], last_host[2]])
    };
    let mut inv_total_dev = stream.alloc_zeros::<u64>(3)?;
    stream.memcpy_htod(&total, &mut inv_total_dev)?;

    let mut out_dev = stream.alloc_zeros::<u64>(n * 3)?;
    let cfg_combine = LaunchConfig {
        grid_dim: (((n as u32) + 255) / 256, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&be.batch_inverse_combine_ext3)
            .arg(&prefix_dev)
            .arg(&suffix_dev)
            .arg(&inv_total_dev)
            .arg(&n_u64)
            .arg(&mut out_dev)
            .launch(cfg_combine)?;
    }

    Ok(out_dev)
}
