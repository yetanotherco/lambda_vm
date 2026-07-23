//! Forward and inverse NTT over Goldilocks base field. Matches the algebraic
//! contract of `math::polynomial::Polynomial::evaluate_fft` /
//! `interpolate_fft`:
//!   input  = n elements in natural order
//!   output = n elements in natural order.
//!
//! Parity is checked by `tests/ntt.rs` against the CPU implementation.

use cudarc::driver::{LaunchConfig, PushKernelArg};
use math::field::element::FieldElement;
use math::field::goldilocks::GoldilocksField;
use math::field::traits::{IsFFTField, IsField};

use crate::Result;
use crate::device::backend;

/// Host-side twiddle table: `[ω^0, ω^1, ..., ω^{n/2-1}]` where ω is the
/// primitive n-th root of unity. Exposed for `device::Backend::cached_twiddles`
/// and for direct use in tests / benches.
pub fn twiddles_forward(log_n: u64) -> Vec<u64> {
    // Smallest meaningful NTT is size 2 (log_n = 1); size-1 has nothing to
    // twiddle. The shift `1 << (log_n - 1)` underflows for log_n = 0.
    assert!(log_n >= 1, "twiddles_forward: log_n must be >= 1");
    let omega = *GoldilocksField::get_primitive_root_of_unity(log_n)
        .expect("primitive root")
        .value();
    powers_of(omega, 1usize << (log_n - 1))
}

/// Inverse twiddle table: `[ω^{-i}]` for i in [0, n/2).
pub fn twiddles_inverse(log_n: u64) -> Vec<u64> {
    assert!(log_n >= 1, "twiddles_inverse: log_n must be >= 1");
    let omega = GoldilocksField::get_primitive_root_of_unity(log_n).expect("primitive root");
    let omega_inv = FieldElement::<GoldilocksField>::inv(&omega).expect("inverse");
    powers_of(*omega_inv.value(), 1usize << (log_n - 1))
}

fn powers_of(base: u64, count: usize) -> Vec<u64> {
    let mut out = Vec::with_capacity(count);
    let mut w = 1u64;
    for _ in 0..count {
        out.push(w);
        w = GoldilocksField::mul(&w, &base);
    }
    out
}

/// Forward NTT on a slice of `n = 2^log_n` Goldilocks coefficients. Takes
/// natural-order input and returns natural-order evaluations.
pub fn forward(coeffs: &[u64]) -> Result<Vec<u64>> {
    crate::nvtx_range!("gpu:ntt_forward");
    ntt_inplace(coeffs, /*forward=*/ true)
}

/// Inverse NTT on a slice of `n = 2^log_n` Goldilocks evaluations. Takes
/// natural-order evaluations and returns natural-order coefficients. Includes
/// the 1/n scaling.
pub fn inverse(evals: &[u64]) -> Result<Vec<u64>> {
    crate::nvtx_range!("gpu:ntt_inverse");
    ntt_inplace(evals, /*forward=*/ false)
}

fn ntt_inplace(input: &[u64], forward: bool) -> Result<Vec<u64>> {
    let n = input.len();
    // Empty / size-1 has no work to do. `is_power_of_two()` returns false for
    // 0, so this branch must come before the assert to avoid panicking on
    // empty input.
    if n <= 1 {
        return Ok(input.to_vec());
    }
    assert!(n.is_power_of_two(), "ntt length must be a power of two");
    assert!(
        n <= u32::MAX as usize,
        "ntt length {n} exceeds u32 range — kernel grid would silently truncate",
    );
    let log_n = n.trailing_zeros() as u64;

    let be = backend()?;
    let stream = be.next_stream();
    crate::gpu_span!(
        &stream,
        "gpu:ntt_{}",
        if forward { "forward" } else { "inverse" }
    );

    let mut x_dev = {
        crate::nvtx_range!("h2d");
        stream.clone_htod(input)?
    };
    let tw_dev = {
        crate::nvtx_range!("twiddles");
        if forward {
            be.fwd_twiddles_for(log_n)?
        } else {
            be.inv_twiddles_for(log_n)?
        }
    };

    let n_u64 = n as u64;

    // 1. Bit-reverse: natural → bit-reversed.
    {
        crate::nvtx_range!("bit_rev");
        unsafe {
            stream
                .launch_builder(&be.bit_reverse_permute)
                .arg(&mut x_dev)
                .arg(&n_u64)
                .arg(&log_n)
                .launch(LaunchConfig::for_num_elems(n as u32))?;
        }
    }

    // 2. DIT butterfly levels. For log_n >= 8 we fuse 8 levels per kernel via
    // the shmem kernel; for very small sizes (< 256 elements) we stick with
    // the per-level kernel because the shmem block dimensions assume n ≥ 256.
    run_ntt_body(stream.as_ref(), &mut x_dev, tw_dev.as_ref(), n_u64, log_n)?;

    // 3. For iNTT, multiply by 1/n.
    if !forward {
        let n_fe = FieldElement::<GoldilocksField>::from(n as u64);
        let inv_n = *n_fe.inv().expect("n is non-zero").value();
        unsafe {
            stream
                .launch_builder(&be.scalar_mul)
                .arg(&mut x_dev)
                .arg(&inv_n)
                .arg(&n_u64)
                .launch(LaunchConfig::for_num_elems(n as u32))?;
        }
    }

    let out = {
        crate::nvtx_range!("d2h");
        stream.clone_dtoh(&x_dev)?
    };
    {
        crate::nvtx_range!("sync");
        crate::timing::timed_sync(
            &stream,
            if forward {
                "gpu:ntt_forward"
            } else {
                "gpu:ntt_inverse"
            },
        )?;
    }
    Ok(out)
}

/// Run the butterfly body of a bit-reversed-input DIT NTT. Split out so the
/// LDE orchestrator can reuse it on the same device buffer.
pub(crate) fn run_ntt_body(
    stream: &cudarc::driver::CudaStream,
    x_dev: &mut cudarc::driver::CudaSlice<u64>,
    tw_dev: &cudarc::driver::CudaSlice<u64>,
    n: u64,
    log_n: u64,
) -> Result<()> {
    crate::nvtx_range!("ntt");
    let be = backend()?;
    // Levels 0..min(log_n, 8): one shmem-fused launch. Loads are fully
    // coalesced (base_step=0 → `row = tid`) and 8 butterfly rounds stay on
    // chip. This is the big DRAM-bandwidth win.
    let fused = core::cmp::min(log_n, 8);
    if fused >= 8 {
        let grid_x = (n / 256) as u32;
        let cfg = LaunchConfig {
            grid_dim: (grid_x, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let base_step = 0u64;
        unsafe {
            stream
                .launch_builder(&be.ntt_dit_8_levels)
                .arg(&mut *x_dev)
                .arg(tw_dev)
                .arg(&n)
                .arg(&log_n)
                .arg(&base_step)
                .launch(cfg)?;
        }
    } else {
        // Sub-256-element NTT. Use per-level.
        let half_cfg = LaunchConfig::for_num_elems((n / 2) as u32);
        for level in 0..fused {
            unsafe {
                stream
                    .launch_builder(&be.ntt_dit_level)
                    .arg(&mut *x_dev)
                    .arg(tw_dev)
                    .arg(&n)
                    .arg(&log_n)
                    .arg(&level)
                    .launch(half_cfg)?;
            }
        }
    }

    // Levels 8..log_n: per-level kernels. Loads are fully coalesced in the
    // per-level path; switching to fused-with-row-remap at base_step>0 tanks
    // DRAM throughput enough to wipe out the launch savings.
    let half_cfg = LaunchConfig::for_num_elems((n / 2) as u32);
    for level in fused..log_n {
        unsafe {
            stream
                .launch_builder(&be.ntt_dit_level)
                .arg(&mut *x_dev)
                .arg(tw_dev)
                .arg(&n)
                .arg(&log_n)
                .arg(&level)
                .launch(half_cfg)?;
        }
    }
    Ok(())
}

/// Pointwise multiply: `x[i] *= w[i]`.
pub fn pointwise_mul(x: &[u64], w: &[u64]) -> Result<Vec<u64>> {
    crate::nvtx_range!("gpu:pointwise_mul");
    assert_eq!(x.len(), w.len());
    let n = x.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    let be = backend()?;
    let stream = be.next_stream();
    crate::gpu_span!(&stream, "gpu:pointwise_mul");

    let mut x_dev = stream.clone_htod(x)?;
    let w_dev = stream.clone_htod(w)?;

    let n_u64 = n as u64;
    unsafe {
        stream
            .launch_builder(&be.pointwise_mul)
            .arg(&mut x_dev)
            .arg(&w_dev)
            .arg(&n_u64)
            .launch(LaunchConfig::for_num_elems(n as u32))?;
    }

    let out = stream.clone_dtoh(&x_dev)?;
    crate::timing::timed_sync(&stream, "gpu:pointwise_mul")?;
    Ok(out)
}
