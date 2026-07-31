//! Parallel Montgomery batch inverse on the GPU for ext3 elements.
//!
//! The kernels live in `kernels/inverse.cu` and implement a multi-block
//! 3-phase Hillis-Steele scan: each block scans its 256 elements in shmem
//! and emits a block total; the block totals are scanned recursively (the
//! same kernels applied to a smaller array); a final pass multiplies each
//! element by the cumulative offset of preceding blocks.
//!
//! Two public entry points:
//!   - `batch_inverse_ext3`: host -> host (parity-test path).
//!   - `batch_inverse_ext3_dev`: device -> device, returns a `CudaSlice<u64>`
//!     handle the caller feeds into the next kernel without a D2H+H2D.
//!
//! Plus the fused convenience `compute_and_invert_denoms_ext3_dev` for the
//! R3 OOD and R4 DEEP denominator pipelines.

use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream, LaunchConfig, PushKernelArg};

use crate::Result;
use crate::device::backend;

const BLOCK_SIZE: u32 = 256;

/// Test-only fault injection. When the `test-faults` feature is on, setting
/// this to a finite value forces the next `compute_and_invert_denoms_ext3_dev`
/// call to return Err and decrement the counter. Tests use this to exercise
/// the CPU-fallback path in `try_compute_and_invert_inv_denoms_dev`.
#[cfg(feature = "test-faults")]
pub static FAULT_INVERSE_REMAINING_UNTIL_ERR: std::sync::atomic::AtomicI64 =
    std::sync::atomic::AtomicI64::new(-1);

#[cfg(feature = "test-faults")]
fn check_inverse_fault_injection() -> Result<()> {
    use std::sync::atomic::Ordering;
    let v = FAULT_INVERSE_REMAINING_UNTIL_ERR.load(Ordering::Relaxed);
    if v < 0 {
        return Ok(());
    }
    let new = FAULT_INVERSE_REMAINING_UNTIL_ERR.fetch_sub(1, Ordering::Relaxed);
    if new == 0 {
        return Err(cudarc::driver::DriverError(
            cudarc::driver::sys::CUresult::CUDA_ERROR_UNKNOWN,
        ));
    }
    Ok(())
}

/// Host-input batch inverse. Returns a fresh `Vec<u64>` of length `3 * n`
/// containing the inverses. Used by the parity-test suite; production
/// callers should prefer `batch_inverse_ext3_dev` to avoid the D2H.
pub fn batch_inverse_ext3(a: &[u64]) -> Result<Vec<u64>> {
    assert!(a.len().is_multiple_of(3));
    let n = a.len() / 3;
    if n == 0 {
        return Ok(Vec::new());
    }
    if n == 1 {
        // Below GPU break-even (one element). Invert on host via the math
        // crate's `Fp3::inv`.
        let inv = invert_ext3_host([a[0], a[1], a[2]])?;
        return Ok(inv.to_vec());
    }

    let be = backend()?;
    let stream = be.next_stream();
    let input_dev = stream.clone_htod(a)?;
    let out_dev = batch_inverse_ext3_dev(&input_dev, n, &stream)?;
    // Result download (3 * n u64s): async D2H through the per-worker pinned
    // slab instead of a blocking pageable copy. The synchronize drains the
    // kernels and the DMA so the pending wait below is instant.
    let pending =
        crate::device::async_dtoh_via(&stream, be.pinned_staging(), &be.ctx, &out_dev, 3 * n)?;
    stream.synchronize()?;
    let mut out = vec![0u64; 3 * n];
    pending.wait_into_u64(&mut out)?;
    Ok(out)
}

/// `p^3 - 2` as little-endian u64 limbs: the Fermat exponent for inversion in
/// the Goldilocks cubic extension (`|F_{p^3}^*| = p^3 - 1`).
const EXT3_FERMAT_EXP: [u64; 3] = ext3_fermat_exponent();

const fn ext3_fermat_exponent() -> [u64; 3] {
    const P: u128 = 0xFFFF_FFFF_0000_0001;
    let p2 = P * P;
    let m0 = ((p2 as u64) as u128) * P;
    let m1 = (p2 >> 64) * P + (m0 >> 64);
    let l0 = m0 as u64;
    // p^3 mod 2^64 ends in ...0001, so subtracting 2 never borrows.
    assert!(l0 >= 2);
    [l0 - 2, m1 as u64, (m1 >> 64) as u64]
}

/// One-thread Fermat inversion of `src[n-1]` into `out[0..3]`, stream-ordered.
///
/// Unlike the host Fermat this used to call, a zero total maps silently to
/// zero instead of `Err`. Unreachable with honest inputs (LogUp/barycentric
/// denominators are nonzero w.h.p. under random Fiat-Shamir challenges);
/// callers must not rely on a zero-total error. Debug builds add a D2H+sync
/// invertibility guard (see below) that panics on a zero total so a
/// construction/kernel bug fails loudly in tests; release elides it to keep
/// the batch inverse fully stream-ordered (no per-batch host round-trip).
fn launch_invert_total(
    stream: &Arc<CudaStream>,
    be: &crate::device::Backend,
    src: &CudaSlice<u64>,
    n: usize,
    out: &mut CudaSlice<u64>,
) -> Result<()> {
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };
    let n_u64 = n as u64;
    let [e0, e1, e2] = EXT3_FERMAT_EXP;
    unsafe {
        stream
            .launch_builder(&be.invert_total_ext3)
            .arg(src)
            .arg(&n_u64)
            .arg(&e0)
            .arg(&e1)
            .arg(&e2)
            .arg(&mut *out)
            .launch(cfg)?;
    }
    // Debug-only invertibility guard. The Fermat kernel maps a zero total
    // (some denominator was zero) silently to zero, so the batch would ship
    // all-zero "inverses" instead of erroring. A valid inverse is never zero,
    // so `out == 0` unambiguously flags a zero total. Gated off release: the
    // D2H+sync would reintroduce the per-batch host block this path exists to
    // avoid, and a zero total is unreachable with honest inputs — a hit here
    // is a construction or kernel bug, which tests/CI are the place to catch.
    #[cfg(debug_assertions)]
    {
        let mut host = [0u64; 3];
        stream.memcpy_dtoh(&out.slice(0..3), &mut host)?;
        stream.synchronize()?;
        assert_ne!(
            host, [0u64; 3],
            "batch inverse: zero total has no inverse (a denominator was zero)"
        );
    }
    Ok(())
}

/// Device-input batch inverse. Allocates and returns a fresh `CudaSlice<u64>`
/// of length `3 * n` holding the inverses. Requires `n >= 1`.
///
/// Stream-ordered end to end: every launch (including the total's Fermat
/// inversion) goes on the caller's `stream`, so downstream same-stream
/// consumers need no synchronize.
pub fn batch_inverse_ext3_dev(
    input: &CudaSlice<u64>,
    n: usize,
    stream: &Arc<CudaStream>,
) -> Result<CudaSlice<u64>> {
    assert!(n >= 1, "batch_inverse_ext3_dev requires n >= 1");
    // Runtime guard (not debug_assert): a u32 grid_dim is truncated past
    // u32::MAX / BLOCK_SIZE, which would silently launch too few blocks
    // and leave a tail uninverted. Reachable on LDE size 2^23+ × multi-
    // eval-point R4. Returning Err lets the dispatcher's Err(_) => None
    // route the caller to the CPU `inplace_batch_inverse` fallback.
    if n > u32::MAX as usize / BLOCK_SIZE as usize {
        return Err(cudarc::driver::DriverError(
            cudarc::driver::sys::CUresult::CUDA_ERROR_INVALID_VALUE,
        ));
    }
    if n == 1 {
        // Single element: one-thread Fermat kernel, skipping the scan +
        // combine machinery (and any host round-trip).
        let be = backend()?;
        let mut out = unsafe { stream.alloc::<u64>(3) }?;
        launch_invert_total(stream, be, input, 1, &mut out)?;
        return Ok(out);
    }

    let be = backend()?;

    // Prefix and suffix scan scratch buffers; fully overwritten by the
    // scan kernels, so `alloc` is safe (no need for `alloc_zeros`).
    // SAFETY: the multi-block scan kernels write every output slot.
    let mut prefix = unsafe { stream.alloc::<u64>(3 * n) }?;
    let mut suffix = unsafe { stream.alloc::<u64>(3 * n) }?;

    scan_into_fwd(stream, be, input, &mut prefix, n)?;
    scan_into_rev(stream, be, input, &mut suffix, n)?;

    // total = prefix[n-1] = suffix[0]. One-thread Fermat inversion on device,
    // keeping the whole batch inverse stream-ordered (the host round-trip here
    // blocked the calling thread once per batch).
    let mut inv_total_dev = unsafe { stream.alloc::<u64>(3) }?;
    launch_invert_total(stream, be, &prefix, n, &mut inv_total_dev)?;

    // Combine: out[i] = prefix[i-1] * inv_total * suffix[i+1].
    // SAFETY: the combine kernel writes every slot before any read.
    let mut out_dev = unsafe { stream.alloc::<u64>(3 * n) }?;
    let cfg = LaunchConfig {
        grid_dim: ((n as u32).div_ceil(BLOCK_SIZE), 1, 1),
        block_dim: (BLOCK_SIZE, 1, 1),
        shared_mem_bytes: 0,
    };
    let n_u64 = n as u64;
    unsafe {
        stream
            .launch_builder(&be.batch_inverse_combine_ext3)
            .arg(&prefix)
            .arg(&suffix)
            .arg(&inv_total_dev)
            .arg(&n_u64)
            .arg(&mut out_dev)
            .launch(cfg)?;
    }
    // No terminal `stream.synchronize()`: the caller's downstream consumers
    // (e.g. `barycentric_*_on_device_with_dev_inv_denoms`,
    // `deep_composition_ext3_with_dev_parts_and_inv_denoms`) run on the
    // same stream and thus observe the combine kernel's writes via
    // CUDA's per-stream FIFO ordering.
    Ok(out_dev)
}

/// Sign convention for `compute_and_invert_denoms_ext3_dev`.
#[derive(Copy, Clone)]
pub enum DenomSign {
    /// `denoms[k*n+i] = z_scalars[k] - x[i]`. Matches CPU
    /// `barycentric_inv_denoms(z, points)` (R3 OOD).
    ZMinusX,
    /// `denoms[k*n+i] = x[i] - z_scalars[k]`. Matches CPU R4 DEEP
    /// `denoms.push(x_i - z_k)`.
    XMinusZ,
}

/// Compute `denoms[k*n + i] = sign-dependent (z, x) combination` on
/// device, then batch-invert. Returns a fresh `CudaSlice<u64>` of length
/// `3 * k_scalars * n` holding the inverted denominators. Entire pipeline
/// stays on device (no PCIe traffic beyond the small `z_scalars` upload).
pub fn compute_and_invert_denoms_ext3_dev(
    x_lde_dev: &CudaSlice<u64>,
    z_scalars_host: &[u64],
    n: usize,
    k_scalars: usize,
    sign: DenomSign,
    stream: &Arc<CudaStream>,
) -> Result<CudaSlice<u64>> {
    // Fault-injection hook lives here (not in the shared `batch_inverse_ext3_dev`)
    // so `schedule_inverse_fault(N)` targets exactly the Nth R3/R4 denominator
    // inversion the fallback test exercises — not the LogUp aux inverses that
    // also route through `batch_inverse_ext3_dev` earlier in the prove.
    #[cfg(feature = "test-faults")]
    check_inverse_fault_injection()?;
    assert_eq!(z_scalars_host.len(), k_scalars * 3);
    assert!(n >= 1 && k_scalars >= 1);

    let be = backend()?;
    let total = k_scalars
        .checked_mul(n)
        .expect("compute_and_invert_denoms_ext3_dev: k_scalars * n overflow");
    // See `batch_inverse_ext3_dev` for the rationale: runtime Err, not
    // debug_assert, so release builds also route past the silent-truncation
    // hazard via the caller's CPU fallback.
    if total > u32::MAX as usize / BLOCK_SIZE as usize {
        return Err(cudarc::driver::DriverError(
            cudarc::driver::sys::CUresult::CUDA_ERROR_INVALID_VALUE,
        ));
    }

    let z_dev = stream.clone_htod(z_scalars_host)?;
    // SAFETY: the compute_denoms_ext3 kernel writes every output slot.
    let mut denoms = unsafe { stream.alloc::<u64>(3 * total) }?;
    let n_u64 = n as u64;
    let k_u64 = k_scalars as u64;
    // Kernel `denom_sign`: 0 = DenomSign::ZMinusX, 1 = DenomSign::XMinusZ.
    let denom_sign_u64: u64 = match sign {
        DenomSign::ZMinusX => 0,
        DenomSign::XMinusZ => 1,
    };

    let cfg = LaunchConfig {
        grid_dim: ((total as u32).div_ceil(BLOCK_SIZE), 1, 1),
        block_dim: (BLOCK_SIZE, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&be.compute_denoms_ext3)
            .arg(x_lde_dev)
            .arg(&z_dev)
            .arg(&n_u64)
            .arg(&k_u64)
            .arg(&denom_sign_u64)
            .arg(&mut denoms)
            .launch(cfg)?;
    }

    batch_inverse_ext3_dev(&denoms, total, stream)
}

// =============================================================================
// Multi-block recursive scan driver
// =============================================================================

/// Recursive driver: writes `prefix_out[i] = product of input[0..=i]` for i in
/// 0..n. `input` and `prefix_out` may NOT alias for the top-level call (they
/// alias inside the recursion when scanning block totals in place).
fn scan_into_fwd(
    stream: &Arc<CudaStream>,
    be: &crate::device::Backend,
    input: &CudaSlice<u64>,
    prefix_out: &mut CudaSlice<u64>,
    n: usize,
) -> Result<()> {
    if n == 0 {
        return Ok(());
    }
    let k = (n as u32).div_ceil(BLOCK_SIZE);
    // SAFETY: phase-1 writes every block_totals slot when the kernel emits
    // the "last in block" value; partial last block also writes its total.
    let mut block_totals = unsafe { stream.alloc::<u64>(3 * k as usize) }?;
    let n_u64 = n as u64;

    let phase_cfg = LaunchConfig {
        grid_dim: (k, 1, 1),
        block_dim: (BLOCK_SIZE, 1, 1),
        shared_mem_bytes: 0,
    };

    // Phase 1: per-block inclusive scan of `input` into `prefix_out`,
    // plus per-block totals into `block_totals`.
    unsafe {
        stream
            .launch_builder(&be.block_inclusive_scan_fwd_ext3)
            .arg(input)
            .arg(&n_u64)
            .arg(&mut *prefix_out)
            .arg(&mut block_totals)
            .launch(phase_cfg)?;
    }

    if k > 1 {
        // Phase 2: recursively scan block_totals in place.
        scan_inplace_fwd(stream, be, &mut block_totals, k as usize)?;

        // Phase 3: each block reads `block_totals_scanned[blockIdx.x - 1]`
        // and multiplies into its in-block scan output.
        unsafe {
            stream
                .launch_builder(&be.apply_block_offsets_fwd_ext3)
                .arg(&mut *prefix_out)
                .arg(&n_u64)
                .arg(&block_totals)
                .launch(phase_cfg)?;
        }
    }
    Ok(())
}

/// In-place forward scan. Used by the recursion: scanning block totals
/// always reads and writes the same buffer.
fn scan_inplace_fwd(
    stream: &Arc<CudaStream>,
    be: &crate::device::Backend,
    buf: &mut CudaSlice<u64>,
    n: usize,
) -> Result<()> {
    if n <= 1 {
        return Ok(());
    }
    let k = (n as u32).div_ceil(BLOCK_SIZE);
    let mut block_totals = unsafe { stream.alloc::<u64>(3 * k as usize) }?;
    let n_u64 = n as u64;

    let phase_cfg = LaunchConfig {
        grid_dim: (k, 1, 1),
        block_dim: (BLOCK_SIZE, 1, 1),
        shared_mem_bytes: 0,
    };

    // Scratch buffer + memcpy_dtod: cudarc's `launch_builder` chains a
    // `&buf` read arg and a `&mut buf` write arg, which the borrow checker
    // rejects even though the kernel is safe in place.
    let mut scratch = unsafe { stream.alloc::<u64>(3 * n) }?;
    unsafe {
        stream
            .launch_builder(&be.block_inclusive_scan_fwd_ext3)
            .arg(&*buf)
            .arg(&n_u64)
            .arg(&mut scratch)
            .arg(&mut block_totals)
            .launch(phase_cfg)?;
    }
    // Copy scratch back into buf for the apply_block_offsets pass to read+write.
    // SAFETY: identical lengths, both on device.
    stream.memcpy_dtod(&scratch, buf)?;

    if k > 1 {
        scan_inplace_fwd(stream, be, &mut block_totals, k as usize)?;
        unsafe {
            stream
                .launch_builder(&be.apply_block_offsets_fwd_ext3)
                .arg(&mut *buf)
                .arg(&n_u64)
                .arg(&block_totals)
                .launch(phase_cfg)?;
        }
    }
    Ok(())
}

/// Mirror of `scan_into_fwd` for the suffix scan.
fn scan_into_rev(
    stream: &Arc<CudaStream>,
    be: &crate::device::Backend,
    input: &CudaSlice<u64>,
    suffix_out: &mut CudaSlice<u64>,
    n: usize,
) -> Result<()> {
    if n == 0 {
        return Ok(());
    }
    let k = (n as u32).div_ceil(BLOCK_SIZE);
    let mut block_totals = unsafe { stream.alloc::<u64>(3 * k as usize) }?;
    let n_u64 = n as u64;

    let phase_cfg = LaunchConfig {
        grid_dim: (k, 1, 1),
        block_dim: (BLOCK_SIZE, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        stream
            .launch_builder(&be.block_inclusive_scan_rev_ext3)
            .arg(input)
            .arg(&n_u64)
            .arg(&mut *suffix_out)
            .arg(&mut block_totals)
            .launch(phase_cfg)?;
    }

    if k > 1 {
        // The reverse-direction phase-2 is itself a forward inclusive scan
        // of the (already reverse-indexed) block totals: block_totals[b]
        // holds the product over the b-th REVERSE block, and we need an
        // inclusive prefix over those for phase 3's offsets.
        scan_inplace_fwd(stream, be, &mut block_totals, k as usize)?;

        unsafe {
            stream
                .launch_builder(&be.apply_block_offsets_rev_ext3)
                .arg(&mut *suffix_out)
                .arg(&n_u64)
                .arg(&block_totals)
                .launch(phase_cfg)?;
        }
    }
    Ok(())
}

// =============================================================================
// Host-side ext3 inverse (one element, used to invert the batch total).
// =============================================================================

/// Invert one ext3 element on the host via the math crate's `Fp3::inv`.
/// Used once per batch inverse to invert the total product; the main batch
/// inverse work stays on GPU. Returns a cudarc `DriverError` on zero norm
/// so the caller's `Err(_) => None` fallback path fires (instead of
/// panicking past it).
fn invert_ext3_host(x: [u64; 3]) -> Result<[u64; 3]> {
    use math::field::element::FieldElement;
    use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField;
    use math::field::goldilocks::GoldilocksField;

    type Fp = FieldElement<GoldilocksField>;
    type Fp3 = FieldElement<Degree3GoldilocksExtensionField>;

    let elem = Fp3::new([Fp::from_raw(x[0]), Fp::from_raw(x[1]), Fp::from_raw(x[2])]);
    let inv = elem.inv().map_err(|_| {
        cudarc::driver::DriverError(cudarc::driver::sys::CUresult::CUDA_ERROR_UNKNOWN)
    })?;
    Ok([
        *inv.value()[0].value(),
        *inv.value()[1].value(),
        *inv.value()[2].value(),
    ])
}
