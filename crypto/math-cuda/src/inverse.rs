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
        // Below GPU break-even (one element). Invert on host via Fermat.
        let inv = invert_ext3_host([a[0], a[1], a[2]]);
        return Ok(inv.to_vec());
    }

    let be = backend()?;
    let stream = be.next_stream();
    let input_dev = stream.clone_htod(a)?;
    let out_dev = batch_inverse_ext3_dev(&input_dev, n, &stream)?;
    let out = stream.clone_dtoh(&out_dev)?;
    stream.synchronize()?;
    Ok(out)
}

/// Device-input batch inverse. Allocates and returns a fresh `CudaSlice<u64>`
/// of length `3 * n` holding the inverses. Requires `n >= 1`.
///
/// The caller's `stream` is used for every launch and synchronised at the
/// end (so the returned slice's data is committed before this function
/// returns).
pub fn batch_inverse_ext3_dev(
    input: &CudaSlice<u64>,
    n: usize,
    stream: &Arc<CudaStream>,
) -> Result<CudaSlice<u64>> {
    assert!(n >= 1, "batch_inverse_ext3_dev requires n >= 1");
    if n == 1 {
        // Single element: D2H, host invert, H2D. Avoids running the
        // scan + combine machinery for a degenerate case.
        let host_view: Vec<u64> = stream.clone_dtoh(&input.slice(0..3))?;
        stream.synchronize()?;
        let inv = invert_ext3_host([host_view[0], host_view[1], host_view[2]]);
        let mut out = unsafe { stream.alloc::<u64>(3) }?;
        stream.memcpy_htod(&inv, &mut out)?;
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

    // total = prefix[n-1] = suffix[0]. Invert on host (one Fermat per batch).
    let last_host: Vec<u64> = stream.clone_dtoh(&prefix.slice((n - 1) * 3..n * 3))?;
    stream.synchronize()?;
    let inv_total = invert_ext3_host([last_host[0], last_host[1], last_host[2]]);
    let mut inv_total_dev = unsafe { stream.alloc::<u64>(3) }?;
    stream.memcpy_htod(&inv_total, &mut inv_total_dev)?;

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
    stream.synchronize()?;
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
    #[cfg(feature = "test-faults")]
    check_inverse_fault_injection()?;
    assert_eq!(z_scalars_host.len(), k_scalars * 3);
    assert!(n >= 1 && k_scalars >= 1);

    let be = backend()?;
    let total = k_scalars
        .checked_mul(n)
        .expect("compute_and_invert_denoms_ext3_dev: k_scalars * n overflow");

    let z_dev = stream.clone_htod(z_scalars_host)?;
    // SAFETY: the compute_denoms_ext3 kernel writes every output slot.
    let mut denoms = unsafe { stream.alloc::<u64>(3 * total) }?;
    let n_u64 = n as u64;
    let k_u64 = k_scalars as u64;
    let subtract_x_u64: u64 = match sign {
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
            .arg(&subtract_x_u64)
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

const GOLDILOCKS_P: u128 = (1u128 << 64) - (1u128 << 32) + 1;

fn gl_mul(a: u64, b: u64) -> u64 {
    let prod = (a as u128) * (b as u128);
    (prod % GOLDILOCKS_P) as u64
}

fn gl_add(a: u64, b: u64) -> u64 {
    let s = (a as u128) + (b as u128);
    (s % GOLDILOCKS_P) as u64
}

fn gl_sub(a: u64, b: u64) -> u64 {
    // Inputs from gl_mul/gl_add are always reduced; this assert defends
    // against future reuse with raw (potentially non-canonical) inputs.
    debug_assert!((b as u128) < GOLDILOCKS_P, "gl_sub: b must be canonical");
    let a128 = a as u128;
    let b128 = b as u128;
    if a128 >= b128 {
        ((a128 - b128) % GOLDILOCKS_P) as u64
    } else {
        (((GOLDILOCKS_P - b128) + a128) % GOLDILOCKS_P) as u64
    }
}

fn gl_pow(mut base: u64, mut exp: u64) -> u64 {
    let mut acc: u64 = 1;
    while exp != 0 {
        if exp & 1 != 0 {
            acc = gl_mul(acc, base);
        }
        base = gl_mul(base, base);
        exp >>= 1;
    }
    acc
}

fn gl_inv(a: u64) -> u64 {
    // Fermat: a^{p-2} (only valid for non-zero a).
    gl_pow(a, GOLDILOCKS_P as u64 - 2)
}

/// Invert one ext3 element on the host. Used once per batch inverse to
/// invert the total product; the main batch inverse work stays on GPU.
fn invert_ext3_host(x: [u64; 3]) -> [u64; 3] {
    let a = x[0];
    let b = x[1];
    let c = x[2];

    // Adjugate over Fp[w]/(w^3 - 2). Mirrors `Degree3GoldilocksExtensionField::inv`.
    let bc = gl_mul(b, c);
    let d = gl_sub(gl_mul(a, a), gl_add(bc, bc)); // a^2 - 2bc
    let cc = gl_mul(c, c);
    let ab = gl_mul(a, b);
    let e = gl_sub(gl_add(cc, cc), ab); // 2c^2 - ab
    let bb = gl_mul(b, b);
    let ac = gl_mul(a, c);
    let f = gl_sub(bb, ac); // b^2 - ac

    let ad = gl_mul(a, d);
    let bf = gl_mul(b, f);
    let ce = gl_mul(c, e);
    let two_bf = gl_add(bf, bf);
    let two_ce = gl_add(ce, ce);
    let norm = gl_add(ad, gl_add(two_bf, two_ce));

    // gl_inv(0) = 0^(p-2) = 0 mod p; without this assert we would silently
    // return [0,0,0] and poison the whole batch inverse.
    assert!(norm != 0, "invert_ext3_host: input has zero norm");
    let inv_norm = gl_inv(norm);
    [
        gl_mul(d, inv_norm),
        gl_mul(e, inv_norm),
        gl_mul(f, inv_norm),
    ]
}

#[cfg(test)]
mod tests {
    use super::invert_ext3_host;
    use math::field::element::FieldElement;
    use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField;
    use math::field::goldilocks::GoldilocksField;
    use math::field::traits::IsPrimeField;

    type Fp = FieldElement<GoldilocksField>;
    type Fp3 = FieldElement<Degree3GoldilocksExtensionField>;

    fn canon3(a: [u64; 3]) -> [u64; 3] {
        [
            GoldilocksField::canonical(&a[0]),
            GoldilocksField::canonical(&a[1]),
            GoldilocksField::canonical(&a[2]),
        ]
    }

    /// Parity: our adjugate-based ext3 inverse must match
    /// `Degree3GoldilocksExtensionField::inv`. This catches a regression if
    /// either implementation diverges from the other (e.g., adjugate sign
    /// or normalisation drift). Runs without a GPU.
    #[test]
    fn invert_ext3_host_matches_lambdaworks() {
        let cases: [[u64; 3]; 8] = [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 1],
            [2, 3, 5],
            [7, 11, 13],
            [u64::MAX - 5, 1234567, 89],
            [0xdead_beef, 0xface_feed, 0x1234_5678_9abc_def0],
        ];
        for raw in cases {
            let lw = Fp3::new([
                Fp::from_raw(raw[0]),
                Fp::from_raw(raw[1]),
                Fp::from_raw(raw[2]),
            ])
            .inv()
            .expect("non-zero norm");
            let lw_raw = canon3([
                *lw.value()[0].value(),
                *lw.value()[1].value(),
                *lw.value()[2].value(),
            ]);
            let ours = canon3(invert_ext3_host(raw));
            assert_eq!(ours, lw_raw, "mismatch for input {raw:?}");
        }
    }
}
