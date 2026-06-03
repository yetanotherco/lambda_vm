//! Parallel Montgomery batch inverse on the GPU for ext3 elements, plus
//! the R3 OOD / R4 DEEP `compute-denoms + invert` convenience fn.

use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};

use crate::Result;
use crate::device::backend;

const SCAN_THREADS: u32 = 256;
const COMBINE_BLOCK: u32 = 256;

/// Parallel batch inverse over ext3 elements. `a` is 3 * n u64s
/// (interleaved). Returns a fresh Vec<u64> with 3 * n inverses.
///
/// Mirrors `FieldElement::inplace_batch_inverse` semantically; parity
/// is gated by the prove+verify round-trip in the stark test suite.
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

    // H2D input.
    let a_dev = stream.clone_htod(a)?;

    // Scratch buffers.
    let mut prefix_dev = stream.alloc_zeros::<u64>(n * 3)?;
    let mut suffix_dev = stream.alloc_zeros::<u64>(n * 3)?;

    // Chunk sizing: SCAN_THREADS threads, one chunk per thread.
    let k: u32 = SCAN_THREADS;
    let c_per_thread: u64 = (n as u64).div_ceil(k as u64);
    let mut chunk_totals = stream.alloc_zeros::<u64>((k as usize) * 3)?;
    let mut chunk_offsets = stream.alloc_zeros::<u64>((k as usize) * 3)?;
    let n_u64 = n as u64;
    let k_u64 = k as u64;

    // Phase 1: chunk prefix scan.
    let cfg_scan = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (k, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&be.chunk_prefix_scan_ext3)
            .arg(&a_dev)
            .arg(&n_u64)
            .arg(&c_per_thread)
            .arg(&mut prefix_dev)
            .arg(&mut chunk_totals)
            .launch(cfg_scan)?;
    }

    // Phase 2: exclusive scan of chunk totals (single thread).
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

    // Phase 3: apply offsets.
    unsafe {
        stream
            .launch_builder(&be.apply_scan_offsets_ext3)
            .arg(&mut prefix_dev)
            .arg(&n_u64)
            .arg(&c_per_thread)
            .arg(&chunk_offsets)
            .launch(cfg_scan)?;
    }

    // Mirror for suffix.
    let mut suffix_chunk_totals = stream.alloc_zeros::<u64>((k as usize) * 3)?;
    let mut suffix_chunk_offsets = stream.alloc_zeros::<u64>((k as usize) * 3)?;
    unsafe {
        stream
            .launch_builder(&be.chunk_suffix_scan_ext3)
            .arg(&a_dev)
            .arg(&n_u64)
            .arg(&c_per_thread)
            .arg(&mut suffix_dev)
            .arg(&mut suffix_chunk_totals)
            .launch(cfg_scan)?;
    }
    unsafe {
        stream
            .launch_builder(&be.exclusive_reverse_scan_of_totals_ext3)
            .arg(&suffix_chunk_totals)
            .arg(&k_u64)
            .arg(&mut suffix_chunk_offsets)
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
            .arg(&suffix_chunk_offsets)
            .launch(cfg_scan)?;
    }

    // Compute total = prefix[n-1], invert on host.
    let total = {
        let last_view = prefix_dev.slice((n - 1) * 3..n * 3);
        let last_host: Vec<u64> = stream.clone_dtoh(&last_view)?;
        stream.synchronize()?;
        invert_ext3_host([last_host[0], last_host[1], last_host[2]])
    };
    let mut inv_total_dev = stream.alloc_zeros::<u64>(3)?;
    stream.memcpy_htod(&total, &mut inv_total_dev)?;

    // Combine.
    let mut out_dev = stream.alloc_zeros::<u64>(n * 3)?;
    let cfg_combine = LaunchConfig {
        grid_dim: ((n as u32).div_ceil(COMBINE_BLOCK), 1, 1),
        block_dim: (COMBINE_BLOCK, 1, 1),
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

    let out = stream.clone_dtoh(&out_dev)?;
    stream.synchronize()?;
    Ok(out)
}

/// Same as [`batch_inverse_ext3`] but the input is already on device
/// (typically from `compute_denoms_ext3`). Avoids one H2D round-trip.
pub fn batch_inverse_ext3_dev(a_dev: &CudaSlice<u64>, n: usize) -> Result<Vec<u64>> {
    if n == 0 {
        return Ok(Vec::new());
    }
    let be = backend()?;
    let stream = be.next_stream();

    let mut prefix_dev = stream.alloc_zeros::<u64>(n * 3)?;
    let mut suffix_dev = stream.alloc_zeros::<u64>(n * 3)?;

    let k: u32 = SCAN_THREADS;
    let c_per_thread: u64 = (n as u64).div_ceil(k as u64);
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

    let mut suffix_chunk_totals = stream.alloc_zeros::<u64>((k as usize) * 3)?;
    let mut suffix_chunk_offsets = stream.alloc_zeros::<u64>((k as usize) * 3)?;
    unsafe {
        stream
            .launch_builder(&be.chunk_suffix_scan_ext3)
            .arg(a_dev)
            .arg(&n_u64)
            .arg(&c_per_thread)
            .arg(&mut suffix_dev)
            .arg(&mut suffix_chunk_totals)
            .launch(cfg_scan)?;
    }
    unsafe {
        stream
            .launch_builder(&be.exclusive_reverse_scan_of_totals_ext3)
            .arg(&suffix_chunk_totals)
            .arg(&k_u64)
            .arg(&mut suffix_chunk_offsets)
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
            .arg(&suffix_chunk_offsets)
            .launch(cfg_scan)?;
    }

    let total = {
        let last_view = prefix_dev.slice((n - 1) * 3..n * 3);
        let last_host: Vec<u64> = stream.clone_dtoh(&last_view)?;
        stream.synchronize()?;
        invert_ext3_host([last_host[0], last_host[1], last_host[2]])
    };
    let mut inv_total_dev = stream.alloc_zeros::<u64>(3)?;
    stream.memcpy_htod(&total, &mut inv_total_dev)?;

    let mut out_dev = stream.alloc_zeros::<u64>(n * 3)?;
    let cfg_combine = LaunchConfig {
        grid_dim: ((n as u32).div_ceil(COMBINE_BLOCK), 1, 1),
        block_dim: (COMBINE_BLOCK, 1, 1),
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

    let out = stream.clone_dtoh(&out_dev)?;
    stream.synchronize()?;
    Ok(out)
}

/// Compute `denoms[k*n + i] = x[i * stride] - z_scalars[k]` for all i, k,
/// then batch-invert in place. Fuses B.1 + B.2 to avoid an intermediate
/// D2H + H2D of the denominator array.
///
/// `x_base` is the LDE coset (base-field, at least `n * stride` u64s).
/// `z_scalars` is `k * 3` u64s (ext3 interleaved). Returns `k * n * 3`
/// u64s (the inverted denoms), flat in k-major then i-major order.
pub fn compute_and_invert_denoms_ext3(
    x_base: &[u64],
    stride: usize,
    z_scalars: &[u64],
    k_scalars: usize,
    n: usize,
) -> Result<Vec<u64>> {
    assert!(x_base.len() >= n * stride);
    assert_eq!(z_scalars.len(), k_scalars * 3);
    let total = k_scalars * n;

    let be = backend()?;
    let stream = be.next_stream();

    let x_dev = stream.clone_htod(&x_base[..n * stride])?;
    let z_dev = stream.clone_htod(z_scalars)?;
    let mut denoms_dev = stream.alloc_zeros::<u64>(total * 3)?;

    let stride_u64 = stride as u64;
    let n_u64 = n as u64;
    let k_u64 = k_scalars as u64;

    // Compute denoms.
    let cfg = LaunchConfig {
        grid_dim: ((total as u32).div_ceil(256), 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&be.compute_denoms_ext3)
            .arg(&x_dev)
            .arg(&stride_u64)
            .arg(&z_dev)
            .arg(&k_u64)
            .arg(&n_u64)
            .arg(&mut denoms_dev)
            .launch(cfg)?;
    }
    stream.synchronize()?;

    // Batch-invert in place (reuses the device buffer).
    batch_inverse_ext3_dev(&denoms_dev, total)
}

// =============================================================================
// Host-side ext3 inverse (used once, for the total of the GPU prefix product).
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
    // Fermat: a^{p-2}
    gl_pow(a, GOLDILOCKS_P as u64 - 2)
}

/// Invert one ext3 element on the host. Used once per batch inverse to
/// invert the total product; the main batch inverse work stays on GPU.
fn invert_ext3_host(x: [u64; 3]) -> [u64; 3] {
    // x = a + b*w + c*w^2 where w^3 = 2.
    // Compute x^{-1} using the extension field's norm:
    //   norm(x) = x * x_conj1 * x_conj2 (where conjugates are Frobenius images)
    // For Fp[w]/(w^3-2) over Fp, the norm lives in Fp.
    //
    // Simpler: do the full ext3 multiplication inverse via
    // classical adjugate over Fp[w].
    //
    // Use the closed-form adjugate for degree-3 extension:
    //   Let x = (a, b, c) representing a + b*w + c*w^2
    //   Then x^{-1} = (d, e, f) / N
    //   where (Newton's identities / cofactor method):
    //     d = a^2 - 2*b*c
    //     e = 2*c^2 - a*b
    //     f = b^2 - a*c
    //     N = a*d + 2*b*f + 2*c*e
    //
    // (This matches the cpu `Degree3GoldilocksExtensionField::inv`.)
    let a = x[0];
    let b = x[1];
    let c = x[2];

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

    let inv_norm = gl_inv(norm);
    [
        gl_mul(d, inv_norm),
        gl_mul(e, inv_norm),
        gl_mul(f, inv_norm),
    ]
}
