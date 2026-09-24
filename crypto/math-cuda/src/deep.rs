//! R4 deep-composition polynomial evaluations on GPU.
//!
//! Mirrors `compute_deep_composition_poly_evaluations` in
//! `crypto/stark/src/prover.rs`. Accepts the main/aux LDEs as device
//! handles (populated by the R1 fused path in `LDETraceTable`) and
//! takes every other tensor (composition parts LDE, OOD evals,
//! gammas, inv-denoms) from host. Returns a `Vec<u64>` of
//! `domain_size * 3` u64s, ext3 interleaved (ready to `transmute` to
//! `FieldElement<Ext3>` when the caller promises layout compatibility).

use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream, LaunchConfig, PushKernelArg};

use math::field::element::FieldElement;
use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField;
use math::field::goldilocks::GoldilocksField;

use crate::Result;
use crate::device::backend;
use crate::lde::{GpuLdeBase, GpuLdeExt3};

type Fp3 = FieldElement<Degree3GoldilocksExtensionField>;

fn fp3_from_raw(raw: &[u64]) -> Fp3 {
    Fp3::new([
        FieldElement::<GoldilocksField>::from_raw(raw[0]),
        FieldElement::<GoldilocksField>::from_raw(raw[1]),
        FieldElement::<GoldilocksField>::from_raw(raw[2]),
    ])
}

/// Fold the per-proof OOD constants the kernel subtracts once per eval point
/// instead of once per column: returns `(1 + num_eval_points) * 3` u64s,
/// ext3 interleaved, laid out as
///   `[K_h, ood_compressed[0], ..., ood_compressed[num_eval_points - 1]]`
/// with `K_h = Σ_j gammas_h[j] * h_ood[j]` and
/// `ood_compressed[k] = Σ_j gammas_tr[j][k] * trace_ood[j][k]` (the same
/// compression the CPU R4 loop in `crypto/stark/src/prover.rs` applies).
fn fold_ood_constants(
    h_ood: &[u64],
    trace_ood: &[u64],
    gammas_h: &[u64],
    gammas_tr: &[u64],
    num_eval_points: usize,
) -> Vec<u64> {
    let mut folded = vec![Fp3::zero(); 1 + num_eval_points];
    for (g, o) in gammas_h.chunks_exact(3).zip(h_ood.chunks_exact(3)) {
        folded[0] += fp3_from_raw(g) * fp3_from_raw(o);
    }
    if num_eval_points > 0 {
        for (idx, (g, o)) in gammas_tr
            .chunks_exact(3)
            .zip(trace_ood.chunks_exact(3))
            .enumerate()
        {
            folded[1 + idx % num_eval_points] += fp3_from_raw(g) * fp3_from_raw(o);
        }
    }
    folded
        .iter()
        .flat_map(|e| e.value().iter().map(|c| *c.value()))
        .collect()
}

/// Compute deep-composition evaluations on device.
///
/// `num_eval_points = trace_terms_gammas_interleaved.len() / ((num_main +
/// num_aux) * 3)`. The caller is responsible for packing each Vec<ext3>
/// into interleaved u64 slices (`[a0, a1, a2, b0, b1, b2, ...]`).
#[allow(clippy::too_many_arguments)]
pub fn deep_composition_ext3(
    main_lde: &GpuLdeBase,
    aux_lde: Option<&GpuLdeExt3>,
    // Host-side inputs (H2D'd internally)
    h_parts_deinterleaved: &[u64], // num_parts * 3 * lde_stride u64
    h_ood: &[u64],                 // num_parts * 3
    trace_ood: &[u64],             // num_total_cols * num_eval_points * 3
    gammas_h: &[u64],              // num_parts * 3
    gammas_tr: &[u64],             // num_total_cols * num_eval_points * 3
    inv_h: &[u64],                 // domain_size * 3
    inv_t: &[u64],                 // num_eval_points * domain_size * 3
    // Shape params
    num_parts: usize,
    num_main: usize,
    num_aux: usize,
    num_eval_points: usize,
    row_stride: usize,
    domain_size: usize,
) -> Result<Vec<u64>> {
    #[cfg(feature = "test-faults")]
    crate::faults::check_sticky(&crate::faults::FAULT_DEEP_STICKY)?;
    let be = backend()?;
    let stream = be.next_stream();
    deep_composition_ext3_impl(
        &stream,
        main_lde,
        aux_lde,
        None,
        h_parts_deinterleaved,
        h_ood,
        trace_ood,
        gammas_h,
        gammas_tr,
        inv_h,
        inv_t,
        num_parts,
        num_main,
        num_aux,
        num_eval_points,
        row_stride,
        domain_size,
    )
}

/// Same as [`deep_composition_ext3`] but reads the composition-parts LDE
/// from a device handle (`GpuLdeExt3`) populated by the R2 fused path,
/// skipping the `num_parts * 3 * lde_size * 8` byte H2D of
/// `h_parts_deinterleaved`.
#[allow(clippy::too_many_arguments)]
pub fn deep_composition_ext3_with_dev_parts(
    main_lde: &GpuLdeBase,
    aux_lde: Option<&GpuLdeExt3>,
    h_parts_dev: &GpuLdeExt3,
    h_ood: &[u64],
    trace_ood: &[u64],
    gammas_h: &[u64],
    gammas_tr: &[u64],
    inv_h: &[u64],
    inv_t: &[u64],
    num_parts: usize,
    num_main: usize,
    num_aux: usize,
    num_eval_points: usize,
    row_stride: usize,
    domain_size: usize,
) -> Result<Vec<u64>> {
    #[cfg(feature = "test-faults")]
    crate::faults::check_sticky(&crate::faults::FAULT_DEEP_STICKY)?;
    let be = backend()?;
    let stream = be.next_stream();
    deep_composition_ext3_impl(
        &stream,
        main_lde,
        aux_lde,
        Some(h_parts_dev),
        &[],
        h_ood,
        trace_ood,
        gammas_h,
        gammas_tr,
        inv_h,
        inv_t,
        num_parts,
        num_main,
        num_aux,
        num_eval_points,
        row_stride,
        domain_size,
    )
}

/// Fully device-resident R4 DEEP path: parts LDE and inverse denominators
/// both arrive as device handles, the caller threads its own stream
/// through so the inv_denoms producer
/// (`compute_and_invert_denoms_ext3_dev`) and this kernel run on the same
/// stream (no cross-stream race). H2Ds only the small OOD/gamma scalars.
///
/// `inv_denoms_dev` is `3 * (1 + num_eval_points) * domain_size` u64s:
/// the first `3 * domain_size` u64s are `inv_h` (H-term denominators),
/// followed by `num_eval_points` blocks of `3 * domain_size` for the
/// trace terms. Same layout `compute_and_invert_denoms_ext3_dev`
/// produces when called with `z_scalars = [z_power, z_shifted[0..]]`.
#[allow(clippy::too_many_arguments)]
fn deep_fully_resident_launch(
    stream: &Arc<CudaStream>,
    main_lde: &GpuLdeBase,
    aux_lde: Option<&GpuLdeExt3>,
    h_parts_dev: &GpuLdeExt3,
    inv_denoms_dev: &CudaSlice<u64>,
    h_ood: &[u64],
    trace_ood: &[u64],
    gammas_h: &[u64],
    gammas_tr: &[u64],
    num_parts: usize,
    num_main: usize,
    num_aux: usize,
    num_eval_points: usize,
    row_stride: usize,
    domain_size: usize,
) -> Result<CudaSlice<u64>> {
    main_lde.wait_ready_on(stream)?;
    if let Some(aux) = aux_lde {
        aux.wait_ready_on(stream)?;
    }
    h_parts_dev.wait_ready_on(stream)?;
    assert_eq!(main_lde.m, num_main);
    assert_eq!(h_parts_dev.m, num_parts);
    assert_eq!(h_parts_dev.lde_size, main_lde.lde_size);
    if let Some(a) = aux_lde {
        assert_eq!(a.m, num_aux);
        assert_eq!(a.lde_size, main_lde.lde_size);
    } else {
        assert_eq!(num_aux, 0);
    }
    assert_eq!(h_ood.len(), num_parts * 3);
    let num_total_cols = num_main + num_aux;
    assert_eq!(trace_ood.len(), num_total_cols * num_eval_points * 3);
    assert_eq!(gammas_h.len(), num_parts * 3);
    assert_eq!(gammas_tr.len(), num_total_cols * num_eval_points * 3);

    let ext3_size = domain_size
        .checked_mul(3)
        .expect("deep composition: domain_size * 3 overflow");
    let expected_inv_denoms = ext3_size
        .checked_mul(1 + num_eval_points)
        .expect("deep composition: inv_denoms length overflow");
    assert_eq!(inv_denoms_dev.len(), expected_inv_denoms);

    if domain_size > 0 {
        let max_row = (domain_size - 1)
            .checked_mul(row_stride)
            .expect("deep composition: (domain_size - 1) * row_stride overflow");
        assert!(
            max_row < main_lde.lde_size,
            "deep composition: kernel row {max_row} out of LDE stride {}",
            main_lde.lde_size
        );
    }

    let be = backend()?;

    // H2D only the small scalars on the caller's stream.
    let ood_folded = fold_ood_constants(h_ood, trace_ood, gammas_h, gammas_tr, num_eval_points);
    let (ood_folded_dev, gammas_h_dev, gammas_tr_dev) = (
        stream.clone_htod(&ood_folded)?,
        stream.clone_htod(gammas_h)?,
        stream.clone_htod(gammas_tr)?,
    );

    // Slice the inv_denoms buffer into the H-term and trace-term views.
    let inv_h_view = inv_denoms_dev.slice(0..ext3_size);
    let inv_t_view = inv_denoms_dev.slice(ext3_size..expected_inv_denoms);

    // SAFETY: every output slot is written by the kernel.
    let mut deep_out = unsafe { stream.alloc::<u64>(domain_size * 3) }?;

    let dummy_aux;
    let aux_slice = if let Some(a) = aux_lde {
        a.buf.as_ref()
    } else {
        dummy_aux = stream.alloc_zeros::<u64>(1)?;
        &dummy_aux
    };

    let lde_stride = main_lde.lde_size as u64;
    let num_main_u = num_main as u64;
    let num_aux_u = num_aux as u64;
    let num_parts_u = num_parts as u64;
    let num_eval_points_u = num_eval_points as u64;
    let row_stride_u = row_stride as u64;
    let domain_size_u = domain_size as u64;

    let grid = (domain_size as u32).div_ceil(128);
    let cfg = LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&be.deep_composition_ext3_row)
            .arg(main_lde.buf.as_ref())
            .arg(aux_slice)
            .arg(h_parts_dev.buf.as_ref())
            .arg(&lde_stride)
            .arg(&num_main_u)
            .arg(&num_aux_u)
            .arg(&num_parts_u)
            .arg(&num_eval_points_u)
            .arg(&row_stride_u)
            .arg(&domain_size_u)
            .arg(&ood_folded_dev)
            .arg(&gammas_h_dev)
            .arg(&gammas_tr_dev)
            .arg(&inv_h_view)
            .arg(&inv_t_view)
            .arg(&mut deep_out)
            .launch(cfg)?;
    }

    Ok(deep_out)
}

/// Fully-resident DEEP composition: every large input is a device handle; the
/// codeword is D2H'd through the per-worker pinned slab.
#[allow(clippy::too_many_arguments)]
pub fn deep_composition_ext3_with_dev_parts_and_inv_denoms(
    stream: &Arc<CudaStream>,
    main_lde: &GpuLdeBase,
    aux_lde: Option<&GpuLdeExt3>,
    h_parts_dev: &GpuLdeExt3,
    inv_denoms_dev: &CudaSlice<u64>,
    h_ood: &[u64],
    trace_ood: &[u64],
    gammas_h: &[u64],
    gammas_tr: &[u64],
    num_parts: usize,
    num_main: usize,
    num_aux: usize,
    num_eval_points: usize,
    row_stride: usize,
    domain_size: usize,
) -> Result<Vec<u64>> {
    #[cfg(feature = "test-faults")]
    crate::faults::check_sticky(&crate::faults::FAULT_DEEP_STICKY)?;
    let deep_out = deep_fully_resident_launch(
        stream,
        main_lde,
        aux_lde,
        h_parts_dev,
        inv_denoms_dev,
        h_ood,
        trace_ood,
        gammas_h,
        gammas_tr,
        num_parts,
        num_main,
        num_aux,
        num_eval_points,
        row_stride,
        domain_size,
    )?;
    let be = backend()?;
    // DEEP output (domain_size * 3 u64s, ~50 MB): async D2H through the
    // per-worker pinned slab instead of a blocking pageable copy.
    let pending = crate::device::async_dtoh_via(
        stream,
        be.pinned_staging(),
        &be.ctx,
        &deep_out,
        domain_size * 3,
    )?;
    stream.synchronize()?;
    let mut out = vec![0u64; domain_size * 3];
    pending.wait_into_u64(&mut out)?;
    Ok(out)
}

/// The DEEP codeword resident on device in FRI (bit-reversed) order, with the
/// stream that produced it.
pub struct GpuDeepCodeword {
    pub(crate) buf: CudaSlice<u64>,
    pub n: usize,
    pub(crate) stream: Arc<CudaStream>,
}

/// [`deep_composition_ext3_with_dev_parts_and_inv_denoms`] keeping the
/// codeword on device, already bit-reverse-permuted into FRI order — the
/// exact input [`crate::fri::FriCommitState::new_dev`] consumes. No D2H.
#[allow(clippy::too_many_arguments)]
pub fn deep_composition_ext3_fully_resident_keep(
    stream: &Arc<CudaStream>,
    main_lde: &GpuLdeBase,
    aux_lde: Option<&GpuLdeExt3>,
    h_parts_dev: &GpuLdeExt3,
    inv_denoms_dev: &CudaSlice<u64>,
    h_ood: &[u64],
    trace_ood: &[u64],
    gammas_h: &[u64],
    gammas_tr: &[u64],
    num_parts: usize,
    num_main: usize,
    num_aux: usize,
    num_eval_points: usize,
    row_stride: usize,
    domain_size: usize,
) -> Result<GpuDeepCodeword> {
    #[cfg(feature = "test-faults")]
    crate::faults::check_sticky(&crate::faults::FAULT_DEEP_STICKY)?;
    assert!(
        domain_size.is_power_of_two() && domain_size >= 2,
        "bit-reverse needs a power-of-two codeword"
    );
    let deep_out = deep_fully_resident_launch(
        stream,
        main_lde,
        aux_lde,
        h_parts_dev,
        inv_denoms_dev,
        h_ood,
        trace_ood,
        gammas_h,
        gammas_tr,
        num_parts,
        num_main,
        num_aux,
        num_eval_points,
        row_stride,
        domain_size,
    )?;
    let be = backend()?;
    // SAFETY: every element is written by the permutation kernel below.
    let mut reversed = unsafe { stream.alloc::<u64>(domain_size * 3) }?;
    let log_n = domain_size.trailing_zeros();
    let n_u64 = domain_size as u64;
    let grid = (domain_size as u32).div_ceil(128).max(1);
    let cfg = LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&be.bit_reverse_ext3_kernel)
            .arg(&deep_out)
            .arg(&mut reversed)
            .arg(&n_u64)
            .arg(&log_n)
            .launch(cfg)?;
    }
    Ok(GpuDeepCodeword {
        buf: reversed,
        n: domain_size,
        stream: stream.clone(),
    })
}

#[allow(clippy::too_many_arguments)]
fn deep_composition_ext3_impl(
    stream: &Arc<CudaStream>,
    main_lde: &GpuLdeBase,
    aux_lde: Option<&GpuLdeExt3>,
    h_parts_dev: Option<&GpuLdeExt3>,
    h_parts_host: &[u64],
    h_ood: &[u64],
    trace_ood: &[u64],
    gammas_h: &[u64],
    gammas_tr: &[u64],
    inv_h: &[u64],
    inv_t: &[u64],
    num_parts: usize,
    num_main: usize,
    num_aux: usize,
    num_eval_points: usize,
    row_stride: usize,
    domain_size: usize,
) -> Result<Vec<u64>> {
    main_lde.wait_ready_on(stream)?;
    if let Some(aux) = aux_lde {
        aux.wait_ready_on(stream)?;
    }
    if let Some(parts) = h_parts_dev {
        parts.wait_ready_on(stream)?;
    }
    assert_eq!(main_lde.m, num_main);
    if let Some(a) = aux_lde {
        assert_eq!(a.m, num_aux);
        assert_eq!(a.lde_size, main_lde.lde_size);
    } else {
        assert_eq!(num_aux, 0);
    }
    if let Some(h) = h_parts_dev {
        assert_eq!(h.m, num_parts);
        assert_eq!(h.lde_size, main_lde.lde_size);
    } else {
        assert_eq!(h_parts_host.len(), num_parts * 3 * main_lde.lde_size);
    }
    assert_eq!(h_ood.len(), num_parts * 3);
    let num_total_cols = num_main + num_aux;
    assert_eq!(trace_ood.len(), num_total_cols * num_eval_points * 3);
    assert_eq!(gammas_h.len(), num_parts * 3);
    assert_eq!(gammas_tr.len(), num_total_cols * num_eval_points * 3);
    assert_eq!(inv_h.len(), domain_size * 3);
    assert_eq!(inv_t.len(), num_eval_points * domain_size * 3);

    // Kernel reads `*_lde[c * lde_stride + i * row_stride]` for i in
    // 0..domain_size. Reject inputs that would walk past the LDE buffer.
    if domain_size > 0 {
        let max_row = (domain_size - 1)
            .checked_mul(row_stride)
            .expect("deep composition: (domain_size - 1) * row_stride overflow");
        assert!(
            max_row < main_lde.lde_size,
            "deep composition: kernel row {max_row} out of LDE stride {}",
            main_lde.lde_size
        );
    }

    let be = backend()?;

    let ood_folded = fold_ood_constants(h_ood, trace_ood, gammas_h, gammas_tr, num_eval_points);
    let (ood_folded_dev, gammas_h_dev, gammas_tr_dev, inv_h_dev, inv_t_dev) = (
        stream.clone_htod(&ood_folded)?,
        stream.clone_htod(gammas_h)?,
        stream.clone_htod(gammas_tr)?,
        stream.clone_htod(inv_h)?,
        stream.clone_htod(inv_t)?,
    );

    let h_lde_host_dev;
    let dummy_aux;

    // SAFETY: the deep_composition kernel writes every output slot before
    // any read, so uninitialised contents are never observed.
    let mut deep_out = unsafe { stream.alloc::<u64>(domain_size * 3) }?;

    let aux_slice = if let Some(a) = aux_lde {
        a.buf.as_ref()
    } else {
        dummy_aux = stream.alloc_zeros::<u64>(1)?;
        &dummy_aux
    };

    let h_lde_slice = if let Some(h) = h_parts_dev {
        h.buf.as_ref()
    } else {
        h_lde_host_dev = stream.clone_htod(h_parts_host)?;
        &h_lde_host_dev
    };

    let lde_stride = main_lde.lde_size as u64;
    let num_main_u = num_main as u64;
    let num_aux_u = num_aux as u64;
    let num_parts_u = num_parts as u64;
    let num_eval_points_u = num_eval_points as u64;
    let row_stride_u = row_stride as u64;
    let domain_size_u = domain_size as u64;

    let grid = (domain_size as u32).div_ceil(128);
    let cfg = LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&be.deep_composition_ext3_row)
            .arg(main_lde.buf.as_ref())
            .arg(aux_slice)
            .arg(h_lde_slice)
            .arg(&lde_stride)
            .arg(&num_main_u)
            .arg(&num_aux_u)
            .arg(&num_parts_u)
            .arg(&num_eval_points_u)
            .arg(&row_stride_u)
            .arg(&domain_size_u)
            .arg(&ood_folded_dev)
            .arg(&gammas_h_dev)
            .arg(&gammas_tr_dev)
            .arg(&inv_h_dev)
            .arg(&inv_t_dev)
            .arg(&mut deep_out)
            .launch(cfg)?;
    }

    // DEEP output (domain_size * 3 u64s, ~50 MB): async D2H through the
    // per-worker pinned slab instead of a blocking pageable copy. The
    // synchronize drains the kernels and the DMA so the pending wait below
    // is instant.
    let pending = crate::device::async_dtoh_via(
        stream,
        be.pinned_staging(),
        &be.ctx,
        &deep_out,
        domain_size * 3,
    )?;
    stream.synchronize()?;
    let mut out = vec![0u64; domain_size * 3];
    pending.wait_into_u64(&mut out)?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn raw(e: &Fp3) -> [u64; 3] {
        let v = e.value();
        [*v[0].value(), *v[1].value(), *v[2].value()]
    }

    fn flatten(v: &[Fp3]) -> Vec<u64> {
        v.iter().flat_map(raw).collect()
    }

    fn fp3(seed: u64) -> Fp3 {
        // Includes non-canonical limbs (>= p) to match what the kernel may see.
        let limb = |s: u64| {
            FieldElement::<GoldilocksField>::from_raw(s.wrapping_mul(0x9E37_79B9_7F4A_7C15))
        };
        Fp3::new([limb(seed), limb(seed ^ 0xFFFF_FFFF_0000_0001), limb(!seed)])
    }

    /// `fold_ood_constants` against the direct per-point sums, including the
    /// `num_eval_points == 0` (H-term only) and empty-parts shapes.
    #[test]
    fn fold_ood_constants_matches_direct_sums() {
        for &(num_parts, num_cols, num_eval_points) in
            &[(2usize, 5usize, 2usize), (1, 7, 3), (4, 3, 0), (0, 4, 5)]
        {
            let s = (num_parts * 100 + num_cols * 10 + num_eval_points) as u64;
            let h_ood: Vec<Fp3> = (0..num_parts as u64).map(|j| fp3(s + j)).collect();
            let gammas_h: Vec<Fp3> = (0..num_parts as u64).map(|j| fp3(s + 50 + j)).collect();
            let n = (num_cols * num_eval_points) as u64;
            let trace_ood: Vec<Fp3> = (0..n).map(|j| fp3(s + 1000 + j)).collect();
            let gammas_tr: Vec<Fp3> = (0..n).map(|j| fp3(s + 5000 + j)).collect();

            let mut expected = vec![Fp3::zero(); 1 + num_eval_points];
            for j in 0..num_parts {
                expected[0] += &gammas_h[j] * &h_ood[j];
            }
            for j in 0..num_cols {
                for k in 0..num_eval_points {
                    let idx = j * num_eval_points + k;
                    expected[1 + k] += &gammas_tr[idx] * &trace_ood[idx];
                }
            }

            let got = fold_ood_constants(
                &flatten(&h_ood),
                &flatten(&trace_ood),
                &flatten(&gammas_h),
                &flatten(&gammas_tr),
                num_eval_points,
            );
            assert_eq!(got.len(), (1 + num_eval_points) * 3);
            for (k, e) in expected.iter().enumerate() {
                assert_eq!(fp3_from_raw(&got[k * 3..k * 3 + 3]), *e, "point {k}");
            }
        }
    }
}
