//! R4 deep-composition polynomial evaluations on GPU.
//!
//! Mirrors `compute_deep_composition_poly_evaluations` in
//! `crypto/stark/src/prover.rs`. Accepts the main/aux LDEs as device
//! handles (populated by the R1 fused path in `LDETraceTable`) and
//! takes every other tensor (composition parts LDE, OOD evals,
//! gammas, inv-denoms) from host. Returns a `Vec<u64>` of
//! `domain_size * 3` u64s, ext3 interleaved (ready to `transmute` to
//! `FieldElement<Ext3>` when the caller promises layout compatibility).

use cudarc::driver::{LaunchConfig, PushKernelArg};

use crate::Result;
use crate::device::backend;
use crate::lde::{GpuLdeBase, GpuLdeExt3};

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
    deep_composition_ext3_impl(
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
    deep_composition_ext3_impl(
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

#[allow(clippy::too_many_arguments)]
fn deep_composition_ext3_impl(
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
    let stream = be.next_stream();

    // H2D only the scalar arrays. h_parts comes from a device handle
    // when available.
    let h_ood_dev = stream.clone_htod(h_ood)?;
    let trace_ood_dev = stream.clone_htod(trace_ood)?;
    let gammas_h_dev = stream.clone_htod(gammas_h)?;
    let gammas_tr_dev = stream.clone_htod(gammas_tr)?;
    let inv_h_dev = stream.clone_htod(inv_h)?;
    let inv_t_dev = stream.clone_htod(inv_t)?;

    // Keep the owned H2D of h_lde alive until kernel completes. Only
    // populated in the host-parts path.
    let h_lde_host_dev;

    // SAFETY: the deep_composition kernel writes every output slot before
    // any read, so uninitialised contents are never observed.
    let mut deep_out = unsafe { stream.alloc::<u64>(domain_size * 3) }?;

    let dummy_aux;
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
            .arg(&h_ood_dev)
            .arg(&trace_ood_dev)
            .arg(&gammas_h_dev)
            .arg(&gammas_tr_dev)
            .arg(&inv_h_dev)
            .arg(&inv_t_dev)
            .arg(&mut deep_out)
            .launch(cfg)?;
    }

    let out = stream.clone_dtoh(&deep_out)?;
    stream.synchronize()?;
    Ok(out)
}
