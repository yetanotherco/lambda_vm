//! R4 deep-composition polynomial evaluations on GPU.
//!
//! Mirrors `Self::compute_deep_composition_poly_evaluations` in
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
    h_ood: &[u64],                  // num_parts * 3
    trace_ood: &[u64],              // num_total_cols * num_eval_points * 3
    gammas_h: &[u64],               // num_parts * 3
    gammas_tr: &[u64],              // num_total_cols * num_eval_points * 3
    inv_h: &[u64],                  // domain_size * 3
    inv_t: &[u64],                  // num_eval_points * domain_size * 3
    // Shape params
    num_parts: usize,
    num_main: usize,
    num_aux: usize,
    num_eval_points: usize,
    blowup_factor: usize,
    domain_size: usize,
) -> Result<Vec<u64>> {
    assert_eq!(main_lde.m, num_main);
    if let Some(a) = aux_lde {
        assert_eq!(a.m, num_aux);
        assert_eq!(a.lde_size, main_lde.lde_size);
    } else {
        assert_eq!(num_aux, 0);
    }
    assert_eq!(h_parts_deinterleaved.len(), num_parts * 3 * main_lde.lde_size);
    assert_eq!(h_ood.len(), num_parts * 3);
    let num_total_cols = num_main + num_aux;
    assert_eq!(trace_ood.len(), num_total_cols * num_eval_points * 3);
    assert_eq!(gammas_h.len(), num_parts * 3);
    assert_eq!(gammas_tr.len(), num_total_cols * num_eval_points * 3);
    assert_eq!(inv_h.len(), domain_size * 3);
    assert_eq!(inv_t.len(), num_eval_points * domain_size * 3);

    let be = backend();
    let stream = be.next_stream();

    // H2D the host-side arrays.
    let h_lde_dev = stream.clone_htod(h_parts_deinterleaved)?;
    let h_ood_dev = stream.clone_htod(h_ood)?;
    let trace_ood_dev = stream.clone_htod(trace_ood)?;
    let gammas_h_dev = stream.clone_htod(gammas_h)?;
    let gammas_tr_dev = stream.clone_htod(gammas_tr)?;
    let inv_h_dev = stream.clone_htod(inv_h)?;
    let inv_t_dev = stream.clone_htod(inv_t)?;

    let mut deep_out = stream.alloc_zeros::<u64>(domain_size * 3)?;

    // Dummy zero-sized aux LDE buffer when num_aux == 0 — the kernel's aux
    // loop skips iteration but the pointer still needs to be valid.
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
    let blowup_u = blowup_factor as u64;
    let domain_size_u = domain_size as u64;

    let grid = ((domain_size as u32) + 128 - 1) / 128;
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
            .arg(&h_lde_dev)
            .arg(&lde_stride)
            .arg(&num_main_u)
            .arg(&num_aux_u)
            .arg(&num_parts_u)
            .arg(&num_eval_points_u)
            .arg(&blowup_u)
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
