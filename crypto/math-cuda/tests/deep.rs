//! Parity: GPU deep_composition_ext3 vs a direct CPU port of the same
//! row-wise summation. Uses random inputs, not the full stark LDE path.

use std::sync::Arc;

use math::field::element::FieldElement;
use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField;
use math::field::goldilocks::GoldilocksField;
use math::field::traits::IsPrimeField;
use math_cuda::deep::deep_composition_ext3;
use math_cuda::device::backend;
use math_cuda::lde::{GpuLdeBase, GpuLdeExt3};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

type Fp = FieldElement<GoldilocksField>;
type Fp3 = FieldElement<Degree3GoldilocksExtensionField>;

fn rand_fp(rng: &mut ChaCha8Rng) -> Fp {
    Fp::from_raw(rng.r#gen::<u64>())
}
fn rand_fp3(rng: &mut ChaCha8Rng) -> Fp3 {
    Fp3::new([rand_fp(rng), rand_fp(rng), rand_fp(rng)])
}

fn ext3_to_raw(e: &Fp3) -> [u64; 3] {
    [
        *e.value()[0].value(),
        *e.value()[1].value(),
        *e.value()[2].value(),
    ]
}

fn canon3(e: &Fp3) -> [u64; 3] {
    [
        GoldilocksField::canonical(e.value()[0].value()),
        GoldilocksField::canonical(e.value()[1].value()),
        GoldilocksField::canonical(e.value()[2].value()),
    ]
}

/// CPU reference: exact port of `compute_deep_composition_poly_evaluations`.
#[allow(clippy::too_many_arguments)]
fn cpu_deep(
    main_lde: &[Vec<Fp>],   // num_main cols * lde_size
    aux_lde: &[Vec<Fp3>],   // num_aux cols * lde_size
    h_lde: &[Vec<Fp3>],     // num_parts * lde_size
    h_ood: &[Fp3],          // num_parts
    trace_ood: &[Vec<Fp3>], // num_total_cols * num_eval_points
    gammas_h: &[Fp3],       // num_parts
    gammas_tr: &[Vec<Fp3>], // num_total_cols * num_eval_points
    inv_h: &[Fp3],          // domain_size
    inv_t: &[Vec<Fp3>],     // num_eval_points * domain_size
    blowup_factor: usize,
    domain_size: usize,
) -> Vec<Fp3> {
    let num_parts = h_lde.len();
    let num_main = main_lde.len();
    let num_aux = aux_lde.len();
    let num_eval_points = if trace_ood.is_empty() {
        0
    } else {
        trace_ood[0].len()
    };

    (0..domain_size)
        .map(|i| {
            let row = i * blowup_factor;
            let mut result = Fp3::zero();
            // H-terms
            for j in 0..num_parts {
                let num = h_lde[j][row] - h_ood[j];
                result += gammas_h[j] * num * inv_h[i];
            }
            // Main
            for j in 0..num_main {
                for k in 0..num_eval_points {
                    let t_val = &main_lde[j][row];
                    let t_ood = &trace_ood[j][k];
                    let num = t_val - t_ood; // base - ext3 = ext3
                    result += gammas_tr[j][k] * num * inv_t[k][i];
                }
            }
            // Aux
            for (j, aux_col) in aux_lde.iter().enumerate().take(num_aux) {
                let trace_j = num_main + j;
                for k in 0..num_eval_points {
                    let t_val = &aux_col[row];
                    let t_ood = &trace_ood[trace_j][k];
                    let num = t_val - t_ood;
                    result += gammas_tr[trace_j][k] * num * inv_t[k][i];
                }
            }
            result
        })
        .collect()
}

fn run_parity(
    log_domain_size: u32,
    blowup_factor: usize,
    num_main: usize,
    num_aux: usize,
    num_parts: usize,
    num_eval_points: usize,
    seed: u64,
) {
    let domain_size = 1usize << log_domain_size;
    let lde_size = domain_size * blowup_factor;
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    let main_lde: Vec<Vec<Fp>> = (0..num_main)
        .map(|_| (0..lde_size).map(|_| rand_fp(&mut rng)).collect())
        .collect();
    let aux_lde: Vec<Vec<Fp3>> = (0..num_aux)
        .map(|_| (0..lde_size).map(|_| rand_fp3(&mut rng)).collect())
        .collect();
    let h_lde: Vec<Vec<Fp3>> = (0..num_parts)
        .map(|_| (0..lde_size).map(|_| rand_fp3(&mut rng)).collect())
        .collect();
    let h_ood: Vec<Fp3> = (0..num_parts).map(|_| rand_fp3(&mut rng)).collect();
    let num_total_cols = num_main + num_aux;
    let trace_ood: Vec<Vec<Fp3>> = (0..num_total_cols)
        .map(|_| (0..num_eval_points).map(|_| rand_fp3(&mut rng)).collect())
        .collect();
    let gammas_h: Vec<Fp3> = (0..num_parts).map(|_| rand_fp3(&mut rng)).collect();
    let gammas_tr: Vec<Vec<Fp3>> = (0..num_total_cols)
        .map(|_| (0..num_eval_points).map(|_| rand_fp3(&mut rng)).collect())
        .collect();
    let inv_h: Vec<Fp3> = (0..domain_size).map(|_| rand_fp3(&mut rng)).collect();
    let inv_t: Vec<Vec<Fp3>> = (0..num_eval_points)
        .map(|_| (0..domain_size).map(|_| rand_fp3(&mut rng)).collect())
        .collect();

    // CPU reference.
    let cpu_out = cpu_deep(
        &main_lde,
        &aux_lde,
        &h_lde,
        &h_ood,
        &trace_ood,
        &gammas_h,
        &gammas_tr,
        &inv_h,
        &inv_t,
        blowup_factor,
        domain_size,
    );

    // GPU: upload main & aux LDEs into device buffers and wrap in handles.
    let be = backend().unwrap();
    let stream = be.next_stream();

    // main_lde to col-major u64: m * lde_size
    let mut main_flat = vec![0u64; num_main * lde_size];
    for (c, col) in main_lde.iter().enumerate() {
        for (r, v) in col.iter().enumerate() {
            main_flat[c * lde_size + r] = *v.value();
        }
    }
    let main_dev = stream.clone_htod(&main_flat).unwrap();

    // aux_lde to de-interleaved: (m*3) * lde_size
    let mut aux_flat = vec![0u64; num_aux * 3 * lde_size];
    for (c, col) in aux_lde.iter().enumerate() {
        for (r, v) in col.iter().enumerate() {
            let [a, b, c0] = ext3_to_raw(v);
            aux_flat[(c * 3) * lde_size + r] = a;
            aux_flat[(c * 3 + 1) * lde_size + r] = b;
            aux_flat[(c * 3 + 2) * lde_size + r] = c0;
        }
    }
    let aux_dev = stream.clone_htod(&aux_flat).unwrap();
    stream.synchronize().unwrap();

    let main_handle = GpuLdeBase {
        buf: Arc::new(main_dev),
        m: num_main,
        lde_size,
    };
    let aux_handle = if num_aux > 0 {
        Some(GpuLdeExt3 {
            buf: Arc::new(aux_dev),
            m: num_aux,
            lde_size,
        })
    } else {
        drop(aux_dev);
        None
    };

    // h_parts to de-interleaved: num_parts*3 * lde_size
    let mut h_flat = vec![0u64; num_parts * 3 * lde_size];
    for (p, col) in h_lde.iter().enumerate() {
        for (r, v) in col.iter().enumerate() {
            let [a, b, c0] = ext3_to_raw(v);
            h_flat[(p * 3) * lde_size + r] = a;
            h_flat[(p * 3 + 1) * lde_size + r] = b;
            h_flat[(p * 3 + 2) * lde_size + r] = c0;
        }
    }

    let mut h_ood_flat = vec![0u64; num_parts * 3];
    for (j, e) in h_ood.iter().enumerate() {
        let [a, b, c] = ext3_to_raw(e);
        h_ood_flat[j * 3] = a;
        h_ood_flat[j * 3 + 1] = b;
        h_ood_flat[j * 3 + 2] = c;
    }
    let mut trace_ood_flat = vec![0u64; num_total_cols * num_eval_points * 3];
    for (j, col) in trace_ood.iter().enumerate() {
        for (k, e) in col.iter().enumerate() {
            let idx = (j * num_eval_points + k) * 3;
            let [a, b, c] = ext3_to_raw(e);
            trace_ood_flat[idx] = a;
            trace_ood_flat[idx + 1] = b;
            trace_ood_flat[idx + 2] = c;
        }
    }
    let mut gammas_h_flat = vec![0u64; num_parts * 3];
    for (j, e) in gammas_h.iter().enumerate() {
        let [a, b, c] = ext3_to_raw(e);
        gammas_h_flat[j * 3] = a;
        gammas_h_flat[j * 3 + 1] = b;
        gammas_h_flat[j * 3 + 2] = c;
    }
    let mut gammas_tr_flat = vec![0u64; num_total_cols * num_eval_points * 3];
    for (j, col) in gammas_tr.iter().enumerate() {
        for (k, e) in col.iter().enumerate() {
            let idx = (j * num_eval_points + k) * 3;
            let [a, b, c] = ext3_to_raw(e);
            gammas_tr_flat[idx] = a;
            gammas_tr_flat[idx + 1] = b;
            gammas_tr_flat[idx + 2] = c;
        }
    }
    let mut inv_h_flat = vec![0u64; domain_size * 3];
    for (i, e) in inv_h.iter().enumerate() {
        let [a, b, c] = ext3_to_raw(e);
        inv_h_flat[i * 3] = a;
        inv_h_flat[i * 3 + 1] = b;
        inv_h_flat[i * 3 + 2] = c;
    }
    let mut inv_t_flat = vec![0u64; num_eval_points * domain_size * 3];
    for (k, layer) in inv_t.iter().enumerate() {
        for (i, e) in layer.iter().enumerate() {
            let idx = (k * domain_size + i) * 3;
            let [a, b, c] = ext3_to_raw(e);
            inv_t_flat[idx] = a;
            inv_t_flat[idx + 1] = b;
            inv_t_flat[idx + 2] = c;
        }
    }

    let gpu_raw = deep_composition_ext3(
        &main_handle,
        aux_handle.as_ref(),
        &h_flat,
        &h_ood_flat,
        &trace_ood_flat,
        &gammas_h_flat,
        &gammas_tr_flat,
        &inv_h_flat,
        &inv_t_flat,
        num_parts,
        num_main,
        num_aux,
        num_eval_points,
        blowup_factor,
        domain_size,
    )
    .unwrap();

    for i in 0..domain_size {
        let gpu = [gpu_raw[i * 3], gpu_raw[i * 3 + 1], gpu_raw[i * 3 + 2]];
        let gpu_canon = [
            GoldilocksField::canonical(&gpu[0]),
            GoldilocksField::canonical(&gpu[1]),
            GoldilocksField::canonical(&gpu[2]),
        ];
        let cpu_canon = canon3(&cpu_out[i]);
        assert_eq!(
            gpu_canon, cpu_canon,
            "row {i} mismatch at log_ds={log_domain_size} main={num_main} aux={num_aux} parts={num_parts}"
        );
    }
}

#[test]
fn deep_parity_small() {
    run_parity(4, 2, 3, 2, 2, 1, 100);
    run_parity(6, 4, 5, 3, 2, 2, 200);
}

#[test]
fn deep_parity_medium() {
    run_parity(10, 2, 10, 5, 4, 3, 1000);
}

#[test]
fn deep_parity_no_aux() {
    run_parity(8, 2, 5, 0, 2, 2, 5000);
}
