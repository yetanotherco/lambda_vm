//! GPU barycentric kernels (`barycentric_base` / `barycentric_ext3`) must produce
//! the same OOD evaluation as the CPU formula in `get_trace_evaluations_from_lde`
//! (`interpolate_coset_eval_ext_with_g_n_inv`). Covers base field and ext3.

use math::field::element::FieldElement;
use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField;
use math::field::goldilocks::GoldilocksField;
use math::field::traits::{IsFFTField, IsField, IsPrimeField, IsSubFieldOf};
use math::polynomial::barycentric_inv_denoms;
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

/// Build coset points `[g * ω^0, g * ω^1, ..., g * ω^{n-1}]` from a
/// coset offset `g` and the primitive root `ω` of the trace domain.
fn coset_points(n: usize, coset_offset: u64) -> Vec<Fp> {
    let log_n = n.trailing_zeros() as u64;
    let omega = GoldilocksField::get_primitive_root_of_unity(log_n).unwrap();
    let g = Fp::from_raw(coset_offset);
    let mut pts = Vec::with_capacity(n);
    let mut cur = g.clone();
    for _ in 0..n {
        pts.push(cur.clone());
        cur = &cur * &omega;
    }
    pts
}

/// CPU barycentric eval for a single base-field column.
/// Mirrors the prover's `get_trace_evaluations_from_lde` inner loop:
///   col_scale[i] = point[i] * inv_denom[i]
///   sum          = Σ lde[i*blowup] * col_scale[i]   (Fp × Fp3 → Fp3)
///   result       = (n_inv * g_n_inv) * (z^N - g^N) * sum
fn cpu_barycentric_base(
    lde_col: &[Fp],
    blowup: usize,
    coset_pts: &[Fp],
    z: &Fp3,
    coset_offset: &Fp,
) -> Fp3 {
    let n = coset_pts.len();
    let n_inv = Fp::from(n as u64).inv().unwrap();
    let g_n = coset_offset.pow(n as u64);
    let g_n_inv = g_n.inv().unwrap();
    let z_pow_n = z.pow(n as u64);

    let inv_denoms = barycentric_inv_denoms::<GoldilocksField, Degree3GoldilocksExtensionField>(
        z,
        coset_pts,
    );

    let col_scale: Vec<Fp3> = coset_pts
        .iter()
        .zip(inv_denoms.iter())
        .map(|(pt, inv_d)| pt * inv_d)
        .collect();

    let sum = col_scale
        .iter()
        .enumerate()
        .fold(Fp3::from(0u64), |acc, (i, scale)| {
            acc + &lde_col[i * blowup] * scale
        });

    let vanishing = z_pow_n.sub_subfield(&g_n);
    let scalar = &n_inv * &g_n_inv;
    &scalar * &(&vanishing * &sum)
}

/// GPU barycentric eval for a single column via `barycentric_base` kernel,
/// followed by the host-side vanishing scaling that the prover applies.
fn gpu_barycentric_base(
    lde_col: &[Fp],
    blowup: usize,
    coset_pts: &[Fp],
    z: &Fp3,
    coset_offset: &Fp,
) -> Fp3 {
    let n = coset_pts.len();
    let lde_size = lde_col.len();

    let n_inv = Fp::from(n as u64).inv().unwrap();
    let g_n = coset_offset.pow(n as u64);
    let g_n_inv = g_n.inv().unwrap();
    let z_pow_n = z.pow(n as u64);

    let inv_denoms_fp3 =
        barycentric_inv_denoms::<GoldilocksField, Degree3GoldilocksExtensionField>(z, coset_pts);

    // Pack for GPU: coset_points as u64, inv_denoms interleaved ext3 u64.
    let pts_u64: Vec<u64> = coset_pts.iter().map(|p| *p.value()).collect();
    let inv_u64: Vec<u64> = inv_denoms_fp3
        .iter()
        .flat_map(|e| {
            [
                *e.value()[0].value(),
                *e.value()[1].value(),
                *e.value()[2].value(),
            ]
        })
        .collect();

    // Pre-strided column (trace points at stride blowup).
    let pre_strided: Vec<u64> = (0..n).map(|i| *lde_col[i * blowup].value()).collect();

    let raw = math_cuda::barycentric::barycentric_base(&pre_strided, n, &pts_u64, &inv_u64, n, 1)
        .expect("GPU barycentric_base");

    // raw is 3 u64s (ext3 interleaved): the unscaled sum S.
    // The prover then applies: result = scalar * (vanishing * S)
    // where scalar = n_inv * g_n_inv, vanishing = z^N - g^N.
    let s = Fp3::new([
        Fp::from_raw(raw[0]),
        Fp::from_raw(raw[1]),
        Fp::from_raw(raw[2]),
    ]);
    let vanishing = z_pow_n.sub_subfield(&g_n);
    let scalar = &n_inv * &g_n_inv;
    &scalar * &(&vanishing * &s)
}

#[test]
fn gpu_barycentric_base_matches_cpu() {
    const COSET_OFFSET: u64 = 7;

    for log_n in [4usize, 6, 8] {
        for blowup in [2usize, 4] {
            let n = 1usize << log_n;
            let lde_size = n * blowup;
            let mut rng = ChaCha8Rng::seed_from_u64((log_n * 100 + blowup) as u64);

            let lde_col: Vec<Fp> = (0..lde_size).map(|_| rand_fp(&mut rng)).collect();
            let z = rand_fp3(&mut rng);
            let coset_offset = Fp::from_raw(COSET_OFFSET);
            let pts = coset_points(n, COSET_OFFSET);

            let cpu = cpu_barycentric_base(&lde_col, blowup, &pts, &z, &coset_offset);
            let gpu = gpu_barycentric_base(&lde_col, blowup, &pts, &z, &coset_offset);

            for k in 0..3 {
                let cpu_k = *cpu.value()[k].value();
                let gpu_k = *gpu.value()[k].value();
                let cpu_c = GoldilocksField::canonical(&cpu_k);
                let gpu_c = GoldilocksField::canonical(&gpu_k);
                assert_eq!(
                    cpu_c, gpu_c,
                    "component {k} mismatch: log_n={log_n} blowup={blowup} \
                     cpu={cpu_c} gpu={gpu_c}"
                );
            }
        }
    }
}

// ── Ext3 aux path ─────────────────────────────────────────────────────────────

/// CPU barycentric for a single ext3 column (aux trace path).
fn cpu_barycentric_ext3(
    lde_col: &[Fp3],
    blowup: usize,
    coset_pts: &[Fp],
    z: &Fp3,
    coset_offset: &Fp,
) -> Fp3 {
    let n = coset_pts.len();
    let n_inv = Fp::from(n as u64).inv().unwrap();
    let g_n = coset_offset.pow(n as u64);
    let g_n_inv = g_n.inv().unwrap();
    let z_pow_n = z.pow(n as u64);

    let inv_denoms = barycentric_inv_denoms::<GoldilocksField, Degree3GoldilocksExtensionField>(
        z,
        coset_pts,
    );

    let col_scale: Vec<Fp3> = coset_pts
        .iter()
        .zip(inv_denoms.iter())
        .map(|(pt, inv_d)| pt * inv_d)
        .collect();

    let sum = col_scale
        .iter()
        .enumerate()
        .fold(Fp3::from(0u64), |acc, (i, scale)| {
            acc + scale * &lde_col[i * blowup]
        });

    let vanishing = z_pow_n.sub_subfield(&g_n);
    let scalar = &n_inv * &g_n_inv;
    &scalar * &(&vanishing * &sum)
}

/// GPU barycentric for a single ext3 column via `barycentric_ext3` kernel.
fn gpu_barycentric_ext3(
    lde_col: &[Fp3],
    blowup: usize,
    coset_pts: &[Fp],
    z: &Fp3,
    coset_offset: &Fp,
) -> Fp3 {
    let n = coset_pts.len();
    let n_inv = Fp::from(n as u64).inv().unwrap();
    let g_n = coset_offset.pow(n as u64);
    let g_n_inv = g_n.inv().unwrap();
    let z_pow_n = z.pow(n as u64);

    let inv_denoms_fp3 =
        barycentric_inv_denoms::<GoldilocksField, Degree3GoldilocksExtensionField>(z, coset_pts);

    let pts_u64: Vec<u64> = coset_pts.iter().map(|p| *p.value()).collect();
    let inv_u64: Vec<u64> = inv_denoms_fp3
        .iter()
        .flat_map(|e| {
            [
                *e.value()[0].value(),
                *e.value()[1].value(),
                *e.value()[2].value(),
            ]
        })
        .collect();

    // Pre-strided ext3 in the de-interleaved (component-major) layout the
    // kernel expects: slab k at offset k*n holds component k of all n points.
    let mut pre_strided: Vec<u64> = vec![0u64; 3 * n];
    for i in 0..n {
        let e = &lde_col[i * blowup];
        pre_strided[i] = *e.value()[0].value();
        pre_strided[n + i] = *e.value()[1].value();
        pre_strided[2 * n + i] = *e.value()[2].value();
    }

    let raw =
        math_cuda::barycentric::barycentric_ext3(&pre_strided, n, &pts_u64, &inv_u64, n, 1)
            .expect("GPU barycentric_ext3");

    let s = Fp3::new([
        Fp::from_raw(raw[0]),
        Fp::from_raw(raw[1]),
        Fp::from_raw(raw[2]),
    ]);
    let vanishing = z_pow_n.sub_subfield(&g_n);
    let scalar = &n_inv * &g_n_inv;
    &scalar * &(&vanishing * &s)
}

#[test]
fn gpu_barycentric_ext3_matches_cpu() {
    const COSET_OFFSET: u64 = 7;

    for log_n in [4usize, 6, 8] {
        for blowup in [2usize, 4] {
            let n = 1usize << log_n;
            let lde_size = n * blowup;
            let mut rng = ChaCha8Rng::seed_from_u64((log_n * 100 + blowup + 5000) as u64);

            let lde_col: Vec<Fp3> = (0..lde_size).map(|_| rand_fp3(&mut rng)).collect();
            let z = rand_fp3(&mut rng);
            let coset_offset = Fp::from_raw(COSET_OFFSET);
            let pts = coset_points(n, COSET_OFFSET);

            let cpu = cpu_barycentric_ext3(&lde_col, blowup, &pts, &z, &coset_offset);
            let gpu = gpu_barycentric_ext3(&lde_col, blowup, &pts, &z, &coset_offset);

            for k in 0..3 {
                let cpu_k = *cpu.value()[k].value();
                let gpu_k = *gpu.value()[k].value();
                let cpu_c = GoldilocksField::canonical(&cpu_k);
                let gpu_c = GoldilocksField::canonical(&gpu_k);
                assert_eq!(
                    cpu_c, gpu_c,
                    "ext3 component {k} mismatch: log_n={log_n} blowup={blowup} \
                     cpu={cpu_c} gpu={gpu_c}"
                );
            }
        }
    }
}
