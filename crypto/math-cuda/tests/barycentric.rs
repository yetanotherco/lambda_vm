//! Parity: GPU barycentric sum vs CPU. We don't call the upstream
//! `interpolate_coset_eval_*_with_g_n_inv` directly because the GPU kernel
//! returns only the unscaled sum and the caller applies the ext3 scale. We
//! replicate the same unscaled sum on CPU for comparison.

use math::field::element::FieldElement;
use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField;
use math::field::goldilocks::GoldilocksField;
use math::field::traits::IsPrimeField;
use math_cuda::barycentric::{barycentric_base, barycentric_ext3};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

type Fp = FieldElement<GoldilocksField>;
type Fp3 = FieldElement<Degree3GoldilocksExtensionField>;

fn canon_triplet(e: &Fp3) -> [u64; 3] {
    [
        GoldilocksField::canonical(e.value()[0].value()),
        GoldilocksField::canonical(e.value()[1].value()),
        GoldilocksField::canonical(e.value()[2].value()),
    ]
}

fn canon_triplet_raw(t: &[u64]) -> [u64; 3] {
    [
        GoldilocksField::canonical(&t[0]),
        GoldilocksField::canonical(&t[1]),
        GoldilocksField::canonical(&t[2]),
    ]
}

fn random_fp(rng: &mut ChaCha8Rng) -> Fp {
    Fp::from_raw(rng.r#gen::<u64>())
}
fn random_fp3(rng: &mut ChaCha8Rng) -> Fp3 {
    Fp3::new([random_fp(rng), random_fp(rng), random_fp(rng)])
}

#[test]
fn barycentric_base_sum_matches_cpu() {
    for &(log_n, num_cols) in &[(4u32, 1usize), (8, 5), (10, 17), (12, 3)] {
        let n = 1 << log_n;
        let mut rng = ChaCha8Rng::seed_from_u64(100 + log_n as u64 * 7 + num_cols as u64);

        let coset_points: Vec<Fp> = (0..n).map(|_| random_fp(&mut rng)).collect();
        let inv_denoms: Vec<Fp3> = (0..n).map(|_| random_fp3(&mut rng)).collect();

        // Lay out columns base: column c contiguous slab of n u64s.
        let cols_fp: Vec<Vec<Fp>> = (0..num_cols)
            .map(|_| (0..n).map(|_| random_fp(&mut rng)).collect())
            .collect();
        let mut columns_flat = vec![0u64; num_cols * n];
        for (c, col) in cols_fp.iter().enumerate() {
            for (r, e) in col.iter().enumerate() {
                columns_flat[c * n + r] = *e.value();
            }
        }
        let points_raw: Vec<u64> = coset_points.iter().map(|e| *e.value()).collect();
        let inv_denoms_raw: Vec<u64> = inv_denoms
            .iter()
            .flat_map(|e| {
                [
                    *e.value()[0].value(),
                    *e.value()[1].value(),
                    *e.value()[2].value(),
                ]
            })
            .collect();

        let gpu =
            barycentric_base(&columns_flat, n, &points_raw, &inv_denoms_raw, n, num_cols).unwrap();

        for (c, col) in cols_fp.iter().enumerate() {
            // CPU reference sum. Force ext3 by embedding the base product.
            let mut sum = Fp3::zero();
            for i in 0..n {
                let pe_base: Fp = &coset_points[i] * &col[i]; // F * F = F
                // Base * ext3 = ext3 (base is on the left, IsSubFieldOf direction).
                let pe_ext3: Fp3 = &pe_base * &inv_denoms[i]; // F * E = E
                sum = &sum + &pe_ext3;
            }
            let gpu_sum = canon_triplet_raw(&gpu[c * 3..(c + 1) * 3]);
            let cpu_sum = canon_triplet(&sum);
            assert_eq!(
                gpu_sum, cpu_sum,
                "base col {c} log_n={log_n} num_cols={num_cols}"
            );
        }
    }
}

#[test]
fn barycentric_ext3_sum_matches_cpu() {
    for &(log_n, num_cols) in &[(4u32, 1usize), (8, 3), (10, 7)] {
        let n = 1 << log_n;
        let mut rng = ChaCha8Rng::seed_from_u64(200 + log_n as u64 * 11 + num_cols as u64);

        let coset_points: Vec<Fp> = (0..n).map(|_| random_fp(&mut rng)).collect();
        let inv_denoms: Vec<Fp3> = (0..n).map(|_| random_fp3(&mut rng)).collect();
        let cols_fp3: Vec<Vec<Fp3>> = (0..num_cols)
            .map(|_| (0..n).map(|_| random_fp3(&mut rng)).collect())
            .collect();

        // De-interleaved layout: 3 base slabs per ext3 column.
        let mut columns_flat = vec![0u64; num_cols * 3 * n];
        for (c, col) in cols_fp3.iter().enumerate() {
            for (r, e) in col.iter().enumerate() {
                columns_flat[(c * 3) * n + r] = *e.value()[0].value();
                columns_flat[(c * 3 + 1) * n + r] = *e.value()[1].value();
                columns_flat[(c * 3 + 2) * n + r] = *e.value()[2].value();
            }
        }
        let points_raw: Vec<u64> = coset_points.iter().map(|e| *e.value()).collect();
        let inv_denoms_raw: Vec<u64> = inv_denoms
            .iter()
            .flat_map(|e| {
                [
                    *e.value()[0].value(),
                    *e.value()[1].value(),
                    *e.value()[2].value(),
                ]
            })
            .collect();

        let gpu =
            barycentric_ext3(&columns_flat, n, &points_raw, &inv_denoms_raw, n, num_cols).unwrap();

        for (c, col) in cols_fp3.iter().enumerate() {
            let mut sum = Fp3::zero();
            for i in 0..n {
                let pe: Fp3 = &coset_points[i] * &col[i]; // F * E = E
                let term: Fp3 = &pe * &inv_denoms[i]; // E * E = E
                sum = &sum + &term;
            }
            let gpu_sum = canon_triplet_raw(&gpu[c * 3..(c + 1) * 3]);
            let cpu_sum = canon_triplet(&sum);
            assert_eq!(
                gpu_sum, cpu_sum,
                "ext3 col {c} log_n={log_n} num_cols={num_cols}"
            );
        }
    }
}
