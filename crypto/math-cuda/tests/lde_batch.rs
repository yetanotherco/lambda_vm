//! Batched coset LDE must agree with running the CPU single-column LDE on
//! each column independently. Sweeps a few realistic (n, blowup, m) tuples.

use math::fft::bowers_fft::LayerTwiddles;
use math::field::element::FieldElement;
use math::field::goldilocks::GoldilocksField;
use math::field::traits::{IsField, IsPrimeField};
use math::polynomial::Polynomial;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

type Fp = FieldElement<GoldilocksField>;

fn coset_weights(n: usize, g: u64) -> Vec<u64> {
    let inv_n = *FieldElement::<GoldilocksField>::from(n as u64)
        .inv()
        .unwrap()
        .value();
    let mut w = Vec::with_capacity(n);
    let mut cur = inv_n;
    for _ in 0..n {
        w.push(cur);
        cur = GoldilocksField::mul(&cur, &g);
    }
    w
}

fn cpu_lde_one(
    col: &[u64],
    blowup: usize,
    weights_fp: &[Fp],
    inv_tw: &LayerTwiddles<GoldilocksField>,
    fwd_tw: &LayerTwiddles<GoldilocksField>,
) -> Vec<u64> {
    let mut buf: Vec<Fp> = col.iter().map(|&x| Fp::from_raw(x)).collect();
    Polynomial::coset_lde_full_expand::<GoldilocksField>(
        &mut buf, blowup, weights_fp, inv_tw, fwd_tw,
    )
    .unwrap();
    buf.into_iter().map(|e| *e.value()).collect()
}

fn canon(xs: &[u64]) -> Vec<u64> {
    xs.iter().map(GoldilocksField::canonical).collect()
}

fn assert_batch(log_n: u64, blowup: usize, m: usize, seed: u64) {
    let n = 1usize << log_n;
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let columns: Vec<Vec<u64>> = (0..m)
        .map(|_| (0..n).map(|_| rng.r#gen::<u64>()).collect())
        .collect();

    let coset_offset: u64 = 7;
    let weights = coset_weights(n, coset_offset);
    let weights_fp: Vec<Fp> = weights.iter().map(|&w| Fp::from_raw(w)).collect();

    let inv_tw = LayerTwiddles::<GoldilocksField>::new_inverse(log_n).unwrap();
    let fwd_tw =
        LayerTwiddles::<GoldilocksField>::new((n * blowup).trailing_zeros() as u64).unwrap();

    let slices: Vec<&[u64]> = columns.iter().map(|c| c.as_slice()).collect();
    let gpu_all = math_cuda::lde::coset_lde_batch_base(&slices, blowup, &weights).unwrap();
    assert_eq!(gpu_all.len(), m);

    for (c, col) in columns.iter().enumerate() {
        let cpu = cpu_lde_one(col, blowup, &weights_fp, &inv_tw, &fwd_tw);
        assert_eq!(
            canon(&gpu_all[c]),
            canon(&cpu),
            "batch mismatch at col {c}, log_n={log_n}, blowup={blowup}"
        );
    }
}

#[test]
fn batch_small() {
    for &m in &[1usize, 4, 16] {
        for log_n in 4..=10 {
            assert_batch(log_n, 4, m, 100 + log_n * 10 + m as u64);
        }
    }
}

#[test]
fn batch_medium() {
    for &m in &[2usize, 32] {
        for log_n in 11..=14 {
            assert_batch(log_n, 4, m, 200 + log_n * 10 + m as u64);
        }
    }
}

#[test]
fn batch_large_one_column() {
    assert_batch(18, 4, 1, 0xCAFE);
}

#[test]
fn batch_large_32_columns() {
    assert_batch(15, 4, 32, 0xBEEF);
}
