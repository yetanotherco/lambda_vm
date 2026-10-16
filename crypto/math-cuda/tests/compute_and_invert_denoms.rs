//! Parity: GPU `compute_and_invert_denoms_ext3_dev` matches the CPU
//! reference `denoms[k * n + i] = x_lde[i] - z[k]` followed by
//! `inplace_batch_inverse`. Mirrors the shapes used by R3 OOD (n =
//! trace_size, k = num_eval_points) and R4 DEEP (n = lde_size, k =
//! 1 + num_eval_points).

use math::field::element::FieldElement;
use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField;
use math::field::goldilocks::GoldilocksField;
use math::field::traits::IsPrimeField;
use math_cuda::device::backend;
use math_cuda::inverse::{DenomSign, compute_and_invert_denoms_ext3_dev};
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

fn ext3_to_u64s(col: &[Fp3]) -> Vec<u64> {
    let mut out = Vec::with_capacity(col.len() * 3);
    for e in col {
        out.push(*e.value()[0].value());
        out.push(*e.value()[1].value());
        out.push(*e.value()[2].value());
    }
    out
}

fn canon3(a: &[u64]) -> Vec<u64> {
    a.iter().map(GoldilocksField::canonical).collect()
}

fn run(n: usize, k_scalars: usize, sign: DenomSign, seed: u64) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    // x_lde: base-field, n elements. Avoid the trivial case where x_lde[i]
    // happens to equal a z_scalars[k] component (that would make a denom
    // zero and trigger the batch-invert zero-norm assert).
    let x_lde: Vec<Fp> = (0..n).map(|_| rand_fp(&mut rng)).collect();
    let z_scalars: Vec<Fp3> = (0..k_scalars).map(|_| rand_fp3(&mut rng)).collect();

    // CPU reference: denom layout depends on `sign`.
    let mut denoms_cpu: Vec<Fp3> = Vec::with_capacity(n * k_scalars);
    for z in &z_scalars {
        for x in &x_lde {
            let x_lifted = Fp3::new([*x, Fp::zero(), Fp::zero()]);
            let d = match sign {
                DenomSign::ZMinusX => z - &x_lifted,
                DenomSign::XMinusZ => &x_lifted - z,
            };
            denoms_cpu.push(d);
        }
    }
    FieldElement::inplace_batch_inverse(&mut denoms_cpu).expect("denoms non-zero");

    // GPU: H2D x_lde, then run the fused compute+invert.
    let be = backend().unwrap();
    let stream = be.next_stream();
    let x_u64: Vec<u64> = x_lde.iter().map(|x| *x.value()).collect();
    let x_dev = stream.clone_htod(&x_u64).unwrap();
    let z_u64 = ext3_to_u64s(&z_scalars);
    let inv_dev =
        compute_and_invert_denoms_ext3_dev(&x_dev, &z_u64, n, k_scalars, sign, &stream).unwrap();
    let gpu_u64: Vec<u64> = stream.clone_dtoh(&inv_dev).unwrap();
    stream.synchronize().unwrap();

    let cpu_u64 = ext3_to_u64s(&denoms_cpu);
    let gpu_canon = canon3(&gpu_u64);
    let cpu_canon = canon3(&cpu_u64);

    for i in 0..(n * k_scalars) {
        let g = &gpu_canon[i * 3..(i + 1) * 3];
        let c = &cpu_canon[i * 3..(i + 1) * 3];
        assert_eq!(
            g,
            c,
            "mismatch at flat={i} (k={}, idx={}) n={n} k_scalars={k_scalars}",
            i / n,
            i % n
        );
    }
}

#[test]
fn denoms_small_both_signs() {
    // Tiny shapes for fast-feedback debugging, both sign conventions.
    run(8, 1, DenomSign::ZMinusX, 100);
    run(8, 1, DenomSign::XMinusZ, 101);
    run(16, 3, DenomSign::ZMinusX, 200);
    run(64, 5, DenomSign::XMinusZ, 300);
}

#[test]
fn denoms_r3_ood_shape() {
    // R3 OOD: n = trace_size, k = num_eval_points (z - x convention).
    run(1 << 14, 4, DenomSign::ZMinusX, 400);
    run(1 << 16, 4, DenomSign::ZMinusX, 500);
}

#[test]
fn denoms_r4_deep_shape() {
    // R4 DEEP: n = lde_size, k = 1 + num_eval_points (x - z convention).
    run(1 << 18, 5, DenomSign::XMinusZ, 600);
}
