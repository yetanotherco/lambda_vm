//! R4 DEEP inverse-denominator parity: GPU `compute_and_invert_denoms_ext3_dev`
//! (with `DenomSign::XMinusZ`, the convention used by the prover's R4 DEEP
//! fast path) must match the CPU helper `build_r4_inv_denoms_cpu` that the
//! prover's CPU fallback also calls into.
//!
//! Pins the three-copy fragility flagged in PR review: kernel construction,
//! CPU fallback in prover.rs, and any test references must all be the same.
//! With this test, drift on either the helper or the kernel breaks the build.
//!
//! Requires the `cuda` feature.

#![cfg(feature = "cuda")]

use math::field::element::FieldElement;
use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField;
use math::field::goldilocks::GoldilocksField;
use math::field::traits::IsPrimeField;
use math_cuda::device::backend;
use math_cuda::inverse::{DenomSign, compute_and_invert_denoms_ext3_dev};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use stark::r4_denoms::build_r4_inv_denoms_cpu;

type Fp = FieldElement<GoldilocksField>;
type Fp3 = FieldElement<Degree3GoldilocksExtensionField>;

fn rand_fp(rng: &mut ChaCha8Rng) -> Fp {
    Fp::from_raw(rng.r#gen::<u64>())
}

fn rand_fp3(rng: &mut ChaCha8Rng) -> Fp3 {
    Fp3::new([rand_fp(rng), rand_fp(rng), rand_fp(rng)])
}

fn canon3(a: &[u64]) -> Vec<u64> {
    a.iter().map(GoldilocksField::canonical).collect()
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

fn run_parity(lde_size: usize, num_eval_points: usize, seed: u64) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let coset: Vec<Fp> = (0..lde_size).map(|_| rand_fp(&mut rng)).collect();
    let z_power = rand_fp3(&mut rng);
    let z_shifted: Vec<Fp3> = (0..num_eval_points).map(|_| rand_fp3(&mut rng)).collect();

    // CPU side via the shared helper used by the prover's fallback.
    let cpu = build_r4_inv_denoms_cpu::<GoldilocksField, Degree3GoldilocksExtensionField>(
        &coset, &z_power, &z_shifted,
    )
    .expect("non-zero denoms");
    let cpu_u64 = canon3(&ext3_to_u64s(&cpu));

    // GPU side via the device pipeline that the prover's fast path calls.
    let be = backend().unwrap();
    let stream = be.next_stream();
    let coset_u64: Vec<u64> = coset.iter().map(|x| *x.value()).collect();
    let coset_dev = stream.clone_htod(&coset_u64).unwrap();
    let mut z_scalars: Vec<Fp3> = Vec::with_capacity(1 + num_eval_points);
    z_scalars.push(z_power);
    z_scalars.extend_from_slice(&z_shifted);
    let z_u64 = ext3_to_u64s(&z_scalars);
    let gpu_dev = compute_and_invert_denoms_ext3_dev(
        &coset_dev,
        &z_u64,
        lde_size,
        1 + num_eval_points,
        DenomSign::XMinusZ,
        &stream,
    )
    .unwrap();
    let gpu_u64 = canon3(&stream.clone_dtoh(&gpu_dev).unwrap());
    stream.synchronize().unwrap();

    assert_eq!(
        cpu_u64.len(),
        gpu_u64.len(),
        "length mismatch lde_size={lde_size} num_eval_points={num_eval_points}"
    );
    for i in 0..(lde_size * (1 + num_eval_points)) {
        let c = &cpu_u64[i * 3..(i + 1) * 3];
        let g = &gpu_u64[i * 3..(i + 1) * 3];
        assert_eq!(
            c,
            g,
            "mismatch at flat={i} (k={}, idx={}) lde_size={lde_size} num_eval_points={num_eval_points}",
            i / lde_size,
            i % lde_size,
        );
    }
}

#[test]
#[ignore = "requires GPU; run with --ignored --nocapture"]
fn r4_denoms_parity_small() {
    run_parity(1 << 14, 2, 1);
    run_parity(1 << 14, 4, 2);
}

#[test]
#[ignore = "requires GPU; run with --ignored --nocapture"]
fn r4_denoms_parity_prover_shape() {
    // fib_iterative_1M / 4M LDE sizes with the common eval-point counts.
    run_parity(1 << 18, 2, 100);
    run_parity(1 << 20, 2, 101);
}
