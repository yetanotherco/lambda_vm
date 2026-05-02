//! Parity: GPU parallel batch inverse matches CPU
//! `FieldElement::inplace_batch_inverse` on ext3 elements.

use math::field::element::FieldElement;
use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField;
use math::field::goldilocks::GoldilocksField;
use math::field::traits::{IsField, IsPrimeField};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

type Fp = FieldElement<GoldilocksField>;
type Fp3 = FieldElement<Degree3GoldilocksExtensionField>;

fn rand_fp(rng: &mut ChaCha8Rng) -> Fp {
    loop {
        let v = rng.r#gen::<u64>();
        // Avoid zero — batch inverse requires all non-zero.
        if v != 0 {
            return Fp::from_raw(v);
        }
    }
}
fn rand_fp3_nonzero(rng: &mut ChaCha8Rng) -> Fp3 {
    // Random non-zero ext3: at least one component non-zero, all in [1, p).
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
    a.iter()
        .enumerate()
        .map(|(i, v)| {
            // Each u64 is canonicalised independently (ext3 = 3 base coords).
            let _ = i;
            GoldilocksField::canonical(v)
        })
        .collect()
}

fn run(n: usize, seed: u64) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let xs: Vec<Fp3> = (0..n).map(|_| rand_fp3_nonzero(&mut rng)).collect();

    // CPU reference: inplace_batch_inverse.
    let mut cpu = xs.clone();
    FieldElement::inplace_batch_inverse(&mut cpu).expect("batch inverse non-zero");

    // GPU.
    let input_u64 = ext3_to_u64s(&xs);
    let gpu_u64 = math_cuda::inverse::batch_inverse_ext3(&input_u64).unwrap();

    let cpu_u64 = ext3_to_u64s(&cpu);
    let gpu_canon = canon3(&gpu_u64);
    let cpu_canon = canon3(&cpu_u64);

    for i in 0..n {
        let g = &gpu_canon[i * 3..(i + 1) * 3];
        let c = &cpu_canon[i * 3..(i + 1) * 3];
        assert_eq!(g, c, "mismatch at i={i} n={n}");
    }
}

#[test]
fn batch_inverse_small() {
    for n in [2usize, 3, 5, 16, 63, 255, 256, 257] {
        run(n, 100 + n as u64);
    }
}

#[test]
fn batch_inverse_medium() {
    for n in [1024usize, 4096, 8192] {
        run(n, 500 + n as u64);
    }
}

#[test]
fn batch_inverse_large() {
    // Matches R3 OOD / R4 DEEP sizes for fib_1M (domain_size = 2^18,
    // num_denoms_max = 2^18 × 4).
    run(1 << 18, 999);
    run(1 << 20, 12345);
}
