//! Parity: GPU parallel batch inverse matches CPU
//! `FieldElement::inplace_batch_inverse` on ext3 elements.
//!
//! Sizes span:
//!  - n=1 (host-only path)
//!  - n in {2..256} small (single-block scan)
//!  - n in {257..2^17} medium (multi-block, single recursion)
//!  - n=2^20, 2^22 large (multi-block, two-level recursion)

use math::field::element::FieldElement;
use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField;
use math::field::goldilocks::GoldilocksField;
use math::field::traits::IsPrimeField;
use math_cuda::inverse::batch_inverse_ext3;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

type Fp = FieldElement<GoldilocksField>;
type Fp3 = FieldElement<Degree3GoldilocksExtensionField>;

fn rand_fp(rng: &mut ChaCha8Rng) -> Fp {
    loop {
        let v = rng.r#gen::<u64>();
        if v != 0 {
            return Fp::from_raw(v);
        }
    }
}

fn rand_fp3_nonzero(rng: &mut ChaCha8Rng) -> Fp3 {
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

fn run(n: usize, seed: u64) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let xs: Vec<Fp3> = (0..n).map(|_| rand_fp3_nonzero(&mut rng)).collect();

    let mut cpu = xs.clone();
    FieldElement::inplace_batch_inverse(&mut cpu).expect("batch inverse non-zero");

    let input_u64 = ext3_to_u64s(&xs);
    let gpu_u64 = batch_inverse_ext3(&input_u64).unwrap();

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
fn batch_inverse_n1() {
    // Host-only special case.
    run(1, 1);
}

/// `batch_inverse_ext3_dev`'s own `n == 1` branch, which the host entry point
/// above never reaches: `batch_inverse_ext3` short-circuits n==1 to
/// `invert_ext3_host`, so only a direct device call exercises the single
/// `invert_total_ext3` launch that serves this case.
#[test]
fn batch_inverse_dev_n1() {
    let mut rng = ChaCha8Rng::seed_from_u64(7);
    let x = rand_fp3_nonzero(&mut rng);
    let expected = x.inv().expect("nonzero is invertible");

    let be = math_cuda::device::backend().expect("cuda backend");
    let stream = be.next_stream();
    let input = stream.clone_htod(&ext3_to_u64s(&[x])).unwrap();

    let out_dev = math_cuda::inverse::batch_inverse_ext3_dev(&input, 1, &stream).unwrap();
    let got = stream.clone_dtoh(&out_dev).unwrap();
    stream.synchronize().unwrap();

    assert_eq!(
        canon3(&got),
        canon3(&ext3_to_u64s(&[expected])),
        "device n==1 inverse"
    );
}

#[test]
fn batch_inverse_single_block() {
    // All single-block sizes (no recursion).
    for n in [2usize, 3, 5, 16, 63, 127, 255, 256] {
        run(n, 100 + n as u64);
    }
}

#[test]
fn batch_inverse_two_block() {
    // Just over single-block: forces phase 1 + 3 with K = 2.
    for n in [257usize, 511, 512, 513, 1024] {
        run(n, 200 + n as u64);
    }
}

#[test]
fn batch_inverse_multi_block() {
    // Multi-block, single level of recursion (K > 1, K <= 256).
    for n in [4096usize, 16384, 65536] {
        run(n, 500 + n as u64);
    }
}

#[test]
fn batch_inverse_recursive() {
    // K > 256: forces two levels of recursion. fib_iterative_1M
    // (lde_size=2^20) and fib_iterative_4M (lde_size=2^22) shapes.
    run(1 << 18, 9001);
    run(1 << 20, 9002);
    run(1 << 22, 9003);
}
