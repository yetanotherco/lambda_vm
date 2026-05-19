//! Parity: GPU forward NTT must agree with `Polynomial::evaluate_fft`
//! as a field element, across a sweep of sizes from 2^4 to 2^20.
//!
//! Non-canonical u64s can differ between CPU and GPU while representing the
//! same element; we canonicalise both sides before comparing.

use math::field::element::FieldElement;
use math::field::goldilocks::GoldilocksField;
use math::field::traits::IsPrimeField;
use math::polynomial::Polynomial;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

type Fp = FieldElement<GoldilocksField>;

fn cpu_fft(coeffs: &[u64]) -> Vec<u64> {
    let elems: Vec<Fp> = coeffs.iter().map(|&x| Fp::from_raw(x)).collect();
    let poly = Polynomial::new(&elems);
    let evals = Polynomial::evaluate_fft::<GoldilocksField>(&poly, 1, None).expect("cpu fft");
    evals.into_iter().map(|e| *e.value()).collect()
}

fn canonicalize(xs: &[u64]) -> Vec<u64> {
    xs.iter().map(GoldilocksField::canonical).collect()
}

fn assert_ntt_match(log_n: u64, seed: u64) {
    let n = 1usize << log_n;
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let input: Vec<u64> = (0..n).map(|_| rng.r#gen::<u64>()).collect();

    let cpu = cpu_fft(&input);
    let gpu = math_cuda::ntt::forward(&input).expect("gpu ntt");

    assert_eq!(cpu.len(), gpu.len(), "length mismatch at log_n = {log_n}");
    let cpu_c = canonicalize(&cpu);
    let gpu_c = canonicalize(&gpu);
    for i in 0..n {
        if cpu_c[i] != gpu_c[i] {
            panic!(
                "log_n={log_n} i={i}: cpu={:#018x} (canon {:#018x}), gpu={:#018x} (canon {:#018x})",
                cpu[i], cpu_c[i], gpu[i], gpu_c[i],
            );
        }
    }
}

#[test]
fn ntt_sizes_small() {
    for log_n in 4..=10 {
        assert_ntt_match(log_n, 100 + log_n);
    }
}

#[test]
fn ntt_sizes_medium() {
    for log_n in 11..=16 {
        assert_ntt_match(log_n, 200 + log_n);
    }
}

#[test]
fn ntt_size_2_to_20() {
    assert_ntt_match(20, 0xDEAD);
}

#[test]
fn ntt_trivial_sizes() {
    assert_ntt_match(1, 1);
    assert_ntt_match(2, 2);
    assert_ntt_match(3, 3);
}

fn assert_intt_match(log_n: u64, seed: u64) {
    let n = 1usize << log_n;
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let evals: Vec<u64> = (0..n).map(|_| rng.r#gen::<u64>()).collect();

    let elems: Vec<Fp> = evals.iter().map(|&x| Fp::from_raw(x)).collect();
    let cpu_poly = Polynomial::interpolate_fft::<GoldilocksField>(&elems).expect("cpu intt");
    let cpu: Vec<u64> = cpu_poly
        .coefficients
        .into_iter()
        .map(|e| *e.value())
        .collect();

    let gpu = math_cuda::ntt::inverse(&evals).expect("gpu intt");

    let cpu_c = canonicalize(&cpu);
    let gpu_c = canonicalize(&gpu);
    for i in 0..n {
        if cpu_c[i] != gpu_c[i] {
            panic!(
                "iNTT log_n={log_n} i={i}: cpu canon {:#018x}, gpu canon {:#018x}",
                cpu_c[i], gpu_c[i],
            );
        }
    }
}

#[test]
fn intt_sizes_small() {
    for log_n in 4..=10 {
        assert_intt_match(log_n, 700 + log_n);
    }
}

#[test]
fn intt_sizes_medium() {
    for log_n in 11..=16 {
        assert_intt_match(log_n, 800 + log_n);
    }
}

#[test]
fn intt_size_2_to_20() {
    assert_intt_match(20, 0xBEEF);
}

#[test]
fn ntt_round_trip() {
    // inverse(forward(x)) == x up to canonical form.
    let log_n = 14;
    let n = 1usize << log_n;
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let x: Vec<u64> = (0..n)
        .map(|_| rng.r#gen::<u64>() % 0xFFFF_FFFF_0000_0001)
        .collect();

    let evals = math_cuda::ntt::forward(&x).expect("forward");
    let back = math_cuda::ntt::inverse(&evals).expect("inverse");

    let x_c = canonicalize(&x);
    let back_c = canonicalize(&back);
    assert_eq!(x_c, back_c, "round trip failed");
}
