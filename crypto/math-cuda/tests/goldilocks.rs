//! GPU must produce bit-identical u64 outputs to `GoldilocksField` for every op.
//! Non-canonical inputs are expected (CPU operates on the full [0, 2^64) range),
//! so the test inputs include values above the prime.

use math::field::goldilocks::GoldilocksField;
use math::field::traits::{IsField, IsPrimeField};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

const N: usize = 10_000;

fn sample_inputs(seed: u64) -> Vec<u64> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    (0..N).map(|_| rng.r#gen::<u64>()).collect()
}

fn assert_raw_eq(op: &str, expected: &[u64], actual: &[u64]) {
    assert_eq!(expected.len(), actual.len());
    for (i, (e, a)) in expected.iter().zip(actual.iter()).enumerate() {
        if e != a {
            panic!(
                "{op} mismatch at {i}: cpu={e:#018x} (canon {:#018x}), gpu={a:#018x} (canon {:#018x})",
                GoldilocksField::canonical(e),
                GoldilocksField::canonical(a),
            );
        }
    }
}

#[test]
fn gpu_vector_add_u64_matches_wrapping() {
    let a = sample_inputs(0xC0FFEE);
    let b = sample_inputs(0xDEADBEEF);
    let expected: Vec<u64> = a.iter().zip(&b).map(|(x, y)| x.wrapping_add(*y)).collect();
    let actual = math_cuda::vector_add_u64(&a, &b).expect("GPU vector_add_u64");
    assert_raw_eq("vector_add (wrapping)", &expected, &actual);
}

#[test]
fn gpu_gl_add_matches_cpu() {
    let a = sample_inputs(1);
    let b = sample_inputs(2);
    let expected: Vec<u64> = a
        .iter()
        .zip(&b)
        .map(|(x, y)| GoldilocksField::add(x, y))
        .collect();
    let actual = math_cuda::gl_add_u64(&a, &b).expect("GPU gl_add");
    assert_raw_eq("gl_add", &expected, &actual);
}

#[test]
fn gpu_gl_sub_matches_cpu() {
    let a = sample_inputs(3);
    let b = sample_inputs(4);
    let expected: Vec<u64> = a
        .iter()
        .zip(&b)
        .map(|(x, y)| GoldilocksField::sub(x, y))
        .collect();
    let actual = math_cuda::gl_sub_u64(&a, &b).expect("GPU gl_sub");
    assert_raw_eq("gl_sub", &expected, &actual);
}

#[test]
fn gpu_gl_mul_matches_cpu() {
    let a = sample_inputs(5);
    let b = sample_inputs(6);
    let expected: Vec<u64> = a
        .iter()
        .zip(&b)
        .map(|(x, y)| GoldilocksField::mul(x, y))
        .collect();
    let actual = math_cuda::gl_mul_u64(&a, &b).expect("GPU gl_mul");
    assert_raw_eq("gl_mul", &expected, &actual);
}

#[test]
fn gpu_gl_neg_matches_cpu() {
    let a = sample_inputs(7);
    let expected: Vec<u64> = a.iter().map(GoldilocksField::neg).collect();
    let actual = math_cuda::gl_neg_u64(&a).expect("GPU gl_neg");
    assert_raw_eq("gl_neg", &expected, &actual);
}

/// Edge cases the random generator is unlikely to hit: 0, 1, p-1, p, p+1, 2p-1,
/// u64::MAX, EPSILON boundary values. Covers double-overflow / double-underflow.
#[test]
fn gpu_goldilocks_edge_cases() {
    const P: u64 = 0xFFFF_FFFF_0000_0001;
    const EPS: u64 = 0xFFFF_FFFF;
    let edge: [u64; 11] = [
        0,
        1,
        P - 1,
        P,
        P + 1,
        2u64.wrapping_mul(P).wrapping_sub(1),
        u64::MAX,
        u64::MAX - EPS,
        u64::MAX - 1,
        EPS,
        EPS - 1,
    ];
    // All pairs via nested loops, materialised as flat a[], b[] of length edge^2.
    let mut a = Vec::with_capacity(edge.len() * edge.len());
    let mut b = Vec::with_capacity(edge.len() * edge.len());
    for &x in &edge {
        for &y in &edge {
            a.push(x);
            b.push(y);
        }
    }

    type GpuOp = fn(&[u64], &[u64]) -> math_cuda::Result<Vec<u64>>;
    type CpuOp = fn(&u64, &u64) -> u64;
    let cases: &[(&str, GpuOp, CpuOp)] = &[
        ("gl_add", math_cuda::gl_add_u64, GoldilocksField::add),
        ("gl_sub", math_cuda::gl_sub_u64, GoldilocksField::sub),
        ("gl_mul", math_cuda::gl_mul_u64, GoldilocksField::mul),
    ];

    for (op, gpu_fn, cpu_fn) in cases {
        let expected: Vec<u64> = a.iter().zip(&b).map(|(x, y)| cpu_fn(x, y)).collect();
        let actual = gpu_fn(&a, &b).expect("GPU op");
        assert_raw_eq(op, &expected, &actual);
    }
}
