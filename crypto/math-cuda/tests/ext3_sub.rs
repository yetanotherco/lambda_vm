//! Parity test for `ext3::sub` in kernels/ext3.cuh. This device
//! function is part of the public ext3 header but is not invoked by any
//! kernel in the PR — every other ext3 caller uses `mul`, `add`, or
//! `mul_base`. The PR's review test infrastructure adds an
//! `ext3_sub_kernel` so we can call it directly here for parity vs the
//! CPU `Degree3GoldilocksExtensionField::sub`.

use math::field::element::FieldElement;
use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField;
use math::field::goldilocks::GoldilocksField;
use math::field::traits::{IsField, IsPrimeField};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

type Fp = FieldElement<GoldilocksField>;
type Fp3 = FieldElement<Degree3GoldilocksExtensionField>;

const N: usize = 10_000;
const P: u64 = 0xFFFF_FFFF_0000_0001;

fn random_fp3s(seed: u64, count: usize) -> Vec<Fp3> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    (0..count)
        .map(|_| {
            Fp3::new([
                Fp::from_raw(rng.r#gen::<u64>()),
                Fp::from_raw(rng.r#gen::<u64>()),
                Fp::from_raw(rng.r#gen::<u64>()),
            ])
        })
        .collect()
}

fn to_u64s(col: &[Fp3]) -> Vec<u64> {
    let mut v = Vec::with_capacity(col.len() * 3);
    for e in col {
        v.push(*e.value()[0].value());
        v.push(*e.value()[1].value());
        v.push(*e.value()[2].value());
    }
    v
}

fn canon(x: u64) -> u64 {
    GoldilocksField::canonical(&x)
}

#[test]
fn ext3_sub_matches_cpu_random() {
    let a = random_fp3s(101, N);
    let b = random_fp3s(202, N);
    let a_raw = to_u64s(&a);
    let b_raw = to_u64s(&b);
    let gpu = math_cuda::ext3_sub_u64(&a_raw, &b_raw).expect("GPU ext3 sub launch");
    assert_eq!(gpu.len(), 3 * N);
    for i in 0..N {
        let cpu = Degree3GoldilocksExtensionField::sub(a[i].value(), b[i].value());
        let cpu_fp3 = Fp3::new(cpu);
        let g = [
            canon(gpu[3 * i]),
            canon(gpu[3 * i + 1]),
            canon(gpu[3 * i + 2]),
        ];
        let c = [
            canon(*cpu_fp3.value()[0].value()),
            canon(*cpu_fp3.value()[1].value()),
            canon(*cpu_fp3.value()[2].value()),
        ];
        assert_eq!(g, c, "ext3 sub mismatch at {i}");
    }
}

#[test]
fn ext3_sub_edge_cases() {
    // Underflow cases: a < b on each component, plus non-canonical p
    // representations.
    let cases: Vec<([u64; 3], [u64; 3])> = vec![
        ([0, 0, 0], [P - 1, P - 1, P - 1]),
        ([1, 2, 3], [P - 1, P - 1, P - 1]),
        ([P - 1, P - 1, P - 1], [0, 0, 0]),
        ([P, P, P], [P, P, P]), // (0,0,0) - (0,0,0)
        ([u64::MAX, u64::MAX, u64::MAX], [0, 0, 0]),
        ([0, 0, 0], [u64::MAX, u64::MAX, u64::MAX]),
    ];
    let mut a_raw = Vec::new();
    let mut b_raw = Vec::new();
    for (a, b) in &cases {
        a_raw.extend_from_slice(a);
        b_raw.extend_from_slice(b);
    }
    let gpu = math_cuda::ext3_sub_u64(&a_raw, &b_raw).expect("GPU ext3 sub launch");
    for (i, (a, b)) in cases.iter().enumerate() {
        let ae = [Fp::from_raw(a[0]), Fp::from_raw(a[1]), Fp::from_raw(a[2])];
        let be = [Fp::from_raw(b[0]), Fp::from_raw(b[1]), Fp::from_raw(b[2])];
        let cpu = Degree3GoldilocksExtensionField::sub(&ae, &be);
        let cpu_fp3 = Fp3::new(cpu);
        let g = [
            canon(gpu[3 * i]),
            canon(gpu[3 * i + 1]),
            canon(gpu[3 * i + 2]),
        ];
        let c = [
            canon(*cpu_fp3.value()[0].value()),
            canon(*cpu_fp3.value()[1].value()),
            canon(*cpu_fp3.value()[2].value()),
        ];
        assert_eq!(g, c, "ext3 sub edge mismatch at {i}: a={a:?} b={b:?}");
    }
}
