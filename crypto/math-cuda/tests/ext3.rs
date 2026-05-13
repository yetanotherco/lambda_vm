//! Parity: GPU ext3 arithmetic must agree (canonically) with CPU
//! `Degree3GoldilocksExtensionField` on random ext3 inputs.

use math::field::element::FieldElement;
use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField;
use math::field::goldilocks::GoldilocksField;
use math::field::traits::{IsField, IsPrimeField};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

type Fp = FieldElement<GoldilocksField>;
type Fp3 = FieldElement<Degree3GoldilocksExtensionField>;

const N: usize = 10_000;

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

#[test]
fn ext3_mul_matches_cpu() {
    let a = random_fp3s(11, N);
    let b = random_fp3s(22, N);
    let a_raw = to_u64s(&a);
    let b_raw = to_u64s(&b);
    let gpu = math_cuda::ext3_mul_u64(&a_raw, &b_raw).unwrap();
    assert_eq!(gpu.len(), 3 * N);
    for i in 0..N {
        use math::field::traits::IsField;
        let cpu = Degree3GoldilocksExtensionField::mul(a[i].value(), b[i].value());
        let cpu_fp3 = Fp3::new(cpu);
        let g = canon_triplet_raw(&gpu[i * 3..(i + 1) * 3]);
        let c = canon_triplet(&cpu_fp3);
        assert_eq!(g, c, "ext3 mul mismatch at {i}");
    }
}

#[test]
fn ext3_add_matches_cpu() {
    let a = random_fp3s(33, N);
    let b = random_fp3s(44, N);
    let a_raw = to_u64s(&a);
    let b_raw = to_u64s(&b);
    let gpu = math_cuda::ext3_add_u64(&a_raw, &b_raw).unwrap();
    for i in 0..N {
        let cpu = Degree3GoldilocksExtensionField::add(a[i].value(), b[i].value());
        let cpu_fp3 = Fp3::new(cpu);
        let g = canon_triplet_raw(&gpu[i * 3..(i + 1) * 3]);
        let c = canon_triplet(&cpu_fp3);
        assert_eq!(g, c, "ext3 add mismatch at {i}");
    }
}
