//! Parity: GPU `coset_lde_base` must match the CPU
//! `Polynomial::coset_lde_full_expand` for a sweep of realistic sizes and
//! blowup factors.

use math::fft::bowers_fft::LayerTwiddles;
use math::field::element::FieldElement;
use math::field::goldilocks::GoldilocksField;
use math::field::traits::{IsField, IsPrimeField};
use math::polynomial::Polynomial;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

type Fp = FieldElement<GoldilocksField>;

/// Build the coset weights `[1/N, g/N, g²/N, ..., g^{n-1}/N]` — this is the
/// layout `crypto/stark/src/prover.rs` uses, with `1/N` pre-folded into the
/// first coefficient so the iFFT step does not need a separate scaling pass.
fn coset_weights(n: usize, coset_offset: u64) -> Vec<u64> {
    let inv_n_fe = FieldElement::<GoldilocksField>::from(n as u64)
        .inv()
        .expect("n is non-zero");
    let mut w = Vec::with_capacity(n);
    let mut cur = *inv_n_fe.value();
    for _ in 0..n {
        w.push(cur);
        cur = GoldilocksField::mul(&cur, &coset_offset);
    }
    w
}

fn cpu_lde(evals: &[u64], blowup_factor: usize, coset_offset: u64) -> Vec<u64> {
    let n = evals.len();
    let log_n = n.trailing_zeros() as u64;
    let log_lde = (n * blowup_factor).trailing_zeros() as u64;

    let inv_tw = LayerTwiddles::<GoldilocksField>::new_inverse(log_n).expect("inv tw");
    let fwd_tw = LayerTwiddles::<GoldilocksField>::new(log_lde).expect("fwd tw");
    let weights_raw = coset_weights(n, coset_offset);
    let weights: Vec<Fp> = weights_raw.iter().map(|&w| Fp::from_raw(w)).collect();

    let mut buf: Vec<Fp> = evals.iter().map(|&x| Fp::from_raw(x)).collect();
    Polynomial::coset_lde_full_expand::<GoldilocksField>(
        &mut buf,
        blowup_factor,
        &weights,
        &inv_tw,
        &fwd_tw,
    )
    .expect("cpu lde");

    buf.into_iter().map(|e| *e.value()).collect()
}

fn canon(xs: &[u64]) -> Vec<u64> {
    xs.iter().map(GoldilocksField::canonical).collect()
}

fn assert_lde_match(log_n: u64, blowup_factor: usize, seed: u64) {
    let n = 1usize << log_n;
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let evals: Vec<u64> = (0..n).map(|_| rng.r#gen::<u64>()).collect();

    // Use a fixed, public coset offset. For lambda-vm the coset offset is the
    // generator of Goldilocks' multiplicative subgroup; any non-trivial element
    // works for an isolated correctness check.
    let coset_offset: u64 = 7;
    let weights = coset_weights(n, coset_offset);

    let cpu = cpu_lde(&evals, blowup_factor, coset_offset);
    let gpu = math_cuda::lde::coset_lde_base(&evals, blowup_factor, &weights).expect("gpu lde");

    assert_eq!(
        cpu.len(),
        gpu.len(),
        "length mismatch (log_n={log_n}, blowup={blowup_factor})"
    );
    let cpu_c = canon(&cpu);
    let gpu_c = canon(&gpu);
    for (i, (e, a)) in cpu_c.iter().zip(&gpu_c).enumerate() {
        if e != a {
            panic!(
                "lde mismatch log_n={log_n} blowup={blowup_factor} i={i}: cpu {e:#018x}, gpu {a:#018x}",
            );
        }
    }
}

#[test]
fn lde_small() {
    for log_n in 4..=10 {
        for &blow in &[2usize, 4, 8] {
            assert_lde_match(log_n, blow, 1_000 + log_n + (blow as u64));
        }
    }
}

#[test]
fn lde_medium() {
    for log_n in 11..=14 {
        for &blow in &[2usize, 4] {
            assert_lde_match(log_n, blow, 2_000 + log_n + (blow as u64));
        }
    }
}

#[test]
fn lde_large_2_to_18() {
    // 2^18 × blowup 4 = 2^20 LDE — representative of Phase A trace columns.
    assert_lde_match(18, 4, 0xCAFE);
}

#[test]
fn lde_largest_2_to_20() {
    // 2^20 LDE is the hot size; blowup 2 keeps total = 2^21 (within TWO_ADICITY).
    assert_lde_match(20, 2, 0xF00D);
}
