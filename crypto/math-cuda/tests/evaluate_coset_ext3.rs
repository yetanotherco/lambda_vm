//! Parity test for `evaluate_poly_coset_batch_ext3_into`.
//!
//! Reference: `math::polynomial::Polynomial::evaluate_offset_fft` on an ext3
//! polynomial, then canonicalise. The GPU path should produce the same
//! evaluations on the offset-coset at `n * blowup` points.

use math::field::element::FieldElement;
use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField;
use math::field::goldilocks::GoldilocksField;
use math::field::traits::{IsField, IsPrimeField};
use math::polynomial::Polynomial;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

type Fp = FieldElement<GoldilocksField>;
type Fp3 = FieldElement<Degree3GoldilocksExtensionField>;

fn offset_weights(n: usize, offset: u64) -> Vec<u64> {
    let mut w = Vec::with_capacity(n);
    let mut cur = 1u64;
    for _ in 0..n {
        w.push(cur);
        cur = GoldilocksField::mul(&cur, &offset);
    }
    w
}

fn rand_ext3(rng: &mut ChaCha8Rng) -> Fp3 {
    Fp3::new([
        Fp::from_raw(rng.r#gen::<u64>()),
        Fp::from_raw(rng.r#gen::<u64>()),
        Fp::from_raw(rng.r#gen::<u64>()),
    ])
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

fn u64s_to_ext3(raw: &[u64]) -> Vec<Fp3> {
    let mut out = Vec::with_capacity(raw.len() / 3);
    for i in 0..raw.len() / 3 {
        out.push(Fp3::new([
            Fp::from_raw(raw[i * 3]),
            Fp::from_raw(raw[i * 3 + 1]),
            Fp::from_raw(raw[i * 3 + 2]),
        ]));
    }
    out
}

fn canon_fp3(e: &Fp3) -> [u64; 3] {
    [
        GoldilocksField::canonical(e.value()[0].value()),
        GoldilocksField::canonical(e.value()[1].value()),
        GoldilocksField::canonical(e.value()[2].value()),
    ]
}

fn assert_evaluate_coset(log_n: u64, blowup: usize, m: usize, offset: u64, seed: u64) {
    let n = 1usize << log_n;
    let lde_size = n * blowup;
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    // M ext3 polynomials, each of degree < n.
    let polys: Vec<Vec<Fp3>> = (0..m)
        .map(|_| (0..n).map(|_| rand_ext3(&mut rng)).collect())
        .collect();

    let weights = offset_weights(n, offset);

    // CPU reference: evaluate each polynomial at `offset`-coset of size lde_size.
    let offset_fp = Fp::from_raw(offset);
    let cpu: Vec<Vec<Fp3>> = polys
        .iter()
        .map(|coefs| {
            let p = Polynomial::new(coefs);
            Polynomial::evaluate_offset_fft::<GoldilocksField>(&p, blowup, Some(n), &offset_fp)
                .unwrap()
        })
        .collect();

    // GPU: flatten each poly to 3n u64s, pre-allocate 3*lde_size u64 outputs.
    let flat_inputs: Vec<Vec<u64>> = polys.iter().map(|p| ext3_to_u64s(p)).collect();
    let input_slices: Vec<&[u64]> = flat_inputs.iter().map(|v| v.as_slice()).collect();
    let mut flat_outputs: Vec<Vec<u64>> = (0..m).map(|_| vec![0u64; 3 * lde_size]).collect();
    {
        let mut out_slices: Vec<&mut [u64]> =
            flat_outputs.iter_mut().map(|v| v.as_mut_slice()).collect();
        math_cuda::lde::evaluate_poly_coset_batch_ext3_into(
            &input_slices,
            n,
            blowup,
            &weights,
            &mut out_slices,
        )
        .unwrap();
    }

    for c in 0..m {
        let gpu: Vec<Fp3> = u64s_to_ext3(&flat_outputs[c]);
        assert_eq!(gpu.len(), cpu[c].len(), "length mismatch");
        for i in 0..gpu.len() {
            let g = canon_fp3(&gpu[i]);
            let cc = canon_fp3(&cpu[c][i]);
            assert_eq!(
                g, cc,
                "eval mismatch col={c} row={i} log_n={log_n} blowup={blowup}"
            );
        }
    }
}

#[test]
fn ext3_evaluate_coset_small() {
    for &m in &[1usize, 4] {
        for log_n in 4..=10 {
            for &blowup in &[2usize, 4] {
                assert_evaluate_coset(log_n, blowup, m, 7, 100 + log_n * 10 + m as u64);
            }
        }
    }
}

#[test]
fn ext3_evaluate_coset_medium() {
    for log_n in 11..=14 {
        assert_evaluate_coset(log_n, 4, 2, 7, 200 + log_n);
    }
}

#[test]
fn ext3_evaluate_coset_large_one_column() {
    assert_evaluate_coset(16, 4, 1, 7, 0xCAFE);
}
