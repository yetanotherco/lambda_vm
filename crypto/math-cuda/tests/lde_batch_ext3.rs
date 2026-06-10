//! Ext3 batched coset LDE must agree with the CPU `coset_lde_full_expand`
//! on each column independently when run over `FieldElement<Ext3>`.

use math::fft::bowers_fft::LayerTwiddles;
use math::field::element::FieldElement;
use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField;
use math::field::goldilocks::GoldilocksField;
use math::field::traits::{IsField, IsPrimeField};
use math::polynomial::Polynomial;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

type Fp = FieldElement<GoldilocksField>;
type Fp3 = FieldElement<Degree3GoldilocksExtensionField>;

fn coset_weights(n: usize, g: u64) -> Vec<u64> {
    let inv_n = *FieldElement::<GoldilocksField>::from(n as u64)
        .inv()
        .unwrap()
        .value();
    let mut w = Vec::with_capacity(n);
    let mut cur = inv_n;
    for _ in 0..n {
        w.push(cur);
        cur = GoldilocksField::mul(&cur, &g);
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
    // Each Fp3 is [u64; 3] in memory; we just flatten componentwise.
    let mut out = Vec::with_capacity(col.len() * 3);
    for e in col {
        out.push(*e.value()[0].value());
        out.push(*e.value()[1].value());
        out.push(*e.value()[2].value());
    }
    out
}

fn u64s_to_ext3(raw: &[u64]) -> Vec<Fp3> {
    assert_eq!(raw.len() % 3, 0);
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

fn cpu_lde_one_ext3(
    col: &[Fp3],
    blowup: usize,
    weights_fp: &[Fp],
    inv_tw: &LayerTwiddles<GoldilocksField>,
    fwd_tw: &LayerTwiddles<GoldilocksField>,
) -> Vec<Fp3> {
    let mut buf = col.to_vec();
    Polynomial::coset_lde_full_expand::<GoldilocksField>(
        &mut buf, blowup, weights_fp, inv_tw, fwd_tw,
    )
    .unwrap();
    buf
}

fn canon(xs: &[u64]) -> Vec<u64> {
    xs.iter().map(GoldilocksField::canonical).collect()
}

fn assert_ext3_batch(log_n: u64, blowup: usize, m: usize, seed: u64) {
    let n = 1usize << log_n;
    let lde_size = n * blowup;
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let columns: Vec<Vec<Fp3>> = (0..m)
        .map(|_| (0..n).map(|_| rand_ext3(&mut rng)).collect())
        .collect();

    let coset_offset: u64 = 7;
    let weights = coset_weights(n, coset_offset);
    let weights_fp: Vec<Fp> = weights.iter().map(|&w| Fp::from_raw(w)).collect();
    let inv_tw = LayerTwiddles::<GoldilocksField>::new_inverse(log_n).unwrap();
    let fwd_tw = LayerTwiddles::<GoldilocksField>::new(lde_size.trailing_zeros() as u64).unwrap();

    // Flatten each ext3 column to 3n u64s for the GPU API.
    let flat_inputs: Vec<Vec<u64>> = columns.iter().map(|c| ext3_to_u64s(c)).collect();
    let input_slices: Vec<&[u64]> = flat_inputs.iter().map(|v| v.as_slice()).collect();

    // Pre-allocate outputs, each 3*lde_size u64s.
    let mut flat_outputs: Vec<Vec<u64>> = (0..m).map(|_| vec![0u64; 3 * lde_size]).collect();
    {
        let mut out_slices: Vec<&mut [u64]> =
            flat_outputs.iter_mut().map(|v| v.as_mut_slice()).collect();
        math_cuda::lde::coset_lde_batch_ext3_into(
            &input_slices,
            n,
            blowup,
            &weights,
            &mut out_slices,
        )
        .unwrap();
    }

    for (c, col) in columns.iter().enumerate() {
        let cpu = cpu_lde_one_ext3(col, blowup, &weights_fp, &inv_tw, &fwd_tw);
        let gpu: Vec<Fp3> = u64s_to_ext3(&flat_outputs[c]);
        assert_eq!(gpu.len(), cpu.len(), "length mismatch");
        for i in 0..cpu.len() {
            for k in 0..3 {
                let cv = *cpu[i].value()[k].value();
                let gv = *gpu[i].value()[k].value();
                let cc = GoldilocksField::canonical(&cv);
                let gc = GoldilocksField::canonical(&gv);
                if cc != gc {
                    panic!(
                        "ext3 batch mismatch col={c} row={i} comp={k} log_n={log_n} blowup={blowup}: cpu={cv:#018x} (canon {cc:#018x}), gpu={gv:#018x} (canon {gc:#018x})",
                    );
                }
            }
        }
    }
    // Also sanity-check raw canonical equality per column.
    for (c, col) in columns.iter().enumerate() {
        let cpu_raw = ext3_to_u64s(&cpu_lde_one_ext3(
            col,
            blowup,
            &weights_fp,
            &inv_tw,
            &fwd_tw,
        ));
        assert_eq!(canon(&cpu_raw), canon(&flat_outputs[c]));
    }
}

#[test]
fn ext3_batch_small() {
    for &m in &[1usize, 4, 16] {
        for log_n in 4..=10 {
            assert_ext3_batch(log_n, 4, m, 100 + log_n * 10 + m as u64);
        }
    }
}

#[test]
fn ext3_batch_medium() {
    for &m in &[2usize, 8] {
        for log_n in 11..=14 {
            assert_ext3_batch(log_n, 4, m, 300 + log_n * 10 + m as u64);
        }
    }
}

#[test]
fn ext3_batch_large_one_column() {
    assert_ext3_batch(16, 4, 1, 0xCAFE);
}
