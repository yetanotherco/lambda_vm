//! Parity for the direct natural-order DEEP -> first FRI fold handoff.

use math::fft::bit_reversing::in_place_bit_reverse_permute;
use math::field::element::FieldElement;
use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField;
use math::field::goldilocks::GoldilocksField;
use math::field::traits::IsPrimeField;
use math_cuda::deep::GpuDeepEvals;
use math_cuda::device::backend;
use math_cuda::fri::FriCommitState;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

type Fp = FieldElement<GoldilocksField>;
type Fp3 = FieldElement<Degree3GoldilocksExtensionField>;

fn rand_fp(rng: &mut ChaCha8Rng) -> Fp {
    Fp::from_raw(rng.r#gen::<u64>())
}

fn rand_fp3(rng: &mut ChaCha8Rng) -> Fp3 {
    Fp3::new([rand_fp(rng), rand_fp(rng), rand_fp(rng)])
}

fn raw3(e: &Fp3) -> [u64; 3] {
    [
        *e.value()[0].value(),
        *e.value()[1].value(),
        *e.value()[2].value(),
    ]
}

fn canonical3(e: &Fp3) -> [u64; 3] {
    [
        GoldilocksField::canonical(e.value()[0].value()),
        GoldilocksField::canonical(e.value()[1].value()),
        GoldilocksField::canonical(e.value()[2].value()),
    ]
}

fn run_parity(log_n: u32, seed: u64) {
    let n = 1usize << log_n;
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let natural: Vec<Fp3> = (0..n).map(|_| rand_fp3(&mut rng)).collect();
    let inv_tw: Vec<Fp> = (0..n / 2).map(|_| rand_fp(&mut rng)).collect();
    let zeta = rand_fp3(&mut rng);

    let mut expected = natural.clone();
    in_place_bit_reverse_permute(&mut expected);
    for j in 0..n / 2 {
        let lo = &expected[2 * j];
        let hi = &expected[2 * j + 1];
        let sum = lo + hi;
        let diff = lo - hi;
        expected[j] = &sum + &(&inv_tw[j] * &(&zeta * &diff));
    }
    expected.truncate(n / 2);

    let mut natural_raw = Vec::with_capacity(3 * n);
    for e in &natural {
        natural_raw.extend_from_slice(&raw3(e));
    }
    let inv_tw_raw: Vec<u64> = inv_tw.iter().map(|x| *x.value()).collect();

    let be = backend().unwrap();
    let stream = be.next_stream();
    let deep_buf = stream.clone_htod(&natural_raw).unwrap();
    stream.synchronize().unwrap();
    let deep = GpuDeepEvals {
        buf: deep_buf,
        len: n,
        stream,
    };
    let mut state = FriCommitState::from_deep(deep, &inv_tw_raw, n).unwrap();
    let (got_raw, _tree, _retained) = state
        .fold_and_commit_layer(raw3(&zeta), false, true)
        .unwrap();

    assert_eq!(got_raw.len(), expected.len() * 3);
    for (i, want) in expected.iter().enumerate() {
        let got = Fp3::new([
            Fp::from_raw(got_raw[i * 3]),
            Fp::from_raw(got_raw[i * 3 + 1]),
            Fp::from_raw(got_raw[i * 3 + 2]),
        ]);
        assert_eq!(
            canonical3(&got),
            canonical3(want),
            "first direct fold mismatch at log_n={log_n}, row={i}"
        );
    }
}

#[test]
fn direct_deep_first_fold_small() {
    for log_n in 3..=10 {
        run_parity(log_n, 0xD33F_0000 + log_n as u64);
    }
}

#[test]
fn direct_deep_first_fold_medium() {
    run_parity(16, 0xD33F_F001);
}
