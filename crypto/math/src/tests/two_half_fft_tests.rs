use crate::fft::bowers_fft::LayerTwiddles;
use crate::fft::bowers_fft_batch::{
    bowers_fft_batch_row_major, bowers_ifft_batch_row_major, in_place_bit_reverse_permute_row_major,
};
use crate::fft::two_half_fft::{TwoHalfTwiddles, fft_batch_two_half};
use crate::field::element::FieldElement;
use crate::field::goldilocks::GoldilocksField;
use alloc::vec::Vec;

type F = GoldilocksField;

fn reference_natural_fft(buf: &mut [FieldElement<F>], m: usize, log_n: usize) {
    let tw = LayerTwiddles::<F>::new(log_n as u64).unwrap();
    bowers_fft_batch_row_major::<F, F>(buf, m, &tw).unwrap();
    in_place_bit_reverse_permute_row_major(buf, m);
}

// Mirrors the LDE's iFFT: bit-reverse then flat Bowers inverse (no 1/n).
fn reference_natural_ifft(buf: &mut [FieldElement<F>], m: usize, log_n: usize) {
    let tw = LayerTwiddles::<F>::new_inverse(log_n as u64).unwrap();
    in_place_bit_reverse_permute_row_major(buf, m);
    bowers_ifft_batch_row_major::<F, F>(buf, m, &tw).unwrap();
}

fn sample(n: usize, m: usize) -> Vec<FieldElement<F>> {
    (0..n * m)
        .map(|i| FieldElement::<F>::from((i as u64).wrapping_mul(2654435761) ^ 0x9e37))
        .collect()
}

#[test]
fn two_half_matches_flat_bowers() {
    for log_n in [2usize, 3, 4, 5, 6, 8, 10] {
        for m in [1usize, 3, 7] {
            let n = 1 << log_n;
            let input = sample(n, m);
            let fwd_tw = TwoHalfTwiddles::<F>::new(log_n, false).unwrap();
            let inv_tw = TwoHalfTwiddles::<F>::new(log_n, true).unwrap();

            let mut a = input.clone();
            let mut c = input.clone();
            reference_natural_fft(&mut a, m, log_n);
            fft_batch_two_half::<F, F>(&mut c, m, &fwd_tw).unwrap();
            assert_eq!(a, c, "two_half fwd mismatch at log_n={log_n}, m={m}");

            let mut d = input.clone();
            let mut e = input.clone();
            reference_natural_ifft(&mut d, m, log_n);
            fft_batch_two_half::<F, F>(&mut e, m, &inv_tw).unwrap();
            assert_eq!(d, e, "two_half ifft mismatch at log_n={log_n}, m={m}");
        }
    }
}

/// Mismatched twiddle size must error rather than silently misbehave.
#[test]
fn wrong_twiddle_size_errors() {
    let m = 4;
    let mut buf = sample(1 << 6, m);
    let tw = TwoHalfTwiddles::<F>::new(5, false).unwrap();
    assert!(fft_batch_two_half::<F, F>(&mut buf, m, &tw).is_err());
}

/// Timing micro-bench (run with `--release --ignored --nocapture`).
#[test]
#[ignore]
fn bench_two_half_vs_flat() {
    use std::time::Instant;
    let m = 64;
    for log_n in [20usize, 21, 22, 23] {
        let n = 1 << log_n;
        let input = sample(n, m);
        let tw = LayerTwiddles::<F>::new(log_n as u64).unwrap();
        let two_tw = TwoHalfTwiddles::<F>::new(log_n, false).unwrap();

        let runs = 5;
        let mut t_flat = f64::INFINITY;
        let mut t_two = f64::INFINITY;
        for _ in 0..runs {
            let mut a = input.clone();
            let s = Instant::now();
            bowers_fft_batch_row_major::<F, F>(&mut a, m, &tw).unwrap();
            in_place_bit_reverse_permute_row_major(&mut a, m);
            t_flat = t_flat.min(s.elapsed().as_secs_f64());

            let mut c = input.clone();
            let s = Instant::now();
            fft_batch_two_half::<F, F>(&mut c, m, &two_tw).unwrap();
            t_two = t_two.min(s.elapsed().as_secs_f64());
        }
        println!(
            "log_n={log_n} m={m}: flat={:.4}s two_half={:.4}s  two/flat={:.2}x",
            t_flat,
            t_two,
            t_flat / t_two
        );
    }
}
