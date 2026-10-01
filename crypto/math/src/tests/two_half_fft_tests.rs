use crate::fft::bit_reversing::in_place_bit_reverse_permute;
use crate::fft::bowers_fft::{LayerTwiddles, bowers_fft_opt_fused, bowers_ifft_opt};
use crate::fft::two_half_fft::{TwoHalfTwiddles, fft_batch_two_half};
use crate::field::element::FieldElement;
use crate::field::goldilocks::GoldilocksField;
use alloc::vec::Vec;

type F = GoldilocksField;

/// Apply a single-column transform `f` independently to each of the `m`
/// columns of a flat `n * m` row-major buffer. The single-column `bowers_fft`
/// is the same algorithm the batched row-major FFT mirrors, so it is the
/// reference oracle for `fft_batch_two_half` (the LDE differential test already
/// proves the row-major transpose-compare end to end).
fn per_column<G: FnMut(&mut Vec<FieldElement<F>>)>(
    buf: &mut [FieldElement<F>],
    m: usize,
    n: usize,
    mut f: G,
) {
    for col in 0..m {
        let mut c: Vec<FieldElement<F>> = (0..n).map(|r| buf[r * m + col]).collect();
        f(&mut c);
        for (r, v) in c.into_iter().enumerate() {
            buf[r * m + col] = v;
        }
    }
}

/// Natural-order forward FFT, per column, via the single-column Bowers FFT
/// (DIF → bit-reversed) followed by the bit-reverse permute back to natural
/// order. Matches `fft_batch_two_half` (forward).
fn reference_natural_fft(buf: &mut [FieldElement<F>], m: usize, log_n: usize) {
    let n = 1usize << log_n;
    let tw = LayerTwiddles::<F>::new(log_n as u64).unwrap();
    per_column(buf, m, n, |c| {
        bowers_fft_opt_fused::<F, F>(c, &tw).unwrap();
        in_place_bit_reverse_permute(c);
    });
}

/// Mirrors the LDE's iFFT: bit-reverse then the single-column Bowers inverse
/// (DIT, no 1/n). Matches `fft_batch_two_half` (inverse).
fn reference_natural_ifft(buf: &mut [FieldElement<F>], m: usize, log_n: usize) {
    let n = 1usize << log_n;
    let tw = LayerTwiddles::<F>::new_inverse(log_n as u64).unwrap();
    per_column(buf, m, n, |c| {
        in_place_bit_reverse_permute(c);
        bowers_ifft_opt::<F, F>(c, &tw).unwrap();
    });
}

fn sample(n: usize, m: usize) -> Vec<FieldElement<F>> {
    (0..n * m)
        .map(|i| FieldElement::<F>::from((i as u64).wrapping_mul(2654435761) ^ 0x9e37))
        .collect()
}

#[test]
fn two_half_matches_single_column() {
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

/// Timing micro-bench (run with `--release --ignored --nocapture`). Compares
/// the batched two-half FFT against the per-column single-column FFT — the
/// path the LDE used before the row-major rework.
#[test]
#[ignore]
fn bench_two_half_vs_single_column() {
    use std::time::Instant;
    let m = 64;
    for log_n in [20usize, 21, 22, 23] {
        let n = 1 << log_n;
        let input = sample(n, m);
        let two_tw = TwoHalfTwiddles::<F>::new(log_n, false).unwrap();

        let runs = 5;
        let mut t_single = f64::INFINITY;
        let mut t_two = f64::INFINITY;
        for _ in 0..runs {
            let mut a = input.clone();
            let s = Instant::now();
            reference_natural_fft(&mut a, m, log_n);
            t_single = t_single.min(s.elapsed().as_secs_f64());

            let mut c = input.clone();
            let s = Instant::now();
            fft_batch_two_half::<F, F>(&mut c, m, &two_tw).unwrap();
            t_two = t_two.min(s.elapsed().as_secs_f64());
        }
        println!(
            "log_n={log_n} m={m}: single={:.4}s two_half={:.4}s  two/single={:.2}x",
            t_single,
            t_two,
            t_single / t_two
        );
    }
}
