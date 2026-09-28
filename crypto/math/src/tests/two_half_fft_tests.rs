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

mod expand {
    //! Raw-bit parity of the padding-free expansion (`fft_batch_two_half_expand`,
    //! `coset_lde_full_expand_row_major[_from]`) against the pipeline it
    //! replaces: zero-`resize` the buffer, then `fft_batch_two_half` over all of
    //! it. The butterflies are the same, so the raw (non-canonical) u64s must be
    //! identical, not just equal as field elements.
    use crate::fft::bit_reversing::{
        bit_reverse_rows_into, in_place_bit_reverse_permute_row_major,
    };
    use crate::fft::two_half_fft::{
        TwoHalfTwiddles, fft_batch_two_half, fft_batch_two_half_expand,
    };
    use crate::field::element::FieldElement;
    use crate::field::extensions_goldilocks::Degree3GoldilocksExtensionField;
    use crate::field::goldilocks::GoldilocksField;
    use crate::field::traits::IsField;
    use crate::polynomial::Polynomial;
    use alloc::vec::Vec;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    type F = GoldilocksField;
    type E3 = Degree3GoldilocksExtensionField;

    /// Raw u64 words of a field-element slice (`FieldElement` is
    /// `repr(transparent)` over u64-word base types).
    fn raw<E: IsField>(xs: &[FieldElement<E>]) -> Vec<u64> {
        let words = core::mem::size_of_val(xs) / 8;
        // SAFETY: Goldilocks and its cubic extension are plain u64 words.
        unsafe { core::slice::from_raw_parts(xs.as_ptr() as *const u64, words) }.to_vec()
    }

    /// Random raw base-field words, including non-canonical ones (>= p).
    fn base(rng: &mut ChaCha8Rng, len: usize) -> Vec<FieldElement<F>> {
        (0..len)
            .map(|_| FieldElement::from_raw(rng.r#gen::<u64>()))
            .collect()
    }

    fn ext3(rng: &mut ChaCha8Rng, len: usize) -> Vec<FieldElement<E3>> {
        (0..len)
            .map(|_| {
                FieldElement::from_raw([
                    FieldElement::from_raw(rng.r#gen::<u64>()),
                    FieldElement::from_raw(rng.r#gen::<u64>()),
                    FieldElement::from_raw(rng.r#gen::<u64>()),
                ])
            })
            .collect()
    }

    /// `(log_n, log_spread)`: fallbacks (no padding; more spread than a
    /// first-half chunk has rows), tail-only shapes, and ones with several
    /// in-place waves before the tail (`2^floor(log_lde/2)` chunks > 32).
    fn shapes() -> Vec<(usize, usize)> {
        let mut v = Vec::new();
        for log_lde in 1..=16usize {
            for log_spread in 0..=4usize.min(log_lde) {
                v.push((log_lde - log_spread, log_spread));
            }
        }
        v
    }

    fn check_expand<E: IsField>(
        prefix: Vec<FieldElement<E>>,
        m: usize,
        log_n: usize,
        log_spread: usize,
    ) where
        F: crate::field::traits::IsSubFieldOf<E>,
        FieldElement<E>: Send + Sync,
    {
        let lde_rows = 1usize << (log_n + log_spread);
        let tw = TwoHalfTwiddles::<F>::new(log_n + log_spread, false).unwrap();

        let mut expected = prefix.clone();
        expected.resize(lde_rows * m, FieldElement::zero());
        fft_batch_two_half::<F, E>(&mut expected, m, &tw).unwrap();

        let mut got = prefix;
        fft_batch_two_half_expand::<F, E>(&mut got, m, lde_rows, &tw).unwrap();
        assert_eq!(got.len(), lde_rows * m);
        assert_eq!(
            raw(&got),
            raw(&expected),
            "log_n={log_n} log_spread={log_spread} m={m}"
        );
    }

    #[test]
    fn expand_matches_resize_then_fft_bits_base() {
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        for (log_n, log_spread) in shapes() {
            for m in [1usize, 3, 5] {
                let prefix = base(&mut rng, (1 << log_n) * m);
                check_expand(prefix, m, log_n, log_spread);
            }
        }
    }

    #[test]
    fn expand_matches_resize_then_fft_bits_ext3() {
        let mut rng = ChaCha8Rng::seed_from_u64(2);
        for (log_n, log_spread) in shapes() {
            let m = 2;
            let prefix = ext3(&mut rng, (1 << log_n) * m);
            check_expand(prefix, m, log_n, log_spread);
        }
    }

    /// Ignores whatever the spare capacity held before (the padding is never
    /// read): poison it through a longer previous use of the same allocation.
    #[test]
    fn expand_ignores_stale_spare_capacity() {
        let mut rng = ChaCha8Rng::seed_from_u64(3);
        let (log_n, log_spread, m) = (12usize, 1usize, 3usize);
        let n = 1usize << log_n;
        let prefix = base(&mut rng, n * m);
        let mut buf = base(&mut rng, (n << log_spread) * m); // full of junk
        buf.truncate(n * m);
        buf.copy_from_slice(&prefix);
        let tw = TwoHalfTwiddles::<F>::new(log_n + log_spread, false).unwrap();
        fft_batch_two_half_expand::<F, F>(&mut buf, m, n << log_spread, &tw).unwrap();

        let mut expected = prefix;
        expected.resize((n << log_spread) * m, FieldElement::zero());
        fft_batch_two_half::<F, F>(&mut expected, m, &tw).unwrap();
        assert_eq!(raw(&buf), raw(&expected));
    }

    #[test]
    fn bit_reverse_rows_into_matches_copy_then_in_place() {
        let mut rng = ChaCha8Rng::seed_from_u64(4);
        for log_n in [0usize, 1, 5, 11, 12] {
            for m in [1usize, 4, 7] {
                let src = base(&mut rng, (1 << log_n) * m);
                let mut expected = src.clone();
                in_place_bit_reverse_permute_row_major(&mut expected, m);
                let mut got = Vec::with_capacity(3 * src.len());
                bit_reverse_rows_into(&src, &mut got, m);
                assert_eq!(raw(&got), raw(&expected), "log_n={log_n} m={m}");
                assert!(got.capacity() >= 3 * src.len(), "capacity kept");
            }
        }
    }

    /// The coset LDE, reference-first: the old pipeline spelled out step by
    /// step (copy, iFFT, scale, zero-resize, forward FFT).
    fn reference_lde(
        src: &[FieldElement<F>],
        m: usize,
        blowup: usize,
        w: &[FieldElement<F>],
    ) -> Vec<FieldElement<F>> {
        let n = src.len() / m;
        let inv = TwoHalfTwiddles::<F>::new(n.trailing_zeros() as usize, true).unwrap();
        let fwd = TwoHalfTwiddles::<F>::new((n * blowup).trailing_zeros() as usize, false).unwrap();
        let mut buf = src.to_vec();
        fft_batch_two_half::<F, F>(&mut buf, m, &inv).unwrap();
        for (r, row) in buf.chunks_exact_mut(m).enumerate() {
            for x in row.iter_mut() {
                *x = w[r] * *x;
            }
        }
        buf.resize(n * blowup * m, FieldElement::zero());
        fft_batch_two_half::<F, F>(&mut buf, m, &fwd).unwrap();
        buf
    }

    #[test]
    fn row_major_coset_lde_from_and_in_place_match_reference_bits() {
        let mut rng = ChaCha8Rng::seed_from_u64(5);
        for log_n in [1usize, 4, 9, 13] {
            for blowup in [1usize, 2, 4, 8] {
                for m in [1usize, 3, 6] {
                    let n = 1usize << log_n;
                    let src = base(&mut rng, n * m);
                    let w = base(&mut rng, n);
                    let inv = TwoHalfTwiddles::<F>::new(log_n, true).unwrap();
                    let fwd =
                        TwoHalfTwiddles::<F>::new((n * blowup).trailing_zeros() as usize, false)
                            .unwrap();
                    let expected = raw(&reference_lde(&src, m, blowup, &w));
                    let ctx = format!("log_n={log_n} blowup={blowup} m={m}");

                    let from =
                        Polynomial::<FieldElement<F>>::coset_lde_full_expand_row_major_from::<F>(
                            &src, m, blowup, &w, &inv, &fwd,
                        )
                        .unwrap();
                    assert_eq!(raw(&from), expected, "_from {ctx}");

                    let mut in_place = Vec::with_capacity(n * blowup * m);
                    in_place.extend_from_slice(&src);
                    Polynomial::<FieldElement<F>>::coset_lde_full_expand_row_major::<F>(
                        &mut in_place,
                        m,
                        blowup,
                        &w,
                        &inv,
                        &fwd,
                    )
                    .unwrap();
                    assert_eq!(raw(&in_place), expected, "in place {ctx}");
                }
            }
        }
    }

    /// Informal timing of the old row-major LDE (copy the input in, iFFT,
    /// scale, zero-resize, forward FFT) against `_from` (bit-reverse the input
    /// in, padding never written). Run with
    /// `cargo test -p math --release --lib bench_row_major_lde_from -- --ignored --nocapture`.
    #[test]
    #[ignore = "informal perf probe; run with --ignored"]
    fn bench_row_major_lde_from_vs_old() {
        extern crate std;
        use std::time::Instant;
        let mut rng = ChaCha8Rng::seed_from_u64(6);
        for (log_n, m) in [(18usize, 64usize), (20, 64), (20, 16)] {
            let n = 1usize << log_n;
            let blowup = 2;
            let src = base(&mut rng, n * m);
            let w = base(&mut rng, n);
            let inv = TwoHalfTwiddles::<F>::new(log_n, true).unwrap();
            let fwd = TwoHalfTwiddles::<F>::new(log_n + 1, false).unwrap();
            let median = |f: &mut dyn FnMut()| {
                f();
                let mut t: Vec<f64> = (0..7)
                    .map(|_| {
                        let t0 = Instant::now();
                        f();
                        t0.elapsed().as_secs_f64() * 1e3
                    })
                    .collect();
                t.sort_by(|a, b| a.partial_cmp(b).unwrap());
                t[3]
            };
            let old = median(&mut || {
                let mut buf = Vec::with_capacity(n * blowup * m);
                buf.extend_from_slice(&src);
                fft_batch_two_half::<F, F>(&mut buf, m, &inv).unwrap();
                // Parallel, like the scale pass the old pipeline ran.
                use rayon::prelude::*;
                buf.par_chunks_exact_mut(m)
                    .enumerate()
                    .for_each(|(r, row)| {
                        for x in row.iter_mut() {
                            *x = w[r] * *x;
                        }
                    });
                buf.resize(n * blowup * m, FieldElement::zero());
                fft_batch_two_half::<F, F>(&mut buf, m, &fwd).unwrap();
                core::hint::black_box(buf);
            });
            let new = median(&mut || {
                let buf = Polynomial::<FieldElement<F>>::coset_lde_full_expand_row_major_from::<F>(
                    &src, m, blowup, &w, &inv, &fwd,
                )
                .unwrap();
                core::hint::black_box(buf);
            });
            std::println!(
                "log_n={log_n} m={m}: old {old:.1} ms | new {new:.1} ms | {:+.1}%",
                100.0 * (new - old) / old
            );
        }
    }
}
