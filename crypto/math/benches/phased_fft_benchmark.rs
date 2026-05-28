//! Phased FFT vs Bowers FFT — single-column and multi-column.

use std::hint::black_box;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use rand::{RngCore, SeedableRng};

use math::fft::bit_reversing::in_place_bit_reverse_permute;
use math::fft::bowers_fft::{LayerTwiddles, bowers_fft_opt_fused_parallel};
use math::fft::phased_fft::{
    PhasedFftContext, bowers_phased_fft_multicol, bowers_phased_fft_with_buf,
};
use math::field::element::FieldElement;
use math::field::goldilocks::GoldilocksField;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

type F = GoldilocksField;
type FE = FieldElement<F>;

fn random_input(log_n: usize, seed: u64) -> Vec<FE> {
    let n = 1usize << log_n;
    let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(seed);
    (0..n).map(|_| FE::from(rng.next_u64())).collect()
}

fn bench_single_col(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft_single_col");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(8));

    for &log_n in &[20usize, 22, 24] {
        let n = 1usize << log_n;
        let input = random_input(log_n, 1);
        let bowers_twiddles = LayerTwiddles::<F>::new(log_n as u64).unwrap();
        let phased_ctx = PhasedFftContext::<F>::new(log_n).unwrap();

        group.bench_with_input(
            BenchmarkId::new("bowers", format!("2^{log_n}")),
            &n,
            |b, _| {
                b.iter_batched(
                    || input.clone(),
                    |mut data| {
                        bowers_fft_opt_fused_parallel(&mut data, &bowers_twiddles).unwrap();
                        in_place_bit_reverse_permute(&mut data);
                        black_box(data);
                    },
                    criterion::BatchSize::LargeInput,
                );
            },
        );

        let mut scratch: Vec<FE> = Vec::with_capacity(n);
        group.bench_with_input(
            BenchmarkId::new("phased", format!("2^{log_n}")),
            &n,
            |b, _| {
                b.iter_batched(
                    || input.clone(),
                    |mut data| {
                        bowers_phased_fft_with_buf::<F, F>(&mut data, &phased_ctx, &mut scratch)
                            .unwrap();
                        black_box(data);
                    },
                    criterion::BatchSize::LargeInput,
                );
            },
        );
    }

    group.finish();
}

/// Multi-column bench: realistic trace LDE shape. CPU table is 74 cols at
/// log_n = 21, MEMW at 49 cols. We probe a few (cols × log_n) combinations.
fn bench_multi_col(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft_multi_col");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(10));

    for &(cols, log_n) in &[(74usize, 21usize), (49, 21), (24, 22), (12, 24)] {
        let n = 1usize << log_n;
        let bowers_twiddles = LayerTwiddles::<F>::new(log_n as u64).unwrap();
        let phased_ctx = PhasedFftContext::<F>::new(log_n).unwrap();
        let columns: Vec<Vec<FE>> = (0..cols)
            .map(|c| random_input(log_n, c as u64 + 1))
            .collect();

        // Baseline 1: rayon par_iter over columns, each calls bowers
        // independently. This mirrors what the prover does today.
        group.bench_with_input(
            BenchmarkId::new("bowers_par", format!("{cols}cols_2^{log_n}")),
            &n,
            |b, _| {
                b.iter_batched(
                    || columns.clone(),
                    |mut data| {
                        #[cfg(feature = "parallel")]
                        data.par_iter_mut().for_each(|col| {
                            bowers_fft_opt_fused_parallel(col, &bowers_twiddles).unwrap();
                            in_place_bit_reverse_permute(col);
                        });
                        #[cfg(not(feature = "parallel"))]
                        for col in data.iter_mut() {
                            bowers_fft_opt_fused_parallel(col, &bowers_twiddles).unwrap();
                            in_place_bit_reverse_permute(col);
                        }
                        black_box(data);
                    },
                    criterion::BatchSize::LargeInput,
                );
            },
        );

        // Phased multi-col: shared ctx + per-worker buffer pool.
        group.bench_with_input(
            BenchmarkId::new("phased_multicol", format!("{cols}cols_2^{log_n}")),
            &n,
            |b, _| {
                b.iter_batched(
                    || columns.clone(),
                    |mut data| {
                        let mut refs: Vec<&mut [FE]> =
                            data.iter_mut().map(|c| c.as_mut_slice()).collect();
                        bowers_phased_fft_multicol::<F, F>(&mut refs, &phased_ctx).unwrap();
                        black_box(data);
                    },
                    criterion::BatchSize::LargeInput,
                );
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_single_col, bench_multi_col);
criterion_main!(benches);
