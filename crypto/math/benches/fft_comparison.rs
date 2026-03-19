//! FFT benchmark: lambda_vm_2 (3-layer fused Bowers) vs Plonky3 (Radix2Dit, Radix2Bowers)
//!
//! Both implementations use the Goldilocks field (p = 2^64 - 2^32 + 1).
//! Plonky3 is benchmarked WITHOUT the `parallel` feature (scalar only, no SIMD).
//!
//! Run with:
//!   cargo bench --bench fft_comparison -p math

use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};
use math::fft::cpu::bit_reversing::in_place_bit_reverse_permute;
use math::fft::cpu::bowers_fft::{LayerTwiddles, bowers_fft_opt_fused};
use math::fft::cpu::roots_of_unity::get_twiddles;
use math::fft::cpu::fft::in_place_nr_2radix_fft;
use math::field::element::FieldElement;
use math::field::goldilocks::GoldilocksField;
use math::field::traits::RootsConfig;
use p3_dft::{Radix2Bowers, Radix2Dit, TwoAdicSubgroupDft};
use p3_goldilocks::Goldilocks;
use p3_matrix::dense::RowMajorMatrix;

type LambdaFE = FieldElement<GoldilocksField>;

/// Create a lambda_vm_2 input vector of length `n` with sequential field elements.
fn lambda_input(n: usize) -> Vec<LambdaFE> {
    (0..n).map(|i| LambdaFE::from(i as u64 + 1)).collect()
}

/// Create a Plonky3 single-column matrix of height `n`.
fn p3_matrix(n: usize) -> RowMajorMatrix<Goldilocks> {
    let values: Vec<Goldilocks> = (0..n).map(|i| Goldilocks::new(i as u64 + 1)).collect();
    RowMajorMatrix::new(values, 1)
}

fn bench_fft_comparison(c: &mut Criterion) {
    // Same log sizes as Plonky3's own FFT benchmark.
    let log_sizes: &[u32] = &[16, 18, 20, 22];

    for &log_n in log_sizes {
        let n = 1usize << log_n;
        let label = format!("log{log_n}");

        // ── lambda_vm_2: classic radix-2 NR DIT (baseline) ──────────────────
        {
            let twiddles = get_twiddles::<GoldilocksField>(log_n.into(), RootsConfig::BitReverse)
                .expect("twiddle generation failed");

            let mut group = c.benchmark_group(format!("fft/lambda_vm2_radix2/{label}"));
            group.sample_size(10);
            group.bench_function(BenchmarkId::from_parameter(&label), |b| {
                b.iter_batched(
                    || lambda_input(n),
                    |mut data| {
                        in_place_nr_2radix_fft::<GoldilocksField, GoldilocksField>(
                            &mut data,
                            &twiddles,
                        );
                        in_place_bit_reverse_permute(&mut data);
                        data
                    },
                    BatchSize::LargeInput,
                );
            });
            group.finish();
        }

        // ── lambda_vm_2: 3-layer fused Bowers (this branch) ─────────────────
        {
            let layer_twiddles =
                LayerTwiddles::<GoldilocksField>::new(log_n.into()).expect("twiddle init failed");

            let mut group = c.benchmark_group(format!("fft/lambda_vm2_bowers_fused/{label}"));
            group.sample_size(10);
            group.bench_function(BenchmarkId::from_parameter(&label), |b| {
                b.iter_batched(
                    || lambda_input(n),
                    |mut data| {
                        bowers_fft_opt_fused(&mut data, &layer_twiddles)
                            .expect("FFT failed");
                        in_place_bit_reverse_permute(&mut data);
                        data
                    },
                    BatchSize::LargeInput,
                );
            });
            group.finish();
        }

        // ── Plonky3: Radix2Dit (no parallel, no SIMD on field ops) ──────────
        {
            let dit: Radix2Dit<Goldilocks> = Radix2Dit::default();

            let mut group = c.benchmark_group(format!("fft/plonky3_radix2_dit/{label}"));
            group.sample_size(10);
            group.bench_function(BenchmarkId::from_parameter(&label), |b| {
                b.iter_batched(
                    || p3_matrix(n),
                    |mat| dit.dft_batch(mat),
                    BatchSize::LargeInput,
                );
            });
            group.finish();
        }

        // ── Plonky3: Radix2Bowers (no parallel, no SIMD on field ops) ───────
        {
            let bowers = Radix2Bowers::default();

            let mut group = c.benchmark_group(format!("fft/plonky3_radix2_bowers/{label}"));
            group.sample_size(10);
            group.bench_function(BenchmarkId::from_parameter(&label), |b| {
                b.iter_batched(
                    || p3_matrix(n),
                    |mat| bowers.dft_batch(mat),
                    BatchSize::LargeInput,
                );
            });
            group.finish();
        }
    }
}

criterion_group!(benches, bench_fft_comparison);
criterion_main!(benches);
