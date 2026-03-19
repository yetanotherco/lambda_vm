//! FFT comparison benchmark: lambda_vm Bowers fused vs Plonky3 (no SIMD, no parallel).
//!
//! Both implementations use the Goldilocks field (p = 2^64 - 2^32 + 1).
//!
//! Run with:
//!   cargo bench --bench fft_comparison -p math

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use math::fft::cpu::bit_reversing::in_place_bit_reverse_permute;
use math::fft::cpu::bowers_fft::{LayerTwiddles, bowers_fft_opt_fused};
use math::fft::cpu::fft::in_place_nr_2radix_fft;
use math::fft::cpu::roots_of_unity::get_twiddles;
use math::field::element::FieldElement;
use math::field::goldilocks::GoldilocksField;
use math::field::traits::RootsConfig;
use p3_dft::{Radix2Bowers, Radix2Dit, TwoAdicSubgroupDft};
use p3_goldilocks::Goldilocks;
use p3_matrix::dense::RowMajorMatrix;
use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha20Rng;

type FE = FieldElement<GoldilocksField>;

const SEED: u64 = 9001;

fn rand_lambda_input(n: usize) -> Vec<FE> {
    let mut rng = ChaCha20Rng::seed_from_u64(SEED);
    (0..n).map(|_| FE::from(rng.next_u64())).collect()
}

fn rand_p3_matrix(n: usize) -> RowMajorMatrix<Goldilocks> {
    let mut rng = ChaCha20Rng::seed_from_u64(SEED);
    let values: Vec<Goldilocks> = (0..n).map(|_| Goldilocks::new(rng.next_u64())).collect();
    RowMajorMatrix::new(values, 1)
}

fn bench_fft(c: &mut Criterion, group_name: &str, log_sizes: &[u32]) {
    for &log_n in log_sizes {
        let n = 1usize << log_n;
        let label = format!("2pow{log_n}");

        let twiddles =
            get_twiddles::<GoldilocksField>(log_n.into(), RootsConfig::BitReverse).unwrap();
        let layer_twiddles = LayerTwiddles::<GoldilocksField>::new(log_n.into()).unwrap();
        let dit: Radix2Dit<Goldilocks> = Radix2Dit::default();
        let bowers = Radix2Bowers::default();

        c.bench_with_input(
            BenchmarkId::new(format!("{group_name}/lambda_vm_radix2"), &label),
            &n,
            |b, &n| {
                b.iter_with_setup(
                    || rand_lambda_input(n),
                    |mut data| {
                        in_place_nr_2radix_fft::<GoldilocksField, GoldilocksField>(
                            &mut data,
                            &twiddles,
                        );
                        in_place_bit_reverse_permute(&mut data);
                        data
                    },
                )
            },
        );

        c.bench_with_input(
            BenchmarkId::new(format!("{group_name}/lambda_vm_bowers_fused"), &label),
            &n,
            |b, &n| {
                b.iter_with_setup(
                    || rand_lambda_input(n),
                    |mut data| {
                        bowers_fft_opt_fused(&mut data, &layer_twiddles).unwrap();
                        in_place_bit_reverse_permute(&mut data);
                        data
                    },
                )
            },
        );

        c.bench_with_input(
            BenchmarkId::new(format!("{group_name}/plonky3_radix2_dit"), &label),
            &n,
            |b, &n| {
                b.iter_with_setup(|| rand_p3_matrix(n), |mat| dit.dft_batch(mat))
            },
        );

        c.bench_with_input(
            BenchmarkId::new(format!("{group_name}/plonky3_bowers"), &label),
            &n,
            |b, &n| {
                b.iter_with_setup(|| rand_p3_matrix(n), |mat| bowers.dft_batch(mat))
            },
        );
    }
}

fn quick_benchmarks(c: &mut Criterion) {
    bench_fft(c, "quick", &[16, 18]);
}

fn thorough_benchmarks(c: &mut Criterion) {
    bench_fft(c, "thorough", &[20, 22]);
}

criterion_group! {
    name = quick;
    config = Criterion::default()
        .sample_size(10)
        .measurement_time(std::time::Duration::from_secs(30));
    targets = quick_benchmarks
}

criterion_group! {
    name = thorough;
    config = Criterion::default()
        .sample_size(10)
        .measurement_time(std::time::Duration::from_secs(60));
    targets = thorough_benchmarks
}

criterion_main!(quick, thorough);
