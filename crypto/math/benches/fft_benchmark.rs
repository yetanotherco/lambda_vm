//! FFT Performance Benchmarks
//!
//! Compares the performance of different FFT implementations:
//! - Native Cooley-Tukey FFT (existing implementation)
//! - Bowers FFT with SoA optimization (new implementation)

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use math::fft::cpu::bit_reversing::in_place_bit_reverse_permute;
use math::fft::cpu::bowers_fft::{LayerTwiddles, bowers_fft_opt_fused};
use math::fft::cpu::fft::in_place_nr_2radix_fft;
use math::fft::cpu::roots_of_unity::get_twiddles;
use math::field::element::FieldElement;
use math::field::fields::fft_friendly::u64_goldilocks::GoldilocksField;
use math::field::traits::RootsConfig;
use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha20Rng;

type F = GoldilocksField;
type FE = FieldElement<F>;

/// Generate random field elements for testing
fn random_field_elements(n: usize) -> Vec<FE> {
    let mut rng = ChaCha20Rng::seed_from_u64(42);
    (0..n).map(|_| FE::from(rng.next_u64())).collect()
}

/// Benchmark native Cooley-Tukey FFT
fn bench_native_fft(c: &mut Criterion) {
    let mut group = c.benchmark_group("FFT/Native Cooley-Tukey");

    // Test sizes: 2^10 to 2^20
    for order in [14, 16, 18, 20] {
        let n = 1 << order;
        let input = random_field_elements(n);
        let twiddles = get_twiddles(order, RootsConfig::BitReverse).unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(order), &order, |b, _| {
            b.iter(|| {
                let mut data = input.clone();
                in_place_nr_2radix_fft::<F, F>(black_box(&mut data), black_box(&twiddles));
                in_place_bit_reverse_permute(black_box(&mut data));
                black_box(data)
            });
        });
    }

    group.finish();
}

/// Benchmark Bowers FFT
fn bench_bowers_fft(c: &mut Criterion) {
    let mut group = c.benchmark_group("FFT/Bowers optimized");

    // Test sizes: 2^10 to 2^20
    for order in [14, 16, 18, 20] {
        let n = 1 << order;
        let input = random_field_elements(n);
        let layer_twiddles = LayerTwiddles::<F>::new(order).unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(order), &order, |b, _| {
            b.iter(|| {
                let mut data = input.clone();
                bowers_fft_opt_fused(black_box(&mut data), black_box(&layer_twiddles)).unwrap();
                in_place_bit_reverse_permute(black_box(&mut data));
                black_box(data)
            });
        });
    }

    group.finish();
}

/// Benchmark twiddle factor precomputation overhead
fn bench_twiddle_precomputation(c: &mut Criterion) {
    let mut group = c.benchmark_group("FFT/Twiddle precomputation");

    for order in [14, 16, 18, 20] {
        group.bench_with_input(BenchmarkId::new("Native", order), &order, |b, &order| {
            b.iter(|| black_box(get_twiddles::<F>(order, RootsConfig::BitReverse).unwrap()));
        });

        group.bench_with_input(BenchmarkId::new("Bowers", order), &order, |b, &order| {
            b.iter(|| black_box(LayerTwiddles::<F>::new(order).unwrap()));
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_native_fft,
    bench_bowers_fft,
    bench_twiddle_precomputation
);
criterion_main!(benches);
