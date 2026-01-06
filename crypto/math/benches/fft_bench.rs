use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use math::fft::cpu::bit_reversing::in_place_bit_reverse_permute;
use math::fft::cpu::fft::{in_place_nr_2radix_fft, in_place_nr_2radix_fft_parallel};
use math::fft::cpu::roots_of_unity::{get_twiddles, TwiddleCache};
use math::field::element::FieldElement;
use math::field::fields::fft_friendly::babybear_u32::Babybear31PrimeField;
use math::field::traits::RootsConfig;

type F = Babybear31PrimeField;
type FE = FieldElement<F>;

fn benchmark_fft_sequential_vs_parallel(c: &mut Criterion) {
    let mut group = c.benchmark_group("FFT Sequential vs Parallel");

    // Test with different sizes
    for order in [12, 14, 16, 18, 20] {
        let size = 1 << order;
        let input: Vec<FE> = (1..=size as u64).map(FE::from).collect();
        let twiddles = get_twiddles::<F>(order, RootsConfig::BitReverse).unwrap();

        group.bench_with_input(
            BenchmarkId::new("Sequential", size),
            &(input.clone(), twiddles.clone()),
            |b, (input, twiddles)| {
                b.iter(|| {
                    let mut result = input.clone();
                    in_place_nr_2radix_fft::<F, F>(black_box(&mut result), black_box(twiddles));
                    in_place_bit_reverse_permute(&mut result);
                    result
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("Parallel", size),
            &(input, twiddles),
            |b, (input, twiddles)| {
                b.iter(|| {
                    let mut result = input.clone();
                    in_place_nr_2radix_fft_parallel::<F, F>(black_box(&mut result), black_box(twiddles));
                    in_place_bit_reverse_permute(&mut result);
                    result
                })
            },
        );
    }

    group.finish();
}

fn benchmark_twiddle_caching(c: &mut Criterion) {
    let mut group = c.benchmark_group("Twiddle Computation vs Cache");

    for order in [12, 16, 20] {
        // Benchmark direct computation
        group.bench_with_input(
            BenchmarkId::new("Direct Computation", 1 << order),
            &order,
            |b, &order| {
                b.iter(|| {
                    get_twiddles::<F>(black_box(order), RootsConfig::BitReverse).unwrap()
                })
            },
        );

        // Benchmark cached retrieval
        let cache: TwiddleCache<F> = TwiddleCache::new();
        cache.precompute(order, RootsConfig::BitReverse).unwrap();

        group.bench_with_input(
            BenchmarkId::new("Cached Retrieval", 1 << order),
            &(order, &cache),
            |b, (order, cache)| {
                b.iter(|| {
                    cache.get_or_compute(black_box(*order), RootsConfig::BitReverse).unwrap()
                })
            },
        );
    }

    group.finish();
}

fn benchmark_end_to_end_fft(c: &mut Criterion) {
    let mut group = c.benchmark_group("End-to-End FFT");

    for order in [14, 16, 18] {
        let size = 1 << order;
        let input: Vec<FE> = (1..=size as u64).map(FE::from).collect();

        // Without cache (includes twiddle computation)
        group.bench_with_input(
            BenchmarkId::new("Sequential + Twiddle Compute", size),
            &input,
            |b, input| {
                b.iter(|| {
                    let twiddles = get_twiddles::<F>(order, RootsConfig::BitReverse).unwrap();
                    let mut result = input.clone();
                    in_place_nr_2radix_fft::<F, F>(&mut result, &twiddles);
                    in_place_bit_reverse_permute(&mut result);
                    result
                })
            },
        );

        // With cache
        let cache: TwiddleCache<F> = TwiddleCache::new();
        cache.precompute(order, RootsConfig::BitReverse).unwrap();

        group.bench_with_input(
            BenchmarkId::new("Sequential + Cached Twiddles", size),
            &(&input, &cache),
            |b, (input, cache)| {
                b.iter(|| {
                    let twiddles = cache.get_or_compute(order, RootsConfig::BitReverse).unwrap();
                    let mut result = (*input).clone();
                    in_place_nr_2radix_fft::<F, F>(&mut result, &twiddles);
                    in_place_bit_reverse_permute(&mut result);
                    result
                })
            },
        );

        // Parallel with cache
        group.bench_with_input(
            BenchmarkId::new("Parallel + Cached Twiddles", size),
            &(&input, &cache),
            |b, (input, cache)| {
                b.iter(|| {
                    let twiddles = cache.get_or_compute(order, RootsConfig::BitReverse).unwrap();
                    let mut result = (*input).clone();
                    in_place_nr_2radix_fft_parallel::<F, F>(&mut result, &twiddles);
                    in_place_bit_reverse_permute(&mut result);
                    result
                })
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    benchmark_fft_sequential_vs_parallel,
    benchmark_twiddle_caching,
    benchmark_end_to_end_fft
);
criterion_main!(benches);
