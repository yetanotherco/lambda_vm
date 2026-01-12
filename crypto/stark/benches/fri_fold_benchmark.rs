use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use math::field::element::FieldElement;
use math::field::fields::fft_friendly::babybear_u32::Babybear31PrimeField;
use math::polynomial::Polynomial;
use rand::Rng;
use stark::fri::fri_functions::{fold_polynomial, fold_polynomial_doubled, fold_polynomial_original};

type FE = FieldElement<Babybear31PrimeField>;

fn generate_random_polynomial(degree: usize) -> Polynomial<FE> {
    let mut rng = rand::thread_rng();
    let coefficients: Vec<FE> = (0..=degree).map(|_| FE::new(rng.r#gen::<u32>())).collect();
    Polynomial::new(&coefficients)
}

fn bench_fold_polynomial(c: &mut Criterion) {
    let mut group = c.benchmark_group("fri_fold");

    // Test different polynomial sizes (powers of 2, as used in FRI)
    // 2^6, 2^8, 2^10, 2^12, 2^14, 2^16, 2^18, 2^20
    for exp in [6, 8, 10, 12, 14, 16, 18, 20] {
        let size = 1usize << exp;
        group.throughput(Throughput::Elements(size as u64));

        let poly = generate_random_polynomial(size - 1);
        let beta = FE::new(12345);

        group.bench_with_input(BenchmarkId::new("original", size), &size, |b, _| {
            b.iter(|| black_box(fold_polynomial_original(&poly, &beta)))
        });

        group.bench_with_input(BenchmarkId::new("optimized", size), &size, |b, _| {
            b.iter(|| black_box(fold_polynomial(&poly, &beta)))
        });
    }

    group.finish();
}

/// Benchmark comparing scalar * fold vs fused fold_polynomial_doubled
/// This simulates the FRI commit phase pattern where each fold is multiplied by 2
fn bench_doubled_fold(c: &mut Criterion) {
    let mut group = c.benchmark_group("fri_fold_doubled");

    let two = FE::from(2);

    for exp in [10, 14, 18, 20] {
        let size = 1usize << exp;
        group.throughput(Throughput::Elements(size as u64));

        let poly = generate_random_polynomial(size - 1);
        let beta = FE::new(12345);

        // Original pattern: 2 * fold_original
        group.bench_with_input(BenchmarkId::new("mul2_x_original", size), &size, |b, _| {
            b.iter(|| black_box(&two * fold_polynomial_original(&poly, &beta)))
        });

        // Separate pattern: 2 * fold_optimized
        group.bench_with_input(BenchmarkId::new("mul2_x_optimized", size), &size, |b, _| {
            b.iter(|| black_box(&two * fold_polynomial(&poly, &beta)))
        });

        // Fused with double(): fold_polynomial_doubled (most efficient)
        group.bench_with_input(BenchmarkId::new("fused_doubled", size), &size, |b, _| {
            b.iter(|| black_box(fold_polynomial_doubled(&poly, &beta)))
        });
    }

    group.finish();
}

fn bench_multiple_folds(c: &mut Criterion) {
    let mut group = c.benchmark_group("fri_multiple_folds");

    // Simulate FRI commit phase with multiple fold operations starting from 2^20
    let initial_size = 1 << 20; // 1,048,576
    let num_folds = 18; // 2^20 -> 2^19 -> ... -> 2^2 (18 folds to get to size 4)

    group.throughput(Throughput::Elements(initial_size as u64));

    let poly = generate_random_polynomial(initial_size - 1);
    let betas: Vec<FE> = (0..num_folds).map(|i| FE::new(1000 + i as u32)).collect();
    let two = FE::from(2);

    // Original pattern: 2 * fold_original (what commit_phase used to do)
    group.bench_function("mul2_original_chain", |b| {
        b.iter(|| {
            let mut current = poly.clone();
            for beta in &betas {
                current = &two * fold_polynomial_original(&current, beta);
            }
            black_box(current)
        })
    });

    // Optimized pattern: 2 * fold_optimized
    group.bench_function("mul2_optimized_chain", |b| {
        b.iter(|| {
            let mut current = poly.clone();
            for beta in &betas {
                current = &two * fold_polynomial(&current, beta);
            }
            black_box(current)
        })
    });

    // Fused with double(): fold_polynomial_doubled (what commit_phase now does)
    group.bench_function("fused_doubled_chain", |b| {
        b.iter(|| {
            let mut current = poly.clone();
            for beta in &betas {
                current = fold_polynomial_doubled(&current, beta);
            }
            black_box(current)
        })
    });

    group.finish();
}

criterion_group!(benches, bench_fold_polynomial, bench_doubled_fold, bench_multiple_folds);

criterion_main!(benches);
