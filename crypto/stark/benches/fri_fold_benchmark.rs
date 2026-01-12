use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use math::field::element::FieldElement;
use math::field::fields::fft_friendly::babybear_u32::Babybear31PrimeField;
use math::polynomial::Polynomial;
use rand::Rng;
use stark::fri::fri_functions::{fold_polynomial, fold_polynomial_doubled, fold_polynomial_legacy};

type FE = FieldElement<Babybear31PrimeField>;

const BETA: u32 = 12345;

fn random_polynomial(degree: usize) -> Polynomial<FE> {
    let mut rng = rand::thread_rng();
    let coefficients: Vec<FE> = (0..=degree).map(|_| FE::new(rng.r#gen::<u32>())).collect();
    Polynomial::new(&coefficients)
}

fn bench_fold_polynomial(c: &mut Criterion) {
    let mut group = c.benchmark_group("fri_fold");
    let beta = FE::new(BETA);

    for exp in [8, 12, 16, 20] {
        let size = 1usize << exp;
        let poly = random_polynomial(size - 1);

        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("legacy", size), &size, |b, _| {
            b.iter(|| black_box(fold_polynomial_legacy(&poly, &beta)))
        });

        group.bench_with_input(BenchmarkId::new("optimized", size), &size, |b, _| {
            b.iter(|| black_box(fold_polynomial(&poly, &beta)))
        });
    }

    group.finish();
}

fn bench_fold_doubled(c: &mut Criterion) {
    let mut group = c.benchmark_group("fri_fold_doubled");
    let beta = FE::new(BETA);
    let two = FE::from(2);

    for exp in [12, 16, 20] {
        let size = 1usize << exp;
        let poly = random_polynomial(size - 1);

        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("legacy_mul2", size), &size, |b, _| {
            b.iter(|| black_box(&two * fold_polynomial_legacy(&poly, &beta)))
        });

        group.bench_with_input(BenchmarkId::new("optimized_mul2", size), &size, |b, _| {
            b.iter(|| black_box(&two * fold_polynomial(&poly, &beta)))
        });

        group.bench_with_input(BenchmarkId::new("fused_double", size), &size, |b, _| {
            b.iter(|| black_box(fold_polynomial_doubled(&poly, &beta)))
        });
    }

    group.finish();
}

fn bench_fri_commit_chain(c: &mut Criterion) {
    let mut group = c.benchmark_group("fri_commit_chain");

    const INITIAL_SIZE: usize = 1 << 20;
    const NUM_FOLDS: usize = 18;

    let poly = random_polynomial(INITIAL_SIZE - 1);
    let betas: Vec<FE> = (0..NUM_FOLDS).map(|i| FE::new(1000 + i as u32)).collect();
    let two = FE::from(2);

    group.throughput(Throughput::Elements(INITIAL_SIZE as u64));

    group.bench_function("legacy", |b| {
        b.iter(|| {
            let mut current = poly.clone();
            for beta in &betas {
                current = &two * fold_polynomial_legacy(&current, beta);
            }
            black_box(current)
        })
    });

    group.bench_function("optimized", |b| {
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

criterion_group!(
    benches,
    bench_fold_polynomial,
    bench_fold_doubled,
    bench_fri_commit_chain
);
criterion_main!(benches);
