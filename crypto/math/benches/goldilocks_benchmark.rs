//! Benchmark for native Goldilocks field operations.

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use rand::{RngCore, SeedableRng};

// Native form (base field)
use math::field::element::FieldElement;
use math::field::goldilocks::GoldilocksField;

type NativeFE = FieldElement<GoldilocksField>;

const SIZES: [usize; 5] = [100, 1000, 10000, 100000, 1000000];

/// Generate random Native field element pairs with fixed seed for reproducibility
fn rand_native_elements(num: usize) -> Vec<(NativeFE, NativeFE)> {
    let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(9001);
    (0..num)
        .map(|_| {
            (
                NativeFE::from(rng.next_u64()),
                NativeFE::from(rng.next_u64()),
            )
        })
        .collect()
}

fn bench_add(c: &mut Criterion) {
    let mut group = c.benchmark_group("goldilocks_add");

    for size in SIZES {
        let native_data = rand_native_elements(size);

        group.bench_with_input(BenchmarkId::new("native", size), &native_data, |b, data| {
            b.iter(|| {
                for (x, y) in data {
                    black_box(black_box(x) + black_box(y));
                }
            })
        });
    }
    group.finish();
}

fn bench_sub(c: &mut Criterion) {
    let mut group = c.benchmark_group("goldilocks_sub");

    for size in SIZES {
        let native_data = rand_native_elements(size);

        group.bench_with_input(BenchmarkId::new("native", size), &native_data, |b, data| {
            b.iter(|| {
                for (x, y) in data {
                    black_box(black_box(x) - black_box(y));
                }
            })
        });
    }
    group.finish();
}

fn bench_mul(c: &mut Criterion) {
    let mut group = c.benchmark_group("goldilocks_mul");

    for size in SIZES {
        let native_data = rand_native_elements(size);

        group.bench_with_input(BenchmarkId::new("native", size), &native_data, |b, data| {
            b.iter(|| {
                for (x, y) in data {
                    black_box(black_box(x) * black_box(y));
                }
            })
        });
    }
    group.finish();
}

fn bench_square(c: &mut Criterion) {
    let mut group = c.benchmark_group("goldilocks_square");

    for size in SIZES {
        let native_data = rand_native_elements(size);

        group.bench_with_input(BenchmarkId::new("native", size), &native_data, |b, data| {
            b.iter(|| {
                for (x, _) in data {
                    black_box(black_box(x).square());
                }
            })
        });
    }
    group.finish();
}

fn bench_inv(c: &mut Criterion) {
    let mut group = c.benchmark_group("goldilocks_inv");

    // Inversion is expensive, use smaller sizes
    const INV_SIZES: [usize; 4] = [100, 1000, 10000, 100000];

    for size in INV_SIZES {
        let native_data = rand_native_elements(size);

        group.bench_with_input(BenchmarkId::new("native", size), &native_data, |b, data| {
            b.iter(|| {
                for (x, _) in data {
                    black_box(black_box(x).inv().unwrap());
                }
            })
        });
    }
    group.finish();
}

fn bench_inplace_batch_inverse(c: &mut Criterion) {
    let mut group = c.benchmark_group("goldilocks_inplace_batch_inverse");

    // 2^14: below the parallel threshold (sequential path).
    // 2^16: at the parallel threshold.
    // 2^20: well above the threshold (parallel path with full chunking).
    const BATCH_SIZES: [usize; 3] = [1 << 14, 1 << 16, 1 << 20];

    for size in BATCH_SIZES {
        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(9001);
        let data: Vec<NativeFE> = (0..size)
            .map(|_| {
                let mut v = rng.next_u64();
                if v == 0 {
                    v = 1;
                }
                NativeFE::from(v)
            })
            .collect();

        group.bench_with_input(BenchmarkId::new("native", size), &data, |b, data| {
            b.iter_batched(
                || data.clone(),
                |mut buf| {
                    NativeFE::inplace_batch_inverse(black_box(&mut buf)).unwrap();
                    black_box(buf);
                },
                criterion::BatchSize::LargeInput,
            )
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_add,
    bench_sub,
    bench_mul,
    bench_square,
    bench_inv,
    bench_inplace_batch_inverse
);
criterion_main!(benches);
