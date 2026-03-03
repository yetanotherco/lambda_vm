//! Benchmark for native Goldilocks field operations.

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use rand::{RngCore, SeedableRng};

// Native form (base field)
use math::field::element::FieldElement;
use math::field::fields::fft_friendly::u64_goldilocks::GoldilocksField;
use math::field::fields::fft_friendly::u64_goldilocks_packed::PackedGoldilocks;
use math::field::packed::PackedField;

type NativeFE = FieldElement<GoldilocksField>;
type Packed = PackedGoldilocks;

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

/// Generate random packed element pairs
fn rand_packed_elements(num: usize) -> Vec<(Packed, Packed)> {
    let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(9001);
    (0..num)
        .map(|_| {
            let a = Packed::from_fn(|_| NativeFE::from(rng.next_u64()));
            let b = Packed::from_fn(|_| NativeFE::from(rng.next_u64()));
            (a, b)
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

// ---- Packed (SIMD) benchmarks ----
// Each packed element holds WIDTH scalars, so packed ops process WIDTH elements at once.

fn bench_packed_add(c: &mut Criterion) {
    let mut group = c.benchmark_group("goldilocks_packed_add");

    for size in SIZES {
        let packed_data = rand_packed_elements(size);

        group.bench_with_input(BenchmarkId::new("packed", size), &packed_data, |b, data| {
            b.iter(|| {
                for (x, y) in data {
                    black_box(black_box(*x) + black_box(*y));
                }
            })
        });
    }
    group.finish();
}

fn bench_packed_mul(c: &mut Criterion) {
    let mut group = c.benchmark_group("goldilocks_packed_mul");

    for size in SIZES {
        let packed_data = rand_packed_elements(size);

        group.bench_with_input(BenchmarkId::new("packed", size), &packed_data, |b, data| {
            b.iter(|| {
                for (x, y) in data {
                    black_box(black_box(*x) * black_box(*y));
                }
            })
        });
    }
    group.finish();
}

fn bench_packed_square(c: &mut Criterion) {
    let mut group = c.benchmark_group("goldilocks_packed_square");

    for size in SIZES {
        let packed_data = rand_packed_elements(size);

        group.bench_with_input(BenchmarkId::new("packed", size), &packed_data, |b, data| {
            b.iter(|| {
                for (x, _) in data {
                    black_box(black_box(*x).square());
                }
            })
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
    bench_packed_add,
    bench_packed_mul,
    bench_packed_square,
);
criterion_main!(benches);
