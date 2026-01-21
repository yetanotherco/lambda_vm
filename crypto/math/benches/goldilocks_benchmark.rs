//! Benchmark comparing Goldilocks Montgomery vs Native implementations
//!
//! Validates that Native form performs better than Montgomery form for field operations.

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use rand::RngCore;
use rand::SeedableRng;

// Montgomery form (base field)
use math::field::element::FieldElement;
use math::field::fields::fft_friendly::u64_goldilocks::U64GoldilocksPrimeField;

// Native form (base field)
use math::field::fields::fft_friendly::u64_goldilocks_native::GoldilocksField;

type MontFE = FieldElement<U64GoldilocksPrimeField>;
type NativeFE = FieldElement<GoldilocksField>;

const SIZES: [usize; 5] = [100, 1000, 10000, 100000, 1000000];

/// Generate random Montgomery field element pairs with fixed seed for reproducibility
fn rand_mont_elements(num: usize) -> Vec<(MontFE, MontFE)> {
    let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(9001);
    (0..num)
        .map(|_| (MontFE::from(rng.next_u64()), MontFE::from(rng.next_u64())))
        .collect()
}

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
        let mont_data = rand_mont_elements(size);
        let native_data = rand_native_elements(size);

        group.bench_with_input(
            BenchmarkId::new("montgomery", size),
            &mont_data,
            |b, data| {
                b.iter(|| {
                    for (x, y) in data {
                        black_box(black_box(x) + black_box(y));
                    }
                })
            },
        );

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
        let mont_data = rand_mont_elements(size);
        let native_data = rand_native_elements(size);

        group.bench_with_input(
            BenchmarkId::new("montgomery", size),
            &mont_data,
            |b, data| {
                b.iter(|| {
                    for (x, y) in data {
                        black_box(black_box(x) - black_box(y));
                    }
                })
            },
        );

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
        let mont_data = rand_mont_elements(size);
        let native_data = rand_native_elements(size);

        group.bench_with_input(
            BenchmarkId::new("montgomery", size),
            &mont_data,
            |b, data| {
                b.iter(|| {
                    for (x, y) in data {
                        black_box(black_box(x) * black_box(y));
                    }
                })
            },
        );

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
        let mont_data = rand_mont_elements(size);
        let native_data = rand_native_elements(size);

        group.bench_with_input(
            BenchmarkId::new("montgomery", size),
            &mont_data,
            |b, data| {
                b.iter(|| {
                    for (x, _) in data {
                        black_box(black_box(x).square());
                    }
                })
            },
        );

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
        let mont_data = rand_mont_elements(size);
        let native_data = rand_native_elements(size);

        group.bench_with_input(
            BenchmarkId::new("montgomery", size),
            &mont_data,
            |b, data| {
                b.iter(|| {
                    for (x, _) in data {
                        black_box(black_box(x).inv().unwrap());
                    }
                })
            },
        );

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

criterion_group!(
    benches,
    bench_add,
    bench_sub,
    bench_mul,
    bench_square,
    bench_inv
);
criterion_main!(benches);
