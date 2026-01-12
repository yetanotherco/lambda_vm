//! Benchmark for Fp2 and Fp3 extension field operations.
//!
//! Run with: cargo bench -p math --features asm-arm64 --bench fp2_alternatives_benchmark

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};

#[cfg(all(feature = "asm-arm64", target_arch = "aarch64"))]
mod fp2_benches {
    use super::*;
    use math::field::fields::fft_friendly::goldilocks_extensions_asm::{
        fp2_mul, fp2_square, fp3_mul, fp3_square, mul_by_7, GOLDILOCKS_PRIME,
    };

    const BATCH_SIZE: usize = 10_000;

    fn generate_fp2_pairs(count: usize, seed: u64) -> Vec<([u64; 2], [u64; 2])> {
        let mut state = if seed == 0 { 1 } else { seed };
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state % GOLDILOCKS_PRIME
        };

        (0..count)
            .map(|_| ([next(), next()], [next(), next()]))
            .collect()
    }

    fn generate_fp3_pairs(count: usize, seed: u64) -> Vec<([u64; 3], [u64; 3])> {
        let mut state = if seed == 0 { 1 } else { seed };
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state % GOLDILOCKS_PRIME
        };

        (0..count)
            .map(|_| ([next(), next(), next()], [next(), next(), next()]))
            .collect()
    }

    fn generate_u64_values(count: usize, seed: u64) -> Vec<u64> {
        let mut state = if seed == 0 { 1 } else { seed };
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state % GOLDILOCKS_PRIME
        };

        (0..count).map(|_| next()).collect()
    }

    pub fn bench_mul_by_7(c: &mut Criterion) {
        let mut group = c.benchmark_group("mul_by_7");
        group.throughput(Throughput::Elements(BATCH_SIZE as u64));

        let values = generate_u64_values(BATCH_SIZE, 11111);

        group.bench_function("u128_shift", |b| {
            b.iter(|| {
                for v in &values {
                    black_box(mul_by_7(*v));
                }
            })
        });

        group.finish();
    }

    pub fn bench_fp2_operations(c: &mut Criterion) {
        let mut group = c.benchmark_group("fp2_operations");
        group.throughput(Throughput::Elements(BATCH_SIZE as u64));

        let pairs = generate_fp2_pairs(BATCH_SIZE, 12345);
        let elements: Vec<[u64; 2]> = pairs.iter().map(|(a, _)| *a).collect();

        group.bench_function("mul", |b| {
            b.iter(|| {
                for (a, b_val) in &pairs {
                    black_box(fp2_mul(*a, *b_val));
                }
            })
        });

        group.bench_function("square", |b| {
            b.iter(|| {
                for a in &elements {
                    black_box(fp2_square(*a));
                }
            })
        });

        group.finish();
    }

    pub fn bench_fp3_operations(c: &mut Criterion) {
        let mut group = c.benchmark_group("fp3_operations");
        group.throughput(Throughput::Elements(BATCH_SIZE as u64));

        let pairs = generate_fp3_pairs(BATCH_SIZE, 54321);
        let elements: Vec<[u64; 3]> = pairs.iter().map(|(a, _)| *a).collect();

        group.bench_function("mul", |b| {
            b.iter(|| {
                for (a, b_val) in &pairs {
                    black_box(fp3_mul(*a, *b_val));
                }
            })
        });

        group.bench_function("square", |b| {
            b.iter(|| {
                for a in &elements {
                    black_box(fp3_square(*a));
                }
            })
        });

        group.finish();
    }
}

#[cfg(not(all(feature = "asm-arm64", target_arch = "aarch64")))]
mod fp2_benches {
    use super::*;

    pub fn bench_mul_by_7(_c: &mut Criterion) {
        println!("This benchmark requires --features asm-arm64 on ARM64 platform");
    }
    pub fn bench_fp2_operations(_c: &mut Criterion) {}
    pub fn bench_fp3_operations(_c: &mut Criterion) {}
}

criterion_group!(
    fp2_benches_group,
    fp2_benches::bench_mul_by_7,
    fp2_benches::bench_fp2_operations,
    fp2_benches::bench_fp3_operations,
);

criterion_main!(fp2_benches_group);
