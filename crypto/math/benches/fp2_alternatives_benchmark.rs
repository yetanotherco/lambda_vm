//! Benchmark comparing different Fp2 multiplication strategies.
//!
//! Run with: cargo bench -p math --features asm-arm64 --bench fp2_alternatives_benchmark

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};

#[cfg(all(feature = "asm-arm64", target_arch = "aarch64"))]
mod fp2_benches {
    use super::*;
    use math::field::fields::fft_friendly::goldilocks_extensions_asm::{
        fp2_mul, fp2_mul_direct, fp2_mul_fused, fp2_mul_karatsuba_delayed,
        fp2_mul_karatsuba_u128_mul7, mul_by_7, mul_by_7_u128,
        fp3_mul, fp3_mul_delayed,
        GOLDILOCKS_PRIME,
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

    // ============== MUL_BY_7 BENCHMARKS ==============
    pub fn bench_mul_by_7(c: &mut Criterion) {
        let mut group = c.benchmark_group("mul_by_7_strategies");
        group.throughput(Throughput::Elements(BATCH_SIZE as u64));

        let values = generate_u64_values(BATCH_SIZE, 11111);

        group.bench_function("current_doubles_adds", |b| {
            b.iter(|| {
                for v in &values {
                    black_box(mul_by_7(*v));
                }
            })
        });

        group.bench_function("u128_shift", |b| {
            b.iter(|| {
                for v in &values {
                    black_box(mul_by_7_u128(*v));
                }
            })
        });

        group.finish();
    }

    // ============== FP2 MULTIPLICATION BENCHMARKS ==============
    pub fn bench_fp2_mul_strategies(c: &mut Criterion) {
        let mut group = c.benchmark_group("fp2_mul_strategies");
        group.throughput(Throughput::Elements(BATCH_SIZE as u64));

        let pairs = generate_fp2_pairs(BATCH_SIZE, 12345);

        // Current implementation: Karatsuba with field ops
        group.bench_function("current_karatsuba", |b| {
            b.iter(|| {
                for (a, b_val) in &pairs {
                    black_box(fp2_mul(*a, *b_val));
                }
            })
        });

        // Alt 1: Direct (Plonky3-style)
        group.bench_function("alt1_direct", |b| {
            b.iter(|| {
                for (a, b_val) in &pairs {
                    black_box(fp2_mul_direct(*a, *b_val));
                }
            })
        });

        // Alt 2: Fully fused with *7 in u128
        group.bench_function("alt2_fused", |b| {
            b.iter(|| {
                for (a, b_val) in &pairs {
                    black_box(fp2_mul_fused(*a, *b_val));
                }
            })
        });

        // Alt 3: Karatsuba with delayed reduction
        group.bench_function("alt3_karatsuba_delayed", |b| {
            b.iter(|| {
                for (a, b_val) in &pairs {
                    black_box(fp2_mul_karatsuba_delayed(*a, *b_val));
                }
            })
        });

        // Alt 4: Karatsuba with u128 mul_by_7
        group.bench_function("alt4_karatsuba_u128_mul7", |b| {
            b.iter(|| {
                for (a, b_val) in &pairs {
                    black_box(fp2_mul_karatsuba_u128_mul7(*a, *b_val));
                }
            })
        });

        group.finish();
    }

    // ============== FP3 MULTIPLICATION BENCHMARKS ==============
    pub fn bench_fp3_mul_strategies(c: &mut Criterion) {
        let mut group = c.benchmark_group("fp3_mul_strategies");
        group.throughput(Throughput::Elements(BATCH_SIZE as u64));

        let pairs = generate_fp3_pairs(BATCH_SIZE, 54321);

        // Current implementation
        group.bench_function("current_karatsuba", |b| {
            b.iter(|| {
                for (a, b_val) in &pairs {
                    black_box(fp3_mul(*a, *b_val));
                }
            })
        });

        // Delayed reduction
        group.bench_function("delayed_reduction", |b| {
            b.iter(|| {
                for (a, b_val) in &pairs {
                    black_box(fp3_mul_delayed(*a, *b_val));
                }
            })
        });

        group.finish();
    }
}

// Non-ARM64 stub
#[cfg(not(all(feature = "asm-arm64", target_arch = "aarch64")))]
mod fp2_benches {
    use super::*;

    pub fn bench_mul_by_7(_c: &mut Criterion) {
        println!("This benchmark requires --features asm-arm64 on ARM64 platform");
    }
    pub fn bench_fp2_mul_strategies(_c: &mut Criterion) {}
    pub fn bench_fp3_mul_strategies(_c: &mut Criterion) {}
}

criterion_group!(
    fp2_benches_group,
    fp2_benches::bench_mul_by_7,
    fp2_benches::bench_fp2_mul_strategies,
    fp2_benches::bench_fp3_mul_strategies,
);

criterion_main!(fp2_benches_group);
