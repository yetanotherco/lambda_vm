//! Dedicated benchmarks for ARM64 assembly Goldilocks field operations.
//!
//! This benchmark compares:
//! 1. Raw ASM functions vs native Rust functions
//! 2. Different multiplication strategies
//!
//! Run with: cargo bench --features asm-arm64 -- asm

#[cfg(all(feature = "asm-arm64", target_arch = "aarch64"))]
use criterion::{BenchmarkId, Throughput, black_box};
use criterion::{Criterion, criterion_group, criterion_main};

#[cfg(all(feature = "asm-arm64", target_arch = "aarch64"))]
mod asm_benches {
    use super::*;
    use math::field::fields::fft_friendly::u64_goldilocks_asm;
    use math::field::fields::fft_friendly::u64_goldilocks_native::GOLDILOCKS_PRIME;

    /// EPSILON = 2^32 - 1
    const EPSILON: u64 = 0xFFFF_FFFF;

    const BATCH_SIZE: usize = 10_000;

    // Native Rust reference implementations
    fn native_mul(a: u64, b: u64) -> u64 {
        let product = (a as u128) * (b as u128);
        native_reduce128(product)
    }

    fn native_reduce128(x: u128) -> u64 {
        let x_lo = x as u64;
        let x_hi = (x >> 64) as u64;
        let x_hi_hi = x_hi >> 32;
        let x_hi_lo = x_hi & EPSILON;

        let (t0, borrow) = x_lo.overflowing_sub(x_hi_hi);
        let t0 = if borrow { t0.wrapping_sub(EPSILON) } else { t0 };

        // Original: uses multiply
        let t1 = x_hi_lo.wrapping_mul(EPSILON);

        let (result, carry) = t0.overflowing_add(t1);
        if carry {
            result.wrapping_add(EPSILON)
        } else {
            result
        }
    }

    fn native_reduce128_shift(x: u128) -> u64 {
        let x_lo = x as u64;
        let x_hi = (x >> 64) as u64;
        let x_hi_hi = x_hi >> 32;
        let x_hi_lo = x_hi & EPSILON;

        let (t0, borrow) = x_lo.overflowing_sub(x_hi_hi);
        let t0 = if borrow { t0.wrapping_sub(EPSILON) } else { t0 };

        // Alternative: uses shift instead of multiply
        let t1 = (x_hi_lo << 32).wrapping_sub(x_hi_lo);

        let (result, carry) = t0.overflowing_add(t1);
        if carry {
            result.wrapping_add(EPSILON)
        } else {
            result
        }
    }

    fn native_mul_shift(a: u64, b: u64) -> u64 {
        let product = (a as u128) * (b as u128);
        native_reduce128_shift(product)
    }

    fn native_add(a: u64, b: u64) -> u64 {
        let (sum, over) = a.overflowing_add(b);
        let (sum, over2) = sum.overflowing_add((over as u64) * EPSILON);
        if over2 {
            sum.wrapping_add(EPSILON)
        } else {
            sum
        }
    }

    fn native_sub(a: u64, b: u64) -> u64 {
        let (diff, under) = a.overflowing_sub(b);
        let (diff, under2) = diff.overflowing_sub((under as u64) * EPSILON);
        if under2 {
            diff.wrapping_sub(EPSILON)
        } else {
            diff
        }
    }

    fn generate_random_pairs(count: usize, seed: u64) -> Vec<(u64, u64)> {
        let mut state = if seed == 0 { 1 } else { seed };
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state % GOLDILOCKS_PRIME
        };

        (0..count).map(|_| (next(), next())).collect()
    }

    // ============== MUL BENCHMARKS ==============
    pub fn bench_mul_comparison(c: &mut Criterion) {
        let mut group = c.benchmark_group("goldilocks_mul_strategies");
        group.throughput(Throughput::Elements(BATCH_SIZE as u64));

        let pairs = generate_random_pairs(BATCH_SIZE, 12345);

        // Native Rust with multiply in reduction
        group.bench_function("native_mul_multiply", |b| {
            b.iter(|| {
                for (a, b_val) in &pairs {
                    black_box(native_mul(*a, *b_val));
                }
            })
        });

        // Native Rust with shift in reduction
        group.bench_function("native_mul_shift", |b| {
            b.iter(|| {
                for (a, b_val) in &pairs {
                    black_box(native_mul_shift(*a, *b_val));
                }
            })
        });

        // ASM module mul (uses native Rust which LLVM optimizes well)
        group.bench_function("asm_mul", |b| {
            b.iter(|| {
                for (a, b_val) in &pairs {
                    black_box(u64_goldilocks_asm::mul(*a, *b_val));
                }
            })
        });

        group.finish();
    }

    // ============== SQUARE BENCHMARKS ==============
    // Note: Square benchmarks removed - native Rust outperforms ASM for squaring
    pub fn bench_square_comparison(_c: &mut Criterion) {
        // Square ASM was slower than native, so we only use native now.
        // This function is kept as a placeholder for the benchmark group.
    }

    // ============== ADD BENCHMARKS ==============
    pub fn bench_add_comparison(c: &mut Criterion) {
        let mut group = c.benchmark_group("goldilocks_add");
        group.throughput(Throughput::Elements(BATCH_SIZE as u64));

        let pairs = generate_random_pairs(BATCH_SIZE, 11111);

        group.bench_function("native_add", |b| {
            b.iter(|| {
                for (a, b_val) in &pairs {
                    black_box(native_add(*a, *b_val));
                }
            })
        });

        group.bench_function("asm_add_fast", |b| {
            b.iter(|| {
                for (a, b_val) in &pairs {
                    black_box(u64_goldilocks_asm::add_fast(*a, *b_val));
                }
            })
        });

        group.finish();
    }

    // ============== SUB BENCHMARKS ==============
    pub fn bench_sub_comparison(c: &mut Criterion) {
        let mut group = c.benchmark_group("goldilocks_sub");
        group.throughput(Throughput::Elements(BATCH_SIZE as u64));

        let pairs = generate_random_pairs(BATCH_SIZE, 22222);

        group.bench_function("native_sub", |b| {
            b.iter(|| {
                for (a, b_val) in &pairs {
                    black_box(native_sub(*a, *b_val));
                }
            })
        });

        group.bench_function("asm_sub_fast", |b| {
            b.iter(|| {
                for (a, b_val) in &pairs {
                    black_box(u64_goldilocks_asm::sub_fast(*a, *b_val));
                }
            })
        });

        group.finish();
    }

    // ============== COMBINED OPERATIONS BENCHMARK ==============
    pub fn bench_combined_ops(c: &mut Criterion) {
        let mut group = c.benchmark_group("goldilocks_combined_ops");
        group.throughput(Throughput::Elements((BATCH_SIZE / 3) as u64));

        // Simulate a typical field computation: (a * b) + (c * d) - e
        let pairs = generate_random_pairs(BATCH_SIZE * 3, 55555);

        group.bench_function("native_combined", |b| {
            b.iter(|| {
                for i in (0..BATCH_SIZE).step_by(3) {
                    let (a, b_val) = pairs[i];
                    let (c, d) = pairs[i + 1];
                    let (e, _) = pairs[i + 2];
                    let ab = native_mul(a, b_val);
                    let cd = native_mul(c, d);
                    let sum = native_add(ab, cd);
                    black_box(native_sub(sum, e));
                }
            })
        });

        group.bench_function("asm_combined", |b| {
            b.iter(|| {
                for i in (0..BATCH_SIZE).step_by(3) {
                    let (a, b_val) = pairs[i];
                    let (c, d) = pairs[i + 1];
                    let (e, _) = pairs[i + 2];
                    let ab = u64_goldilocks_asm::mul(a, b_val);
                    let cd = u64_goldilocks_asm::mul(c, d);
                    let sum = u64_goldilocks_asm::add_fast(ab, cd);
                    black_box(u64_goldilocks_asm::sub_fast(sum, e));
                }
            })
        });

        group.finish();
    }

    // ============== SCALING BENCHMARK ==============
    pub fn bench_scaling(c: &mut Criterion) {
        let mut group = c.benchmark_group("goldilocks_scaling");

        for size in [1000, 10000, 100000].iter() {
            let pairs = generate_random_pairs(*size, 66666);

            group.throughput(Throughput::Elements(*size as u64));

            group.bench_with_input(BenchmarkId::new("native_mul", size), size, |b, &size| {
                b.iter(|| {
                    for i in 0..size {
                        let (a, b_val) = pairs[i];
                        black_box(native_mul(a, b_val));
                    }
                })
            });

            group.bench_with_input(BenchmarkId::new("asm_mul", size), size, |b, &size| {
                b.iter(|| {
                    for i in 0..size {
                        let (a, b_val) = pairs[i];
                        black_box(u64_goldilocks_asm::mul(a, b_val));
                    }
                })
            });
        }

        group.finish();
    }
}

// Non-ARM64 stub
#[cfg(not(all(feature = "asm-arm64", target_arch = "aarch64")))]
mod asm_benches {
    use super::*;

    pub fn bench_mul_comparison(_c: &mut Criterion) {
        println!("ASM benchmarks require --features asm-arm64 on ARM64 platform");
    }
    pub fn bench_square_comparison(_c: &mut Criterion) {}
    pub fn bench_add_comparison(_c: &mut Criterion) {}
    pub fn bench_sub_comparison(_c: &mut Criterion) {}
    pub fn bench_combined_ops(_c: &mut Criterion) {}
    pub fn bench_scaling(_c: &mut Criterion) {}
}

criterion_group!(
    asm_benches_group,
    asm_benches::bench_mul_comparison,
    asm_benches::bench_square_comparison,
    asm_benches::bench_add_comparison,
    asm_benches::bench_sub_comparison,
    asm_benches::bench_combined_ops,
    asm_benches::bench_scaling,
);

criterion_main!(asm_benches_group);
