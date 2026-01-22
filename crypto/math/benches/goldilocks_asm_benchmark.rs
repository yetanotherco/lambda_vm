//! Comprehensive benchmarks for ARM64 assembly Goldilocks field operations.
//!
//! This benchmark compares:
//! 1. Raw ASM functions vs native Rust functions for base field
//! 2. Fp2 extension field operations (ASM vs native)
//! 3. Fp3 extension field operations (ASM vs native)
//!
//! Run with: cargo bench --features asm-arm64 --bench goldilocks_asm_benchmark

#[cfg(all(feature = "asm-arm64", target_arch = "aarch64"))]
use criterion::{BenchmarkId, black_box};
use criterion::{Criterion, criterion_group, criterion_main};

#[cfg(all(feature = "asm-arm64", target_arch = "aarch64"))]
mod asm_benches {
    use super::*;
    use math::field::fields::fft_friendly::goldilocks_extensions_asm;
    use math::field::fields::fft_friendly::u64_goldilocks::GOLDILOCKS_PRIME;
    use math::field::fields::fft_friendly::u64_goldilocks_asm;

    /// EPSILON = 2^32 - 1
    const EPSILON: u64 = 0xFFFF_FFFF;

    const BATCH_SIZE: usize = 10_000;

    // ============================================================
    // NATIVE RUST REFERENCE IMPLEMENTATIONS - BASE FIELD
    // ============================================================

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

        let t1 = x_hi_lo.wrapping_mul(EPSILON);

        let (result, carry) = t0.overflowing_add(t1);
        if carry {
            result.wrapping_add(EPSILON)
        } else {
            result
        }
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

    fn native_double(a: u64) -> u64 {
        native_add(a, a)
    }

    fn native_square(a: u64) -> u64 {
        native_mul(a, a)
    }

    // ============================================================
    // NATIVE RUST REFERENCE IMPLEMENTATIONS - FP2
    // ============================================================

    fn native_fp2_add(a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
        [native_add(a[0], b[0]), native_add(a[1], b[1])]
    }

    fn native_fp2_mul(a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
        let a0b0 = native_mul(a[0], b[0]);
        let a1b1 = native_mul(a[1], b[1]);
        let a0b1 = native_mul(a[0], b[1]);
        let a1b0 = native_mul(a[1], b[0]);

        // 7 * a1b1
        let a1b1_2 = native_double(a1b1);
        let a1b1_4 = native_double(a1b1_2);
        let temp = native_add(a1b1, a1b1_2);
        let w_a1b1 = native_add(temp, a1b1_4);

        let c0 = native_add(a0b0, w_a1b1);
        let c1 = native_add(a0b1, a1b0);

        [c0, c1]
    }

    fn native_fp2_square(a: [u64; 2]) -> [u64; 2] {
        let a0_sq = native_square(a[0]);
        let a1_sq = native_square(a[1]);
        let a0a1 = native_mul(a[0], a[1]);

        // 7 * a1_sq
        let a1_sq_2 = native_double(a1_sq);
        let a1_sq_4 = native_double(a1_sq_2);
        let temp = native_add(a1_sq, a1_sq_2);
        let w_a1_sq = native_add(temp, a1_sq_4);

        let c0 = native_add(a0_sq, w_a1_sq);
        let c1 = native_double(a0a1);

        [c0, c1]
    }

    // ============================================================
    // NATIVE RUST REFERENCE IMPLEMENTATIONS - FP3
    // ============================================================

    fn native_fp3_add(a: [u64; 3], b: [u64; 3]) -> [u64; 3] {
        [
            native_add(a[0], b[0]),
            native_add(a[1], b[1]),
            native_add(a[2], b[2]),
        ]
    }

    fn native_fp3_mul(a: [u64; 3], b: [u64; 3]) -> [u64; 3] {
        // Karatsuba algorithm
        let v0 = native_mul(a[0], b[0]);
        let v1 = native_mul(a[1], b[1]);
        let v2 = native_mul(a[2], b[2]);

        let a1_plus_a2 = native_add(a[1], a[2]);
        let b1_plus_b2 = native_add(b[1], b[2]);
        let temp0 = native_mul(a1_plus_a2, b1_plus_b2);
        let temp0 = native_sub(temp0, v1);
        let t0 = native_sub(temp0, v2);

        let a0_plus_a1 = native_add(a[0], a[1]);
        let b0_plus_b1 = native_add(b[0], b[1]);
        let temp1 = native_mul(a0_plus_a1, b0_plus_b1);
        let temp1 = native_sub(temp1, v0);
        let t1 = native_sub(temp1, v1);

        let a0_plus_a2 = native_add(a[0], a[2]);
        let b0_plus_b2 = native_add(b[0], b[2]);
        let temp2 = native_mul(a0_plus_a2, b0_plus_b2);
        let temp2 = native_sub(temp2, v0);
        let t2 = native_sub(temp2, v2);

        let c0 = native_add(v0, native_double(t0));
        let c1 = native_add(t1, native_double(v2));
        let c2 = native_add(t2, v1);

        [c0, c1, c2]
    }

    fn native_fp3_square(a: [u64; 3]) -> [u64; 3] {
        let s0 = native_square(a[0]);
        let s1 = native_square(a[1]);
        let s2 = native_square(a[2]);
        let a01 = native_mul(a[0], a[1]);
        let a02 = native_mul(a[0], a[2]);
        let a12 = native_mul(a[1], a[2]);

        // c0 = s0 + 4*a12
        let a12_2 = native_double(a12);
        let a12_4 = native_double(a12_2);
        let c0 = native_add(s0, a12_4);

        // c1 = 2*a01 + 2*s2
        let c1 = native_add(native_double(a01), native_double(s2));

        // c2 = 2*a02 + s1
        let c2 = native_add(native_double(a02), s1);

        [c0, c1, c2]
    }

    // ============================================================
    // RANDOM DATA GENERATION
    // ============================================================

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

    fn generate_random_fp2_pairs(count: usize, seed: u64) -> Vec<([u64; 2], [u64; 2])> {
        let mut state = if seed == 0 { 1 } else { seed };
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state % GOLDILOCKS_PRIME
        };

        (0..count)
            .map(|_| {
                let a = [next(), next()];
                let b = [next(), next()];
                (a, b)
            })
            .collect()
    }

    fn generate_random_fp3_pairs(count: usize, seed: u64) -> Vec<([u64; 3], [u64; 3])> {
        let mut state = if seed == 0 { 1 } else { seed };
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state % GOLDILOCKS_PRIME
        };

        (0..count)
            .map(|_| {
                let a = [next(), next(), next()];
                let b = [next(), next(), next()];
                (a, b)
            })
            .collect()
    }

    // ============================================================
    // BASE FIELD BENCHMARKS
    // ============================================================

    pub fn bench_base_add(c: &mut Criterion) {
        let mut group = c.benchmark_group("base_field_add");

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

    pub fn bench_base_sub(c: &mut Criterion) {
        let mut group = c.benchmark_group("base_field_sub");

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

    pub fn bench_base_mul(c: &mut Criterion) {
        let mut group = c.benchmark_group("base_field_mul");

        let pairs = generate_random_pairs(BATCH_SIZE, 33333);

        group.bench_function("native_mul", |b| {
            b.iter(|| {
                for (a, b_val) in &pairs {
                    black_box(native_mul(*a, *b_val));
                }
            })
        });

        group.bench_function("asm_mul", |b| {
            b.iter(|| {
                for (a, b_val) in &pairs {
                    black_box(u64_goldilocks_asm::mul(*a, *b_val));
                }
            })
        });

        group.finish();
    }

    pub fn bench_base_square(c: &mut Criterion) {
        let mut group = c.benchmark_group("base_field_square");

        let pairs = generate_random_pairs(BATCH_SIZE, 44444);

        group.bench_function("native_square", |b| {
            b.iter(|| {
                for (a, _) in &pairs {
                    black_box(native_square(*a));
                }
            })
        });

        group.bench_function("asm_square", |b| {
            b.iter(|| {
                for (a, _) in &pairs {
                    black_box(goldilocks_extensions_asm::square(*a));
                }
            })
        });

        group.finish();
    }

    // ============================================================
    // FP2 EXTENSION FIELD BENCHMARKS
    // ============================================================

    pub fn bench_fp2_add(c: &mut Criterion) {
        let mut group = c.benchmark_group("fp2_add");

        let pairs = generate_random_fp2_pairs(BATCH_SIZE, 55555);

        group.bench_function("native_fp2_add", |b| {
            b.iter(|| {
                for (a, b_val) in &pairs {
                    black_box(native_fp2_add(*a, *b_val));
                }
            })
        });

        group.bench_function("asm_fp2_add", |b| {
            b.iter(|| {
                for (a, b_val) in &pairs {
                    black_box(goldilocks_extensions_asm::fp2_add(*a, *b_val));
                }
            })
        });

        group.finish();
    }

    pub fn bench_fp2_mul(c: &mut Criterion) {
        let mut group = c.benchmark_group("fp2_mul");

        let pairs = generate_random_fp2_pairs(BATCH_SIZE, 66666);

        group.bench_function("native_fp2_mul", |b| {
            b.iter(|| {
                for (a, b_val) in &pairs {
                    black_box(native_fp2_mul(*a, *b_val));
                }
            })
        });

        group.bench_function("asm_fp2_mul", |b| {
            b.iter(|| {
                for (a, b_val) in &pairs {
                    black_box(goldilocks_extensions_asm::fp2_mul(*a, *b_val));
                }
            })
        });

        group.finish();
    }

    pub fn bench_fp2_square(c: &mut Criterion) {
        let mut group = c.benchmark_group("fp2_square");

        let pairs = generate_random_fp2_pairs(BATCH_SIZE, 77777);

        group.bench_function("native_fp2_square", |b| {
            b.iter(|| {
                for (a, _) in &pairs {
                    black_box(native_fp2_square(*a));
                }
            })
        });

        group.bench_function("asm_fp2_square", |b| {
            b.iter(|| {
                for (a, _) in &pairs {
                    black_box(goldilocks_extensions_asm::fp2_square(*a));
                }
            })
        });

        group.finish();
    }

    // ============================================================
    // FP3 EXTENSION FIELD BENCHMARKS
    // ============================================================

    pub fn bench_fp3_add(c: &mut Criterion) {
        let mut group = c.benchmark_group("fp3_add");

        let pairs = generate_random_fp3_pairs(BATCH_SIZE, 88888);

        group.bench_function("native_fp3_add", |b| {
            b.iter(|| {
                for (a, b_val) in &pairs {
                    black_box(native_fp3_add(*a, *b_val));
                }
            })
        });

        group.bench_function("asm_fp3_add", |b| {
            b.iter(|| {
                for (a, b_val) in &pairs {
                    black_box(goldilocks_extensions_asm::fp3_add(*a, *b_val));
                }
            })
        });

        group.finish();
    }

    pub fn bench_fp3_mul(c: &mut Criterion) {
        let mut group = c.benchmark_group("fp3_mul");

        let pairs = generate_random_fp3_pairs(BATCH_SIZE, 99999);

        group.bench_function("native_fp3_mul", |b| {
            b.iter(|| {
                for (a, b_val) in &pairs {
                    black_box(native_fp3_mul(*a, *b_val));
                }
            })
        });

        group.bench_function("asm_fp3_mul", |b| {
            b.iter(|| {
                for (a, b_val) in &pairs {
                    black_box(goldilocks_extensions_asm::fp3_mul(*a, *b_val));
                }
            })
        });

        group.finish();
    }

    pub fn bench_fp3_square(c: &mut Criterion) {
        let mut group = c.benchmark_group("fp3_square");

        let pairs = generate_random_fp3_pairs(BATCH_SIZE, 12121);

        group.bench_function("native_fp3_square", |b| {
            b.iter(|| {
                for (a, _) in &pairs {
                    black_box(native_fp3_square(*a));
                }
            })
        });

        group.bench_function("asm_fp3_square", |b| {
            b.iter(|| {
                for (a, _) in &pairs {
                    black_box(goldilocks_extensions_asm::fp3_square(*a));
                }
            })
        });

        group.finish();
    }

    // ============================================================
    // COMBINED OPERATIONS BENCHMARKS
    // ============================================================

    pub fn bench_combined_base_ops(c: &mut Criterion) {
        let mut group = c.benchmark_group("combined_base_ops");

        // Simulate: (a * b) + (c * d) - e
        let pairs = generate_random_pairs(BATCH_SIZE * 3, 13131);

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

    pub fn bench_combined_fp2_ops(c: &mut Criterion) {
        let mut group = c.benchmark_group("combined_fp2_ops");

        // Simulate: (a * b) + c^2
        let pairs = generate_random_fp2_pairs(BATCH_SIZE * 2, 14141);

        group.bench_function("native_fp2_combined", |b| {
            b.iter(|| {
                for i in (0..BATCH_SIZE).step_by(2) {
                    let (a, b_val) = pairs[i];
                    let (c, _) = pairs[i + 1];
                    let ab = native_fp2_mul(a, b_val);
                    let c2 = native_fp2_square(c);
                    black_box(native_fp2_add(ab, c2));
                }
            })
        });

        group.bench_function("asm_fp2_combined", |b| {
            b.iter(|| {
                for i in (0..BATCH_SIZE).step_by(2) {
                    let (a, b_val) = pairs[i];
                    let (c, _) = pairs[i + 1];
                    let ab = goldilocks_extensions_asm::fp2_mul(a, b_val);
                    let c2 = goldilocks_extensions_asm::fp2_square(c);
                    black_box(goldilocks_extensions_asm::fp2_add(ab, c2));
                }
            })
        });

        group.finish();
    }

    pub fn bench_combined_fp3_ops(c: &mut Criterion) {
        let mut group = c.benchmark_group("combined_fp3_ops");

        // Simulate: (a * b) + c^2
        let pairs = generate_random_fp3_pairs(BATCH_SIZE * 2, 15151);

        group.bench_function("native_fp3_combined", |b| {
            b.iter(|| {
                for i in (0..BATCH_SIZE).step_by(2) {
                    let (a, b_val) = pairs[i];
                    let (c, _) = pairs[i + 1];
                    let ab = native_fp3_mul(a, b_val);
                    let c2 = native_fp3_square(c);
                    black_box(native_fp3_add(ab, c2));
                }
            })
        });

        group.bench_function("asm_fp3_combined", |b| {
            b.iter(|| {
                for i in (0..BATCH_SIZE).step_by(2) {
                    let (a, b_val) = pairs[i];
                    let (c, _) = pairs[i + 1];
                    let ab = goldilocks_extensions_asm::fp3_mul(a, b_val);
                    let c2 = goldilocks_extensions_asm::fp3_square(c);
                    black_box(goldilocks_extensions_asm::fp3_add(ab, c2));
                }
            })
        });

        group.finish();
    }

    // ============================================================
    // SCALING BENCHMARKS
    // ============================================================

    pub fn bench_scaling(c: &mut Criterion) {
        let mut group = c.benchmark_group("scaling_base_mul");

        for size in [1000, 10000, 100000].iter() {
            let pairs = generate_random_pairs(*size, 16161);

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

    pub fn bench_fp2_scaling(c: &mut Criterion) {
        let mut group = c.benchmark_group("scaling_fp2_mul");

        for size in [1000, 10000, 50000].iter() {
            let pairs = generate_random_fp2_pairs(*size, 17171);

            group.bench_with_input(
                BenchmarkId::new("native_fp2_mul", size),
                size,
                |b, &size| {
                    b.iter(|| {
                        for i in 0..size {
                            let (a, b_val) = pairs[i];
                            black_box(native_fp2_mul(a, b_val));
                        }
                    })
                },
            );

            group.bench_with_input(BenchmarkId::new("asm_fp2_mul", size), size, |b, &size| {
                b.iter(|| {
                    for i in 0..size {
                        let (a, b_val) = pairs[i];
                        black_box(goldilocks_extensions_asm::fp2_mul(a, b_val));
                    }
                })
            });
        }

        group.finish();
    }
}

// ============================================================
// NON-ARM64 STUB
// ============================================================

#[cfg(not(all(feature = "asm-arm64", target_arch = "aarch64")))]
mod asm_benches {
    use super::*;

    pub fn bench_base_add(_c: &mut Criterion) {
        println!("ASM benchmarks require --features asm-arm64 on ARM64 platform");
    }
    pub fn bench_base_sub(_c: &mut Criterion) {}
    pub fn bench_base_mul(_c: &mut Criterion) {}
    pub fn bench_base_square(_c: &mut Criterion) {}
    pub fn bench_fp2_add(_c: &mut Criterion) {}
    pub fn bench_fp2_mul(_c: &mut Criterion) {}
    pub fn bench_fp2_square(_c: &mut Criterion) {}
    pub fn bench_fp3_add(_c: &mut Criterion) {}
    pub fn bench_fp3_mul(_c: &mut Criterion) {}
    pub fn bench_fp3_square(_c: &mut Criterion) {}
    pub fn bench_combined_base_ops(_c: &mut Criterion) {}
    pub fn bench_combined_fp2_ops(_c: &mut Criterion) {}
    pub fn bench_combined_fp3_ops(_c: &mut Criterion) {}
    pub fn bench_scaling(_c: &mut Criterion) {}
    pub fn bench_fp2_scaling(_c: &mut Criterion) {}
}

criterion_group!(
    goldilocks_asm_benches,
    // Base field operations
    asm_benches::bench_base_add,
    asm_benches::bench_base_sub,
    asm_benches::bench_base_mul,
    asm_benches::bench_base_square,
    // Fp2 extension field operations
    asm_benches::bench_fp2_add,
    asm_benches::bench_fp2_mul,
    asm_benches::bench_fp2_square,
    // Fp3 extension field operations
    asm_benches::bench_fp3_add,
    asm_benches::bench_fp3_mul,
    asm_benches::bench_fp3_square,
    // Combined operations
    asm_benches::bench_combined_base_ops,
    asm_benches::bench_combined_fp2_ops,
    asm_benches::bench_combined_fp3_ops,
    // Scaling
    asm_benches::bench_scaling,
    asm_benches::bench_fp2_scaling,
);

criterion_main!(goldilocks_asm_benches);
