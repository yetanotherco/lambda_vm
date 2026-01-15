//! Benchmarks for PackedGoldilocks2 and PackedFp2x2 SIMD operations.
//!
//! Run with: cargo bench --bench packed_goldilocks

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use math::field::fields::u64_goldilocks_field::Goldilocks64Field;
use math::field::simd::{Fp2Raw, Fp3Raw, PackedFp2x2, PackedFp3x2, PackedGoldilocks2};
use math::field::traits::IsField;

const P: u64 = 0xFFFF_FFFF_0000_0001;

fn bench_add(c: &mut Criterion) {
    let mut group = c.benchmark_group("goldilocks_add");

    // Scalar operations (2 elements)
    let a0 = P - 12345;
    let a1 = P - 67890;
    let b0 = 54321u64;
    let b1 = 98765u64;

    group.throughput(Throughput::Elements(2));

    group.bench_function("scalar_2x", |bench| {
        bench.iter(|| {
            let r0 = Goldilocks64Field::add(black_box(&a0), black_box(&b0));
            let r1 = Goldilocks64Field::add(black_box(&a1), black_box(&b1));
            black_box((r0, r1))
        });
    });

    // SIMD operations
    let a_packed = PackedGoldilocks2::from_array([a0, a1]);
    let b_packed = PackedGoldilocks2::from_array([b0, b1]);

    group.bench_function("simd_2x", |bench| {
        bench.iter(|| black_box(black_box(a_packed) + black_box(b_packed)));
    });

    group.finish();
}

fn bench_sub(c: &mut Criterion) {
    let mut group = c.benchmark_group("goldilocks_sub");

    let a0 = 12345u64;
    let a1 = 67890u64;
    let b0 = P - 54321;
    let b1 = P - 98765;

    group.throughput(Throughput::Elements(2));

    group.bench_function("scalar_2x", |bench| {
        bench.iter(|| {
            let r0 = Goldilocks64Field::sub(black_box(&a0), black_box(&b0));
            let r1 = Goldilocks64Field::sub(black_box(&a1), black_box(&b1));
            black_box((r0, r1))
        });
    });

    let a_packed = PackedGoldilocks2::from_array([a0, a1]);
    let b_packed = PackedGoldilocks2::from_array([b0, b1]);

    group.bench_function("simd_2x", |bench| {
        bench.iter(|| black_box(black_box(a_packed) - black_box(b_packed)));
    });

    group.finish();
}

fn bench_mul(c: &mut Criterion) {
    let mut group = c.benchmark_group("goldilocks_mul");

    let a0 = P - 12345;
    let a1 = P - 67890;
    let b0 = P - 54321;
    let b1 = P - 98765;

    group.throughput(Throughput::Elements(2));

    group.bench_function("scalar_2x", |bench| {
        bench.iter(|| {
            let r0 = Goldilocks64Field::mul(black_box(&a0), black_box(&b0));
            let r1 = Goldilocks64Field::mul(black_box(&a1), black_box(&b1));
            black_box((r0, r1))
        });
    });

    let a_packed = PackedGoldilocks2::from_array([a0, a1]);
    let b_packed = PackedGoldilocks2::from_array([b0, b1]);

    group.bench_function("simd_2x", |bench| {
        bench.iter(|| black_box(black_box(a_packed) * black_box(b_packed)));
    });

    group.finish();
}

fn bench_neg(c: &mut Criterion) {
    let mut group = c.benchmark_group("goldilocks_neg");

    let a0 = P - 12345;
    let a1 = P - 67890;

    group.throughput(Throughput::Elements(2));

    group.bench_function("scalar_2x", |bench| {
        bench.iter(|| {
            let r0 = Goldilocks64Field::neg(black_box(&a0));
            let r1 = Goldilocks64Field::neg(black_box(&a1));
            black_box((r0, r1))
        });
    });

    let a_packed = PackedGoldilocks2::from_array([a0, a1]);

    group.bench_function("simd_2x", |bench| {
        bench.iter(|| black_box(-black_box(a_packed)));
    });

    group.finish();
}

fn bench_combined_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("goldilocks_combined");

    // Simulate a typical FFT butterfly: (a + wb, a - wb)
    let a0 = 12345u64;
    let a1 = 67890u64;
    let w0 = P - 111;
    let w1 = P - 222;
    let b0 = 54321u64;
    let b1 = 98765u64;

    group.throughput(Throughput::Elements(2));

    group.bench_function("butterfly_scalar_2x", |bench| {
        bench.iter(|| {
            let wb0 = Goldilocks64Field::mul(black_box(&w0), black_box(&b0));
            let wb1 = Goldilocks64Field::mul(black_box(&w1), black_box(&b1));
            let y0_0 = Goldilocks64Field::add(black_box(&a0), &wb0);
            let y0_1 = Goldilocks64Field::add(black_box(&a1), &wb1);
            let y1_0 = Goldilocks64Field::sub(black_box(&a0), &wb0);
            let y1_1 = Goldilocks64Field::sub(black_box(&a1), &wb1);
            black_box((y0_0, y0_1, y1_0, y1_1))
        });
    });

    let a_packed = PackedGoldilocks2::from_array([a0, a1]);
    let w_packed = PackedGoldilocks2::from_array([w0, w1]);
    let b_packed = PackedGoldilocks2::from_array([b0, b1]);

    group.bench_function("butterfly_simd_2x", |bench| {
        bench.iter(|| {
            let wb = black_box(w_packed) * black_box(b_packed);
            let y0 = black_box(a_packed) + wb;
            let y1 = black_box(a_packed) - wb;
            black_box((y0, y1))
        });
    });

    group.finish();
}

fn bench_batch_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("goldilocks_batch");

    const BATCH_SIZE: usize = 1024;

    // Create test data
    let scalar_a: Vec<u64> = (0..BATCH_SIZE).map(|i| (i as u64 * 12345) % P).collect();
    let scalar_b: Vec<u64> = (0..BATCH_SIZE).map(|i| (i as u64 * 67890) % P).collect();

    let packed_a: Vec<PackedGoldilocks2> = scalar_a
        .chunks(2)
        .map(|c| PackedGoldilocks2::from_array([c[0], c[1]]))
        .collect();
    let packed_b: Vec<PackedGoldilocks2> = scalar_b
        .chunks(2)
        .map(|c| PackedGoldilocks2::from_array([c[0], c[1]]))
        .collect();

    group.throughput(Throughput::Elements(BATCH_SIZE as u64));

    group.bench_with_input(
        BenchmarkId::new("add_scalar", BATCH_SIZE),
        &(&scalar_a, &scalar_b),
        |bench, (a, b)| {
            bench.iter(|| {
                let mut result = Vec::with_capacity(BATCH_SIZE);
                for i in 0..BATCH_SIZE {
                    result.push(Goldilocks64Field::add(&a[i], &b[i]));
                }
                black_box(result)
            });
        },
    );

    group.bench_with_input(
        BenchmarkId::new("add_simd", BATCH_SIZE),
        &(&packed_a, &packed_b),
        |bench, (a, b)| {
            bench.iter(|| {
                let mut result = Vec::with_capacity(BATCH_SIZE / 2);
                for i in 0..BATCH_SIZE / 2 {
                    result.push(a[i] + b[i]);
                }
                black_box(result)
            });
        },
    );

    group.bench_with_input(
        BenchmarkId::new("mul_scalar", BATCH_SIZE),
        &(&scalar_a, &scalar_b),
        |bench, (a, b)| {
            bench.iter(|| {
                let mut result = Vec::with_capacity(BATCH_SIZE);
                for i in 0..BATCH_SIZE {
                    result.push(Goldilocks64Field::mul(&a[i], &b[i]));
                }
                black_box(result)
            });
        },
    );

    group.bench_with_input(
        BenchmarkId::new("mul_simd", BATCH_SIZE),
        &(&packed_a, &packed_b),
        |bench, (a, b)| {
            bench.iter(|| {
                let mut result = Vec::with_capacity(BATCH_SIZE / 2);
                for i in 0..BATCH_SIZE / 2 {
                    result.push(a[i] * b[i]);
                }
                black_box(result)
            });
        },
    );

    group.finish();
}

// ============================================================================
// Fp2 Benchmarks
// ============================================================================

fn bench_fp2_mul(c: &mut Criterion) {
    let mut group = c.benchmark_group("fp2_mul");

    let a0 = Fp2Raw::new(P - 12345, P - 67890);
    let a1 = Fp2Raw::new(P - 11111, P - 22222);
    let b0 = Fp2Raw::new(P - 54321, P - 98765);
    let b1 = Fp2Raw::new(P - 33333, P - 44444);

    group.throughput(Throughput::Elements(2));

    group.bench_function("scalar_2x", |bench| {
        bench.iter(|| {
            let r0 = Fp2Raw::mul(black_box(a0), black_box(b0));
            let r1 = Fp2Raw::mul(black_box(a1), black_box(b1));
            black_box((r0, r1))
        });
    });

    let packed_a = PackedFp2x2::from_array([a0, a1]);
    let packed_b = PackedFp2x2::from_array([b0, b1]);

    group.bench_function("simd_2x", |bench| {
        bench.iter(|| black_box(black_box(packed_a) * black_box(packed_b)));
    });

    group.finish();
}

fn bench_fp2_add(c: &mut Criterion) {
    let mut group = c.benchmark_group("fp2_add");

    let a0 = Fp2Raw::new(P - 12345, P - 67890);
    let a1 = Fp2Raw::new(P - 11111, P - 22222);
    let b0 = Fp2Raw::new(54321, 98765);
    let b1 = Fp2Raw::new(33333, 44444);

    group.throughput(Throughput::Elements(2));

    group.bench_function("scalar_2x", |bench| {
        bench.iter(|| {
            let r0 = Fp2Raw::add(black_box(a0), black_box(b0));
            let r1 = Fp2Raw::add(black_box(a1), black_box(b1));
            black_box((r0, r1))
        });
    });

    let packed_a = PackedFp2x2::from_array([a0, a1]);
    let packed_b = PackedFp2x2::from_array([b0, b1]);

    group.bench_function("simd_2x", |bench| {
        bench.iter(|| black_box(black_box(packed_a) + black_box(packed_b)));
    });

    group.finish();
}

fn bench_fp_times_fp2(c: &mut Criterion) {
    let mut group = c.benchmark_group("fp_times_fp2");

    // Scalar * Fp2 element
    let scalar0 = P - 12345;
    let scalar1 = P - 67890;
    let fp2_0 = Fp2Raw::new(54321, 98765);
    let fp2_1 = Fp2Raw::new(33333, 44444);

    group.throughput(Throughput::Elements(2));

    group.bench_function("scalar_2x", |bench| {
        bench.iter(|| {
            let r0 = Fp2Raw::new(
                Goldilocks64Field::mul(&scalar0, &fp2_0.real),
                Goldilocks64Field::mul(&scalar0, &fp2_0.imag),
            );
            let r1 = Fp2Raw::new(
                Goldilocks64Field::mul(&scalar1, &fp2_1.real),
                Goldilocks64Field::mul(&scalar1, &fp2_1.imag),
            );
            black_box((r0, r1))
        });
    });

    let packed_scalar = PackedGoldilocks2::from_array([scalar0, scalar1]);
    let packed_fp2 = PackedFp2x2::from_array([fp2_0, fp2_1]);

    group.bench_function("simd_2x", |bench| {
        bench.iter(|| black_box(black_box(packed_fp2).mul_by_fp(black_box(packed_scalar))));
    });

    group.finish();
}

fn bench_fp2_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("fp2_batch");

    const BATCH_SIZE: usize = 512;

    // Create test data
    let scalar_a: Vec<Fp2Raw> = (0..BATCH_SIZE)
        .map(|i| Fp2Raw::new((i as u64 * 12345) % P, (i as u64 * 67890) % P))
        .collect();
    let scalar_b: Vec<Fp2Raw> = (0..BATCH_SIZE)
        .map(|i| Fp2Raw::new((i as u64 * 11111) % P, (i as u64 * 22222) % P))
        .collect();

    let packed_a: Vec<PackedFp2x2> = scalar_a
        .chunks(2)
        .map(|c| PackedFp2x2::from_array([c[0], c[1]]))
        .collect();
    let packed_b: Vec<PackedFp2x2> = scalar_b
        .chunks(2)
        .map(|c| PackedFp2x2::from_array([c[0], c[1]]))
        .collect();

    group.throughput(Throughput::Elements(BATCH_SIZE as u64));

    group.bench_with_input(
        BenchmarkId::new("mul_scalar", BATCH_SIZE),
        &(&scalar_a, &scalar_b),
        |bench, (a, b)| {
            bench.iter(|| {
                let mut result = Vec::with_capacity(BATCH_SIZE);
                for i in 0..BATCH_SIZE {
                    result.push(Fp2Raw::mul(a[i], b[i]));
                }
                black_box(result)
            });
        },
    );

    group.bench_with_input(
        BenchmarkId::new("mul_simd", BATCH_SIZE),
        &(&packed_a, &packed_b),
        |bench, (a, b)| {
            bench.iter(|| {
                let mut result = Vec::with_capacity(BATCH_SIZE / 2);
                for i in 0..BATCH_SIZE / 2 {
                    result.push(a[i] * b[i]);
                }
                black_box(result)
            });
        },
    );

    group.finish();
}

// ============================================================================
// Fp3 Benchmarks
// ============================================================================

fn bench_fp3_add(c: &mut Criterion) {
    let mut group = c.benchmark_group("fp3_add");

    let a0 = Fp3Raw::new(P - 12345, P - 67890, P - 11111);
    let a1 = Fp3Raw::new(P - 22222, P - 33333, P - 44444);
    let b0 = Fp3Raw::new(54321, 98765, 55555);
    let b1 = Fp3Raw::new(66666, 77777, 88888);

    group.throughput(Throughput::Elements(2));

    group.bench_function("scalar_2x", |bench| {
        bench.iter(|| {
            let r0 = Fp3Raw::add(black_box(a0), black_box(b0));
            let r1 = Fp3Raw::add(black_box(a1), black_box(b1));
            black_box((r0, r1))
        });
    });

    let packed_a = PackedFp3x2::from_array([a0, a1]);
    let packed_b = PackedFp3x2::from_array([b0, b1]);

    group.bench_function("simd_2x", |bench| {
        bench.iter(|| black_box(black_box(packed_a) + black_box(packed_b)));
    });

    group.finish();
}

fn bench_fp3_mul(c: &mut Criterion) {
    let mut group = c.benchmark_group("fp3_mul");

    let a0 = Fp3Raw::new(P - 12345, P - 67890, P - 11111);
    let a1 = Fp3Raw::new(P - 22222, P - 33333, P - 44444);
    let b0 = Fp3Raw::new(P - 54321, P - 98765, P - 55555);
    let b1 = Fp3Raw::new(P - 66666, P - 77777, P - 88888);

    group.throughput(Throughput::Elements(2));

    group.bench_function("scalar_2x", |bench| {
        bench.iter(|| {
            let r0 = Fp3Raw::mul(black_box(a0), black_box(b0));
            let r1 = Fp3Raw::mul(black_box(a1), black_box(b1));
            black_box((r0, r1))
        });
    });

    let packed_a = PackedFp3x2::from_array([a0, a1]);
    let packed_b = PackedFp3x2::from_array([b0, b1]);

    group.bench_function("simd_2x", |bench| {
        bench.iter(|| black_box(black_box(packed_a) * black_box(packed_b)));
    });

    group.finish();
}

fn bench_fp_times_fp3(c: &mut Criterion) {
    let mut group = c.benchmark_group("fp_times_fp3");

    // Scalar * Fp3 element
    let scalar0 = P - 12345;
    let scalar1 = P - 67890;
    let fp3_0 = Fp3Raw::new(54321, 98765, 11111);
    let fp3_1 = Fp3Raw::new(33333, 44444, 55555);

    group.throughput(Throughput::Elements(2));

    group.bench_function("scalar_2x", |bench| {
        bench.iter(|| {
            let r0 = Fp3Raw::mul_by_fp(black_box(fp3_0), black_box(scalar0));
            let r1 = Fp3Raw::mul_by_fp(black_box(fp3_1), black_box(scalar1));
            black_box((r0, r1))
        });
    });

    let packed_scalar = PackedGoldilocks2::from_array([scalar0, scalar1]);
    let packed_fp3 = PackedFp3x2::from_array([fp3_0, fp3_1]);

    group.bench_function("simd_2x", |bench| {
        bench.iter(|| black_box(black_box(packed_fp3).mul_by_fp(black_box(packed_scalar))));
    });

    group.finish();
}

fn bench_fp3_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("fp3_batch");

    const BATCH_SIZE: usize = 512;

    // Create test data
    let scalar_a: Vec<Fp3Raw> = (0..BATCH_SIZE)
        .map(|i| {
            Fp3Raw::new(
                (i as u64 * 12345) % P,
                (i as u64 * 67890) % P,
                (i as u64 * 11111) % P,
            )
        })
        .collect();
    let scalar_b: Vec<Fp3Raw> = (0..BATCH_SIZE)
        .map(|i| {
            Fp3Raw::new(
                (i as u64 * 11111) % P,
                (i as u64 * 22222) % P,
                (i as u64 * 33333) % P,
            )
        })
        .collect();

    let packed_a: Vec<PackedFp3x2> = scalar_a
        .chunks(2)
        .map(|c| PackedFp3x2::from_array([c[0], c[1]]))
        .collect();
    let packed_b: Vec<PackedFp3x2> = scalar_b
        .chunks(2)
        .map(|c| PackedFp3x2::from_array([c[0], c[1]]))
        .collect();

    group.throughput(Throughput::Elements(BATCH_SIZE as u64));

    group.bench_with_input(
        BenchmarkId::new("mul_scalar", BATCH_SIZE),
        &(&scalar_a, &scalar_b),
        |bench, (a, b)| {
            bench.iter(|| {
                let mut result = Vec::with_capacity(BATCH_SIZE);
                for i in 0..BATCH_SIZE {
                    result.push(Fp3Raw::mul(a[i], b[i]));
                }
                black_box(result)
            });
        },
    );

    group.bench_with_input(
        BenchmarkId::new("mul_simd", BATCH_SIZE),
        &(&packed_a, &packed_b),
        |bench, (a, b)| {
            bench.iter(|| {
                let mut result = Vec::with_capacity(BATCH_SIZE / 2);
                for i in 0..BATCH_SIZE / 2 {
                    result.push(a[i] * b[i]);
                }
                black_box(result)
            });
        },
    );

    group.finish();
}

criterion_group!(
    benches,
    bench_add,
    bench_sub,
    bench_mul,
    bench_neg,
    bench_combined_ops,
    bench_batch_operations,
    bench_fp2_add,
    bench_fp2_mul,
    bench_fp_times_fp2,
    bench_fp2_batch,
    bench_fp3_add,
    bench_fp3_mul,
    bench_fp_times_fp3,
    bench_fp3_batch,
);

criterion_main!(benches);
