//! Benchmarks for LogUp lookup implementation.
//!
//! Run with: `cargo bench -p stark --bench lookup_bench`

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use math::field::element::FieldElement;
use math::field::fields::fft_friendly::{
    extensions_goldilocks::Degree2GoldilocksExtensionField, u64_goldilocks::U64GoldilocksPrimeField,
};
use rand::Rng;
use stark::lookup::{Packing, SHIFT_8, SHIFT_16};

type F = U64GoldilocksPrimeField;
type E = Degree2GoldilocksExtensionField;
type FE = FieldElement<F>;
type EE = FieldElement<E>;

/// Generate random base field elements
fn generate_random_field_elements(n: usize) -> Vec<FE> {
    let mut rng = rand::thread_rng();
    (0..n)
        .map(|_| FE::from(rng.r#gen::<u64>() % 18446744069414584320))
        .collect()
}

/// Generate random extension field elements
fn generate_random_extension_elements(n: usize) -> Vec<EE> {
    let mut rng = rand::thread_rng();
    (0..n)
        .map(|_| {
            EE::new([
                FE::from(rng.r#gen::<u64>() % 18446744069414584320),
                FE::from(rng.r#gen::<u64>() % 18446744069414584320),
            ])
        })
        .collect()
}

/// Benchmark batch inversion vs individual inversions
fn bench_batch_vs_individual_inverse(c: &mut Criterion) {
    let mut group = c.benchmark_group("inversion");

    for n in [100, 1000, 10000] {
        // Individual inversions
        group.bench_with_input(BenchmarkId::new("individual", n), &n, |b, &n| {
            let elements = generate_random_field_elements(n);
            b.iter(|| {
                elements
                    .iter()
                    .map(|e| e.inv().unwrap())
                    .collect::<Vec<_>>()
            })
        });

        // Batch inversion
        group.bench_with_input(BenchmarkId::new("batch", n), &n, |b, &n| {
            b.iter_batched(
                || generate_random_field_elements(n),
                |mut elements| {
                    FieldElement::inplace_batch_inverse(&mut elements).unwrap();
                    elements
                },
                criterion::BatchSize::SmallInput,
            )
        });
    }
    group.finish();
}

/// Benchmark batch inversion for extension field
fn bench_extension_batch_inverse(c: &mut Criterion) {
    let mut group = c.benchmark_group("extension_inversion");

    for n in [100, 1000, 10000] {
        // Individual inversions
        group.bench_with_input(BenchmarkId::new("individual", n), &n, |b, &n| {
            let elements = generate_random_extension_elements(n);
            b.iter(|| {
                elements
                    .iter()
                    .map(|e| e.inv().unwrap())
                    .collect::<Vec<_>>()
            })
        });

        // Batch inversion
        group.bench_with_input(BenchmarkId::new("batch", n), &n, |b, &n| {
            b.iter_batched(
                || generate_random_extension_elements(n),
                |mut elements| {
                    FieldElement::inplace_batch_inverse(&mut elements).unwrap();
                    elements
                },
                criterion::BatchSize::SmallInput,
            )
        });
    }
    group.finish();
}

/// Benchmark packing combine operations
fn bench_packing_combine(c: &mut Criterion) {
    let mut group = c.benchmark_group("packing_combine");

    // Direct (1 column -> 1 element)
    group.bench_function("Direct", |b| {
        let columns = generate_random_field_elements(1);
        b.iter(|| Packing::Direct.combine(&columns))
    });

    // Word2L (2 columns -> 1 element)
    group.bench_function("Word2L", |b| {
        let columns = generate_random_field_elements(2);
        b.iter(|| Packing::Word2L.combine(&columns))
    });

    // Word4L (4 columns -> 1 element)
    group.bench_function("Word4L", |b| {
        let columns = generate_random_field_elements(4);
        b.iter(|| Packing::Word4L.combine(&columns))
    });

    // DWordHL (4 columns -> 2 elements)
    group.bench_function("DWordHL", |b| {
        let columns = generate_random_field_elements(4);
        b.iter(|| Packing::DWordHL.combine(&columns))
    });

    // DWordBL (8 columns -> 2 elements)
    group.bench_function("DWordBL", |b| {
        let columns = generate_random_field_elements(8);
        b.iter(|| Packing::DWordBL.combine(&columns))
    });

    // QuadHL (8 columns -> 4 elements)
    group.bench_function("QuadHL", |b| {
        let columns = generate_random_field_elements(8);
        b.iter(|| Packing::QuadHL.combine(&columns))
    });

    group.finish();
}

/// Benchmark shift constant creation (to measure overhead)
fn bench_shift_constant_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("shift_constants");

    // Creating shift constants every time (current approach)
    group.bench_function("create_each_time", |b| {
        b.iter(|| {
            let shift_8 = FE::from(SHIFT_8);
            let shift_16 = FE::from(SHIFT_16);
            let shift_24 = &shift_8 * &shift_16;
            (shift_8, shift_16, shift_24)
        })
    });

    // Precomputed constants (optimized approach)
    let shift_8 = FE::from(SHIFT_8);
    let shift_16 = FE::from(SHIFT_16);
    let shift_24 = &shift_8 * &shift_16;

    group.bench_function("use_precomputed", |b| {
        b.iter(|| (shift_8.clone(), shift_16.clone(), shift_24.clone()))
    });

    group.finish();
}

/// Benchmark alpha power computation
fn bench_alpha_powers(c: &mut Criterion) {
    let mut group = c.benchmark_group("alpha_powers");

    let alpha = generate_random_extension_elements(1)[0].clone();

    for num_elements in [3, 5, 8, 16] {
        // Compute powers on-the-fly using pow
        group.bench_with_input(
            BenchmarkId::new("pow_each_time", num_elements),
            &num_elements,
            |b, &n| {
                b.iter(|| {
                    (0..n)
                        .map(|i| alpha.pow(i))
                        .collect::<Vec<_>>()
                })
            },
        );

        // Compute powers incrementally (more efficient)
        group.bench_with_input(
            BenchmarkId::new("incremental", num_elements),
            &num_elements,
            |b, &n| {
                b.iter(|| {
                    let mut powers = Vec::with_capacity(n);
                    let mut current = EE::one();
                    for _ in 0..n {
                        powers.push(current.clone());
                        current = &current * &alpha;
                    }
                    powers
                })
            },
        );
    }

    group.finish();
}

/// Benchmark fingerprint computation pattern
fn bench_fingerprint_computation(c: &mut Criterion) {
    let mut group = c.benchmark_group("fingerprint");

    let z = generate_random_extension_elements(1)[0].clone();
    let alpha = generate_random_extension_elements(1)[0].clone();

    for num_elements in [3, 5, 8] {
        let values = generate_random_extension_elements(num_elements);

        // Current approach: compute alpha powers each time
        group.bench_with_input(
            BenchmarkId::new("current", num_elements),
            &num_elements,
            |b, &n| {
                b.iter(|| {
                    let alpha_powers: Vec<EE> = (0..n).map(|i| alpha.pow(i)).collect();
                    let linear_combination: EE = values
                        .iter()
                        .zip(alpha_powers.iter())
                        .map(|(v, coeff)| v * coeff)
                        .sum();
                    &z - &linear_combination
                })
            },
        );

        // Optimized: precomputed alpha powers
        let precomputed_alpha_powers: Vec<EE> = (0..num_elements).map(|i| alpha.pow(i)).collect();

        group.bench_with_input(
            BenchmarkId::new("precomputed", num_elements),
            &num_elements,
            |b, _| {
                b.iter(|| {
                    let linear_combination: EE = values
                        .iter()
                        .zip(precomputed_alpha_powers.iter())
                        .map(|(v, coeff)| v * coeff)
                        .sum();
                    &z - &linear_combination
                })
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_batch_vs_individual_inverse,
    bench_extension_batch_inverse,
    bench_packing_combine,
    bench_shift_constant_creation,
    bench_alpha_powers,
    bench_fingerprint_computation,
);
criterion_main!(benches);
