use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::Rng;

// Lambda VM types
use math::field::element::FieldElement;
use math::field::fields::fft_friendly::babybear::Babybear31PrimeField;
use math::field::fields::fft_friendly::u64_goldilocks::U64GoldilocksPrimeField;
use math::field::fields::fft_friendly::u64_goldilocks_native::GoldilocksField as GoldilocksNative;
use math::polynomial::Polynomial;

// Plonky3 types
use p3_baby_bear::BabyBear;
use p3_goldilocks::Goldilocks;
use p3_dft::{Radix2Dit, TwoAdicSubgroupDft};
use p3_matrix::dense::RowMajorMatrix;

type LambdaBabyBear = FieldElement<Babybear31PrimeField>;
type LambdaGoldilocksMont = FieldElement<U64GoldilocksPrimeField>;
type LambdaGoldilocksNative = FieldElement<GoldilocksNative>;

// ============== BabyBear generators ==============
fn generate_lambda_babybear_poly(size: usize) -> Polynomial<LambdaBabyBear> {
    let mut rng = rand::thread_rng();
    let coeffs: Vec<LambdaBabyBear> = (0..size)
        .map(|_| LambdaBabyBear::from(rng.r#gen::<u64>()))
        .collect();
    Polynomial::new(&coeffs)
}

fn generate_lambda_babybear_evals(size: usize) -> Vec<LambdaBabyBear> {
    let mut rng = rand::thread_rng();
    (0..size)
        .map(|_| LambdaBabyBear::from(rng.r#gen::<u64>()))
        .collect()
}

fn generate_plonky3_babybear_vec(size: usize) -> Vec<BabyBear> {
    let mut rng = rand::thread_rng();
    (0..size)
        .map(|_| BabyBear::new(rng.r#gen::<u32>() % (1 << 31)))
        .collect()
}

// ============== Goldilocks generators (Montgomery) ==============
fn generate_lambda_goldilocks_mont_poly(size: usize) -> Polynomial<LambdaGoldilocksMont> {
    let mut rng = rand::thread_rng();
    let coeffs: Vec<LambdaGoldilocksMont> = (0..size)
        .map(|_| LambdaGoldilocksMont::from(rng.r#gen::<u64>()))
        .collect();
    Polynomial::new(&coeffs)
}

fn generate_lambda_goldilocks_mont_evals(size: usize) -> Vec<LambdaGoldilocksMont> {
    let mut rng = rand::thread_rng();
    (0..size)
        .map(|_| LambdaGoldilocksMont::from(rng.r#gen::<u64>()))
        .collect()
}

// ============== Goldilocks generators (Native) ==============
fn generate_lambda_goldilocks_native_poly(size: usize) -> Polynomial<LambdaGoldilocksNative> {
    let mut rng = rand::thread_rng();
    let coeffs: Vec<LambdaGoldilocksNative> = (0..size)
        .map(|_| LambdaGoldilocksNative::from(rng.r#gen::<u64>()))
        .collect();
    Polynomial::new(&coeffs)
}

fn generate_lambda_goldilocks_native_evals(size: usize) -> Vec<LambdaGoldilocksNative> {
    let mut rng = rand::thread_rng();
    (0..size)
        .map(|_| LambdaGoldilocksNative::from(rng.r#gen::<u64>()))
        .collect()
}

fn generate_plonky3_goldilocks_vec(size: usize) -> Vec<Goldilocks> {
    let mut rng = rand::thread_rng();
    (0..size)
        .map(|_| Goldilocks::new(rng.r#gen::<u64>()))
        .collect()
}

// ============== BABYBEAR FFT BENCHMARKS ==============
fn bench_babybear_fft_evaluate(c: &mut Criterion) {
    let mut group = c.benchmark_group("babybear_fft_evaluate");

    for log_size in [10, 12, 14, 16] {
        let size = 1 << log_size;
        group.throughput(Throughput::Elements(size as u64));

        let lambda_poly = generate_lambda_babybear_poly(size);
        let plonky3_coeffs = generate_plonky3_babybear_vec(size);

        group.bench_with_input(
            BenchmarkId::new("lambda_vm", log_size),
            &log_size,
            |b, _| {
                b.iter(|| {
                    black_box(
                        Polynomial::evaluate_fft::<Babybear31PrimeField>(&lambda_poly, 1, None)
                            .unwrap(),
                    )
                })
            },
        );

        let dft = Radix2Dit::default();
        group.bench_with_input(
            BenchmarkId::new("plonky3", log_size),
            &log_size,
            |b, _| {
                b.iter(|| {
                    let mat = RowMajorMatrix::new(plonky3_coeffs.clone(), 1);
                    black_box(dft.dft_batch(mat))
                })
            },
        );
    }

    group.finish();
}

fn bench_babybear_fft_interpolate(c: &mut Criterion) {
    let mut group = c.benchmark_group("babybear_fft_interpolate");

    for log_size in [10, 12, 14, 16] {
        let size = 1 << log_size;
        group.throughput(Throughput::Elements(size as u64));

        let lambda_evals = generate_lambda_babybear_evals(size);
        let plonky3_evals = generate_plonky3_babybear_vec(size);

        group.bench_with_input(
            BenchmarkId::new("lambda_vm", log_size),
            &log_size,
            |b, _| {
                b.iter(|| {
                    black_box(
                        Polynomial::interpolate_fft::<Babybear31PrimeField>(&lambda_evals).unwrap(),
                    )
                })
            },
        );

        let dft = Radix2Dit::default();
        group.bench_with_input(
            BenchmarkId::new("plonky3", log_size),
            &log_size,
            |b, _| {
                b.iter(|| {
                    let mat = RowMajorMatrix::new(plonky3_evals.clone(), 1);
                    black_box(dft.idft_batch(mat))
                })
            },
        );
    }

    group.finish();
}

fn bench_babybear_fft_coset(c: &mut Criterion) {
    let mut group = c.benchmark_group("babybear_fft_coset");
    let offset = LambdaBabyBear::from(7u64);

    for log_size in [10, 12, 14] {
        let size = 1 << log_size;
        let blowup = 2;
        group.throughput(Throughput::Elements((size * blowup) as u64));

        let lambda_poly = generate_lambda_babybear_poly(size);
        let plonky3_coeffs = generate_plonky3_babybear_vec(size);

        group.bench_with_input(
            BenchmarkId::new("lambda_vm", log_size),
            &log_size,
            |b, _| {
                b.iter(|| {
                    black_box(
                        Polynomial::evaluate_offset_fft::<Babybear31PrimeField>(
                            &lambda_poly,
                            blowup,
                            None,
                            &offset,
                        )
                        .unwrap(),
                    )
                })
            },
        );

        let dft = Radix2Dit::default();
        let p3_shift = BabyBear::new(7);
        group.bench_with_input(
            BenchmarkId::new("plonky3", log_size),
            &log_size,
            |b, _| {
                b.iter(|| {
                    let mat = RowMajorMatrix::new(plonky3_coeffs.clone(), 1);
                    black_box(dft.coset_lde_batch(mat, blowup.trailing_zeros() as usize, p3_shift))
                })
            },
        );
    }

    group.finish();
}

// ============== GOLDILOCKS FFT BENCHMARKS ==============
fn bench_goldilocks_fft_evaluate(c: &mut Criterion) {
    let mut group = c.benchmark_group("goldilocks_fft_evaluate");

    for log_size in [10, 12, 14, 16] {
        let size = 1 << log_size;
        group.throughput(Throughput::Elements(size as u64));

        let mont_poly = generate_lambda_goldilocks_mont_poly(size);
        let native_poly = generate_lambda_goldilocks_native_poly(size);
        let plonky3_coeffs = generate_plonky3_goldilocks_vec(size);

        group.bench_with_input(
            BenchmarkId::new("lambda_vm_montgomery", log_size),
            &log_size,
            |b, _| {
                b.iter(|| {
                    black_box(
                        Polynomial::evaluate_fft::<U64GoldilocksPrimeField>(&mont_poly, 1, None)
                            .unwrap(),
                    )
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("lambda_vm_native", log_size),
            &log_size,
            |b, _| {
                b.iter(|| {
                    black_box(
                        Polynomial::evaluate_fft::<GoldilocksNative>(&native_poly, 1, None)
                            .unwrap(),
                    )
                })
            },
        );

        let dft = Radix2Dit::default();
        group.bench_with_input(
            BenchmarkId::new("plonky3", log_size),
            &log_size,
            |b, _| {
                b.iter(|| {
                    let mat = RowMajorMatrix::new(plonky3_coeffs.clone(), 1);
                    black_box(dft.dft_batch(mat))
                })
            },
        );
    }

    group.finish();
}

fn bench_goldilocks_fft_interpolate(c: &mut Criterion) {
    let mut group = c.benchmark_group("goldilocks_fft_interpolate");

    for log_size in [10, 12, 14, 16] {
        let size = 1 << log_size;
        group.throughput(Throughput::Elements(size as u64));

        let mont_evals = generate_lambda_goldilocks_mont_evals(size);
        let native_evals = generate_lambda_goldilocks_native_evals(size);
        let plonky3_evals = generate_plonky3_goldilocks_vec(size);

        group.bench_with_input(
            BenchmarkId::new("lambda_vm_montgomery", log_size),
            &log_size,
            |b, _| {
                b.iter(|| {
                    black_box(
                        Polynomial::interpolate_fft::<U64GoldilocksPrimeField>(&mont_evals).unwrap(),
                    )
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("lambda_vm_native", log_size),
            &log_size,
            |b, _| {
                b.iter(|| {
                    black_box(
                        Polynomial::interpolate_fft::<GoldilocksNative>(&native_evals).unwrap(),
                    )
                })
            },
        );

        let dft = Radix2Dit::default();
        group.bench_with_input(
            BenchmarkId::new("plonky3", log_size),
            &log_size,
            |b, _| {
                b.iter(|| {
                    let mat = RowMajorMatrix::new(plonky3_evals.clone(), 1);
                    black_box(dft.idft_batch(mat))
                })
            },
        );
    }

    group.finish();
}

fn bench_goldilocks_fft_coset(c: &mut Criterion) {
    let mut group = c.benchmark_group("goldilocks_fft_coset");
    let mont_offset = LambdaGoldilocksMont::from(7u64);
    let native_offset = LambdaGoldilocksNative::from(7u64);

    for log_size in [10, 12, 14] {
        let size = 1 << log_size;
        let blowup = 2;
        group.throughput(Throughput::Elements((size * blowup) as u64));

        let mont_poly = generate_lambda_goldilocks_mont_poly(size);
        let native_poly = generate_lambda_goldilocks_native_poly(size);
        let plonky3_coeffs = generate_plonky3_goldilocks_vec(size);

        group.bench_with_input(
            BenchmarkId::new("lambda_vm_montgomery", log_size),
            &log_size,
            |b, _| {
                b.iter(|| {
                    black_box(
                        Polynomial::evaluate_offset_fft::<U64GoldilocksPrimeField>(
                            &mont_poly,
                            blowup,
                            None,
                            &mont_offset,
                        )
                        .unwrap(),
                    )
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("lambda_vm_native", log_size),
            &log_size,
            |b, _| {
                b.iter(|| {
                    black_box(
                        Polynomial::evaluate_offset_fft::<GoldilocksNative>(
                            &native_poly,
                            blowup,
                            None,
                            &native_offset,
                        )
                        .unwrap(),
                    )
                })
            },
        );

        let dft = Radix2Dit::default();
        let p3_shift = Goldilocks::new(7);
        group.bench_with_input(
            BenchmarkId::new("plonky3", log_size),
            &log_size,
            |b, _| {
                b.iter(|| {
                    let mat = RowMajorMatrix::new(plonky3_coeffs.clone(), 1);
                    black_box(dft.coset_lde_batch(mat, blowup.trailing_zeros() as usize, p3_shift))
                })
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    // BabyBear FFT
    bench_babybear_fft_evaluate,
    bench_babybear_fft_interpolate,
    bench_babybear_fft_coset,
    // Goldilocks FFT
    bench_goldilocks_fft_evaluate,
    bench_goldilocks_fft_interpolate,
    bench_goldilocks_fft_coset,
);
criterion_main!(benches);
