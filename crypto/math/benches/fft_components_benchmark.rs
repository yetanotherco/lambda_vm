use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::Rng;

// Lambda VM types
use math::field::element::FieldElement;
use math::field::fields::fft_friendly::u64_goldilocks::U64GoldilocksPrimeField;
use math::field::fields::fft_friendly::u64_goldilocks_native::GoldilocksField as GoldilocksNative;
use math::field::traits::RootsConfig;
use math::polynomial::Polynomial;

// Plonky3 types
use p3_goldilocks::Goldilocks;
use p3_dft::{Radix2Dit, TwoAdicSubgroupDft};
use p3_field::{PrimeCharacteristicRing, TwoAdicField};
use p3_matrix::dense::RowMajorMatrix;

type LambdaGoldilocksMont = FieldElement<U64GoldilocksPrimeField>;
type LambdaGoldilocksNative = FieldElement<GoldilocksNative>;

// ============== Generators ==============
fn generate_lambda_mont_poly(size: usize) -> Polynomial<LambdaGoldilocksMont> {
    let mut rng = rand::thread_rng();
    let coeffs: Vec<LambdaGoldilocksMont> = (0..size)
        .map(|_| LambdaGoldilocksMont::from(rng.r#gen::<u64>()))
        .collect();
    Polynomial::new(&coeffs)
}

fn generate_lambda_native_poly(size: usize) -> Polynomial<LambdaGoldilocksNative> {
    let mut rng = rand::thread_rng();
    let coeffs: Vec<LambdaGoldilocksNative> = (0..size)
        .map(|_| LambdaGoldilocksNative::from(rng.r#gen::<u64>()))
        .collect();
    Polynomial::new(&coeffs)
}

fn generate_lambda_mont_vec(size: usize) -> Vec<LambdaGoldilocksMont> {
    let mut rng = rand::thread_rng();
    (0..size)
        .map(|_| LambdaGoldilocksMont::from(rng.r#gen::<u64>()))
        .collect()
}

fn generate_lambda_native_vec(size: usize) -> Vec<LambdaGoldilocksNative> {
    let mut rng = rand::thread_rng();
    (0..size)
        .map(|_| LambdaGoldilocksNative::from(rng.r#gen::<u64>()))
        .collect()
}

fn generate_plonky3_vec(size: usize) -> Vec<Goldilocks> {
    let mut rng = rand::thread_rng();
    (0..size)
        .map(|_| Goldilocks::new(rng.r#gen::<u64>()))
        .collect()
}

// ============== COMPONENT 1: Twiddle Generation ==============
fn bench_twiddle_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("twiddle_generation");

    for log_size in [10, 12, 14, 16] {
        let size = 1 << log_size;
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(
            BenchmarkId::new("lambda_montgomery", log_size),
            &log_size,
            |b, &log_size| {
                b.iter(|| {
                    black_box(
                        math::fft::cpu::roots_of_unity::get_twiddles::<U64GoldilocksPrimeField>(
                            log_size as u64,
                            RootsConfig::BitReverse,
                        )
                        .unwrap(),
                    )
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("lambda_native", log_size),
            &log_size,
            |b, &log_size| {
                b.iter(|| {
                    black_box(
                        math::fft::cpu::roots_of_unity::get_twiddles::<GoldilocksNative>(
                            log_size as u64,
                            RootsConfig::BitReverse,
                        )
                        .unwrap(),
                    )
                })
            },
        );

        // Plonky3 generates twiddles on-the-fly in their DFT, but we can measure
        // primitive root computation for comparison
        group.bench_with_input(
            BenchmarkId::new("plonky3_root_powers", log_size),
            &log_size,
            |b, &log_size| {
                b.iter(|| {
                    let root = Goldilocks::two_adic_generator(log_size);
                    let mut powers = Vec::with_capacity(size / 2);
                    let mut current = Goldilocks::ONE;
                    for _ in 0..size / 2 {
                        powers.push(current);
                        current = current * root;
                    }
                    black_box(powers)
                })
            },
        );
    }

    group.finish();
}

// ============== COMPONENT 2: Polynomial Scaling (LDE preprocessing) ==============
fn bench_polynomial_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("polynomial_scaling");

    for log_size in [10, 12, 14, 16] {
        let size = 1 << log_size;
        group.throughput(Throughput::Elements(size as u64));

        let mont_poly = generate_lambda_mont_poly(size);
        let native_poly = generate_lambda_native_poly(size);
        let mont_offset = LambdaGoldilocksMont::from(7u64);
        let native_offset = LambdaGoldilocksNative::from(7u64);

        group.bench_with_input(
            BenchmarkId::new("lambda_montgomery", log_size),
            &log_size,
            |b, _| {
                b.iter(|| black_box(mont_poly.scale(&mont_offset)))
            },
        );

        group.bench_with_input(
            BenchmarkId::new("lambda_native", log_size),
            &log_size,
            |b, _| {
                b.iter(|| black_box(native_poly.scale(&native_offset)))
            },
        );

        // Plonky3's coset_shift applies shift to the domain, not polynomial coefficients
        // Their equivalent is to multiply each coeff[i] by shift^i
        let plonky3_coeffs = generate_plonky3_vec(size);
        let p3_shift = Goldilocks::new(7);
        group.bench_with_input(
            BenchmarkId::new("plonky3_coeff_scale", log_size),
            &log_size,
            |b, _| {
                b.iter(|| {
                    let mut scaled = Vec::with_capacity(size);
                    let mut power = Goldilocks::ONE;
                    for coeff in &plonky3_coeffs {
                        scaled.push(*coeff * power);
                        power = power * p3_shift;
                    }
                    black_box(scaled)
                })
            },
        );
    }

    group.finish();
}

// ============== COMPONENT 3: Bit Reversal Permutation ==============
fn bench_bit_reversal(c: &mut Criterion) {
    use criterion::BatchSize;
    let mut group = c.benchmark_group("bit_reversal");

    for log_size in [10, 12, 14, 16] {
        let size = 1 << log_size;
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(
            BenchmarkId::new("lambda_montgomery", log_size),
            &log_size,
            |b, _| {
                b.iter_batched(
                    || generate_lambda_mont_vec(size),
                    |mut data| {
                        math::fft::cpu::bit_reversing::in_place_bit_reverse_permute(&mut data);
                        data
                    },
                    BatchSize::SmallInput,
                )
            },
        );

        group.bench_with_input(
            BenchmarkId::new("lambda_native", log_size),
            &log_size,
            |b, _| {
                b.iter_batched(
                    || generate_lambda_native_vec(size),
                    |mut data| {
                        math::fft::cpu::bit_reversing::in_place_bit_reverse_permute(&mut data);
                        data
                    },
                    BatchSize::SmallInput,
                )
            },
        );

        // Plonky3's bit reversal (p3_util::reverse_slice_index_bits)
        group.bench_with_input(
            BenchmarkId::new("plonky3", log_size),
            &log_size,
            |b, _| {
                b.iter_batched(
                    || generate_plonky3_vec(size),
                    |mut data| {
                        p3_util::reverse_slice_index_bits(&mut data);
                        data
                    },
                    BatchSize::SmallInput,
                )
            },
        );
    }

    group.finish();
}

// ============== COMPONENT 4: FFT Butterflies Only ==============
fn bench_fft_butterflies(c: &mut Criterion) {
    use criterion::BatchSize;
    use math::field::traits::IsFFTField;
    let mut group = c.benchmark_group("fft_butterflies_only");

    for log_size in [10, 12, 14, 16] {
        let size = 1 << log_size;
        group.throughput(Throughput::Elements(size as u64));

        // Pre-compute twiddles (not counted in benchmark)
        let mont_twiddles = math::fft::cpu::roots_of_unity::get_twiddles::<U64GoldilocksPrimeField>(
            log_size as u64,
            RootsConfig::BitReverse,
        )
        .unwrap();

        let native_twiddles = math::fft::cpu::roots_of_unity::get_twiddles::<GoldilocksNative>(
            log_size as u64,
            RootsConfig::BitReverse,
        )
        .unwrap();

        // Get primitive root for on-the-fly FFT
        let native_primitive_root =
            GoldilocksNative::get_primitive_root_of_unity(log_size as u64).unwrap();

        group.bench_with_input(
            BenchmarkId::new("lambda_montgomery", log_size),
            &log_size,
            |b, _| {
                b.iter_batched(
                    || generate_lambda_mont_vec(size),
                    |mut data| {
                        math::fft::cpu::fft::in_place_nr_2radix_fft(&mut data, &mont_twiddles);
                        data
                    },
                    BatchSize::SmallInput,
                )
            },
        );

        group.bench_with_input(
            BenchmarkId::new("lambda_native", log_size),
            &log_size,
            |b, _| {
                b.iter_batched(
                    || generate_lambda_native_vec(size),
                    |mut data| {
                        math::fft::cpu::fft::in_place_nr_2radix_fft(&mut data, &native_twiddles);
                        data
                    },
                    BatchSize::SmallInput,
                )
            },
        );

        // On-the-fly twiddle generation variant
        group.bench_with_input(
            BenchmarkId::new("lambda_native_on_the_fly", log_size),
            &log_size,
            |b, _| {
                b.iter_batched(
                    || generate_lambda_native_vec(size),
                    |mut data| {
                        math::fft::cpu::fft::in_place_nr_2radix_fft_on_the_fly(
                            &mut data,
                            &native_primitive_root,
                        );
                        data
                    },
                    BatchSize::SmallInput,
                )
            },
        );

        // For Plonky3, we benchmark full DFT since butterflies aren't exposed
        let dft = Radix2Dit::default();
        group.bench_with_input(
            BenchmarkId::new("plonky3_full_dft", log_size),
            &log_size,
            |b, _| {
                b.iter_batched(
                    || generate_plonky3_vec(size),
                    |data| {
                        let mat = RowMajorMatrix::new(data, 1);
                        dft.dft_batch(mat)
                    },
                    BatchSize::SmallInput,
                )
            },
        );
    }

    group.finish();
}

// ============== COMPONENT 5: Full FFT Pipeline ==============
fn bench_full_fft_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("full_fft_pipeline");

    for log_size in [10, 12, 14, 16] {
        let size = 1 << log_size;
        group.throughput(Throughput::Elements(size as u64));

        let mont_poly = generate_lambda_mont_poly(size);
        let native_poly = generate_lambda_native_poly(size);
        let plonky3_coeffs = generate_plonky3_vec(size);

        group.bench_with_input(
            BenchmarkId::new("lambda_montgomery", log_size),
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
            BenchmarkId::new("lambda_native", log_size),
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

// ============== COMPONENT 6: LDE (Coset FFT) Pipeline ==============
fn bench_lde_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("lde_pipeline");
    group.sample_size(10);

    for log_size in [10, 12, 14, 16, 18, 20] {
        let size = 1 << log_size;
        let blowup = 2;
        group.throughput(Throughput::Elements((size * blowup) as u64));

        let mont_poly = generate_lambda_mont_poly(size);
        let native_poly = generate_lambda_native_poly(size);
        let plonky3_coeffs = generate_plonky3_vec(size);
        let mont_offset = LambdaGoldilocksMont::from(7u64);
        let native_offset = LambdaGoldilocksNative::from(7u64);

        group.bench_with_input(
            BenchmarkId::new("lambda_montgomery", log_size),
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
            BenchmarkId::new("lambda_native", log_size),
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

// ============== COMPONENT 7: Field Operations in Butterfly Context ==============
fn bench_butterfly_field_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("butterfly_field_ops");

    // Single butterfly: (a, b) = (a + wb, a - wb)
    // Operations: 1 mul, 2 add/sub
    let iterations = 1_000_000;
    group.throughput(Throughput::Elements(iterations));

    let mut rng = rand::thread_rng();

    // Montgomery
    let mont_a = LambdaGoldilocksMont::from(rng.r#gen::<u64>());
    let mont_b = LambdaGoldilocksMont::from(rng.r#gen::<u64>());
    let mont_w = LambdaGoldilocksMont::from(rng.r#gen::<u64>());

    group.bench_function("lambda_montgomery_butterfly", |b| {
        b.iter(|| {
            let mut a = mont_a;
            let mut b_val = mont_b;
            for _ in 0..iterations {
                let wb = &mont_w * &b_val;
                let y0 = &a + &wb;
                let y1 = &a - &wb;
                a = y0;
                b_val = y1;
            }
            black_box((a, b_val))
        })
    });

    // Native
    let native_a = LambdaGoldilocksNative::from(rng.r#gen::<u64>());
    let native_b = LambdaGoldilocksNative::from(rng.r#gen::<u64>());
    let native_w = LambdaGoldilocksNative::from(rng.r#gen::<u64>());

    group.bench_function("lambda_native_butterfly", |b| {
        b.iter(|| {
            let mut a = native_a;
            let mut b_val = native_b;
            for _ in 0..iterations {
                let wb = &native_w * &b_val;
                let y0 = &a + &wb;
                let y1 = &a - &wb;
                a = y0;
                b_val = y1;
            }
            black_box((a, b_val))
        })
    });

    // Plonky3
    let p3_a = Goldilocks::new(rng.r#gen::<u64>());
    let p3_b = Goldilocks::new(rng.r#gen::<u64>());
    let p3_w = Goldilocks::new(rng.r#gen::<u64>());

    group.bench_function("plonky3_butterfly", |b| {
        b.iter(|| {
            let mut a = p3_a;
            let mut b_val = p3_b;
            for _ in 0..iterations {
                let wb = p3_w * b_val;
                let y0 = a + wb;
                let y1 = a - wb;
                a = y0;
                b_val = y1;
            }
            black_box((a, b_val))
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_twiddle_generation,
    bench_polynomial_scaling,
    bench_bit_reversal,
    bench_fft_butterflies,
    bench_full_fft_pipeline,
    bench_lde_pipeline,
    bench_butterfly_field_ops,
);
criterion_main!(benches);
