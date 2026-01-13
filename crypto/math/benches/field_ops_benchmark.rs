use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use rand::Rng;

// Lambda VM field types
use math::field::element::FieldElement;
use math::field::fields::fft_friendly::babybear::Babybear31PrimeField;
use math::field::fields::fft_friendly::u64_goldilocks::U64GoldilocksPrimeField;
use math::field::fields::fft_friendly::u64_goldilocks_native::GoldilocksField as GoldilocksNative;
use math::field::fields::fft_friendly::extensions_goldilocks_native::{
    Degree2GoldilocksNativeExtensionField, Degree3GoldilocksNativeExtensionField,
};

// Plonky3 field types
use p3_baby_bear::BabyBear;
use p3_goldilocks::Goldilocks;
use p3_field::{Field, PrimeCharacteristicRing, extension::BinomialExtensionField};

type LambdaBabyBear = FieldElement<Babybear31PrimeField>;
type LambdaGoldilocksMont = FieldElement<U64GoldilocksPrimeField>;  // Montgomery form
type LambdaGoldilocksNative = FieldElement<GoldilocksNative>;       // Native form
type LambdaGoldilocksFp2 = FieldElement<Degree2GoldilocksNativeExtensionField>;
type LambdaGoldilocksFp3 = FieldElement<Degree3GoldilocksNativeExtensionField>;

// Plonky3 quadratic extension of Goldilocks
type P3GoldilocksFp2 = BinomialExtensionField<Goldilocks, 2>;

const BATCH_SIZE: usize = 1000;

// ============== BabyBear generators ==============
fn generate_lambda_babybear(count: usize) -> Vec<LambdaBabyBear> {
    let mut rng = rand::thread_rng();
    (0..count)
        .map(|_| LambdaBabyBear::from(rng.r#gen::<u64>()))
        .collect()
}

fn generate_plonky3_babybear(count: usize) -> Vec<BabyBear> {
    let mut rng = rand::thread_rng();
    (0..count)
        .map(|_| BabyBear::new(rng.r#gen::<u32>() % (1 << 31)))
        .collect()
}

// ============== Goldilocks generators ==============
fn generate_lambda_goldilocks_mont(count: usize) -> Vec<LambdaGoldilocksMont> {
    let mut rng = rand::thread_rng();
    (0..count)
        .map(|_| LambdaGoldilocksMont::from(rng.r#gen::<u64>()))
        .collect()
}

fn generate_lambda_goldilocks_native(count: usize) -> Vec<LambdaGoldilocksNative> {
    let mut rng = rand::thread_rng();
    (0..count)
        .map(|_| LambdaGoldilocksNative::from(rng.r#gen::<u64>()))
        .collect()
}

fn generate_plonky3_goldilocks(count: usize) -> Vec<Goldilocks> {
    let mut rng = rand::thread_rng();
    (0..count)
        .map(|_| Goldilocks::new(rng.r#gen::<u64>()))
        .collect()
}

// ============== Goldilocks Fp2 generators ==============
fn generate_lambda_goldilocks_fp2(count: usize) -> Vec<LambdaGoldilocksFp2> {
    let mut rng = rand::thread_rng();
    (0..count)
        .map(|_| {
            LambdaGoldilocksFp2::new([
                LambdaGoldilocksNative::from(rng.r#gen::<u64>()),
                LambdaGoldilocksNative::from(rng.r#gen::<u64>()),
            ])
        })
        .collect()
}

fn generate_plonky3_goldilocks_fp2(count: usize) -> Vec<P3GoldilocksFp2> {
    let mut rng = rand::thread_rng();
    (0..count)
        .map(|_| {
            P3GoldilocksFp2::new([
                Goldilocks::new(rng.r#gen::<u64>()),
                Goldilocks::new(rng.r#gen::<u64>()),
            ])
        })
        .collect()
}

// ============== Goldilocks Fp3 generators ==============
fn generate_lambda_goldilocks_fp3(count: usize) -> Vec<LambdaGoldilocksFp3> {
    let mut rng = rand::thread_rng();
    (0..count)
        .map(|_| {
            LambdaGoldilocksFp3::new([
                LambdaGoldilocksNative::from(rng.r#gen::<u64>()),
                LambdaGoldilocksNative::from(rng.r#gen::<u64>()),
                LambdaGoldilocksNative::from(rng.r#gen::<u64>()),
            ])
        })
        .collect()
}

// ============== BABYBEAR BENCHMARKS ==============
fn bench_babybear_add(c: &mut Criterion) {
    let mut group = c.benchmark_group("babybear_add");
    group.throughput(Throughput::Elements(BATCH_SIZE as u64));

    let lambda_a = generate_lambda_babybear(BATCH_SIZE);
    let lambda_b = generate_lambda_babybear(BATCH_SIZE);
    let plonky3_a = generate_plonky3_babybear(BATCH_SIZE);
    let plonky3_b = generate_plonky3_babybear(BATCH_SIZE);

    group.bench_function("lambda_vm", |b| {
        b.iter(|| {
            for i in 0..BATCH_SIZE {
                black_box(&lambda_a[i] + &lambda_b[i]);
            }
        })
    });

    group.bench_function("plonky3", |b| {
        b.iter(|| {
            for i in 0..BATCH_SIZE {
                black_box(plonky3_a[i] + plonky3_b[i]);
            }
        })
    });

    group.finish();
}

fn bench_babybear_mul(c: &mut Criterion) {
    let mut group = c.benchmark_group("babybear_mul");
    group.throughput(Throughput::Elements(BATCH_SIZE as u64));

    let lambda_a = generate_lambda_babybear(BATCH_SIZE);
    let lambda_b = generate_lambda_babybear(BATCH_SIZE);
    let plonky3_a = generate_plonky3_babybear(BATCH_SIZE);
    let plonky3_b = generate_plonky3_babybear(BATCH_SIZE);

    group.bench_function("lambda_vm", |b| {
        b.iter(|| {
            for i in 0..BATCH_SIZE {
                black_box(&lambda_a[i] * &lambda_b[i]);
            }
        })
    });

    group.bench_function("plonky3", |b| {
        b.iter(|| {
            for i in 0..BATCH_SIZE {
                black_box(plonky3_a[i] * plonky3_b[i]);
            }
        })
    });

    group.finish();
}

fn bench_babybear_inv(c: &mut Criterion) {
    let mut group = c.benchmark_group("babybear_inv");
    group.throughput(Throughput::Elements(BATCH_SIZE as u64));

    let lambda_a: Vec<_> = generate_lambda_babybear(BATCH_SIZE)
        .into_iter()
        .map(|x| if x == LambdaBabyBear::zero() { LambdaBabyBear::one() } else { x })
        .collect();
    let plonky3_a: Vec<_> = generate_plonky3_babybear(BATCH_SIZE)
        .into_iter()
        .map(|x| if x == BabyBear::ZERO { BabyBear::ONE } else { x })
        .collect();

    group.bench_function("lambda_vm", |b| {
        b.iter(|| {
            for i in 0..BATCH_SIZE {
                black_box(lambda_a[i].inv().unwrap());
            }
        })
    });

    group.bench_function("plonky3", |b| {
        b.iter(|| {
            for i in 0..BATCH_SIZE {
                black_box(plonky3_a[i].inverse());
            }
        })
    });

    group.finish();
}

// ============== GOLDILOCKS BENCHMARKS ==============
fn bench_goldilocks_add(c: &mut Criterion) {
    let mut group = c.benchmark_group("goldilocks_add");
    group.throughput(Throughput::Elements(BATCH_SIZE as u64));

    let mont_a = generate_lambda_goldilocks_mont(BATCH_SIZE);
    let mont_b = generate_lambda_goldilocks_mont(BATCH_SIZE);
    let native_a = generate_lambda_goldilocks_native(BATCH_SIZE);
    let native_b = generate_lambda_goldilocks_native(BATCH_SIZE);
    let plonky3_a = generate_plonky3_goldilocks(BATCH_SIZE);
    let plonky3_b = generate_plonky3_goldilocks(BATCH_SIZE);

    group.bench_function("lambda_vm_montgomery", |b| {
        b.iter(|| {
            for i in 0..BATCH_SIZE {
                black_box(&mont_a[i] + &mont_b[i]);
            }
        })
    });

    group.bench_function("lambda_vm_native", |b| {
        b.iter(|| {
            for i in 0..BATCH_SIZE {
                black_box(&native_a[i] + &native_b[i]);
            }
        })
    });

    group.bench_function("plonky3", |b| {
        b.iter(|| {
            for i in 0..BATCH_SIZE {
                black_box(plonky3_a[i] + plonky3_b[i]);
            }
        })
    });

    group.finish();
}

fn bench_goldilocks_mul(c: &mut Criterion) {
    let mut group = c.benchmark_group("goldilocks_mul");
    group.throughput(Throughput::Elements(BATCH_SIZE as u64));

    let mont_a = generate_lambda_goldilocks_mont(BATCH_SIZE);
    let mont_b = generate_lambda_goldilocks_mont(BATCH_SIZE);
    let native_a = generate_lambda_goldilocks_native(BATCH_SIZE);
    let native_b = generate_lambda_goldilocks_native(BATCH_SIZE);
    let plonky3_a = generate_plonky3_goldilocks(BATCH_SIZE);
    let plonky3_b = generate_plonky3_goldilocks(BATCH_SIZE);

    group.bench_function("lambda_vm_montgomery", |b| {
        b.iter(|| {
            for i in 0..BATCH_SIZE {
                black_box(&mont_a[i] * &mont_b[i]);
            }
        })
    });

    group.bench_function("lambda_vm_native", |b| {
        b.iter(|| {
            for i in 0..BATCH_SIZE {
                black_box(&native_a[i] * &native_b[i]);
            }
        })
    });

    group.bench_function("plonky3", |b| {
        b.iter(|| {
            for i in 0..BATCH_SIZE {
                black_box(plonky3_a[i] * plonky3_b[i]);
            }
        })
    });

    group.finish();
}

fn bench_goldilocks_inv(c: &mut Criterion) {
    let mut group = c.benchmark_group("goldilocks_inv");
    group.throughput(Throughput::Elements(BATCH_SIZE as u64));

    let mont_a: Vec<_> = generate_lambda_goldilocks_mont(BATCH_SIZE)
        .into_iter()
        .map(|x| if x == LambdaGoldilocksMont::zero() { LambdaGoldilocksMont::one() } else { x })
        .collect();
    let native_a: Vec<_> = generate_lambda_goldilocks_native(BATCH_SIZE)
        .into_iter()
        .map(|x| if x == LambdaGoldilocksNative::zero() { LambdaGoldilocksNative::one() } else { x })
        .collect();
    let plonky3_a: Vec<_> = generate_plonky3_goldilocks(BATCH_SIZE)
        .into_iter()
        .map(|x| if x == Goldilocks::ZERO { Goldilocks::ONE } else { x })
        .collect();

    group.bench_function("lambda_vm_montgomery", |b| {
        b.iter(|| {
            for i in 0..BATCH_SIZE {
                black_box(mont_a[i].inv().unwrap());
            }
        })
    });

    group.bench_function("lambda_vm_native", |b| {
        b.iter(|| {
            for i in 0..BATCH_SIZE {
                black_box(native_a[i].inv().unwrap());
            }
        })
    });

    group.bench_function("plonky3", |b| {
        b.iter(|| {
            for i in 0..BATCH_SIZE {
                black_box(plonky3_a[i].inverse());
            }
        })
    });

    group.finish();
}

// ============== GOLDILOCKS FP2 BENCHMARKS ==============
fn bench_goldilocks_fp2_add(c: &mut Criterion) {
    let mut group = c.benchmark_group("goldilocks_fp2_add");
    group.throughput(Throughput::Elements(BATCH_SIZE as u64));

    let lambda_a = generate_lambda_goldilocks_fp2(BATCH_SIZE);
    let lambda_b = generate_lambda_goldilocks_fp2(BATCH_SIZE);
    let plonky3_a = generate_plonky3_goldilocks_fp2(BATCH_SIZE);
    let plonky3_b = generate_plonky3_goldilocks_fp2(BATCH_SIZE);

    group.bench_function("lambda_vm", |b| {
        b.iter(|| {
            for i in 0..BATCH_SIZE {
                black_box(&lambda_a[i] + &lambda_b[i]);
            }
        })
    });

    group.bench_function("plonky3", |b| {
        b.iter(|| {
            for i in 0..BATCH_SIZE {
                black_box(plonky3_a[i] + plonky3_b[i]);
            }
        })
    });

    group.finish();
}

fn bench_goldilocks_fp2_mul(c: &mut Criterion) {
    let mut group = c.benchmark_group("goldilocks_fp2_mul");
    group.throughput(Throughput::Elements(BATCH_SIZE as u64));

    let lambda_a = generate_lambda_goldilocks_fp2(BATCH_SIZE);
    let lambda_b = generate_lambda_goldilocks_fp2(BATCH_SIZE);
    let plonky3_a = generate_plonky3_goldilocks_fp2(BATCH_SIZE);
    let plonky3_b = generate_plonky3_goldilocks_fp2(BATCH_SIZE);

    group.bench_function("lambda_vm", |b| {
        b.iter(|| {
            for i in 0..BATCH_SIZE {
                black_box(&lambda_a[i] * &lambda_b[i]);
            }
        })
    });

    group.bench_function("plonky3", |b| {
        b.iter(|| {
            for i in 0..BATCH_SIZE {
                black_box(plonky3_a[i] * plonky3_b[i]);
            }
        })
    });

    group.finish();
}

fn bench_goldilocks_fp2_inv(c: &mut Criterion) {
    let mut group = c.benchmark_group("goldilocks_fp2_inv");
    group.throughput(Throughput::Elements(BATCH_SIZE as u64));

    let lambda_a: Vec<_> = generate_lambda_goldilocks_fp2(BATCH_SIZE)
        .into_iter()
        .map(|x| if x == LambdaGoldilocksFp2::zero() { LambdaGoldilocksFp2::one() } else { x })
        .collect();
    let plonky3_a: Vec<_> = generate_plonky3_goldilocks_fp2(BATCH_SIZE)
        .into_iter()
        .map(|x| if x == P3GoldilocksFp2::ZERO { P3GoldilocksFp2::ONE } else { x })
        .collect();

    group.bench_function("lambda_vm", |b| {
        b.iter(|| {
            for i in 0..BATCH_SIZE {
                black_box(lambda_a[i].inv().unwrap());
            }
        })
    });

    group.bench_function("plonky3", |b| {
        b.iter(|| {
            for i in 0..BATCH_SIZE {
                black_box(plonky3_a[i].inverse());
            }
        })
    });

    group.finish();
}

// ============== GOLDILOCKS FP3 BENCHMARKS (Lambda only) ==============
fn bench_goldilocks_fp3_add(c: &mut Criterion) {
    let mut group = c.benchmark_group("goldilocks_fp3_add");
    group.throughput(Throughput::Elements(BATCH_SIZE as u64));

    let lambda_a = generate_lambda_goldilocks_fp3(BATCH_SIZE);
    let lambda_b = generate_lambda_goldilocks_fp3(BATCH_SIZE);

    group.bench_function("lambda_vm", |b| {
        b.iter(|| {
            for i in 0..BATCH_SIZE {
                black_box(&lambda_a[i] + &lambda_b[i]);
            }
        })
    });

    group.finish();
}

fn bench_goldilocks_fp3_mul(c: &mut Criterion) {
    let mut group = c.benchmark_group("goldilocks_fp3_mul");
    group.throughput(Throughput::Elements(BATCH_SIZE as u64));

    let lambda_a = generate_lambda_goldilocks_fp3(BATCH_SIZE);
    let lambda_b = generate_lambda_goldilocks_fp3(BATCH_SIZE);

    group.bench_function("lambda_vm", |b| {
        b.iter(|| {
            for i in 0..BATCH_SIZE {
                black_box(&lambda_a[i] * &lambda_b[i]);
            }
        })
    });

    group.finish();
}

fn bench_goldilocks_fp3_inv(c: &mut Criterion) {
    let mut group = c.benchmark_group("goldilocks_fp3_inv");
    group.throughput(Throughput::Elements(BATCH_SIZE as u64));

    let lambda_a: Vec<_> = generate_lambda_goldilocks_fp3(BATCH_SIZE)
        .into_iter()
        .map(|x| if x == LambdaGoldilocksFp3::zero() { LambdaGoldilocksFp3::one() } else { x })
        .collect();

    group.bench_function("lambda_vm", |b| {
        b.iter(|| {
            for i in 0..BATCH_SIZE {
                black_box(lambda_a[i].inv().unwrap());
            }
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    // BabyBear
    bench_babybear_add,
    bench_babybear_mul,
    bench_babybear_inv,
    // Goldilocks
    bench_goldilocks_add,
    bench_goldilocks_mul,
    bench_goldilocks_inv,
    // Goldilocks Fp2
    bench_goldilocks_fp2_add,
    bench_goldilocks_fp2_mul,
    bench_goldilocks_fp2_inv,
    // Goldilocks Fp3 (Lambda only)
    bench_goldilocks_fp3_add,
    bench_goldilocks_fp3_mul,
    bench_goldilocks_fp3_inv,
);
criterion_main!(benches);
