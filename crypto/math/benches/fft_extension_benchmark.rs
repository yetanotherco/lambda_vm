//! Benchmark comparing FFT performance between base field and cubic extension field.
//!
//! Run with: cargo bench -p math --features asm-arm64 --bench fft_extension_benchmark

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use rand::Rng;

use math::field::element::FieldElement;
use math::field::fields::fft_friendly::extensions_goldilocks_native::Fp3E;
use math::field::fields::fft_friendly::u64_goldilocks_native::GoldilocksField;
use math::polynomial::Polynomial;

type FpE = FieldElement<GoldilocksField>;

const LOG_SIZE: u32 = 20;
const SIZE: usize = 1 << LOG_SIZE;

fn generate_base_field_elements(size: usize) -> Vec<FpE> {
    let mut rng = rand::thread_rng();
    (0..size).map(|_| FpE::from(rng.r#gen::<u64>())).collect()
}

fn generate_extension_field_elements(base_elements: &[FpE]) -> Vec<Fp3E> {
    // Cast base field elements to cubic extension by embedding
    // (a, 0, 0) representation in Fp3
    base_elements
        .iter()
        .map(|e| {
            let zero = FpE::zero();
            Fp3E::new([e.clone(), zero.clone(), zero])
        })
        .collect()
}

fn bench_fft_base_vs_extension(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft_base_vs_extension_2^20");

    // Generate random base field elements
    let base_elements = generate_base_field_elements(SIZE);
    let base_poly = Polynomial::new(&base_elements);

    // Cast to extension field
    let ext_elements = generate_extension_field_elements(&base_elements);
    let ext_poly = Polynomial::new(&ext_elements);

    println!("\nBenchmarking FFT for 2^{} = {} elements", LOG_SIZE, SIZE);

    // Benchmark base field FFT
    group.bench_function("goldilocks_base_field", |b| {
        b.iter(|| {
            black_box(Polynomial::evaluate_fft::<GoldilocksField>(&base_poly, 1, None).unwrap())
        })
    });

    // Benchmark extension field FFT
    group.bench_function("goldilocks_cubic_extension", |b| {
        b.iter(|| {
            black_box(Polynomial::evaluate_fft::<GoldilocksField>(&ext_poly, 1, None).unwrap())
        })
    });

    group.finish();
}

fn bench_fft_interpolate_base_vs_extension(c: &mut Criterion) {
    let mut group = c.benchmark_group("ifft_base_vs_extension_2^20");

    // Generate random base field elements (as evaluations)
    let base_evals = generate_base_field_elements(SIZE);

    // Cast to extension field
    let ext_evals = generate_extension_field_elements(&base_evals);

    // Benchmark base field IFFT
    group.bench_function("goldilocks_base_field", |b| {
        b.iter(|| black_box(Polynomial::interpolate_fft::<GoldilocksField>(&base_evals).unwrap()))
    });

    // Benchmark extension field IFFT
    group.bench_function("goldilocks_cubic_extension", |b| {
        b.iter(|| black_box(Polynomial::interpolate_fft::<GoldilocksField>(&ext_evals).unwrap()))
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_fft_base_vs_extension,
    bench_fft_interpolate_base_vs_extension,
);
criterion_main!(benches);
