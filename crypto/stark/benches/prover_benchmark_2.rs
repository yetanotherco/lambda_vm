use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use math::field::fields::fft_friendly::{
    babybear::Babybear31PrimeField, quartic_babybear::Degree4BabyBearExtensionField,
};
use stark::{
    examples::multi_table_lookup::{
        generate_random_traces, new_add_air_with_lookup, new_cpu_air_with_lookup,
        new_mul_air_with_lookup,
    },
    proof::options::ProofOptions,
    prover::{IsStarkProver, Prover},
    traits::AIR,
    verifier::{IsStarkVerifier, Verifier},
};

type F = Babybear31PrimeField;
type E = Degree4BabyBearExtensionField;

/// Configuration for a benchmark case
struct BenchConfig {
    name: &'static str,
    trace_length: usize,
}

impl BenchConfig {
    const fn new(name: &'static str, trace_length: usize) -> Self {
        Self { name, trace_length }
    }
}

/// Quick benchmark configurations
const QUICK_CONFIGS: &[BenchConfig] = &[
    BenchConfig::new("config_1", 8192),
    BenchConfig::new("config_2", 16384),
];

/// Thorough benchmark configurations
const THOROUGH_CONFIGS: &[BenchConfig] = &[
    BenchConfig::new("config_3", 65536),      // 2^16 = 65.536
    BenchConfig::new("config_4", 4294967296), // 2^32 = 4.294.967.296
];

/// Creates proof options suitable for benchmarking
fn benchmark_proof_options() -> ProofOptions {
    ProofOptions {
        blowup_factor: 4,
        fri_number_of_queries: 30,
        coset_offset: 3,
        grinding_factor: 0,
    }
}

/// Benchmark proving for a single configuration
fn bench_prove(c: &mut Criterion, group_name: &str, config: &BenchConfig) {
    let proof_options = benchmark_proof_options();

    c.bench_with_input(
        BenchmarkId::new(format!("{}/prove", group_name), config.name),
        config,
        |b, config| {
            b.iter_with_setup(
                || {
                    let (cpu_trace, add_trace, mul_trace) =
                        generate_random_traces(config.trace_length, None);
                    let cpu_air = new_cpu_air_with_lookup(&proof_options);
                    let add_air = new_add_air_with_lookup(&proof_options);
                    let mul_air = new_mul_air_with_lookup(&proof_options);
                    (cpu_trace, add_trace, mul_trace, cpu_air, add_air, mul_air)
                },
                |(mut cpu_trace, mut add_trace, mut mul_trace, cpu_air, add_air, mul_air)| {
                    let air_trace_pairs: Vec<(
                        &dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>,
                        _,
                        _,
                    )> = vec![
                        (&cpu_air, &mut cpu_trace, &()),
                        (&add_air, &mut add_trace, &()),
                        (&mul_air, &mut mul_trace, &()),
                    ];

                    Prover::<F, E, _>::multi_prove(
                        air_trace_pairs,
                        &mut DefaultTranscript::<E>::new(&[]),
                    )
                    .unwrap()
                },
            )
        },
    );
}

/// Benchmark verification for a single configuration
fn bench_verify(c: &mut Criterion, group_name: &str, config: &BenchConfig) {
    // Pre-generate the proof and AIRs
    let proof_options = benchmark_proof_options();
    let (mut cpu_trace, mut add_trace, mut mul_trace) =
        generate_random_traces(config.trace_length, None);

    let cpu_air = new_cpu_air_with_lookup(&proof_options);
    let add_air = new_add_air_with_lookup(&proof_options);
    let mul_air = new_mul_air_with_lookup(&proof_options);

    let air_trace_pairs: Vec<(
        &dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>,
        _,
        _,
    )> = vec![
        (&cpu_air, &mut cpu_trace, &()),
        (&add_air, &mut add_trace, &()),
        (&mul_air, &mut mul_trace, &()),
    ];

    let proof =
        Prover::multi_prove(air_trace_pairs, &mut DefaultTranscript::<E>::new(&[])).unwrap();

    c.bench_with_input(
        BenchmarkId::new(format!("{}/verify", group_name), config.name),
        config,
        |b, _| {
            b.iter(|| {
                let airs: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>> =
                    vec![&cpu_air, &add_air, &mul_air];
                Verifier::<F, E, _>::multi_verify(
                    &airs,
                    &proof,
                    &mut DefaultTranscript::<E>::new(&[]),
                )
            })
        },
    );
}

/// Quick benchmarks
fn quick_benchmarks(c: &mut Criterion) {
    for config in QUICK_CONFIGS {
        bench_prove(c, "quick", config);
        bench_verify(c, "quick", config);
    }
}

/// Thorough benchmarks
fn thorough_benchmarks(c: &mut Criterion) {
    for config in THOROUGH_CONFIGS {
        bench_prove(c, "thorough", config);
        bench_verify(c, "thorough", config);
    }
}

criterion_group! {
    name = quick;
    config = Criterion::default()
        .sample_size(10)
        .measurement_time(std::time::Duration::from_secs(30));
    targets = quick_benchmarks
}

criterion_group! {
    name = thorough;
    config = Criterion::default()
        .sample_size(10)
        .measurement_time(std::time::Duration::from_secs(60));
    targets = thorough_benchmarks
}

criterion_main!(quick);
