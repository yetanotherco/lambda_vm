use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use math::field::element::FieldElement;
use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField;
use math::field::goldilocks::GoldilocksField;
use stark::examples::fibonacci_multi_column::{
    FibonacciMultiColumnAIR, FibonacciMultiColumnPublicInputs, compute_trace, create_public_inputs,
};
use stark::proof::options::ProofOptions;
use stark::proof::stark::StarkProof;
use stark::prover::{IsStarkProver, Prover};
use stark::verifier::{IsStarkVerifier, Verifier};

type F = GoldilocksField;
type E = Degree3GoldilocksExtensionField;
type FE = FieldElement<F>;

/// Configuration for a benchmark case
struct BenchConfig {
    name: &'static str,
    num_columns: usize,
    trace_length: usize,
}

impl BenchConfig {
    const fn new(name: &'static str, num_columns: usize, trace_length: usize) -> Self {
        Self {
            name,
            num_columns,
            trace_length,
        }
    }
}

/// Quick benchmark configurations
const QUICK_CONFIGS: &[BenchConfig] = &[
    BenchConfig::new("fib_2col_16k", 2, 16384),
    BenchConfig::new("fib_4col_8k", 4, 8192),
    BenchConfig::new("fib_4col_16k", 4, 16384),
];

/// Thorough benchmark configurations
const THOROUGH_CONFIGS: &[BenchConfig] = &[
    BenchConfig::new("fib_8col_16k", 8, 16384),
    BenchConfig::new("fib_12col_16k", 12, 16384),
    BenchConfig::new("fib_16col_8k", 16, 8192),
    BenchConfig::new("fib_16col_16k", 16, 16384),
];

/// Creates initial values for the specified number of columns
fn create_initial_values(num_columns: usize) -> Vec<(FE, FE)> {
    (0..num_columns)
        .map(|i| (FE::from((i + 1) as u64), FE::from((i + 2) as u64)))
        .collect()
}

/// Creates proof options suitable for benchmarking
fn benchmark_proof_options() -> ProofOptions {
    ProofOptions {
        blowup_factor: 4,
        fri_number_of_queries: 30,
        coset_offset: 3,
        grinding_factor: 0,
        fri_final_poly_log_degree: 7,
    }
}

/// Generates a proof for the given configuration
fn generate_proof(
    config: &BenchConfig,
) -> (
    StarkProof<F, E, FibonacciMultiColumnPublicInputs<F>>,
    FibonacciMultiColumnAIR<F, E>,
) {
    let proof_options = benchmark_proof_options();
    let initial_values = create_initial_values(config.num_columns);
    let mut trace = compute_trace::<F, E>(&initial_values, config.trace_length);
    let pub_inputs = create_public_inputs(initial_values);
    let air = FibonacciMultiColumnAIR::<F, E>::with_num_columns(&proof_options, config.num_columns);

    let proof = Prover::<F, E, _>::prove(
        &air,
        &mut trace,
        &pub_inputs,
        &mut DefaultTranscript::<E>::new(&[]),
    )
    .unwrap();

    (proof, air)
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
                    let initial_values = create_initial_values(config.num_columns);
                    let trace = compute_trace::<F, E>(&initial_values, config.trace_length);
                    let pub_inputs = create_public_inputs(initial_values);
                    let air = FibonacciMultiColumnAIR::<F, E>::with_num_columns(
                        &proof_options,
                        config.num_columns,
                    );
                    (trace, pub_inputs, air)
                },
                |(mut trace, pub_inputs, air)| {
                    Prover::<F, E, _>::prove(
                        &air,
                        &mut trace,
                        &pub_inputs,
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
    // Pre-generate the proof
    let (proof, air) = generate_proof(config);

    c.bench_with_input(
        BenchmarkId::new(format!("{}/verify", group_name), config.name),
        &(proof, air),
        |b, (proof, air)| {
            b.iter(|| {
                Verifier::<F, E, _>::verify(proof, air, &mut DefaultTranscript::<E>::new(&[]))
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

criterion_main!(quick, thorough);
