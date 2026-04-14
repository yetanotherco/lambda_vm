use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use math::field::element::FieldElement;
use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField;
use math::field::goldilocks::GoldilocksField;
use p3_uni_stark::{prove as p3_prove, verify as p3_verify};
use stark::examples::fibonacci_multi_column::{
    FibonacciMultiColumnAIR, compute_trace, create_public_inputs,
};
use stark::proof::options::ProofOptions;
use stark::prover::{IsStarkProver, Prover};
use stark::verifier::{IsStarkVerifier, Verifier};

use bench_vs_plonky3::plonky3_config;
use bench_vs_plonky3::plonky3_fibonacci;

type F = GoldilocksField;
type E = Degree3GoldilocksExtensionField;
type FE = FieldElement<F>;

/// Number of independent Fibonacci sequences (Lambda columns).
const NUM_SEQUENCES: usize = 2;

/// Lambda trace lengths to benchmark.
/// Plonky3 uses half the rows (2 cols per sequence).
const TRACE_SIZES: &[(usize, &str)] = &[
    (1 << 12, "2^12"),
    (1 << 14, "2^14"),
    (1 << 16, "2^16"),
    (1 << 18, "2^18"),
    (1 << 20, "2^20"),
];

/// Lambda benchmark proof options: blowup=4, 30 queries, no grinding.
fn benchmark_proof_options() -> ProofOptions {
    ProofOptions {
        blowup_factor: 4,
        fri_number_of_queries: 30,
        coset_offset: 3,
        grinding_factor: 0,
    }
}

fn lambda_initial_values() -> Vec<(FE, FE)> {
    (0..NUM_SEQUENCES)
        .map(|i| (FE::from((i + 1) as u64), FE::from((i + 2) as u64)))
        .collect()
}

fn bench_lambda_prove(c: &mut Criterion) {
    let mut group = c.benchmark_group("lambda_stark/prove");
    let proof_options = benchmark_proof_options();

    for &(trace_length, label) in TRACE_SIZES {
        group.bench_with_input(
            BenchmarkId::new("fibonacci", label),
            &trace_length,
            |b, &trace_length| {
                b.iter_with_setup(
                    || {
                        let initial_values = lambda_initial_values();
                        let trace = compute_trace::<F, E>(&initial_values, trace_length);
                        let pub_inputs = create_public_inputs(initial_values);
                        let air = FibonacciMultiColumnAIR::<F, E>::with_num_columns(
                            &proof_options,
                            NUM_SEQUENCES,
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
                );
            },
        );
    }
    group.finish();
}

fn bench_plonky3_prove(c: &mut Criterion) {
    let mut group = c.benchmark_group("plonky3_stark/prove");

    for &(trace_length, label) in TRACE_SIZES {
        // Plonky3 uses half the rows (2 cols per sequence vs Lambda's 1)
        let p3_rows = trace_length / 2;

        group.bench_with_input(
            BenchmarkId::new("fibonacci", label),
            &p3_rows,
            |b, &p3_rows| {
                b.iter_with_setup(
                    || {
                        let config = plonky3_config::matched_params_config();
                        let air = plonky3_fibonacci::P3FibonacciAir {
                            num_sequences: NUM_SEQUENCES,
                        };
                        let trace =
                            plonky3_fibonacci::generate_fibonacci_trace(NUM_SEQUENCES, p3_rows);
                        (config, air, trace)
                    },
                    |(config, air, trace)| p3_prove(&config, &air, trace, &[]),
                );
            },
        );
    }
    group.finish();
}

fn bench_lambda_verify(c: &mut Criterion) {
    let mut group = c.benchmark_group("lambda_stark/verify");
    let proof_options = benchmark_proof_options();

    for &(trace_length, label) in TRACE_SIZES {
        // Pre-generate proof
        let initial_values = lambda_initial_values();
        let mut trace = compute_trace::<F, E>(&initial_values, trace_length);
        let pub_inputs = create_public_inputs(initial_values);
        let air = FibonacciMultiColumnAIR::<F, E>::with_num_columns(
            &proof_options,
            NUM_SEQUENCES,
        );
        let proof = Prover::<F, E, _>::prove(
            &air,
            &mut trace,
            &pub_inputs,
            &mut DefaultTranscript::<E>::new(&[]),
        )
        .unwrap();

        group.bench_with_input(
            BenchmarkId::new("fibonacci", label),
            &trace_length,
            |b, _| {
                b.iter(|| {
                    assert!(Verifier::<F, E, _>::verify(
                        &proof,
                        &air,
                        &mut DefaultTranscript::<E>::new(&[]),
                    ))
                });
            },
        );
    }
    group.finish();
}

fn bench_plonky3_verify(c: &mut Criterion) {
    let mut group = c.benchmark_group("plonky3_stark/verify");

    for &(trace_length, label) in TRACE_SIZES {
        let p3_rows = trace_length / 2;

        // Pre-generate proof
        let air = plonky3_fibonacci::P3FibonacciAir {
            num_sequences: NUM_SEQUENCES,
        };
        let trace = plonky3_fibonacci::generate_fibonacci_trace(NUM_SEQUENCES, p3_rows);
        let config = plonky3_config::matched_params_config();
        let proof = p3_prove(&config, &air, trace, &[]);

        group.bench_with_input(
            BenchmarkId::new("fibonacci", label),
            &p3_rows,
            |b, _| {
                b.iter(|| {
                    let config = plonky3_config::matched_params_config();
                    p3_verify(&config, &air, &proof, &[]).unwrap();
                });
            },
        );
    }
    group.finish();
}

criterion_group! {
    name = prove_comparison;
    config = Criterion::default()
        .sample_size(10)
        .measurement_time(std::time::Duration::from_secs(60));
    targets = bench_lambda_prove, bench_plonky3_prove
}

criterion_group! {
    name = verify_comparison;
    config = Criterion::default()
        .sample_size(10)
        .measurement_time(std::time::Duration::from_secs(30));
    targets = bench_lambda_verify, bench_plonky3_verify
}

criterion_main!(prove_comparison, verify_comparison);
