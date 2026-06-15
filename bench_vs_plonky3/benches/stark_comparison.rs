use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use math::field::element::FieldElement;
use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField;
use math::field::goldilocks::GoldilocksField;
use p3_uni_stark::{prove as p3_prove, verify as p3_verify};
use stark::proof::options::ProofOptions;
use stark::prover::{IsStarkProver, Prover};
use stark::verifier::{IsStarkVerifier, Verifier};

use bench_vs_plonky3::lambda_fibonacci_pair;
use bench_vs_plonky3::plonky3_config;
use bench_vs_plonky3::plonky3_fibonacci;

type F = GoldilocksField;
type E = Degree3GoldilocksExtensionField;
type FE = FieldElement<F>;

/// Number of independent Fibonacci sequences.
const NUM_SEQUENCES: usize = 16;

/// Rows (same for both Lambda and Plonky3 — identical AIR shape).
///
/// 2^18 rows × 2 Fibonacci steps packed per row = 2^19 effective Fibonacci
/// steps per sequence, matching Lambda's original `FibonacciMultiColumnAIR`
/// at 2^19 rows × 1 step/row.
const ROWS: usize = 1 << 18;
const TRACE_LABEL: &str = "fib_pair_16seq_2^18";

/// Production proof options: blowup=2, 219 queries (from
/// `GoldilocksCubicProofOptions::with_blowup(2)`), grinding=0 (excluded
/// from benchmark — identical PoW work on both sides, not informative).
fn benchmark_proof_options() -> ProofOptions {
    ProofOptions {
        blowup_factor: 2,
        fri_number_of_queries: 219,
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
    group.throughput(Throughput::Elements((ROWS * 2 * NUM_SEQUENCES) as u64));
    let proof_options = benchmark_proof_options();

    group.bench_with_input(
        BenchmarkId::new("fibonacci", TRACE_LABEL),
        &ROWS,
        |b, &rows| {
            b.iter_with_setup(
                || {
                    let initial_values = lambda_initial_values();
                    let trace = lambda_fibonacci_pair::compute_trace::<F, E>(&initial_values, rows);
                    let pub_inputs = lambda_fibonacci_pair::create_public_inputs(initial_values);
                    let air =
                        lambda_fibonacci_pair::FibonacciPairMultiColAIR::<F, E>::with_num_sequences(
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
    group.finish();
}

fn bench_plonky3_prove(c: &mut Criterion) {
    let mut group = c.benchmark_group("plonky3_stark/prove");
    group.throughput(Throughput::Elements((ROWS * 2 * NUM_SEQUENCES) as u64));

    group.bench_with_input(
        BenchmarkId::new("fibonacci", TRACE_LABEL),
        &ROWS,
        |b, &rows| {
            b.iter_with_setup(
                || {
                    let config = plonky3_config::matched_params_config();
                    let air = plonky3_fibonacci::P3FibonacciAir {
                        num_sequences: NUM_SEQUENCES,
                    };
                    let trace = plonky3_fibonacci::generate_fibonacci_trace(NUM_SEQUENCES, rows);
                    let pis = plonky3_fibonacci::public_values(NUM_SEQUENCES);
                    (config, air, trace, pis)
                },
                |(config, air, trace, pis)| p3_prove(&config, &air, trace, &pis),
            );
        },
    );
    group.finish();
}

fn bench_lambda_verify(c: &mut Criterion) {
    let mut group = c.benchmark_group("lambda_stark/verify");
    group.throughput(Throughput::Elements((ROWS * 2 * NUM_SEQUENCES) as u64));
    let proof_options = benchmark_proof_options();

    let initial_values = lambda_initial_values();
    let mut trace = lambda_fibonacci_pair::compute_trace::<F, E>(&initial_values, ROWS);
    let pub_inputs = lambda_fibonacci_pair::create_public_inputs(initial_values);
    let air = lambda_fibonacci_pair::FibonacciPairMultiColAIR::<F, E>::with_num_sequences(
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

    group.bench_with_input(BenchmarkId::new("fibonacci", TRACE_LABEL), &ROWS, |b, _| {
        b.iter(|| {
            assert!(Verifier::<F, E, _>::verify(
                &proof,
                &air,
                &mut DefaultTranscript::<E>::new(&[]),
            ))
        });
    });
    group.finish();
}

fn bench_plonky3_verify(c: &mut Criterion) {
    let mut group = c.benchmark_group("plonky3_stark/verify");
    group.throughput(Throughput::Elements((ROWS * 2 * NUM_SEQUENCES) as u64));

    let air = plonky3_fibonacci::P3FibonacciAir {
        num_sequences: NUM_SEQUENCES,
    };
    let trace = plonky3_fibonacci::generate_fibonacci_trace(NUM_SEQUENCES, ROWS);
    let pis = plonky3_fibonacci::public_values(NUM_SEQUENCES);
    let config = plonky3_config::matched_params_config();
    let proof = p3_prove(&config, &air, trace, &pis);

    let verify_config = plonky3_config::matched_params_config();
    group.bench_with_input(BenchmarkId::new("fibonacci", TRACE_LABEL), &ROWS, |b, _| {
        b.iter(|| {
            p3_verify(&verify_config, &air, &proof, &pis).unwrap();
        });
    });
    group.finish();
}

criterion_group! {
    name = prove_comparison;
    config = Criterion::default()
        .sample_size(10)
        .measurement_time(std::time::Duration::from_secs(120));
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
