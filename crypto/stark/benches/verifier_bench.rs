use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use math::field::fields::fft_friendly::stark_252_prime_field::Stark252PrimeField;
use stark::Felt252;
use stark::examples::simple_fibonacci::{self, FibonacciAIR, FibonacciPublicInputs};
use stark::proof::options::ProofOptions;
use stark::proof::stark::StarkProof;
use stark::prover::{IsStarkProver, Prover};
use stark::traits::AIR;
use stark::transcript::StoneProverTranscript;
use stark::verifier::{IsStarkVerifier, Verifier};

type F = Stark252PrimeField;

fn generate_proof(
    trace_length: usize,
    num_queries: usize,
) -> (StarkProof<F, F>, FibonacciAIR<F>, FibonacciPublicInputs<F>) {
    let mut trace =
        simple_fibonacci::fibonacci_trace([Felt252::from(1), Felt252::from(1)], trace_length);

    // Use custom options with specified number of queries
    let mut proof_options = ProofOptions::default_test_options();
    proof_options.fri_number_of_queries = num_queries;

    let pub_inputs = FibonacciPublicInputs {
        a0: Felt252::one(),
        a1: Felt252::one(),
    };

    let air = FibonacciAIR::new(trace.num_rows(), &pub_inputs, &proof_options);

    let proof = Prover::prove(&air, &mut trace, &mut StoneProverTranscript::new(&[])).unwrap();

    (proof, air, pub_inputs)
}

fn bench_verifier(c: &mut Criterion) {
    let mut group = c.benchmark_group("STARK Verifier");
    group.sample_size(20);

    // Test with production-like query counts (31 queries = 80-bit security)
    for (trace_length, num_queries) in [(1024, 31), (4096, 31), (4096, 55)].iter() {
        let (proof, air, _pub_inputs) = generate_proof(*trace_length, *num_queries);

        group.bench_with_input(
            BenchmarkId::new(format!("trace={}", trace_length), num_queries),
            &(*trace_length, *num_queries),
            |b, _| {
                b.iter(|| Verifier::verify(&proof, &air, &mut StoneProverTranscript::new(&[])));
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_verifier);
criterion_main!(benches);
