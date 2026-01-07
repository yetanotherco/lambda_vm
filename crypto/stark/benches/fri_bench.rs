use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use stark::Felt252;
use stark::examples::simple_fibonacci::{self, FibonacciAIR, FibonacciPublicInputs};
use stark::proof::options::ProofOptions;
use stark::prover::{IsStarkProver, Prover};
use stark::traits::AIR;
use stark::transcript::StoneProverTranscript;
use stark::verifier::{IsStarkVerifier, Verifier};

fn fibonacci_prove_verify(trace_length: usize) {
    let mut trace =
        simple_fibonacci::fibonacci_trace([Felt252::from(1), Felt252::from(1)], trace_length);

    let proof_options = ProofOptions::default_test_options();

    let pub_inputs = FibonacciPublicInputs {
        a0: Felt252::one(),
        a1: Felt252::one(),
    };

    let air = FibonacciAIR::new(trace.num_rows(), &pub_inputs, &proof_options);

    let proof = Prover::prove(&air, &mut trace, &mut StoneProverTranscript::new(&[])).unwrap();

    assert!(Verifier::verify(
        &proof,
        &air,
        &mut StoneProverTranscript::new(&[]),
    ));
}

fn bench_fri_prover(c: &mut Criterion) {
    let mut group = c.benchmark_group("STARK Prove/Verify");
    group.sample_size(10);

    for trace_length in [256, 1024, 4096, 16384].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(trace_length),
            trace_length,
            |b, &length| {
                b.iter(|| fibonacci_prove_verify(length));
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_fri_prover);
criterion_main!(benches);
