use stark::Felt252;
use stark::examples::simple_fibonacci::{self, FibonacciAIR, FibonacciPublicInputs};
use stark::proof::options::ProofOptions;
use stark::prover::{IsStarkProver, Prover};
use stark::traits::AIR;
use stark::transcript::StoneProverTranscript;
use stark::verifier::{IsStarkVerifier, Verifier};
use std::time::Instant;

fn main() {
    let trace_length: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(4096);

    let num_queries: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(31);

    println!(
        "Generating proof: trace_length={}, num_queries={}",
        trace_length, num_queries
    );

    let mut trace =
        simple_fibonacci::fibonacci_trace([Felt252::from(1), Felt252::from(1)], trace_length);

    let mut proof_options = ProofOptions::default_test_options();
    proof_options.fri_number_of_queries = num_queries;

    let pub_inputs = FibonacciPublicInputs {
        a0: Felt252::one(),
        a1: Felt252::one(),
    };

    let air = FibonacciAIR::new(trace.num_rows(), &pub_inputs, &proof_options);
    let proof = Prover::prove(&air, &mut trace, &mut StoneProverTranscript::new(&[])).unwrap();

    println!("Proof generated. Running verifier {} times...", 100);

    // Warmup
    for _ in 0..5 {
        let _ = Verifier::verify(&proof, &air, &mut StoneProverTranscript::new(&[]));
    }

    // Timed runs
    let start = Instant::now();
    let iterations = 100;
    for _ in 0..iterations {
        assert!(Verifier::verify(
            &proof,
            &air,
            &mut StoneProverTranscript::new(&[])
        ));
    }
    let elapsed = start.elapsed();

    let avg_us = elapsed.as_micros() as f64 / iterations as f64;
    println!(
        "Average verification time: {:.2} µs ({:.2} ms)",
        avg_us,
        avg_us / 1000.0
    );
    println!("Total time for {} iterations: {:?}", iterations, elapsed);
}
