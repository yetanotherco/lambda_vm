use stark::Felt252;
use stark::examples::simple_fibonacci::{self, FibonacciAIR, FibonacciPublicInputs};
use stark::proof::options::ProofOptions;
use stark::prover::{IsStarkProver, Prover};
use stark::traits::AIR;
use stark::transcript::StoneProverTranscript;
use std::time::Instant;

fn main() {
    let trace_length: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(4096);

    let iterations: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);

    println!(
        "Running STARK prover {} times with trace_length={}",
        iterations, trace_length
    );

    let proof_options = ProofOptions::default_test_options();

    let pub_inputs = FibonacciPublicInputs {
        a0: Felt252::one(),
        a1: Felt252::one(),
    };

    // Warmup
    for _ in 0..2 {
        let mut trace =
            simple_fibonacci::fibonacci_trace([Felt252::from(1), Felt252::from(1)], trace_length);
        let air = FibonacciAIR::new(trace.num_rows(), &pub_inputs, &proof_options);
        let _ = Prover::prove(&air, &mut trace, &mut StoneProverTranscript::new(&[])).unwrap();
    }

    // Timed runs
    let start = Instant::now();
    for _ in 0..iterations {
        let mut trace =
            simple_fibonacci::fibonacci_trace([Felt252::from(1), Felt252::from(1)], trace_length);
        let air = FibonacciAIR::new(trace.num_rows(), &pub_inputs, &proof_options);
        let _ = Prover::prove(&air, &mut trace, &mut StoneProverTranscript::new(&[])).unwrap();
    }
    let elapsed = start.elapsed();

    let avg_ms = elapsed.as_millis() as f64 / iterations as f64;
    println!("Average proving time: {:.2} ms", avg_ms);
    println!("Total time for {} iterations: {:?}", iterations, elapsed);
}
