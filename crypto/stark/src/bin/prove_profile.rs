//! STARK prover profiling binary
//!
//! This binary is used to profile STARK proving and verification.
//! Run with various trace sizes to identify performance bottlenecks.
//!
//! Usage:
//!   cargo build --release -p stark
//!
//!   # Generate flamegraph
//!   cargo flamegraph --release -p stark --bin prove_profile -- --log-trace-size 14
//!
//!   # Use samply (better for macOS)
//!   samply record ./target/release/prove_profile --log-trace-size 14

use stark::{
    examples::simple_fibonacci::{fibonacci_trace, FibonacciAIR, FibonacciPublicInputs},
    proof::options::ProofOptions,
    prover::{IsStarkProver, Prover},
    traits::AIR,
    transcript::StoneProverTranscript,
    verifier::{IsStarkVerifier, Verifier},
};
use math::field::{
    element::FieldElement,
    fields::fft_friendly::stark_252_prime_field::Stark252PrimeField,
};
use std::{env, time::Instant};

type F = Stark252PrimeField;
type Felt = FieldElement<F>;

fn main() {
    let args: Vec<String> = env::args().collect();

    // Parse command line arguments
    let mut log_trace_size = 14; // Default: 2^14 = 16384 rows
    let mut iterations = 1;
    let mut verify = true;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--log-trace-size" | "-s" => {
                i += 1;
                if i < args.len() {
                    log_trace_size = args[i].parse().unwrap_or(14);
                }
            }
            "--iterations" | "-n" => {
                i += 1;
                if i < args.len() {
                    iterations = args[i].parse().unwrap_or(1);
                }
            }
            "--no-verify" => {
                verify = false;
            }
            "--help" | "-h" => {
                print_help();
                return;
            }
            _ => {
                eprintln!("Unknown argument: {}", args[i]);
                print_help();
                return;
            }
        }
        i += 1;
    }

    let trace_size = 1 << log_trace_size;

    println!("=== STARK Prover Profiling ===");
    println!("Trace size: 2^{} = {} rows", log_trace_size, trace_size);
    println!("Iterations: {}", iterations);
    println!("Verify: {}", verify);
    println!();

    // Run profiling
    for iter in 0..iterations {
        if iterations > 1 {
            println!("--- Iteration {} ---", iter + 1);
        }

        run_fibonacci_proof(trace_size, verify);

        println!();
    }
}

fn run_fibonacci_proof(trace_size: usize, do_verify: bool) {
    // Setup
    let setup_start = Instant::now();

    let pub_inputs = FibonacciPublicInputs {
        a0: Felt::one(),
        a1: Felt::one(),
    };

    let proof_options = ProofOptions::default_test_options();

    // Generate trace
    let trace_start = Instant::now();
    let mut trace = fibonacci_trace([Felt::one(), Felt::one()], trace_size);
    let trace_time = trace_start.elapsed();

    // Create AIR
    let air = FibonacciAIR::<F>::new(trace.num_rows(), &pub_inputs, &proof_options);

    let setup_time = setup_start.elapsed();
    println!("Setup time: {:?} (trace gen: {:?})", setup_time, trace_time);

    // Prove
    let prove_start = Instant::now();
    let proof = Prover::prove(
        &air,
        &mut trace,
        &mut StoneProverTranscript::new(&[]),
    ).expect("Proving failed");
    let prove_time = prove_start.elapsed();

    println!("Prove time: {:?}", prove_time);
    println!("FRI layers: {}", proof.fri_layers_merkle_roots.len());
    println!("Query count: {}", proof.query_list.len());

    // Verify
    if do_verify {
        let verify_start = Instant::now();
        let valid = Verifier::verify(
            &proof,
            &air,
            &mut StoneProverTranscript::new(&[]),
        );
        let verify_time = verify_start.elapsed();

        println!("Verify time: {:?}", verify_time);
        println!("Proof valid: {}", valid);
    }

    // Summary
    let total_time = setup_start.elapsed();
    println!("Total time: {:?}", total_time);
}

fn print_help() {
    println!("STARK Prover Profiling Tool");
    println!();
    println!("Usage: prove_profile [OPTIONS]");
    println!();
    println!("Options:");
    println!("  -s, --log-trace-size <N>  Log2 of trace size (default: 14, meaning 2^14 rows)");
    println!("  -n, --iterations <N>      Number of iterations (default: 1)");
    println!("      --no-verify           Skip verification");
    println!("  -h, --help                Print this help");
    println!();
    println!("Examples:");
    println!("  prove_profile --log-trace-size 16");
    println!("  cargo flamegraph --release -p stark --bin prove_profile -- -s 14");
    println!("  samply record ./target/release/prove_profile -s 14");
}
