// Profiling binary for STARK prover to generate flamegraph data
// It can run using `samply record cargo bench --bench profile_prover --features parallel`.
use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use math::field::element::FieldElement;
use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField;
use math::field::goldilocks::GoldilocksField;
use stark::examples::fibonacci_multi_column::{
    FibonacciMultiColumnAIR, compute_trace, create_public_inputs,
};
use stark::proof::options::ProofOptions;
use stark::prover::{IsStarkProver, Prover};

type F = GoldilocksField;
type E = Degree3GoldilocksExtensionField;
type FE = FieldElement<F>;

fn main() {
    // Use a representative workload for profiling
    let proof_options = ProofOptions {
        blowup_factor: 4,
        fri_number_of_queries: 100,
        coset_offset: 3,
        grinding_factor: 0,
        fri_final_poly_log_degree: 7,
    };

    let num_columns = 16;
    let trace_length = 1048576; // 2^20 = 1.048.576

    println!("Starting STARK prover profiling...");
    println!("Configuration:");
    println!("  - Number of Columns: {}", num_columns);
    println!("  - Trace length: {}", trace_length);
    println!("  - FRI queries: {}", proof_options.fri_number_of_queries);
    println!("  - Blowup factor: {}", proof_options.blowup_factor);

    #[cfg(feature = "parallel")]
    println!(
        "  - Parallel: ENABLED (rayon threads: {})",
        rayon::current_num_threads()
    );

    #[cfg(not(feature = "parallel"))]
    println!("  - Parallel: DISABLED");

    let initial_values: Vec<(FE, FE)> = (0..num_columns)
        .map(|i| (FE::from((i + 1) as u64), FE::from((i + 2) as u64)))
        .collect();

    println!("\nGenerating trace...");
    let mut trace = compute_trace::<F, E>(&initial_values, trace_length);
    let pub_inputs = create_public_inputs(initial_values);
    let air = FibonacciMultiColumnAIR::<F, E>::with_num_columns(&proof_options, num_columns);

    println!("Starting proof generation (this could take a while)...");
    let start = std::time::Instant::now();

    let _proof = Prover::<F, E, _>::prove(
        &air,
        &mut trace,
        &pub_inputs,
        &mut DefaultTranscript::<E>::new(&[]),
    )
    .expect("Failed to generate proof");

    let elapsed = start.elapsed();
    println!("\nProof generation completed in {:?}", elapsed);
    println!(
        "Profiling complete. Run with 'samply record' or 'cargo flamegraph' to generate flamegraph."
    );
}
