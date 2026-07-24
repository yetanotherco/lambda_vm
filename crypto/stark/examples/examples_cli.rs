//! Prove/verify CLI over the stark example AIRs, for cross-version
//! verification of the constraint system (see
//! `scripts/cross_verify_examples.sh`).
//!
//! Usage:
//!   examples_cli prove  <example-name> -o <proof.bin>
//!   examples_cli verify <example-name> <proof.bin>
//!
//! Proofs are bincode-serialized (this example's own format); `bin/cli`'s
//! VM-proof format is now rkyv, so this no longer mirrors it.
//! Trace sizes and public inputs mirror the existing stark tests
//! (`src/tests/air_tests.rs`, `src/tests/small_trace_tests.rs`,
//! `src/tests/bus_tests/completeness_tests.rs`) so a proof produced by one
//! version of the constraint system can be checked by another.
//!
//! Exit code 0 = success (prove written / verify accepted); nonzero = failure.

use std::path::PathBuf;
use std::process::ExitCode;

use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use math::field::{
    element::FieldElement, extensions_goldilocks::Degree3GoldilocksExtensionField,
    goldilocks::GoldilocksField,
};

use stark::examples::{
    dummy_air::{self, DummyAIR},
    fibonacci_2_cols_shifted::{self, Fibonacci2ColsShifted},
    fibonacci_2_columns::{self, Fibonacci2ColsAIR},
    fibonacci_multi_column::{self, FibonacciMultiColumnAIR, FibonacciMultiColumnPublicInputs},
    fibonacci_rap::{FibonacciRAP, FibonacciRAPPublicInputs, fibonacci_rap_trace},
    multi_table_lookup::{
        new_add_air_with_lookup, new_cpu_air_with_lookup, new_mul_air_with_lookup,
    },
    quadratic_air::{self, QuadraticAIR, QuadraticPublicInputs},
    read_only_memory::{ReadOnlyPublicInputs, ReadOnlyRAP, sort_rap_trace},
    read_only_memory_logup::{LogReadOnlyPublicInputs, LogReadOnlyRAP, read_only_logup_trace},
    simple_addition::{SimpleAdditionAIR, SimpleAdditionPublicInputs, simple_addition_trace},
    simple_fibonacci::{self, FibonacciAIR, FibonacciPublicInputs},
};
use stark::proof::options::ProofOptions;
use stark::proof::stark::{MultiProof, StarkProof};
use stark::prover::{IsStarkProver, Prover};
use stark::trace::TraceTable;
use stark::traits::AIR;
use stark::verifier::{IsStarkVerifier, Verifier};

type Gl = GoldilocksField;
type Gl3 = Degree3GoldilocksExtensionField;
type Felt = FieldElement<Gl>;

const EXAMPLES: &[&str] = &[
    "simple_fibonacci",
    "fibonacci_2_columns",
    "fibonacci_2_cols_shifted",
    "fibonacci_multi_column",
    "quadratic_air",
    "fibonacci_rap",
    "dummy_air",
    "simple_addition",
    "read_only_memory",
    "read_only_memory_logup",
    "multi_table_lookup",
];

fn ser<T: serde::Serialize>(proof: &T) -> Result<Vec<u8>, String> {
    bincode::serialize(proof).map_err(|e| format!("failed to serialize proof: {e}"))
}

fn de<T: serde::de::DeserializeOwned>(bytes: &[u8]) -> Result<T, String> {
    bincode::deserialize(bytes).map_err(|e| format!("failed to deserialize proof: {e}"))
}

// =============================================================================
// simple_fibonacci — mirrors air_tests::test_prove_fib
// =============================================================================

fn prove_simple_fibonacci() -> Result<Vec<u8>, String> {
    let mut trace = simple_fibonacci::fibonacci_trace([Felt::from(1), Felt::from(1)], 8);
    let pub_inputs = FibonacciPublicInputs {
        a0: Felt::one(),
        a1: Felt::one(),
    };
    let air = FibonacciAIR::<Gl>::new(&ProofOptions::default_test_options());
    let proof = Prover::prove(
        &air,
        &mut trace,
        &pub_inputs,
        &mut DefaultTranscript::<Gl>::new(&[]),
    )
    .map_err(|e| format!("prove failed: {e:?}"))?;
    ser(&proof)
}

fn verify_simple_fibonacci(bytes: &[u8]) -> Result<bool, String> {
    let proof: StarkProof<Gl, Gl, FibonacciPublicInputs<Gl>> = de(bytes)?;
    let air = FibonacciAIR::<Gl>::new(&ProofOptions::default_test_options());
    Ok(Verifier::verify(
        &proof,
        &air,
        &mut DefaultTranscript::<Gl>::new(&[]),
    ))
}

// =============================================================================
// fibonacci_2_columns — mirrors air_tests::test_prove_fib_2_cols
// =============================================================================

fn prove_fibonacci_2_columns() -> Result<Vec<u8>, String> {
    let mut trace = fibonacci_2_columns::compute_trace([Felt::from(1), Felt::from(1)], 16);
    let pub_inputs = FibonacciPublicInputs {
        a0: Felt::one(),
        a1: Felt::one(),
    };
    let air = Fibonacci2ColsAIR::<Gl>::new(&ProofOptions::default_test_options());
    let proof = Prover::prove(
        &air,
        &mut trace,
        &pub_inputs,
        &mut DefaultTranscript::<Gl>::new(&[]),
    )
    .map_err(|e| format!("prove failed: {e:?}"))?;
    ser(&proof)
}

fn verify_fibonacci_2_columns(bytes: &[u8]) -> Result<bool, String> {
    let proof: StarkProof<Gl, Gl, FibonacciPublicInputs<Gl>> = de(bytes)?;
    let air = Fibonacci2ColsAIR::<Gl>::new(&ProofOptions::default_test_options());
    Ok(Verifier::verify(
        &proof,
        &air,
        &mut DefaultTranscript::<Gl>::new(&[]),
    ))
}

// =============================================================================
// fibonacci_2_cols_shifted — mirrors air_tests::test_prove_fib_2_cols_shifted
// =============================================================================

fn prove_fibonacci_2_cols_shifted() -> Result<Vec<u8>, String> {
    let mut trace = fibonacci_2_cols_shifted::compute_trace(FieldElement::one(), 16);
    let claimed_index = 14;
    let claimed_value = trace.main_table.get_row(claimed_index)[0];
    let pub_inputs = fibonacci_2_cols_shifted::PublicInputs {
        claimed_value,
        claimed_index,
    };
    let air = Fibonacci2ColsShifted::<Gl>::new(&ProofOptions::default_test_options());
    let proof = Prover::prove(
        &air,
        &mut trace,
        &pub_inputs,
        &mut DefaultTranscript::<Gl>::new(&[]),
    )
    .map_err(|e| format!("prove failed: {e:?}"))?;
    ser(&proof)
}

fn verify_fibonacci_2_cols_shifted(bytes: &[u8]) -> Result<bool, String> {
    let proof: StarkProof<Gl, Gl, fibonacci_2_cols_shifted::PublicInputs<Gl>> = de(bytes)?;
    let air = Fibonacci2ColsShifted::<Gl>::new(&ProofOptions::default_test_options());
    Ok(Verifier::verify(
        &proof,
        &air,
        &mut DefaultTranscript::<Gl>::new(&[]),
    ))
}

// =============================================================================
// fibonacci_multi_column — mirrors air_tests::test_multi_column_fibonacci_2_cols
// =============================================================================

fn multi_column_initial_values() -> Vec<(Felt, Felt)> {
    (0..2u64)
        .map(|i| (Felt::from(i + 1), Felt::from(i + 2)))
        .collect()
}

fn prove_fibonacci_multi_column() -> Result<Vec<u8>, String> {
    let initial_values = multi_column_initial_values();
    let mut trace = fibonacci_multi_column::compute_trace::<Gl, Gl3>(&initial_values, 16);
    let pub_inputs = fibonacci_multi_column::create_public_inputs(initial_values);
    let air = FibonacciMultiColumnAIR::<Gl, Gl3>::with_num_columns(
        &ProofOptions::default_test_options(),
        2,
    );
    let proof = Prover::<Gl, Gl3, _>::prove(
        &air,
        &mut trace,
        &pub_inputs,
        &mut DefaultTranscript::<Gl3>::new(&[]),
    )
    .map_err(|e| format!("prove failed: {e:?}"))?;
    ser(&proof)
}

fn verify_fibonacci_multi_column(bytes: &[u8]) -> Result<bool, String> {
    let proof: StarkProof<Gl, Gl3, FibonacciMultiColumnPublicInputs<Gl>> = de(bytes)?;
    let air = FibonacciMultiColumnAIR::<Gl, Gl3>::with_num_columns(
        &ProofOptions::default_test_options(),
        2,
    );
    Ok(Verifier::<Gl, Gl3, _>::verify(
        &proof,
        &air,
        &mut DefaultTranscript::<Gl3>::new(&[]),
    ))
}

// =============================================================================
// quadratic_air — mirrors air_tests::test_prove_quadratic
// =============================================================================

fn prove_quadratic_air() -> Result<Vec<u8>, String> {
    let mut trace = quadratic_air::quadratic_trace(Felt::from(3), 32);
    let pub_inputs = QuadraticPublicInputs { a0: Felt::from(3) };
    let air = QuadraticAIR::<Gl>::new(&ProofOptions::default_test_options());
    let proof = Prover::prove(
        &air,
        &mut trace,
        &pub_inputs,
        &mut DefaultTranscript::<Gl>::new(&[]),
    )
    .map_err(|e| format!("prove failed: {e:?}"))?;
    ser(&proof)
}

fn verify_quadratic_air(bytes: &[u8]) -> Result<bool, String> {
    let proof: StarkProof<Gl, Gl, QuadraticPublicInputs<Gl>> = de(bytes)?;
    let air = QuadraticAIR::<Gl>::new(&ProofOptions::default_test_options());
    Ok(Verifier::verify(
        &proof,
        &air,
        &mut DefaultTranscript::<Gl>::new(&[]),
    ))
}

// =============================================================================
// fibonacci_rap — mirrors air_tests::test_prove_rap_fib
// =============================================================================

fn prove_fibonacci_rap() -> Result<Vec<u8>, String> {
    let steps = 16;
    let mut trace = fibonacci_rap_trace([Felt::from(1), Felt::from(1)], steps);
    let pub_inputs = FibonacciRAPPublicInputs {
        steps,
        a0: Felt::one(),
        a1: Felt::one(),
    };
    let air = FibonacciRAP::<Gl>::new(&ProofOptions::default_test_options());
    let proof = Prover::prove(
        &air,
        &mut trace,
        &pub_inputs,
        &mut DefaultTranscript::<Gl>::new(&[]),
    )
    .map_err(|e| format!("prove failed: {e:?}"))?;
    ser(&proof)
}

fn verify_fibonacci_rap(bytes: &[u8]) -> Result<bool, String> {
    let proof: StarkProof<Gl, Gl, FibonacciRAPPublicInputs<Gl>> = de(bytes)?;
    let air = FibonacciRAP::<Gl>::new(&ProofOptions::default_test_options());
    Ok(Verifier::verify(
        &proof,
        &air,
        &mut DefaultTranscript::<Gl>::new(&[]),
    ))
}

// =============================================================================
// dummy_air — mirrors air_tests::test_prove_dummy
// =============================================================================

fn prove_dummy_air() -> Result<Vec<u8>, String> {
    let mut trace = dummy_air::dummy_trace(16);
    let air = DummyAIR::new(&ProofOptions::default_test_options());
    let proof = Prover::prove(
        &air,
        &mut trace,
        &(),
        &mut DefaultTranscript::<Gl>::new(&[]),
    )
    .map_err(|e| format!("prove failed: {e:?}"))?;
    ser(&proof)
}

fn verify_dummy_air(bytes: &[u8]) -> Result<bool, String> {
    let proof: StarkProof<Gl, Gl, ()> = de(bytes)?;
    let air = DummyAIR::new(&ProofOptions::default_test_options());
    Ok(Verifier::verify(
        &proof,
        &air,
        &mut DefaultTranscript::<Gl>::new(&[]),
    ))
}

// =============================================================================
// simple_addition — mirrors small_trace_tests::test_prove_verify_single_row
// =============================================================================

fn prove_simple_addition() -> Result<Vec<u8>, String> {
    let mut trace = simple_addition_trace::<Gl>(1);
    let pub_inputs = SimpleAdditionPublicInputs {
        a: Felt::from(1u64),
        b: Felt::from(2u64),
    };
    let air = SimpleAdditionAIR::<Gl>::new(&ProofOptions::default_test_options());
    let proof = Prover::prove(
        &air,
        &mut trace,
        &pub_inputs,
        &mut DefaultTranscript::<Gl>::new(&[]),
    )
    .map_err(|e| format!("prove failed: {e:?}"))?;
    ser(&proof)
}

fn verify_simple_addition(bytes: &[u8]) -> Result<bool, String> {
    let proof: StarkProof<Gl, Gl, SimpleAdditionPublicInputs<Gl>> = de(bytes)?;
    let air = SimpleAdditionAIR::<Gl>::new(&ProofOptions::default_test_options());
    Ok(Verifier::verify(
        &proof,
        &air,
        &mut DefaultTranscript::<Gl>::new(&[]),
    ))
}

// =============================================================================
// read_only_memory — mirrors air_tests::test_prove_read_only_memory
// =============================================================================

fn read_only_memory_columns() -> (Vec<Felt>, Vec<Felt>) {
    let address_col = vec![
        Felt::from(3), // a0
        Felt::from(2), // a1
        Felt::from(2), // a2
        Felt::from(3), // a3
        Felt::from(4), // a4
        Felt::from(5), // a5
        Felt::from(1), // a6
        Felt::from(3), // a7
    ];
    let value_col = vec![
        Felt::from(10), // v0
        Felt::from(5),  // v1
        Felt::from(5),  // v2
        Felt::from(10), // v3
        Felt::from(25), // v4
        Felt::from(25), // v5
        Felt::from(7),  // v6
        Felt::from(10), // v7
    ];
    (address_col, value_col)
}

fn prove_read_only_memory() -> Result<Vec<u8>, String> {
    let (address_col, value_col) = read_only_memory_columns();
    let pub_inputs = ReadOnlyPublicInputs {
        a0: Felt::from(3),
        v0: Felt::from(10),
        a_sorted0: Felt::from(1), // a6
        v_sorted0: Felt::from(7), // v6
    };
    let mut trace = sort_rap_trace(address_col, value_col);
    let air = ReadOnlyRAP::<Gl>::new(&ProofOptions::default_test_options());
    let proof = Prover::prove(
        &air,
        &mut trace,
        &pub_inputs,
        &mut DefaultTranscript::<Gl>::new(&[]),
    )
    .map_err(|e| format!("prove failed: {e:?}"))?;
    ser(&proof)
}

fn verify_read_only_memory(bytes: &[u8]) -> Result<bool, String> {
    let proof: StarkProof<Gl, Gl, ReadOnlyPublicInputs<Gl>> = de(bytes)?;
    let air = ReadOnlyRAP::<Gl>::new(&ProofOptions::default_test_options());
    Ok(Verifier::verify(
        &proof,
        &air,
        &mut DefaultTranscript::<Gl>::new(&[]),
    ))
}

// =============================================================================
// read_only_memory_logup — mirrors air_tests::test_prove_log_read_only_memory
// =============================================================================

fn read_only_memory_logup_columns() -> (Vec<Felt>, Vec<Felt>) {
    let address_col = vec![
        Felt::from(3), // a0
        Felt::from(2), // a1
        Felt::from(2), // a2
        Felt::from(3), // a3
        Felt::from(4), // a4
        Felt::from(5), // a5
        Felt::from(1), // a6
        Felt::from(3), // a7
    ];
    let value_col = vec![
        Felt::from(30), // v0
        Felt::from(20), // v1
        Felt::from(20), // v2
        Felt::from(30), // v3
        Felt::from(40), // v4
        Felt::from(50), // v5
        Felt::from(10), // v6
        Felt::from(30), // v7
    ];
    (address_col, value_col)
}

fn prove_read_only_memory_logup() -> Result<Vec<u8>, String> {
    let (address_col, value_col) = read_only_memory_logup_columns();
    let pub_inputs = LogReadOnlyPublicInputs {
        a0: Felt::from(3),
        v0: Felt::from(30),
        a_sorted_0: Felt::from(1),
        v_sorted_0: Felt::from(10),
        m0: Felt::from(1),
    };
    let mut trace = read_only_logup_trace(address_col, value_col);
    let air = LogReadOnlyRAP::<Gl, Gl3>::new(&ProofOptions::default_test_options());
    let proof = Prover::prove(
        &air,
        &mut trace,
        &pub_inputs,
        &mut DefaultTranscript::<Gl3>::new(&[]),
    )
    .map_err(|e| format!("prove failed: {e:?}"))?;
    ser(&proof)
}

fn verify_read_only_memory_logup(bytes: &[u8]) -> Result<bool, String> {
    let proof: StarkProof<Gl, Gl3, LogReadOnlyPublicInputs<Gl>> = de(bytes)?;
    let air = LogReadOnlyRAP::<Gl, Gl3>::new(&ProofOptions::default_test_options());
    Ok(Verifier::verify(
        &proof,
        &air,
        &mut DefaultTranscript::<Gl3>::new(&[]),
    ))
}

// =============================================================================
// multi_table_lookup — mirrors bus_tests::completeness_tests::test_multi_table_proof
// =============================================================================

fn multi_table_traces() -> (
    TraceTable<Gl, Gl3>,
    TraceTable<Gl, Gl3>,
    TraceTable<Gl, Gl3>,
) {
    // CPU Trace (8 rows): dispatches operations to ADD and MUL tables
    let add_column = vec![
        Felt::one(),
        Felt::zero(),
        Felt::one(),
        Felt::zero(),
        Felt::one(),
        Felt::one(),
        Felt::zero(),
        Felt::zero(),
    ];
    let mul_column = vec![
        Felt::zero(),
        Felt::one(),
        Felt::zero(),
        Felt::one(),
        Felt::zero(),
        Felt::zero(),
        Felt::one(),
        Felt::one(),
    ];
    let a_column = vec![
        Felt::from(1),
        Felt::from(2),
        Felt::from(3),
        Felt::from(4),
        Felt::from(5),
        Felt::from(6),
        Felt::from(7),
        Felt::from(8),
    ];
    let b_column = vec![
        Felt::from(10),
        Felt::from(20),
        Felt::from(30),
        Felt::from(40),
        Felt::from(50),
        Felt::from(60),
        Felt::from(70),
        Felt::from(80),
    ];
    let c_column = vec![
        Felt::from(11),  // 1 + 10
        Felt::from(40),  // 2 * 20
        Felt::from(33),  // 3 + 30
        Felt::from(160), // 4 * 40
        Felt::from(55),  // 5 + 50
        Felt::from(66),  // 6 + 60
        Felt::from(490), // 7 * 70
        Felt::from(640), // 8 * 80
    ];
    let cpu_trace = TraceTable::from_columns_main(
        vec![add_column, mul_column, a_column, b_column, c_column],
        1,
    );

    // ADD Trace (4 rows): receives addition operations
    let add_trace = TraceTable::from_columns_main(
        vec![
            vec![Felt::from(1), Felt::from(3), Felt::from(5), Felt::from(6)],
            vec![
                Felt::from(10),
                Felt::from(30),
                Felt::from(50),
                Felt::from(60),
            ],
            vec![
                Felt::from(11),
                Felt::from(33),
                Felt::from(55),
                Felt::from(66),
            ],
            vec![Felt::one(), Felt::one(), Felt::one(), Felt::one()],
        ],
        1,
    );

    // MUL Trace (4 rows): receives multiplication operations
    let mul_trace = TraceTable::from_columns_main(
        vec![
            vec![Felt::from(2), Felt::from(4), Felt::from(7), Felt::from(8)],
            vec![
                Felt::from(20),
                Felt::from(40),
                Felt::from(70),
                Felt::from(80),
            ],
            vec![
                Felt::from(40),
                Felt::from(160),
                Felt::from(490),
                Felt::from(640),
            ],
            vec![Felt::one(), Felt::one(), Felt::one(), Felt::one()],
        ],
        1,
    );

    (cpu_trace, add_trace, mul_trace)
}

fn prove_multi_table_lookup() -> Result<Vec<u8>, String> {
    let (mut cpu_trace, mut add_trace, mut mul_trace) = multi_table_traces();
    let proof_options = ProofOptions::default_test_options();
    let cpu_air = new_cpu_air_with_lookup(&proof_options);
    let add_air = new_add_air_with_lookup(&proof_options);
    let mul_air = new_mul_air_with_lookup(&proof_options);

    let air_trace_pairs: Vec<(
        &dyn AIR<Field = Gl, FieldExtension = Gl3, PublicInputs = ()>,
        _,
        _,
    )> = vec![
        (&cpu_air, &mut cpu_trace, &()),
        (&add_air, &mut add_trace, &()),
        (&mul_air, &mut mul_trace, &()),
    ];

    let multi_proof = Prover::<Gl, Gl3, ()>::multi_prove(
        air_trace_pairs,
        &mut DefaultTranscript::<Gl3>::new(&[]),
        #[cfg(feature = "disk-spill")]
        stark::storage_mode::StorageMode::Ram,
    )
    .map_err(|e| format!("prove failed: {e:?}"))?;
    ser(&multi_proof)
}

fn verify_multi_table_lookup(bytes: &[u8]) -> Result<bool, String> {
    let multi_proof: MultiProof<Gl, Gl3, ()> = de(bytes)?;
    let proof_options = ProofOptions::default_test_options();
    let cpu_air = new_cpu_air_with_lookup(&proof_options);
    let add_air = new_add_air_with_lookup(&proof_options);
    let mul_air = new_mul_air_with_lookup(&proof_options);
    let airs: Vec<&dyn AIR<Field = Gl, FieldExtension = Gl3, PublicInputs = ()>> =
        vec![&cpu_air, &add_air, &mul_air];
    Ok(Verifier::multi_verify(
        &airs,
        &multi_proof,
        &mut DefaultTranscript::<Gl3>::new(&[]),
        &FieldElement::zero(),
    ))
}

// =============================================================================
// Dispatch + main
// =============================================================================

fn prove_example(name: &str) -> Result<Vec<u8>, String> {
    match name {
        "simple_fibonacci" => prove_simple_fibonacci(),
        "fibonacci_2_columns" => prove_fibonacci_2_columns(),
        "fibonacci_2_cols_shifted" => prove_fibonacci_2_cols_shifted(),
        "fibonacci_multi_column" => prove_fibonacci_multi_column(),
        "quadratic_air" => prove_quadratic_air(),
        "fibonacci_rap" => prove_fibonacci_rap(),
        "dummy_air" => prove_dummy_air(),
        "simple_addition" => prove_simple_addition(),
        "read_only_memory" => prove_read_only_memory(),
        "read_only_memory_logup" => prove_read_only_memory_logup(),
        "multi_table_lookup" => prove_multi_table_lookup(),
        _ => Err(format!(
            "unknown example '{name}'; available: {}",
            EXAMPLES.join(", ")
        )),
    }
}

fn verify_example(name: &str, bytes: &[u8]) -> Result<bool, String> {
    match name {
        "simple_fibonacci" => verify_simple_fibonacci(bytes),
        "fibonacci_2_columns" => verify_fibonacci_2_columns(bytes),
        "fibonacci_2_cols_shifted" => verify_fibonacci_2_cols_shifted(bytes),
        "fibonacci_multi_column" => verify_fibonacci_multi_column(bytes),
        "quadratic_air" => verify_quadratic_air(bytes),
        "fibonacci_rap" => verify_fibonacci_rap(bytes),
        "dummy_air" => verify_dummy_air(bytes),
        "simple_addition" => verify_simple_addition(bytes),
        "read_only_memory" => verify_read_only_memory(bytes),
        "read_only_memory_logup" => verify_read_only_memory_logup(bytes),
        "multi_table_lookup" => verify_multi_table_lookup(bytes),
        _ => Err(format!(
            "unknown example '{name}'; available: {}",
            EXAMPLES.join(", ")
        )),
    }
}

fn usage() -> ExitCode {
    eprintln!("Usage:");
    eprintln!("  examples_cli prove  <example-name> -o <proof.bin>");
    eprintln!("  examples_cli verify <example-name> <proof.bin>");
    eprintln!("Examples: {}", EXAMPLES.join(", "));
    ExitCode::FAILURE
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("prove") => {
            let (Some(name), Some(flag), Some(out)) = (args.get(2), args.get(3), args.get(4))
            else {
                return usage();
            };
            if flag != "-o" {
                return usage();
            }
            let out = PathBuf::from(out);
            match prove_example(name) {
                Ok(bytes) => {
                    if let Err(e) = std::fs::write(&out, &bytes) {
                        eprintln!("failed to write proof to {out:?}: {e}");
                        return ExitCode::FAILURE;
                    }
                    eprintln!(
                        "proof for '{name}' written to {out:?} ({} bytes)",
                        bytes.len()
                    );
                    ExitCode::SUCCESS
                }
                Err(e) => {
                    eprintln!("{e}");
                    ExitCode::FAILURE
                }
            }
        }
        Some("verify") => {
            let (Some(name), Some(path)) = (args.get(2), args.get(3)) else {
                return usage();
            };
            let bytes = match std::fs::read(path) {
                Ok(b) => b,
                Err(e) => {
                    eprintln!("failed to read proof file {path}: {e}");
                    return ExitCode::FAILURE;
                }
            };
            match verify_example(name, &bytes) {
                Ok(true) => {
                    eprintln!("verification succeeded for '{name}'");
                    ExitCode::SUCCESS
                }
                Ok(false) => {
                    eprintln!("verification FAILED for '{name}'");
                    ExitCode::FAILURE
                }
                Err(e) => {
                    eprintln!("{e}");
                    ExitCode::FAILURE
                }
            }
        }
        _ => usage(),
    }
}
