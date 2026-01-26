//! Realistic VM prover benchmark with lookups.
//!
//! This benchmark exercises the full proving pipeline with real VM tables:
//! - CPU table (2^15 rows = 32768 instructions)
//! - Bitwise table (receives AND/OR/XOR/MSB lookups)
//! - LT table (receives less-than comparisons)
//!
//! Uses an actual ELF program that runs through the executor, generating
//! real execution traces with lookups.
//!
//! Run with `cargo bench --bench vm_prover_benchmark`.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use stark::proof::options::ProofOptions;
use stark::proof::stark::MultiProof;
use stark::prover::{IsStarkProver, Prover};
use stark::trace::TraceTable;
use stark::traits::AIR;
use stark::verifier::{IsStarkVerifier, Verifier};

use prover::tables::lt::generate_lt_trace;
use prover::tables::trace_builder::Traces;

// Import shared utilities
use prover::test_utils::{
    E, F, VmAir, collect_bitwise_lookups_from_logs, collect_bitwise_lookups_from_lt,
    collect_lt_lookups_from_logs, create_bitwise_air, create_cpu_air, create_lt_air,
    generate_minimal_bitwise_trace, run_asm_elf,
};

/// Configuration for a benchmark case
struct BenchConfig {
    name: &'static str,
    elf_name: &'static str,
    expected_steps: usize,
}

impl BenchConfig {
    const fn new(name: &'static str, elf_name: &'static str, expected_steps: usize) -> Self {
        Self {
            name,
            elf_name,
            expected_steps,
        }
    }
}

/// Benchmark configurations
const CONFIGS: &[BenchConfig] = &[
    BenchConfig::new("vm_32k", "bench_32k", 32768), // 2^15 = 32768 rows.
];

/// Creates proof options suitable for benchmarking
fn benchmark_proof_options() -> ProofOptions {
    ProofOptions {
        blowup_factor: 4,
        fri_number_of_queries: 30,
        coset_offset: 3,
        grinding_factor: 0,
    }
}

// Lookup collection and AIR creation functions moved to prover::test_utils module

fn create_airs(proof_options: &ProofOptions) -> (VmAir, VmAir, VmAir) {
    (
        create_cpu_air(proof_options),
        create_bitwise_air(proof_options),
        create_lt_air(proof_options),
    )
}

// =============================================================================
// Helper functions for benchmarking
// =============================================================================

/// Execute ELF and generate traces (ELF execution + trace generation)
fn execute_program(
    elf_name: &str,
    expected_steps: usize,
) -> (TraceTable<F, E>, TraceTable<F, E>, TraceTable<F, E>) {
    // Run the ELF program
    let (logs, instructions) = run_asm_elf(elf_name);
    assert_eq!(
        logs.len(),
        expected_steps,
        "{}.elf should have {} steps, got {}",
        elf_name,
        expected_steps,
        logs.len()
    );

    // Generate CPU trace using Traces::from_logs
    let traces = Traces::from_logs(&logs, instructions.clone()).unwrap();
    let cpu_trace = traces.cpu;

    // Collect LT operations and generate LT trace
    let lt_ops = collect_lt_lookups_from_logs(&logs, &instructions);
    let lt_trace = generate_lt_trace(&lt_ops);

    // Collect all bitwise lookups (from CPU + from LT table)
    let mut bitwise_lookups = collect_bitwise_lookups_from_logs(&logs, &instructions);
    let lt_bitwise_lookups = collect_bitwise_lookups_from_lt(&lt_ops);
    bitwise_lookups.extend(lt_bitwise_lookups);

    // Generate minimal bitwise trace for speed
    let bitwise_trace = generate_minimal_bitwise_trace(&bitwise_lookups);

    (cpu_trace, bitwise_trace, lt_trace)
}

/// Run multi_prove on the given traces and AIRs
fn prove(
    cpu_air: &VmAir,
    bitwise_air: &VmAir,
    lt_air: &VmAir,
    cpu_trace: &mut TraceTable<F, E>,
    bitwise_trace: &mut TraceTable<F, E>,
    lt_trace: &mut TraceTable<F, E>,
) -> MultiProof<F, E, ()> {
    let air_trace_pairs: Vec<(
        &dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>,
        _,
        _,
    )> = vec![
        (cpu_air, cpu_trace, &()),
        (bitwise_air, bitwise_trace, &()),
        (lt_air, lt_trace, &()),
    ];

    Prover::multi_prove(air_trace_pairs, &mut DefaultTranscript::<E>::new(&[])).unwrap()
}

/// Run multi_verify on the given proof and AIRs
fn verify(
    cpu_air: &VmAir,
    bitwise_air: &VmAir,
    lt_air: &VmAir,
    proof: &MultiProof<F, E, ()>,
) -> bool {
    let airs: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>> =
        vec![cpu_air, bitwise_air, lt_air];
    Verifier::multi_verify(&airs, proof, &mut DefaultTranscript::<E>::new(&[]))
}

// =============================================================================
// Benchmark functions
// =============================================================================

/// Benchmark execute + prove (ELF execution + trace generation + proving)
///
/// **What's measured (timed):**
/// - ELF execution through the executor
/// - Trace generation for CPU, Bitwise, and LT tables
/// - Multi-table STARK proving (commit, FRI, query phases)
///
/// **What's excluded from timing (setup):**
/// - AIR (Algebraic Intermediate Representation) creation
/// - Proof options configuration
///
/// **What the numbers represent:**
/// Time to go from ELF binary to complete STARK proof, including all table
/// generation and cryptographic operations. This represents the prover's
/// end-to-end latency for a given program size.
fn bench_execute_and_prove(c: &mut Criterion, config: &BenchConfig) {
    let proof_options = benchmark_proof_options();

    c.bench_with_input(
        BenchmarkId::new("vm_prover/execute_and_prove", config.name),
        config,
        |b, config| {
            b.iter_with_setup(
                || create_airs(&proof_options),
                |(cpu_air, bitwise_air, lt_air)| {
                    let (mut cpu_trace, mut bitwise_trace, mut lt_trace) =
                        execute_program(config.elf_name, config.expected_steps);
                    prove(
                        &cpu_air,
                        &bitwise_air,
                        &lt_air,
                        &mut cpu_trace,
                        &mut bitwise_trace,
                        &mut lt_trace,
                    )
                },
            )
        },
    );
}

/// Benchmark prove only (just the multi_prove function, traces pre-generated)
///
/// **What's measured (timed):**
/// - Multi-table STARK proving only (commit phase, FRI, query generation)
/// - Auxiliary trace construction (bus argument columns)
/// - Merkle tree commitments and FRI folding
///
/// **What's excluded from timing (setup):**
/// - ELF execution (done once before benchmark loop)
/// - Trace generation (done once before benchmark loop)
/// - AIR creation
/// - Trace cloning (to provide fresh traces for each iteration)
///
/// **What the numbers represent:**
/// Pure proving time from execution traces to STARK proof. This isolates the
/// cryptographic proving cost from program execution and trace generation.
/// Useful for understanding prover performance scaling with table sizes.
fn bench_prove(c: &mut Criterion, config: &BenchConfig) {
    let proof_options = benchmark_proof_options();

    // Pre-generate traces once
    let (cpu_trace, bitwise_trace, lt_trace) =
        execute_program(config.elf_name, config.expected_steps);

    println!(
        "{}: CPU {} rows, Bitwise {} rows, LT {} rows",
        config.name,
        cpu_trace.main_table.height,
        bitwise_trace.main_table.height,
        lt_trace.main_table.height,
    );

    c.bench_with_input(
        BenchmarkId::new("vm_prover/prove", config.name),
        config,
        |b, _config| {
            b.iter_with_setup(
                || {
                    let airs = create_airs(&proof_options);
                    // Clone traces for each iteration
                    (
                        airs,
                        cpu_trace.clone(),
                        bitwise_trace.clone(),
                        lt_trace.clone(),
                    )
                },
                |(
                    (cpu_air, bitwise_air, lt_air),
                    mut cpu_trace,
                    mut bitwise_trace,
                    mut lt_trace,
                )| {
                    prove(
                        &cpu_air,
                        &bitwise_air,
                        &lt_air,
                        &mut cpu_trace,
                        &mut bitwise_trace,
                        &mut lt_trace,
                    )
                },
            )
        },
    );
}

/// Benchmark verify only (proof pre-generated)
///
/// **What's measured (timed):**
/// - Multi-table STARK verification
/// - Constraint evaluation at queried positions
/// - FRI verification (verifying low-degree property)
/// - Merkle proof verification
///
/// **What's excluded from timing (setup):**
/// - ELF execution
/// - Trace generation
/// - Proof generation
/// - AIR creation
///
/// **What the numbers represent:**
/// Pure verification time for a STARK proof. This is the time a verifier
/// (who doesn't have the execution trace) needs to verify correctness.
/// Should be much faster than proving and scale logarithmically with
/// trace size.
fn bench_verify(c: &mut Criterion, config: &BenchConfig) {
    let proof_options = benchmark_proof_options();

    // Pre-generate traces and proof
    let (mut cpu_trace, mut bitwise_trace, mut lt_trace) =
        execute_program(config.elf_name, config.expected_steps);

    let (cpu_air, bitwise_air, lt_air) = create_airs(&proof_options);

    let proof = prove(
        &cpu_air,
        &bitwise_air,
        &lt_air,
        &mut cpu_trace,
        &mut bitwise_trace,
        &mut lt_trace,
    );

    c.bench_with_input(
        BenchmarkId::new("vm_prover/verify", config.name),
        &(proof, cpu_air, bitwise_air, lt_air),
        |b, (proof, cpu_air, bitwise_air, lt_air)| {
            b.iter(|| verify(cpu_air, bitwise_air, lt_air, proof))
        },
    );
}

/// Benchmark execute + prove + verify (full pipeline)
///
/// **What's measured (timed):**
/// - Complete end-to-end pipeline:
///   1. ELF execution through the executor
///   2. Trace generation for all tables (CPU, Bitwise, LT)
///   3. STARK proof generation (commit, FRI, queries)
///   4. STARK proof verification
///
/// **What's excluded from timing (setup):**
/// - AIR creation
/// - Proof options configuration
///
/// **What the numbers represent:**
/// Total time from ELF binary to verified proof, representing the complete
/// zkVM proving system latency. This is the most realistic benchmark for
/// end-to-end performance, including both prover and verifier costs.
fn bench_execute_prove_and_verify(c: &mut Criterion, config: &BenchConfig) {
    let proof_options = benchmark_proof_options();

    c.bench_with_input(
        BenchmarkId::new("vm_prover/execute_prove_and_verify", config.name),
        config,
        |b, config| {
            b.iter_with_setup(
                || create_airs(&proof_options),
                |(cpu_air, bitwise_air, lt_air)| {
                    // Execute
                    let (mut cpu_trace, mut bitwise_trace, mut lt_trace) =
                        execute_program(config.elf_name, config.expected_steps);

                    // Prove
                    let proof = prove(
                        &cpu_air,
                        &bitwise_air,
                        &lt_air,
                        &mut cpu_trace,
                        &mut bitwise_trace,
                        &mut lt_trace,
                    );

                    // Verify
                    verify(&cpu_air, &bitwise_air, &lt_air, &proof)
                },
            )
        },
    );
}

/// VM prover benchmarks
fn vm_prover_benchmarks(c: &mut Criterion) {
    for config in CONFIGS {
        bench_execute_and_prove(c, config);
        bench_prove(c, config);
        bench_verify(c, config);
        bench_execute_prove_and_verify(c, config);
    }
}

criterion_group! {
    name = vm_prover;
    // Use 10 samples instead of Criterion's default 100 because each benchmark
    // iteration takes 5+ seconds. This provides sufficient statistical confidence
    // while keeping total benchmark time reasonable (some minutes vs hours).
    config = Criterion::default()
        .sample_size(10)
        .measurement_time(std::time::Duration::from_secs(60));
    targets = vm_prover_benchmarks
}

criterion_main!(vm_prover);
