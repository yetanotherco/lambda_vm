//! Realistic VM prover benchmark with all tables.
//!
//! Uses the public `prove` and `verify` API from lambda_vm_prover,
//! exercising the full proving pipeline with all 9 VM tables.
//!
//! Run with `cargo bench --bench vm_prover_benchmark`.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

use lambda_vm_prover::test_utils::asm_elf_bytes;
use stark::proof::options::ProofOptions;

/// Configuration for a benchmark case
struct BenchConfig {
    name: &'static str,
    elf_name: &'static str,
}

impl BenchConfig {
    const fn new(name: &'static str, elf_name: &'static str) -> Self {
        Self { name, elf_name }
    }
}

/// Benchmark configurations
const CONFIGS: &[BenchConfig] = &[
    BenchConfig::new("vm_32k", "bench_32k"), // 2^15 = 32768 rows.
];

// =============================================================================
// Benchmark functions
// =============================================================================

/// Benchmark prove (ELF execution + trace generation + proving)
fn bench_prove(c: &mut Criterion, config: &BenchConfig) {
    let elf_bytes = asm_elf_bytes(config.elf_name);
    let proof_options = ProofOptions::default_test_options();

    c.bench_with_input(
        BenchmarkId::new("vm_prover/prove", config.name),
        config,
        |b, _config| b.iter(|| lambda_vm_prover::prove(&elf_bytes, &proof_options).unwrap()),
    );
}

/// Benchmark verify only (proof pre-generated)
fn bench_verify(c: &mut Criterion, config: &BenchConfig) {
    let elf_bytes = asm_elf_bytes(config.elf_name);
    let proof_options = ProofOptions::default_test_options();
    let proof = lambda_vm_prover::prove(&elf_bytes, &proof_options).unwrap();

    c.bench_with_input(
        BenchmarkId::new("vm_prover/verify", config.name),
        &proof,
        |b, proof| {
            b.iter(|| lambda_vm_prover::verify(proof, &elf_bytes, &proof_options).unwrap())
        },
    );
}

/// Benchmark full pipeline (prove + verify)
fn bench_prove_and_verify(c: &mut Criterion, config: &BenchConfig) {
    let elf_bytes = asm_elf_bytes(config.elf_name);
    let proof_options = ProofOptions::default_test_options();

    c.bench_with_input(
        BenchmarkId::new("vm_prover/prove_and_verify", config.name),
        config,
        |b, _config| {
            b.iter(|| lambda_vm_prover::prove_and_verify(&elf_bytes, &proof_options).unwrap())
        },
    );
}

/// VM prover benchmarks
fn vm_prover_benchmarks(c: &mut Criterion) {
    for config in CONFIGS {
        bench_prove(c, config);
        bench_verify(c, config);
        bench_prove_and_verify(c, config);
    }
}

criterion_group! {
    name = vm_prover;
    config = Criterion::default()
        .sample_size(10)
        .measurement_time(std::time::Duration::from_secs(60));
    targets = vm_prover_benchmarks
}

criterion_main!(vm_prover);
