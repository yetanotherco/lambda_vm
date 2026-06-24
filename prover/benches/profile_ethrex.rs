// Profiling binary for proving an ethrex block (realistic workload).
//
// Run with instrumented phase breakdown:
//   cargo bench -p lambda-vm-prover --bench profile_ethrex --features "parallel,instruments"
// Flamegraph:
//   samply record -- cargo bench -p lambda-vm-prover --bench profile_ethrex --features parallel
//
// Defaults: guest = executor/program_artifacts/rust/ethrex.elf,
//           fixture = executor/tests/ethrex_2_transfers.bin
// Override the fixture name (in executor/tests, without extension) as the first arg:
//   ... --bench profile_ethrex --features parallel -- ethrex_simple_tx

use std::path::PathBuf;

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root")
        .to_path_buf()
}

fn main() {
    let fixture = std::env::args()
        .skip(1)
        .find(|a| !a.starts_with('-'))
        .unwrap_or_else(|| "ethrex_2_transfers".to_string());

    let root = workspace_root();
    let elf_path = root.join("executor/program_artifacts/rust/ethrex.elf");
    let fixture_path = root.join("executor/tests").join(format!("{fixture}.bin"));

    let elf_bytes = std::fs::read(&elf_path)
        .unwrap_or_else(|e| panic!("read guest ELF {}: {e}", elf_path.display()));
    let private_inputs = std::fs::read(&fixture_path)
        .unwrap_or_else(|e| panic!("read fixture {}: {e}", fixture_path.display()));

    println!("Starting ethrex prover profiling...");
    println!("Configuration:");
    println!("  - Guest ELF: {}", elf_path.display());
    println!("  - Fixture: {} ({} bytes)", fixture, private_inputs.len());

    #[cfg(feature = "parallel")]
    println!(
        "  - Parallel: ENABLED (rayon threads: {})",
        rayon::current_num_threads()
    );

    println!("\nGenerating proof (this will take a while)...");
    let start = std::time::Instant::now();

    let _proof = lambda_vm_prover::prove_with_inputs(&elf_bytes, &private_inputs)
        .expect("Failed to generate proof");

    println!("\nProof generation completed in {:?}", start.elapsed());
}
