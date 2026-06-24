// Profiling binary for proving an ethrex block (realistic workload).
//
// Accurate per-step wall-clock timeline (build with the `instruments` feature):
//   LAMBDA_VM_TIMELINE_JSON=/tmp/t.json \
//   cargo bench -p lambda-vm-prover --bench profile_ethrex --features "parallel,instruments" -- ethrex_5_transfers
//
// Defaults: guest = executor/program_artifacts/rust/ethrex.elf,
//           fixture = executor/tests/ethrex_2_transfers.bin
// First positional arg overrides the fixture name (in executor/tests, no extension).

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
    println!("  - Guest ELF: {}", elf_path.display());
    println!("  - Fixture: {} ({} bytes)", fixture, private_inputs.len());
    #[cfg(feature = "parallel")]
    println!("  - rayon threads: {}", rayon::current_num_threads());

    let start = std::time::Instant::now();
    let _proof = lambda_vm_prover::prove_with_inputs(&elf_bytes, &private_inputs)
        .expect("Failed to generate proof");
    println!("\nProof generation completed in {:?}", start.elapsed());
}
