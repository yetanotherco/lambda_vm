//! Host driver: prove an inner empty program on lambda-vm, then execute the
//! lambda-vm verifier inside SP1's executor, printing the cycle count.
//!
//! Set `TRACE_FILE=profiles/verifier.json` to capture a DWARF-attributed
//! profile (1 sample = 1 cycle). The output can be opened with
//! `samply load profiles/verifier.json`.

use std::path::PathBuf;

use sp1_sdk::blocking::{Prover, ProverClient};
use sp1_sdk::{SP1Stdin, include_elf};

const VERIFIER_ELF: sp1_sdk::Elf = include_elf!("verifier-program");

fn workspace_root() -> PathBuf {
    // CARGO_MANIFEST_DIR for this crate is `<root>/bench_vs/sp1/verifier/script`.
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(4)
        .expect("workspace root")
        .to_path_buf()
}

fn main() {
    sp1_sdk::utils::setup_logger();

    let root = workspace_root();
    let empty_elf_path = root
        .join("bench_vs/lambda/empty/target/riscv64im-lambda-vm-elf/release/empty-bench");
    assert!(
        empty_elf_path.exists(),
        "empty-bench ELF not found at {} — run `bash bench_vs/build_recursion_elfs.sh` first",
        empty_elf_path.display(),
    );
    let inner_elf = std::fs::read(&empty_elf_path).expect("read empty-bench");

    let options = stark::proof::options::ProofOptions {
        blowup_factor: 2,
        fri_number_of_queries: 1,
        coset_offset: 3,
        grinding_factor: 1,
    };

    println!("[sp1-verifier] proving inner (empty, blowup=2, 1 query) ...");
    let inner_proof = lambda_vm_prover::prove_with_options_and_inputs(
        &inner_elf,
        &[],
        &options,
        &lambda_vm_prover::MaxRowsConfig::default(),
    )
    .expect("inner prove should succeed");

    let blob = postcard::to_allocvec(&(&inner_proof, &inner_elf, &options))
        .expect("postcard encode failed");
    println!("[sp1-verifier] postcard blob: {} bytes", blob.len());

    let client = ProverClient::from_env();
    let mut stdin = SP1Stdin::new();
    stdin.write_vec(blob);

    println!("[sp1-verifier] executing verifier in SP1 ...");
    let (_, report) = client
        .execute(VERIFIER_ELF.clone(), stdin)
        .run()
        .expect("execute failed");

    let cycles = report.total_instruction_count();
    println!();
    println!("============================================================");
    println!("  SP1 EXECUTION SUMMARY — lambda-vm verifier inside SP1");
    println!("============================================================");
    println!("  Total cycles : {cycles}");
    println!();
    println!("  Compare against lambda-vm in-VM count (~40.5B for the same");
    println!("  proof). Both VMs target riscv64im, so word width is symmetric.");
    println!("  Main remaining asymmetry: lambda-vm's KeccakPermute precompile");
    println!("  is patched on its guests but SP1 does not patch `keccak` (only");
    println!("  `tiny-keccak`), so Keccak rounds run as software in SP1 here.");
    println!();
    println!("  If TRACE_FILE was set, the profile was written there.");
    println!("  Render with: samply load <trace>");
    println!("============================================================");
}
