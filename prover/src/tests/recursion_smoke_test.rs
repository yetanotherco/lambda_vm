//! End-to-end naive recursion pipeline smoke tests.
//!
//! Each test:
//! 1. Proves an inner program on the host.
//! 2. Serializes `(VmProof, inner_elf)` with postcard.
//! 3. Hands that as private input to the recursion guest.
//! 4. Proves the recursion guest's execution.
//! 5. Verifies the outer proof.
//!
//! The ELFs are built on demand by `bench_vs/build_recursion_elfs.sh`.
//!
//! Tests are `#[ignore]`d because the outer proof runs the full STARK verifier
//! inside the VM (minutes per run, large memory footprint).

use std::path::PathBuf;
use std::process::Command;

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root")
        .to_path_buf()
}

fn build_elfs(root: &std::path::Path) {
    let status = Command::new("bash")
        .arg(root.join("bench_vs/build_recursion_elfs.sh"))
        .status()
        .expect("failed to spawn build helper");
    assert!(status.success(), "ELF build script failed");
}

/// Read a guest ELF artifact from a bench_vs/lambda/<name>/ build.
fn read_guest_elf(root: &std::path::Path, name: &str, bin_name: &str) -> Vec<u8> {
    let path = root.join(format!(
        "bench_vs/lambda/{name}/target/riscv64im-lambda-vm-elf/release/{bin_name}"
    ));
    std::fs::read(&path).unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()))
}

/// Core pipeline: prove an inner program, hand the proof+ELF to the recursion
/// guest, then prove and verify the outer proof.
///
/// Uses `blowup=8` for the inner proof to keep the outer prove memory tractable.
fn run_recursion_pipeline(label: &str, inner_elf_bytes: &[u8], inner_private_input: &[u8]) {
    let root = workspace_root();
    build_elfs(&root);
    let recursion_elf_bytes = read_guest_elf(&root, "recursion", "recursion-bench");

    let inner_proof_options = stark::proof::options::GoldilocksCubicProofOptions::with_blowup(8)
        .expect("blowup=8 is always valid");

    eprintln!("[{label}] proving inner (blowup=8) ...");
    let inner_proof = crate::prove_with_options_and_inputs(
        inner_elf_bytes,
        inner_private_input,
        &inner_proof_options,
        &crate::MaxRowsConfig::default(),
    )
    .expect("inner prove should succeed");
    eprintln!("[{label}] inner proof generated");

    assert!(
        crate::verify_with_options(&inner_proof, inner_elf_bytes, &inner_proof_options)
            .expect("inner verify errored"),
        "inner proof must verify on host"
    );

    let blob = postcard::to_allocvec(&(&inner_proof, &inner_elf_bytes))
        .expect("postcard encode failed");
    eprintln!(
        "[{label}] postcard blob: {} bytes (limit: MAX_PRIVATE_INPUT_SIZE)",
        blob.len()
    );
    assert!(
        blob.len() < executor::constants::MAX_PRIVATE_INPUT_SIZE as usize,
        "recursion input exceeds MAX_PRIVATE_INPUT_SIZE"
    );

    eprintln!("[{label}] proving outer (recursion guest) ...");
    let outer_proof = crate::prove_with_inputs(&recursion_elf_bytes, &blob)
        .expect("outer prove should succeed");
    eprintln!("[{label}] outer proof generated");

    assert!(
        crate::verify(&outer_proof, &recursion_elf_bytes).expect("outer verify errored"),
        "outer proof must verify on host"
    );

    assert_eq!(
        outer_proof.public_output,
        vec![1u8],
        "guest should commit success marker"
    );
}

/// Inner program: empty (halt immediately). Useful for measuring the
/// lambda-vm verifier's intrinsic recursion overhead — i.e. what it costs
/// to verify the smallest possible lambda-vm proof, with no inner workload.
#[test]
#[ignore = "slow: runs the full STARK verifier inside the VM"]
fn test_recursion_smoke_empty() {
    let root = workspace_root();
    build_elfs(&root);
    let empty_elf_bytes = read_guest_elf(&root, "empty", "empty-bench");
    run_recursion_pipeline("recursion-empty", &empty_elf_bytes, &[]);
}

/// Inner program: fibonacci(10).
#[test]
#[ignore = "slow: runs the full STARK verifier inside the VM"]
fn test_recursion_smoke() {
    let root = workspace_root();
    build_elfs(&root);
    let fib_elf_bytes = read_guest_elf(&root, "fibonacci", "fibonacci-bench");

    let n: u64 = 10;
    let inner_private_input = n.to_le_bytes().to_vec();

    run_recursion_pipeline("recursion-smoke", &fib_elf_bytes, &inner_private_input);
}
