//! End-to-end naive recursion pipeline smoke tests.
//!
//! Each test:
//! 1. Proves an inner program on the host.
//! 2. Serializes `(VmProof, inner_elf, opts)` with postcard.
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

/// Path to a guest ELF artifact from a bench_vs/lambda/<name>/ build.
fn guest_elf_path(root: &std::path::Path, name: &str, bin_name: &str) -> PathBuf {
    root.join(format!(
        "bench_vs/lambda/{name}/target/riscv64im-lambda-vm-elf/release/{bin_name}"
    ))
}

/// Read a guest ELF artifact from a bench_vs/lambda/<name>/ build.
fn read_guest_elf(root: &std::path::Path, name: &str, bin_name: &str) -> Vec<u8> {
    let path = guest_elf_path(root, name, bin_name);
    std::fs::read(&path).unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()))
}

/// Minimum-security FRI parameters: blowup=2, a single FRI query. Security is
/// intentionally terrible — used by the capacity-probing test, where the goal
/// is the smallest possible inner proof, not a sound one.
/// (`GoldilocksCubicProofOptions::with_blowup` derives a query count from a
/// 128-bit target, far more than we want here.)
const MIN_PROOF_OPTIONS: stark::proof::options::ProofOptions =
    stark::proof::options::ProofOptions {
        blowup_factor: 2,
        fri_number_of_queries: 1,
        coset_offset: 3,
        grinding_factor: 1,
    };

/// Prove `inner_elf` (fed `inner_input`) under `opts`, then package
/// `(proof, elf, opts)` into the postcard blob the recursion guest consumes as
/// its private input. `tag` prefixes the progress lines. Returns the inner
/// proof — callers that re-verify it on the host need it — next to the encoded
/// blob.
fn prove_inner_and_encode_blob(
    tag: &str,
    inner_elf: &[u8],
    inner_input: &[u8],
    opts: &stark::proof::options::ProofOptions,
) -> (crate::VmProof, Vec<u8>) {
    eprintln!(
        "[{tag}] proving inner (blowup={}, fri_queries={}) ...",
        opts.blowup_factor, opts.fri_number_of_queries
    );
    let inner_proof = crate::prove_with_options_and_inputs(
        inner_elf,
        inner_input,
        opts,
        &crate::MaxRowsConfig::default(),
    )
    .expect("inner prove should succeed");

    let blob = postcard::to_allocvec(&(&inner_proof, &inner_elf, opts))
        .expect("postcard encode failed");
    eprintln!("[{tag}] postcard blob: {} bytes", blob.len());
    (inner_proof, blob)
}

/// Core pipeline: prove an inner program with the given options, hand the
/// proof+ELF+options to the recursion guest, then prove and verify the outer
/// proof.
fn run_recursion_pipeline_with_options(
    label: &str,
    inner_elf_bytes: &[u8],
    inner_private_input: &[u8],
    inner_proof_options: stark::proof::options::ProofOptions,
) {
    let root = workspace_root();
    build_elfs(&root);
    let recursion_elf_bytes = read_guest_elf(&root, "recursion", "recursion-bench");

    let (inner_proof, blob) =
        prove_inner_and_encode_blob(label, inner_elf_bytes, inner_private_input, &inner_proof_options);

    assert!(
        crate::verify_with_options(&inner_proof, inner_elf_bytes, &inner_proof_options, None, None)
            .expect("inner verify errored"),
        "inner proof must verify on host"
    );
    assert!(
        blob.len() < executor::constants::MAX_PRIVATE_INPUT_SIZE as usize,
        "recursion input exceeds MAX_PRIVATE_INPUT_SIZE"
    );

    eprintln!("[{label}] proving outer (recursion guest) ...");
    let outer_proof =
        crate::prove_with_inputs(&recursion_elf_bytes, &blob).expect("outer prove should succeed");
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

/// Convenience wrapper using `blowup=8` for the inner proof — the default for
/// the existing smoke tests, chosen to keep outer-prove memory tractable.
fn run_recursion_pipeline(label: &str, inner_elf_bytes: &[u8], inner_private_input: &[u8]) {
    let inner_proof_options = stark::proof::options::GoldilocksCubicProofOptions::with_blowup(8)
        .expect("blowup=8 is always valid");
    run_recursion_pipeline_with_options(
        label,
        inner_elf_bytes,
        inner_private_input,
        inner_proof_options,
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

/// Inner program: empty, but with the absolute-minimum FRI parameters
/// (blowup=2, **fri_number_of_queries=1**). This is a "can the pipeline even
/// run end-to-end on a 125 GB box" experiment — security is intentionally
/// terrible. Use only for capacity probing.
#[test]
#[ignore = "slow: runs the full STARK verifier inside the VM"]
fn test_recursion_smoke_1query() {
    let root = workspace_root();
    build_elfs(&root);
    let empty_elf_bytes = read_guest_elf(&root, "empty", "empty-bench");

    run_recursion_pipeline_with_options(
        "recursion-1query",
        &empty_elf_bytes,
        &[],
        MIN_PROOF_OPTIONS,
    );
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
