//! End-to-end naive recursion pipeline smoke test.
//!
//! 1. Prove an inner program (fibonacci) on the host.
//! 2. Serialize `(VmProof, inner_elf)` with postcard.
//! 3. Hand that as private input to the recursion guest.
//! 4. Prove the recursion guest's execution.
//! 5. Verify the outer proof.
//!
//! Both ELFs are built on demand by the shell helper script:
//!   `bench_vs/build_recursion_elfs.sh`
//!
//! Marked `#[ignore]` because the outer proof is large (the guest runs the
//! full STARK verifier in software keccak — minutes per run).

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

#[test]
#[ignore = "slow: runs the full STARK verifier inside the VM with soft keccak"]
fn test_recursion_smoke() {
    let root = workspace_root();
    build_elfs(&root);

    let fib_elf_bytes =
        std::fs::read(root.join(
            "bench_vs/lambda/fibonacci/target/riscv64im-lambda-vm-elf/release/fibonacci-bench",
        ))
        .expect("fibonacci-bench ELF not found");
    let recursion_elf_bytes =
        std::fs::read(root.join(
            "bench_vs/lambda/recursion/target/riscv64im-lambda-vm-elf/release/recursion-bench",
        ))
        .expect("recursion-bench ELF not found");

    // Inner program: compute fib(10).
    let n: u64 = 10;
    let mut inner_private_input = Vec::with_capacity(8);
    inner_private_input.extend_from_slice(&n.to_le_bytes());

    eprintln!("[recursion-smoke] proving inner (fibonacci) ...");
    let inner_proof = crate::prove_with_inputs(&fib_elf_bytes, &inner_private_input)
        .expect("inner prove should succeed");
    eprintln!("[recursion-smoke] inner proof generated");

    assert!(
        crate::verify(&inner_proof, &fib_elf_bytes).expect("inner verify errored"),
        "inner proof must verify on host"
    );

    // Build the recursion guest's private input: postcard-encoded `(VmProof, Vec<u8>)`.
    let blob =
        postcard::to_allocvec(&(&inner_proof, &fib_elf_bytes)).expect("postcard encode failed");
    eprintln!(
        "[recursion-smoke] postcard blob: {} bytes (limit: MAX_PRIVATE_INPUT_SIZE)",
        blob.len()
    );
    assert!(
        blob.len() < executor::constants::MAX_PRIVATE_INPUT_SIZE as usize,
        "recursion input exceeds MAX_PRIVATE_INPUT_SIZE"
    );

    eprintln!("[recursion-smoke] proving outer (recursion guest) ...");
    let outer_proof =
        crate::prove_with_inputs(&recursion_elf_bytes, &blob).expect("outer prove should succeed");
    eprintln!("[recursion-smoke] outer proof generated");

    assert!(
        crate::verify(&outer_proof, &recursion_elf_bytes).expect("outer verify errored"),
        "outer proof must verify on host"
    );

    // The recursion guest commits a single `1` byte on success.
    assert_eq!(
        outer_proof.public_output,
        vec![1u8],
        "guest should commit success marker"
    );
}
