//! Runtime end-to-end evidence that the `KeccakPermute` precompile is wired
//! through the guest hashing path.
//!
//! Static evidence (ELF disassembly) already shows that `keccak::p1600` in the
//! `keccak-roundtrip-bench` guest is a ~20-byte ecall stub (a7 = u64::MAX-1,
//! the `KeccakPermute` syscall number) rather than ~1500 bytes of pure-Rust
//! Keccak rounds. That proves the patch is *wired*, but not that it runs and
//! produces the right answer at execution time.
//!
//! This test closes that gap end-to-end:
//!
//! 1. Builds the `keccak-roundtrip-bench` guest, which uses `sha3::Keccak256`
//!    (which delegates to `keccak::p1600`, which on `target_arch = "riscv64"`
//!    is the lambda-vm precompile ecall via the `keccak-patched` crate).
//! 2. Runs the guest inside the lambda-vm prover for each FIPS-202 test
//!    vector and proves the execution.
//! 3. Verifies the proof on the host.
//! 4. Asserts that the committed `public_output` matches the reference
//!    Keccak256 digest of the input message.
//!
//! If the precompile were unwired, mis-wired, or computed the wrong
//! permutation, the digest committed by the guest would not match the
//! reference vector and the test would fail.
//!
//! As an additional diagnostic, the test also runs the guest through the
//! executor directly to count the number of `KeccakPermute` syscall ecalls,
//! confirming the precompile is actually exercised at runtime.

use std::path::{Path, PathBuf};
use std::process::Command;

use executor::constants::KECCAK_SYSCALL_NUMBER;
use executor::elf::Elf;
use executor::vm::execution::Executor;
use executor::vm::instruction::decoding::Instruction;

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root")
        .to_path_buf()
}

fn build_elfs(root: &Path) {
    let status = Command::new("bash")
        .arg(root.join("bench_vs/build_recursion_elfs.sh"))
        .status()
        .expect("failed to spawn build helper");
    assert!(status.success(), "ELF build script failed");
}

fn read_guest_elf(root: &Path, name: &str, bin_name: &str) -> Vec<u8> {
    let path = root.join(format!(
        "bench_vs/lambda/{name}/target/riscv64im-lambda-vm-elf/release/{bin_name}"
    ));
    std::fs::read(&path).unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()))
}

/// FIPS-202 Keccak256 reference vectors.
///
/// Sources:
/// * empty input — `c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470`
/// * `"abc"`     — `4e03657aea45a94fc7d47ba826c8d667c0d1e6e33a64a036ec44f58fa12d6c45`
/// * `"The quick brown fox jumps over the lazy dog"`
///   — `4d741b6f1eb29cb2a9b9911c82f56fa8d73b04959d3d9d222895df6c0b28aa15`
const TEST_VECTORS: &[(&str, &[u8], [u8; 32])] = &[
    (
        "empty",
        b"",
        hex32("c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470"),
    ),
    (
        "abc",
        b"abc",
        hex32("4e03657aea45a94fc7d47ba826c8d667c0d1e6e33a64a036ec44f58fa12d6c45"),
    ),
    (
        "fox",
        b"The quick brown fox jumps over the lazy dog",
        hex32("4d741b6f1eb29cb2a9b9911c82f56fa8d73b04959d3d9d222895df6c0b28aa15"),
    ),
];

/// `const fn` hex-to-`[u8; 32]` for the test vectors above.
const fn hex32(s: &str) -> [u8; 32] {
    let b = s.as_bytes();
    assert!(b.len() == 64, "expected 64 hex chars");
    let mut out = [0u8; 32];
    let mut i = 0;
    while i < 32 {
        out[i] = hex_byte(b[2 * i]) * 16 + hex_byte(b[2 * i + 1]);
        i += 1;
    }
    out
}

const fn hex_byte(c: u8) -> u8 {
    match c {
        b'0'..=b'9' => c - b'0',
        b'a'..=b'f' => c - b'a' + 10,
        b'A'..=b'F' => c - b'A' + 10,
        _ => panic!("invalid hex char"),
    }
}

/// Count `KeccakPermute` syscall invocations by running the guest through the
/// executor and inspecting the log of executed instructions.
///
/// Returns `(total_cycles, keccak_syscalls)`.
fn count_keccak_syscalls(elf_bytes: &[u8], private_input: &[u8]) -> (usize, usize) {
    let program = Elf::load(elf_bytes).expect("ELF load failed");
    let executor = Executor::new(&program, private_input.to_vec()).expect("Executor::new failed");
    let result = executor.run().expect("executor.run() failed");

    let mut keccak_syscalls = 0usize;
    for log in &result.logs {
        if let Some(instr) = result.instructions.get(&log.current_pc)
            && matches!(instr, Instruction::EcallEbreak)
            && log.src1_val == KECCAK_SYSCALL_NUMBER
        {
            keccak_syscalls += 1;
        }
    }
    (result.logs.len(), keccak_syscalls)
}

#[test]
#[ignore = "slow: runs the lambda-vm prover end-to-end on real ELFs"]
fn test_keccak_precompile_runtime() {
    let root = workspace_root();
    build_elfs(&root);
    let elf_bytes = read_guest_elf(&root, "keccak-roundtrip", "keccak-roundtrip-bench");

    for (label, msg, expected) in TEST_VECTORS {
        eprintln!("[keccak-precompile/{label}] message len = {}", msg.len());

        // Diagnostic: confirm the KeccakPermute precompile is actually hit.
        let (cycles, keccak_syscalls) = count_keccak_syscalls(&elf_bytes, msg);
        eprintln!(
            "[keccak-precompile/{label}] cycles = {cycles}, KeccakPermute syscalls = {keccak_syscalls}",
        );
        assert!(
            keccak_syscalls > 0,
            "{label}: guest must invoke the KeccakPermute precompile at least once",
        );

        // End-to-end: prove → verify → check public_output == reference digest.
        let vm_proof = crate::prove_with_inputs(&elf_bytes, msg).expect("prove_with_inputs failed");
        assert!(
            crate::verify(&vm_proof, &elf_bytes).expect("verify errored"),
            "{label}: proof must verify on host",
        );
        assert_eq!(
            vm_proof.public_output,
            expected.to_vec(),
            "{label}: committed digest does not match FIPS-202 reference; \
             the precompile is unwired or computes the wrong permutation",
        );
    }
}

/// Cheaper sibling: same correctness check but only runs the executor (no
/// STARK prove/verify). Useful for fast regression CI and to inspect cycle /
/// syscall counts without paying the prove cost.
#[test]
fn test_keccak_precompile_executor_only() {
    let root = workspace_root();
    build_elfs(&root);
    let elf_bytes = read_guest_elf(&root, "keccak-roundtrip", "keccak-roundtrip-bench");

    for (label, msg, expected) in TEST_VECTORS {
        let program = Elf::load(&elf_bytes).expect("ELF load");
        let executor = Executor::new(&program, msg.to_vec()).expect("Executor::new");
        let result = executor.run().expect("executor.run()");

        let mut keccak_syscalls = 0usize;
        for log in &result.logs {
            if let Some(instr) = result.instructions.get(&log.current_pc)
                && matches!(instr, Instruction::EcallEbreak)
                && log.src1_val == KECCAK_SYSCALL_NUMBER
            {
                keccak_syscalls += 1;
            }
        }

        eprintln!(
            "[keccak-precompile/{label}] cycles = {}, KeccakPermute syscalls = {keccak_syscalls}",
            result.logs.len(),
        );
        assert!(
            keccak_syscalls > 0,
            "{label}: guest must invoke the KeccakPermute precompile",
        );
        assert_eq!(
            result.return_values.memory_values,
            expected.to_vec(),
            "{label}: committed digest does not match FIPS-202 reference",
        );
    }
}
