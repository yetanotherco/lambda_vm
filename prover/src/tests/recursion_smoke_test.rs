//! End-to-end naive recursion pipeline smoke tests.
//!
//! Each test:
//! 1. Proves an inner program on the host.
//! 2. Serializes `(VmProof, inner_elf, opts)` with postcard.
//! 3. Hands that as private input to the recursion guest.
//! 4. Either **proves** the recursion guest's execution and verifies the outer
//!    proof (`OuterMode::Prove`), or merely **executes** the guest in-VM and
//!    reads the committed marker off the trace (`OuterMode::ExecuteOnly`) — a
//!    cheaper tier that skips the LDE/FRI that dominate the full pipeline.
//!
//! The guest ELFs are built by `make compile-recursion-elfs` (which the
//! `test-prover-all` make target depends on) and read from
//! `executor/program_artifacts/recursion/`, like every other program test.
//!
//! Tests are `#[ignore]`d because the outer proof runs the full STARK verifier
//! inside the VM (minutes per run, large memory footprint).

use std::path::PathBuf;

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root")
        .to_path_buf()
}

/// Read a recursion-suite guest ELF artifact, built by `make compile-recursion-elfs`.
fn read_guest_elf(root: &std::path::Path, name: &str) -> Vec<u8> {
    let path = root.join(format!("executor/program_artifacts/recursion/{name}.elf"));
    std::fs::read(&path).unwrap_or_else(|e| {
        panic!(
            "failed to read {} — run `make compile-recursion-elfs`: {e}",
            path.display()
        )
    })
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

    let blob =
        postcard::to_allocvec(&(&inner_proof, &inner_elf, opts)).expect("postcard encode failed");
    eprintln!("[{tag}] postcard blob: {} bytes", blob.len());
    (inner_proof, blob)
}

/// How far to take the recursion guest after it has been handed the inner
/// proof. The guest under test is the verifier either way — this only chooses
/// whether we also prove the guest's own execution.
#[derive(Clone, Copy, Debug)]
enum OuterMode {
    /// Execute the guest in-VM and read the committed marker off the trace.
    /// Skips the LDE blowup + FRI commit that dominate the full pipeline's
    /// footprint, so it needs materially less RAM than `Prove`.
    ///
    /// "Less" is not "little": `Executor::run` retains a per-instruction log
    /// and `Traces` materializes the full execution trace, so verifying even a
    /// 1-query inner proof still needs tens of GB — it OOMs on a 36 GB box.
    ExecuteOnly,
    /// Prove the guest's execution and verify the outer proof on the host. The
    /// full STARK verifier inside the VM — minutes per run, ~125 GB.
    Prove,
}

/// Execute the recursion guest in-VM on `blob` and return the bytes it
/// committed (the success marker the in-VM verifier emits).
fn execute_outer_and_commit(label: &str, recursion_elf_bytes: &[u8], blob: &[u8]) -> Vec<u8> {
    use executor::elf::Elf;
    use executor::vm::execution::Executor;

    eprintln!("[{label}] executing outer (recursion guest, in-VM verify) ...");
    let program = Elf::load(recursion_elf_bytes).expect("load recursion elf");
    let result = Executor::new(&program, blob.to_vec())
        .expect("executor new")
        .run()
        .expect("recursion guest execution failed (verify panicked in-VM?)");

    let traces = crate::tables::trace_builder::Traces::from_elf_and_logs(
        &program,
        &result.logs,
        &crate::MaxRowsConfig::default(),
        blob,
        #[cfg(feature = "disk-spill")]
        stark::storage_mode::StorageMode::Ram,
    )
    .expect("trace build");

    eprintln!(
        "[{label}] committed {} bytes: {:?} (as str: {:?})",
        traces.public_output_bytes.len(),
        traces.public_output_bytes,
        String::from_utf8_lossy(&traces.public_output_bytes),
    );
    traces.public_output_bytes
}

/// Prove the recursion guest's execution on `blob`, verify the outer proof on
/// the host, and return the bytes the guest committed.
fn prove_outer_and_commit(label: &str, recursion_elf_bytes: &[u8], blob: &[u8]) -> Vec<u8> {
    eprintln!("[{label}] proving outer (recursion guest) ...");
    let outer_proof =
        crate::prove_with_inputs(recursion_elf_bytes, blob).expect("outer prove should succeed");
    eprintln!("[{label}] outer proof generated");

    assert!(
        crate::verify(&outer_proof, recursion_elf_bytes).expect("outer verify errored"),
        "outer proof must verify on host"
    );
    outer_proof.public_output
}

/// Core pipeline: prove an inner program with the given options, hand the
/// proof+ELF+options to the recursion guest, then take the guest to `mode`
/// (execute-only or full prove) and assert it committed the `[1]` success
/// marker — i.e. the in-VM verifier accepted the inner proof.
fn run_recursion_pipeline_with_options(
    label: &str,
    inner_elf_bytes: &[u8],
    inner_private_input: &[u8],
    inner_proof_options: stark::proof::options::ProofOptions,
    mode: OuterMode,
) {
    let root = workspace_root();
    let recursion_elf_bytes = read_guest_elf(&root, "recursion");

    let (inner_proof, blob) = prove_inner_and_encode_blob(
        label,
        inner_elf_bytes,
        inner_private_input,
        &inner_proof_options,
    );

    assert!(
        crate::verify_with_options(
            &inner_proof,
            inner_elf_bytes,
            &inner_proof_options,
            None,
            None
        )
        .expect("inner verify errored"),
        "inner proof must verify on host"
    );
    assert!(
        blob.len() < executor::vm::memory::MAX_PRIVATE_INPUT_SIZE as usize,
        "recursion input exceeds MAX_PRIVATE_INPUT_SIZE"
    );

    let committed = match mode {
        OuterMode::ExecuteOnly => execute_outer_and_commit(label, &recursion_elf_bytes, &blob),
        OuterMode::Prove => prove_outer_and_commit(label, &recursion_elf_bytes, &blob),
    };

    assert_eq!(
        committed,
        vec![1u8],
        "recursion guest must commit the [1] success marker (in-VM verify accepted)"
    );
    eprintln!("[{label}] guest committed [1]: in-VM verify accepted ✓");
}

/// Convenience wrapper using `blowup=8` for the inner proof — the default for
/// the `empty` and `fibonacci` cases, chosen to keep outer-prove memory tractable.
fn run_recursion_pipeline(
    label: &str,
    inner_elf_bytes: &[u8],
    inner_private_input: &[u8],
    mode: OuterMode,
) {
    let inner_proof_options = stark::proof::options::GoldilocksCubicProofOptions::with_blowup(8)
        .expect("blowup=8 is always valid");
    run_recursion_pipeline_with_options(
        label,
        inner_elf_bytes,
        inner_private_input,
        inner_proof_options,
        mode,
    );
}

/// Reproduce the recursion guest's EXACT path on the host — decode the postcard
/// blob into `(VmProof, Vec<u8>, ProofOptions)` and call `verify_with_options`.
/// The cheapest regression guard in this file: no VM execution, just the
/// encode/decode contract plus a host verify, so it catches drift in the proof
/// format or the blob layout in seconds. Unlike the guest, a failure here
/// surfaces the actual error instead of an infinite abort loop.
#[test]
#[ignore = "needs prebuilt guest ELF (make compile-recursion-elfs)"]
fn test_recursion_blob_decodes_and_verifies_on_host() {
    let root = workspace_root();
    let empty_elf_bytes = read_guest_elf(&root, "empty");
    let (_inner, blob) =
        prove_inner_and_encode_blob("roundtrip", &empty_elf_bytes, &[], &MIN_PROOF_OPTIONS);

    // Decode exactly as the guest does.
    let decoded: Result<(crate::VmProof, Vec<u8>, crate::ProofOptions), _> =
        postcard::from_bytes(&blob);
    let (vm_proof, inner_elf, options) = match decoded {
        Ok(t) => t,
        Err(e) => panic!("[roundtrip] postcard DECODE failed (this is the guest panic): {e}"),
    };
    eprintln!(
        "[roundtrip] decode ok: elf {} bytes, blowup {}",
        inner_elf.len(),
        options.blowup_factor
    );

    match crate::verify_with_options(&vm_proof, &inner_elf, &options, None, None) {
        Ok(true) => eprintln!("[roundtrip] verify ok=true — guest path is sound"),
        Ok(false) => panic!(
            "[roundtrip] verify returned FALSE (guest hits assert!(ok)) — proof did not survive the postcard round-trip"
        ),
        Err(e) => panic!("[roundtrip] verify ERRORED (guest hits .expect): {e:?}"),
    }
}

// === Execute-only tier ========================================================
// Mirrors the proving tests below, but stops at `OuterMode::ExecuteOnly`: the
// guest runs in-VM and we read the committed marker off the trace, skipping the
// outer STARK prove. Needs tens of GB (execution trace), not the ~125 GB the
// full outer prove wants — but still OOMs on a 36 GB box.

/// Execute-only mirror of `test_recursion_prove_empty`: verify a `blowup=8`
/// proof of the empty program in-VM.
#[test]
#[ignore = "needs prebuilt recursion guest ELF + tens of GB RAM (execution trace)"]
fn test_recursion_execute_empty() {
    let root = workspace_root();
    let empty_elf_bytes = read_guest_elf(&root, "empty");
    run_recursion_pipeline(
        "recursion-exec-empty",
        &empty_elf_bytes,
        &[],
        OuterMode::ExecuteOnly,
    );
}

/// Execute-only mirror of `test_recursion_prove_1query`: smallest possible
/// inner proof (blowup=2, 1 query) → least guest work.
#[test]
#[ignore = "needs prebuilt recursion guest ELF + tens of GB RAM (execution trace)"]
fn test_recursion_execute_1query() {
    let root = workspace_root();
    let empty_elf_bytes = read_guest_elf(&root, "empty");
    run_recursion_pipeline_with_options(
        "recursion-exec-1query",
        &empty_elf_bytes,
        &[],
        MIN_PROOF_OPTIONS,
        OuterMode::ExecuteOnly,
    );
}

/// Execute-only mirror of `test_recursion_prove`: verify a `blowup=8` proof of
/// fibonacci(10) in-VM.
#[test]
#[ignore = "needs prebuilt recursion guest ELF + tens of GB RAM (execution trace)"]
fn test_recursion_execute() {
    let root = workspace_root();
    let fib_elf_bytes = read_guest_elf(&root, "fibonacci");

    let n: u64 = 10;
    let inner_private_input = n.to_le_bytes().to_vec();

    run_recursion_pipeline(
        "recursion-exec-fib",
        &fib_elf_bytes,
        &inner_private_input,
        OuterMode::ExecuteOnly,
    );
}

// === Full-prove tier ==========================================================

/// Inner program: empty (halt immediately). Useful for measuring the
/// lambda-vm verifier's intrinsic recursion overhead — i.e. what it costs
/// to verify the smallest possible lambda-vm proof, with no inner workload.
#[test]
#[ignore = "slow: runs the full STARK verifier inside the VM"]
fn test_recursion_prove_empty() {
    let root = workspace_root();
    let empty_elf_bytes = read_guest_elf(&root, "empty");
    run_recursion_pipeline(
        "recursion-prove-empty",
        &empty_elf_bytes,
        &[],
        OuterMode::Prove,
    );
}

/// Inner program: empty, but with the absolute-minimum FRI parameters
/// (blowup=2, **fri_number_of_queries=1**). This is a "can the pipeline even
/// run end-to-end on a 125 GB box" experiment — security is intentionally
/// terrible. Use only for capacity probing.
#[test]
#[ignore = "slow: runs the full STARK verifier inside the VM"]
fn test_recursion_prove_1query() {
    let root = workspace_root();
    let empty_elf_bytes = read_guest_elf(&root, "empty");

    run_recursion_pipeline_with_options(
        "recursion-prove-1query",
        &empty_elf_bytes,
        &[],
        MIN_PROOF_OPTIONS,
        OuterMode::Prove,
    );
}

/// Inner program: fibonacci(10).
#[test]
#[ignore = "slow: runs the full STARK verifier inside the VM"]
fn test_recursion_prove() {
    let root = workspace_root();
    let fib_elf_bytes = read_guest_elf(&root, "fibonacci");

    let n: u64 = 10;
    let inner_private_input = n.to_le_bytes().to_vec();

    run_recursion_pipeline(
        "recursion-prove-fib",
        &fib_elf_bytes,
        &inner_private_input,
        OuterMode::Prove,
    );
}
