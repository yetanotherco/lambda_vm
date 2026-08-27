//! End-to-end exercise of the device-only downgrade recovery: with
//! `LAMBDA_VM_GPU_FORCE_DOWNGRADE` set, every device-only table declines its
//! device R2 path, downloads its resident LDEs back to host
//! (`materialize_lde_trace_host`) and finishes on the host evaluator — and
//! the proof must still verify. The device-only threshold is lowered so the
//! small fixture actually produces device-only tables.
//!
//! Lives in its own integration-test binary: the env hooks are cached in
//! process-wide `OnceLock`s, so they must be set before any other test's GPU
//! dispatch initializes them.
//!
//! Requires the `cuda` feature and a visible GPU. Run with:
//!
//! ```text
//! cargo test -p lambda-vm-prover --release --features cuda \
//!     --test gpu_force_downgrade -- --ignored --nocapture
//! ```
#![cfg(feature = "cuda")]

#[test]
#[ignore = "requires GPU; run with --ignored --nocapture"]
fn forced_downgrade_prove_verifies() {
    // SAFETY: single test in this binary, set before any GPU dispatch.
    unsafe {
        std::env::set_var("LAMBDA_VM_GPU_FORCE_DOWNGRADE", "1");
        std::env::set_var("LAMBDA_VM_GPU_DEVICE_ONLY_THRESHOLD", "16384");
    }
    let ws = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root")
        .to_path_buf();
    let elf = std::fs::read(ws.join("executor/program_artifacts/rust/ethrex.elf"))
        .expect("need ethrex.elf — run `make compile-programs-rust`");
    let input = std::fs::read(ws.join("executor/tests/ethrex_simple_tx.bin")).expect("fixture");

    // Empty arena: the executor answers this guest's hint requests during the single
    // execution `prove_with_inputs` already performs, so nothing has to be supplied here.
    let proof = lambda_vm_prover::prove_with_inputs(&elf, &input, &[]).expect("prove");
    assert!(
        stark::gpu_lde::gpu_device_only_downgrades() > 0,
        "no table took the forced downgrade — the hook or the device-only gate moved"
    );
    assert!(
        lambda_vm_prover::verify(&proof, &elf).expect("verify"),
        "downgraded proof must verify"
    );
}
