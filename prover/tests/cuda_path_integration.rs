//! End-to-end GPU path coverage. Proves a real ELF with the `cuda` feature on,
//! asserts every dispatch counter introduced by the cuda backend actually
//! fired, then verifies the produced proof. Catches both silent CPU-fallback
//! regressions (GPU path skipped while the proof still verifies) and bad-proof
//! regressions (GPU path fired but produced output that fails verification).
//!
//! `#[ignore]`'d so the no-GPU CI path skips it. Run via `make test-cuda-integration`
//! or `cargo test -p lambda-vm-prover --release --features cuda --test cuda_path_integration -- --ignored --nocapture`.
#![cfg(feature = "cuda")]

use lambda_vm_prover::test_utils::asm_elf_bytes;
use lambda_vm_prover::{prove, verify};
use stark::gpu_lde::{
    gpu_bary_calls, gpu_batch_invert_calls, gpu_deep_calls, gpu_extend_halves_calls, gpu_fri_calls,
    gpu_lde_calls, reset_all_gpu_call_counters,
};

#[test]
#[ignore = "requires GPU; run with --ignored --nocapture"]
fn gpu_path_fires_end_to_end() {
    // Warm-up amortises PTX load + pool warm-up + first-call pinned alloc
    // so the profiled-pass counter assertions reflect only steady-state work.
    let elf = asm_elf_bytes("fib_iterative_1M");
    let _ = prove(&elf).expect("warm-up prove");

    reset_all_gpu_call_counters();

    let proof = prove(&elf).expect("prove");

    // R1 main + aux fused LDE+Merkle. Fires for every table above the LDE
    // threshold; fib_iterative_1M has plenty.
    assert!(gpu_lde_calls() > 0, "R1 GPU LDE path did not fire");

    // R3 OOD barycentric reads the R1 main + aux device handles. Fires once
    // per (eval-point, base/ext3) pair for every table that took the R1 GPU
    // path.
    assert!(gpu_bary_calls() > 0, "R3 GPU barycentric did not fire");

    // R2 fused composition LDE + no-tree keep. After #699/#700 every VM AIR's
    // composition poly has `number_of_parts == 2`, so the degree-2 quotient
    // decomposition routes through `try_extend_two_halves_gpu_keep`: one call does
    // the LDE of both halves and retains the device handle for R4 DEEP. A silent
    // fallback to the CPU `extend_half_to_lde` would drop this to zero.
    assert!(
        gpu_extend_halves_calls() > 0,
        "R2 fused composition LDE keep path did not fire"
    );

    // DEEP fires once per table that took the R1 GPU path.
    assert!(gpu_deep_calls() > 0, "R4 GPU DEEP composition did not fire");

    // FRI commit fires once per table (commit_phase_from_evaluations).
    assert!(gpu_fri_calls() > 0, "R4 GPU FRI commit did not fire");

    // GPU batch-invert dispatch fires for the R3 OOD and R4 DEEP
    // inv_denoms pipelines. A regression where either silently fell back
    // to host inv_denoms would drop this to zero.
    assert!(
        gpu_batch_invert_calls() > 0,
        "GPU batch-invert dispatch did not fire on R3 + R4"
    );

    // Counters only prove the dispatches ran; this checks the GPU proof
    // actually satisfies the verifier.
    let ok = verify(&proof, &elf).expect("verify");
    assert!(ok, "GPU-produced proof failed verification");
}

/// Focused validation of the GPU row-pair trace commitment: proves a large
/// trace with the GPU path and verifies the resulting proof. Independent of the
/// per-round counter assertions in `gpu_path_fires_end_to_end` (the R2 parts-LDE
/// assertion bit-rotted on main and cuts off before the verify). A wrong GPU
/// trace-commit leaf layout (1-row vs the new row-pair) would fail verification.
#[test]
#[ignore = "requires GPU; run with --ignored --nocapture"]
fn gpu_proof_verifies_row_pair_commitment() {
    let elf = asm_elf_bytes("fib_iterative_1M");
    reset_all_gpu_call_counters();
    let proof = prove(&elf).expect("prove");
    assert!(
        gpu_lde_calls() > 0,
        "GPU LDE path did not fire (silent CPU fallback would not test the GPU commit)"
    );
    assert!(
        verify(&proof, &elf).expect("verify"),
        "GPU-produced proof (row-pair commitment) failed verification"
    );
}
