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
    gpu_bary_calls, gpu_batch_invert_calls, gpu_comp_poly_tree_calls, gpu_composition_calls,
    gpu_deep_calls, gpu_device_only_calls, gpu_extend_halves_calls, gpu_fri_calls, gpu_lde_calls,
    gpu_logup_calls, gpu_opening_gather_calls, gpu_parts_lde_calls, reset_all_gpu_call_counters,
};

/// The R2 GPU composition-poly path (fused `H = z·Σβᵢ·Cᵢ + boundary`) fires and
/// yields a verifying proof. Guards both a silent CPU fallback (counter == 0)
/// and a bad-`H` regression (fires but the proof fails verification).
#[test]
#[ignore = "requires GPU; run with --ignored --nocapture"]
fn gpu_composition_path_fires_and_verifies() {
    let elf = asm_elf_bytes("fib_iterative_1M");
    reset_all_gpu_call_counters();
    let proof = prove(&elf).expect("prove");
    assert!(
        gpu_composition_calls() > 0,
        "GPU composition path did not fire (tables below threshold or gate fell back to CPU)"
    );
    assert!(
        verify(&proof, &elf).expect("verify"),
        "GPU-produced proof (fused composition) failed verification"
    );
}

/// The GPU LogUp aux-build path fires and still yields a verifying proof.
#[test]
#[ignore = "requires GPU; run with --ignored --nocapture"]
fn gpu_logup_aux_build_fires_and_verifies() {
    let elf = asm_elf_bytes("fib_iterative_1M");
    reset_all_gpu_call_counters();
    let proof = prove(&elf).expect("prove");
    assert!(
        gpu_logup_calls() > 0,
        "GPU LogUp aux-build path did not fire (tables below threshold or fell back)"
    );
    assert!(
        verify(&proof, &elf).expect("verify"),
        "proof failed to verify"
    );
}

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

    // R2 GPU composition-poly LDE. Fires via one of two paths depending on the
    // AIR's `number_of_parts`: the fused two-halves quotient decomposition for
    // the common degree-2 case (`== 2`, counted by `gpu_extend_halves_calls`),
    // or the batched parts LDE for `> 2` (counted by `gpu_parts_lde_calls`).
    // fib_iterative_1M only exercises the degree-2 path, so assert on either.
    assert!(
        gpu_extend_halves_calls() + gpu_parts_lde_calls() > 0,
        "R2 GPU composition LDE did not fire (neither two-halves d2 nor parts>2 path)"
    );

    // R2 comp-poly Merkle tree build. Dispatched unconditionally (independent of
    // the parts-count branch above), so it fires for the common degree-2 case
    // too; a silent CPU fallback would still verify, so this counter is what
    // guards the GPU comp-poly-tree dispatch.
    assert!(
        gpu_comp_poly_tree_calls() > 0,
        "R2 GPU comp-poly tree did not fire"
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

/// Focused validation of the GPU FRI early-termination commit: proves a large
/// trace (which exceeds the GPU FRI threshold), confirms the GPU FRI commit
/// path fired, and verifies the resulting proof. Independent of the per-round
/// counter assertions in `gpu_path_fires_end_to_end` (some of which are
/// sensitive to AIR/LDE shape and may bit-rot across LDE reworks).
#[test]
#[ignore = "requires GPU; run with --ignored --nocapture"]
fn gpu_fri_commit_produces_verifiable_proof() {
    let elf = asm_elf_bytes("fib_iterative_1M");
    reset_all_gpu_call_counters();
    let proof = prove(&elf).expect("prove");
    assert!(
        gpu_fri_calls() > 0,
        "GPU FRI commit path did not fire on a 1M-row trace"
    );
    assert!(
        verify(&proof, &elf).expect("verify"),
        "GPU-produced proof (early-termination FRI) failed verification"
    );
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

/// The full-residency Stage-2 R4 opening path fires: query row values are
/// gathered straight off the device LDE (not the host trace) and, guarded by the
/// in-prover cross-check against the host gather, still yield a verifying proof.
/// Guards a silent regression where the openings quietly revert to the host
/// path (which would still verify but drop the data-residency win). The
/// cross-check `assert_eq!`s inside `open_deep_composition_poly` are
/// release-active, so a divergent device gather would panic the prove here.
#[test]
#[ignore = "requires GPU; run with --ignored --nocapture"]
fn gpu_opening_gather_fires_and_verifies() {
    let elf = asm_elf_bytes("fib_iterative_1M");
    reset_all_gpu_call_counters();
    let proof = prove(&elf).expect("prove");
    assert!(
        gpu_opening_gather_calls() > 0,
        "device-resident opening gather did not fire (openings fell back to the host LDE)"
    );
    assert!(
        verify(&proof, &elf).expect("verify"),
        "GPU-produced proof (device-gathered openings) failed verification"
    );
}

/// The full-residency Stage-3 device-only path fires: at least one table keeps
/// its round-1 LDE device-resident (the host D2H is skipped), and the proof
/// still verifies. This exercises every `host_trace_empty` hard-abort guard on
/// the happy path (none may fire) plus the GPU-only R2/R3/R4 paths reading the
/// device LDE with no host trace behind them. A regression that silently
/// reverts to the host D2H drops the counter to 0 (while the proof would still
/// verify), and a mis-gate that forces a host fallback panics one of the guards.
#[test]
#[ignore = "requires GPU; run with --ignored --nocapture"]
fn gpu_device_only_residency_fires_and_verifies() {
    let elf = asm_elf_bytes("fib_iterative_1M");
    reset_all_gpu_call_counters();
    let proof = prove(&elf).expect("prove");
    assert!(
        gpu_device_only_calls() > 0,
        "device-only residency path did not fire (every table kept its host trace)"
    );
    assert_eq!(
        stark::gpu_lde::gpu_device_only_downgrades(),
        0,
        "a device-only table was downgraded back to host on the happy path \
         (a device dispatch declined that the gate should mirror)"
    );
    assert!(
        verify(&proof, &elf).expect("verify"),
        "GPU-produced proof (device-only residency) failed verification"
    );
}
