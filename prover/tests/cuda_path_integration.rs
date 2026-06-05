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
    gpu_bary_calls, gpu_comp_poly_tree_calls, gpu_deep_calls, gpu_fri_calls, gpu_lde_calls,
    gpu_parts_lde_calls, reset_all_gpu_call_counters,
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

    // R2 ext3 LDE of composition-poly parts. Only fires when an AIR's
    // `number_of_parts > 2`. The branch and shift tables have degree-3
    // transition constraints, so this triggers on any non-trivial prove.
    assert!(gpu_parts_lde_calls() > 0, "R2 GPU parts LDE did not fire");

    // R2 comp-poly Merkle tree build, paired with the parts LDE above.
    assert!(
        gpu_comp_poly_tree_calls() > 0,
        "R2 GPU comp-poly tree did not fire"
    );

    // R4 DEEP composition. Reuses the R1 main/aux handles and (when R2
    // parts LDE took the keep path) the parts handle on Round2. Fires
    // once per table whose lde_size clears the threshold.
    assert!(gpu_deep_calls() > 0, "R4 GPU DEEP composition did not fire");

    // R4 FRI commit phase. One call per table (per
    // `commit_phase_from_evaluations`); each call drives all FRI layers
    // device-side internally.
    assert!(gpu_fri_calls() > 0, "R4 GPU FRI commit did not fire");

    // Verify the GPU-produced proof. Catches the worst regression class:
    // GPU dispatches fire but the device path produces a proof that
    // doesn't satisfy the verifier.
    let ok = verify(&proof, &elf).expect("verify");
    assert!(ok, "GPU-produced proof failed verification");
}
