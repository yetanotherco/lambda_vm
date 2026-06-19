//! GPU error-path coverage. Forces math-cuda dispatches to return Err at
//! chosen points and asserts the CPU fallback inside the dispatchers
//! still produces a verifying proof. Distinct from
//! `cuda_path_integration`, which covers the happy path.
//!
//! Requires the `test-cuda-faults` feature, which compiles the fault-
//! injection hook into math-cuda. Run with:
//!
//! ```text
//! cargo test -p lambda-vm-prover --release --features test-cuda-faults \
//!     --test cuda_fallback_tests -- --ignored --nocapture
//! ```
#![cfg(feature = "test-cuda-faults")]

use lambda_vm_prover::test_utils::asm_elf_bytes;
use lambda_vm_prover::{prove, verify};
use stark::gpu_lde::{gpu_batch_invert_calls, gpu_fri_calls, reset_all_gpu_call_counters};

/// FRI commit-phase CPU fallback: when the GPU dispatch errors after the
/// first transcript mutation, `try_fri_commit_gpu` must restore the
/// transcript from its snapshot and return None so the CPU loop produces
/// a verifying proof. Without the snapshot/restore the CPU fallback would
/// resume against an advanced transcript and emit an invalid proof.
///
/// We assert only that the recovered proof verifies, not that it matches
/// a baseline byte-for-byte: `prove()` is not currently deterministic
/// across runs (parallel reductions, grinding-nonce search), so proof-
/// equality is the wrong invariant. The right invariant is "the recovered
/// proof is valid".
#[test]
#[ignore = "requires GPU + test-cuda-faults; run with --ignored --nocapture"]
fn gpu_fri_fault_falls_back_to_cpu() {
    let elf = asm_elf_bytes("fib_iterative_1M");
    // Baseline: a clean prove tells us how many GPU FRI calls fire when
    // nothing is forced to fail. The per-fault runs must show exactly one
    // fewer (the table that hit the injected Err).
    reset_all_gpu_call_counters();
    let _ = prove(&elf).expect("warm-up");
    let clean = gpu_fri_calls();
    assert!(clean > 0, "GPU FRI never ran, cannot test fallback");

    for n in 1..=3i64 {
        // Force the Nth FRI fold call (across all tables) to return Err.
        // The hook auto-resets to -1 after firing.
        stark::gpu_lde::schedule_fri_fold_fault(n);
        reset_all_gpu_call_counters();

        let recovered = prove(&elf).expect("prove after fault");
        assert_eq!(
            gpu_fri_calls(),
            clean - 1,
            "expected exactly one GPU FRI fallback (fault #{n})"
        );
        assert!(
            verify(&recovered, &elf).expect("verify recovered"),
            "post-fallback proof failed verification (fault at fold #{n})"
        );
    }

    // Reset injection state for any subsequent tests in the same process.
    stark::gpu_lde::schedule_fri_fold_fault(-1);
}

/// Batch-invert CPU fallback: when `compute_and_invert_denoms_ext3_dev`
/// errors, `try_compute_and_invert_inv_denoms_dev` must return None so the
/// caller (R3 OOD in `trace.rs` or R4 DEEP in `prover.rs`) builds inv_denoms
/// on CPU and the remaining GPU path keeps running.
///
/// The injection fires the Nth time the math-cuda entry point is reached,
/// across all tables. We assert that a single fault drops `gpu_batch_invert_calls`
/// by exactly one (one table fell back, the rest succeeded) and that the
/// recovered proof still verifies.
#[test]
#[ignore = "requires GPU + test-cuda-faults; run with --ignored --nocapture"]
fn gpu_batch_invert_fault_falls_back_to_cpu() {
    let elf = asm_elf_bytes("fib_iterative_1M");
    reset_all_gpu_call_counters();
    let _ = prove(&elf).expect("warm-up");
    let clean = gpu_batch_invert_calls();
    assert!(
        clean > 0,
        "GPU batch-invert never ran, cannot test fallback"
    );

    for n in 1..=3i64 {
        stark::gpu_lde::schedule_inverse_fault(n);
        reset_all_gpu_call_counters();

        let recovered = prove(&elf).expect("prove after fault");
        assert_eq!(
            gpu_batch_invert_calls(),
            clean - 1,
            "expected exactly one GPU batch-invert fallback (fault #{n})"
        );
        assert!(
            verify(&recovered, &elf).expect("verify recovered"),
            "post-fallback proof failed verification (batch-invert fault #{n})"
        );
    }

    stark::gpu_lde::schedule_inverse_fault(-1);
}
