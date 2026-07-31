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
    // Baseline: a clean prove tells us how many GPU FRI commits fire when
    // nothing is forced to fail. A faulted run lands on `clean` or `clean - 1`,
    // never anything else.
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
        // The injection must have been consumed; otherwise this iteration
        // never reached the error path and the checks below are vacuous.
        assert!(
            stark::gpu_lde::fri_fold_fault_fired(),
            "injected FRI fold fault #{n} never fired"
        );
        // Not `clean - 1`: the prover gets two shots at the GPU commit (first
        // from the device-resident DEEP codeword, then from host evals), and
        // the hook disarms itself once it fires, so the retry usually succeeds
        // and the count comes back to `clean`. A fault landing on the host
        // entry (no retry behind it) removes exactly one; anything outside
        // {clean - 1, clean} means dispatches were double-counted or the GPU
        // path collapsed entirely.
        let count = gpu_fri_calls();
        assert!(
            (clean - 1..=clean).contains(&count),
            "fault #{n} left GPU FRI commits at {count} (clean {clean})"
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
/// across all tables. We assert that the fault really fired, that the
/// dispatch count stays in {clean - 1, clean}, and that the recovered proof
/// still verifies. The count is not pinned to `clean - 1`: R4 retries the
/// dispatch on its host DEEP arm, so only a fault landing on R3 (CPU-only
/// fallback) removes one.
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
        assert!(
            stark::gpu_lde::inverse_fault_fired(),
            "injected batch-invert fault #{n} never fired"
        );
        let count = gpu_batch_invert_calls();
        assert!(
            (clean - 1..=clean).contains(&count),
            "fault #{n} left GPU batch-invert dispatches at {count} (clean {clean})"
        );
        assert!(
            verify(&recovered, &elf).expect("verify recovered"),
            "post-fallback proof failed verification (batch-invert fault #{n})"
        );
    }

    stark::gpu_lde::schedule_inverse_fault(-1);
}
