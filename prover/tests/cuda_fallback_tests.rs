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
use stark::gpu_lde::{
    gpu_batch_invert_calls, gpu_bitwise_trace_calls, gpu_branch_trace_calls,
    gpu_bytewise_trace_calls, gpu_cpu32_trace_calls, gpu_cpu_trace_calls, gpu_decode_trace_calls,
    gpu_dvrm_trace_calls, gpu_fri_calls, gpu_keccak_trace_calls, gpu_load_trace_calls,
    gpu_lt_trace_calls, gpu_memw_aligned_trace_calls, gpu_memw_register_trace_calls,
    gpu_memw_trace_calls, gpu_mul_trace_calls, gpu_page_trace_calls, gpu_shift_trace_calls,
    gpu_store_trace_calls, reset_all_gpu_call_counters,
};

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

/// PAGE-trace CPU fallback: forcing `generate_page_trace_dev` to Err must
/// make the prover-side wrapper return None and the CPU loop in
/// `generate_page_trace` run to completion. Recovered proof must verify.
///
/// `fib_iterative_1M` only has a single memory page → one GPU page-trace
/// call per prove. We schedule the single fault and check the counter
/// drops to zero (one CPU fallback fired).
#[test]
#[ignore = "requires GPU + test-cuda-faults; run with --ignored --nocapture"]
fn gpu_page_trace_fault_falls_back_to_cpu() {
    let elf = asm_elf_bytes("fib_iterative_1M");
    reset_all_gpu_call_counters();
    let _ = prove(&elf).expect("warm-up");
    let clean = gpu_page_trace_calls();
    assert!(clean > 0, "GPU page-trace never ran, cannot test fallback");

    stark::gpu_lde::schedule_page_trace_fault(1);
    reset_all_gpu_call_counters();
    let recovered = prove(&elf).expect("prove after fault");
    assert_eq!(
        gpu_page_trace_calls(),
        clean - 1,
        "expected exactly one GPU page-trace fallback",
    );
    assert!(
        verify(&recovered, &elf).expect("verify recovered"),
        "post-fallback proof failed verification (page-trace fault)",
    );

    stark::gpu_lde::schedule_page_trace_fault(-1);
}

/// DECODE-trace CPU fallback: same shape as `gpu_page_trace_fault_...`.
/// `generate_decode_trace_dev` fires once per prove, so a single injected
/// fault drops the counter to zero and the entire decode trace is built
/// by the existing CPU loop.
#[test]
#[ignore = "requires GPU + test-cuda-faults; run with --ignored --nocapture"]
fn gpu_decode_trace_fault_falls_back_to_cpu() {
    let elf = asm_elf_bytes("fib_iterative_1M");
    reset_all_gpu_call_counters();
    let _ = prove(&elf).expect("warm-up");
    let clean = gpu_decode_trace_calls();
    assert!(clean > 0, "GPU decode-trace never ran, cannot test fallback");

    stark::gpu_lde::schedule_decode_trace_fault(1);
    reset_all_gpu_call_counters();
    let recovered = prove(&elf).expect("prove after fault");
    assert_eq!(
        gpu_decode_trace_calls(),
        clean - 1,
        "expected exactly one GPU decode-trace fallback",
    );
    assert!(
        verify(&recovered, &elf).expect("verify recovered"),
        "post-fallback proof failed verification (decode-trace fault)",
    );

    stark::gpu_lde::schedule_decode_trace_fault(-1);
}

// =============================================================================
// PR-6 trace-expansion fallback tests (one per ported table).
//
// Every kernel-side wrapper in `stark::gpu_lde` returns Option<…>, and the
// prover's `try_generate_X_trace_gpu` helper propagates None on cudarc errors
// so the existing CPU loop runs. These tests force that path: schedule a
// single Err from the math-cuda dispatch and assert the table's GPU counter
// drops by exactly one (one row built on CPU) while the recovered proof
// still verifies.
//
// Tables that didn't fire on the chosen ELF print a "skipping" message and
// pass — the harness can't fault-inject what never ran. Pick an ELF that
// exercises the target table to extend coverage. Known gaps:
//   - ECSM has no test ELF on disk (`test_ecsm.elf` is referenced by src
//     but the artifact is missing — same pre-existing gap as `heap_alloc`).
// =============================================================================

/// Shared fallback-test driver. Runs a clean prove, captures the table's
/// dispatch count, then schedules a single GPU error for that table and
/// re-proves. Asserts:
///   1. The counter dropped by exactly 1 (one CPU fallback for that table).
///   2. The recovered proof verifies.
///
/// Skips (returns OK) when the table never fires on the chosen ELF —
/// callers can swap in a different ELF to extend coverage.
fn fallback_for(
    elf_name: &str,
    table_label: &str,
    count_fn: fn() -> u64,
    schedule_fault: fn(i64),
) {
    let elf = asm_elf_bytes(elf_name);
    reset_all_gpu_call_counters();
    let _ = prove(&elf).expect("warm-up prove");
    let clean = count_fn();
    if clean == 0 {
        eprintln!(
            "[{table_label}] table never fired on `{elf_name}`; skipping fallback test"
        );
        return;
    }

    schedule_fault(1);
    reset_all_gpu_call_counters();
    let recovered = prove(&elf).expect("prove after fault");
    assert_eq!(
        count_fn(),
        clean - 1,
        "[{table_label}] expected exactly one GPU fallback (counter dropped by 1)",
    );
    assert!(
        verify(&recovered, &elf).expect("verify recovered"),
        "[{table_label}] post-fallback proof failed verification",
    );

    schedule_fault(-1);
}

#[test]
#[ignore = "requires GPU + test-cuda-faults; run with --ignored --nocapture"]
fn gpu_bitwise_trace_fault_falls_back_to_cpu() {
    fallback_for(
        "all_instructions_64",
        "bitwise",
        gpu_bitwise_trace_calls,
        stark::gpu_lde::schedule_bitwise_trace_fault,
    );
}

#[test]
#[ignore = "requires GPU + test-cuda-faults; run with --ignored --nocapture"]
fn gpu_load_trace_fault_falls_back_to_cpu() {
    fallback_for(
        "all_instructions_64",
        "load",
        gpu_load_trace_calls,
        stark::gpu_lde::schedule_load_trace_fault,
    );
}

#[test]
#[ignore = "requires GPU + test-cuda-faults; run with --ignored --nocapture"]
fn gpu_store_trace_fault_falls_back_to_cpu() {
    fallback_for(
        "all_instructions_64",
        "store",
        gpu_store_trace_calls,
        stark::gpu_lde::schedule_store_trace_fault,
    );
}

#[test]
#[ignore = "requires GPU + test-cuda-faults; run with --ignored --nocapture"]
fn gpu_bytewise_trace_fault_falls_back_to_cpu() {
    fallback_for(
        "all_instructions_64",
        "bytewise",
        gpu_bytewise_trace_calls,
        stark::gpu_lde::schedule_bytewise_trace_fault,
    );
}

#[test]
#[ignore = "requires GPU + test-cuda-faults; run with --ignored --nocapture"]
fn gpu_shift_trace_fault_falls_back_to_cpu() {
    fallback_for(
        "all_instructions_64",
        "shift",
        gpu_shift_trace_calls,
        stark::gpu_lde::schedule_shift_trace_fault,
    );
}

#[test]
#[ignore = "requires GPU + test-cuda-faults; run with --ignored --nocapture"]
fn gpu_memw_aligned_trace_fault_falls_back_to_cpu() {
    fallback_for(
        "all_instructions_64",
        "memw_aligned",
        gpu_memw_aligned_trace_calls,
        stark::gpu_lde::schedule_memw_aligned_trace_fault,
    );
}

#[test]
#[ignore = "requires GPU + test-cuda-faults; run with --ignored --nocapture"]
fn gpu_memw_register_trace_fault_falls_back_to_cpu() {
    fallback_for(
        "all_instructions_64",
        "memw_register",
        gpu_memw_register_trace_calls,
        stark::gpu_lde::schedule_memw_register_trace_fault,
    );
}

#[test]
#[ignore = "requires GPU + test-cuda-faults; run with --ignored --nocapture"]
fn gpu_lt_trace_fault_falls_back_to_cpu() {
    fallback_for(
        "all_instructions_64",
        "lt",
        gpu_lt_trace_calls,
        stark::gpu_lde::schedule_lt_trace_fault,
    );
}

#[test]
#[ignore = "requires GPU + test-cuda-faults; run with --ignored --nocapture"]
fn gpu_mul_trace_fault_falls_back_to_cpu() {
    fallback_for(
        "all_instructions_64",
        "mul",
        gpu_mul_trace_calls,
        stark::gpu_lde::schedule_mul_trace_fault,
    );
}

#[test]
#[ignore = "requires GPU + test-cuda-faults; run with --ignored --nocapture"]
fn gpu_cpu_trace_fault_falls_back_to_cpu() {
    fallback_for(
        "all_instructions_64",
        "cpu",
        gpu_cpu_trace_calls,
        stark::gpu_lde::schedule_cpu_trace_fault,
    );
}

#[test]
#[ignore = "requires GPU + test-cuda-faults; run with --ignored --nocapture"]
fn gpu_branch_trace_fault_falls_back_to_cpu() {
    fallback_for(
        "all_instructions_64",
        "branch",
        gpu_branch_trace_calls,
        stark::gpu_lde::schedule_branch_trace_fault,
    );
}

#[test]
#[ignore = "requires GPU + test-cuda-faults; run with --ignored --nocapture"]
fn gpu_cpu32_trace_fault_falls_back_to_cpu() {
    fallback_for(
        "all_instructions_64",
        "cpu32",
        gpu_cpu32_trace_calls,
        stark::gpu_lde::schedule_cpu32_trace_fault,
    );
}

#[test]
#[ignore = "requires GPU + test-cuda-faults; run with --ignored --nocapture"]
fn gpu_dvrm_trace_fault_falls_back_to_cpu() {
    fallback_for(
        "all_instructions_64",
        "dvrm",
        gpu_dvrm_trace_calls,
        stark::gpu_lde::schedule_dvrm_trace_fault,
    );
}

#[test]
#[ignore = "requires GPU + test-cuda-faults; run with --ignored --nocapture"]
fn gpu_memw_trace_fault_falls_back_to_cpu() {
    fallback_for(
        "all_instructions_64",
        "memw",
        gpu_memw_trace_calls,
        stark::gpu_lde::schedule_memw_trace_fault,
    );
}

#[test]
#[ignore = "requires GPU + test-cuda-faults; run with --ignored --nocapture"]
fn gpu_keccak_trace_fault_falls_back_to_cpu() {
    // Keccak only fires when the ECALL=Keccak syscall is invoked. `test_keccak`
    // calls keccak_permute once, so the counter starts at 1 and the fault
    // drops it to 0.
    fallback_for(
        "test_keccak",
        "keccak",
        gpu_keccak_trace_calls,
        stark::gpu_lde::schedule_keccak_trace_fault,
    );
}
