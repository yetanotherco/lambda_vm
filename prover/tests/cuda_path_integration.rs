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
    gpu_bary_calls, gpu_batch_invert_calls, gpu_bitwise_trace_calls, gpu_branch_trace_calls,
    gpu_bytewise_trace_calls, gpu_comp_poly_tree_calls, gpu_cpu32_trace_calls, gpu_cpu_trace_calls,
    gpu_decode_trace_calls, gpu_deep_calls, gpu_dvrm_trace_calls, gpu_fri_calls,
    gpu_keccak_trace_calls, gpu_lde_calls, gpu_load_trace_calls, gpu_lt_trace_calls,
    gpu_memw_aligned_trace_calls, gpu_memw_register_trace_calls, gpu_memw_trace_calls,
    gpu_mul_trace_calls, gpu_page_trace_calls, gpu_parts_lde_calls, gpu_shift_trace_calls,
    gpu_store_trace_calls, reset_all_gpu_call_counters,
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

    // PR-6 trace-expansion ports: PAGE and DECODE main-column generation.
    // PAGE fires once per memory page in the program (always >= 1).
    // DECODE fires once per prove (single table build).
    assert!(
        gpu_page_trace_calls() > 0,
        "PAGE trace expansion did not fire on GPU"
    );
    assert!(
        gpu_decode_trace_calls() > 0,
        "DECODE trace expansion did not fire on GPU"
    );

    // Counters only prove the dispatches ran; this checks the GPU proof
    // actually satisfies the verifier.
    let ok = verify(&proof, &elf).expect("verify");
    assert!(ok, "GPU-produced proof failed verification");
}

/// Every PR-6 trace-expansion port fires on a comprehensive ELF.
///
/// `all_instructions_64` exercises every non-syscall RISC-V instruction
/// family, so every ported table that isn't gated on a specific syscall
/// (keccak/ecsm) should fire at least once during the prove. A counter
/// stuck at zero means either:
///   - the prover dispatch wiring is wrong (helper returned None without
///     ever calling `try_generate_*_trace_gpu_raw`), or
///   - the kernel always errored and silently fell back to CPU.
///
/// Syscall tables (KECCAK, ECSM) have their own dedicated tests below or
/// in `cuda_fallback_tests.rs`.
#[test]
#[ignore = "requires GPU; run with --ignored --nocapture"]
fn gpu_all_table_ports_fire() {
    let elf = asm_elf_bytes("all_instructions_64");
    // Warm-up amortises PTX load + pool warm-up so the profiled-pass
    // counter assertions reflect steady-state work.
    let _ = prove(&elf).expect("warm-up prove");

    reset_all_gpu_call_counters();
    let proof = prove(&elf).expect("prove");

    // Per-port assertions: each `gpu_X_trace_calls() > 0` proves the
    // prover-side fast path was reached and math-cuda's kernel returned
    // Ok at least once.
    macro_rules! assert_fired {
        ($name:expr, $count_fn:ident) => {
            assert!(
                $count_fn() > 0,
                "GPU {} trace port did not fire on all_instructions_64",
                $name,
            );
        };
    }
    assert_fired!("page", gpu_page_trace_calls);
    assert_fired!("decode", gpu_decode_trace_calls);
    assert_fired!("bitwise", gpu_bitwise_trace_calls);
    assert_fired!("load", gpu_load_trace_calls);
    assert_fired!("store", gpu_store_trace_calls);
    assert_fired!("bytewise", gpu_bytewise_trace_calls);
    assert_fired!("shift", gpu_shift_trace_calls);
    assert_fired!("memw_aligned", gpu_memw_aligned_trace_calls);
    assert_fired!("memw_register", gpu_memw_register_trace_calls);
    assert_fired!("lt", gpu_lt_trace_calls);
    assert_fired!("mul", gpu_mul_trace_calls);
    assert_fired!("cpu", gpu_cpu_trace_calls);
    assert_fired!("branch", gpu_branch_trace_calls);
    assert_fired!("cpu32", gpu_cpu32_trace_calls);
    assert_fired!("dvrm", gpu_dvrm_trace_calls);
    assert_fired!("memw", gpu_memw_trace_calls);

    let ok = verify(&proof, &elf).expect("verify");
    assert!(ok, "GPU-produced proof failed verification");
}

/// KECCAK syscall trace port fires when invoked from a program. Separate
/// test because `all_instructions_64` doesn't issue keccak_permute.
#[test]
#[ignore = "requires GPU; run with --ignored --nocapture"]
fn gpu_keccak_port_fires() {
    let elf = asm_elf_bytes("test_keccak");
    let _ = prove(&elf).expect("warm-up prove");
    reset_all_gpu_call_counters();
    let proof = prove(&elf).expect("prove");
    assert!(
        gpu_keccak_trace_calls() > 0,
        "GPU keccak trace port did not fire on test_keccak"
    );
    let ok = verify(&proof, &elf).expect("verify");
    assert!(ok, "GPU keccak proof failed verification");
}
