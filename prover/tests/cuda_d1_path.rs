//! End-to-end coverage for the num_parts==1 (DECODE) device DEEP/FRI path.
//!
//! The steady-state fixtures in `cuda_path_integration.rs` only exercise the
//! degree-2 composition path: every asm fixture is a `fib_iterative_*` whose
//! DECODE ROM sits below the default GPU LDE threshold, so no num_parts==1 table
//! ever engages the device path there. This binary lowers
//! `LAMBDA_VM_GPU_LDE_THRESHOLD` (via `make test-cuda-d1`) so DECODE crosses it
//! and the whole d=1 wiring — de-interleave -> R2 commit -> R3 OOD -> R4 DEEP ->
//! FRI -> openings — runs end to end, validated by the release query-0
//! composition canary and the final verify.
//!
//! Its own binary (not another test in `cuda_path_integration.rs`) on purpose:
//! `gpu_lde_threshold()` caches the env in a `OnceLock` on first read, so the
//! lowered value must be the one the process sees before any prove — which only
//! holds if this is the sole test in the process.
//!
//! `#[ignore]`'d so the no-GPU CI path skips it. Single test thread: the dispatch
//! counters it asserts on are process-global.
#![cfg(feature = "cuda")]

use lambda_vm_prover::test_utils::asm_elf_bytes;
use lambda_vm_prover::{prove, verify};
use stark::gpu_lde::{gpu_comp_h_slabs_calls, reset_all_gpu_call_counters};

/// With the LDE threshold lowered so the DECODE (num_parts==1) table engages,
/// the device de-interleave path (`gpu_comp_h_slabs_calls`) must fire and the
/// proof — whose DECODE DEEP/FRI now ran on device — must still verify. Guards a
/// silent CPU fallback (counter == 0) and a bad-layout regression (fires but the
/// proof fails verification); the in-prove release query-0 canary guards the
/// composition-row gather on top.
#[test]
#[ignore = "requires GPU + a lowered LAMBDA_VM_GPU_LDE_THRESHOLD; run via `make test-cuda-d1`"]
fn gpu_num_parts_1_decode_path_fires_and_verifies() {
    // Meaningful only with the threshold lowered (the make target sets it). Fail
    // with a pointed message rather than a confusing "path did not fire".
    let thr: usize = std::env::var("LAMBDA_VM_GPU_LDE_THRESHOLD")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    assert!(
        thr > 0 && thr < 1 << 14,
        "run via `make test-cuda-d1`: needs LAMBDA_VM_GPU_LDE_THRESHOLD set below the \
         default (1<<14) so the DECODE (num_parts==1) table engages the device path; got {thr}"
    );

    let elf = asm_elf_bytes("fib_iterative_1M");
    // Warm-up amortises PTX load + pool warm-up so the measured prove reflects
    // steady state (mirrors cuda_path_integration.rs).
    let _ = prove(&elf).expect("warm-up prove");
    reset_all_gpu_call_counters();

    let proof = prove(&elf).expect("prove");

    assert!(
        gpu_comp_h_slabs_calls() > 0,
        "num_parts==1 device de-interleave path did not fire: no DECODE-shaped table \
         crossed the lowered LDE threshold, so the d=1 DEEP/FRI wiring was not exercised"
    );
    assert!(
        verify(&proof, &elf).expect("verify"),
        "num_parts==1 device DEEP/FRI proof failed verification"
    );
}
