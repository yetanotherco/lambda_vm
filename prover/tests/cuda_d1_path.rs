//! End-to-end coverage for the num_parts==1 (DECODE) device DEEP/FRI path.
//!
//! No fixture crosses the default GPU LDE threshold for a num_parts==1 table, so
//! this binary lowers `LAMBDA_VM_GPU_LDE_THRESHOLD` (via `make test-cuda-d1`)
//! until DECODE engages and the whole d=1 wiring — de-interleave -> R2 commit ->
//! R3 OOD -> R4 DEEP -> FRI -> openings — runs end to end, validated by the
//! release query-0 composition canary and the final verify.
//!
//! Fixture and threshold are one choice. There are exactly two d=1 tables (a d=1
//! table is one with a single bus interaction): DECODE, sized from the guest's
//! instruction count, and KECCAK_RC, fixed at `NUM_ROWS = 32` => LDE 64. DECODE's
//! ROM comes from the ELF and not from cycles, so every `fib_iterative_*` variant
//! is 13 executable words => 16 rows => LDE 32 — below KECCAK_RC's 64, which means
//! no threshold isolates DECODE with a fib fixture. `all_instructions_64` is 66
//! executable words => 128 rows => LDE 256, so at threshold 128 DECODE engages and
//! KECCAK_RC declines: a nonzero counter uniquely attributes to DECODE.
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

/// The fixture whose DECODE ROM crosses the lowered threshold: 66 executable
/// words -> 128 rows -> LDE 256.
const FIXTURE: &str = "all_instructions_64";
/// DECODE's LDE for [`FIXTURE`], and the LDE of the only other d=1 table. The
/// threshold must fall between them so the counter attributes to DECODE alone.
const DECODE_LDE: usize = 256;
const KECCAK_RC_LDE: usize = 64;

/// With the LDE threshold lowered so the DECODE (num_parts==1) table engages, the
/// device de-interleave path (`gpu_comp_h_slabs_calls`) must fire and the proof —
/// whose DECODE DEEP/FRI now ran on device — must still verify. Guards a silent
/// CPU fallback (counter == 0) and a bad-layout regression (fires but the proof
/// fails verification); the in-prove release query-0 canary guards the
/// composition-row gather on top.
#[test]
#[ignore = "requires GPU + a lowered LAMBDA_VM_GPU_LDE_THRESHOLD; run via `make test-cuda-d1`"]
fn gpu_num_parts_1_decode_path_fires_and_verifies() {
    // Pin the window rather than just "below the default": a threshold anywhere
    // outside (KECCAK_RC_LDE, DECODE_LDE] silently measures the wrong table (or no
    // table), which is exactly the failure this constant pair exists to prevent.
    let thr: usize = std::env::var("LAMBDA_VM_GPU_LDE_THRESHOLD")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    assert!(
        thr > KECCAK_RC_LDE && thr <= DECODE_LDE,
        "run via `make test-cuda-d1`: LAMBDA_VM_GPU_LDE_THRESHOLD must land in \
         ({KECCAK_RC_LDE}, {DECODE_LDE}] so {FIXTURE}'s DECODE (LDE {DECODE_LDE}) engages the \
         device path while KECCAK_RC (LDE {KECCAK_RC_LDE}) declines; got {thr}"
    );

    let elf = asm_elf_bytes(FIXTURE);
    // Warm-up amortises PTX load + pool warm-up so the measured prove reflects
    // steady state (mirrors cuda_path_integration.rs).
    let _ = prove(&elf).expect("warm-up prove");
    reset_all_gpu_call_counters();

    let proof = prove(&elf).expect("prove");

    assert!(
        gpu_comp_h_slabs_calls() > 0,
        "num_parts==1 device de-interleave path did not fire: DECODE (the only d=1 table \
         above the threshold for {FIXTURE}) did not take it, so the d=1 DEEP/FRI wiring \
         was not exercised"
    );
    assert!(
        verify(&proof, &elf).expect("verify"),
        "num_parts==1 device DEEP/FRI proof failed verification"
    );
}
