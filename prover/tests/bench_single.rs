//! Single-prove bench for profiling with nsys / ncu.
use lambda_vm_prover::test_utils::asm_elf_bytes;

#[test]
#[ignore = "bench; run with --ignored --nocapture"]
fn prove_fib_1m_once() {
    let elf = asm_elf_bytes("fib_iterative_1M");
    // Warm-up pays one-time costs (PTX load, pool warm-up).
    let _ = lambda_vm_prover::prove(&elf).expect("warm-up");
    // Reset GPU counters so the profiled-pass assert below reflects only the
    // second run, not warm-up + profiled combined.
    #[cfg(feature = "cuda")]
    stark::gpu_lde::reset_all_gpu_call_counters();
    // The profiled run:
    let _ = lambda_vm_prover::prove(&elf).expect("prove");
    // Catch silent regressions where the table sizes drop below the GPU LDE
    // threshold and we'd be measuring CPU numbers without noticing.
    #[cfg(feature = "cuda")]
    assert!(
        stark::gpu_lde::gpu_lde_calls() > 0,
        "GPU LDE path did not fire — fib_iterative_1M may have dropped below the GPU threshold"
    );
}
