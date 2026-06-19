//! Single-prove bench for profiling with nsys / ncu.
//!
//! Runs one warm-up + one profiled `lambda_vm_prover::prove` of
//! `fib_iterative_1M`, prints the profiled-pass wall time, and (under
//! `cuda`) asserts the GPU LDE path actually fired so silent CPU-fallback
//! regressions are caught.
//!
//! Run via `make bench-prover` or
//! `cargo test -p lambda-vm-prover --release --test bench_single -- --ignored --nocapture`.

use std::time::Instant;

use lambda_vm_prover::prove;
use lambda_vm_prover::test_utils::asm_elf_bytes;
#[cfg(feature = "cuda")]
use stark::gpu_lde::{gpu_lde_calls, reset_all_gpu_call_counters};

#[test]
#[ignore = "bench; run with --ignored --nocapture"]
fn prove_fib_1m_once() {
    let elf = asm_elf_bytes("fib_iterative_1M");

    // Warm-up pays one-time costs (PTX load, pool warm-up).
    let _ = prove(&elf).expect("warm-up");

    // Reset GPU counters so the profiled-pass assert below reflects only the
    // second run, not warm-up + profiled combined.
    #[cfg(feature = "cuda")]
    reset_all_gpu_call_counters();

    // The profiled run.
    let start = Instant::now();
    let _ = prove(&elf).expect("prove");
    let elapsed = start.elapsed();
    println!("bench: prove(fib_iterative_1M) = {:?}", elapsed);

    // Catch silent regressions where the table sizes drop below the GPU LDE
    // threshold and we'd be measuring CPU numbers without noticing.
    #[cfg(feature = "cuda")]
    assert!(
        gpu_lde_calls() > 0,
        "GPU LDE path did not fire: fib_iterative_1M may have dropped below the GPU threshold"
    );
}
