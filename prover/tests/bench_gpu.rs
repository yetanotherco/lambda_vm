//! End-to-end timing probe: prove `fib_iterative_1M` (≈1M instructions) once
//! and print wall-clock time. Intended to be run twice — once with the `cuda`
//! feature, once without — so the caller can compare. Ignored by default.
//!
//! Usage:
//!   cargo test -p lambda-vm-prover --release --test bench_gpu -- --ignored --nocapture
//!   cargo test -p lambda-vm-prover --release --features cuda --test bench_gpu -- --ignored --nocapture

use std::time::Instant;

use lambda_vm_prover::test_utils::asm_elf_bytes;

fn bench_prove(name: &str, trials: u32) {
    let elf = asm_elf_bytes(name);
    // Warm up — first prove pays lazy one-time costs (PTX load on the GPU side,
    // buffer pool warm-up on the CPU side).
    let _ = lambda_vm_prover::prove(&elf).expect("warm-up prove");

    #[cfg(feature = "cuda")]
    stark::gpu_lde::reset_gpu_lde_calls();

    let t0 = Instant::now();
    for _ in 0..trials {
        let _ = lambda_vm_prover::prove(&elf).expect("prove");
    }
    let elapsed = t0.elapsed().as_secs_f64() / trials as f64;

    let gpu = if cfg!(feature = "cuda") { "gpu" } else { "cpu" };
    println!("prove({name}) [{gpu}]: {elapsed:.3}s avg over {trials} trials");

    #[cfg(feature = "cuda")]
    {
        let calls = stark::gpu_lde::gpu_lde_calls();
        let eh = stark::gpu_lde::gpu_extend_halves_calls();
        let r4 = stark::gpu_lde::gpu_r4_lde_calls();
        let parts = stark::gpu_lde::gpu_parts_lde_calls();
        let leaf = stark::gpu_lde::gpu_leaf_hash_calls();
        println!("  GPU LDE calls across {trials} proves: {calls}");
        println!("  GPU extend_two_halves calls: {eh}");
        println!("  GPU R4 deep-poly LDE calls: {r4}");
        println!("  GPU R2 parts LDE calls: {parts}");
        println!("  GPU leaf-hash calls: {leaf}");
    }
}

#[test]
#[ignore = "bench; run with --ignored --nocapture"]
fn bench_prove_fib_1m() {
    bench_prove("fib_iterative_1M", 5);
}

#[test]
#[ignore = "bench; run with --ignored --nocapture"]
fn bench_prove_fib_2m() {
    bench_prove("fib_iterative_2M", 5);
}

#[test]
#[ignore = "bench; run with --ignored --nocapture"]
fn bench_prove_fib_4m() {
    bench_prove("fib_iterative_4M", 3);
}
