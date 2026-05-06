//! Single-prove bench for profiling with nsys / ncu.
use lambda_vm_prover::test_utils::asm_elf_bytes;

#[test]
#[ignore = "bench; run with --ignored --nocapture"]
fn prove_fib_1m_once() {
    let elf = asm_elf_bytes("fib_iterative_1M");
    // Warm-up pays one-time costs (PTX load, pool warm-up).
    let _ = lambda_vm_prover::prove(&elf).expect("warm-up");
    // The profiled run:
    let _ = lambda_vm_prover::prove(&elf).expect("prove");
}
