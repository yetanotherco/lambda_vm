//! End-to-end validation of the p5 single-build resident seam: a FULL prove+verify of the bench
//! ELF through production `build_traces`. With `LAMBDA_VM_GPU_RESIDENT_CHIPS=1` this exercises the
//! shared-devops path — the cpu_op field SoA is uploaded to the device ONCE and every chip table
//! (CPU32/LOAD/STORE/SHIFT/EQ/BYTEWISE/MUL/DVRM/BRANCH) is built from those resident buffers in
//! place across the parallel p5 rayon scope. The proof must still verify, proving the single-build
//! integration (shared devops lifetime + concurrent read + table slotting) is correct.
//!
//! `LAMBDA_VM_BENCH_ELF=.../rust/ethrex.elf LAMBDA_VM_BENCH_INPUT=.../ethrex_5tx.bin \
//!   LAMBDA_VM_GPU_RESIDENT_CHIPS=1 \
//!   cargo test -p lambda-vm-prover --release --features cuda --lib gpu_resident_e2e -- --ignored --nocapture`

use std::env;
use std::fs;

#[test]
#[ignore = "requires GPU + LAMBDA_VM_BENCH_ELF (ethrex); run --ignored --nocapture"]
fn gpu_resident_e2e_prove_verify() {
    let path = env::var("LAMBDA_VM_BENCH_ELF").expect("set LAMBDA_VM_BENCH_ELF (ethrex.elf)");
    let elf_bytes = fs::read(&path).expect("read ELF");
    let input = env::var("LAMBDA_VM_BENCH_INPUT")
        .ok()
        .map(|p| fs::read(p).expect("read input"))
        .unwrap_or_default();
    let resident = env::var("LAMBDA_VM_GPU_RESIDENT_CHIPS").is_ok_and(|v| v == "1");
    println!("gpu_resident_e2e: resident-chips single-build seam = {resident}");

    let proof = crate::prove_with_inputs(&elf_bytes, &input).expect("prove");
    assert!(
        crate::verify(&proof, &elf_bytes).expect("verify"),
        "proof must verify (resident={resident})"
    );
    println!("gpu_resident_e2e OK: full prove+verify passed (resident single-build={resident})");
}
