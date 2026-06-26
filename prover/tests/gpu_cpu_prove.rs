//! End-to-end: prove + verify with the GPU-built CPU trace table.
//!
//! Uses a single-chunk program (≤ 2^19 CPU rows) so the GPU CPU-table path
//! fires, then asserts the proof verifies and that the device path actually ran
//! (not a silent CPU fallback). If the GPU-built table were wrong, the Merkle
//! commitment would diverge and verification would fail.
//!
//! `#[ignore]`'d (needs a GPU). Run with:
//!   cargo test -p lambda-vm-prover --release --features cuda \
//!       --test gpu_cpu_prove -- --ignored --nocapture
#![cfg(feature = "cuda")]

use lambda_vm_prover::tables::gpu_trace::{gpu_cpu_table_builds, gpu_lt_table_builds};
use lambda_vm_prover::test_utils::asm_elf_bytes;
use lambda_vm_prover::{prove, verify};
use stark::gpu_lde::gpu_lde_from_device_calls;

fn prove_verify(program: &str) {
    let elf = asm_elf_bytes(program);

    let builds_before = gpu_cpu_table_builds();
    let lde_dev_before = gpu_lde_from_device_calls();
    let proof = prove(&elf).unwrap_or_else(|e| panic!("{program}: prove failed: {e:?}"));

    assert!(
        gpu_cpu_table_builds() > builds_before,
        "{program}: GPU CPU-table build did not fire (silent CPU fallback)"
    );
    assert!(
        gpu_lde_from_device_calls() > lde_dev_before,
        "{program}: device-resident LDE did not fire (fell back to host-input LDE)"
    );
    assert!(
        verify(&proof, &elf).expect("verify"),
        "{program}: proof built with the GPU CPU table + device-resident LDE failed to verify"
    );
    println!("{program}: prove+verify OK with GPU CPU table + device-resident LDE");
}

#[test]
#[ignore = "requires GPU; run with --ignored --nocapture"]
fn gpu_cpu_table_prove_verify() {
    // Single chunk: 372k CPU ops < 2^19 (524288).
    prove_verify("fib_iterative_372k");
    // Multi-chunk: 1M CPU ops > 2^19 → 2 chunks (exercises row_offset on chunk 1).
    prove_verify("fib_iterative_1M");
}

#[test]
#[ignore = "requires GPU; run with --ignored --nocapture"]
fn gpu_lt_table_prove_verify() {
    // comprehensive_test exercises SLT/SLTU/BLT/BGE → LT ops.
    let elf = asm_elf_bytes("comprehensive_test");
    let lt_before = gpu_lt_table_builds();
    let proof = prove(&elf).expect("prove");
    assert!(
        gpu_lt_table_builds() > lt_before,
        "GPU LT-table path did not fire (no LT ops, or silent CPU fallback)"
    );
    assert!(
        verify(&proof, &elf).expect("verify"),
        "proof built with the GPU LT table failed to verify"
    );
    println!("comprehensive_test: prove+verify OK with GPU LT table");
}
