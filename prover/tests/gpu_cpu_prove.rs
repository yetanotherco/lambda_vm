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

use lambda_vm_prover::tables::gpu_trace::{
    gpu_bytewise_table_builds, gpu_cpu_table_builds, gpu_dvrm_table_builds, gpu_eq_table_builds,
    gpu_lt_table_builds, gpu_mul_table_builds, gpu_shift_table_builds,
};
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
fn gpu_alu_tables_prove_verify() {
    // all_instructions_64 exercises LT (SLT/BLT/BGE), EQ (BEQ/BNE/SEQ), and
    // BYTEWISE (AND/OR/XOR) → all three GPU ALU tables fire.
    let elf = asm_elf_bytes("all_instructions_64");
    let (lt0, eq0, bw0, sh0, mul0, dv0) = (
        gpu_lt_table_builds(),
        gpu_eq_table_builds(),
        gpu_bytewise_table_builds(),
        gpu_shift_table_builds(),
        gpu_mul_table_builds(),
        gpu_dvrm_table_builds(),
    );
    let proof = prove(&elf).expect("prove");
    assert!(gpu_lt_table_builds() > lt0, "GPU LT table did not fire");
    assert!(gpu_eq_table_builds() > eq0, "GPU EQ table did not fire");
    assert!(
        gpu_bytewise_table_builds() > bw0,
        "GPU BYTEWISE table did not fire"
    );
    assert!(gpu_shift_table_builds() > sh0, "GPU SHIFT table did not fire");
    assert!(gpu_mul_table_builds() > mul0, "GPU MUL table did not fire");
    assert!(gpu_dvrm_table_builds() > dv0, "GPU DVRM table did not fire");
    assert!(
        verify(&proof, &elf).expect("verify"),
        "proof built with GPU LT/EQ/BYTEWISE/SHIFT/MUL/DVRM tables failed to verify"
    );
    println!(
        "all_instructions_64: prove+verify OK with GPU LT+EQ+BYTEWISE+SHIFT+MUL+DVRM tables"
    );
}
