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
    gpu_load_table_builds, gpu_lt_table_builds, gpu_memw_aligned_table_builds,
    gpu_memw_register_table_builds, gpu_memw_table_builds, gpu_mul_table_builds,
    gpu_shift_table_builds, gpu_store_table_builds,
};
use lambda_vm_prover::test_utils::asm_elf_bytes;
use lambda_vm_prover::{prove, verify};
use stark::gpu_lde::gpu_lde_from_device_calls;

fn prove_verify(program: &str) {
    let elf = asm_elf_bytes(program);

    let builds_before = gpu_cpu_table_builds();
    let lde_dev_before = gpu_lde_from_device_calls();
    // The fib programs loop on BNE → tens of thousands of EQ ops, so this is
    // where the GPU EQ table is actually exercised (all_instructions_64 has no
    // BEQ/BNE, hence no EQ ops — see gpu_alu_tables_prove_verify).
    let eq_before = gpu_eq_table_builds();
    let proof = prove(&elf).unwrap_or_else(|e| panic!("{program}: prove failed: {e:?}"));

    assert!(
        gpu_cpu_table_builds() > builds_before,
        "{program}: GPU CPU-table build did not fire (silent CPU fallback)"
    );
    assert!(
        gpu_eq_table_builds() > eq_before,
        "{program}: GPU EQ table did not fire"
    );
    assert!(
        gpu_lde_from_device_calls() > lde_dev_before,
        "{program}: device-resident LDE did not fire (fell back to host-input LDE)"
    );
    assert!(
        verify(&proof, &elf).expect("verify"),
        "{program}: proof built with the GPU CPU table + device-resident LDE failed to verify"
    );
    println!("{program}: prove+verify OK with GPU CPU + EQ tables + device-resident LDE");
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
    // all_instructions_64 exercises LT (SLT/BLT/BGE), BYTEWISE (AND/OR/XOR),
    // SHIFT, MUL and DVRM → those GPU ALU tables fire. It has no BEQ/BNE, so it
    // produces no EQ ops; the EQ table is covered by gpu_cpu_table_prove_verify
    // (fib loops on BNE → 74k+ EQ ops).
    let elf = asm_elf_bytes("all_instructions_64");
    let (lt0, bw0, sh0, mul0, dv0, mr0) = (
        gpu_lt_table_builds(),
        gpu_bytewise_table_builds(),
        gpu_shift_table_builds(),
        gpu_mul_table_builds(),
        gpu_dvrm_table_builds(),
        gpu_memw_register_table_builds(),
    );
    let proof = prove(&elf).expect("prove");
    assert!(
        gpu_memw_register_table_builds() > mr0,
        "GPU MEMW_R table did not fire"
    );
    assert!(gpu_lt_table_builds() > lt0, "GPU LT table did not fire");
    assert!(
        gpu_bytewise_table_builds() > bw0,
        "GPU BYTEWISE table did not fire"
    );
    assert!(gpu_shift_table_builds() > sh0, "GPU SHIFT table did not fire");
    assert!(gpu_mul_table_builds() > mul0, "GPU MUL table did not fire");
    assert!(gpu_dvrm_table_builds() > dv0, "GPU DVRM table did not fire");
    assert!(
        verify(&proof, &elf).expect("verify"),
        "proof built with GPU LT/BYTEWISE/SHIFT/MUL/DVRM tables failed to verify"
    );
    println!(
        "all_instructions_64: prove+verify OK with GPU LT+BYTEWISE+SHIFT+MUL+DVRM tables"
    );
}

#[test]
#[ignore = "requires GPU; run with --ignored --nocapture"]
fn gpu_memory_tables_prove_verify() {
    // all_loadstore_32 exercises register accesses (MEMW_R), aligned and
    // general/unaligned memory writes (MEMW_A / MEMW), and load/store of every
    // width (LOAD / STORE) → all five GPU memory tables fire. If any GPU-built
    // memory table diverged from the CPU builder, the memory-argument bus would
    // unbalance and verification would fail.
    let elf = asm_elf_bytes("all_loadstore_32");
    let (mr0, ma0, mw0, ld0, st0) = (
        gpu_memw_register_table_builds(),
        gpu_memw_aligned_table_builds(),
        gpu_memw_table_builds(),
        gpu_load_table_builds(),
        gpu_store_table_builds(),
    );
    let proof = prove(&elf).expect("prove");
    assert!(
        gpu_memw_register_table_builds() > mr0,
        "GPU MEMW_R table did not fire"
    );
    assert!(
        gpu_memw_aligned_table_builds() > ma0,
        "GPU MEMW_A table did not fire"
    );
    assert!(gpu_memw_table_builds() > mw0, "GPU MEMW table did not fire");
    assert!(gpu_load_table_builds() > ld0, "GPU LOAD table did not fire");
    assert!(
        gpu_store_table_builds() > st0,
        "GPU STORE table did not fire"
    );
    assert!(
        verify(&proof, &elf).expect("verify"),
        "proof built with GPU MEMW_R/MEMW_A/MEMW/LOAD/STORE tables failed to verify"
    );
    println!(
        "all_loadstore_32: prove+verify OK with GPU MEMW_R+MEMW_A+MEMW+LOAD+STORE tables"
    );
}
