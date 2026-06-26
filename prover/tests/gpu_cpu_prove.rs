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
use lambda_vm_prover::tables::bitwise::{
    self, BitwiseOperation, BitwiseOperationType, update_multiplicities,
};
use lambda_vm_prover::tables::commit::{self, CommitOperation};
use lambda_vm_prover::tables::cpu32::{self, Cpu32Operation};
use lambda_vm_prover::tables::gpu_trace::{
    gpu_build_bitwise_table, gpu_build_commit_table, gpu_build_cpu32_trace_tables,
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

/// Byte-parity + timing: the GPU BITWISE table (fixed cols + multiplicity
/// histogram) must equal the CPU `generate_bitwise_trace` + `update_multiplicities`
/// on a 12.6M-op set (ethrex scale), and reports the wall-time of each.
#[test]
#[ignore = "requires GPU; run with --ignored --nocapture"]
fn gpu_bitwise_matches_cpu_and_timing() {
    use std::time::Instant;

    // Deterministic ~12.6M lookups spread over many (x, y, z, type) bins.
    let types = [
        BitwiseOperationType::Msb8,
        BitwiseOperationType::Msb16,
        BitwiseOperationType::Zero,
        BitwiseOperationType::AreBytes,
        BitwiseOperationType::IsHalf,
        BitwiseOperationType::IsB20,
        BitwiseOperationType::Hwsl,
        BitwiseOperationType::ByteAluAnd,
        BitwiseOperationType::ByteAluOr,
        BitwiseOperationType::ByteAluXor,
    ];
    let n = 12_600_000usize;
    let mut ops = Vec::with_capacity(n);
    let mut s = 0x1234_5678u64;
    for _ in 0..n {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let x = (s >> 33) as u8;
        let y = (s >> 41) as u8;
        let z = ((s >> 49) & 0xF) as u8;
        let t = types[((s >> 53) as usize) % types.len()];
        ops.push(BitwiseOperation::new(t, x, y, z));
    }

    let t0 = Instant::now();
    let mut cpu = bitwise::generate_bitwise_trace();
    update_multiplicities(&mut cpu, &ops);
    let cpu_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let t1 = Instant::now();
    let gpu = gpu_build_bitwise_table(&ops).expect("gpu bitwise");
    let gpu_ms = t1.elapsed().as_secs_f64() * 1000.0;

    let cpu_cols = cpu.columns_main();
    let gpu_cols = gpu.columns_main();
    assert_eq!(cpu_cols.len(), gpu_cols.len(), "column count");
    for (c, (cc, gc)) in cpu_cols.iter().zip(gpu_cols.iter()).enumerate() {
        assert_eq!(cc, gc, "BITWISE column {c} differs (GPU vs CPU)");
    }
    println!(
        "BITWISE {n} ops: CPU {cpu_ms:.1}ms vs GPU {gpu_ms:.1}ms ({:.1}x), byte-identical",
        cpu_ms / gpu_ms
    );
}

/// Byte-parity: the GPU COMMIT table must equal the CPU `generate_commit_trace`
/// (COMMIT only appears in ethrex blocks, so the prove tests above don't
/// exercise it — validate directly on synthetic ops, including padding).
#[test]
#[ignore = "requires GPU; run with --ignored --nocapture"]
fn gpu_commit_matches_cpu() {
    // A few commit sequences with first/middle/end rows + varied counts.
    let mut ops = Vec::new();
    for k in 0..50u64 {
        let count = 5 - (k % 5); // 5,4,3,2,1 then repeats
        ops.push(CommitOperation {
            timestamp: 100 + k * 4,
            index: k,
            address: 0x4000 + k,
            count,
            first: k % 5 == 0,
            end: count == 0,
            value: (k & 0xFF) as u8,
        });
    }
    // An explicit end row (count == 0).
    ops.push(CommitOperation {
        timestamp: 999,
        index: 50,
        address: 0x4100,
        count: 0,
        first: false,
        end: true,
        value: 0,
    });

    let cpu = commit::generate_commit_trace(&ops);
    let gpu = gpu_build_commit_table(&ops).expect("gpu commit");
    let cpu_cols = cpu.columns_main();
    let gpu_cols = gpu.columns_main();
    assert_eq!(cpu_cols.len(), gpu_cols.len(), "column count");
    for (c, (cc, gc)) in cpu_cols.iter().zip(gpu_cols.iter()).enumerate() {
        assert_eq!(cc, gc, "COMMIT column {c} differs (GPU vs CPU)");
    }
    println!("COMMIT {} ops: GPU byte-identical to CPU", ops.len());
}

/// Byte-parity: the GPU CPU32 table must equal the CPU `generate_cpu32_trace`
/// on synthetic *W ops spanning signed/unsigned, rv2-vs-imm, and sign-extension
/// cases. (CPU32 may not appear in the prove programs above; validate directly.)
#[test]
#[ignore = "requires GPU; run with --ignored --nocapture"]
fn gpu_cpu32_matches_cpu() {
    let mut ops = Vec::new();
    let mut s = 0xC0FFEEu64;
    for k in 0..300u64 {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let signed = (k % 2) == 0;
        let use_imm = (k % 3) == 0;
        ops.push(Cpu32Operation {
            timestamp: 4 + k * 4,
            pc: 0x1000 + k * 2,
            rs1: (k % 32) as u8,
            read_register1: true,
            rv1: s ^ (k << 17),
            rs2: ((k + 7) % 32) as u8,
            read_register2: !use_imm,
            rv2: if use_imm { 0 } else { (s >> 7) & 0xFFFF_FFFF },
            imm: if use_imm { (s >> 11) & 0xFFFF_FFFF } else { 0 },
            res: s.rotate_left(13),
            rd: ((k + 3) % 32) as u8,
            write_register: true,
            alu: true,
            alu_flags: if signed { 1 << 5 } else { 0 },
            add: (k % 4) == 0,
            sub: (k % 4) == 1,
            half_instruction_length: 2,
        });
    }

    let cpu_tables = {
        // Single chunk: mirror chunk_and_generate with a large max_rows.
        vec![cpu32::generate_cpu32_trace(&ops)]
    };
    let gpu_tables = gpu_build_cpu32_trace_tables(&ops, 1 << 20).expect("gpu cpu32");
    assert_eq!(cpu_tables.len(), gpu_tables.len(), "table count");
    for (cpu, gpu) in cpu_tables.iter().zip(gpu_tables.iter()) {
        let cc = cpu.columns_main();
        let gc = gpu.columns_main();
        assert_eq!(cc.len(), gc.len(), "column count");
        for (c, (a, b)) in cc.iter().zip(gc.iter()).enumerate() {
            assert_eq!(a, b, "CPU32 column {c} differs (GPU vs CPU)");
        }
    }
    println!("CPU32 {} ops: GPU byte-identical to CPU", ops.len());
}
