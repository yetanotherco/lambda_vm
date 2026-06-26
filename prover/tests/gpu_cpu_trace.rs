//! Parity: the GPU CPU-table kernel (`math_cuda::trace::gpu_build_cpu_trace`)
//! vs the CPU reference (`generate_cpu_trace_from_logs`). Asserts byte-identical
//! column-major output across several asm programs chosen for opcode coverage
//! (loads/stores, word instructions, branches, ALU, ecall).
//!
//! `#[ignore]`'d so the no-GPU path skips it. Run with:
//!   cargo test -p lambda-vm-prover --release --features cuda \
//!       --test gpu_cpu_trace -- --ignored --nocapture
#![cfg(feature = "cuda")]

use lambda_vm_prover::tables::cpu::{cols, generate_cpu_trace_from_logs};
use lambda_vm_prover::tables::types::DecodeEntry;
use lambda_vm_prover::test_utils::run_asm_elf;
use math_cuda::trace::{
    self, DEC_ALU_FLAGS, DEC_FLAGS, DEC_HIL, DEC_IMM, DEC_MEM_FLAGS, DEC_RD, DEC_RS1, DEC_RS2,
    DEC_STRIDE, F_ADD, F_ALU, F_BRANCH, F_ECALL, F_MEMORY, F_READ_REGISTER1, F_READ_REGISTER2,
    F_SUB, F_WORD_INSTR, F_WRITE_REGISTER,
};

fn check(program: &str) {
    let (_elf, logs, instructions) = run_asm_elf(program);
    let n = logs.len();
    assert!(n > 0, "{program}: produced no logs");

    // CPU reference table (one un-chunked table of n.next_power_of_two() rows).
    let cpu_trace = generate_cpu_trace_from_logs(&logs, &instructions)
        .unwrap_or_else(|e| panic!("{program}: cpu trace build failed: {e:?}"));
    let nrows = n.next_power_of_two().max(4);

    // PackedDecode dense array indexed by (pc - base) >> 1 (RISC-V is 2-byte
    // aligned). Built by calling the SAME DecodeEntry::from_instruction the CPU
    // path uses, so the per-PC fields are guaranteed to match (no re-derivation).
    let base = *instructions.keys().min().unwrap();
    let maxpc = *instructions.keys().max().unwrap();
    let slots = (((maxpc - base) >> 1) + 1) as usize;
    let mut decode = vec![0u64; slots * DEC_STRIDE];
    for (&pc, &instr) in instructions.iter() {
        let de = DecodeEntry::from_instruction(pc, instr, 4);
        let f = de.fields;
        let o = (((pc - base) >> 1) as usize) * DEC_STRIDE;
        let mut flags = 0u64;
        flags |= (f.read_register1 as u64) << F_READ_REGISTER1;
        flags |= (f.read_register2 as u64) << F_READ_REGISTER2;
        flags |= (f.write_register as u64) << F_WRITE_REGISTER;
        flags |= (f.word_instr as u64) << F_WORD_INSTR;
        flags |= (f.alu as u64) << F_ALU;
        flags |= (f.add as u64) << F_ADD;
        flags |= (f.sub as u64) << F_SUB;
        flags |= (f.memory as u64) << F_MEMORY;
        flags |= (f.branch as u64) << F_BRANCH;
        flags |= (f.ecall as u64) << F_ECALL;
        decode[o + DEC_FLAGS] = flags;
        decode[o + DEC_RS1] = f.rs1 as u64;
        decode[o + DEC_RS2] = f.rs2 as u64;
        decode[o + DEC_RD] = f.rd as u64;
        decode[o + DEC_HIL] = f.half_instruction_length as u64;
        decode[o + DEC_ALU_FLAGS] = f.alu_flags as u64;
        decode[o + DEC_MEM_FLAGS] = f.mem_flags as u64;
        decode[o + DEC_IMM] = de.imm;
    }

    // Flatten logs: [current_pc, next_pc, src1_val, src2_val, dst_val] per row.
    let mut logs_flat = Vec::with_capacity(5 * n);
    for l in &logs {
        logs_flat.extend_from_slice(&[
            l.current_pc,
            l.next_pc,
            l.src1_val,
            l.src2_val,
            l.dst_val,
        ]);
    }

    // Single un-chunked table → row_offset = 0.
    let dev = trace::gpu_build_cpu_trace(&logs_flat, &decode, base, 0, n, nrows)
        .unwrap_or_else(|e| panic!("{program}: gpu build failed: {e:?}"));
    assert_eq!(dev.ncols, cols::NUM_COLUMNS, "{program}: ncols");
    assert_eq!(dev.nrows, nrows, "{program}: nrows");
    let gpu = dev.to_host().unwrap();

    // Byte-compare every column-major cell against the CPU table.
    for c in 0..cols::NUM_COLUMNS {
        for r in 0..nrows {
            let cpu_v = *cpu_trace.get_main(r, c).value();
            let gpu_v = gpu[c * nrows + r];
            assert_eq!(
                cpu_v, gpu_v,
                "{program}: mismatch at col {c} row {r} (cpu={cpu_v} gpu={gpu_v})"
            );
        }
    }
    println!("{program}: OK — {n} ops, {nrows} rows, {} cols", cols::NUM_COLUMNS);
}

#[test]
#[ignore = "requires GPU; run with --ignored --nocapture"]
fn gpu_cpu_trace_matches_cpu() {
    for program in [
        "all_instructions_64",
        "comprehensive_test",
        "all_loadstore_32",
        "all_branches_16",
        "basic_arith_32",
        "fib_iterative_160k",
    ] {
        check(program);
    }
}
