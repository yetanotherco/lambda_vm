//! Phase-3c byte-parity: the device CPU32 (word `*W`) chip-op builder
//! ([`math_cuda::trace_ops::gpu_build_cpu32_ops`]) must reproduce, row-for-row in program
//! order, the packed CPU32 rows the host builds via `build_cpu32_op` + `pack_cpu32_op` —
//! including `res`, which the device computes with the ported SHIFT/MUL/DVRM arithmetic
//! (compute_aux + cpu32_res). State-free (pure cpu_op projection), so it validates on the
//! real guest (ethrex_5tx).
//!
//! `LAMBDA_VM_BENCH_ELF=.../rust/ethrex.elf LAMBDA_VM_BENCH_INPUT=.../ethrex_5tx.bin \
//!   cargo test -p lambda-vm-prover --release --features cuda --lib gpu_build_cpu32 -- --ignored --nocapture`

use std::env;
use std::fs;

use executor::elf::Elf;
use executor::vm::execution::Executor;

use crate::tables::cpu::CpuOperation;
use crate::tables::decode;
use crate::tables::gpu_trace::pack_cpu32_op;
use crate::tables::trace_builder::build_cpu32_op;
use crate::tables::types::DecodeEntry;

#[test]
#[ignore = "requires GPU + LAMBDA_VM_BENCH_ELF (ethrex); run --ignored --nocapture"]
fn gpu_build_cpu32_ops_matches_build_cpu32_op() {
    if let Err(e) = math_cuda::device::backend() {
        eprintln!("skipping gpu_build_cpu32_ops_matches_build_cpu32_op: no CUDA backend: {e:?}");
        return;
    }
    let path = env::var("LAMBDA_VM_BENCH_ELF").expect("set LAMBDA_VM_BENCH_ELF (ethrex.elf)");
    let bytes = fs::read(&path).expect("read ELF");
    let elf = Elf::load(&bytes).expect("load ELF");
    let input = env::var("LAMBDA_VM_BENCH_INPUT")
        .ok()
        .map(|p| fs::read(p).expect("read input"))
        .unwrap_or_default();
    let executor = Executor::new(&elf, input).expect("executor");
    let result = executor.run().expect("run");
    let instructions = decode::instructions_from_elf(&elf).expect("decode");

    let n = result.logs.len();
    let mut packed = Vec::with_capacity(n);
    let mut rv1 = Vec::with_capacity(n);
    let mut rv2 = Vec::with_capacity(n);
    let mut imm = Vec::with_capacity(n);
    let mut pc = Vec::with_capacity(n);
    // Expected packed CPU32 rows (8 u64 each), program order.
    let mut expected: Vec<[u64; 8]> = Vec::new();
    for (i, log) in result.logs.iter().enumerate() {
        let instr = *instructions
            .get(&log.current_pc)
            .expect("instruction for pc");
        let ts = (i as u64) * 4 + 4;
        let op = CpuOperation::from_log_and_instruction(log, ts, instr);
        let d = DecodeEntry::from_instruction(log.current_pc, instr, 4);
        if op.decode.fields.word_instr {
            expected.push(pack_cpu32_op(&build_cpu32_op(&op)));
        }
        packed.push(d.fields.pack());
        rv1.push(op.rv1);
        rv2.push(op.rv2);
        imm.push(op.decode.imm);
        pc.push(op.decode.pc);
    }

    let (flat, rows) =
        math_cuda::trace_ops::gpu_build_cpu32_ops(&packed, &rv1, &rv2, &imm, &pc).expect("device");
    assert_eq!(rows, expected.len(), "cpu32 row count");
    assert_eq!(flat.len(), rows * 8, "flat buffer size");
    for (r, exp) in expected.iter().enumerate() {
        for c in 0..8 {
            assert_eq!(
                flat[r * 8 + c],
                exp[c],
                "cpu32 row {r} col {c} (0=ts 1=pc 2=rv1 3=rv2 4=imm 5=res 6=flags 7=bytes)"
            );
        }
    }
    println!("gpu_build_cpu32_ops parity OK over {n} cycles ({rows} CPU32 rows, res computed on device)");
}

/// End-to-end RESIDENT proof: the CPU32 table filled via the device→device chain
/// (`gpu_build_cpu32_resident`: device op-build → device fill, no intermediate host
/// round-trip) must be byte-identical to the host path (host `build_cpu32_op` → pack →
/// `gpu_build_cpu32_trace_host`). This is the first chip run fully on-device from cpu_op
/// fields through the filled trace matrix.
#[test]
#[ignore = "requires GPU + LAMBDA_VM_BENCH_ELF (ethrex); run --ignored --nocapture"]
fn gpu_cpu32_resident_matches_host_path() {
    if let Err(e) = math_cuda::device::backend() {
        eprintln!("skipping gpu_cpu32_resident_matches_host_path: no CUDA backend: {e:?}");
        return;
    }
    let path = env::var("LAMBDA_VM_BENCH_ELF").expect("set LAMBDA_VM_BENCH_ELF (ethrex.elf)");
    let bytes = fs::read(&path).expect("read ELF");
    let elf = Elf::load(&bytes).expect("load ELF");
    let input = env::var("LAMBDA_VM_BENCH_INPUT")
        .ok()
        .map(|p| fs::read(p).expect("read input"))
        .unwrap_or_default();
    let executor = Executor::new(&elf, input).expect("executor");
    let result = executor.run().expect("run");
    let instructions = decode::instructions_from_elf(&elf).expect("decode");

    let n = result.logs.len();
    let (mut packed, mut rv1, mut rv2, mut imm, mut pc) = (
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    );
    let mut host_packed_ops: Vec<u64> = Vec::new();
    let mut cpu32_rows = 0usize;
    for (i, log) in result.logs.iter().enumerate() {
        let instr = *instructions.get(&log.current_pc).expect("instruction");
        let op = CpuOperation::from_log_and_instruction(log, (i as u64) * 4 + 4, instr);
        let d = DecodeEntry::from_instruction(log.current_pc, instr, 4);
        if op.decode.fields.word_instr {
            host_packed_ops.extend_from_slice(&pack_cpu32_op(&build_cpu32_op(&op)));
            cpu32_rows += 1;
        }
        packed.push(d.fields.pack());
        rv1.push(op.rv1);
        rv2.push(op.rv2);
        imm.push(op.decode.imm);
        pc.push(op.decode.pc);
    }
    let num_rows = cpu32_rows.next_power_of_two().max(4);

    // Host path: host-built ops → device fill (uploads the ops).
    let host_table =
        math_cuda::trace_cpu::gpu_build_cpu32_trace_host(&host_packed_ops, cpu32_rows, num_rows)
            .expect("host-path fill");
    // Resident path: device op-build → device fill, no intermediate host round-trip.
    let resident_table =
        math_cuda::trace_ops::gpu_build_cpu32_resident(&packed, &rv1, &rv2, &imm, &pc, num_rows)
            .expect("resident fill");

    assert_eq!(host_table.len(), resident_table.len(), "table size");
    let mut mism = 0usize;
    for k in 0..host_table.len() {
        if host_table[k] != resident_table[k] {
            if mism < 10 {
                eprintln!("mismatch @ {k}: host={} resident={}", host_table[k], resident_table[k]);
            }
            mism += 1;
        }
    }
    assert_eq!(mism, 0, "{mism} table-cell mismatches");
    println!(
        "gpu_cpu32_resident OK: device→device chain byte-identical to host path \
         ({cpu32_rows} rows, {num_rows} padded, {} cells)",
        host_table.len()
    );
}
