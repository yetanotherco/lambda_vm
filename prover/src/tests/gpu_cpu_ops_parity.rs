//! Phase-0 byte-parity: the device `CpuOperation` builder
//! ([`math_cuda::trace_ops::gpu_build_cpu_ops`]) must reproduce, field-for-field, what
//! the host `CpuOperation::from_log` computes for every cycle — the resident seam the
//! rest of GPU trace-gen reads. Runs on the configured guest program (ethrex_5tx).
//!
//! `LAMBDA_VM_BENCH_ELF=.../rust/ethrex.elf LAMBDA_VM_BENCH_INPUT=.../ethrex_5tx.bin \
//!   cargo test -p lambda-vm-prover --release --features cuda --lib gpu_build_cpu_ops -- --ignored --nocapture`

use std::env;
use std::fs;

use executor::elf::Elf;
use executor::vm::execution::Executor;

use crate::tables::cpu::CpuOperation;
use crate::tables::decode;
use crate::tables::types::DecodeEntry;

#[test]
#[ignore = "requires GPU + LAMBDA_VM_BENCH_ELF (ethrex); run --ignored --nocapture"]
fn gpu_build_cpu_ops_matches_from_log() {
    if let Err(e) = math_cuda::device::backend() {
        eprintln!("skipping gpu_build_cpu_ops_matches_from_log: no CUDA backend: {e:?}");
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

    // CPU reference (from_log) + the device-builder input SoA, built consistently.
    let n = result.logs.len();
    let (mut cpc, mut npc, mut s1, mut s2, mut dv) = (
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    );
    let (mut pc, mut imm, mut packed) = (
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    );
    let mut cpu_ops = Vec::with_capacity(n);
    for (i, log) in result.logs.iter().enumerate() {
        let instr = *instructions
            .get(&log.current_pc)
            .expect("instruction for pc");
        let ts = (i as u64) * 4 + 4;
        cpu_ops.push(CpuOperation::from_log_and_instruction(log, ts, instr));
        let d = DecodeEntry::from_instruction(log.current_pc, instr, 4);
        cpc.push(log.current_pc);
        npc.push(log.next_pc);
        s1.push(log.src1_val);
        s2.push(log.src2_val);
        dv.push(log.dst_val);
        pc.push(d.pc);
        imm.push(d.imm);
        packed.push(d.fields.pack());
    }

    let dev =
        math_cuda::trace_ops::gpu_build_cpu_ops(&cpc, &npc, &s1, &s2, &dv, &pc, &imm, &packed)
            .expect("device build_cpu_ops");

    for (i, op) in cpu_ops.iter().enumerate() {
        assert_eq!(dev.rv1[i], op.rv1, "rv1 @ {i}");
        assert_eq!(dev.rv2[i], op.rv2, "rv2 @ {i}");
        assert_eq!(dev.arg2[i], op.arg2, "arg2 @ {i}");
        assert_eq!(dev.res[i], op.res, "res @ {i}");
        assert_eq!(dev.rvd[i], op.rvd, "rvd @ {i}");
        assert_eq!(dev.next_pc[i], op.next_pc, "next_pc @ {i}");
        let fl = dev.flags[i];
        assert_eq!(fl & 1 != 0, op.branch_cond, "branch_cond @ {i}");
        assert_eq!(fl & 2 != 0, op.ecall_commit, "ecall_commit @ {i}");
        assert_eq!(fl & 4 != 0, op.ecall_keccak, "ecall_keccak @ {i}");
        assert_eq!(fl & 8 != 0, op.ecall_ecsm, "ecall_ecsm @ {i}");
        assert_eq!(
            dev.commit_buf_addr[i], op.commit_buf_addr,
            "commit_buf_addr @ {i}"
        );
        assert_eq!(dev.commit_count[i], op.commit_count, "commit_count @ {i}");
        assert_eq!(
            dev.keccak_state_addr[i], op.keccak_state_addr,
            "keccak_state_addr @ {i}"
        );
    }
    println!("gpu_build_cpu_ops parity OK over {n} cycles");
}
