//! C2-c1 parity: the device REGISTER FINAL-STATE snapshot
//! (`math_cuda::trace_walk::gpu_register_final_snapshot`) must reproduce, per word address, the
//! `(timestamp, value)` that the host `RegisterState::to_final_state_map` produces after the full
//! sequential advance (regular M1/M3/M5 + interleaved ecall register accesses). This is the device
//! source for `register_final_state` — the first step of dropping the host `cpu_ops`/advance (Camino 2).
//!
//! `LAMBDA_VM_BENCH_ELF=.../rust/ethrex.elf LAMBDA_VM_BENCH_INPUT=.../ethrex_5tx.bin \
//!   cargo test -p lambda-vm-prover --release --features cuda --lib gpu_reg_final -- --ignored --nocapture`

use std::env;
use std::fs;

use executor::elf::Elf;
use executor::vm::execution::Executor;

use crate::tables::cpu::CpuOperation;
use crate::tables::decode;
use crate::tables::trace_builder::{
    MemoryState, REG_WALK_NBINS, RegisterState, build_initial_image, collect_ops_from_cpu,
    walk_seed_from_register_init,
};

#[test]
#[ignore = "requires GPU + LAMBDA_VM_BENCH_ELF (ethrex); run --ignored --nocapture"]
fn gpu_reg_final_snapshot_matches_cpu() {
    if math_cuda::device::backend().is_err() {
        eprintln!("skipping gpu_reg_final_snapshot_matches_cpu: no CUDA backend");
        return;
    }
    let path = env::var("LAMBDA_VM_BENCH_ELF").expect("set LAMBDA_VM_BENCH_ELF (ethrex.elf)");
    let bytes = fs::read(&path).expect("read ELF");
    let elf = Elf::load(&bytes).expect("load ELF");
    let input = env::var("LAMBDA_VM_BENCH_INPUT")
        .ok()
        .map(|p| fs::read(p).expect("read input"))
        .unwrap_or_default();
    let executor = Executor::new(&elf, input.clone()).expect("executor");
    let result = executor.run().expect("run");
    let instructions = decode::instructions_from_elf(&elf).expect("decode");

    let n = result.logs.len();
    let mut cpu_ops = Vec::with_capacity(n);
    let (mut packed, mut rv1, mut rv2, mut rvd, mut next_pc) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let (mut commit_flag, mut commit_count) = (Vec::new(), Vec::new());
    for (i, log) in result.logs.iter().enumerate() {
        let instr = *instructions.get(&log.current_pc).expect("instruction");
        let op = CpuOperation::from_log_and_instruction(log, (i as u64) * 4 + 4, instr);
        packed.push(op.decode.fields.pack());
        rv1.push(op.rv1);
        rv2.push(op.rv2);
        rvd.push(op.rvd);
        next_pc.push(op.next_pc);
        commit_flag.push(u8::from(op.ecall_commit));
        commit_count.push(op.commit_count);
        cpu_ops.push(op);
    }
    let register_init = crate::tables::register::register_init_from_entry_point(elf.entry_point);

    // Host reference: full sequential advance → the final-state map + the ecall register accesses.
    let mut mem = MemoryState::from_image(&build_initial_image(&elf, &input));
    let mut reg = RegisterState::from_init(&register_init);
    let (_memw, _lo, _lt, _sh, _bw, _cm, _kc, _c32, _ec, _ed, ecall_accesses) =
        collect_ops_from_cpu(&cpu_ops, &mut mem, &mut reg, false);
    let host_map = reg.to_final_state_map();

    // Device snapshot: seed = the walk seed (regs 0-31 + pc). x254 (addr 508) is derived on device via
    // the commit-index scan (below), from `start_commit_index` = the init index.
    let init_value = walk_seed_from_register_init(&register_init);
    let start_commit_index = register_init
        .get(crate::tables::register::X254_INDEX)
        .copied()
        .unwrap_or(0) as u64;

    let e_oi: Vec<u32> = ecall_accesses.reg_op_index.clone();
    let e_addr: Vec<u32> = ecall_accesses.reg.iter().map(|a| a.reg_addr as u32).collect();
    let e_ts: Vec<u64> = ecall_accesses.reg.iter().map(|a| a.timestamp).collect();
    let e_val: Vec<u64> = ecall_accesses.reg.iter().map(|a| a.value).collect();
    let e_ir: Vec<u8> = ecall_accesses.reg.iter().map(|a| u8::from(a.is_read)).collect();

    let (dev_val, dev_ts) = math_cuda::trace_walk::gpu_register_final_snapshot(
        &packed, &rv1, &rv2, &rvd, &next_pc, &e_oi, &e_addr, &e_ts, &e_val, &e_ir, &init_value, 1,
        &commit_flag, &commit_count, start_commit_index,
    )
    .expect("device register final snapshot");
    assert_eq!(dev_val.len(), REG_WALK_NBINS as usize);

    // Compare every host word-state against the device snapshot. The device stores the full u64 value
    // at the PRIMARY (even / 508 / 510) address; the hi word (odd / 511) reads value>>32 there.
    // x254 (addr 508, the commit index) is derived on device by the C2-c2 commit-index scan.
    let mut checked = 0usize;
    for (&addr, want) in &host_map {
        let (primary, want_val) = if addr == 511 {
            (510u64, (dev_val[510] >> 32) as u32)
        } else if addr == 510 {
            (510, (dev_val[510] & 0xFFFF_FFFF) as u32)
        } else if addr == 508 {
            (508, dev_val[508] as u32)
        } else if addr % 2 == 1 {
            (addr - 1, (dev_val[(addr - 1) as usize] >> 32) as u32)
        } else {
            (addr, (dev_val[addr as usize] & 0xFFFF_FFFF) as u32)
        };
        assert_eq!(
            want.value, want_val,
            "value @ addr {addr}: host {} != device {want_val}",
            want.value
        );
        assert_eq!(
            want.timestamp, dev_ts[primary as usize],
            "ts @ addr {addr}: host {} != device {}",
            want.timestamp, dev_ts[primary as usize]
        );
        checked += 1;
    }
    println!(
        "gpu_reg_final_snapshot parity OK over {checked} register word-states ({} ecall reg accesses)",
        ecall_accesses.reg.len()
    );
}
