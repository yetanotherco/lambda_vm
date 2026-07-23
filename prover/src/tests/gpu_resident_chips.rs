//! Resident-chain proofs for the per-row memory chips (LOAD, STORE): the table filled via
//! the device→device chain (`gpu_build_{load,store}_resident`: device op-build → device fill,
//! no intermediate host round-trip) must be byte-identical to the host path (host op-build →
//! pack → `gpu_build_{load,store}_trace_host`). Extends the CPU32 resident proof to the
//! memory-chip table side (the MEMW rows / old_ts remain Phase 2). ethrex_5tx.
//!
//! `LAMBDA_VM_BENCH_ELF=.../rust/ethrex.elf LAMBDA_VM_BENCH_INPUT=.../ethrex_5tx.bin \
//!   cargo test -p lambda-vm-prover --release --features cuda --lib gpu_resident -- --ignored --nocapture`

use std::env;
use std::fs;

use executor::elf::Elf;
use executor::vm::execution::Executor;

use crate::tables::cpu::CpuOperation;
use crate::tables::decode;
use crate::tables::gpu_trace::{pack_load_op, pack_store_op};
use crate::tables::load::LoadOperation;
use crate::tables::store::StoreOperation;
use crate::tables::types::DecodeEntry;

#[test]
#[ignore = "requires GPU + LAMBDA_VM_BENCH_ELF (ethrex); run --ignored --nocapture"]
fn gpu_load_store_resident_matches_host_path() {
    if let Err(e) = math_cuda::device::backend() {
        eprintln!("skipping gpu_load_store_resident_matches_host_path: no CUDA backend: {e:?}");
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
    let (mut packed, mut res, mut rvd, mut rv2) = (
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    );
    let mut load_packed_ops: Vec<u64> = Vec::new();
    let mut store_packed_ops: Vec<u64> = Vec::new();
    let (mut load_rows, mut store_rows) = (0usize, 0usize);
    for (i, log) in result.logs.iter().enumerate() {
        let instr = *instructions.get(&log.current_pc).expect("instruction");
        let op = CpuOperation::from_log_and_instruction(log, (i as u64) * 4 + 4, instr);
        let d = DecodeEntry::from_instruction(log.current_pc, instr, 4);
        let f = op.decode.fields;
        if f.is_load() {
            let bc = f.mem_bytes();
            let signed = f.mem_signed();
            let loaded = op.rvd;
            let mut vb = [0u32; 8];
            for (j, b) in vb.iter_mut().take(bc).enumerate() {
                *b = ((loaded >> (j * 8)) & 0xFF) as u32;
            }
            let mut rb = vb;
            if bc < 8 {
                let fill = if signed && (vb[bc - 1] >> 7) & 1 == 1 { 0xFF } else { 0 };
                for b in rb.iter_mut().skip(bc) {
                    *b = fill;
                }
            }
            load_packed_ops.extend_from_slice(&pack_load_op(&LoadOperation::new(
                op.res,
                op.timestamp,
                bc as u8,
                signed,
                rb.map(u64::from),
            )));
            load_rows += 1;
        }
        if f.is_store() {
            store_packed_ops.extend_from_slice(&pack_store_op(&StoreOperation::new(
                op.res,
                op.timestamp,
                op.rv2,
                f.mem_bytes() as u8,
            )));
            store_rows += 1;
        }
        packed.push(d.fields.pack());
        res.push(op.res);
        rvd.push(op.rvd);
        rv2.push(op.rv2);
    }

    // LOAD
    let load_num_rows = load_rows.next_power_of_two().max(4);
    let load_host =
        math_cuda::trace_cpu::gpu_build_load_trace_host(&load_packed_ops, load_rows, load_num_rows)
            .expect("load host fill");
    let load_res =
        math_cuda::trace_ops::gpu_build_load_resident(&packed, &res, &rvd, load_num_rows)
            .expect("load resident");
    assert_eq!(load_host, load_res, "LOAD resident != host path");

    // STORE
    let store_num_rows = store_rows.next_power_of_two().max(4);
    let store_host = math_cuda::trace_cpu::gpu_build_store_trace_host(
        &store_packed_ops,
        store_rows,
        store_num_rows,
    )
    .expect("store host fill");
    let store_res =
        math_cuda::trace_ops::gpu_build_store_resident(&packed, &res, &rv2, store_num_rows)
            .expect("store resident");
    assert_eq!(store_host, store_res, "STORE resident != host path");

    println!(
        "resident LOAD ({load_rows} rows) + STORE ({store_rows} rows) chains byte-identical to host path"
    );
}
