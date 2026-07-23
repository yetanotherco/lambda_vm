//! Phase-3d byte-parity: the device LOAD chip-op builder
//! ([`math_cuda::trace_ops::gpu_build_load_ops`]) must reproduce, row-for-row in program
//! order, the packed LOAD rows the host builds via `collect_load_op_from_cpu`'s
//! `LoadOperation` + `pack_load_op` — including the sign/zero-extended `res_bytes`, computed
//! on device. The LOAD chip table is a pure state-free cpu_op projection (the MEMW read
//! row's old_ts is the Phase-2 walk, separate), so it validates on the real guest.
//!
//! `LAMBDA_VM_BENCH_ELF=.../rust/ethrex.elf LAMBDA_VM_BENCH_INPUT=.../ethrex_5tx.bin \
//!   cargo test -p lambda-vm-prover --release --features cuda --lib gpu_build_load -- --ignored --nocapture`

use std::env;
use std::fs;

use executor::elf::Elf;
use executor::vm::execution::Executor;

use crate::tables::cpu::CpuOperation;
use crate::tables::decode;
use crate::tables::gpu_trace::pack_load_op;
use crate::tables::load::LoadOperation;
use crate::tables::types::DecodeEntry;

#[test]
#[ignore = "requires GPU + LAMBDA_VM_BENCH_ELF (ethrex); run --ignored --nocapture"]
fn gpu_build_load_ops_matches_collect() {
    if let Err(e) = math_cuda::device::backend() {
        eprintln!("skipping gpu_build_load_ops_matches_collect: no CUDA backend: {e:?}");
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
    let mut res = Vec::with_capacity(n);
    let mut rvd = Vec::with_capacity(n);
    let mut expected: Vec<[u64; 7]> = Vec::new();
    for (i, log) in result.logs.iter().enumerate() {
        let instr = *instructions
            .get(&log.current_pc)
            .expect("instruction for pc");
        let ts = (i as u64) * 4 + 4;
        let op = CpuOperation::from_log_and_instruction(log, ts, instr);
        let d = DecodeEntry::from_instruction(log.current_pc, instr, 4);
        let f = op.decode.fields;
        if f.is_load() {
            // Mirror collect_load_op_from_cpu's res_bytes sign/zero extension.
            let byte_count = f.mem_bytes();
            let signed = f.mem_signed();
            let loaded = op.rvd;
            let mut value_bytes = [0u32; 8];
            for (j, b) in value_bytes.iter_mut().take(byte_count).enumerate() {
                *b = ((loaded >> (j * 8)) & 0xFF) as u32;
            }
            let mut res_bytes = value_bytes;
            if byte_count < 8 {
                let msb = value_bytes[byte_count - 1];
                let sign_bit = (msb >> 7) & 1;
                let fill = if signed && sign_bit == 1 { 0xFF } else { 0 };
                for b in res_bytes.iter_mut().skip(byte_count) {
                    *b = fill;
                }
            }
            let load_op =
                LoadOperation::new(op.res, op.timestamp, byte_count as u8, signed, res_bytes.map(u64::from));
            expected.push(pack_load_op(&load_op));
        }
        packed.push(d.fields.pack());
        res.push(op.res);
        rvd.push(op.rvd);
    }

    let (flat, rows) =
        math_cuda::trace_ops::gpu_build_load_ops(&packed, &res, &rvd).expect("device");
    assert_eq!(rows, expected.len(), "load row count");
    assert_eq!(flat.len(), rows * 7, "flat buffer size");
    for (r, exp) in expected.iter().enumerate() {
        for c in 0..7 {
            assert_eq!(
                flat[r * 7 + c],
                exp[c],
                "load row {r} col {c} (0=flags 1=base 2=ts 3=r01 4=r23 5=r45 6=r67)"
            );
        }
    }
    println!("gpu_build_load_ops parity OK over {n} cycles ({rows} LOAD rows, res_bytes on device)");
}
