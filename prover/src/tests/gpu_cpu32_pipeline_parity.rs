//! Pipeline-integration parity: the p5 resident CPU32 path
//! ([`crate::tables::gpu_trace::build_cpu32_resident_tables`], fed the resident `cpu_ops`)
//! must produce the exact same device-resident CPU32 trace table as the existing host-op
//! device path (`gpu_build_cpu32_tables`, fed host-built `cpu32_ops`). This validates the
//! integration wrapper wired into `trace_builder` p5 behind `LAMBDA_VM_GPU_RESIDENT_CHIPS`.
//!
//! `LAMBDA_VM_BENCH_ELF=.../rust/ethrex.elf LAMBDA_VM_BENCH_INPUT=.../ethrex_5tx.bin \
//!   cargo test -p lambda-vm-prover --release --features cuda --lib gpu_cpu32_pipeline -- --ignored --nocapture`

use std::env;
use std::fs;

use executor::elf::Elf;
use executor::vm::execution::Executor;

use crate::tables::cpu::CpuOperation;
use crate::tables::decode;
use std::collections::HashMap;

use crate::tables::branch::BranchOperation;
use crate::tables::bytewise::BytewiseOperation;
use crate::tables::dvrm::DvrmOperation;
use crate::tables::eq::EqOperation;
use crate::tables::gpu_trace::{
    build_branch_resident_tables, build_bytewise_resident_tables, build_cpu32_resident_tables,
    build_dvrm_resident_tables, build_eq_resident_tables, build_load_resident_tables,
    build_mul_resident_tables, build_shift_resident_tables, build_store_resident_tables,
    gpu_build_branch_tables, gpu_build_bytewise_tables, gpu_build_cpu32_tables,
    gpu_build_dvrm_tables, gpu_build_eq_tables, gpu_build_load_tables, gpu_build_mul_tables,
    gpu_build_shift_tables, gpu_build_store_tables,
};
use crate::tables::load::LoadOperation;
use crate::tables::mul::MulOperation;
use crate::tables::shift::ShiftOperation;
use crate::tables::store::StoreOperation;
use crate::tables::trace_builder::{build_cpu32_op, cpu32_chip_op};
use crate::tables::types::DecodeEntry;

#[test]
#[ignore = "requires GPU + LAMBDA_VM_BENCH_ELF (ethrex); run --ignored --nocapture"]
fn gpu_cpu32_pipeline_resident_matches_host_op_path() {
    let Ok(be) = math_cuda::device::backend() else {
        eprintln!("skipping: no CUDA backend");
        return;
    };
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

    let mut cpu_ops = Vec::with_capacity(result.logs.len());
    let mut cpu32_ops = Vec::new();
    for (i, log) in result.logs.iter().enumerate() {
        let instr = *instructions.get(&log.current_pc).expect("instruction");
        let op = CpuOperation::from_log_and_instruction(log, (i as u64) * 4 + 4, instr);
        if op.decode.fields.word_instr {
            cpu32_ops.push(build_cpu32_op(&op));
        }
        cpu_ops.push(op);
    }
    let max_rows = cpu32_ops.len().max(1) * 2; // single chunk for both paths

    let host_tables = gpu_build_cpu32_tables(&cpu32_ops, max_rows).expect("host-op device path");
    let res_tables = build_cpu32_resident_tables(&cpu_ops, max_rows).expect("resident path");
    assert_eq!(host_tables.len(), 1, "host chunks");
    assert_eq!(res_tables.len(), 1, "resident chunks");

    let stream = be.next_stream();
    let host_buf = stream
        .clone_dtoh(host_tables[0].main_input_dev().expect("host dev buf"))
        .expect("dtoh host");
    let res_buf = stream
        .clone_dtoh(res_tables[0].main_input_dev().expect("resident dev buf"))
        .expect("dtoh resident");
    stream.synchronize().expect("sync");

    assert_eq!(host_buf.len(), res_buf.len(), "table size");
    assert_eq!(host_buf, res_buf, "resident CPU32 table != host-op device table");
    println!(
        "gpu_cpu32_pipeline parity OK: p5 resident CPU32 table byte-identical to host-op device \
         path ({} cpu32 rows, {} cells)",
        cpu32_ops.len(),
        host_buf.len()
    );
}

/// Same integration check for the per-row memory chips: the p5 resident LOAD/STORE tables
/// (`build_{load,store}_resident_tables`) must be byte-identical to the host-op device path.
#[test]
#[ignore = "requires GPU + LAMBDA_VM_BENCH_ELF (ethrex); run --ignored --nocapture"]
fn gpu_load_store_pipeline_resident_matches_host_op_path() {
    let Ok(be) = math_cuda::device::backend() else {
        eprintln!("skipping: no CUDA backend");
        return;
    };
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

    let mut cpu_ops = Vec::with_capacity(result.logs.len());
    let mut load_ops = Vec::new();
    let mut store_ops = Vec::new();
    for (i, log) in result.logs.iter().enumerate() {
        let instr = *instructions.get(&log.current_pc).expect("instruction");
        let op = CpuOperation::from_log_and_instruction(log, (i as u64) * 4 + 4, instr);
        let f = op.decode.fields;
        if f.is_load() {
            let bc = f.mem_bytes();
            let signed = f.mem_signed();
            let mut vb = [0u32; 8];
            for (j, b) in vb.iter_mut().take(bc).enumerate() {
                *b = ((op.rvd >> (j * 8)) & 0xFF) as u32;
            }
            let mut rb = vb;
            if bc < 8 {
                let fill = if signed && (vb[bc - 1] >> 7) & 1 == 1 { 0xFF } else { 0 };
                for b in rb.iter_mut().skip(bc) {
                    *b = fill;
                }
            }
            load_ops.push(LoadOperation::new(op.res, op.timestamp, bc as u8, signed, rb.map(u64::from)));
        }
        if f.is_store() {
            store_ops.push(StoreOperation::new(op.res, op.timestamp, op.rv2, f.mem_bytes() as u8));
        }
        cpu_ops.push(op);
    }

    let stream = be.next_stream();
    // LOAD
    {
        let max_rows = load_ops.len().max(1) * 2;
        let host = gpu_build_load_tables(&load_ops, max_rows).expect("host load");
        let res = build_load_resident_tables(&cpu_ops, max_rows).expect("resident load");
        let hb = stream.clone_dtoh(host[0].main_input_dev().unwrap()).unwrap();
        let rb = stream.clone_dtoh(res[0].main_input_dev().unwrap()).unwrap();
        stream.synchronize().unwrap();
        assert_eq!(hb, rb, "LOAD resident table != host-op device table");
        println!("  LOAD pipeline OK ({} rows)", load_ops.len());
    }
    // STORE
    {
        let max_rows = store_ops.len().max(1) * 2;
        let host = gpu_build_store_tables(&store_ops, max_rows).expect("host store");
        let res = build_store_resident_tables(&cpu_ops, max_rows).expect("resident store");
        let hb = stream.clone_dtoh(host[0].main_input_dev().unwrap()).unwrap();
        let rb = stream.clone_dtoh(res[0].main_input_dev().unwrap()).unwrap();
        stream.synchronize().unwrap();
        assert_eq!(hb, rb, "STORE resident table != host-op device table");
        println!("  STORE pipeline OK ({} rows)", store_ops.len());
    }
    println!("gpu_load_store_pipeline parity OK: p5 resident LOAD+STORE tables byte-identical to host-op device path");
}

/// SHIFT pipeline integration: the p5 resident SHIFT table (`build_shift_resident_tables`,
/// merging instruction + cpu32 shifts on device) must be byte-identical to the host-op device
/// path (`gpu_build_shift_tables` fed instruction ++ cpu32 shift ops in that order).
#[test]
#[ignore = "requires GPU + LAMBDA_VM_BENCH_ELF (ethrex); run --ignored --nocapture"]
fn gpu_shift_pipeline_resident_matches_host_op_path() {
    let Ok(be) = math_cuda::device::backend() else {
        eprintln!("skipping: no CUDA backend");
        return;
    };
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

    let mut cpu_ops = Vec::with_capacity(result.logs.len());
    let mut shift_ops: Vec<ShiftOperation> = Vec::new();
    let mut cpu32_shifts: Vec<ShiftOperation> = Vec::new();
    for (i, log) in result.logs.iter().enumerate() {
        let instr = *instructions.get(&log.current_pc).expect("instruction");
        let op = CpuOperation::from_log_and_instruction(log, (i as u64) * 4 + 4, instr);
        let f = op.decode.fields;
        if !f.word_instr && f.is_shift() {
            shift_ops.push(ShiftOperation::new(
                op.rv1,
                op.arg2,
                f.alu_signed2_or_invert(),
                f.alu_signed(),
                f.word_instr,
            ));
        }
        if f.word_instr {
            let c = build_cpu32_op(&op);
            let (mut m, mut dv) = (Vec::new(), Vec::new());
            cpu32_chip_op(&c, &mut cpu32_shifts, &mut m, &mut dv);
        }
        cpu_ops.push(op);
    }
    shift_ops.extend(cpu32_shifts);

    let max_rows = shift_ops.len().max(1) * 2;
    let host = gpu_build_shift_tables(&shift_ops, max_rows).expect("host shift");
    let res = build_shift_resident_tables(&cpu_ops, max_rows).expect("resident shift");
    let stream = be.next_stream();
    let hb = stream.clone_dtoh(host[0].main_input_dev().unwrap()).unwrap();
    let rb = stream.clone_dtoh(res[0].main_input_dev().unwrap()).unwrap();
    stream.synchronize().unwrap();
    assert_eq!(hb, rb, "SHIFT resident table != host-op device table");
    println!("gpu_shift_pipeline parity OK: p5 resident SHIFT table byte-identical to host-op device path ({} rows)", shift_ops.len());
}

/// Deduped-chip pipeline integration: the p5 resident EQ/BYTEWISE tables
/// (`build_{eq,bytewise}_resident_tables`, auto-sized to the device unique count) must equal
/// the host-op device path (`gpu_build_{eq,bytewise}_tables`) as a MULTISET of rows (resident
/// emits sorted vs host HashMap order — both valid, LogUp bus is order-independent).
#[test]
#[ignore = "requires GPU + LAMBDA_VM_BENCH_ELF (ethrex); run --ignored --nocapture"]
fn gpu_eq_bytewise_pipeline_resident_matches_host_op_path() {
    let Ok(be) = math_cuda::device::backend() else {
        eprintln!("skipping: no CUDA backend");
        return;
    };
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

    let mut cpu_ops = Vec::with_capacity(result.logs.len());
    let mut eq_ops = Vec::new();
    let mut bytewise_ops = Vec::new();
    for (i, log) in result.logs.iter().enumerate() {
        let instr = *instructions.get(&log.current_pc).expect("instruction");
        let op = CpuOperation::from_log_and_instruction(log, (i as u64) * 4 + 4, instr);
        let f = op.decode.fields;
        if !f.word_instr && f.is_eq() {
            eq_ops.push(EqOperation::new(op.rv1, op.arg2, f.alu_signed2_or_invert()));
        }
        if !f.word_instr && (f.is_and() || f.is_or() || f.is_xor()) {
            bytewise_ops.push(BytewiseOperation::new(op.rv1, op.arg2, f.alu_op()));
        }
        cpu_ops.push(op);
    }

    let stream = be.next_stream();
    let row_ms = |t: &[u64], nc: usize| -> HashMap<Vec<u64>, usize> {
        let mut m = HashMap::new();
        for r in t.chunks_exact(nc) {
            *m.entry(r.to_vec()).or_insert(0) += 1;
        }
        m
    };
    // EQ
    {
        let nc = crate::tables::eq::cols::NUM_COLUMNS;
        let max_rows = eq_ops.len().max(1) * 2;
        let host = gpu_build_eq_tables(&eq_ops, max_rows).expect("host eq");
        let res = build_eq_resident_tables(&cpu_ops, max_rows).expect("resident eq");
        let hb = stream.clone_dtoh(host[0].main_input_dev().unwrap()).unwrap();
        let rb = stream.clone_dtoh(res[0].main_input_dev().unwrap()).unwrap();
        stream.synchronize().unwrap();
        assert_eq!(row_ms(&hb, nc), row_ms(&rb, nc), "EQ p5 table multiset != host");
        println!("  EQ pipeline OK ({} raw ops, multiset-identical)", eq_ops.len());
    }
    // BYTEWISE
    {
        let nc = crate::tables::bytewise::cols::NUM_COLUMNS;
        let max_rows = bytewise_ops.len().max(1) * 2;
        let host = gpu_build_bytewise_tables(&bytewise_ops, max_rows).expect("host bytewise");
        let res = build_bytewise_resident_tables(&cpu_ops, max_rows).expect("resident bytewise");
        let hb = stream.clone_dtoh(host[0].main_input_dev().unwrap()).unwrap();
        let rb = stream.clone_dtoh(res[0].main_input_dev().unwrap()).unwrap();
        stream.synchronize().unwrap();
        assert_eq!(row_ms(&hb, nc), row_ms(&rb, nc), "BYTEWISE p5 table multiset != host");
        println!("  BYTEWISE pipeline OK ({} raw ops, multiset-identical)", bytewise_ops.len());
    }
    println!("gpu_eq_bytewise_pipeline parity OK: p5 resident EQ+BYTEWISE tables multiset-identical to host-op device path");
}

/// DVRM (instruction ⊕ cpu32) + BRANCH (branch_cond) pipeline integration — both complete
/// deduped chips. p5 resident tables must be MULTISET-identical to the host-op device path.
#[test]
#[ignore = "requires GPU + LAMBDA_VM_BENCH_ELF (ethrex); run --ignored --nocapture"]
fn gpu_dvrm_branch_pipeline_resident_matches_host_op_path() {
    let Ok(be) = math_cuda::device::backend() else {
        eprintln!("skipping: no CUDA backend");
        return;
    };
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

    let mut cpu_ops = Vec::with_capacity(result.logs.len());
    let mut dvrm_ops: Vec<(DvrmOperation, bool)> = Vec::new();
    let mut branch_ops: Vec<BranchOperation> = Vec::new();
    for (i, log) in result.logs.iter().enumerate() {
        let instr = *instructions.get(&log.current_pc).expect("instruction");
        let op = CpuOperation::from_log_and_instruction(log, (i as u64) * 4 + 4, instr);
        let d = DecodeEntry::from_instruction(log.current_pc, instr, 4);
        let f = op.decode.fields;
        if !f.word_instr && f.is_divrem() {
            dvrm_ops.push((DvrmOperation::new(op.rv1, op.arg2, f.alu_signed()), f.alu_muldiv()));
        }
        if f.word_instr {
            let c = build_cpu32_op(&op);
            let (mut s, mut m) = (Vec::new(), Vec::new());
            cpu32_chip_op(&c, &mut s, &mut m, &mut dvrm_ops);
        }
        if op.branch_cond {
            branch_ops.push(BranchOperation::new(d.pc, d.imm, op.rv1, f.jalr()));
        }
        cpu_ops.push(op);
    }

    let stream = be.next_stream();
    let row_ms = |t: &[u64], nc: usize| -> HashMap<Vec<u64>, usize> {
        let mut m = HashMap::new();
        for r in t.chunks_exact(nc) {
            *m.entry(r.to_vec()).or_insert(0) += 1;
        }
        m
    };
    // DVRM
    {
        let nc = crate::tables::dvrm::cols::NUM_COLUMNS;
        let max_rows = dvrm_ops.len().max(1) * 2;
        let host = gpu_build_dvrm_tables(&dvrm_ops, max_rows).expect("host dvrm");
        let res = build_dvrm_resident_tables(&cpu_ops, max_rows).expect("resident dvrm");
        let hb = stream.clone_dtoh(host[0].main_input_dev().unwrap()).unwrap();
        let rb = stream.clone_dtoh(res[0].main_input_dev().unwrap()).unwrap();
        stream.synchronize().unwrap();
        assert_eq!(row_ms(&hb, nc), row_ms(&rb, nc), "DVRM p5 table multiset != host");
        println!("  DVRM pipeline OK ({} raw ops, multiset-identical)", dvrm_ops.len());
    }
    // BRANCH
    {
        let nc = crate::tables::branch::cols::NUM_COLUMNS;
        let max_rows = branch_ops.len().max(1) * 2;
        let host = gpu_build_branch_tables(&branch_ops, max_rows).expect("host branch");
        let res = build_branch_resident_tables(&cpu_ops, max_rows).expect("resident branch");
        let hb = stream.clone_dtoh(host[0].main_input_dev().unwrap()).unwrap();
        let rb = stream.clone_dtoh(res[0].main_input_dev().unwrap()).unwrap();
        stream.synchronize().unwrap();
        assert_eq!(row_ms(&hb, nc), row_ms(&rb, nc), "BRANCH p5 table multiset != host");
        println!("  BRANCH pipeline OK ({} raw ops, multiset-identical)", branch_ops.len());
    }
    println!("gpu_dvrm_branch_pipeline parity OK: p5 resident DVRM+BRANCH tables multiset-identical to host-op device path");
}

/// MUL pipeline integration — the COMPLETE table (all 4 sources: instruction ⊕
/// instruction-dvrm→mul ⊕ cpu32 ⊕ cpu32-dvrm→mul). The p5 resident MUL table
/// (`build_mul_resident_tables`) must be MULTISET-identical to the host-op device path
/// (`gpu_build_mul_tables` fed the production `mul_ops` built the same way).
#[test]
#[ignore = "requires GPU + LAMBDA_VM_BENCH_ELF (ethrex); run --ignored --nocapture"]
fn gpu_mul_pipeline_resident_matches_host_op_path() {
    let Ok(be) = math_cuda::device::backend() else {
        eprintln!("skipping: no CUDA backend");
        return;
    };
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

    let mut cpu_ops = Vec::with_capacity(result.logs.len());
    let mut mul_ops: Vec<(MulOperation, bool)> = Vec::new();
    let mut dvrm_ops: Vec<(DvrmOperation, bool)> = Vec::new();
    for (i, log) in result.logs.iter().enumerate() {
        let instr = *instructions.get(&log.current_pc).expect("instruction");
        let op = CpuOperation::from_log_and_instruction(log, (i as u64) * 4 + 4, instr);
        let f = op.decode.fields;
        if !f.word_instr && f.is_mul() {
            mul_ops.push((
                MulOperation::new(op.rv1, f.alu_signed(), op.arg2, f.alu_signed2_or_invert()),
                f.alu_muldiv(),
            ));
        }
        if !f.word_instr && f.is_divrem() {
            dvrm_ops.push((DvrmOperation::new(op.rv1, op.arg2, f.alu_signed()), f.alu_muldiv()));
        }
        if f.word_instr {
            let c = build_cpu32_op(&op);
            let mut s = Vec::new();
            cpu32_chip_op(&c, &mut s, &mut mul_ops, &mut dvrm_ops);
        }
        cpu_ops.push(op);
    }
    // dvrm→mul (C13 lo + C14 hi) over ALL dvrm ops (instruction + cpu32) — matches production.
    for (dv, _) in dvrm_ops.clone() {
        let m = MulOperation::new(dv.d, dv.signed, dv.compute_quotient(), dv.sign_q());
        mul_ops.push((m.clone(), false));
        mul_ops.push((m, true));
    }

    let nc = crate::tables::mul::cols::NUM_COLUMNS;
    let max_rows = mul_ops.len().max(1) * 2;
    let host = gpu_build_mul_tables(&mul_ops, max_rows).expect("host mul");
    let res = build_mul_resident_tables(&cpu_ops, max_rows).expect("resident mul");
    let stream = be.next_stream();
    let hb = stream.clone_dtoh(host[0].main_input_dev().unwrap()).unwrap();
    let rb = stream.clone_dtoh(res[0].main_input_dev().unwrap()).unwrap();
    stream.synchronize().unwrap();
    let ms = |t: &[u64]| -> HashMap<Vec<u64>, usize> {
        let mut m = HashMap::new();
        for r in t.chunks_exact(nc) {
            *m.entry(r.to_vec()).or_insert(0) += 1;
        }
        m
    };
    assert_eq!(ms(&hb), ms(&rb), "MUL p5 table multiset != host");
    println!(
        "gpu_mul_pipeline parity OK: p5 resident MUL table (all 4 sources) multiset-identical to \
         host-op device path ({} raw mul ops)",
        mul_ops.len()
    );
}
