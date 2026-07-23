//! Resident deduped-chip capstone: the LT trace table built via the fully-device chain
//! (`gpu_build_lt_resident`: cpu_op fields → device route+extract → device dedup → device
//! pack → device `lt_fill`, no host round-trip) must match the host path (CPU `lt_ops` →
//! HashMap dedup → pack → `gpu_build_lt_trace_host`) as a MULTISET of table rows. LT's LogUp
//! bus is order-independent, so device (sorted) vs host (HashMap) row order is irrelevant —
//! only the multiset of rows must match. This is the deduped-chip analog of the CPU32/LOAD/
//! STORE resident proofs, and the template for the other 6 deduped chips. ethrex_5tx.
//!
//! `LAMBDA_VM_BENCH_ELF=.../rust/ethrex.elf LAMBDA_VM_BENCH_INPUT=.../ethrex_5tx.bin \
//!   cargo test -p lambda-vm-prover --release --features cuda --lib gpu_lt_resident -- --ignored --nocapture`

use std::collections::HashMap;
use std::env;
use std::fs;

use executor::elf::Elf;
use executor::vm::execution::Executor;

use crate::tables::bytewise::BytewiseOperation;
use crate::tables::cpu::CpuOperation;
use crate::tables::decode;
use crate::tables::dvrm::DvrmOperation;
use crate::tables::eq::EqOperation;
use crate::tables::gpu_trace::{pack_bytewise_op, pack_eq_op, pack_lt_op};
use crate::tables::lt::LtOperation;
use crate::tables::types::DecodeEntry;

/// Multiset of table rows (row = LT_NCOLS-wide slice) → {row: count}.
fn row_multiset(table: &[u64], ncols: usize) -> HashMap<Vec<u64>, usize> {
    let mut m = HashMap::new();
    for row in table.chunks_exact(ncols) {
        *m.entry(row.to_vec()).or_insert(0) += 1;
    }
    m
}

#[test]
#[ignore = "requires GPU + LAMBDA_VM_BENCH_ELF (ethrex); run --ignored --nocapture"]
fn gpu_lt_resident_matches_host_path() {
    if let Err(e) = math_cuda::device::backend() {
        eprintln!("skipping gpu_lt_resident_matches_host_path: no CUDA backend: {e:?}");
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
    let (mut packed, mut rv1, mut arg2) =
        (Vec::with_capacity(n), Vec::with_capacity(n), Vec::with_capacity(n));
    let mut host_map: HashMap<LtOperation, u64> = HashMap::new();
    for (i, log) in result.logs.iter().enumerate() {
        let instr = *instructions.get(&log.current_pc).expect("instruction");
        let op = CpuOperation::from_log_and_instruction(log, (i as u64) * 4 + 4, instr);
        let d = DecodeEntry::from_instruction(log.current_pc, instr, 4);
        let f = op.decode.fields;
        if !f.word_instr && f.is_lt() {
            let lt = LtOperation::new_with_invert(
                op.rv1,
                op.arg2,
                f.alu_signed(),
                f.alu_signed2_or_invert(),
            );
            *host_map.entry(lt).or_insert(0) += 1;
        }
        packed.push(d.fields.pack());
        rv1.push(op.rv1);
        arg2.push(op.arg2);
    }

    let n_unique = host_map.len();
    let num_rows = n_unique.next_power_of_two().max(4);
    let ncols = math_cuda::trace_cpu::LT_NCOLS;

    // Host path: HashMap-deduped ops → device fill.
    let mut host_packed = Vec::with_capacity(n_unique * math_cuda::trace_cpu::LT_STRIDE);
    for (op, mult) in &host_map {
        host_packed.extend_from_slice(&pack_lt_op(op, *mult));
    }
    let host_table = math_cuda::trace_cpu::gpu_build_lt_trace_host(&host_packed, n_unique, num_rows)
        .expect("host lt fill");

    // Resident path: fully on-device.
    let res_table = math_cuda::trace_ops::gpu_build_lt_resident(&packed, &rv1, &arg2, num_rows)
        .expect("resident lt");

    assert_eq!(host_table.len(), res_table.len(), "table size");
    let host_ms = row_multiset(&host_table, ncols);
    let res_ms = row_multiset(&res_table, ncols);
    assert_eq!(
        host_ms.len(),
        res_ms.len(),
        "distinct-row count: host {} vs resident {}",
        host_ms.len(),
        res_ms.len()
    );
    for (row, &hc) in &host_ms {
        match res_ms.get(row) {
            Some(&rc) => assert_eq!(rc, hc, "row multiplicity mismatch"),
            None => panic!("resident missing a host row"),
        }
    }
    println!(
        "gpu_lt_resident OK: device→device LT chain multiset-identical to host path \
         ({n_unique} unique rows, {num_rows} padded, {} cells)",
        host_table.len()
    );
}

#[test]
#[ignore = "requires GPU + LAMBDA_VM_BENCH_ELF (ethrex); run --ignored --nocapture"]
fn gpu_lt_instr_dvrm_resident_matches_host_path() {
    if let Err(e) = math_cuda::device::backend() {
        eprintln!("skipping: no CUDA backend: {e:?}");
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
    let (mut packed, mut rv1, mut arg2) =
        (Vec::with_capacity(n), Vec::with_capacity(n), Vec::with_capacity(n));
    // Host merged source: instruction-driven LT ⊕ dvrm-derived LT.
    let mut host_map: HashMap<LtOperation, u64> = HashMap::new();
    for (i, log) in result.logs.iter().enumerate() {
        let instr = *instructions.get(&log.current_pc).expect("instruction");
        let op = CpuOperation::from_log_and_instruction(log, (i as u64) * 4 + 4, instr);
        let d = DecodeEntry::from_instruction(log.current_pc, instr, 4);
        let f = op.decode.fields;
        if !f.word_instr {
            if f.is_lt() {
                *host_map
                    .entry(LtOperation::new_with_invert(
                        op.rv1,
                        op.arg2,
                        f.alu_signed(),
                        f.alu_signed2_or_invert(),
                    ))
                    .or_insert(0) += 1;
            }
            if f.is_divrem() {
                let dv = DvrmOperation::new(op.rv1, op.arg2, f.alu_signed());
                *host_map
                    .entry(LtOperation::new(dv.abs_r(), dv.abs_d(), false))
                    .or_insert(0) += 1;
            }
        }
        packed.push(d.fields.pack());
        rv1.push(op.rv1);
        arg2.push(op.arg2);
    }

    let n_unique = host_map.len();
    let num_rows = n_unique.next_power_of_two().max(4);
    let ncols = math_cuda::trace_cpu::LT_NCOLS;
    let mut host_packed = Vec::with_capacity(n_unique * math_cuda::trace_cpu::LT_STRIDE);
    for (op, mult) in &host_map {
        host_packed.extend_from_slice(&pack_lt_op(op, *mult));
    }
    let host_table = math_cuda::trace_cpu::gpu_build_lt_trace_host(&host_packed, n_unique, num_rows)
        .expect("host lt fill");
    let res_table =
        math_cuda::trace_ops::gpu_build_lt_instr_dvrm_resident(&packed, &rv1, &arg2, num_rows)
            .expect("resident merged lt");

    assert_eq!(host_table.len(), res_table.len(), "table size");
    let host_ms = row_multiset(&host_table, ncols);
    let res_ms = row_multiset(&res_table, ncols);
    assert_eq!(host_ms.len(), res_ms.len(), "distinct-row count");
    for (row, &hc) in &host_ms {
        assert_eq!(res_ms.get(row), Some(&hc), "row multiplicity mismatch");
    }
    println!(
        "gpu_lt_instr_dvrm_resident OK: device merges instruction+dvrm LT sources → \
         {n_unique} unique rows, multiset-identical to host"
    );
}

#[test]
#[ignore = "requires GPU + LAMBDA_VM_BENCH_ELF (ethrex); run --ignored --nocapture"]
fn gpu_eq_resident_matches_host_path() {
    if let Err(e) = math_cuda::device::backend() {
        eprintln!("skipping gpu_eq_resident_matches_host_path: no CUDA backend: {e:?}");
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
    let (mut packed, mut rv1, mut arg2) =
        (Vec::with_capacity(n), Vec::with_capacity(n), Vec::with_capacity(n));
    let mut host_map: HashMap<EqOperation, u64> = HashMap::new();
    for (i, log) in result.logs.iter().enumerate() {
        let instr = *instructions.get(&log.current_pc).expect("instruction");
        let op = CpuOperation::from_log_and_instruction(log, (i as u64) * 4 + 4, instr);
        let d = DecodeEntry::from_instruction(log.current_pc, instr, 4);
        let f = op.decode.fields;
        if !f.word_instr && f.is_eq() {
            let eq = EqOperation::new(op.rv1, op.arg2, f.alu_signed2_or_invert());
            *host_map.entry(eq).or_insert(0) += 1;
        }
        packed.push(d.fields.pack());
        rv1.push(op.rv1);
        arg2.push(op.arg2);
    }

    let n_unique = host_map.len();
    let num_rows = n_unique.next_power_of_two().max(4);
    let ncols = math_cuda::trace_cpu::EQ_NCOLS;
    let mut host_packed = Vec::with_capacity(n_unique * math_cuda::trace_cpu::EQ_STRIDE);
    for (op, mult) in &host_map {
        host_packed.extend_from_slice(&pack_eq_op(op, *mult));
    }
    let host_table = math_cuda::trace_cpu::gpu_build_eq_trace_host(&host_packed, n_unique, num_rows)
        .expect("host eq fill");
    let res_table = math_cuda::trace_ops::gpu_build_eq_resident(&packed, &rv1, &arg2, num_rows)
        .expect("resident eq");

    assert_eq!(host_table.len(), res_table.len(), "table size");
    let host_ms = row_multiset(&host_table, ncols);
    let res_ms = row_multiset(&res_table, ncols);
    assert_eq!(host_ms.len(), res_ms.len(), "distinct-row count");
    for (row, &hc) in &host_ms {
        assert_eq!(res_ms.get(row), Some(&hc), "row multiplicity mismatch");
    }
    println!(
        "gpu_eq_resident OK: device→device EQ chain multiset-identical to host path \
         ({n_unique} unique rows, {num_rows} padded)"
    );
}

#[test]
#[ignore = "requires GPU + LAMBDA_VM_BENCH_ELF (ethrex); run --ignored --nocapture"]
fn gpu_bytewise_resident_matches_host_path() {
    if let Err(e) = math_cuda::device::backend() {
        eprintln!("skipping gpu_bytewise_resident_matches_host_path: no CUDA backend: {e:?}");
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
    let (mut packed, mut rv1, mut arg2) =
        (Vec::with_capacity(n), Vec::with_capacity(n), Vec::with_capacity(n));
    let mut host_map: HashMap<BytewiseOperation, u64> = HashMap::new();
    for (i, log) in result.logs.iter().enumerate() {
        let instr = *instructions.get(&log.current_pc).expect("instruction");
        let op = CpuOperation::from_log_and_instruction(log, (i as u64) * 4 + 4, instr);
        let d = DecodeEntry::from_instruction(log.current_pc, instr, 4);
        let f = op.decode.fields;
        if !f.word_instr && (f.is_and() || f.is_or() || f.is_xor()) {
            let b = BytewiseOperation::new(op.rv1, op.arg2, f.alu_op());
            *host_map.entry(b).or_insert(0) += 1;
        }
        packed.push(d.fields.pack());
        rv1.push(op.rv1);
        arg2.push(op.arg2);
    }

    let n_unique = host_map.len();
    let num_rows = n_unique.next_power_of_two().max(4);
    let ncols = math_cuda::trace_cpu::BYTEWISE_NCOLS;
    let mut host_packed = Vec::with_capacity(n_unique * math_cuda::trace_cpu::BYTEWISE_STRIDE);
    for (op, mult) in &host_map {
        host_packed.extend_from_slice(&pack_bytewise_op(op, *mult));
    }
    let host_table =
        math_cuda::trace_cpu::gpu_build_bytewise_trace_host(&host_packed, n_unique, num_rows)
            .expect("host bytewise fill");
    let res_table =
        math_cuda::trace_ops::gpu_build_bytewise_resident(&packed, &rv1, &arg2, num_rows)
            .expect("resident bytewise");

    assert_eq!(host_table.len(), res_table.len(), "table size");
    let host_ms = row_multiset(&host_table, ncols);
    let res_ms = row_multiset(&res_table, ncols);
    assert_eq!(host_ms.len(), res_ms.len(), "distinct-row count");
    for (row, &hc) in &host_ms {
        assert_eq!(res_ms.get(row), Some(&hc), "row multiplicity mismatch");
    }
    println!(
        "gpu_bytewise_resident OK: device→device BYTEWISE chain multiset-identical to host path \
         ({n_unique} unique rows, {num_rows} padded)"
    );
}
