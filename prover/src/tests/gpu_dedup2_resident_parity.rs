//! Resident chains for the remaining chips: MUL/DVRM (dual multiplicity mu_lo/mu_hi,
//! mu_q/mu_r), SHIFT (per-row, no dedup), BRANCH (4-field key → dedup4). Each device→device
//! table must match the host path as a MULTISET of rows (order-independent LogUp buses).
//! Covers the instruction-driven op source (see the scope note in the plan). ethrex_5tx.
//!
//! `LAMBDA_VM_BENCH_ELF=.../rust/ethrex.elf LAMBDA_VM_BENCH_INPUT=.../ethrex_5tx.bin \
//!   cargo test -p lambda-vm-prover --release --features cuda --lib gpu_dedup2_resident -- --ignored --nocapture`

use std::collections::HashMap;
use std::env;
use std::fs;

use executor::elf::Elf;
use executor::vm::execution::Executor;

use crate::tables::branch::BranchOperation;
use crate::tables::cpu::CpuOperation;
use crate::tables::decode;
use crate::tables::dvrm::DvrmOperation;
use crate::tables::gpu_trace::{pack_branch_op, pack_dvrm_op, pack_mul_op, pack_shift_op};
use crate::tables::mul::MulOperation;
use crate::tables::shift::ShiftOperation;
use crate::tables::types::DecodeEntry;

fn row_multiset(table: &[u64], ncols: usize) -> HashMap<Vec<u64>, usize> {
    let mut m = HashMap::new();
    for row in table.chunks_exact(ncols) {
        *m.entry(row.to_vec()).or_insert(0) += 1;
    }
    m
}

struct Run {
    elf: Elf,
    input: Vec<u8>,
}
fn setup() -> Option<Run> {
    if let Err(e) = math_cuda::device::backend() {
        eprintln!("skipping: no CUDA backend: {e:?}");
        return None;
    }
    let path = env::var("LAMBDA_VM_BENCH_ELF").expect("set LAMBDA_VM_BENCH_ELF (ethrex.elf)");
    let bytes = fs::read(&path).expect("read ELF");
    let elf = Elf::load(&bytes).expect("load ELF");
    let input = env::var("LAMBDA_VM_BENCH_INPUT")
        .ok()
        .map(|p| fs::read(p).expect("read input"))
        .unwrap_or_default();
    Some(Run { elf, input })
}

fn assert_multiset_eq(host: &[u64], res: &[u64], ncols: usize, name: &str) {
    assert_eq!(host.len(), res.len(), "{name}: table size");
    let hm = row_multiset(host, ncols);
    let rm = row_multiset(res, ncols);
    assert_eq!(hm.len(), rm.len(), "{name}: distinct-row count");
    for (row, &hc) in &hm {
        assert_eq!(rm.get(row), Some(&hc), "{name}: row multiplicity mismatch");
    }
}

#[test]
#[ignore = "requires GPU + LAMBDA_VM_BENCH_ELF (ethrex); run --ignored --nocapture"]
fn gpu_dedup2_resident_all() {
    let Some(run) = setup() else { return };
    let executor = Executor::new(&run.elf, run.input.clone()).expect("executor");
    let result = executor.run().expect("run");
    let instructions = decode::instructions_from_elf(&run.elf).expect("decode");

    let n = result.logs.len();
    let (mut packed, mut rv1, mut arg2, mut pc, mut imm, mut flags) = (
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    );
    let mut mul_map: HashMap<MulOperation, (u64, u64)> = HashMap::new();
    let mut dvrm_map: HashMap<DvrmOperation, (u64, u64)> = HashMap::new();
    let mut branch_map: HashMap<BranchOperation, u64> = HashMap::new();
    let mut shift_ops: Vec<ShiftOperation> = Vec::new();
    for (i, log) in result.logs.iter().enumerate() {
        let instr = *instructions.get(&log.current_pc).expect("instruction");
        let op = CpuOperation::from_log_and_instruction(log, (i as u64) * 4 + 4, instr);
        let d = DecodeEntry::from_instruction(log.current_pc, instr, 4);
        let f = op.decode.fields;
        if !f.word_instr {
            if f.is_mul() {
                let m = MulOperation::new(op.rv1, f.alu_signed(), op.arg2, f.alu_signed2_or_invert());
                let e = mul_map.entry(m).or_insert((0, 0));
                if f.alu_muldiv() { e.1 += 1 } else { e.0 += 1 }
            }
            if f.is_divrem() {
                let dv = DvrmOperation::new(op.rv1, op.arg2, f.alu_signed());
                let e = dvrm_map.entry(dv).or_insert((0, 0));
                if f.alu_muldiv() { e.1 += 1 } else { e.0 += 1 }
            }
            if f.is_shift() {
                shift_ops.push(ShiftOperation::new(
                    op.rv1,
                    op.arg2,
                    f.alu_signed2_or_invert(),
                    f.alu_signed(),
                    f.word_instr,
                ));
            }
        }
        if op.branch_cond {
            *branch_map
                .entry(BranchOperation::new(d.pc, d.imm, op.rv1, f.jalr()))
                .or_insert(0) += 1;
        }
        packed.push(d.fields.pack());
        rv1.push(op.rv1);
        arg2.push(op.arg2);
        pc.push(d.pc);
        imm.push(d.imm);
        flags.push(op.branch_cond as u8);
    }

    // MUL
    {
        let nu = mul_map.len();
        let num_rows = nu.next_power_of_two().max(4);
        let mut hp = Vec::with_capacity(nu * math_cuda::trace_cpu::MUL_STRIDE);
        for (op, (lo, hi)) in &mul_map {
            hp.extend_from_slice(&pack_mul_op(op, *lo, *hi));
        }
        let host = math_cuda::trace_cpu::gpu_build_mul_trace_host(&hp, nu, num_rows).expect("mul host");
        let res = math_cuda::trace_ops::gpu_build_mul_resident(&packed, &rv1, &arg2, num_rows)
            .expect("mul resident");
        assert_multiset_eq(&host, &res, math_cuda::trace_cpu::MUL_NCOLS, "MUL");
        println!("  MUL resident OK ({nu} unique rows)");
    }
    // DVRM
    {
        let nu = dvrm_map.len();
        let num_rows = nu.next_power_of_two().max(4);
        let mut hp = Vec::with_capacity(nu * math_cuda::trace_cpu::DVRM_STRIDE);
        for (op, (q, r)) in &dvrm_map {
            hp.extend_from_slice(&pack_dvrm_op(op, *q, *r));
        }
        let host = math_cuda::trace_cpu::gpu_build_dvrm_trace_host(&hp, nu, num_rows).expect("dvrm host");
        let res = math_cuda::trace_ops::gpu_build_dvrm_resident(&packed, &rv1, &arg2, num_rows)
            .expect("dvrm resident");
        assert_multiset_eq(&host, &res, math_cuda::trace_cpu::DVRM_NCOLS, "DVRM");
        println!("  DVRM resident OK ({nu} unique rows)");
    }
    // SHIFT (per-row)
    {
        let nrows = shift_ops.len();
        let num_rows = nrows.next_power_of_two().max(4);
        let mut hp = Vec::with_capacity(nrows * math_cuda::trace_cpu::SHIFT_STRIDE);
        for op in &shift_ops {
            hp.extend_from_slice(&pack_shift_op(op));
        }
        let host =
            math_cuda::trace_cpu::gpu_build_shift_trace_host(&hp, nrows, num_rows).expect("shift host");
        let res = math_cuda::trace_ops::gpu_build_shift_resident(&packed, &rv1, &arg2, num_rows)
            .expect("shift resident");
        assert_multiset_eq(&host, &res, math_cuda::trace_cpu::SHIFT_NCOLS, "SHIFT");
        println!("  SHIFT resident OK ({nrows} rows)");
    }
    // BRANCH (dedup4)
    {
        let nu = branch_map.len();
        let num_rows = nu.next_power_of_two().max(4);
        let mut hp = Vec::with_capacity(nu * math_cuda::trace_cpu::BRANCH_STRIDE);
        for (op, mult) in &branch_map {
            hp.extend_from_slice(&pack_branch_op(op, *mult));
        }
        let host = math_cuda::trace_cpu::gpu_build_branch_trace_host(&hp, nu, num_rows)
            .expect("branch host");
        let res =
            math_cuda::trace_ops::gpu_build_branch_resident(&packed, &flags, &pc, &imm, &rv1, num_rows)
                .expect("branch resident");
        assert_multiset_eq(&host, &res, math_cuda::trace_cpu::BRANCH_NCOLS, "BRANCH");
        println!("  BRANCH resident OK ({nu} unique rows)");
    }
    println!("gpu_dedup2_resident_all OK: MUL/DVRM/SHIFT/BRANCH device→device multiset-identical to host");
}

/// MUL with a MERGED source: instruction-driven MUL ⊕ dvrm-derived MUL (each is_divrem cycle
/// adds `MulOperation::new(d, d_signed, q, q_signed)` to both mu_lo and mu_hi). Device
/// `gpu_build_mul_instr_dvrm_resident` must match the host merge as a multiset of rows.
#[test]
#[ignore = "requires GPU + LAMBDA_VM_BENCH_ELF (ethrex); run --ignored --nocapture"]
fn gpu_mul_instr_dvrm_resident() {
    let Some(run) = setup() else { return };
    let executor = Executor::new(&run.elf, run.input.clone()).expect("executor");
    let result = executor.run().expect("run");
    let instructions = decode::instructions_from_elf(&run.elf).expect("decode");

    let n = result.logs.len();
    let (mut packed, mut rv1, mut arg2) =
        (Vec::with_capacity(n), Vec::with_capacity(n), Vec::with_capacity(n));
    let mut mul_map: HashMap<MulOperation, (u64, u64)> = HashMap::new();
    for (i, log) in result.logs.iter().enumerate() {
        let instr = *instructions.get(&log.current_pc).expect("instruction");
        let op = CpuOperation::from_log_and_instruction(log, (i as u64) * 4 + 4, instr);
        let d = DecodeEntry::from_instruction(log.current_pc, instr, 4);
        let f = op.decode.fields;
        if !f.word_instr {
            if f.is_mul() {
                let m = MulOperation::new(op.rv1, f.alu_signed(), op.arg2, f.alu_signed2_or_invert());
                let e = mul_map.entry(m).or_insert((0, 0));
                if f.alu_muldiv() { e.1 += 1 } else { e.0 += 1 }
            }
            if f.is_divrem() {
                let dv = DvrmOperation::new(op.rv1, op.arg2, f.alu_signed());
                let m = MulOperation::new(op.arg2, f.alu_signed(), dv.compute_quotient(), dv.sign_q());
                let e = mul_map.entry(m).or_insert((0, 0));
                e.0 += 1; // C13 lo
                e.1 += 1; // C14 hi
            }
        }
        packed.push(d.fields.pack());
        rv1.push(op.rv1);
        arg2.push(op.arg2);
    }

    let nu = mul_map.len();
    let num_rows = nu.next_power_of_two().max(4);
    let mut hp = Vec::with_capacity(nu * math_cuda::trace_cpu::MUL_STRIDE);
    for (op, (lo, hi)) in &mul_map {
        hp.extend_from_slice(&pack_mul_op(op, *lo, *hi));
    }
    let host = math_cuda::trace_cpu::gpu_build_mul_trace_host(&hp, nu, num_rows).expect("mul host");
    let res = math_cuda::trace_ops::gpu_build_mul_instr_dvrm_resident(&packed, &rv1, &arg2, num_rows)
        .expect("mul merged resident");
    assert_multiset_eq(&host, &res, math_cuda::trace_cpu::MUL_NCOLS, "MUL+dvrm");
    println!("gpu_mul_instr_dvrm_resident OK: device merges instruction+dvrm MUL → {nu} unique rows");
}

/// SHIFT with a MERGED source: instruction-driven SHIFT (word=0) ⊕ cpu32-derived SHIFT
/// (word instructions via `cpu32_chip_op`, word=1). Per-row; device
/// `gpu_build_shift_full_resident` must match the host merge as a multiset of rows.
#[test]
#[ignore = "requires GPU + LAMBDA_VM_BENCH_ELF (ethrex); run --ignored --nocapture"]
fn gpu_shift_full_resident() {
    let Some(run) = setup() else { return };
    let executor = Executor::new(&run.elf, run.input.clone()).expect("executor");
    let result = executor.run().expect("run");
    let instructions = decode::instructions_from_elf(&run.elf).expect("decode");

    let n = result.logs.len();
    let (mut packed, mut rv1, mut rv2, mut arg2, mut imm) = (
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    );
    // Host: instruction shifts, then cpu32-derived shifts (matches trace_builder order).
    let mut shift_ops: Vec<ShiftOperation> = Vec::new();
    let mut cpu32_shifts: Vec<ShiftOperation> = Vec::new();
    for (i, log) in result.logs.iter().enumerate() {
        let instr = *instructions.get(&log.current_pc).expect("instruction");
        let op = CpuOperation::from_log_and_instruction(log, (i as u64) * 4 + 4, instr);
        let d = DecodeEntry::from_instruction(log.current_pc, instr, 4);
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
            let c = crate::tables::trace_builder::build_cpu32_op(&op);
            let (mut m, mut dv) = (Vec::new(), Vec::new());
            crate::tables::trace_builder::cpu32_chip_op(&c, &mut cpu32_shifts, &mut m, &mut dv);
        }
        packed.push(d.fields.pack());
        rv1.push(op.rv1);
        rv2.push(op.rv2);
        arg2.push(op.arg2);
        imm.push(op.decode.imm);
    }
    shift_ops.extend(cpu32_shifts);

    let nrows = shift_ops.len();
    let num_rows = nrows.next_power_of_two().max(4);
    let mut hp = Vec::with_capacity(nrows * math_cuda::trace_cpu::SHIFT_STRIDE);
    for op in &shift_ops {
        hp.extend_from_slice(&pack_shift_op(op));
    }
    let host = math_cuda::trace_cpu::gpu_build_shift_trace_host(&hp, nrows, num_rows).expect("shift host");
    let res =
        math_cuda::trace_ops::gpu_build_shift_full_resident(&packed, &rv1, &rv2, &arg2, &imm, num_rows)
            .expect("shift full resident");
    assert_multiset_eq(&host, &res, math_cuda::trace_cpu::SHIFT_NCOLS, "SHIFT+cpu32");
    println!("gpu_shift_full_resident OK: device merges instruction+cpu32 SHIFT → {nrows} rows");
}

/// COMPLETE production DVRM table on device: DVRM's only two sources are instruction-driven
/// and cpu32-derived (no dvrm-from-other), so `gpu_build_dvrm_full_resident` reproduces the
/// full table. Also validates MUL's 3-source merge (instruction ⊕ instruction-dvrm-derived ⊕
/// cpu32) — honestly a subset of production MUL (missing the dvrm→mul from cpu32-dvrm).
#[test]
#[ignore = "requires GPU + LAMBDA_VM_BENCH_ELF (ethrex); run --ignored --nocapture"]
fn gpu_dvrm_full_and_mul_3source_resident() {
    let Some(run) = setup() else { return };
    let executor = Executor::new(&run.elf, run.input.clone()).expect("executor");
    let result = executor.run().expect("run");
    let instructions = decode::instructions_from_elf(&run.elf).expect("decode");

    let n = result.logs.len();
    let (mut packed, mut rv1, mut rv2, mut arg2, mut imm) = (
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    );
    let mut dvrm_map: HashMap<DvrmOperation, (u64, u64)> = HashMap::new();
    let mut mul_map: HashMap<MulOperation, (u64, u64)> = HashMap::new();
    for (i, log) in result.logs.iter().enumerate() {
        let instr = *instructions.get(&log.current_pc).expect("instruction");
        let op = CpuOperation::from_log_and_instruction(log, (i as u64) * 4 + 4, instr);
        let d = DecodeEntry::from_instruction(log.current_pc, instr, 4);
        let f = op.decode.fields;
        if !f.word_instr {
            if f.is_mul() {
                let m = MulOperation::new(op.rv1, f.alu_signed(), op.arg2, f.alu_signed2_or_invert());
                let e = mul_map.entry(m).or_insert((0, 0));
                if f.alu_muldiv() { e.1 += 1 } else { e.0 += 1 }
            }
            if f.is_divrem() {
                let dv = DvrmOperation::new(op.rv1, op.arg2, f.alu_signed());
                let e = dvrm_map.entry(dv.clone()).or_insert((0, 0));
                if f.alu_muldiv() { e.1 += 1 } else { e.0 += 1 }
                // dvrm→mul (C13 lo + C14 hi) from this instruction dvrm op.
                let m = MulOperation::new(op.arg2, f.alu_signed(), dv.compute_quotient(), dv.sign_q());
                let e2 = mul_map.entry(m).or_insert((0, 0));
                e2.0 += 1;
                e2.1 += 1;
            }
        }
        if f.word_instr {
            let c = crate::tables::trace_builder::build_cpu32_op(&op);
            let (mut s, mut m, mut dv) = (Vec::new(), Vec::new(), Vec::new());
            crate::tables::trace_builder::cpu32_chip_op(&c, &mut s, &mut m, &mut dv);
            for (mop, muldiv) in m {
                let e = mul_map.entry(mop).or_insert((0, 0));
                if muldiv { e.1 += 1 } else { e.0 += 1 }
            }
            for (dop, muldiv) in dv {
                // dvrm→mul (C13 lo + C14 hi) from this cpu32-derived dvrm op — MUL's 4th source.
                let mm = MulOperation::new(dop.d, dop.signed, dop.compute_quotient(), dop.sign_q());
                let em = mul_map.entry(mm).or_insert((0, 0));
                em.0 += 1;
                em.1 += 1;
                let e = dvrm_map.entry(dop).or_insert((0, 0));
                if muldiv { e.1 += 1 } else { e.0 += 1 }
            }
        }
        packed.push(d.fields.pack());
        rv1.push(op.rv1);
        rv2.push(op.rv2);
        arg2.push(op.arg2);
        imm.push(op.decode.imm);
    }

    // DVRM (complete production table).
    {
        let nu = dvrm_map.len();
        let num_rows = nu.next_power_of_two().max(4);
        let mut hp = Vec::with_capacity(nu * math_cuda::trace_cpu::DVRM_STRIDE);
        for (op, (q, r)) in &dvrm_map {
            hp.extend_from_slice(&pack_dvrm_op(op, *q, *r));
        }
        let host = math_cuda::trace_cpu::gpu_build_dvrm_trace_host(&hp, nu, num_rows).expect("dvrm host");
        let res = math_cuda::trace_ops::gpu_build_dvrm_full_resident(&packed, &rv1, &rv2, &arg2, &imm, num_rows)
            .expect("dvrm full resident");
        assert_multiset_eq(&host, &res, math_cuda::trace_cpu::DVRM_NCOLS, "DVRM-full");
        println!("  DVRM FULL (instruction+cpu32 = complete production table) OK ({nu} unique rows)");
    }
    // MUL (COMPLETE — all 4 sources: instruction + instruction-dvrm + cpu32 + cpu32-dvrm).
    {
        let nu = mul_map.len();
        let num_rows = nu.next_power_of_two().max(4);
        let mut hp = Vec::with_capacity(nu * math_cuda::trace_cpu::MUL_STRIDE);
        for (op, (lo, hi)) in &mul_map {
            hp.extend_from_slice(&pack_mul_op(op, *lo, *hi));
        }
        let host = math_cuda::trace_cpu::gpu_build_mul_trace_host(&hp, nu, num_rows).expect("mul host");
        let res = math_cuda::trace_ops::gpu_build_mul_full_resident(&packed, &rv1, &rv2, &arg2, &imm, num_rows)
            .expect("mul full resident");
        assert_multiset_eq(&host, &res, math_cuda::trace_cpu::MUL_NCOLS, "MUL-full");
        println!("  MUL FULL (all 4 sources = complete production table) OK ({nu} unique rows)");
    }
    println!("gpu_dvrm_full_and_mul_resident OK");
}
