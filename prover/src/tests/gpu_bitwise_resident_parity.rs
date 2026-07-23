//! RESIDENT BITWISE in-walk histogram parity: the per-CPU-op range-check bumps scattered straight
//! from the device-resident `packed`+`res` buffers (`gpu_bitwise_hist_in_walk_devbuf`, no host SoA
//! rebuild) must be BIN-FOR-BIN identical to the existing host-SoA GPU path (`gpu_bitwise_hist`
//! with memw empty), which is itself validated against the CPU `collect_bitwise_ops`. This is the
//! first source of the resident BITWISE Phase-4 histogram — the keystone that lets the host
//! `p2a_collect` eventually be dropped. ethrex_5tx.
//!
//! `LAMBDA_VM_BENCH_ELF=.../rust/ethrex.elf LAMBDA_VM_BENCH_INPUT=.../ethrex_5tx.bin \
//!   cargo test -p lambda-vm-prover --release --features cuda --lib gpu_bitwise_resident -- --ignored --nocapture`

use std::env;
use std::fs;

use executor::elf::Elf;
use executor::vm::execution::Executor;

use crate::tables::bitwise;
use crate::tables::cpu::CpuOperation;
use crate::tables::decode;
use crate::tables::types::DecodeEntry;

#[test]
#[ignore = "requires GPU + LAMBDA_VM_BENCH_ELF (ethrex); run --ignored --nocapture"]
fn gpu_bitwise_in_walk_resident_matches_host_soa() {
    if math_cuda::device::backend().is_err() {
        eprintln!("skipping: no CUDA backend");
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
    // Host CpuOpFields SoA (the reference path's input) + the devops SoA (the resident input).
    let (mut rs1, mut rs2, mut rd, mut hil, mut alu, mut mem, mut res, mut word) = (
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    );
    let (mut packed, mut imm, mut pc, mut rv1, mut rv2, mut arg2, mut rvd, mut flags) = (
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::<u8>::with_capacity(n),
    );
    for (i, log) in result.logs.iter().enumerate() {
        let instr = *instructions.get(&log.current_pc).expect("instruction");
        let op = CpuOperation::from_log_and_instruction(log, (i as u64) * 4 + 4, instr);
        let d = DecodeEntry::from_instruction(log.current_pc, instr, 4);
        let f = &op.decode.fields;
        rs1.push(f.rs1);
        rs2.push(f.rs2);
        rd.push(f.rd);
        hil.push(f.half_instruction_length);
        alu.push(f.alu_flags);
        mem.push(f.mem_flags);
        res.push(op.res);
        word.push(u8::from(f.word_instr));
        packed.push(d.fields.pack());
        imm.push(d.imm);
        pc.push(d.pc);
        rv1.push(op.rv1);
        rv2.push(op.rv2);
        arg2.push(op.arg2);
        rvd.push(op.rvd);
        flags.push(op.branch_cond as u8);
    }

    let num_rows = bitwise::NUM_ROWS;
    let num_types = bitwise::NUM_LOOKUP_TYPES;

    // Reference: existing host-SoA GPU path, in-walk only (memw empty) — validated vs CPU.
    let host_fields = math_cuda::bitwise_hist::CpuOpFields {
        rs1: &rs1,
        rs2: &rs2,
        rd: &rd,
        hil: &hil,
        alu_flags: &alu,
        mem_flags: &mem,
        res: &res,
        word: &word,
    };
    let reference =
        math_cuda::bitwise_hist::gpu_bitwise_hist(&host_fields, &[], &[], num_rows, num_types)
            .expect("host-soa hist");

    // Resident: scatter straight from the device-resident packed+res (ONE upload, no SoA rebuild).
    let devops = math_cuda::trace_ops::gpu_upload_cpu_ops_resident(
        &packed, &imm, &pc, &rv1, &rv2, &arg2, &res, &rvd, &flags,
    )
    .expect("upload devops");
    let resident = math_cuda::bitwise_hist::gpu_bitwise_hist_in_walk_devbuf(
        &devops.packed,
        &devops.res,
        n,
        num_rows,
        num_types,
    )
    .expect("resident hist");

    assert_eq!(reference.len(), resident.len(), "counter array length");
    assert_eq!(reference, resident, "resident in-walk histogram != host-SoA path");
    let bumps: u64 = resident.iter().sum();
    println!(
        "gpu_bitwise_in_walk_resident OK: {n} cpu_ops → {bumps} in-walk BITWISE bumps, \
         bin-for-bin identical to host-SoA GPU path, ZERO host SoA rebuild/upload"
    );
}

/// PAGE-source histogram parity: the GPU `bitwise_hist_page` scatter (one ARE_BYTES[init,fini] per
/// byte) must be bin-for-bin identical to a direct CPU scatter of the same init/fini bytes. Second
/// feeder of the all-GPU counting table. Self-contained (synthetic init/fini); requires a GPU.
#[test]
#[ignore = "requires GPU; run --ignored --nocapture"]
fn gpu_bitwise_page_scatter_matches_cpu() {
    if math_cuda::device::backend().is_err() {
        eprintln!("skipping: no CUDA backend");
        return;
    }
    let num_rows = bitwise::NUM_ROWS;
    let num_types = bitwise::NUM_LOOKUP_TYPES;
    // Synthetic per-byte init/fini (well-mixed, full 0..255 range, incl init==fini "untouched").
    let n = 300_000usize;
    let mut init = Vec::with_capacity(n);
    let mut fini = Vec::with_capacity(n);
    for i in 0..n {
        let a = (i.wrapping_mul(2_654_435_761) & 0xFF) as u8;
        let b = if i % 5 == 0 { a } else { (i.wrapping_mul(40_503) >> 3 & 0xFF) as u8 };
        init.push(a);
        fini.push(b);
    }

    let mut cpu = vec![0u64; num_rows * num_types];
    for i in 0..n {
        cpu[3 * num_rows + init[i] as usize + fini[i] as usize * 256] += 1;
    }

    let gpu = math_cuda::bitwise_hist::gpu_bitwise_hist_page_only(&init, &fini, num_rows, num_types)
        .expect("gpu page scatter");
    assert_eq!(gpu.len(), cpu.len());
    assert_eq!(gpu, cpu, "GPU page scatter != CPU reference");
    println!("gpu_bitwise_page_scatter OK: {n} ARE_BYTES[init,fini] bumps bin-for-bin identical to CPU");
}

/// MEMW_R-source histogram parity (P3): the masked device scatter (one IS_HALF per emitting register
/// row, `row_index >= 0`, keyed by ts_lo - old_ts_lo - 1) must be bin-for-bin identical to a direct
/// CPU scatter over the same ts/old_ts/row_index. Reads the resident-walk stream format; non-emitting
/// PC writes (row_index=-1) contribute nothing. Self-contained; requires a GPU.
#[test]
#[ignore = "requires GPU; run --ignored --nocapture"]
fn gpu_bitwise_memw_reg_masked_matches_cpu() {
    if math_cuda::device::backend().is_err() {
        eprintln!("skipping: no CUDA backend");
        return;
    }
    let num_rows = bitwise::NUM_ROWS;
    let num_types = bitwise::NUM_LOOKUP_TYPES;
    let n = 500_000usize;
    let (mut ts, mut old_ts, mut row_index) =
        (Vec::with_capacity(n), Vec::with_capacity(n), Vec::with_capacity(n));
    for i in 0..n {
        let t = (i as u64) * 4 + 100;
        // old_ts a bounded delta below ts; a few large-delta rows exercise the u16 wrap.
        let delta = (i as u64 % 300) + 1;
        ts.push(t);
        old_ts.push(t.wrapping_sub(delta));
        // Every ~4th access is a non-emitting PC write (row_index = -1).
        row_index.push(if i % 4 == 3 { -1 } else { (i as i64) });
    }

    let mut cpu = vec![0u64; num_rows * num_types];
    for i in 0..n {
        if row_index[i] < 0 {
            continue;
        }
        let diff = ((ts[i] as u32).wrapping_sub(old_ts[i] as u32).wrapping_sub(1) & 0xFFFF) as usize;
        cpu[4 * num_rows + diff] += 1;
    }

    let gpu = math_cuda::bitwise_hist::gpu_bitwise_hist_memw_reg_masked(
        &ts, &old_ts, &row_index, num_rows, num_types,
    )
    .expect("gpu masked memw_reg scatter");
    assert_eq!(gpu.len(), cpu.len());
    assert_eq!(gpu, cpu, "GPU masked memw_reg scatter != CPU reference");
    let emitting = row_index.iter().filter(|&&r| r >= 0).count();
    println!("gpu_bitwise_memw_reg_masked OK: {emitting} emitting IS_HALF bumps bin-for-bin identical to CPU");
}

/// MEMW_ALIGNED op-vec source parity (P4): one IS_HALF[base_low + mask(width)] per aligned op,
/// bin-for-bin identical to a direct CPU scatter over the same base/width/aligned. Self-contained.
#[test]
#[ignore = "requires GPU; run --ignored --nocapture"]
fn gpu_bitwise_memw_aligned_matches_cpu() {
    if math_cuda::device::backend().is_err() {
        eprintln!("skipping: no CUDA backend");
        return;
    }
    let num_rows = bitwise::NUM_ROWS;
    let num_types = bitwise::NUM_LOOKUP_TYPES;
    let n = 400_000usize;
    let widths = [2u32, 4, 8, 1];
    let (mut base, mut width, mut aligned) =
        (Vec::with_capacity(n), Vec::with_capacity(n), Vec::with_capacity(n));
    for i in 0..n {
        let w = widths[i % 4];
        // Aligned base (multiple of w) so base_low + mask stays within the halfword.
        base.push((i as u64 % 0x2000) * (w as u64).max(1) + 0x1_0000 * (i as u64 % 7));
        width.push(w);
        aligned.push(if i % 6 == 5 { 0 } else { 1 }); // some general (skipped)
    }

    let mut cpu = vec![0u64; num_rows * num_types];
    for i in 0..n {
        if aligned[i] == 0 {
            continue;
        }
        let mask: u64 = match width[i] {
            2 => 1,
            4 => 3,
            8 => 7,
            _ => 0,
        };
        let v = ((base[i] & 0xFFFF) + mask) & 0xFFFF;
        cpu[4 * num_rows + v as usize] += 1;
    }

    let gpu =
        math_cuda::bitwise_hist::gpu_bitwise_hist_memw_aligned(&base, &width, &aligned, num_rows, num_types)
            .expect("gpu memw_aligned scatter");
    assert_eq!(gpu, cpu, "GPU memw_aligned scatter != CPU reference");
    let cnt = aligned.iter().filter(|&&a| a != 0).count();
    println!("gpu_bitwise_memw_aligned OK: {cnt} aligned IS_HALF bumps bin-for-bin identical to CPU");
}
