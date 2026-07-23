//! Resident cpu_ops seam: building the CPU32 table from the device-resident cpu_ops
//! ([`math_cuda::trace_ops::gpu_build_cpu_ops_resident`] → `gpu_build_cpu32_resident_from_devops`,
//! a single log-SoA upload then everything on device, no per-chip re-upload) must be byte-
//! identical to the host-input resident path (`gpu_build_cpu32_resident`). Proves the
//! "one upload → device cpu_ops → chips read in place" architecture. ethrex_5tx.
//!
//! `LAMBDA_VM_BENCH_ELF=.../rust/ethrex.elf LAMBDA_VM_BENCH_INPUT=.../ethrex_5tx.bin \
//!   cargo test -p lambda-vm-prover --release --features cuda --lib gpu_resident_seam -- --ignored --nocapture`

use std::env;
use std::fs;

use executor::elf::Elf;
use executor::vm::execution::Executor;

use crate::tables::cpu::CpuOperation;
use crate::tables::decode;
use crate::tables::types::DecodeEntry;

#[test]
#[ignore = "requires GPU + LAMBDA_VM_BENCH_ELF (ethrex); run --ignored --nocapture"]
fn gpu_cpu_ops_resident_seam_cpu32() {
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

    let n = result.logs.len();
    // Log SoA (the one-time upload) + decode SoA + host cpu_op fields (for the host-input path).
    let (mut cpc, mut npc, mut s1, mut s2, mut dv, mut pc, mut imm, mut packed) = (
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    );
    let (mut rv1, mut rv2, mut res, mut rvd, mut arg2, mut flags) = (
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
        cpc.push(log.current_pc);
        npc.push(log.next_pc);
        s1.push(log.src1_val);
        s2.push(log.src2_val);
        dv.push(log.dst_val);
        pc.push(d.pc);
        imm.push(d.imm);
        packed.push(d.fields.pack());
        rv1.push(op.rv1);
        rv2.push(op.rv2);
        res.push(op.res);
        rvd.push(op.rvd);
        arg2.push(op.arg2);
        flags.push(op.branch_cond as u8);
    }

    // Resident seam: ONE upload of the log+decode SoA → device cpu_ops → every chip reads them
    // in place (no per-chip re-upload). Build CPU32 + LOAD + STORE from the SAME devops.
    let devops = math_cuda::trace_ops::gpu_build_cpu_ops_resident(
        &cpc, &npc, &s1, &s2, &dv, &pc, &imm, &packed,
    )
    .expect("resident cpu_ops");
    let stream = be.next_stream();
    let dl = |buf| stream.clone_dtoh(buf).expect("dtoh");

    // CPU32
    let (cpu32_buf, cpu32_nr) =
        math_cuda::trace_ops::gpu_build_cpu32_resident_from_devops(&devops).expect("cpu32");
    let cpu32_host =
        math_cuda::trace_ops::gpu_build_cpu32_resident(&packed, &rv1, &rv2, &imm, &pc, cpu32_nr)
            .expect("cpu32 host");
    assert_eq!(dl(&cpu32_buf), cpu32_host, "CPU32 resident-seam != host-input");

    // LOAD
    let (load_buf, load_nr) =
        math_cuda::trace_ops::gpu_build_load_resident_from_devops(&devops).expect("load");
    let load_host =
        math_cuda::trace_ops::gpu_build_load_resident(&packed, &res, &rvd, load_nr).expect("load host");
    assert_eq!(dl(&load_buf), load_host, "LOAD resident-seam != host-input");

    // STORE
    let (store_buf, store_nr) =
        math_cuda::trace_ops::gpu_build_store_resident_from_devops(&devops).expect("store");
    let store_host =
        math_cuda::trace_ops::gpu_build_store_resident(&packed, &res, &rv2, store_nr).expect("store host");
    assert_eq!(dl(&store_buf), store_host, "STORE resident-seam != host-input");

    // EQ (deduped): resident-seam reads devops.{packed,rv1,arg2} in place. Deduped output is
    // SORTED (device radix) vs the host HashMap order → compare as a MULTISET.
    let (eq_buf, eq_nr) =
        math_cuda::trace_ops::gpu_build_eq_resident_from_devops(&devops).expect("eq");
    let eq_host =
        math_cuda::trace_ops::gpu_build_eq_resident_dev(&packed, &rv1, &arg2).expect("eq host");
    assert_eq!(eq_nr, eq_host.1, "EQ resident-seam row count != host-input");
    let mut eq_seam = dl(&eq_buf);
    let mut eq_hbuf = dl(&eq_host.0);
    eq_seam.sort_unstable();
    eq_hbuf.sort_unstable();
    assert_eq!(eq_seam, eq_hbuf, "EQ resident-seam multiset != host-input");

    // BYTEWISE (deduped): same seam, same multiset check.
    let (bw_buf, bw_nr) =
        math_cuda::trace_ops::gpu_build_bytewise_resident_from_devops(&devops).expect("bytewise");
    let bw_host =
        math_cuda::trace_ops::gpu_build_bytewise_resident_dev(&packed, &rv1, &arg2).expect("bw host");
    assert_eq!(bw_nr, bw_host.1, "BYTEWISE resident-seam row count != host-input");
    let mut bw_seam = dl(&bw_buf);
    let mut bw_hbuf = dl(&bw_host.0);
    bw_seam.sort_unstable();
    bw_hbuf.sort_unstable();
    assert_eq!(bw_seam, bw_hbuf, "BYTEWISE resident-seam multiset != host-input");

    // SHIFT (per-row, instruction ⊕ cpu32 in fixed source order → byte-identical). Autosizes
    // from the device row count; the host path is pinned to that same height.
    let (shift_buf, shift_nr) =
        math_cuda::trace_ops::gpu_build_shift_full_resident_from_devops(&devops).expect("shift");
    let shift_host =
        math_cuda::trace_ops::gpu_build_shift_full_resident_dev(&packed, &rv1, &rv2, &arg2, &imm, shift_nr)
            .expect("shift host");
    assert_eq!(dl(&shift_buf), dl(&shift_host), "SHIFT resident-seam != host-input");

    // MUL (deduped, four sources merged): multiset check.
    let (mul_buf, mul_nr) =
        math_cuda::trace_ops::gpu_build_mul_full_resident_from_devops(&devops).expect("mul");
    let mul_host =
        math_cuda::trace_ops::gpu_build_mul_full_resident_dev(&packed, &rv1, &rv2, &arg2, &imm).expect("mul host");
    assert_eq!(mul_nr, mul_host.1, "MUL resident-seam row count != host-input");
    let mut mul_seam = dl(&mul_buf);
    let mut mul_hbuf = dl(&mul_host.0);
    mul_seam.sort_unstable();
    mul_hbuf.sort_unstable();
    assert_eq!(mul_seam, mul_hbuf, "MUL resident-seam multiset != host-input");

    // DVRM (deduped, instruction ⊕ cpu32): multiset check.
    let (dvrm_buf, dvrm_nr) =
        math_cuda::trace_ops::gpu_build_dvrm_full_resident_from_devops(&devops).expect("dvrm");
    let dvrm_host =
        math_cuda::trace_ops::gpu_build_dvrm_full_resident_dev(&packed, &rv1, &rv2, &arg2, &imm).expect("dvrm host");
    assert_eq!(dvrm_nr, dvrm_host.1, "DVRM resident-seam row count != host-input");
    let mut dvrm_seam = dl(&dvrm_buf);
    let mut dvrm_hbuf = dl(&dvrm_host.0);
    dvrm_seam.sort_unstable();
    dvrm_hbuf.sort_unstable();
    assert_eq!(dvrm_seam, dvrm_hbuf, "DVRM resident-seam multiset != host-input");

    // BRANCH (deduped, 4-key): multiset check.
    let (br_buf, br_nr) =
        math_cuda::trace_ops::gpu_build_branch_resident_from_devops(&devops).expect("branch");
    let br_host =
        math_cuda::trace_ops::gpu_build_branch_resident_dev(&packed, &flags, &pc, &imm, &rv1).expect("branch host");
    assert_eq!(br_nr, br_host.1, "BRANCH resident-seam row count != host-input");
    let mut br_seam = dl(&br_buf);
    let mut br_hbuf = dl(&br_host.0);
    br_seam.sort_unstable();
    br_hbuf.sort_unstable();
    assert_eq!(br_seam, br_hbuf, "BRANCH resident-seam multiset != host-input");

    stream.synchronize().expect("sync");
    println!(
        "gpu_cpu_ops_resident_seam OK: ONE log-SoA upload → device cpu_ops → \
         CPU32 ({cpu32_nr}) + LOAD ({load_nr}) + STORE ({store_nr}) + SHIFT ({shift_nr}) \
         [per-row, byte-identical] + EQ ({eq_nr}) + BYTEWISE ({bw_nr}) + MUL ({mul_nr}) + \
         DVRM ({dvrm_nr}) + BRANCH ({br_nr}) [deduped, multiset-identical] — all 9 chips from \
         ONE upload, zero per-chip re-uploads"
    );
}
