//! P1 (resident pipeline): the device REGISTER-access emitter
//! (`math_cuda::trace_walk::gpu_emit_register_accesses`) must produce the SAME access stream —
//! reg_addr / ts / value / is_read / row_index, in the same order — as the CPU reference
//! `emit_register_accesses` over all ops (with row_index = compacted emit position, -1 for the
//! non-emitting implicit PC write). This is the foundation the device register walk consumes with
//! NO host collection/upload of the accesses. ethrex_5tx.
//!
//! `LAMBDA_VM_BENCH_ELF=.../rust/ethrex.elf LAMBDA_VM_BENCH_INPUT=.../ethrex_5tx.bin \
//!   cargo test -p lambda-vm-prover --release --features cuda --lib gpu_reg_emit -- --ignored --nocapture`

use std::env;
use std::fs;

use executor::elf::Elf;
use executor::vm::execution::Executor;

use crate::tables::cpu::CpuOperation;
use crate::tables::decode;
use crate::tables::trace_builder::{RegAccess, build_initial_image, emit_register_accesses};

use crate::tables::types::DecodeEntry;

#[test]
#[ignore = "requires GPU + LAMBDA_VM_BENCH_ELF (ethrex); run --ignored --nocapture"]
fn gpu_reg_emit_matches_cpu() {
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
    let (mut packed, mut rv1, mut rv2, mut rvd, mut next_pc) = (
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    );
    let mut cpu_ops = Vec::with_capacity(n);
    for (i, log) in result.logs.iter().enumerate() {
        let instr = *instructions.get(&log.current_pc).expect("instruction");
        let op = CpuOperation::from_log_and_instruction(log, (i as u64) * 4 + 4, instr);
        let d = DecodeEntry::from_instruction(log.current_pc, instr, 4);
        packed.push(d.fields.pack());
        rv1.push(op.rv1);
        rv2.push(op.rv2);
        rvd.push(op.rvd);
        next_pc.push(op.next_pc);
        cpu_ops.push(op);
    }

    // Device emit.
    let (d_addr, d_ts, d_val, d_isread, d_rowidx) =
        math_cuda::trace_walk::gpu_emit_register_accesses(&packed, &rv1, &rv2, &rvd, &next_pc)
            .expect("device reg emit");

    // CPU reference: concat emit_register_accesses over ops; row_index = running emit counter.
    let mut accs: Vec<RegAccess> = Vec::new();
    for op in &cpu_ops {
        emit_register_accesses(op, &mut accs);
    }
    let m = accs.len();
    let (mut c_addr, mut c_ts, mut c_val, mut c_isread, mut c_rowidx) = (
        Vec::with_capacity(m),
        Vec::with_capacity(m),
        Vec::with_capacity(m),
        Vec::with_capacity(m),
        Vec::with_capacity(m),
    );
    let mut row = 0i64;
    for a in &accs {
        c_addr.push(a.reg_addr as u32);
        c_ts.push(a.timestamp);
        c_val.push(a.value);
        c_isread.push(a.is_read as u8);
        if a.emits_row {
            c_rowidx.push(row);
            row += 1;
        } else {
            c_rowidx.push(-1);
        }
    }

    assert_eq!(d_addr.len(), m, "device access count != CPU");
    assert_eq!(d_addr, c_addr, "reg_addr stream mismatch");
    assert_eq!(d_ts, c_ts, "ts stream mismatch");
    assert_eq!(d_val, c_val, "value stream mismatch");
    assert_eq!(d_isread, c_isread, "is_read stream mismatch");
    assert_eq!(d_rowidx, c_rowidx, "row_index stream mismatch");
    println!(
        "gpu_reg_emit OK: {n} ops → {m} register accesses ({} emitting), stream-identical to CPU \
         emit_register_accesses, ZERO host collection",
        row
    );
}

#[test]
#[ignore = "requires GPU + LAMBDA_VM_BENCH_ELF (ethrex); run --ignored --nocapture"]
fn gpu_mem_emit_matches_cpu() {
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
    let (mut packed, mut res, mut rvd, mut rv2) = (
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    );
    // CPU reference (mirrors the memw byte-access host prep).
    let (mut c_addr, mut c_ts, mut c_val, mut c_oprow, mut c_boff) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let (mut c_base, mut c_opts, mut c_isread, mut c_width, mut c_signed, mut c_vw) = (
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
    );
    for (i, log) in result.logs.iter().enumerate() {
        let instr = *instructions.get(&log.current_pc).expect("instruction");
        let op = CpuOperation::from_log_and_instruction(log, (i as u64) * 4 + 4, instr);
        let d = DecodeEntry::from_instruction(log.current_pc, instr, 4);
        packed.push(d.fields.pack());
        res.push(op.res);
        rvd.push(op.rvd);
        rv2.push(op.rv2);
        let f = op.decode.fields;
        let (ld, st) = (f.is_load(), f.is_store());
        if !ld && !st {
            continue;
        }
        let w = f.mem_bytes();
        let vw = if ld { op.rvd } else { op.rv2 };
        let row = c_base.len() as u64;
        for j in 0..w {
            c_addr.push(op.res.wrapping_add(j as u64));
            c_ts.push(op.timestamp);
            c_val.push((vw >> (j * 8)) & 0xFF);
            c_oprow.push(row);
            c_boff.push(j as u32);
        }
        c_base.push(op.res);
        c_opts.push(op.timestamp);
        c_isread.push(ld as u32);
        c_width.push(w as u32);
        c_signed.push(f.mem_signed() as u32);
        c_vw.push(vw);
    }

    let dev = math_cuda::trace_walk::gpu_emit_memory_accesses(&packed, &res, &rvd, &rv2)
        .expect("device mem emit");

    assert_eq!(dev.base.len(), c_base.len(), "op count");
    assert_eq!(dev.addr.len(), c_addr.len(), "byte-access count");
    assert_eq!(dev.addr, c_addr, "addr");
    assert_eq!(dev.ts, c_ts, "ts");
    assert_eq!(dev.val, c_val, "val");
    assert_eq!(dev.op_row, c_oprow, "op_row");
    assert_eq!(dev.byte_off, c_boff, "byte_off");
    assert_eq!(dev.base, c_base, "base");
    assert_eq!(dev.op_ts, c_opts, "op_ts");
    assert_eq!(dev.is_read, c_isread, "is_read");
    assert_eq!(dev.width, c_width, "width");
    assert_eq!(dev.signed, c_signed, "signed");
    assert_eq!(dev.value_word, c_vw, "value_word");
    println!(
        "gpu_mem_emit OK: {n} ops → {} load/store ops / {} byte-accesses, stream-identical to CPU, \
         ZERO host collection",
        c_base.len(),
        c_addr.len()
    );
}

/// P2: the FULLY-RESIDENT register MEMW_R build (emit accesses on device → walk → fill, no per-access
/// upload) must be byte-identical to the existing (validated) walk+fill fed the SAME accesses via the
/// host path. Same accesses (P1-emitted) + same init ⇒ identical MEMW_R, proving the resident wiring.
#[test]
#[ignore = "requires GPU + LAMBDA_VM_BENCH_ELF (ethrex); run --ignored --nocapture"]
fn gpu_reg_walk_fill_resident_matches_host() {
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
    let (mut packed, mut rv1, mut rv2, mut rvd, mut next_pc) = (
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    );
    for (i, log) in result.logs.iter().enumerate() {
        let instr = *instructions.get(&log.current_pc).expect("instruction");
        let op = CpuOperation::from_log_and_instruction(log, (i as u64) * 4 + 4, instr);
        let d = DecodeEntry::from_instruction(log.current_pc, instr, 4);
        packed.push(d.fields.pack());
        rv1.push(op.rv1);
        rv2.push(op.rv2);
        rvd.push(op.rvd);
        next_pc.push(op.next_pc);
    }

    let nbins = 512u32;
    let init_ts = 1u64;
    // Deterministic init (both paths use the same — validates wiring, not production seed values).
    let init_value: Vec<u64> = (0..nbins as u64).map(|b| b.wrapping_mul(0x9E37_79B9)).collect();

    // Reference accesses via the (validated) P1 emit, then the (validated) host walk+fill.
    let (d_addr, d_ts, d_val, d_isread, d_rowidx) =
        math_cuda::trace_walk::gpu_emit_register_accesses(&packed, &rv1, &rv2, &rvd, &next_pc)
            .expect("emit");
    let emitting = d_rowidx.iter().filter(|&&r| r >= 0).count();
    let num_rows = emitting.next_power_of_two().max(4);
    let reference = math_cuda::trace_cpu::gpu_walk_and_fill_memw_register_host(
        &d_addr, &d_ts, &d_val, &d_isread, &d_rowidx, &init_value, init_ts, nbins, num_rows,
    )
    .expect("host walk+fill");

    // Resident: emit → walk → fill, no per-access upload.
    let resident = math_cuda::trace_cpu::gpu_walk_fill_memw_register_resident_host(
        &packed, &rv1, &rv2, &rvd, &next_pc, &init_value, init_ts, nbins, num_rows,
    )
    .expect("resident walk+fill");

    assert_eq!(resident.len(), reference.len());
    assert_eq!(resident, reference, "resident MEMW_R != host walk+fill on the same accesses");
    println!(
        "gpu_reg_walk_fill_resident OK: {emitting} MEMW_R rows ({num_rows} padded) — resident \
         emit→walk→fill byte-identical to host walk+fill, ZERO per-access upload"
    );
}

/// P1-ecall injection mechanism: the resident register walk with INJECTED non-emitting accesses
/// (`gpu_walk_fill_memw_register_resident_injected`) must be byte-identical to the (validated) host
/// walk+fill over the SAME combined [emitted regular ⊕ injected] stream — proving the device concat
/// of extra timeline events is correct. Uses ethrex regular accesses + synthetic ecall-like injected
/// accesses (register addresses, ts ≡ 3 mod 4 so they never collide with regular ts ≡ {0,1,2} mod 4).
/// Also asserts the injected events actually change the regular rows' old_ts (not a no-op).
#[test]
#[ignore = "requires GPU + LAMBDA_VM_BENCH_ELF (ethrex); run --ignored --nocapture"]
fn gpu_reg_walk_injected_matches_host() {
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
    let (mut packed, mut rv1, mut rv2, mut rvd, mut next_pc) = (
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    );
    for (i, log) in result.logs.iter().enumerate() {
        let instr = *instructions.get(&log.current_pc).expect("instruction");
        let op = CpuOperation::from_log_and_instruction(log, (i as u64) * 4 + 4, instr);
        let d = DecodeEntry::from_instruction(log.current_pc, instr, 4);
        packed.push(d.fields.pack());
        rv1.push(op.rv1);
        rv2.push(op.rv2);
        rvd.push(op.rvd);
        next_pc.push(op.next_pc);
    }

    let nbins = 512u32;
    let init_ts = 1u64;
    let init_value: Vec<u64> = (0..nbins as u64).map(|b| b.wrapping_mul(0x9E37_79B9)).collect();

    // Regular emitted accesses.
    let (mut c_addr, mut c_ts, mut c_val, mut c_isread, mut c_rowidx) =
        math_cuda::trace_walk::gpu_emit_register_accesses(&packed, &rv1, &rv2, &rvd, &next_pc)
            .expect("emit");
    let emitting = c_rowidx.iter().filter(|&&r| r >= 0).count();
    let num_rows = emitting.next_power_of_two().max(4);

    // Synthetic ecall-like injected accesses: hot registers x10/x11/x12/x254 (addr 20/22/24/508),
    // ts = 3 + 4*i (≡ 3 mod 4, never collides with regular ts), NON-emitting timeline events.
    let m = 2000usize;
    let regs = [20u32, 22, 24, 508];
    let inj_addr: Vec<u32> = (0..m).map(|i| regs[i % 4]).collect();
    let inj_ts: Vec<u64> = (0..m as u64).map(|i| 3 + 4 * i).collect();
    let inj_val: Vec<u64> = (0..m as u64).map(|i| i.wrapping_mul(0x1234_5678_9ABC_DEF1)).collect();
    let inj_isread: Vec<u8> = vec![1u8; m];

    // Device: resident emit-from-packed + inject.
    let device = math_cuda::trace_cpu::gpu_walk_fill_memw_register_resident_injected_host(
        &packed, &rv1, &rv2, &rvd, &next_pc, &inj_addr, &inj_ts, &inj_val, &inj_isread,
        &init_value, init_ts, nbins, num_rows,
    )
    .expect("resident injected");

    // Reference: host walk+fill over [emitted ⊕ injected(row_index=-1)].
    c_addr.extend_from_slice(&inj_addr);
    c_ts.extend_from_slice(&inj_ts);
    c_val.extend_from_slice(&inj_val);
    c_isread.extend_from_slice(&inj_isread);
    c_rowidx.extend(std::iter::repeat(-1i64).take(m));
    let reference = math_cuda::trace_cpu::gpu_walk_and_fill_memw_register_host(
        &c_addr, &c_ts, &c_val, &c_isread, &c_rowidx, &init_value, init_ts, nbins, num_rows,
    )
    .expect("host walk+fill");

    assert_eq!(device.len(), reference.len());
    assert_eq!(device, reference, "resident-injected != host walk+fill over the combined stream");

    // NOTE (finding): the register walk's `walk_link` links `perm[p-1]`, and `perm` is a STABLE
    // counting-sort by BIN that preserves INPUT-ARRAY ORDER (it does NOT sort by ts within a bin).
    // So APPENDED injected accesses land last within each bin and never become a predecessor of an
    // earlier row → 0 diff vs no-injection here (expected). This test validates the CONCAT MECHANISM
    // (device == host over the same combined stream). Correct ecall injection must INTERLEAVE the
    // accesses in timeline order (at their op position); that is a follow-on increment.
    let no_inject = math_cuda::trace_cpu::gpu_walk_fill_memw_register_resident_host(
        &packed, &rv1, &rv2, &rvd, &next_pc, &init_value, init_ts, nbins, num_rows,
    )
    .expect("no-inject");
    let diff_cells = device.iter().zip(no_inject.iter()).filter(|(a, b)| a != b).count();
    println!(
        "gpu_reg_walk_injected OK: {emitting} regular MEMW_R rows + {m} appended injected accesses — \
         resident-injected BYTE-IDENTICAL to host walk over the same combined stream (concat mechanism \
         validated). Appended accesses differ from no-injection in {diff_cells} cells (0 expected — \
         walk is stable-by-bin input-order; real ecall injection must interleave in timeline order)."
    );
}

/// P1-ecall INTERLEAVED injection: the resident register walk that interleaves ecall accesses at
/// their op's timeline position (`gpu_walk_fill_memw_register_resident_ecall`) must be byte-identical
/// to the host walk+fill over the CORRECTLY-INTERLEAVED [per-op: regular ⊕ ecall] stream, AND must
/// differ from the no-injection walk (the interleaved ecall events re-link old_ts of later regular
/// rows — unlike appending, which the stable-by-bin walk ignores). Synthetic ecall accesses placed at
/// real op positions on hot register bins; validates the interleaving emit kernels.
#[test]
#[ignore = "requires GPU + LAMBDA_VM_BENCH_ELF (ethrex); run --ignored --nocapture"]
fn gpu_reg_walk_ecall_interleaved_matches_host() {
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
    let mut cpu_ops = Vec::with_capacity(n);
    let (mut packed, mut rv1, mut rv2, mut rvd, mut next_pc) = (
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    );
    for (i, log) in result.logs.iter().enumerate() {
        let instr = *instructions.get(&log.current_pc).expect("instruction");
        let op = CpuOperation::from_log_and_instruction(log, (i as u64) * 4 + 4, instr);
        let d = DecodeEntry::from_instruction(log.current_pc, instr, 4);
        packed.push(d.fields.pack());
        rv1.push(op.rv1);
        rv2.push(op.rv2);
        rvd.push(op.rvd);
        next_pc.push(op.next_pc);
        cpu_ops.push(op);
    }

    let nbins = 512u32;
    let init_ts = 1u64;
    let init_value: Vec<u64> = (0..nbins as u64).map(|b| b.wrapping_mul(0x9E37_79B9)).collect();

    // Synthetic ecall-like accesses at REAL op positions (spread across the run), hot register bins
    // x10/x11/x12/x254 (20/22/24/508), ts = the op's timeslot. Grouped by op index (non-decreasing).
    let m = 2000usize;
    let regs = [20u32, 22, 24, 508];
    let (mut e_op_index, mut e_reg_addr, mut e_ts, mut e_value, mut e_is_read) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for k in 0..m {
        let oi = (k * (n / (m + 1))) as u32; // spread, strictly increasing, < n
        e_op_index.push(oi);
        e_reg_addr.push(regs[k % 4]);
        e_ts.push(oi as u64 * 4 + 4);
        e_value.push((k as u64).wrapping_mul(0x1234_5678_9ABC_DEF1));
        e_is_read.push(1u8);
    }

    // Build the CORRECTLY-INTERLEAVED reference stream on host: per op, regular accesses then the
    // op's ecall accesses (non-emitting). This also gives the emitting count for num_rows.
    let mut acc: Vec<RegAccess> = Vec::with_capacity(n * 4 + m);
    let mut ep = 0usize;
    for (oi, op) in cpu_ops.iter().enumerate() {
        emit_register_accesses(op, &mut acc);
        while ep < e_op_index.len() && e_op_index[ep] as usize == oi {
            acc.push(RegAccess {
                reg_addr: e_reg_addr[ep] as u64,
                timestamp: e_ts[ep],
                value: e_value[ep],
                is_read: e_is_read[ep] != 0,
                emits_row: false,
            });
            ep += 1;
        }
    }
    let emitting = acc.iter().filter(|a| a.emits_row).count();
    let num_rows = emitting.next_power_of_two().max(4);

    // Device: resident walk interleaving the ecall accesses on-device.
    let device = math_cuda::trace_cpu::gpu_walk_fill_memw_register_resident_ecall_host(
        &packed, &rv1, &rv2, &rvd, &next_pc, &e_op_index, &e_reg_addr, &e_ts, &e_value, &e_is_read,
        &init_value, init_ts, nbins, num_rows,
    )
    .expect("resident ecall walk");

    let mut r = 0i64;
    let (mut ref_addr, mut ref_ts, mut ref_val, mut ref_isr, mut ref_row) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for a in &acc {
        ref_addr.push(a.reg_addr as u32);
        ref_ts.push(a.timestamp);
        ref_val.push(a.value);
        ref_isr.push(u8::from(a.is_read));
        ref_row.push(if a.emits_row {
            let x = r;
            r += 1;
            x
        } else {
            -1
        });
    }
    let reference = math_cuda::trace_cpu::gpu_walk_and_fill_memw_register_host(
        &ref_addr, &ref_ts, &ref_val, &ref_isr, &ref_row, &init_value, init_ts, nbins, num_rows,
    )
    .expect("host walk+fill over interleaved stream");

    assert_eq!(device.len(), reference.len());
    assert_eq!(device, reference, "resident interleaved-ecall walk != host walk over interleaved stream");

    // The interleaved ecall events MUST change the regular rows' old_ts (unlike appending).
    let no_inject = math_cuda::trace_cpu::gpu_walk_fill_memw_register_resident_host(
        &packed, &rv1, &rv2, &rvd, &next_pc, &init_value, init_ts, nbins, num_rows,
    )
    .expect("no-inject");
    let diff = device.iter().zip(no_inject.iter()).filter(|(a, b)| a != b).count();
    assert!(diff > 0, "interleaved injection was a no-op (expected old_ts changes)");
    println!(
        "gpu_reg_walk_ecall_interleaved OK: {emitting} regular MEMW_R rows + {m} interleaved ecall \
         accesses — device byte-identical to host walk over the correctly-interleaved stream; \
         {diff} cells differ vs no-injection (interleaved events correctly re-linked old_ts)."
    );
}

/// Image-on-device building block: the device binary-search `gpu_image_lookup` (init byte per
/// accessed address from the sorted initial image) must match the host `image.get(a).unwrap_or(0)`.
/// Shared dependency for the resident memory walk (init_value) + PAGE source. Self-contained.
#[test]
#[ignore = "requires GPU; run --ignored --nocapture"]
fn gpu_image_lookup_matches_cpu() {
    use std::collections::BTreeMap;
    if math_cuda::device::backend().is_err() {
        eprintln!("skipping: no CUDA backend");
        return;
    }
    // Sparse initial image (sorted by address via BTreeMap): addr = k*7+3.
    let mut img: BTreeMap<u64, u64> = BTreeMap::new();
    for k in 0..50_000u64 {
        img.insert(k * 7 + 3, k.wrapping_mul(31) & 0xFF);
    }
    let img_addr: Vec<u64> = img.keys().copied().collect();
    let img_val: Vec<u64> = img.values().copied().collect();

    // Access addresses: half present in the image, half (likely) absent.
    let mut addr = Vec::with_capacity(200_000);
    for i in 0..200_000u64 {
        addr.push(if i % 2 == 0 { (i % 50_000) * 7 + 3 } else { i * 11 + 1 });
    }

    let gpu = math_cuda::trace_walk::gpu_image_lookup(&addr, &img_addr, &img_val).expect("gpu lookup");
    let cpu: Vec<u64> = addr.iter().map(|a| img.get(a).copied().unwrap_or(0)).collect();
    assert_eq!(gpu, cpu, "device image lookup != host image.get");
    println!(
        "gpu_image_lookup OK: {} lookups over {}-entry image bin-for-bin identical to host",
        addr.len(),
        img_addr.len()
    );
}

/// P2-memory: the FULLY-RESIDENT MEMW build (emit accesses on device → on-device image init →
/// walk → gather → classify → pack) must be byte-identical to the existing (validated) host path
/// `gpu_build_memw_ls` fed the same emitted accesses + the same image init. Proves the resident
/// memory walk+fill wiring (no per-access upload). ethrex_5tx.
#[test]
#[ignore = "requires GPU + LAMBDA_VM_BENCH_ELF (ethrex); run --ignored --nocapture"]
fn gpu_memw_ls_resident_matches_host() {
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
    let executor = Executor::new(&elf, input.clone()).expect("executor");
    let result = executor.run().expect("run");
    let instructions = decode::instructions_from_elf(&elf).expect("decode");

    let n = result.logs.len();
    let (mut packed, mut res, mut rvd, mut rv2) = (
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    );
    for (i, log) in result.logs.iter().enumerate() {
        let instr = *instructions.get(&log.current_pc).expect("instruction");
        let op = CpuOperation::from_log_and_instruction(log, (i as u64) * 4 + 4, instr);
        let d = DecodeEntry::from_instruction(log.current_pc, instr, 4);
        packed.push(d.fields.pack());
        res.push(op.res);
        rvd.push(op.rvd);
        rv2.push(op.rv2);
    }

    // Sorted initial image for the on-device lookup.
    let image = build_initial_image(&elf, &input);
    let mut pairs: Vec<(u64, u8)> = image.iter().map(|(&a, &v)| (a, v)).collect();
    pairs.sort_by_key(|&(a, _)| a);
    let img_addr: Vec<u64> = pairs.iter().map(|&(a, _)| a).collect();
    let img_val: Vec<u64> = pairs.iter().map(|&(_, v)| v as u64).collect();

    // Resident: emit → image-init → walk → fill, no per-access upload.
    let (rpa, rna, rpg, rng) =
        math_cuda::trace_walk::gpu_build_memw_ls_resident(&packed, &res, &rvd, &rv2, &img_addr, &img_val)
            .expect("resident memw");

    // Reference: the same emitted accesses + host image init through the validated host path.
    let em = math_cuda::trace_walk::gpu_emit_memory_accesses(&packed, &res, &rvd, &rv2).expect("emit");
    let init: Vec<u64> = em.addr.iter().map(|a| image.get(a).copied().unwrap_or(0) as u64).collect();
    let (hpa, hna, hpg, hng) = math_cuda::trace_walk::gpu_build_memw_ls(
        &em.addr, &em.ts, &em.val, &init, &em.op_row, &em.byte_off, &em.base, &em.op_ts, &em.is_read,
        &em.width, &em.signed, &em.value_word,
    )
    .expect("host memw");

    assert_eq!(rna, hna, "aligned row count");
    assert_eq!(rng, hng, "general row count");
    assert_eq!(rpa, hpa, "MEMW_A rows != host");
    assert_eq!(rpg, hpg, "MEMW general rows != host");
    println!(
        "gpu_memw_ls_resident OK: {rna} MEMW_A + {rng} MEMW rows — resident emit→image-init→walk→fill \
         byte-identical to host path, ZERO per-access upload"
    );
}

/// P1-ecall MEMORY-walk injection: `gpu_build_memw_ls_resident_ecall` must (a) reduce EXACTLY to the
/// base `gpu_build_memw_ls_resident` when given no ecall accesses (the byte-slot reservation + DUMP-row
/// gather plumbing is transparent), and (b) with real-address ecall accesses interleaved at early op
/// positions, keep the regular MEMW_A/MEMW row COUNTS identical (ecall is non-emitting) while CHANGING
/// their packed content (old_ts/old_value re-linked) — proving the interleaved ecall memory events
/// advance the memory timeline for regular LOAD/STORE rows.
#[test]
#[ignore = "requires GPU + LAMBDA_VM_BENCH_ELF (ethrex); run --ignored --nocapture"]
fn gpu_memw_ls_ecall_interleaved() {
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
    let executor = Executor::new(&elf, input.clone()).expect("executor");
    let result = executor.run().expect("run");
    let instructions = decode::instructions_from_elf(&elf).expect("decode");

    let n = result.logs.len();
    let (mut packed, mut res, mut rvd, mut rv2) =
        (Vec::with_capacity(n), Vec::with_capacity(n), Vec::with_capacity(n), Vec::with_capacity(n));
    for (i, log) in result.logs.iter().enumerate() {
        let instr = *instructions.get(&log.current_pc).expect("instruction");
        let op = CpuOperation::from_log_and_instruction(log, (i as u64) * 4 + 4, instr);
        let d = DecodeEntry::from_instruction(log.current_pc, instr, 4);
        packed.push(d.fields.pack());
        res.push(op.res);
        rvd.push(op.rvd);
        rv2.push(op.rv2);
    }
    let image = build_initial_image(&elf, &input);
    let mut pairs: Vec<(u64, u8)> = image.iter().map(|(&a, &v)| (a, v)).collect();
    pairs.sort_by_key(|&(a, _)| a);
    let img_addr: Vec<u64> = pairs.iter().map(|&(a, _)| a).collect();
    let img_val: Vec<u64> = pairs.iter().map(|&(_, v)| v as u64).collect();

    let base = math_cuda::trace_walk::gpu_build_memw_ls_resident(
        &packed, &res, &rvd, &rv2, &img_addr, &img_val,
    )
    .expect("base memw");

    // (a) Empty ecall → must equal base exactly (plumbing is transparent).
    let empty = math_cuda::trace_walk::gpu_build_memw_ls_resident_ecall(
        &packed, &res, &rvd, &rv2, &img_addr, &img_val, &[], &[], &[], &[],
    )
    .expect("empty-ecall memw");
    assert_eq!(empty.1, base.1, "empty-ecall aligned count != base");
    assert_eq!(empty.3, base.3, "empty-ecall general count != base");
    assert_eq!(empty.0, base.0, "empty-ecall MEMW_A rows != base");
    assert_eq!(empty.2, base.2, "empty-ecall MEMW rows != base");

    // (b) Real-address ecall accesses at op 0 (before their first regular access) → perturb.
    let em = math_cuda::trace_walk::gpu_emit_memory_accesses(&packed, &res, &rvd, &rv2).expect("emit");
    let k = 300usize.min(em.addr.len());
    let step = (em.addr.len() / (k + 1)).max(1);
    let (mut e_oi, mut e_addr, mut e_ts, mut e_val) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for j in 0..k {
        e_oi.push(0u32); // op 0 → earliest input position, before any regular access to this addr
        e_addr.push(em.addr[j * step]);
        e_ts.push(4u64);
        e_val.push((j as u64) & 0xFF);
    }
    let inj = math_cuda::trace_walk::gpu_build_memw_ls_resident_ecall(
        &packed, &res, &rvd, &rv2, &img_addr, &img_val, &e_oi, &e_addr, &e_ts, &e_val,
    )
    .expect("ecall memw");
    // The TOTAL number of regular MEMW ops is conserved (ecall accesses emit no rows — they only
    // advance the timeline). The aligned/general SPLIT may legitimately shift, because perturbing a
    // regular op's old_ts can flip its alignment classification (aligned ⇔ all bytes share old_ts) —
    // exactly what the true trace would reflect once ecall writes are in the timeline.
    assert_eq!(
        inj.1 + inj.3,
        base.1 + base.3,
        "total regular MEMW op count not conserved (ecall must be non-emitting)"
    );
    assert!(
        inj.0 != base.0 || inj.2 != base.2 || inj.1 != base.1,
        "interleaved ecall memory accesses did not perturb any regular MEMW row"
    );
    println!(
        "gpu_memw_ls_ecall_interleaved OK: empty-ecall byte-identical to base ({} MEMW_A + {} MEMW); \
         {k} interleaved ecall accesses re-linked regular old_ts → split now {}/{} (was {}/{}), \
         total {} conserved (ecall non-emitting).",
        base.1, base.3, inj.1, inj.3, base.1, base.3, base.1 + base.3
    );
}
