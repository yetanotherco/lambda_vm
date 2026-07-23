//! Phase-2 MEMW routing: the device walk+gather ([`math_cuda::trace_walk::gpu_mem_walk_memw`])
//! must build each LOAD/STORE op's MEMW row `(old_timestamp[0..width], old_value[0..width])`
//! — and hence the aligned/general classification — identically to the sequential
//! `MemoryState` walk. Exercised over the real guest's (ethrex_5tx) LOAD+STORE stream, seeded
//! from the initial image. (Ecall memory accesses are Phase-6; register accesses are the
//! separate register walk. This validates the memory-walk → MEMW-row connection.)
//!
//! `LAMBDA_VM_BENCH_ELF=.../rust/ethrex.elf LAMBDA_VM_BENCH_INPUT=.../ethrex_5tx.bin \
//!   cargo test -p lambda-vm-prover --release --features cuda --lib gpu_memw_routing -- --ignored --nocapture`

use std::collections::HashMap;
use std::env;
use std::fs;

use executor::elf::Elf;
use executor::vm::execution::Executor;

use crate::tables::cpu::CpuOperation;
use crate::tables::decode;
use crate::tables::trace_builder::build_initial_image;

/// is_aligned_op over a per-op old_timestamp[0..width] slice (mirrors trace_builder.rs).
fn is_aligned(base: u64, width: usize, old_ts: &[u64]) -> bool {
    let low = (base & 0xFFFF_FFFF) as u32;
    let w = width as u32;
    if w > 1 && (low & (w - 1)) != 0 {
        return false;
    }
    for i in 1..width {
        if old_ts[i] != old_ts[0] {
            return false;
        }
    }
    true
}

#[test]
#[ignore = "requires GPU + LAMBDA_VM_BENCH_ELF (ethrex); run --ignored --nocapture"]
fn gpu_memw_routing_matches_reference() {
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
    let executor = Executor::new(&elf, input.clone()).expect("executor");
    let result = executor.run().expect("run");
    let instructions = decode::instructions_from_elf(&elf).expect("decode");

    // Emit per-byte accesses for LOAD/STORE ops, in program order, with per-op mapping.
    let mut addr = Vec::new();
    let mut ts = Vec::new();
    let mut value = Vec::new();
    let mut op_row = Vec::new();
    let mut byte_off = Vec::new();
    let mut op_base = Vec::new(); // per op: base address
    let mut op_width = Vec::new(); // per op: byte_count
    for (i, log) in result.logs.iter().enumerate() {
        let instr = *instructions.get(&log.current_pc).expect("instruction");
        let op = CpuOperation::from_log_and_instruction(log, (i as u64) * 4 + 4, instr);
        let f = op.decode.fields;
        let (is_load, is_store) = (f.is_load(), f.is_store());
        if !is_load && !is_store {
            continue;
        }
        let base = op.res;
        let nbytes = f.mem_bytes();
        let word = if is_load { op.rvd } else { op.rv2 };
        let row = op_base.len() as u64;
        for j in 0..nbytes {
            addr.push(base.wrapping_add(j as u64));
            ts.push(op.timestamp);
            value.push((word >> (j * 8)) & 0xFF);
            op_row.push(row);
            byte_off.push(j as u32);
        }
        op_base.push(base);
        op_width.push(nbytes);
    }
    let num_ops = op_base.len();
    let n_acc = addr.len();

    // Initial image seed.
    let image = build_initial_image(&elf, &input);
    let init: Vec<u64> = addr.iter().map(|a| image.get(a).copied().unwrap_or(0) as u64).collect();

    // Reference: sequential walk (image-seeded), gathered per-op.
    let mut state: HashMap<u64, (u64, u64)> = HashMap::new();
    for (&a, &b) in &image {
        state.insert(a, (b as u64, 0));
    }
    let mut ref_ts = vec![0u64; num_ops * 8];
    let mut ref_val = vec![0u64; num_ops * 8];
    for k in 0..n_acc {
        let (ov, ot) = state.get(&addr[k]).copied().unwrap_or((0, 0));
        let slot = op_row[k] as usize * 8 + byte_off[k] as usize;
        ref_ts[slot] = ot;
        ref_val[slot] = ov;
        state.insert(addr[k], (value[k], ts[k]));
    }

    let (dev_ts, dev_val) =
        math_cuda::trace_walk::gpu_mem_walk_memw(&addr, &ts, &value, &init, &op_row, &byte_off, num_ops)
            .expect("device walk+gather");

    let mut mism = 0usize;
    let mut aligned = 0usize;
    for op in 0..num_ops {
        let w = op_width[op];
        for j in 0..w {
            let slot = op * 8 + j;
            if dev_ts[slot] != ref_ts[slot] || dev_val[slot] != ref_val[slot] {
                if mism < 10 {
                    eprintln!(
                        "op {op} byte {j}: dev=({},{}) ref=({},{})",
                        dev_val[slot], dev_ts[slot], ref_val[slot], ref_ts[slot]
                    );
                }
                mism += 1;
            }
        }
        // Classification agrees (derived from old_ts, which must match).
        let da = is_aligned(op_base[op], w, &dev_ts[op * 8..op * 8 + 8]);
        let ra = is_aligned(op_base[op], w, &ref_ts[op * 8..op * 8 + 8]);
        assert_eq!(da, ra, "classification mismatch @ op {op}");
        if da {
            aligned += 1;
        }
    }
    assert_eq!(mism, 0, "{mism} MEMW-row byte mismatches");
    println!(
        "gpu_memw_routing parity OK over {num_ops} LOAD/STORE ops ({n_acc} byte-accesses); \
         classify: {aligned} aligned, {} general",
        num_ops - aligned
    );
}
