//! Phase-2 byte-parity: the device memory memory-model walk
//! ([`math_cuda::trace_walk::gpu_mem_walk`], stable LSD radix sort by 64-bit byte address +
//! predecessor link) must reproduce the sequential `MemoryState` read-old/write-new walk —
//! for every byte-access `old_value`/`old_ts` = the previous access to that byte address
//! (or the init seed at ts 0). Exercised over the real guest's (ethrex_5tx) LOAD+STORE
//! byte-access stream.
//!
//! Scope: validates the walk over the LOAD/STORE access set, seeded from the real initial
//! memory image (`build_initial_image`) so first accesses to code/data addresses read the
//! image byte at ts 0 — matching `MemoryState::from_image`. Ecall (COMMIT/KECCAK/ECSM)
//! memory accesses are a later integration step (Phase 6 seam); they are simply not in this
//! input set, so the device and the reference walk over the identical set and must agree.
//!
//! `LAMBDA_VM_BENCH_ELF=.../rust/ethrex.elf LAMBDA_VM_BENCH_INPUT=.../ethrex_5tx.bin \
//!   cargo test -p lambda-vm-prover --release --features cuda --lib gpu_mem_walk -- --ignored --nocapture`

use std::collections::HashMap;
use std::env;
use std::fs;

use executor::elf::Elf;
use executor::vm::execution::Executor;

use crate::tables::cpu::CpuOperation;
use crate::tables::decode;
use crate::tables::trace_builder::build_initial_image;

#[test]
#[ignore = "requires GPU + LAMBDA_VM_BENCH_ELF (ethrex); run --ignored --nocapture"]
fn gpu_mem_walk_matches_reference() {
    if let Err(e) = math_cuda::device::backend() {
        eprintln!("skipping gpu_mem_walk_matches_reference: no CUDA backend: {e:?}");
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

    // Emit per-byte memory accesses for LOAD/STORE cycles, in program (ts) order.
    // LOAD writes the loaded value (op.rvd) back at its ts; STORE writes op.rv2. Both write
    // `mem_bytes` bytes starting at op.res (mirrors `write_bytes` in collect_load/store).
    let mut addr = Vec::new();
    let mut ts = Vec::new();
    let mut value = Vec::new();
    for (i, log) in result.logs.iter().enumerate() {
        let instr = *instructions
            .get(&log.current_pc)
            .expect("instruction for pc");
        let t = (i as u64) * 4 + 4;
        let op = CpuOperation::from_log_and_instruction(log, t, instr);
        let f = op.decode.fields;
        let (is_load, is_store) = (f.is_load(), f.is_store());
        if !is_load && !is_store {
            continue;
        }
        let base = op.res;
        let nbytes = f.mem_bytes();
        let word = if is_load { op.rvd } else { op.rv2 };
        for j in 0..nbytes {
            addr.push(base.wrapping_add(j as u64));
            ts.push(op.timestamp);
            value.push((word >> (j * 8)) & 0xFF);
        }
    }
    let n = addr.len();

    // Initial memory image (code/data bytes, seeded at ts 0) — the same source
    // `MemoryState::from_image` uses. Per-access `init_value[i]` = image byte at addr[i].
    let image = build_initial_image(&elf, &input);
    let init: Vec<u64> = addr
        .iter()
        .map(|a| image.get(a).copied().unwrap_or(0) as u64)
        .collect();

    // Reference: sequential read-old/write-new walk, pre-seeded from the image at ts 0.
    let mut state: HashMap<u64, (u64, u64)> = HashMap::new();
    for (&a, &b) in &image {
        state.insert(a, (b as u64, 0));
    }
    let mut ref_old_value = vec![0u64; n];
    let mut ref_old_ts = vec![0u64; n];
    let mut seeded_first_hits = 0usize;
    for k in 0..n {
        let (ov, ot) = state.get(&addr[k]).copied().unwrap_or((0, 0));
        ref_old_value[k] = ov;
        ref_old_ts[k] = ot;
        if ot == 0 && ov != 0 {
            seeded_first_hits += 1;
        }
        state.insert(addr[k], (value[k], ts[k]));
    }

    let (dev_old_value, dev_old_ts) =
        math_cuda::trace_walk::gpu_mem_walk(&addr, &ts, &value, &init).expect("device mem walk");

    assert_eq!(dev_old_value.len(), n);
    assert_eq!(dev_old_ts.len(), n);
    let mut mismatches = 0usize;
    for k in 0..n {
        if dev_old_ts[k] != ref_old_ts[k] || dev_old_value[k] != ref_old_value[k] {
            if mismatches < 10 {
                eprintln!(
                    "mismatch @ {k}: addr={:#x} ts={} dev=({},{}) ref=({},{})",
                    addr[k], ts[k], dev_old_value[k], dev_old_ts[k], ref_old_value[k], ref_old_ts[k]
                );
            }
            mismatches += 1;
        }
    }
    assert_eq!(mismatches, 0, "{mismatches} mismatches out of {n} accesses");
    println!(
        "gpu_mem_walk parity OK over {n} byte-accesses (image size {}, {seeded_first_hits} \
         first-accesses read a nonzero image byte at ts 0)",
        image.len()
    );
}
