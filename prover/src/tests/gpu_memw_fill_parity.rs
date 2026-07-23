//! Phase-2 MEMW table fill: the device MEMW_A/MEMW packed-row assembly
//! ([`math_cuda::trace_walk::gpu_build_memw_ls`]) must reproduce, in program order, the
//! aligned/general MEMW rows for the LOAD/STORE stream — value[8], old[8], old_timestamp,
//! flags, base, ts, and the aligned/general split — from the device walk. Positions
//! [width,8) of old_ts/old_value are unconstrained (zero bus multiplicity) and set to 0 on
//! both sides (a valid trace; production fills them from its read-8, which doesn't affect the
//! proof — the constrained [0,width) content is validated here and by gpu_memw_routing).
//!
//! `LAMBDA_VM_BENCH_ELF=.../rust/ethrex.elf LAMBDA_VM_BENCH_INPUT=.../ethrex_5tx.bin \
//!   cargo test -p lambda-vm-prover --release --features cuda --lib gpu_memw_fill -- --ignored --nocapture`

use std::collections::HashMap;
use std::env;
use std::fs;

use executor::elf::Elf;
use executor::vm::execution::Executor;

use crate::tables::cpu::CpuOperation;
use crate::tables::decode;
use crate::tables::trace_builder::build_initial_image;

fn pack2(b: &[u32; 8]) -> [u64; 4] {
    [
        b[0] as u64 | ((b[1] as u64) << 32),
        b[2] as u64 | ((b[3] as u64) << 32),
        b[4] as u64 | ((b[5] as u64) << 32),
        b[6] as u64 | ((b[7] as u64) << 32),
    ]
}

#[test]
#[ignore = "requires GPU + LAMBDA_VM_BENCH_ELF (ethrex); run --ignored --nocapture"]
fn gpu_memw_fill_matches_reference() {
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

    // Per-op + byte-access inputs.
    let (mut addr, mut ts_a, mut val_a, mut op_row, mut byte_off) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let (mut base, mut op_ts, mut is_read, mut width, mut signed, mut value_word) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for (i, log) in result.logs.iter().enumerate() {
        let instr = *instructions.get(&log.current_pc).expect("instruction");
        let op = CpuOperation::from_log_and_instruction(log, (i as u64) * 4 + 4, instr);
        let f = op.decode.fields;
        let (ld, st) = (f.is_load(), f.is_store());
        if !ld && !st {
            continue;
        }
        let w = f.mem_bytes();
        let vw = if ld { op.rvd } else { op.rv2 };
        let row = base.len() as u64;
        for j in 0..w {
            addr.push(op.res.wrapping_add(j as u64));
            ts_a.push(op.timestamp);
            val_a.push((vw >> (j * 8)) & 0xFF);
            op_row.push(row);
            byte_off.push(j as u32);
        }
        base.push(op.res);
        op_ts.push(op.timestamp);
        is_read.push(ld as u32);
        width.push(w as u32);
        signed.push(f.mem_signed() as u32);
        value_word.push(vw);
    }
    let num_ops = base.len();
    let image = build_initial_image(&elf, &input);
    let init: Vec<u64> = addr.iter().map(|a| image.get(a).copied().unwrap_or(0) as u64).collect();

    // Reference walk (image-seeded) → per-op old_ts/old_value, then build the same MEMW rows.
    let mut state: HashMap<u64, (u64, u64)> = HashMap::new();
    for (&a, &b) in &image {
        state.insert(a, (b as u64, 0));
    }
    let n_acc = addr.len();
    let mut walk_ts = vec![0u64; num_ops * 8];
    let mut walk_val = vec![0u64; num_ops * 8];
    for k in 0..n_acc {
        let (ov, ot) = state.get(&addr[k]).copied().unwrap_or((0, 0));
        let slot = op_row[k] as usize * 8 + byte_off[k] as usize;
        walk_ts[slot] = ot;
        walk_val[slot] = ov;
        state.insert(addr[k], (val_a[k], ts_a[k]));
    }

    let mut ref_aligned: Vec<[u64; 12]> = Vec::new();
    let mut ref_general: Vec<[u64; 19]> = Vec::new();
    for op in 0..num_ops {
        let w = width[op] as usize;
        let vw = value_word[op];
        let mut vb = [0u32; 8];
        let mut ob = [0u32; 8];
        if is_read[op] == 1 {
            for (j, b) in vb.iter_mut().take(w).enumerate() {
                *b = ((vw >> (j * 8)) & 0xFF) as u32;
            }
            if w < 8 {
                let sb = (vb[w - 1] >> 7) & 1;
                let fill = if signed[op] == 1 && sb == 1 { 0xFF } else { 0 };
                for b in vb.iter_mut().skip(w) {
                    *b = fill;
                }
            }
            ob = vb; // LOAD old = own
        } else {
            for (j, b) in vb.iter_mut().enumerate() {
                *b = ((vw >> (j * 8)) & 0xFF) as u32;
            }
            for (j, b) in ob.iter_mut().take(w).enumerate() {
                *b = (walk_val[op * 8 + j] & 0xFF) as u32;
            }
        }
        let flags = ((is_read[op] as u64) << 1) | ((w as u64) << 8);
        let v = pack2(&vb);
        let o = pack2(&ob);
        // classify aligned
        let low = (base[op] & 0xFFFF_FFFF) as u32;
        let mut aligned = !(w > 1 && (low & (w as u32 - 1)) != 0);
        if aligned {
            for i in 1..w {
                if walk_ts[op * 8 + i] != walk_ts[op * 8] {
                    aligned = false;
                    break;
                }
            }
        }
        if aligned {
            ref_aligned.push([
                flags, base[op], op_ts[op], walk_ts[op * 8],
                v[0], v[1], v[2], v[3], o[0], o[1], o[2], o[3],
            ]);
        } else {
            let mut ot8 = [0u64; 8];
            for (j, t) in ot8.iter_mut().take(w).enumerate() {
                *t = walk_ts[op * 8 + j];
            }
            ref_general.push([
                flags, base[op], op_ts[op],
                v[0], v[1], v[2], v[3], o[0], o[1], o[2], o[3],
                ot8[0], ot8[1], ot8[2], ot8[3], ot8[4], ot8[5], ot8[6], ot8[7],
            ]);
        }
    }

    let (pa, na, pg, ng) = math_cuda::trace_walk::gpu_build_memw_ls(
        &addr, &ts_a, &val_a, &init, &op_row, &byte_off, &base, &op_ts, &is_read, &width, &signed,
        &value_word,
    )
    .expect("device memw build");

    assert_eq!(na, ref_aligned.len(), "aligned count: dev {na} vs ref {}", ref_aligned.len());
    assert_eq!(ng, ref_general.len(), "general count: dev {ng} vs ref {}", ref_general.len());
    for (r, row) in ref_aligned.iter().enumerate() {
        for c in 0..12 {
            assert_eq!(pa[r * 12 + c], row[c], "MEMW_A row {r} col {c}");
        }
    }
    for (r, row) in ref_general.iter().enumerate() {
        for c in 0..19 {
            assert_eq!(pg[r * 19 + c], row[c], "MEMW row {r} col {c}");
        }
    }
    println!(
        "gpu_memw_fill parity OK: {num_ops} LOAD/STORE ops → {na} MEMW_A + {ng} MEMW rows, \
         packed rows byte-identical to reference"
    );
}
