//! Device-dedup parity: `math_cuda::trace_walk::gpu_dedup3` (sort by full op key +
//! segmented multiplicity reduce) must reproduce the host `HashMap<Op, mult>` dedup — the
//! step that makes the deduped chips (LT/SHIFT/EQ/BYTEWISE/MUL/DVRM/BRANCH) resident.
//! Validated on the LT op stream from ethrex_5tx, compared as a MULTISET (the LogUp bus is
//! order-independent, so sorted device order vs HashMap order is irrelevant — only the set
//! of (unique op, mult) must match).
//!
//! `LAMBDA_VM_BENCH_ELF=.../rust/ethrex.elf LAMBDA_VM_BENCH_INPUT=.../ethrex_5tx.bin \
//!   cargo test -p lambda-vm-prover --release --features cuda --lib gpu_dedup -- --ignored --nocapture`

use std::collections::HashMap;
use std::env;
use std::fs;

use executor::elf::Elf;
use executor::vm::execution::Executor;

use crate::tables::cpu::CpuOperation;
use crate::tables::decode;

#[test]
#[ignore = "requires GPU + LAMBDA_VM_BENCH_ELF (ethrex); run --ignored --nocapture"]
fn gpu_dedup3_matches_hashmap_lt() {
    if let Err(e) = math_cuda::device::backend() {
        eprintln!("skipping gpu_dedup3_matches_hashmap_lt: no CUDA backend: {e:?}");
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

    // LT op keys: k0 = flags (signed | invert<<1), k1 = lhs (rv1), k2 = rhs (arg2).
    let mut k0 = Vec::new();
    let mut k1 = Vec::new();
    let mut k2 = Vec::new();
    let mut host: HashMap<(u64, u64, u64), u64> = HashMap::new();
    for (i, log) in result.logs.iter().enumerate() {
        let instr = *instructions.get(&log.current_pc).expect("instruction");
        let op = CpuOperation::from_log_and_instruction(log, (i as u64) * 4 + 4, instr);
        let f = op.decode.fields;
        if !f.word_instr && f.is_lt() {
            let flags = (f.alu_signed() as u64) | ((f.alu_signed2_or_invert() as u64) << 1);
            k0.push(flags);
            k1.push(op.rv1);
            k2.push(op.arg2);
            *host.entry((op.rv1, op.arg2, flags)).or_insert(0) += 1;
        }
    }
    let total_ops = k0.len();

    let dd = math_cuda::trace_walk::gpu_dedup3(&k0, &k1, &k2).expect("device dedup");
    let m = dd.mult.len();
    assert_eq!(dd.k0.len(), m);
    assert_eq!(dd.k1.len(), m);
    assert_eq!(dd.k2.len(), m);

    // Build device map {(lhs, rhs, flags): mult} and compare to host as a multiset.
    let mut dev: HashMap<(u64, u64, u64), u64> = HashMap::with_capacity(m);
    let mut dev_total = 0u64;
    for i in 0..m {
        let key = (dd.k1[i], dd.k2[i], dd.k0[i]); // (lhs, rhs, flags)
        assert!(dev.insert(key, dd.mult[i]).is_none(), "device emitted a duplicate unique key");
        dev_total += dd.mult[i];
    }
    assert_eq!(m, host.len(), "unique-row count: device {m} vs host {}", host.len());
    assert_eq!(dev_total as usize, total_ops, "multiplicities must sum to total ops");
    for (key, &hmult) in &host {
        match dev.get(key) {
            Some(&dmult) => assert_eq!(dmult, hmult, "mult mismatch for {key:?}"),
            None => panic!("device missing key {key:?}"),
        }
    }
    println!("gpu_dedup3 parity OK: {total_ops} LT ops → {m} unique rows (multiset == host HashMap)");
}
