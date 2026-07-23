//! Phase-3 byte-parity: the device state-free ALU chip-op extractor
//! ([`math_cuda::trace_ops::gpu_extract_alu_chipops`]) must reproduce, in program order,
//! the six instruction-driven ALU chip-op streams the host builds as per-cycle
//! `cpu_ops.iter().filter().map()` projections — LT/SHIFT (SLT/BLT/BGE, SLL/SRL/SRA), EQ
//! (BEQ/BNE), BYTEWISE (AND/OR/XOR), MUL, DVRM. These derive purely from resident cpu_op
//! fields (no memory/register state), so they validate on the real guest (ethrex_5tx)
//! independent of the memory walk / precompiles.
//!
//! `LAMBDA_VM_BENCH_ELF=.../rust/ethrex.elf LAMBDA_VM_BENCH_INPUT=.../ethrex_5tx.bin \
//!   cargo test -p lambda-vm-prover --release --features cuda --lib gpu_extract_alu -- --ignored --nocapture`

use std::env;
use std::fs;

use executor::elf::Elf;
use executor::vm::execution::Executor;

use crate::tables::cpu::CpuOperation;
use crate::tables::decode;
use crate::tables::types::DecodeEntry;

/// Expected chip op: the raw operands + the semantic `alu_flags` bits, in program order.
#[derive(Default)]
struct Exp {
    ops: Vec<(u64, u64, bool, bool, bool, u8)>, // (rv1, arg2, signed, signed2/invert, muldiv, alu_op)
}

#[test]
#[ignore = "requires GPU + LAMBDA_VM_BENCH_ELF (ethrex); run --ignored --nocapture"]
fn gpu_extract_alu_chipops_matches_collect() {
    if let Err(e) = math_cuda::device::backend() {
        eprintln!("skipping gpu_extract_alu_chipops_matches_collect: no CUDA backend: {e:?}");
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
    let mut packed = Vec::with_capacity(n);
    let mut rv1 = Vec::with_capacity(n);
    let mut arg2 = Vec::with_capacity(n);
    // BRANCH/STORE input SoA + per-cycle flags byte (bit0 = branch_cond).
    let (mut flags, mut pc, mut imm, mut res, mut ts_v, mut rv2) = (
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    );
    let (mut lt, mut shift, mut eq, mut bytewise, mut mul, mut dvrm) = (
        Exp::default(),
        Exp::default(),
        Exp::default(),
        Exp::default(),
        Exp::default(),
        Exp::default(),
    );
    // BRANCH/STORE expected 4-tuples: (c0, c1, c2, packed), program order.
    let mut branch_exp: Vec<(u64, u64, u64, u64)> = Vec::new();
    let mut store_exp: Vec<(u64, u64, u64, u64)> = Vec::new();
    for (i, log) in result.logs.iter().enumerate() {
        let instr = *instructions
            .get(&log.current_pc)
            .expect("instruction for pc");
        let ts = (i as u64) * 4 + 4;
        let op = CpuOperation::from_log_and_instruction(log, ts, instr);
        let d = DecodeEntry::from_instruction(log.current_pc, instr, 4);
        let f = op.decode.fields;
        let pk = d.fields.pack();
        // Same predicates as chipop_alu_route (all under `!word_instr`, via the `alu`-gated
        // accessors). The tuple is (rv1, arg2, signed, signed2/invert, muldiv, alu_op).
        let tup = (
            op.rv1,
            op.arg2,
            f.alu_signed(),
            f.alu_signed2_or_invert(),
            f.alu_muldiv(),
            f.alu_op(),
        );
        if !f.word_instr {
            if f.is_lt() {
                lt.ops.push(tup);
            }
            if f.is_shift() {
                shift.ops.push(tup);
            }
            if f.is_eq() {
                eq.ops.push(tup);
            }
            if f.is_and() || f.is_or() || f.is_xor() {
                bytewise.ops.push(tup);
            }
            if f.is_mul() {
                mul.ops.push(tup);
            }
            if f.is_divrem() {
                dvrm.ops.push(tup);
            }
        }
        // BRANCH: filter branch_cond → (pc, imm, rv1, packed). STORE: filter is_store →
        // (res, timestamp, rv2, packed).
        if op.branch_cond {
            branch_exp.push((d.pc, d.imm, op.rv1, pk));
        }
        if f.is_store() {
            store_exp.push((op.res, op.timestamp, op.rv2, pk));
        }
        let flag_byte = (op.branch_cond as u8)
            | ((op.ecall_commit as u8) << 1)
            | ((op.ecall_keccak as u8) << 2)
            | ((op.ecall_ecsm as u8) << 3);
        packed.push(pk);
        rv1.push(op.rv1);
        arg2.push(op.arg2);
        flags.push(flag_byte);
        pc.push(d.pc);
        imm.push(d.imm);
        res.push(op.res);
        ts_v.push(op.timestamp);
        rv2.push(op.rv2);
    }

    let dev =
        math_cuda::trace_ops::gpu_extract_alu_chipops(&packed, &rv1, &arg2).expect("device extract");

    let check = |name: &str, d: &math_cuda::trace_ops::DeviceAluChipOps, exp: &Exp| {
        assert_eq!(d.a.len(), exp.ops.len(), "{name}: count");
        assert_eq!(d.b.len(), exp.ops.len(), "{name}: count b");
        assert_eq!(d.alu_flags.len(), exp.ops.len(), "{name}: count flags");
        for (i, &(a, b, signed, s2, muldiv, alu_op)) in exp.ops.iter().enumerate() {
            assert_eq!(d.a[i], a, "{name}.a @ {i}");
            assert_eq!(d.b[i], b, "{name}.b @ {i}");
            let fl = d.alu_flags[i];
            assert_eq!((fl >> 5) & 1 == 1, signed, "{name}.signed @ {i}");
            assert_eq!((fl >> 6) & 1 == 1, s2, "{name}.signed2/invert @ {i}");
            assert_eq!((fl >> 7) & 1 == 1, muldiv, "{name}.muldiv @ {i}");
            assert_eq!(fl & 0x1F, alu_op, "{name}.alu_op @ {i}");
        }
    };
    check("lt", &dev.lt, &lt);
    check("shift", &dev.shift, &shift);
    check("eq", &dev.eq, &eq);
    check("bytewise", &dev.bytewise, &bytewise);
    check("mul", &dev.mul, &mul);
    check("dvrm", &dev.dvrm, &dvrm);

    // BRANCH + STORE (4-column state-free projections).
    let (branch, store) =
        math_cuda::trace_ops::gpu_extract_branch_store(&packed, &flags, &pc, &imm, &rv1, &res, &ts_v, &rv2)
            .expect("device extract branch/store");
    let check4 = |name: &str, d: &math_cuda::trace_ops::DeviceGather4, exp: &[(u64, u64, u64, u64)]| {
        assert_eq!(d.c0.len(), exp.len(), "{name}: count");
        for (i, &(c0, c1, c2, c3)) in exp.iter().enumerate() {
            assert_eq!(d.c0[i], c0, "{name}.c0 @ {i}");
            assert_eq!(d.c1[i], c1, "{name}.c1 @ {i}");
            assert_eq!(d.c2[i], c2, "{name}.c2 @ {i}");
            assert_eq!(d.c3[i], c3, "{name}.c3(packed) @ {i}");
        }
    };
    check4("branch", &branch, &branch_exp);
    check4("store", &store, &store_exp);

    println!(
        "gpu_extract_alu_chipops parity OK over {n} cycles \
         (LT {}, SHIFT {}, EQ {}, BYTEWISE {}, MUL {}, DVRM {}, BRANCH {}, STORE {})",
        lt.ops.len(),
        shift.ops.len(),
        eq.ops.len(),
        bytewise.ops.len(),
        mul.ops.len(),
        dvrm.ops.len(),
        branch_exp.len(),
        store_exp.len(),
    );
}
