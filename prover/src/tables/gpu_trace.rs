//! On-GPU trace generation: build trace tables directly in device memory so
//! they feed the already-GPU LDE/commit without a host round-trip.
//!
//! This module is compiled only under the `cuda` feature. It hosts the
//! device-build dispatch (added table-by-table) plus the kill-switch used to
//! A/B the GPU path against the CPU trace generator.
//!
//! Design: `reports/tracegen/GPU-TRACEGEN-DESIGN-V2.md`.
#![cfg(feature = "cuda")]

use std::sync::{Arc, OnceLock};

use executor::vm::logs::Log;
use stark::trace::TraceTable;

use std::collections::HashMap;

use super::branch::{self, BranchOperation};
use super::bytewise::{self, BytewiseOperation};
use super::commit;
use super::cpu::{self, CpuOperation};
use super::cpu32::{self, Cpu32Operation};
use super::dvrm::{self, DvrmOperation};
use super::ecdas;
use super::ecsm;
use super::eq::{self, EqOperation};
use super::keccak;
use super::keccak_rnd;
use super::load::{self, LoadOperation};
use super::lt::{self, LtOperation};
use super::memw::{self, MemwOperation};
use super::memw_aligned;
use super::memw_register::{self, RegRow};
use super::mul::{self, MulOperation};
use super::shift::{self, ShiftOperation};
use super::store::{self, StoreOperation};
use super::types::{GoldilocksExtension, GoldilocksField};

/// When set (`LAMBDA_VM_CPU_TRACE=1`), all GPU trace-build dispatchers return
/// `None` so callers fall back to the CPU trace generator. This is the one-flag
/// A/B switch: same binary, `LAMBDA_VM_CPU_TRACE=1` runs the CPU baseline,
/// unset runs the GPU path. Read once and cached.
pub(crate) fn gpu_trace_disabled() -> bool {
    static DISABLED: OnceLock<bool> = OnceLock::new();
    *DISABLED.get_or_init(|| {
        std::env::var("LAMBDA_VM_CPU_TRACE")
            .map(|v| v != "0" && !v.is_empty())
            .unwrap_or(false)
    })
}

/// Master switch (`LAMBDA_VM_GPU_FULL=1`): enable the WHOLE resident trace-gen pipeline at once —
/// resident chips + device register walk + memory_state-drop + bitwise histogram + resident memw_reg.
/// Each individual gate ORs this in, so one flag turns everything on. Read once and cached. The
/// `LAMBDA_VM_CPU_TRACE=1` kill-switch still overrides (all-CPU).
pub(crate) fn gpu_full_enabled() -> bool {
    static FULL: OnceLock<bool> = OnceLock::new();
    *FULL.get_or_init(|| std::env::var("LAMBDA_VM_GPU_FULL").is_ok_and(|v| v == "1"))
}

/// When set (`LAMBDA_VM_GPU_RESIDENT_CHIPS=1`), p5 builds the eligible chip tables via the
/// fully-resident device→device chains (cpu_op fields → device op-build → device fill), instead
/// of building the chip ops on the host and uploading them. Opt-in: warm-measured ~flat/slightly
/// slower than the multicore-CPU chip build (no net win until the WHOLE pipeline is resident), so it
/// stays opt-in rather than default. Read once and cached.
pub(crate) fn gpu_resident_chips_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("LAMBDA_VM_GPU_RESIDENT_CHIPS").is_ok_and(|v| v == "1") || gpu_full_enabled()
    })
}

/// Build the CPU32 trace table via the resident device chain fed directly from the resident
/// `cpu_ops` (no host CPU32-op build/pack). Single table (CPU32 is per-row, one source), so
/// returns `None` if it would exceed `max_rows` (falls back to the host-op device path or CPU).
/// Byte-identical to `gpu_build_cpu32_tables` (validated by gpu_cpu32_parity).
pub(crate) fn build_cpu32_resident_tables(
    cpu_ops: &[CpuOperation],
    max_rows: usize,
) -> Option<Vec<TraceTable<GoldilocksField, GoldilocksExtension>>> {
    if gpu_trace_disabled() {
        return None;
    }
    let rows = cpu_ops.iter().filter(|op| op.decode.fields.word_instr).count();
    if rows > max_rows {
        return None; // chunking not yet supported in the resident path
    }
    let num_rows = rows.next_power_of_two().max(4);
    let n = cpu_ops.len();
    let (mut packed, mut rv1, mut rv2, mut imm, mut pc) = (
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    );
    for op in cpu_ops {
        packed.push(op.decode.fields.pack());
        rv1.push(op.rv1);
        rv2.push(op.rv2);
        imm.push(op.decode.imm);
        pc.push(op.decode.pc);
    }
    let dev = math_cuda::trace_ops::gpu_build_cpu32_resident_dev(
        &packed, &rv1, &rv2, &imm, &pc, num_rows,
    )
    .ok()?;
    let mut trace = TraceTable::new_main(
        crate::tables::types::zeroed_fe_vec(num_rows * cpu32::cols::NUM_COLUMNS),
        cpu32::cols::NUM_COLUMNS,
        1,
    );
    trace.set_main_input_dev(Arc::new(dev));
    Some(vec![trace])
}

/// Build the LOAD trace table via the resident device chain fed from the resident `cpu_ops`
/// (no host LOAD-op build/pack). Per-row, single chunk — `None` if it would exceed `max_rows`.
/// Byte-identical to `gpu_build_load_tables` (validated by gpu_load_parity).
pub(crate) fn build_load_resident_tables(
    cpu_ops: &[CpuOperation],
    max_rows: usize,
) -> Option<Vec<TraceTable<GoldilocksField, GoldilocksExtension>>> {
    if gpu_trace_disabled() {
        return None;
    }
    let rows = cpu_ops.iter().filter(|op| op.decode.fields.is_load()).count();
    if rows > max_rows {
        return None;
    }
    let num_rows = rows.next_power_of_two().max(4);
    let n = cpu_ops.len();
    let (mut packed, mut res, mut rvd) =
        (Vec::with_capacity(n), Vec::with_capacity(n), Vec::with_capacity(n));
    for op in cpu_ops {
        packed.push(op.decode.fields.pack());
        res.push(op.res);
        rvd.push(op.rvd);
    }
    let dev = math_cuda::trace_ops::gpu_build_load_resident_dev(&packed, &res, &rvd, num_rows).ok()?;
    let mut trace = TraceTable::new_main(
        crate::tables::types::zeroed_fe_vec(num_rows * load::cols::NUM_COLUMNS),
        load::cols::NUM_COLUMNS,
        1,
    );
    trace.set_main_input_dev(Arc::new(dev));
    Some(vec![trace])
}

/// Build the STORE trace table via the resident device chain fed from the resident `cpu_ops`.
/// Per-row, single chunk — `None` if it would exceed `max_rows`. Byte-identical to
/// `gpu_build_store_tables` (validated by gpu_resident_chips).
pub(crate) fn build_store_resident_tables(
    cpu_ops: &[CpuOperation],
    max_rows: usize,
) -> Option<Vec<TraceTable<GoldilocksField, GoldilocksExtension>>> {
    if gpu_trace_disabled() {
        return None;
    }
    let rows = cpu_ops.iter().filter(|op| op.decode.fields.is_store()).count();
    if rows > max_rows {
        return None;
    }
    let num_rows = rows.next_power_of_two().max(4);
    let n = cpu_ops.len();
    let (mut packed, mut res, mut rv2) =
        (Vec::with_capacity(n), Vec::with_capacity(n), Vec::with_capacity(n));
    for op in cpu_ops {
        packed.push(op.decode.fields.pack());
        res.push(op.res);
        rv2.push(op.rv2);
    }
    let dev = math_cuda::trace_ops::gpu_build_store_resident_dev(&packed, &res, &rv2, num_rows).ok()?;
    let mut trace = TraceTable::new_main(
        crate::tables::types::zeroed_fe_vec(num_rows * store::cols::NUM_COLUMNS),
        store::cols::NUM_COLUMNS,
        1,
    );
    trace.set_main_input_dev(Arc::new(dev));
    Some(vec![trace])
}

/// Build the SHIFT trace table via the resident device chain fed from the resident `cpu_ops`
/// (merges instruction-driven + cpu32-derived shifts on device). Per-row, single chunk —
/// `None` if it would exceed `max_rows`. Byte-identical to `gpu_build_shift_tables`.
pub(crate) fn build_shift_resident_tables(
    cpu_ops: &[CpuOperation],
    max_rows: usize,
) -> Option<Vec<TraceTable<GoldilocksField, GoldilocksExtension>>> {
    if gpu_trace_disabled() {
        return None;
    }
    let rows: usize = cpu_ops
        .iter()
        .filter(|op| {
            let f = &op.decode.fields;
            (!f.word_instr && f.is_shift()) || (f.word_instr && !f.add && !f.sub && f.is_shift())
        })
        .count();
    if rows > max_rows {
        return None;
    }
    let num_rows = rows.next_power_of_two().max(4);
    let n = cpu_ops.len();
    let (mut packed, mut rv1, mut rv2, mut arg2, mut imm) = (
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    );
    for op in cpu_ops {
        packed.push(op.decode.fields.pack());
        rv1.push(op.rv1);
        rv2.push(op.rv2);
        arg2.push(op.arg2);
        imm.push(op.decode.imm);
    }
    let dev =
        math_cuda::trace_ops::gpu_build_shift_full_resident_dev(&packed, &rv1, &rv2, &arg2, &imm, num_rows)
            .ok()?;
    let mut trace = TraceTable::new_main(
        crate::tables::types::zeroed_fe_vec(num_rows * shift::cols::NUM_COLUMNS),
        shift::cols::NUM_COLUMNS,
        1,
    );
    trace.set_main_input_dev(Arc::new(dev));
    Some(vec![trace])
}

/// Build the EQ trace table via the resident device chain (auto-sized to the device-computed
/// unique count). Global dedup ⇒ single table; returns `None` if the raw is_eq op count exceeds
/// `max_rows` (the CPU path would chunk with per-chunk dedup, which global dedup can't match).
/// Multiset-equal to `gpu_build_eq_tables` (sorted vs HashMap row order — both valid traces).
pub(crate) fn build_eq_resident_tables(
    cpu_ops: &[CpuOperation],
    max_rows: usize,
) -> Option<Vec<TraceTable<GoldilocksField, GoldilocksExtension>>> {
    if gpu_trace_disabled() {
        return None;
    }
    let raw = cpu_ops
        .iter()
        .filter(|op| !op.decode.fields.word_instr && op.decode.fields.is_eq())
        .count();
    if raw > max_rows {
        return None;
    }
    let (packed, rv1, arg2) = alu_soa(cpu_ops);
    let (dev, num_rows) = math_cuda::trace_ops::gpu_build_eq_resident_dev(&packed, &rv1, &arg2).ok()?;
    let mut trace = TraceTable::new_main(
        crate::tables::types::zeroed_fe_vec(num_rows * eq::cols::NUM_COLUMNS),
        eq::cols::NUM_COLUMNS,
        1,
    );
    trace.set_main_input_dev(Arc::new(dev));
    Some(vec![trace])
}

/// Build the BYTEWISE trace table via the resident device chain (auto-sized). Same
/// single-chunk / multiset semantics as `build_eq_resident_tables`.
pub(crate) fn build_bytewise_resident_tables(
    cpu_ops: &[CpuOperation],
    max_rows: usize,
) -> Option<Vec<TraceTable<GoldilocksField, GoldilocksExtension>>> {
    if gpu_trace_disabled() {
        return None;
    }
    let raw = cpu_ops
        .iter()
        .filter(|op| {
            let f = &op.decode.fields;
            !f.word_instr && (f.is_and() || f.is_or() || f.is_xor())
        })
        .count();
    if raw > max_rows {
        return None;
    }
    let (packed, rv1, arg2) = alu_soa(cpu_ops);
    let (dev, num_rows) =
        math_cuda::trace_ops::gpu_build_bytewise_resident_dev(&packed, &rv1, &arg2).ok()?;
    let mut trace = TraceTable::new_main(
        crate::tables::types::zeroed_fe_vec(num_rows * bytewise::cols::NUM_COLUMNS),
        bytewise::cols::NUM_COLUMNS,
        1,
    );
    trace.set_main_input_dev(Arc::new(dev));
    Some(vec![trace])
}

/// Build the DVRM trace table via the resident device chain (instruction ⊕ cpu32 sources,
/// auto-sized). Single-chunk guard on the raw dvrm op count. Multiset-equal to
/// `gpu_build_dvrm_tables`.
pub(crate) fn build_dvrm_resident_tables(
    cpu_ops: &[CpuOperation],
    max_rows: usize,
) -> Option<Vec<TraceTable<GoldilocksField, GoldilocksExtension>>> {
    if gpu_trace_disabled() {
        return None;
    }
    let raw = cpu_ops
        .iter()
        .filter(|op| {
            let f = &op.decode.fields;
            (!f.word_instr && f.is_divrem()) || (f.word_instr && !f.add && !f.sub && f.is_divrem())
        })
        .count();
    if raw > max_rows {
        return None;
    }
    let n = cpu_ops.len();
    let (mut packed, mut rv1, mut rv2, mut arg2, mut imm) = (
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    );
    for op in cpu_ops {
        packed.push(op.decode.fields.pack());
        rv1.push(op.rv1);
        rv2.push(op.rv2);
        arg2.push(op.arg2);
        imm.push(op.decode.imm);
    }
    let (dev, num_rows) =
        math_cuda::trace_ops::gpu_build_dvrm_full_resident_dev(&packed, &rv1, &rv2, &arg2, &imm).ok()?;
    let mut trace = TraceTable::new_main(
        crate::tables::types::zeroed_fe_vec(num_rows * dvrm::cols::NUM_COLUMNS),
        dvrm::cols::NUM_COLUMNS,
        1,
    );
    trace.set_main_input_dev(Arc::new(dev));
    Some(vec![trace])
}

/// Build the MUL trace table via the resident device chain — ALL four sources (instruction ⊕
/// instruction-dvrm→mul ⊕ cpu32 ⊕ cpu32-dvrm→mul), auto-sized. Single-chunk guard on the raw
/// mul-op count. Multiset-equal to `gpu_build_mul_tables`.
pub(crate) fn build_mul_resident_tables(
    cpu_ops: &[CpuOperation],
    max_rows: usize,
) -> Option<Vec<TraceTable<GoldilocksField, GoldilocksExtension>>> {
    if gpu_trace_disabled() {
        return None;
    }
    let raw: usize = cpu_ops
        .iter()
        .map(|op| {
            let f = &op.decode.fields;
            let elig = if f.word_instr { !f.add && !f.sub } else { true };
            if !elig {
                0
            } else if f.is_mul() {
                1 // MUL op
            } else if f.is_divrem() {
                2 // dvrm→mul C13 + C14
            } else {
                0
            }
        })
        .sum();
    if raw > max_rows {
        return None;
    }
    let n = cpu_ops.len();
    let (mut packed, mut rv1, mut rv2, mut arg2, mut imm) = (
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    );
    for op in cpu_ops {
        packed.push(op.decode.fields.pack());
        rv1.push(op.rv1);
        rv2.push(op.rv2);
        arg2.push(op.arg2);
        imm.push(op.decode.imm);
    }
    let (dev, num_rows) =
        math_cuda::trace_ops::gpu_build_mul_full_resident_dev(&packed, &rv1, &rv2, &arg2, &imm).ok()?;
    let mut trace = TraceTable::new_main(
        crate::tables::types::zeroed_fe_vec(num_rows * mul::cols::NUM_COLUMNS),
        mul::cols::NUM_COLUMNS,
        1,
    );
    trace.set_main_input_dev(Arc::new(dev));
    Some(vec![trace])
}

/// Build the BRANCH trace table via the resident device chain (dedup4, auto-sized). Single
/// source (branch_cond). Multiset-equal to `gpu_build_branch_tables`.
pub(crate) fn build_branch_resident_tables(
    cpu_ops: &[CpuOperation],
    max_rows: usize,
) -> Option<Vec<TraceTable<GoldilocksField, GoldilocksExtension>>> {
    if gpu_trace_disabled() {
        return None;
    }
    let raw = cpu_ops.iter().filter(|op| op.branch_cond).count();
    if raw > max_rows {
        return None;
    }
    let n = cpu_ops.len();
    let (mut packed, mut flags, mut pc, mut imm, mut rv1) = (
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    );
    for op in cpu_ops {
        packed.push(op.decode.fields.pack());
        flags.push(op.branch_cond as u8);
        pc.push(op.decode.pc);
        imm.push(op.decode.imm);
        rv1.push(op.rv1);
    }
    let (dev, num_rows) =
        math_cuda::trace_ops::gpu_build_branch_resident_dev(&packed, &flags, &pc, &imm, &rv1).ok()?;
    let mut trace = TraceTable::new_main(
        crate::tables::types::zeroed_fe_vec(num_rows * branch::cols::NUM_COLUMNS),
        branch::cols::NUM_COLUMNS,
        1,
    );
    trace.set_main_input_dev(Arc::new(dev));
    Some(vec![trace])
}

// ── p5 single-build seam ──────────────────────────────────────────────────────────────────
// Build the device-resident cpu_ops ONCE (`build_shared_devops`) and let every chip's
// `*_from_devops` table builder read the SAME device buffers in place — no per-chip SoA
// extraction or upload. Each builder keeps its own cheap host-side row count only for the
// single-chunk `max_rows` guard (returning `None` ⇒ p5 falls back to the per-chip path/CPU).

/// Extract the full cpu_op field SoA (packed/imm/pc/rv1/rv2/arg2/res/rvd/flags) ONCE and upload
/// it into a resident device buffer set shared by all chip builders. Returns `None` when the GPU
/// path is disabled or the resident-chips flag is off (so p5 skips the single-build entirely).
pub(crate) fn build_shared_devops(
    cpu_ops: &[CpuOperation],
    logs: &[Log],
) -> Option<math_cuda::trace_ops::DeviceCpuOpsResident> {
    if gpu_trace_disabled() || !gpu_resident_chips_enabled() {
        return None;
    }
    let n = cpu_ops.len();

    // Step A (device cpu_ops seam): under gpu_full, RECOMPUTE the `from_log` fields
    // (rv1/rv2/arg2/res/rvd/next_pc/branch_cond + ecall metadata) ON DEVICE from the raw Log SoA
    // + the decode SoA, instead of extracting the host-computed `CpuOperation` fields and
    // uploading them. Only the decode SoA (packed/imm/pc — already decoded in p1) is extracted
    // from `cpu_ops`; the raw log fields come straight from `logs`. Bit-exact with the host
    // `from_log` (parity: `gpu_cpu_ops_parity`, 4.03M cycles). Falls back to the upload path when
    // gpu_full is off (resident-chips-only mode) or if the log/op lengths ever disagree.
    if gpu_full_enabled() && logs.len() == n {
        let (mut packed, mut imm, mut pc) = (
            Vec::with_capacity(n),
            Vec::with_capacity(n),
            Vec::with_capacity(n),
        );
        for op in cpu_ops {
            packed.push(op.decode.fields.pack());
            imm.push(op.decode.imm);
            pc.push(op.decode.pc);
        }
        let (mut cpc, mut npc, mut s1, mut s2, mut dv) = (
            Vec::with_capacity(n),
            Vec::with_capacity(n),
            Vec::with_capacity(n),
            Vec::with_capacity(n),
            Vec::with_capacity(n),
        );
        for log in logs {
            cpc.push(log.current_pc);
            npc.push(log.next_pc);
            s1.push(log.src1_val);
            s2.push(log.src2_val);
            dv.push(log.dst_val);
        }
        return math_cuda::trace_ops::gpu_build_cpu_ops_resident(
            &cpc, &npc, &s1, &s2, &dv, &pc, &imm, &packed,
        )
        .ok();
    }

    let (mut packed, mut imm, mut pc, mut rv1, mut rv2, mut arg2, mut res, mut rvd, mut flags) = (
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::<u8>::with_capacity(n),
    );
    for op in cpu_ops {
        packed.push(op.decode.fields.pack());
        imm.push(op.decode.imm);
        pc.push(op.decode.pc);
        rv1.push(op.rv1);
        rv2.push(op.rv2);
        arg2.push(op.arg2);
        res.push(op.res);
        rvd.push(op.rvd);
        flags.push(op.branch_cond as u8);
    }
    math_cuda::trace_ops::gpu_upload_cpu_ops_resident(
        &packed, &imm, &pc, &rv1, &rv2, &arg2, &res, &rvd, &flags,
    )
    .ok()
}

/// C2-c (wiring): the REGISTER table's final-state map, derived ON DEVICE (regs 0-31 + PC via the
/// register access-stream snapshot; x254 via the commit-index scan) instead of the host
/// `RegisterState::to_final_state_map` (which needs the 4M-op sequential advance). Layout is
/// bit-identical to `to_final_state_map`. `None` when the GPU path is off (→ caller keeps the host map).
/// Parity-validated standalone (`gpu_reg_final_parity`); wired under gpu_full so the e2e proof itself
/// re-verifies it (a wrong map ⇒ the REGISTER table unbalances ⇒ verify fails).
pub(crate) fn device_register_final_state(
    cpu_ops: &[CpuOperation],
    ecall_accesses: &super::trace_builder::EcallAccesses,
    register_init: &[u32],
    is_final: bool,
    // The REGISTER table's PC (x255) final token is finalized by the caller (HALT: pc=1 at
    // `halt_ts + 4*padding + 1`), NOT by the register walk. When `Some((value, ts))`, use it for
    // addrs 510/511 instead of the device snapshot's last PC access.
    pc_final: Option<(u64, u64)>,
) -> Option<super::register::FinalRegisterStateMap> {
    use super::register::{FinalRegisterWordState, register_base_address};
    if gpu_trace_disabled() || !gpu_full_enabled() {
        return None;
    }
    let n = cpu_ops.len();
    let (mut packed, mut rv1, mut rv2, mut rvd, mut next_pc) = (
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    );
    let (mut commit_flag, mut commit_count) = (Vec::with_capacity(n), Vec::with_capacity(n));
    for op in cpu_ops {
        packed.push(op.decode.fields.pack());
        rv1.push(op.rv1);
        rv2.push(op.rv2);
        rvd.push(op.rvd);
        next_pc.push(op.next_pc);
        commit_flag.push(u8::from(op.ecall_commit));
        commit_count.push(op.commit_count);
    }
    let e_oi = ecall_accesses.reg_op_index.clone();
    let e_addr: Vec<u32> = ecall_accesses.reg.iter().map(|a| a.reg_addr as u32).collect();
    let e_ts: Vec<u64> = ecall_accesses.reg.iter().map(|a| a.timestamp).collect();
    let e_val: Vec<u64> = ecall_accesses.reg.iter().map(|a| a.value).collect();
    let e_ir: Vec<u8> = ecall_accesses.reg.iter().map(|a| u8::from(a.is_read)).collect();
    let seed = super::trace_builder::walk_seed_from_register_init(register_init);
    let start_commit_index =
        register_init.get(super::register::X254_INDEX).copied().unwrap_or(0) as u64;

    let (val, ts) = math_cuda::trace_walk::gpu_register_final_snapshot(
        &packed, &rv1, &rv2, &rvd, &next_pc, &e_oi, &e_addr, &e_ts, &e_val, &e_ir, &seed, 1,
        &commit_flag, &commit_count, start_commit_index,
    )
    .ok()?;

    let mut map = super::register::FinalRegisterStateMap::new();
    for r in 0..32u8 {
        let a = (2 * r) as usize;
        // HALT finalization (is_final): x1-x31 are written 0 at ts=u64::MAX (see `collect_halt_ops`),
        // which the device walk (regular+ecall accesses only) doesn't see. x0 is never written (stays
        // init). Mirror the post-halt host state here.
        let (v, t) = if is_final && r >= 1 {
            (0u64, u64::MAX)
        } else {
            (val[a], ts[a])
        };
        let base = register_base_address(r);
        map.insert(base, FinalRegisterWordState { timestamp: t, value: (v & 0xFFFF_FFFF) as u32 });
        map.insert(base + 1, FinalRegisterWordState { timestamp: t, value: (v >> 32) as u32 });
    }
    // x254 (508, single word) — not affected by HALT.
    map.insert(508, FinalRegisterWordState { timestamp: ts[508], value: val[508] as u32 });
    // PC (510/511): the caller-finalized token when provided (HALT), else the device snapshot's last PC.
    let (pv, pt) = pc_final.unwrap_or((val[510], ts[510]));
    map.insert(510, FinalRegisterWordState { timestamp: pt, value: (pv & 0xFFFF_FFFF) as u32 });
    map.insert(511, FinalRegisterWordState { timestamp: pt, value: (pv >> 32) as u32 });
    Some(map)
}

/// Wrap an autosized `(device buffer, num_rows)` resident-chip result into a single main
/// `TraceTable` with `ncols` columns (device-input; the host main is a zeroed placeholder).
fn devops_table(
    dev: math_cuda::CudaSlice<u64>,
    num_rows: usize,
    ncols: usize,
) -> Vec<TraceTable<GoldilocksField, GoldilocksExtension>> {
    let mut trace =
        TraceTable::new_main(crate::tables::types::zeroed_fe_vec(num_rows * ncols), ncols, 1);
    trace.set_main_input_dev(Arc::new(dev));
    vec![trace]
}

type Devops = math_cuda::trace_ops::DeviceCpuOpsResident;
type Tables = Vec<TraceTable<GoldilocksField, GoldilocksExtension>>;

/// LT-resident-table STEP 2B: the FULL LT table (instruction ⊕ dvrm→lt ⊕ memw→lt) built resident on
/// device, CHUNKED to `max_rows` rows/table (LT dedups poorly → multiple chunks). `memw_lt_lhs/rhs` are
/// the device-derived memw→lt operand pairs. Replaces the host `gpu_build_lt_tables(&lt_ops)` under
/// gpu_full → no host `lt_ops`. Each chunk is a device-input main table (`set_main_input_dev`).
pub(crate) fn build_lt_resident_tables_from_devops(
    devops: &Devops,
    memw_lt_lhs: &[u64],
    memw_lt_rhs: &[u64],
    max_rows: usize,
) -> Option<Tables> {
    if gpu_trace_disabled() {
        return None;
    }
    let chunks = math_cuda::trace_ops::gpu_build_lt_full_resident_from_devops(
        devops, memw_lt_lhs, memw_lt_rhs, max_rows,
    )
    .ok()?;
    let mut tables = Vec::with_capacity(chunks.len());
    for (dev, num_rows) in chunks {
        let mut trace = TraceTable::new_main(
            crate::tables::types::zeroed_fe_vec(num_rows * lt::cols::NUM_COLUMNS),
            lt::cols::NUM_COLUMNS,
            1,
        );
        trace.set_main_input_dev(Arc::new(dev));
        tables.push(trace);
    }
    Some(tables)
}

pub(crate) fn build_cpu32_resident_tables_from_devops(
    cpu_ops: &[CpuOperation],
    devops: &Devops,
    max_rows: usize,
) -> Option<Tables> {
    if gpu_trace_disabled() {
        return None;
    }
    let raw = cpu_ops.iter().filter(|op| op.decode.fields.word_instr).count();
    if raw > max_rows {
        return None;
    }
    let (dev, num_rows) = math_cuda::trace_ops::gpu_build_cpu32_resident_from_devops(devops).ok()?;
    Some(devops_table(dev, num_rows, cpu32::cols::NUM_COLUMNS))
}

pub(crate) fn build_load_resident_tables_from_devops(
    cpu_ops: &[CpuOperation],
    devops: &Devops,
    max_rows: usize,
) -> Option<Tables> {
    if gpu_trace_disabled() {
        return None;
    }
    let raw = cpu_ops.iter().filter(|op| op.decode.fields.is_load()).count();
    if raw > max_rows {
        return None;
    }
    let (dev, num_rows) = math_cuda::trace_ops::gpu_build_load_resident_from_devops(devops).ok()?;
    Some(devops_table(dev, num_rows, load::cols::NUM_COLUMNS))
}

pub(crate) fn build_store_resident_tables_from_devops(
    cpu_ops: &[CpuOperation],
    devops: &Devops,
    max_rows: usize,
) -> Option<Tables> {
    if gpu_trace_disabled() {
        return None;
    }
    let raw = cpu_ops.iter().filter(|op| op.decode.fields.is_store()).count();
    if raw > max_rows {
        return None;
    }
    let (dev, num_rows) = math_cuda::trace_ops::gpu_build_store_resident_from_devops(devops).ok()?;
    Some(devops_table(dev, num_rows, store::cols::NUM_COLUMNS))
}

pub(crate) fn build_shift_resident_tables_from_devops(
    cpu_ops: &[CpuOperation],
    devops: &Devops,
    max_rows: usize,
) -> Option<Tables> {
    if gpu_trace_disabled() {
        return None;
    }
    let raw: usize = cpu_ops
        .iter()
        .filter(|op| {
            let f = &op.decode.fields;
            (!f.word_instr && f.is_shift()) || (f.word_instr && !f.add && !f.sub && f.is_shift())
        })
        .count();
    if raw > max_rows {
        return None;
    }
    let (dev, num_rows) =
        math_cuda::trace_ops::gpu_build_shift_full_resident_from_devops(devops).ok()?;
    Some(devops_table(dev, num_rows, shift::cols::NUM_COLUMNS))
}

pub(crate) fn build_eq_resident_tables_from_devops(
    cpu_ops: &[CpuOperation],
    devops: &Devops,
    max_rows: usize,
) -> Option<Tables> {
    if gpu_trace_disabled() {
        return None;
    }
    let raw = cpu_ops
        .iter()
        .filter(|op| !op.decode.fields.word_instr && op.decode.fields.is_eq())
        .count();
    if raw > max_rows {
        return None;
    }
    let (dev, num_rows) = math_cuda::trace_ops::gpu_build_eq_resident_from_devops(devops).ok()?;
    Some(devops_table(dev, num_rows, eq::cols::NUM_COLUMNS))
}

pub(crate) fn build_bytewise_resident_tables_from_devops(
    cpu_ops: &[CpuOperation],
    devops: &Devops,
    max_rows: usize,
) -> Option<Tables> {
    if gpu_trace_disabled() {
        return None;
    }
    let raw = cpu_ops
        .iter()
        .filter(|op| {
            let f = &op.decode.fields;
            !f.word_instr && (f.is_and() || f.is_or() || f.is_xor())
        })
        .count();
    if raw > max_rows {
        return None;
    }
    let (dev, num_rows) =
        math_cuda::trace_ops::gpu_build_bytewise_resident_from_devops(devops).ok()?;
    Some(devops_table(dev, num_rows, bytewise::cols::NUM_COLUMNS))
}

pub(crate) fn build_dvrm_resident_tables_from_devops(
    cpu_ops: &[CpuOperation],
    devops: &Devops,
    max_rows: usize,
) -> Option<Tables> {
    if gpu_trace_disabled() {
        return None;
    }
    let raw = cpu_ops
        .iter()
        .filter(|op| {
            let f = &op.decode.fields;
            (!f.word_instr && f.is_divrem()) || (f.word_instr && !f.add && !f.sub && f.is_divrem())
        })
        .count();
    if raw > max_rows {
        return None;
    }
    let (dev, num_rows) =
        math_cuda::trace_ops::gpu_build_dvrm_full_resident_from_devops(devops).ok()?;
    Some(devops_table(dev, num_rows, dvrm::cols::NUM_COLUMNS))
}

pub(crate) fn build_mul_resident_tables_from_devops(
    cpu_ops: &[CpuOperation],
    devops: &Devops,
    max_rows: usize,
) -> Option<Tables> {
    if gpu_trace_disabled() {
        return None;
    }
    let raw: usize = cpu_ops
        .iter()
        .map(|op| {
            let f = &op.decode.fields;
            let elig = if f.word_instr { !f.add && !f.sub } else { true };
            if !elig {
                0
            } else if f.is_mul() {
                1
            } else if f.is_divrem() {
                2
            } else {
                0
            }
        })
        .sum();
    if raw > max_rows {
        return None;
    }
    let (dev, num_rows) =
        math_cuda::trace_ops::gpu_build_mul_full_resident_from_devops(devops).ok()?;
    Some(devops_table(dev, num_rows, mul::cols::NUM_COLUMNS))
}

pub(crate) fn build_branch_resident_tables_from_devops(
    cpu_ops: &[CpuOperation],
    devops: &Devops,
    max_rows: usize,
) -> Option<Tables> {
    if gpu_trace_disabled() {
        return None;
    }
    let raw = cpu_ops.iter().filter(|op| op.branch_cond).count();
    if raw > max_rows {
        return None;
    }
    let (dev, num_rows) = math_cuda::trace_ops::gpu_build_branch_resident_from_devops(devops).ok()?;
    Some(devops_table(dev, num_rows, branch::cols::NUM_COLUMNS))
}

/// Build the COMMIT (ECALL) trace table via the device fill (Phase 6, precompiles). The commit
/// ops are collected on host (they read memory); this moves the per-byte recursive fill to device.
/// Single table (never chunked, like `commit::generate_commit_trace`). `None` when the GPU path is
/// disabled / the resident flag is off. Byte-identical to the CPU generator.
pub(crate) fn build_commit_resident_table(
    commit_ops: &[commit::CommitOperation],
) -> Option<TraceTable<GoldilocksField, GoldilocksExtension>> {
    if gpu_trace_disabled() || !gpu_resident_chips_enabled() {
        return None;
    }
    let n = commit_ops.len();
    let mut flat = Vec::with_capacity(n * math_cuda::precompile::COMMIT_STRIDE);
    for op in commit_ops {
        flat.push(op.timestamp);
        flat.push(op.index);
        flat.push(op.address);
        flat.push(op.count);
        flat.push(op.first as u64);
        flat.push(op.end as u64);
        flat.push(op.value as u64);
    }
    let (dev, num_rows) = math_cuda::precompile::gpu_build_commit_trace_dev(&flat, n).ok()?;
    Some(devops_table(dev, num_rows, commit::cols::NUM_COLUMNS).pop().unwrap())
}

/// Build the main KECCAK (permute) trace table via the device fill (Phase 6). The keccak ops are
/// collected on host (input read from memory, output = `keccak_f1600`); this moves the per-op fill
/// (ts / addr bytes / input+output state bytes / state_ptr halfwords) to device. `None` when the
/// GPU path is disabled / the resident flag is off. Byte-identical to `keccak::generate_keccak_trace`.
pub(crate) fn build_keccak_resident_table(
    keccak_ops: &[keccak::KeccakOperation],
) -> Option<TraceTable<GoldilocksField, GoldilocksExtension>> {
    if gpu_trace_disabled() || !gpu_resident_chips_enabled() {
        return None;
    }
    let n = keccak_ops.len();
    let mut flat = Vec::with_capacity(n * math_cuda::precompile::KECCAK_TBL_STRIDE);
    for op in keccak_ops {
        flat.push(op.timestamp);
        flat.push(op.state_addr);
        flat.extend_from_slice(&op.input);
        flat.extend_from_slice(&op.output);
    }
    let (dev, num_rows) = math_cuda::precompile::gpu_build_keccak_trace_dev(&flat, n).ok()?;
    Some(devops_table(dev, num_rows, keccak::cols::NUM_COLUMNS).pop().unwrap())
}

/// Step C: recompute the per-step ECDAS carries (`c0/c1/c2`) ON DEVICE and write them back into
/// `ops` — which arrived carryless from `ecsm::compute_witness_carryless` (the ~190ms `conv`
/// limb-convolution bulk skipped on CPU). Packs the point + quotient limbs (same layout the ecdas
/// fill reads) and runs the `ecdas_carries` kernel (bit-exact with the CPU `carries_lambda/xr/yr`,
/// parity: `gpu_ecdas_carries_parity`). Returns `false` if the GPU path is unavailable — the caller
/// (under gpu_full) treats that as fatal, since the carryless steps would otherwise keep zero carries.
pub(crate) fn fill_ecdas_carries_device(ops: &mut [ecdas::EcdasOperation]) -> bool {
    if gpu_trace_disabled() || ops.is_empty() {
        return false;
    }
    let n = ops.len();
    let mut bytes = Vec::with_capacity(n * math_cuda::precompile::ECDAS_BSTRIDE);
    for op in ops.iter() {
        let s = &op.step;
        bytes.extend_from_slice(&s.x_g);
        bytes.extend_from_slice(&s.y_g);
        bytes.extend_from_slice(&s.x_a);
        bytes.extend_from_slice(&s.y_a);
        bytes.push(s.round);
        bytes.push(s.op);
        bytes.extend_from_slice(&s.x_r);
        bytes.extend_from_slice(&s.y_r);
        bytes.extend_from_slice(&s.lambda);
        bytes.extend_from_slice(&s.q0);
        bytes.extend_from_slice(&s.q1);
        bytes.extend_from_slice(&s.q2);
        bytes.push(s.next_op);
    }
    let carries = match math_cuda::precompile::gpu_build_ecdas_carries(&bytes, n) {
        Ok(c) => c,
        Err(_) => return false,
    };
    let cs = math_cuda::precompile::ECDAS_CSTRIDE;
    for (i, op) in ops.iter_mut().enumerate() {
        let base = i * cs;
        op.step.c0.copy_from_slice(&carries[base..base + 64]);
        op.step.c1.copy_from_slice(&carries[base + 64..base + 128]);
        op.step.c2.copy_from_slice(&carries[base + 128..base + 192]);
    }
    true
}

/// Build the ECDAS trace table via the device fill (Phase 6). Pure formatting of the precomputed
/// witness (byte coords + signed carries); the EC/modular math ran on CPU during execution. `None`
/// when the GPU path is disabled / the resident flag is off. Byte-identical to the CPU generator.
pub(crate) fn build_ecdas_resident_table(
    ops: &[ecdas::EcdasOperation],
) -> Option<TraceTable<GoldilocksField, GoldilocksExtension>> {
    if gpu_trace_disabled() || !gpu_resident_chips_enabled() {
        return None;
    }
    let n = ops.len();
    let mut bytes = Vec::with_capacity(n * math_cuda::precompile::ECDAS_BSTRIDE);
    let mut carries = Vec::with_capacity(n * math_cuda::precompile::ECDAS_CSTRIDE);
    let mut ts = Vec::with_capacity(n);
    for op in ops {
        let s = &op.step;
        bytes.extend_from_slice(&s.x_g);
        bytes.extend_from_slice(&s.y_g);
        bytes.extend_from_slice(&s.x_a);
        bytes.extend_from_slice(&s.y_a);
        bytes.push(s.round);
        bytes.push(s.op);
        bytes.extend_from_slice(&s.x_r);
        bytes.extend_from_slice(&s.y_r);
        bytes.extend_from_slice(&s.lambda);
        bytes.extend_from_slice(&s.q0);
        bytes.extend_from_slice(&s.q1);
        bytes.extend_from_slice(&s.q2);
        bytes.push(s.next_op);
        carries.extend_from_slice(&s.c0);
        carries.extend_from_slice(&s.c1);
        carries.extend_from_slice(&s.c2);
        ts.push(op.timestamp);
    }
    let (dev, num_rows) =
        math_cuda::precompile::gpu_build_ecdas_trace_dev(&bytes, &carries, &ts, n).ok()?;
    Some(devops_table(dev, num_rows, ecdas::cols::NUM_COLUMNS).pop().unwrap())
}

/// Build the ECSM trace table via the device fill (Phase 6). Pure formatting of the precomputed
/// witness (byte coords, k bits, halfwords, signed carries). `None` when the GPU path is disabled /
/// the resident flag is off. Byte-identical to the CPU generator.
pub(crate) fn build_ecsm_resident_table(
    ops: &[ecsm::EcsmOperation],
) -> Option<TraceTable<GoldilocksField, GoldilocksExtension>> {
    if gpu_trace_disabled() || !gpu_resident_chips_enabled() {
        return None;
    }
    let n = ops.len();
    let mut bytes = Vec::with_capacity(n * math_cuda::precompile::ECSM_BSTRIDE);
    let mut carries = Vec::with_capacity(n * math_cuda::precompile::ECSM_CSTRIDE);
    let mut addrs = Vec::with_capacity(n * math_cuda::precompile::ECSM_ASTRIDE);
    for op in ops {
        let wt = &op.witness;
        bytes.extend_from_slice(&wt.x_r);
        bytes.extend_from_slice(&wt.y_r);
        bytes.extend_from_slice(&wt.k);
        bytes.extend_from_slice(&wt.x_g);
        bytes.extend_from_slice(&wt.y_g);
        bytes.extend_from_slice(&wt.x2);
        bytes.extend_from_slice(&wt.q0);
        bytes.extend_from_slice(&wt.q1);
        bytes.extend_from_slice(&wt.x_g_sub_p);
        bytes.extend_from_slice(&wt.k_sub_n);
        bytes.extend_from_slice(&wt.x_r_sub_p);
        bytes.push(wt.len_k);
        carries.extend_from_slice(&wt.c0);
        carries.extend_from_slice(&wt.c1);
        addrs.push(op.timestamp);
        addrs.push(op.addr_xg);
        addrs.push(op.addr_k);
        addrs.push(op.addr_xr);
    }
    let (dev, num_rows) =
        math_cuda::precompile::gpu_build_ecsm_trace_dev(&bytes, &carries, &addrs, n).ok()?;
    Some(devops_table(dev, num_rows, ecsm::cols::NUM_COLUMNS).pop().unwrap())
}

/// Build the KECCAK_RND (per-round) trace table via the device fill (Phase 6) — recomputes the 24
/// permutation rounds per op on device. `keccak_ops` are the same KeccakOperations that drive the
/// main keccak table; only `input`+`timestamp` are used (rounds recompute the rest). `None` when the
/// GPU path is disabled / resident flag off. Byte-identical to `keccak_rnd::generate_keccak_rnd_trace`.
pub(crate) fn build_keccak_rnd_resident_table(
    keccak_ops: &[keccak::KeccakOperation],
) -> Option<TraceTable<GoldilocksField, GoldilocksExtension>> {
    if gpu_trace_disabled() || !gpu_resident_chips_enabled() {
        return None;
    }
    let n = keccak_ops.len();
    let mut flat = Vec::with_capacity(n * math_cuda::precompile::KECCAK_RND_STRIDE);
    for op in keccak_ops {
        flat.push(op.timestamp);
        flat.extend_from_slice(&op.input);
    }
    let (dev, num_rows) = math_cuda::precompile::gpu_build_keccak_rnd_trace_dev(&flat, n).ok()?;
    Some(devops_table(dev, num_rows, keccak_rnd::cols::NUM_COLUMNS).pop().unwrap())
}

/// Extract the (packed decode, rv1, arg2) SoA the resident ALU chip chains read from cpu_ops.
fn alu_soa(cpu_ops: &[CpuOperation]) -> (Vec<u64>, Vec<u64>, Vec<u64>) {
    let n = cpu_ops.len();
    let (mut packed, mut rv1, mut arg2) =
        (Vec::with_capacity(n), Vec::with_capacity(n), Vec::with_capacity(n));
    for op in cpu_ops {
        packed.push(op.decode.fields.pack());
        rv1.push(op.rv1);
        arg2.push(op.arg2);
    }
    (packed, rv1, arg2)
}

/// Shared GPU table-build dispatcher. Mirrors `chunk_and_generate`'s chunking
/// (`ops.chunks(max_rows)`, with one empty chunk when there are no ops so the
/// table still emits its padded minimum), builds each chunk on device via
/// `build_chunk`, and collects the resident tables. Returns `None` when the
/// kill-switch is set (`LAMBDA_VM_CPU_TRACE`) or any chunk fails to build, so the
/// caller falls back to the CPU generator. Every `gpu_build_*_tables` entry point
/// is a thin wrapper over this.
fn gpu_build_tables<T>(
    ops: &[T],
    max_rows: usize,
    build_chunk: impl Fn(&[T]) -> Option<TraceTable<GoldilocksField, GoldilocksExtension>>,
) -> Option<Vec<TraceTable<GoldilocksField, GoldilocksExtension>>> {
    if gpu_trace_disabled() {
        return None;
    }
    let chunks: Vec<&[T]> = if ops.is_empty() {
        vec![&[][..]]
    } else {
        ops.chunks(max_rows).collect()
    };
    let mut tables = Vec::with_capacity(chunks.len());
    for chunk in chunks {
        tables.push(build_chunk(chunk)?);
    }
    Some(tables)
}

// =============================================================================
// CPU table (the first table built on device — see GPU-TRACEGEN-DESIGN-V2 §P1)
// =============================================================================

/// Marshal one chunk of `CpuOperation`s into the packed layout the `trace_cpu`
/// kernel consumes (stride `CPU_OP_STRIDE` u64/op). The kernel does the same
/// bit-slicing as `cpu::generate_cpu_trace`, so this only copies fields — no
/// per-column encoding on the host.
pub(crate) fn pack_cpu_ops(chunk: &[CpuOperation]) -> Vec<u64> {
    let stride = math_cuda::trace_cpu::CPU_OP_STRIDE;
    let mut packed = vec![0u64; chunk.len() * stride];
    for (i, op) in chunk.iter().enumerate() {
        let f = &op.decode.fields;
        let flags = (f.word_instr as u64)
            | ((f.read_register1 as u64) << 1)
            | ((f.read_register2 as u64) << 2)
            | ((f.write_register as u64) << 3)
            | ((f.alu as u64) << 4)
            | ((f.add as u64) << 5)
            | ((f.sub as u64) << 6)
            | ((f.memory as u64) << 7)
            | ((f.branch as u64) << 8)
            | ((f.ecall as u64) << 9)
            | ((op.branch_cond as u64) << 10);
        let bytes = (f.rs1 as u64)
            | ((f.rs2 as u64) << 8)
            | ((f.rd as u64) << 16)
            | ((f.half_instruction_length as u64) << 24)
            | ((f.alu_flags as u64) << 32)
            | ((f.mem_flags as u64) << 40);
        let b = i * stride;
        packed[b] = op.timestamp;
        packed[b + 1] = op.decode.pc;
        packed[b + 2] = op.decode.imm;
        packed[b + 3] = op.next_pc;
        packed[b + 4] = op.rvd;
        packed[b + 5] = op.rv1;
        packed[b + 6] = op.rv2;
        packed[b + 7] = op.arg2;
        packed[b + 8] = op.res;
        packed[b + 9] = flags;
        packed[b + 10] = bytes;
    }
    packed
}

/// Build one CPU trace-table chunk on device: pack ops → GPU fill → a
/// `TraceTable` whose main matrix is resident on device (fed to the LDE with no
/// upload). The host main table is a zeroed placeholder sized for the correct
/// `num_rows`; it is never read on the GPU commit path (commit consumes the
/// device buffer, the aux build reads the resident snapshot, queries gather from
/// the device tree). Returns `None` if the GPU build fails, so the caller can
/// fall back to the CPU generator.
fn build_cpu_chunk(
    chunk: &[CpuOperation],
) -> Option<TraceTable<GoldilocksField, GoldilocksExtension>> {
    let n = chunk.len();
    let num_rows = n.next_power_of_two().max(4);
    let last_ts = chunk.last().map(|op| op.timestamp).unwrap_or(0);
    let packed = pack_cpu_ops(chunk);
    let dev = math_cuda::trace_cpu::gpu_build_cpu_trace(&packed, n, num_rows, last_ts).ok()?;
    let mut trace = TraceTable::new_main(
        crate::tables::types::zeroed_fe_vec(num_rows * cpu::cols::NUM_COLUMNS),
        cpu::cols::NUM_COLUMNS,
        1,
    );
    trace.set_main_input_dev(Arc::new(dev));
    Some(trace)
}

/// Build all CPU trace-table chunks on device, mirroring `chunk_and_generate`'s
/// chunking (`max_rows`, one empty chunk when there are no ops). Returns `None`
/// when the kill-switch is set or any chunk fails to build, so the caller falls
/// back to the CPU generator.
pub(crate) fn gpu_build_cpu_trace_tables(
    cpu_ops: &[CpuOperation],
    max_rows: usize,
) -> Option<Vec<TraceTable<GoldilocksField, GoldilocksExtension>>> {
    gpu_build_tables(cpu_ops, max_rows, build_cpu_chunk)
}

// =============================================================================
// MEMW_R (register fast path — the biggest table, ~15M rows on ethrex)
// =============================================================================

/// Build one MEMW_R trace-table chunk on device: marshal the walked `RegRow`s into
/// the SoA the `memw_register_fill` kernel consumes, fill the 10 columns row-major
/// on device, and leave the matrix RESIDENT (fed to the LDE with no full-column
/// upload). The host main table is a zeroed placeholder sized to `num_rows`; it is
/// never read on the GPU commit path. The `old_*` come from the (correct,
/// precompile-inclusive) sequential walk — so this is program-agnostic — with only
/// the compact `RegRow` fields uploaded, not the full column matrix. Returns `None`
/// on GPU failure so the caller can fall back to the CPU fill.
fn build_memw_register_chunk(
    chunk: &[RegRow],
) -> Option<TraceTable<GoldilocksField, GoldilocksExtension>> {
    let n = chunk.len();
    let num_rows = n.next_power_of_two().max(4);

    let mut reg_addr = Vec::with_capacity(n);
    let mut ts = Vec::with_capacity(n);
    let mut value = Vec::with_capacity(n);
    let mut is_read = Vec::with_capacity(n);
    let mut old_value = Vec::with_capacity(n);
    let mut old_ts = Vec::with_capacity(n);
    for r in chunk {
        let (ra, t, v, ir, ov, ot) = r.fill_soa();
        reg_addr.push(ra);
        ts.push(t);
        value.push(v);
        is_read.push(ir);
        old_value.push(ov);
        old_ts.push(ot);
    }

    let dev = math_cuda::trace_cpu::gpu_fill_memw_register(
        &reg_addr, &ts, &value, &is_read, &old_value, &old_ts, num_rows,
    )
    .ok()?;

    let mut trace = TraceTable::new_main(
        crate::tables::types::zeroed_fe_vec(num_rows * memw_register::cols::NUM_COLUMNS),
        memw_register::cols::NUM_COLUMNS,
        1,
    );
    trace.set_main_input_dev(Arc::new(dev));
    Some(trace)
}

/// Build all MEMW_R trace-table chunks on device, mirroring `chunk_and_generate`'s
/// chunking (`max_rows`, one empty chunk when there are no rows). Returns `None`
/// when the kill-switch is set or any chunk fails to build, so the caller falls
/// back to the CPU fill.
pub(crate) fn gpu_build_memw_register_tables(
    rows: &[RegRow],
    max_rows: usize,
) -> Option<Vec<TraceTable<GoldilocksField, GoldilocksExtension>>> {
    gpu_build_tables(rows, max_rows, build_memw_register_chunk)
}

/// Build the MEMW_R chunk tables DIRECTLY on device from the register walk (no 447MB row
/// download + 9.3M host `RegRow` build). Runs `gpu_walk_route_memw_register_ecall_chunked` — one
/// device walk fills every capped chunk table resident (via `fill_chunk_on`) — then wraps each
/// resident `[height*NCOLS]` buffer into a `TraceTable` (placeholder host matrix + `set_main_input_dev`,
/// exactly like `build_memw_register_chunk`). Returns the tables + the rare fallback subset
/// `[reg_addr, ts, value, old_value, old_ts, is_read]` (emit order) for the host to route to
/// aligned/general. The MEMW_R memory bus is partition-invariant across chunks, so the multi-chunk
/// split is correct regardless of which chunk a row lands in. `None` on GPU failure.
#[allow(clippy::too_many_arguments)]
pub(crate) fn gpu_build_memw_register_tables_from_walk(
    packed: &[u64],
    rv1: &[u64],
    rv2: &[u64],
    rvd: &[u64],
    next_pc: &[u64],
    ecall_op_index: &[u32],
    ecall_reg_addr: &[u32],
    ecall_ts: &[u64],
    ecall_value: &[u64],
    ecall_is_read: &[u8],
    init_value: &[u64],
    init_ts: u64,
    nbins: u32,
    max_rows: usize,
) -> Option<(Vec<TraceTable<GoldilocksField, GoldilocksExtension>>, Vec<[u64; 6]>, Vec<u64>)> {
    if gpu_trace_disabled() {
        return None;
    }
    // `is_half` is the MEMW_R IS_HALFWORD ts-delta histogram (65536 bins) the walk already computes as
    // a byproduct. Surfaced here so the caller can feed it to the BITWISE merge and SKIP the redundant
    // second register walk inside `gpu_bitwise_hist_full` (fix (a): walk-once).
    let (bufs, num_rows, is_half, fallback) =
        math_cuda::trace_walk::gpu_walk_route_memw_register_ecall_chunked(
            packed, rv1, rv2, rvd, next_pc, ecall_op_index, ecall_reg_addr, ecall_ts, ecall_value,
            ecall_is_read, init_value, init_ts, nbins, max_rows,
        )
        .ok()?;
    let ncols = memw_register::cols::NUM_COLUMNS;
    let tables: Vec<_> = bufs
        .into_iter()
        .enumerate()
        .map(|(c, buf)| {
            let lo = c * max_rows;
            let hi = ((c + 1) * max_rows).min(num_rows);
            // Match `fill_chunk_on`'s height exactly (empty walk → one height-4 table).
            let height = (hi.saturating_sub(lo)).next_power_of_two().max(4);
            let mut trace = TraceTable::new_main(
                crate::tables::types::zeroed_fe_vec(height * ncols),
                ncols,
                1,
            );
            trace.set_main_input_dev(Arc::new(buf));
            trace
        })
        .collect();
    Some((tables, fallback, is_half))
}

/// A1 — build the resident PAGE trace tables (one per page, ascending `page_bases`) DIRECTLY on
/// device from the initial image + the device final-memory snapshot, retiring the ~285ms host
/// `generate_page_trace_from_dense` column fill. Each `[page_size*5]` resident buffer is wrapped in
/// a `TraceTable` (placeholder host matrix + `set_main_input_dev`, like the resident chips).
/// Bit-identical to the host page trace (exclude_touched=false). `None` on GPU failure.
/// A1 NO-GO (kept as ready infra, currently OFF): device PAGE tables VERIFY (device fill → host
/// matrix → preprocessed commit), but `gen_pages` is parallel-hidden in p5 so this is a net loss
/// (see trace_builder.rs "A1 NO-GO"). Retained for a future fully-resident preprocessed path.
#[allow(dead_code)]
pub(crate) fn gpu_build_page_tables_from_snapshot(
    page_bases: &[u64],
    img_addr: &[u64],
    img_val: &[u64],
    snap_addr: &[u64],
    snap_val: &[u64],
    snap_ts: &[u64],
) -> Option<Vec<TraceTable<GoldilocksField, GoldilocksExtension>>> {
    if gpu_trace_disabled() {
        return None;
    }
    let page_size = crate::tables::page::DEFAULT_PAGE_SIZE;
    let ncols = crate::tables::page::cols::NUM_COLUMNS;
    let bufs = math_cuda::trace_cpu::gpu_page_fill_snapshot(
        page_bases, page_size as u64, img_addr, img_val, snap_addr, snap_val, snap_ts, ncols,
    )
    .ok()?;
    // Each `buf` is a row-major `[page_size*ncols]` canonical-u64 PAGE table filled on device and
    // downloaded. Wrap it as a REAL host trace matrix (not a resident device buffer): PAGE is a
    // preprocessed 2-tree table whose commit AND R2 aux build both read the host matrix, so the
    // device buffer must be materialized here. Bit-identical to `generate_page_trace_from_dense`.
    let tables = bufs
        .into_iter()
        .map(|buf| {
            debug_assert_eq!(buf.len(), page_size * ncols);
            TraceTable::new_main(crate::tables::types::fe_vec_from_u64(buf), ncols, 1)
        })
        .collect();
    Some(tables)
}

// =============================================================================
// MEMW_A (aligned memory — the biggest remaining uploader, ~2M rows on ethrex)
// =============================================================================

/// Pack one aligned `MemwOperation` into the stride-`MEMW_ALIGNED_STRIDE` layout
/// the `memw_aligned_fill` kernel consumes (see `trace_cpu.cu`). The op is already
/// walked (old_value/old_timestamp filled), so this only copies fields; value/old
/// (`[u32; 8]` each) pack two-per-u64.
pub(crate) fn pack_memw_aligned_op(
    op: &MemwOperation,
) -> [u64; math_cuda::trace_cpu::MEMW_ALIGNED_STRIDE] {
    let flags = (op.is_register as u64) | ((op.is_read as u64) << 1) | ((op.width as u64) << 8);
    let v = &op.value;
    let o = &op.old;
    [
        flags,
        op.base_address,
        op.timestamp,
        op.old_timestamp[0],
        v[0] as u64 | ((v[1] as u64) << 32),
        v[2] as u64 | ((v[3] as u64) << 32),
        v[4] as u64 | ((v[5] as u64) << 32),
        v[6] as u64 | ((v[7] as u64) << 32),
        o[0] as u64 | ((o[1] as u64) << 32),
        o[2] as u64 | ((o[3] as u64) << 32),
        o[4] as u64 | ((o[5] as u64) << 32),
        o[6] as u64 | ((o[7] as u64) << 32),
    ]
}

/// Inverse of [`pack_memw_aligned_op`]: reconstruct a `MemwOperation` from a device MEMW_A row
/// (option B). The 12-stride carries only `old_timestamp[0]` — aligned ops share one timestamp
/// across all bytes (`is_aligned_op`), and both the MEMW_A fill and `collect_lt_from_memw_aligned`
/// read only index 0 — so the remaining `old_timestamp[1..8]` are semantically absent and left 0.
pub(crate) fn unpack_memw_aligned(r: &[u64; math_cuda::trace_cpu::MEMW_ALIGNED_STRIDE]) -> MemwOperation {
    let flags = r[0];
    let is_register = (flags & 1) != 0;
    let is_read = (flags & 2) != 0;
    let width = ((flags >> 8) & 0xFF) as u8;
    let unpack4 = |lo: usize| {
        let mut b = [0u32; 8];
        for k in 0..4 {
            b[2 * k] = (r[lo + k] & 0xFFFF_FFFF) as u32;
            b[2 * k + 1] = (r[lo + k] >> 32) as u32;
        }
        b
    };
    let value = unpack4(4);
    let old = unpack4(8);
    let mut old_timestamp = [0u64; 8];
    old_timestamp[0] = r[3];
    MemwOperation::new(is_register, r[1], value, r[2], width, is_read).with_old(old, old_timestamp)
}

/// Build one MEMW_A trace-table chunk on device: pack the walked ops → GPU fill →
/// a resident matrix fed to the LDE with no full-column upload (only the compact
/// packed ops are H2D'd). Returns `None` on GPU failure so the caller falls back.
fn build_memw_aligned_chunk(
    chunk: &[MemwOperation],
) -> Option<TraceTable<GoldilocksField, GoldilocksExtension>> {
    let n = chunk.len();
    let num_rows = n.next_power_of_two().max(4);
    let mut packed = Vec::with_capacity(n * math_cuda::trace_cpu::MEMW_ALIGNED_STRIDE);
    for op in chunk {
        packed.extend_from_slice(&pack_memw_aligned_op(op));
    }
    let dev = math_cuda::trace_cpu::gpu_build_memw_aligned_trace(&packed, n, num_rows).ok()?;
    let mut trace = TraceTable::new_main(
        crate::tables::types::zeroed_fe_vec(num_rows * memw_aligned::cols::NUM_COLUMNS),
        memw_aligned::cols::NUM_COLUMNS,
        1,
    );
    trace.set_main_input_dev(Arc::new(dev));
    Some(trace)
}

/// Build all MEMW_A trace-table chunks on device, mirroring `chunk_and_generate`'s
/// chunking. Returns `None` when the kill-switch is set or any chunk fails to build.
pub(crate) fn gpu_build_memw_aligned_tables(
    ops: &[MemwOperation],
    max_rows: usize,
) -> Option<Vec<TraceTable<GoldilocksField, GoldilocksExtension>>> {
    gpu_build_tables(ops, max_rows, build_memw_aligned_chunk)
}

// =============================================================================
// LOAD / STORE (per-row-map memory tables — same shape as MEMW_A)
// =============================================================================

/// Pack one `LoadOperation` into the `load_fill` stride (see `trace_cpu.cu`).
pub(crate) fn pack_load_op(op: &LoadOperation) -> [u64; math_cuda::trace_cpu::LOAD_STRIDE] {
    let flags = (op.signed as u64) | ((op.width as u64) << 8);
    let r = &op.res;
    [
        flags,
        op.base_address,
        op.timestamp,
        r[0] | (r[1] << 32),
        r[2] | (r[3] << 32),
        r[4] | (r[5] << 32),
        r[6] | (r[7] << 32),
    ]
}

/// Pack one `StoreOperation` into the `store_fill` stride (see `trace_cpu.cu`).
pub(crate) fn pack_store_op(op: &StoreOperation) -> [u64; math_cuda::trace_cpu::STORE_STRIDE] {
    let flags = (op.write2 as u64) | ((op.write4 as u64) << 1) | ((op.write8 as u64) << 2);
    [flags, op.base_address, op.timestamp, op.value]
}

fn build_load_chunk(
    chunk: &[LoadOperation],
) -> Option<TraceTable<GoldilocksField, GoldilocksExtension>> {
    let n = chunk.len();
    let num_rows = n.next_power_of_two().max(4);
    let mut packed = Vec::with_capacity(n * math_cuda::trace_cpu::LOAD_STRIDE);
    for op in chunk {
        packed.extend_from_slice(&pack_load_op(op));
    }
    let dev = math_cuda::trace_cpu::gpu_build_load_trace(&packed, n, num_rows).ok()?;
    let mut trace = TraceTable::new_main(
        crate::tables::types::zeroed_fe_vec(num_rows * load::cols::NUM_COLUMNS),
        load::cols::NUM_COLUMNS,
        1,
    );
    trace.set_main_input_dev(Arc::new(dev));
    Some(trace)
}

pub(crate) fn gpu_build_load_tables(
    ops: &[LoadOperation],
    max_rows: usize,
) -> Option<Vec<TraceTable<GoldilocksField, GoldilocksExtension>>> {
    gpu_build_tables(ops, max_rows, build_load_chunk)
}

fn build_store_chunk(
    chunk: &[StoreOperation],
) -> Option<TraceTable<GoldilocksField, GoldilocksExtension>> {
    let n = chunk.len();
    let num_rows = n.next_power_of_two().max(4);
    let mut packed = Vec::with_capacity(n * math_cuda::trace_cpu::STORE_STRIDE);
    for op in chunk {
        packed.extend_from_slice(&pack_store_op(op));
    }
    let dev = math_cuda::trace_cpu::gpu_build_store_trace(&packed, n, num_rows).ok()?;
    let mut trace = TraceTable::new_main(
        crate::tables::types::zeroed_fe_vec(num_rows * store::cols::NUM_COLUMNS),
        store::cols::NUM_COLUMNS,
        1,
    );
    trace.set_main_input_dev(Arc::new(dev));
    Some(trace)
}

pub(crate) fn gpu_build_store_tables(
    ops: &[StoreOperation],
    max_rows: usize,
) -> Option<Vec<TraceTable<GoldilocksField, GoldilocksExtension>>> {
    gpu_build_tables(ops, max_rows, build_store_chunk)
}

// =============================================================================
// SHIFT (ALU table, no dedup — the kernel recomputes the shift aux on device)
// =============================================================================

/// Pack one `ShiftOperation` into the `shift_fill` stride (see `trace_cpu.cu`):
/// value (4×u16 in_halves), full shift_amount, and the flag bits. The kernel
/// recomputes bit_shift/zbs/x/y/limb_shift/out, so only 3 u64/op upload.
pub(crate) fn pack_shift_op(op: &ShiftOperation) -> [u64; math_cuda::trace_cpu::SHIFT_STRIDE] {
    let h = &op.in_halves;
    let value =
        (h[0] as u64) | ((h[1] as u64) << 16) | ((h[2] as u64) << 32) | ((h[3] as u64) << 48);
    let flags = (op.direction as u64) | ((op.signed as u64) << 1) | ((op.word_instr as u64) << 2);
    [value, op.shift_amount, flags]
}

fn build_shift_chunk(
    chunk: &[ShiftOperation],
) -> Option<TraceTable<GoldilocksField, GoldilocksExtension>> {
    let n = chunk.len();
    let num_rows = n.next_power_of_two().max(4);
    let mut packed = Vec::with_capacity(n * math_cuda::trace_cpu::SHIFT_STRIDE);
    for op in chunk {
        packed.extend_from_slice(&pack_shift_op(op));
    }
    let dev = math_cuda::trace_cpu::gpu_build_shift_trace(&packed, n, num_rows).ok()?;
    let mut trace = TraceTable::new_main(
        crate::tables::types::zeroed_fe_vec(num_rows * shift::cols::NUM_COLUMNS),
        shift::cols::NUM_COLUMNS,
        1,
    );
    trace.set_main_input_dev(Arc::new(dev));
    Some(trace)
}

pub(crate) fn gpu_build_shift_tables(
    ops: &[ShiftOperation],
    max_rows: usize,
) -> Option<Vec<TraceTable<GoldilocksField, GoldilocksExtension>>> {
    gpu_build_tables(ops, max_rows, build_shift_chunk)
}

// =============================================================================
// LT (ALU dedup table): host per-chunk HashMap dedup → device fill (compute)
// =============================================================================

/// Pack one unique `LtOperation` + its multiplicity into the `lt_fill` stride.
pub(crate) fn pack_lt_op(op: &LtOperation, mult: u64) -> [u64; math_cuda::trace_cpu::LT_STRIDE] {
    let flags = (op.signed as u64) | ((op.invert as u64) << 1);
    [op.lhs, op.rhs, flags, mult]
}

/// Build one LT trace-table chunk on device. Dedup happens HERE on the host (the
/// same per-chunk HashMap `generate_lt_trace` uses), then the unique ops + summed
/// multiplicities are filled on device. LT rides the permutation-invariant ALU
/// bus, so any row order is valid (validated by multiset/prove, not byte order).
fn build_lt_chunk(
    chunk: &[LtOperation],
) -> Option<TraceTable<GoldilocksField, GoldilocksExtension>> {
    let mut map: HashMap<LtOperation, u64> = HashMap::new();
    for op in chunk {
        *map.entry(op.clone()).or_insert(0) += 1;
    }
    let unique: Vec<(LtOperation, u64)> = map.into_iter().collect();
    let n = unique.len();
    let num_rows = n.next_power_of_two().max(4);
    let mut packed = Vec::with_capacity(n * math_cuda::trace_cpu::LT_STRIDE);
    for (op, mult) in &unique {
        packed.extend_from_slice(&pack_lt_op(op, *mult));
    }
    let dev = math_cuda::trace_cpu::gpu_build_lt_trace(&packed, n, num_rows).ok()?;
    let mut trace = TraceTable::new_main(
        crate::tables::types::zeroed_fe_vec(num_rows * lt::cols::NUM_COLUMNS),
        lt::cols::NUM_COLUMNS,
        1,
    );
    trace.set_main_input_dev(Arc::new(dev));
    Some(trace)
}

pub(crate) fn gpu_build_lt_tables(
    ops: &[LtOperation],
    max_rows: usize,
) -> Option<Vec<TraceTable<GoldilocksField, GoldilocksExtension>>> {
    // Each chunk dedups independently (matching `generate_lt_trace` per chunk).
    gpu_build_tables(ops, max_rows, build_lt_chunk)
}

// =============================================================================
// EQ (ALU dedup table): host per-chunk HashMap dedup → device fill (compute)
// =============================================================================

/// Pack one unique `EqOperation` + its multiplicity into the `eq_fill` stride.
pub(crate) fn pack_eq_op(op: &EqOperation, mult: u64) -> [u64; math_cuda::trace_cpu::EQ_STRIDE] {
    let flags = op.invert as u64;
    [op.a, op.b, flags, mult]
}

/// Build one EQ trace-table chunk on device. Dedup happens HERE on the host (the
/// same per-chunk HashMap `generate_eq_trace` uses), then the unique ops + summed
/// multiplicities are filled on device. EQ rides the permutation-invariant ALU
/// bus, so any row order is valid (validated by multiset/prove, not byte order).
fn build_eq_chunk(
    chunk: &[EqOperation],
) -> Option<TraceTable<GoldilocksField, GoldilocksExtension>> {
    let mut map: HashMap<EqOperation, u64> = HashMap::new();
    for op in chunk {
        *map.entry(op.clone()).or_insert(0) += 1;
    }
    let unique: Vec<(EqOperation, u64)> = map.into_iter().collect();
    let n = unique.len();
    let num_rows = n.next_power_of_two().max(4);
    let mut packed = Vec::with_capacity(n * math_cuda::trace_cpu::EQ_STRIDE);
    for (op, mult) in &unique {
        packed.extend_from_slice(&pack_eq_op(op, *mult));
    }
    let dev = math_cuda::trace_cpu::gpu_build_eq_trace(&packed, n, num_rows).ok()?;
    let mut trace = TraceTable::new_main(
        crate::tables::types::zeroed_fe_vec(num_rows * eq::cols::NUM_COLUMNS),
        eq::cols::NUM_COLUMNS,
        1,
    );
    trace.set_main_input_dev(Arc::new(dev));
    Some(trace)
}

pub(crate) fn gpu_build_eq_tables(
    ops: &[EqOperation],
    max_rows: usize,
) -> Option<Vec<TraceTable<GoldilocksField, GoldilocksExtension>>> {
    // Each chunk dedups independently (matching `generate_eq_trace` per chunk).
    gpu_build_tables(ops, max_rows, build_eq_chunk)
}

// =============================================================================
// BYTEWISE (ALU dedup table): host per-chunk HashMap dedup → device fill (compute)
// =============================================================================

/// Pack one unique `BytewiseOperation` + its multiplicity into the `bytewise_fill`
/// stride.
pub(crate) fn pack_bytewise_op(
    op: &BytewiseOperation,
    mult: u64,
) -> [u64; math_cuda::trace_cpu::BYTEWISE_STRIDE] {
    [op.a, op.b, op.op as u64, mult]
}

/// Build one BYTEWISE trace-table chunk on device. Dedup happens HERE on the host
/// (the same per-chunk HashMap `generate_bytewise_trace` uses), then the unique ops
/// with summed multiplicities are filled on device. BYTEWISE rides the
/// permutation-invariant ALU bus, so any row order is valid (validated by
/// multiset/prove, not byte order).
fn build_bytewise_chunk(
    chunk: &[BytewiseOperation],
) -> Option<TraceTable<GoldilocksField, GoldilocksExtension>> {
    let mut map: HashMap<BytewiseOperation, u64> = HashMap::new();
    for op in chunk {
        *map.entry(op.clone()).or_insert(0) += 1;
    }
    let unique: Vec<(BytewiseOperation, u64)> = map.into_iter().collect();
    let n = unique.len();
    let num_rows = n.next_power_of_two().max(4);
    let mut packed = Vec::with_capacity(n * math_cuda::trace_cpu::BYTEWISE_STRIDE);
    for (op, mult) in &unique {
        packed.extend_from_slice(&pack_bytewise_op(op, *mult));
    }
    let dev = math_cuda::trace_cpu::gpu_build_bytewise_trace(&packed, n, num_rows).ok()?;
    let mut trace = TraceTable::new_main(
        crate::tables::types::zeroed_fe_vec(num_rows * bytewise::cols::NUM_COLUMNS),
        bytewise::cols::NUM_COLUMNS,
        1,
    );
    trace.set_main_input_dev(Arc::new(dev));
    Some(trace)
}

pub(crate) fn gpu_build_bytewise_tables(
    ops: &[BytewiseOperation],
    max_rows: usize,
) -> Option<Vec<TraceTable<GoldilocksField, GoldilocksExtension>>> {
    // Each chunk dedups independently (matching `generate_bytewise_trace` per chunk).
    gpu_build_tables(ops, max_rows, build_bytewise_chunk)
}

// =============================================================================
// MUL (ALU dedup table): host per-chunk HashMap dedup → device fill (128-bit
// product + sign-extended convolution recomputed on device)
// =============================================================================

/// Pack one unique `MulOperation` + its split multiplicities into the `mul_fill`
/// stride.
pub(crate) fn pack_mul_op(
    op: &MulOperation,
    mu_lo: u64,
    mu_hi: u64,
) -> [u64; math_cuda::trace_cpu::MUL_STRIDE] {
    let flags = (op.lhs_signed as u64) | ((op.rhs_signed as u64) << 1);
    [op.lhs, op.rhs, flags, mu_lo, mu_hi]
}

/// Build one MUL trace-table chunk on device. Dedup happens HERE on the host (the
/// same per-chunk HashMap `generate_mul_trace` uses, keyed by op with mu_lo/mu_hi
/// accumulated from each `wants_hi`), then the unique ops + both multiplicities are
/// filled on device (the kernel recomputes the 128-bit product and raw_products).
/// MUL rides the permutation-invariant ALU bus, so any row order is valid
/// (validated by multiset/prove, not byte order).
fn build_mul_chunk(
    chunk: &[(MulOperation, bool)],
) -> Option<TraceTable<GoldilocksField, GoldilocksExtension>> {
    // (mu_lo, mu_hi) per unique op, matching `MulMultiplicities`.
    let mut map: HashMap<MulOperation, (u64, u64)> = HashMap::new();
    for (op, wants_hi) in chunk {
        let e = map.entry(op.clone()).or_insert((0, 0));
        if *wants_hi {
            e.1 += 1;
        } else {
            e.0 += 1;
        }
    }
    let unique: Vec<(MulOperation, (u64, u64))> = map.into_iter().collect();
    let n = unique.len();
    let num_rows = n.next_power_of_two().max(4);
    let mut packed = Vec::with_capacity(n * math_cuda::trace_cpu::MUL_STRIDE);
    for (op, (mu_lo, mu_hi)) in &unique {
        packed.extend_from_slice(&pack_mul_op(op, *mu_lo, *mu_hi));
    }
    let dev = math_cuda::trace_cpu::gpu_build_mul_trace(&packed, n, num_rows).ok()?;
    let mut trace = TraceTable::new_main(
        crate::tables::types::zeroed_fe_vec(num_rows * mul::cols::NUM_COLUMNS),
        mul::cols::NUM_COLUMNS,
        1,
    );
    trace.set_main_input_dev(Arc::new(dev));
    Some(trace)
}

pub(crate) fn gpu_build_mul_tables(
    ops: &[(MulOperation, bool)],
    max_rows: usize,
) -> Option<Vec<TraceTable<GoldilocksField, GoldilocksExtension>>> {
    // Each chunk dedups independently (matching `generate_mul_trace` per chunk).
    gpu_build_tables(ops, max_rows, build_mul_chunk)
}

// =============================================================================
// DVRM (ALU dedup table): host per-chunk HashMap dedup → device fill (signed/
// unsigned division & remainder + abs/sign aux recomputed on device)
// =============================================================================

/// Pack one unique `DvrmOperation` + its split multiplicities into the `dvrm_fill`
/// stride.
pub(crate) fn pack_dvrm_op(
    op: &DvrmOperation,
    mu_q: u64,
    mu_r: u64,
) -> [u64; math_cuda::trace_cpu::DVRM_STRIDE] {
    let flags = op.signed as u64;
    [op.n, op.d, flags, mu_q, mu_r]
}

/// Build one DVRM trace-table chunk on device. Dedup happens HERE on the host (the
/// same per-chunk HashMap `generate_dvrm_trace` uses, keyed by op with mu_q/mu_r
/// accumulated from each `wants_remainder`), then the unique ops + both
/// multiplicities are filled on device (the kernel recomputes the quotient,
/// remainder, and abs/sign aux). DVRM rides the permutation-invariant ALU bus, so
/// any row order is valid (validated by multiset/prove, not byte order).
fn build_dvrm_chunk(
    chunk: &[(DvrmOperation, bool)],
) -> Option<TraceTable<GoldilocksField, GoldilocksExtension>> {
    // (mu_q, mu_r) per unique op, matching `DvrmMultiplicities`.
    let mut map: HashMap<DvrmOperation, (u64, u64)> = HashMap::new();
    for (op, wants_remainder) in chunk {
        let e = map.entry(op.clone()).or_insert((0, 0));
        if *wants_remainder {
            e.1 += 1;
        } else {
            e.0 += 1;
        }
    }
    let unique: Vec<(DvrmOperation, (u64, u64))> = map.into_iter().collect();
    let n = unique.len();
    let num_rows = n.next_power_of_two().max(4);
    let mut packed = Vec::with_capacity(n * math_cuda::trace_cpu::DVRM_STRIDE);
    for (op, (mu_q, mu_r)) in &unique {
        packed.extend_from_slice(&pack_dvrm_op(op, *mu_q, *mu_r));
    }
    let dev = math_cuda::trace_cpu::gpu_build_dvrm_trace(&packed, n, num_rows).ok()?;
    let mut trace = TraceTable::new_main(
        crate::tables::types::zeroed_fe_vec(num_rows * dvrm::cols::NUM_COLUMNS),
        dvrm::cols::NUM_COLUMNS,
        1,
    );
    trace.set_main_input_dev(Arc::new(dev));
    Some(trace)
}

pub(crate) fn gpu_build_dvrm_tables(
    ops: &[(DvrmOperation, bool)],
    max_rows: usize,
) -> Option<Vec<TraceTable<GoldilocksField, GoldilocksExtension>>> {
    // Each chunk dedups independently (matching `generate_dvrm_trace` per chunk).
    gpu_build_tables(ops, max_rows, build_dvrm_chunk)
}

// =============================================================================
// BRANCH (branch/jump target dedup table): host per-chunk HashMap dedup → device
// fill (next_pc = (base + offset) & ~1 recomputed on device)
// =============================================================================

/// Pack one unique `BranchOperation` + its multiplicity into the `branch_fill`
/// stride.
pub(crate) fn pack_branch_op(
    op: &BranchOperation,
    mult: u64,
) -> [u64; math_cuda::trace_cpu::BRANCH_STRIDE] {
    let flags = op.jalr as u64;
    [op.pc, op.offset, op.register, flags, mult]
}

/// Build one BRANCH trace-table chunk on device. Dedup happens HERE on the host
/// (the same per-chunk HashMap `generate_branch_trace` uses), then the unique ops +
/// summed multiplicities are filled on device (the kernel recomputes next_pc and
/// its byte/half decomposition). BRANCH is a permutation-invariant lookup table, so
/// any row order is valid (validated by multiset/prove, not byte order).
fn build_branch_chunk(
    chunk: &[BranchOperation],
) -> Option<TraceTable<GoldilocksField, GoldilocksExtension>> {
    let mut map: HashMap<BranchOperation, u64> = HashMap::new();
    for op in chunk {
        *map.entry(op.clone()).or_insert(0) += 1;
    }
    let unique: Vec<(BranchOperation, u64)> = map.into_iter().collect();
    let n = unique.len();
    let num_rows = n.next_power_of_two().max(4);
    let mut packed = Vec::with_capacity(n * math_cuda::trace_cpu::BRANCH_STRIDE);
    for (op, mult) in &unique {
        packed.extend_from_slice(&pack_branch_op(op, *mult));
    }
    let dev = math_cuda::trace_cpu::gpu_build_branch_trace(&packed, n, num_rows).ok()?;
    let mut trace = TraceTable::new_main(
        crate::tables::types::zeroed_fe_vec(num_rows * branch::cols::NUM_COLUMNS),
        branch::cols::NUM_COLUMNS,
        1,
    );
    trace.set_main_input_dev(Arc::new(dev));
    Some(trace)
}

pub(crate) fn gpu_build_branch_tables(
    ops: &[BranchOperation],
    max_rows: usize,
) -> Option<Vec<TraceTable<GoldilocksField, GoldilocksExtension>>> {
    // Each chunk dedups independently (matching `generate_branch_trace` per chunk).
    gpu_build_tables(ops, max_rows, build_branch_chunk)
}

// =============================================================================
// CPU32 (delegated *W instructions — per-row, no dedup, like the CPU table)
// =============================================================================

/// Pack one `Cpu32Operation` into the `cpu32_fill` stride (see `trace_cpu.cu`). The
/// kernel recomputes the sign-extension aux (arg1/arg2/rvd, sign bits), so this
/// only copies the raw fields.
pub(crate) fn pack_cpu32_op(op: &Cpu32Operation) -> [u64; math_cuda::trace_cpu::CPU32_STRIDE] {
    let flags = (op.read_register1 as u64)
        | ((op.read_register2 as u64) << 1)
        | ((op.write_register as u64) << 2)
        | ((op.alu as u64) << 3)
        | ((op.add as u64) << 4)
        | ((op.sub as u64) << 5);
    let bytes = (op.rs1 as u64)
        | ((op.rs2 as u64) << 8)
        | ((op.rd as u64) << 16)
        | ((op.alu_flags as u64) << 24)
        | ((op.half_instruction_length as u64) << 32);
    [
        op.timestamp,
        op.pc,
        op.rv1,
        op.rv2,
        op.imm,
        op.res,
        flags,
        bytes,
    ]
}

/// Build one CPU32 trace-table chunk on device: pack the ops → GPU fill → a resident
/// matrix fed to the LDE with no full-column upload. Per-row (μ=1, no dedup), so the
/// device fill is byte-identical to `generate_cpu32_trace`.
fn build_cpu32_chunk(
    chunk: &[Cpu32Operation],
) -> Option<TraceTable<GoldilocksField, GoldilocksExtension>> {
    let n = chunk.len();
    let num_rows = n.next_power_of_two().max(4);
    let mut packed = Vec::with_capacity(n * math_cuda::trace_cpu::CPU32_STRIDE);
    for op in chunk {
        packed.extend_from_slice(&pack_cpu32_op(op));
    }
    let dev = math_cuda::trace_cpu::gpu_build_cpu32_trace(&packed, n, num_rows).ok()?;
    let mut trace = TraceTable::new_main(
        crate::tables::types::zeroed_fe_vec(num_rows * cpu32::cols::NUM_COLUMNS),
        cpu32::cols::NUM_COLUMNS,
        1,
    );
    trace.set_main_input_dev(Arc::new(dev));
    Some(trace)
}

pub(crate) fn gpu_build_cpu32_tables(
    ops: &[Cpu32Operation],
    max_rows: usize,
) -> Option<Vec<TraceTable<GoldilocksField, GoldilocksExtension>>> {
    gpu_build_tables(ops, max_rows, build_cpu32_chunk)
}

// =============================================================================
// MEMW (general / unaligned memory — per-op, same MemwOperation as MEMW_A)
// =============================================================================

/// Pack one walked `MemwOperation` into the `memw_fill` stride (see `trace_cpu.cu`).
/// value/old (`[u32; 8]` each) pack two-per-u64; the 8 old_timestamps are full u64.
pub(crate) fn pack_memw_op(op: &MemwOperation) -> [u64; math_cuda::trace_cpu::MEMW_STRIDE] {
    let flags = (op.is_register as u64) | ((op.is_read as u64) << 1) | ((op.width as u64) << 8);
    let v = &op.value;
    let o = &op.old;
    let ot = &op.old_timestamp;
    [
        flags,
        op.base_address,
        op.timestamp,
        v[0] as u64 | ((v[1] as u64) << 32),
        v[2] as u64 | ((v[3] as u64) << 32),
        v[4] as u64 | ((v[5] as u64) << 32),
        v[6] as u64 | ((v[7] as u64) << 32),
        o[0] as u64 | ((o[1] as u64) << 32),
        o[2] as u64 | ((o[3] as u64) << 32),
        o[4] as u64 | ((o[5] as u64) << 32),
        o[6] as u64 | ((o[7] as u64) << 32),
        ot[0],
        ot[1],
        ot[2],
        ot[3],
        ot[4],
        ot[5],
        ot[6],
        ot[7],
    ]
}

/// Inverse of [`pack_memw_op`]: reconstruct a `MemwOperation` from a device MEMW (general) row
/// (option B). The 19-stride captures the entire struct (all 8 old bytes + all 8 old_timestamps),
/// so this is fully lossless. The device walk zeros old_value/old_timestamp BEYOND the access
/// width (don't-care per store.toml — those bytes are not on the memory bus), matching the walk's
/// validated output; within-width fields are bit-exact vs the CPU.
pub(crate) fn unpack_memw(r: &[u64; math_cuda::trace_cpu::MEMW_STRIDE]) -> MemwOperation {
    let flags = r[0];
    let is_register = (flags & 1) != 0;
    let is_read = (flags & 2) != 0;
    let width = ((flags >> 8) & 0xFF) as u8;
    let unpack4 = |lo: usize| {
        let mut b = [0u32; 8];
        for k in 0..4 {
            b[2 * k] = (r[lo + k] & 0xFFFF_FFFF) as u32;
            b[2 * k + 1] = (r[lo + k] >> 32) as u32;
        }
        b
    };
    let value = unpack4(3);
    let old = unpack4(7);
    let mut old_timestamp = [0u64; 8];
    for (j, slot) in old_timestamp.iter_mut().enumerate() {
        *slot = r[11 + j];
    }
    MemwOperation::new(is_register, r[1], value, r[2], width, is_read).with_old(old, old_timestamp)
}

/// Build one MEMW (general) trace-table chunk on device: pack the walked ops → GPU
/// fill → a resident matrix fed to the LDE with no full-column upload. Per-row (no
/// dedup), so the device fill is byte-identical to `generate_memw_trace`.
fn build_memw_chunk(
    chunk: &[MemwOperation],
) -> Option<TraceTable<GoldilocksField, GoldilocksExtension>> {
    let n = chunk.len();
    let num_rows = n.next_power_of_two().max(4);
    let mut packed = Vec::with_capacity(n * math_cuda::trace_cpu::MEMW_STRIDE);
    for op in chunk {
        packed.extend_from_slice(&pack_memw_op(op));
    }
    let dev = math_cuda::trace_cpu::gpu_build_memw_trace(&packed, n, num_rows).ok()?;
    let mut trace = TraceTable::new_main(
        crate::tables::types::zeroed_fe_vec(num_rows * memw::cols::NUM_COLUMNS),
        memw::cols::NUM_COLUMNS,
        1,
    );
    trace.set_main_input_dev(Arc::new(dev));
    Some(trace)
}

pub(crate) fn gpu_build_memw_tables(
    ops: &[MemwOperation],
    max_rows: usize,
) -> Option<Vec<TraceTable<GoldilocksField, GoldilocksExtension>>> {
    gpu_build_tables(ops, max_rows, build_memw_chunk)
}
