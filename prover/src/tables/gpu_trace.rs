//! GPU trace-generation glue (cuda-gated).
//!
//! Builds the per-program `PackedDecode` array on the host (the instruction
//! decoder stays on CPU — it runs once per program) and drives the math-cuda
//! CPU-table kernel, producing a host `TraceTable` byte-identical to
//! `cpu::generate_cpu_trace_from_logs`. The kernel does the per-cycle
//! `from_log` + column fill on device; this module only marshals inputs and
//! wraps the result.

use executor::vm::instruction::decoding::Instruction;
use executor::vm::logs::Log;
use executor::vm::memory::U64HashMap;
use math_cuda::trace::{
    self, DEC_ALU_FLAGS, DEC_FLAGS, DEC_HIL, DEC_IMM, DEC_MEM_FLAGS, DEC_RD, DEC_RS1, DEC_RS2,
    DEC_STRIDE, F_ADD, F_ALU, F_BRANCH, F_ECALL, F_MEMORY, F_READ_REGISTER1, F_READ_REGISTER2,
    F_SUB, F_WORD_INSTR, F_WRITE_REGISTER,
};
use std::sync::atomic::{AtomicU64, Ordering};

use stark::trace::TraceTable;

use super::bitwise::BitwiseOperation;
use super::branch::BranchOperation;
use super::bytewise::BytewiseOperation;
use super::commit::CommitOperation;
use super::cpu32::Cpu32Operation;
use super::register::{
    self, FinalRegisterStateMap, NUM_REGISTER_ADDRESSES,
};
use super::cpu::cols;
use super::dvrm::DvrmOperation;
use super::eq::EqOperation;
use super::load::LoadOperation;
use super::lt::LtOperation;
use super::memw::MemwOperation;
use super::mul::MulOperation;
use super::store::StoreOperation;
use super::shift::ShiftOperation;
use super::types::{DecodeEntry, FE, GoldilocksExtension, GoldilocksField};

/// Counts successful GPU CPU-table builds. Lets tests confirm the device path
/// actually fired (vs a silent CPU fallback).
pub static GPU_CPU_TABLE_BUILDS: AtomicU64 = AtomicU64::new(0);

/// Number of CPU tables built on the GPU so far this process.
pub fn gpu_cpu_table_builds() -> u64 {
    GPU_CPU_TABLE_BUILDS.load(Ordering::Relaxed)
}

/// Counts successful GPU LT-table builds (per `from_elf_and_logs` call with
/// non-empty LT ops).
pub static GPU_LT_TABLE_BUILDS: AtomicU64 = AtomicU64::new(0);

/// Number of LT tables built on the GPU so far this process.
pub fn gpu_lt_table_builds() -> u64 {
    GPU_LT_TABLE_BUILDS.load(Ordering::Relaxed)
}

/// Counts successful GPU EQ-table builds.
pub static GPU_EQ_TABLE_BUILDS: AtomicU64 = AtomicU64::new(0);
pub fn gpu_eq_table_builds() -> u64 {
    GPU_EQ_TABLE_BUILDS.load(Ordering::Relaxed)
}

/// Counts successful GPU BYTEWISE-table builds.
pub static GPU_BYTEWISE_TABLE_BUILDS: AtomicU64 = AtomicU64::new(0);
pub fn gpu_bytewise_table_builds() -> u64 {
    GPU_BYTEWISE_TABLE_BUILDS.load(Ordering::Relaxed)
}

/// Counts successful GPU SHIFT-table builds.
pub static GPU_SHIFT_TABLE_BUILDS: AtomicU64 = AtomicU64::new(0);
pub fn gpu_shift_table_builds() -> u64 {
    GPU_SHIFT_TABLE_BUILDS.load(Ordering::Relaxed)
}

/// Counts successful GPU MUL-table builds.
pub static GPU_MUL_TABLE_BUILDS: AtomicU64 = AtomicU64::new(0);
pub fn gpu_mul_table_builds() -> u64 {
    GPU_MUL_TABLE_BUILDS.load(Ordering::Relaxed)
}

/// Counts successful GPU DVRM-table builds.
pub static GPU_DVRM_TABLE_BUILDS: AtomicU64 = AtomicU64::new(0);
pub fn gpu_dvrm_table_builds() -> u64 {
    GPU_DVRM_TABLE_BUILDS.load(Ordering::Relaxed)
}

/// Counts successful GPU MEMW_R-table builds.
pub static GPU_MEMW_REGISTER_TABLE_BUILDS: AtomicU64 = AtomicU64::new(0);
pub fn gpu_memw_register_table_builds() -> u64 {
    GPU_MEMW_REGISTER_TABLE_BUILDS.load(Ordering::Relaxed)
}

/// Counts successful GPU LOAD-table builds.
pub static GPU_LOAD_TABLE_BUILDS: AtomicU64 = AtomicU64::new(0);
pub fn gpu_load_table_builds() -> u64 {
    GPU_LOAD_TABLE_BUILDS.load(Ordering::Relaxed)
}

/// Counts successful GPU STORE-table builds.
pub static GPU_STORE_TABLE_BUILDS: AtomicU64 = AtomicU64::new(0);
pub fn gpu_store_table_builds() -> u64 {
    GPU_STORE_TABLE_BUILDS.load(Ordering::Relaxed)
}

/// Counts successful GPU MEMW_A-table builds.
pub static GPU_MEMW_ALIGNED_TABLE_BUILDS: AtomicU64 = AtomicU64::new(0);
pub fn gpu_memw_aligned_table_builds() -> u64 {
    GPU_MEMW_ALIGNED_TABLE_BUILDS.load(Ordering::Relaxed)
}

/// Counts successful GPU MEMW-table builds.
pub static GPU_MEMW_TABLE_BUILDS: AtomicU64 = AtomicU64::new(0);
pub fn gpu_memw_table_builds() -> u64 {
    GPU_MEMW_TABLE_BUILDS.load(Ordering::Relaxed)
}

/// Counts successful GPU BITWISE-table builds.
pub static GPU_BITWISE_TABLE_BUILDS: AtomicU64 = AtomicU64::new(0);
pub fn gpu_bitwise_table_builds() -> u64 {
    GPU_BITWISE_TABLE_BUILDS.load(Ordering::Relaxed)
}

/// Counts successful GPU PAGE-table builds (one per page).
pub static GPU_PAGE_TABLE_BUILDS: AtomicU64 = AtomicU64::new(0);
pub fn gpu_page_table_builds() -> u64 {
    GPU_PAGE_TABLE_BUILDS.load(Ordering::Relaxed)
}

/// Counts successful GPU BRANCH-table builds.
pub static GPU_BRANCH_TABLE_BUILDS: AtomicU64 = AtomicU64::new(0);
pub fn gpu_branch_table_builds() -> u64 {
    GPU_BRANCH_TABLE_BUILDS.load(Ordering::Relaxed)
}

/// Counts successful GPU COMMIT-table builds.
pub static GPU_COMMIT_TABLE_BUILDS: AtomicU64 = AtomicU64::new(0);
pub fn gpu_commit_table_builds() -> u64 {
    GPU_COMMIT_TABLE_BUILDS.load(Ordering::Relaxed)
}

/// Counts successful GPU REGISTER-table builds.
pub static GPU_REGISTER_TABLE_BUILDS: AtomicU64 = AtomicU64::new(0);
pub fn gpu_register_table_builds() -> u64 {
    GPU_REGISTER_TABLE_BUILDS.load(Ordering::Relaxed)
}

/// Counts successful GPU HALT-table builds.
pub static GPU_HALT_TABLE_BUILDS: AtomicU64 = AtomicU64::new(0);
pub fn gpu_halt_table_builds() -> u64 {
    GPU_HALT_TABLE_BUILDS.load(Ordering::Relaxed)
}

/// Counts successful GPU CPU32-table builds.
pub static GPU_CPU32_TABLE_BUILDS: AtomicU64 = AtomicU64::new(0);
pub fn gpu_cpu32_table_builds() -> u64 {
    GPU_CPU32_TABLE_BUILDS.load(Ordering::Relaxed)
}

/// Convert a device column-major buffer into a host `TraceTable` (no device
/// handle — for small/preprocessed-split tables that stay on the host commit
/// path).
fn device_to_host_table(
    dev: math_cuda::trace::DeviceMainCols,
) -> Option<TraceTable<GoldilocksField, GoldilocksExtension>> {
    let (ncols, nrows) = (dev.ncols, dev.nrows);
    let host = dev.to_host().ok()?;
    let columns: Vec<Vec<FE>> = (0..ncols)
        .map(|c| {
            host[c * nrows..c * nrows + nrows]
                .iter()
                .map(|&v| FE::from(v))
                .collect()
        })
        .collect();
    Some(TraceTable::from_columns_main(columns, 1))
}

/// Build the dense `PackedDecode` array (`DEC_STRIDE` u64 per PC, indexed by
/// `(pc - base) >> 1` — RISC-V is 2-byte aligned) from the program's
/// instruction map. Calls the same `DecodeEntry::from_instruction` the CPU
/// trace path uses, so the per-PC fields match exactly. Returns `(array, base)`.
pub fn build_packed_decode(instructions: &U64HashMap<Instruction>) -> (Vec<u64>, u64) {
    let base = *instructions.keys().min().expect("non-empty program");
    let maxpc = *instructions.keys().max().unwrap();
    let slots = (((maxpc - base) >> 1) + 1) as usize;
    let mut arr = vec![0u64; slots * DEC_STRIDE];
    for (&pc, &instr) in instructions.iter() {
        let de = DecodeEntry::from_instruction(pc, instr, 4);
        let f = de.fields;
        let o = (((pc - base) >> 1) as usize) * DEC_STRIDE;
        let mut flags = 0u64;
        flags |= (f.read_register1 as u64) << F_READ_REGISTER1;
        flags |= (f.read_register2 as u64) << F_READ_REGISTER2;
        flags |= (f.write_register as u64) << F_WRITE_REGISTER;
        flags |= (f.word_instr as u64) << F_WORD_INSTR;
        flags |= (f.alu as u64) << F_ALU;
        flags |= (f.add as u64) << F_ADD;
        flags |= (f.sub as u64) << F_SUB;
        flags |= (f.memory as u64) << F_MEMORY;
        flags |= (f.branch as u64) << F_BRANCH;
        flags |= (f.ecall as u64) << F_ECALL;
        arr[o + DEC_FLAGS] = flags;
        arr[o + DEC_RS1] = f.rs1 as u64;
        arr[o + DEC_RS2] = f.rs2 as u64;
        arr[o + DEC_RD] = f.rd as u64;
        arr[o + DEC_HIL] = f.half_instruction_length as u64;
        arr[o + DEC_ALU_FLAGS] = f.alu_flags as u64;
        arr[o + DEC_MEM_FLAGS] = f.mem_flags as u64;
        arr[o + DEC_IMM] = de.imm;
    }
    (arr, base)
}

/// Build the CPU trace tables on GPU, chunked exactly like the CPU path
/// (`cpu_ops.chunks(max_rows)` → one `TraceTable` per chunk; see
/// `chunk_and_generate`). Each chunk is byte-identical to the CPU builder.
///
/// `max_rows` is the CPU chunk size (`MaxRowsConfig::cpu`). Timestamps stay
/// global via the kernel's `row_offset = chunk_index * max_rows`. Returns
/// `None` on empty input or any GPU error, so the caller falls back to the
/// CPU-built tables.
pub fn gpu_build_cpu_trace_tables(
    logs: &[Log],
    instructions: &U64HashMap<Instruction>,
    max_rows: usize,
) -> Option<Vec<TraceTable<GoldilocksField, GoldilocksExtension>>> {
    if logs.is_empty() {
        return None; // let the CPU path handle the empty/padding-only case
    }
    let (decode, base) = build_packed_decode(instructions);
    let ncols = cols::NUM_COLUMNS;

    let mut tables = Vec::with_capacity(logs.len().div_ceil(max_rows));
    for (chunk_index, chunk) in logs.chunks(max_rows).enumerate() {
        let n = chunk.len();
        let nrows = n.next_power_of_two().max(4);
        let row_offset = (chunk_index * max_rows) as u64;

        let mut logs_flat = Vec::with_capacity(5 * n);
        for l in chunk {
            logs_flat.extend_from_slice(&[
                l.current_pc,
                l.next_pc,
                l.src1_val,
                l.src2_val,
                l.dst_val,
            ]);
        }

        let dev = trace::gpu_build_cpu_trace(&logs_flat, &decode, base, row_offset, n, nrows).ok()?;
        let host = dev.to_host().ok()?; // column-major: NUM_COLUMNS * nrows
        let columns: Vec<Vec<FE>> = (0..ncols)
            .map(|c| host[c * nrows..c * nrows + nrows].iter().map(|&v| FE::from(v)).collect())
            .collect();
        // Keep the host columns (for num_rows / CPU consumers) AND attach the
        // resident device buffer so `commit_main_trace` runs the LDE from it.
        let mut tt = TraceTable::from_columns_main(columns, 1);
        tt.set_gpu_main_input(dev);
        tables.push(tt);
    }

    GPU_CPU_TABLE_BUILDS.fetch_add(1, Ordering::Relaxed);
    Some(tables)
}

/// Build the LT trace tables on GPU, chunked exactly like the CPU path
/// (`lt_ops.chunks(max_rows)` → one `TraceTable` per chunk, deduped within the
/// chunk; see `generate_lt_trace` + `chunk_and_generate`). Dedup runs on host
/// (`HashMap<LtOperation, multiplicity>`); the per-row column fill runs on
/// device. Row order need not match the CPU (the LT lookup bus is permutation-
/// invariant). Returns `None` on empty input or any GPU error → CPU fallback.
/// D2H a device-built column buffer into a host `TraceTable` (for num_rows /
/// CPU consumers) while keeping the resident handle attached for the LDE seam.
fn device_to_trace_table(
    dev: math_cuda::trace::DeviceMainCols,
) -> Option<TraceTable<GoldilocksField, GoldilocksExtension>> {
    let (ncols, nrows) = (dev.ncols, dev.nrows);
    let host = dev.to_host().ok()?;
    let columns: Vec<Vec<FE>> = (0..ncols)
        .map(|c| host[c * nrows..c * nrows + nrows].iter().map(|&v| FE::from(v)).collect())
        .collect();
    let mut tt = TraceTable::from_columns_main(columns, 1);
    tt.set_gpu_main_input(dev);
    Some(tt)
}

pub fn gpu_build_lt_trace_tables(
    lt_ops: &[LtOperation],
    max_rows: usize,
) -> Option<Vec<TraceTable<GoldilocksField, GoldilocksExtension>>> {
    if lt_ops.is_empty() {
        return None;
    }
    let mut tables = Vec::with_capacity(lt_ops.len().div_ceil(max_rows));
    for chunk in lt_ops.chunks(max_rows) {
        // Marshal raw ops (no host dedup) — the GPU groups them.
        let n = chunk.len();
        let mut a = vec![0u64; n];
        let mut b = vec![0u64; n];
        let mut c = vec![0u64; n];
        let sel = vec![0u64; n]; // single counter
        for (i, op) in chunk.iter().enumerate() {
            a[i] = op.lhs;
            b[i] = op.rhs;
            c[i] = (op.signed as u64) | ((op.invert as u64) << 1);
        }
        let dev = trace::gpu_build_lt_trace_deduped(&a, &b, &c, &sel).ok()?;
        tables.push(device_to_trace_table(dev)?);
    }
    GPU_LT_TABLE_BUILDS.fetch_add(1, Ordering::Relaxed);
    Some(tables)
}

/// EQ tables on GPU (GPU dedup + fill). Key = (a, b, invert).
pub fn gpu_build_eq_trace_tables(
    eq_ops: &[EqOperation],
    max_rows: usize,
) -> Option<Vec<TraceTable<GoldilocksField, GoldilocksExtension>>> {
    if eq_ops.is_empty() {
        return None;
    }
    let mut tables = Vec::with_capacity(eq_ops.len().div_ceil(max_rows));
    for chunk in eq_ops.chunks(max_rows) {
        let n = chunk.len();
        let mut a = vec![0u64; n];
        let mut b = vec![0u64; n];
        let mut c = vec![0u64; n];
        let sel = vec![0u64; n];
        for (i, op) in chunk.iter().enumerate() {
            a[i] = op.a;
            b[i] = op.b;
            c[i] = op.invert as u64;
        }
        let dev = trace::gpu_build_eq_trace_deduped(&a, &b, &c, &sel).ok()?;
        tables.push(device_to_trace_table(dev)?);
    }
    GPU_EQ_TABLE_BUILDS.fetch_add(1, Ordering::Relaxed);
    Some(tables)
}

/// BYTEWISE tables on GPU (GPU dedup + fill). Key = (a, b, op).
pub fn gpu_build_bytewise_trace_tables(
    bytewise_ops: &[BytewiseOperation],
    max_rows: usize,
) -> Option<Vec<TraceTable<GoldilocksField, GoldilocksExtension>>> {
    if bytewise_ops.is_empty() {
        return None;
    }
    let mut tables = Vec::with_capacity(bytewise_ops.len().div_ceil(max_rows));
    for chunk in bytewise_ops.chunks(max_rows) {
        let n = chunk.len();
        let mut a = vec![0u64; n];
        let mut b = vec![0u64; n];
        let mut c = vec![0u64; n];
        let sel = vec![0u64; n];
        for (i, op) in chunk.iter().enumerate() {
            a[i] = op.a;
            b[i] = op.b;
            c[i] = op.op as u64;
        }
        let dev = trace::gpu_build_bytewise_trace_deduped(&a, &b, &c, &sel).ok()?;
        tables.push(device_to_trace_table(dev)?);
    }
    GPU_BYTEWISE_TABLE_BUILDS.fetch_add(1, Ordering::Relaxed);
    Some(tables)
}

/// SHIFT tables on GPU. No dedup (one row per op, μ=1); chunked like the CPU
/// path. Mirrors generate_shift_trace + ShiftOperation::compute_aux.
pub fn gpu_build_shift_trace_tables(
    shift_ops: &[ShiftOperation],
    max_rows: usize,
) -> Option<Vec<TraceTable<GoldilocksField, GoldilocksExtension>>> {
    if shift_ops.is_empty() {
        return None;
    }
    let mut tables = Vec::with_capacity(shift_ops.len().div_ceil(max_rows));
    for chunk in shift_ops.chunks(max_rows) {
        let n = chunk.len();
        let nrows = n.next_power_of_two().max(4);
        let mut value = vec![0u64; n];
        let mut sa = vec![0u64; n];
        let mut flags = vec![0u64; n];
        for (i, op) in chunk.iter().enumerate() {
            value[i] = (op.in_halves[0] as u64)
                | ((op.in_halves[1] as u64) << 16)
                | ((op.in_halves[2] as u64) << 32)
                | ((op.in_halves[3] as u64) << 48);
            sa[i] = op.shift_amount;
            flags[i] = (op.direction as u64)
                | ((op.signed as u64) << 1)
                | ((op.word_instr as u64) << 2);
        }
        let dev = trace::gpu_build_shift_trace(&value, &sa, &flags, n, nrows).ok()?;
        tables.push(device_to_trace_table(dev)?);
    }
    GPU_SHIFT_TABLE_BUILDS.fetch_add(1, Ordering::Relaxed);
    Some(tables)
}

/// MUL tables on GPU (GPU dedup with dual counters + fill). Key = (lhs, rhs,
/// signed flags); selector = wants_hi. Input is (op, wants_hi) pairs.
pub fn gpu_build_mul_trace_tables(
    mul_ops: &[(MulOperation, bool)],
    max_rows: usize,
) -> Option<Vec<TraceTable<GoldilocksField, GoldilocksExtension>>> {
    if mul_ops.is_empty() {
        return None;
    }
    let mut tables = Vec::with_capacity(mul_ops.len().div_ceil(max_rows));
    for chunk in mul_ops.chunks(max_rows) {
        let n = chunk.len();
        let mut a = vec![0u64; n];
        let mut b = vec![0u64; n];
        let mut c = vec![0u64; n];
        let mut sel = vec![0u64; n];
        for (i, (op, wants_hi)) in chunk.iter().enumerate() {
            a[i] = op.lhs;
            b[i] = op.rhs;
            c[i] = (op.lhs_signed as u64) | ((op.rhs_signed as u64) << 1);
            sel[i] = *wants_hi as u64;
        }
        let dev = trace::gpu_build_mul_trace_deduped(&a, &b, &c, &sel).ok()?;
        tables.push(device_to_trace_table(dev)?);
    }
    GPU_MUL_TABLE_BUILDS.fetch_add(1, Ordering::Relaxed);
    Some(tables)
}

/// DVRM tables on GPU (GPU dedup with dual counters + fill). Key = (n, d,
/// signed); selector = wants_remainder. Input is (op, wants_remainder) pairs.
pub fn gpu_build_dvrm_trace_tables(
    dvrm_ops: &[(DvrmOperation, bool)],
    max_rows: usize,
) -> Option<Vec<TraceTable<GoldilocksField, GoldilocksExtension>>> {
    if dvrm_ops.is_empty() {
        return None;
    }
    let mut tables = Vec::with_capacity(dvrm_ops.len().div_ceil(max_rows));
    for chunk in dvrm_ops.chunks(max_rows) {
        let n = chunk.len();
        let mut a = vec![0u64; n];
        let mut b = vec![0u64; n];
        let mut c = vec![0u64; n];
        let mut sel = vec![0u64; n];
        for (i, (op, wants_remainder)) in chunk.iter().enumerate() {
            a[i] = op.n;
            b[i] = op.d;
            c[i] = op.signed as u64;
            sel[i] = *wants_remainder as u64;
        }
        let dev = trace::gpu_build_dvrm_trace_deduped(&a, &b, &c, &sel).ok()?;
        tables.push(device_to_trace_table(dev)?);
    }
    GPU_DVRM_TABLE_BUILDS.fetch_add(1, Ordering::Relaxed);
    Some(tables)
}

/// MEMW_R (register memory) tables on GPU. No dedup (one row per op); chunked
/// like the CPU path. The ops already carry `old`/`old_timestamp` from the
/// (CPU) memory-model walk — this only moves the per-row fill to the device.
/// Mirrors `generate_memw_register_trace`.
pub fn gpu_build_memw_register_trace_tables(
    memw_register_ops: &[MemwOperation],
    max_rows: usize,
) -> Option<Vec<TraceTable<GoldilocksField, GoldilocksExtension>>> {
    if memw_register_ops.is_empty() {
        return None;
    }
    let stride = math_cuda::trace::MEMW_REGISTER_STRIDE;
    let mut tables = Vec::with_capacity(memw_register_ops.len().div_ceil(max_rows));
    for chunk in memw_register_ops.chunks(max_rows) {
        let n = chunk.len();
        let nrows = n.next_power_of_two().max(4);
        let mut input = vec![0u64; n * stride];
        for (i, op) in chunk.iter().enumerate() {
            let base = i * stride;
            input[base] = op.base_address;
            input[base + 1] = op.timestamp;
            input[base + 2] = op.value[0];
            input[base + 3] = op.value[1];
            input[base + 4] = op.old[0];
            input[base + 5] = op.old[1];
            input[base + 6] = op.old_timestamp[0];
            input[base + 7] = op.is_read as u64;
        }
        let dev = trace::gpu_build_memw_register_trace(&input, n, nrows).ok()?;
        tables.push(device_to_trace_table(dev)?);
    }
    GPU_MEMW_REGISTER_TABLE_BUILDS.fetch_add(1, Ordering::Relaxed);
    Some(tables)
}

/// LOAD tables on GPU. No dedup (one row per op); chunked like the CPU path.
/// Mirrors `generate_load_trace`.
pub fn gpu_build_load_trace_tables(
    load_ops: &[LoadOperation],
    max_rows: usize,
) -> Option<Vec<TraceTable<GoldilocksField, GoldilocksExtension>>> {
    if load_ops.is_empty() {
        return None;
    }
    let stride = math_cuda::trace::LOAD_STRIDE;
    let mut tables = Vec::with_capacity(load_ops.len().div_ceil(max_rows));
    for chunk in load_ops.chunks(max_rows) {
        let n = chunk.len();
        let nrows = n.next_power_of_two().max(4);
        let mut input = vec![0u64; n * stride];
        for (i, op) in chunk.iter().enumerate() {
            let base = i * stride;
            input[base] = op.base_address;
            input[base + 1] = op.timestamp;
            input[base + 2] = op.width as u64;
            input[base + 3] = op.signed as u64;
            for j in 0..8 {
                input[base + 4 + j] = op.res[j];
            }
        }
        let dev = trace::gpu_build_load_trace(&input, n, nrows).ok()?;
        tables.push(device_to_trace_table(dev)?);
    }
    GPU_LOAD_TABLE_BUILDS.fetch_add(1, Ordering::Relaxed);
    Some(tables)
}

/// STORE tables on GPU. No dedup (one row per op); chunked like the CPU path.
/// Mirrors `generate_store_trace`.
pub fn gpu_build_store_trace_tables(
    store_ops: &[StoreOperation],
    max_rows: usize,
) -> Option<Vec<TraceTable<GoldilocksField, GoldilocksExtension>>> {
    if store_ops.is_empty() {
        return None;
    }
    let stride = math_cuda::trace::STORE_STRIDE;
    let mut tables = Vec::with_capacity(store_ops.len().div_ceil(max_rows));
    for chunk in store_ops.chunks(max_rows) {
        let n = chunk.len();
        let nrows = n.next_power_of_two().max(4);
        let mut input = vec![0u64; n * stride];
        for (i, op) in chunk.iter().enumerate() {
            let base = i * stride;
            input[base] = op.base_address;
            input[base + 1] = op.timestamp;
            input[base + 2] = op.value;
            input[base + 3] =
                (op.write2 as u64) | ((op.write4 as u64) << 1) | ((op.write8 as u64) << 2);
        }
        let dev = trace::gpu_build_store_trace(&input, n, nrows).ok()?;
        tables.push(device_to_trace_table(dev)?);
    }
    GPU_STORE_TABLE_BUILDS.fetch_add(1, Ordering::Relaxed);
    Some(tables)
}

/// MEMW_A (aligned memory) tables on GPU. No dedup (one row per op). Ops carry
/// old/old_timestamp from the (CPU) memory-model walk. Mirrors
/// `generate_memw_aligned_trace`.
pub fn gpu_build_memw_aligned_trace_tables(
    memw_aligned_ops: &[MemwOperation],
    max_rows: usize,
) -> Option<Vec<TraceTable<GoldilocksField, GoldilocksExtension>>> {
    if memw_aligned_ops.is_empty() {
        return None;
    }
    let stride = math_cuda::trace::MEMW_ALIGNED_STRIDE;
    let mut tables = Vec::with_capacity(memw_aligned_ops.len().div_ceil(max_rows));
    for chunk in memw_aligned_ops.chunks(max_rows) {
        let n = chunk.len();
        let nrows = n.next_power_of_two().max(4);
        let mut input = vec![0u64; n * stride];
        for (i, op) in chunk.iter().enumerate() {
            let base = i * stride;
            input[base] = op.is_register as u64;
            input[base + 1] = op.base_address;
            for j in 0..8 {
                input[base + 2 + j] = op.value[j];
            }
            input[base + 10] = op.timestamp;
            input[base + 11] = op.width as u64;
            for j in 0..8 {
                input[base + 12 + j] = op.old[j];
            }
            input[base + 20] = op.old_timestamp[0];
            input[base + 21] = op.is_read as u64;
        }
        let dev = trace::gpu_build_memw_aligned_trace(&input, n, nrows).ok()?;
        tables.push(device_to_trace_table(dev)?);
    }
    GPU_MEMW_ALIGNED_TABLE_BUILDS.fetch_add(1, Ordering::Relaxed);
    Some(tables)
}

/// MEMW (general memory) tables on GPU. No dedup (one row per op). Ops carry
/// old/old_timestamp (per-byte) from the (CPU) memory-model walk. Mirrors
/// `generate_memw_trace`.
pub fn gpu_build_memw_trace_tables(
    memw_ops: &[MemwOperation],
    max_rows: usize,
) -> Option<Vec<TraceTable<GoldilocksField, GoldilocksExtension>>> {
    if memw_ops.is_empty() {
        return None;
    }
    let stride = math_cuda::trace::MEMW_STRIDE;
    let mut tables = Vec::with_capacity(memw_ops.len().div_ceil(max_rows));
    for chunk in memw_ops.chunks(max_rows) {
        let n = chunk.len();
        let nrows = n.next_power_of_two().max(4);
        let mut input = vec![0u64; n * stride];
        for (i, op) in chunk.iter().enumerate() {
            let base = i * stride;
            input[base] = op.is_register as u64;
            input[base + 1] = op.base_address;
            for j in 0..8 {
                input[base + 2 + j] = op.value[j];
            }
            input[base + 10] = op.timestamp;
            input[base + 11] = op.width as u64;
            for j in 0..8 {
                input[base + 12 + j] = op.old[j];
            }
            for j in 0..8 {
                input[base + 20 + j] = op.old_timestamp[j];
            }
            input[base + 28] = op.is_read as u64;
        }
        let dev = trace::gpu_build_memw_trace(&input, n, nrows).ok()?;
        tables.push(device_to_trace_table(dev)?);
    }
    GPU_MEMW_TABLE_BUILDS.fetch_add(1, Ordering::Relaxed);
    Some(tables)
}

/// BITWISE table on GPU: the fixed 2^20-row lookup + the multiplicity histogram
/// over `bitwise_ops` (atomic scatter-add). Replaces `generate_bitwise_trace` +
/// `update_multiplicities`. Returns a host `TraceTable` with NO device handle —
/// BITWISE has a preprocessed/multiplicity commit split handled on the host
/// commit path, so we don't route it through the device-resident LDE seam.
pub fn gpu_build_bitwise_table(
    bitwise_ops: &[BitwiseOperation],
) -> Option<TraceTable<GoldilocksField, GoldilocksExtension>> {
    // Pack each lookup into x | y<<8 | z<<16 | type<<20 (type = enum discriminant
    // 0..9, matching mu_col = 11 + type in the kernel / update_multiplicities).
    let packed: Vec<u32> = bitwise_ops
        .iter()
        .map(|op| {
            (op.x as u32)
                | ((op.y as u32) << 8)
                | ((op.z as u32) << 16)
                | ((op.lookup_type as u32) << 20)
        })
        .collect();

    let dev = trace::gpu_build_bitwise_trace(&packed).ok()?;
    let (ncols, nrows) = (dev.ncols, dev.nrows);
    let host = dev.to_host().ok()?;
    let columns: Vec<Vec<FE>> = (0..ncols)
        .map(|c| {
            host[c * nrows..c * nrows + nrows]
                .iter()
                .map(|&v| FE::from(v))
                .collect()
        })
        .collect();
    let tt = TraceTable::from_columns_main(columns, 1);
    GPU_BITWISE_TABLE_BUILDS.fetch_add(1, Ordering::Relaxed);
    Some(tt)
}

/// Build one PAGE table on GPU: dense fill (OFFSET, INIT, FINI=INIT, TS=0) +
/// scatter of the page's touched cells. Returns a host `TraceTable` with no
/// device handle (PAGE has a preprocessed OFFSET/INIT split on the host commit
/// path). Mirrors `generate_page_trace`. `init_values` is the page's initial
/// bytes (empty = all-zero page); `cells` are `(offset, value, timestamp)`.
pub fn gpu_build_page_table(
    init_values: &[u8],
    page_size: usize,
    cells: &[(u32, u8, u64)],
) -> Option<TraceTable<GoldilocksField, GoldilocksExtension>> {
    let mut offsets = Vec::with_capacity(cells.len());
    let mut values = Vec::with_capacity(cells.len());
    let mut timestamps = Vec::with_capacity(cells.len());
    for &(off, val, ts) in cells {
        offsets.push(off);
        values.push(val);
        timestamps.push(ts);
    }
    let dev =
        trace::gpu_build_page_trace(init_values, page_size, &offsets, &values, &timestamps).ok()?;
    let (ncols, nrows) = (dev.ncols, dev.nrows);
    let host = dev.to_host().ok()?;
    let columns: Vec<Vec<FE>> = (0..ncols)
        .map(|c| {
            host[c * nrows..c * nrows + nrows]
                .iter()
                .map(|&v| FE::from(v))
                .collect()
        })
        .collect();
    let tt = TraceTable::from_columns_main(columns, 1);
    GPU_PAGE_TABLE_BUILDS.fetch_add(1, Ordering::Relaxed);
    Some(tt)
}

/// BRANCH tables on GPU. Dedup by (pc, offset, register, jalr) with summed
/// multiplicity on the host (small table), then a per-row GPU fill (next_pc
/// decomposition). Chunked like the CPU path. Mirrors `generate_branch_trace`.
pub fn gpu_build_branch_trace_tables(
    branch_ops: &[BranchOperation],
    max_rows: usize,
) -> Option<Vec<TraceTable<GoldilocksField, GoldilocksExtension>>> {
    if branch_ops.is_empty() {
        return None;
    }
    let stride = math_cuda::trace::BRANCH_STRIDE;
    let mut tables = Vec::with_capacity(branch_ops.len().div_ceil(max_rows));
    for chunk in branch_ops.chunks(max_rows) {
        // Host dedup (same per-chunk grouping as generate_branch_trace).
        let mut op_map: std::collections::HashMap<&BranchOperation, u64> =
            std::collections::HashMap::new();
        for op in chunk {
            *op_map.entry(op).or_insert(0) += 1;
        }
        let n = op_map.len();
        let nrows = n.next_power_of_two().max(4);
        let mut input = vec![0u64; n * stride];
        for (i, (op, mult)) in op_map.into_iter().enumerate() {
            let base = i * stride;
            input[base] = op.pc;
            input[base + 1] = op.offset;
            input[base + 2] = op.register;
            input[base + 3] = op.jalr as u64;
            input[base + 4] = mult;
        }
        let dev = trace::gpu_build_branch_trace(&input, n, nrows).ok()?;
        tables.push(device_to_trace_table(dev)?);
    }
    GPU_BRANCH_TABLE_BUILDS.fetch_add(1, Ordering::Relaxed);
    Some(tables)
}

/// COMMIT table on GPU (single table, no chunking; per-op fill, mu=1). Mirrors
/// `generate_commit_trace`.
pub fn gpu_build_commit_table(
    commit_ops: &[CommitOperation],
) -> Option<TraceTable<GoldilocksField, GoldilocksExtension>> {
    if commit_ops.is_empty() {
        return None;
    }
    let stride = math_cuda::trace::COMMIT_STRIDE;
    let n = commit_ops.len();
    let nrows = n.next_power_of_two().max(4);
    let mut input = vec![0u64; n * stride];
    for (i, op) in commit_ops.iter().enumerate() {
        let base = i * stride;
        input[base] = op.timestamp;
        input[base + 1] = op.index;
        input[base + 2] = op.address;
        input[base + 3] = op.count;
        input[base + 4] = op.first as u64;
        input[base + 5] = op.end as u64;
        input[base + 6] = op.value as u64;
    }
    let dev = trace::gpu_build_commit_trace(&input, n, nrows).ok()?;
    device_to_trace_table(dev).inspect(|_| {
        GPU_COMMIT_TABLE_BUILDS.fetch_add(1, Ordering::Relaxed);
    })
}

/// REGISTER table on GPU: dense fill (OFFSET/INIT/FINI/TS) + scatter of accessed
/// register words. OFFSET/INIT are preprocessed → host TraceTable (no device
/// handle). Mirrors `generate_register_trace`.
pub fn gpu_build_register_table(
    final_state: &FinalRegisterStateMap,
    entry_point: u64,
) -> Option<TraceTable<GoldilocksField, GoldilocksExtension>> {
    let addr_list = register::register_word_address_list();
    let init: Vec<u64> = addr_list
        .iter()
        .map(|&a| register::init_value_for_address(a, entry_point) as u64)
        .collect();
    let addrs: Vec<u64> = addr_list.to_vec();
    let nrows = NUM_REGISTER_ADDRESSES.next_power_of_two();

    // Resolve each accessed word to its row (position in addr_list).
    let mut row_of: std::collections::HashMap<u64, u32> = std::collections::HashMap::new();
    for (i, &a) in addr_list.iter().enumerate() {
        row_of.insert(a, i as u32);
    }
    let mut rows = Vec::new();
    let mut values = Vec::new();
    let mut tss = Vec::new();
    for (&word_addr, state) in final_state.iter() {
        if let Some(&row) = row_of.get(&word_addr) {
            rows.push(row);
            values.push(state.value as u64);
            tss.push(state.timestamp);
        }
    }

    let dev = trace::gpu_build_register_trace(&addrs, &init, nrows, &rows, &values, &tss).ok()?;
    device_to_host_table(dev).inspect(|_| {
        GPU_REGISTER_TABLE_BUILDS.fetch_add(1, Ordering::Relaxed);
    })
}

/// HALT table on GPU (single row). Mirrors `generate_halt_trace`.
pub fn gpu_build_halt_table(
    timestamp: u64,
    next_pc: u64,
) -> Option<TraceTable<GoldilocksField, GoldilocksExtension>> {
    let dev = trace::gpu_build_halt_trace(timestamp, next_pc).ok()?;
    device_to_host_table(dev).inspect(|_| {
        GPU_HALT_TABLE_BUILDS.fetch_add(1, Ordering::Relaxed);
    })
}

/// CPU32 tables on GPU (delegated *W instructions). No dedup (one row per op);
/// chunked like the CPU path. Mirrors `generate_cpu32_trace`.
pub fn gpu_build_cpu32_trace_tables(
    cpu32_ops: &[Cpu32Operation],
    max_rows: usize,
) -> Option<Vec<TraceTable<GoldilocksField, GoldilocksExtension>>> {
    if cpu32_ops.is_empty() {
        return None;
    }
    let stride = math_cuda::trace::CPU32_STRIDE;
    let mut tables = Vec::with_capacity(cpu32_ops.len().div_ceil(max_rows));
    for chunk in cpu32_ops.chunks(max_rows) {
        let n = chunk.len();
        let nrows = n.next_power_of_two().max(4);
        let mut input = vec![0u64; n * stride];
        for (i, op) in chunk.iter().enumerate() {
            let b = i * stride;
            input[b] = op.timestamp;
            input[b + 1] = op.pc;
            input[b + 2] = op.rs1 as u64;
            input[b + 3] = op.read_register1 as u64;
            input[b + 4] = op.rv1;
            input[b + 5] = op.rs2 as u64;
            input[b + 6] = op.read_register2 as u64;
            input[b + 7] = op.rv2;
            input[b + 8] = op.imm;
            input[b + 9] = op.res;
            input[b + 10] = op.rd as u64;
            input[b + 11] = op.write_register as u64;
            input[b + 12] = op.alu as u64;
            input[b + 13] = op.alu_flags as u64;
            input[b + 14] = op.add as u64;
            input[b + 15] = op.sub as u64;
            input[b + 16] = op.half_instruction_length as u64;
        }
        let dev = trace::gpu_build_cpu32_trace(&input, n, nrows).ok()?;
        tables.push(device_to_trace_table(dev)?);
    }
    GPU_CPU32_TABLE_BUILDS.fetch_add(1, Ordering::Relaxed);
    Some(tables)
}
