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

use super::bytewise::{self, BytewiseOperation};
use super::cpu::cols;
use super::dvrm::{self, DvrmOperation};
use super::eq::{self, EqOperation};
use super::lt::{self, LtOperation};
use super::mul::{self, MulOperation};
use super::shift::{self, ShiftOperation};
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
pub fn gpu_build_lt_trace_tables(
    lt_ops: &[LtOperation],
    max_rows: usize,
) -> Option<Vec<TraceTable<GoldilocksField, GoldilocksExtension>>> {
    use std::collections::HashMap;
    if lt_ops.is_empty() {
        return None;
    }
    let ncols = lt::cols::NUM_COLUMNS;
    let mut tables = Vec::with_capacity(lt_ops.len().div_ceil(max_rows));
    for chunk in lt_ops.chunks(max_rows) {
        // Per-chunk dedup, mirroring generate_lt_trace.
        let mut map: HashMap<LtOperation, u64> = HashMap::new();
        for op in chunk {
            *map.entry(op.clone()).or_insert(0) += 1;
        }
        let unique: Vec<(LtOperation, u64)> = map.into_iter().collect();
        let n = unique.len();
        let nrows = n.next_power_of_two().max(4);

        let mut lhs = vec![0u64; n];
        let mut rhs = vec![0u64; n];
        let mut flags = vec![0u64; n];
        let mut mult = vec![0u64; n];
        for (i, (op, m)) in unique.iter().enumerate() {
            lhs[i] = op.lhs;
            rhs[i] = op.rhs;
            flags[i] = (op.signed as u64) | ((op.invert as u64) << 1);
            mult[i] = *m;
        }

        let dev = trace::gpu_build_lt_trace(&lhs, &rhs, &flags, &mult, n, nrows).ok()?;
        let host = dev.to_host().ok()?;
        let columns: Vec<Vec<FE>> = (0..ncols)
            .map(|c| host[c * nrows..c * nrows + nrows].iter().map(|&v| FE::from(v)).collect())
            .collect();
        let mut tt = TraceTable::from_columns_main(columns, 1);
        tt.set_gpu_main_input(dev);
        tables.push(tt);
    }

    GPU_LT_TABLE_BUILDS.fetch_add(1, Ordering::Relaxed);
    Some(tables)
}

/// EQ tables on GPU (per-chunk host dedup → GPU fill). Mirrors generate_eq_trace.
pub fn gpu_build_eq_trace_tables(
    eq_ops: &[EqOperation],
    max_rows: usize,
) -> Option<Vec<TraceTable<GoldilocksField, GoldilocksExtension>>> {
    use std::collections::HashMap;
    if eq_ops.is_empty() {
        return None;
    }
    let ncols = eq::cols::NUM_COLUMNS;
    let mut tables = Vec::with_capacity(eq_ops.len().div_ceil(max_rows));
    for chunk in eq_ops.chunks(max_rows) {
        let mut map: HashMap<EqOperation, u64> = HashMap::new();
        for op in chunk {
            *map.entry(op.clone()).or_insert(0) += 1;
        }
        let unique: Vec<(EqOperation, u64)> = map.into_iter().collect();
        let n = unique.len();
        let nrows = n.next_power_of_two().max(4);
        let mut a = vec![0u64; n];
        let mut b = vec![0u64; n];
        let mut flags = vec![0u64; n];
        let mut mult = vec![0u64; n];
        for (i, (op, m)) in unique.iter().enumerate() {
            a[i] = op.a;
            b[i] = op.b;
            flags[i] = op.invert as u64;
            mult[i] = *m;
        }
        let dev = trace::gpu_build_eq_trace(&a, &b, &flags, &mult, n, nrows).ok()?;
        let host = dev.to_host().ok()?;
        let columns: Vec<Vec<FE>> = (0..ncols)
            .map(|c| host[c * nrows..c * nrows + nrows].iter().map(|&v| FE::from(v)).collect())
            .collect();
        let mut tt = TraceTable::from_columns_main(columns, 1);
        tt.set_gpu_main_input(dev);
        tables.push(tt);
    }
    GPU_EQ_TABLE_BUILDS.fetch_add(1, Ordering::Relaxed);
    Some(tables)
}

/// BYTEWISE tables on GPU (per-chunk host dedup → GPU fill). Mirrors
/// generate_bytewise_trace.
pub fn gpu_build_bytewise_trace_tables(
    bytewise_ops: &[BytewiseOperation],
    max_rows: usize,
) -> Option<Vec<TraceTable<GoldilocksField, GoldilocksExtension>>> {
    use std::collections::HashMap;
    if bytewise_ops.is_empty() {
        return None;
    }
    let ncols = bytewise::cols::NUM_COLUMNS;
    let mut tables = Vec::with_capacity(bytewise_ops.len().div_ceil(max_rows));
    for chunk in bytewise_ops.chunks(max_rows) {
        let mut map: HashMap<BytewiseOperation, u64> = HashMap::new();
        for op in chunk {
            *map.entry(op.clone()).or_insert(0) += 1;
        }
        let unique: Vec<(BytewiseOperation, u64)> = map.into_iter().collect();
        let n = unique.len();
        let nrows = n.next_power_of_two().max(4);
        let mut a = vec![0u64; n];
        let mut b = vec![0u64; n];
        let mut op_col = vec![0u64; n];
        let mut mult = vec![0u64; n];
        for (i, (op, m)) in unique.iter().enumerate() {
            a[i] = op.a;
            b[i] = op.b;
            op_col[i] = op.op as u64;
            mult[i] = *m;
        }
        let dev = trace::gpu_build_bytewise_trace(&a, &b, &op_col, &mult, n, nrows).ok()?;
        let host = dev.to_host().ok()?;
        let columns: Vec<Vec<FE>> = (0..ncols)
            .map(|c| host[c * nrows..c * nrows + nrows].iter().map(|&v| FE::from(v)).collect())
            .collect();
        let mut tt = TraceTable::from_columns_main(columns, 1);
        tt.set_gpu_main_input(dev);
        tables.push(tt);
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
    let ncols = shift::cols::NUM_COLUMNS;
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
        let host = dev.to_host().ok()?;
        let columns: Vec<Vec<FE>> = (0..ncols)
            .map(|c| host[c * nrows..c * nrows + nrows].iter().map(|&v| FE::from(v)).collect())
            .collect();
        let mut tt = TraceTable::from_columns_main(columns, 1);
        tt.set_gpu_main_input(dev);
        tables.push(tt);
    }
    GPU_SHIFT_TABLE_BUILDS.fetch_add(1, Ordering::Relaxed);
    Some(tables)
}

/// MUL tables on GPU (per-chunk host dedup with dual mu_lo/mu_hi counters →
/// GPU fill). Mirrors generate_mul_trace. Input is (op, wants_hi) pairs.
pub fn gpu_build_mul_trace_tables(
    mul_ops: &[(MulOperation, bool)],
    max_rows: usize,
) -> Option<Vec<TraceTable<GoldilocksField, GoldilocksExtension>>> {
    use std::collections::HashMap;
    if mul_ops.is_empty() {
        return None;
    }
    let ncols = mul::cols::NUM_COLUMNS;
    let mut tables = Vec::with_capacity(mul_ops.len().div_ceil(max_rows));
    for chunk in mul_ops.chunks(max_rows) {
        // Dedup by (lhs, lhs_signed, rhs, rhs_signed); wants_hi selects counter.
        let mut map: HashMap<(u64, bool, u64, bool), (u64, u64)> = HashMap::new();
        for (op, wants_hi) in chunk {
            let e = map
                .entry((op.lhs, op.lhs_signed, op.rhs, op.rhs_signed))
                .or_default();
            if *wants_hi {
                e.1 += 1;
            } else {
                e.0 += 1;
            }
        }
        let unique: Vec<((u64, bool, u64, bool), (u64, u64))> = map.into_iter().collect();
        let n = unique.len();
        let nrows = n.next_power_of_two().max(4);
        let mut lhs = vec![0u64; n];
        let mut rhs = vec![0u64; n];
        let mut flags = vec![0u64; n];
        let mut mu_lo = vec![0u64; n];
        let mut mu_hi = vec![0u64; n];
        for (i, ((l, ls, rr, rs), (mlo, mhi))) in unique.iter().enumerate() {
            lhs[i] = *l;
            rhs[i] = *rr;
            flags[i] = (*ls as u64) | ((*rs as u64) << 1);
            mu_lo[i] = *mlo;
            mu_hi[i] = *mhi;
        }
        let dev = trace::gpu_build_mul_trace(&lhs, &rhs, &flags, &mu_lo, &mu_hi, n, nrows).ok()?;
        let host = dev.to_host().ok()?;
        let columns: Vec<Vec<FE>> = (0..ncols)
            .map(|c| host[c * nrows..c * nrows + nrows].iter().map(|&v| FE::from(v)).collect())
            .collect();
        let mut tt = TraceTable::from_columns_main(columns, 1);
        tt.set_gpu_main_input(dev);
        tables.push(tt);
    }
    GPU_MUL_TABLE_BUILDS.fetch_add(1, Ordering::Relaxed);
    Some(tables)
}

/// DVRM tables on GPU (per-chunk host dedup with dual mu_q/mu_r counters → GPU
/// fill). Mirrors generate_dvrm_trace. Input is (op, wants_remainder) pairs.
pub fn gpu_build_dvrm_trace_tables(
    dvrm_ops: &[(DvrmOperation, bool)],
    max_rows: usize,
) -> Option<Vec<TraceTable<GoldilocksField, GoldilocksExtension>>> {
    use std::collections::HashMap;
    if dvrm_ops.is_empty() {
        return None;
    }
    let ncols = dvrm::cols::NUM_COLUMNS;
    let mut tables = Vec::with_capacity(dvrm_ops.len().div_ceil(max_rows));
    for chunk in dvrm_ops.chunks(max_rows) {
        // Dedup by (n, d, signed); wants_remainder selects mu_r else mu_q.
        let mut map: HashMap<(u64, u64, bool), (u64, u64)> = HashMap::new();
        for (op, wants_remainder) in chunk {
            let e = map.entry((op.n, op.d, op.signed)).or_default();
            if *wants_remainder {
                e.1 += 1;
            } else {
                e.0 += 1;
            }
        }
        let unique: Vec<((u64, u64, bool), (u64, u64))> = map.into_iter().collect();
        let n = unique.len();
        let nrows = n.next_power_of_two().max(4);
        let mut n_num = vec![0u64; n];
        let mut d_den = vec![0u64; n];
        let mut flags = vec![0u64; n];
        let mut mu_q = vec![0u64; n];
        let mut mu_r = vec![0u64; n];
        for (i, ((nn, dd, signed), (mq, mr))) in unique.iter().enumerate() {
            n_num[i] = *nn;
            d_den[i] = *dd;
            flags[i] = *signed as u64;
            mu_q[i] = *mq;
            mu_r[i] = *mr;
        }
        let dev = trace::gpu_build_dvrm_trace(&n_num, &d_den, &flags, &mu_q, &mu_r, n, nrows).ok()?;
        let host = dev.to_host().ok()?;
        let columns: Vec<Vec<FE>> = (0..ncols)
            .map(|c| host[c * nrows..c * nrows + nrows].iter().map(|&v| FE::from(v)).collect())
            .collect();
        let mut tt = TraceTable::from_columns_main(columns, 1);
        tt.set_gpu_main_input(dev);
        tables.push(tt);
    }
    GPU_DVRM_TABLE_BUILDS.fetch_add(1, Ordering::Relaxed);
    Some(tables)
}
