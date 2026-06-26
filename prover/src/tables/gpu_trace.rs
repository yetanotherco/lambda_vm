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

use super::cpu::cols;
use super::lt::{self, LtOperation};
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
