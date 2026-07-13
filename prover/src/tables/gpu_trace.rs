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

use stark::trace::TraceTable;

use std::collections::HashMap;

use super::cpu::{self, CpuOperation};
use super::load::{self, LoadOperation};
use super::lt::{self, LtOperation};
use super::memw::MemwOperation;
use super::memw_aligned;
use super::memw_register::{self, RegRow};
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

// =============================================================================
// CPU table (the first table built on device — see GPU-TRACEGEN-DESIGN-V2 §P1)
// =============================================================================

/// Marshal one chunk of `CpuOperation`s into the packed layout the `trace_cpu`
/// kernel consumes (stride `CPU_OP_STRIDE` u64/op). The kernel does the same
/// bit-slicing as `cpu::generate_cpu_trace`, so this only copies fields — no
/// per-column encoding on the host.
fn pack_cpu_ops(chunk: &[CpuOperation]) -> Vec<u64> {
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
    if gpu_trace_disabled() {
        return None;
    }
    let chunks: Vec<&[CpuOperation]> = if cpu_ops.is_empty() {
        vec![&[][..]]
    } else {
        cpu_ops.chunks(max_rows).collect()
    };
    let mut tables = Vec::with_capacity(chunks.len());
    for chunk in chunks {
        tables.push(build_cpu_chunk(chunk)?);
    }
    Some(tables)
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
    if gpu_trace_disabled() {
        return None;
    }
    let chunks: Vec<&[RegRow]> = if rows.is_empty() {
        vec![&[][..]]
    } else {
        rows.chunks(max_rows).collect()
    };
    let mut tables = Vec::with_capacity(chunks.len());
    for chunk in chunks {
        tables.push(build_memw_register_chunk(chunk)?);
    }
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
    if gpu_trace_disabled() {
        return None;
    }
    let chunks: Vec<&[MemwOperation]> = if ops.is_empty() {
        vec![&[][..]]
    } else {
        ops.chunks(max_rows).collect()
    };
    let mut tables = Vec::with_capacity(chunks.len());
    for chunk in chunks {
        tables.push(build_memw_aligned_chunk(chunk)?);
    }
    Some(tables)
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
    if gpu_trace_disabled() {
        return None;
    }
    let chunks: Vec<&[LoadOperation]> = if ops.is_empty() {
        vec![&[][..]]
    } else {
        ops.chunks(max_rows).collect()
    };
    let mut tables = Vec::with_capacity(chunks.len());
    for chunk in chunks {
        tables.push(build_load_chunk(chunk)?);
    }
    Some(tables)
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
    if gpu_trace_disabled() {
        return None;
    }
    let chunks: Vec<&[StoreOperation]> = if ops.is_empty() {
        vec![&[][..]]
    } else {
        ops.chunks(max_rows).collect()
    };
    let mut tables = Vec::with_capacity(chunks.len());
    for chunk in chunks {
        tables.push(build_store_chunk(chunk)?);
    }
    Some(tables)
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
    if gpu_trace_disabled() {
        return None;
    }
    let chunks: Vec<&[ShiftOperation]> = if ops.is_empty() {
        vec![&[][..]]
    } else {
        ops.chunks(max_rows).collect()
    };
    let mut tables = Vec::with_capacity(chunks.len());
    for chunk in chunks {
        tables.push(build_shift_chunk(chunk)?);
    }
    Some(tables)
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
    if gpu_trace_disabled() {
        return None;
    }
    // Chunk the RAW ops exactly like `chunk_and_generate`; each chunk dedups
    // independently (matching `generate_lt_trace` per chunk).
    let chunks: Vec<&[LtOperation]> = if ops.is_empty() {
        vec![&[][..]]
    } else {
        ops.chunks(max_rows).collect()
    };
    let mut tables = Vec::with_capacity(chunks.len());
    for chunk in chunks {
        tables.push(build_lt_chunk(chunk)?);
    }
    Some(tables)
}
