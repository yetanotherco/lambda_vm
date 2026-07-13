//! GPU device-fill parity tests: byte-parity (and multiset, for the
//! order-independent dedup ALU tables) between each table's on-device fill and its
//! CPU `generate_*_trace`, so the GPU trace tables are bit-identical to the CPU
//! path. Each test skips cleanly with no CUDA backend. Grouped here (rather than
//! inline in the table modules) so production code carries no test code; the whole
//! module is `cuda`-gated at its `mod.rs` registration.

use std::collections::HashMap;

use crate::tables::cpu::CpuOperation;
use crate::tables::memw::MemwOperation;
use crate::tables::{cpu, load, lt, memw_aligned, memw_register, shift, store};

/// CPU-table device fill must be byte-identical to `cpu::generate_cpu_trace`.
/// The CPU kernel is the most intricate of the seven (word-delegate column
/// masking, `PC_DOUBLE_READ`/`PREV_PC_TIMESTAMP_BORROW`, and the `+4` padding
/// cadence with PC=1), so this guards it the same way its six siblings are
/// guarded. The fill is a pure function of the packed op fields, so the synthetic
/// ops need only be diverse (not a valid execution): word-delegate rows, x255 PC
/// reads (`pc_double_read`), `ts_lo < 3` rows (`prev_pc_timestamp_borrow`), x0
/// registers, and high-word values (DWordWL/DWordHL splits). `n = 300 < 512`
/// forces padding rows, exercising the `+4` cadence off `last_ts`. Skips cleanly
/// with no GPU.
#[test]
fn gpu_cpu_fill_matches_cpu() {
    use crate::tables::types::{DecodeEntry, ShrunkDecode};
    if math_cuda::device::backend().is_err() {
        eprintln!("skipping gpu_cpu_fill_matches_cpu: no CUDA backend");
        return;
    }
    let mut ops = Vec::new();
    for i in 0..300u64 {
        let word = i % 5 == 0;
        // rs1 cycles through x255 (PC register), x0, and normal registers so the
        // pc_double_read and register-zero-suppression paths are both hit.
        let rs1 = match i % 4 {
            0 => 255u8,
            1 => 0,
            _ => (i % 31 + 1) as u8,
        };
        let fields = ShrunkDecode {
            read_register1: i % 3 != 0,
            read_register2: i % 2 == 0,
            write_register: i % 3 == 0,
            word_instr: word,
            alu: i % 6 == 0,
            add: i % 6 == 1,
            sub: i % 6 == 2,
            memory: i % 6 == 3,
            branch: i % 6 == 4,
            ecall: i % 6 == 5,
            rs1,
            rs2: (i % 32) as u8,
            rd: ((i + 7) % 32) as u8,
            half_instruction_length: if i % 2 == 0 { 1 } else { 2 },
            alu_flags: (i & 0xFF) as u8,
            mem_flags: ((i >> 1) & 0xFF) as u8,
        };
        // A few timestamps with ts_lo < 3 (0,1,2) trip prev_pc_timestamp_borrow;
        // the rest are large and continue a +4 cadence.
        let timestamp = match i {
            0 => 0,
            1 => 1,
            2 => 2,
            _ => 4 * i + 100,
        };
        // PC/imm/next_pc carry high words (>32 bits) to exercise the DWordWL split
        // into PC_1/NEXT_PC_1/IMM_1.
        let decode = DecodeEntry {
            pc: 0x1000 + i * 4 + ((i % 3) << 33),
            imm: i.wrapping_mul(0x9E37_79B9) | ((i % 4) << 40),
            fields,
        };
        ops.push(CpuOperation {
            decode,
            timestamp,
            next_pc: 0x2000 + i * 4 + ((i % 2) << 34),
            rvd: i.wrapping_mul(0x1234_5678_9ABC),
            rv1: i.wrapping_mul(0xDEAD_BEEF).rotate_left(13),
            rv2: (i ^ 0xFFFF_0000_1111) << 3,
            arg2: i.wrapping_add(0x7777_8888_9999),
            res: i.wrapping_mul(0xABCD_1234_5678) ^ (i << 48),
            branch_cond: i % 3 == 1,
            ..Default::default()
        });
    }
    let n = ops.len();
    let num_rows = n.next_power_of_two().max(4);
    let last_ts = ops.last().map(|op| op.timestamp).unwrap_or(0);

    let cpu_table = cpu::generate_cpu_trace(&ops);
    let (cpu_fe, w) = cpu_table.main_data_row_major();
    assert_eq!(w, math_cuda::trace_cpu::CPU_NCOLS);
    let cpu_u64: Vec<u64> = cpu_fe
        .iter()
        .map(|e| unsafe { *(e.value() as *const u64) })
        .collect();

    let packed = crate::tables::gpu_trace::pack_cpu_ops(&ops);
    let gpu_u64 = math_cuda::trace_cpu::gpu_build_cpu_trace_host(&packed, n, num_rows, last_ts)
        .expect("device CPU build must run on a box with a CUDA backend");

    assert_eq!(gpu_u64.len(), num_rows * math_cuda::trace_cpu::CPU_NCOLS);
    assert_eq!(
        gpu_u64, cpu_u64,
        "device CPU table must be byte-identical to the CPU fill"
    );
}

/// MEMW_R (register fast-path) device fill must be byte-identical to the CPU
/// `generate_memw_register_trace_from_rows`. The rows are the walked register
/// rows; here we build synthetic `RegRow`s (read + write, varied values/old/ts)
/// and fill both ways.
#[test]
fn gpu_memw_register_fill_matches_cpu() {
    if math_cuda::device::backend().is_err() {
        eprintln!("skipping gpu_memw_register_fill_matches_cpu: no CUDA backend");
        return;
    }
    let mut rows = Vec::new();
    for i in 0..500u64 {
        let reg_addr = 2 * (i % 40); // even word-address = 2*reg_index
        let ts = 4 * i + 100;
        let old_ts = 4 * i + 50;
        let val = i.wrapping_mul(2_654_435_761);
        let old = i.wrapping_mul(40_503) ^ 0xABCD_1234;
        let is_read = i % 3 == 0;
        rows.push(memw_register::RegRow::new(
            reg_addr,
            ts,
            val as u32,
            (val >> 32) as u32,
            old as u32,
            (old >> 32) as u32,
            old_ts,
            is_read,
        ));
    }
    let num_rows = rows.len().next_power_of_two().max(4);

    let cpu_table = memw_register::generate_memw_register_trace_from_rows(&rows);
    let (cpu_fe, w) = cpu_table.main_data_row_major();
    assert_eq!(w, math_cuda::trace_cpu::MEMW_REGISTER_NCOLS);
    let cpu_u64: Vec<u64> = cpu_fe
        .iter()
        .map(|e| unsafe { *(e.value() as *const u64) })
        .collect();

    let mut reg_addr = Vec::new();
    let mut ts = Vec::new();
    let mut value = Vec::new();
    let mut is_read = Vec::new();
    let mut old_value = Vec::new();
    let mut old_tsv = Vec::new();
    for r in &rows {
        let (ra, t, v, ir, ov, ot) = r.fill_soa();
        reg_addr.push(ra);
        ts.push(t);
        value.push(v);
        is_read.push(ir);
        old_value.push(ov);
        old_tsv.push(ot);
    }
    let gpu_u64 = math_cuda::trace_cpu::gpu_fill_memw_register_host(
        &reg_addr, &ts, &value, &is_read, &old_value, &old_tsv, num_rows,
    )
    .expect("device MEMW_R fill must run on a box with a CUDA backend");

    assert_eq!(
        gpu_u64.len(),
        num_rows * math_cuda::trace_cpu::MEMW_REGISTER_NCOLS
    );
    assert_eq!(
        gpu_u64, cpu_u64,
        "device MEMW_R fill must be byte-identical to the CPU fill"
    );
}

/// MEMW_A (aligned memory — the biggest remaining uploader) device fill must be
/// byte-identical to the CPU `generate_memw_aligned_trace`. Exercises memory
/// (widths 2/4/8), register fallback (`is_register`), high address bits, and
/// the 2×u32-per-u64 value/old packing. Skips cleanly with no GPU.
#[test]
fn gpu_memw_aligned_fill_matches_cpu() {
    if math_cuda::device::backend().is_err() {
        eprintln!("skipping gpu_memw_aligned_fill_matches_cpu: no CUDA backend");
        return;
    }
    let mut ops = Vec::new();
    for i in 0..500u64 {
        let width = [2u8, 4, 8][(i % 3) as usize];
        let is_read = i % 2 == 0;
        let is_register = i % 7 == 0;
        // Vary the high address word too (whh split has a 32-bit high word).
        let base = (0x1_0000_0000u64 * (i % 4)) + 0x8000 + i * 8;
        let value = [
            (i as u32).wrapping_mul(2654435761),
            (i as u32) ^ 0xABCD_1234,
            i as u32,
            0,
            7,
            0,
            0,
            (i as u32) & 0xFF,
        ];
        let old = [
            (i as u32).wrapping_add(99),
            (i as u32) ^ 0x0BAD_F00D,
            0,
            3,
            0,
            0,
            0,
            0,
        ];
        let ts = 4 * i + 100;
        let old_ts = [4 * i + 50; 8];
        ops.push(
            MemwOperation::new(is_register, base, value, ts, width, is_read).with_old(old, old_ts),
        );
    }
    let n = ops.len();
    let num_rows = n.next_power_of_two().max(4);

    let cpu_table = memw_aligned::generate_memw_aligned_trace(&ops);
    let (cpu_fe, width_cols) = cpu_table.main_data_row_major();
    assert_eq!(width_cols, math_cuda::trace_cpu::MEMW_ALIGNED_NCOLS);
    let cpu_u64: Vec<u64> = cpu_fe
        .iter()
        .map(|e| unsafe { *(e.value() as *const u64) })
        .collect();

    let mut packed = Vec::with_capacity(n * math_cuda::trace_cpu::MEMW_ALIGNED_STRIDE);
    for op in &ops {
        packed.extend_from_slice(&crate::tables::gpu_trace::pack_memw_aligned_op(op));
    }
    let gpu_u64 = math_cuda::trace_cpu::gpu_build_memw_aligned_trace_host(&packed, n, num_rows)
        .expect("device MEMW_A build must run on a box with a CUDA backend");

    assert_eq!(
        gpu_u64.len(),
        num_rows * math_cuda::trace_cpu::MEMW_ALIGNED_NCOLS
    );
    assert_eq!(
        gpu_u64, cpu_u64,
        "device MEMW_A table must be byte-identical to the CPU fill"
    );
}

/// LOAD device fill byte-parity (widths 1/2/4/8, signed/unsigned, sign-bit,
/// high address bits, res-byte packing). Skips cleanly with no GPU.
#[test]
fn gpu_load_fill_matches_cpu() {
    if math_cuda::device::backend().is_err() {
        eprintln!("skipping gpu_load_fill_matches_cpu: no CUDA backend");
        return;
    }
    let mut ops = Vec::new();
    for i in 0..400u64 {
        let width = [1u8, 2, 4, 8][(i % 4) as usize];
        let signed = i % 2 == 0;
        let base = (0x1_0000_0000u64 * (i % 3)) + 0x400 + i * 8;
        let ts = 4 * i + 7;
        let mut res = [0u64; 8];
        for (j, r) in res.iter_mut().enumerate() {
            *r = (i.wrapping_mul(31) + j as u64) & 0xFF;
        }
        ops.push(load::LoadOperation::new(base, ts, width, signed, res));
    }
    let n = ops.len();
    let num_rows = n.next_power_of_two().max(4);
    let cpu = load::generate_load_trace(&ops);
    let (fe, w) = cpu.main_data_row_major();
    assert_eq!(w, math_cuda::trace_cpu::LOAD_NCOLS);
    let cpu_u64: Vec<u64> = fe
        .iter()
        .map(|e| unsafe { *(e.value() as *const u64) })
        .collect();
    let mut packed = Vec::with_capacity(n * math_cuda::trace_cpu::LOAD_STRIDE);
    for op in &ops {
        packed.extend_from_slice(&crate::tables::gpu_trace::pack_load_op(op));
    }
    let gpu = math_cuda::trace_cpu::gpu_build_load_trace_host(&packed, n, num_rows)
        .expect("device LOAD build must run on a box with a CUDA backend");
    assert_eq!(
        gpu, cpu_u64,
        "device LOAD table must be byte-identical to the CPU fill"
    );
}

/// STORE device fill byte-parity (widths 1/2/4/8 via `bytes`, full-value
/// DWordBL split, high address bits). Skips cleanly with no GPU.
#[test]
fn gpu_store_fill_matches_cpu() {
    if math_cuda::device::backend().is_err() {
        eprintln!("skipping gpu_store_fill_matches_cpu: no CUDA backend");
        return;
    }
    let mut ops = Vec::new();
    for i in 0..400u64 {
        let bytes = [1u8, 2, 4, 8][(i % 4) as usize];
        let base = (0x1_0000_0000u64 * (i % 3)) + 0x800 + i * 8;
        let ts = 4 * i + 9;
        let value = i.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        ops.push(store::StoreOperation::new(base, ts, value, bytes));
    }
    let n = ops.len();
    let num_rows = n.next_power_of_two().max(4);
    let cpu = store::generate_store_trace(&ops);
    let (fe, w) = cpu.main_data_row_major();
    assert_eq!(w, math_cuda::trace_cpu::STORE_NCOLS);
    let cpu_u64: Vec<u64> = fe
        .iter()
        .map(|e| unsafe { *(e.value() as *const u64) })
        .collect();
    let mut packed = Vec::with_capacity(n * math_cuda::trace_cpu::STORE_STRIDE);
    for op in &ops {
        packed.extend_from_slice(&crate::tables::gpu_trace::pack_store_op(op));
    }
    let gpu = math_cuda::trace_cpu::gpu_build_store_trace_host(&packed, n, num_rows)
        .expect("device STORE build must run on a box with a CUDA backend");
    assert_eq!(
        gpu, cpu_u64,
        "device STORE table must be byte-identical to the CPU fill"
    );
}

/// SHIFT device fill byte-parity: the kernel recomputes the full aux
/// (bit_shift/zbs/x/y/limb_shift/out) — covers left/right, signed/unsigned,
/// word/64-bit, shift amounts spanning 0 / >16 / >32 / >64, negative MSB inputs,
/// and the padding rows (ZBS=1). SHIFT is per-row (μ=1), so byte-parity holds.
#[test]
fn gpu_shift_fill_matches_cpu() {
    if math_cuda::device::backend().is_err() {
        eprintln!("skipping gpu_shift_fill_matches_cpu: no CUDA backend");
        return;
    }
    let mut ops = Vec::new();
    // Exhaustive over shift byte 0..256 × all flags × MSB-set / edge values
    // (so the signed arithmetic-right-shift extension path is fully covered).
    let values: [u64; 8] = [
        0,
        1,
        0x8000_0000_0000_0000,
        0xFFFF_FFFF_FFFF_FFFF,
        0x1234_5678_9ABC_DEF0,
        0xFEDC_BA98_7654_3210,
        0x0000_0000_FFFF_FFFF,
        0xFFFF_FFFF_0000_0000,
    ];
    for &value in &values {
        for shift_amount in 0u64..256 {
            for &direction in &[false, true] {
                for &signed in &[false, true] {
                    for &word_instr in &[false, true] {
                        ops.push(shift::ShiftOperation::new(
                            value,
                            shift_amount,
                            direction,
                            signed,
                            word_instr,
                        ));
                    }
                }
            }
        }
    }
    // Also a few large shift_amounts (high SHIFT_B1/H1/HIGH limbs) — real ops
    // carry the full rv2 on the ALU bus.
    for &sa in &[0x1_0000u64, 0xFFFF_FFFFu64, 0x1234_5678_9ABCu64, u64::MAX] {
        ops.push(shift::ShiftOperation::new(
            0xDEAD_BEEF_CAFE_1234,
            sa,
            false,
            true,
            false,
        ));
    }
    let n = ops.len();
    let num_rows = n.next_power_of_two().max(4);
    let cpu = shift::generate_shift_trace(&ops);
    let (fe, w) = cpu.main_data_row_major();
    assert_eq!(w, math_cuda::trace_cpu::SHIFT_NCOLS);
    let cpu_u64: Vec<u64> = fe
        .iter()
        .map(|e| unsafe { *(e.value() as *const u64) })
        .collect();
    let mut packed = Vec::with_capacity(n * math_cuda::trace_cpu::SHIFT_STRIDE);
    for op in &ops {
        packed.extend_from_slice(&crate::tables::gpu_trace::pack_shift_op(op));
    }
    let gpu = math_cuda::trace_cpu::gpu_build_shift_trace_host(&packed, n, num_rows)
        .expect("device SHIFT build must run on a box with a CUDA backend");
    assert_eq!(
        gpu, cpu_u64,
        "device SHIFT table must be byte-identical to the CPU fill"
    );
}

/// LT device fill: dedup rides the permutation-invariant ALU bus, so the row
/// ORDER is non-deterministic (HashMap iteration). Validate as a MULTISET —
/// the set of real rows (μ>0), incl. summed multiplicities, must match the CPU
/// `generate_lt_trace`. Covers signed/unsigned, invert, MSBs, and duplicates
/// (μ>1). Skips cleanly with no GPU.
#[test]
fn gpu_lt_fill_matches_cpu_multiset() {
    if math_cuda::device::backend().is_err() {
        eprintln!("skipping gpu_lt_fill_matches_cpu_multiset: no CUDA backend");
        return;
    }
    // Raw ops with deliberate duplicates (some pushed twice → μ=2).
    let mut raw = Vec::new();
    for i in 0..800u64 {
        let lhs = i.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (i << 50);
        let rhs = i.wrapping_mul(0x1234_5678_9ABC_DEF1).rotate_left(17);
        let signed = i % 2 == 0;
        let invert = i % 3 == 0;
        let op = lt::LtOperation::new_with_invert(lhs, rhs, signed, invert);
        raw.push(op.clone());
        if i % 4 == 0 {
            raw.push(op); // duplicate → multiplicity 2
        }
    }

    let ncols = math_cuda::trace_cpu::LT_NCOLS;
    // Extract the real (μ>0) rows of a row-major u64 buffer as a sorted multiset.
    let real_rows = |flat: &[u64]| -> Vec<Vec<u64>> {
        let mut rows: Vec<Vec<u64>> = flat
            .chunks(ncols)
            .filter(|row| row[lt::cols::MU] > 0)
            .map(|row| row.to_vec())
            .collect();
        rows.sort();
        rows
    };

    let cpu_table = lt::generate_lt_trace(&raw);
    let (cpu_fe, w) = cpu_table.main_data_row_major();
    assert_eq!(w, ncols);
    let cpu_u64: Vec<u64> = cpu_fe
        .iter()
        .map(|e| unsafe { *(e.value() as *const u64) })
        .collect();

    // Dedup on the host exactly like `gpu_build_lt_tables`, then device-fill.
    let mut map: std::collections::HashMap<lt::LtOperation, u64> = HashMap::new();
    for op in &raw {
        *map.entry(op.clone()).or_insert(0) += 1;
    }
    let unique: Vec<(lt::LtOperation, u64)> = map.into_iter().collect();
    let n = unique.len();
    let num_rows = n.next_power_of_two().max(4);
    let mut packed = Vec::with_capacity(n * math_cuda::trace_cpu::LT_STRIDE);
    for (op, mult) in &unique {
        packed.extend_from_slice(&crate::tables::gpu_trace::pack_lt_op(op, *mult));
    }
    let gpu_u64 = math_cuda::trace_cpu::gpu_build_lt_trace_host(&packed, n, num_rows)
        .expect("device LT build must run on a box with a CUDA backend");

    assert_eq!(
        real_rows(&gpu_u64),
        real_rows(&cpu_u64),
        "device LT rows must match the CPU fill as a multiset"
    );
}
