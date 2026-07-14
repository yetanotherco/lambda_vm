//! GPU device-fill parity tests: byte-parity (and multiset, for the
//! order-independent dedup ALU tables) between each table's on-device fill and its
//! CPU `generate_*_trace`, so the GPU trace tables are bit-identical to the CPU
//! path. Each test skips cleanly with no CUDA backend. Grouped here (rather than
//! inline in the table modules) so production code carries no test code; the whole
//! module is `cuda`-gated at its `mod.rs` registration.

use std::collections::HashMap;

use crate::tables::cpu::CpuOperation;
use crate::tables::memw::MemwOperation;
use crate::tables::{
    branch, bytewise, cpu, cpu32, dvrm, eq, load, lt, memw, memw_aligned, memw_register, mul,
    shift, store,
};

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

/// EQ device fill: like LT, dedup rides the permutation-invariant ALU bus, so
/// the row ORDER is non-deterministic (HashMap iteration). Validate as a
/// MULTISET — the real rows (μ>0), incl. summed multiplicities, must match the
/// CPU `generate_eq_trace`. Covers a==b and a!=b, invert, high words, and
/// duplicates (μ>1). Skips cleanly with no GPU.
#[test]
fn gpu_eq_fill_matches_cpu_multiset() {
    if math_cuda::device::backend().is_err() {
        eprintln!("skipping gpu_eq_fill_matches_cpu_multiset: no CUDA backend");
        return;
    }
    // Raw ops with equal/unequal operands, high words, and deliberate duplicates.
    let mut raw = Vec::new();
    for i in 0..800u64 {
        let a = i.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (i << 40);
        // Every 5th op has a == b to exercise the eq=1 path.
        let b = if i % 5 == 0 {
            a
        } else {
            i.wrapping_mul(0x1234_5678_9ABC_DEF1).rotate_left(11)
        };
        let invert = i % 3 == 0;
        let op = eq::EqOperation::new(a, b, invert);
        raw.push(op.clone());
        if i % 4 == 0 {
            raw.push(op); // duplicate → multiplicity 2
        }
    }

    let ncols = math_cuda::trace_cpu::EQ_NCOLS;
    // Extract the real (μ>0) rows of a row-major u64 buffer as a sorted multiset.
    let real_rows = |flat: &[u64]| -> Vec<Vec<u64>> {
        let mut rows: Vec<Vec<u64>> = flat
            .chunks(ncols)
            .filter(|row| row[eq::cols::MU] > 0)
            .map(|row| row.to_vec())
            .collect();
        rows.sort();
        rows
    };

    let cpu_table = eq::generate_eq_trace(&raw);
    let (cpu_fe, w) = cpu_table.main_data_row_major();
    assert_eq!(w, ncols);
    let cpu_u64: Vec<u64> = cpu_fe
        .iter()
        .map(|e| unsafe { *(e.value() as *const u64) })
        .collect();

    // Dedup on the host exactly like `gpu_build_eq_tables`, then device-fill.
    let mut map: std::collections::HashMap<eq::EqOperation, u64> = HashMap::new();
    for op in &raw {
        *map.entry(op.clone()).or_insert(0) += 1;
    }
    let unique: Vec<(eq::EqOperation, u64)> = map.into_iter().collect();
    let n = unique.len();
    let num_rows = n.next_power_of_two().max(4);
    let mut packed = Vec::with_capacity(n * math_cuda::trace_cpu::EQ_STRIDE);
    for (op, mult) in &unique {
        packed.extend_from_slice(&crate::tables::gpu_trace::pack_eq_op(op, *mult));
    }
    let gpu_u64 = math_cuda::trace_cpu::gpu_build_eq_trace_host(&packed, n, num_rows)
        .expect("device EQ build must run on a box with a CUDA backend");

    assert_eq!(
        real_rows(&gpu_u64),
        real_rows(&cpu_u64),
        "device EQ rows must match the CPU fill as a multiset"
    );
}

/// BYTEWISE device fill: like LT/EQ, dedup rides the permutation-invariant ALU
/// bus, so the row ORDER is non-deterministic. Validate as a MULTISET — the real
/// rows (μ>0), incl. summed multiplicities, must match `generate_bytewise_trace`.
/// Covers AND/OR/XOR, full 64-bit operands, and duplicates (μ>1). Skips cleanly
/// with no GPU.
#[test]
fn gpu_bytewise_fill_matches_cpu_multiset() {
    use crate::tables::types::alu_op;
    if math_cuda::device::backend().is_err() {
        eprintln!("skipping gpu_bytewise_fill_matches_cpu_multiset: no CUDA backend");
        return;
    }
    // Raw ops cycling AND/OR/XOR, full-word operands, with deliberate duplicates.
    let mut raw = Vec::new();
    for i in 0..900u64 {
        let a = i.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (i << 33);
        let b = i.wrapping_mul(0x1234_5678_9ABC_DEF1).rotate_left(19);
        let op = [alu_op::AND, alu_op::OR, alu_op::XOR][(i % 3) as usize];
        let bw = bytewise::BytewiseOperation::new(a, b, op);
        raw.push(bw.clone());
        if i % 4 == 0 {
            raw.push(bw); // duplicate → multiplicity 2
        }
    }

    let ncols = math_cuda::trace_cpu::BYTEWISE_NCOLS;
    // Extract the real (μ>0) rows of a row-major u64 buffer as a sorted multiset.
    let real_rows = |flat: &[u64]| -> Vec<Vec<u64>> {
        let mut rows: Vec<Vec<u64>> = flat
            .chunks(ncols)
            .filter(|row| row[bytewise::cols::MU] > 0)
            .map(|row| row.to_vec())
            .collect();
        rows.sort();
        rows
    };

    let cpu_table = bytewise::generate_bytewise_trace(&raw);
    let (cpu_fe, w) = cpu_table.main_data_row_major();
    assert_eq!(w, ncols);
    let cpu_u64: Vec<u64> = cpu_fe
        .iter()
        .map(|e| unsafe { *(e.value() as *const u64) })
        .collect();

    // Dedup on the host exactly like `gpu_build_bytewise_tables`, then device-fill.
    let mut map: std::collections::HashMap<bytewise::BytewiseOperation, u64> = HashMap::new();
    for op in &raw {
        *map.entry(op.clone()).or_insert(0) += 1;
    }
    let unique: Vec<(bytewise::BytewiseOperation, u64)> = map.into_iter().collect();
    let n = unique.len();
    let num_rows = n.next_power_of_two().max(4);
    let mut packed = Vec::with_capacity(n * math_cuda::trace_cpu::BYTEWISE_STRIDE);
    for (op, mult) in &unique {
        packed.extend_from_slice(&crate::tables::gpu_trace::pack_bytewise_op(op, *mult));
    }
    let gpu_u64 = math_cuda::trace_cpu::gpu_build_bytewise_trace_host(&packed, n, num_rows)
        .expect("device BYTEWISE build must run on a box with a CUDA backend");

    assert_eq!(
        real_rows(&gpu_u64),
        real_rows(&cpu_u64),
        "device BYTEWISE rows must match the CPU fill as a multiset"
    );
}

/// MUL device fill: like the other ALU tables, dedup rides the
/// permutation-invariant ALU bus, so the row ORDER is non-deterministic.
/// Validate as a MULTISET — the real rows (mu_lo>0 or mu_hi>0), incl. the split
/// multiplicities, must match `generate_mul_trace`. Covers all four
/// signed/unsigned combos, negative operands, the 128-bit product + raw_product
/// convolution, and lo/hi (wants_hi) requests with duplicates. Skips cleanly
/// with no GPU.
#[test]
fn gpu_mul_fill_matches_cpu_multiset() {
    if math_cuda::device::backend().is_err() {
        eprintln!("skipping gpu_mul_fill_matches_cpu_multiset: no CUDA backend");
        return;
    }
    // Raw (op, wants_hi) pairs: varied operands, all sign combos, both lo and hi
    // requests, with deliberate duplicates so mu_lo/mu_hi accumulate.
    let mut raw = Vec::new();
    for i in 0..800u64 {
        let lhs = i.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (i << 40);
        let rhs = i.wrapping_mul(0x1234_5678_9ABC_DEF1).rotate_left(23);
        let lhs_signed = i % 2 == 0;
        let rhs_signed = i % 3 == 0;
        let op = mul::MulOperation::new(lhs, lhs_signed, rhs, rhs_signed);
        raw.push((op.clone(), i % 2 == 0));
        if i % 3 == 0 {
            raw.push((op.clone(), true)); // extra hi request
        }
        if i % 5 == 0 {
            raw.push((op, false)); // extra lo request (duplicate)
        }
    }
    // Explicit sign edge cases: i64::MIN, -1, treated signed and unsigned.
    for &(a, b) in &[
        (0x8000_0000_0000_0000u64, 0xFFFF_FFFF_FFFF_FFFFu64),
        (0xFFFF_FFFF_FFFF_FFFFu64, 0x8000_0000_0000_0000u64),
    ] {
        raw.push((mul::MulOperation::new(a, true, b, true), false));
        raw.push((mul::MulOperation::new(a, false, b, false), true));
    }

    let ncols = math_cuda::trace_cpu::MUL_NCOLS;
    // Real rows: mu_lo>0 or mu_hi>0.
    let real_rows = |flat: &[u64]| -> Vec<Vec<u64>> {
        let mut rows: Vec<Vec<u64>> = flat
            .chunks(ncols)
            .filter(|row| row[mul::cols::MU_LO] > 0 || row[mul::cols::MU_HI] > 0)
            .map(|row| row.to_vec())
            .collect();
        rows.sort();
        rows
    };

    let cpu_table = mul::generate_mul_trace(&raw);
    let (cpu_fe, w) = cpu_table.main_data_row_major();
    assert_eq!(w, ncols);
    let cpu_u64: Vec<u64> = cpu_fe
        .iter()
        .map(|e| unsafe { *(e.value() as *const u64) })
        .collect();

    // Dedup on the host exactly like `gpu_build_mul_tables`, then device-fill.
    let mut map: std::collections::HashMap<mul::MulOperation, (u64, u64)> = HashMap::new();
    for (op, wants_hi) in &raw {
        let e = map.entry(op.clone()).or_insert((0, 0));
        if *wants_hi {
            e.1 += 1;
        } else {
            e.0 += 1;
        }
    }
    let unique: Vec<(mul::MulOperation, (u64, u64))> = map.into_iter().collect();
    let n = unique.len();
    let num_rows = n.next_power_of_two().max(4);
    let mut packed = Vec::with_capacity(n * math_cuda::trace_cpu::MUL_STRIDE);
    for (op, (mu_lo, mu_hi)) in &unique {
        packed.extend_from_slice(&crate::tables::gpu_trace::pack_mul_op(op, *mu_lo, *mu_hi));
    }
    let gpu_u64 = math_cuda::trace_cpu::gpu_build_mul_trace_host(&packed, n, num_rows)
        .expect("device MUL build must run on a box with a CUDA backend");

    assert_eq!(
        real_rows(&gpu_u64),
        real_rows(&cpu_u64),
        "device MUL rows must match the CPU fill as a multiset"
    );
}

/// DVRM device fill: like the other ALU tables, dedup rides the
/// permutation-invariant ALU bus, so the row ORDER is non-deterministic.
/// Validate as a MULTISET — the real rows (mu_q>0 or mu_r>0), incl. the split
/// multiplicities, must match `generate_dvrm_trace`. Covers signed/unsigned,
/// division-by-zero, the MIN/-1 signed overflow, negative operands, and
/// quotient/remainder (wants_remainder) requests with duplicates. Skips cleanly
/// with no GPU.
#[test]
fn gpu_dvrm_fill_matches_cpu_multiset() {
    if math_cuda::device::backend().is_err() {
        eprintln!("skipping gpu_dvrm_fill_matches_cpu_multiset: no CUDA backend");
        return;
    }
    // Raw (op, wants_remainder) pairs: varied operands (incl. periodic d==0),
    // both signednesses, q and r requests, with deliberate duplicates.
    let mut raw = Vec::new();
    for i in 0..800u64 {
        let n = i.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (i << 40);
        let d = if i % 11 == 0 {
            0
        } else {
            i.wrapping_mul(0x1234_5678_9ABC_DEF1).rotate_left(29)
        };
        let signed = i % 2 == 0;
        let op = dvrm::DvrmOperation::new(n, d, signed);
        raw.push((op.clone(), i % 2 == 0));
        if i % 3 == 0 {
            raw.push((op.clone(), true)); // extra remainder request
        }
        if i % 5 == 0 {
            raw.push((op, false)); // extra quotient request (duplicate)
        }
    }
    // Explicit edge cases: signed overflow (MIN/-1), div-by-zero, MIN numerator,
    // -1 denominator.
    let min = 0x8000_0000_0000_0000u64;
    let neg1 = 0xFFFF_FFFF_FFFF_FFFFu64;
    for &(n, d, s) in &[
        (min, neg1, true),     // signed overflow
        (min, neg1, false),    // unsigned: /(2^64-1), no overflow
        (123u64, 0u64, true),  // div-by-zero, signed
        (123u64, 0u64, false), // div-by-zero, unsigned
        (min, 7u64, true),     // negative numerator
        (100u64, neg1, true),  // negative denominator (n != MIN)
    ] {
        raw.push((dvrm::DvrmOperation::new(n, d, s), false));
        raw.push((dvrm::DvrmOperation::new(n, d, s), true));
    }

    let ncols = math_cuda::trace_cpu::DVRM_NCOLS;
    // Real rows: mu_q>0 or mu_r>0.
    let real_rows = |flat: &[u64]| -> Vec<Vec<u64>> {
        let mut rows: Vec<Vec<u64>> = flat
            .chunks(ncols)
            .filter(|row| row[dvrm::cols::MU_Q] > 0 || row[dvrm::cols::MU_R] > 0)
            .map(|row| row.to_vec())
            .collect();
        rows.sort();
        rows
    };

    let cpu_table = dvrm::generate_dvrm_trace(&raw);
    let (cpu_fe, w) = cpu_table.main_data_row_major();
    assert_eq!(w, ncols);
    let cpu_u64: Vec<u64> = cpu_fe
        .iter()
        .map(|e| unsafe { *(e.value() as *const u64) })
        .collect();

    // Dedup on the host exactly like `gpu_build_dvrm_tables`, then device-fill.
    let mut map: std::collections::HashMap<dvrm::DvrmOperation, (u64, u64)> = HashMap::new();
    for (op, wants_remainder) in &raw {
        let e = map.entry(op.clone()).or_insert((0, 0));
        if *wants_remainder {
            e.1 += 1;
        } else {
            e.0 += 1;
        }
    }
    let unique: Vec<(dvrm::DvrmOperation, (u64, u64))> = map.into_iter().collect();
    let n = unique.len();
    let num_rows = n.next_power_of_two().max(4);
    let mut packed = Vec::with_capacity(n * math_cuda::trace_cpu::DVRM_STRIDE);
    for (op, (mu_q, mu_r)) in &unique {
        packed.extend_from_slice(&crate::tables::gpu_trace::pack_dvrm_op(op, *mu_q, *mu_r));
    }
    let gpu_u64 = math_cuda::trace_cpu::gpu_build_dvrm_trace_host(&packed, n, num_rows)
        .expect("device DVRM build must run on a box with a CUDA backend");

    assert_eq!(
        real_rows(&gpu_u64),
        real_rows(&cpu_u64),
        "device DVRM rows must match the CPU fill as a multiset"
    );
}

/// BRANCH device fill: a permutation-invariant lookup table (dedup + summed
/// multiplicity), so the row ORDER is non-deterministic. Validate as a MULTISET
/// — the real rows (μ>0) must match `generate_branch_trace`. Covers JALR (base =
/// register) vs PC-relative, wrapping add, odd offsets (so LSB masking differs
/// from the unmasked low byte), high address bits, and duplicates (μ>1). Skips
/// cleanly with no GPU.
#[test]
fn gpu_branch_fill_matches_cpu_multiset() {
    if math_cuda::device::backend().is_err() {
        eprintln!("skipping gpu_branch_fill_matches_cpu_multiset: no CUDA backend");
        return;
    }
    let mut raw = Vec::new();
    for i in 0..800u64 {
        let pc = 0x1000u64.wrapping_add(i.wrapping_mul(4)) ^ (i << 34);
        // Odd offsets so `next_pc` (LSB masked) differs from the unmasked low
        // byte; high bits set to exercise the wrapping add.
        let offset = i.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        let register = i.wrapping_mul(0x1234_5678_9ABC_DEF1).rotate_left(7);
        let jalr = i % 2 == 0;
        let op = branch::BranchOperation::new(pc, offset, register, jalr);
        raw.push(op.clone());
        if i % 4 == 0 {
            raw.push(op); // duplicate → μ=2
        }
    }

    let ncols = math_cuda::trace_cpu::BRANCH_NCOLS;
    let real_rows = |flat: &[u64]| -> Vec<Vec<u64>> {
        let mut rows: Vec<Vec<u64>> = flat
            .chunks(ncols)
            .filter(|row| row[branch::cols::MU] > 0)
            .map(|row| row.to_vec())
            .collect();
        rows.sort();
        rows
    };

    let cpu_table = branch::generate_branch_trace(&raw);
    let (cpu_fe, w) = cpu_table.main_data_row_major();
    assert_eq!(w, ncols);
    let cpu_u64: Vec<u64> = cpu_fe
        .iter()
        .map(|e| unsafe { *(e.value() as *const u64) })
        .collect();

    // Dedup on the host exactly like `gpu_build_branch_tables`, then device-fill.
    let mut map: std::collections::HashMap<branch::BranchOperation, u64> = HashMap::new();
    for op in &raw {
        *map.entry(op.clone()).or_insert(0) += 1;
    }
    let unique: Vec<(branch::BranchOperation, u64)> = map.into_iter().collect();
    let n = unique.len();
    let num_rows = n.next_power_of_two().max(4);
    let mut packed = Vec::with_capacity(n * math_cuda::trace_cpu::BRANCH_STRIDE);
    for (op, mult) in &unique {
        packed.extend_from_slice(&crate::tables::gpu_trace::pack_branch_op(op, *mult));
    }
    let gpu_u64 = math_cuda::trace_cpu::gpu_build_branch_trace_host(&packed, n, num_rows)
        .expect("device BRANCH build must run on a box with a CUDA backend");

    assert_eq!(
        real_rows(&gpu_u64),
        real_rows(&cpu_u64),
        "device BRANCH rows must match the CPU fill as a multiset"
    );
}

/// CPU32 device fill must be byte-identical to `cpu32::generate_cpu32_trace`.
/// Per-row (μ=1, no dedup), so full byte parity holds. Covers signed/unsigned
/// (alu_flags bit 5), negative rv1/rv2/res (sign-extension into arg1/arg2/rvd),
/// imm-vs-rv2 operands, flag/register combinations, high words, and padding
/// (n=300 < 512). Skips cleanly with no GPU.
#[test]
fn gpu_cpu32_fill_matches_cpu() {
    if math_cuda::device::backend().is_err() {
        eprintln!("skipping gpu_cpu32_fill_matches_cpu: no CUDA backend");
        return;
    }
    let mut ops = Vec::new();
    for i in 0..300u64 {
        let signed = i % 2 == 0;
        // alu_flags: bit 5 = signed; low bits carry a varied opcode.
        let alu_flags = ((i % 20) as u8) | if signed { 1 << 5 } else { 0 };
        // Alternate imm-driven vs rv2-driven arg2 (decode assumption: one nonzero).
        let use_imm = i % 3 == 0;
        let rv2 = if use_imm {
            0
        } else {
            i.wrapping_mul(0xDEAD_BEEF).rotate_left(9)
        };
        let imm = if use_imm {
            i.wrapping_mul(0x9E37_79B9) | (i << 40)
        } else {
            0
        };
        ops.push(cpu32::Cpu32Operation {
            timestamp: 4 * i + 8 + (i << 34),
            pc: 0x1000 + i * 4 + ((i % 3) << 33),
            rs1: (i % 32) as u8,
            read_register1: i % 3 != 0,
            rv1: i.wrapping_mul(0x1234_5678_9ABC) ^ (i << 31), // exercise bit 31
            rs2: ((i + 5) % 32) as u8,
            read_register2: !use_imm,
            rv2,
            imm,
            res: i.wrapping_mul(0xABCD_1234) ^ (i << 31),
            rd: ((i + 7) % 32) as u8,
            write_register: i % 4 != 0,
            alu: i % 5 != 0,
            alu_flags,
            add: i % 5 == 1,
            sub: i % 5 == 2,
            half_instruction_length: if i % 2 == 0 { 1 } else { 2 },
        });
    }
    let n = ops.len();
    let num_rows = n.next_power_of_two().max(4);

    let cpu_table = cpu32::generate_cpu32_trace(&ops);
    let (cpu_fe, w) = cpu_table.main_data_row_major();
    assert_eq!(w, math_cuda::trace_cpu::CPU32_NCOLS);
    let cpu_u64: Vec<u64> = cpu_fe
        .iter()
        .map(|e| unsafe { *(e.value() as *const u64) })
        .collect();

    let mut packed = Vec::with_capacity(n * math_cuda::trace_cpu::CPU32_STRIDE);
    for op in &ops {
        packed.extend_from_slice(&crate::tables::gpu_trace::pack_cpu32_op(op));
    }
    let gpu_u64 = math_cuda::trace_cpu::gpu_build_cpu32_trace_host(&packed, n, num_rows)
        .expect("device CPU32 build must run on a box with a CUDA backend");

    assert_eq!(gpu_u64.len(), num_rows * math_cuda::trace_cpu::CPU32_NCOLS);
    assert_eq!(
        gpu_u64, cpu_u64,
        "device CPU32 table must be byte-identical to the CPU fill"
    );
}

/// MEMW (general/unaligned) device fill must be byte-identical to
/// `memw::generate_memw_trace`. Per-row (no dedup). Covers memory and register
/// accesses, widths 1/2/4/8, read/write, base addresses that straddle the 2^32
/// boundary (so `carry[i]` fires), distinct per-byte old_timestamps (the
/// split-timestamp path), and full-u32 value/old limbs (exercises both halves of
/// the 2×u32/u64 packing). Skips cleanly with no GPU.
#[test]
fn gpu_memw_fill_matches_cpu() {
    if math_cuda::device::backend().is_err() {
        eprintln!("skipping gpu_memw_fill_matches_cpu: no CUDA backend");
        return;
    }
    let mut ops = Vec::new();
    for i in 0..400u64 {
        let width = [1u8, 2, 4, 8][(i % 4) as usize];
        let is_read = i % 2 == 0;
        let is_register = i % 7 == 0;
        // Low word near 2^32 so `base_lo + (i+1)` overflows for some rows.
        let base = (0x1_0000_0000u64 * (i % 3)) + (0xFFFF_FFF8u64 - (i % 16)) + i;
        let value = [
            (i as u32).wrapping_mul(2_654_435_761),
            (i as u32) ^ 0xABCD_1234,
            i as u32,
            0xDEAD_0000 | (i as u32 & 0xFFFF),
            7,
            0,
            (i as u32).wrapping_add(99),
            (i as u32) & 0xFF,
        ];
        let old = [
            (i as u32).wrapping_add(3),
            (i as u32).wrapping_mul(17),
            0,
            (i as u32) ^ 0x0BAD_F00D,
            (i as u32).wrapping_add(5),
            (i as u32).wrapping_mul(23),
            0,
            (i as u32).wrapping_add(7),
        ];
        let ts = 4 * i + 100;
        // Distinct per-byte old_timestamps (the unaligned split-timestamp path).
        let mut old_ts = [0u64; 8];
        for (j, o) in old_ts.iter_mut().enumerate() {
            *o = (4 * i + 3 + j as u64) ^ ((j as u64) << 33);
        }
        ops.push(
            memw::MemwOperation::new(is_register, base, value, ts, width, is_read)
                .with_old(old, old_ts),
        );
    }
    let n = ops.len();
    let num_rows = n.next_power_of_two().max(4);

    let cpu_table = memw::generate_memw_trace(&ops);
    let (cpu_fe, w) = cpu_table.main_data_row_major();
    assert_eq!(w, math_cuda::trace_cpu::MEMW_NCOLS);
    let cpu_u64: Vec<u64> = cpu_fe
        .iter()
        .map(|e| unsafe { *(e.value() as *const u64) })
        .collect();

    let mut packed = Vec::with_capacity(n * math_cuda::trace_cpu::MEMW_STRIDE);
    for op in &ops {
        packed.extend_from_slice(&crate::tables::gpu_trace::pack_memw_op(op));
    }
    let gpu_u64 = math_cuda::trace_cpu::gpu_build_memw_trace_host(&packed, n, num_rows)
        .expect("device MEMW build must run on a box with a CUDA backend");

    assert_eq!(gpu_u64.len(), num_rows * math_cuda::trace_cpu::MEMW_NCOLS);
    assert_eq!(
        gpu_u64, cpu_u64,
        "device MEMW table must be byte-identical to the CPU fill"
    );
}
