//! OP-VEC BITWISE histogram parity (P4a): each device `bitwise_hist_<src>` kernel must be
//! BIN-FOR-BIN identical to its CPU collector's decomposition. Every test builds a synthetic
//! op vector, runs the REAL CPU collector to produce the reference counter array (scattered by
//! `lookup_type_index`/`row_index`), runs the GPU kernel, and compares the full
//! `[NUM_ROWS * NUM_LOOKUP_TYPES]` array. Self-contained (no ELF); requires a GPU.
//!
//! `cargo test -p lambda-vm-prover --release --features cuda --lib gpu_bitwise_opvec -- --ignored --nocapture`

use crate::tables::bitwise::{self, BitwiseOperation};
use crate::tables::branch::BranchOperation;
use crate::tables::bytewise::BytewiseOperation;
use crate::tables::cpu32::Cpu32Operation;
use crate::tables::eq::EqOperation;
use crate::tables::dvrm::DvrmOperation;
use crate::tables::load::LoadOperation;
use crate::tables::lt::LtOperation;
use crate::tables::mul::MulOperation;
use crate::tables::shift::{collect_bitwise_from_shift, ShiftOperation};
use crate::tables::store::StoreOperation;
use crate::tables::trace_builder::{
    collect_bitwise_from_branch, collect_bitwise_from_dvrm, collect_bitwise_from_lt,
    collect_bitwise_from_mul, collect_cpu32_bitwise,
};

/// A deterministic well-mixed 64-bit value (seed `i`, stream `s`) exercising all four halves.
fn mix(i: usize, s: u64) -> u64 {
    let mut x = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ s.wrapping_mul(0xD1B5_4A32_D192_ED03);
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// Scatter a CPU op vector into the dense `[NUM_ROWS * NUM_LOOKUP_TYPES]` counter array,
/// matching `BitwiseHistogram::bump` exactly.
fn scatter_ref(ops: &[BitwiseOperation], num_rows: usize, num_types: usize) -> Vec<u64> {
    let mut cpu = vec![0u64; num_rows * num_types];
    for op in ops {
        let idx = bitwise::lookup_type_index(op.lookup_type) * num_rows
            + bitwise::row_index(op.x, op.y, op.z);
        cpu[idx] += 1;
    }
    cpu
}

fn have_gpu() -> bool {
    if math_cuda::device::backend().is_err() {
        eprintln!("skipping: no CUDA backend");
        false
    } else {
        true
    }
}

#[test]
#[ignore = "requires GPU; run --ignored --nocapture"]
fn gpu_bitwise_lt_matches_cpu() {
    if !have_gpu() {
        return;
    }
    let (nr, nt) = (bitwise::NUM_ROWS, bitwise::NUM_LOOKUP_TYPES);
    let n = 200_000usize;
    let ops: Vec<LtOperation> =
        (0..n).map(|i| LtOperation::new(mix(i, 1), mix(i, 2), i % 2 == 0)).collect();
    let reference = scatter_ref(&collect_bitwise_from_lt(&ops), nr, nt);

    let lhs: Vec<u64> = ops.iter().map(|o| o.lhs).collect();
    let rhs: Vec<u64> = ops.iter().map(|o| o.rhs).collect();
    let gpu = math_cuda::bitwise_hist::gpu_bitwise_hist_lt(&lhs, &rhs, nr, nt).expect("gpu lt hist");
    assert_eq!(gpu, reference, "GPU lt histogram != CPU collect_bitwise_from_lt");
    let bumps: u64 = gpu.iter().sum();
    println!("gpu_bitwise_lt OK: {n} LT ops → {bumps} bumps bin-for-bin identical to CPU");
}

#[test]
#[ignore = "requires GPU; run --ignored --nocapture"]
fn gpu_bitwise_store_matches_cpu() {
    if !have_gpu() {
        return;
    }
    let (nr, nt) = (bitwise::NUM_ROWS, bitwise::NUM_LOOKUP_TYPES);
    let n = 200_000usize;
    let ops: Vec<StoreOperation> =
        (0..n).map(|i| StoreOperation::new(0x1000 + i as u64, i as u64 * 4, mix(i, 3), 8)).collect();
    let mut ref_ops = Vec::new();
    for op in &ops {
        ref_ops.extend(op.collect_bitwise_ops());
    }
    let reference = scatter_ref(&ref_ops, nr, nt);

    let value: Vec<u64> = ops.iter().map(|o| o.value).collect();
    let gpu = math_cuda::bitwise_hist::gpu_bitwise_hist_store(&value, nr, nt).expect("gpu store hist");
    assert_eq!(gpu, reference, "GPU store histogram != CPU StoreOperation::collect_bitwise_ops");
    let bumps: u64 = gpu.iter().sum();
    println!("gpu_bitwise_store OK: {n} STORE ops → {bumps} bumps bin-for-bin identical to CPU");
}

#[test]
#[ignore = "requires GPU; run --ignored --nocapture"]
fn gpu_bitwise_bytewise_matches_cpu() {
    if !have_gpu() {
        return;
    }
    let (nr, nt) = (bitwise::NUM_ROWS, bitwise::NUM_LOOKUP_TYPES);
    let n = 200_000usize;
    let ops: Vec<BytewiseOperation> =
        (0..n).map(|i| BytewiseOperation::new(mix(i, 4), mix(i, 5), (i % 3) as u8)).collect();
    let mut ref_ops = Vec::new();
    for op in &ops {
        ref_ops.extend(op.collect_bitwise_ops());
    }
    let reference = scatter_ref(&ref_ops, nr, nt);

    let a: Vec<u64> = ops.iter().map(|o| o.a).collect();
    let b: Vec<u64> = ops.iter().map(|o| o.b).collect();
    let op: Vec<u8> = ops.iter().map(|o| o.op).collect();
    let gpu = math_cuda::bitwise_hist::gpu_bitwise_hist_bytewise(&a, &b, &op, nr, nt)
        .expect("gpu bytewise hist");
    assert_eq!(gpu, reference, "GPU bytewise histogram != CPU BytewiseOperation::collect_bitwise_ops");
    let bumps: u64 = gpu.iter().sum();
    println!("gpu_bitwise_bytewise OK: {n} BYTEWISE ops → {bumps} bumps bin-for-bin identical to CPU");
}

#[test]
#[ignore = "requires GPU; run --ignored --nocapture"]
fn gpu_bitwise_eq_matches_cpu() {
    if !have_gpu() {
        return;
    }
    let (nr, nt) = (bitwise::NUM_ROWS, bitwise::NUM_LOOKUP_TYPES);
    let n = 200_000usize;
    let ops: Vec<EqOperation> =
        (0..n).map(|i| EqOperation::new(mix(i, 6), mix(i, 7), i % 2 == 0)).collect();
    let mut ref_ops = Vec::new();
    for op in &ops {
        ref_ops.extend(op.collect_bitwise_ops());
    }
    let reference = scatter_ref(&ref_ops, nr, nt);

    let a: Vec<u64> = ops.iter().map(|o| o.a).collect();
    let b: Vec<u64> = ops.iter().map(|o| o.b).collect();
    let gpu = math_cuda::bitwise_hist::gpu_bitwise_hist_eq(&a, &b, nr, nt).expect("gpu eq hist");
    assert_eq!(gpu, reference, "GPU eq histogram != CPU EqOperation::collect_bitwise_ops");
    let bumps: u64 = gpu.iter().sum();
    println!("gpu_bitwise_eq OK: {n} EQ ops → {bumps} bumps bin-for-bin identical to CPU");
}

#[test]
#[ignore = "requires GPU; run --ignored --nocapture"]
fn gpu_bitwise_load_matches_cpu() {
    if !have_gpu() {
        return;
    }
    let (nr, nt) = (bitwise::NUM_ROWS, bitwise::NUM_LOOKUP_TYPES);
    let n = 200_000usize;
    let widths = [1u8, 2, 4, 8];
    let ops: Vec<LoadOperation> = (0..n)
        .map(|i| {
            let w = widths[i % 4];
            let v = mix(i, 8);
            let res: [u64; 8] = std::array::from_fn(|j| (v >> (j * 8)) & 0xFF);
            LoadOperation::new(0x2000 + i as u64, i as u64 * 4, w, i % 2 == 0, res)
        })
        .collect();
    let mut ref_ops = Vec::new();
    for op in &ops {
        ref_ops.extend(op.collect_bitwise_ops());
    }
    let reference = scatter_ref(&ref_ops, nr, nt);

    let mut res_flat = Vec::with_capacity(n * 8);
    for op in &ops {
        res_flat.extend_from_slice(&op.res);
    }
    let width: Vec<u32> = ops.iter().map(|o| o.width as u32).collect();
    let gpu = math_cuda::bitwise_hist::gpu_bitwise_hist_load(&res_flat, &width, nr, nt)
        .expect("gpu load hist");
    assert_eq!(gpu, reference, "GPU load histogram != CPU LoadOperation::collect_bitwise_ops");
    let bumps: u64 = gpu.iter().sum();
    println!("gpu_bitwise_load OK: {n} LOAD ops → {bumps} MSB8 bumps bin-for-bin identical to CPU");
}

#[test]
#[ignore = "requires GPU; run --ignored --nocapture"]
fn gpu_bitwise_cpu32_matches_cpu() {
    if !have_gpu() {
        return;
    }
    let (nr, nt) = (bitwise::NUM_ROWS, bitwise::NUM_LOOKUP_TYPES);
    let n = 100_000usize;
    let ops: Vec<Cpu32Operation> = (0..n)
        .map(|i| {
            let m = mix(i, 9);
            Cpu32Operation {
                timestamp: i as u64 * 4,
                pc: 0,
                rs1: (m & 0xFF) as u8,
                read_register1: true,
                rv1: mix(i, 10),
                rs2: ((m >> 8) & 0xFF) as u8,
                read_register2: true,
                rv2: mix(i, 11),
                imm: 0,
                res: mix(i, 12),
                rd: ((m >> 16) & 0xFF) as u8,
                write_register: true,
                alu: true,
                // Vary the SIGNED bit (5) and other alu_flags bits.
                alu_flags: ((m >> 24) & 0x3F) as u8,
                add: false,
                sub: false,
                half_instruction_length: ((m >> 32) & 0xFF) as u8,
            }
        })
        .collect();
    let mut ref_ops = Vec::new();
    for op in &ops {
        ref_ops.extend(collect_cpu32_bitwise(op));
    }
    let reference = scatter_ref(&ref_ops, nr, nt);

    let hil: Vec<u8> = ops.iter().map(|o| o.half_instruction_length).collect();
    let alu_flags: Vec<u8> = ops.iter().map(|o| o.alu_flags).collect();
    let rs1: Vec<u8> = ops.iter().map(|o| o.rs1).collect();
    let rs2: Vec<u8> = ops.iter().map(|o| o.rs2).collect();
    let rd: Vec<u8> = ops.iter().map(|o| o.rd).collect();
    let rv1: Vec<u64> = ops.iter().map(|o| o.rv1).collect();
    let rv2: Vec<u64> = ops.iter().map(|o| o.rv2).collect();
    let res: Vec<u64> = ops.iter().map(|o| o.res).collect();
    let gpu = math_cuda::bitwise_hist::gpu_bitwise_hist_cpu32(
        &hil, &alu_flags, &rs1, &rs2, &rd, &rv1, &rv2, &res, nr, nt,
    )
    .expect("gpu cpu32 hist");
    assert_eq!(gpu, reference, "GPU cpu32 histogram != CPU collect_cpu32_bitwise");
    let bumps: u64 = gpu.iter().sum();
    println!("gpu_bitwise_cpu32 OK: {n} CPU32 ops → {bumps} bumps bin-for-bin identical to CPU");
}

#[test]
#[ignore = "requires GPU; run --ignored --nocapture"]
fn gpu_bitwise_shift_matches_cpu() {
    if !have_gpu() {
        return;
    }
    let (nr, nt) = (bitwise::NUM_ROWS, bitwise::NUM_LOOKUP_TYPES);
    let n = 200_000usize;
    // Vary value (all halves), shift amount (incl. non-16-multiples → zbs=0), direction,
    // signed, word_instr — exercising every branch of compute_aux + the C1/C2/C4-C7 splits.
    let ops: Vec<ShiftOperation> = (0..n)
        .map(|i| {
            let value = mix(i, 20);
            let shift_amount = mix(i, 21);
            ShiftOperation::new(value, shift_amount, i % 2 == 0, i % 3 == 0, i % 5 == 0)
        })
        .collect();
    let reference = scatter_ref(&collect_bitwise_from_shift(&ops), nr, nt);

    let value: Vec<u64> = ops
        .iter()
        .map(|o| {
            (o.in_halves[0] as u64)
                | ((o.in_halves[1] as u64) << 16)
                | ((o.in_halves[2] as u64) << 32)
                | ((o.in_halves[3] as u64) << 48)
        })
        .collect();
    let shift: Vec<u8> = ops.iter().map(|o| o.shift).collect();
    let shift_amount: Vec<u64> = ops.iter().map(|o| o.shift_amount).collect();
    let flags: Vec<u32> = ops
        .iter()
        .map(|o| (o.direction as u32) | ((o.signed as u32) << 1) | ((o.word_instr as u32) << 2))
        .collect();
    let gpu = math_cuda::bitwise_hist::gpu_bitwise_hist_shift(
        &value, &shift, &shift_amount, &flags, nr, nt,
    )
    .expect("gpu shift hist");
    assert_eq!(gpu, reference, "GPU shift histogram != CPU collect_bitwise_from_shift");
    let bumps: u64 = gpu.iter().sum();
    println!("gpu_bitwise_shift OK: {n} SHIFT ops → {bumps} bumps bin-for-bin identical to CPU");
}

#[test]
#[ignore = "requires GPU; run --ignored --nocapture"]
fn gpu_bitwise_mul_perop_matches_cpu() {
    if !have_gpu() {
        return;
    }
    let (nr, nt) = (bitwise::NUM_ROWS, bitwise::NUM_LOOKUP_TYPES);
    let n = 100_000usize;
    // UNSIGNED ops → the collector's per-chunk MSB16 dedup emits nothing, so the full
    // collector output equals the PER-OP part the kernel computes. Exercises the 128-bit
    // product, raw-product, and carry math.
    let ops: Vec<(MulOperation, bool)> = (0..n)
        .map(|i| (MulOperation::new(mix(i, 30), false, mix(i, 31), false), i % 2 == 0))
        .collect();
    let reference = scatter_ref(&collect_bitwise_from_mul(&ops, n), nr, nt);

    let lhs: Vec<u64> = ops.iter().map(|(o, _)| o.lhs).collect();
    let rhs: Vec<u64> = ops.iter().map(|(o, _)| o.rhs).collect();
    let flags: Vec<u32> = vec![0u32; n]; // unsigned
    let gpu = math_cuda::bitwise_hist::gpu_bitwise_hist_mul_perop(&lhs, &rhs, &flags, nr, nt)
        .expect("gpu mul_perop hist");
    assert_eq!(gpu, reference, "GPU mul per-op histogram != CPU collect_bitwise_from_mul (unsigned)");
    let bumps: u64 = gpu.iter().sum();
    println!("gpu_bitwise_mul_perop OK: {n} unsigned MUL ops → {bumps} per-op bumps bin-for-bin identical to CPU");
}

#[test]
#[ignore = "requires GPU; run --ignored --nocapture"]
fn gpu_bitwise_dvrm_perop_matches_cpu() {
    if !have_gpu() {
        return;
    }
    let (nr, nt) = (bitwise::NUM_ROWS, bitwise::NUM_LOOKUP_TYPES);
    let n = 100_000usize;
    // UNSIGNED ops → MSB16 + NEG-template ZERO (both signed-only) contribute nothing, so the
    // full collector equals the PER-OP part. Includes d==0 (div-by-zero special case).
    let ops: Vec<(DvrmOperation, bool)> = (0..n)
        .map(|i| {
            let d = if i % 97 == 0 { 0 } else { mix(i, 41) };
            (DvrmOperation::new(mix(i, 40), d, false), i % 2 == 0)
        })
        .collect();
    let reference = scatter_ref(&collect_bitwise_from_dvrm(&ops, n), nr, nt);

    let n_vals: Vec<u64> = ops.iter().map(|(o, _)| o.n).collect();
    let d_vals: Vec<u64> = ops.iter().map(|(o, _)| o.d).collect();
    let flags: Vec<u32> = vec![0u32; n]; // unsigned
    let gpu = math_cuda::bitwise_hist::gpu_bitwise_hist_dvrm_perop(&n_vals, &d_vals, &flags, nr, nt)
        .expect("gpu dvrm_perop hist");
    assert_eq!(gpu, reference, "GPU dvrm per-op histogram != CPU collect_bitwise_from_dvrm (unsigned)");
    let bumps: u64 = gpu.iter().sum();
    println!("gpu_bitwise_dvrm_perop OK: {n} unsigned DVRM ops → {bumps} per-op bumps bin-for-bin identical to CPU");
}

#[test]
#[ignore = "requires GPU; run --ignored --nocapture"]
fn gpu_bitwise_branch_matches_cpu() {
    if !have_gpu() {
        return;
    }
    let (nr, nt) = (bitwise::NUM_ROWS, bitwise::NUM_LOOKUP_TYPES);
    let n = 200_000usize;
    let ops: Vec<BranchOperation> = (0..n)
        .map(|i| {
            let jalr = i % 3 == 0;
            BranchOperation::new(mix(i, 13) & 0xFFFF_FFFF, mix(i, 14), mix(i, 15), jalr)
        })
        .collect();
    let reference = scatter_ref(&collect_bitwise_from_branch(&ops), nr, nt);

    let next_pc: Vec<u64> = ops.iter().map(|o| o.compute_next_pc()).collect();
    let next_pc_unmasked: Vec<u64> = ops.iter().map(|o| o.compute_next_pc_unmasked()).collect();
    let gpu = math_cuda::bitwise_hist::gpu_bitwise_hist_branch(&next_pc, &next_pc_unmasked, nr, nt)
        .expect("gpu branch hist");
    assert_eq!(gpu, reference, "GPU branch histogram != CPU collect_bitwise_from_branch");
    let bumps: u64 = gpu.iter().sum();
    println!("gpu_bitwise_branch OK: {n} BRANCH ops → {bumps} bumps bin-for-bin identical to CPU");
}

/// S3: the RESIDENT-seam BRANCH+LOAD kernel (`bitwise_hist_branch_load_packed`) — self-routes by
/// `packed`/`flags` and computes next_pc (branch) / Msb8 (load) on-device from pc/imm/rv1/rvd — must
/// equal the SUM of the CPU `collect_bitwise_from_branch` + `LoadOperation::collect_bitwise_ops`
/// over the same cycles. Synthetic resident seam: half BRANCH (jalr + non-jalr), half LOAD
/// (widths 1/2/4/8), disjoint as in real traces. Validates the on-device arithmetic in isolation;
/// the e2e (bus balance) validates the real-data wiring.
#[test]
#[ignore = "requires GPU; run --ignored --nocapture"]
fn gpu_bitwise_branch_load_packed_matches_cpu() {
    if !have_gpu() {
        return;
    }
    let (nr, nt) = (bitwise::NUM_ROWS, bitwise::NUM_LOOKUP_TYPES);
    let n = 200_000usize;
    let mut packed = vec![0u64; n];
    let mut flags = vec![0u8; n];
    let mut pc = vec![0u64; n];
    let mut imm = vec![0u64; n];
    let mut rv1 = vec![0u64; n];
    let mut rvd = vec![0u64; n];
    let mut branch: Vec<BranchOperation> = Vec::new();
    let mut loads: Vec<LoadOperation> = Vec::new();
    let widths = [1u8, 2, 4, 8];
    // Packed decode bit layout the kernel reads (matches ShrunkDecode::pack / BH_PD_*):
    // memory @ bit 7, mem_flags byte @ bit 50 (bit0 = jalr/store, 2B @ bit2, 4B @ bit3, 8B @ bit4).
    for i in 0..n {
        if i % 2 == 0 {
            // BRANCH: branch_cond = flags bit0; jalr = mem_flags bit0.
            let jalr = i % 3 == 0;
            flags[i] = 1;
            packed[i] = (jalr as u64) << 50;
            pc[i] = mix(i, 13) & 0xFFFF_FFFF;
            imm[i] = mix(i, 14);
            rv1[i] = mix(i, 15);
            branch.push(BranchOperation::new(pc[i], imm[i], rv1[i], jalr));
        } else {
            // LOAD: memory=1, load (mem_flags bit0=0); width bits per `mem_bytes`.
            let w = widths[(i / 2) % 4];
            let mem_flags: u64 = match w {
                2 => 1 << 2,
                4 => 1 << 3,
                8 => 1 << 4,
                _ => 0, // width 1: no width bit
            };
            packed[i] = (mem_flags << 50) | (1u64 << 7);
            let v = mix(i, 8);
            rvd[i] = v;
            let res: [u64; 8] = std::array::from_fn(|j| (v >> (j * 8)) & 0xFF);
            loads.push(LoadOperation::new(0x2000 + i as u64, i as u64 * 4, w, i % 2 == 0, res));
        }
    }
    let mut ref_ops = collect_bitwise_from_branch(&branch);
    for o in &loads {
        ref_ops.extend(o.collect_bitwise_ops());
    }
    let reference = scatter_ref(&ref_ops, nr, nt);

    let gpu = math_cuda::bitwise_hist::gpu_bitwise_hist_branch_load_packed(
        &packed, &flags, &pc, &imm, &rv1, &rvd, nr, nt,
    )
    .expect("gpu branch_load packed hist");
    assert_eq!(
        gpu, reference,
        "GPU branch_load packed histogram != CPU branch + load collectors"
    );
    let bumps: u64 = gpu.iter().sum();
    println!(
        "gpu_bitwise_branch_load_packed OK: {} BRANCH + {} LOAD ops → {bumps} bumps bin-for-bin identical to CPU",
        branch.len(),
        loads.len()
    );
}

/// S3: the CPU32 histogram source computed from the PACKED device op rows
/// (`bitwise_hist_cpu32_packed`, fed rows in `pack_cpu32_op` layout) must equal the CPU
/// `collect_cpu32_bitwise`. In the real pipeline the rows come from the device `build_cpu32_ops`
/// (res validated == build_cpu32_op elsewhere); here we pack host ops to isolate the scatter's
/// packed-row unpack + bump logic. Same op set as `gpu_bitwise_cpu32_matches_cpu`.
#[test]
#[ignore = "requires GPU; run --ignored --nocapture"]
fn gpu_bitwise_cpu32_packed_matches_cpu() {
    if !have_gpu() {
        return;
    }
    let (nr, nt) = (bitwise::NUM_ROWS, bitwise::NUM_LOOKUP_TYPES);
    let n = 100_000usize;
    let ops: Vec<Cpu32Operation> = (0..n)
        .map(|i| {
            let m = mix(i, 9);
            Cpu32Operation {
                timestamp: i as u64 * 4,
                pc: 0,
                rs1: (m & 0xFF) as u8,
                read_register1: true,
                rv1: mix(i, 10),
                rs2: ((m >> 8) & 0xFF) as u8,
                read_register2: true,
                rv2: mix(i, 11),
                imm: 0,
                res: mix(i, 12),
                rd: ((m >> 16) & 0xFF) as u8,
                write_register: true,
                alu: true,
                alu_flags: ((m >> 24) & 0x3F) as u8,
                add: false,
                sub: false,
                half_instruction_length: ((m >> 32) & 0xFF) as u8,
            }
        })
        .collect();
    let mut ref_ops = Vec::new();
    for op in &ops {
        ref_ops.extend(collect_cpu32_bitwise(op));
    }
    let reference = scatter_ref(&ref_ops, nr, nt);

    // Pack the ops exactly as the device op-build emits them (pack_cpu32_op).
    let mut rows = Vec::with_capacity(n * 8);
    for op in &ops {
        rows.extend_from_slice(&crate::tables::gpu_trace::pack_cpu32_op(op));
    }
    let gpu = math_cuda::bitwise_hist::gpu_bitwise_hist_cpu32_packed(&rows, nr, nt)
        .expect("gpu cpu32 packed hist");
    assert_eq!(gpu, reference, "GPU cpu32 packed histogram != CPU collect_cpu32_bitwise");
    let bumps: u64 = gpu.iter().sum();
    println!("gpu_bitwise_cpu32_packed OK: {n} CPU32 ops → {bumps} bumps bin-for-bin identical to CPU");
}

/// S3: the SHIFT histogram source from the PACKED device op rows (`bitwise_hist_shift_packed`, fed
/// 3-u64 rows [value, shift_amount, flags]) must equal the CPU `collect_bitwise_from_shift`. In the
/// real pipeline the rows come from `build_shift_ops`/`cpu32_shift_ops`; here we pack host ops to
/// isolate the scatter (shift = shift_amount low byte, derived on device). Same op set as
/// `gpu_bitwise_shift_matches_cpu`.
#[test]
#[ignore = "requires GPU; run --ignored --nocapture"]
fn gpu_bitwise_shift_packed_matches_cpu() {
    if !have_gpu() {
        return;
    }
    let (nr, nt) = (bitwise::NUM_ROWS, bitwise::NUM_LOOKUP_TYPES);
    let n = 200_000usize;
    let ops: Vec<ShiftOperation> = (0..n)
        .map(|i| ShiftOperation::new(mix(i, 20), mix(i, 21), i % 2 == 0, i % 3 == 0, i % 5 == 0))
        .collect();
    let reference = scatter_ref(&collect_bitwise_from_shift(&ops), nr, nt);

    // Pack rows exactly as build_shift_ops/cpu32_shift_ops emit: [value, shift_amount, flags].
    let mut rows = Vec::with_capacity(n * 3);
    for o in &ops {
        let value = (o.in_halves[0] as u64)
            | ((o.in_halves[1] as u64) << 16)
            | ((o.in_halves[2] as u64) << 32)
            | ((o.in_halves[3] as u64) << 48);
        let flags =
            (o.direction as u64) | ((o.signed as u64) << 1) | ((o.word_instr as u64) << 2);
        rows.push(value);
        rows.push(o.shift_amount);
        rows.push(flags);
    }
    let gpu = math_cuda::bitwise_hist::gpu_bitwise_hist_shift_packed(&rows, nr, nt)
        .expect("gpu shift packed hist");
    assert_eq!(gpu, reference, "GPU shift packed histogram != CPU collect_bitwise_from_shift");
    let bumps: u64 = gpu.iter().sum();
    println!("gpu_bitwise_shift_packed OK: {n} SHIFT ops → {bumps} bumps bin-for-bin identical to CPU");
}

/// S3: the MUL per-op histogram source from the MERGED device key stream (`bitwise_hist_mul_perop_packed`,
/// fed k0=flags/k1=lhs/k2=rhs) must equal the CPU `collect_bitwise_from_mul` per-op part. UNSIGNED ops
/// (the collector's MSB16 dedup contributes nothing → per-op == full). Same op set as
/// `gpu_bitwise_mul_perop_matches_cpu`.
#[test]
#[ignore = "requires GPU; run --ignored --nocapture"]
fn gpu_bitwise_mul_perop_packed_matches_cpu() {
    if !have_gpu() {
        return;
    }
    let (nr, nt) = (bitwise::NUM_ROWS, bitwise::NUM_LOOKUP_TYPES);
    let n = 100_000usize;
    let ops: Vec<(MulOperation, bool)> = (0..n)
        .map(|i| (MulOperation::new(mix(i, 30), false, mix(i, 31), false), i % 2 == 0))
        .collect();
    let reference = scatter_ref(&collect_bitwise_from_mul(&ops, n), nr, nt);

    // Merged key stream layout (mul_key_gather): k0=flags (lhs_signed|rhs_signed<<1), k1=lhs, k2=rhs.
    let k0: Vec<u64> =
        ops.iter().map(|(o, _)| (o.lhs_signed as u64) | ((o.rhs_signed as u64) << 1)).collect();
    let k1: Vec<u64> = ops.iter().map(|(o, _)| o.lhs).collect();
    let k2: Vec<u64> = ops.iter().map(|(o, _)| o.rhs).collect();
    let gpu = math_cuda::bitwise_hist::gpu_bitwise_hist_mul_perop_packed(&k0, &k1, &k2, nr, nt)
        .expect("gpu mul_perop packed hist");
    assert_eq!(gpu, reference, "GPU mul per-op packed histogram != CPU collect_bitwise_from_mul (unsigned)");
    let bumps: u64 = gpu.iter().sum();
    println!("gpu_bitwise_mul_perop_packed OK: {n} unsigned MUL ops → {bumps} per-op bumps bin-for-bin identical to CPU");
}

/// S3: the DVRM per-op histogram source from the MERGED device key stream (`bitwise_hist_dvrm_perop_packed`,
/// fed k0=flags(signed)/k1=n/k2=d) must equal the CPU `collect_bitwise_from_dvrm` per-op part (UNSIGNED,
/// incl. div-by-zero). Same op set as `gpu_bitwise_dvrm_perop_matches_cpu`.
#[test]
#[ignore = "requires GPU; run --ignored --nocapture"]
fn gpu_bitwise_dvrm_perop_packed_matches_cpu() {
    if !have_gpu() {
        return;
    }
    let (nr, nt) = (bitwise::NUM_ROWS, bitwise::NUM_LOOKUP_TYPES);
    let n = 100_000usize;
    let ops: Vec<(DvrmOperation, bool)> = (0..n)
        .map(|i| {
            let d = if i % 97 == 0 { 0 } else { mix(i, 41) };
            (DvrmOperation::new(mix(i, 40), d, false), i % 2 == 0)
        })
        .collect();
    let reference = scatter_ref(&collect_bitwise_from_dvrm(&ops, n), nr, nt);

    // Merged key stream layout (dvrm_key_gather): k0=flags(signed), k1=n, k2=d.
    let k0: Vec<u64> = ops.iter().map(|(o, _)| o.signed as u64).collect();
    let k1: Vec<u64> = ops.iter().map(|(o, _)| o.n).collect();
    let k2: Vec<u64> = ops.iter().map(|(o, _)| o.d).collect();
    let gpu = math_cuda::bitwise_hist::gpu_bitwise_hist_dvrm_perop_packed(&k0, &k1, &k2, nr, nt)
        .expect("gpu dvrm_perop packed hist");
    assert_eq!(gpu, reference, "GPU dvrm per-op packed histogram != CPU collect_bitwise_from_dvrm (unsigned)");
    let bumps: u64 = gpu.iter().sum();
    println!("gpu_bitwise_dvrm_perop_packed OK: {n} unsigned DVRM ops → {bumps} per-op bumps bin-for-bin identical to CPU");
}

/// P4b assembly: `gpu_bitwise_hist_opvec` scatters ALL fully-covered op-vec sources (lt, store,
/// bytewise, eq, load, cpu32, branch, shift) into ONE histogram; the total must equal the sum of the
/// real CPU collectors over the same ops. Confirms the multi-source scatter+reduce has no cross-source
/// corruption (each source's kernel is already validated 1:1 above).
#[test]
#[ignore = "requires GPU; run --ignored --nocapture"]
fn gpu_bitwise_opvec_assembly_matches_cpu() {
    if !have_gpu() {
        return;
    }
    let (nr, nt) = (bitwise::NUM_ROWS, bitwise::NUM_LOOKUP_TYPES);
    let n = 50_000usize;

    let lt: Vec<LtOperation> = (0..n).map(|i| LtOperation::new(mix(i, 1), mix(i, 2), i % 2 == 0)).collect();
    let store: Vec<StoreOperation> =
        (0..n).map(|i| StoreOperation::new(0x1000 + i as u64, i as u64 * 4, mix(i, 3), 8)).collect();
    let byw: Vec<BytewiseOperation> =
        (0..n).map(|i| BytewiseOperation::new(mix(i, 4), mix(i, 5), (i % 3) as u8)).collect();
    let eq: Vec<EqOperation> = (0..n).map(|i| EqOperation::new(mix(i, 6), mix(i, 7), i % 2 == 0)).collect();
    let widths = [1u8, 2, 4, 8];
    let load: Vec<LoadOperation> = (0..n)
        .map(|i| {
            let v = mix(i, 8);
            let res: [u64; 8] = std::array::from_fn(|j| (v >> (j * 8)) & 0xFF);
            LoadOperation::new(0x2000 + i as u64, i as u64 * 4, widths[i % 4], i % 2 == 0, res)
        })
        .collect();
    let cpu32: Vec<Cpu32Operation> = (0..n)
        .map(|i| {
            let m = mix(i, 9);
            Cpu32Operation {
                timestamp: i as u64 * 4, pc: 0, rs1: (m & 0xFF) as u8, read_register1: true,
                rv1: mix(i, 10), rs2: ((m >> 8) & 0xFF) as u8, read_register2: true, rv2: mix(i, 11),
                imm: 0, res: mix(i, 12), rd: ((m >> 16) & 0xFF) as u8, write_register: true, alu: true,
                alu_flags: ((m >> 24) & 0x3F) as u8, add: false, sub: false,
                half_instruction_length: ((m >> 32) & 0xFF) as u8,
            }
        })
        .collect();
    let branch: Vec<BranchOperation> =
        (0..n).map(|i| BranchOperation::new(mix(i, 13) & 0xFFFF_FFFF, mix(i, 14), mix(i, 15), i % 3 == 0)).collect();
    let shift: Vec<ShiftOperation> =
        (0..n).map(|i| ShiftOperation::new(mix(i, 20), mix(i, 21), i % 2 == 0, i % 3 == 0, i % 5 == 0)).collect();

    // CPU reference: sum of all source collectors.
    let mut refh = vec![0u64; nr * nt];
    let mut add = |ops: &[BitwiseOperation]| {
        for op in ops {
            refh[bitwise::lookup_type_index(op.lookup_type) * nr + bitwise::row_index(op.x, op.y, op.z)] += 1;
        }
    };
    add(&collect_bitwise_from_lt(&lt));
    for o in &store { add(&o.collect_bitwise_ops()); }
    for o in &byw { add(&o.collect_bitwise_ops()); }
    for o in &eq { add(&o.collect_bitwise_ops()); }
    for o in &load { add(&o.collect_bitwise_ops()); }
    for c in &cpu32 { add(&collect_cpu32_bitwise(c)); }
    add(&collect_bitwise_from_branch(&branch));
    add(&collect_bitwise_from_shift(&shift));

    // GPU assembly inputs.
    let mut load_res = Vec::with_capacity(n * 8);
    for o in &load { load_res.extend_from_slice(&o.res); }
    let src = math_cuda::bitwise_hist::OpVecSources {
        lt_lhs: &lt.iter().map(|o| o.lhs).collect::<Vec<_>>(),
        lt_rhs: &lt.iter().map(|o| o.rhs).collect::<Vec<_>>(),
        store_val: &store.iter().map(|o| o.value).collect::<Vec<_>>(),
        bytewise_a: &byw.iter().map(|o| o.a).collect::<Vec<_>>(),
        bytewise_b: &byw.iter().map(|o| o.b).collect::<Vec<_>>(),
        bytewise_op: &byw.iter().map(|o| o.op).collect::<Vec<_>>(),
        eq_a: &eq.iter().map(|o| o.a).collect::<Vec<_>>(),
        eq_b: &eq.iter().map(|o| o.b).collect::<Vec<_>>(),
        load_res: &load_res,
        load_width: &load.iter().map(|o| o.width as u32).collect::<Vec<_>>(),
        cpu32_hil: &cpu32.iter().map(|o| o.half_instruction_length).collect::<Vec<_>>(),
        cpu32_alu: &cpu32.iter().map(|o| o.alu_flags).collect::<Vec<_>>(),
        cpu32_rs1: &cpu32.iter().map(|o| o.rs1).collect::<Vec<_>>(),
        cpu32_rs2: &cpu32.iter().map(|o| o.rs2).collect::<Vec<_>>(),
        cpu32_rd: &cpu32.iter().map(|o| o.rd).collect::<Vec<_>>(),
        cpu32_rv1: &cpu32.iter().map(|o| o.rv1).collect::<Vec<_>>(),
        cpu32_rv2: &cpu32.iter().map(|o| o.rv2).collect::<Vec<_>>(),
        cpu32_res: &cpu32.iter().map(|o| o.res).collect::<Vec<_>>(),
        branch_next_pc: &branch.iter().map(|o| o.compute_next_pc()).collect::<Vec<_>>(),
        branch_unmasked: &branch.iter().map(|o| o.compute_next_pc_unmasked()).collect::<Vec<_>>(),
        shift_value: &shift.iter().map(|o| {
            (o.in_halves[0] as u64) | ((o.in_halves[1] as u64) << 16)
                | ((o.in_halves[2] as u64) << 32) | ((o.in_halves[3] as u64) << 48)
        }).collect::<Vec<_>>(),
        shift_shift: &shift.iter().map(|o| o.shift).collect::<Vec<_>>(),
        shift_amount: &shift.iter().map(|o| o.shift_amount).collect::<Vec<_>>(),
        shift_flags: &shift.iter().map(|o| {
            (o.direction as u32) | ((o.signed as u32) << 1) | ((o.word_instr as u32) << 2)
        }).collect::<Vec<_>>(),
        ..Default::default()
    };
    let gpu = math_cuda::bitwise_hist::gpu_bitwise_hist_opvec(&src, nr, nt).expect("gpu opvec assembly");
    assert_eq!(gpu, refh, "assembled GPU op-vec histogram != sum of CPU collectors");
    let bumps: u64 = gpu.iter().sum();
    println!("gpu_bitwise_opvec_assembly OK: 8 sources × {n} ops → {bumps} bumps, assembled total bin-for-bin identical to CPU");
}
