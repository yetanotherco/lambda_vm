//! Parity for the on-GPU BITWISE histogram [`gpu_bitwise_hist`]: the device atomic-add
//! scatter (replicated + reduced) must reproduce, bin-for-bin, the counter array a
//! sequential CPU pass over the same lookups produces — for the in-walk per-op source
//! (`collect_bitwise_ops`: 3 ARE_BYTES + 4 IS_HALF) and the MEMW_R source (1 IS_HALF per
//! row, keyed by the ts delta), together. Layout `hist[type*num_rows + x + y*256]`.
//! Skips without CUDA.

use math_cuda::bitwise_hist::{CpuOpFields, gpu_bitwise_hist};
use math_cuda::device::backend;

const NUM_ROWS: usize = 256 * 256 * 16; // 2^20, matches prover bitwise::NUM_ROWS
const NUM_TYPES: usize = 10;
const ARE: usize = 3 * NUM_ROWS; // AreBytes lane
const ISH: usize = 4 * NUM_ROWS; // IsHalf lane

/// Sequential reference: replicate `collect_bitwise_ops` + `bump` for each op.
#[allow(clippy::too_many_arguments)]
fn cpu_hist(
    rs1: &[u8],
    rs2: &[u8],
    rd: &[u8],
    hil: &[u8],
    alu: &[u8],
    mem: &[u8],
    res: &[u64],
    word: &[u8],
) -> Vec<u64> {
    let mut h = vec![0u64; NUM_ROWS * NUM_TYPES];
    for i in 0..rs1.len() {
        let w = word[i] != 0;
        let z = |v: u8| if w { 0usize } else { v as usize };
        let hl = hil[i] as usize;
        let r = if w { 0u64 } else { res[i] };
        h[ARE + z(rs1[i]) + z(rs2[i]) * 256] += 1;
        h[ARE + z(rd[i]) + hl * 256] += 1;
        h[ARE + z(alu[i]) + z(mem[i]) * 256] += 1;
        for k in 0..4 {
            let half = ((r >> (k * 16)) & 0xFFFF) as usize;
            h[ISH + half] += 1;
        }
    }
    h
}

/// MEMW_R reference: one IS_HALF per row, keyed by `diff = ts_lo - old_ts_lo - 1`.
fn cpu_memw(h: &mut [u64], ts: &[u64], old_ts: &[u64]) {
    for i in 0..ts.len() {
        let ts_lo = (ts[i] & 0xFFFF_FFFF) as u32;
        let ot_lo = (old_ts[i] & 0xFFFF_FFFF) as u32;
        let diff = (ts_lo.wrapping_sub(ot_lo).wrapping_sub(1) & 0xFFFF) as usize;
        h[ISH + diff] += 1;
    }
}

#[test]
fn gpu_bitwise_hist_matches_cpu() {
    if backend().is_err() {
        eprintln!("skipping gpu_bitwise_hist_matches_cpu: no CUDA backend");
        return;
    }
    let n = 200_000usize;
    let mut rs1 = Vec::with_capacity(n);
    let mut rs2 = Vec::with_capacity(n);
    let mut rd = Vec::with_capacity(n);
    let mut hil = Vec::with_capacity(n);
    let mut alu = Vec::with_capacity(n);
    let mut mem = Vec::with_capacity(n);
    let mut res = Vec::with_capacity(n);
    let mut word = Vec::with_capacity(n);
    for i in 0..n as u64 {
        // Registers 0..31 (hot, clustered → high atomic contention); flags/hil small;
        // res spreads over the IS_HALF lane. Every 9th op is a word instruction.
        rs1.push((i % 32) as u8);
        rs2.push(((i / 32) % 32) as u8);
        rd.push(((i / 7) % 32) as u8);
        hil.push((2 + (i % 3)) as u8);
        alu.push((i % 200) as u8);
        mem.push((i % 40) as u8);
        res.push(i.wrapping_mul(0x9E37_79B9_7F4A_7C15));
        word.push(u8::from(i % 9 == 0));
    }

    // MEMW_R source: rows with varied ts deltas (in-range → small diff).
    let m = 120_000usize;
    let mut ts = Vec::with_capacity(m);
    let mut old_ts = Vec::with_capacity(m);
    for i in 0..m as u64 {
        old_ts.push(i * 4 + 4);
        ts.push(i * 4 + 4 + 1 + (i % 300)); // diff = ts_lo-old_ts_lo-1 = i%300
    }

    let fields = CpuOpFields {
        rs1: &rs1,
        rs2: &rs2,
        rd: &rd,
        hil: &hil,
        alu_flags: &alu,
        mem_flags: &mem,
        res: &res,
        word: &word,
    };
    let gpu =
        gpu_bitwise_hist(&fields, &ts, &old_ts, NUM_ROWS, NUM_TYPES).expect("device bitwise hist");
    let mut cpu = cpu_hist(&rs1, &rs2, &rd, &hil, &alu, &mem, &res, &word);
    cpu_memw(&mut cpu, &ts, &old_ts);

    assert_eq!(gpu.len(), cpu.len());
    assert!(
        gpu == cpu,
        "device BITWISE histogram (in-walk + MEMW_R) must match the CPU counts"
    );
    // Sanity: 7 bumps per cpu op + 1 per memw row.
    let total: u64 = gpu.iter().sum();
    assert_eq!(total, 7 * n as u64 + m as u64);
}
