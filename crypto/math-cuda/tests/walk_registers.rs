//! Parity for the on-GPU register memory-model walk [`gpu_walk_registers`]: the
//! device stable group-by + predecessor link must recover the same per-access
//! `(old_value, old_ts)` as a straightforward sequential per-bucket reference —
//! the read-old/write-new semantics the prover's `walk_register_accesses`
//! implements (every access advances the cell timeline, so `old` is the previous
//! *access* at the same bucket). Skips cleanly without a CUDA backend.

use math_cuda::device::backend;
use math_cuda::trace_cpu::{MEMW_REGISTER_NCOLS, gpu_walk_and_fill_memw_register_host};
use math_cuda::trace_walk::gpu_walk_registers;

/// Sequential reference: `old` = the previous access at the same bucket in input
/// order, seeded per bucket from `init_value` at timestamp 1.
fn cpu_walk(key: &[u32], ts: &[u64], value: &[u64], init_value: &[u64]) -> (Vec<u64>, Vec<u64>) {
    let mut cell: Vec<(u64, u64)> = init_value.iter().map(|&v| (v, 1u64)).collect();
    let n = key.len();
    let mut old_value = vec![0u64; n];
    let mut old_ts = vec![0u64; n];
    for i in 0..n {
        let b = key[i] as usize;
        old_value[i] = cell[b].0;
        old_ts[i] = cell[b].1;
        cell[b] = (value[i], ts[i]);
    }
    (old_value, old_ts)
}

#[test]
fn gpu_walk_matches_cpu_reference() {
    if backend().is_err() {
        eprintln!("skipping gpu_walk_matches_cpu_reference: no CUDA backend");
        return;
    }
    let nbins: u32 = 512;
    // > seg_size (4096) → many segments / multi-block, exercising the group-by.
    let n = 60_000usize;
    let mut key = Vec::with_capacity(n);
    let mut ts = Vec::with_capacity(n);
    let mut value = Vec::with_capacity(n);
    for i in 0..n as u64 {
        // Skewed keys: bucket 510 (the PC) is hot (~1/4 of accesses), the rest
        // spread over 0..300 — exercises hot buckets and empty buckets.
        let k = if i % 4 == 0 { 510 } else { (i % 300) as u32 };
        key.push(k);
        ts.push(i + 100);
        value.push(i.wrapping_mul(0x9E37_79B9).wrapping_add(1));
    }
    let init_value: Vec<u64> = (0..nbins as u64).map(|b| b * 1000).collect();

    let (gpu_ov, gpu_ot) =
        gpu_walk_registers(&key, &ts, &value, &init_value, 1, nbins).expect("device walk");
    let (cpu_ov, cpu_ot) = cpu_walk(&key, &ts, &value, &init_value);

    assert_eq!(gpu_ov, cpu_ov, "old_value mismatch");
    assert_eq!(gpu_ot, cpu_ot, "old_ts mismatch");
}

/// Reference for the combined device walk + MEMW_R fill: walk every access
/// (timeline-only accesses included), then lay out the 10 MEMW_R columns for the
/// emitting rows (`row_index >= 0`), mirroring the `memw_register_fill` kernel.
#[allow(clippy::too_many_arguments)]
fn cpu_walk_and_fill(
    reg_addr: &[u32],
    ts: &[u64],
    value: &[u64],
    is_read: &[u8],
    row_index: &[i64],
    init_value: &[u64],
    init_ts: u64,
    num_rows: usize,
) -> Vec<u64> {
    let ncols = MEMW_REGISTER_NCOLS;
    let mut cell: Vec<(u64, u64)> = init_value.iter().map(|&v| (v, init_ts)).collect();
    let mut buf = vec![0u64; num_rows * ncols];
    for i in 0..reg_addr.len() {
        let b = reg_addr[i] as usize;
        let (ov, ot) = cell[b];
        cell[b] = (value[i], ts[i]); // every access advances the cell timeline
        let row = row_index[i];
        if row < 0 {
            continue; // timeline-only access (e.g. implicit PC write): no row
        }
        let base = row as usize * ncols;
        let (v, t) = (value[i], ts[i]);
        buf[base] = (reg_addr[i] / 2) as u64;
        buf[base + 1] = t & 0xFFFF_FFFF;
        buf[base + 2] = t >> 32;
        buf[base + 3] = v & 0xFFFF_FFFF;
        buf[base + 4] = v >> 32;
        buf[base + 5] = ov & 0xFFFF_FFFF;
        buf[base + 6] = ov >> 32;
        buf[base + 7] = ot & 0xFFFF_FFFF;
        buf[base + 8] = u64::from(is_read[i] != 0);
        buf[base + 9] = u64::from(is_read[i] == 0);
    }
    buf
}

/// The device walk feeding the device MEMW_R fill (no host walk, no `old_*`
/// upload) must produce the exact row-major MEMW_R buffer the sequential
/// walk + fill would. Includes timeline-only accesses (`row_index = -1`, the
/// implicit PC bump) and trailing zero-padded rows.
#[test]
fn gpu_walk_and_fill_matches_cpu_reference() {
    if backend().is_err() {
        eprintln!("skipping gpu_walk_and_fill_matches_cpu_reference: no CUDA backend");
        return;
    }
    let nbins: u32 = 512;
    let n = 40_000usize;
    let mut reg_addr = Vec::with_capacity(n);
    let mut ts = Vec::with_capacity(n);
    let mut value = Vec::with_capacity(n);
    let mut is_read = Vec::with_capacity(n);
    let mut row_index: Vec<i64> = Vec::with_capacity(n);
    let mut next_row: i64 = 0;
    for i in 0..n as u64 {
        // Even accesses hit the PC word address (510); odd spread over 0..250 (all
        // even word addresses, i.e. registers). Reads and writes interleave.
        let k = if i % 2 == 0 {
            510
        } else {
            (2 * (i % 250)) as u32
        };
        reg_addr.push(k);
        ts.push(i + 5);
        value.push(i.wrapping_mul(0x9E37_79B9).wrapping_add(7));
        is_read.push((i % 3 == 0) as u8);
        // Every 5th access is timeline-only (mimics the implicit PC write): it
        // advances the cell but emits no MEMW_R row.
        if i % 5 == 4 {
            row_index.push(-1);
        } else {
            row_index.push(next_row);
            next_row += 1;
        }
    }
    // Pad past the emitting-row count to exercise trailing zero rows.
    let num_rows = next_row as usize + 16;
    let init_value: Vec<u64> = (0..nbins as u64)
        .map(|b| b.wrapping_mul(2654435761))
        .collect();
    let init_ts = 1u64;

    let gpu = gpu_walk_and_fill_memw_register_host(
        &reg_addr,
        &ts,
        &value,
        &is_read,
        &row_index,
        &init_value,
        init_ts,
        nbins,
        num_rows,
    )
    .expect("device walk+fill");
    let cpu = cpu_walk_and_fill(
        &reg_addr,
        &ts,
        &value,
        &is_read,
        &row_index,
        &init_value,
        init_ts,
        num_rows,
    );

    assert_eq!(gpu.len(), cpu.len(), "buffer length mismatch");
    assert_eq!(gpu, cpu, "MEMW_R row-major buffer mismatch");
}
