//! Byte-parity for the device MEMW_R build via the on-GPU walk + route + compact
//! ([`math_cuda::trace_walk::gpu_walk_route_memw_register_host`], the "option A"
//! path). The device pipeline — walk → route (`reg_ts_delta_in_range`) → compact →
//! fill + IS_HALFWORD histogram + fallback gather — must reproduce, bit-for-bit, the
//! sequential CPU path (`walk_register_accesses` into [`MemwBuckets`], then
//! `generate_memw_register_trace_from_rows` and `collect_bitwise_from_memw_register`).
//! The MEMW_R matrix comparison against the real prover fill validates the routing
//! (a misrouted access changes the row set); an independent per-bucket walk validates
//! the gathered fallback records. Skips cleanly without a CUDA backend.

use crate::tables::memw_register;
use crate::tables::trace_builder::{
    MemwBuckets, PC_WORD_ADDR, RegAccess, RegisterState, walk_register_accesses,
};

const NBINS: u32 = 512;

/// Device fallback-record layout: `[reg_addr, ts, value, old_value, old_ts, is_read]`.
type FbRecord = [u64; 6];

/// Independent sequential oracle for the fallback subset: walk every access
/// (timeline included), and for each emitting, out-of-range access emit its device
/// record in emit order. Mirrors the routing predicate the device applies.
fn expected_fallbacks(accesses: &[RegAccess], init_value: &[u64], init_ts: u64) -> Vec<FbRecord> {
    let mut cell: Vec<(u64, u64)> = init_value.iter().map(|&v| (v, init_ts)).collect();
    let mut out = Vec::new();
    for a in accesses {
        let b = a.reg_addr as usize;
        let (ov, ot) = cell[b];
        cell[b] = (a.value, a.timestamp);
        if !a.emits_row {
            continue;
        }
        let ts_lo = a.timestamp & 0xFFFF_FFFF;
        let ot_lo = ot & 0xFFFF_FFFF;
        let in_range =
            (a.timestamp >> 32) == (ot >> 32) && ts_lo > ot_lo && (ts_lo - ot_lo) <= 0x10000;
        if !in_range {
            out.push([
                a.reg_addr,
                a.timestamp,
                a.value,
                ov,
                ot,
                u64::from(a.is_read),
            ]);
        }
    }
    out
}

#[test]
fn gpu_walk_route_matches_cpu_memw_register() {
    if math_cuda::device::backend().is_err() {
        eprintln!("skipping gpu_walk_route_matches_cpu_memw_register: no CUDA backend");
        return;
    }
    let entry_point = 0x1000u64;
    let init = RegisterState::new(entry_point);

    // Build a register-only access stream (no memory ops → aligned/general hold only
    // register fallbacks). Mostly in-range small deltas, interleaved timeline-only PC
    // writes and PC reads, then two engineered fallbacks. > SCAN_TARGET_EPB accesses
    // so the two-level device scan spans multiple blocks.
    let mut accesses: Vec<RegAccess> = Vec::new();
    let mut ts = 10u64;
    let push = |accesses: &mut Vec<RegAccess>, reg_addr, timestamp, value, is_read, emits_row| {
        accesses.push(RegAccess {
            reg_addr,
            timestamp,
            value,
            is_read,
            emits_row,
        });
    };
    for i in 0..20_000u64 {
        let reg = 1 + (i % 20); // regs x1..x20 (skip x0)
        push(
            &mut accesses,
            2 * reg,
            ts,
            i.wrapping_mul(0x9E37_79B9).wrapping_add(1),
            i % 3 == 0,
            true,
        );
        ts += 5;
        // Implicit PC write (advances the PC timeline, emits no row).
        if i % 4 == 0 {
            push(
                &mut accesses,
                PC_WORD_ADDR,
                ts,
                entry_point + i,
                false,
                false,
            );
            ts += 1;
        }
        // PC read (emits a MEMW_R row).
        if i % 7 == 0 {
            push(&mut accesses, PC_WORD_ADDR, ts, entry_point + i, true, true);
            ts += 1;
        }
    }
    // Fallback 1: large low-word delta (> 2^16) at the same bucket.
    push(&mut accesses, 2 * 5, ts + 200_000, 0xDEAD, false, true);
    ts += 200_001;
    // Fallback 2: high-word mismatch (crosses the 2^32 boundary), placed last so no
    // other bucket is dragged across the boundary.
    push(&mut accesses, 2 * 6, (1u64 << 32) + ts, 0xBEEF, true, true);

    // Per-bucket seed: value from the initial register state, timestamp 1.
    let mut init_value = vec![0u64; NBINS as usize];
    for r in 0..32u8 {
        init_value[2 * r as usize] = init.read(r).0;
    }
    init_value[PC_WORD_ADDR as usize] = entry_point;
    let init_ts = 1u64;

    // ---- CPU reference (the real prover path) ----
    let mut bucket = MemwBuckets::with_register_capacity(accesses.len());
    walk_register_accesses(&accesses, &init, &mut bucket);
    let rows = bucket.register_rows;

    let cpu_table = memw_register::generate_memw_register_trace_from_rows(&rows);
    let (cpu_fe, w) = cpu_table.main_data_row_major();
    assert_eq!(w, math_cuda::trace_cpu::MEMW_REGISTER_NCOLS);
    let cpu_u64: Vec<u64> = cpu_fe
        .iter()
        .map(|e| unsafe { *(e.value() as *const u64) })
        .collect();
    let max_rows = cpu_u64.len() / w;

    // IS_HALFWORD delta multiplicities the MEMW_R rows send (one +1 per row).
    let mut cpu_is_half = vec![0u64; 1 << 16];
    for r in &rows {
        let (_ra, t, _v, _ir, _ov, ot) = r.fill_soa();
        let d = ((t & 0xFFFF_FFFF) as u32)
            .wrapping_sub((ot & 0xFFFF_FFFF) as u32)
            .wrapping_sub(1) as u16;
        cpu_is_half[d as usize] += 1;
    }

    // ---- Device path ----
    let reg_addr: Vec<u32> = accesses.iter().map(|a| a.reg_addr as u32).collect();
    let ts_v: Vec<u64> = accesses.iter().map(|a| a.timestamp).collect();
    let value_v: Vec<u64> = accesses.iter().map(|a| a.value).collect();
    let is_read_v: Vec<u8> = accesses.iter().map(|a| u8::from(a.is_read)).collect();
    let emits_v: Vec<u8> = accesses.iter().map(|a| u8::from(a.emits_row)).collect();

    let (gpu_buf, gpu_rows, gpu_is_half, gpu_fb) =
        math_cuda::trace_walk::gpu_walk_route_memw_register_host(
            &reg_addr,
            &ts_v,
            &value_v,
            &is_read_v,
            &emits_v,
            &init_value,
            init_ts,
            NBINS,
            max_rows,
        )
        .expect("device walk+route+build");

    // ---- Assertions ----
    assert_eq!(gpu_rows, rows.len(), "MEMW_R row count mismatch");
    assert_eq!(
        gpu_buf, cpu_u64,
        "device MEMW_R matrix must be byte-identical to the CPU fill"
    );
    assert_eq!(
        gpu_is_half, cpu_is_half,
        "device IS_HALFWORD delta histogram must match the CPU multiplicities"
    );

    let emitting = accesses.iter().filter(|a| a.emits_row).count();
    assert_eq!(
        gpu_fb.len(),
        emitting - rows.len(),
        "fallback count must be (emitting rows) - (MEMW_R rows)"
    );
    assert!(
        gpu_fb.len() >= 2,
        "the two engineered fallbacks must route out"
    );
    assert_eq!(
        gpu_fb,
        expected_fallbacks(&accesses, &init_value, init_ts),
        "gathered fallback records must match the sequential oracle"
    );
}
