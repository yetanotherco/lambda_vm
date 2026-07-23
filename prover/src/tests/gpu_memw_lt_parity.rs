//! LT-resident-table (session 5) — device `memw→lt` pair generation parity.
//!
//! `math_cuda::trace_walk::gpu_memw_lt_pairs` produces the timestamp-ordering LT operands from the
//! packed MEMW_A / MEMW general rows on device. It must be a MULTISET match (the LT bus is
//! order-free) to the host `collect_lt_from_memw` / `collect_lt_from_memw_aligned`:
//!   aligned row → 1 LT (lhs=old_timestamp[0], rhs=timestamp)
//!   general row → `width` LTs (lhs=old_timestamp[i], rhs=timestamp), i in 0..width
//! Self-contained (builds the packed rows by hand per the `unpack_memw`/`unpack_memw_aligned`
//! layout); requires a GPU.
//!
//! `cargo test -p lambda-vm-prover --release --features cuda --lib gpu_memw_lt -- --ignored --nocapture`

const ALIGNED_STRIDE: usize = 12; // MEMW_ALIGNED_STRIDE
const GENERAL_STRIDE: usize = 19; // MEMW_STRIDE

fn mix(i: usize, s: u64) -> u64 {
    let mut x = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ s.wrapping_mul(0xD1B5_4A32_D192_ED03);
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 27;
    x ^ (x >> 31)
}

#[test]
#[ignore = "requires GPU; run --ignored --nocapture"]
fn gpu_memw_lt_pairs_matches_cpu() {
    if math_cuda::device::backend().is_err() {
        eprintln!("skipping gpu_memw_lt_pairs_matches_cpu: no CUDA backend");
        return;
    }
    let widths = [1u8, 2, 4, 8];
    let na = 50_000usize;
    let ng = 40_000usize;

    // Build packed rows + the expected (lhs, rhs) multiset from the same synthetic data.
    let mut pa = vec![0u64; na * ALIGNED_STRIDE];
    let mut pg = vec![0u64; ng * GENERAL_STRIDE];
    let mut expected: Vec<(u64, u64)> = Vec::new();

    for i in 0..na {
        let w = widths[i % 4];
        let ts = mix(i, 1) | 1; // nonzero-ish
        let old0 = mix(i, 2);
        let r = &mut pa[i * ALIGNED_STRIDE..(i + 1) * ALIGNED_STRIDE];
        r[0] = (w as u64) << 8; // flags: width @ bits 8-15
        r[2] = ts;
        r[3] = old0; // old_timestamp[0]
        expected.push((old0, ts)); // aligned → 1 LT
    }
    for i in 0..ng {
        let w = widths[i % 4];
        let ts = mix(i, 3);
        let r = &mut pg[i * GENERAL_STRIDE..(i + 1) * GENERAL_STRIDE];
        r[0] = (w as u64) << 8;
        r[2] = ts;
        for j in 0..w as usize {
            let old = mix(i, 10 + j as u64);
            r[11 + j] = old; // old_timestamp[j]
            expected.push((old, ts)); // general → width LTs
        }
    }

    let (lhs, rhs) =
        math_cuda::trace_walk::gpu_memw_lt_pairs(&pa, na, &pg, ng).expect("gpu_memw_lt_pairs");
    assert_eq!(lhs.len(), rhs.len(), "lhs/rhs length mismatch");
    assert_eq!(lhs.len(), expected.len(), "pair count mismatch");

    let mut got: Vec<(u64, u64)> = lhs.into_iter().zip(rhs).collect();
    got.sort_unstable();
    expected.sort_unstable();
    assert_eq!(got, expected, "device memw→lt pairs != CPU collect_lt_from_memw (multiset)");
    println!(
        "gpu_memw_lt_pairs OK: {na} aligned + {ng} general rows → {} LT pairs, multiset-identical to CPU",
        got.len()
    );
}
