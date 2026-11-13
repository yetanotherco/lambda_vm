//! The WHIR encoding's column engine at spacings 32 and 64 (blowups 32 and
//! 64, `lde_cm::MAX_SPREAD_LOG_BLOWUP`): where the transform's first pass
//! holds the spacing (`lde_cm::spread_supports`), the engine's codeword is the
//! legacy per-level encoding's (`lift_spread` + the tiled NTT, which shares no
//! pass code with it) in field values, in both plans, with and without the
//! fused low Möbius levels, and a committed stack's root is the legacy one's.
//! The engine ran where the rule admits the shape (its counter moved) and the
//! legacy path ran where it does not.
//!
//! Needs a GPU (the box):
//!
//!   cargo test -p math-cuda --release --test whir_spread_blowup64 -- --test-threads=1
//!   cargo test -p math-cuda --release --test whir_spread_blowup64 -- --ignored --nocapture --test-threads=1

use std::time::Instant;

use math_cuda::lde_cm::{codewords_encoded, spread_supports, with_engine, with_ntt_wide};
use math_cuda::whir::{
    commit_codeword, encode_codeword_parts, mobius_fused_encodes, with_mobius_fused,
    with_mobius_wide,
};

const P: u64 = 0xFFFF_FFFF_0000_0001;

/// Raw words with an edge value at every seventh slot.
fn words(seed: u64, n: usize) -> Vec<u64> {
    const EDGES: [u64; 8] = [0, 1, P - 1, P, P + 1, u64::MAX, 1 << 63, 0xFFFF_FFFF];
    let mut z = seed;
    (0..n)
        .map(|i| {
            z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut x = z;
            x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            if i % 7 == 0 {
                EDGES[(i / 7) % EDGES.len()]
            } else {
                x ^ (x >> 31)
            }
        })
        .collect()
}

/// The field values of raw words: every word in `[0, p)`.
fn canonical(mut v: Vec<u64>) -> Vec<u64> {
    for x in &mut v {
        if *x >= P {
            *x -= P;
        }
    }
    v
}

/// `evals` (one stack of `2^l`) encoded at spacing `2^lb`: the raw codeword and
/// how many codewords the column engine encoded.
fn encode(what: &str, evals: &[u64], l: usize, lb: usize) -> (Vec<u64>, u64) {
    let before = codewords_encoded();
    let raw = encode_codeword_parts(&[(evals, 0)], l, lb, false)
        .unwrap_or_else(|e| panic!("{what}: the encode failed: {e:?}"))
        .to_host()
        .unwrap_or_else(|e| panic!("{what}: the read-back failed: {e:?}"));
    (raw, codewords_encoded() - before)
}

/// ★ Spacings 32 and 64 at sizes from 2^19 to 2^30 (C2's encode: a 2^24
/// stack at blowup 64), today's and the wide plan, the fused Möbius levels on
/// and off: the engine's codeword equals the legacy one in field values, the
/// engine ran (the rule admits every one of these shapes), and the legacy arm
/// ran no engine pass. A short shape the rule declines (2^9 at spacing 64 in
/// today's plan, first pass [5, 4]) takes the legacy path. Every comparison
/// prints its `WHIR SPREAD64 PARITY` line before it asserts.
#[test]
fn the_engine_encodes_the_legacy_codeword_at_blowups_32_and_64() {
    // (log_evals, log_blowup)
    const SHAPES: [(usize, usize); 7] = [
        (14, 5),
        (14, 6),
        (17, 6),
        (19, 5),
        (20, 6),
        (22, 6),
        (24, 6),
    ];
    for (l, lb) in SHAPES {
        let log_n = l + lb;
        let evals = words(l as u64 * 8 + lb as u64, 1 << l);
        let (legacy, legacy_k) = with_engine(false, || encode("the legacy encode", &evals, l, lb));
        let legacy = canonical(legacy);
        assert_eq!(legacy_k, 0, "2^{log_n}: the legacy arm ran the engine");
        let arms: &[(&str, bool, bool)] = if log_n >= 30 {
            // The production shape: the default plan and Möbius setting only.
            &[("wide fused", true, true)]
        } else {
            &[
                ("today", false, false),
                ("today fused", false, true),
                ("wide", true, false),
                ("wide fused", true, true),
            ]
        };
        for &(arm, wide, fused) in arms {
            let ((raw, engine_k), fused_k) = with_ntt_wide(wide, || {
                with_mobius_wide(fused, || {
                    with_mobius_fused(fused, || {
                        assert!(
                            spread_supports(log_n as u32, lb as u32),
                            "2^{log_n} at 2^{lb}: the rule declines a shape the test expects it to take"
                        );
                        let m0 = mobius_fused_encodes();
                        let out = encode(arm, &evals, l, lb);
                        (out, mobius_fused_encodes() - m0)
                    })
                })
            });
            let equal = canonical(raw) == legacy;
            println!(
                "WHIR SPREAD64 PARITY: 2^{log_n} (2^{l} at spacing 2^{lb}) · {arm} · equal={equal} engine={engine_k} mobius_fused={fused_k}"
            );
            assert_eq!(
                engine_k, 1,
                "2^{log_n} {arm}: the column engine did not run"
            );
            assert!(
                equal,
                "2^{log_n} {arm}: the engine's codeword differs from the legacy encoding"
            );
        }
    }
    // A shape the rule declines stays on the legacy path, and its codeword is
    // the legacy one by construction.
    with_ntt_wide(false, || {
        assert!(!spread_supports(9, 6));
        let evals = words(7, 1 << 3);
        let (_, engine_k) = encode("a declined shape", &evals, 3, 6);
        println!("WHIR SPREAD64 DECLINED: 2^9 at spacing 2^6 · engine={engine_k}");
        assert_eq!(
            engine_k, 0,
            "2^9 at spacing 2^6 ran the engine past the rule"
        );
    });
}

/// A whole stack committed at blowup 64 (C2's round-0 commit, first fold 6):
/// the engine's root is the legacy encoding's, and one moved stack word moves
/// the codeword and the root (the comparison can fail).
#[test]
fn a_blowup_64_commit_has_the_legacy_root() {
    let commit = |what: &str, engine: bool, evals: &[u64]| {
        with_engine(engine, || {
            commit_codeword(evals, 6, 6, true, math_cuda::DeviceHash::Poseidon1)
                .unwrap_or_else(|e| panic!("{what}: the commit failed: {e:?}"))
        })
    };
    let l = 20;
    let evals = words(64, 1 << l);
    let (a, ra) = commit("the legacy commit", false, &evals);
    let k0 = codewords_encoded();
    let (b, rb) = commit("the engine commit", true, &evals);
    let engine_k = codewords_encoded() - k0;
    let (wa, wb) = (
        canonical(a.to_host().expect("read back")),
        canonical(b.to_host().expect("read back")),
    );
    println!(
        "WHIR SPREAD64 ROOT PARITY: 2^{} · equal={} root_equal={} engine={engine_k}",
        l + 6,
        wa == wb,
        ra == rb
    );
    assert_eq!(engine_k, 1, "the engine commit did not run the engine");
    assert_eq!(wa, wb, "the blowup-64 codeword differs");
    assert_eq!(ra, rb, "the blowup-64 root differs");
    let mut moved = evals.clone();
    moved[54321] ^= 1;
    let (c, rc) = commit("the moved commit", true, &moved);
    let differs = canonical(c.to_host().expect("read back")) != wb;
    println!(
        "WHIR SPREAD64 MOVED WORD: differs={differs} root_differs={}",
        rc != rb
    );
    assert!(differs, "a moved word must move the codeword");
    assert_ne!(rc, rb, "a moved word must move the root");
}

/// C2's encode, timed and its card memory read: a 2^24 stack to its 2^30
/// codeword at blowup 64 under the engine and under the legacy path, the
/// legacy path first cold (its 2^30 twiddle table built and uploaded, as C2
/// pays today) then warm, median of 5 after one warm-up, the codewords
/// checked equal first. Card memory: the device's free bytes before an encode
/// (the pool trimmed) against after it with the codeword alive — the pool's
/// high-water over the encode. Prints `WHIR SPREAD64 BENCH` and
/// `WHIR SPREAD64 MEMORY` lines.
#[test]
#[ignore = "box tier: a measurement"]
fn the_blowup_64_encode_at_c2s_shape() {
    let be = math_cuda::device::backend().expect("a card");
    let (l, lb) = (24usize, 6usize);
    let evals = words(2424, 1 << l);
    let timed = |engine: bool| {
        with_engine(engine, || {
            let t = Instant::now();
            let cw = encode_codeword_parts(&[(&evals, 0)], l, lb, false).expect("an encode");
            cw.synchronize().expect("synchronize");
            let ms = t.elapsed().as_secs_f64() * 1e3;
            drop(cw);
            ms
        })
    };
    let cold = timed(false);
    let engine_cw = canonical(encode("the engine encode", &evals, l, lb).0);
    let legacy_cw = canonical(with_engine(false, || encode("the legacy encode", &evals, l, lb)).0);
    let equal = engine_cw == legacy_cw;
    drop((engine_cw, legacy_cw));
    println!("WHIR SPREAD64 PARITY: 2^30 (2^24 at spacing 2^6) · bench shape · equal={equal}");
    assert!(equal, "the bench shape's codewords differ");
    let mut times: [Vec<f64>; 2] = Default::default();
    for rep in 0..6 {
        let order = if rep % 2 == 0 {
            [true, false]
        } else {
            [false, true]
        };
        for engine in order {
            let ms = timed(engine);
            if rep > 0 {
                times[usize::from(!engine)].push(ms);
            }
        }
    }
    let median = |v: &mut Vec<f64>| {
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    };
    let (engine_ms, legacy_ms) = (median(&mut times[0]), median(&mut times[1]));
    println!(
        "WHIR SPREAD64 BENCH: 2^30 (2^24 at spacing 2^6) · engine {engine_ms:.1} ms · legacy warm {legacy_ms:.1} ms · legacy cold (twiddle table built) {cold:.1} ms"
    );
    for (arm, engine) in [("engine", true), ("legacy", false)] {
        be.trim_mempool_to(0);
        let (free0, total) = be.device_mem_info().expect("mem info");
        let cw = with_engine(engine, || {
            let cw = encode_codeword_parts(&[(&evals, 0)], l, lb, false).expect("an encode");
            cw.synchronize().expect("synchronize");
            cw
        });
        let (free1, _) = be.device_mem_info().expect("mem info");
        drop(cw);
        println!(
            "WHIR SPREAD64 MEMORY: {arm} · used over the encode {:.2} GiB (free {:.2} → {:.2} of {:.2} GiB; the codeword is {:.2} GiB)",
            free0.saturating_sub(free1) as f64 / f64::from(1u32 << 30),
            free0 as f64 / f64::from(1u32 << 30),
            free1 as f64 / f64::from(1u32 << 30),
            total as f64 / f64::from(1u32 << 30),
            ((1u64 << (l + lb)) * 8) as f64 / f64::from(1u32 << 30)
        );
    }
}
