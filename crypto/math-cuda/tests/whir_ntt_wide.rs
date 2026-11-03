//! The WHIR encoding's wide NTT (`LAMBDA_VM_WHIR_NTT_WIDE`: passes of up to
//! 10 levels, three instead of four at the production 2^28) encodes the
//! 8-level plan's codeword: the same field values — word for word once canonicalised, since a
//! different pass split multiplies by different twist factors and may leave a
//! value in its other representation — and the same tree root.
//!
//! Both plans run in one process (`lde_cm::with_ntt_wide`) over stacks of raw
//! words with edge values planted, at every codeword size where the plans
//! differ from 2^9 to 2^29 and at two where they agree; the legacy per-level
//! encoding (`lde_cm::with_engine(false, ..)`), which shares no pass code with
//! either plan, is a second reference. The encoding's other two knobs (the
//! wide Möbius transform, the fused low levels), on by default, are held off
//! ([`alone`]) except where a comparison sets them. Needs a GPU (the box):
//!
//!   cargo test -p math-cuda --release --test whir_ntt_wide -- --test-threads=1
//!   cargo test -p math-cuda --release --test whir_ntt_wide -- --ignored --nocapture --test-threads=1

use std::time::Instant;

use math_cuda::columns::DeviceColumns;
use math_cuda::lde_cm::{wide_codewords_encoded, with_engine, with_ntt_wide};
use math_cuda::whir::{
    commit_codeword, encode_codeword_parts, encode_codeword_resident, mobius_wide_gathers,
    with_mobius_fused, with_mobius_wide,
};

/// Runs `f` with the wide Möbius transform and the fused low levels off on
/// this thread, so the plans' arms differ in the NTT only; a comparison that
/// sets the Möbius transform itself overrides it inside.
fn alone<R>(f: impl FnOnce() -> R) -> R {
    with_mobius_fused(false, || with_mobius_wide(false, f))
}

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

/// The field values of raw words, in place: every word in `[0, p)`.
fn canonical(mut v: Vec<u64>) -> Vec<u64> {
    for x in &mut v {
        if *x >= P {
            *x -= P;
        }
    }
    v
}

/// Whether the wide plan of a `2^log_n` transform has a pass of more than 8
/// levels (`lde_cm::plan_wide`: `ceil(l / 10)` passes, as even as possible).
/// Where it does not, it is today's plan.
fn wide_kernels_run(log_n: usize) -> bool {
    let p = log_n.div_ceil(10);
    log_n.div_ceil(p) > 8
}

/// A stack of `2^l` words as parts `(column, offset)`: a half, an eighth, a
/// ragged run at an odd offset, a 5-word stub near the end, gaps between them.
fn layout(l: usize) -> Vec<(Vec<u64>, usize)> {
    let n = 1usize << l;
    let ragged = 1000.min(n / 4 - 32);
    let parts = vec![
        (words(11, n / 2), 0),
        (words(12, n / 8), n / 2),
        (words(13, ragged), n / 2 + n / 8 + 17),
        (words(14, 5), n - 9),
    ];
    let mut end = 0;
    for (k, (c, o)) in parts.iter().enumerate() {
        assert!(
            *o >= end && o + c.len() <= n,
            "2^{l}: the test's part {k} ({} words at {o}) overlaps or overruns the stack",
            c.len()
        );
        end = o + c.len();
    }
    parts
}

/// An encode's codeword read back, a named failure if either step fails.
fn read(what: &str, cw: math_cuda::Result<math_cuda::whir::DeviceCodeword>) -> Vec<u64> {
    cw.unwrap_or_else(|e| panic!("{what}: the encode failed: {e:?}"))
        .to_host()
        .unwrap_or_else(|e| panic!("{what}: the read-back failed: {e:?}"))
}

/// Encode `evals` (one whole stack of `2^l`) at spacing `2^lb` under one plan:
/// the raw codeword, and how many encodings ran a wide pass.
fn encode(what: &str, wide: bool, evals: &[u64], l: usize, lb: usize) -> (Vec<u64>, u64) {
    with_ntt_wide(wide, || {
        let before = wide_codewords_encoded();
        let raw = read(what, encode_codeword_parts(&[(evals, 0)], l, lb, false));
        (raw, wide_codewords_encoded() - before)
    })
}

/// ★ Every codeword size where the plans differ from 2^9 to 2^29 (one or two
/// wide passes, a short strided one, the production [10, 9, 9] and [10, 10,
/// 9]), spacings 2, 4 and 16, and two sizes where they agree: the wide plan's
/// codeword is today's in field values, its wide kernels ran exactly where the
/// plans differ, and a committed stack's root is today's. Every comparison
/// prints its `WHIR NTT WIDE … PARITY` line before it asserts, so the box
/// read-out counts the compares that ran.
#[test]
fn the_wide_ntt_encodes_todays_codeword() {
    alone(|| {
        // (log_evals, log_blowup): the codeword is 2^(log_evals + log_blowup).
        const SHAPES: [(usize, usize); 15] = [
            (8, 1),
            (9, 1),
            (15, 1),
            (16, 1),
            (17, 1),
            (18, 1),
            (19, 1),
            (16, 2),
            (15, 4),
            (23, 1),
            (24, 1),
            (25, 1),
            (26, 1),
            (27, 1),
            (27, 2),
        ];
        for (l, lb) in SHAPES {
            let log_n = l + lb;
            let evals = words(l as u64 * 8 + lb as u64, 1 << l);
            let (today, today_k) = encode("today's encode", false, &evals, l, lb);
            let (wide, wide_k) = encode("the wide encode", true, &evals, l, lb);
            let legacy = with_engine(false, || {
                read(
                    "the legacy encode",
                    encode_codeword_parts(&[(&evals, 0)], l, lb, false),
                )
            });
            let expect = u64::from(wide_kernels_run(log_n));
            let raw_equal = today == wide;
            let (today, wide) = (canonical(today), canonical(wide));
            let equal = today == wide;
            let legacy_equal = canonical(legacy) == wide;
            println!(
                "WHIR NTT WIDE SMALL PARITY: 2^{log_n} (2^{l} at spacing 2^{lb}) · equal={equal} legacy_equal={legacy_equal} raw_equal={raw_equal} wide_kernels={wide_k} expected={expect} today_kernels={today_k}"
            );
            assert_eq!(today_k, 0, "2^{log_n}: today's plan ran a wide pass");
            assert_eq!(
                wide_k, expect,
                "2^{log_n}: the wide kernels did not run where the plans differ"
            );
            assert!(
                equal,
                "2^{log_n}: the wide plan's codeword differs in field values"
            );
            assert!(
                legacy_equal,
                "2^{log_n}: the wide plan's codeword differs from the legacy encoding"
            );
        }

        // A resident stack under both wide transforms (the Möbius gathered, the
        // NTT wide) against today's: the two levers together.
        for l in [19usize, 24] {
            let parts = layout(l);
            let cols: Vec<&[u64]> = parts.iter().map(|(c, _)| c.as_slice()).collect();
            let store = DeviceColumns::upload(&cols).expect("room for the store");
            let at: Vec<(usize, usize)> = parts
                .iter()
                .enumerate()
                .map(|(k, (_, o))| (k, *o))
                .collect();
            let today = with_mobius_wide(false, || {
                with_ntt_wide(false, || {
                    read(
                        "today's resident encode",
                        encode_codeword_resident(&store, &at, l, 1, false),
                    )
                })
            });
            let (g0, w0) = (mobius_wide_gathers(), wide_codewords_encoded());
            let both = with_mobius_wide(true, || {
                with_ntt_wide(true, || {
                    read(
                        "the wide-wide resident encode",
                        encode_codeword_resident(&store, &at, l, 1, false),
                    )
                })
            });
            let (gathered, wide_k) = (mobius_wide_gathers() - g0, wide_codewords_encoded() - w0);
            let equal = canonical(today) == canonical(both);
            println!(
                "WHIR NTT WIDE BOTH PARITY: 2^{} · resident · equal={equal} gathered={gathered} wide_kernels={wide_k}",
                l + 1
            );
            assert_eq!(gathered, 1, "2^{l}: the resident source was not gathered");
            assert_eq!(wide_k, 1, "2^{l}: the wide kernels did not run");
            assert!(equal, "2^{l}: both wide transforms together differ");
        }

        // A whole stack, committed: the same field values give the same root.
        let commit = |what: &str, wide: bool, evals: &[u64]| {
            with_ntt_wide(wide, || {
                commit_codeword(evals, 1, 6, true, math_cuda::DeviceHash::Poseidon1)
                    .unwrap_or_else(|e| panic!("{what}: the commit failed: {e:?}"))
            })
        };
        let mut kept = None;
        for l in [19usize, 27] {
            let evals = words(90 + l as u64, 1 << l);
            let (a, ra) = commit("today's whole commit", false, &evals);
            let w0 = wide_codewords_encoded();
            let (b, rb) = commit("the wide whole commit", true, &evals);
            let wide_k = wide_codewords_encoded() - w0;
            let (wa, wb) = (
                canonical(a.to_host().expect("read back")),
                canonical(b.to_host().expect("read back")),
            );
            println!(
                "WHIR NTT WIDE ROOT PARITY: 2^{} · equal={} root_equal={} wide_kernels={wide_k}",
                l + 1,
                wa == wb,
                ra == rb
            );
            assert_eq!(wide_k, 1, "2^{l}: the wide commit ran no wide pass");
            assert_eq!(wa, wb, "2^{l}: a whole stack's codeword differs");
            assert_eq!(ra, rb, "2^{l}: a whole stack's root differs");
            if l == 19 {
                kept = Some((evals, wa, ra));
            }
        }

        // The comparison can fail: one stack word moved moves the codeword and
        // the root.
        let (evals, wa, ra) = kept.expect("the 2^19 commit");
        let mut moved = evals.clone();
        moved[12345] ^= 1;
        let (c, rc) = commit("the wide moved commit", true, &moved);
        let differs = canonical(c.to_host().expect("read back")) != wa;
        println!(
            "WHIR NTT WIDE MOVED WORD: differs={differs} root_differs={}",
            rc != ra
        );
        assert!(differs, "a moved word must move the codeword");
        assert_ne!(rc, ra, "a moved word must move the root");
    })
}

/// The production shape's encode, timed: a resident stack of 2^27 (64 columns
/// of 2^21, as a group's tables stack) to its 2^28 codeword under four arms —
/// today, the wide Möbius alone, the wide NTT alone, both — interleaved (the
/// order rotates a rep), median of 9 after 2 warm-ups. Each arm's codeword is
/// first checked against today's in field values. Prints one `WHIR NTT WIDE
/// BENCH` line.
#[test]
#[ignore = "box tier: a measurement"]
fn the_wide_ntt_encode_at_the_production_shape() {
    alone(|| {
        let l = 27;
        let label = "64 columns of 2^21";
        const ARMS: [(&str, bool, bool); 4] = [
            ("today", false, false),
            ("mobius", true, false),
            ("ntt", false, true),
            ("both", true, true),
        ];
        let cols: Vec<Vec<u64>> = (0..64).map(|k| words(k + 1, 1 << 21)).collect();
        let refs: Vec<&[u64]> = cols.iter().map(|c| c.as_slice()).collect();
        let store = DeviceColumns::upload(&refs).expect("room for the store");
        let at: Vec<(usize, usize)> = (0..64).map(|k| (k, k << 21)).collect();
        let encode = |mobius: bool, ntt: bool| {
            with_mobius_wide(mobius, || {
                with_ntt_wide(ntt, || encode_codeword_resident(&store, &at, l, 1, false))
            })
        };

        // The production shape's codewords in field values, before any timing.
        let today = canonical(read("today's encode", encode(false, false)));
        for &(arm, mobius, ntt) in &ARMS[1..] {
            let equal = canonical(read(arm, encode(mobius, ntt))) == today;
            println!("WHIR NTT WIDE PARITY: 2^{l} · {label} · {arm} · equal={equal}");
            assert!(
                equal,
                "{label}: the {arm} arm's codeword differs at the production shape"
            );
        }
        drop(today);

        let run = |mobius: bool, ntt: bool| {
            let (g0, w0) = (mobius_wide_gathers(), wide_codewords_encoded());
            let t = Instant::now();
            let cw = encode(mobius, ntt).expect("a timed encode");
            cw.synchronize().unwrap();
            let ms = t.elapsed().as_secs_f64() * 1e3;
            drop(cw);
            // Each arm took exactly its own transforms: the times are the levers'.
            assert_eq!(
                mobius_wide_gathers() - g0,
                u64::from(mobius),
                "the gather count is not the arm's"
            );
            assert_eq!(
                wide_codewords_encoded() - w0,
                u64::from(ntt),
                "the wide-NTT count is not the arm's"
            );
            ms
        };
        let mut times: [Vec<f64>; 4] = Default::default();
        for rep in 0..11 {
            for i in 0..ARMS.len() {
                let a = (rep + i) % ARMS.len();
                let ms = run(ARMS[a].1, ARMS[a].2);
                if rep >= 2 {
                    times[a].push(ms);
                }
            }
        }
        let med: Vec<f64> = times
            .iter_mut()
            .map(|t| {
                t.sort_by(f64::total_cmp);
                t[t.len() / 2]
            })
            .collect();
        let (today, mobius, ntt, both) = (med[0], med[1], med[2], med[3]);
        println!(
            "WHIR NTT WIDE SPREAD: {}",
            ARMS.iter()
                .zip(&times)
                .map(|((arm, ..), t)| format!("{arm} {:.3}–{:.3} ms", t[0], t[t.len() - 1]))
                .collect::<Vec<_>>()
                .join(" · ")
        );
        println!(
            "WHIR NTT WIDE BENCH: resident 2^{l} → 2^{} · {label} · today {today:.3} ms · mobius {mobius:.3} ms · ntt {ntt:.3} ms · both {both:.3} ms · R_N {:.3} · R_BM {:.3} · saved {:.3} ms alone · {:.3} ms on the wide Möbius",
            l + 1,
            ntt / today,
            both / mobius,
            today - ntt,
            mobius - both
        );
    })
}
