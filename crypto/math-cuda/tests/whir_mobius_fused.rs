//! The WHIR encoding with its low Möbius levels folded into the NTT's first
//! pass (`LAMBDA_VM_WHIR_MOBIUS_FUSED=1`: the strided levels first, the first
//! of them gathering the stack, then the low `12 − log_blowup` levels on the
//! coefficients the NTT's first pass stages) encodes today's codeword: the same
//! field values — word for word once canonicalised, since the subtractions run
//! in another order — and the same tree root.
//!
//! The fused and today's encodings run in one process over stacks of raw words
//! with edge values planted, from every source a commit takes, under both NTT
//! plans. Needs a GPU (the box):
//!
//!   cargo test -p math-cuda --release --test whir_mobius_fused -- --test-threads=1
//!   cargo test -p math-cuda --release --test whir_mobius_fused -- --ignored --nocapture --test-threads=1

use std::time::Instant;

use math_cuda::columns::DeviceColumns;
use math_cuda::lde_cm::{wide_codewords_encoded, with_ntt_wide};
use math_cuda::whir::{
    commit_codeword, encode_codeword_parts, encode_codeword_resident, encode_codeword_slices,
    mobius_fused_encodes, mobius_wide_gathers, with_mobius_fused, with_mobius_wide,
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

/// The field values of raw words, in place: every word in `[0, p)`.
fn canonical(mut v: Vec<u64>) -> Vec<u64> {
    for x in &mut v {
        if *x >= P {
            *x -= P;
        }
    }
    v
}

/// Whether the fused encoding applies to a `2^l` stack at spacing `2^lb`: the
/// NTT's first pass holds `min(12, l + lb) − lb` low levels, and at least one
/// strided level must be left (`whir::mobius_fused_levels`).
fn fuses(l: usize, lb: usize) -> bool {
    (l + lb).min(12) - lb < l
}

/// A stack of `2^l` words as parts `(column, offset)`: a half, an eighth, a
/// ragged run at an odd offset, a 5-word stub near the end, gaps between them.
fn layout(l: usize) -> Vec<(Vec<u64>, usize)> {
    let n = 1usize << l;
    let ragged = 1000.min(n / 4 - 32);
    let parts = vec![
        (words(21, n / 2), 0),
        (words(22, n / 8), n / 2),
        (words(23, ragged), n / 2 + n / 8 + 17),
        (words(24, 5), n - 9),
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

/// Run `f` under one arm: the fused encoding on or off, the wide NTT on or
/// off, the wide Möbius transform off. Returns its result and how many
/// encodings fused and gathered.
fn arm<R>(fused: bool, ntt: bool, f: impl FnOnce() -> R) -> (R, u64, u64) {
    with_mobius_wide(false, || {
        with_mobius_fused(fused, || {
            with_ntt_wide(ntt, || {
                let (f0, g0) = (mobius_fused_encodes(), mobius_wide_gathers());
                let r = f();
                (r, mobius_fused_encodes() - f0, mobius_wide_gathers() - g0)
            })
        })
    })
}

/// ★ Every source a commit takes, at stacks of 2^11 (the first pass holds them
/// whole: nothing fuses) to 2^22, under both NTT plans; spacings 4 and 16 on a
/// host source; and a committed stack's root at 2^20 and 2^28: the fused
/// encoding's codeword is today's in field values, it fused exactly where it
/// applies, and the device sources gathered. Every comparison prints its
/// `WHIR MOBIUS FUSED … PARITY` line before it asserts, so the box read-out
/// counts the compares that ran.
#[test]
fn the_fused_mobius_encodes_todays_codeword() {
    let be = math_cuda::device::backend().expect("a GPU");
    for l in [11usize, 12, 13, 19, 20, 22] {
        let parts = layout(l);
        let host: Vec<(&[u64], usize)> = parts.iter().map(|(c, o)| (c.as_slice(), *o)).collect();
        let (today, ..) = arm(false, false, || {
            canonical(read(
                "today's host encode",
                encode_codeword_parts(&host, l, 1, false),
            ))
        });
        let cols: Vec<&[u64]> = parts.iter().map(|(c, _)| c.as_slice()).collect();
        let store = DeviceColumns::upload(&cols).expect("room for the store");
        let at: Vec<(usize, usize)> = parts
            .iter()
            .enumerate()
            .map(|(k, (_, o))| (k, *o))
            .collect();
        let stream = be.next_stream();
        let mut flat = Vec::new();
        let mut slices = Vec::new();
        for (c, o) in parts.iter().rev() {
            slices.push((flat.len(), c.len(), *o));
            flat.extend_from_slice(c);
        }
        let buffer = stream.clone_htod(&flat).expect("the slices' buffer");
        stream.synchronize().expect("the slices' upload");
        let applies = fuses(l, 1);
        for ntt in [false, true] {
            let plan = if ntt { "wide" } else { "today" };
            for source in ["host", "resident", "slices"] {
                let (cw, fused, gathered) = arm(true, ntt, || {
                    let what = format!("the fused {source} encode");
                    canonical(match source {
                        "host" => read(&what, encode_codeword_parts(&host, l, 1, false)),
                        "resident" => {
                            read(&what, encode_codeword_resident(&store, &at, l, 1, false))
                        }
                        _ => read(&what, encode_codeword_slices(&buffer, &slices, l, 1, false)),
                    })
                });
                let equal = cw == today;
                let want_gather = u64::from(applies && source != "host");
                println!(
                    "WHIR MOBIUS FUSED SMALL PARITY: 2^{l} · {source} · ntt {plan} · equal={equal} applies={applies} fused={fused} gathered={gathered} want_gathered={want_gather}"
                );
                assert_eq!(
                    fused,
                    u64::from(applies),
                    "2^{l} {source}: fused where it does not apply, or not where it does"
                );
                assert_eq!(
                    gathered, want_gather,
                    "2^{l} {source}: the gather count is not the source's"
                );
                assert!(
                    equal,
                    "2^{l} {source} ntt {plan}: the fused codeword differs"
                );
            }
        }
    }

    // Other spacings: the first pass then holds 12 − lb levels.
    for (l, lb) in [(16usize, 2usize), (15, 4)] {
        let evals = words(30 + lb as u64, 1 << l);
        let (today, ..) = arm(false, false, || {
            canonical(read(
                "today's encode",
                encode_codeword_parts(&[(&evals, 0)], l, lb, false),
            ))
        });
        let (cw, fused, _) = arm(true, true, || {
            canonical(read(
                "the fused encode",
                encode_codeword_parts(&[(&evals, 0)], l, lb, false),
            ))
        });
        let equal = cw == today;
        println!(
            "WHIR MOBIUS FUSED SPACING PARITY: 2^{l} at spacing 2^{lb} · equal={equal} fused={fused}"
        );
        assert_eq!(fused, 1, "2^{l} at 2^{lb}: not fused");
        assert!(equal, "2^{l} at 2^{lb}: the fused codeword differs");
    }

    // A whole stack, committed: the same field values give the same root.
    let commit = |what: &str, fused: bool, evals: &[u64]| {
        arm(fused, true, || {
            commit_codeword(evals, 1, 6, true, math_cuda::DeviceHash::Poseidon1)
                .unwrap_or_else(|e| panic!("{what}: the commit failed: {e:?}"))
        })
    };
    let mut kept = None;
    for l in [19usize, 27] {
        let evals = words(80 + l as u64, 1 << l);
        let ((a, ra), ..) = commit("today's whole commit", false, &evals);
        let ((b, rb), fused, _) = commit("the fused whole commit", true, &evals);
        let (wa, wb) = (
            canonical(a.to_host().expect("read back")),
            canonical(b.to_host().expect("read back")),
        );
        println!(
            "WHIR MOBIUS FUSED ROOT PARITY: 2^{} · equal={} root_equal={} fused={fused}",
            l + 1,
            wa == wb,
            ra == rb
        );
        assert_eq!(fused, 1, "2^{l}: the commit did not fuse");
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
    moved[54321] ^= 1;
    let ((c, rc), ..) = commit("the fused moved commit", true, &moved);
    let differs = canonical(c.to_host().expect("read back")) != wa;
    println!(
        "WHIR MOBIUS FUSED MOVED WORD: differs={differs} root_differs={}",
        rc != ra
    );
    assert!(differs, "a moved word must move the codeword");
    assert_ne!(rc, ra, "a moved word must move the root");
}

/// The production shape's encode, timed: a resident stack of 2^27 (64 columns
/// of 2^21) to its 2^28 codeword under five arms — today (T), the wide Möbius
/// (M), the fused encoding (MD), the wide Möbius and NTT (B), the fused
/// encoding and the wide NTT (BD) — interleaved (the order rotates a rep),
/// median of 9 after 2 warm-ups. Each arm's codeword is first checked against
/// today's in field values. Prints one `WHIR MOBIUS FUSED BENCH` line.
#[test]
#[ignore = "box tier: a measurement"]
fn the_fused_mobius_encode_at_the_production_shape() {
    let l = 27;
    let label = "64 columns of 2^21";
    // (name, wide Möbius, fused, wide NTT)
    const ARMS: [(&str, bool, bool, bool); 5] = [
        ("today", false, false, false),
        ("mobius", true, false, false),
        ("fused", false, true, false),
        ("both", true, false, true),
        ("fused_ntt", false, true, true),
    ];
    let cols: Vec<Vec<u64>> = (0..64).map(|k| words(k + 1, 1 << 21)).collect();
    let refs: Vec<&[u64]> = cols.iter().map(|c| c.as_slice()).collect();
    let store = DeviceColumns::upload(&refs).expect("room for the store");
    let at: Vec<(usize, usize)> = (0..64).map(|k| (k, k << 21)).collect();
    let encode = |(_, mobius, fused, ntt): (&str, bool, bool, bool)| {
        with_mobius_wide(mobius, || {
            with_mobius_fused(fused, || {
                with_ntt_wide(ntt, || encode_codeword_resident(&store, &at, l, 1, false))
            })
        })
    };

    // The production shape's codewords in field values, before any timing.
    let today = canonical(read("today's encode", encode(ARMS[0])));
    for &a in &ARMS[1..] {
        let equal = canonical(read(a.0, encode(a))) == today;
        println!(
            "WHIR MOBIUS FUSED PARITY: 2^{l} · {label} · {} · equal={equal}",
            a.0
        );
        assert!(
            equal,
            "{label}: the {} arm's codeword differs at the production shape",
            a.0
        );
    }
    drop(today);

    let run = |a: (&str, bool, bool, bool)| {
        let (g0, f0, w0) = (
            mobius_wide_gathers(),
            mobius_fused_encodes(),
            wide_codewords_encoded(),
        );
        let t = Instant::now();
        let cw = encode(a).expect("a timed encode");
        cw.synchronize().unwrap();
        let ms = t.elapsed().as_secs_f64() * 1e3;
        drop(cw);
        // Each arm took exactly its own transforms: the times are the levers'.
        assert_eq!(
            mobius_wide_gathers() - g0,
            u64::from(a.1 || a.2),
            "{}: the gather count is not the arm's",
            a.0
        );
        assert_eq!(
            mobius_fused_encodes() - f0,
            u64::from(a.2),
            "{}: the fused count is not the arm's",
            a.0
        );
        assert_eq!(
            wide_codewords_encoded() - w0,
            u64::from(a.3),
            "{}: the wide-NTT count is not the arm's",
            a.0
        );
        ms
    };
    let mut times: [Vec<f64>; 5] = Default::default();
    for rep in 0..11 {
        for i in 0..ARMS.len() {
            let a = (rep + i) % ARMS.len();
            let ms = run(ARMS[a]);
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
    println!(
        "WHIR MOBIUS FUSED SPREAD: {}",
        ARMS.iter()
            .zip(&times)
            .map(|((arm, ..), t)| format!("{arm} {:.3}–{:.3} ms", t[0], t[t.len() - 1]))
            .collect::<Vec<_>>()
            .join(" · ")
    );
    let (today, mobius, fused, both, fused_ntt) = (med[0], med[1], med[2], med[3], med[4]);
    println!(
        "WHIR MOBIUS FUSED BENCH: resident 2^{l} → 2^{} · {label} · today {today:.3} ms · mobius {mobius:.3} ms · fused {fused:.3} ms · both {both:.3} ms · fused_ntt {fused_ntt:.3} ms · R_DM {:.3} · R_D {:.3}",
        l + 1,
        fused / mobius,
        fused_ntt / both
    );
}
