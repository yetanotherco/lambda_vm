//! The WHIR encoding's wide Möbius transform
//! (the default: three passes instead of five, and the
//! stack gathered from its parts instead of copied into a zeroed buffer)
//! encodes the five-pass transform's codeword, word for word, from every
//! source a commit takes. The encoding's other two knobs (the fused low
//! levels, the wide NTT), on by default, are held off ([`alone`]), so the two
//! arms differ in this transform only.
//!
//! Both transforms run in one process (`whir::with_mobius_wide`) over stacks of
//! raw words — edge values planted, parts of mixed sizes at unaligned offsets
//! with gaps of padding — and the codewords are compared whole. Needs a GPU
//! (the box):
//!
//!   cargo test -p math-cuda --release --test whir_mobius_wide
//!   cargo test -p math-cuda --release --test whir_mobius_wide -- --ignored --nocapture

use std::time::Instant;

use math_cuda::columns::DeviceColumns;
use math_cuda::lde_cm::with_ntt_wide;
use math_cuda::whir::{
    commit_codeword, encode_codeword_parts, encode_codeword_resident, encode_codeword_slices,
    mobius_wide_gathers, with_mobius_fused, with_mobius_wide,
};

/// Runs `f` with the fused low levels and the wide NTT off on this thread, so
/// each arm's encoding differs from the other's in the Möbius transform only.
fn alone<R>(f: impl FnOnce() -> R) -> R {
    with_mobius_fused(false, || with_ntt_wide(false, f))
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

/// A stack of `2^l` words as parts `(column, offset)`: a half, an eighth,
/// a ragged run (1,000 words, or less on a small stack) at an odd offset, a
/// 5-word stub near the end, and gaps of padding between them. Checked to fit:
/// a layout that overruns is the test's own fault, and it must say so rather
/// than panic inside a copy.
fn layout(l: usize) -> Vec<(Vec<u64>, usize)> {
    let n = 1usize << l;
    let ragged = 1000.min(n / 4 - 32);
    let parts = vec![
        (words(1, n / 2), 0),
        (words(2, n / 8), n / 2),
        (words(3, ragged), n / 2 + n / 8 + 17),
        (words(4, 5), n - 9),
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

/// ★ Every source a commit takes, at stacks of 2^11 (one contiguous pass) to
/// 2^22 (both strided passes, one short): the wide transform's codeword is
/// today's, word for word, and the device sources took the gather. Every
/// comparison prints its `WHIR MOBIUS WIDE … PARITY` line before it asserts,
/// so the box read-out counts the compares that ran.
#[test]
fn the_wide_mobius_encodes_todays_codeword() {
    alone(|| {
        for l in [11usize, 12, 13, 19, 20, 22] {
            let parts = layout(l);
            // A host source: the wide transform, no gather.
            let host: Vec<(&[u64], usize)> =
                parts.iter().map(|(c, o)| (c.as_slice(), *o)).collect();
            let today = with_mobius_wide(false, || {
                read(
                    "today's host encode",
                    encode_codeword_parts(&host, l, 1, false),
                )
            });
            let wide = with_mobius_wide(true, || {
                read(
                    "the wide host encode",
                    encode_codeword_parts(&host, l, 1, false),
                )
            });
            println!(
                "WHIR MOBIUS WIDE SMALL PARITY: 2^{l} · host · equal={}",
                today == wide
            );
            assert_eq!(
                today, wide,
                "2^{l}: host parts differ under the wide transform"
            );

            // A resident store: gathered.
            let cols: Vec<&[u64]> = parts.iter().map(|(c, _)| c.as_slice()).collect();
            let store = DeviceColumns::upload(&cols).expect("room for the store");
            let at: Vec<(usize, usize)> = parts
                .iter()
                .enumerate()
                .map(|(k, (_, o))| (k, *o))
                .collect();
            let before = mobius_wide_gathers();
            let wide = with_mobius_wide(true, || {
                read(
                    "the wide resident encode",
                    encode_codeword_resident(&store, &at, l, 1, false),
                )
            });
            let gathered = mobius_wide_gathers() == before + 1;
            let today_r = with_mobius_wide(false, || {
                read(
                    "today's resident encode",
                    encode_codeword_resident(&store, &at, l, 1, false),
                )
            });
            println!(
                "WHIR MOBIUS WIDE SMALL PARITY: 2^{l} · resident · equal={} gathered={gathered} today_equal={}",
                wide == today,
                today_r == today
            );
            assert!(gathered, "2^{l}: the resident source was not gathered");
            assert_eq!(
                today_r, today,
                "2^{l}: the resident and host sources differ today"
            );
            assert_eq!(wide, today, "2^{l}: the gathered resident source differs");

            // A buffer's slices, out of order in the list: gathered.
            let be = math_cuda::device::backend().expect("a GPU");
            let stream = be.next_stream();
            let mut flat = Vec::new();
            let mut slices = Vec::new();
            for (c, o) in parts.iter().rev() {
                slices.push((flat.len(), c.len(), *o));
                flat.extend_from_slice(c);
            }
            let buffer = stream.clone_htod(&flat).expect("the slices' buffer");
            stream.synchronize().expect("the slices' upload");
            let before = mobius_wide_gathers();
            let wide = with_mobius_wide(true, || {
                read(
                    "the wide sliced encode",
                    encode_codeword_slices(&buffer, &slices, l, 1, false),
                )
            });
            let gathered = mobius_wide_gathers() == before + 1;
            println!(
                "WHIR MOBIUS WIDE SMALL PARITY: 2^{l} · slices · equal={} gathered={gathered}",
                wide == today
            );
            assert!(gathered, "2^{l}: the sliced source was not gathered");
            assert_eq!(wide, today, "2^{l}: the gathered slices differ");
        }

        // A whole stack, committed: the same codeword gives the same root.
        let l = 16;
        let evals = words(9, 1 << l);
        let commit = |what: &str, wide: bool, evals: &[u64]| {
            with_mobius_wide(wide, || {
                commit_codeword(evals, 1, 6, true, math_cuda::DeviceHash::Poseidon1)
                    .unwrap_or_else(|e| panic!("{what}: the commit failed: {e:?}"))
            })
        };
        let (a, ra) = commit("today's whole commit", false, &evals);
        let (b, rb) = commit("the wide whole commit", true, &evals);
        let (wa, wb) = (
            a.to_host().expect("read back"),
            b.to_host().expect("read back"),
        );
        println!(
            "WHIR MOBIUS WIDE WHOLE PARITY: 2^{l} · equal={} root_equal={}",
            wa == wb,
            ra == rb
        );
        assert_eq!(wa, wb, "a whole stack's codeword differs");
        assert_eq!(ra, rb, "a whole stack's root differs");

        // The comparison can fail: one stack word moved moves the codeword.
        let mut moved = evals.clone();
        moved[12345] ^= 1;
        let (c, _) = commit("the wide moved commit", true, &moved);
        let differs = c.to_host().expect("read back") != wa;
        println!("WHIR MOBIUS WIDE MOVED WORD: differs={differs}");
        assert!(differs, "a moved word must move the codeword");
    })
}

/// The production shape's encode, timed: a resident stack of 2^27 (64 columns
/// of 2^21, as a group's tables stack) to its 2^28 codeword, today's transform
/// against the wide one, interleaved, median of 9 after 2 warm-ups. Prints a
/// `WHIR MOBIUS WIDE BENCH` line a shape.
#[test]
#[ignore = "box tier: a measurement"]
fn the_wide_mobius_encode_at_the_production_shape() {
    alone(|| {
        let l = 27;
        let shapes: [(&str, Vec<usize>); 2] = [
            ("64 columns of 2^21", vec![1 << 21; 64]),
            ("mixed: 48 of 2^21, 24 of 2^20, 40 of 2^16 and gaps", {
                let mut v = vec![1 << 21; 48];
                v.extend(vec![1 << 20; 24]);
                v.extend(vec![1 << 16; 40]);
                v
            }),
        ];
        for (label, lens) in shapes {
            let cols: Vec<Vec<u64>> = lens
                .iter()
                .enumerate()
                .map(|(k, &n)| words(k as u64 + 1, n))
                .collect();
            let refs: Vec<&[u64]> = cols.iter().map(|c| c.as_slice()).collect();
            let store = DeviceColumns::upload(&refs).expect("room for the store");
            // Columns at their running offsets, a 4,096-word gap after each 2^16 one.
            let mut at = Vec::new();
            let mut off = 0usize;
            for (k, &n) in lens.iter().enumerate() {
                at.push((k, off));
                off += n + if n == 1 << 16 { 4096 } else { 0 };
            }
            assert!(off <= 1 << l, "{label}: the layout overruns the stack");
            // The production shape's codewords, word for word, before any timing.
            let today = with_mobius_wide(false, || {
                encode_codeword_resident(&store, &at, l, 1, false)
                    .unwrap()
                    .to_host()
                    .unwrap()
            });
            let wide = with_mobius_wide(true, || {
                encode_codeword_resident(&store, &at, l, 1, false)
                    .unwrap()
                    .to_host()
                    .unwrap()
            });
            let equal = today == wide;
            println!("WHIR MOBIUS WIDE PARITY: 2^{l} · {label} · equal={equal}");
            assert!(
                equal,
                "{label}: the wide transform's codeword differs at the production shape"
            );
            drop((today, wide));
            let run = |wide: bool| {
                with_mobius_wide(wide, || {
                    let t = Instant::now();
                    let before = mobius_wide_gathers();
                    let cw =
                        encode_codeword_resident(&store, &at, l, 1, false).expect("a timed encode");
                    cw.synchronize().unwrap();
                    let ms = t.elapsed().as_secs_f64() * 1e3;
                    drop(cw);
                    // The wide arm took the gather and today's did not: the times are the lever's.
                    assert_eq!(
                        mobius_wide_gathers() - before,
                        u64::from(wide),
                        "{label}: the gather count is not the arm's"
                    );
                    ms
                })
            };
            let (mut a, mut b) = (Vec::new(), Vec::new());
            for rep in 0..11 {
                let (x, y) = if rep % 2 == 0 {
                    (run(false), run(true))
                } else {
                    let y = run(true);
                    (run(false), y)
                };
                if rep >= 2 {
                    a.push(x);
                    b.push(y);
                }
            }
            a.sort_by(f64::total_cmp);
            b.sort_by(f64::total_cmp);
            let (ma, mb) = (a[a.len() / 2], b[b.len() / 2]);
            println!(
                "WHIR MOBIUS WIDE BENCH: resident 2^{l} → 2^{} · {label} · today {ma:.3} ms · wide {mb:.3} ms · R {:.3} · saved {:.3} ms",
                l + 1,
                mb / ma,
                ma - mb
            );
        }
    })
}
