//! ★ C5's falsifier: what one table's Round-1 LDE costs to REBUILD on device.
//!
//! # The question, and why a probe decides it
//!
//! Lane P5's pass-5 profile put **27.1 s** of the block's card idle inside Round
//! 1, and traced it to one fact: ✓ the `VramGate` throttles *construction*, not
//! *residency* — a table's permit is released when its commit task returns, but
//! the LDE, the tree and the snapshot stay on the card until its fused task
//! consumes them. **C5** is the proposed answer: drop a table's LDE once its root
//! is absorbed and rebuild it inside the fused task, so the gate's budget is not
//! spent on tables that are merely waiting.
//!
//! C5's cost is one extra coset LDE per released table. Its benefit is bounded by
//! the 27.1 s pool. ⇒ **the ratio of those two is the whole decision, and it is a
//! measurement.**
//!
//! # ★★ The scope is decided before any measurement, and it decides C5
//!
//! ✓ Measured: the block runs 64 proves whose AIR counts sum to **1,109** tables
//! (27×16 · 26×3 · 58×1 · 14×19 · 11×25). So:
//!
//! | policy | rebuilds/block | break-even per rebuild |
//! |---|---|---|
//! | every table | 1,109 | **24.4 ms** |
//! | the heaviest table only | 64 | **423 ms** |
//! | the heaviest two | 128 | 212 ms |
//! | the heaviest three | 192 | 141 ms |
//!
//! A forward NTT over `LFM_HASH`'s 2^20 × 329 at blowup 4 touches 1.38 G base
//! elements, which is nowhere near 24 ms. ⇒ **the all-tables policy is dead on
//! arithmetic, before this probe runs.** But ✓ one table is 83% of the Round-1
//! budget, so releasing *that one* is most of the admission win — and a top-1
//! policy has 17× the budget. **Anyone who builds C5 all-tables has built the
//! dead version.**
//!
//! # What this probe does NOT settle
//!
//! ⚠ It prices the **cost** only. The **benefit** is an admission model, and the
//! gate bounds construction rather than residency, so the benefit needs its own
//! arm. ⚠ And the rebuild sits on the fused task's critical path, not a
//! background stream: a rebuild that is cheap in total device time can still
//! lengthen the table that owns it.
//!
//! ⛔ **Rebase hazard, by design and not as a footnote.** `pt/shared-vramgate` @
//! `fd11d97b` makes `VramGate` a process-wide `OnceLock`. C5 must be built ON TOP
//! of it: a release has to hand bytes back to the gate, and with a process-wide
//! gate those bytes reach the *other* prove too — which is where most of C5's
//! value lands once the base and the wraps overlap. Built against today's
//! per-prove gate the release credits only its own prove and is worth strictly
//! less.

/// One shape to price: a label, trace rows, base-field columns, blowup.
struct Shape {
    name: &'static str,
    n: usize,
    cols: usize,
    blowup: usize,
}

/// The default ladder.
///
/// ★ A LADDER rather than three named tables, deliberately. C5's decision is
/// **per table** — the policy releases the heaviest ones — so the useful output
/// is a curve any later table shape can be read against, not three points that
/// go stale the moment the census moves. The anchor is exact: ✓ `LFM_HASH` at
/// 2^20 × 329 @ blowup 4 is the measured fan-in-2 shape, and the same table at
/// 2^21 is the one that aborted fan-in 3 at 25.95 GiB. The rest bracket the
/// wrap's 14 tables and the node's 11.
const LADDER: &[Shape] = &[
    Shape {
        name: "LFM_HASH-anchor",
        n: 1 << 20,
        cols: 329,
        blowup: 4,
    },
    Shape {
        name: "LFM_HASH-fanin3",
        n: 1 << 21,
        cols: 329,
        blowup: 4,
    },
    Shape {
        name: "wide-2^19",
        n: 1 << 19,
        cols: 329,
        blowup: 4,
    },
    Shape {
        name: "mid-2^20",
        n: 1 << 20,
        cols: 128,
        blowup: 4,
    },
    Shape {
        name: "mid-2^21",
        n: 1 << 21,
        cols: 128,
        blowup: 4,
    },
    Shape {
        name: "narrow-2^20",
        n: 1 << 20,
        cols: 32,
        blowup: 4,
    },
    Shape {
        name: "narrow-2^22",
        n: 1 << 22,
        cols: 32,
        blowup: 4,
    },
];

/// `C5_PROBE_SHAPES="name:n:cols:blowup,…"` replaces the ladder, for pricing a
/// shape the census panel turns up later without editing this file.
fn shapes() -> Vec<Shape> {
    let Ok(spec) = std::env::var("C5_PROBE_SHAPES") else {
        return LADDER
            .iter()
            .map(|s| Shape {
                name: s.name,
                n: s.n,
                cols: s.cols,
                blowup: s.blowup,
            })
            .collect();
    };
    spec.split(',')
        .filter(|s| !s.is_empty())
        .map(|entry| {
            let f: Vec<&str> = entry.split(':').collect();
            assert_eq!(
                f.len(),
                4,
                "C5_PROBE_SHAPES entries are name:n:cols:blowup, got `{entry}`"
            );
            let num = |i: usize| -> usize {
                f[i].parse()
                    .unwrap_or_else(|_| panic!("`{}` is not a number in `{entry}`", f[i]))
            };
            let (n, cols, blowup) = (num(1), num(2), num(3));
            assert!(n.is_power_of_two() && blowup.is_power_of_two(), "{entry}");
            // Leaked so the label can stay `&'static str` alongside the ladder's;
            // a handful of short strings in a measurement-only test.
            Shape {
                name: Box::leak(f[0].to_string().into_boxed_str()),
                n,
                cols,
                blowup,
            }
        })
        .collect()
}

/// ★★★ THE PROBE. Prints one line per shape and the kill thresholds beside them.
///
/// Measurement only: it touches no proof, no transcript and no admission path,
/// and `coset_lde_row_major_no_tree` has no caller on the proving path.
///
/// ⛔ The values in the buffers are arbitrary because the timing is
/// shape-determined, not data-determined: the kernels are a pointwise coset
/// multiply and a radix NTT, both branch-free over their inputs. Only `n`,
/// `cols` and `blowup` move the clock.
///
/// ⛔ And it synchronises inside the timed region (see
/// `coset_lde_row_major_no_tree`): the launches are async, so timing without the
/// sync would report a queue submission and declare the rebuild free.
#[test]
#[ignore = "box tier: needs an idle card; allocates up to ~22 GiB of device memory"]
fn c5_probe_prices_the_round_1_lde_rebuild() {
    const REPS: usize = 5;
    const GIB: f64 = (1u64 << 30) as f64;
    const MIB: f64 = (1u64 << 20) as f64;

    println!(
        "\n★★★ C5 PROBE — what a Round-1 residency release costs to undo.\n\
         Break-even against the 27.1 s Round-1 idle pool, by policy:\n\
         \x20  every table (1,109 rebuilds/block): 24 ms  ⇒ dead above it\n\
         \x20  heaviest THREE (192):              141 ms\n\
         \x20  heaviest TWO (128):                212 ms\n\
         \x20  heaviest ONE (64):                 423 ms  ⇒ KILL C5 above this\n"
    );

    let mut any = false;
    for s in shapes() {
        let lde_bytes = (s.n as u64) * (s.blowup as u64) * (s.cols as u64) * 8;
        // The host input is the trace, not the LDE: n × cols.
        let row_major = vec![0x1234_5678_9abc_def0u64; s.n * s.cols];
        let weights = vec![0x0f0f_0f0f_0f0f_0f0fu64; s.n];

        // Warm-up: the first call populates the pool and pays a cold pinned
        // staging allocation. A production release reuses a warm pool, so a
        // cold number would price an allocation that will not happen.
        if math_cuda::lde::coset_lde_row_major_no_tree(&row_major, s.n, s.cols, s.blowup, &weights)
            .is_err()
        {
            println!(
                "   C5 PROBE {}: DECLINED by the device (shape does not fit)",
                s.name
            );
            continue;
        }

        let mut ms: Vec<f64> = Vec::with_capacity(REPS);
        for _ in 0..REPS {
            let t = std::time::Instant::now();
            let out = math_cuda::lde::coset_lde_row_major_no_tree(
                &row_major, s.n, s.cols, s.blowup, &weights,
            )
            .expect("the warm-up already succeeded for this shape");
            ms.push(t.elapsed().as_secs_f64() * 1000.0);
            drop(out);
        }
        ms.sort_by(|a, b| a.partial_cmp(b).expect("no NaN from a clock"));
        let median = ms[REPS / 2];
        let releases = lde_bytes as f64 / GIB;
        let ratio = (lde_bytes as f64 / MIB) / median;
        let verdict = if median > 423.0 {
            "⛔ above the top-1 break-even"
        } else if median > 212.0 {
            "top-1 only"
        } else if median > 141.0 {
            "top-2"
        } else if median > 24.4 {
            "top-3"
        } else {
            "★ every table clears"
        };
        println!(
            "   C5 PROBE {}: n={} cols={} blowup={} · rebuild {median:.1} ms · \
             releases {releases:.2} GiB · ratio {ratio:.0} MiB/ms · {verdict} \
             (reps {ms:.0?})",
            s.name, s.n, s.cols, s.blowup,
        );
        any = true;
    }

    // ⛔ THE CHECK THAT MAKES THIS A TEST. Without a card the loop above prints
    // nothing, returns, and reads as a clean pass — which is how a probe comes
    // to be quoted as evidence that C5 is cheap.
    assert!(
        any,
        "★ NO SHAPE WAS MEASURED. Every shape was declined by the device, or \
         there is no device. This is NOT a result about C5 and must not be \
         reported as one."
    );
}
