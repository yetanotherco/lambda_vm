//! A GKR input layer kept on the card without its padding against the same
//! layer stored whole (`DeviceFractionTree::from_truncated_input`,
//! `LAMBDA_VM_ARGUE_INPUT_REAL`).
//!
//! ```text
//! cargo test --release -p math-cuda --test gkr_gruen_truncated -- --test-threads=1
//! ```
//!
//! Needs a GPU, like the other tests here. The interactions are padded up to a
//! power of two with `0/1`; a kept layer stores only the real fractions and its
//! rounds read the padding as those constants. Everything the card hands back
//! — the tree's output and levels, every round's sums, the halves after the
//! last round — is compared limb for limb with today's: the kept layer is
//! meant to be raw-identical, not just equal.
//!
//! The ignored test is the card micro-benchmark: the input layer's write, the
//! tree's folds and the input layer's own rounds at a CPU table's shape and a
//! Keccak-round table's, whole against kept.

use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream};
use math_cuda::device::backend;
use math_cuda::gkr::{DeviceFractionTree, GruenLayer, InputPlan};

static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn serial() -> std::sync::MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

const P: u64 = 0xFFFF_FFFF_0000_0001;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// A raw limb, every eighth one a non-canonical representative in
    /// `[p, 2^64)` — what the card may hold.
    fn limb(&mut self) -> u64 {
        let v = self.next() % P;
        if self.next() & 7 == 0 && v < 0xFFFF_FFFF {
            v + P
        } else {
            v
        }
    }

    fn ext(&mut self) -> [u64; 3] {
        [self.limb(), self.limb(), self.limb()]
    }
}

/// An input layer of `interactions` slots rounded up to a power of two, `rows`
/// a slot, the slot in the high bits: random fractions, the padding written
/// as today writes it (`p` 0, `q` 1).
fn padded_layer(interactions: usize, rows: usize, seed: u64) -> (Vec<u64>, Vec<u64>) {
    let full = interactions.next_power_of_two() * rows;
    let real = interactions * rows;
    let mut rng = Rng(seed | 1);
    let mut p = Vec::with_capacity(full * 3);
    let mut q = Vec::with_capacity(full * 3);
    for _ in 0..real {
        p.extend_from_slice(&rng.ext());
        q.extend_from_slice(&rng.ext());
    }
    for _ in real..full {
        p.extend_from_slice(&[0, 0, 0]);
        q.extend_from_slice(&[1, 0, 0]);
    }
    (p, q)
}

fn upload(stream: &Arc<CudaStream>, values: &[u64]) -> CudaSlice<u64> {
    math_cuda::device::htod_or_trim(stream, values).expect("upload")
}

/// Both trees over one layer: today's, carrying the padding, and the kept one.
fn both_trees(
    interactions: usize,
    rows: usize,
    seed: u64,
) -> (DeviceFractionTree, DeviceFractionTree, usize, usize) {
    let be = backend().expect("a device");
    let full = interactions.next_power_of_two() * rows;
    let real = interactions * rows;
    let (p, q) = padded_layer(interactions, rows, seed);
    let stream = be.next_stream();
    let whole =
        DeviceFractionTree::from_device(stream.clone(), upload(&stream, &p), upload(&stream, &q))
            .expect("today's tree");
    let stream = be.next_stream();
    let kept = DeviceFractionTree::from_truncated_input(
        stream.clone(),
        upload(&stream, &p[..real * 3]),
        upload(&stream, &q[..real * 3]),
        real,
        full.trailing_zeros() as usize,
    )
    .expect("the kept tree");
    (whole, kept, real, full)
}

/// Every round of a layer's Gruen session and its finish, raw: the sums of
/// each round, then the halves.
fn run_layer(
    layer: &mut GruenLayer,
    lambda: &[u64; 3],
    challenges: &[[u64; 3]],
    h0: impl Fn(usize) -> bool,
) -> (Vec<Vec<u64>>, Vec<u64>) {
    let mut sums = Vec::new();
    for j in 0..layer.rounds() {
        let fold = (j > 0).then(|| &challenges[j - 1]);
        sums.push(layer.round(fold, lambda, h0(j)).expect("a round"));
    }
    let last = (layer.rounds() > 0).then(|| &challenges[layer.rounds() - 1]);
    (sums, layer.finish(last).expect("the finish"))
}

/// ★ A kept tree is today's tree: the same output and levels, and its input
/// layer's Gruen rounds hand back today's sums every round and today's halves
/// at the end, limb for limb — at shapes whose padding starts one row into
/// the upper halves, at a precompile's three eighths, at a CPU table's twelve
/// slots of 32 and a Keccak-round table's 1,017 of 2,048, with no padding,
/// for every host tail and with `H(0)` summed on the card or not.
#[test]
fn a_kept_layer_runs_todays_rounds_limb_for_limb() {
    let _serial = serial();
    if backend().is_err() {
        eprintln!("no device; skipping");
        return;
    }
    let shapes: [(usize, usize); 9] = [
        (5, 8),
        (3, 4),
        (7, 32),
        (17, 1 << 6),
        (20, 1 << 10),
        (1031, 1 << 3),
        (9, 1 << 11),
        (8, 1 << 9),
        (1025, 1),
    ];
    let mut checked = 0;
    for (n, &(interactions, rows)) in shapes.iter().enumerate() {
        for (low, h0) in [(6usize, 0u8), (0, 1), (9, 2), (2, 0)] {
            let what = format!("{interactions} × {rows} rows, low {low}, h0 {h0}");
            let seed = 0xA5A5 + (n * 16 + low) as u64;
            let (whole, kept, real, full) = both_trees(interactions, rows, seed);
            assert_eq!(
                whole.num_layers(),
                kept.num_layers(),
                "{what}: the same levels"
            );
            assert_eq!(
                whole.output().unwrap(),
                kept.output().unwrap(),
                "{what}: the output fraction"
            );
            assert!(
                kept.holds_input(),
                "{what}: the kept tree holds its input layer"
            );
            assert_eq!(
                kept.input_truncated(),
                real < full,
                "{what}: kept without its padding"
            );
            let input = whole.num_layers() - 1;
            if input >= 1 {
                assert_eq!(
                    whole.layer_to_host(input - 1).unwrap(),
                    kept.layer_to_host(input - 1).unwrap(),
                    "{what}: the level above the input layer, raw"
                );
            }
            let m = whole.layer_num_vars(input) - 1;
            let mut rng = Rng(seed ^ 0x77);
            let point: Vec<u64> = (0..m).flat_map(|_| rng.ext()).collect();
            let lambda = rng.ext();
            let challenges: Vec<[u64; 3]> = (0..m).map(|_| rng.ext()).collect();
            let h0_of = |j: usize| h0 == 1 || (h0 == 2 && j % 2 == 1);
            let mut a = whole.layer_gruen(input, &point, low).unwrap();
            let mut b = kept.layer_gruen(input, &point, low).unwrap();
            assert!(!a.truncated(), "{what}: today's layer is whole");
            assert_eq!(
                b.truncated(),
                real < full,
                "{what}: the kept layer is opened kept"
            );
            let (sums_a, halves_a) = run_layer(&mut a, &lambda, &challenges, h0_of);
            let (sums_b, halves_b) = run_layer(&mut b, &lambda, &challenges, h0_of);
            for (j, (x, y)) in sums_a.iter().zip(&sums_b).enumerate() {
                assert_eq!(x, y, "{what}: round {j}'s sums, raw");
            }
            assert_eq!(
                halves_a, halves_b,
                "{what}: the halves handed to the host, raw"
            );
            checked += 1;
        }
    }
    eprintln!("gkr gruen truncated: {checked} layers, every round and finish raw-identical");
}

/// The same at a CPU table's shape (20 interactions of 32 slots, 2^21 rows)
/// and a Keccak-round table's (1,031 of 2,048, 2^16 rows), one tree at a time:
/// today's values first, then the kept tree's against them.
#[test]
fn a_kept_layer_runs_todays_rounds_at_production_shapes() {
    let _serial = serial();
    let Ok(be) = backend() else {
        eprintln!("no device; skipping");
        return;
    };
    for (name, interactions, rows) in [("CPU", 20usize, 1usize << 21), ("KECCAK", 1031, 1 << 16)] {
        let full = interactions.next_power_of_two() * rows;
        let real = interactions * rows;
        let num_vars = full.trailing_zeros() as usize;
        let (p, q) = padded_layer(interactions, rows, 0xC0FFEE);
        let mut rng = Rng(0xBEEF);
        let point: Vec<u64> = (0..num_vars - 1).flat_map(|_| rng.ext()).collect();
        let lambda = rng.ext();
        let challenges: Vec<[u64; 3]> = (0..num_vars).map(|_| rng.ext()).collect();
        let low = 6;
        let run = |tree: &DeviceFractionTree| {
            let input = tree.num_layers() - 1;
            let mut layer = tree.layer_gruen(input, &point, low).unwrap();
            let truncated = layer.truncated();
            let (sums, halves) = run_layer(&mut layer, &lambda, &challenges, |_| false);
            (tree.output().unwrap(), sums, halves, truncated)
        };
        let want = {
            let stream = be.next_stream();
            let tree = DeviceFractionTree::from_device(
                stream.clone(),
                upload(&stream, &p),
                upload(&stream, &q),
            )
            .expect("today's tree");
            run(&tree)
        };
        let got = {
            let stream = be.next_stream();
            let tree = DeviceFractionTree::from_truncated_input(
                stream.clone(),
                upload(&stream, &p[..real * 3]),
                upload(&stream, &q[..real * 3]),
                real,
                num_vars,
            )
            .expect("the kept tree");
            run(&tree)
        };
        assert!(!want.3 && got.3, "{name}: today's whole, the kept one kept");
        assert_eq!(want.0, got.0, "{name}: the output fraction");
        assert_eq!(want.1.len(), got.1.len(), "{name}: the rounds");
        for (j, (x, y)) in want.1.iter().zip(&got.1).enumerate() {
            assert_eq!(x, y, "{name}: round {j}'s sums, raw");
        }
        assert_eq!(want.2, got.2, "{name}: the halves handed to the host, raw");
        eprintln!(
            "gkr gruen truncated: {name} ({interactions} of {} slots × 2^{} rows): {} rounds and the finish raw-identical",
            interactions.next_power_of_two(),
            rows.trailing_zeros(),
            got.1.len()
        );
    }
}

/// ⛔ Only the Gruen rounds read a kept layer: today's session over it and the
/// cross-check's shadow refuse it, before anything runs. The same tree's
/// levels above it are whole, and their Gruen sessions open as before.
#[test]
fn todays_session_and_the_shadow_refuse_a_kept_layer() {
    let _serial = serial();
    if backend().is_err() {
        eprintln!("no device; skipping");
        return;
    }
    let invalid = cudarc::driver::sys::CUresult::CUDA_ERROR_INVALID_VALUE;
    let (_whole, kept, _, _) = both_trees(5, 1 << 8, 0x51);
    let input = kept.num_layers() - 1;
    let m = kept.layer_num_vars(input) - 1;
    let point = vec![7u64; m * 3];
    let refused = kept.layer_sumcheck(input, &point, &[], &[], 1, 0);
    assert_eq!(
        refused.err().map(|e| e.0),
        Some(invalid),
        "today's session refuses it"
    );
    let layer = kept.layer_gruen(input, &point, 4).unwrap();
    assert_eq!(
        layer.shadow(&[], &[], 1, 0).err().map(|e| e.0),
        Some(invalid),
        "the shadow refuses it"
    );
    let above = kept
        .layer_gruen(input - 1, &point[3..], 4)
        .expect("a level above opens as before");
    assert!(!above.truncated(), "the levels above are whole");
}

/// ⛔ A shape no count rounded up to a power of two gives is refused, never
/// folded: half the cube or less, more than the cube, halves that disagree.
#[test]
fn a_kept_tree_of_a_shape_no_count_gives_is_refused() {
    let _serial = serial();
    let Ok(be) = backend() else {
        eprintln!("no device; skipping");
        return;
    };
    let invalid = cudarc::driver::sys::CUresult::CUDA_ERROR_INVALID_VALUE;
    let num_vars = 6;
    let (p, q) = padded_layer(5, 8, 3);
    let stream = be.next_stream();
    let attempt = |p: &[u64], q: &[u64], real: usize| {
        DeviceFractionTree::from_truncated_input(
            stream.clone(),
            upload(&stream, p),
            upload(&stream, q),
            real,
            num_vars,
        )
        .err()
        .map(|e| e.0)
    };
    assert_eq!(
        attempt(&p[..32 * 3], &q[..32 * 3], 32),
        Some(invalid),
        "half the cube"
    );
    assert_eq!(
        attempt(&p[..60], &q[..60], 20),
        Some(invalid),
        "less than half"
    );
    let mut longer = p.clone();
    longer.extend_from_slice(&[0, 0, 0]);
    assert_eq!(
        attempt(&longer, &longer, 65),
        Some(invalid),
        "more than the cube"
    );
    assert_eq!(
        attempt(&p[..40 * 3], &q[..41 * 3], 40),
        Some(invalid),
        "halves that disagree"
    );
    assert!(
        attempt(&p[..40 * 3], &q[..40 * 3], 40).is_none(),
        "the real shape builds"
    );
}

/// The kept tree holds less on the card: the pool's high water over the
/// input layer's upload, the build and the input layer's rounds, against
/// today's, by at least nine tenths of the padding it does not store (48
/// bytes a cell).
#[test]
fn a_kept_tree_holds_less_than_todays() {
    let _serial = serial();
    let Ok(be) = backend() else {
        eprintln!("no device; skipping");
        return;
    };
    let (interactions, rows) = (20usize, 1usize << 16);
    let full = interactions.next_power_of_two() * rows;
    let real = interactions * rows;
    let (p, q) = padded_layer(interactions, rows, 9);
    let peak = |kept: bool| -> u64 {
        math_cuda::device::drain_and_trim().unwrap();
        math_cuda::device::reset_pool_high_water().unwrap();
        let stream = be.next_stream();
        let (dp, dq) = if kept {
            (
                upload(&stream, &p[..real * 3]),
                upload(&stream, &q[..real * 3]),
            )
        } else {
            (upload(&stream, &p), upload(&stream, &q))
        };
        let tree = if kept {
            DeviceFractionTree::from_truncated_input(
                stream.clone(),
                dp,
                dq,
                real,
                full.trailing_zeros() as usize,
            )
        } else {
            DeviceFractionTree::from_device(stream.clone(), dp, dq)
        }
        .unwrap();
        let input = tree.num_layers() - 1;
        let m = tree.layer_num_vars(input) - 1;
        let mut layer = tree.layer_gruen(input, &vec![5u64; m * 3], 6).unwrap();
        run_layer(&mut layer, &[3, 0, 0], &vec![[2, 0, 0]; m], |_| false);
        stream.synchronize().unwrap();
        drop(layer);
        drop(tree);
        math_cuda::device::drain_and_trim().unwrap();
        math_cuda::device::pool_used_bytes().unwrap().1
    };
    let (whole, kept) = (peak(false), peak(true));
    let padding = ((full - real) * 48) as u64;
    eprintln!(
        "gkr gruen truncated: pool high water today {} MiB, kept {} MiB, padding {} MiB",
        whole >> 20,
        kept >> 20,
        padding >> 20
    );
    assert!(
        kept + padding * 9 / 10 <= whole,
        "the kept tree holds at least nine tenths of the padding less than today's"
    );
}

// ── the card micro-benchmark ────────────────────────────────────────────────

/// What one tree hands back: its output fraction, its input layer's sums
/// round by round, and the halves after the last round.
type LayerValues = (([u64; 3], [u64; 3]), Vec<Vec<u64>>, Vec<u64>);

/// Median, min and max of a sample, as `median/min/max` in milliseconds.
fn median_min_max(mut v: Vec<f64>) -> String {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    format!("{:.3}/{:.3}/{:.3}", v[v.len() / 2], v[0], v[v.len() - 1])
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

/// A table of `width` base columns over `rows`, and a plan of `interactions`
/// whose numerators are a multiplicity column times an integer and whose
/// denominators are `z − Σ α^k·column` over two to four columns — the shape
/// of a bus interaction.
fn columns_and_plan(
    stream: &Arc<CudaStream>,
    width: usize,
    rows: usize,
    interactions: usize,
    seed: u64,
) -> (CudaSlice<u64>, InputPlan) {
    let mut rng = Rng(seed | 1);
    let columns: Vec<u64> = (0..width * rows).map(|_| rng.next() % P).collect();
    let mut side_start = vec![0u32];
    let (mut terms, mut coeffs, mut constants) = (Vec::new(), Vec::new(), Vec::new());
    for _ in 0..interactions {
        // numerator: ± multiplicity
        let column = (rng.next() as usize) % width;
        terms.extend_from_slice(&[(column * rows) as u64, 0]);
        coeffs.extend_from_slice(&[rng.next() % 3 + 1, 0, 0]);
        constants.extend_from_slice(&[0, 0, 0]);
        side_start.push((terms.len() / 2) as u32);
        // denominator: z − fingerprint
        for _ in 0..2 + rng.next() % 3 {
            let column = (rng.next() as usize) % width;
            terms.extend_from_slice(&[(column * rows) as u64, rng.next() % 2]);
            coeffs.extend_from_slice(&rng.ext());
        }
        constants.extend_from_slice(&rng.ext());
        side_start.push((terms.len() / 2) as u32);
    }
    let plan = InputPlan::upload(stream, &side_start, &terms, &coeffs, &constants).unwrap();
    (upload(stream, &columns), plan)
}

/// ⏱ The kill test's measurement (run with `--ignored --nocapture`): at a CPU
/// table's shape and a Keccak-round table's, the input layer's write (padded
/// against real slots), the tree's folds (today's against the kept tree's,
/// whose first fold reads the padding as `0/1`), and the input layer's own
/// Gruen rounds to the host tail plus the finish — 3 warm-ups and 10 timed
/// each, every step waited for, medians. The first timed repetition's values
/// are compared raw. `ARGUE BENCH`, `ARGUE VALUES` and `ARGUE MEM` lines.
#[test]
#[ignore]
fn the_kept_input_layer_at_production_shapes() {
    let _serial = serial();
    let be = backend().expect("a device");
    const WARM: usize = 3;
    const TIMED: usize = 10;
    let time = |f: &mut dyn FnMut()| {
        let from = std::time::Instant::now();
        f();
        from.elapsed().as_secs_f64() * 1e3
    };
    for (name, interactions, rows, width) in [
        ("CPU", 20usize, 1usize << 21, 38usize),
        ("KECCAK", 1031, 1 << 16, 96),
    ] {
        let full = interactions.next_power_of_two() * rows;
        let real = interactions * rows;
        let num_vars = full.trailing_zeros() as usize;

        // The write, from columns resident on the card.
        {
            let stream = be.next_stream();
            let (columns, plan) = columns_and_plan(&stream, width, rows, interactions, 0x11);
            let view = columns.slice(..);
            for padding in [true, false] {
                let mut times = Vec::new();
                for rep in 0..WARM + TIMED {
                    stream.synchronize().unwrap();
                    let ms = time(&mut || {
                        let halves = math_cuda::gkr::input_from_columns(
                            &stream, &view, rows, &plan, padding,
                        )
                        .unwrap();
                        stream.synchronize().unwrap();
                        drop(halves);
                    });
                    if rep >= WARM {
                        times.push(ms);
                    }
                }
                println!(
                    "ARGUE BENCH shape={name} form={} step=write reps={TIMED} ms={}",
                    if padding { "whole" } else { "kept" },
                    median_min_max(times)
                );
            }
        }

        // The folds and the input layer's rounds.
        let (p, q) = padded_layer(interactions, rows, 0xF00D);
        let mut rng = Rng(0xD00D);
        let point: Vec<u64> = (0..num_vars - 1).flat_map(|_| rng.ext()).collect();
        let lambda = rng.ext();
        let challenges: Vec<[u64; 3]> = (0..num_vars).map(|_| rng.ext()).collect();
        let low = 6;
        let mut values: Vec<LayerValues> = Vec::new();
        let mut medians = Vec::new();
        let mut peaks = Vec::new();
        for kept in [false, true] {
            let form = if kept { "kept" } else { "whole" };
            let stream = be.next_stream();
            let cells = if kept { real } else { full };
            let pristine_p = upload(&stream, &p[..cells * 3]);
            let pristine_q = upload(&stream, &q[..cells * 3]);
            stream.synchronize().unwrap();
            let (mut folds, mut rounds, mut first3) = (Vec::new(), Vec::new(), Vec::new());
            let mut peak = 0u64;
            for rep in 0..WARM + TIMED {
                math_cuda::device::reset_pool_high_water().unwrap();
                let mut input = Some((
                    stream.clone_dtod(&pristine_p).unwrap(),
                    stream.clone_dtod(&pristine_q).unwrap(),
                ));
                stream.synchronize().unwrap();
                let mut tree = None;
                let fold_ms = time(&mut || {
                    let (dp, dq) = input.take().unwrap();
                    let built = if kept {
                        DeviceFractionTree::from_truncated_input(
                            stream.clone(),
                            dp,
                            dq,
                            real,
                            num_vars,
                        )
                    } else {
                        DeviceFractionTree::from_device(stream.clone(), dp, dq)
                    }
                    .unwrap();
                    stream.synchronize().unwrap();
                    tree = Some(built);
                });
                let tree = tree.unwrap();
                let input = tree.num_layers() - 1;
                let mut layer = tree.layer_gruen(input, &point, low).unwrap();
                let mut sums = Vec::new();
                let mut three = 0.0;
                let rounds_ms = time(&mut || {
                    for j in 0..layer.rounds() {
                        let from = std::time::Instant::now();
                        let fold = (j > 0).then(|| &challenges[j - 1]);
                        sums.push(layer.round(fold, &lambda, false).unwrap());
                        if j < 3 {
                            three += from.elapsed().as_secs_f64() * 1e3;
                        }
                    }
                });
                let last = (layer.rounds() > 0).then(|| &challenges[layer.rounds() - 1]);
                let finish_from = std::time::Instant::now();
                let halves = layer.finish(last).unwrap();
                let rounds_ms = rounds_ms + finish_from.elapsed().as_secs_f64() * 1e3;
                // The window's high water less the pristine copy held beside it.
                let held = (cells * 48) as u64;
                peak = peak.max(
                    math_cuda::device::pool_used_bytes()
                        .unwrap()
                        .1
                        .saturating_sub(held),
                );
                if rep == WARM {
                    values.push((tree.output().unwrap(), sums, halves));
                }
                if rep >= WARM {
                    folds.push(fold_ms);
                    rounds.push(rounds_ms);
                    first3.push(three);
                }
                drop(layer);
                drop(tree);
            }
            println!(
                "ARGUE BENCH shape={name} form={form} step=folds reps={TIMED} ms={}",
                median_min_max(folds.clone())
            );
            println!(
                "ARGUE BENCH shape={name} form={form} step=rounds reps={TIMED} ms={} first3_ms={}",
                median_min_max(rounds.clone()),
                median_min_max(first3)
            );
            medians.push((median(folds), median(rounds)));
            peaks.push(peak);
        }
        let same_output = values[0].0 == values[1].0;
        let same_rounds = values[0].1 == values[1].1;
        let same_halves = values[0].2 == values[1].2;
        println!(
            "ARGUE VALUES shape={name} output_equal={same_output} rounds_equal={same_rounds} halves_equal={same_halves} rounds={}",
            values[1].1.len()
        );
        println!(
            "ARGUE MEM shape={name} promise_whole_mib={} promise_kept_mib={} pool_peak_whole_mib={} pool_peak_kept_mib={}",
            ((4 * full) as u64 * 24) >> 20,
            math_cuda::gkr::truncated_peak_bytes(real, full) >> 20,
            peaks[0] >> 20,
            peaks[1] >> 20
        );
        println!(
            "ARGUE RATIO shape={name} rounds_kept_over_whole={:.3} folds_kept_over_whole={:.3}",
            medians[1].1 / medians[0].1,
            medians[1].0 / medians[0].0
        );
        assert!(
            same_output && same_rounds && same_halves,
            "{name}: the kept layer's values"
        );
    }
}
