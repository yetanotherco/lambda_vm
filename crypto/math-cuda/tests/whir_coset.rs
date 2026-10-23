//! The first-round cosets evaluated from the polynomial
//! ([`math_cuda::whir_coset`]) against the re-encoded codeword they replace,
//! on the device.
//!
//! Runs on a GPU box. The parity test compares, value by value, the coset
//! blocks with the same leaves gathered out of
//! [`math_cuda::whir::encode_codeword_resident`]'s codeword (the re-encode a
//! retired stack is revived with) and with the host algorithm
//! ([`CosetPlan::evaluate_on_host`], pinned against `encode(lift_coefficients)`
//! in the module's own tests). The measurement arm times both at the
//! production shape and prints one `COSET EVAL …` line per reading.

use std::time::Instant;

use math_cuda::columns::DeviceColumns;
use math_cuda::whir_coset::{CosetPlan, evaluate_resident};

const P: u64 = 0xFFFF_FFFF_0000_0001;
/// Production geometry: blowup 4, first fold 6, 256 leaves under a kept node.
const LOG_BLOWUP: usize = 2;
const LOG_FOLD: usize = 6;
const KEPT: usize = 8;

fn canonical(v: u64) -> u64 {
    if v >= P { v - P } else { v }
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

/// A stack of `2^num_vars` values as the block prover lays one out: fifteen
/// columns of `2^(num_vars-4)` end to end and a zero tail of one more, each a
/// separate resident column. Returns the store, its parts and the columns.
fn stack(num_vars: usize, seed: u64) -> (DeviceColumns, Vec<(usize, usize)>, Vec<Vec<u64>>) {
    let mut rng = Rng(seed);
    let column = 1usize << (num_vars - 4);
    let columns: Vec<Vec<u64>> = (0..15)
        .map(|_| (0..column).map(|_| rng.next() % P).collect())
        .collect();
    let refs: Vec<&[u64]> = columns.iter().map(Vec::as_slice).collect();
    let store = DeviceColumns::upload(&refs).expect("device columns (needs a GPU)");
    let parts: Vec<(usize, usize)> = (0..15).map(|k| (k, k * column)).collect();
    (store, parts, columns)
}

/// The stacked polynomial on the host, its zero tail included.
fn whole(columns: &[Vec<u64>], num_vars: usize) -> Vec<u64> {
    let mut whole = columns.concat();
    whole.resize(1 << num_vars, 0);
    whole
}

fn queries(num_vars: usize, count: usize, seed: u64) -> Vec<u64> {
    queries_at(num_vars, LOG_BLOWUP, count, seed)
}

/// [`queries`] over a codeword of blowup `2^log_blowup`.
fn queries_at(num_vars: usize, log_blowup: usize, count: usize, seed: u64) -> Vec<u64> {
    let mut rng = Rng(seed ^ 0x9e37_79b9_7f4a_7c15);
    let num_leaves = 1u64 << (num_vars + log_blowup - LOG_FOLD);
    (0..count).map(|_| rng.next() % num_leaves).collect()
}

/// The re-encoded codeword's values at every block's leaves, in the blocks'
/// order: block `q`'s leaf `r` value `t` at `(q·2^kept + r)·2^fold + t`.
fn reencoded_at_blocks(
    store: &DeviceColumns,
    parts: &[(usize, usize)],
    num_vars: usize,
    plan: &CosetPlan,
) -> Vec<u64> {
    reencoded_at_blocks_at(store, parts, num_vars, LOG_BLOWUP, plan)
}

/// [`reencoded_at_blocks`] for a codeword of blowup `2^log_blowup`.
fn reencoded_at_blocks_at(
    store: &DeviceColumns,
    parts: &[(usize, usize)],
    num_vars: usize,
    log_blowup: usize,
    plan: &CosetPlan,
) -> Vec<u64> {
    let codeword =
        math_cuda::whir::encode_codeword_resident(store, parts, num_vars, log_blowup, true)
            .expect("re-encode");
    codeword.synchronize().expect("re-encode done");
    let leaves: Vec<u64> = (0..plan.classes().len())
        .flat_map(|q| plan.block_leaves(q))
        .collect();
    let num_leaves = 1usize << (num_vars + log_blowup - LOG_FOLD);
    codeword
        .cosets(&leaves, num_leaves, 1 << LOG_FOLD)
        .expect("gather the leaves")
}

fn mismatches(a: &[u64], b: &[u64]) -> usize {
    assert_eq!(a.len(), b.len(), "the same number of values");
    a.iter()
        .zip(b)
        .filter(|(x, y)| canonical(**x) != canonical(**y))
        .count()
}

/// Every coset block equals the re-encoded codeword at its leaves and the
/// host algorithm, at three stack heights: 2 bits folded (tiles only), 6
/// (tiles only), 10 (8 tile + 2 rest); the split path once.
#[test]
fn coset_blocks_equal_the_reencoded_codeword() {
    coset_blocks_at(LOG_BLOWUP, 114);
}

/// Blowup 2 (`LAMBDA_VM_ZF_WHIR_BLOWUP=2`) and its 229 queries: the same
/// parity over a codeword half as long, whose kept cosets are the same width.
#[test]
fn coset_blocks_at_blowup_two_equal_the_reencoded_codeword() {
    coset_blocks_at(1, 229);
}

/// [`coset_blocks_equal_the_reencoded_codeword`] at blowup `2^log_blowup` with
/// `count` queries.
fn coset_blocks_at(log_blowup: usize, count: usize) {
    for (num_vars, seed) in [(16usize, 11u64), (20, 12), (24, 13)] {
        let (store, parts, columns) = stack(num_vars, seed);
        let leaves = queries_at(num_vars, log_blowup, count, seed);
        let plan = CosetPlan::new(num_vars, log_blowup, LOG_FOLD, KEPT, &leaves).expect("plan");
        let mut split = Vec::new();
        let device = evaluate_resident(&store, &parts, &plan, Some(&mut split))
            .expect("coset evaluation")
            .to_host()
            .expect("read back");
        assert_eq!(split.len(), 4, "four phases timed: {split:?}");
        let queued = evaluate_resident(&store, &parts, &plan, None)
            .expect("coset evaluation")
            .to_host()
            .expect("read back");
        let gathered = reencoded_at_blocks_at(&store, &parts, num_vars, log_blowup, &plan);
        let host = plan.evaluate_on_host(&whole(&columns, num_vars));
        let (vs_codeword, vs_host, vs_queued) = (
            mismatches(&device, &gathered),
            mismatches(&device, &host),
            mismatches(&device, &queued),
        );
        // The blowup-4 line names no rate; blowup 2's names its rate at the
        // end.
        let rate = if log_blowup == LOG_BLOWUP {
            String::new()
        } else {
            format!(" blowup_log={log_blowup}")
        };
        println!(
            "COSET EVAL PARITY n={num_vars} cosets={} values={} vs_codeword={vs_codeword} vs_host={vs_host} vs_queued={vs_queued}{rate}",
            plan.classes().len(),
            device.len()
        );
        assert_eq!(
            (vs_codeword, vs_host, vs_queued),
            (0, 0, 0),
            "n {num_vars} blowup 2^{log_blowup}: the cosets differ"
        );
        for &leaf in &leaves {
            let q = plan.block_of(leaf).expect("a block");
            assert!(plan.block_leaves(q).contains(&leaf));
        }
    }
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn summary(arm: &str, ms: &mut [f64]) {
    ms.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
    let at = |p: f64| ms[((ms.len() - 1) as f64 * p).round() as usize];
    println!(
        "COSET EVAL SUMMARY arm={arm} reps={} median_ms={:.3} min_ms={:.3} p10_ms={:.3} p90_ms={:.3} max_ms={:.3}",
        ms.len(),
        at(0.5),
        ms[0],
        at(0.1),
        at(0.9),
        ms[ms.len() - 1]
    );
}

/// Peak pool bytes `run` holds beyond what was held before it, in MiB.
fn peak_mib(run: impl FnOnce()) -> f64 {
    math_cuda::device::drain_and_trim().expect("drain");
    let (before, _) = math_cuda::device::pool_used_bytes().expect("pool");
    math_cuda::device::reset_pool_high_water().expect("reset");
    run();
    math_cuda::device::drain_and_trim().expect("drain");
    let (_, high) = math_cuda::device::pool_used_bytes().expect("pool");
    (high.max(before) - before) as f64 / (1 << 20) as f64
}

/// The measurement: one production-shape polynomial (`2^27` values, blowup 4,
/// first fold 6, 256 leaves under a kept node, 114 queries) — (A) today's
/// re-encode as a retired stack is revived (`encode_codeword_resident` on
/// resident columns, then a synchronize) against (B) the plan built from the
/// queries and the cosets evaluated (then a synchronize), interleaved A B A B,
/// `COSET_EVAL_WARM` warm-ups then `COSET_EVAL_REPS` timed each; the same for
/// a group of three polynomials one after another (A3, B3); B's phases; both
/// arms' peak pool bytes; and the check: B's values equal A's codeword at
/// every block's leaves.
///
/// `COSET_EVAL_LOG` (27), `COSET_EVAL_QUERIES` (114), `COSET_EVAL_WARM` (3),
/// `COSET_EVAL_REPS` (20).
#[test]
#[ignore = "measurement arm: a minute on a card; run on the box"]
fn coset_eval_bench() {
    let num_vars = env_usize("COSET_EVAL_LOG", 27);
    let count = env_usize("COSET_EVAL_QUERIES", 114);
    let warm = env_usize("COSET_EVAL_WARM", 3);
    let reps = env_usize("COSET_EVAL_REPS", 20);
    let stacks: Vec<_> = (0..3).map(|k| stack(num_vars, 100 + k)).collect();
    let sets: Vec<Vec<u64>> = (0..3).map(|k| queries(num_vars, count, 200 + k)).collect();
    let plan0 = CosetPlan::new(num_vars, LOG_BLOWUP, LOG_FOLD, KEPT, &sets[0]).expect("plan");
    println!(
        "COSET EVAL SHAPE n={num_vars} blowup_log={LOG_BLOWUP} fold={LOG_FOLD} kept={KEPT} queries={count} cosets={} coset_len={} device_mib_b={:.1} codeword_mib_a={:.1}",
        plan0.classes().len(),
        plan0.coset_len(),
        plan0.device_bytes() as f64 / (1 << 20) as f64,
        ((8usize << num_vars) + (8usize << (num_vars + LOG_BLOWUP))) as f64 / (1 << 20) as f64,
    );

    let encode_one = |k: usize| {
        let (store, parts, _) = &stacks[k];
        let codeword =
            math_cuda::whir::encode_codeword_resident(store, parts, num_vars, LOG_BLOWUP, false)
                .expect("re-encode");
        codeword.synchronize().expect("re-encode done");
    };
    let cosets_one = |k: usize| {
        let (store, parts, _) = &stacks[k];
        let plan = CosetPlan::new(num_vars, LOG_BLOWUP, LOG_FOLD, KEPT, &sets[k]).expect("plan");
        let values = evaluate_resident(store, parts, &plan, None).expect("cosets");
        values.synchronize().expect("cosets done");
    };
    let time = |f: &dyn Fn()| {
        let start = Instant::now();
        f();
        start.elapsed().as_secs_f64() * 1e3
    };

    let (mut a, mut b, mut a3, mut b3) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for rep in 0..warm + reps {
        let ta = time(&|| encode_one(0));
        let tb = time(&|| cosets_one(0));
        let ta3 = time(&|| (0..3).for_each(encode_one));
        let tb3 = time(&|| (0..3).for_each(cosets_one));
        let label = if rep < warm { "warm" } else { "timed" };
        println!(
            "COSET EVAL REP {label} rep={rep} a_ms={ta:.3} b_ms={tb:.3} a3_ms={ta3:.3} b3_ms={tb3:.3}"
        );
        if rep >= warm {
            a.push(ta);
            b.push(tb);
            a3.push(ta3);
            b3.push(tb3);
        }
    }
    summary("A", &mut a);
    summary("B", &mut b);
    summary("A3", &mut a3);
    summary("B3", &mut b3);

    // B's phases, each synchronized (so they sum to a little more than B).
    let (store, parts, _) = &stacks[0];
    let mut phases: Vec<Vec<f64>> = vec![Vec::new(); 5];
    for _ in 0..5 {
        let start = Instant::now();
        let plan = CosetPlan::new(num_vars, LOG_BLOWUP, LOG_FOLD, KEPT, &sets[0]).expect("plan");
        phases[0].push(start.elapsed().as_secs_f64() * 1e3);
        let mut split = Vec::new();
        evaluate_resident(store, parts, &plan, Some(&mut split)).expect("cosets");
        for (i, (_, ms)) in split.iter().enumerate() {
            phases[i + 1].push(*ms);
        }
    }
    let median = |v: &mut Vec<f64>| {
        v.sort_by(|x, y| x.partial_cmp(y).expect("finite"));
        v[v.len() / 2]
    };
    println!(
        "COSET EVAL SPLIT plan_ms={:.3} assemble_ms={:.3} tiles_ms={:.3} rest_ms={:.3} transform_ms={:.3}",
        median(&mut phases[0]),
        median(&mut phases[1]),
        median(&mut phases[2]),
        median(&mut phases[3]),
        median(&mut phases[4]),
    );

    let peak_a = peak_mib(|| encode_one(0));
    let peak_b = peak_mib(|| cosets_one(0));
    println!("COSET EVAL MEM peak_mib_a={peak_a:.1} peak_mib_b={peak_b:.1}");

    let device = evaluate_resident(store, parts, &plan0, None)
        .expect("cosets")
        .to_host()
        .expect("read back");
    let gathered = reencoded_at_blocks(store, parts, num_vars, &plan0);
    let differ = mismatches(&device, &gathered);
    println!(
        "COSET EVAL CHECK positions={} mismatches={differ} cosets={} queries={count}",
        device.len(),
        plan0.classes().len()
    );
    assert_eq!(differ, 0, "the cosets differ from the re-encoded codeword");
}
