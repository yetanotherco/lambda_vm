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
//!
//! The stack read where it lies (`evaluate_slices` / `coset_openings` with
//! `in_place`) is compared with the assembled stack word for word, raw: it
//! changes what the tile kernel reads, not its arithmetic.

use std::time::Instant;

use math_cuda::columns::DeviceColumns;
use math_cuda::whir_coset::{
    CosetPlan, StackParts, StackSlices, coset_openings, evaluate_resident, evaluate_slices,
};

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

/// A stack laid out as `StackedLayout::build` lays the block prover's: two
/// tall columns (3/8 of the stack), then about 340 more, three in four short
/// (`2^4`…`2^8` values, under a 256-value tile) and the rest `2^9`…
/// `2^(num_vars-6)`, as many as fit in 94 % of the stack; taller first, every
/// seventh pair out of order, each at a multiple of its height with the
/// cursor only growing (so alignment gaps); then the room left up to 94 %
/// filled, taller columns first (a zero tail of 6 % or more, as the block's
/// stacks have). Every fifth column is 5 values short of its height (an end
/// mid-tile). The store holds a junk column before each column and a tile of
/// junk after the last, none of it in the stack, so a read past a part's end
/// shows as a wrong value. Returns the store and the parts.
fn production_stack(num_vars: usize, seed: u64) -> (DeviceColumns, Vec<(usize, usize)>) {
    let mut rng = Rng(seed);
    let placed = production_layout(num_vars, &mut rng);
    let mut columns: Vec<Vec<u64>> = Vec::new();
    let mut parts = Vec::new();
    for (offset, len) in placed {
        columns.push((0..1 + rng.next() % 37).map(|_| rng.next() % P).collect());
        parts.push((columns.len(), offset));
        columns.push((0..len).map(|_| rng.next() % P).collect());
    }
    columns.push((0..256).map(|_| rng.next() % P).collect());
    let refs: Vec<&[u64]> = columns.iter().map(Vec::as_slice).collect();
    let store = DeviceColumns::upload(&refs).expect("device columns (needs a GPU)");
    (store, parts)
}

/// [`production_stack`]'s columns: `(offset in the stack, length)`, in the
/// order placed.
fn production_layout(num_vars: usize, rng: &mut Rng) -> Vec<(usize, usize)> {
    let cap = (1usize << num_vars) * 94 / 100;
    let mut room = cap - (3usize << (num_vars - 3));
    let mut heights = Vec::new();
    for _ in 0..338 {
        let h = if rng.next().is_multiple_of(4) {
            9 + (rng.next() % (num_vars as u64 - 14)) as usize
        } else {
            4 + (rng.next() % 5) as usize
        };
        if (1usize << h) <= room {
            room -= 1 << h;
            heights.push(h);
        }
    }
    heights.sort_unstable_by(|a, b| b.cmp(a));
    for i in (0..heights.len().saturating_sub(1)).step_by(7) {
        heights.swap(i, i + 1);
    }
    let heights: Vec<usize> = [num_vars - 2, num_vars - 3]
        .into_iter()
        .chain(heights)
        .collect();
    let mut placed = Vec::new();
    let mut cursor = 0usize;
    for (i, &h) in heights.iter().enumerate() {
        let size = 1usize << h;
        let offset = cursor.next_multiple_of(size);
        if offset + size > cap {
            continue;
        }
        cursor = offset + size;
        placed.push((offset, if i % 5 == 4 { size - 5 } else { size }));
    }
    for h in (4..=num_vars - 4).rev() {
        let size = 1usize << h;
        while cursor.next_multiple_of(size) + size <= cap {
            let offset = cursor.next_multiple_of(size);
            cursor = offset + size;
            placed.push((offset, size));
        }
    }
    placed
}

/// [`stack`]'s columns with a zero tail of 15 %: thirteen whole columns and
/// six tenths of the fourteenth, a tile of junk after it in the store (not in
/// the stack).
fn tail_stack(num_vars: usize, seed: u64) -> (DeviceColumns, Vec<(usize, usize)>) {
    let mut rng = Rng(seed);
    let column = 1usize << (num_vars - 4);
    let columns: Vec<Vec<u64>> = (0..15)
        .map(|k| {
            let len = match k {
                13 => column * 6 / 10,
                14 => 256,
                _ => column,
            };
            (0..len).map(|_| rng.next() % P).collect()
        })
        .collect();
    let refs: Vec<&[u64]> = columns.iter().map(Vec::as_slice).collect();
    let store = DeviceColumns::upload(&refs).expect("device columns (needs a GPU)");
    let parts: Vec<(usize, usize)> = (0..14).map(|k| (k, k * column)).collect();
    (store, parts)
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
/// order: block `q`'s leaf `r` value `t` at `(q·2^kept + r)·2^fold + t`, for a
/// codeword of blowup `2^log_blowup`.
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

/// The tiles a stack of `2^num_vars` values folds in: `2^tile_bits` values
/// each, as many bits as fold below the kept coset, eight at most.
fn tiles(num_vars: usize) -> u64 {
    let tile_bits = (num_vars - LOG_FOLD - KEPT).min(8);
    1 << (num_vars - tile_bits)
}

/// The production-like layouts the card tests and the measurement use (no
/// card needed): about 340 columns or more, 85-94 % of the stack filled,
/// alignment gaps, and tiles of every kind at the plan's tile width.
#[test]
fn the_production_like_stacks_have_every_tile_kind() {
    for (num_vars, seed) in [
        (16usize, 21u64),
        (20, 22),
        (24, 23),
        (20, 31),
        (24, 32),
        (27, 100),
    ] {
        let placed = production_layout(num_vars, &mut Rng(seed));
        let n = 1usize << num_vars;
        let filled: usize = placed.iter().map(|&(_, len)| len).sum();
        let mut sorted = placed.clone();
        sorted.sort_unstable();
        let (mut gaps, mut end) = (0usize, 0usize);
        for &(offset, len) in &sorted {
            assert!(offset >= end, "n {num_vars}: columns overlap");
            gaps += offset - end;
            end = offset + len;
        }
        let parts: Vec<(usize, usize, usize)> = placed
            .iter()
            .map(|&(offset, len)| (offset, len, offset))
            .collect();
        let tile_bits = (num_vars - LOG_FOLD - KEPT).min(8);
        let kinds = StackParts::new(&parts, num_vars, n)
            .expect("the parts place")
            .tile_kinds(num_vars, tile_bits);
        println!(
            "COSET EVAL LAYOUT n={num_vars} seed={seed} parts={} fill={:.3} gaps={:.4} tiles_full={} tiles_zero={} tiles_mixed={}",
            placed.len(),
            filled as f64 / n as f64,
            gaps as f64 / n as f64,
            kinds.full,
            kinds.zero,
            kinds.mixed
        );
        assert!(
            placed.len() >= 150,
            "n {num_vars}: {} columns",
            placed.len()
        );
        assert!(
            (n * 85 / 100..=n * 94 / 100).contains(&filled),
            "n {num_vars}: {filled} filled"
        );
        assert!(gaps > 0, "n {num_vars}: no alignment gap");
        assert!(kinds.zero > 0 && kinds.mixed > 0, "n {num_vars}: {kinds:?}");
    }
}

/// ★ The stack read where it lies is the assembled stack: the cosets equal
/// word for word, raw (the same arithmetic over the same words), and equal
/// the re-encoded codeword at every block's leaves; the tiles' kinds add up.
/// At blowups 4 and 2, stacks of `2^16` (4-value tiles), `2^20` (64) and
/// `2^24` (256, two rest bits), three layouts each: [`stack`]'s, a
/// production-like one ([`production_stack`]: short columns, ends mid-tile,
/// alignment gaps, junk between the columns in the store) and a 15 % zero
/// tail ([`tail_stack`]).
#[test]
fn coset_blocks_read_in_place_equal_the_assembled_stack() {
    for (log_blowup, count) in [(LOG_BLOWUP, 114usize), (1, 229)] {
        for (num_vars, seed) in [(16usize, 21u64), (20, 22), (24, 23)] {
            let (bench, bench_parts, _) = stack(num_vars, seed);
            let (production, production_parts) = production_stack(num_vars, seed);
            let (tail, tail_parts) = tail_stack(num_vars, seed);
            for (shape, store, parts) in [
                ("bench", &bench, &bench_parts),
                ("production", &production, &production_parts),
                ("tail", &tail, &tail_parts),
            ] {
                let label = format!("n {num_vars} blowup 2^{log_blowup} {shape}");
                let leaves = queries_at(num_vars, log_blowup, count, seed);
                let plan =
                    CosetPlan::new(num_vars, log_blowup, LOG_FOLD, KEPT, &leaves).expect("plan");
                let slices = StackSlices::from_resident(store, parts, num_vars, log_blowup)
                    .expect("stack slices");
                let (assembled, none) =
                    evaluate_slices(&slices, &plan, false, None).expect("assembled");
                assert_eq!(none, None, "{label}: assembled, no tile kinds");
                let assembled = assembled.to_host().expect("read back");
                let mut split = Vec::new();
                let (in_place, kinds) =
                    evaluate_slices(&slices, &plan, true, Some(&mut split)).expect("in place");
                assert_eq!(split.len(), 4, "{label}: four phases timed: {split:?}");
                let in_place = in_place.to_host().expect("read back");
                let kinds = kinds.unwrap_or_else(|| panic!("{label}: the parts must place"));
                let resident = evaluate_resident(store, parts, &plan, None)
                    .expect("resident")
                    .to_host()
                    .expect("read back");
                let gathered = reencoded_at_blocks_at(store, parts, num_vars, log_blowup, &plan);
                let raw = in_place
                    .iter()
                    .zip(&assembled)
                    .filter(|(a, b)| a != b)
                    .count();
                let vs_codeword = mismatches(&in_place, &gathered);
                println!(
                    "COSET EVAL INPLACE n={num_vars} blowup_log={log_blowup} shape={shape} tiles_full={} tiles_zero={} tiles_mixed={} raw_vs_assembled={raw} vs_codeword={vs_codeword}",
                    kinds.full, kinds.zero, kinds.mixed
                );
                assert_eq!(in_place.len(), assembled.len(), "{label}: lengths");
                assert_eq!(
                    (raw, vs_codeword),
                    (0, 0),
                    "{label}: the stack read in place differs"
                );
                assert_eq!(assembled, resident, "{label}: the two assembled paths");
                assert_eq!(
                    kinds.full + kinds.zero + kinds.mixed,
                    tiles(num_vars),
                    "{label}: every tile has one kind"
                );
                if shape != "bench" {
                    assert!(kinds.zero > 0, "{label}: a zero tail: {kinds:?}");
                }
                if shape == "production" {
                    assert!(kinds.mixed > 0, "{label}: mixed tiles: {kinds:?}");
                }
            }
        }
    }
}

/// ★ A round-0 tree's openings with the stack read where it lies are the
/// assembled stack's: the same subtrees' nodes, byte for byte, and the same
/// queried values; only the in-place run reports tile kinds.
#[test]
fn coset_openings_read_in_place_are_the_assembled_stacks() {
    for (num_vars, seed) in [(20usize, 31u64), (24, 32)] {
        let (store, parts) = production_stack(num_vars, seed);
        let leaves = queries(num_vars, 114, seed);
        let plan = CosetPlan::new(num_vars, LOG_BLOWUP, LOG_FOLD, KEPT, &leaves).expect("plan");
        let slices =
            StackSlices::from_resident(&store, &parts, num_vars, LOG_BLOWUP).expect("slices");
        let hash = math_cuda::DeviceHash::Poseidon1;
        let assembled = coset_openings(&slices, &plan, &leaves, hash, false).expect("assembled");
        let in_place = coset_openings(&slices, &plan, &leaves, hash, true).expect("in place");
        println!(
            "COSET EVAL OPENINGS n={num_vars} nodes_equal={} values_equal={} in_place={:?}",
            in_place.nodes == assembled.nodes,
            in_place.values == assembled.values,
            in_place.in_place
        );
        assert_eq!(assembled.in_place, None, "n {num_vars}: assembled");
        assert!(in_place.in_place.is_some(), "n {num_vars}: read in place");
        assert_eq!(in_place.leaves, assembled.leaves, "n {num_vars}: leaves");
        assert!(
            in_place.nodes == assembled.nodes,
            "n {num_vars}: the subtrees' nodes differ"
        );
        assert_eq!(
            in_place.values, assembled.values,
            "n {num_vars}: the queried values"
        );
    }
}

/// ★ The long kernels' grids cut to 1, 3 or 85 blocks (`device::with_grid_cap`,
/// as a background thread runs them beside latency-bound kernels): the same
/// words, raw, as the full grid, the stack assembled and read in place, and
/// the same openings' subtrees and values. At blowups 4 and 2, `2^20` and
/// `2^24`, the production-like stack.
#[test]
fn coset_kernels_under_a_grid_cap_give_the_full_grids_words() {
    use math_cuda::device::{grid_cap, with_grid_cap};
    for (log_blowup, count) in [(LOG_BLOWUP, 114usize), (1, 229)] {
        for (num_vars, seed) in [(20usize, 51u64), (24, 52)] {
            let (store, parts) = production_stack(num_vars, seed);
            let leaves = queries_at(num_vars, log_blowup, count, seed);
            let plan = CosetPlan::new(num_vars, log_blowup, LOG_FOLD, KEPT, &leaves).expect("plan");
            let slices =
                StackSlices::from_resident(&store, &parts, num_vars, log_blowup).expect("slices");
            let hash = math_cuda::DeviceHash::Poseidon1;
            let run = || {
                let assembled = evaluate_resident(&store, &parts, &plan, None)
                    .expect("assembled")
                    .to_host()
                    .expect("read back");
                let (in_place, kinds) =
                    evaluate_slices(&slices, &plan, true, None).expect("in place");
                assert!(kinds.is_some(), "read in place");
                let opened = coset_openings(&slices, &plan, &leaves, hash, true).expect("openings");
                (
                    assembled,
                    in_place.to_host().expect("read back"),
                    opened.nodes,
                    opened.values,
                )
            };
            assert_eq!(grid_cap(), None, "no cap outside with_grid_cap");
            let full = run();
            for cap in [1u32, 3, 85] {
                let label = format!("n {num_vars} blowup 2^{log_blowup}, {cap} blocks");
                let capped = with_grid_cap(Some(cap), || {
                    assert_eq!(grid_cap(), Some(cap));
                    run()
                });
                assert_eq!(grid_cap(), None, "{label}: the cap restored");
                let raw = |a: &[u64], b: &[u64]| a.iter().zip(b).filter(|(x, y)| x != y).count();
                let (assembled, in_place) = (raw(&capped.0, &full.0), raw(&capped.1, &full.1));
                println!(
                    "COSET EVAL CAPPED n={num_vars} blowup_log={log_blowup} cap={cap} assembled_vs_full={assembled} in_place_vs_full={in_place} nodes_equal={} values_equal={}",
                    capped.2 == full.2,
                    capped.3 == full.3
                );
                assert_eq!(
                    (capped.0.len(), capped.1.len(), assembled, in_place),
                    (full.0.len(), full.1.len(), 0, 0),
                    "{label}: the cosets differ from the full grid's"
                );
                assert!(capped.2 == full.2, "{label}: the subtrees' nodes differ");
                assert_eq!(capped.3, full.3, "{label}: the queried values differ");
            }
        }
    }
}

/// ★ Parts the in-place kernels cannot place (two columns overlapping in the
/// stack: no stacked layout makes them, a stand-in for whatever might) are
/// assembled after all, the in-place request reporting no tile kinds — what
/// the caller counts as assembled after all — with the assembled values.
#[test]
fn parts_the_kernels_cannot_place_are_assembled_after_all() {
    let num_vars = 20;
    let (store, _, _) = stack(num_vars, 41);
    let column = 1usize << (num_vars - 4);
    let parts = [(0usize, 0usize), (1, column - 7), (2, 3 * column)];
    let leaves = queries(num_vars, 114, 41);
    let plan = CosetPlan::new(num_vars, LOG_BLOWUP, LOG_FOLD, KEPT, &leaves).expect("plan");
    let slices = StackSlices::from_resident(&store, &parts, num_vars, LOG_BLOWUP).expect("slices");
    assert!(
        slices.parts_in_place().is_none(),
        "overlapping parts refused"
    );
    let (asked, kinds) = evaluate_slices(&slices, &plan, true, None).expect("asked in place");
    let (off, _) = evaluate_slices(&slices, &plan, false, None).expect("assembled");
    let opened = coset_openings(
        &slices,
        &plan,
        &leaves,
        math_cuda::DeviceHash::Poseidon1,
        true,
    )
    .expect("openings");
    assert_eq!(kinds, None, "no tile kinds: assembled");
    assert_eq!(opened.in_place, None, "the openings assembled too");
    assert_eq!(
        asked.to_host().expect("read back"),
        off.to_host().expect("read back"),
        "the assembled values"
    );
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
/// queries and the cosets evaluated from the assembled stack, and (B1) from
/// the stack read where it lies (each then a synchronize), interleaved
/// A B B1, `COSET_EVAL_WARM` warm-ups then `COSET_EVAL_REPS` timed each; the
/// same for a group of three polynomials one after another (A3, B3, B13); B's
/// and B1's phases; the arms' peak pool bytes; and the checks: B's and B1's
/// values equal A's codeword at every block's leaves, B1's equal B's raw.
///
/// `COSET_EVAL_LOG` (27), `COSET_EVAL_QUERIES` (114), `COSET_EVAL_WARM` (3),
/// `COSET_EVAL_REPS` (20), `COSET_EVAL_BLOWUP` (the blowup's log, 2),
/// `COSET_EVAL_STACK` (`bench`: [`stack`]'s fifteen columns; `production`:
/// [`production_stack`]).
#[test]
#[ignore = "measurement arm: a minute on a card; run on the box"]
fn coset_eval_bench() {
    let num_vars = env_usize("COSET_EVAL_LOG", 27);
    let count = env_usize("COSET_EVAL_QUERIES", 114);
    let warm = env_usize("COSET_EVAL_WARM", 3);
    let reps = env_usize("COSET_EVAL_REPS", 20);
    let log_blowup = env_usize("COSET_EVAL_BLOWUP", LOG_BLOWUP);
    let shape = std::env::var("COSET_EVAL_STACK").unwrap_or_else(|_| "bench".into());
    let stacks: Vec<(DeviceColumns, Vec<(usize, usize)>)> = (0..3)
        .map(|k| match shape.as_str() {
            "bench" => {
                let (store, parts, _) = stack(num_vars, 100 + k);
                (store, parts)
            }
            "production" => production_stack(num_vars, 100 + k),
            other => panic!("COSET_EVAL_STACK={other}: bench or production"),
        })
        .collect();
    let sets: Vec<Vec<u64>> = (0..3)
        .map(|k| queries_at(num_vars, log_blowup, count, 200 + k))
        .collect();
    let plan0 = CosetPlan::new(num_vars, log_blowup, LOG_FOLD, KEPT, &sets[0]).expect("plan");
    let slices: Vec<StackSlices> = stacks
        .iter()
        .map(|(store, parts)| {
            StackSlices::from_resident(store, parts, num_vars, log_blowup).expect("slices")
        })
        .collect();
    println!(
        "COSET EVAL SHAPE n={num_vars} blowup_log={log_blowup} fold={LOG_FOLD} kept={KEPT} queries={count} cosets={} coset_len={} device_mib_b={:.1} codeword_mib_a={:.1} stack={shape} parts={}",
        plan0.classes().len(),
        plan0.coset_len(),
        plan0.device_bytes() as f64 / (1 << 20) as f64,
        ((8usize << num_vars) + (8usize << (num_vars + log_blowup))) as f64 / (1 << 20) as f64,
        stacks[0].1.len(),
    );

    let encode_one = |k: usize| {
        let (store, parts) = &stacks[k];
        let codeword =
            math_cuda::whir::encode_codeword_resident(store, parts, num_vars, log_blowup, false)
                .expect("re-encode");
        codeword.synchronize().expect("re-encode done");
    };
    let cosets_one = |k: usize| {
        let (store, parts) = &stacks[k];
        let plan = CosetPlan::new(num_vars, log_blowup, LOG_FOLD, KEPT, &sets[k]).expect("plan");
        let values = evaluate_resident(store, parts, &plan, None).expect("cosets");
        values.synchronize().expect("cosets done");
    };
    let in_place_one = |k: usize| {
        let plan = CosetPlan::new(num_vars, log_blowup, LOG_FOLD, KEPT, &sets[k]).expect("plan");
        let (values, kinds) = evaluate_slices(&slices[k], &plan, true, None).expect("cosets");
        assert!(kinds.is_some(), "B1 read the stack in place");
        values.synchronize().expect("cosets done");
    };
    let time = |f: &dyn Fn()| {
        let start = Instant::now();
        f();
        start.elapsed().as_secs_f64() * 1e3
    };

    let mut arms: Vec<Vec<f64>> = vec![Vec::new(); 6];
    for rep in 0..warm + reps {
        let ta = time(&|| encode_one(0));
        let tb = time(&|| cosets_one(0));
        let tb1 = time(&|| in_place_one(0));
        let ta3 = time(&|| (0..3).for_each(encode_one));
        let tb3 = time(&|| (0..3).for_each(cosets_one));
        let tb13 = time(&|| (0..3).for_each(in_place_one));
        let label = if rep < warm { "warm" } else { "timed" };
        println!(
            "COSET EVAL REP {label} rep={rep} a_ms={ta:.3} b_ms={tb:.3} a3_ms={ta3:.3} b3_ms={tb3:.3} b1_ms={tb1:.3} b13_ms={tb13:.3}"
        );
        if rep >= warm {
            for (arm, t) in arms.iter_mut().zip([ta, tb, ta3, tb3, tb1, tb13]) {
                arm.push(t);
            }
        }
    }
    for (name, arm) in ["A", "B", "A3", "B3", "B1", "B13"]
        .iter()
        .zip(arms.iter_mut())
    {
        summary(name, arm);
    }

    // B's and B1's phases, each synchronized (so they sum to a little more
    // than the arm). B1's assemble is its parts' placement only; its tiles
    // include the tile map.
    let median = |v: &mut Vec<f64>| {
        v.sort_by(|x, y| x.partial_cmp(y).expect("finite"));
        v[v.len() / 2]
    };
    for in_place in [false, true] {
        let mut phases: Vec<Vec<f64>> = vec![Vec::new(); 5];
        for _ in 0..5 {
            let start = Instant::now();
            let plan =
                CosetPlan::new(num_vars, log_blowup, LOG_FOLD, KEPT, &sets[0]).expect("plan");
            phases[0].push(start.elapsed().as_secs_f64() * 1e3);
            let mut split = Vec::new();
            if in_place {
                evaluate_slices(&slices[0], &plan, true, Some(&mut split)).expect("cosets");
            } else {
                let (store, parts) = &stacks[0];
                evaluate_resident(store, parts, &plan, Some(&mut split)).expect("cosets");
            }
            for (i, (_, ms)) in split.iter().enumerate() {
                phases[i + 1].push(*ms);
            }
        }
        // B's line keeps its words; B1's names its arm at the end.
        let tag = if in_place { " arm=B1" } else { "" };
        println!(
            "COSET EVAL SPLIT plan_ms={:.3} assemble_ms={:.3} tiles_ms={:.3} rest_ms={:.3} transform_ms={:.3}{tag}",
            median(&mut phases[0]),
            median(&mut phases[1]),
            median(&mut phases[2]),
            median(&mut phases[3]),
            median(&mut phases[4]),
        );
    }

    let peak_a = peak_mib(|| encode_one(0));
    let peak_b = peak_mib(|| cosets_one(0));
    let peak_b1 = peak_mib(|| in_place_one(0));
    println!(
        "COSET EVAL MEM peak_mib_a={peak_a:.1} peak_mib_b={peak_b:.1} peak_mib_b1={peak_b1:.1}"
    );

    let (store, parts) = &stacks[0];
    let device = evaluate_resident(store, parts, &plan0, None)
        .expect("cosets")
        .to_host()
        .expect("read back");
    let (in_place, kinds) = evaluate_slices(&slices[0], &plan0, true, None).expect("cosets");
    let in_place = in_place.to_host().expect("read back");
    let kinds = kinds.expect("B1 read the stack in place");
    let gathered = reencoded_at_blocks_at(store, parts, num_vars, log_blowup, &plan0);
    let differ = mismatches(&device, &gathered);
    println!(
        "COSET EVAL CHECK positions={} mismatches={differ} cosets={} queries={count}",
        device.len(),
        plan0.classes().len()
    );
    let differ_b1 = mismatches(&in_place, &gathered);
    let raw = in_place.iter().zip(&device).filter(|(a, b)| a != b).count();
    println!(
        "COSET EVAL CHECK B1 positions={} mismatches={differ_b1} raw_vs_b={raw} tiles_full={} tiles_zero={} tiles_mixed={}",
        in_place.len(),
        kinds.full,
        kinds.zero,
        kinds.mixed
    );
    assert_eq!(differ, 0, "the cosets differ from the re-encoded codeword");
    assert_eq!(
        (differ_b1, raw, in_place.len()),
        (0, 0, device.len()),
        "the stack read in place differs"
    );
}
