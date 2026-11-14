//! ★ LT's rows of the memory accesses' timestamp comparisons by merging equal
//! neighbours ([`StreamSkip::lt_merge`]): what the rows are, that one row
//! builder feeds the tables and every count of their per-row lookup
//! `ARE_BYTES[μ, 0]`, and what changes in a build with it.
//!
//! - A merged list of rows holds each op's count in rows of μ in 1..=MU_MAX,
//!   satisfies LT's constraints, and puts on the ALU, MSB16 and IS_HALF buses
//!   what the map's rows put there; only the μ bound's lookups differ.
//! - A chunk's rows ([`lt_chunk_rows`]) merge only the memory parts, the map
//!   takes the rest; without merged parts they are the map's rows exactly.
//!   Counted on the pool or by collectors, the per-row lookups are those of
//!   the tables laid out from the rows, chunk by chunk, a chunk boundary
//!   inside a run included.
//! - A windowed build with it changes LT's tables and BITWISE's table only;
//!   LT's tuples (each with its μ summed over LT's tables) are today's, and
//!   every inner bus balances. Counting the map's rows under merged tables
//!   (the fault) leaves the μ bound's lookups unbalanced.

use std::collections::{BTreeMap, HashMap};

use stark::lookup::BusInteraction;

use super::bus_net_tests::{multiplicity, named, unbalanced};
use super::finish_tests::{contents, run, windowed};
use super::*;
use crate::tables::MaxRowsConfig;
use crate::tables::lt::{LtConstraints, cols};
use crate::tables::types::{BusId, FE, MU_MAX, merge_rows_into};
use crate::test_utils::{busless_air, validate_busless};

type Table = TraceTable<GoldilocksField, GoldilocksExtension>;

fn merged(ops: &[LtOperation]) -> Vec<(LtOperation, u64)> {
    let mut rows = Vec::new();
    merge_rows_into(&mut rows, ops.iter());
    rows
}

/// Each op's μ summed over `rows`.
fn totals(rows: &[(LtOperation, u64)]) -> HashMap<LtOperation, u64> {
    let mut out = HashMap::new();
    for (op, mu) in rows {
        *out.entry(op.clone()).or_insert(0) += mu;
    }
    out
}

/// Each op's count in `ops`.
fn counts(ops: &[LtOperation]) -> HashMap<LtOperation, u64> {
    let mut out = HashMap::new();
    for op in ops {
        *out.entry(op.clone()).or_insert(0) += 1;
    }
    out
}

fn assert_rows_hold(rows: &[(LtOperation, u64)], ops: &[LtOperation], what: &str) {
    for (op, mu) in rows {
        assert!(
            (1..=MU_MAX).contains(mu),
            "{what}: {op:?} has μ {mu}, outside 1..={MU_MAX}"
        );
    }
    assert_eq!(totals(rows), counts(ops), "{what}: each op's count");
}

/// A memory-like list: runs of one comparison (a load's bytes, an ECALL's
/// dwords), runs past `MU_MAX`, comparisons that come back after others, and
/// distinct ones; `n` ops from `seed`.
fn memory_like(n: usize, seed: u64) -> Vec<LtOperation> {
    let mut x = seed | 1;
    let mut step = || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x
    };
    let runs = [1usize, 1, 1, 2, 4, 8, 254, 255, 256, 600];
    let mut out = Vec::with_capacity(n);
    let mut recent: Vec<LtOperation> = Vec::new();
    let mut ts = 1_000u64;
    while out.len() < n {
        let r = step();
        let op = if r % 7 == 0 && !recent.is_empty() {
            // One that came before, coming back.
            recent[(r as usize / 7) % recent.len()].clone()
        } else {
            ts += 4;
            LtOperation::new(ts - 4 * (1 + r % 50), ts, false)
        };
        let times = runs[(r >> 8) as usize % runs.len()];
        for _ in 0..times.min(n - out.len()) {
            out.push(op.clone());
        }
        recent.push(op);
        if recent.len() > 16 {
            recent.remove(0);
        }
    }
    out
}

/// The CPU's kind: a few values repeated all over the list.
fn cpu_like(n: usize) -> Vec<LtOperation> {
    (0..n as u64)
        .map(|k| {
            let v = (k * 0x9e37_79b9) % 97;
            LtOperation::new_with_invert(v, v ^ 0x55, v % 3 == 0, v % 5 == 0)
        })
        .collect()
}

/// ★ Merged rows: a run of one op is rows of `MU_MAX` and the rest, an op
/// that comes back starts a row, every μ is in 1..=MU_MAX and each op's count
/// is kept; the list handed over a slice at a time merges as one.
#[test]
fn merged_rows_keep_each_ops_count_within_a_byte() {
    let a = LtOperation::new(3, 9, false);
    let b = LtOperation::new(4, 9, false);
    let run = |op: &LtOperation, n: usize| std::iter::repeat_n(op.clone(), n);
    let mus = |rows: &[(LtOperation, u64)]| rows.iter().map(|(_, mu)| *mu).collect::<Vec<_>>();

    assert!(merged(&[]).is_empty());
    assert_eq!(mus(&merged(&run(&a, 255).collect::<Vec<_>>())), vec![255]);
    assert_eq!(
        mus(&merged(&run(&a, 256).collect::<Vec<_>>())),
        vec![255, 1]
    );
    assert_eq!(
        mus(&merged(&run(&a, 600).collect::<Vec<_>>())),
        vec![255, 255, 90]
    );
    // A tuple that returns after another: a row of its own each time.
    let back: Vec<LtOperation> = run(&a, 2).chain(run(&b, 1)).chain(run(&a, 3)).collect();
    let rows = merged(&back);
    assert_eq!(rows, vec![(a.clone(), 2), (b.clone(), 1), (a.clone(), 3)]);
    assert_eq!(dedup_rows(&back).len(), 2, "the map's two rows");

    for seed in [1u64, 7, 99] {
        let ops = memory_like(50_000, seed);
        let rows = merged(&ops);
        assert_rows_hold(&rows, &ops, &format!("seed {seed}"));
        assert!(rows.len() >= dedup_rows(&ops).len());
        for slice in [1usize, 7, 4096] {
            let mut pieces = Vec::new();
            for part in ops.chunks(slice) {
                merge_rows_into(&mut pieces, part.iter());
            }
            assert_eq!(pieces, rows, "seed {seed}, slices of {slice}");
        }
    }
}

/// Every bus `trace` sends or receives on, the net of each tuple (sends
/// positive), with `interactions`.
fn nets(trace: &Table, interactions: &[BusInteraction]) -> BTreeMap<(u64, Vec<u64>), FE> {
    let mut net: BTreeMap<(u64, Vec<u64>), FE> = BTreeMap::new();
    for row in 0..trace.num_rows() {
        let g = |c: usize| *trace.get_main(row, c);
        for bi in interactions {
            let m = multiplicity(&bi.multiplicity, &g);
            if m == FE::zero() {
                continue;
            }
            let tuple: Vec<u64> = bi
                .values
                .iter()
                .flat_map(|v| v.combine_from(g))
                .map(|x: FE| x.canonical_u64())
                .collect();
            let e = net.entry((bi.bus_id, tuple)).or_insert(FE::zero());
            *e = if bi.is_sender { *e + m } else { *e - m };
        }
    }
    net.retain(|_, v| *v != FE::zero());
    net
}

/// ★ LT tables laid out from merged rows satisfy
/// LT's constraints, put on the ALU, MSB16 and IS_HALF buses exactly what the
/// map's rows put there, and send the μ bound once a row, padding rows
/// included ([`lt_rows_lookups`]) — the only difference, which the counters
/// follow. Over memory-like lists with runs past 255 and tuples that return.
#[test]
fn merged_tables_satisfy_lt_and_put_the_maps_tuples_on_the_buses() {
    let air = busless_air(cols::NUM_COLUMNS, LtConstraints);
    let interactions = crate::tables::lt::bus_interactions();
    let bound = u64::from(BusId::AreBytes);
    let mut differed = 0usize;
    for (seed, n) in [(3u64, 1_000usize), (11, 5_000), (23, 20_000)] {
        let mut ops = memory_like(n, seed);
        ops.extend(cpu_like(n / 4));
        let rows = merged(&ops);
        let map = dedup_rows(&ops);
        let (t_merged, t_map) = (
            lt::lt_trace_of_rows(&rows, TraceForm::Wide),
            lt::lt_trace_of_rows(&map, TraceForm::Wide),
        );
        assert!(
            validate_busless(&air, &t_merged),
            "seed {seed}: LT's constraints"
        );
        let split = |net: BTreeMap<(u64, Vec<u64>), FE>| {
            net.into_iter()
                .partition::<BTreeMap<_, _>, _>(|((bus, _), _)| *bus == bound)
        };
        let ((bound_a, a), (bound_b, b)) = (
            split(nets(&t_merged, &interactions)),
            split(nets(&t_map, &interactions)),
        );
        assert!(
            !a.is_empty(),
            "seed {seed}: the ALU, MSB16 and IS_HALF tuples"
        );
        assert_eq!(a, b, "seed {seed}: the nets but the μ bound's");
        // The μ bound's lookups are the merged rows', as the counters count them.
        let mut want: BTreeMap<(u64, Vec<u64>), FE> = BTreeMap::new();
        for lookup in lt_rows_lookups(&rows) {
            let e = want
                .entry((bound, vec![u64::from(lookup.x), 0]))
                .or_insert(FE::zero());
            *e += FE::one();
        }
        want.retain(|_, v| *v != FE::zero());
        assert_eq!(bound_a, want, "seed {seed}: the μ bound, a row each");
        differed += usize::from(bound_a != bound_b);
    }
    assert!(
        differed > 0,
        "the merged rows never differed from the map's"
    );
}

/// An LT list as the finish keeps it ([`LtLists`]' parts): the walk's ops in
/// two blocks, then the four memory parts (two compact), then HINT's.
struct Lists {
    walk: [Vec<LtOperation>; 2],
    memw: CompactLt,
    from_memw: Vec<LtOperation>,
    memw_a: CompactLt,
    from_memw_a: Vec<LtOperation>,
    hints: Vec<LtOperation>,
}

impl Lists {
    fn new(n: usize) -> Self {
        let mut memw = CompactLt::with_block_bytes(64);
        memw.extend(&memory_like(n, 5));
        let mut memw_a = CompactLt::default();
        memw_a.extend(&memory_like(n / 2, 9));
        Self {
            walk: [cpu_like(n / 3), cpu_like(n / 5)],
            memw,
            from_memw: memory_like(n / 7, 13),
            memw_a,
            from_memw_a: memory_like(n / 9, 17),
            hints: cpu_like(30),
        }
    }

    fn segments(&self) -> Segmented<'_, LtOperation> {
        Segmented {
            parts: vec![
                Part::Ops(&self.walk[0]),
                Part::Ops(&self.walk[1]),
                Part::Compact(&self.memw),
                Part::Ops(&self.from_memw),
                Part::Compact(&self.memw_a),
                Part::Ops(&self.from_memw_a),
                Part::Ops(&self.hints),
            ],
        }
    }
}

const MERGED: std::ops::Range<usize> = 2..6;

/// The μ histogram (index μ) of a table's MU column, padding rows included.
fn mu_column(t: &Table) -> Vec<u64> {
    let mut out = vec![0u64; 256];
    for row in 0..t.num_rows() {
        out[t.get_main(row, cols::MU).canonical_u64() as usize] += 1;
    }
    out
}

/// ★ A chunk's rows ([`lt_chunk_rows`]): without merged parts the map's rows
/// of the range, word for word; with them the map's rows of the walk's and
/// HINT's ops, then the memory parts' ops merged in list order, each op's
/// count kept and every μ in 1..=MU_MAX, the same rows each time. The chunk
/// sizes put chunk boundaries inside runs and across every part.
#[test]
fn a_chunks_rows_merge_only_the_memory_parts() {
    let lists = Lists::new(30_000);
    let list = lists.segments();
    let some = Some(MERGED);
    let mut split_run = false;
    for rows in [64usize, 1_000, 4_096, 1 << 15] {
        for (start, end) in list_chunks(list.len(), rows) {
            let what = format!("{start}..{end} at {rows}");
            let range = list.range(start, end);
            assert_eq!(
                lt_chunk_rows(&list, &None, start, end),
                dedup_rows(&range),
                "{what}: no merged parts"
            );
            let got = lt_chunk_rows(&list, &some, start, end);
            assert_rows_hold(&got, &range, &what);
            assert_eq!(
                got,
                lt_chunk_rows(&list, &some, start, end),
                "{what}: again"
            );
            // The map's part and the merged part, as the rule says.
            let (mut mapped, mut memory) = (Vec::new(), Vec::new());
            list.for_each_part_slice(start, end, 7, &mut |part, ops| {
                if MERGED.contains(&part) {
                    memory.extend_from_slice(ops)
                } else {
                    mapped.extend_from_slice(ops)
                }
            });
            let mut want = dedup_rows(&mapped);
            want.extend(merged(&memory));
            assert_eq!(got, want, "{what}: map, then merged");
            split_run |= end < list.len() && range.last() == list.range(end, end + 1).first();
        }
    }
    assert!(split_run, "no chunk boundary fell inside a run");
}

/// ★ Builder against counter: with merged parts, BITWISE's per-row lookups
/// counted on the pool (one map a chunk or by parts) and by each chunk's
/// collector are the μ column of the LT tables laid out from the chunks'
/// rows, chunk by chunk, chunk boundaries inside runs; and they differ from
/// what the map's rows would send, so the count reads the rule.
#[test]
fn the_per_row_lookups_counted_are_the_merged_tables_mu_column() {
    let lists = Lists::new(30_000);
    let list = lists.segments();
    let some = Some(MERGED);
    let empty: Vec<BranchOperation> = Vec::new();
    let no_bytewise: Vec<bytewise::BytewiseOperation> = Vec::new();
    let branch = Segmented {
        parts: vec![Part::Ops(&empty)],
    };
    let bytewise_s = Segmented {
        parts: vec![Part::Ops(&no_bytewise)],
    };
    for rows in [64usize, 4_096, 1 << 15] {
        let max_rows = MaxRowsConfig::uniform(rows);
        let mut laid_out = vec![0u64; 256];
        let mut collected = bitwise::BitwiseHistogram::new();
        let mut map_rows = vec![0u64; 256];
        for (start, end) in list_chunks(list.len(), rows) {
            let rows = lt_chunk_rows(&list, &some, start, end);
            let table = lt::lt_trace_of_rows(&rows, TraceForm::Wide);
            for (mu, n) in mu_column(&table).into_iter().enumerate() {
                laid_out[mu] += n;
            }
            collected.add_ops(&lt_rows_lookups(&rows));
            let map = dedup_rows(&list.range(start, end));
            for (mu, n) in mu_column(&lt::lt_trace_of_rows(&map, TraceForm::Wide))
                .into_iter()
                .enumerate()
            {
                map_rows[mu] += n;
            }
        }
        let mut want = bitwise::BitwiseHistogram::new();
        for (mu, n) in laid_out.iter().enumerate() {
            want.bump_n(
                BitwiseOperation::single_byte(BitwiseOperationType::AreBytes, mu as u8),
                *n,
            );
        }
        assert!(collected == want, "{rows} a chunk: the collectors");
        for by in [RowCount::Map, RowCount::Parts] {
            let mut counted = bitwise::BitwiseHistogram::new();
            count_row_lookups(&list, &some, &branch, &bytewise_s, &max_rows, by)
                .bump_into(&mut counted);
            assert!(counted == want, "{rows} a chunk, {by:?}: on the pool");
        }
        assert_ne!(laid_out, map_rows, "{rows} a chunk: the map's rows differ");
    }
}

// ---- builds ------------------------------------------------------------------------------------

type Configure = Box<dyn Fn(WindowedTraceBuilder<'_>) -> WindowedTraceBuilder<'_>>;

/// The windowed configurations, each without and with the merge: the
/// default's, the block's (streamed ops dropped, the per-row lookups on the
/// pool, by parts), and the raw, concatenated A arm (nothing merges).
fn configs() -> Vec<(&'static str, Configure, Configure)> {
    vec![
        ("plain", Box::new(|b| b), Box::new(|b| b.lt_merge())),
        (
            "dropped",
            Box::new(|b| b.drop_streamed_ops().expect("before any window")),
            Box::new(|b| b.drop_streamed_ops().expect("before any window").lt_merge()),
        ),
        (
            "dropped, rows on the pool",
            Box::new(|b| {
                b.drop_streamed_ops()
                    .expect("before any window")
                    .p4_rows_on_pool()
                    .p4_lookups_direct()
            }),
            Box::new(|b| {
                b.drop_streamed_ops()
                    .expect("before any window")
                    .p4_rows_on_pool()
                    .p4_lookups_direct()
                    .lt_merge()
            }),
        ),
        (
            "dropped, rows by parts",
            Box::new(|b| {
                b.drop_streamed_ops()
                    .expect("before any window")
                    .p4_rows_by_parts()
            }),
            Box::new(|b| {
                b.drop_streamed_ops()
                    .expect("before any window")
                    .p4_rows_by_parts()
                    .lt_merge()
            }),
        ),
        (
            "dropped, raw + concatenated LT",
            Box::new(|b| {
                b.drop_streamed_ops()
                    .expect("before any window")
                    .raw_memw_lt()
                    .expect("before any window")
                    .concat_lt()
            }),
            Box::new(|b| {
                b.drop_streamed_ops()
                    .expect("before any window")
                    .raw_memw_lt()
                    .expect("before any window")
                    .concat_lt()
                    .lt_merge()
            }),
        ),
    ]
}

/// Loops past `MU_MAX`, every instruction, unaligned accesses, comparisons,
/// loads, KECCAK and COPY (ECALL dwords).
const PROGRAMS: [&str; 7] = [
    "loop_4096",
    "all_instructions_64",
    "misalign_sd",
    "poc_slt_commit",
    "poc_lb_commit",
    "test_keccak_multi",
    "test_commit_split",
];

/// Every LT tuple of a build, with its μ summed over LT's tables.
fn lt_tuples(t: &Traces) -> BTreeMap<(u64, u64, bool, bool), u64> {
    let mut out = BTreeMap::new();
    for table in &t.lts {
        let g = |r: usize, c: usize| table.get_main(r, c).canonical_u64();
        for r in 0..table.num_rows() {
            let mu = g(r, cols::MU);
            if mu == 0 {
                continue;
            }
            let lhs = g(r, cols::LHS_0) | (g(r, cols::LHS_1) << 32) | (g(r, cols::LHS_2) << 48);
            let rhs = g(r, cols::RHS_0) | (g(r, cols::RHS_1) << 32) | (g(r, cols::RHS_2) << 48);
            let key = (lhs, rhs, g(r, cols::SIGNED) == 1, g(r, cols::INVERT) == 1);
            *out.entry(key).or_insert(0) += mu;
        }
    }
    out
}

/// ★ A windowed build with the merge against the same build without: every
/// table but LT's and BITWISE is the same word for word, the counts and what
/// the statement reads are the same, LT's tuples with their μ summed are the
/// same, and LT's tables do change somewhere (the test sees the knob).
#[test]
fn with_the_merge_only_lt_and_bitwise_change() {
    let mut changed = 0usize;
    for max_rows in [MaxRowsConfig::small(), MaxRowsConfig::uniform(4)] {
        for name in PROGRAMS {
            let (program, logs) = run(name);
            for window in [7, 1000] {
                for (config, today, merge) in &configs() {
                    let what = format!("{name} window {window} {config}");
                    let a = windowed(&program, &logs, &max_rows, window, today.as_ref());
                    let b = windowed(&program, &logs, &max_rows, window, merge.as_ref());
                    let keep = |(name, _): &(String, String)| {
                        !(name.starts_with("LT[") || name == "BITWISE")
                    };
                    let (ca, cb): (Vec<_>, Vec<_>) = (
                        contents(&a).into_iter().filter(keep).collect(),
                        contents(&b).into_iter().filter(keep).collect(),
                    );
                    assert_eq!(ca.len(), cb.len(), "{what}: the tables");
                    for ((n, x), (_, y)) in ca.iter().zip(&cb) {
                        assert_eq!(x, y, "{what}: {n}");
                    }
                    assert_eq!(lt_tuples(&a), lt_tuples(&b), "{what}: LT's tuples");
                    let (la, lb): (Vec<_>, Vec<_>) = (
                        contents(&a)
                            .into_iter()
                            .filter(|(n, _)| n.starts_with("LT["))
                            .collect(),
                        contents(&b)
                            .into_iter()
                            .filter(|(n, _)| n.starts_with("LT["))
                            .collect(),
                    );
                    if config.contains("concatenated") {
                        assert_eq!(la, lb, "{what}: nothing merges with LT concatenated");
                    }
                    changed += usize::from(la != lb);
                }
            }
        }
    }
    assert!(changed > 0, "the merge changed no LT table");
}

/// ★ Every inner bus balances on the merged builds (the μ bound's lookups
/// counted from the rows the tables were laid out from), and counting the
/// map's rows under merged tables (`lt_merge_count_fault`, the mutant)
/// leaves tuples on the μ bound's bus, and on no other, in some build: the
/// count is load-bearing.
#[test]
fn with_the_merge_every_inner_bus_balances_and_a_map_count_does_not() {
    let mut failed = Vec::new();
    let mut caught = 0usize;
    for max_rows in [MaxRowsConfig::small(), MaxRowsConfig::uniform(4)] {
        for name in PROGRAMS {
            let (program, logs) = run(name);
            for window in [7, 1000] {
                for (config, _, merge) in &configs() {
                    let what = format!("{name} window {window} {config}");
                    let mut t = windowed(&program, &logs, &max_rows, window, merge.as_ref());
                    let left = unbalanced(&program, &mut t);
                    if !left.is_empty() {
                        failed.push(format!("{what}: {}", named(&left)));
                    }
                    let mut t = windowed(&program, &logs, &max_rows, window, &|b| {
                        merge(b).lt_merge_count_fault()
                    });
                    let left = unbalanced(&program, &mut t);
                    let bound = u64::from(BusId::AreBytes);
                    assert!(
                        left.keys().all(|bus| *bus == bound),
                        "{what}: the fault left {}",
                        named(&left)
                    );
                    caught += usize::from(!left.is_empty());
                }
            }
        }
    }
    assert!(failed.is_empty(), "{}", failed.join("\n"));
    assert!(
        caught > 0,
        "no build caught the map's count under merged tables"
    );
}

/// ★ A MEMW op's comparisons are `old_timestamp[i] < timestamp` for each of
/// its `width` bytes (MEMW-C4…C7), in byte order; a MEMW_A row's is its one
/// pair. LT's memory list fed pairs ([`CompactLt::extend_pairs`]) holds the
/// ops fed whole would ([`CompactLt::extend`]), compact or raw.
#[test]
fn a_memory_accesss_comparisons_are_its_bytes_pairs() {
    let mut ops = Vec::new();
    for (k, width) in [1u8, 2, 4, 8].into_iter().enumerate() {
        let ts = 1_000 + 10 * k as u64;
        let op = MemwOperation {
            base_address: 0x1000 + 8 * k as u64,
            timestamp: ts,
            old_timestamp: std::array::from_fn(|i| ts - 1 - i as u64),
            value: [0; 8],
            old: [0; 8],
            is_register: false,
            width,
            is_read: k % 2 == 0,
        };
        let want: Vec<(u64, u64)> = (0..width as u64).map(|i| (ts - 1 - i, ts)).collect();
        assert_eq!(
            memw_lt_pairs(&op).collect::<Vec<_>>(),
            want,
            "width {width}"
        );
        ops.push(op);
    }
    let derived = collect_lt_from_memw(&ops);
    let pairs: Vec<(u64, u64)> = ops.iter().flat_map(memw_lt_pairs).collect();
    assert_eq!(derived.len(), 1 + 2 + 4 + 8);
    assert!(
        derived
            .iter()
            .zip(&pairs)
            .all(|(op, &(old, ts))| *op == LtOperation::new(old, ts, false))
    );
    for (mut whole, mut fed) in [
        (CompactLt::default(), CompactLt::default()),
        (
            CompactLt::with_block_bytes(40),
            CompactLt::with_block_bytes(40),
        ),
        (CompactLt::raw(), CompactLt::raw()),
    ] {
        whole.extend(&derived);
        fed.extend_pairs(pairs.iter().copied());
        let (mut a, mut b) = (Vec::new(), Vec::new());
        whole.expand_into(0, whole.len(), &mut a);
        fed.expand_into(0, fed.len(), &mut b);
        assert_eq!(a, derived);
        assert_eq!(b, derived);
    }
}
