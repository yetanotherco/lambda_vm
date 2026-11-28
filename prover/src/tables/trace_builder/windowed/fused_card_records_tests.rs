//! MEM_LOAD's and MEM_STORE's chunks (the fused memory format) handed out as
//! their rows for the card to fill ([`WindowedTraceBuilder::card_fill_streamed_rows`]),
//! beside the in-row CPU chunks held as their logs and every other row
//! family: the build's tables, each records chunk filled on the host, are the
//! fused build's without the card, every table word for word, whatever the
//! windows, walkers and room; each family asked alone is the only one held; a
//! chunk dropped as its rows comes back as them; and the test fault on their
//! rows makes tables the run did not, so an identity check reads them.

use std::collections::HashMap;

use executor::elf::Elf;
use executor::vm::logs::Log;

use super::card_records_tests::{Walk, always, for_first, held_by};
use super::{CardFillStamps, RecordsRoom, RowFamilies, StreamTable, Table, WindowedTraceBuilder};
use crate::Error;
use crate::tables::MaxRowsConfig;
use crate::tables::fused_memory::MemoryFormat;
use crate::tables::memw_aligned::MemwAlignedOrder;
use crate::tables::register::RegisterFormat;
use crate::tables::trace_builder::Traces;
use crate::tables::trace_builder::finish_tests::windowed;
use crate::tests::windowed_builder_tests::{run, same_traces};

/// Chunks of four rows for the row families and the fused tables, so that the
/// small programs fill several, their rows in one window part or many.
fn fused_max() -> MaxRowsConfig {
    MaxRowsConfig {
        memw_aligned: 1 << 2,
        load: 1 << 2,
        store: 1 << 2,
        memw: 1 << 2,
        shift: 1 << 2,
        mem_load: 1 << 2,
        mem_store: 1 << 2,
        ..MaxRowsConfig::small()
    }
}

/// The programs: every width, sign and alignment of the CPU's loads and
/// stores, every instruction, loads and stores at every width, BLAKE3's
/// memory rows.
const PROGRAMS: [&str; 4] = [
    "fused_mem_cases",
    "all_loadstore_32",
    "all_instructions_64",
    "test_blake3",
];

/// The block's formats with the fused memory format: in-row registers, MEMW_A
/// in the step's limbs.
fn fused(b: WindowedTraceBuilder<'_>) -> Result<WindowedTraceBuilder<'_>, Error> {
    b.register_format(RegisterFormat::InRow)?
        .memw_aligned_order(MemwAlignedOrder::Limbs)?
        .memory_format(MemoryFormat::Fused)
}

/// The fused build of the run without the card.
fn reference(program: &Elf, logs: &[Log], max_rows: &MaxRowsConfig) -> Traces {
    windowed(program, logs, max_rows, 1000, &|b| {
        fused(b).expect("before any window")
    })
}

/// The fused build with its in-row CPU chunks held as their logs and the
/// families in `rows` as their rows while `room` admits them, each test-
/// faulted by `fault` ([`WindowedTraceBuilder::card_fill_streamed_fault`]),
/// each records chunk filled on the host; how many chunks were records; and
/// the builder's stamps.
fn held_fused(
    (program, logs): (&Elf, &[Log]),
    max_rows: &MaxRowsConfig,
    (window, walk): (usize, Walk),
    room: RecordsRoom,
    rows: RowFamilies,
    fault: u64,
) -> (Traces, usize, CardFillStamps) {
    let builder = WindowedTraceBuilder::new(program, &[], max_rows)
        .and_then(fused)
        .and_then(WindowedTraceBuilder::drop_streamed_ops)
        .and_then(|b| b.card_fill_streamed(room, true, false))
        .and_then(|b| b.card_fill_streamed_rows(rows))
        .expect("the builder")
        .card_fill_streamed_fault(fault);
    held_by(builder, program, logs, window, walk, rows.any())
}

/// The fused families' `[as records, on the host]`: MEM_LOAD, MEM_STORE.
fn fused_counts(s: &CardFillStamps) -> [[usize; 2]; 2] {
    [s.mem_load, s.mem_store]
}

/// Every family but CPU and the fused two: `[as records, on the host]`.
fn other_counts(s: &CardFillStamps) -> [[usize; 2]; 6] {
    [s.memw_r, s.memw_a, s.load, s.store, s.memw, s.shift]
}

/// ★ Every full MEM_LOAD and MEM_STORE chunk handed out as its rows (32 B a
/// load, 40 a store), beside the in-row CPU chunks as their logs and every
/// other row family as its rows, fills on the host to the table the fused
/// build makes without the card: the whole build, every table, word for word,
/// windows of 1, 7 and 33 cycles walked by the builder or on 1 and 4 threads.
#[test]
fn fused_chunks_held_as_rows_are_the_runs_tables() {
    let max_rows = fused_max();
    let mut walks = vec![Walk::Own];
    #[cfg(feature = "parallel")]
    walks.extend([Walk::Threads(1), Walk::Threads(4)]);
    let mut held_most = [0usize; 2];
    for name in PROGRAMS {
        let (program, logs) = run(name);
        let want = reference(&program, &logs, &max_rows);
        for window in [1, 7, 33] {
            for &walk in &walks {
                let what = format!("{name}, window {window}, {walk:?}");
                let (got, records, s) = held_fused(
                    (&program, &logs),
                    &max_rows,
                    (window, walk),
                    always(),
                    RowFamilies::ALL,
                    0,
                );
                same_traces(&want, &got);
                assert_eq!(s.room_out, None, "{what}");
                assert_eq!(s.out_of_step, 0, "{what}");
                let fused = fused_counts(&s);
                assert!(
                    fused.iter().chain(&other_counts(&s)).all(|f| f[1] == 0),
                    "{what}: {s:?}"
                );
                assert_eq!(s.memw_r, [0, 0], "{what}: MEMW_R is in-row's residual");
                assert_eq!(
                    records,
                    s.cpu[0]
                        + fused
                            .iter()
                            .chain(&other_counts(&s))
                            .map(|f| f[0])
                            .sum::<usize>(),
                    "{what}"
                );
                assert_eq!(
                    s.records_bytes,
                    s.cpu[0] * max_rows.cpu * (40 + 20)
                        + s.memw_a[0] * max_rows.memw_aligned * 48
                        + s.load[0] * max_rows.load * 32
                        + s.store[0] * max_rows.store * 32
                        + s.memw[0] * max_rows.memw * 152
                        + s.shift[0] * max_rows.shift * 24
                        + s.mem_load[0] * max_rows.mem_load * 32
                        + s.mem_store[0] * max_rows.mem_store * 40,
                    "{what}: 32 B a MEM_LOAD row, 40 a MEM_STORE row"
                );
                for (most, f) in held_most.iter_mut().zip(fused) {
                    *most = (*most).max(f[0]);
                }
            }
        }
    }
    assert!(held_most.iter().all(|&n| n > 1), "{held_most:?}");
}

/// ★ MEM_LOAD or MEM_STORE asked alone is the only row family held: the
/// other fused family and every other row family are filled on the host,
/// uncounted, and the tables are the run's.
#[test]
fn a_fused_family_asked_alone_is_the_only_one_held() {
    let max_rows = fused_max();
    let (program, logs) = run("fused_mem_cases");
    let want = reference(&program, &logs, &max_rows);
    for k in 0..2 {
        let mut rows = RowFamilies::NONE;
        *[&mut rows.mem_load, &mut rows.mem_store][k] = true;
        let (got, _, s) = held_fused(
            (&program, &logs),
            &max_rows,
            (7, Walk::Own),
            always(),
            rows,
            0,
        );
        same_traces(&want, &got);
        let fused = fused_counts(&s);
        assert!(fused[k][0] > 0 && fused[k][1] == 0, "family {k}: {s:?}");
        assert_eq!(fused[1 - k], [0, 0], "family {k}: {s:?}");
        assert!(
            other_counts(&s).iter().all(|f| *f == [0, 0]),
            "family {k}: {s:?}"
        );
    }
}

/// ★ The room rule reads the fused chunks as it reads any family's: whatever
/// chunk it first refuses, the tables are the run's, and the fused chunks
/// held and filled on the host add up to the same count; refused from the
/// first chunk on, none is held.
#[test]
fn the_room_rule_sends_fused_chunks_to_the_host() {
    let max_rows = fused_max();
    let (program, logs) = run("all_loadstore_32");
    let want = reference(&program, &logs, &max_rows);
    let mut totals = Vec::new();
    let mut split = false;
    for n in [0, 3, 12, 40] {
        let (got, _, s) = held_fused(
            (&program, &logs),
            &max_rows,
            (7, Walk::Own),
            for_first(n, 1),
            RowFamilies::ALL,
            0,
        );
        same_traces(&want, &got);
        let fused = fused_counts(&s);
        totals.push(fused.map(|f| f[0] + f[1]));
        if n == 0 {
            assert!(fused.iter().all(|f| f[0] == 0), "{s:?}");
        }
        split |= fused.iter().any(|f| f[0] > 0 && f[1] > 0);
    }
    assert!(totals.windows(2).all(|w| w[0] == w[1]), "{totals:?}");
    assert!(totals[0].iter().all(|&n| n > 1), "{totals:?}");
    assert!(split, "some cut leaves fused chunks on both sides");
}

/// Rows of `a` and `b` that differ, over every table of each list.
fn rows_differ(a: &[Table], b: &[Table]) -> usize {
    assert_eq!(a.len(), b.len(), "table count");
    a.iter()
        .zip(b)
        .map(|(x, y)| {
            assert_eq!(x.main_table.height, y.main_table.height, "height");
            (0..x.main_table.height)
                .filter(|&r| x.main_table.get_row(r) != y.main_table.get_row(r))
                .count()
        })
        .sum()
}

/// ★ The mutant an identity check must catch: each fused row held as a
/// record moved `cycles` later (the test fault) fills MEM_LOAD's or
/// MEM_STORE's tables with rows the run did not make, while the family not
/// asked stays the run's. Faulted by 0, the build is the run's.
#[test]
fn the_fused_rows_fault_makes_tables_the_run_did_not() {
    let max_rows = fused_max();
    let (program, logs) = run("fused_mem_cases");
    let want = reference(&program, &logs, &max_rows);
    for k in 0..2 {
        let mut rows = RowFamilies::NONE;
        *[&mut rows.mem_load, &mut rows.mem_store][k] = true;
        let build = |fault| {
            held_fused(
                (&program, &logs),
                &max_rows,
                (7, Walk::Own),
                always(),
                rows,
                fault,
            )
            .0
        };
        same_traces(&want, &build(0));
        let got = build(4);
        let lists = |t: &Traces| [t.mem_loads.clone(), t.mem_stores.clone()];
        let (got, want) = (lists(&got), lists(&want));
        assert!(
            rows_differ(&got[k], &want[k]) > 0,
            "family {k}: the fault moved no row"
        );
        assert_eq!(
            rows_differ(&got[1 - k], &want[1 - k]),
            0,
            "family {k}: the family not asked moved"
        );
    }
}

/// ★ A fused chunk dropped as its rows comes back as them: walked again by
/// the regenerator, each MEM_LOAD and MEM_STORE chunk phase A held as its rows
/// is rebuilt with the digest of those rows, its packed columns their host
/// fill; a records slot takes them, and refuses another chunk of the family's
/// columns. Not dropped as records, the same columns with no digest.
#[test]
fn a_fused_chunk_dropped_as_its_rows_is_rebuilt_as_them() {
    use super::RegenBuilder;
    use crate::tables::gpack::TraceForm;
    use stark::regen::{RegenError, RegenWindow};

    let max_rows = fused_max();
    let mut crossed = [0usize; 2];
    for name in ["fused_mem_cases", "all_loadstore_32"] {
        let (program, logs) = run(name);
        let window = 7;
        let body = logs.len() - 1;
        let cut = body - body % window;
        let mut held = HashMap::new();
        {
            let mut rows = RowFamilies::NONE;
            rows.mem_load = true;
            rows.mem_store = true;
            let mut builder = WindowedTraceBuilder::new(&program, &[], &max_rows)
                .and_then(fused)
                .and_then(WindowedTraceBuilder::drop_streamed_ops)
                .and_then(|b| b.card_fill_streamed(always(), false, false))
                .and_then(|b| b.card_fill_streamed_rows(rows))
                .expect("the builder");
            let (mut walker, mut accumulator) = builder.split();
            for w in logs[..cut].chunks(window) {
                let walked = walker.walk_owned(w.to_vec()).expect("a window");
                for job in accumulator.absorb(walked) {
                    let (table, index) = (job.table, job.index);
                    if let Some(records) = job.generate().trace.take_records_main() {
                        held.insert((table, index), records);
                    }
                }
            }
        }
        assert!(
            held.keys()
                .all(|(t, _)| matches!(t, StreamTable::MemLoad | StreamTable::MemStore)),
            "{name}: only the fused families held"
        );
        let replay = |as_records: bool| {
            let mut regen = RegenBuilder::new(&program, &[], &max_rows)
                .and_then(|b| b.register_format(RegisterFormat::InRow))
                .and_then(|b| b.memw_aligned_order(MemwAlignedOrder::Limbs))
                .and_then(|b| b.memory_format(MemoryFormat::Fused))
                .expect("the regenerator");
            let mut rebuilt = HashMap::new();
            for w in logs.chunks(window) {
                for job in regen.push_owned(w.to_vec()).expect("a window") {
                    let (table, index) = (job.table, job.index);
                    let (digest, chunk) = job.regenerate(TraceForm::Narrow, as_records);
                    let mut trace = chunk.trace;
                    trace.pack_main_narrow();
                    rebuilt.insert((table, index), (digest, trace.take_narrow_main()));
                }
            }
            rebuilt
        };
        let rebuilt = replay(true);
        let (slots, _producer) = RegenWindow::new(1 << 30);
        for (rank, (key, records)) in held.iter().enumerate() {
            let what = format!("{name} {key:?}");
            let (digest, packed) = rebuilt.get(key).cloned().expect("rebuilt");
            let packed = packed.expect("packed");
            assert_eq!(digest, Some(records.digest()), "{what}: the rows' digest");
            let host = records.host_fill().expect("the host fills phase A's rows");
            assert_eq!(
                (packed.widths(), packed.data()),
                (host.widths(), host.data()),
                "{what}: the packed columns"
            );
            slots
                .slot_for_records(records, rank as u64)
                .deposit_checked(packed, digest.expect("a digest"))
                .unwrap_or_else(|e| panic!("{what}: {e:?}"));
        }
        for (family, crossed) in [StreamTable::MemLoad, StreamTable::MemStore]
            .into_iter()
            .zip(&mut crossed)
        {
            let mut keys: Vec<_> = held.keys().filter(|(t, _)| *t == family).collect();
            keys.sort_by_key(|(_, index)| *index);
            let [own, other, ..] = keys[..] else {
                continue;
            };
            *crossed += 1;
            let (digest, packed) = rebuilt.get(other).cloned().expect("rebuilt");
            let slot = slots.slot_for_records(&held[own], held.len() as u64);
            assert_eq!(
                slot.deposit_checked(packed.expect("packed"), digest.expect("a digest")),
                Err(RegenError::Mismatch),
                "{name}: {other:?}'s columns for {own:?}"
            );
        }
        let plain = replay(false);
        for key in held.keys() {
            let (digest, packed) = plain.get(key).cloned().expect("rebuilt");
            assert_eq!(digest, None, "{name} {key:?}");
            assert_eq!(
                packed.map(|p| p.data().to_vec()),
                rebuilt[key].1.as_ref().map(|p| p.data().to_vec()),
                "{name} {key:?}: the columns"
            );
        }
    }
    assert!(crossed.iter().all(|&n| n > 0), "{crossed:?}");
}
