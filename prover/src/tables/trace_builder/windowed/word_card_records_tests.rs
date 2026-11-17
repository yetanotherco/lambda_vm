//! The word memory format's families handed out in place
//! ([`WindowedTraceBuilder::word_chunks_in_place`]) and held as their rows for
//! the card to fill (`RowFamilies::words`), beside the in-row CPU chunks held
//! as their logs and every other row family: the build's tables, each records
//! chunk filled on the host, are the word build's without the card, every
//! table word for word, whatever the windows, walkers and room; the word
//! families asked alone are the only ones held; without the in-place hand-out
//! none is; and the test fault on their rows makes tables the run did not, so
//! an identity check reads them.

use executor::elf::Elf;
use executor::vm::logs::Log;

use super::card_records_tests::{Walk, always, for_first, held_by};
use super::{CardFillStamps, RecordsRoom, RowFamilies, Table, WindowedTraceBuilder};
use crate::Error;
use crate::tables::MaxRowsConfig;
use crate::tables::fused_memory::MemoryFormat;
use crate::tables::memw_aligned::MemwAlignedOrder;
use crate::tables::register::RegisterFormat;
use crate::tables::trace_builder::Traces;
use crate::tables::trace_builder::finish_tests::windowed;
use crate::tests::windowed_builder_tests::{run, same_traces};

/// Word caps of four rows (`mem_load`, `mem_store`), and the row families'
/// chunks of four, so that the small programs fill several, their rows in one
/// window part or many.
fn word_max() -> MaxRowsConfig {
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

/// The programs: the CPU's loads and stores of every width and alignment
/// (crossing words among them), and a run that calls COPY (MEMW_W's rows).
const PROGRAMS: [&str; 4] = [
    "fused_mem_cases",
    "all_loadstore_32",
    "misalign_sd",
    "test_copy_ecalls",
];

/// The block's formats with the word memory format: in-row registers, MEMW_A
/// in the step's limbs, KECCAK_RND in the block's form.
fn words(b: WindowedTraceBuilder<'_>) -> Result<WindowedTraceBuilder<'_>, Error> {
    b.register_format(RegisterFormat::InRow)?
        .memw_aligned_order(MemwAlignedOrder::Limbs)?
        .keccak_rnd_form(crate::block_whir::BlockFormat::production().keccak_rnd)
        .memory_format(MemoryFormat::Words)
}

/// The word build of the run without the card, its families built at the
/// finish.
fn reference(program: &Elf, logs: &[Log], max_rows: &MaxRowsConfig) -> Traces {
    windowed(program, logs, max_rows, 1000, &|b| {
        words(b).expect("before any window")
    })
}

/// The word build with its families streamed, its in-row CPU chunks held as
/// their logs and the families in `rows` as their rows while `room` admits
/// them, each test-faulted by `fault`, each records chunk filled on the host;
/// the word chunks handed out in place when `in_place`. How many chunks were
/// records, and the builder's stamps.
fn held_words(
    (program, logs): (&Elf, &[Log]),
    max_rows: &MaxRowsConfig,
    (window, walk): (usize, Walk),
    (room, rows, in_place): (RecordsRoom, RowFamilies, bool),
    fault: u64,
) -> (Traces, usize, CardFillStamps) {
    let builder = WindowedTraceBuilder::new(program, &[], max_rows)
        .and_then(words)
        .and_then(WindowedTraceBuilder::drop_streamed_ops)
        .and_then(WindowedTraceBuilder::stream_word_tables)
        .and_then(|b| {
            if in_place {
                b.word_chunks_in_place()
            } else {
                Ok(b)
            }
        })
        .and_then(|b| b.card_fill_streamed(room, true, false))
        .and_then(|b| b.card_fill_streamed_rows(rows))
        .expect("the builder")
        .card_fill_streamed_fault(fault);
    held_by(builder, program, logs, window, walk, rows.any())
}

/// The word families and every row family.
fn all_rows() -> RowFamilies {
    RowFamilies {
        words: true,
        ..RowFamilies::ALL
    }
}

/// The word families alone.
fn word_rows() -> RowFamilies {
    RowFamilies {
        words: true,
        ..RowFamilies::NONE
    }
}

/// Every row family but the word families: `[as records, on the host]`.
fn other_counts(s: &CardFillStamps) -> [[usize; 2]; 8] {
    [
        s.memw_r,
        s.memw_a,
        s.load,
        s.store,
        s.memw,
        s.shift,
        s.mem_load,
        s.mem_store,
    ]
}

/// ★ Every full word chunk handed out in place and held as its rows, beside
/// the in-row CPU chunks as their logs and every other row family as its rows,
/// fills on the host to the table the word build makes without the card: the
/// whole build, every table, word for word, windows of 1, 7 and 33 cycles
/// walked by the builder or on 1 and 4 threads.
#[test]
fn word_chunks_held_as_rows_are_the_runs_tables() {
    let max_rows = word_max();
    let mut walks = vec![Walk::Own];
    #[cfg(feature = "parallel")]
    walks.extend([Walk::Threads(1), Walk::Threads(4)]);
    let mut held_most = 0usize;
    for name in PROGRAMS {
        let (program, logs) = run(name);
        let want = reference(&program, &logs, &max_rows);
        for window in [1, 7, 33] {
            for &walk in &walks {
                let what = format!("{name}, window {window}, {walk:?}");
                let (got, records, s) = held_words(
                    (&program, &logs),
                    &max_rows,
                    (window, walk),
                    (always(), all_rows(), true),
                    0,
                );
                same_traces(&want, &got);
                assert_eq!(s.room_out, None, "{what}");
                assert_eq!(s.words[1], 0, "{what}: {s:?}");
                assert_eq!(
                    records,
                    s.cpu[0] + s.words[0] + other_counts(&s).iter().map(|f| f[0]).sum::<usize>(),
                    "{what}"
                );
                held_most = held_most.max(s.words[0]);
            }
        }
    }
    assert!(held_most > 4, "{held_most} word chunks held at most");
}

/// ★ The word families asked alone are the only row families held, and they
/// are held only when handed out in place: copied out, each is filled on the
/// host as before, uncounted. The tables are the run's either way.
#[test]
fn the_word_families_asked_alone_are_the_only_ones_held() {
    let max_rows = word_max();
    let (program, logs) = run("fused_mem_cases");
    let want = reference(&program, &logs, &max_rows);
    for in_place in [true, false] {
        let (got, _, s) = held_words(
            (&program, &logs),
            &max_rows,
            (7, Walk::Own),
            (always(), word_rows(), in_place),
            0,
        );
        same_traces(&want, &got);
        if in_place {
            assert!(s.words[0] > 0 && s.words[1] == 0, "{s:?}");
        } else {
            assert_eq!(s.words, [0, 0], "copied out, none is held: {s:?}");
        }
        assert!(
            other_counts(&s).iter().all(|f| *f == [0, 0]),
            "{in_place}: {s:?}"
        );
    }
}

/// ★ The room rule reads the word chunks as it reads any family's: whatever
/// chunk it first refuses, the tables are the run's, and the word chunks held
/// and filled on the host add up to the same count; refused from the first
/// chunk on, none is held.
#[test]
fn the_room_rule_sends_word_chunks_to_the_host() {
    let max_rows = word_max();
    let (program, logs) = run("all_loadstore_32");
    let want = reference(&program, &logs, &max_rows);
    let mut totals = Vec::new();
    let mut split = false;
    for n in [0, 3, 12, 40] {
        let (got, _, s) = held_words(
            (&program, &logs),
            &max_rows,
            (7, Walk::Own),
            (for_first(n, 1), word_rows(), true),
            0,
        );
        same_traces(&want, &got);
        totals.push(s.words[0] + s.words[1]);
        if n == 0 {
            assert_eq!(s.words[0], 0, "{s:?}");
        }
        split |= s.words[0] > 0 && s.words[1] > 0;
    }
    assert!(totals.windows(2).all(|w| w[0] == w[1]), "{totals:?}");
    assert!(totals[0] > 1, "{totals:?}");
    assert!(split, "some cut leaves word chunks on both sides");
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

/// ★ The mutant an identity check must catch: each word row held as a record
/// moved `cycles` later (the test fault) fills its family's tables with rows
/// the run did not make, while a row family not asked stays the run's.
/// Faulted by 0, the build is the run's.
#[test]
fn the_word_rows_fault_makes_tables_the_run_did_not() {
    let max_rows = word_max();
    let (program, logs) = run("fused_mem_cases");
    let want = reference(&program, &logs, &max_rows);
    let build = |fault| {
        held_words(
            (&program, &logs),
            &max_rows,
            (7, Walk::Own),
            (always(), word_rows(), true),
            fault,
        )
        .0
    };
    same_traces(&want, &build(0));
    let got = build(4);
    let moved: usize = (0..7)
        .map(|k| rows_differ(&got.word_tables[k], &want.word_tables[k]))
        .sum();
    assert!(moved > 0, "the fault moved no word row");
    // A row family not asked stays the run's (the fault also moves the CPU
    // chunks held as logs, its own family's test).
    assert_eq!(rows_differ(&got.shifts, &want.shifts), 0, "SHIFT moved");
}
