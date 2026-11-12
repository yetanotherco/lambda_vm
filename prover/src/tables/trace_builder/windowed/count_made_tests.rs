//! Routing and counting where the rows are made
//! (`WindowedTraceBuilder::count_where_made`): a build whose walkers route each
//! batch and count its rows builds the whole-run tables, through the split,
//! from the windows' entries and from the executor's history, the hand-out's
//! counting on its thread or not, at chunk sizes that hand out every streamed
//! table; and a window absorbed without its walker's counts is refused.

use crate::tables::MaxRowsConfig;
use crate::tables::trace_builder::Traces;
use crate::tests::windowed_builder_tests::{run, same_traces, whole};

use super::WindowedTraceBuilder;

const PROGRAMS: [&str; 8] = [
    "all_instructions_64",
    "lw_sw_offset_odd",
    "test_memw_split_ts",
    "test_keccak_multi",
    "test_ecsm",
    "test_ecrec",
    "test_copy_ecalls",
    "test_commit_split",
];

/// The chunk sizes: the tests' small ones, CPU chunks that pad, and chunks of
/// four rows (one op for the rarer tables), so every streamed table hands
/// out chunks.
fn configs() -> [(&'static str, MaxRowsConfig); 3] {
    [
        ("small", MaxRowsConfig::small()),
        (
            "padded",
            MaxRowsConfig {
                cpu: 24,
                ..MaxRowsConfig::small()
            },
        ),
        (
            "one op",
            MaxRowsConfig {
                memw: 1,
                load: 1,
                shift: 1,
                store: 1,
                ..MaxRowsConfig::uniform(4)
            },
        ),
    ]
}

/// The windowed build of `program`'s run over windows of `window` cycles,
/// dropping its streamed ops, the walkers counting where they make the rows,
/// split (walker and accumulator) or through `push`, the hand-out's counting
/// on a thread of its own or not; and the windows its walkers counted.
fn made_build(
    program: &executor::elf::Elf,
    logs: &[executor::vm::logs::Log],
    max_rows: &MaxRowsConfig,
    window: usize,
    split: bool,
    count_thread: bool,
    tamper: bool,
) -> (Result<Traces, crate::Error>, usize) {
    let mut builder = WindowedTraceBuilder::new(program, &[], max_rows)
        .expect("the builder")
        .count_handout_on_thread(count_thread)
        .drop_streamed_ops()
        .expect("before the first window")
        .count_where_made(true);
    assert!(builder.counts_where_made(), "the build counts where made");
    let body = logs.len() - 1;
    let cut = body - body % window;
    let (mut chunks, mut counted) = (Vec::new(), 0);
    for (k, w) in logs[..cut].chunks(window).enumerate() {
        if split || tamper {
            let (mut walker, mut accumulator) = builder.split();
            if tamper && k == 0 {
                // A walker that counts nothing: its window comes without.
                walker.counts = None;
            }
            let walked = walker.walk(w).expect("a window");
            counted += usize::from(walked.made.is_some());
            chunks.extend(
                accumulator
                    .absorb(walked)
                    .into_iter()
                    .map(|job| job.generate()),
            );
        } else {
            chunks.extend(builder.push(w).expect("a window"));
            counted += 1;
        }
    }
    let traces = builder.finish(&logs[cut..]).and_then(|mut traces| {
        traces.insert_streamed(chunks)?;
        Ok(traces)
    });
    (traces, counted)
}

/// ★ Counted where the rows are made, through the split and through `push`, the hand-out's counting on its thread
/// or not, at every window length and the three chunk sizes, the windows
/// build the whole-run tables.
#[test]
fn counting_where_made_builds_the_whole_run_tables() {
    let _ecrec = crate::tables::trace_builder::ecrec_forge_free();
    for (what, max_rows) in configs() {
        for name in PROGRAMS {
            let (program, logs) = run(name);
            let reference = whole(&program, &logs, &max_rows);
            for window in [1, 7, 33, 1000] {
                for (split, count_thread) in
                    [(true, true), (true, false), (false, true), (false, false)]
                {
                    let (traces, counted) = made_build(
                        &program,
                        &logs,
                        &max_rows,
                        window,
                        split,
                        count_thread,
                        false,
                    );
                    let traces = traces.unwrap_or_else(|e| {
                        panic!("{what} {name}, window {window}, split {split}: {e:?}")
                    });
                    same_traces(&reference, &traces);
                    let body = (logs.len() - 1) / window;
                    if split {
                        assert_eq!(counted, body, "{name}: every window counted by its walker");
                    }
                }
            }
        }
    }
}

/// ★ Under the mode, a window absorbed without its walker's counts (its rows
/// would go uncounted) is refused at the finish.
#[test]
fn a_window_without_its_walkers_counts_is_refused() {
    let (program, logs) = run("all_instructions_64");
    let (traces, counted) = made_build(
        &program,
        &logs,
        &MaxRowsConfig::small(),
        7,
        true,
        false,
        true,
    );
    assert!(counted > 0);
    match traces {
        Err(crate::Error::Prover(e)) => {
            assert!(e.contains("without their walker's counts"), "{e}")
        }
        Err(e) => panic!("another error: {e:?}"),
        Ok(_) => panic!("not refused"),
    }
}

/// Without [`WindowedTraceBuilder::drop_streamed_ops`] (a build that keeps its
/// windows) the mode does not apply, and the build is today's.
#[test]
fn counting_where_made_applies_only_to_a_build_that_drops_its_ops() {
    let (program, logs) = run("all_instructions_64");
    let max_rows = MaxRowsConfig::small();
    let builder = WindowedTraceBuilder::new(&program, &[], &max_rows)
        .expect("the builder")
        .count_where_made(true);
    assert!(!builder.counts_where_made());
    let reference = whole(&program, &logs, &max_rows);
    let mut builder = builder;
    let mut chunks = Vec::new();
    let body = logs.len() - 1;
    let cut = body - body % 7;
    for w in logs[..cut].chunks(7) {
        chunks.extend(builder.push(w).expect("a window"));
    }
    let mut traces = builder.finish(&logs[cut..]).expect("the finish");
    traces.insert_streamed(chunks).expect("placeholders");
    same_traces(&reference, &traces);
}

/// The windowed build of `program`'s run over windows of `window` cycles,
/// split, dropping its streamed ops, with LT's memory comparisons merged
/// (`lt_merge`); with `levers`, also the ECREC witnesses deferred and the rows
/// counted where made (the accumulator routing).
fn merged_build(
    program: &executor::elf::Elf,
    logs: &[executor::vm::logs::Log],
    max_rows: &MaxRowsConfig,
    window: usize,
    levers: bool,
) -> Traces {
    let mut builder = WindowedTraceBuilder::new(program, &[], max_rows)
        .expect("the builder")
        .drop_streamed_ops()
        .expect("before the first window")
        .lt_merge()
        .defer_ecrec_witnesses(levers)
        .count_where_made(levers);
    assert_eq!(builder.counts_where_made(), levers);
    let body = logs.len() - 1;
    let cut = body - body % window;
    let mut chunks = Vec::new();
    for w in logs[..cut].chunks(window) {
        let (mut walker, mut accumulator) = builder.split();
        let walked = walker.walk(w).expect("a window");
        chunks.extend(
            accumulator
                .absorb(walked)
                .into_iter()
                .map(|job| job.generate()),
        );
    }
    let mut traces = builder.finish(&logs[cut..]).expect("the finish");
    traces.insert_streamed(chunks).expect("placeholders");
    traces
}

/// ★ The three levers stacked: with LT's memory comparisons merged, the ECREC
/// witnesses deferred and the rows counted where made build what the merge
/// alone builds, table for table, at the small chunks and at one op a chunk.
#[test]
fn the_levers_stack_on_the_lt_merge() {
    let _ecrec = crate::tables::trace_builder::ecrec_forge_free();
    for (_, max_rows) in configs() {
        for name in [
            "test_ecrec",
            "test_ecsm",
            "all_instructions_64",
            "test_memw_split_ts",
        ] {
            let (program, logs) = run(name);
            for window in [7, 33] {
                let reference = merged_build(&program, &logs, &max_rows, window, false);
                let stacked = merged_build(&program, &logs, &max_rows, window, true);
                same_traces(&reference, &stacked);
            }
        }
    }
}
