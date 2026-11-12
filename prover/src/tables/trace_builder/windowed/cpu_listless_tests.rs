//! Walking without the CPU list (`WindowedTraceBuilder::drop_cpu_list`): with
//! the card filling CPU from the logs, the walkers keep no list of a window's
//! CPU ops; the accumulator routes from the logs, each CPU chunk is cut from
//! its logs and, where the host fills it, its ops are rebuilt from them; the
//! finish rebuilds its tail's. The tables are the whole-run build's, through
//! the split walker and on threads, whatever the room says; and the mode
//! changes nothing where the card does not fill CPU.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use executor::elf::Elf;
use executor::vm::logs::Log;
use math::field::element::FieldElement;

use crate::tables::MaxRowsConfig;
use crate::tables::trace_builder::{StreamedChunk, Traces};
use crate::tables::types::GoldilocksField;
use crate::tests::windowed_builder_tests::{run, same_traces, whole};

use super::{ChunkJob, RecordsRoom, Table, WindowedTraceBuilder};

/// `chunk` with its records, if it holds them, replaced by their host fill at
/// eight bytes a cell (as the card fills them).
fn unheld(mut chunk: StreamedChunk) -> StreamedChunk {
    let Some(records) = chunk.trace.take_records_main() else {
        return chunk;
    };
    let packed = records.host_fill().expect("the host fills its records");
    let columns = packed
        .widen::<GoldilocksField>()
        .expect("the packed columns widen");
    let (rows, cols) = (records.rows(), records.cols());
    let mut data: Vec<FieldElement<GoldilocksField>> = Vec::with_capacity(rows * cols);
    for r in 0..rows {
        data.extend(columns.iter().map(|column| column.evals()[r]));
    }
    chunk.trace = Table::new_main(data, cols, 1);
    chunk
}

/// The room rules: every chunk, none (every CPU chunk filled on the host
/// from its rebuilt ops), and every other one.
fn rooms() -> Vec<(&'static str, RecordsRoom)> {
    let every_other = AtomicUsize::new(0);
    vec![
        ("always", Arc::new(|_| true) as RecordsRoom),
        ("never", Arc::new(|_| false) as RecordsRoom),
        (
            "every other",
            Arc::new(move |_| {
                every_other
                    .fetch_add(1, Ordering::Relaxed)
                    .is_multiple_of(2)
            }) as RecordsRoom,
        ),
    ]
}

/// How the windows are walked.
#[derive(Clone, Copy, Debug)]
enum Walk {
    /// The split walker, each window's logs handed over owned.
    Owned,
    /// On this many threads, from the windows' entries.
    Threads(usize),
}

/// The run of `program` by windows of `window` cycles, dropping its streamed
/// ops, the card filling CPU and MEMW_R while `room` admits (`cpu`: CPU too),
/// the walkers counting where made and, with `listless`, keeping no CPU list;
/// every chunk then filled on the host and put back. With the windows the
/// walkers kept no list of.
#[allow(clippy::too_many_arguments)]
fn built(
    program: &Elf,
    logs: &[Log],
    max_rows: &MaxRowsConfig,
    window: usize,
    walk: Walk,
    room: RecordsRoom,
    cpu: bool,
    listless: bool,
) -> (Traces, usize) {
    let mut builder = WindowedTraceBuilder::new(program, &[], max_rows)
        .and_then(WindowedTraceBuilder::drop_streamed_ops)
        .and_then(|b| b.card_fill_streamed(room, cpu, true))
        .expect("the builder")
        .count_where_made(true)
        .drop_cpu_list(listless);
    assert_eq!(builder.cpu_listless(), listless && cpu, "the mode applies");
    let mut jobs: Vec<ChunkJob> = Vec::new();
    let mut without = 0usize;
    let last = match walk {
        Walk::Owned => {
            let body = logs.len() - 1;
            let cut = body - body % window;
            let (mut walker, mut accumulator) = builder.split();
            for w in logs[..cut].chunks(window) {
                let walked = walker.walk_owned(w.to_vec()).expect("a window");
                without += usize::from(walked.cpu_ops.is_empty() && walked.cycles > 0);
                jobs.extend(accumulator.absorb(walked));
            }
            logs[cut..].to_vec()
        }
        Walk::Threads(walkers) => {
            use executor::vm::execution::Executor;
            let (tx, rx) = std::sync::mpsc::sync_channel(2);
            let report = std::thread::scope(|scope| {
                scope.spawn(move || {
                    let mut executor = Executor::new(program, Vec::new()).expect("the executor");
                    let mut start = Some(executor.memory().snapshot());
                    while let Some(logs) = executor
                        .resume_with_limit(window)
                        .expect("the run")
                        .map(<[Log]>::to_vec)
                    {
                        let next = Some(executor.memory().snapshot());
                        if tx
                            .send((logs, std::mem::replace(&mut start, next)))
                            .is_err()
                        {
                            break;
                        }
                    }
                });
                super::parallel::walk_windows(&mut builder, rx, walkers, |accumulator, w, _| {
                    without += usize::from(w.cpu_ops.is_empty() && w.cycles > 0);
                    jobs.extend(accumulator.absorb(w));
                    Ok(())
                })
            })
            .expect("the walk");
            report.last
        }
    };
    let chunks: Vec<StreamedChunk> = jobs.into_iter().map(|j| unheld(j.generate())).collect();
    let mut traces = builder.finish(&last).expect("the last window");
    traces
        .insert_streamed(chunks)
        .expect("every chunk has a placeholder");
    (traces, without)
}

/// ★ Without the CPU list, through the split walker and on 1 and 3 threads,
/// at windows of 1, 7 and 33 cycles, whatever the room admits (every CPU
/// chunk, none: each filled on the host from its rebuilt ops, every other),
/// the windows build the whole-run tables, and their walkers kept no list.
#[test]
fn without_the_cpu_list_the_windows_build_the_whole_run_tables() {
    let _ecrec = crate::tables::trace_builder::ecrec_forge_free();
    let max_rows = MaxRowsConfig::small();
    for name in ["all_instructions_64", "test_ecrec", "test_keccak_multi"] {
        let (program, logs) = run(name);
        let reference = whole(&program, &logs, &max_rows);
        for window in [1, 7, 33] {
            for walk in [Walk::Owned, Walk::Threads(1), Walk::Threads(3)] {
                for (what, room) in rooms() {
                    let (traces, without) =
                        built(&program, &logs, &max_rows, window, walk, room, true, true);
                    same_traces(&reference, &traces);
                    let body = (logs.len() - 1) / window;
                    assert!(
                        without == body || matches!(walk, Walk::Threads(_)),
                        "{name}/{window} {walk:?} {what}: {without} of {body} windows listless"
                    );
                    assert!(
                        without > 0 || body == 0,
                        "{name}/{window} {walk:?} {what}: no window walked without its list"
                    );
                }
            }
        }
    }
}

/// ★ Where the card does not fill CPU the mode does not apply: the walkers
/// keep their lists and the host fills each CPU chunk from them.
#[test]
fn without_the_cards_cpu_fill_the_walkers_keep_their_lists() {
    let max_rows = MaxRowsConfig::small();
    let (program, logs) = run("all_instructions_64");
    let reference = whole(&program, &logs, &max_rows);
    for walk in [Walk::Owned, Walk::Threads(2)] {
        let (traces, without) = built(
            &program,
            &logs,
            &max_rows,
            7,
            walk,
            Arc::new(|_| true),
            false,
            true,
        );
        same_traces(&reference, &traces);
        assert_eq!(without, 0, "{walk:?}: every window kept its list");
    }
}
