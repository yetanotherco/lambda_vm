//! CPU and MEMW_R chunks handed out as the records their rows come from
//! ([`WindowedTraceBuilder::card_fill_streamed`]): their host fill is the
//! table the build makes without it, whatever windows, walkers and room; the
//! logs out of step with the ops, and every chunk past the first the room rule
//! refuses, are filled on the host.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use executor::elf::Elf;
use executor::vm::logs::Log;
use math::field::element::FieldElement;

use super::super::super::types::GoldilocksField;
use super::{
    CardFillStamps, ChunkJob, RecordsRoom, StreamTable, StreamedChunk, Table, WalkedWindow,
    WindowedTraceBuilder,
};
use crate::tables::MaxRowsConfig;
use crate::tables::trace_builder::Traces;
use crate::tests::windowed_builder_tests::{run, same_traces, whole};

/// A room rule that admits every chunk.
fn always() -> RecordsRoom {
    Arc::new(|_| true)
}

/// A room rule that admits the first `n` chunks it is asked about, then none
/// for `gap` asks, then every one again: a rule that changes its mind.
fn for_first(n: usize, gap: usize) -> RecordsRoom {
    let asked = AtomicUsize::new(0);
    Arc::new(move |_| {
        let k = asked.fetch_add(1, Ordering::Relaxed);
        k < n || k >= n + gap
    })
}

/// `chunk` with its records, if it holds them, replaced by their host fill at
/// eight bytes a cell; and whether it held them.
fn unheld(mut chunk: StreamedChunk) -> (StreamedChunk, bool) {
    let Some(records) = chunk.trace.take_records_main() else {
        return (chunk, false);
    };
    let packed = records.host_fill().expect("the host fills its records");
    assert_eq!(
        (packed.rows(), packed.cols()),
        (records.rows(), records.cols()),
        "{:?}[{}]: the host fill's shape",
        chunk.table,
        chunk.index
    );
    let columns = packed
        .widen::<GoldilocksField>()
        .expect("the packed columns widen");
    let (rows, cols) = (records.rows(), records.cols());
    let mut data: Vec<FieldElement<GoldilocksField>> = Vec::with_capacity(rows * cols);
    for r in 0..rows {
        data.extend(columns.iter().map(|column| column.evals()[r]));
    }
    chunk.trace = Table::new_main(data, cols, 1);
    (chunk, true)
}

/// How a test hands the accumulator its windows.
#[derive(Clone, Copy, Debug)]
enum Walk {
    /// The builder's own walker, keeping each window's logs.
    Own,
    /// The builder's own walker, window `k`'s logs swapped with window
    /// `k + 1`'s (two windows of one length): logs out of step with the ops.
    Swapped(usize),
    /// On this many threads, from the windows' entries.
    #[cfg(feature = "parallel")]
    Threads(usize),
}

/// The run of `program` by windows of `window` cycles, its CPU and MEMW_R
/// chunks handed out as records while `room` admits them and each records
/// chunk then filled on the host, with the chunks put back; how many chunks
/// were records; and the builder's stamps.
fn held(
    program: &Elf,
    logs: &[Log],
    max_rows: &MaxRowsConfig,
    window: usize,
    walk: Walk,
    room: RecordsRoom,
) -> (Traces, usize, CardFillStamps) {
    let mut builder = WindowedTraceBuilder::new(program, &[], max_rows)
        .and_then(WindowedTraceBuilder::drop_streamed_ops)
        .and_then(|b| b.card_fill_streamed(room, true, true))
        .expect("the builder");
    assert!(builder.keeps_logs());
    let mut jobs: Vec<ChunkJob> = Vec::new();
    let last = match walk {
        Walk::Own | Walk::Swapped(_) => {
            let body = logs.len() - 1;
            let cut = body - body % window;
            let windows: Vec<&[Log]> = logs[..cut].chunks(window).collect();
            let (mut walker, mut accumulator) = builder.split();
            let mut walked: Vec<WalkedWindow> = windows
                .iter()
                .map(|w| walker.walk(w).expect("a window"))
                .collect();
            for (k, w) in walked.iter_mut().enumerate() {
                let from = match walk {
                    Walk::Swapped(s) if k == s => s + 1,
                    Walk::Swapped(s) if k == s + 1 => s,
                    _ => k,
                };
                w.keep_logs(Arc::new(windows[from].to_vec()));
            }
            for w in walked {
                jobs.extend(accumulator.absorb(w));
            }
            logs[cut..].to_vec()
        }
        #[cfg(feature = "parallel")]
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
                    jobs.extend(accumulator.absorb(w));
                    Ok(())
                })
            })
            .expect("the walk");
            report.last
        }
    };
    let stamps = builder.stamps().card_fill;
    let mut records = 0usize;
    let chunks: Vec<StreamedChunk> = jobs
        .into_iter()
        .map(|job| {
            let was = job.is_records();
            let (chunk, held) = unheld(job.generate());
            assert_eq!(
                held, was,
                "{:?}[{}]: held as records",
                chunk.table, chunk.index
            );
            if held {
                assert!(
                    matches!(chunk.table, StreamTable::Cpu | StreamTable::MemwRegister),
                    "{:?} held as records",
                    chunk.table
                );
                records += 1;
            }
            chunk
        })
        .collect();
    let mut traces = builder.finish(&last).expect("the last window");
    traces
        .insert_streamed(chunks)
        .expect("every chunk has a placeholder");
    (traces, records, stamps)
}

/// ★ Every full CPU and MEMW_R chunk handed out as records, its logs or rows
/// in the window parts it was cut from (windows of 1, 7, 32 and 33 cycles,
/// walked by the builder or on 1 and 4 threads), fills on the host to the
/// table the whole-run build makes; every other table is the same.
#[test]
fn cpu_and_memw_r_chunks_held_as_records_are_the_runs_tables() {
    let max_rows = MaxRowsConfig::small();
    let mut walks = vec![Walk::Own];
    #[cfg(feature = "parallel")]
    walks.extend([Walk::Threads(1), Walk::Threads(4)]);
    let mut held_most = [0usize; 2];
    for name in [
        "all_instructions_64",
        "test_keccak_multi",
        "test_ecsm",
        "test_commit_split",
    ] {
        let (program, logs) = run(name);
        let reference = whole(&program, &logs, &max_rows);
        for window in [1, 7, 32, 33] {
            for &walk in &walks {
                let what = format!("{name}, window {window}, {walk:?}");
                let (traces, records, stamps) =
                    held(&program, &logs, &max_rows, window, walk, always());
                same_traces(&reference, &traces);
                assert_eq!(stamps.cpu[1] + stamps.memw_r[1], 0, "{what}: {stamps:?}");
                assert_eq!(stamps.out_of_step, 0, "{what}");
                assert_eq!(stamps.room_out, None, "{what}");
                assert_eq!(records, stamps.cpu[0] + stamps.memw_r[0], "{what}");
                held_most[0] = held_most[0].max(stamps.cpu[0]);
                held_most[1] = held_most[1].max(stamps.memw_r[0]);
                assert_eq!(
                    stamps.records_bytes,
                    stamps.cpu[0] * max_rows.cpu * 40
                        + stamps.memw_r[0] * max_rows.memw_register * 32,
                    "{what}: 40 B a cycle, 32 a register row"
                );
            }
        }
    }
    assert!(held_most[0] > 1 && held_most[1] > 1, "{held_most:?}");
}

/// ★ A window's logs swapped with the next window's (one length, so the
/// tails stay the same length) are out of step with the CPU ops: every CPU
/// chunk they fall in is filled on the host, counted, and the tables are the
/// run's.
#[test]
fn cpu_logs_out_of_step_with_the_ops_are_filled_on_the_host() {
    let max_rows = MaxRowsConfig::small();
    let mut held_elsewhere = 0;
    for name in ["all_instructions_64", "test_keccak_multi"] {
        let (program, logs) = run(name);
        let reference = whole(&program, &logs, &max_rows);
        for (window, swapped) in [(32, 0), (7, 1), (33, 0)] {
            // Both windows before the last.
            if (logs.len() - 1) / window < swapped + 2 {
                continue;
            }
            let what = format!(
                "{name}, window {window}, windows {swapped} and {} swapped",
                swapped + 1
            );
            let (traces, _, stamps) = held(
                &program,
                &logs,
                &max_rows,
                window,
                Walk::Swapped(swapped),
                always(),
            );
            same_traces(&reference, &traces);
            assert!(stamps.out_of_step > 0, "{what}: {stamps:?}");
            assert!(stamps.cpu[1] >= stamps.out_of_step, "{what}: {stamps:?}");
            held_elsewhere += stamps.cpu[0];
        }
    }
    assert!(
        held_elsewhere > 0,
        "the chunks the swap misses are still held"
    );
}

/// ★ The room rule is read chunk by chunk, and from the first chunk it refuses
/// on every chunk is filled on the host, whatever it says after: what is held
/// as records is the prefix it admitted. The line names that chunk, and the
/// tables are the run's.
#[test]
fn from_the_first_chunk_without_room_every_chunk_is_filled_on_the_host() {
    let max_rows = MaxRowsConfig::small();
    let (program, logs) = run("all_instructions_64");
    let reference = whole(&program, &logs, &max_rows);
    let (_, _, all) = held(&program, &logs, &max_rows, 32, Walk::Own, always());
    let chunks = all.cpu[0] + all.memw_r[0];
    assert!(chunks > 4, "{all:?}");
    for admitted in [0, 1, 3, chunks - 1] {
        let what = format!("{admitted} admitted");
        let (traces, records, stamps) = held(
            &program,
            &logs,
            &max_rows,
            32,
            Walk::Own,
            for_first(admitted, 1),
        );
        same_traces(&reference, &traces);
        assert_eq!(records, admitted, "{what}: {stamps:?}");
        assert_eq!(stamps.cpu[0] + stamps.memw_r[0], admitted, "{what}");
        assert_eq!(
            stamps.cpu[1] + stamps.memw_r[1],
            chunks - admitted,
            "{what}: every later chunk on the host"
        );
        assert!(
            stamps.room_out.is_some(),
            "{what}: the chunk room ran out at"
        );
    }
    // Never refused: no chunk named.
    assert_eq!(all.room_out, None);
}

/// A chunk of any height but a whole table's is not held as records: the
/// host's fill pads it.
#[test]
fn only_a_whole_table_is_held_as_records() {
    use super::Segments;
    use super::card_records::{CardDecode, CpuLogRecords, RegisterRowRecords};
    let (program, logs) = run("all_instructions_64");
    let builder =
        WindowedTraceBuilder::new(&program, &[], &MaxRowsConfig::small()).expect("the builder");
    let decode = Arc::new(CardDecode::new(Arc::clone(&builder.decode)).expect("a program"));
    for n in [0usize, 1, 2, 3, 5, 31, 33] {
        let some = Segments::one(logs[..n].to_vec());
        assert!(
            CpuLogRecords::new(some, 0, Arc::clone(&decode)).is_none(),
            "{n} logs"
        );
    }
    for n in [4usize, 32] {
        let some = Segments::one(logs[..n].to_vec());
        assert!(
            CpuLogRecords::new(some, 0, Arc::clone(&decode)).is_some(),
            "{n} logs"
        );
    }
    let row = crate::tables::memw_register::RegRow::new(0, 9, 0, 0, 0, 0, 8, false);
    for n in [0usize, 3, 7] {
        assert!(
            RegisterRowRecords::new(Segments::one(vec![row; n])).is_err(),
            "{n} rows"
        );
    }
    assert!(RegisterRowRecords::new(Segments::one(vec![row; 8])).is_ok());
}

/// The card's DECODE rows find every pc's row as the host's table does, and
/// no row for a pc the program has none at.
#[test]
fn the_cards_decode_rows_find_the_hosts_rows() {
    for name in ["all_instructions_64", "test_keccak_multi", "test_ecsm"] {
        let (program, logs) = run(name);
        let builder =
            WindowedTraceBuilder::new(&program, &[], &MaxRowsConfig::small()).expect("the builder");
        let host = &builder.decode;
        let card = host.card_records().expect("a program");
        let mut pcs: Vec<u64> = logs.iter().map(|log| log.current_pc).collect();
        pcs.extend(card.entries().iter().map(|e| e.pc));
        let probes: Vec<u64> = pcs
            .iter()
            .flat_map(|&pc| {
                [
                    pc,
                    pc.wrapping_add(2),
                    pc.wrapping_sub(2),
                    pc.wrapping_add(4),
                ]
            })
            .chain([0, u64::MAX])
            .collect();
        for pc in probes {
            assert_eq!(card.row(pc), host.row(pc), "{name}: pc {pc:#x}");
        }
        for (row, entry) in card.entries().iter().enumerate() {
            let host_entry = host.entry(row as u32);
            assert_eq!(
                (entry.pc, entry.imm, entry.rd, entry.ecall != 0),
                (
                    host_entry.pc,
                    host_entry.imm,
                    host_entry.fields.rd,
                    host_entry.fields.ecall
                ),
                "{name}: row {row}"
            );
        }
    }
}

/// ★ A chunk held as records has no recipe and no rank: the regeneration
/// recorder keeps its place in the hand-out, so every other chunk's rank is
/// its block index still, but names no chunk there, so no regenerator drops
/// it or waits on its fate.
#[test]
fn a_chunk_held_as_records_has_no_recipe_and_no_rank() {
    use crate::block_whir::regen::Recorder;
    let max_rows = MaxRowsConfig::small();
    let (program, logs) = run("test_keccak_multi");
    for (room, admitted) in [(always(), None), (for_first(3, 1000), Some(3))] {
        let mut builder = WindowedTraceBuilder::new(&program, &[], &max_rows)
            .and_then(WindowedTraceBuilder::drop_streamed_ops)
            .and_then(|b| b.card_fill_streamed(room, true, true))
            .expect("the builder");
        let recorder = Recorder::new(false);
        let mut handed: Vec<(StreamTable, usize, bool)> = Vec::new();
        {
            let (mut walker, mut accumulator) = builder.split();
            let body = logs.len() - 1;
            for w in logs[..body - body % 32].chunks(32) {
                let walked = walker.walk_owned(w.to_vec()).expect("a window");
                let (jobs, added) = accumulator.absorb_counted(walked);
                recorder.window(added, &jobs);
                handed.extend(jobs.iter().map(|j| (j.table, j.index, j.is_records())));
            }
        }
        let held = handed.iter().filter(|(_, _, r)| *r).count();
        assert!(held > 0, "{held} of {}", handed.len());
        if let Some(admitted) = admitted {
            assert!(
                held == admitted && handed.len() > held,
                "{held} of {}",
                handed.len()
            );
        }
        for (t, &(table, index, records)) in handed.iter().enumerate() {
            let want = (!records).then_some((table, index));
            assert_eq!(recorder.chunk_at(t), want, "block index {t}");
        }
        assert_eq!(recorder.chunk_at(handed.len()), None);
        let (recipes, stray) = recorder.recipes();
        assert_eq!(stray, 0);
        let kept: Vec<(StreamTable, usize, usize)> = handed
            .iter()
            .enumerate()
            .filter(|(_, (_, _, records))| !records)
            .map(|(t, &(table, index, _))| (table, index, t))
            .collect();
        let recorded: Vec<(StreamTable, usize, usize)> = recipes
            .iter()
            .map(|r| (r.table, r.index, r.order))
            .collect();
        assert_eq!(
            recorded, kept,
            "a recipe a chunk not held, at its block index"
        );
    }
}
