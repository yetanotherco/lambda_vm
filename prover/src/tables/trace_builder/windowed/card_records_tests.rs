//! CPU and MEMW_R chunks, and the row families' (MEMW_A, LOAD, STORE, MEMW,
//! SHIFT), handed out as the
//! records their rows come from ([`WindowedTraceBuilder::card_fill_streamed`],
//! [`WindowedTraceBuilder::card_fill_streamed_rows`]): their host fill is the
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
    CardFillStamps, ChunkJob, RecordsRoom, RowFamilies, StreamTable, StreamedChunk, Table,
    WalkedWindow, WindowedTraceBuilder,
};
use crate::tables::MaxRowsConfig;
use crate::tables::trace_builder::Traces;
use crate::tests::windowed_builder_tests::{run, same_traces, whole};

/// The chunk heights the row families' tests run at: chunks of four rows, so
/// that the small programs fill several.
fn rows_max() -> MaxRowsConfig {
    MaxRowsConfig {
        memw_aligned: 1 << 2,
        load: 1 << 2,
        store: 1 << 2,
        memw: 1 << 2,
        shift: 1 << 2,
        ..MaxRowsConfig::small()
    }
}

/// Whether `table` is a row family's.
fn row_table(table: StreamTable) -> bool {
    matches!(
        table,
        StreamTable::MemwAligned
            | StreamTable::Load
            | StreamTable::Store
            | StreamTable::Memw
            | StreamTable::Shift
    )
}

/// Every family's `[as records, on the host]`: CPU, MEMW_R, then the row
/// families.
fn families(s: &CardFillStamps) -> [[usize; 2]; 7] {
    [s.cpu, s.memw_r, s.memw_a, s.load, s.store, s.memw, s.shift]
}

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
/// chunks (with `rows`, every row family's too) handed out as records
/// while `room` admits them and each records chunk then filled on the host,
/// with the chunks put back; how many chunks were records; and the builder's
/// stamps.
fn held(
    program: &Elf,
    logs: &[Log],
    max_rows: &MaxRowsConfig,
    window: usize,
    walk: Walk,
    room: RecordsRoom,
    rows: bool,
) -> (Traces, usize, CardFillStamps) {
    let mut builder = WindowedTraceBuilder::new(program, &[], max_rows)
        .and_then(WindowedTraceBuilder::drop_streamed_ops)
        .and_then(|b| b.card_fill_streamed(room, true, true))
        .and_then(|b| {
            if rows {
                b.card_fill_streamed_rows(RowFamilies::ALL)
            } else {
                Ok(b)
            }
        })
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
                let row_family = row_table(chunk.table);
                assert!(
                    matches!(chunk.table, StreamTable::Cpu | StreamTable::MemwRegister)
                        || (rows && row_family),
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
/// table the whole-run build makes; every other table is the same, and no
/// row family's chunk is held unless asked.
#[test]
fn cpu_and_memw_r_chunks_held_as_records_are_the_runs_tables() {
    chunks_held_as_records_are_the_runs_tables(false);
}

/// ★ The same with every full chunk of the row families held as its rows too
/// ([`WindowedTraceBuilder::card_fill_streamed_rows`]; chunks of four rows,
/// and programs that load, store and shift more): 48 B an aligned row, 32 a
/// load or a store, 152 a MEMW op, 24 a shift.
#[test]
fn row_family_chunks_held_as_records_are_the_runs_tables() {
    chunks_held_as_records_are_the_runs_tables(true);
}

fn chunks_held_as_records_are_the_runs_tables(rows: bool) {
    let (max_rows, more) = if rows {
        (rows_max(), &["all_loadstore_32", "test_blake3"][..])
    } else {
        (MaxRowsConfig::small(), &[][..])
    };
    let mut walks = vec![Walk::Own];
    #[cfg(feature = "parallel")]
    walks.extend([Walk::Threads(1), Walk::Threads(4)]);
    let mut held_most = [0usize; 7];
    let names = [
        "all_instructions_64",
        "test_keccak_multi",
        "test_ecsm",
        "test_commit_split",
    ];
    for &name in names.iter().chain(more) {
        let (program, logs) = run(name);
        let reference = whole(&program, &logs, &max_rows);
        for window in [1, 7, 32, 33] {
            for &walk in &walks {
                let what = format!("{name}, window {window}, {walk:?}");
                let (traces, records, stamps) =
                    held(&program, &logs, &max_rows, window, walk, always(), rows);
                same_traces(&reference, &traces);
                let families = families(&stamps);
                assert!(families.iter().all(|f| f[1] == 0), "{what}: {stamps:?}");
                if !rows {
                    assert!(families[2..].iter().all(|f| f[0] == 0), "{what}");
                }
                assert_eq!(stamps.out_of_step, 0, "{what}");
                assert_eq!(stamps.room_out, None, "{what}");
                assert_eq!(
                    records,
                    families.iter().map(|f| f[0]).sum::<usize>(),
                    "{what}"
                );
                for (most, family) in held_most.iter_mut().zip(families) {
                    *most = (*most).max(family[0]);
                }
                assert_eq!(
                    stamps.records_bytes,
                    stamps.cpu[0] * max_rows.cpu * 40
                        + stamps.memw_r[0] * max_rows.memw_register * 32
                        + stamps.memw_a[0] * max_rows.memw_aligned * 48
                        + stamps.load[0] * max_rows.load * 32
                        + stamps.store[0] * max_rows.store * 32
                        + stamps.memw[0] * max_rows.memw * 152
                        + stamps.shift[0] * max_rows.shift * 24,
                    "{what}: 40 B a cycle, 32 a register row, 48 an aligned row, 32 a load or \
                     store, 152 a MEMW op, 24 a shift"
                );
            }
        }
    }
    // No small program makes four MEMW ops: every other family fills several.
    let asked: &[usize] = if rows { &[0, 1, 2, 3, 4, 6] } else { &[0, 1] };
    assert!(asked.iter().all(|&k| held_most[k] > 1), "{held_most:?}");
}

/// Each row family asked alone is the only one held as rows: the others are
/// filled on the host, uncounted, and the tables are the run's.
#[test]
fn a_row_family_asked_alone_is_the_only_one_held() {
    let max_rows = rows_max();
    let (program, logs) = run("test_blake3");
    let reference = whole(&program, &logs, &max_rows);
    let alone = |k: usize| {
        let mut rows = RowFamilies::NONE;
        *[
            &mut rows.memw_a,
            &mut rows.load,
            &mut rows.store,
            &mut rows.memw,
            &mut rows.shift,
        ][k] = true;
        rows
    };
    for k in 0..5 {
        let mut builder = WindowedTraceBuilder::new(&program, &[], &max_rows)
            .and_then(WindowedTraceBuilder::drop_streamed_ops)
            .and_then(|b| b.card_fill_streamed(always(), false, false))
            .and_then(|b| b.card_fill_streamed_rows(alone(k)))
            .expect("the builder");
        let body = logs.len() - 1;
        let cut = body - body % 8;
        let mut chunks = Vec::new();
        {
            let (mut walker, mut accumulator) = builder.split();
            for w in logs[..cut].chunks(8) {
                let walked = walker.walk_owned(w.to_vec()).expect("a window");
                chunks.extend(
                    accumulator
                        .absorb(walked)
                        .into_iter()
                        .map(|job| unheld(job.generate()).0),
                );
            }
        }
        let stamps = builder.stamps().card_fill;
        let mut traces = builder.finish(&logs[cut..]).expect("the last window");
        traces
            .insert_streamed(chunks)
            .expect("every chunk has a placeholder");
        same_traces(&reference, &traces);
        let rows = &families(&stamps)[2..];
        for (j, counts) in rows.iter().enumerate() {
            if j == k {
                // test_blake3 makes no four MEMW ops.
                assert_eq!(counts[0] > 0, j != 3, "family {k}: {stamps:?}");
            } else {
                assert_eq!(
                    *counts,
                    [0, 0],
                    "family {k}: family {j} counted: {stamps:?}"
                );
            }
        }
        assert_eq!(stamps.cpu, [0, 0], "{stamps:?}");
    }
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
                false,
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

/// ★ The room rule is read chunk by chunk, across the families held, and from
/// the first chunk it refuses on every chunk is filled on the host, whatever
/// it says after: what is held as records is the prefix it admitted. The line
/// names that chunk, and the tables are the run's.
#[test]
fn from_the_first_chunk_without_room_every_chunk_is_filled_on_the_host() {
    let sum = |s: &CardFillStamps, k: usize| families(s).iter().map(|f| f[k]).sum::<usize>();
    for rows in [false, true] {
        let (name, max_rows) = if rows {
            ("test_blake3", rows_max())
        } else {
            ("all_instructions_64", MaxRowsConfig::small())
        };
        let (program, logs) = run(name);
        let reference = whole(&program, &logs, &max_rows);
        let (_, _, all) = held(&program, &logs, &max_rows, 32, Walk::Own, always(), rows);
        let chunks = sum(&all, 0);
        assert!(chunks > 4, "{all:?}");
        if rows {
            assert!(
                families(&all)[2..].iter().filter(|f| f[0] > 0).count() >= 2,
                "{all:?}"
            );
        }
        for admitted in [0, 1, 3, chunks - 1] {
            let what = format!("rows {rows}, {admitted} admitted");
            let (traces, records, stamps) = held(
                &program,
                &logs,
                &max_rows,
                32,
                Walk::Own,
                for_first(admitted, 1),
                rows,
            );
            same_traces(&reference, &traces);
            assert_eq!(records, admitted, "{what}: {stamps:?}");
            assert_eq!(sum(&stamps, 0), admitted, "{what}");
            assert_eq!(
                sum(&stamps, 1),
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
}

/// A chunk of any height but a whole table's is not held as records: the
/// host's fill pads it.
#[test]
fn only_a_whole_table_is_held_as_records() {
    use super::Segments;
    use super::card_records::{CardDecode, CpuLogRecords, RegisterRowRecords, RowRecords};
    use crate::tables::load::LoadOperation;
    use crate::tables::memw_aligned::AlignedRow;
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
    let aligned = AlignedRow::default();
    let load = LoadOperation::new(0x100, 9, 4, true, [1; 8]);
    for n in [0usize, 2, 3, 7, 33] {
        assert!(
            RowRecords::new(Segments::one(vec![aligned; n])).is_err(),
            "{n} aligned rows"
        );
        assert!(
            RowRecords::new(Segments::one(vec![load.clone(); n])).is_err(),
            "{n} loads"
        );
    }
    for n in [4usize, 32] {
        assert!(RowRecords::new(Segments::one(vec![aligned; n])).is_ok());
        assert!(RowRecords::new(Segments::one(vec![load.clone(); n])).is_ok());
    }
}

/// ★ A CPU chunk held as its logs is laid out for the CPU AIR the block lays a
/// streamed chunk out with: the records declare the preprocessed TIMESTAMP
/// column the fill writes from the row index, so the layout's check of it
/// passes. A MEMW_R chunk declares none, as its AIR has none.
#[test]
fn records_chunks_are_laid_out_for_their_tables_airs() {
    use super::Segments;
    use super::card_records::{CardDecode, CpuLogRecords, RegisterRowRecords};
    use crate::tables::cpu;
    use crate::test_utils::{create_cpu_air, create_memw_register_air};
    use multilinear::records::{Records, TableRecords};
    use stark::proof::options::ProofOptions;

    let (program, logs) = run("all_instructions_64");
    let builder =
        WindowedTraceBuilder::new(&program, &[], &MaxRowsConfig::small()).expect("the builder");
    let decode = Arc::new(CardDecode::new(Arc::clone(&builder.decode)).expect("a program"));
    let opts = ProofOptions::default_test_options();
    let height = 32usize;
    let cpu_air = create_cpu_air(&opts, 0).with_lazy_preprocessed_columns(
        cpu::lazy_timestamp_commitment(&opts, height),
        cpu::NUM_PRECOMPUTED_COLS,
        Arc::new(move || vec![cpu::timestamp_column(height)]),
    );
    type Air = dyn stark::traits::AIR<
            Field = GoldilocksField,
            FieldExtension = crate::tables::types::GoldilocksExtension,
            PublicInputs = (),
        >;
    let lay_out = |records: Records, air: &Air| {
        let cols = records.cols();
        let rows = records.rows();
        let mut trace = Table::try_from_records_main(records, 1).expect("a records trace");
        crate::block_whir::table_of_records(air, &mut trace, (cols, rows.trailing_zeros() as usize))
            .map(|_| ())
    };
    let cpu_records = CpuLogRecords::new(Segments::one(logs[..height].to_vec()), 0, decode)
        .expect("a whole table");
    assert_eq!(cpu_records.row_index_columns(), cpu::NUM_PRECOMPUTED_COLS);
    lay_out(Records(Arc::new(cpu_records)), &cpu_air).expect("CPU laid out");
    let row = crate::tables::memw_register::RegRow::new(0, 9, 0, 0, 0, 0, 8, false);
    let Ok(memw_r) = RegisterRowRecords::new(Segments::one(vec![row; 8])) else {
        panic!("8 rows are a whole table");
    };
    assert_eq!(memw_r.row_index_columns(), 0);
    lay_out(Records(Arc::new(memw_r)), &create_memw_register_air(&opts)).expect("MEMW_R laid out");
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

/// ★ A chunk held as records parks and drops like any chunk: the regeneration
/// recorder gives it a recipe and a rank, its block index. With records kept
/// (`LAMBDA_VM_BLOCK_RECORDS_DROP=keep`) it has neither: the recorder keeps its
/// place in the hand-out, so every other chunk's rank is still its block
/// index, but names no chunk there, so no regenerator drops it or waits on its
/// fate.
#[test]
fn a_chunk_held_as_records_has_a_rank_unless_records_are_kept() {
    use crate::block_whir::regen::Recorder;
    let (program, logs) = run("test_keccak_multi");
    for (keep, rows) in [(false, false), (true, false), (false, true), (true, true)] {
        let max_rows = if rows {
            rows_max()
        } else {
            MaxRowsConfig::small()
        };
        for (room, admitted) in [(always(), None), (for_first(3, 1000), Some(3))] {
            let what = format!("keep {keep}, rows {rows}, admitted {admitted:?}");
            let mut builder = WindowedTraceBuilder::new(&program, &[], &max_rows)
                .and_then(WindowedTraceBuilder::drop_streamed_ops)
                .and_then(|b| b.card_fill_streamed(room, true, true))
                .and_then(|b| {
                    if rows {
                        b.card_fill_streamed_rows(RowFamilies::ALL)
                    } else {
                        Ok(b)
                    }
                })
                .expect("the builder");
            let recorder = if keep {
                Recorder::new(false).keep_records()
            } else {
                Recorder::new(false)
            };
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
            assert!(held > 0, "{what}: {held} of {}", handed.len());
            let row_held = handed
                .iter()
                .any(|&(table, _, records)| records && row_table(table));
            if !rows || admitted.is_none() {
                assert_eq!(row_held, rows, "{what}");
            }
            if let Some(admitted) = admitted {
                assert!(
                    held == admitted && handed.len() > held,
                    "{what}: {held} of {}",
                    handed.len()
                );
            }
            for (t, &(table, index, records)) in handed.iter().enumerate() {
                let want = (!(records && keep)).then_some((table, index));
                assert_eq!(recorder.chunk_at(t), want, "{what}: block index {t}");
            }
            assert_eq!(recorder.chunk_at(handed.len()), None);
            let (recipes, stray) = recorder.recipes();
            assert_eq!(stray, 0);
            let ranked: Vec<(StreamTable, usize, usize)> = handed
                .iter()
                .enumerate()
                .filter(|(_, (_, _, records))| !(*records && keep))
                .map(|(t, &(table, index, _))| (table, index, t))
                .collect();
            let recorded: Vec<(StreamTable, usize, usize)> = recipes
                .iter()
                .map(|r| (r.table, r.index, r.order))
                .collect();
            assert_eq!(recorded, ranked, "{what}: a recipe at its block index");
        }
    }
}

/// ★ A chunk dropped as its records comes back as them: walked again from the
/// same logs, with the CPU logs kept, every CPU, MEMW_R and row family's chunk
/// phase A held as records is rebuilt with records of the same digest, and the
/// packed columns made from them are the host fill of phase A's records. A
/// records slot takes those columns; with one log changed in the replay, the
/// digest moves and the slot refuses the deposit, as a row family's slot
/// refuses another chunk's columns. A chunk not dropped as records is rebuilt
/// with no digest.
#[test]
fn a_chunk_dropped_as_records_is_rebuilt_as_them() {
    use std::collections::HashMap;

    use super::RegenBuilder;
    use crate::tables::gpack::TraceForm;
    use stark::regen::{RegenError, RegenWindow};

    let max_rows = rows_max();
    // MEMW_A's, LOAD's, STORE's and SHIFT's chunks offered another's columns
    // (no small program makes four MEMW ops).
    let mut crossed = [0usize; 4];
    for name in ["all_instructions_64", "test_keccak_multi", "test_blake3"] {
        let (program, logs) = run(name);
        let window = 7;
        let body = logs.len() - 1;
        let cut = body - body % window;
        // Phase A's records, by chunk.
        let mut held: HashMap<(StreamTable, usize), multilinear::records::Records> = HashMap::new();
        {
            let mut builder = WindowedTraceBuilder::new(&program, &[], &max_rows)
                .and_then(WindowedTraceBuilder::drop_streamed_ops)
                .and_then(|b| b.card_fill_streamed(always(), true, true))
                .and_then(|b| b.card_fill_streamed_rows(RowFamilies::ALL))
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
        assert!(held.len() > 2, "{name}: {} chunks held", held.len());
        // The replay, every window (the last one included), with one log
        // changed or not.
        let replay = |tamper: Option<usize>, as_records: bool| {
            let mut regen = RegenBuilder::new(&program, &[], &max_rows)
                .and_then(RegenBuilder::keep_cpu_logs)
                .expect("the regenerator");
            let mut rebuilt = HashMap::new();
            let mut at = 0usize;
            for w in logs.chunks(window) {
                let mut w = w.to_vec();
                if let Some(k) = tamper.filter(|&k| (at..at + w.len()).contains(&k)) {
                    w[k - at].dst_val ^= 1;
                }
                at += w.len();
                for job in regen.push_owned(w).expect("a window") {
                    let (table, index) = (job.table, job.index);
                    let (digest, chunk) = job.regenerate(TraceForm::Narrow, as_records);
                    let mut trace = chunk.trace;
                    trace.pack_main_narrow();
                    rebuilt.insert((table, index), (digest, trace.take_narrow_main()));
                }
            }
            rebuilt
        };
        let rebuilt = replay(None, true);
        let (slots, _producer) = RegenWindow::new(1 << 30);
        for (rank, (key, records)) in held.iter().enumerate() {
            let what = format!("{name} {key:?}");
            let (digest, packed) = rebuilt.get(key).cloned().expect("rebuilt");
            let packed = packed.expect("packed");
            assert_eq!(
                digest,
                Some(records.digest()),
                "{what}: the records' digest"
            );
            let want = records
                .host_fill()
                .expect("the host fills phase A's records");
            assert_eq!(
                (packed.widths(), packed.data()),
                (want.widths(), want.data()),
                "{what}: the packed columns"
            );
            let slot = slots.slot_for_records(records, rank as u64);
            slot.deposit_checked(packed, digest.expect("a digest"))
                .unwrap_or_else(|e| panic!("{what}: {e:?}"));
        }
        // One log changed in the first CPU chunk held: its records' digest
        // moves, and its slot refuses the deposit.
        let (&(table, index), records) = held
            .iter()
            .filter(|((t, _), _)| *t == StreamTable::Cpu)
            .min_by_key(|((_, i), _)| *i)
            .expect("a CPU chunk held");
        let bent = replay(Some(index * max_rows.cpu + 1), true);
        let (digest, packed) = bent.get(&(table, index)).cloned().expect("rebuilt");
        let digest = digest.expect("a digest");
        assert_ne!(digest, records.digest(), "{name}: one log changed");
        let slot = slots.slot_for_records(records, held.len() as u64);
        assert_eq!(
            slot.deposit_checked(packed.expect("packed"), digest),
            Err(RegenError::Mismatch),
            "{name}: the changed log's chunk"
        );
        // Another chunk's columns, and their digest, offered to a row family's
        // records slot.
        for (family, crossed) in [
            StreamTable::MemwAligned,
            StreamTable::Load,
            StreamTable::Store,
            StreamTable::Shift,
        ]
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
            let slot = slots.slot_for_records(&held[own], (held.len() + 1) as u64);
            assert_eq!(
                slot.deposit_checked(packed.expect("packed"), digest.expect("a digest")),
                Err(RegenError::Mismatch),
                "{name}: {other:?}'s columns for {own:?}"
            );
        }
        // Not dropped as records: the same columns, no digest.
        let plain = replay(None, false);
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
