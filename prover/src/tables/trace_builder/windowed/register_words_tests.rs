//! Words mode (`LAMBDA_VM_CARD_ROWS=words`, [`WindowedTraceBuilder::card_writes_words`]),
//! on the host (CPU only): the walkers write MEMW_R's register rows as the
//! 8-byte words the card builds rows from, and every host reader of a chunk's
//! rows decodes them from the logs. Record-mode runs built this way must give
//! the whole-run tables (every MEMW_R chunk held as words, filled on the host
//! from its decoded rows; the finish's last rows decoded; the IsHalfword
//! counts made from the words), and each held chunk's digest must be the
//! rows-mode chunk's. Each mutant of the walker's rule or the decode must
//! change a table or fail the build.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use executor::elf::Elf;
use executor::vm::logs::Log;
use math::field::element::FieldElement;

use super::super::register_words::{
    FAULT_WORDS, WORDS_FAULT_BEFORE, WORDS_FAULT_DECODE_SLOT, WORDS_FAULT_ECALL,
};
use super::{ChunkJob, RecordsRoom, StreamTable, Table, WindowedTraceBuilder};
use crate::tables::MaxRowsConfig;
use crate::tables::trace_builder::{StreamedChunk, Traces};
use crate::tables::types::GoldilocksField;
use crate::tests::windowed_builder_tests::{run, same_traces, whole};

/// The asm programs of the record-mode tests: every ECALL the walk handles
/// but HINT, misaligned and page-crossing accesses.
const PROGRAMS: [&str; 14] = [
    "all_instructions_64",
    "all_loadstore_32",
    "test_keccak_multi",
    "test_keccak256",
    "test_blake3",
    "test_blake3_absorb",
    "test_ecsm",
    "test_ecrec",
    "test_commit_split",
    "test_copy_ecalls",
    "test_copy_overlaps",
    "misalign_ld",
    "misalign_sd",
    "misalign_sh",
];

/// What a build held: MEMW_R chunks as words, and every MEMW_R chunk's
/// records digest in hand-out order.
#[derive(Default)]
struct Held {
    words: usize,
    digests: Vec<[u64; 2]>,
}

/// `chunk` with its records, if it holds them, replaced by their host fill at
/// eight bytes a cell (as the card fills them).
fn unheld(mut chunk: StreamedChunk, held: &mut Held) -> StreamedChunk {
    let Some(records) = chunk.trace.take_records_main() else {
        return chunk;
    };
    if chunk.table == StreamTable::MemwRegister {
        held.words += usize::from(records.register_words().is_some());
        held.digests.push(records.digest());
    }
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

/// The record-mode run of `program` by windows of `window` cycles on
/// `walkers` threads, dropping its streamed ops, the card filling CPU and
/// MEMW_R while `room` admits, the walkers counting where made, writing
/// words when `words`; every chunk then filled on the host and put back.
fn built(
    program: &Elf,
    max_rows: &MaxRowsConfig,
    window: usize,
    walkers: usize,
    room: RecordsRoom,
    words: bool,
    where_made: bool,
) -> (Traces, Held) {
    use executor::vm::fast::FastExecutor;
    let mut builder = WindowedTraceBuilder::new(program, &[], max_rows)
        .and_then(WindowedTraceBuilder::drop_streamed_ops)
        .and_then(|b| b.card_fill_streamed(room, true, true))
        .expect("the builder")
        .count_where_made(where_made);
    if words {
        builder = builder.card_writes_words().expect("words mode");
    }
    assert_eq!(builder.writes_words().is_some(), words);
    let mut jobs: Vec<ChunkJob> = Vec::new();
    let (tx, rx) = std::sync::mpsc::sync_channel::<super::parallel::ExecWindow>(2);
    let report = std::thread::scope(|scope| {
        scope.spawn(move || {
            let mut fast = FastExecutor::new(program, Vec::new()).expect("the fast engine");
            fast.enable_history().expect("history mode");
            while let Some((logs, history)) = fast.resume_history(window).expect("the run") {
                let last = (fast.pc() == 0)
                    .then(|| fast.take_history_final().map(Box::new))
                    .flatten();
                let sent = tx.send(super::parallel::ExecWindow {
                    logs: Arc::new(logs),
                    start: None,
                    history: Some(Box::new(history)),
                    last,
                });
                if sent.is_err() {
                    break;
                }
            }
        });
        super::parallel::walk_windows_from_history(&mut builder, rx, walkers, 0, |a, w, _| {
            jobs.extend(a.absorb(w));
            Ok(())
        })
    })
    .expect("the walk");
    let mut held = Held::default();
    let chunks: Vec<StreamedChunk> = jobs
        .into_iter()
        .map(|j| unheld(j.generate(), &mut held))
        .collect();
    let last: Vec<Log> = report.last;
    let mut traces = builder.finish(&last).expect("the last window");
    traces
        .insert_streamed(chunks)
        .expect("every chunk has a placeholder");
    (traces, held)
}

/// The room rules: every chunk, none (every MEMW_R chunk's rows decoded and
/// filled on the host as rows), and every other one.
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

/// ★ Every program, windows of 7 and 33 cycles, one and three walkers, every
/// room rule: words mode builds the whole-run tables, holds MEMW_R chunks as
/// words, and each held chunk's digest is the rows-mode chunk's. Then each
/// mutant, one at a time, on the programs with ECALLs: the walker sending an
/// ECALL cycle's rows as words, and the decode reading a write's old value
/// whatever its slot, must change a table or fail the build; the walker
/// sending a write whose old access lies before its window as a word is the
/// card's mutant (its kernel refuses the row), reported here.
#[test]
fn words_mode_builds_the_whole_run_tables_and_each_mutant_moves_them() {
    let _ecrec = crate::tables::trace_builder::ecrec_forge_free();
    let max_rows = MaxRowsConfig::small();
    let mut words_chunks = 0usize;
    let mut compared = 0usize;
    for name in PROGRAMS {
        let (program, logs) = run(name);
        let reference = whole(&program, &logs, &max_rows);
        for window in [7usize, 33] {
            for walkers in [1usize, 3] {
                for (what, room) in rooms() {
                    let at = format!("{name}, window {window}, {walkers} walkers, room {what}");
                    let (traces, held) =
                        built(&program, &max_rows, window, walkers, room, true, true);
                    same_traces(&reference, &traces);
                    words_chunks += held.words;
                    if what == "always" {
                        // The accumulator counting the handed-out words.
                        let (traces, _) = built(
                            &program,
                            &max_rows,
                            window,
                            walkers,
                            Arc::new(|_| true),
                            true,
                            false,
                        );
                        same_traces(&reference, &traces);
                        let (_, rows) = built(
                            &program,
                            &max_rows,
                            window,
                            walkers,
                            Arc::new(|_| true),
                            false,
                            true,
                        );
                        assert_eq!(held.digests, rows.digests, "{at}: the chunks' digests");
                        compared += held.digests.len();
                    }
                }
            }
        }
    }
    println!(
        "REGISTER WORDS HOST: {words_chunks} MEMW_R chunks held as words · {compared} digests equal to rows mode's"
    );
    assert!(
        words_chunks > 0 && compared > 0,
        "words mode never held a chunk"
    );

    for (mutant, bit, must) in [
        ("an ECALL cycle's rows as words", WORDS_FAULT_ECALL, true),
        (
            "old value read whatever its slot",
            WORDS_FAULT_DECODE_SLOT,
            true,
        ),
        (
            "a write before its window as a word",
            WORDS_FAULT_BEFORE,
            false,
        ),
    ] {
        let mut moved = 0usize;
        let mut runs = 0usize;
        for name in [
            "all_instructions_64",
            "test_keccak_multi",
            "test_ecsm",
            "test_commit_split",
            "test_copy_ecalls",
        ] {
            let (program, logs) = run(name);
            let reference = whole(&program, &logs, &max_rows);
            for window in [7usize, 33] {
                FAULT_WORDS.store(bit, Ordering::Relaxed);
                let outcome = catch_unwind(AssertUnwindSafe(|| {
                    let (traces, _) = built(
                        &program,
                        &max_rows,
                        window,
                        1,
                        Arc::new(|_| true),
                        true,
                        true,
                    );
                    same_traces(&reference, &traces);
                }));
                FAULT_WORDS.store(0, Ordering::Relaxed);
                runs += 1;
                moved += usize::from(outcome.is_err());
            }
        }
        println!("REGISTER WORDS MUTANT {mutant}: {moved} of {runs} runs moved");
        if must {
            assert!(moved > 0, "{mutant}: the mutant moved no run");
        }
    }
}

/// Words mode needs MEMW_R held as records and the CPU logs kept, and a run
/// whose builder does not ask for it writes rows.
#[test]
fn words_mode_needs_memw_r_as_records_and_the_cpu_logs() {
    let max_rows = MaxRowsConfig::small();
    let (program, _) = run("test_keccak_multi");
    let no_memw_r = WindowedTraceBuilder::new(&program, &[], &max_rows)
        .and_then(WindowedTraceBuilder::drop_streamed_ops)
        .and_then(|b| b.card_fill_streamed(Arc::new(|_| true), true, false))
        .and_then(WindowedTraceBuilder::card_writes_words);
    assert!(no_memw_r.is_err(), "words without MEMW_R as records");
    let no_logs = WindowedTraceBuilder::new(&program, &[], &max_rows)
        .and_then(WindowedTraceBuilder::drop_streamed_ops)
        .and_then(|b| b.card_fill_streamed(Arc::new(|_| true), false, true))
        .and_then(WindowedTraceBuilder::card_writes_words);
    assert!(no_logs.is_err(), "words without the CPU logs");
    let rows = WindowedTraceBuilder::new(&program, &[], &max_rows)
        .and_then(WindowedTraceBuilder::drop_streamed_ops)
        .and_then(|b| b.card_fill_streamed(Arc::new(|_| true), true, true))
        .expect("the builder");
    assert!(rows.writes_words().is_none());
}

/// The MEMW_R chunks a record-mode run of `program` hands out as records, in
/// hand-out order, its walkers writing words when `words`.
fn memw_r_chunks(
    program: &Elf,
    input: &[u8],
    max_rows: &MaxRowsConfig,
    window: usize,
    words: bool,
) -> Vec<multilinear::records::Records> {
    use executor::vm::fast::FastExecutor;
    let mut builder = WindowedTraceBuilder::new(program, input, max_rows)
        .and_then(WindowedTraceBuilder::drop_streamed_ops)
        .and_then(|b| b.card_fill_streamed(Arc::new(|_| true), true, true))
        .expect("the builder")
        .count_where_made(true);
    if words {
        builder = builder.card_writes_words().expect("words mode");
    }
    let mut held = Vec::new();
    let (tx, rx) = std::sync::mpsc::sync_channel::<super::parallel::ExecWindow>(2);
    std::thread::scope(|scope| {
        scope.spawn(move || {
            let mut fast = FastExecutor::new(program, input.to_vec()).expect("the fast engine");
            fast.enable_history().expect("history mode");
            while let Some((logs, history)) = fast.resume_history(window).expect("the run") {
                let last = (fast.pc() == 0)
                    .then(|| fast.take_history_final().map(Box::new))
                    .flatten();
                let sent = tx.send(super::parallel::ExecWindow {
                    logs: Arc::new(logs),
                    start: None,
                    history: Some(Box::new(history)),
                    last,
                });
                if sent.is_err() {
                    break;
                }
            }
        });
        super::parallel::walk_windows_from_history(&mut builder, rx, 4, 0, |a, w, _| {
            for job in a.absorb(w) {
                if job.table == StreamTable::MemwRegister && job.is_records() {
                    let mut chunk = job.generate();
                    held.extend(chunk.trace.take_records_main());
                }
            }
            Ok(())
        })
        .expect("the walk");
    });
    held
}

/// The real block (`BLOCK_WHIR_ELF`, `BLOCK_WHIR_INPUT`) at the production
/// heights in record mode, every window, walked as rows and as words: each
/// MEMW_R chunk's rows decoded from its words are the walk's rows, field for
/// field, and its digest is the rows' digest. A box test: both runs' chunks
/// are held at once.
#[test]
#[ignore = "a real block's whole record-mode run, twice: box only"]
fn the_real_blocks_words_decode_to_the_walks_rows() {
    let elf =
        std::fs::read(std::env::var("BLOCK_WHIR_ELF").expect("BLOCK_WHIR_ELF")).expect("the ELF");
    let input =
        std::fs::read(std::env::var("BLOCK_WHIR_INPUT").expect("BLOCK_WHIR_INPUT")).expect("input");
    let program = Elf::load(&elf).expect("the ELF loads");
    let options = crate::block_whir::BlockOptions::production();
    let window = 1usize << options.window_log2.expect("the block streams");
    let clock = std::time::Instant::now();
    let rows = memw_r_chunks(&program, &input, &options.max_rows, window, false);
    let words = memw_r_chunks(&program, &input, &options.max_rows, window, true);
    assert_eq!(rows.len(), words.len(), "the same MEMW_R chunks");
    let (mut as_words, mut decoded) = (0usize, 0usize);
    for (k, (r, w)) in rows.into_iter().zip(words).enumerate() {
        as_words += usize::from(w.register_words().is_some());
        let flat = |x: &multilinear::records::Records| -> Vec<multilinear::records::RegisterRow> {
            match x.fill() {
                multilinear::records::RecordsFill::MemwR { rows } => {
                    rows.iter().flat_map(|p| p.iter()).copied().collect()
                }
                other => panic!("chunk {k}: not MEMW_R: {other:?}"),
            }
        };
        let (a, b) = (flat(&r), flat(&w));
        assert_eq!(a.len(), b.len(), "chunk {k}: rows");
        if let Some(i) = (0..a.len()).find(|&i| a[i] != b[i]) {
            panic!(
                "chunk {k} row {i}: words decode to {:?}, the walk wrote {:?}",
                b[i], a[i]
            );
        }
        assert_eq!(r.digest(), w.digest(), "chunk {k}: digest");
        decoded += b.len();
    }
    println!(
        "REGISTER WORDS REAL: {as_words} MEMW_R chunks held as words · rows decoded from words equal to the walk's {decoded} · digests equal · {:.1}s",
        clock.elapsed().as_secs_f64()
    );
    assert!(as_words > 0 && decoded > 0, "nothing held as words");
}
