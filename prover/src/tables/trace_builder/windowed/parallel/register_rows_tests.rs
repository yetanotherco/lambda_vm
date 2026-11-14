//! The card-built register rows' known-answer vectors (CPU only): every
//! MEMW_R chunk a record-mode run hands out as words
//! ([`WindowedTraceBuilder::card_writes_words`]), with the logs of the CPU
//! tables the card is taken to hold, the program's DECODE rows and the words
//! the host fill writes, for the host replay of `card_rows.cu`'s
//! `derive_registers_from_cache`
//! (`crypto/math-cuda/tests/host_kat/derive_host_kat.cpp`).

use std::sync::{Arc, Mutex};

use executor::elf::Elf;
use executor::vm::fast::FastExecutor;
use executor::vm::logs::Log;
use multilinear::records::{Records, RegisterRow};

use super::super::{StreamTable, WindowedTraceBuilder};
use super::{ExecWindow, walk_windows_from_history};
use crate::tables::MaxRowsConfig;
use crate::tables::decode::DecodeTable;
use crate::tests::windowed_builder_tests::run;

/// A record-mode run of `program` in windows of `window` cycles, walked from
/// the fast engine's history on two walkers writing MEMW_R's rows as words:
/// every log of the run, and the MEMW_R chunks it handed out as records.
struct Walked {
    logs: Vec<Log>,
    chunks: Vec<Records>,
    decode: Arc<DecodeTable>,
}

fn walked(program: &Elf, max_rows: &MaxRowsConfig, window: usize) -> Walked {
    let mut builder = WindowedTraceBuilder::new(program, &[], max_rows)
        .and_then(WindowedTraceBuilder::drop_streamed_ops)
        .and_then(|b| b.card_fill_streamed(Arc::new(|_| true), true, true))
        .and_then(WindowedTraceBuilder::card_writes_words)
        .expect("the builder");
    let decode = Arc::clone(&builder.decode);
    let (tx, rx) = std::sync::mpsc::sync_channel::<ExecWindow>(2);
    let logs = Mutex::new(Vec::new());
    let mut chunks = Vec::new();
    std::thread::scope(|scope| {
        let logs = &logs;
        scope.spawn(move || {
            let mut fast = FastExecutor::new(program, Vec::new()).expect("the fast engine");
            fast.enable_history().expect("history mode");
            while let Some((window_logs, history)) = fast.resume_history(window).expect("the run") {
                logs.lock()
                    .expect("the logs")
                    .extend_from_slice(&window_logs);
                let last = (fast.pc() == 0)
                    .then(|| fast.take_history_final().map(Box::new))
                    .flatten();
                let sent = tx.send(ExecWindow {
                    logs: Arc::new(window_logs),
                    start: None,
                    history: Some(Box::new(history)),
                    last,
                });
                if sent.is_err() {
                    break;
                }
            }
        });
        walk_windows_from_history(&mut builder, rx, 2, 0, |accumulator, walked, _| {
            for job in accumulator.absorb(walked) {
                if job.table == StreamTable::MemwRegister && job.is_records() {
                    let mut chunk = job.generate();
                    chunks.extend(chunk.trace.take_records_main());
                }
            }
            Ok(())
        })
        .expect("the walk");
    });
    Walked {
        logs: logs.into_inner().expect("the logs"),
        chunks,
        decode,
    }
}

/// A words-mode MEMW_R table for the kernel's vectors: its words, its whole
/// rows, the run index of the first, and the host fill's words.
type WordsTable = (Vec<u64>, Vec<RegisterRow>, u64, Vec<u64>);

/// `v`'s bytes as they lie (plain `#[repr(C)]` records without padding).
fn raw<T: Copy>(v: &[T]) -> &[u8] {
    // SAFETY: `T` is one of the records below, all fields, no padding.
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), std::mem::size_of_val(v)) }
}

fn put(out: &mut Vec<u8>, words: &[u64]) {
    words
        .iter()
        .for_each(|w| out.extend_from_slice(&w.to_le_bytes()));
}

/// Writes the card kernel's known-answer vectors to `REGISTER_ROWS_VECTORS`
/// for `crypto/math-cuda/tests/host_kat/derive_host_kat.cpp`, which replays
/// `derive_registers_from_cache` on the host: record-mode runs walked as
/// words, their full CPU tables' logs, the program's DECODE rows, and each
/// MEMW_R chunk whose cycles those tables hold as the card gets it in words
/// mode (its words as the walk wrote them, its whole rows, the run index of
/// the first), with the words the host fill of its decoded rows writes.
/// Nothing is checked here but that the vectors hold tables past the first
/// chunk; the replay checks.
#[test]
#[ignore = "writes the kernel's vectors to REGISTER_ROWS_VECTORS"]
fn write_the_register_rows_kernel_vectors() {
    let path = std::env::var("REGISTER_ROWS_VECTORS").expect("REGISTER_ROWS_VECTORS");
    let max_rows = MaxRowsConfig::small();
    let cases = [
        ("test_keccak_multi", 7usize),
        ("all_instructions_64", 33),
        ("test_commit_split", 7),
        ("test_ecsm", 33),
        ("test_copy_ecalls", 7),
        ("misalign_sd", 33),
    ];
    let mut out: Vec<u8> = b"CRREG003".to_vec();
    put(&mut out, &[cases.len() as u64]);
    let (mut words_count, mut words_based) = (0usize, 0usize);
    for (name, window) in cases {
        let (program, _) = run(name);
        let walked = walked(&program, &max_rows, window);
        let cpu = max_rows.cpu as u64;
        let held: Vec<(u64, u64)> = (0..walked.logs.len() as u64 / cpu)
            .map(|k| (k * cpu, cpu))
            .collect();
        put(&mut out, &[held.len() as u64]);
        for &(first, n) in &held {
            put(&mut out, &[first, n]);
            for log in &walked.logs[first as usize..(first + n) as usize] {
                put(
                    &mut out,
                    &[
                        log.current_pc,
                        log.next_pc,
                        log.src1_val,
                        log.src2_val,
                        log.dst_val,
                    ],
                );
            }
        }
        let decode = walked.decode.card_records().expect("the DECODE rows");
        put(&mut out, &[decode.entries().len() as u64]);
        out.extend_from_slice(raw(decode.entries()));
        put(&mut out, &[decode.runs().len() as u64]);
        out.extend_from_slice(raw(decode.runs()));
        let end = held.last().map_or(0, |&(f, n)| f + n);
        let words_tables: Vec<WordsTable> = walked
            .chunks
            .iter()
            .filter_map(|records| {
                let w = records.register_words()?;
                let words: Vec<u64> = w.words.concat();
                let whole: Vec<RegisterRow> = w.whole.concat();
                let ts = |x: u64| {
                    if x as u32 == 0 {
                        whole[((x >> 32) - w.whole_base) as usize].timestamp
                    } else {
                        x & 0xFFFF_FFFF
                    }
                };
                let last = (ts(*words.last()?) - 4) / 4;
                if last >= end {
                    return None;
                }
                let packed = records.host_fill().expect("the host fill");
                let want = (0..packed.cols()).flat_map(|c| packed.column(c)).collect();
                Some((words, whole, w.whole_base, want))
            })
            .collect();
        put(&mut out, &[words_tables.len() as u64]);
        for (words, whole, base, want) in &words_tables {
            words_count += 1;
            words_based += usize::from(*base > 0);
            put(&mut out, &[words.len() as u64, want.len() as u64, *base]);
            put(&mut out, words);
            put(&mut out, &[whole.len() as u64]);
            out.extend_from_slice(raw(whole));
            put(&mut out, want);
        }
    }
    std::fs::write(&path, &out).expect("the vectors written");
    println!(
        "REGISTER VECTORS: {} cases, {words_count} tables as words ({words_based} after a whole row of an earlier chunk), {} bytes → {path}",
        cases.len(),
        out.len()
    );
    assert!(
        words_count >= 2 && words_based > 0,
        "the vectors need words tables past the first chunk"
    );
}
