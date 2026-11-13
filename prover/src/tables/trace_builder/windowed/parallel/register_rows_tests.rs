//! The card-built register rows' rule, on the host (CPU only): every MEMW_R
//! chunk a record-mode run hands out with its ECALL cycles
//! ([`WindowedTraceBuilder::card_marks_ecalls`]) is planned against the
//! run's CPU tables (`math_cuda::card_rows::plan_registers`), and each row the
//! plan sends as its timestamps must be the walk's row as the card reads it
//! from the logs (`card_rows.cu`, `derive_registers_from_cache`, restated
//! here): its cycle's DECODE register and value by slot, a read for slots 0
//! and 1, and a write's old value from the log of the cycle its old
//! timestamp names, by that timestamp's slot, in the same CPU table. No row
//! of an ECALL cycle, and no write whose old timestamp names one, may go as
//! its timestamps; every row that goes whole is the walk's own.

use std::sync::{Arc, Mutex};

use executor::elf::Elf;
use executor::vm::fast::FastExecutor;
use executor::vm::logs::Log;
use math_cuda::card_rows::{RegisterPlan, WHOLE_ROW, WholeRows, plan_registers};
use math_cuda::fill::RegRowC;
use multilinear::records::{Records, RecordsFill};

use super::super::super::CpuOperation;
use super::super::{StreamTable, WindowedTraceBuilder};
use super::{ExecWindow, walk_windows_from_history};
use crate::tables::MaxRowsConfig;
use crate::tables::decode::DecodeTable;
use crate::tests::windowed_builder_tests::run;

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

/// A record-mode run of `program` in windows of `window` cycles, walked from
/// the fast engine's history on two walkers: every log of the run, and the
/// MEMW_R chunks it handed out as records.
struct Walked {
    logs: Vec<Log>,
    chunks: Vec<Records>,
    decode: Arc<DecodeTable>,
}

fn walked(program: &Elf, max_rows: &MaxRowsConfig, window: usize) -> Walked {
    let mut builder = WindowedTraceBuilder::new(program, &[], max_rows)
        .and_then(WindowedTraceBuilder::drop_streamed_ops)
        .and_then(|b| b.card_fill_streamed(Arc::new(|_| true), true, true))
        .and_then(WindowedTraceBuilder::card_marks_ecalls)
        .expect("the builder");
    assert!(builder.marks_ecalls());
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

/// A MEMW_R chunk's rows as the card's struct.
fn rows_of(records: &Records) -> Vec<RegRowC> {
    let RecordsFill::MemwR { rows } = records.fill() else {
        panic!("not a MEMW_R chunk: {records:?}");
    };
    rows.iter()
        .flat_map(|piece| piece.iter())
        .map(|r| RegRowC {
            timestamp: r.timestamp,
            val0: r.val0,
            val1: r.val1,
            old0: r.old0,
            old1: r.old1,
            old_ts_lo: r.old_ts_lo,
            address: r.address,
            is_read: r.is_read,
            pad: r.pad,
        })
        .collect()
}

/// What the plans of a run did.
#[derive(Default, Debug)]
struct Tally {
    chunks: usize,
    planned: usize,
    built: u64,
    why: WholeRows,
}

/// Checks every chunk of `run` against the rule, `held` the CPU tables the
/// card is taken to hold (first cycle, count).
fn check(name: &str, run: &Walked, held: &[(u64, u64)], tally: &mut Tally) {
    let mut ops: Vec<CpuOperation> = Vec::with_capacity(run.logs.len());
    super::super::super::collect_cpu_ops_into(&run.logs, &run.decode, 0, &mut ops)
        .expect("the run's CPU ops");
    let fields = |cycle: u64| run.decode.op(&ops[cycle as usize]).decode.fields;
    let table_of = |cycle: u64| {
        held.iter()
            .find(|&&(first, n)| first <= cycle && cycle < first + n)
            .map(|&(first, _)| first)
    };
    for records in &run.chunks {
        tally.chunks += 1;
        let ecalls = records
            .ecall_cycles()
            .unwrap_or_else(|| panic!("{name}: a MEMW_R chunk without its ECALL cycles"));
        assert!(
            ecalls.windows(2).all(|w| w[0] < w[1]),
            "{name}: ECALL cycles ascending"
        );
        let rows = rows_of(records);
        let Some(plan) = plan_registers(&[&rows[..]], held, ecalls, 0) else {
            continue;
        };
        tally.planned += 1;
        let RegisterPlan {
            words, whole, why, ..
        } = &plan;
        tally.why = WholeRows {
            ecall: tally.why.ecall + why.ecall,
            after_ecall: tally.why.after_ecall + why.after_ecall,
            before_table: tally.why.before_table + why.before_table,
            not_held: tally.why.not_held + why.not_held,
            other: tally.why.other + why.other,
        };
        let mut next = 0usize;
        for (r, (row, &word)) in rows.iter().zip(words).enumerate() {
            let at = format!("{name}: row {r} at {}", row.timestamp);
            if word as u32 == WHOLE_ROW {
                assert_eq!(whole[next], *row, "{at}: a whole row is the walk's");
                next += 1;
                continue;
            }
            tally.built += 1;
            let (ts, old) = (word & 0xFFFF_FFFF, word >> 32);
            assert_eq!(ts, row.timestamp, "{at}");
            let (cycle, slot) = ((ts - 4) / 4, (ts - 4) % 4);
            let op = &ops[cycle as usize];
            let f = fields(cycle);
            assert!(!f.ecall, "{at}: an ECALL cycle's row as its timestamps");
            let (on, reg, value) = match slot {
                0 => (f.read_register1, f.rs1, op.rv1),
                1 => (f.read_register2, f.rs2, op.rv2),
                2 => (f.write_register, f.rd, op.rvd),
                _ => panic!("{at}: slot 3"),
            };
            assert!(on && reg != 0, "{at}: the cycle makes no such access");
            let before = if slot < 2 {
                value
            } else {
                let first = table_of(cycle).expect("a held cycle");
                assert!(old >= 4 * first + 4, "{at}: old before its table");
                let (j, sj) = ((old - 4) / 4, (old - 4) % 4);
                assert!(!fields(j).ecall, "{at}: old names an ECALL cycle");
                let log = &run.logs[j as usize];
                match sj {
                    0 => log.src1_val,
                    1 => log.src2_val,
                    2 => log.dst_val,
                    _ => panic!("{at}: old at slot 3"),
                }
            };
            let want = RegRowC {
                timestamp: ts,
                val0: value as u32,
                val1: (value >> 32) as u32,
                old0: before as u32,
                old1: (before >> 32) as u32,
                old_ts_lo: old as u32,
                address: u16::from(reg),
                is_read: u8::from(slot < 2),
                pad: 0,
            };
            assert_eq!(*row, want, "{at}: the card's row is not the walk's");
        }
        assert_eq!(next, whole.len(), "{name}: every whole row placed");
    }
}

/// ★ Every program, windows of 7, 33 and 4096 cycles, CPU tables of 32: each
/// row built from the logs is the walk's, and the rule both builds rows and
/// sends rows whole for each of its reasons.
#[test]
fn the_card_built_register_rows_are_the_walks_rows() {
    let max_rows = MaxRowsConfig::small();
    let mut tally = Tally::default();
    let mut sparse = Tally::default();
    for name in PROGRAMS {
        let (program, _) = run(name);
        for window in [7usize, 33, 4096] {
            let walked = walked(&program, &max_rows, window);
            let cpu = max_rows.cpu as u64;
            let tables = walked.logs.len() as u64 / cpu;
            let held: Vec<(u64, u64)> = (0..tables).map(|k| (k * cpu, cpu)).collect();
            check(
                &format!("{name}, window {window}"),
                &walked,
                &held,
                &mut tally,
            );
            // Every other CPU table: their rows cross whole.
            let some: Vec<(u64, u64)> = held.iter().copied().step_by(2).collect();
            check(
                &format!("{name}, window {window}, half held"),
                &walked,
                &some,
                &mut sparse,
            );
        }
    }
    println!("REGISTER RULE: {tally:?} · half held {sparse:?}");
    assert!(
        tally.planned > 0 && tally.built > 0,
        "no row built: {tally:?}"
    );
    let w = tally.why;
    assert!(
        w.ecall > 0 && w.after_ecall > 0 && w.before_table > 0,
        "a reason never met: {w:?}"
    );
    assert_eq!(w.other, 0, "rows the rule does not read: {w:?}");
    assert!(sparse.why.not_held > 0 && sparse.built > 0, "{sparse:?}");
}

/// Without the executor's history (record mode off), the walkers have no
/// ECALL cycles to read: a builder marking them hands its MEMW_R chunks out
/// without them (the card then sends MEMW_R as records). And marking needs
/// MEMW_R held as records.
#[test]
fn without_the_history_a_memw_r_chunk_carries_no_ecall_cycles() {
    let max_rows = MaxRowsConfig::small();
    let (program, _) = run("test_keccak_multi");
    let refused = WindowedTraceBuilder::new(&program, &[], &max_rows)
        .and_then(WindowedTraceBuilder::drop_streamed_ops)
        .and_then(|b| b.card_fill_streamed(Arc::new(|_| true), true, false))
        .and_then(WindowedTraceBuilder::card_marks_ecalls);
    assert!(refused.is_err(), "marking without MEMW_R as records");
    let mut builder = WindowedTraceBuilder::new(&program, &[], &max_rows)
        .and_then(WindowedTraceBuilder::drop_streamed_ops)
        .and_then(|b| b.card_fill_streamed(Arc::new(|_| true), true, true))
        .and_then(WindowedTraceBuilder::card_marks_ecalls)
        .expect("the builder");
    let (mut walker, mut accumulator) = builder.split();
    let mut executor =
        executor::vm::execution::Executor::new(&program, Vec::new()).expect("the executor");
    let mut seen = 0usize;
    let mut ahead: Option<Vec<Log>> = None;
    while let Some(logs) = executor.resume_with_limit(7).expect("the run") {
        // The run's last window is the finish's.
        if let Some(previous) = ahead.replace(logs.to_vec()) {
            let w = walker.walk_owned(previous).expect("a window");
            for job in accumulator.absorb(w) {
                if job.table == StreamTable::MemwRegister && job.is_records() {
                    let mut chunk = job.generate();
                    if let Some(records) = chunk.trace.take_records_main() {
                        assert!(records.ecall_cycles().is_none());
                        seen += 1;
                    }
                }
            }
        }
    }
    assert!(seen > 0, "no MEMW_R chunk held as records");
}

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

fn put_plan(out: &mut Vec<u8>, plan: &RegisterPlan) {
    put(out, &[plan.words.len() as u64]);
    put(out, &plan.words);
    put(out, &[plan.whole.len() as u64]);
    out.extend_from_slice(raw(&plan.whole));
}

/// Writes the card kernel's known-answer vectors to `REGISTER_ROWS_VECTORS`
/// for `crypto/math-cuda/tests/host_kat/derive_host_kat.cpp`, which replays
/// `derive_registers_from_cache` on the host: record-mode runs' MEMW_R
/// chunks as planned against every full CPU table, their CPU tables' logs,
/// the program's DECODE rows, the words the host fill writes, and each
/// chunk's plan under the host mutants that change it. Nothing is checked
/// here; the replay checks.
#[test]
#[ignore = "writes the kernel's vectors to REGISTER_ROWS_VECTORS"]
fn write_the_register_rows_kernel_vectors() {
    use math_cuda::card_rows::{REGISTER_FAULT_ECALL, REGISTER_FAULT_START};
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
    let mut out: Vec<u8> = b"CRREG001".to_vec();
    put(&mut out, &[cases.len() as u64]);
    let mut tables = 0usize;
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
        let planned: Vec<(RegisterPlan, Vec<Option<RegisterPlan>>, Vec<u64>)> = walked
            .chunks
            .iter()
            .filter_map(|records| {
                let rows = rows_of(records);
                let ecalls = records.ecall_cycles().expect("the ECALL cycles");
                let plan = plan_registers(&[&rows[..]], &held, ecalls, 0)?;
                let faulted = [REGISTER_FAULT_START, REGISTER_FAULT_ECALL]
                    .map(|f| plan_registers(&[&rows[..]], &held, ecalls, f).filter(|p| *p != plan))
                    .to_vec();
                let packed = records.host_fill().expect("the host fill");
                let words = (0..packed.cols()).flat_map(|c| packed.column(c)).collect();
                Some((plan, faulted, words))
            })
            .collect();
        put(&mut out, &[planned.len() as u64]);
        for (plan, faulted, words) in &planned {
            tables += 1;
            put(&mut out, &[plan.words.len() as u64, words.len() as u64]);
            put_plan(&mut out, plan);
            put(&mut out, words);
            for f in faulted {
                match f {
                    Some(p) => {
                        put(&mut out, &[1]);
                        put_plan(&mut out, p);
                    }
                    None => put(&mut out, &[0]),
                }
            }
        }
    }
    std::fs::write(&path, &out).expect("the vectors written");
    println!(
        "REGISTER VECTORS: {} cases, {tables} tables, {} bytes → {path}",
        cases.len(),
        out.len()
    );
}
