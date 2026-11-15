//! The in-row side records derived from the executor's history
//! ([`super::history_sides`]) against the walk's own, record for record, on
//! record-mode runs of every ECALL and misalignment program, in several window
//! and CPU chunk heights; and the rules each part of the derivation stands on,
//! each broken once.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use executor::elf::Elf;
use executor::vm::fast::FastExecutor;
use executor::vm::history::HistExtra;
use executor::vm::logs::Log;
use multilinear::records::SideException;

use super::WindowedTraceBuilder;
use super::history_sides::{chunk_first, cpu_sides_from_history, exceptions_of_walk};
use super::parallel::{ExecWindow, walk_windows_from_history};
use crate::tables::MaxRowsConfig;
use crate::tables::cpu::CpuSide;
use crate::tables::decode::DecodeTable;
use crate::tables::register::RegisterFormat;
use crate::tests::windowed_builder_tests::run;

/// The asm programs: every ECALL the walk handles but HINT, misaligned and
/// page-crossing accesses (the record-mode tests' list).
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

/// A walked window: its first cycle, its logs, its history records and the
/// side records its walk wrote.
struct Window {
    first: u64,
    logs: Vec<Log>,
    extras: Vec<HistExtra>,
    sides: Vec<CpuSide>,
}

/// `program` run in history mode in windows of `window` cycles and walked
/// from the history under the in-row register format on two walkers: every
/// window the walkers walked (the run's last, the finish's, is not), and the
/// program's DECODE rows.
fn walked(program: &Elf, window: usize) -> (Vec<Window>, Arc<DecodeTable>) {
    walked_with(program, &[], window)
}

/// The windows the executor sent, by first cycle: their logs and history
/// records, taken as the walkers hand each window back.
type Sent = Mutex<BTreeMap<u64, (Vec<Log>, Vec<HistExtra>)>>;

/// The walked windows as one run: logs, history records and the walk's side
/// records from cycle 0, and each window's first cycle.
fn concat(windows: &[Window]) -> (Vec<Log>, Vec<HistExtra>, Vec<CpuSide>, Vec<u64>) {
    let mut out = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for w in windows {
        assert_eq!(w.first as usize, out.0.len(), "the windows are in order");
        assert_eq!(w.logs.len(), w.extras.len());
        assert_eq!(w.logs.len(), w.sides.len(), "one side record a log");
        out.0.extend_from_slice(&w.logs);
        out.1.extend_from_slice(&w.extras);
        out.2.extend_from_slice(&w.sides);
        out.3.push(w.first);
    }
    out
}

/// Every window's exceptions for CPU chunks of `chunk_rows` cycles, in order.
fn exceptions(windows: &[Window], chunk_rows: u64, decode: &DecodeTable) -> Vec<SideException> {
    windows
        .iter()
        .flat_map(|w| {
            exceptions_of_walk(&w.logs, &w.extras, &w.sides, w.first, chunk_rows, decode)
                .expect("the window's exceptions")
        })
        .collect()
}

/// The derivation of the run's walked cycles cut into chunks of
/// `chunk_rows`, each with the exceptions of its cycles.
fn derived(
    logs: &[Log],
    extras: &[HistExtra],
    exceptions: &[SideException],
    chunk_rows: usize,
    decode: &DecodeTable,
) -> Result<Vec<CpuSide>, String> {
    let mut sides = Vec::with_capacity(logs.len());
    for start in (0..logs.len()).step_by(chunk_rows) {
        let end = (start + chunk_rows).min(logs.len());
        let ex: Vec<SideException> = exceptions
            .iter()
            .copied()
            .filter(|x| (start..end).contains(&(x.cycle as usize)))
            .collect();
        sides.extend(cpu_sides_from_history(
            &[&logs[start..end]],
            &[&extras[start..end]],
            start as u64,
            decode,
            &[&ex],
        )?);
    }
    Ok(sides)
}

/// Every walked row's side record derived from the history is the walk's:
/// 14 programs × windows 1, 7, 33, 4096 × CPU chunks of 64 and 4096 cycles
/// (chunk starts inside windows and windows inside chunks). The exceptions
/// stay a small share of the rows.
#[test]
fn the_derived_side_records_are_the_walks() {
    let (mut rows, mut listed, mut with_rd) = (0usize, 0usize, 0usize);
    for name in PROGRAMS {
        let (program, _) = run(name);
        for window in [1usize, 7, 33, 4096] {
            let (windows, decode) = walked(&program, window);
            let (logs, extras, sides, _) = concat(&windows);
            for chunk_rows in [64usize, 4096] {
                let what = format!("{name}, window {window}, CPU chunks of {chunk_rows}");
                let ex = exceptions(&windows, chunk_rows as u64, &decode);
                let got = derived(&logs, &extras, &ex, chunk_rows, &decode)
                    .unwrap_or_else(|e| panic!("{what}: {e}"));
                assert_eq!(got.len(), sides.len(), "{what}");
                if let Some(at) = (0..got.len()).find(|&i| got[i] != sides[i]) {
                    panic!(
                        "{what}: cycle {at}: derived {:?}, walked {:?}",
                        got[at], sides[at]
                    );
                }
                rows += got.len();
                listed += ex.len();
                with_rd += sides.iter().filter(|s| s.prev_ts[2] != 0).count();
            }
        }
    }
    assert!(rows > 0 && with_rd > 0, "{rows} rows, {with_rd} with rd");
    // Windows of one cycle list nearly every rd write: the share is over all
    // window heights, and must stay well under one.
    assert!(listed < with_rd, "{listed} of {with_rd} rd rows listed");
}

/// Each rule is load-bearing: a run that lists no exception is refused (not
/// derived wrong) somewhere; an exception's value moved changes that row's
/// record and no other; a history record's `rs2` stamp moved changes the
/// row's or is refused.
#[test]
fn each_rule_of_the_derivation_is_load_bearing() {
    let (mut refused_without_exceptions, mut tried) = (0usize, 0usize);
    for name in [
        "test_keccak_multi",
        "test_commit_split",
        "all_instructions_64",
    ] {
        let (program, _) = run(name);
        let (windows, decode) = walked(&program, 7);
        let (logs, extras, sides, _) = concat(&windows);
        let chunk_rows = 16usize;
        let ex = exceptions(&windows, chunk_rows as u64, &decode);
        println!(
            "{name}: {} walked cycles, {} exceptions",
            logs.len(),
            ex.len()
        );
        if ex.is_empty() {
            continue;
        }
        tried += 1;
        // No exceptions: each listed row the derivation cannot read is refused.
        if derived(&logs, &extras, &[], chunk_rows, &decode).is_err() {
            refused_without_exceptions += 1;
        }
        // An exception's value moved: that row, and only it, differs.
        let mut moved = ex.clone();
        moved[0].old_rd[0] ^= 1;
        let got = derived(&logs, &extras, &moved, chunk_rows, &decode).expect("derived");
        let differ: Vec<usize> = (0..got.len()).filter(|&i| got[i] != sides[i]).collect();
        assert_eq!(differ, vec![moved[0].cycle as usize], "{name}");
        // An rs2 stamp moved by one: the row's previous timestamp moves too
        // (or the step past 2^16 makes it a keep-alive's), or it is refused.
        let at = (0..logs.len())
            .find(|&i| sides[i].prev_ts[1] != 0 && sides[i].prev_ts[1] == extras[i].regs_ts[1])
            .expect("a row that reads rs2 within the step");
        let mut stamped = extras.clone();
        stamped[at].regs_ts[1] -= 1;
        if let Ok(got) = derived(&logs, &stamped, &ex, chunk_rows, &decode) {
            assert_ne!(got[at], sides[at], "{name}: cycle {at}");
        }
        // A list out of order is refused.
        if ex.len() >= 2 {
            let mut swapped = ex.clone();
            swapped.swap(0, 1);
            assert!(derived(&logs, &extras, &swapped, logs.len(), &decode).is_err());
        }
    }
    assert!(tried >= 2, "{tried} runs with exceptions");
    assert!(
        refused_without_exceptions > 0,
        "no run needed its exceptions"
    );
}

/// A CPU chunk's first cycle is its height's multiple.
#[test]
fn chunks_start_at_their_heights_multiples() {
    assert_eq!(chunk_first(0, 64), 0);
    assert_eq!(chunk_first(63, 64), 0);
    assert_eq!(chunk_first(64, 64), 64);
    assert_eq!(chunk_first(200, 64), 192);
}

/// A Rust guest's ELF from `program_artifacts/rust`.
fn rust_elf(name: &str) -> Elf {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../executor/program_artifacts/rust")
        .join(format!("{name}.elf"));
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    Elf::load(&bytes).expect("the ELF loads")
}

/// [`walked`] with a private input.
fn walked_with(program: &Elf, input: &[u8], window: usize) -> (Vec<Window>, Arc<DecodeTable>) {
    let mut builder = WindowedTraceBuilder::new(program, input, &MaxRowsConfig::small())
        .and_then(|b| b.register_format(RegisterFormat::InRow))
        .expect("the builder");
    let decode = Arc::clone(&builder.decode);
    let (tx, rx) = std::sync::mpsc::sync_channel::<ExecWindow>(2);
    let sent: Sent = Mutex::new(BTreeMap::new());
    let mut windows = Vec::new();
    std::thread::scope(|scope| {
        let sent = &sent;
        scope.spawn(move || {
            let mut fast = FastExecutor::new(program, input.to_vec()).expect("the fast engine");
            fast.enable_history().expect("history mode");
            while let Some((logs, history)) = fast.resume_history(window).expect("the run") {
                sent.lock()
                    .expect("the windows")
                    .insert(history.first, (logs.clone(), history.extras.clone()));
                let last = (fast.pc() == 0)
                    .then(|| fast.take_history_final().map(Box::new))
                    .flatten();
                let window = ExecWindow {
                    logs: Arc::new(logs),
                    start: None,
                    history: Some(Box::new(history)),
                    last,
                };
                if tx.send(window).is_err() {
                    break;
                }
            }
        });
        walk_windows_from_history(&mut builder, rx, 2, 0, |accumulator, walked, _| {
            let first = walked.first as u64;
            let sides = walked.walk.cpu_side.clone();
            let (logs, extras) = sent
                .lock()
                .expect("the windows")
                .remove(&first)
                .expect("the window was sent");
            windows.push(Window {
                first,
                logs,
                extras,
                sides,
            });
            let _ = accumulator.absorb(walked);
            Ok(())
        })
        .expect("the walk");
    });
    (windows, decode)
}

/// The derivation against the walk on the Rust guests (HINT, DMA, KECCAK)
/// and two ethrex blocks, in CPU chunks of 4096 and 2^17 cycles: real
/// registers, ECALLs between an access and the next, chunks of many windows.
#[test]
fn rust_guests_and_ethrex_derive_the_walks_side_records() {
    let fixtures = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../executor/tests");
    let mut cases: Vec<(String, Elf, Vec<u8>, usize)> = [
        "hint_min",
        "hint_multi",
        "dma_memset_cases",
        "dma_memmove_cases",
        "keccak_precompile",
    ]
    .iter()
    .map(|name| (name.to_string(), rust_elf(name), Vec::new(), 97))
    .collect();
    for fixture in ["ethrex_empty_block", "ethrex_10_transfers"] {
        let input = std::fs::read(fixtures.join(format!("{fixture}.bin"))).expect("the fixture");
        cases.push((fixture.to_string(), rust_elf("ethrex"), input, 1 << 16));
    }
    let (mut rows, mut listed, mut at_ecall) = (0usize, 0usize, 0usize);
    for (name, program, input, window) in &cases {
        let (windows, decode) = walked_with(program, input, *window);
        let (logs, extras, sides, _) = concat(&windows);
        for chunk_rows in [4096usize, 1 << 17] {
            let what = format!("{name}, window {window}, CPU chunks of {chunk_rows}");
            let ex = exceptions(&windows, chunk_rows as u64, &decode);
            let got = derived(&logs, &extras, &ex, chunk_rows, &decode)
                .unwrap_or_else(|e| panic!("{what}: {e}"));
            if let Some(at) = (0..got.len()).find(|&i| got[i] != sides[i]) {
                panic!(
                    "{what}: cycle {at}: derived {:?}, walked {:?}",
                    got[at], sides[at]
                );
            }
            // The rows listed because their old access is in an ECALL cycle.
            at_ecall += ex
                .iter()
                .filter(|x| {
                    let o = u64::from(extras[x.cycle as usize].regs_ts[2]);
                    o >= 4
                        && logs.get(((o - 4) / 4) as usize).is_some_and(|l| {
                            decode
                                .row(l.current_pc)
                                .is_some_and(|r| decode.entry(r).fields.ecall)
                        })
                })
                .count();
            rows += got.len();
            listed += ex.len();
        }
        println!("{name}: {} walked cycles", logs.len());
    }
    println!("rows {rows}, listed {listed}, of them at an ECALL {at_ecall}");
    assert!(rows > 1_000_000, "{rows} rows");
    assert!(at_ecall > 0, "no row's rd was last touched by an ECALL");
    assert!(listed * 100 < rows, "{listed} of {rows} rows listed");
}

/// `v`'s bytes as they lie (plain `#[repr(C)]` records without padding).
fn raw<T: Copy>(v: &[T]) -> &[u8] {
    // SAFETY: `T` is one of the records written below, all fields, no padding.
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), std::mem::size_of_val(v)) }
}

fn put(out: &mut Vec<u8>, words: &[u64]) {
    words
        .iter()
        .for_each(|w| out.extend_from_slice(&w.to_le_bytes()));
}

/// Writes `derive_cpu_sides`' known-answer vectors to `SIDE_RECORDS_VECTORS`
/// for `crypto/math-cuda/tests/host_kat/sides_host_kat.cpp`, which replays
/// the kernel on the host and then the CPU fill with what it derived: record
/// walks of asm programs (CPU chunks of 16 cycles), Rust guests and an ethrex
/// block (chunks of 4096), each whole chunk's logs, history records and
/// exceptions as the card gets them, the program's DECODE rows, and the words
/// the host fills of the chunk's CPU table with the walk's side records.
/// Nothing is checked here but that the vectors hold chunks with
/// exceptions and with a keep-alive (a step past 2^16) for the replay's
/// mutants to move.
#[test]
#[ignore = "writes the kernel's vectors to SIDE_RECORDS_VECTORS"]
fn write_the_side_records_kernel_vectors() {
    use crate::tables::card_fill_inputs::{
        col_major, cpu_in_row_host, cpu_ops, decode_records, log_record,
    };
    use crate::tables::gpack::TraceForm;
    let path = std::env::var("SIDE_RECORDS_VECTORS").expect("SIDE_RECORDS_VECTORS");
    let fixtures = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../executor/tests");
    let mut cases: Vec<(String, Elf, Vec<u8>, usize, usize, usize)> = [
        "all_instructions_64",
        "test_keccak_multi",
        "test_commit_split",
        "test_copy_ecalls",
        "misalign_sd",
    ]
    .iter()
    .map(|name| (name.to_string(), run(name).0, Vec::new(), 7, 16, usize::MAX))
    .collect();
    for name in ["hint_multi", "dma_memmove_cases", "keccak_precompile"] {
        cases.push((
            name.to_string(),
            rust_elf(name),
            Vec::new(),
            97,
            4096,
            usize::MAX,
        ));
    }
    let input = std::fs::read(fixtures.join("ethrex_empty_block.bin")).expect("the fixture");
    cases.push((
        "ethrex_empty_block".to_string(),
        rust_elf("ethrex"),
        input,
        1 << 16,
        4096,
        16,
    ));
    let mut out: Vec<u8> = b"CTSIDE01".to_vec();
    put(&mut out, &[cases.len() as u64]);
    let (mut chunks_n, mut with_ex, mut with_keep) = (0usize, 0usize, 0usize);
    for (name, program, input, window, chunk_rows, most) in &cases {
        let (windows, decode) = walked_with(program, input, *window);
        let (logs, extras, sides, _) = concat(&windows);
        let ex = exceptions(&windows, *chunk_rows as u64, &decode);
        let (entries, runs) = decode_records(&decode);
        put(&mut out, &[entries.len() as u64]);
        out.extend_from_slice(raw(&entries));
        put(&mut out, &[runs.len() as u64]);
        out.extend_from_slice(raw(&runs));
        let whole = (logs.len() / chunk_rows).min(*most);
        put(&mut out, &[whole as u64]);
        for k in 0..whole {
            let (start, end) = (k * chunk_rows, (k + 1) * chunk_rows);
            let chunk_ex: Vec<SideException> = ex
                .iter()
                .copied()
                .filter(|x| (start..end).contains(&(x.cycle as usize)))
                .collect();
            let ops = cpu_ops(&logs[start..end], start as u64, &decode);
            let want = col_major(&cpu_in_row_host(
                &ops,
                &sides[start..end],
                &decode,
                TraceForm::Wide,
            ));
            let records: Vec<_> = logs[start..end].iter().map(log_record).collect();
            put(&mut out, &[start as u64, *chunk_rows as u64]);
            out.extend_from_slice(raw(&records));
            out.extend_from_slice(raw(super::history_sides::card_stamps(&extras[start..end])));
            put(&mut out, &[chunk_ex.len() as u64]);
            out.extend_from_slice(raw(&chunk_ex));
            put(&mut out, &[want.len() as u64]);
            put(&mut out, &want);
            chunks_n += 1;
            with_ex += usize::from(!chunk_ex.is_empty());
            with_keep += usize::from((start..end).any(|i| {
                let s = &sides[i];
                (0..3).any(|k| s.prev_ts[k] != 0 && s.prev_ts[k] != extras[i].regs_ts[k])
            }));
        }
        println!("{name}: {whole} chunks of {chunk_rows}");
    }
    std::fs::write(&path, &out).expect("the vectors written");
    println!(
        "SIDE VECTORS: {} cases, {chunks_n} chunks ({with_ex} with exceptions, {with_keep} with a \
         keep-alive), {} bytes → {path}",
        cases.len(),
        out.len()
    );
    assert!(
        with_ex > 0 && with_keep > 0,
        "{with_ex} with exceptions, {with_keep} with a keep-alive"
    );
}
