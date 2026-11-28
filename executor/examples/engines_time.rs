//! The executor alone, as the block prover's executor thread drives it, on
//! either engine: windows of 2^20 cycles handed out with `resume_owned` and a
//! memory snapshot at each window's start (what the walkers read), the
//! buffers fresh or taken back.
//!
//! ```text
//! cargo run --release -p executor --example engines_time -- \
//!     <guest.elf> <input.bin|-> <reference|fast> <fresh|recycle>[+history|+wordhistory] [rounds]
//! cargo run --release -p executor --example engines_time -- \
//!     <guest.elf> <input.bin|-> fast ident
//! ```
//!
//! Prints one `ENGINES TIME:` line a round (setup, execute, hand-off, total
//! seconds, cycles, a digest of every log) and the median total. `+history`
//! keeps the byte history, `+wordhistory` the word history (a timestamp a
//! word). `ident` runs both histories side by side and prints one
//! `ENGINES TIME IDENT:` line: the windows' logs, register stamps and access
//! records compared, and every word's final stamp against its bytes' largest.
use executor::elf::Elf;
use executor::vm::engine::{Engine, EngineKind};
use executor::vm::logs::Log;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 5 {
        eprintln!("usage: engines_time <elf> <input|-> <reference|fast> <fresh|recycle> [rounds]");
        std::process::exit(2);
    }
    let elf = std::fs::read(&args[1]).expect("the ELF");
    let program = Elf::load(&elf).expect("an ELF the loader takes");
    let input = if args[2] == "-" {
        Vec::new()
    } else {
        std::fs::read(&args[2]).expect("the input")
    };
    let kind = EngineKind::from_value(Some(&args[3])).expect("an engine");
    if args[4] == "ident" {
        std::process::exit(ident(&program, &input));
    }
    // `fresh`, `recycle` (buffers handed back two windows later) or
    // `prefault` (buffers prepared already written); `+history` keeps the
    // history (fast engine), `+wordhistory` the word history, its extras
    // handed back or prepared alike.
    let (buffers, history, words) = match args[4].split_once('+') {
        Some((b, "history")) => (b.to_string(), true, false),
        Some((b, "wordhistory")) => (b.to_string(), true, true),
        _ => (args[4].clone(), false, false),
    };
    let recycle = match buffers.as_str() {
        "fresh" => 0,
        "recycle" => 1,
        "prefault" => 2,
        other => panic!("{other}: fresh, recycle or prefault [+history|+wordhistory]"),
    };
    let rounds: usize = args.get(5).map_or(3, |r| r.parse().expect("a round count"));
    let window = 1usize << 20;
    // `ENGINES_TIME_SIDES=1`: each window's accesses by kind (history mode).
    let diag = std::env::var("ENGINES_TIME_SIDES").is_ok_and(|v| v == "1");
    let mut totals = Vec::new();
    for round in 0..rounds {
        let t = Instant::now();
        let mut engine = Engine::new(kind, &program, input.clone()).expect("the run starts");
        if words {
            engine
                .enable_word_history()
                .expect("the word history (fast engine)");
        } else if history {
            engine.enable_history().expect("history mode (fast engine)");
        }
        let setup = t.elapsed().as_secs_f64();
        let mut start = (!history).then(|| engine.snapshot());
        let mut extras_held: std::collections::VecDeque<Vec<executor::vm::history::HistExtra>> =
            Default::default();
        let mut side_bytes = 0usize;
        let (mut execute, mut handoff, mut cycles, mut digest) = (0.0, 0.0, 0u64, 0u64);
        // Windows still "downstream": two, as the prover's channel holds.
        let mut downstream: std::collections::VecDeque<Vec<Log>> = Default::default();
        loop {
            if recycle == 2 {
                engine.recycle(executor::vm::logs::prefaulted_logs(window));
                if history {
                    engine.recycle_extras(executor::vm::logs::prefaulted(
                        window,
                        executor::vm::history::HistExtra::default(),
                    ));
                }
            }
            let t = Instant::now();
            let logs = if history {
                let resumed = if words {
                    engine.resume_word_history(window)
                } else {
                    engine.resume_history(window)
                };
                resumed.expect("the run").map(|(logs, h)| {
                    side_bytes += h.heap_bytes()
                        - h.extras.capacity()
                            * std::mem::size_of::<executor::vm::history::HistExtra>();
                    if diag && words {
                        eprintln!(
                            "ENGINES TIME SIDES: cycles {} accesses {} crossing {} ecalls {} \
                                 ecall words {} register cells {}",
                            h.extras.len(),
                            h.word_accesses.len(),
                            h.next_words.len(),
                            h.groups.len(),
                            h.word_values.len(),
                            h.reg_cells.len(),
                        );
                    } else if diag {
                        use executor::vm::history::NO_ACCESS;
                        let mixed = h.accesses.iter().filter(|a| a.mixed != NO_ACCESS).count();
                        eprintln!(
                            "ENGINES TIME SIDES: cycles {} uniform {} mixed {} ecalls {} \
                                 ecall bytes {} register cells {}",
                            h.extras.len(),
                            h.accesses.len() - mixed,
                            mixed,
                            h.groups.len(),
                            h.values.len(),
                            h.reg_cells.len(),
                        );
                    }
                    extras_held.push_back(h.extras);
                    logs
                })
            } else {
                engine.resume_owned(window).expect("the run")
            };
            let ran = Instant::now();
            execute += (ran - t).as_secs_f64();
            let Some(logs) = logs else {
                break;
            };
            if !history {
                let next = Some(engine.snapshot());
                drop(std::mem::replace(&mut start, next));
            }
            handoff += ran.elapsed().as_secs_f64();
            cycles += logs.len() as u64;
            for log in logs.iter().step_by(4099) {
                digest = digest.rotate_left(7)
                    ^ log.current_pc
                    ^ log.next_pc.rotate_left(13)
                    ^ log.src1_val.rotate_left(29)
                    ^ log.src2_val.rotate_left(41)
                    ^ log.dst_val.rotate_left(53);
            }
            downstream.push_back(logs);
            if downstream.len() > 2 {
                let done = downstream.pop_front().expect("three held");
                if recycle == 1 {
                    engine.recycle(done);
                }
            }
            if extras_held.len() > 2 {
                let done = extras_held.pop_front().expect("three held");
                if recycle == 1 {
                    engine.recycle_extras(done);
                }
            }
        }
        let output = engine.public_output().to_vec();
        drop(downstream);
        let total = t.elapsed().as_secs_f64();
        totals.push(total);
        println!(
            "ENGINES TIME: round {round} engine {} buffers {} setup {setup:.5} execute {execute:.5} \
             hand-off {handoff:.3} total {total:.3} cycles {cycles} logs {digest:016x} output {} bytes \
             side lists {:.1} MiB",
            kind.name(),
            args[4],
            output.len(),
            side_bytes as f64 / (1 << 20) as f64,
        );
    }
    totals.sort_by(f64::total_cmp);
    println!(
        "ENGINES TIME MEDIAN: engine {} buffers {} total {:.3}",
        kind.name(),
        args[4],
        totals[totals.len() / 2]
    );
}

/// The word history against the byte history on one run, window by window
/// (2^20 cycles): the logs, each cycle's register stamps and access kind, the
/// ECALL register cells and the number of access records must be equal; each
/// load or store's word cells must hold the bytes the byte record holds, at a
/// stamp no older than theirs (equal for an aligned 8-byte access, whose word
/// is its bytes); and at the end every word's stamp must be its bytes'
/// largest, in every page either history stamped. Returns the exit code: 0
/// when nothing differs.
fn ident(program: &Elf, input: &[u8]) -> i32 {
    use executor::vm::history::{ECALL, NO_ACCESS, WORD_BYTES};
    let window = 1usize << 20;
    let mut bytes = Engine::new(EngineKind::Fast, program, input.to_vec()).expect("the run starts");
    let mut words = Engine::new(EngineKind::Fast, program, input.to_vec()).expect("the run starts");
    bytes.enable_history().expect("history mode");
    words.enable_word_history().expect("the word history");
    let (mut windows, mut cycles, mut accesses, mut exact, mut crossing, mut ecalls) =
        (0u64, 0u64, 0u64, 0u64, 0u64, 0u64);
    let mut differ = 0u64;
    let mut first_difference: Option<String> = None;
    let mut note = |why: String, differ: &mut u64| {
        *differ += 1;
        if first_difference.is_none() {
            first_difference = Some(why);
        }
    };
    loop {
        let b = bytes.resume_history(window).expect("the byte run");
        let w = words.resume_word_history(window).expect("the word run");
        let ((bl, bh), (wl, wh)) = match (b, w) {
            (Some(b), Some(w)) => (b, w),
            (None, None) => break,
            (b, w) => {
                note(
                    format!(
                        "window {windows}: one run ended ({} {})",
                        b.is_some(),
                        w.is_some()
                    ),
                    &mut differ,
                );
                break;
            }
        };
        let record = |l: &Log| [l.current_pc, l.next_pc, l.src1_val, l.src2_val, l.dst_val];
        if bl.len() != wl.len() || bl.iter().zip(&wl).any(|(b, w)| record(b) != record(w)) {
            note(format!("window {windows}: the logs differ"), &mut differ);
        }
        if bh.accesses.len() != wh.word_accesses.len() || bh.groups.len() != wh.groups.len() {
            note(
                format!("window {windows}: the record counts differ"),
                &mut differ,
            );
        }
        crossing += wh.next_words.len() as u64;
        for i in 0..bh.extras.len().min(wh.extras.len()) {
            let (be, we) = (bh.extras[i], wh.extras[i]);
            let kind = |side: u32| match side {
                NO_ACCESS => 0,
                s if s & ECALL != 0 => 2,
                _ => 1,
            };
            if be.regs_ts != we.regs_ts || kind(be.side) != kind(we.side) {
                note(
                    format!("window {windows} cycle {i}: stamps or kind"),
                    &mut differ,
                );
                continue;
            }
            if bh.cycle_registers(i) != wh.cycle_registers(i) {
                note(
                    format!("window {windows} cycle {i}: ECALL registers"),
                    &mut differ,
                );
            }
            match kind(be.side) {
                1 => {
                    accesses += 1;
                    let a = bh.accesses[be.side as usize];
                    let wa = wh.word_accesses[we.side as usize];
                    let mut cells = std::collections::HashMap::new();
                    wh.cycle_cells(i, |address, value, ts| {
                        cells.insert(address, (value, ts));
                    });
                    let mixed = (a.mixed != NO_ACCESS).then(|| bh.mixed[a.mixed as usize]);
                    let mut largest = 0u32;
                    for (k, byte) in a.old.to_le_bytes().into_iter().enumerate() {
                        let ts = mixed.map_or(a.ts, |t| t[k]);
                        largest = largest.max(ts);
                        if let Some(&(value, word_ts)) = cells.get(&a.addr.wrapping_add(k as u64))
                            && (value != byte || word_ts < ts)
                        {
                            note(
                                format!("window {windows} cycle {i}: byte {k} of {:#x}", a.addr),
                                &mut differ,
                            );
                        }
                    }
                    if wa.addr != a.addr {
                        note(
                            format!("window {windows} cycle {i}: the address"),
                            &mut differ,
                        );
                    } else if a.addr.is_multiple_of(WORD_BYTES) {
                        // The byte record holds the whole word: its stamp is
                        // their largest.
                        exact += 1;
                        if wa.ts != largest {
                            note(
                                format!("window {windows} cycle {i}: the word's stamp"),
                                &mut differ,
                            );
                        }
                    }
                }
                2 => ecalls += 1,
                _ => {}
            }
        }
        windows += 1;
        cycles += bl.len() as u64;
    }
    let (bf, wf) = (
        bytes.take_history_final().expect("the byte history's end"),
        words
            .take_word_history_final()
            .expect("the word history's end"),
    );
    // Each stamped page and its words (a byte page's cells over eight).
    let mut pages: Vec<(u64, u64)> = bf
        .shadow
        .pages()
        .into_iter()
        .map(|(b, cells)| (b, cells.len() as u64 / WORD_BYTES))
        .collect();
    pages.extend(
        wf.shadow
            .pages()
            .into_iter()
            .map(|(b, cells)| (b, cells.len() as u64)),
    );
    pages.sort_unstable();
    pages.dedup();
    let (mut final_words, mut final_differ) = (0u64, 0u64);
    for (base, page_words) in pages {
        for k in 0..page_words {
            let word = base + k * WORD_BYTES;
            let largest = (0..WORD_BYTES)
                .map(|j| bf.shadow.get(word + j))
                .max()
                .unwrap_or(0);
            final_words += 1;
            final_differ += u64::from(wf.shadow.get(word) != largest);
        }
    }
    println!(
        "ENGINES TIME IDENT: windows {windows} cycles {cycles} accesses {accesses} aligned-exact \
         {exact} crossing {crossing} ecalls {ecalls} differ {differ} final words {final_words} \
         differ {final_differ}{}",
        first_difference.map_or(String::new(), |why| format!(" first {why}"))
    );
    i32::from(differ != 0 || final_differ != 0)
}
