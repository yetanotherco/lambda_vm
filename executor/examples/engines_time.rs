//! The executor alone, as the block prover's executor thread drives it, on
//! either engine: windows of 2^20 cycles handed out with `resume_owned` and a
//! memory snapshot at each window's start (what the walkers read), the
//! buffers fresh or taken back.
//!
//! ```text
//! cargo run --release -p executor --example engines_time -- \
//!     <guest.elf> <input.bin|-> <reference|fast> <fresh|recycle> [rounds]
//! ```
//!
//! Prints one `ENGINES TIME:` line a round (setup, execute, hand-off, total
//! seconds, cycles, a digest of every log) and the median total.
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
    // `fresh`, `recycle` (buffers handed back two windows later) or
    // `prefault` (buffers prepared already written); `+history` keeps the
    // history (fast engine), its extras handed back or prepared alike.
    let (buffers, history) = match args[4].split_once('+') {
        Some((b, "history")) => (b.to_string(), true),
        _ => (args[4].clone(), false),
    };
    let recycle = match buffers.as_str() {
        "fresh" => 0,
        "recycle" => 1,
        "prefault" => 2,
        other => panic!("{other}: fresh, recycle or prefault [+history]"),
    };
    let rounds: usize = args.get(5).map_or(3, |r| r.parse().expect("a round count"));
    let window = 1usize << 20;
    // `ENGINES_TIME_SIDES=1`: each window's accesses by kind (history mode).
    let diag = std::env::var("ENGINES_TIME_SIDES").is_ok_and(|v| v == "1");
    let mut totals = Vec::new();
    for round in 0..rounds {
        let t = Instant::now();
        let mut engine = Engine::new(kind, &program, input.clone()).expect("the run starts");
        if history {
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
                engine
                    .resume_history(window)
                    .expect("the run")
                    .map(|(logs, h)| {
                        side_bytes += h.heap_bytes() - h.extras.capacity() * 40;
                        if diag {
                            use executor::vm::history::{MIXED, NO_ACCESS, UNIFORM};
                            let count = |f: &dyn Fn(u32) -> bool| {
                                h.extras.iter().filter(|e| f(e.side)).count()
                            };
                            eprintln!(
                                "ENGINES TIME SIDES: cycles {} uniform {} mixed {} ecalls {} \
                                 ecall bytes {} register cells {}",
                                h.extras.len(),
                                count(&|s| s == UNIFORM),
                                count(&|s| s == MIXED),
                                count(&|s| s != NO_ACCESS && s != UNIFORM && s != MIXED),
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
