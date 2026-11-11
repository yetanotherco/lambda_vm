//! Record mode: the fast engine's history against the lean pass, window by
//! window, and the run built by walkers reading it against the whole-run
//! build; and the sampled re-walk refusing a history put wrong on purpose.

use std::sync::Arc;

use executor::elf::Elf;
use executor::vm::execution::{Executor, InstructionCache};
use executor::vm::fast::FastExecutor;
use executor::vm::history::{UNIFORM, WindowHistory};
use executor::vm::instruction::decoding::Instruction;
use executor::vm::logs::Log;

use super::super::super::RegisterState;
use super::super::lean_pass::LeanPass;
use super::super::{ChunkJob, WindowedTraceBuilder};
use super::{ExecWindow, register_state, walk_windows_from_history};
use crate::Error;
use crate::tables::MaxRowsConfig;
use crate::tables::trace_builder::Traces;
use crate::tests::windowed_builder_tests::{run, same_traces, whole};

/// The asm programs every record-mode test runs: every ECALL the walk
/// handles but HINT (the Rust guests below have it), misaligned and
/// page-crossing accesses.
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

/// A Rust guest's ELF from `program_artifacts/rust`.
fn rust_elf(name: &str) -> Elf {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../executor/program_artifacts/rust")
        .join(format!("{name}.elf"));
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    Elf::load(&bytes).expect("the ELF loads")
}

fn regs_equal(a: &RegisterState, b: &RegisterState) -> bool {
    a.regs == b.regs && a.index_register == b.index_register && a.pc_register == b.pc_register
}

/// Runs `program` on the fast engine in history mode and, beside it, the
/// reference executor and the lean pass, in windows of `window` cycles: every
/// window's logs, its starting register cells and the bytes it touches first
/// with their timestamps and values before it must be the lean pass's and the
/// reference memory's. Returns the cycles run.
fn history_is_the_lean_passs(name: &str, program: &Elf, input: &[u8], window: usize) -> usize {
    let builder =
        WindowedTraceBuilder::new(program, input, &MaxRowsConfig::small()).expect("the builder");
    let mut pass =
        LeanPass::new(Arc::clone(&builder.decode), &builder.register_state).expect("the pass");
    let mut reference = Executor::new(program, input.to_vec()).expect("the executor");
    let mut fast = FastExecutor::new(program, input.to_vec()).expect("the fast engine");
    fast.enable_history().expect("history mode");
    let (mut first, mut k) = (0usize, 0u32);
    loop {
        let start = reference.memory().snapshot();
        let want = reference
            .resume_with_limit(window)
            .expect("the reference run")
            .map(<[Log]>::to_vec);
        let got = fast.resume_history(window).expect("the history run");
        let (want, (logs, history)) = match (want, got) {
            (None, None) => break,
            (Some(w), Some(g)) => (w, g),
            (w, g) => panic!(
                "{name}/{window}: one run ended first: {} {}",
                w.is_some(),
                g.is_some()
            ),
        };
        let record = |l: &Log| [l.current_pc, l.next_pc, l.src1_val, l.src2_val, l.dst_val];
        assert!(
            want.iter().map(record).eq(logs.iter().map(record)),
            "{name}/{window}: window {k}'s logs"
        );
        assert_eq!(
            history.first, first as u64,
            "{name}/{window}: window {k}'s first cycle"
        );
        assert_eq!(
            history.extras.len(),
            logs.len(),
            "{name}/{window}: window {k}'s records"
        );
        let registers = register_state(history.registers.as_ref().expect("registers"));
        assert!(
            regs_equal(&registers, &pass.registers()),
            "{name}/{window}: window {k}'s starting registers"
        );
        // The last window is the finish's; the pass passes every other. After
        // it, the cells HALT reads are the run end's.
        if logs.last().is_some_and(|l| l.next_pc == 0) {
            pass.begin(k).expect("a window");
            pass.pass(&want, first).expect("the pass");
            let end = fast.take_history_final().expect("the run's end");
            assert!(
                regs_equal(&register_state(&end.registers), &pass.registers()),
                "{name}/{window}: the register cells at the end"
            );
            break;
        }
        pass.begin(k).expect("a window");
        pass.pass(&want, first).expect("the pass");
        let mut entries = pass.take_entries();
        let touches = history.first_touches();
        for &(address, value, _) in &touches {
            assert_eq!(
                value,
                start.load_byte(address),
                "{name}/{window}: window {k}: the value before it at {address:#x}"
            );
        }
        let mut mine: Vec<(u64, u64)> = touches.iter().map(|&(a, _, t)| (a, t)).collect();
        entries.sort_unstable();
        mine.sort_unstable();
        assert_eq!(mine, entries, "{name}/{window}: window {k}'s first touches");
        first += logs.len();
        k += 1;
    }
    first
}

/// ★ Every asm program and Rust guest, and the ethrex fixtures, at window
/// lengths from one cycle to many: the fast engine's history is the lean
/// pass's and its logs the reference's.
#[test]
fn the_executors_history_is_the_lean_passs() {
    for name in PROGRAMS {
        let (program, _) = run(name);
        for window in [1, 7, 32, 33, 4096] {
            history_is_the_lean_passs(name, &program, &[], window);
        }
    }
    for name in [
        "hint_min",
        "hint_multi",
        "dma_memset_cases",
        "dma_memmove_cases",
        "keccak_precompile",
    ] {
        let program = rust_elf(name);
        for window in [97, 4096] {
            history_is_the_lean_passs(name, &program, &[], window);
        }
    }
    let ethrex = rust_elf("ethrex");
    let fixtures = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../executor/tests");
    for fixture in ["ethrex_empty_block", "ethrex_10_transfers"] {
        let input = std::fs::read(fixtures.join(format!("{fixture}.bin"))).expect("the fixture");
        let cycles = history_is_the_lean_passs(fixture, &ethrex, &input, 1 << 16);
        assert!(cycles > 100_000, "{fixture}: {cycles} cycles");
    }
}

/// A mistake put into the history on purpose, to show a check catches it.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Tamper {
    None,
    /// The `nth` uniform load or store of the first window from `window` that
    /// has one, one timestamp later.
    LaterAccess {
        window: usize,
        nth: usize,
    },
    /// The `nth` uniform store of the first window from `window` that has
    /// one, its old bytes changed.
    OtherValue {
        window: usize,
        nth: usize,
    },
    /// The first side cell of the first window from `window` that has one,
    /// one timestamp later.
    LaterEcallCell {
        window: usize,
    },
    /// Window `window` starting with x5 one larger.
    Register {
        window: usize,
    },
    /// The `nth` cycle reading `rs1` of the first window from `window` that
    /// has one, its old timestamp one later.
    RegisterTs {
        window: usize,
        nth: usize,
    },
    /// The first ECALL register cell of the first window from `window` that
    /// has one, one timestamp later.
    EcallRegister {
        window: usize,
    },
}

fn tamper_with(
    history: &mut WindowHistory,
    logs: &[Log],
    code: &InstructionCache,
    k: usize,
    tamper: Tamper,
) -> bool {
    // A store's old bytes are its MEMW row's old values (a load's row takes
    // the loaded value, its old bytes are never read).
    let stores = |h: &WindowHistory, nth: usize| {
        h.extras
            .iter()
            .zip(logs)
            .enumerate()
            .filter(|(_, (e, log))| {
                e.side == UNIFORM
                    && matches!(code.get(log.current_pc), Some(Instruction::Store { .. }))
            })
            .nth(nth)
            .map(|(i, _)| i)
    };
    let uniform = |h: &WindowHistory, nth: usize| {
        h.extras
            .iter()
            .enumerate()
            .filter(|(_, e)| e.side == UNIFORM)
            .nth(nth)
            .map(|(i, _)| i)
    };
    match tamper {
        Tamper::LaterAccess { window, nth } if k >= window => {
            if let Some(i) = uniform(history, nth) {
                history.extras[i].ts += 1;
                return true;
            }
        }
        Tamper::OtherValue { window, nth } if k >= window => {
            if let Some(i) = stores(history, nth) {
                history.extras[i].old ^= 0x5a;
                return true;
            }
        }
        Tamper::LaterEcallCell { window } if k >= window => {
            if let Some(ts) = history.ts.first_mut() {
                *ts += 1;
                return true;
            }
        }
        Tamper::Register { window } if k >= window => {
            if let Some(cells) = &mut history.registers {
                cells.regs[5].0 += 1;
                return true;
            }
        }
        Tamper::RegisterTs { window, nth } if k >= window => {
            if let Some(e) = history
                .extras
                .iter_mut()
                .filter(|e| e.regs_ts[0] > 0)
                .nth(nth)
            {
                e.regs_ts[0] += 1;
                return true;
            }
        }
        Tamper::EcallRegister { window } if k >= window => {
            if let Some(cell) = history.reg_cells.first_mut() {
                cell.ts += 1;
                return true;
            }
        }
        _ => {}
    }
    false
}

/// The run of `program` built by windows of `window` cycles walked from the
/// fast engine's history on `walkers` threads ([`walk_windows_from_history`],
/// re-walking every `recheck`-th window), the executor on a thread of its own
/// as the block's prove runs it, its history passed through `tamper`.
fn walked_from_history(
    program: &Elf,
    input: &[u8],
    max_rows: &MaxRowsConfig,
    window: usize,
    walkers: usize,
    recheck: usize,
    tamper: Tamper,
) -> Result<Traces, Error> {
    let mut builder = WindowedTraceBuilder::new(program, input, max_rows)?.drop_streamed_ops()?;
    let (tx, rx) = std::sync::mpsc::sync_channel::<ExecWindow>(2);
    let mut chunks = Vec::new();
    let tampered = std::sync::atomic::AtomicBool::new(false);
    let tampered = &tampered;
    let report = std::thread::scope(|scope| {
        scope.spawn(move || {
            let mut fast = FastExecutor::new(program, input.to_vec()).expect("the fast engine");
            fast.enable_history().expect("history mode");
            let code = InstructionCache::new(&program.data).expect("the program decodes");
            let mut k = 0;
            loop {
                // A re-walked window comes with its start memory.
                let start = (recheck > 0 && k % recheck == 0).then(|| fast.snapshot());
                let Some((logs, mut history)) = fast.resume_history(window).expect("the run")
                else {
                    break;
                };
                // The first window from the tamper's that it applies to, but
                // the run's last: the finish walks that one from a state its
                // own history starts (see `finish_state`).
                if !tampered.load(std::sync::atomic::Ordering::Relaxed)
                    && fast.pc() != 0
                    && tamper_with(&mut history, &logs, &code, k, tamper)
                {
                    tampered.store(true, std::sync::atomic::Ordering::Relaxed);
                }
                let last = (fast.pc() == 0)
                    .then(|| fast.take_history_final().map(Box::new))
                    .flatten();
                let sent = tx.send(ExecWindow {
                    logs: Arc::new(logs),
                    start,
                    history: Some(Box::new(history)),
                    last,
                });
                if sent.is_err() {
                    break;
                }
                k += 1;
            }
        });
        walk_windows_from_history(
            &mut builder,
            rx,
            walkers,
            recheck,
            |accumulator, walked, _| {
                chunks.extend(
                    accumulator
                        .absorb(walked)
                        .into_iter()
                        .map(ChunkJob::generate),
                );
                Ok(())
            },
        )
    });
    assert_eq!(
        tampered.load(std::sync::atomic::Ordering::Relaxed),
        tamper != Tamper::None,
        "{tamper:?} applied"
    );
    let report = report?;
    assert!(report.from_history);
    assert_eq!(report.entries, 0);
    let mut traces = builder.finish(&report.last)?;
    traces.insert_streamed(chunks)?;
    Ok(traces)
}

/// ★ Walked from the executor's history on 1, 2 and 3 threads, at window
/// lengths 1, 7 and 33, rechecking every window or none, the windows build
/// the whole-run tables.
#[test]
fn windows_walked_from_history_build_the_whole_run_tables() {
    let max_rows = MaxRowsConfig::small();
    for name in PROGRAMS {
        let (program, logs) = run(name);
        let reference = whole(&program, &logs, &max_rows);
        for window in [1, 7, 33] {
            for (walkers, recheck) in [(1, 0), (2, 1), (3, 0), (3, 2)] {
                let traces = walked_from_history(
                    &program,
                    &[],
                    &max_rows,
                    window,
                    walkers,
                    recheck,
                    Tamper::None,
                )
                .unwrap_or_else(|e| {
                    panic!("{name}, window {window}, {walkers} walkers, recheck {recheck}: {e:?}")
                });
                same_traces(&reference, &traces);
            }
        }
    }
}

/// ★ A Rust guest with HINT and DMA calls, walked from history, builds the
/// whole-run tables.
#[test]
fn rust_guests_walked_from_history_build_the_whole_run_tables() {
    let max_rows = MaxRowsConfig::default();
    for name in ["hint_min", "dma_memset_cases"] {
        let program = rust_elf(name);
        let logs = Executor::new(&program, Vec::new())
            .expect("the executor")
            .run()
            .expect("the run")
            .logs;
        let reference = whole(&program, &logs, &max_rows);
        let traces = walked_from_history(&program, &[], &max_rows, 1 << 12, 3, 3, Tamper::None)
            .unwrap_or_else(|e| panic!("{name}: {e:?}"));
        same_traces(&reference, &traces);
    }
}

/// ★ With every window rechecked, each mistake in the history is refused;
/// a wrong starting register is refused with no recheck at all.
#[test]
fn a_wrong_history_is_refused() {
    let max_rows = MaxRowsConfig::small();
    let windows = 8;
    let cases = [
        (
            "all_loadstore_32",
            Tamper::LaterAccess { window: 1, nth: 0 },
            1,
        ),
        (
            "all_loadstore_32",
            Tamper::LaterAccess { window: 2, nth: 1 },
            1,
        ),
        (
            "all_loadstore_32",
            Tamper::OtherValue { window: 1, nth: 0 },
            1,
        ),
        (
            "all_loadstore_32",
            Tamper::OtherValue { window: 2, nth: 1 },
            1,
        ),
        ("test_copy_ecalls", Tamper::LaterEcallCell { window: 1 }, 1),
        ("all_loadstore_32", Tamper::Register { window: 2 }, 1),
        ("all_loadstore_32", Tamper::Register { window: 2 }, 0),
        (
            "all_loadstore_32",
            Tamper::RegisterTs { window: 1, nth: 2 },
            0,
        ),
        ("test_copy_ecalls", Tamper::EcallRegister { window: 1 }, 0),
        ("test_commit_split", Tamper::EcallRegister { window: 0 }, 0),
    ];
    for (name, tamper, recheck) in cases {
        let (program, _) = run(name);
        let built = walked_from_history(&program, &[], &max_rows, windows, 2, recheck, tamper);
        assert!(
            matches!(&built, Err(Error::Prover(_))),
            "{name}: {tamper:?}, recheck {recheck}: {:?}",
            built.as_ref().err()
        );
        // The control: no tampering, every window rechecked.
        walked_from_history(&program, &[], &max_rows, windows, 2, 1, Tamper::None)
            .expect("an untouched history");
    }
}
