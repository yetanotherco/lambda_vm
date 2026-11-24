//! `LAMBDA_VM_HOST_HOIST=state`: record mode's final memory state built on
//! the builder's setup state, its new pages on the pool and the run's end read
//! as the history keeps its timestamps, is the state [`finish_state`] builds,
//! under the byte and the word history, on runs whose last window touches
//! fresh pages, image pages and words across pages; a setup state that is not
//! the image's is refused; and a record-mode build with it on makes the
//! tables it makes with it off, the hoisted state built once, never with the
//! knob off.

use std::sync::Arc;

use executor::elf::Elf;
use executor::vm::execution::Executor;
use executor::vm::fast::FastExecutor;
use executor::vm::history::WindowHistory;

use super::super::super::{HostHoist, MemoryState, build_initial_image_paged};
use super::super::{ChunkJob, WindowedTraceBuilder};
use super::{ExecWindow, RunEnd, finish_state, finish_state_hoisted, walk_windows_from_history};
use crate::Error;
use crate::tables::MaxRowsConfig;
use crate::tables::fused_memory::MemoryFormat;
use crate::tables::memw_aligned::MemwAlignedOrder;
use crate::tables::register::RegisterFormat;
use crate::tables::trace_builder::Traces;
use crate::tables::trace_builder::host_hoist::calls;
use crate::test_utils::asm_elf_bytes;
use crate::tests::windowed_builder_tests::same_traces;

/// The hoisted-state counter is process-wide: these tests take turns.
fn turn() -> std::sync::MutexGuard<'static, ()> {
    calls::turn()
}

/// Programs whose last windows touch fresh pages (`page_runs`, the
/// `*_4gib` runs past the image), image pages and words across a page end.
const PROGRAMS: [&str; 6] = [
    "fused_mem_cases",
    "page_runs",
    "cross_4gib",
    "copy_4gib",
    "all_loadstore_32",
    "test_keccak256",
];

fn program(name: &str) -> Elf {
    Elf::load(&asm_elf_bytes(name)).expect("the ELF loads")
}

/// `program` run on the fast engine by windows of `window` cycles under the
/// byte or the word history: the last window's history and the run's end.
fn last_window(program: &Elf, input: &[u8], window: usize, words: bool) -> (WindowHistory, RunEnd) {
    let mut fast = FastExecutor::new(program, input.to_vec()).expect("the fast engine");
    if words {
        fast.enable_word_history().expect("the word history");
    } else {
        fast.enable_history().expect("the history");
    }
    let mut last = None;
    loop {
        let next = if words {
            fast.resume_word_history(window).expect("the run")
        } else {
            fast.resume_history(window).expect("the run")
        };
        let Some((_, history)) = next else {
            break;
        };
        last = Some(history);
        if fast.pc() == 0 {
            break;
        }
    }
    let end = if words {
        RunEnd::Words(fast.take_word_history_final().expect("the run's end"))
    } else {
        RunEnd::Bytes(fast.take_history_final().expect("the run's end"))
    };
    (last.expect("a window"), end)
}

/// A state's pages, each page's cells and its set cells.
fn same_state(what: &str, a: &MemoryState, b: &MemoryState) {
    let bases: Vec<u64> = a.cells.page_bases().collect();
    assert_eq!(
        bases,
        b.cells.page_bases().collect::<Vec<_>>(),
        "{what}: pages"
    );
    for base in bases {
        assert!(
            a.cells.page_data(base) == b.cells.page_data(base),
            "{what}: page {base:#x}"
        );
    }
    assert!(a.cells.iter().eq(b.cells.iter()), "{what}: the set cells");
    assert_eq!(a.lean, b.lean, "{what}: lean");
    assert!(
        a.overlay.is_none() && b.overlay.is_none(),
        "{what}: overlay"
    );
}

/// ★ The hoisted final state is [`finish_state`]'s, page for page and cell
/// for cell, under both histories, at windows that leave a short last
/// window, a long one and the whole run in one, with and without a private
/// input; across the runs the state holds pages the image lacks and image
/// pages written by the run.
#[test]
fn the_hoisted_final_state_is_the_final_state() {
    let _turn = turn();
    let mut fresh = 0usize;
    let mut written_image_pages = 0usize;
    for name in PROGRAMS {
        let program = program(name);
        for input in [Vec::new(), (0..300u32).map(|i| i as u8).collect()] {
            let image = build_initial_image_paged(&program, &input);
            for words in [false, true] {
                for window in [7, 33, 1 << 20] {
                    for lean in [false, true] {
                        let what = format!(
                            "{name}, {} input bytes, words {words}, window {window}, lean {lean}",
                            input.len()
                        );
                        let (history, end) = last_window(&program, &input, window, words);
                        let (want, _) = finish_state(&end, &history, &image, lean);
                        let (got, _) = finish_state_hoisted(
                            &end,
                            &history,
                            &image,
                            MemoryState::from_pages(&image),
                            lean,
                        )
                        .expect("the image's state");
                        same_state(&what, &got, &want);
                        let image_bases: Vec<u64> = image.page_bases().collect();
                        fresh += got
                            .cells
                            .page_bases()
                            .filter(|base| image_bases.binary_search(base).is_err())
                            .count();
                        written_image_pages += image_bases
                            .iter()
                            .filter(|&&base| {
                                got.cells
                                    .page_data(base)
                                    .is_some_and(|cells| cells.iter().any(|&(_, ts)| ts > 0))
                            })
                            .count();
                    }
                }
            }
        }
    }
    assert!(fresh > 0, "a page the image lacks");
    assert!(written_image_pages > 0, "an image page the run wrote");
}

/// A setup state that is not the image's (a cell more, a page more, or
/// another program's image) is refused, not built on.
#[test]
fn a_setup_state_not_the_images_is_refused() {
    let _turn = turn();
    let program = program("fused_mem_cases");
    let image = build_initial_image_paged(&program, &[]);
    let (history, end) = last_window(&program, &[], 33, true);
    let set: std::collections::HashSet<u64> = image.iter().map(|(addr, _)| addr).collect();
    let first = image.page_bases().next().expect("an image page");
    let unset = (first..)
        .find(|addr| !set.contains(addr))
        .expect("an unset byte");
    let mut cell_more = build_initial_image_paged(&program, &[]);
    cell_more.set(unset, 0);
    let mut page_more = build_initial_image_paged(&program, &[]);
    page_more.set(0x7000_0000, 1);
    let other = build_initial_image_paged(&self::program("page_runs"), &[]);
    for (what, wrong) in [
        ("a cell more", &cell_more),
        ("a page more", &page_more),
        ("another image", &other),
    ] {
        let refused =
            finish_state_hoisted(&end, &history, &image, MemoryState::from_pages(wrong), true);
        assert!(
            matches!(&refused, Err(Error::Prover(m)) if m.contains("not the image's")),
            "{what}: {:?}",
            refused.as_ref().err()
        );
    }
    finish_state_hoisted(
        &end,
        &history,
        &image,
        MemoryState::from_pages(&image),
        true,
    )
    .expect("the image's own state");
}

/// `program` on `input` built in record mode by windows of `window` cycles
/// on `walkers` threads, from the byte history or (`words`, the word memory
/// format) the word history, the builder's parts `hoist`.
fn record_build(
    program: &Elf,
    input: &[u8],
    window: usize,
    walkers: usize,
    words: bool,
    hoist: HostHoist,
) -> Result<Traces, Error> {
    let max_rows = MaxRowsConfig::small();
    let mut builder =
        WindowedTraceBuilder::new_with(program, input, &max_rows, hoist)?.drop_streamed_ops()?;
    if words {
        builder = builder
            .register_format(RegisterFormat::InRow)?
            .memw_aligned_order(MemwAlignedOrder::Limbs)?
            .keccak_rnd_form(crate::block_whir::BlockFormat::production().keccak_rnd)
            .ecdas_form(crate::block_whir::BlockFormat::production().ecdas)
            .memory_format(MemoryFormat::Words)?
            .light_walk(2)?;
    }
    let (tx, rx) = std::sync::mpsc::sync_channel::<ExecWindow>(2);
    let mut chunks = Vec::new();
    let report = std::thread::scope(|scope| {
        scope.spawn(move || {
            let mut fast = FastExecutor::new(program, input.to_vec()).expect("the fast engine");
            if words {
                fast.enable_word_history().expect("the word history");
            } else {
                fast.enable_history().expect("the history");
            }
            loop {
                let next = if words {
                    fast.resume_word_history(window).expect("the run")
                } else {
                    fast.resume_history(window).expect("the run")
                };
                let Some((logs, history)) = next else {
                    break;
                };
                let last = (fast.pc() == 0)
                    .then(|| {
                        if words {
                            fast.take_word_history_final().map(RunEnd::boxed_words)
                        } else {
                            fast.take_history_final().map(RunEnd::boxed)
                        }
                    })
                    .flatten();
                let sent = tx.send(ExecWindow {
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
        walk_windows_from_history(&mut builder, rx, walkers, 0, |accumulator, walked, _| {
            chunks.extend(
                accumulator
                    .absorb(walked)
                    .into_iter()
                    .map(ChunkJob::generate),
            );
            Ok(())
        })
    })?;
    let mut traces = builder.finish(&report.last)?;
    traces.insert_streamed(chunks)?;
    Ok(traces)
}

/// ★ A record-mode build with the final state hoisted (alone and with every
/// part) makes the tables it makes with none, under both histories, with and
/// without a private input: the hoisted state built once a build, and never
/// with the knob off.
#[test]
fn record_mode_with_the_state_hoisted_builds_the_same_tables() {
    let _turn = turn();
    let _ecrec = crate::tables::trace_builder::ecrec_forge_free();
    // Counted: another test's block prove beside this one, under the knob's
    // default, hoists a state this count must not see.
    let state = HostHoist {
        state: true,
        ..HostHoist::none()
    }
    .counted();
    for name in [
        "fused_mem_cases",
        "page_runs",
        "cross_4gib",
        "test_keccak256",
    ] {
        let program = program(name);
        for input in [Vec::new(), (0..300u32).map(|i| i as u8).collect()] {
            // The executor must run the program on this input.
            Executor::new(&program, input.clone())
                .expect("the executor starts")
                .run()
                .expect("the program runs");
            for words in [false, true] {
                for (window, walkers) in [(7, 1), (33, 3)] {
                    let what = format!(
                        "{name}, {} input bytes, words {words}, window {window}, {walkers} walkers",
                        input.len()
                    );
                    let _ = calls::take_states();
                    let want = record_build(
                        &program,
                        &input,
                        window,
                        walkers,
                        words,
                        HostHoist::none().counted(),
                    )
                    .unwrap_or_else(|e| panic!("{what}: {e:?}"));
                    assert_eq!(calls::take_states(), 0, "{what}: off never hoists");
                    for hoist in [state, HostHoist::all().counted()] {
                        let got = record_build(&program, &input, window, walkers, words, hoist)
                            .unwrap_or_else(|e| panic!("{what}, {hoist:?}: {e:?}"));
                        assert_eq!(calls::take_states(), 1, "{what}, {hoist:?}: built once");
                        same_traces(&want, &got);
                    }
                }
            }
        }
    }
}

/// The state part takes the builder's setup state only before the run's
/// first cycle: a sequential build (no record mode) with it on never builds
/// the hoisted state.
#[test]
fn the_state_part_outside_record_mode_changes_nothing() {
    let _turn = turn();
    let program = program("fused_mem_cases");
    let logs = Executor::new(&program, Vec::new())
        .expect("the executor starts")
        .run()
        .expect("the program runs")
        .logs;
    let max_rows = MaxRowsConfig::small();
    let build = |hoist| {
        let mut builder =
            WindowedTraceBuilder::new_with(&program, &[], &max_rows, hoist).expect("the builder");
        let body = logs.len() - 1;
        let cut = body - body % 33;
        let mut chunks = Vec::new();
        for w in logs[..cut].chunks(33) {
            chunks.extend(builder.push(w).expect("a window"));
        }
        let mut traces = builder.finish(&logs[cut..]).expect("the finish");
        traces.insert_streamed(chunks).expect("placeholders");
        traces
    };
    let _ = calls::take_states();
    let want = build(HostHoist::none().counted());
    let got = build(
        HostHoist {
            state: true,
            ..HostHoist::none()
        }
        .counted(),
    );
    assert_eq!(calls::take_states(), 0);
    same_traces(&want, &got);
}
