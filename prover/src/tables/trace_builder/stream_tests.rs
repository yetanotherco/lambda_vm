//! ★ The streamed finish ([`super::FinishPlan::emit_streamed`]) hands on the
//! tables the monolithic finish ([`super::FinishPlan::emit_all`]) builds, word
//! for word, in the run's AIR order with every streamed chunk's slot in its
//! place, at any byte budget (one byte, 2 GiB, none). The byte gate's
//! lowest-pending rule keeps a one-byte budget from deadlocking; a job's error
//! and a stopped packer each stop the emitter and every waiter; a header that
//! miscounts is refused. The cut tables' padding rule
//! ([`CutPadding`]) is one of the configurations: the header, the stream and
//! the lists kept for phase B count the same tables under either rule. Built
//! as the gate admits each table instead of in waves
//! ([`WindowedTraceBuilder::slide_rest`]), the stream is the same, and a slow
//! table holds back only the hand-on of the tables after it.

use std::sync::Arc;
use std::time::Duration;

use executor::elf::Elf;
use executor::vm::execution::Executor;
use executor::vm::logs::Log;

use super::super::gate::ByteGate;
use super::super::*;
use super::{Emitted, FAMILIES, Job, Unit, emit_streamed_units};
use crate::tables::MaxRowsConfig;
use crate::test_utils::asm_elf_bytes;

type Table = TraceTable<GoldilocksField, GoldilocksExtension>;

fn run(name: &str) -> (Elf, Vec<Log>) {
    let program = Elf::load(&asm_elf_bytes(name)).expect("the ELF loads");
    let logs = Executor::new(&program, Vec::new())
        .expect("the executor starts")
        .run()
        .expect("the program runs")
        .logs;
    (program, logs)
}

/// `f` on a thread of its own; `None` when it has not returned within `secs`
/// (the thread is left behind: a hang fails the test instead of the suite).
fn within<T: Send + 'static>(secs: u64, f: impl FnOnce() -> T + Send + 'static) -> Option<T> {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(f());
    });
    rx.recv_timeout(Duration::from_secs(secs)).ok()
}

/// A table as (columns, rows, words row-major), held packed or wide.
fn shape_and_words(t: &Table) -> (usize, usize, Vec<u64>) {
    match t.narrow_main() {
        Some(packed) => {
            let columns: Vec<Vec<u64>> = (0..packed.cols()).map(|c| packed.column(c)).collect();
            let words = (0..packed.rows())
                .flat_map(|r| columns.iter().map(move |column| column[r]))
                .collect();
            (packed.cols(), packed.rows(), words)
        }
        None => {
            let main = &t.main_table;
            let words = (0..main.height)
                .flat_map(|r| (0..main.width).map(move |c| *main.get(r, c).value()))
                .collect();
            (main.width, main.height, words)
        }
    }
}

/// A build's tables in the run's AIR order (`VmAirs::air_trace_pairs`, HALT
/// in, BLAKE3 as the counts include it), streamed chunks' slots as `None`.
fn air_order(t: &Traces) -> Vec<Option<&Table>> {
    fn slot(table: &Table) -> Option<&Table> {
        (table.main_table.width != 0).then_some(table)
    }
    let mut out = vec![
        slot(&t.bitwise),
        slot(&t.decode),
        slot(&t.keccak_rc),
        slot(&t.register),
        slot(&t.halt),
    ];
    for list in [
        &t.copies,
        &t.keccaks,
        &t.keccak_rnds,
        &t.ecsms,
        &t.ecdases,
        &t.hints,
    ] {
        out.extend(list.iter().map(slot));
    }
    if t.table_counts().blake3 == 1 {
        out.push(slot(&t.blake3));
    }
    for list in [
        &t.is_whs,
        &t.cpus,
        &t.lts,
        &t.shifts,
        &t.memws,
        &t.memw_aligneds,
        &t.loads,
        &t.muls,
        &t.dvrms,
        &t.branches,
        &t.pages,
        &t.memw_registers,
        &t.eqs,
        &t.bytewises,
        &t.stores,
        &t.cpu32s,
    ] {
        out.extend(list.iter().map(slot));
    }
    out
}

type Configure = Box<dyn Fn(WindowedTraceBuilder<'_>) -> WindowedTraceBuilder<'_>>;

/// The windowed configurations the block uses (as the plan's tests do).
fn configs() -> Vec<(&'static str, Configure)> {
    vec![
        ("plain", Box::new(|b| b)),
        (
            "dropped",
            Box::new(|b| b.drop_streamed_ops().expect("before any window")),
        ),
        (
            "dropped, raw + concatenated LT",
            Box::new(|b| {
                b.drop_streamed_ops()
                    .expect("before any window")
                    .raw_memw_lt()
                    .expect("before any window")
                    .concat_lt()
            }),
        ),
        (
            "dropped, KECCAK_RND at the finish, cuts, packed, written packed",
            Box::new(|b| {
                b.keccak_rnd_chunks_at_finish(32)
                    .expect("rows")
                    .cuts_at_finish(2, 4, 2, 2)
                    .expect("rows")
                    .pack_finished_tables()
                    .generate_packed()
                    .drop_streamed_ops()
                    .expect("before any window")
            }),
        ),
        (
            "KECCAK_RND streamed, packed",
            Box::new(|b| {
                b.keccak_rnd_chunks(32)
                    .expect("rows")
                    .pack_finished_tables()
                    .drop_streamed_ops()
                    .expect("before any window")
            }),
        ),
        (
            "MEMW-derived LT streamed",
            Box::new(|b| {
                b.stream_memw_lt()
                    .drop_streamed_ops()
                    .expect("before any window")
            }),
        ),
        (
            "dropped, KECCAK_RND at the finish, cuts, packed, written packed, cut at the last \
             real row",
            Box::new(|b| cut(b, CutPadding::LastTable)),
        ),
    ]
}

/// The block's configuration with every family cut small (KECCAK_RND at 32
/// rows, KECCAK and ECSM at one row, ECDAS at 16), the cut tables padded as
/// `padding`.
fn cut(b: WindowedTraceBuilder<'_>, padding: CutPadding) -> WindowedTraceBuilder<'_> {
    b.keccak_rnd_chunks_at_finish(32)
        .expect("rows")
        .cuts_at_finish(1, 1, 16, 1)
        .expect("rows")
        .cut_padding(padding)
        .pack_finished_tables()
        .generate_packed()
        .drop_streamed_ops()
        .expect("before any window")
}

/// ★ The header, the streamed tables and the lists kept for phase B count the
/// same KECCAK_RND tables under either padding rule: test_keccak_multi's 72
/// rows at 32 rows a table are four tables with the whole run padded and three
/// cut at the last real row, and phase B's jobs build each of them again word
/// for word. KECCAK (3 rows at one row a table) is four tables against three.
#[test]
fn the_header_the_stream_and_the_rebuild_count_the_rules_tables() {
    use super::RegenFamily;
    let (program, logs) = run("test_keccak_multi");
    let max_rows = MaxRowsConfig::small();
    for (padding, tables) in [(CutPadding::WholeRun, 4usize), (CutPadding::LastTable, 3)] {
        let what = format!("{padding:?}");
        let configure = move |b| cut(b, padding);
        let (builder, last) = pushed(&program, &logs, &max_rows, 7, &configure);
        let whole = builder.finish(last).expect("the monolithic finish");
        assert_eq!(
            whole.keccak_rnds.len(),
            tables,
            "{what}: the finish's tables"
        );
        assert_eq!(whole.keccaks.len(), tables, "{what}: KECCAK's tables");

        let (builder, last) = pushed(&program, &logs, &max_rows, 7, &configure);
        let (header, got, _) = streamed(builder, last, usize::MAX);
        assert_eq!(header.table_counts.keccak_rnd, tables, "{what}: the header");
        assert_eq!(header.table_counts.keccak, tables, "{what}: the header");
        assert_eq!(
            got.len(),
            air_order(&whole).len(),
            "{what}: the stream's tables"
        );

        let (builder, last) = pushed(&program, &logs, &max_rows, 7, &configure);
        let (got, regen) = streamed_keeping(builder, last, true);
        let jobs = regen
            .expect("the lists kept")
            .take(RegenFamily::KeccakRnd)
            .expect("the jobs");
        assert_eq!(jobs.len(), tables, "{what}: phase B's jobs");
        let first = regen_ranges(&whole)[0].start;
        for (j, job) in jobs.into_iter().enumerate() {
            let again = job.expect("no chunk was streamed")().expect("the job builds again");
            let table = got[first + j].table.as_ref().expect("a table, not a slot");
            assert_eq!(
                shape_and_words(&again),
                shape_and_words(table),
                "{what}: KECCAK_RND[{j}] built again"
            );
        }
    }
}

/// A windowed builder over windows of `window` cycles, `configure` applied,
/// every window but the last pushed: the last window's logs with it.
fn pushed<'p>(
    program: &'p Elf,
    logs: &'p [Log],
    max_rows: &MaxRowsConfig,
    window: usize,
    configure: &dyn Fn(WindowedTraceBuilder<'p>) -> WindowedTraceBuilder<'p>,
) -> (WindowedTraceBuilder<'p>, &'p [Log]) {
    let mut builder =
        configure(WindowedTraceBuilder::new(program, &[], max_rows).expect("the builder"));
    let body = logs.len() - 1;
    let cut = body - body % window;
    for w in logs[..cut].chunks(window) {
        builder.push(w).expect("a window");
    }
    (builder, &logs[cut..])
}

/// What a streamed finish handed on, with the header, the gate placing each
/// table as it arrives (the packer's part).
fn streamed(
    builder: WindowedTraceBuilder<'_>,
    last: &[Log],
    budget: usize,
) -> (RestHeader, Vec<Emitted>, usize) {
    let gate = ByteGate::new(budget);
    let mut header = None;
    let mut out = Vec::new();
    let mut placed = 0usize;
    builder
        .finish_streamed(last, &gate, |h| {
            header = Some(h);
            Ok(Box::new(|emitted: Emitted| {
                if emitted.table.is_some() {
                    gate.release(placed);
                    placed += 1;
                }
                out.push(emitted);
                Ok(())
            }))
        })
        .expect("the streamed finish");
    let most = gate.most();
    (header.expect("the header came first"), out, most)
}

/// [`streamed`] with the lists KECCAK_RND's and LT's tables are built from
/// kept (`keep`), and what was kept.
fn streamed_keeping(
    builder: WindowedTraceBuilder<'_>,
    last: &[Log],
    keep: bool,
) -> (Vec<Emitted>, Option<super::RestRegen>) {
    let gate = ByteGate::new(usize::MAX);
    let mut out = Vec::new();
    let mut placed = 0usize;
    let regen = builder
        .finish_streamed_keeping(last, &gate, keep, |_| {
            Ok(Box::new(|emitted: Emitted| {
                if emitted.table.is_some() {
                    gate.release(placed);
                    placed += 1;
                }
                out.push(emitted);
                Ok(())
            }))
        })
        .expect("the streamed finish");
    (out, regen)
}

/// The AIR positions of a build's KECCAK_RND tables and of its LT tables
/// (`air_order`'s layout).
fn regen_ranges(t: &Traces) -> [std::ops::Range<usize>; 2] {
    let kr = 5 + t.copies.len() + t.keccaks.len();
    let kr_end = kr + t.keccak_rnds.len();
    let blake3 = usize::from(t.table_counts().blake3 == 1);
    let lt = kr_end
        + t.ecsms.len()
        + t.ecdases.len()
        + t.hints.len()
        + blake3
        + t.is_whs.len()
        + t.cpus.len();
    [kr..kr_end, lt..lt + t.lts.len()]
}

/// ★ N3: what phase B builds again — KECCAK_RND's and LT's tables, from the
/// lists the streamed finish kept ([`super::RestRegen`]) — is what the finish
/// handed on, word for word and packed byte for byte, for every table of the
/// two families, in every windowed configuration; a streamed chunk's slot
/// stays a slot; a family is taken once; nothing is kept unless asked.
#[test]
fn the_kept_lists_build_the_finishs_keccak_rnd_and_lt_tables_again() {
    use super::RegenFamily;
    let mut rebuilt = [0usize; 2];
    for max_rows in [MaxRowsConfig::small(), MaxRowsConfig::uniform(4)] {
        for name in [
            "test_keccak_multi",
            "all_instructions_64",
            "test_ecsm_multi",
        ] {
            let (program, logs) = run(name);
            for window in [7, 1000] {
                for (config, configure) in &configs() {
                    let what = format!("{name} window {window} {config}");
                    let (builder, last) =
                        pushed(&program, &logs, &max_rows, window, configure.as_ref());
                    let whole = builder.finish(last).expect("the monolithic finish");
                    let ranges = regen_ranges(&whole);
                    let (builder, last) =
                        pushed(&program, &logs, &max_rows, window, configure.as_ref());
                    let (got, regen) = streamed_keeping(builder, last, true);
                    let mut regen = regen.expect("the lists kept");
                    for (f, (family, range)) in RegenFamily::ALL.into_iter().zip(ranges).enumerate()
                    {
                        let jobs = regen.take(family).expect("the family's jobs");
                        assert_eq!(jobs.len(), range.len(), "{what}: {family:?}'s tables");
                        for (j, job) in jobs.into_iter().enumerate() {
                            let at = range.start + j;
                            match (job, &got[at].table) {
                                (None, None) => {}
                                (Some(job), Some(table)) => {
                                    let again = job().expect("the job builds again");
                                    assert_eq!(
                                        shape_and_words(&again),
                                        shape_and_words(table),
                                        "{what}: {family:?}[{j}]"
                                    );
                                    assert_eq!(
                                        again
                                            .narrow_main()
                                            .map(|p| (p.widths().to_vec(), p.data().to_vec())),
                                        table
                                            .narrow_main()
                                            .map(|p| (p.widths().to_vec(), p.data().to_vec())),
                                        "{what}: {family:?}[{j}] packed"
                                    );
                                    rebuilt[f] += 1;
                                }
                                (job, table) => panic!(
                                    "{what}: {family:?}[{j}] a job {} against a table {}",
                                    job.is_some(),
                                    table.is_some()
                                ),
                            }
                        }
                        assert!(
                            regen.take(family).is_err(),
                            "{what}: {family:?} taken twice"
                        );
                    }
                    let (builder, last) =
                        pushed(&program, &logs, &max_rows, window, configure.as_ref());
                    assert!(
                        streamed_keeping(builder, last, false).1.is_none(),
                        "{what}: kept unasked"
                    );
                }
            }
        }
    }
    assert!(
        rebuilt.iter().all(|&n| n > 0),
        "both families rebuilt: {rebuilt:?}"
    );
}

/// Every windowed configuration, chunked small and at four rows, windows of 7
/// and 1000 cycles, budgets of one byte, 2 GiB and none: the streamed finish
/// hands on, in AIR order, the monolithic finish's tables word for word and
/// its streamed slots, under its header.
#[test]
fn the_stream_is_the_monolithic_finish_in_air_order() {
    for max_rows in [MaxRowsConfig::small(), MaxRowsConfig::uniform(4)] {
        for name in [
            "all_instructions_64",
            "misalign_sd",
            "test_keccak_multi",
            "test_ecsm_multi",
            "test_commit_split",
            "test_blake3_absorb",
            "test_dense_pages",
        ] {
            let (program, logs) = run(name);
            for window in [7, 1000] {
                for (config, configure) in &configs() {
                    let (builder, last) =
                        pushed(&program, &logs, &max_rows, window, configure.as_ref());
                    let whole = builder.finish(last).expect("the monolithic finish");
                    let want = air_order(&whole);
                    for budget in [1usize, 2 << 30, usize::MAX] {
                        let what = format!("{name} window {window} {config} budget {budget}");
                        let (builder, last) =
                            pushed(&program, &logs, &max_rows, window, configure.as_ref());
                        let (header, got, _) = streamed(builder, last, budget);
                        assert_eq!(
                            format!("{:?}", header.table_counts),
                            format!("{:?}", whole.table_counts()),
                            "{what}: the header's counts"
                        );
                        assert_eq!(got.len(), want.len(), "{what}: the tables in AIR order");
                        for (k, (emitted, table)) in got.iter().zip(&want).enumerate() {
                            assert_eq!(emitted.position, k, "{what}: position {k}");
                            match (&emitted.table, table) {
                                (None, None) => {}
                                (Some(a), Some(b)) => assert_eq!(
                                    shape_and_words(a),
                                    shape_and_words(b),
                                    "{what}: table {k}"
                                ),
                                (a, b) => panic!(
                                    "{what}: table {k} streamed {} but built {}",
                                    a.is_some(),
                                    b.is_some()
                                ),
                            }
                        }
                    }
                }
            }
        }
    }
}

/// The streamed tables against the run's AIRs built from the header: one a
/// position, each table that AIR's width.
#[test]
fn the_streamed_tables_are_the_airs_in_order() {
    let opts = stark::proof::options::ProofOptions::default_test_options();
    for name in [
        "all_instructions_64",
        "test_keccak_multi",
        "test_ecsm_multi",
        "test_blake3",
    ] {
        let (program, logs) = run(name);
        let configure: Configure = Box::new(|b| {
            b.keccak_rnd_chunks_at_finish(32)
                .expect("rows")
                .cuts_at_finish(2, 4, 2, 2)
                .expect("rows")
                .pack_finished_tables()
                .generate_packed()
                .drop_streamed_ops()
                .expect("before any window")
        });
        let (builder, last) = pushed(
            &program,
            &logs,
            &MaxRowsConfig::small(),
            7,
            configure.as_ref(),
        );
        let (header, got, _) = streamed(builder, last, 2 << 30);
        let airs = crate::VmAirs::new(
            &program,
            &opts,
            false,
            &header.page_configs,
            &header.table_counts,
            &header.cpu_heights,
            None,
            true,
            None,
            crate::test_utils::halted_register_commitment(
                &opts,
                &program,
                &header.public_output_bytes,
            )
            .preprocessed(),
        );
        let refs = airs.air_refs();
        assert_eq!(refs.len(), got.len(), "{name}: one table an AIR");
        for emitted in &got {
            if let Some(table) = &emitted.table {
                assert_eq!(
                    table.main_table.width,
                    refs[emitted.position].trace_layout().0,
                    "{name}: {} at {}",
                    refs[emitted.position].name(),
                    emitted.position
                );
            }
        }
    }
}

/// A one-byte budget, every table larger: the lowest table not yet placed
/// always goes, so the finish completes (a named timeout, not a hang, if it
/// did not).
#[test]
fn a_one_byte_budget_streams_the_finish() {
    let done = within(120, || {
        let (program, logs) = run("test_ecsm_multi");
        let configure: Configure = Box::new(|b| {
            b.cuts_at_finish(2, 4, 2, 2)
                .expect("rows")
                .pack_finished_tables()
                .drop_streamed_ops()
                .expect("before any window")
        });
        let (builder, last) = pushed(
            &program,
            &logs,
            &MaxRowsConfig::small(),
            7,
            configure.as_ref(),
        );
        let (_, got, most) = streamed(builder, last, 1);
        (got.len(), most)
    });
    let (tables, most) = done.expect("a one-byte budget must not deadlock the finish");
    assert!(tables > 0);
    assert!(most > 1, "every table went alone, over the budget");
}

/// Synthetic units: `n` tables of one family, each `bytes` estimated, the job
/// `fail` errs; the rest build a one-row table.
fn units<'a>(n: usize, fail: Option<usize>) -> Vec<Unit<'a>> {
    (0..n)
        .map(|k| Unit::Build {
            family: 0,
            rows: 1 << 20,
            job: Box::new(move || {
                if fail == Some(k) {
                    return Err(Error::Prover(format!("job {k} failed")));
                }
                let width = FAMILIES[0].1;
                Ok(TraceTable::new_main(
                    crate::tables::types::zeroed_fe_vec(width),
                    width,
                    1,
                ))
            }) as Job<'a>,
        })
        .collect()
}

/// A job's error ends the stream with that error and reaches a waiter on the
/// gate, within a timeout.
#[test]
fn a_jobs_error_reaches_every_waiter() {
    for slide in [false, true] {
        let done = within(30, move || {
            let gate = Arc::new(ByteGate::new(1));
            let waiter = {
                let gate = Arc::clone(&gate);
                std::thread::spawn(move || gate.acquire(1000, 1 << 40))
            };
            let mut placed = 0usize;
            let result = emit_streamed_units(
                Ok(units(4, Some(2))),
                &gate,
                &mut |emitted: Emitted| {
                    if emitted.table.is_some() {
                        gate.release(placed);
                        placed += 1;
                    }
                    Ok(())
                },
                slide,
            );
            (result, waiter.join().expect("the waiter"))
        });
        let (result, waited) = done.expect("an error must not hang the stream");
        let err = result.expect_err("the stream fails");
        assert!(
            format!("{err:?}").contains("job 2 failed"),
            "slide {slide}: {err:?}"
        );
        assert!(waited.is_err(), "slide {slide}: the waiter is stopped too");
    }
}

/// A packer that stops (its gate stopped, nothing placed) stops the emitter
/// waiting for its next table's bytes; a sink that refuses stops it too.
#[test]
fn a_stopped_packer_stops_the_emitter() {
    for slide in [false, true] {
        let done = within(30, move || {
            let gate = Arc::new(ByteGate::new(1));
            let stopper = {
                let gate = Arc::clone(&gate);
                std::thread::spawn(move || {
                    std::thread::sleep(Duration::from_millis(100));
                    gate.stop("the packer stopped");
                })
            };
            let result = emit_streamed_units(Ok(units(3, None)), &gate, &mut |_| Ok(()), slide);
            stopper.join().expect("the stopper");
            result
        });
        let err = done
            .expect("a stopped packer must not hang the emitter")
            .expect_err("stopped");
        assert!(
            format!("{err:?}").contains("the packer stopped"),
            "slide {slide}: {err:?}"
        );
        let gate = ByteGate::new(usize::MAX);
        let err = emit_streamed_units(
            Ok(units(3, None)),
            &gate,
            &mut |_| Err(Error::Prover("the layout thread stopped".into())),
            slide,
        )
        .expect_err("the sink refused");
        assert!(
            format!("{err:?}").contains("the layout thread stopped"),
            "slide {slide}: {err:?}"
        );
    }
}

/// Built as the gate admits each table: every configuration, chunked small,
/// windows of 7 and 1000 cycles, budgets of one byte, 2 GiB and none, hands on
/// the monolithic finish's tables word for word in AIR order, as the waves do.
#[test]
fn the_sliding_stream_is_the_monolithic_finish_in_air_order() {
    let max_rows = MaxRowsConfig::small();
    for name in [
        "misalign_sd",
        "test_keccak_multi",
        "test_ecsm_multi",
        "test_blake3_absorb",
    ] {
        let (program, logs) = run(name);
        for window in [7, 1000] {
            for (config, configure) in &configs() {
                let (builder, last) =
                    pushed(&program, &logs, &max_rows, window, configure.as_ref());
                let want = air_order(&builder.finish(last).expect("the monolithic finish"))
                    .into_iter()
                    .map(|t| t.map(shape_and_words))
                    .collect::<Vec<_>>();
                for budget in [1usize, 2 << 30, usize::MAX] {
                    let what = format!("{name} window {window} {config} budget {budget}");
                    let (builder, last) = pushed(&program, &logs, &max_rows, window, &|b| {
                        configure(b).slide_rest()
                    });
                    let (_, got, _) = streamed(builder, last, budget);
                    let got: Vec<_> = got
                        .iter()
                        .enumerate()
                        .map(|(k, e)| {
                            assert_eq!(e.position, k, "{what}: position {k}");
                            e.table.as_ref().map(shape_and_words)
                        })
                        .collect();
                    assert!(got == want, "{what}: the tables in AIR order");
                }
            }
        }
    }
}

/// Synthetic units that sleep: unit `slow` takes `long` ms, the others `short`;
/// each notes when it starts building.
#[cfg(feature = "parallel")]
fn sleeping_units<'a>(
    n: usize,
    slow: usize,
    long: u64,
    short: u64,
    started: &'a std::sync::Mutex<Vec<(usize, std::time::Instant)>>,
) -> Vec<Unit<'a>> {
    (0..n)
        .map(|k| Unit::Build {
            family: 0,
            rows: 1,
            job: Box::new(move || {
                started
                    .lock()
                    .expect("the start log")
                    .push((k, std::time::Instant::now()));
                std::thread::sleep(Duration::from_millis(if k == slow { long } else { short }));
                let width = FAMILIES[0].1;
                Ok(TraceTable::new_main(
                    crate::tables::types::zeroed_fe_vec(width),
                    width,
                    1,
                ))
            }) as Job<'a>,
        })
        .collect()
}

/// ★ Built as the gate admits each table, a slow first table does not hold
/// back the next tables' builds (they all start before it ends), only their
/// hand-on, which stays in order; built in waves of half the gate's budget,
/// the tables after the first wave wait for its slowest table. The budget is
/// the `n` tables' estimated bytes and the packer places each table as it
/// comes.
#[test]
#[cfg(feature = "parallel")]
fn a_slow_table_holds_back_only_the_hand_on() {
    let n = rayon::current_num_threads().min(6);
    assert!(n >= 2, "a pool of at least two threads");
    let one = FAMILIES[0].1 * 8;
    for slide in [true, false] {
        let started = std::sync::Mutex::new(Vec::new());
        let gate = ByteGate::new(n * one);
        let t0 = std::time::Instant::now();
        let (mut order, mut first_sent) = (Vec::new(), None);
        emit_streamed_units(
            Ok(sleeping_units(n, 0, 600, 20, &started)),
            &gate,
            &mut |e: Emitted| {
                first_sent.get_or_insert(t0.elapsed());
                gate.release(order.len());
                order.push(e.position);
                Ok(())
            },
            slide,
        )
        .expect("the stream");
        assert_eq!(
            order,
            (0..n).collect::<Vec<_>>(),
            "slide {slide}: in AIR order"
        );
        let started = started.into_inner().expect("the start log");
        assert_eq!(started.len(), n, "slide {slide}");
        let latest = started
            .iter()
            .map(|(_, at)| at.duration_since(t0))
            .max()
            .expect("starts");
        let first_sent = first_sent.expect("a table was handed on");
        assert!(
            first_sent >= Duration::from_millis(600),
            "slide {slide}: the slow table was handed on at {first_sent:?}"
        );
        if slide {
            assert!(
                latest < Duration::from_millis(400),
                "the last table started at {latest:?}, behind the slow one"
            );
        } else {
            assert!(
                latest >= Duration::from_millis(600),
                "in waves the last table started at {latest:?}, before the slow one ended"
            );
        }
    }
}

/// A job that panics ends the sliding stream with an error naming its table,
/// within a timeout, instead of hanging it.
#[test]
#[cfg(feature = "parallel")]
fn a_panicking_job_ends_the_sliding_stream() {
    let done = within(30, || {
        let gate = ByteGate::new(usize::MAX);
        let mut units = units(3, None);
        units[1] = Unit::Build {
            family: 0,
            rows: 1,
            job: Box::new(|| panic!("a generator bug")),
        };
        emit_streamed_units(Ok(units), &gate, &mut |_| Ok(()), true)
    });
    let err = done
        .expect("a panicking job must not hang the stream")
        .expect_err("the stream fails");
    assert!(format!("{err:?}").contains("table 1 panicked"), "{err:?}");
}

/// A header that miscounts any family by one is refused before a table is
/// handed on.
#[test]
fn a_miscounted_header_is_refused_by_the_stream() {
    let (program, logs) = run("test_keccak_multi");
    let max_rows = MaxRowsConfig::small();
    let (builder, last) = pushed(&program, &logs, &max_rows, 1000, &|b| b);
    let (header, _, _) = streamed(builder, last, usize::MAX);
    let counts = header.table_counts;
    let wrong: Vec<crate::TableCounts> = (0..22)
        .map(|f| {
            let mut c = counts.clone();
            let field = [
                &mut c.cpu,
                &mut c.lt,
                &mut c.memw,
                &mut c.memw_aligned,
                &mut c.load,
                &mut c.mul,
                &mut c.dvrm,
                &mut c.shift,
                &mut c.branch,
                &mut c.memw_register,
                &mut c.eq,
                &mut c.bytewise,
                &mut c.store,
                &mut c.cpu32,
                &mut c.keccak,
                &mut c.keccak_rnd,
                &mut c.ecsm,
                &mut c.ecdas,
                &mut c.hint,
                &mut c.copy,
                &mut c.blake3,
                &mut c.is_wh,
            ]
            .into_iter()
            .nth(f)
            .expect("22 fields");
            *field += 1;
            c
        })
        .collect();
    let artifacts = DecodeArtifacts::from_elf(&program).expect("the artifacts");
    let image = build_initial_image(&program, &[]);
    let register_init = register::register_init_from_entry_point(program.entry_point);
    for (f, counts) in wrong.into_iter().enumerate() {
        let CollectedEpoch {
            ops,
            memory_state,
            register_state,
        } = Traces::collect_epoch(&artifacts, &image, &register_init, &logs, true)
            .expect("collected");
        let plan = super::FinishPlan::new(
            ops,
            Some(&image),
            &memory_state,
            &register_init,
            artifacts.decode_trace.clone(),
            &artifacts.decode,
            register_state,
            &max_rows,
            #[cfg(feature = "disk-spill")]
            StorageMode::Ram,
            &[],
            true,
            false,
            &StreamSkip::default(),
            None,
        )
        .expect("the plan")
        .with_counts(counts);
        let gate = ByteGate::new(usize::MAX);
        let mut handed = 0usize;
        let result = plan.emit_streamed(&gate, &mut |_| {
            handed += 1;
            Ok(())
        });
        assert!(
            result.is_err(),
            "family {f} miscounted by one was not refused"
        );
        assert_eq!(
            handed, 0,
            "family {f}: a table was handed on before the refusal"
        );
    }
}

/// A table's packed columns: widths and bytes.
fn packed_bytes(t: &Table) -> Option<(Vec<u8>, Vec<u8>)> {
    t.narrow_main()
        .map(|p| (p.widths().to_vec(), p.data().to_vec()))
}

/// Where two tables' packed columns first differ, for a short failure.
fn first_difference(
    a: &Option<(Vec<u8>, Vec<u8>)>,
    b: &Option<(Vec<u8>, Vec<u8>)>,
) -> Option<String> {
    match (a, b) {
        (Some((wa, da)), Some((wb, db))) => {
            if let Some(c) = (0..wa.len().max(wb.len())).find(|&c| wa.get(c) != wb.get(c)) {
                return Some(format!("column {c}'s width"));
            }
            (0..da.len().max(db.len()))
                .find(|&i| da.get(i) != db.get(i))
                .map(|i| format!("byte {i} of {}", da.len()))
        }
        (None, None) => None,
        _ => Some("one is packed, the other not".into()),
    }
}

/// ★ With the card filling KECCAK_RND ([`WindowedTraceBuilder::card_fill_keccak_rnd`]),
/// the streamed finish hands on each KECCAK_RND table as the keccak operations
/// its rows come from, and those records' host fill is the table the finish
/// builds without it, packed byte for byte: every cut (into an op at either
/// end, and padding rows alone) under both padding rules, windows of 7 and
/// 1000 cycles. Every other table is the same, the records are 208 bytes an
/// op of the table, and phase B is left nothing of KECCAK_RND to build again
/// (LT's jobs are the same).
#[test]
fn held_as_records_the_finishs_keccak_rnd_tables_are_its_tables() {
    use super::RegenFamily;
    let mut held = 0usize;
    for name in [
        "test_keccak_multi",
        "all_instructions_64",
        "test_ecsm_multi",
    ] {
        let (program, logs) = run(name);
        for padding in [CutPadding::LastTable, CutPadding::WholeRun] {
            for window in [7, 1000] {
                let what = format!("{name} {padding:?} window {window}");
                let max_rows = MaxRowsConfig::small();
                let host = move |b| cut(b, padding);
                let card = move |b| cut(b, padding).card_fill_keccak_rnd();
                let (builder, last) = pushed(&program, &logs, &max_rows, window, &host);
                let whole = builder.finish(last).expect("the monolithic finish");
                let [kr, _] = regen_ranges(&whole);
                let (builder, last) = pushed(&program, &logs, &max_rows, window, &host);
                let (want, want_regen) = streamed_keeping(builder, last, true);
                let (builder, last) = pushed(&program, &logs, &max_rows, window, &card);
                let (got, got_regen) = streamed_keeping(builder, last, true);
                assert_eq!(got.len(), want.len(), "{what}: the tables");
                for (k, (g, w)) in got.iter().zip(&want).enumerate() {
                    assert_eq!(g.position, w.position, "{what}: position {k}");
                    let (Some(g), Some(w)) = (&g.table, &w.table) else {
                        assert!(g.table.is_none() && w.table.is_none(), "{what}: slot {k}");
                        continue;
                    };
                    match g.records_main() {
                        Some(records) => {
                            assert!(kr.contains(&k), "{what}: records at {k}, not KECCAK_RND's");
                            let filled = records.host_fill().expect("the host fills it");
                            let filled = Some((filled.widths().to_vec(), filled.data().to_vec()));
                            assert_eq!(
                                first_difference(&filled, &packed_bytes(w)),
                                None,
                                "{what}: KECCAK_RND at {k}, filled from its records"
                            );
                            assert_eq!(
                                (g.main_table.width, g.main_table.height),
                                (w.main_table.width, w.main_table.height),
                                "{what}: KECCAK_RND at {k}'s shape"
                            );
                            let ops = match records.fill() {
                                multilinear::records::RecordsFill::KeccakRnd { ops, skip } => {
                                    assert!(
                                        skip < 24 || ops.is_empty(),
                                        "{what}: cut within its first op, if any"
                                    );
                                    ops.len()
                                }
                            };
                            assert_eq!(records.bytes(), 208 * ops, "{what}: 208 B an op");
                            held += 1;
                        }
                        None => {
                            assert!(!kr.contains(&k), "{what}: KECCAK_RND at {k} built");
                            assert_eq!(
                                first_difference(&packed_bytes(g), &packed_bytes(w)),
                                None,
                                "{what}: table {k}"
                            );
                            assert!(
                                shape_and_words(g) == shape_and_words(w),
                                "{what}: table {k}'s words"
                            );
                        }
                    }
                }
                let mut got_regen = got_regen.expect("the lists kept");
                let mut want_regen = want_regen.expect("the lists kept");
                assert!(
                    got_regen
                        .take(RegenFamily::KeccakRnd)
                        .expect("no jobs")
                        .is_empty(),
                    "{what}: KECCAK_RND left to phase B"
                );
                let lt_jobs = |r: &mut super::RestRegen| -> Vec<_> {
                    r.take(RegenFamily::Lt)
                        .expect("LT's jobs")
                        .into_iter()
                        .map(|job| job.map(|job| packed_bytes(&job().expect("LT again"))))
                        .collect()
                };
                assert!(
                    lt_jobs(&mut got_regen) == lt_jobs(&mut want_regen),
                    "{what}: LT's jobs"
                );
            }
        }
    }
    assert!(held > 0, "some KECCAK_RND table was held as records");
}
