//! ★ The streamed finish ([`super::FinishPlan::emit_streamed`]) hands on the
//! tables the monolithic finish ([`super::FinishPlan::emit_all`]) builds, word
//! for word, in the run's AIR order with every streamed chunk's slot in its
//! place, at any byte budget (one byte, 2 GiB, none). The byte gate's
//! lowest-pending rule keeps a one-byte budget from deadlocking; a job's error
//! and a stopped packer each stop the emitter and every waiter; a header that
//! miscounts is refused.

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
        &t.commits,
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
                    .cuts_at_finish(2, 4, 2)
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
    ]
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
                .cuts_at_finish(2, 4, 2)
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
            None,
            true,
            None,
            None,
            None,
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
            b.cuts_at_finish(2, 4, 2)
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
    let done = within(30, || {
        let gate = Arc::new(ByteGate::new(1));
        let waiter = {
            let gate = Arc::clone(&gate);
            std::thread::spawn(move || gate.acquire(1000, 1 << 40))
        };
        let mut placed = 0usize;
        let result = emit_streamed_units(Ok(units(4, Some(2))), &gate, &mut |emitted: Emitted| {
            if emitted.table.is_some() {
                gate.release(placed);
                placed += 1;
            }
            Ok(())
        });
        (result, waiter.join().expect("the waiter"))
    });
    let (result, waited) = done.expect("an error must not hang the stream");
    let err = result.expect_err("the stream fails");
    assert!(format!("{err:?}").contains("job 2 failed"), "{err:?}");
    assert!(waited.is_err(), "the waiter is stopped too");
}

/// A packer that stops (its gate stopped, nothing placed) stops the emitter
/// waiting for its next table's bytes; a sink that refuses stops it too.
#[test]
fn a_stopped_packer_stops_the_emitter() {
    let done = within(30, || {
        let gate = Arc::new(ByteGate::new(1));
        let stopper = {
            let gate = Arc::clone(&gate);
            std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(100));
                gate.stop("the packer stopped");
            })
        };
        let result = emit_streamed_units(Ok(units(3, None)), &gate, &mut |_| Ok(()));
        stopper.join().expect("the stopper");
        result
    });
    let err = done
        .expect("a stopped packer must not hang the emitter")
        .expect_err("stopped");
    assert!(format!("{err:?}").contains("the packer stopped"), "{err:?}");
    let gate = ByteGate::new(usize::MAX);
    let err = emit_streamed_units(Ok(units(3, None)), &gate, &mut |_| {
        Err(Error::Prover("the layout thread stopped".into()))
    })
    .expect_err("the sink refused");
    assert!(
        format!("{err:?}").contains("the layout thread stopped"),
        "{err:?}"
    );
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
    let wrong: Vec<crate::TableCounts> = (0..21)
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
                &mut c.commit,
                &mut c.blake3,
            ]
            .into_iter()
            .nth(f)
            .expect("21 fields");
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
            &artifacts.decode_pc_to_row,
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
