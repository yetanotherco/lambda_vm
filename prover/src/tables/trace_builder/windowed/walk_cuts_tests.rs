//! The walk's cuts (`LAMBDA_VM_WALK_CUTS`, [`WalkCuts`]) against the walk
//! without them: window by window, each part alone and every part, the lists
//! a walker emits, its routing, its last ECALL and over the run its counts are
//! the same, at the light walk's level 2 (the walkers' default: routed where
//! walked) and at level 0; a run built with them on one walker or three builds
//! the same tables; and a window that is not the run's last is refused for a
//! halting cycle with or without the pre-scan.

use std::sync::Arc;

use executor::elf::Elf;
use executor::vm::fast::FastExecutor;
use executor::vm::history::WindowHistory;
use executor::vm::logs::Log;

use super::super::{HistoryOverlay, MemoryState, RegisterState, WalkCuts, WalkSizes};
use super::light_walk_tests::{
    PROGRAMS, RUST_GUESTS, WORD_PROGRAMS, built, built_with_memory, counted, dbg, lists, rust_elf,
};
use super::{LogSource, WalkCounts, WalkedWindow, WindowedTraceBuilder, walk_window};
use crate::Error;
use crate::tables::MaxRowsConfig;
use crate::tables::decode::DecodeTable;
use crate::tables::fused_memory::MemoryFormat;
use crate::tables::memw_aligned::MemwAlignedOrder;
use crate::tables::register::RegisterFormat;
use crate::tables::trace_builder::walk_cuts::each_part_and_all;
use crate::tests::windowed_builder_tests::{run, same_traces};

/// How a window is walked: its formats and batch.
#[derive(Clone, Copy)]
struct Shape {
    order: MemwAlignedOrder,
    format: RegisterFormat,
    memory: MemoryFormat,
    batch: usize,
}

/// A record-mode walk of one window as the walkers walk it, at `light` and
/// `route`, with `cuts`; its CPU list kept. The window and whether its cells
/// were refused, or the walk's error.
#[allow(clippy::too_many_arguments)]
fn walked(
    decode: &DecodeTable,
    shape: Shape,
    (light, route): (bool, bool),
    cuts: WalkCuts,
    counts: &mut WalkCounts,
    sizes: &mut WalkSizes,
    first: usize,
    logs: &[Log],
    history: &Arc<WindowHistory>,
) -> Result<(WalkedWindow, bool), Error> {
    let cells = history.registers.expect("the window's register cells");
    let mut record = MemoryState::new();
    record.overlay = Some(Box::new(
        HistoryOverlay::new(Arc::clone(history)).light(light),
    ));
    let mut registers = RegisterState {
        regs: cells.regs,
        index_register: cells.index,
        pc_register: cells.pc,
    };
    let walked = walk_window(
        decode,
        shape.order,
        shape.format,
        shape.memory,
        false,
        shape.batch,
        false.into(),
        cuts,
        Some(counts),
        true,
        route,
        sizes,
        &mut record,
        &mut registers,
        first,
        &mut LogSource::Logs(logs),
        None,
    )?;
    let refused = record.take_missing();
    Ok((walked, refused))
}

/// The run's windows from the fast engine's history (the word history under
/// the word format), each with its first cycle, the run's last left out.
fn windows(
    program: &Elf,
    input: &[u8],
    window: usize,
    words: bool,
) -> Vec<(usize, Vec<Log>, Arc<WindowHistory>)> {
    let mut fast = FastExecutor::new(program, input.to_vec()).expect("the fast engine");
    if words {
        fast.enable_word_history().expect("the word history");
    } else {
        fast.enable_history().expect("history mode");
    }
    let mut out = Vec::new();
    let mut first = 0usize;
    loop {
        let next = if words {
            fast.resume_word_history(window)
        } else {
            fast.resume_history(window)
        };
        let Some((logs, history)) = next.expect("the run") else {
            break;
        };
        if logs.last().is_some_and(|l| l.next_pc == 0) {
            break;
        }
        let len = logs.len();
        out.push((first, logs, Arc::new(history)));
        first += len;
    }
    out
}

/// Walks every window of `program`'s run but the last at the light walk's
/// levels 2 and 0, without cuts and with each arm of [`each_part_and_all`]:
/// each arm must emit the lists, the routing and the last ECALL the walk
/// without cuts emits, refuse nothing, and over the run count the same
/// lookups and DECODE rows. Returns the windows compared.
fn cuts_are_todays(name: &str, program: &Elf, input: &[u8], window: usize, shape: Shape) -> usize {
    let max_rows = MaxRowsConfig::small();
    let builder = WindowedTraceBuilder::new(program, input, &max_rows).expect("the builder");
    let decode = Arc::clone(&builder.decode);
    drop(builder);
    let mut arms = each_part_and_all();
    let default = WalkCuts::parse(None).unwrap();
    arms.extend([
        ("defaults", default),
        (
            "defaults,rows-sized",
            WalkCuts {
                rows_sized: true,
                ..default
            },
        ),
        (
            "defaults,count-direct",
            WalkCuts {
                count_direct: true,
                ..default
            },
        ),
    ]);
    let levels = [(true, true), (false, false)];
    // By level: the walk without cuts, then each arm.
    let mut counts: Vec<Vec<WalkCounts>> = levels
        .iter()
        .map(|_| (0..=arms.len()).map(|_| WalkCounts::default()).collect())
        .collect();
    let mut sizes: Vec<Vec<WalkSizes>> = levels
        .iter()
        .map(|_| (0..=arms.len()).map(|_| WalkSizes::default()).collect())
        .collect();
    let windows = windows(program, input, window, shape.memory == MemoryFormat::Words);
    for (k, (first, logs, history)) in windows.iter().enumerate() {
        for (l, &level) in levels.iter().enumerate() {
            let what = format!(
                "{name}/{window}/{}/{:?}/{:?}/{:?} window {k} level {level:?}",
                shape.batch, shape.order, shape.format, shape.memory
            );
            let (today, refused) = walked(
                &decode,
                shape,
                level,
                WalkCuts::default(),
                &mut counts[l][0],
                &mut sizes[l][0],
                *first,
                logs,
                history,
            )
            .expect("the walk without cuts");
            assert!(!refused, "{what}: refused without cuts");
            let want = lists(&today);
            for (a, &(arm, cuts)) in arms.iter().enumerate() {
                let what = format!("{what} cuts {arm}");
                let (got, refused) = walked(
                    &decode,
                    shape,
                    level,
                    cuts,
                    &mut counts[l][a + 1],
                    &mut sizes[l][a + 1],
                    *first,
                    logs,
                    history,
                )
                .unwrap_or_else(|e| panic!("{what}: {e:?}"));
                assert!(!refused, "{what}: refused the window");
                for ((list, got), (_, wanted)) in lists(&got).iter().zip(&want) {
                    assert!(got == wanted, "{what}: {list} differ");
                }
                assert!(
                    dbg(&got.walk.ecrec_ops) == dbg(&today.walk.ecrec_ops),
                    "{what}: ECREC differ"
                );
                assert_eq!(
                    got.made.as_ref().map(|m| m.last_ecall),
                    today.made.as_ref().map(|m| m.last_ecall),
                    "{what}: the last ECALL"
                );
                assert!(
                    dbg(&got.routed) == dbg(&today.routed),
                    "{what}: the routing"
                );
                assert_eq!(got.cycles, today.cycles, "{what}: the cycles");
            }
        }
    }
    for (l, level) in levels.iter().enumerate() {
        // Each arm's counts settled as its walker settles them at its end.
        for counts in counts[l].iter_mut() {
            counts.settle(&decode).expect("the counts settle");
        }
        let (want, want_rows) = counted(&counts[l][0]);
        for (a, (arm, _)) in arms.iter().enumerate() {
            let what = format!("{name}/{window}/{} level {level:?} cuts {arm}", shape.batch);
            let (histogram, rows) = counted(&counts[l][a + 1]);
            assert!(histogram == want, "{what}: the lookups");
            assert_eq!(rows, want_rows, "{what}: DECODE's rows");
        }
    }
    windows.len()
}

fn split_shape(order: MemwAlignedOrder, format: RegisterFormat, batch: usize) -> Shape {
    Shape {
        order,
        format,
        memory: MemoryFormat::Split,
        batch,
    }
}

/// ★ Every asm program and Rust guest, and an ethrex fixture, at windows from
/// a few cycles to many, batches that cross them, both register formats and
/// the word memory format: each cut alone and every cut emit the walk's
/// lists, routing and counts.
mod the_walk_cuts_emit_today_s_lists {
    use super::*;

    fn asm_split_memory(cases: &[(usize, Shape)]) -> usize {
        let _ecrec = crate::tables::trace_builder::ecrec_forge_free();
        let mut windows = 0;
        for name in PROGRAMS {
            let (program, _) = run(name);
            for &(window, shape) in cases {
                windows += cuts_are_todays(name, &program, &[], window, shape);
            }
        }
        windows
    }

    #[test]
    fn asm_split_memory_short_windows() {
        let windows = asm_split_memory(&[
            (
                7,
                split_shape(MemwAlignedOrder::Limbs, RegisterFormat::InRow, 3),
            ),
            (
                33,
                split_shape(MemwAlignedOrder::Lt, RegisterFormat::InRow, 2048),
            ),
            (
                33,
                split_shape(MemwAlignedOrder::Limbs, RegisterFormat::Tokens32, 7),
            ),
        ]);
        assert!(windows > 1000, "{windows} windows compared");
    }

    #[test]
    fn asm_split_memory_long_windows() {
        asm_split_memory(&[
            (
                4096,
                split_shape(MemwAlignedOrder::Limbs, RegisterFormat::InRow, 5),
            ),
            (
                4096,
                split_shape(MemwAlignedOrder::Limbs, RegisterFormat::InRow, 2048),
            ),
        ]);
    }

    #[test]
    fn rust_guests() {
        let mut windows = 0;
        for name in RUST_GUESTS {
            let program = rust_elf(name);
            windows += cuts_are_todays(
                name,
                &program,
                &[],
                4096,
                split_shape(MemwAlignedOrder::Limbs, RegisterFormat::InRow, 7),
            );
        }
        assert!(windows > 0, "compared Rust guest windows");
    }

    #[test]
    fn asm_word_memory() {
        let _ecrec = crate::tables::trace_builder::ecrec_forge_free();
        let mut windows = 0;
        for name in WORD_PROGRAMS.into_iter().chain(["test_ecrec"]) {
            let (program, _) = run(name);
            for (window, batch) in [(7, 3), (4096, 5)] {
                windows += cuts_are_todays(
                    name,
                    &program,
                    &[],
                    window,
                    Shape {
                        order: MemwAlignedOrder::Limbs,
                        format: RegisterFormat::InRow,
                        memory: MemoryFormat::Words,
                        batch,
                    },
                );
            }
        }
        assert!(windows > 0, "compared word-memory windows");
    }

    #[test]
    fn ethrex_transfers() {
        let ethrex = rust_elf("ethrex");
        let input = ethrex_transfers_input();
        let windows = cuts_are_todays(
            "ethrex_10_transfers",
            &ethrex,
            &input,
            1 << 16,
            Shape {
                order: MemwAlignedOrder::Limbs,
                format: RegisterFormat::InRow,
                memory: MemoryFormat::Words,
                batch: 2048,
            },
        );
        assert!(windows > 0, "compared ethrex windows");
    }
}

fn ethrex_transfers_input() -> Vec<u8> {
    let fixtures = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../executor/tests");
    std::fs::read(fixtures.join("ethrex_10_transfers.bin")).expect("the fixture")
}

#[test]
fn assert_walk_cuts_build_the_same_tables_for_word_memory_and_ethrex_transfers() {
    let (assembly, _) = run("fused_mem_cases");
    let ethrex = rust_elf("ethrex");
    let transfers = ethrex_transfers_input();
    let default = WalkCuts::parse(None).unwrap();
    let cuts = [
        WalkCuts {
            rows_sized: true,
            ..default
        },
        WalkCuts {
            count_direct: true,
            ..default
        },
        WalkCuts {
            rows_sized: true,
            count_direct: true,
            ..default
        },
    ];
    for (name, program, input, window, max_rows) in [
        (
            "word memory",
            &assembly,
            &[][..],
            7,
            MaxRowsConfig::uniform(4),
        ),
        (
            "ethrex transfers",
            &ethrex,
            transfers.as_slice(),
            1 << 16,
            MaxRowsConfig::uniform(1 << 16),
        ),
    ] {
        let build = |walkers, cuts| {
            built_with_memory(
                program,
                input,
                &max_rows,
                window,
                walkers,
                (2, cuts),
                (RegisterFormat::InRow, MemoryFormat::Words),
            )
        };
        let (want, want_order) = build(1, default);
        assert!(!want_order.is_empty(), "{name}: streamed chunks");
        for walkers in [1, 3, 6] {
            for cuts in std::iter::once(default).chain(cuts) {
                let (got, order) = build(walkers, cuts);
                assert_eq!(
                    order, want_order,
                    "{name}, {walkers} walkers, {cuts:?}: the hand-out"
                );
                same_traces(&want, &got);
            }
        }
    }
}

/// ★ A window that is not the run's last is refused for a halting cycle
/// (`next_pc == 0`) anywhere in it, first, last, at a batch's edge or inside
/// a later batch, with the pre-scan and without it (the collect refuses it
/// as it reads the log), with the same error; the window as executed walks
/// either way.
#[test]
fn a_halting_cycle_before_the_last_window_is_refused_with_or_without_the_prescan() {
    let (program, _) = run("all_instructions_64");
    let max_rows = MaxRowsConfig::small();
    let builder = WindowedTraceBuilder::new(&program, &[], &max_rows).expect("the builder");
    let decode = Arc::clone(&builder.decode);
    drop(builder);
    let shape = Shape {
        order: MemwAlignedOrder::Limbs,
        format: RegisterFormat::InRow,
        memory: MemoryFormat::Split,
        batch: 3,
    };
    let (first, logs, history) = windows(&program, &[], 7, false)
        .into_iter()
        .nth(1)
        .expect("a second window before the last");
    assert_eq!(logs.len(), 7, "a whole window");
    let prescan = WalkCuts::parse(Some("prescan")).expect("the part");
    let walk = |logs: &[Log], cuts: WalkCuts, level| {
        walked(
            &decode,
            shape,
            level,
            cuts,
            &mut WalkCounts::default(),
            &mut WalkSizes::default(),
            first,
            logs,
            &history,
        )
    };
    for level in [(true, true), (false, false)] {
        for cuts in [WalkCuts::default(), prescan, WalkCuts::all()] {
            let (_, refused) = walk(&logs, cuts, level).expect("the window as executed");
            assert!(!refused, "{cuts:?} {level:?}: the window as executed");
            for at in [0, 2, 3, 5, 6] {
                let mut halting = logs.clone();
                halting[at].next_pc = 0;
                let got = walk(&halting, cuts, level);
                assert!(
                    matches!(got, Err(Error::HaltInNonFinalEpoch)),
                    "{cuts:?} {level:?}: a halt at cycle {at} walked: {:?}",
                    got.err()
                );
            }
        }
    }
}

/// ★ A run built with each cut alone on three walkers and with every cut on
/// one and on three, under both register formats, hands its chunks out in the order
/// and builds the tables, row for row, the run without cuts does.
fn assert_walk_cuts_build_the_same_tables(
    format: RegisterFormat,
    window: usize,
    programs: &[&str],
) {
    // Runs `test_ecrec`: the ECREC forgery switch stays off meanwhile.
    let _ecrec = crate::tables::trace_builder::ecrec_forge_free();
    let max_rows = MaxRowsConfig::small();
    let none = WalkCuts::default();
    let mut chunks = 0usize;
    for &name in programs {
        let (program, _) = run(name);
        let (want, want_order) = built(&program, &max_rows, window, 2, (2, none), format);
        chunks += want_order.len();
        for (arm, cuts) in each_part_and_all() {
            // Each part on three walkers; every part on one and three.
            let walkers: &[usize] = if arm == "all" { &[1, 3] } else { &[3] };
            for &walkers in walkers {
                let what = format!("{name}/{format:?}/{window}/{arm}/{walkers}");
                let (got, order) = built(&program, &max_rows, window, walkers, (2, cuts), format);
                assert_eq!(order, want_order, "{what}: the hand-out");
                same_traces(&want, &got);
            }
        }
    }
    assert!(chunks > 0, "no chunk handed out");
}

macro_rules! walk_table_cases {
    ($name:ident, $format:ident, $window:literal) => {
        mod $name {
            use super::*;

            #[test]
            fn keccak_guests() {
                assert_walk_cuts_build_the_same_tables(
                    RegisterFormat::$format,
                    $window,
                    &PROGRAMS[..7],
                );
            }

            #[test]
            fn ecrec_guests() {
                assert_walk_cuts_build_the_same_tables(
                    RegisterFormat::$format,
                    $window,
                    &PROGRAMS[7..],
                );
            }
        }
    };
}

walk_table_cases!(walk_cuts_build_the_same_in_row_tables_at_window_7, InRow, 7);
walk_table_cases!(
    walk_cuts_build_the_same_in_row_tables_at_window_4096,
    InRow,
    4096
);
walk_table_cases!(
    walk_cuts_build_the_same_token_tables_at_window_7,
    Tokens32,
    7
);
walk_table_cases!(
    walk_cuts_build_the_same_token_tables_at_window_4096,
    Tokens32,
    4096
);

/// Under `const` a walker's counts are settled only if every CPU op it
/// counted was routed without its instruction lookups: a walker that routed
/// more or fewer than it counted, or counted a DECODE row the table lacks, is
/// refused rather than settled; settled, the counts are each op's seven.
#[test]
fn counts_settled_under_const_must_route_every_op_they_count() {
    let (program, _) = run("all_instructions_64");
    let max_rows = MaxRowsConfig::small();
    let builder = WindowedTraceBuilder::new(&program, &[], &max_rows).expect("the builder");
    let decode = Arc::clone(&builder.decode);
    drop(builder);
    let rows = decode.padding_row();
    assert!(rows >= 3, "{rows} DECODE rows");
    let counts = |decode_rows: Vec<u64>, const_cycles: u64| WalkCounts {
        histogram: None,
        decode_rows,
        const_cycles,
    };
    for (what, decode_rows, routed) in [
        ("fewer routed", vec![2, 1], 2),
        ("more routed", vec![2, 1], 4),
    ] {
        assert!(
            counts(decode_rows, routed).settle(&decode).is_err(),
            "{what}: settled"
        );
    }
    let mut beyond = vec![0; rows + 1];
    beyond[rows] = 1;
    assert!(
        counts(beyond, 1).settle(&decode).is_err(),
        "a row beyond the table: settled"
    );
    // Settled: each counted op's three instruction lookups, times its count.
    let mut settled = counts(vec![3, 0, 2], 5);
    settled.settle(&decode).expect("every op routed");
    let mut want = crate::tables::bitwise::BitwiseHistogram::new();
    for (row, n) in [(0u32, 3), (2, 2)] {
        for _ in 0..n {
            for op in crate::tables::cpu::const_lookups(decode.entry(row)) {
                want.bump(op);
            }
        }
    }
    assert!(
        settled.histogram.as_ref() == Some(&want),
        "the settled lookups"
    );
    // Without `const` nothing is deferred: settling changes nothing.
    let mut plain = counts(vec![3, 0, 2], 0);
    plain.settle(&decode).expect("nothing to settle");
    assert!(plain.histogram.is_none(), "settled without const");
}

#[test]
fn rows_sized_reuses_seven_buffers_for_an_equal_window_and_can_grow() {
    use super::super::{WalkOutputs, word_walk::WordRows};
    use crate::tables::{
        mem_load2w::Load2WRow,
        mem_load8::Load8Row,
        mem_loadw::LoadWRow,
        mem_store2w::Store2WRow,
        mem_store8::Store8Row,
        mem_storew::StoreWRow,
        memw_w::{MemwWordRow, MessageKind},
    };

    let fill = |rows: &mut WordRows, multiple| {
        rows.load8
            .resize(191 * multiple, Load8Row::new(8, 42, 0, 1).unwrap());
        rows.store8
            .resize(307 * multiple, Store8Row::new(8, 42, 0, 1, 2).unwrap());
        rows.loadw
            .resize(109 * multiple, LoadWRow::new(8, 42, 0, 2, true, 1).unwrap());
        rows.storew
            .resize(89 * multiple, StoreWRow::new(8, 42, 0, 2, 1, 2).unwrap());
        rows.load2w.resize(
            53 * multiple,
            Load2WRow::new(15, 42, [0, 1], 2, true, [1, 2]).unwrap(),
        );
        rows.store2w.resize(
            29 * multiple,
            Store2WRow::new(15, 42, [0, 1], 2, 1, [1, 2]).unwrap(),
        );
        rows.memw.resize(
            71 * multiple,
            MemwWordRow::new(MessageKind::Access, 8, 42, 0, 1, 2).unwrap(),
        );
    };
    let cycles = 8192;
    let window = |sizes: &WalkSizes, sized| {
        WalkOutputs::for_walk_sized(cycles, false, sizes, sized, MemwAlignedOrder::Limbs)
    };
    let mut first = window(&WalkSizes::default(), true);
    assert_eq!(first.word_rows.heap_bytes(), 0);
    fill(&mut first.word_rows, 1usize);
    let sizes = WalkSizes::of(&first, cycles);
    let mut equal = window(&sizes, true);
    let allocations = |rows: &WordRows| {
        (
            rows.buffers().map(|(ptr, _)| ptr),
            [
                rows.load8.capacity(),
                rows.store8.capacity(),
                rows.loadw.capacity(),
                rows.storew.capacity(),
                rows.load2w.capacity(),
                rows.store2w.capacity(),
                rows.memw.capacity(),
            ],
        )
    };
    let before = allocations(&equal.word_rows);
    fill(&mut equal.word_rows, 1usize);
    assert_eq!(
        allocations(&equal.word_rows),
        before,
        "no word-row list moved or grew"
    );
    assert_eq!(equal.word_rows, first.word_rows);
    for (&n, cap) in first.word_rows.lens().iter().zip(before.1) {
        assert!(
            (n..=n + n / 8 + 16).contains(&cap),
            "{n} rows, capacity {cap}"
        );
    }
    assert_eq!(equal.word_rows.copy.capacity(), 0);
    let mut heavy = window(&sizes, true);
    let mut control = window(&sizes, false);
    fill(&mut heavy.word_rows, 2usize);
    fill(&mut control.word_rows, 2usize);
    assert_eq!(heavy.word_rows, control.word_rows);
}
