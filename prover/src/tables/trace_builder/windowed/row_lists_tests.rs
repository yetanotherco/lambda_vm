//! The walkers' row lists (`WindowedTraceBuilder::card_row_lists`) against
//! the rows they stand for: window by window, a walk that lists LOAD's,
//! STORE's and MEMW_A's rows lists exactly the rows today's walk makes, in
//! their order, makes every other list as today's and counts what today's
//! walk and hand-out count; the builder and the walk refuse what the lists
//! cannot take; and a run whose listed rows no chunk takes is refused at its
//! finish.

use std::fmt::Debug;
use std::sync::Arc;

use executor::elf::Elf;
use executor::vm::fast::FastExecutor;
use executor::vm::history::WindowHistory;
use executor::vm::logs::Log;

use super::super::{HistoryOverlay, MemoryState, RegisterState, WalkSizes};
use super::parallel::{ExecWindow, walk_windows_from_history};
use super::{
    ALIGNED_ECALL, AlignedRef, RecordsRoom, RowLists, WalkCounts, WalkedWindow,
    WindowedTraceBuilder, walk_window,
};
use crate::tables::MaxRowsConfig;
use crate::tables::bitwise::BitwiseHistogram;
use crate::tables::decode::DecodeTable;
use crate::tables::memw_aligned::MemwAlignedOrder;
use crate::tables::register::RegisterFormat;
use crate::tests::windowed_builder_tests::run;

/// The asm programs: every ECALL the walk handles but HINT, misaligned and
/// page-crossing accesses, a register idle past the slot's reach.
const PROGRAMS: [&str; 15] = [
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
    "keep_alive_regs",
];

/// The Rust guests with HINT, DMA and Keccak calls.
const RUST_GUESTS: [&str; 3] = ["hint_multi", "dma_memmove_cases", "keccak_precompile"];

fn rust_elf(name: &str) -> Elf {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../executor/program_artifacts/rust")
        .join(format!("{name}.elf"));
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    Elf::load(&bytes).expect("the ELF loads")
}

fn dbg<T: Debug + ?Sized>(x: &T) -> String {
    format!("{x:?}")
}

fn counted(counts: &WalkCounts) -> (BitwiseHistogram, Vec<u64>) {
    let mut histogram = BitwiseHistogram::new();
    if let Some(h) = counts.histogram.as_ref() {
        histogram.merge(h);
    }
    let mut rows = counts.decode_rows.clone();
    while rows.last() == Some(&0) {
        rows.pop();
    }
    (histogram, rows)
}

/// One record-mode window walked with the walkers' routing (and at `light`
/// without the cells of cycles that are no ECALL), its rows listed as
/// `lists` says; its CPU list kept.
#[allow(clippy::too_many_arguments)]
fn walked(
    decode: &DecodeTable,
    batch: usize,
    light: bool,
    lists: RowLists,
    counts: &mut WalkCounts,
    sizes: &mut WalkSizes,
    first: usize,
    logs: &[Log],
    history: &Arc<WindowHistory>,
) -> WalkedWindow {
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
        MemwAlignedOrder::Limbs,
        RegisterFormat::InRow,
        false,
        batch,
        false,
        Some(counts),
        true,
        true,
        lists,
        sizes,
        &mut record,
        &mut registers,
        first,
        logs,
        None,
        None,
    )
    .expect("the walk");
    assert!(!record.take_missing(), "the walk refused the window");
    walked
}

/// Runs `program` in history mode by windows of `window` cycles and walks
/// each window but the last twice, routed by its walker: making every row,
/// and listing LOAD's, STORE's and MEMW_A's. The lists must be the rows'
/// cycles in the rows' order (an ECALL cycle's MEMW_A rows whole and
/// marked), the window's other lists and its routing the same, and over the
/// run the counts the same once the STORE rows' lookups, which the hand-out
/// counts from today's STORE ops, are added to today's. Returns the windows
/// compared.
fn lists_are_the_rows(
    name: &str,
    program: &Elf,
    input: &[u8],
    window: usize,
    batch: usize,
    light: bool,
) -> usize {
    let max_rows = MaxRowsConfig::small();
    let builder = WindowedTraceBuilder::new(program, input, &max_rows).expect("the builder");
    let decode = Arc::clone(&builder.decode);
    drop(builder);
    let all = RowLists {
        load: true,
        store: true,
        memw_a: true,
    };
    let (mut today_counts, mut listed_counts) = (WalkCounts::default(), WalkCounts::default());
    let (mut today_sizes, mut listed_sizes) = (WalkSizes::default(), WalkSizes::default());
    let mut store_lookups = BitwiseHistogram::new();
    let mut fast = FastExecutor::new(program, input.to_vec()).expect("the fast engine");
    fast.enable_history().expect("history mode");
    let (mut first, mut k) = (0usize, 0usize);
    loop {
        let Some((logs, history)) = fast.resume_history(window).expect("the run") else {
            break;
        };
        if logs.last().is_some_and(|l| l.next_pc == 0) {
            break;
        }
        let what = format!("{name}/{window}/{batch}/light {light} window {k}");
        let history = Arc::new(history);
        let today = walked(
            &decode,
            batch,
            light,
            RowLists::NONE,
            &mut today_counts,
            &mut today_sizes,
            first,
            &logs,
            &history,
        );
        let listed = walked(
            &decode,
            batch,
            light,
            all,
            &mut listed_counts,
            &mut listed_sizes,
            first,
            &logs,
            &history,
        );
        let (t, l) = (&today.walk, &listed.walk);
        let lists = listed.lists.as_deref().expect("the window's lists");
        // LOAD: each row's cycle, in order.
        let cycle = |ts: u64| ((ts - 4) / 4) as u32;
        let load: Vec<u32> = t.load_ops.iter().map(|op| cycle(op.timestamp)).collect();
        assert_eq!(lists.load_cycles, load, "{what}: LOAD's cycles");
        assert!(l.load_ops.is_empty(), "{what}: LOAD ops made");
        // MEMW_A: a regular row's cycle and access, an ECALL cycle's row
        // whole.
        let mut rows = Vec::new();
        let mut whole = Vec::new();
        for row in &t.memw.aligned {
            let c = cycle(row.timestamp());
            let raw = &today.cpu_ops[c as usize - first];
            if decode.op(raw).decode.fields.ecall {
                rows.push(AlignedRef {
                    cycle: ALIGNED_ECALL | c,
                    at: whole.len() as u32,
                });
                whole.push(*row);
            } else {
                let at = history.extras[c as usize - first].side;
                let a = history.accesses[at as usize];
                assert_eq!(a.addr, row.base_address(), "{what}: cycle {c}'s access");
                rows.push(AlignedRef { cycle: c, at });
            }
        }
        assert_eq!(lists.aligned_rows, rows, "{what}: MEMW_A's list");
        assert!(
            dbg(&lists.aligned_whole) == dbg(&whole),
            "{what}: MEMW_A's ECALL rows"
        );
        assert!(l.memw.aligned.is_empty(), "{what}: MEMW_A rows made");
        // Every other list as today's.
        let other = |w: &super::super::WalkOutputs| {
            [
                dbg(&w.memw.register_rows),
                dbg(&w.memw.general),
                dbg(&w.lt_ops),
                dbg(&w.shift_ops),
                dbg(&w.cpu_side),
                dbg(&w.bitwise_ops),
                dbg(&w.keccak_ops),
                dbg(&w.blake3_ops),
                dbg(&w.blake3_absorb_ops),
                dbg(&w.cpu32_ops),
                dbg(&w.ecsm_ops),
                dbg(&w.ecdas_ops),
                dbg(&w.copy_ops),
                dbg(&w.hint_ops),
                dbg(&w.ecrec_ops),
            ]
        };
        assert!(other(t) == other(l), "{what}: another list differs");
        assert!(
            dbg(&today.cpu_ops) == dbg(&listed.cpu_ops),
            "{what}: CPU ops"
        );
        // STORE: each routed store's cycle; every other segment as today's.
        let (rt, rl) = (
            today.routed.as_ref().expect("today's routing"),
            listed.routed.as_ref().expect("the listed walk's routing"),
        );
        let store: Vec<u32> = rt.store_ops.iter().map(|op| cycle(op.timestamp)).collect();
        assert_eq!(
            rl.store_cycles.as_deref(),
            Some(&store[..]),
            "{what}: STORE's cycles"
        );
        assert!(rl.store_ops.is_empty(), "{what}: STORE ops routed");
        assert!(
            rt.store_cycles.is_none(),
            "{what}: today's routing listed cycles"
        );
        let strip = |s: &super::super::RoutedSegments| {
            [
                dbg(&s.branch_ops),
                dbg(&s.mul_filter),
                dbg(&s.dvrm_filter),
                dbg(&s.eq_ops),
                dbg(&s.bytewise_ops),
                dbg(&s.shift_cpu32),
                dbg(&s.mul_cpu32),
                dbg(&s.dvrm_cpu32),
                dbg(&s.bitwise_cpu32),
                dbg(&s.lt_dvrm_filter),
                dbg(&s.lt_dvrm_cpu32),
                dbg(&s.mul_dvrm_filter),
                dbg(&s.mul_dvrm_cpu32),
            ]
        };
        assert!(
            strip(rt) == strip(rl),
            "{what}: another routed segment differs"
        );
        for op in &rt.store_ops {
            store_lookups.add_ops(&op.collect_bitwise_ops());
        }
        first += logs.len();
        k += 1;
    }
    let (today, today_rows) = counted(&today_counts);
    let (listed, listed_rows) = counted(&listed_counts);
    let mut want = BitwiseHistogram::new();
    want.merge(&today);
    want.merge(&store_lookups);
    let what = format!("{name}/{window}/{batch}/light {light}");
    assert!(
        listed == want,
        "{what}: the listed walk's lookups are not today's and the hand-out's"
    );
    assert_eq!(listed_rows, today_rows, "{what}: DECODE's rows");
    k
}

/// ★ Every asm program, three Rust guests and an ethrex fixture, at windows
/// from one cycle to many and batches that cross them, with and without the
/// light walk's cells skipped: the lists are the rows today's walk makes.
#[test]
fn the_walkers_list_the_rows_they_no_longer_make() {
    let mut windows = 0;
    for name in PROGRAMS {
        let (program, _) = run(name);
        for (window, batch) in [(1, 2048), (7, 3), (33, 2048), (4096, 5), (4096, 2048)] {
            for light in [false, true] {
                windows += lists_are_the_rows(name, &program, &[], window, batch, light);
            }
        }
    }
    for name in RUST_GUESTS {
        let program = rust_elf(name);
        windows += lists_are_the_rows(name, &program, &[], 4096, 7, true);
    }
    let ethrex = rust_elf("ethrex");
    let fixtures = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../executor/tests");
    let input = std::fs::read(fixtures.join("ethrex_10_transfers.bin")).expect("the fixture");
    windows += lists_are_the_rows("ethrex_10_transfers", &ethrex, &input, 1 << 16, 2048, true);
    assert!(windows > 1000, "{windows} windows compared");
}

fn always() -> RecordsRoom {
    Arc::new(|_| true)
}

/// A builder of the windowed record-mode build the lists need, at light
/// walk `level`, its MEMW_A form `order` and register format `format`.
fn ready<'a>(
    program: &'a Elf,
    max_rows: &'a MaxRowsConfig,
    format: RegisterFormat,
    order: MemwAlignedOrder,
    level: u8,
) -> WindowedTraceBuilder<'a> {
    WindowedTraceBuilder::new(program, &[], max_rows)
        .and_then(|b| b.register_format(format))
        .and_then(|b| b.memw_aligned_order(order))
        .and_then(WindowedTraceBuilder::drop_streamed_ops)
        .and_then(|b| b.card_fill_streamed(always(), true, false))
        .map(|b| b.count_where_made(true).drop_cpu_list(true))
        .and_then(|b| b.light_walk(level))
        .expect("the builder")
}

/// The run of `builder`'s program by windows of `window` cycles on `walkers`
/// threads in record mode: its finish's outcome.
fn finished(
    mut builder: WindowedTraceBuilder<'_>,
    program: &Elf,
    window: usize,
    walkers: usize,
    recheck: usize,
) -> Result<(), crate::Error> {
    let (tx, rx) = std::sync::mpsc::sync_channel::<ExecWindow>(2);
    let report = std::thread::scope(|scope| {
        scope.spawn(move || {
            let mut fast = FastExecutor::new(program, Vec::new()).expect("the fast engine");
            fast.enable_history().expect("history mode");
            let mut k = 0usize;
            loop {
                let start = (recheck > 0 && k.is_multiple_of(recheck)).then(|| fast.snapshot());
                let Some((logs, history)) = fast.resume_history(window).expect("the run") else {
                    break;
                };
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
                drop(accumulator.absorb(walked));
                Ok(())
            },
        )
    })?;
    builder.finish(&report.last).map(drop)
}

/// ★ The builder and the walk refuse what the lists cannot take; a run whose
/// listed rows no chunk takes is refused at its finish, LOAD's and MEMW_A's
/// as STORE's, and nothing panics; with no list the run finishes.
#[test]
fn listed_rows_no_chunk_takes_refuse_the_run() {
    let (program, _) = run("all_loadstore_32");
    let max_rows = MaxRowsConfig::small();
    let all = RowLists {
        load: true,
        store: true,
        memw_a: true,
    };
    let inrow = |level| {
        ready(
            &program,
            &max_rows,
            RegisterFormat::InRow,
            MemwAlignedOrder::Limbs,
            level,
        )
    };
    // The builder's refusals.
    assert!(
        ready(
            &program,
            &max_rows,
            RegisterFormat::Tokens32,
            MemwAlignedOrder::Limbs,
            2
        )
        .card_row_lists(all)
        .is_err(),
        "the format before"
    );
    assert!(
        ready(
            &program,
            &max_rows,
            RegisterFormat::InRow,
            MemwAlignedOrder::Lt,
            2
        )
        .card_row_lists(RowLists {
            memw_a: true,
            ..RowLists::NONE
        })
        .is_err(),
        "MEMW_A in its LT form"
    );
    assert!(
        ready(
            &program,
            &max_rows,
            RegisterFormat::InRow,
            MemwAlignedOrder::Lt,
            2
        )
        .card_row_lists(RowLists {
            load: true,
            ..RowLists::NONE
        })
        .is_ok(),
        "LOAD alone takes either form"
    );
    // The walk's refusals.
    let store = RowLists {
        store: true,
        ..RowLists::NONE
    };
    let b = inrow(1).card_row_lists(store).expect("the lists");
    assert!(
        finished(b, &program, 33, 2, 0).is_err(),
        "STORE without the walkers routing"
    );
    let b = inrow(2).card_row_lists(all).expect("the lists");
    assert!(finished(b, &program, 33, 2, 3).is_err(), "a re-walk");
    // The finish's: no chunk of this build takes the rows.
    for (what, lists) in [
        (
            "LOAD",
            RowLists {
                load: true,
                ..RowLists::NONE
            },
        ),
        ("STORE", store),
        (
            "MEMW_A",
            RowLists {
                memw_a: true,
                ..RowLists::NONE
            },
        ),
    ] {
        let b = inrow(2).card_row_lists(lists).expect("the lists");
        let finish = finished(b, &program, 33, 3, 0);
        assert!(
            matches!(&finish, Err(crate::Error::Prover(line)) if line.contains("no chunk took")),
            "{what}: {finish:?}"
        );
    }
    // No list: the run finishes.
    assert!(finished(inrow(2), &program, 33, 3, 0).is_ok(), "no list");
}
