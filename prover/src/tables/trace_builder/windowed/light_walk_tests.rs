//! The light walk (`WindowedTraceBuilder::light_walk`) against today's
//! record-mode walk: window by window, at both levels, every list the walk
//! emits, its counts and its refusals are today's (at level 2, with the
//! routing's segments and the CPU ops' lookups made by the walker); and a run
//! built at either level, on one walker or three, builds today's tables and
//! hands its chunks out in today's order.

use std::fmt::Debug;
use std::sync::Arc;

use executor::elf::Elf;
use executor::vm::fast::FastExecutor;
use executor::vm::history::{NO_ACCESS, WindowHistory};
use executor::vm::logs::Log;

use super::super::{HistoryOverlay, MemoryState, RegisterState, WalkSizes, route_ops_one_pass};
use super::parallel::{ExecWindow, walk_windows_from_history};
use super::{
    ChunkJob, RecordsRoom, StreamTable, StreamedChunk, Table, WalkCounts, WalkedWindow,
    WindowedTraceBuilder, walk_window,
};
use crate::tables::MaxRowsConfig;
use crate::tables::bitwise::BitwiseHistogram;
use crate::tables::decode::DecodeTable;
use crate::tables::fused_memory::MemoryFormat;
use crate::tables::memw_aligned::MemwAlignedOrder;
use crate::tables::register::RegisterFormat;
use crate::tables::trace_builder::Traces;
use crate::tests::windowed_builder_tests::{run, same_traces};

/// The asm programs of the record-mode tests: every ECALL the walk handles
/// but HINT (the Rust guests have it), misaligned and page-crossing accesses,
/// and a register idle past the in-row slot's reach.
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
const RUST_GUESTS: [&str; 5] = [
    "hint_min",
    "hint_multi",
    "dma_memset_cases",
    "dma_memmove_cases",
    "keccak_precompile",
];

/// A Rust guest's ELF from `program_artifacts/rust`.
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

/// A walker's counts as one histogram and DECODE's rows, for comparing.
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

/// A record-mode walk of one window, as the walkers walk it: at `light`
/// without the cells of a cycle that is no ECALL, with `route` routed where
/// walked; its CPU list kept. Returns the window and whether it was refused.
#[allow(clippy::too_many_arguments)]
fn walked(
    decode: &DecodeTable,
    order: MemwAlignedOrder,
    format: RegisterFormat,
    memory: MemoryFormat,
    batch: usize,
    (light, route): (bool, bool),
    counts: &mut WalkCounts,
    sizes: &mut WalkSizes,
    first: usize,
    logs: &[Log],
    history: &Arc<WindowHistory>,
) -> (WalkedWindow, bool) {
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
        order,
        format,
        memory,
        false,
        batch,
        false,
        Some(counts),
        true,
        route,
        sizes,
        &mut record,
        &mut registers,
        first,
        logs,
        None,
    )
    .expect("the walk");
    let refused = record.take_missing();
    (walked, refused)
}

/// Every list a walked window emits, as text.
fn lists(w: &WalkedWindow) -> [(&'static str, String); 18] {
    let l = &w.walk;
    [
        ("CPU", dbg(&w.cpu_ops)),
        ("register rows", dbg(&l.memw.register_rows)),
        ("MEMW_A", dbg(&l.memw.aligned)),
        ("MEMW", dbg(&l.memw.general)),
        ("LOAD", dbg(&l.load_ops)),
        ("LT", dbg(&l.lt_ops)),
        ("SHIFT", dbg(&l.shift_ops)),
        ("side records", dbg(&l.cpu_side)),
        ("in-walk lookups", dbg(&l.bitwise_ops)),
        ("KECCAK", dbg(&l.keccak_ops)),
        ("BLAKE3", dbg(&l.blake3_ops)),
        ("BLAKE3 absorb", dbg(&l.blake3_absorb_ops)),
        ("CPU32", dbg(&l.cpu32_ops)),
        ("ECSM", dbg(&l.ecsm_ops)),
        ("ECDAS", dbg(&l.ecdas_ops)),
        ("COPY", dbg(&l.copy_ops)),
        ("HINT", dbg(&l.hint_ops)),
        ("word rows", dbg(&l.word_rows)),
    ]
}

/// Runs `program` in history mode by windows of `window` cycles and walks
/// each window but the run's last three ways, in batches of `batch` cycles:
/// as today, at level 1 and at level 2. Each must emit today's lists (ECREC's
/// too), refuse what today's refuses, find today's last ECALL; level 2 must
/// route as today's accumulator routes. Over the run, the counts at level 1
/// must be today's walk's, at level 2 today's with the routing's lookups of
/// the CPU ops. Returns the windows compared.
fn light_is_todays(
    name: &str,
    program: &Elf,
    input: &[u8],
    window: usize,
    batch: usize,
    order: MemwAlignedOrder,
    format: RegisterFormat,
) -> usize {
    light_is_todays_under(
        name,
        program,
        input,
        (window, batch),
        order,
        format,
        MemoryFormat::Split,
    )
}

/// [`light_is_todays`] under the memory format `memory`: under the word
/// format the run's history is the word history, and the windows' word rows
/// are among the lists compared.
fn light_is_todays_under(
    name: &str,
    program: &Elf,
    input: &[u8],
    (window, batch): (usize, usize),
    order: MemwAlignedOrder,
    format: RegisterFormat,
    memory: MemoryFormat,
) -> usize {
    let words = memory == MemoryFormat::Words;
    let max_rows = MaxRowsConfig::small();
    let builder = WindowedTraceBuilder::new(program, input, &max_rows).expect("the builder");
    let decode = Arc::clone(&builder.decode);
    drop(builder);
    let levels = [(false, false), (true, false), (true, true)];
    let mut counts = levels.map(|_| WalkCounts::default());
    let mut sizes = levels.map(|_| WalkSizes::default());
    let mut cpu_lookups = BitwiseHistogram::new();
    let mut fast = FastExecutor::new(program, input.to_vec()).expect("the fast engine");
    if words {
        fast.enable_word_history().expect("the word history");
    } else {
        fast.enable_history().expect("history mode");
    }
    let (mut first, mut k) = (0usize, 0usize);
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
        let what = format!("{name}/{window}/{batch}/{order:?}/{format:?}/{memory:?} window {k}");
        let history = Arc::new(history);
        let mut walks = Vec::with_capacity(3);
        for (i, level) in levels.into_iter().enumerate() {
            walks.push(walked(
                &decode,
                order,
                format,
                memory,
                batch,
                level,
                &mut counts[i],
                &mut sizes[i],
                first,
                &logs,
                &history,
            ));
        }
        let (today, refused) = &walks[0];
        assert!(!refused, "{what}: today's walk refused the window");
        for op in &today.cpu_ops {
            decode.op(op).count_bitwise_into(&mut cpu_lookups);
        }
        // The accumulator routes the CPU's stores to STORE under the split
        // format alone.
        let routed = route_ops_one_pass(
            &today.cpu_ops,
            &decode,
            &today.walk.cpu32_ops,
            memory == MemoryFormat::Split,
        );
        let want = lists(today);
        for (level, (light, refused)) in walks.iter().enumerate().skip(1) {
            let what = format!("{what} level {level}");
            assert!(!refused, "{what}: refused the window");
            for ((list, got), (_, wanted)) in lists(light).iter().zip(&want) {
                assert!(got == wanted, "{what}: {list} differ");
            }
            assert!(
                dbg(&light.walk.ecrec_ops) == dbg(&today.walk.ecrec_ops),
                "{what}: ECREC differ"
            );
            assert_eq!(
                light.made.as_ref().map(|m| m.last_ecall),
                today.made.as_ref().map(|m| m.last_ecall),
                "{what}: the last ECALL"
            );
            match (level, light.routed.as_ref()) {
                (1, None) => {}
                (2, Some(s)) => {
                    assert!(
                        dbg(s) == dbg(&routed),
                        "{what}: the walker's routing is not the accumulator's"
                    );
                }
                (_, s) => panic!("{what}: routed {}", s.is_some()),
            }
        }
        first += logs.len();
        k += 1;
    }
    let (today, today_rows) = counted(&counts[0]);
    let mut with_routing = BitwiseHistogram::new();
    with_routing.merge(&today);
    with_routing.merge(&cpu_lookups);
    for (level, want) in [(1, &today), (2, &with_routing)] {
        let what = format!("{name}/{window}/{batch}/{order:?}/{format:?} level {level}");
        let (histogram, rows) = counted(&counts[level]);
        assert!(histogram == *want, "{what}: the lookups are not today's");
        assert_eq!(rows, today_rows, "{what}: DECODE's rows");
    }
    k
}

/// ★ Every asm program and Rust guest, and an ethrex fixture, at windows from
/// one cycle to many, batches that cross them, both MEMW_A forms and both
/// register formats: the light walk emits today's lists and counts, and its
/// walker routes as today's accumulator.
#[test]
fn the_light_walk_emits_today_s_lists() {
    let mut windows = 0;
    for name in PROGRAMS {
        let (program, _) = run(name);
        for (order, format) in [
            (MemwAlignedOrder::Limbs, RegisterFormat::InRow),
            (MemwAlignedOrder::Lt, RegisterFormat::InRow),
            (MemwAlignedOrder::Limbs, RegisterFormat::Tokens32),
        ] {
            for (window, batch) in [(1, 2048), (7, 3), (33, 2048), (4096, 5), (4096, 2048)] {
                windows += light_is_todays(name, &program, &[], window, batch, order, format);
            }
        }
    }
    for name in RUST_GUESTS {
        let program = rust_elf(name);
        for (window, batch) in [(97, 2048), (4096, 7)] {
            windows += light_is_todays(
                name,
                &program,
                &[],
                window,
                batch,
                MemwAlignedOrder::Limbs,
                RegisterFormat::InRow,
            );
        }
    }
    let ethrex = rust_elf("ethrex");
    let fixtures = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../executor/tests");
    let input = std::fs::read(fixtures.join("ethrex_10_transfers.bin")).expect("the fixture");
    windows += light_is_todays(
        "ethrex_10_transfers",
        &ethrex,
        &input,
        1 << 16,
        2048,
        MemwAlignedOrder::Limbs,
        RegisterFormat::InRow,
    );
    assert!(windows > 1000, "{windows} windows compared");
}

/// ★ The light walk refuses a window whose history misses a load's or a
/// store's access, or names another address for it, as today's walk refuses
/// it for the cells the cycle lacks.
#[test]
fn the_light_walk_refuses_a_history_without_the_access() {
    let (program, _) = run("all_loadstore_32");
    let max_rows = MaxRowsConfig::small();
    let builder = WindowedTraceBuilder::new(&program, &[], &max_rows).expect("the builder");
    let decode = Arc::clone(&builder.decode);
    drop(builder);
    let mut fast = FastExecutor::new(&program, Vec::new()).expect("the fast engine");
    fast.enable_history().expect("history mode");
    let accessed = |h: &WindowHistory| {
        h.extras
            .iter()
            .position(|e| e.side != NO_ACCESS && e.side & executor::vm::history::ECALL == 0)
    };
    // The run's first window with a load or store, before the one that halts.
    let (first, logs, history, at) = {
        let mut first = 0usize;
        loop {
            let (logs, history) = fast
                .resume_history(33)
                .expect("the run")
                .expect("a window with a load or store");
            assert!(
                logs.last().is_none_or(|l| l.next_pc != 0),
                "the run halted before a load or store"
            );
            if let Some(at) = accessed(&history) {
                break (first, logs, history, at);
            }
            first += logs.len();
        }
    };
    let mut no_access = history.clone();
    no_access.extras[at].side = NO_ACCESS;
    let mut moved = history.clone();
    let index = moved.extras[at].side as usize;
    moved.accesses[index].addr ^= 8;
    for (what, tampered) in [("no access", no_access), ("another address", moved)] {
        let tampered = Arc::new(tampered);
        for level in [(false, false), (true, false), (true, true)] {
            let (_, refused) = walked(
                &decode,
                MemwAlignedOrder::Limbs,
                RegisterFormat::InRow,
                MemoryFormat::Split,
                2048,
                level,
                &mut WalkCounts::default(),
                &mut WalkSizes::default(),
                first,
                &logs,
                &tampered,
            );
            assert!(refused, "{what}: level {level:?} walked the window");
        }
    }
    // The history as recorded: walked at every level.
    let history = Arc::new(history);
    for level in [(false, false), (true, false), (true, true)] {
        let (_, refused) = walked(
            &decode,
            MemwAlignedOrder::Limbs,
            RegisterFormat::InRow,
            MemoryFormat::Split,
            2048,
            level,
            &mut WalkCounts::default(),
            &mut WalkSizes::default(),
            first,
            &logs,
            &history,
        );
        assert!(!refused, "the recorded history: level {level:?} refused it");
    }
}

/// The programs the word memory format walks: the CPU's loads and stores of
/// every width and alignment (across `2^32` too), and the chips it takes
/// (COPY, keccak256, ECREC).
const WORD_PROGRAMS: [&str; 10] = [
    "all_instructions_64",
    "all_loadstore_32",
    "cross_4gib",
    "copy_4gib",
    "misalign_ld",
    "misalign_sd",
    "misalign_sh",
    "test_commit_split",
    "test_copy_ecalls",
    "test_keccak256",
];

/// ★ Under the word memory format the light walk emits today's lists: every
/// window of each run, walked from the word history at levels 1 and 2 (a
/// load or store reading its words from its word access record, no cell
/// made), emits the word rows and every other list the cell walk emits,
/// refuses nothing, and counts today's lookups; at windows from one cycle to
/// many and batches that cross them. ECREC's run too (its rows compared on
/// their own).
#[test]
fn the_light_walk_emits_today_s_lists_under_the_word_format() {
    let mut windows = 0;
    for name in WORD_PROGRAMS.into_iter().chain(["test_ecrec"]) {
        let (program, _) = run(name);
        for (window, batch) in [(1, 2048), (7, 3), (33, 2048), (4096, 5)] {
            windows += light_is_todays_under(
                name,
                &program,
                &[],
                (window, batch),
                MemwAlignedOrder::Limbs,
                RegisterFormat::InRow,
                MemoryFormat::Words,
            );
        }
    }
    assert!(windows > 500, "{windows} windows compared");
}

/// ★ Under the word memory format the light walk refuses a window whose word
/// history misses a load's or a store's word record, or names another address
/// for it, as the cell walk refuses it for the cells the cycle lacks; the
/// history as recorded is walked at every level.
#[test]
fn the_light_walk_refuses_a_word_history_without_the_access() {
    let (program, _) = run("all_loadstore_32");
    let max_rows = MaxRowsConfig::small();
    let builder = WindowedTraceBuilder::new(&program, &[], &max_rows).expect("the builder");
    let decode = Arc::clone(&builder.decode);
    drop(builder);
    let mut fast = FastExecutor::new(&program, Vec::new()).expect("the fast engine");
    fast.enable_word_history().expect("the word history");
    let accessed = |h: &WindowHistory| {
        h.extras
            .iter()
            .position(|e| e.side != NO_ACCESS && e.side & executor::vm::history::ECALL == 0)
    };
    let (first, logs, history, at) = {
        let mut first = 0usize;
        loop {
            let (logs, history) = fast
                .resume_word_history(33)
                .expect("the run")
                .expect("a window with a load or store");
            assert!(history.words, "a word history");
            assert!(
                logs.last().is_none_or(|l| l.next_pc != 0),
                "the run halted before a load or store"
            );
            if let Some(at) = accessed(&history) {
                break (first, logs, history, at);
            }
            first += logs.len();
        }
    };
    let mut no_access = history.clone();
    no_access.extras[at].side = NO_ACCESS;
    let mut moved = history.clone();
    let index = moved.extras[at].side as usize;
    moved.word_accesses[index].addr ^= 8;
    let walk = |history: &Arc<WindowHistory>, level| {
        walked(
            &decode,
            MemwAlignedOrder::Limbs,
            RegisterFormat::InRow,
            MemoryFormat::Words,
            2048,
            level,
            &mut WalkCounts::default(),
            &mut WalkSizes::default(),
            first,
            &logs,
            history,
        )
        .1
    };
    for (what, tampered) in [("no access", no_access), ("another address", moved)] {
        let tampered = Arc::new(tampered);
        for level in [(false, false), (true, false), (true, true)] {
            assert!(
                walk(&tampered, level),
                "{what}: level {level:?} walked the window"
            );
        }
    }
    let history = Arc::new(history);
    for level in [(false, false), (true, false), (true, true)] {
        assert!(
            !walk(&history, level),
            "the recorded history: level {level:?} refused it"
        );
    }
}

/// A room rule that admits every chunk.
fn always() -> RecordsRoom {
    Arc::new(|_| true)
}

/// `chunk` with its records replaced by their host fill at eight bytes a
/// cell.
fn unheld(mut chunk: StreamedChunk) -> StreamedChunk {
    use math::field::element::FieldElement;
    let Some(records) = chunk.trace.take_records_main() else {
        return chunk;
    };
    let packed = records.host_fill().expect("the host fills its records");
    let columns = packed
        .widen::<crate::tables::types::GoldilocksField>()
        .expect("the packed columns widen");
    let (rows, cols) = (records.rows(), records.cols());
    let mut data: Vec<FieldElement<crate::tables::types::GoldilocksField>> =
        Vec::with_capacity(rows * cols);
    for r in 0..rows {
        data.extend(columns.iter().map(|column| column.evals()[r]));
    }
    chunk.trace = Table::new_main(data, cols, 1);
    chunk
}

/// The run of `program` by windows of `window` cycles under `format`, walked
/// from the fast engine's history on `walkers` threads at light walk `level`,
/// its CPU chunks held as their logs, counted where made, no CPU list: its
/// traces (every chunk filled on the host) and the chunks in hand-out order.
fn built(
    program: &Elf,
    max_rows: &MaxRowsConfig,
    window: usize,
    walkers: usize,
    level: u8,
    format: RegisterFormat,
) -> (Traces, Vec<(StreamTable, usize)>) {
    let mut builder = WindowedTraceBuilder::new(program, &[], max_rows)
        .and_then(|b| b.register_format(format))
        .and_then(|b| b.memw_aligned_order(MemwAlignedOrder::Limbs))
        .and_then(WindowedTraceBuilder::drop_streamed_ops)
        .and_then(|b| b.card_fill_streamed(always(), true, format != RegisterFormat::InRow))
        .map(|b| b.count_where_made(true).drop_cpu_list(true))
        .and_then(|b| b.light_walk(level))
        .expect("the builder");
    let (tx, rx) = std::sync::mpsc::sync_channel::<ExecWindow>(2);
    let mut jobs: Vec<ChunkJob> = Vec::new();
    let report = std::thread::scope(|scope| {
        scope.spawn(move || {
            let mut fast = FastExecutor::new(program, Vec::new()).expect("the fast engine");
            fast.enable_history().expect("history mode");
            while let Some((logs, history)) = fast.resume_history(window).expect("the run") {
                let last = (fast.pc() == 0)
                    .then(|| {
                        fast.take_history_final()
                            .map(crate::tables::trace_builder::windowed::parallel::RunEnd::boxed)
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
            jobs.extend(accumulator.absorb(walked));
            Ok(())
        })
    })
    .expect("the walk");
    let order: Vec<(StreamTable, usize)> = jobs.iter().map(|job| (job.table, job.index)).collect();
    let chunks: Vec<StreamedChunk> = jobs.into_iter().map(|job| unheld(job.generate())).collect();
    let mut traces = builder.finish(&report.last).expect("the finish");
    traces
        .insert_streamed(chunks)
        .expect("every chunk has a placeholder");
    (traces, order)
}

/// ★ A run built at either level of the light walk, on one walker and on
/// three, under both register formats, hands its chunks out in today's order
/// and builds today's tables, row for row: CPU's side records, MEMW_A, LOAD,
/// BITWISE, DECODE, LT and the routed tables among them.
#[test]
fn the_light_walk_builds_today_s_tables() {
    let max_rows = MaxRowsConfig::small();
    let mut chunks = 0usize;
    for name in PROGRAMS {
        let (program, _) = run(name);
        for format in [RegisterFormat::InRow, RegisterFormat::Tokens32] {
            for window in [7, 33, 4096] {
                let (want, want_order) = built(&program, &max_rows, window, 2, 0, format);
                chunks += want_order.len();
                for (level, walkers) in [(1, 3), (2, 1), (2, 3)] {
                    let what = format!("{name}/{format:?}/{window}/level {level}/{walkers}");
                    let (got, order) = built(&program, &max_rows, window, walkers, level, format);
                    assert_eq!(order, want_order, "{what}: the hand-out");
                    same_traces(&want, &got);
                }
            }
        }
    }
    assert!(chunks > 0, "no chunk handed out");
}

/// The level refuses what it does not take, and only before the first window.
#[test]
fn the_light_walk_takes_levels_0_to_2() {
    let (program, _) = run("all_loadstore_32");
    let max_rows = MaxRowsConfig::small();
    let new = || WindowedTraceBuilder::new(&program, &[], &max_rows).expect("the builder");
    for level in 0..=2 {
        assert!(new().light_walk(level).is_ok(), "level {level}");
    }
    assert!(new().light_walk(3).is_err(), "level 3");
    // Routing where walked needs the counts where made.
    let b = new()
        .drop_streamed_ops()
        .map(|b| b.count_where_made(false))
        .and_then(|b| b.light_walk(2))
        .expect("the builder");
    assert_eq!(
        b.light_walks(),
        (true, false),
        "no routing without the counts"
    );
}
