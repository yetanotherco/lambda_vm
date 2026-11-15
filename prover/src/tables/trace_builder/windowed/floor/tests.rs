//! The floor arm against today's record mode, under the in-row register
//! format: window by window, its light pass counts what the walk counts (and,
//! routing on the walkers, what the accumulator's routing counts), keeps the
//! lists the build still keeps and counts each family's rows; and a run built
//! with it builds every table but the families' as today's build does, its
//! chunks handed out in the same order.

use std::fmt::Debug;
use std::sync::Arc;

use executor::elf::Elf;
use executor::vm::fast::FastExecutor;
use math::field::element::FieldElement;

use super::super::super::{
    HistoryOverlay, MemoryState, RegisterState, WalkSizes, route_ops_one_pass,
};
use super::super::card_records::RowChunk;
use super::super::parallel::{ExecWindow, walk_windows_from_history};
use super::super::{
    ChunkJob, RecordsRoom, StreamTable, StreamedChunk, Table, WalkCounts, WindowedTraceBuilder,
    walk_window,
};
use super::{FAMILIES, FloorRoute, FloorWorker, Placeholder, light_window};
use crate::tables::MaxRowsConfig;
use crate::tables::bitwise::BitwiseHistogram;
use crate::tables::cpu;
use crate::tables::memw_aligned::MemwAlignedOrder;
use crate::tables::register::RegisterFormat;
use crate::tables::trace_builder::Traces;
use crate::tables::types::GoldilocksField;
use crate::tests::windowed_builder_tests::run;

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

/// Runs `program` in history mode by windows of `window` cycles and, for each
/// window but the run's last, walks it as record mode does (in-row registers,
/// MEMW_A in `order`, counted where made) and gives it the light pass under
/// both routes, all in batches of `batch` cycles. Each light window must hold
/// the walk's MEMW ops, LT, SHIFT, CPU32 and accelerator ops, MEMW_A's LT
/// pairs, each family's rows and the last ECALL; routing on the walkers, the
/// routing's segments. Over the run, the light pass's counts must be the
/// walk's (routing on the walkers, and the routing's lookups of the CPU ops).
/// Returns the windows compared.
fn light_is_the_walks(
    name: &str,
    program: &Elf,
    input: &[u8],
    window: usize,
    batch: usize,
    order: MemwAlignedOrder,
) -> usize {
    let max_rows = MaxRowsConfig::small();
    let builder = WindowedTraceBuilder::new(program, input, &max_rows).expect("the builder");
    let decode = Arc::clone(&builder.decode);
    drop(builder);
    let routes = [FloorRoute::Accumulator, FloorRoute::Walkers];
    let mut workers = routes.map(|_| FloorWorker::new(order));
    let mut light_counts = routes.map(|_| WalkCounts::default());
    let mut walk_counts = WalkCounts::default();
    // The CPU ops' own lookups, which the accumulator's routing counts.
    let mut cpu_lookups = BitwiseHistogram::new();
    let mut sizes = WalkSizes::default();
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
        let what = format!("{name}/{window}/{batch}/{order:?} window {k}");
        let history = Arc::new(history);
        let cells = history.registers.expect("the window's register cells");
        // Today's record-mode walk, its CPU list kept for the comparison.
        let mut record = MemoryState::new();
        record.overlay = Some(Box::new(HistoryOverlay::new(Arc::clone(&history))));
        let mut registers = RegisterState {
            regs: cells.regs,
            index_register: cells.index,
            pc_register: cells.pc,
        };
        let walked = walk_window(
            &decode,
            order,
            RegisterFormat::InRow,
            false,
            batch,
            false,
            Some(&mut walk_counts),
            true,
            &mut sizes,
            &mut record,
            &mut registers,
            first,
            &logs,
            None,
        )
        .unwrap_or_else(|e| panic!("{what}: the walk: {e:?}"));
        assert!(!record.take_missing(), "{what}: the walk missed a cell");
        let w = &walked.walk;
        for op in &walked.cpu_ops {
            decode.op(op).count_bitwise_into(&mut cpu_lookups);
        }
        let routed = route_ops_one_pass(&walked.cpu_ops, &decode, &w.cpu32_ops);
        let want_last = walked.made.as_ref().expect("counted where made").last_ecall;
        for (r, route) in routes.into_iter().enumerate() {
            let what = format!("{what} {route:?}");
            let mut light_record = MemoryState::new();
            light_record.overlay = Some(Box::new(HistoryOverlay::new(Arc::clone(&history))));
            let light = light_window(
                &mut workers[r],
                &mut light_counts[r],
                &decode,
                order,
                batch,
                false,
                route,
                &mut light_record,
                first,
                &logs,
                &history,
            )
            .unwrap_or_else(|e| panic!("{what}: the light pass: {e:?}"));
            assert!(
                !light_record.take_missing(),
                "{what}: the light pass missed a cell"
            );
            let f = light.floor.as_deref().expect("a light window");
            assert_eq!(light.cycles, logs.len(), "{what}: cycles");
            assert!(light.cpu_ops.is_empty(), "{what}: no CPU op kept");
            let l = &light.walk;
            assert!(
                l.memw.register_rows.is_empty()
                    && l.memw.aligned.is_empty()
                    && l.load_ops.is_empty()
                    && l.cpu_side.is_empty(),
                "{what}: a residual row, MEMW_A row, LOAD op or side record made"
            );
            assert_eq!(
                f.rows,
                [
                    w.memw.register_rows.len(),
                    w.memw.aligned.len(),
                    w.load_ops.len()
                ],
                "{what}: the families' rows"
            );
            let pairs: Vec<(u64, u64)> =
                w.memw.aligned.iter().filter_map(|r| r.lt_pair()).collect();
            assert_eq!(f.pairs, pairs, "{what}: MEMW_A's LT pairs");
            let lists: [(&str, String, String); 12] = [
                ("MEMW", dbg(&l.memw.general), dbg(&w.memw.general)),
                ("LT", dbg(&l.lt_ops), dbg(&w.lt_ops)),
                ("SHIFT", dbg(&l.shift_ops), dbg(&w.shift_ops)),
                ("CPU32", dbg(&l.cpu32_ops), dbg(&w.cpu32_ops)),
                ("KECCAK", dbg(&l.keccak_ops), dbg(&w.keccak_ops)),
                ("BLAKE3", dbg(&l.blake3_ops), dbg(&w.blake3_ops)),
                (
                    "BLAKE3 absorb",
                    dbg(&l.blake3_absorb_ops),
                    dbg(&w.blake3_absorb_ops),
                ),
                ("ECSM", dbg(&l.ecsm_ops), dbg(&w.ecsm_ops)),
                ("ECDAS", dbg(&l.ecdas_ops), dbg(&w.ecdas_ops)),
                ("COPY", dbg(&l.copy_ops), dbg(&w.copy_ops)),
                ("HINT", dbg(&l.hint_ops), dbg(&w.hint_ops)),
                ("ECREC", dbg(&l.ecrec_ops), dbg(&w.ecrec_ops)),
            ];
            for (list, got, wanted) in lists {
                assert!(got == wanted, "{what}: {list} ops differ");
            }
            assert!(l.bitwise_ops.is_empty(), "{what}: lookups listed");
            assert_eq!(
                light.made.as_ref().map(|m| m.last_ecall),
                Some(want_last),
                "{what}: the last ECALL"
            );
            match (route, f.segments.as_ref()) {
                (FloorRoute::Accumulator, None) => {}
                (FloorRoute::Walkers, Some(s)) => {
                    let segments: [(&str, String, String); 14] = [
                        ("branch", dbg(&s.branch_ops), dbg(&routed.branch_ops)),
                        ("mul", dbg(&s.mul_filter), dbg(&routed.mul_filter)),
                        ("dvrm", dbg(&s.dvrm_filter), dbg(&routed.dvrm_filter)),
                        ("eq", dbg(&s.eq_ops), dbg(&routed.eq_ops)),
                        ("bytewise", dbg(&s.bytewise_ops), dbg(&routed.bytewise_ops)),
                        ("store", dbg(&s.store_ops), dbg(&routed.store_ops)),
                        ("shift cpu32", dbg(&s.shift_cpu32), dbg(&routed.shift_cpu32)),
                        ("mul cpu32", dbg(&s.mul_cpu32), dbg(&routed.mul_cpu32)),
                        ("dvrm cpu32", dbg(&s.dvrm_cpu32), dbg(&routed.dvrm_cpu32)),
                        (
                            "bitwise cpu32",
                            dbg(&s.bitwise_cpu32),
                            dbg(&routed.bitwise_cpu32),
                        ),
                        (
                            "lt dvrm",
                            dbg(&s.lt_dvrm_filter),
                            dbg(&routed.lt_dvrm_filter),
                        ),
                        (
                            "lt dvrm cpu32",
                            dbg(&s.lt_dvrm_cpu32),
                            dbg(&routed.lt_dvrm_cpu32),
                        ),
                        (
                            "mul dvrm",
                            dbg(&s.mul_dvrm_filter),
                            dbg(&routed.mul_dvrm_filter),
                        ),
                        (
                            "mul dvrm cpu32",
                            dbg(&s.mul_dvrm_cpu32),
                            dbg(&routed.mul_dvrm_cpu32),
                        ),
                    ];
                    for (segment, got, wanted) in segments {
                        assert!(got == wanted, "{what}: the {segment} segment differs");
                    }
                }
                (route, s) => panic!("{what}: {route:?} with segments {}", s.is_some()),
            }
        }
        first += logs.len();
        k += 1;
    }
    let (walk_histogram, walk_rows) = counted(&walk_counts);
    let mut with_routing = BitwiseHistogram::new();
    with_routing.merge(&walk_histogram);
    with_routing.merge(&cpu_lookups);
    for (r, route) in routes.into_iter().enumerate() {
        let what = format!("{name}/{window}/{batch}/{order:?} {route:?}");
        let (histogram, rows) = counted(&light_counts[r]);
        let want = match route {
            FloorRoute::Accumulator => &walk_histogram,
            FloorRoute::Walkers => &with_routing,
        };
        assert!(histogram == *want, "{what}: the lookups are not the walk's");
        assert_eq!(rows, walk_rows, "{what}: DECODE's rows");
    }
    k
}

/// ★ Every asm program and Rust guest, and an ethrex fixture, at windows from
/// one cycle to many, batches that cross them and both MEMW_A forms: the
/// light pass counts what today's walk counts (and, routing, what the
/// accumulator's routing counts), and keeps what the build still keeps.
#[test]
fn the_light_pass_counts_what_the_walk_counts() {
    let mut windows = 0;
    for name in PROGRAMS {
        let (program, _) = run(name);
        for order in [MemwAlignedOrder::Limbs, MemwAlignedOrder::Lt] {
            for (window, batch) in [(1, 2048), (7, 3), (33, 2048), (4096, 5), (4096, 2048)] {
                windows += light_is_the_walks(name, &program, &[], window, batch, order);
            }
        }
    }
    for name in RUST_GUESTS {
        let program = rust_elf(name);
        for (window, batch) in [(97, 2048), (4096, 7)] {
            windows +=
                light_is_the_walks(name, &program, &[], window, batch, MemwAlignedOrder::Limbs);
        }
    }
    let ethrex = rust_elf("ethrex");
    let fixtures = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../executor/tests");
    let input = std::fs::read(fixtures.join("ethrex_10_transfers.bin")).expect("the fixture");
    windows += light_is_the_walks(
        "ethrex_10_transfers",
        &ethrex,
        &input,
        1 << 16,
        2048,
        MemwAlignedOrder::Limbs,
    );
    assert!(windows > 1000, "{windows} windows compared");
}

/// A room rule that admits every chunk.
fn always() -> RecordsRoom {
    Arc::new(|_| true)
}

/// `chunk` with its records replaced by their host fill at eight bytes a
/// cell; a placeholder (no host fill) as it is.
fn unheld(mut chunk: StreamedChunk) -> StreamedChunk {
    if chunk
        .trace
        .records_main()
        .is_none_or(|records| records.host_fill().is_none())
    {
        return chunk;
    }
    let records = chunk.trace.take_records_main().expect("held as records");
    let packed = records.host_fill().expect("a host fill");
    let (rows, cols) = (records.rows(), records.cols());
    let columns = packed
        .widen::<GoldilocksField>()
        .expect("the packed columns widen");
    let mut data: Vec<FieldElement<GoldilocksField>> = Vec::with_capacity(rows * cols);
    for r in 0..rows {
        data.extend(columns.iter().map(|column| column.evals()[r]));
    }
    chunk.trace = Table::new_main(data, cols, 1);
    chunk
}

/// The run of `program` by windows of `window` cycles under the in-row
/// register format (MEMW_A in `order`), walked from the fast engine's
/// history on `walkers` threads, its CPU chunks held as their logs, the
/// walkers counting where made and keeping no CPU list, with the floor arm
/// (`floor`) or without: its traces (every chunk filled on the host, a
/// placeholder left as it is), the chunks in hand-out order, and the arm's
/// stamps.
fn built(
    program: &Elf,
    max_rows: &MaxRowsConfig,
    window: usize,
    walkers: usize,
    floor: Option<FloorRoute>,
    defer_ecrec: bool,
    order: MemwAlignedOrder,
) -> (
    Traces,
    Vec<(StreamTable, usize)>,
    Option<super::FloorStamps>,
) {
    let mut builder = WindowedTraceBuilder::new(program, &[], max_rows)
        .and_then(|b| b.register_format(RegisterFormat::InRow))
        .and_then(|b| b.memw_aligned_order(order))
        .and_then(WindowedTraceBuilder::drop_streamed_ops)
        .and_then(|b| b.card_fill_streamed(always(), true, false))
        .map(|b| b.defer_ecrec_witnesses(defer_ecrec))
        .expect("the builder")
        .count_where_made(true)
        .drop_cpu_list(true);
    if let Some(route) = floor {
        builder = builder.history_floor(route).expect("the floor arm");
    }
    let (tx, rx) = std::sync::mpsc::sync_channel::<ExecWindow>(2);
    let mut jobs: Vec<ChunkJob> = Vec::new();
    let report = std::thread::scope(|scope| {
        scope.spawn(move || {
            let mut fast = FastExecutor::new(program, Vec::new()).expect("the fast engine");
            fast.enable_history().expect("history mode");
            while let Some((logs, history)) = fast.resume_history(window).expect("the run") {
                let last = (fast.pc() == 0)
                    .then(|| fast.take_history_final().map(Box::new))
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
    let stamps = builder.floor_stamps();
    let order: Vec<(StreamTable, usize)> = jobs.iter().map(|job| (job.table, job.index)).collect();
    let chunks: Vec<StreamedChunk> = jobs.into_iter().map(|job| unheld(job.generate())).collect();
    let mut traces = builder.finish(&report.last).expect("the finish");
    traces
        .insert_streamed(chunks)
        .expect("every chunk has a placeholder");
    (traces, order, stamps)
}

/// The first `cols` columns of each row of `t`.
fn rows_to(t: &Table, cols: usize) -> Vec<Vec<FieldElement<GoldilocksField>>> {
    (0..t.main_table.height)
        .map(|r| t.main_table.get_row(r)[..cols].to_vec())
        .collect()
}

/// Equal tables, row for row (their first `cols` columns, or all).
fn same(what: &str, a: &Table, b: &Table, cols: Option<usize>) {
    assert_eq!(
        (a.main_table.width, a.main_table.height),
        (b.main_table.width, b.main_table.height),
        "{what}: shape"
    );
    let cols = cols.unwrap_or(a.main_table.width);
    assert!(rows_to(a, cols) == rows_to(b, cols), "{what}: rows differ");
}

fn same_list(what: &str, a: &[Table], b: &[Table]) {
    assert_eq!(a.len(), b.len(), "{what}: table count");
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        same(&format!("{what}[{i}]"), x, y, None);
    }
}

/// ★ A run built with the floor arm, routed either way, on one walker and on
/// three, hands its chunks out in today's order and builds every table but
/// the families' as today's build does: CPU's columns but the side records',
/// and BITWISE's and DECODE's counts and LT's rows, which take the families'
/// lookups and MEMW_A's LT pairs from the light pass. Every program with
/// MEMW_A's limbs form (the block's), the ones with loads and stores with its
/// LT form too, whose rows send LT pairs.
#[test]
fn the_floor_arm_builds_every_table_but_the_families() {
    let max_rows = MaxRowsConfig::small();
    // The placeholders each family got over every build, and the CPU chunks.
    let (mut placeholders, mut cpu_chunks) = ([0usize; 3], 0usize);
    let lt_form = [
        "all_loadstore_32",
        "misalign_ld",
        "misalign_sd",
        "test_keccak_multi",
        "test_copy_ecalls",
    ];
    let runs = PROGRAMS
        .into_iter()
        .map(|name| (name, MemwAlignedOrder::Limbs))
        .chain(lt_form.into_iter().map(|name| (name, MemwAlignedOrder::Lt)));
    for (name, form) in runs {
        let (program, logs) = run(name);
        // ECREC's witnesses computed in the walk, and (on its program)
        // deferred to the pool in both builds.
        let defers: &[bool] = if name == "test_ecrec" {
            &[false, true]
        } else {
            &[false]
        };
        for (window, &defer) in [7, 33, 4096]
            .into_iter()
            .flat_map(|w| defers.iter().map(move |d| (w, d)))
        {
            let (want, want_order, none) = built(&program, &max_rows, window, 2, None, defer, form);
            assert!(none.is_none());
            for (route, walkers) in [
                (FloorRoute::Accumulator, 1),
                (FloorRoute::Accumulator, 3),
                (FloorRoute::Walkers, 1),
                (FloorRoute::Walkers, 3),
            ] {
                let what = format!("{name}/{form:?}/{window}/{route:?}/{walkers}/defer {defer}");
                let (got, order, stamps) = built(
                    &program,
                    &max_rows,
                    window,
                    walkers,
                    Some(route),
                    defer,
                    form,
                );
                assert_eq!(order, want_order, "{what}: the hand-out");
                let stamps = stamps.expect("the arm's stamps");
                for (f, table) in FAMILIES.iter().enumerate() {
                    let handed = want_order.iter().filter(|(t, _)| t == table).count();
                    assert_eq!(
                        stamps.placeholders[f], handed,
                        "{what}: {table:?} placeholders"
                    );
                    placeholders[f] += handed;
                }
                cpu_chunks += order.iter().filter(|(t, _)| *t == StreamTable::Cpu).count();
                assert_eq!(
                    stamps.windows + 1,
                    logs.len().div_ceil(window).max(1),
                    "{what}: windows"
                );
                assert_eq!(want.cpus.len(), got.cpus.len(), "{what}: CPU tables");
                for (i, (a, b)) in want.cpus.iter().zip(&got.cpus).enumerate() {
                    same(
                        &format!("{what} CPU[{i}]"),
                        a,
                        b,
                        Some(cpu::cols::NUM_COLUMNS),
                    );
                }
                same_list(&format!("{what} MEMW"), &want.memws, &got.memws);
                same_list(&format!("{what} STORE"), &want.stores, &got.stores);
                same_list(&format!("{what} SHIFT"), &want.shifts, &got.shifts);
                same_list(&format!("{what} CPU32"), &want.cpu32s, &got.cpu32s);
                same_list(&format!("{what} COPY"), &want.copies, &got.copies);
                same_list(&format!("{what} KECCAK"), &want.keccaks, &got.keccaks);
                same_list(
                    &format!("{what} KECCAK_RND"),
                    &want.keccak_rnds,
                    &got.keccak_rnds,
                );
                same_list(&format!("{what} ECSM"), &want.ecsms, &got.ecsms);
                same_list(&format!("{what} ECDAS"), &want.ecdases, &got.ecdases);
                same_list(&format!("{what} HINT"), &want.hints, &got.hints);
                same_list(&format!("{what} PAGE"), &want.pages, &got.pages);
                same_list(&format!("{what} LT"), &want.lts, &got.lts);
                same_list(&format!("{what} MUL"), &want.muls, &got.muls);
                same_list(&format!("{what} DVRM"), &want.dvrms, &got.dvrms);
                same_list(&format!("{what} BRANCH"), &want.branches, &got.branches);
                same_list(&format!("{what} EQ"), &want.eqs, &got.eqs);
                same_list(&format!("{what} BYTEWISE"), &want.bytewises, &got.bytewises);
                same(
                    &format!("{what} BITWISE"),
                    &want.bitwise,
                    &got.bitwise,
                    None,
                );
                same(&format!("{what} DECODE"), &want.decode, &got.decode, None);
                same(
                    &format!("{what} REGISTER"),
                    &want.register,
                    &got.register,
                    None,
                );
                same(&format!("{what} HALT"), &want.halt, &got.halt, None);
                same(&format!("{what} BLAKE3"), &want.blake3, &got.blake3, None);
                // The families: the full chunks placeholders (no host
                // columns); then the finish's tables, which lack the rows
                // the walked windows left past the last full chunk (fewer
                // than a chunk's): at most one table fewer.
                for (f, (family, a, b)) in [
                    ("residual", &want.memw_registers, &got.memw_registers),
                    ("MEMW_A", &want.memw_aligneds, &got.memw_aligneds),
                    ("LOAD", &want.loads, &got.loads),
                ]
                .into_iter()
                .enumerate()
                {
                    let chunks = stamps.placeholders[f];
                    for (k, table) in b.iter().take(chunks).enumerate() {
                        let records = table.records_main().unwrap_or_else(|| {
                            panic!("{what}: {family}[{k}] is not a placeholder")
                        });
                        assert!(records.host_fill().is_none(), "{what}: {family}[{k}]");
                    }
                    assert!(
                        b.len() >= chunks && b.len() <= a.len() && a.len() <= b.len() + 1,
                        "{what}: {family}: {} tables today, {} with the arm, {chunks} chunks",
                        a.len(),
                        b.len()
                    );
                }
            }
        }
    }
    // The comparison covers each family's placeholders and CPU's zero side
    // records, not only runs too short for a chunk.
    assert!(
        placeholders.iter().all(|&n| n > 0) && cpu_chunks > 0,
        "placeholders {placeholders:?}, CPU chunks {cpu_chunks}"
    );
}

/// ★ The arm refuses what it cannot build: the register format before the
/// in-row one, no dropped streamed ops, no CPU chunks held as their logs, no
/// counting where made, chunk heights that are not whole tables, a re-walk;
/// and its placeholders are their family's shape with no host columns.
#[test]
fn the_floor_arm_refuses_what_it_cannot_build() {
    let (program, _) = run("all_loadstore_32");
    let max_rows = MaxRowsConfig::small();
    let route = FloorRoute::Walkers;
    let ready = |max_rows: &MaxRowsConfig, format: RegisterFormat| {
        WindowedTraceBuilder::new(&program, &[], max_rows)
            .and_then(|b| b.register_format(format))
            .and_then(WindowedTraceBuilder::drop_streamed_ops)
            .and_then(|b| b.card_fill_streamed(always(), true, false))
            .map(|b| b.count_where_made(true).drop_cpu_list(true))
    };
    assert!(
        ready(&max_rows, RegisterFormat::InRow)
            .and_then(|b| b.history_floor(route))
            .is_ok(),
        "the arm's build"
    );
    assert!(
        ready(&max_rows, RegisterFormat::Tokens32)
            .and_then(|b| b.history_floor(route))
            .is_err(),
        "the register format before"
    );
    assert!(
        WindowedTraceBuilder::new(&program, &[], &max_rows)
            .and_then(|b| b.register_format(RegisterFormat::InRow))
            .and_then(|b| b.history_floor(route))
            .is_err(),
        "the streamed ops kept"
    );
    assert!(
        WindowedTraceBuilder::new(&program, &[], &max_rows)
            .and_then(|b| b.register_format(RegisterFormat::InRow))
            .and_then(WindowedTraceBuilder::drop_streamed_ops)
            .and_then(|b| b.history_floor(route))
            .is_err(),
        "no CPU logs"
    );
    assert!(
        ready(&max_rows, RegisterFormat::InRow)
            .map(|b| b.count_where_made(false))
            .and_then(|b| b.history_floor(route))
            .is_err(),
        "not counted where made"
    );
    let odd = MaxRowsConfig {
        load: 24,
        ..MaxRowsConfig::small()
    };
    assert!(
        ready(&odd, RegisterFormat::InRow)
            .and_then(|b| b.history_floor(route))
            .is_err(),
        "a LOAD chunk of 24 rows"
    );
    // With a re-walk: the walk refuses the arm.
    let mut builder = ready(&max_rows, RegisterFormat::InRow)
        .and_then(|b| b.history_floor(route))
        .expect("the arm");
    let (tx, rx) = std::sync::mpsc::sync_channel::<ExecWindow>(2);
    drop(tx);
    let walked = walk_windows_from_history(&mut builder, rx, 2, 3, |_, _, _| Ok(()));
    assert!(walked.is_err(), "a re-walk");
    // A placeholder: its family's width, its height, no host fill.
    for table in FAMILIES {
        let p = Placeholder { table, rows: 32 };
        let cols = p.columns();
        let trace = Box::new(p).into_trace();
        assert_eq!(
            (trace.main_table.width, trace.main_table.height),
            (cols, 32),
            "{table:?}"
        );
        let records = trace.records_main().expect("held as records");
        assert!(records.host_fill().is_none(), "{table:?}: no host fill");
        assert_eq!(records.bytes(), 0, "{table:?}: no records");
    }
}
