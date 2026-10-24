//! ★ The finish's plan ([`super::finish`]) builds the tables the frozen table
//! phase (`finish_oracle`) built, word for word, on every path that reaches
//! it: whole-run builds, non-final epochs with and without the L2G bookend,
//! and windowed builds (streamed ops dropped, KECCAK_RND chunked or cut at the
//! finish, KECCAK / ECSM / ECDAS cut, tables packed or written packed, the
//! derived LT ops raw or compact, LT concatenated or kept as segments). Its
//! header is the oracle's counts and pages before any table exists, and a
//! header that miscounts is refused. With the cut tables stopped at the last
//! real row ([`CutPadding::LastTable`]) the plan builds the oracle's tables
//! without the ones of padding rows alone, and every other table as it is.

use executor::elf::Elf;
use executor::vm::execution::Executor;
use executor::vm::logs::Log;

use super::cut_padding_tests::is_today_without_its_padding_tables;
use super::finish::{FinishPlan, counts_of};
use super::*;
use crate::tables::MaxRowsConfig;
use crate::test_utils::asm_elf_bytes;

type Table = TraceTable<GoldilocksField, GoldilocksExtension>;

pub(super) fn run(name: &str) -> (Elf, Vec<Log>) {
    let program = Elf::load(&asm_elf_bytes(name)).expect("the ELF loads");
    let logs = Executor::new(&program, Vec::new())
        .expect("the executor starts")
        .run()
        .expect("the program runs")
        .logs;
    (program, logs)
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

/// Every table of a build, named, and what the statement reads off it.
fn contents(t: &Traces) -> Vec<(String, String)> {
    let lists: [(&str, &[Table]); 21] = [
        ("CPU", &t.cpus),
        ("MEMW_R", &t.memw_registers),
        ("MEMW_A", &t.memw_aligneds),
        ("MEMW", &t.memws),
        ("LOAD", &t.loads),
        ("STORE", &t.stores),
        ("SHIFT", &t.shifts),
        ("CPU32", &t.cpu32s),
        ("COPY", &t.copies),
        ("KECCAK", &t.keccaks),
        ("KECCAK_RND", &t.keccak_rnds),
        ("ECSM", &t.ecsms),
        ("ECDAS", &t.ecdases),
        ("HINT", &t.hints),
        ("PAGE", &t.pages),
        ("LT", &t.lts),
        ("MUL", &t.muls),
        ("DVRM", &t.dvrms),
        ("BRANCH", &t.branches),
        ("EQ", &t.eqs),
        ("BYTEWISE", &t.bytewises),
    ];
    let mut out = Vec::new();
    for (name, list) in lists {
        for (i, table) in list.iter().enumerate() {
            let (cols, rows, words) = shape_and_words(table);
            out.push((
                format!("{name}[{i}]"),
                format!("{cols}x{rows} {:x}", fold(&words)),
            ));
        }
        out.push((format!("{name} count"), list.len().to_string()));
    }
    for (name, table) in [
        ("BITWISE", &t.bitwise),
        ("DECODE", &t.decode),
        ("REGISTER", &t.register),
        ("HALT", &t.halt),
        ("KECCAK_RC", &t.keccak_rc),
        ("BLAKE3", &t.blake3),
        ("L2G", &t.local_to_global),
    ] {
        let (cols, rows, words) = shape_and_words(table);
        out.push((
            name.to_string(),
            format!("{cols}x{rows} {:x}", fold(&words)),
        ));
    }
    out.push((
        "public output".into(),
        format!("{:?}", t.public_output_bytes),
    ));
    out.push(("blake3 ops".into(), t.num_blake3_ops.to_string()));
    out.push(("counts".into(), format!("{:?}", t.table_counts())));
    out.push((
        "runtime pages".into(),
        format!("{:?}", t.runtime_page_ranges()),
    ));
    out.push((
        "page configs".into(),
        format!(
            "{:?}",
            t.page_configs
                .iter()
                .map(|c| (c.page_base, c.init_values.is_some(), c.is_private_input))
                .collect::<Vec<_>>()
        ),
    ));
    out.push((
        "touched cells".into(),
        format!("{:?}", t.touched_memory_cells),
    ));
    out
}

/// An order-sensitive digest of a table's words (a mismatch names the table).
fn fold(words: &[u64]) -> u64 {
    words.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &w| {
        (h ^ w).wrapping_mul(0x0000_0100_0000_01b3).rotate_left(5)
    })
}

/// `build` run as the oracle, which must have run.
fn oracle<T>(build: impl FnOnce() -> T) -> T {
    let before = finish_oracle::runs();
    let traces = finish_oracle::with(build);
    assert!(finish_oracle::runs() > before, "the oracle did not run");
    traces
}

fn same(what: &str, oracle: &Traces, plan: &Traces) {
    let (a, b) = (contents(oracle), contents(plan));
    assert_eq!(a.len(), b.len(), "{what}: the builds hold different tables");
    for ((name, x), (_, y)) in a.iter().zip(&b) {
        assert_eq!(x, y, "{what}: {name}");
    }
}

/// A windowed build over windows of `window` cycles, `configure` applied, its
/// streamed chunks put back.
pub(super) fn windowed(
    program: &Elf,
    logs: &[Log],
    max_rows: &MaxRowsConfig,
    window: usize,
    configure: &dyn Fn(WindowedTraceBuilder<'_>) -> WindowedTraceBuilder<'_>,
) -> Traces {
    let mut builder =
        configure(WindowedTraceBuilder::new(program, &[], max_rows).expect("the builder"));
    let body = logs.len() - 1;
    let cut = body - body % window;
    let mut chunks = Vec::new();
    for w in logs[..cut].chunks(window) {
        chunks.extend(builder.push(w).expect("a window"));
    }
    let mut traces = builder.finish(&logs[cut..]).expect("the last window");
    traces
        .insert_streamed(chunks)
        .expect("every chunk has a placeholder");
    traces
}

const PROGRAMS: [&str; 13] = [
    "sub",
    "all_instructions_64",
    "all_loadstore_32",
    "misalign_sd",
    "test_keccak",
    "test_keccak_multi",
    "test_ecsm",
    "test_ecsm_multi",
    "test_ecsm_split",
    "test_commit_split",
    "test_blake3",
    "test_blake3_absorb",
    "test_dense_pages",
];

/// Whole-run builds, at small chunks (every chunked table split) and at the
/// production sizes: the plan's tables are the oracle's.
#[test]
fn the_plan_builds_the_oracles_whole_run_tables() {
    for max_rows in [
        MaxRowsConfig::small(),
        MaxRowsConfig::uniform(4),
        MaxRowsConfig::default(),
    ] {
        for name in PROGRAMS {
            let (program, logs) = run(name);
            let build = || {
                Traces::from_elf_and_logs(
                    &program,
                    &logs,
                    &max_rows,
                    &[],
                    #[cfg(feature = "disk-spill")]
                    StorageMode::Ram,
                )
                .expect("the build")
            };
            let frozen = oracle(build);
            same(&format!("{name} whole run"), &frozen, &build());
        }
    }
}

/// A run cut into two epochs, the first non-final, with and without the L2G
/// bookend (no PAGE, touched cells): the plan's tables are the oracle's, and
/// so is the final epoch's.
#[test]
fn the_plan_builds_the_oracles_epoch_tables() {
    let max_rows = MaxRowsConfig::small();
    for name in [
        "all_instructions_64",
        "test_keccak_multi",
        "test_ecsm_multi",
    ] {
        let (program, logs) = run(name);
        let artifacts = DecodeArtifacts::from_elf(&program).expect("the artifacts");
        let image = build_initial_image(&program, &[]);
        let register_init = register::register_init_from_entry_point(program.entry_point);
        let half = logs.len() / 2;
        for l2g in [false, true] {
            for (part, is_final) in [(&logs[..half], false), (&logs[..], true)] {
                let build = || {
                    Traces::from_image_and_logs_with_decode(
                        &artifacts,
                        &image,
                        &register_init,
                        part,
                        &max_rows,
                        &[],
                        is_final,
                        l2g,
                        #[cfg(feature = "disk-spill")]
                        StorageMode::Ram,
                    )
                    .expect("the build")
                };
                let frozen = oracle(build);
                same(
                    &format!("{name} epoch final {is_final} l2g {l2g}"),
                    &frozen,
                    &build(),
                );
            }
        }
    }
}

/// Windowed builds in every configuration the block uses (and their A arms):
/// the plan's tables are the oracle's, packed or not. The configurations that
/// cut pad the whole run, as the oracle does whatever the settings say (cut at
/// the last real row:
/// [`a_build_cut_at_the_last_real_row_is_the_oracles_without_its_padding_tables`]).
#[test]
fn the_plan_builds_the_oracles_windowed_tables() {
    type Configure = Box<dyn Fn(WindowedTraceBuilder<'_>) -> WindowedTraceBuilder<'_>>;
    let configs: Vec<(&str, Configure)> = vec![
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
                    .cut_padding(CutPadding::WholeRun)
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
                    .cut_padding(CutPadding::WholeRun)
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
    ];
    for max_rows in [MaxRowsConfig::small(), MaxRowsConfig::uniform(4)] {
        for name in [
            "all_instructions_64",
            "misalign_sd",
            "test_keccak_multi",
            "test_ecsm_multi",
            "test_commit_split",
            "test_blake3_absorb",
        ] {
            let (program, logs) = run(name);
            for window in [7, 1000] {
                for (config, configure) in &configs {
                    let build = || windowed(&program, &logs, &max_rows, window, configure.as_ref());
                    let frozen = oracle(build);
                    same(
                        &format!("{name} window {window} {config}"),
                        &frozen,
                        &build(),
                    );
                }
            }
        }
    }
}

/// The delta-coded lists ([`super::delta`]) marked every three ops instead of
/// every 4096, so nearly every chunk and every p4 slice starts past an index
/// mark: the windowed builds make the same tables word for word.
#[test]
fn the_tables_are_the_same_with_an_index_mark_every_three_ops() {
    type Configure = Box<dyn Fn(WindowedTraceBuilder<'_>) -> WindowedTraceBuilder<'_>>;
    let configs: Vec<(&str, Configure)> = vec![
        ("plain", Box::new(|b| b)),
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
    ];
    for max_rows in [MaxRowsConfig::small(), MaxRowsConfig::uniform(4)] {
        for name in [
            "all_instructions_64",
            "misalign_sd",
            "test_keccak_multi",
            "test_ecsm_multi",
        ] {
            let (program, logs) = run(name);
            for window in [7, 1000] {
                for (config, configure) in &configs {
                    let build = || windowed(&program, &logs, &max_rows, window, configure.as_ref());
                    let dense = super::delta::with_index_every(3, build);
                    same(
                        &format!("{name} window {window} {config}, marks every 3"),
                        &build(),
                        &dense,
                    );
                }
            }
        }
    }
}

/// A plan over a collected run, as `build_from_collected` makes one.
fn plan_of<'a>(
    ops: CollectedOps,
    memory_state: &'a MemoryState,
    register_state: RegisterState,
    artifacts: &'a DecodeArtifacts,
    image: &'a HashMap<u64, u8>,
    register_init: &'a [u32],
    max_rows: &'a MaxRowsConfig,
) -> FinishPlan<'a> {
    FinishPlan::new(
        ops,
        Some(image),
        memory_state,
        register_init,
        artifacts.decode_trace.clone(),
        &artifacts.decode,
        register_state,
        max_rows,
        #[cfg(feature = "disk-spill")]
        StorageMode::Ram,
        &[],
        true,
        false,
        &StreamSkip::default(),
        None,
    )
    .expect("the plan")
}

/// The header is the oracle's counts, pages, public output and BLAKE3 count
/// before any table is built; a header that miscounts any one family by one
/// is refused when the tables are built, not proved.
#[test]
fn the_header_is_the_tables_counts_and_a_wrong_one_is_refused() {
    let max_rows = MaxRowsConfig::small();
    for name in [
        "all_instructions_64",
        "test_keccak_multi",
        "test_ecsm_multi",
        "test_blake3",
    ] {
        let (program, logs) = run(name);
        let artifacts = DecodeArtifacts::from_elf(&program).expect("the artifacts");
        let image = build_initial_image(&program, &[]);
        let register_init = register::register_init_from_entry_point(program.entry_point);
        let frozen = oracle(|| {
            Traces::from_elf_and_logs(
                &program,
                &logs,
                &max_rows,
                &[],
                #[cfg(feature = "disk-spill")]
                StorageMode::Ram,
            )
            .expect("the build")
        });
        let collect = || {
            Traces::collect_epoch(&artifacts, &image, &register_init, &logs, true)
                .expect("collected")
        };
        let CollectedEpoch {
            ops,
            memory_state,
            register_state,
        } = collect();
        let plan = plan_of(
            ops,
            &memory_state,
            register_state,
            &artifacts,
            &image,
            &register_init,
            &max_rows,
        );
        let header = plan.header();
        assert_eq!(
            counts_of(&header.table_counts),
            counts_of(&frozen.table_counts()),
            "{name}: counts"
        );
        assert_eq!(
            format!("{:?}", header.runtime_page_ranges()),
            format!("{:?}", frozen.runtime_page_ranges()),
            "{name}: runtime pages"
        );
        assert_eq!(
            header.page_configs.len(),
            frozen.page_configs.len(),
            "{name}: pages"
        );
        assert_eq!(
            header.public_output_bytes, frozen.public_output_bytes,
            "{name}"
        );
        assert_eq!(header.num_blake3_ops, frozen.num_blake3_ops, "{name}");
        same(
            &format!("{name} planned"),
            &frozen,
            &plan.emit_all().expect("the tables"),
        );
        // Each family miscounted by one, up or down (where it has a table).
        for field in 0..22 {
            for up in [true, false] {
                let CollectedEpoch {
                    ops,
                    memory_state,
                    register_state,
                } = collect();
                let plan = plan_of(
                    ops,
                    &memory_state,
                    register_state,
                    &artifacts,
                    &image,
                    &register_init,
                    &max_rows,
                );
                let mut wrong = plan.header().table_counts.clone();
                let slot: &mut usize = match field {
                    0 => &mut wrong.cpu,
                    1 => &mut wrong.lt,
                    2 => &mut wrong.memw,
                    3 => &mut wrong.memw_aligned,
                    4 => &mut wrong.load,
                    5 => &mut wrong.mul,
                    6 => &mut wrong.dvrm,
                    7 => &mut wrong.shift,
                    8 => &mut wrong.branch,
                    9 => &mut wrong.memw_register,
                    10 => &mut wrong.eq,
                    11 => &mut wrong.bytewise,
                    12 => &mut wrong.store,
                    13 => &mut wrong.cpu32,
                    14 => &mut wrong.keccak,
                    15 => &mut wrong.keccak_rnd,
                    16 => &mut wrong.ecsm,
                    17 => &mut wrong.ecdas,
                    18 => &mut wrong.hint,
                    19 => &mut wrong.copy,
                    20 => &mut wrong.blake3,
                    _ => &mut wrong.is_wh,
                };
                if !up && *slot == 0 {
                    continue;
                }
                *slot = if up { *slot + 1 } else { *slot - 1 };
                assert!(
                    matches!(plan.with_counts(wrong).emit_all(), Err(Error::Prover(_))),
                    "{name}: count {field} off by one ({up}) was not refused"
                );
            }
        }
    }
}

/// The four families a block cuts at a fixed number of rows: each one's
/// tables in a build and its bus interactions.
fn cut_families(t: &Traces) -> [(&'static str, &[Table], Vec<stark::lookup::BusInteraction>); 5] {
    [
        ("COPY", &t.copies, copy::bus_interactions()),
        ("KECCAK", &t.keccaks, keccak::bus_interactions()),
        ("KECCAK_RND", &t.keccak_rnds, keccak_rnd::bus_interactions()),
        ("ECSM", &t.ecsms, ecsm::bus_interactions()),
        ("ECDAS", &t.ecdases, ecdas::bus_interactions()),
    ]
}

/// [`contents`] without the five cut families' tables and counts.
fn contents_but_the_cut_families(t: &Traces) -> Vec<(String, String)> {
    let cut = ["COPY", "KECCAK", "KECCAK_RND", "ECSM", "ECDAS"];
    contents(t)
        .into_iter()
        .filter(|(name, _)| {
            name != "counts"
                && !cut.iter().any(|family| {
                    *name == format!("{family} count") || name.starts_with(&format!("{family}["))
                })
        })
        .collect()
}

/// The block's windowed build with every family cut: KECCAK_RND at 32 rows,
/// KECCAK, ECSM, ECDAS and COPY at `cuts`, wide (`packed` false) or packed, written
/// packed and with the streamed ops dropped, the cut tables padded as
/// `padding`.
fn cut_build(
    program: &Elf,
    logs: &[Log],
    window: usize,
    cuts: (usize, usize, usize, usize),
    packed: bool,
    padding: CutPadding,
) -> Traces {
    windowed(program, logs, &MaxRowsConfig::small(), window, &|b| {
        let b = b
            .keccak_rnd_chunks_at_finish(32)
            .expect("rows")
            .cuts_at_finish(cuts.0, cuts.1, cuts.2, cuts.3)
            .expect("rows")
            .cut_padding(padding);
        if packed {
            b.pack_finished_tables()
                .generate_packed()
                .drop_streamed_ops()
                .expect("before any window")
        } else {
            b
        }
    })
}

/// ★ Cut at the last real row, a windowed build is the frozen build's — which
/// pads the whole run, whatever the settings say — without the cut families'
/// tables of padding rows alone: each family's tables are the frozen build's
/// first ones word for word, every table left out has all its bus
/// multiplicities at zero but its padding byte checks, the last one kept holds
/// a real row, and every other table of the build (KECCAK_RC's among them) is
/// the frozen build's. BITWISE's counts are too, but one: KECCAK_RND's θ/ρ byte
/// checks have multiplicity 1 (main's multiplicity fix), so each padding row
/// sends them as `ARE_BYTES[0, 0]`, and the frozen build's BITWISE answers the
/// left-out tables' rows as well — `ARE_BYTES_PER_ROW` a row, in that one cell.
#[test]
fn a_build_cut_at_the_last_real_row_is_the_oracles_without_its_padding_tables() {
    let mut left_out = [0usize; 5];
    for name in [
        "all_instructions_64",
        "test_keccak",
        "test_keccak_multi",
        "test_ecsm",
        "test_ecsm_multi",
        "test_ecsm_split",
        "test_copy_ecalls",
    ] {
        let (program, logs) = run(name);
        for cuts in [(1, 1, 1, 1), (2, 2, 16, 4), (2, 4, 2, 2)] {
            for (window, packed) in [(7, true), (1000, true), (1000, false)] {
                let what = format!("{name} cuts {cuts:?} window {window} packed {packed}");
                let build =
                    || cut_build(&program, &logs, window, cuts, packed, CutPadding::LastTable);
                let frozen = oracle(build);
                let last = build();
                let mut keccak_rnd_rows_left_out = 0usize;
                for (f, ((family, kept, interactions), (_, today, _))) in cut_families(&last)
                    .into_iter()
                    .zip(cut_families(&frozen))
                    .enumerate()
                {
                    is_today_without_its_padding_tables(kept, today, &interactions)
                        .unwrap_or_else(|e| panic!("{what}: {family}: {e}"));
                    left_out[f] += today.len() - kept.len();
                    if family == "KECCAK_RND" {
                        keccak_rnd_rows_left_out += today[kept.len()..]
                            .iter()
                            .map(|t| t.num_rows())
                            .sum::<usize>();
                    }
                }
                let (a, b) = (
                    contents_but_the_cut_families(&frozen),
                    contents_but_the_cut_families(&last),
                );
                assert_eq!(a.len(), b.len(), "{what}: the other tables");
                for ((table, x), (_, y)) in a.iter().zip(&b) {
                    if table != "BITWISE" {
                        assert_eq!(x, y, "{what}: {table}");
                    }
                }
                let ((cols, rows, frozen_words), (_, _, words)) = (
                    shape_and_words(&frozen.bitwise),
                    shape_and_words(&last.bitwise),
                );
                assert_eq!(
                    (rows, frozen_words.len()),
                    (rows, words.len()),
                    "{what}: BITWISE"
                );
                let cell = bitwise::row_index(0, 0, 0) * cols + bitwise::cols::MU_ARE_BYTES;
                for (i, (x, y)) in frozen_words.iter().zip(&words).enumerate() {
                    if i == cell {
                        assert_eq!(
                            *x - *y,
                            (keccak_rnd::ARE_BYTES_PER_ROW * keccak_rnd_rows_left_out) as u64,
                            "{what}: BITWISE's ARE_BYTES[0, 0]"
                        );
                    } else {
                        assert_eq!(x, y, "{what}: BITWISE word {i}");
                    }
                }
            }
        }
    }
    assert!(
        left_out.iter().all(|&n| n > 0),
        "each cut family left a table of padding rows out somewhere: {left_out:?}"
    );
}

/// The cut families' tables of a build as (family, tables, rows of each).
fn cut_shapes(t: &Traces) -> Vec<(&'static str, usize, Vec<usize>)> {
    cut_families(t)
        .into_iter()
        .map(|(family, tables, _)| {
            (
                family,
                tables.len(),
                tables
                    .iter()
                    .map(|table| shape_and_words(table).1)
                    .collect(),
            )
        })
        .collect()
}

/// ★ The default stops at the last real row, and the knob at `run` builds
/// exactly the tables the block always built before, on two pinned programs:
/// the type's default and what an unset variable reads as keep the tables
/// with a real row; `run` is the whole run padded, table for table as listed
/// here and word for word the frozen build's. test_keccak_multi is 3
/// permutations: 3 KECCAK rows (4 padded) and 72 KECCAK_RND rows (128
/// padded), and its 200-byte `write` is 25 COPY rows (32 padded).
/// test_ecsm_multi is 3 scalar multiplications: 3 ECSM rows (4 padded) and 42
/// ECDAS steps (64 padded), and its three 32-byte `write`s are 12 COPY rows
/// (16 padded).
#[test]
fn the_knob_at_run_builds_the_whole_runs_tables_on_pinned_programs() {
    let unset = CutPadding::parse(None).expect("an unset variable");
    let run_word = CutPadding::parse(Some("run")).expect("the word run");
    let keccak_last = [
        ("COPY", 13, 2),
        ("KECCAK", 2, 2),
        ("KECCAK_RND", 3, 32),
        ("ECSM", 0, 0),
        ("ECDAS", 0, 0),
    ];
    let ecsm_last = [
        ("COPY", 6, 2),
        ("KECCAK", 0, 0),
        ("KECCAK_RND", 0, 0),
        ("ECSM", 2, 2),
        ("ECDAS", 3, 16),
    ];
    for (name, padding, want) in [
        ("test_keccak_multi", CutPadding::default(), keccak_last),
        ("test_keccak_multi", unset, keccak_last),
        ("test_ecsm_multi", CutPadding::default(), ecsm_last),
        ("test_ecsm_multi", unset, ecsm_last),
        (
            "test_keccak_multi",
            run_word,
            [
                ("COPY", 16, 2),
                ("KECCAK", 2, 2),
                ("KECCAK_RND", 4, 32),
                ("ECSM", 0, 0),
                ("ECDAS", 0, 0),
            ],
        ),
        (
            "test_ecsm_multi",
            run_word,
            [
                ("COPY", 8, 2),
                ("KECCAK", 0, 0),
                ("KECCAK_RND", 0, 0),
                ("ECSM", 2, 2),
                ("ECDAS", 4, 16),
            ],
        ),
    ] {
        let (program, logs) = run(name);
        let build = || cut_build(&program, &logs, 1000, (2, 2, 16, 2), true, padding);
        let built = build();
        let want: Vec<(&str, usize, Vec<usize>)> = want
            .into_iter()
            .map(|(family, tables, rows)| (family, tables, vec![rows; tables]))
            .collect();
        assert_eq!(cut_shapes(&built), want, "{name} {padding:?}");
        if padding == CutPadding::WholeRun {
            same(&format!("{name} knob at run"), &oracle(build), &built);
        }
    }
}

/// The readout's line names the rule and, per cut family, the tables built
/// and how many of them hold a real row.
#[test]
fn the_cut_line_names_the_rule_and_the_tables_with_a_real_row() {
    let (program, logs) = run("test_keccak_multi");
    let artifacts = DecodeArtifacts::from_elf(&program).expect("the artifacts");
    let image = build_initial_image(&program, &[]);
    let register_init = register::register_init_from_entry_point(program.entry_point);
    let max_rows = MaxRowsConfig::small();
    let line = |skip: StreamSkip| {
        let CollectedEpoch {
            ops,
            memory_state,
            register_state,
        } = Traces::collect_epoch(&artifacts, &image, &register_init, &logs, true)
            .expect("collected");
        FinishPlan::new(
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
            &skip,
            None,
        )
        .expect("the plan")
        .cut_line()
        .map(str::to_string)
    };
    let cut = |cut_padding| StreamSkip {
        keccak_rnd_rows: 32,
        keccak_rows: 1,
        ecsm_rows: 1,
        ecdas_rows: 1,
        cut_padding,
        ..StreamSkip::default()
    };
    assert_eq!(
        line(cut(CutPadding::WholeRun)).as_deref(),
        Some(
            "BLOCK CUT PADDING: the whole run padded, then cut · tables (of them with a real \
             row): KECCAK 4 (3) · KECCAK_RND 4 (3) · ECSM 0 (0) · ECDAS 0 (0)"
        )
    );
    assert_eq!(
        line(cut(CutPadding::LastTable)).as_deref(),
        Some(
            "BLOCK CUT PADDING: cut, the last table padded (the default) · tables (of them \
             with a real row): KECCAK 3 (3) · KECCAK_RND 3 (3) · ECSM 0 (0) · ECDAS 0 (0)"
        )
    );
    // A build that cuts nothing has no line.
    assert_eq!(line(StreamSkip::default()), None);
}
