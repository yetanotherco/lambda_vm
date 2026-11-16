//! ★ The walk under the word memory format (`super::word_walk`), on the
//! windowed builder's traces, with no proof: every inner bus of a chip-free
//! run's build balances under the verifier's AIR set for the format (BITWISE's
//! multiplicities as the build counted them), every word table's and PAGE_W's
//! constraints read zero, every CPU load and store is exactly one word table's
//! row, and the format refuses a run that makes any syscall but the halt and
//! the COPY chip's. Runs that call the COPY chip (memcpy and memmove, memset,
//! the commit syscall) build COPY_W and MEMW_W tables that balance the same
//! way, the commit bus against the public output as the verifier rebuilds it.

use std::collections::{BTreeMap, HashMap};

use executor::elf::Elf;
use executor::vm::logs::Log;
use stark::constraints::builder::ConstraintSet;
use stark::proof::options::ProofOptions;
use stark::trace::TraceTable;

use super::Traces;
use super::bus_net_tests::multiplicity;
use super::finish_tests::{run, windowed};
use super::in_row_forgery_tests::with_verifier_files;
use crate::tables::MaxRowsConfig;
use crate::tables::fused_memory::MemoryFormat;
use crate::tables::load::cols as ld;
use crate::tables::memw_aligned::MemwAlignedOrder;
use crate::tables::register::RegisterFormat;
use crate::tables::store::cols as st;
use crate::tables::types::{BusId, FE, GoldilocksExtension, GoldilocksField};
use crate::tables::word_cpu::{WordClass, word_class};
use crate::tables::{
    copy_w, mem_load2w, mem_load8, mem_loadw, mem_store2w, mem_store8, mem_storew, memw_w, page_w,
};

type Table = TraceTable<GoldilocksField, GoldilocksExtension>;

/// The field's order: `−k` reads as `P − k`.
const P: u64 = 0xFFFF_FFFF_0000_0001;

/// The chip-free programs: every width, sign and alignment of the CPU's loads
/// and stores (`fused_mem_cases`), every instruction, the misaligned ones.
const PROGRAMS: [&str; 13] = [
    "fused_mem_cases",
    "fused_mem_one_sb",
    "all_instructions_64",
    "all_loadstore_32",
    "loop_4096",
    "misalign_ld",
    "misalign_lh",
    "misalign_lhu",
    "misalign_lw",
    "misalign_lwu",
    "misalign_sd",
    "misalign_sh",
    "misalign_sw",
];

/// The asm and Rust programs that call the COPY chip: memcpy and memmove,
/// memset, the commit syscall.
const COPY_ASM: [&str; 4] = [
    "test_copy_ecalls",
    "test_copy_overlaps",
    "test_commit_4",
    "test_commit_split",
];
const COPY_RUST: [&str; 7] = [
    "dma_memcpy_cases",
    "dma_memmove_cases",
    "dma_memset_cases",
    "dma_memcpy_min",
    "dma_memset_min",
    "commit",
    "pure_commit",
];

/// A Rust guest's run, from `program_artifacts/rust`.
fn run_rust(name: &str) -> (Elf, Vec<Log>) {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../executor/program_artifacts/rust")
        .join(format!("{name}.elf"));
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let program = Elf::load(&bytes).expect("the ELF loads");
    let logs = crate::Executor::new(&program, Vec::new())
        .expect("the executor starts")
        .run()
        .expect("the program runs")
        .logs;
    (program, logs)
}

/// The buses a block's statement closes from outside the tables (the commit
/// bus too, which [`left`] closes with the public output as the verifier
/// does).
const OUTSIDE: [BusId; 2] = [BusId::GlobalMemory, BusId::Decode];

fn get(t: &Table, row: usize, col: usize) -> u64 {
    t.get_main(row, col).canonical_u64()
}

/// `program`'s run built by windows of `window` cycles under the word memory
/// format, the registers in `registers`, the small chunks.
fn word_build(program: &Elf, logs: &[Log], registers: RegisterFormat, window: usize) -> Traces {
    windowed(program, logs, &MaxRowsConfig::small(), window, &|b| {
        b.register_format(registers)
            .expect("before any window")
            .memw_aligned_order(MemwAlignedOrder::Limbs)
            .expect("before any window")
            .memory_format(MemoryFormat::Words)
            .expect("before any window")
    })
}

/// The same run under the split format (the accesses' reference).
fn split_build(program: &Elf, logs: &[Log]) -> Traces {
    windowed(program, logs, &MaxRowsConfig::small(), 1000, &|b| {
        b.register_format(RegisterFormat::InRow)
            .expect("before any window")
            .memw_aligned_order(MemwAlignedOrder::Limbs)
            .expect("before any window")
    })
}

/// What a build's nets leave over: every bus but the statement's with a
/// tuple left over, its tuples and their signed nets (BITWISE's multiplicities
/// read as the build counted them, so a range check left over shows here).
fn left(
    program: &Elf,
    build: &mut Traces,
    registers: RegisterFormat,
) -> BTreeMap<u64, Vec<(Vec<u64>, i128)>> {
    if registers == RegisterFormat::InRow {
        with_verifier_files(program, build);
    }
    let options = ProofOptions::default_test_options();
    let heights = crate::cpu_heights_from_traces(build);
    let inputs = crate::BlockAirInputs::new(&options, program, heights, &build.public_output_bytes)
        .expect("the block's inputs");
    let format = crate::block_whir::BlockFormat::production().with_memory(MemoryFormat::Words);
    let mut airs = crate::block_airs(
        program,
        &options,
        &build.page_configs,
        &build.table_counts(),
        &inputs,
        format.keccak_rnd,
        format.memw_aligned,
        registers,
    );
    airs.with_fused_memory(&options, build.fused_counts());
    airs.with_word_memory(&options, build.word_counts(), &build.page_configs)
        .expect("the word format's set");
    let outside: Vec<u64> = OUTSIDE.iter().map(|b| u64::from(*b)).collect();
    let mut net: HashMap<(u64, Vec<u64>), FE> = HashMap::new();
    for (air, trace, _) in airs.air_trace_pairs(build) {
        let interactions = air.bus_interactions();
        for row in 0..trace.num_rows() {
            let g = |c: usize| *trace.get_main(row, c);
            for bi in interactions {
                if outside.contains(&bi.bus_id) {
                    continue;
                }
                let m = multiplicity(&bi.multiplicity, &g);
                if m == FE::zero() {
                    continue;
                }
                let tuple: Vec<u64> = bi
                    .values
                    .iter()
                    .flat_map(|v| v.combine_from(g))
                    .map(|x: FE| x.canonical_u64())
                    .collect();
                let e = net.entry((bi.bus_id, tuple)).or_insert(FE::zero());
                *e = if bi.is_sender { *e + m } else { *e - m };
            }
        }
    }
    // The verifier's receives on the commit bus: `(index, byte)` for each
    // byte of the public output.
    for (i, &byte) in build.public_output_bytes.iter().enumerate() {
        let e = net
            .entry((u64::from(BusId::Commit), vec![i as u64, u64::from(byte)]))
            .or_insert(FE::zero());
        *e = *e - FE::one();
    }
    let mut out: BTreeMap<u64, Vec<(Vec<u64>, i128)>> = BTreeMap::new();
    for ((bus, tuple), v) in net {
        if v == FE::zero() {
            continue;
        }
        let v = v.canonical_u64();
        let signed = if v > u64::MAX / 2 {
            -((P - v) as i128)
        } else {
            v as i128
        };
        out.entry(bus).or_default().push((tuple, signed));
    }
    for tuples in out.values_mut() {
        tuples.sort();
    }
    out
}

/// A constraint set's values on one row.
fn constraint_values<CS: ConstraintSet<GoldilocksField, GoldilocksExtension>>(
    set: &CS,
    row: &[FE],
) -> Vec<FE> {
    use math::field::element::FieldElement;
    use stark::constraints::builder::ProverEvalFolder;
    use stark::frame::Frame;
    use stark::table::TableView;
    use stark::traits::TransitionEvaluationContext;
    let n = set.meta().len();
    let frame = Frame::<GoldilocksField, GoldilocksExtension>::new(vec![TableView::new(
        vec![row.to_vec()],
        vec![vec![]],
    )]);
    let no_e: Vec<FieldElement<GoldilocksExtension>> = vec![];
    let offset_e = FieldElement::<GoldilocksExtension>::zero();
    let ctx =
        TransitionEvaluationContext::new_prover(frame.as_row_frame(), &no_e, &no_e, &offset_e);
    let mut base = vec![FE::zero(); n];
    let mut ext = vec![FieldElement::<GoldilocksExtension>::zero(); n];
    let mut folder = ProverEvalFolder::new(&ctx, &mut base, &mut ext);
    set.eval(&mut folder);
    base
}

/// Every `(family, table, row, constraint)` of the word tables that does not
/// read zero.
fn broken(build: &Traces) -> Vec<(usize, usize, usize, usize)> {
    fn each<CS: ConstraintSet<GoldilocksField, GoldilocksExtension>>(
        family: usize,
        tables: &[Table],
        set: &CS,
        out: &mut Vec<(usize, usize, usize, usize)>,
    ) {
        for (i, t) in tables.iter().enumerate() {
            for r in 0..t.num_rows() {
                let row: Vec<FE> = (0..t.main_table.width).map(|c| *t.get_main(r, c)).collect();
                let values = constraint_values(set, &row);
                out.extend(
                    (0..values.len())
                        .filter(|&k| values[k] != FE::zero())
                        .map(|k| (family, i, r, k)),
                );
            }
        }
    }
    let w = &build.word_tables;
    let mut out = Vec::new();
    each(0, &w[0], &mem_load8::MemLoad8Constraints, &mut out);
    each(1, &w[1], &mem_store8::MemStore8Constraints, &mut out);
    each(2, &w[2], &mem_loadw::MemLoadWConstraints, &mut out);
    each(3, &w[3], &mem_storew::MemStoreWConstraints, &mut out);
    each(4, &w[4], &mem_load2w::MemLoad2WConstraints, &mut out);
    each(5, &w[5], &mem_store2w::MemStore2WConstraints, &mut out);
    each(6, &w[6], &memw_w::MemwWordConstraints, &mut out);
    // COPY's tables, COPY_W's under the word format.
    each(7, &build.copies, &copy_w::CopyWordConstraints, &mut out);
    out
}

/// The `(family, timestamp, address)` of every active row of the word
/// tables, sorted. Every family keeps `ADDRESS_LO`, `ADDRESS_HI`,
/// `TIMESTAMP` and `MU` at the same places.
fn word_accesses(build: &Traces) -> Vec<(usize, u64, u64)> {
    let columns = [
        (
            mem_load8::cols::ADDRESS_LO,
            mem_load8::cols::ADDRESS_HI,
            mem_load8::cols::TIMESTAMP,
            mem_load8::cols::MU,
        ),
        (
            mem_store8::cols::ADDRESS_LO,
            mem_store8::cols::ADDRESS_HI,
            mem_store8::cols::TIMESTAMP,
            mem_store8::cols::MU,
        ),
        (
            mem_loadw::cols::ADDRESS_LO,
            mem_loadw::cols::ADDRESS_HI,
            mem_loadw::cols::TIMESTAMP,
            mem_loadw::cols::MU,
        ),
        (
            mem_storew::cols::ADDRESS_LO,
            mem_storew::cols::ADDRESS_HI,
            mem_storew::cols::TIMESTAMP,
            mem_storew::cols::MU,
        ),
        (
            mem_load2w::cols::ADDRESS_LO,
            mem_load2w::cols::ADDRESS_HI,
            mem_load2w::cols::TIMESTAMP,
            mem_load2w::cols::MU,
        ),
        (
            mem_store2w::cols::ADDRESS_LO,
            mem_store2w::cols::ADDRESS_HI,
            mem_store2w::cols::TIMESTAMP,
            mem_store2w::cols::MU,
        ),
    ];
    let mut out = Vec::new();
    for (family, &(lo, hi, ts, mu)) in columns.iter().enumerate() {
        for t in &build.word_tables[family] {
            for r in (0..t.num_rows()).filter(|&r| get(t, r, mu) == 1) {
                out.push((family, get(t, r, ts), get(t, r, lo) | get(t, r, hi) << 32));
            }
        }
    }
    out.sort();
    out
}

/// The width of a LOAD or STORE row from its three width flags.
fn width_of(t: &Table, r: usize, flags: [usize; 3]) -> u8 {
    match flags.map(|c| get(t, r, c)) {
        [1, 0, 0] => 2,
        [0, 1, 0] => 4,
        [0, 0, 1] => 8,
        _ => 1,
    }
}

/// The `(family, timestamp, address)` each CPU load and store of the split
/// build of the same run belongs to by [`word_class`], sorted: the word
/// tables' rows a word build must hold.
fn expected_accesses(split: &Traces) -> Vec<(usize, u64, u64)> {
    let family = |class: WordClass, load: bool| match (class, load) {
        (WordClass::Whole, true) => 0,
        (WordClass::Whole, false) => 1,
        (WordClass::Within, true) => 2,
        (WordClass::Within, false) => 3,
        (WordClass::Crossing, true) => 4,
        (WordClass::Crossing, false) => 5,
    };
    let mut out = Vec::new();
    for t in &split.loads {
        for r in (0..t.num_rows()).filter(|&r| get(t, r, ld::MU) == 1) {
            let address = get(t, r, ld::BASE_ADDRESS_0) | get(t, r, ld::BASE_ADDRESS_1) << 32;
            let ts = get(t, r, ld::TIMESTAMP_0) | get(t, r, ld::TIMESTAMP_1) << 32;
            let width = width_of(t, r, [ld::READ2, ld::READ4, ld::READ8]);
            let class = word_class(address, width).expect("a word class");
            out.push((family(class, true), ts, address));
        }
    }
    for t in &split.stores {
        for r in (0..t.num_rows()).filter(|&r| get(t, r, st::MU) == 1) {
            let address = get(t, r, st::BASE_ADDRESS_0) | get(t, r, st::BASE_ADDRESS_1) << 32;
            let ts = get(t, r, st::TIMESTAMP_0) | get(t, r, st::TIMESTAMP_1) << 32;
            let width = width_of(t, r, [st::WRITE2, st::WRITE4, st::WRITE8]);
            let class = word_class(address, width).expect("a word class");
            out.push((family(class, false), ts, address));
        }
    }
    out.sort();
    out
}

/// Every PAGE_W table's preprocessed columns are the verifier's for its page.
fn pages_hold_the_verifiers_columns(build: &Traces) -> Result<(), String> {
    for (config, t) in build.page_configs.iter().zip(&build.pages) {
        let columns = page_w::preprocessed_columns_of(config);
        for (c, column) in columns.iter().enumerate() {
            for (r, value) in column.iter().enumerate() {
                if *t.get_main(r, c) != *value {
                    return Err(format!(
                        "PAGE_W {:#x}: column {c} row {r} is not the verifier's",
                        config.page_base
                    ));
                }
            }
        }
    }
    Ok(())
}

/// ★ Under the word memory format, every chip-free run's windowed build (the
/// in-row register format; windows of 1000 cycles, and of 7 so words cross
/// windows)
/// balances every inner bus under the verifier's AIR set for the format, with
/// BITWISE's multiplicities as the build counted them; every word table's row
/// meets its constraints; every PAGE_W table holds the verifier's columns.
/// No LOAD, STORE, MEMW, MEMW_A or fused table is built, and every PAGE table
/// is a PAGE_W table.
#[test]
fn the_word_walk_builds_balanced_traces() {
    for name in PROGRAMS {
        let (program, logs) = run(name);
        let registers = RegisterFormat::InRow;
        {
            for window in [1000, 7] {
                let mut build = word_build(&program, &logs, registers, window);
                let counts = build.table_counts();
                assert_eq!(
                    (counts.load, counts.store, counts.memw, counts.memw_aligned),
                    (0, 0, 0, 0),
                    "{name} {registers:?} window {window}: the retired tables, {} MEMW_A rows",
                    build
                        .memw_aligneds
                        .iter()
                        .map(|t| t.num_rows())
                        .sum::<usize>()
                );
                assert_eq!(build.fused_counts(), [0, 0], "{name}");
                assert_eq!(build.word_counts()[6], 0, "{name}: no MEMW_W");
                assert!(
                    build
                        .pages
                        .iter()
                        .all(|t| t.main_table.width == page_w::cols::NUM_COLUMNS
                            && t.num_rows() == page_w::ROWS),
                    "{name}: PAGE_W tables"
                );
                pages_hold_the_verifiers_columns(&build).unwrap_or_else(|e| panic!("{name}: {e}"));
                let left = left(&program, &mut build, registers);
                assert!(
                    left.is_empty(),
                    "{name} {registers:?} window {window}: {}",
                    super::bus_net_tests::named(&left)
                );
                let broken = broken(&build);
                assert!(broken.is_empty(), "{name}: {broken:?}");
            }
        }
    }
}

/// ★ Under the word memory format, every run that calls the COPY chip (the
/// asm programs of every row shape, overlapping copies, commits split across
/// calls; the Rust guests' memcpy, memmove, memset and commit cases) builds,
/// by windows of 1000 cycles and of 7: COPY's tables are COPY_W's, MEMW_W
/// holds the chips' word accesses, no retired table is built, every inner bus
/// balances under the verifier's AIR set (the commit bus against the public
/// output as the verifier rebuilds it), every word table's and COPY_W's row
/// meets its constraints, and the public output is the split build's.
#[test]
fn copy_calls_build_balanced_traces_under_the_word_walk() {
    let runs = COPY_ASM
        .iter()
        .map(|&name| (name, run(name)))
        .chain(COPY_RUST.iter().map(|&name| (name, run_rust(name))));
    for (name, (program, logs)) in runs {
        let split = split_build(&program, &logs);
        let registers = RegisterFormat::InRow;
        for window in [1000, 7] {
            let mut build = word_build(&program, &logs, registers, window);
            let counts = build.table_counts();
            assert_eq!(
                (counts.load, counts.store, counts.memw, counts.memw_aligned),
                (0, 0, 0, 0),
                "{name} window {window}: the retired tables"
            );
            assert_eq!(build.fused_counts(), [0, 0], "{name}");
            assert!(
                counts.copy > 0 && build.word_counts()[6] > 0,
                "{name}: COPY_W and MEMW_W tables, {counts:?} {:?}",
                build.word_counts()
            );
            assert!(
                build
                    .copies
                    .iter()
                    .all(|t| t.main_table.width == copy_w::cols::NUM_COLUMNS),
                "{name}: COPY's tables are COPY_W's"
            );
            assert_eq!(
                build.public_output_bytes, split.public_output_bytes,
                "{name}: the public output"
            );
            pages_hold_the_verifiers_columns(&build).unwrap_or_else(|e| panic!("{name}: {e}"));
            let left = left(&program, &mut build, registers);
            assert!(
                left.is_empty(),
                "{name} window {window}: {}",
                super::bus_net_tests::named(&left)
            );
            let broken = broken(&build);
            assert!(broken.is_empty(), "{name}: {broken:?}");
        }
    }
}

/// ★ Every CPU load and store is exactly one word table's row, of the family
/// its width and address name: the word build's rows, as (family,
/// timestamp, address), are the split build's LOAD and STORE rows classed by
/// [`word_class`]. On `fused_mem_cases` every family holds a row; on
/// `all_loadstore_32` the loads and stores are as many as LOAD's and STORE's.
#[test]
fn every_load_and_store_is_one_word_tables_row() {
    for name in PROGRAMS {
        let (program, logs) = run(name);
        let split = split_build(&program, &logs);
        let expected = expected_accesses(&split);
        let registers = RegisterFormat::InRow;
        {
            let build = word_build(&program, &logs, registers, 7);
            let made = word_accesses(&build);
            assert_eq!(made, expected, "{name} {registers:?}");
            if name == "fused_mem_cases" {
                let mut per = [0usize; 6];
                for &(family, _, _) in &made {
                    per[family] += 1;
                }
                assert!(per.iter().all(|&n| n > 0), "{name}: {per:?}");
            }
        }
    }
}

/// ★ The word format proves no syscall but the halt and the COPY chip's yet:
/// a run that hashes is refused by the walk, naming the syscall, before any
/// table is built; the same runs build under the fused format.
#[test]
fn a_run_calling_a_chip_is_refused_by_the_word_walk() {
    use super::windowed::WindowedTraceBuilder;
    for name in ["test_keccak", "test_blake3", "test_keccak256"] {
        let (program, logs) = run(name);
        let mut builder = WindowedTraceBuilder::new(&program, &[], &MaxRowsConfig::small())
            .expect("the builder")
            .register_format(RegisterFormat::InRow)
            .expect("before any window")
            .memw_aligned_order(MemwAlignedOrder::Limbs)
            .expect("before any window")
            .memory_format(MemoryFormat::Words)
            .expect("before any window");
        let body = logs.len() - 1;
        let mut refused = None;
        for w in logs[..body].chunks(1000) {
            if let Err(e) = builder.push(w) {
                refused = Some(e);
                break;
            }
        }
        let refused = match refused {
            Some(e) => Err(e),
            None => builder.finish(&logs[body..]).map(|_| ()),
        };
        assert!(
            matches!(&refused, Err(crate::Error::Prover(why)) if why.contains("no syscall but the halt")),
            "{name}: {refused:?}"
        );
    }
}

/// ★ The word format needs the in-row register format: under the 32-bit
/// token format a register access MEMW_R cannot take goes to MEMW or MEMW_A,
/// which the format retires, so the walk refuses the run.
#[test]
fn the_word_walk_needs_the_in_row_register_format() {
    use super::windowed::WindowedTraceBuilder;
    let (program, logs) = run("fused_mem_cases");
    let mut builder = WindowedTraceBuilder::new(&program, &[], &MaxRowsConfig::small())
        .expect("the builder")
        .register_format(RegisterFormat::Tokens32)
        .expect("before any window")
        .memory_format(MemoryFormat::Words)
        .expect("before any window");
    let body = logs.len() - 1;
    let refused = builder.push(&logs[..body]).map(|_| ());
    assert!(
        matches!(&refused, Err(crate::Error::Prover(why)) if why.contains("in-row register format")),
        "{refused:?}"
    );
}

fn set(t: &mut Table, row: usize, col: usize, value: u64) {
    assert!(t.narrow_main().is_none(), "the forgeries edit wide tables");
    t.main_table.set(row, col, FE::from(value));
}

/// ★ A forged word build is refused on the buses, its honest build balancing
/// (the control): each family's first row moved four cycles later (its
/// dispatch then names no CPU row, its tokens no word's stamp), and a PAGE_W
/// row's final word changed (the last token it consumes is no one's).
#[test]
fn a_moved_word_row_or_a_changed_final_word_is_refused_on_the_buses() {
    let registers = RegisterFormat::InRow;
    let (program, logs) = run("fused_mem_cases");
    let mut honest = word_build(&program, &logs, registers, 7);
    assert!(
        left(&program, &mut honest, registers).is_empty(),
        "the honest build"
    );
    let timestamps = [
        mem_load8::cols::TIMESTAMP,
        mem_store8::cols::TIMESTAMP,
        mem_loadw::cols::TIMESTAMP,
        mem_storew::cols::TIMESTAMP,
        mem_load2w::cols::TIMESTAMP,
        mem_store2w::cols::TIMESTAMP,
    ];
    for (family, &ts) in timestamps.iter().enumerate() {
        let mut forged = word_build(&program, &logs, registers, 7);
        let table = &mut forged.word_tables[family][0];
        let moved = get(table, 0, ts) + 16;
        set(table, 0, ts, moved);
        let left = left(&program, &mut forged, registers);
        assert!(
            left.contains_key(&u64::from(BusId::MemoryOp))
                && left.contains_key(&u64::from(BusId::MemWord)),
            "family {family}: {}",
            super::bus_net_tests::named(&left)
        );
    }
    // The stack's last word written: its PAGE_W row's final word plus one.
    let mut forged = word_build(&program, &logs, registers, 7);
    let (page, row) = forged
        .pages
        .iter()
        .enumerate()
        .find_map(|(p, t)| {
            (0..t.num_rows())
                .find(|&r| get(t, r, page_w::cols::TIMESTAMP) > 0)
                .map(|r| (p, r))
        })
        .expect("a touched word");
    let fini = get(&forged.pages[page], row, page_w::cols::FINI_LO);
    set(
        &mut forged.pages[page],
        row,
        page_w::cols::FINI_LO,
        fini ^ 1,
    );
    let left = left(&program, &mut forged, registers);
    assert!(
        left.contains_key(&u64::from(BusId::MemWord)),
        "{}",
        super::bus_net_tests::named(&left)
    );
}

/// `program`'s run under the word memory format walked from the fast engine's
/// word history (record mode) on `walkers` threads at light walk `level`, by
/// windows of `window` cycles, its streamed ops dropped; with `recheck`, every
/// `recheck`-th window is also walked from the lean pass's entries and its
/// first cells compared with the history's, by word.
fn word_record_build(
    program: &Elf,
    window: usize,
    walkers: usize,
    level: u8,
    recheck: usize,
) -> Result<Traces, crate::Error> {
    use super::windowed::parallel::{ExecWindow, RunEnd, walk_windows_from_history};
    use super::windowed::{ChunkJob, WindowedTraceBuilder};
    use executor::vm::fast::FastExecutor;
    let max_rows = MaxRowsConfig::small();
    let mut builder = WindowedTraceBuilder::new(program, &[], &max_rows)?
        .register_format(RegisterFormat::InRow)?
        .memw_aligned_order(MemwAlignedOrder::Limbs)?
        .memory_format(MemoryFormat::Words)?
        .drop_streamed_ops()?
        .light_walk(level)?;
    let (tx, rx) = std::sync::mpsc::sync_channel::<ExecWindow>(2);
    let mut chunks = Vec::new();
    let report = std::thread::scope(|scope| {
        scope.spawn(move || {
            let mut fast = FastExecutor::new(program, Vec::new()).expect("the fast engine");
            fast.enable_word_history().expect("the word history");
            let mut k = 0usize;
            loop {
                let start = (recheck > 0 && k.is_multiple_of(recheck)).then(|| fast.snapshot());
                let Some((logs, history)) = fast.resume_word_history(window).expect("the run")
                else {
                    break;
                };
                assert!(history.words, "a word history");
                let last = (fast.pc() == 0)
                    .then(|| fast.take_word_history_final().map(RunEnd::boxed_words))
                    .flatten();
                let sent = tx.send(ExecWindow {
                    logs: std::sync::Arc::new(logs),
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
    })?;
    let mut traces = builder.finish(&report.last)?;
    traces.insert_streamed(chunks)?;
    Ok(traces)
}

/// ★ Record mode under the word memory format: a run walked from the
/// executor's word history (a timestamp an aligned word, each word's bytes
/// at it), on one walker and on three, at light walk levels 0 and 2, with
/// every second window re-walked from the lean pass's entries and its first
/// cells compared with the history's by word, builds the tables the
/// sequential windowed build makes, and they balance every bus.
#[test]
fn record_mode_under_the_word_format_builds_the_windowed_tables() {
    for name in [
        "fused_mem_cases",
        "all_loadstore_32",
        "misalign_sw",
        "loop_4096",
        "test_copy_ecalls",
        "test_copy_overlaps",
        "test_commit_split",
    ] {
        let (program, logs) = run(name);
        let want = word_build(&program, &logs, RegisterFormat::InRow, 7);
        for window in [7, 33] {
            for (walkers, level) in [(1, 0u8), (3, 0), (3, 2)] {
                for recheck in [0, 2] {
                    let mut got = word_record_build(&program, window, walkers, level, recheck)
                        .unwrap_or_else(|e| {
                            panic!("{name}/{window}/{walkers}/{level}/{recheck}: {e:?}")
                        });
                    crate::tests::windowed_builder_tests::same_traces(&want, &got);
                    let left = left(&program, &mut got, RegisterFormat::InRow);
                    assert!(
                        left.is_empty(),
                        "{name}: {}",
                        super::bus_net_tests::named(&left)
                    );
                }
            }
        }
    }
}
