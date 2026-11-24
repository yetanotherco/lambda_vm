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
    copy_w, ecrec_w, keccak_w, mem_load2w, mem_load8, mem_loadw, mem_store2w, mem_store8,
    mem_storew, memw_w, page_w,
};

type Table = TraceTable<GoldilocksField, GoldilocksExtension>;

/// The field's order: `−k` reads as `P − k`.
const P: u64 = 0xFFFF_FFFF_0000_0001;

/// The chip-free programs: every width, sign and alignment of the CPU's loads
/// and stores (`fused_mem_cases`), every instruction, the misaligned ones, and
/// loads and stores across `2^32` (`cross_4gib`).
const PROGRAMS: [&str; 14] = [
    "fused_mem_cases",
    "cross_4gib",
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
/// memset, the commit syscall (across `2^32` too: `copy_4gib`).
const COPY_ASM: [&str; 5] = [
    "test_copy_ecalls",
    "copy_4gib",
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
    let keccak_rnd = crate::block_whir::BlockFormat::production().keccak_rnd;
    windowed(program, logs, &MaxRowsConfig::small(), window, &|b| {
        b.register_format(registers)
            .expect("before any window")
            .memw_aligned_order(MemwAlignedOrder::Limbs)
            .expect("before any window")
            .keccak_rnd_form(keccak_rnd)
            .ecdas_form(crate::block_whir::BlockFormat::production().ecdas)
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
    let format = crate::block_whir::BlockFormat::production()
        .with_memory(MemoryFormat::Words)
        .with_page_run(0);
    let mut airs = crate::block_airs(
        program,
        &options,
        &build.page_configs,
        &build.table_counts(),
        &inputs,
        format.keccak_rnd,
        format.memw_aligned,
        registers,
        format.ecdas,
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
    // COPY's tables, COPY_W's under the word format; KECCAK's, KECCAK_W's.
    each(7, &build.copies, &copy_w::CopyWordConstraints, &mut out);
    each(
        8,
        &build.keccaks,
        &keccak_w::KeccakWordConstraints,
        &mut out,
    );
    each(9, &build.ecrecs, &ecrec_w::EcrecWordConstraints, &mut out);
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

/// ★ Under the word memory format, a run of keccak256 calls (every input
/// offset and the lengths at each block edge, the digest at offsets of its
/// own and once over its own input, then a commit of the digests) builds, by
/// windows of 1000 cycles and of 7: KECCAK's tables are KECCAK_W's, MEMW_W
/// holds their word accesses, every inner bus balances under the verifier's
/// set (the round chip's included, the commit bus against the public
/// output), every word table's, COPY_W's and KECCAK_W's row meets its
/// constraints, and the digests committed are the split build's.
#[test]
fn keccak256_calls_build_balanced_traces_under_the_word_walk() {
    let name = "test_keccak256";
    let (program, logs) = run(name);
    let split = split_build(&program, &logs);
    let registers = RegisterFormat::InRow;
    for window in [1000, 7] {
        let mut build = word_build(&program, &logs, registers, window);
        let counts = build.table_counts();
        assert_eq!(
            (counts.load, counts.store, counts.memw, counts.memw_aligned),
            (0, 0, 0, 0),
            "window {window}: the retired tables"
        );
        assert!(
            counts.keccak > 0 && counts.keccak_rnd > 0 && build.word_counts()[6] > 0,
            "KECCAK_W, its rounds and MEMW_W: {counts:?} {:?}",
            build.word_counts()
        );
        assert!(
            build
                .keccaks
                .iter()
                .all(|t| t.main_table.width == keccak_w::cols::NUM_COLUMNS),
            "KECCAK's tables are KECCAK_W's"
        );
        assert_eq!(
            build.public_output_bytes, split.public_output_bytes,
            "the digests committed"
        );
        pages_hold_the_verifiers_columns(&build).unwrap_or_else(|e| panic!("{e}"));
        let left = left(&program, &mut build, registers);
        assert!(
            left.is_empty(),
            "window {window}: {}",
            super::bus_net_tests::named(&left)
        );
        let broken = broken(&build);
        assert!(broken.is_empty(), "{broken:?}");
    }
}

/// ★ Under the word memory format, a run of ECREC calls (their operands
/// aligned, as the executor demands) builds by windows of 1000 cycles and
/// of 7: ECREC's tables are ECREC_W's, MEMW_W holds their word accesses,
/// every inner bus balances under the verifier's set (ECDAS's chains
/// included), every word table's and ECREC_W's row meets its constraints, and
/// the keys committed are the split build's.
#[test]
fn ecrec_calls_build_balanced_traces_under_the_word_walk() {
    // Runs `test_ecrec`: the ECREC forgery switch stays off meanwhile.
    let _ecrec = crate::tables::trace_builder::ecrec_forge_free();
    let name = "test_ecrec";
    let (program, logs) = run(name);
    let split = split_build(&program, &logs);
    for window in [1000, 7] {
        let mut build = word_build(&program, &logs, RegisterFormat::InRow, window);
        let counts = build.table_counts();
        assert!(
            counts.ecrec > 0 && counts.ecdas > 0 && build.word_counts()[6] > 0,
            "ECREC_W, ECDAS and MEMW_W: {counts:?} {:?}",
            build.word_counts()
        );
        assert!(
            build
                .ecrecs
                .iter()
                .all(|t| t.main_table.width == ecrec_w::cols::NUM_COLUMNS),
            "ECREC's tables are ECREC_W's"
        );
        assert_eq!(build.public_output_bytes, split.public_output_bytes);
        let left = left(&program, &mut build, RegisterFormat::InRow);
        assert!(
            left.is_empty(),
            "window {window}: {}",
            super::bus_net_tests::named(&left)
        );
        let broken = broken(&build);
        assert!(broken.is_empty(), "{broken:?}");
    }
}

/// ★ A run whose loads and stores cross `2^32` at every width (`cross_4gib`:
/// from the last aligned word of the low 4 GiB range into low word 0 of high
/// word 1, the large block's LD at `2^32 − 4` among them) halts on its success
/// path, and its word build carries each of those rows' second key into the
/// high word (`CARRY` 1 on 9 MEM_LOAD2W and 3 MEM_STORE2W rows, every other
/// row 0), at windows of 1000 and 7 cycles. The build balances and meets
/// every constraint (`the_word_walk_builds_balanced_traces`, which runs it).
#[test]
fn a_run_crossing_two_to_the_32_carries_into_the_high_word() {
    let (program, logs) = run("cross_4gib");
    // `li a0, 0; li a7, 93; ecall` ends the success path (the failure path
    // sets a0 to 1).
    assert_eq!(logs[logs.len() - 3].dst_val, 0, "the program's checks hold");
    for window in [1000, 7] {
        let build = word_build(&program, &logs, RegisterFormat::InRow, window);
        let count = |family: usize, carry: usize, mu: usize, value: u64| -> usize {
            build.word_tables[family]
                .iter()
                .map(|t| {
                    (0..t.num_rows())
                        .filter(|&r| get(t, r, mu) == 1 && get(t, r, carry) == value)
                        .count()
                })
                .sum()
        };
        for (family, carry, mu, carried, others) in [
            (4, mem_load2w::cols::CARRY, mem_load2w::cols::MU, 9, 1),
            (5, mem_store2w::cols::CARRY, mem_store2w::cols::MU, 3, 0),
        ] {
            assert_eq!(
                (count(family, carry, mu, 1), count(family, carry, mu, 0)),
                (carried, others),
                "family {family}, window {window}: (carried, not carried) rows"
            );
        }
    }
}

/// ★ A run whose COPY chip calls cross `2^32` (`copy_4gib`: memcpy and memmove
/// with the source or the destination across it, one whose unread first
/// source word is the last below it, a memset across it, a commit from a
/// source across it) halts on its success path, and its word build carries
/// keys into the high word: every one of COPY_W's carries is 1 on some row,
/// at windows of 1000 and 7 cycles. The build balances, its commits against
/// the public output (`copy_calls_build_balanced_traces_under_the_word_walk`,
/// which runs it).
#[test]
fn a_run_copying_across_two_to_the_32_carries_its_keys() {
    let (program, logs) = run("copy_4gib");
    assert_eq!(logs[logs.len() - 3].dst_val, 0, "the program's checks hold");
    for window in [1000, 7] {
        let build = word_build(&program, &logs, RegisterFormat::InRow, window);
        for carry in [
            copy_w::cols::SRC_CARRY,
            copy_w::cols::DST_CARRY,
            copy_w::cols::OP_CARRY,
        ] {
            let rows: usize = build
                .copies
                .iter()
                .map(|t| {
                    (0..t.num_rows())
                        .filter(|&r| get(t, r, copy_w::cols::MU) == 1 && get(t, r, carry) == 1)
                        .count()
                })
                .sum();
            assert!(
                rows > 0,
                "window {window}: carry column {carry} is 1 on no row"
            );
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

/// ★ The word format proves no syscall but the halt, the COPY chip's,
/// keccak256 and ECREC yet: a run that permutes a Keccak state, compresses
/// with BLAKE3 or multiplies a curve point is refused by the walk, naming the
/// syscall, before any table is built.
#[test]
fn a_run_calling_a_chip_is_refused_by_the_word_walk() {
    use super::windowed::WindowedTraceBuilder;
    for name in ["test_keccak", "test_blake3", "test_ecsm"] {
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
    word_record_build_with(
        program,
        window,
        walkers,
        level,
        recheck,
        &MaxRowsConfig::small(),
        false,
    )
}

/// [`word_record_build`] at `max_rows`, the word families streamed in chunks
/// as the walk fills them when `stream`
/// ([`super::windowed::WindowedTraceBuilder::stream_word_tables`]).
fn word_record_build_with(
    program: &Elf,
    window: usize,
    walkers: usize,
    level: u8,
    recheck: usize,
    max_rows: &MaxRowsConfig,
    stream: bool,
) -> Result<Traces, crate::Error> {
    use super::windowed::parallel::{ExecWindow, RunEnd, walk_windows_from_history};
    use super::windowed::{ChunkJob, WindowedTraceBuilder};
    use executor::vm::fast::FastExecutor;
    let mut builder = WindowedTraceBuilder::new(program, &[], max_rows)?
        .register_format(RegisterFormat::InRow)?
        .memw_aligned_order(MemwAlignedOrder::Limbs)?
        .keccak_rnd_form(crate::block_whir::BlockFormat::production().keccak_rnd)
        .ecdas_form(crate::block_whir::BlockFormat::production().ecdas)
        .memory_format(MemoryFormat::Words)?
        .drop_streamed_ops()?
        .light_walk(level)?;
    if stream {
        builder = builder.stream_word_tables()?;
    }
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
    // Runs `test_ecrec`: the ECREC forgery switch stays off meanwhile.
    let _ecrec = crate::tables::trace_builder::ecrec_forge_free();
    for name in [
        "fused_mem_cases",
        "cross_4gib",
        "all_loadstore_32",
        "misalign_sw",
        "loop_4096",
        "test_copy_ecalls",
        "copy_4gib",
        "test_copy_overlaps",
        "test_commit_split",
        "test_keccak256",
        "test_ecrec",
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

/// ★ A run that commits output, stated as if it called no chip. As built,
/// with its COPY_W tables and its output, it leaves nothing on any bus. The
/// forger's statement drops the COPY_W tables and the output: each of the
/// run's commit calls stays on the syscall bus, one send of `(T, 0, 64, 0)`
/// no table receives (in a chip-free word frame HALT's is the one receiver,
/// at 93: `word_memory_tests::the_chip_free_word_frame_has_one_syscall_receiver`).
/// The other leftovers (the copy's word messages, its register reads and
/// byte checks) a forger could rebuild; the CPU's send it cannot.
#[test]
fn a_run_that_commits_stated_without_its_copy_tables_leaves_its_calls_on_the_syscall_bus() {
    let (program, logs) = run("test_commit_split");
    let mut honest = word_build(&program, &logs, RegisterFormat::InRow, 1000);
    let calls: usize = honest
        .copies
        .iter()
        .map(|t| {
            (0..t.num_rows())
                .filter(|&r| {
                    get(t, r, copy_w::cols::FIRST) == 1 && get(t, r, copy_w::cols::IS_WRITE) == 1
                })
                .count()
        })
        .sum();
    assert!(calls >= 2, "the run commits twice or more: {calls}");
    let as_built = left(&program, &mut honest, RegisterFormat::InRow);
    assert!(
        as_built.is_empty(),
        "the honest build balances: {as_built:?}"
    );
    let mut forged = word_build(&program, &logs, RegisterFormat::InRow, 1000);
    forged.copies.clear();
    forged.public_output_bytes.clear();
    let l = left(&program, &mut forged, RegisterFormat::InRow);
    let ecall = l
        .get(&u64::from(BusId::Ecall))
        .expect("the syscall bus keeps the calls");
    assert_eq!(ecall.len(), calls, "{ecall:?}");
    assert!(
        ecall
            .iter()
            .all(|(tuple, net)| *net == 1 && tuple[1..] == [0, 64, 0]),
        "one unreceived commit send each: {ecall:?}"
    );
}

/// ★ The word memory format under the 32-bit register tokens, which the
/// prover refuses (the walk here, the prove's gate before it): a forger's
/// walk skips the refusal. Its build states a MEMW_A table (the register
/// accesses MEMW_R cannot take under the 32-bit tokens), and the verifier's
/// retired-table rule refuses that statement. Control: the same build
/// balances on every bus under the word format's set with those counts and
/// breaks no word-table constraint, so the rule is the one check (its
/// proof-level twin is `tests::word_memory_block_tests::a_word_block_under_the_32_bit_registers_is_refused_by_the_retired_table_rule_alone`).
#[test]
fn a_word_build_under_the_32_bit_registers_is_refused_by_the_retired_table_rule() {
    use crate::tables::trace_builder::WORDS_UNDER_TOKENS32;
    for name in ["all_instructions_64", "fused_mem_cases"] {
        let (program, logs) = run(name);
        WORDS_UNDER_TOKENS32.with(|c| c.set(true));
        let built = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            word_build(&program, &logs, RegisterFormat::Tokens32, 1000)
        }));
        WORDS_UNDER_TOKENS32.with(|c| c.set(false));
        let mut build = built.unwrap_or_else(|_| panic!("{name}: the forger's walk"));
        let counts = build.table_counts();
        assert!(counts.memw_aligned >= 1, "{name}: a MEMW_A table");
        let refused = crate::block_whir::word_tables_under(
            MemoryFormat::Words,
            &counts,
            build.fused_counts().map(|n| n as u64),
        );
        assert!(
            matches!(&refused, Err(crate::Error::InvalidTableCounts(why)) if why.contains("no MEMW_A table")),
            "{name}: {refused:?}"
        );
        let l = left(&program, &mut build, RegisterFormat::Tokens32);
        assert!(l.is_empty(), "{name}: control: {l:?}");
        assert!(broken(&build).is_empty(), "{name}: control");
    }
    // Without the switch the walk refuses.
    let (program, logs) = run("all_instructions_64");
    let walked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        word_build(&program, &logs, RegisterFormat::Tokens32, 1000)
    }));
    assert!(walked.is_err(), "the honest walk refuses the 32-bit tokens");
}

/// ★ Completeness: word families cut into several tables. Two runs built
/// with word tables of at most 4 rows, at windows of 1000 and 7 cycles: a
/// family holds two tables or more, the rows are the uncut build's, every
/// bus balances and every word-table constraint holds.
#[test]
fn a_word_family_cut_into_several_tables_balances() {
    let tiny = MaxRowsConfig {
        mem_load: 4,
        mem_store: 4,
        ..MaxRowsConfig::small()
    };
    for name in ["fused_mem_cases", "all_loadstore_32"] {
        let (program, logs) = run(name);
        for window in [1000, 7] {
            let mut build = windowed(&program, &logs, &tiny, window, &|b| {
                b.register_format(RegisterFormat::InRow)
                    .expect("before any window")
                    .memw_aligned_order(MemwAlignedOrder::Limbs)
                    .expect("before any window")
                    .memory_format(MemoryFormat::Words)
                    .expect("before any window")
            });
            let counts = build.word_counts();
            let uncut = word_build(&program, &logs, RegisterFormat::InRow, window);
            assert!(
                counts.iter().any(|&n| n > 1),
                "{name} {window}: a family is cut"
            );
            assert_eq!(
                word_accesses(&build),
                word_accesses(&uncut),
                "{name} {window}: the uncut build's rows"
            );
            let l = left(&program, &mut build, RegisterFormat::InRow);
            assert!(l.is_empty(), "{name} {window}: {l:?}");
            assert!(broken(&build).is_empty(), "{name} {window}");
        }
    }
}

/// Word caps of 4 rows a table (`mem_load`, `mem_store`): every family with
/// rows cuts into chunks.
fn tiny() -> MaxRowsConfig {
    MaxRowsConfig {
        mem_load: 4,
        mem_store: 4,
        ..MaxRowsConfig::small()
    }
}

/// `program`'s run built by windows of `window` cycles under the word memory
/// format at `max_rows`, the word families streamed when `stream`, the
/// streamed ops dropped as handed out when `drop_ops`; each chunk passed
/// through `edit` before it is put back. The traces and the word chunks
/// handed out.
fn streamed_word_build(
    program: &Elf,
    logs: &[Log],
    max_rows: &MaxRowsConfig,
    window: usize,
    (stream, drop_ops): (bool, bool),
    edit: impl Fn(&mut Vec<super::windowed::StreamedChunk>),
) -> (Traces, usize) {
    use super::windowed::WindowedTraceBuilder;
    let keccak_rnd = crate::block_whir::BlockFormat::production().keccak_rnd;
    let mut builder = WindowedTraceBuilder::new(program, &[], max_rows)
        .and_then(|b| b.register_format(RegisterFormat::InRow))
        .and_then(|b| b.memw_aligned_order(MemwAlignedOrder::Limbs))
        .map(|b| b.keccak_rnd_form(keccak_rnd))
        .map(|b| b.ecdas_form(crate::block_whir::BlockFormat::production().ecdas))
        .and_then(|b| b.memory_format(MemoryFormat::Words))
        .expect("the builder");
    if stream {
        builder = builder.stream_word_tables().expect("before any window");
    }
    if drop_ops {
        builder = builder.drop_streamed_ops().expect("before any window");
    }
    let body = logs.len() - 1;
    let cut = body - body % window;
    let mut chunks = Vec::new();
    for w in logs[..cut].chunks(window) {
        chunks.extend(builder.push(w).expect("a window"));
    }
    let words = chunks
        .iter()
        .filter(|c| c.table.word_family().is_some())
        .count();
    let mut traces = builder.finish(&logs[cut..]).expect("the last window");
    edit(&mut chunks);
    traces
        .insert_streamed(chunks)
        .expect("every chunk has a placeholder");
    (traces, words)
}

/// The runs a streamed word build is checked on: the CPU's loads and stores
/// of every width and alignment, and the chips the word format takes.
const STREAMED: [&str; 6] = [
    "fused_mem_cases",
    "all_loadstore_32",
    "misalign_sd",
    "test_copy_ecalls",
    "test_keccak256",
    "test_ecrec",
];

/// ★ The word families streamed in chunks as the walk fills them
/// (`stream_word_tables`) build the finish-built tables, byte for byte: each
/// run at word caps of 4 rows (families cut into many chunks), at windows of
/// 7, 33 and 4096 cycles, with the windows kept whole and with the streamed
/// ops dropped as handed out (the block's tails); word chunks are handed out
/// in every run. Off (the default), none is.
#[test]
fn streamed_word_families_are_the_finish_built_tables() {
    // Runs `test_ecrec`: the ECREC forgery switch stays off meanwhile.
    let _ecrec = crate::tables::trace_builder::ecrec_forge_free();
    let max_rows = tiny();
    let mut streamed = 0usize;
    for name in STREAMED {
        let (program, logs) = run(name);
        for window in [7, 33, 4096] {
            let (want, none) =
                streamed_word_build(&program, &logs, &max_rows, window, (false, false), |_| {});
            assert_eq!(none, 0, "{name}/{window}: off streams no word chunk");
            for drop_ops in [false, true] {
                let (got, words) = streamed_word_build(
                    &program,
                    &logs,
                    &max_rows,
                    window,
                    (true, drop_ops),
                    |_| {},
                );
                streamed += words;
                crate::tests::windowed_builder_tests::same_traces(&want, &got);
                assert_eq!(
                    got.word_counts(),
                    want.word_counts(),
                    "{name}/{window}/{drop_ops}"
                );
            }
        }
    }
    assert!(streamed > 20, "{streamed} word chunks handed out");
}

/// ★ Record mode with the word families streamed: a run walked from the
/// executor's word history on one walker and on three, at light walk levels
/// 0, 1 and 2, builds the finish-built tables at word caps of 4 rows, and
/// they balance every bus.
#[test]
fn record_mode_with_streamed_word_families_builds_the_finish_built_tables() {
    // Runs `test_ecrec`: the ECREC forgery switch stays off meanwhile.
    let _ecrec = crate::tables::trace_builder::ecrec_forge_free();
    let max_rows = tiny();
    for name in STREAMED {
        let (program, logs) = run(name);
        let (want, _) = streamed_word_build(&program, &logs, &max_rows, 7, (false, false), |_| {});
        for window in [7, 33] {
            for (walkers, level) in [(1, 0u8), (3, 1), (3, 2)] {
                let mut got =
                    word_record_build_with(&program, window, walkers, level, 0, &max_rows, true)
                        .unwrap_or_else(|e| panic!("{name}/{window}/{walkers}/{level}: {e:?}"));
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

/// ★ The parity above is load-bearing: a streamed word build that loses one
/// chunk of a family (its slot left a placeholder), or whose two chunks of
/// one family come back swapped, is not the finish-built one.
#[test]
fn a_dropped_or_swapped_word_chunk_reds_the_parity() {
    use super::windowed::StreamTable;
    let max_rows = tiny();
    let (program, logs) = run("fused_mem_cases");
    let (want, _) = streamed_word_build(&program, &logs, &max_rows, 7, (false, false), |_| {});
    let differs = |got: &Traces| {
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            crate::tests::windowed_builder_tests::same_traces(&want, got)
        }))
        .is_err()
    };
    let of = |chunks: &[super::windowed::StreamedChunk], table: StreamTable| -> Vec<usize> {
        chunks
            .iter()
            .enumerate()
            .filter(|(_, c)| c.table == table)
            .map(|(i, _)| i)
            .collect()
    };
    // The family with the most chunks handed out.
    let most = std::cell::Cell::new(StreamTable::MemLoad8);
    streamed_word_build(&program, &logs, &max_rows, 7, (true, false), |chunks| {
        let best = StreamTable::WORDS
            .into_iter()
            .max_by_key(|&t| of(chunks, t).len())
            .expect("a word family");
        assert!(
            of(chunks, best).len() >= 2,
            "a family with two chunks handed out"
        );
        most.set(best);
    });
    let table = most.get();
    for drop_ops in [false, true] {
        let (got, _) =
            streamed_word_build(&program, &logs, &max_rows, 7, (true, drop_ops), |chunks| {
                let at = of(chunks, table);
                chunks.remove(at[0]);
            });
        assert!(
            differs(&got),
            "{table:?} {drop_ops}: a dropped chunk passed"
        );
        let (got, _) =
            streamed_word_build(&program, &logs, &max_rows, 7, (true, drop_ops), |chunks| {
                let at = of(chunks, table);
                assert!(at.len() >= 2, "two {table:?} chunks");
                let (a, b) = (chunks[at[0]].index, chunks[at[1]].index);
                chunks[at[0]].index = b;
                chunks[at[1]].index = a;
            });
        assert!(differs(&got), "{drop_ops}: swapped chunks passed");
    }
}

// --- The executor's lean stream ----------------------------------------------

/// A window's history edited before it is sent (a forger's executor).
type Tamper<'a> = &'a (dyn Fn(&mut executor::vm::history::WindowHistory) + Sync);

/// How [`word_stream_build`] runs the executor and the walk.
#[derive(Clone, Copy, Default)]
struct StreamRun<'a> {
    /// The executor's stream lean ([`executor::vm::fast::FastExecutor::resume_word_lean`]).
    lean: bool,
    /// The builder given the lean stream's replay
    /// ([`super::windowed::WindowedTraceBuilder::lean_stream`]).
    replay: bool,
    /// Window `.0`'s history edited by `.1` before it is sent.
    tamper: Option<(usize, Tamper<'a>)>,
    /// A lean window sent with a log a cycle (a mismatched pair).
    logs_with_lean: bool,
}

/// One streamed chunk as phase A hands it out: its table, index and records
/// digest (`None` for a chunk not held as records).
type HandedChunk = (String, usize, Option<[u64; 2]>);

/// `program`'s run in record mode under the word memory format, by windows of
/// `window` cycles on `walkers` threads at light walk `level`, its streamed
/// ops dropped and its CPU chunks held as their records (their logs kept, as
/// the card fills them): the tables, and each streamed chunk with its records'
/// digest (the one regeneration checks).
fn word_stream_build(
    program: &Elf,
    window: usize,
    walkers: usize,
    level: u8,
    run: StreamRun<'_>,
) -> Result<(Traces, Vec<HandedChunk>), crate::Error> {
    use super::windowed::WindowedTraceBuilder;
    use super::windowed::parallel::{ExecWindow, RunEnd, walk_windows_from_history};
    use executor::vm::fast::{FastExecutor, Replay};
    let mut builder = WindowedTraceBuilder::new(program, &[], &MaxRowsConfig::small())?
        .register_format(RegisterFormat::InRow)?
        .memw_aligned_order(MemwAlignedOrder::Limbs)?
        .keccak_rnd_form(crate::block_whir::BlockFormat::production().keccak_rnd)
        .ecdas_form(crate::block_whir::BlockFormat::production().ecdas)
        .memory_format(MemoryFormat::Words)?
        .drop_streamed_ops()?
        .card_fill_streamed(std::sync::Arc::new(|_| true), true, false)?
        .light_walk(level)?;
    assert!(builder.keeps_logs(), "the CPU chunks keep their logs");
    if run.replay {
        let replay = Replay::new(program).map_err(|e| crate::Error::Execution(format!("{e}")))?;
        builder = builder.lean_stream(std::sync::Arc::new(replay), false)?;
    }
    let (tx, rx) = std::sync::mpsc::sync_channel::<ExecWindow>(2);
    let mut chunks = Vec::new();
    let mut handed = Vec::new();
    let report = std::thread::scope(|scope| {
        scope.spawn(move || {
            let mut fast = FastExecutor::new(program, Vec::new()).expect("the fast engine");
            fast.enable_word_history().expect("the word history");
            if run.lean {
                fast.enable_lean_stream().expect("the lean stream");
            }
            let mut k = 0usize;
            loop {
                let window = if run.lean {
                    fast.resume_word_lean(window)
                        .expect("the run")
                        .map(|(cycles, h)| {
                            let logs = if run.logs_with_lean {
                                let log = executor::vm::logs::Log {
                                    current_pc: 0,
                                    next_pc: 0,
                                    src1_val: 0,
                                    src2_val: 0,
                                    dst_val: 0,
                                };
                                vec![log; cycles]
                            } else {
                                Vec::new()
                            };
                            (logs, h)
                        })
                } else {
                    fast.resume_word_history(window).expect("the run")
                };
                let Some((logs, mut history)) = window else {
                    break;
                };
                assert_eq!(history.lean, run.lean);
                if let Some((at, tamper)) = run.tamper
                    && at == k
                {
                    tamper(&mut history);
                }
                let last = (fast.pc() == 0)
                    .then(|| fast.take_word_history_final().map(RunEnd::boxed_words))
                    .flatten();
                let sent = tx.send(ExecWindow {
                    logs: std::sync::Arc::new(logs),
                    start: None,
                    history: Some(Box::new(history)),
                    last,
                });
                if sent.is_err() {
                    break;
                }
                k += 1;
            }
        });
        walk_windows_from_history(&mut builder, rx, walkers, 0, |accumulator, walked, _| {
            for job in accumulator.absorb(walked) {
                let mut chunk = job.generate();
                // A chunk held as records: their digest, then the host's
                // fill of them (the card's fill, on the host).
                let digest = chunk.trace.take_records_main().map(|records| {
                    let digest = records.digest();
                    let packed = records.host_fill().expect("the host fills its records");
                    let columns = packed
                        .widen::<GoldilocksField>()
                        .expect("the packed columns widen");
                    let (rows, cols) = (records.rows(), records.cols());
                    let mut data = Vec::with_capacity(rows * cols);
                    for r in 0..rows {
                        data.extend(columns.iter().map(|column| column.evals()[r]));
                    }
                    chunk.trace = Table::new_main(data, cols, 1);
                    digest
                });
                handed.push((format!("{:?}", chunk.table), chunk.index, digest));
                chunks.push(chunk);
            }
            Ok(())
        })
    })?;
    let mut traces = builder.finish(&report.last)?;
    traces.insert_streamed(chunks)?;
    Ok((traces, handed))
}

/// The fixtures the lean stream is checked on: every width, sign and
/// alignment of loads and stores and across `2^32`, every instruction, the
/// misaligned ones, a loop, and the ECALLs the word format takes (copy,
/// commit, Keccak, signature recovery; the executor's tests replay every
/// ECALL, BLAKE3's among them).
const LEAN_PROGRAMS: [&str; 11] = [
    "fused_mem_cases",
    "cross_4gib",
    "all_loadstore_32",
    "misalign_sw",
    "loop_4096",
    "test_copy_ecalls",
    "copy_4gib",
    "test_copy_overlaps",
    "test_commit_split",
    "test_keccak256",
    "test_ecrec",
];

/// ★ The executor's lean stream: each window sent as its access records,
/// ECALLs and starting cells only, re-run by its walker from them, a batch at
/// a time. On every fixture, at windows of 7 and 33 cycles (every run's last
/// window halts inside it), on one walker and on three, at light walk 1 and
/// 2: the tables are the full stream's, every streamed chunk is handed out
/// alike with the same records digest (the CPU chunks' kept logs byte for
/// byte, as regeneration checks them), and every bus balances.
#[test]
fn the_lean_stream_builds_the_full_streams_tables_and_records() {
    let _ecrec = crate::tables::trace_builder::ecrec_forge_free();
    let lean = StreamRun {
        lean: true,
        replay: true,
        ..StreamRun::default()
    };
    // The chunks held as records (a CPU chunk is when it is cut full).
    let mut held = 0usize;
    for name in LEAN_PROGRAMS {
        let (program, _) = run(name);
        for window in [7, 33] {
            for (walkers, level) in [(1, 1u8), (3, 2)] {
                let what = format!("{name}/{window}/{walkers}/{level}");
                eprintln!("{what}: full");
                let (want, want_chunks) =
                    word_stream_build(&program, window, walkers, level, StreamRun::default())
                        .unwrap_or_else(|e| panic!("{what}: full: {e:?}"));
                eprintln!("{what}: lean");
                let (mut got, got_chunks) =
                    word_stream_build(&program, window, walkers, level, lean)
                        .unwrap_or_else(|e| panic!("{what}: lean: {e:?}"));
                crate::tests::windowed_builder_tests::same_traces(&want, &got);
                assert_eq!(got_chunks, want_chunks, "{what}");
                held += got_chunks.iter().filter(|(_, _, d)| d.is_some()).count();
                if window == 7 && walkers == 3 {
                    let left = left(&program, &mut got, RegisterFormat::InRow);
                    assert!(
                        left.is_empty(),
                        "{what}: {}",
                        super::bus_net_tests::named(&left)
                    );
                }
            }
        }
    }
    assert!(held > 0, "a chunk held as records, its logs kept");
}

/// ★ The lean stream is refused, never run as another mode: a lean window
/// sent to a walk with no replay, or with a log a cycle beside it.
#[test]
fn a_lean_window_without_its_replay_or_with_logs_is_refused() {
    let (program, _) = run("loop_4096");
    let no_replay = StreamRun {
        lean: true,
        ..StreamRun::default()
    };
    let err = word_stream_build(&program, 33, 2, 2, no_replay).map(|_| ());
    assert!(
        matches!(&err, Err(crate::Error::Prover(m)) if m.contains("needs the lean stream's replay")),
        "{err:?}"
    );
    let paired = StreamRun {
        lean: true,
        replay: true,
        logs_with_lean: true,
        ..StreamRun::default()
    };
    let err = word_stream_build(&program, 33, 2, 2, paired).map(|_| ());
    assert!(
        matches!(&err, Err(crate::Error::Prover(m)) if m.contains("record mode: window 0's history")),
        "{err:?}"
    );
    // At light walk 0 a cycle's record is found by its index, which a lean
    // window has none of.
    let lean = StreamRun {
        lean: true,
        replay: true,
        ..StreamRun::default()
    };
    let err = word_stream_build(&program, 33, 2, 0, lean).map(|_| ());
    assert!(
        matches!(&err, Err(crate::Error::Prover(m)) if m.contains("the light walk")),
        "{err:?}"
    );
}

/// `program`'s lean windows at `window` cycles, to the end.
fn lean_windows(program: &Elf, window: usize) -> Vec<executor::vm::history::WindowHistory> {
    use executor::vm::fast::FastExecutor;
    let mut fast = FastExecutor::new(program, Vec::new()).unwrap();
    fast.enable_word_history().unwrap();
    fast.enable_lean_stream().unwrap();
    let mut out = Vec::new();
    while let Some((_, h)) = fast.resume_word_lean(window).unwrap() {
        out.push(h);
    }
    out
}

/// The access indices of the lean window `h`'s loads and stores whose word,
/// changed, changes the replay's registers at the window's end (`reaches`)
/// or leaves them (and the replay runs to its end).
fn bent_accesses(
    program: &Elf,
    h: &executor::vm::history::WindowHistory,
    reaches: bool,
) -> Vec<usize> {
    use executor::vm::fast::Replay;
    let replay = Replay::new(program).unwrap();
    let end = |h: &executor::vm::history::WindowHistory| {
        let mut at = Replay::start(h).ok()?;
        let mut out = Vec::new();
        replay.run(h, &mut at, h.cycles, &mut out).ok()?;
        Replay::finish(h, &at).ok()?;
        Some(at.regs)
    };
    let honest = end(h).expect("the honest window replays");
    (0..h.word_accesses.len())
        .filter(|&a| {
            let mut bent = h.clone();
            bent.word_accesses[a].word ^= 0x0101_0101_0101_0101;
            end(&bent).is_some_and(|regs| (regs != honest) == reaches)
        })
        .collect()
}

/// ★ Mutants of a lean window's records (a window that is not the run's
/// last), each refused by its walker: an access record at another address,
/// an access record dropped, an ECALL placed a cycle later or after another
/// count of accesses; and a changed word on a live load, one whose bent value
/// reaches the window's end registers (the walk's end is checked against the
/// next window's start). Known limit, completeness only: a changed word whose
/// value never reaches the window's end or a later address is not seen by
/// the walker; the tables it builds then leave the memory argument
/// unbalanced (the word read is not the one stored), so the verifier rejects
/// the proof (the last case).
#[test]
fn a_lean_windows_tampered_records_are_refused() {
    let _ecrec = crate::tables::trace_builder::ecrec_forge_free();
    const WINDOW: usize = 33;
    for name in ["fused_mem_cases", "test_copy_ecalls"] {
        let (program, _) = run(name);
        let windows = lean_windows(&program, WINDOW);
        let body = &windows[..windows.len() - 1];
        let walk = |k: usize, tamper: Tamper<'_>| {
            let run = StreamRun {
                lean: true,
                replay: true,
                tamper: Some((k, tamper)),
                ..StreamRun::default()
            };
            word_stream_build(&program, WINDOW, 2, 2, run).map(|_| ())
        };
        // A window with a live load (the loads and stores fixture has one).
        let live = body
            .iter()
            .enumerate()
            .find_map(|(k, h)| bent_accesses(&program, h, true).first().map(|&a| (k, a)));
        assert!(
            live.is_some() || name != "fused_mem_cases",
            "{name}: a live load"
        );
        if let Some((k, live)) = live {
            assert!(
                walk(k, &|h| h.word_accesses[0].addr ^= 8).is_err(),
                "{name}: an access at another address"
            );
            assert!(
                walk(k, &|h| {
                    h.word_accesses.pop();
                })
                .is_err(),
                "{name}: an access record dropped"
            );
            assert!(
                walk(k, &move |h| h.word_accesses[live].word ^=
                    0x0101_0101_0101_0101)
                .is_err(),
                "{name}: a changed word on a live load"
            );
        }
        // The known limit: a changed word whose value dies in its window.
        if let Some((k, dead)) = body
            .iter()
            .enumerate()
            .find_map(|(k, h)| bent_accesses(&program, h, false).first().map(|&a| (k, a)))
        {
            let run = StreamRun {
                lean: true,
                replay: true,
                tamper: Some((k, &move |h| {
                    h.word_accesses[dead].word ^= 0x0101_0101_0101_0101
                })),
                ..StreamRun::default()
            };
            if let Ok((mut traces, _)) = word_stream_build(&program, WINDOW, 2, 2, run) {
                let left = left(&program, &mut traces, RegisterFormat::InRow);
                assert!(
                    !left.is_empty(),
                    "{name}: a dead load's changed word balanced every bus"
                );
            }
        }
        // A window with an ECALL.
        if let Some(k) = body.iter().position(|h| !h.ecall_at.is_empty()) {
            assert!(
                walk(k, &|h| h.ecall_at[0] += 1).is_err(),
                "{name}: an ECALL a cycle later"
            );
            assert!(
                walk(k, &|h| h.accesses_before[0] += 1).is_err(),
                "{name}: an ECALL after another count of accesses"
            );
        } else {
            assert_ne!(name, "test_copy_ecalls", "a window with an ECALL");
        }
    }
}

/// ★ A run that fetches an instruction from memory (a jump into a data
/// segment): today's record mode refuses its window (the prover's DECODE
/// table holds no instruction there: `MissingInstruction`), and the lean
/// stream refuses it at the executor, with an error, not a panic.
#[test]
fn a_window_fetching_an_instruction_from_memory_is_refused_by_both_streams() {
    use executor::elf::Segment;
    use executor::vm::fast::FastExecutor;
    // I-type: imm[11:0] rs1 funct3 rd opcode.
    let i = |opcode: u32, imm: i32, rs1: u32, funct3: u32, rd: u32| {
        ((imm as u32 & 0xFFF) << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | opcode
    };
    let data = 0x200u64;
    let program = Elf {
        entry_point: 0x1000,
        data: vec![
            Segment {
                base_addr: data,
                // addi x6, x0, 7; addi x17, x0, 93; ecall (halt)
                values: vec![i(0x13, 7, 0, 0, 6), i(0x13, 93, 0, 0, 17), 0x73],
                is_executable: false,
            },
            Segment {
                base_addr: 0x1000,
                // addi x5, x0, 0x200; jalr x0, 0(x5)
                values: vec![i(0x13, data as i32, 0, 0, 5), i(0x67, 0, 5, 0, 0)],
                is_executable: true,
            },
        ],
    };
    // The engine runs it.
    let mut fast = FastExecutor::new(&program, Vec::new()).unwrap();
    fast.enable_word_history().unwrap();
    while fast.resume_word_history(4).unwrap().is_some() {}
    assert_eq!(fast.registers().read(6).unwrap(), 7, "the fetched addi ran");
    // Today's record mode: refused at the walk.
    let full = word_stream_build(&program, 4, 1, 2, StreamRun::default()).map(|_| ());
    assert!(
        matches!(&full, Err(crate::Error::MissingInstruction(pc)) if (data..data + 12).contains(pc)),
        "{full:?}"
    );
    // The lean stream: refused at the executor.
    let mut lean = FastExecutor::new(&program, Vec::new()).unwrap();
    lean.enable_word_history().unwrap();
    lean.enable_lean_stream().unwrap();
    let refused = loop {
        match lean.resume_word_lean(4) {
            Ok(Some(_)) => continue,
            other => break other.map(|o| o.map(|(c, _)| c)),
        }
    };
    assert!(
        matches!(&refused, Err(e) if format!("{e}").contains("outside the program's decoded segments")),
        "{refused:?}"
    );
}
