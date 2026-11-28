//! ★ The fused memory format (`tables::fused_memory`), on the builders'
//! traces, with no proof: MEM_LOAD's and MEM_STORE's rows against the LOAD,
//! STORE and MEMW_A rows they replace, and each soundness check of the two
//! tables shown load-bearing by a forgery.
//!
//! The honest fused build of a run ([`fuse`]) is its split build (the block's
//! production register format and MEMW_A order) with every CPU load or store
//! whose MEMW message MEMW_A received turned into one fused row, and that
//! LOAD (STORE) row and MEMW_A row taken out. Every inner bus of it balances,
//! the Memory bus included (PAGE closes it), and every fused row carries the
//! very messages its pair sent outward ([`each_fused_row_sends_what_its_pair_sent_outward`]).
//!
//! A forgery is a malicious prover's edit of an honest fused build, read on
//! every inner bus ([`left`]). The byte, half-word and sign-bit checks are
//! BITWISE's: its tuples are preprocessed and only its multiplicities are the
//! prover's, so a check left over is refused only when no BITWISE row answers
//! it. Each forgery a check refuses also runs with that one check taken out of
//! its table (a control mutant) and then balances every bus, so the check is
//! what refuses it. A forgery a constraint refuses balances every bus and
//! breaks that constraint alone.

use std::collections::{BTreeMap, HashMap};

use executor::elf::Elf;
use executor::vm::logs::Log;
use stark::constraints::builder::ConstraintSet;
use stark::lookup::{BusInteraction, Multiplicity};
use stark::proof::options::ProofOptions;
use stark::trace::TraceTable;

use super::Traces;
use super::bus_net_tests::multiplicity;
use super::finish_tests::{run, windowed};
use super::in_row_forgery_tests::with_verifier_files;
use crate::tables::MaxRowsConfig;
use crate::tables::gpack::TraceForm;
use crate::tables::load::{self, LoadOperation, cols as ld};
use crate::tables::mem_load::{self, MemLoadConstraints, MemLoadRow, cols as ml};
use crate::tables::mem_store::{self, MemStoreConstraints, MemStoreRow, cols as ms};
use crate::tables::memw_aligned::{self, AlignedRow, MemwAlignedOrder, cols as ma};
use crate::tables::page::cols as pg;
use crate::tables::register::RegisterFormat;
use crate::tables::store::{self, StoreOperation, cols as st};
use crate::tables::types::{BusId, FE, GoldilocksExtension, GoldilocksField};
use crate::test_utils::VmAir;

type Table = TraceTable<GoldilocksField, GoldilocksExtension>;

/// The field's order: `−k` reads as `P − k`.
const P: u64 = 0xFFFF_FFFF_0000_0001;
const TWO_16: u64 = 1 << 16;

// ---- the honest fused build -------------------------------------------------------------------------------------------

/// The split build of `program`'s run: windows of 1000 cycles, the small
/// chunks, every table wide, the registers in the CPU row and MEMW_A in the
/// limbs' form (the block's production formats).
fn split_build(program: &Elf, logs: &[Log]) -> Traces {
    windowed(program, logs, &MaxRowsConfig::small(), 1000, &|b| {
        b.register_format(RegisterFormat::InRow)
            .expect("before any window")
            .memw_aligned_order(MemwAlignedOrder::Limbs)
            .expect("before any window")
    })
}

fn get(t: &Table, row: usize, col: usize) -> u64 {
    t.get_main(row, col).canonical_u64()
}

fn set(t: &mut Table, row: usize, col: usize, value: u64) {
    assert!(t.narrow_main().is_none(), "the forgeries edit wide tables");
    t.main_table.set(row, col, FE::from(value));
}

fn width_of(t: &Table, row: usize, flags: [usize; 3]) -> u8 {
    match flags.map(|c| get(t, row, c)) {
        [1, 0, 0] => 2,
        [0, 1, 0] => 4,
        [0, 0, 1] => 8,
        _ => 1,
    }
}

/// Eight byte columns of a row, little-endian.
fn bytes_of(t: &Table, row: usize, columns: [usize; 8]) -> u64 {
    columns
        .iter()
        .enumerate()
        .map(|(k, &c)| get(t, row, c) << (8 * k))
        .sum()
}

/// The active rows of LOAD tables as their ops.
fn load_ops(tables: &[Table]) -> Vec<LoadOperation> {
    let mut ops = Vec::new();
    for t in tables {
        for r in (0..t.num_rows()).filter(|&r| get(t, r, ld::MU) == 1) {
            let res = bytes_of(t, r, ld::RES).to_le_bytes();
            ops.push(LoadOperation::new(
                get(t, r, ld::BASE_ADDRESS_0) | get(t, r, ld::BASE_ADDRESS_1) << 32,
                get(t, r, ld::TIMESTAMP_0) | get(t, r, ld::TIMESTAMP_1) << 32,
                width_of(t, r, [ld::READ2, ld::READ4, ld::READ8]),
                get(t, r, ld::SIGNED) == 1,
                res,
            ));
        }
    }
    ops
}

/// The active rows of STORE tables as their ops.
fn store_ops(tables: &[Table]) -> Vec<StoreOperation> {
    let mut ops = Vec::new();
    for t in tables {
        for r in (0..t.num_rows()).filter(|&r| get(t, r, st::MU) == 1) {
            ops.push(StoreOperation::new(
                get(t, r, st::BASE_ADDRESS_0) | get(t, r, st::BASE_ADDRESS_1) << 32,
                get(t, r, st::TIMESTAMP_0) | get(t, r, st::TIMESTAMP_1) << 32,
                bytes_of(t, r, st::VALUE),
                width_of(t, r, [st::WRITE2, st::WRITE4, st::WRITE8]),
            ));
        }
    }
    ops
}

/// The active rows of MEMW_A tables in the limbs' form as their rows.
fn aligned_rows(tables: &[Table]) -> Vec<AlignedRow> {
    let mut rows = Vec::new();
    for t in tables {
        for r in 0..t.num_rows() {
            let (read, write) = (get(t, r, ma::MU_READ), get(t, r, ma::MU_WRITE));
            if read + write == 0 {
                continue;
            }
            let is_register = get(t, r, ma::IS_REGISTER) == 1;
            let pack = |columns: [usize; 8]| {
                if is_register {
                    get(t, r, columns[0]) | get(t, r, columns[1]) << 32
                } else {
                    bytes_of(t, r, columns)
                }
            };
            let ts = get(t, r, ma::TIMESTAMP_0) | get(t, r, ma::TIMESTAMP_1) << 32;
            let step = 1 + get(t, r, ma::STEP_0) + (get(t, r, ma::STEP_1) << 16);
            let address = get(t, r, ma::BASE_ADDRESS[0])
                | get(t, r, ma::BASE_ADDRESS[1]) << 16
                | get(t, r, ma::BASE_ADDRESS[2]) << 32;
            rows.push(
                AlignedRow::from_parts(
                    address,
                    ts,
                    ts - step,
                    pack(ma::VALUE),
                    pack(ma::OLD),
                    width_of(t, r, [ma::WRITE2, ma::WRITE4, ma::WRITE8]),
                    read == 1,
                    is_register,
                )
                .in_order(MemwAlignedOrder::Limbs),
            );
        }
    }
    rows
}

/// What [`fuse`] joined: each fused row with the LOAD (STORE) op and the
/// MEMW_A row it replaces.
#[derive(Default)]
struct Joined {
    loads: Vec<(MemLoadRow, LoadOperation, AlignedRow)>,
    stores: Vec<(MemStoreRow, StoreOperation, AlignedRow)>,
}

/// `traces` (a split build) with each CPU load or store that MEMW_A answered
/// as one fused row: the honest fused build.
fn fuse(mut traces: Traces) -> (Traces, Joined) {
    let loads = load_ops(&traces.loads);
    let stores = store_ops(&traces.stores);
    let aligned = aligned_rows(&traces.memw_aligneds);
    // A memory access's MEMW_A row by its address, timestamp and kind: one a
    // CPU cycle, so the key is unique.
    let mut by_access: HashMap<(u64, u64, bool), usize> = HashMap::new();
    for (i, row) in aligned.iter().enumerate() {
        if !row.is_register() {
            let key = (row.base_address(), row.timestamp(), row.is_read());
            assert!(by_access.insert(key, i).is_none(), "two rows for {key:?}");
        }
    }
    let mut taken = vec![false; aligned.len()];
    let mut joined = Joined::default();
    let mut rest_loads = Vec::new();
    for op in loads {
        match by_access.get(&(op.base_address, op.timestamp, true)) {
            Some(&i) => {
                taken[i] = true;
                let row = MemLoadRow::new(
                    op.base_address,
                    op.timestamp,
                    aligned[i].old_timestamp(),
                    op.width,
                    op.signed,
                    op.res,
                )
                .expect("an access MEMW_A takes fits a fused row");
                joined.loads.push((row, op, aligned[i]));
            }
            None => rest_loads.push(op),
        }
    }
    let mut rest_stores = Vec::new();
    for op in stores {
        match by_access.get(&(op.base_address, op.timestamp, false)) {
            Some(&i) => {
                taken[i] = true;
                let width = match (op.write2, op.write4, op.write8) {
                    (true, _, _) => 2,
                    (_, true, _) => 4,
                    (_, _, true) => 8,
                    _ => 1,
                };
                let old = aligned[i]
                    .old()
                    .iter()
                    .enumerate()
                    .map(|(k, &b)| u64::from(b) << (8 * k))
                    .sum();
                let row = MemStoreRow::new(
                    op.base_address,
                    op.timestamp,
                    aligned[i].old_timestamp(),
                    width,
                    op.value,
                    old,
                )
                .expect("an access MEMW_A takes fits a fused row");
                joined.stores.push((row, op, aligned[i]));
            }
            None => rest_stores.push(op),
        }
    }
    let rest_aligned: Vec<AlignedRow> = aligned
        .iter()
        .zip(&taken)
        .filter(|(_, t)| !**t)
        .map(|(r, _)| *r)
        .collect();
    traces.loads = vec![load::generate_load_trace_segments_as(
        &[&rest_loads],
        TraceForm::Wide,
    )];
    traces.stores = vec![store::generate_store_trace_segments_as(
        &[&rest_stores],
        TraceForm::Wide,
    )];
    traces.memw_aligneds = vec![memw_aligned::generate_memw_aligned_trace_segments_as(
        &[&rest_aligned],
        TraceForm::Wide,
    )];
    let fused_loads: Vec<MemLoadRow> = joined.loads.iter().map(|j| j.0).collect();
    let fused_stores: Vec<MemStoreRow> = joined.stores.iter().map(|j| j.0).collect();
    traces.mem_loads = vec![mem_load::generate_mem_load_trace_segments_as(
        &[&fused_loads],
        TraceForm::Wide,
    )];
    traces.mem_stores = vec![mem_store::generate_mem_store_trace_segments_as(
        &[&fused_stores],
        TraceForm::Wide,
    )];
    (traces, joined)
}

/// The honest fused build of `logs`' run of `program`.
fn fused_build(program: &Elf, logs: &[Log]) -> (Traces, Joined) {
    fuse(split_build(program, logs))
}

// ---- reading a build's nets --------------------------------------------------------------------------------------------

/// The buses BITWISE answers that a fused row sends on.
const CHECKS: [BusId; 3] = [BusId::Msb8, BusId::AreBytes, BusId::IsHalfword];

/// The buses a block's statement closes from outside the tables.
const OUTSIDE: [BusId; 3] = [BusId::Commit, BusId::GlobalMemory, BusId::Decode];

/// Whether a BITWISE row answers `tuple` on `bus` (one of [`CHECKS`]).
fn answered(bus: BusId, tuple: &[u64]) -> bool {
    match (bus, tuple) {
        (BusId::Msb8, [x, msb]) => *x < 256 && *msb == x >> 7,
        (BusId::AreBytes, [x, y]) => *x < 256 && *y < 256,
        (BusId::IsHalfword, [x]) => *x < TWO_16,
        _ => false,
    }
}

/// What the nets of a build leave that the prover cannot cancel.
#[derive(Debug, Default)]
struct Left {
    /// Every bus but [`CHECKS`] with a tuple left over, with its tuples.
    buses: BTreeMap<u64, Vec<(Vec<u64>, i128)>>,
    /// The checks left over that no BITWISE row answers, sorted.
    unanswered: Vec<(u64, Vec<u64>)>,
}

impl Left {
    fn accepted(&self) -> bool {
        self.buses.is_empty() && self.unanswered.is_empty()
    }
}

impl std::fmt::Display for Left {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} · unanswered checks {:?}",
            super::bus_net_tests::named(&self.buses),
            self.unanswered
        )
    }
}

/// A net as a small signed count: a send left over is positive.
fn signed(v: FE) -> i128 {
    let v = v.canonical_u64();
    if v > u64::MAX / 2 {
        -((P - v) as i128)
    } else {
        v as i128
    }
}

/// A table's interactions under a test's `over` (by the table's name), else
/// its AIR's.
type Over<'a> = &'a dyn Fn(&str) -> Option<Vec<BusInteraction>>;

/// The nets of `build` under the block's AIRs at the fused format (a table's
/// interactions from `over` where it names the table), REGISTER's files the
/// verifier's.
fn left(program: &Elf, build: &mut Traces, over: Over<'_>) -> Left {
    left_with(program, build, over, false)
}

/// [`left`], or with `exact` BITWISE's multiplicities read as the build
/// counted them on every bus: then a check left over at all is unanswered
/// (the build's collectors are what is read).
fn left_with(program: &Elf, build: &mut Traces, over: Over<'_>, exact: bool) -> Left {
    use crate::test_utils::{create_memw_aligned_air_for, halted_register_commitment};
    with_verifier_files(program, build);
    let options = ProofOptions::default_test_options();
    let heights = crate::cpu_heights_from_traces(build);
    let mut airs = crate::VmAirs::new(
        program,
        &options,
        false,
        &build.page_configs,
        &build.table_counts(),
        &heights,
        None,
        true,
        None,
        halted_register_commitment(&options, program, &build.public_output_bytes).preprocessed(),
    );
    let inputs = crate::BlockAirInputs::new(&options, program, heights, &build.public_output_bytes)
        .expect("the block's inputs");
    airs.swap_in_row_registers(&options, &inputs);
    airs.memw_aligneds = (0..airs.memw_aligneds.len())
        .map(|i| {
            Box::new(
                create_memw_aligned_air_for(&options, MemwAlignedOrder::Limbs)
                    .with_name(&format!("MEMW_A[{i}]")),
            ) as VmAir
        })
        .collect();
    airs.with_fused_memory(&options, build.fused_counts());

    let checks: Vec<u64> = CHECKS.iter().map(|b| u64::from(*b)).collect();
    let outside: Vec<u64> = OUTSIDE.iter().map(|b| u64::from(*b)).collect();
    let mut net: HashMap<(u64, Vec<u64>), FE> = HashMap::new();
    for (air, trace, _) in airs.air_trace_pairs(build) {
        let name = air.name();
        let bitwise = name == "BITWISE" && !exact;
        let owned = over(name);
        let interactions = owned.as_deref().unwrap_or(air.bus_interactions());
        for row in 0..trace.num_rows() {
            let g = |c: usize| *trace.get_main(row, c);
            for bi in interactions {
                if outside.contains(&bi.bus_id) || (bitwise && checks.contains(&bi.bus_id)) {
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
    let mut out = Left::default();
    for ((bus, tuple), v) in net {
        if v == FE::zero() {
            continue;
        }
        if checks.contains(&bus) {
            let id = BusId::try_from(bus).expect("a bus id");
            if exact || !answered(id, &tuple) {
                out.unanswered.push((bus, tuple));
            }
        } else {
            out.buses.entry(bus).or_default().push((tuple, signed(v)));
        }
    }
    out.unanswered.sort();
    for tuples in out.buses.values_mut() {
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

fn row_of(t: &Table, r: usize) -> Vec<FE> {
    (0..t.main_table.width).map(|c| *t.get_main(r, c)).collect()
}

/// Every `(table, row, constraint)` of the fused tables that does not read
/// zero.
fn broken(build: &Traces) -> Vec<(&'static str, usize, usize, usize)> {
    let mut out = Vec::new();
    for (i, t) in build.mem_loads.iter().enumerate() {
        for r in 0..t.num_rows() {
            let values = constraint_values(&MemLoadConstraints, &row_of(t, r));
            out.extend(
                (0..values.len())
                    .filter(|&k| values[k] != FE::zero())
                    .map(|k| ("MEM_LOAD", i, r, k)),
            );
        }
    }
    for (i, t) in build.mem_stores.iter().enumerate() {
        for r in 0..t.num_rows() {
            let values = constraint_values(&MemStoreConstraints, &row_of(t, r));
            out.extend(
                (0..values.len())
                    .filter(|&k| values[k] != FE::zero())
                    .map(|k| ("MEM_STORE", i, r, k)),
            );
        }
    }
    out
}

/// The honest fused build of `name`: every inner bus balances and every
/// fused row's constraints read zero (the control each forgery is read
/// against).
fn honest(name: &str) -> (Elf, Vec<Log>, Traces) {
    let (program, logs) = run(name);
    let (mut build, _) = fused_build(&program, &logs);
    let control = left(&program, &mut build, &|_| None);
    assert!(control.accepted(), "{name}, honest: {control}");
    assert!(
        broken(&build).is_empty(),
        "{name}, honest: {:?}",
        broken(&build)
    );
    (program, logs, build)
}

// ---- finding rows -----------------------------------------------------------------------------------------------------

/// The MEM_LOAD rows of `build` (table, row) for which `matches` holds.
fn load_rows(build: &Traces, matches: impl Fn(&Table, usize) -> bool) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    for (i, t) in build.mem_loads.iter().enumerate() {
        for r in (0..t.num_rows()).filter(|&r| get(t, r, ml::MU) == 1 && matches(t, r)) {
            out.push((i, r));
        }
    }
    out
}

/// The MEM_STORE rows of `build` (table, row) for which `matches` holds.
fn store_rows(build: &Traces, matches: impl Fn(&Table, usize) -> bool) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    for (i, t) in build.mem_stores.iter().enumerate() {
        for r in (0..t.num_rows()).filter(|&r| get(t, r, ms::MU) == 1 && matches(t, r)) {
            out.push((i, r));
        }
    }
    out
}

/// The one row of `rows`.
fn one(rows: Vec<(usize, usize)>, what: &str) -> (usize, usize) {
    assert_eq!(rows.len(), 1, "one {what}: {rows:?}");
    rows[0]
}

fn address_of(t: &Table, r: usize, base: [usize; 3]) -> u64 {
    get(t, r, base[0]) | get(t, r, base[1]) << 16 | get(t, r, base[2]) << 32
}

/// The log of the cycle at CPU timestamp `ts` (`4·cycle + 4`).
fn cycle_of(ts: u64) -> usize {
    assert!(ts.is_multiple_of(4) && ts >= 4, "a cycle's timestamp: {ts}");
    (ts / 4 - 1) as usize
}

/// The row of `build`'s PAGE tables that holds `address`: (table, row).
fn page_row(build: &Traces, address: u64) -> (usize, usize) {
    for (i, config) in build.page_configs.iter().enumerate() {
        let rows = build.pages[i].num_rows() as u64;
        if let Some(offset) = address.checked_sub(config.page_base)
            && offset < rows
        {
            return (i, offset as usize);
        }
    }
    panic!("no page holds {address:#x}");
}

// ---- the tests --------------------------------------------------------------------------------------------------------

/// The programs the honest builds are read on: every width, sign and
/// alignment of the CPU's loads and stores and bytes of mixed timestamps
/// (`fused_mem_cases`), every instruction, the commit syscall's copy, KECCAK's
/// and BLAKE3's memory rows (MEMW_A's ECALL rows), a loop.
const PROGRAMS: [&str; 8] = [
    "fused_mem_cases",
    "fused_mem_one_sb",
    "all_instructions_64",
    "all_loadstore_32",
    "test_commit_split",
    "test_keccak",
    "test_blake3",
    "loop_4096",
];

/// ★ The two tables' shapes: MEM_LOAD 21 columns and 23 interactions, MEM_STORE
/// 27 and 24, each 24 GKR input slots at 2^21 rows, where a split load's LOAD
/// and MEMW_A rows take 8 + 24 and a split store's STORE and MEMW_A rows 16 +
/// 24. Neither sends a MEMW message, every column is read, and the register
/// flag of every token is the constant 0.
#[test]
fn the_fused_tables_take_their_columns_interactions_and_slots() {
    use multilinear::whir_chain::GkrPack;
    use stark::multilinear_table::batched::{ArgueShape, chunking};
    let slots = |interactions: &[BusInteraction], num_vars: usize| {
        let n = interactions.len();
        let shape = ArgueShape {
            num_vars,
            input_vars: multilinear::logup::input_layer_vars(n, num_vars),
            degree: 2,
            shifted: false,
            interactions: n,
        };
        let c = chunking(&shape, GkrPack::PRODUCTION);
        c.count << c.slot_vars
    };
    let fused_load = mem_load::bus_interactions();
    let fused_store = mem_store::bus_interactions();
    assert_eq!(
        (ml::NUM_COLUMNS, fused_load.len()),
        (21, 23),
        "MEM_LOAD's columns and interactions"
    );
    assert_eq!(
        (ms::NUM_COLUMNS, fused_store.len()),
        (27, 24),
        "MEM_STORE's columns and interactions"
    );
    let limbs = memw_aligned::bus_interactions_for(MemwAlignedOrder::Limbs);
    assert_eq!(slots(&fused_load, 21), 24);
    assert_eq!(slots(&fused_store, 21), 24);
    assert_eq!(slots(&load::bus_interactions(), 21), 8);
    assert_eq!(slots(&store::bus_interactions(), 21), 16);
    assert_eq!(slots(&limbs, 21), 24);
    for (name, interactions, width) in [
        ("MEM_LOAD", &fused_load, ml::NUM_COLUMNS),
        ("MEM_STORE", &fused_store, ms::NUM_COLUMNS),
    ] {
        assert!(
            interactions
                .iter()
                .all(|i| i.bus_id != u64::from(BusId::Memw)),
            "{name} sends no MEMW message"
        );
        let read: std::collections::BTreeSet<usize> = interactions
            .iter()
            .flat_map(BusInteraction::columns_read)
            .collect();
        // MEM_STORE's OLD bytes past the width and the sign bit's partner
        // columns are read by some interaction; every column is.
        assert_eq!(
            read.into_iter().collect::<Vec<_>>(),
            (0..width).collect::<Vec<_>>(),
            "{name} reads every column and none past its width"
        );
        let memory = u64::from(BusId::Memory);
        let tokens: Vec<&BusInteraction> =
            interactions.iter().filter(|i| i.bus_id == memory).collect();
        assert_eq!(tokens.len(), 16, "{name}: 16 byte tokens");
        let zero = |row: &[FE]| {
            tokens
                .iter()
                .all(|i| i.values[0].combine_from(|c| row[c])[0] == FE::zero())
        };
        let ones = vec![FE::one(); width];
        assert!(zero(&ones), "{name}: every token's register flag is 0");
    }
    assert_eq!(MemLoadConstraints.meta().len(), 14);
    assert_eq!(MemLoadConstraints.max_degree(), 3);
    assert_eq!(MemStoreConstraints.meta().len(), 6);
    assert_eq!(MemStoreConstraints.max_degree(), 2);
    assert_eq!(std::mem::size_of::<MemLoadRow>(), 32);
    assert_eq!(std::mem::size_of::<MemStoreRow>(), 40);
}

/// ★ The honest fused builds balance every inner bus, the Memory bus
/// included, and break no fused row's constraint; on `fused_mem_cases` the
/// accesses left split are exactly its unaligned ones and the one whose bytes
/// were last touched at different times, and every other load and store is
/// fused, so every load is a LOAD row or a fused row and no access is lost.
#[test]
fn the_honest_fused_builds_balance_every_bus() {
    for name in PROGRAMS {
        let (program, logs) = run(name);
        let split = split_build(&program, &logs);
        let (loads, stores) = (load_ops(&split.loads).len(), store_ops(&split.stores).len());
        let (mut build, joined) = fuse(split);
        let fused_loads = joined.loads.len();
        let fused_stores = joined.stores.len();
        assert_eq!(
            load_ops(&build.loads).len() + fused_loads,
            loads,
            "{name}: every load is LOAD's or fused"
        );
        assert_eq!(
            store_ops(&build.stores).len() + fused_stores,
            stores,
            "{name}: every store is STORE's or fused"
        );
        let left = left(&program, &mut build, &|_| None);
        assert!(left.accepted(), "{name}: {left}");
        assert!(broken(&build).is_empty(), "{name}: {:?}", broken(&build));
        if name == "fused_mem_cases" {
            // Unaligned: lw 2, lhu 3, ld 1; mixed: ld 32. sh 9, sd 17.
            let rest_loads: Vec<(u64, u8)> = load_ops(&build.loads)
                .iter()
                .map(|op| (op.base_address & 0x3F, op.width))
                .collect();
            let rest_stores: Vec<(u64, u8)> = store_ops(&build.stores)
                .iter()
                .map(|op| {
                    let w = if op.write8 {
                        8
                    } else if op.write4 {
                        4
                    } else if op.write2 {
                        2
                    } else {
                        1
                    };
                    (op.base_address & 0x3F, w)
                })
                .collect();
            assert_eq!(
                rest_loads.len() + rest_stores.len(),
                6,
                "{name}: split loads {rest_loads:?}, stores {rest_stores:?}"
            );
            assert_eq!(
                (fused_loads, fused_stores),
                (15, 10),
                "{name}: the aligned loads and stores of one old timestamp"
            );
            // Every width, both signs.
            let widths: std::collections::BTreeSet<(u8, bool)> = joined
                .loads
                .iter()
                .map(|(r, _, _)| (r.width(), r.signed()))
                .collect();
            assert_eq!(
                widths.len(),
                7,
                "{name}: LB LBU LH LHU LW LWU LD: {widths:?}"
            );
        }
    }
}

/// The messages a row of `interactions` sends with a nonzero multiplicity:
/// `(bus, sends, tuple, multiplicity)`.
fn messages(interactions: &[BusInteraction], row: &[FE]) -> Vec<(u64, bool, Vec<u64>, u64)> {
    let g = |c: usize| row[c];
    let mut out: Vec<(u64, bool, Vec<u64>, u64)> = interactions
        .iter()
        .filter_map(|i| {
            let m = multiplicity(&i.multiplicity, &g);
            (m != FE::zero()).then(|| {
                let tuple = i
                    .values
                    .iter()
                    .flat_map(|v| v.combine_from(g))
                    .map(|x: FE| x.canonical_u64())
                    .collect();
                (i.bus_id, i.is_sender, tuple, m.canonical_u64())
            })
        })
        .collect();
    out.sort();
    out
}

/// A one-row table's row 0.
fn first_row(t: Table) -> Vec<FE> {
    row_of(&t, 0)
}

/// `split` less its MEMW messages, which must be one send and one receive of
/// the same tuple: the hop the fused row does away with.
fn without_the_hop(mut split: Vec<(u64, bool, Vec<u64>, u64)>) -> Vec<(u64, bool, Vec<u64>, u64)> {
    let memw = u64::from(BusId::Memw);
    let hop: Vec<_> = split.iter().filter(|m| m.0 == memw).cloned().collect();
    assert_eq!(hop.len(), 2, "one MEMW send and one receive: {hop:?}");
    assert!(hop[0].1 != hop[1].1, "a send and a receive: {hop:?}");
    assert_eq!(
        (&hop[0].2, hop[0].3),
        (&hop[1].2, hop[1].3),
        "the hop's two ends carry one tuple"
    );
    split.retain(|m| m.0 != memw);
    split
}

/// ★ Each fused row sends what its pair sent outward. On every honest build,
/// a fused load's messages are, as a multiset, its LOAD row's and its MEMW_A
/// row's less their MEMW message, whose two ends carry one tuple; a fused
/// store's likewise, except that its eight single-byte checks are four pairs
/// of the same eight bytes.
#[test]
fn each_fused_row_sends_what_its_pair_sent_outward() {
    let load_air = load::bus_interactions();
    let store_air = store::bus_interactions();
    let aligned_air = memw_aligned::bus_interactions_for(MemwAlignedOrder::Limbs);
    let fused_load_air = mem_load::bus_interactions();
    let fused_store_air = mem_store::bus_interactions();
    let are_bytes = u64::from(BusId::AreBytes);
    let (mut loads_seen, mut stores_seen) = (0, 0);
    for name in PROGRAMS {
        let (program, logs) = run(name);
        let (_, joined) = fused_build(&program, &logs);
        for (fused, op, aligned) in &joined.loads {
            let row = first_row(mem_load::generate_mem_load_trace_segments_as(
                &[&[*fused]],
                TraceForm::Wide,
            ));
            let load_row = first_row(load::generate_load_trace_segments_as(
                &[std::slice::from_ref(op)],
                TraceForm::Wide,
            ));
            let aligned_row = first_row(memw_aligned::generate_memw_aligned_trace_segments_as(
                &[&[*aligned]],
                TraceForm::Wide,
            ));
            let mut split = messages(&load_air, &load_row);
            split.extend(messages(&aligned_air, &aligned_row));
            split.sort();
            assert_eq!(
                messages(&fused_load_air, &row),
                without_the_hop(split),
                "{name}: {op:?}"
            );
            loads_seen += 1;
        }
        for (fused, op, aligned) in &joined.stores {
            let row = first_row(mem_store::generate_mem_store_trace_segments_as(
                &[&[*fused]],
                TraceForm::Wide,
            ));
            let store_row = first_row(store::generate_store_trace_segments_as(
                &[std::slice::from_ref(op)],
                TraceForm::Wide,
            ));
            let aligned_row = first_row(memw_aligned::generate_memw_aligned_trace_segments_as(
                &[&[*aligned]],
                TraceForm::Wide,
            ));
            let mut split = messages(&store_air, &store_row);
            split.extend(messages(&aligned_air, &aligned_row));
            split.sort();
            let split = without_the_hop(split);
            let fused_messages = messages(&fused_store_air, &row);
            let (split_checks, split_rest): (Vec<_>, Vec<_>) =
                split.into_iter().partition(|m| m.0 == are_bytes);
            let (fused_checks, fused_rest): (Vec<_>, Vec<_>) =
                fused_messages.into_iter().partition(|m| m.0 == are_bytes);
            assert_eq!(fused_rest, split_rest, "{name}: {op:?}");
            // Eight single-byte checks, `(byte, 0)`; four pairs.
            let mut singles: Vec<u64> = Vec::new();
            for (_, sends, tuple, m) in &split_checks {
                assert!(*sends && tuple[1] == 0, "{name}: a single-byte check");
                singles.extend(std::iter::repeat_n(tuple[0], *m as usize));
            }
            let mut pairs: Vec<u64> = Vec::new();
            for (_, sends, tuple, m) in &fused_checks {
                assert!(*sends, "{name}: a check sent");
                for _ in 0..*m {
                    pairs.extend([tuple[0], tuple[1]]);
                }
            }
            singles.sort_unstable();
            pairs.sort_unstable();
            assert_eq!(singles.len(), 8, "{name}: eight bytes checked");
            assert_eq!(pairs, singles, "{name}: the same eight bytes: {op:?}");
            stores_seen += 1;
        }
    }
    assert!(
        loads_seen >= 30 && stores_seen >= 30,
        "{loads_seen} {stores_seen}"
    );
}

/// `table`'s interactions less the ones `drop` picks, which must be `count`.
fn without(
    interactions: Vec<BusInteraction>,
    count: usize,
    drop: impl Fn(usize, &BusInteraction) -> bool,
) -> Vec<BusInteraction> {
    let kept: Vec<BusInteraction> = interactions
        .iter()
        .enumerate()
        .filter(|(k, i)| !drop(*k, i))
        .map(|(_, i)| i.clone())
        .collect();
    assert_eq!(
        kept.len() + count,
        interactions.len(),
        "the mutant drops {count}"
    );
    kept
}

/// MEM_LOAD tables' interactions from `list`, the others' their own.
fn over_mem_load(list: Vec<BusInteraction>) -> impl Fn(&str) -> Option<Vec<BusInteraction>> {
    move |name: &str| name.starts_with("MEM_LOAD[").then(|| list.clone())
}

/// MEM_STORE tables' interactions from `list`, the others' their own.
fn over_mem_store(list: Vec<BusInteraction>) -> impl Fn(&str) -> Option<Vec<BusInteraction>> {
    move |name: &str| name.starts_with("MEM_STORE[").then(|| list.clone())
}

/// A program's run with the load at the CPU timestamp `ts` claiming `value`:
/// the malicious prover's execution, whose fused build the tests then edit.
fn claimed(program: &Elf, mut logs: Vec<Log>, ts: u64, value: u64) -> (Traces, Joined) {
    logs[cycle_of(ts)].dst_val = value;
    fused_build(program, &logs)
}

/// The claimed load at `ts` of `build`: its (table, row).
fn claimed_row(build: &Traces, ts: u64) -> (usize, usize) {
    one(
        load_rows(build, |t, r| get(t, r, ml::TIMESTAMP_0) == ts),
        "the claimed load",
    )
}

/// The honest fused load of `fused_mem_cases` that `matches`, and its
/// timestamp.
fn load_at(honest: &Traces, what: &str, matches: impl Fn(&Table, usize) -> bool) -> u64 {
    let (i, r) = one(load_rows(honest, matches), what);
    get(&honest.mem_loads[i], r, ml::TIMESTAMP_0)
}

fn is_byte_load(t: &Table, r: usize, signed: u64, byte: u64) -> bool {
    get(t, r, ml::SIGNED) == signed
        && width_of(t, r, [ml::READ2, ml::READ4, ml::READ8]) == 1
        && get(t, r, ml::RES[0]) == byte
}

/// ★ One dispatch, one receiver, `MU` a bit. `lb` of 0x05 claimed as
/// 0xFF…05: the fused row's sign bit set and its result extended (its
/// constraints hold) leaves `MSB8[0x05] → 1` alone, a tuple BITWISE does not
/// hold. A `MU = −1 / +1` pair of zero-extended byte rows of 0x05 (a table of
/// their own), the first with the sign bit and the second without, cancels on
/// the dispatch and the tokens and nets `MSB8[0x05] → 1` once against it:
/// every bus balances, and the `MU = −1` row's `MU` bit constraint alone
/// refuses the build.
#[test]
fn a_sign_bit_forged_through_a_minus_one_row_breaks_the_mu_bit_constraint_alone() {
    let (program, logs, honest) = honest("fused_mem_cases");
    let ts = load_at(&honest, "lb of 0x05", |t, r| is_byte_load(t, r, 1, 0x05));
    let (mut build, _) = claimed(&program, logs, ts, 0xFFFF_FFFF_FFFF_FF05);
    let (i, r) = claimed_row(&build, ts);
    for &c in &ml::RES[1..] {
        set(&mut build.mem_loads[i], r, c, 0xFF);
    }
    set(&mut build.mem_loads[i], r, ml::SIGN_BIT, 1);
    let alone = left(&program, &mut build, &|_| None);
    assert!(alone.buses.is_empty(), "{alone}");
    assert_eq!(alone.unanswered, vec![(u64::from(BusId::Msb8), vec![5, 1])]);
    let mut pair = mem_load::generate_mem_load_trace_segments_as(&[], TraceForm::Wide);
    for (row, (mu, sign_bit)) in [(P - 1, 1), (1, 0)].into_iter().enumerate() {
        set(&mut pair, row, ml::RES[0], 0x05);
        set(&mut pair, row, ml::MU, mu);
        set(&mut pair, row, ml::SIGN_BIT, sign_bit);
    }
    build.mem_loads.push(pair);
    let forged = left(&program, &mut build, &|_| None);
    assert!(
        forged.accepted(),
        "the forgery balances every bus: {forged}"
    );
    let pair = build.mem_loads.len() - 1;
    assert_eq!(
        broken(&build),
        vec![("MEM_LOAD", pair, 0, 13)],
        "the MU bit constraint alone, on the MU = −1 row"
    );
}

/// ★ One dispatch, one receiver, `MU` a bit, with a real CPU load at `MU =
/// 2`. `lb` of 0x81 claimed as 0x81, zero-extended (sign bit clear, upper
/// bytes 0; its constraints hold), leaves `MSB8[0x81] → 0` alone. The next
/// instruction's `lbu` of the same byte, its fused row at `MU = 2`, and a twin
/// of that row at `MU = −1` with the sign bit clear (a table of its own) net
/// the `lbu`'s dispatch and tokens once, as the CPU sent them, and
/// `MSB8[0x81] → 0` once against the claimed load: every bus balances, and the
/// `MU` bit constraint alone refuses the build, on those two rows.
#[test]
fn a_sign_bit_forged_through_a_load_at_mu_two_breaks_the_mu_bit_constraint_alone() {
    let (program, logs, honest) = honest("fused_mem_cases");
    let ts = load_at(&honest, "lb of 0x81", |t, r| is_byte_load(t, r, 1, 0x81));
    let lbu_ts = load_at(&honest, "lbu of 0x81", |t, r| is_byte_load(t, r, 0, 0x81));
    let (mut build, _) = claimed(&program, logs, ts, 0x81);
    let (i, r) = claimed_row(&build, ts);
    for &c in &ml::RES[1..] {
        set(&mut build.mem_loads[i], r, c, 0);
    }
    set(&mut build.mem_loads[i], r, ml::SIGN_BIT, 0);
    let alone = left(&program, &mut build, &|_| None);
    assert!(alone.buses.is_empty(), "{alone}");
    assert_eq!(
        alone.unanswered,
        vec![(u64::from(BusId::Msb8), vec![0x81, 0])]
    );
    let (j, s) = claimed_row(&build, lbu_ts);
    set(&mut build.mem_loads[j], s, ml::MU, 2);
    let mut twin = mem_load::generate_mem_load_trace_segments_as(&[], TraceForm::Wide);
    for c in 0..ml::NUM_COLUMNS {
        set(&mut twin, 0, c, get(&build.mem_loads[j], s, c));
    }
    set(&mut twin, 0, ml::MU, P - 1);
    set(&mut twin, 0, ml::SIGN_BIT, 0);
    build.mem_loads.push(twin);
    let forged = left(&program, &mut build, &|_| None);
    assert!(
        forged.accepted(),
        "the forgery balances every bus: {forged}"
    );
    let twin = build.mem_loads.len() - 1;
    let mut want = vec![("MEM_LOAD", j, s, 13), ("MEM_LOAD", twin, 0, 13)];
    want.sort();
    let mut got = broken(&build);
    got.sort();
    assert_eq!(got, want, "the MU bit constraint alone, on the two rows");
}

/// ★ Byte multiplicities by width. In a program whose one byte store writes
/// 0x34 over a doubleword of zeros, the next byte read back claimed as 0x12
/// (the stored register's byte 1) needs the byte store to have written byte
/// 1 too: with the read moved onto the store's timestamp, the store's byte-1
/// tokens are missing from the Memory bus (the doubleword's zero is never
/// taken, the claimed 0x12 never made). With the store's byte 1 at its row's
/// `MU` (a control mutant), the forgery balances every bus.
#[test]
fn a_byte_store_that_also_writes_byte_one_is_refused_on_the_memory_bus() {
    let (program, logs, honest) = honest("fused_mem_one_sb");
    let sb = one(
        store_rows(&honest, |t, r| {
            width_of(t, r, [ms::WRITE2, ms::WRITE4, ms::WRITE8]) == 1
        }),
        "the byte store",
    );
    let (sb_ts, sb_address) = {
        let t = &honest.mem_stores[sb.0];
        (
            get(t, sb.1, ms::TIMESTAMP_0),
            address_of(t, sb.1, ms::BASE_ADDRESS),
        )
    };
    let lbu = one(
        load_rows(&honest, |t, r| {
            address_of(t, r, ml::BASE_ADDRESS) == sb_address + 1
        }),
        "the byte read back",
    );
    let ts = get(&honest.mem_loads[lbu.0], lbu.1, ml::TIMESTAMP_0);
    let (mut build, _) = claimed(&program, logs, ts, 0x12);
    let (i, r) = one(
        load_rows(&build, |t, r| get(t, r, ml::TIMESTAMP_0) == ts),
        "the claimed read",
    );
    // The read takes its byte from the store's timestamp.
    let step = ts - sb_ts - 1;
    set(&mut build.mem_loads[i], r, ml::STEP_0, step & 0xFFFF);
    set(&mut build.mem_loads[i], r, ml::STEP_1, step >> 16);
    let forged = left(&program, &mut build, &|_| None);
    assert!(forged.unanswered.is_empty(), "{forged}");
    let memory = forged
        .buses
        .get(&u64::from(BusId::Memory))
        .expect("the Memory bus is left");
    assert_eq!(forged.buses.len(), 1, "the Memory bus alone: {forged}");
    assert_eq!(
        memory.len(),
        2,
        "the zero never taken, the 0x12 never made: {forged}"
    );
    let byte_one = (sb_address + 1) & 0xFFFF_FFFF;
    assert!(memory.iter().all(|(t, _)| t[1] == byte_one), "{forged}");
    let byte_one_at_mu = mem_store::bus_interactions()
        .into_iter()
        .enumerate()
        .map(|(k, mut i)| {
            if matches!(k, 5 | 6) {
                assert_eq!(
                    i.multiplicity,
                    Multiplicity::Sum3(ms::WRITE2, ms::WRITE4, ms::WRITE8)
                );
                i.multiplicity = Multiplicity::Column(ms::MU);
            }
            i
        })
        .collect();
    let control = left(&program, &mut build, &over_mem_store(byte_one_at_mu));
    assert!(
        control.accepted(),
        "with byte 1 at MU the forgery balances: {control}"
    );
    assert!(broken(&build).is_empty(), "{:?}", broken(&build));
}

/// ★ The step rule, `old_ts = ts − 1 − STEP_0 − 2^16·STEP_1`. A byte read
/// claimed as 9 where memory holds 7 consumes and produces its own token,
/// `old_ts = ts`, so the two cancel and the claimed byte comes from nowhere;
/// the byte's store goes on to the page's final token. That needs the step
/// less one to be −1: through the low limb, `STEP_0 = p − 1`, or through the
/// high one, `STEP_0 = 2^16 − 1` and `STEP_1 = p − 1`. Each leaves exactly
/// its limb's half-word check unanswered; with that check out of MEM_LOAD (a
/// control mutant) each balances every bus.
#[test]
fn a_read_consuming_its_own_token_is_refused_by_each_limbs_check() {
    let (program, logs, honest) = honest("fused_mem_cases");
    let sb_rows = store_rows(&honest, |t, r| {
        width_of(t, r, [ms::WRITE2, ms::WRITE4, ms::WRITE8]) == 1 && get(t, r, ms::VALUE[0]) == 7
    });
    let (sb_ts, address) = {
        let (i, r) = one(sb_rows, "the store of 7");
        let t = &honest.mem_stores[i];
        (
            get(t, r, ms::TIMESTAMP_0),
            address_of(t, r, ms::BASE_ADDRESS),
        )
    };
    let read = one(
        load_rows(&honest, |t, r| {
            address_of(t, r, ml::BASE_ADDRESS) == address
        }),
        "the read of 7",
    );
    let ts = get(&honest.mem_loads[read.0], read.1, ml::TIMESTAMP_0);
    for (limb, steps) in [(1usize, (P - 1, 0)), (2, (TWO_16 - 1, P - 1))] {
        let (mut build, _) = claimed(&program, logs.clone(), ts, 9);
        let (i, r) = one(
            load_rows(&build, |t, r| get(t, r, ml::TIMESTAMP_0) == ts),
            "the claimed read",
        );
        set(&mut build.mem_loads[i], r, ml::STEP_0, steps.0);
        set(&mut build.mem_loads[i], r, ml::STEP_1, steps.1);
        let (page, row) = page_row(&build, address);
        set(&mut build.pages[page], row, pg::FINI, 7);
        set(&mut build.pages[page], row, pg::TIMESTAMP_LO, sb_ts);
        let forged = left(&program, &mut build, &|_| None);
        assert!(forged.buses.is_empty(), "limb {limb}: {forged}");
        assert_eq!(
            forged.unanswered,
            vec![(u64::from(BusId::IsHalfword), vec![P - 1])],
            "limb {limb}: its check alone"
        );
        let mutant = without(mem_load::bus_interactions(), 1, |k, _| k == limb);
        let control = left(&program, &mut build, &over_mem_load(mutant));
        assert!(
            control.accepted(),
            "limb {limb}: without its check: {control}"
        );
        assert!(broken(&build).is_empty(), "{:?}", broken(&build));
    }
}

/// ★ A store's value bytes are bytes. A doubleword store, read back and
/// overwritten: the store's bytes 6 and 7 as `b6 + 256` and `b7 − 1` keep the
/// register's two words, so the dispatch balances, and carry the non-byte
/// `b6 + 256` into memory; the read takes and gives back the same tokens (an
/// 8-byte read checks no sign bit) and the overwrite consumes them as its old
/// bytes, which no check reads. Only the `(6, 7)` pair's byte check is left
/// unanswered; with it out of MEM_STORE (a control mutant) every bus balances.
#[test]
fn a_store_value_byte_past_255_is_refused_by_its_pair_check() {
    let (program, _, mut build) = honest("fused_mem_cases");
    let first = one(
        store_rows(&build, |t, r| {
            width_of(t, r, [ms::WRITE2, ms::WRITE4, ms::WRITE8]) == 8
                && bytes_of(t, r, ms::VALUE) == 0x0102_0304_0506_0708
        }),
        "the doubleword store",
    );
    let address = address_of(&build.mem_stores[first.0], first.1, ms::BASE_ADDRESS);
    let read = one(
        load_rows(&build, |t, r| {
            address_of(t, r, ml::BASE_ADDRESS) == address
                && width_of(t, r, [ml::READ2, ml::READ4, ml::READ8]) == 8
        }),
        "the read back",
    );
    let overwrite = one(
        store_rows(&build, |t, r| {
            address_of(t, r, ms::BASE_ADDRESS) == address && bytes_of(t, r, ms::VALUE) == 0
        }),
        "the overwrite",
    );
    let (b6, b7) = (0x02, 0x01);
    set(
        &mut build.mem_stores[first.0],
        first.1,
        ms::VALUE[6],
        b6 + 256,
    );
    set(
        &mut build.mem_stores[first.0],
        first.1,
        ms::VALUE[7],
        b7 - 1,
    );
    set(&mut build.mem_loads[read.0], read.1, ml::RES[6], b6 + 256);
    set(&mut build.mem_loads[read.0], read.1, ml::RES[7], b7 - 1);
    set(
        &mut build.mem_stores[overwrite.0],
        overwrite.1,
        ms::OLD[6],
        b6 + 256,
    );
    set(
        &mut build.mem_stores[overwrite.0],
        overwrite.1,
        ms::OLD[7],
        b7 - 1,
    );
    let forged = left(&program, &mut build, &|_| None);
    assert!(forged.buses.is_empty(), "{forged}");
    assert_eq!(
        forged.unanswered,
        vec![(u64::from(BusId::AreBytes), vec![b6 + 256, b7 - 1])],
        "the (6, 7) pair's check alone"
    );
    let mutant = without(mem_store::bus_interactions(), 1, |k, _| k == 23);
    let control = left(&program, &mut build, &over_mem_store(mutant));
    assert!(control.accepted(), "without the (6, 7) check: {control}");
    assert!(broken(&build).is_empty(), "{:?}", broken(&build));
}

/// ★ Sign extension. `lb` of 0x80 claimed as 0x80, zero-extended: the fused
/// row's sign bit clear and its result's upper bytes 0 (the extension
/// constraints hold, `signed · 0 · 255 = 0`) leaves exactly `MSB8[0x80] → 0`,
/// which BITWISE does not hold. With the one-byte sign-bit send out of
/// MEM_LOAD (a control mutant) every bus balances.
#[test]
fn a_sign_bit_cleared_on_a_negative_byte_is_refused_by_its_msb8_check() {
    let (program, logs, honest) = honest("fused_mem_cases");
    let ts = load_at(&honest, "lb of 0x80", |t, r| is_byte_load(t, r, 1, 0x80));
    let (mut build, _) = claimed(&program, logs, ts, 0x80);
    let (i, r) = claimed_row(&build, ts);
    for &c in &ml::RES[1..] {
        set(&mut build.mem_loads[i], r, c, 0);
    }
    set(&mut build.mem_loads[i], r, ml::SIGN_BIT, 0);
    assert!(
        broken(&build).is_empty(),
        "the constraints hold: {:?}",
        broken(&build)
    );
    let forged = left(&program, &mut build, &|_| None);
    assert!(forged.buses.is_empty(), "{forged}");
    assert_eq!(
        forged.unanswered,
        vec![(u64::from(BusId::Msb8), vec![0x80, 0])],
        "the one-byte sign-bit check alone"
    );
    let msb8 = u64::from(BusId::Msb8);
    let first_msb8 = mem_load::bus_interactions()
        .iter()
        .position(|i| i.bus_id == msb8)
        .expect("a sign-bit send");
    let mutant = without(mem_load::bus_interactions(), 1, |k, _| k == first_msb8);
    let control = left(&program, &mut build, &over_mem_load(mutant));
    assert!(
        control.accepted(),
        "without the one-byte sign-bit check: {control}"
    );
}

/// ★ The span check stays MEMW_A's. MEM_LOAD's and MEM_STORE's first
/// interaction is MEMW_A's span check on their own columns,
/// `IS_HALF[base_address[0] + write2 + 3·write4 + 7·write8]` at the row's
/// multiplicity, and in all three tables an access of each width whose last
/// byte is the low half's last (0xFFFF) is answered while one a byte further
/// (0x10000, an access across the half) is not.
#[test]
fn the_span_check_is_memw_as_in_both_fused_tables() {
    let halfword = u64::from(BusId::IsHalfword);
    // (name, interactions, width, base_address[0], the flags, the
    // multiplicity columns set to 1).
    type Case = (
        &'static str,
        Vec<BusInteraction>,
        usize,
        usize,
        [usize; 3],
        Vec<usize>,
    );
    let tables: [Case; 3] = [
        (
            "MEMW_A",
            memw_aligned::bus_interactions_for(MemwAlignedOrder::Limbs),
            ma::NUM_COLUMNS,
            ma::BASE_ADDRESS[0],
            [ma::WRITE2, ma::WRITE4, ma::WRITE8],
            vec![ma::MU_READ],
        ),
        (
            "MEM_LOAD",
            mem_load::bus_interactions(),
            ml::NUM_COLUMNS,
            ml::BASE_ADDRESS[0],
            [ml::READ2, ml::READ4, ml::READ8],
            vec![ml::MU],
        ),
        (
            "MEM_STORE",
            mem_store::bus_interactions(),
            ms::NUM_COLUMNS,
            ms::BASE_ADDRESS[0],
            [ms::WRITE2, ms::WRITE4, ms::WRITE8],
            vec![ms::MU],
        ),
    ];
    let span = |interactions: &[BusInteraction], row: &[FE]| -> Vec<u64> {
        messages(&interactions[..1], row)
            .into_iter()
            .filter(|m| m.0 == halfword && m.1)
            .map(|m| m.2[0])
            .collect()
    };
    for (name, interactions, width, base, flags, mus) in &tables {
        assert_eq!(
            interactions[0].bus_id, halfword,
            "{name}: the span check first"
        );
        for (w, flag) in [
            (1u64, None),
            (2, Some(flags[0])),
            (4, Some(flags[1])),
            (8, Some(flags[2])),
        ] {
            for (low, answered_span) in [(0x1_0000 - w, true), (0x1_0000 - w + 1, false)] {
                let mut row = vec![FE::zero(); *width];
                row[*base] = FE::from(low);
                if let Some(f) = flag {
                    row[f] = FE::one();
                }
                for &m in mus {
                    row[m] = FE::one();
                }
                let sent = span(interactions, &row);
                assert_eq!(sent, vec![low + w - 1], "{name}: width {w} at {low:#x}");
                assert_eq!(
                    sent[0] < TWO_16,
                    answered_span,
                    "{name}: width {w} at {low:#x}"
                );
            }
        }
    }
}

/// Field values for synthetic rows: bits more often than not, else any
/// element (a small linear congruential stream, so the rows are the same each
/// run).
fn synthetic_rows(width: usize, rows: usize) -> Vec<Vec<FE>> {
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = move || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        state
    };
    (0..rows)
        .map(|_| {
            (0..width)
                .map(|_| {
                    let x = next();
                    match x % 4 {
                        0 => FE::zero(),
                        1 => FE::one(),
                        2 => FE::from(x >> 56),
                        _ => FE::from(x >> 1),
                    }
                })
                .collect()
        })
        .collect()
}

/// ★ A fused row's rule is its split pair's. On synthetic rows (bits and any
/// field elements), MEM_LOAD's 14 constraints read what LOAD's read on the
/// same values in LOAD's columns, one for one, and MEM_STORE's 6 what
/// STORE's read. MEMW_A's eight, with `μ_read` the fused row's `MU` (a load)
/// or `μ_write` (a store) and the other 0, read nothing the fused row's do
/// not: each value is zero or one of the fused row's.
#[test]
fn the_fused_constraints_are_the_split_pairs_rules() {
    use crate::tables::load::LoadConstraints;
    use crate::tables::memw_aligned::MemwAlignedConstraints;
    use crate::tables::store::StoreConstraints;
    let load_pairs: Vec<(usize, usize)> = [
        (ml::READ2, ld::READ2),
        (ml::READ4, ld::READ4),
        (ml::READ8, ld::READ8),
        (ml::SIGNED, ld::SIGNED),
        (ml::SIGN_BIT, ld::SIGN_BIT),
        (ml::MU, ld::MU),
    ]
    .into_iter()
    .chain(ml::RES.into_iter().zip(ld::RES))
    .collect();
    for row in synthetic_rows(ml::NUM_COLUMNS, 4096) {
        let mut split = vec![FE::zero(); ld::NUM_COLUMNS];
        for &(f, s) in &load_pairs {
            split[s] = row[f];
        }
        let fused = constraint_values(&MemLoadConstraints, &row);
        assert_eq!(fused, constraint_values(&LoadConstraints, &split));
        let mut aligned = vec![FE::zero(); ma::NUM_COLUMNS];
        aligned[ma::WRITE2] = row[ml::READ2];
        aligned[ma::WRITE4] = row[ml::READ4];
        aligned[ma::WRITE8] = row[ml::READ8];
        aligned[ma::MU_READ] = row[ml::MU];
        for v in constraint_values(&MemwAlignedConstraints, &aligned) {
            assert!(
                v == FE::zero() || fused.contains(&v),
                "MEMW_A's {v:?} on {row:?}"
            );
        }
    }
    let store_pairs = [
        (ms::WRITE2, st::WRITE2),
        (ms::WRITE4, st::WRITE4),
        (ms::WRITE8, st::WRITE8),
        (ms::MU, st::MU),
    ];
    for row in synthetic_rows(ms::NUM_COLUMNS, 4096) {
        let mut split = vec![FE::zero(); st::NUM_COLUMNS];
        for &(f, s) in &store_pairs {
            split[s] = row[f];
        }
        let fused = constraint_values(&MemStoreConstraints, &row);
        assert_eq!(fused, constraint_values(&StoreConstraints, &split));
        let mut aligned = vec![FE::zero(); ma::NUM_COLUMNS];
        aligned[ma::WRITE2] = row[ms::WRITE2];
        aligned[ma::WRITE4] = row[ms::WRITE4];
        aligned[ma::WRITE8] = row[ms::WRITE8];
        aligned[ma::MU_WRITE] = row[ms::MU];
        for v in constraint_values(&MemwAlignedConstraints, &aligned) {
            assert!(
                v == FE::zero() || fused.contains(&v),
                "MEMW_A's {v:?} on {row:?}"
            );
        }
    }
}

/// The constraints of `set` that do not read zero on `row`.
fn broken_on<CS: ConstraintSet<GoldilocksField, GoldilocksExtension>>(
    set: &CS,
    row: &[FE],
) -> Vec<usize> {
    let values = constraint_values(set, row);
    (0..values.len())
        .filter(|&k| values[k] != FE::zero())
        .collect()
}

/// ★ Each fused constraint reads zero on an honest row and not on a row that
/// violates it: on an honest `lh` (bytes 0x80 0x92, sign-extended), an `lb`
/// of 0x80 and an `sw`, each edit breaks exactly the constraints named.
#[test]
fn each_fused_constraint_refuses_its_violating_rows() {
    let two = FE::from(2u64);
    let lh = {
        let mut row = vec![FE::zero(); ml::NUM_COLUMNS];
        row[ml::READ2] = FE::one();
        row[ml::SIGNED] = FE::one();
        row[ml::RES[0]] = FE::from(0x80u64);
        row[ml::RES[1]] = FE::from(0x92u64);
        row[ml::SIGN_BIT] = FE::one();
        for &c in &ml::RES[2..] {
            row[c] = FE::from(0xFFu64);
        }
        row[ml::MU] = FE::one();
        row
    };
    let lb = {
        let mut row = vec![FE::zero(); ml::NUM_COLUMNS];
        row[ml::SIGNED] = FE::one();
        row[ml::RES[0]] = FE::from(0x80u64);
        row[ml::SIGN_BIT] = FE::one();
        for &c in &ml::RES[1..] {
            row[c] = FE::from(0xFFu64);
        }
        row[ml::MU] = FE::one();
        row
    };
    let sw = {
        let mut row = vec![FE::zero(); ms::NUM_COLUMNS];
        row[ms::WRITE4] = FE::one();
        row[ms::MU] = FE::one();
        row
    };
    assert!(broken_on(&MemLoadConstraints, &lh).is_empty());
    assert!(broken_on(&MemLoadConstraints, &lb).is_empty());
    assert!(broken_on(&MemStoreConstraints, &sw).is_empty());
    type Case<'a> = (&'a str, &'a Vec<FE>, usize, FE, Vec<usize>);
    let load_cases: [Case<'_>; 9] = [
        (
            "lh, SIGNED = 2",
            &lh,
            ml::SIGNED,
            two,
            vec![0, 6, 7, 8, 9, 10, 11],
        ),
        ("lh, READ2 = 2", &lh, ml::READ2, two, vec![1, 4, 12]),
        ("lh, READ4 set too", &lh, ml::READ4, FE::one(), vec![4, 12]),
        ("lb, READ8 = 2", &lb, ml::READ8, two, vec![3, 4]),
        ("lh, without MU", &lh, ml::MU, FE::zero(), vec![5]),
        ("lh, MU = 2", &lh, ml::MU, two, vec![5, 13]),
        (
            "lh, byte 5 not the extension",
            &lh,
            ml::RES[5],
            FE::zero(),
            vec![7],
        ),
        (
            "lh, byte 2 not the extension",
            &lh,
            ml::RES[2],
            FE::zero(),
            vec![10],
        ),
        (
            "lb, byte 1 not the extension",
            &lb,
            ml::RES[1],
            FE::zero(),
            vec![12],
        ),
    ];
    for (what, row, column, value, want) in load_cases {
        let mut row = row.clone();
        row[column] = value;
        assert_eq!(broken_on(&MemLoadConstraints, &row), want, "{what}");
    }
    let store_cases: [(&str, usize, FE, Vec<usize>); 6] = [
        ("WRITE2 = 2", ms::WRITE2, two, vec![0, 4]),
        ("WRITE4 = 2", ms::WRITE4, two, vec![1, 4]),
        ("WRITE8 set too", ms::WRITE8, FE::one(), vec![4]),
        ("without MU", ms::MU, FE::zero(), vec![5]),
        ("MU = 2", ms::MU, two, vec![3, 5]),
        ("WRITE8 = 2 alone", ms::WRITE8, two, vec![2, 4]),
    ];
    for (what, column, value, want) in store_cases {
        let mut row = sw.clone();
        row[column] = value;
        if what == "WRITE8 = 2 alone" {
            row[ms::WRITE4] = FE::zero();
        }
        assert_eq!(broken_on(&MemStoreConstraints, &row), want, "{what}");
    }
}

/// ★ The memory format is a verifier constant: the block's production format
/// (the base's) is fused, its pinned preset (the tree's) split, a builder's
/// default split, and the fused format needs MEMW_A's limbs.
#[test]
fn the_production_format_is_fused_and_the_fused_format_needs_the_limbs() {
    use crate::block_whir::{BlockFormat, check_memory_format};
    use crate::tables::fused_memory::MemoryFormat;
    assert_eq!(BlockFormat::production().memory, MemoryFormat::Fused);
    assert_eq!(BlockFormat::pinned().memory, MemoryFormat::Split);
    assert_eq!(MemoryFormat::default(), MemoryFormat::Split);
    assert!(check_memory_format(&BlockFormat::production()).is_ok());
    assert!(check_memory_format(&BlockFormat::pinned()).is_ok());
    let fused = BlockFormat::pinned().with_memory(MemoryFormat::Fused);
    assert_eq!(fused.memory, MemoryFormat::Fused);
    assert!(check_memory_format(&fused).is_ok());
    let lt = fused.with_memw_aligned(MemwAlignedOrder::Lt);
    assert!(check_memory_format(&lt).is_err());
}

/// ★ What a fused row counts into BITWISE is what its row checks. On every
/// honest build, each fused row's [`MemLoadRow::count_bitwise_into`]
/// ([`MemStoreRow::count_bitwise_into`]) bumps exactly the sign-bit, byte-pair
/// and half-word lookups its generated row sends, and the record keeps its
/// pair's access: address, timestamp, old timestamp, width, result or value.
#[test]
fn a_fused_rows_lookup_count_is_its_rows_checks() {
    use crate::tables::bitwise::{BitwiseHistogram, BitwiseOperation, BitwiseOperationType};
    let as_lookup = |bus: u64, tuple: &[u64]| match (BusId::try_from(bus), tuple) {
        (Ok(BusId::Msb8), [x, _]) => {
            BitwiseOperation::single_byte(BitwiseOperationType::Msb8, *x as u8)
        }
        (Ok(BusId::AreBytes), [x, y]) => {
            BitwiseOperation::byte_op(BitwiseOperationType::AreBytes, *x as u8, *y as u8)
        }
        (Ok(BusId::IsHalfword), [v]) => {
            BitwiseOperation::halfword(BitwiseOperationType::IsHalf, *v as u8, (*v >> 8) as u8)
        }
        other => panic!("not a fused row's check: {other:?}"),
    };
    let checks: Vec<u64> = CHECKS.iter().map(|b| u64::from(*b)).collect();
    let (mut counted, mut sent) = (BitwiseHistogram::new(), BitwiseHistogram::new());
    let mut rows = 0;
    for name in PROGRAMS {
        let (program, logs) = run(name);
        let (_, joined) = fused_build(&program, &logs);
        for (fused, op, aligned) in &joined.loads {
            assert_eq!(
                (fused.address(), fused.timestamp(), fused.width()),
                (op.base_address, op.timestamp, op.width),
                "{name}"
            );
            assert_eq!((fused.res(), fused.signed()), (op.res, op.signed), "{name}");
            assert_eq!(fused.old_timestamp(), aligned.old_timestamp(), "{name}");
            assert_eq!(
                u64::from(fused.step()),
                fused.timestamp() - fused.old_timestamp() - 1
            );
            fused.count_bitwise_into(&mut counted);
            let row = first_row(mem_load::generate_mem_load_trace_segments_as(
                &[&[*fused]],
                TraceForm::Wide,
            ));
            for (bus, _, tuple, m) in messages(&mem_load::bus_interactions(), &row) {
                assert!(
                    checks.contains(&bus)
                        || bus == u64::from(BusId::Memory)
                        || bus == u64::from(BusId::MemoryOp)
                );
                if checks.contains(&bus) {
                    sent.bump_n(as_lookup(bus, &tuple), m);
                }
            }
            rows += 1;
        }
        for (fused, op, aligned) in &joined.stores {
            assert_eq!(
                (fused.address(), fused.timestamp(), fused.value()),
                (op.base_address, op.timestamp, op.value),
                "{name}"
            );
            assert_eq!(fused.old_timestamp(), aligned.old_timestamp(), "{name}");
            assert_eq!(fused.width(), aligned.width(), "{name}");
            assert_eq!(
                u64::from(fused.step()),
                fused.timestamp() - fused.old_timestamp() - 1
            );
            let old: [u8; 8] = aligned.old().map(|b| b as u8);
            assert_eq!(fused.old(), u64::from_le_bytes(old), "{name}");
            fused.count_bitwise_into(&mut counted);
            let row = first_row(mem_store::generate_mem_store_trace_segments_as(
                &[&[*fused]],
                TraceForm::Wide,
            ));
            for (bus, _, tuple, m) in messages(&mem_store::bus_interactions(), &row) {
                if checks.contains(&bus) {
                    sent.bump_n(as_lookup(bus, &tuple), m);
                }
            }
            rows += 1;
        }
    }
    assert!(rows > 60, "{rows} fused rows");
    assert_eq!(
        counted.digest(),
        sent.digest(),
        "the counts are the rows' checks"
    );
}

/// The walk's fused build of `program`'s run: windows of `window` cycles,
/// tables of `max_rows`, the block's production register format and MEMW_A
/// order and the fused memory format, the streamed ops dropped or kept.
fn walk_fused(
    program: &Elf,
    logs: &[Log],
    max_rows: &MaxRowsConfig,
    window: usize,
    dropped: bool,
) -> Traces {
    use crate::tables::fused_memory::MemoryFormat;
    windowed(program, logs, max_rows, window, &|b| {
        let b = b
            .register_format(RegisterFormat::InRow)
            .expect("before any window")
            .memw_aligned_order(MemwAlignedOrder::Limbs)
            .expect("before any window")
            .memory_format(MemoryFormat::Fused)
            .expect("before any window");
        if dropped {
            b.drop_streamed_ops().expect("before any window")
        } else {
            b
        }
    })
}

/// The active rows of `tables` (the rows whose `mu` columns sum to nonzero),
/// every column, in order.
fn active_rows(tables: &[Table], mu: &[usize]) -> Vec<Vec<u64>> {
    let mut out = Vec::new();
    for t in tables {
        for r in 0..t.num_rows() {
            if mu.iter().map(|&c| get(t, r, c)).sum::<u64>() != 0 {
                out.push(row_of(t, r).iter().map(FE::canonical_u64).collect());
            }
        }
    }
    out
}

/// ★ The walk makes the fused tables the conversion makes. On every program,
/// at small chunks and at chunks of 4 rows, windows of 7 and of 1000 cycles,
/// the streamed ops kept or dropped, the walk's fused build holds, row for
/// row and in order, the MEM_LOAD, MEM_STORE, LOAD, STORE and MEMW_A rows of
/// the split build's conversion ([`fuse`]); every inner bus balances with
/// BITWISE's multiplicities as the build counted them (its collectors count
/// each fused row's checks); and every load and store of the run is a LOAD,
/// STORE or fused row, once.
fn assert_walk_reproduces_fused_tables(max_rows: &MaxRowsConfig) {
    let mut failed = Vec::new();
    let mut fused_rows = 0usize;
    for name in PROGRAMS {
        let (program, logs) = run(name);
        let split = windowed(&program, &logs, max_rows, 1000, &|b| {
            b.register_format(RegisterFormat::InRow)
                .expect("before any window")
                .memw_aligned_order(MemwAlignedOrder::Limbs)
                .expect("before any window")
        });
        let (loads, stores) = (load_ops(&split.loads).len(), store_ops(&split.stores).len());
        let (want, _) = fuse(split);
        for window in [7, 1000] {
            for dropped in [false, true] {
                let what = format!("{name} window {window} dropped {dropped}");
                let mut got = walk_fused(&program, &logs, max_rows, window, dropped);
                for (family, a, b, mu) in [
                    ("MEM_LOAD", &got.mem_loads, &want.mem_loads, vec![ml::MU]),
                    ("MEM_STORE", &got.mem_stores, &want.mem_stores, vec![ms::MU]),
                    ("LOAD", &got.loads, &want.loads, vec![ld::MU]),
                    ("STORE", &got.stores, &want.stores, vec![st::MU]),
                    (
                        "MEMW_A",
                        &got.memw_aligneds,
                        &want.memw_aligneds,
                        vec![ma::MU_READ, ma::MU_WRITE],
                    ),
                ] {
                    if active_rows(a, &mu) != active_rows(b, &mu) {
                        failed.push(format!("{what}: {family}'s rows differ"));
                    }
                }
                let (l, s) = (
                    active_rows(&got.mem_loads, &[ml::MU]).len(),
                    active_rows(&got.mem_stores, &[ms::MU]).len(),
                );
                if load_ops(&got.loads).len() + l != loads
                    || store_ops(&got.stores).len() + s != stores
                {
                    failed.push(format!("{what}: an access lost or doubled"));
                }
                fused_rows += l + s;
                let left = left_with(&program, &mut got, &|_| None, true);
                if !left.accepted() {
                    failed.push(format!("{what}: {left}"));
                }
            }
        }
    }
    assert!(failed.is_empty(), "{}", failed.join("\n"));
    assert!(fused_rows > 0);
}

#[test]
fn the_walk_reproduces_small_fused_tables() {
    assert_walk_reproduces_fused_tables(&MaxRowsConfig::small());
}

#[test]
fn the_walk_reproduces_uniform_fused_tables() {
    assert_walk_reproduces_fused_tables(&MaxRowsConfig::uniform(4));
}

/// ★ The verifier's frame of a fused statement, with no proof: the statement
/// a fused build states (its tables' heights in AIR order, the fused pair) is
/// framed under the fused format, the fused tables right after STORE's; a
/// pair one off either way, the same statement under the split format, and a
/// fused table past the block's chunk are each refused.
#[cfg(feature = "parallel")]
#[test]
fn the_verifier_frames_a_fused_statement_and_refuses_its_forgeries() {
    use crate::block_whir::{BLOCK_FUSED_MAX_VARS, BlockFormat, OwnedBlockStatement, block_frame};
    use crate::tables::fused_memory::MemoryFormat;
    let (program, logs) = run("fused_mem_cases");
    let elf = crate::test_utils::asm_elf_bytes("fused_mem_cases");
    let mut build = walk_fused(&program, &logs, &MaxRowsConfig::small(), 1000, true);
    let options = ProofOptions::default_test_options();
    let heights = crate::cpu_heights_from_traces(&build);
    let inputs =
        crate::BlockAirInputs::new(&options, &program, heights, &build.public_output_bytes)
            .expect("the block's inputs");
    let format = BlockFormat::pinned().with_memory(MemoryFormat::Fused);
    let mut airs = crate::block_airs(
        &program,
        &options,
        &build.page_configs,
        &build.table_counts(),
        &inputs,
        format.keccak_rnd,
        format.memw_aligned,
        format.registers,
        format.ecdas,
    );
    let fused = build.fused_counts();
    airs.with_fused_memory(&options, fused);
    let pairs = airs.air_trace_pairs(&mut build);
    let names: Vec<String> = pairs
        .iter()
        .map(|(air, _, _)| air.name().to_string())
        .collect();
    let table_num_vars: Vec<u8> = pairs
        .iter()
        .map(|(_, trace, _)| trace.num_rows().trailing_zeros() as u8)
        .collect();
    drop(pairs);
    // The fused tables sit right after STORE's.
    let last_store = names
        .iter()
        .rposition(|n| n.starts_with("STORE["))
        .expect("STORE");
    let first_fused = names
        .iter()
        .position(|n| n.starts_with("MEM_LOAD["))
        .expect("MEM_LOAD");
    assert_eq!(first_fused, last_store + 1, "{names:?}");
    let statement = OwnedBlockStatement {
        table_num_vars: table_num_vars.clone(),
        runtime_page_ranges: build.runtime_page_ranges(),
        table_counts: build.table_counts(),
        public_output: build.public_output_bytes.clone(),
        num_private_input_pages: 0,
        groups: (0..table_num_vars.len() as u32).map(|t| vec![t]).collect(),
        fused_memory: fused.map(|n| n as u64),
        word_memory: [0; 7],
    };
    assert!(fused[0] > 0 && fused[1] > 0, "{fused:?}");
    let frame = |s: &OwnedBlockStatement, f: &BlockFormat| block_frame(s.view(), &elf, &options, f);
    assert!(
        frame(&statement, &format).is_ok(),
        "the honest statement frames"
    );
    for pair in [
        [fused[0] as u64 + 1, fused[1] as u64],
        [fused[0] as u64, fused[1] as u64 - 1],
    ] {
        let off = OwnedBlockStatement {
            fused_memory: pair,
            ..statement.clone()
        };
        assert!(frame(&off, &format).is_err(), "{pair:?} refused");
    }
    let split = BlockFormat::pinned();
    assert!(
        frame(&statement, &split).is_err(),
        "the split format refuses a fused pair"
    );
    let unstated = OwnedBlockStatement {
        fused_memory: [0, 0],
        ..statement.clone()
    };
    assert!(
        frame(&unstated, &split).is_err(),
        "the split format has no slot for them"
    );
    let mut tall = statement.clone();
    tall.table_num_vars[first_fused] = BLOCK_FUSED_MAX_VARS as u8 + 1;
    assert!(
        frame(&tall, &format).is_err(),
        "a fused table past the chunk"
    );
}
