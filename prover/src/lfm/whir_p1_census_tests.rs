//! D-WHIR-P1 S5's laptop census: the Poseidon1 leaf against RPX's at the
//! median block's shape, by emission alone (a WHIR block statement is shape
//! only, so no proof). The statement is I-PADLEAF §7's instrument's (its
//! measurement branch `padleaf/whir-census-pow`, 18d7d6b20).
//!
//! Run alone: `LAMBDA_VM_WHIR_HASH=rpx … --ignored --exact
//! lfm::whir_p1_census_tests::whir_p1_leaf_census --nocapture`. Prints
//! `CENSUS …` lines: each base's plan, and leaves of one to four full groups
//! and the plan's own heaviest leaf, rows by instruction and padded cells by
//! chip.

use std::collections::BTreeMap;

use super::instr::{BaseOp, Instr};
use super::whir_block::{BlockPartition, WhirBlockPlan};
use crate::block_whir::{BlockFormat, BlockStatement, OwnedBlockStatement};
use stark::proof::options::BaseFormat;

/// The median 25475471's tables (ULTRA mc12's BLOCK CENSUS, #1013), AIR order
/// after the fixed tables: `(kind, [(rows, count)])`, and its 198 pages.
const MEDIAN_MIX: &[(&str, &[(usize, usize)])] = &[
    ("COMMIT", &[(256, 1)]),
    ("KECCAK", &[(1 << 17, 1)]),
    ("KECCAK_RND", &[(1 << 16, 35), (1 << 15, 1)]),
    ("ECSM", &[(2048, 1)]),
    ("ECDAS", &[(1 << 17, 4)]),
    ("HINT", &[(1024, 1)]),
    ("CPU", &[(1 << 21, 142), (1 << 20, 1)]),
    ("LT", &[(1 << 21, 49), (1 << 20, 49)]),
    ("SHIFT", &[(1 << 21, 10), (1 << 20, 1)]),
    ("MEMW", &[(1 << 21, 5)]),
    ("MEMW_A", &[(1 << 21, 50), (1 << 17, 1)]),
    ("LOAD", &[(1 << 21, 31)]),
    ("MUL", &[(1 << 20, 1), (1 << 17, 1)]),
    ("DVRM", &[(1 << 14, 1)]),
    (
        "BRANCH",
        &[(1 << 19, 11), (1 << 18, 1), (1 << 17, 1), (1 << 14, 2)],
    ),
    ("PAGE", &[]),
    ("MEMW_R", &[(1 << 21, 292), (1 << 19, 1)]),
    (
        "EQ",
        &[(1 << 18, 4), (1 << 17, 1), (1 << 15, 1), (1 << 14, 2)],
    ),
    (
        "BYTEWISE",
        &[(1 << 20, 7), (1 << 19, 1), (1 << 18, 1), (1 << 17, 2)],
    ),
    ("STORE", &[(1 << 21, 23), (1 << 17, 1)]),
    ("CPU32", &[(1 << 20, 1)]),
];
const MEDIAN_PAGES: usize = 198;
const RUNTIME_PAGE_BASE: u64 = 0x4000_0000;

/// The statement of [`MEDIAN_MIX`] over `elf_bytes`, its groups packed as the
/// prover packs them ([`stark::multilinear_block::block_groups`]).
fn median_statement(
    elf_bytes: &[u8],
    opts: &crate::ProofOptions,
    format: &BlockFormat,
) -> OwnedBlockStatement {
    let elf = executor::elf::Elf::load(elf_bytes).expect("load the ELF");
    let elf_pages = crate::tables::trace_builder::Traces::page_configs_from_elf(&elf).len();
    let mut counts = crate::TableCounts {
        cpu: 0,
        lt: 0,
        memw: 0,
        memw_aligned: 0,
        load: 0,
        mul: 0,
        dvrm: 0,
        shift: 0,
        branch: 0,
        memw_register: 0,
        eq: 0,
        bytewise: 0,
        store: 0,
        cpu32: 0,
        keccak: 0,
        keccak_rnd: 0,
        ecsm: 0,
        ecdas: 0,
        hint: 0,
        commit: 0,
        blake3: 0,
    };
    let mut rows: Vec<usize> = vec![32; crate::FIXED_TABLE_COUNT];
    for &(kind, runs) in MEDIAN_MIX {
        if kind == "PAGE" {
            // Every page table is one row a byte of its page (`page::DEFAULT_PAGE_SIZE`).
            rows.extend(std::iter::repeat_n(
                crate::tables::page::DEFAULT_PAGE_SIZE,
                elf_pages,
            ));
            rows.extend(std::iter::repeat_n(1 << 18, MEDIAN_PAGES - elf_pages));
            continue;
        }
        let n: usize = runs.iter().map(|&(_, c)| c).sum();
        rows.extend(runs.iter().flat_map(|&(r, c)| std::iter::repeat_n(r, c)));
        let slot = match kind {
            "COMMIT" => &mut counts.commit,
            "KECCAK" => &mut counts.keccak,
            "KECCAK_RND" => &mut counts.keccak_rnd,
            "ECSM" => &mut counts.ecsm,
            "ECDAS" => &mut counts.ecdas,
            "HINT" => &mut counts.hint,
            "CPU" => &mut counts.cpu,
            "LT" => &mut counts.lt,
            "SHIFT" => &mut counts.shift,
            "MEMW" => &mut counts.memw,
            "MEMW_A" => &mut counts.memw_aligned,
            "LOAD" => &mut counts.load,
            "MUL" => &mut counts.mul,
            "DVRM" => &mut counts.dvrm,
            "BRANCH" => &mut counts.branch,
            "MEMW_R" => &mut counts.memw_register,
            "EQ" => &mut counts.eq,
            "BYTEWISE" => &mut counts.bytewise,
            "STORE" => &mut counts.store,
            "CPU32" => &mut counts.cpu32,
            other => panic!("{other}"),
        };
        *slot = n;
    }
    let ranges = vec![crate::RuntimePageRange {
        base: RUNTIME_PAGE_BASE,
        count: (MEDIAN_PAGES - elf_pages) as u64,
    }];
    let page_configs = crate::tables::trace_builder::Traces::page_configs_from_elf_and_runtime(
        &elf,
        &ranges,
        0,
        rows.len(),
    )
    .expect("page configs");
    let airs = crate::VmAirs::new(
        &elf,
        opts,
        false,
        &page_configs,
        &counts,
        None,
        true,
        None,
        None,
        None,
    );
    // The fixed tables at their preprocessed columns' heights, so group 0
    // emits too (HALT, which has none, stays at the placeholder).
    for (i, air) in airs
        .air_refs()
        .iter()
        .enumerate()
        .take(crate::FIXED_TABLE_COUNT)
    {
        if let Some(column) = air.precomputed_columns().first() {
            rows[i] = column.len();
        }
    }
    println!("CENSUS FIXED ROWS: {:?}", &rows[..crate::FIXED_TABLE_COUNT]);
    let shapes: Vec<(usize, usize)> = airs
        .air_refs()
        .iter()
        .zip(&rows)
        .map(|(air, &r)| (air.trace_layout().0, r.trailing_zeros() as usize))
        .collect();
    let config = format.chain_config(&shapes);
    let sizes =
        stark::multilinear_block::block_groups(&shapes, config.format.stack, format.group_polys)
            .expect("groups");
    let mut at = 0u32;
    let groups: Vec<Vec<u32>> = sizes
        .iter()
        .map(|&size| {
            let g = (at..at + size as u32).collect();
            at += size as u32;
            g
        })
        .collect();
    OwnedBlockStatement {
        table_num_vars: rows.iter().map(|r| r.trailing_zeros() as u8).collect(),
        runtime_page_ranges: ranges,
        table_counts: counts,
        public_output: vec![0; 4],
        num_private_input_pages: 0,
        groups,
    }
}

fn kind(instr: &Instr) -> &'static str {
    match instr {
        Instr::Const { .. } => "CONST",
        Instr::BaseAlu { op, .. } => match op {
            BaseOp::Add => "BALU.add",
            BaseOp::Sub => "BALU.sub",
            BaseOp::Mul => "BALU.mul",
            BaseOp::Div => "BALU.div",
            BaseOp::MulAdd => "BALU.muladd",
        },
        Instr::ExtAlu { .. } => "XALU",
        Instr::Select { .. } => "SELECT",
        Instr::BitDec { .. } => "BITDEC",
        Instr::Hash { .. } => "HASH",
        Instr::Hash16(_) => "HASH16",
        Instr::KeccakF(_) => "KECCAK",
        Instr::Blake3(_) => "BLAKE3",
        Instr::Hint { .. } => "HINT",
        Instr::Pack { .. } => "LANES.pack",
        Instr::Unpack { .. } => "LANES.unpack",
        Instr::Public { .. } => "PUBLIC",
    }
}

/// One leaf's census line: instruction rows by kind, then padded cells by chip
/// under the program's own hasher.
fn census_leaf(label: &str, program: &super::compiler::LfmProgram) -> u64 {
    let mut rows: BTreeMap<&'static str, usize> = BTreeMap::new();
    for instr in &program.instrs {
        *rows.entry(kind(instr)).or_default() += 1;
    }
    let balu: usize = rows
        .iter()
        .filter(|(k, _)| k.starts_with("BALU"))
        .map(|(_, r)| *r)
        .sum();
    let chips = super::airs::lfm_chip_census_with_hasher(
        program,
        program.hasher(crate::hash_pin::BLOCK_HASHER),
    );
    let cells: u64 = chips.iter().map(|c| c.main_cells() + c.aux_cells()).sum();
    println!(
        "CENSUS {label} ROWS: BALU={balu} {}",
        rows.iter()
            .map(|(k, r)| format!("{k}={r}"))
            .collect::<Vec<_>>()
            .join(" ")
    );
    println!(
        "CENSUS {label} CHIPS: cells {cells} · {}",
        chips
            .iter()
            .filter(|c| c.real_rows > 0)
            .map(|c| format!(
                "{}={}/{}x{}",
                c.name,
                c.real_rows,
                c.rows,
                c.main_cols + c.aux_cols
            ))
            .collect::<Vec<_>>()
            .join(" ")
    );
    cells
}

/// ★ The S5 census (laptop instrument, `--exact`): per base, the median mix's
/// plan, leaves of 1–4 full groups and the plan's heaviest leaf.
#[test]
#[ignore = "laptop instrument: run alone with --exact"]
fn whir_p1_leaf_census() {
    let elf_bytes = crate::test_utils::asm_elf_bytes("poc_rodata_commit");
    let opts = super::proof::block_base_options();
    let mut four: Vec<(String, u64)> = Vec::new();
    for (name, base) in [("RPX", BaseFormat::RPX), ("P1", BaseFormat::P1_WHIR)] {
        let production = BlockFormat::production();
        let format = BlockFormat {
            zf: production.zf.with_base(base),
            max_groups: 1024,
            ..production
        };
        let owned = median_statement(&elf_bytes, &opts, &format);
        let statement: BlockStatement<'_> = owned.view();
        let plan = WhirBlockPlan::derive_with(
            &elf_bytes,
            &opts,
            &format,
            statement,
            None,
            super::whir_block::BLOCK_FAN_IN,
            None,
        )
        .expect("the plan derives");
        let costs = plan.costs().to_vec();
        println!(
            "CENSUS {name} PLAN: {} tables · {} groups · {} leaves · cap {} · group costs {:?}",
            owned.table_num_vars.len(),
            plan.num_groups(),
            plan.partition().num_leaves(),
            super::whir_block::leaf_cap_for(&format.zf.base),
            costs
        );
        let heaviest = (0..plan.partition().num_leaves())
            .max_by_key(|&k| {
                plan.partition()
                    .leaf(k)
                    .iter()
                    .map(|&g| costs[g])
                    .sum::<usize>()
            })
            .expect("a leaf");
        let program = plan.leaf_program(heaviest).expect("the leaf emits");
        census_leaf(
            &format!(
                "{name} PLAN LEAF {heaviest} (groups {:?})",
                plan.partition().leaf(heaviest)
            ),
            &program,
        );
        let n = plan.num_groups();
        let mut lists: Vec<Vec<usize>> =
            vec![vec![1], vec![2, 3], vec![4, 5, 6], vec![7, 8, 9, 10]];
        let used: Vec<usize> = lists.iter().flatten().copied().collect();
        lists.push((0..n).filter(|g| !used.contains(g)).collect());
        let plan = plan.with_partition(BlockPartition::new(lists.clone(), n).expect("covers once"));
        for (k, list) in lists.iter().enumerate().take(4) {
            let program = plan.leaf_program(k).expect("the leaf emits");
            let cells = census_leaf(&format!("{name} LEAF {} GROUPS", list.len()), &program);
            if list.len() == 4 {
                four.push((name.to_string(), cells));
            }
        }
    }
    // The Poseidon1 plan at a 2^17 socket-row cap (three full groups a leaf):
    // the hash table one doubling down from D8a's.
    {
        let production = BlockFormat::production();
        let format = BlockFormat {
            zf: production.zf.with_base(BaseFormat::P1_WHIR),
            max_groups: 1024,
            ..production
        };
        let owned = median_statement(&elf_bytes, &opts, &format);
        let plan = WhirBlockPlan::derive_capped(
            &elf_bytes,
            &opts,
            &format,
            owned.view(),
            None,
            super::whir_block::BLOCK_FAN_IN,
            None,
            1 << 17,
        )
        .expect("the plan derives at 2^17");
        let costs = plan.costs().to_vec();
        let heaviest = (0..plan.partition().num_leaves())
            .max_by_key(|&k| {
                plan.partition()
                    .leaf(k)
                    .iter()
                    .map(|&g| costs[g])
                    .sum::<usize>()
            })
            .expect("a leaf");
        println!(
            "CENSUS P1@2^17 PLAN: {} leaves · heaviest leaf {heaviest} groups {:?}",
            plan.partition().num_leaves(),
            plan.partition().leaf(heaviest)
        );
        let program = plan.leaf_program(heaviest).expect("the leaf emits");
        census_leaf(&format!("P1@2^17 PLAN LEAF {heaviest}"), &program);
    }
    if let [(_, rpx), (_, p1)] = four.as_slice() {
        println!(
            "CENSUS 4-GROUP LEAF CELLS: RPX {rpx} · P1 {p1} · P1/RPX {:.4}",
            *p1 as f64 / *rpx as f64
        );
    }
}

/// A leaf's emitted rows against the plan's: every chip but the constants
/// exactly on the carrier, and on any other leaf less one vector — the COMMIT
/// bus target's rows, the same for every such leaf (`target` keeps the first);
/// the constants (interned) at most the plan's.
fn assert_planned(
    label: &str,
    carrier: bool,
    emitted: ChipRows,
    planned: ChipRows,
    target: &mut Option<ChipRows>,
) {
    assert!(
        emitted.consts <= planned.consts,
        "{label}: {} constants against the plan's {}",
        emitted.consts,
        planned.consts
    );
    let (e, p) = (
        ChipRows {
            consts: 0,
            ..emitted
        },
        ChipRows {
            consts: 0,
            ..planned
        },
    );
    if carrier {
        assert_eq!(e, p, "{label}: the carrier's rows are the plan's");
        return;
    }
    assert!(e.under(p), "{label}: {e:?} past the plan's {p:?}");
    let (ea, pa) = (e.to_array(), p.to_array());
    let gap = ChipRows::from_array(core::array::from_fn(|c| pa[c] - ea[c]));
    match target {
        None => *target = Some(gap),
        Some(t) => assert_eq!(*t, gap, "{label}: a leaf less another target"),
    }
}

/// ★ Every group of the median mix as a one-group Poseidon1 leaf, against the
/// plan's rows for it — the lead's acceptance criterion (10-06; every chip
/// since S6b): the rows the partition admits leaves by ARE the emitted ones,
/// for every group shape the mix has, both prepared kinds included (DECODE's
/// stack in group 0, the ELF pages' genesis), with no fallback
/// ([`assert_planned`]). Then every leaf the plan builds, emitted, stays under
/// every padded height.
///
/// Both prepared kinds need a real guest (a small one's pages are not dense
/// enough for a genesis stack): `CENSUS_ELF=<path>` or `BLOCK_WHIR_ELF` (the
/// box's), which then also asserts both kinds are present; else the small one.
/// Prints `CENSUS GROUP …` and `CENSUS PLAN LEAF …` lines.
#[test]
#[ignore = "instrument (laptop with CENSUS_ELF, box with BLOCK_WHIR_ELF): run with --exact"]
fn whir_p1_group_census() {
    let real = std::env::var("CENSUS_ELF")
        .or_else(|_| std::env::var("BLOCK_WHIR_ELF"))
        .ok();
    let elf_bytes = match &real {
        Some(path) => std::fs::read(path).expect("read the guest ELF"),
        None => crate::test_utils::asm_elf_bytes("poc_rodata_commit"),
    };
    let opts = super::proof::block_base_options();
    let format = p1_format();
    let owned = median_statement(&elf_bytes, &opts, &format);
    let derive = || {
        WhirBlockPlan::derive_with(
            &elf_bytes,
            &opts,
            &format,
            owned.view(),
            None,
            super::whir_block::BLOCK_FAN_IN,
            None,
        )
        .expect("the plan derives")
    };
    let started = std::time::Instant::now();
    let plan = derive();
    println!(
        "CENSUS PLAN DERIVED in {:.2} s (every group emitted once)",
        started.elapsed().as_secs_f64()
    );
    let started = std::time::Instant::now();
    let again = super::whir_block::chip_loads_by_emission(&plan).expect("the probe");
    println!(
        "CENSUS PROBE: {} groups measured in {:.2} s",
        plan.num_groups(),
        started.elapsed().as_secs_f64()
    );
    let n = plan.num_groups();
    let prepared: Vec<usize> = plan.prepared().iter().map(|p| p.group).collect();
    let (loads, front) = plan.chip_loads();
    assert_eq!((loads.to_vec(), front), again, "the probe is deterministic");
    let (loads, front) = (loads.to_vec(), front);
    if real.is_some() {
        assert!(
            prepared.contains(&0) && prepared.iter().any(|&g| g != 0),
            "a real guest's mix has both prepared kinds, got groups {prepared:?}"
        );
    }
    let singles = plan.with_partition(
        BlockPartition::new((0..n).map(|g| vec![g]).collect(), n).expect("one group a leaf"),
    );
    let mut target = None;
    let mut max = ChipRows::default();
    for (g, &load) in loads.iter().enumerate() {
        let emitted = ChipRows::of(&singles.leaf_program(g).expect("the leaf emits"));
        assert_planned(
            &format!("group {g} (prepared {})", prepared.contains(&g)),
            g == CARRIER,
            emitted,
            front.plus(load),
            &mut target,
        );
        println!(
            "CENSUS GROUP {g}: tables {} · prepared {} · cost {} · {load:?}",
            owned.groups[g].len(),
            prepared.contains(&g),
            singles.costs()[g],
        );
        let (m, l) = (max.to_array(), load.to_array());
        max = ChipRows::from_array(core::array::from_fn(|c| m[c].max(l[c])));
    }
    println!("CENSUS GROUP MAX: {max:?}");
    println!("CENSUS FRONT: {front:?} · a non-carrier leaf less {target:?}");

    // The plan at the production cap: every leaf, against its chips' heights.
    let plan = derive();
    let mut target = None;
    for k in 0..plan.partition().num_leaves() {
        let leaf = plan.partition().leaf(k);
        let emitted = ChipRows::of(&plan.leaf_program(k).expect("the leaf emits"));
        let planned = leaf.iter().fold(front, |acc, &g| acc.plus(loads[g]));
        assert_planned(
            &format!("leaf {k}"),
            k == CARRIER,
            emitted,
            planned,
            &mut target,
        );
        println!("CENSUS PLAN LEAF {k}: groups {leaf:?} · {emitted:?}");
        assert!(
            emitted.under(P1_LEAF_HEIGHTS),
            "leaf {k} is past a chip's height: {emitted:?}"
        );
    }
}

// ======================== the leaves' chip heights ========================

use super::whir_block::{CARRIER, P1_LEAF_HEIGHTS, leaf_partition_rows};
use super::whir_chain::ChipRows;

fn p1_format() -> BlockFormat {
    let production = BlockFormat::production();
    BlockFormat {
        zf: production.zf.with_base(BaseFormat::P1_WHIR),
        max_groups: 1024,
        ..production
    }
}

fn rows(hash: usize, select: usize, balu: usize, hint: usize) -> ChipRows {
    ChipRows {
        hash,
        select,
        balu,
        hint,
        ..ChipRows::default()
    }
}

/// The census's shapes (median mix, an ethrex ELF; `whir_p1_group_census`): a
/// full group, a full one heavy in the extension ALU (KECCAK_RND's tables), a
/// prepared one (the ELF pages' genesis), and the front.
const FULL: ChipRows = ChipRows {
    hash: 35_744,
    select: 139_878,
    balu: 142_236,
    xalu: 193_562,
    lanes: 129_555,
    hint: 135_333,
    bitdec: 2_415,
    consts: 527,
    public: 0,
    accel: 0,
};
const HEAVY: ChipRows = ChipRows {
    hash: 38_358,
    xalu: 321_505,
    lanes: 147_538,
    hint: 140_290,
    consts: 519,
    ..FULL
};
const PREPARED: ChipRows = ChipRows {
    hash: 48_367,
    select: 172_938,
    balu: 180_090,
    xalu: 350_252,
    lanes: 190_294,
    hint: 185_114,
    bitdec: 3_105,
    consts: 706,
    public: 0,
    accel: 0,
};
const FRONT: ChipRows = ChipRows {
    hash: 287,
    select: 0,
    balu: 0,
    xalu: 17,
    lanes: 544,
    hint: 237,
    bitdec: 0,
    consts: 554,
    public: 5,
    accel: 0,
};

fn sums(loads: &[ChipRows], leaf: &[usize], front: ChipRows) -> ChipRows {
    leaf.iter().fold(front, |acc, &g| acc.plus(loads[g]))
}

/// ★ The partition admits no leaf past any chip's height, whichever chip
/// binds: groups light in the socket but heavy in `Select`, base ALU, hints,
/// lanes, bit decompositions or extension ALU are split where a partition
/// blind to that chip would double it silently. Seven full groups share a
/// leaf, an eighth does not (one 2^18 socket table); six heavy ones do, a
/// seventh not. Every group is placed once, over no more leaves than the rows
/// need, and a group no leaf holds is refused.
#[test]
fn the_p1_partition_keeps_every_chip_under_its_height() {
    let front = FRONT;
    let small = ChipRows {
        xalu: 90_000,
        lanes: 50_000,
        ..rows(9_000, 60_000, 40_000, 50_000)
    };
    for loads in [
        vec![PREPARED, FULL, FULL],
        vec![PREPARED, FULL, HEAVY, FULL, HEAVY, FULL, PREPARED, small],
        [
            vec![PREPARED; 2],
            vec![HEAVY; 8],
            vec![FULL; 68],
            vec![small],
        ]
        .concat(),
    ] {
        let partition = leaf_partition_rows(&loads, front, P1_LEAF_HEIGHTS).expect("partitions");
        for leaf in partition.leaves() {
            assert!(
                sums(&loads, leaf, front).under(P1_LEAF_HEIGHTS),
                "leaf {leaf:?} of {} groups is past a height",
                loads.len()
            );
        }
        // No fewer than the socket's sum allows.
        let hashes: usize = loads.iter().map(|l| l.hash).sum();
        assert!(partition.num_leaves() >= hashes.div_ceil(P1_LEAF_HEIGHTS.hash - front.hash));
        // Deterministic.
        assert_eq!(
            partition,
            leaf_partition_rows(&loads, front, P1_LEAF_HEIGHTS).expect("again")
        );
    }
    let leaves = |loads: &[ChipRows]| {
        leaf_partition_rows(loads, front, P1_LEAF_HEIGHTS)
            .expect("fits")
            .num_leaves()
    };
    assert_eq!(leaves(&[FULL; 7]), 1, "seven full groups are one leaf");
    assert_eq!(leaves(&[FULL; 8]), 2, "an eighth is past the socket");
    assert_eq!(leaves(&[HEAVY; 6]), 1, "six heavy groups are one leaf");
    assert_eq!(leaves(&[HEAVY; 7]), 2, "a seventh is past the socket");
    // The 1× block's nine groups close in two leaves; the median's ninety in
    // thirteen, the socket's lower bound — where balancing alone puts the two
    // prepared groups on two leaves of five full ones and takes fourteen,
    // first fit packs them on one with four.
    let nine = [vec![FULL; 7], vec![PREPARED; 2]].concat();
    assert_eq!(leaves(&nine), 2, "the 1× block's nine groups");
    let ninety = [vec![PREPARED; 2], vec![FULL; 88]].concat();
    assert_eq!(leaves(&ninety), 13, "the median's ninety groups");
    // Any one chip binds: groups light in the socket are split.
    for heavy in [
        rows(10_000, 600_000, 10_000, 10_000),
        rows(10_000, 10_000, 600_000, 10_000),
        rows(10_000, 10_000, 10_000, 600_000),
        ChipRows {
            lanes: 600_000,
            ..rows(10_000, 10_000, 10_000, 10_000)
        },
        ChipRows {
            xalu: 1_100_000,
            ..rows(10_000, 10_000, 10_000, 10_000)
        },
        ChipRows {
            bitdec: 17_000,
            ..rows(10_000, 10_000, 10_000, 10_000)
        },
        ChipRows {
            consts: 4_200,
            ..rows(10_000, 10_000, 10_000, 10_000)
        },
    ] {
        let p = leaf_partition_rows(&[heavy, heavy], front, P1_LEAF_HEIGHTS).expect("fits");
        assert_eq!(
            p.num_leaves(),
            2,
            "{heavy:?} twice is past a height: split, not doubled"
        );
    }
    // ★ Four groups that fit every chip S6 bounded (socket, Select, base ALU,
    // words) but not the extension ALU (2.8 M): the partition splits them; with
    // the XALU bound off — S6's partition — they share a leaf whose XALU
    // crosses 2^21 (arm C on RYZEN 062 padded seven groups' to 2^21 unbounded).
    let xaluy = ChipRows {
        xalu: 700_000,
        ..rows(40_000, 150_000, 150_000, 150_000)
    };
    let set = [xaluy; 4];
    let p = leaf_partition_rows(&set, front, P1_LEAF_HEIGHTS).expect("fits");
    assert_eq!(p.num_leaves(), 2, "split by the extension ALU");
    for leaf in p.leaves() {
        assert!(sums(&set, leaf, front).under(P1_LEAF_HEIGHTS));
    }
    let unbounded = ChipRows {
        xalu: usize::MAX / 2,
        ..P1_LEAF_HEIGHTS
    };
    let blind = leaf_partition_rows(&set, front, unbounded).expect("the XALU-blind partition");
    assert!(
        blind
            .leaves()
            .iter()
            .any(|leaf| sums(&set, leaf, front).xalu > P1_LEAF_HEIGHTS.xalu),
        "the XALU bound is what keeps it under 2^21: {:?}",
        blind.leaves()
    );
    // Refusals: a group past a height, an accelerator row (a WHIR leaf has
    // none), a front past a height.
    let huge = rows(1_000, 1_200_000, 0, 0);
    assert!(leaf_partition_rows(&[FULL, huge], front, P1_LEAF_HEIGHTS).is_err());
    let accel = ChipRows { accel: 1, ..FULL };
    assert!(leaf_partition_rows(&[FULL, accel], front, P1_LEAF_HEIGHTS).is_err());
    assert!(leaf_partition_rows(&[FULL], rows(0, 1 << 20, 0, 0), P1_LEAF_HEIGHTS).is_err());
    assert!(
        leaf_partition_rows(
            &[FULL],
            ChipRows {
                xalu: (1 << 21) + 1,
                ..front
            },
            P1_LEAF_HEIGHTS
        )
        .is_err()
    );
}

/// ★ The heights hold on the median mix, as emitted: the plan's rows are the
/// emitted ones ([`assert_planned`]: groups 0 (prepared, the carrier) and 1
/// (heavy) alone, and a leaf of seven heavy groups — past the socket and the
/// extension ALU, a leaf the partition never builds, its rows still the sum),
/// and every leaf the plan builds stays under every chip's height.
#[test]
fn the_p1_leaves_stay_under_their_heights_on_the_median_mix() {
    let elf_bytes = crate::test_utils::asm_elf_bytes("poc_rodata_commit");
    let opts = super::proof::block_base_options();
    let format = p1_format();
    let owned = median_statement(&elf_bytes, &opts, &format);
    let plan = WhirBlockPlan::derive_with(
        &elf_bytes,
        &opts,
        &format,
        owned.view(),
        None,
        super::whir_block::BLOCK_FAN_IN,
        None,
    )
    .expect("the plan derives");
    let (loads, front) = plan.chip_loads();
    let (loads, n) = (loads.to_vec(), plan.num_groups());
    assert_eq!(loads.len(), n);
    // Every leaf the plan built, by its rows.
    for leaf in plan.partition().leaves() {
        assert!(
            sums(&loads, leaf, front).under(P1_LEAF_HEIGHTS),
            "leaf {leaf:?}"
        );
    }
    // The rows are the emitted ones.
    let lists = [vec![0usize], vec![1], (2..9).collect(), (9..n).collect()];
    let probe = plan.with_partition(BlockPartition::new(lists.to_vec(), n).expect("covers once"));
    let mut target = None;
    for (k, list) in lists.iter().enumerate().take(3) {
        let emitted = ChipRows::of(&probe.leaf_program(k).expect("the leaf emits"));
        assert_planned(
            &format!("groups {list:?}"),
            k == CARRIER,
            emitted,
            sums(&loads, list, front),
            &mut target,
        );
        if k == 2 {
            println!("P1 SEVEN HEAVY GROUPS: {emitted:?} against {P1_LEAF_HEIGHTS:?}");
            assert!(
                !emitted.under(P1_LEAF_HEIGHTS),
                "seven heavy groups: {emitted:?}"
            );
        }
    }
}

/// The LFM chip a [`ChipRows`] field counts, by [`ChipRows::NAMES`] order.
const CHIP_NAMES: [&str; 10] = [
    "LFM_HASH",
    "LFM_SELECT",
    "LFM_BALU",
    "LFM_XALU",
    "LFM_LANES",
    "LFM_HINT",
    "LFM_BITDEC",
    "LFM_CONST",
    "LFM_PUBLIC",
    "",
];

/// RYZEN 062's card model of a leaf (I-WHIR-P1 §S6.4): ≈ 0.28 s plus
/// ≈ 0.0122 s a million padded cells outside the socket.
fn card_seconds(non_hash_cells: u64) -> f64 {
    0.28 + 0.0122 * non_hash_cells as f64 / 1e6
}

/// ★ The S6b census (instrument, `--exact`; `CENSUS_ELF` for a real guest):
/// the median mix's leaves under RPX's partition, S6's (178 k socket rows, four
/// groups a leaf) and S6b's (every chip bounded) — every leaf emitted: its
/// padded chip heights, its cells (all, and outside the socket), the tree above
/// it, and 062's card model summed over the leaves (L0 is card-serial). Then
/// the 1× block's shape, nine groups (seven full, both prepared) of the same
/// mix, under S6 and S6b, by the plan's rows and the chips' widths. Prints
/// `S6b …` lines; asserts S6b's leaves under every height.
#[test]
#[ignore = "instrument: run alone with --exact (CENSUS_ELF for a real guest)"]
fn whir_s6b_census() {
    let elf_bytes = match std::env::var("CENSUS_ELF") {
        Ok(path) => std::fs::read(path).expect("read CENSUS_ELF"),
        Err(_) => crate::test_utils::asm_elf_bytes("poc_rodata_commit"),
    };
    let opts = super::proof::block_base_options();
    let production = BlockFormat::production();
    let arms: [(&str, BaseFormat, Option<usize>); 3] = [
        ("RPX", BaseFormat::RPX, None),
        ("P1-S6 (178 k)", BaseFormat::P1_WHIR, Some(178_000)),
        ("P1-S6b", BaseFormat::P1_WHIR, None),
    ];
    // Each chip's columns a row (main + aux), from the leaves' census.
    let mut widths: BTreeMap<&'static str, u64> = BTreeMap::new();
    let mut s6b: Option<(Vec<ChipRows>, ChipRows, Vec<usize>)> = None;
    for (name, base, cap) in arms {
        let format = BlockFormat {
            zf: production.zf.with_base(base),
            max_groups: 1024,
            ..production
        };
        let owned = median_statement(&elf_bytes, &opts, &format);
        let started = std::time::Instant::now();
        let plan = WhirBlockPlan::derive_capped(
            &elf_bytes,
            &opts,
            &format,
            owned.view(),
            None,
            super::whir_block::BLOCK_FAN_IN,
            None,
            cap.unwrap_or_else(|| super::whir_block::leaf_cap_for(&format.zf.base)),
        )
        .expect("the plan derives");
        let derived = started.elapsed().as_secs_f64();
        let levels = plan.levels();
        let nodes: usize = levels.iter().map(|l| l.arities.len()).sum();
        let mut max: BTreeMap<&'static str, u64> = BTreeMap::new();
        let (mut cells_sum, mut non_hash_sum, mut card) = (0u64, 0u64, 0f64);
        let mut groups_max = 0usize;
        let started = std::time::Instant::now();
        for k in 0..plan.partition().num_leaves() {
            let program = plan.leaf_program(k).expect("the leaf emits");
            let chips = super::airs::lfm_chip_census_with_hasher(
                &program,
                program.hasher(crate::hash_pin::BLOCK_HASHER),
            );
            let cells: u64 = chips.iter().map(|c| c.main_cells() + c.aux_cells()).sum();
            let hash: u64 = chips
                .iter()
                .filter(|c| c.name == "LFM_HASH")
                .map(|c| c.main_cells() + c.aux_cells())
                .sum();
            for c in &chips {
                let e = max.entry(c.name).or_default();
                *e = (*e).max(c.rows);
                if c.rows > 0 {
                    widths.insert(c.name, (c.main_cols + c.aux_cols) as u64);
                }
            }
            if base == BaseFormat::P1_WHIR && cap.is_none() {
                let rows = ChipRows::of(&program);
                assert!(
                    rows.under(P1_LEAF_HEIGHTS),
                    "S6b leaf {k} past a bounded height: {rows:?}"
                );
            }
            cells_sum += cells;
            non_hash_sum += cells - hash;
            card += card_seconds(cells - hash);
            groups_max = groups_max.max(plan.partition().leaf(k).len());
        }
        println!(
            "S6b {name}: {} groups · {} leaves (≤ {groups_max} groups) · {nodes} nodes over {} node levels · leaf cells Σ {:.1} M, outside the socket Σ {:.1} M · card model Σ {card:.2} s · plan derived {derived:.2} s, leaves emitted {:.2} s · padded max {}",
            plan.num_groups(),
            plan.partition().num_leaves(),
            levels.len(),
            cells_sum as f64 / 1e6,
            non_hash_sum as f64 / 1e6,
            started.elapsed().as_secs_f64(),
            max.iter()
                .map(|(n, r)| format!("{n}={r}"))
                .collect::<Vec<_>>()
                .join(" ")
        );
        if base == BaseFormat::P1_WHIR && cap.is_none() {
            let (loads, front) = plan.chip_loads();
            let prepared = plan.prepared().iter().map(|p| p.group).collect();
            s6b = Some((loads.to_vec(), front, prepared));
        }
    }
    println!("S6b CHIP WIDTHS: {widths:?}");

    // The 1× shape: nine groups of the mix — both prepared and seven full —
    // under S6's heights and S6b's, by the plan's rows.
    let (loads, front, prepared) = s6b.expect("the S6b arm ran");
    let full: Vec<usize> = (0..loads.len())
        .filter(|g| !prepared.contains(g))
        .take(7)
        .collect();
    let nine: Vec<ChipRows> = prepared.iter().chain(&full).map(|&g| loads[g]).collect();
    let leaf_cells = |r: ChipRows| -> u64 {
        r.to_array()
            .iter()
            .zip(CHIP_NAMES)
            .filter(|&(&rows, name)| rows > 0 && !name.is_empty() && name != "LFM_HASH")
            .map(|(&rows, name)| {
                rows.next_power_of_two() as u64 * widths.get(name).copied().unwrap_or(0)
            })
            .sum::<u64>()
            + 65_536 * widths.get("LFM_RANGE").copied().unwrap_or(0)
    };
    for (name, hash) in [("P1-S6 (178 k)", 178_000), ("P1-S6b", P1_LEAF_HEIGHTS.hash)] {
        let heights = ChipRows {
            hash,
            ..P1_LEAF_HEIGHTS
        };
        let p = leaf_partition_rows(&nine, front, heights).expect("the nine partition");
        let leaves: Vec<ChipRows> = p.leaves().iter().map(|l| sums(&nine, l, front)).collect();
        let card: f64 = leaves.iter().map(|&r| card_seconds(leaf_cells(r))).sum();
        println!(
            "S6b X1-SHAPE {name}: {} leaves {:?} · outside the socket {:?} M · card model Σ {card:.2} s",
            p.num_leaves(),
            p.leaves(),
            leaves
                .iter()
                .map(|&r| format!("{:.1}", leaf_cells(r) as f64 / 1e6))
                .collect::<Vec<_>>(),
        );
    }
}
