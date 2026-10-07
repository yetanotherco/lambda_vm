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

/// ★ Every group of the median mix as a one-group Poseidon1 leaf, against the
/// plan's rows for it — the lead's acceptance criterion (10-06): the rows the
/// partition admits leaves by ARE the emitted ones, for every group shape the
/// mix has, both prepared kinds included (DECODE's stack in group 0, the ELF
/// pages' genesis), with no fallback: Select, base ALU and words exactly (the
/// words from the arena counts alone), socket rows at most the plan's and
/// within 64 of it. Then every leaf the plan builds, emitted, stays under
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
    let production = BlockFormat::production();
    let format = BlockFormat {
        zf: production.zf.with_base(BaseFormat::P1_WHIR),
        max_groups: 1024,
        ..production
    };
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
    let n = plan.num_groups();
    let costs = plan.costs().to_vec();
    let prepared: Vec<usize> = plan.prepared().iter().map(|p| p.group).collect();
    let (loads, front) = plan.chip_loads();
    let (loads, front) = (loads.to_vec(), front);
    if real.is_some() {
        assert!(
            prepared.contains(&0) && prepared.iter().any(|&g| g != 0),
            "a real guest's mix has both prepared kinds, got groups {prepared:?}"
        );
    }
    let plan = plan.with_partition(
        BlockPartition::new((0..n).map(|g| vec![g]).collect(), n).expect("one group a leaf"),
    );
    let mut max: BTreeMap<&'static str, (usize, usize)> = BTreeMap::new();
    // Group 0 (the fixed tables at the instrument's placeholder heights and
    // the accelerators) does not emit here; the real block's leaf census does.
    for (g, &cost) in costs.iter().enumerate() {
        let program = plan.leaf_program(g).expect("the leaf emits");
        let mut rows: BTreeMap<&'static str, usize> = BTreeMap::new();
        for instr in &program.instrs {
            let k = match kind(instr) {
                k if k.starts_with("BALU") => "BALU",
                k => k,
            };
            *rows.entry(k).or_default() += 1;
        }
        let get = |k: &str| rows.get(k).copied().unwrap_or(0);
        let planned = loads[g].plus(front);
        assert_eq!(
            (get("SELECT"), get("BALU"), get("HINT")),
            (planned.select, planned.balu, planned.hint),
            "group {g} (prepared {}): Select, base ALU and words are the plan's",
            prepared.contains(&g)
        );
        assert!(
            get("HASH16") <= planned.hash && get("HASH16") + 64 >= planned.hash,
            "group {g}: socket rows {} against the plan's {}",
            get("HASH16"),
            planned.hash
        );
        println!(
            "CENSUS GROUP {g}: tables {} · prepared {} · cost {cost} · HASH16 {} · SELECT {} · HINT {} · BALU {} · XALU {}",
            owned.groups[g].len(),
            prepared.contains(&g),
            get("HASH16"),
            get("SELECT"),
            get("HINT"),
            get("BALU"),
            get("XALU"),
        );
        for k in ["HASH16", "SELECT", "HINT", "BALU", "XALU"] {
            let e = max.entry(k).or_insert((0, 0));
            if get(k) > e.0 {
                *e = (get(k), g);
            }
        }
    }
    println!("CENSUS GROUP MAX: {max:?}");

    // The plan at the production cap: every leaf, against its chips' heights.
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
    for k in 0..plan.partition().num_leaves() {
        let program = plan.leaf_program(k).expect("the leaf emits");
        let mut rows: BTreeMap<&'static str, usize> = BTreeMap::new();
        for instr in &program.instrs {
            let kk = match kind(instr) {
                kk if kk.starts_with("BALU") => "BALU",
                kk => kk,
            };
            *rows.entry(kk).or_default() += 1;
        }
        let get = |kk: &str| rows.get(kk).copied().unwrap_or(0);
        let over: Vec<&str> = [
            ("HASH16", P1_LEAF_HEIGHTS.hash),
            ("SELECT", P1_LEAF_HEIGHTS.select),
            ("HINT", P1_LEAF_HEIGHTS.hint),
            ("BALU", P1_LEAF_HEIGHTS.balu),
        ]
        .iter()
        .filter(|(kk, h)| get(kk) > *h)
        .map(|(kk, _)| *kk)
        .collect();
        println!(
            "CENSUS PLAN LEAF {k}: groups {:?} · HASH16 {} · SELECT {} · HINT {} · BALU {} · over {over:?}",
            plan.partition().leaf(k),
            get("HASH16"),
            get("SELECT"),
            get("HINT"),
            get("BALU"),
        );
        assert!(
            over.is_empty(),
            "leaf {k} is past a chip's height: {over:?}"
        );
    }
}

// ======================== the leaves' chip heights ========================

use super::whir_block::{P1_LEAF_HEIGHTS, leaf_partition_rows};
use super::whir_chain::ChipRows;

fn rows(hash: usize, select: usize, balu: usize, hint: usize) -> ChipRows {
    ChipRows {
        hash,
        select,
        balu,
        hint,
    }
}

/// The census's shapes (median mix, an ethrex ELF): a full group, a prepared
/// one (DECODE's stack, the ELF pages' genesis).
const FULL: ChipRows = ChipRows {
    hash: 38_646,
    select: 139_878,
    balu: 142_236,
    hint: 140_290,
};
const PREPARED: ChipRows = ChipRows {
    hash: 48_656,
    select: 172_938,
    balu: 180_090,
    hint: 185_114,
};

fn sums(loads: &[ChipRows], leaf: &[usize], front: ChipRows) -> ChipRows {
    leaf.iter().fold(front, |acc, &g| acc.plus(loads[g]))
}

fn under(r: ChipRows, h: ChipRows) -> bool {
    r.hash <= h.hash && r.select <= h.select && r.balu <= h.balu && r.hint <= h.hint
}

/// ★ The partition admits no leaf past a chip's height, whichever chip binds:
/// groups light in the socket but heavy in `Select` (or hints) are split
/// where a socket-only partition would double that chip silently. Four full
/// groups share a leaf, a fifth does not (the socket cap holds RPX's leaf
/// count). Every group is placed once, over no more leaves than the rows need,
/// and a group no leaf holds is refused.
#[test]
fn the_p1_partition_keeps_every_chip_under_its_height() {
    let front = rows(300, 0, 0, 240);
    let small = rows(9_000, 60_000, 40_000, 50_000);
    for loads in [
        vec![PREPARED, FULL, FULL],
        vec![PREPARED, FULL, FULL, FULL, FULL, FULL, PREPARED, small],
        [vec![PREPARED; 2], vec![FULL; 77], vec![small]].concat(),
    ] {
        let partition = leaf_partition_rows(&loads, front, P1_LEAF_HEIGHTS).expect("partitions");
        for leaf in partition.leaves() {
            assert!(
                under(sums(&loads, leaf, front), P1_LEAF_HEIGHTS),
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
    // Four full groups share a leaf, and so do a prepared one and three full;
    // a fifth full group does not fit the socket cap. The 1× block's nine
    // groups close in three leaves, the median's ninety in RPX's twenty-three.
    let p = leaf_partition_rows(&[FULL; 4], front, P1_LEAF_HEIGHTS).expect("fits");
    assert_eq!(p.num_leaves(), 1, "four full groups are one leaf");
    let p =
        leaf_partition_rows(&[PREPARED, FULL, FULL, FULL], front, P1_LEAF_HEIGHTS).expect("fits");
    assert_eq!(p.num_leaves(), 1, "a prepared group and three full ones");
    let p = leaf_partition_rows(&[FULL; 5], front, P1_LEAF_HEIGHTS).expect("fits");
    assert_eq!(
        p.num_leaves(),
        2,
        "a fifth full group is past the socket cap"
    );
    let nine = [vec![FULL; 7], vec![PREPARED; 2]].concat();
    let p = leaf_partition_rows(&nine, front, P1_LEAF_HEIGHTS).expect("fits");
    assert_eq!(p.num_leaves(), 3, "the 1× block's nine groups");
    let ninety = [vec![FULL; 88], vec![PREPARED; 2]].concat();
    let p = leaf_partition_rows(&ninety, front, P1_LEAF_HEIGHTS).expect("fits");
    assert_eq!(
        p.num_leaves(),
        23,
        "the median's ninety groups, RPX's leaf count"
    );
    // Select binds, or the hints: groups light in the socket are split.
    for heavy in [
        rows(10_000, 600_000, 10_000, 10_000),
        rows(10_000, 10_000, 10_000, 600_000),
    ] {
        let p = leaf_partition_rows(&[heavy, heavy], front, P1_LEAF_HEIGHTS).expect("fits");
        assert_eq!(
            p.num_leaves(),
            2,
            "{heavy:?} twice is past 2^20: split, not doubled"
        );
    }
    // Four groups that fit the socket cap (160 k) but not Select (1.2 M): the
    // partition splits them; with its chip check off — the socket-only
    // partition RPX keeps — they share a leaf whose Select crosses 2^20.
    let selecty = rows(40_000, 300_000, 100_000, 100_000);
    let set = [selecty; 4];
    let p = leaf_partition_rows(&set, front, P1_LEAF_HEIGHTS).expect("fits");
    assert_eq!(p.num_leaves(), 2, "split by Select");
    for leaf in p.leaves() {
        assert!(under(sums(&set, leaf, front), P1_LEAF_HEIGHTS));
    }
    let costs: Vec<usize> = set.iter().map(|r| r.hash).collect();
    let socket_only =
        super::whir_block::leaf_partition(&costs, None, P1_LEAF_HEIGHTS.hash - front.hash)
            .expect("the socket-only partition");
    assert!(
        socket_only
            .leaves()
            .iter()
            .any(|leaf| sums(&set, leaf, front).select > P1_LEAF_HEIGHTS.select),
        "the check is what keeps Select under 2^20: {:?}",
        socket_only.leaves()
    );
    // Refusals: a group past a height, a front that fills a chip.
    let huge = rows(1_000, 1_200_000, 0, 0);
    assert!(leaf_partition_rows(&[FULL, huge], front, P1_LEAF_HEIGHTS).is_err());
    assert!(leaf_partition_rows(&[FULL], rows(0, 1 << 20, 0, 0), P1_LEAF_HEIGHTS).is_err());
}

/// ★ The heights hold on the median mix, as emitted (the lead's condition on
/// child order, 10-06): the plan's per-group rows are the emitted ones (a
/// group's Select and base ALU are its chains', its words its arena's), a leaf
/// of four full groups stays under 2^20 `Select`, hint and base-ALU rows and
/// the socket cap (one 2^18 table), and so does every leaf the plan builds. A
/// change that grows a group's rows past four a leaf fails here, and the
/// partition then takes another leaf rather than a doubled chip.
#[test]
fn the_p1_leaves_stay_under_their_heights_on_the_median_mix() {
    let elf_bytes = crate::test_utils::asm_elf_bytes("poc_rodata_commit");
    let opts = super::proof::block_base_options();
    let production = BlockFormat::production();
    let format = BlockFormat {
        zf: production.zf.with_base(BaseFormat::P1_WHIR),
        max_groups: 1024,
        ..production
    };
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
            under(sums(&loads, leaf, front), P1_LEAF_HEIGHTS),
            "leaf {leaf:?}"
        );
    }
    // The rows are the emitted ones: groups 1 (full) and 0 (prepared) alone.
    let partition = BlockPartition::new(
        [
            vec![vec![0], vec![1], vec![2, 3, 4, 5]],
            vec![(6..n).collect()],
        ]
        .concat(),
        n,
    )
    .expect("covers once");
    let probe = plan.with_partition(partition);
    for (k, list) in [vec![0usize], vec![1]].iter().enumerate() {
        let emitted = ChipRows::of(&probe.leaf_program(k).expect("the leaf emits"));
        let planned = sums(&loads, list, front);
        assert_eq!(
            (emitted.select, emitted.balu, emitted.hint),
            (planned.select, planned.balu, planned.hint),
            "group {list:?}: Select, base ALU and words are the plan's"
        );
        assert!(
            emitted.hash <= planned.hash && emitted.hash + 64 >= planned.hash,
            "group {list:?}: socket rows {} against the plan's {}",
            emitted.hash,
            planned.hash
        );
    }
    // Four full groups, emitted.
    let four = ChipRows::of(&probe.leaf_program(2).expect("the leaf emits"));
    println!("P1 FOUR FULL GROUPS: {four:?} against {P1_LEAF_HEIGHTS:?}");
    assert!(under(four, P1_LEAF_HEIGHTS), "four full groups: {four:?}");
}

/// ★ The S6 census (instrument, `--exact`; `CENSUS_ELF` for a real guest): the
/// median mix's leaves under each candidate partition — RPX's, P1 at S6's
/// [`LEAF_P1_CAP`](super::whir_block::LEAF_P1_CAP) (four groups a leaf), at S5's
/// 2^17 (three), and at a 2^18 room (all the socket table holds) — every leaf
/// emitted: its padded chip heights, its cells, the tree above it, and 056's
/// time model (a leaf ≈ 1.24 s + 0.0121 s a million cells; I-WHIR-P1 §S5.8).
/// Prints `S6 …` lines; asserts S6's leaves under every bounded height.
#[test]
#[ignore = "instrument: run alone with --exact (CENSUS_ELF for a real guest)"]
fn whir_s6_arm_census() {
    let elf_bytes = match std::env::var("CENSUS_ELF") {
        Ok(path) => std::fs::read(path).expect("read CENSUS_ELF"),
        Err(_) => crate::test_utils::asm_elf_bytes("poc_rodata_commit"),
    };
    let opts = super::proof::block_base_options();
    let production = BlockFormat::production();
    let arms: [(&str, BaseFormat, Option<usize>); 4] = [
        ("RPX", BaseFormat::RPX, None),
        ("P1-S6", BaseFormat::P1_WHIR, None),
        ("P1-S5-2^17", BaseFormat::P1_WHIR, Some(1 << 17)),
        ("P1-2^18", BaseFormat::P1_WHIR, Some(262_000)),
    ];
    for (name, base, cap) in arms {
        let format = BlockFormat {
            zf: production.zf.with_base(base),
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
            cap.unwrap_or_else(|| super::whir_block::leaf_cap_for(&format.zf.base)),
        )
        .expect("the plan derives");
        let levels = plan.levels();
        let nodes: usize = levels.iter().map(|l| l.arities.len()).sum();
        let mut max: BTreeMap<&'static str, u64> = BTreeMap::new();
        let (mut cells_sum, mut model, mut cells_max) = (0u64, 0f64, 0u64);
        let mut groups_max = 0usize;
        for k in 0..plan.partition().num_leaves() {
            let program = plan.leaf_program(k).expect("the leaf emits");
            let chips = super::airs::lfm_chip_census_with_hasher(
                &program,
                program.hasher(crate::hash_pin::BLOCK_HASHER),
            );
            let cells: u64 = chips.iter().map(|c| c.main_cells() + c.aux_cells()).sum();
            for c in &chips {
                let e = max.entry(c.name).or_default();
                *e = (*e).max(c.rows);
            }
            if base == BaseFormat::P1_WHIR && cap.is_none() {
                let rows = ChipRows::of(&program);
                assert!(
                    under(rows, P1_LEAF_HEIGHTS),
                    "S6 leaf {k} past a bounded height: {rows:?}"
                );
            }
            cells_sum += cells;
            cells_max = cells_max.max(cells);
            groups_max = groups_max.max(plan.partition().leaf(k).len());
            model += 1.24 + 0.0121 * cells as f64 / 1e6;
        }
        println!(
            "S6 {name}: {} groups · {} leaves (≤ {groups_max} groups) · {nodes} nodes over {} node levels · leaf cells Σ {:.1} M, max {:.1} M · model Σ leaf time {model:.1} s · padded max {}",
            plan.num_groups(),
            plan.partition().num_leaves(),
            levels.len(),
            cells_sum as f64 / 1e6,
            cells_max as f64 / 1e6,
            max.iter()
                .map(|(n, r)| format!("{n}={r}"))
                .collect::<Vec<_>>()
                .join(" ")
        );
    }
}
