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
            rows.extend(std::iter::repeat_n(32, elf_pages));
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
            .map(|c| format!("{}={}/{}x{}", c.name, c.real_rows, c.rows, c.main_cols + c.aux_cols))
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
            .max_by_key(|&k| plan.partition().leaf(k).iter().map(|&g| costs[g]).sum::<usize>())
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
            .max_by_key(|&k| plan.partition().leaf(k).iter().map(|&g| costs[g]).sum::<usize>())
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
