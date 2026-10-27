//! D-P3B P2's node census: the W3 tree's nodes over the median block's
//! Poseidon1-base leaves, by emission alone, under today's recursion (the
//! legacy RPX family: binary trees, the format's `auto` cap) and under the
//! Poseidon1 family at 4-ary caps C1, C2 and C4 — the rows `c_L` is chosen
//! from. A node's cells depend on its children's table heights and their
//! trees, never on a root, so the children's artifacts are their heights only
//! (`registry::census_artifacts`): no commit, no proof.
//!
//! Run alone: `cargo test --release -p lambda-vm-prover --lib --
//! --ignored --exact lfm::p1_node_census_tests::p1_node_census --nocapture`
//! (`CENSUS_ELF` for a real guest; `CENSUS_LEVELS=1` stops after level 1;
//! `CENSUS_ARMS=c1,c2` runs only those of `rpx`, `c1`, `c2`, `c4`).
//! Prints one `NODES …` line per level and arm and one `NODES … total` line
//! per arm.

use std::collections::BTreeMap;

use stark::proof::options::{BaseFormat, ProofOptions};

use crate::block_whir::BlockFormat;
use crate::hash_pin::{LegacyFamily, P1Family, ProofFamily};

use super::compiler::LfmProgram;
use super::p1_commit::{P1StarkHashCapped, P1Transcript};
use super::per_table_aggregator::DerivedChild;
use super::registry::{LfmArtifacts, census_artifacts};
use super::whir_block::WhirBlockPlan;

/// The Poseidon1 family at 4-ary cap `C`: the census's other heights.
struct P1At<const C: u8>;

impl<const C: u8> ProofFamily for P1At<C> {
    type Stark = P1StarkHashCapped<C>;
    type Transcript = P1Transcript;
    const STATEMENT_TAG: &'static [u8] = P1Family::STATEMENT_TAG;
    const WRAP: super::edsl::WrapHash = P1Family::WRAP;
    const SOCKET: super::hash::HasherKind = P1Family::SOCKET;
    fn transcript(seed: &[u8]) -> Self::Transcript {
        P1Family::transcript(seed)
    }
}

/// A program's cells over every chip: padded (main + aux), its `LFM_HASH`
/// share, every chip's padded rows, and the unpadded cells and `LFM_SELECT`
/// rows (what the walk order moves below a padding step).
struct Cells {
    padded: u64,
    hash: u64,
    rows: BTreeMap<&'static str, u64>,
    real: u64,
    select_real: u64,
}

fn cells_of(program: &LfmProgram) -> Cells {
    let chips = super::airs::lfm_chip_census_with_hasher(
        program,
        program.hasher(crate::hash_pin::BLOCK_HASHER),
    );
    let mut out = Cells {
        padded: 0,
        hash: 0,
        rows: BTreeMap::new(),
        real: 0,
        select_real: 0,
    };
    for c in &chips {
        let cells = c.main_cells() + c.aux_cells();
        out.padded += cells;
        out.real += c.real_rows * (c.main_cols + c.aux_cols) as u64;
        if c.name == "LFM_HASH" {
            out.hash += cells;
        }
        if c.name == "LFM_SELECT" {
            out.select_real += c.real_rows;
        }
        let e: &mut u64 = out.rows.entry(c.name).or_default();
        *e = (*e).max(c.rows);
    }
    out
}

/// A program's heights-only artifacts, under the socket its hash rows run on.
fn heights_of(p: &LfmProgram) -> LfmArtifacts {
    census_artifacts(p, p.hasher(crate::hash_pin::BLOCK_HASHER))
}

/// Every node level of `plan`'s tree over the leaves (their heights-only
/// artifacts) under the family `Fam`: each node emitted over its children's
/// shapes (theirs under `Fam`), counted and dropped, level by level, so one
/// compiled program is alive at a time.
fn census<Fam: ProofFamily>(
    label: &str,
    plan: &WhirBlockPlan,
    leaves: &[LfmArtifacts],
    wrap: &ProofOptions,
    max_levels: usize,
) {
    let words = plan.child_layout().total();
    let derive = |artifacts: &LfmArtifacts| {
        DerivedChild::from_artifacts_under::<Fam>(artifacts, wrap, words)
            .unwrap_or_else(|e| panic!("{label}: {e}"))
    };
    let started = std::time::Instant::now();
    let mut below: Vec<DerivedChild> = leaves.iter().map(derive).collect();
    let shape = plan.levels();
    let (mut total, mut total_hash, mut nodes) = (0u64, 0u64, 0usize);
    let (mut total_real, mut total_select) = (0u64, 0u64);
    for (lv, arities) in shape.iter().enumerate().take(max_levels) {
        let top = lv + 1 == shape.len();
        let (mut level_cells, mut level_hash, mut level_max) = (0u64, 0u64, 0u64);
        let (mut level_real, mut level_select, mut select_max) = (0u64, 0u64, 0u64);
        let mut chips: BTreeMap<&'static str, u64> = BTreeMap::new();
        let mut next = Vec::with_capacity(arities.arities.len());
        let mut at = 0usize;
        for &a in &arities.arities {
            let kids: Vec<&DerivedChild> = below[at..at + a].iter().collect();
            at += a;
            let program = plan
                .node_program_under::<Fam>(&kids, top)
                .unwrap_or_else(|e| panic!("{label} level {}: {e}", lv + 1));
            let Cells {
                padded: cells,
                hash,
                rows,
                real,
                select_real,
            } = cells_of(&program);
            level_cells += cells;
            level_hash += hash;
            level_max = level_max.max(cells);
            level_real += real;
            level_select += select_real;
            select_max = select_max.max(select_real);
            for (n, r) in rows {
                let e = chips.entry(n).or_default();
                *e = (*e).max(r);
            }
            let heights = heights_of(&program);
            drop(program);
            next.push(derive(&heights));
        }
        nodes += arities.arities.len();
        total += level_cells;
        total_hash += level_hash;
        total_real += level_real;
        total_select += level_select;
        println!(
            "NODES {label} L{}: {} nodes · cells Σ {:.1} M (max {:.1} M) · LFM_HASH Σ {:.1} M · \
             unpadded Σ {:.1} M · LFM_SELECT real Σ {} (max {}) · padded max {}",
            lv + 1,
            arities.arities.len(),
            level_cells as f64 / 1e6,
            level_max as f64 / 1e6,
            level_hash as f64 / 1e6,
            level_real as f64 / 1e6,
            level_select,
            select_max,
            chips
                .iter()
                .filter(|(_, r)| **r > 0)
                .map(|(n, r)| format!("{n}=2^{}", r.trailing_zeros()))
                .collect::<Vec<_>>()
                .join(" ")
        );
        below = next;
    }
    println!(
        "NODES {label} total: {nodes} nodes · cells Σ {:.1} M · LFM_HASH Σ {:.1} M · unpadded Σ {:.1} M \
         · LFM_SELECT real Σ {total_select} · {:.1} s",
        total as f64 / 1e6,
        total_hash as f64 / 1e6,
        total_real as f64 / 1e6,
        started.elapsed().as_secs_f64()
    );
}

/// ★ The census (instrument, `--exact`): the median mix under the Poseidon1
/// base (`BaseFormat::P1_WHIR`, its leaf partition), every leaf emitted once,
/// then the node levels under today's RPX recursion and under P1 at C1, C2
/// and C4.
#[test]
#[ignore = "instrument: run alone with --exact (CENSUS_ELF for a real guest)"]
fn p1_node_census() {
    let elf_bytes = match std::env::var("CENSUS_ELF") {
        Ok(path) => std::fs::read(path).expect("read CENSUS_ELF"),
        Err(_) => crate::test_utils::asm_elf_bytes("poc_rodata_commit"),
    };
    let max_levels = std::env::var("CENSUS_LEVELS")
        .ok()
        .map(|v| v.parse().expect("CENSUS_LEVELS is a level count"))
        .unwrap_or(usize::MAX);
    let opts = super::proof::block_base_options();
    let production = BlockFormat::production();
    let format = BlockFormat {
        zf: production.zf.with_base(BaseFormat::P1_WHIR),
        ..production
    };
    let owned = super::whir_p1_census_tests::median_statement(&elf_bytes, &opts, &format);
    let plan = WhirBlockPlan::derive(&elf_bytes, &opts, &format, owned.view(), None)
        .expect("the plan derives");
    let started = std::time::Instant::now();
    // One leaf program at a time: only its heights are kept.
    let leaves: Vec<LfmArtifacts> = (0..plan.partition().num_leaves())
        .map(|k| heights_of(&plan.leaf_program(k).expect("the leaf emits")))
        .collect();
    println!(
        "NODES plan: {} groups · {} leaves · levels {:?} · leaves emitted {:.1} s",
        plan.num_groups(),
        leaves.len(),
        plan.levels()
            .iter()
            .map(|l| l.arities.clone())
            .collect::<Vec<_>>(),
        started.elapsed().as_secs_f64()
    );
    let wrap = super::proof::aggregation_wrap_options();
    let arms = std::env::var("CENSUS_ARMS").unwrap_or_else(|_| "rpx,c1,c2,c4".to_string());
    let arm = |name: &str| arms.split(',').any(|a| a == name);
    if arm("rpx") {
        census::<LegacyFamily>("today (RPX recursion)", &plan, &leaves, &wrap, max_levels);
    }
    if arm("c1") {
        census::<P1At<1>>("P1 C1", &plan, &leaves, &wrap, max_levels);
    }
    if arm("c2") {
        census::<P1At<2>>("P1 C2", &plan, &leaves, &wrap, max_levels);
    }
    if arm("c4") {
        census::<P1At<4>>("P1 C4", &plan, &leaves, &wrap, max_levels);
    }
}

/// The census's heights-only artifacts describe the program's real shape:
/// the trivial program's heights, chunk lists and chip set are its real
/// artifacts', so a node derived from either has the same tables.
#[test]
fn the_census_artifacts_are_the_real_ones_shape() {
    let opts = stark::proof::options::GoldilocksCubicProofOptions::with_blowup(2).expect("opts");
    let program = super::programs::trivial_program();
    let real = super::registry::build_artifacts(&program, &opts);
    let census = census_artifacts(&program, real.hasher);
    assert_eq!(census.log_heights, real.log_heights);
    assert_eq!(census.keccak_rnd_chunks, real.keccak_rnd_chunks);
    assert_eq!(
        census.blake3_chunk_log_heights,
        real.blake3_chunk_log_heights
    );
    assert_eq!(census.hash_chunk_log_heights, real.hash_chunk_log_heights);
    assert_eq!(census.chip_set, real.chip_set);
    let words = 4;
    let a = DerivedChild::from_artifacts(&census, &opts, words).expect("derives");
    let b = DerivedChild::from_artifacts(&real, &opts, words).expect("derives");
    assert_eq!(a.log2_trace_lengths(), b.log2_trace_lengths());
}
