//! I-PADLEAF stage 1: the padded partition's chip model against emitted leaves,
//! and the block-mix instrument that sizes v3 against v2 (I-PADLEAF §5).

use std::time::Instant;

use super::block_leaf::{BlockPartition, seed_leaves};
use super::block_plan::{
    BlockShape, BlockTreePlan, LEAF_PERMS_CAP, PARTITION_COST_MODEL, partition_for, partition_model,
};
use super::leaf_pack::{
    ChipModel, chip_rows, pack, padded_partition, partition_cells, spread_start,
};

/// A block's instance mix in AIR order after the fixed tables: each table
/// kind's `(rows, count)` runs, and the zero-init runtime pages (2^18 rows each)
/// beside the ELF's own.
struct Mix {
    tables: &'static [(&'static str, &'static [(usize, usize)])],
    pages: usize,
}

/// ULTRA mc12 00-warmup's BLOCK CENSUS (#1013, the 1× bench block 25368371):
/// 137 instances, 43 pages.
const BENCH_MIX: Mix = Mix {
    tables: &[
        ("COMMIT", &[(256, 1)]),
        ("KECCAK", &[(1 << 14, 1)]),
        ("KECCAK_RND", &[(1 << 16, 4)]),
        ("ECSM", &[(128, 1)]),
        ("ECDAS", &[(1 << 16, 1)]),
        ("HINT", &[(128, 1)]),
        ("CPU", &[(1 << 21, 15)]),
        ("LT", &[(1 << 21, 6), (1 << 20, 2), (1 << 19, 3)]),
        ("SHIFT", &[(1 << 21, 1), (1 << 17, 1)]),
        ("MEMW", &[(1 << 21, 1)]),
        ("MEMW_A", &[(1 << 21, 5), (1 << 17, 1)]),
        ("LOAD", &[(1 << 21, 3), (1 << 18, 1)]),
        ("MUL", &[(1 << 17, 1)]),
        ("DVRM", &[(1 << 10, 1)]),
        ("BRANCH", &[(1 << 19, 1), (1 << 17, 1)]),
        ("MEMW_R", &[(1 << 21, 30)]),
        ("EQ", &[(1 << 16, 1)]),
        ("BYTEWISE", &[(1 << 20, 1), (1 << 16, 1)]),
        ("STORE", &[(1 << 21, 2), (1 << 20, 1)]),
        ("CPU32", &[(1 << 17, 1)]),
    ],
    pages: 43,
};

/// ULTRA mc12 01-run's BLOCK CENSUS (the median 25475471): 941 instances, 198
/// pages. Per kind the census gives the count and the total rows; the runs
/// below match both (LT, whose real split the census does not give, as
/// 2^21 and 2^20 halves).
const MEDIAN_MIX: Mix = Mix {
    tables: &[
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
    ],
    pages: 198,
};

/// ULTRA mc14 01-run's BLOCK CENSUS (full gas 25431071, the p90 proxy): 1725
/// instances, 291 pages; LT, BRANCH and BYTEWISE tails folded as in
/// [`MEDIAN_MIX`].
const FULL_GAS_MIX: Mix = Mix {
    tables: &[
        ("COMMIT", &[(256, 1)]),
        ("KECCAK", &[(1 << 18, 1)]),
        ("KECCAK_RND", &[(1 << 16, 49)]),
        ("ECSM", &[(2048, 1)]),
        ("ECDAS", &[(1 << 17, 4), (1 << 16, 1)]),
        ("HINT", &[(2048, 1)]),
        ("CPU", &[(1 << 21, 288)]),
        ("LT", &[(1 << 21, 101), (1 << 20, 77)]),
        ("SHIFT", &[(1 << 21, 25)]),
        ("MEMW", &[(1 << 21, 7), (1 << 20, 1)]),
        ("MEMW_A", &[(1 << 21, 101)]),
        ("LOAD", &[(1 << 21, 62), (1 << 20, 1)]),
        ("MUL", &[(1 << 20, 1), (1 << 19, 1), (1 << 18, 1)]),
        ("DVRM", &[(1 << 16, 1)]),
        (
            "BRANCH",
            &[(1 << 19, 15), (1 << 18, 1), (1 << 17, 1), (1 << 14, 11)],
        ),
        ("MEMW_R", &[(1 << 21, 591), (1 << 20, 1)]),
        ("EQ", &[(1 << 17, 14), (1 << 15, 1)]),
        (
            "BYTEWISE",
            &[(1 << 20, 11), (1 << 18, 1), (1 << 16, 1), (1 << 14, 9)],
        ),
        ("STORE", &[(1 << 21, 44), (1 << 18, 1)]),
        ("CPU32", &[(1 << 21, 1), (1 << 18, 1)]),
    ],
    pages: 291,
};

/// Every table kind at a small height: leaf programs a test emits in
/// milliseconds, over the production kinds.
const SMALL_MIX: Mix = Mix {
    tables: &[
        ("COMMIT", &[(256, 1)]),
        ("KECCAK", &[(1 << 12, 1)]),
        ("KECCAK_RND", &[(1 << 12, 2)]),
        ("ECSM", &[(128, 1)]),
        ("ECDAS", &[(1 << 12, 2)]),
        ("HINT", &[(128, 1)]),
        ("CPU", &[(1 << 12, 3), (1 << 11, 1)]),
        ("LT", &[(1 << 12, 2)]),
        ("SHIFT", &[(1 << 12, 1)]),
        ("MEMW", &[(1 << 12, 1)]),
        ("MEMW_A", &[(1 << 12, 2)]),
        ("LOAD", &[(1 << 12, 1)]),
        ("MUL", &[(1 << 12, 1)]),
        ("DVRM", &[(1 << 10, 1)]),
        ("BRANCH", &[(1 << 12, 1)]),
        ("MEMW_R", &[(1 << 12, 3), (1 << 10, 1)]),
        ("EQ", &[(1 << 12, 1)]),
        ("BYTEWISE", &[(1 << 12, 1)]),
        ("STORE", &[(1 << 12, 1)]),
        ("CPU32", &[(1 << 12, 1)]),
    ],
    pages: 6,
};

/// The zero-init runtime pages' base: page-aligned and above the fixture ELF.
const RUNTIME_PAGE_BASE: u64 = 0x4000_0000;

/// `mix` as a block shape over `elf`: the fixed tables and the ELF's pages at
/// the honest fixture's heights, every other instance at its mix height, and
/// runtime pages up to the mix's page count.
fn mix_shape(elf: &executor::elf::Elf, mix: &Mix) -> BlockShape {
    let elf_pages = crate::tables::trace_builder::Traces::page_configs_from_elf(elf).len();
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
    let runs = |kind: &str| -> Vec<usize> {
        mix.tables
            .iter()
            .filter(|(k, _)| *k == kind)
            .flat_map(|(_, runs)| runs.iter())
            .flat_map(|&(rows, n)| std::iter::repeat_n(rows, n))
            .collect()
    };
    let mut lengths = vec![32usize; crate::FIXED_TABLE_COUNT];
    // AIR order (`VmAirs::air_refs`): the accelerators, HINT, CPU … BRANCH,
    // the pages, MEMW_R … CPU32.
    for kind in [
        "COMMIT",
        "KECCAK",
        "KECCAK_RND",
        "ECSM",
        "ECDAS",
        "HINT",
        "CPU",
        "LT",
        "SHIFT",
        "MEMW",
        "MEMW_A",
        "LOAD",
        "MUL",
        "DVRM",
        "BRANCH",
        "PAGE",
        "MEMW_R",
        "EQ",
        "BYTEWISE",
        "STORE",
        "CPU32",
    ] {
        if kind == "PAGE" {
            lengths.extend(std::iter::repeat_n(32, elf_pages));
            lengths.extend(std::iter::repeat_n(1 << 18, mix.pages - elf_pages));
            continue;
        }
        let rows = runs(kind);
        let n = rows.len();
        match kind {
            "COMMIT" => counts.commit = n,
            "KECCAK" => counts.keccak = n,
            "KECCAK_RND" => counts.keccak_rnd = n,
            "ECSM" => counts.ecsm = n,
            "ECDAS" => counts.ecdas = n,
            "HINT" => counts.hint = n,
            "CPU" => counts.cpu = n,
            "LT" => counts.lt = n,
            "SHIFT" => counts.shift = n,
            "MEMW" => counts.memw = n,
            "MEMW_A" => counts.memw_aligned = n,
            "LOAD" => counts.load = n,
            "MUL" => counts.mul = n,
            "DVRM" => counts.dvrm = n,
            "BRANCH" => counts.branch = n,
            "MEMW_R" => counts.memw_register = n,
            "EQ" => counts.eq = n,
            "BYTEWISE" => counts.bytewise = n,
            "STORE" => counts.store = n,
            "CPU32" => counts.cpu32 = n,
            other => unreachable!("{other}"),
        }
        lengths.extend(rows);
    }
    BlockShape {
        table_counts: counts,
        runtime_page_ranges: vec![crate::RuntimePageRange {
            base: RUNTIME_PAGE_BASE,
            count: (mix.pages - elf_pages) as u64,
        }],
        num_private_input_pages: 0,
        public_output_len: 4,
        trace_lengths: lengths,
    }
}

fn mix_plan(mix: &Mix) -> BlockTreePlan {
    let elf_bytes = crate::test_utils::asm_elf_bytes("poc_rodata_commit");
    let elf = executor::elf::Elf::load(&elf_bytes).expect("load the ELF");
    let shape = mix_shape(&elf, mix);
    BlockTreePlan::derive(&elf_bytes, &super::proof::block_base_options(), &shape)
        .expect("the mix's plan derives")
}

fn join(v: &[u64]) -> String {
    v.iter().map(u64::to_string).collect::<Vec<_>>().join(",")
}

/// ★ I-PADLEAF stage 1 instrument (laptop, `--exact`): a block mix's v2 plan,
/// its chip model, and the model against emitted v2 leaves.
/// `PADLEAF_MIX=bench|median|full` picks the mix (median by default);
/// `PADLEAF_VALIDATE=<n>` emits the first `n` leaves and the last (3 by
/// default). Prints `PADLEAF …` lines a reader parses.
#[test]
#[ignore = "laptop instrument: run alone with --exact"]
fn padleaf_mix_instrument() {
    let which = std::env::var("PADLEAF_MIX").unwrap_or_else(|_| "median".to_string());
    let mix = match which.as_str() {
        "bench" => &BENCH_MIX,
        "median" => &MEDIAN_MIX,
        "full" => &FULL_GAS_MIX,
        other => panic!("PADLEAF_MIX must be bench, median or full, got {other}"),
    };
    let t = Instant::now();
    let plan = mix_plan(mix);
    println!(
        "PADLEAF PLAN {which}: {} instances · v{} · {} leaves · derive {:.2}s",
        plan.num_instances(),
        plan.cost_model(),
        plan.partition().num_leaves(),
        t.elapsed().as_secs_f64()
    );
    let t = Instant::now();
    let model = ChipModel::probe(&plan).expect("the model probes");
    println!("PADLEAF PROBE: {:.2}s", t.elapsed().as_secs_f64());
    println!("PADLEAF CHIPS {}", model.names().join(","));
    println!("PADLEAF WIDTHS {}", join(model.widths()));
    println!("PADLEAF FRONT {}", join(model.front()));
    println!("PADLEAF CARRIER {}", join(model.carrier()));
    let costs = plan.costs();
    for (i, cost) in costs.iter().enumerate() {
        println!(
            "PADLEAF INST {i} {} {cost} {}",
            plan.instance(i).name,
            join(model.instance(i))
        );
    }
    let names: Vec<&str> = plan.instances().iter().map(|i| i.name.as_str()).collect();
    let partition = &partition_for(&names, &costs).expect("v2 partitions");
    let mut v2_cells = 0u64;
    for (k, list) in partition.leaves().iter().enumerate() {
        let rows = model.leaf_rows(list, k == plan.carrier());
        v2_cells += model.padded_cells(&rows);
        println!(
            "PADLEAF V2 {k} {} {}",
            list.iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(","),
            join(&rows)
        );
    }
    println!("PADLEAF V2 CELLS {v2_cells}");
    println!(
        "PADLEAF V2 STEPS {}",
        steps(&model, partition.leaves(), plan.carrier())
    );
    let (seeds, seeded) = seed_leaves(&names, partition.num_leaves());
    let start = spread_start(&model, &seeds, &seeded, plan.carrier());
    for (label, lists) in [
        (
            "from v2",
            pack(&model, partition.leaves(), &seeded, plan.carrier()),
        ),
        ("spread", start.clone()),
        ("from spread", pack(&model, &start, &seeded, plan.carrier())),
    ] {
        let cells = partition_cells(&model, &lists, plan.carrier());
        println!(
            "PADLEAF V3 {label}: {cells} ({:+.1} M) · {}",
            (cells as f64 - v2_cells as f64) / 1e6,
            steps(&model, &lists, plan.carrier())
        );
    }
    let t = Instant::now();
    let v3 = super::leaf_pack::padded_partition(&plan).expect("v3 derives");
    let v3_cells = partition_cells(&model, v3.leaves(), plan.carrier());
    println!(
        "PADLEAF V3 CELLS {v3_cells} ({:+.1} M, {:+.2} %) · pack {:.2}s",
        (v3_cells as f64 - v2_cells as f64) / 1e6,
        100.0 * (v3_cells as f64 - v2_cells as f64) / v2_cells as f64,
        t.elapsed().as_secs_f64()
    );
    println!(
        "PADLEAF V3 STEPS {}",
        steps(&model, v3.leaves(), plan.carrier())
    );
    let n: usize = std::env::var("PADLEAF_VALIDATE")
        .ok()
        .map(|v| v.parse().expect("PADLEAF_VALIDATE is a count"))
        .unwrap_or(3);
    let leaves = partition.num_leaves();
    let mut ks: Vec<usize> = (0..n.min(leaves)).collect();
    if !ks.contains(&(leaves - 1)) {
        ks.push(leaves - 1);
    }
    for (label, p) in [("v2", partition), ("v3", &v3)] {
        for &k in &ks {
            let mut b = super::block_plan::leaf_builder();
            super::block_leaf::emit_block_leaf_over(&mut b, &plan, p, k, k == plan.carrier());
            let program = super::compiler::compile(b.finish());
            let actual: Vec<u64> = chip_rows(&program).iter().map(|c| c.1).collect();
            let modelled = model.leaf_rows(p.leaf(k), k == plan.carrier());
            println!(
                "PADLEAF CHECK {label} {k} actual {} model {}",
                join(&actual),
                join(&modelled)
            );
        }
    }
}

/// Per stepped chip, how many leaves sit at each padded height.
fn steps(model: &ChipModel, lists: &[Vec<usize>], carrier: usize) -> String {
    let rows: Vec<Vec<u64>> = lists
        .iter()
        .enumerate()
        .map(|(l, list)| model.leaf_rows(list, l == carrier))
        .collect();
    let mut out = Vec::new();
    for (c, name) in model.names().iter().enumerate() {
        let mut heights: std::collections::BTreeMap<u64, usize> = Default::default();
        for r in &rows {
            if *name == "LFM_HASH" {
                continue;
            }
            *heights
                .entry(super::layout::padded_rows(r[c] as usize) as u64)
                .or_default() += 1;
        }
        if heights.len() > 1 {
            let h: Vec<String> = heights.iter().map(|(p, n)| format!("{p}x{n}")).collect();
            out.push(format!("{name}:{}", h.join(",")));
        }
    }
    out.join(" ")
}

/// ★ The chip model is the leaves it models: over a three-leaf partition of
/// every table kind, each emitted leaf's real rows equal the front, the
/// carrier term and its instances' forks — exactly, but for the two terms the
/// model does not add up by construction: `LFM_CONST` (constants pooled by
/// value, modelled as the largest probe's pool) and `LFM_XALU`'s one
/// extension add per bus contribution summed after the first.
#[test]
fn every_leaf_is_its_chip_model() {
    let plan = mix_plan(&SMALL_MIX);
    let model = ChipModel::probe(&plan).expect("the model probes");
    let n = plan.num_instances();
    let lists: Vec<Vec<usize>> = (0..3)
        .map(|k| (0..n).filter(|i| i % 3 == k).collect())
        .collect();
    let partition = BlockPartition::new(lists, n).expect("covers once");
    let names = model.names();
    let xalu = names.iter().position(|&c| c == "LFM_XALU").expect("XALU");
    for k in 0..3 {
        let carries = k == plan.carrier();
        let mut b = super::block_plan::leaf_builder();
        super::block_leaf::emit_block_leaf_over(&mut b, &plan, &partition, k, carries);
        let actual = chip_rows(&super::compiler::compile(b.finish()));
        let modelled = model.leaf_rows(partition.leaf(k), carries);
        assert_eq!(actual.len(), names.len(), "leaf {k}: the chip set");
        for (c, (name, rows, _)) in actual.iter().enumerate() {
            assert_eq!(*name, names[c], "leaf {k}: chip order");
            match *name {
                "LFM_CONST" => {}
                "LFM_XALU" => assert!(
                    rows.abs_diff(modelled[c]) <= partition.leaf(k).len() as u64,
                    "leaf {k}: XALU {rows} against {}",
                    modelled[c]
                ),
                _ => assert_eq!(*rows, modelled[c], "leaf {k}: {name}"),
            }
        }
        assert!(
            actual[xalu].1 >= modelled[xalu],
            "leaf {k}: the adds are extra"
        );
    }
}

/// v2 is the default and its partition is untouched: a plan derived without
/// `NOEPOCH_PARTITION_MODEL` carries version 2 and the rule's partition.
#[test]
fn the_partition_model_defaults_to_v2() {
    if std::env::var_os("NOEPOCH_PARTITION_MODEL").is_some() {
        return;
    }
    assert_eq!(partition_model(), PARTITION_COST_MODEL);
    let plan = mix_plan(&SMALL_MIX);
    assert_eq!(plan.cost_model(), 2);
    let names: Vec<&str> = plan.instances().iter().map(|i| i.name.as_str()).collect();
    assert_eq!(
        plan.partition(),
        &partition_for(&names, &plan.costs()).expect("v2 partitions")
    );
}

/// ★ v3 on the bench block's mix (8 leaves): it models strictly fewer cells
/// than v2 at v2's leaf count, keeps every seed on the rule's leaf, keeps
/// every leaf's `LFM_HASH` rows under the cap, and derives the same lists
/// twice.
#[test]
fn the_padded_partition_models_fewer_cells_and_keeps_the_seeds() {
    let plan = mix_plan(&BENCH_MIX);
    let names: Vec<&str> = plan.instances().iter().map(|i| i.name.as_str()).collect();
    let v2 = partition_for(&names, &plan.costs()).expect("v2 partitions");
    let v3 = padded_partition(&plan).expect("v3 partitions");
    assert_eq!(v3, padded_partition(&plan).expect("again"), "deterministic");
    assert_eq!(v3.num_leaves(), v2.num_leaves());
    let model = ChipModel::probe(&plan).expect("the model probes");
    let carrier = plan.carrier();
    let (cells2, cells3) = (
        partition_cells(&model, v2.leaves(), carrier),
        partition_cells(&model, v3.leaves(), carrier),
    );
    assert!(cells3 < cells2, "v3 {cells3} against v2 {cells2}");
    let leaf_of = |p: &BlockPartition, i: usize| p.leaves().iter().position(|l| l.contains(&i));
    let (_, seeded) = seed_leaves(&names, v2.num_leaves());
    for i in (0..names.len()).filter(|&i| seeded[i]) {
        assert_eq!(leaf_of(&v3, i), leaf_of(&v2, i), "seed {} moved", names[i]);
    }
    for (k, list) in v3.leaves().iter().enumerate() {
        let rows = model.leaf_rows(list, k == carrier);
        assert!(model.hash_rows(&rows) <= LEAF_PERMS_CAP as u64, "leaf {k}");
    }
}

/// The padded partition is part of the verifier's identity (it decides every
/// leaf's instance list at version 3), so its lists on the bench mix are
/// pinned: a change to the probes, the cost, the search or its starts fails
/// here until [`super::block_plan::PARTITION_MODEL_PADDED`] is bumped with it.
/// The lists, not the modelled cells: the cells carry the pooled
/// `LFM_CONST` term, which moves with the fixture ELF's bytes (laptop and box
/// clang differ) while no decision does.
#[test]
fn the_padded_partition_is_pinned_to_its_version() {
    let plan = mix_plan(&BENCH_MIX);
    let v3 = padded_partition(&plan).expect("v3 partitions");
    let bytes: Vec<u8> = v3
        .leaves()
        .iter()
        .flat_map(|l| l.iter().map(|&i| i as u16).chain([u16::MAX]))
        .flat_map(u16::to_le_bytes)
        .collect();
    let digest = crate::statement::elf_digest(&bytes);
    assert_eq!(
        (
            super::block_plan::PARTITION_MODEL_PADDED,
            v3.num_leaves(),
            digest[..8]
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect::<String>(),
        ),
        (3, 8, "5c8f0f2312f3d795".to_string()),
        "the padded partition's output moved: bump PARTITION_MODEL_PADDED with it"
    );
}
