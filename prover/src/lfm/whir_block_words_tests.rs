//! The block leaf under the word memory format ([`MemoryFormat::Words`]): the
//! one table family it checks differently, PAGE_W, and the checks every route
//! is held to.
//!
//! - The page leg over PAGE_W's columns computes what the host's rebuild of
//!   the table's own preprocessed columns computes, on the three page kinds
//!   (an ELF page with data, a zero page, a private-input page), and refuses
//!   each preprocessed column's value changed; PAGE's leg, under the split
//!   format, is the control.
//! - The page route is the verifier's memory format's family, refused for a
//!   table of the other family's name.
//! - Every route checks exactly the columns its table declares, a page
//!   route's at the table's height: each mutant (a private page on an ELF
//!   page's route and the reverse, the other family's columns, a prepared
//!   opening settling one column fewer) is refused, and with the check taken
//!   out the leg admits a value nobody checks.
//! - A word-format plan derives from a hand-built statement and every leaf
//!   emits; under each route mutant the leaves are refused by the named check.
//!
//! The leaves over real word-format base proofs (forgeries included) are the
//! box tier's (`#[ignore]`): they prove blocks.

use multilinear::mle::Mle;
use stark::proof::options::ProofOptions;
use stark::traits::AIR;

use crate::block_whir::{BlockFormat, OwnedBlockStatement};
use crate::tables::fused_memory::MemoryFormat;
use crate::tables::page::PageConfig;
use crate::tables::types::{FE, FEE, GoldilocksExtension};

use super::super::builder::LfmBuilder;
use super::super::compiler::{LfmProgram, compile};
use super::super::executor::execute;
use super::super::validator::validate;
use super::super::whir_epoch::{emit_preprocessed_leg, preprocessed_targets};
use super::super::whir_table::TableVerdictWires;
use super::super::word::ext_word;
use super::route_faults::{self, Fault};
use super::{DynAir, LeafScale, Route, Shapes, WhirBlockPlan};
use super::{check_route_columns, page_columns, preprocessed_plan};

/// A fixed-seed xorshift, so a failure names one reproducible value.
fn pseudo(seed: u64, count: usize) -> Vec<FEE> {
    let mut state = seed | 1;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        FE::from(state >> 2)
    };
    (0..count)
        .map(|_| FEE::new([next(), next(), next()]))
        .collect()
}

const BASE: u64 = 0x2000_0000;

/// The three page kinds: an ELF page with data (zeros among it, words with
/// only some halves set), a zero page, a private-input page.
fn page_kinds() -> Vec<(&'static str, PageConfig)> {
    let data: Vec<u8> = (0..5_000u32)
        .map(|i| {
            if i % 7 == 3 || i % 11 == 0 {
                0
            } else {
                (i.wrapping_mul(37) + 11) as u8
            }
        })
        .collect();
    let private: Vec<u8> = (0..700u32).map(|i| (i * 13 + 5) as u8).collect();
    let size = crate::tables::page::DEFAULT_PAGE_SIZE as u64;
    vec![
        ("elf", PageConfig::with_data(BASE, data)),
        ("zero", PageConfig::zero_init(BASE + size)),
        (
            "private",
            PageConfig::with_private_input(BASE + 2 * size, private),
        ),
    ]
}

fn opts() -> ProofOptions {
    ProofOptions::default_test_options()
}

/// The PAGE_W table of `config`, the verifier's own (`VmAirs::with_word_memory`).
fn page_w_air(config: &PageConfig) -> crate::VmAir {
    crate::test_utils::create_page_w_air(&opts(), config).expect("a page-aligned page")
}

/// The leg alone over `air` at `num_vars` under `route`: the point and every
/// committed column's reduced value hinted (the point first), the leg
/// emitted as the leaf emits it ([`preprocessed_plan`]); where each of the
/// table's `declared` preprocessed columns' values sits among the column
/// values; and the number of column values.
fn leg_program(
    air: &DynAir<'_>,
    route: &Route,
    num_vars: usize,
    declared: usize,
) -> (LfmProgram, Vec<usize>, usize) {
    let width = air.trace_layout().0;
    let refs = [air];
    let owned = Shapes::build(&refs, &[(width, num_vars)], &[0]).expect("the page lays out");
    let shapes = owned.table_shapes();
    let shape = &shapes[0];
    let slot_of = owned.layouts[0].slot_of();
    let view = route.view();
    let plan = preprocessed_plan(route, &view, num_vars);
    let columns = shape.num_columns;
    let mut b = LfmBuilder::new().with_wrap_hash(super::super::edsl::WrapHash::production());
    let arena = b.declare_arena((num_vars + columns) as u32);
    let point: Vec<_> = (0..num_vars)
        .map(|i| b.hint_word(arena, i as u32).as_ext())
        .collect();
    let column_values: Vec<_> = (0..columns)
        .map(|i| b.hint_word(arena, (num_vars + i) as u32).as_ext())
        .collect();
    let zero = b.ext_const(&FEE::zero());
    let verdict = TableVerdictWires {
        bus_output: (zero, zero),
        point,
        column_values,
    };
    emit_preprocessed_leg(&mut b, &plan, slot_of, shape, &verdict);
    b.public(verdict.point[0].as_cell());
    let program = compile(b.finish());
    validate(&program).expect("the page leg must be admissible");
    let targets = preprocessed_targets(slot_of, shape.kinds, declared);
    assert!(
        targets.iter().all(|&t| t < columns),
        "{targets:?} of {columns} columns"
    );
    (program, targets, columns)
}

/// Executes the leg at `point` with `values`.
fn run_leg(program: &LfmProgram, point: &[FEE], values: &[FEE]) -> bool {
    let arena: Vec<_> = point.iter().chain(values).map(ext_word).collect();
    execute(program, &[arena], &crate::hash_pin::BLOCK_HASHER).is_ok()
}

/// The host's values: each of `columns` (the table's own preprocessed
/// columns) folded at `point`, at its target; the other columns `seed`'s.
fn honest_values(
    columns: &[Vec<FE>],
    targets: &[usize],
    width: usize,
    point: &[FEE],
    seed: u64,
) -> Vec<FEE> {
    let mut values = pseudo(seed, width);
    for (column, &target) in columns.iter().zip(targets) {
        values[target] = Mle::new(column.clone())
            .expect("a power-of-two column")
            .evaluate_in::<GoldilocksExtension>(point)
            .expect("the point has the column's variables");
    }
    values
}

/// ★ The leaf's page leg over PAGE_W computes what the host computes.
/// On each page kind the leg, on the route the leaf takes under the word
/// format ([`page_columns`], held by [`check_route_columns`]), executes at the
/// values the host's rebuild of the TABLE'S OWN preprocessed columns gives
/// (`air.precomputed_columns()`, the host verifier's route), at two points;
/// each preprocessed column's value one more is refused; a committed column
/// that is not preprocessed is free (the leg reads its targets alone). The
/// route checks 5 columns of `2^15` rows on an ELF and a zero page, OFFSET
/// alone on a private-input page.
#[test]
fn the_page_w_leg_computes_what_the_hosts_rebuild_computes() {
    let num_vars = crate::tables::page_w::NUM_VARS;
    for (kind, config) in page_kinds() {
        let air = page_w_air(&config);
        let columns = page_columns(MemoryFormat::Words, &config, air.name()).expect("a route");
        let declared = air.num_precomputed_columns();
        assert_eq!(
            (declared, columns.len()),
            if config.is_private_input {
                (1, 1)
            } else {
                (5, 5)
            },
            "{kind}"
        );
        assert!(columns.iter().all(|c| c.len() == 1 << num_vars), "{kind}");
        let host = air.precomputed_columns();
        assert_eq!(
            host, columns,
            "{kind}: the route's columns are the table's own"
        );
        let route = Route::Page(columns);
        check_route_columns(&route, declared, num_vars, air.name(), 0).expect("held");
        let (program, targets, width) = leg_program(air.as_ref(), &route, num_vars, declared);
        for seed in [0x5eed_0001u64, 0x5eed_0002] {
            let point = pseudo(seed, num_vars);
            let values = honest_values(&host, &targets, width, &point, seed ^ 0xA5);
            assert!(run_leg(&program, &point, &values), "{kind}: the honest leg");
            for (c, &t) in targets.iter().enumerate() {
                let mut wrong = values.clone();
                wrong[t] += FEE::one();
                assert!(
                    !run_leg(&program, &point, &wrong),
                    "{kind}: preprocessed column {c} changed is refused"
                );
            }
            let free = (0..width)
                .find(|c| !targets.contains(c))
                .expect("a committed column");
            let mut moved = values.clone();
            moved[free] += FEE::one();
            assert!(
                run_leg(&program, &point, &moved),
                "{kind}: column {free} is no preprocessed column"
            );
        }
        // The anti-vacuity of the INIT halves: on the ELF page every half
        // column has a nonzero entry, so the sparse leg folds something.
        if kind == "elf" {
            for (h, column) in host[1..].iter().enumerate() {
                assert!(
                    column.iter().any(|v| *v != FE::zero()),
                    "INIT half {h} has data"
                );
            }
        }
    }
}

/// The control: PAGE's leg under the split format, through the same route
/// code, on the same three page kinds (OFFSET and the byte INIT at `2^18`
/// rows, OFFSET alone on a private-input page).
#[test]
fn the_page_leg_under_the_split_format_is_unchanged() {
    let num_vars = crate::continuation::PAGE_NUM_VARS;
    for (kind, config) in page_kinds() {
        let air = crate::test_utils::create_page_air(&opts(), config.page_base);
        let columns = page_columns(MemoryFormat::Split, &config, air.name()).expect("a route");
        let declared = if config.is_private_input {
            crate::tables::page::NUM_PREPROCESSED_COLS_PRIVATE
        } else {
            crate::tables::page::NUM_PREPROCESSED_COLS
        };
        assert_eq!(columns.len(), declared, "{kind}");
        let route = Route::Page(columns.clone());
        check_route_columns(&route, declared, num_vars, air.name(), 0).expect("held");
        // The bare PAGE AIR (VmAirs attaches its columns): the leg reads the
        // declared count.
        let (program, targets, width) = leg_program(&air, &route, num_vars, declared);
        let point = pseudo(0x5eed_0003, num_vars);
        let values = honest_values(&columns, &targets, width, &point, 7);
        assert!(run_leg(&program, &point, &values), "{kind}: the honest leg");
        for &t in &targets {
            let mut wrong = values.clone();
            wrong[t] += FEE::one();
            assert!(!run_leg(&program, &point, &wrong), "{kind}: column {t}");
        }
    }
}

/// ★ The page route is the VERIFIER's memory format's family: PAGE_W's
/// columns under the word format, PAGE's under the split and fused ones, each
/// refused for a table named the other family's (whatever the proof's
/// format).
#[test]
fn the_page_route_is_the_verifiers_memory_formats_family() {
    for (kind, config) in page_kinds() {
        let w_name = page_w_air(&config).name().to_string();
        let p_name = crate::test_utils::create_page_air(&opts(), config.page_base)
            .name()
            .to_string();
        let words = page_columns(MemoryFormat::Words, &config, &w_name).expect("words");
        assert!(
            words.iter().all(|c| c.len() == crate::tables::page_w::ROWS),
            "{kind}"
        );
        for memory in [MemoryFormat::Split, MemoryFormat::Fused] {
            let split = page_columns(memory, &config, &p_name).expect("split");
            assert!(
                split
                    .iter()
                    .all(|c| c.len() == crate::tables::page::DEFAULT_PAGE_SIZE),
                "{kind} {memory:?}"
            );
            let refused = page_columns(memory, &config, &w_name).expect_err("a PAGE_W table");
            assert!(refused.contains("its pages are PAGE: tables"), "{refused}");
        }
        let refused =
            page_columns(MemoryFormat::Words, &config, &p_name).expect_err("a PAGE table");
        assert!(
            refused.contains("its pages are PAGE_W: tables"),
            "{refused}"
        );
    }
}

/// ★ The count check's mutants, at the route. On PAGE_W:
/// - a private-input page on an ELF page's route (5 columns for its 1) and an
///   ELF or zero page on a private page's (1 for 5): refused by the count;
/// - the other family's columns (PAGE's): refused by the count on an ELF or
///   zero page (2 for 5), by the height on a private page (`2^18` rows for
///   `2^15`);
/// - a prepared opening settling 4 of 5: refused by the count.
///
/// Each with its control, the check taken out: the swapped and the
/// other-family routes then pass the route checks.
#[test]
fn every_route_mutant_is_refused_by_its_named_check() {
    let num_vars = crate::tables::page_w::NUM_VARS;
    let route_of = |config: &PageConfig, name: &str| {
        let air = page_w_air(config);
        page_columns(MemoryFormat::Words, config, name).and_then(|columns| {
            let route = Route::Page(columns);
            check_route_columns(&route, air.num_precomputed_columns(), num_vars, name, 0)
                .map(|()| route.columns_checked())
        })
    };
    for (kind, config) in page_kinds() {
        let name = page_w_air(&config).name().to_string();
        let declared = page_w_air(&config).num_precomputed_columns();
        assert_eq!(route_of(&config, &name), Ok(declared), "{kind}: honest");

        let swapped = route_faults::with(Fault::PrivateRouteSwapped, || route_of(&config, &name));
        let refused = swapped.expect_err("the swapped route");
        assert!(
            refused.contains("preprocessed columns and its route"),
            "{kind}: {refused}"
        );
        let control = route_faults::with(Fault::PrivateRouteSwapped, || {
            route_faults::with(Fault::NoCountCheck, || route_of(&config, &name))
        });
        assert_eq!(
            control,
            Ok(if config.is_private_input { 5 } else { 1 }),
            "{kind}: without the count check the swapped route passes"
        );

        let other = route_faults::with(Fault::OtherPageFamily, || route_of(&config, &name));
        let refused = other.expect_err("the other family");
        let by = if config.is_private_input {
            "states 2^15 rows and its route's columns are 262144 rows"
        } else {
            "preprocessed columns and its route"
        };
        assert!(refused.contains(by), "{kind}: {refused}");
        let control = route_faults::with(Fault::OtherPageFamily, || {
            route_faults::with(Fault::NoCountCheck, || {
                route_faults::with(Fault::NoHeightCheck, || route_of(&config, &name))
            })
        });
        assert!(control.is_ok(), "{kind}: without both checks: {control:?}");
    }
    let five = Route::Prepared(4);
    let refused = check_route_columns(&five, 5, num_vars, "PAGE_W:0x0", 0).expect_err("4 of 5");
    assert!(refused.contains("declares 5"), "{refused}");
    assert!(
        route_faults::with(Fault::NoCountCheck, || {
            check_route_columns(&five, 5, num_vars, "PAGE_W:0x0", 0)
        })
        .is_ok()
    );
}

/// ★ The count check is load-bearing in the LEG: an ELF page on a private
/// page's route (OFFSET alone), the count check taken out, admits an INIT
/// half nobody checks; the leaf's own route refuses it (above), and on the
/// leaf's route the same value is refused by the leg.
#[test]
fn an_elf_page_on_the_private_route_leaves_its_init_unchecked() {
    let num_vars = crate::tables::page_w::NUM_VARS;
    let (_, config) = page_kinds().into_iter().next().expect("the ELF page");
    assert!(!config.is_private_input && config.init_values.is_some());
    let air = page_w_air(&config);
    let host = air.precomputed_columns();
    let full = Route::Page(page_columns(MemoryFormat::Words, &config, air.name()).expect("route"));
    let swapped = route_faults::with(Fault::PrivateRouteSwapped, || {
        page_columns(MemoryFormat::Words, &config, air.name()).map(Route::Page)
    })
    .expect("the swapped route's columns");
    assert_eq!(swapped.columns_checked(), 1);
    let point = pseudo(0x5eed_0004, num_vars);
    for (route, admitted) in [(full, false), (swapped, true)] {
        let (program, targets, width) = leg_program(air.as_ref(), &route, num_vars, 5);
        let values = honest_values(&host, &targets, width, &point, 9);
        assert!(run_leg(&program, &point, &values), "honest");
        let mut wrong = values.clone();
        wrong[targets[1]] += FEE::one();
        assert_eq!(
            run_leg(&program, &point, &wrong),
            admitted,
            "INIT half 0 changed, {} columns checked",
            route.columns_checked()
        );
    }
}

// =============================================================================
// Word-format plans over hand-built statements (laptop: emission, no proof)
// =============================================================================

/// A table family's tables, `(rows, count)` runs, by the family name the AIR
/// set gives its tables (`CPU`, `MEM_LOAD8`, `COPY_W`, …).
type Mix<'a> = &'a [(&'a str, &'a [(usize, usize)])];

const RUNTIME_PAGE_BASE: u64 = 0x4000_0000;

/// The block format these plans take: the production one over the Poseidon1
/// base (the tree's), `memory`'s loads and stores, at `leaf_scale`.
fn p1_format(memory: MemoryFormat, leaf_scale: LeafScale) -> BlockFormat {
    let production = BlockFormat::production();
    BlockFormat {
        zf: production
            .zf
            .with_base(stark::proof::options::BaseFormat::P1_WHIR),
        memory,
        leaf_scale,
        ..production
    }
}

/// A block statement over `elf_bytes` under `format`: `mix`'s tables, each
/// family's in run order; the ELF's pages, then `runtime_pages` zero pages
/// and `private_pages` private-input pages; the fixed tables at their
/// preprocessed columns' heights (HALT, which has none, at 32 rows); its
/// groups packed as the prover packs them
/// ([`stark::multilinear_block::block_groups`]). The AIR set is the frame's
/// (`crate::block_airs`, and under the word format
/// `VmAirs::with_word_memory`), so the heights land on the tables the
/// verifier builds. The word families' counts are its `word_memory`.
fn mix_statement(
    elf_bytes: &[u8],
    opts: &ProofOptions,
    format: &BlockFormat,
    mix: Mix<'_>,
    runtime_pages: usize,
    private_pages: usize,
) -> OwnedBlockStatement {
    use crate::tables::word_cpu::WORD_TABLES;
    let elf = executor::elf::Elf::load(elf_bytes).expect("load the ELF");
    let count = |family: &str| -> usize {
        mix.iter()
            .filter(|(f, _)| *f == family)
            .flat_map(|(_, runs)| runs.iter().map(|&(_, c)| c))
            .sum()
    };
    let counts = crate::TableCounts {
        cpu: count("CPU"),
        lt: count("LT"),
        memw: count("MEMW"),
        memw_aligned: count("MEMW_A"),
        load: count("LOAD"),
        mul: count("MUL"),
        dvrm: count("DVRM"),
        shift: count("SHIFT"),
        branch: count("BRANCH"),
        memw_register: count("MEMW_R"),
        eq: count("EQ"),
        bytewise: count("BYTEWISE"),
        store: count("STORE"),
        cpu32: count("CPU32"),
        keccak: count("KECCAK") + count(crate::tables::keccak_w::NAME),
        keccak_rnd: count("KECCAK_RND"),
        ecsm: count("ECSM"),
        ecdas: count("ECDAS"),
        hint: count("HINT"),
        copy: count("COPY") + count(crate::tables::copy_w::NAME),
        blake3: 0,
        is_wh: 0,
        ecrec: count("ECREC") + count(crate::tables::ecrec_w::NAME),
    };
    let words: [usize; 7] = WORD_TABLES.map(count);
    let ranges = if runtime_pages > 0 {
        vec![crate::RuntimePageRange {
            base: RUNTIME_PAGE_BASE,
            count: runtime_pages as u64,
        }]
    } else {
        Vec::new()
    };
    let page_configs = crate::tables::trace_builder::Traces::page_configs_from_elf_and_runtime(
        &elf,
        &ranges,
        private_pages,
        1 << 16,
    )
    .expect("page configs");
    let mut runs: std::collections::HashMap<&str, Vec<usize>> = mix
        .iter()
        .map(|&(family, runs)| {
            (
                family,
                runs.iter()
                    .flat_map(|&(rows, c)| std::iter::repeat_n(rows, c))
                    .rev()
                    .collect(),
            )
        })
        .collect();
    let cpu_heights: Vec<usize> = runs
        .get("CPU")
        .map(|r| r.iter().rev().copied().collect())
        .unwrap_or_default();
    let inputs =
        crate::BlockAirInputs::new(opts, &elf, cpu_heights, &[0; 4]).expect("the block's inputs");
    let mut airs = crate::block_airs(
        &elf,
        opts,
        &page_configs,
        &counts,
        &inputs,
        format.keccak_rnd,
        format.memw_aligned,
        format.registers,
    );
    if format.memory == MemoryFormat::Words {
        airs.with_word_memory(opts, words, &page_configs)
            .expect("the word tables");
    }
    let rows: Vec<usize> = airs
        .air_refs()
        .iter()
        .enumerate()
        .map(|(i, air)| {
            let name = air.name();
            if i < crate::FIXED_TABLE_COUNT {
                return air.precomputed_columns().first().map_or(32, Vec::len);
            }
            if name.starts_with(crate::tables::page_w::NAME_PREFIX) {
                return crate::tables::page_w::ROWS;
            }
            if name.starts_with(super::PAGE_NAME_PREFIX) {
                return crate::tables::page::DEFAULT_PAGE_SIZE;
            }
            let family = name.split('[').next().expect("a name");
            runs.get_mut(family)
                .and_then(Vec::pop)
                .unwrap_or_else(|| panic!("{name}: no height in the mix"))
        })
        .collect();
    assert!(
        runs.values().all(Vec::is_empty),
        "every height of the mix lands on a table"
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
        num_private_input_pages: private_pages,
        groups,
        fused_memory: [0, 0],
        word_memory: words.map(|n| n as u64),
    }
}

/// A small word-format block: every CPU word family, MEMW_W and COPY_W, a
/// few CPU and ALU tables, two zero pages and a private-input page.
const SMALL_WORDS: Mix<'static> = &[
    ("CPU", &[(1 << 12, 2)]),
    ("LT", &[(1 << 10, 1)]),
    ("BRANCH", &[(1 << 9, 1)]),
    ("MEMW_R", &[(1 << 10, 1)]),
    ("COPY_W", &[(1 << 8, 1)]),
    ("MEM_LOAD8", &[(1 << 11, 1), (1 << 9, 1)]),
    ("MEM_STORE8", &[(1 << 10, 1)]),
    ("MEM_LOADW", &[(1 << 9, 1)]),
    ("MEM_STOREW", &[(1 << 8, 1)]),
    ("MEM_LOAD2W", &[(1 << 6, 1)]),
    ("MEM_STORE2W", &[(1 << 6, 1)]),
    ("MEMW_W", &[(1 << 9, 1)]),
];

fn small_words_plan(format: &BlockFormat) -> Result<WhirBlockPlan, String> {
    let elf: &[u8] = include_bytes!("testdata/poc_rodata_commit.elf");
    let opts = super::super::proof::block_base_options();
    let owned = mix_statement(elf, &opts, format, SMALL_WORDS, 2, 1);
    WhirBlockPlan::derive(elf, &opts, format, owned.view(), None)
}

/// ★ A word-format plan derives from a hand-built statement (every word
/// family, PAGE_W tables of each kind) and every leaf emits: the leaf takes
/// the word format's tables. Under the split format the same statement is
/// refused by the frame (it states word tables), so a word block is never
/// planned as a split one.
#[test]
fn a_word_plan_derives_and_every_leaf_emits() {
    let words = p1_format(MemoryFormat::Words, LeafScale::DoubledOneTop);
    let plan = small_words_plan(&words).expect("the word plan derives");
    let names: Vec<String> = plan
        .frame
        .airs
        .air_refs()
        .iter()
        .map(|a| a.name().to_string())
        .collect();
    for family in crate::tables::word_cpu::WORD_TABLES {
        assert!(
            names.iter().any(|n| n.starts_with(&format!("{family}["))),
            "{family} in the plan"
        );
    }
    let pages: Vec<&PageConfig> = plan.frame.page_configs.iter().collect();
    assert!(pages.iter().any(|c| c.is_private_input), "a private page");
    assert!(
        pages
            .iter()
            .any(|c| c.init_values.is_some() && !c.is_private_input)
    );
    assert!(pages.iter().any(|c| c.init_values.is_none()));
    assert!(
        names
            .iter()
            .any(|n| n.starts_with(crate::tables::page_w::NAME_PREFIX))
            && !names.iter().any(|n| n.starts_with(super::PAGE_NAME_PREFIX))
    );
    for k in 0..plan.partition().num_leaves() {
        plan.leaf_program(k).expect("the leaf emits");
    }
    println!(
        "WORD PLAN: {} groups · {} leaves · {} prepared",
        plan.num_groups(),
        plan.partition().num_leaves(),
        plan.prepared().len()
    );
    let opts = super::super::proof::block_base_options();
    let elf: &[u8] = include_bytes!("testdata/poc_rodata_commit.elf");
    let owned = mix_statement(elf, &opts, &words, SMALL_WORDS, 2, 1);
    for other in [MemoryFormat::Split, MemoryFormat::Fused] {
        let refused = WhirBlockPlan::derive(
            elf,
            &opts,
            &p1_format(other, LeafScale::DoubledOneTop),
            owned.view(),
            None,
        )
        .err()
        .expect("refused");
        assert!(
            refused.contains("states no word table"),
            "{other:?}: {refused}"
        );
    }
}

/// Every leaf of `plan` emitted, or the first refusal.
fn emit_leaves(plan: &WhirBlockPlan) -> Result<(), String> {
    (0..plan.partition().num_leaves()).try_for_each(|k| plan.leaf_program(k).map(|_| ()))
}

/// ★ The route mutants through a whole plan's leaves (emitted on this
/// thread, as the tree emits each): each refuses the leaves by its named
/// check, and with that check taken out (the control) every leaf emits. The
/// prepared mutant over pages forced onto the prepared opening (every ELF
/// page with data, as the dense rule takes a dense one). The other family's
/// columns with both route checks out reach the leg, whose own emit-time
/// assertion is a second line, not the refusal.
#[test]
fn every_route_mutant_refuses_the_word_leaves() {
    let words = p1_format(MemoryFormat::Words, LeafScale::DoubledOneTop);
    let plan = small_words_plan(&words).expect("the honest plan");
    emit_leaves(&plan).expect("the honest leaves");
    let under = |faults: &[Fault]| -> Result<(), String> {
        faults.iter().rev().fold(
            Box::new(|| emit_leaves(&plan)) as Box<dyn Fn() -> _>,
            |f, &fault| Box::new(move || route_faults::with(fault, &f)),
        )()
    };
    let swapped = under(&[Fault::PrivateRouteSwapped]).expect_err("refused");
    assert!(
        swapped.contains("preprocessed columns and its route checks"),
        "{swapped}"
    );
    under(&[Fault::PrivateRouteSwapped, Fault::NoCountCheck]).expect("without the count check");
    let other = under(&[Fault::OtherPageFamily]).expect_err("refused");
    assert!(
        other.contains("preprocessed columns and its route checks"),
        "{other}"
    );
    let leg = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        under(&[
            Fault::OtherPageFamily,
            Fault::NoCountCheck,
            Fault::NoHeightCheck,
        ])
    }));
    let message = leg
        .expect_err("the leg's own assertion")
        .downcast::<String>()
        .map(|s| *s)
        .unwrap_or_default();
    assert!(
        message.contains("one entry per row of its table"),
        "{message}"
    );
    // The prepared opening: DECODE and every ELF page with data.
    crate::block_whir::DENSE_PAGES_FORCED.with(|c| c.set(true));
    let checked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let plan = small_words_plan(&words).expect("the honest plan over prepared pages");
        let pages: usize = plan
            .prepared()
            .iter()
            .flat_map(|p| p.tables.iter())
            .filter(|&&(_, _, n)| n == crate::tables::page_w::NUM_PREPROCESSED_COLS)
            .count();
        assert!(pages > 0, "a PAGE_W table on the prepared opening");
        emit_leaves(&plan).expect("the honest leaves over prepared pages");
        let fewer =
            route_faults::with(Fault::SettleOneFewer, || emit_leaves(&plan)).expect_err("refused");
        assert!(
            fewer.contains("preprocessed columns and its route checks"),
            "{fewer}"
        );
        route_faults::with(Fault::SettleOneFewer, || {
            route_faults::with(Fault::NoCountCheck, || emit_leaves(&plan))
        })
        .expect("without the count check");
    }));
    crate::block_whir::DENSE_PAGES_FORCED.with(|c| c.set(false));
    if let Err(panic) = checked {
        std::panic::resume_unwind(panic);
    }
}
