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
    let leaves = |faults: &[Fault]| under(faults, || emit_leaves(&plan));
    let swapped = leaves(&[Fault::PrivateRouteSwapped]).expect_err("refused");
    assert!(
        swapped.contains("preprocessed columns and its route checks"),
        "{swapped}"
    );
    leaves(&[Fault::PrivateRouteSwapped, Fault::NoCountCheck]).expect("without the count check");
    let other = leaves(&[Fault::OtherPageFamily]).expect_err("refused");
    assert!(
        other.contains("preprocessed columns and its route checks"),
        "{other}"
    );
    let leg = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        leaves(&[
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

// =============================================================================
// The leaf census under the word format
// =============================================================================

/// `rows` in tables of at most `cap` rows: full ones, then the rest padded to
/// a power of two.
fn runs_of(rows: usize, cap: usize) -> Vec<(usize, usize)> {
    let mut runs = vec![(cap, rows / cap)];
    if rows % cap > 0 {
        runs.push(((rows % cap).next_power_of_two(), 1));
    }
    runs.retain(|&(_, c)| c > 0);
    runs
}

/// A block's tables, by family: the families the word format keeps, the
/// split format's memory tables, the word format's, and the pages.
struct CensusMix {
    kept: Vec<(&'static str, Vec<(usize, usize)>)>,
    split_memory: Vec<(&'static str, Vec<(usize, usize)>)>,
    words: Vec<(&'static str, Vec<(usize, usize)>)>,
    pages: usize,
}

/// A family's tables of `heights` (variables), as runs.
fn of_vars(heights: &[usize]) -> Vec<(usize, usize)> {
    heights.iter().map(|&v| (1usize << v, 1)).collect()
}

/// ✗ A MODEL of the median and the large block (Poseidon1 base, in-row
/// registers), never a measurement: each family's table count, and LT's,
/// KECCAK_RND's, COPY's, KECCAK's and ECDAS's heights, are the split
/// pipeline's on the box (`W3 GROUP TABLES`, `W3 LT HEIGHTS`, `W3 CHUNKED`
/// lines); the other families' heights are guessed at their caps (CPU, SHIFT,
/// the memory tables, MEMW_R at 2^21; BYTEWISE, MUL, CPU32 at 2^20; BRANCH at
/// 2^19; EQ at 2^18; DVRM, ECREC at 2^14). The word families' rows are the
/// word-format design's counts of the CPU's accesses (median, large), at
/// 2^21 rows a table; MEMW_W's (the chips' word accesses) are guessed at 9 M
/// and 18 M rows. The census prints its tables and cells beside the box's
/// (split 403 / 12.657 G and 781 / 25.486 G; words 346 / 10.469 G and 666 /
/// 21.043 G) so the model's error is read, not assumed.
fn census_mix(large: bool) -> CensusMix {
    const LT_MEDIAN: [usize; 38] = [
        20, 20, 20, 17, 15, 18, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 17, 15, 16, 16, 18, 19,
        19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 18, 18, 19, 17,
    ];
    const LT_LARGE: [usize; 74] = [
        20, 20, 20, 20, 20, 20, 18, 15, 15, 15, 15, 18, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19,
        19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 18, 15, 15, 15, 15, 16, 16, 15,
        17, 19, 19, 19, 19, 19, 19, 19, 19, 20, 20, 20, 20, 20, 20, 19, 20, 19, 19, 18, 19, 19, 18,
        17, 17, 19, 19, 18,
    ];
    let k = if large { 2 } else { 1 };
    let n = |median: usize, large_n: usize| if large { large_n } else { median };
    let m = 1usize << 20;
    // The word families' rows (millions): MEM_LOAD8, MEM_STORE8, MEM_LOADW,
    // MEM_STOREW, MEM_LOAD2W, MEM_STORE2W, MEMW_W.
    let word_rows: [f64; 7] = if large {
        [46.42, 33.79, 13.22, 4.09, 4.74, 3.92, 18.0]
    } else {
        [22.72, 16.68, 6.30, 1.95, 2.31, 1.90, 9.0]
    };
    let words = crate::tables::word_cpu::WORD_TABLES
        .iter()
        .zip(word_rows)
        .map(|(&family, rows)| (family, runs_of((rows * 1e6) as usize, 2 * m)))
        .collect();
    CensusMix {
        kept: vec![
            ("CPU", vec![(2 * m, n(58, 117))]),
            (
                "LT",
                of_vars(if large { &LT_LARGE[..] } else { &LT_MEDIAN[..] }),
            ),
            ("KECCAK_RND", vec![(1 << 16, 33 * k)]),
            ("SHIFT", vec![(2 * m, n(5, 9))]),
            ("BRANCH", vec![(1 << 19, n(5, 11))]),
            ("EQ", vec![(1 << 18, n(5, 9))]),
            ("BYTEWISE", vec![(m, n(4, 7))]),
            ("ECDAS", vec![(1 << 17, 2 * k)]),
            ("MUL", vec![(m, k)]),
            ("DVRM", vec![(1 << 14, 1)]),
            ("MEMW_R", vec![(2 * m, k)]),
            ("CPU32", vec![(m, 1)]),
            ("COPY", vec![(m, 4 * k)]),
            ("KECCAK", vec![(1 << 17, k)]),
            ("ECREC", vec![(1 << 14, 1)]),
        ],
        split_memory: vec![
            ("MEMW_A", vec![(2 * m, n(25, 51))]),
            ("LOAD", vec![(2 * m, n(15, 31))]),
            ("STORE", vec![(2 * m, n(10, 20))]),
            ("MEMW", vec![(2 * m, n(5, 9))]),
        ],
        words,
        pages: n(183, 351),
    }
}

/// The mix's tables under `memory`, by the names its AIR set gives them.
fn census_tables(
    mix: &CensusMix,
    memory: MemoryFormat,
) -> Vec<(&'static str, Vec<(usize, usize)>)> {
    let rename = |family: &'static str| match (memory, family) {
        (MemoryFormat::Words, "COPY") => crate::tables::copy_w::NAME,
        (MemoryFormat::Words, "KECCAK") => crate::tables::keccak_w::NAME,
        (MemoryFormat::Words, "ECREC") => crate::tables::ecrec_w::NAME,
        _ => family,
    };
    let mut tables: Vec<(&'static str, Vec<(usize, usize)>)> = mix
        .kept
        .iter()
        .map(|(f, runs)| (rename(f), runs.clone()))
        .collect();
    let memory_tables = if memory == MemoryFormat::Words {
        &mix.words
    } else {
        &mix.split_memory
    };
    tables.extend(memory_tables.iter().cloned());
    tables
}

/// One census line for `plan`: its tables, cells, groups and leaves; each
/// group's mean rows in the leaf's binding chips; and how many such groups a
/// leaf at the plan's heights holds, chip by chip.
fn census_line(label: &str, plan: &WhirBlockPlan) -> (usize, usize) {
    let cells: f64 = plan
        .frame
        .shapes
        .iter()
        .map(|&(w, v)| (w as f64) * (1u64 << v) as f64)
        .sum::<f64>()
        / 1e9;
    let (loads, front) = plan.chip_loads();
    let n = loads.len();
    let heights = plan
        .leaf_bound()
        .map_or(super::P1_LEAF_HEIGHTS_2X, |b| b.heights);
    let names = super::super::whir_chain::ChipRows::NAMES;
    let sums = loads.iter().fold([0usize; 10], |mut acc, l| {
        for (a, r) in acc.iter_mut().zip(l.to_array()) {
            *a += r;
        }
        acc
    });
    let room: Vec<String> = names
        .iter()
        .zip(sums)
        .zip(heights.to_array())
        .zip(front.to_array())
        .filter(|(((_, s), h), _)| *s > 0 && *h < usize::MAX && *h > 0)
        .map(|(((name, s), h), f)| {
            let mean = s as f64 / n as f64;
            format!("{name} {:.0}/group → {:.1}", mean, (h - f) as f64 / mean)
        })
        .collect();
    let leaves = plan.partition().num_leaves();
    println!(
        "WORDS CENSUS {label}: tables {} · cells {cells:.3} G · groups {n} · leaves {leaves} \
         ({:?}) · groups a leaf {:?} · costs Σ {} · a doubled leaf holds, by chip: {}",
        plan.frame.shapes.len(),
        plan.leaf_bound(),
        plan.partition()
            .leaves()
            .iter()
            .map(Vec::len)
            .collect::<Vec<_>>(),
        plan.costs().iter().sum::<usize>(),
        room.join(" · ")
    );
    (n, leaves)
}

/// ★ The word format's leaf census (laptop instrument, `--exact`; ✗ a model,
/// [`census_mix`]): the median and the large block's plans under the split
/// format (the control) and the word format, at the default leaf scale (every
/// leaf doubled, one top over at most three), each leaf emitted and held
/// under its heights. Prints, per plan, the groups, the leaves, and how many
/// of its groups a doubled leaf holds by each chip: the number that says
/// whether the box's word-format group counts (29 at the median, 57–59 at
/// the large, under a BLAKE3 base) take 2 and 3 leaves or 2 and 4.
#[test]
#[ignore = "laptop instrument: run alone with --exact"]
fn words_leaf_census() {
    let elf: &[u8] = include_bytes!("testdata/poc_rodata_commit.elf");
    let opts = super::super::proof::block_base_options();
    let elf_pages = crate::tables::trace_builder::Traces::page_configs_from_elf(
        &executor::elf::Elf::load(elf).expect("the ELF"),
    )
    .len();
    for large in [false, true] {
        let mix = census_mix(large);
        for memory in [MemoryFormat::Split, MemoryFormat::Words] {
            let format = p1_format(memory, LeafScale::DoubledOneTop);
            let tables = census_tables(&mix, memory);
            let refs: Vec<(&str, &[(usize, usize)])> = tables
                .iter()
                .map(|(f, runs)| (*f, runs.as_slice()))
                .collect();
            let owned = mix_statement(elf, &opts, &format, &refs, mix.pages - elf_pages, 0);
            let plan = WhirBlockPlan::derive(elf, &opts, &format, owned.view(), None)
                .expect("the plan derives");
            let label = format!("{} {memory:?}", if large { "large" } else { "median" });
            census_line(&label, &plan);
            let bound = plan.leaf_bound().expect("bounded");
            for k in 0..plan.partition().num_leaves() {
                let rows = super::super::whir_chain::ChipRows::of(
                    &plan.leaf_program(k).expect("the leaf emits"),
                );
                assert!(rows.under(bound.heights), "{label} leaf {k}: {rows:?}");
            }
        }
    }
}

/// ★ The census over a REAL base proof (instrument, `--exact`): the plan of
/// the base proof in `CENSUS_BASE_FILE` (a `prove-block-base -o` file) over
/// the ELF in `CENSUS_ELF`, under `CENSUS_MEMORY` (`split` or `words`,
/// default `words`) and the Poseidon1 base, at the default leaf scale; every
/// leaf emitted and held under its heights. The box runs it on the median
/// and the large block's word-format base proofs.
#[test]
#[ignore = "instrument: CENSUS_BASE_FILE and CENSUS_ELF, run alone with --exact"]
fn words_leaf_census_of_a_base_file() {
    let path = std::env::var("CENSUS_BASE_FILE").expect("CENSUS_BASE_FILE");
    let elf = std::fs::read(std::env::var("CENSUS_ELF").expect("CENSUS_ELF")).expect("the ELF");
    let memory = match std::env::var("CENSUS_MEMORY").as_deref() {
        Ok("split") => MemoryFormat::Split,
        Ok("fused") => MemoryFormat::Fused,
        _ => MemoryFormat::Words,
    };
    let bytes = std::fs::read(&path).expect("the base proof");
    let proof: crate::block_whir::BlockWhirProof =
        rkyv::from_bytes::<_, rkyv::rancor::Error>(&bytes).expect("a base proof");
    let opts = super::super::proof::block_base_options();
    let format = p1_format(memory, LeafScale::DoubledOneTop);
    let plan = WhirBlockPlan::derive(&elf, &opts, &format, proof.statement(), None)
        .expect("the plan derives");
    let (groups, leaves) = census_line(&format!("{path} {memory:?}"), &plan);
    let bound = plan.leaf_bound().expect("bounded");
    for k in 0..leaves {
        let rows =
            super::super::whir_chain::ChipRows::of(&plan.leaf_program(k).expect("the leaf emits"));
        println!("WORDS CENSUS LEAF {k}: {rows:?}");
        assert!(rows.under(bound.heights), "leaf {k}: {rows:?}");
    }
    println!(
        "WORDS CENSUS LEVELS: {:?} · {groups} groups",
        plan.levels()
            .into_iter()
            .map(|l| l.arities)
            .collect::<Vec<_>>()
    );
}

/// ★ One statement's bytes under the three memory formats are three:
/// the split format absorbs no format word, the fused one `MEMFUSED` and its
/// two counts, the word one `MEMWORDS` and its seven, so the three give other
/// challenges and other block identities (the id is a digest of the bytes).
/// The bytes are the frame's own function, as the plan calls it.
#[test]
fn the_three_memory_formats_give_three_statements() {
    let elf: &[u8] = include_bytes!("testdata/poc_rodata_commit.elf");
    let opts = super::super::proof::block_base_options();
    let words = p1_format(MemoryFormat::Words, LeafScale::DoubledOneTop);
    let owned = mix_statement(elf, &opts, &words, SMALL_WORDS, 2, 1);
    let plan = small_words_plan(&words).expect("the word plan derives");
    let tag = crate::block_whir::block_statement_tag(&words.zf.base);
    let digest = crate::statement::elf_digest(elf);
    let bytes = |memory: MemoryFormat| {
        crate::block_whir::block_statement_bytes(
            owned.view(),
            &tag,
            &digest,
            &plan.frame.config,
            memory,
        )
    };
    let all = [
        bytes(MemoryFormat::Split),
        bytes(MemoryFormat::Fused),
        bytes(MemoryFormat::Words),
    ];
    assert_ne!(all[0], all[1]);
    assert_ne!(all[0], all[2]);
    assert_ne!(all[1], all[2]);
    assert_eq!(
        crate::statement::elf_digest(&all[2]),
        *plan.id(),
        "the word plan's id is the digest of the word statement's bytes"
    );
    // The seven counts are in the bytes: each one moved moves them.
    for family in 0..7 {
        let mut moved = owned.clone();
        moved.word_memory[family] += 1;
        assert_ne!(
            crate::block_whir::block_statement_bytes(
                moved.view(),
                &tag,
                &digest,
                &plan.frame.config,
                MemoryFormat::Words,
            ),
            all[2],
            "word count {family}"
        );
    }
}

// =============================================================================
// The leaves over real word-format base proofs (box tier)
// =============================================================================

use crate::block_whir::{BlockWhirProof, Deviations, verify_block_whir};
use crate::tables::MaxRowsConfig;
use crate::test_utils::asm_elf_bytes;

use super::super::word::{LfmWord, word_as_ext};
use super::{LeafChecks, block_leaf_arena, id_words, leaf_program_with, out_halves};

/// The word format the box-tier tests prove and plan under: the Poseidon1
/// base (the tree's), the default leaf scale.
fn words_format() -> BlockFormat {
    p1_format(MemoryFormat::Words, LeafScale::DoubledOneTop)
}

fn test_opts() -> ProofOptions {
    ProofOptions::default_test_options()
}

/// `elf` given `private` input, proved as a streamed block under `format`
/// and verified by the host.
fn prove_words(elf: &[u8], private: &[u8], format: &BlockFormat) -> BlockWhirProof {
    let options = crate::tests::block_whir_tests::streamed(MaxRowsConfig::small(), 3, 9);
    let (proof, _) = crate::block_whir::prove_block_whir_with(
        elf,
        private,
        &test_opts(),
        format,
        &options,
        &Deviations::default(),
    )
    .expect("the word block proves");
    assert!(
        verify_block_whir(&proof, elf, &test_opts(), format).expect("verify runs"),
        "the host verifier accepts the word block"
    );
    proof
}

fn plan_over(elf: &[u8], proof: &BlockWhirProof, format: &BlockFormat) -> WhirBlockPlan {
    WhirBlockPlan::derive(elf, &test_opts(), format, proof.statement(), None)
        .expect("the plan derives")
}

/// Leaf `k` executed over the proof's arena: its published words, or why
/// not.
fn run_leaf(
    plan: &WhirBlockPlan,
    proof: &BlockWhirProof,
    k: usize,
    checks: LeafChecks,
) -> Result<Vec<LfmWord>, String> {
    let program = leaf_program_with(plan, k, checks)?;
    let arenas = block_leaf_arena(plan, proof, k)?;
    execute(
        &program,
        &arenas,
        &program.hasher(<crate::hash_pin::BlockFamily as crate::hash_pin::ProofFamily>::SOCKET),
    )
    .map(|e| e.public_words.iter().map(|(_, w)| *w).collect())
    .map_err(|e| format!("{e:?}"))
}

/// Every leaf over `proof`, or the first that does not execute.
fn run_leaves(
    plan: &WhirBlockPlan,
    proof: &BlockWhirProof,
    checks: LeafChecks,
) -> Result<Vec<Vec<LfmWord>>, String> {
    (0..plan.partition().num_leaves())
        .map(|k| run_leaf(plan, proof, k, checks).map_err(|e| format!("leaf {k}: {e}")))
        .collect()
}

/// The leaves' bus shares, summed: zero exactly when the bus closes.
fn share_sum(plan: &WhirBlockPlan, words: &[Vec<LfmWord>]) -> FEE {
    let layout = plan.child_layout();
    words
        .iter()
        .map(|w| word_as_ext(&w[layout.sum()]).expect("an extension word"))
        .fold(FEE::zero(), |a, b| a + b)
}

/// What every leaf publishes is the block's: the plan's id, one state, the
/// public output; and the shares cancel.
fn assert_leaves_claim_the_block(
    name: &str,
    plan: &WhirBlockPlan,
    proof: &BlockWhirProof,
    words: &[Vec<LfmWord>],
) {
    let layout = plan.child_layout();
    let id = id_words(plan.id());
    let out: Vec<LfmWord> = out_halves(&proof.public_output)
        .into_iter()
        .map(super::super::word::base_word)
        .collect();
    for w in words {
        assert_eq!(w.len(), layout.total(), "{name}");
        assert_eq!([w[layout.id(0)], w[layout.id(1)]], id, "{name}: the id");
        assert_eq!(
            w[layout.state(0)],
            words[0][layout.state(0)],
            "{name}: one state"
        );
        for (i, half) in out.iter().enumerate() {
            assert_eq!(w[layout.out_half(i)], *half, "{name}: the output");
        }
    }
    assert_eq!(
        share_sum(plan, words),
        FEE::zero(),
        "{name}: the shares cancel"
    );
}

/// ★ Every leaf of a word-format block executes over its base proof and the
/// leaves close the bus: a chip-free run (every CPU word family), the same
/// with private input (a private-input PAGE_W page), runs that call COPY
/// (COPY_W and MEMW_W), keccak256 (KECCAK_W) and ECREC (ECREC_W), and a run
/// with no load or store. Each plan's pages are PAGE_W tables.
#[test]
#[ignore = "proves small word-format blocks and executes their leaves; box tier"]
fn word_block_leaves_execute_and_close_the_bus() {
    let format = words_format();
    let private: Vec<u8> = (0..300u32).map(|i| (i * 7 + 3) as u8).collect();
    let runs: [(&str, &str, &[u8]); 6] = [
        ("chip-free", "fused_mem_cases", &[]),
        ("private input", "fused_mem_cases", &private),
        ("COPY", "test_copy_ecalls", &[]),
        ("keccak256", "test_keccak256", &[]),
        ("ECREC", "test_ecrec", &[]),
        ("no memory", "all_instructions_64", &[]),
    ];
    for (name, program, input) in runs {
        let elf = asm_elf_bytes(program);
        let proof = prove_words(&elf, input, &format);
        let plan = plan_over(&elf, &proof, &format);
        assert!(
            plan.frame
                .airs
                .air_refs()
                .iter()
                .filter(|a| a.name().starts_with(crate::tables::page_w::NAME_PREFIX))
                .count()
                == plan.frame.page_configs.len(),
            "{name}: every page a PAGE_W table"
        );
        assert_eq!(
            plan.frame.page_configs.iter().any(|c| c.is_private_input),
            !input.is_empty(),
            "{name}"
        );
        let words = run_leaves(&plan, &proof, LeafChecks::ALL)
            .unwrap_or_else(|e| panic!("{name}: an honest leaf executes: {e}"));
        assert_leaves_claim_the_block(name, &plan, &proof, &words);
        println!(
            "WORD BLOCK LEAVES {name}: {} groups · {} leaves · word tables {:?} · {} prepared",
            plan.num_groups(),
            plan.partition().num_leaves(),
            proof.word_memory,
            plan.prepared().len()
        );
    }
}

/// `elf`'s run given `private`, built by the windowed builder under `format`
/// (windows of 1000 cycles, the small chunks), REGISTER's files the
/// verifier's: the traces a forger edits.
fn word_build(
    elf: &[u8],
    format: &BlockFormat,
    private: &[u8],
) -> crate::tables::trace_builder::Traces {
    use crate::tables::trace_builder::WindowedTraceBuilder;
    use crate::tables::trace_builder::in_row_forgery_tests::with_verifier_files;
    let program = executor::elf::Elf::load(elf).expect("the ELF loads");
    let logs = executor::vm::execution::Executor::new(&program, private.to_vec())
        .expect("the executor")
        .run()
        .expect("the run")
        .logs;
    let mut builder = WindowedTraceBuilder::new(&program, private, &MaxRowsConfig::small())
        .and_then(|b| b.register_format(format.registers))
        .and_then(|b| b.memw_aligned_order(format.memw_aligned))
        .map(|b| b.keccak_rnd_form(format.keccak_rnd))
        .and_then(|b| b.memory_format(format.memory))
        .expect("the builder");
    let window = 1000;
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
    with_verifier_files(&program, &mut traces);
    traces
}

/// `traces` proved by a prover that skips its own check of the preprocessed
/// columns: a forger's, so the verifier answers for them.
fn prove_forged(
    elf: &[u8],
    traces: &mut crate::tables::trace_builder::Traces,
    format: &BlockFormat,
) -> BlockWhirProof {
    let program = executor::elf::Elf::load(elf).expect("the ELF loads");
    let options = crate::tests::block_whir_tests::streamed(MaxRowsConfig::small(), 3, 9);
    let mut stamps = crate::block_whir::BlockStamps::default();
    crate::block_whir::prove_traces(
        &program,
        elf,
        traces,
        &test_opts(),
        format,
        &options,
        &Deviations {
            unchecked_preprocessed: true,
            ..Deviations::default()
        },
        true,
        &|_, _| {},
        &mut stamps,
    )
    .expect("the forger proves")
}

/// The last row of PAGE_W table `page` no access touched (final timestamp 0).
fn untouched(traces: &crate::tables::trace_builder::Traces, page: usize) -> usize {
    use crate::tables::page_w::cols as pw;
    let t = &traces.pages[page];
    (1..t.num_rows() - 1)
        .rev()
        .find(|&r| t.get_main(r, pw::TIMESTAMP).canonical_u64() == 0)
        .expect("an untouched word")
}

/// One edit of a PAGE_W row no access touched, whose first and last tokens
/// are one token the row makes and takes, so every bus still balances:
/// `"offset"` one more (the next word's key), `"init"` its initial word one
/// more (and its final with it), `"half"` its initial low half `2^16` more
/// (a private half past 16 bits; its final with it).
fn edit_page(
    traces: &mut crate::tables::trace_builder::Traces,
    page: usize,
    row: usize,
    edit: &str,
) {
    use crate::tables::page_w::cols as pw;
    let t = &mut traces.pages[page];
    let mut bump = |col: usize, by: u64| {
        let v = *t.get_main(row, col);
        t.main_table.set(row, col, v + FE::from(by));
    };
    match edit {
        "offset" => bump(pw::OFFSET, 1),
        "init" => {
            bump(pw::INIT[0], 1);
            bump(pw::FINI_LO, 1);
        }
        _ => {
            bump(pw::INIT[0], 1 << 16);
            bump(pw::FINI_LO, 1 << 16);
        }
    }
}

/// The edited block's proof, refused by the host verifier.
fn forged_page(
    elf: &[u8],
    format: &BlockFormat,
    private: &[u8],
    page: usize,
    edit: &str,
) -> BlockWhirProof {
    let mut forged = word_build(elf, format, private);
    let row = untouched(&forged, page);
    edit_page(&mut forged, page, row, edit);
    let proof = prove_forged(elf, &mut forged, format);
    assert!(
        !matches!(
            verify_block_whir(&proof, elf, &test_opts(), format),
            Ok(true)
        ),
        "page {page} {edit}: the host verifier refuses it"
    );
    proof
}

/// Runs `f` with each of `faults` on.
fn under<R>(faults: &[Fault], f: impl FnOnce() -> R) -> R {
    match faults.split_first() {
        None => f(),
        Some((&first, rest)) => route_faults::with(first, || under(rest, f)),
    }
}

/// ★ On real proofs: a word block whose zero page or ELF page has one
/// untouched row's OFFSET moved (a key minted twice) or initial word changed,
/// or whose private-input page has one untouched row's OFFSET moved, proved
/// by a forger (the host verifier refuses each): a leaf refuses it. The
/// controls, each taking out one check:
/// - the page leg off (the route checks nothing, the count check off): every
///   leaf executes and the shares cancel, so the page leg is what refuses;
/// - an ELF or zero page on a private page's route (OFFSET alone, the count
///   check off): an initial word changed executes, so the count check is
///   what keeps the halves checked.
///
/// A private row's initial half past 16 bits is the bus's: every leaf
/// executes and the shares do not cancel.
#[test]
#[ignore = "proves forged word-format blocks and executes their leaves; box tier"]
fn a_word_block_leaf_refuses_forged_page_columns() {
    let format = words_format();
    let elf = asm_elf_bytes("fused_mem_cases");
    let honest = word_build(&elf, &format, &[]);
    let zero = honest
        .page_configs
        .iter()
        .position(|c| !c.is_private_input && c.init_values.is_none())
        .expect("a zero page");
    let elf_page = honest
        .page_configs
        .iter()
        .position(|c| !c.is_private_input && c.init_values.is_some())
        .expect("an ELF page");
    let mut wrong = Vec::new();
    for (page, kind) in [(zero, "zero"), (elf_page, "ELF")] {
        for edit in ["offset", "init"] {
            let proof = forged_page(&elf, &format, &[], page, edit);
            let plan = plan_over(&elf, &proof, &format);
            let all = run_leaves(&plan, &proof, LeafChecks::ALL);
            let off = under(&[Fault::NoPageLeg, Fault::NoCountCheck], || {
                run_leaves(&plan, &proof, LeafChecks::ALL)
            });
            let closes = off
                .as_ref()
                .is_ok_and(|w| share_sum(&plan, w) == FEE::zero());
            if all.is_ok() || !closes {
                wrong.push(format!(
                    "{kind} {edit}: every check {:?} · page leg off {:?}",
                    all.err(),
                    off.map(|w| share_sum(&plan, &w))
                ));
            }
            if edit == "init" {
                let swapped = under(&[Fault::PrivateRouteSwapped, Fault::NoCountCheck], || {
                    run_leaves(&plan, &proof, LeafChecks::ALL)
                });
                if swapped.is_err() {
                    wrong.push(format!(
                        "{kind} init on the private route: {:?}",
                        swapped.err()
                    ));
                }
            }
        }
    }
    let private: Vec<u8> = (0..300u32).map(|i| (i * 7 + 3) as u8).collect();
    let page = word_build(&elf, &format, &private)
        .page_configs
        .iter()
        .position(|c| c.is_private_input)
        .expect("a private page");
    let proof = forged_page(&elf, &format, &private, page, "offset");
    let plan = plan_over(&elf, &proof, &format);
    if run_leaves(&plan, &proof, LeafChecks::ALL).is_ok() {
        wrong.push("private offset: every leaf executes".to_string());
    }
    let off = under(&[Fault::NoPageLeg, Fault::NoCountCheck], || {
        run_leaves(&plan, &proof, LeafChecks::ALL)
    });
    if !off
        .as_ref()
        .is_ok_and(|w| share_sum(&plan, w) == FEE::zero())
    {
        wrong.push(format!("private offset, page leg off: {:?}", off.err()));
    }
    let proof = forged_page(&elf, &format, &private, page, "half");
    let plan = plan_over(&elf, &proof, &format);
    match run_leaves(&plan, &proof, LeafChecks::ALL) {
        Ok(w) if share_sum(&plan, &w) != FEE::zero() => {}
        other => wrong.push(format!(
            "private half: every leaf executes, the shares do not cancel: {:?}",
            other.map(|w| share_sum(&plan, &w))
        )),
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

/// ★ On real proofs: with every page that has initial data on the
/// prepared opening on both sides (on this thread), a word block's plan
/// settles a PAGE_W table's five columns by the opening and every leaf
/// executes; an ELF page's initial word changed on an untouched row is
/// refused by a leaf, and executes with the prepared openings left out (the
/// control), so the opening is what refuses it.
#[test]
#[ignore = "proves word-format blocks over the prepared opening; box tier"]
fn a_word_block_leaf_holds_its_prepared_pages() {
    let format = words_format();
    let elf = asm_elf_bytes("fused_mem_cases");
    crate::block_whir::DENSE_PAGES_FORCED.with(|c| c.set(true));
    let checked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut honest = word_build(&elf, &format, &[]);
        let proof = prove_forged(&elf, &mut honest, &format);
        assert!(matches!(
            verify_block_whir(&proof, &elf, &test_opts(), &format),
            Ok(true)
        ));
        let plan = plan_over(&elf, &proof, &format);
        let settled: Vec<usize> = plan
            .prepared()
            .iter()
            .flat_map(|p| p.tables.iter().map(|&(_, _, n)| n))
            .collect();
        assert!(
            settled.contains(&crate::tables::page_w::NUM_PREPROCESSED_COLS),
            "{settled:?}"
        );
        let words = run_leaves(&plan, &proof, LeafChecks::ALL).expect("the honest leaves");
        assert_leaves_claim_the_block("prepared pages", &plan, &proof, &words);
        let page = honest
            .page_configs
            .iter()
            .position(|c| !c.is_private_input && c.init_values.is_some())
            .expect("an ELF page");
        let proof = forged_page(&elf, &format, &[], page, "init");
        let plan = plan_over(&elf, &proof, &format);
        assert!(
            run_leaves(&plan, &proof, LeafChecks::ALL).is_err(),
            "the opening refuses the edited page"
        );
        let skip = LeafChecks {
            prepared: false,
            ..LeafChecks::ALL
        };
        assert!(
            run_leaves(&plan, &proof, skip).is_ok(),
            "with the openings left out it executes"
        );
    }));
    crate::block_whir::DENSE_PAGES_FORCED.with(|c| c.set(false));
    if let Err(panic) = checked {
        std::panic::resume_unwind(panic);
    }
}

/// ★ A word block's leaf front draws the host's challenges: the leaf
/// replays the statement the host verifier absorbs (the word format's word
/// and its seven counts after the table counts) and draws the host's `z`,
/// `α`, `β`, the fork's first draw and a group's point.
#[test]
#[ignore = "executes a word block's leaf front; box tier"]
fn the_word_block_front_draws_the_hosts_challenges() {
    let format = words_format();
    let elf = asm_elf_bytes("fused_mem_cases");
    let proof = prove_words(&elf, &[], &format);
    assert!(proof.word_memory.iter().any(|&n| n > 0));
    super::super::whir_block_tests::front_draws_the_hosts_challenges(&elf, &proof, &format);
}

/// ★ A word count moved from one family to another (the total kept, so
/// the statement still states one height a table) is refused: its plan
/// refuses, or derives another block's identity and no leaf of it executes
/// over the proof.
#[test]
#[ignore = "proves a word-format block and executes its leaves; box tier"]
fn a_moved_word_count_is_another_block() {
    let format = words_format();
    let elf = asm_elf_bytes("fused_mem_cases");
    let proof = prove_words(&elf, &[], &format);
    let honest = plan_over(&elf, &proof, &format);
    let (from, to) = (0usize, 1usize);
    assert!(proof.word_memory[from] > 0, "{:?}", proof.word_memory);
    let mut moved = proof.clone();
    moved.word_memory[from] -= 1;
    moved.word_memory[to] += 1;
    match WhirBlockPlan::derive(&elf, &test_opts(), &format, moved.statement(), None) {
        Err(e) => println!("MOVED WORD COUNT: the plan refuses: {e}"),
        Ok(plan) => {
            assert_ne!(plan.id(), honest.id(), "another statement, another id");
            assert!(
                (0..plan.partition().num_leaves()).all(|k| run_leaf(
                    &plan,
                    &moved,
                    k,
                    LeafChecks::ALL
                )
                .is_err()),
                "no leaf of the moved statement executes"
            );
        }
    }
}
