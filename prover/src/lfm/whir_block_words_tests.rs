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
use super::{
    DECODE_NAME, DynAir, KECCAK_RC_NAME, LeafScale, REGISTER_NAME, Route, Shapes, WhirBlockPlan,
};
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
        // PAGE_W's preprocessed columns are its leading committed columns: the
        // prefix a prepared opening settles is the columns this leg checks.
        assert_eq!(
            targets,
            (0..declared).collect::<Vec<_>>(),
            "{kind}: PAGE_W's preprocessed columns lead the table"
        );
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
    // A fixed table's folded columns and a prepared table's opened ones are
    // held to the stated height too: a REGISTER-like column of 32 rows at a
    // stated 2^2, an opening of 2^7-row columns at a stated 2^2, each refused
    // by the height check, and passed by the route checks without it.
    let register = Route::ConstMle(vec![vec![FE::zero(); 32]]);
    let refused = check_route_columns(&register, 1, 2, "REGISTER", 3).expect_err("32 rows at 2^2");
    assert!(
        refused.contains("states 2^2 rows and its route's columns are 32 rows"),
        "{refused}"
    );
    let decode = Route::Prepared {
        settled: 3,
        num_vars: 7,
    };
    let refused = check_route_columns(&decode, 3, 2, "DECODE", 1).expect_err("2^7 at 2^2");
    assert!(
        refused.contains("states 2^2 rows and its prepared columns are 2^7 rows"),
        "{refused}"
    );
    for route in [&register, &decode] {
        let declared = route.columns_checked();
        assert!(
            route_faults::with(Fault::NoHeightCheck, || {
                check_route_columns(route, declared, 2, "fixed", 0)
            })
            .is_ok()
        );
    }
    let five = Route::Prepared {
        settled: 4,
        num_vars,
    };
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
        format.ecdas,
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
    if !rows.is_multiple_of(cap) {
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
        rkyv::from_bytes::<_, rkyv::rancor::Error>(&bytes).expect(
            "a base proof written by this build (a base proof file has no version: one written by another \
             build reads as some other shape)",
        );
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

/// The box-tier blocks' options: the small chunks, streamed, KECCAK_RND cut at
/// the block's own height ([`crate::block_whir::BLOCK_KECCAK_RND_ROWS_LOG2`]).
/// Cut at 2^3 rows, a keccak256 run's rounds are some eighty tables, which the
/// production stack puts in one group whose leaf rows no leaf holds
/// ([`a_groups_heavy_tables_are_bounded_by_one_leaf`]).
fn word_block_options() -> crate::block_whir::BlockOptions {
    crate::tests::block_whir_tests::streamed(
        MaxRowsConfig::small(),
        crate::block_whir::BLOCK_KECCAK_RND_ROWS_LOG2,
        9,
    )
}

/// `elf` given `private` input, proved as a streamed block under `format`
/// and verified by the host.
fn prove_words(elf: &[u8], private: &[u8], format: &BlockFormat) -> BlockWhirProof {
    let options = word_block_options();
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
    // Runs `test_ecrec`: the ECREC forgery switch stays off meanwhile.
    let _ecrec = crate::tables::trace_builder::ecrec_forge_free();
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
    let options = word_block_options();
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

// =============================================================================
// The tree's verifiers under a named memory format
// =============================================================================

use super::super::whir_block_tree::{WhirBlockTreeProof, verify_whir_block_tree_proof_at};

/// A top with no proof in it: every check here stops before the top is read
/// (the frame's refusals), so what reaches it is a statement the frame took.
fn empty_top() -> super::super::proof::LfmProof {
    super::super::proof::LfmProof {
        proof: super::super::proof::LfmStarkProof {
            multi: stark::proof::stark::MultiProof { proofs: vec![] },
            commit_ood: vec![],
        },
        public_words: vec![],
    }
}

/// The small word block's statement, packed under the verifier's pinned
/// format with `memory`'s loads and stores (so its groups are the ones that
/// verifier stacks).
fn pinned_statement(memory: MemoryFormat, mix: Mix<'_>) -> OwnedBlockStatement {
    let elf: &[u8] = include_bytes!("testdata/poc_rodata_commit.elf");
    let format = BlockFormat::pinned().with_memory(memory);
    let opts = crate::lfm::proof::block_base_options_under(&format.zf);
    mix_statement(elf, &opts, &format, mix, 2, 1)
}

/// The file verifier at every pinned named choice and `memory`.
fn verify_file_under(file: &WhirBlockTreeProof, memory: MemoryFormat) -> Result<(), String> {
    let elf: &[u8] = include_bytes!("testdata/poc_rodata_commit.elf");
    let pinned = BlockFormat::pinned();
    verify_whir_block_tree_proof_at(
        elf,
        file,
        pinned.group,
        crate::zf_format::WhirBlowup::Two,
        pinned.zf.whir_leaves,
        pinned.zf.whir_trees,
        pinned.leaf_scale,
        pinned.leaf_query,
        multilinear::whir_chain::GkrPack::PRODUCTION,
        memory,
    )
}

/// The verifier's plan of `owned` under the pinned named choices and
/// `memory`: what the file verifier derives before it reads the top.
fn verifier_plan_under(
    owned: &OwnedBlockStatement,
    memory: MemoryFormat,
) -> Result<WhirBlockPlan, String> {
    let elf: &[u8] = include_bytes!("testdata/poc_rodata_commit.elf");
    let pinned = BlockFormat::pinned();
    let presets = super::verifier_presets_at(
        pinned.group,
        crate::zf_format::WhirBlowup::Two,
        pinned.zf.whir_leaves,
        pinned.zf.whir_trees,
        pinned.leaf_scale,
        pinned.leaf_query,
        multilinear::whir_chain::GkrPack::PRODUCTION,
        memory,
    )?;
    super::verifier_plan(elf, owned.view(), &presets)
}

/// The small split block: SMALL_WORDS' tables with the split format's memory
/// tables in place of the word families.
const SMALL_SPLIT: Mix<'static> = &[
    ("CPU", &[(1 << 12, 2)]),
    ("LT", &[(1 << 10, 1)]),
    ("BRANCH", &[(1 << 9, 1)]),
    ("MEMW_R", &[(1 << 10, 1)]),
    ("COPY", &[(1 << 8, 1)]),
    ("LOAD", &[(1 << 11, 1)]),
    ("STORE", &[(1 << 10, 1)]),
    ("MEMW_A", &[(1 << 11, 1)]),
    ("MEMW", &[(1 << 9, 1)]),
];

/// ★ At the tree's verifier: the memory format is the caller's named
/// choice and the file's counts are read under it. A word block's file is
/// refused under the split and the fused format by their rule (they state no
/// word table), and its verifier plan under the word format is the prover's
/// own plan of it (the same block identity, so a top the prover's tree claims
/// is the one this verifier asks for); a split block's file under the word
/// format is refused by the retired-table rule, and under the split format
/// its plan is the pinned verifier's (`pinned_verifier_presets`), as before
/// the named choice. Each refusal's control: its rule taken out, the file
/// passes the frame or is refused by another rule.
#[test]
fn the_tree_verifier_reads_a_file_under_its_named_memory_format() {
    let words = pinned_statement(MemoryFormat::Words, SMALL_WORDS);
    let file = WhirBlockTreeProof::new(words.clone(), empty_top());
    assert_eq!(file.word_memory, words.word_memory);
    for other in [MemoryFormat::Split, MemoryFormat::Fused] {
        let refused = verify_file_under(&file, other).expect_err("refused");
        assert!(
            refused.contains(&format!("the {other:?} memory format states no word table")),
            "{other:?}: {refused}"
        );
        crate::block_whir::WORD_COUNTS_UNCHECKED.with(|c| c.set(true));
        let control = verify_file_under(&file, other);
        crate::block_whir::WORD_COUNTS_UNCHECKED.with(|c| c.set(false));
        let control = control.expect_err("refused by another rule");
        assert!(
            !control.contains("states no word table"),
            "{other:?} control: {control}"
        );
    }
    let elf: &[u8] = include_bytes!("testdata/poc_rodata_commit.elf");
    let verifier = verifier_plan_under(&words, MemoryFormat::Words).expect("the word plan");
    let format = BlockFormat::pinned().with_memory(MemoryFormat::Words);
    let prover = WhirBlockPlan::derive(
        elf,
        &crate::lfm::proof::block_base_options_under(&format.zf),
        &format,
        words.view(),
        None,
    )
    .expect("the prover's plan");
    assert_eq!(
        verifier.id(),
        prover.id(),
        "the verifier asks for the prover's block"
    );

    let split = pinned_statement(MemoryFormat::Split, SMALL_SPLIT);
    let split_file = WhirBlockTreeProof::new(split.clone(), empty_top());
    assert_eq!(
        (split_file.fused_memory, split_file.word_memory),
        ([0, 0], [0; 7])
    );
    let refused = verify_file_under(&split_file, MemoryFormat::Words).expect_err("refused");
    assert!(refused.contains("states no LOAD table"), "{refused}");
    crate::block_whir::RETIRED_COUNTS_UNCHECKED.with(|c| c.set(true));
    let control = verify_file_under(&split_file, MemoryFormat::Words);
    crate::block_whir::RETIRED_COUNTS_UNCHECKED.with(|c| c.set(false));
    assert!(
        !control
            .expect_err("refused by another rule")
            .contains("states no LOAD table"),
        "the retired rule's control"
    );
    let named = verifier_plan_under(&split, MemoryFormat::Split).expect("the split plan");
    let pinned = super::verifier_plan(
        elf,
        split.view(),
        &super::pinned_verifier_presets(BlockFormat::pinned().group),
    )
    .expect("the pinned plan");
    assert_eq!(
        named.id(),
        pinned.id(),
        "the named split verifier is the pinned one"
    );
    assert_ne!(named.id(), verifier.id());
}

/// ★ At the tree's verifier, under the word format: each of the
/// frame's word rules refuses a file that breaks it, through the file
/// verifier (not only the host block verifier) — a retired table stated, a
/// chip's table the format takes no call of yet, a PAGE_W table at another
/// height, a word table past 2^21 rows — and a word count moved from one
/// family to another (the total kept) derives another block's identity.
/// The two count rules' controls: the rule taken out, another refuses.
#[test]
fn every_word_rule_of_the_frame_refuses_the_tree_verifiers_file() {
    let honest = pinned_statement(MemoryFormat::Words, SMALL_WORDS);
    let honest_id = *verifier_plan_under(&honest, MemoryFormat::Words)
        .expect("the honest plan")
        .id();
    let refused = |owned: &OwnedBlockStatement| {
        verify_file_under(
            &WhirBlockTreeProof::new(owned.clone(), empty_top()),
            MemoryFormat::Words,
        )
        .expect_err("refused")
    };
    let mut retired = honest.clone();
    retired.table_counts.load = 1;
    let e = refused(&retired);
    assert!(e.contains("states no LOAD table"), "{e}");
    crate::block_whir::RETIRED_COUNTS_UNCHECKED.with(|c| c.set(true));
    let control = refused(&retired);
    crate::block_whir::RETIRED_COUNTS_UNCHECKED.with(|c| c.set(false));
    assert!(!control.contains("states no LOAD table"), "{control}");

    let mut chip = honest.clone();
    chip.table_counts.hint = 1;
    let e = refused(&chip);
    assert!(e.contains("takes no HINT call yet: no HINT table"), "{e}");
    crate::block_whir::CHIP_COUNTS_UNCHECKED.with(|c| c.set(true));
    let control = refused(&chip);
    crate::block_whir::CHIP_COUNTS_UNCHECKED.with(|c| c.set(false));
    assert!(!control.contains("takes no HINT call yet"), "{control}");

    let plan = verifier_plan_under(&honest, MemoryFormat::Words).expect("the honest plan");
    let names: Vec<String> = plan
        .frame
        .airs
        .air_refs()
        .iter()
        .map(|a| a.name().to_string())
        .collect();
    let page = names
        .iter()
        .position(|n| n.starts_with(crate::tables::page_w::NAME_PREFIX))
        .expect("a PAGE_W table");
    let mut tall = honest.clone();
    tall.table_num_vars[page] += 1;
    let e = refused(&tall);
    assert!(e.contains("a PAGE_W table is 2^15"), "{e}");
    let word = names
        .iter()
        .position(|n| n.starts_with("MEM_LOAD8["))
        .expect("a MEM_LOAD8 table");
    let mut long = honest.clone();
    long.table_num_vars[word] = 22;
    let e = refused(&long);
    assert!(e.contains("a block takes at most 2^21"), "{e}");

    let mut moved = honest.clone();
    assert!(moved.word_memory[0] > 1);
    moved.word_memory[0] -= 1;
    moved.word_memory[1] += 1;
    match verifier_plan_under(&moved, MemoryFormat::Words) {
        Err(e) => println!("MOVED WORD COUNT: refused: {e}"),
        Ok(plan) => assert_ne!(*plan.id(), honest_id, "a moved count is another block"),
    }
}

/// ★ A group's leaf room bounds the heavy tables it holds, under any memory
/// format: a KECCAK_RND table's leg costs a leaf about 47 thousand ext-ALU
/// rows whatever its height (its constraint graph), so a block whose round
/// chip is cut into tables of 2^3 rows puts some eighty of them in one group
/// under the production stack, and the planner refuses it by naming the
/// chip; forty still derive; the same rounds in one table of 2^16 rows (the
/// block's own cut) cost one table's rows. A completeness bound, never a
/// check a proof can pass: such a block is refused before any program is
/// emitted. Under the word and the split format alike.
#[test]
fn a_groups_heavy_tables_are_bounded_by_one_leaf() {
    let elf: &[u8] = include_bytes!("testdata/poc_rodata_commit.elf");
    for memory in [MemoryFormat::Words, MemoryFormat::Split] {
        let format = BlockFormat {
            memory,
            ..words_format()
        };
        let opts = super::super::proof::block_base_options_under(&format.zf);
        let derive = |rows: usize, n: usize| {
            let keccak: &[(usize, usize)] = &[(1 << 8, 1)];
            let rnd: &[(usize, usize)] = &[(rows, n)];
            let (base, family) = if memory == MemoryFormat::Words {
                (SMALL_WORDS, crate::tables::keccak_w::NAME)
            } else {
                (SMALL_SPLIT, "KECCAK")
            };
            let mut mix: Vec<(&str, &[(usize, usize)])> = base.to_vec();
            mix.push((family, keccak));
            mix.push(("KECCAK_RND", rnd));
            let owned = mix_statement(elf, &opts, &format, &mix, 2, 0);
            WhirBlockPlan::derive(elf, &opts, &format, owned.view(), None)
        };
        let xalu = |plan: &WhirBlockPlan| plan.chip_loads().0.iter().map(|l| l.xalu).max();
        let one = derive(8, 1).expect("one table");
        let ten = derive(8, 10).expect("ten tables");
        let per_table = (xalu(&ten).expect("a group") - xalu(&one).expect("a group")) / 9;
        assert!(
            (40_000..60_000).contains(&per_table),
            "{memory:?}: {per_table} ext-ALU rows a KECCAK_RND table"
        );
        assert_eq!(ten.num_groups(), 1, "{memory:?}: one group");
        derive(8, 40).expect("forty tables fit a leaf");
        let refused = derive(8, 80).err().expect("eighty tables pass a leaf");
        assert!(
            refused.contains("ext-ALU rows — a leaf takes at most"),
            "{memory:?}: {refused}"
        );
        let whole = derive(1 << crate::block_whir::BLOCK_KECCAK_RND_ROWS_LOG2, 1)
            .expect("one table at the block's cut");
        assert!(
            xalu(&whole) < xalu(&ten),
            "{memory:?}: one tall table costs one table's rows"
        );
    }
}

/// Instrument (`--exact`): one group's leaf rows as it gains zero pages, under
/// the word and the split format (the production stack holds every page of
/// these statements in one group).
#[test]
#[ignore = "laptop instrument: run alone with --exact"]
fn pages_in_one_group_census() {
    let elf: &[u8] = include_bytes!("testdata/poc_rodata_commit.elf");
    for memory in [MemoryFormat::Words, MemoryFormat::Split] {
        let format = BlockFormat {
            memory,
            ..words_format()
        };
        let opts = super::super::proof::block_base_options_under(&format.zf);
        let mix = if memory == MemoryFormat::Words {
            SMALL_WORDS
        } else {
            SMALL_SPLIT
        };
        for pages in [2usize, 50, 100, 200, 400] {
            let owned = mix_statement(elf, &opts, &format, mix, pages, 0);
            match WhirBlockPlan::derive(elf, &opts, &format, owned.view(), None) {
                Ok(plan) => {
                    let (loads, _) = plan.chip_loads();
                    println!(
                        "PAGES {memory:?} {pages}: {} groups · ext-ALU by group {:?} · socket by group {:?}",
                        plan.num_groups(),
                        loads.iter().map(|l| l.xalu).collect::<Vec<_>>(),
                        loads.iter().map(|l| l.hash).collect::<Vec<_>>(),
                    );
                }
                Err(e) => println!("PAGES {memory:?} {pages}: refused: {e}"),
            }
        }
    }
}

// =============================================================================
// The statement the leaves replay, fused at the tree, the tree prover's
// register guard, and the pinned leaf programs
// =============================================================================

/// ★ The leaves replay the WORD statement: the calls a word plan's leaves
/// absorb concatenate to the word statement's bytes, not the split one's, and
/// their digest is the plan's identity (taken from the bytes by another
/// call, so the identity tests alone would not see a leaf replaying another
/// statement).
#[test]
fn the_word_plans_statement_calls_are_the_word_statements() {
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
    let calls: Vec<u8> = plan.statement_calls.concat();
    assert_eq!(calls, bytes(MemoryFormat::Words), "the word statement");
    assert_ne!(calls, bytes(MemoryFormat::Split));
    assert_eq!(crate::statement::elf_digest(&calls), *plan.id());
}

/// A small fused block's statement at the verifier's pinned presets:
/// SMALL_SPLIT's tables and one MEM_LOAD and one MEM_STORE table (the AIR
/// set's last two), in a group of their own.
fn fused_statement() -> OwnedBlockStatement {
    let mut owned = pinned_statement(MemoryFormat::Fused, SMALL_SPLIT);
    let n = owned.table_num_vars.len() as u32;
    owned.table_num_vars.extend([9u8, 8]);
    owned.fused_memory = [1, 1];
    owned.groups.push(vec![n, n + 1]);
    owned
}

/// ★ Fused at the tree, and a split file's counts. A fused statement plans
/// at the verifier's named choice and every leaf emits (MEM_LOAD and
/// MEM_STORE among its tables); its file is refused under the split and the
/// word format by the fused rule. Each of a split file's nine counts set to 1
/// is refused under the split format by its rule; with the fused rule taken
/// out (the control), a fused pair on a split file is no longer refused by
/// it.
#[test]
fn fused_at_the_tree_and_a_split_files_counts() {
    let fused = fused_statement();
    let plan = verifier_plan_under(&fused, MemoryFormat::Fused).expect("the fused plan");
    let names: Vec<String> = plan
        .frame
        .airs
        .air_refs()
        .iter()
        .map(|a| a.name().to_string())
        .collect();
    assert!(names.iter().any(|n| n.starts_with("MEM_LOAD[")));
    assert!(names.iter().any(|n| n.starts_with("MEM_STORE[")));
    emit_leaves(&plan).expect("the fused leaves emit");
    let file = WhirBlockTreeProof::new(fused.clone(), empty_top());
    assert_eq!(file.fused_memory, [1, 1]);
    for other in [MemoryFormat::Split, MemoryFormat::Words] {
        let refused = verify_file_under(&file, other).expect_err("refused");
        assert!(
            refused.contains("states no fused table"),
            "{other:?}: {refused}"
        );
    }
    let split = pinned_statement(MemoryFormat::Split, SMALL_SPLIT);
    for slot in 0..9 {
        let mut moved = split.clone();
        if slot < 2 {
            moved.fused_memory[slot] = 1;
        } else {
            moved.word_memory[slot - 2] = 1;
        }
        let file = WhirBlockTreeProof::new(moved, empty_top());
        let refused = verify_file_under(&file, MemoryFormat::Split).expect_err("refused");
        let want = if slot < 2 {
            "states no fused table"
        } else {
            "states no word table"
        };
        assert!(refused.contains(want), "slot {slot}: {refused}");
        if slot < 2 {
            crate::block_whir::FUSED_COUNTS_UNCHECKED.with(|c| c.set(true));
            let control = verify_file_under(&file, MemoryFormat::Split);
            crate::block_whir::FUSED_COUNTS_UNCHECKED.with(|c| c.set(false));
            let control = control.expect_err("refused by another rule");
            assert!(
                !control.contains("states no fused table"),
                "slot {slot} control: {control}"
            );
        }
    }
}

/// ★ The tree's prover refuses the word memory format under the 32-bit
/// register tokens, before anything is proved, naming why; the word format
/// checks the registers in the CPU row.
#[test]
fn the_tree_prover_refuses_words_under_the_32_bit_register_tokens() {
    use super::super::whir_block_tree::{StderrSink, WhirTreeConfig, prove_whir_block_tree};
    let mut cfg = WhirTreeConfig::from_env().expect("the tree's configuration");
    cfg.format.memory = MemoryFormat::Words;
    cfg.format.registers = crate::tables::register::RegisterFormat::Tokens32;
    match prove_whir_block_tree(&[], &[], &cfg, &StderrSink) {
        Ok(_) => panic!("the tree proved words under the 32-bit register tokens"),
        Err(refused) => assert!(
            refused.contains("checks the registers in the CPU row"),
            "{refused}"
        ),
    }
}

/// ★ With a compression step planned, the tree's prover refuses a memory
/// format other than the pinned one before anything is proved, naming the
/// top's uncompressed file as the way to prove it: every compressed file's
/// verifier checks the pinned format. Controls: the pinned format passes the
/// check with compression planned, and every format passes it without.
#[test]
fn the_tree_prover_refuses_to_compress_another_memory_format() {
    use super::super::whir_block_tree::{
        StderrSink, WhirTreeConfig, compressed_memory_check, prove_whir_block_tree,
    };
    let pinned = BlockFormat::pinned().memory;
    assert_eq!(pinned, MemoryFormat::Split, "the pinned memory format");
    for memory in [MemoryFormat::Fused, MemoryFormat::Words] {
        let mut cfg = WhirTreeConfig::from_env().expect("the tree's configuration");
        cfg.format.memory = memory;
        cfg.format.registers = crate::tables::register::RegisterFormat::InRow;
        cfg.compress = Some(super::super::wlfm::WlfmRate::Two);
        match prove_whir_block_tree(&[], &[], &cfg, &StderrSink) {
            Ok(_) => panic!("the tree proved a compressed {memory:?} block"),
            Err(refused) => assert!(
                refused.contains(&format!(
                    "a compressed block is proved at the Split memory format alone, not {memory:?}"
                )) && refused.contains("uncompressed file"),
                "{memory:?}: {refused}"
            ),
        }
    }
    assert_eq!(compressed_memory_check(pinned, true), Ok(()));
    for memory in [
        MemoryFormat::Split,
        MemoryFormat::Fused,
        MemoryFormat::Words,
    ] {
        assert_eq!(compressed_memory_check(memory, false), Ok(()), "{memory:?}");
    }
}

/// The leaf digests of `owned` under `format` (each leaf, in order) and the
/// plan's identity's first eight bytes, with the prepared opening forced onto
/// every ELF page with data when `dense`.
fn leaf_digests(
    format: &BlockFormat,
    owned: &OwnedBlockStatement,
    dense: bool,
) -> (Vec<String>, [u8; 8]) {
    let elf: &[u8] = include_bytes!("testdata/poc_rodata_commit.elf");
    let opts = super::super::proof::block_base_options_under(&format.zf);
    crate::block_whir::DENSE_PAGES_FORCED.with(|c| c.set(dense));
    let derived = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let plan = WhirBlockPlan::derive(elf, &opts, format, owned.view(), None)
            .expect("the plan derives");
        let digests = (0..plan.partition().num_leaves())
            .map(|k| {
                super::super::whir_chain_tests::program_digest(
                    &plan.leaf_program(k).expect("the leaf emits"),
                )
            })
            .collect();
        let id: [u8; 8] = plan.id()[..8].try_into().expect("eight bytes");
        (digests, id)
    }));
    crate::block_whir::DENSE_PAGES_FORCED.with(|c| c.set(false));
    derived.unwrap_or_else(|panic| std::panic::resume_unwind(panic))
}

/// ★ The split leaf programs with private-input pages, pinned: the small
/// split block with one private-input page, its ELF pages on the sparse leg
/// and on the prepared opening, at both leaf scales. The digests are the
/// ones the same statements give at the word-tokens head this work started
/// from (built there and here, equal), so the split format's leaves are
/// unchanged on every page kind. Not blind: with a private page on an ELF
/// page's route (its count check out), the leaf moves.
#[test]
fn the_split_leaves_with_a_private_page_are_pinned() {
    const ID: [u8; 8] = [0x8d, 0x0b, 0x67, 0xfc, 0x2c, 0x1a, 0x33, 0x65];
    const SPARSE: &str = "179d541abd410363f71e85372d945298d3bfef3ffce665f81efc79a9d0d6a6b4";
    const PREPARED: &str = "d394a340579e07ff2085997f5b8ad09626e051726aff0930b7981227bbb4e4fa";
    let elf: &[u8] = include_bytes!("testdata/poc_rodata_commit.elf");
    let opts = super::super::proof::block_base_options();
    for scale in [LeafScale::DoubledOneTop, LeafScale::Fixed] {
        let format = p1_format(MemoryFormat::Split, scale);
        let owned = mix_statement(elf, &opts, &format, SMALL_SPLIT, 2, 1);
        for (dense, want) in [(false, SPARSE), (true, PREPARED)] {
            let (digests, id) = leaf_digests(&format, &owned, dense);
            assert_eq!(id, ID, "{scale}: the plan's identity");
            assert_eq!(digests, vec![want.to_string()], "{scale}, dense {dense}");
        }
        let moved = route_faults::with(Fault::PrivateRouteSwapped, || {
            route_faults::with(Fault::NoCountCheck, || {
                route_faults::with(Fault::NoHeightCheck, || {
                    leaf_digests(&format, &owned, false)
                })
            })
        });
        assert_ne!(
            moved.0,
            vec![SPARSE.to_string()],
            "{scale}: the pin sees a page's route"
        );
    }
}

/// ★ The word-format and fused leaf programs of the small plans, pinned: the
/// small word block (every word family, a private-input page) and the small
/// fused block, each plan's identity and every leaf's digest. A change to a
/// word leaf (a route, a leg, a word table) moves them. Not blind: with the
/// page leg taken out, the word leaf moves.
#[test]
fn the_word_and_fused_leaves_are_pinned() {
    const WORD_ID: [u8; 8] = [0xe7, 0xf5, 0x5a, 0xc7, 0x00, 0x9a, 0x6e, 0x04];
    const WORD_LEAVES: &[&str] =
        &["c2a6da70844ff4ae56241bdc4c7e339102d35d79bd6eefdf76491af522ca85fe"];
    const FUSED_ID: [u8; 8] = [0x3e, 0x9b, 0xfb, 0x43, 0xf6, 0xcd, 0x9a, 0x49];
    const FUSED_LEAVES: &[&str] =
        &["1fe8687986e96b6e132dbdea0ed95f5d6f2dca9a6974f2f1fb663bcf23f8e93f"];
    let elf: &[u8] = include_bytes!("testdata/poc_rodata_commit.elf");
    let words = p1_format(MemoryFormat::Words, LeafScale::DoubledOneTop);
    let opts = super::super::proof::block_base_options_under(&words.zf);
    let owned = mix_statement(elf, &opts, &words, SMALL_WORDS, 2, 1);
    let (digests, id) = leaf_digests(&words, &owned, false);
    let fused_format = BlockFormat::pinned().with_memory(MemoryFormat::Fused);
    let (fused_digests, fused_id) = leaf_digests(&fused_format, &fused_statement(), false);
    assert_eq!(id, WORD_ID, "the word plan's identity");
    assert_eq!(digests, WORD_LEAVES, "the word leaves");
    assert_eq!(fused_id, FUSED_ID, "the fused plan's identity");
    assert_eq!(fused_digests, FUSED_LEAVES, "the fused leaves");
    let moved = route_faults::with(Fault::NoPageLeg, || {
        route_faults::with(Fault::NoCountCheck, || leaf_digests(&words, &owned, false))
    });
    assert_ne!(moved.0, digests, "the pin sees the page leg");
}

/// ★ Each fixed or prepared table stated off its columns' height, one at a
/// time, is refused by the plan's route check naming that table: DECODE (on
/// the prepared opening) by its opened columns' height, KECCAK_RC and
/// REGISTER (folded constants) by their folded columns'. One table moves at a
/// time, so each refusal is the one check's: with that check taken out, the
/// plan's probe of the moved table asserts instead and the plan is refused
/// with another error ("probe never ran"), which this test would not accept.
/// (The checks' controls themselves are the unit test's,
/// [`every_route_mutant_is_refused_by_its_named_check`]: the probes run on
/// threads of their own, which the test faults do not reach.)
#[test]
fn each_fixed_or_prepared_table_off_its_height_is_refused_by_name() {
    let elf: &[u8] = include_bytes!("testdata/poc_rodata_commit.elf");
    let format = p1_format(MemoryFormat::Words, LeafScale::DoubledOneTop);
    let opts = super::super::proof::block_base_options_under(&format.zf);
    let honest = mix_statement(elf, &opts, &format, SMALL_WORDS, 2, 1);
    let plan =
        WhirBlockPlan::derive(elf, &opts, &format, honest.view(), None).expect("the honest plan");
    let names: Vec<String> = plan
        .frame
        .airs
        .air_refs()
        .iter()
        .map(|a| a.name().to_string())
        .collect();
    for (name, check) in [
        (DECODE_NAME, "prepared columns are"),
        (KECCAK_RC_NAME, "route's columns are"),
        (REGISTER_NAME, "route's columns are"),
    ] {
        let t = names.iter().position(|n| n == name).expect("a fixed table");
        let mut moved = honest.clone();
        moved.table_num_vars[t] = 2;
        assert_ne!(
            moved.table_num_vars[t], honest.table_num_vars[t],
            "{name} moves"
        );
        let refused = WhirBlockPlan::derive(elf, &opts, &format, moved.view(), None)
            .err()
            .unwrap_or_else(|| panic!("{name} at 2^2 rows is refused"));
        assert!(
            refused.contains(&format!(
                "table {t} ({name}) states 2^2 rows and its {check}"
            )),
            "{name}: {refused}"
        );
    }
}

/// The test functions in one source file that run the `test_ecrec` program
/// without the ECREC forgery lock (`trace_builder::ecrec_forge_free`, or the
/// lock itself), by name and line. A source reading, by item:
/// - an item names the program if its text holds `"test_ecrec"`, or the name
///   of another item of the file that names it: a program list (`const` or
///   `static`) or a helper (a function that is not a test) that does not take
///   the lock itself, followed to a fixed point;
/// - a test function (`#[test]` among the attributes and comments above it)
///   that names the program must take the lock in its own body.
fn tests_running_test_ecrec_unlocked(text: &str) -> Vec<String> {
    const LOCKS: [&str; 2] = ["ecrec_forge_free", "ECREC_FORGE_TEST_LOCK"];
    struct Item<'a> {
        name: &'a str,
        line: usize,
        body: &'a str,
        test: bool,
    }
    fn ident_at(s: &str) -> &str {
        let end = s
            .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
            .unwrap_or(s.len());
        &s[..end]
    }
    // The text from `from`'s first `{` to its matching `}`.
    fn braced(text: &str, from: usize) -> Option<&str> {
        let open = from + text[from..].find('{')?;
        let mut depth = 0usize;
        for (i, c) in text[open..].char_indices() {
            match c {
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        return Some(&text[open..=open + i]);
                    }
                }
                _ => {}
            }
        }
        None
    }
    fn names(body: &str, name: &str) -> bool {
        if name.starts_with('"') {
            return body.contains(name);
        }
        let word = |c: char| c.is_ascii_alphanumeric() || c == '_';
        body.match_indices(name).any(|(i, _)| {
            !body[..i].chars().next_back().is_some_and(word)
                && !body[i + name.len()..].chars().next().is_some_and(word)
        })
    }
    let lines: Vec<&str> = text.lines().collect();
    let mut starts = Vec::with_capacity(lines.len());
    let mut at = 0;
    for line in &lines {
        starts.push(at);
        at += line.len() + 1;
    }
    let mut items = Vec::new();
    for (n, line) in lines.iter().enumerate() {
        let mut rest = line.trim_start();
        for vis in ["pub(crate) ", "pub(super) ", "pub "] {
            rest = rest.strip_prefix(vis).unwrap_or(rest);
        }
        let (kind, after) = match ["fn ", "const ", "static "]
            .iter()
            .find_map(|k| rest.strip_prefix(k).map(|a| (*k, a)))
        {
            Some(found) => found,
            None => continue,
        };
        let name = ident_at(after);
        if name.is_empty() {
            continue;
        }
        let from = starts[n];
        let body = if kind == "fn " {
            // Its body, or none for a declaration (a trait's): the first `{`
            // or `;` outside the signature's parentheses and brackets.
            let mut depth = 0i32;
            let open = text[from..].chars().find(|&c| {
                match c {
                    '(' | '[' => depth += 1,
                    ')' | ']' => depth -= 1,
                    '{' | ';' => return depth == 0,
                    _ => {}
                }
                false
            });
            match (open, braced(text, from)) {
                (Some('{'), Some(body)) => body,
                _ => continue,
            }
        } else {
            let end = text[from..]
                .find(";\n")
                .map_or(text.len(), |i| from + i + 1);
            &text[from..end]
        };
        let test = lines[..n]
            .iter()
            .rev()
            .map(|l| l.trim())
            .take_while(|l| l.starts_with("#[") || l.starts_with("//"))
            .any(|l| l == "#[test]");
        items.push(Item {
            name,
            line: n + 1,
            body,
            test: test && kind == "fn ",
        });
    }
    let locked = |body: &str| LOCKS.iter().any(|l| body.contains(l));
    let mut naming: Vec<&str> = vec!["\"test_ecrec\""];
    loop {
        let more: Vec<&str> = items
            .iter()
            .filter(|it| !it.test && !naming.contains(&it.name) && !locked(it.body))
            .filter(|it| naming.iter().any(|n| names(it.body, n)))
            .map(|it| it.name)
            .collect();
        if more.is_empty() {
            break;
        }
        naming.extend(more);
    }
    items
        .iter()
        .filter(|it| it.test && !locked(it.body))
        .filter(|it| naming.iter().any(|n| names(it.body, n)))
        .map(|it| format!("{}:{}", it.name, it.line))
        .collect()
}

/// ★ Every test function that runs the `test_ecrec` program takes the ECREC
/// forgery lock in its own body: a test that runs `test_ecrec` beside the
/// forging test without it builds the forged call, and its block does not
/// verify (or its two builds differ). The program is reached by name, through
/// a program list, or through a helper ([`tests_running_test_ecrec_unlocked`]).
/// A source check, so a test added later without the lock fails here rather
/// than now and then in a threaded whole-library run.
#[test]
fn every_test_running_test_ecrec_holds_the_forgery_lock() {
    fn walk(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
        for entry in std::fs::read_dir(dir).expect("a source directory") {
            let path = entry.expect("an entry").path();
            if path.is_dir() {
                walk(&path, out);
            } else if path.extension().is_some_and(|e| e == "rs") {
                out.push(path);
            }
        }
    }
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut files = Vec::new();
    walk(&root, &mut files);
    let mut naming = 0;
    let mut unlocked = Vec::new();
    for path in &files {
        let text = std::fs::read_to_string(path).expect("a source file");
        if !text.contains("\"test_ecrec\"") {
            continue;
        }
        naming += 1;
        let shown = path
            .strip_prefix(&root)
            .unwrap_or(path)
            .display()
            .to_string();
        unlocked.extend(
            tests_running_test_ecrec_unlocked(&text)
                .into_iter()
                .map(|t| format!("{shown}: {t}")),
        );
    }
    assert!(naming >= 10, "{naming} files name test_ecrec");
    assert!(
        unlocked.is_empty(),
        "these tests run test_ecrec without the ECREC forgery lock: {unlocked:?}"
    );
}

/// The reading of [`tests_running_test_ecrec_unlocked`] on a made-up file: a
/// test that names the program, one that runs a list holding it, and two that
/// call a helper running it (one with a `;` in its signature), each without
/// the lock, are named; the same
/// with the lock, a test that names neither, a helper taking the lock itself
/// and a trait's declaration are not. Not blind: the lock taken out of one
/// test names that test.
#[test]
fn the_ecrec_lock_reading_names_each_unlocked_test() {
    let file = r#"
const PROGRAMS: [&str; 2] = ["all_instructions_64", ECREC_PROGRAM];
const OTHERS: [&str; 1] = ["test_keccak256"];

trait Runs {
    fn run(name: &str);
}

fn proves(name: &str) {
    let _ = name;
}

fn proves_ecrec() {
    proves(ECREC_PROGRAM);
}

fn listed() -> [&'static str; 1] {
    [ECREC_PROGRAM]
}

fn proves_ecrec_locked() {
    let _ecrec = crate::tables::trace_builder::ecrec_forge_free();
    proves(ECREC_PROGRAM);
}

#[test]
fn by_name_unlocked() {
    proves(ECREC_PROGRAM);
}

/// A list.
#[test]
#[ignore = "a box test"]
fn by_list_unlocked() {
    for name in PROGRAMS {
        proves(name);
    }
}

#[test]
fn by_helper_unlocked() {
    proves_ecrec();
}

#[test]
fn by_a_listing_helper_unlocked() {
    for name in listed() {
        proves(name);
    }
}

#[test]
fn by_name_locked() {
    let _ecrec = crate::tables::trace_builder::ecrec_forge_free();
    proves(ECREC_PROGRAM);
}

#[test]
fn by_list_locked() {
    let _ecrec = ecrec_forge_free();
    for name in PROGRAMS {
        proves(name);
    }
}

#[test]
fn by_a_locking_helper() {
    proves_ecrec_locked();
}

#[test]
fn others() {
    for name in OTHERS {
        proves(name);
    }
}
"#;
    // The program's name spelled apart, so the real check of this file does
    // not read the made-up tests as its own.
    let file = &file.replace("ECREC_PROGRAM", concat!("\"test", "_ecrec\""));
    assert_eq!(
        tests_running_test_ecrec_unlocked(file),
        vec![
            "by_name_unlocked:27".to_string(),
            "by_list_unlocked:34".to_string(),
            "by_helper_unlocked:41".to_string(),
            "by_a_listing_helper_unlocked:46".to_string(),
        ]
    );
    let unlocked = file.replacen(
        "fn by_name_locked() {\n    let _ecrec = crate::tables::trace_builder::ecrec_forge_free();\n",
        "fn by_name_locked() {\n",
        1,
    );
    assert_ne!(&unlocked, file, "the mutant moves the file");
    assert!(
        tests_running_test_ecrec_unlocked(&unlocked).contains(&"by_name_locked:53".to_string()),
        "{:?}",
        tests_running_test_ecrec_unlocked(&unlocked)
    );
}
