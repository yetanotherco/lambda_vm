//! ★ Page runs (`page_w::page_tables`): the rule that cuts a block's page
//! list into its page tables, a pure function of the list the verifier
//! derives. The tables are a partition of the list in its order; each block is
//! one class's run of contiguous pages, aligned to its own span (so it never
//! crosses `2^32` and its words' addresses never carry into the high word),
//! and the largest aligned block that fits where it starts; the prover's view
//! of the list (private pages with their bytes) and the verifier's (without)
//! give the same tables; the page sets of a median and a large block give
//! their counts; malformed lists and run caps are refused.

use super::page::{DEFAULT_PAGE_SIZE, PageConfig};
use super::page_w::{
    self, MAX_PAGE_RUN_LOG2, PageTable, PageTableKind, check_page_table, page_kind, page_tables,
};

const PAGE: u64 = DEFAULT_PAGE_SIZE as u64;
const INPUT_BASE: u64 = executor::vm::memory::PRIVATE_INPUT_START_INDEX;
const STACK_TOP_PAGE: u64 = 0xFFFF_FFFF_FFFC_0000;

/// A page of program data (a nonzero byte).
fn data(base: u64) -> PageConfig {
    PageConfig::with_data(base, vec![0, 0, 7])
}

/// A private-input page as the prover holds it (its bytes) or as the
/// verifier does (none).
fn private(base: u64, prover: bool) -> PageConfig {
    if prover {
        PageConfig::with_private_input(base, vec![1, 2, 3])
    } else {
        PageConfig {
            page_base: base,
            init_values: None,
            is_private_input: true,
        }
    }
}

/// `count` pages from `base`, each `make(base)`.
fn run(base: u64, count: u64, make: impl Fn(u64) -> PageConfig) -> Vec<PageConfig> {
    (0..count).map(|i| make(base + i * PAGE)).collect()
}

/// The page sets of a block's three runs: the program's 9 data pages and
/// `heap` zero pages from 0, `input` private-input pages, the stack's top page.
fn block_pages(heap: u64, input: u64, prover: bool) -> Vec<PageConfig> {
    let mut pages = run(0, 9, data);
    pages.extend(run(9 * PAGE, heap, PageConfig::zero_init));
    pages.extend(run(INPUT_BASE, input, |b| private(b, prover)));
    pages.push(PageConfig::zero_init(STACK_TOP_PAGE));
    pages
}

/// The median block's 183 pages and the large block's 351.
fn median() -> Vec<PageConfig> {
    block_pages(138, 35, false)
}

fn large() -> Vec<PageConfig> {
    block_pages(263, 78, false)
}

/// Every property a cut must have, checked independently of the rule's code:
/// a partition of `configs` in order; each table aligned to its span, at most
/// `2^run_log2` pages (one at `run_log2 = 0`), within `2^32` of its base's
/// high word, its pages contiguous and of its class; a data page alone; each
/// block the largest aligned block of its run that fits where it starts.
fn check_cut(configs: &[PageConfig], tables: &[PageTable], run_log2: u8) {
    let mut next = 0usize;
    for (t, table) in tables.iter().enumerate() {
        assert_eq!(table.first, next, "table {t} starts where the last ended");
        assert_eq!(
            table.base, configs[table.first].page_base,
            "table {t}'s base"
        );
        check_page_table(table).unwrap_or_else(|e| panic!("table {t}: {e}"));
        table
            .configs(configs)
            .unwrap_or_else(|e| panic!("table {t}: {e}"));
        assert!(table.log_pages <= run_log2, "table {t} past the cap");
        // Its words' addresses: `base_lo + 8·OFFSET` below 2^32 for every row.
        let base_lo = table.base & 0xFFFF_FFFF;
        assert!(
            base_lo + table.span() <= 1 << 32,
            "table {t} crosses 2^32: {table:?}"
        );
        assert_eq!(table.rows() as u64 * 8, table.span(), "a row a word");
        match table.kind {
            PageTableKind::Page => {
                assert_eq!(table.log_pages, 0, "a page table is one page");
                if run_log2 > 0 {
                    assert_eq!(page_kind(&configs[table.first]), PageTableKind::Page);
                }
            }
            kind => {
                assert!(run_log2 > 0, "no block with page runs off");
                // The largest aligned block that fits: one twice its size
                // either is misaligned, past the cap, or past the run's end.
                let index = table.base / PAGE;
                let doubled = table.log_pages + 1;
                let fits = (0..(1usize << doubled)).all(|k| {
                    configs.get(table.first + k).is_some_and(|c| {
                        page_kind(c) == kind
                            && Some(c.page_base) == table.base.checked_add(k as u64 * PAGE)
                    })
                });
                assert!(
                    doubled > run_log2 || !index.is_multiple_of(1 << doubled) || !fits,
                    "table {t} is not the largest block that fits: {table:?}"
                );
            }
        }
        next += table.pages();
    }
    assert_eq!(next, configs.len(), "the tables cover the list");
}

/// The tables' count by class: (pages, zero blocks, private blocks).
fn census(tables: &[PageTable]) -> (usize, usize, usize) {
    let of = |kind| tables.iter().filter(|t| t.kind == kind).count();
    (
        of(PageTableKind::Page),
        of(PageTableKind::Zero),
        of(PageTableKind::Private),
    )
}

/// ★ The median block's 183 pages and the large block's 351 cut at every cap:
/// at 16 pages, 27 tables (9 data pages, 14 zero blocks, 4 private) and 36
/// (9, 20, 7); one a page with runs off; no block crosses `2^32`, the large
/// private run cut at it by alignment alone.
#[test]
fn the_median_and_large_page_sets_cut_to_their_counts() {
    let expect: [(u8, usize, usize); 5] = [
        (0, 183, 351),
        (2, 57, 97),
        (3, 37, 56),
        (4, 27, 36),
        (6, 22, 23),
    ];
    let (m, l) = (median(), large());
    assert_eq!((m.len(), l.len()), (183, 351));
    for (run_log2, median_tables, large_tables) in expect {
        let tm = page_tables(&m, run_log2).expect("median");
        let tl = page_tables(&l, run_log2).expect("large");
        check_cut(&m, &tm, run_log2);
        check_cut(&l, &tl, run_log2);
        assert_eq!(
            (tm.len(), tl.len()),
            (median_tables, large_tables),
            "at 2^{run_log2}"
        );
    }
    let tm = page_tables(&m, 4).expect("median");
    let tl = page_tables(&l, 4).expect("large");
    assert_eq!(census(&tm), (9, 14, 4));
    assert_eq!(census(&tl), (9, 20, 7));
    // The large private run (78 pages from 0xFF000000) crosses 2^32 at its
    // 65th page: a block ends there and the next starts at 2^32.
    assert!(
        tl.iter()
            .any(|t| t.kind == PageTableKind::Private && t.base == 1 << 32)
    );
    assert!(
        tl.iter()
            .filter(|t| t.kind == PageTableKind::Private)
            .all(|t| t.base >= 1 << 32 || t.base + t.span() <= 1 << 32)
    );
    // The heap's run from page 9: 1, 2, 4 pages to reach alignment, then 16s.
    let heap: Vec<u8> = tm
        .iter()
        .filter(|t| t.kind == PageTableKind::Zero && t.base < INPUT_BASE)
        .map(|t| t.log_pages)
        .collect();
    assert_eq!(&heap[..4], &[0, 1, 2, 4]);
    // Runs off: today's tables, one a page, in the list's order.
    let today = page_tables(&m, 0).expect("median");
    assert!(today.iter().enumerate().all(|(i, t)| {
        t.kind == PageTableKind::Page
            && t.first == i
            && t.log_pages == 0
            && t.base == m[i].page_base
    }));
    assert_eq!(today[0].name(), "PAGE_W:0x0");
}

/// A deterministic stream of page lists: runs of each class with gaps,
/// starting anywhere (low pages, around `2^32`, the top of the address space).
fn lists() -> Vec<Vec<PageConfig>> {
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let starts = [0u64, 9 * PAGE, (1 << 32) - 37 * PAGE, INPUT_BASE, 1 << 40];
    let mut out = Vec::new();
    for case in 0..200 {
        let mut pages: Vec<PageConfig> = Vec::new();
        let mut base = starts[case % starts.len()] + (next() % 64) * PAGE;
        for _ in 0..(1 + next() % 8) {
            let count = 1 + next() % 40;
            let class = next() % 4;
            for i in 0..count {
                let b = base + i * PAGE;
                pages.push(match class {
                    0 => data(b),
                    1 => private(b, case % 2 == 0),
                    // An all-zero page of the program: a zero page.
                    2 => PageConfig::with_data(b, vec![0; 64]),
                    _ => PageConfig::zero_init(b),
                });
            }
            base += (count + next() % 3) * PAGE;
        }
        out.push(pages);
    }
    // The top of the address space: the stack's top page and its neighbour.
    out.push(run(STACK_TOP_PAGE - PAGE, 2, PageConfig::zero_init));
    out.push(vec![PageConfig::zero_init(STACK_TOP_PAGE)]);
    out.push(Vec::new());
    out
}

/// ★ On every list of the stream and at every cap, the cut has every
/// property [`check_cut`] checks, and is a function of the list alone: the
/// same list gives the same tables, and the prover's view of its private pages
/// (their bytes) the verifier's.
#[test]
fn every_cut_is_an_aligned_partition_and_the_lists_alone_decide_it() {
    for (case, list) in lists().iter().enumerate() {
        let verifier: Vec<PageConfig> = list
            .iter()
            .map(|c| {
                if c.is_private_input {
                    private(c.page_base, false)
                } else {
                    c.clone()
                }
            })
            .collect();
        for run_log2 in 0..=MAX_PAGE_RUN_LOG2 {
            let tables = page_tables(list, run_log2).unwrap_or_else(|e| panic!("{case}: {e}"));
            check_cut(list, &tables, run_log2);
            assert_eq!(
                page_tables(list, run_log2).expect("again"),
                tables,
                "{case}: deterministic"
            );
            assert_eq!(
                page_tables(&verifier, run_log2).expect("the verifier's view"),
                tables,
                "{case}: the verifier's view"
            );
            // A table's name says its class, base and pages.
            for table in &tables {
                let name = table.name();
                let prefix = match table.kind {
                    PageTableKind::Page => page_w::NAME_PREFIX,
                    PageTableKind::Zero => page_w::ZERO_NAME_PREFIX,
                    PageTableKind::Private => page_w::PRIVATE_NAME_PREFIX,
                };
                assert!(name.starts_with(prefix), "{name}");
                assert!(name.contains(&format!("0x{:x}", table.base)), "{name}");
            }
        }
    }
}

/// The edges: a single page of each class; a run whose first page is odd
/// (blocks grow 1, 2, 4, … to alignment, then shrink to the run's end); a
/// data page splitting a zero run; same-class pages with a gap; an all-zero
/// program page joining a runtime zero run; the top page.
#[test]
fn the_edges_cut_as_the_rule_says() {
    let logs = |list: &[PageConfig], run_log2| -> Vec<(PageTableKind, u8)> {
        page_tables(list, run_log2)
            .expect("cut")
            .iter()
            .map(|t| (t.kind, t.log_pages))
            .collect()
    };
    use PageTableKind::{Page, Private, Zero};
    // Single pages: a block of one each, but a data page.
    let singles = vec![
        data(0),
        PageConfig::zero_init(4 * PAGE),
        private(INPUT_BASE, true),
    ];
    assert_eq!(logs(&singles, 4), [(Page, 0), (Zero, 0), (Private, 0)]);
    // From page 3, 29 pages: 1 (3), 4 (4..8), 8 (8..16), 16 (16..32).
    assert_eq!(
        logs(&run(3 * PAGE, 29, PageConfig::zero_init), 4),
        [(Zero, 0), (Zero, 2), (Zero, 3), (Zero, 4)]
    );
    // At cap 2: 1, then 4s.
    assert_eq!(
        logs(&run(3 * PAGE, 9, PageConfig::zero_init), 2),
        [(Zero, 0), (Zero, 2), (Zero, 2)]
    );
    // A run ending unaligned: 16 then 2 then 1.
    assert_eq!(
        logs(&run(0, 19, PageConfig::zero_init), 4),
        [(Zero, 4), (Zero, 1), (Zero, 0)]
    );
    // A data page splits a zero run.
    let mut split = run(0, 4, PageConfig::zero_init);
    split[2] = data(2 * PAGE);
    assert_eq!(logs(&split, 4), [(Zero, 1), (Page, 0), (Zero, 0)]);
    // A gap splits a run of one class.
    let gap = vec![PageConfig::zero_init(0), PageConfig::zero_init(2 * PAGE)];
    assert_eq!(logs(&gap, 4), [(Zero, 0), (Zero, 0)]);
    // An all-zero page of the program is a zero page.
    let joined = vec![
        PageConfig::with_data(0, vec![0; 100]),
        PageConfig::zero_init(PAGE),
    ];
    assert_eq!(logs(&joined, 4), [(Zero, 1)]);
    // Private pages and zero pages never share a block.
    let mixed = vec![private(0, true), PageConfig::zero_init(PAGE)];
    assert_eq!(logs(&mixed, 4), [(Private, 0), (Zero, 0)]);
    // The top of the address space: two pages, one block, no overflow.
    let top = run(STACK_TOP_PAGE - PAGE, 2, PageConfig::zero_init);
    let tables = page_tables(&top, 4).expect("top");
    assert_eq!(tables.len(), 1);
    assert_eq!(tables[0].base & 0xFFFF_FFFF, 0xFFF8_0000);
    assert_eq!((tables[0].base & 0xFFFF_FFFF) + tables[0].span(), 1 << 32);
    assert_eq!(tables[0].name(), "PAGE_Z:0xfffffffffff80000:2");
    assert!(page_tables(&[], 4).expect("empty").is_empty());
}

/// Refused: a cap past [`MAX_PAGE_RUN_LOG2`]; a list out of order, with a
/// duplicate base or an unaligned one; a table not aligned to its span, past
/// the cap, or whose pages are not its own.
#[test]
fn malformed_lists_and_tables_are_refused() {
    let list = run(0, 4, PageConfig::zero_init);
    assert!(page_tables(&list, MAX_PAGE_RUN_LOG2).is_ok());
    assert!(page_tables(&list, MAX_PAGE_RUN_LOG2 + 1).is_err());
    let mut reversed = list.clone();
    reversed.reverse();
    let mut duplicate = list.clone();
    duplicate[1].page_base = 0;
    let mut unaligned = list.clone();
    unaligned[3].page_base += 8;
    for (name, bad) in [
        ("reversed", reversed),
        ("duplicate", duplicate),
        ("unaligned", unaligned),
    ] {
        for run_log2 in [0, 4] {
            assert!(page_tables(&bad, run_log2).is_err(), "{name} at {run_log2}");
        }
    }
    let block = |base, log_pages, kind| PageTable {
        kind,
        base,
        log_pages,
        first: 0,
    };
    assert!(check_page_table(&block(4 * PAGE, 2, PageTableKind::Zero)).is_ok());
    assert!(check_page_table(&block(2 * PAGE, 2, PageTableKind::Zero)).is_err());
    assert!(check_page_table(&block(0, MAX_PAGE_RUN_LOG2 + 1, PageTableKind::Zero)).is_err());
    // Its pages must be its own: their bases and class.
    assert!(block(0, 2, PageTableKind::Zero).configs(&list).is_ok());
    assert!(block(0, 2, PageTableKind::Private).configs(&list).is_err());
    assert!(block(0, 3, PageTableKind::Zero).configs(&list).is_err());
    let mut with_data = list.clone();
    with_data[1] = data(PAGE);
    assert!(
        block(0, 2, PageTableKind::Zero)
            .configs(&with_data)
            .is_err()
    );
    assert!(block(PAGE, 1, PageTableKind::Zero).configs(&list).is_err());
}

// =========================================================================
// The page tables on the bench (`word_memory_tests`' model, no proof)
// =========================================================================

use super::memw_w::cols as mw;
use super::page_w::{FinalWordMap, cols as pw, zero_cols as zw};
use super::types::{BusId, FE};
use super::word_memory_tests::{
    Bench, CPU_WORD_TABLES, Model, Piece, chip_free_counts, memw_w_row, set_chip_word, verdict,
};
use crate::block_whir::{self, BlockFormat, OwnedBlockStatement};
use crate::tables::fused_memory::MemoryFormat;
use stark::lookup::{BusValue, LinearTerm};
use stark::proof::options::ProofOptions;

const DATA_BASE: u64 = 8 * PAGE;
const ZERO_BLOCK: u64 = 4 * PAGE;

/// A page table's bench piece, as the verifier builds its table: its
/// interactions and preprocessed columns, its trace from `words`.
fn piece_of(table: &PageTable, configs: &[PageConfig], words: &FinalWordMap) -> Piece {
    Piece {
        name: table.name(),
        interactions: page_w::page_table_interactions(table, configs).expect("its interactions"),
        trace: page_w::generate_page_table_trace(table, configs, words).expect("its trace"),
        preprocessed: page_w::page_table_preprocessed_columns(table, configs).expect("its columns"),
        constraints: None,
    }
}

/// `m`'s bench, its pages in the page tables of runs of at most `2^run_log2`
/// pages.
fn bench_runs(m: &Model, run_log2: u8) -> Bench {
    let tables = page_tables(&m.pages, run_log2).expect("the pages cut");
    m.bench_with(
        tables
            .iter()
            .map(|t| piece_of(t, &m.pages, &m.words))
            .collect(),
    )
}

fn elf_data() -> Vec<u8> {
    (0..4096u32).map(|i| (i * 37 + 11) as u8).collect()
}

/// The pages of the page-run benches: a zero page at index 3 (odd: a block
/// of one), zero pages 4..8 (one block of four from runs of 4 pages on), a
/// page of data at 8, two private-input pages (one block of two from runs of
/// 2 on), the stack's top page.
fn run_pages() -> Vec<PageConfig> {
    let input: Vec<u8> = (0..(DEFAULT_PAGE_SIZE + 4096) as u32)
        .map(|i| (i * 101 + 3) as u8)
        .collect();
    let mut pages = run(3 * PAGE, 5, PageConfig::zero_init);
    pages.push(PageConfig::with_data(DATA_BASE, elf_data()));
    pages.push(PageConfig::with_private_input(
        INPUT_BASE,
        input[..DEFAULT_PAGE_SIZE].to_vec(),
    ));
    pages.push(PageConfig::with_private_input(
        INPUT_BASE + PAGE,
        input[DEFAULT_PAGE_SIZE..].to_vec(),
    ));
    pages.push(PageConfig::zero_init(STACK_TOP_PAGE));
    pages
}

/// `pages` with the honest accesses of the page-run benches, but the read of
/// the page of data unless `data_read`: two words across the zero block's
/// first internal page boundary written and read back, a zero page read, a
/// word of the second private page read, the stack written and read.
fn run_model(pages: Vec<PageConfig>, data_read: bool) -> Model {
    let mut m = Model::new(pages);
    m.write(ZERO_BLOCK + PAGE - 8, 8, [0x1122_3344_5566_7788, u64::MAX]);
    m.read(ZERO_BLOCK + PAGE - 8, 12);
    m.read(3 * PAGE + 0x100, 16);
    if data_read {
        m.read(DATA_BASE + 0x40, 20);
    }
    m.read(INPUT_BASE + PAGE + 0x10, 24);
    m.write(STACK_TOP_PAGE + 0x3_FF00, 28, [0xDEAD_BEEF, 0xCAFE]);
    m.read(STACK_TOP_PAGE + 0x3_FF00, 32);
    m
}

/// The names of a bench's pieces.
fn names(bench: &Bench) -> Vec<String> {
    bench.pieces.iter().map(|p| p.name.clone()).collect()
}

/// ★ The honest bench at every run: accepted, the same accesses balancing
/// whatever tables the pages are cut into. With runs off its pieces are
/// today's, piece for piece (names, interactions, every cell, the
/// preprocessed columns); at runs of 4 pages the zero run 3..8 is two tables,
/// the private pages one, the page of data and the stack's top page their
/// own.
#[test]
fn the_page_run_bench_is_accepted_at_every_run_and_is_todays_with_runs_off() {
    let m = run_model(run_pages(), true);
    let today = m.bench();
    let off = bench_runs(&m, 0);
    assert_eq!(names(&off), names(&today));
    for (a, b) in off.pieces.iter().zip(&today.pieces) {
        assert!(a.interactions == b.interactions, "{}", a.name);
        assert_eq!(a.preprocessed, b.preprocessed, "{}", a.name);
        let (s, t) = (&a.trace, &b.trace);
        assert_eq!(
            (s.num_rows(), s.main_table.width),
            (t.num_rows(), t.main_table.width)
        );
        for r in 0..s.num_rows() {
            for c in 0..s.main_table.width {
                assert_eq!(s.get_main(r, c), t.get_main(r, c), "{} {r} {c}", a.name);
            }
        }
    }
    for run_log2 in 0..=MAX_PAGE_RUN_LOG2 {
        let v = verdict(&bench_runs(&m, run_log2));
        assert!(v.accepted(), "at 2^{run_log2}: {v:?}");
    }
    let at2 = bench_runs(&m, 2);
    assert_eq!(
        names(&at2)[..5],
        [
            "PAGE_Z:0xc0000:1",
            "PAGE_Z:0x100000:4",
            "PAGE_W:0x200000",
            "PAGE_P:0xff000000:2",
            "PAGE_Z:0xfffffffffffc0000:1",
        ]
    );
    // The zero block: 4 columns, the words across its internal boundary its
    // rows 2^15 − 1 and 2^15.
    let block = at2.piece("PAGE_Z:0x100000:4");
    assert_eq!(block.trace.main_table.width, zw::NUM_COLUMNS);
    assert_eq!(block.trace.num_rows(), 4 * page_w::ROWS);
    let at = |r: usize, c: usize| block.trace.get_main(r, c).canonical_u64();
    let edge = page_w::ROWS;
    assert_eq!(
        (at(edge - 1, zw::OFFSET), at(edge, zw::OFFSET)),
        (edge as u64 - 1, edge as u64)
    );
    assert_eq!(
        (at(edge - 1, zw::TIMESTAMP), at(edge, zw::TIMESTAMP)),
        (12, 12)
    );
    assert_eq!(
        (at(edge, zw::FINI_LO), at(edge, zw::FINI_HI)),
        (0xFFFF_FFFF, 0xFFFF_FFFF)
    );
    // Its mint: `(base_lo + 8·OFFSET, base_hi, 0, 0, 0)`.
    assert_eq!(
        block.interactions[0].values,
        vec![
            BusValue::linear(vec![
                LinearTerm::Constant(ZERO_BLOCK as i64),
                super::word_memory::term(8, zw::OFFSET),
            ]),
            BusValue::constant(0),
            BusValue::constant(0),
            BusValue::constant(0),
            BusValue::constant(0),
        ]
    );
}

/// A bench's piece `name` replaced by `piece`.
fn replace(bench: &mut Bench, name: &str, piece: Piece) {
    let at = bench
        .pieces
        .iter()
        .position(|p| p.name == name)
        .unwrap_or_else(|| panic!("the bench holds {name}"));
    bench.pieces[at] = piece;
}

/// A zero block's piece laid out by hand, whatever its alignment or height:
/// `interactions` (a zero block's), `rows` rows from `base`, each row's final
/// state from `words` at its byte address `base + 8·row`, `OFFSET` the row
/// index and preprocessed at that height.
fn zero_piece_by_hand(
    name: &str,
    base: u64,
    rows: usize,
    words: &FinalWordMap,
    interactions: Vec<stark::lookup::BusInteraction>,
) -> Piece {
    let trace = super::word_memory_tests::table(rows, zw::NUM_COLUMNS, |t, r| {
        let state = words
            .get(&(base + 8 * r as u64))
            .copied()
            .unwrap_or_default();
        let (lo, hi) = super::word_memory::halves(state.value);
        t.main_table.set(r, zw::OFFSET, FE::from(r as u64));
        t.main_table.set(r, zw::FINI_LO, FE::from(u64::from(lo)));
        t.main_table.set(r, zw::FINI_HI, FE::from(u64::from(hi)));
        t.main_table
            .set(r, zw::TIMESTAMP, FE::from(state.timestamp));
    });
    Piece {
        name: name.into(),
        interactions,
        trace,
        preprocessed: vec![page_w::offset_column_of(rows)],
        constraints: None,
    }
}

/// ★ A block taller than its pages: the zero block of pages 4..8 stated
/// at 8 pages' rows (its interactions its own, its `OFFSET` at that height, as
/// a verifier without the height rule would build it) mints the next page's
/// words a second time, the page of data's included, with the word 0; a
/// forged read of the data page's first words as 0 then balances every bus,
/// the data page's own table left untouched. The frame's height rule refuses
/// that statement ([`a_page_table_is_held_to_its_place_name_and_height`]).
#[test]
fn a_taller_block_would_mint_the_next_pages_words_twice() {
    // The forger's accesses: the honest ones, and the data page's first two
    // words read as 0 (what the taller block minted for them).
    let mut forger_pages = run_pages();
    let data = forger_pages
        .iter()
        .position(|c| c.page_base == DATA_BASE)
        .expect("the data page");
    forger_pages[data] = PageConfig::zero_init(DATA_BASE);
    let mut m = run_model(forger_pages, false);
    assert_eq!(m.read(DATA_BASE, 36), [0, 0], "read as the word 0");
    let honest = page_tables(&run_pages(), 2).expect("cut");
    let mut forged = m.bench_with(
        honest
            .iter()
            .map(|t| {
                // The data page's own table as the verifier has it, untouched.
                let words = if t.base == DATA_BASE {
                    FinalWordMap::new()
                } else {
                    m.words.clone()
                };
                piece_of(t, &run_pages(), &words)
            })
            .collect(),
    );
    let block = PageTable {
        kind: PageTableKind::Zero,
        base: ZERO_BLOCK,
        log_pages: 2,
        first: 0,
    };
    let own_pages = run(ZERO_BLOCK, 4, PageConfig::zero_init);
    let own = page_w::page_table_interactions(&block, &own_pages).expect("its own");
    let taller = zero_piece_by_hand("TALLER", ZERO_BLOCK, 8 * page_w::ROWS, &m.words, own);
    replace(&mut forged, &block.name(), taller);
    let v = verdict(&forged);
    assert!(
        v.accepted(),
        "control: a taller block's forgery balances: {v:?}"
    );
    // With the block at its own height the read of 0 is left unanswered.
    let mut held = forged.clone();
    replace(&mut held, "TALLER", piece_of(&block, &own_pages, &m.words));
    let v = verdict(&held);
    assert!(v.refused_only_on(BusId::MemWord), "{v:?}");
}

/// ★ A page of data presented as zero: the verifier's cut never puts a
/// data page in a block, and a zero block over one is refused by the table's
/// own constructor; so the forger's read of the data page as 0 consumes a
/// token no table of the verifier's mints, and only the word bus refuses it.
/// Control: the classification taken from the forger (the page zero, its
/// run one block), and the forged read balances.
#[test]
fn a_page_of_data_in_a_zero_block_is_refused_by_the_verifiers_cut() {
    let pages = run_pages();
    let mut forger_pages = pages.clone();
    let data = forger_pages
        .iter()
        .position(|c| c.page_base == DATA_BASE)
        .expect("the data page");
    forger_pages[data] = PageConfig::zero_init(DATA_BASE);
    let mut m = run_model(forger_pages.clone(), false);
    assert_eq!(m.read(DATA_BASE, 36), [0, 0]);
    // The verifier's tables over the forger's accesses.
    let tables = page_tables(&pages, 3).expect("cut");
    assert!(
        tables
            .iter()
            .any(|t| t.kind == PageTableKind::Page && t.base == DATA_BASE),
        "the data page is a table of its own"
    );
    let verifiers = m.bench_with(
        tables
            .iter()
            .map(|t| piece_of(t, &pages, &m.words))
            .collect(),
    );
    let v = verdict(&verifiers);
    assert!(v.refused_only_on(BusId::MemWord), "{v:?}");
    // The forger's cut: the data page joins the zero run 3..9 as a block.
    let theirs = page_tables(&forger_pages, 3).expect("cut");
    let block = theirs
        .iter()
        .find(|t| {
            t.kind == PageTableKind::Zero && t.base <= DATA_BASE && DATA_BASE < t.base + t.span()
        })
        .expect("the forger's zero table over the data page");
    let v = verdict(&bench_runs(&m, 3));
    assert!(v.accepted(), "control: {v:?}");
    // That table over the true pages: no constructor builds it.
    assert!(block.configs(&pages).is_err());
    assert!(page_w::page_table_interactions(block, &pages).is_err());
    assert!(page_w::generate_page_table_trace(block, &pages, &FinalWordMap::new()).is_err());
    let opts = ProofOptions::default_test_options();
    assert!(crate::test_utils::create_page_table_air(&opts, block, &pages).is_err());
}

/// A zero block's interactions at any `base`, aligned or not: an aligned
/// block's with its base's two constants moved.
fn zero_block_interactions_at(base: u64) -> Vec<stark::lookup::BusInteraction> {
    let aligned = PageTable {
        kind: PageTableKind::Zero,
        base: 0,
        log_pages: 1,
        first: 0,
    };
    let mut out = page_w::page_table_interactions(&aligned, &run(0, 2, PageConfig::zero_init))
        .expect("aligned");
    for bi in &mut out {
        bi.values[0] = BusValue::linear(vec![
            LinearTerm::Constant((base & 0xFFFF_FFFF) as i64),
            super::word_memory::term(8, zw::OFFSET),
        ]);
        bi.values[1] = BusValue::constant(base >> 32);
    }
    out
}

/// ★ A block off its span's alignment: the derivation never cuts one
/// (`every_cut_is_an_aligned_partition_and_the_lists_alone_decide_it`) and
/// no constructor builds one. Its words would run past `2^32` in the key's
/// low half: two pages from `2^32 − 2^18` as one block mint `(2^32, 0)`, a key
/// no aligned table names (the word at `2^32` is `(0, 1)`). A read whose
/// second word is named `(2^32, 0)` (a pointer at `2^32 − 8`, its next word
/// `+ 8` in the low half) consumes a token no aligned table minted and is
/// refused on the word bus alone; control: the misaligned block, which mints
/// it, and the read balances.
#[test]
fn a_block_off_its_alignment_would_mint_past_two_to_the_32() {
    let low = (1u64 << 32) - PAGE;
    let pages = vec![PageConfig::zero_init(low), PageConfig::zero_init(1 << 32)];
    let misaligned = PageTable {
        kind: PageTableKind::Zero,
        base: low,
        log_pages: 1,
        first: 0,
    };
    assert!(check_page_table(&misaligned).is_err());
    let opts = ProofOptions::default_test_options();
    assert!(crate::test_utils::create_page_table_air(&opts, &misaligned, &pages).is_err());
    assert!(page_w::page_table_interactions(&misaligned, &pages).is_err());
    assert!(
        page_tables(&pages, 1)
            .expect("cut")
            .iter()
            .all(|t| t.log_pages == 0),
        "the verifier cuts two pages of one"
    );
    // The read at `2^32 − 8`, its second word's key `(2^32, 0)`: the model's
    // rows at the true word `2^32`, its MEMW_W row's key moved.
    let mut m = Model::new(pages.clone());
    m.read((1 << 32) - 8, 8);
    let row = memw_w_row(&m, 1 << 32, 8);
    let mut bench = bench_runs(&m, 0);
    bench.set_u64("MEMW_W", row, mw::ADDRESS_LO, 1 << 32);
    bench.set_u64("MEMW_W", row, mw::ADDRESS_HI, 0);
    // The aligned tables: the read's tokens at `(0, 1)` were the page's.
    let at_2_32 = page_tables(&pages, 0).expect("cut")[1];
    replace(
        &mut bench,
        &at_2_32.name(),
        piece_of(&at_2_32, &pages, &FinalWordMap::new()),
    );
    let v = verdict(&bench);
    assert!(v.refused_only_on(BusId::MemWord), "{v:?}");
    // Control: one misaligned block over both pages, its rows by hand.
    // Its row 2^15 is the word at 2^32, its key `(2^32, 0)`.
    let mut control = bench_runs(&m, 0);
    control.set_u64("MEMW_W", row, mw::ADDRESS_LO, 1 << 32);
    control.set_u64("MEMW_W", row, mw::ADDRESS_HI, 0);
    control
        .pieces
        .retain(|p| !p.name.starts_with(page_w::NAME_PREFIX));
    control.pieces.push(zero_piece_by_hand(
        "MISALIGNED",
        low,
        2 * page_w::ROWS,
        &m.words,
        zero_block_interactions_at(low),
    ));
    let v = verdict(&control);
    assert!(v.accepted(), "control: {v:?}");
}

/// ★ A block's internal page boundary (rows `2^15 − 1` and `2^15` of the
/// zero block, the last word of its first page and the first of its second,
/// both written and read): a final word or timestamp moved there is refused
/// by the word bus alone. Its `OFFSET` is the verifier's across the boundary:
/// a forger's row of the second page re-minting the first page's word 0
/// (`OFFSET` 0) answers a read of 0 after that word was written, and is
/// refused by the preprocessed check alone; control: that `OFFSET` the
/// forger's, and the forged read balances.
#[test]
fn a_blocks_internal_page_boundary_is_held() {
    let name = "PAGE_Z:0x100000:4";
    let edge = page_w::ROWS;
    let m = run_model(run_pages(), true);
    let bench = bench_runs(&m, 2);
    for (row, col) in [
        (edge, zw::FINI_LO),
        (edge, zw::FINI_HI),
        (edge - 1, zw::TIMESTAMP),
        (edge, zw::TIMESTAMP),
    ] {
        let mut forged = bench.clone();
        let v = forged.get(name, row, col) + FE::one();
        forged.set(name, row, col, v);
        let v = verdict(&forged);
        assert!(v.refused_only_on(BusId::MemWord), "{row} {col}: {v:?}");
    }

    // The re-mint: page 4's word 0 written at 36, then read at 40 as 0 from a
    // second chain minted by row 2^15 + 2 (page 5's word 2) with OFFSET 0.
    let mut m = run_model(run_pages(), true);
    m.write(ZERO_BLOCK, 36, [0x77, 0x88]);
    m.read(ZERO_BLOCK, 40);
    let mut forged = bench_runs(&m, 2);
    let r = memw_w_row(&m, ZERO_BLOCK, 40);
    for (col, value) in [
        (mw::OLD_LO, 0),
        (mw::OLD_HI, 0),
        (mw::NEW_LO, 0),
        (mw::NEW_HI, 0),
        (mw::STEP_0, 39),
        (mw::STEP_1, 0),
    ] {
        forged.set_u64("MEMW_W", r, col, value);
    }
    let reader = m.readers.iter().position(|c| c.ts == 40).expect("the read");
    set_chip_word(&mut forged, "READER", reader, 0, 0, 0);
    // Chain A ends at the write; chain B at the forged read, on row 2^15 + 2.
    forged.set_u64(name, 0, zw::TIMESTAMP, 36);
    forged.set_u64(name, 0, zw::FINI_LO, 0x77);
    let b = edge + 2;
    forged.set_u64(name, b, zw::OFFSET, 0);
    forged.set_u64(name, b, zw::TIMESTAMP, 40);
    forged.set_u64(name, b, zw::FINI_LO, 0);
    forged.set_u64(name, b, zw::FINI_HI, 0);
    let v = verdict(&forged);
    assert_eq!(
        v.preprocessed,
        vec![(name.to_string(), b, zw::OFFSET)],
        "{v:?}"
    );
    assert!(
        v.left.is_empty() && v.unanswered.is_empty() && v.broken.is_empty(),
        "{v:?}"
    );
    let column = (0..forged.piece(name).trace.num_rows())
        .map(|r| forged.get(name, r, zw::OFFSET))
        .collect();
    forged.piece_mut(name).preprocessed[zw::OFFSET] = column;
    let v = verdict(&forged);
    assert!(v.accepted(), "control: {v:?}");
}

/// ★ A private block's initial words are canonical, each 16-bit half by
/// its own `IS_HALF`, on its second page too: one half of a word of the
/// second private page moved past 16 bits (carried through the read to the
/// block's final word) is refused by that half's lookup alone; control: that
/// lookup removed, the read takes a word that is not one.
#[test]
fn a_private_blocks_init_word_past_32_bits_is_refused_by_its_half_check() {
    let m = run_model(run_pages(), true);
    let bench = bench_runs(&m, 2);
    let table = page_tables(&m.pages, 2)
        .expect("cut")
        .into_iter()
        .find(|t| t.kind == PageTableKind::Private)
        .expect("the private block");
    assert_eq!(table.pages(), 2);
    let name = table.name();
    let word = page_w::ROWS + 0x10 / 8;
    let row = memw_w_row(&m, INPUT_BASE + PAGE + 0x10, 24);
    let reader = m.readers.iter().position(|r| r.ts == 24).expect("the read");
    let read = m.readers[reader].words[0];
    for k in 0..4 {
        let mut forged = bench.clone();
        let (delta, upper) = match k {
            0 | 2 => (FE::from(1u64 << 32), k == 2),
            _ => (FE::from(1u64 << 16), k == 3),
        };
        let h = forged.get(&name, word, pw::INIT[k]) + delta;
        forged.set(&name, word, pw::INIT[k], h);
        let (fini, old, new) = if upper {
            (pw::FINI_HI, mw::OLD_HI, mw::NEW_HI)
        } else {
            (pw::FINI_LO, mw::OLD_LO, mw::NEW_LO)
        };
        let half = forged.get("MEMW_W", row, old) + FE::from(1u64 << 32);
        forged.set(&name, word, fini, half);
        forged.set("MEMW_W", row, old, half);
        forged.set("MEMW_W", row, new, half);
        let (lo, hi) = (read & 0xFFFF_FFFF, read >> 32);
        let (lo, hi) = if upper {
            (lo, hi + (1 << 32))
        } else {
            (lo + (1 << 32), hi)
        };
        set_chip_word(&mut forged, "READER", reader, 0, lo, hi);
        let v = verdict(&forged);
        assert_eq!(
            v.unanswered,
            vec![(u64::from(BusId::IsHalfword), vec![h.canonical_u64()])],
            "INIT_{k}: {v:?}"
        );
        assert!(v.left.is_empty() && v.broken.is_empty() && v.preprocessed.is_empty());
        let mut control = page_w::page_table_interactions(&table, &m.pages).expect("its own");
        control.remove(2 + k);
        let v = verdict(&forged.with_interactions(&name, control));
        assert!(v.accepted(), "INIT_{k} control: {v:?}");
    }
}

// =========================================================================
// The frame: the page tables are the verifier's
// =========================================================================

/// A word-format statement under `format` (no chip call, one table of each
/// CPU word family): the program `all_instructions_64`'s pages of data, a
/// heap of 20 zero pages from page 64, two private-input pages and the
/// stack's top page; every table at 2^2 rows but BITWISE's 2^20 and each page
/// table at its own height; one group a table. With the page tables it was
/// cut into.
fn statement_at(format: &BlockFormat) -> (OwnedBlockStatement, Vec<PageTable>) {
    statement_with_heap(format, 20)
}

/// [`statement_at`] with a heap of `heap` zero pages from page 64.
fn statement_with_heap(format: &BlockFormat, heap: u64) -> (OwnedBlockStatement, Vec<PageTable>) {
    let elf = crate::test_utils::asm_elf_bytes("all_instructions_64");
    let program = executor::elf::Elf::load(&elf).expect("the ELF loads");
    let opts = ProofOptions::default_test_options();
    let runtime_page_ranges = vec![
        crate::RuntimePageRange {
            base: 64 * PAGE,
            count: heap,
        },
        crate::RuntimePageRange {
            base: STACK_TOP_PAGE,
            count: 1,
        },
    ];
    let pages = crate::tables::trace_builder::Traces::page_configs_from_elf_and_runtime(
        &program,
        &runtime_page_ranges,
        2,
        1 << 20,
    )
    .expect("pages");
    let counts = chip_free_counts();
    let inputs =
        crate::BlockAirInputs::new(&opts, &program, vec![4; counts.cpu], &[]).expect("inputs");
    let mut airs = crate::block_airs(
        &program,
        &opts,
        &pages,
        &counts,
        &inputs,
        format.keccak_rnd,
        format.memw_aligned,
        format.registers,
    );
    airs.with_word_memory_at(
        &opts,
        CPU_WORD_TABLES.map(|n| n as usize),
        &pages,
        format.page_run_log2,
    )
    .expect("the word tables");
    let tables = page_tables(&pages, format.page_run_log2).expect("cut");
    let first = block_whir::first_page_index(&airs).expect("the pages");
    let table_num_vars: Vec<u8> = airs
        .air_refs()
        .iter()
        .enumerate()
        .map(|(i, air)| {
            if air.name() == "BITWISE" {
                crate::tables::bitwise::NUM_VARS as u8
            } else if (first..first + tables.len()).contains(&i) {
                tables[i - first].num_vars() as u8
            } else {
                2
            }
        })
        .collect();
    let statement = OwnedBlockStatement {
        groups: (0..table_num_vars.len() as u32).map(|t| vec![t]).collect(),
        table_num_vars,
        runtime_page_ranges,
        table_counts: counts,
        public_output: Vec::new(),
        num_private_input_pages: 2,
        fused_memory: [0, 0],
        word_memory: CPU_WORD_TABLES,
    };
    (statement, tables)
}

fn words() -> BlockFormat {
    BlockFormat::pinned().with_memory(MemoryFormat::Words)
}

/// ★ The verifier frames a word statement at its own page run, its page
/// tables cut by the rule from its own page list: their AIRs in PAGE's place,
/// named for their class, base and pages, at their widths and with `OFFSET`
/// (and a page of data's `INIT`) preprocessed. A statement cut at another run
/// is refused whenever the cuts differ; where they are the same cut, the
/// verifiers are one.
#[test]
fn a_word_statement_frames_at_its_own_page_run_alone() {
    let runs = [0u8, 1, 2, 4];
    for run_log2 in runs {
        let format = words().with_page_run(run_log2);
        let (statement, tables) = statement_at(&format);
        let frame = super::word_memory_tests::frame_of(&statement, &format)
            .unwrap_or_else(|e| panic!("at 2^{run_log2}: {e:?}"));
        assert_eq!(frame.page_tables, tables);
        let first = block_whir::first_page_index(&frame.airs).expect("the pages");
        let refs = frame.airs.air_refs();
        for (i, table) in tables.iter().enumerate() {
            let air = refs[first + i];
            assert_eq!(air.name(), table.name());
            assert_eq!(air.trace_layout().0, page_w::page_table_width(table));
            assert_eq!(
                air.num_precomputed_columns(),
                page_w::page_table_num_preprocessed_cols(table, &frame.page_configs)
                    .expect("its columns")
            );
        }
        if run_log2 == 4 {
            // The heap's 20 pages from 64: 16 then 4.
            assert!(tables.iter().any(|t| t.name() == "PAGE_Z:0x1000000:16"));
            assert!(tables.iter().any(|t| t.name() == "PAGE_Z:0x1400000:4"));
            assert!(tables.iter().any(|t| t.name() == "PAGE_P:0xff000000:2"));
        }
        for other in runs.into_iter().filter(|&o| o != run_log2) {
            let theirs = words().with_page_run(other);
            let same = page_tables(&frame.page_configs, other).expect("cut") == tables;
            let framed = super::word_memory_tests::frame_of(&statement, &theirs);
            assert_eq!(
                framed.is_ok(),
                same,
                "cut at 2^{run_log2}, framed at 2^{other}"
            );
        }
    }
}

/// ★ In the frame, every page table is held to its place, name and
/// height, which the verifier derives. A page table stated a row taller (or
/// shorter) is refused; so is a statement with two tables' heights swapped,
/// whose total is the same. `check_page_table_heights` refuses a table at a
/// page table's place under another base, page count or class.
#[test]
fn a_page_table_is_held_to_its_place_name_and_height() {
    let format = words().with_page_run(4);
    let (statement, tables) = statement_at(&format);
    let frame = super::word_memory_tests::frame_of(&statement, &format).expect("frames");
    let first = block_whir::first_page_index(&frame.airs).expect("the pages");
    let big = tables
        .iter()
        .position(|t| t.pages() == 16)
        .expect("a block of 16")
        + first;
    let small = tables
        .iter()
        .position(|t| t.pages() == 4)
        .expect("a block of 4")
        + first;
    for (at, by) in [(big, 1i8), (big, -1), (small, 1), (small, -1)] {
        let mut bad = statement.clone();
        bad.table_num_vars[at] = bad.table_num_vars[at].wrapping_add_signed(by);
        let refused = super::word_memory_tests::frame_of(&bad, &format);
        assert!(
            matches!(&refused, Err(crate::Error::InvalidTableCounts(e)) if e.contains("PAGE_Z")),
            "{at} {by}: {:?}",
            refused.err()
        );
    }
    let mut swapped = statement.clone();
    swapped.table_num_vars.swap(big, small);
    assert!(super::word_memory_tests::frame_of(&swapped, &format).is_err());

    // By name: another base, page count or class at a page table's place.
    let named = |names: &[String]| -> Vec<(String, u8)> {
        names
            .iter()
            .zip(&tables)
            .map(|(n, t)| (n.clone(), t.num_vars() as u8))
            .collect()
    };
    let own: Vec<String> = tables.iter().map(PageTable::name).collect();
    let check = |list: Vec<(String, u8)>| {
        block_whir::check_page_table_heights(&tables, 0, list.iter().map(|(n, v)| (n.as_str(), *v)))
    };
    assert!(check(named(&own)).is_ok());
    let b = big - first;
    for other in [
        PageTable {
            base: tables[b].base + 16 * PAGE,
            ..tables[b]
        },
        PageTable {
            log_pages: 3,
            ..tables[b]
        },
        PageTable {
            kind: PageTableKind::Private,
            ..tables[b]
        },
    ] {
        let mut names = own.clone();
        names[b] = other.name();
        assert!(check(named(&names)).is_err(), "{}", other.name());
    }
    // One page table short or one more.
    assert!(check(named(&own[..own.len() - 1])).is_err());
}

/// Page runs are the word memory format's alone, and at most
/// [`MAX_PAGE_RUN_LOG2`]: the verifier refuses either before framing.
#[test]
fn page_runs_need_the_word_format_and_stay_under_their_cap() {
    let (statement, _) = statement_at(&words().with_page_run(2));
    for format in [
        BlockFormat::pinned().with_page_run(2),
        BlockFormat::pinned()
            .with_memory(MemoryFormat::Fused)
            .with_page_run(2),
        words().with_page_run(MAX_PAGE_RUN_LOG2 + 1),
    ] {
        assert!(block_whir::check_memory_format(&format).is_err());
        assert!(super::word_memory_tests::frame_of(&statement, &format).is_err());
    }
    assert!(block_whir::check_memory_format(&words().with_page_run(MAX_PAGE_RUN_LOG2)).is_ok());
    assert!(block_whir::check_memory_format(&BlockFormat::pinned()).is_ok());
}

// =========================================================================
// The prepared opening's pages and the page list's own checks
// =========================================================================

/// Every page of data (a nonzero byte) is the first page of a page table of
/// its own at every run, on every list of the stream: the genesis rule's
/// dense pages, which the prepared opening settles, are tables of their own
/// whatever the run.
#[test]
fn every_page_of_data_is_a_table_of_its_own_at_every_run() {
    for (case, list) in lists().iter().enumerate() {
        for run_log2 in 0..=MAX_PAGE_RUN_LOG2 {
            let tables = page_tables(list, run_log2).expect("cut");
            for (i, config) in list.iter().enumerate() {
                if crate::continuation::nonzero_entries(config) > 0 && !config.is_private_input {
                    assert!(
                        tables.iter().any(|t| t.first == i
                            && t.kind == PageTableKind::Page
                            && t.pages() == 1),
                        "{case}: page {i} at 2^{run_log2}"
                    );
                }
            }
        }
    }
}

/// ★ More pages than tables: with page runs on, a page table holds up to
/// `2^k` pages, so an honest statement may list more pages than it states
/// tables. A heap of 1,024 zero pages at runs of 64 is 16 blocks, the
/// statement some 40 tables: it frames, and with runs off the same pages (one
/// table a page) are refused by the cap as before. The cap, `tables × 2^k`,
/// still refuses before the page list is built: a range one page past it, and
/// one of `2^62` pages.
#[test]
fn a_statement_with_more_pages_than_tables_frames_under_page_runs() {
    let format = words().with_page_run(6);
    let (statement, tables) = statement_with_heap(&format, 1024);
    let pages: usize = tables.iter().map(PageTable::pages).sum();
    let num_tables = statement.table_num_vars.len();
    assert!(
        pages > 1024 && num_tables < 64,
        "{pages} pages, {num_tables} tables"
    );
    let frame = super::word_memory_tests::frame_of(&statement, &format)
        .unwrap_or_else(|e| panic!("more pages than tables: {e:?}"));
    assert_eq!(frame.page_configs.len(), pages);
    assert_eq!(frame.page_tables, tables);
    assert_eq!(
        block_whir::max_block_pages(num_tables, &format),
        num_tables << 6
    );
    assert_eq!(
        block_whir::max_block_pages(num_tables, &words()),
        num_tables
    );
    // Runs off: one table a page, the same page list past the table count.
    let off = super::word_memory_tests::frame_of(&statement, &words());
    assert!(
        matches!(&off, Err(crate::Error::MalformedPageLayout(e)) if e.contains("needs more than")),
        "{:?}",
        off.err()
    );
    // Past the cap: refused by the cap, the range itself past it before any of
    // it is listed. At the cap exactly the list is built, and the statement is
    // refused by its table count instead (the pages need more tables).
    let cap = (num_tables << 6) as u64;
    let listed = pages as u64;
    for (count, by_cap) in [
        (cap - listed, false),
        (cap - listed + 1, true),
        (cap + 1, true),
        (1 << 62, true),
    ] {
        let mut bad = statement.clone();
        bad.runtime_page_ranges.push(crate::RuntimePageRange {
            base: 1 << 40,
            count,
        });
        let refused = super::word_memory_tests::frame_of(&bad, &format);
        let capped = matches!(&refused, Err(crate::Error::MalformedPageLayout(e)) if e.contains("needs more than"));
        assert!(
            refused.is_err() && capped == by_cap,
            "{count}: {:?}",
            refused.err()
        );
    }
}

/// The prepared tables of `format`'s frame of [`statement_at`], each by its
/// table's name: what the opening settles, and on which table.
fn prepared_by_name(format: &BlockFormat) -> Vec<(String, Vec<Vec<FE>>)> {
    let (statement, _) = statement_at(format);
    let frame = super::word_memory_tests::frame_of(&statement, format).expect("frames");
    let refs = frame.airs.air_refs();
    block_whir::prepared_tables(&frame.airs, &frame.page_configs, format)
        .expect("the prepared tables")
        .into_iter()
        .map(|(table, columns)| (refs[table].name().to_string(), columns))
        .collect()
}

/// The prepared tables of an AIR set over `pages` (any page list: the
/// program's DECODE, chip-free counts) under `format`, each by its table's
/// name.
fn prepared_over(pages: &[PageConfig], format: &BlockFormat) -> Vec<(String, Vec<Vec<FE>>)> {
    let elf = crate::test_utils::asm_elf_bytes("all_instructions_64");
    let program = executor::elf::Elf::load(&elf).expect("the ELF loads");
    let opts = ProofOptions::default_test_options();
    let counts = chip_free_counts();
    let inputs =
        crate::BlockAirInputs::new(&opts, &program, vec![4; counts.cpu], &[]).expect("inputs");
    let mut airs = crate::block_airs(
        &program,
        &opts,
        pages,
        &counts,
        &inputs,
        format.keccak_rnd,
        format.memw_aligned,
        format.registers,
    );
    airs.with_word_memory_at(
        &opts,
        CPU_WORD_TABLES.map(|n| n as usize),
        pages,
        format.page_run_log2,
    )
    .expect("the word tables");
    let refs = airs.air_refs();
    block_whir::prepared_tables(&airs, pages, format)
        .expect("the prepared tables")
        .into_iter()
        .map(|(table, columns)| (refs[table].name().to_string(), columns))
        .collect()
}

/// ★ The prepared opening settles the same pages at every run: the genesis
/// rule reads the pages (its page count and each page's candidacy), never the
/// tables, and a dense page is a page of data, a table of its own, so the
/// opening's columns and the PAGE_W tables they settle (by name) are the same
/// with runs off and on. On a frame's pages as the program's pages fall (none
/// dense: DECODE alone) and with every page of data dense (forced, on this
/// thread); and on a page list where pages of data follow blocks (so a page
/// table's index is not its page's), forced dense.
#[test]
fn the_dense_pages_are_the_same_at_every_page_run() {
    // Zero pages 64..68 (a block of four), a page of data at 68, zero pages
    // 69..72, a page of data at 72, two private-input pages.
    let data = |base| PageConfig::with_data(base, elf_data());
    let mut pages = run(64 * PAGE, 4, PageConfig::zero_init);
    pages.push(data(68 * PAGE));
    pages.extend(run(69 * PAGE, 3, PageConfig::zero_init));
    pages.push(data(72 * PAGE));
    pages.extend(run(INPUT_BASE, 2, |b| private(b, true)));
    let tables = page_tables(&pages, 4).expect("cut");
    assert!(
        tables
            .iter()
            .enumerate()
            .any(|(i, t)| t.kind == PageTableKind::Page && t.first != i),
        "a page of data whose table index is not its page's"
    );
    for forced in [false, true] {
        block_whir::DENSE_PAGES_FORCED.with(|c| c.set(forced));
        let checked = std::panic::catch_unwind(|| {
            let today = prepared_by_name(&words());
            assert_eq!(today[0].0, "DECODE");
            if forced {
                assert!(
                    today
                        .iter()
                        .any(|(n, _)| n.starts_with(page_w::NAME_PREFIX)),
                    "a page settled by the opening"
                );
            }
            for run_log2 in [1, 2, 4] {
                assert!(
                    prepared_by_name(&words().with_page_run(run_log2)) == today,
                    "forced {forced}: the opening at 2^{run_log2} is today's"
                );
            }
            if forced {
                let today = prepared_over(&pages, &words());
                let settled: Vec<&str> = today.iter().map(|(n, _)| n.as_str()).collect();
                assert_eq!(settled, ["DECODE", "PAGE_W:0x1100000", "PAGE_W:0x1200000"]);
                for run_log2 in [1, 2, 4] {
                    assert!(
                        prepared_over(&pages, &words().with_page_run(run_log2)) == today,
                        "the opening over blocks at 2^{run_log2} is today's"
                    );
                }
            }
        });
        block_whir::DENSE_PAGES_FORCED.with(|c| c.set(false));
        if let Err(panic) = checked {
            std::panic::resume_unwind(panic);
        }
    }
}

/// ★ A runtime range overlapping a page of the program, or another
/// range, is refused by the page list's one-page-a-base check before any
/// table is cut, at every run; and the rule refuses such a list itself.
#[test]
fn an_overlapping_runtime_range_is_refused_at_every_run() {
    let elf = crate::test_utils::asm_elf_bytes("all_instructions_64");
    let program = executor::elf::Elf::load(&elf).expect("the ELF loads");
    let elf_page =
        crate::tables::trace_builder::Traces::page_configs_from_elf(&program)[0].page_base;
    for run_log2 in [0u8, 2, 4] {
        let format = words().with_page_run(run_log2);
        let (statement, _) = statement_at(&format);
        for overlap in [
            crate::RuntimePageRange {
                base: elf_page,
                count: 1,
            },
            crate::RuntimePageRange {
                base: 70 * PAGE,
                count: 2,
            },
        ] {
            let base = overlap.base;
            let mut bad = statement.clone();
            bad.runtime_page_ranges.push(overlap);
            let refused = super::word_memory_tests::frame_of(&bad, &format);
            assert!(
                matches!(&refused, Err(crate::Error::MalformedPageLayout(e)) if e.contains("two page tables cover")),
                "at 2^{run_log2}, {base:#x}: {:?}",
                refused.err()
            );
        }
    }
    let mut doubled = run(64 * PAGE, 4, PageConfig::zero_init);
    doubled.push(PageConfig::zero_init(65 * PAGE));
    doubled.sort_by_key(|c| c.page_base);
    for run_log2 in 0..=MAX_PAGE_RUN_LOG2 {
        assert!(page_tables(&doubled, run_log2).is_err());
    }
}

/// ★ The census reads the page blocks: at runs of 16 pages every page table of
/// the verifier's set passes its page rule (one mint at timestamp 0 at
/// `base + 8·OFFSET`, `OFFSET` preprocessed, the base aligned to the table's
/// span, each initial-word column preprocessed or held to a half). Refused: a
/// zero block minting a committed column as its word, a private block without
/// one `INIT` half's check, a block whose name says a span its base is not
/// aligned to.
#[test]
fn the_census_reads_the_page_blocks() {
    use super::word_memory_tests::census_table;
    let format = words().with_page_run(4);
    let (statement, tables) = statement_at(&format);
    let frame = super::word_memory_tests::frame_of(&statement, &format).expect("frames");
    let first = block_whir::first_page_index(&frame.airs).expect("the pages");
    let refs = frame.airs.air_refs();
    for (i, table) in tables.iter().enumerate() {
        let air = refs[first + i];
        census_table(
            air.name(),
            air.bus_interactions(),
            air.num_precomputed_columns(),
        )
        .unwrap_or_else(|e| panic!("{}: {e}", table.name()));
    }
    let zero = tables
        .iter()
        .find(|t| t.kind == PageTableKind::Zero && t.pages() == 16)
        .expect("a block of 16");
    let own = page_w::page_table_interactions(zero, &frame.page_configs).expect("its own");
    let mut minted = own.clone();
    minted[0].values[3] = super::word_memory::column(zw::FINI_LO);
    assert!(census_table(&zero.name(), &minted, 1).is_err());
    let wide = PageTable {
        log_pages: 7,
        ..*zero
    };
    assert!(
        census_table(&wide.name(), &own, 1).is_err(),
        "{}",
        wide.name()
    );
    let private = tables
        .iter()
        .find(|t| t.kind == PageTableKind::Private)
        .expect("the private block");
    let mut unchecked =
        page_w::page_table_interactions(private, &frame.page_configs).expect("its own");
    assert!(census_table(&private.name(), &unchecked, 1).is_ok());
    unchecked.remove(2);
    assert!(census_table(&private.name(), &unchecked, 1).is_err());
}
