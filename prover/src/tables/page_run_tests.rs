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
